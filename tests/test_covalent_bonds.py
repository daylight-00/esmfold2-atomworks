"""Covalent connectivity the model cannot infer must be carried across.

ESMFold2 rebuilds bonds inside a residue from the CCD and the polymer backbone
from the sequence. Everything else -- a ligand bonded to a side chain, a
crosslink between chains -- exists only in the source structure, and if it is
not declared the model folds the pieces unconnected and returns a confident
prediction of a different molecule.

The test case is 2HHB's proximal histidine: chain A His87 NE2 coordinates the
haem iron in chain E. The deposited file does not record it as a `struct_conn`,
so it is added here; the chemistry is real and the bond is cross-chain, which
is the case that matters.
"""

from __future__ import annotations

import numpy as np

from esmfold2_foundry.data.atomworks_to_esm import (
    AdapterReport,
    atom_array_to_structure_prediction_input,
)
from esmfold2_foundry.data.bonds import covalent_bond_candidates
from esmfold2_foundry.parity.compare import compare_features, featurize


def _with_proximal_histidine_bond(atoms):
    """2HHB with the A/His87 NE2 -- E/HEM FE coordination added."""
    import biotite.structure as struc

    chain = np.asarray(atoms.chain_id).astype(str)
    res_id = np.asarray(atoms.res_id).astype(int)
    name = np.asarray(atoms.atom_name).astype(str)

    his = np.where((chain == "A") & (res_id == 87) & (name == "NE2"))[0]
    iron = np.where((chain == "E") & (name == "FE"))[0]
    assert len(his) == 1 and len(iron) == 1, "fixture no longer has the expected atoms"

    bonded = atoms.copy()
    bonded.bonds.add_bond(int(his[0]), int(iron[0]), struc.BondType.SINGLE)
    return bonded


def test_backbone_and_intra_residue_bonds_are_not_declared(parsed):
    """The common case must cost nothing and declare nothing.

    2HHB has 9302 bonds and 570 inter-residue ones, every one of them an
    ordinary peptide link. Forwarding those would be wrong twice over: the
    model already knows them, and declaring a bond forces its chains out of
    entity deduplication.
    """
    atoms, _chain_info = parsed("hemoglobin")
    assert covalent_bond_candidates(atoms) == []


def test_a_cross_chain_bond_is_detected_and_placed(parsed, ccd):
    atoms, chain_info = parsed("hemoglobin")
    bonded = _with_proximal_histidine_bond(atoms)

    candidates = covalent_bond_candidates(bonded)
    assert len(candidates) == 1
    assert candidates[0].describe() == "A/87/NE2 - E/142/FE"

    report = AdapterReport()
    spi = atom_array_to_structure_prediction_input(
        bonded, chain_info=chain_info, report=report
    )
    assert spi.covalent_bonds is not None
    (bond,) = spi.covalent_bonds
    assert report.unresolved_covalent_bonds == []

    # His87 is the 87th residue of a chain numbered from 1, so index 86; NE2 is
    # the tenth heavy atom of histidine in the tokenizer's ordering
    # (N CA C O CB CG ND1 CD2 CE1 NE2).
    assert (bond.chain_id1, bond.res_idx1, bond.atom_idx1) == ("A", 86, 9)
    assert bond.chain_id2 == "E"
    assert bond.res_idx2 == 0  # the haem is the chain's only residue

    # Plain Python types: these serialize to JSON for the hosted API.
    for value in (bond.res_idx1, bond.atom_idx1, bond.res_idx2, bond.atom_idx2):
        assert type(value) is int
    assert type(bond.chain_id1) is str


def test_declaring_the_bond_changes_only_the_bond_features(parsed, ccd):
    """Proves the declaration reaches the model, and touches nothing else.

    Without this the test above would only show that a `CovalentBond` object
    was constructed, not that ESMFold2 acted on it.
    """
    atoms, chain_info = parsed("hemoglobin")
    plain = atom_array_to_structure_prediction_input(atoms, chain_info=chain_info)
    bonded = atom_array_to_structure_prediction_input(
        _with_proximal_histidine_bond(atoms), chain_info=chain_info
    )

    diff = compare_features(featurize(plain), featurize(bonded))
    assert not diff.identical, "declaring a covalent bond had no effect on the features"
    changed = set(diff.value_mismatch) | set(diff.shape_mismatch)
    assert changed == {"token_bonds"}, changed


def test_bonds_can_be_switched_off(parsed, ccd):
    atoms, chain_info = parsed("hemoglobin")
    bonded = _with_proximal_histidine_bond(atoms)
    spi = atom_array_to_structure_prediction_input(
        bonded, chain_info=chain_info, declare_covalent_bonds=False
    )
    assert spi.covalent_bonds is None


def test_bonds_to_dropped_chains_are_not_declared(parsed, ccd):
    """An index into a chain the model never saw is an error upstream."""
    atoms, _chain_info = parsed("hemoglobin")
    bonded = _with_proximal_histidine_bond(atoms)
    # Pretend chain E never reaches the model.
    candidates = covalent_bond_candidates(bonded, keep_chains=frozenset({"A", "B"}))
    assert candidates == []


def _tiny(spec, bond_pairs):
    """A minimal AtomArray: ``spec`` is ``(chain, res_id, atom_name, element)``."""
    import biotite.structure as struc

    array = struc.AtomArray(len(spec))
    array.coord = np.zeros((len(spec), 3), dtype=np.float32)
    array.set_annotation("chain_id", np.array([s[0] for s in spec], dtype="U4"))
    array.set_annotation("res_id", np.array([s[1] for s in spec], dtype=int))
    array.set_annotation("res_name", np.array(["XXX"] * len(spec), dtype="U5"))
    array.set_annotation("atom_name", np.array([s[2] for s in spec], dtype="U6"))
    array.set_annotation("element", np.array([s[3] for s in spec], dtype="U2"))
    array.bonds = struc.BondList(len(spec))
    for i, j in bond_pairs:
        array.bonds.add_bond(i, j, struc.BondType.SINGLE)
    return array


def test_a_bond_to_hydrogen_is_ignored():
    """ESMFold2 models heavy atoms, so an H endpoint cannot be addressed at all."""
    atoms = _tiny(
        [("A", 1, "SG", "S"), ("B", 1, "H1", "H")],
        [(0, 1)],
    )
    assert covalent_bond_candidates(atoms) == []


def test_a_cross_chain_heavy_atom_bond_is_kept():
    """The same shape with a heavy partner must survive, or the test above is vacuous."""
    atoms = _tiny(
        [("A", 1, "SG", "S"), ("B", 1, "C1", "C")],
        [(0, 1)],
    )
    (candidate,) = covalent_bond_candidates(atoms)
    assert candidate.describe() == "A/1/SG - B/1/C1"


def test_a_peptide_link_between_consecutive_residues_is_ignored():
    atoms = _tiny(
        [("A", 1, "C", "C"), ("A", 2, "N", "N")],
        [(0, 1)],
    )
    assert covalent_bond_candidates(atoms) == []


def test_a_non_backbone_bond_between_consecutive_residues_is_kept():
    """A disulphide between neighbours is not the backbone and must be declared."""
    atoms = _tiny(
        [("A", 1, "SG", "S"), ("A", 2, "SG", "S")],
        [(0, 1)],
    )
    (candidate,) = covalent_bond_candidates(atoms)
    assert candidate.describe() == "A/1/SG - A/2/SG"
