"""The bond list of a folded structure, rebuilt from the chemistry it was given.

No weights: the structure a fold would return is assembled from the featurization
with placeholder coordinates, through esm's own builder, so the atoms and names
are the ones a real fold returns.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("esm.models.esmfold2")
pytest.importorskip("rdkit")
struc = pytest.importorskip("biotite.structure")

from esmfold2_atomworks.data.spec import TopologyError

SINGLE, DOUBLE = int(struc.BondType.SINGLE), int(struc.BondType.DOUBLE)


def _folded(spi):
    """What ``fold_atom_array`` would hand back for *spi*, without the model."""
    import torch
    from esm.models.esmfold2.output import build_molecular_complex_from_features
    from esm.models.esmfold2.prepare_input import prepare_esmfold2_input
    from esm.models.esmfold2.processor import clean_esmfold2_input

    from esmfold2_atomworks.data.molecular_complex import (
        molecular_complex_to_atom_array,
    )

    features, chain_infos = prepare_esmfold2_input(clean_esmfold2_input(spi), seed=0)
    complex_ = build_molecular_complex_from_features(
        coords=torch.zeros(features["ref_pos"].shape[0], 3),
        plddt=torch.ones(features["res_type"].shape[-1]),
        atom_mask=features["atom_attention_mask"],
        ref_element=features["ref_element"],
        ref_atom_name_chars=features["ref_atom_name_chars"],
        chain_infos=chain_infos,
        complex_id="t",
    )
    return molecular_complex_to_atom_array(complex_)


def _bond_kind(atoms, first, second):
    """The type of the bond between ``(chain, res_id, name)`` atoms, or ``None``."""

    def locate(chain, res_id, name):
        hit = np.flatnonzero(
            (np.asarray(atoms.chain_id) == chain)
            & (np.asarray(atoms.res_id) == res_id)
            & (np.asarray(atoms.atom_name) == name)
        )
        assert hit.size == 1, f"{chain}/{res_id}/{name} is not one atom"
        return int(hit[0])

    i, j = locate(*first), locate(*second)
    for a, b, kind in atoms.bonds.as_array():
        if {int(a), int(b)} == {i, j}:
            return int(kind)
    return None


def _spi(*sequences, bonds=None):
    from esm.models.esmfold2.types import StructurePredictionInput

    return StructurePredictionInput(sequences=list(sequences), covalent_bonds=bonds)


def test_a_peptide_is_bonded_inside_and_between_its_residues(ccd):
    from esm.models.esmfold2.types import ProteinInput

    from esmfold2_atomworks.data.topology import build_bond_list

    spi = _spi(ProteinInput(id="A", sequence="GAG"))
    atoms = _folded(spi)
    atoms.bonds = build_bond_list(atoms, spi)

    assert _bond_kind(atoms, ("A", 2, "CA"), ("A", 2, "CB")) == SINGLE
    assert _bond_kind(atoms, ("A", 2, "C"), ("A", 2, "O")) == DOUBLE
    assert _bond_kind(atoms, ("A", 1, "C"), ("A", 2, "N")) == SINGLE
    assert _bond_kind(atoms, ("A", 2, "C"), ("A", 3, "N")) == SINGLE
    assert _bond_kind(atoms, ("A", 1, "C"), ("A", 3, "N")) is None
    assert _bond_kind(atoms, ("A", 1, "N"), ("A", 3, "C")) is None


def test_an_aromatic_ring_keeps_its_bond_orders(ccd):
    from esm.models.esmfold2.types import ProteinInput

    from esmfold2_atomworks.data.topology import build_bond_list

    spi = _spi(ProteinInput(id="A", sequence="GFG"))
    atoms = _folded(spi)
    atoms.bonds = build_bond_list(atoms, spi)

    kinds = {
        _bond_kind(atoms, ("A", 2, a), ("A", 2, b))
        for a, b in (("CG", "CD1"), ("CD1", "CE1"), ("CE1", "CZ"))
    }
    assert kinds <= {
        int(struc.BondType.AROMATIC_SINGLE),
        int(struc.BondType.AROMATIC_DOUBLE),
    }
    assert len(kinds) == 2


def test_a_modified_residue_is_bonded_by_its_own_component(ccd):
    from esm.models.esmfold2.types import Modification, ProteinInput

    from esmfold2_atomworks.data.topology import build_bond_list

    spi = _spi(
        ProteinInput(
            id="A",
            sequence="AMA",
            modifications=[Modification(position=1, ccd="MSE")],
        )
    )
    atoms = _folded(spi)
    atoms.bonds = build_bond_list(atoms, spi)

    assert _bond_kind(atoms, ("A", 2, "CG"), ("A", 2, "SE")) == SINGLE
    assert _bond_kind(atoms, ("A", 2, "SE"), ("A", 2, "CE")) == SINGLE
    assert _bond_kind(atoms, ("A", 1, "C"), ("A", 2, "N")) == SINGLE


def test_a_nucleic_acid_is_linked_through_its_phosphates(ccd):
    from esm.models.esmfold2.types import DNAInput

    from esmfold2_atomworks.data.topology import build_bond_list

    spi = _spi(DNAInput(id="A", sequence="ACG"))
    atoms = _folded(spi)
    atoms.bonds = build_bond_list(atoms, spi)

    assert _bond_kind(atoms, ("A", 1, "O3'"), ("A", 2, "P")) == SINGLE
    assert _bond_kind(atoms, ("A", 2, "O3'"), ("A", 3, "P")) == SINGLE


def test_a_smiles_ligand_gets_the_bonds_of_its_smiles(ccd):
    from esm.models.esmfold2.types import LigandInput, ProteinInput

    from esmfold2_atomworks.data.topology import build_bond_list

    spi = _spi(
        ProteinInput(id="A", sequence="G"),
        LigandInput(id="B", smiles="CC(=O)O"),
    )
    atoms = _folded(spi)
    atoms.bonds = build_bond_list(atoms, spi)

    ligand = np.flatnonzero(np.asarray(atoms.chain_id) == "B")
    inside = [
        (int(a), int(b), int(k))
        for a, b, k in atoms.bonds.as_array()
        if a in ligand and b in ligand
    ]
    assert len(inside) == 3  # acetic acid: C-C, C=O, C-O
    assert sorted(kind for *_, kind in inside) == [SINGLE, SINGLE, DOUBLE]


def test_a_ring_in_a_smiles_ligand_is_aromatic(ccd):
    from esm.models.esmfold2.types import LigandInput

    from esmfold2_atomworks.data.topology import build_bond_list

    spi = _spi(LigandInput(id="B", smiles="c1ccccc1"))
    atoms = _folded(spi)
    atoms.bonds = build_bond_list(atoms, spi)

    kinds = [int(kind) for *_, kind in atoms.bonds.as_array()]
    assert kinds == [int(struc.BondType.AROMATIC)] * 6


def test_a_smiles_ligand_named_for_a_ccd_component_collides_with_it(ccd):
    from esm.models.esmfold2.types import LigandInput, ProteinInput

    from esmfold2_atomworks.data.molecular_complex import rename_ligand_residues
    from esmfold2_atomworks.data.topology import build_bond_list, ccd_name_collisions

    spi = _spi(
        ProteinInput(id="A", sequence="G"), LigandInput(id="B", smiles="CC(=O)O")
    )
    atoms = _folded(spi)
    atoms.bonds = build_bond_list(atoms, spi)

    assert ccd_name_collisions(atoms) == [
        "LIG"
    ]  # LIG is a CCD code, for another molecule
    assert ccd_name_collisions(rename_ligand_residues(atoms, "Q9Q9Q")) == []


def test_a_ccd_ligand_does_not_collide_with_its_own_component(ccd):
    from esm.models.esmfold2.types import CovalentBond, LigandInput, ProteinInput

    from esmfold2_atomworks.data.topology import build_bond_list, ccd_name_collisions

    plain = _spi(LigandInput(id="B", ccd=["NAG"]))
    atoms = _folded(plain)
    atoms.bonds = build_bond_list(atoms, plain)
    assert ccd_name_collisions(atoms) == []

    # Bonded, the leaving atoms are gone: a subset of the component, still itself.
    bonded = _spi(
        ProteinInput(id="A", sequence="GGG"),
        LigandInput(id="B", ccd=["NAG"]),
        bonds=[CovalentBond("A", 1, 2, "B", 0, 10)],
    )
    atoms = _folded(bonded)
    atoms.bonds = build_bond_list(atoms, bonded)
    assert ccd_name_collisions(atoms) == []


def test_the_collision_check_needs_no_bond_list(ccd):
    from esm.models.esmfold2.types import LigandInput

    from esmfold2_atomworks.data.topology import ccd_name_collisions

    atoms = _folded(_spi(LigandInput(id="B", smiles="c1ccccc1")))
    assert atoms.bonds is None
    assert ccd_name_collisions(atoms) == ["LIG"]


def _written_and_reread(atoms, tmp_path):
    """The structure as AtomWorks reads it back from the file ``dump`` wrote."""
    pytest.importorskip("atomworks")
    from atomworks.io import parse

    from esmfold2_atomworks.inference.engine import ESMFold2Output

    path = ESMFold2Output(atom_array=atoms, example_id="x").dump(
        tmp_path, verbose=False
    )
    return parse(path)["asym_unit"][0]


def _ligand_and_bonds(atoms):
    """Chain B's atom names and the number of bonds between its atoms."""
    chain = np.flatnonzero(np.asarray(atoms.chain_id) == "B")
    inside = [1 for a, b, _ in atoms.bonds.as_array() if a in chain and b in chain]
    return sorted(map(str, np.asarray(atoms.atom_name)[chain])), len(inside)


def _acetic_acid(ccd, label):
    from esm.models.esmfold2.types import LigandInput, ProteinInput

    from esmfold2_atomworks.data.molecular_complex import rename_ligand_residues
    from esmfold2_atomworks.data.topology import build_bond_list

    spi = _spi(
        ProteinInput(id="A", sequence="G"), LigandInput(id="B", smiles="CC(=O)O")
    )
    atoms = _folded(spi)
    atoms.bonds = build_bond_list(atoms, spi)
    return rename_ligand_residues(atoms, label)


def test_a_smiles_ligand_with_a_non_ccd_name_is_read_back_as_written(ccd, tmp_path):
    atoms = _acetic_acid(ccd, "Q9Q9Q")

    back = _written_and_reread(atoms, tmp_path)

    assert _ligand_and_bonds(back) == _ligand_and_bonds(atoms)


def test_a_smiles_ligand_named_for_a_ccd_component_is_not_read_back_as_written(
    ccd, tmp_path
):
    atoms = _acetic_acid(ccd, "LIG")
    try:
        back = _written_and_reread(atoms, tmp_path)
    except ValueError:  # 3.x: atoms the component does not have
        return
    assert _ligand_and_bonds(back) != _ligand_and_bonds(atoms)  # 2.x: rebuilt


def test_a_declared_bond_keeps_the_order_the_source_had(ccd):
    from esm.models.esmfold2.types import CovalentBond, ProteinInput

    from esmfold2_atomworks.data.topology import build_bond_list

    bond = CovalentBond("A", 0, 3, "B", 0, 3)  # A1 O (index 3) - B1 O
    bond.source_bond_type = DOUBLE
    spi = _spi(
        ProteinInput(id="A", sequence="G"),
        ProteinInput(id="B", sequence="G"),
        bonds=[bond],
    )
    atoms = _folded(spi)
    atoms.bonds = build_bond_list(atoms, spi)

    assert _bond_kind(atoms, ("A", 1, "O"), ("B", 1, "O")) == DOUBLE


def test_a_bond_to_a_ligand_atom_lands_after_the_leaving_atoms_are_dropped(ccd):
    from esm.models.esmfold2.types import CovalentBond, LigandInput, ProteinInput

    from esmfold2_atomworks.data.topology import build_bond_list

    protein = ProteinInput(id="A", sequence="GGG")
    sugar = LigandInput(id="B", ccd=["NAG"])
    # NAG's O1 is flagged leaving, so in a bonded chain O4 is atom 10, not 11.
    bond = CovalentBond("A", 1, 2, "B", 0, 10)
    atoms = _folded(_spi(protein, sugar, bonds=[bond]))
    atoms.bonds = build_bond_list(atoms, _spi(protein, sugar, bonds=[bond]))

    assert "O1" not in set(map(str, atoms.atom_name[np.asarray(atoms.chain_id) == "B"]))
    assert _bond_kind(atoms, ("A", 2, "C"), ("B", 1, "O4")) == SINGLE
    assert _bond_kind(atoms, ("B", 1, "C1"), ("B", 1, "C2")) == SINGLE


def test_atoms_that_do_not_match_the_chemistry_are_refused(ccd):
    from esm.models.esmfold2.types import ProteinInput

    from esmfold2_atomworks.data.topology import build_bond_list

    spi = _spi(ProteinInput(id="A", sequence="GAG"))
    atoms = _folded(spi)
    names = np.asarray(atoms.atom_name).copy()
    names[1] = "XX"
    atoms.set_annotation("atom_name", names)
    with pytest.raises(TopologyError, match="not atoms of CCD component"):
        build_bond_list(atoms, spi)


def test_a_chain_the_input_does_not_name_is_refused(ccd):
    from esm.models.esmfold2.types import ProteinInput

    from esmfold2_atomworks.data.topology import build_bond_list

    spi = _spi(ProteinInput(id="A", sequence="GAG"))
    atoms = _folded(spi)
    other = _spi(ProteinInput(id="Z", sequence="GAG"))
    with pytest.raises(TopologyError, match="not in the structure"):
        build_bond_list(atoms, other)


@pytest.mark.parametrize("name", ["lysozyme", "hemoglobin", "modified"])
def test_the_bonds_equal_biotites_own_assignment_by_residue_name(parsed, ccd, name):
    """An independent route to the same bonds, with the same types.

    ``connect_via_residue_names`` reads the CCD through biotite and links
    consecutive residues; the rebuilt list has to agree on every bond.
    """
    from esmfold2_atomworks.data.atomworks_to_esm import (
        atom_array_to_structure_prediction_input,
    )
    from esmfold2_atomworks.data.topology import build_bond_list

    atoms, chain_info = parsed(name)
    spi = atom_array_to_structure_prediction_input(atoms, chain_info=chain_info)
    folded = _folded(spi)
    ours = build_bond_list(folded, spi)
    reference = struc.connect_via_residue_names(folded, inter_residue=True)

    def as_dict(bonds):
        return {
            (min(int(a), int(b)), max(int(a), int(b))): int(kind)
            for a, b, kind in bonds.as_array()
        }

    assert as_dict(ours) == as_dict(reference)
