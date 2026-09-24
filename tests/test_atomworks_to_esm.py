"""The adapter reads chains, sequences and ligand identities off a structure."""

from __future__ import annotations

import pytest

from esmfold2_atomworks.data.atomworks_to_esm import (
    AdapterReport,
    atom_array_to_structure_prediction_input,
    chain_records,
)
from esmfold2_atomworks.data.spec import LigandIdentityError, LigandSpec


def _kinds(records) -> dict[str, str]:
    return {r.chain_id: r.kind for r in records}


def test_monomer_is_one_protein_chain(parsed):
    atoms, chain_info = parsed("lysozyme")
    records = chain_records(atoms, chain_info=chain_info)
    assert _kinds(records) == {"A": "protein"}
    assert records[0].n_residues == 129


def test_multimer_with_ligands_is_partitioned(parsed):
    atoms, chain_info = parsed("hemoglobin")
    kinds = _kinds(chain_records(atoms, chain_info=chain_info))
    assert [c for c, k in kinds.items() if k == "protein"] == ["A", "B", "C", "D"]
    assert [c for c, k in kinds.items() if k == "ligand"] == ["E", "G", "H", "J"]


def test_hemoglobin_produces_four_chains_and_four_hems(parsed, ccd):
    from esm.models.esmfold2.types import LigandInput, ProteinInput

    atoms, chain_info = parsed("hemoglobin")
    spi = atom_array_to_structure_prediction_input(atoms, chain_info=chain_info)

    proteins = [s for s in spi.sequences if isinstance(s, ProteinInput)]
    ligands = [s for s in spi.sequences if isinstance(s, LigandInput)]
    assert len(proteins) == 4
    assert len(ligands) == 4
    assert all(lig.ccd == ["HEM"] for lig in ligands)
    # Haemoglobin is a2b2: two distinct sequences, each appearing twice.
    assert len({p.sequence for p in proteins}) == 2


def test_sequence_comes_from_chain_info_not_observed_atoms(parsed, ccd):
    """The full entity sequence is preferred over the residues that happen to be modelled.

    Folding the observed residues only would silently fold a construct with its
    unmodelled loops deleted -- a different molecule that reports perfectly
    good confidence.
    """
    atoms, chain_info = parsed("hemoglobin")
    report = AdapterReport()
    atom_array_to_structure_prediction_input(
        atoms, chain_info=chain_info, report=report
    )
    assert set(report.sequence_source.values()) == {"chain_info"}


def test_sequence_override_is_recorded(parsed, ccd):
    """The design path: fold *this* sequence on *that* system."""
    atoms, chain_info = parsed("lysozyme")
    designed = "A" * 129
    report = AdapterReport()
    spi = atom_array_to_structure_prediction_input(
        atoms, chain_info=chain_info, sequences={"A": designed}, report=report
    )
    assert spi.sequences[0].sequence == designed
    assert report.sequence_source["A"] == "override"


def test_modified_residues_become_modifications(parsed, ccd):
    """MSE is declared by CCD code, not quietly folded as methionine."""
    atoms, chain_info = parsed("modified")
    report = AdapterReport()
    spi = atom_array_to_structure_prediction_input(
        atoms, chain_info=chain_info, report=report
    )
    protein = spi.sequences[0]
    assert protein.modifications is not None
    assert [m.ccd for m in protein.modifications] == ["MSE"] * 4
    # Positions verified against chain_info: MSE sits where the canonical
    # sequence reads "M".
    assert [m.position for m in protein.modifications] == [0, 34, 63, 64]
    assert all(protein.sequence[m.position] == "M" for m in protein.modifications)
    assert report.modifications["A"]


def test_modifications_can_be_disabled(parsed, ccd):
    atoms, chain_info = parsed("modified")
    spi = atom_array_to_structure_prediction_input(
        atoms, chain_info=chain_info, emit_modifications=False
    )
    assert spi.sequences[0].modifications is None


def test_overridden_sequence_drops_structure_modifications(parsed, ccd):
    """A designed sequence is a different molecule; its positions are not the structure's."""
    atoms, chain_info = parsed("modified")
    spi = atom_array_to_structure_prediction_input(
        atoms, chain_info=chain_info, sequences={"A": "G" * 70}
    )
    assert spi.sequences[0].modifications is None


def test_generic_ligand_label_is_refused(parsed, ccd):
    """``UNL`` is a real CCD code and also what a model writes on anything.

    Reconciling it against the dictionary is the silent wrong-molecule failure
    this project exists to prevent, so it must be declared instead.
    """
    atoms, chain_info = parsed("unl")
    with pytest.raises(LigandIdentityError, match="no chemical meaning"):
        atom_array_to_structure_prediction_input(atoms, chain_info=chain_info)


def test_declared_ligand_replaces_the_refused_label(parsed, ccd):
    atoms, chain_info = parsed("unl")
    records = chain_records(atoms, chain_info=chain_info)
    ligand_chain = next(r.chain_id for r in records if r.kind == "ligand")

    spi = atom_array_to_structure_prediction_input(
        atoms,
        chain_info=chain_info,
        ligands=(LigandSpec(chain_id=ligand_chain, smiles="c1ccccc1"),),
    )
    declared = [s for s in spi.sequences if getattr(s, "smiles", None)]
    assert len(declared) == 1
    assert declared[0].smiles == "c1ccccc1"


def test_ligand_composition_is_verified_not_trusted(parsed, ccd):
    """A declaration that disagrees with the atoms present is an error, not a warning."""
    atoms, chain_info = parsed("hemoglobin")
    wrong = LigandSpec(chain_id="E", ccd=("HEM",), expected_formula={"C": 1})
    with pytest.raises(LigandIdentityError, match="declared"):
        atom_array_to_structure_prediction_input(
            atoms, chain_info=chain_info, ligands=(wrong,)
        )


def test_undeclared_ccd_ligand_can_be_refused(parsed, ccd):
    atoms, chain_info = parsed("hemoglobin")
    with pytest.raises(LigandIdentityError, match="allow_undeclared_ccd_ligands"):
        atom_array_to_structure_prediction_input(
            atoms, chain_info=chain_info, allow_undeclared_ccd_ligands=False
        )


def test_report_names_everything_dropped(parsed, ccd):
    """Nothing leaves the adapter silently."""
    atoms, chain_info = parsed("hemoglobin")
    report = AdapterReport()
    atom_array_to_structure_prediction_input(
        atoms, chain_info=chain_info, report=report
    )
    assert report.dropped == []
    assert len(report.chains) == 8
