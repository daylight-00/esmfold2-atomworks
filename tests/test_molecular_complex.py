"""The return leg, ``MolecularComplex`` -> ``AtomArray``, pinned on its promises.

What the model reported is neither lost, duplicated nor invented on the way back:
every atom keeps its position, name and element; the token spans partition the
atoms, or the conversion refuses; the polymer/ligand split is the model's own
hetero flags, never a CCD lookup; chain indices resolve to their letters; and a
ligand relabel touches only hetero residues.

Synthetic complexes, so all of it holds without weights or a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from esmfold2_atomworks.data.molecular_complex import (
    molecular_complex_to_atom_array,
    rename_ligand_residues,
)

pytest.importorskip("biotite.structure")


class FakeMetadata:
    """``chain_lookup`` as ESM reports it: a one-element list holding the dict."""

    def __init__(self, lookup):
        self.chain_lookup = [lookup]


class FakeComplex:
    """The flat arrays ``build_molecular_complex_from_features`` fills in."""

    def __init__(
        self,
        sequence,
        atom_names,
        atom_elements,
        atom_hetero,
        token_to_atoms,
        chain_id,
        chain_lookup=None,
    ):
        n_atoms = len(atom_names)
        self.sequence = np.asarray(sequence, dtype="U5")
        self.atom_names = np.asarray(atom_names, dtype=object)
        self.atom_elements = np.asarray(atom_elements, dtype=object)
        self.atom_hetero = np.asarray(atom_hetero, dtype=bool)
        self.token_to_atoms = np.asarray(token_to_atoms, dtype=np.int32)
        self.chain_id = np.asarray(chain_id)
        # Distinct per atom, so a reordering or a dropped atom shows.
        self.atom_positions = np.arange(n_atoms * 3, dtype=np.float32).reshape(-1, 3)
        self.metadata = FakeMetadata(chain_lookup or {0: "A", 1: "B"})


@pytest.fixture
def protein_and_ligand():
    """Two residues of chain A (three atoms each), then a four-atom ligand, B."""
    return FakeComplex(
        sequence=["ALA", "GLY", "LIG"],
        atom_names=["N", "CA", "C", "N", "CA", "C", "C1", "N1", "O1", "FE"],
        atom_elements=["N", "C", "C", "N", "C", "C", "C", "N", "O", "Fe"],
        atom_hetero=[False] * 6 + [True] * 4,
        # [start, end) ranges, as ESM emits them.
        token_to_atoms=[[0, 3], [3, 6], [6, 10]],
        chain_id=[0, 0, 1],
    )


def _hetero(atoms):
    return np.asarray(atoms.hetero, dtype=bool)


# -- nothing lost, nothing invented --------------------------------------------


def test_every_atom_keeps_its_position_name_and_element(protein_and_ligand):
    atoms = molecular_complex_to_atom_array(protein_and_ligand)

    assert len(atoms) == 10
    np.testing.assert_array_equal(atoms.coord, protein_and_ligand.atom_positions)
    assert list(atoms.atom_name) == list(protein_and_ligand.atom_names)
    # Upper-cased, which is biotite's convention for element symbols.
    assert list(atoms.element) == [e.upper() for e in protein_and_ligand.atom_elements]


def test_every_ligand_atom_survives(protein_and_ligand):
    ligand = molecular_complex_to_atom_array(protein_and_ligand)
    ligand = ligand[_hetero(ligand)]

    assert sorted(map(str, ligand.atom_name)) == ["C1", "FE", "N1", "O1"]


def test_hetero_comes_from_the_model_even_against_the_ccd():
    """``ALA`` is a peptide residue to the CCD; the model says hetero, and wins.

    The flags decide the polymer/ligand split, and a CCD lookup is exactly the
    reconciliation that once turned a 19-atom ligand into 8 carbons.
    """
    complex_ = FakeComplex(
        sequence=["GLY", "ALA"],
        atom_names=["N", "CA", "C", "N", "CA", "C"],
        atom_elements=["N", "C", "C", "N", "C", "C"],
        atom_hetero=[False] * 3 + [True] * 3,
        token_to_atoms=[[0, 3], [3, 6]],
        chain_id=[0, 1],
    )

    atoms = molecular_complex_to_atom_array(complex_)

    assert _hetero(atoms).tolist() == [False] * 3 + [True] * 3
    assert np.asarray(atoms.is_polymer).tolist() == [True] * 3 + [False] * 3


def test_chain_indices_resolve_to_letters(protein_and_ligand):
    """``chain_id`` holds integer indices; the letters live in ``chain_lookup``.
    Stringified, the integers would name the chains "0" and "1", which match no
    chain a caller asks about by its letter."""
    atoms = molecular_complex_to_atom_array(protein_and_ligand)

    assert list(np.unique(atoms.chain_id)) == ["A", "B"]


def test_residue_ids_are_numbered_per_chain(protein_and_ligand):
    atoms = molecular_complex_to_atom_array(protein_and_ligand)

    assert list(atoms[atoms.chain_id == "A"].res_id) == [1, 1, 1, 2, 2, 2]
    assert list(atoms[atoms.chain_id == "B"].res_id) == [1, 1, 1, 1]


# -- the spans partition the atoms, or nothing is built ----------------------------


def _alanine(token_to_atoms, sequence=("ALA",), chain_id=(0,)):
    return FakeComplex(
        sequence=list(sequence),
        atom_names=["N", "CA", "C", "O"],
        atom_elements=["N", "C", "C", "O"],
        atom_hetero=[False] * 4,
        token_to_atoms=token_to_atoms,
        chain_id=list(chain_id),
    )


def test_an_atom_claimed_by_no_span_is_refused():
    """Dropping it would be the silent loss this conversion exists to prevent."""
    with pytest.raises(ValueError, match="claimed"):
        molecular_complex_to_atom_array(_alanine([[0, 2]]))


def test_an_atom_claimed_by_two_spans_is_refused():
    """Padded index rows, where an overlap is expressible: atom 2 in both.

    Built anyway, the second token would silently take the atom from the first.
    """
    complex_ = _alanine(
        [[0, 1, 2], [2, 3, -1]], sequence=("ALA", "GLY"), chain_id=(0, 0)
    )

    with pytest.raises(ValueError, match="claimed"):
        molecular_complex_to_atom_array(complex_)


def test_a_span_that_claims_no_atom_is_refused():
    """A residue with no atoms would vanish from the structure but not from
    the complex's per-residue arrays, misaligning the two."""
    complex_ = _alanine(
        [[0, 1, 2, 3], [-1, -1, -1, -1]], sequence=("ALA", "GLY"), chain_id=(0, 0)
    )

    with pytest.raises(ValueError, match="claim none"):
        molecular_complex_to_atom_array(complex_)


def test_a_complex_whose_arrays_disagree_is_refused():
    complex_ = FakeComplex(
        sequence=["ALA", "GLY"],
        atom_names=["N", "CA"],
        atom_elements=["N", "C"],
        atom_hetero=[False, False],
        token_to_atoms=[[0, 2]],
        chain_id=[0],
    )

    with pytest.raises(ValueError, match="inconsistent"):
        molecular_complex_to_atom_array(complex_)


# -- the relabel ---------------------------------------------------------------------


def test_the_relabel_touches_only_hetero_residues(protein_and_ligand):
    atoms = molecular_complex_to_atom_array(protein_and_ligand)

    renamed = rename_ligand_residues(atoms, "PHT")

    assert set(renamed.res_name[_hetero(renamed)]) == {"PHT"}
    assert set(renamed.res_name[~_hetero(renamed)]) == {"ALA", "GLY"}
    assert set(atoms.res_name[_hetero(atoms)]) == {"LIG"}, "the input was modified"


def test_the_relabel_is_a_no_op_without_a_hetero_residue():
    atoms = molecular_complex_to_atom_array(_alanine([[0, 4]]))

    assert list(rename_ligand_residues(atoms, "PHT").res_name) == list(atoms.res_name)


@pytest.mark.parametrize("name", ["", "LIGAND"])
def test_a_name_that_cannot_be_a_residue_name_is_refused(protein_and_ligand, name):
    """Truncated to five characters, ``LIGAND`` would come back as a name the
    caller never gave."""
    atoms = molecular_complex_to_atom_array(protein_and_ligand)

    with pytest.raises(ValueError, match="one to five characters"):
        rename_ligand_residues(atoms, name)
