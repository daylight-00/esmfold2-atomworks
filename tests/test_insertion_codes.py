"""Residue identity is ``(chain, res_id, ins_code)``, not ``(chain, res_id)``.

A deposited chain may number residues 100, 100A, 100B. Keying on ``res_id``
alone makes those one residue, with two consequences that are both silent:

* a genuine bond between two of them looks intra-residue and is dropped, so the
  model folds them unconnected;
* the residue map keeps whichever entry was written last, so labels land on the
  wrong residue.

Neither is acceptable in a package whose stated preference is to refuse rather
than fold the wrong molecule, so both are fixed where they can be and reported
where they cannot. ``chain_info`` records no insertion codes, so for a chain
whose sequence comes from there the codes cannot be tied to sequence positions;
that chain is named in the report instead of guessed at.
"""

from __future__ import annotations

import numpy as np

from esmfold2_atomworks.data.atomworks_to_esm import (
    AdapterReport,
    _residue_index_map,
    chain_records,
)
from esmfold2_atomworks.data.bonds import covalent_bond_candidates


def _chain_with_insertion_codes():
    """One chain numbered 100, 100A, 100B, with a bond between the first two."""
    import biotite.structure as struc

    spec = [
        ("A", 100, "", "CA", "C"),
        ("A", 100, "", "SG", "S"),
        ("A", 100, "A", "CA", "C"),
        ("A", 100, "A", "SG", "S"),
        ("A", 100, "B", "CA", "C"),
        ("A", 100, "B", "SG", "S"),
    ]
    array = struc.AtomArray(len(spec))
    array.coord = np.arange(len(spec) * 3, dtype=np.float32).reshape(-1, 3)
    array.set_annotation("chain_id", np.array([s[0] for s in spec], dtype="U4"))
    array.set_annotation("res_id", np.array([s[1] for s in spec], dtype=int))
    array.set_annotation("ins_code", np.array([s[2] for s in spec], dtype="U1"))
    array.set_annotation("res_name", np.array(["CYS"] * len(spec), dtype="U5"))
    array.set_annotation("atom_name", np.array([s[3] for s in spec], dtype="U6"))
    array.set_annotation("element", np.array([s[4] for s in spec], dtype="U2"))
    array.set_annotation("is_polymer", np.ones(len(spec), dtype=bool))
    array.bonds = struc.BondList(len(spec))
    # 100/SG -- 100A/SG: two different residues, so a real inter-residue bond.
    array.bonds.add_bond(1, 3, struc.BondType.SINGLE)
    return array


def test_a_bond_between_insertion_coded_residues_is_not_mistaken_for_intra_residue():
    """This bond was previously dropped: both endpoints have res_id 100."""
    candidates = covalent_bond_candidates(_chain_with_insertion_codes())
    assert len(candidates) == 1, "the bond between 100 and 100A was dropped again"
    assert candidates[0].describe() == "A/100/SG - A/100A/SG"


def test_insertion_codes_reach_the_candidate():
    candidate = covalent_bond_candidates(_chain_with_insertion_codes())[0]
    assert candidate.residue_1 == ("A", 100, "")
    assert candidate.residue_2 == ("A", 100, "A")


def _numbered_chain(residues, bond):
    """A chain of GLY residues given as ``(res_id, ins_code)``, with one bond.

    ``bond`` is ``((residue index, atom name), (residue index, atom name))``.
    Every residue carries N, C and SG so any of the three can be bonded.
    """
    import biotite.structure as struc

    names = ("N", "C", "SG")
    elements = ("N", "C", "S")
    n = len(residues) * len(names)
    array = struc.AtomArray(n)
    array.coord = np.zeros((n, 3), dtype=np.float32)
    array.set_annotation("chain_id", np.array(["A"] * n, dtype="U4"))
    array.set_annotation(
        "res_id", np.array([r[0] for r in residues for _ in names], dtype=int)
    )
    array.set_annotation(
        "ins_code", np.array([r[1] for r in residues for _ in names], dtype="U1")
    )
    array.set_annotation("res_name", np.array(["CYS"] * n, dtype="U5"))
    array.set_annotation(
        "atom_name", np.array([a for _ in residues for a in names], dtype="U6")
    )
    array.set_annotation(
        "element", np.array([e for _ in residues for e in elements], dtype="U2")
    )
    array.bonds = struc.BondList(n)
    (res_a, atom_a), (res_b, atom_b) = bond
    i = res_a * len(names) + names.index(atom_a)
    j = res_b * len(names) + names.index(atom_b)
    array.bonds.add_bond(i, j, struc.BondType.SINGLE)
    return array


# 100, 100A, 100B, 101 are four consecutive residues of one polymer -- the
# ordinary meaning of an insertion code.
INSERTED = [(100, ""), (100, "A"), (100, "B"), (101, "")]


def test_a_peptide_bond_across_an_insertion_code_is_not_declared():
    """100/C -- 100A/N is plain backbone, whatever the numbering suggests.

    Declaring it would hand the model a `token_bonds` edge that an ordinary
    chain never has: upstream adds no token bond for a standard residue's
    backbone at all (`compute_token_bonds` skips them with
    "Standard residue - no peptide bond added here").
    """
    array = _numbered_chain(INSERTED, ((0, "C"), (1, "N")))
    assert covalent_bond_candidates(array) == []


def test_a_peptide_bond_between_two_insertion_codes_is_not_declared():
    """100A/C -- 100B/N is the same case one residue along."""
    array = _numbered_chain(INSERTED, ((1, "C"), (2, "N")))
    assert covalent_bond_candidates(array) == []


def test_a_disulphide_across_an_insertion_code_is_declared():
    """Same residue pair, non-backbone atoms: a real crosslink, and it stays."""
    array = _numbered_chain(INSERTED, ((0, "SG"), (1, "SG")))
    (candidate,) = covalent_bond_candidates(array)
    assert candidate.describe() == "A/100/SG - A/100A/SG"


def test_a_backbone_bond_between_non_adjacent_residues_is_declared():
    """100/C -- 100B/N skips a residue, so it is not the backbone."""
    array = _numbered_chain(INSERTED, ((0, "C"), (2, "N")))
    (candidate,) = covalent_bond_candidates(array)
    assert candidate.describe() == "A/100/C - A/100B/N"


def test_an_ordinary_backbone_link_is_still_exempt():
    """Guards the change above from turning every peptide bond into a declaration."""
    import biotite.structure as struc

    array = struc.AtomArray(2)
    array.coord = np.zeros((2, 3), dtype=np.float32)
    array.set_annotation("chain_id", np.array(["A", "A"], dtype="U4"))
    array.set_annotation("res_id", np.array([100, 101], dtype=int))
    array.set_annotation("ins_code", np.array(["", ""], dtype="U1"))
    array.set_annotation("res_name", np.array(["GLY", "GLY"], dtype="U5"))
    array.set_annotation("atom_name", np.array(["C", "N"], dtype="U6"))
    array.set_annotation("element", np.array(["C", "N"], dtype="U2"))
    array.bonds = struc.BondList(2)
    array.bonds.add_bond(0, 1, struc.BondType.SINGLE)

    assert covalent_bond_candidates(array) == []


def test_the_residue_map_distinguishes_insertion_codes_without_chain_info():
    """Without chain_info the observed order *is* the model order, so codes work."""
    atoms = _chain_with_insertion_codes()
    records = chain_records(atoms)
    mapping = _residue_index_map(records, None)

    assert mapping[("A", 100, "")] == 0
    assert mapping[("A", 100, "A")] == 1
    assert mapping[("A", 100, "B")] == 2


def test_a_chain_info_backed_chain_with_insertion_codes_is_refused_not_guessed():
    """chain_info records no insertion codes, so the alignment is unknowable.

    Previously the map simply overwrote, keeping the last residue -- which
    attaches labels and bonds to the wrong one without saying so.
    """
    atoms = _chain_with_insertion_codes()
    records = chain_records(atoms)
    report = AdapterReport()

    mapping = _residue_index_map(
        records, {"A": {"res_id": [100, 100, 100], "res_name": ["CYS"] * 3}}, report
    )
    assert mapping == {}
    assert report.unrepresentable_insertion_codes == ["A"]


def test_an_unplaceable_bond_raises_rather_than_folding_disconnected(parsed, ccd):
    """Skipping would fold a connected system as though it were not.

    The direct API returns no report, so a caller has nothing to inspect; the
    only way the fact reaches them is by raising.
    """
    import pytest

    from esmfold2_atomworks.data.atomworks_to_esm import (
        atom_array_to_structure_prediction_input,
    )
    from esmfold2_atomworks.data.spec import CovalentBondResolutionError

    atoms, chain_info = parsed("hemoglobin")
    bonded = _bond_to_a_residue_outside_the_model(atoms)

    with pytest.raises(CovalentBondResolutionError, match="could not be placed"):
        atom_array_to_structure_prediction_input(bonded, chain_info=chain_info)


def test_the_caller_can_opt_into_the_disconnected_reading(parsed, ccd):
    from esmfold2_atomworks.data.atomworks_to_esm import (
        AdapterReport,
        atom_array_to_structure_prediction_input,
    )

    atoms, chain_info = parsed("hemoglobin")
    bonded = _bond_to_a_residue_outside_the_model(atoms)

    report = AdapterReport()
    atom_array_to_structure_prediction_input(
        bonded,
        chain_info=chain_info,
        allow_unresolved_covalent_bonds=True,
        report=report,
    )
    assert report.unresolved_covalent_bonds


def _bond_to_a_residue_outside_the_model(atoms):
    """A bond whose endpoint the model's indexing cannot name.

    Water is dropped before the model sees it, so a bond reaching it is real in
    the source and unplaceable downstream -- exactly the case that must not be
    silently dropped.
    """
    import biotite.structure as struc

    chain = np.asarray(atoms.chain_id).astype(str)
    name = np.asarray(atoms.atom_name).astype(str)
    res_id = np.asarray(atoms.res_id).astype(int)

    his = np.where((chain == "A") & (res_id == 87) & (name == "NE2"))[0]
    iron = np.where((chain == "E") & (name == "FE"))[0]
    assert len(his) == 1 and len(iron) == 1

    bonded = atoms.copy()
    bonded.bonds.add_bond(int(his[0]), int(iron[0]), struc.BondType.SINGLE)
    # Renumber the haem so the residue map no longer covers it.
    res_ids = np.asarray(bonded.res_id).copy()
    res_ids[np.asarray(bonded.chain_id).astype(str) == "E"] = 9999
    bonded.set_annotation("res_id", res_ids)
    return bonded


def test_a_chain_kind_guessed_without_chain_type_is_marked_as_such():
    """Without `chain_type` the kind is inferred, and says so.

    `is_polymer` cannot distinguish protein from nucleic acid, so a DNA chain
    arriving this way is called protein. Refusing would reject every hand-built
    AtomArray, so the guess stands -- but a caller can tell it from a statement.
    Anything from `atomworks.io.parse` or the component assembler carries
    `chain_type` and never takes this path.
    """
    atoms = _chain_with_insertion_codes()  # has is_polymer, no chain_type
    (record,) = chain_records(atoms)
    assert record.chain_type is None
    assert record.kind == "protein"
    assert record.kind_is_inferred is True


def test_a_chain_kind_from_chain_type_is_not_marked_inferred(parsed):
    atoms, chain_info = parsed("hemoglobin")
    for record in chain_records(atoms, chain_info=chain_info):
        assert record.kind_is_inferred is False
