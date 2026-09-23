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

from esmfold2_foundry.data.atomworks_to_esm import (
    AdapterReport,
    _residue_index_map,
    chain_records,
)
from esmfold2_foundry.data.bonds import covalent_bond_candidates


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


def test_a_backbone_link_across_an_insertion_code_is_still_declared():
    """100/C -- 100A/N is not "consecutive" in the numbering, so it is declared."""
    import biotite.structure as struc

    spec = [("A", 100, "", "C", "C"), ("A", 100, "A", "N", "N")]
    array = struc.AtomArray(len(spec))
    array.coord = np.zeros((len(spec), 3), dtype=np.float32)
    array.set_annotation("chain_id", np.array(["A"] * len(spec), dtype="U4"))
    array.set_annotation("res_id", np.array([s[1] for s in spec], dtype=int))
    array.set_annotation("ins_code", np.array([s[2] for s in spec], dtype="U1"))
    array.set_annotation("res_name", np.array(["GLY"] * len(spec), dtype="U5"))
    array.set_annotation("atom_name", np.array([s[3] for s in spec], dtype="U6"))
    array.set_annotation("element", np.array([s[4] for s in spec], dtype="U2"))
    array.bonds = struc.BondList(len(spec))
    array.bonds.add_bond(0, 1, struc.BondType.SINGLE)

    (candidate,) = covalent_bond_candidates(array)
    assert candidate.describe() == "A/100/C - A/100A/N"


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
