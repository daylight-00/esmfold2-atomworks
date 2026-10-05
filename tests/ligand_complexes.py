"""A protein chain and one small ligand, built by hand, for tests of how a ligand's
atoms are matched to ESMFold2's.

Chain A is three glycines and chain B the ligand. Every atom gets its own
coordinate, so a label that landed on the wrong atom is visible by its value.
"""

from __future__ import annotations

import numpy as np

SINGLE, DOUBLE = 1, 2  # biotite.structure.BondType


def protein_and_ligand(names, bonds, *, bond_to=None):
    """Chain A: GGG. Chain B: a ligand of atoms named *names* (the element is the
    first letter), bonded inside by *bonds* (index pair, bond type), with Gly 2's C
    bonded to atom *bond_to* when that is given.
    """
    import biotite.structure as struc

    rows = [
        ("A", i, name, name[0]) for i in (1, 2, 3) for name in ("N", "CA", "C", "O")
    ]
    rows += [("B", 1, name, name[0]) for name in names]
    atoms = struc.AtomArray(len(rows))
    atoms.coord = np.arange(len(rows) * 3, dtype=np.float32).reshape(-1, 3)
    atoms.set_annotation("chain_id", np.array([r[0] for r in rows], dtype="U4"))
    atoms.set_annotation("res_id", np.array([r[1] for r in rows]))
    atoms.set_annotation("ins_code", np.array([""] * len(rows), dtype="U1"))
    atoms.set_annotation(
        "res_name",
        np.array(["GLY" if r[0] == "A" else "LIG" for r in rows], dtype="U5"),
    )
    atoms.set_annotation("atom_name", np.array([r[2] for r in rows], dtype="U6"))
    atoms.set_annotation("element", np.array([r[3] for r in rows], dtype="U2"))
    atoms.set_annotation("is_polymer", np.array([r[0] == "A" for r in rows]))
    atoms.bonds = struc.BondList(len(rows))
    first = 12
    for a, b, kind in bonds:
        atoms.bonds.add_bond(first + a, first + b, kind)
    if bond_to is not None:
        carbonyl = next(i for i, r in enumerate(rows) if r[:3] == ("A", 2, "C"))
        atoms.bonds.add_bond(carbonyl, first + bond_to, struc.BondType.SINGLE)
    return atoms


def protein_and_acetic_acid(names, bonds, *, bond_to):
    """The same with four atoms (C, C, O, O), the ligand drawn as CC(=O)O."""
    return protein_and_ligand(names, bonds, bond_to=bond_to)
