"""Covalent bonds that ESMFold2 cannot infer, carried across from AtomWorks.

ESMFold2 rebuilds most connectivity by itself: bonds *within* a residue come
from the CCD component, and the polymer backbone comes from the sequence. What
it cannot know is everything else -- a ligand bonded to a cysteine, a crosslink
between two chains, a sugar attached to an asparagine. Those must be declared
through ``StructurePredictionInput.covalent_bonds`` or the model folds the
pieces as though they were unconnected, which is a confident prediction of the
wrong molecule.

**Indices are read back from the tokenizer, not re-derived.** ESM's
``CovalentBond`` addresses an atom as ``(chain_id, res_idx, atom_idx)`` where
``res_idx`` is the tokenizer's residue index within the chain and ``atom_idx``
indexes that residue's atoms *in the order the tokenizer built them* -- CCD
order for a ligand, ``PROTEIN_HEAVY_ATOMS`` order for a standard residue, minus
leaving atoms. Reimplementing that ordering here would duplicate upstream logic
that changes independently of this package, which is the mistake D-001 exists
to prevent. So the structure is featurized once without bonds, the atom
ordering is decoded from the resulting ``ref_atom_name_chars`` and
``atom_to_token``, and the indices are looked up in it.

The cost is one extra featurization, paid only when a structure actually has a
bond to declare -- detection is a scan of the ``BondList`` and most structures
have none.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from biotite.structure import AtomArray

__all__ = [
    "BondCandidate",
    "covalent_bond_candidates",
    "resolve_covalent_bonds",
]

#: Backbone links ESMFold2 reconstructs from the sequence. A bond between
#: consecutive residues of one chain using these atom names is the polymer
#: backbone itself and must not be declared -- doing so would also force the
#: chain out of entity deduplication for no reason.
_POLYMER_LINKS = frozenset({("C", "N"), ("N", "C"), ("O3'", "P"), ("P", "O3'")})


@dataclass(frozen=True)
class BondCandidate:
    """One covalent bond, in the source structure's own terms.

    A residue is identified by ``(chain, res_id, ins_code)``. The insertion
    code is not decoration: a deposited chain may number residues 100, 100A,
    100B, and keying on ``res_id`` alone makes those one residue -- which turns
    a genuine bond between two of them into an intra-residue bond and drops it.
    """

    chain_1: str
    res_id_1: int
    ins_code_1: str
    atom_name_1: str
    chain_2: str
    res_id_2: int
    ins_code_2: str
    atom_name_2: str

    @property
    def residue_1(self) -> tuple[str, int, str]:
        return (self.chain_1, self.res_id_1, self.ins_code_1)

    @property
    def residue_2(self) -> tuple[str, int, str]:
        return (self.chain_2, self.res_id_2, self.ins_code_2)

    def describe(self) -> str:
        return (
            f"{self.chain_1}/{self.res_id_1}{self.ins_code_1}/{self.atom_name_1}"
            f" - {self.chain_2}/{self.res_id_2}{self.ins_code_2}/{self.atom_name_2}"
        )


def covalent_bond_candidates(
    atoms: AtomArray,
    *,
    chain_key: str = "chain_id",
    keep_chains: frozenset[str] | None = None,
) -> list[BondCandidate]:
    """Bonds in *atoms* that ESMFold2 would not otherwise know about.

    Excluded, in order: bonds inside one residue (the CCD supplies them), the
    polymer backbone between consecutive residues (the sequence supplies it),
    anything touching hydrogen (ESMFold2 models heavy atoms), and anything
    reaching a chain that will not be sent to the model at all.

    Args:
        keep_chains: chains that survive into the model input. Bonds to any
            other chain are dropped, because an index into a chain that does
            not exist is an error upstream rather than a no-op.
    """
    if atoms.bonds is None:
        return []

    bond_array = atoms.bonds.as_array()
    if len(bond_array) == 0:
        return []

    chain = np.asarray(atoms.get_annotation(chain_key)).astype(str)
    res_id = np.asarray(atoms.res_id).astype(int)
    atom_name = np.asarray(atoms.atom_name).astype(str)
    element = np.asarray(atoms.element).astype(str)
    if "ins_code" in set(atoms.get_annotation_categories()):
        ins_code = np.asarray(atoms.get_annotation("ins_code")).astype(str)
    else:
        ins_code = np.full(len(atoms), "", dtype="U1")

    candidates: list[BondCandidate] = []
    seen: set[tuple] = set()
    for i, j, _bond_type in bond_array:
        if element[i].upper() in ("H", "D") or element[j].upper() in ("H", "D"):
            continue
        chain_i, chain_j = chain[i], chain[j]
        if keep_chains is not None and (
            chain_i not in keep_chains or chain_j not in keep_chains
        ):
            continue

        same_residue = (
            chain_i == chain_j and res_id[i] == res_id[j] and ins_code[i] == ins_code[j]
        )
        if same_residue:
            continue

        # The backbone exemption applies only between consecutive *plain*
        # residues. Between 100 and 100A the numbering does not say they are
        # adjacent, so such a bond is declared rather than assumed.
        if (
            chain_i == chain_j
            and not ins_code[i]
            and not ins_code[j]
            and abs(int(res_id[i]) - int(res_id[j])) == 1
            and (atom_name[i], atom_name[j]) in _POLYMER_LINKS
        ):
            continue

        # Order the endpoints so the same bond is not emitted twice.
        key = tuple(
            sorted(
                [
                    (chain_i, int(res_id[i]), ins_code[i], atom_name[i]),
                    (chain_j, int(res_id[j]), ins_code[j], atom_name[j]),
                ]
            )
        )
        if key in seen:
            continue
        seen.add(key)
        (c1, r1, i1, a1), (c2, r2, i2, a2) = key
        candidates.append(BondCandidate(c1, r1, i1, a1, c2, r2, i2, a2))
    return candidates


def _atom_names_by_residue(
    features: dict[str, Any], chain_infos: list[Any]
) -> dict[tuple[str, int], list[str]]:
    """``(chain_id, residue_index) -> atom names`` in the tokenizer's order.

    Decoded from the featurizer's own output: ``ref_atom_name_chars`` holds each
    atom's name as ``ord(c) - 32`` in four slots, and ``atom_to_token`` says
    which token each atom belongs to. ``ChainInfo.tokens`` then maps a token
    back to its chain and residue.
    """
    name_chars = features["ref_atom_name_chars"]
    atom_to_token = features["atom_to_token"]
    atom_mask = features["atom_attention_mask"]
    if hasattr(name_chars, "detach"):
        name_chars = name_chars.detach().cpu()
        atom_to_token = atom_to_token.detach().cpu()
        atom_mask = atom_mask.detach().cpu()
    name_chars = np.asarray(name_chars)
    atom_to_token = np.asarray(atom_to_token).astype(int)
    atom_mask = np.asarray(atom_mask).astype(bool)

    token_location: dict[int, tuple[str, int]] = {}
    for info in chain_infos:
        for token in info.tokens:
            token_location[int(token.token_index)] = (
                str(info.chain_id),
                int(token.residue_index),
            )

    ordering: dict[tuple[str, int], list[str]] = {}
    for atom_index in range(len(atom_to_token)):
        if not atom_mask[atom_index]:
            continue
        location = token_location.get(int(atom_to_token[atom_index]))
        if location is None:
            continue
        name = "".join(
            chr(int(code) + 32) for code in name_chars[atom_index] if int(code) != 0
        ).strip()
        ordering.setdefault(location, []).append(name)
    return ordering


def resolve_covalent_bonds(
    candidates: list[BondCandidate],
    features: dict[str, Any],
    chain_infos: list[Any],
    residue_index_of: dict[tuple[str, int], int],
) -> tuple[list[Any], list[str]]:
    """Turn source-space bonds into ESM ``CovalentBond`` objects.

    Args:
        residue_index_of: ``(chain_id, res_id, ins_code) -> tokenizer residue
            index``. The source numbering is the deposited one: it need not
            start at zero, need not be contiguous, and may repeat a number
            under different insertion codes, so it cannot be used directly.

    Returns:
        ``(bonds, skipped)`` -- the resolved bonds, and a human-readable reason
        per candidate that could not be placed. A bond is skipped rather than
        guessed: upstream raises on an out-of-range index, and a *wrong* index
        would silently bond the wrong pair of atoms.
    """
    from esm.models.esmfold2.types import CovalentBond

    ordering = _atom_names_by_residue(features, chain_infos)

    bonds: list[Any] = []
    skipped: list[str] = []
    for candidate in candidates:
        resolved = []
        failure = None
        for (chain_id, res_id, ins_code), atom_name in (
            (candidate.residue_1, candidate.atom_name_1),
            (candidate.residue_2, candidate.atom_name_2),
        ):
            residue_index = residue_index_of.get((chain_id, res_id, ins_code))
            if residue_index is None:
                failure = (
                    f"residue {chain_id}/{res_id}{ins_code} is not in the model input"
                )
                break
            names = ordering.get((chain_id, residue_index))
            if names is None:
                failure = (
                    f"residue index {residue_index} of chain {chain_id} has no atoms"
                )
                break
            if atom_name not in names:
                failure = (
                    f"atom {atom_name!r} is not among the tokenizer's atoms for "
                    f"{chain_id}/{res_id}{ins_code} ({names})"
                )
                break
            resolved.append((chain_id, residue_index, names.index(atom_name)))

        if failure is not None:
            skipped.append(f"{candidate.describe()}: {failure}")
            continue

        (chain_1, res_1, atom_1), (chain_2, res_2, atom_2) = resolved
        bonds.append(
            CovalentBond(
                # Plain str, not numpy's: these are serialized to JSON for the
                # hosted API, where a numpy scalar is not encodable.
                chain_id1=str(chain_1),
                res_idx1=int(res_1),
                atom_idx1=int(atom_1),
                chain_id2=str(chain_2),
                res_idx2=int(res_2),
                atom_idx2=int(atom_2),
            )
        )
    return bonds, skipped
