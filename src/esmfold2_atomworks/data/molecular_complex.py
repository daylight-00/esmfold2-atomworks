"""``MolecularComplex`` -> ``AtomArray``: the return leg of the adapter.

The obvious route is ``MolecularComplex.to_mmcif()`` -> ``atomworks.io.parse()``.
It is in memory, and it hands back AtomWorks' full annotation set for free.

It is also **silently lossy for ligands**. ESMFold2 labels a SMILES-specified
ligand ``LIG``; ``LIG`` is a real CCD code for an unrelated molecule, so
AtomWorks reconciles the ligand against that component and keeps only the atoms
whose names happen to match. In one observed case that turned 19 correct ligand
atoms into 8 carbons -- and the result still parsed and still validated.
Nothing downstream noticed.

So the flat arrays are read directly instead. ``MolecularComplex`` already
stores per-atom positions, names, elements and hetero flags, and a token->atom
index table; nothing here is inferred that the model did not report.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from biotite.structure import AtomArray

__all__ = [
    "molecular_complex_to_atom_array",
    "rename_ligand_residues",
    "result_to_atom_array",
]


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _token_atom_spans(token_to_atoms: Any, n_atoms: int) -> list[np.ndarray]:
    """Per-token atom indices, tolerant of the two shapes this field takes.

    Either an ``(n_tokens, 2)`` ``[start, end)`` range table -- what the current
    ``MolecularComplex`` documents -- or an ``(n_tokens, k)`` padded index table.
    Deciding from the data rather than pinning a version keeps this working
    across ESM releases, which is worth the few lines given that both upstreams
    are explicitly mid-cleanup.
    """
    table = _to_numpy(token_to_atoms)
    if table.ndim == 1:
        return [np.asarray([int(i)]) for i in table if 0 <= int(i) < n_atoms]

    if table.shape[1] == 2 and bool(np.all(table[:, 1] >= table[:, 0])):
        starts, ends = table[:, 0].astype(int), table[:, 1].astype(int)
        # An end is exclusive if the largest one reaches n_atoms.
        exclusive = int(ends.max()) >= n_atoms
        spans = [
            np.arange(s, e if exclusive else e + 1)
            for s, e in zip(starts, ends, strict=True)
        ]
        if all(len(s) > 0 for s in spans) and sum(len(s) for s in spans) == n_atoms:
            return spans

    return [row[(row >= 0) & (row < n_atoms)].astype(int) for row in table]


def _chain_labels(complex_: Any) -> list[str]:
    """Per-token chain labels, as letters rather than indices.

    ``MolecularComplex.chain_id`` holds integer asym ids; the original letters
    live in ``metadata.chain_lookup``. Stringifying the integers would name the
    chains "0"/"1", which then fail to match the chain the caller asked about.
    """
    raw = _to_numpy(complex_.chain_id).reshape(-1)
    lookup: dict[int, str] = {}
    metadata = getattr(complex_, "metadata", None)
    candidate = getattr(metadata, "chain_lookup", None)
    if isinstance(candidate, (list, tuple)) and candidate:
        candidate = candidate[0]
    if isinstance(candidate, dict):
        lookup = {int(k): str(v) for k, v in candidate.items()}

    if np.issubdtype(raw.dtype, np.number):
        return [lookup.get(int(value), str(int(value))) for value in raw]
    return [str(value) for value in raw]


def molecular_complex_to_atom_array(complex_: Any) -> AtomArray:
    """Build an ``AtomArray`` straight from the complex's flat arrays.

    ``res_id`` is assigned per chain in token order, so it is consecutive and
    cannot be shifted by an insertion code that a predicted structure does not
    have.
    """
    import biotite.structure as struc

    positions = _to_numpy(complex_.atom_positions).reshape(-1, 3)
    atom_names = _to_numpy(complex_.atom_names).astype(str).reshape(-1)
    elements = _to_numpy(complex_.atom_elements).astype(str).reshape(-1)
    hetero = _to_numpy(complex_.atom_hetero).astype(bool).reshape(-1)
    n_atoms = len(positions)

    residue_names = [str(name) for name in complex_.sequence]
    chain_ids = _chain_labels(complex_)
    spans = _token_atom_spans(complex_.token_to_atoms, n_atoms)

    if not (len(residue_names) == len(chain_ids) == len(spans)):
        raise ValueError(
            "MolecularComplex is inconsistent: "
            f"{len(residue_names)} residue names, {len(chain_ids)} chain ids, "
            f"{len(spans)} token spans"
        )

    per_atom_chain = np.empty(n_atoms, dtype="U8")
    per_atom_res_name = np.empty(n_atoms, dtype="U5")
    per_atom_res_id = np.zeros(n_atoms, dtype=int)

    counters: dict[str, int] = {}
    covered = np.zeros(n_atoms, dtype=bool)
    for token, span in enumerate(spans):
        if len(span) == 0:
            continue
        chain = str(chain_ids[token])
        counters[chain] = counters.get(chain, 0) + 1
        per_atom_chain[span] = chain
        per_atom_res_name[span] = residue_names[token][:5]
        per_atom_res_id[span] = counters[chain]
        covered[span] = True

    if not covered.all():
        # Every atom must belong to a token, or atoms vanish silently -- which
        # is the failure this module exists to remove.
        raise ValueError(
            f"{int((~covered).sum())} of {n_atoms} atoms are claimed by no token "
            "span; refusing to build a partial structure"
        )

    array = struc.AtomArray(n_atoms)
    array.coord = positions.astype(np.float32)
    array.set_annotation("chain_id", per_atom_chain)
    array.set_annotation("res_id", per_atom_res_id)
    array.set_annotation("res_name", per_atom_res_name)
    array.set_annotation("atom_name", atom_names.astype("U6"))
    array.set_annotation("element", np.char.upper(elements).astype("U2"))
    array.set_annotation("hetero", hetero)
    array.set_annotation("ins_code", np.full(n_atoms, "", dtype="U1"))
    # The polymer/ligand split comes from the model's own hetero flags rather
    # than a CCD lookup, which is what lets the ligand survive intact.
    array.set_annotation("is_polymer", ~hetero)
    return array


def rename_ligand_residues(atoms: AtomArray, residue_name: str) -> AtomArray:
    """Relabel non-polymer residues, returning a copy.

    ESMFold2's generic ``LIG`` is not the caller's residue name, and anything
    that matches ligands by name -- Rosetta params, a metric that selects the
    ligand chain -- needs the two sides to agree.
    """
    result = atoms.copy()
    mask = np.asarray(result.hetero, dtype=bool)
    if mask.any():
        res_name = np.asarray(result.res_name, dtype="U5").copy()
        res_name[mask] = residue_name[:5]
        result.set_annotation("res_name", res_name)
    return result


def result_to_atom_array(
    result: Any,
    *,
    ligand_residue_name: str | None = None,
) -> AtomArray:
    """The ``AtomArray`` for a ``MolecularComplexResult``.

    Args:
        result: what ``ESMFold2InputBuilder.fold`` returned.
        ligand_residue_name: relabel non-polymer residues to this name.
    """
    atoms = molecular_complex_to_atom_array(result.complex)
    if ligand_residue_name is not None:
        atoms = rename_ligand_residues(atoms, ligand_residue_name)
    return atoms
