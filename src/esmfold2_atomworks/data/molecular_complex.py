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

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from biotite.structure import AtomArray

__all__ = [
    "apply_ligand_labels",
    "check_residue_name",
    "copy_chain_types",
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

    # The spans have to partition the atoms: every atom in exactly one, and no
    # span empty. An unclaimed atom would vanish, a doubly claimed one would
    # be silently handed to whichever token came last, and an empty span would
    # drop a residue that the complex's per-residue arrays still count -- each
    # the failure this module exists to remove.
    claims = np.zeros(n_atoms, dtype=int)
    for span in spans:
        np.add.at(claims, span, 1)
    unclaimed = int((claims == 0).sum())
    shared = int((claims > 1).sum())
    empty = sum(1 for span in spans if len(span) == 0)
    if unclaimed or shared or empty:
        raise ValueError(
            f"{unclaimed} of {n_atoms} atoms are claimed by no token span and "
            f"{shared} by more than one, and {empty} token span(s) claim none: the "
            "spans must partition the atoms, so refusing to build a structure "
            "that drops or reassigns atoms, or loses a residue"
        )

    per_atom_chain = np.empty(n_atoms, dtype="U8")
    per_atom_res_name = np.empty(n_atoms, dtype="U5")
    per_atom_res_id = np.zeros(n_atoms, dtype=int)

    counters: dict[str, int] = {}
    for token, span in enumerate(spans):
        chain = str(chain_ids[token])
        counters[chain] = counters.get(chain, 0) + 1
        per_atom_chain[span] = chain
        per_atom_res_name[span] = residue_names[token][:5]
        per_atom_res_id[span] = counters[chain]

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


def check_residue_name(residue_name: str) -> None:
    """Raise unless *residue_name* fits a residue name: one to five characters."""
    if not isinstance(residue_name, str) or not 1 <= len(residue_name) <= 5:
        raise ValueError(
            f"{residue_name!r} cannot be a residue name, which holds one to five "
            "characters"
        )


def rename_ligand_residues(atoms: AtomArray, residue_name: str) -> AtomArray:
    """Give every hetero residue *residue_name*, returning a copy.

    ESMFold2's generic ``LIG`` is not the caller's residue name, and anything
    that matches ligands by name -- Rosetta params, a metric that selects the
    ligand chain -- needs the two sides to agree.

    **Every** hetero residue: ESMFold2 marks each non-polymer chain hetero and
    nothing else (a modified polymer residue is not), so an ion or a cofactor
    beside the ligand is renamed with it. This is for an output whose hetero
    residues are all one molecule. For several, declare each chain with a
    ``LigandSpec`` and use :func:`apply_ligand_labels`.

    Raises:
        ValueError: a name a residue name cannot hold -- empty, or longer than
            its five characters. Truncating it would hand back a name the
            caller never gave, which then matches nothing.
    """
    check_residue_name(residue_name)
    result = atoms.copy()
    mask = np.asarray(result.hetero, dtype=bool)
    if mask.any():
        res_name = np.asarray(result.res_name, dtype="U5").copy()
        res_name[mask] = residue_name
        result.set_annotation("res_name", res_name)
    return result


def apply_ligand_labels(structure: AtomArray, ligands: Mapping[str, Any]) -> AtomArray:
    """Give each declared ligand chain its ``LigandSpec.label``, as a copy.

    *ligands* maps output chain labels to specs. A multi-component declaration
    has no label and is left as the model wrote it.

    Raises:
        ValueError: a declared chain missing from the output or holding
            polymer atoms.
    """
    chains = np.asarray(structure.chain_id).astype(str)
    hetero = np.asarray(structure.hetero, dtype=bool)
    res_name = np.asarray(structure.res_name, dtype="U5").copy()
    for chain, spec in ligands.items():
        label = spec.label
        if label is None:
            continue
        mask = chains == str(chain)
        if not mask.any():
            raise ValueError(f"ligand chain {chain!r} is not in the folded structure")
        if not hetero[mask].all():
            raise ValueError(f"ligand chain {chain!r} holds polymer atoms")
        res_name[mask] = label
    result = structure.copy()
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
        ligand_residue_name: give every hetero residue this name; see
            :func:`rename_ligand_residues` for what "every" includes.
    """
    atoms = molecular_complex_to_atom_array(result.complex)
    if ligand_residue_name is not None:
        atoms = rename_ligand_residues(atoms, ligand_residue_name)
    return atoms


def copy_chain_types(
    structure: AtomArray, source: AtomArray, *, chain_key: str = "chain_id"
) -> AtomArray:
    """Give ``structure`` the ``chain_type`` its chains had in ``source``, as a copy.

    A folded structure does not carry the AtomWorks ``chain_type`` of the
    chains it came from, and cannot recover it: the model's output knows
    polymer from non-polymer, not an L- from a D-polypeptide, or DNA from RNA.
    The source structure does, so this copies it back, chain by chain -- the
    output's chain labels are the source's ``chain_key`` labels.

    Only a source that carries ``chain_type`` has one to copy; otherwise
    ``structure`` is returned unchanged, and a kind declared through
    ``chain_kinds`` is not turned into a ``ChainType``, which would state more
    than the declaration did. An output chain the source does not name raises
    rather than being left unannotated beside annotated ones.
    """
    if "chain_type" not in source.get_annotation_categories():
        return structure
    labels = np.asarray(source.get_annotation(chain_key)).astype(str)
    types = np.asarray(source.get_annotation("chain_type"))
    by_chain: dict[str, Any] = {}
    for label in np.unique(labels):
        values = np.unique(types[labels == label])
        if len(values) != 1:
            raise ValueError(
                f"source chain {label!r} holds {len(values)} chain types; one "
                "chain must be one molecule"
            )
        by_chain[str(label)] = values[0]

    out_labels = np.asarray(structure.chain_id).astype(str)
    unknown = sorted({str(label) for label in out_labels} - set(by_chain))
    if unknown:
        raise ValueError(
            f"output chains {unknown} have no chain in the source under "
            f"{chain_key!r}; their chain_type is unknown"
        )
    result = structure.copy()
    result.set_annotation(
        "chain_type",
        np.array([by_chain[str(label)] for label in out_labels], dtype=types.dtype),
    )
    return result
