"""Confidence values read off a ``MolecularComplexResult``.

Two rules shape this module.

**A metric the model did not produce stays absent.** Never defaulted to 0.0: a
fabricated zero reads downstream as a real measurement, and for pLDDT (where
higher is better) it reads as a catastrophic fold, while for PAE it reads as a
perfect one. Absence is recoverable; a plausible wrong number is not.

**Global means can hide the confidence between entities.** On a 112-residue
protein with a 19-atom ligand, three quarters of the PAE matrix is
protein-internal, so ``mean_pae`` mostly reports how well the monomer folded; a
prediction can improve it by folding the core better while placing the ligand
worse. The cross-entity and ligand-restricted values are therefore emitted
beside the global ones.

Units: **pLDDT here is on 0--1**, as the model reports it -- not the 0--100 of
AlphaFold's B-factor column. Scale at the point of display, not here.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

__all__ = [
    "NON_POLYMER",
    "SCALAR_METRIC_SOURCES",
    "fold_metrics",
    "plddt_per_residue",
    "plddt_per_token",
]

#: How ``MolecularComplexMetadata.entity_lookup`` spells a non-polymer entity.
#: The other value is ``"polymer"``.
NON_POLYMER = "non-polymer"

#: Scalar fields of the result, mapped to the metric names they are emitted as.
#:
#: Public so that a consumer can check the namespace without copying the table:
#: a copy diverges, and a key that stops being emitted then looks exactly like a
#: legitimately absent metric. Read-only, because :func:`fold_metrics` reads it
#: and a consumer's edit would change what it emits for everyone. *Scalar*
#: rather than "all metrics": :func:`fold_metrics` also emits conditional ones
#: that are not read off a single field -- the cross-entity PAE and the pLDDT
#: split.
#:
#: ``esm.mean_plddt`` is the mean of ``result.plddt``, which is **token** space:
#: an atomized ligand weighs in once per atom, not once as a residue. See
#: :func:`plddt_per_token`.
SCALAR_METRIC_SOURCES: Mapping[str, str] = MappingProxyType(
    {
        "esm.ptm": "ptm",
        "esm.iptm": "iptm",
        "esm.mean_plddt": "plddt",
    }
)


def _as_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu()
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return array if array.size else None


def _scalar(value: Any) -> float | None:
    array = _as_array(value)
    if array is None:
        return None
    return float(array.reshape(-1)[0]) if array.size == 1 else float(array.mean())


def fold_metrics(result: Any) -> dict[str, float]:
    """Namespaced confidence values for one prediction."""
    metrics: dict[str, float] = {}
    for key, attribute in SCALAR_METRIC_SOURCES.items():
        value = _scalar(getattr(result, attribute, None))
        if value is not None:
            metrics[key] = value

    pae = getattr(result, "pae", None)
    mean_pae = _scalar(pae)
    if mean_pae is not None:
        metrics["esm.mean_pae"] = mean_pae

    metrics.update(_interface_metrics(result, pae))
    metrics.update(_plddt_split(result))
    return metrics


def _interface_metrics(result: Any, pae: Any) -> dict[str, float]:
    """Cross-entity confidence, which the global means dilute in a complex.

    ``pair_chains_iptm`` is **asymmetric** -- the model reports a per-chain-pair
    matrix whose two directions differ -- so there is no single "the" pair ipTM
    to emit. The two reductions below are well defined for any chain count and
    are named for what they are; inventing a canonical direction would bury a
    choice inside something that looks like a measurement.
    """
    metrics: dict[str, float] = {}

    entity_id = _as_array(getattr(result, "entity_id", None))
    pae_array = _as_array(pae)
    if entity_id is not None and pae_array is not None and pae_array.ndim == 2:
        cross = entity_id[:, None] != entity_id[None, :]
        if cross.any() and cross.shape == pae_array.shape:
            metrics["esm.interface_pae"] = float(pae_array[cross].mean())

    pair = _as_array(getattr(result, "pair_chains_iptm", None))
    if pair is not None and pair.ndim == 2 and pair.shape[0] == pair.shape[1] >= 2:
        off_diagonal = pair[~np.eye(pair.shape[0], dtype=bool)]
        metrics["esm.pair_chain_iptm_min"] = float(off_diagonal.min())
        metrics["esm.pair_chain_iptm_mean"] = float(off_diagonal.mean())
    return metrics


def _entity_kinds(result: Any) -> dict[int, str] | None:
    """``entity_id -> "polymer" | "non-polymer"``, as the model declared it.

    Written by ``build_molecular_complex_from_features`` from each chain's
    ``mol_type``, so it is the model's own statement about what it folded,
    keyed by the same ids ``result.entity_id`` labels each token with.
    """
    metadata = getattr(getattr(result, "complex", None), "metadata", None)
    lookup = getattr(metadata, "entity_lookup", None)
    if not isinstance(lookup, dict) or not lookup:
        return None
    try:
        return {int(entity): str(kind) for entity, kind in lookup.items()}
    except (TypeError, ValueError):
        return None


def _plddt_split(result: Any) -> dict[str, float]:
    """pLDDT restricted to the ligand, contrasted against the protein's.

    **Which entity is the polymer is read off the result, never assumed.**
    Entity *order* would usually work -- the adapter emits polymers before
    ligands, and entities are numbered in input order -- but that is a property
    of how the input happened to be assembled, not of the result. A caller
    passing a ligand first would swap the two metrics while both stayed in
    range, and a swapped split reports the protein's 0.9 as the ligand's, which
    reads exactly like the success the metric exists to detect.

    So the split is either established from ``entity_lookup`` or the metrics are
    absent: no lookup, a token whose entity the lookup does not describe, or a
    complex with only one side, and nothing is emitted.
    """
    plddt = _as_array(getattr(result, "plddt", None))
    entity_id = _as_array(getattr(result, "entity_id", None))
    kinds = _entity_kinds(result)
    if plddt is None or entity_id is None or kinds is None:
        return {}
    if plddt.ndim != 1 or entity_id.shape != plddt.shape:
        return {}

    present = np.unique(entity_id)
    labels = [kinds.get(int(entity)) for entity in present]
    if any(label is None for label in labels):
        # A token carrying an entity the model did not describe cannot be put on
        # either side, and assigning it to the majority is the guess this
        # function exists to avoid.
        return {}

    ligand_entities = [
        entity
        for entity, label in zip(present, labels, strict=True)
        if label == NON_POLYMER
    ]
    if not ligand_entities:
        return {}
    ligand = np.isin(entity_id, ligand_entities)
    if ligand.all():
        # No polymer to contrast against; a "protein pLDDT" over zero residues
        # is not a measurement.
        return {}

    return {
        "esm.protein_plddt": float(plddt[~ligand].mean()),
        "esm.ligand_plddt": float(plddt[ligand].mean()),
        # The worst per-entity mean: with two ligands of unequal size the pooled
        # mean is dominated by the larger, so a small ligand placed badly
        # disappears into a number that still looks fine.
        "esm.ligand_plddt_min": min(
            float(plddt[entity_id == entity].mean()) for entity in ligand_entities
        ),
    }


def plddt_per_token(result: Any) -> np.ndarray | None:
    """pLDDT in **model-token** space, or None.

    One value per token the model scored. A ligand is atomized, so it
    contributes one token per atom, and a modified residue one per atom too.
    This is ``result.plddt``, the axis every other per-token field -- PAE,
    ``entity_id`` -- is indexed by, and the one ``esm.mean_plddt`` averages.
    """
    return _as_array(getattr(result, "plddt", None))


def plddt_per_residue(result: Any) -> np.ndarray | None:
    """pLDDT in collapsed **residue** space, or None.

    One value per output residue, each the mean of that residue's tokens, and a
    non-polymer chain collapsed into one residue (``esm/models/esmfold2/output.py``).
    This is ``result.complex.plddt``: the axis of the residues of the
    ``AtomArray`` that :func:`~esmfold2_atomworks.data.molecular_complex.result_to_atom_array`
    returns, in their order.

    **The two axes differ in length whenever a ligand or a modified residue is
    present.** A 112-residue protein with a 19-atom ligand gives 131 tokens and
    113 residues. Indexing one with the other is silently wrong on exactly the
    systems this package exists for, and right on a plain monomer, where the two
    coincide -- which is why each has a name.
    """
    return _as_array(getattr(getattr(result, "complex", None), "plddt", None))
