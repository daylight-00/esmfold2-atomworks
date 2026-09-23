"""Source coordinates, aligned to ESMFold2's atom ordering.

``StructurePredictionInput`` carries no coordinates: it is a sequence- and
chemistry-level description, and ESMFold2 derives all geometry from CCD
reference conformers. So the structure's own coordinates never reach the model,
and a pipeline built only from the adapter has inputs but no supervision
targets. ``feats["gt_coords"]`` is not them either -- it is built from the
prediction input and is zeros at inference.

This module supplies the missing half: the deposited coordinates, permuted into
the order the model's atom axis uses, with a mask saying which of those atoms
the source actually provides.

**It deliberately stops short of a loss.** Where the supervision goes --
diffusion, distogram, an interface term, something else -- is a modelling
decision belonging to whatever is being trained, and several such projects are
expected to share this package. Two consequences for the design:

* **Unresolved atoms are masked, never imputed.** Filling them with a
  placeholder would be a choice with a silent effect on any loss that averages
  over atoms. The mask is reported and the coordinates left at NaN, so a
  consumer that forgets to apply it gets NaN rather than a plausible number.
* **Nothing is centred, aligned or normalised.** Those belong to the objective.

Alignment is by name: ESMFold2's atom ordering is decoded from its own
featurization (see :mod:`esmfold2_foundry.data.bonds` for why it is read back
rather than re-derived) and matched against the source by
``(chain, residue index, atom name)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from biotite.structure import AtomArray

__all__ = ["StructureLabels", "structure_labels"]


@dataclass
class StructureLabels:
    """Deposited coordinates on the model's atom axis.

    Attributes:
        atom_coords: ``(A, 3)`` float32, NaN where the source has no atom.
        atom_mask: ``(A,)`` bool -- True where a source atom was found.
        matched: how many model atoms were matched.
        unmatched_examples: a few ``(chain, residue index, atom name)`` triples
            the source did not provide, for diagnosis.
    """

    atom_coords: np.ndarray
    atom_mask: np.ndarray
    matched: int
    unmatched_examples: list[tuple[str, int, str]]

    @property
    def coverage(self) -> float:
        """Fraction of the model's atoms the source supplied."""
        total = int(self.atom_mask.size)
        return float(self.matched) / total if total else 0.0

    def as_dict(self) -> dict[str, np.ndarray]:
        """The plain arrays, for putting under ``example["labels"]``."""
        return {"atom_coords": self.atom_coords, "atom_mask": self.atom_mask}


def _model_atom_identities(
    features: dict[str, Any], chain_infos: list[Any]
) -> list[tuple[str, int, str] | None]:
    """Per model atom, its ``(chain_id, residue index, atom name)``.

    ``None`` for padding. Decoded from the featurizer's own output so that the
    ordering is the model's own rather than this package's guess at it.
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

    identities: list[tuple[str, int, str] | None] = []
    for index in range(len(atom_to_token)):
        if not atom_mask[index]:
            identities.append(None)
            continue
        location = token_location.get(int(atom_to_token[index]))
        if location is None:
            identities.append(None)
            continue
        name = "".join(
            chr(int(code) + 32) for code in name_chars[index] if int(code) != 0
        ).strip()
        identities.append((location[0], location[1], name))
    return identities


def structure_labels(
    atoms: AtomArray,
    features: dict[str, Any],
    chain_infos: list[Any],
    residue_index_of: dict[tuple[str, int], int],
    *,
    chain_key: str = "chain_id",
    max_unmatched_examples: int = 8,
) -> StructureLabels:
    """Permute *atoms*' coordinates onto the model's atom axis.

    Args:
        residue_index_of: ``(chain_id, source res_id) -> model residue index``,
            as the adapter builds it. The deposited numbering need not start at
            one or be contiguous, so the two are related only through the
            residue list the sequence came from.

    Returns:
        :class:`StructureLabels`. Atoms the source does not provide -- an
        unresolved side chain, a hydrogen the model does not model, a residue
        present in the sequence but not in the density -- are left NaN and
        masked out.
    """
    identities = _model_atom_identities(features, chain_infos)

    chain = np.asarray(atoms.get_annotation(chain_key)).astype(str)
    res_id = np.asarray(atoms.res_id).astype(int)
    atom_name = np.asarray(atoms.atom_name).astype(str)
    coords = np.asarray(atoms.coord, dtype=np.float32)

    # (chain, model residue index, atom name) -> source row. Built once; a
    # per-atom scan would be quadratic on anything real.
    source: dict[tuple[str, int, str], int] = {}
    for index in range(len(atoms)):
        model_residue = residue_index_of.get((chain[index], int(res_id[index])))
        if model_residue is None:
            continue
        source.setdefault((chain[index], model_residue, atom_name[index]), index)

    n_atoms = len(identities)
    out = np.full((n_atoms, 3), np.nan, dtype=np.float32)
    mask = np.zeros(n_atoms, dtype=bool)
    unmatched: list[tuple[str, int, str]] = []

    for position, identity in enumerate(identities):
        if identity is None:
            continue
        row = source.get(identity)
        if row is None:
            if len(unmatched) < max_unmatched_examples:
                unmatched.append(identity)
            continue
        coordinate = coords[row]
        # A source atom whose coordinate is NaN is not a measurement; treat it
        # as absent rather than propagating it into a label that looks present.
        if not np.isfinite(coordinate).all():
            if len(unmatched) < max_unmatched_examples:
                unmatched.append(identity)
            continue
        out[position] = coordinate
        mask[position] = True

    return StructureLabels(
        atom_coords=out,
        atom_mask=mask,
        matched=int(mask.sum()),
        unmatched_examples=unmatched,
    )
