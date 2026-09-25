"""AtomWorks' loaded alignments, as the MSAs ESMFold2 takes.

``atomworks.ml.transforms.msa.LoadPolymerMSAs`` attaches, per polymer chain,
``data["polymer_msas_by_chain_id"][chain] = {"msa", "ins", "tax_ids", ...}``:
an integer-encoded alignment, the count of insertions to the left of each
column, and a taxonomy id per row. ESMFold2 takes an ``esm.utils.msa.MSA`` per
protein chain, derives ``has_deletion`` / ``deletion_value`` from per-column
deletion counts, and pairs chains only through a ``key=<taxid>`` token in a
row's header.

This module is that translation and nothing more: rows keep their order, the
insertion counts become the deletion counts unchanged, and a numeric taxonomy
id becomes the pairing key. Whether the result binds to the chain -- query row
equal to the folded sequence -- is checked where every MSA is checked, in the
adapter. The transform that applies it to a pipeline example is
:class:`esmfold2_atomworks.data.pipelines.PolymerMSAsToESMFold2`.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

__all__ = ["polymer_msa_to_esm"]

#: Upstream reads a pairing key only in this form (``key=(-?\\d+)``).
_PAIRING_KEY = re.compile(r"-?\d+")


def _protein_letters() -> np.ndarray:
    """AtomWorks' MSA integer -> the protein one-letter code, as bytes."""
    from atomworks.ml.transforms.msa._msa_constants import (
        AMINO_ACID_ONE_LETTER_TO_INT,
    )

    letters = np.full(max(AMINO_ACID_ONE_LETTER_TO_INT.values()) + 1, b"", dtype="S1")
    for letter, code in AMINO_ACID_ONE_LETTER_TO_INT.items():
        letters[code] = letter.encode()
    return letters


def polymer_msa_to_esm(msa_data: dict[str, Any], *, chain_id: str = "") -> Any:
    """One chain's loaded alignment as an ``esm.utils.msa.MSA``.

    Args:
        msa_data: one value of ``polymer_msas_by_chain_id``, as
            ``LoadPolymerMSAs`` leaves it -- before any cropping, pairing or
            padding transform has touched it.
        chain_id: for error messages only.

    Raises:
        ValueError: for an alignment that is not a protein alignment as loaded:
            codes outside the protein alphabet, padded positions, or arrays
            whose shapes disagree.
    """
    from esm.utils.msa import MSA
    from esm.utils.parsing import FastaEntry

    msa = np.asarray(msa_data["msa"])
    ins = np.asarray(msa_data["ins"])
    tax_ids = [str(t) for t in msa_data.get("tax_ids", [""] * len(msa))]
    where = f"the loaded alignment for chain {chain_id!r}"
    if msa.ndim != 2 or ins.shape != msa.shape or len(tax_ids) != msa.shape[0]:
        raise ValueError(
            f"{where} has inconsistent shapes: msa {msa.shape}, ins {ins.shape}, "
            f"{len(tax_ids)} taxonomy ids"
        )
    padded = msa_data.get("msa_is_padded_mask")
    if padded is not None and np.asarray(padded).any():
        raise ValueError(
            f"{where} carries padded positions; convert it straight after "
            "LoadPolymerMSAs, before a transform that pads or merges alignments"
        )
    letters = _protein_letters()
    in_range = msa.size == 0 or (msa.min() >= 0 and msa.max() < len(letters))
    if not in_range or not letters[msa].all():
        raise ValueError(f"{where} holds codes outside the protein alphabet")

    entries = []
    for index, row in enumerate(letters[msa]):
        header = "query" if index == 0 else f"hit{index}"
        taxid = tax_ids[index].strip()
        if index and _PAIRING_KEY.fullmatch(taxid):
            header += f" key={taxid}"
        entries.append(FastaEntry(header, row.tobytes().decode()))
    return MSA(entries, deletions=ins.astype(np.float32))
