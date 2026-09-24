"""MSAs must reach the right chain, and pair across chains by taxonomy.

MSA support is part of ESMFold2's native input surface, so the port is not
complete without it. It is also where a mistake is quietest: an alignment
attached to the wrong chain, or cross-chain pairing that silently does not
happen, changes the prediction while every tensor keeps its shape.

Pairing is driven **only** by a ``key=<taxid>`` token in the FASTA header
(upstream matches ``key=(-?\\d+)``). A pipeline that builds MSAs without those
keys gets no pairing at all and nothing in the feature shapes reveals it, which
is why the heteromer tests below assert on row *content*.

The alignments are synthetic and built from the frozen gold sequences, so they
stay consistent with the fixtures without a large a3m in the tree.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from esmfold2_foundry.data.atomworks_to_esm import (
    atom_array_to_structure_prediction_input,
)
from esmfold2_foundry.parity.compare import compare_features, featurize

GAP = 1  # MSA_GAP_TOKEN_ID
#: Positions mutated in the synthetic hits, so a row can be identified by the
#: column at which it departs from the query.
POS_SHARED = 5
POS_UNSHARED = 9


def _msa(query: str, variants: list[tuple[str, int]]):
    """An ``MSA`` over *query* plus ``(sequence, taxid)`` rows."""
    from esm.utils.msa import MSA

    lines = [">query", query]
    for index, (sequence, taxid) in enumerate(variants):
        lines += [f">hit{index} key={taxid}", sequence]
    return MSA.from_a3m(io.StringIO("\n".join(lines) + "\n"))


def _mutate(sequence: str, position: int, residue: str = "W") -> str:
    chars = list(sequence)
    chars[position] = residue
    return "".join(chars)


@pytest.fixture
def lysozyme_msa(gold_document):
    sequence = gold_document("lysozyme")["chains"][0]["sequence"]
    return sequence, _msa(
        sequence,
        [
            (_mutate(sequence, POS_SHARED), 1001),
            (_mutate(sequence, POS_UNSHARED), 2002),
        ],
    )


def test_an_msa_reaches_the_model(parsed, ccd, lysozyme_msa):
    """Without one the model runs in single-sequence mode; with one it must not."""
    _sequence, msa = lysozyme_msa
    atoms, chain_info = parsed("lysozyme")

    without = atom_array_to_structure_prediction_input(atoms, chain_info=chain_info)
    with_msa = atom_array_to_structure_prediction_input(
        atoms, chain_info=chain_info, msas={"A": msa}
    )
    assert without.sequences[0].msa is None
    assert with_msa.sequences[0].msa is not None

    plain, aligned = featurize(without), featurize(with_msa)
    assert plain["msa"].shape[0] == 1, "single-sequence mode should synthesise one row"
    assert aligned["msa"].shape[0] == 3
    assert aligned["msa"].shape[1] == plain["msa"].shape[1]

    diff = compare_features(plain, aligned)
    changed = set(diff.value_mismatch) | set(diff.shape_mismatch)
    assert {"msa", "msa_attention_mask"} <= changed, changed


def test_the_msa_features_match_a_hand_written_input(parsed, ccd, lysozyme_msa):
    """The adapter must not alter the alignment on its way through."""
    from esm.models.esmfold2.types import ProteinInput, StructurePredictionInput

    sequence, msa = lysozyme_msa
    atoms, chain_info = parsed("lysozyme")

    adapted = atom_array_to_structure_prediction_input(
        atoms, chain_info=chain_info, msas={"A": msa}
    )
    native = StructurePredictionInput(
        sequences=[ProteinInput(id="A", sequence=sequence, msa=msa)]
    )
    diff = compare_features(featurize(native), featurize(adapted))
    assert diff.identical, diff.report()


def _heteromer_msas(gold_document):
    """Chain A: unshared hit FIRST, shared hit SECOND. Chain B: the shared one.

    The ordering is the point. If pairing followed row order, the paired row
    would carry A's *first* hit; if it follows the taxid, it carries the second.
    """
    document = gold_document("hemoglobin")
    alpha = document["chains"][0]["sequence"]
    beta = document["chains"][1]["sequence"]
    msa_a = _msa(
        alpha,
        [
            (_mutate(alpha, POS_UNSHARED), 2002),
            (_mutate(alpha, POS_SHARED), 1001),
        ],
    )
    msa_b = _msa(beta, [(_mutate(beta, POS_SHARED), 1001)])
    return msa_a, msa_b


def _rows(features):
    msa = np.asarray(features["msa"])
    asym = np.asarray(features["asym_id"])
    return msa, asym == 0, asym == 1


def test_shared_taxids_pair_across_chains(parsed, ccd, gold_document):
    """A shared ``key=`` must put both chains' hits in one row.

    Asserting only that some row covers both chains would not show pairing:
    two *unpaired* hits from different chains also share a row index, laid out
    block-diagonally. So the paired row is identified by content -- it must
    carry the hit whose taxid matched, which is A's second hit.
    """
    msa_a, msa_b = _heteromer_msas(gold_document)
    atoms, chain_info = parsed("hemoglobin")

    features = featurize(
        atom_array_to_structure_prediction_input(
            atoms, chain_info=chain_info, msas={"A": msa_a, "B": msa_b}
        )
    )
    msa, cols_a, cols_b = _rows(features)

    covers_both = [
        row
        for row in range(1, msa.shape[0])  # row 0 is the query
        if (msa[row, cols_a] != GAP).any() and (msa[row, cols_b] != GAP).any()
    ]
    assert len(covers_both) == 1, f"expected exactly one paired row, got {covers_both}"

    paired = covers_both[0]
    differs_at = np.where(msa[paired, cols_a] != msa[0, cols_a])[0].tolist()
    assert differs_at == [POS_SHARED], (
        f"the paired row carries A's hit mutated at {differs_at}; pairing by taxid "
        f"requires the shared hit (position {POS_SHARED}), not whichever came first"
    )


def test_an_unpairable_hit_stays_unpaired(parsed, ccd, gold_document):
    """A's unshared hit must occupy a row that touches A only."""
    msa_a, msa_b = _heteromer_msas(gold_document)
    atoms, chain_info = parsed("hemoglobin")

    features = featurize(
        atom_array_to_structure_prediction_input(
            atoms, chain_info=chain_info, msas={"A": msa_a, "B": msa_b}
        )
    )
    msa, cols_a, cols_b = _rows(features)

    a_only = [
        row
        for row in range(1, msa.shape[0])
        if (msa[row, cols_a] != GAP).any() and not (msa[row, cols_b] != GAP).any()
    ]
    assert len(a_only) == 1, f"expected one A-only row, got {a_only}"
    differs_at = np.where(msa[a_only[0], cols_a] != msa[0, cols_a])[0].tolist()
    assert differs_at == [POS_UNSHARED]


def test_alignments_are_looked_up_by_chain_id(parsed, ccd, gold_document):
    """Guards the wiring: the chains must not be interchangeable.

    If the adapter routed alignments by position rather than by chain id, or
    dropped them, the parity test above would still pass. Chains A and C carry
    the same sequence, so each of two different alignments binds to either:
    swapping them must change the input, and the order they are declared in
    must not. (Swapping alpha's and beta's alignments is refused outright --
    see tests/test_declarations.py.)
    """
    alpha = gold_document("hemoglobin")["chains"][0]["sequence"]
    first = _msa(alpha, [(_mutate(alpha, POS_SHARED), 1001)])
    second = _msa(alpha, [(_mutate(alpha, POS_UNSHARED), 2002)])
    atoms, chain_info = parsed("hemoglobin")

    def features(msas):
        return featurize(
            atom_array_to_structure_prediction_input(
                atoms, chain_info=chain_info, msas=msas
            )
        )

    declared = features({"A": first, "C": second})
    swapped = compare_features(declared, features({"A": second, "C": first}))
    assert not swapped.ok, "swapping the two chains' alignments changed nothing"
    reordered = compare_features(declared, features({"C": second, "A": first}))
    assert reordered.identical, reordered.report()
