"""The milestone: ``F(AtomArray)`` featurizes identically to an independent input.

These run on CPU, need no weights, and are exact.

**What feature parity does and does not claim.** ``prepare_esmfold2_input`` is a
pure function of the ``StructurePredictionInput``, so identical inputs give
identical feature tensors -- that part is by construction. ``forward`` is *not*
pure: the structure head is a diffusion sampler that consumes RNG, so identical
features do not give identical coordinates. What follows is that the adapter
presents the model with the same conditioning, and therefore the same sampling
distribution. That is the strong claim, and it is the one worth making;
``tests/test_output_parity_gpu.py`` then checks that the realised samples differ
no more than the sampler differs from itself.

**The comparison is against a frozen fixture, not a reconstruction.** An earlier
version rebuilt the "native" input from the adapter's own output, which meant a
systematic error -- a ligand mapped to the wrong CCD code, say -- would be
copied into both sides and cancel. ``tests/data/gold/*.json`` is generated from
AtomWorks' own parse output, checked in, and anchored to facts about the PDB
entries by ``test_gold_fixtures.py``.
"""

from __future__ import annotations

import pytest

from esmfold2_foundry.data.atomworks_to_esm import (
    atom_array_to_structure_prediction_input,
)
from esmfold2_foundry.parity.compare import (
    MODEL_CONSUMED_FEATURES,
    compare_features,
    featurize,
)

FIXTURES = ["lysozyme", "hemoglobin", "modified", "zinc", "flavoprotein"]


@pytest.mark.parametrize("fixture", FIXTURES)
def test_adapter_featurizes_identically_to_the_gold_input(parsed, ccd, gold, fixture):
    """Every tensor matches, for monomer, multimer, metal, cofactor and NCAA cases."""
    atoms, chain_info = parsed(fixture)
    adapted = atom_array_to_structure_prediction_input(atoms, chain_info=chain_info)

    diff = compare_features(featurize(gold(fixture)), featurize(adapted))
    assert diff.ok, diff.report()
    assert diff.identical, diff.report()
    assert MODEL_CONSUMED_FEATURES <= set(diff.compared)


def test_declaring_a_modification_actually_changes_the_features(parsed, ccd):
    """Guards the parity assertions above from being vacuous.

    If declaring ``MSE`` made no difference, the modified-residue parity case
    would pass whatever the adapter did with modifications. It must change
    tokenization: a modified residue becomes one token per atom.
    """
    atoms, chain_info = parsed("modified")
    with_mods = atom_array_to_structure_prediction_input(
        atoms, chain_info=chain_info, emit_modifications=True
    )
    without = atom_array_to_structure_prediction_input(
        atoms, chain_info=chain_info, emit_modifications=False
    )
    n_with = featurize(with_mods)["res_type"].shape[0]
    n_without = featurize(without)["res_type"].shape[0]
    assert n_with > n_without, (
        f"declaring 4 MSE did not change the token count ({n_with} vs {n_without}); "
        "the parity test above would then be vacuous"
    )


def test_a_wrong_ligand_would_be_caught(parsed, ccd, gold):
    """Guards the parity assertions from passing on a mis-identified ligand.

    Substituting one real CCD code for another is exactly the failure D-004
    exists to prevent, and it must show up as a feature difference rather than
    being absorbed. Without this, "all 29 tensors identical" would be
    reassuring without being informative.
    """
    from esm.models.esmfold2.types import (
        LigandInput,
        ProteinInput,
        StructurePredictionInput,
    )

    correct = gold("hemoglobin")
    tampered = StructurePredictionInput(
        sequences=[
            entry
            if isinstance(entry, ProteinInput)
            else LigandInput(id=entry.id, ccd=["HEC"])  # haem C, not haem B
            for entry in correct.sequences
        ]
    )
    diff = compare_features(featurize(correct), featurize(tampered))
    assert not diff.ok, "swapping HEM for HEC produced identical features"


def test_chain_order_is_stable(parsed, ccd):
    """Entity numbering follows input order, so the order must be reproducible."""
    atoms, chain_info = parsed("hemoglobin")
    first = atom_array_to_structure_prediction_input(atoms, chain_info=chain_info)
    second = atom_array_to_structure_prediction_input(atoms, chain_info=chain_info)
    assert [s.id for s in first.sequences] == [s.id for s in second.sequences]
    diff = compare_features(featurize(first), featurize(second))
    assert diff.identical, diff.report()
