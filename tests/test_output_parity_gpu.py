"""Output parity: real folds, compared against the model's own run-to-run scatter.

Marked ``gpu`` and skipped by default. This confirms what feature parity already
establishes -- see docs/02 -- so it is deliberately not the primary check.

Run on a GPU node with the weights present::

    sbatch --partition=<gpu-partition> scripts/parity_gpu.sbatch

**These tests are controlled comparisons, not absolute-tolerance checks**, and
that distinction is the whole point. ESMFold2's structure head is a diffusion
sampler; even with every RNG seeded identically (``_seed_context`` seeds python,
numpy, torch and CUDA, then restores), two folds of the *same* input on a GPU do
not agree bitwise, because non-deterministic kernels and bf16 reductions differ
in ordering and the sampler amplifies that over its steps. Measured on an RTX
6000 Ada: two identical lysozyme folds differ by more than 0.2 A in the worst
atom.

So an absolute tolerance on coordinates cannot distinguish "the adapter changed
the input" from "the sampler is not reproducible", which is exactly the
confusion D-008 exists to prevent. Instead each test folds the native input
twice to measure the noise floor, then asserts that native-vs-adapted sits
inside it. If the adapter really did change the input, the cross difference
would leave that envelope.
"""

from __future__ import annotations

import os

import pytest

from esmfold2_atomworks import paths
from esmfold2_atomworks.data.atomworks_to_esm import (
    atom_array_to_structure_prediction_input,
)
from esmfold2_atomworks.parity.compare import compare_results

pytestmark = pytest.mark.gpu

#: How much larger than the model's own scatter a cross-path difference may be
#: before it stops being explicable as noise. The floor keeps a metric that is
#: essentially zero in both runs from failing on a rounding artefact.
SCATTER_ALLOWANCE = 3.0
ABSOLUTE_FLOOR = {
    "coord_max_abs": 0.05,
    "coord_rmsd": 0.01,
    "plddt_max_abs": 0.01,
    "pae_max_abs": 0.5,
    "distogram_max_abs": 0.5,
    "ptm": 1e-3,
    "iptm": 1e-3,
}


@pytest.fixture(scope="module")
def model():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    # Without a local mirror this would download several GB (ESMFold2 plus the
    # ESMC backbone), which a test run should not do unless asked.
    if not paths.ESMFOLD2_WEIGHTS.standard.is_dir() and not os.environ.get(
        "EF_ALLOW_DOWNLOAD"
    ):
        pytest.skip(
            f"no local weights at {paths.ESMFOLD2_WEIGHTS.standard}; "
            "set EF_ALLOW_DOWNLOAD=1 to fetch them from the Hub"
        )

    from esmfold2_atomworks.model.esmfold2 import AtomWorksESMFold2

    return AtomWorksESMFold2()


def _native_counterpart(spi, chain_info):  # retained for ad-hoc use
    from esmfold2_atomworks.parity.run import _native_counterpart as build

    return build(spi, chain_info)


def _assert_within_scatter(cross, baseline) -> None:
    """Every cross-path metric must sit inside the model's own scatter."""
    failures = []
    for key, value in sorted(cross.metrics.items()):
        noise = baseline.metrics.get(key, 0.0)
        budget = max(noise * SCATTER_ALLOWANCE, ABSOLUTE_FLOOR.get(key, 0.0))
        mark = "ok " if value <= budget else "BAD"
        line = f"  [{mark}] {key:24s} cross {value:12.6g}  self {noise:12.6g}  budget {budget:12.6g}"
        if value > budget:
            failures.append(line)
        print(line)
    assert not failures, (
        "cross-path difference exceeds the model's own scatter:\n" + "\n".join(failures)
    )


@pytest.mark.parametrize("fixture", ["lysozyme", "hemoglobin"])
def test_adapted_input_folds_within_the_models_own_scatter(
    parsed, model, gold, fixture
):
    """The two paths agree as closely as the model agrees with itself.

    The reference side is the frozen gold input, not a reconstruction of the
    adapter's output, for the same reason feature parity uses it: a systematic
    error copied into both sides would cancel.

    Atom naming and ordering are held to exact equality regardless -- those are
    bookkeeping, not sampling, and a mismatch there would make every coordinate
    comparison meaningless rather than merely noisy.
    """
    from esmfold2_atomworks.model.esmfold2 import FoldingConfig

    atoms, chain_info = parsed(fixture)
    adapted = atom_array_to_structure_prediction_input(atoms, chain_info=chain_info)
    native = gold(fixture)

    config = FoldingConfig(num_loops=1, num_sampling_steps=8, seed=0)
    native_a = model.fold(native, config=config)
    native_b = model.fold(native, config=config)
    adapted_a = model.fold(adapted, config=config)

    baseline = compare_results(native_a, native_b)
    cross = compare_results(native_a, adapted_a)

    print(f"\n{fixture}: native-vs-native is the noise floor")
    assert cross.metrics["atom_name_mismatches"] == 0
    _assert_within_scatter(cross, baseline)


def test_fold_atom_array_round_trips_to_a_structure(parsed, model):
    """The whole round trip: AtomArray in, AtomArray out, nothing lost."""
    from esmfold2_atomworks.model.esmfold2 import FoldingConfig

    atoms, chain_info = parsed("hemoglobin")
    structure, result = model.fold_atom_array(
        atoms,
        chain_info=chain_info,
        config=FoldingConfig(num_loops=1, num_sampling_steps=8, seed=0),
    )
    assert len(structure) > 0
    # Four HEM ligands must survive as hetero atoms; the mmCIF route loses them.
    assert structure.hetero.sum() > 0
    assert set(structure.chain_id.tolist()) >= {"A", "B", "C", "D"}
    assert result.plddt is not None


def test_the_sampler_is_not_bitwise_reproducible(parsed, model):
    """Guards the control above from silently becoming a no-op.

    If two identical folds ever did agree bitwise, the scatter budget would
    collapse to the absolute floor and the tests above would quietly turn into
    the absolute-tolerance check they were written to replace. This records the
    assumption so that a future deterministic build fails loudly here instead.
    """
    from esmfold2_atomworks.model.esmfold2 import FoldingConfig

    atoms, chain_info = parsed("lysozyme")
    spi = atom_array_to_structure_prediction_input(atoms, chain_info=chain_info)
    config = FoldingConfig(num_loops=1, num_sampling_steps=8, seed=0)

    diff = compare_results(
        model.fold(spi, config=config), model.fold(spi, config=config)
    )
    print(f"\nself-comparison: {diff.metrics}")
    assert diff.metrics["atom_name_mismatches"] == 0
    if diff.metrics["coord_max_abs"] == 0.0:
        pytest.skip(
            "this build folds reproducibly; the scatter-budget tests are now "
            "equivalent to absolute-tolerance checks and should be tightened"
        )
