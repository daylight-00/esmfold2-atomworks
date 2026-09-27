"""Output parity: real folds of the two paths, compared on a GPU.

Marked ``gpu`` and skipped by default. This confirms what feature parity already
establishes -- see docs/02 -- so it is deliberately not the primary check.

Run on a GPU node with the weights present::

    sbatch --partition=<gpu-partition> scripts/parity_gpu.sbatch

**The comparison runs under deterministic kernels.** ESMFold2's structure head
is a diffusion sampler, and with every RNG seeded identically two folds of the
same input still differ on a GPU when the kernels are allowed to be
non-deterministic -- by tenths of an Angstrom here, by far more on some inputs.
An absolute tolerance cannot tell that apart from an adapter that changed the
input, and a budget scaled to it is only as tight as the scatter happens to be.
With ``torch.use_deterministic_algorithms(True)`` and a fixed cuBLAS workspace
the scatter goes away: the same input folds identically twice, and the two
paths, which hand the model identical tensors, must then fold identically too.

The scatter itself is characterized separately, as a measurement rather than a
gate: it describes the execution configuration, not the adapter.
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


@pytest.mark.parametrize(
    "fixture", ["lysozyme", "hemoglobin", "modified", "zinc", "flavoprotein"]
)
def test_adapted_input_folds_identically(parsed, model, gold, fixture, deterministic):
    """The two paths fold to the same structure, exactly.

    The reference side is the frozen gold input, not a reconstruction of the
    adapter's output, for the same reason feature parity uses it: a systematic
    error copied into both sides would cancel. The native input is folded twice
    first, so that a difference is never read as the adapter's while the
    execution itself is not repeatable.
    """
    from esmfold2_atomworks.model.esmfold2 import FoldingConfig

    atoms, chain_info = parsed(fixture)
    adapted = atom_array_to_structure_prediction_input(atoms, chain_info=chain_info)
    native = gold(fixture)

    config = FoldingConfig(num_loops=1, num_sampling_steps=8, seed=0)
    native_a = model.fold(native, config=config)
    native_b = model.fold(native, config=config)
    adapted_a = model.fold(adapted, config=config)

    repeat = compare_results(native_a, native_b)
    cross = compare_results(native_a, adapted_a)
    print(f"\n{fixture}: repeat {repeat.metrics}\n{fixture}: cross  {cross.metrics}")
    assert all(value == 0.0 for value in repeat.metrics.values()), (
        "the same input did not fold identically twice, so deterministic "
        f"execution is not in force and no cross-path claim can be made: {repeat.metrics}"
    )
    assert all(value == 0.0 for value in cross.metrics.values()), cross.metrics


def test_nondeterministic_scatter_is_characterized(model, gold):
    """What the same seed leaves to chance when kernels may be non-deterministic.

    A measurement, not a gate: the numbers depend on the device, the software
    stack and the input, and describe the execution configuration rather than
    the adapter. Only the bookkeeping is asserted -- atom names and order are
    not sampled, so they agree whatever the kernels do.
    """
    torch = pytest.importorskip("torch")
    from esmfold2_atomworks.model.esmfold2 import FoldingConfig

    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        spi = gold("lysozyme")
        config = FoldingConfig(num_loops=1, num_sampling_steps=8, seed=0)
        diff = compare_results(
            model.fold(spi, config=config), model.fold(spi, config=config)
        )
    finally:
        torch.use_deterministic_algorithms(previous)
    print(f"\nnon-deterministic repeat: {diff.metrics}")
    assert diff.metrics["atom_name_mismatches"] == 0


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


def test_the_loaded_model_answers_through_its_seams(model, gold):
    """The seams against the real module rather than a fake of it.

    A backbone is attached -- from the local mirror or bundled, never
    ``"none"`` under the default ``load_esmc=True`` -- and ``.esmc`` is that
    module; every width is real; and a fold that leaves ``num_loops`` unset
    reaches the model, which takes the count from its own config.
    """
    from esmfold2_atomworks.model.esmfold2 import FoldingConfig

    assert model.esmc is not None
    assert model.esmc is model.net.esmc
    assert model.provenance()["esmfold2.esmc"] != "none"
    assert all(width > 0 for width in model.representation_dims().values())

    result = model.fold(
        gold("lysozyme"), config=FoldingConfig(num_sampling_steps=8, seed=0)
    )
    assert result.plddt is not None


@pytest.fixture
def deterministic():
    """Deterministic kernels for one test, restored after it.

    cuBLAS reads its workspace setting when CUDA starts, so the variable has to
    be in the environment of the process, not set here; without it the test
    skips rather than claiming a determinism it cannot have.
    """
    torch = pytest.importorskip("torch")
    if not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        pytest.skip(
            "run with CUBLAS_WORKSPACE_CONFIG=:4096:8 for deterministic kernels"
        )
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def test_supplied_states_reproduce_the_models_own_fold(model, gold, deterministic):
    """Folding with the states the backbone computes is the same fold.

    The fold that carries supplied states replicates upstream's ``fold``; under
    deterministic kernels, handing it exactly what ``forward`` would compute
    must give back exactly what the ordinary path gives.
    """
    from esmfold2_atomworks.model.esmfold2 import FoldingConfig

    assert model.config.lm_mask_pct == 0.0, "a masked backbone pass is not replayable"
    spi = gold("lysozyme")
    config = FoldingConfig(num_loops=1, num_sampling_steps=8, seed=0)

    own_record: dict = {}
    own = model.fold(spi, config=config, record=own_record)
    features, _ = model.featurize(spi, seed=0)
    states = model.compute_lm_hidden_states(features)
    supplied_record: dict = {}
    supplied = model.fold(
        spi, config=config, lm_hidden_states=states, record=supplied_record
    )

    diff = compare_results(own, supplied)
    print(f"\nown vs supplied: {diff.metrics}")
    assert own_record["esmfold2.lm_source"] == "model"
    assert supplied_record["esmfold2.lm_source"] == "caller-supplied"
    assert own_record["esmfold2.deterministic_algorithms"] is True
    assert diff.metrics["atom_name_mismatches"] == 0
    assert diff.metrics["coord_max_abs"] == 0.0
    assert diff.metrics["plddt_max_abs"] == 0.0


def test_a_checkpoint_without_its_backbone_refuses_to_fold(model, gold):
    """The separate-layout checkpoint, loaded without its backbone.

    Before the guard this folded without the LM prior and returned a
    plausible structure. It must refuse, and fold once handed the states.
    """
    from esmfold2_atomworks.model.esmfold2 import (
        AtomWorksESMFold2,
        FoldingConfig,
        MissingLanguageModelError,
    )

    weights = paths.ESMFOLD2_WEIGHTS.experimental
    if not weights.is_dir():
        pytest.skip(f"no experimental checkpoint at {weights}")
    trunk = AtomWorksESMFold2(weights, load_esmc=False)
    try:
        assert trunk.esmc is None
        spi = gold("lysozyme")
        config = FoldingConfig(num_loops=1, num_sampling_steps=8, seed=0)
        with pytest.raises(MissingLanguageModelError):
            trunk.fold(spi, config=config)

        features, _ = model.featurize(spi, seed=0)
        states = model.compute_lm_hidden_states(features).to(trunk.device)
        result = trunk.fold(spi, config=config, lm_hidden_states=states)
        assert result.plddt is not None
    finally:
        del trunk
