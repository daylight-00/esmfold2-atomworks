"""The LM prior of a fold: always present, and caller-supplied only when checked.

The native ``forward`` skips its language-model pathway when no ESMC backbone
is attached and no hidden states are given, and folds anyway. Supplied hidden
states are the one input that legitimately takes the backbone's place, so they
are checked against the live module before they reach it, and ``lm_mask_pct``,
which acts only inside the backbone, is refused beside them.

The refusals need nothing installed and run in the offline job; a fold that
goes through needs torch, the checks esm too, and the shim test a local
checkpoint mirror.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from esmfold2_atomworks import paths
from esmfold2_atomworks.model.esmfold2 import (
    REPLICATED_FOLD_PARAMETERS,
    AtomWorksESMFold2,
    MissingLanguageModelError,
)

#: An input with no entities: enough for a record's digests.
_SPI = SimpleNamespace(sequences=[])


class _Builder:
    """Records whether a fold reached upstream."""

    def __init__(self) -> None:
        self.calls = 0

    def fold(self, net, spi, **kwargs):
        self.calls += 1
        return "result"


def _wrapper(esmc=None) -> AtomWorksESMFold2:
    model = AtomWorksESMFold2.__new__(AtomWorksESMFold2)
    config = SimpleNamespace(type="release", num_loops=3, lm_mask_pct=0.0)
    model.net = SimpleNamespace(esmc=esmc, config=config)
    model.builder = _Builder()
    return model


# -- offline -----------------------------------------------------------------


@pytest.mark.offline
def test_a_fold_without_backbone_or_states_is_refused():
    model = _wrapper(esmc=None)
    with pytest.raises(MissingLanguageModelError, match="without the LM prior"):
        model.fold(object())
    assert model.builder.calls == 0


def test_the_guard_reads_the_resident_backbone_not_the_load_flag():
    # A bundled checkpoint loaded with load_esmc=False still has its backbone.
    model = _wrapper(esmc=object())
    record: dict = {}
    assert model.fold(_SPI, record=record) == "result"
    assert record["esmfold2.lm_source"] == "model"


@pytest.mark.offline
def test_lm_mask_pct_beside_supplied_states_is_refused():
    model = _wrapper(esmc=None)
    with pytest.raises(ValueError, match="lm_mask_pct"):
        model.fold(object(), lm_hidden_states=object(), lm_mask_pct=0.1)
    assert model.builder.calls == 0


@pytest.mark.offline
def test_a_record_holding_an_earlier_calls_entries_is_refused():
    model = _wrapper(esmc=object())
    record = {"caller.run": "r1", "esmfold2.lm_source": "model"}
    with pytest.raises(ValueError, match="fresh dict"):
        model.fold(_SPI, record=record)
    assert model.builder.calls == 0


def test_a_record_keeps_the_callers_own_keys():
    model = _wrapper(esmc=object())
    record = {"caller.run": "r1"}
    model.fold(_SPI, record=record)
    assert record["caller.run"] == "r1"
    assert record["esmfold2.lm_source"] == "model"


def test_a_record_observes_the_determinism_settings(monkeypatch):
    """Observed, not certified: the flag and the variable as they stand."""
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    record: dict = {}
    _wrapper(esmc=object()).fold(_SPI, record=record)
    assert record["esmfold2.deterministic_algorithms"] is (
        torch.are_deterministic_algorithms_enabled()
    )
    assert record["esmfold2.cublas_workspace_config"] == ":4096:8"

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    record = {}
    _wrapper(esmc=object()).fold(_SPI, record=record)
    assert record["esmfold2.cublas_workspace_config"] is None


@pytest.mark.offline
@pytest.mark.parametrize("name", ["model", "input"])
@pytest.mark.parametrize("states", [None, object()], ids=["model-lm", "supplied-lm"])
def test_the_model_and_input_are_not_overrides(name, states):
    """Both paths refuse them alike, rather than one failing and one ignoring."""
    model = _wrapper(esmc=object())
    with pytest.raises(TypeError, match="not settings to override"):
        model.fold(object(), lm_hidden_states=states, **{name: object()})
    assert model.builder.calls == 0


@pytest.mark.offline
def test_computing_states_reads_the_mask_fraction_rather_than_defaulting_it():
    model = _wrapper(esmc=object())
    model.net._compute_lm_hidden_states = lambda *args, **kwargs: None
    del model.net.config.lm_mask_pct
    with pytest.raises(AttributeError, match="lm_mask_pct"):
        model.compute_lm_hidden_states({})


@pytest.mark.offline
def test_computing_states_without_a_backbone_is_refused():
    with pytest.raises(MissingLanguageModelError, match="load_esmc=True"):
        _wrapper(esmc=None).compute_lm_hidden_states({})


# -- against the installed upstream ------------------------------------------


def test_the_replicated_fold_matches_upstreams_signature():
    """The fold that carries supplied states replicates upstream's body.

    A parameter upstream adds would be dropped by the replica, so the set is
    pinned: this fails, and the fold itself raises, until the replica is
    brought up to date. When upstream takes ``lm_hidden_states`` itself, this
    is where that shows first.
    """
    processor = pytest.importorskip("esm.models.esmfold2.processor")
    parameters = set(inspect.signature(processor.ESMFold2InputBuilder.fold).parameters)
    assert parameters - {"self"} == REPLICATED_FOLD_PARAMETERS


def _shim_model(d_z: int = 8, d_model: int = 16, num_layers: int = 3):
    torch = pytest.importorskip("torch")
    layers = pytest.importorskip("esm.models.esmfold2.layers")
    model = AtomWorksESMFold2.__new__(AtomWorksESMFold2)
    model.net = SimpleNamespace(
        language_model=layers.LanguageModelShim(
            d_z=d_z, d_model=d_model, num_layers=num_layers
        )
    )
    features = {"res_type": torch.zeros(1, 5, dtype=torch.long)}
    return torch, model, features


def test_states_of_the_right_shape_pass():
    torch, model, features = _shim_model()
    model._check_lm_hidden_states(torch.zeros(1, 5, 4, 16), features)


@pytest.mark.parametrize(
    "shape", [(1, 6, 4, 16), (2, 5, 4, 16), (1, 5, 3, 16), (1, 5, 4, 15), (5, 4, 16)]
)
def test_states_of_another_shape_are_refused(shape):
    torch, model, features = _shim_model()
    with pytest.raises(ValueError, match=r"\(1, 5, 4, 16\)"):
        model._check_lm_hidden_states(torch.zeros(*shape), features)


def test_states_on_another_device_are_refused():
    torch, model, features = _shim_model()
    with pytest.raises(ValueError, match="meta"):
        model._check_lm_hidden_states(torch.zeros(1, 5, 4, 16, device="meta"), features)


def test_integer_states_are_refused():
    torch, model, features = _shim_model()
    with pytest.raises(TypeError, match="floating point"):
        model._check_lm_hidden_states(
            torch.zeros(1, 5, 4, 16, dtype=torch.long), features
        )


def test_a_module_without_an_lm_pathway_is_refused():
    torch, model, features = _shim_model()
    model.net = SimpleNamespace()
    with pytest.raises(AttributeError, match="language_model"):
        model._check_lm_hidden_states(torch.zeros(1, 5, 4, 16), features)


def _published_shim():
    """The LM shim of a local checkpoint mirror, with its trained weights."""
    torch = pytest.importorskip("torch")
    layers = pytest.importorskip("esm.models.esmfold2.layers")
    safetensors = pytest.importorskip("safetensors")
    for flavour in ("experimental", "standard"):
        directory = getattr(paths.ESMFOLD2_WEIGHTS, flavour)
        state = {}
        for shard in (
            sorted(directory.glob("*.safetensors")) if directory.is_dir() else []
        ):
            with safetensors.safe_open(str(shard), "pt") as handle:
                for key in handle.keys():  # noqa: SIM118 -- a safetensors handle, not a dict
                    if key.startswith("language_model."):
                        state[key.removeprefix("language_model.")] = handle.get_tensor(
                            key
                        )
        if state:
            d_model = state["base_z_linear.0.weight"].shape[0]
            d_z = state["base_z_linear.1.weight"].shape[0]
            num_layers = state["base_z_combine"].shape[0] - 1
            shim = layers.LanguageModelShim(
                d_z=d_z, d_model=d_model, num_layers=num_layers
            )
            shim.load_state_dict(state)
            return torch, shim
    pytest.skip("no local checkpoint mirror to read the LM shim from")


def test_the_published_shim_maps_zero_states_to_a_nonzero_pair_term():
    """Why the guard is unconditional, protein or not.

    Upstream fills every non-protein token with zero hidden states, so a
    protein-free input with a backbone resident hands the shim all zeros. The
    trained shim does not map them to zero, so leaving the pathway out -- what
    ``forward`` does without a backbone -- is a different fold even then.
    """
    torch, shim = _published_shim()
    d_model = shim.base_z_linear[0].normalized_shape[-1]
    layers = shim.base_z_combine.numel()
    with torch.no_grad():
        lm_z = shim(torch.zeros(1, 4, layers, d_model))
    assert lm_z.abs().max().item() > 0
