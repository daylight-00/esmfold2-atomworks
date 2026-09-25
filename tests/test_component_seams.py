"""The named component seams and the dimension helper answer truthfully.

``.esmc``, ``.folding_trunk`` and ``.structure_head`` exist so that separating a
component later is an implementation change rather than a change to every call
site, and ``representation_dims()`` is where a caller is told to read widths.
Each can fail by returning a plausible value -- ``None`` read as "no backbone
attached", ``0`` read as a width -- so each is pinned here against the layout
the native module actually has.

The fakes need nothing installed and run in the offline job. The last two go
through ``EsmFold2Config``, which is where the field names move: once with a
pre-alignment config it has to migrate, and once with the local mirror's own
``config.json``, whichever layout that mirror is in.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest

from esmfold2_atomworks import paths
from esmfold2_atomworks.model.esmfold2 import AtomWorksESMFold2


def _wrapper(net) -> AtomWorksESMFold2:
    """A wrapper around ``net`` without loading anything."""
    model = AtomWorksESMFold2.__new__(AtomWorksESMFold2)
    model.net = net
    return model


# -- component seams ---------------------------------------------------------


@pytest.mark.offline
def test_esmc_is_read_under_the_native_name():
    backbone = object()
    assert _wrapper(SimpleNamespace(esmc=backbone)).esmc is backbone


@pytest.mark.offline
def test_esmc_falls_back_to_a_private_layout():
    backbone = object()
    assert _wrapper(SimpleNamespace(_esmc=backbone)).esmc is backbone


@pytest.mark.offline
def test_esmc_is_none_only_when_nothing_is_attached():
    # The native module keeps flags beside the backbone; neither is it.
    net = SimpleNamespace(esmc=None, _esmc_fp8=False, _offload_esmc=False)
    assert _wrapper(net).esmc is None


@pytest.mark.offline
def test_trunk_and_head_are_the_native_modules():
    trunk, head = object(), object()
    model = _wrapper(SimpleNamespace(folding_trunk=trunk, structure_head=head))
    assert model.folding_trunk is trunk
    assert model.structure_head is head


@pytest.mark.offline
def test_a_missing_trunk_raises_instead_of_answering_none():
    # No supported ESMFold2 lacks one, so None would only hide a renamed module.
    with pytest.raises(AttributeError, match="folding_trunk"):
        _ = _wrapper(SimpleNamespace()).folding_trunk


@pytest.mark.offline
def test_a_missing_head_raises_instead_of_answering_none():
    with pytest.raises(AttributeError, match="structure_head"):
        _ = _wrapper(SimpleNamespace()).structure_head


# -- representation_dims -----------------------------------------------------


def _config(**top):
    """A config shaped like ``EsmFold2Config`` after its field migration."""
    diffusion = SimpleNamespace(
        token_hidden_size=768,
        atom_encoder=SimpleNamespace(hidden_size=128),
        c_s_inputs=451,
    )
    fields = {
        "hidden_size": 384,
        "pairwise_hidden_size": 256,
        "structure_head": SimpleNamespace(diffusion_module=diffusion),
    }
    fields.update(top)
    return SimpleNamespace(**fields)


EXPECTED = {
    "d_pair": 256,
    "d_single_declared": 384,
    "c_token": 768,
    "c_atom": 128,
    "c_s_inputs": 451,
}


@pytest.mark.offline
def test_dims_read_the_live_field_names():
    model = _wrapper(SimpleNamespace(config=_config()))
    assert model.representation_dims() == EXPECTED


@pytest.mark.offline
def test_dims_fall_back_to_the_pre_migration_names():
    legacy = SimpleNamespace(
        d_single=384,
        d_pair=256,
        structure_head=SimpleNamespace(
            diffusion_module=SimpleNamespace(c_token=768, c_atom=128, c_s_inputs=451)
        ),
    )
    assert _wrapper(SimpleNamespace(config=legacy)).representation_dims() == EXPECTED


@pytest.mark.offline
def test_a_missing_width_raises_instead_of_reading_zero():
    config = _config()
    del config.pairwise_hidden_size
    with pytest.raises(AttributeError, match="pairwise_hidden_size"):
        _wrapper(SimpleNamespace(config=config)).representation_dims()


@pytest.mark.offline
def test_a_zero_width_raises():
    config = _config(hidden_size=0)
    with pytest.raises(ValueError, match="hidden_size"):
        _wrapper(SimpleNamespace(config=config)).representation_dims()


def test_dims_survive_the_upstream_field_migration():
    """A pre-alignment config, as the reference checkpoint ships, renamed on load."""
    pytest.importorskip("torch")
    config_module = pytest.importorskip("esm.models.esmfold2.config")
    legacy = {
        "d_single": 384,
        "d_pair": 256,
        "structure_head": {
            "diffusion_module": {"c_token": 768, "c_atom": 128, "c_s_inputs": 451}
        },
    }
    with pytest.warns(FutureWarning, match="pre-alignment"):
        config = config_module.EsmFold2Config(**legacy)
    dims = _wrapper(SimpleNamespace(config=config)).representation_dims()
    assert dims == EXPECTED


def test_dims_of_the_local_checkpoint_are_all_real():
    """The mirror's own config.json, pre-alignment or not."""
    pytest.importorskip("torch")
    config_module = pytest.importorskip("esm.models.esmfold2.config")
    weights = paths.ESMFOLD2_WEIGHTS.standard
    if not (weights / "config.json").is_file():
        pytest.skip(f"no local ESMFold2 mirror at {weights}")
    with warnings.catch_warnings():
        # A pre-alignment config announces its migration; that is not a failure.
        warnings.simplefilter("ignore", FutureWarning)
        config = config_module.EsmFold2Config.from_pretrained(str(weights))
    dims = _wrapper(SimpleNamespace(config=config)).representation_dims()
    assert dims == EXPECTED
