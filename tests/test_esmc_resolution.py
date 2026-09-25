"""The ESMC backbone is found the way the folding weights are.

A checkpoint that does not bundle its backbone names one in ``config.esmc_id``,
and what that string holds varies by where the mirror came from: a Hub id, or an
absolute path on the machine that wrote it. These pin that the name is resolved
through ``paths`` -- a local mirror first -- and that the wrapper attaches what
was resolved rather than what was written. Nothing here loads weights.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from esmfold2_atomworks import paths
from esmfold2_atomworks.model.esmfold2 import attach_esmc

pytestmark = pytest.mark.offline


@pytest.fixture
def models(tmp_path, monkeypatch):
    """An ``EF_MODELS`` holding an ``ESMC-6B`` mirror."""
    (tmp_path / "ESMC-6B").mkdir()
    monkeypatch.setattr(paths, "MODELS", tmp_path)
    return tmp_path


# -- resolve_esmc ------------------------------------------------------------


def test_a_hub_id_uses_the_mirror_beside_it(models):
    assert paths.resolve_esmc("biohub/ESMC-6B") == models / "ESMC-6B"


def test_a_stale_absolute_path_uses_the_mirror_by_name(models):
    moved = "/somewhere/that/moved/ESMC-6B"
    assert paths.resolve_esmc(moved) == models / "ESMC-6B"


def test_the_name_is_the_checkpoints_not_a_default(models):
    # A mirror of one backbone must not stand in for another.
    assert paths.resolve_esmc("biohub/ESMC-600M") == "biohub/ESMC-600M"


def test_same_basename_from_another_hub_namespace_is_not_replaced(models):
    # Same directory name, different identity: the biohub mirror is not it.
    assert paths.resolve_esmc("otherorg/ESMC-6B") == "otherorg/ESMC-6B"


def test_a_stale_path_naming_no_known_mirror_raises(models):
    with pytest.raises(FileNotFoundError, match="ESMC-600M"):
        paths.resolve_esmc("/somewhere/that/moved/ESMC-600M")


def test_mirror_directories_are_unique():
    # The stale-path rule relies on a directory naming one backbone only.
    names = list(paths.ESMC_MIRRORS.values())
    assert len(names) == len(set(names))


def test_an_existing_directory_is_used_when_there_is_no_mirror(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MODELS", tmp_path / "empty")
    elsewhere = tmp_path / "staged" / "ESMC-6B"
    elsewhere.mkdir(parents=True)
    assert paths.resolve_esmc(str(elsewhere)) == elsewhere


def test_a_hub_id_without_a_mirror_is_downloaded(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MODELS", tmp_path)
    assert paths.resolve_esmc("biohub/ESMC-6B") == "biohub/ESMC-6B"


def test_a_path_that_exists_nowhere_raises(tmp_path, monkeypatch):
    # A known name, but no mirror under EF_MODELS to recover it from.
    monkeypatch.setattr(paths, "MODELS", tmp_path)
    with pytest.raises(FileNotFoundError, match="not the name of a mirror"):
        paths.resolve_esmc("/somewhere/that/moved/ESMC-6B")


# -- attach_esmc -------------------------------------------------------------


class _ReleaseNet:
    def __init__(self, **config):
        self.config = SimpleNamespace(esmc_config=None, **config)
        self.calls = []

    def load_esmc(self, esmc_model_path, precision="bf16"):
        self.calls.append((esmc_model_path, precision))


class _ExperimentalNet(_ReleaseNet):
    # Upstream's experimental load_esmc takes no precision and loads bf16.
    def load_esmc(self, esmc_model_path):
        self.calls.append((esmc_model_path,))


def test_a_bundled_backbone_is_left_alone(models):
    net = _ReleaseNet()
    net.config.esmc_config = object()
    assert attach_esmc(net) == "bundled"
    assert net.calls == []


def test_a_separate_backbone_is_attached_from_the_resolved_location(models):
    net = _ReleaseNet(esmc_id="/somewhere/that/moved/ESMC-6B")
    source = attach_esmc(net, "fp32")
    assert source == str(models / "ESMC-6B")
    assert net.calls == [(source, "fp32")]


def test_the_experimental_signature_is_respected(models):
    net = _ExperimentalNet(esmc_id="biohub/ESMC-6B")
    source = attach_esmc(net, "fp32")
    assert net.calls == [(source,)]
