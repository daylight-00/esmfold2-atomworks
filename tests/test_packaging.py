"""The Hydra configs must be findable in both layouts.

Configs live at the repo root in a source checkout and inside the package once
installed, because ``pkg://esmfold2_atomworks.configs`` resolves there. These
tests cover the source side; ``scripts/check_packaging.py`` covers the
installed side and runs in CI against a real wheel, which is the only place the
packaging failure is visible.
"""

from __future__ import annotations

import pytest
import tomllib
from omegaconf import OmegaConf

from esmfold2_atomworks import paths

EXPECTED_GROUP_TARGETS = {
    "model/esmfold2.yaml": "esmfold2_atomworks.model.esmfold2.AtomWorksESMFold2",
    "data/atomworks.yaml": "esmfold2_atomworks.data.pipelines.build_esmfold2_pipeline",
    "inference_engine/esmfold2.yaml": (
        "esmfold2_atomworks.inference.engine.ESMFold2InferenceEngine"
    ),
}


@pytest.mark.offline
def test_config_dir_resolves():
    directory = paths.config_dir()
    assert directory.is_dir()
    assert (directory / "inference.yaml").exists()


@pytest.mark.offline
@pytest.mark.parametrize("relative", sorted(EXPECTED_GROUP_TARGETS))
def test_group_configs_carry_their_targets(relative):
    node = OmegaConf.load(paths.config_dir() / relative)
    assert node.get("_target_") == EXPECTED_GROUP_TARGETS[relative]


@pytest.mark.offline
def test_the_wheel_is_told_to_ship_the_configs():
    """Guards the force-include, without building a wheel.

    A source checkout works whether or not this mapping is declared, so nothing
    else in the fast test suite would notice it being dropped.
    """
    pyproject = tomllib.loads((paths.REPO_ROOT / "pyproject.toml").read_text())
    force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"][
        "force-include"
    ]
    assert force_include.get("configs") == "esmfold2_atomworks/configs", (
        "pyproject must map configs/ into the package, or an installed wheel "
        "cannot resolve pkg://esmfold2_atomworks.configs"
    )


@pytest.mark.offline
def test_every_group_file_referenced_by_a_default_exists():
    """A defaults entry naming a missing file fails only at compose time."""
    config_dir = paths.config_dir()
    for entry in sorted(config_dir.rglob("*.yaml")):
        node = OmegaConf.load(entry)
        for default in node.get("defaults") or []:
            if not isinstance(default, str) or default == "_self_":
                continue
            candidate = (entry.parent / f"{default}.yaml").resolve()
            assert candidate.exists(), (
                f"{entry.relative_to(config_dir)} lists default {default!r}, "
                f"but {candidate} does not exist"
            )


@pytest.mark.offline
def test_searchpath_names_only_resolvable_packages():
    """Hydra warns on every compose for a searchpath it cannot resolve."""
    node = OmegaConf.load(paths.config_dir() / "inference.yaml")
    searchpath = list(node.hydra.searchpath)
    assert searchpath == ["pkg://esmfold2_atomworks.configs"], searchpath
