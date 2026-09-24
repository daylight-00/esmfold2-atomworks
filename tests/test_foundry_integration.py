"""The optional Foundry integration holds against the Foundry it is tested with.

docs/06 describes the integration and the Foundry contracts it relies on, but
description is not verification: every claim in it is about a repository that
moves independently of this one. These tests check the claims against the
installed Foundry, so that "Foundry-shaped" becomes "Foundry-compatible".

They do **not** modify the Foundry checkout. Copying this package into it and
editing its ``pyproject.toml`` would verify the same things while leaving a
dirty tree behind, so instead each contract is checked where it lives: the
trainer base class by subclassing it, the engine surface by comparing it, the
registration instructions by reading Foundry's own ``pyproject.toml``.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import tomllib

from esmfold2_atomworks import paths

foundry_trainers = pytest.importorskip("foundry.trainers.fabric")
foundry_engines = pytest.importorskip("foundry.inference_engines.base")


def test_our_trainer_is_a_concrete_fabric_trainer():
    """Subclassing is the contract; leaving an abstract method is the failure."""
    from esmfold2_atomworks.training.trainer import ESMFold2Trainer

    assert issubclass(ESMFold2Trainer, foundry_trainers.FabricTrainer)
    assert not getattr(ESMFold2Trainer, "__abstractmethods__", frozenset()), (
        f"unimplemented: {sorted(ESMFold2Trainer.__abstractmethods__)}"
    )


def test_our_trainer_matches_the_abstract_signatures():
    """A renamed parameter upstream would break the call, not the subclass check."""
    from esmfold2_atomworks.training.trainer import ESMFold2Trainer

    base = foundry_trainers.FabricTrainer
    for name in ("training_step", "validation_step"):
        ours = inspect.signature(getattr(ESMFold2Trainer, name))
        theirs = inspect.signature(getattr(base, name))
        assert list(ours.parameters) == list(theirs.parameters), (
            f"{name}: {list(ours.parameters)} != {list(theirs.parameters)}"
        )


def test_our_engine_offers_the_base_engine_surface():
    """Deliberately not a subclass (docs/03), so the surface is checked instead."""
    from esmfold2_atomworks.inference.engine import ESMFold2InferenceEngine

    base = foundry_engines.BaseInferenceEngine
    for name in ("initialize", "run", "forward", "__call__", "__enter__", "__exit__"):
        assert hasattr(base, name), f"Foundry's engine no longer has {name}"
        assert hasattr(ESMFold2InferenceEngine, name), f"ours is missing {name}"


def test_the_checkpoint_registry_takes_the_fields_docs_06_claims():
    """docs/06 tells the reader to add a RegisteredCheckpoint(url, filename, description)."""
    registry = pytest.importorskip("foundry.inference_engines.checkpoint_registry")

    assert isinstance(registry.REGISTERED_CHECKPOINTS, dict)
    fields = set(inspect.signature(registry.RegisteredCheckpoint).parameters)
    assert {"url", "filename", "description"} <= fields, fields


def _foundry_checkout() -> Path:
    """The Foundry source checkout -- provided it is the Foundry under test.

    The repository contracts below read files only a checkout has: its root
    ``pyproject.toml`` and its ``models/``. Beside an installed rc-foundry those
    describe a different Foundry from the one the runtime tests above import,
    so they skip rather than mix two versions in one run.
    """
    import foundry

    tree = paths.SOURCE_TREES["foundry"].resolve()
    if not Path(foundry.__file__).resolve().is_relative_to(tree):
        pytest.skip("the Foundry under test is not the checkout these contracts read")
    return tree.parent


def _foundry_pyproject() -> dict:
    path = _foundry_checkout() / "pyproject.toml"
    if not path.exists():
        pytest.skip(f"no Foundry pyproject at {path}")
    return tomllib.loads(path.read_text())


def test_the_registration_points_docs_06_names_still_exist():
    """Every table docs/06 tells the reader to edit must be there to edit.

    Upstream added two of these after this package was written; if it adds a
    third, the instructions go stale silently.
    """
    pyproject = _foundry_pyproject()
    wheel = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]

    assert "optional-dependencies" in pyproject["project"]
    assert "scripts" in pyproject["project"]
    assert "packages" in wheel
    assert "force-include" in wheel
    assert "files" in pyproject["tool"]["mypy"]
    assert "testpaths" in pyproject["tool"]["pytest"]["ini_options"]


def test_models_are_registered_in_the_root_pyproject_not_their_own():
    """The premise of docs/06: CONTRIBUTING describes per-model pyprojects, the repo has none."""
    pyproject = _foundry_pyproject()
    packages = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert any(entry.startswith("models/") for entry in packages), packages

    models_dir = _foundry_checkout() / "models"
    stray = sorted(p for p in models_dir.glob("*/pyproject.toml"))
    assert not stray, (
        f"a model now carries its own pyproject ({stray}); docs/06 says the repo "
        "registers models centrally and should be revisited"
    )


def test_the_data_pipeline_config_instantiates(parsed, ccd):
    """The config Foundry would instantiate must actually build a working pipeline."""
    hydra = pytest.importorskip("hydra")
    from omegaconf import OmegaConf

    node = OmegaConf.load(paths.config_dir() / "data" / "atomworks_inference.yaml")
    # `defaults` is Hydra's composition key and is not part of instantiate().
    node.pop("defaults", None)
    node._target_ = "esmfold2_atomworks.data.pipelines.build_esmfold2_pipeline"

    pipeline = hydra.utils.instantiate(node, _recursive_=False)

    atoms, chain_info = parsed("lysozyme")
    out = pipeline(
        {"example_id": "lysozyme", "atom_array": atoms, "chain_info": chain_info}
    )
    assert len(out["feats"]) == 29
    assert out["chain_infos"]
