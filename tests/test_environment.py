"""The environment this repo assumes, as assertions.

Same checks as ``esmfold2-foundry doctor``, so a broken environment fails here
rather than deep inside a fold.
"""

from __future__ import annotations

import importlib.util

import pytest

from esmfold2_foundry import paths


def test_design_root_is_discovered_not_assumed():
    """Discovery must find the real tree, including from a git worktree.

    A worktree sits several levels below the repo, so ``REPO_ROOT.parent`` would
    resolve into ``.claude/worktrees`` and every source tree would vanish.
    """
    for marker in ("esm", "atomworks", "foundry"):
        assert (paths.DESIGN_ROOT / marker).is_dir(), (
            f"{marker} not found under DESIGN_ROOT={paths.DESIGN_ROOT}"
        )


@pytest.mark.parametrize("module", ["esm", "atomworks", "foundry", "mpnn"])
def test_source_tree_exists(module):
    assert paths.SOURCE_TREES[module].is_dir()


@pytest.mark.parametrize(
    "module", ["numpy", "torch", "biotite", "atomworks", "esm", "foundry"]
)
def test_module_importable(module):
    assert importlib.util.find_spec(module) is not None


def test_a_native_model_class_is_importable():
    """One of the two packagings of the ESMFold2 module must be present.

    Up to esm 3.3 the module ships in a fork of ``transformers``; from esm 3.4
    it is in ``esm`` itself. The input pipeline imports fine without either, so
    this is worth asserting separately.
    """
    from esmfold2_foundry.doctor import _find_model_module

    found, detail = _find_model_module()
    assert found, detail


def test_biotite_version_keeps_the_import_atomworks_needs():
    """1.7 moved connect_via_residue_names out of biotite.structure.bonds."""
    from biotite.structure.bonds import connect_via_residue_names  # noqa: F401


def test_pythonpath_entries_are_reported_in_order():
    entries = paths.pythonpath_entries()
    assert entries[-1].name == "src"
    assert len(entries) == len(paths.SOURCE_TREES) + 1
