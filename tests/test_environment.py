"""The environment this repo assumes, as assertions.

Same checks as ``esmfold2-atomworks doctor``, so a broken environment fails here
rather than deep inside a fold.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from esmfold2_atomworks import paths


def _uses_source_trees() -> bool:
    """Whether esm comes from a source tree -- the reference environment -- or,
    as in an ordinary install, from a package."""
    spec = importlib.util.find_spec("esm")
    if spec is None or spec.origin is None:
        return False
    return not {"site-packages", "dist-packages"} & set(Path(spec.origin).parts)


reference_environment_only = pytest.mark.skipif(
    not _uses_source_trees(),
    reason="upstreams are installed packages here, not the reference source trees",
)


@reference_environment_only
def test_design_root_is_discovered_not_assumed():
    """Discovery must find the real tree, including from a git worktree.

    A worktree can sit several levels below the repository, where
    ``REPO_ROOT.parent`` names the directory holding it and every source tree
    would vanish.
    """
    for marker in paths.REQUIRED_TREES:
        assert (paths.DESIGN_ROOT / marker).is_dir(), (
            f"{marker} not found under DESIGN_ROOT={paths.DESIGN_ROOT}"
        )


@reference_environment_only
@pytest.mark.parametrize("module", paths.REQUIRED_TREES)
def test_required_source_tree_exists(module):
    assert paths.SOURCE_TREES[module].is_dir()


@pytest.mark.parametrize("module", ["numpy", "torch", "biotite", "atomworks", "esm"])
def test_module_importable(module):
    assert importlib.util.find_spec(module) is not None


def test_foundry_is_not_required():
    """The core is AtomWorks <-> ESMFold2; Foundry backs one optional integration."""
    assert "foundry" in paths.OPTIONAL_TREES
    assert "foundry" not in paths.REQUIRED_TREES


def test_doctor_passes_a_workspace_without_foundry(monkeypatch, tmp_path, capsys):
    """Absent is reported, not failed: the adapter needs nothing from Foundry."""
    from esmfold2_atomworks import doctor

    monkeypatch.setitem(paths.SOURCE_TREES, "foundry", tmp_path / "absent")
    importable = doctor._importable
    monkeypatch.setattr(
        doctor, "_importable", lambda module: module != "foundry" and importable(module)
    )

    assert doctor.check_source_trees()
    assert doctor.check_imports()
    assert doctor.check_upstream_revisions()
    assert "FAIL" not in capsys.readouterr().out


def test_a_native_model_class_is_importable():
    """ESMFold2's native module must be importable."""
    from esmfold2_atomworks.doctor import _find_model_module

    found, detail = _find_model_module()
    assert found, detail


def test_biotite_version_keeps_the_import_atomworks_needs():
    """1.7 moved connect_via_residue_names out of biotite.structure.bonds."""
    from biotite.structure.bonds import connect_via_residue_names  # noqa: F401


def test_pythonpath_entries_are_reported_in_order():
    entries = paths.pythonpath_entries()
    assert entries[-1].name == "src"
    assert len(entries) == len(paths.SOURCE_TREES) + 1


def test_doctor_passes_an_ordinary_install(monkeypatch, tmp_path, capsys):
    """No source trees, no UPSTREAM.lock, no AtomWorks test data: nothing fails.

    That is what `pip install` gives, and it is a complete installation -- those
    three belong to the reference environment, not to the package.
    """
    from esmfold2_atomworks import doctor

    for module in list(paths.SOURCE_TREES):
        monkeypatch.setitem(paths.SOURCE_TREES, module, tmp_path / "trees" / module)
    monkeypatch.setattr(paths, "UPSTREAM_LOCK", tmp_path / "UPSTREAM.lock")
    monkeypatch.setattr(paths, "DESIGN_ROOT", tmp_path)

    assert doctor.check_source_trees()
    assert doctor.check_upstream_revisions()
    assert doctor.check_adapter()
    assert "FAIL" not in capsys.readouterr().out


def test_doctor_says_when_feature_parity_was_not_run(monkeypatch, tmp_path, capsys):
    """A pass that skipped the parity check must not read as one that ran it."""
    from esmfold2_atomworks import doctor, paths

    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(paths, "DESIGN_ROOT", tmp_path)
    for name in (
        "check_source_trees",
        "check_upstream_revisions",
        "check_imports",
        "check_weights",
    ):
        monkeypatch.setattr(doctor, name, lambda: True)

    assert doctor.main() == 0
    assert "all checks passed (feature parity not run)" in capsys.readouterr().out


def test_doctor_runs_parity_on_the_repositorys_own_structure(monkeypatch, tmp_path):
    from esmfold2_atomworks import doctor, paths

    own = tmp_path / "tests" / "data" / "structures"
    own.mkdir(parents=True)
    (own / "2hhb.cif.gz").write_bytes(b"")
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(paths, "DESIGN_ROOT", tmp_path / "elsewhere")

    assert doctor._parity_fixture() == own / "2hhb.cif.gz"
