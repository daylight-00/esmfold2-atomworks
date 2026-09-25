"""Verify that the environment this repo assumes is actually present.

``python -m esmfold2_atomworks.doctor`` (or ``esmfold2-atomworks doctor``)
reports where the upstreams come from, checks the imports and the weights, and
runs a real parity check when AtomWorks' test structures are available,
printing what is wrong rather than failing later inside a fold. ``tests/test_environment.py`` asserts the same things.

Only a missing import or a failed parity check is a failure. Source trees,
``UPSTREAM.lock`` and AtomWorks' test structures belong to the reference
environment (``reproducibility/``); an ordinary install has none of them, and
is no less complete for it. Foundry is optional -- it backs the training
integration and nothing else.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

from esmfold2_atomworks import paths


def _check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {label}{(': ' + detail) if detail else ''}")
    return ok


def _optional(label: str, detail: str) -> None:
    print(f"  [ -- ] {label}: {detail}")


_ONLY_FOR_FOUNDRY = "absent; needed only for the optional Foundry integration"


#: The distribution that provides each upstream module when it is installed
#: rather than put on PYTHONPATH as a source tree.
_DISTRIBUTIONS = {"esm": "esm", "atomworks": "atomworks", "foundry": "rc-foundry"}


def _origin(module: str) -> Path | None:
    """Where *module* is imported from, or ``None`` if it is not importable."""
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    return Path(spec.origin).resolve() if spec and spec.origin else None


def _installed_version(module: str) -> str | None:
    try:
        return importlib.metadata.version(_DISTRIBUTIONS[module])
    except (KeyError, importlib.metadata.PackageNotFoundError):
        return None


def _from_source_tree(module: str, origin: Path) -> bool:
    tree = paths.SOURCE_TREES.get(module)
    return tree is not None and origin.is_relative_to(tree.resolve())


def check_source_trees() -> bool:
    """Where each upstream comes from: a source tree, or an installed package.

    Either is fine, so this never fails; the import check below is what
    requires the modules to be there at all.
    """
    print(f"upstreams (DESIGN_ROOT = {paths.DESIGN_ROOT})")
    for module in paths.SOURCE_TREES:
        origin = _origin(module) if _importable(module) else None
        if origin is None:
            detail = _ONLY_FOR_FOUNDRY if module in paths.OPTIONAL_TREES else "absent"
            _optional(f"{module:10s}", detail)
        elif _from_source_tree(module, origin):
            _check(f"{module:10s} source tree {paths.SOURCE_TREES[module]}", True)
        elif (version := _installed_version(module)) is not None:
            _check(f"{module:10s} installed package {version}", True)
        else:
            _check(f"{module:10s} {origin.parent}", True)
    return True


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def check_imports() -> bool:
    print("imports")
    ok = True
    for module in ("numpy", "torch", "biotite", "atomworks", "esm"):
        ok &= _check(module, _importable(module))
    for module in paths.OPTIONAL_TREES:
        if _importable(module):
            _check(module, True, "optional")
        else:
            _optional(module, _ONLY_FOR_FOUNDRY)

    # The model module is packaged differently across esm versions, and its
    # absence is a distinct and easily-missed failure -- the input pipeline
    # imports fine without it.
    found, detail = _find_model_module()
    ok &= _check("ESMFold2 model class", found, detail)
    return ok


def _find_model_module() -> tuple[bool, str]:
    """Whether a native ESMFold2 module is importable, and from where."""
    for module, detail in (
        ("esm.models.esmfold2.model", "esm >= 3.4"),
        (
            "transformers.models.esmfold2.modeling_esmfold2",
            "transformers fork, esm <= 3.3",
        ),
    ):
        try:
            if importlib.util.find_spec(module) is not None:
                return True, detail
        except (ImportError, ValueError, ModuleNotFoundError):
            continue
    return False, "install esm >= 3.4, which ships the model"


def check_upstream_revisions() -> bool:
    """Report the source trees' revisions against ``UPSTREAM.lock``.

    Drift is reported but never fails the check. The adapter is deliberately
    version-tolerant, so a newer tree is something to re-verify rather than
    something to refuse -- and refusing would make ``doctor`` useless on the
    day an upstream releases.
    """
    print("upstream revisions")
    locked = paths.read_upstream_lock()
    if not locked:
        _optional(
            "UPSTREAM.lock", "not found; it ships with the repository, not the package"
        )
        return True

    for tree in paths.LOCKED_TREES:
        origin = _origin(tree) if _importable(tree) else None
        want = locked.get(tree, {})
        pinned = want.get("commit")
        version = want.get("version", "?")
        if origin is None:
            detail = _ONLY_FOR_FOUNDRY if tree in paths.OPTIONAL_TREES else "absent"
            _optional(f"{tree:10s}", detail)
            continue
        if not _from_source_tree(tree, origin):
            # An installed package: its version is all there is to compare.
            installed = _installed_version(tree)
            detail = (
                f"matches locked version ({version})"
                if installed == version
                else f"locked {version}; re-verify parity"
            )
            _check(f"{tree:10s} package {installed}", True, detail)
            continue
        actual = paths.tree_revision(tree)
        if actual is None:
            _check(f"{tree:10s} (not a git checkout)", True, f"locked {version}")
        elif pinned and actual == pinned:
            _check(f"{tree:10s} {actual[:12]}", True, f"matches lock ({version})")
        else:
            _check(
                f"{tree:10s} {actual[:12]}",
                True,
                f"DRIFT from locked {str(pinned)[:12]} ({version}); re-verify parity",
            )
    return True


def check_weights() -> bool:
    """Report where the weights would come from.

    Never a failure: with no local mirror the Hub id is used and
    ``from_pretrained`` downloads on first use. Only the CPU-only checks below
    are required for the adapter, which needs no weights at all.
    """
    print("weights")
    for name in paths.HUB_IDS:
        local = getattr(paths.ESMFOLD2_WEIGHTS, name)
        resolved = paths.ESMFOLD2_WEIGHTS.resolve(name)
        if local.is_dir():
            detail = f"local mirror; {_esmc_detail(local)}"
        else:
            detail = "will download from the Hub"
        _check(f"{name:12s} {resolved}", True, detail)
    return True


def _esmc_detail(checkpoint: Path) -> str:
    """Which ESMC backbone a local checkpoint would be given, from its config.json.

    The two layouts differ in what decides it: a bundled checkpoint carries its
    own, and a separate one names it, which ``paths.resolve_esmc`` turns into a
    location. A name that resolves nowhere is reported here, where it is cheap,
    instead of at load time.
    """
    try:
        raw = json.loads((checkpoint / "config.json").read_text())
    except (OSError, ValueError) as error:
        return f"config.json unreadable ({error})"
    if raw.get("esmc_config") is not None:
        return "ESMC bundled"
    esmc_id = raw.get("esmc_id")
    if esmc_id is None:
        return "ESMC named by the upstream default"
    try:
        return f"ESMC from {paths.resolve_esmc(esmc_id)}"
    except FileNotFoundError:
        return f"ESMC {esmc_id!r} resolves nowhere; loading will fail"


def check_adapter() -> bool:
    """Exercise the adapter end to end on CPU, without weights.

    This is the check that matters most: it is the core milestone in
    miniature, and it fails for every reason the others do plus a few of its
    own.
    """
    print("adapter")
    fixture = paths.DESIGN_ROOT / "atomworks" / "tests" / "data" / "io" / "2hhb.cif.gz"
    if not fixture.exists():
        _optional(
            "feature parity",
            "needs AtomWorks' test structures, which come with an atomworks "
            "source checkout (see reproducibility/)",
        )
        return True
    try:
        from esmfold2_atomworks.parity.run import parity_for_structure

        outcome = parity_for_structure(fixture)
    except Exception as error:  # noqa: BLE001
        return _check("feature parity", False, f"{type(error).__name__}: {error}")
    return _check("feature parity (2hhb)", outcome.ok, outcome.detail)


def main() -> int:
    print(f"esmfold2-atomworks doctor (python {sys.version.split()[0]})")
    # REPO_ROOT is a checkout's root; from an installed wheel it points above
    # site-packages, which is no place worth printing.
    if (paths.REPO_ROOT / "pyproject.toml").is_file():
        print(f"  repo: {paths.REPO_ROOT}")
    else:
        print(f"  package: {Path(__file__).resolve().parent}")
    results = [
        check_source_trees(),
        check_upstream_revisions(),
        check_imports(),
        check_weights(),
        check_adapter(),
    ]
    ok = all(results)
    print("\n" + ("all checks passed" if ok else "some checks FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
