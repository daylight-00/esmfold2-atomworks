"""Verify that the environment this repo assumes is actually present.

``python -m esmfold2_atomworks.doctor`` (or ``esmfold2-atomworks doctor``) checks the
source trees, the weights and the imports, and prints what is wrong rather than
failing later inside a fold. ``tests/test_environment.py`` asserts the same
things.

Foundry is optional -- it backs the training integration and nothing else -- so
its absence is reported, never counted as a failure.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys

from esmfold2_atomworks import paths


def _check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {label}{(': ' + detail) if detail else ''}")
    return ok


def _optional(label: str, detail: str) -> None:
    print(f"  [ -- ] {label}: {detail}")


_ONLY_FOR_FOUNDRY = "absent; needed only for the optional Foundry integration"


def check_source_trees() -> bool:
    print(f"source trees (DESIGN_ROOT = {paths.DESIGN_ROOT})")
    ok = True
    for module, tree in paths.SOURCE_TREES.items():
        if module in paths.OPTIONAL_TREES and not tree.is_dir():
            _optional(f"{module:10s} {tree}", _ONLY_FOR_FOUNDRY)
            continue
        ok &= _check(f"{module:10s} {tree}", tree.is_dir())
    return ok


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
    return False, "install esm >= 3.4, or the Biohub transformers fork for esm <= 3.3"


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
        _check("UPSTREAM.lock", False, "missing; parity results are unanchored")
        return True

    for tree in paths.LOCKED_TREES:
        if tree in paths.OPTIONAL_TREES and not paths.SOURCE_TREES[tree].is_dir():
            _optional(f"{tree:10s}", _ONLY_FOR_FOUNDRY)
            continue
        want = locked.get(tree, {})
        pinned = want.get("commit")
        actual = paths.tree_revision(tree)
        version = want.get("version", "?")
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
    for name in ("standard", "fast"):
        local = getattr(paths.ESMFOLD2_WEIGHTS, name)
        resolved = paths.ESMFOLD2_WEIGHTS.resolve(name)
        detail = "local mirror" if local.is_dir() else "will download from the Hub"
        _check(f"{name:8s} {resolved}", True, detail)
    return True


def check_adapter() -> bool:
    """Exercise the adapter end to end on CPU, without weights.

    This is the check that matters most: it is the core milestone in
    miniature, and it fails for every reason the others do plus a few of its
    own.
    """
    print("adapter")
    fixture = paths.DESIGN_ROOT / "atomworks" / "tests" / "data" / "io" / "2hhb.cif.gz"
    if not fixture.exists():
        return _check("feature parity", False, f"fixture missing: {fixture}")
    try:
        from esmfold2_atomworks.parity.run import parity_for_structure

        outcome = parity_for_structure(fixture)
    except Exception as error:  # noqa: BLE001
        return _check("feature parity", False, f"{type(error).__name__}: {error}")
    return _check("feature parity (2hhb)", outcome.ok, outcome.detail)


def main() -> int:
    print(f"esmfold2-atomworks doctor (python {sys.version.split()[0]})")
    print(f"  repo: {paths.REPO_ROOT}")
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
