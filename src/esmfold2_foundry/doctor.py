"""Verify that the environment this repo assumes is actually present.

``python -m esmfold2_foundry.doctor`` (or ``esmfold2-foundry doctor``) checks the
source trees, the weights and the imports, and prints what is wrong rather than
failing later inside a fold. ``tests/test_environment.py`` asserts the same
things.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys

from esmfold2_foundry import paths


def _check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {label}{(': ' + detail) if detail else ''}")
    return ok


def check_source_trees() -> bool:
    print(f"source trees (DESIGN_ROOT = {paths.DESIGN_ROOT})")
    ok = True
    for module, tree in paths.SOURCE_TREES.items():
        ok &= _check(f"{module:10s} {tree}", tree.is_dir())
    return ok


def check_imports() -> bool:
    print("imports")
    ok = True
    for module in ("numpy", "torch", "biotite", "atomworks", "esm", "foundry"):
        spec = None
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        ok &= _check(module, spec is not None)

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

    This is the check that matters most: it is the Phase 1 milestone in
    miniature, and it fails for every reason the others do plus a few of its
    own.
    """
    print("adapter")
    fixture = paths.DESIGN_ROOT / "atomworks" / "tests" / "data" / "io" / "2hhb.cif.gz"
    if not fixture.exists():
        return _check("feature parity", False, f"fixture missing: {fixture}")
    try:
        from esmfold2_foundry.parity.run import parity_for_structure

        outcome = parity_for_structure(fixture)
    except Exception as error:  # noqa: BLE001
        return _check("feature parity", False, f"{type(error).__name__}: {error}")
    return _check("feature parity (2hhb)", outcome.ok, outcome.detail)


def main() -> int:
    print(f"esmfold2-foundry doctor (python {sys.version.split()[0]})")
    print(f"  repo: {paths.REPO_ROOT}")
    results = [
        check_source_trees(),
        check_imports(),
        check_weights(),
        check_adapter(),
    ]
    ok = all(results)
    print("\n" + ("all checks passed" if ok else "some checks FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
