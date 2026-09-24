"""The core is AtomWorks <-> ESMFold2; Foundry backs one optional integration.

That is a claim about imports, so it is checked as one: importing every module
of the package pulls in no part of Foundry, even with Foundry on the path. Only
the training integration reaches for it, and only when a trainer is built.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

_IMPORT_EVERYTHING = textwrap.dedent(
    """
    import importlib
    import pkgutil
    import sys

    import esmfold2_atomworks

    for info in pkgutil.walk_packages(
        esmfold2_atomworks.__path__, "esmfold2_atomworks."
    ):
        importlib.import_module(info.name)
    loaded = [n for n in sys.modules if n == "foundry" or n.startswith("foundry.")]
    print(" ".join(sorted(loaded)))
    """
)


def test_importing_the_package_never_imports_foundry():
    for dependency in ("torch", "esm", "atomworks"):
        pytest.importorskip(dependency)
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_EVERYTHING],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "", f"importing the package loaded {result.stdout}"
