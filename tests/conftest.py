"""Fixtures for esmfold2-foundry tests.

Structures come from the AtomWorks checkout rather than being copied in. They
are the same files AtomWorks tests against, so a parse-behaviour change upstream
shows up here as a test failure instead of being masked by a stale copy.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from esmfold2_foundry import paths

#: Structures used across the parity tests, by short name.
FIXTURES: dict[str, str] = {
    "lysozyme": "6lyz.bcif",  # clean monomer, 129 residues
    "myoglobin": "101m_arginine_nh1nh2_swapped.cif",  # monomer + HEM + NBN
    "hemoglobin": "2hhb.cif.gz",  # A2B2 multimer + 4 HEM
    "modified": "1a8o_modified.cif",  # monomer with 4 MSE
    "zinc": "test_cif_loading_4q8n.cif.gz",  # protein + ZN
    "flavoprotein": "8cjg_from_af3.cif",  # 663 residues + FAD + UV3
    "afdb": "UniRef50_A0A0S8JQ92_AF2_predicted.pdb",  # predicted monomer
    "unl": "example_distillation_output.cif",  # protein + UNL: must be refused
}


def _atomworks_data_dir() -> Path:
    return paths.DESIGN_ROOT / "atomworks" / "tests" / "data" / "io"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    directory = _atomworks_data_dir()
    if not directory.is_dir():
        pytest.skip(f"AtomWorks test data not found at {directory}")
    return directory


@pytest.fixture(scope="session")
def ccd() -> None:
    """Load the CCD dictionary once for the whole session (~9 s, ~50k entries)."""
    load_ccd = pytest.importorskip("esm.models.esmfold2").load_ccd
    load_ccd()


@pytest.fixture(scope="session")
def parsed(data_dir):
    """``name -> (atom_array, chain_info)``, parsed once and memoized."""
    from atomworks.io import parse

    cache: dict[str, tuple] = {}

    def _get(name: str):
        if name not in cache:
            path = data_dir / FIXTURES[name]
            if not path.exists():
                pytest.skip(f"fixture {name} missing at {path}")
            result = parse(path)
            cache[name] = (result["asym_unit"][0], result["chain_info"])
        return cache[name]

    return _get


def requires_weights() -> bool:
    return paths.ESMFOLD2_WEIGHTS.standard.exists() or bool(
        os.environ.get("EF_WEIGHTS")
    )
