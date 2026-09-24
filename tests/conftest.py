"""Fixtures for esmfold2-atomworks tests.

Structures come from the AtomWorks checkout rather than being copied in. They
are the same files AtomWorks tests against, so a parse-behaviour change upstream
shows up here as a test failure instead of being masked by a stale copy.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from esmfold2_atomworks import paths

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
    """``name -> (atom_array, chain_info)``, parsed once and memoized.

    Skips rather than errors when the source trees are absent, so that a test
    requesting this fixture behind a `gpu`/weights guard reports "skipped"
    instead of a collection error in an environment that was never expected to
    have them.
    """
    parse = pytest.importorskip("atomworks.io").parse

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


GOLD_DIR = Path(__file__).parent / "data" / "gold"


def load_gold_document(name: str) -> dict:
    """The frozen expected content for *name*, as checked in."""
    import json

    path = GOLD_DIR / f"{name}.json"
    if not path.exists():
        pytest.skip(f"no gold fixture for {name}")
    return json.loads(path.read_text())


def gold_structure_prediction_input(name: str):
    """Build a ``StructurePredictionInput`` from the frozen gold document.

    This is the **independent** side of the parity comparison. It is built from
    a checked-in file and never touches
    :mod:`esmfold2_atomworks.data.atomworks_to_esm`, so a systematic mistake in
    the adapter -- mapping a ligand to the wrong CCD code, say -- cannot be
    copied into the thing the adapter is compared against.
    """
    from esm.models.esmfold2.types import (
        LigandInput,
        Modification,
        ProteinInput,
        StructurePredictionInput,
    )

    document = load_gold_document(name)
    entries: list = []
    for chain in document["chains"]:
        if chain["kind"] == "protein":
            mods = [
                Modification(position=m["position"], ccd=m["ccd"])
                for m in chain.get("modifications", [])
            ]
            entries.append(
                ProteinInput(
                    id=chain["id"],
                    sequence=chain["sequence"],
                    modifications=mods or None,
                )
            )
        elif chain["kind"] == "ligand":
            entries.append(LigandInput(id=chain["id"], ccd=list(chain["ccd"])))
        else:
            raise AssertionError(
                f"gold fixture {name} has an unhandled chain kind {chain['kind']!r}"
            )
    return StructurePredictionInput(sequences=entries)


@pytest.fixture(scope="session")
def gold():
    """``name -> StructurePredictionInput`` from the frozen fixtures."""
    return gold_structure_prediction_input


@pytest.fixture(scope="session")
def gold_document():
    """``name -> dict`` for the frozen fixtures."""
    return load_gold_document


def requires_weights() -> bool:
    return paths.ESMFOLD2_WEIGHTS.standard.exists() or bool(
        os.environ.get("EF_WEIGHTS")
    )
