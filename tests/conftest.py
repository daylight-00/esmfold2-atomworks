"""Fixtures for esmfold2-atomworks tests.

Structures come from the AtomWorks checkout rather than being copied in. They
are the same files AtomWorks tests against, so a parse-behaviour change upstream
shows up here as a test failure instead of being masked by a stale copy.
"""

from __future__ import annotations

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


#: Structures that ship with the repository (see data/structures/README.md); the
#: rest are read from an AtomWorks checkout beside it.
VENDORED_DIR = Path(__file__).parent / "data" / "structures"
VENDORED: dict[str, str] = {
    "lysozyme": "6lyz.bcif",
    "hemoglobin": "2hhb.cif.gz",
    "zinc": "4q8n.cif.gz",
    "modified": "1a8o.cif",
}


def locate_fixture(name: str) -> Path | None:
    """The file for fixture *name*, or ``None`` when it is not available."""
    if name in VENDORED:
        return VENDORED_DIR / VENDORED[name]
    path = _atomworks_data_dir() / FIXTURES[name]
    return path if path.exists() else None


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
    load_ccd(paths.ccd_dir())


@pytest.fixture(scope="session")
def parsed():
    """``name -> (atom_array, chain_info)``, parsed once and memoized.

    Skips rather than errors when AtomWorks is absent or the file is not
    available, so that a test requesting this fixture behind a `gpu`/weights
    guard reports "skipped" instead of a collection error in an environment that
    was never expected to have them.
    """
    parse = pytest.importorskip("atomworks.io").parse

    cache: dict[str, tuple] = {}

    def _get(name: str):
        if name not in cache:
            path = locate_fixture(name)
            if path is None:
                pytest.skip(
                    f"fixture {name} is not available (needs an AtomWorks checkout)"
                )
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
