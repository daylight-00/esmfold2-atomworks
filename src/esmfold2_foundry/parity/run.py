"""Run feature parity over structure files and report per structure.

This is the batch form of what ``tests/test_feature_parity.py`` asserts. It is
separate from the tests because the useful thing to do with a new corpus --
point it at a directory of PDB entries and see which ones the adapter cannot yet
reproduce -- is a survey, not a pass/fail gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["ParityOutcome", "parity_for_structure", "run_feature_parity"]


class ParityOutcome:
    """The result of checking one structure."""

    def __init__(self, path: Path, ok: bool, detail: str) -> None:
        self.path = path
        self.ok = ok
        self.detail = detail

    def __str__(self) -> str:
        return f"[{'ok ' if self.ok else 'FAIL'}] {self.path.name}: {self.detail}"


def _native_counterpart(spi: Any, chain_info: dict) -> Any:
    """Rebuild *spi* with sequences re-read from ``chain_info``.

    **This is a self-consistency check, not an independent one**, and the
    difference matters. Chain ids, ligand CCD codes and modifications are taken
    from *spi* -- the adapter's own output -- so a systematic error (a ligand
    mapped to the wrong CCD code, say) would be copied into both sides and
    cancel. Only the sequences come from an independent source.

    That is acceptable here because this function backs the corpus *survey*,
    whose job is to find structures the adapter cannot process at all. The
    assertion suite does not use it: ``tests/data/gold/*.json`` holds frozen
    inputs generated from AtomWorks' own parse output and anchored to facts
    about the entries by ``tests/test_gold_fixtures.py``. Use those when adding
    a case that is meant to prove correctness rather than survey coverage.
    """
    from esm.models.esmfold2.types import (
        DNAInput,
        LigandInput,
        ProteinInput,
        RNAInput,
        StructurePredictionInput,
    )

    def canonical(chain_id: str) -> str | None:
        for key, value in chain_info.items():
            if str(key) == str(chain_id):
                return value.get("processed_entity_canonical_sequence")
        return None

    entries: list[Any] = []
    for entry in spi.sequences:
        if isinstance(entry, (ProteinInput, DNAInput, RNAInput)):
            sequence = canonical(entry.id) or entry.sequence
            entries.append(
                type(entry)(
                    id=entry.id,
                    sequence=sequence,
                    modifications=entry.modifications,
                )
            )
        else:
            entries.append(LigandInput(id=entry.id, smiles=entry.smiles, ccd=entry.ccd))
    return StructurePredictionInput(sequences=entries)


def parity_for_structure(path: Path, *, seed: int = 0) -> ParityOutcome:
    """Check one structure, converting any failure into an outcome rather than a raise."""
    from atomworks.io import parse

    from esmfold2_foundry.data.atomworks_to_esm import (
        atom_array_to_structure_prediction_input,
    )
    from esmfold2_foundry.parity.compare import compare_features, featurize

    try:
        parsed = parse(path)
        atoms, chain_info = parsed["asym_unit"][0], parsed["chain_info"]
        adapted = atom_array_to_structure_prediction_input(atoms, chain_info=chain_info)
        native = _native_counterpart(adapted, chain_info)
        diff = compare_features(
            featurize(native, seed=seed), featurize(adapted, seed=seed)
        )
    except Exception as error:  # noqa: BLE001 - a survey reports, it does not abort
        return ParityOutcome(path, False, f"{type(error).__name__}: {error}")
    return ParityOutcome(path, diff.ok, diff.report().splitlines()[0])


def run_feature_parity(
    structures: list[Path] | list[str],
    *,
    seed: int = 0,
    verbose: bool = False,
) -> list[ParityOutcome]:
    """Check each structure; return the failures.

    Loads the CCD dictionary once up front -- it costs about nine seconds and is
    process-global, so paying for it per structure would dominate a survey.
    """
    from esm.models.esmfold2.conformers import load_ccd

    load_ccd()

    failures: list[ParityOutcome] = []
    for item in structures:
        outcome = parity_for_structure(Path(item), seed=seed)
        if verbose or not outcome.ok:
            print(outcome)
        if not outcome.ok:
            failures.append(outcome)

    total = len(structures)
    print(f"\nfeature parity: {total - len(failures)}/{total} structures reproduce")
    return failures
