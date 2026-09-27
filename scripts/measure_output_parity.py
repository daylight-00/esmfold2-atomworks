"""Measure output parity and the seeded scatter, with and without deterministic kernels.

The GPU test suite gates exact output parity on short folds; this records the
fuller measurement docs/02 reports -- the checkpoint's own schedule as well as
the short one, and the scatter the default kernels leave -- as a frozen record
beside the environment it was produced in::

    CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/measure_output_parity.py \\
        --json reproducibility/output_parity.json

For every case the frozen native input (``tests/data/gold``) is folded twice
and the adapted input once, in each execution mode. ``repeat`` compares the two
native folds, ``cross`` the first native fold with the adapted one; wall time
is per fold. Needs a GPU, the weights and AtomWorks' test structures, and runs
in the reference environment. cuBLAS reads ``CUBLAS_WORKSPACE_CONFIG`` when
CUDA starts, so it has to be set before Python is; the script refuses to run
without it rather than record a deterministic mode that was not.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "scripts"))

#: (fixture, schedule) pairs measured. "short" is what the GPU suite gates on;
#: "checkpoint" leaves num_loops to the checkpoint and samples 100 steps.
CASES = [
    ("lysozyme", "short"),
    ("lysozyme", "checkpoint"),
    ("hemoglobin", "short"),
    ("hemoglobin", "checkpoint"),
    ("modified", "short"),
    ("zinc", "short"),
    ("flavoprotein", "short"),
]


def _config(schedule: str) -> Any:
    from esmfold2_atomworks.model.esmfold2 import FoldingConfig

    if schedule == "short":
        return FoldingConfig(num_loops=1, num_sampling_steps=8, seed=0)
    return FoldingConfig(num_sampling_steps=100, seed=0)


def _timed(model: Any, spi: Any, config: Any) -> tuple[Any, float]:
    import torch

    torch.cuda.synchronize()
    start = time.perf_counter()
    result = model.fold(spi, config=config)
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def _located(value: str, identity: Any) -> Any:
    """A provenance value, with a local path replaced by what it identifies.

    A path names a directory on one machine; the record names the Hub repo and
    revision it holds (and the file within it), as the checkpoint entry does.
    """
    path = Path(value)
    if not path.is_absolute() or not path.exists():
        return value
    if path.is_file():
        return {**identity(path.parent), "file": path.name}
    return identity(path)


def _commit(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    if not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        print(
            "set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python",
            file=sys.stderr,
        )
        return 2

    import torch
    from atomworks.io import parse
    from compare_checkpoints import identity
    from conftest import FIXTURES, _atomworks_data_dir, gold_structure_prediction_input

    from esmfold2_atomworks import paths
    from esmfold2_atomworks.data.atomworks_to_esm import (
        atom_array_to_structure_prediction_input,
    )
    from esmfold2_atomworks.model.esmfold2 import AtomWorksESMFold2
    from esmfold2_atomworks.parity.compare import compare_results

    model = AtomWorksESMFold2()
    record: dict[str, Any] = {
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "checkpoint": identity(Path(model.weights)),
        "checkpoint_num_loops": int(model.config.num_loops),
        "provenance": {
            key: _located(value, identity)
            for key, value in model.provenance().items()
            if key != "esmfold2.weights"
        },
        "device": torch.cuda.get_device_name(model.device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "esmfold2_atomworks_commit": _commit(REPO),
        "upstream_lock": paths.read_upstream_lock(paths.UPSTREAM_LOCK),
        "seed": 0,
        "cases": [],
    }

    previous = torch.are_deterministic_algorithms_enabled()
    try:
        for fixture, schedule in CASES:
            parsed = parse(_atomworks_data_dir() / FIXTURES[fixture])
            adapted = atom_array_to_structure_prediction_input(
                parsed["asym_unit"][0], chain_info=parsed["chain_info"]
            )
            native = gold_structure_prediction_input(fixture)
            config = _config(schedule)
            for deterministic in (True, False):
                torch.use_deterministic_algorithms(deterministic)
                a, t_a = _timed(model, native, config)
                b, t_b = _timed(model, native, config)
                c, t_c = _timed(model, adapted, config)
                case = {
                    "fixture": fixture,
                    "schedule": schedule,
                    "num_loops": config.num_loops,
                    "num_sampling_steps": config.num_sampling_steps,
                    "deterministic": deterministic,
                    "repeat": compare_results(a, b).metrics,
                    "cross": compare_results(a, c).metrics,
                    "seconds_per_fold": [round(t, 2) for t in (t_a, t_b, t_c)],
                }
                print(json.dumps(case), flush=True)
                record["cases"].append(case)
    finally:
        torch.use_deterministic_algorithms(previous)

    args.json.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
