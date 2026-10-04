"""Training steps through the Foundry integration, on a GPU.

``scripts/smoke_training_step.py`` does the work in its own process: the
experimental checkpoint, Foundry's ``fit`` loop, a distogram objective against the
structure's own coordinates. It needs a GPU, Foundry and the experimental
checkpoint (``EF_EXPERIMENTAL_WEIGHTS``, or the mirror).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.gpu
def test_the_trainer_takes_steps_that_lower_the_loss_and_change_the_weights():
    torch = pytest.importorskip("torch")
    pytest.importorskip("foundry")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    weights = os.environ.get("EF_EXPERIMENTAL_WEIGHTS")
    if not weights:
        pytest.skip("set EF_EXPERIMENTAL_WEIGHTS to the experimental checkpoint")

    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "smoke_training_step.py"),
            "--structure",
            str(ROOT / "tests" / "data" / "structures" / "1a8o.cif"),
            "--weights",
            weights,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert done.stdout.rstrip().endswith("OK")
