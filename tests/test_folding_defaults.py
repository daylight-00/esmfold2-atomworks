"""The recycle count is the loaded checkpoint's unless the caller sets one.

Checkpoints disagree about ``num_loops`` -- 3 in the reference checkpoint, 20
in the one ``biohub/ESMFold2`` publishes today -- so a fixed default overrides
one of them silently. ``None`` reaches the model, which reads
``config.num_loops``; the key must be present, because
``ESMFold2InputBuilder.fold`` substitutes its own 20 for a missing one.
"""

from __future__ import annotations

import pytest

from esmfold2_atomworks.model.esmfold2 import FoldingConfig

pytestmark = pytest.mark.offline


def test_num_loops_defers_to_the_checkpoint():
    kwargs = FoldingConfig().as_fold_kwargs()
    assert "num_loops" in kwargs
    assert kwargs["num_loops"] is None


def test_a_pinned_count_is_passed_through():
    assert FoldingConfig(num_loops=3).as_fold_kwargs()["num_loops"] == 3
