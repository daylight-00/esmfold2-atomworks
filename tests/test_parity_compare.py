"""The comparators decide every parity claim, so what they refuse is pinned here.

Both need nothing but numpy, so these run in the offline job.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from esmfold2_atomworks.parity.compare import (
    MODEL_CONSUMED_FEATURES,
    TRAINING_ONLY_FEATURES,
    compare_features,
    compare_results,
)

pytestmark = pytest.mark.offline


def _features(**replace):
    features = {
        key: np.zeros((2, 3), dtype=np.float32)
        for key in sorted(MODEL_CONSUMED_FEATURES | TRAINING_ONLY_FEATURES)
    }
    features.update(replace)
    return features


def test_the_two_feature_sets_partition_the_featurizer_output():
    assert not MODEL_CONSUMED_FEATURES & TRAINING_ONLY_FEATURES
    assert len(MODEL_CONSUMED_FEATURES) == 25
    assert len(TRAINING_ONLY_FEATURES) == 4


def test_identical_features_are_identical():
    diff = compare_features(_features(), _features())
    assert diff.ok and diff.identical
    assert len(diff.compared) == 29


@pytest.mark.parametrize("name", ["disto_cond", "disto_cond_mask", "res_type"])
def test_a_difference_in_a_tensor_the_model_reads_fails(name):
    changed = _features(**{name: np.ones((2, 3), dtype=np.float32)})
    diff = compare_features(_features(), changed)
    assert not diff.ok and not diff.identical
    assert diff.consumed_mismatches == [name]


@pytest.mark.parametrize("name", sorted(TRAINING_ONLY_FEATURES))
def test_a_difference_in_a_discarded_tensor_is_reported_without_failing(name):
    changed = _features(**{name: np.ones((2, 3), dtype=np.float32)})
    diff = compare_features(_features(), changed)
    assert diff.ok and not diff.identical
    assert name in diff.report()


def test_a_tensor_that_cannot_be_compared_is_not_a_pass():
    diff = compare_features(_features(), _features(msa=None))
    assert not diff.ok and not diff.identical
    assert diff.uncomparable == ["msa"]


def _result(positions, names=("N", "CA", "C"), plddt=0.5):
    n = len(positions)
    return SimpleNamespace(
        complex=SimpleNamespace(
            atom_positions=np.asarray(positions, dtype=np.float32),
            atom_names=np.asarray(names[:n]),
        ),
        ptm=0.5,
        iptm=0.4,
        plddt=np.full(n, plddt),
        distogram=np.zeros((n, n, 4)),
        pae=np.zeros((n, n)),
    )


_XYZ = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 1.4, 0.0]]


def test_identical_results_compare_on_every_quantity():
    diff = compare_results(_result(_XYZ), _result(_XYZ))
    assert diff.ok
    assert {
        "coord_max_abs",
        "coord_rmsd",
        "atom_name_mismatches",
        "plddt_max_abs",
        "distogram_max_abs",
        "pae_max_abs",
        "ptm",
        "iptm",
    } <= set(diff.metrics)
    assert all(value == 0.0 for value in diff.metrics.values())


def test_a_result_with_no_coordinates_is_not_a_pass():
    missing = SimpleNamespace(complex=SimpleNamespace(atom_positions=None))
    diff = compare_results(_result(_XYZ), missing)
    assert not diff.ok
    assert "no coordinates" in " ".join(diff.notes)


def test_a_diff_that_compared_nothing_is_not_a_pass():
    from esmfold2_atomworks.parity.compare import ResultDiff

    assert not ResultDiff().ok


def test_a_different_atom_count_fails():
    diff = compare_results(_result(_XYZ), _result(_XYZ[:2]))
    assert not diff.ok
    assert diff.metrics["atom_count_delta"] == 1.0


def test_different_atom_names_fail_before_any_coordinate_is_read():
    diff = compare_results(_result(_XYZ), _result(_XYZ, names=("N", "CB", "C")))
    assert not diff.ok
    assert diff.metrics["atom_name_mismatches"] == 1.0


def test_a_coordinate_shift_beyond_the_tolerance_fails():
    shifted = [[x + 0.01 for x in row] for row in _XYZ]
    diff = compare_results(_result(_XYZ), _result(shifted))
    assert not diff.ok
    assert diff.metrics["coord_max_abs"] == pytest.approx(0.01, abs=1e-6)
