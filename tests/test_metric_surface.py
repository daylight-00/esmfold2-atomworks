"""The metric surface a consumer can rely on without copying anything.

The scalar metric table, and the two axes pLDDT comes in. Both need nothing but
numpy, so these run in the offline job.
"""

from __future__ import annotations

import numpy as np
import pytest

from esmfold2_atomworks.metrics import (
    SCALAR_METRIC_SOURCES,
    fold_metrics,
    plddt_per_residue,
    plddt_per_token,
)

pytestmark = pytest.mark.offline


class _Complex:
    def __init__(self, plddt):
        self.plddt = np.asarray(plddt, dtype=float)


class _Result:
    """Token space and residue space, at the lengths a ligand gives them."""

    def __init__(self, n_tokens=131, n_residues=113):
        self.plddt = np.linspace(0.1, 0.9, n_tokens)
        self.complex = _Complex(np.linspace(0.3, 0.6, n_residues))
        self.ptm = np.array(0.75)
        self.iptm = np.array(0.61)


def test_the_scalar_table_is_public_and_namespaced():
    assert SCALAR_METRIC_SOURCES
    assert all(key.startswith("esm.") for key in SCALAR_METRIC_SOURCES)
    # Scalar, not "all metrics": the conditional ones are not read off a field.
    assert "esm.interface_pae" not in SCALAR_METRIC_SOURCES
    assert "esm.ligand_plddt" not in SCALAR_METRIC_SOURCES


def test_the_scalar_table_cannot_be_edited_in_place():
    """fold_metrics reads it, so an edit would change what it emits for everyone."""
    with pytest.raises(TypeError):
        SCALAR_METRIC_SOURCES["esm.extra"] = "extra"  # type: ignore[index]

    assert "esm.extra" not in SCALAR_METRIC_SOURCES


def test_every_scalar_in_the_table_is_emitted_when_present():
    metrics = fold_metrics(_Result())

    assert set(SCALAR_METRIC_SOURCES) <= set(metrics)


def test_mean_plddt_is_a_token_space_mean():
    result = _Result()

    mean = fold_metrics(result)["esm.mean_plddt"]

    assert mean == pytest.approx(result.plddt.mean())
    assert mean != pytest.approx(result.complex.plddt.mean())


def test_the_two_plddt_axes_are_different_arrays():
    """On a plain monomer they coincide, which is what hides reading the wrong
    one until a ligand appears."""
    result = _Result(n_tokens=131, n_residues=113)

    np.testing.assert_array_equal(plddt_per_token(result), result.plddt)
    np.testing.assert_array_equal(plddt_per_residue(result), result.complex.plddt)
    assert plddt_per_token(result).size == 131
    assert plddt_per_residue(result).size == 113


def test_an_axis_the_result_does_not_carry_is_absent_rather_than_fabricated():
    class Empty:
        pass

    assert plddt_per_token(Empty()) is None
    assert plddt_per_residue(Empty()) is None
