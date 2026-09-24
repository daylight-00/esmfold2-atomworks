"""The AtomWorks pipeline emits the same features the adapter does."""

from __future__ import annotations

import pytest

from esmfold2_atomworks.data.atomworks_to_esm import (
    atom_array_to_structure_prediction_input,
)
from esmfold2_atomworks.parity.compare import compare_features, featurize

pytest.importorskip("atomworks.ml.transforms.base")


def _example(parsed, name: str) -> dict:
    atoms, chain_info = parsed(name)
    return {"example_id": name, "atom_array": atoms, "chain_info": chain_info}


def test_pipeline_produces_features_and_decode_metadata(parsed, ccd):
    from esmfold2_atomworks.data.pipelines import build_esmfold2_pipeline

    pipeline = build_esmfold2_pipeline(is_inference=True, seed=0)
    out = pipeline(_example(parsed, "hemoglobin"))

    assert set(out) >= {"example_id", "feats", "chain_infos"}
    assert len(out["feats"]) == 29
    # chain_infos is the only token -> (chain, residue, atom span) record, so
    # losing it makes a prediction impossible to decode back to a structure.
    assert len(out["chain_infos"]) == 8


def test_pipeline_matches_the_direct_adapter(parsed, ccd):
    """Going through Compose must not change the featurization."""
    from esmfold2_atomworks.data.pipelines import build_esmfold2_pipeline

    atoms, chain_info = parsed("hemoglobin")
    direct = featurize(
        atom_array_to_structure_prediction_input(atoms, chain_info=chain_info), seed=0
    )
    through_pipeline = build_esmfold2_pipeline(is_inference=True, seed=0)(
        _example(parsed, "hemoglobin")
    )["feats"]

    diff = compare_features(direct, through_pipeline)
    assert diff.identical, diff.report()


def test_inference_pipeline_keeps_the_input_structure(parsed, ccd):
    from esmfold2_atomworks.data.pipelines import build_esmfold2_pipeline

    out = build_esmfold2_pipeline(is_inference=True, seed=0)(
        _example(parsed, "lysozyme")
    )
    assert "atom_array" in out
    assert out["adapter_report"].dropped == []


def test_training_pipeline_drops_the_structure(parsed, ccd):
    from esmfold2_atomworks.data.pipelines import build_esmfold2_pipeline

    out = build_esmfold2_pipeline(is_inference=False, seed=0)(
        _example(parsed, "lysozyme")
    )
    assert "atom_array" not in out
    assert "feats" in out
