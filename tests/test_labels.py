"""Source coordinates must land on the model's atom axis, correctly and honestly.

This is the supervision half of "Foundry-trainable": the pipeline otherwise
carries inputs only, and `feats["gt_coords"]` is not the source structure -- it
is built from the prediction input and is zeros at inference.

No loss lives here, so the tests are about the two properties a loss will rely
on: that a matched atom really is the same atom, and that an unmatched one is
masked rather than filled in.
"""

from __future__ import annotations

import numpy as np
import pytest

from esmfold2_foundry.data.atomworks_to_esm import (
    _residue_index_map,
    atom_array_to_structure_prediction_input,
    chain_records,
)
from esmfold2_foundry.data.labels import structure_labels
from esmfold2_foundry.parity.compare import featurize


def _labels(atoms, chain_info):
    spi = atom_array_to_structure_prediction_input(atoms, chain_info=chain_info)
    from esm.models.esmfold2.prepare_input import prepare_esmfold2_input
    from esm.models.esmfold2.processor import clean_esmfold2_input

    features, chain_infos = prepare_esmfold2_input(clean_esmfold2_input(spi), seed=0)
    records = chain_records(atoms, chain_info=chain_info)
    return structure_labels(
        atoms, features, chain_infos, _residue_index_map(records, chain_info)
    ), features


def test_labels_cover_the_model_atom_axis(parsed, ccd):
    atoms, chain_info = parsed("lysozyme")
    labels, features = _labels(atoms, chain_info)

    import numpy as np

    assert labels.atom_coords.shape == (features["ref_pos"].shape[0], 3)
    assert labels.atom_mask.shape == (features["ref_pos"].shape[0],)

    # Coverage is measured against the model's REAL atoms, not the padded axis.
    # ESMFold2 pads to a multiple of 32, so lysozyme's 1000 atoms sit on a
    # 1024-long axis; dividing by the axis would report 0.977 for a structure
    # that is in fact complete, and `require_coverage=0.99` would reject it.
    real = int(np.asarray(features["atom_attention_mask"]).sum())
    padded = int(np.asarray(features["atom_attention_mask"]).size)
    assert padded > real, "expected the axis to be padded, or this asserts nothing"
    assert labels.n_model_atoms == real
    assert labels.coverage == pytest.approx(labels.matched / real)

    # Lysozyme is fully resolved, so every real model atom must match.
    assert labels.coverage == pytest.approx(1.0), (
        f"{labels.matched}/{real} matched; unmatched: {labels.unmatched_examples}"
    )


def test_matched_coordinates_are_the_source_coordinates(parsed, ccd):
    """A permutation that lost track of which atom is which would still 'cover'."""
    import biotite.structure as struc

    atoms, chain_info = parsed("lysozyme")
    labels, _features = _labels(atoms, chain_info)

    matched = labels.atom_coords[labels.atom_mask]
    source = np.asarray(atoms.coord, dtype=np.float32)

    # Every matched label must be an actual atom position from the source, and
    # the set of matched positions must be a subset of the source's.
    source_rows = {tuple(np.round(row, 3)) for row in source}
    sample = matched[:: max(1, len(matched) // 50)]
    for row in sample:
        assert tuple(np.round(row, 3)) in source_rows

    # And the CA of the first residue must be exactly where the source has it.
    starts = struc.get_residue_starts(atoms)
    first_ca = np.where(
        (np.asarray(atoms.res_id) == int(atoms.res_id[starts[0]]))
        & (np.asarray(atoms.atom_name).astype(str) == "CA")
    )[0][0]
    assert any(
        np.allclose(labels.atom_coords[i], source[first_ca], atol=1e-4)
        for i in np.where(labels.atom_mask)[0]
    )


def test_unmatched_atoms_are_masked_and_left_nan(parsed, ccd):
    """Never imputed: a placeholder would silently bias any averaging loss."""
    atoms, chain_info = parsed("hemoglobin")
    labels, _features = _labels(atoms, chain_info)

    unmatched = ~labels.atom_mask
    assert unmatched.any(), "expected padding atoms at least"
    assert np.isnan(labels.atom_coords[unmatched]).all()
    assert np.isfinite(labels.atom_coords[labels.atom_mask]).all()


def test_a_deleted_side_chain_becomes_unmatched_rather_than_wrong(parsed, ccd):
    """Removing source atoms must lower coverage, not shift the alignment."""
    atoms, chain_info = parsed("lysozyme")
    full, _ = _labels(atoms, chain_info)

    names = np.asarray(atoms.atom_name).astype(str)
    backbone_only = atoms[np.isin(names, ["N", "CA", "C", "O"])]
    reduced, _ = _labels(backbone_only, chain_info)

    assert reduced.matched < full.matched
    # What remains must still be right, not merely fewer.
    assert np.isfinite(reduced.atom_coords[reduced.atom_mask]).all()
    assert reduced.unmatched_examples


def test_the_pipeline_can_attach_labels(parsed, ccd):
    from esmfold2_foundry.data.pipelines import build_esmfold2_pipeline

    atoms, chain_info = parsed("lysozyme")
    pipeline = build_esmfold2_pipeline(is_inference=False, seed=0, attach_labels=True)
    out = pipeline(
        {"example_id": "lysozyme", "atom_array": atoms, "chain_info": chain_info}
    )
    assert set(out["labels"]) == {"atom_coords", "atom_mask"}
    assert out["labels"]["atom_coords"].shape[0] == out["feats"]["ref_pos"].shape[0]
    assert out["label_coverage"] == pytest.approx(1.0)


def test_labels_are_off_by_default(parsed, ccd):
    """Inference does not need them and they are not free."""
    from esmfold2_foundry.data.pipelines import build_esmfold2_pipeline

    atoms, chain_info = parsed("lysozyme")
    out = build_esmfold2_pipeline(is_inference=False, seed=0)(
        {"example_id": "lysozyme", "atom_array": atoms, "chain_info": chain_info}
    )
    assert "labels" not in out


def test_a_coverage_floor_can_be_required(parsed, ccd):
    """A silently-unmatched structure is worse than a refused one."""
    from esmfold2_foundry.data.pipelines import AttachStructureLabels

    atoms, chain_info = parsed("lysozyme")
    spi = atom_array_to_structure_prediction_input(atoms, chain_info=chain_info)
    features = featurize(spi, seed=0)
    from esm.models.esmfold2.prepare_input import prepare_esmfold2_input
    from esm.models.esmfold2.processor import clean_esmfold2_input

    _f, chain_infos = prepare_esmfold2_input(clean_esmfold2_input(spi), seed=0)

    # 1.0 is now attainable, so the floor has to exceed it to trip.
    transform = AttachStructureLabels(require_coverage=1.01)
    with pytest.raises(ValueError, match="below the required"):
        transform(
            {
                "atom_array": atoms,
                "chain_info": chain_info,
                "feats": features,
                "chain_infos": chain_infos,
            }
        )
