"""A fold's record says what it was asked, what ran, and on what.

``fold(record=...)`` is the call-level half of provenance; ``provenance()`` the
model-level half. Both are checked here against fakes of the native module and
builder, whose ``fold`` signature stands in for upstream's, so none of it needs
weights. The digest tests use esm's own input types.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from esmfold2_atomworks.model.esmfold2 import (
    AtomWorksESMFold2,
    FoldingConfig,
    input_digests,
)

pytest.importorskip("torch")


class _Builder:
    """Upstream's ``fold`` signature, doing nothing."""

    def fold(
        self,
        model,
        input,
        *,
        num_loops=20,
        num_sampling_steps=200,
        num_diffusion_samples=1,
        seed=None,
        lm_mask_pct=None,
        lm_dropout=0.3,
        msa_max_depth=1024,
    ):
        return "result"


def _model(**config) -> AtomWorksESMFold2:
    model = AtomWorksESMFold2.__new__(AtomWorksESMFold2)
    config = {"type": "release", "num_loops": 3, "lm_mask_pct": 0.1, **config}
    model.net = SimpleNamespace(esmc=object(), config=SimpleNamespace(**config))
    model.builder = _Builder()
    return model


def _spi():
    return SimpleNamespace(sequences=[])


def test_the_record_holds_every_argument_the_fold_ran_with():
    record: dict = {}
    _model().fold(
        _spi(), config=FoldingConfig(num_sampling_steps=8, seed=7), record=record
    )
    # Requested...
    assert record["esmfold2.fold.num_sampling_steps"] == 8
    assert record["esmfold2.fold.seed"] == 7
    # ...and upstream's own default for what was not.
    assert record["esmfold2.fold.msa_max_depth"] == 1024
    assert "esmfold2.fold.model" not in record
    assert "esmfold2.fold.input" not in record


def test_a_schedule_left_to_the_checkpoint_is_resolved():
    record: dict = {}
    _model(num_loops=3).fold(_spi(), record=record)
    assert record["esmfold2.fold.num_loops"] is None
    assert record["esmfold2.effective.num_loops"] == 3

    record = {}
    _model(num_loops=3).fold(_spi(), config=FoldingConfig(num_loops=5), record=record)
    assert record["esmfold2.effective.num_loops"] == 5


def test_the_mask_fraction_is_the_checkpoints_when_the_backbone_runs():
    record: dict = {}
    _model(lm_mask_pct=0.1).fold(_spi(), record=record)
    assert record["esmfold2.effective.lm_mask_pct"] == 0.1


def test_supplied_states_apply_no_mask_at_the_fold():
    entries = _model(lm_mask_pct=0.1)._call_record(_spi(), {}, "caller-supplied")
    assert entries["esmfold2.lm_source"] == "caller-supplied"
    assert entries["esmfold2.effective.lm_mask_pct"] is None


def test_one_record_describes_every_sample_and_names_none():
    record: dict = {}
    _model().fold(_spi(), config=FoldingConfig(num_diffusion_samples=4), record=record)
    assert record["esmfold2.fold.num_diffusion_samples"] == 4
    assert not any("sample_index" in key for key in record)


def test_fold_atom_array_records_where_each_sequence_came_from(monkeypatch):
    import esmfold2_atomworks.data.atomworks_to_esm as adapter
    import esmfold2_atomworks.data.molecular_complex as reverse

    def convert(atoms, *, chain_info=None, report=None, **kwargs):
        report.sequence_source["A"] = "override"
        return _spi()

    monkeypatch.setattr(adapter, "atom_array_to_structure_prediction_input", convert)
    monkeypatch.setattr(reverse, "result_to_atom_array", lambda result, **_: "atoms")

    record: dict = {"caller.run": "r1"}
    atoms, result = _model().fold_atom_array(object(), record=record)
    assert (atoms, result) == ("atoms", "result")
    assert record["esmfold2.sequence_source"] == {"A": "override"}
    assert record["caller.run"] == "r1"


# -- digests -----------------------------------------------------------------


def _types():
    return pytest.importorskip("esm.utils.structure.input_builder")


def test_the_digest_names_the_chemistry_not_the_chain():
    t = _types()
    a = t.StructurePredictionInput(sequences=[t.ProteinInput(id="A", sequence="MKV")])
    b = t.StructurePredictionInput(sequences=[t.ProteinInput(id="Z", sequence="MKV")])
    c = t.StructurePredictionInput(sequences=[t.ProteinInput(id="A", sequence="MKA")])
    (da,), (db,), (dc,) = input_digests(a), input_digests(b), input_digests(c)
    assert da["sha256"] == db["sha256"] != dc["sha256"]
    assert da["ids"] == ["A"] and db["ids"] == ["Z"]
    assert da["kind"] == "protein" and da["length"] == 3 and da["msa"] is False


def test_a_modification_changes_the_digest():
    t = _types()
    plain = t.ProteinInput(id="A", sequence="MKV")
    modified = t.ProteinInput(
        id="A", sequence="MKV", modifications=[t.Modification(position=0, ccd="MSE")]
    )
    (dp,) = input_digests(t.StructurePredictionInput(sequences=[plain]))
    (dm,) = input_digests(t.StructurePredictionInput(sequences=[modified]))
    assert dp["sha256"] != dm["sha256"]


def test_a_ligand_is_named_by_its_ccd_codes():
    t = _types()
    hem = t.LigandInput(id=["C", "D"], ccd=["HEM"])
    zn = t.LigandInput(id="C", ccd=["ZN"])
    (dh,) = input_digests(t.StructurePredictionInput(sequences=[hem]))
    (dz,) = input_digests(t.StructurePredictionInput(sequences=[zn]))
    assert dh["kind"] == "ligand" and dh["length"] is None
    assert dh["ids"] == ["C", "D"]
    assert dh["sha256"] != dz["sha256"]


# -- model provenance --------------------------------------------------------


def _loaded(weights: Path, esmc_source: str = "bundled") -> AtomWorksESMFold2:
    import torch

    model = AtomWorksESMFold2.__new__(AtomWorksESMFold2)
    model.net = SimpleNamespace(config=SimpleNamespace(type="release"))
    model.weights = str(weights)
    model.device = torch.device("cpu")
    model.flavour = "esm"
    model.esmc_source = esmc_source
    model.ccd_source = "ccd.pkl"
    return model


def _snapshot(root: Path, repo: str, revision: str) -> Path:
    org, name = repo.split("/")
    path = root / f"models--{org}--{name}" / "snapshots" / revision
    path.mkdir(parents=True)
    return path


def test_provenance_names_the_checkpoint_by_repo_and_revision(tmp_path):
    snapshot = _snapshot(tmp_path, "biohub/ESMFold2", "69869f73")
    pin = tmp_path / "ESMFold2"
    pin.symlink_to(snapshot)
    record = _loaded(pin).provenance()
    assert record["esmfold2.checkpoint.repo"] == "biohub/ESMFold2"
    assert record["esmfold2.checkpoint.revision"] == "69869f73"
    assert record["esmfold2.device"] == "cpu"
    assert record["esmfold2.torch"]
    assert all(isinstance(value, str) for value in record.values())
    assert "esmfold2.esmc.revision" not in record


def test_provenance_leaves_an_unknown_revision_empty(tmp_path):
    record = _loaded(tmp_path).provenance()
    assert record["esmfold2.checkpoint.revision"] == ""


def test_provenance_names_a_separate_backbone(tmp_path):
    trunk = _snapshot(tmp_path, "biohub/ESMFold2-Experimental", "61717221")
    esmc = _snapshot(tmp_path, "biohub/ESMC-6B", "89c554c4")
    record = _loaded(trunk, esmc_source=str(esmc)).provenance()
    assert record["esmfold2.esmc.repo"] == "biohub/ESMC-6B"
    assert record["esmfold2.esmc.revision"] == "89c554c4"


# -- the engine --------------------------------------------------------------


class _EngineModel:
    """Answers fold_atom_array with ``samples`` structures and fills the record."""

    def __init__(self, samples: int) -> None:
        self.samples = samples

    def provenance(self):
        return {"esmfold2.checkpoint.revision": "69869f73"}

    def fold_atom_array(self, atoms, *, record, **kwargs):
        record["esmfold2.fold.seed"] = 0
        if self.samples == 1:
            return "atoms", "result"
        return ["atoms"] * self.samples, ["result"] * self.samples


@pytest.mark.parametrize("samples", [1, 3])
def test_every_engine_output_describes_itself(monkeypatch, samples):
    from esmfold2_atomworks import metrics
    from esmfold2_atomworks.inference.engine import ESMFold2InferenceEngine

    monkeypatch.setattr(metrics, "fold_metrics", lambda result: {"esm.ptm": 0.5})
    engine = ESMFold2InferenceEngine()
    engine._model = _EngineModel(samples)

    outputs = engine.run({"x": object()})
    assert len(outputs) == samples
    for index, output in enumerate(outputs):
        assert output.metadata["esm.ptm"] == 0.5
        assert output.metadata["esmfold2.checkpoint.revision"] == "69869f73"
        assert output.metadata["esmfold2.fold.seed"] == 0
        # The sample index is the output's own, never the call record's.
        assert output.metadata.get("sample") == (index if samples > 1 else None)
