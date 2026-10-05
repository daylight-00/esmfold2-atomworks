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
    input_record,
    tensor_record,
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
        noise_scale=None,
        step_scale=None,
        max_inference_sigma=None,
    ):
        return "result"


def _model(**config) -> AtomWorksESMFold2:
    model = AtomWorksESMFold2.__new__(AtomWorksESMFold2)
    config = {"type": "release", "num_loops": 3, "lm_mask_pct": 0.1, **config}
    model.net = SimpleNamespace(esmc=object(), config=SimpleNamespace(**config))
    model.builder = _Builder()
    return model


def _spi():
    return SimpleNamespace(
        sequences=[], pocket=None, distogram_conditioning=None, covalent_bonds=None
    )


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
    from esmfold2_atomworks.data import topology

    def convert(atoms, *, chain_info=None, report=None, **kwargs):
        report.sequence_source["A"] = "override"
        return _spi()

    monkeypatch.setattr(adapter, "atom_array_to_structure_prediction_input", convert)
    monkeypatch.setattr(reverse, "result_to_atom_array", lambda result, **_: "atoms")
    monkeypatch.setattr(topology, "ccd_name_collisions", lambda atoms, spi: ["LIG"])

    import biotite.structure as struc

    record: dict = {"caller.run": "r1"}
    atoms, result = _model().fold_atom_array(
        struc.AtomArray(0), record=record, bonds=False
    )
    assert (atoms, result) == ("atoms", "result")
    assert record["esmfold2.bonds"] is False
    assert record["esmfold2.ccd_name_collisions"] == ["LIG"]
    assert record["esmfold2.sequence_source"] == {"A": "override"}
    assert record["caller.run"] == "r1"


# -- digests -----------------------------------------------------------------


def _types():
    return pytest.importorskip("esm.utils.structure.input_builder")


def _entities(t, *entries, **conditions):
    spi = t.StructurePredictionInput(sequences=list(entries), **conditions)
    return input_record(spi)["esmfold2.inputs"]


def test_the_digest_names_the_chemistry_not_the_chain():
    t = _types()
    (da,) = _entities(t, t.ProteinInput(id="A", sequence="MKV"))
    (db,) = _entities(t, t.ProteinInput(id="Z", sequence="MKV"))
    (dc,) = _entities(t, t.ProteinInput(id="A", sequence="MKA"))
    assert da["chemistry_sha256"] == db["chemistry_sha256"] != dc["chemistry_sha256"]
    assert len(da["chemistry_sha256"]) == 64
    assert da["ids"] == ["A"] and db["ids"] == ["Z"]
    assert da["kind"] == "protein" and da["length"] == 3
    assert da["msa_sha256"] is None


def test_a_modification_changes_the_digest():
    t = _types()
    plain = t.ProteinInput(id="A", sequence="MKV")
    modified = t.ProteinInput(
        id="A", sequence="MKV", modifications=[t.Modification(position=0, ccd="MSE")]
    )
    ((dp,), (dm,)) = _entities(t, plain), _entities(t, modified)
    assert dp["chemistry_sha256"] != dm["chemistry_sha256"]


def test_a_ligand_is_named_by_its_ccd_codes():
    t = _types()
    (dh,) = _entities(t, t.LigandInput(id=["C", "D"], ccd=["HEM"]))
    (dz,) = _entities(t, t.LigandInput(id="C", ccd=["ZN"]))
    assert dh["kind"] == "ligand" and dh["length"] is None
    assert dh["ids"] == ["C", "D"]
    assert dh["chemistry_sha256"] != dz["chemistry_sha256"]


def test_chainbreaks_are_recorded_as_the_entities_actually_folded():
    """Upstream splits "AAA|AAA|BBB" into two entities; so does the record."""
    t = _types()
    entities = _entities(t, t.ProteinInput(id="X", sequence="MKV|MKV|GGA"))
    assert [e["ids"] for e in entities] == [["X_0", "X_1"], ["X_2"]]
    assert [e["length"] for e in entities] == [3, 3]


def _msa(rows, deletions=None):
    msa = pytest.importorskip("esm.utils.msa.msa")
    parsing = pytest.importorskip("esm.utils.parsing")
    entries = [parsing.FastaEntry(header, sequence) for header, sequence in rows]
    return msa.MSA(entries=entries, deletions=deletions)


def test_the_alignment_is_part_of_the_record():
    """Same sequence, another alignment: a different conditioning."""
    import numpy as np

    t = _types()
    a = _msa([("query", "MKV"), ("key=9606", "MRV")])
    b = _msa([("query", "MKV"), ("key=10090", "MRV")])  # only the taxonomy differs
    c = _msa(
        [("query", "MKV"), ("key=9606", "MRV")],
        deletions=np.array([[0, 0, 0], [0, 2, 0]]),
    )
    digests = [
        _entities(t, t.ProteinInput(id="A", sequence="MKV", msa=m))[0]["msa_sha256"]
        for m in (a, b, c)
    ]
    assert all(d is not None and len(d) == 64 for d in digests)
    assert len(set(digests)) == 3


def test_covalent_bonds_are_recorded_in_canonical_order():
    t = _types()
    bonds = [
        t.CovalentBond("B", 0, 1, "A", 5, 2),
        t.CovalentBond("A", 1, 0, "B", 0, 3),
    ]
    record = input_record(
        t.StructurePredictionInput(
            sequences=[t.ProteinInput(id="A", sequence="MKVCAC")], covalent_bonds=bonds
        )
    )
    assert record["esmfold2.covalent_bonds"] == [
        ["A", 1, 0, "B", 0, 3],
        ["B", 0, 1, "A", 5, 2],
    ]
    assert record["esmfold2.pocket_sha256"] is None
    assert record["esmfold2.distogram_conditioning_sha256"] is None


def test_supplied_states_are_named_by_their_content():
    import torch

    a = torch.zeros(1, 4, 3, 5, dtype=torch.bfloat16)
    b = a.clone()
    b[0, 1, 2, 3] = 1.0
    ra, rb = tensor_record(a), tensor_record(b)
    assert ra["sha256"] != rb["sha256"] and len(ra["sha256"]) == 64
    assert ra["shape"] == [1, 4, 3, 5] and ra["dtype"] == "bfloat16"
    # Chunking does not change the digest.
    assert tensor_record(b, chunk_elements=7)["sha256"] == rb["sha256"]


# -- model provenance --------------------------------------------------------


def _loaded(weights: Path, esmc_source: str = "bundled") -> AtomWorksESMFold2:
    import torch

    model = AtomWorksESMFold2.__new__(AtomWorksESMFold2)
    model.net = SimpleNamespace(config=SimpleNamespace(type="release"))
    model.weights = str(weights)
    weights.mkdir(parents=True, exist_ok=True)
    (weights / "config.json").write_text('{"type": "release"}')
    model.device = torch.device("cpu")
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


def test_an_unversioned_directory_is_said_to_be_one(tmp_path):
    """Its name is not an identity; the record says so, with the config digest."""
    record = _loaded(tmp_path / "my-finetune").provenance()
    assert record["esmfold2.checkpoint.revision"] == ""
    assert record["esmfold2.checkpoint.versioning"] == "unversioned"
    assert len(record["esmfold2.checkpoint.config_sha256"]) == 64
    assert "my-finetune" not in record["esmfold2.checkpoint.repo"]


def test_the_device_is_where_the_module_is(tmp_path):
    """Read off the module, not the process's current device."""
    model = _loaded(tmp_path / "w")
    model.net.device = "meta"
    assert model.provenance()["esmfold2.device"] == "meta"


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


# -- effective settings ------------------------------------------------------


class _Forward:
    """A module whose forward declares the sampler knobs, as esm >= 3.4 does."""

    def forward(self, *, noise_scale=None, step_scale=None, max_inference_sigma=256.0):
        return None


def _live(**config):
    model = _model(**config)
    net = _Forward()
    net.esmc = object()
    net.config = model.net.config
    net.structure_head = SimpleNamespace(
        inference_num_steps=200, noise_scale=1.003, step_scale=1.5
    )
    model.net = net
    return model


def test_every_setting_left_to_the_model_is_resolved_against_it():
    model = _live(
        num_diffusion_samples=5,
        msa_encoder=SimpleNamespace(max_depth=512, column_mask_rate=0.2),
        lm_encoder=SimpleNamespace(lm_dropout=0.1, per_loop_lm_dropout=True),
    )
    args = {"num_loops": None, "num_diffusion_samples": None, "lm_dropout": None}
    entries = model._effective_settings(args, "model")
    assert entries["esmfold2.effective.num_loops"] == 3
    assert entries["esmfold2.effective.num_diffusion_samples"] == 5
    assert entries["esmfold2.effective.num_sampling_steps"] == 200
    assert entries["esmfold2.effective.noise_scale"] == 1.003
    assert entries["esmfold2.effective.step_scale"] == 1.5
    assert entries["esmfold2.effective.max_inference_sigma"] == 256.0
    assert entries["esmfold2.effective.msa_max_depth"] == 512
    assert entries["esmfold2.effective.msa_column_mask_rate"] == 0.2
    assert entries["esmfold2.effective.lm_mask_pct"] == 0.1
    assert entries["esmfold2.effective.lm_dropout"] == 0.1


def test_a_requested_setting_is_the_effective_one():
    entries = _live()._effective_settings(
        {"num_sampling_steps": 8, "noise_scale": 1.0, "lm_dropout": 0.3}, "model"
    )
    assert entries["esmfold2.effective.num_sampling_steps"] == 8
    assert entries["esmfold2.effective.noise_scale"] == 1.0
    assert entries["esmfold2.effective.lm_dropout"] == 0.3


def test_release_lm_dropout_applies_only_per_loop():
    model = _live(lm_encoder=SimpleNamespace(lm_dropout=0.2, per_loop_lm_dropout=False))
    assert model._configured_lm_dropout() == 0.0


def test_the_reported_lm_dropout_is_the_rate_upstream_applies():
    """Upstream's ``_lm_dropout_context`` decides what runs; the record must agree.

    It leaves the checkpoint's rate for ``0``, so a request of ``0`` reports the
    configured one. When upstream makes ``0`` switch the dropout off this fails,
    and the report in ``_effective_settings`` is what to change.
    """
    context = pytest.importorskip("esm.models.esmfold2.processor")._lm_dropout_context
    model = _live(lm_encoder=SimpleNamespace(lm_dropout=0.1, per_loop_lm_dropout=True))
    for requested in (None, 0.0, 0.3):
        with context(model.net, requested):
            applied = model.net.config.lm_encoder.lm_dropout
        entries = model._effective_settings({"lm_dropout": requested}, "model")
        assert entries["esmfold2.effective.lm_dropout"] == applied


def test_experimental_lm_dropout_is_its_configured_rate():
    model = _live(type="experimental", lm_dropout=0.25)
    assert model._configured_lm_dropout() == 0.25


def test_a_setting_the_module_does_not_expose_is_none_not_guessed():
    entries = _model()._effective_settings({}, "model")
    assert entries["esmfold2.effective.num_sampling_steps"] is None
    assert entries["esmfold2.effective.max_inference_sigma"] is None


# -- sampler knobs -----------------------------------------------------------


def test_a_declared_sampler_knob_is_not_warned_about(recwarn):
    _live().fold(_spi(), noise_scale=1.0)
    assert not [w for w in recwarn if "discarded" in str(w.message)]


def test_a_knob_the_module_does_not_declare_is_warned_about():
    model = _model()

    class _Bare:  # a forward that declares no sampler knob
        def forward(self, **kwargs):
            return None

    bare = _Bare()
    bare.esmc = object()
    bare.config = model.net.config
    model.net = bare
    with pytest.warns(RuntimeWarning, match="noise_scale"):
        model.fold(_spi(), noise_scale=1.0)


# -- Hub resolution ----------------------------------------------------------


def test_a_directory_is_its_own_snapshot(tmp_path):
    from esmfold2_atomworks.model.esmfold2 import snapshot_dir

    assert snapshot_dir(str(tmp_path)) == str(tmp_path)


def test_a_hub_id_is_resolved_to_its_snapshot(monkeypatch, tmp_path):
    hub = pytest.importorskip("esm.models.hub")
    from esmfold2_atomworks.model.esmfold2 import snapshot_dir

    seen = []
    monkeypatch.setattr(
        hub, "resolve_model_dir", lambda source: seen.append(source) or str(tmp_path)
    )
    assert snapshot_dir("biohub/ESMFold2") == str(tmp_path)
    assert seen == ["biohub/ESMFold2"]


def test_provenance_names_the_snapshot_a_hub_id_resolved_to(tmp_path):
    snapshot = _snapshot(tmp_path, "biohub/ESMFold2", "69869f73")
    model = _loaded(snapshot)
    model.weights = "biohub/ESMFold2"
    model.weights_resolved = str(snapshot)
    record = model.provenance()
    assert record["esmfold2.weights"] == "biohub/ESMFold2"
    assert record["esmfold2.checkpoint.revision"] == "69869f73"
    assert record["esmfold2.checkpoint.versioning"] == "hub-snapshot"


def test_the_sampler_knobs_are_config_fields_passed_only_when_set():
    assert "noise_scale" not in FoldingConfig().as_fold_kwargs()
    kwargs = FoldingConfig(noise_scale=1.0, max_inference_sigma=80.0).as_fold_kwargs()
    assert kwargs["noise_scale"] == 1.0 and kwargs["max_inference_sigma"] == 80.0
    assert "step_scale" not in kwargs


def test_provenance_records_the_numerics_chosen_at_construction(tmp_path):
    model = _loaded(tmp_path / "w")
    model.esmc_precision, model.chunk_size, model.kernel_backend = "fp8", 64, None
    record = model.provenance()
    assert record["esmfold2.esmc_precision"] == "fp8"
    assert record["esmfold2.chunk_size"] == "64"
    assert record["esmfold2.kernel_backend"] == "None"


def test_a_setter_is_called_with_none_too_and_its_absence_is_recorded():
    from esmfold2_atomworks.model.esmfold2 import _apply

    calls = []
    net = SimpleNamespace(set_chunk_size=calls.append)
    assert _apply(net, "set_chunk_size", None) == "None"
    assert calls == [None]  # None disables chunking upstream; it is not skipped
    assert _apply(SimpleNamespace(), "set_chunk_size", 64).startswith("not applied")


@pytest.mark.parametrize(
    ("config_type", "esmc_source", "requested", "applied"),
    [
        ("release", "bundled", "fp8", "fp8"),
        ("experimental", "/mirror/ESMC-6B", "fp8", "bf16"),
        ("release", "none", "bf16", "none: no backbone"),
    ],
)
def test_the_backbone_precision_recorded_is_the_one_applied(
    config_type, esmc_source, requested, applied
):
    model = AtomWorksESMFold2.__new__(AtomWorksESMFold2)
    model.net = SimpleNamespace(config=SimpleNamespace(type=config_type))
    model.esmc_source = esmc_source
    assert model._applied_esmc_precision(requested) == applied


def test_fold_atom_array_records_the_source_atoms_a_link_leaves_out(monkeypatch):
    import biotite.structure as struc

    import esmfold2_atomworks.data.atomworks_to_esm as adapter
    import esmfold2_atomworks.data.molecular_complex as reverse
    from esmfold2_atomworks.data import topology

    def convert(atoms, *, chain_info=None, report=None, **kwargs):
        report.link_atoms_left_out["B"] = ["NAG1/O1"]
        return _spi()

    monkeypatch.setattr(adapter, "atom_array_to_structure_prediction_input", convert)
    monkeypatch.setattr(reverse, "result_to_atom_array", lambda result, **_: "atoms")
    monkeypatch.setattr(topology, "ccd_name_collisions", lambda atoms, spi: [])

    record: dict = {}
    _model().fold_atom_array(struc.AtomArray(0), record=record, bonds=False)
    assert record["esmfold2.link_atoms_left_out"] == {"B": ["NAG1/O1"]}

    quiet: dict = {}
    monkeypatch.setattr(
        adapter,
        "atom_array_to_structure_prediction_input",
        lambda atoms, *, chain_info=None, report=None, **kw: _spi(),
    )
    _model().fold_atom_array(struc.AtomArray(0), record=quiet, bonds=False)
    assert "esmfold2.link_atoms_left_out" not in quiet
