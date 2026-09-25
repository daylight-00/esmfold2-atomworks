"""The artifact record behind the published results is complete and consistent.

``reproducibility/ARTIFACTS.lock`` pins what the reference runs read -- the
checkpoint, the ESMC backbone paired with it, the CCD pickle -- and
``checkpoint_equivalence.json``, written by ``scripts/compare_checkpoints.py``,
is the evidence for any statement about the re-published checkpoint. These
check the record's form and that the two agree with each other; they need no
weights. The last one checks where a model says its CCD pickle came from.
"""

from __future__ import annotations

import json
import re

import pytest

from esmfold2_atomworks import paths

LOCK = paths.REPO_ROOT / "reproducibility" / "ARTIFACTS.lock"
EQUIVALENCE = paths.REPO_ROOT / "reproducibility" / "checkpoint_equivalence.json"

SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_SHA1 = re.compile(r"git-sha1:[0-9a-f]{40}")
REVISION = re.compile(r"[0-9a-f]{40}")
FILE_KEYS = re.compile(r".+\.(safetensors|json|pkl)$")


@pytest.fixture(scope="module")
def lock():
    if not LOCK.is_file():
        pytest.skip("ARTIFACTS.lock ships with the repository, not the package")
    return paths.read_upstream_lock(LOCK)


@pytest.mark.offline
def test_every_artifact_is_pinned_by_revision_and_digest(lock):
    assert set(lock) == {"esmfold2", "esmc", "ccd"}
    for name, entry in lock.items():
        assert entry["repo"].startswith("biohub/"), name
        assert REVISION.fullmatch(entry["revision"]), name
        files = {k: v for k, v in entry.items() if FILE_KEYS.match(k)}
        assert files, f"{name} pins no file"
        for key, digest in files.items():
            assert SHA256.fullmatch(digest) or GIT_SHA1.fullmatch(digest), (name, key)
    # The weights themselves, not only their configs.
    assert SHA256.fullmatch(lock["esmfold2"]["model.safetensors"])
    shards = [k for k in lock["esmc"] if k.endswith(".safetensors")]
    assert len(shards) == 6


@pytest.mark.offline
def test_the_equivalence_record_compares_against_the_pinned_reference(lock):
    if not EQUIVALENCE.is_file():
        pytest.skip("checkpoint_equivalence.json ships with the repository")
    record = json.loads(EQUIVALENCE.read_text())
    reference = record["reference"]
    assert reference["repo"] == lock["esmfold2"]["repo"]
    assert reference["revision"] == lock["esmfold2"]["revision"]
    assert reference["esmc"]["repo"] == lock["esmc"]["repo"]
    assert reference["esmc"]["revision"] == lock["esmc"]["revision"]
    assert record["candidate"]["esmc"] == "bundled"
    # What docs/04 states rests on exactly this.
    assert record["same_weights"] is True
    for part in ("trunk", "esmc"):
        assert record[part]["differ"] == [], part
        assert record[part]["compared"] == record[part]["identical"] > 0, part


def test_the_ccd_source_follows_esms_order(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    conformers = pytest.importorskip("esm.models.esmfold2.conformers")
    from esmfold2_atomworks.model.esmfold2 import ccd_source

    monkeypatch.setattr(conformers, "CCD_PICKLE_PATH", None)
    monkeypatch.setattr(conformers, "_CCD_MOLECULES", None)  # nothing loaded yet
    assert ccd_source(tmp_path) == str(tmp_path / "ccd.pkl")
    assert "Hub" in ccd_source(None)

    captured = tmp_path / "captured.pkl"
    captured.write_bytes(b"")
    monkeypatch.setattr(conformers, "CCD_PICKLE_PATH", captured)
    # Captured at import, it outranks the location a caller passes.
    assert ccd_source(tmp_path / "elsewhere") == str(captured)


def test_a_dictionary_loaded_earlier_is_reported_as_unknown(tmp_path, monkeypatch):
    """The builder's location is never read then; naming it would be a guess."""
    pytest.importorskip("torch")
    conformers = pytest.importorskip("esm.models.esmfold2.conformers")
    from esmfold2_atomworks.model.esmfold2 import ccd_source

    monkeypatch.setattr(conformers, "CCD_PICKLE_PATH", None)
    monkeypatch.setattr(conformers, "_CCD_MOLECULES", {"already": "loaded"})
    assert ccd_source(tmp_path).startswith("unknown")

    # A captured path still pins it: every load, the earlier one included,
    # read that path.
    captured = tmp_path / "captured.pkl"
    captured.write_bytes(b"")
    monkeypatch.setattr(conformers, "CCD_PICKLE_PATH", captured)
    assert ccd_source(tmp_path) == str(captured)


def test_esm_still_keeps_the_ccd_where_ccd_source_looks():
    """ccd_source reads two module names of esm's, one private.

    A rename upstream would make it misreport silently, so it fails here.
    """
    pytest.importorskip("torch")
    conformers = pytest.importorskip("esm.models.esmfold2.conformers")
    assert hasattr(conformers, "_CCD_MOLECULES")
    assert hasattr(conformers, "CCD_PICKLE_PATH")
