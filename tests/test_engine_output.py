"""What the engine writes and records for an input it parsed.

The structure side needs biotite and the CIF writer needs AtomWorks; the fold
itself is replaced, so no weights or GPU are involved.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("atomworks")
struc = pytest.importorskip("biotite.structure")


def _output_structure():
    """What ``result_to_atom_array`` returns: no bond list."""
    atoms = struc.AtomArray(6)
    atoms.coord = np.arange(18, dtype=np.float32).reshape(-1, 3)
    atoms.set_annotation("chain_id", np.array(["A"] * 6, dtype="U4"))
    atoms.set_annotation("res_id", np.array([1, 1, 1, 2, 2, 2]))
    atoms.set_annotation("ins_code", np.array([""] * 6, dtype="U1"))
    atoms.set_annotation("res_name", np.array(["ALA"] * 3 + ["GLY"] * 3, dtype="U5"))
    atoms.set_annotation("atom_name", np.array(["N", "CA", "C"] * 2, dtype="U6"))
    atoms.set_annotation("element", np.array(["N", "C", "C"] * 2, dtype="U2"))
    atoms.set_annotation("hetero", np.zeros(6, dtype=bool))
    atoms.set_annotation("is_polymer", np.ones(6, dtype=bool))
    assert atoms.bonds is None
    return atoms


def test_dump_writes_a_structure_that_has_no_bond_list(tmp_path):
    from biotite.structure.io.pdbx import CIFFile, get_structure

    from esmfold2_atomworks.inference.engine import ESMFold2Output

    output = ESMFold2Output(
        atom_array=_output_structure(), metadata={"k": 1.0}, example_id="x"
    )
    path = output.dump(tmp_path, verbose=False)

    assert path == tmp_path / "x.cif"
    assert json.loads((tmp_path / "x.json").read_text()) == {"k": 1.0}
    written = get_structure(CIFFile.read(path), model=1)
    assert written.array_length() == 6
    assert output.atom_array.bonds is None, "dump must not change the output"


def test_a_path_is_parsed_with_the_given_configuration(monkeypatch, tmp_path):
    from esmfold2_atomworks.data import loading
    from esmfold2_atomworks.inference.engine import _load

    seen = {}

    def fake_parse(source, *, config=None):
        seen.update(config=config, source=source)
        return {"asym_unit": ["model-1"], "chain_info": {"A": {}}}

    monkeypatch.setattr("atomworks.io.parse", fake_parse)
    atoms, chain_info = _load(tmp_path / "s.cif", {"hydrogen_policy": "remove"})
    assert (atoms, chain_info) == ("model-1", {"A": {}})
    assert seen["config"].hydrogen_policy == "remove"
    assert loading.parse_provenance({"hydrogen_policy": "remove"})[
        "atomworks.parse_config"
    ] == {"hydrogen_policy": "remove"}


def test_a_parse_configuration_is_a_config_a_preset_or_a_mapping():
    from atomworks.io.config import ParseConfig

    from esmfold2_atomworks.data.loading import resolve_parse_config

    config = ParseConfig(hydrogen_policy="remove")
    assert resolve_parse_config(config) is config
    assert resolve_parse_config(None) == ParseConfig()
    assert resolve_parse_config("rcsb") == ParseConfig.from_preset("rcsb")
    assert resolve_parse_config({"altloc": "first"}).altloc == "first"


def test_a_parse_option_that_does_not_exist_is_refused_not_ignored():
    from esmfold2_atomworks.data.loading import resolve_parse_config

    with pytest.raises(TypeError):
        resolve_parse_config({"hydrogen_polcy": "remove"})
    with pytest.raises(TypeError, match="not int"):
        resolve_parse_config(3)


def test_one_result_per_model_is_refused_not_resolved(monkeypatch, tmp_path):
    from esmfold2_atomworks.inference.engine import _load

    monkeypatch.setattr("atomworks.io.parse", lambda source, **_: [{}, {}])
    with pytest.raises(ValueError, match="one per model"):
        _load(tmp_path / "nmr.cif")


def test_the_provenance_names_the_version_and_the_default_configuration():
    from esmfold2_atomworks.data.loading import atomworks_version, parse_provenance

    entries = parse_provenance()
    assert entries["atomworks.version"] == atomworks_version()
    assert entries["atomworks.parse_config"] == "default"
