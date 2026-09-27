"""The folded structure carries its source chains' ``chain_type``, and no more.

The model's output cannot tell an L- from a D-polypeptide, or DNA from RNA, so
``result_to_atom_array`` never writes ``chain_type``. ``fold_atom_array`` has
the source, and copies each chain's back -- only where the source had one.
"""

from __future__ import annotations

import numpy as np
import pytest

struc = pytest.importorskip("biotite.structure")

from esmfold2_atomworks.data.molecular_complex import copy_chain_types

POLYPEPTIDE_L, RNA, NON_POLYMER = 6, 10, 8  # values, not their meaning, matter here


def _atoms(chains: list[str], types: list[int] | None = None, key: str = "chain_id"):
    array = struc.AtomArray(len(chains))
    array.set_annotation(key, np.array(chains))
    if key != "chain_id":
        array.set_annotation("chain_id", np.array(["X"] * len(chains)))
    if types is not None:
        array.set_annotation("chain_type", np.array(types, dtype=np.int8))
    return array


def test_each_output_chain_gets_its_source_chains_type():
    source = _atoms(
        ["A", "A", "B", "C"], [POLYPEPTIDE_L, POLYPEPTIDE_L, RNA, NON_POLYMER]
    )
    folded = _atoms(["A", "A", "A", "B", "C", "C"])
    out = copy_chain_types(folded, source)
    assert out.chain_type.tolist() == [POLYPEPTIDE_L] * 3 + [RNA] + [NON_POLYMER] * 2
    assert out.chain_type.dtype == source.chain_type.dtype
    assert "chain_type" not in folded.get_annotation_categories()


def test_a_source_without_chain_type_is_left_alone():
    folded = _atoms(["A", "B"])
    out = copy_chain_types(folded, _atoms(["A", "B"]))
    assert out is folded
    assert "chain_type" not in out.get_annotation_categories()


def test_a_dropped_source_chain_is_not_needed():
    # Water dropped by policy never reaches the output.
    source = _atoms(["A", "W"], [POLYPEPTIDE_L, 12])
    out = copy_chain_types(_atoms(["A", "A"]), source)
    assert out.chain_type.tolist() == [POLYPEPTIDE_L] * 2


def test_an_output_chain_the_source_does_not_name_raises():
    source = _atoms(["A"], [POLYPEPTIDE_L])
    with pytest.raises(ValueError, match=r"\['B'\]"):
        copy_chain_types(_atoms(["A", "B"]), source)


def test_chains_are_matched_under_the_adapters_chain_key():
    source = _atoms(["A1", "B1"], [POLYPEPTIDE_L, RNA], key="chain_iid")
    out = copy_chain_types(_atoms(["A1", "B1"]), source, chain_key="chain_iid")
    assert out.chain_type.tolist() == [POLYPEPTIDE_L, RNA]


def test_fold_atom_array_copies_under_a_non_default_chain_key(monkeypatch):
    """The whole plumbing, not just the helper: chain_key reaches the copy."""
    from test_roundtrip_ligand_name import _Result

    from esmfold2_atomworks.model.esmfold2 import AtomWorksESMFold2

    model = AtomWorksESMFold2.__new__(AtomWorksESMFold2)
    monkeypatch.setattr(model, "fold", lambda spi, **kwargs: _Result(), raising=False)
    monkeypatch.setattr(
        "esmfold2_atomworks.data.atomworks_to_esm.atom_array_to_structure_prediction_input",
        lambda atoms, **kwargs: object(),
    )
    # The decoded complex names its chains A and B; the source keys them by
    # chain_iid, with a chain_id that would match nothing.
    source = _atoms(["A", "B"], [POLYPEPTIDE_L, NON_POLYMER], key="chain_iid")
    folded, _ = model.fold_atom_array(source, adapter_kwargs={"chain_key": "chain_iid"})
    by_chain = dict(
        zip(folded.chain_id.tolist(), folded.chain_type.tolist(), strict=True)
    )
    assert by_chain == {"A": POLYPEPTIDE_L, "B": NON_POLYMER}
