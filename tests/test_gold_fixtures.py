"""The gold fixtures are anchored to facts about the PDB entries themselves.

A frozen snapshot closes the circularity in the parity suite -- the adapter is
no longer compared against something rebuilt from its own output -- but only if
the snapshot is *right*. A snapshot generated once from a buggy pipeline would
be just as self-consistent and just as wrong, only harder to notice.

So these tests assert what is independently known about each entry: chain
counts, sequence lengths, the identity of the cofactors, where the
selenomethionines sit. They are checkable against the PDB without running any
of this code, which is the property that makes them worth having.
"""

from __future__ import annotations

import pytest

from esmfold2_atomworks.data.atomworks_to_esm import (
    atom_array_to_structure_prediction_input,
)

# Hen egg-white lysozyme, and the haemoglobin chain termini -- typed from the
# entries, not read out of the fixtures they are checking.
LYSOZYME_HEAD = "KVFGRCELAAAMKRHGLDNYRGYS"
HBA_HEAD = "VLSPADKTNVKAAWGKV"
HBB_HEAD = "VHLTPEEKSAVTALWGKV"


def _by_kind(document: dict, kind: str) -> list[dict]:
    return [c for c in document["chains"] if c["kind"] == kind]


@pytest.mark.offline
def test_lysozyme_gold_is_the_entry_we_think_it_is(gold_document):
    document = gold_document("lysozyme")
    chains = document["chains"]
    assert len(chains) == 1
    (chain,) = chains
    assert chain["kind"] == "protein"
    assert len(chain["sequence"]) == 129
    assert chain["sequence"].startswith(LYSOZYME_HEAD)
    assert "modifications" not in chain


@pytest.mark.offline
def test_haemoglobin_gold_is_a2b2_with_four_haems(gold_document):
    document = gold_document("hemoglobin")
    proteins = _by_kind(document, "protein")
    ligands = _by_kind(document, "ligand")

    assert [c["id"] for c in proteins] == ["A", "B", "C", "D"]
    assert [c["id"] for c in ligands] == ["E", "G", "H", "J"]

    alpha = [c for c in proteins if len(c["sequence"]) == 141]
    beta = [c for c in proteins if len(c["sequence"]) == 146]
    assert len(alpha) == 2 and len(beta) == 2
    assert all(c["sequence"].startswith(HBA_HEAD) for c in alpha)
    assert all(c["sequence"].startswith(HBB_HEAD) for c in beta)
    # The two copies of each subunit really are the same molecule.
    assert alpha[0]["sequence"] == alpha[1]["sequence"]
    assert beta[0]["sequence"] == beta[1]["sequence"]

    assert all(c["ccd"] == ["HEM"] for c in ligands)


@pytest.mark.offline
def test_modified_gold_places_four_selenomethionines(gold_document):
    document = gold_document("modified")
    (chain,) = document["chains"]
    assert len(chain["sequence"]) == 70
    mods = chain["modifications"]
    assert [m["ccd"] for m in mods] == ["MSE"] * 4
    assert [m["position"] for m in mods] == [0, 34, 63, 64]
    # Selenomethionine stands in for methionine, so the canonical sequence must
    # read M at each of those positions. If it did not, the position index and
    # the sequence would be describing different things.
    assert all(chain["sequence"][m["position"]] == "M" for m in mods)


@pytest.mark.offline
def test_cofactor_golds_name_the_right_ligands(gold_document):
    assert [c["ccd"] for c in _by_kind(gold_document("zinc"), "ligand")] == [["ZN"]]
    assert [c["ccd"] for c in _by_kind(gold_document("flavoprotein"), "ligand")] == [
        ["FAD"],
        ["UV3"],
    ]


@pytest.mark.parametrize(
    "fixture", ["lysozyme", "hemoglobin", "modified", "zinc", "flavoprotein"]
)
def test_adapter_reproduces_the_gold_input(parsed, ccd, gold_document, fixture):
    """The adapter's input must agree with the frozen one, field by field.

    This is the semantic comparison that sits underneath feature parity. It
    fails with a readable message naming the chain and the field, where a
    tensor comparison would only say that ``ref_element`` differs.
    """
    from esm.models.esmfold2.types import LigandInput, ProteinInput

    atoms, chain_info = parsed(fixture)
    produced = atom_array_to_structure_prediction_input(atoms, chain_info=chain_info)
    expected = gold_document(fixture)["chains"]

    assert len(produced.sequences) == len(expected)
    for entry, want in zip(produced.sequences, expected, strict=True):
        assert str(entry.id) == want["id"]
        if want["kind"] == "protein":
            assert isinstance(entry, ProteinInput)
            assert entry.sequence == want["sequence"]
            got_mods = [
                {"position": m.position, "ccd": m.ccd}
                for m in (entry.modifications or [])
            ]
            assert got_mods == want.get("modifications", [])
        else:
            assert isinstance(entry, LigandInput)
            assert list(entry.ccd or []) == want["ccd"]
            assert entry.smiles is None
