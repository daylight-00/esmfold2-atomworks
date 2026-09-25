"""Alignments found by AtomWorks' own loader reach ESMFold2 unchanged.

``LoadPolymerMSAs`` is how an AtomWorks pipeline gets its MSAs. The end-to-end
tests run the real loader on a real a3m -- AtomWorks' 1wym test case, whose
hits carry insertions and taxonomy ids -- and require the features to be
exactly those of the same file handed to the adapter directly. The rest pin the
translation itself on small hand-made alignments.
"""

from __future__ import annotations

import numpy as np
import pytest

from esmfold2_atomworks import paths

CASE = (
    paths.DESIGN_ROOT
    / "atomworks"
    / "tests"
    / "ml"
    / "pipelines"
    / "test_data"
    / "1wym"
)


def _codes(sequence: str) -> list[int]:
    from atomworks.ml.transforms.msa._msa_constants import (
        AMINO_ACID_ONE_LETTER_TO_INT,
    )

    return [AMINO_ACID_ONE_LETTER_TO_INT[ch] for ch in sequence]


def _alignment(rows: list[str], ins: list[list[int]], tax_ids: list[str]) -> dict:
    msa = np.array([_codes(row) for row in rows], dtype=np.int8)
    return {
        "msa": msa,
        "ins": np.array(ins, dtype=np.int16),
        "tax_ids": np.array(tax_ids),
        "sequence_similarity": np.ones(len(rows)),
        "msa_is_padded_mask": np.zeros(msa.shape, dtype=bool),
    }


# -- the translation ---------------------------------------------------------


def test_rows_deletions_and_pairing_keys_carry_over():
    pytest.importorskip("atomworks")
    pytest.importorskip("esm")
    from esmfold2_atomworks.data.msa import polymer_msa_to_esm

    msa = polymer_msa_to_esm(
        _alignment(
            ["ARNDC", "AR-DC", "WRNDC"],
            [[0, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 0, 0, 1]],
            ["", "9606", "unclassified"],
        )
    )
    assert msa.sequences == ["ARNDC", "AR-DC", "WRNDC"]
    # Only a numeric id is a key upstream will read; the query row needs none.
    assert msa.headers == ["query", "hit1 key=9606", "hit2"]
    np.testing.assert_array_equal(
        msa.deletions, [[0, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 0, 0, 1]]
    )


def test_a_padded_alignment_is_refused():
    pytest.importorskip("atomworks")
    pytest.importorskip("esm")
    from esmfold2_atomworks.data.msa import polymer_msa_to_esm

    data = _alignment(["ARN", "A-N"], [[0, 0, 0], [0, 0, 0]], ["", ""])
    data["msa_is_padded_mask"][1, 2] = True
    with pytest.raises(ValueError, match="padded"):
        polymer_msa_to_esm(data, chain_id="A")


def test_codes_outside_the_protein_alphabet_are_refused():
    pytest.importorskip("atomworks")
    pytest.importorskip("esm")
    from esmfold2_atomworks.data.msa import polymer_msa_to_esm

    data = _alignment(["ARN"], [[0, 0, 0]], [""])
    data["msa"][0, 1] = 30  # an RNA code (U)
    with pytest.raises(ValueError, match="protein alphabet"):
        polymer_msa_to_esm(data, chain_id="A")


def _two_chain_example():
    struc = pytest.importorskip("biotite.structure")
    from atomworks.enums import ChainType

    atoms = struc.AtomArray(2)
    atoms.chain_id = np.array(["A", "B"])
    atoms.add_annotation("chain_type", dtype=int)
    atoms.chain_type = np.array([ChainType.POLYPEPTIDE_L.value, ChainType.RNA.value])
    alignments = {
        "A": _alignment(["ARN", "A-N"], [[0, 0, 0], [0, 1, 0]], ["", "9606"]),
        "B": {"msa": np.array([[27, 28]]), "ins": np.zeros((1, 2)), "tax_ids": [""]},
    }
    return {"atom_array": atoms, "polymer_msas_by_chain_id": alignments}


def test_an_rna_alignment_is_left_out_and_named():
    pytest.importorskip("esm")
    pytest.importorskip("atomworks.ml.transforms.base")
    from esmfold2_atomworks.data.pipelines import PolymerMSAsToESMFold2

    data = PolymerMSAsToESMFold2().forward(_two_chain_example())
    assert set(data["msas"]) == {"A"}
    assert data["msas_left_out"] == {"B": "RNA"}


def test_a_second_alignment_for_the_same_chain_is_a_conflict():
    pytest.importorskip("esm")
    pytest.importorskip("atomworks.ml.transforms.base")
    from esmfold2_atomworks.data.pipelines import PolymerMSAsToESMFold2

    data = _two_chain_example()
    data["msas"] = {"A": object()}
    with pytest.raises(ValueError, match="keep one"):
        PolymerMSAsToESMFold2().forward(data)


# -- end to end: the real loader on files it parses ---------------------------


def _hits(query: str, taxids: list[int | None]) -> list[tuple[str, int | None]]:
    """Rows departing from *query*: a substitution, a gap run, a3m insertions."""
    rows = []
    for index, taxid in enumerate(taxids):
        chars = list(query)
        chars[3 + index] = "W" if chars[3 + index] != "W" else "Y"
        for position in range(12 + index, 15 + index):
            chars[position] = "-"
        # Lowercase letters are insertions: no column, a deletion count.
        chars[20 + index] = "gk"[: index + 1] + chars[20 + index]
        rows.append(("".join(chars), taxid))
    return rows


def _write_a3m(path, query: str, hits, *, native: bool) -> None:
    """As a sequence search writes it (UniRef headers), or with upstream's keys."""
    lines = [">query", query]
    for index, (row, taxid) in enumerate(hits, 1):
        if native:
            lines.append(
                f">hit{index}" + (f" key={taxid}" if taxid is not None else "")
            )
        else:
            tax = f" TaxID={taxid}" if taxid is not None else ""
            lines.append(
                f">UniRef100_X{index} protein n=1 Tax=Some species{tax} RepID=X{index}"
            )
        lines.append(row)
    path.write_text("\n".join(lines) + "\n")


def _through_the_loader(atoms, chain_info, msa_paths: dict[str, str]):
    """LoadPolymerMSAs then the translation, composed as a pipeline composes them."""
    import copy

    from atomworks.ml.transforms.base import Compose
    from atomworks.ml.transforms.msa.msa import LoadPolymerMSAs

    from esmfold2_atomworks.data.pipelines import PolymerMSAsToESMFold2

    info = copy.deepcopy(chain_info)
    for chain, path in msa_paths.items():
        info[chain]["msa_path"] = str(path)
    pipeline = Compose(
        [LoadPolymerMSAs(use_paths_in_chain_info=True), PolymerMSAsToESMFold2()]
    )
    return pipeline({"atom_array": atoms, "chain_info": info})


def _features(atoms, chain_info, msas):
    from esmfold2_atomworks.data.atomworks_to_esm import (
        atom_array_to_structure_prediction_input,
    )
    from esmfold2_atomworks.parity.compare import featurize

    return featurize(
        atom_array_to_structure_prediction_input(
            atoms, chain_info=chain_info, msas=msas
        )
    )


def test_a_monomer_loads_exactly_like_its_file(parsed, ccd, tmp_path):
    pytest.importorskip("atomworks.ml.transforms.msa.msa")
    from esm.utils.msa import MSA

    from esmfold2_atomworks.parity.compare import compare_features

    atoms, chain_info = parsed("lysozyme")
    query = chain_info["A"]["processed_entity_canonical_sequence"]
    hits = _hits(query, [1001, 2002, None])
    searched = tmp_path / "lysozyme.a3m"
    _write_a3m(searched, query, hits, native=False)

    data = _through_the_loader(atoms, chain_info, {"A": searched})
    assert data["msas_left_out"] == {}
    loaded = data["msas"]["A"]
    assert loaded.depth == 4 and loaded.deletions.any()

    direct = MSA.from_a3m(searched, remove_insertions=True)
    diff = compare_features(
        _features(atoms, chain_info, {"A": direct}),
        _features(atoms, chain_info, {"A": loaded}),
    )
    assert diff.identical, diff.report()
    assert {"msa", "has_deletion", "deletion_value"} <= set(diff.compared)


def test_a_heteromer_pairs_by_the_loaders_taxonomy(parsed, ccd, tmp_path):
    """Hemoglobin: rows sharing a TaxID across alpha and beta must pair."""
    pytest.importorskip("atomworks.ml.transforms.msa.msa")
    from esm.utils.msa import MSA

    from esmfold2_atomworks.parity.compare import compare_features

    atoms, chain_info = parsed("hemoglobin")
    by_sequence: dict[str, list[str]] = {}
    for chain, info in chain_info.items():
        if getattr(info["chain_type"], "is_protein", lambda: False)():
            by_sequence.setdefault(
                info["processed_entity_canonical_sequence"], []
            ).append(chain)
    assert len(by_sequence) == 2, "2hhb should hold two protein entities"

    searched, native = {}, {}
    for entity, (query, chains) in enumerate(sorted(by_sequence.items())):
        # 1001 is in both entities' alignments and pairs; the rest do not.
        hits = _hits(query, [1001, 3000 + entity, None])
        found, keyed = tmp_path / f"e{entity}.a3m", tmp_path / f"e{entity}_keyed.a3m"
        _write_a3m(found, query, hits, native=False)
        _write_a3m(keyed, query, hits, native=True)
        direct = MSA.from_a3m(keyed, remove_insertions=True)
        for chain in chains:
            searched[chain], native[chain] = found, direct

    data = _through_the_loader(atoms, chain_info, searched)
    loaded = _features(atoms, chain_info, data["msas"])
    diff = compare_features(_features(atoms, chain_info, native), loaded)
    assert diff.identical, diff.report()

    # Not vacuous: without the keys the same rows do not pair, and the
    # features differ -- so the loader's taxonomy is what paired them.
    unkeyed = {
        c: MSA.from_a3m(path, remove_insertions=True) for c, path in searched.items()
    }
    assert not compare_features(_features(atoms, chain_info, unkeyed), loaded).identical


def test_an_alignment_for_another_protein_is_refused():
    """AtomWorks' 1wym test case pairs the structure with a 327-residue a3m.

    Its only chain has 155 residues, so the loader finds an alignment for a
    different protein. Upstream would clamp it into place column by column;
    here it must be refused.
    """
    parse = pytest.importorskip("atomworks.io").parse
    if not (CASE / "1wym.a3m").is_file():
        pytest.skip(f"AtomWorks' 1wym MSA test case is not at {CASE}")
    from esmfold2_atomworks.data.atomworks_to_esm import (
        atom_array_to_structure_prediction_input,
    )
    from esmfold2_atomworks.data.spec import ChainDeclarationError

    parsed = parse(CASE / "1wym.cif.gz")
    atoms, chain_info = parsed["asym_unit"][0], parsed["chain_info"]
    data = _through_the_loader(atoms, chain_info, {"A": CASE / "1wym.a3m"})
    assert data["msas"]["A"].depth == 15
    with pytest.raises(ChainDeclarationError, match="327 aligned columns"):
        atom_array_to_structure_prediction_input(
            atoms, chain_info=chain_info, msas=data["msas"]
        )


def test_the_pipeline_option_wires_the_loader_in(parsed, ccd, tmp_path):
    pytest.importorskip("atomworks.ml.transforms.msa.msa")
    import copy

    from atomworks.ml.transforms.msa.msa import LoadPolymerMSAs

    from esmfold2_atomworks.data.pipelines import build_esmfold2_pipeline

    atoms, chain_info = parsed("lysozyme")
    query = chain_info["A"]["processed_entity_canonical_sequence"]
    searched = tmp_path / "lysozyme.a3m"
    _write_a3m(searched, query, _hits(query, [1001, 2002, None]), native=False)
    info = copy.deepcopy(chain_info)
    info["A"]["msa_path"] = str(searched)

    pipeline = build_esmfold2_pipeline(
        is_inference=True,
        seed=0,
        msa_loader=LoadPolymerMSAs(use_paths_in_chain_info=True),
    )
    out = pipeline({"example_id": "lysozyme", "atom_array": atoms, "chain_info": info})
    assert out["msas_left_out"] == {}
    assert out["feats"]["msa"].shape[0] == 4
