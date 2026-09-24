"""Every declaration binds to exactly the chain it names, or the call fails.

``sequences``, ``msas`` and ``ligands`` are statements about particular chains.
The conversion reads each only for chains of particular kinds, so a declaration
that names another chain -- or none -- would be passed over without a trace,
leaving the caller believing it had been applied. That is an invalid request
rather than an approximation, so these raise ``ChainDeclarationError`` and
there is no opt-in.
"""

from __future__ import annotations

import io

import pytest

from esmfold2_atomworks.data.atomworks_to_esm import (
    _by_chain,
    _check_msa_binds,
    _index_ligand_specs,
    atom_array_to_structure_prediction_input,
)
from esmfold2_atomworks.data.spec import ChainDeclarationError, LigandSpec


def _msa(query: str, hits: tuple[str, ...] = ()):
    from esm.utils.msa import MSA

    lines = [">query", query]
    for index, hit in enumerate(hits):
        lines += [f">hit{index}", hit]
    return MSA.from_a3m(io.StringIO("\n".join(lines) + "\n"))


# -- a declaration must name a chain that is there ---------------------------


@pytest.mark.parametrize(
    "declaration",
    [
        {"sequences": {"Z": "GAAG"}},
        {"msas": {"Z": None}},
        {"ligands": (LigandSpec(chain_id="Z", smiles="CCO"),)},
    ],
    ids=["sequence", "msa", "ligand"],
)
def test_a_declaration_for_a_missing_chain_is_refused(parsed, ccd, declaration):
    """Otherwise it is ignored, and the structure folded as though it applied."""
    atoms, chain_info = parsed("lysozyme")
    with pytest.raises(ChainDeclarationError, match="no such chain"):
        atom_array_to_structure_prediction_input(
            atoms, chain_info=chain_info, **declaration
        )


# -- ... of a kind it applies to ---------------------------------------------


def test_a_sequence_override_for_a_ligand_chain_is_refused(parsed, ccd):
    atoms, chain_info = parsed("hemoglobin")
    with pytest.raises(ChainDeclarationError, match="kind is 'ligand'"):
        atom_array_to_structure_prediction_input(
            atoms, chain_info=chain_info, sequences={"E": "GAAG"}
        )


def test_an_msa_for_a_nucleic_acid_chain_is_refused(ccd):
    """ESMFold2 reads MSAs for protein chains only, so this one would vanish."""
    pytest.importorskip("atomworks.io.tools.inference")
    from atomworks.io.tools.inference import DNA, components_to_atom_array
    from esm.utils.msa import MSA

    atoms = components_to_atom_array([DNA(seq="ATGCATGC", chain_id="A")])
    with pytest.raises(ChainDeclarationError, match="kind is 'dna'"):
        atom_array_to_structure_prediction_input(
            atoms, msas={"A": MSA.from_sequences(["ATGCATGC"])}
        )


def test_a_ligand_spec_for_a_polymer_chain_is_refused(parsed, ccd):
    """A LigandSpec is read only for non-polymer chains; this one never would be."""
    atoms, chain_info = parsed("lysozyme")
    with pytest.raises(ChainDeclarationError, match="kind is 'protein'"):
        atom_array_to_structure_prediction_input(
            atoms,
            chain_info=chain_info,
            ligands=(LigandSpec(chain_id="A", smiles="CCO"),),
        )


# -- ... and must not contradict itself --------------------------------------


@pytest.mark.offline
def test_a_ligand_key_must_name_the_chain_its_spec_describes():
    """Otherwise the spec for chain 'C' is applied to chain 'B'."""
    with pytest.raises(ChainDeclarationError, match="same chain"):
        _index_ligand_specs({"B": LigandSpec(chain_id="C", smiles="CCO")})


@pytest.mark.offline
def test_a_chain_has_one_ligand_identity():
    """Otherwise the second spec silently replaces the first."""
    first = LigandSpec(chain_id="B", smiles="CCO")
    second = LigandSpec(chain_id="B", ccd=("HEM",))
    with pytest.raises(ChainDeclarationError, match="two LigandSpecs"):
        _index_ligand_specs((first, second))


@pytest.mark.offline
def test_a_numeric_key_names_the_chain_it_spells():
    """Keys are compared as strings, as the structure's own labels are.

    ``{1: ...}`` -- which a config file can produce for chain ``"1"`` -- would
    otherwise match no chain at all.
    """
    assert _by_chain("sequences", {1: "GAAG"}) == {"1": "GAAG"}
    spec = LigandSpec(chain_id="1", smiles="CCO")
    assert _index_ligand_specs({1: spec}) == {"1": spec}
    with pytest.raises(ChainDeclarationError, match="twice"):
        _by_chain("sequences", {1: "GAAG", "1": "GGAA"})


def test_the_ligand_binding_is_checked_on_the_adapter_path(parsed, ccd):
    atoms, chain_info = parsed("hemoglobin")
    with pytest.raises(ChainDeclarationError, match="same chain"):
        atom_array_to_structure_prediction_input(
            atoms,
            chain_info=chain_info,
            ligands={"E": LigandSpec(chain_id="G", ccd=("HEM",))},
        )


# -- an override describes one whole chain -----------------------------------


def test_an_empty_override_in_a_multichain_structure_is_refused(parsed, ccd):
    """Alone it would fail anyway, with nothing left to fold; beside others it would vanish."""
    atoms, chain_info = parsed("hemoglobin")
    with pytest.raises(ChainDeclarationError, match="empty override"):
        atom_array_to_structure_prediction_input(
            atoms, chain_info=chain_info, sequences={"B": ""}
        )


def test_a_chain_break_in_an_override_is_refused(parsed, ccd):
    atoms, chain_info = parsed("lysozyme")
    with pytest.raises(ChainDeclarationError, match="chain break"):
        atom_array_to_structure_prediction_input(
            atoms, chain_info=chain_info, sequences={"A": "GAAG:GGAA"}
        )


def test_upstream_splits_a_protein_sequence_at_a_chain_break():
    """Why the override above is refused: every fold passes through this."""
    pytest.importorskip("esm.models.esmfold2.processor")
    from esm.models.esmfold2.processor import clean_esmfold2_input
    from esm.models.esmfold2.types import ProteinInput, StructurePredictionInput

    cleaned = clean_esmfold2_input(
        StructurePredictionInput(sequences=[ProteinInput(id="A", sequence="GAAG:GGAA")])
    )
    assert [entry.id for entry in cleaned.sequences] == [["A_0"], ["A_1"]]


# -- an MSA binds only to the sequence it aligns -----------------------------


def test_an_msa_built_for_another_chain_is_refused(parsed, ccd, gold_document):
    """Hemoglobin's alpha alignment on a beta chain: upstream would clamp it in.

    146 residues against 141 columns -- the last five would all read column 141.
    """
    alpha = gold_document("hemoglobin")["chains"][0]["sequence"]
    atoms, chain_info = parsed("hemoglobin")
    with pytest.raises(ChainDeclarationError, match="past its end"):
        atom_array_to_structure_prediction_input(
            atoms, chain_info=chain_info, msas={"B": _msa(alpha)}
        )


def test_an_msa_of_the_right_width_for_another_sequence_is_refused(
    parsed, ccd, gold_document
):
    """Same width, different query: nothing about the shapes would show it."""
    sequence = gold_document("lysozyme")["chains"][0]["sequence"]
    other = ("A" if sequence[0] != "A" else "G") + sequence[1:]
    atoms, chain_info = parsed("lysozyme")
    with pytest.raises(ChainDeclarationError, match=f"1 of {len(sequence)} positions"):
        atom_array_to_structure_prediction_input(
            atoms, chain_info=chain_info, msas={"A": _msa(other)}
        )


def test_an_msa_follows_the_sequence_being_folded(parsed, ccd, gold_document):
    """With an override, the alignment's query must be the design, not its parent."""
    native = gold_document("lysozyme")["chains"][0]["sequence"]
    designed = "G" * len(native)
    atoms, chain_info = parsed("lysozyme")
    with pytest.raises(ChainDeclarationError, match="replace the query row"):
        atom_array_to_structure_prediction_input(
            atoms,
            chain_info=chain_info,
            sequences={"A": designed},
            msas={"A": _msa(native)},
        )

    # The parent's alignment with the design as its query row binds.
    spi = atom_array_to_structure_prediction_input(
        atoms,
        chain_info=chain_info,
        sequences={"A": designed},
        msas={"A": _msa(designed, (native,))},
    )
    assert spi.sequences[0].msa is not None


@pytest.mark.offline
def test_something_that_is_not_an_msa_is_refused():
    with pytest.raises(TypeError, match="not an esm.utils.msa.MSA"):
        _check_msa_binds("A", "GAAG", "GAAG")
