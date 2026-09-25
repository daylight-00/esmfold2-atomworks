"""A chain's kind is declared and then verified -- never guessed.

``chain_records`` classifies from AtomWorks' ``chain_type``. An array that did
not come through ``atomworks.io.parse`` carries neither that nor ``is_polymer``
and is classified ``"unsupported"``, because the alternative is guessing from
residue names. A caller who knows what a chain is says so with ``chain_kinds``,
and the declaration is checked against whatever the array still carries --
against the CCD, residue by residue, when it carries nothing.

The failure the checks exist for is one chain label holding more than one
molecule. ``atomworks.io.parse`` never produces one. An mmCIF read with author
fields routinely does, and a declaration covering the whole chain would fold the
rest as part of it -- or, under a sequence override, leave it out entirely.
"""

from __future__ import annotations

import numpy as np
import pytest

from esmfold2_atomworks.data.atomworks_to_esm import (
    DECLARABLE_CHAIN_KINDS,
    AdapterReport,
    atom_array_to_structure_prediction_input,
    chain_records,
)
from esmfold2_atomworks.data.spec import (
    ChainDeclarationError,
    InferredChainKindError,
    LigandIdentityError,
    LigandSpec,
    MixedChainError,
    UnsupportedChainError,
)

struc = pytest.importorskip("biotite.structure")
pytest.importorskip("atomworks.enums")

BACKBONE = ("N", "CA", "C", "O")


def _atoms(*residues: tuple[str, str, tuple[str, ...]], **annotations) -> object:
    """An ``AtomArray`` from ``(chain_id, res_name, atom_names)`` per residue.

    Only what biotite gives any file -- no ``chain_type``, no ``is_polymer`` --
    plus whatever per-atom *annotations* a test adds.
    """
    rows = [
        (chain, res_id, res_name, atom)
        for res_id, (chain, res_name, atom_names) in enumerate(residues, start=1)
        for atom in atom_names
    ]
    atoms = struc.AtomArray(len(rows))
    atoms.coord = np.zeros((len(rows), 3), dtype=np.float32)
    atoms.chain_id = np.array([row[0] for row in rows])
    atoms.res_id = np.array([row[1] for row in rows])
    atoms.res_name = np.array([row[2] for row in rows])
    atoms.atom_name = np.array([row[3] for row in rows])
    atoms.element = np.array([row[3][0] for row in rows])
    for name, values in annotations.items():
        atoms.set_annotation(name, np.asarray(values))
    return atoms


def _protein_and_ligand(**annotations):
    """Chain A, two alanines; chain B, one ligand residue."""
    return _atoms(
        ("A", "ALA", BACKBONE),
        ("A", "ALA", BACKBONE),
        ("B", "HEM", ("C1", "C2", "O1")),
        **annotations,
    )


def _by_id(records):
    return {record.chain_id: record for record in records}


@pytest.fixture
def convert():
    """The adapter, where the test needs a model input and not just records."""
    pytest.importorskip("esm.models.esmfold2.types")
    return atom_array_to_structure_prediction_input


# -- the seam ---------------------------------------------------------------


def test_an_unannotated_array_is_unsupported_without_a_declaration():
    records = _by_id(chain_records(_protein_and_ligand()))

    assert records["A"].kind == records["B"].kind == "unsupported"


def test_a_declaration_supplies_the_kind_and_is_not_an_inference():
    records = _by_id(
        chain_records(
            _protein_and_ligand(), chain_kinds={"A": "protein", "B": "ligand"}
        )
    )

    assert (records["A"].kind, records["B"].kind) == ("protein", "ligand")
    # Nothing was guessed, so nothing asks for allow_inferred_chain_kind.
    assert not records["A"].kind_is_inferred
    assert not records["B"].kind_is_inferred
    assert records["A"].sequence == "AA"


def test_a_declaration_leaves_undeclared_chains_as_they_were():
    records = _by_id(chain_records(_protein_and_ligand(), chain_kinds={"A": "protein"}))

    assert records["A"].kind == "protein"
    assert records["B"].kind == "unsupported"


# -- a declaration, not an override -------------------------------------------


def _chain_types(protein: str, ligand: str) -> list[int]:
    from atomworks.enums import ChainType

    return [int(ChainType[protein])] * 8 + [int(ChainType[ligand])] * 3


def test_a_declaration_agreeing_with_chain_type_is_accepted():
    atoms = _protein_and_ligand(chain_type=_chain_types("POLYPEPTIDE_L", "NON_POLYMER"))

    records = _by_id(chain_records(atoms, chain_kinds={"A": "protein", "B": "ligand"}))

    assert (records["A"].kind, records["B"].kind) == ("protein", "ligand")


def test_a_declaration_contradicting_chain_type_is_refused():
    atoms = _protein_and_ligand(chain_type=_chain_types("POLYPEPTIDE_L", "NON_POLYMER"))

    with pytest.raises(ChainDeclarationError, match="chain_type"):
        chain_records(atoms, chain_kinds={"A": "dna"})


def test_a_declaration_contradicting_is_polymer_is_refused():
    atoms = _protein_and_ligand(is_polymer=[True] * 8 + [False] * 3)

    with pytest.raises(ChainDeclarationError, match="is_polymer"):
        chain_records(atoms, chain_kinds={"B": "protein"})


# -- declarations must bind ---------------------------------------------------


def test_a_declaration_for_a_chain_that_is_not_there_is_refused():
    with pytest.raises(ChainDeclarationError, match="no such chain"):
        chain_records(_protein_and_ligand(), chain_kinds={"Z": "protein"})


def test_only_the_declarable_kinds_can_be_declared():
    """``unsupported`` names the absence of anything to go on; it is not a kind."""
    assert DECLARABLE_CHAIN_KINDS == {"protein", "dna", "rna", "ligand", "water"}

    for kind in ("unsupported", "peptide", "Protein"):
        with pytest.raises(ChainDeclarationError, match="declarable kinds"):
            chain_records(_protein_and_ligand(), chain_kinds={"A": kind})


def test_one_chain_declared_twice_is_refused():
    """Keys are compared as strings, so ``1`` and ``"1"`` are the same chain."""
    atoms = _atoms(("1", "ALA", BACKBONE))

    with pytest.raises(ChainDeclarationError, match="twice"):
        chain_records(atoms, chain_kinds={1: "protein", "1": "ligand"})


# -- verified against the CCD ---------------------------------------------------


def test_a_protein_chain_holding_water_is_refused():
    atoms = _atoms(("A", "ALA", BACKBONE), ("A", "HOH", ("O",)), ("A", "HOH", ("O",)))

    with pytest.raises(ChainDeclarationError, match=r"2 x HOH \(water\)"):
        chain_records(atoms, chain_kinds={"A": "protein"})


def test_a_protein_chain_holding_a_ligand_is_refused():
    atoms = _atoms(("A", "ALA", BACKBONE), ("A", "HEM", ("C1", "C2", "O1")))

    with pytest.raises(ChainDeclarationError, match="HEM"):
        chain_records(atoms, chain_kinds={"A": "protein"})


def test_a_kind_the_residues_contradict_is_refused():
    atoms = _atoms(("A", "ALA", BACKBONE), ("A", "GLY", BACKBONE))

    with pytest.raises(ChainDeclarationError, match="DNA residues"):
        chain_records(atoms, chain_kinds={"A": "dna"})


def test_modified_and_d_residues_are_protein_residues():
    atoms = _atoms(
        ("A", "ALA", BACKBONE), ("A", "MSE", BACKBONE), ("A", "DAL", BACKBONE)
    )

    records = _by_id(chain_records(atoms, chain_kinds={"A": "protein"}))

    assert records["A"].kind == "protein"


def test_a_residue_the_ccd_does_not_know_cannot_be_verified():
    atoms = _atoms(("A", "ALA", BACKBONE), ("A", "WAT", ("O",)))

    with pytest.raises(ChainDeclarationError, match="not in the CCD"):
        chain_records(atoms, chain_kinds={"A": "protein"})


def test_a_free_amino_acid_can_be_declared_a_ligand():
    atoms = _atoms(("B", "ARG", (*BACKBONE, "CB")))

    records = _by_id(chain_records(atoms, chain_kinds={"B": "ligand"}))

    assert records["B"].kind == "ligand"


def test_a_run_of_amino_acids_is_not_a_ligand():
    atoms = _atoms(("B", "ALA", BACKBONE), ("B", "GLY", BACKBONE))

    with pytest.raises(ChainDeclarationError, match="only on its own"):
        chain_records(atoms, chain_kinds={"B": "ligand"})


def test_a_ligand_chain_holding_water_is_refused():
    atoms = _atoms(("B", "HEM", ("C1", "O1")), ("B", "HOH", ("O",)))

    with pytest.raises(ChainDeclarationError, match="HOH"):
        chain_records(atoms, chain_kinds={"B": "ligand"})


def test_a_declared_water_chain_has_to_be_water():
    water = _atoms(("W", "HOH", ("O",)), ("W", "DOD", ("O",)))
    assert _by_id(chain_records(water, chain_kinds={"W": "water"}))["W"].kind == "water"

    ligand = _atoms(("W", "HEM", ("C1", "O1")))
    with pytest.raises(ChainDeclarationError, match="not water"):
        chain_records(ligand, chain_kinds={"W": "water"})


# -- one chain, one molecule: the annotations have to agree -------------------


def test_a_chain_carrying_two_chain_types_is_refused():
    """Classified by its first atom, the heme would fold as part of the protein."""
    from atomworks.enums import ChainType

    atoms = _atoms(
        ("A", "ALA", BACKBONE),
        ("A", "HEM", ("C1", "C2", "O1")),
        chain_type=[int(ChainType.POLYPEPTIDE_L)] * 4
        + [int(ChainType.NON_POLYMER)] * 3,
    )

    with pytest.raises(MixedChainError, match="NON_POLYMER"):
        chain_records(atoms)


def test_a_chain_mixing_polymer_and_non_polymer_cannot_be_inferred(convert):
    """Not a guess that can be accepted: no single guess describes the chain."""
    atoms = _atoms(
        ("A", "ALA", BACKBONE),
        ("A", "HEM", ("C1", "C2", "O1")),
        is_polymer=[True] * 4 + [False] * 3,
    )

    with pytest.raises(MixedChainError, match="is_polymer"):
        convert(atoms, allow_inferred_chain_kind=True)


# -- what the adapter does with a declaration -----------------------------------


def test_a_declared_water_chain_is_dropped_under_the_water_policy(convert):
    atoms = _atoms(("A", "ALA", BACKBONE), ("W", "HOH", ("O",)))
    report = AdapterReport()

    spi = convert(atoms, chain_kinds={"A": "protein", "W": "water"}, report=report)

    assert [entry.id for entry in spi.sequences] == ["A"]
    assert report.dropped == [("W", "water")]
    with pytest.raises(ValueError, match="water"):
        convert(atoms, chain_kinds={"A": "protein", "W": "water"}, drop_water=False)


def test_a_declared_ligand_still_needs_an_identity(convert):
    """The kind says what a chain is; it does not say which molecule."""
    atoms = _atoms(("A", "ALA", BACKBONE), ("B", "LIG", ("C1", "O1")))
    kinds = {"A": "protein", "B": "ligand"}

    with pytest.raises(LigandIdentityError, match="LIG"):
        convert(atoms, chain_kinds=kinds)

    spec = LigandSpec(chain_id="B", smiles="CO")
    spi = convert(atoms, chain_kinds=kinds, ligands={"B": spec})
    assert [type(entry).__name__ for entry in spi.sequences] == [
        "ProteinInput",
        "LigandInput",
    ]


def test_an_unannotated_chain_is_told_how_to_declare_it(convert):
    with pytest.raises(UnsupportedChainError, match="chain_kinds="):
        convert(_protein_and_ligand())


def test_an_inferred_kind_is_told_how_to_declare_it(convert):
    atoms = _protein_and_ligand(is_polymer=[True] * 8 + [False] * 3)

    with pytest.raises(InferredChainKindError, match="chain_kinds="):
        convert(atoms)


# -- a real author chain --------------------------------------------------------


def _read_101m(data_dir, *, author: bool):
    """101M as biotite reads it: author chains by default, label chains if asked."""
    pdbx = pytest.importorskip("biotite.structure.io.pdbx")
    path = data_dir / "101m_arginine_nh1nh2_swapped.cif"
    if not path.exists():
        pytest.skip(f"fixture missing at {path}")
    return pdbx.get_structure(
        pdbx.CIFFile.read(str(path)), model=1, use_author_fields=author
    )


@pytest.mark.parametrize("override", [False, True], ids=["as-is", "sequence-override"])
def test_an_author_chain_holding_a_protein_its_ligands_and_water_is_refused(
    data_dir, convert, override
):
    """Author chain A of 101M is the protein, a heme, NBN, a sulfate and 138 waters.

    Declared ``protein``, it would fold them as residues of the protein. Under a
    sequence override none of them would reach the model,
    with nothing dropped and nothing raised.
    """
    atoms = _read_101m(data_dir, author=True)
    assert set(np.unique(atoms.chain_id)) == {"A"}
    sequences = {"A": "M" * 154} if override else None

    with pytest.raises(ChainDeclarationError, match="HEM") as refused:
        convert(atoms, chain_kinds={"A": "protein"}, sequences=sequences)
    assert "138 x HOH" in str(refused.value)


def test_the_same_structure_with_a_chain_per_molecule_converts(data_dir, convert, ccd):
    atoms = _read_101m(data_dir, author=False)
    kinds = {"A": "protein", "B": "ligand", "C": "ligand", "D": "ligand", "E": "water"}
    assert set(np.unique(atoms.chain_id)) == set(kinds)
    report = AdapterReport()

    spi = convert(atoms, chain_kinds=kinds, report=report)

    assert [(type(e).__name__, e.id) for e in spi.sequences] == [
        ("ProteinInput", "A"),
        ("LigandInput", "B"),
        ("LigandInput", "C"),
        ("LigandInput", "D"),
    ]
    assert len(spi.sequences[0].sequence) == 154
    assert sorted(ccd for e in spi.sequences[1:] for ccd in e.ccd) == [
        "HEM",
        "NBN",
        "SO4",
    ]
    assert report.dropped == [("E", "water")]
    assert report.inferred_chain_kinds == []
