"""One case per implemented branch of the adapter's input surface.

The adapter has code for ``DNAInput``, ``RNAInput`` and SMILES ligands, but
until these tests existed no fixture exercised them, so they were untested
rather than known-good. The repo's rule is that what is supported is tested and
what is not supported is refused explicitly; this closes the first half.

Structures are built with AtomWorks' own component assembler rather than taken
from the PDB, because the point is to cover the branches, and a synthetic
duplex is a smaller and more legible fixture than a real nucleoprotein complex.
"""

from __future__ import annotations

import numpy as np
import pytest

from esmfold2_atomworks.data.atomworks_to_esm import (
    AdapterReport,
    atom_array_to_structure_prediction_input,
    chain_records,
)
from esmfold2_atomworks.data.spec import LigandIdentityError, LigandSpec
from esmfold2_atomworks.parity.compare import compare_features, featurize

# esm/models/esmfold2/constants.py
MOL_TYPE_PROTEIN, MOL_TYPE_DNA, MOL_TYPE_RNA, MOL_TYPE_NONPOLYMER = 0, 1, 2, 3

PROTEIN = "MKTAYIAKQRQISFVKSHFSRQ"


@pytest.fixture(scope="module")
def build():
    """``components -> AtomArray`` via AtomWorks' inference assembler."""
    pytest.importorskip("atomworks.io.tools.inference")
    from atomworks.io.tools.inference import components_to_atom_array

    return components_to_atom_array


def _components():
    from atomworks.io.tools.inference import DNA, RNA, Protein, SmilesComponent

    return DNA, RNA, Protein, SmilesComponent


def _mol_types(spi) -> set[int]:
    return set(np.asarray(featurize(spi)["mol_type"]).tolist())


def test_dna_duplex(build, ccd):
    DNA, _RNA, _Protein, _Smiles = _components()
    atoms = build(
        [DNA(seq="ATGCATGC", chain_id="A"), DNA(seq="GCATGCAT", chain_id="B")]
    )

    assert [r.kind for r in chain_records(atoms)] == ["dna", "dna"]
    spi = atom_array_to_structure_prediction_input(atoms)
    assert [type(s).__name__ for s in spi.sequences] == ["DNAInput", "DNAInput"]
    assert [s.sequence for s in spi.sequences] == ["ATGCATGC", "GCATGCAT"]
    assert _mol_types(spi) == {MOL_TYPE_DNA}


def test_rna_monomer(build, ccd):
    _DNA, RNA, _Protein, _Smiles = _components()
    atoms = build([RNA(seq="AUGCAUGC", chain_id="A")])

    spi = atom_array_to_structure_prediction_input(atoms)
    assert [type(s).__name__ for s in spi.sequences] == ["RNAInput"]
    assert spi.sequences[0].sequence == "AUGCAUGC"
    assert _mol_types(spi) == {MOL_TYPE_RNA}


def test_protein_nucleic_complex(build, ccd):
    """Both molecule types must survive in one input, and be labelled as such."""
    DNA, _RNA, Protein, _Smiles = _components()
    atoms = build(
        [Protein(seq=PROTEIN, chain_id="A"), DNA(seq="ATGCATGC", chain_id="B")]
    )

    spi = atom_array_to_structure_prediction_input(atoms)
    assert [type(s).__name__ for s in spi.sequences] == ["ProteinInput", "DNAInput"]
    assert _mol_types(spi) == {MOL_TYPE_PROTEIN, MOL_TYPE_DNA}


def test_a_smiles_built_ligand_is_refused_without_a_declaration(build, ccd):
    """AtomWorks names it ``L:0``, which is not a CCD code and must not be used.

    Before this was checked the placeholder reached the featurizer and failed
    there with "CCD component L:0 not found" -- a message that names neither
    the chain nor the remedy.
    """
    _DNA, _RNA, Protein, SmilesComponent = _components()
    atoms = build(
        [
            Protein(seq=PROTEIN, chain_id="A"),
            SmilesComponent(smiles="c1ccccc1O", chain_id="B"),
        ]
    )
    with pytest.raises(LigandIdentityError, match="not in the chemical component"):
        atom_array_to_structure_prediction_input(atoms)


def test_a_declared_smiles_ligand_folds(build, ccd):
    """Declaring it is the supported route, and it must reach the model."""
    _DNA, _RNA, Protein, SmilesComponent = _components()
    smiles = "c1ccccc1O"
    atoms = build(
        [
            Protein(seq=PROTEIN, chain_id="A"),
            SmilesComponent(smiles=smiles, chain_id="B"),
        ]
    )
    report = AdapterReport()
    spi = atom_array_to_structure_prediction_input(
        atoms,
        ligands=(LigandSpec(chain_id="B", smiles=smiles),),
        report=report,
    )
    ligand = spi.sequences[1]
    assert ligand.smiles == smiles
    assert ligand.ccd is None
    assert _mol_types(spi) == {MOL_TYPE_PROTEIN, MOL_TYPE_NONPOLYMER}


def test_a_smiles_ligand_is_reproducible_at_a_fixed_seed(build, ccd):
    """Its conformer is embedded by RDKit at call time, so the seed matters.

    This is the one input for which featurization is not seed-independent, and
    the coverage matrix in docs/02 says so. Same seed must give identical
    tensors; that is what makes a parity comparison possible at all.
    """
    _DNA, _RNA, _Protein, SmilesComponent = _components()
    smiles = "c1ccccc1O"
    atoms = build([SmilesComponent(smiles=smiles, chain_id="B")])
    spi = atom_array_to_structure_prediction_input(
        atoms, ligands=(LigandSpec(chain_id="B", smiles=smiles),)
    )
    diff = compare_features(featurize(spi, seed=0), featurize(spi, seed=0))
    assert diff.identical, diff.report()
