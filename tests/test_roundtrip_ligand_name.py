"""The caller's ligand residue name survives ``fold_atom_array``.

ESMFold2 writes ``LIG`` on a ligand it was given as SMILES, and anything that
matches ligands by name needs the caller's name instead. ``fold`` and the
adapter are stubbed, so this tests the plumbing, not the model.
"""

from __future__ import annotations

import numpy as np
import pytest

from esmfold2_atomworks.model.esmfold2 import AtomWorksESMFold2

pytest.importorskip("biotite.structure")


class _Metadata:
    def __init__(self):
        self.chain_lookup = [{0: "A", 1: "B"}]


class _Complex:
    """One alanine and a two-atom ligand, in the flat form the decoder emits."""

    def __init__(self):
        self.sequence = np.asarray(["ALA", "LIG"], dtype="U5")
        self.atom_names = np.asarray(["N", "CA", "C", "C1", "O1"], dtype=object)
        self.atom_elements = np.asarray(["N", "C", "C", "C", "O"], dtype=object)
        self.atom_hetero = np.asarray([False, False, False, True, True])
        self.token_to_atoms = np.asarray([[0, 3], [3, 5]], dtype=np.int32)
        self.chain_id = np.asarray([0, 1])
        self.atom_positions = np.zeros((5, 3), dtype=np.float32)
        self.metadata = _Metadata()


class _Result:
    def __init__(self):
        self.complex = _Complex()


def _source():
    """A source structure with no ``chain_type``, so none is copied back."""
    import biotite.structure as struc

    return struc.AtomArray(0)


@pytest.fixture
def model(monkeypatch):
    """An ``AtomWorksESMFold2`` that loads no weights and converts nothing.

    ``fold`` counts its calls, so a test can tell whether folding happened.
    """
    model = AtomWorksESMFold2.__new__(AtomWorksESMFold2)
    model.folds = 0

    def fold(spi, config=None, **overrides):
        model.folds += 1
        return _Result()

    monkeypatch.setattr(model, "fold", fold, raising=False)
    monkeypatch.setattr(
        "esmfold2_atomworks.data.atomworks_to_esm.atom_array_to_structure_prediction_input",
        lambda atoms, **kwargs: object(),
    )
    return model


def _names(atoms, *, hetero: bool) -> set[str]:
    mask = np.asarray(atoms.hetero, dtype=bool)
    return {str(name) for name in atoms.res_name[mask if hetero else ~mask]}


def test_the_callers_name_reaches_the_returned_structure(model):
    atoms, _ = model.fold_atom_array(_source(), ligand_residue_name="PHT")

    assert _names(atoms, hetero=True) == {"PHT"}
    assert _names(atoms, hetero=False) == {"ALA"}


def test_without_a_name_the_models_label_is_kept(model):
    atoms, _ = model.fold_atom_array(_source())

    assert _names(atoms, hetero=True) == {"LIG"}


def test_every_sample_of_a_multi_sample_fold_is_relabelled(model, monkeypatch):
    monkeypatch.setattr(model, "fold", lambda spi, config=None, **kw: [_Result()] * 2)

    arrays, results = model.fold_atom_array(_source(), ligand_residue_name="PHT")

    assert len(arrays) == len(results) == 2
    assert all(_names(atoms, hetero=True) == {"PHT"} for atoms in arrays)


def test_a_name_that_cannot_be_kept_is_refused_before_folding(model):
    with pytest.raises(ValueError, match="one to five characters"):
        model.fold_atom_array(_source(), ligand_residue_name="LIGAND")

    assert model.folds == 0


class _TwoLigandComplex(_Complex):
    """One alanine, a SMILES ligand on B and a zinc on C."""

    def __init__(self):
        super().__init__()
        self.sequence = np.asarray(["ALA", "LIG", "ZN"], dtype="U5")
        self.atom_names = np.asarray(["N", "CA", "C", "C1", "O1", "ZN"], dtype=object)
        self.atom_elements = np.asarray(["N", "C", "C", "C", "O", "ZN"], dtype=object)
        self.atom_hetero = np.asarray([False, False, False, True, True, True])
        self.token_to_atoms = np.asarray([[0, 3], [3, 5], [5, 6]], dtype=np.int32)
        self.chain_id = np.asarray([0, 1, 2])
        self.atom_positions = np.zeros((6, 3), dtype=np.float32)
        self.metadata.chain_lookup = [{0: "A", 1: "B", 2: "C"}]


def _by_chain(atoms) -> dict[str, set[str]]:
    return {
        str(chain): {str(n) for n in atoms.res_name[atoms.chain_id == chain]}
        for chain in np.unique(atoms.chain_id)
    }


@pytest.fixture
def two_ligands(model, monkeypatch):
    def fold(spi, config=None, **overrides):
        model.folds += 1
        result = _Result()
        result.complex = _TwoLigandComplex()
        return result

    monkeypatch.setattr(model, "fold", fold, raising=False)
    return model


def test_each_declared_ligand_keeps_its_own_name(two_ligands):
    from esmfold2_atomworks.data.spec import LigandSpec

    ligands = [
        LigandSpec(chain_id="B", smiles="CCO", residue_name="ETO"),
        LigandSpec(chain_id="C", ccd=("ZN",)),
    ]
    atoms, _ = two_ligands.fold_atom_array(
        _source(), adapter_kwargs={"ligands": ligands}
    )

    assert _by_chain(atoms) == {"A": {"ALA"}, "B": {"ETO"}, "C": {"ZN"}}


def test_a_blanket_name_that_contradicts_a_declaration_is_refused(two_ligands):
    from esmfold2_atomworks.data.spec import LigandSpec

    ligands = [LigandSpec(chain_id="B", smiles="CCO", residue_name="ETO")]
    with pytest.raises(ValueError, match="contradicts the LigandSpec"):
        two_ligands.fold_atom_array(
            _source(), ligand_residue_name="PHT", adapter_kwargs={"ligands": ligands}
        )

    assert two_ligands.folds == 0


@pytest.mark.parametrize(
    "spec_kwargs",
    [
        {"ccd": ("ZN",)},
        {"ccd": ("NAG", "BMA")},
        {"smiles": "CCO"},
    ],
)
def test_a_blanket_name_cannot_overwrite_a_declared_default(two_ligands, spec_kwargs):
    from esmfold2_atomworks.data.spec import LigandSpec

    ligands = [LigandSpec(chain_id="C", **spec_kwargs)]
    with pytest.raises(ValueError, match="contradicts the LigandSpec"):
        two_ligands.fold_atom_array(
            _source(), ligand_residue_name="PHT", adapter_kwargs={"ligands": ligands}
        )

    assert two_ligands.folds == 0


def test_a_blanket_name_that_restates_every_declaration_is_accepted(two_ligands):
    from esmfold2_atomworks.data.spec import LigandSpec

    ligands = [LigandSpec(chain_id="B", smiles="CCO", residue_name="PHT")]
    atoms, _ = two_ligands.fold_atom_array(
        _source(), ligand_residue_name="PHT", adapter_kwargs={"ligands": ligands}
    )

    assert _by_chain(atoms)["B"] == {"PHT"}


def test_a_multi_component_declaration_is_left_as_the_model_wrote_it(two_ligands):
    from esmfold2_atomworks.data.spec import LigandSpec

    spec = LigandSpec(chain_id="B", ccd=("NAG", "NAG"))
    assert spec.label is None
    atoms, _ = two_ligands.fold_atom_array(
        _source(), adapter_kwargs={"ligands": [spec]}
    )

    assert _by_chain(atoms)["B"] == {"LIG"}


def test_one_name_for_several_components_is_refused():
    from esmfold2_atomworks.data.spec import LigandSpec

    with pytest.raises(ValueError, match="cannot label 2 CCD components"):
        LigandSpec(chain_id="B", ccd=("NAG", "BMA"), residue_name="GLY")


def test_a_declared_chain_missing_from_the_output_is_refused(two_ligands):
    from esmfold2_atomworks.data.spec import LigandSpec

    with pytest.raises(ValueError, match="not in the folded structure"):
        two_ligands.fold_atom_array(
            _source(),
            adapter_kwargs={"ligands": [LigandSpec(chain_id="D", ccd=("HEM",))]},
        )
