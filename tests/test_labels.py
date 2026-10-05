"""Source coordinates must land on the model's atom axis, correctly and honestly.

This is the supervision half of training on AtomWorks data: the pipeline otherwise
carries inputs only, and `feats["gt_coords"]` is not the source structure -- it
is built from the prediction input and is zeros at inference.

No loss lives here, so the tests are about the two properties a loss will rely
on: that a matched atom really is the same atom, and that an unmatched one is
masked rather than filled in.
"""

from __future__ import annotations

import numpy as np
import pytest
from ligand_complexes import DOUBLE, SINGLE, protein_and_ligand

from esmfold2_atomworks.data.atomworks_to_esm import (
    _residue_index_map,
    atom_array_to_structure_prediction_input,
    chain_records,
)
from esmfold2_atomworks.data.labels import _model_atom_identities, structure_labels
from esmfold2_atomworks.parity.compare import featurize


def _labels(atoms, chain_info):
    spi = atom_array_to_structure_prediction_input(atoms, chain_info=chain_info)
    from esm.models.esmfold2.prepare_input import prepare_esmfold2_input
    from esm.models.esmfold2.processor import clean_esmfold2_input

    features, chain_infos = prepare_esmfold2_input(clean_esmfold2_input(spi), seed=0)
    records = chain_records(atoms, chain_info=chain_info)
    return structure_labels(
        atoms, features, chain_infos, _residue_index_map(records, chain_info)
    ), features


def test_labels_cover_the_model_atom_axis(parsed, ccd):
    atoms, chain_info = parsed("lysozyme")
    labels, features = _labels(atoms, chain_info)

    import numpy as np

    assert labels.atom_coords.shape == (features["ref_pos"].shape[0], 3)
    assert labels.atom_mask.shape == (features["ref_pos"].shape[0],)

    # Coverage is measured against the model's REAL atoms, not the padded axis.
    # ESMFold2 pads to a multiple of 32, so lysozyme's 1000 atoms sit on a
    # 1024-long axis; dividing by the axis would report 0.977 for a structure
    # that is in fact complete, and `require_coverage=0.99` would reject it.
    real = int(np.asarray(features["atom_attention_mask"]).sum())
    padded = int(np.asarray(features["atom_attention_mask"]).size)
    assert padded > real, "expected the axis to be padded, or this asserts nothing"
    assert labels.n_model_atoms == real
    assert labels.coverage == pytest.approx(labels.matched / real)

    # Lysozyme is fully resolved, so every real model atom must match.
    assert labels.coverage == pytest.approx(1.0), (
        f"{labels.matched}/{real} matched; unmatched: {labels.unmatched_examples}"
    )


def test_matched_coordinates_are_the_source_coordinates(parsed, ccd):
    """Each matched label sits on the source atom of the same name and residue.

    A permutation that lost track of which atom is which would still "cover", so
    the check is per atom: the source atom a label's coordinate belongs to must
    carry the model atom's name, and the model's residue index must differ from
    the source residue's ordinal by one constant.
    """
    import biotite.structure as struc

    atoms, chain_info = parsed("lysozyme")
    labels, features = _labels(atoms, chain_info)

    name_chars = np.asarray(features["ref_atom_name_chars"])
    to_token = np.asarray(features["atom_to_token"])
    residue_of_token = np.asarray(features["residue_index"])
    source_names = np.asarray(atoms.atom_name).astype(str)
    is_start = np.zeros(len(atoms), dtype=int)
    is_start[struc.get_residue_starts(atoms)] = 1
    source_residue = np.cumsum(is_start) - 1

    # Coordinates identify atoms uniquely here, so a label names its source row.
    row_of = {tuple(np.round(row, 3)): i for i, row in enumerate(atoms.coord)}
    assert len(row_of) == len(atoms)

    offsets = set()
    for position in np.where(labels.atom_mask)[0]:
        row = row_of[tuple(np.round(labels.atom_coords[position], 3))]
        model_name = "".join(
            chr(int(code) + 32) for code in name_chars[position] if int(code) != 0
        ).strip()
        assert model_name == source_names[row]
        offsets.add(
            int(residue_of_token[to_token[position]]) - int(source_residue[row])
        )
    assert len(offsets) == 1, f"residue indices are not one constant apart: {offsets}"


def test_unmatched_atoms_are_masked_and_left_nan(parsed, ccd):
    """Never imputed: a placeholder would silently bias any averaging loss."""
    atoms, chain_info = parsed("hemoglobin")
    labels, _features = _labels(atoms, chain_info)

    unmatched = ~labels.atom_mask
    assert unmatched.any(), "expected padding atoms at least"
    assert np.isnan(labels.atom_coords[unmatched]).all()
    assert np.isfinite(labels.atom_coords[labels.atom_mask]).all()


def test_a_deleted_side_chain_becomes_unmatched_rather_than_wrong(parsed, ccd):
    """Removing source atoms must lower coverage, not shift the alignment."""
    atoms, chain_info = parsed("lysozyme")
    full, _ = _labels(atoms, chain_info)

    names = np.asarray(atoms.atom_name).astype(str)
    backbone_only = atoms[np.isin(names, ["N", "CA", "C", "O"])]
    reduced, _ = _labels(backbone_only, chain_info)

    assert reduced.matched < full.matched
    # What remains must still be right, not merely fewer.
    assert np.isfinite(reduced.atom_coords[reduced.atom_mask]).all()
    assert reduced.unmatched_examples


def test_the_pipeline_can_attach_labels(parsed, ccd):
    from esmfold2_atomworks.data.pipelines import build_esmfold2_pipeline

    atoms, chain_info = parsed("lysozyme")
    pipeline = build_esmfold2_pipeline(is_inference=False, seed=0, attach_labels=True)
    out = pipeline(
        {"example_id": "lysozyme", "atom_array": atoms, "chain_info": chain_info}
    )
    assert set(out["labels"]) == {"atom_coords", "atom_mask"}
    assert out["labels"]["atom_coords"].shape[0] == out["feats"]["ref_pos"].shape[0]
    assert out["label_coverage"] == pytest.approx(1.0)


def test_labels_are_off_by_default(parsed, ccd):
    """Inference does not need them and they are not free."""
    from esmfold2_atomworks.data.pipelines import build_esmfold2_pipeline

    atoms, chain_info = parsed("lysozyme")
    out = build_esmfold2_pipeline(is_inference=False, seed=0)(
        {"example_id": "lysozyme", "atom_array": atoms, "chain_info": chain_info}
    )
    assert "labels" not in out


def test_a_coverage_floor_can_be_required(parsed, ccd):
    """A silently-unmatched structure is worse than a refused one."""
    from esmfold2_atomworks.data.pipelines import AttachStructureLabels

    atoms, chain_info = parsed("lysozyme")
    spi = atom_array_to_structure_prediction_input(atoms, chain_info=chain_info)
    features = featurize(spi, seed=0)
    from esm.models.esmfold2.prepare_input import prepare_esmfold2_input
    from esm.models.esmfold2.processor import clean_esmfold2_input

    _f, chain_infos = prepare_esmfold2_input(clean_esmfold2_input(spi), seed=0)

    # Full coverage is attainable, so the floor has to exceed it to trip.
    transform = AttachStructureLabels(require_coverage=1.01)
    with pytest.raises(ValueError, match="below the required"):
        transform(
            {
                "atom_array": atoms,
                "chain_info": chain_info,
                "feats": features,
                "chain_infos": chain_infos,
                "structure_prediction_input": spi,
            }
        )


def test_label_skipped_chains_survives_the_pipeline(parsed, ccd):
    """It is written by the transform, and must not be dropped by SubsetToKeys.

    Reporting a skipped chain into a dict that is then filtered away is the
    same as not reporting it at all.
    """
    from esmfold2_atomworks.data.pipelines import build_esmfold2_pipeline

    atoms, chain_info = parsed("lysozyme")
    out = build_esmfold2_pipeline(is_inference=False, seed=0, attach_labels=True)(
        {"example_id": "lysozyme", "atom_array": atoms, "chain_info": chain_info}
    )
    # Always present, so a consumer need not check for the key, and an empty
    # list means "nothing skipped" rather than "not computed".
    assert out["label_skipped_chains"] == []
    assert out["label_unmatched_smiles_chains"] == {}


def test_the_pipeline_passes_the_smiles_match_option_on():
    from esmfold2_atomworks.data.pipelines import build_esmfold2_pipeline

    with pytest.raises(ValueError, match="on_smiles_match_failure"):
        build_esmfold2_pipeline(
            is_inference=False, attach_labels=True, on_smiles_match_failure="ignore"
        )


# -- SMILES ligands ----------------------------------------------------------
# ESMFold2 names such a ligand's atoms by element and rank (acetic acid: CH3 is
# C8, the carbonyl C is C7, =O is O5, -OH is O6), so a source's name for an atom
# says nothing about which atom it is. A label put on the atom with the same name
# can be on the wrong atom and still cover every atom of the model.


def _attach(atoms, smiles="CC(=O)O", **options):
    pytest.importorskip("atomworks.ml.transforms.base")
    from esm.models.esmfold2.prepare_input import prepare_esmfold2_input
    from esm.models.esmfold2.processor import clean_esmfold2_input

    from esmfold2_atomworks.data.pipelines import AttachStructureLabels
    from esmfold2_atomworks.data.spec import LigandSpec

    spi = atom_array_to_structure_prediction_input(
        atoms,
        chain_kinds={"A": "protein", "B": "ligand"},
        ligands=[LigandSpec(chain_id="B", smiles=smiles)],
    )
    features, chain_infos = prepare_esmfold2_input(clean_esmfold2_input(spi), seed=0)
    data = AttachStructureLabels(**options)(
        {
            "atom_array": atoms,
            "feats": features,
            "chain_infos": chain_infos,
            "structure_prediction_input": spi,
        }
    )
    return data, features, chain_infos


def _label(data, features, chain_infos, name):
    """``(coordinate, mask)`` of the label on the ligand atom ESMFold2 calls *name*."""
    identities = _model_atom_identities(features, chain_infos)
    (position,) = [
        i
        for i, ident in enumerate(identities)
        if ident and ident[0] == "B" and ident[2] == name
    ]
    return data["labels"]["atom_coords"][position], data["labels"]["atom_mask"][
        position
    ]


def _source(atoms, name):
    (row,) = np.flatnonzero(
        (np.asarray(atoms.chain_id) == "B") & (np.asarray(atoms.atom_name) == name)
    )
    return atoms.coord[row]


ACETIC_ACID = [(0, 1, SINGLE), (1, 2, SINGLE), (1, 3, DOUBLE)]  # CH3, C, -OH, =O


def test_labels_follow_the_chemistry_of_a_smiles_ligand_not_its_atom_names(ccd):
    # The source's names are ESMFold2's, but on other atoms: its C7 is the methyl, its
    # C8 the carbonyl, its O5 the hydroxyl and its O6 the carbonyl oxygen.
    atoms = protein_and_ligand(["C7", "C8", "O5", "O6"], ACETIC_ACID)
    data, features, chain_infos = _attach(atoms)

    expected = {
        "C8": "C7",
        "C7": "C8",
        "O6": "O5",
        "O5": "O6",
    }  # ESMFold2 name -> source
    for model_name, source_name in expected.items():
        coordinate, mask = _label(data, features, chain_infos, model_name)
        assert mask
        np.testing.assert_array_equal(coordinate, _source(atoms, source_name))
    assert data["label_coverage"] == pytest.approx(1.0)
    assert data["label_unmatched_smiles_chains"] == {}


def test_a_smiles_ligand_with_esms_own_names_keeps_its_labels(ccd):
    atoms = protein_and_ligand(
        ["C8", "C7", "O5", "O6"], [(0, 1, SINGLE), (1, 2, DOUBLE), (1, 3, SINGLE)]
    )
    data, features, chain_infos = _attach(atoms)
    for name in ("C8", "C7", "O5", "O6"):
        coordinate, _ = _label(data, features, chain_infos, name)
        np.testing.assert_array_equal(coordinate, _source(atoms, name))


def test_atoms_the_graph_cannot_tell_apart_are_labelled_one_to_one(ccd):
    # Fluoroform: the three fluorines are equivalent, so which source fluorine
    # lands on which of ESMFold2's is not determined -- but it is one to one.
    atoms = protein_and_ligand(
        ["F1", "C1", "F2", "F3"], [(1, 0, SINGLE), (1, 2, SINGLE), (1, 3, SINGLE)]
    )
    data, features, chain_infos = _attach(atoms, smiles="FC(F)F")

    ligand = np.asarray(atoms.chain_id) == "B"
    labelled = [
        _label(data, features, chain_infos, name)
        for name in _ligand_names(features, chain_infos)
    ]
    assert all(mask for _, mask in labelled)
    assert sorted(map(tuple, (c for c, _ in labelled))) == sorted(
        map(tuple, atoms.coord[ligand])
    )


def _ligand_names(features, chain_infos):
    return [
        i[2] for i in _model_atom_identities(features, chain_infos) if i and i[0] == "B"
    ]


def test_a_smiles_ligand_that_cannot_be_matched_is_refused_or_masked_by_choice(ccd):
    from esmfold2_atomworks.data.spec import LigandIdentityError

    # A chain of four atoms, not acetic acid.
    atoms = protein_and_ligand(
        ["C7", "C8", "O5", "O6"], [(0, 1, SINGLE), (1, 2, SINGLE), (2, 3, SINGLE)]
    )
    with pytest.raises(LigandIdentityError, match="not that of the SMILES"):
        _attach(atoms)

    data, features, chain_infos = _attach(atoms, on_smiles_match_failure="mask")
    assert set(data["label_unmatched_smiles_chains"]) == {"B"}
    assert "not that of the SMILES" in data["label_unmatched_smiles_chains"]["B"]
    ligand = [
        i
        for i, ident in enumerate(_model_atom_identities(features, chain_infos))
        if ident and ident[0] == "B"
    ]
    assert not data["labels"]["atom_mask"][ligand].any()
    assert np.isnan(data["labels"]["atom_coords"][ligand]).all()
    assert data["labels"]["atom_mask"].sum() == 12  # the protein's atoms are untouched
    assert data["label_coverage"] < 1.0


def test_the_unmatched_option_names_what_it_accepts():
    pytest.importorskip("atomworks.ml.transforms.base")
    from esmfold2_atomworks.data.pipelines import AttachStructureLabels

    with pytest.raises(ValueError, match="on_smiles_match_failure"):
        AttachStructureLabels(on_smiles_match_failure="ignore")


def test_a_bond_and_a_label_agree_on_the_atom_they_mean(ccd):
    """One mapping for both: the atom a declared bond lands on carries the
    coordinate of the source atom the bond was declared on."""
    from esmfold2_atomworks.data.bonds import _atom_names_by_residue

    atoms = protein_and_ligand(
        ["C7", "C8", "O5", "O6"], ACETIC_ACID, bond_to=2
    )  # the hydroxyl
    data, features, chain_infos = _attach(atoms)

    from esmfold2_atomworks.data.atomworks_to_esm import (
        atom_array_to_structure_prediction_input as adapt,
    )
    from esmfold2_atomworks.data.spec import LigandSpec

    spi = adapt(
        atoms,
        chain_kinds={"A": "protein", "B": "ligand"},
        ligands=[LigandSpec(chain_id="B", smiles="CC(=O)O")],
    )
    (bond,) = spi.covalent_bonds
    placed = _atom_names_by_residue(features, chain_infos)[("B", bond.res_idx2)][
        bond.atom_idx2
    ]
    coordinate, _ = _label(data, features, chain_infos, placed)
    np.testing.assert_array_equal(coordinate, _source(atoms, "O5"))
