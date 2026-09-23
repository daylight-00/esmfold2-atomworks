"""The milestone: ``F(AtomArray)`` featurizes identically to the native input.

These run on CPU, need no weights, and are exact. ``prepare_esmfold2_input`` is
a pure function of the ``StructurePredictionInput``, so two inputs that
featurize to the same 29 tensors produce the same prediction by construction --
which is why this, and not a tolerance check on a sampled structure, is the
primary evidence that the adapter is faithful.
"""

from __future__ import annotations

import pytest

from esmfold2_foundry.data.atomworks_to_esm import (
    atom_array_to_structure_prediction_input,
)
from esmfold2_foundry.parity.compare import (
    MODEL_CONSUMED_FEATURES,
    compare_features,
    featurize,
)


def _native(sequences):
    from esm.models.esmfold2.types import StructurePredictionInput

    return StructurePredictionInput(sequences=list(sequences))


def _canonical(chain_info, chain_id):
    for key, value in chain_info.items():
        if str(key) == chain_id:
            return value["processed_entity_canonical_sequence"]
    raise KeyError(chain_id)


@pytest.mark.parametrize("fixture", ["lysozyme", "hemoglobin", "zinc", "flavoprotein"])
def test_adapter_featurizes_identically_to_a_hand_written_input(parsed, ccd, fixture):
    """Every tensor matches, for monomer, multimer, metal and cofactor cases."""
    from esm.models.esmfold2.types import LigandInput, ProteinInput

    atoms, chain_info = parsed(fixture)
    adapted = atom_array_to_structure_prediction_input(atoms, chain_info=chain_info)

    # Rebuild the same system the way a user writes it by hand today: one entry
    # per chain, sequences typed out, ligands named by CCD code.
    native_entries = []
    for entry in adapted.sequences:
        if isinstance(entry, ProteinInput):
            native_entries.append(
                ProteinInput(
                    id=entry.id,
                    sequence=_canonical(chain_info, entry.id),
                    modifications=entry.modifications,
                )
            )
        else:
            native_entries.append(LigandInput(id=entry.id, ccd=entry.ccd))

    diff = compare_features(featurize(_native(native_entries)), featurize(adapted))
    assert diff.ok, diff.report()
    assert diff.identical, diff.report()
    assert MODEL_CONSUMED_FEATURES <= set(diff.compared)


def test_modified_residue_survives_featurization(parsed, ccd):
    """A declared MSE changes tokenization -- and does so identically both ways."""
    from esm.models.esmfold2.types import ProteinInput

    atoms, chain_info = parsed("modified")
    adapted = atom_array_to_structure_prediction_input(atoms, chain_info=chain_info)
    native = _native(
        [
            ProteinInput(
                id="A",
                sequence=_canonical(chain_info, "A"),
                modifications=adapted.sequences[0].modifications,
            )
        ]
    )
    diff = compare_features(featurize(native), featurize(adapted))
    assert diff.identical, diff.report()


def test_declaring_a_modification_actually_changes_the_features(parsed, ccd):
    """Guards the test above from being vacuous.

    If declaring ``MSE`` made no difference, the parity assertion would pass
    whatever the adapter did with modifications. It must change tokenization:
    a modified residue becomes one token per atom.
    """
    atoms, chain_info = parsed("modified")
    with_mods = atom_array_to_structure_prediction_input(
        atoms, chain_info=chain_info, emit_modifications=True
    )
    without = atom_array_to_structure_prediction_input(
        atoms, chain_info=chain_info, emit_modifications=False
    )
    n_with = featurize(with_mods)["res_type"].shape[0]
    n_without = featurize(without)["res_type"].shape[0]
    assert n_with > n_without, (
        f"declaring 4 MSE did not change the token count ({n_with} vs {n_without}); "
        "the parity test above would then be vacuous"
    )


def test_chain_order_is_stable(parsed, ccd):
    """Entity numbering follows input order, so the order must be reproducible."""
    atoms, chain_info = parsed("hemoglobin")
    first = atom_array_to_structure_prediction_input(atoms, chain_info=chain_info)
    second = atom_array_to_structure_prediction_input(atoms, chain_info=chain_info)
    assert [s.id for s in first.sequences] == [s.id for s in second.sequences]
    diff = compare_features(featurize(first), featurize(second))
    assert diff.identical, diff.report()
