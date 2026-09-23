"""No silent semantic degradation: the adapter's contract, tested as a whole.

The failure this guards against has one shape: a degradation *recorded* in
`AdapterReport` rather than raised -- and `fold_atom_array`, the path the README
quickstart uses, returns no report. Recorded-but-silent is therefore invisible
exactly where it matters.

So the policy is tested here as a unit: every degradation the adapter can
detect raises by default, each is accepted only by name, and the table that
lists them is checked for completeness rather than trusted.
"""

from __future__ import annotations

import inspect

import pytest

from esmfold2_foundry.data import spec
from esmfold2_foundry.data.atomworks_to_esm import (
    DEGRADATIONS,
    AdapterReport,
    allow_kwargs,
    atom_array_to_structure_prediction_input,
)

# -- the table itself --------------------------------------------------------


@pytest.mark.offline
def test_every_degradation_has_an_opt_in_that_defaults_to_strict():
    """The structural guard: a new degradation cannot be added half-way.

    Each entry must be a keyword of the adapter, and that keyword must default
    to False -- otherwise the degradation happens unless someone remembers to
    forbid it, which is the failure mode this whole module exists to prevent.
    """
    parameters = inspect.signature(atom_array_to_structure_prediction_input).parameters
    for name in DEGRADATIONS:
        keyword = f"allow_{name}"
        assert keyword in parameters, f"{name}: the adapter has no {keyword}"
        assert parameters[keyword].default is False, f"{keyword} is not strict"


@pytest.mark.offline
def test_every_degradation_raises_a_public_error():
    for name, error in DEGRADATIONS.items():
        assert issubclass(error, Exception), name
        assert error.__name__ in spec.__all__, f"{error.__name__} is not exported"


@pytest.mark.offline
def test_allow_kwargs_maps_names_to_keywords():
    assert allow_kwargs(["unsupported_chains"]) == {"allow_unsupported_chains": True}
    assert allow_kwargs("inferred_chain_kind") == {"allow_inferred_chain_kind": True}
    assert allow_kwargs(None) == {}
    assert allow_kwargs(()) == {}


@pytest.mark.offline
def test_a_misspelt_opt_in_is_refused_rather_than_ignored():
    """Ignoring it would leave a caller believing they had accepted something."""
    with pytest.raises(ValueError, match="unknown degradation"):
        allow_kwargs(["unsuported_chains"])


# -- inferred chain kind -----------------------------------------------------


def _dna_without_chain_type():
    pytest.importorskip("atomworks.io.tools.inference")
    from atomworks.io.tools.inference import DNA, components_to_atom_array

    atoms = components_to_atom_array([DNA(seq="ATGCATGC", chain_id="A")])
    atoms.del_annotation("chain_type")
    return atoms


def test_a_dna_chain_without_chain_type_is_not_folded_as_protein(ccd):
    """Accepted, it would become ``ProteinInput(sequence="XXXXXXXX")``."""
    with pytest.raises(spec.InferredChainKindError, match="protein"):
        atom_array_to_structure_prediction_input(_dna_without_chain_type())


def test_the_inference_can_be_accepted_by_name(ccd):
    """And accepting it shows why it is off by default."""
    report = AdapterReport()
    spi = atom_array_to_structure_prediction_input(
        _dna_without_chain_type(), allow_inferred_chain_kind=True, report=report
    )
    assert type(spi.sequences[0]).__name__ == "ProteinInput"
    assert report.inferred_chain_kinds == ["A"]


def test_a_chain_with_chain_type_is_never_treated_as_inferred(parsed, ccd):
    atoms, chain_info = parsed("hemoglobin")
    report = AdapterReport()
    atom_array_to_structure_prediction_input(
        atoms, chain_info=chain_info, report=report
    )
    assert report.inferred_chain_kinds == []


# -- unplaceable modifications -----------------------------------------------


def _misaligned_modified(parsed):
    """1A8O (4 x MSE) against an entity record two residues longer, without res_name."""
    atoms, chain_info = parsed("modified")
    sequence = chain_info["A"]["processed_entity_canonical_sequence"]
    return atoms, {"A": {"processed_entity_canonical_sequence": "GG" + sequence}}


def test_a_modification_that_cannot_be_placed_raises(parsed, ccd):
    """Proceeding would fold four methionines where there are selenomethionines."""
    atoms, stale = _misaligned_modified(parsed)
    with pytest.raises(spec.ModificationResolutionError, match="MSE"):
        atom_array_to_structure_prediction_input(atoms, chain_info=stale)


def test_the_parent_residue_approximation_can_be_accepted_by_name(parsed, ccd):
    atoms, stale = _misaligned_modified(parsed)
    report = AdapterReport()
    spi = atom_array_to_structure_prediction_input(
        atoms,
        chain_info=stale,
        allow_unplaceable_modifications=True,
        report=report,
    )
    assert spi.sequences[0].modifications is None
    assert report.unplaceable_modifications == ["A"]


def test_misalignment_without_any_nonstandard_residue_is_not_an_alarm(parsed, ccd):
    """Nothing needed placing, so nothing was approximated.

    Raising here would be a false positive on every standard chain whose entity
    record happens to be longer than its model -- which is most of the PDB.
    """
    atoms, chain_info = parsed("lysozyme")
    sequence = chain_info["A"]["processed_entity_canonical_sequence"]
    stale = {"A": {"processed_entity_canonical_sequence": "GG" + sequence}}
    report = AdapterReport()
    atom_array_to_structure_prediction_input(atoms, chain_info=stale, report=report)
    assert report.unplaceable_modifications == []


# -- the report says what happened, not what was refused --------------------


def test_a_refused_modification_leaves_no_trace_in_the_report(parsed, ccd):
    """The degradation fields describe the input returned; after a raise, none was."""
    atoms, stale = _misaligned_modified(parsed)
    report = AdapterReport()
    with pytest.raises(spec.ModificationResolutionError):
        atom_array_to_structure_prediction_input(atoms, chain_info=stale, report=report)
    assert report.unplaceable_modifications == []


def test_a_chain_with_no_annotation_at_all_says_so(ccd):
    """Neither chain_type nor is_polymer: nothing to infer from, so unsupported."""
    atoms = _dna_without_chain_type()
    atoms.del_annotation("is_polymer")
    report = AdapterReport()
    with pytest.raises(spec.UnsupportedChainError, match="no chain_type or is_polymer"):
        atom_array_to_structure_prediction_input(atoms, report=report)
    assert report.dropped == []


# -- empty sequence ----------------------------------------------------------


def test_an_empty_override_in_a_multichain_structure_is_not_a_silent_drop(parsed, ccd):
    """Alone it would fail anyway, with nothing left to fold; beside others it would vanish."""
    atoms, chain_info = parsed("hemoglobin")
    with pytest.raises(ValueError, match="empty override"):
        atom_array_to_structure_prediction_input(
            atoms, chain_info=chain_info, sequences={"B": ""}
        )


# -- every public path can express the policy -------------------------------


def test_the_pipeline_is_strict_and_accepts_names(ccd):
    """Otherwise its error would name a remedy the pipeline cannot pass."""
    pytest.importorskip("atomworks.ml.transforms.base")
    from esmfold2_foundry.data.pipelines import build_esmfold2_pipeline

    example = {"example_id": "dna", "atom_array": _dna_without_chain_type()}
    with pytest.raises(Exception, match="chain_type"):
        build_esmfold2_pipeline(is_inference=True, seed=0)(dict(example))

    out = build_esmfold2_pipeline(
        is_inference=True, seed=0, allow=["inferred_chain_kind"]
    )(dict(example))
    assert out["adapter_report"].inferred_chain_kinds == ["A"]


def test_a_misspelt_pipeline_opt_in_fails_at_build_time():
    pytest.importorskip("atomworks.ml.transforms.base")
    from esmfold2_foundry.data.pipelines import build_esmfold2_pipeline

    with pytest.raises(ValueError, match="unknown degradation"):
        build_esmfold2_pipeline(is_inference=True, allow=["inferred_chain_kinds"])


def test_the_engine_validates_its_policy_before_loading_a_model():
    """A typo must fail in milliseconds, not after the weights download."""
    from esmfold2_foundry.inference.engine import ESMFold2InferenceEngine

    engine = ESMFold2InferenceEngine(allow=["unsupported_chains"])
    assert engine.adapter_policy == {"allow_unsupported_chains": True}
    assert engine._model is None, "construction must not load the model"

    with pytest.raises(ValueError, match="unknown degradation"):
        ESMFold2InferenceEngine(allow=["bogus"])


def test_the_engine_records_what_it_accepted_per_output():
    """An opt-in says a degradation is acceptable, not which structure it hit."""
    from esmfold2_foundry.inference.engine import _accepted_degradations

    clean = AdapterReport()
    clean.dropped.append(("W", "water"))  # policy, not a degradation
    assert _accepted_degradations(clean) == {}

    degraded = AdapterReport()
    degraded.dropped.append(("X", "unsupported chain_type 1"))
    degraded.unplaceable_modifications.append("A")
    recorded = _accepted_degradations(degraded)["adapter.degradations"]
    assert recorded == {"dropped_chains": ["X"], "unplaceable_modifications": ["A"]}


def test_the_cli_refuses_an_unknown_opt_in_before_loading_anything():
    typer_testing = pytest.importorskip("typer.testing")
    from esmfold2_foundry.cli import app

    result = typer_testing.CliRunner().invoke(
        app, ["fold", "missing.cif", "--allow", "bogus"]
    )
    assert result.exit_code != 0
    assert "unknown degradation" in str(result.exception)
