"""AtomWorks transform pipeline that emits ESMFold2 features.

This is what lets a Foundry dataset -- PDB, AFDB, a distillation parquet, a
PLINDER protein-ligand corpus -- feed ESMFold2 without any of them knowing about
each other. The pipeline is an ordinary ``atomworks.ml.transforms.Compose``, so
it composes with the crop, filter and MSA transforms the other Foundry models
already use.

**The featurizer is one transform, and it goes last.** ESMFold2 builds its own
tensors from a ``StructurePredictionInput``; it does not consume AtomWorks'
``feats`` dict, because their conventions differ (see
:mod:`esmfold2_foundry.data.atomworks_to_esm`). So the pipeline uses AtomWorks
for everything up to and including structure selection, then converts once, at
the end. Transforms that write ``data["feats"]`` in AF3 terms --
``AggregateFeaturesLikeAF3`` and friends -- are deliberately not part of it;
they would compute a featurization nothing here reads.

The consequence worth stating plainly: cropping composes, but a cropped chain is
a *different sequence*, and ESMFold2 folds sequences. Crop before the featurizer
and you fold the crop, which is usually what training wants and almost never
what an evaluation wants.

Importing this module requires ``atomworks.ml``. The adapter itself
(:mod:`esmfold2_foundry.data.atomworks_to_esm`) does not, so a caller that only
wants ``AtomArray -> StructurePredictionInput`` need not pay for it.
"""

from __future__ import annotations

from typing import Any

from atomworks.ml.transforms._checks import check_contains_keys
from atomworks.ml.transforms.base import Compose, SubsetToKeys, Transform

__all__ = [
    "AttachStructureLabels",
    "FeaturizeForESMFold2",
    "StructurePredictionInputTransform",
    "build_esmfold2_pipeline",
]


class StructurePredictionInputTransform(Transform):
    """``data["atom_array"]`` -> ``data["structure_prediction_input"]``.

    Reads ``chain_info`` from the example when the loader supplied it, which it
    does for anything parsed by ``atomworks.io``. Without it, sequences fall
    back to the residues actually present -- a weaker guarantee, for the reason
    given in :func:`~esmfold2_foundry.data.atomworks_to_esm.sequence_of_chain`.
    """

    def __init__(
        self,
        *,
        ligands: dict[str, Any] | tuple[Any, ...] = (),
        allow_undeclared_ccd_ligands: bool = True,
        emit_modifications: bool = True,
        keep_report: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.ligands = ligands
        self.allow_undeclared_ccd_ligands = allow_undeclared_ccd_ligands
        self.emit_modifications = emit_modifications
        self.keep_report = keep_report

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        from esmfold2_foundry.data.atomworks_to_esm import (
            AdapterReport,
            atom_array_to_structure_prediction_input,
        )

        report = AdapterReport()
        data["structure_prediction_input"] = atom_array_to_structure_prediction_input(
            data["atom_array"],
            chain_info=data.get("chain_info"),
            msas=data.get("msas"),
            ligands=self.ligands,
            allow_undeclared_ccd_ligands=self.allow_undeclared_ccd_ligands,
            emit_modifications=self.emit_modifications,
            report=report,
        )
        if self.keep_report:
            # Kept so a training run can be asked afterwards what it dropped,
            # rather than only what it used.
            data["adapter_report"] = report
        return data


class FeaturizeForESMFold2(Transform):
    """``structure_prediction_input`` -> ``data["feats"]``, unbatched, on CPU.

    ``chain_infos`` is stored beside the features because it is the only record
    of ``token -> (chain, residue, atom span)``, and decoding a prediction back
    to a structure is impossible without it.
    """

    def __init__(self, *, seed: int | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.seed = seed

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["structure_prediction_input"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        from esm.models.esmfold2.prepare_input import prepare_esmfold2_input
        from esm.models.esmfold2.processor import clean_esmfold2_input

        features, chain_infos = prepare_esmfold2_input(
            clean_esmfold2_input(data["structure_prediction_input"]), seed=self.seed
        )
        data["feats"] = features
        data["chain_infos"] = chain_infos
        return data


class AttachStructureLabels(Transform):
    """Add ``data["labels"]``: the source coordinates on the model's atom axis.

    Opt-in, because inference does not need it and it is not free. It supplies
    supervision *targets* and deliberately no loss -- see
    :mod:`esmfold2_foundry.data.labels` for why that split is where it is.

    Must run after :class:`FeaturizeForESMFold2`, since the alignment is against
    the atom ordering that featurization produces.
    """

    def __init__(self, *, require_coverage: float | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.require_coverage = require_coverage

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array", "feats", "chain_infos"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        from esmfold2_foundry.data.atomworks_to_esm import (
            AdapterReport,
            _residue_index_map,
            chain_records,
        )
        from esmfold2_foundry.data.labels import structure_labels

        atoms = data["atom_array"]
        records = chain_records(atoms, chain_info=data.get("chain_info"))
        report = AdapterReport()
        labels = structure_labels(
            atoms,
            data["feats"],
            data["chain_infos"],
            _residue_index_map(records, data.get("chain_info"), report),
        )
        # Always set, so a consumer can read it without a key check and an
        # empty list means "nothing skipped" rather than "not computed".
        data["label_skipped_chains"] = list(report.unrepresentable_insertion_codes)
        if (
            self.require_coverage is not None
            and labels.coverage < self.require_coverage
        ):
            raise ValueError(
                f"structure labels cover {labels.coverage:.1%} of the model's atoms, "
                f"below the required {self.require_coverage:.1%}; unmatched examples: "
                f"{labels.unmatched_examples}"
            )
        data["labels"] = labels.as_dict()
        data["label_coverage"] = labels.coverage
        return data


def build_esmfold2_pipeline(
    *,
    is_inference: bool,
    seed: int | None = None,
    ligands: dict[str, Any] | tuple[Any, ...] = (),
    allow_undeclared_ccd_ligands: bool = True,
    emit_modifications: bool = True,
    attach_labels: bool = False,
    pre_transforms: list[Transform] | None = None,
    keys_to_keep: list[str] | None = None,
) -> Compose:
    """Compose the ESMFold2 training/inference pipeline.

    Args:
        is_inference: keeps ``atom_array`` in the output, as Foundry's pipelines
            do, so a prediction can be compared against its input.
        seed: conformer-generation seed. Leave ``None`` for training, where
            per-example variation in a SMILES conformer is augmentation; set it
            for evaluation, where the same variation is noise.
        ligands: declared ligand identities, keyed by chain id.
        attach_labels: also emit ``data["labels"]`` -- the source coordinates on
            the model's atom axis, plus a mask. Supervision targets only; the
            objective stays with the caller.
        pre_transforms: AtomWorks transforms to run first -- crops, filters, MSA
            loading. Everything structural belongs here.
        keys_to_keep: final ``SubsetToKeys``. Defaults to the features, the
            decode metadata and the example id.

    Returns:
        An ``atomworks.ml.transforms.Compose``.
    """
    transforms: list[Transform] = list(pre_transforms or [])
    transforms.append(
        StructurePredictionInputTransform(
            ligands=ligands,
            allow_undeclared_ccd_ligands=allow_undeclared_ccd_ligands,
            emit_modifications=emit_modifications,
        )
    )
    transforms.append(FeaturizeForESMFold2(seed=seed))
    if attach_labels:
        transforms.append(AttachStructureLabels())

    if keys_to_keep is None:
        keys_to_keep = ["example_id", "feats", "chain_infos", "extra_info"]
        if attach_labels:
            keys_to_keep += ["labels", "label_coverage", "label_skipped_chains"]
        if is_inference:
            keys_to_keep += ["atom_array", "adapter_report"]
    transforms.append(SubsetToKeys(keys_to_keep))
    return Compose(transforms)
