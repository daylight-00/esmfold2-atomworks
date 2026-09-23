"""Equivalence checks between the native and AtomWorks-fed ESMFold2 paths.

The milestone this repo is built around is::

    ESMFold2(I_native) ~= ESMFold2(F(A))

where ``F`` is :func:`~esmfold2_foundry.data.atomworks_to_esm.atom_array_to_structure_prediction_input`.
It is checked at two levels, and the cheaper one is by far the more useful:

**Feature parity** (:func:`compare_features`) compares the 29 tensors that
``prepare_esmfold2_input`` produces. It needs no GPU, no weights and no
sampling, it is exact rather than tolerance-based, and when it passes, output
parity follows by construction -- the model is a pure function of these tensors.
When it fails it names the tensor, which is a diagnosis rather than a symptom.

**Output parity** (:func:`compare_results`) compares coordinates, pLDDT, pTM,
ipTM and the distogram from a real fold. It needs a GPU and it is the only
level at which sampling noise exists, so it is a tolerance check and a
confirmation, not the primary test.

Running only the second is a trap: the structure head is a diffusion sampler, so
a single paired run cannot distinguish an implementation difference from
run-to-run scatter, and a shared seed does not guarantee a shared trajectory
when the two paths consume randomness differently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "MODEL_CONSUMED_FEATURES",
    "TRAINING_ONLY_FEATURES",
    "FeatureDiff",
    "ResultDiff",
    "compare_features",
    "compare_results",
    "featurize",
]

#: The feature tensors the release ``ESMFold2Model.forward`` actually declares
#: as parameters. ``prepare_esmfold2_input`` emits 29 tensors; these 23 are the
#: ones the model reads.
MODEL_CONSUMED_FEATURES = frozenset(
    {
        "token_index",
        "residue_index",
        "asym_id",
        "entity_id",
        "sym_id",
        "mol_type",
        "res_type",
        "input_ids",
        "token_bonds",
        "token_attention_mask",
        "ref_pos",
        "ref_element",
        "ref_charge",
        "ref_atom_name_chars",
        "ref_space_uid",
        "atom_attention_mask",
        "atom_to_token",
        "distogram_atom_idx",
        "msa",
        "deletion_value",
        "has_deletion",
        "deletion_mean",
        "msa_attention_mask",
    }
)

#: The remaining six. They are produced by the featurizer and land in
#: ``forward(**kwargs)``, where they are discarded -- ``gt_coords`` and
#: ``is_resolved`` are zeros/False at inference by construction, and
#: ``pocket_feature`` is zeroed unconditionally (``prepare_input.py`` marks it
#: ``# --- Pocket (dropped) ---``). A difference here cannot change a
#: prediction, so it is reported separately rather than failing parity.
TRAINING_ONLY_FEATURES = frozenset(
    {
        "gt_coords",
        "is_resolved",
        "frames_idx",
        "disto_cond",
        "disto_cond_mask",
        "pocket_feature",
    }
)


def featurize(spi: Any, *, seed: int | None = 0) -> dict[str, Any]:
    """The ESMFold2 feature dict for *spi*, unbatched and on CPU.

    Deliberately calls ``prepare_esmfold2_input`` rather than
    ``ESMFold2InputBuilder.prepare_input``: the latter adds a batch dimension
    and moves tensors to a device, neither of which a comparison needs.
    ``clean_esmfold2_input`` is applied first, because that is what the builder
    does and skipping it would compare something the model never sees.
    """
    from esm.models.esmfold2.prepare_input import prepare_esmfold2_input
    from esm.models.esmfold2.processor import clean_esmfold2_input

    features, _chains = prepare_esmfold2_input(clean_esmfold2_input(spi), seed=seed)
    return features


def _as_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    try:
        return np.asarray(value)
    except (TypeError, ValueError):
        return None


@dataclass
class FeatureDiff:
    """Per-tensor comparison of two ESMFold2 feature dicts."""

    only_in_native: list[str] = field(default_factory=list)
    only_in_adapted: list[str] = field(default_factory=list)
    shape_mismatch: dict[str, tuple[tuple, tuple]] = field(default_factory=dict)
    value_mismatch: dict[str, float] = field(default_factory=dict)
    uncomparable: list[str] = field(default_factory=list)
    compared: list[str] = field(default_factory=list)

    def _fatal(self, names: list[str] | dict) -> list[str]:
        """Those of *names* that the model actually reads."""
        return sorted(set(names) & MODEL_CONSUMED_FEATURES)

    @property
    def consumed_mismatches(self) -> list[str]:
        """Differing tensors that ``forward`` declares as parameters."""
        return sorted(
            set(
                self._fatal(self.only_in_native)
                + self._fatal(self.only_in_adapted)
                + self._fatal(self.shape_mismatch)
                + self._fatal(self.value_mismatch)
            )
        )

    @property
    def ok(self) -> bool:
        """True when nothing the model reads differs.

        Deliberately not "nothing differs at all": six of the 29 tensors are
        discarded by ``forward``, so a difference there cannot change a
        prediction, and failing on it would make the check cry wolf.
        :attr:`identical` is the stricter statement.
        """
        return not self.consumed_mismatches

    @property
    def identical(self) -> bool:
        """True when all 29 tensors match, dropped ones included."""
        return not (
            self.only_in_native
            or self.only_in_adapted
            or self.shape_mismatch
            or self.value_mismatch
        )

    def report(self) -> str:
        if self.identical:
            return f"feature parity: all {len(self.compared)} tensors identical"
        if self.ok:
            differing = sorted(
                set(list(self.shape_mismatch) + list(self.value_mismatch))
                - MODEL_CONSUMED_FEATURES
            )
            return (
                f"feature parity: all {len(MODEL_CONSUMED_FEATURES & set(self.compared))} "
                f"model-consumed tensors identical; differs only in discarded "
                f"tensors {differing}"
            )
        lines = [f"feature parity FAILED on {self.consumed_mismatches}"]
        if self.only_in_native:
            lines.append(f"  only in native:  {sorted(self.only_in_native)}")
        if self.only_in_adapted:
            lines.append(f"  only in adapted: {sorted(self.only_in_adapted)}")
        for key, (a, b) in sorted(self.shape_mismatch.items()):
            lines.append(f"  shape  {key}: native {a} vs adapted {b}")
        for key, delta in sorted(self.value_mismatch.items()):
            lines.append(f"  values {key}: max |diff| = {delta:g}")
        if self.uncomparable:
            lines.append(f"  not comparable: {sorted(self.uncomparable)}")
        return "\n".join(lines)


def compare_features(
    native: dict[str, Any],
    adapted: dict[str, Any],
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
    ignore: frozenset[str] = frozenset(),
) -> FeatureDiff:
    """Compare two feature dicts tensor by tensor.

    Defaults to an **exact** comparison. Both dicts come from the same
    featurizer applied to two descriptions of the same system, so any
    difference is a difference in the description -- there is no arithmetic in
    between to accumulate error. Loosen *atol* only for a ligand whose conformer
    is generated from SMILES, where RDKit embedding is the one genuinely
    stochastic step.
    """
    diff = FeatureDiff()
    keys_a = set(native) - ignore
    keys_b = set(adapted) - ignore

    diff.only_in_native = sorted(keys_a - keys_b)
    diff.only_in_adapted = sorted(keys_b - keys_a)

    for key in sorted(keys_a & keys_b):
        a = _as_numpy(native[key])
        b = _as_numpy(adapted[key])
        if a is None or b is None:
            diff.uncomparable.append(key)
            continue
        if a.shape != b.shape:
            diff.shape_mismatch[key] = (a.shape, b.shape)
            continue

        diff.compared.append(key)
        if a.dtype == bool or np.issubdtype(a.dtype, np.integer):
            if not np.array_equal(a, b):
                diff.value_mismatch[key] = float((a != b).sum())
            continue
        # Float: NaN is meaningful here (unresolved coordinates), so compare
        # NaN placement explicitly instead of letting it fail every check.
        nan_a, nan_b = np.isnan(a), np.isnan(b)
        if not np.array_equal(nan_a, nan_b):
            diff.value_mismatch[key] = float(np.logical_xor(nan_a, nan_b).sum())
            continue
        finite = ~nan_a
        if finite.any() and not np.allclose(a[finite], b[finite], rtol=rtol, atol=atol):
            diff.value_mismatch[key] = float(np.abs(a[finite] - b[finite]).max())
    return diff


@dataclass
class ResultDiff:
    """Comparison of two ``MolecularComplexResult`` objects."""

    metrics: dict[str, float] = field(default_factory=dict)
    tolerances: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(
            value <= self.tolerances.get(key, 0.0)
            for key, value in self.metrics.items()
        )

    def report(self) -> str:
        lines = ["output parity: " + ("PASS" if self.ok else "FAIL")]
        for key, value in sorted(self.metrics.items()):
            tol = self.tolerances.get(key)
            mark = "ok " if tol is not None and value <= tol else "BAD"
            lines.append(f"  [{mark}] {key:24s} {value:12.6g}  (tol {tol})")
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


def compare_results(
    native: Any,
    adapted: Any,
    *,
    coord_atol: float = 1e-4,
    scalar_atol: float = 1e-4,
    plddt_atol: float = 1e-3,
    distogram_atol: float = 1e-3,
) -> ResultDiff:
    """Compare two folds of the same system.

    Coordinates are compared **as reported**, not after superposition: the two
    inputs describe the same system in the same order, so a rigid-body
    difference would itself be a finding, and aligning first would hide it.
    Atom order is checked before any coordinate comparison, because comparing
    two differently-ordered atom lists produces a large number that looks like a
    modelling difference and is really a bookkeeping one.
    """
    diff = ResultDiff()

    a_xyz = _as_numpy(native.complex.atom_positions)
    b_xyz = _as_numpy(adapted.complex.atom_positions)
    if a_xyz is None or b_xyz is None:
        diff.notes.append("one result has no coordinates")
        return diff
    if a_xyz.shape != b_xyz.shape:
        diff.notes.append(f"atom count differs: {a_xyz.shape} vs {b_xyz.shape}")
        diff.metrics["atom_count_delta"] = float(abs(a_xyz.shape[0] - b_xyz.shape[0]))
        diff.tolerances["atom_count_delta"] = 0.0
        return diff

    a_names = _as_numpy(native.complex.atom_names)
    b_names = _as_numpy(adapted.complex.atom_names)
    if a_names is not None and b_names is not None:
        mismatched = int((a_names.astype(str) != b_names.astype(str)).sum())
        diff.metrics["atom_name_mismatches"] = float(mismatched)
        diff.tolerances["atom_name_mismatches"] = 0.0

    delta = a_xyz - b_xyz
    diff.metrics["coord_max_abs"] = float(np.abs(delta).max())
    diff.tolerances["coord_max_abs"] = coord_atol
    diff.metrics["coord_rmsd"] = float(np.sqrt((delta**2).sum(axis=-1).mean()))
    diff.tolerances["coord_rmsd"] = coord_atol

    for name, tol in (("ptm", scalar_atol), ("iptm", scalar_atol)):
        a_val, b_val = getattr(native, name, None), getattr(adapted, name, None)
        if a_val is None or b_val is None:
            continue
        diff.metrics[name] = float(abs(float(a_val) - float(b_val)))
        diff.tolerances[name] = tol

    for name, tol in (
        ("plddt", plddt_atol),
        ("distogram", distogram_atol),
        ("pae", plddt_atol),
    ):
        a_arr = _as_numpy(getattr(native, name, None))
        b_arr = _as_numpy(getattr(adapted, name, None))
        if a_arr is None or b_arr is None or a_arr.shape != b_arr.shape:
            if a_arr is not None and b_arr is not None:
                diff.notes.append(f"{name}: shape {a_arr.shape} vs {b_arr.shape}")
            continue
        diff.metrics[f"{name}_max_abs"] = float(np.abs(a_arr - b_arr).max())
        diff.tolerances[f"{name}_max_abs"] = tol

    return diff
