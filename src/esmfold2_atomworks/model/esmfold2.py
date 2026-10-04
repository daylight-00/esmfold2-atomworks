"""The ESMFold2 model wrapper.

ESMFold2, driven from AtomWorks **without its architecture being rewritten**.
``AtomWorksESMFold2`` holds the native ESMFold2 module and delegates to it;
whatever trains or serves it -- the optional Foundry integration in
:mod:`esmfold2_atomworks.training`, or anything else -- works with the native
module rather than a reimplementation.

The module ships in ``esm`` >= 3.4.

The reason for the indirection: both upstreams' APIs change between releases, so
a full rewrite would have to track them, and any subtle divergence would show up
as a model that runs, reports plausible confidence, and is quietly wrong.
Wrapping keeps the weights and the numerics
exactly as published, and leaves one named seam per component.

Behaviours of the native model that callers need to know:

1. **Gradients depend on the inputs, not just the checkpoint.** The release
   ``forward`` is ``@torch.inference_mode()`` and can never produce them. The
   experimental one gates autograd on ``res_type_soft`` being supplied, so
   loading it is necessary and not sufficient -- see
   :meth:`AtomWorksESMFold2.will_produce_gradients`.
2. **A sampler knob reaches the model only if its ``forward`` declares it.**
   esm >= 3.4 declares ``noise_scale``, ``step_scale`` and
   ``max_inference_sigma`` and hands them to the structure head. :meth:`fold`
   warns about any a module would drop, read off the loaded module's own
   signature.
   ``early_exit`` is deprecated and ignored by upstream's ``fold``.
3. **pLDDT is on 0--1, not 0--100**, and ``result.plddt`` (model tokens) is a
   different length from ``result.complex.plddt`` (collapsed residues) whenever
   a ligand or modified residue is present. Do not index one with the other;
   :func:`esmfold2_atomworks.metrics.plddt_per_token` and
   :func:`~esmfold2_atomworks.metrics.plddt_per_residue` name the two.
4. **Without an ESMC backbone, ``forward`` folds anyway**, leaving the LM
   pathway out -- a different computation from the checkpoint's, for any
   input. :meth:`AtomWorksESMFold2.fold` refuses unless a backbone is resident
   or the caller supplies the hidden states.
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from esmfold2_atomworks import paths

if TYPE_CHECKING:
    from biotite.structure import AtomArray

__all__ = [
    "GRADIENT_GATE",
    "LM_SOURCES",
    "REPLICATED_FOLD_PARAMETERS",
    "SAMPLER_KNOBS",
    "AtomWorksESMFold2",
    "FoldingConfig",
    "MissingLanguageModelError",
    "attach_esmc",
    "ccd_source",
    "input_record",
    "snapshot_dir",
    "tensor_record",
]

#: The input that switches the experimental forward into a gradient-bearing
#: mode. Upstream gates autograd on it directly --
#: ``torch.set_grad_enabled(res_type_soft is not None)`` -- so its presence,
#: not the checkpoint flavour alone, is what decides whether a loss is
#: differentiable.
GRADIENT_GATE = "res_type_soft"

#: Sampler knobs ``fold()`` forwards to ``forward`` when set. A module whose
#: ``forward`` does not declare one drops it, so :meth:`fold` checks the loaded
#: module's signature and warns about any it would drop.
#: (``early_exit`` is deprecated upstream and ignored by ``fold`` itself, with
#: a ``DeprecationWarning`` of its own.)
SAMPLER_KNOBS = ("noise_scale", "step_scale", "max_inference_sigma")


#: Where the LM prior of a fold came from, as a call record states it:
#: computed by the resident ESMC backbone, or handed in by the caller.
LM_SOURCES = ("model", "caller-supplied")

#: ``ESMFold2InputBuilder.fold``'s parameters, every one of which the fold that
#: carries caller-supplied LM hidden states reproduces. That fold replicates
#: upstream's body because upstream's takes no ``lm_hidden_states``; a parameter
#: upstream adds later would be dropped by the replica, so a signature that no
#: longer matches this set raises instead (see ``_fold_with_lm_states``).
REPLICATED_FOLD_PARAMETERS = frozenset(
    {
        "model",
        "input",
        "num_loops",
        "num_sampling_steps",
        "num_diffusion_samples",
        "seed",
        "noise_scale",
        "step_scale",
        "max_inference_sigma",
        "lm_mask_pct",
        "lm_dropout",
        "early_exit",
        "msa_max_depth",
        "msa_column_mask_rate",
        "msa_subsample_at_inference",
        "include_embeddings",
        "complex_id",
    }
)


#: The parameters of ``ESMFold2InputBuilder.fold`` this wrapper fills itself.
#: Passed as overrides, the ordinary path fails on them ("multiple values") and
#: the replica would ignore them, so both refuse them up front.
_FOLD_POSITIONALS = frozenset({"model", "input"})


class MissingLanguageModelError(RuntimeError):
    """A fold would run without the LM prior the checkpoint was trained with.

    The native ``forward`` skips the language-model pathway when no ESMC
    backbone is attached and no hidden states are given, and folds anyway. That
    is not the published model with a smaller input: its LM shim maps even the
    all-zero states of a protein-free input to a non-zero pair term, so leaving
    the pathway out changes every fold, with or without protein chains.
    """


def ccd_source(ccd_cache: Any) -> str:
    """Where esm reads the CCD pickle for a builder given ``ccd_cache``.

    In esm's own order: ``ESMCFOLD_CCD_PATH`` as captured when
    ``esm.models.esmfold2.conformers`` was imported, which every load prefers;
    then ``<ccd_cache>/ccd.pkl``; then a download of the Hub's latest copy.

    The dictionary is process-global and esm keeps no record of where it came
    from. When something loaded it before this call, with no captured path to
    pin the source, the builder's own location is never read, and the source is
    reported as unknown rather than as that location.
    """
    from esm.models.esmfold2 import conformers

    captured = getattr(conformers, "CCD_PICKLE_PATH", None)
    if captured is not None and Path(captured).exists():
        return str(captured)
    if getattr(conformers, "_CCD_MOLECULES", None) is not None:
        return "unknown: loaded before this model, from a source esm does not record"
    if ccd_cache is not None:
        return str(Path(ccd_cache) / "ccd.pkl")
    return "biohub/ESMFold2 ccd.pkl, the Hub's latest revision"


def _bundles_esmc(config: Any) -> bool:
    return getattr(config, "esmc_config", None) is not None


def _apply(net: Any, setter: str, value: Any) -> str:
    """Call ``net.<setter>(value)`` and say what was applied.

    A module without the setter cannot honour the request, so the record says
    so rather than reporting the value as if it had taken effect.
    """
    method = getattr(net, setter, None)
    if method is None:
        return f"not applied: {type(net).__name__} has no {setter}"
    method(value)
    return str(value)


def snapshot_dir(source: str) -> str:
    """``source`` as a local directory: itself if it is one, else its Hub snapshot.

    A Hub id is resolved through upstream's own ``resolve_model_dir``, into the
    Hugging Face store (``HF_HUB_CACHE``), downloading only what is missing --
    the step ``from_pretrained`` would take anyway, taken first so that the
    snapshot, and so its revision, is known to whoever loaded it.
    """
    if Path(source).is_dir():
        return str(source)
    from esm.models.hub import resolve_model_dir

    return str(resolve_model_dir(source))


def attach_esmc(net: Any, precision: str = "bf16") -> str:
    """Attach the ESMC backbone ``net``'s checkpoint needs; say where it came from.

    Returns ``"bundled"`` for a checkpoint that carries its backbone, which the
    native loader has already attached. Otherwise the checkpoint names a
    separate one in ``config.esmc_id``, which is resolved through
    :func:`esmfold2_atomworks.paths.resolve_esmc` -- a local mirror first --
    rather than handed to ``from_pretrained`` as written; a Hub id with no
    mirror is resolved to its snapshot (:func:`snapshot_dir`). The local
    directory it was loaded from is returned.

    ``precision`` reaches the release model only: the experimental model's
    ``load_esmc`` takes none and always loads bf16, as its own
    ``from_pretrained`` does.
    """
    if _bundles_esmc(net.config):
        return "bundled"
    source = snapshot_dir(str(paths.resolve_esmc(net.config.esmc_id)))
    if "precision" in inspect.signature(net.load_esmc).parameters:
        net.load_esmc(source, precision=precision)
    else:
        net.load_esmc(source)
    return source


def _field(obj: Any, path: str) -> Any:
    """``obj.a.b`` for ``path="a.b"``, or ``None`` if any step is absent."""
    for name in path.split("."):
        obj = getattr(obj, name, None)
        if obj is None:
            return None
    return obj


def _dimension(obj: Any, *paths: str) -> int:
    """The first of ``paths`` present on ``obj``, as a positive width.

    Raises rather than defaulting: a width that cannot be found is unknown, and
    a default would pass it off as measured.
    """
    for path in paths:
        value = _field(obj, path)
        if value is None:
            continue
        width = int(value)
        if width <= 0:
            raise ValueError(
                f"config field {path!r} is {width}; a width must be positive"
            )
        return width
    raise AttributeError(
        f"none of {list(paths)} is present on {type(obj).__name__}; the config "
        "schema has moved again, and the width is unknown rather than zero"
    )


def _check_record(record: dict[str, Any] | None) -> None:
    """Raise if ``record`` already holds entries of an earlier call."""
    if record is None:
        return
    stale = sorted(key for key in record if str(key).startswith("esmfold2."))
    if stale:
        raise ValueError(
            f"record already holds {stale}; pass a fresh dict per call, so two "
            "calls' entries cannot mix"
        )


def _sha256(payload: Any) -> str:
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def _msa_sha256(msa: Any) -> str | None:
    """Every row of an alignment -- header and sequence -- and its deletion matrix.

    Headers are included because they carry the taxonomy cross-chain pairing
    reads, and the deletions because they become model features.
    """
    import hashlib

    import numpy as np

    if msa is None:
        return None
    digest = hashlib.sha256()
    for entry in msa.entries:
        digest.update(str(entry.header).encode() + b"\0")
        digest.update(str(entry.sequence).encode() + b"\n")
    deletions = getattr(msa, "deletions", None)
    if deletions is not None:
        deletions = np.ascontiguousarray(deletions)
        digest.update(f"{deletions.dtype}{deletions.shape}".encode())
        digest.update(deletions.tobytes())
    return digest.hexdigest()


def input_record(spi: Any) -> dict[str, Any]:
    """What a ``StructurePredictionInput`` hands the model, as record entries.

    Read after upstream's own ``clean_esmfold2_input``, which splits chainbreaks
    into the entities that are actually folded. ``esmfold2.inputs`` lists each:
    ids, kind, length (``None`` for a ligand), ``chemistry_sha256`` over the
    sequence and modifications or a ligand's CCD codes or SMILES -- not the chain
    ids, so the same molecule under another name agrees -- and ``msa_sha256``
    over its alignment, or ``None``. ``esmfold2.covalent_bonds`` is the bond list
    in canonical order; ``esmfold2.pocket_sha256`` and
    ``esmfold2.distogram_conditioning_sha256`` digest those conditions, ``None``
    when absent. Every digest is a full SHA-256.
    """
    import numpy as np
    from esm.models.esmfold2.processor import clean_esmfold2_input

    cleaned = clean_esmfold2_input(spi)
    entities = []
    for entry in cleaned.sequences:
        kind = type(entry).__name__.removesuffix("Input").lower()
        sequence = getattr(entry, "sequence", None)
        chemistry = {
            "kind": kind,
            "sequence": sequence,
            "modifications": sorted(
                (m.position, m.ccd, getattr(m, "smiles", None))
                for m in (getattr(entry, "modifications", None) or [])
            ),
            "ccd": list(getattr(entry, "ccd", None) or []),
            "smiles": getattr(entry, "smiles", None),
        }
        ids = entry.id if isinstance(entry.id, list) else [entry.id]
        entities.append(
            {
                "ids": [str(i) for i in ids],
                "kind": kind,
                "length": None if sequence is None else len(sequence),
                "chemistry_sha256": _sha256(chemistry),
                "msa_sha256": _msa_sha256(getattr(entry, "msa", None)),
            }
        )
    bonds = sorted(
        [
            str(b.chain_id1),
            int(b.res_idx1),
            int(b.atom_idx1),
            str(b.chain_id2),
            int(b.res_idx2),
            int(b.atom_idx2),
        ]
        for b in (cleaned.covalent_bonds or [])
    )
    pocket = cleaned.pocket
    distogram = cleaned.distogram_conditioning
    return {
        "esmfold2.inputs": entities,
        "esmfold2.covalent_bonds": bonds,
        "esmfold2.pocket_sha256": None
        if pocket is None
        else _sha256([pocket.binder_chain_id, sorted(map(list, pocket.contacts))]),
        "esmfold2.distogram_conditioning_sha256": None
        if not distogram
        else _sha256(
            [
                [
                    d.chain_id,
                    np.asarray(d.distogram, dtype=np.float32).shape,
                    np.asarray(d.distogram, dtype=np.float32).tobytes().hex(),
                ]
                for d in distogram
            ]
        ),
    }


def tensor_record(tensor: Any, *, chunk_elements: int = 1 << 24) -> dict[str, Any]:
    """A tensor's full SHA-256 over its bytes, with its shape and dtype.

    Hashed in chunks copied to the host one at a time, so the host holds one
    chunk rather than the tensor.
    """
    import hashlib

    import torch

    flat = tensor.detach().contiguous().reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, flat.numel(), chunk_elements):
        piece = flat[start : start + chunk_elements].cpu()
        digest.update(piece.view(torch.uint8).numpy().tobytes())
    return {
        "sha256": digest.hexdigest(),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
    }


def _observed_execution() -> dict[str, Any]:
    """The kernel-determinism settings in force, as observed -- not certified.

    ``esmfold2.deterministic_algorithms`` is torch's flag at this moment, and
    ``esmfold2.cublas_workspace_config`` the environment variable's value
    (``None`` when unset). cuBLAS reads that variable when CUDA starts, so a
    value set later is recorded but was never applied; nothing here can tell
    the two apart, which is why the entries are observations. Deterministic
    kernels make a fold repeatable on one device and software stack; the seed
    alone does not (docs/02).
    """
    import os

    import torch

    return {
        "esmfold2.deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "esmfold2.cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


@dataclass
class FoldingConfig:
    """Inference-time knobs, mirroring the SDK's ``FoldingConfig``.

    ``num_loops=None`` means the checkpoint's own ``config.num_loops``, which
    is what the model uses when it is given no count -- and the checkpoints
    disagree: 3 in revision ``e1e189d0`` of ``biohub/ESMFold2``, 20 in ``69869f73``
    (docs/04). A fixed default here would silently override
    whichever of them is loaded. Set it to pin a schedule.
    """

    num_loops: int | None = None
    num_sampling_steps: int = 100
    num_diffusion_samples: int = 1
    lm_dropout: float | None = 0.3
    lm_mask_pct: float | None = None
    seed: int | None = None
    #: The sampler's knobs (:data:`SAMPLER_KNOBS`). ``None`` leaves each to the
    #: loaded model: the scales to its structure head's config, the sigma cap
    #: to ``forward``'s default -- as the call record's ``esmfold2.effective.*``
    #: then states.
    noise_scale: float | None = None
    step_scale: float | None = None
    max_inference_sigma: float | None = None
    #: Further ``ESMFold2InputBuilder.fold`` arguments, passed through as given.
    extra: dict[str, Any] = field(default_factory=dict)

    def as_fold_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            # Passed even when None: ESMFold2InputBuilder.fold has a default of
            # its own (20), so leaving the key out would not reach the model's
            # fallback to config.num_loops.
            "num_loops": self.num_loops,
            "num_sampling_steps": self.num_sampling_steps,
            "num_diffusion_samples": self.num_diffusion_samples,
            "lm_dropout": self.lm_dropout,
            "seed": self.seed,
        }
        if self.lm_mask_pct is not None:
            kwargs["lm_mask_pct"] = self.lm_mask_pct
        for knob in SAMPLER_KNOBS:
            if getattr(self, knob) is not None:
                kwargs[knob] = getattr(self, knob)
        kwargs.update(self.extra)
        return kwargs


class AtomWorksESMFold2:
    """A resident ESMFold2, fed from AtomWorks and answering in AtomWorks terms.

    Not an ``nn.Module``: it holds no parameters of its own, and wrapping it in a
    module would add a parameter namespace every checkpoint has to agree about.
    :attr:`net` is the native module and is what a trainer should register.
    """

    def __init__(
        self,
        weights: str | Any | None = None,
        *,
        device: Any | None = None,
        load_esmc: bool = True,
        esmc_precision: str = "bf16",
        ccd_cache: Any | None = None,
        chunk_size: int | None = 64,
        kernel_backend: str | None = None,
    ) -> None:
        import torch
        from esm.models.esmfold2.processor import ESMFold2InputBuilder

        if weights is None:
            # A local mirror when there is one, otherwise the Hub id -- so a
            # fresh clone downloads rather than failing on a missing directory.
            weights = paths.ESMFOLD2_WEIGHTS.resolve("standard")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.weights = str(weights)
        #: The directory the weights were read from: ``weights`` itself when it
        #: is one, otherwise the Hub snapshot it names, resolved before loading
        #: so that the revision that ran is known (``provenance()``).
        self.weights_resolved = self.weights
        self.device = torch.device(device)
        try:
            from esm.models.esmfold2.model import EsmFold2Model
        except ImportError as error:
            raise ImportError(
                "No ESMFold2 model class found. Install esm >= 3.4.1.post1, which "
                "ships the model. See docs/04_ENVIRONMENT.md."
            ) from error

        self.weights_resolved = snapshot_dir(self.weights)
        # esm places the model on `device` during construction, which avoids
        # materializing it on CPU first. A bundled backbone comes in with the
        # trunk regardless of `load_esmc`; a separate one is attached here rather
        # than from the checkpoint's `esmc_id` (see attach_esmc).
        self.net = EsmFold2Model.from_pretrained(
            self.weights_resolved,
            load_esmc=False,
            esmc_precision=esmc_precision,
            device=str(self.device),
        ).eval()
        if load_esmc:
            self.esmc_source = attach_esmc(self.net, esmc_precision)
        elif _bundles_esmc(self.net.config):
            self.esmc_source = "bundled"
        else:
            self.esmc_source = "none"

        # Both are applied as given, None included: set_chunk_size(None)
        # disables chunking, and set_kernel_backend(None) selects upstream's
        # reference path (its default). torch.compile and the Triton kernels do
        # not stack; upstream requires clearing the backend before compiling.
        # What provenance() records is what was applied, never merely asked for.
        # Changes made later directly on .net are not tracked.
        self.chunk_size = _apply(self.net, "set_chunk_size", chunk_size)
        self.kernel_backend = _apply(self.net, "set_kernel_backend", kernel_backend)
        self.esmc_precision = self._applied_esmc_precision(esmc_precision)

        # Stateless, and its __init__ loads the ~50k-entry CCD dictionary
        # (~9 s), so one builder is shared by every call. The dictionary comes
        # from the mirror when there is one (paths.ccd_dir), not from the Hub.
        if ccd_cache is None:
            ccd_cache = paths.ccd_dir()
        self.ccd_source = ccd_source(ccd_cache)
        self.builder = ESMFold2InputBuilder(ccd_cache=ccd_cache)

    # -- introspection -----------------------------------------------------

    @property
    def config(self) -> Any:
        """The live config. Read dimensions off this, never off the dataclass defaults."""
        return self.net.config

    @property
    def supports_soft_sequence_design(self) -> bool:
        """Whether this checkpoint *can* produce gradients at all.

        **Necessary, not sufficient.** The release model's ``forward`` is
        ``@torch.inference_mode()`` and can never produce them. The
        experimental model can, but only under the gate it applies internally::

            torch.set_grad_enabled(res_type_soft is not None)  # experimental.py

        so an experimental checkpoint fed an ordinary integer ``res_type``
        still runs with autograd *off*. Use :meth:`will_produce_gradients` to
        ask the question that actually matters.
        """
        return getattr(self.config, "type", "release") == "experimental"

    def will_produce_gradients(self, inputs: dict[str, Any]) -> bool:
        """Whether ``forward(**inputs)`` will build an autograd graph.

        Both conditions, checked together: an experimental checkpoint *and* a
        soft sequence among the inputs.
        """
        return (
            self.supports_soft_sequence_design and inputs.get(GRADIENT_GATE) is not None
        )

    def explain_gradient_status(self, inputs: dict[str, Any]) -> str:
        """Why gradients are or are not available, in one line."""
        if not self.supports_soft_sequence_design:
            return (
                f"no gradients: this is a '{getattr(self.config, 'type', 'release')}' "
                "checkpoint, whose forward is @torch.inference_mode(). Load the "
                "experimental checkpoint."
            )
        if inputs.get(GRADIENT_GATE) is None:
            return (
                f"no gradients: the experimental forward gates autograd on "
                f"`{GRADIENT_GATE}`, which is absent from the inputs. Pass a soft "
                "sequence (e.g. a softmax over design logits) instead of relying "
                "on the integer res_type."
            )
        return "gradients available"

    def representation_dims(self) -> dict[str, int]:
        """The single and pair widths, read off the live config.

        ``EsmFold2Config`` migrates a pre-alignment ``config.json`` on load and
        drops the old field names, so ``d_pair`` or ``c_token`` are absent from a
        config that was written with them; the current names are read, and the
        old ones only when a current one is missing. A width that is missing
        under both, or not positive, raises: an unknown dimension reported as
        ``0`` reads downstream as a measurement.

        The keys are this method's own, not upstream's.
        """
        config = self.config
        diffusion = _field(config, "structure_head.diffusion_module")
        return {
            "d_pair": _dimension(config, "pairwise_hidden_size", "d_pair"),
            "d_single_declared": _dimension(config, "hidden_size", "d_single"),
            "c_token": _dimension(diffusion, "token_hidden_size", "c_token"),
            "c_atom": _dimension(diffusion, "atom_encoder.hidden_size", "c_atom"),
            "c_s_inputs": _dimension(diffusion, "c_s_inputs"),
        }

    # -- component seams ---------------------------------------------------
    # Named accessors for the ESMC backbone, the folding trunk and the structure
    # head. Only `.esmc` has a real absent state; the trunk and the head raise
    # rather than answer None, which a caller would read as a component that is
    # legitimately missing.

    @property
    def esmc(self) -> Any:
        """The frozen ESMC language model backbone, or ``None`` when none is attached.

        ``None`` means exactly that -- constructed with ``load_esmc=False`` from
        a checkpoint that does not bundle one -- so the attribute is read under
        the name the native module stores it: ``esmc``, with ``_esmc`` as a
        fallback for a layout that kept it private.
        """
        esmc = getattr(self.net, "esmc", None)
        if esmc is not None:
            return esmc
        return getattr(self.net, "_esmc", None)

    @property
    def folding_trunk(self) -> Any:
        """The pair-representation trunk. ``AttributeError`` if the module has none."""
        return self.net.folding_trunk

    @property
    def structure_head(self) -> Any:
        """The diffusion structure head. ``AttributeError`` if the module has none."""
        return self.net.structure_head

    # -- the model as a function ------------------------------------------

    def featurize(self, spi: Any, *, seed: int | None = None) -> tuple[dict, list]:
        """``StructurePredictionInput`` -> ``(features, chain_infos)`` on this device.

        ``chain_infos`` is not optional bookkeeping: it is the only record of
        ``token -> (chain, residue, atom span)``, and ``decode`` cannot run
        without it.
        """
        return self.builder.prepare_input(spi, seed=seed, device=self.device)

    def compute_lm_hidden_states(
        self, features: dict[str, Any], *, lm_mask_pct: float | None = None
    ) -> Any:
        """The ESMC hidden states a fold of ``features`` would compute itself.

        ``[B, L, n_layers + 1, d_model]``, detached, on the model's device --
        what :meth:`fold` accepts as ``lm_hidden_states``. ``features`` is the
        first element of :meth:`featurize`.

        Computed the way ``forward`` computes them, through the native module's
        own method, which restores an offloaded backbone and applies the FP8
        precision context and padding; the bare
        ``esm.models.esmfold2.layers.compute_lm_hidden_states`` does neither.
        That method is private upstream, so this is the one place that depends
        on it.

        Args:
            lm_mask_pct: fraction of residues masked before the backbone.
                ``None`` means the checkpoint's own ``config.lm_mask_pct``, which
                is what ``forward`` uses when it computes the states itself. A
                non-zero fraction draws from the torch RNG, unseeded here.

        Raises:
            MissingLanguageModelError: no backbone is resident.
        """
        if self.esmc is None:
            raise MissingLanguageModelError(
                "no ESMC backbone is resident, so there is nothing to compute LM "
                "hidden states with; construct the model with load_esmc=True"
            )
        compute = getattr(self.net, "_compute_lm_hidden_states", None)
        if compute is None:
            raise AttributeError(
                f"{type(self.net).__name__} has no _compute_lm_hidden_states; the "
                "native module's LM path has moved, and computing the states "
                "another way would not be the computation forward performs"
            )
        if lm_mask_pct is None:
            # Read, not defaulted: a config that no longer carries it has moved,
            # and 0.0 would be a fraction nobody measured.
            lm_mask_pct = self.config.lm_mask_pct

        import torch

        with torch.no_grad():
            return compute(
                features["input_ids"],
                features["asym_id"],
                features["residue_index"],
                features["mol_type"],
                # forward's `tok_mask` is this feature, unchanged.
                features["token_attention_mask"],
                lm_mask_pct=lm_mask_pct,
            )

    def _check_lm_hidden_states(self, states: Any, features: dict[str, Any]) -> None:
        """Raise unless ``states`` can stand in for the ones ``forward`` computes.

        Checked against the live module and the prepared features, not the
        config: the layer count and width are read off the LM shim's own
        parameters, and the batch and token axes off ``res_type``. dtype is left
        alone, because the precision path decides it.
        """
        import torch

        if not isinstance(states, torch.Tensor):
            raise TypeError(
                f"lm_hidden_states must be a tensor, not {type(states).__name__}"
            )
        shim = getattr(self.net, "language_model", None)
        if shim is None:
            raise AttributeError(
                f"{type(self.net).__name__} has no language_model; there is no LM "
                "pathway for the hidden states to enter"
            )
        tokens = features["res_type"]
        expected = (
            *tuple(tokens.shape[:2]),
            int(shim.base_z_combine.numel()),
            int(shim.base_z_linear[0].normalized_shape[-1]),
        )
        if tuple(states.shape) != expected:
            raise ValueError(
                f"lm_hidden_states has shape {tuple(states.shape)}; this model and "
                f"these features need {expected} (batch, tokens, layers, width)"
            )
        if not states.is_floating_point():
            raise TypeError(
                f"lm_hidden_states must be floating point, not {states.dtype}"
            )
        if states.device != tokens.device:
            raise ValueError(
                f"lm_hidden_states is on {states.device}, the features on "
                f"{tokens.device}"
            )

    def fold(
        self,
        spi: Any,
        config: FoldingConfig | None = None,
        *,
        lm_hidden_states: Any | None = None,
        record: dict[str, Any] | None = None,
        **overrides: Any,
    ) -> Any:
        """Fold a ``StructurePredictionInput``.

        Returns a single ``MolecularComplexResult``, or a list of them when
        ``num_diffusion_samples > 1`` -- the native ``decode`` collapses the
        one-sample case, and callers must handle both.

        Args:
            lm_hidden_states: ESMC hidden states to fold with instead of those
                the resident backbone would compute, e.g. from
                :meth:`compute_lm_hidden_states`, cached or substituted. Checked
                against this model and the features ``spi`` prepares to. They
                enter the LM shim detached, as upstream detaches them, so no
                gradient reaches them. ``lm_mask_pct`` acts only inside the
                backbone, which this skips, so combining the two raises.
            record: a dict to write this call's ``esmfold2.*`` entries into --
                every argument the fold ran with (``esmfold2.fold.*``), what the
                checkpoint resolved (``esmfold2.effective.*``),
                ``esmfold2.lm_source`` (one of :data:`LM_SOURCES`), each folded
                entity and condition by digest (see :func:`input_record`), the
                supplied LM states by digest (``esmfold2.lm_states``, see
                :func:`tensor_record`)
                and the execution state observed when the call began (see
                :func:`_observed_execution`). One record per call: with
                ``num_diffusion_samples > 1`` it describes every sample, and
                holds no sample index. The caller's other keys are kept; an
                ``esmfold2.*`` key already present raises, so a record reused
                from an earlier fold cannot mix two calls' entries.

        Raises:
            MissingLanguageModelError: no backbone is resident and no
                ``lm_hidden_states`` are given.
        """
        _check_record(record)
        positional = sorted(set(overrides) & _FOLD_POSITIONALS)
        if positional:
            raise TypeError(
                f"fold() got {positional} as keyword overrides; the model and the "
                "input are this call's own, not settings to override"
            )
        config = config or FoldingConfig()
        kwargs = config.as_fold_kwargs()
        kwargs.update(overrides)

        ignored = [
            k
            for k in SAMPLER_KNOBS
            if kwargs.get(k) is not None and k not in self._forward_parameters()
        ]
        if ignored:
            warnings.warn(
                f"{ignored} are accepted by ESMFold2InputBuilder.fold but are not "
                f"declared by {type(self.net).__name__}.forward, so they are "
                "discarded. Set them on config.structure_head instead.",
                RuntimeWarning,
                stacklevel=2,
            )

        execution = _observed_execution() if record is not None else {}
        if lm_hidden_states is None:
            if self.esmc is None:
                raise MissingLanguageModelError(
                    "no ESMC backbone is resident and no lm_hidden_states were "
                    "given, so this fold would run without the LM prior -- a "
                    "different computation from the checkpoint's, whatever the "
                    "input holds. Construct the model with load_esmc=True, or "
                    "pass lm_hidden_states."
                )
            import torch

            with torch.no_grad():
                result = self.builder.fold(self.net, spi, **kwargs)
            lm_source = "model"
        else:
            if kwargs.get("lm_mask_pct") is not None:
                raise ValueError(
                    "lm_mask_pct masks residues inside the ESMC backbone, which "
                    "supplied lm_hidden_states bypass; apply the mask when "
                    "computing the states (compute_lm_hidden_states(lm_mask_pct=))"
                )
            if record is not None:
                # Hashed before the fold: the record names the states that ran.
                execution["esmfold2.lm_states"] = tensor_record(lm_hidden_states)
            result = self._fold_with_lm_states(spi, kwargs, lm_hidden_states)
            lm_source = "caller-supplied"

        if record is not None:
            record.update(self._call_record(spi, kwargs, lm_source))
            record.update(execution)
        return result

    def _forward_parameters(self) -> dict[str, Any]:
        """The loaded module's ``forward`` parameters, by name."""
        forward = getattr(type(self.net), "forward", None)
        if forward is None:
            return {}
        return dict(inspect.signature(forward).parameters)

    def _effective_settings(
        self, args: dict[str, Any], lm_source: str
    ) -> dict[str, Any]:
        """What the model ran with, where a request left a setting to it.

        Read off the live module and config the way upstream's forward at the
        pinned revision resolves each ``None``: the loop count and sample count
        from the config, the step count and both sampler scales from the
        structure head, the sigma cap from ``forward``'s own default (``fold``
        omits it when unset), the MSA depth and column-mask rate from the MSA
        encoder's config (they act only on an input with an alignment), the
        mask fraction from the config when the backbone runs (``None`` with
        supplied states, which no mask reaches), and the LM dropout from the
        config when the call sets none. A setting the loaded module does not
        expose is recorded as ``None`` rather than guessed.
        """

        def chosen(name: str, fallback: Any) -> Any:
            value = args.get(name)
            return fallback() if value is None else value

        config = self.config
        head = getattr(self.net, "structure_head", None)
        msa = getattr(config, "msa_encoder", None)
        sigma = self._forward_parameters().get("max_inference_sigma")
        effective = {
            "num_loops": chosen("num_loops", lambda: config.num_loops),
            "num_diffusion_samples": chosen(
                "num_diffusion_samples",
                lambda: getattr(config, "num_diffusion_samples", None),
            ),
            "num_sampling_steps": chosen(
                "num_sampling_steps", lambda: getattr(head, "inference_num_steps", None)
            ),
            "noise_scale": chosen(
                "noise_scale", lambda: getattr(head, "noise_scale", None)
            ),
            "step_scale": chosen(
                "step_scale", lambda: getattr(head, "step_scale", None)
            ),
            "max_inference_sigma": chosen(
                "max_inference_sigma",
                lambda: (
                    None
                    if sigma is None or sigma.default is inspect.Parameter.empty
                    else sigma.default
                ),
            ),
            "msa_max_depth": chosen(
                "msa_max_depth", lambda: getattr(msa, "max_depth", None)
            ),
            "msa_column_mask_rate": chosen(
                "msa_column_mask_rate", lambda: getattr(msa, "column_mask_rate", None)
            ),
            "lm_mask_pct": chosen("lm_mask_pct", lambda: config.lm_mask_pct)
            if lm_source == "model"
            else None,
            "lm_dropout": args.get("lm_dropout") or self._configured_lm_dropout(),
        }
        return {
            f"esmfold2.effective.{name}": value
            for name, value in sorted(effective.items())
        }

    def _configured_lm_dropout(self) -> float:
        """The LM dropout the checkpoint's config applies when a call sets none.

        The release model drops out per loop only when its LM encoder config
        asks for it; the experimental model's shim applies its configured rate
        on every fold.
        """
        config = self.config
        encoder = getattr(config, "lm_encoder", None)
        if encoder is not None and getattr(config, "type", None) != "experimental":
            rate = float(getattr(encoder, "lm_dropout", 0.0) or 0.0)
            return rate if getattr(encoder, "per_loop_lm_dropout", False) else 0.0
        return float(getattr(config, "lm_dropout", 0.0) or 0.0)

    def _call_record(
        self, spi: Any, kwargs: dict[str, Any], lm_source: str
    ) -> dict[str, Any]:
        """This call's ``esmfold2.*`` entries: what was asked, what ran, on what.

        ``esmfold2.fold.<name>`` is every argument of upstream's ``fold`` call,
        its defaults included -- what was requested. ``esmfold2.effective.*``
        is what the model executed with, every ``None`` resolved against the
        live module (:meth:`_effective_settings`). :func:`input_record` names every entity folded and every
        condition by digest, so a record says which sequence and which
        alignment ran, not only where they came from.
        """
        args = self._fold_arguments(kwargs)
        return (
            {f"esmfold2.fold.{name}": value for name, value in sorted(args.items())}
            | self._effective_settings(args, lm_source)
            | {"esmfold2.lm_source": lm_source}
            | input_record(spi)
        )

    def _fold_arguments(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """What ``ESMFold2InputBuilder.fold`` runs with: its defaults, then ``kwargs``.

        Read off upstream's signature, so a default upstream changes is the one
        recorded, and an argument it does not declare raises as upstream would.
        """
        parameters = inspect.signature(type(self.builder).fold).parameters
        declared = set(parameters) - {"self"} - _FOLD_POSITIONALS
        takes_any = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        unknown = set() if takes_any else set(kwargs) - declared
        if unknown:
            raise TypeError(
                f"fold() got unexpected keyword argument(s) {sorted(unknown)}"
            )
        args = {
            name: parameter.default
            for name, parameter in parameters.items()
            if name in declared and parameter.default is not inspect.Parameter.empty
        }
        args.update(kwargs)
        return args

    def _fold_with_lm_states(
        self, spi: Any, kwargs: dict[str, Any], lm_hidden_states: Any
    ) -> Any:
        """``ESMFold2InputBuilder.fold``, carrying ``lm_hidden_states`` to ``forward``.

        Upstream's ``fold`` has a fixed keyword list without
        ``lm_hidden_states``, although ``forward`` declares it. This is that
        method's body with the one argument added: its defaults are read off
        its signature, and seeding, the dropout context and decoding are its
        own helpers, called as it calls them. The helpers are private upstream;
        this is the one place that imports them. A signature that has moved
        away from :data:`REPLICATED_FOLD_PARAMETERS` raises, because the
        replica would silently drop what it does not know.
        """
        from contextlib import nullcontext

        import torch
        from esm.models.esmfold2.processor import _lm_dropout_context, _seed_context

        known = set(inspect.signature(type(self.builder).fold).parameters) - {"self"}
        if known != REPLICATED_FOLD_PARAMETERS:
            raise NotImplementedError(
                "ESMFold2InputBuilder.fold's parameters have changed "
                f"(added {sorted(known - REPLICATED_FOLD_PARAMETERS)}, removed "
                f"{sorted(REPLICATED_FOLD_PARAMETERS - known)}); the fold that "
                "carries lm_hidden_states replicates it and must be brought up to "
                "date before it can be trusted"
            )
        args = self._fold_arguments(kwargs)

        if (
            args["early_exit"] is not None
            or args["msa_subsample_at_inference"] is not None
        ):
            warnings.warn(
                "fold(): ignoring early_exit and msa_subsample_at_inference. "
                "Use msa_max_depth instead; early_exit was never supported.",
                DeprecationWarning,
                stacklevel=3,
            )

        seed = args["seed"]
        features, chain_infos = self.builder.prepare_input(
            spi, seed=seed, device=self.net.device
        )
        self._check_lm_hidden_states(lm_hidden_states, features)

        sampler_kwargs = {
            name: args[name]
            for name in ("noise_scale", "step_scale", "max_inference_sigma")
            if args[name] is not None
        }
        with (
            torch.no_grad(),
            _seed_context(seed) if seed is not None else nullcontext(),
            _lm_dropout_context(self.net, args["lm_dropout"]),
        ):
            output = self.net(
                **features,
                lm_hidden_states=lm_hidden_states,
                num_loops=args["num_loops"],
                num_sampling_steps=args["num_sampling_steps"],
                num_diffusion_samples=args["num_diffusion_samples"],
                msa_max_depth=args["msa_max_depth"],
                msa_column_mask_rate=args["msa_column_mask_rate"],
                include_embeddings=args["include_embeddings"],
                **sampler_kwargs,
            )

        return self.builder.decode(
            output,
            features,
            chain_infos,
            num_diffusion_samples=args["num_diffusion_samples"],
            complex_id=args["complex_id"],
        )

    def fold_atom_array(
        self,
        atoms: AtomArray,
        *,
        chain_info: dict | None = None,
        config: FoldingConfig | None = None,
        ligand_residue_name: str | None = None,
        adapter_kwargs: dict[str, Any] | None = None,
        lm_hidden_states: Any | None = None,
        record: dict[str, Any] | None = None,
        bonds: bool = True,
        **overrides: Any,
    ) -> tuple[AtomArray, Any] | tuple[list[AtomArray], list[Any]]:
        """Fold an AtomWorks structure and answer with one.

        This is the whole AtomWorks round trip in a single call::

            AtomArray -> StructurePredictionInput -> ESMFold2 -> AtomArray

        Note:
            Strict by default. Anything that would fold a different molecule
            than the one described -- an unsupported chain, a covalent bond
            that cannot be placed, a chain kind that would have to be guessed,
            a non-standard residue whose position is unknown -- raises, because
            this path returns no report and a recorded-but-silent degradation
            would be invisible here. Accept one by name through
            ``adapter_kwargs={"allow_<name>": True}``; the names are
            :data:`esmfold2_atomworks.data.atomworks_to_esm.DEGRADATIONS`. To see
            what happened under an opt-in, pass an ``AdapterReport`` as
            ``adapter_kwargs={"report": report}`` and read
            ``report.accepted_degradations()`` afterwards. A chain kind,
            sequence override, MSA or ``LigandSpec`` passed the same way must
            bind to the chain it names, or it raises ``ChainDeclarationError``
            -- with no opt-in. For a structure that carries no ``chain_type``,
            ``adapter_kwargs={"chain_kinds": {...}}`` says what each chain is.

        Args:
            ligand_residue_name: the residue name to give every hetero residue
                of the returned structure. ESMFold2 writes ``LIG`` on a ligand
                it was given as SMILES -- the model's label, not the caller's --
                and anything that matches ligands by name, from a parameter file
                to atom correspondence between two structures, needs the
                caller's. **Every** hetero residue means every non-polymer
                chain, an ion or cofactor beside the ligand included, so this is
                for an output whose hetero residues are all one molecule. For
                several, declare each with a ``LigandSpec`` in
                ``adapter_kwargs["ligands"]``; each declared chain gets its
                spec's label. Giving both is refused unless every declared
                chain's label equals this name. Checked before the fold: a name longer than a
                residue name's five characters raises rather than being
                truncated.
            lm_hidden_states: as for :meth:`fold`.
            record: as for :meth:`fold`, plus ``esmfold2.sequence_source``:
                where each chain's folded sequence came from
                (``AdapterReport.sequence_source``).
            bonds: give the returned structure a ``BondList``, rebuilt from the
                chemistry the model was given (see
                :mod:`esmfold2_atomworks.data.topology`). ``TopologyError`` when
                the structure and the input disagree; ``False`` returns the
                structure without bonds.

        Returns:
            ``(atom_array, result)`` -- the structure, and the native result
            beside it so that confidence values remain available without being
            smuggled through annotations. With ``num_diffusion_samples > 1``,
            a list of each, and every sample is relabelled. When ``atoms``
            carries ``chain_type``, each output chain carries its source
            chain's (:func:`~esmfold2_atomworks.data.molecular_complex.copy_chain_types`);
            a kind known only from ``chain_kinds`` is not turned into one.
        """
        from esmfold2_atomworks.data.atomworks_to_esm import (
            _index_ligand_specs,
            atom_array_to_structure_prediction_input,
        )
        from esmfold2_atomworks.data.molecular_complex import (
            apply_ligand_labels,
            check_residue_name,
            copy_chain_types,
            rename_ligand_residues,
            result_to_atom_array,
        )
        from esmfold2_atomworks.data.topology import (
            build_bond_list,
            ccd_name_collisions,
        )

        _check_record(record)
        adapter_kwargs = dict(adapter_kwargs or {})
        ligand_specs = _index_ligand_specs(adapter_kwargs.get("ligands"))
        if ligand_residue_name is not None:
            check_residue_name(ligand_residue_name)
            conflicting = sorted(
                chain
                for chain, spec in ligand_specs.items()
                if spec.label != ligand_residue_name
            )
            if conflicting:
                raise ValueError(
                    f"ligand_residue_name={ligand_residue_name!r} contradicts the "
                    f"LigandSpec of chains {conflicting}"
                )
        if record is not None and adapter_kwargs.get("report") is None:
            from esmfold2_atomworks.data.atomworks_to_esm import AdapterReport

            adapter_kwargs["report"] = AdapterReport()
        spi = atom_array_to_structure_prediction_input(
            atoms, chain_info=chain_info, **adapter_kwargs
        )
        result = self.fold(
            spi,
            config=config,
            lm_hidden_states=lm_hidden_states,
            record=record,
            **overrides,
        )
        if record is not None:
            from esmfold2_atomworks.data.loading import atomworks_version

            record["esmfold2.sequence_source"] = dict(
                adapter_kwargs["report"].sequence_source
            )
            record["atomworks.version"] = atomworks_version()
            record["esmfold2.bonds"] = bool(bonds)

        chain_key = (adapter_kwargs or {}).get("chain_key", "chain_id")
        collided: set[str] = set()

        def structure(one: Any) -> AtomArray:
            folded = result_to_atom_array(one)
            if ligand_specs:
                folded = apply_ligand_labels(folded, ligand_specs)
            if ligand_residue_name is not None:
                folded = rename_ligand_residues(folded, ligand_residue_name)
            folded = copy_chain_types(folded, atoms, chain_key=chain_key)
            if bonds:
                folded.bonds = build_bond_list(folded, spi)
            collided.update(ccd_name_collisions(folded))
            return folded

        built = (
            [structure(r) for r in result]
            if isinstance(result, list)
            else structure(result)
        )
        if record is not None and collided:
            record["esmfold2.ccd_name_collisions"] = sorted(collided)
        return built, result

    def provenance(self) -> dict[str, str]:
        """What this model is and where it runs: one entry per fact, as strings.

        The weights by path and, where the directory says, by repo and
        revision (``paths.checkpoint_identity``); the device resolved to an
        index, with its name; torch and its CUDA build. What a single fold was
        asked and ran with is the call's own record (``fold(record=...)``).
        """
        import torch

        resolved = Path(getattr(self, "weights_resolved", self.weights))
        checkpoint = (
            paths.checkpoint_identity(resolved)
            if resolved.is_dir()
            else {"repo": self.weights, "revision": ""}
        )
        device = self._placement()
        device_name = (
            torch.cuda.get_device_name(device) if device.type == "cuda" else device.type
        )
        return {
            "esmfold2.weights": self.weights,
            "esmfold2.weights_resolved": str(resolved),
            "esmfold2.checkpoint.repo": checkpoint.get("repo", ""),
            "esmfold2.checkpoint.revision": checkpoint.get("revision", ""),
            "esmfold2.checkpoint.versioning": checkpoint.get(
                "versioning", "unresolved"
            ),
            "esmfold2.checkpoint.config_sha256": checkpoint.get("config_sha256", ""),
            "esmfold2.device": str(device),
            "esmfold2.device_name": device_name,
            "esmfold2.torch": torch.__version__,
            "esmfold2.torch_cuda": str(torch.version.cuda),
            "esmfold2.config_type": str(getattr(self.config, "type", "release")),
            # Numerics the wrapper chose at construction: the backbone's
            # precision (bf16 or fp8) changes the LM states; the chunk size and
            # kernel backend change the order of reductions.
            "esmfold2.esmc_precision": str(getattr(self, "esmc_precision", "")),
            "esmfold2.chunk_size": str(getattr(self, "chunk_size", "")),
            "esmfold2.kernel_backend": str(getattr(self, "kernel_backend", "")),
            # "bundled", "none", or where a separate backbone was loaded from.
            "esmfold2.esmc": self.esmc_source,
            # The CCD pickle ligand and modified-residue conformers come from.
            "esmfold2.ccd": self.ccd_source,
        } | self._esmc_identity()

    def _applied_esmc_precision(self, requested: str) -> str:
        """The precision the backbone actually runs in, not the one requested.

        The experimental model's loader discards the argument and always loads
        the backbone in bf16; a model without a backbone has none.
        """
        if self.esmc_source == "none":
            return "none: no backbone"
        if getattr(self.config, "type", None) == "experimental":
            return "bf16"
        return str(requested)

    def _placement(self) -> Any:
        """Where the module's parameters are, read off the module itself.

        Not ``torch.cuda.current_device()``: that is the process's current
        device, which can change after the model was placed.
        """
        import torch

        device = getattr(self.net, "device", None)
        if device is not None:
            return torch.device(device)
        parameters = getattr(self.net, "parameters", None)
        if callable(parameters):
            first = next(iter(parameters()), None)
            if first is not None:
                return first.device
        return self.device

    def _esmc_identity(self) -> dict[str, str]:
        """A separately attached backbone's repo and revision, when its directory says."""
        if not Path(self.esmc_source).is_dir():
            return {}
        identity = paths.checkpoint_identity(Path(self.esmc_source))
        return {
            "esmfold2.esmc.repo": identity.get("repo", ""),
            "esmfold2.esmc.revision": identity.get("revision", ""),
        }
