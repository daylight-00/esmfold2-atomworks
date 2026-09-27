"""The ESMFold2 model wrapper.

ESMFold2, driven from AtomWorks **without its architecture being rewritten**.
``AtomWorksESMFold2`` holds the native ESMFold2 module and delegates to it;
whatever trains or serves it -- the optional Foundry integration in
:mod:`esmfold2_atomworks.training`, or anything else -- works with the native
module rather than a reimplementation.

The module ships in ``esm`` >= 3.4. Up to 3.3 it lived in a fork of
``transformers``, under a slightly different name; :func:`load_native_model_class`
still recognises that layout, though the fork is no longer published.

The reason for the indirection is not ceremony. Both upstreams move fast --
AtomWorks' README calls it mid-cleanup, and the ESMFold2 module has changed
homes twice -- so a full rewrite would spend its first months chasing upstream
API changes, and any
subtle divergence would show up as a model that runs, reports plausible
confidence, and is quietly wrong. Wrapping keeps the weights and the numerics
exactly as published, and leaves one named seam per component.

Four behaviours of the native model are worth knowing before you wire
anything to it; each is asserted or surfaced below rather than left as folklore.

1. **Gradients depend on the inputs, not just the checkpoint.** The release
   ``forward`` is ``@torch.inference_mode()`` and can never produce them. The
   experimental one gates autograd on ``res_type_soft`` being supplied, so
   loading it is necessary and not sufficient -- see
   :meth:`AtomWorksESMFold2.will_produce_gradients`.
2. **``fold()`` accepts sampler knobs the release model ignores.**
   ``noise_scale``, ``step_scale``, ``max_inference_sigma`` and ``early_exit``
   are forwarded into ``forward(**kwargs)`` and silently discarded; only
   ``lm_mask_pct`` is a declared parameter. Passing them and expecting an
   effect is a real trap, so :meth:`fold` names them.
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
    "AtomWorksESMFold2",
    "FoldingConfig",
    "MissingLanguageModelError",
    "attach_esmc",
    "ccd_source",
    "load_native_model_class",
]

#: The input that switches the experimental forward into a gradient-bearing
#: mode. Upstream gates autograd on it directly --
#: ``torch.set_grad_enabled(res_type_soft is not None)`` -- so its presence,
#: not the checkpoint flavour alone, is what decides whether a loss is
#: differentiable.
GRADIENT_GATE = "res_type_soft"

#: ``fold()`` forwards these into the release ``forward``, which does not
#: declare them, so they land in ``**kwargs`` and are dropped: the sampler is
#: called with hardcoded defaults. ``early_exit`` is additionally deprecated in
#: esm >= 3.4, where passing it raises a ``DeprecationWarning`` and does nothing
#: ("early_exit was never supported").
SILENTLY_IGNORED_BY_RELEASE = (
    "noise_scale",
    "step_scale",
    "max_inference_sigma",
    "early_exit",
)


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


class MissingLanguageModelError(RuntimeError):
    """A fold would run without the LM prior the checkpoint was trained with.

    The native ``forward`` skips the language-model pathway when no ESMC
    backbone is attached and no hidden states are given, and folds anyway. That
    is not the published model with a smaller input: its LM shim maps even the
    all-zero states of a protein-free input to a non-zero pair term, so leaving
    the pathway out changes every fold, with or without protein chains.
    """


def load_native_model_class() -> tuple[type, str]:
    """The native ESMFold2 class, and which packaging it came from.

    ESMFold2 moved house. Up to esm 3.3 the ``esm`` package shipped only the
    input pipeline and the ``nn.Module`` lived in a fork of ``transformers``;
    from esm 3.4 the model is in ``esm`` itself and the fork is gone. The two
    also differ in name (``ESMFold2Model`` vs ``EsmFold2Model``) and in how the
    device is chosen, so supporting both is a few lines here rather than a
    version constraint the caller has to satisfy.

    Returns:
        ``(class, flavour)`` where flavour is ``"esm"`` (>= 3.4) or
        ``"transformers-fork"`` (<= 3.3).
    """
    try:
        from esm.models.esmfold2.model import EsmFold2Model

        return EsmFold2Model, "esm"
    except ImportError:
        pass
    try:
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

        return ESMFold2Model, "transformers-fork"
    except ImportError as error:
        raise ImportError(
            "No ESMFold2 model class found. Install esm >= 3.4, which ships "
            "the model. (esm <= 3.3 relied on a fork of transformers that is no "
            "longer published.) See docs/04_ENVIRONMENT.md."
        ) from error


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


def attach_esmc(net: Any, precision: str = "bf16") -> str:
    """Attach the ESMC backbone ``net``'s checkpoint needs; say where it came from.

    Returns ``"bundled"`` for a checkpoint that carries its backbone, which the
    native loader has already attached. Otherwise the checkpoint names a
    separate one in ``config.esmc_id``, which is resolved through
    :func:`esmfold2_atomworks.paths.resolve_esmc` -- a local mirror first --
    rather than handed to ``from_pretrained`` as written. The resolved location
    is returned.

    ``precision`` reaches the release model only: the experimental model's
    ``load_esmc`` takes none and always loads bf16, as its own
    ``from_pretrained`` does.
    """
    if _bundles_esmc(net.config):
        return "bundled"
    source = str(paths.resolve_esmc(net.config.esmc_id))
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


@dataclass
class FoldingConfig:
    """Inference-time knobs, mirroring the SDK's ``FoldingConfig``.

    ``num_loops=None`` means the checkpoint's own ``config.num_loops``, which
    is what the model uses when it is given no count -- and the checkpoints
    disagree: 3 in the reference checkpoint, 20 in the one ``biohub/ESMFold2``
    publishes today (docs/04). A fixed default here would silently override
    whichever of them is loaded. Set it to pin a schedule.
    """

    num_loops: int | None = None
    num_sampling_steps: int = 100
    num_diffusion_samples: int = 1
    lm_dropout: float | None = 0.3
    lm_mask_pct: float | None = None
    seed: int | None = None
    #: Forwarded for completeness; see SILENTLY_IGNORED_BY_RELEASE.
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
        kwargs.update(self.extra)
        return kwargs


class AtomWorksESMFold2:
    """A resident ESMFold2, fed from AtomWorks and answering in AtomWorks terms.

    This is intentionally *not* an ``nn.Module`` subclass yet. Until something
    introduces trainable parameters of its own, wrapping in a module would add a
    parameter namespace that every checkpoint has to agree about, for no gain.
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
        self.device = torch.device(device)
        model_class, self.flavour = load_native_model_class()

        if self.flavour == "esm":
            # esm >= 3.4 places the model on `device` during construction, which
            # avoids materializing it on CPU first. A bundled backbone comes in
            # with the trunk regardless of `load_esmc`; a separate one is
            # attached here rather than from the checkpoint's `esmc_id` (see
            # attach_esmc).
            self.net = model_class.from_pretrained(
                self.weights,
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
        else:
            self.esmc_source = "checkpoint esmc_id" if load_esmc else "none"
            self.net = (
                model_class.from_pretrained(
                    self.weights, load_esmc=load_esmc, esmc_precision=esmc_precision
                )
                .to(self.device)
                .eval()
            )

        # Both are performance knobs and neither is part of the model contract,
        # so their absence on a future revision is not an error.
        if chunk_size is not None and hasattr(self.net, "set_chunk_size"):
            self.net.set_chunk_size(chunk_size)
        if kernel_backend is not None and hasattr(self.net, "set_kernel_backend"):
            # torch.compile and the Triton kernels do not stack; upstream
            # requires clearing the backend before compiling.
            self.net.set_kernel_backend(kernel_backend)

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

        Deliberately not named ``supports_gradients``: that name invited
        exactly the reading that loading the experimental checkpoint was
        enough, which produces a loss with no ``grad_fn`` and a flat training
        curve whose cause has to be guessed at.
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

        Read off the live config, under the names it uses now: ``EsmFold2Config``
        migrates a pre-alignment ``config.json`` on load and drops the old field
        names, so ``d_pair`` or ``c_token`` are absent from a config that was
        written with them. The old name is read only when the new one is
        missing, for a module that predates the migration. A width that is
        missing under both, or not positive, raises: an unknown dimension
        reported as ``0`` reads downstream as a measurement.

        The keys are this method's own and did not move with upstream.
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
    # Named accessors for the three components the plan eventually separates.
    # They exist now so that a later change is a change of implementation
    # rather than a change of every call site. Only `.esmc` has a real absent
    # state; the trunk and the head raise rather than answer None, which a
    # caller would read as a component that is legitimately missing.

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
            lm_mask_pct = getattr(self.config, "lm_mask_pct", 0.0)

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
            record: a dict to write this call's ``esmfold2.*`` entries into:
                ``esmfold2.lm_source``, one of :data:`LM_SOURCES`. The caller's
                other keys are kept; an ``esmfold2.*`` key already present
                raises, so a record reused from an earlier fold cannot mix two
                calls' entries.

        Raises:
            MissingLanguageModelError: no backbone is resident and no
                ``lm_hidden_states`` are given.
        """
        _check_record(record)
        config = config or FoldingConfig()
        kwargs = config.as_fold_kwargs()
        kwargs.update(overrides)

        ignored = [k for k in SILENTLY_IGNORED_BY_RELEASE if k in kwargs]
        if ignored and not self.supports_soft_sequence_design:
            warnings.warn(
                f"{ignored} are accepted by ESMFold2InputBuilder.fold but are not "
                "declared by the release ESMFold2Model.forward, so they are "
                "discarded. Set them on config.structure_head instead.",
                RuntimeWarning,
                stacklevel=2,
            )

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
            result = self._fold_with_lm_states(spi, kwargs, lm_hidden_states)
            lm_source = "caller-supplied"

        if record is not None:
            record["esmfold2.lm_source"] = lm_source
        return result

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

        parameters = inspect.signature(type(self.builder).fold).parameters
        known = set(parameters) - {"self"}
        if known != REPLICATED_FOLD_PARAMETERS:
            raise NotImplementedError(
                "ESMFold2InputBuilder.fold's parameters have changed "
                f"(added {sorted(known - REPLICATED_FOLD_PARAMETERS)}, removed "
                f"{sorted(REPLICATED_FOLD_PARAMETERS - known)}); the fold that "
                "carries lm_hidden_states replicates it and must be brought up to "
                "date before it can be trusted"
            )
        unknown = set(kwargs) - known
        if unknown:
            raise TypeError(
                f"fold() got unexpected keyword argument(s) {sorted(unknown)}"
            )
        args = {
            name: parameter.default
            for name, parameter in parameters.items()
            if parameter.default is not inspect.Parameter.empty
        }
        args.update(kwargs)

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
        **overrides: Any,
    ) -> tuple[AtomArray, Any]:
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
                several, omit it and relabel chain by chain. Checked before the
                fold: a name longer than a residue name's five characters
                raises rather than being truncated.

        Returns:
            ``(atom_array, result)`` -- the structure, and the native result
            beside it so that confidence values remain available without being
            smuggled through annotations. With ``num_diffusion_samples > 1``,
            a list of each, and every sample is relabelled.
        """
        from esmfold2_atomworks.data.atomworks_to_esm import (
            atom_array_to_structure_prediction_input,
        )
        from esmfold2_atomworks.data.molecular_complex import (
            check_residue_name,
            result_to_atom_array,
        )

        _check_record(record)
        if ligand_residue_name is not None:
            check_residue_name(ligand_residue_name)
        spi = atom_array_to_structure_prediction_input(
            atoms, chain_info=chain_info, **(adapter_kwargs or {})
        )
        result = self.fold(
            spi,
            config=config,
            lm_hidden_states=lm_hidden_states,
            record=record,
            **overrides,
        )

        def structure(one: Any) -> AtomArray:
            return result_to_atom_array(one, ligand_residue_name=ligand_residue_name)

        if isinstance(result, list):
            return [structure(r) for r in result], result
        return structure(result), result

    def provenance(self) -> dict[str, str]:
        return {
            "esmfold2.weights": self.weights,
            "esmfold2.device": str(self.device),
            "esmfold2.config_type": str(getattr(self.config, "type", "release")),
            # Which packaging supplied the module; the two are different code
            # paths, so a result is only comparable against one of them.
            "esmfold2.flavour": self.flavour,
            # "bundled", "none", or where a separate backbone was loaded from.
            "esmfold2.esmc": self.esmc_source,
            # The CCD pickle ligand and modified-residue conformers come from.
            "esmfold2.ccd": self.ccd_source,
        }
