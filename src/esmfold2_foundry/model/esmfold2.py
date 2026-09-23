"""The Foundry-facing ESMFold2 model.

Phase 2 of the plan: ESMFold2 becomes a Foundry model **without its architecture
being rewritten**. ``FoundryESMFold2`` holds the native ESMFold2 module and
delegates to it; Foundry contributes the dataset, trainer, config, distributed
execution, logging and checkpointing around it.

Which module that is depends on the ``esm`` version -- up to 3.3 it lived in a
fork of ``transformers``, from 3.4 it is in ``esm`` itself under a slightly
different name. :func:`load_native_model_class` resolves both.

The reason for the indirection is not ceremony. Both AtomWorks and the Biohub
fork are explicitly mid-cleanup -- AtomWorks' README says so outright -- so a
full rewrite would spend its first months chasing upstream API changes, and any
subtle divergence would show up as a model that runs, reports plausible
confidence, and is quietly wrong. Wrapping keeps the weights and the numerics
exactly as published, and leaves one seam per component for Phase 3 to open.

Three behaviours of the native model are worth knowing before you wire
anything to it; each is asserted or surfaced below rather than left as folklore.

1. **Release ``forward`` is decorated ``@torch.inference_mode()``.** Its outputs
   cannot enter an autograd graph. Training therefore needs
   ``ESMFold2ExperimentalModel``, whose ``forward`` is not so decorated -- see
   :attr:`FoundryESMFold2.supports_gradients`.
2. **``fold()`` accepts sampler knobs the release model ignores.**
   ``noise_scale``, ``step_scale``, ``max_inference_sigma`` and ``early_exit``
   are forwarded into ``forward(**kwargs)`` and silently discarded; only
   ``lm_mask_pct`` is a declared parameter. Passing them and expecting an
   effect is a real trap, so :meth:`fold` names them.
3. **pLDDT is on 0--1, not 0--100**, and ``result.plddt`` (model tokens) is a
   different length from ``result.complex.plddt`` (collapsed residues) whenever
   a ligand or modified residue is present. Do not index one with the other.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from esmfold2_foundry import paths

if TYPE_CHECKING:
    from biotite.structure import AtomArray

__all__ = ["FoldingConfig", "FoundryESMFold2", "load_native_model_class"]

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
            "No ESMFold2 model class found. Either install esm >= 3.4 (which "
            "ships the model), or, for esm <= 3.3, install the Biohub fork of "
            "transformers that carries it. See docs/04_ENVIRONMENT.md."
        ) from error


@dataclass
class FoldingConfig:
    """Inference-time knobs, mirroring the SDK's ``FoldingConfig``.

    Defaults follow the shipped ``biohub/ESMFold2`` config rather than the
    dataclass defaults in ``configuration_esmfold2.py``, which differ
    substantially (that config ships ``num_loops=3``, not 20).
    """

    num_loops: int = 3
    num_sampling_steps: int = 100
    num_diffusion_samples: int = 1
    lm_dropout: float | None = 0.3
    lm_mask_pct: float | None = None
    seed: int | None = None
    #: Forwarded for completeness; see SILENTLY_IGNORED_BY_RELEASE.
    extra: dict[str, Any] = field(default_factory=dict)

    def as_fold_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
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


class FoundryESMFold2:
    """A resident ESMFold2, fed from AtomWorks and answering in AtomWorks terms.

    This is intentionally *not* an ``nn.Module`` subclass yet. Until Phase 3
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
            # avoids materializing it on CPU first.
            self.net = model_class.from_pretrained(
                self.weights,
                load_esmc=load_esmc,
                esmc_precision=esmc_precision,
                device=str(self.device),
            ).eval()
        else:
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
        # (~9 s), so one builder is shared by every call.
        self.builder = ESMFold2InputBuilder(ccd_cache=ccd_cache)

    # -- introspection -----------------------------------------------------

    @property
    def config(self) -> Any:
        """The live config. Read dimensions off this, never off the dataclass defaults."""
        return self.net.config

    @property
    def supports_gradients(self) -> bool:
        """Whether ``forward`` can participate in autograd.

        The release model's ``forward`` is ``@torch.inference_mode()``, so it
        cannot. Phase 3 training must use the experimental variant, and finding
        that out from a wrong-looking loss curve is much worse than from here.
        """
        return getattr(self.config, "type", "release") == "experimental"

    def representation_dims(self) -> dict[str, int]:
        """The single/pair widths, for comparison against RFD3's ``c_s``/``c_z``."""
        config = self.config
        structure = getattr(config, "structure_head", None)
        diffusion = getattr(structure, "diffusion_module", None) if structure else None
        dims = {
            "d_pair": int(getattr(config, "d_pair", 0)),
            "d_single_declared": int(getattr(config, "d_single", 0)),
        }
        if diffusion is not None:
            dims["c_token"] = int(getattr(diffusion, "c_token", 0))
            dims["c_atom"] = int(getattr(diffusion, "c_atom", 0))
            dims["c_s_inputs"] = int(getattr(diffusion, "c_s_inputs", 0))
        return dims

    # -- Phase 3 seams -----------------------------------------------------
    # Named accessors for the three components the plan eventually separates.
    # They exist now so that a later change is a change of implementation
    # rather than a change of every call site.

    @property
    def esmc(self) -> Any:
        """The frozen ESMC language model backbone."""
        return getattr(self.net, "_esmc", None)

    @property
    def folding_trunk(self) -> Any:
        """The pair-representation trunk."""
        return getattr(self.net, "folding_trunk", None)

    @property
    def structure_head(self) -> Any:
        """The diffusion structure head."""
        return getattr(self.net, "structure_head", None)

    # -- the model as a function ------------------------------------------

    def featurize(self, spi: Any, *, seed: int | None = None) -> tuple[dict, list]:
        """``StructurePredictionInput`` -> ``(features, chain_infos)`` on this device.

        ``chain_infos`` is not optional bookkeeping: it is the only record of
        ``token -> (chain, residue, atom span)``, and ``decode`` cannot run
        without it.
        """
        return self.builder.prepare_input(spi, seed=seed, device=self.device)

    def fold(
        self,
        spi: Any,
        config: FoldingConfig | None = None,
        **overrides: Any,
    ) -> Any:
        """Fold a ``StructurePredictionInput``.

        Returns a single ``MolecularComplexResult``, or a list of them when
        ``num_diffusion_samples > 1`` -- the native ``decode`` collapses the
        one-sample case, and callers must handle both.
        """
        import torch

        config = config or FoldingConfig()
        kwargs = config.as_fold_kwargs()
        kwargs.update(overrides)

        ignored = [k for k in SILENTLY_IGNORED_BY_RELEASE if k in kwargs]
        if ignored and not self.supports_gradients:
            warnings.warn(
                f"{ignored} are accepted by ESMFold2InputBuilder.fold but are not "
                "declared by the release ESMFold2Model.forward, so they are "
                "discarded. Set them on config.structure_head instead.",
                RuntimeWarning,
                stacklevel=2,
            )

        with torch.no_grad():
            return self.builder.fold(self.net, spi, **kwargs)

    def fold_atom_array(
        self,
        atoms: AtomArray,
        *,
        chain_info: dict | None = None,
        config: FoldingConfig | None = None,
        adapter_kwargs: dict[str, Any] | None = None,
        **overrides: Any,
    ) -> tuple[AtomArray, Any]:
        """Fold an AtomWorks structure and answer with one.

        This is the whole Phase 1 loop in a single call::

            AtomArray -> StructurePredictionInput -> ESMFold2 -> AtomArray

        Returns:
            ``(atom_array, result)`` -- the structure, and the native result
            beside it so that confidence values remain available without being
            smuggled through annotations.
        """
        from esmfold2_foundry.data.atomworks_to_esm import (
            atom_array_to_structure_prediction_input,
        )
        from esmfold2_foundry.data.molecular_complex import result_to_atom_array

        spi = atom_array_to_structure_prediction_input(
            atoms, chain_info=chain_info, **(adapter_kwargs or {})
        )
        result = self.fold(spi, config=config, **overrides)
        if isinstance(result, list):
            return [result_to_atom_array(r) for r in result], result
        return result_to_atom_array(result), result

    def provenance(self) -> dict[str, str]:
        return {
            "esmfold2.weights": self.weights,
            "esmfold2.device": str(self.device),
            "esmfold2.config_type": str(getattr(self.config, "type", "release")),
            # Which packaging supplied the module; the two are different code
            # paths, so a result is only comparable against one of them.
            "esmfold2.flavour": self.flavour,
        }
