"""Foundry ``FabricTrainer`` integration for ESMFold2.

Status: **the contract is implemented, the gradient path is blocked upstream.**
That is a property of the released model, not an omission here, and it is worth
stating precisely because it determines what Phase 3 has to do first.

``transformers.models.esmfold2.ESMFold2Model.forward`` is decorated
``@torch.inference_mode()``. Tensors produced under that decorator are marked
inference tensors: they carry no autograd history and cannot be made to. So a
loss computed from a release-model forward has nothing to differentiate, and no
amount of trainer engineering changes it. ``ESMFold2ExperimentalModel.forward``
carries no such decorator and additionally accepts ``res_type_soft`` for
soft-sequence design, which is why it -- not the release model -- is the
starting point for the generative surgery in Phase 3.

:meth:`ESMFold2Trainer.training_step` therefore refuses a release model with an
explanation, instead of producing a loss that never decreases and leaving the
reason to be discovered from a flat curve.

What *is* wired up: the Foundry trainer contract (``training_step`` /
``validation_step``, ``self.state``, ``construct_model``), so that a working
forward is the only missing piece rather than the whole integration.
"""

from __future__ import annotations

from typing import Any

# ``ESMFold2Trainer`` is supplied by the module-level ``__getattr__`` (PEP 562)
# at the bottom of this file, so that importing this module does not import
# Foundry -- whose ``__init__`` installs beartype/jaxtyping import hooks when
# TYPE_CHECK is set. It resolves normally for ``hydra`` ``_target_`` strings and
# for ``from ... import ESMFold2Trainer``; only static analysis cannot see it.
__all__ = ["ESMFold2Trainer", "GradientsUnavailableError"]  # noqa: F822


class GradientsUnavailableError(RuntimeError):
    """The loaded ESMFold2 cannot participate in autograd."""


def _fabric_trainer_base() -> Any:
    from foundry.trainers.fabric import FabricTrainer

    return FabricTrainer


def make_trainer_class() -> Any:
    """Build the trainer class, importing Foundry lazily.

    Foundry's ``__init__`` installs beartype/jaxtyping import hooks when
    ``TYPE_CHECK`` is set, so importing it at module scope would impose that on
    anyone who only wants the adapter.
    """
    FabricTrainer = _fabric_trainer_base()

    class ESMFold2Trainer(FabricTrainer):
        """Trains ESMFold2 inside Foundry's Fabric loop.

        The two abstract methods of ``FabricTrainer`` are ``training_step`` and
        ``validation_step``; everything else -- optimizer and scheduler
        construction, checkpointing, DDP, EMA, logging -- is inherited.

        Note that ``training_step`` returns ``None``: Foundry's convention is to
        call ``self.fabric.backward(loss)`` inside the step and stash detached
        outputs on ``self._current_train_return`` for the callbacks to read.
        """

        def __init__(
            self,
            *,
            loss: Any = None,
            metrics: Any = None,
            seed: int | None = None,
            require_gradients: bool = True,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.loss_cfg = loss
            self.metrics_cfg = metrics
            self.seed = seed
            self.require_gradients = require_gradients

        # -- model -------------------------------------------------------

        def construct_model(self) -> None:
            """Instantiate the wrapper and register its native module.

            Foundry's base implementation instantiates ``train_cfg.model.net``
            and optionally wraps it in EMA. ESMFold2's architecture comes from
            the checkpoint rather than from config, so this instantiates the
            wrapper and registers ``wrapper.net`` -- the actual ``nn.Module`` --
            as the trainer's model, keeping optimizer and checkpoint handling
            pointed at real parameters.
            """
            import hydra

            wrapper = hydra.utils.instantiate(
                self.state["train_cfg"].model, _recursive_=False
            )
            if self.require_gradients and not wrapper.supports_gradients:
                raise GradientsUnavailableError(_RELEASE_MODEL_MESSAGE)

            self.wrapper = wrapper
            self.initialize_or_update_trainer_state({"model": wrapper.net})

        # -- steps -------------------------------------------------------

        def _assemble_network_inputs(self, example: dict) -> dict:
            """The feature dict, batched, on the fabric device.

            The pipeline emits unbatched CPU tensors (that is what
            ``prepare_esmfold2_input`` returns); the leading batch dimension is
            added here, exactly as ``ESMFold2InputBuilder.prepare_input`` does.
            """
            import torch

            feats = example["feats"]
            return {
                key: (
                    value[None].to(self.fabric.device)
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in feats.items()
            }

        def training_step(
            self, batch: Any, batch_idx: int, is_accumulating: bool
        ) -> None:
            example = batch[0] if not isinstance(batch, dict) else batch
            model = self.state["model"]

            if self.require_gradients and not getattr(self, "wrapper", None):
                raise GradientsUnavailableError(_RELEASE_MODEL_MESSAGE)

            inputs = self._assemble_network_inputs(example)
            output = model(**inputs)
            loss = self.compute_loss(output, example)

            self.fabric.backward(loss)
            self._current_train_return = {
                "loss": loss.detach(),
                "example_id": example.get("example_id"),
            }

        def validation_step(
            self, batch: Any, batch_idx: int, val_loader_name: str | None = None
        ) -> dict:
            import torch

            example = batch[0] if not isinstance(batch, dict) else batch
            model = self.state["model"]
            with torch.no_grad():
                output = model(**self._assemble_network_inputs(example))
            return {
                "example_id": example.get("example_id"),
                **self.compute_metrics(output, example),
            }

        # -- to be supplied by Phase 3 -----------------------------------

        def compute_loss(self, output: dict, example: dict) -> Any:
            """The training objective.

            Left unimplemented on purpose. ESMFold2 ships no training loss, and
            the objective is the experiment -- diffusion loss on the structure,
            distogram loss on the trunk, or the joint latent/structure objective
            the project is aiming at. Guessing one here would put an arbitrary
            choice in the base class where it would be inherited silently.
            """
            raise NotImplementedError(
                "Supply a loss: subclass ESMFold2Trainer and implement "
                "compute_loss(output, example). See docs/05_ROADMAP.md."
            )

        def compute_metrics(self, output: dict, example: dict) -> dict:
            return {}

    return ESMFold2Trainer


_RELEASE_MODEL_MESSAGE = (
    "The release ESMFold2Model.forward is decorated @torch.inference_mode(), so "
    "its outputs carry no autograd history and cannot be trained through. Load "
    "ESMFold2ExperimentalModel (config.type == 'experimental') instead, or pass "
    "require_gradients=False to run this trainer for evaluation only."
)


def __getattr__(name: str) -> Any:
    if name == "ESMFold2Trainer":
        return make_trainer_class()
    raise AttributeError(name)
