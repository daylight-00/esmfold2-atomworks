"""``ESMFold2Trainer`` against Foundry's trainer, with the model replaced.

What is checked: the config layout the trainer reads (``model.net``,
``model.optimizer``), that the native module is what the optimizer is given, and
that the gradient preconditions refuse with their reasons. The model is a stand-in
that holds the same interface as ``AtomWorksESMFold2``; no weights, no GPU.
"""

from __future__ import annotations

import pytest

pytest.importorskip("foundry")
torch = pytest.importorskip("torch")
omegaconf = pytest.importorskip("omegaconf")


class FakeWrapper:
    """The part of ``AtomWorksESMFold2`` the trainer uses."""

    def __init__(self, soft_sequence: bool = True) -> None:
        self.net = torch.nn.Linear(3, 1)
        self.esmc = torch.nn.Linear(3, 3)
        self.soft_sequence = soft_sequence

    @property
    def supports_soft_sequence_design(self) -> bool:
        return self.soft_sequence

    def will_produce_gradients(self, inputs: dict) -> bool:
        return self.soft_sequence and "res_type_soft" in inputs

    def explain_gradient_status(self, inputs: dict) -> str:
        return "no soft sequence" if self.soft_sequence else "no experimental model"


def _trainer(**kwargs):
    from esmfold2_atomworks.training.trainer import make_trainer_class

    class Trainer(make_trainer_class()):
        def compute_loss(self, output, example):
            return output.sum()

    trainer = Trainer(
        accelerator="cpu", devices_per_node=1, strategy="auto", precision="32-true"
    )
    trainer.__dict__.update(kwargs)
    return trainer


def _config(**wrapper):
    return omegaconf.OmegaConf.create(
        {
            "model": {
                "net": {"_target_": "test_trainer_contract.FakeWrapper", **wrapper},
                "optimizer": {"_target_": "torch.optim.AdamW", "lr": 1e-3},
                "lr_scheduler": None,
                "ema": None,
            }
        }
    )


def _built(**wrapper):
    trainer = _trainer()
    trainer.initialize_or_update_trainer_state({"train_cfg": _config(**wrapper)})
    trainer.construct_model()
    return trainer


def test_the_wrapper_comes_from_model_net_and_its_module_is_registered():
    trainer = _built()
    trainer.construct_optimizer()

    assert trainer.state["model"] is trainer.wrapper.net
    (group,) = trainer.state["optimizer"].param_groups
    assert {id(p) for p in group["params"]} == {
        id(p) for p in trainer.wrapper.net.parameters()
    }


def test_the_language_model_backbone_is_frozen():
    trainer = _built()
    assert not any(p.requires_grad for p in trainer.wrapper.esmc.parameters())
    assert all(p.requires_grad for p in trainer.wrapper.net.parameters())


def test_a_model_that_cannot_produce_gradients_is_refused_at_construction():
    from esmfold2_atomworks.training.trainer import GradientsUnavailableError

    trainer = _trainer()
    trainer.initialize_or_update_trainer_state(
        {"train_cfg": _config(soft_sequence=False)}
    )
    with pytest.raises(GradientsUnavailableError, match="no experimental model"):
        trainer.construct_model()


def test_a_step_without_a_soft_sequence_is_refused_with_the_reason():
    from esmfold2_atomworks.training.trainer import GradientsUnavailableError

    trainer = _built()
    example = {"feats": {"res_type": torch.zeros(4, dtype=torch.long)}}
    with pytest.raises(GradientsUnavailableError, match="no soft sequence"):
        trainer.training_step([example], 0, False)


def test_a_step_without_construct_model_is_refused():
    from esmfold2_atomworks.training.trainer import GradientsUnavailableError

    trainer = _trainer()
    trainer.initialize_or_update_trainer_state({"model": torch.nn.Linear(1, 1)})
    with pytest.raises(GradientsUnavailableError, match="never checked"):
        trainer.training_step([{"feats": {}}], 0, False)


def test_a_step_runs_the_model_the_loss_and_the_backward():
    trainer = _built()
    net = trainer.wrapper.net
    # The stand-in "model" takes the batched features as keywords.
    net.forward = lambda **inputs: net.weight.sum() * inputs["res_type_soft"].sum()
    example = {"feats": {"res_type_soft": torch.ones(2, 3)}, "example_id": "e"}

    trainer.training_step([example], 0, False)

    assert net.weight.grad is not None and net.weight.grad.abs().sum() > 0
    assert trainer._current_train_return["example_id"] == "e"
    assert torch.isfinite(trainer._current_train_return["loss"])


def test_an_objective_must_be_supplied():
    from esmfold2_atomworks.training.trainer import make_trainer_class

    trainer = make_trainer_class()(accelerator="cpu", devices_per_node=1)
    with pytest.raises(NotImplementedError, match="Supply a loss"):
        trainer.compute_loss({}, {})
