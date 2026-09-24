"""What it takes to get a gradient out of ESMFold2, asserted rather than assumed.

The contract is easy to state and easy to get wrong: the release forward is
``@torch.inference_mode()`` and can never produce gradients, and the
experimental forward gates them on an **input** --

    torch.set_grad_enabled(res_type_soft is not None)   # experimental.py

-- so loading the experimental checkpoint is necessary and not sufficient. An
earlier version of this package exposed ``supports_gradients`` that returned
``True`` for any experimental checkpoint, which reads as "you can train now" and
is not what upstream does. Training against it would produce a loss with no
``grad_fn`` and a flat curve.

The logic tests below need no weights and run everywhere. The integration test
proves the contract end to end and needs a GPU and the experimental checkpoint.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from esmfold2_atomworks.model.esmfold2 import GRADIENT_GATE, AtomWorksESMFold2


def _stub(config_type: str) -> SimpleNamespace:
    """A stand-in carrying only what the contract methods read."""
    return SimpleNamespace(
        config=SimpleNamespace(type=config_type),
        supports_soft_sequence_design=(config_type == "experimental"),
    )


@pytest.mark.offline
def test_the_gate_is_named_after_the_upstream_condition():
    assert GRADIENT_GATE == "res_type_soft"


@pytest.mark.offline
def test_release_checkpoint_never_produces_gradients():
    stub = _stub("release")
    for inputs in ({}, {GRADIENT_GATE: object()}):
        assert not AtomWorksESMFold2.will_produce_gradients(stub, inputs)
    assert "inference_mode" in AtomWorksESMFold2.explain_gradient_status(stub, {})


@pytest.mark.offline
def test_experimental_checkpoint_alone_is_not_enough():
    """The precise mistake this contract exists to prevent."""
    stub = _stub("experimental")
    assert stub.supports_soft_sequence_design is True
    assert not AtomWorksESMFold2.will_produce_gradients(stub, {"res_type": object()})

    explanation = AtomWorksESMFold2.explain_gradient_status(
        stub, {"res_type": object()}
    )
    assert GRADIENT_GATE in explanation
    assert "no gradients" in explanation


@pytest.mark.offline
def test_experimental_plus_soft_sequence_is_enough():
    stub = _stub("experimental")
    inputs = {GRADIENT_GATE: object()}
    assert AtomWorksESMFold2.will_produce_gradients(stub, inputs)
    assert (
        AtomWorksESMFold2.explain_gradient_status(stub, inputs) == "gradients available"
    )


@pytest.mark.offline
def test_a_none_valued_gate_does_not_count():
    """``res_type_soft=None`` is how upstream spells "off"."""
    stub = _stub("experimental")
    assert not AtomWorksESMFold2.will_produce_gradients(stub, {GRADIENT_GATE: None})


@pytest.mark.gpu
def test_gradients_actually_reach_the_design_logits(parsed, ccd):
    """End to end: a structural objective must move the sequence logits.

    This is the check that defines where a differentiable design path starts.
    It needs the experimental checkpoint, which is a separate download; point
    ``EF_EXPERIMENTAL_WEIGHTS`` at it (or a Hub id) to run this.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    weights = os.environ.get("EF_EXPERIMENTAL_WEIGHTS")
    if not weights:
        pytest.skip(
            "set EF_EXPERIMENTAL_WEIGHTS to the experimental checkpoint to run "
            "the gradient integration test"
        )

    from esmfold2_atomworks.data.atomworks_to_esm import (
        atom_array_to_structure_prediction_input,
    )

    model = AtomWorksESMFold2(weights)
    assert model.supports_soft_sequence_design, (
        f"{weights} is not an experimental checkpoint "
        f"(config.type={getattr(model.config, 'type', None)!r})"
    )

    atoms, chain_info = parsed("lysozyme")
    spi = atom_array_to_structure_prediction_input(atoms, chain_info=chain_info)
    features, _chain_infos = model.featurize(spi)

    n_tokens = features["res_type"].shape[-1]
    n_res_types = int(features["res_type"].max().item()) + 1
    logits = torch.zeros(
        1, n_tokens, max(n_res_types, 33), device=model.device, requires_grad=True
    )
    features[GRADIENT_GATE] = logits.softmax(dim=-1)

    assert model.will_produce_gradients(features), model.explain_gradient_status(
        features
    )

    output = model.net(**features)
    loss = output["distogram_logits"].float().sum()
    loss.backward()

    assert logits.grad is not None, "no gradient reached the design logits"
    assert torch.isfinite(logits.grad).all(), "gradient contains NaN or inf"
    assert logits.grad.norm() > 0, "gradient is identically zero"
