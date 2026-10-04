# 06 — Optional Foundry integration

Foundry is not part of the core AtomWorks ↔ ESMFold2 contract. It provides one
supported training and execution backend: the `FabricTrainer` subclass in
`training/` and the Hydra configs in `configs/`. Nothing else in the package
imports it — `tests/test_core_boundary.py` checks that at import time — and a
workspace without Foundry is complete for everything else. It is the package's
`foundry` extra — `pip install -e ".[foundry]"`, which brings rc-foundry and so
needs Python 3.12 — and a plain install leaves it out.

`tests/test_foundry_integration.py` checks the contracts the integration relies
on, in two kinds. The *runtime* contracts are checked against the Foundry that is
imported: the trainer subclasses `FabricTrainer` with no abstract method left and
matching signatures, the engine offers `BaseInferenceEngine`'s surface,
`RegisteredCheckpoint` takes the fields named below, and the data-pipeline config
instantiates. They pass against both the pinned Foundry checkout and rc-foundry
0.2.0. The *repository* contracts read a Foundry checkout's own files — the six
registration tables exist, and models are registered centrally rather than per
model — so they run only when that checkout is the Foundry under test. No Foundry
file is modified. The other statements on this page about Foundry's repository
layout describe the pinned checkout and are not tested.

## The integration keeps Foundry's model-package conventions

The repository is laid out like `foundry/models/<name>/`, so that moving it into
Foundry needs only the move and a handful of registrations:

```
esmfold2-atomworks/              foundry/models/esmfold2/
├── configs/          <────────> ├── configs/
├── src/esmfold2_atomworks/ <──> ├── src/esmfold2/
├── tests/            <────────> ├── tests/
└── docs/             <────────> └── docs/
```

Its formatting and pytest settings (`[tool.ruff.format]`,
`[tool.pytest.ini_options]`) follow Foundry's. It is its own repository because the core has value without Foundry at
all — anything that has an `AtomArray` can fold it with ESMFold2.

## Registering the package inside Foundry

Foundry registers models centrally, in its root `pyproject.toml` and a checkpoint
registry; none of its models has a `pyproject.toml` of its own. Registration is
six edits to the root `pyproject.toml` plus one to the registry:

```toml
[project.optional-dependencies]
# esm >= 3.4 ships the ESMFold2 module itself. See docs/04_ENVIRONMENT.md.
esmfold2 = ["esm>=3.4.1.post1"]
all = [..., "rc-foundry[esmfold2]"]        # optional: mpnn skips both of these

[project.scripts]
esmfold2 = "esmfold2.cli:app"

[tool.hatch.build.targets.wheel]
packages = [..., "models/esmfold2/src/esmfold2"]

[tool.hatch.build.targets.wheel.force-include]
"models/esmfold2/configs" = "esmfold2/configs"

[tool.mypy]
files = [..., "models/esmfold2/src/esmfold2"]

[tool.pytest.ini_options]
testpaths = [..., "models/esmfold2/tests"]
```

plus an `"esmfold2"` entry in
`src/foundry/inference_engines/checkpoint_registry.py` —
`RegisteredCheckpoint(url=..., filename=..., description=...)` — so
`ckpt_path=esmfold2` resolves, and a docs symlink:

```bash
ln -s ../../../models/esmfold2/docs foundry/docs/source/models/esmfold2
```

Further requirements:

- **mypy.** Foundry type-checks each model package under per-package strict
  overrides (`disallow_untyped_defs`, `check_untyped_defs`); add
  `module = ["esmfold2.*"]` to them.
- **pytest.** `testpaths` lists every model's tests, and `markers` are declared
  centrally under `--strict-markers`. Foundry declares `gpu` and `integration`,
  so any other marker this package uses must be added there.
- **sdist.** `[tool.hatch.build.targets.sdist] exclude` is glob-based
  (`models/*/tests/`) and needs no edit.
- **`configs/__init__.py`** must exist (empty) for `pkg://esmfold2.configs` to
  resolve.
- **Hydra searchpath.** Foundry's models also list `pkg://configs`. Outside
  Foundry that package does not exist, and Hydra warns on every compose for an
  entry it cannot resolve, so `configs/inference.yaml` lists only
  `pkg://esmfold2_atomworks.configs`; moved in, `pkg://configs` goes back beside
  it.
- **`tests/conftest.py`.** Foundry's `rf3` and `mpnn` use
  `foundry.testing.configure_pytest`, plus `collect_ignore` for tests that need a
  GPU or checkpoints.

## Training through Foundry

Foundry's loaders use `collate_fn=lambda x: x` with `batch_size: 1`, so a batch
is a one-element list and the trainer unwraps it with
`batch[0] if not isinstance(batch, dict) else batch`.
`ESMFold2Trainer._assemble_network_inputs` then adds the batch dimension to the
pipeline's unbatched tensors ([03](03_MODEL.md#pipeline)), and checks the
gradient precondition
([03](03_MODEL.md#gradients-depend-on-the-inputs-not-only-the-checkpoint))
against the assembled inputs on every step. What it deliberately lacks is an
objective — see [05](05_ROADMAP.md).

**Configuration.** `train_cfg.model` follows Foundry's layout: `net` is the
wrapper's config (`_target_: esmfold2_atomworks.model.esmfold2.AtomWorksESMFold2`,
`weights: ...`), and `optimizer` and `lr_scheduler` sit beside it, built by
Foundry's own `construct_optimizer` and `construct_scheduler`. `construct_model`
freezes the ESMC backbone — about 6.3 B parameters that the model runs without
autograd and hands the trunk a detached copy of — so the optimizer's parameters
are the trunk, embedders and heads (about 226 M for the experimental
checkpoint).

**What can be trained.** The experimental `forward` turns autograd on when it is
given a soft sequence (`res_type_soft`, `[L, 33]`) and returns `distogram_logits`
with a gradient path through the trunk. The diffusion sampler runs under
`torch.no_grad()`, so a coordinate or diffusion loss reaches nothing: the trainable
objective is on the trunk's distogram, or on the soft sequence. The pipeline's
`attach_labels=True` supplies target coordinates, and the example has to carry
`res_type_soft`.

**A run.** `scripts/smoke_training_step.py` takes steps through Foundry's `fit`
loop on one GPU — the experimental checkpoint built by `construct_model`, one
AtomWorks structure through the pipeline, an AdamW optimizer, and a distogram
cross-entropy against the structure's own coordinates — and checks that each loss
is finite, every step has gradients, and the trunk's parameters change.
[`reproducibility/training_smoke.json`](../reproducibility/training_smoke.json)
records a run; `tests/test_training_gpu.py` repeats it under `pytest -m gpu`. The
objective there is the script's own, and its bin range is its choice, not the
checkpoint's.
