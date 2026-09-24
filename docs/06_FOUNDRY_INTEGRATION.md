# 06 — Optional Foundry integration

Foundry is not part of the core AtomWorks ↔ ESMFold2 contract. It provides one
supported training and execution backend: the `FabricTrainer` subclass in
`training/` and the Hydra configs in `configs/`. Nothing else in the package
imports it — `tests/test_core_boundary.py` holds it to that — and a workspace
without Foundry is complete for everything else. It is the package's `foundry`
extra — `pip install -e ".[foundry]"`, which brings rc-foundry and so needs
Python 3.12 — and a plain install leaves it out. The contract tests pass
against both the pinned Foundry tree and the rc-foundry 0.2.0 release.

> Everything on this page is **verified against Foundry's current integration
> contracts** by
> `tests/test_foundry_integration.py`: the trainer really subclasses
> `FabricTrainer` with no abstract method left and matching signatures, the
> engine really offers `BaseInferenceEngine`'s surface, `RegisteredCheckpoint`
> really takes the fields named below, all six registration tables really
> exist, models really are registered centrally rather than per-model, and the
> data-pipeline config really instantiates into a working pipeline. The tests
> do not modify the Foundry checkout — they check each claim where it lives.
> What they do *not* do is copy the package in, patch Foundry's `pyproject.toml`
> and install the result; that would verify the same contracts while leaving a
> dirty tree, and would need re-doing on every upstream release.

## The integration keeps Foundry's model-package conventions

It is kept compatible with them on purpose, so that an upstream Foundry
integration can stay thin. The repository is laid out like
`foundry/models/<name>/`:

```
esmfold2-atomworks/              foundry/models/esmfold2/
├── configs/          <────────> ├── configs/
├── src/esmfold2_atomworks/ <──> ├── src/esmfold2/
├── tests/            <────────> ├── tests/
└── docs/             <────────> └── docs/
```

Moving it in is a move plus a handful of registrations. It is its own
repository because the core has value without Foundry at all — anything that
has an `AtomArray` can fold it with ESMFold2 — and because vendoring it into a
fast-moving upstream would couple this project's history to theirs.

## What the CONTRIBUTING recipe says, and what the repo actually does

`foundry/CONTRIBUTING.md` says to create `models/<name>/pyproject.toml` and
install the model separately. **No model in the repo does this.** There is
exactly one `pyproject.toml` in Foundry; `rf3`, `rfd3`, `rfd3na` and `mpnn` have
none. Follow the code, not the doc.

Real registration is six edits to the root `pyproject.toml` plus one to a
registry:

```toml
[project.optional-dependencies]
# esm >= 3.4 ships the ESMFold2 module itself. See docs/04_ENVIRONMENT.md.
esmfold2 = ["esm>=3.4"]
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

The last two `pyproject.toml` entries are newer than the rest and easy to miss:

- **mypy has no ignore ratchet any more.** It was driven to zero for all four
  models and the `ignore_errors` block deleted; strictness is now opt-in per
  package via `disallow_untyped_defs`/`check_untyped_defs` overrides. A new
  package is therefore type-checked from the day it is added, so annotate it
  fully and add `module = ["esmfold2.*"]` to the strict overrides rather than
  asking for an exemption.
- **`testpaths` now lists every model's tests**, and `markers` is declared
  centrally under `--strict-markers`. Only `gpu` and `integration` are declared
  upstream, so any additional marker this package uses must be added there too.

`[tool.hatch.build.targets.sdist] exclude` is glob-based
(`models/*/tests/`), so it needs no edit.

`configs/__init__.py` must exist (empty) for `pkg://esmfold2.configs` to
resolve. `rfd3` and `rfd3na` have it; `rf3` omits it despite declaring the
searchpath. Do not copy `rf3` here.

Two conventions have a current and a legacy form — copy the current one:

| | current | legacy |
|---|---|---|
| `tests/conftest.py` | `rf3`, `mpnn` (`foundry.testing.configure_pytest`, plus `collect_ignore` for tests that need a GPU or checkpoints) | `rfd3` (hand-rolled `rootutils.setup_root`) |
| `configs/__init__.py` | `rfd3`, `rfd3na` (present) | `rf3` (absent) |

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
