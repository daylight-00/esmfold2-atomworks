# 03 — Foundry integration

## The repo is shaped like `foundry/models/<name>/`

```
esmfold2-foundry/              foundry/models/esmfold2/
├── configs/          <──────> ├── configs/
├── src/esmfold2_foundry/ <──> ├── src/esmfold2/
├── tests/            <──────> ├── tests/
└── docs/             <──────> └── docs/
```

Dropping it in is a move plus a handful of registrations. It is kept as its own
repo because Phase 1 has value without Foundry at all — the adapter is useful to
anything that has an `AtomArray` and wants ESMFold2 — and because vendoring it
into a fast-moving upstream would couple this project's history to theirs.

## What the CONTRIBUTING recipe says, and what the repo actually does

`foundry/CONTRIBUTING.md` says to create `models/<name>/pyproject.toml` and
install the model separately. **No model in the repo does this.** There is
exactly one `pyproject.toml` in Foundry; `rf3`, `rfd3`, `rfd3na` and `mpnn` have
none. Follow the code, not the doc.

Real registration is six edits to the root `pyproject.toml` plus one to a
registry:

```toml
[project.optional-dependencies]
esmfold2 = ["transformers @ git+https://github.com/Biohub/transformers.git@main"]
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

## The model wrapper holds the native module

```python
class FoundryESMFold2:
    self.net = ESMFold2Model.from_pretrained(...)   # unmodified
```

Foundry contributes dataset, trainer, config, distributed execution, logging and
checkpointing. The architecture stays exactly as published, so the weights keep
meaning what they meant.

`FoundryESMFold2` is deliberately **not** an `nn.Module` yet: until Phase 3 adds
parameters of its own, wrapping would introduce a parameter namespace every
checkpoint has to agree about, for nothing. `.net` is the module a trainer
registers.

There is no `model.net` sub-config of layer widths, because ESMFold2's
architecture comes from the checkpoint's `config.json`. **Read dimensions off
`model.config`, never off the dataclass defaults in
`configuration_esmfold2.py`** — they disagree substantially. The shipped
`biohub/ESMFold2` has `folding_trunk.n_layers = 48` (dataclass says 24),
`num_loops = 3` (says 20), `structure_head.distogram_bins = 64` (says 128).
`FoundryESMFold2.representation_dims()` reads the live config.

## Gradients: the release model cannot be trained

`ESMFold2Model.forward` is decorated `@torch.inference_mode()`. Its outputs are
inference tensors — they carry no autograd history and cannot be given any. A
loss computed from them has nothing to differentiate.

`ESMFold2ExperimentalModel.forward` carries no such decorator, and additionally
accepts `res_type_soft` for soft-sequence design. It is the starting point for
Phase 3.

`ESMFold2Trainer.construct_model` raises `GradientsUnavailableError` on a release
model rather than letting a training run produce a flat loss curve whose cause
has to be guessed at.

## Sampler knobs that do nothing

`ESMFold2InputBuilder.fold()` accepts `noise_scale`, `step_scale`,
`max_inference_sigma` and `early_exit`, forwards them into `forward(**kwargs)`,
and the release `forward` does not declare them — so they are discarded. The
sampler is called with hardcoded defaults. Only `lm_mask_pct` is a real
parameter. (`early_exit` *is* honoured by the experimental model.)

They are therefore **not** exposed as config keys; `FoundryESMFold2.fold` warns
if they are passed. Set them on `config.structure_head` instead.

## Pipeline

`build_esmfold2_pipeline` returns an ordinary `atomworks.ml.transforms.Compose`,
so it composes with the crop, filter and MSA transforms the other Foundry models
use. Two properties differ from `rf3`/`rfd3`:

- **The featurizer goes last and is the only ESMFold2-specific transform.**
  ESMFold2 builds its own tensors from a `StructurePredictionInput` and does not
  consume AtomWorks' `feats` dict (D-001), so `AggregateFeaturesLikeAF3` and
  friends are not in this pipeline — they would compute a featurization nothing
  reads.
- **Cropping changes the molecule.** ESMFold2 folds sequences, so a crop applied
  before the featurizer folds the crop. That is usually right for training and
  usually wrong for evaluation.

Foundry's loaders use `collate_fn=lambda x: x` with `batch_size: 1`, so a batch
is a one-element list and the trainer unwraps it with
`batch[0] if not isinstance(batch, dict) else batch`. The pipeline emits
unbatched CPU tensors; `ESMFold2Trainer._assemble_network_inputs` adds the batch
dimension, exactly as `ESMFold2InputBuilder.prepare_input` does.

## Inference engine

`ESMFold2InferenceEngine` mirrors `BaseInferenceEngine`'s surface
(`initialize` / `run` / `__call__` / context manager) but does not subclass it.
`BaseInferenceEngine.__init__` resolves a checkpoint against Foundry's registry,
loads a `.pt` carrying the training `cfg`, and builds its pipeline from
`cfg.datasets.val`'s first dataset. ESMFold2 has none of that: HF safetensors
plus `config.json`, a second repo for the ESMC backbone, and a featurizer that
takes no config. Inheriting would mean overriding every checkpoint-touching
method and leaving the parent half-initialised.

## Reading confidence

- **pLDDT is on 0–1**, not 0–100. Scale at display time.
- `result.plddt` is in model-token space; `result.complex.plddt` is in collapsed
  residue space. A ligand chain collapses to **one** output residue, so the two
  have different lengths whenever a ligand or modified residue is present. Do
  not index one with the other.
- `pair_chains_iptm` is **asymmetric**. There is no single "the" pair ipTM;
  `metrics.py` emits the min and mean over off-diagonal entries and names them
  for what they are.
- Which entity is the ligand is read from `complex.metadata.entity_lookup`, not
  from entity order. Order works today only because the adapter emits polymers
  first — a property of the input assembly, not of the result.
