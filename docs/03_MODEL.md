# 03 — The model, the pipeline and the engine

What sits between the adapter and a prediction: the wrapper that holds the
published module, the AtomWorks pipeline that feeds it examples, the engine
that keeps it resident for inference, and how to read what it returns. None of
it needs Foundry.

## The model wrapper holds the native module

```python
class AtomWorksESMFold2:
    self.net = ESMFold2Model.from_pretrained(...)   # unmodified
```

The architecture stays exactly as published, so the weights keep meaning what
they meant. Whatever trains or serves it — the optional Foundry integration
([06](06_FOUNDRY_INTEGRATION.md)) or anything else — works with `.net` rather
than with a reimplementation.

`AtomWorksESMFold2` is deliberately **not** an `nn.Module` yet: until something adds
parameters of its own, wrapping would introduce a parameter namespace every
checkpoint has to agree about, for nothing. `.net` is the module a trainer
registers.

There is no `model.net` sub-config of layer widths, because ESMFold2's
architecture comes from the checkpoint's `config.json`. **Read dimensions off
`model.config`, never off the dataclass defaults in
`configuration_esmfold2.py`** — they disagree substantially. The shipped
`biohub/ESMFold2` has `folding_trunk.n_layers = 48` (dataclass says 24),
`num_loops = 3` (says 20), `structure_head.distogram_bins = 64` (says 128).
`AtomWorksESMFold2.representation_dims()` reads the live config.

## Gradients depend on the inputs, not only the checkpoint

The release `forward` is decorated `@torch.inference_mode()`. Its outputs are
inference tensors — they carry no autograd history and cannot be given any, so
a loss computed from them has nothing to differentiate.

The experimental model can produce gradients, but gates them on an **input**:

```python
torch.set_grad_enabled(res_type_soft is not None)   # experimental.py
```

Loading the experimental checkpoint is therefore **necessary and not
sufficient** — fed an ordinary integer `res_type` it still runs with autograd
off. `AtomWorksESMFold2.supports_soft_sequence_design` reports the first half,
`will_produce_gradients(inputs)` reports both, and
`explain_gradient_status(inputs)` names whichever is missing.
`ESMFold2Trainer` checks the real condition against the assembled inputs on
every step, not just the checkpoint flavour once at construction.

## Sampler knobs that do nothing

`ESMFold2InputBuilder.fold()` accepts `noise_scale`, `step_scale`,
`max_inference_sigma` and `early_exit`, forwards them into `forward(**kwargs)`,
and the release `forward` does not declare them — so they are discarded. The
sampler is called with hardcoded defaults. Only `lm_mask_pct` is a real
parameter. (`early_exit` *is* honoured by the experimental model.)

They are therefore **not** exposed as config keys; `AtomWorksESMFold2.fold` warns
if they are passed. Set them on `config.structure_head` instead.

## Pipeline

`build_esmfold2_pipeline` returns an ordinary `atomworks.ml.transforms.Compose`,
so it composes with AtomWorks' own crop, filter and MSA transforms — the ones
Foundry's models are built from too. Two properties differ from the AF3-style
pipelines of `rf3` and `rfd3`:

- **The featurizer goes last and is the only ESMFold2-specific transform.**
  ESMFold2 builds its own tensors from a `StructurePredictionInput` and does not
  consume AtomWorks' `feats` dict (D-001), so `AggregateFeaturesLikeAF3` and
  friends are not in this pipeline — they would compute a featurization nothing
  reads.
- **Cropping changes the molecule.** ESMFold2 folds sequences, so a crop applied
  before the featurizer folds the crop. That is usually right for training and
  usually wrong for evaluation.

The pipeline emits unbatched CPU tensors, one example at a time; whatever
trains on them adds the batch dimension, exactly as
`ESMFold2InputBuilder.prepare_input` does.

## Inference engine

`ESMFold2InferenceEngine` keeps one model resident and folds AtomWorks structures
with it. Its surface mirrors Foundry's `BaseInferenceEngine` (`initialize` /
`run` / `__call__` / context manager), so call sites read the same, but it
neither subclasses nor imports it.
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
  have different lengths whenever a ligand or modified residue is present — a
  112-residue protein with a 19-atom ligand gives 131 tokens and 113 residues.
  Do not index one with the other. `metrics.plddt_per_token(result)` and
  `metrics.plddt_per_residue(result)` name the two, and the residue axis is the
  axis of the residues `result_to_atom_array` returns.
- `esm.mean_plddt` is the mean of `result.plddt`, so it is a **token-space**
  mean: an atomized ligand counts once per atom, not once as a residue.
  `metrics.SCALAR_METRIC_SOURCES` is the read-only table of the scalar metrics
  and the fields they are read from.
- `pair_chains_iptm` is **asymmetric**. There is no single "the" pair ipTM;
  `metrics.py` emits the min and mean over off-diagonal entries and names them
  for what they are.
- Which entity is the ligand is read from `complex.metadata.entity_lookup`, not
  from entity order. Order works today only because the adapter emits polymers
  first — a property of the input assembly, not of the result.
