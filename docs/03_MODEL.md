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
`model.config`, never off the dataclass defaults in `EsmFold2Config`** — they
disagree substantially, and checkpoints disagree with each other: the reference
checkpoint has `num_loops = 3` where the one `biohub/ESMFold2` publishes today
has 20 (see [04](04_ENVIRONMENT.md) for the two layouts).

`AtomWorksESMFold2.representation_dims()` reads the widths off the live config.
`EsmFold2Config` renames fields on load — a `config.json` written with `d_pair`
or `structure_head.diffusion_module.c_token` comes back with
`pairwise_hidden_size` and `token_hidden_size`, the old names gone — so the
helper reads the current names, falls back to the old ones, and raises on a
width it cannot find rather than reporting 0.

For the same reason `FoldingConfig.num_loops` defaults to `None`, which leaves
the count to the loaded checkpoint; set it to pin a schedule.

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

## Sampler knobs reach the model only if its `forward` declares them

`ESMFold2InputBuilder.fold()` accepts `noise_scale`, `step_scale` and
`max_inference_sigma` and forwards each one that is set. esm >= 3.4 declares
all three on `forward` and hands them to the structure head's sampler; the
`transformers` fork of esm <= 3.3 did not, and dropped them. So
`AtomWorksESMFold2.fold` reads the loaded module's own `forward` signature
and warns about any knob it would drop, rather than assuming either
packaging. The three are typed fields of `FoldingConfig` and keys of the
engine's Hydra config (`configs/inference_engine/base.yaml`), `null` by
default. Left unset, the two scales and the step count come from the
structure head's config and the sigma cap from `forward`'s default -- which
the call record states as `esmfold2.effective.*` (below). `early_exit` is
deprecated upstream and ignored by `fold` with a warning of its own.

## The LM prior is part of the model

`forward` declares `lm_hidden_states`; a caller that supplies it skips the
ESMC pass. `ESMFold2InputBuilder.fold()` does not take it, so
`AtomWorksESMFold2.fold` and `fold_atom_array` carry it themselves
(`lm_hidden_states=`), through a replica of upstream's `fold` body that adds
only that argument. The replica reads upstream's defaults off its signature
and pins its parameter list (`REPLICATED_FOLD_PARAMETERS`): a parameter
upstream adds makes it raise rather than be dropped.
`compute_lm_hidden_states(features)` returns the states the resident backbone
would compute, through the native module's own method, which also restores an
offloaded backbone and applies the FP8 path; the module-level
`esm.models.esmfold2.layers.compute_lm_hidden_states` does neither.

Supplied states are checked against the live module and the prepared
features -- batch and token axes against `res_type`, layer count and width
against the LM shim's own parameters, and the device -- and enter the shim
detached, as upstream detaches them: no gradient reaches them. `lm_mask_pct`
acts inside the backbone the states bypass, so passing both raises; mask when
computing the states instead. `lm_dropout` acts after the shim and applies
either way.

**A fold without the LM prior is refused.** With no backbone attached and no
states supplied, the native `forward` leaves the LM pathway out and folds
anyway. That is not the published model on a smaller input, even for a
protein-free one: upstream hands the shim zero states for every non-protein
token, and the trained shim maps zeros to a non-zero pair term, which leaving
the pathway out removes. So `fold` raises `MissingLanguageModelError` unless a
backbone is resident (`.esmc`, whatever `load_esmc` said -- a bundled
checkpoint carries one regardless) or states are given. The raw `.net` keeps
upstream's flexibility for whoever means to ablate the prior.

`lm_source` is part of the call's record, below; `provenance()["esmfold2.esmc"]`
stays the backbone that was loaded.

## Provenance: the model's, and each call's

Two halves, kept apart because they change at different rates.

`provenance()` is the model's, one string per fact: the weights by path and,
where the directory says, by Hub repo and revision
(`esmfold2.checkpoint.repo`/`.revision`, read by `paths.checkpoint_identity`
through the workspace pin into the Hugging Face store), with
`esmfold2.checkpoint.versioning` saying which answer applies -- `hub-snapshot`,
`hub-local-dir`, or `unversioned` for a directory whose name is all there is
(its repo and revision stay empty rather than read off the name). Weights
named by Hub id are resolved to their snapshot in the Hugging Face store
before loading (upstream's `resolve_model_dir`, the step `from_pretrained`
takes anyway), so the revision that ran is known;
`esmfold2.weights` keeps what was asked for and `esmfold2.weights_resolved`
the directory read. A separate ESMC backbone without a mirror is resolved the
same way. The numerics the wrapper chose at construction are recorded too:
`esmfold2.esmc_precision` (bf16 or fp8 changes the LM states),
`esmfold2.chunk_size` and `esmfold2.kernel_backend` (both change the order of
reductions); a change made later directly on `.net` is not tracked. Alongside, and `esmfold2.checkpoint.config_sha256`, the digest
of its `config.json`, which names the architecture and defaults but not the
weights; the same for a separately attached ESMC backbone; the device the
module's parameters are on, read off the module rather than the process's
current device, with its name; torch and its CUDA build; the config type, the
packaging, the backbone's source and the CCD's.

`fold(record=...)` and `fold_atom_array(record=...)` fill a caller's dict with
the call's own entries:

- `esmfold2.fold.<name>`: every argument the fold ran with, upstream's defaults
  included, read off its signature -- so `num_sampling_steps=8` and `100` are
  visibly different calls, and a default upstream changes is the one recorded.
- `esmfold2.effective.<name>`: what the model executed with -- every setting a
  request left to the checkpoint resolved against the live module, as the
  pinned upstream forward resolves it: loop and sample counts from the config;
  step count, noise and step scale from the structure head; the sigma cap from
  `forward`'s default; MSA depth and column-mask rate from the MSA encoder's
  config; the mask fraction from the config when the backbone runs (`None`
  with supplied states); LM dropout from the config when the call sets none.
  `esmfold2.fold.*` is the request, this is the execution.
- `esmfold2.lm_source`: `model` or `caller-supplied`; with supplied states,
  `esmfold2.lm_states` names them by a full SHA-256 of their bytes, shape and
  dtype, hashed in chunks before the fold, so two different states cannot
  share a record.
- `esmfold2.inputs`: every entity folded, read after upstream's
  `clean_esmfold2_input` (a chainbreak is recorded as the entities it becomes)
  -- ids, kind, length, `chemistry_sha256` over the sequence and modifications
  or the CCD codes or SMILES, not over the chain ids, and `msa_sha256` over the
  alignment's headers, sequences and deletion matrix, or `None`. Where a
  sequence came from is one question; which sequence and which alignment ran
  is another, and a caller that always overrides the sequence learns nothing
  from the first.
- `esmfold2.covalent_bonds` in canonical order, and
  `esmfold2.pocket_sha256` / `esmfold2.distogram_conditioning_sha256`, `None`
  when the condition is absent. Every digest is a full SHA-256.
- `esmfold2.sequence_source` (from `fold_atom_array`): where each chain's
  folded sequence came from, as `AdapterReport` records it.
- `esmfold2.deterministic_algorithms`, `esmfold2.cublas_workspace_config`: the
  execution state observed at the call ([02](02_PARITY.md)).

One record per call. With `num_diffusion_samples > 1` it describes every
sample and names none; a sample index belongs to the output. A record that
already holds an `esmfold2.*` key raises, so one reused from an earlier call
cannot mix two calls' entries. The engine writes both halves into each
output's JSON, with the sample index when there are several, so a prediction
can be read without the run that made it.

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
- **Alignments come from AtomWorks' own loader.** `msa_loader=LoadPolymerMSAs(...)`
  runs it after `pre_transforms` and hands what it finds to the adapter
  ([01](01_ADAPTER.md#declarations-must-bind)). A crop shortens the folded
  sequence, and a full-chain alignment then no longer binds: it is refused, not
  clamped.

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
