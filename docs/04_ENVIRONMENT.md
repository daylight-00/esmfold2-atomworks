# 04 — Environment

## Installing

In a clone, `pip install -e .` installs the package and resolves `esm` and
`atomworks` like any other dependency; `pip install -e ".[foundry]"` adds the
Foundry integration. What a resolver picks follows the upstreams' own pins; the
versions the suite has run against are in the table below. Two of those pins
constrain the interpreter:

- atomworks 2.2.1 pins `biotite==1.4.0`, which has wheels for CPython 3.11–3.13
  only; on 3.14 it builds from source.
- rc-foundry supports Python 3.12 only, and so does the `foundry` extra.

`.python-version` therefore selects 3.12 for `uv sync`.

## Verified against

| | ordinary install | reference environment |
|---|---|---|
| CPython | 3.12 | 3.14.6 |
| esm | 3.4.1.post1 | source tree at the `UPSTREAM.lock` revision |
| atomworks | 2.2.1 | source tree at the `UPSTREAM.lock` revision |
| rc-foundry (`foundry` extra) | 0.2.0 | source tree at the `UPSTREAM.lock` revision |
| torch | 2.11, CPU | 2.14.0, CUDA 13.2 |
| biotite | 1.4.0 | 1.6 |

The suite passes in both, feature parity and the Foundry contract tests included.
Four structures (lysozyme, haemoglobin, a zinc site, selenomethionine) are in
`tests/data/structures`. What still comes from an AtomWorks checkout beside this
repository, and skips without one, is the corpus survey, the AF3-derived
`flavoprotein` and `unl` structures and AtomWorks' `1wym` MSA case. Other skips
are the Foundry tests without the `foundry` extra, the Hydra config tests without
`omegaconf`, and the checks of the source-tree environment in an ordinary install.
The GPU tests run only in the reference environment. A version outside this table is
untested, not refused: `doctor` reports how an installed upstream differs from
`UPSTREAM.lock`.

The published parity results were produced in a different, pinned environment:
Python 3.14 and torch 2.14, with the upstreams as source trees at the revisions
in `reproducibility/UPSTREAM.lock`. It is defined, and its choices explained, under
[`reproducibility/`](../reproducibility/README.md).

## The ESMFold2 module

esm ≥ 3.4 ships it: `esm.models.esmfold2.model.EsmFold2Model`, loaded with
`from_pretrained(..., device=...)`. `doctor` checks that it is importable.

Two things about esm 3.4 worth knowing:

- `esm.models.esmfold2.__init__` imports the whole model stack eagerly, which
  pulls in `huggingface_hub`, `safetensors` and `accelerate`, so importing any
  module of that package loads it.
- esm 3.4 pins `torch>=2.11,<2.12`, which is why an ordinary install gets
  torch 2.11. The reference environment runs 2.14 by consuming esm as a source
  tree instead.

Weights come from a local mirror if one exists at `$EF_MODELS/ESMFold2`, and
otherwise from the Hub id `biohub/ESMFold2`, downloaded on first use
(`paths.ESMFOLD2_WEIGHTS`; likewise `-Fast` and `-Experimental`).

The CCD dictionary (~50k entries, ~9 s) is loaded once per process into
module-global state in `esm.models.esmfold2.conformers`. It is **not
thread-safe**, which is why `AtomWorksESMFold2.__init__` warms it. It comes from
the `ccd.pkl` in the `ESMFold2` mirror (`paths.ccd_dir()`), which serves every
checkpoint; without a mirror copy, esm downloads it from the Hub's latest
revision, which can change under a fixed checkpoint. The first load in a
process decides, and esm's own lazy lookups name no location, so
`reproducibility/env.sh` also exports `ESMCFOLD_CCD_PATH` before Python starts:
`esm.models.esmfold2.conformers` captures it at import and prefers it over any
location a caller passes. `doctor` reports which source applies, and
`provenance()["esmfold2.ccd"]` records it for each model -- or records it as
unknown when something loaded the dictionary before the model, since esm keeps
no record of where it came from.

`transformers` ≥ 5 ships its own ESMFold2 and cannot share an environment with
esm 3.4 (esm requires `transformers<5`).

### Two checkpoint layouts, and where the ESMC backbone comes from

A Hub repository is mutable and its revisions differ in layout, so "the
checkpoint" needs a revision:

| layout | backbone | `biohub/` revisions |
|---|---|---|
| **separate** — one trunk file, ~1 GB | named in `config.esmc_id`, stored elsewhere | `ESMFold2` at `e1e189d0`; `ESMFold2-Experimental` |
| **bundled** — sharded, ~27 GB | carried in the checkpoint (`config.esmc_config`) | `ESMFold2` at `69869f73` and `ESMFold2-Fast` |

The output-parity record in [02](02_PARITY.md) was produced with the bundled
`biohub/ESMFold2` at revision `69869f73`, whose weights are identical to those of
the separate-layout **reference checkpoint**, revision `e1e189d0`. The two
configs also differ in defaults — `num_loops` is 3 in `e1e189d0` and 20 in
`69869f73` — which is why no schedule is hardcoded here ([03](03_MODEL.md)).

The reference checkpoint, the ESMC-6B revision it was paired with and the CCD
pickle are pinned by revision and digest in
[`reproducibility/ARTIFACTS.lock`](../reproducibility/ARTIFACTS.lock).
[`scripts/compare_checkpoints.py`](../scripts/compare_checkpoints.py) loads a
checkpoint and that reference through esm's own loader and compares every
tensor. Its record for the bundled `biohub/ESMFold2` (revision `69869f73`),
[`reproducibility/checkpoint_equivalence.json`](../reproducibility/checkpoint_equivalence.json),
finds all 1590 trunk tensors and all 802 tensors of the bundled backbone
identical to the reference. The only exclusions are the three confidence-head
modules upstream allocates and never reads, which the bundled checkpoint no
longer carries. The bundled revision differs in packaging, tensor names and
config defaults, not in weights.

With a separate backbone, `esmc_id` is only a name, and what it holds depends
on where the mirror came from: a Hub id (`biohub/ESMC-6B`), or an absolute path
on the machine that wrote the `config.json`. Left to `from_pretrained`, an
absolute path breaks when the tree moves, and a Hub id downloads 25 GB again
although a mirror sits beside the checkpoint. `AtomWorksESMFold2` therefore
loads the trunk with `load_esmc=False` and attaches the backbone itself
(`attach_esmc`), resolved by `paths.resolve_esmc` by identity: a Hub id that has
a local mirror (`paths.ESMC_MIRRORS`: `biohub/ESMC-6B` → `$EF_MODELS/ESMC-6B`)
goes to its mirror; any other Hub id is passed on unchanged, so a mirror never
stands in for a backbone of another namespace; an existing directory is used
as is; and an absolute path that no longer exists — which records a directory
on another machine, not an identity — is recovered only when its last
component names a known mirror. A bundled checkpoint is left exactly as
upstream loads it. `provenance()["esmfold2.esmc"]` records which applied, and `doctor`
reports it for every local mirror.

`load_esmc=False` leaves a separate-layout checkpoint without a backbone. Such a
model folds only when handed `lm_hidden_states`; without them `fold` refuses rather than folding
without the LM prior (see [03](03_MODEL.md)). It cannot remove a bundled
backbone, which arrives with the trunk.

## Upstream revision lock

`reproducibility/UPSTREAM.lock` records the upstream revisions — commit and
version — that the published parity results were verified against. `doctor` compares what it finds
with it: a source tree's commit in the reference environment, an installed
package's version in an ordinary install. Drift is **reported, never
enforced**: the adapter is deliberately version-tolerant, so a newer upstream is
something to re-verify, not something to refuse. How the lock is used to
rebuild the reference environment is described in
[`reproducibility/`](../reproducibility/README.md).

## Checks

```bash
esmfold2-atomworks doctor    # where the upstreams come from, imports, weights, parity
pytest -q                    # the same, as assertions
```

`pytest` is in the `dev` dependency group, which `uv sync` installs along with
the package; `pip install -e .` does not, so on that path add it yourself.

`doctor` fails only on a missing import or a failed parity check. Source trees
belong to the reference environment; an ordinary install is compared with
`UPSTREAM.lock` by package version instead. The parity check reads this
repository's own `2hhb` (`tests/data/structures`), else AtomWorks' test data
(`atomworks/tests/data/io`) beside it; with neither, a wheel for instance, it is
not run, and `doctor` says so beside its verdict rather than reporting a plain
pass.

CI (`.github/workflows/ci.yml`) runs: ruff; the `offline`-marked tests; a wheel
build whose packaged configs must compose, with a check that the declared
dependencies resolve; and a well-formedness check on `UPSTREAM.lock`. The core
suite, feature parity included, needs esm, atomworks, torch and a CCD
download — only the Foundry contract tests also need `foundry` — and is run
locally; its results are recorded in [02_PARITY.md](02_PARITY.md).

Everything in the core — the adapter, the pipeline, feature parity,
`doctor` — runs on **CPU in well under a minute** and needs no weights. That is
a deliberate property, not a coincidence: see [02_PARITY.md](02_PARITY.md).

## Gotchas worth knowing

- **Importing `atomworks` monkey-patches biotite globally**, for every consumer
  in the process.
- **Hydrogens that AtomWorks 2.x's `parse` adds (`hydrogen_policy="infer"`) can
  carry NaN coordinates.** This adapter is unaffected — ESMFold2 takes sequences and CCD
  codes, not input coordinates — but anything computing an RMSD downstream is.
- **`processed_entity_canonical_sequence` is meaningful for polymers only.**
  Read it only for polymers; this repo does.
- **Optional accelerators compile from source.** `flash-attn` and
  `transformer-engine` are not dependencies; ESMFold2 runs without them via
  torch SDPA. If you install them, cap the build (`MAX_JOBS=8`) — an uncapped
  parallel compile will exhaust a shared machine.

## GPU

Folding and output parity are meant for a GPU (`--device cpu` works, slowly);
`scripts/parity_gpu.sbatch` is a Slurm template for the GPU tests (set the
partition and resource flags for your site). It runs in the reference
environment. Everything else in this repository runs on CPU.

## Environment variables

| variable | meaning | default |
|---|---|---|
| `DESIGN_ROOT` | directory holding the `esm/` and `atomworks/` source trees (reference environment) | discovered by walking up from the repository |
| `EF_MODELS` | directory of local weight mirrors (`ESMFold2/`, `ESMFold2-Fast/`, `ESMFold2-Experimental/`, `ESMC-6B/`) | `$DESIGN_ROOT/biohub` |
| `EF_CHECKPOINTS`, `EF_RUNS` | checkpoint and run output directories | `$DESIGN_ROOT/checkpoints`, `<repo>/runs` |
| `ESMCFOLD_CCD_PATH` | the CCD pickle esm loads; read once at import | `$EF_MODELS/ESMFold2/ccd.pkl` when present, via `reproducibility/env.sh` |
| `EF_VENV` | environment `reproducibility/env.sh` activates | `reproducibility/.venv` |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` for the deterministic output-parity tests; read when CUDA starts | unset |
| `EF_ALLOW_DOWNLOAD`, `EF_EXPERIMENTAL_WEIGHTS` | tests only: allow Hub downloads in the GPU tests; the experimental checkpoint for the gradient test | unset |
