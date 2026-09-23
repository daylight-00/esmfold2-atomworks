# 04 — Environment

Python **3.14**. `source env.sh` before anything.

## Source trees, not packages

`esm`, `atomworks` and `foundry` are consumed as **source** on `PYTHONPATH`
rather than installed, because their metadata pins `python<3.13`, `torch<2.8`
and `biotite==1.4.0`, which would hold the whole environment back:

| module | expected location |
|---|---|
| `esm` | `$DESIGN_ROOT/esm` |
| `atomworks` | `$DESIGN_ROOT/atomworks/src` |
| `foundry` | `$DESIGN_ROOT/foundry/src` |
| `mpnn` | `$DESIGN_ROOT/foundry/models/mpnn/src` |

`pyproject.toml` therefore declares the **union of those trees' runtime
imports**, not the trees. When a tree grows a new import, add it to the matching
dependency group — do not add the tree itself.

Clone them next to this repo:

```bash
git clone https://github.com/Biohub/esm
git clone https://github.com/RosettaCommons/atomworks
git clone https://github.com/RosettaCommons/foundry
git clone <this repo>
```

## `DESIGN_ROOT` is discovered, never hardcoded

Both `env.sh` and `paths.py` walk **up** from the repo until they find a
directory holding `esm/`, `atomworks/` and `foundry/`. A fixed
`REPO_ROOT.parent` is wrong from a git worktree, which sits several levels
deeper. Set `DESIGN_ROOT` explicitly when the trees live somewhere else.

`EF_VENV` points at an existing virtualenv to activate; otherwise `env.sh` uses
this repo's own `.venv` when present.

## Where the ESMFold2 module lives depends on the `esm` version

This moved, and the two layouts are incompatible:

| `esm` | the `nn.Module` | class name | device |
|---|---|---|---|
| **≥ 3.4** | in `esm` itself, `esm.models.esmfold2.model` | `EsmFold2Model` | `from_pretrained(..., device=...)` |
| **≤ 3.3** | in a **fork of `transformers`** | `ESMFold2Model` | `.to(device)` after loading |

`load_native_model_class()` resolves either, and `FoundryESMFold2.provenance()`
records which one was used — results are only comparable within one of them.
`doctor` reports it too, because the input pipeline imports perfectly well
without any model module, so a missing one surfaces late otherwise.

For esm ≤ 3.3 install the fork with the `esm-legacy-model` group; it conflicts
with stock `transformers`, so use one or the other:

```bash
uv sync --group esm-legacy-model
```

Two knock-on effects of the 3.4 move:

- `esm.models.esmfold2.__init__` now imports the whole model stack eagerly,
  which pulls in `huggingface_hub`, `safetensors` and `accelerate`. This repo
  imports from the **leaf modules** (`.types`, `.processor`, `.prepare_input`,
  `.conformers`) instead — stable in both versions, and it keeps the CPU-only
  parity path from loading the model stack at all.
- esm 3.4 pins `torch>=2.11,<2.12` while this project asks for `torch>=2.12`.
  Resolve that deliberately rather than letting a solver pick.

Weights come from a local mirror if one exists at `$EF_MODELS/ESMFold2`, and
otherwise from the Hub id `biohub/ESMFold2`, downloaded on first use. The ESMC
backbone (`biohub/ESMC-6B`) is a separate repository fetched during
`from_pretrained`; pass `load_esmc=False` to share one instance across several
folding models.

The CCD dictionary (~50k entries, ~9 s) is loaded once per process into
module-global state in `esm.models.esmfold2.conformers`. It is **not
thread-safe**, which is why `FoundryESMFold2.__init__` warms it.

## Pinning: `UPSTREAM.lock`

The trees come from `PYTHONPATH`, so nothing else records *which* revision
produced a given result — and this project's central claim is **exact** parity,
which raises the bar for reproducibility well above an ordinary wrapper's.
`UPSTREAM.lock` records the commit and version of each tree that the published
parity numbers were verified against, and `doctor` compares them:

```
upstream revisions
  [ok  ] esm        43b4548b8676: matches lock (3.4.1.post1)
  [ok  ] atomworks  fcf7af8c127e: matches lock (2.2.1)
  [ok  ] foundry    b02eed6a6bdf: matches lock (0.1.0)
```

Drift is **reported, never enforced**. The adapter is deliberately
version-tolerant — it supports both ESMFold2 packagings — so a newer tree is
something to re-verify and re-pin, not something to refuse. Refusing would also
make `doctor` useless on the day an upstream releases.

## Checks

```bash
uv sync && source env.sh
esmfold2-foundry doctor        # trees, revisions, imports, weights, a real parity check
pytest -q                      # the same, as assertions
```

CI (`.github/workflows/ci.yml`) runs only what needs nothing but this
repository: ruff, the `offline`-marked gold-fixture invariants, and a
well-formedness check on `UPSTREAM.lock`. The full parity suite needs the three
source trees plus torch and a CCD download, which is minutes of setup for a
check that is run locally and recorded in [02_PARITY.md](02_PARITY.md); the
`offline` subset runs in well under a second.

Everything needed for Phase 1 — the adapter, the pipeline, feature parity,
`doctor` — runs on **CPU in well under a minute** and needs no weights. That is
a deliberate property, not a coincidence: see [02_PARITY.md](02_PARITY.md).

## Gotchas worth knowing

- **`biotite` must satisfy both sides.** AtomWorks pins `==1.4.0`, but 1.4 has
  no cp314 wheel, and biotite 1.7 moved `connect_via_residue_names` out of
  `biotite.structure.bonds` — which `atomworks.io.utils.io_utils` imports by
  that exact path. This project therefore pins `>=1.5,<1.7`. Do not widen it
  without checking that import.
- **AtomWorks 2.2.1 removed the upper bounds on its own dependencies**
  (`torch`, `numpy`, `rdkit`, `pandas`, `pyarrow`). A resolve that inherits
  those constraints is now unbounded, so pin them here rather than relying on
  the tree.
- **Importing `atomworks` monkey-patches biotite globally**, for every consumer
  in the process.
- **`atomworks.io.parse` hydrogenates, and added hydrogens can carry NaN
  coordinates.** This adapter is unaffected — ESMFold2 takes sequences and CCD
  codes, not input coordinates — but anything computing an RMSD downstream is.
- **`chain_info` has no `processed_entity_canonical_sequence` for non-polymer
  chains.** Read it only for polymers; this repo does.
- **GPU-only extras compile from source.** `flash` and `te` are excluded from
  `default-groups` precisely so `uv sync` installs wheels only and builds
  nothing. If you do install them, cap the build (`MAX_JOBS=8`) — an uncapped
  parallel compile will exhaust a shared machine.
- **The prebuilt `flash-attn` cp314 wheel is ABI-incompatible with torch 2.14.**
  ESMFold2 runs without it via torch SDPA; it is an optional acceleration path.

## GPU

Only folding and output parity need a GPU; `scripts/parity_gpu.sbatch` is a
Slurm template for it (set the partition and resource flags for your site).
Everything else in this repository runs on CPU.
