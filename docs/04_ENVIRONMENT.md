# 04 — Environment

## Installing

In a clone, `pip install -e .` installs the package and resolves `esm` and
`atomworks` like any other dependency; `pip install -e ".[foundry]"` adds the
Foundry integration. What a resolver picks follows the upstreams' own pins —
at the time of writing esm 3.4.1.post1, atomworks 2.2.1, torch 2.11 and
biotite 1.4.0. Two of those pins constrain the interpreter:

- atomworks pins `biotite==1.4.0`, which has wheels for CPython 3.11–3.13 only;
  on 3.14 it builds from source.
- rc-foundry supports Python 3.12 only, and so does the `foundry` extra.

`.python-version` therefore selects 3.12 for `uv sync`. Installed that way —
CPython 3.12, the PyPI releases of all three upstreams, torch 2.11 on CPU — the
suite passes, feature parity and the Foundry contract tests included, given
AtomWorks' test structures: they come with an atomworks checkout rather than
with the package, and without them the tests that read them skip.

The published parity results were produced in a different, pinned environment:
Python 3.14 and torch 2.14, with the upstreams as source trees at the revisions
in `reproducibility/UPSTREAM.lock`. It is defined, and its choices explained, under
[`reproducibility/`](../reproducibility/README.md).

## The ESMFold2 module

esm ≥ 3.4 ships it: `esm.models.esmfold2.model.EsmFold2Model`, loaded with
`from_pretrained(..., device=...)`. `doctor` checks that it is importable,
because the input pipeline imports perfectly well without any model module, so
a missing one would otherwise surface late.

Two things about esm 3.4 worth knowing:

- `esm.models.esmfold2.__init__` imports the whole model stack eagerly,
  which pulls in `huggingface_hub`, `safetensors` and `accelerate`. This repo
  imports from the **leaf modules** (`.types`, `.processor`, `.prepare_input`,
  `.conformers`) instead, which keeps the CPU-only parity path from loading
  the model stack at all.
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
`reproducibility/env.sh` also exports `ESMCFOLD_CCD_PATH`, which esm consults on
every load. `doctor` reports which source applies.

**Historically**, esm ≤ 3.3 shipped only the input pipeline, and the module
lived in a fork of `transformers` (`ESMFold2Model`, moved with `.to(device)`
after loading). `load_native_model_class()` still recognises that layout, and
parity was verified against it while it was published; it no longer is.
ESMFold2 has since also moved into Hugging Face `transformers` ≥ 5.16, which
cannot share an environment with esm 3.4 (esm requires `transformers<5`).
`AtomWorksESMFold2.provenance()` records which packaging produced a result.

### Two checkpoint layouts, and where the ESMC backbone comes from

A Hub repository is mutable, and `biohub/ESMFold2` has been re-published in a
different layout, so "the checkpoint" needs saying which:

| layout | backbone | `biohub/` repos, as of writing |
|---|---|---|
| **separate** — one trunk file, ~1 GB | named in `config.esmc_id`, stored elsewhere | `ESMFold2` before Hub revision `4dc54ad`; `ESMFold2-Experimental` |
| **bundled** — sharded, ~27 GB | carried in the checkpoint (`config.esmc_config`) | `ESMFold2` and `ESMFold2-Fast` from then on |

The parity results in [02](02_PARITY.md) were produced with the separate
`biohub/ESMFold2` at Hub revision `e1e189d0` — the **reference checkpoint**; a
fresh download today gets the bundled one. The two configs also differ in
defaults — `num_loops` is 3 in the reference checkpoint and 20 in the current
one — which is why no schedule is hardcoded here ([03](03_MODEL.md)).

With a separate backbone, `esmc_id` is only a name, and what it holds depends
on where the mirror came from: a Hub id (`biohub/ESMC-6B`), or an absolute path
on the machine that wrote the `config.json`. Left to `from_pretrained`, an
absolute path breaks when the tree moves, and a Hub id downloads 25 GB again
although a mirror sits beside the checkpoint. `AtomWorksESMFold2` therefore
loads the trunk with `load_esmc=False` and attaches the backbone itself
(`attach_esmc`), resolved by `paths.resolve_esmc` by identity: a Hub id the
workspace mirrors (`paths.ESMC_MIRRORS`: `biohub/ESMC-6B` → `$EF_MODELS/ESMC-6B`)
goes to its mirror; any other Hub id is passed on unchanged, so a mirror never
stands in for a backbone of another namespace; an existing directory is used
as is; and an absolute path that no longer exists — which records a directory
on another machine, not an identity — is recovered only when its last
component names a known mirror. A bundled checkpoint is left exactly as
upstream loads it. `provenance()["esmfold2.esmc"]` records which applied, and `doctor`
reports it for every local mirror.

`load_esmc=False` leaves a separate-layout checkpoint without a backbone, so
several folding models can share one instance, or a caller can supply
`lm_hidden_states` itself (see [03](03_MODEL.md)). It cannot remove a bundled
backbone, which arrives with the trunk.

## Provenance: `UPSTREAM.lock`

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

`doctor` fails only on a missing import or a failed parity check. Source trees,
`UPSTREAM.lock` and AtomWorks' test structures belong to the reference
environment; an ordinary install has none of them and is reported as such. The
parity check and most tests read structures from AtomWorks' test data
(`atomworks/tests/data/io`), which comes with an atomworks source checkout
beside this repository; without one they skip rather than fail.

CI (`.github/workflows/ci.yml`) runs only what needs nothing but this
repository: ruff; the `offline`-marked tests — the gold-fixture invariants and
the strictness and declaration checks; a wheel build whose packaged configs
must compose; and a well-formedness check on `UPSTREAM.lock`. The core suite,
feature parity included, needs the `esm` and `atomworks` trees plus torch and a
CCD download — only the Foundry contract tests also need `foundry` — which is
minutes of setup for a check that is run locally and recorded in
[02_PARITY.md](02_PARITY.md); the `offline` subset runs in well under a second.

Everything in the core — the adapter, the pipeline, feature parity,
`doctor` — runs on **CPU in well under a minute** and needs no weights. That is
a deliberate property, not a coincidence: see [02_PARITY.md](02_PARITY.md).

## Gotchas worth knowing

- **Importing `atomworks` monkey-patches biotite globally**, for every consumer
  in the process.
- **`atomworks.io.parse` hydrogenates, and added hydrogens can carry NaN
  coordinates.** This adapter is unaffected — ESMFold2 takes sequences and CCD
  codes, not input coordinates — but anything computing an RMSD downstream is.
- **`chain_info` has no `processed_entity_canonical_sequence` for non-polymer
  chains.** Read it only for polymers; this repo does.
- **Optional accelerators compile from source.** `flash-attn` and
  `transformer-engine` are not dependencies; ESMFold2 runs without them via
  torch SDPA. If you install them, cap the build (`MAX_JOBS=8`) — an uncapped
  parallel compile will exhaust a shared machine.

## GPU

Only folding and output parity need a GPU; `scripts/parity_gpu.sbatch` is a
Slurm template for it (set the partition and resource flags for your site). It
runs in the reference environment.
Everything else in this repository runs on CPU.
