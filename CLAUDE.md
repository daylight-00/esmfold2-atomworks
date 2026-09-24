# Notes for contributors (human or agent)

Read [`docs/`](docs/README.md) first. The design record there is authoritative
and decisions carry IDs (`D-001` … `D-008`) that the code cites; this file only
points at the things most easily got wrong.

## Setup

```bash
source env.sh                  # discovers DESIGN_ROOT, sets PYTHONPATH, activates a venv
esmfold2-foundry doctor        # trees, imports, weights, and a real parity check
pytest -q                      # ~30 s, CPU only, no weights needed
ruff check src tests && ruff format --check src tests
```

## The four rules most easily broken

- **Convert at `StructurePredictionInput`, never at the tensors** (`D-001`).
  AtomWorks and ESMFold2 both featurize AF3-like and their conventions differ.
  Rebuilding one from the other yields a model that runs, reports plausible
  confidence, and is wrong.
- **A residue name is a label, not an identity** (`D-004`). `LIG`/`UNL`/`UNK`
  are refused. There is no residue-name→CCD fallback, by design.
- **A metric the model did not produce stays absent** (`D-006`). Never `0.0` —
  for pLDDT that reads as a catastrophic fold, for PAE as a perfect one.
- **Nothing degrades silently** (`D-005`). A new way for the adapter to drop or
  approximate part of the input goes into `atomworks_to_esm.DEGRADATIONS`,
  raises by default, and is accepted only by name; `tests/test_strictness.py`
  then enforces the rest. Recording it in `AdapterReport` alone is not enough —
  `fold_atom_array` returns no report. A declaration that would not reach the
  chain it names (`sequences`/`msas`/`ligands`) raises `ChainDeclarationError`,
  with no opt-in.

## When changing the adapter

Keep `test_declaring_a_modification_actually_changes_the_features` passing. It
asserts that declaring an `MSE` changes the token count, which is what stops the
parity test beside it from being vacuous.

Parity is checked at the **feature** level first (`D-008`): exact, CPU, no
weights, and it names the offending tensor. Do not replace it with an
output-level check — the structure head is a diffusion sampler, so a single
paired run cannot separate an implementation difference from sampling scatter.

## Upstream facts that cost time to rediscover

All are recorded with references in [docs/03](docs/03_FOUNDRY_INTEGRATION.md)
and [docs/04](docs/04_ENVIRONMENT.md). The short list:

- The ESMFold2 module is in a `transformers` fork for esm ≤ 3.3 and in `esm`
  itself for ≥ 3.4, under a different class name. Import from the **leaf**
  modules (`.types`, `.processor`, `.prepare_input`, `.conformers`), which are
  stable across both and avoid loading the model stack to featurize.
- Release `forward` is `@torch.inference_mode()` — it cannot be trained through.
- `fold()` silently discards `noise_scale` / `step_scale` /
  `max_inference_sigma`; `early_exit` is deprecated and ignored.
- 6 of the 29 feature tensors are discarded by `forward`. `FeatureDiff.ok`
  accounts for that; `.identical` is the stricter claim.
- Pocket conditioning is declared in the input schema but never read.
- The shipped checkpoint config differs from the dataclass defaults. Read
  dimensions off `model.config`.
- pLDDT is 0–1, and `result.plddt` and `result.complex.plddt` have different
  lengths whenever a ligand or modified residue is present.
- Importing `atomworks` monkey-patches biotite globally, and the CCD dictionary
  is process-global, not thread-safe, and costs ~9 s.
