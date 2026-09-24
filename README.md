# esmfold2-atomworks

**ESMFold2 on AtomWorks structures, with native ESMFold2 feature semantics
preserved exactly.**

Not a rewrite. The published model is held intact, fed from and returned to
AtomWorks structures — then proved to be the same model. An optional Foundry
integration adds training, configuration and checkpointing on Foundry's
trainer, without changing the native model.

```
AtomWorks AtomArray ──adapter──> ESMFold2 (unmodified) ──adapter──> AtomWorks AtomArray
                         │
                         └── feature parity, exact, on CPU, no weights
```

## Status

| part | state |
|---|---|
| **AtomWorks ↔ ESMFold2** — adapter and its reverse, AtomWorks pipeline, supervision labels | done; feature parity exact on 5 fixtures |
| **Inference** — model wrapper, engine, CLI | done; output parity within the model's own scatter on a GPU |
| **Foundry integration** (optional) — trainer, Hydra configs | contracts verified against the pinned Foundry checkout, the runtime ones also against rc-foundry 0.2.0; the objective is the caller's ([docs/05](docs/05_ROADMAP.md)) |

The core milestone is met: for monomer, multimer, metal, cofactor and
modified-residue systems, the input the adapter derives from a structure
featurizes to **the same 29 tensors** as an independently frozen reference
input. Featurization is a pure function **at a fixed seed**, so that
equality is exact. (Only a SMILES ligand makes the seed matter: its conformer
is embedded at call time.)

The model itself is *not* deterministic — its structure head is a diffusion
sampler — so identical features mean the two paths condition the model
identically, not that they return identical coordinates. Confirmed on a GPU:
the cross-path deviation falls within the model's own run-to-run scatter,
measured rather than assumed ([docs/02](docs/02_PARITY.md)).

## Quickstart

```bash
pip install -e .                 # esm and atomworks resolve like any dependency
pip install -e ".[foundry]"      # optional: training through Foundry (Python 3.12)

esmfold2-atomworks doctor        # imports, weights; parity given AtomWorks' test data

esmfold2-atomworks parity structures/*.cif        # survey a corpus
esmfold2-atomworks fold input.cif --out-dir runs/ # needs a GPU
```

```python
from atomworks.io import parse
from esmfold2_atomworks.model.esmfold2 import AtomWorksESMFold2

parsed = parse("2hhb.cif.gz")
model = AtomWorksESMFold2()
structure, result = model.fold_atom_array(
    parsed["asym_unit"][0], chain_info=parsed["chain_info"]
)
```

## What it refuses to do

The adapter fails loudly on the cases that otherwise produce a confident
prediction of the wrong molecule:

- a ligand labelled `LIG`/`UNL`/`UNK` is **refused**, because those are real CCD
  codes *and* what a model writes on anything it was handed as SMILES;
- a declared ligand whose heavy-atom composition disagrees with the atoms present
  is an **error**;
- a sequence is taken from the full entity record, not from the residues that
  happen to be modelled, so unresolved loops are not silently deleted;
- a non-standard residue is declared by CCD code, not folded as its parent;
- a chain kind, sequence override, MSA or ligand declaration that does not bind
  to exactly the chain it names — a typo, the wrong kind of chain, an alignment
  built for another sequence — is an **error**, not a no-op;
- a chain that holds more than one molecule — a protein, its ligands and its
  waters under one author chain id — is an **error**, whether its annotations
  say so or a declared kind is contradicted by the CCD;
- a chain that cannot be expressed, a covalent bond that cannot be placed, a
  chain kind that would have to be guessed, or a modification whose position is
  unknown each **raises** by default — accepted only by name, on every path,
  and recorded per output when accepted.

Each of these has a decision ID in [docs/00_SCOPE.md](docs/00_SCOPE.md) and a
test; the strictness policy is laid out in
[docs/01](docs/01_ADAPTER.md#strictness-no-silent-semantic-degradation).

## Layout

```
src/esmfold2_atomworks/
  data/                 the adapter, the reverse adapter, the AtomWorks pipeline, labels
  model/                AtomWorksESMFold2 — holds the native module
  inference/            engine: one resident model, many structures
  parity/               feature and output comparison
  metrics.py  paths.py  doctor.py  cli.py
  training/             optional Foundry integration: FabricTrainer subclass
configs/                optional Foundry integration: Hydra configs
tests/                  parity and adapter tests; CPU-only by default
docs/                   the design record — read docs/README.md first
```

Nothing outside `training/` and `configs/` imports Foundry, and
`tests/test_core_boundary.py` holds it to that. The optional integration is
described in [docs/06](docs/06_FOUNDRY_INTEGRATION.md).

## Environments

`pip install -e .` is all the package needs: `esm` and `atomworks` are ordinary
dependencies, resolved from PyPI (Python ≥ 3.12; the `foundry` extra needs 3.12,
as rc-foundry does). The structure-based tests, feature parity among them,
read AtomWorks' test structures, which come with an atomworks checkout rather
than with the package: given those, the suite passes against the PyPI releases
of the upstreams, and without them those tests skip. See
[docs/04](docs/04_ENVIRONMENT.md).

The published parity results were produced in one pinned reference
environment: Python 3.14, torch 2.14, and the upstreams as source trees at the
revisions in `reproducibility/UPSTREAM.lock`. [`reproducibility/`](reproducibility/README.md)
defines it (`uv sync --project reproducibility`, then
`source reproducibility/env.sh`). It is not needed for ordinary use.

## License

MIT — see [LICENSE](LICENSE). This project wraps, but does not vendor,
[ESM](https://github.com/Biohub/esm) (MIT) and
[AtomWorks](https://github.com/RosettaCommons/atomworks) /
[Foundry](https://github.com/RosettaCommons/foundry) (BSD-3-Clause), each of
which carries its own license and must be obtained separately.
