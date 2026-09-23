# esmfold2-foundry

**ESMFold2, made AtomWorks-compatible and Foundry-trainable.**

Not a rewrite of ESMFold2 in Foundry idiom. The native model is held intact and
made reachable from AtomWorks data — then proved to be the same model.

```
AtomWorks AtomArray ──adapter──> ESMFold2 (unmodified) ──adapter──> AtomWorks AtomArray
                         │
                         └── feature parity, exact, on CPU, no weights
```

## Status

| phase | state |
|---|---|
| **1 — AtomWorks ↔ ESMFold2 adapter** | done; feature parity exact on 5 fixtures |
| **2 — ESMFold2 as a Foundry model** | wrapper, pipeline, inference engine, configs, CLI |
| **3 — generative surgery** | seams exposed; blocked upstream on gradients ([docs/05](docs/05_ROADMAP.md)) |

The Phase 1 milestone is met: for monomer, multimer, metal, cofactor and
modified-residue systems, the input the adapter derives from a structure
featurizes to **the same 29 tensors** as the input a user writes by hand. Since
`prepare_esmfold2_input` and `forward` are both pure functions of their inputs,
identical features mean identical predictions by construction.

Confirmed on a GPU: across real folds the two paths differ no more than the
model differs from *itself* between two runs of the same input — which is the
only meaningful statement available about a diffusion sampler, and is measured
rather than assumed ([docs/02](docs/02_PARITY.md)).

## Quickstart

```bash
source env.sh
esmfold2-foundry doctor              # trees, imports, weights + a real parity check
pytest -q                            # ~50 s, CPU, no GPU and no weights needed

esmfold2-foundry parity structures/*.cif          # survey a corpus
esmfold2-foundry fold input.cif --out-dir runs/   # needs a GPU
```

```python
from atomworks.io import parse
from esmfold2_foundry.model.esmfold2 import FoundryESMFold2

parsed = parse("2hhb.cif.gz")
model = FoundryESMFold2()
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
- every chain that does not reach the model is **named** in the report.

Each of these has a decision ID in [docs/00_SCOPE.md](docs/00_SCOPE.md) and a
test.

## Layout

```
configs/                Hydra configs, shaped like models/rfd3/configs
src/esmfold2_foundry/
  data/                 the adapter, the reverse adapter, the AtomWorks pipeline
  model/                FoundryESMFold2 — holds the native module
  inference/            engine, BaseInferenceEngine-shaped
  training/             FabricTrainer subclass
  parity/               feature and output comparison
  metrics.py  paths.py  doctor.py  cli.py
tests/                  parity and adapter tests; CPU-only by default
docs/                   the design record — read docs/README.md first
```

It mirrors `foundry/models/<name>/` so it can be moved there unchanged; see
[docs/03](docs/03_FOUNDRY_INTEGRATION.md) for the five registrations that needs.

## Requirements

Python 3.14, and the `esm`, `atomworks` and `foundry` source trees beside this
repo (discovered automatically — see [docs/04](docs/04_ENVIRONMENT.md)).

Where the ESMFold2 `nn.Module` comes from depends on the `esm` version — a
`transformers` fork for esm ≤ 3.3, and `esm` itself from 3.4. Both are
supported; `doctor` reports which one is in use.

## License

MIT — see [LICENSE](LICENSE). This project wraps, but does not vendor,
[ESM](https://github.com/Biohub/esm) (MIT) and
[AtomWorks](https://github.com/RosettaCommons/atomworks) /
[Foundry](https://github.com/RosettaCommons/foundry) (BSD-3-Clause), each of
which carries its own license and must be obtained separately.
