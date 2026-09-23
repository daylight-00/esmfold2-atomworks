# 05 — What is not done yet

This project is scoped to one thing: making the published ESMFold2 reachable
from AtomWorks data and usable as a Foundry model, faithfully. What follows is
what that scope still lacks — not a research plan. Downstream projects that
build on this repository keep their own.

## Done

- **AtomWorks → ESMFold2 adapter**, with exact feature parity against frozen
  reference inputs ([02_PARITY.md](02_PARITY.md)).
- **Output parity on a GPU**, as a controlled comparison against the model's own
  run-to-run scatter.
- **Both ESMFold2 packagings** (esm ≤ 3.3 via the `transformers` fork, esm ≥ 3.4
  in-package), verified on CPU and GPU.
- **Inference**: engine, CLI, configs, AtomWorks round trip.

## Missing for inference completeness

**Corpus coverage.** `esmfold2-foundry parity <dir>/*.cif` over a larger set,
to enumerate the chain types and ligands the adapter cannot yet express. The
useful output is the failure list, not the pass rate.

**Input-surface coverage.** The adapter implements `DNAInput`, `RNAInput` and
SMILES ligands, but no fixture exercises them, so they are untested rather than
known-good. One parity case per implemented branch would close that; see the
matrix in [02_PARITY.md](02_PARITY.md).

**`LoadPolymerMSAs` wiring.** MSA transfer and pairing are verified
([02_PARITY.md](02_PARITY.md)), but the alignments in those tests are
constructed directly. Composing AtomWorks' own MSA loader into
`pre_transforms` and re-checking would close the last step of that path.

## Missing for training

The Foundry trainer contract is wired up (`training_step` / `validation_step`,
`construct_model`, state, checkpointing). Two things stand between that and a
training run, and both are properties of the port rather than of any particular
objective.

**1. Supervision targets do not exist in the pipeline.**

`StructurePredictionInput` carries no coordinates: it is a sequence- and
chemistry-level description, and ESMFold2 derives all geometry from CCD
reference conformers. So the AtomWorks structure's own coordinates are *not*
transferred by the adapter, by construction.

This has a consequence that is easy to misread. `feats["gt_coords"]` is part of
the 29 tensors compared by feature parity, and it matches — but it is built from
the *prediction input* and is zeros at inference. **Parity on `gt_coords` does
not mean the source coordinates were carried across.** Both sides simply hold
the same placeholder.

A supervised structural loss therefore needs something the pipeline does not
produce: the source atoms aligned to ESMFold2's atom ordering, in a namespace
of their own (`example["labels"]`, say) rather than overwriting an input
tensor. The alignment is the work — `(chain, residue_index, atom_name)` has to
map onto the tokenizer's ordering, and ligands, modified residues and
unresolved atoms each break the naive correspondence. `chain_infos` is the
bridge: `ChainInfo.tokens[].{token_index, atom_start, atom_count}` is the only
record of it.

**2. The gradient path has a precondition.**

The release `forward` is `@torch.inference_mode()` and can never yield
gradients. The experimental one can, but gates them on an input:

```python
torch.set_grad_enabled(res_type_soft is not None)   # experimental.py
```

Loading the experimental checkpoint is therefore necessary and **not**
sufficient. `FoundryESMFold2.will_produce_gradients(inputs)` checks the real
condition and `explain_gradient_status(inputs)` says which half is missing;
`tests/test_gradient_contract.py` asserts both, and — given a GPU and the
experimental checkpoint — that a structural objective really does move soft
sequence logits.

**3. No objective.** `compute_loss` is deliberately unimplemented: ESMFold2
ships no training loss, and picking one is a modelling decision that does not
belong in a base class.

## Not planned here

Rewriting ESMFold2 module by module in Foundry idiom. See `D-001` and the
non-goals in [00_SCOPE.md](00_SCOPE.md). Components get separated when something
concrete needs them separated, one at a time, each behind a parity check —
`FoundryESMFold2` exposes `.esmc`, `.folding_trunk` and `.structure_head` as
named seams so that such a change is an implementation change rather than a
change to every call site.
