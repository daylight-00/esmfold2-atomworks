# 05 — What is not done yet

This project is scoped to one thing: making the published ESMFold2 reachable
from AtomWorks structures, faithfully — with Foundry as one optional training
backend. What follows is
what that scope still lacks — not a research plan. Downstream projects that
build on this repository keep their own.

## Done

- **AtomWorks → ESMFold2 adapter**, with exact feature parity against frozen
  reference inputs ([02_PARITY.md](02_PARITY.md)).
- **Output parity on a GPU**, as a controlled comparison against the model's own
  run-to-run scatter.
- **esm ≥ 3.4**, verified on CPU and GPU — as was the esm ≤ 3.3 `transformers`
  fork while it was still published.
- **Inference**: engine, CLI, configs, AtomWorks round trip.
- **Covalent bonds** carried across, with indices read back from the tokenizer.
- **MSA** transfer and cross-chain pairing by `key=<taxid>`.
- **Nucleic acid and SMILES branches** covered.
- **Packaged configs** compose from an installed wheel (checked in CI).
- **Optional Foundry integration** verified against the pinned checkout, not just
  described.

## Missing for inference completeness

**Corpus coverage.** `esmfold2-atomworks parity <dir>/*.cif` over a larger set,
to enumerate the chain types and ligands the adapter cannot yet express. The
useful output is the failure list, not the pass rate.

**`LoadPolymerMSAs` wiring.** MSA transfer and pairing are verified
([02_PARITY.md](02_PARITY.md)), but the alignments in those tests are
constructed directly. Composing AtomWorks' own MSA loader into
`pre_transforms` and re-checking would close the last step of that path.
Whatever the loader attaches must bind ([01](01_ADAPTER.md#declarations-must-bind)):
an alignment is accepted only for a protein chain and only with the folded
sequence as its query row, so an RNA alignment — ESMFold2 has no input for one —
has to be left out rather than passed through.

## Missing for training

The Foundry trainer contract is wired up (`training_step` / `validation_step`,
`construct_model`, state, checkpointing), and supervision targets are
available. What remains is described below: the alignment's one caveat, the
gradient precondition, and the objective -- which stays with the caller.

**1. Supervision targets: available, but the alignment is by name.**

`StructurePredictionInput` carries no coordinates: it is a sequence- and
chemistry-level description, and ESMFold2 derives all geometry from CCD
reference conformers. So the AtomWorks structure's own coordinates are *not*
transferred by the adapter, by construction.

This has a consequence that is easy to misread. `feats["gt_coords"]` is part of
the 29 tensors compared by feature parity, and it matches — but it is built from
the *prediction input* and is zeros at inference. **Parity on `gt_coords` does
not mean the source coordinates were carried across.** Both sides simply hold
the same placeholder.

`build_esmfold2_pipeline(attach_labels=True)` now supplies the missing half:
`example["labels"]` holds the source coordinates permuted onto the model's atom
axis, plus a mask. Unresolved atoms are **masked, never imputed** — a
placeholder would silently bias any loss that averages over atoms — and nothing
is centred or normalised, because that belongs to the objective.

The remaining caveat is the alignment itself. It matches on
`(chain, residue index, atom name)`, which is exact for standard residues and
CCD ligands but has no answer where the source and the CCD disagree about atom
naming. `StructureLabels.coverage` and `unmatched_examples` report what did not
match, and `AttachStructureLabels(require_coverage=...)` will refuse a structure
that falls below a threshold — a silently half-matched label set being worse
than a refused one.

**2. The gradient path has a precondition.**

The release `forward` is `@torch.inference_mode()` and can never yield
gradients. The experimental one can, but gates them on an input:

```python
torch.set_grad_enabled(res_type_soft is not None)   # experimental.py
```

Loading the experimental checkpoint is therefore necessary and **not**
sufficient. `AtomWorksESMFold2.will_produce_gradients(inputs)` checks the real
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
`AtomWorksESMFold2` exposes `.esmc`, `.folding_trunk` and `.structure_head` as
named seams so that such a change is an implementation change rather than a
change to every call site.
