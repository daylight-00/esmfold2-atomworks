# 00 — Scope and decisions

## What this project is

> **ESMFold2 on AtomWorks structures, with native ESMFold2 feature semantics
> preserved exactly.**

Not a port. A port starts by rewriting an architecture; this makes an existing,
published, working model reachable from AtomWorks — the representation it is
fed from and returned to — and proves that the reachable version is the same
model. The whole promise fits in one sentence:

> Given the same biological system, using an AtomWorks structure through this
> adapter must produce the same ESMFold2 features as constructing the
> equivalent native ESMFold2 input directly.

```
AtomWorks structure  ─F─>  StructurePredictionInput  ──>  ESMFold2's own input path

ESMFold2(I_native)  ==  ESMFold2(F(A_AtomWorks))
```

That milestone is **met** — see [02_PARITY.md](02_PARITY.md). It involves no
training framework, and neither does the rest of the core: the adapter and its
reverse, the AtomWorks data pipeline, the supervision labels, the model
wrapper, the inference engine and the CLI.

Foundry is one supported integration on top of that core: training,
configuration and checkpointing on Foundry's trainer, without changing the
native model ([06_FOUNDRY_INTEGRATION.md](06_FOUNDRY_INTEGRATION.md)). It keeps
Foundry's model-package conventions, so an upstream integration stays thin.

## Decisions

Decisions carry IDs so the code can cite them.

### D-001 — The adapter converts at `StructurePredictionInput`, not at the tensors

Both stacks featurize to an AF3-like layout and the names overlap enough to look
interchangeable. They are not:

| quantity | AtomWorks | ESMFold2 |
|---|---|---|
| residue type | `restype` `(N, 32)` one-hot | `res_type` `(N,)` index |
| atom→token | `atom_to_token_map` | `atom_to_token` |
| element | `ref_element` `(A, 128)` one-hot | `ref_element` `(A,)` atomic number |
| atom names | `ref_atom_name_chars` `(A, 4, 64)` | `ref_atom_name_chars` `(A, 4)`, `ord(c)-32` |
| molecule kind | `is_protein`/`is_rna`/`is_dna`/`is_ligand` | `mol_type` `(N,)` enum |
| MSA | `msa_stack` `(R, M, N, 34)` | `msa`, `has_deletion`, `deletion_value`, split |

Rebuilding ESMFold2's tensors from AtomWorks' would mean reproducing every one
of those conventions exactly. A subtle mismatch yields a model that runs,
reports plausible confidence, and is wrong — the failure this project exists to
prevent. `StructurePredictionInput` is the declarative layer above both, so
converting there lets ESMFold2's own `prepare_esmfold2_input` build the tensors
it expects.

**Consequence:** parity is checkable at the feature level, exactly, on CPU.

### D-002 — Chain classification comes from `chain_type`, never from residue names

AtomWorks annotates every atom with its `chain_type` (`atomworks.enums.ChainType`).
The adapter maps that enum onto ESMFold2's four input classes. A chain type the
mapping does not know **raises** (`UnsupportedChainError`), rather than falling
into whichever branch happened to be the default. So does a chain with no
`chain_type` at all (`InferredChainKindError`): the only fallback, `is_polymer`,
cannot tell protein from DNA or RNA, and a DNA chain read that way folds as a
protein of unknown residues.

### D-003 — Sequences come from `chain_info`, not from the residues that were modelled

`processed_entity_canonical_sequence` includes residues that were never
resolved. Reading the sequence off the atoms instead silently deletes every
unmodelled loop, and folds a construct the caller never asked for — which then
reports perfectly good confidence, because it *is* a confident prediction of a
different molecule.

The adapter prefers `chain_info` and records which source it used per chain
(`AdapterReport.sequence_source`).

### D-004 — Ligand identity is declared and verified, never inferred from a label

A residue name is a label, not an identity. `LIG`, `UNL` and `UNK` are all real
CCD codes *and* the strings a model writes on anything it was given as SMILES.
Reconciling such a ligand against the dictionary keeps only the atoms whose
names happen to match: in one observed case this turned 19 correct ligand atoms
into 8 carbons, leaving a structure that still parsed and still validated.

So: a generic label is **refused**. A CCD code that came through
`atomworks.io.parse` has been reconciled against the component dictionary and is
a real statement about chemistry, so it is accepted — and a declared
`expected_formula` is checked against the heavy atoms present.

### D-005 — Nothing is dropped or approximated silently

A conversion that quietly discards a chain, a covalent bond or a modification is
the same class of failure as D-004: the fold succeeds, the metrics look
reasonable, and the model was given a different system than the caller believes.

Recording such a degradation is not enough, because the direct path
(`fold_atom_array`) returns no report. So every degradation the adapter can
detect **raises** by default and is accepted only by naming it
(`atomworks_to_esm.DEGRADATIONS`); what was accepted is then named in
`AdapterReport` and, on the engine path, in each output's metadata. Dropping
water is an explicit policy (`drop_water`), not a degradation.

The same holds for what the caller declares. A sequence override, an MSA or a
ligand identity that would not reach the chain it names — a typo, a chain of the
wrong kind, an alignment built for another sequence — is refused
(`ChainDeclarationError`): ignoring it is the same failure seen from the
caller's side.

### D-006 — A metric the model did not produce stays absent

Never defaulted to `0.0`. A fabricated zero reads downstream as a real
measurement — and for pLDDT it reads as a catastrophic fold, for PAE as a
perfect one.

### D-007 — Non-standard residues are declared by CCD code

ESMFold2's one-letter alphabet covers the standard residues only. Anything else
— selenomethionine, a D-amino acid, a phosphoserine — must become a
`Modification(position, ccd=...)`, or the model folds the **parent** residue.

This changes tokenization (a modified residue becomes one token per atom), which
is the point: it is how the real chemistry enters the model. `tests/` asserts
that declaring it actually changes the token count, so the parity test above it
cannot be vacuous.

### D-008 — Parity is checked at the feature level first, the output level second

Feature parity is exact, needs no GPU and no weights, and names the offending
tensor when it fails. Output parity is a tolerance check on a sampled structure
and can only ever confirm. Relying on the second alone means judging an
implementation from a single paired stochastic run: the structure head is a
diffusion sampler, a shared seed does not guarantee a shared trajectory when two
paths consume randomness differently, and the resulting scatter is easy to read
as a real difference — or to mistake a real difference for scatter.

## Non-goals, for now

- **Rewriting the architecture.** Both AtomWorks and the Biohub fork are
  explicitly mid-cleanup (AtomWorks' README says so). A full rewrite would spend
  its first months chasing upstream API changes.
- **Training the released model.** It cannot be trained; see
  [03_MODEL.md](03_MODEL.md) §Gradients.
