# 05 — Status and open items

This project is scoped to one thing: making the published ESMFold2 reachable
from AtomWorks structures, faithfully — with Foundry as one optional training
backend. This page lists what is done in that scope, what is limited by an
upstream, and what remains; it is not a research plan.

## Done

- **AtomWorks → ESMFold2 adapter**, with exact feature parity against frozen
  reference inputs ([02_PARITY.md](02_PARITY.md)).
- **Output parity on a GPU**, exact under deterministic kernels, with the
  non-deterministic scatter characterized separately.
- **esm 3.4.1.post1**, verified on CPU and GPU.
- **Inference**: engine, CLI, configs, AtomWorks round trip — atoms, names,
  chains and bonds ([01](01_ADAPTER.md), "The bond list of the returned
  structure").
- **Covalent bonds** carried across, with indices read back from the tokenizer.
- **MSA** transfer and cross-chain pairing by `key=<taxid>`.
- **Nucleic acid and SMILES branches** covered.
- **The packaged engine config** composes from an installed wheel (checked in CI).
- **Optional Foundry integration** verified against the pinned checkout, not just
  described.
- **Declared chain kinds**, verified against `chain_type`, `is_polymer` and the CCD
  (`chain_kinds=`); a chain that holds more than one molecule is refused
  ([01](01_ADAPTER.md)).
- **The return leg's names and kinds**: each declared ligand chain keeps its name, and
  each chain's source `chain_type` is carried to the output.
- **Folding with supplied LM hidden states** (`compute_lm_hidden_states`,
  `fold(lm_hidden_states=)`); a fold with no LM prior is refused
  ([03](03_MODEL.md)).
- **Provenance**: a record per fold call (`fold(record=)`) and the model's by revision
  (`provenance()`); the checkpoint, ESMC backbone and CCD pinned by digest in
  `reproducibility/ARTIFACTS.lock`.
- **Metrics**: the scalar sources as a read-only table (`SCALAR_METRIC_SOURCES`), and
  the two pLDDT axes by name (`plddt_per_token`, `plddt_per_residue`).
- **AtomWorks' own MSA loader**, `LoadPolymerMSAs`, wired in with
  `build_esmfold2_pipeline(msa_loader=...)`: what it loads features exactly like
  the file it came from, heteromer rows pair by its TaxIDs, and an alignment it
  finds for a different protein is refused ([02](02_PARITY.md)).

## Ongoing validation

Inference is complete for the scope above; what follows has no natural end, so
it is validation that continues rather than a missing piece.

**Corpus coverage.** `esmfold2-atomworks parity <dir>/*.cif` over a larger set,
to enumerate the chain types and ligands the adapter cannot yet express. The
useful output is the failure list, not the pass rate.

## Limited by an upstream

Each of these is a gap in what an upstream records, so the fix belongs there;
the package refuses rather than guesses meanwhile.

- **Residue boundaries of a multi-component ligand** (esm). ESMFold2's output
  builder returns each non-polymer chain as one residue, although its
  tokenizer keeps the component index of every atom. Once it returns one
  residue per component, each declared component comes back under its own code
  with no change here.
- **Bonds in the output** (esm). `MolecularComplex` has no bond field;
  `fold_atom_array` rebuilds the structure's bond list from the chemistry the
  model was given ([01](01_ADAPTER.md), "The bond list of the returned
  structure"), and `result_to_atom_array` alone returns none.
- **Insertion-coded positions in PDB files** (AtomWorks). Parsed from a PDB
  file, `res_id` is the author numbering and `chain_info` lists a repeated
  number once per insertion code with no code beside it, so covalent bonds and
  labels on such a chain are refused or skipped. AtomWorks' parser also drops
  the coordinates of all but one residue sharing a number, restoring them as
  unresolved, so such a chain's labels would be wrong even if placed. mmCIF
  input is unaffected
  ([02](02_PARITY.md), "Chains whose insertion codes cannot be placed").
- **`lm_dropout=0`** (esm). `fold` documents `0` as switching the LM dropout
  off, but upstream treats `0` like `None` and leaves the checkpoint's rate,
  `0.25` in the released ESMFold2 and ESMFold2-Fast configs.
  `esmfold2.effective.lm_dropout` states the rate that applies, so a fold's
  record shows it; switching the dropout off means setting it on the loaded
  model's config.

## Missing for training

The Foundry trainer contract is wired up (`training_step` / `validation_step`,
`construct_model`, state, checkpointing), and supervision targets are
available. What remains is described below: the alignment's one caveat, the
gradient precondition, and the objective -- which stays with the caller.

**1. Supervision targets: available, but the alignment is by name.**

`StructurePredictionInput` carries no coordinates: it is a sequence- and
chemistry-level description, and ESMFold2 derives all geometry from CCD
reference conformers. So the AtomWorks structure's own coordinates are *not*
transferred by the adapter, by construction, and parity on `gt_coords` says
nothing about them ([02](02_PARITY.md), "What parity does *not* cover").

`build_esmfold2_pipeline(attach_labels=True)` supplies them:
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

A SMILES ligand is the case where names say nothing: ESMFold2 names its atoms by
element and rank, so a source atom of the same name can be another atom, and a
label put on it covers every atom and passes any coverage floor. Its atoms are
matched through the ligand's graph instead, by the rule that places a covalent
bond on it ([01](01_ADAPTER.md), "Covalent bonds"), so a bond and a label always
mean the same atom. A ligand that cannot be matched — no bond list, an unstated
bond order, another graph — raises. `on_smiles_match_failure="mask"`, on
`AttachStructureLabels` and `build_esmfold2_pipeline`, leaves its chain unlabelled
instead and names it, with the reason, in `label_unmatched_smiles_chains`.

The match is of connectivity and bond orders. It separates atoms in different
chemical environments; it does not pick between atoms the graph cannot tell apart
(a CF3's fluorines, the carbons of a symmetric ring), so which of those carries
which coordinate is one of several equivalent assignments. An objective that is
sensitive to atom permutation has to handle that symmetry itself, as it already
does for a standard residue's `OD1`/`OD2`.

**2. The gradient path has a precondition.**

The release `forward` can never yield gradients; the experimental one gates them
on an input (`res_type_soft`), so loading the experimental checkpoint is
necessary and **not** sufficient ([03](03_MODEL.md), "Gradients depend on the
inputs"). `AtomWorksESMFold2.will_produce_gradients(inputs)` checks the real
condition and `explain_gradient_status(inputs)` says which half is missing;
`tests/test_gradient_contract.py` asserts both, and — given a GPU and the
experimental checkpoint — that a structural objective moves soft sequence
logits.

**3. No objective.** `compute_loss` is deliberately unimplemented: ESMFold2
ships no training loss, and picking one is a modelling decision that does not
belong in a base class.

## Not planned here

Rewriting ESMFold2 module by module in Foundry idiom. See `D-001` and the
non-goals in [00_SCOPE.md](00_SCOPE.md). Components get separated when something
concrete needs them separated, one at a time, each behind a parity check.
