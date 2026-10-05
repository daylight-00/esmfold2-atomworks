# 02 — Parity: method and result

## Result

**Feature parity holds exactly on every fixture tested.** All 29 tensors that
`prepare_esmfold2_input` emits are identical between a frozen reference
`StructurePredictionInput` and the one the adapter derives from the same
structure:

| fixture | system | tensors identical |
|---|---|---|
| `6lyz` | lysozyme, 129-residue monomer | 29 / 29 |
| `2hhb` | haemoglobin, α₂β₂ + 4 × HEM | 29 / 29 |
| `test_cif_loading_4q8n` | protein + Zn²⁺ | 29 / 29 |
| `8cjg_from_af3` | 663 residues + FAD + UV3 | 29 / 29 |
| `1a8o_modified` | monomer with 4 × MSE | 29 / 29 |

## Corpus survey

`esmfold2-atomworks parity -v` over every structure file under AtomWorks'
`tests/data` (17 at the pinned revision):

```
[ok ] 101m_arginine_nh1nh2_swapped.cif           all 29 tensors identical
[ok ] 1a8o_modified.cif                          all 29 tensors identical
[ok ] 1qfe.pdb                                   all 29 tensors identical
[ok ] 2hhb.cif.gz                                all 29 tensors identical
[ok ] 6lyz.bcif                                  all 29 tensors identical
[ok ] 7ubd_from_af3.cif                          all 29 tensors identical
[ok ] 8cjg_from_af3.cif                          all 29 tensors identical
[ok ] example_conditional_generation_output.cif  all 29 tensors identical
[ok ] test_cif_loading_4q8n.cif.gz               all 29 tensors identical
[ok ] UniRef50_A0A0S8JQ92_AF2_predicted.pdb      all 29 tensors identical
[ok ] UniRef50_A0A1H9L980.cif                    all 29 tensors identical
[ok ] UniRef50_A0A1Q4X5U9.cif                    all 29 tensors identical
[ok ] UniRef50_UPI000A006E95.cif                 all 29 tensors identical
[FAIL] 9cox_with_unknown_ccd.cif          LigandIdentityError: chain 'C' is labelled ['UNKNOWN_CCD']
[FAIL] example_distillation_output.cif    LigandIdentityError: chain 'B' is labelled ['UNL']
[FAIL] example_ncaa.cif                   LigandIdentityError: chain 'B' is labelled ['C:0']
[FAIL] test_unl_ligand_with_bonds.cif     LigandIdentityError: chain 'A' is labelled ['UNL']

feature parity: 13/17 structures reproduce
```

The four failures are refusals by design:

- `9cox_with_unknown_ccd` carries a residue named `UNKNOWN_CCD`, which is not in
  the component dictionary, so it cannot be used as a CCD code and there is no
  chemistry to fold. The adapter checks the name against the CCD and refuses it
  by name, rather than letting it fail inside the featurizer with
  `CCD component UNKNOWN_CCD not found`, which names neither the chain nor the
  remedy.
- `example_ncaa` carries a component named `C:0`, a placeholder rather than a CCD
  code (AtomWorks names its own SMILES components `L:n`); the same check catches
  it.
- `example_distillation_output` and `test_unl_ligand_with_bonds` carry `UNL`
  ligands, which D-004 refuses. Declaring one with a `LigandSpec` makes it
  fold — `tests/` asserts exactly that round trip.

That distinction is the reason the survey prints the exception rather than a
count: a refusal and a bug look the same in a pass rate. For the same reason
`tests/test_corpus_survey.py` pins each file's outcome — reproduces, or refused
with the named error on the named chain — and fails on a corpus file it does not
classify, so a new AtomWorks revision shows up as a test to update rather than
a stale table.

The survey is a self-consistency check, not an independent one: its reference
side is the adapter's own input with the sequences re-read from `chain_info`
(`parity/run.py`). It finds structures the adapter cannot process; it does not
prove that the ones it can are right. That is what the gold fixtures above are
for.

Coverage in the thirteen that pass: PDB and mmCIF and binary CIF, monomer and
multimer, AFDB-style predictions, cofactor/metal ligands (HEM, FAD, UV3, ZN,
NBN), a covalently bound ligand on each of two chains (DHS on `1qfe`), a cyclic
D-peptide given as one eight-component non-polymer chain (`7ubd`), and four
selenomethionines (`1a8o`).

Reproduce with:

```bash
pytest tests/test_feature_parity.py -q      # CPU, no weights
esmfold2-atomworks doctor                    # same check on 2hhb, plus the environment
```

Four of the five parity cases (lysozyme, haemoglobin, a zinc site,
selenomethionine) read structures that are in this repository
(`tests/data/structures`), so they run from a clone. The `flavoprotein` and `unl`
cases are AF3-derived, which carry output terms of use and stay out; they read
AtomWorks' test structures, which come with an atomworks checkout beside this
repository, and skip without one. They pass in an ordinary install as well as in
the reference environment ([`reproducibility/`](../reproducibility/README.md)).

## Coverage

What is tested, and what is merely implemented. The distinction matters: the
adapter has code paths for inputs no fixture exercises, and those are untested
rather than known-good.

| input | status |
|---|---|
| protein monomer | exact feature parity |
| protein multimer | exact feature parity |
| CCD ligand / cofactor / metal | exact feature parity (HEM, FAD, UV3, ZN); survey only for NBN and DHS |
| modified residue | exact feature parity (4 × MSE, positions asserted) |
| D-amino acids as a non-polymer chain | survey only (`7ubd`): self-consistent, not compared against a frozen input |
| `POLYPEPTIDE_D`, `CYCLIC_PSEUDO_PEPTIDE`, `BRANCHED` and `MACROLIDE` chains | mapped (docs/01), not exercised: no fixture carries one |
| multi-component non-polymer chain | survey only (`7ubd`, eight components); returned as one residue (docs/01, "What `G` does not return") |
| covalent bond | survey only for deposited bonds (`1qfe`, `7ubd`); carried and placed; changes `token_bonds` and nothing else. Backbone adjacency judged by residue order, so an insertion-coded peptide bond is not declared; an unplaceable bond raises |
| MSA, single chain | exact feature parity against a hand-written input |
| MSA, paired heteromer | pairing verified by row content, not just shape |
| MSA binding | refused unless the query row is the folded sequence, which upstream would clamp into place; lookup by chain id verified by swapping alignments between identical chains |
| MSA from AtomWorks' loader | `LoadPolymerMSAs` output features exactly like the same a3m handed over directly, insertions included; heteromer rows pair by the loader's TaxIDs, and not without them; the alignment AtomWorks' own `1wym` case pairs with that structure is for another protein, and is refused |
| DNA | branch covered (`mol_type` 1, duplex) |
| RNA | branch covered (`mol_type` 2) |
| protein–nucleic complex | branch covered (`mol_type` {0, 1}) |
| SMILES ligand | branch covered, declared; reproducible at a fixed seed |
| generic `UNL` / unknown CCD | **refused**, with a test asserting the refusal |
| SMILES placeholder name (`L:0`) | **refused**, with the remedy in the message |

The rule the repo follows: *what is supported is tested or listed above as
untested, and what is not supported is refused explicitly.*

The nucleic-acid and SMILES rows are branch coverage rather than parity against
a frozen reference: their fixtures are built with AtomWorks' component
assembler rather than deposited, so there is no independent description to
compare against. They assert that the right input class, sequence and
`mol_type` come out, which is what the adapter decides.

## What the adapter is compared *against*

`tests/data/gold/*.json` holds a frozen `StructurePredictionInput` per fixture:
chains, sequences, ligand CCD codes, modification positions.

The reference side is never built from the adapter's output. A reference that
copied chain ids, ligand codes and modifications from it would be circular: an
adapter that consistently mapped a ligand to the wrong CCD code would have that
error copied into the reference, and parity would pass.

The gold files are generated once from AtomWorks' own `parse()` output —
`chain_info`, `chain_type`, residue names — never from this package, and then
checked in so they cannot follow a change in adapter behaviour.

A frozen file is only as good as its contents, so `tests/test_gold_fixtures.py`
anchors them to facts about the entries themselves: 6LYZ is one 129-residue
chain beginning `KVFGRCELAAAM…`; 2HHB is α₂β₂ with two 141-residue and two
146-residue chains and four `HEM`; 1A8O carries `MSE` at positions 0, 34, 63 and
64, each where the canonical sequence reads `M`. Those are checkable against the
PDB without running any of this code.

`test_a_wrong_ligand_would_be_caught` completes the argument from the other
side: substituting `HEC` for `HEM` must break feature parity. Without it,
"all 29 tensors identical" would be reassuring without being informative.

## Why the feature level is the primary check

`prepare_esmfold2_input` is a pure function of the `StructurePredictionInput`
**at a fixed seed**, so two inputs that describe the same system produce
byte-identical tensors: there is no arithmetic in between to accumulate error,
which is why this comparison is exact rather than tolerance-based.

A ligand given as SMILES gets an RDKit
conformer embedded at call time, which is seeded but stochastic — the same
`LigandInput` at two different seeds yields different `ref_pos`. CCD-specified
ligands read a stored conformer and are unaffected. All the fixtures here are
CCD, which is why the comparison holds exactly; a SMILES case needs the same
seed on both sides, or a tolerance.

`forward` is **not** pure. The structure head is a diffusion sampler; it
consumes RNG, and on a GPU with torch's deterministic algorithms off it is not
even reproducible across two identical seeded calls (see below). So feature parity
by itself does *not* say that the two paths produce the same coordinates. What
it says is:

> the adapter presents the model with exactly the same conditioning, and
> therefore the same conditional sampling distribution

Everything downstream — coordinates, pLDDT, PAE — is then a draw from one
distribution rather than from two. The GPU check below goes
further: with the kernels made deterministic, the realised draws are
identical.

It is also the diagnostic level. A failure names the tensor: a wrong
`res_type` is a sequence bug, a wrong `ref_element` is a ligand-identity bug, a
wrong `asym_id` is a chain-ordering bug. Output parity would report "the
coordinates differ by 4 Å" for all three.

## The 29 tensors are not equally important

Only **25** are declared parameters of the release `EsmFold2Model.forward`. The
other four land in `**kwargs` and are discarded:

```
gt_coords  is_resolved  frames_idx  pocket_feature
```

`gt_coords` and `is_resolved` are zeros/all-False at inference by construction
(there are no experimental coordinates to supply), and `pocket_feature` is
unconditionally zeroed — upstream labels it `# --- Pocket (dropped) ---` in
`prepare_input.py`. They are training-time features.

`FeatureDiff` therefore separates the two sets: `.ok` means nothing the model
reads differs, `.identical` means all 29 match. A difference in a discarded
tensor cannot change a prediction, so it is reported without failing the check;
a difference in one the model reads always fails it.

**On the fixtures above, `.identical` holds** — the stronger statement.

## Output parity: exact, under deterministic execution

```bash
sbatch --partition=<gpu-partition> scripts/parity_gpu.sbatch
```

Two statements, kept apart because they are about different things:

1. **Under deterministic kernels the two paths fold identically.** With
   `torch.use_deterministic_algorithms(True)` and a fixed cuBLAS workspace
   (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, which the sbatch script exports before
   Python starts), the same input folds to the same structure twice, bit for
   bit -- and the adapted input folds to exactly the structure of the frozen
   native input. This is the output-level parity claim.
2. **With torch's deterministic algorithms off, a seeded fold is not
   repeatable**, the fixed cuBLAS workspace notwithstanding. That is a property
   of the execution configuration, not of either path, and it is measured
   below as a characterization rather than used as a tolerance.

Measured on an RTX 6000 Ada with the current `biohub/ESMFold2` (revision
`69869f73`, whose weights are identical to the reference checkpoint's --
[`checkpoint_equivalence.json`](../reproducibility/checkpoint_equivalence.json)),
`seed=0`, deterministic kernels. "Short" is 1 loop and 8 steps, what the GPU
suite gates on every run; "checkpoint" leaves the loop count to the checkpoint
(20) and samples 100 steps. The full record, with the device, software stack
and commit (`662802b`, with AtomWorks 3.0.0 as a package), is
[`output_parity.json`](../reproducibility/output_parity.json), regenerated by
[`scripts/measure_output_parity.py`](../scripts/measure_output_parity.py):

| fixture | schedule | native, folded twice | native vs adapted |
|---|---|---|---|
| lysozyme | short | identical | identical |
| lysozyme | checkpoint | identical | identical |
| haemoglobin | short | identical | identical |
| haemoglobin | checkpoint | identical | identical |
| 1a8o (4 × MSE) | short | identical | identical |
| 4q8n (protein + Zn²⁺) | short | identical | identical |
| 8cjg (663 residues + FAD + UV3) | short | identical | identical |

"Identical" is every compared quantity at zero difference: coordinates, pLDDT,
PAE, distogram, pTM and ipTM, and atom names. The test folds the native input
twice before comparing paths, so a cross-path difference is never read as the
adapter's while the execution itself is not repeatable. Enabling torch's
deterministic algorithms cost about 12-16% in wall time on these folds (the faster
of the second and third fold of each case, the first carrying warm-up; one
haemoglobin fold at the checkpoint's schedule took 319 s against 236 s for its
repeats, on a card shared with other jobs), measured with the fixed cuBLAS
workspace in place in both modes -- the cost of the flag under that workspace, not
of deterministic execution against a fully default one.

Identical, not merely close, is what the exact feature parity above predicts:
the model receives the same tensors, and with the kernels made deterministic
nothing else differs. The claim holds for one device and software stack.
Deterministic kernels make a fold repeatable there; they do not make two
devices, or two builds, agree.

### The non-deterministic scatter, characterized

With torch's deterministic algorithms off and every RNG seeded identically,
folding the same input twice gives the numbers below. Both modes ran in one
process started with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, as the GPU suite's
characterization test does too, so this is the scatter left under the fixed
cuBLAS workspace with the flag off -- not that of a fully default execution,
which would need a separate process started without the variable.

| fixture | schedule | coordinates, largest difference (Å) | pLDDT, worst token |
|---|---|---|---|
| lysozyme | short | 0.32 | 0.060 |
| lysozyme | checkpoint | 0.33 | 0.001 |
| haemoglobin | short | 0.44 | 0.043 |
| haemoglobin | checkpoint | 0.29 | 0.017 |
| 1a8o | short | 0.58 | 0.032 |
| 4q8n | short | 0.30 | 0.043 |
| 8cjg | short | 0.75 | 0.038 |

These describe one pair of runs on this device and stack; the scatter varies
from run to run, with the input, and with the schedule in no fixed direction,
so they characterize its size rather than bound it. `fold` applies
`lm_dropout=0.3` by default (the checkpoints' own rate is 0.25), and the dropout
stays active at inference. Its mask is drawn from the seeded RNG like everything
else, but whether it contributes to this scatter is not established: passing
`lm_dropout=0` is no control, because upstream leaves the checkpoint's own rate
in place for it ([05](05_ROADMAP.md), "Limited by an upstream"). What removed
the scatter here is turning torch's deterministic algorithms on under that
workspace, which is why the parity check runs with both, and why a scatter
budget -- only as tight as the scatter happens to be on the input at hand -- is
not used as one.

`test_nondeterministic_scatter_is_characterized` prints the measurement and
asserts only the bookkeeping: atom names and order are not sampled, so they
agree whatever the kernels do. A fold's own record states the settings it ran
under (`fold(record=...)`: `esmfold2.deterministic_algorithms`,
`esmfold2.cublas_workspace_config`), as observed at the call -- cuBLAS reads the
variable when CUDA starts, so a value set later is recorded but was never
applied, and the record does not claim otherwise.

## What parity does *not* cover

- **The source structure's coordinates.** `gt_coords` is among the 29 tensors
  and it matches — but `StructurePredictionInput` carries no
  coordinates at all, and ESMFold2 derives geometry from CCD reference
  conformers. `gt_coords` is built from the *prediction input* and is zeros at
  inference, so both sides agree on a placeholder. **Matching `gt_coords` does
  not mean the AtomWorks coordinates were transferred.** The default feature
  path does not transfer them at all; `build_esmfold2_pipeline(attach_labels=True)`
  does, into `example["labels"]` — see [05](05_ROADMAP.md).
- **SMILES ligands.** RDKit conformer embedding is the one genuinely stochastic
  step in featurization. With a fixed seed it is reproducible, but a ligand
  declared as SMILES on one side and CCD on the other will differ in `ref_pos`
  legitimately. `compare_features` takes `atol` for this case.
- **Chains whose insertion codes cannot be placed.** `chain_info` records no
  insertion code. From mmCIF that does not matter: AtomWorks' `res_id` is
  `label_seq_id`, one number per position, so an insertion-coded chain maps
  like any other (`tests/test_insertion_codes.py` maps such a chain). From a PDB
  file `res_id` is the author
  numbering, where 100/100A/100B share a number and `chain_info` repeats it, so
  residues cannot be tied to sequence positions. For such a chain the fold
  input is unaffected, but its labels are skipped (and named in
  `label_skipped_chains`) and a covalent bond on it raises as unplaceable,
  rather than either being attached to whichever residue happened to be
  written last. The mmCIF form of the same entry avoids it.
- **Anything with a `chain_type` the mapping does not know.** Raises
  `UnsupportedChainError` rather than being folded; it can be dropped by name
  (`allow_unsupported_chains`).
