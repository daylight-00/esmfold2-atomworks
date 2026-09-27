# 02 — Parity: method and result

## Result

**Feature parity holds exactly on every fixture tested.** All 29 tensors that
`prepare_esmfold2_input` emits are identical between a hand-written
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

`esmfold2-atomworks parity` over every structure in the AtomWorks test corpus:

```
[ok ] 6lyz.bcif                           all 29 tensors identical
[ok ] 2hhb.cif.gz                         all 29 tensors identical
[ok ] 1a8o_modified.cif                   all 29 tensors identical
[ok ] 101m_arginine_nh1nh2_swapped.cif    all 29 tensors identical
[ok ] 8cjg_from_af3.cif                   all 29 tensors identical
[ok ] test_cif_loading_4q8n.cif.gz        all 29 tensors identical
[ok ] 1qfe.pdb                            all 29 tensors identical
[ok ] 7ubd_from_af3.cif                   all 29 tensors identical
[ok ] UniRef50_..._AF2_predicted.pdb      all 29 tensors identical
[FAIL] 9cox_with_unknown_ccd.cif          LigandIdentityError: chain 'C' is labelled ['UNKNOWN_CCD']
[FAIL] example_distillation_output.cif    LigandIdentityError: chain 'B' is labelled ['UNL']

feature parity: 9/11 structures reproduce
```

Both failures are **correct refusals**, not gaps:

- `9cox_with_unknown_ccd` carries a residue named `UNKNOWN_CCD`, which is not in
  the component dictionary, so it cannot be used as a CCD code and there is no
  chemistry to fold. The adapter checks the name against the CCD and refuses it
  by name; previously it was passed through and failed inside the featurizer
  with `CCD component UNKNOWN_CCD not found`, which named neither the chain nor
  the remedy. The same check catches AtomWorks' placeholder names for ligands
  built from SMILES or an SDF (`L:0`, `C:0`).
- `example_distillation_output` carries a `UNL` ligand, which D-004 refuses.
  Declaring it with a `LigandSpec` makes it fold — `tests/` asserts exactly that
  round trip.

That distinction is the reason the survey prints the exception rather than a
count: a refusal and a bug look the same in a pass rate.

Coverage in the nine that pass: PDB and mmCIF and binary CIF, monomer and
multimer, an AFDB-style prediction, four cofactor/metal ligand types
(HEM, FAD, UV3, ZN, NBN, DHS), D-amino acids as a non-polymer chain (`7ubd`),
and four selenomethionines (`1a8o`).

Reproduce with:

```bash
pytest tests/test_feature_parity.py -q      # CPU, no weights
esmfold2-atomworks doctor                    # same check on 2hhb, plus the environment
```

Both read AtomWorks' test structures, which come with an atomworks checkout
beside this repository. Given those, they pass in an ordinary install as well
as in the pinned reference environment the published numbers come from
([`reproducibility/`](../reproducibility/README.md)).

## Coverage

What is tested, and what is merely implemented. The distinction matters: the
adapter has code paths for inputs no fixture exercises, and those are untested
rather than known-good.

| input | status |
|---|---|
| protein monomer | exact feature parity |
| protein multimer | exact feature parity |
| CCD ligand / cofactor / metal | exact feature parity (HEM, FAD, UV3, ZN, NBN, DHS) |
| modified residue | exact feature parity (4 × MSE, positions asserted) |
| D-amino acids | exact feature parity (`7ubd`) |
| covalent bond | carried and placed; changes `token_bonds` and nothing else. Backbone adjacency judged by residue order, so an insertion-coded peptide bond is not declared; an unplaceable bond raises |
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

The rule the repo follows: *what is supported is tested, what is not supported
is refused explicitly.*

The nucleic-acid and SMILES rows are branch coverage rather than parity against
a frozen reference: their fixtures are built with AtomWorks' component
assembler rather than deposited, so there is no independent description to
compare against. They assert that the right input class, sequence and
`mol_type` come out, which is what the adapter decides.

## What the adapter is compared *against*

`tests/data/gold/*.json` holds a frozen `StructurePredictionInput` per fixture:
chains, sequences, ligand CCD codes, modification positions.

This matters more than it looks. An earlier version of the suite built the
reference side by copying chain ids, ligand codes and modifications out of the
adapter's own output and re-reading only the sequences independently. That is
circular: an adapter that consistently mapped a ligand to the wrong CCD code
would have that error copied into the reference, and parity would pass.

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

The seed qualifier is not pedantry. A ligand given as SMILES gets an RDKit
conformer embedded at call time, which is seeded but stochastic — the same
`LigandInput` at two different seeds yields different `ref_pos`. CCD-specified
ligands read a stored conformer and are unaffected. All the fixtures here are
CCD, which is why the comparison holds exactly; a SMILES case needs the same
seed on both sides, or a tolerance.

`forward` is **not** pure. The structure head is a diffusion sampler; it
consumes RNG, and on a GPU with the default kernels it is not even
reproducible across two identical seeded calls (see below). So feature parity
by itself does *not* say that the two paths produce the same coordinates. What
it says is:

> the adapter presents the model with exactly the same conditioning, and
> therefore the same conditional sampling distribution

which is the claim worth making, and the strongest one available for a
stochastic model. Everything downstream — coordinates, pLDDT, PAE — is then a
draw from one distribution rather than from two. The GPU check below goes
further: with the kernels made deterministic, the realised draws are
identical.

It is also the diagnostic level. A failure names the tensor: a wrong
`res_type` is a sequence bug, a wrong `ref_element` is a ligand-identity bug, a
wrong `asym_id` is a chain-ordering bug. Output parity would report "the
coordinates differ by 4 Å" for all three.

## The 29 tensors are not equally important

Only **23** are declared parameters of the release `ESMFold2Model.forward`. The
other six land in `**kwargs` and are discarded:

```
gt_coords  is_resolved  frames_idx  disto_cond  disto_cond_mask  pocket_feature
```

`gt_coords` and `is_resolved` are zeros/all-False at inference by construction
(there are no experimental coordinates to supply), and `pocket_feature` is
unconditionally zeroed — upstream labels it `# --- Pocket (dropped) ---` in
`prepare_input.py`. They are training-time features.

`FeatureDiff` therefore separates the two sets: `.ok` means nothing the model
reads differs, `.identical` means all 29 match. Failing on a discarded tensor
would make the check cry wolf; ignoring the distinction would let a real
regression hide behind "well, something differs".

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
2. **Without them, a seeded fold is not repeatable.** That is a property of
   the execution configuration, not of either path, and it is measured below
   as a characterization rather than used as a tolerance.

Measured on an RTX 6000 Ada with the current `biohub/ESMFold2` (revision
`69869f73`, whose weights are identical to the reference checkpoint's --
[`checkpoint_equivalence.json`](../reproducibility/checkpoint_equivalence.json)),
`seed=0`, deterministic kernels. "Short" is 1 loop and 8 steps, what the GPU
suite gates on every run; "checkpoint" leaves the loop count to the checkpoint
(20) and samples 100 steps. The full record, with the device, software stack
and commit, is [`output_parity.json`](../reproducibility/output_parity.json),
regenerated by
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
adapter's while the execution itself is not repeatable. Deterministic kernels
cost 12-16% in wall time on these folds (the second and third fold of each
case, the first carrying warm-up).

Identical, not merely close, is what the exact feature parity above predicts:
the model receives the same tensors, and with the kernels made deterministic
nothing else differs. The claim holds for one device and software stack.
Deterministic kernels make a fold repeatable there; they do not make two
devices, or two builds, agree.

### The non-deterministic scatter, characterized

With the kernels left non-deterministic, as by default, and every RNG seeded
identically, folding the same input twice gives:

| fixture | schedule | coordinates, worst atom (Å) | pLDDT, worst token |
|---|---|---|---|
| lysozyme | short | 0.12 | 0.060 |
| lysozyme | checkpoint | 0.18 | 0.001 |
| haemoglobin | short | 0.30 | 0.044 |
| haemoglobin | checkpoint | 0.10 | 0.053 |
| 1a8o | short | 0.42 | 0.052 |
| 4q8n | short | 0.19 | 0.060 |
| 8cjg | short | 0.76 | 0.042 |

These describe one pair of runs on this device and stack; the scatter varies
from run to run, with the input, and with the schedule in no fixed direction,
so they characterize its size rather than bound it. `lm_dropout` is not
the cause: it defaults to `0.3` and stays active at inference on purpose (it
is the ensembling mechanism), but measured on the reference checkpoint,
setting it to `0` left the scatter where it was -- its mask is drawn from the
seeded RNG like everything else. What removes the scatter is the
deterministic configuration above, which is why the parity check runs under
it, and why a scatter budget -- only as tight as the scatter happens to be on
the input at hand -- is not used as one.

`test_nondeterministic_scatter_is_characterized` prints the measurement and
asserts only the bookkeeping: atom names and order are not sampled, so they
agree whatever the kernels do. A fold's own record states the settings it ran
under (`fold(record=...)`: `esmfold2.deterministic_algorithms`,
`esmfold2.cublas_workspace_config`), as observed at the call -- cuBLAS reads the
variable when CUDA starts, so a value set later is recorded but was never
applied, and the record does not claim otherwise.

## What parity does *not* cover

- **The source structure's coordinates.** This one is worth stating plainly
  because the parity table invites the opposite reading. `gt_coords` is among
  the 29 tensors and it matches — but `StructurePredictionInput` carries no
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
  insertion code, so for a chain numbered 100/100A/100B whose sequence comes
  from there, residues cannot be tied to sequence positions. The fold input is
  unaffected, but the chain's labels are skipped (and named in
  `label_skipped_chains`) and a covalent bond on it raises as unplaceable,
  rather than either being attached to whichever residue happened to be
  written last.
- **Anything with a `chain_type` the mapping does not know.** Raises
  `UnsupportedChainError` rather than being folded; it can be dropped by name
  (`allow_unsupported_chains`).
