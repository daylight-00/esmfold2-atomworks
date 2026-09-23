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

`esmfold2-foundry parity` over every structure in the AtomWorks test corpus:

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
[FAIL] 9cox_with_unknown_ccd.cif          ValueError: CCD component UNKNOWN_CCD not found
[FAIL] example_distillation_output.cif    LigandIdentityError: chain 'B' is labelled ['UNL']

feature parity: 9/11 structures reproduce
```

Both failures are **correct refusals**, not gaps:

- `9cox_with_unknown_ccd` carries a residue named `UNKNOWN_CCD`, which is not in
  the component dictionary. ESMFold2's own conformer lookup raises; there is no
  chemistry to fold.
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
source env.sh
pytest tests/test_feature_parity.py -q      # ~30 s, CPU, no weights
esmfold2-foundry doctor                      # same check on 2hhb, plus the environment
```

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
| covalent bond | carried and placed; verified to change `token_bonds` and nothing else |
| MSA, single chain | exact feature parity against a hand-written input |
| MSA, paired heteromer | pairing verified by row content, not just shape |
| DNA | **implemented, untested** |
| RNA | **implemented, untested** |
| protein–nucleic complex | **implemented, untested** |
| SMILES ligand | **implemented, untested** (needs a seeded-conformer tolerance) |
| generic `UNL` / unknown CCD | **refused**, with a test asserting the refusal |

The rule the repo follows: *what is supported is tested, what is not supported
is refused explicitly.* The untested rows are the remaining gap between those
two.

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
consumes RNG, and on a GPU it is not even reproducible across two identical
calls (see below). So feature parity does *not* say that the two paths produce
the same coordinates. What it says is:

> the adapter presents the model with exactly the same conditioning, and
> therefore the same conditional sampling distribution

which is the claim worth making, and the strongest one available for a
stochastic model. Everything downstream — coordinates, pLDDT, PAE — is then a
draw from one distribution rather than from two, and the GPU check below
verifies that the realised draws behave accordingly.

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

## Output parity: measured, and why it is a *controlled* comparison

```bash
sbatch --partition=<gpu-partition> scripts/parity_gpu.sbatch
```

**The sampler is not reproducible, even seeded.** `_seed_context` seeds python,
numpy, torch and CUDA identically for every fold, and the two paths hand the
model identical tensors -- yet folding the *same* input twice on an RTX 6000 Ada
gives coordinates differing by **0.25 A** in the worst atom.

The obvious suspect is `lm_dropout`, which defaults to `0.3` and is left active
at inference on purpose (it is the ensembling mechanism). It is not the cause.
Folding lysozyme twice at each setting:

| `lm_dropout` | `coord_max_abs` (A) | `coord_rmsd` (A) | `plddt_max_abs` |
|---|---|---|---|
| 0.3 (default) | 0.176 | 0.061 | 0.060 |
| **0.0** | 0.193 | 0.048 | 0.033 |

Turning dropout off entirely leaves the scatter where it was. What that
establishes is that **`lm_dropout` is not the cause**; the residue is consistent
with non-deterministic GPU kernels and bf16 reduction ordering amplified over
the diffusion steps, but that has not been isolated here and is stated as the
remaining explanation rather than a demonstrated one. **Do not expect
`lm_dropout=0` to buy reproducibility** — it does not, and a scatter budget is
needed either way.

So an absolute tolerance on coordinates cannot tell "the adapter changed the
input" from "the sampler is not reproducible". The first version of this test
used one and failed at 0.27 A -- which said nothing about the adapter. That is
D-008's warning arriving in practice.

The test therefore folds the native input **twice** to measure the noise floor,
then asserts the native-vs-adapted difference sits inside it. Measured on an
RTX 6000 Ada, `num_loops=1`, `num_sampling_steps=8`, `seed=0`:

| | lysozyme cross | lysozyme self | haemoglobin cross | haemoglobin self |
|---|---|---|---|---|
| `coord_max_abs` (A) | 0.214 | **0.372** | 0.356 | 0.350 |
| `coord_rmsd` (A) | 0.048 | **0.085** | 0.049 | 0.049 |
| `plddt_max_abs` | 0.059 | 0.059 | 0.044 | 0.044 |
| `distogram_max_abs` | 3.46 | 3.60 | 6.12 | 5.48 |
| `pae_max_abs` | 5.56 | 5.74 | 5.75 | 7.19 |
| `atom_name_mismatches` | 0 | 0 | 0 | 0 |

**The cross-path deviation falls within the observed self-scatter** — on
lysozyme it is smaller than it. That is a single paired observation per fixture,
not a distributional claim, and it does not need to be more: the load-bearing
evidence is exact feature parity, and this only has to show that nothing
unexpected happens once the sampler runs.

Three deliberate choices:

- **Atom names and ordering are held to exact equality**, whatever the scatter
  budget. Those are bookkeeping, not sampling; a mismatch there would make every
  coordinate comparison meaningless rather than merely noisy.
- **Coordinates are compared as reported, not after superposition.** The two
  inputs describe the same system in the same order, so a rigid-body difference
  would itself be a finding, and aligning first would hide it.
- **`test_the_sampler_is_not_bitwise_reproducible` guards the control.** If a
  future build folded deterministically, the scatter budget would collapse to
  its floor and these tests would quietly revert to the absolute-tolerance check
  they exist to replace. That test fails loudly instead.

## What parity does *not* cover

- **The source structure's coordinates.** This one is worth stating plainly
  because the parity table invites the opposite reading. `gt_coords` is among
  the 29 tensors and it matches — but `StructurePredictionInput` carries no
  coordinates at all, and ESMFold2 derives geometry from CCD reference
  conformers. `gt_coords` is built from the *prediction input* and is zeros at
  inference, so both sides agree on a placeholder. **Matching `gt_coords` does
  not mean the AtomWorks coordinates were transferred.** Nothing in this
  pipeline transfers them; see [05](05_ROADMAP.md).
- **SMILES ligands.** RDKit conformer embedding is the one genuinely stochastic
  step in featurization. With a fixed seed it is reproducible, but a ligand
  declared as SMILES on one side and CCD on the other will differ in `ref_pos`
  legitimately. `compare_features` takes `atol` for this case.
- **MSAs.** The fixtures fold in single-sequence mode. Pairing is driven purely
  by `key=<taxid>` in the FASTA header, so an adapter that builds MSAs without
  injecting those keys gets no cross-chain pairing at all — silently, and with
  no shape change to reveal it. That is the next parity case to add.
- **Anything with a `chain_type` the mapping does not know.** Reported as
  `unsupported` rather than folded.
