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

## Why the feature level is the primary check

`prepare_esmfold2_input` is a pure function of the `StructurePredictionInput`,
and `ESMFold2Model.forward` is a pure function of its output. So two inputs that
featurize to the same tensors produce the same prediction **by construction** —
there is no arithmetic in between to accumulate error, which is why the
comparison is exact rather than tolerance-based.

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
gives coordinates differing by **0.25 A** in the worst atom. Non-deterministic
GPU kernels and bf16 reduction ordering differ between launches, and the
diffusion steps amplify it.

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

The cross-path difference is **indistinguishable from the model's disagreement
with itself**, and on lysozyme it is smaller. Together with exact feature
parity, that is as strong as an output-level statement about a stochastic
sampler can be.

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
