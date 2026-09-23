# 01 — The adapter

```
AtomWorks AtomArray  ──F──>  StructurePredictionInput  ──>  ESMFold2  ──>  MolecularComplexResult  ──G──>  AtomArray
```

`F` is `data/atomworks_to_esm.py`, `G` is `data/molecular_complex.py`. ESMFold2
itself is untouched.

## What `F` reads

| input | source | why |
|---|---|---|
| chain kind | `chain_type` annotation → `atomworks.enums.ChainType` | D-002 |
| sequence | `chain_info[c]["processed_entity_canonical_sequence"]` | D-003 |
| modifications | `chain_info[c]["res_name"]`, aligned 1:1 with the canonical sequence | D-007 |
| ligand identity | a declared `LigandSpec`, else a verified CCD code | D-004 |
| MSA | passed in per chain | — |

`ChainType` → ESMFold2 input class:

```
POLYPEPTIDE_L / POLYPEPTIDE_D / CYCLIC_PSEUDO_PEPTIDE  -> ProteinInput
DNA                                                     -> DNAInput
RNA                                                     -> RNAInput
NON_POLYMER / BRANCHED / MACROLIDE                      -> LigandInput
WATER                                                   -> dropped, and reported
anything else                                           -> unsupported, and reported
```

## Modification positions are verified, not assumed

`Modification.position` indexes the sequence being folded. That only means
anything if the residue list and the sequence line up, so the adapter uses
`chain_info[c]["res_name"]`, which does.

The alignment is guaranteed **by construction**, not merely observed: AtomWorks
builds the canonical sequence by mapping one-letter codes over that same
`res_name` list, one character per entry, with `"X"`/`"N"`/`"-"` fallbacks on
every path. So `len(sequence) == len(res_name)` holds for any polymer chain.
For polymers `res_name` comes from `entity_poly_seq` — the full deposited
sequence, including residues that were never resolved — which is also why it,
and not the observed residues, is the right thing to fold (D-003).

Checked concretely on `1a8o_modified`: 70 residue names, 70 sequence characters,
and the four `MSE` entries at indices 0, 34, 63, 64 — where the canonical
sequence reads `M`. `tests/test_atomworks_to_esm.py` asserts that, including the
`sequence[position] == "M"` relation, so a change that shifts the alignment
fails loudly rather than folding four wrong residues.

Non-polymer chains have no canonical sequence at all, so the lookup is only made
for polymers.

When the alignment cannot be established the adapter emits **no** modifications
and records the chain in `AdapterReport.unplaceable_modifications`. Folding the
parent residues is an approximation; placing a modification by coincidence is a
wrong molecule.

An overridden sequence (the design path) drops the structure's modifications
entirely — a designed sequence is a different molecule, and the structure's
positions do not apply to it.

## Covalent bonds

ESMFold2 rebuilds connectivity inside a residue from the CCD, and the polymer
backbone from the sequence. Anything else — a ligand bonded to a side chain, a
crosslink between chains — exists only in the source, and the adapter carries it
across as `StructurePredictionInput.covalent_bonds`.

Three decisions worth knowing:

- **Backbone adjacency is judged by residue *order*, not residue number.**
  A chain numbered 100, 100A, 100B, 101 has four consecutive residues, so
  `100/C – 100A/N` is a plain peptide bond. Declaring it would hand the model a
  `token_bonds` edge that an ordinary chain never has, because upstream adds no
  token bond for a standard residue's backbone at all. `100/SG – 100A/SG` is a
  disulphide and is declared; `100/C – 100B/N` skips a residue and is declared.
- **Indices are read back from the tokenizer**, not re-derived — see
  [03](03_FOUNDRY_INTEGRATION.md) and `data/bonds.py` for why.
- **An unplaceable bond raises.** Dropping it folds a connected system as though
  it were disconnected, and the direct `fold_atom_array` path returns no report,
  so nothing would tell the caller. `allow_unresolved_covalent_bonds=True` opts
  into that reading deliberately.

## Strictness: what raises, and what is dropped by policy

The adapter would rather stop than hand back a confident prediction of a
different system. `fold_atom_array` returns no report, so anything it merely
recorded would be invisible to the caller — which makes raising the only way
some facts arrive.

| situation | default | opt out with |
|---|---|---|
| water chain | dropped | `drop_water=False` to refuse instead |
| chain no ESMFold2 input can express | **raises** `UnsupportedChainError` | `allow_unsupported_chains=True` |
| covalent bond that cannot be placed | **raises** `CovalentBondResolutionError` | `allow_unresolved_covalent_bonds=True` |
| ligand labelled `LIG`/`UNL`/`UNK` | **raises** `LigandIdentityError` | declare a `LigandSpec` |
| residue name absent from the CCD | **raises** `LigandIdentityError` | declare a `LigandSpec` |

The two bond-related rows are independent on purpose. Accepting that a chain is
dropped is not the same as accepting that a bond to it disappears, so opting
into the first still raises on the second.

Detection of bonds is deliberately kept separate from that policy:
`covalent_bond_candidates` returns a bond whose endpoint is not in the model
rather than filtering it out, so the decision belongs to the caller. An earlier
version filtered inside detection, which put such bonds beyond the strict check
entirely — an unsupported chain and a real bond to it could both vanish in
silence.

## What `F` refuses

```python
LigandIdentityError: non-polymer chain 'B' is labelled ['UNL'], which carries no
chemical meaning even though each is a real CCD code. Declare it with
LigandSpec(chain_id=..., smiles=...) or ccd=...
```

`LIG`, `UNL`, `UNK` and `UNX` are refused (D-004). A CCD code that came through
`atomworks.io.parse` is accepted, because parsing reconciled it against the
component dictionary — it is a real statement about chemistry rather than a
label a model wrote on its own output.

`LigandSpec.verify_against` compares **heavy atoms only**. Whether hydrogens are
present at all depends on the source: `parse` hydrogenates, Rosetta rebuilds
them on load, ESMFold2 reports none. Heavy atoms are the only comparison that
means the same thing on every side.

## Why `G` does not go through mmCIF

The obvious return route is `MolecularComplex.to_mmcif()` → `atomworks.io.parse()`.
It is in memory and hands back AtomWorks' full annotation set for free.

It is also **silently lossy for ligands**, for exactly the D-004 reason: ESMFold2
labels a SMILES ligand `LIG`, `LIG` is a real CCD code for an unrelated molecule,
and `parse` reconciles against that component, keeping only the atoms whose names
happen to match. In one observed case that turned 19 correct ligand atoms into
8 carbons — and the structure still parsed and still validated.

So `G` reads `MolecularComplex`'s flat per-atom arrays directly. It is faithful
because nothing is inferred that the model did not report, and it raises rather
than returning a partial structure if any atom is claimed by no token span.

## The report

`AdapterReport` records, per conversion: every chain and its classification,
everything dropped and why, the sequence source per chain, unmapped residues,
the modifications emitted, and any chain whose modifications could not be placed.

This exists because of D-005. A conversion that quietly drops a chain produces a
successful fold with reasonable-looking metrics of a system the caller did not
describe, and nothing downstream is in a position to notice.
