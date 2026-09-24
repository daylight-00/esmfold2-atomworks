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
| MSA | passed in per protein chain; binds only if its query row is the folded sequence | D-005 |

`ChainType` → ESMFold2 input class:

```
POLYPEPTIDE_L / POLYPEPTIDE_D / CYCLIC_PSEUDO_PEPTIDE  -> ProteinInput
DNA                                                     -> DNAInput
RNA                                                     -> RNAInput
NON_POLYMER / BRANCHED / MACROLIDE                      -> LigandInput
WATER                                                   -> dropped by policy (drop_water), and reported
anything else                                           -> raises UnsupportedChainError
no chain_type annotation                                -> raises InferredChainKindError
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

When the alignment cannot be established — no `res_name` in `chain_info`, and
modelled residues that do not line up with the sequence — a chain carrying a
non-standard residue **raises** `ModificationResolutionError`, because folding
the parent residue in its place is a different molecule. Placing a modification
by coincidence would be worse still, so accepting the approximation
(`allow_unplaceable_modifications=True`) emits **no** modifications and names the
chain in `AdapterReport.unplaceable_modifications`. A chain of standard residues
has nothing to place, so the same misalignment is not an error there.

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
- **Indices are read back from the tokenizer**, not re-derived — `data/bonds.py`
  says why.
- **An unplaceable bond raises.** Dropping it folds a connected system as though
  it were disconnected, and the direct `fold_atom_array` path returns no report,
  so nothing would tell the caller. `allow_unresolved_covalent_bonds=True` opts
  into that reading deliberately.

## Strictness: no silent semantic degradation

The adapter would rather stop than hand back a confident prediction of a
different system. `fold_atom_array` returns no report, so anything it merely
*recorded* would be invisible to the caller — which makes raising the only way
some facts arrive. The failure this guards against always has that shape: a
degradation noted in `AdapterReport` and nowhere else.

The degradations the adapter can detect are listed once, in
`atomworks_to_esm.DEGRADATIONS`, and each is accepted only by name:

| degradation | default | accept with |
|---|---|---|
| a chain no ESMFold2 input can express | **raises** `UnsupportedChainError` | `unsupported_chains` |
| a covalent bond that cannot be placed | **raises** `CovalentBondResolutionError` | `unresolved_covalent_bonds` |
| a chain kind that would be guessed (no `chain_type`) | **raises** `InferredChainKindError` | `inferred_chain_kind` |
| a non-standard residue whose position is unknown | **raises** `ModificationResolutionError` | `unplaceable_modifications` |

The same name works on every path, so no error names a remedy its caller
cannot reach:

```python
atom_array_to_structure_prediction_input(atoms, allow_inferred_chain_kind=True)
model.fold_atom_array(atoms, adapter_kwargs={"allow_inferred_chain_kind": True})
build_esmfold2_pipeline(is_inference=False, allow=["inferred_chain_kind"])
ESMFold2InferenceEngine(allow=["inferred_chain_kind"])
```
```bash
esmfold2-atomworks fold input.cif --allow inferred_chain_kind
```

A misspelt name raises rather than being ignored — an opt-in that silently did
nothing would leave a caller believing a run was permissive, or strict, when it
was not. `tests/test_strictness.py` checks the table in both directions: every
entry must be an adapter keyword that defaults to `False`, and every `allow_*`
keyword of the adapter must be an entry — bar `allow_undeclared_ccd_ligands`,
which is on by default because using a parsed CCD code is D-004's rule rather
than an approximation. A third check makes each entry say where its acceptance
is recorded.

Opting in says a degradation is acceptable, not which structure it hit. The
engine therefore records, per output, what actually happened under
`adapter.degradations` in the JSON beside each CIF, keyed by the same names;
direct callers pass an `AdapterReport` through `adapter_kwargs={"report": report}`
and read `report.accepted_degradations()`.

Two refusals have no opt-in, because there is nothing reasonable to proceed
with; and water is not a degradation but a policy with its own switch:

| situation | behaviour |
|---|---|
| ligand labelled `LIG`/`UNL`/`UNK`, or a name absent from the CCD | **raises** `LigandIdentityError`; declare a `LigandSpec` |
| a declaration that does not bind ([below](#declarations-must-bind)) | **raises** `ChainDeclarationError` |
| water | dropped under the explicit `drop_water` policy (`drop_water=False` refuses instead) |

**The boundary of the policy.** It acts on what the adapter can establish.
Without `chain_info`, the sequence comes from the modelled residues, so an
unresolved loop is simply absent (D-003). A gap in the numbering can hint at
that, but not reliably — numbering may skip legitimately — and it can never say
*what* is missing. So this case is recorded (`sequence_source == "atoms"`) and
documented rather than raised; supplying `chain_info` removes it entirely.

Dropping a chain and losing a bond are independent on purpose. Accepting that a
chain is dropped is not the same as accepting that a bond to it disappears, so
opting into the first still raises on the second.

Detection of bonds is deliberately kept separate from that policy:
`covalent_bond_candidates` returns a bond whose endpoint is not in the model
rather than filtering it out, so the decision belongs to the caller. An earlier
version filtered inside detection, which put such bonds beyond the strict check
entirely — an unsupported chain and a real bond to it could both vanish in
silence.

## Declarations must bind

`sequences`, `msas` and `ligands` are statements about particular chains, and
the conversion reads each only for chains of particular kinds. One that names
another chain — or none — would be passed over without a trace, and the fold
would come back as though it had been applied. So each is checked before
conversion starts, and one that does not bind raises `ChainDeclarationError`:

| declaration | binds only if | otherwise |
|---|---|---|
| `sequences[c]` | `c` is a protein, DNA or RNA chain, the override is not empty, and a protein override has no chain break (`:` or `\|`) | ignored in favour of the structure's own sequence; the chain omitted; or split by upstream into chains `c_0`, `c_1` |
| `msas[c]` | `c` is a protein chain, and the alignment's query row is the sequence folded there | ignored, leaving the protein in single-sequence mode; or clamped into place |
| `ligands[c]` | `c` is a non-polymer chain, a mapping key equals its spec's `chain_id`, and no other spec names `c` | never read; applied to one chain while describing another; or replaced, the last one winning |

The MSA rule is the one upstream cannot enforce for itself. `construct_paired_msa`
clamps each residue's column to the alignment's width, so an alignment built for
another sequence — the other chain of a heteromer, a construct with a different
tag, the parent of a design — is used column by column with no error, and its
first row contradicts the residues the model is given. To fold a design against
its parent's alignment on purpose, make the design the alignment's query row.

Keys are compared as strings, as the structure's own labels are, so a key `1`
binds to chain `"1"` instead of silently matching nothing. These are invalid
requests rather than approximations, so unlike `DEGRADATIONS` there is no
opt-in.

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
the modifications emitted, the covalent bonds found — and, for each degradation
the caller accepted by name, exactly what it hit.

It is the audit trail, not the safeguard (D-005). A conversion that quietly
drops a chain produces a successful fold with reasonable-looking metrics of a
system the caller did not describe, and nothing downstream is in a position to
notice. A report cannot prevent that — `fold_atom_array` does not even return
one — so the safeguard is the raise, and the report's degradation fields are
filled only once a degradation has been accepted. After a raise, the error
message carries the detail.
