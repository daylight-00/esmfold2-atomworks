# 01 — The adapter

```
AtomWorks AtomArray  ──F──>  StructurePredictionInput  ──>  ESMFold2  ──>  MolecularComplexResult  ──G──>  AtomArray
```

`F` is `data/atomworks_to_esm.py`, `G` is `data/molecular_complex.py`. ESMFold2
itself is untouched.

## What `F` reads

| input | source | why |
|---|---|---|
| chain kind | `chain_type` annotation → `atomworks.enums.ChainType`; without one, a `chain_kinds` declaration, verified | D-002 |
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
no chain_type, declared in chain_kinds                  -> the declared kind, verified (below)
no chain_type, only is_polymer                          -> raises InferredChainKindError
no chain_type, no is_polymer                            -> raises UnsupportedChainError
```

A chain is classified as a whole, so every atom of it has to agree: a chain
whose atoms carry two `chain_type` values, or both `is_polymer` values, raises
`MixedChainError`; classified by its first atom, the rest would be folded as
part of that molecule.

## One chain, one molecule

ESMFold2 takes one input per chain, and everything above assumes that one label
of `chain_key` is one molecule. `atomworks.io.parse` makes it so: it gives the
polymer and non-polymer residues of an author chain chains of their own. An
author chain read any other way need not be one molecule. Read with author
fields — biotite's default — chain `A` of 101M is the protein, a heme, NBN, a
sulfate and 138 waters.

A built assembly has the same problem in another form: its copies repeat a
`chain_id` under each `transformation_id`, and one `chain_id` read as one chain
would fold a homodimer as a monomer. A chain label that spans several
`transformation_id` values raises `MixedChainError`; `chain_key="chain_iid"`
labels each copy.

Such a structure carries no `chain_type`, so it is refused until the caller says
what its chains are. `chain_kinds` is how:

```python
atom_array_to_structure_prediction_input(atoms, chain_kinds={"A": "protein", "B": "ligand"})
model.fold_atom_array(atoms, adapter_kwargs={"chain_kinds": {"A": "protein", "B": "ligand"}})
```

It is a declaration, not an override, and it is verified rather than trusted:

| the chain carries | the declaration is checked against |
|---|---|
| `chain_type` | `chain_type` alone: the parse is authoritative, and a contradiction raises |
| `is_polymer` only | `is_polymer`, then the CCD |
| neither | the CCD, residue by residue |

Against the CCD, each declared kind has to be true of every residue:

| declared | every residue must be |
|---|---|
| `protein` / `dna` / `rna` | of that polymer, by its CCD `_chem_comp.type` — modified and D-residues included |
| `ligand` | not water, and a polymer residue only when it is the chain's only residue (a free amino acid is a ligand; a run of them is a peptide) |
| `water` | water in AtomWorks' sense, `HOH` or `DOD`, and then `drop_water` decides, as for a parsed water chain |

A residue that does not fit raises `ChainDeclarationError`, listing it with
what the CCD calls it. That covers the case this exists for: declared `protein`,
101M's author chain A would otherwise fold every water and the heme as residues
of the protein — and under a sequence override, leave them out
of the model input with nothing dropped and nothing raised. The CCD is only ever
used to refuse; it never supplies a kind, so this is verification and not the
residue-name guess the classification exists to avoid. It cannot tell which of
two things went wrong, and the error says so: the residues may be separate
molecules sharing a label, or the chain's own residues under names the CCD uses
for something else — Amber's `HIE` and `CYX` are unrelated small molecules
there. Nor can it catch a name the CCD files under the same polymer: Amber's
`HIP` is phosphonohistidine in the CCD, passes, and is folded as that
modification. Residue names have to be CCD codes.

The fix for a chain that holds more than one molecule is a label per molecule:
parse with `atomworks.io.parse`, read an mmCIF with
`pdbx.get_structure(..., use_author_fields=False)`, or build an annotation that
separates them and pass it as `chain_key=`. `unsupported` cannot be declared —
it names the absence of anything to classify by, not something a chain is.

## Modification positions are verified, not assumed

`Modification.position` indexes the sequence being folded. That only means
anything if the residue list and the sequence line up, so the adapter uses
`chain_info[c]["res_name"]`, which does.

The alignment is guaranteed **by construction**, not merely observed: AtomWorks
builds the canonical sequence by mapping one-letter codes over that same
`res_name` list, one character per entry, with `"X"`/`"N"`/`"-"` fallbacks on
every path. So `len(sequence) == len(res_name)` holds for any polymer chain.
For an mmCIF with an `entity_poly_seq` record, `res_name` comes from it — the
full deposited sequence, including residues that were never resolved — which is
also why it, and not the observed residues, is the right thing to fold (D-003).
For other input (a PDB-format file, an mmCIF without the record) `chain_info` is
inferred from the modelled residues and the alignment holds in the same way.

Checked concretely on `1a8o_modified`: 70 residue names, 70 sequence characters
(`tests/test_gold_fixtures.py`), and the four `MSE` entries at indices 0, 34, 63,
64 — where the canonical sequence reads `M`. `tests/test_atomworks_to_esm.py`
asserts those positions and the `sequence[position] == "M"` relation, so a change
that shifts the alignment fails loudly rather than folding four wrong residues.

The lookup is made for polymer chains only.

When the alignment cannot be established — no `res_name` in `chain_info`, and
modelled residues that do not line up with the sequence — a chain carrying a
non-standard residue **raises** `ModificationResolutionError`, because folding
the parent residue in its place is a different molecule. Placing a modification
by coincidence would be worse still, so accepting the approximation
(`allow_unplaceable_modifications=True`) emits **no** modifications and names the
chain in `AdapterReport.unplaceable_modifications`. A chain of standard residues
has nothing to place, so the same misalignment is not an error there.

An overridden sequence drops the structure's modifications entirely — it is a
different molecule, and the structure's positions do not apply to it.

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
  says why. They come from a tokenization in which the bonded chains already
  count as bonded, because the tokenizer drops a CCD ligand's leaving atoms (NAG's
  `O1`) once its chain takes part in a bond, which moves every later atom up one
  place.
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

Three refusals have no opt-in, because there is nothing reasonable to proceed
with; and water is not a degradation but a policy with its own switch:

| situation | behaviour |
|---|---|
| ligand labelled `LIG`/`UNL`/`UNK`/`UNX`, or a name absent from the CCD | **raises** `LigandIdentityError`; declare a `LigandSpec` |
| a declaration that does not bind ([below](#declarations-must-bind)) | **raises** `ChainDeclarationError` |
| a chain whose atoms carry more than one `chain_type` or `is_polymer` value | **raises** `MixedChainError`; give each molecule its own chain ([above](#one-chain-one-molecule)) |
| water | dropped under the explicit `drop_water` policy (`drop_water=False` refuses instead) |

**The boundary of the policy.** It acts on what the adapter can establish.
Without `chain_info`, the sequence comes from the modelled residues, so an
unresolved loop is simply absent (D-003). A gap in the numbering can hint at
that, but not reliably — numbering may skip legitimately — and it can never say
*what* is missing. So this case is recorded (`sequence_source == "atoms"`) and
documented rather than raised; supplying a `chain_info` that records the full
entity sequence (mmCIF `entity_poly_seq`) removes it.

Dropping a chain and losing a bond are independent on purpose. Accepting that a
chain is dropped is not the same as accepting that a bond to it disappears, so
opting into the first still raises on the second.

Detection of bonds is deliberately kept separate from that policy:
`covalent_bond_candidates` returns a bond whose endpoint is not in the model
rather than filtering it out, so the decision belongs to the caller.

## Declarations must bind

`chain_kinds`, `sequences`, `msas` and `ligands` are statements about
particular chains, and the conversion reads each only for chains of particular
kinds. One that names another chain — or none — would be passed over without a
trace, and the fold would come back as though it had been applied. So each is
checked before conversion starts, and one that does not bind raises
`ChainDeclarationError`:

| declaration | binds only if | otherwise |
|---|---|---|
| `chain_kinds[c]` | `c` is a chain, the kind is `protein`, `dna`, `rna`, `ligand` or `water`, and neither the chain's annotations nor its residues contradict it ([above](#one-chain-one-molecule)) | ignored; overriding what the parse recorded; or folding what else shares the label as part of the declared molecule |
| `sequences[c]` | `c` is a protein, DNA or RNA chain, the override is not empty, and a protein override has no chain break (`:` or `\|`) | ignored in favour of the structure's own sequence; the chain omitted; or split by upstream into chains `c_0`, `c_1` |
| `msas[c]` | `c` is a protein chain, and the alignment's query row is the sequence folded there | ignored, leaving the protein in single-sequence mode; or clamped into place |
| `ligands[c]` | `c` is a non-polymer chain, a mapping key equals its spec's `chain_id`, and no other spec names `c` | never read; applied to one chain while describing another; or replaced, the last one winning |

The MSA rule is the one upstream cannot enforce for itself. `construct_paired_msa`
clamps each residue's column to the alignment's width, so an alignment built for
another sequence — the other chain of a heteromer, a construct with a different
tag, the wild type of a variant — is used column by column with no error, and
its first row contradicts the residues the model is given. To fold a variant
against another sequence's alignment on purpose, make the variant the
alignment's query row.

Alignments found by AtomWorks' `LoadPolymerMSAs` reach the same check through
`PolymerMSAsToESMFold2`, which `build_esmfold2_pipeline(msa_loader=...)` places
after the loader: rows, insertion counts and TaxIDs become an `MSA` whose
headers carry upstream's `key=<taxid>`, so heteromer rows pair. ESMFold2 takes
no RNA alignment, so the loader's RNA alignments are left out and named in
`data["msas_left_out"]`.

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
label a model wrote on its own output. The check looks the name up in esm's CCD
dictionary; a dictionary that cannot be loaded raises instead of accepting every
name.

`LigandSpec.verify_against` compares **heavy atoms only**. Whether hydrogens are
present at all depends on the source and the parser settings: `parse` keeps
them where the source has them and, in AtomWorks 2.x, can add them
(`hydrogen_policy="infer"`; 3.x protonates in a separate step), Rosetta rebuilds
them on load, ESMFold2 reports none. Heavy atoms are the only
comparison that means the same thing on every side.

## Why `G` does not go through mmCIF

The obvious return route is `MolecularComplex.to_mmcif()` → `atomworks.io.parse()`.
It is in memory and hands back AtomWorks' full annotation set for free.

It is also **silently lossy for ligands**, for exactly the D-004 reason: ESMFold2
labels a SMILES ligand `LIG`, `LIG` is a real CCD code for an unrelated molecule,
and `parse` reconciles against that component, keeping only the atoms whose names
happen to match — and the structure still parses and still validates.

So `G` reads `MolecularComplex`'s flat per-atom arrays directly. It is faithful
because nothing is inferred that the model did not report, and it raises rather
than build anything if the token spans do not partition the atoms: an atom
claimed by no span would vanish, one claimed by two would be handed silently to
the later token, and a span claiming none would drop a residue that
`result.complex.plddt` still counts. The polymer/ligand split is the model's own
hetero flags, never a CCD lookup.

Ligand names are restored per chain. ESMFold2 writes `LIG` on a ligand it was
given as SMILES, and anything that matches ligands by name needs the caller's
name instead. `fold_atom_array` gives each chain declared with a `LigandSpec`
that spec's `residue_name` (`apply_ligand_labels`), so a ligand, a cofactor and
an ion beside it keep their own names. A declaration of several CCD components
has no single name — they are several residues — so `residue_name` is refused
there and the chain is left as the model wrote it (see below).

`ligand_residue_name` — on `result_to_atom_array`, and on `fold_atom_array`,
which passes it through — is the blanket form: **every** hetero residue gets
the caller's name. ESMFold2 marks each non-polymer chain hetero and nothing
else, so an ion or cofactor beside the ligand is renamed with it; the option is
for an output whose hetero residues are all one molecule. Given together with
`LigandSpec`s, it is accepted only if it restates every declared chain's label
— a spec's default label counts, and a multi-component declaration has none —
and refused rather than overriding a declaration otherwise. A name longer than a residue name's five
characters raises rather than being truncated.

`chain_type` is not something `G` can write: the model's output knows polymer
from non-polymer, not an L- from a D-polypeptide or DNA from RNA, so
`result_to_atom_array` leaves it out rather than guess. `fold_atom_array` has
the source, and copies each chain's `chain_type` back onto the chain the
output names the same (`copy_chain_types`, matched under the adapter's
`chain_key`), so a round trip keeps what the source said and the output can be
folded again without a declaration. Only a source that carries `chain_type`
has one to copy: a kind known from a `chain_kinds` declaration is not turned
into a `ChainType`, which would state more than the declaration did. Residue
numbering is a separate matter -- the output numbers each chain from 1, and
the folded sequence can include residues the source never resolved, so the
source's `res_id` and insertion codes do not map back one to one.

### What `G` does not return

The model's output is the predicted geometry and atom-level labels —
coordinates, atom names, elements, residue names, chains, polymer/non-polymer —
and not the molecule's topology. Two things a source structure can carry do not
come back from it:

- **Bonds.** `MolecularComplex` records none, so `result_to_atom_array` returns
  an `AtomArray` with no `BondList`, and nothing the model reports would support
  one. `fold_atom_array` adds one rebuilt from what the model was given
  ([below](#the-bond-list-of-the-returned-structure)).
- **Components of a multi-component ligand.** A non-polymer chain declared as
  several CCD components — a glycan, or `7ubd`'s cyclic D-peptide as one
  eight-component ligand — is tokenized component by component, but ESMFold2's
  output builder returns each non-polymer chain as one residue named by its
  first component. The atoms are all there; the residue boundaries between
  them are not. That is upstream behaviour, recorded here so a round trip is not
  read as preserving them.

### The bond list of the returned structure

`fold_atom_array(bonds=True)`, the default, sets `structure.bonds`, built by
`data/topology.py` from the chemistry the model was given and from nothing it
reported:

- inside a residue or CCD ligand component, the CCD's bonds with the types its
  file gives them (`value_order` and the aromatic flag, so
  `AROMATIC_SINGLE`/`AROMATIC_DOUBLE` in a ring), among the atoms present;
- inside a SMILES ligand, RDKit's bonds (aromatic bonds as `AROMATIC`), on the
  atom names ESMFold2 gives it;
- between consecutive residues of a polymer, `C`–`N` (protein) or `O3'`–`P`
  (nucleic acid);
- the covalent bonds that were declared, with the type the source had.

A list that looks complete and is not would be worse than none, so every atom the
chemistry names has to be found under that name, every atom of the structure has
to belong to a chain of the input, and a ligand's components have to account for
its atoms — including the leaving atoms the tokenizer drops from a bonded chain.
Otherwise `TopologyError` is raised; `bonds=False` returns the structure without
a bond list. For a standard residue the result equals biotite's own assignment
by residue name (`connect_via_residue_names`), bond for bond and type for type
(`tests/test_topology.py`); AtomWorks' parse agrees on every connection and may
alternate an aromatic ring's single and double bonds differently.

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
