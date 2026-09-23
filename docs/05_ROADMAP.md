# 05 — Roadmap

Phases 1 and 2 are done; see [02_PARITY.md](02_PARITY.md) and
[03_FOUNDRY_INTEGRATION.md](03_FOUNDRY_INTEGRATION.md). This is what is left.

## Immediate

1. **Output parity on a GPU.** `scripts/parity_gpu.sbatch`. Feature parity makes
   this a confirmation, not a discovery, but it should be recorded.
2. **MSA parity.** The current fixtures fold in single-sequence mode. Pairing is
   driven purely by `key=<taxid>` in FASTA headers, so an adapter that builds
   MSAs without injecting those keys gets no cross-chain pairing — silently, and
   with no shape change to reveal it. Wire `LoadPolymerMSAs` into
   `pre_transforms` and add the case.
3. **A corpus survey.** `esmfold2-foundry parity <dir>/*.cif` over a few thousand
   PDB entries, to find which chain types and ligands the adapter cannot yet
   reproduce. The point is the failure list, not the pass rate.

## Phase 3 — generative surgery

The reason for all of the above. Today:

```
S -> ESMC(S) -> h -> FoldingTrunk(h) -> StructureDiffusion -> X
```

The plan is to expose those as separable modules and add a second path into the
trunk:

```
                  ESMC(sequence)  ──┐
                                    ├──> FoldingTrunk -> pair z -> StructureDiffusion
  noise z -> LatentGenerator      ──┘
```

`FoundryESMFold2` already exposes `.esmc`, `.folding_trunk` and
`.structure_head` as named seams, so opening one is a change of implementation
rather than of every call site.

**The first blocker is gradients, not architecture.** The release model's
`forward` is `@torch.inference_mode()`; Phase 3 starts by moving to
`ESMFold2ExperimentalModel`, whose `forward` is not decorated and which accepts
`res_type_soft` for soft-sequence design.

**The second is that the trunk is pair-only.** There is no single-representation
stream — `s_trunk=None` is passed to the structure head, and `d_single=384` is
declared but unused by the release trunk. Anything that expects an RF3-style
`(c_s, c_z)` pair of streams has to account for that; the comparable quantity is
`c_token = 768` inside the diffusion module, not `d_single`.

**The third is that pocket conditioning does not exist yet.** `PocketConditioning`
is in the input schema and round-trips through serialization, but
`prepare_esmfold2_input` never reads it — `pocket_feature` is
`torch.zeros(n_tokens)`, and upstream labels the block `# --- Pocket (dropped) ---`.
Binder work of the form *target FIXED / binder DIFFUSE* needs that path built,
not merely passed.

### The experiment this enables

With RFD3 and ESMFold2 both reachable from the same `AtomArray`, the comparison
stops being "two different architectures" and becomes a comparison of the
**representation prior**:

| arm | representation | generator |
|---|---|---|
| A | RFD3 | RFD3 |
| B | ESMC | same/similar |
| C | ESMFold2 folding trunk | same/similar |

which is the actual research question.

## Not planned

Rewriting ESMFold2 module by module in Foundry idiom. See D-001 and the "Non-goals"
section of [00_SCOPE.md](00_SCOPE.md). Components get separated when an
experiment needs them separated, one at a time, each behind a parity check.
