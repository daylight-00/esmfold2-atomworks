# The reference environment

The environment the suite and the GPU parity checks are run in — kept here so
they can be reproduced, not because the package needs it. To *use* the
package, `pip install -e .` in the repository root is enough; see
[docs/04](../docs/04_ENVIRONMENT.md).

| | |
|---|---|
| interpreter | CPython 3.14.6, linux x86_64 |
| torch | 2.14.0, CUDA 13.2 |
| atomworks | the package, `atomworks[ml]==3.0.0`, pinned in [`uv.lock`](uv.lock) |
| esm, foundry | source trees at the revisions in [`UPSTREAM.lock`](UPSTREAM.lock), on `PYTHONPATH` |
| model artifacts | the checkpoint, its ESMC backbone and the CCD pickle, by revision and digest, in [`ARTIFACTS.lock`](ARTIFACTS.lock) |
| everything else | [`pyproject.toml`](pyproject.toml) pins what the trees and the package import; [`uv.lock`](uv.lock) pins the rest — 180 packages, each at the version that environment had |

## Rebuilding it

```bash
# esm and foundry, as siblings of this repository, at the revisions in UPSTREAM.lock
git clone https://github.com/Biohub/esm
git clone https://github.com/RosettaCommons/foundry      # only for the Foundry integration
#   then `git -C <tree> checkout <commit>` for each commit recorded there

# optional: AtomWorks' test structures (the corpus survey, the two AF3-derived
# cases and the 1wym MSA case skip without them); any revision, the code comes
# from the package
git clone https://github.com/RosettaCommons/atomworks

uv sync --project reproducibility    # builds reproducibility/.venv from uv.lock
source reproducibility/env.sh        # puts esm, foundry and ../src on PYTHONPATH
python -m esmfold2_atomworks.doctor  # upstreams, revisions against the lock, a real parity check
pytest -q                            # the full suite, Foundry contract tests included
```

`scripts/parity_gpu.sbatch` runs the GPU output-parity check in the same
environment.

## Why esm and foundry are source trees, and atomworks is not

The package declares `esm` and `atomworks` as ordinary dependencies, and a
resolver honours their own pins. This environment is newer than esm's: esm 3.4
pins `torch>=2.11,<2.12`, and this one runs torch 2.14. rc-foundry supports
Python 3.12 only, and this one is 3.14. Installing them as packages would pull the
environment back, so they are consumed as source, and `pyproject.toml` here lists
the union of their runtime imports — plus the package's own — instead of the
upstreams themselves. When a tree grows a new import, add it to the matching
group; do not add the tree.

AtomWorks 3.0 asks for nothing that pulls the environment back — it pins
`biotite==1.6.0`, which has cp314 wheels — so it is installed, and its own
requirements are in the lock. Two consequences:

- **`biotite` is held by atomworks to 1.6.0.** Do not move it: AtomWorks patches
  biotite internals, and it refuses another version at import.
- **atomworks 3.0 raised a few floors**: `urllib3>=2.8` (the lock moved it from
  2.7.0), `pyarrow>=23.0.1`, and its `ml` extra brings `numba` and `llvmlite`,
  which the pipelines' MSA loader needs. `hydride`, which 2.2.1 required, is no
  longer in the environment.

## `DESIGN_ROOT` is discovered, never hardcoded

`env.sh` and the package's `paths.py` walk **up** from the repository until they
find a directory holding `esm/` and `atomworks/` checkouts. A fixed
`REPO_ROOT.parent` would be wrong from a nested checkout such as a git worktree.
Set `DESIGN_ROOT` explicitly when they live elsewhere, and `EF_VENV` to use an
existing environment instead of the one `uv sync` builds.

## `UPSTREAM.lock`

esm and foundry come from `PYTHONPATH`, so nothing else records *which* revision
produced a result — and exact parity raises the bar for that well above an
ordinary wrapper's. `UPSTREAM.lock` records the commit and version of each tree,
and the release of the atomworks package. It sits beside `pyproject.toml` and
`uv.lock` because the three define the environment together, and `doctor`
compares against it:

```
upstream revisions
  [ok  ] esm        43b4548b8676: matches lock (3.4.1.post1)
  [ok  ] atomworks  package 3.0.0: matches locked version (3.0.0)
  [ok  ] foundry    b02eed6a6bdf: matches lock (0.1.0)
```

Drift is **reported, never enforced**: the adapter is deliberately
version-tolerant, so a newer upstream is something to re-verify and re-pin, not
something to refuse. In an ordinary install `doctor` compares the installed
packages' versions instead.

## `ARTIFACTS.lock`

What the code read, as `UPSTREAM.lock` is the code. Feature parity needs only
the CCD pickle -- it passes two inputs through the same featurizer and involves
no weights; the GPU output-parity numbers need a checkpoint with the pinned
weights (the reference checkpoint, or the bundled revision recorded in
`checkpoint_equivalence.json`), the ESMC backbone for the separate layout, and
the CCD pickle. A Hub repository can be re-published under the same name, so
each entry names a revision and a digest per file.
[`scripts/compare_checkpoints.py`](../scripts/compare_checkpoints.py) compares a
checkpoint with the pinned one, and
[`checkpoint_equivalence.json`](checkpoint_equivalence.json) is its record for
the bundled `biohub/ESMFold2`.

## `output_parity.json`

The measurement behind the output-parity tables in
[docs/02](../docs/02_PARITY.md), written by
[`scripts/measure_output_parity.py`](../scripts/measure_output_parity.py): every
fixture and schedule folded in both execution modes, with the checkpoint (by
repo and revision), the device, the software stack and the commit it ran at.
The GPU suite gates the short schedule on every run; this record holds what is
too slow to gate -- the checkpoint's own schedule, the scatter, the wall time --
so that the numbers can be regenerated rather than taken on trust.

It was made in this environment, at commit `662802b` with AtomWorks 3.0.0 as a package,
and says so in its own `upstream_lock` field.

## Not part of it

The optional accelerators `flash-attn` and `transformer-engine` were not
installed for the reference run; ESMFold2 runs without them via torch SDPA. Both
compile from source — cap the build (`MAX_JOBS=8`), because an uncapped
parallel compile will exhaust a shared machine — and the prebuilt `flash-attn`
cp314 wheel is ABI-incompatible with torch 2.14.
