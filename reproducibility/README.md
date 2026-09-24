# The reference environment

The environment the published parity results were produced in — kept here so
they can be reproduced, not because the package needs it. To *use* the
package, `pip install -e .` in the repository root is enough; see
[docs/04](../docs/04_ENVIRONMENT.md).

| | |
|---|---|
| interpreter | CPython 3.14.6, linux x86_64 |
| torch | 2.14.0, CUDA 13.2 |
| esm, atomworks, foundry | source trees at the revisions in [`UPSTREAM.lock`](UPSTREAM.lock), on `PYTHONPATH` |
| everything else | [`pyproject.toml`](pyproject.toml) pins what the trees and the package import; [`uv.lock`](uv.lock) pins the rest — 177 packages, each at the version that environment had |

## Rebuilding it

```bash
# the upstreams, as siblings of this repository, at the revisions in UPSTREAM.lock
git clone https://github.com/Biohub/esm
git clone https://github.com/RosettaCommons/atomworks
git clone https://github.com/RosettaCommons/foundry      # only for the Foundry integration
#   then `git -C <tree> checkout <commit>` for each commit recorded there

uv sync --project reproducibility    # builds reproducibility/.venv from uv.lock
source reproducibility/env.sh        # puts the trees and ../src on PYTHONPATH
esmfold2-atomworks doctor            # trees, revisions against the lock, a real parity check
pytest -q                            # the full suite, Foundry contract tests included
```

`scripts/parity_gpu.sbatch` runs the GPU output-parity check in the same
environment.

## Why source trees rather than packages

The package declares `esm` and `atomworks` as ordinary dependencies, and a
resolver honours their own pins: torch 2.11 and biotite 1.4.0. This environment
is newer than those pins allow — esm 3.4 pins `torch>=2.11,<2.12`, and atomworks
pins `biotite==1.4.0`, which has no cp314 wheel. (Foundry's own pins conflict
with nothing here, but it depends on atomworks.) Installing the upstreams as
packages would pull the environment back, so they are consumed as source, and
`pyproject.toml` here lists the union of their runtime imports — plus the
package's own — instead of the upstreams themselves. When a tree grows a new
import, add it to the matching group; do not add the tree.

Two consequences of that choice:

- **`biotite` has to satisfy both sides.** biotite 1.7 moved
  `connect_via_residue_names` out of `biotite.structure.bonds`, which
  `atomworks.io.utils.io_utils` imports by that exact path; 1.6 keeps it and has
  cp314 wheels, so 1.6 is what runs here. Do not move past it without checking
  that import.
- **AtomWorks 2.2.1 has no upper bounds on its own dependencies** (`torch`,
  `numpy`, `rdkit`, `pandas`, `pyarrow`), so nothing from the tree constrains
  them; the pins here do.

## `DESIGN_ROOT` is discovered, never hardcoded

`env.sh` and the package's `paths.py` walk **up** from the repository until they
find a directory holding `esm/` and `atomworks/`. A fixed `REPO_ROOT.parent`
would be wrong from a git worktree, which sits several levels deeper. Set
`DESIGN_ROOT` explicitly when the trees live elsewhere, and `EF_VENV` to use an
existing environment instead of the one `uv sync` builds.

## `UPSTREAM.lock`

The trees come from `PYTHONPATH`, so nothing else records *which* revision
produced a result — and exact parity raises the bar for that well above an
ordinary wrapper's. `UPSTREAM.lock` records the commit and version of each tree.
It sits beside `pyproject.toml` and `uv.lock` because the three define the
environment together — the trees' revisions, and everything around them — and
`doctor` compares against it:

```
upstream revisions
  [ok  ] esm        43b4548b8676: matches lock (3.4.1.post1)
  [ok  ] atomworks  fcf7af8c127e: matches lock (2.2.1)
  [ok  ] foundry    b02eed6a6bdf: matches lock (0.1.0)
```

Drift is **reported, never enforced**: the adapter is deliberately
version-tolerant, so a newer tree is something to re-verify and re-pin, not
something to refuse. In an ordinary install `doctor` compares the installed
packages' versions instead.

## Not part of it

The optional accelerators `flash-attn` and `transformer-engine` were not
installed for the reference run; ESMFold2 runs without them via torch SDPA. Both
compile from source — cap the build (`MAX_JOBS=8`), because an uncapped
parallel compile will exhaust a shared machine — and the prebuilt `flash-attn`
cp314 wheel is ABI-incompatible with torch 2.14.
