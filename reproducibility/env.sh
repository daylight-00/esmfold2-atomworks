#!/usr/bin/env bash
# The reference environment: the one the published parity results were
# produced in. Not needed to use the package -- `pip install -e .` is.
#   uv sync --project reproducibility && source reproducibility/env.sh
#
# The upstreams are consumed as SOURCE trees at the revisions in UPSTREAM.lock,
# not as packages, because their package metadata would pull this environment
# back: esm pins torch<2.12, and atomworks pins biotite==1.4.0, which has no
# cp314 wheel (Foundry depends on atomworks, so installing it would bring that
# pin along). esm and atomworks are required; foundry is needed only by the
# optional Foundry integration. See reproducibility/README.md.

# The repository root, one level above this file.
EF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EF_ROOT

# ---------------------------------------------------------------- source trees
# DESIGN_ROOT is DISCOVERED, never hardcoded: walk up from this repo until a
# directory holds both required source trees. That keeps the same env.sh correct
# from the repo itself, from a git worktree (which sits several levels deeper),
# and after the tree is reorganized again -- which has already happened twice.
if [ -z "${DESIGN_ROOT:-}" ]; then
    _ef_dir="${EF_ROOT}"
    while [ "${_ef_dir}" != "/" ]; do
        if [ -d "${_ef_dir}/esm" ] && [ -d "${_ef_dir}/atomworks" ]; then
            DESIGN_ROOT="${_ef_dir}"
            break
        fi
        _ef_dir="$(dirname "${_ef_dir}")"
    done
    # Fall back to the parent directory, which is the layout when the trees are
    # staged but incomplete; `doctor` reports precisely what is missing.
    DESIGN_ROOT="${DESIGN_ROOT:-$(cd "${EF_ROOT}/.." && pwd)}"
    unset _ef_dir
fi
export DESIGN_ROOT

# Optional: only the Foundry integration imports it, and an absent entry is
# harmless.
export PYTHONPATH="${DESIGN_ROOT}/foundry/src:${PYTHONPATH:-}"
export PYTHONPATH="${DESIGN_ROOT}/atomworks/src:${PYTHONPATH:-}"
export PYTHONPATH="${DESIGN_ROOT}/esm:${PYTHONPATH:-}"
export PYTHONPATH="${EF_ROOT}/src:${PYTHONPATH:-}"

# ------------------------------------------------------------------- artifacts
export EF_MODELS="${EF_MODELS:-${DESIGN_ROOT}/biohub}"
export EF_CHECKPOINTS="${EF_CHECKPOINTS:-${DESIGN_ROOT}/checkpoints}"
export EF_RUNS="${EF_RUNS:-${EF_ROOT}/runs}"

# The CCD pickle ESMFold2 builds conformers from. esm captures this variable
# once, when esm.models.esmfold2.conformers is imported, and prefers it over any
# location a caller passes; the dictionary is then loaded once per process by
# whichever call comes first -- and esm's own lazy lookups pass no location,
# which downloads the Hub's latest revision into the Hugging Face cache.
# Exported here, before Python starts, it makes every call agree. See
# paths.ccd_dir().
if [ -z "${ESMCFOLD_CCD_PATH:-}" ] && [ -f "${EF_MODELS}/ESMFold2/ccd.pkl" ]; then
    export ESMCFOLD_CCD_PATH="${EF_MODELS}/ESMFold2/ccd.pkl"
fi

# ---------------------------------------------------------------- thread policy
# Featurization parity is many small CPU jobs; BLAS threads only oversubscribe.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# ---------------------------------------------------------------------- venv
# EF_VENV points at an existing environment; otherwise the one
# `uv sync --project reproducibility` builds is used when present. Set EF_VENV
# to share one environment across several projects that consume the same source
# trees. If neither exists and nothing is already active, the current
# interpreter is left alone -- `doctor` then reports which imports are missing.
if [ -n "${EF_VENV:-}" ] && [ -f "${EF_VENV}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${EF_VENV}/bin/activate"
elif [ -f "${EF_ROOT}/reproducibility/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${EF_ROOT}/reproducibility/.venv/bin/activate"
fi
