#!/usr/bin/env bash
#SBATCH --job-name=ef2-parity
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=runs/parity_gpu_%j.out
#
# Output parity: one real fold through each path, compared.
#
# Feature parity (CPU, exact, no weights) is the primary check and does not
# need this -- see docs/02_PARITY.md. This confirms it on a sampled structure.
#
# Submit from the repository root, adding whatever your site requires:
#   sbatch --partition=<gpu-partition> scripts/parity_gpu.sbatch
#
# Set EF_VENV if the environment is not this repo's own .venv, and
# EF_ALLOW_DOWNLOAD=1 if there is no local weight mirror. Both propagate
# through sbatch's default --export=ALL.

set -euo pipefail

# Slurm copies the batch script into a spool directory before running it, so
# ${BASH_SOURCE[0]} points there and not into the repository. SLURM_SUBMIT_DIR
# is the directory the job was submitted from; fall back to the script's own
# location so this still works when run directly as a shell script.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO}"

if [ ! -f env.sh ]; then
    echo "expected to run from the repository root; got ${REPO}" >&2
    exit 2
fi

# shellcheck disable=SC1091
source env.sh

mkdir -p runs
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || true
python -m pytest tests/test_output_parity_gpu.py -m gpu -v -s
