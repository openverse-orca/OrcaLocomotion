#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"

if [[ -f "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
  conda activate "${ORCA_CONDA_ENV:-orcalab}"
fi

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/third_party/unitree_rl_mjlab:${PYTHONPATH:-}"

exec python -m orca_rl.play_g1_mjlab_keyboard \
  --checkpoint "${ROOT_DIR}/checkpoints/test_model_G1_mjlab_Flat.pt" \
  "$@"
