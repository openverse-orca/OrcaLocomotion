#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"

if [[ -f "${CONDA_SH}" ]]; then
  # The orcalab env already has Python 3.12 and onnxruntime in this workspace.
  source "${CONDA_SH}"
  conda activate "${ORCA_CONDA_ENV:-orcalab}"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

python -m orca_rl.play_g1_twist2_keyboard "$@"
