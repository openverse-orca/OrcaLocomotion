#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
PYTHON_BIN="${ORCA_HEFT_PYTHON:-}"

if [[ -z "${PYTHON_BIN}" && -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
elif [[ -z "${PYTHON_BIN}" && -f "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
  conda activate "${ORCA_CONDA_ENV:-orcalab}"
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"

ORCA_HEFT_PYTHON="${PYTHON_BIN}" "${ROOT_DIR}/scripts/check_heft_install.sh" --runtime

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/third_party/unitree_rl_mjlab:${PYTHONPATH:-}"

exec "${PYTHON_BIN}" -m orca_rl.play_g1_heft_velocity \
  --policy "${ROOT_DIR}/checkpoints/heft/G1_PMG/policy.onnx" \
  --motion-dir "${ROOT_DIR}/assets/heft/recorded_commands" \
  "$@"
