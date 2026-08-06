#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
PYTHON_BIN="${ORCA_HEFT_PYTHON:-}"

if [[ -z "${PYTHON_BIN}" && -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
elif [[ -z "${PYTHON_BIN}" && -f "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
  conda activate "${ORCA_CONDA_ENV:-orcalab}"
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"

if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

exec "${PYTHON_BIN}" "${ROOT_DIR}/scripts/check_pico_xrobot_stream.py" "$@"
