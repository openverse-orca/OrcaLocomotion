#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ORCA_HEFT_VENV:-${ROOT_DIR}/.venv}"

find_python() {
  if [[ -n "${ORCA_HEFT_PYTHON:-}" ]]; then
    printf '%s\n' "${ORCA_HEFT_PYTHON}"
    return
  fi
  local candidate
  for candidate in python3.13 python3.12 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      if "${candidate}" -c 'import sys; raise SystemExit(sys.version_info < (3, 12))'; then
        command -v "${candidate}"
        return
      fi
    fi
  done
  return 1
}

cd "${ROOT_DIR}"

if command -v git-lfs >/dev/null 2>&1; then
  git lfs pull --include='checkpoints/heft/**,assets/heft/**'
else
  echo "[HEFT install] git-lfs is required. Install Git LFS, then rerun this script." >&2
  exit 1
fi

if ! PYTHON_BIN="$(find_python)"; then
  echo "[HEFT install] Python 3.12+ was not found." >&2
  echo "Set ORCA_HEFT_PYTHON=/path/to/python3.12 and rerun." >&2
  exit 1
fi

echo "[HEFT install] Creating environment: ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install -r requirements.txt
"${VENV_DIR}/bin/python" -m pip install --no-deps -e .

ORCA_HEFT_PYTHON="${VENV_DIR}/bin/python" "${ROOT_DIR}/scripts/check_heft_install.sh" --runtime

echo
echo "[HEFT install] Ready. Start OrcaLab, load the G1 + Dex3-1 scene, then run:"
echo "  ./play_g1_heft.sh"
