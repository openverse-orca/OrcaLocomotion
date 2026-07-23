#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ORCA_HEFT_PYTHON:-${ROOT_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[HEFT check] Python was not found. Run ./install_heft.sh first." >&2
  exit 1
fi

"${PYTHON_BIN}" -c 'import sys; raise SystemExit("HEFT requires Python 3.12+; found " + sys.version.split()[0] if sys.version_info < (3, 12) else 0)'

cd "${ROOT_DIR}"
sha256sum --check --quiet assets/heft/SHA256SUMS

if [[ "${1:-}" == "--runtime" ]]; then
  "${PYTHON_BIN}" - <<'PY'
import importlib
import importlib.util

required = ("mujoco", "numpy", "onnxruntime", "orca_gym")
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if importlib.util.find_spec("pynput") is None:
    missing.append("pynput: package not found")
if missing:
    raise SystemExit("Missing or broken HEFT runtime dependencies:\n  " + "\n  ".join(missing))
PY
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/smoke_test_heft.py"
fi

echo "[HEFT check] Assets${1:+ and runtime dependencies} are ready."
