#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${XROBOT_BUILD_ROOT:-${TMPDIR:-/tmp}/orca-xrobottoolkit-sdk}"
PYBIND_DIR="${BUILD_ROOT}/XRoboToolkit-PC-Service-Pybind"
SERVICE_DIR="${BUILD_ROOT}/XRoboToolkit-PC-Service"

if [[ -n "${ORCA_HEFT_PYTHON:-}" ]]; then
  PYTHON_BIN="${ORCA_HEFT_PYTHON}"
elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

for command in git cmake g++ pkg-config protoc grpc_cpp_plugin; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "[PICO SDK] Missing required build command: ${command}" >&2
    exit 1
  fi
done
if ! pkg-config --exists protobuf || ! pkg-config --exists grpc++; then
  cat >&2 <<'EOF'
[PICO SDK] Missing C++ Protobuf/gRPC development packages.

On a Conda-based OrcaLab environment, install them with:
  conda install -n orcalab -c conda-forge grpc-cpp protobuf pkg-config pybind11

On a system Python environment, install the equivalent Protobuf/gRPC development packages.
EOF
  exit 1
fi

if ! "${PYTHON_BIN}" -c 'import pybind11, setuptools, wheel' >/dev/null 2>&1; then
  echo "[PICO SDK] ${PYTHON_BIN} needs pybind11, setuptools, and wheel." >&2
  exit 1
fi

mkdir -p "${BUILD_ROOT}"
if [[ ! -d "${PYBIND_DIR}/.git" ]]; then
  git clone https://github.com/Axellwppr/XRoboToolkit-PC-Service-Pybind "${PYBIND_DIR}"
fi
if [[ ! -d "${SERVICE_DIR}/.git" ]]; then
  git clone --branch main --single-branch https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git "${SERVICE_DIR}"
fi

case "$(uname -m)" in
  x86_64|amd64)
    PROTO_SUBDIR="linux_x86"
    LINUX_DEFINE="LINUX_x86"
    ;;
  aarch64|arm64)
    PROTO_SUBDIR="linux_aarch64"
    LINUX_DEFINE="LINUX_aarch64"
    ;;
  *)
    echo "[PICO SDK] Unsupported architecture: $(uname -m)" >&2
    exit 1
    ;;
esac

PROTO_DIR="${SERVICE_DIR}/RoboticsService/PXREAService/${PROTO_SUBDIR}"
SDK_SRC_DIR="${SERVICE_DIR}/RoboticsService/PXREARobotSDK"
SDK_OUTPUT_DIR="${BUILD_ROOT}/out"
SDK_LIBRARY="${SDK_OUTPUT_DIR}/libPXREARobotSDK.so"

cd "${PROTO_DIR}"
protoc \
  -I "${PROTO_DIR}" \
  -I /usr/include \
  --plugin=protoc-gen-grpc="$(command -v grpc_cpp_plugin)" \
  --grpc_out="${PROTO_DIR}" \
  --cpp_out="${PROTO_DIR}" \
  PXREAService.proto

cmake \
  -S "${ROOT_DIR}/scripts/xrobottoolkit_sdk" \
  -B "${BUILD_ROOT}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPXREA_SDK_SRC_DIR="${SDK_SRC_DIR}" \
  -DPXREA_PROTO_DIR="${PROTO_DIR}" \
  -DPXREA_LINUX_DEFINE="${LINUX_DEFINE}" \
  -DPXREA_OUTPUT_DIR="${SDK_OUTPUT_DIR}"
cmake --build "${BUILD_ROOT}/build" --target PXREARobotSDK -j2

PXREA_SDK_LIBRARY="${SDK_LIBRARY}" "${PYTHON_BIN}" -m pip install --no-build-isolation --force-reinstall "${PYBIND_DIR}"

SDK_LIBRARY_DIR="$(dirname "${SDK_LIBRARY}")"
LD_LIBRARY_PATH="${CONDA_PREFIX:+${CONDA_PREFIX}/lib:}${SDK_LIBRARY_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "${PYTHON_BIN}" - <<'PY'
import xrobotoolkit_sdk as xrt

required = ("init", "register_frame_callback", "clear_frame_callback", "has_frame_callback")
missing = [name for name in required if not hasattr(xrt, name)]
if missing:
    raise RuntimeError(f"xrobotoolkit_sdk missing callback APIs: {missing}")
print("[PICO SDK] installed:", xrt.__file__)
PY
