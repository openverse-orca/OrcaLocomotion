#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="${ROOT_DIR}/test_model_Go2_mjlab_Flat.pt"

MODE="${1:-walk}"

case "${MODE}" in
  walk)
    LIN_VEL_X=0.5
    LIN_VEL_Y=0.0
    ANG_VEL_Z=0.0
    ;;
  stand)
    LIN_VEL_X=0.0
    LIN_VEL_Y=0.0
    ANG_VEL_Z=0.0
    ;;
  spin)
    LIN_VEL_X=0.0
    LIN_VEL_Y=0.0
    ANG_VEL_Z=0.6
    ;;
  spin_cw)
    LIN_VEL_X=0.0
    LIN_VEL_Y=0.0
    ANG_VEL_Z=-0.6
    ;;
  -h|--help)
    cat <<EOF
Usage:
  ./play_go2_balance.sh [walk|stand|spin|spin_cw]

Examples:
  ./play_go2_balance.sh
  ./play_go2_balance.sh walk
  ./play_go2_balance.sh stand

Checkpoint:
  ./test_model_Go2_mjlab_Flat.pt
EOF
    exit 0
    ;;
  *)
    echo "Unsupported mode: ${MODE}. Expected one of: walk, stand, spin, spin_cw." >&2
    exit 2
    ;;
esac

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint file not found: ${CHECKPOINT}" >&2
  echo "Run: git lfs install && git lfs pull" >&2
  exit 1
fi

cd "${ROOT_DIR}"

python -m orca_rl.run_play \
  --config Unitree-Go2-Flat \
  --policy-backend mjlab \
  --checkpoint "${CHECKPOINT}" \
  --lin-vel-x "${LIN_VEL_X}" \
  --lin-vel-y "${LIN_VEL_Y}" \
  --ang-vel-z "${ANG_VEL_Z}"
