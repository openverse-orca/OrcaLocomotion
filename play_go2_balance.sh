#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="${ROOT_DIR}/checkpoints/test_model_Go2_mjlab_Flat.pt"
BACKEND="orcalab"
VIEWER="auto"

MODE="${1:-walk}"
if [[ "$#" -gt 0 ]]; then
  MODE=""
fi

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --mjlab)
      BACKEND="mjlab"
      shift
      ;;
    --orcalab)
      BACKEND="orcalab"
      shift
      ;;
    --viewer)
      if [[ "$#" -lt 2 ]]; then
        echo "--viewer requires one of: auto, native, viser" >&2
        exit 2
      fi
      VIEWER="$2"
      shift 2
      ;;
    -h|--help)
      MODE="--help"
      shift
      ;;
    walk|stand|spin|spin_cw)
      if [[ -n "${MODE}" ]]; then
        echo "Only one mode can be provided. Got both '${MODE}' and '$1'." >&2
        exit 2
      fi
      MODE="$1"
      shift
      ;;
    *)
      echo "Unsupported argument: $1" >&2
      exit 2
      ;;
  esac
done

MODE="${MODE:-walk}"

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
  ./play_go2_balance.sh [walk|stand|spin|spin_cw] [--orcalab|--mjlab] [--viewer auto|native|viser]

Examples:
  # OrcaLab play, default backend.
  ./play_go2_balance.sh
  ./play_go2_balance.sh walk
  ./play_go2_balance.sh stand

  # Native mjlab replay for comparison.
  ./play_go2_balance.sh walk --mjlab
  ./play_go2_balance.sh spin --mjlab --viewer native
  ./play_go2_balance.sh stand --mjlab --viewer viser

Checkpoint:
  ./checkpoints/test_model_Go2_mjlab_Flat.pt
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

if [[ "${BACKEND}" == "orcalab" ]]; then
  python -m orca_rl.run_play \
    --config Unitree-Go2-Flat \
    --policy-backend mjlab \
    --checkpoint "${CHECKPOINT}" \
    --lin-vel-x "${LIN_VEL_X}" \
    --lin-vel-y "${LIN_VEL_Y}" \
    --ang-vel-z "${ANG_VEL_Z}"
  exit 0
fi

PYTHONPATH="${ROOT_DIR}/third_party/unitree_rl_mjlab:${PYTHONPATH:-}" \
python - "${CHECKPOINT}" "${MODE}" "${LIN_VEL_X}" "${LIN_VEL_Y}" "${ANG_VEL_Z}" "${VIEWER}" <<'PY'
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


def resolve_viewer(viewer: str) -> str:
    if viewer != "auto":
        return viewer
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return "native" if has_display else "viser"


checkpoint = Path(sys.argv[1]).expanduser().resolve()
mode = sys.argv[2]
command = tuple(float(value) for value in sys.argv[3:6])
viewer = resolve_viewer(sys.argv[6])

task_id = "Unitree-Go2-Flat"
configure_torch_backends()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

env_cfg = load_env_cfg(task_id, play=True)
agent_cfg = load_rl_cfg(task_id)
env_cfg.scene.num_envs = 1

twist_cmd = env_cfg.commands["twist"]
twist_cmd.resampling_time_range = (1.0e9, 1.0e9)
twist_cmd.heading_command = False
twist_cmd.rel_heading_envs = 0.0
twist_cmd.rel_standing_envs = 0.0
twist_cmd.ranges.heading = None
twist_cmd.ranges.lin_vel_x = (command[0], command[0])
twist_cmd.ranges.lin_vel_y = (command[1], command[1])
twist_cmd.ranges.ang_vel_z = (command[2], command[2])

env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
runner = runner_cls(env, asdict(agent_cfg), device=device)
runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

print(f"[INFO] Backend: mjlab")
print(f"[INFO] Task: {task_id}")
print(f"[INFO] Mode: {mode}")
print(f"[INFO] Command: lin_x={command[0]:.2f}, lin_y={command[1]:.2f}, yaw={command[2]:.2f}")
print(f"[INFO] Viewer: {viewer}")
print(f"[INFO] Checkpoint: {checkpoint}")

if viewer == "native":
    NativeMujocoViewer(env, policy).run()
elif viewer == "viser":
    ViserPlayViewer(env, policy).run()
else:
    raise RuntimeError(f"Unsupported viewer backend: {viewer}")

env.close()
PY
