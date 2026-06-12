#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec python -m orca_rl.run_play \
  --config DeepRobotics-Lite3-Flat \
  --mjlab \
  --checkpoint checkpoints/model_3100_Lite3.pt \
  "$@"
