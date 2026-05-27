#!/usr/bin/env bash
set -euo pipefail

checkpoint="${1:-checkpoints/test_model_Go2_mjlab_Rough.pt}"

python -m orca_rl.run_play \
  --config Unitree-Go2-Rough \
  --policy-backend mjlab \
  --checkpoint "$checkpoint"
