#!/usr/bin/env bash
set -euo pipefail

checkpoint="${1:-checkpoints/test_model_G1_mjlab_Flat.pt}"

python -m orca_rl.run_play \
  --config Unitree-G1-Flat \
  --policy-backend mjlab \
  --checkpoint "$checkpoint"
