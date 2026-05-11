# TODO: Add Headless / No-Rendering Training Mode

Status: implemented.

## Background

Current training launches the simulator with rendering enabled. This makes training much slower, especially when running many parallel environments.

For RL training, we usually do not need visual rendering. Rendering should be optional and disabled by default during training.

## Goal

Add a headless / no-rendering mode to the training pipeline so that simulation can run faster without opening the viewer or rendering frames.

## Tasks

### 1. Add a command-line argument

Added training arguments:

```bash
--headless
```

Alias:

```bash
--no-render
```

Example usage:

```bash
python -m orca_rl.run_train \
  --config orca_rl/tasks/velocity/config/g1/env_cfgs.py \
  --headless
```

### 2. Pass the option into the simulator / environment

Done. The headless flag is forwarded from `run_train` to the environment factory, VecEnv adapter, and task backend.

Expected behavior:

- `--headless` enabled: no viewer/render calls, faster training.
- `--render` enabled: normal visual simulation for debugging.

### 3. Disable unnecessary rendering during training

Done. When running in headless mode, training disables:

- Viewer window
- Camera rendering
- Frame display
- Any visual debug rendering that slows down simulation

### 4. Keep rendering available for play / debugging

Done. `run_play` still uses human rendering so trained policies can be visually inspected.

### 5. Document usage in README

Done. README examples now include:

```bash
# Fast headless training
python -m orca_rl.run_train \
  --config orca_rl/tasks/velocity/config/g1/env_cfgs.py \
  --headless

# Visual debugging / playback
python -m orca_rl.run_play \
  --config orca_rl/tasks/velocity/config/g1/env_cfgs.py \
  --ckpt <path_to_checkpoint>
```

## Notes

This is important because training with the simulator window open is currently too slow. Headless mode should be the default choice for large-scale RL training.

Follow-up completed: G1 headless training now uses a generated local MJCF batch by default, so it can run without OrcaLab viewport rendering, without a gRPC server, and without publishing actors into the interactive scene.
