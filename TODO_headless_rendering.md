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
  --config Unitree-G1-Flat \
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
  --config Unitree-G1-Flat \
  --headless

# Visual debugging / playback
python -m orca_rl.run_play \
  --config Unitree-G1-Flat \
  --ckpt <path_to_checkpoint>
```

## Notes

This is important because training with the simulator window open is currently too slow. Headless mode should be the default choice for large-scale RL training.

Follow-up completed:

- G1 headless training now uses a generated local MJCF batch by default, so it can run without OrcaLab viewport rendering, without a gRPC server, and without publishing actors into the interactive scene.
- G1 rough local MJCF training now inserts the generated rough heightfield into MuJoCo as a physical `hfield` collision geom.
- Standard registered task names are available through `--config Unitree-G1-Flat`, `Unitree-G1-Rough`, `Unitree-GO2-Flat`, and `Unitree-GO2-Rough`.
- `run_play` defaults back to the OrcaLab scene path for visual inspection. For G1 it disables local MJCF, uses the scene binding resolver, and can auto-publish the configured G1 asset if the scene is missing one.
- `run_play --local-mujoco` keeps the generated local MuJoCo playback path available for debugging without OrcaLab scene binding.
