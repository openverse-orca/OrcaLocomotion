# TODO: Add Headless / No-Rendering Training Mode

## Background

Current training launches the simulator with rendering enabled. This makes training much slower, especially when running many parallel environments.

For RL training, we usually do not need visual rendering. Rendering should be optional and disabled by default during training.

## Goal

Add a headless / no-rendering mode to the training pipeline so that simulation can run faster without opening the viewer or rendering frames.

## Tasks

### 1. Add a command-line argument

Add a training argument such as:

```bash
--headless
```

or:

```bash
--no-render
```

Example target usage:

```bash
python -m orca_rl.run_train \
  --config orca_rl/tasks/velocity/config/g1/env_cfgs.py \
  --headless
```

### 2. Pass the option into the simulator / environment

Make sure the headless flag is forwarded from `run_train` to the environment creation code and then to the OrcaGym / OrcaLab simulator backend.

Expected behavior:

- `--headless` enabled: no viewer, no rendering, faster training.
- `--headless` disabled: normal visual simulation for debugging.

### 3. Disable unnecessary rendering during training

When running in headless mode, disable:

- Viewer window
- Camera rendering
- Frame display
- Any visual debug rendering that slows down simulation

### 4. Keep rendering available for play / debugging

`run_play` or debug mode should still support rendering so trained policies can be visually inspected.

### 5. Document usage in README

Add examples to the README:

```bash
# Fast headless training
python -m orca_rl.run_train \
  --config orca_rl/tasks/velocity/config/g1/env_cfgs.py \
  --headless

# Visual debugging / playback
python -m orca_rl.run_play \
  --config orca_rl/tasks/velocity/config/g1/env_cfgs.py \
  --checkpoint <path_to_checkpoint>
```

## Notes

This is important because training with the simulator window open is currently too slow. Headless mode should be the default choice for large-scale RL training.
