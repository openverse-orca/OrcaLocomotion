# RSL-RL Locomotion Restructure Report

## Goal

After the G1 feasibility test passed, the locomotion RSL-RL integration was promoted out of `examples/` into the
top-level `orca_rl` package. The package now follows an mjlab/IsaacLab-style layout: task configs live with the task
family, while the low-level OrcaGym-to-RSL-RL conversion is isolated under an adapter package.

References used for the target shape:

- https://github.com/mujocolab/mjlab/tree/main/src/mjlab
- https://github.com/mujocolab/mjlab/tree/main/src/mjlab/tasks/velocity/config
- https://github.com/mujocolab/mjlab/tree/main/src/mjlab/tasks/velocity/config/go1

## New Layout

Task configs now live here:

```text
orca_rl/tasks/velocity/config/g1/env_cfgs.py
orca_rl/tasks/velocity/config/g1/rl_cfg.py
orca_rl/tasks/velocity/config/go2/env_cfgs.py
orca_rl/tasks/velocity/config/go2/rl_cfg.py
```

Adapter code now lives here:

```text
orca_rl/rsl_env/adapters/factory.py
orca_rl/rsl_env/adapters/vecenv.py
orca_rl/rsl_env/locomotion_task.py
```

The shared MDP-style config scaffolding now lives here:

```text
orca_rl/tasks/velocity/config_types.py
orca_rl/tasks/velocity/velocity_env_cfg.py
orca_rl/tasks/velocity/mdp/
```

Old YAML config shims, robot-named VecEnv adapters, and robot-named task wrappers were removed.

## Behavior

- `run_train`, `run_play`, and `run_eval` keep using the same loader path.
- Adapter package exports are lazy, so importing the factory does not load `rsl_rl` until an env is actually created.
- User-facing imports are now available from `orca_rl`, for example
  `from orca_rl import load_task_and_train_cfg, make_locomotion_vec_env`.
- Built-in velocity config factories can be imported from `orca_rl.tasks.velocity.config`.
- G1 and GO2 now share one config-driven `OrcaRslRlVecEnv` and one `OrcaLocomotionTask` runtime.
- Robot-specific scene binding is selected by `scene_binding.resolver` in each task config. Built-ins use short aliases:
  `g1` and `go2`. Dotted import paths are still supported for custom robots.
- Python config loading now supports `TASK_CONFIG_FACTORY` and `RL_CONFIG_FACTORY`.
- YAML locomotion config loading was removed from the active path; canonical configs are Python cfg files only.
- G1 still uses the discovered asset path:
  `assets/e071469a36d3c8aa/default_project/prefabs/g1_29dof_old_usda`.
- G1 still maps 29 actions; GO2 still maps 12 actions.
- Actor/critic observation grouping remains in each robot's `rl_cfg.py`.
- Rewards, observations, commands, events, and terminations are declared as MDP-style terms in Python cfg objects, then
  converted to the legacy dict shape consumed by the current Orca RSL-RL runtime.
- RSL-RL checkpoint alias saving and W&B CLI flags are unchanged.
- Train/play/eval now print an IsaacLab/MJLab-style terminal runtime summary before policy construction or inference:
  device/GPU, action and observation dimensions, rewards, terminations, commands, domain randomization, and scene binding.
- `run_train`, `run_play`, and `run_eval` default `--config` values now point at the canonical GO2 config path.
- `.orcalab/config.toml` now points the menu entries at the canonical task config paths.

## Validation

Ran these checks:

```bash
python - <<'PY'
from pathlib import Path
import ast
for path in Path('orca_rl').rglob('*.py'):
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
print('ast ok')
PY
```

```bash
/home/huan-hu/miniconda3/envs/orcalab/bin/python - <<'PY'
from orca_rl.utils import load_task_and_train_cfg
configs = [
    'orca_rl/tasks/velocity/config/g1/env_cfgs.py',
    'orca_rl/tasks/velocity/config/go2/env_cfgs.py',
]
for cfg in configs:
    task, train = load_task_and_train_cfg(cfg)
    print(cfg, task['robot'], task['name'], train['run_name'], len(task['control']['max_delta']))
PY
```

Observed:

```text
orca_rl/tasks/velocity/config/g1/env_cfgs.py g1 g1_flat_velocity g1_flat_velocity 29
orca_rl/tasks/velocity/config/go2/env_cfgs.py go2 go2_flat_velocity go2_flat_velocity 12
```

Also checked:

- `python -m orca_rl.run_train --help` in the OrcaLab conda env.
- `python -m orca_rl.run_play --help` in the OrcaLab conda env.
- `python -m orca_rl.run_eval --help` in the OrcaLab conda env.
- Importing `make_locomotion_vec_env` with the system Python does not eagerly require the RSL-RL runtime.
- `.orcalab/config.toml` parses and points RSL-RL menu entries at `orca_rl/tasks/velocity/config/{g1,go2}/env_cfgs.py`.
- `orca_rl.diagnostics.print_runtime_summary` was exercised with a fake env to verify terminal formatting without
  requiring a live OrcaLab server.
- Removed the obsolete `orca_rl/configs/` compatibility directory and old VecEnv import shims.
- Removed robot-named task wrappers. Adding another flat-velocity robot should be config-package work: define
  `env_cfgs.py`, `rl_cfg.py`, and either reuse a resolver alias or add a custom dotted resolver path when the asset
  naming differs.
- Removed the former examples-scoped package path. Runtime entrypoints are now `orca_rl.run_train`, `orca_rl.run_play`,
  and `orca_rl.run_eval`.

## Recommended Commands

G1 visual training smoke:

```bash
python -m orca_rl.run_train \
  --config orca_rl/tasks/velocity/config/g1/env_cfgs.py \
  --remote localhost:50051 \
  --num-iterations 1
```

G1 W&B training:

```bash
python -m orca_rl.run_train \
  --config orca_rl/tasks/velocity/config/g1/env_cfgs.py \
  --remote localhost:50051 \
  --logger wandb \
  --wandb-project orca_locomotion \
  --wandb-mode online
```
