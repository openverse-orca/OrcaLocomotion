# Orca RL

Standalone RSL-RL bridge package for OrcaLab / OrcaGym.

Current targets:

- Unitree GO2
- Unitree G1
- flat velocity tracking
- asymmetric actor-critic with privileged observations
- bounded residual joint-target actions

Install the runtime dependencies inside the OrcaLab environment:

```bash
pip install -r requirements.txt
```

Before launching GO2, place exactly one GO2 actor in the OrcaLab scene. The scene binding requires the GO2 joints,
actuators, contact sites, foot bodies, and touch sensors to match the asset suffixes used by
`envs.legged_gym.robot_config.go2_config`.

The canonical task configs follow an mjlab/IsaacLab-style Python layout:

- `tasks/velocity/config/g1/env_cfgs.py`
- `tasks/velocity/config/g1/rl_cfg.py`
- `tasks/velocity/config/go2/env_cfgs.py`
- `tasks/velocity/config/go2/rl_cfg.py`

The Orca runtime side is robot-neutral: `rsl_env/adapters/vecenv.py` creates the RSL-RL VecEnv, and
`rsl_env/locomotion_task.py` runs one bound robot instance. Robot-specific asset discovery is selected through each
config's local `resolve_scene_binding` function.

For G1, use `orca_rl/tasks/velocity/config/g1/env_cfgs.py`. It scans the existing G1 scene first; if no complete
G1 is found, it tries to publish `g1_000` from:
`assets/e071469a36d3c8aa/default_project/prefabs/g1_29dof_old_usda`.

Train GO2:

```bash
python -m orca_rl.run_train \
  --config orca_rl/tasks/velocity/config/go2/env_cfgs.py
```

Train G1:

```bash
python -m orca_rl.run_train \
  --config orca_rl/tasks/velocity/config/g1/env_cfgs.py
```

Train G1 with W&B logging:

```bash
python -m orca_rl.run_train \
  --config orca_rl/tasks/velocity/config/g1/env_cfgs.py \
  --logger wandb \
  --wandb-project orca_locomotion \
  --wandb-mode online
```

The training entrypoint always writes stable checkpoint aliases after `runner.learn()`:

- `model_last.pt`
- `model_final.pt`

If OrcaGym is not listening on `localhost:50051`, override the address:

```bash
python -m orca_rl.run_train \
  --config orca_rl/tasks/velocity/config/g1/env_cfgs.py \
  --remote 192.168.1.10:50051
```

Play a checkpoint:

```bash
python -m orca_rl.run_play \
  --config orca_rl/tasks/velocity/config/g1/env_cfgs.py \
  --ckpt trained_models_tmp/rsl_rl_locomotion/<run>/model_<iter>.pt
```

Evaluate:

```bash
python -m orca_rl.run_eval \
  --config orca_rl/tasks/velocity/config/g1/env_cfgs.py \
  --ckpt trained_models_tmp/rsl_rl_locomotion/<run>/model_<iter>.pt
```
