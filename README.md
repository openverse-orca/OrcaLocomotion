# Orca RL

Standalone RSL-RL bridge package for OrcaLab / OrcaGym.

Current targets:

- Unitree GO2
- Unitree G1
- flat and rough velocity tracking
- asymmetric actor-critic with privileged observations
- bounded residual joint-target actions

Install the runtime dependencies inside the OrcaLab environment:

```bash
pip install -r requirements.txt
```

When working from the original OrcaPlayground tree instead of the standalone `orca_rl` repository, use
`pip install -r orca_rl/requirements.txt`.

Before launching GO2, place exactly one GO2 actor in the OrcaLab scene. The scene binding requires the GO2 joints,
actuators, contact sites, foot bodies, and touch sensors to match the asset suffixes used by
`orca_rl.rsl_env.robot_configs.GO2_CONFIG`.

The canonical task configs follow an mjlab/IsaacLab-style Python layout:

- `tasks/velocity/config/g1/env_cfgs.py`
- `tasks/velocity/config/g1/rl_cfg.py`
- `tasks/velocity/config/go2/env_cfgs.py`
- `tasks/velocity/config/go2/rl_cfg.py`

Programmatic use is exposed through the top-level package:

```python
from orca_rl import load_task_and_train_cfg, make_locomotion_vec_env
from orca_rl.tasks.velocity.config import (
    unitree_g1_flat_env_cfg,
    unitree_g1_rough_env_cfg,
    unitree_go2_flat_env_cfg,
    unitree_go2_rough_env_cfg,
)
```

The Orca runtime side is robot-neutral: `rsl_env/adapters/vecenv.py` creates the RSL-RL VecEnv, and
`rsl_env/locomotion_task.py` runs one bound robot instance. Robot-specific asset discovery is selected through each
config's `scene_binding.resolver` alias. Built-in aliases are `g1` and `go2`; custom import paths are still supported
for new assets.

When train/play/eval starts, Orca RL prints a terminal runtime summary with the selected device and GPU, observation
dimensions, action dimensions, reward terms, termination terms, commands, domain randomization, terrain, sensors,
curriculum, and scene binding.

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

Train GO2 rough terrain config:

```bash
python -m orca_rl.run_train \
  --config orca_rl/tasks/velocity/config/go2/env_cfgs.py:unitree_go2_rough_env_cfg
```

Train G1 rough terrain config:

```bash
python -m orca_rl.run_train \
  --config orca_rl/tasks/velocity/config/g1/env_cfgs.py:unitree_g1_rough_env_cfg
```

The `file.py:factory_name` form is the development-mode task selector until the later registry layer lands.

Export the generated rough terrain mesh for later OrcaLab import:

```bash
python -m orca_rl.terrains.export \
  --config orca_rl/tasks/velocity/config/go2/env_cfgs.py:unitree_go2_rough_env_cfg \
  --out generated_terrains/go2_rough.obj
```

The rough task already generates a heightfield and uses it for height-scan observations. Physics collision remains
disabled until OrcaLab allows local mesh/asset import or exposes a runtime terrain publish API.

## Backend Capability Gaps

OrcaLab runs MuJoCo behind a gRPC service, so client-side Python cannot safely mutate remote `mjModel` parameters unless
the server exposes explicit APIs. The following items are intentionally not marked complete.

Requires OrcaLab server/gRPC changes:

- **Rough terrain collision**: needs `AddCollisionMesh`, `ReplaceTerrainMesh`, native heightfield upload, or scene asset
  import/publish support. Required input from `orca_rl`: generated vertices/faces or heightfield grid, friction, pose,
  and collision group/material settings.
- **Runtime terrain switching**: needs a server API to swap terrain mesh/heightfield or activate a terrain tile without
  restarting the whole simulation. Required for reset-time terrain randomization and curriculum.
- **Friction randomization**: needs server-side `SetGeomFriction` or equivalent for named terrain/foot geoms. Client-only
  writes to a local model mirror are not enough.
- **Body-mass randomization**: needs server-side `SetBodyMass`, and ideally `SetBodyInertia` / `SetBodyCOM`, for named
  robot bodies.
- **Other physics randomization**: actuator gains, damping, armature, joint friction, solver params, and contact params
  need explicit remote setters if we want IsaacLab-style domain randomization.
- **True OrcaLab raycaster**: only useful if it raycasts against the same imported terrain mesh/heightfield. Until then,
  Python heightfield sampling should remain the source of truth for height scan observations.
- **Reliable non-foot contact sensors**: needs server-side contact filtering or force reporting by named geom/body pairs
  if we want `illegal_contact` to match the declared `ContactSensorCfg`.

Can be finished inside `orca_rl` after those APIs exist:

- Implement the terrain publisher/importer backend and make `terrain.physics_enabled=True` fail loudly when unavailable.
- Apply sampled friction/body-mass randomization through the new remote setters.
- Wire reset-time terrain tile selection and terrain curriculum progression.
- Compute rough rewards currently declared but not active: `feet_air_time`, `foot_clearance`, `body_ang_vel_l2`,
  `stand_still`, and `joint_deviation_l1`.
- Implement `illegal_contact` termination from the declared non-foot contact selector.
- Add a debug command that exports the terrain mesh plus a compact height-scan preview for visual inspection.

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
