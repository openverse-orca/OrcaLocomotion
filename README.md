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
pip install --extra-index-url https://py.mujoco.org -r requirements.txt
```

When working from the original OrcaPlayground tree instead of the standalone `orca_rl` repository, use
`pip install --extra-index-url https://py.mujoco.org -r orca_rl/requirements.txt`.

`requirements.txt` now follows the MJWarp experiment line and installs MuJoCo 3.8, `warp-lang`, and
`mujoco-warp`. `orca-gym 26.4.3` still declares `mujoco==3.5.0`, so pip may report a resolver warning. For current
headless training this is acceptable because the G1 path uses local MuJoCo directly, and the tested CPU local path still
runs under MuJoCo 3.8. Keep this in mind if you later rely on an OrcaGym release feature that assumes exactly 3.5.

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

Registered task names are available for normal command-line use:

```bash
python -m orca_rl.run_train --list-tasks
```

Built-in registered tasks:

- `Unitree-G1-Flat`
- `Unitree-G1-Rough`
- `Unitree-GO2-Flat`
- `Unitree-GO2-Rough`

The Orca runtime side is robot-neutral: `rsl_env/adapters/vecenv.py` creates the RSL-RL VecEnv, and
`rsl_env/batched_locomotion_task.py` runs all robot agents inside one local MuJoCo runtime per simulator group.
Robot-specific binding is selected through each config's `scene_binding.resolver` alias. Built-in aliases are `g1` and
`go2`; custom import paths are still supported for new assets.

When train/play/eval starts, Orca RL prints a terminal runtime summary with the selected device and GPU, observation
dimensions, action dimensions, reward terms, termination terms, commands, domain randomization, terrain, sensors,
curriculum, and scene binding.

For G1, headless training uses a local MJCF clone-tiling path by default: one source G1 XML is cloned into
`g1_000`, `g1_001`, ... inside a generated local MuJoCo XML under
`/tmp/orca_rl_mjcf`. This path does not require an OrcaLab scene, does not require the `localhost:50051` gRPC service,
and does not publish actors into the OrcaLab layout.

RSL-RL parallelism is configured with the usual `num_envs` meaning. Use the config's `num_envs` field or override it
from the CLI:

```bash
python -m orca_rl.run_train \
  --config Unitree-G1-Flat \
  --num-envs 8
```

The adapter resolves exactly that many complete scene robot bindings and exposes exactly that many environments to
RSL-RL. There is no user-facing `subenv_num * agent_num` product in the Orca RL config.

Orca RL batches all resolved agents into one local MuJoCo runtime per simulator group. For example, G1
`--num-envs 4096` creates one RSL-RL VecEnv with 4096 logical environments but one simulator group, so stepping no
longer creates or advances 4096 separate `OrcaGymLocalEnv` Python wrappers. Multiple comma-separated remote addresses
still split non-local scene-backed runs across one simulator group per address.

G1 auto-publish is disabled by default to avoid accidentally inserting thousands of actors into the OrcaLab scene. If
you want to train against an OrcaLab-authored scene instead of the generated local MJCF, set
`scene_binding.local_xml_path = None` and prepare the scene explicitly.

Train GO2:

```bash
python -m orca_rl.run_train \
  --config Unitree-GO2-Flat \
  --headless
```

Train G1:

```bash
python -m orca_rl.run_train \
  --config Unitree-G1-Flat \
  --headless
```

Training is headless by default, so `--headless` / `--no-render` is mostly there to make the intent explicit in launch
scripts. Use `--render` only for short training-debug runs where you want the viewer updated during learning.

Experimental MJWarp stepping can be enabled for local headless training:

```bash
python -m orca_rl.run_train \
  --config Unitree-G1-Flat \
  --headless \
  --num-envs 24 \
  --sim-backend mjwarp
```

This keeps `OrcaGymLocalEnv` as the loader/metadata layer, but replaces the local `mujoco.mj_step(...)` call with
`mujoco_warp.step(...)`. After each control step the GPU state is synchronized back to CPU `MjData` so the existing
observation, reward, reset, and contact code remains compatible. The first launch compiles Warp kernels and can be slow;
subsequent launches use the Warp cache. This is a compatibility bridge, not yet mjlab's full zero-copy `nworld`
architecture, so it may be slower than CPU for small batches until observations/contact handling move fully onto GPU.
The default backend remains `orca_cpu`.

Visual playback/debugging stays in `run_play`:

```bash
python -m orca_rl.run_play \
  --config Unitree-G1-Flat \
  --ckpt <path_to_checkpoint>
```

`run_play` defaults to the visual OrcaLab scene path instead of local MuJoCo for G1. It disables
`scene_binding.local_xml_path`, requires the OrcaGym gRPC server, and can auto-publish the configured G1 asset when the
scene does not already contain a complete G1 binding. This keeps play useful for visual debugging in OrcaStudio.

To play through the generated local MuJoCo XML anyway:

```bash
python -m orca_rl.run_play \
  --config Unitree-G1-Flat \
  --ckpt <path_to_checkpoint> \
  --local-mujoco
```

Train GO2 rough terrain config:

```bash
python -m orca_rl.run_train \
  --config Unitree-GO2-Rough \
  --headless
```

Train G1 rough terrain config:

```bash
python -m orca_rl.run_train \
  --config Unitree-G1-Rough \
  --headless
```

The `file.py:factory_name` form still works as a development-mode task selector when editing new configs directly.

Export the generated rough terrain mesh for later OrcaLab import:

```bash
python -m orca_rl.terrains.export \
  --config orca_rl/tasks/velocity/config/go2/env_cfgs.py:unitree_go2_rough_env_cfg \
  --out generated_terrains/go2_rough.obj
```

For G1 local MJCF training, rough terrain is physical now. The generated heightfield is inserted into the generated
MuJoCo XML as an `hfield` collision geom, the old floor plane is removed, and height-scan observations/reset base
height use the same seeded terrain runtime. Generated rough XML files include a terrain hash suffix such as:

```text
/tmp/orca_rl_mjcf/g1_29dof_old_batch_24_terrain_ed885449.xml
```

The default backend is still CPU MuJoCo physics. `--sim-backend mjwarp` is the experimental GPU stepping path.

## Current Task Status

Flat velocity is a training-capable baseline, not a full IsaacLab/mjlab-equivalent environment. The following pieces are
active:

- RSL-RL VecEnv integration for G1/GO2.
- Actor and privileged critic observations.
- Bounded residual joint-target actions.
- PD torque control through OrcaGym.
- Velocity command sampling.
- Core flat rewards: linear/yaw velocity tracking, vertical velocity, orientation, base height, torque, action-rate,
  joint-limit, foot-slip, and termination penalty.
- Core terminations: base height range, tilt, invalid state, base contact, and timeout.

Flat velocity current notes:

- `randomize_friction` and `randomize_body_mass` now apply to the local MuJoCo model used by `OrcaGymLocalEnv`.
  Friction scales matched ground/terrain geoms, and base mass is reset from a stored baseline before each sampled delta
  is applied.
- `manager_terms` is used for config structure and diagnostics. It does not mean every declared MDP term is independently
  executed by a full manager framework.
- `ContactSensorCfg` is declarative metadata. Actual foot contact still uses the available OrcaGym touch/contact query
  path.
- Push disturbance, actuator gain/strength randomization, action latency, solver/contact parameter randomization, and
  base inertia/COM randomization are active in the local MuJoCo training path.
- Flat walking is now feature-complete enough for first serious RSL-RL experiments: policy/critic observations, bounded
  actions, rewards, terminations, reset noise, domain randomization, W&B/checkpointing, and runtime diagnostics are all
  wired. Remaining flat work is mostly parity/quality work, not a blocking interface gap.

Rough velocity adds procedural terrain observations on top of the flat baseline:

- Procedural heightfield generation is active.
- Height scan observations come from that generated heightfield and are no longer all-zero placeholders.
- OBJ export is active for later OrcaLab import.
- Rough runner/config selection is active through registered task names and `file.py:factory_name`.
- G1 local MJCF rough terrain inserts the heightfield into MuJoCo physics as an `hfield` collision geom.
- G1 rough reset/height scan/foot ground height use the same terrain config and seed as the generated MJCF.

Rough velocity current notes:

- G1 local MJCF rough terrain has physical collision.
- OrcaLab scene-backed rough terrain still needs a scene-side terrain import/upload API if you want visual play against
  the same generated terrain inside OrcaStudio.
- `RayCasterCfg` is metadata; height scans currently use Python heightfield sampling, not OrcaLab raycast.
- Rough rewards now computed by the runtime: `feet_air_time`, `foot_clearance`, `body_ang_vel_l2`, `stand_still`, and
  `joint_deviation_l1`.
- `illegal_contact` now uses OrcaGym contact queries and applies the configured force threshold when contact force is
  available.
- Physical terrain is compiled into the local MJCF, so reset-time terrain resampling is disabled on that path. Terrain
  curriculum metadata exists, but the success/failure progression loop is not active yet.

## Backend Capability Gaps

Current training uses `OrcaGymLocalEnv`, whose MuJoCo step runs in the local Python process. That path can mutate local
`mjModel` / `mjData` fields, so friction, mass, inertia/COM, actuator, latency, push, solver, and contact
randomization are implemented there. A future pure `RemoteEnv` path would still need explicit server/gRPC setters for
the same behavior.

Requires OrcaLab server/gRPC capability or currently restricted release permissions:

- **OrcaStudio rough terrain collision/play**: local G1 training already has physical hfield terrain in MuJoCo XML. The
  visual OrcaLab scene path still needs `AddCollisionMesh`, `ReplaceTerrainMesh`, native heightfield upload, or scene
  asset import/publish support if it should display and collide with the same generated rough terrain.
- **Runtime terrain switching**: needs a server API to swap terrain mesh/heightfield or activate a terrain tile without
  restarting the whole simulation. Required for reset-time terrain randomization and curriculum.
- **RemoteEnv domain randomization**: the local training path is implemented. A pure remote path still needs server-side
  setters for friction, mass, inertia/COM, actuator gains/limits, solver params, contact params, and push disturbance.
- **Unimplemented local physics randomization variants**: damping, armature, joint friction, motor delay dynamics beyond
  integer action latency, and full body-inertia tensor randomization beyond base inertia scaling are not active yet.
- **True OrcaLab raycaster**: only useful if it raycasts against the same imported terrain mesh/heightfield. Until then,
  Python heightfield sampling should remain the source of truth for height scan observations.
- **Reliable force-thresholded non-foot contact sensors**: `illegal_contact` is active through body/contact matching, but
  matching the declared `ContactSensorCfg(force_threshold=...)` exactly needs filtered named-pair force reporting.

Can be finished inside `orca_rl` after those APIs exist:

- Implement the terrain publisher/importer backend and make `terrain.physics_enabled=True` fail loudly when unavailable.
- Wire physical terrain tile selection and terrain curriculum progression after terrain import/publish exists.
- Add optional RemoteEnv setters for domain randomization if training ever moves away from `OrcaGymLocalEnv`.
- Upgrade `illegal_contact` to use force-thresholded named-pair filtering when OrcaLab exposes it.
- Add a debug command that exports the terrain mesh plus a compact height-scan preview for visual inspection.

## Remaining Unwired Items

Flat velocity has no known blocking interface gaps for first-stage RSL-RL training. The items below are the remaining
unwired or partially wired pieces, with the reason they are not complete and what needs to happen next.

Observed Orca / OrcaGym API state:

- `orca-gym` has a hybrid design: local MuJoCo execution through `OrcaGymLocalEnv` and a gRPC remote/server API.
- Public gRPC/proto paths currently expose model/data queries, `AddActor`, `PublishScene`, `LoadLocalEnv`,
  `LoadContentFile`, `SetOptConfig`, `SetGeomFriction`, `QueryContactSimple`, and `QueryContactForce`.
- The scene API can add a named actor from an already known `spawnable_name`. This is good for published assets such as
  G1/GO2, but it is not the same as uploading an arbitrary generated terrain mesh at runtime.
- I did not find a public RPC named like `AddCollisionMesh`, `AddHeightField`, `ReplaceTerrain`, `AddGeom`, or
  `SetMjModel`. That means scene-backed rough terrain collision for visual play needs either an OrcaLab asset/publish
  path or a new simulator-side API.
- Collision is not blocked because "simulation must decide whether to collide" in some abstract way. MuJoCo already
  computes contacts, but only between geoms that exist in the compiled `mjModel` and have valid collision settings
  (`contype`, `conaffinity`, geom type, material/contact params, pose, scale). The local G1 training path now inserts
  the generated heightfield into the compiled MJCF; the visual OrcaLab scene path still needs an equivalent terrain
  upload/import capability.

Remaining items:

- **OrcaLab scene rough terrain physics collision**
  - Current state: G1 local MJCF training has physical rough terrain collision; heightfield generation, height scan, and
    OBJ export are implemented.
  - Why not fully connected for play: the visual OrcaLab scene path still cannot upload the generated terrain into the
    scene as a collision asset through a known public API.
  - Best next path: ask OrcaLab to support one of these, in order of preference:
    1. Runtime terrain upload API: `AddHeightField` / `AddCollisionMesh` / `ReplaceTerrain`, returning geom names.
    2. Asset publish/import API: upload our OBJ/heightfield as a temporary spawnable, then `AddActor` it.
    3. Scene XML patch/reload path for play, if OrcaLab exposes a supported way to reload a patched visual scene.

- **Physical terrain randomization**
  - Current state: local MJCF rough terrain is compiled once per generated XML.
  - Why not fully connected: MuJoCo hfield geometry is part of the compiled model, so reset-time terrain changes need
    a pool/tile selection strategy or model reload.
  - Needed follow-up: keep a pool of terrain tiles and move spawn origins through terrain levels, or reload patched MJCF
    only at coarse curriculum boundaries.

- **Terrain curriculum**
  - Current state: `terrain_levels` metadata exists.
  - Why not fully connected: curriculum needs physical terrain levels. Before collision terrain exists, increasing
    observation difficulty alone would train against mismatched physics.
  - Needed follow-up: add episode success/failure metrics, then update terrain tile difficulty on reset.

- **True OrcaLab raycaster**
  - Current state: `RayCasterCfg` metadata exists; scans use Python heightfield sampling.
  - Why not fully connected: a true raycaster is useful only if it hits the same imported terrain geom that the robot
    collides with. Otherwise observation and collision can disagree.
  - Needed follow-up: after terrain import is verified, either add/use an OrcaLab raycast API, or keep Python scanning
    and add a debug checker comparing Python scan against OrcaLab ray hits.

- **Force-thresholded non-foot contact sensor**
  - Current state: `illegal_contact` is active through body/contact matching. `QueryContactForce` exists in OrcaGym, so
    force data appears available by contact id.
  - Why not fully connected: current runtime has not yet wrapped `QueryContactSimple` + `QueryContactForce` into the
    declared `ContactSensorCfg` abstraction with named geom/body filters and thresholds.
  - Needed follow-up: implement a contact sensor manager that maps contact ids to geom/body names, queries force by id,
    filters `nonfoot_ground_contact`, and applies `force_threshold`.

- **Pure `RemoteEnv` domain randomization**
  - Current state: local `OrcaGymLocalEnv` randomization is implemented by writing local `mjModel` / `mjData`.
  - Why not fully connected: pure remote mode has some setters (`SetGeomFriction`, `SetOptConfig`, actuator gain/bias),
    but I did not find complete setters for body mass, inertia/COM, dof damping/armature, contact geom params, or direct
    qvel push as named high-level APIs.
  - Needed follow-up: only needed if training moves away from `OrcaGymLocalEnv`. Add remote setters or expose a guarded
    model-edit API on the OrcaLab side.

- **Damping randomization**
  - Current state: not implemented locally.
  - Why not fully connected: we have not yet mapped controlled joint names to MuJoCo DoF damping slots and captured a
    reset-safe baseline.
  - Needed follow-up: use joint `qvel_idx_start` / DoF address data to write `model.dof_damping` for controlled joints.

- **Armature randomization**
  - Current state: not implemented locally.
  - Why not fully connected: same mapping/baseline issue as damping.
  - Needed follow-up: capture and randomize `model.dof_armature` for controlled joint DoFs.

- **Joint friction / passive loss randomization**
  - Current state: not implemented locally.
  - Why not fully connected: need to confirm whether G1/GO2 Orca assets use MuJoCo passive joint friction/loss fields in
    a way that matters for these actuators.
  - Needed follow-up: inspect compiled joint/DoF fields, then add reset-safe writes for the correct MuJoCo friction/loss
    field.

- **Richer motor delay dynamics**
  - Current state: integer action latency is implemented.
  - Why not fully connected: the current buffer does not model actuator bandwidth, low-pass filtering, dropped packets,
    or per-joint latency.
  - Needed follow-up: add optional first-order action filtering, per-joint delay, and dropout after baseline training is
    stable.

- **Full inertia tensor randomization**
  - Current state: base inertia scale is implemented.
  - Why not fully connected: current implementation scales only the base diagonal inertia vector; it does not randomize
    every body or validate arbitrary physical inertia tensors.
  - Needed follow-up: add body selection, positive-definite inertia checks, and per-body tensor scaling.

- **Contact parameter coverage**
  - Current state: matched ground/terrain geom `solref`, `solimp`, and `margin` are randomized.
  - Why not fully connected: it does not yet cover explicit robot foot collision geoms, geom-pair overrides, or global
    contact options.
  - Needed follow-up: after scene geom names are stable, add foot geom matching and optional pair/material-level
    randomization.

- **Terrain visual/debug preview**
  - Current state: OBJ export exists.
  - Why not fully connected: there is no compact command that renders the heightfield and scan overlay for inspection.
  - Needed follow-up: add a CLI that exports OBJ plus PNG/NumPy summaries so terrain/height-scan alignment can be checked
    before training.

## Flat/Rough Implementation Checklist

- Done: flat RSL-RL VecEnv, actor/critic observations, bounded residual joint actions, PD torque control, command
  sampling, core flat rewards, base/tilt/height/contact/time terminations, and checkpoint/W&B plumbing.
- Done: local MuJoCo friction randomization for matched ground/terrain geoms.
- Done: reset-safe local base mass, base inertia scale, and base COM-offset randomization.
- Done: actuator PD gain randomization, torque-strength randomization, action latency, periodic push disturbance, local
  solver parameter randomization, and contact parameter randomization.
- Done: rough heightfield generation, yaw-aware height scans, reset-time observation-terrain resampling, and OBJ export.
- Done: rough runtime rewards for `feet_air_time`, `foot_clearance`, `body_ang_vel_l2`, `stand_still`, and
  `joint_deviation_l1`.
- Done: body/contact based `illegal_contact` termination.
- Pending OrcaLab permission/API: insert generated terrain into physics collision and verify imported mesh matches the
  heightfield used by observations.
- Pending after terrain physics: terrain curriculum progression and physical terrain tile switching.
- Pending optional parity work: damping/armature/joint-friction randomization, richer delay dynamics, full inertia tensor
  randomization, and a true OrcaLab raycaster backend.

Train G1 with W&B logging:

```bash
python -m orca_rl.run_train \
  --config orca_rl/tasks/velocity/config/g1/env_cfgs.py \
  --headless \
  --logger wandb \
  --wandb-project orca_locomotion \
  --wandb-mode online
```

The training entrypoint always writes stable checkpoint aliases after `runner.learn()`:

- `model_last.pt`
- `model_final.pt`

Training logs and checkpoints use the same layout as mjlab-style RSL-RL runs:

- `logs/rsl_rl/g1_velocity/<date_time>_flat`
- `logs/rsl_rl/g1_velocity/<date_time>_rough`
- `logs/rsl_rl/go2_velocity/<date_time>_flat`
- `logs/rsl_rl/go2_velocity/<date_time>_rough`

The built-in G1/GO2 runner configs use the mjlab-style PPO setup: Gaussian policy distribution with scalar
`init_std=1.0`, actor/critic hidden dims `(512, 256, 128)`, `learning_rate=1e-3`, `num_steps_per_env=24`,
`save_interval=50`, and `max_iterations=30000`. Use `--num-iterations` to shorten a smoke test.

Short multi-env smoke test:

```bash
python -m orca_rl.run_train \
  --config orca_rl/tasks/velocity/config/g1/env_cfgs.py \
  --num-envs 2 \
  --num-iterations 1 \
  --headless
```

If OrcaGym is not listening on `localhost:50051`, override the address:

```bash
python -m orca_rl.run_train \
  --config orca_rl/tasks/velocity/config/g1/env_cfgs.py \
  --remote 192.168.1.10:50051 \
  --headless
```

Play a checkpoint:

```bash
python -m orca_rl.run_play \
  --config orca_rl/tasks/velocity/config/g1/env_cfgs.py \
  --ckpt logs/rsl_rl/g1_velocity/<date_time>_flat/model_<iter>.pt
```

Evaluate:

```bash
python -m orca_rl.run_eval \
  --config orca_rl/tasks/velocity/config/g1/env_cfgs.py \
  --ckpt logs/rsl_rl/g1_velocity/<date_time>_flat/model_<iter>.pt
```
