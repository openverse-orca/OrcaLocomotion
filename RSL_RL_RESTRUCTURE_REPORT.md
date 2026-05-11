# RSL-RL OrcaLab Integration Report

## Summary

This package wires RSL-RL locomotion training into OrcaLab / OrcaGym through a robot-neutral Python package named
`orca_rl`. The current runtime is now centered on a batched local MuJoCo bridge: one simulator group creates one
`OrcaGymLocalEnv`, and all robot agents inside that local MuJoCo model are exposed to RSL-RL as logical vector
environments.

The most important recent change is the removal of the old per-environment runtime. Earlier, `num_envs=4096` meant
4096 Python task wrappers, 4096 local-env initializations, and 4096 serial step calls. The new path uses
`BatchedOrcaLocomotionTask`, so G1 `num_envs=4096` means one simulator group with 4096 logical RSL-RL environments.

The G1 training path now also has a local MJCF clone-tiling mode. In headless training it can generate a batched MuJoCo
XML directly from a local source XML, so it no longer needs an OrcaLab scene, does not require the `localhost:50051`
gRPC service, and does not publish actors into the interactive layout.

## Goals

- Use RSL-RL's standard `VecEnv` interface for locomotion training.
- Keep task configuration close to mjlab / IsaacLab: Python config factories, robot-specific config packages, and
  MDP-style declarations for rewards, observations, commands, terminations, and events.
- Run with visual rendering disabled during training.
- Keep `run_play` visual for policy inspection.
- Avoid one Python simulator wrapper per logical environment.
- Avoid surprise large-scale OrcaLab scene publishing when a large `--num-envs` value is used.
- Provide a local MJCF path for G1 so headless training can start without OrcaStudio scene setup.

## Current Layout

Task and runner configs:

```text
orca_rl/tasks/velocity/config/g1/env_cfgs.py
orca_rl/tasks/velocity/config/g1/rl_cfg.py
orca_rl/tasks/velocity/config/go2/env_cfgs.py
orca_rl/tasks/velocity/config/go2/rl_cfg.py
```

RSL-RL adapter and batched runtime:

```text
orca_rl/rsl_env/adapters/factory.py
orca_rl/rsl_env/adapters/vecenv.py
orca_rl/rsl_env/batched_locomotion_task.py
orca_rl/rsl_env/local_mjcf.py
orca_rl/rsl_env/rendering.py
```

Shared runtime managers:

```text
orca_rl/rsl_env/action_mapper.py
orca_rl/rsl_env/obs_builder.py
orca_rl/rsl_env/reward_manager.py
orca_rl/rsl_env/termination_manager.py
orca_rl/rsl_env/randomization.py
orca_rl/rsl_env/terrain_runtime.py
```

Scene binding and asset discovery:

```text
orca_rl/rsl_env/scene_binding.py
orca_rl/rsl_env/model_scanner.py
orca_rl/rsl_env/scene_resolvers.py
orca_rl/rsl_env/robot_configs.py
```

Support packages:

```text
orca_rl/sensor/
orca_rl/terrains/
orca_rl/managers/
```

The old single-agent runtime file `orca_rl/rsl_env/locomotion_task.py` has been removed from the active tree. The active
runtime path is batched-only.

## Launch Flow

Training starts in `orca_rl.run_train`:

1. Load the task config with `load_task_and_train_cfg`.
2. Apply CLI overrides such as `--num-envs`, `--remote`, logging, device, and rendering mode.
3. Force training to headless by default:
   - `headless=True`
   - `render_mode="none"`
4. Create an RSL-RL environment through `make_locomotion_vec_env`.
5. Print runtime diagnostics.
6. Construct `rsl_rl.runners.OnPolicyRunner`.
7. Call `runner.learn(...)`.

Playback starts in `orca_rl.run_play` and deliberately uses:

```text
headless=False
render_mode="human"
```

Evaluation starts in `orca_rl.run_eval` and uses:

```text
headless=True
render_mode="none"
```

## Scene Binding And Local MJCF

The adapter does not hard-code G1 or GO2 names. Each config sets:

```python
scene_binding = {
    "resolver": "g1",  # or "go2"
    ...
}
```

There are now two binding sources.

For G1 headless training, the default config sets:

```python
"local_xml_path": "auto"
```

That selects the local MJCF path:

1. Locate a source G1 XML. The default candidates include `/home/huan-hu/OrcaPlayground/examples/g1/g1_29dof_old.xml`,
   and `ORCA_RL_G1_XML` can override this.
2. Clone the robot body, actuators, sensors, contact exclusions, and references into `g1_000`, `g1_001`, ...
3. Offset each cloned root body on a grid.
4. Write the generated batch XML to `/tmp/orca_rl_mjcf`.
5. Return those generated `agent_names` plus the generated `model_xml_path`.

For scene-backed GO2 or explicit OrcaLab workflows, the resolver scans the compiled OrcaLab / OrcaGym model for complete
robot instances. A complete instance must provide the expected joints, actuators, sites, bodies, and sensors. The
resolver returns:

- `agent_names`
- `robot_config`
- optional `model_xml_path`

The batched runtime then passes all `agent_names` into a single `OrcaGymLocalEnv`. If `model_xml_path` is present, it
loads that local XML directly and bypasses OrcaLab scene loading.

## Why 4096 Used To Stall

The old implementation treated each logical RSL-RL environment as its own `OrcaLocomotionTask`. That meant:

- one `OrcaGymLocalEnv` object per logical env;
- one gRPC/local model initialization per logical env;
- one reset path per logical env;
- one `task.step(action)` Python call per logical env;
- one `do_simulation` / `update_data` path per logical env;
- repeated contact, site, and sensor queries per logical env.

For `--num-envs 4096`, this was effectively 4096 separate Python simulator wrappers. Headless mode helped by skipping
render calls, but it did not change this per-env simulator architecture.

## Batched Runtime Improvement

The new runtime is `BatchedOrcaLocomotionTask`.

For each simulator group:

1. Resolve or generate the requested robot agents.
2. Create one `OrcaGymLocalEnv` with all those `agent_names`.
3. If a generated local MJCF path is provided, initialize `OrcaGymLocal(None)` and skip gRPC-only setup.
4. Build per-agent metadata once:
   - qpos/qvel offsets;
   - actuator ids;
   - joint limits;
   - torque limits;
   - foot contact bodies/sites/sensors;
   - base contact bodies;
   - observation/reward/termination managers.
5. Stack the key indices into NumPy arrays.
6. During step:
   - receive an action matrix shaped `(num_envs_on_address, num_actions)`;
   - compute all residual joint targets and PD torques in a batched NumPy operation;
   - write all actuator commands into one full MuJoCo control vector;
   - call `mj_step` once per decimation loop for the whole scene;
   - call `update_data` once;
   - query contacts and foot positions in grouped queries;
   - compute per-agent observations, rewards, terminations, and logs.

The RSL-RL adapter still exposes a standard VecEnv:

```text
obs, rewards, dones, extras = env.step(actions)
```

The difference is that internally it now loops over simulator groups, not over every logical environment.

## Simulator Groups

`OrcaRslRlVecEnv` now creates one batched simulator group per local MJCF batch or per OrcaGym address.

Examples:

```text
--num-envs 4096
```

Result:

```text
num_envs: 4096
num_sim_groups: 1
```

```text
--num-envs 4096 --remote host0:50051,host1:50051,host2:50051,host3:50051
```

Result:

```text
num_envs: 4096
num_sim_groups: 4
```

Each simulator group owns one local MuJoCo runtime. The runtime summary prints `num_sim_groups`, `source`, and
`model_xml_path` so accidental regression back to per-env wrapping or scene-backed loading is visible immediately.

## Rendering / Headless Changes

Training now defaults to no rendering:

```bash
python -m orca_rl.run_train \
  --config orca_rl/tasks/velocity/config/g1/env_cfgs.py \
  --headless
```

`--headless` and `--no-render` are aliases. `--render` exists only for short debugging runs.

The rendering resolver normalizes:

- `headless=True` -> `render_mode="none"`
- `headless=False` with no explicit mode -> `render_mode="human"`

`run_play` keeps rendering enabled so a trained policy can be inspected visually.

For G1 local MJCF headless training there is no OrcaStudio render loop involved at all. For scene-backed visual or
debug runs, this still only disables render calls from `orca_rl`; an already-running OrcaStudio viewport can still
consume GPU resources.

## Scene Auto-Publish Change

G1 previously had:

```python
"spawn_if_missing": True
```

That was convenient for small smoke tests, but dangerous for large training. If `--num-envs 4096` was requested and the
scene had fewer complete G1 actors, the resolver could attempt to publish thousands of G1 actors into OrcaLab. This is
the likely source of "strange things being added to the scene" and large render/compile overhead.

G1 now defaults to:

```python
"spawn_if_missing": False
"max_auto_spawn_count": 1
"local_xml_path": "auto"
```

The local XML setting means normal headless G1 training no longer scans or publishes the OrcaLab scene. If
`local_xml_path` is set to `None` and `spawn_if_missing` is explicitly enabled, the resolver still refuses large
auto-publish requests. If missing actors exceed `max_auto_spawn_count`, it raises a clear error instead of filling the
scene.

For large G1 training, the expected workflow is now:

1. Keep `scene_binding.local_xml_path = "auto"`.
2. Start training with `--headless`.
3. Verify the runtime summary:

```text
num_envs: <requested count>
num_sim_groups: 1
headless: True
render_mode: none
source: local_mjcf
```

## RSL-RL Interface

The adapter implements the fields RSL-RL expects:

- `num_envs`
- `num_actions`
- `max_episode_length`
- `episode_length_buf`
- `get_observations()`
- `step(actions)`
- `reset()`
- `close()`

Observations are returned as a `TensorDict` with:

- `policy`
- `privileged`

The runner config maps RSL-RL actor/critic groups to these observation names:

```python
obs_groups = {
    "actor": ["policy"],
    "critic": ["policy", "privileged"],
}
```

The policy receives the actor group, while the critic can receive privileged terms.

## Action Pipeline

The policy outputs normalized residual joint-position actions.

Runtime flow:

1. Clip policy actions.
2. Convert actions to target joint positions around nominal qpos.
3. Clamp targets to safety-scaled joint limits.
4. Compute PD torque:

```text
torque = kp * (target_qpos - qpos) - kd * qvel
```

5. Clamp torque to effort limits.
6. Write all agent torques into one full MuJoCo `ctrl` vector.
7. Step the whole MuJoCo scene.

The batched runtime vectorizes target and torque computation across agents on the same OrcaGym address.

## Observation Pipeline

For each agent, the runtime reads:

- base position and orientation;
- base linear/angular velocity;
- joint position and velocity;
- command;
- last action;
- last torque;
- foot position/velocity/contact;
- domain randomization state;
- optional height scan samples.

The actor observation includes:

- base angular velocity;
- projected gravity;
- command;
- joint position relative to nominal pose;
- joint velocity;
- last action;
- optional height scan.

The privileged observation includes:

- base linear velocity;
- base angular velocity;
- projected gravity;
- base height;
- foot contacts;
- foot heights;
- foot velocities;
- last torque;
- domain randomization values;
- optional height scan.

## Reward / Termination Pipeline

The flat task computes:

- linear velocity tracking;
- yaw velocity tracking;
- vertical velocity penalty;
- orientation penalty;
- base height penalty;
- torque penalty;
- action-rate penalty;
- joint-limit penalty;
- foot-slip penalty;
- termination penalty.

The rough task additionally wires:

- feet air time;
- foot clearance;
- body angular velocity;
- stand-still regularization;
- joint deviation;
- illegal contact.

Terminations include:

- too low;
- too high;
- too tilted;
- base contact;
- illegal contact when configured;
- invalid numerical state;
- timeout.

## Domain Randomization

The local MuJoCo path supports:

- friction scaling for matched ground/terrain geoms;
- base mass delta;
- base inertia scale;
- base COM offset;
- actuator kp/kd scaling;
- torque strength scaling;
- integer action latency;
- push disturbance;
- solver iteration/tolerance randomization;
- contact solref/solimp/margin randomization.

Some randomization values are global to the shared MuJoCo model, while others are per-agent. In a batched scene, global
model fields cannot be independently different for every actor at the exact same time without deeper model duplication
or per-geom/per-body field mapping for every replicated actor. The current implementation keeps the per-agent state for
observations and per-agent actuator scaling, while global model parameters are applied at reset using a representative
sample for the shared model.

## Rough Terrain Status

Rough terrain support is active as an observation/config scaffold:

- procedural heightfield generation;
- height scan observations;
- OBJ export;
- rough reward terms;
- illegal contact termination.

Physical rough-terrain collision is still gated because the generated terrain is not inserted into the compiled OrcaLab
MuJoCo model by this package. Until OrcaLab exposes a reliable runtime terrain upload/import/reload path, robot feet
still collide with the scene's existing terrain while height scans come from the Python heightfield.

## Why Startup May Still Be Slow

The batched runtime removes per-env Python simulator wrappers and per-env step calls. The local G1 MJCF path also removes
OrcaStudio scene publishing from the headless training hot path.

Startup can still be slow if:

- the generated batch XML is very large;
- MuJoCo compiles thousands of cloned articulated bodies;
- contact/site/sensor dictionaries for thousands of actors are expensive to build;
- a scene-backed workflow is used and OrcaStudio is rendering or compiling the stage.

This is why G1 auto-publish is disabled by default, local MJCF is the default for G1 headless training, and the runtime
summary prints the active source.

## Remaining Gap Versus mjlab / IsaacLab

mjlab and IsaacLab can launch very large batches quickly because they usually avoid interactive scene publishing in the
hot path. They either build a local batched model directly, use a tensorized simulator, or clone environments inside an
already optimized simulator scene.

The remaining high-impact work for Orca RL is:

1. Extend the local MJCF clone-tiling path to GO2 and rough-terrain collision assets.
2. Cache or reuse generated and compiled batched scenes where MuJoCo allows it.
3. Move observation/reward/termination calculation further from per-agent Python loops to vectorized NumPy/Torch.
4. Add an OrcaLab true server-headless launch path for scene-backed visual/debug workflows.
5. Add performance diagnostics that separately time XML generation, MuJoCo compile, env creation, physics step, data sync,
   contact query, observation build, reward build, and RSL-RL update.

## Recommended Large-Batch Workflow

For G1 headless training, use the local MJCF path.

```bash
python -m orca_rl.run_train \
  --config orca_rl/tasks/velocity/config/g1/env_cfgs.py \
  --num-envs 4096 \
  --headless
```

Check the runtime summary before letting the run continue:

```text
num_envs: 4096
num_sim_groups: 1
headless: True
render_mode: none
source: local_mjcf
model_xml_path: /tmp/orca_rl_mjcf/...
```

If `source` is `orcalab_scene`, the run is using a scene-backed workflow. That can be useful for visual debugging or
GO2, but it is not the intended fast G1 headless path.

## Validation

Static checks performed:

```bash
python -m compileall orca_rl
python -m orca_rl.run_train --help
python -m orca_rl.run_play --help
python -m orca_rl.run_eval --help
git diff --check
```

Import checks in the OrcaLab conda environment:

```bash
/home/huan-hu/miniconda3/envs/orcalab/bin/python - <<'PY'
from orca_rl.rsl_env.batched_locomotion_task import BatchedOrcaLocomotionTask
from orca_rl.rsl_env.adapters.vecenv import OrcaRslRlVecEnv, _split_num_envs
print(BatchedOrcaLocomotionTask.__name__, OrcaRslRlVecEnv.__name__)
print(_split_num_envs(4096, 1), _split_num_envs(4096, 4))
PY
```

Observed:

```text
BatchedOrcaLocomotionTask OrcaRslRlVecEnv
[4096] [1024, 1024, 1024, 1024]
```

Local MJCF validation in the OrcaLab conda environment:

```bash
/home/huan-hu/miniconda3/envs/orcalab/bin/python - <<'PY'
from orca_rl.rsl_env.local_mjcf import build_local_mjcf_batch, resolve_existing_xml_path
from orca_rl.rsl_env.scene_binding import G1_LOCAL_XML_CANDIDATES
import mujoco
source = resolve_existing_xml_path("auto", G1_LOCAL_XML_CANDIDATES)
path = build_local_mjcf_batch(source_xml_path=source, agent_names=[f"g1_{i:03d}" for i in range(24)])
model = mujoco.MjModel.from_xml_path(path)
print(source, path, model.nq, model.nv, model.nu)
PY
```

Observed:

```text
/home/huan-hu/OrcaPlayground/examples/g1/g1_29dof_old.xml /tmp/orca_rl_mjcf/g1_29dof_old_batch_24.xml 864 840 696
```

Live smoke test without an OrcaLab gRPC server:

```bash
/home/huan-hu/miniconda3/envs/orcalab/bin/python -m orca_rl.run_train \
  --config orca_rl/tasks/velocity/config/g1/env_cfgs.py \
  --headless \
  --num-envs 24 \
  --num-iterations 1
```

Observed:

```text
num_envs: 24
num_sim_groups: 1
source: local_mjcf
model_xml_path: /tmp/orca_rl_mjcf/g1_29dof_old_batch_24.xml
Steps per second: 520
```
