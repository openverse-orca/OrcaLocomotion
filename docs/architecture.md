# Architecture

## Supported surface

OrcaLab-RSLRL follows a strict public/private boundary:

```text
Application
  ├─ `orca` CLI
  └─ `orcalab_rslrl.orca` Python API
          ↓
Task registry and validated runtime config
          ↓
RslRlVecEnvAdapter
          ↓
ManagerBasedRLEnv
          ↓
OrcaPhysics contract
          ↓
Private GPU physics implementation
```

Everything under `orcalab_rslrl._internal` is an implementation detail and may
change without deprecation. Product code must not import it. The public
contract consists of:

- `OrcaRuntimeConfig`
- `make_env`, `list_tasks`, and `register_task`
- `OrcaPhysics`, `OrcaState`, and `OrcaCapabilities` for task authors

There is intentionally no public backend selector. Backend upgrades must not
change task configuration, checkpoints, command lines, or MDP terms.

## Runtime ownership

`ManagerBasedRLEnv` owns the complete step/reset lifecycle. Tasks configure
commands, observations, rewards, terminations, events, action transforms, and
decimation. Terms read stable `env.orca` tensor views and never call private
solver kernels.

Every runtime state tensor starts with `num_envs`. Stepping, action writes,
sensor reads, reward computation, termination checks, and local resets remain
on the configured device. CPU synchronization is permitted only at explicit
boundaries such as logging, checkpointing, diagnostics, or rendering.

## Training and rendering

Training is headless. RSL-RL sees only `RslRlVecEnvAdapter`; it has no knowledge
of robot assets or the physics implementation. W&B consumes iteration-level
aggregates so media and network operations cannot stall rollout collection.

OrcaLab rendering consumes batched poses through a separate bridge. It may run
at a lower frequency than control and can be disabled without changing task
semantics. Renderer-specific layout offsets never modify physical state.

## Asset policy

Released robot XML, meshes, actuator definitions, and default poses live below
`orcalab_rslrl/assets/robots`. A task must validate all named joints, bodies,
actuators, and sensors at startup. Missing or reordered names are fatal errors;
silent index fallback is prohibited.

External asset paths are explicit overrides through `OrcaRuntimeConfig.asset`
or CLI `--asset`. A released task must work with packaged assets by default.

## Dependency rules

- `orca`, `envs`, `managers`, `mdp`, and `rslrl` cannot import a concrete
  physics implementation.
- Only task factories and private runtime modules construct concrete physics.
- MDP terms depend on `env.orca`, never private fields.
- Renderer code may use only public runtime mappings and state views.
- A new public symbol requires documentation, typing, and a contract test.

## Compatibility

Version 0.2 establishes the Orca-only surface. Older standalone console scripts
are consolidated under `orca`; `--mjcf` becomes `--asset`, and the backend
selection flag is removed. Checkpoints and registered task identifiers remain
unchanged.
