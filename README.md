# Orca RL

Orca RL 是一个把 RSL-RL 接入 OrcaLab / OrcaGym / MuJoCo 生态的训练与播放项目。当前重点是 Unitree G1 的速度跟踪训练、headless 本地 MuJoCo 训练、Unitree/mjlab checkpoint 回放到 OrcaLab scene，以及实验性 MJWarp 接入。

当前可用能力：

- Unitree G1 flat / rough velocity task
- Unitree GO2 flat / rough 配置骨架
- RSL-RL `OnPolicyRunner` 训练入口
- 本地 MuJoCo headless training
- `--headless` / `--no-render` 无渲染训练模式
- G1 batched local MJCF 训练
- G1 rough physical hfield terrain
- 实验性 `mujoco_warp` step backend
- OrcaLab scene play / debug
- Unitree/mjlab G1 checkpoint 在 OrcaLab scene 中 play

## 当前架构

默认训练路径：

```text
RSL-RL
  -> OrcaRslRlVecEnv
  -> BatchedOrcaLocomotionTask
  -> OrcaGymLocalEnv
  -> local MuJoCo MjModel / MjData
```

G1 headless 训练默认不再要求手动打开 OrcaLab 场景，也不依赖 OrcaLab viewport 渲染。它会从本地 G1 MJCF 生成 batched XML，然后用 `OrcaGymLocalEnv` 在本地进程里跑 MuJoCo。

实验性 MJWarp 路径：

```text
OrcaGymLocalEnv loads local MuJoCo model
  -> MjWarpRuntime
  -> mujoco_warp.step(...)
  -> sync back to CPU MjData
  -> reuse numpy obs / reward / contact
```

这条路径能跑，但不是最终快路径。真正像 mjlab 一样快，需要单机器人 model + `mujoco_warp.put_data(nworld=num_envs)` + torch/warp 版 observation / reward / reset / contact，避免每个 control step 同步回 CPU。

## 安装

在 OrcaLab Python 环境里安装：

```bash
pip install -r requirements.txt
```

`requirements.txt` 当前按本地 MuJoCo / MJWarp 实验路线写入：

```text
mujoco>=3.8.0.dev0
warp-lang>=1.12.0
mujoco-warp
rsl-rl-lib
torch
tensordict
onnx / onnxscript
tensorboard / wandb
```

注意：`orca-gym 26.4.3` 仍声明固定依赖 `mujoco==3.5.0`。本项目现在为了 MJWarp 实验使用 MuJoCo 3.8，本地 G1 training 已验证可启动；如果后续使用 OrcaGym 中强依赖 3.5 的功能，需要重新验证。

## 快速命令

列出任务：

```bash
python -m orca_rl.run_train --list-tasks
```

当前注册任务：

```text
Unitree-G1-Flat
Unitree-G1-Rough
Unitree-GO2-Flat
Unitree-GO2-Rough
```

训练 G1 flat。训练默认就是 headless，显式写 `--headless` 或 `--no-render` 是为了避免误开 viewer：

```bash
python -m orca_rl.run_train \
  --config Unitree-G1-Flat \
  --headless \
  --num-envs 24
```

等价写法：

```bash
python -m orca_rl.run_train \
  --config Unitree-G1-Flat \
  --no-render \
  --num-envs 24
```

训练 G1 rough：

```bash
python -m orca_rl.run_train \
  --config Unitree-G1-Rough \
  --headless \
  --num-envs 24
```

实验性 MJWarp step：

```bash
python -m orca_rl.run_train \
  --config Unitree-G1-Flat \
  --headless \
  --num-envs 24 \
  --sim-backend mjwarp
```

短程可视化 debug 训练：

```bash
python -m orca_rl.run_train \
  --config Unitree-G1-Flat \
  --render \
  --num-envs 1
```

OrcaLab scene 中播放 Orca RL checkpoint：

```bash
python -m orca_rl.run_play \
  --config Unitree-G1-Flat \
  --ckpt <path_to_checkpoint>
```

OrcaLab scene 中播放 Unitree/mjlab G1 checkpoint：

```bash
python -m orca_rl.run_play \
  --config Unitree-G1-Flat \
  --policy-backend mjlab \
  --ckpt /home/huan-hu/orca_rl/test_model_G1_mjlab_Flat.pt \
  --lin-vel-x 0.5 \
  --lin-vel-y 0.0 \
  --ang-vel-z 0.0
```

不要加 `--local-mujoco`，这样才会播放到 OrcaLab scene 里的 G1。若不传 `--ckpt`，会自动寻找：

```text
third_party/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/*/model_*.pt
```

G1 scene binding / auto-publish 默认使用 OrcaLab 资产：

```text
assets/e071469a36d3c8aa/unitree_robots/prefabs/g1_29dof_usda
```

本地 MuJoCo play：

```bash
python -m orca_rl.run_play \
  --config Unitree-G1-Flat \
  --ckpt <path_to_checkpoint> \
  --local-mujoco
```

`run_play` 默认保留 human rendering，用于看策略动作；`--local-mujoco` 只作为不接 OrcaLab scene 的调试路径。

评估 checkpoint：

```bash
python -m orca_rl.run_eval \
  --config Unitree-G1-Flat \
  --ckpt <path_to_checkpoint> \
  --steps 2000
```

导出 rough terrain OBJ：

```bash
python -m orca_rl.terrains.export \
  --config Unitree-G1-Rough \
  --out generated_terrains/g1_rough.obj
```

## Headless / No-Rendering 训练

这部分原来记录在 `TODO_headless_rendering.md`，现在已经合并到主 README。

训练入口支持：

```text
--headless
--no-render
```

它们会把配置写成：

```text
sim.headless = True
sim.render_mode = "none"
```

并沿着下面的路径传入后端：

```text
run_train
  -> make_locomotion_vec_env(...)
  -> OrcaRslRlVecEnv
  -> BatchedOrcaLocomotionTask
  -> OrcaGymLocalEnv / MuJoCo backend
```

headless 模式的目标是训练热路径里不打开 viewer、不渲染 camera frame、不做 frame display，也不把大批量机器人发布到 OrcaLab 交互场景里。G1 headless 训练默认使用 generated local MJCF batch，因此可以不依赖 OrcaLab viewport、不依赖 gRPC server，也不会因为场景里存在大量可视化 actor 而拖慢训练。

play/debug 则反过来：`run_play` 默认使用 human rendering，并优先走 OrcaLab scene binding。G1 play 会禁用 local MJCF 路径，使用 scene binding resolver；如果场景里没有 G1，会尝试自动发布配置里的 G1 asset，`--local-mujoco` 则保留为完全不接 OrcaLab scene 的调试路径。这样训练和播放分工清楚：

```text
train: headless local MuJoCo, 追求吞吐
play: OrcaLab scene / human render, 追求可视化检查
```

G1 rough local MJCF 训练还会把 rough heightfield 插入 MuJoCo，作为真实 `hfield` collision geom，而不是只在 reward 里使用高度采样。

## 后端

### `orca_cpu`

默认训练后端：

```text
local MJCF
  -> OrcaGymLocalEnv
  -> mujoco.mj_step
  -> numpy obs / reward / contact
```

优点：

- 当前最稳定
- G1 flat / rough smoke 通过
- rough hfield 已经是真 MuJoCo collision geom
- 与现有 OrcaGym query / scene binding 兼容

限制：

- 仍是 CPU MuJoCo
- 4096 环境靠 clone 到同一个 XML，不是 mjlab 的 `nworld`
- observation / reward / contact 仍主要在 CPU / numpy

### `mjwarp`

实验性后端：

```text
OrcaGymLocalEnv 初始化 model/data
  -> MjWarpRuntime
  -> mujoco_warp.step
  -> sync_to_cpu
  -> 原有 obs / reward / contact
```

它已经能跑 G1 flat / rough 的 1-iteration smoke，并对 `frame_skip` step 做 CUDA graph capture。但每个 control step 后仍同步回 CPU，观测、奖励、contact 仍复用 numpy 代码，所以小 batch 下可能比 CPU 慢。

### 未来 `mjwarp_nworld`

真正高吞吐路线应该新增独立后端：

```text
single robot MJCF
  -> mujoco_warp.put_model(model)
  -> mujoco_warp.put_data(model, data, nworld=num_envs)
  -> torch/warp obs
  -> torch/warp reward
  -> torch/warp done/reset
  -> RSL-RL
```

OrcaGym 可以继续用于 play、asset 管理和 scene 可视化，但训练热路径要避免 CPU 同步。

## Policy ABI

训练出的 policy 能否放回 OrcaGym / OrcaLab play，取决于 policy ABI 是否一致：

```text
actor observation shape
actor observation order
observation scale
body-frame / world-frame transform
quaternion format: MuJoCo wxyz
action dim
joint order
actuator order
action -> target qpos / torque semantics
command format
rough height_scan dim and semantics
```

Orca RL 旧 G1 flat actor observation：

```text
base_ang_vel_body
projected_gravity
command
joint_pos_rel
joint_vel
last_action
```

Unitree/mjlab G1 flat actor observation 是 98 维：

```text
base_ang_vel_body
projected_gravity
command
phase(sin, cos)
joint_pos_rel
joint_vel
last_action
```

所以播放 Unitree/mjlab checkpoint 时必须使用：

```bash
--policy-backend mjlab
```

这个 backend 做的事情：

```text
1. 读取 Unitree/mjlab RSL-RL checkpoint 里的 actor_state_dict
2. 构造 mjlab 的 G1 actor observation
3. 使用 mjlab action_scale = 0.25 * effort / stiffness
4. 将 OrcaLab runtime G1 joint range / armature / damping / frictionloss 对齐到 mjlab 训练配置
5. 将 OrcaLab runtime motor 改成 mjlab 风格的 MuJoCo position actuator 语义
6. play step 写入 target joint position
7. 让 MuJoCo 根据 stiffness / damping / force limit 产生控制力
```

启动后应看到类似日志：

```text
[orca_rl.play] Mjlab runtime alignment: tasks=1, agents=1, joints=29, actuators=29, position_actuator_tasks=1
```

## mjlab G1 到 OrcaLab Play 的对齐

OrcaLab scene play 中没有修改 OrcaLab 源码、USDA 资产或缓存 XML 文件。对齐发生在当前 Python play 进程里：

```text
OrcaLab scene G1 asset
  -> OrcaGym 下发 runtime MuJoCo model
  -> orca_rl 拿到 gym._mjModel
  -> patch 当前 MjModel 的 joint / actuator 参数
  -> policy action
  -> target_qpos
  -> task.set_ctrl(target_qpos)
  -> mj_step()
```

具体 patch：

```text
model.jnt_range[joint_id] = mjlab_joint_range
model.jnt_limited[joint_id] = True
model.dof_armature[dof_id] = mjlab_armature
model.dof_damping[dof_id] = 0.0
model.dof_frictionloss[dof_id] = mjlab_frictionloss
```

执行器从原始 motor 语义改成 position actuator 语义：

```text
actuator_dyntype = mjDYN_NONE
actuator_gaintype = mjGAIN_FIXED
actuator_biastype = mjBIAS_AFFINE
gainprm[0] = kp
biasprm[1] = -kp
biasprm[2] = -kd
forcelimited = True
forcerange = [-effort_limit, effort_limit]
ctrllimited = False
```

因此现在给 OrcaLab 发送的是：

```text
ctrl = target_qpos
```

不是直接发送 torque。PD force 由 MuJoCo 在 `mj_step()` 中根据 `gainprm/biasprm/forcerange` 计算。

当前限制：这套 runtime patch 针对 CPU MuJoCo / OrcaLab scene play。如果使用 `sim.backend=mjwarp`，MJWarp 数据已经搬到 GPU 侧，play 对齐会跳过。

## OrcaLab Runtime XML 差异

对比对象：

```text
mjlab 裸机器人:
third_party/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/g1.xml

OrcaLab runtime:
/home/huan-hu/.orcagym/tmp/93915E65_A25D_4200_B557_C9224A8C5C10.xml
```

关键差异：

| 项目 | mjlab `g1.xml` | OrcaLab runtime |
|---|---:|---:|
| `torso_link` mass | 7.818 | 9.598 |
| `waist_yaw_link` mass | 0.214 | 0.244 |
| `waist_roll_link` mass | 0.086 | 0.047 |
| pelvis pos z | 0.79 | 0.792999983 |
| shoulder pitch z | 0.24778 | 0.23778 |
| waist roll pos z | 0.044 | 0.035 |
| IMU site | `imu_in_pelvis` | `imu` + `secondary_imu` |
| joint damping | 0 default | 0.05 |
| joint armature | 0 default | 0.01 |
| joint frictionloss | 0 default | 0.2, wrist 0.1 |
| actuator | none, `nu=0` | 29 motor actuators |
| sensors | 4 | 95 |
| geom/contact | visual mesh + simplified collision | mesh/sphere/cylinder collision |

### Actuator Ctrl 参数

`g1.xml` 本身没有 actuator：

```text
nu = 0
```

mjlab 训练时由 Python `G1_ARTICULATION` 动态创建 MuJoCo position actuator：

```text
ctrl = target joint position
force = kp * (ctrl - qpos) - kd * qvel
force is clipped by forcerange
```

OrcaLab runtime XML 里已有 29 个 motor actuator，例如原始 hip roll 类似：

```xml
<motor name="g1_000_left_hip_roll"
       joint="g1_000_left_hip_roll_joint"
       ctrlrange="-88 88"/>
```

原始 motor 语义更接近 torque command，`ctrlrange` 直接限制输入。mjlab bridge 现在会在运行时把它改成 position actuator 语义。

### Geom / Collision

`mjlab g1.xml` 是 visual 和 collision 分离：

```text
total geoms: 71
visual/non-collision: 36
collision-enabled: 35
```

visual mesh 默认：

```xml
contype="0" conaffinity="0" group="2"
```

collision geom 主要是 capsule / sphere：

```xml
type="capsule"
condim="6"
group="3"
```

OrcaLab runtime XML 里机器人 geom 基本都是 collision-enabled：

```text
total geoms in captured runtime XML: 39
visual/non-collision: 0
collision-enabled: 39
```

常见属性：

```xml
contype="1"
conaffinity="1"
condim="3"
group="0"
```

所以 OrcaLab runtime 不是简单把 visual mesh 关掉碰撞，而是很多 mesh/sphere/cylinder 直接参与 collision。这会影响接触模型，即使 joint/controller 已经对齐，contact 仍然可能和 mjlab 训练环境不同。

## 项目结构

```text
orca_rl/
├── README.md
├── RSL_RL_RESTRUCTURE_REPORT.md
├── pyproject.toml
├── requirements.txt
├── third_party/
│   └── unitree_rl_mjlab/
└── orca_rl/
    ├── __init__.py
    ├── diagnostics.py
    ├── list_tasks.py
    ├── registry.py
    ├── run_train.py
    ├── run_play.py
    ├── run_eval.py
    ├── utils.py
    ├── managers/
    │   └── __init__.py
    ├── sensor/
    │   ├── __init__.py
    │   └── config.py
    ├── terrains/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── export.py
    │   └── generator.py
    ├── rsl_env/
    │   ├── __init__.py
    │   ├── action_mapper.py
    │   ├── batched_locomotion_task.py
    │   ├── curriculum.py
    │   ├── local_mjcf.py
    │   ├── math_utils.py
    │   ├── mjlab_policy.py
    │   ├── mjwarp_runtime.py
    │   ├── model_scanner.py
    │   ├── obs_builder.py
    │   ├── randomization.py
    │   ├── rendering.py
    │   ├── reward_manager.py
    │   ├── robot_configs.py
    │   ├── runtime_policy.py
    │   ├── scene_binding.py
    │   ├── scene_resolvers.py
    │   ├── termination_manager.py
    │   ├── terrain_runtime.py
    │   └── adapters/
    │       ├── __init__.py
    │       ├── factory.py
    │       └── vecenv.py
    └── tasks/
        └── velocity/
            ├── __init__.py
            ├── config_types.py
            ├── velocity_env_cfg.py
            ├── config/
            │   ├── g1/
            │   │   ├── __init__.py
            │   │   └── env_cfgs.py
            │   └── go2/
            │       ├── __init__.py
            │       └── env_cfgs.py
            └── mdp/
                ├── __init__.py
                ├── actions.py
                ├── observations.py
                ├── rewards.py
                └── terminations.py
```

### 入口

- `run_train.py`: 训练入口，加载 task / runner config，创建 vec env，启动 RSL-RL。
- `run_play.py`: 可视化 play 入口，支持 Orca checkpoint 和 mjlab checkpoint。
- `run_eval.py`: checkpoint rollout / 评估入口。
- `registry.py`: 注册任务名，如 `Unitree-G1-Flat`。
- `utils.py`: 配置加载、checkpoint 查找、runtime summary、OrcaGym 地址检查。
- `diagnostics.py`: 打印环境、配置、依赖和 runtime 信息。

### `rsl_env`

- `adapters/vecenv.py`: RSL-RL `VecEnv` 适配器，负责把 Orca task 包成 RSL-RL 期待的接口。
- `adapters/factory.py`: 创建 locomotion vec env。
- `batched_locomotion_task.py`: 多机器人 batched MuJoCo task，负责 step/reset/state/contact。
- `local_mjcf.py`: 从单机器人 MJCF 生成 batched local MJCF。
- `mjlab_policy.py`: Unitree/mjlab checkpoint loader、mjlab observation bridge、OrcaLab runtime model 对齐。
- `mjwarp_runtime.py`: 实验性 MJWarp wrapper。
- `scene_binding.py`: G1 / GO2 scene binding、local XML fallback、auto-spawn 配置。
- `model_scanner.py`: 扫描 OrcaLab scene 中是否存在完整机器人实例。
- `action_mapper.py`: Orca RL 原生 residual joint target 到 PD torque 的映射。
- `obs_builder.py`: actor / privileged critic observation 构造。
- `reward_manager.py`: locomotion reward。
- `termination_manager.py`: too low、tilt、contact 等终止条件。
- `randomization.py`: domain randomization。
- `terrain_runtime.py`: rough terrain runtime sampling / hfield 交互。
- `rendering.py`: headless / human render mode 解析。
- `robot_configs.py`: GO2 等机器人静态配置。
- `runtime_policy.py`: Orca RL checkpoint 推理加载。

### `tasks/velocity`

- `config_types.py`: velocity task dataclass 配置。
- `velocity_env_cfg.py`: 通用 velocity task 默认配置。
- `config/g1/env_cfgs.py`: G1 flat / rough 注册配置。
- `config/go2/env_cfgs.py`: GO2 flat / rough 注册配置。
- `mdp/actions.py`: MDP action placeholder / config helper。
- `mdp/observations.py`: observation placeholder / config helper。
- `mdp/rewards.py`: reward placeholder / config helper。
- `mdp/terminations.py`: termination placeholder / config helper。

### 其他

- `terrains/generator.py`: rough terrain heightfield 生成。
- `terrains/export.py`: terrain OBJ 导出命令。
- `terrains/config.py`: terrain 参数配置。
- `sensor/config.py`: contact / sensor config dataclass。
- `managers/__init__.py`: manager-style 结构占位。
- `RSL_RL_RESTRUCTURE_REPORT.md`: 早期 RSL-RL 接入 OrcaLab 的重构记录。

## 验证命令

```bash
/home/huan-hu/miniconda3/envs/orcalab/bin/python -m py_compile \
  orca_rl/rsl_env/mjlab_policy.py \
  orca_rl/run_play.py
```

```bash
/home/huan-hu/miniconda3/envs/orcalab/bin/python -m compileall orca_rl
git diff --check
```

## 当前未解决问题

### 1. MJWarp 兼容桥仍然慢

现在的 MJWarp 后端每个 control step 后同步回 CPU `MjData`，然后继续用 numpy observation / reward / contact。它证明了能跑，但不是最终 GPU 训练架构。

### 2. OrcaGymLocalEnv 不是 nworld tensor API

当前 API 围绕 CPU `MjData`：

```text
query_contact_simple
query_sensor_data
query_site_pos_and_quat
get_body_xpos_xmat_xquat
set_joint_qpos
set_joint_qvel
update_data
```

最终需要 OrcaGym 或本项目新增 `nworld` tensor API。

### 3. OrcaLab scene rough terrain 还不能自动导入

G1 local MJCF rough 已经是真 hfield collision，但 OrcaLab scene play 中还没有公开 API 可以把生成 terrain 作为 collision asset 上传、替换或临时导入。

### 4. OrcaLab viewer 主视口不能从 Python 自动跟随机器人

当前 `render()` 只通过 `UpdateLocalEnv(qpos, time)` 同步机器人状态，没有公开 `SetViewerCamera` / `FollowActor` / `SetCameraLookAt` 这类主视口 API。现有 `SetCameraSensorInfo`、`MakeCameraViewportActive` 更偏 camera sensor，不是用户看的主 viewer orbit camera。

需要 OrcaLab / OrcaGym 增加：

```text
SetViewerCamera
SetViewerLookAt
FollowActor
```

或者在 OrcaLab viewer UI 内部加 `Follow Robot` toggle。

### 5. mjlab checkpoint 在 OrcaLab play 仍可能受 contact model 影响

runtime patch 已对齐 joint / actuator controller，但 OrcaLab runtime 的 collision geom 仍和 mjlab 训练环境不同。OrcaLab runtime 中很多 mesh/sphere/cylinder geom 直接参与 collision，mjlab 则是 visual mesh + simplified capsule/sphere collision。

### 6. Terrain curriculum 还没有真正接上

当前有 metadata，但没有按成功/失败动态切 terrain level。可行方向是生成大 terrain grid，并在 reset 时移动 env origin。

### 7. 部分 domain randomization 还没做完

已实现 friction、base mass、base inertia scale、base COM offset、kp/kd scale、torque scale、action latency、push、solver params、contact params。

未完整实现：

```text
damping randomization
armature randomization
joint friction / passive loss
per-joint motor delay
motor bandwidth / low-pass filter
full inertia tensor randomization
foot geom / pair-level contact randomization
```

### 8. Contact sensor 还不是完整 manager 抽象

当前 illegal contact 能跑，但还没有完整 `ContactSensorCfg` manager，包括 pair filtering、force threshold、history buffer、per-foot force tensor 等。
