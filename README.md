# Orca RL

Orca RL 是一个把 RSL-RL 接到 OrcaLab / OrcaGym / MuJoCo 生态里的独立训练项目。当前主要目标是让 Unitree G1 / GO2 的速度跟踪任务可以用 RSL-RL 训练，并逐步靠近 mjlab / IsaacLab 的配置风格、任务结构和大规模并行训练体验。

当前支持：

- Unitree G1 flat velocity training
- Unitree G1 rough velocity training
- Unitree GO2 flat velocity scene binding
- Unitree GO2 rough velocity config metadata
- RSL-RL OnPolicyRunner 接入
- actor / privileged critic 观测
- residual joint target action
- local MuJoCo headless training
- G1 batched local MJCF 训练
- G1 local rough terrain physical hfield
- 实验性 `mujoco_warp` step backend
- OrcaLab scene play / debug 路径

## 当前结论

训练主路径现在是：

```text
RSL-RL
  -> OrcaRslRlVecEnv
  -> BatchedOrcaLocomotionTask
  -> OrcaGymLocalEnv
  -> local MuJoCo MjModel / MjData
```

G1 headless 训练默认不再要求手动打开 OrcaLab 场景，也不通过远端 gRPC 跑仿真。它会从本地 G1 MJCF 生成一个 batched XML，然后用 `OrcaGymLocalEnv` 在本地进程里跑 MuJoCo。

实验性 MJWarp 路径是：

```text
OrcaGymLocalEnv 加载本地 MuJoCo model
  -> orca_rl 拿到 _mjModel / _mjData
  -> mujoco_warp.step(...)
  -> 同步回 CPU MjData
  -> 复用现有 numpy obs / reward / contact
```

它能跑，但不是最终快路径。真正像 mjlab 一样快，需要单机器人 model + `mujoco_warp.put_data(nworld=num_envs)` + torch/warp 版 observation / reward / reset / contact，不应该每步同步回 CPU。

## 安装

在 OrcaLab Python 环境里安装：

```bash
pip install -r requirements.txt
```

`requirements.txt` 已经包含：

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

注意：`orca-gym 26.4.3` 仍声明固定依赖 `mujoco==3.5.0`。本项目现在按 MJWarp 实验路线使用 MuJoCo 3.8。当前 G1 local training 在 MuJoCo 3.8 下测试通过，但如果后面使用 OrcaGym 某些强依赖 3.5 的功能，需要重新验证。

## 快速命令

列出注册任务：

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

训练 G1 flat：

```bash
python -m orca_rl.run_train \
  --config Unitree-G1-Flat \
  --headless \
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

OrcaLab scene 可视化 play：

```bash
python -m orca_rl.run_play \
  --config Unitree-G1-Flat \
  --ckpt <path_to_checkpoint>
```

Unitree/mjlab 训练出的 G1 checkpoint 在 OrcaLab scene 中 play：

```bash
python -m orca_rl.run_play \
  --config Unitree-G1-Flat \
  --policy-backend mjlab \
  --ckpt third_party/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/<run>/model_<iter>.pt
```

如果不传 `--ckpt`，会自动寻找：

```text
third_party/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/*/model_*.pt
```

本地 MuJoCo play：

```bash
python -m orca_rl.run_play \
  --config Unitree-G1-Flat \
  --ckpt <path_to_checkpoint> \
  --local-mujoco
```

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

## 后端说明

### `orca_cpu`

默认训练后端。

```text
local MJCF
  -> OrcaGymLocalEnv
  -> mujoco.mj_step
  -> numpy obs/reward/contact
```

优点：

- 当前最稳定。
- G1 flat / rough smoke 通过。
- rough hfield 已经是真 MuJoCo collision geom。
- 与现有 OrcaGym query / scene binding 兼容。

缺点：

- 还是 CPU MuJoCo。
- 4096 环境靠 clone 到同一个 XML，不是 mjlab 那种 `nworld`。
- 大量 contact / observation / reward 仍然在 CPU / numpy。

### `mjwarp`

实验性后端。

```text
OrcaGymLocalEnv 初始化 model/data
  -> MjWarpRuntime
  -> mujoco_warp.step
  -> sync_to_cpu
  -> 原有 obs/reward/contact
```

优点：

- 已经证明 G1 flat / rough 可以通过 `mujoco_warp.step` 跑完 1 iteration。
- `frame_skip` step 已做 CUDA graph capture。
- 可以作为接入 MuJoCo 3.8 / MJWarp 的实验桥。

缺点：

- 每个 control step 后仍同步回 CPU。
- 观测、奖励、contact 仍然走原来的 numpy 代码。
- 小 batch 下可能比 CPU 慢。
- 不是 mjlab 的最终 GPU 架构。

### 未来 `mjwarp_nworld`

真正想要 mjlab 速度，应当新增独立后端：

```text
single robot MJCF
  -> mujoco_warp.put_model(model)
  -> mujoco_warp.put_data(model, data, nworld=num_envs)
  -> torch obs / reward / done / reset
  -> RSL-RL
```

这条路径不应该依赖 `OrcaGymLocalEnv` 的 CPU query API。OrcaGym 可以继续用于 play、asset 管理、scene 可视化，但训练热路径要避免 CPU 同步。

## Policy ABI

训练出来的 policy 是否能放回 OrcaGym play，取决于 policy ABI 是否一致。

必须保持一致的内容：

- actor observation shape
- actor observation 顺序
- observation scale
- body-frame / world-frame 坐标变换
- quaternion 格式，当前是 MuJoCo `wxyz`
- action 维度
- joint / actuator 顺序
- residual joint target 到 PD torque 的映射
- command 表示
- rough height scan 维度和语义

当前 G1 flat actor observation：

```text
base_ang_vel_body
projected_gravity
command
joint_pos_rel
joint_vel
last_action
```

Unitree/mjlab G1 flat policy 不是这个旧 Orca 观测。它的 actor observation 是 98 维：

```text
base_ang_vel_body
projected_gravity
command
phase(sin, cos)
joint_pos_rel
joint_vel
last_action
```

所以 OrcaLab play 接 mjlab checkpoint 时必须加：

```bash
--policy-backend mjlab
```

这个后端会：

```text
1. 直接读取 Unitree/mjlab RSL-RL checkpoint 里的 actor_state_dict
2. 构造 mjlab 的 98 维 G1 actor observation
3. 使用 mjlab 的 action_scale = 0.25 * effort / stiffness
4. 使用 mjlab 的 G1 stiffness / damping / effort limit 把 residual joint target 转成 motor torque
5. 把 torque 写回 OrcaLab / OrcaGym 的 G1 motor actuator
```

也就是说，训练热路径可以完全走 `unitree_rl_mjlab`，可视化和场景调试可以回到 OrcaLab play。

G1 rough actor observation 额外包含：

```text
height_scan
```

G1 action 是 29 维，顺序来自：

```text
G1_JOINT_SUFFIXES
G1_ACTUATOR_SUFFIXES
```

只要未来 MJWarp nworld 后端严格复刻这些 ABI，训练出的 actor 就可以放回 OrcaGym play。

## 项目结构

```text
orca_rl/
├── README.md
├── RSL_RL_RESTRUCTURE_REPORT.md
├── TODO_headless_rendering.md
├── pyproject.toml
├── requirements.txt
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
        ├── __init__.py
        └── velocity/
            ├── __init__.py
            ├── config_types.py
            ├── velocity_env_cfg.py
            ├── config/
            │   ├── __init__.py
            │   ├── g1/
            │   │   ├── __init__.py
            │   │   ├── env_cfgs.py
            │   │   └── rl_cfg.py
            │   └── go2/
            │       ├── __init__.py
            │       ├── env_cfgs.py
            │       └── rl_cfg.py
            └── mdp/
                ├── __init__.py
                ├── actions.py
                ├── commands.py
                ├── curriculums.py
                ├── events.py
                ├── observations.py
                ├── rewards.py
                ├── terminations.py
                └── terrain_utils.py
```

## 根目录文件

### `README.md`

项目主说明文档。说明当前后端、训练命令、项目结构、已实现功能和未解决问题。

### `RSL_RL_RESTRUCTURE_REPORT.md`

早期 RSL-RL 接入 OrcaLab 的重构报告。记录过渡版本、接口设计、观测/action 对齐思路，以及为什么后面切到本地 MJCF 路线。

### `TODO_headless_rendering.md`

headless/no-rendering 任务记录。现在主要作为历史 TODO 和完成状态记录。

### `requirements.txt`

运行依赖。当前包含 MuJoCo 3.8、MJWarp、Warp、RSL-RL、Torch、TensorDict、ONNX、TensorBoard、W&B。

### `pyproject.toml`

Python 包元数据。定义 `orca-rl` 包名、依赖、setuptools 包发现规则。

### `.gitignore`

Git 忽略规则。避免日志、缓存、构建产物进入仓库。

## 顶层 Python 入口

### `orca_rl/__init__.py`

顶层懒加载入口。导出：

- `OrcaRslRlVecEnv`
- `make_locomotion_vec_env`
- `load_task_and_train_cfg`
- `print_runtime_summary`
- `list_tasks`
- `get_task_spec`

### `orca_rl/run_train.py`

训练入口。负责：

- 解析 CLI 参数。
- 加载 task config 和 RSL-RL runner config。
- 应用 `--num-envs`、`--headless`、`--render`、`--remote`、`--logger`、`--wandb` 等覆盖。
- 支持 `--sim-backend orca_cpu|mjwarp`。
- 创建 RSL-RL VecEnv。
- 创建 `OnPolicyRunner`。
- 执行 `runner.learn()`。
- 保存 `model_last.pt` 和 `model_final.pt`。
- 导出 ONNX / JIT policy。

### `orca_rl/run_play.py`

play / 可视化入口。负责：

- 加载 checkpoint。
- 默认使用 OrcaLab scene 路径。
- G1 play 默认关闭 `local_xml_path`，让 scene binding 回到 OrcaLab actor 申请/绑定方式。
- 支持 `--local-mujoco`，强制使用本地 MJCF play。
- 按 `control_dt` 实时 sleep，便于视觉检查策略。

### `orca_rl/run_eval.py`

评估入口。负责：

- 加载 checkpoint。
- headless 运行固定步数。
- 统计平均 reward 和 done 数量。

### `orca_rl/list_tasks.py`

小 CLI。打印当前注册任务列表。

### `orca_rl/registry.py`

任务注册表。负责：

- `register_task`
- `list_tasks`
- `get_task_spec`
- `load_registered_task`

内置注册：

- `Unitree-G1-Flat`
- `Unitree-G1-Rough`
- `Unitree-GO2-Flat`
- `Unitree-GO2-Rough`

### `orca_rl/diagnostics.py`

训练/play/eval 启动前的 runtime summary。打印：

- 任务名
- robot
- device / GPU
- num_envs
- num_actions
- max episode length
- control dt
- sim backend
- obs 维度
- action 参数
- reward / termination / command / randomization / terrain / sensor / curriculum
- scene binding source
- generated XML path

### `orca_rl/utils.py`

通用工具。负责：

- 项目根目录加入 `sys.path`
- 加载 Python config
- 解析 `file.py:factory_name`
- 加载注册任务
- 应用 remote override
- W&B 参数覆盖
- 检查 OrcaGym 地址
- local MJCF headless 时跳过 gRPC 检查
- 创建 log dir
- 查找最新 checkpoint
- 保存 checkpoint alias

## RSL-RL 环境层

### `orca_rl/rsl_env/adapters/factory.py`

环境工厂。`make_locomotion_vec_env()` 会返回 `OrcaRslRlVecEnv`。

### `orca_rl/rsl_env/adapters/vecenv.py`

RSL-RL `VecEnv` 适配器。负责：

- 接收 task config。
- 解析 headless/render mode。
- 调 scene binding resolver。
- 创建一个或多个 `BatchedOrcaLocomotionTask`。
- 拼接多个 simulator group 的 obs/reward/done。
- 把 numpy 结果转成 torch `TensorDict`。
- 聚合 log extras。

### `orca_rl/rsl_env/batched_locomotion_task.py`

当前训练核心。它继承 `OrcaGymLocalEnv`，一个实例里同时管理多个 robot agent。

负责：

- 加载 local MJCF 或 OrcaLab scene。
- 初始化 agent runtime。
- 解析 qpos/qvel/actuator/body/site/sensor offset。
- 批量 action -> torque。
- 调 MuJoCo step。
- 读取 state。
- 构造 obs。
- 计算 reward。
- 计算 termination。
- reset done env。
- local domain randomization。
- rough terrain runtime 对齐。
- 可选 `mjwarp` step backend。

### `orca_rl/rsl_env/mjwarp_runtime.py`

实验性 MJWarp 桥。负责：

- 从已有 `MjModel/MjData` 创建 `mjwarp.Model/Data`。
- 暴露 qpos/qvel/ctrl 等 torch view。
- 用 `mujoco_warp.step()` 跑 GPU physics。
- 捕获 `frame_skip` step CUDA graph。
- 将 GPU state 同步回 CPU `MjData`。

当前限制：

- 只是 `OrcaGymLocalEnv` 兼容桥。
- `nworld` 当前固定为 1。
- 不是最终 mjlab nworld 快路径。

### `orca_rl/rsl_env/action_mapper.py`

action 映射。负责：

- policy action clip。
- `[-1, 1]` action 到 residual joint target。
- joint limit safety scale。
- `max_delta` 限制。
- PD torque 计算。
- torque limit clip。

### `orca_rl/rsl_env/obs_builder.py`

观测构造。负责：

- `LocomotionTaskState` 数据结构。
- actor obs 拼接。
- privileged critic obs 拼接。
- body frame velocity。
- projected gravity。
- command scale。
- dof pos/vel scale。
- height scan padding。
- actor observation noise。

### `orca_rl/rsl_env/reward_manager.py`

速度任务 reward。负责：

- linear velocity tracking。
- yaw velocity tracking。
- z velocity penalty。
- orientation penalty。
- base height penalty。
- torque penalty。
- action rate penalty。
- joint limit penalty。
- foot slip penalty。
- feet air time。
- foot clearance。
- body angular velocity。
- stand still。
- joint deviation。
- termination penalty。

### `orca_rl/rsl_env/termination_manager.py`

termination 判断。负责：

- too low。
- too high。
- too tilted。
- invalid state。
- base contact。
- illegal contact。

### `orca_rl/rsl_env/curriculum.py`

命令采样。当前主要是 `FlatVelocityCommandSampler`：

- 采样 `lin_vel_x`
- 采样 `lin_vel_y`
- 采样 `yaw_vel`
- 支持 command resample interval

### `orca_rl/rsl_env/randomization.py`

domain randomization。负责：

- friction scale。
- base mass delta。
- base inertia scale。
- base COM offset。
- kp scale。
- kd scale。
- torque scale。
- action delay steps。
- push velocity。
- solver iterations。
- solver tolerance。
- contact solref/solimp/margin scale。

### `orca_rl/rsl_env/terrain_runtime.py`

Python 侧 terrain runtime。负责：

- 从 task cfg 生成 heightfield。
- height at `(x, y)` 查询。
- yaw-aware height scan。
- 导出 terrain mesh。
- rough physical terrain 已编入 MJCF 时禁止 reset-time resample，避免物理/观测不一致。

### `orca_rl/rsl_env/local_mjcf.py`

本地 MJCF 生成器。负责：

- 找到 G1 source XML。
- clone robot root body。
- prefix body/joint/actuator/site/sensor 名字。
- rewrite XML references。
- 删除 keyframe。
- 生成 batched local XML。
- rough 时加入 MuJoCo `hfield` collision geom。
- 移除旧 floor/plane。
- 根据 `num_envs` 调整 terrain 覆盖范围。
- 给 rough XML 加 terrain hash 后缀。

### `orca_rl/rsl_env/scene_binding.py`

robot scene binding。负责：

- G1 / GO2 scene resolver。
- G1 local XML path resolver。
- G1 local MJCF batch 生成。
- OrcaLab scene 扫描。
- G1 auto-publish。
- robot config 组装。

### `orca_rl/rsl_env/scene_resolvers.py`

resolver alias 映射。负责：

- `"g1"` -> `resolve_g1_scene_binding`
- `"go2"` -> `resolve_go2_scene_binding`
- 支持自定义 import path resolver。

### `orca_rl/rsl_env/model_scanner.py`

OrcaLab scene 模型扫描。负责：

- 根据 suffix template 扫描 complete robot match。
- 判断 joints / actuators / sites / sensors 是否完整。
- 在 scene 缺 robot 时给提示。

### `orca_rl/rsl_env/robot_configs.py`

GO2 robot config。包含：

- base joint 名字。
- leg joint 名字。
- actuator 名字。
- contact site 名字。
- foot body 名字。
- motor 参数。
- action scale。

G1 config 目前主要在 `scene_binding.py` 内生成，因为 G1 joint/action 列表比较长。

### `orca_rl/rsl_env/runtime_policy.py`

policy runtime 工具。负责：

- 加载 RSL-RL inference runner。
- 导出 JIT。
- 导出 ONNX。
- 保存 inference policy。

### `orca_rl/rsl_env/rendering.py`

渲染模式解析。负责：

- headless 默认值。
- `render_mode` 规范化。
- `human` / `none` 区分。

### `orca_rl/rsl_env/math_utils.py`

数学工具。负责：

- MuJoCo `wxyz` quaternion 到 rotation matrix。
- yaw quaternion。
- quaternion multiply。
- safe clip。

## Task / Config 层

### `orca_rl/tasks/velocity/config_types.py`

mjlab / IsaacLab 风格的配置 dataclass。定义：

- `SceneEntityCfg`
- `TermCfg`
- `EventTermCfg`
- `ObservationTermCfg`
- `ObservationGroupCfg`
- `UniformVelocityCommandCfg`
- `JointPositionActionCfg`
- `LocomotionEnvCfg`
- RSL-RL actor/critic/algorithm/runner cfg

同时负责把结构化配置转成当前 runtime 使用的 legacy dict。

### `orca_rl/tasks/velocity/velocity_env_cfg.py`

velocity task 通用配置工厂。负责：

- flat velocity base config。
- rough velocity base config。
- actor/critic observation term。
- reward term。
- termination term。
- reset event。
- randomization event。
- terrain scan metadata。
- nonfoot contact metadata。
- rough reward/termination/curriculum metadata。

### `orca_rl/tasks/velocity/config/g1/env_cfgs.py`

G1 env config。负责：

- G1 flat config。
- G1 rough config。
- G1 action max delta。
- G1 scene binding。
- G1 reset noise。
- G1 base height / tilt limit。
- G1 reward scale。
- G1 local XML 默认开启。

### `orca_rl/tasks/velocity/config/g1/rl_cfg.py`

G1 RSL-RL PPO runner config。负责：

- actor MLP 结构。
- critic MLP 结构。
- Gaussian distribution。
- PPO 超参。
- experiment name。
- run name。
- save interval。
- rollout length。
- max iteration。

### `orca_rl/tasks/velocity/config/go2/env_cfgs.py`

GO2 env config。负责：

- GO2 flat config。
- GO2 rough config metadata。
- GO2 action max delta。
- GO2 scene binding。
- GO2 reward scale。
- GO2 rough physical terrain 当前关闭，避免 scene-backed 路径误以为有 local hfield。

### `orca_rl/tasks/velocity/config/go2/rl_cfg.py`

GO2 RSL-RL PPO runner config。结构同 G1。

## MDP 声明层

### `orca_rl/tasks/velocity/mdp/actions.py`

动作 term 占位/声明。用于配置 metadata，运行时由 `action_mapper.py` 实现。

### `orca_rl/tasks/velocity/mdp/commands.py`

命令 term 占位/声明。用于配置 metadata，运行时由 `curriculum.py` 采样。

### `orca_rl/tasks/velocity/mdp/observations.py`

观测 term 占位/声明。用于配置 metadata，运行时由 `obs_builder.py` 实现。

### `orca_rl/tasks/velocity/mdp/rewards.py`

reward term 占位/声明。用于配置 metadata，运行时由 `reward_manager.py` 实现。

### `orca_rl/tasks/velocity/mdp/terminations.py`

termination term 占位/声明。用于配置 metadata，运行时由 `termination_manager.py` 实现。

### `orca_rl/tasks/velocity/mdp/events.py`

event / randomization term 占位/声明。用于配置 metadata，运行时由 `randomization.py` 和 reset 逻辑实现。

### `orca_rl/tasks/velocity/mdp/curriculums.py`

curriculum term 占位/声明。当前 terrain level metadata 已有，但真实 progression 还未接上。

### `orca_rl/tasks/velocity/mdp/terrain_utils.py`

terrain 相关 MDP 工具占位/辅助。用于 rough config 的 terrain metadata 对齐。

## Sensor 层

### `orca_rl/sensor/config.py`

sensor dataclass。定义：

- `ContactMatchCfg`
- `ContactSensorCfg`
- `GridPatternCfg`
- `RayCasterCfg`

这些当前主要是配置 metadata。实际 contact / height scan 由 runtime 手写逻辑实现。

## Terrain 层

### `orca_rl/terrains/config.py`

terrain config dataclass。定义：

- `SubTerrainCfg`
- `TerrainGeneratorCfg`
- `TerrainCfg`

### `orca_rl/terrains/generator.py`

heightfield 生成器。负责：

- plane heightfield。
- random uniform。
- pyramid stairs。
- discrete obstacles。
- wave terrain。
- height interpolation。
- height scan sampling。
- heightfield 转 OBJ mesh。

### `orca_rl/terrains/export.py`

terrain 导出 CLI。把当前任务配置中的 terrain 导出成 OBJ，并打印 heightfield shape / resolution / origin。

## Managers 目录

### `orca_rl/managers/__init__.py`

当前是占位包。项目曾经向 mjlab manager-style config 靠拢，但实际运行时 manager 目前分散在 `rsl_env` 的 action / obs / reward / termination / randomization 模块里。

## 训练数据流

### CPU local 训练

```text
run_train.py
  -> load_task_and_train_cfg
  -> make_locomotion_vec_env
  -> OrcaRslRlVecEnv
  -> resolve_g1_scene_binding
  -> build_local_mjcf_batch
  -> BatchedOrcaLocomotionTask
  -> OrcaGymLocalEnv.initialize_simulation
  -> mujoco.MjModel.from_xml_path
  -> runner.learn
```

每步：

```text
policy(obs)
  -> action mapper
  -> PD torque
  -> MuJoCo ctrl
  -> mujoco.mj_step
  -> update_data
  -> query contact / foot pos
  -> read state
  -> reward / done
  -> observation builder
  -> RSL-RL update
```

### MJWarp 实验训练

```text
BatchedOrcaLocomotionTask
  -> OrcaGymLocalEnv creates _mjModel/_mjData
  -> MjWarpRuntime creates wp_model/wp_data
  -> mujoco_warp.step
  -> sync_to_cpu
  -> existing obs/reward/contact
```

这条路径只是实验桥。它证明 G1 XML 和 rough hfield 能在 MJWarp 中跑，但由于 CPU 同步和 numpy 逻辑还在，暂时不是高性能训练方案。

## 当前测试过的命令

CPU local：

```bash
python -m orca_rl.run_train \
  --config Unitree-G1-Flat \
  --headless \
  --num-envs 2 \
  --num-iterations 1
```

MJWarp flat：

```bash
python -m orca_rl.run_train \
  --config Unitree-G1-Flat \
  --headless \
  --num-envs 2 \
  --num-iterations 1 \
  --sim-backend mjwarp
```

MJWarp rough：

```bash
python -m orca_rl.run_train \
  --config Unitree-G1-Rough \
  --headless \
  --num-envs 2 \
  --num-iterations 1 \
  --sim-backend mjwarp
```

静态检查：

```bash
python -m compileall orca_rl
git diff --check
```

## 最后碰到但暂时不能解决的问题

### 1. MJWarp 兼容桥收集速度慢

现象：

```text
orca_cpu 小 batch 约 100 SPS
mjwarp 兼容桥小 batch 约 45 SPS
```

原因：

```text
mujoco_warp.step 在 GPU 上跑
  -> 每个 control step 同步回 CPU MjData
  -> CPU 上跑 observation / reward / contact / reset
```

这会抵消 GPU step 的收益。CUDA graph 已经加了，但只能减少 kernel launch 开销，不能消除 CPU 同步。

真正解决方案：

```text
single G1 MJCF
  -> mujoco_warp.put_data(nworld=num_envs)
  -> torch/warp obs
  -> torch/warp reward
  -> torch/warp contact
  -> torch/warp reset
```

这相当于新增 mjlab-style backend，而不是继续 patch `OrcaGymLocalEnv`。

### 2. OrcaGymLocalEnv 不是 GPU nworld API

当前 `OrcaGymLocalEnv` 的 API 围绕 CPU `MjData`：

```text
query_contact_simple
query_sensor_data
query_site_pos_and_quat
get_body_xpos_xmat_xquat
set_joint_qpos
set_joint_qvel
update_data
```

这些都天然要求 CPU 数据。要让训练真正 GPU 化，最好在 OrcaGym 内部新增：

```text
backend="mjwarp"
nworld=num_envs
query_qpos_tensor
query_qvel_tensor
query_body_xpos_tensor
query_sensor_tensor
query_contact_tensor
set_ctrl_tensor
reset_envs
```

否则 `orca_rl` 在外面偷拿 `_mjModel/_mjData` 做 GPU step，会一直是过渡方案。

### 3. 完全 mjlab 化后 policy ABI 必须严格对齐

如果未来新增 `mjwarp_nworld` 后端，必须保证训练出的 policy 能放回 OrcaGym play。

需要锁死：

```text
obs shape
obs order
obs scale
action dim
joint order
actuator order
PD mapping
command format
rough height_scan dim
```

否则训练能跑，但 play 时动作会错位或观测不匹配。

### 4. OrcaLab scene rough terrain 还不能自动导入

G1 local MJCF rough 已经是真物理 hfield。但 OrcaLab scene play 里还没有发现公开 API 可以把生成 terrain 作为 collision asset 上传/替换。

需要 OrcaLab 支持其中一种：

```text
AddHeightField
AddCollisionMesh
ReplaceTerrain
temporary asset import + AddActor
supported scene XML patch/reload
```

否则 rough 训练和 OrcaStudio 可视化 terrain 不能完全一致。

### 5. Terrain curriculum 没有真正接上

当前有 `terrain_levels` metadata，但没有真正按成功/失败切换 terrain level。

原因：

```text
local hfield 是编译进 MJCF 的
reset 时不能随便换物理 terrain
```

可行方案：

```text
生成大 terrain grid
每个 env reset 到不同 origin
按成功/失败移动 origin level
```

这需要新的 terrain origin manager。

### 6. GO2 rough 还不是 physical local terrain

G1 有 local XML 和 hfield path。GO2 当前主要是 scene-backed resolver，rough config 有 metadata，但 physical terrain 默认关闭，避免误以为 scene 里已有同一块地形。

要补 GO2，需要：

```text
GO2 local MJCF source
GO2 local batch builder support
GO2 rough hfield insertion
GO2 foot/body/contact 名称对齐
```

### 7. 一些 domain randomization 还没做完

已实现：

```text
friction
base mass
base inertia scale
base COM offset
kp/kd scale
torque scale
action latency
push
solver params
contact params
```

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

当前 illegal contact 能跑，但还不是完整 `ContactSensorCfg` manager。

缺口：

```text
named pair filtering
force threshold abstraction
history buffer
per-foot force tensor
non-foot ground contact force tensor
```

未来 mjwarp_nworld 后端尤其需要把 contact tensor 化。

### 9. 真正的 OrcaLab raycast 还没接

当前 height scan 用 Python heightfield sampling。它和 local hfield 生成配置一致，所以训练可用。

但如果在 OrcaLab scene play 中使用真实 terrain，最好有 OrcaLab raycast 或 MJWarp raycast 来确认：

```text
观测 height_scan
物理 collision terrain
可视化 terrain
```

三者一致。

### 10. 训练速度最终问题不能靠当前桥彻底解决

现在的问题不是单纯“把 `mj_step` 换成 `mujoco_warp.step`”。

真正瓶颈是：

```text
数据结构还是 CPU Env
obs/reward/contact 还是 numpy
机器人并行还是 batched XML clone
不是 nworld
```

所以最终方向必须是：

```text
保留 OrcaGym 后端用于 play/debug
新增 mjlab-style training backend
或者推动 OrcaGymLocalEnv 原生支持 MJWarp nworld tensor API
```
