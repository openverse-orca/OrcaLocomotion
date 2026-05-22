# Orca RL

Orca RL 是一个把 RSL-RL 接入 OrcaLab / OrcaGym / MuJoCo 生态的训练与播放项目。当前重点是 Unitree G1 / GO2 的速度跟踪训练、headless 本地 MuJoCo 训练、Unitree/mjlab checkpoint 回放到 OrcaLab scene，以及实验性 MJWarp 接入。

核心目标：

| 目标 | 说明 |
|---|---|
| RSL-RL 训练工具链接入 OrcaGym | 让 RSL-RL 的 PPO runner 能通过标准 VecEnv 接口训练 G1 / GO2 velocity task |
| Headless 本地训练 | 训练热路径默认不依赖 OrcaLab viewport，不把大批量 actor 发布进交互场景 |
| mjlab checkpoint 回放 | 让 Unitree/mjlab 训练出的 G1 / GO2 `.pt` checkpoint 可以直接在 OrcaLab scene 里 play |
| 后端演进 | 保留稳定 CPU MuJoCo 后端，同时探索 MJWarp / nworld tensor 化训练路线 |

当前可用能力：

- Unitree G1 flat / rough velocity task
- Unitree GO2 flat / rough velocity task
- RSL-RL `OnPolicyRunner` 训练入口
- 本地 MuJoCo headless training
- `--headless` / `--no-render` 无渲染训练模式
- G1 batched local MJCF 训练
- G1 rough physical hfield terrain
- 实验性 `mujoco_warp` step backend
- OrcaLab scene play / debug
- Unitree/mjlab G1 / GO2 checkpoint 在 OrcaLab scene 中 play

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

训练时的批量设计：

```text
MuJoCo mjModel
  ├── g1_000 / go2_000
  ├── g1_001 / go2_001
  └── ...
      ↓
一次 mj_step 推进整个模型
      ↓
按 qpos/qvel/actuator/contact offset 切回 RSL-RL logical env
```

旧实现里 `num_envs=4096` 会更接近 4096 个 Python task wrapper、4096 次 local-env 初始化和大量串行 step 调用。现在 `BatchedOrcaLocomotionTask` 把同一个 simulator group 里的机器人放进一个 MuJoCo model，RSL-RL 仍然看到 `(num_envs, num_actions)` 的标准向量环境。

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

本仓库包含两个用于 OrcaLab play smoke 的 mjlab checkpoint，使用 Git LFS 存储：

```text
checkpoints/test_model_G1_mjlab_Flat.pt
checkpoints/test_model_Go2_mjlab_Flat.pt
```

首次 clone 前建议先安装 Git LFS：

```bash
sudo apt-get update
sudo apt-get install -y git-lfs
git lfs install
git clone <repo-url>
cd orca_rl
git lfs pull
```

如果已经 clone 了仓库，但 `.pt` 文件只是 LFS pointer 或缺失：

```bash
git lfs install
git lfs pull
```

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
  --checkpoint <path_to_checkpoint>
```

OrcaLab scene 中播放 Unitree/mjlab G1 checkpoint：

```bash
python -m orca_rl.run_play \
  --config Unitree-G1-Flat \
  --policy-backend mjlab \
  --checkpoint ./checkpoints/test_model_G1_mjlab_Flat.pt \
  --lin-vel-x 0.5 \
  --lin-vel-y 0.0 \
  --ang-vel-z 0.0
```

不要加 `--local-mujoco`，这样才会播放到 OrcaLab scene 里的 G1。若不传 `--checkpoint`，会自动寻找：

```text
third_party/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/*/model_*.pt
```

G1 scene binding / auto-publish 默认使用 OrcaLab 资产：

```text
assets/e071469a36d3c8aa/unitree_robots/prefabs/g1_29dof_usda
```

OrcaLab scene 中播放 Unitree/mjlab GO2 checkpoint：

```bash
python -m orca_rl.run_play \
  --config Unitree-GO2-Flat \
  --policy-backend mjlab \
  --checkpoint ./checkpoints/test_model_Go2_mjlab_Flat.pt \
  --lin-vel-x 0.5 \
  --lin-vel-y 0.0 \
  --ang-vel-z 0.0
```

若不传 `--checkpoint`，GO2 配置会自动寻找：

```text
third_party/unitree_rl_mjlab/logs/rsl_rl/go2_velocity/*/model_*.pt
```

GO2 scene binding / auto-publish 默认使用 OrcaLab 资产：

```text
assets/e071469a36d3c8aa/unitree_robots/prefabs/go2_usda
```

OrcaLab 可上传 primitive 地形资产：

```text
smb://192.168.110.53/share/OrcaPrimitiveTerrainXml.zip
```

上传项目名建议填：

```text
OrcaPrimitiveTerrainXml
```

本地文件：

```text
assets/terrain/orca_primitive_terrain_xml/terrain.xml
assets/terrain/orca_primitive_terrain_xml/terrain_height_field.npz
assets/terrain/OrcaPrimitiveTerrainXml.zip
```

这份资产只使用 MuJoCo primitive geom：

```text
plane / box / cylinder
```

OrcaLab 的 XML 上传链路目前可以接受 primitive geom，但不接受 MuJoCo `hfield` 高度图。所以本地 MuJoCo play 默认也改成同一份 `terrain.xml`，并用 `terrain_height_field.npz` 做 height scan 对齐。用本地 MuJoCo play 覆盖地形：

```bash
python -m orca_rl.run_play \
  --config Unitree-Go2-Flat \
  --policy-backend mjlab \
  --checkpoint checkpoints/test_model_Go2_mjlab_Flat.pt \
  --local-terrain-map \
  --lin-vel-x 0.5 \
  --command-arrow
```

`--local-terrain-map` 会自动打开 `--local-mujoco`，并把本地生成的 robot batch XML 地面替换成这份 OrcaLab 也能导入的 primitive terrain。注意这仍然是简化版静态地形，不是 mjlab 完整课程学习地形系统。

OrcaLab command arrow debug：

```bash
python -m orca_rl.run_play \
  --config Unitree-GO2-Flat \
  --policy-backend mjlab \
  --checkpoint ./checkpoints/test_model_Go2_mjlab_Flat.pt \
  --lin-vel-x 0.5 \
  --lin-vel-y 0.0 \
  --ang-vel-z 0.0 \
  --command-arrow \
  --command-arrow-scale 0.55 \
  --heading-arrow-scale 0.5 \
  --command-arrow-actor cmd_arrow_000
```

`--command-arrow` 默认会在机器人 auto-publish 时把正式 OrcaLab 箭头资产同批发布进 scene，避免单独发布箭头覆盖现有机器人场景：

```text
assets/001d46537b9e555b/commandarrow/prefabs/command_arrow_usda
```

默认会同时绑定两根 debug arrow：

| 箭头 | 默认 actor | 颜色 | 含义 |
|---|---|---|---|
| command arrow | `cmd_arrow_000` | cyan | 理想上层速度命令方向，body-frame command 转到 world |
| velocity arrow | `heading_arrow_000` | orange | 机器人当前实际 base linear velocity 的 world XY 方向 |

orange velocity arrow 仍复用原 heading arrow 资产，正式资产默认按下面路径查找：

```text
assets/001d46537b9e555b/heading_arrow/prefabs/heading_arrow_usda
```

如果平台生成的 orange arrow asset 路径不同，运行时显式传：

```bash
--heading-arrow-asset <generated_heading_arrow_asset_path>
```

如果只想更新已经手动放好的箭头，不让 `run_play` 自动发布：

```bash
--no-command-arrow-auto-publish
```

如果只想看 command arrow，不显示机器人当前朝向：

```bash
--no-heading-arrow
```

箭头尺寸是在 OrcaLab scene publish 时写入 actor scale 的，不是每帧 qpos 更新的一部分；已经存在于 scene 里的旧箭头不会因为 CLI 参数自动变小。默认尺寸已经调小为：

```text
command arrow scale = 0.55
velocity arrow scale = 0.50
```

如果仍然遮挡机器人，可以重新发布 scene 时继续缩小：

```bash
--command-arrow-scale 0.35 --heading-arrow-scale 0.32
```

如果只是想测试“twist 幅值变化时策略和 debug arrow 的响应”，可以打开命令扫描。它会把命令从 0 平滑扫到指定 twist，再扫回 0：

```bash
--lin-vel-x 0.5 --ang-vel-z 0.6 --command-sweep --command-sweep-period 6.0
```

当前 debug arrow 使用旧的固定 mesh 箭头，只更新 freejoint 的位置和方向。两个箭头的 XY 跟随点已经对齐，只用 Z 分层：

```text
command arrow: base position + (0, 0, command_arrow_z)
velocity arrow: base position + (0, 0, command_arrow_z + 0.15)
```

已上传并恢复为旧固定箭头的调试包：

```text
smb://192.168.110.53/share/CommandArrow.zip
smb://192.168.110.53/share/HeadingArrow.zip
```

本地调试包仍保留在：

```text
assets/debug/command_arrow.usdz
assets/debug/command_arrow.usda
assets/debug/heading_arrow.usdz
assets/debug/heading_arrow.usda
assets/debug/command_arrow_mjcf.xml
assets/debug/arrow_x.usd
```

使用协议：

```text
asset local +X = arrow forward
actor name     = cmd_arrow_000 / heading_arrow_000
joint type     = freejoint / 6DoF
collision      = off
```

其中 `assets/debug/CommandArrow.zip` 和 `assets/debug/HeadingArrow.zip` 是上传生成正式 OrcaLab 箭头资产的 USDZ 源包；`command_arrow_mjcf.xml` 是带 freejoint 的 MuJoCo/Orca wrapper 示例：

```xml
<body name="cmd_arrow_000" pos="0 0 0">
  <freejoint name="cmd_arrow_000_freejoint"/>
  <geom name="cmd_arrow_000_visual" type="mesh" mesh="command_arrow_mesh" contype="0" conaffinity="0"/>
</body>
```

`run_play` 会自动尝试寻找 `cmd_arrow_000_base_joint`、`cmd_arrow_000_freejoint`、`cmd_arrow_000_joint`、`command_arrow_freejoint`、`arrow_x_freejoint` 等常见 joint 名。如果 OrcaLab 里实际 joint 名不同，显式传：

```bash
--command-arrow-joint <freejoint_name>
--heading-arrow-joint <freejoint_name>
```

当前旧固定箭头 pivot 不在箭尾，默认用 `--command-arrow-tail-x -0.25` 做补偿：

```bash
--command-arrow-tail-x -0.25
```

GO2 样例脚本默认走 OrcaLab play：

```bash
./play_go2_balance.sh walk
./play_go2_balance.sh stand
./play_go2_balance.sh spin
```

同一个脚本加 `--mjlab` 可以直接用 vendored mjlab 原生 viewer replay，用来和 OrcaLab play 对比：

```bash
./play_go2_balance.sh walk --mjlab
./play_go2_balance.sh spin --mjlab --viewer native
```

本地 MuJoCo play：

```bash
python -m orca_rl.run_play \
  --config Unitree-G1-Flat \
  --checkpoint <path_to_checkpoint> \
  --local-mujoco
```

`run_play` 默认保留 human rendering，用于看策略动作；`--local-mujoco` 只作为不接 OrcaLab scene 的调试路径。

评估 checkpoint：

```bash
python -m orca_rl.run_eval \
  --config Unitree-G1-Flat \
  --checkpoint <path_to_checkpoint> \
  --steps 2000
```

导出 rough terrain OBJ：

```bash
python -m orca_rl.terrains.export \
  --config Unitree-G1-Rough \
  --out generated_terrains/g1_rough.obj
```

## Headless / No-Rendering 训练

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

大批量训练推荐先看 runtime summary，确认没有退回 scene-backed workflow：

```text
num_envs: <requested count>
num_sim_groups: 1
headless: True
render_mode: none
source: local_mjcf
model_xml_path: /tmp/orca_rl_mjcf/...
```

如果 `source` 是 `orcalab_scene`，说明当前 run 正在使用 OrcaLab 交互场景。它适合 play / debug，但不是 G1 大批量 headless 训练的快路径。

## RSL-RL 接入细节

`OrcaRslRlVecEnv` 向 RSL-RL 暴露标准接口：

```text
num_envs
num_actions
max_episode_length
episode_length_buf
get_observations()
step(actions)
reset()
close()
```

训练循环看到的是：

```text
actions shape: (num_envs, num_actions)
rewards shape: (num_envs,)
dones shape: (num_envs,)
obs: TensorDict
extras: dict
```

observation 使用两个组：

```python
obs_groups = {
    "actor": ["policy"],
    "critic": ["policy", "privileged"],
}
```

actor policy 只读 `policy`，critic 可以额外读 `privileged`。

### Action Pipeline

Orca RL 原生 checkpoint 的 action 是归一化 residual joint-position action：

```text
policy action
  -> clip
  -> target_qpos around nominal pose
  -> safety-scaled joint limit clamp
  -> torque = kp * (target_qpos - qpos) - kd * qvel
  -> torque clamp by effort limit
  -> write full MuJoCo ctrl
  -> mj_step()
```

同一个 simulator group 内，target 和 torque 计算按 `(num_agents, num_actions)` 的 NumPy 数组批量处理，再一次性写入全局 `ctrl`。

### Observation Pipeline

每个 agent 从 MuJoCo 读取：

```text
base position / quaternion
base linear / angular velocity
joint position / velocity
velocity command
last action / last torque
foot position / velocity / contact
domain randomization state
optional height scan
```

actor observation 主要包括：

```text
base angular velocity
projected gravity
command
joint position relative to nominal pose
joint velocity
last action
optional height scan
```

privileged observation 额外包括 base linear velocity、base height、foot contacts、foot heights、foot velocities、last torque、domain randomization values 等。

### Reward / Termination Pipeline

flat task 使用：

```text
linear velocity tracking
yaw velocity tracking
vertical velocity penalty
orientation penalty
base height penalty
torque penalty
action-rate penalty
joint-limit penalty
foot-slip penalty
termination penalty
```

rough task 额外接入 feet air time、foot clearance、body angular velocity、stand-still regularization、joint deviation、illegal contact 等项。

termination 包括：

```text
too low / too high
too tilted
base contact
illegal contact
invalid numerical state
timeout
```

### Domain Randomization

当前 local MuJoCo 路径支持：

```text
friction scaling
base mass delta
base inertia scale
base COM offset
actuator kp/kd scaling
torque strength scaling
integer action latency
push disturbance
solver iteration / tolerance randomization
contact solref / solimp / margin randomization
```

注意：一个 batched MuJoCo model 里有些字段是全局模型字段，不能在同一时刻天然做到每个 actor 都不同。当前实现保留 per-agent randomization state，并对 actuator scaling 等局部项逐 agent 生效；全局 model 参数则在 reset 时用代表性采样写入共享 model。

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

Unitree/mjlab GO2 flat actor observation 是 47 维：

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

这个 backend 会按当前 `--config` 的机器人选择 G1 或 GO2 bridge。共同做的事情：

```text
1. 读取 Unitree/mjlab RSL-RL checkpoint 里的 actor_state_dict
2. 构造 mjlab 的 actor observation
3. 使用 mjlab action scale
4. 将 OrcaLab runtime joint range / armature / damping / frictionloss 对齐到 mjlab 训练配置
5. 将 OrcaLab runtime motor 改成 mjlab 风格的 MuJoCo position actuator 语义
6. 优先使用 runtime IMU gyro sensor 构造 body-frame base angular velocity
7. play step 写入 target joint position
8. 让 MuJoCo 根据 stiffness / damping / force limit 产生控制力
```

启动后应看到类似日志：

```text
[orca_rl.play] Mjlab runtime alignment: tasks=1, agents=1, joints=29, actuators=29, position_actuator_tasks=1
```

G1 mjlab play 还会打印 contact / sensor 对齐：

```text
[orca_rl.play] G1 mjlab contact alignment: contact_geoms=..., foot_contact_geoms=8, nonfoot_contact_geoms=..., imu_gyro_sensors=1
```

G1 的 mjlab 训练配置把脚底 collision 设为 `condim=3`、`priority=1`、主摩擦 `0.6`，其它 collision 设为 `condim=1`。OrcaLab runtime 里的 G1 脚底是左右 ankle roll link 下的 8 个 sphere geom，所以 G1 bridge 会在当前 play 进程里把这些 foot sphere patch 成 mjlab 风格。

如果 `foot_contact_geoms=0`，说明当前 G1 runtime XML 的脚底 geom 结构和已知 `g1_29dof_usda` 不同，需要重新抓 runtime XML。
如果 `imu_gyro_sensors=0`，说明当前 G1 runtime XML 没有暴露 `*_imu_gyro` sensor，bridge 会退回到 qvel 推导的角速度。

GO2 的 mjlab bridge 额外会把 nominal joint pose 对齐到 `unitree_go2/go2_constants.py` 的 `INIT_STATE`：

```text
FL/RL hip = -0.1
FR/RR hip = 0.1
thigh = 0.9
calf = -1.8
```

这一步很重要，因为 GO2 policy 的 `joint_pos_rel` 和 `action -> target_qpos` 都以 mjlab default pose 为零点。

GO2 bridge 还会做额外的 OrcaLab runtime 对齐：

```text
base reset height = 0.32
reset xy/yaw/joint noise = 0
foot sphere contact condim = 3
foot sphere friction = [0.6, 0.02, 0.01]
foot sphere solimp = [0.9, 0.95, 0.023, ...]
non-foot robot collision condim = 1
non-foot robot collision conaffinity = 0
```

GO2 原地转向时尤其依赖正确的角速度观测。Unitree/mjlab deploy 使用 IMU gyro 作为 body-frame angular velocity；OrcaLab play bridge 现在优先读取 runtime XML 中的 `*_imu_gyro` sensor，而不是从 freejoint `qvel[3:6]` 再猜坐标系。

GO2 mjlab play 时应额外看到：

```text
[orca_rl.play] GO2 mjlab contact alignment: contact_geoms=23, foot_contact_geoms=4, nonfoot_contact_geoms=19, base_height_resets=1, imu_gyro_sensors=1
```

如果 `foot_contact_geoms=0`，说明当前 OrcaLab 下发的 GO2 资产脚底 geom 结构和已知 `go2_usda` 不同，需要重新抓 runtime XML。
如果 `imu_gyro_sensors=0`，说明 runtime XML 没有暴露 `*_imu_gyro` sensor，play bridge 会退回到 qvel 推导的角速度，原地旋转策略更容易出现慢性漂移。

## mjlab G1 / GO2 到 OrcaLab Play 的对齐

OrcaLab scene play 中没有修改 OrcaLab 源码、USDA 资产或缓存 XML 文件。对齐发生在当前 Python play 进程里：

```text
OrcaLab scene G1 / GO2 asset
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

GO2 额外 patch contact / reset / IMU 读取：

```text
model.geom_condim[foot_geom_id] = 3
model.geom_friction[foot_geom_id] = [0.6, 0.02, 0.01]
model.geom_solimp[foot_geom_id][:3] = [0.9, 0.95, 0.023]
model.geom_condim[nonfoot_robot_geom_id] = 1
model.geom_conaffinity[robot_geom_id] = 0
task.cfg["reset"]["base_height"] = 0.32
base_ang_vel_body = query_sensor_data("*_imu_gyro")
```

G1 也会 patch runtime contact geom，但不改 reset height：

```text
model.geom_condim[g1_foot_sphere_geom_id] = 3
model.geom_priority[g1_foot_sphere_geom_id] = 1
model.geom_friction[g1_foot_sphere_geom_id][0] = 0.6
model.geom_condim[g1_nonfoot_robot_geom_id] = 1
base_ang_vel_body = query_sensor_data("*_imu_gyro")
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

## 交付摘要

| 组件 | 说明 |
|---|---|
| `orca_rl/rsl_env/adapters/` | RSL-RL VecEnv 适配层 |
| `orca_rl/rsl_env/batched_locomotion_task.py` | 批量 MuJoCo locomotion runtime |
| `orca_rl/rsl_env/local_mjcf.py` | 本地 MJCF 批量生成器 |
| `orca_rl/rsl_env/scene_binding.py` | OrcaLab scene 自动发现、asset auto-publish、G1 / GO2 binding |
| `orca_rl/rsl_env/mjlab_policy.py` | Unitree/mjlab checkpoint loader、observation bridge、runtime actuator patch |
| `orca_rl/rsl_env/mjwarp_runtime.py` | 实验性 MJWarp step wrapper |
| `orca_rl/tasks/velocity/config/{g1,go2}/` | G1 / GO2 flat / rough task 配置 |
| `orca_rl/terrains/` | 程序化 rough terrain 生成与导出 |
| `third_party/unitree_rl_mjlab/` | Unitree/mjlab 资产、XML、常量 vendoring |

当前技术范围：

| 指标 | 当前状态 |
|---|---|
| 支持机器人 | G1 29 DoF、GO2 12 DoF |
| 任务 | flat / rough velocity tracking |
| 训练接口 | RSL-RL `OnPolicyRunner` |
| 稳定物理后端 | CPU MuJoCo via OrcaGymLocalEnv |
| 实验物理后端 | `mujoco_warp` compatibility bridge |
| 可视化 | OrcaLab scene play |
| mjlab checkpoint play | G1 / GO2 velocity checkpoint |
| checkpoint 参数名 | `--checkpoint` |

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

### 5. mjlab checkpoint 在 OrcaLab play 仍可能受完整物理模型差异影响

runtime patch 已对齐 joint / actuator controller，并对 GO2 的脚底 sphere contact、reset height、IMU gyro 观测做了 mjlab 风格对齐。但 OrcaLab runtime 的完整 collision geom、body mass、inertia、mesh/cylinder/sphere 组合仍可能和 mjlab 训练 XML 不完全一致。OrcaLab runtime 中很多 mesh/sphere/cylinder geom 直接参与 collision，mjlab 则是 visual mesh + simplified collision。

GO2 play 的健康启动日志应包含：

```text
foot_contact_geoms=4
imu_gyro_sensors=1
```

如果这两个都正常但策略仍摔倒，下一步应继续对比 body mass / inertia、foot geom size / pose、command sign、以及 joint/action order。

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
