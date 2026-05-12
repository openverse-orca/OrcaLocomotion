# Orca RL 工程接入工作报告

> 写给 Kimi 做 PPT 的结构化输入稿  
> 角色：harness engineer（接入工程师）  
> 日期：2026-05-12

---

## 项目全景

**Orca RL** 是一个独立 Python 包，核心目标有两件事：

| # | 工作内容 | 一句话概述 |
|---|---------|-----------|
| 1 | RSL-RL 训练工具链接入 OrcaGym | 让 RSL-RL（GPU 强化学习训练框架）能在 OrcaGym / CPU MuJoCo 环境下训练 G1/GO2 机器人 |
| 2 | MJLab 策略模型兼容包 | 让 Unitree 官方 mjlab 训练出的 `.pt` 模型，直接在 OrcaLab 里 Play 出来，不需要装 mjlab 环境 |

物理架构：

```
RSL-RL (PyTorch PPO)
    ↓
OrcaRslRlVecEnv (VecEnv 适配器)
    ↓
BatchedOrcaLocomotionTask (CPU 批量仿真)
    ↓
OrcaGymLocalEnv / MuJoCo 3.8 (本地进程内物理引擎)
    ↓ 可选 GPU 加速路径
mujoco_warp (实验性)
```

支持两种仿真后端：
- **CPU MuJoCo** (`orca_cpu`)：默认路径，`mj_step()` 串行推进
- **GPU MJWarp**（实验性）：`mujoco_warp.step()` 在 GPU 上并行跑物理，同步回 CPU 做 obs/reward

---

# 第一部分：RSL-RL 训练工具链接入 OrcaGym

## 1.1 目标

让 **RSL-RL**（一个基于 PyTorch 的 PPO 强化学习训练框架）跑在 **OrcaGym / CPU MuJoCo** 上，训练 Unitree G1（人形 29 自由度）和 GO2（四足 12 自由度）的速度跟踪任务，对标 mjlab / IsaacLab 的配置风格和训练体验。

## 1.2 核心挑战

| 挑战 | 问题 | 解决方案 |
|------|------|---------|
| **批量并行** | RSL-RL 要求一次 step 推进数百到数千个环境，传统做法是每个 env 一个 Python 包装器 | 设计 `BatchedOrcaLocomotionTask`，一个 MuJoCo 场景放 N 个机器人，一次 `mj_step()` 推进全部 |
| **接口适配** | RSL-RL 需要标准的 `VecEnv` 接口（`step()`, `reset()`, `get_observations()`） | 实现 `OrcaRslRlVecEnv`，把批量任务的返回拼接成 RSL-RL 期望的 `TensorDict` |
| **配置风格对齐** | mjlab 使用 Python 工厂函数生成配置，不是 YAML | 在 `orca_rl/tasks/velocity/config/` 目录下用 `env_cfgs.py` / `rl_cfg.py` 工厂模式 |
| **本地离线训练** | 不想依赖 gRPC 远端 OrcaLab 场景 | 实现 `local_mjcf.py`，从本地 MJCF XML 克隆 N 个机器人并贴地形，生成自包含的批量 XML |
| **头部无渲染** | 训练不需要可视化，但结构要保持可切换 | 实现 `rendering.py` 的 `resolve_rendering()` 统一 headless / human 模式 |

## 1.3 架构分层

```
┌──────────────────────────────────────────────────────────────────┐
│  run_train.py / run_play.py / run_eval.py   — 入口脚本           │
├──────────────────────────────────────────────────────────────────┤
│  registry.py  — 任务注册表                                       │
│  tasks/velocity/config/{g1,go2}/  — 机器人配置文件工厂          │
├──────────────────────────────────────────────────────────────────┤
│  rsl_env/adapters/vecenv.py  — OrcaRslRlVecEnv (VecEnv 适配器)  │
│  rsl_env/adapters/factory.py  — make_locomotion_vec_env()       │
├──────────────────────────────────────────────────────────────────┤
│  rsl_env/batched_locomotion_task.py  — 批量仿真引擎（核心）      │
│   ├── action_mapper.py       — 动作→关节目标+PD扭矩             │
│   ├── obs_builder.py         — 观测构建（策略/特权观测）        │
│   ├── reward_manager.py      — 15项奖励函数                     │
│   ├── termination_manager.py — 6种终止条件                      │
│   ├── curriculum.py          — 速度指令采样/课程                │
│   ├── randomization.py       — 域随机化（20+参数）              │
│   └── terrain_runtime.py     — 地形高度场/扫描                  │
├──────────────────────────────────────────────────────────────────┤
│  rsl_env/scene_binding.py    — 场景绑定（G1/GO2 自动发现）      │
│  rsl_env/model_scanner.py    — 场景模型扫描与模板匹配           │
│  rsl_env/scene_resolvers.py  — 场景绑定解析器分发               │
│  rsl_env/local_mjcf.py       — 本地批量 MJCF 生成               │
│  rsl_env/mjwarp_runtime.py   — GPU 物理加速（实验性）           │
└──────────────────────────────────────────────────────────────────┘
```

## 1.4 批量仿真引擎设计 (BatchedOrcaLocomotionTask)

**核心理念**：一个 MuJoCo 模型里放 N 个机器人，用一个 `mj_step()` 推进所有机器人，向量化处理观测/奖励/终止。

```
┌─────────────────────────────────────────────────┐
│  MuJoCo mjModel                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ robot_000│  │ robot_001│  │ robot_N  │      │
│  │ (G1/GO2) │  │ (G1/GO2) │  │ (G1/GO2) │ ...  │
│  └──────────┘  └──────────┘  └──────────┘      │
│                     ↓                           │
│         一次 mj_step() 推进全部机器人             │
│                     ↓                           │
│  向量化索引 → 并行读取 qpos/qvel/contacts        │
│  向量化计算 → reward/obs/termination             │
└─────────────────────────────────────────────────┘
```

**关键数据结构 `_AgentRuntime`**：为每个机器人预计算所有索引
- 关节 qpos/qvel 在全局数组中的偏移位置
- 执行器 ID 在 `ctrl` 数组中的位置
- 关节限位、名义位姿、PD 增益

**前后端支持**：
- CPU 模式：`mj_step(nstep=frame_skip)` × decimation 次
- GPU 模式：`MjWarpRuntime.step(nstep=frame_skip)` × decimation 次，再 `sync_to_cpu()` 复用 CPU 端 obs/reward 代码

## 1.5 场景绑定与本地 MJCF 生成

**两条加载路径**：

| 路径 | 场景来源 | 使用场景 |
|------|---------|---------|
| OrcaGym 远端 | 通过 gRPC 连接 OrcaLab 场景，用 `scene_binding` 扫描发现机器人实例 | Play / Debug |
| 本地 MJCF | 从本地 XML 文件克隆机器人，`local_mjcf.py` 生成批量 XML | 训练（默认） |

**本地 MJCF 生成流程**：
1. 读取源 `scene_g1.xml`（含单机器人完整定义）
2. 解析 XML 树，定位 `worldbody` 下的根 body
3. 对每个 agent，深拷贝根 body 并偏移 `pos`（网格布局）
4. 如果有地形配置，附加地形 height field
5. 写入临时文件，返回路径给 `OrcaGymLocalEnv` 加载

**自动场景扫描** (`model_scanner.py`)：
- `scan_scene_for_template()`：连接 OrcaGym，探测场景中所有 body/joint/actuator/site/sensor
- `match_robot_instances()`：用后缀模板（如 `left_hip_pitch_joint`）匹配每个机器人实例的完整组件
- `require_complete_matches()`：验证数量满足训练需求，不合格抛异常

## 1.6 任务配置体系

**注册任务**（`registry.py`）：
- `Unitree-G1-Flat`：G1 平地速度跟踪
- `Unitree-G1-Rough`：G1 崎岖地形
- `Unitree-GO2-Flat`：GO2 平地
- `Unitree-GO2-Rough`：GO2 崎岖地形

**配置工厂模式**：
- `tasks/velocity/config/{robot}/env_cfgs.py` → 环境配置（仿真参数、观测、奖励、域随机化）
- `tasks/velocity/config/{robot}/rl_cfg.py` → 训练配置（PPO 超参数、日志、迭代次数）

**G1 训练规格**：
- 时间步长：0.001s，控制频率：50Hz（decimation=4×frame_skip=5）
- 观测维度：约 48 维（策略）/ 约 235 维（特权 Critic）
- 动作空间：29 维连续，residual joint target + PD
- 速度指令范围：vx ∈ [0, 1.0] m/s（flat），vy/ω 可变

---

# 第二部分：MJLab 策略模型兼容包

## 2.1 目标

Unitree 官方用 **mjlab**（基于 IsaacLab）训练了一批 G1 的强化学习策略模型（`.pt` checkpoint），训练环境与 OrcaLab 物理环境一致，但：

- 模型格式不同：mjlab checkpoint 的 state_dict 结构与 RSL-RL 标准不同
- 观测构建不同：mjlab 的观测维度、顺序、归一化方式为自定义格式
- 动作映射不同：mjlab 使用特定的 PD 增益、力矩限幅、关节缩放因子

**目标**：做一个"兼容包"，让这些 `.pt` 模型能在 OrcaLab 环境里直接 Play，**不需要安装 mjlab/IsaacLab**。

## 2.2 核心挑战与解决方案

| 挑战 | 方案 |
|------|------|
| **模型加载** | 解析 mjlab checkpoint 结构，提取 actor MLP 权重和观测归一化器，用纯 PyTorch 重建前向推理 |
| **观测桥接** | `MjlabG1OrcaPlayBridge` 把 OrcaGym 的原始物理状态重新拼接成 mjlab 格式的 98 维观测 |
| **动作桥接** | 加载 Unitree 官方动作参数（scale/kp/kd/effort_limit），执行 `action → target_qpos → PD torque` |
| **相位信号** | mjlab 需要 gait phase（sin/cos），按 0.6 秒周期合成 |
| **高度扫描** | 脚部周围地形高度采样，与 mjlab 的 height scan 格式对齐 |
| **无依赖加载** | 所有 mjlab 常量和配置 vendored 到 `third_party/unitree_rl_mjlab/` |

## 2.3 数据流架构

```
┌─────────────────────────────────────────────────────────────────┐
│  run_play.py --mjlab                                            │
│       │                                                         │
│       ├─ 模型加载                                               │
│       │   MjlabRslRlActorPolicy.from_checkpoint(ckpt.pt)        │
│       │   ├─ 提取 actor MLP (权重+偏置+ELU)                     │
│       │   └─ 提取 obs normalizer (mean/std)                     │
│       │                                                         │
│       ├─ 动作参数                                               │
│       │   MjlabG1ActionSpec ← unitree_rl_mjlab                  │
│       │   ├─ joint_scale: 每个关节的动作缩放                    │
│       │   ├─ kp/kd: PD 控制器增益                              │
│       │   └─ effort_limit: 力矩上限                             │
│       │                                                         │
│       └─ 推理循环 (每步)                                        │
│           ┌──────────────────────────────────────────────┐      │
│           │  MjlabG1OrcaPlayBridge.step(policy_action)   │      │
│           │                                              │      │
│           │  1. get_observations()                       │      │
│           │     ├─ 读取 qpos/qvel (所有机器人)           │      │
│           │     ├─ 读取脚部接触状态 (批量)                │      │
│           │     ├─ _build_mjlab_g1_actor_obs()           │      │
│           │     │   → 角速度、投影重力、指令、相位、      │      │
│           │     │      关位差、关速、上一步动作、高度扫描  │      │
│           │     └─ 叠成 98 维观测向量                     │      │
│           │                                              │      │
│           │  2. policy.act_numpy(obs) → 29 维动作         │      │
│           │                                              │      │
│           │  3. _step_task_with_mjlab_g1_actions()        │      │
│           │     ├─ action × scale → target_qpos           │      │
│           │     ├─ kp*(target-qpos) - kd*qvel → torque   │      │
│           │     ├─ clip torque → ctrl 数组                │      │
│           │     └─ mj_step() 推进物理                     │      │
│           └──────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

## 2.4 关键组件详解

### 2.4.1 MjlabRslRlActorPolicy (`mjlab_policy.py`)
- **类方法 `from_checkpoint(path)`**：
  1. `torch.load()` 加载 `.pt` 文件
  2. 从 `model_state_dict` 提取 `actor.` 前缀的所有参数
  3. 从 `obs_normalizer.` 前缀提取 `running_mean` / `running_var`
  4. 自动推断输入输出维度（`mlp.0.weight` 的 shape）
  5. 构建多层 MLP（每层 Linear + ELU）
- **前向推理 `forward(obs)`**：标准化输入 → MLP → 输出
- **NumPy 接口 `act_numpy(obs, device)`**：`torch.no_grad()` 推理 → CPU numpy

### 2.4.2 MjlabG1OrcaPlayBridge (`mjlab_policy.py`)
**观测构建 `_build_mjlab_g1_actor_obs()`** — 构建与 mjlab 完全一致的 98 维观测：
1. 本体角速度（身体坐标系）→ 3 维
2. 投影重力（身体坐标系）→ 3 维
3. 速度指令（vx, vy, ωz）→ 3 维
4. 步态相位（sin/cos，0.6s 周期）→ 2 维
5. 关节位置偏差（qpos - 名义位姿）→ N 维
6. 关节速度 → N 维
7. 上一步动作 → N 维
8. 脚部高度扫描 → 补齐到 98 维

**动作执行 `_step_task_with_mjlab_g1_actions()`**：
- 归一化动作到 [-1, 1]
- target_qpos = nominal_qpos + action × scale
- PD 扭矩 = kp × (target - qpos) - kd × qvel
- 力矩限幅 → 写入 ctrl 数组
- 支持 CPU MuJoCo 和 GPU MJWarp 两种后端

### 2.4.3 MjlabG1ActionSpec (`mjlab_policy.py`)
冻结数据类，从 vendored `unitree_rl_mjlab` 模块动态加载：
- 关节动作缩放因子（来自 mjlab action config）
- PD 增益（kp/kd，来自 mjlab actuator config）
- 力矩限幅（effort_limit，来自 mjlab motor spec）

### 2.4.4 runtime_policy.py
Orca 自训练模型的加载接口（与 mjlab 无关）：
- `load_inference_runner()`：加载 RSL-RL 标准 checkpoint
- `export_policy()`：导出 JIT `.pt` 和 ONNX `.onnx`

## 2.5 Play 入口集成 (`run_play.py`)

```bash
# 使用 mjlab 训练的 G1 模型 Play
python -m orca_rl.run_play --mjlab

# 指定 mjlab checkpoint 路径
python -m orca_rl.run_play --policy-backend mjlab --ckpt path/to/model.pt

# 固定速度指令 Play
python -m orca_rl.run_play --mjlab --lin-vel-x 0.3 --lin-vel-y 0.0 --ang-vel-z 0.0
```

Play 模式会自动：
1. 加载最新 mjlab checkpoint（从 `third_party/unitree_rl_mjlab/logs/` 搜索）
2. 设置 `render_mode="human"`，打开可视化
3. 关闭域随机化和观测噪声（Play 需要确定性推理）
4. 关闭 episode 时间限制（持续走不被 reset）

---

# 工程总结

## 交付物清单

| 交付物 | 说明 |
|--------|------|
| `orca_rl/` Python 包 | 完整的独立训练+推理包 |
| `orca_rl/rsl_env/adapters/` | RSL-RL VecEnv 适配层 |
| `orca_rl/rsl_env/batched_locomotion_task.py` | 批量仿真引擎（核心创新） |
| `orca_rl/rsl_env/local_mjcf.py` | 本地 MJCF 批量生成器 |
| `orca_rl/rsl_env/scene_binding.py` | 场景自动发现与绑定（G1/GO2） |
| `orca_rl/rsl_env/mjlab_policy.py` | MJLab 策略兼容层（核心创新） |
| `orca_rl/rsl_env/mjwarp_runtime.py` | GPU 物理加速包装器 |
| `orca_rl/tasks/velocity/config/{g1,go2}/` | 4 套任务配置（flat/rough × G1/GO2） |
| `orca_rl/sensor/` | 接触传感器配置声明 |
| `orca_rl/terrains/` | 程序化地形生成 |
| `third_party/unitree_rl_mjlab/` | MJLab 原始资产 vendoring |

## 技术指标

| 指标 | 数值 |
|------|------|
| 支持机器人 | G1（29 DOF 人形）、GO2（12 DOF 四足） |
| 训练规模 | 1 ~ 4096 并行环境 |
| 物理引擎 | MuJoCo 3.8 CPU / mujoco_warp GPU（实验性） |
| 控制频率 | 50 Hz |
| 策略网络 | MLP（ELU 激活） |
| 观测维度 | ~48（策略）/ ~235（特权 Critic） |
| 奖励项数 | 15 项（速度跟踪 + 正则惩罚） |
| 域随机化参数 | 20+ 项 |
| 终止条件 | 6 种 |
| MJLab 兼容观测维度 | 98 维 |
| 模型导出格式 | JIT `.pt` / ONNX `.onnx` |
