# OrcaLocomotion · Orca Warp

面向 **OrcaLab** 的 GPU 并行机器人强化学习框架。Orca Warp 将 manager-based
任务设计、MuJoCo Warp 批量物理、RSL-RL PPO 训练和 OrcaLab 可视化连接在同一条
工作流中：用一个命令训练，用同一 checkpoint 做 headless 或 OrcaLab 回放。

> `orca_warp` 是专注 OrcaLab 的实现分支。任务代码仅依赖
> `orcalab_rslrl.orca` 公共接口；运行时实现属于内部模块，不应被应用代码直接导入。

## ✨ 特性

```text
┌───────────────────────┐       ┌─────────────────────────┐
│  G1 manager-based MDP │       │       RSL-RL PPO        │
│ commands · rewards    │ ───▶  │ GPU-parallel rollouts   │
│ observations · events │       │ checkpoints · W&B       │
└───────────┬───────────┘       └───────────┬─────────────┘
            │                                │
            └──────── Orca Warp ────────────┘
                         │
                         ▼
                OrcaLab live playback
```

- **GPU 并行训练**：基于 MuJoCo Warp 的批量环境与 RSL-RL PPO。
- **单一任务接口**：训练、检查、性能测试和回放均通过 `orca` CLI 进入。
- **两种回放方式**：高吞吐 headless 回放，或连接 OrcaLab 进行实时可视化。
- **清晰的产品边界**：任务通过稳定的 `orcalab_rslrl.orca` API 访问 Orca 运行时。

## 支持的任务

| 任务 | 机器人 | 训练 | 回放 |
| --- | --- | --- | --- |
| `G1-Velocity-Flat` | Unitree G1 29DoF | GPU 并行 RSL-RL | Headless / OrcaLab |

## 🚀 快速开始

需要 Ubuntu 22.04、Python 3.12、NVIDIA GPU 与兼容的 CUDA 驱动。

```bash
git clone --branch orca_warp https://github.com/openverse-orca/OrcaLocomotion.git
cd OrcaLocomotion

conda env create -f environment.yml
conda activate orcalab-rslrl

# 建立 8 个环境并执行一次零动作 step，验证安装与任务契约
orca inspect --task G1-Velocity-Flat --num-envs 8 --device cuda:0
```

已有可用的 OrcaLab Python 环境时，可改用：

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
orca inspect --task G1-Velocity-Flat --num-envs 8 --device cuda:0
```

运行完整测试：

```bash
pytest -q
# 没有兼容 GPU 时：
pytest -q -m "not orca_gpu"
```

## 🏃 训练与回放

### 1. 训练速度跟踪策略

```bash
orca train --task G1-Velocity-Flat \
  --num-envs 4096 \
  --device cuda:0 \
  --runner-config configs/train/ppo.yaml \
  --wandb-mode online
```

checkpoint 默认写入 `logs/rsl_rl/`；完成时为
`logs/rsl_rl/model_final.pt`。不使用 W&B 云端记录时：

```bash
orca train --task G1-Velocity-Flat --wandb-mode offline
orca train --task G1-Velocity-Flat --wandb-mode disabled
```

从 checkpoint 续训：

```bash
orca train --task G1-Velocity-Flat \
  --resume logs/rsl_rl/model_100.pt \
  --num-envs 4096 --device cuda:0
```

### 2. Headless 并行回放

该模式不需要 OrcaLab，适合服务器检查策略或批量评估：

```bash
orca play --task G1-Velocity-Flat \
  --checkpoint logs/rsl_rl/model_final.pt \
  --num-envs 300 --device cuda:0
```

### 3. 训练时或回放时连接 OrcaLab

首次使用前，需在 OrcaLab 资产平台订阅 **`unitree_robots`**：

1. 打开资产平台并登录；
2. 搜索 `unitree_robots`（作者为 **Orca**）；
3. 点击 **订阅**，确认状态显示为 **已订阅**；
4. 刷新或重启 OrcaLab，使资产同步到本地。

默认 G1 prefab 位于：

```text
assets/e071469a36d3c8aa/unitree_robots/prefabs/g1_29dof_usda
```

未订阅该资产时，OrcaLab 无法正确创建和显示 G1；纯 headless 训练和回放使用
仓库内置的 XML/mesh，不需要启动 OrcaLab。

订阅后，训练时实时查看前 16 个环境：

```bash
orca train --task G1-Velocity-Flat \
  --num-envs 4096 --device cuda:0 \
  --orcalab --render-num-envs 16 --render-fps 30
```

或者先在 OrcaLab 中依次选择 **运行** → **开始模拟** → **无仿真程序启动** →
**启动**，再执行回放：

```bash
orca play --task G1-Velocity-Flat \
  --checkpoint logs/rsl_rl/model_final.pt \
  --num-envs 300 --device cuda:0 \
  --orcalab --render-fps 30
```

连接另一台机器上的 OrcaLab bridge：

```bash
orca play --task G1-Velocity-Flat \
  --checkpoint logs/rsl_rl/model_final.pt \
  --orcalab --orca-addr 192.168.1.20:50051
```

> OrcaLab 实时渲染会带来 GPU→CPU 同步和网络开销。基准测试与正式训练请关闭
> `--orcalab`。

## 常用命令

| 命令 | 用途 | 示例 |
| --- | --- | --- |
| `orca inspect` | 验证模型、传感器、动作维度与运行环境 | `orca inspect --task G1-Velocity-Flat --num-envs 8` |
| `orca benchmark` | 测量批量环境吞吐 | `orca benchmark --task G1-Velocity-Flat --num-envs 4096 --steps 500` |
| `orca train` | 训练或从 checkpoint 续训 | `orca train --task G1-Velocity-Flat --device cuda:0` |
| `orca play` | 回放 checkpoint | `orca play --task G1-Velocity-Flat --checkpoint <path>` |
| `orca list` | 列出可用的 CLI 任务 | `orca list` |

## 配置与扩展

配置按职责分成三层：

| 层级 | 入口 | 负责内容 |
| --- | --- | --- |
| 运行参数 | `orca train/play/inspect` | 设备、环境数、checkpoint、渲染和 bridge 地址 |
| 算法参数 | `configs/train/ppo.yaml` | PPO、网络、rollout 和 W&B |
| 任务参数 | `orcalab_rslrl/tasks/` | MDP、奖励、观测、随机化与终止条件 |

- 参数优先级和 Python 配置方式：[配置指南](docs/configuration.md)
- 自定义训练、回放与资产命令：[案例手册](docs/examples.md)
- 运行时边界与模块职责：[架构文档](docs/architecture.md)

本地 G1 训练资产在：

```text
orcalab_rslrl/assets/robots/unitree_g1/
```

- `--asset <local.xml>`：覆盖本地训练物理模型。
- `--asset-path <orca_asset_path>`：覆盖 OrcaLab 中已订阅的可视化 prefab。

## Python API

```python
from orcalab_rslrl.orca import OrcaRuntimeConfig, make_env

config = OrcaRuntimeConfig(
    num_envs=1024,
    device="cuda:0",
    headless=True,
    seed=42,
)

env = make_env("G1-Velocity-Flat", config)
try:
    observations = env.get_observations()
finally:
    env.close()
```

MDP term 可经由 `env.orca` 读取批量状态，例如 `env.orca.state.qpos`、
`env.orca.joint_qvel()` 和 `env.orca.sensor(name)`。请不要导入 `_internal` 模块。
