# OrcaLocomotion · Orca Warp

`orca_warp` 是 OrcaLocomotion 的 GPU 并行训练分支。它提供 manager-based
强化学习环境、RSL-RL PPO、W&B 实验记录和 OrcaLab 批量策略回放。物理实现封装在
Orca 运行时内部，任务代码只使用 `orcalab_rslrl.orca` 公共接口。

## 支持的任务

| 任务 | 机器人 | 训练 | OrcaLab 回放 |
| --- | --- | --- | --- |
| `G1-Velocity-Flat` | Unitree G1 29DoF | GPU 并行 RSL-RL | 单机或批量回放 |

## 安装

推荐 Ubuntu 22.04、Python 3.12、NVIDIA GPU 和兼容的 CUDA 驱动。

```bash
git clone -b orca_warp https://github.com/openverse-orca/OrcaLocomotion.git
cd OrcaLocomotion

conda env create -f environment.yml
conda activate orcalab-rslrl

pytest -q
orca list
```

复用已有的 OrcaLab Python 环境时：

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

## OrcaLab 资产订阅（实时渲染前必做）

使用 OrcaLab 实时查看训练或回放前，必须在 OrcaLab 资产平台订阅 **`unitree_robots`**：

1. 打开 OrcaLab 资产平台并登录；
2. 搜索 `unitree_robots`（作者为 **Orca**）；
3. 点击 **订阅**，确认状态显示为 **已订阅**；
4. 刷新或重新启动 OrcaLab，使资产同步到本地。

默认使用其中的 G1 prefab：

```text
assets/e071469a36d3c8aa/unitree_robots/prefabs/g1_29dof_usda
```

未订阅该资产时，OrcaLab 无法正确创建和显示 G1。纯 headless 训练使用仓库内置
XML/mesh，不需要启动 OrcaLab。

## 快速检查

创建训练任务并执行一个零动作 step，用于检查模型、传感器、动作维度与运行环境：

```bash
orca inspect --task G1-Velocity-Flat --num-envs 8 --device cuda:0
```

性能检查：

```bash
orca benchmark --task G1-Velocity-Flat \
  --num-envs 4096 --steps 500 --device cuda:0
```

## 训练

```bash
orca train --task G1-Velocity-Flat \
  --num-envs 4096 \
  --device cuda:0 \
  --runner-config configs/train/ppo.yaml \
  --wandb-mode online
```

checkpoint 默认写入 `logs/rsl_rl/`，训练结束后保存
`logs/rsl_rl/model_final.pt`。不使用云端 W&B 时可选择：

```bash
orca train --task G1-Velocity-Flat --wandb-mode offline
orca train --task G1-Velocity-Flat --wandb-mode disabled
```

断点续训：

```bash
orca train --task G1-Velocity-Flat \
  --resume logs/rsl_rl/model_100.pt \
  --num-envs 4096 --device cuda:0
```

训练时连接 OrcaLab 实时查看前 16 个环境：

```bash
orca train --task G1-Velocity-Flat \
  --num-envs 4096 --device cuda:0 \
  --orcalab --render-num-envs 16 --render-fps 30
```

实时渲染会产生 GPU 到 CPU 同步和网络开销，正式性能训练建议关闭 `--orcalab`。

## 回放

不连接 OrcaLab 的 headless 并行回放：

```bash
orca play --task G1-Velocity-Flat \
  --checkpoint logs/rsl_rl/model_final.pt \
  --num-envs 300 --device cuda:0
```

启动 OrcaLab 后，依次选择 **运行** → **开始模拟** →
**无仿真程序启动** → **启动**，再运行：

```bash
orca play --task G1-Velocity-Flat \
  --checkpoint logs/rsl_rl/model_final.pt \
  --num-envs 300 --device cuda:0 \
  --orcalab --render-fps 30
```

连接其它 OrcaLab bridge：

```bash
orca play --task G1-Velocity-Flat \
  --checkpoint logs/rsl_rl/model_final.pt \
  --orcalab --orca-addr 192.168.1.20:50051
```

## 配置

项目配置分为三层：

| 层级 | 入口 | 用途 |
| --- | --- | --- |
| 运行参数 | `orca train/play/inspect` | 设备、并行环境数、checkpoint、渲染 |
| 算法参数 | `configs/train/ppo.yaml` | PPO、网络结构、rollout、W&B |
| 任务参数 | `orcalab_rslrl/tasks/` | MDP、奖励、观测、随机化、终止条件 |

常用参数、优先级和 Python 配置方式见
[配置指南](docs/configuration.md)。训练、回放与自定义资产的完整命令见
[案例手册](docs/examples.md)。

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

任务 MDP term 通过 `env.orca` 读取批量状态，例如
`env.orca.state.qpos`、`env.orca.joint_qvel()` 和
`env.orca.sensor(name)`。应用不应导入 `_internal` 模块。

## 资产

仓库包含 G1 训练所需的 XML 与 mesh：

```text
orcalab_rslrl/assets/robots/unitree_g1/
```

- `--asset <local.xml>`：覆盖本地训练物理模型。
- `--asset-path <orca_asset_path>`：覆盖 OrcaLab 中已订阅的可视化 prefab。

## 验证

```bash
pytest -q
pytest -q -m "not orca_gpu"  # 没有兼容 GPU 时
```

更详细的模块边界见 [架构文档](docs/architecture.md)。
