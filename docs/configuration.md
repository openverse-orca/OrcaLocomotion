# Orca Warp 配置指南

## 配置层级

Orca Warp 将配置拆成运行参数、算法参数和任务参数。命令行只负责一次实验的运行差异，
YAML 保存可复用的训练器配置，任务 Python 模块定义环境语义。

```text
CLI runtime options
       ↓
registered task + task MDP config
       ↓
RSL-RL runner YAML
       ↓
Orca runtime
```

## 运行参数

### 训练

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--task` | `G1-Velocity-Flat` | 注册任务 ID |
| `--num-envs` | `4096` | GPU 并行环境数 |
| `--device` | `cuda:0` | PyTorch/Orca 运行设备 |
| `--runner-config` | `configs/train/ppo.yaml` | RSL-RL 配置 |
| `--iterations` | YAML 中的值 | 临时覆盖训练迭代数 |
| `--log-dir` | `logs/rsl_rl` | checkpoint 和日志目录 |
| `--resume` | 无 | 恢复 checkpoint |
| `--asset` | 内置 G1 XML | 本地机器人模型覆盖 |
| `--wandb-mode` | `online` | `online`、`offline` 或 `disabled` |
| `--check-for-nan` | 关闭 | 每 step 检查 NaN，会降低性能 |

命令行的 `--iterations` 优先于 YAML 中的 `max_iterations`。其它 PPO 参数由
`--runner-config` 指定的 YAML 提供。

### 回放

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--checkpoint` | 必填 | RSL-RL checkpoint |
| `--num-envs` | `300` | 并行回放数量 |
| `--steps` | `100000` | 最大控制步数 |
| `--orcalab` | 关闭 | 启用 OrcaStudio 实时渲染 |
| `--orca-addr` | `localhost:50051` | OrcaLab bridge 地址 |
| `--render-fps` | `30` | OrcaStudio 推流帧率 |
| `--physics-timestep` | 任务默认值 | 回放物理 timestep |
| `--asset-path` | G1 prefab | OrcaStudio 机器人资产 |

## PPO YAML

默认文件为 `configs/train/ppo.yaml`。建议复制后修改，不直接覆盖默认配置：

```bash
cp configs/train/ppo.yaml configs/train/ppo_experiment.yaml
orca train --runner-config configs/train/ppo_experiment.yaml
```

常用字段：

| 字段 | 说明 |
| --- | --- |
| `seed` | 训练器随机种子 |
| `num_steps_per_env` | 每次更新前每个环境采集的 step 数 |
| `max_iterations` | 最大 PPO iteration |
| `save_interval` | checkpoint 保存间隔 |
| `actor.hidden_dims` | Actor MLP 隐层 |
| `critic.hidden_dims` | Critic MLP 隐层 |
| `algorithm.learning_rate` | 学习率 |
| `algorithm.desired_kl` | adaptive schedule 的目标 KL |
| `algorithm.gamma` / `lam` | 回报折扣与 GAE 参数 |

## Python 运行配置

程序化使用时通过 `OrcaRuntimeConfig` 配置环境：

```python
from orcalab_rslrl.orca import OrcaRuntimeConfig, make_env

config = OrcaRuntimeConfig(
    num_envs=256,
    device="cuda:0",
    headless=True,
    play=False,
    asset=None,
    physics_timestep=0.005,
    seed=1,
)
env = make_env("G1-Velocity-Flat", config)
```

| 字段 | 说明 |
| --- | --- |
| `asset` | 本地物理 XML；不同于 OrcaStudio 的 `--asset-path` |
| `physics_timestep` | 物理 step；控制周期还会乘以任务 decimation |
| `play` | 关闭训练噪声和 push，使用回放任务设置 |
| `seed` | 模型构建及环境随机化种子 |

## 任务配置

`orcalab_rslrl/tasks/g1_velocity.py` 组合以下内容：

- command：速度范围、heading control、standing 比例与重采样周期；
- observation：actor 噪声观测和 critic privileged observation；
- reward：速度跟踪、姿态、步态、足端与正则项；
- event：reset、push、encoder bias；
- termination：超时和倾倒；
- action：scale、default offset 和 decimation。

需要修改任务语义时，创建新的任务模块和任务 ID，避免让不同实验共享同一个名称却产生
不同的 MDP 定义。

## 资产配置

物理资产与渲染资产是两个独立入口：

```text
--asset robot.xml                  # Orca 物理模型
--asset-path assets/.../g1_usda    # OrcaStudio 机器人 prefab
```

使用默认 G1 prefab 前，必须先在 OrcaLab 资产平台订阅 `unitree_robots`。

修改机器人资产后必须检查 joint、actuator、body 和 sensor 名称。名称不一致会在环境创建
阶段报错，不应通过修改索引绕过。
