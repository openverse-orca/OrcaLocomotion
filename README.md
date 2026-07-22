# OrcaLab-RSLRL

面向 OrcaLab 的 GPU 并行机器人强化学习框架。项目提供 manager-based 环境、
任务注册、RSL-RL 训练、W&B 实验记录与 OrcaLab 批量回放；物理求解器被封装在
运行时内部，应用和任务代码只依赖 `orca` 接口。

## 安装

推荐使用项目锁定的 Conda 环境：

```bash
conda env create -f environment.yml
conda activate orcalab-rslrl
pytest -q
```

复用已有 `orcalab` 环境时：

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

要求 Python 3.12、兼容的 NVIDIA 驱动与 CUDA。只有实时可视化需要启动
OrcaStudio；训练和检查默认无窗口运行。

## 命令行

所有用户功能统一在 `orca` 命令下：

```bash
orca list
orca inspect --task G1-Velocity-Flat --num-envs 256
orca train --task G1-Velocity-Flat --num-envs 4096 --device cuda:0
orca benchmark --task G1-Velocity-Flat --num-envs 4096 --steps 500
```

训练产物默认写入 `logs/rsl_rl/`，结束时保存 `model_final.pt`。在线 W&B 是
默认模式，也可选择离线或关闭：

```bash
orca train --task G1-Velocity-Flat --wandb-mode offline
orca train --task G1-Velocity-Flat --wandb-mode disabled
```

无 OrcaStudio 的并行策略回放：

```bash
orca play --task G1-Velocity-Flat \
  --checkpoint logs/rsl_rl/model_final.pt \
  --num-envs 300 --device cuda:0
```

启动 OrcaStudio 后增加 `--orcalab` 即可实时显示整个 batch：

```bash
orca play --task G1-Velocity-Flat \
  --checkpoint logs/rsl_rl/model_final.pt \
  --num-envs 300 --orcalab --render-fps 30
```

密集布局可使用 `--spacing 0.6 --spawn-range 4.0 --root-xy-scale 0.2`。
楼梯等场景需同时选择物理地形和对应的 OrcaStudio prefab：

```bash
orca play --task G1-Velocity-Flat \
  --checkpoint logs/rsl_rl/model_final.pt --orcalab \
  --terrain-kind stair-mid-flat \
  --terrain-asset-path assets/<project>/prefabs/terrain_stair_mid_flat_usda
```

## Python API

支持的公共模块只有 `orcalab_rslrl.orca`：

```python
from orcalab_rslrl.orca import OrcaRuntimeConfig, make_env

cfg = OrcaRuntimeConfig(
    num_envs=1024,
    device="cuda:0",
    headless=True,
    seed=42,
)
env = make_env("G1-Velocity-Flat", cfg)
try:
    observations = env.get_observations()
finally:
    env.close()
```

任务 MDP term 通过 `env.orca` 读取批量状态，例如
`env.orca.state.qpos`、`env.orca.joint_qvel()` 和
`env.orca.sensor(name)`。任务代码不应导入或调用任何求解器模块。

## 新任务接入

任务工厂接收运行参数并返回 RSL-RL 向量环境：

```python
from orcalab_rslrl.orca import register_task

def make_train_env(*, num_envs: int, device: str, headless: bool = True):
    ...

register_task("MyRobot-Velocity-Flat", make_train_env, "MyRobot 平地速度控制")
```

生产任务建议按以下边界组织：

```text
tasks/<task>.py       组合机器人、场景、MDP 与训练配置
robots/<robot>.py     机器人模型、执行器和 name→index 校验
assets/robots/...     可版本化的 XML、mesh 与默认姿态
mdp/...               可复用 observation/reward/event/termination term
configs/train/...     算法超参数
```

## 架构边界

```text
orca CLI / orcalab_rslrl.orca
        ↓
task registry → ManagerBasedRLEnv → MDP terms
        ↓
OrcaPhysics contract → private GPU runtime
        ↓
RSL-RL / W&B                 OrcaLab renderer
```

- 公共层只使用 Orca 命名，不提供物理后端选择开关。
- 所有状态张量第一维固定为 `num_envs`，训练热路径不做全量 CPU 同步。
- 任务只组合 term，不改写环境 step/reset 生命周期。
- 训练器只依赖 `RslRlVecEnvAdapter`；渲染是独立消费者，不参与物理求解。
- 本地机器人资产随包发布，可用 `--asset` 临时覆盖，避免依赖外部 checkout。

更详细的稳定边界和内部依赖规则见 [架构文档](docs/architecture.md)。

## 验证

```bash
pytest -q
orca inspect --task G1-Velocity-Flat --num-envs 8
orca train --task G1-Velocity-Flat --num-envs 256 --iterations 3 \
  --wandb-mode disabled
```

GPU smoke 测试带有 `orca_gpu` marker；没有兼容 GPU 环境时可运行：

```bash
pytest -q -m "not orca_gpu"
```
