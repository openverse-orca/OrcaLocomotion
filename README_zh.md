# Orca RL

Orca RL 是 OrcaLab 的训练扩展组件，支持训练 Unitree 机器人的运动控制策略，
并可在 OrcaLab 中播放训练好的策略。

## 支持的任务

| 任务 | 训练 | OrcaLab 回放 |
| --- | --- | --- |
| `Unitree-Go2-Flat` | Unitree/mjlab | `orca_rl.run_play` |
| `Unitree-Go2-Rough` | Unitree/mjlab | `orca_rl.run_play` |
| `Unitree-G1-Flat` | Unitree/mjlab | `orca_rl.run_play` |

## 安装

推荐使用 Ubuntu 22.04 + NVIDIA GPU 进行训练。仅在 OrcaLab 中回放策略时需要显示器。

```bash
git clone https://github.com/BenHuHuan/orca_rl.git
cd orca_rl

# 编译 C++ 扩展所需的系统依赖
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev

conda create -n orca_rl python=3.11
conda activate orca_rl

pip install -r requirements.txt
```

> **OrcaLab 前置条件：** 回放需要在 OrcaLab 中订阅 `unitree_robots` 资产。
> 本项目使用的 GO2 和 G1 机器人预制件均来自该资产包。

## 训练

使用捆绑的 Unitree/mjlab 项目进行训练。以下命令以无头模式运行 mjlab
训练器，checkpoint 写入 `third_party/unitree_rl_mjlab/logs/rsl_rl/`。

```bash
cd third_party/unitree_rl_mjlab
python scripts/train.py Unitree-Go2-Flat --env.scene.num-envs=4096
python scripts/train.py Unitree-Go2-Rough --env.scene.num-envs=4096
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096
```

训练产出的 checkpoint 可用于 Orca RL 回放。

## 回放

启动 OrcaLab，然后在仓库根目录运行回放命令：

```bash
python -m orca_rl.run_play --config Unitree-Go2-Flat --checkpoint <checkpoint.pt>
python -m orca_rl.run_play --config Unitree-Go2-Rough --checkpoint <checkpoint.pt>
python -m orca_rl.run_play --config Unitree-G1-Flat --checkpoint <checkpoint.pt>
```

使用 Unitree/mjlab 训练步骤产出的 checkpoint。

> **注意：** `Unitree-Go2-Rough` 需要在 OrcaLab 中订阅地形资产。如需导入额外资产，
> 请 [提交 issue](https://github.com/BenHuHuan/orca_rl/issues)。

## 资产

在地形回放前，将 XML 地形文件上传到 OrcaLab：

| 资产 | 用途 |
| --- | --- |
| `assets/terrain/orca_primitive_terrain_xml/terrain.xml` | 地形测试 |
| `assets/terrain/mjlab_rough_5x5_xml/terrain.xml` | Go2 粗糙地形回放 |

在 OrcaLab 的 XML 资产上传流程中上传即可。

## 致谢

训练任务和机器人资产基于 [mujocolab/mjlab](https://github.com/mujocolab/mjlab)
和捆绑的 [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) 项目，用户可自行替换为其他第三方训练库。
