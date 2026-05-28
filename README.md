# OrcaLocomotion

OrcaLocomotion 是 OrcaLab 的训练扩展组件，支持训练 Unitree 机器人的运动控制策略，
并可在 OrcaLab 中播放训练好的策略。

| 仿真 | 实机 |
| --- | --- |
| <img src="output.gif" alt="OrcaLocomotion simulation preview" width="480"> | <img src="physical.gif" alt="OrcaLocomotion physical preview" width="480"> |

## 支持的任务

| 任务 | 训练 | OrcaLab 回放 |
| --- | --- | --- |
| `Unitree-Go2-Flat` | Unitree/mjlab | `orca_rl.run_play` |
| `Unitree-Go2-Rough` | Unitree/mjlab | `orca_rl.run_play` |
| `Unitree-G1-Flat` | Unitree/mjlab | `orca_rl.run_play` |

## 安装

推荐使用 Ubuntu 22.04 + NVIDIA GPU 进行训练。仅在 OrcaLab 中回放策略时需要显示器。

```bash
git clone https://github.com/openverse-orca/OrcaLocomotion.git

# 编译 C++ 扩展所需的系统依赖
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev

# 创建并激活 OrcaLab Python 3.12 环境
conda create -n orcalab python=3.12
conda activate orcalab

# 进入刚下载的 OrcaLocomotion 仓库根目录
cd OrcaLocomotion
pip install --no-deps mjlab==1.2.0
pip install -r requirements.txt
```

> **OrcaLab 前置条件：** 回放需要在 OrcaLab 中订阅 `unitree_robots` 资产。
> 本项目使用的 GO2 和 G1 机器人预制件均来自该资产包。

## 训练

使用基于 mjlab 项目进行训练。以下命令以无头模式运行 mjlab 训练器，
checkpoint 写入 `third_party/mjlab_rl/logs/rsl_rl/`。

```bash
cd third_party/mjlab_rl
python scripts/train.py Unitree-Go2-Flat --env.scene.num-envs=4096
python scripts/train.py Unitree-Go2-Rough --env.scene.num-envs=4096
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096
```

训练产出的 checkpoint 可用于 OrcaLocomotion 回放。

## 回放

启动 OrcaLab 后，依次选择 **运行** -> **开始模拟** -> **无仿真程序启动** -> **启动**。
然后在仓库根目录运行回放命令：

```bash
python -m orca_rl.run_play --config Unitree-Go2-Flat --checkpoint <checkpoint.pt>
python -m orca_rl.run_play --config Unitree-Go2-Rough --checkpoint <checkpoint.pt>
python -m orca_rl.run_play --config Unitree-G1-Flat --checkpoint <checkpoint.pt>
```

使用 Unitree/mjlab 训练步骤产出的 checkpoint。

> **注意：** `Unitree-Go2-Rough` 需要在 OrcaLab 中订阅地形资产。如需导入额外资产，
> 请 [提交 issue](https://github.com/openverse-orca/OrcaLocomotion/issues)。

## 资产

`Unitree-Go2-Rough` 在 OrcaLab scene 回放时会默认发布这个粗糙地形可视化资产：

```text
assets/e071469a36d3c8aa/test_terrain/prefabs/terrain_usda
```

也可以使用另一个 5x5 地形资产：

```text
assets/e071469a36d3c8aa/terrain_5x5/prefabs/terrain_usda
```

> **必须订阅地形渲染资产：** 在 Orca 云端资产管理平台订阅
> **`OrcaPrimitiveTerrainXml`** 和 **`MjlabRough5x5xml_20260527`**。
> 它们是两个地形高度图对应的渲染资产。

如果导入了自己的 OrcaLab 地形资产，需要在 `python -m orca_rl.run_play` 中通过
`--rough-terrain-asset <orca_asset_path>` 改成新的 spawnable asset 路径；必要时也可以用
`--rough-terrain-actor <actor_name>` 改发布到 scene 里的 actor 名称。

仓库中同时保留了对应的 rough 地形 XML 源文件，便于检查或重新导入资产：

| 资产 | 用途 |
| --- | --- |
| `assets/terrain/mjlab_rough_5x5_xml/terrain.xml` | Go2 粗糙地形回放 |

## 致谢

训练任务和机器人资产参考 [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)
项目构建，并使用 [mujocolab/mjlab](https://github.com/mujocolab/mjlab) 作为训练后端。
用户可自行选择基于 mjlab 的第三方扩展项目。
