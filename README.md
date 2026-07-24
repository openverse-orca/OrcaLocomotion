# Orca HEFT G1 + Dex3-1

HEFT G1 PMG 在 OrcaLab 的专用运行包。仓库只保留 G1 + Dex3-1 播放所需的代码、ONNX 策略、
动作片段、安装脚本和上游许可证。

策略动作维度固定为 29，只控制 G1 本体。Dex3-1 的 14 个 actuator 不进入策略输入输出，运行时
双手保持被动张开，避免手指扰动本体控制。

## 安装

要求 Ubuntu x86_64、Python 3.12+、Git LFS 和 OrcaLab/OrcaStudio `26.6.3`。

```bash
git lfs install
git clone --branch heft_dex3-1 https://github.com/openverse-orca/OrcaLocomotion.git
cd OrcaLocomotion
./install_heft.sh
```

默认安装到仓库内 `.venv`。自定义解释器或环境位置：

```bash
ORCA_HEFT_PYTHON=/path/to/python3.12 \
ORCA_HEFT_VENV=/data/venvs/orca-heft \
./install_heft.sh
```

## 运行

先启动 OrcaLab/OrcaStudio，加载带 G1 29DoF + Dex3-1 的场景，并选择外部仿真程序，然后执行：

```bash
./play_g1_heft.sh
```

非默认服务地址：

```bash
./play_g1_heft.sh --remote 127.0.0.1:50051
```

| 按键 | 动作 |
| --- | --- |
| `W/S`、上下方向键 | 前进/后退 |
| `A/D`、左右方向键 | 左移/右移 |
| `Z/C` | 左转/右转 |
| `F1/F2/F3` | HEFT walk1/walk2/walk3 |
| 空格 | 站立 |
| `R` | 复位 |
| `Q`、`Esc` | 退出 |

SSH/无全局键盘权限时使用：

```bash
./play_g1_heft.sh --keyboard-backend terminal
```

固定命令联调：

```bash
./play_g1_heft.sh --keyboard-backend none --lin-vel-x 0.5 --seconds 20
```

兼容入口 `./play_g1_heft_velocity.sh` 保留。完整部署与排障见
[`docs/HEFT_DEX3.md`](docs/HEFT_DEX3.md)。

## Citation

This runtime package is based on [HEFT](https://arxiv.org/abs/2607.02332).
If you use it in academic work, please cite:

```bibtex
@misc{liu2026heftheavypayloadfullsizehumanoid,
  title={HEFT: Heavy-Payload Full-size Humanoid Teleoperation with Privileged Motion Guidance and Windowed Payload Curriculum},
  author={Chenxin Liu and Qingzhou Lu and Guangxiao Yang and Xuanyang Shi and Chenghan Yang and Yanjiang Guo and Jianyu Chen},
  year={2026},
  eprint={2607.02332},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2607.02332},
}
```
