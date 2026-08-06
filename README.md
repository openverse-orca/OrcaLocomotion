# Orca HEFT G1 + Dex3-1

HEFT G1 PMG 在 OrcaLab 的专用运行包。仓库只保留 G1 + Dex3-1 播放所需的代码、ONNX 策略、
动作片段、安装脚本和上游许可证。

策略动作维度固定为 29，只控制 G1 本体。Dex3-1 的 14 个 actuator 不进入
策略输入输出：启动时双手保持被动闭合，可在 PICO 遥操作中切换开合。

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

## PICO 骨架遥操作（实验）

`./play_g1_heft_pico.sh` 使用与 `unitree_sdk_for_demo` 相同的 XRoboToolkit
`xrobotoolkit_sdk` 链路与 MJViser 浏览器查看器。入口校验完整的 24-joint PICO skeleton，
但 IK 只读取肩、肘、腕链，发布给 HEFT 的也只有双臂 14 DoF。其余 15 个 G1 关节
始终保持启动时的站立参考；Dex3-1 不做手指重定向。

安装 XRoboToolkit Python binding（与 HEFT 使用同一 Python 环境）：

```bash
ORCA_HEFT_PYTHON="$(pwd)/.venv/bin/python" ./scripts/install_xrobottoolkit_sdk.sh
```

启动顺序：

1. 启动 OrcaLab/OrcaStudio，加载 G1 29DoF + Dex3-1 场景并选择外部仿真程序。
2. 运行 `./scripts/start_xrobottoolkit_service.sh`。
3. 在 PICO XRoboToolkit App 中连接 teleop 主机的 LAN IPv4，开启 Body 和 Controller tracking。
4. 运行 `./scripts/check_pico_xrobot_stream.sh --seconds 10` 确认数据已到达。
5. 保持中立站姿，运行 `./play_g1_heft_pico.sh`；首个有效骨架帧会用于标定。

默认可在 `http://localhost:8080` 查看在线参考。`O` / `C` 张开/闭合 Dex3-1，
`R` 复位机器人并在下一个有效 PICO 帧重新标定，`Q` / `Esc` 退出。无 PICO 时可先跑
离线链路测试：

```bash
./play_g1_heft_pico.sh --demo-hand-forward --demo-side right --seconds 6
```

详细安装、远程连接、参数、标定与安全边界见
[`docs/PICO_TELEOP.md`](docs/PICO_TELEOP.md)。该入口目前仅用于仿真联调，不要直接连接真机。

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
