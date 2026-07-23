# HEFT G1 + Dex3-1 部署指南

这套运行包把 HEFT G1 PMG ONNX 策略直接接入 OrcaLab。策略只控制 G1 本体 29 个关节；
Dex3-1 的 14 个手部 actuator 不进入 observation/action，并在运行时保持被动张开，因而策略 ABI
始终是 29 维。

## 环境要求

- Ubuntu 22.04/24.04 x86_64
- Python 3.12 或 3.13
- Git LFS
- 可运行的 OrcaLab/OrcaStudio，版本 `26.6.3`
- OrcaStudio 工程中可用的 G1 29DoF + Dex3-1 prefab

模型、动作和依赖总体积较大，首次安装需要可访问 PyPI 和 MuJoCo Python 包源。

## 安装

```bash
git lfs install
git clone --branch heft_dex3-1 https://github.com/openverse-orca/OrcaLocomotion.git
cd OrcaLocomotion
./install_heft.sh
```

安装脚本默认创建仓库内的 `.venv`，拉取 HEFT LFS 资产，安装锁定版本的运行依赖，并校验模型、
动作文件和 Python import。也可以指定解释器或虚拟环境位置：

```bash
ORCA_HEFT_PYTHON=/opt/python3.12/bin/python \
ORCA_HEFT_VENV=/data/venvs/orca-heft \
./install_heft.sh
```

只检查当前安装，不修改环境：

```bash
./scripts/check_heft_install.sh --runtime
```

## OrcaLab 准备

1. 启动 OrcaLab，加载场景。
2. 订阅资产 G1 29DoF + Dex3-1；默认 prefab 路径为
   `assets/13951baeb514b4b9/default_project/prefabs/g1_pick_usda`。
3. 仿真程序选择“外部/无仿真程序”，确保 OrcaGym 服务地址可连接。

如服务不是默认地址，在启动命令后追加 `--remote host:port`。

## 一键运行

```bash
./play_g1_heft.sh
```

快捷键：

| 按键 | 动作 |
| --- | --- |
| `W/S` 或上下方向键 | 前进/后退 |
| `A/D` 或左右方向键 | 左移/右移 |
| `Z/C` | 左转/右转 |
| `F1/F2/F3` | HEFT 原始 walk1/walk2/walk3 |
| 空格 | 站立 |
| `R` | 复位机器人并清零命令 |
| `Q` 或 `Esc` | 退出 |

固定命令和自动结束适合联调：

```bash
./play_g1_heft.sh --keyboard-backend none --lin-vel-x 0.5 --seconds 20
```

## 常见问题

- `SHA256SUMS: FAILED`：通常是 Git LFS 未拉完整，执行
  `git lfs pull --include='checkpoints/heft/**,assets/heft/**'` 后重新检查。
- 找不到 `orca_lab`、`mujoco` 或 `pynput`：重新运行 `./install_heft.sh`，不要用系统 Python
  覆盖 `.venv`。
- 找不到完整 G1：确认 OrcaStudio 中已有 G1 + Dex3-1，或用 `--remote` 指向正确服务。
- `action_dim` 不是 29：场景本体关节命名或模型版本不兼容；Dex3 actuator 不应加入 locomotion
  action 列表。
- 手指抖动：启动日志应出现 `G1 Dex3 passive mode`，并显示 14 个 hand actuator 被禁用。

## HEFT 引用

G1 PMG 策略和 `walk1/walk2/walk3` 动作来自
[Axellwppr/motion_tracking](https://github.com/Axellwppr/motion_tracking) 的 `sim2real` 分支，
固定版本为 [`0d5ba31e33397f3543d350d98b637e26d92f470a`](https://github.com/Axellwppr/motion_tracking/commit/0d5ba31e33397f3543d350d98b637e26d92f470a)，
采用 MIT License；Copyright (c) 2026 Axell。
