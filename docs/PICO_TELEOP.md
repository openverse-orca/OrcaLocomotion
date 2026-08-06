# PICO 骨架遥操作（实验）

此入口使用 `unitree_sdk_for_demo` 中相同的 XRoboToolkit SDK：
`xrobotoolkit_sdk` 的 `register_frame_callback` 回调，以及 `mjviser + viser`
浏览器查看器。

本实现接收并校验 XRoboToolkit 的完整 24-joint PICO skeleton，但 IK 只读取肩、肘、腕链，
其变量和输出严格只有 G1
双臂的 14 DoF（左右各 7）。其余 15 个 G1 关节（骨盆、双腿和三个腰部）始终保持启动时的
默认站立姿态。Dex3-1 仅是腕端上的独立“假手”，不会接收 PICO 手指重定向。

为避免 HEFT 接收 PICO 抖动带来的分布外目标，双臂 IK 与发布给策略的每帧 qpos 都有速度限幅。
浏览器只显示实际参与重定向的肩、肘、腕链。

位置重定向以每侧肩部的**首次标定局部坐标系**为准，再映射到对应的 G1 肩部坐标系；不会把 PICO
全局世界坐标直接当作机器人前方。因而操作者与机器人在标定时朝向不同、或操作者随后转动躯干时，
单臂前伸仍应保持为机器人相同的局部前伸方向。腕部姿态同时采用 GMR `xrobot_to_g1` 的左右肩帧偏置。

## 前置条件

1. 按常规 HEFT 步骤安装项目：`./install_heft.sh`。
2. 使用与参考项目相同的 XRoboToolkit PC service 和 Python binding：
   [XRoboToolkit-PC-Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service) 与
   [XRoboToolkit-PC-Service-Pybind](https://github.com/Axellwppr/XRoboToolkit-PC-Service-Pybind)。
   该仓库的安装命令与参考项目使用相同源码和 build 路径，安装目标是运行 HEFT 的同一个 Python 环境：

   ```bash
   conda activate orcalab
   ORCA_HEFT_PYTHON="$(which python)" ./scripts/install_xrobottoolkit_sdk.sh
   ```
3. 在 teleop 主机启动 PC service：

   ```bash
   cd /opt/apps/roboticsservice
   bash runService.sh
   ```

4. 在 PICO 的 XRoboToolkit client 中连接该主机，并启用完整 Body tracking 与两个 Controller tracking。
5. 在 OrcaLab/OrcaStudio 中加载带 actuator 的 G1 29DoF + Dex3-1 场景，选择外部仿真程序。

连接后、启动 HEFT 前先确认数据确实到达 PC service：

```bash
./scripts/check_pico_xrobot_stream.sh --seconds 10
```

至少应看到 `headset=yes` 与 `body=yes`。完整 skeleton 可用后，
`./play_g1_heft_pico.sh` 的 `frames=` 才会增长。

先验证 SDK 在当前环境中可用：

```bash
python - <<'PY'
import xrobotoolkit_sdk as xrt
xrt.init()
print("callback API:", xrt.has_frame_callback())
xrt.close()
PY
```

## 启动与连接顺序

在运行 HEFT player 前，按下面顺序启动。PICO 本身没有可由本仓库执行的 shell 命令；它需要在
头显的 XRoboToolkit App 中完成连接。

终端 1，启动 XRoboToolkit PC service（保持这个终端运行）：

```bash
./scripts/start_xrobottoolkit_service.sh
```

默认服务位置是 `/opt/apps/roboticsservice/runService.sh`。若安装在其他路径：

```bash
XROBOT_SERVICE_DIR=/opt/xrobot/roboticsservice ./scripts/start_xrobottoolkit_service.sh
```

PICO XRoboToolkit App 中需要开启：

1. 与 PC service 在同一局域网，并填入 **运行 service 的主机 LAN IPv4 地址**（不要填
   `127.0.0.1`、Docker 或代理软件的虚拟网卡地址）。可用 `hostname -I` 查看候选地址，选择与 PICO
   处在同一网段的地址。
2. 点击连接，确认 App 显示已连接到 PC service。
3. 开启 **完整 Body tracking** 与两个 **Controller tracking**。
4. 保持中立站立姿态；首次完整 skeleton 帧会被用作校准。

终端 2，PICO 已连接后启动 HEFT：

```bash
./play_g1_heft_pico.sh
```

远程 OrcaGym：

```bash
./play_g1_heft_pico.sh --remote 192.168.1.20:50051
```

PICO 没有连接时，可用同一条 HEFT 在线参考链路发送单臂向前伸展动作；此模式**不加载 XRoboToolkit
SDK，也不会等待 PICO 数据**：

```bash
./play_g1_heft_pico.sh --demo-hand-forward --demo-side right
```

动作从当前站立参考平滑过渡到右臂前伸并保持。用 `--demo-side left` 切换左臂；`Q`/`Esc` 退出，或加
`--seconds 6` 自动结束。浏览器中的 `/teleop/reference_g1` 红色骨架会显示实际发送的参考姿态。

需要验证较长时间的在线参考与可视化时，使用 60 秒的单臂连续轨迹：从肩部抬离躯干、肩前伸、
肩外扫、肩上举、腕部绕转、再次前伸、收回、回到站立。它同样不需要 PICO 或 XRoboToolkit SDK，
且只修改选中一侧的 7 DoF；另一只手、腰和双腿始终保持启动站立参考：

```bash
./play_g1_heft_pico.sh --demo-arm-sequence --demo-side right
```

默认运行 60 秒后自动回到站立并退出；可通过 `--demo-sequence-seconds 90` 调整总时长，也可加
`--seconds 20` 仅截取前 20 秒。

当终端持续显示 `frames=` 且 `rejected=` 不再增加时，表示已经收到了可用于完整骨架 retarget 的
PICO pose。若 `frames=0`，先检查 PICO App 是否连接到正确的主机 LAN IP、完整 Body tracking 是否已开启，
以及 `xrobotoolkit_sdk` 是否安装在当前 Python 环境。

默认会启用 MJViser 浏览器查看器。打开 `http://localhost:8080`；远程运行时请转发该端口。
黄色点和坐标轴是**经过首次中立帧标定后**的肩、肘、腕 IK 目标，因此与 G1 处于同一坐标系；
不是未经对齐的 PICO 世界坐标。机器人模型显示实际发送给 HEFT 的、经过限速后的 14-DoF 手臂目标。
如果 OrcaLab USD 网格无法被 MJViser 转换，`/teleop/reference_g1` 下的红色躯干和双臂骨架仍会
显示同一套实时 HEFT reference qpos。它直接由发送给 HEFT 的 29-D qpos 与 G1 模型的真实局部
关节轴生成；不显示固定的双腿，避免在裁切视角中误认为手臂连在脚上。它是浏览器可视化的可靠兜底，
而非第二套 retarget。

首次收到完整 skeleton 时，脚本把该帧设为中立姿态。因此启动后请保持舒适的站立和中立手臂姿势
一小段时间。之后移动手臂即可；完整 skeleton 只作为输入格式，重定向只读取肩、肘、腕链并只发送
双臂的结果给 HEFT。

Dex3-1 在启动时默认完全闭合。PICO player 的键盘快捷键为：`O` 完全张开、`C` 完全闭合、`Q` 或
`Esc` 退出。`R` 复位仿真与 HEFT 历史，并在下一个有效 PICO 帧重新标定。没有全局键盘权限时，
用 `--keyboard-backend terminal`。

## 参数与安全边界

```bash
./play_g1_heft_pico.sh \
  --position-scale 0.8 \
  --ik-iterations 6 \
  --max-joint-step 0.08
```

- `--position-scale`：人手臂位移的缩放系数；较大幅度时建议先降低。
- `--max-joint-step`：每个 PICO 帧允许的单关节最大 IK 更新量（弧度），用于限制跳变。
- `--max-reference-step`：每个 PICO 帧允许发布到 HEFT 的手臂 qpos 最大变化（弧度）；默认 `0.06`，
  用于限制 OOD 跳变。
- `--no-visualize`：没有 `mjviser/viser` 或不需要浏览器时关闭查看器。
- `Ctrl-C`：清除 XR callback、关闭查看器和 OrcaGym 环境。

这是仿真联调入口，不应直接用于真机。先在 OrcaLab 中确认姿态方向、关节限位与 HEFT 跟踪稳定性，
再考虑接入任何真实机器人命令通道。
