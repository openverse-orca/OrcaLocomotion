# Orca Warp 案例手册

以下命令均从 OrcaLocomotion 仓库根目录执行。

## 1. 最小训练检查

使用少量环境和 iteration 验证完整训练链路：

```bash
orca train --task G1-Velocity-Flat \
  --num-envs 256 --iterations 3 \
  --device cuda:0 --wandb-mode disabled \
  --log-dir logs/smoke
```

预期生成 `logs/smoke/model_final.pt`。

## 2. 标准训练

```bash
orca train --task G1-Velocity-Flat \
  --num-envs 4096 --device cuda:0 \
  --runner-config configs/train/ppo.yaml \
  --log-dir logs/g1_flat \
  --wandb-mode online
```

## 3. 断点续训

```bash
orca train --task G1-Velocity-Flat \
  --resume logs/g1_flat/model_100.pt \
  --num-envs 4096 --device cuda:0 \
  --log-dir logs/g1_flat_resume
```

## 4. Headless 并行回放

```bash
orca play --task G1-Velocity-Flat \
  --checkpoint logs/g1_flat/model_final.pt \
  --num-envs 300 --steps 10000 \
  --device cuda:0 --no-realtime
```

## 5. OrcaLab 批量回放

启动 OrcaLab 无仿真程序模式后：

```bash
orca play --task G1-Velocity-Flat \
  --checkpoint logs/g1_flat/model_final.pt \
  --num-envs 300 --device cuda:0 \
  --orcalab --render-fps 30
```

首次执行会发布机器人场景。后续复用已发布场景时可增加 `--no-publish`。

## 6. 密集排列 300 个机器人

```bash
orca play --task G1-Velocity-Flat \
  --checkpoint logs/g1_flat/model_final.pt \
  --num-envs 300 --device cuda:0 --orcalab \
  --spacing 0.6 \
  --spawn-range 4.0 \
  --root-xy-scale 0.2
```

- `--spacing`：机器人网格间距；
- `--spawn-range`：布局中心区域半宽；
- `--root-xy-scale`：仅压缩渲染层 root x/y 位移，不修改物理状态。

## 7. 自定义 G1 XML

训练：

```bash
orca train --task G1-Velocity-Flat \
  --asset /absolute/path/to/g1.xml \
  --num-envs 4096 --device cuda:0
```

回放时应使用同一模型：

```bash
orca play --task G1-Velocity-Flat \
  --asset /absolute/path/to/g1.xml \
  --checkpoint logs/rsl_rl/model_final.pt \
  --num-envs 300 --device cuda:0
```

自定义 XML 必须保持任务使用的 joint、actuator、body 和 sensor 名称。

## 8. 自定义 OrcaLab 机器人 prefab

先在 OrcaLab 资产平台订阅 `unitree_robots`。使用其它机器人 prefab 时，也必须先订阅
该资产所在的资产包。

```bash
orca play --task G1-Velocity-Flat \
  --checkpoint logs/rsl_rl/model_final.pt \
  --orcalab \
  --asset-path assets/<project>/prefabs/<g1_asset>
```

`--asset-path` 只影响 OrcaLab 渲染；训练模型仍由内置 XML 或 `--asset` 决定。
