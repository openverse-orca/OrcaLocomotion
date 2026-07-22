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

启动 OrcaStudio 无仿真程序模式后：

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

## 7. 楼梯地形

本地物理地形和 OrcaStudio prefab 必须同时配置：

```bash
orca play --task G1-Velocity-Flat \
  --checkpoint logs/g1_flat/model_final.pt \
  --num-envs 300 --device cuda:0 --orcalab \
  --terrain-kind stair-mid-flat \
  --terrain-asset-path \
    assets/e071469a36d3c8aa/default_project/prefabs/terrain_stair_mid_flat_usda
```

如果订阅后的资产路径不同，请替换为 OrcaStudio 中实际的 spawnable asset 路径。

视觉位置偏移可通过以下参数校准：

```bash
--terrain-align-offset X Y Z
--render-root-offset X Y Z
```

前者移动机器人布局中心，后者只修改每帧显示的 root 位置；两者都不改变本地碰撞几何。

## 8. 自定义 G1 XML

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

## 9. 自定义 OrcaStudio 机器人 prefab

```bash
orca play --task G1-Velocity-Flat \
  --checkpoint logs/rsl_rl/model_final.pt \
  --orcalab \
  --asset-path assets/<project>/prefabs/<g1_asset>
```

`--asset-path` 只影响 OrcaStudio 渲染；物理模型仍由内置 XML 或 `--asset` 决定。

## 10. 排查 OrcaLab 地形碰撞

导出 OrcaLab 编译场景中的非机器人碰撞几何：

```bash
orca play --task G1-Velocity-Flat \
  --checkpoint logs/rsl_rl/model_final.pt \
  --num-envs 1 --device cuda:0 --orcalab \
  --terrain-asset-path assets/<project>/prefabs/<terrain_asset> \
  --dump-orcalab-geoms logs/orcalab_geoms.json \
  --dump-orcalab-geoms-only
```

JSON 为空通常表示 prefab 没有下发可碰撞几何。增加
`--dump-orcalab-all-geoms` 可同时检查 visual-only geom。
