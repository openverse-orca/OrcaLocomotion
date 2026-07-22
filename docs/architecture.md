# Orca Warp 架构

## 设计目标

Orca Warp 对应用只提供一个统一入口：`orca`。训练、检查、性能测试和回放使用同一套
任务注册与运行配置，调用方不需要了解运行时内部结构。

```text
应用
  ├─ orca 命令行
  └─ orcalab_rslrl.orca Python API
          ↓
任务注册表 + OrcaRuntimeConfig
          ↓
RslRlVecEnvAdapter
          ↓
ManagerBasedRLEnv
          ↓
OrcaPhysics
          ↓
Orca GPU Runtime
          ├─ 批量训练状态
          └─ OrcaLab 实时回放
```

稳定的公共接口包括：

- `OrcaRuntimeConfig`：设备、并行环境数、随机种子和资产覆盖等运行参数；
- `make_env`、`list_tasks`、`register_task`：环境创建与任务注册；
- `OrcaPhysics`、`OrcaState`、`OrcaCapabilities`：任务开发所需的状态与能力接口；
- `orca train/play/inspect/benchmark`：产品命令行入口。

应用和任务代码不应导入 `orcalab_rslrl._internal`。该目录只承载运行时内部实现，
不属于公共接口。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `orca` | 公共 API、任务注册、运行配置与命令行入口 |
| `tasks` | 组合机器人、场景、MDP 与训练器配置 |
| `envs` | 管理批量环境的 step、reset 与 decimation 生命周期 |
| `managers` / `mdp` | command、observation、reward、event、termination 与 action term |
| `robots` / `assets` | 机器人定义、默认姿态及仓库内置训练资产 |
| `rslrl` | 将 Orca 环境适配为 RSL-RL 向量环境 |
| `recording` | 记录明确启用的运行数据 |
| `tools` | 性能检查和开发诊断工具 |

## 环境生命周期

`ManagerBasedRLEnv` 负责完整的 step/reset 生命周期。任务配置 command、observation、
reward、termination、event、action transform 和 decimation；各 term 通过 `env.orca`
读取稳定的批量 tensor，不直接调用运行时内部函数。

所有运行状态的首维都是 `num_envs`。动作写入、状态读取、奖励计算、终止判断与局部
reset 均保留在配置的设备上。只有日志、checkpoint、诊断和实时回放等明确边界允许
进行 CPU 同步。

## 训练数据流

```text
策略动作
   ↓
动作缩放与默认姿态偏移
   ↓
Orca 批量 step
   ↓
状态、传感器与 command
   ↓
observation / reward / termination
   ↓
RslRlVecEnvAdapter
   ↓
RSL-RL PPO
```

训练默认采用 headless 模式。RSL-RL 只依赖 `RslRlVecEnvAdapter`，W&B 只接收按
iteration 汇总的指标，避免媒体处理和网络操作阻塞 rollout。

## OrcaLab 实时回放

OrcaLab bridge 从环境读取批量位姿并按 `--render-fps` 推送。渲染频率可以低于控制
频率，关闭回放连接不会改变任务语义。`--spacing`、`--spawn-range` 和
`--root-xy-scale` 只改变 OrcaLab 中的显示布局，不修改训练状态。

实时回放前必须在 OrcaLab 资产平台订阅作者为 **Orca** 的 `unitree_robots`，并确认
状态为“已订阅”。训练所需的 XML 与 mesh 随仓库发布，不依赖 OrcaLab 连接。

## 资产约束

发布的机器人 XML、mesh、actuator 定义和默认姿态位于
`orcalab_rslrl/assets/robots`。任务启动时必须检查 joint、body、actuator 和 sensor
名称；缺失或名称不一致时直接报错，不允许静默回退到固定索引。

`OrcaRuntimeConfig.asset` 或 `--asset` 可显式覆盖训练模型；发布任务默认应能直接使用
仓库内置资产。`--asset-path` 仅用于指定 OrcaLab 中已订阅的显示资产。

## 扩展约束

- 新任务通过注册表提供独立任务 ID，不修改已有任务的语义；
- MDP term 只读取 `env.orca` 的公共状态与映射；
- OrcaLab bridge 只使用公共状态视图，不反向修改环境状态；
- 新增公共符号时必须同时提供类型标注、文档和接口测试；
- 运行时内部调整不得改变任务配置、checkpoint、命令行或 MDP term 的使用方式。
