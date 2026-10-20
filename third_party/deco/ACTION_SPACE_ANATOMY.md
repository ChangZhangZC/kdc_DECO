# DECO Action Space Anatomy

本文档详细梳理了 DECO 模型的输入输出维度、物理含义及其在源码中的对应关系，作为修改网络维度（如适配 Kuavo 机器人）的参考指南。

## 1. 核心维度概述

| 参数 | 数值 | 描述 |
| :--- | :--- | :--- |
| `act_dim` | 28 | 动作/本体感知状态的特征维度 |
| `chunk_size` | 32 | 时间预测步长 (Horizon $H$) |
| `observation_shape` | (B, 28) | 模型输入的本体感知状态形状 |
| `action_shape` | (B, 32, 28) | 模型预测的动作轨迹形状 |

---

## 2. 28 维动作空间详细映射 (Joint Mapping)

在模型训练与推理过程中，28 维向量的排列顺序严格遵循 **左侧全集 -> 右侧全集 -> 头部** 的逻辑：

| 维度索引 (Index) | 自由度 (DOF) | 物理含义 | 对应硬件部件 |
| :--- | :--- | :--- | :--- |
| **0 - 6** | 7 | 左臂关节状态 (Left Arm) | 手臂电机 (7-DOF) |
| **7 - 12** | 6 | 左手手指状态 (Left Hand) | Inspire 灵巧手 (6-DOF) |
| **13 - 19** | 7 | 右臂关节状态 (Right Arm) | 手臂电机 (7-DOF) |
| **20 - 25** | 6 | 右手手指状态 (Right Hand) | Inspire 灵巧手 (6-DOF) |
| **26 - 27** | 2 | 头部/相机状态 (Head) | Active Cam (Yaw, Pitch) |

---

## 3. 触觉空间 (Tactile Data)

当 `use_tactile: True` 时，模型额外处理双手的触觉信息：

- **DECO 原生输入**: 左右手各 1062 维 Inspire Hand 触觉像素点。
- **Kuavo-DECO 当前输入**: 左右手各 15 维 normal force，合计 30 维。
- **归一化边界**: 数据转换阶段的 `/100` 只负责把 Kuavo 原始 normal force 转成牛顿；进入模型前仍需由 wrapper / dataset 按 `tactile_left_max` 与 `tactile_right_max` 做 DECO-style `tac / tactile_max` 归一化。
- **模型手术结果**: `init_tac_regions` 的 1062 维区域均值逻辑已从 `third_party/deco` Kuavo 定制副本中移除；模型直接使用 `tac1=[B,15]` 与 `tac2=[B,15]`。
- **触觉 token 维度**: `tactile_encoder` 执行 `30D -> 34D`，再与左右手 15D 归一化触觉拼接，形成 `15 + 15 + 34 = 64` 个触觉条件位置进入跨模态注意力机制。

---

## 4. 源码查证指南

### 数据输入侧
- [dataset.py](dataset.py): 
    - `line 112`: 拼接 `left_obs`, `right_obs`, `head_obs` 构建本体感知向量。
    - `line 115-117`: 拼接 `left_action`, `right_action`, `head_action` 构建专家动作。
- [inference.py](inference.py): 
    - `preprocess` 函数对 28 维输入进行均值/标准差归一化。

### 模型定义侧
- [models/deco/deco.py](models/deco/deco.py):
    - `__init__`: 初始化 `obs_encoder` 和 `action_encoder`，维度均为 `act_dim`。
    - `self.linear`: 最终动作预测头，输出维度为 `act_dim`。

### 部署控制侧
- [deploy/deploy_h1.py](deploy/deploy_h1.py):
    - `line 198-210`: 实时读取硬件状态并按索引 `0-6, 7-12, 13-19, 20-25, 26-27` 填入状态向量。
    - `line 251-267`: 将模型预测的 28 维动作切片，分发给手臂控制器、手指控制器和头部云台。

---

## ⚠️ 重要注意事项：顺序一致性

在原始录制的 `data.json` 中，数据块的存储顺序可能为 `L_arm, R_arm, L_hand, R_hand, head`。但在进入 DECO 模型之前，**必须** 转换为上述 `L_arm, L_hand, R_arm, R_hand, head` 的顺序。

> **如果需要适配新机器人（如 Kuavo）**：
> 1. 修改 `config/deco.yaml` 中的 `action_dim`。
> 2. 确认 `observation.state` / `action` 保持 `L_arm, L_hand, R_arm, R_hand, head` 的 28 维顺序。
> 3. 确认触觉进入模型前已经拆成左手 15D 与右手 15D，并完成 DECO-style tactile max 归一化。
> 4. 更新部署侧控制分发逻辑，避免把 30Hz action chunk 直接当作 10Hz 控制频率使用。
