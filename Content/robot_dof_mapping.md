# Kuavo 机器人自由度 (DOF) 与索引映射对照表

本文件汇总了 Kuavo 机器人（重点针对上半身）的自由度统计及在运控脚本（如 `.tact` 动作文件）中的索引映射关系。

## 1. 自由度统计 (Upper Body DOF)

| 部件 | 自由度 (DOF) | 说明 |
| :--- | :---: | :--- |
| **手臂 (Arms)** | 14 | 左臂 7 + 右臂 7 |
| **灵巧手 (Hands)** | 12 | 左手 6 + 右手 6 (Qiangnao 灵巧手) |
| **头部 (Head)** | 2 | Yaw (左右) + Pitch (上下) |
| **总计 (Upper Body)** | **28** | 运控脚本中标准的 28 位 servos 数组 |

> [!NOTE]
> **关于腰部 (Waist)**：物理机器人通常包含 1 个腰部自由度（V52版本），但在高层动作序列（.tact）中通常不包含在 28 位 servos 数组内，而是由底层控制器维护。

---

## 2. 运控脚本索引映射 (Standard 28-DOF Mapping)

此映射关系适用于 `src/demo/examples_code/hand_plan_arm_trajectory/` 下的示例脚本及 `.tact` 动作文件。

### 2.1 手臂 (Index 0 - 13)

| Index | 关节名称 | 描述 |
| :---: | :--- | :--- |
| 0 | `l_arm_pitch` | 左臂俯仰 |
| 1 | `l_arm_roll` | 左臂横滚 |
| 2 | `l_arm_yaw` | 左臂偏航 |
| 3 | `l_forearm_pitch` | 左前臂俯仰 |
| 4 | `l_hand_yaw` | 左手腕偏航 |
| 5 | `l_hand_pitch` | 左手腕俯仰 |
| 6 | `l_hand_roll` | 左手腕横滚 |
| 7 | `r_arm_pitch` | 右臂俯仰 |
| 8 | `r_arm_roll` | 右臂横滚 |
| 9 | `r_arm_yaw` | 右臂偏航 |
| 10 | `r_forearm_pitch` | 右前臂俯仰 |
| 11 | `r_hand_yaw` | 右手腕偏航 |
| 12 | `r_hand_pitch` | 右手腕俯仰 |
| 13 | `r_hand_roll` | 右手腕横滚 |

### 2.2 灵巧手 (Index 14 - 25)

| Index | 关节名称 | 描述 |
| :---: | :--- | :--- |
| 14 | `l_thumb` | 左手大拇指 |
| 15 | `l_thumb_aux` | 左手大拇指辅助 |
| 16 | `l_index` | 左手食指 |
| 17 | `l_middle` | 左手中指 |
| 18 | `l_ring` | 左手无名指 |
| 19 | `l_pinky` | 左手小拇指 |
| 20 | `r_thumb` | 右手大拇指 |
| 21 | `r_thumb_aux` | 右手大拇指辅助 |
| 22 | `r_index` | 右手食指 |
| 23 | `r_middle` | 右手中指 |
| 24 | `r_ring` | 右手无名指 |
| 25 | `r_pinky` | 右手小拇指 |

### 2.3 头部 (Index 26 - 27)

| Index | 关节名称 | 描述 |
| :---: | :--- | :--- |
| 26 | `head_yaw` | 头部左右 |
| 27 | `head_pitch` | 头部上下 |

---

## 3. 底层传感器数据索引 (`sensors_data_raw`)

在 `/sensors_data_raw` 话题或 `robotState.msg` 的 `joint_data.joint_q` 数组中，由于包含腿部和腰部，索引会发生偏移：

| 关节组 | Kuavo V4x / V49 索引 | Kuavo V52 索引 |
| :--- | :---: | :---: |
| **腿部 (Legs)** | 0 - 11 | 0 - 11 |
| **腰部 (Waist)** | - | 12 |
| **手臂 (Arms)** | 12 - 25 | 13 - 26 |
| **头部 (Head)** | 26 - 27 | 27 - 28 |

---

## 5. ROS 话题与功能对应关系

为了便于在录制和开发中针对性地寻找话题，以下将常用话题按功能区域进行了分类：

### 5.1 上半身 (Upper Body: Arms & Head)
主要控制手臂姿态、轨迹以及头部的运动。

| 话题名称 | 功能描述 | 备注 |
| :--- | :--- | :--- |
| `/kuavo_arm_traj` | 手臂关节空间轨迹 | 用于记录/控制双臂 14 个关节 |
| `/robot_head_motion_data` | 头部运动控制 | 控制 Yaw/Pitch 2 个自由度 |
| `/arm_joint_states` | 手臂关节状态反馈 | 实时反馈手臂角度、速度等 |

### 5.2 灵巧手 (Dexterous Hands)
专门用于控制和反馈大连乐聚或强脑（Qiangnao）灵巧手的开合程度。

| 话题名称 | 功能描述 | 备注 |
| :--- | :--- | :--- |
| `/control_robot_hand_position` | 灵巧手目标位置 | 发送 12 通道的手指开合指令 |
| `/robot_hand_position` | 灵巧手状态反馈 | 反馈当前手指的实际开合度 |
| `/dexhand/touch_state` | 触觉传感器数据 | 灵巧手阵列触觉信息（若配备） |

### 5.3 下半身与核心 (Lower Body & Core)
包含腿部运动、足端位姿、腰部控制以及全身基础数据。

| 话题名称 | 功能描述 | 备注 |
| :--- | :--- | :--- |
| `/sensors_data_raw` | **全身原始传感器数据** | 核心话题：包含腿(12)+腰(1)+臂(14)+头(2) |
| `/robot_waist_motion_data` | 腰部运动控制 | 控制腰部的俯仰或偏航（视版本而定） |
| `/robot_state` | 机器人整机状态 | 包含当前所有关节的位姿和运动学状态 |
| `/foot_pose` | 足端位姿 | 记录左右脚在世界坐标系或机体坐标系的位姿 |
| `/odom` | 里程计数据 | 机器人整体的移动轨迹和速度估计 |

---

## 6. 录制状态核查 (rosbag_tool.py)

目前 `record_topics.json` 的配置状态如下：
*   [x] **上半身 (Arm/Head)**：已开启
*   [x] **灵巧手 (Hand)**：已开启
*   [x] **视觉 (Camera)**：已开启
*   [ ] **下半身 (Leg/Waist/Odom)**：**暂未开启**（若需记录，请手动将 `/sensors_data_raw` 或 `/robot_waist_motion_data` 加入 json）
---

## 7. 话题冗余与嵌套关系说明

在录制的数据中，您可能会发现某些话题的内容高度相似或存在重复，这通常是由以下逻辑导致的：

### 7.1 “全身包”与“局部包”的包含关系
*   **核心话题**：`/sensors_data_raw`
*   **嵌套逻辑**：这是一个“巨型数组”，按照特定索引顺序（见第3节）塞进了腿、腰、臂、头的所有数据。
*   **冗余表现**：`/kuavo_arm_traj` 或 `/arm_joint_states` 中的数据实际上是 `/sensors_data_raw` 中手臂部分的“提取版”。系统单独发布它们是为了让只关注手臂的算法（如视觉抓取）更方便地订阅。

### 7.2 “控制指令 (CMD)”与“状态反馈 (State)”
*   **成对关系**：
    *   指令：`/joint_cmd` / `/kuavo_arm_traj` (大脑想让机器人做的)
    *   反馈：`/sensors_data_raw` / `/dexhand/state` (电机实际做到的)
*   **训练意义**：在模仿学习（Imitation Learning）训练中，通常需要成对记录。指令话题用于作为训练的 Label（标签），而反馈话题用于确认执行效果。

### 7.3 “原始角度”与“派生位姿”
*   **逻辑逻辑**：`/humanoid_controller/wbc_arm_eef_pose`（末端位姿）是通过 `/sensors_data_raw`（关节角度）经由运动学计算（Forward Kinematics）得出的。
*   **存在意义**：为了避免在数据处理阶段重复进行复杂的运动学解算，直接录制计算好的位姿可以大幅提高训练效率。

### 7.4 录制策略建议
*   **数据完整性优先**：务必保留 `/sensors_data_raw`，它是所有关节数据的“后悔药”。
*   **训练便捷性优先**：保留 `/humanoid_controller/wbc_arm_eef_pose` 和 `/tf`，它们能让你在不了解机器人模型的情况下直接获得空间位置。
*   **特殊传感器不可替代**：`/dexhand/touch_state`（触觉）等话题包含非角度类信息，无法从其他话题中推算，必须单独录制。
