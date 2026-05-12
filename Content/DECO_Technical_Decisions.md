# DECO 技术决策记录

> 最后更新：2026-05-12
> 用途：记录 Kuavo-DECO 集成过程中已经确认、仍待验证、以及被放弃的关键技术方案。后续阶段二、阶段三的实现细节也应继续追加到本文档，便于长期 review。

---

## 1. 文档定位

本文档不是运行手册，而是技术决策记录。它用于回答：

- 为什么阶段一要新建 DECO 专用 rosbag 转换脚本。
- 哪些 topic、字段和维度会进入 LeRobot 数据集。
- 当前采用了哪些实现策略，备选方案是什么。
- 哪些细节必须等待 Inspector 或用户 review 后才能冻结。

阶段一代码实现时，应优先遵守本文档；若后续 Inspector 结果或真实数据结构与本文档冲突，应先更新本文档，再修改代码。

---

## 2. 阶段一：数据引擎技术决策

### 2.1 目标

阶段一的目标是新建 `kuavo_data/CvtRosbag2Lerobot_DECO.py`，将 Kuavo rosbag 转换为 DECO 训练 wrapper 可以直接消费的 LeRobot 数据集。

目标输出字段：

| LeRobot 字段 | 目标形状 | 语义 |
| --- | --- | --- |
| `observation.state` | `(28,)` | DECO 本体状态：左臂 7 + 左手 6 + 右臂 7 + 右手 6 + 头部 2 |
| `action` | `(28,)` | DECO 动作：左臂 7 + 左手 6 + 右臂 7 + 右手 6 + 头部 2 |
| `observation.tactile` | `(30,)` | 左手 15 + 右手 15 的法向触觉力，单位为牛顿 |
| `observation.images.*` | 自动推断 | 头部 RGB 图像，最终策略等待 Inspector 冻结 |

---

### 2.2 数据配置文件

已确认方案：

- 新建配置文件：`configs/data/KuavoRosbag2Lerobot_deco.yaml`
- 默认训练频率：`train_hz: 10`
- 深度图：`use_depth: false`
- 转换阶段不做图像 resize。

重要注释要求：

- 在 `_deco.yaml` 中明确标注 `train_hz` 是 10Hz/30Hz 切换点。
- 当前先用 10Hz 是因为现阶段数据与硬件链路未确认能稳定对齐 30Hz。
- 若后续切换到 30Hz，需要同步确认主相机帧率、触觉频率、动作 topic 频率以及 LeRobot 写入速度。

备选方案：

- 若后续确认 rosbag 全模态稳定支持 30Hz，可将 `train_hz` 改为 `30`。
- 若某些老数据包只有 10Hz 可用，保留 10Hz 配置作为兼容路径。

---

### 2.3 视觉源策略

当前已知 rosbag 记录 topic 包含：

- `/cam_h/color/image_raw/compressed`
- `/cam_l/color/image_raw/compressed`
- `/cam_r/color/image_raw/compressed`
- `/cam_h/depth/image_raw/compressed`
- `/cam_l/depth/image_rect_raw/compressed`
- `/cam_r/depth/image_rect_raw/compressed`

已确认约束：

- DECO 原生模型只需要两路 RGB 图像输入，不使用 depth。
- 当前阶段一必须忽略所有 depth topic。
- 机器人头部相机 Gemini-335L 的 RGB 输出很可能是单路 RGB 彩色图，而不是左右拼接双目图。
- 用户在 Foxglove 中看到 `/cam_h/color/image_raw/compressed` 更像一整块头部相机图像。

当前推荐默认策略：

- 阶段一先通过 Inspector 确认 `/cam_h/color/image_raw/compressed` 的第一帧尺寸、宽高比和可视化结果。
- 若 `/cam_h` 是单目整图：LeRobot 数据集中只保存 `observation.images.head_cam_h`，阶段四 wrapper 中将同一张图复制为 DECO 的 `img1/img2`。
- 若 `/cam_h` 是左右拼接图：转换阶段将其按宽度中线切分为 `observation.images.head_cam_left` 和 `observation.images.head_cam_right`。

备选方案：

- 若 Inspector 或硬件文档确认 `/cam_l`、`/cam_r` 是头部左右目，而不是腕部相机，则可以使用 `/cam_l` 和 `/cam_r` 作为 DECO 的 `img1/img2`。
- 若 `/cam_l`、`/cam_r` 是腕部相机，则不作为 DECO 默认视觉源，以避免偏离 DECO 原生头部双视角假设。

Review 决策点：

- Inspector 必须导出 `cam_h_full.jpg`、`cam_h_left_half.jpg`、`cam_h_right_half.jpg`。
- 用户根据导出图判断 `/cam_h` 是否需要切分。
- 最终视觉策略冻结后，再进入完整转换脚本实现。

---

### 2.4 图像 resize 策略

已确认方案：

- 转换阶段不 resize 图像。
- 图像尺寸从第一帧自动推断，并用于注册 LeRobot image feature。
- 后续在训练 wrapper 或模型预处理阶段统一 resize/letterbox 到 DECO 所需的 `256x256`。

原因：

- 转换阶段保留原始 RGB 信息，避免过早破坏视野比例和细节。
- 训练阶段可以更灵活地替换 resize、letterbox、crop 和数据增强策略。

备选方案：

- 若数据量过大或写入速度成为瓶颈，可在 `_deco.yaml` 中增加显式 `resize` 配置，并在转换阶段保存较小图像。
- 该备选方案需要在代码注释中清楚说明会改变数据集中的视觉原始信息。

---

### 2.5 28 维 state/action 映射

DECO 固定 28 维顺序：

```text
左臂 7 + 左手 6 + 右臂 7 + 右手 6 + 头部 2
```

已确认方案：

- 输出必须固定为 28 维。
- 头部两个自由度当前在实机中锁定，但模型接口必须保留。
- `observation.state[26:28]` 优先使用 Inspector 验证为稳定的 `/sensors_data_raw.joint_data.joint_q[26:28]` 实测固定角度；若读不到、字段异常或跨帧不稳定，则回退为 `[0.0, 0.0]`。
- `action[26:28]` 当前始终补 `[0.0, 0.0]`，表示阶段一 DECO 策略暂不输出头部控制。

推荐 state 来源：

| 维度 | 来源 topic | 字段/切片 |
| --- | --- | --- |
| 左臂 7 | `/sensors_data_raw` | `joint_data.joint_q[12:19]` |
| 左手 6 | `/dexhand/state` | `position[:6]` |
| 右臂 7 | `/sensors_data_raw` | `joint_data.joint_q[19:26]` |
| 右手 6 | `/dexhand/state` | `position[6:12]` |
| 头部 2 | `/sensors_data_raw` 优先，fallback 补零 | `joint_data.joint_q[26:28]`；若缺失或不稳定则 `[0.0, 0.0]` |

推荐 action 来源：

| 维度 | 首选来源 | fallback |
| --- | --- | --- |
| 左臂 7 | `/kuavo_arm_traj` 的左臂部分 | `/joint_cmd` 的上肢部分 |
| 左手 6 | `/control_robot_hand_position` 的左手部分 | 必要时检查 `/joint_cmd` 是否含手部命令 |
| 右臂 7 | `/kuavo_arm_traj` 的右臂部分 | `/joint_cmd` 的上肢部分 |
| 右手 6 | `/control_robot_hand_position` 的右手部分 | 必要时检查 `/joint_cmd` 是否含手部命令 |
| 头部 2 | 补零 `[0.0, 0.0]` | 暂不使用 `/robot_head_motion_data` |

待 Inspector 确认：

- `/kuavo_arm_traj` 的 `position` 是否始终为 14 维，且顺序为左臂 7 + 右臂 7。
- `/control_robot_hand_position` 的左右手字段名称和长度。
- `/joint_cmd` 中是否包含完整 28 维命令，以及手部/头部索引是否与计划一致。
- `/sensors_data_raw.joint_data.joint_q[26:28]` 是否存在且跨帧稳定；Inspector 需要打印前若干帧、最小值、最大值和稳定性判断，用于决定 `observation.state[26:28]` 使用实测固定角还是补零。

---

### 2.6 触觉策略

已确认方案：

- topic：`/dexhand/touch_state`
- 消息类型预期：`kuavo_msgs/dexhandTouchState`
- 只提取每个手指的 `normal_force1/2/3`。
- 舍弃切向力、切向方向、接近觉和状态字段。
- 单手维度：`5 指 × 3 点 = 15`
- 双手维度：`30`
- 数值换算：原始值为 `100 * N`，转换时除以 `100.0`，得到牛顿。

输出顺序：

```text
左手 finger0 normal_force1/2/3, ..., finger4 normal_force1/2/3,
右手 finger0 normal_force1/2/3, ..., finger4 normal_force1/2/3
```

缺失策略：

- `/dexhand/touch_state` 缺失时直接报错，不静默补零。
- 原因：阶段一目标就是构造触觉版 DECO 数据，触觉缺失会让训练目标变质。

---

### 2.7 时间对齐策略

已确认方案：

- 使用主视觉时间轴作为采样时间轴。
- 默认 `train_hz: 10`。
- 对每个目标时间戳，在其他 topic 中选择绝对时间差最小的一帧，即 nearest-neighbor 对齐。
- 这是当前 `CvtRosbag2Lerobot.py` 中已使用的核心思想，但 DECO 版本先放在新脚本中实现，不修改公共 reader。

备选方案：

- 后续若动作或触觉存在明显抖动，可考虑线性插值 arm/head state/action。
- 触觉仍建议 nearest-neighbor 或窗口统计，因为触觉接触信号可能包含瞬态峰值，简单插值可能改变物理含义。

---

### 2.8 缺失 topic 策略

已确认方案：

必须存在，否则报错：

- `/cam_h/color/image_raw/compressed`
- `/sensors_data_raw`
- `/dexhand/state`
- `/dexhand/touch_state`
- `/kuavo_arm_traj` 或 `/joint_cmd` 至少一个可用于 arm action
- `/control_robot_hand_position`

允许补零：

- `action[26:28]` 当前始终补零，因为阶段一暂不让策略输出头部控制。
- `observation.state[26:28]` 仅在 Inspector 读不到稳定实测固定角时补零；若 `/sensors_data_raw.joint_data.joint_q[26:28]` 可读且稳定，则应使用该固定角。

暂不使用但保留扩展注释：

- `/robot_head_motion_data`
- depth 相关 topic
- `/cam_l/color/image_raw/compressed`
- `/cam_r/color/image_raw/compressed`
- `/humanoid_controller/wbc_arm_eef_pose`

---

### 2.9 公共 reader 策略

已确认方案：

- 阶段一暂不修改 `kuavo_data/common/kuavo_dataset.py`。
- 新建 `kuavo_data/CvtRosbag2Lerobot_DECO.py`，内部实现 DECO 专用 topic map、字段解析和 mapper。

原因：

- 避免影响 ACT/DP 既有数据转换链路。
- DECO 需要固定 28 维 state/action 和 30 维 tactile，与当前配置化拼接逻辑差异较大。

备选方案：

- 当 DECO 转换流程稳定后，可将通用 rosbag 读取、时间对齐、图像解码逻辑抽回公共模块，减少重复。

---

## 3. 阶段一执行切分

### 3.1 已完成：1.0 技术决策记录与方案冻结

- 更新 `PLANS.md` 阶段一结构。
- 新建本文档。
- 记录已确认方案、备选方案和待 Inspector 确认项。

### 3.2 下一步：1.1 Rosbag Schema Inspector

计划新建：

```text
kuavo_data/inspect_deco_stage1_schema.py
```

脚本要求：

- 只读 rosbag，不写 LeRobot 数据集。
- 不修改原始 rosbag。
- 默认读取 `data_example/vr_record_2026-04-15-15-57-47.bag`。
- 输出关键 topic 的存在性、消息数量、时间范围、估算频率、字段长度。
- 导出 `/cam_h/color/image_raw/compressed` 第一帧完整图和左右半图。
- 打印 `/sensors_data_raw.joint_data.joint_q[26:28]` 的前若干帧、最小值、最大值和稳定性判断，作为头部 `observation.state` 是否使用实测固定角的冻结依据。

预期导出目录：

```text
data_example/inspect_outputs/
```

预期导出图片：

```text
cam_h_full.jpg
cam_h_left_half.jpg
cam_h_right_half.jpg
```

### 3.3 后续：1.2 Review 与 1.3 完整转换

- 用户运行 Inspector 后反馈终端输出和三张图的观察结论。
- 根据反馈冻结视觉策略。
- 再进入 `CvtRosbag2Lerobot_DECO.py` 和 `_deco.yaml` 的完整实现。

---

## 4. 跨阶段提醒

### 4.1 Wrapper loss 修正

DECO 源码中的训练目标不是旧计划中写的 `F.mse_loss(act, noise)`。

应保留原生 Flow Matching 目标：

```python
loss = F.mse_loss(out, noise - action)
```

其中：

- `out`：DECO 网络预测的速度场/残差方向。
- `noise - action`：从专家动作指向噪声样本的目标速度场。

该决策已同步写入 `PLANS.md` 阶段四，后续 `DECOPolicyWrapper.forward` 必须遵守。

---

## 5. 待补充章节

- 阶段二：DECO 代码物理迁移、依赖剥离、路径桥接。
- 阶段三：触觉编码器手术、模型维度修改、预训练权重兼容性。
- 阶段四：LeRobot wrapper、训练 forward、推理 select_action。
- 阶段五：DECO policy config 与 normalization mapping。
- 阶段六：部署侧 `select_action` 接口与真机/仿真闭环。
