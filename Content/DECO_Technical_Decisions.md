# DECO 技术决策记录

> 最后更新：2026-05-13  
> 用途：记录 Kuavo-DECO 集成过程中已经确认、仍待验证、以及被放弃的关键技术方案。本文档应与 `PLANS.md` 保持一致；若二者冲突，以最新 `PLANS.md` 和本文档中标注的“当前冻结方案”为准。

---

## 1. 文档定位

本文档不是运行手册，而是技术决策记录。它用于回答：

- 为什么 Kuavo-DECO 不再沿用 DECO 原生双 RGB 视觉入口。
- 为什么新方案采用 Kuavo/ACT 风格 RGB-D 视觉前端。
- 哪些 topic、字段和维度会进入 LeRobot 数据集。
- 哪些模型结构保留 DECO，哪些结构移植 Kuavo 工具链。
- 为什么保留 DECO 原生 tactile plugin / low-rank adapter 微调范式，以及它和标准 PEFT LoRA 的区别。
- 30Hz 数据频率与 10Hz 部署控制频率如何解耦。

阶段一到阶段六实现时，应优先遵守本文档；若真实 rosbag 或训练框架行为与本文档冲突，应先更新本文档，再修改代码。

---

## 2. 当前冻结的总体技术路线

### 2.1 当前架构

当前冻结方案：

```text
Kuavo RGB + depth + state + action + tactile rosbag
  -> 30Hz LeRobot RGB-D 数据集
  -> Kuavo RGB_Augmenter + Normalizer
  -> RGB ResNet34 + Depth ResNet34
  -> ACT 风格 RGB-depth cross attention fusion
  -> DECO action-token Flow Matching transformer
  -> 可选 tactile PI_Adapter / plugin 低秩触觉适配
  -> 28D action chunk
  -> 部署阶段按 10Hz 控制频率消费动作队列
```

核心原则：

- **视觉输入方式向 Kuavo 工具链靠齐**：使用 RGB + depth，而不是 DECO 原生双 RGB。
- **模型主干向 DECO 靠齐**：保留 DECO action token、Flow Matching 训练目标和去噪推理。
- **backbone 默认向 DECO 容量靠齐**：默认 `resnet34`，允许配置切换 `resnet18`。
- **触觉微调向 DECO 靠齐**：保留源码中的 `plugin=True` / `PI_Adapter` 低秩 adapter 范式，默认冻结预训练主干，只微调触觉 adapter 和必要的 Kuavo 新增桥接模块。
- **频率处理分层**：数据转换阶段负责 30Hz 训练数据；部署 wrapper 负责 10Hz 控制输出。

### 2.2 被替代的旧方案

以下旧方案不再作为主路线：

- `use_depth: false`
- 只保存 `/cam_h` RGB，不保存 depth
- 将 `/cam_h` 单目图复制成 DECO 的 `img1/img2`
- 将 `/cam_h` 左右裁切成伪双目
- 默认 10Hz LeRobot 数据转换频率
- 在阶段一把 depth 剥离以严格保持 DECO 原生假设

保留说明：

- 这些旧方案可以作为 ablation 或紧急 fallback，但不得作为主实现路线写入新代码。

---

## 3. 阶段一：RGB-D 数据引擎技术决策

### 3.1 目标输出字段

新建 `kuavo_data/CvtRosbag2Lerobot_DECO.py`，将 Kuavo rosbag 转换为 DECO wrapper 可以直接消费的 LeRobot 数据集。

目标输出字段：

| LeRobot 字段 | 目标形状 | 语义 |
| --- | --- | --- |
| `observation.state` | `(28,)` | 左臂 7 + 左手 6 + 右臂 7 + 右手 6 + 头部 2 |
| `action` | `(28,)` | 左臂 7 + 左手 6 + 右臂 7 + 右手 6 + 头部 2 |
| `observation.tactile` | `(30,)` | 左手 15 + 右手 15 法向触觉力，单位为牛顿 |
| `observation.images.head_cam_h` | 自动推断或配置指定 | 头部 RGB 图像 |
| `observation.depth_h` 或等价 depth key | 自动推断或配置指定 | 与头部 RGB 对齐的深度图 |

### 3.2 数据配置文件

当前方案：

- 新建配置文件：`configs/data/KuavoRosbag2Lerobot_deco.yaml`
- 默认训练数据频率：`train_hz: 30`
- 深度图：`use_depth: true`
- 主视觉时间轴：优先使用头部 RGB 时间轴。

重要要求：

- 原始采集流可能是 100Hz 或更高，转换脚本必须用真实时间戳生成 30Hz 目标时间轴。
- 不得依赖 `MAIN_TIMELINE_FPS // TRAIN_HZ` 的整数跳帧假设。
- 如果主相机或 depth 实际只有 30Hz，目标时间轴仍以 30Hz 为准。
- 如果某个 episode 的视觉/depth/action/tactile 覆盖时间不一致，应截取所有必需模态共同覆盖的时间段。

### 3.3 RGB-D 视觉源策略

Inspector 已确认：

- `/cam_h/color/image_raw/compressed` 为 848×480 完整头部 RGB 画面。
- 左右半图只是同一画面的裁切，不是真实双目。
- `/cam_h/depth/image_raw/compressed` 存在，后续作为头部 depth 候选源。

当前冻结策略：

- 保存 `observation.images.head_cam_h` 作为 RGB 输入。
- 保存与头部 RGB 对齐的 depth feature。
- DECO wrapper 通过 Kuavo/ACT 风格 RGB-D 视觉前端处理这两路视觉输入。
- 不再把单目 RGB 复制成 `img1/img2`。
- 不再把 `/cam_h` 按宽度中线切成伪双目。

### 3.4 RGB 与 depth 的增强策略

RGB 复用 Kuavo 现有 `RGB_Augmenter`：

- Identity
- ColorJitter：brightness / contrast / saturation / hue
- SharpnessJitter
- RandomMask
- RandomBorderCutout
- GaussianNoise
- GammaCorrection

Depth 策略：

- 与 RGB 共享 crop/resize 的空间变换，保证 RGB-depth 对齐。
- 不做 ColorJitter、Hue、Saturation、Brightness、Gamma 等 RGB photometric augmentation。
- depth 的归一化遵守 Kuavo 现有 depth 配置，不把 depth 当作普通 RGB 图像处理。

### 3.5 28 维 state/action 映射

DECO 固定 28 维顺序：

```text
左臂 7 + 左手 6 + 右臂 7 + 右手 6 + 头部 2
```

已确认方案：

- 输出必须固定为 28 维。
- 当前 rosbag 的 `/sensors_data_raw.joint_data.joint_q` 长度为 28，头部索引使用 V4x/V49 方案 `joint_q[26:28]`。
- 当前样本头部固定姿态均值约为 `[-0.001657, 0.433904]` rad，即 `[-0.09494°, 24.86089°]`。
- `/robot_head_motion_data` 显示 `[0.0, 25.0]`，可交叉确认 pitch 约 25°。
- `observation.state[26:28]` 使用每个 episode 内 `joint_q[26:28]` 的实测固定均值并广播到所有帧。
- `action[26:28]` 当前始终补 `[0.0, 0.0]`，表示阶段一 DECO 策略暂不输出头部控制。

推荐 state 来源：

| 维度 | 来源 topic | 字段/切片 |
| --- | --- | --- |
| 左臂 7 | `/sensors_data_raw` | `joint_data.joint_q[12:19]` |
| 左手 6 | `/dexhand/state` | `position[:6]` |
| 右臂 7 | `/sensors_data_raw` | `joint_data.joint_q[19:26]` |
| 右手 6 | `/dexhand/state` | `position[6:12]` |
| 头部 2 | `/sensors_data_raw` | `joint_data.joint_q[26:28]` 的 episode 均值 |

推荐 action 来源：

| 维度 | 首选来源 | fallback |
| --- | --- | --- |
| 左臂 7 | `/kuavo_arm_traj` 左臂部分 | `/joint_cmd` 上肢部分 |
| 左手 6 | `/control_robot_hand_position` 左手部分 | 必要时检查 `/joint_cmd` |
| 右臂 7 | `/kuavo_arm_traj` 右臂部分 | `/joint_cmd` 上肢部分 |
| 右手 6 | `/control_robot_hand_position` 右手部分 | 必要时检查 `/joint_cmd` |
| 头部 2 | 补零 `[0.0, 0.0]` | 暂不使用 `/robot_head_motion_data` |

### 3.6 触觉策略

当前方案：

- topic：`/dexhand/touch_state`
- 只提取每个手指的 `normal_force1/2/3`
- 舍弃切向力、切向方向、接近觉和状态字段
- 单手维度：5 指 × 3 点 = 15
- 双手维度：30
- 数值换算：原始值为 `100 * N`，转换时除以 `100.0` 得到牛顿

输出顺序：

```text
左手 finger0 normal_force1/2/3, ..., finger4 normal_force1/2/3,
右手 finger0 normal_force1/2/3, ..., finger4 normal_force1/2/3
```

缺失策略：

- `/dexhand/touch_state` 缺失时直接报错，不静默补零。
- 原因：当前目标是触觉版 DECO，触觉缺失会改变训练任务本质。

### 3.7 时间对齐策略

当前方案：

- 使用主视觉时间范围作为基础，但目标时间戳由 `train_hz: 30` 显式生成。
- 对每个 30Hz 目标时间戳，在 RGB、depth、state、action、tactile 中选择绝对时间差最小的一帧。
- 使用所有必需模态共同覆盖的时间区间，避免某个模态首尾缺帧导致训练样本无效。

后续备选：

- 若 30Hz action 抖动明显，可评估 arm/head state/action 的线性插值。
- 触觉仍建议 nearest-neighbor 或窗口统计，因为触觉接触峰值不应被简单线性插值抹平。

### 3.8 缺失 topic 策略

必须存在，否则报错：

- 头部 RGB topic，例如 `/cam_h/color/image_raw/compressed`
- 头部 depth topic，例如 `/cam_h/depth/image_raw/compressed`
- `/sensors_data_raw`
- `/dexhand/state`
- `/dexhand/touch_state`
- `/kuavo_arm_traj` 或 `/joint_cmd` 至少一个可用于 arm action
- `/control_robot_hand_position`

允许补零：

- `action[26:28]` 当前始终补零。

不允许未 review 时静默补零：

- `observation.state[26:28]`
- `observation.tactile`
- RGB/depth 视觉输入

### 3.9 公共 reader 策略

当前方案：

- 阶段一暂不修改 `kuavo_data/common/kuavo_dataset.py`。
- 新建 `kuavo_data/CvtRosbag2Lerobot_DECO.py`，内部实现 DECO 专用 topic map、字段解析、时间轴生成和 mapper。

原因：

- 避免影响 ACT/DP 既有数据转换链路。
- DECO 需要固定 28 维 state/action、30 维 tactile、30Hz RGB-D 对齐，与当前配置化拼接逻辑差异较大。

---

## 4. 阶段三/四：模型与 Wrapper 技术决策

### 4.1 视觉前端替换策略

DECO 原生视觉入口：

```text
img1, img2
  -> shared ResNet34
  -> img_head Conv2d
  -> image tokens
  -> DECO MMAttention
```

Kuavo-DECO 新视觉入口：

```text
RGB image, depth image
  -> RGB ResNet34 + Depth ResNet34
  -> feature map projection
  -> ACT-style RGB-depth cross attention fusion
  -> fused visual tokens
  -> DECO MMAttention / action-token Flow Matching 主干
```

决策：

- 默认使用 `resnet34`，允许配置切换 `resnet18`。
- 保留空间 token，不优先使用 SpatialSoftmax 压成全局向量。
- 新视觉前端应尽量写在 Kuavo wrapper / 适配层中，减少对 DECO 原包的侵入。

### 4.2 ResNet34 默认值的理由

Pros：

- 更接近 DECO 原始视觉容量。
- 对复杂双手灵巧操作、遮挡和细粒度接触场景可能更强。
- 与 DECO 原始 ResNet34 设计保持心理模型一致。

Contra：

- 比 ResNet18 慢，显存更高。
- 部署延迟风险更高。
- 若数据量不足，过拟合风险高于 ResNet18。

最终决策：

- `vision_backbone: resnet34` 和 `depth_backbone: resnet34` 作为默认配置。
- `resnet18` 作为低延迟、低显存备选配置。

### 4.3 DECO 主干保留策略

必须保留：

- `action_encoder`
- `action_embedd`
- action token 序列建模
- Flow Matching 加噪与去噪逻辑
- 训练目标 `F.mse_loss(out, noise - action)`
- 推理阶段 denoising loop

必须修正的误解：

- `chunk_size` 是 action chunk 长度，不是视觉帧数。
- `inf_step` 是 Flow Matching denoising step，不是 10Hz 控制频率。
- “10Hz 推理”在当前项目语境中指部署控制频率，应由 wrapper/deploy action queue 控制。

### 4.4 30Hz 数据与 10Hz 控制解耦

当前冻结方案：

- 洗数据阶段：`dataset_hz = train_hz = 30`
- 部署阶段：`control_hz = 10`
- wrapper/deploy 使用 `action_stride = dataset_hz // control_hz = 3`

含义：

- 模型训练时看到 30Hz 时间语义的动作序列。
- 部署时每 3 个 30Hz 动作取 1 个进入 10Hz 控制队列。
- 若后续 `dataset_hz` 或 `control_hz` 不是整数倍关系，必须显式实现基于时间戳的动作重采样，而不是硬编码 stride。

### 4.5 触觉模型手术

当前冻结方案：

- 原始 Inspire Hand 1062 维触觉输入不适用于 Kuavo。
- Kuavo 输入为左右手各 15 维 normal force。
- 触觉分支需要将输入维度改为 30，并取消 1062 维区域均值逻辑。
- 触觉预训练权重不可直接严格复用。
- Kuavo 触觉 token 仍应进入 DECO 的 tactile cross-attention，而不是绕过 DECO 主干另接动作头。

### 4.6 Tactile Plugin / LoRA-style Adapter 机制

源码事实：

- DECO 中没有依赖外部 PEFT LoRA；源码中的触觉低秩微调模块叫 `PI_Adapter`。
- `PI_Adapter` 结构是 `Linear(dim -> rank)` 加 `Linear(rank -> out_dim)`，默认 `rank=32`。
- 当 `use_tactile=True` 且 `plugin=True` 时，`MMAttention` 会为 image stream 和 action stream 分别增加低秩 residual adapter：
  - `img_qkv_pi`
  - `img_proj_pi`
  - `img_mlp_pi`
  - `act_qkv_pi`
  - `act_proj_pi`
  - `act_mlp_pi`
- forward 中 adapter 输出以 residual delta 形式加到原始 QKV、attention projection 和 MLP 输出上。

冻结逻辑：

- 当 `pretrain_model_path`、`use_tactile=True`、`plugin=True` 且未指定 `adapter_model_path` 时，DECO 会加载预训练主干权重。
- checkpoint 中存在且 shape 匹配的参数会被加载。
- checkpoint 中已有参数默认被冻结，即 `requires_grad=False`。
- 新出现的参数保持可训练，包括 tactile encoder、tactile cross-attention、PI_Adapter，以及 Kuavo RGB-D 改造后必要的新 bridge 参数。
- 优化器只接收 `requires_grad=True` 的参数，因此冻结策略会真正影响训练。

Kuavo 配置命名：

- 用户配置层使用 `use_tactile_lora` 表示是否启用该触觉低秩 adapter。
- wrapper 内部将 `use_tactile_lora` 映射到 DECO 原生 `plugin`。
- 用户配置层使用 `tactile_lora_rank` 表示低秩 rank，内部映射到 `plugin_rank`。
- 默认值：`use_tactile_lora: true`、`tactile_lora_rank: 32`、`freeze_pretrained_main: true`。

### 4.7 分阶段训练与验证策略

由于 Kuavo-DECO 同时替换了视觉前端并改造了触觉输入，直接开启触觉 adapter 会让视觉问题和触觉问题混在一起。当前冻结分阶段策略如下：

1. **视觉-only 阶段**
   - 配置：`use_tactile: false`、`use_tactile_lora: false`。
   - 目标：验证 RGB-D 前端、30Hz 数据、28 维 action、Flow Matching 主干能独立闭环。
   - 允许训练范围：RGB-D bridge、视觉投影层以及按配置允许的 ResNet backbone。

2. **触觉 adapter 阶段**
   - 配置：`use_tactile: true`、`use_tactile_lora: true`。
   - 加载视觉-only 或 RGB-D 主干 checkpoint。
   - 冻结预训练主干，训练 tactile encoder、tactile cross-attention、PI_Adapter 和必要的 Kuavo 触觉桥接参数。

3. **部署验证阶段**
   - 先关闭触觉进入仿真，确认 RGB-D 主链路稳定。
   - 再开启触觉 adapter 做仿真、离线 replay 或低风险真机验证。
   - 最后经过 dry-run、低速限幅、完整闭环三步上实机。

仿真阶段的好结果标准：

- 无 NaN/Inf。
- 无关节位置或速度越界。
- 10Hz action queue 节奏稳定。
- 左右臂、左右手映射正确。
- 头部 action 补零不被部署端误解释为真实头部控制。
- RGB-depth 对齐，depth 不被 RGB photometric augmentation 破坏。
- 任务行为接近同数据上的 ACT/DP 基线，至少不出现系统性方向反转、剧烈抖动或明显时序滞后。

---

## 5. 待同步文档与实现清单

- [ ] 更新 `README_DECO.md`，说明 Kuavo-DECO 当前采用 RGB-D 前端，而不是原生双 RGB。
- [ ] 新建 `configs/data/KuavoRosbag2Lerobot_deco.yaml`，默认 30Hz、use_depth true。
- [ ] 新建 `configs/policy/deco_config.yaml`，默认 ResNet34 RGB-D、control_hz 10、action_stride 3，并显式包含 `use_tactile_lora`、`tactile_lora_rank`、`freeze_pretrained_main`、`pretrain_model_path`、`adapter_model_path`。
- [ ] 新建 `kuavo_data/CvtRosbag2Lerobot_DECO.py`。
- [ ] 新建 `kuavo_data/validate_deco_lerobot_dataset.py`，检查单 rosbag 转换结果的字段、维度、30Hz 时间轴、RGB-depth 对齐、28 维 action 映射与 30 维 tactile 量纲。
- [ ] 新建 `kuavo_train/wrapper/policy/deco/DECOConfigWrapper.py`。
- [ ] 新建 `kuavo_train/wrapper/policy/deco/DECOPolicyWrapper.py`。
- [ ] 评估是否需要将 DECO 原包迁移到 `third_party/deco/` 后再做最小补丁。
