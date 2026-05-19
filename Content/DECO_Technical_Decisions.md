# DECO 技术决策记录

> 最后更新：2026-05-19
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
Kuavo RGB + depth + state + action + optional tactile rosbag
  -> 30Hz LeRobot RGB-D 数据集（end_effector_profile 决定 state/action 维度与是否写入 tactile）
  -> DECO 专用 preprocessor：RGB/depth 同步 256x256 letterbox，RGB 再进入随机增强池
  -> LeRobot Normalizer：RGB/depth/state/action 按配置归一化，tactile 作为 TACTILE 保持 IDENTITY
  -> 三条平行主枝：
     1) state branch：profile action_dim 对应的 observation.state -> obs_encoder -> time embedding condition
     2) RGB-D visual branch：RGB ResNet34 + 1-channel Depth ResNet34 -> cross attention -> fused RGB/depth tokens
     3) tactile branch：仅 qiangnao_tactile 可启用；30D observation.tactile -> left/right tactile max normalize -> tactile encoder / PI_Adapter
  -> DECO action-token Flow Matching transformer
  -> profile action_dim 对应的 action chunk（qiangnao 28D / gripper 18D）
  -> 部署阶段按 10Hz 控制频率消费动作队列
```

核心原则：

- **视觉输入方式向 Kuavo 工具链靠齐**：使用 RGB + depth，而不是 DECO 原生双 RGB。
- **数据清洗实现向 Kuavo 现有链路靠齐**：除 DECO 专属 profile schema、30Hz 时间轴、可选 30 维触觉和 depth 单通道语义外，topic map、message decoder、state/action 来源优先复用 ACT/DP 当前清洗脚本。
- **末端执行器 schema 向 ACT/DP 入口体验靠齐**：用户在数据清洗 config 中通过 `dataset.eef_type` 选择 `qiangnao`、`leju_claw` 或 `rq2f85`；`deco.end_effector_profile: auto` 会自动推导最终 dataset schema，也允许显式填写后做一致性校验。`qiangnao_tactile` 对应 28D 灵巧手 + 可选触觉，`gripper_no_tactile` 对应 18D 二夹爪且无触觉。
- **模型主干向 DECO 靠齐**：保留 DECO action token、Flow Matching 训练目标和去噪推理。
- **backbone 默认向 DECO 容量靠齐**：默认 `resnet34`，允许配置切换 `resnet18`。
- **RGB-D 编码分流**：RGB 与 depth 使用独立 ResNet backbone，depth 为 1-channel ResNet；二者不共享同一个 ResNet，只在 ResNet 后的 token 层做 cross-modal fusion。
- **保留 DECO 两路视觉 token 假设**：阶段三第一版把原生双路视觉语义替换为 `fused_rgb/fused_depth`，仍以两路 visual tokens 接入 DECO `MMAttention`，暂不采用单路 visual token 重构。
- **视觉预处理分层**：`Resize/Letterbox` 是 RGB 与 depth 共享的确定性空间预处理；`RGB_Augmenter` 是训练期随机增强池。二者不得混淆。默认空间输入采用 DECO 原生 `256x256 letterbox`，RGB padding 为灰色 `fill=128`，depth padding 单独配置。
- **RGB 增强吸收 DECO blur 经验**：随机增强池以 Kuavo ACT 权重策略为主，并新增 GaussianBlur；默认保留一部分原图，另一部分从增强池中抽样。
- **state 接入方式向 DECO 靠齐**：`observation.state` 由 LeRobot preprocessor 基于 dataset stats 归一化，但进入模型后仍走 DECO 的 `obs_encoder + time embedding` 条件路线，不改成 ACT 的 state token / VAE encoder 路线。
- **tactile 接入方式向 DECO 靠齐**：洗数据阶段 `/100` 只做单位换算，把 Kuavo normal force 转为牛顿；LeRobot feature mapping 中将 `observation.tactile` 识别为独立 `TACTILE` 并使用 `IDENTITY`；进入模型前按左右手 tactile max 做 DECO-style 归一化并默认 clamp 到 `[0, 1]`，避免被 LeRobot 当作普通 STATE 执行 `MEAN_STD`。
- **gripper_no_tactile 禁止触觉分支**：`leju_claw` 与 `rq2f85` 在清洗入口 topic 和数值尺度不同，但进入 DECO 后共享 `gripper_no_tactile` 的 18D state/action schema；该 profile 不写入、不要求、不使用 `observation.tactile`，也不得进入 tactile LoRA / PI_Adapter 二阶段训练。
- **loss 向 DECO 原生靠齐**：训练目标保持 `F.mse_loss(out, noise - action)`，第一版不额外乘 `action_is_pad` mask。
- **触觉微调向 DECO 靠齐**：保留源码中的 `plugin=True` / `PI_Adapter` 低秩 adapter 范式，默认冻结预训练主干，只微调触觉 adapter 和必要的 Kuavo 新增桥接模块。
- **wrapper 边界清晰化**：`lerobot_patches/` 只做 LeRobot 全局 feature/type 兼容补丁，例如新增 `FeatureType.TACTILE`；DECO 专用 RGB-D `Resize/Letterbox`、RGB augmentation 接入顺序、tactile max normalization 和两阶段训练逻辑放在 `kuavo_train/wrapper/policy/deco/`。
- **两阶段训练是两次独立启动**：第一阶段 `visual_main` 完整训练 RGB-D + state 主干并保存 Kuavo run 目录；第二阶段 `tactile_adapter` 再加载第一阶段选定 epoch 的 policy 权重，冻结主干，只训练 tactile encoder、tactile cross-attention、PI_Adapter 等新参数。若不使用触觉，则第一阶段 run 目录加选定 epoch 权重就是最终部署资产；部署阶段必须同时允许灵巧手无触觉、灵巧手带触觉和二夹爪无触觉三种推理模式。
- **权重格式语义分层**：`.safetensors` policy 权重目录是 Kuavo-DECO 正式训练、续训和加载权重的入口；`.pth` 只作为 DECO 原生 checkpoint 或历史 PyTorch 权重导入兼容入口。部署资产则沿用 Kuavo 原逻辑，以 `outputs/train/<task>/<method>/<timestamp>/` run 根目录为单位，`epoch<epoch>/` 只是其中被选择的权重子目录；最终保存的 policy 会清空外部初始化路径并关闭外部初始化读取，避免迁移后依赖原始初始化文件。
- **频率处理分层**：数据转换阶段负责 30Hz 训练数据；部署 wrapper 负责 10Hz 控制输出。

### 2.2 被替代的旧方案

以下旧方案不再作为主路线：

- `use_depth: false`
- 只保存 `/cam_h` RGB，不保存 depth
- 将 `/cam_h` 单目图复制成 DECO 的双路视觉输入
- 将 `/cam_h` 左右裁切成伪双目
- 默认 10Hz LeRobot 数据转换频率
- 在阶段一把 depth 剥离以严格保持 DECO 原生假设

保留说明：

- 这些旧方案可以作为 ablation 或紧急 fallback，但不得作为主实现路线写入新代码。

---

## 3. 阶段一：RGB-D 数据引擎技术决策

### 3.1 目标输出字段

新建 `kuavo_data/CvtRosbag2Lerobot_DECO.py`，将 Kuavo rosbag 转换为 DECO wrapper 可以直接消费的 LeRobot 数据集。

目标输出字段按 `end_effector_profile` 决定：

| Profile | LeRobot 字段 | 目标形状 | 语义 |
| ------- | ------------ | -------- | ---- |
| `qiangnao_tactile` | `observation.state` | `(28,)` | 左臂 7 + 左手 6 + 右臂 7 + 右手 6 + 头部 2 |
| `qiangnao_tactile` | `action` | `(28,)` | 左臂 7 + 左手 6 + 右臂 7 + 右手 6 + 头部 2 |
| `qiangnao_tactile` | `observation.tactile` | `(30,)` | 左手 15 + 右手 15 法向触觉力，单位为牛顿；仅在配置要求触觉时写入/校验 |
| `gripper_no_tactile` | `observation.state` | `(18,)` | 左臂 7 + 左夹爪 1 + 右臂 7 + 右夹爪 1 + 头部 2 |
| `gripper_no_tactile` | `action` | `(18,)` | 左臂 7 + 左夹爪 1 + 右臂 7 + 右夹爪 1 + 头部 2 |
| all | `observation.images.head_cam_h` | 自动推断或配置指定 | 头部 RGB 图像 |
| all | `observation.depth_h` 或等价 depth key | 自动推断或配置指定 | 与头部 RGB 对齐的深度图 |

兼容说明：

- `dataset.eef_type=qiangnao` 在 `end_effector_profile=auto` 时推导为 `qiangnao_tactile`；若显式填写，也必须与 `qiangnao_tactile` 一致。
- `dataset.eef_type=leju_claw` 或 `rq2f85` 在 `end_effector_profile=auto` 时推导为 `gripper_no_tactile`；若显式填写，也必须与 `gripper_no_tactile` 一致。
- 不同 profile 不应混在同一个 LeRobot dataset 或同一个训练 run 中，因为 LeRobot feature schema、Normalizer stats、DECO 输入/输出线性层维度都必须固定。

历史目标输出字段（当前 `qiangnao_tactile` 第一版实现）：

| LeRobot 字段                           | 目标形状           | 语义                                       |
| -------------------------------------- | ------------------ | ------------------------------------------ |
| `observation.state`                    | `(28,)`            | 左臂 7 + 左手 6 + 右臂 7 + 右手 6 + 头部 2 |
| `action`                               | `(28,)`            | 左臂 7 + 左手 6 + 右臂 7 + 右手 6 + 头部 2 |
| `observation.tactile`                  | `(30,)`            | 左手 15 + 右手 15 法向触觉力，单位为牛顿   |
| `observation.images.head_cam_h`        | 自动推断或配置指定 | 头部 RGB 图像                              |
| `observation.depth_h` 或等价 depth key | 自动推断或配置指定 | 与头部 RGB 对齐的深度图                    |

### 3.2 数据配置文件

当前方案：

- 新建配置文件：`configs/data/KuavoRosbag2Lerobot_deco.yaml`
- 默认训练数据频率：`train_hz: 30`
- 深度图：`use_depth: true`
- 默认 RGB topic：`/cam_h/color/image_raw/compressed`
- 默认 depth topic：`/cam_h/depth/image_raw/compressed`
- depth topic 兼容候选：`/cam_h/depth/image_raw/compressed` 使用 `compressed_image` 直解，`/cam_h/depth/image_raw/compressedDepth` 使用 `compressedDepth_png` 跳过 PNG 前缀头后解码。
- 默认 depth decoder：`compressed_image`，但转换脚本会在单个 rosbag 打开后按实际存在的 depth topic 自动选择绑定 decoder。
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
- 用户截图中实采与官方模拟数据的 RGB topic 保持一致：头部 RGB 都可使用 `/cam_h/color/image_raw/compressed`，左右相机 RGB 也沿用 `/cam_l|r/color/image_raw/compressed`。
- depth topic 存在两种等价命名：官方模拟数据可使用 `/cam_h/depth/image_raw/compressed`，消息语义为普通 `sensor_msgs/CompressedImage`；实采数据可能使用 `/cam_h/depth/image_raw/compressedDepth`，需要按 ROS compressedDepth 格式跳过 PNG magic header 前的配置头。
- 用户补充的 `/camera/depth/image_rect_raw`、`encoding=16UC1` 暂作为 raw depth 候选源；该信息可能与当前 rosbag 不一致，不能替代默认冻结 topic，必须通过 Inspector/validator 复核后才能启用。

当前冻结策略：

- 保存 `observation.images.head_cam_h` 作为 RGB 输入。
- 保存与头部 RGB 对齐的 depth feature；转换脚本按候选表兼容 `/cam_h/depth/image_raw/compressed` 与 `/cam_h/depth/image_raw/compressedDepth` 两种 topic。
- DECO converter 支持四类 depth decoder 入口：`compressed_image`、`compressedDepth_png`、`raw_16uc1` 与仅供候选表省略 encoding 时使用的 `auto`。
- DECO wrapper 通过 Kuavo/ACT 风格 RGB-D 视觉前端处理这两路视觉输入。
- 不再把单目 RGB 复制成双路视觉输入。
- 不再把 `/cam_h` 按宽度中线切成伪双目。

### 3.4 RGB 与 depth 的增强策略

当前冻结策略把视觉处理拆成两层：

1. **确定性空间预处理**：
   - `Resize/Letterbox` 属于输入尺寸规范化，不属于随机增强池。
   - 训练与推理必须一致执行。
   - RGB 与 depth 必须共享同一组空间参数，保证像素级对齐。
   - 默认采用 DECO 原生 `256x256 letterbox`，即按最长边缩放后补齐到正方形。
   - RGB resize 使用双线性插值；depth resize 使用 nearest 插值。
   - RGB padding 默认继承 DECO 的灰色 `fill=128`；depth padding 必须单独配置，默认建议为 `0` 或明确的 invalid depth 值。

2. **训练期 RGB 随机增强池**：
   - 随机增强只作用于 RGB。
   - 默认采样逻辑参考 Kuavo ACT：`max_num_transforms: 1`、`random_order: true`、Identity/Notransform 权重较高，其他增强权重相同。
   - 这样一部分样本保持原图分布，另一部分从增强池里抽取一种扰动。

RGB 随机增强池复用 Kuavo 现有 `RGB_Augmenter`，并新增 DECO blur 候选：

- Identity
- ColorJitter：brightness / contrast / saturation / hue
- SharpnessJitter
- RandomMask
- RandomBorderCutout
- GaussianNoise
- GammaCorrection
- GaussianBlur

权重冻结建议：

```yaml
notransform:
  weight: 3.0
其他 RGB 增强:
  weight: 1.0
max_num_transforms: 1
```

DECO 原生 GaussianBlur 的源码行为是：

```text
RandomApply(GaussianBlur(kernel_size=random.choice([3, 5, 7]),
                         sigma=random.uniform(0.1, 2.0)),
            p=0.5)
```

Kuavo-DECO 不直接复刻 `p=0.5` 的触发概率，而是把 blur 放进 Kuavo 增强池。当前第一版配置使用固定 `kernel_size: 5`，并保留 `sigma: [0.1, 2.0]` 参数范围，由 Kuavo `RGB_Augmenter` 的权重控制采样概率。

Depth 策略：

- 与 RGB 共享 `Resize/Letterbox` 的空间变换，保证 RGB-depth 对齐。
- 不做 ColorJitter、Hue、Saturation、Brightness、Gamma 等 RGB photometric augmentation。
- 不做 GaussianBlur、GaussianNoise、RandomMask、RandomBorderCutout 等会改变 depth 边界或缺失模式的 RGB 增强。
- depth 的归一化遵守 Kuavo 现有 depth 配置，不把 depth 当作普通 RGB 图像处理。
- 当前脚本默认优先 `compressed_image`：直接对 `sensor_msgs/CompressedImage.data` 调用 `cv2.imdecode(..., IMREAD_UNCHANGED)`。
- 若当前 rosbag 只有 `/cam_h/depth/image_raw/compressedDepth`，脚本会切换到 `compressedDepth_png`：先定位 PNG magic header，再对 PNG payload 调用 `cv2.imdecode(..., IMREAD_UNCHANGED)`。
- raw `16UC1` 仍只作为配置化 fallback，不作为当前默认路线。
- 若使用 raw `16UC1` fallback，则必须按 `height`、`width`、`step`、`is_bigendian`、`data` 正确 reshape，并在 validation report 中记录启用原因。
- 旧转换脚本存在把 depth clip/归一化为 `uint8` 后 repeat 为 3 通道的做法；Kuavo-DECO 主路线仍应保留单通道 depth 语义，除非 wrapper/config 明确声明使用 3-channel depth 兼容模式。

#### 3.4.1 depth 存储的一版策略与备选策略

**第一版冻结策略：沿用现有转换结果**

```text
uint16/mm depth
  -> 按 dataset.depth_range clip
  -> per-frame normalize 到 0-255
  -> uint8
  -> repeat 成 3-channel image/video
  -> wrapper 中取单通道送入 1-channel depth backbone
```

Pros：

- 与当前已实现的 `CvtRosbag2Lerobot_DECO.py` 和 LeRobot image/video writer 兼容。
- 不阻塞阶段三 RGB-D 主链路实现。
- 与现有 ACT/DP 兼容存储习惯接近，短期工程风险最低。

Contra：

- per-frame normalize 会丢失跨帧绝对毫米尺度。
- 模型看到的是归一化深度图，不是严格 metric depth。
- 后续若任务强依赖绝对距离，需要回到数据转换与 validator 层升级 depth 存储。

**备选策略 A：uint16 / 单通道 metric depth feature**

```text
uint16/mm depth
  -> 保留单通道 depth 数组或视频
  -> dataset stats / depth normalizer 统一归一化
  -> wrapper 直接作为 [B, 1, H, W] 输入
```

Pros：

- 保留毫米尺度与跨帧绝对距离关系。
- 更符合机器人 RGB-D 感知中的 metric depth 语义。

Contra：

- 需要确认当前 LeRobot 版本对 `uint16` depth feature、video writer、stats 和 preprocessor 的完整支持。
- 需要改数据转换、validator、policy config 与 wrapper，工程面更大。

**备选策略 B：保留 raw depth 并在 wrapper 中转换**

```text
raw/compressed uint16 depth
  -> 数据集保存原始或近原始 depth
  -> wrapper 中按统一 depth_range / dataset stats 转 float depth
```

Pros：

- 转换阶段尽量少损失信息。
- 后续可以调整 depth normalization 而不必重新解码 rosbag。

Contra：

- 对数据集格式和读取链路要求更高。
- 当前阶段会扩大实现范围，不适合作为第一版。

最终决策：

- 阶段三第一版沿用现有 3-channel uint8 兼容存储策略。
- 在 `PLANS.md` 与 `configs/policy/deco_config.yaml` 中显式声明该策略不是严格 metric depth。
- 第二版或后续 ablation 再评估单通道 `uint16` / metric depth 存储。

### 3.5 28 维 state/action 映射

`qiangnao_tactile` profile 使用 28 维顺序：

```text
左臂 7 + 左手 6 + 右臂 7 + 右手 6 + 头部 2
```

已确认方案：

- `qiangnao_tactile` 输出必须固定为 28 维。
- `gripper_no_tactile` 后续输出必须固定为 18 维，顺序为 `左臂 7 + 左夹爪 1 + 右臂 7 + 右夹爪 1 + 头部 2`。
- 当前 rosbag 的 `/sensors_data_raw.joint_data.joint_q` 长度为 28，头部索引使用 V4x/V49 方案 `joint_q[26:28]`。
- 当前样本头部固定姿态均值约为 `[-0.001657, 0.433904]` rad，即 `[-0.09494°, 24.86089°]`。
- `/robot_head_motion_data` 显示 `[0.0, 25.0]`，可交叉确认 pitch 约 25°。
- `observation.state[26:28]` 使用每个 episode 内 `joint_q[26:28]` 的实测固定均值并广播到所有帧。
- `action[26:28]` 当前始终补 `[0.0, 0.0]`，表示阶段一 DECO 策略暂不输出头部控制。

推荐 state 来源：

| 维度   | 来源 topic          | 字段/切片                                   |
| ------ | ------------------- | ------------------------------------------- |
| 左臂 7 | `/sensors_data_raw` | `joint_data.joint_q[12:19]`                 |
| 左手 6 | `/dexhand/state`    | `position[:6]`                              |
| 右臂 7 | `/sensors_data_raw` | `joint_data.joint_q[19:26]`                 |
| 右手 6 | `/dexhand/state`    | `position[6:12]`                            |
| 头部 2 | `/sensors_data_raw` | `joint_data.joint_q[26:28]` 的 episode 均值 |

推荐 action 来源：

| 维度   | 首选来源                                               | fallback                           |
| ------ | ------------------------------------------------------ | ---------------------------------- |
| 左臂 7 | `/kuavo_arm_traj_synced` 或 `/kuavo_arm_traj` 左臂部分 | `/joint_cmd` 上肢部分              |
| 左手 6 | `/control_robot_hand_position` 左手部分                | 必要时检查 `/joint_cmd`            |
| 右臂 7 | `/kuavo_arm_traj_synced` 或 `/kuavo_arm_traj` 右臂部分 | `/joint_cmd` 上肢部分              |
| 右手 6 | `/control_robot_hand_position` 右手部分                | 必要时检查 `/joint_cmd`            |
| 头部 2 | 补零 `[0.0, 0.0]`                                      | 暂不使用 `/robot_head_motion_data` |

现有清洗脚本的 action 结论：

- `action` 原始读取 `/joint_cmd`。
- arm action 随后被 `/kuavo_arm_traj` 覆盖；如果存在 `/kuavo_arm_traj_synced`，则优先使用 synced 版本。
- hand action 使用 `/control_robot_hand_position`。
- `qiangnao_tactile` 固定双手各 6 DoF，因此不得沿用 ACT/DP 默认 `dex_dof_needed: 1` 的手指压缩策略。
- `gripper_no_tactile` 不使用 dexhand 6 DoF；`leju_claw` 读取 `/leju_claw_state` 与 `/leju_claw_command`，`rq2f85` 读取 `/gripper/state` 与 `/gripper/command`。二者在清洗入口 topic 与归一化尺度不同，但最终都映射到同一个 18D 二夹爪 schema。

### 3.6 触觉策略

当前方案：

- topic：`/dexhand/touch_state`，仅 `qiangnao_tactile` profile 使用。
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

- `qiangnao_tactile` 且配置要求触觉时，`/dexhand/touch_state` 缺失应直接报错，不静默补零。
- `gripper_no_tactile` 不写入 tactile，也不把缺失触觉视为错误。
- 原因：触觉 LoRA / PI_Adapter 的训练依赖真实 tactile 语义；二夹爪任务没有该模态，不能用全零 tactile 假装无接触。

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
- 头部 depth topic，候选为 `/cam_h/depth/image_raw/compressed` 或 `/cam_h/depth/image_raw/compressedDepth`
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
- 新建 `kuavo_data/CvtRosbag2Lerobot_DECO.py`，优先复用现有 `KuavoRosbagReader` 的 topic map、message processor 和 nearest-neighbor 对齐思路。
- 仅在 DECO profile 化 state/action、可选 30 维 tactile、30Hz 目标时间轴、depth 单通道语义和 validation report 等位置增加专用逻辑。

原因：

- 避免影响 ACT/DP 既有数据转换链路。
- DECO 需要 profile 化 state/action、可选 30 维 tactile、30Hz RGB-D 对齐，与当前 ACT/DP 配置化拼接逻辑相近但输出 schema 和触觉训练约束不同。
- depth topic/decoder 当前已有稳定实现，优先继承该实现比引入未经复核的新 raw topic 风险更低。

### 3.10 阶段一实现状态

截至 2026-05-18，阶段一数据转换入口已完成 profile 化静态实现：

- 新增 `configs/data/KuavoRosbag2Lerobot_deco.yaml`，记录 30Hz、RGB-D、`end_effector_profile: auto`、可选触觉、profile 化 state/action、depth fallback 和覆盖保护配置。
- `qiangnao_tactile` 保持 28D + 可选 tactile；`gripper_no_tactile` 新增 18D + no tactile，并复用 `dataset.eef_type=leju_claw/rq2f85` 的 ACT/DP 入口体验。
- 新增 `kuavo_data/CvtRosbag2Lerobot_DECO.py`，保持与原 ACT/DP 脚本并行，不修改公共 reader。
- 转换脚本保持 RGB 默认 `/cam_h/color/image_raw/compressed`，并对 depth 增加候选 topic 自动解析：`/cam_h/depth/image_raw/compressed` 绑定 `compressed_image`，`/cam_h/depth/image_raw/compressedDepth` 绑定 `compressedDepth_png`。
- 转换脚本同时支持 `raw_16uc1` fallback 与 `auto` depth decoder；raw 路径仍必须由后续 Inspector/validator 复核后启用。
- 第一版 depth 在 LeRobot 磁盘 schema 中按 3-channel depth image 保存，以兼容 image/video writer；后续 DECO wrapper 必须把它按 depth 语义还原为 1-channel depth backbone 输入。
- 本次实现只做静态审查，未在当前 Codex 机器执行 Python、rosbag 转换或训练。

---

## 4. 阶段三/四：模型与 Wrapper 技术决策

### 4.1 视觉前端替换策略

DECO 原生视觉入口：

```text
two visual images
  -> shared ResNet34
  -> img_head Conv2d
  -> image tokens
  -> DECO MMAttention
```

Kuavo-DECO 新视觉入口：

```text
RGB image, depth image
  -> RGB ResNet34 + 1-channel Depth ResNet34
  -> feature map projection
  -> ACT-style RGB-depth cross attention fusion
  -> fused RGB tokens + fused depth tokens
  -> DECO MMAttention / action-token Flow Matching 主干
```

决策：

- 默认使用 `resnet34`，允许配置切换 `resnet18`。
- RGB 与 depth 使用独立 backbone，不共享同一个 ResNet；depth backbone 第一层为 1-channel conv，并参考 ACT 用 RGB conv1 权重的通道均值初始化。
- 阶段三第一版保留 DECO 原生“两路视觉 token”结构，但把原生双路视觉语义替换为 `fused_rgb/fused_depth`。
- `MMAttention` 内基于 `total_img_len / 2` 的视觉 token 分流逻辑可以保留，但代码注释必须说明当前两半不再是两张 RGB 图，而是 RGB stream 与 depth stream。
- 保留空间 token，不优先使用 SpatialSoftmax 压成全局向量。
- 暂不采用单路 `visual_tokens: [B, L, D]` 方案；该方案理论可行，但需要改写 `img_encoding`、`pos_idx_embedd`、RoPE 应用和 `MMAttention` 的两路视觉假设，作为后续 ablation 或二期重构候选。
- 新视觉前端应尽量写在 Kuavo wrapper / 适配层中，减少对 DECO 原包的侵入。

#### 4.1.1 两路 visual tokens 与单路 visual token 的取舍

**方案 A：保留两路 visual tokens（当前冻结方案）**

```text
RGB -> RGB ResNet34 -> RGB tokens
depth -> Depth ResNet34 -> depth tokens
RGB/depth cross attention
-> fused_rgb tokens + fused_depth tokens
-> [B, 2L, dim]
-> DECO MMAttention
```

Pros：

- 最接近 DECO 原生双路视觉结构。
- 对 DECO 主干改动更小，`total_img_len / 2`、RoPE 分半应用等逻辑可以保留。
- RGB 和 depth 的语义在进入主干时仍可区分，便于后续分析和 ablation。

Contra：

- token 数为 `2L`，显存和 attention 计算量高于单路方案。
- 原生“双 RGB 图像”的语义被替换成“RGB/depth 双流”，需要通过注释和文档清晰记录。

**方案 B：融合为单路 visual token（暂不作为第一版）**

```text
RGB tokens + depth tokens
-> cross attention / projection
-> visual tokens [B, L, dim]
-> 改写 DECO MMAttention，使其只处理一路视觉流
```

Pros：

- 架构更简洁，token 数更少。
- 更像“RGB-D 已融合后的统一视觉表示”。

Contra：

- 需要更深地改写 DECO core：`img_encoding`、`pos_idx_embedd`、RoPE 和 `MMAttention` 的两路视觉假设都要变。
- 预训练权重兼容性更弱。
- 当前机器遵守 No-Runtime 约束，无法通过 forward 运行验证第一版复杂重构。

最终决策：

- 阶段三第一版采用方案 A。
- 方案 B 仅作为后续在 RGB-D 主链路稳定后的可选实验。

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

Loss 决策：

- 第一版严格遵循 DECO 原生训练代码：虽然数据集中可能存在 `action_is_pad` 或等价 mask，Flow Matching loss 仍直接使用 `F.mse_loss(out, noise - action)`。
- 暂不额外乘 padding mask，避免阶段三同时改变模型结构和 loss 语义。
- 若后续实测发现 episode 末尾 padding 对训练造成明显偏置，再作为单独 ablation 引入 masked loss，而不是混入第一版主线。

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

### 4.5 obs/state 归一化与模型接入策略

源码差异：

- DECO 原生流程在 `dataset.py` / `inference.py` 中手动归一化 `obs` 与 `action`，统计量来自 `config/deco.yaml` 的 `observation_mean/std` 或 `observation_min/max`。
- DECO 原生模型中，`obs` 不是 transformer token；它先经过 `obs_encoder: 28D -> dim`，再加到 time embedding 上，用于调制 `MMAttention` 中的 AdaLN scale、shift 和 gate。
- Kuavo ACT / LeRobot 流程由 preprocessor 根据 dataset stats 统一归一化 `observation.state`，模型内部把 state 作为 transformer encoder token；训练时 state 还进入 VAE encoder，与 action chunk 一起编码 latent。

Kuavo-DECO 冻结策略：

- `observation.state` 使用 Kuavo/LeRobot preprocessor 与 dataset stats 做一次归一化，默认保持 `STATE: MEAN_STD`。
- DECO wrapper 内不得再次套用 DECO 原生 `config/deco.yaml` 的手动 state 归一化，避免二次归一化导致尺度错误。
- 归一化后的 state 仍按 DECO 路线进入模型：`obs_encoder(action_dim -> dim) -> time embedding condition -> MMAttention`。
- 不把 state 改成 ACT 的 encoder token，也不引入 ACT 的 VAE latent 训练路线。
- `observation.tactile` 必须从 batch 中单独取出，进入 tactile encoder / tactile cross-attention / PI_Adapter；不得与 `observation.state` 拼接成一个更长 state。
- `lerobot_patches/custom_patches.py` 需要把 `observation.tactile` 显式识别为 `FeatureType.TACTILE`，而不是普通 `FeatureType.STATE`。
- DECO config 的 normalization mapping 中，`TACTILE` 默认使用 `IDENTITY`。这样 LeRobot preprocessor 只负责把 tactile 张量移动到正确 device，不对它执行 `MEAN_STD` 或 `MIN_MAX`。
- DECO wrapper 在 `use_tactile: true` 时再做 DECO-style tactile max normalization；在 `use_tactile: false` 时不启用 tactile 分支，未训练的 tactile/PI_Adapter 不参与 forward。
- `gripper_no_tactile` profile 的 `action_dim=18`，不会创建或消费 tactile 分支；其 state/action 在进入主干前同样通过 `Linear(18 -> dim)` 投影成 DECO hidden token，因此主干 attention 层仍处理固定 hidden dim，而不是直接处理原始自由度维度。

直观解释：

```text
LeRobot preprocessor 负责数值尺度：
  raw state -> normalized state

DECO 主干负责条件注入方式：
  normalized state -> obs_encoder(action_dim -> dim) -> 与 time embedding 相加 -> 调制 action-token Flow Matching 主干

触觉不走这条 state 路线：
  tactile(Newton) -> TACTILE: IDENTITY -> DECO-style left/right max normalization -> tactile encoder / PI_Adapter
```

### 4.6 触觉模型手术

当前冻结方案：

- 原始 Inspire Hand 1062 维触觉输入不适用于 Kuavo。
- Kuavo 输入为左右手各 15 维 normal force。
- 洗数据脚本中的 `/100` 操作只表示单位换算：把 Kuavo 原始 normal force 转换为牛顿，不等价于 DECO 原生 tactile normalization。
- 进入 DECO 模型前应沿用 DECO 原生思想：左手 15 维除以 `tactile_left_max`，右手 15 维除以 `tactile_right_max`，并默认 clamp 到 `[0, 1]`；这两个 max 应基于已经转换成牛顿的 Kuavo 数据统计或配置，不能直接复用 DECO Inspire Hand 原始单位下的 `3486/4050`。
- `tactile_left_max` 与 `tactile_right_max` 的含义：
  - `tactile_left_max`：左手 15 维 normal force 牛顿值的归一化上限。
  - `tactile_right_max`：右手 15 维 normal force 牛顿值的归一化上限。
  - 二者用于执行 DECO-style `tac / tactile_max`，不是 LeRobot STATE `MEAN_STD` 统计量。
- 可填写内容：
  - `null`：只允许作为待统计占位，适合先写配置模板或只跑 `use_tactile: false` 的视觉阶段。
  - 正数：正式触觉训练必须填写，可来自训练集左右手 tactile 最大值、稳健分位数上限（例如高分位统计）或人工审定安全上限。
  - 不允许在 `use_tactile: true` 时保留 `null`、`0` 或负数；wrapper/config 应显式报错。
- `clip_tactile_to_unit: true` 作为默认配置，含义是在 `tactile / tactile_max` 后执行 clamp `[0, 1]`。这与 DECO 原生 inference 中 `clamp(0, 1.0)` 的处理一致，可抑制真实触觉传感器尖峰对 adapter 的不受控影响。
- 触觉分支需要将输入维度改为 30，并取消 1062 维区域均值逻辑。
- Kuavo 30 维 tactile 可被视为已经抽取好的 30 个触觉区域值：左手 15 + 右手 15。
- `tactile_encoder` 改为 `30D -> 34D`，再与左右手 15D 归一化触觉拼接，形成 `15 + 15 + 34 = 64` 个 tactile condition positions。
- 原生 `gated = nn.Linear(68, 68)` 需调整为 `nn.Linear(64, 64)`；`pos_tac_embedd` 继续把每个 tactile condition position 映射到 DECO hidden dim。
- `observation.tactile` 不应跟随 LeRobot STATE `MEAN_STD`，否则会偏离 DECO 原生“先按触觉最大值缩放，再进 tactile encoder”的处理方式。
- 触觉预训练权重不可直接严格复用。
- Kuavo 触觉 token 仍应进入 DECO 的 tactile cross-attention，而不是绕过 DECO 主干另接动作头。
- `gripper_no_tactile` 不进入本节触觉手术路径；如果数据源是 `leju_claw` 或 `rq2f85`，配置层必须保持 `use_tactile: false` 与 `use_tactile_lora: false`。

直观流程：

```text
rosbag normal_force
  -> 转换脚本 /100，得到牛顿值
  -> LeRobot preprocessor 按 TACTILE: IDENTITY 保持原尺度
  -> DECO wrapper 按左右手 tactile max 归一化并默认 clamp 到 [0, 1]
  -> tac1/tac2: [B, 15] + [B, 15]
  -> tactile_encoder(30D -> 34D)
  -> concat: 15 + 15 + 34 = 64
  -> gated + pos_tac_embedd
  -> tactile cross attention / PI_Adapter
```

### 4.7 Tactile Plugin / LoRA-style Adapter 机制

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

- 当 `base_policy_path` 或 `deco_init_pth_path`、`use_tactile=True`、`use_tactile_lora=True` 且未指定 `adapter_model_path` 时，wrapper 会加载第一阶段 `.safetensors` policy 或兼容 `.pth` 初始化，并冻结主干，只保留 tactile/PI_Adapter 相关参数可训练。
- checkpoint 中存在且 shape 匹配的参数会被加载。
- checkpoint 中已有参数默认被冻结，即 `requires_grad=False`。
- 新出现的参数保持可训练，包括 tactile encoder、tactile cross-attention、PI_Adapter，以及 Kuavo RGB-D 改造后必要的新 bridge 参数。
- 优化器只接收 `requires_grad=True` 的参数，因此冻结策略会真正影响训练。

Kuavo 配置命名：

- 用户配置层使用 `use_tactile_lora` 表示是否启用该触觉低秩 adapter。
- wrapper 内部将 `use_tactile_lora` 映射到 DECO 原生 `plugin`。
- 用户配置层使用 `tactile_lora_rank` 表示低秩 rank，内部映射到 `plugin_rank`。
- 第一阶段默认值：`use_tactile: false`、`use_tactile_lora: false`、`tactile_lora_rank: 32`、`freeze_pretrained_main: true`。
- 第二阶段触觉 adapter override：`use_tactile: true`、`use_tactile_lora: true`、`base_policy_path` 指向第一阶段选定 epoch 的 `.safetensors` policy 权重目录，`freeze_pretrained_main: true`。
- 如果用户完全不使用触觉，`use_tactile_lora: false` 时 PI_Adapter 不参与 forward；即使代码中存在未训练 adapter 参数，也不得影响推理输出。
- 如果用户选择 `gripper_no_tactile`，不仅默认不使用触觉，而且必须禁止 tactile adapter 二阶段；否则会出现“没有 tactile 数据却训练 PI_Adapter”的语义错误。

### 4.8 分阶段训练与验证策略

由于 Kuavo-DECO 同时替换了视觉前端并改造了触觉输入，直接开启触觉 adapter 会让视觉问题和触觉问题混在一起。当前冻结分阶段策略如下：

1. **视觉-only 阶段**
   - 配置：`training_stage: visual_main`、`use_tactile: false`、`use_tactile_lora: false`。
   - 目标：验证 RGB-D 前端、30Hz 数据、profile 对应 action_dim、Flow Matching 主干能独立闭环。
   - 允许训练范围：RGB-D bridge、视觉投影层以及按配置允许的 ResNet backbone。
   - 该阶段允许 `tactile_left_max` / `tactile_right_max` 仍为 `null`，因为 tactile 分支不参与 forward 与 loss。
   - 训练完成后通过 LeRobot `save_pretrained()` 保存 run 根目录与 `.safetensors` epoch 权重；若用户不使用触觉，部署资产就是该 run 根目录加选定 epoch 权重。

2. **触觉 adapter 阶段**
   - 配置：`training_stage: tactile_adapter`、`use_tactile: true`、`use_tactile_lora: true`。
   - 作为第二次独立训练启动，通过 `base_policy_path` 加载视觉-only / RGB-D 主干的选定 `.safetensors` policy 权重目录。
   - 冻结预训练主干，训练 tactile encoder、tactile cross-attention、PI_Adapter 和必要的 Kuavo 触觉桥接参数。
   - 进入该阶段前必须填写正数 `tactile_left_max` / `tactile_right_max`，否则触觉尺度不符合 DECO-style normalization。
   - 第二阶段虽然只更新 tactile/adapter 参数，但 RGB-D visual branch 与 state branch 仍必须参与 forward，因为 tactile adapter 学到的是“在当前视觉和状态上下文下如何修正动作预测”。
   - 该阶段仅允许 `qiangnao_tactile` profile；`gripper_no_tactile` 没有 `observation.tactile`，必须停留在 `visual_main` 路线。

3. **部署验证阶段**
   - 先关闭触觉进入仿真或真机 dry-run，确认 `qiangnao_no_tactile` 与 `gripper_no_tactile` 的 RGB-D 主链路稳定。
   - 再开启触觉 adapter，以 `qiangnao_tactile` 做仿真、离线 replay 或低风险真机验证。
   - 最后经过 dry-run、低速限幅、完整闭环三步上实机。

重要澄清：

- 两阶段训练不是在同一个 epoch loop 中“每个 epoch 先跑主干、再跑 LoRA”。
- 两阶段训练也不是 `train_policy.py` 在第一阶段 100 epoch 结束后自动切换配置并继续第二阶段。
- 当前冻结方案是两次独立启动：

```text
启动 1：visual_main
  epoch 1..N：训练 RGB-D + state + DECO Flow Matching 主干
  输出：第一阶段 run 根目录 + 选定 .safetensors epoch 权重

启动 2：tactile_adapter
  加载启动 1 的选定 epoch policy 权重
  冻结已经加载成功且 shape 匹配的主干参数
  epoch 1..M：训练 tactile encoder / tactile cross-attention / PI_Adapter
  输出：第二阶段 run 根目录 + 选定 .safetensors epoch 权重
```

- 若用户选择完全不使用触觉，则只执行启动 1，不执行启动 2。
- 如需自动衔接两阶段，后续可新增 DECO 专用 orchestrator 脚本；不建议把 stage switching 写进通用 `train_policy.py` 核心循环。

### 4.9 DECO 专用 preprocessor 放置边界

当前冻结方案：

- `lerobot_patches/` 只做 LeRobot 全局兼容补丁，例如新增 `FeatureType.TACTILE`、修改 `dataset_to_policy_features` 的字段类型识别。
- DECO 专用 preprocessor 放在 `kuavo_train/wrapper/policy/deco/`，建议文件名为 `DECOProcessor.py` 或等价命名。
- DECO preprocessor 的职责：
  - 对 RGB/depth 执行同步 deterministic `Resize/Letterbox`。
  - RGB 使用 bilinear 插值与灰色 padding `128`。
  - depth 使用 nearest 插值与独立 padding，默认 `0`。
  - 确保 RGB 随机增强只作用于 RGB，且顺序为 letterbox 之后、LeRobot normalizer 之前。
  - 训练与推理使用一致的空间预处理参数。
- `DECOPolicyWrapper.forward()` 不做 raw-pixel 语义的 resize/padding；它只接收已经预处理和归一化后的 batch，完成字段解包、tactile max normalization、DECO 模型调用和 loss 计算。

这样分层的原因：

- 如果在 `policy.forward()` 中执行 letterbox，输入已经可能经过 LeRobot normalizer，`fill=128` 将不再代表 raw RGB 灰色 padding。
- 把 DECO 专用逻辑放进 `lerobot_patches/` 会污染 ACT/DP 等其他 policy 的全局行为。
- 把 preprocessor 和 wrapper 放在同一 policy 目录中，便于后续维护 DECO 专属训练/部署一致性。

### 4.10 权重格式与路径语义

当前冻结语义：

- `base_policy_path`：Kuavo/LeRobot 第一阶段 `visual_main` 选定 epoch 的 policy 权重目录，内部至少包含 `.safetensors` 与 `config.json`。第二阶段 `tactile_adapter` 优先从这里加载主干权重。
- `adapter_model_path`：第二阶段或部署阶段可选加载的 adapter/full policy 目录，面向 Kuavo `.safetensors` 资产。
- `deco_init_pth_path`：仅用于兼容 DECO 原生 `.pth` 或历史 PyTorch checkpoint 初始化，不作为 Kuavo-DECO 正式部署资产。
- `load_external_init_weights`：训练初始化时为 `true`；最终保存 policy 时置为 `false`，并清空上述三个路径，避免部署或迁移时再次访问外部权重文件。

最终部署规则：

- 部署配置按原 Kuavo 三层路径填写：`task/method/timestamp` 对应 `outputs/train/<task>/<method>/<timestamp>/` run 根目录，`epoch` 只选择 `epoch<epoch>` 权重子目录。
- `policy_preprocessor.json` 与 `policy_postprocessor.json` 保存在 run 根目录；权重和 `config.json` 保存在所选 `epoch<epoch>/` 子目录。因此 `epochbest/` 单独拷贝不是完整可部署 policy 包。
- 通用 `configs/deploy/kuavo_env.yaml` 保持 ACT/DP 默认语义；DECO 使用 `configs/deploy/kuavo_deco_env.yaml` 作为唯一部署入口，记录 RGB-D depth topic、run-root 部署路径、推理模式和本地/服务端运行模式。
- `configs/deploy/kuavo_deco_env.yaml` 必须明确 `deco.inference_mode` 的三个可选值：
  - `qiangnao_tactile`：灵巧手 28D + 30D `observation.tactile`，要求加载已经保存 `use_tactile=true`、`use_tactile_lora=true` 的二阶段 tactile adapter checkpoint。
  - `qiangnao_no_tactile`：灵巧手 28D，但不订阅、不输入 `observation.tactile`，要求加载已经保存 `use_tactile=false`、`use_tactile_lora=false` 的视觉主干 checkpoint。
  - `gripper_no_tactile`：二爪夹 18D，不订阅、不输入 `observation.tactile`，支持 `eef_type=leju_claw` 与 `eef_type=rq2f85`；二者共享 18D state/action schema，只在 topic、状态读取和下发缩放上沿用当前工具链差异。
- 部署配置只能选择和校验权重，不应强行覆盖 checkpoint 中已保存的模型结构字段。`deco.inference_mode`、`state_layout`、`eef_type`、`use_tactile`、`use_tactile_lora`、`action_dim` 必须与所选 epoch 子目录中的 `config.json` 语义一致，否则应在加载阶段早失败。
- `configs/deploy/kuavo_deco_env.yaml` 必须明确 `head_state_source` 的两个可选值：
  - `live_joint_q`：从 `/sensors_data_raw.joint_data.joint_q[26:28]` 实时读取头部 yaw/pitch，更贴近机器人当前真实状态，但可能带入传感器微小抖动。
  - `fixed_config`：使用部署配置中的 `head_init` 作为固定头部 state，更稳定，适合头部部署期间保持固定姿态的任务，但必须确认与训练数据头部姿态一致。
- 第一轮部署实现优先打通本地单进程推理闭环，保持与当前 ACT/DP 主路径一致：`raw obs -> run-root preprocessor -> policy.select_action -> run-root postprocessor -> env.step`。优先覆盖 `real_single_test.py` 与 `sim_auto_test.py`。
- server/client 推理采用 Kuavo ACT 原版语义，不把 processor 迁移到 server：
  - eval/client 调用侧继续执行 `raw obs -> run-root preprocessor -> PolicyClient.select_action -> run-root postprocessor -> env.step`。
  - server 只接收已经由调用侧预处理过的 observation，执行 `policy.select_action(processed_obs)`，并返回尚未 postprocess 的模型 action。
  - 该边界可同时服务 ACT、DP 与 DECO，避免 client/server 双重归一化，也避免 server 混入 ROS raw observation 与硬件执行职责。
  - `kuavo_deploy/kuavo_service/server.py` 应通过启动参数或 `KUAVO_DEPLOY_CONFIG` 选择 `configs/deploy/kuavo_deco_env.yaml` 等部署配置，并按 `policy_type` 加载 `act`、`diffusion`、`deco`；DECO 加载阶段必须复用 checkpoint/config 一致性校验。
  - `kuavo_deploy/kuavo_service/client.py` 保持 `PolicyClient.select_action(obs_dict)` 接口不变，仅允许补充 timeout、api_token 和 server error 处理。
- `ObsBuffer` 对 DECO depth 使用 `depth_encoding: compressed_image` 时直接解码 `/cam_h/depth/image_raw/compressed`；旧 `compressedDepth_png` 解码仍保留给兼容 topic。
- `ObsBuffer` 仅在 `deco.inference_mode=qiangnao_tactile` 时对 `/dexhand/touch_state` 构造 30D `observation.tactile`，顺序为左手 15 + 右手 15，量纲换算保持 normal force `/100`。
- 部署侧应新增 DECO 映射 helper 集中封装 28D/18D state/action 逻辑，避免把 profile 细节散落在 `KuavoBaseRosEnv` 中。
- `state_layout: deco_28d` 的在线 state/action 顺序与离线 converter 一致：左臂 7 + 左手 6 + 右臂 7 + 右手 6 + 头部 2；该 layout 同时服务 `qiangnao_tactile` 与 `qiangnao_no_tactile`。
- `state_layout: deco_18d` 的在线 state/action 顺序与离线 converter 一致：左臂 7 + 左夹爪 1 + 右臂 7 + 右夹爪 1 + 头部 2；该 layout 服务 `gripper_no_tactile`，并同时支持 `leju_claw` 与 `rq2f85`。
- `KuavoBaseRosEnv.step()` 对 DECO 28D action 执行双臂 14D 与双手 12D；对 DECO 18D action 执行双臂 14D 与左右夹爪 2D。两种 layout 的 head action 当前都只保留维度，不下发头部控制。

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

- [x] 更新 `README_DECO.md`，说明 Kuavo-DECO 当前采用 RGB-D 前端，而不是原生双 RGB。
- [x] 新建 `configs/data/KuavoRosbag2Lerobot_deco.yaml`，默认 30Hz、use_depth true。
- [x] 在数据配置中默认记录 `rgb_topic: /cam_h/color/image_raw/compressed`、`depth_topic: /cam_h/depth/image_raw/compressed`、`depth_encoding: compressed_image`；在转换脚本中兼容 `/cam_h/depth/image_raw/compressedDepth` + `compressedDepth_png`，并把 `/camera/depth/image_rect_raw` / `raw_16uc1` 继续标为待复核候选。
- [x] 新建 `configs/policy/deco_config.yaml`，默认 ResNet34 RGB-D、control_hz 10、action_stride 3，并显式包含 `training_stage`、`use_tactile_lora`、`tactile_lora_rank`、`freeze_pretrained_main`、`base_policy_path`、`adapter_model_path`、`deco_init_pth_path`。
- [x] 在 `configs/policy/deco_config.yaml` 中明确 `Resize/Letterbox` 是确定性 RGB-depth 空间预处理，不属于随机增强池；默认采用 DECO `256x256 letterbox`，RGB padding `128`，depth padding 独立配置。
- [x] 在 `configs/policy/deco_config.yaml` 中扩展 `RGB_Augmenter`：保留 Kuavo ACT 的 Identity/ColorJitter/SharpnessJitter/RandomMask/RandomBorderCutout/GaussianNoise/GammaCorrection，并新增 GaussianBlur；默认 Identity/Notransform 权重 `3.0`，其他增强权重 `1.0`，`max_num_transforms: 1`。
- [x] 在 DECO trunk 与 wrapper 中明确 RGB/depth 独立 ResNet，cross attention 后仍按两路 visual tokens 接入 DECO；不额外保留与当前模型逻辑无关的视觉 token 模式开关。
- [x] 在 DECO wrapper 中明确 `observation.state` 由 LeRobot preprocessor 归一化后直接进入 DECO `obs_encoder + time embedding` 路线，不使用 DECO 原生手动归一化，也不改成 ACT state token / VAE 路线。
- [x] 在 `lerobot_patches/custom_patches.py` 中新增或扩展 `FeatureType.TACTILE`，确保 `observation.tactile` 不被识别为 STATE。
- [x] 新建 `kuavo_train/wrapper/policy/deco/DECOProcessor.py` 或等价模块，实现 DECO 专用 RGB-D preprocessor。
- [x] 在 DECO wrapper 中明确 `observation.tactile` 不走 STATE `MEAN_STD`，而是在 `/100` 牛顿单位基础上按左右手 tactile max 做 DECO-style 归一化并默认 clamp 到 `[0, 1]`。
- [x] 在 `configs/policy/deco_config.yaml` 中为 `tactile_left_max` / `tactile_right_max` 写明含义、单位、可填 `null` 的场景，以及 `use_tactile: true` 时必须为正数的约束。
- [x] 在 `configs/policy/deco_config.yaml` 中明确两阶段训练是两次独立启动：`visual_main` 产出可部署 run 目录与 `.safetensors` epoch 权重，`tactile_adapter` 加载第一阶段 policy 并冻结主干训练 tactile/PI_Adapter。
- [x] 在数据配置与转换脚本中新增 `end_effector_profile`：`qiangnao_tactile` 使用 `dataset.eef_type=qiangnao`，输出 28D + 可选 tactile；`gripper_no_tactile` 使用 `dataset.eef_type=leju_claw/rq2f85`，输出 18D + no tactile。
- [x] 在 `configs/policy/deco_config.yaml` 和 `DECOConfigWrapper` 中新增 `gripper_no_tactile` 训练约束：`action_dim=18`、`use_tactile=false`、`use_tactile_lora=false`、禁止 `training_stage=tactile_adapter`。
- [x] 在 DECO wrapper 中明确 loss 严格使用 `F.mse_loss(out, noise - action)`，第一版不使用 `action_is_pad` mask。
- [x] 在 DECO wrapper/config 中明确最终保存的 policy 权重不依赖外部初始化路径：保存时清空外部初始化路径，加载最终 `.safetensors` 时不再读取第一阶段目录或 `.pth`；完整部署包仍以 run 根目录为单位。
- [x] 在部署入口中静态注册 `deco` / `DECO` policy 类型，并导入 DECOProcessor 以注册 `deco_rgbd_letterbox_processor`。
- [x] 新建 `configs/deploy/kuavo_deco_env.yaml`，并明确 DECO 部署资产采用 Kuavo 原有 run 根目录：`outputs/train/<task>/<method>/<timestamp>/`；`epochbest/` 只是权重子目录，不是完整部署包。
- [x] 在 `kuavo_deploy` 中静态接入 DECO 在线部署链路：阶段六已覆盖本地单进程推理闭环，并同时支持 `qiangnao_tactile`、`qiangnao_no_tactile`、`gripper_no_tactile` 三种模式；server/client 按 Kuavo ACT 原版语义完成静态接入，processor 仍归 eval/client 调用侧，server 只负责 policy 推理。
- [x] 新建 `kuavo_data/CvtRosbag2Lerobot_DECO.py`。
- [x] 新建/扩展 `kuavo_data/validate_deco_lerobot_dataset.py`，检查单 rosbag 转换结果的字段、维度、30Hz 时间轴、RGB-depth 对齐、depth decoder、profile 对应 action 映射与可选 30 维 tactile 量纲。
- [x] 新建 `kuavo_train/wrapper/policy/deco/DECOConfigWrapper.py`。
- [x] 新建 `kuavo_train/wrapper/policy/deco/DECOPolicyWrapper.py`。
- [x] 已确定阶段四改动核心目标为 `third_party/deco/` 下的 Kuavo 定制 DECO 主干，并仅在 wrapper/config/patch 侧保留当前模型逻辑需要的最小字段。
