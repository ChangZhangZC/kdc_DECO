# Kuavo-DECO RGB-D 架构迁移与系统集成宏观计划书

> **生成日期**：2026-05-08  
> **重构日期**：2026-05-18
> **核心架构策略**：**Wrapper 融入模式** + **Kuavo/ACT 风格 RGB-D 视觉前端移植** + **DECO Action-Token Flow Matching 主干保留** + **Tactile Plugin/LoRA 低秩微调保留** + **30Hz 数据 / 10Hz 控制解耦**  
> **使用说明**：本计划书为 Kuavo-DECO 集成的当前唯一“真理源 (Single Source of Truth)”。后续 AI Agent 应在每次会话开始时读取此文件与 `AI_Logs.md`，并在完成任务后更新 Checkbox 状态。

---

## 当前分支覆盖方案：3View RGB 前端

> **记录日期**：2026-07-26
> **适用分支**：`deco/feature/3view-rgb`
> **状态**：代码静态实现已完成；训练、仿真和实机运行验收须在允许执行代码的环境中完成。
> **覆盖关系**：本节是当前分支的有效架构定义；下方 RGB-D、depth 与 early cross-attention 章节仅作为历史方案和实验记录保留。

### 冻结架构

```text
固定三路 RGB（head、left wrist、right wrist）
  -> 同步 letterbox 到 256x256
  -> [B, 3, 3, 256, 256]
  -> 沿视角维合并为 [B*3, 3, 256, 256]
  -> 共享 ResNet34 + img_head
  -> [B, 3, 64, 512]
  -> 三个 camera embedding + 每个视角独立的 8x8 二维 RoPE
  -> 按 [head][left wrist][right wrist] 拼接为 192 visual tokens
  -> DECO MMAttention + Flow Matching action tokens
```

- [x] 固定 `rgb_keys` 为恰好三个有序入口，默认顺序为 head、left wrist、right wrist；不提供 N-view 或 head-only fallback。
- [x] 三路 RGB 共用一套 ResNet34 与 `img_head`，通过三项 camera embedding 区分物理视角。
- [x] 删除 depth 输入、depth backbone、depth normalization、RGB-D fusion 与前端 early cross-attention。
- [x] 当前训练版本的 DECO rosbag 转换链路已同步禁用 depth topic 读取、时间对齐、LeRobot feature 创建与帧写入；历史实现以中文注释保留，输出仅包含 RGB、state、action 与可选 tactile。
- [x] DECO rosbag 转换链路已固定写入 `head_cam_h`、`wrist_cam_l`、`wrist_cam_r` 三路 RGB；按 head 时间轴对齐并与 3View RGB policy 的键名和顺序一致。
- [x] 保留原生 DECO `MMAttention`、state、tactile、Flow Matching loss、action chunk 与 action dispatch。
- [x] 训练 processor、policy wrapper、原生 DECO 推理入口、实机、仿真及 server/client 部署统一使用三个 key 的固定顺序。
- [x] 任一路相机缺失、重复、shape 不一致、部署 topic 不可提供或帧同步失败时显式报错，不复制 head、不复用旧帧、不静默降级。
- [x] 从头训练，默认关闭外部 checkpoint 初始化，不实现旧 RGB-D checkpoint 的 shape 兼容或参数迁移。
- [x] 保持 `kuavo_data/CvtRosbag2Lerobot_DECO.py`、`configs/data/KuavoRosbag2Lerobot_deco.yaml` 与已有 LeRobot 数据不变；训练仅从数据集读取三路 RGB。
- [x] 动作后端收敛为两种互斥模式：`receding_horizon` 与 `temporal_ensemble`。
- [x] 默认 `chunk_size=32`、`n_action_steps=16`，以30Hz连续执行原始 action index `0..15`，约0.533s后使用最新观测重新推理。
- [x] 删除部署期 stride 降频；两种保留模式均按 checkpoint 的 `dataset_hz` 下发动作。
- [x] 修复 Hydra 将嵌套 `PolicyFeature` 展平为普通字典后，3View RGB 配置在 `__post_init__` 提前访问 `.type` 的问题；构造期间仅按 key 过滤 depth，随后复用训练入口已有的 input/output feature 类型恢复逻辑。
- [x] 删除 `.gitignore` 中过宽的 `DECO/` 规则，避免误忽略路径中名为 `deco` 的训练 wrapper 和后续新增文件。
- [x] 完成 `git diff --check`、禁止项文本检索、关键 tensor shape 与 key 顺序的逐文件静态审查。
- [x] 删除不参与入口选择、动作下发或安全控制的 `deco.runtime_mode` 字段；仿真、真机与 server/client 模式统一由实际启动入口决定。
- [ ] 在允许运行代码的环境中验证三路 batch 构造、训练 forward/loss 与 checkpoint 保存加载。
- [ ] 在仿真和实机分别验证左右腕物理对应、相机丢帧 fail-fast、30Hz Receding Horizon 与三种末端执行器 profile。
- [ ] 测量 Receding Horizon 重规划周期的推理阻塞、实际控制频率与 action jitter。
- [ ] 对比原 `optimal-depth` 方案的 loss、成功率、推理延迟、显存占用与空抓比例。

## 0. 当前冻结的总体架构

### 0.1 一句话架构

将 DECO 原生的“双 RGB 图像输入 + 自带 ResNet34 图像编码”替换为 Kuavo 现有 ACT/DP 工具链中的 RGB-D 输入与视觉增强方案；保留 DECO 的 action token 表示、Flow Matching 训练目标和动作去噪推理主干；末端执行器从单一 28 维灵巧手接口放宽为配置化 profile：`qiangnao_tactile` 使用 28 维灵巧手 + 可选触觉，`gripper_no_tactile` 使用 18 维二夹爪且不启用触觉 LoRA 二阶段。

```text
Kuavo rosbag RGB + depth + state + action + optional tactile
  -> 30Hz LeRobot RGB-D 数据集（profile 决定 state/action 维度与是否写入 tactile）
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

### 0.2 已确认技术决策

- DECO 不再沿用原生“双目 RGB 近似深度”的视觉假设；Kuavo 版本采用现有工具链的 **RGB + depth** 输入方式。
- 视觉前端默认 `vision_backbone: resnet34`，允许通过配置切换为 `resnet18`。
- RGB 与 depth 分别通过独立 ResNet backbone 编码；depth backbone 使用 1-channel 输入，初始化策略参考 ACT：用 RGB ResNet 第一层权重在通道维求均值初始化 depth conv1。RGB 与 depth 不共用同一个 ResNet。
- 阶段三第一版保留 DECO 原生“两路视觉 token”结构，但把语义改为 `RGB/depth` 或 `fused_rgb/fused_depth`；暂不把 RGB-D 过早压成单路 visual token，以降低 DECO 主干改动风险。
- RGB 视觉处理拆成两层：`Resize/Letterbox` 属于确定性空间预处理，不放入随机增强池；`RGB_Augmenter` 属于训练期随机增强池。默认空间输入保持 DECO 原生 `256x256 letterbox`：RGB padding 使用灰色 `fill=128`，depth padding 单独配置，默认使用 `0` 或 invalid depth。
- RGB 增强复用 Kuavo 现有 `RGB_Augmenter`，并吸收 DECO 原生 blur 思路：Identity/Notransform、ColorJitter、SharpnessJitter、RandomMask、RandomBorderCutout、GaussianNoise、GammaCorrection、GaussianBlur 等。
- RGB 增强采样权重参考 Kuavo ACT：保留原图的 Identity/Notransform 权重较高（默认 `3.0`），其他增强默认 `1.0`，每次默认从增强池中采样一个变换。
- depth 不做颜色类增强；仅与 RGB 共享 `Resize/Letterbox` 等空间同步变换，RGB 使用双线性插值，depth 使用 nearest 插值，避免破坏深度物理含义。
- depth topic 与 decoder 以当前 rosbag 实际存在的 topic 为准：默认优先 `/cam_h/depth/image_raw/compressed` + `compressed_image`，同时兼容 `/cam_h/depth/image_raw/compressedDepth` + `compressedDepth_png`；raw `16UC1` 仍仅作为显式启用的 fallback。
- depth 第一版沿用现有转换策略：`uint16/mm depth -> depth_range clip -> per-frame normalize -> uint8 -> repeat 3 channels`，wrapper 再取单通道进入 1-channel depth backbone。该策略兼容当前 LeRobot image/video writer，但不保留跨帧绝对毫米尺度；后续可升级为 `uint16` 或单通道 metric depth 存储方案。
- 除 DECO 专属 profile schema、30Hz 目标时间轴、可选 30 维触觉解析、depth 单通道语义保留外，数据清洗的 topic map、RGB-D 读取方式、state/action 来源应尽可能复用现有 `CvtRosbag2Lerobot.py` 与 `kuavo_data/common/kuavo_dataset.py` 的稳定逻辑。
- DECO 数据清洗阶段新增 `end_effector_profile` 语义：`auto` 会从 `dataset.eef_type=qiangnao` 推导 `qiangnao_tactile`，输出 28D state/action，并可写入 30D `observation.tactile`；从 `dataset.eef_type=leju_claw` 或 `rq2f85` 推导 `gripper_no_tactile`，输出 18D state/action，不写入也不要求 `observation.tactile`。若显式填写 profile，必须与 `dataset.eef_type` 一致。
- `observation.state` 的归一化采用 Kuavo/LeRobot preprocessor 与 dataset stats，避免沿用 DECO 原生 `dataset.py` / `inference.py` 中的手动二次归一化；但进入模型的方式保留 DECO：归一化后的 profile 维 state 经 `obs_encoder(action_dim -> dim)` 后加到 time embedding，用于调制 MMAttention，不改成 ACT 的 state token / VAE encoder 路线。
- `observation.tactile` 仅在 `qiangnao_tactile` 且 `use_tactile: true` 时作为独立触觉模态进入 tactile encoder / cross-attention / PI_Adapter，不与 `observation.state` 混拼；`gripper_no_tactile` 不允许启用 tactile 或 tactile LoRA 二阶段。
- 触觉洗数据阶段的 `/100` 只是把 Kuavo normal force 转成牛顿；进入 DECO 模型前，触觉遵循 DECO 原生 tactile 处理思想，按左右手各自的 tactile max 做归一化并默认 clamp 到 `[0, 1]`，再进入 Kuavo 30 维触觉 encoder。不得让 `observation.tactile` 被当作普通 STATE 走 LeRobot `MEAN_STD` 归一化。
- `observation.tactile` 在 LeRobot feature mapping 中应被识别为独立 `FeatureType.TACTILE`，normalization mapping 默认使用 `IDENTITY`；`lerobot_patches/` 只负责这种全局 feature/type 兼容补丁，不承载 DECO 专用视觉预处理或模型逻辑。
- DECO 专用 preprocessor 放在 `kuavo_train/wrapper/policy/deco/` 下，负责 RGB/depth 同步 `Resize/Letterbox`、RGB-only 随机增强接入顺序和训练/推理一致性；该逻辑不放在 `policy.forward()` 中，避免在 LeRobot normalizer 之后再做 raw-pixel 语义的 padding。
- DECO 主干保留 Flow Matching 训练目标：`F.mse_loss(out, noise - action)`；第一版不额外乘 `action_is_pad` mask，以严格对齐 DECO 原生 loss 行为。
- DECO 主干保留 action token 形式：动作序列先被编码成 token，再与视觉/state/tactile 条件进行 attention。
- DECO 触觉微调保留源码中的 `plugin=True` / `PI_Adapter` 机制：它不是外部 PEFT LoRA，而是 DECO 自实现的低秩 down/up residual adapter。
- 触觉 adapter 训练默认遵循 DECO 原始范式，但在 Kuavo/LeRobot 工具链中采用“两次独立启动”：第一阶段完整训练 RGB-D + state 主干并保存 Kuavo run 目录；第二阶段再加载第一阶段选定 epoch 的 policy 权重，冻结 checkpoint 中已经存在且 shape 匹配的主干参数，只训练新出现的 tactile encoder、tactile cross-attention、PI_Adapter 和必要桥接参数。若用户不使用触觉，则第一阶段 run 目录 + 选定 epoch 权重即可直接用于仿真或实机部署。
- `.pth` 仅作为导入 DECO 原生权重或历史 PyTorch checkpoint 的兼容入口；Kuavo-DECO 正式训练、续训和部署资产采用原 Kuavo 约定：`outputs/train/<task>/<method>/<timestamp>/` 是完整 run 根目录，`epoch<epoch>/` 只是其中被选择的权重子目录，`epochbest/` 单独拷贝不是完整部署包。
- `train_policy.py` 的核心训练循环、optimizer step、epoch loop 和 dataloader loop 不应为 DECO 改写；阶段四只允许做最小策略注册（例如在 policy registry/dict 中加入 `deco`）以及必要的 processor/factory 轻量接入。
- 原始采集流可能为 100Hz 或更高；LeRobot 训练数据统一重采样到 **30Hz**。
- 部署控制频率必须与 checkpoint 的 `dataset_hz` 一致；当前数据为30Hz时，两种 dispatcher 均按30Hz消费动作，不再做部署期 stride 降频。

### 0.3 关键解释

- `chunk_size` 表示 DECO 一次预测的 action chunk 长度，不表示视觉帧数。
- `inf_step` 表示 Flow Matching 推理时的去噪步数，不表示机器人控制频率。
- `train_hz` 属于洗数据阶段，决定 LeRobot 数据集中相邻样本的时间间隔。
- `env.ros_rate` 属于部署阶段，必须与 checkpoint 的 `dataset_hz` 一致。
- `use_tactile_lora` 是 Kuavo 配置层面对 DECO 原生 `plugin` 的语义化开关；开启后才使用 tactile PI_Adapter 低秩微调。
- `end_effector_profile=qiangnao_tactile` 表示 DECO 使用灵巧手 schema：左臂 7 + 左手 6 + 右臂 7 + 右手 6 + 头部 2，共 28D，可选择进入 tactile adapter 二阶段。
- `end_effector_profile=gripper_no_tactile` 表示 DECO 使用二夹爪 schema：左臂 7 + 左夹爪 1 + 右臂 7 + 右夹爪 1 + 头部 2，共 18D；该 profile 禁止 `use_tactile=true`、`use_tactile_lora=true` 和 `training_stage=tactile_adapter`。
- `freeze_pretrained_main` 表示是否冻结已经从 `base_policy_path` 或 `deco_init_pth_path` 加载成功的主干参数；默认开启，以保留 DECO 触觉 adapter 微调策略。保存最终 policy 时外部初始化路径会被清空，继续训练 tactile adapter 时仍按参数名冻结主干，只保留 tactile/PI_Adapter 相关参数可训练。
- `training_stage=visual_main` 表示第一阶段主干训练；`training_stage=tactile_adapter` 表示第二阶段触觉 adapter 微调。二者是两次独立训练启动，不是在同一个 epoch loop 中自动交替。
- `base_policy_path` 表示 Kuavo/LeRobot 第一阶段选定 epoch 的 `.safetensors` policy 权重目录；`deco_init_pth_path` 或等价字段若存在，仅表示从 DECO 原生 `.pth` 权重做初始化兼容；`load_external_init_weights` 只在训练初始化阶段为 `true`，最终保存的 policy 会设为 `false`。部署配置仍按 `task/method/timestamp + epoch` 组合定位 run 根目录和权重子目录。

---

## 版本 2.0 修正计划：RGB-D 视觉融合消融与空抓问题排查

> **记录日期**：2026-06-02  
> **状态**：代码接入已完成，尚待用户在允许运行的环境中执行 MuJoCo 训练/部署 ablation。
> **边界**：本章节独立记录后续 v2.0 修正计划，不回填修改前文已完成阶段的历史记录。

### 背景现象

当前 Kuavo-DECO 在 MuJoCo 模仿学习闭环中出现“动作形态合理，但视觉与动作没有稳定对齐”的现象，具体表现为空抓、任务成功率为零。初步判断动作先验并非完全错误，优先怀疑 RGB-D 视觉 grounding、动作触发位置或训练/部署视觉语义存在偏移。

### 当前风险假设

- 当前 DECO 视觉前端在 RGB/depth 各自 ResNet 后、进入 `pack_visual_token_sequences()` 之前执行 RGB-depth 双向 cross attention。
- 这一步发生在显式二维 RoPE 与 stream embedding 之前，因此 RGB token 可以全局 attend 到任意 depth token；在低数据量或弱视觉监督下，它可能提前扰乱 RGB-D 原本的空间对应关系。
- 对抓取任务而言，RGB 提供纹理、边界和语义，depth 提供几何距离和空间结构。如果 early cross attention 学到错误跨模态关联，后续 action token 看到的视觉 token 可能已经被污染，从而表现为空抓或目标定位偏移。
- 当前 DECO 默认以30Hz连续执行 action chunk 前16步，约0.533s 后根据最新观测重新推理。
- 对抓取后放置这类接触敏感阶段，仍需评估 Receding Horizon 开环长度是否影响物体滑动、篮筐碰撞或末端偏移后的及时修正。

### stream embedding 与 RoPE 语义

- `RoPE` 负责表达二维空间位置：RGB token 与 depth token 共享同一套空间坐标，因为二者来自对齐 RGB-D 输入。
- `stream embedding` 负责表达 token 来源：`stream_id=0` 表示 RGB，`stream_id=1` 表示 depth，作用类似 BERT 的 segment embedding。
- 若去除 early cross attention，RGB/depth 在进入 DECO 主干前不发生内容交汇；它们只会被 concat 成 `[RGB tokens, depth tokens]`，分别加 stream embedding，并在 `MMAttention` 中分别施加同一套二维 RoPE。
- RGB/depth/action 的真正信息交互推迟到 DECO 主干 `MMAttention` 内部完成，因此该方案不是彻底隔离 RGB 与 depth，而是把交汇点从 ResNet 后、RoPE 前推迟到 RoPE/stream embedding 之后。

### v2.0 计划

- [x] 在 `configs/policy/deco_config.yaml` 中新增 `policy.visual_fusion_mode`，默认保持 `cross_attention`，避免静默改变既有训练语义。
- [x] 支持 `visual_fusion_mode: cross_attention`：沿用当前 RGB/depth ResNet token 先经过双向 cross attention，再以 `fused_rgb/fused_depth` 两路 token 进入 DECO 主干。
- [x] 支持 `visual_fusion_mode: direct_tokens`：RGB/depth ResNet token 不做 early cross attention，直接进入 `pack_visual_token_sequences()`，再加 stream embedding、RoPE，并进入 DECO 主干。
- [x] 在 `kuavo_train/wrapper/policy/deco/DECOConfigWrapper.py` 中注册并校验 `visual_fusion_mode`，只允许 `cross_attention` 与 `direct_tokens`。
- [x] 在 `kuavo_train/wrapper/policy/deco/DECOPolicyWrapper.py` 中向 `DECO(...)` 透传 `visual_fusion_mode`，不改变 batch 输入字段、loss、action queue、tactile 逻辑或 pre/postprocessor 顺序。
- [x] 在 `third_party/deco/models/deco/deco.py` 中保留 `RGBDepthCrossAttentionFusion` 类和当前 cross attention 路径，同时新增 `direct_tokens` 分支：`rgb_tokens/depth_tokens -> pack_visual_token_sequences(...)`。
- [x] 静态确认两种模式输出 shape 均为 `[B, 2L, dim]`，保证 `MMAttention` 中 `feat_len = total_img_len / 2` 的假设继续成立。
- [x] 使用 `n_action_steps` 控制 Receding Horizon 每次推理后连续消费的原始 action 数量；`null` 表示完整消费 chunk。
- [x] 在 `DECOConfigWrapper` 与部署配置中校验 `n_action_steps` 为正整数或 `null`，且不超过 `chunk_size`。
- [x] `n_action_steps` 只影响推理队列刷新频率，不改变模型结构、训练 loss、`chunk_size`、`action_delta_indices` 或权重 shape。
- [x] 在 `configs/deploy/kuavo_deco_env.yaml` 与 DECO 部署加载入口中新增 `deco.n_action_steps` runtime override，使旧 checkpoint 可在不重新训练的情况下直接做 `null/8/4/2/1` 队列长度消融。

### 后续实验设计

- [ ] Baseline A：`visual_fusion_mode=cross_attention`，保持当前实现作为对照。
- [ ] Ablation B：`visual_fusion_mode=direct_tokens`，除视觉融合模式外保持相同数据、相同超参数、相同训练轮数和相同部署配置。
- [ ] `direct_tokens` 优先从头训练，不建议直接严格加载 `cross_attention` checkpoint 续训。
- [ ] 对比指标重点记录 MuJoCo 成功率、空抓比例、抓取触发时目标与末端的空间关系、末端轨迹是否朝目标收敛、动作是否平滑。
- [ ] Receding Horizon 队列消融：保持当前 checkpoint、`chunk_size=32` 与30Hz不变，对比 `n_action_steps=null`、`16`、`8`、`4`、`2`、`1`。重点记录模型真实推理频率、动作 jerk、action clip 比例、放置阶段 target-current joint error 和 MuJoCo 成功率。
- [ ] `n_action_steps` 消融建议优先搭配 `inf_step=5/10/20` 做小网格测试；若 `n_action_steps=1/2` 造成推理耗时超过 10Hz 控制周期，应优先回退到 `4` 或降低 `inf_step`。
- [ ] 若 `direct_tokens` 明显改善空抓，后续再评估更温和的融合方式，例如 RoPE 后局部 cross attention、只在主干中融合，或保留 cross attention 但加入局部窗口/位置约束。
- [ ] 若 `direct_tokens` 无明显改善，则优先复查 RGB-depth 对齐、action 时间偏移、depth normalization、训练/部署 preprocessor 一致性、MuJoCo 相机视角与数据采集视角一致性。

---

## 阶段一：RGB-D 数据引擎阶段 (Data Engine Phase)

**核心目标**：新建 DECO 专用 rosbag 转 LeRobot 数据链路，保留 Kuavo 工具链的数据组织方式，但输出适配 DECO 的 profile 化字段：30Hz RGB-D、多模态 state/action、可选 tactile、灵巧手 28 维或二夹爪 18 维动作空间。

- [x] **1.0 技术决策记录与方案冻结**
  - [x] 新建 `Content/DECO_Technical_Decisions.md`，作为 DECO 集成技术决策记录文件。
  - [x] 记录阶段一早期方案：新建 `_deco.yaml` 数据配置、第一版固定 28 维 state/action、头部 state 使用 Inspector 验证后的实测固定角度、action 头部补零、暂不修改公共 reader。
  - [x] 将 Inspector 作为正式检查点：先通过只读脚本确认 rosbag schema、视觉源、depth 源、触觉字段、头部自由度。
- [x] **1.1 Rosbag Schema Inspector**
  - [x] 新建只读脚本 `kuavo_data/inspect_deco_stage1_schema.py`，检查关键 topic 的存在性、消息数量、频率、字段结构。
  - [x] 对 `/cam_h/color/image_raw/compressed` 解码并导出 `cam_h_full.jpg`、`cam_h_left_half.jpg`、`cam_h_right_half.jpg`。
  - [x] 检查 `/dexhand/touch_state`、`/dexhand/state`、`/sensors_data_raw`、`/kuavo_arm_traj`、`/joint_cmd`、`/control_robot_hand_position` 等字段长度。
- [x] **1.2 视觉策略重构冻结**
  - [x] Inspector 显示 `/cam_h/color/image_raw/compressed` 为 848×480 的完整头部 RGB 画面；左右半图只是裁切同一视野，不是真实双目。
  - [x] 放弃“单目复制成双路输入”和“左右切分伪双目”作为主路线。
  - [x] 新主路线冻结为：保存 Kuavo 头部 RGB + 对齐 depth，训练时通过 Kuavo/ACT 风格 RGB-D 视觉前端替换 DECO 原生双路视觉编码入口。
  - [x] 将该视觉策略同步写入 `README_DECO.md` 的数据转换与训练说明。
- [x] **1.3 完整 DECO RGB-D 数据转换流程实现**
  - [x] 新建 `configs/data/KuavoRosbag2Lerobot_deco.yaml`，默认 `train_hz: 30`、`use_depth: true`、`main_timeline_fps` 按真实主视觉流填写。
  - [x] 默认以当前实际数据配置为准：RGB 使用 `/cam_h/color/image_raw/compressed`；depth 默认使用 `/cam_h/depth/image_raw/compressed` + `compressed_image`，并兼容 `/cam_h/depth/image_raw/compressedDepth` + `compressedDepth_png`。
  - [x] 新建 `kuavo_data/CvtRosbag2Lerobot_DECO.py`，实现 DECO 专用 rosbag reader、时间戳采样、RGB-D 对齐、第一版固定 28 维 state/action 映射、30 维触觉提取和 LeRobot dataset 写入。
  - [x] DECO converter 优先复用现有 `KuavoRosbagReader` 的 topic map、message processor 与 nearest-neighbor 对齐思路；仅在 DECO 第一版固定 schema、30Hz 目标时间轴、30 维 tactile、28 维 action/state 和 depth 保存语义处做专用适配。
  - [x] 使用真实时间戳生成 30Hz 目标时间轴；不得依赖 `MAIN_TIMELINE_FPS // TRAIN_HZ` 的整数跳帧假设，以兼容 100Hz 或更高频采集流。
  - [x] 对 RGB、depth、state、action、tactile 统一采用 nearest-neighbor 时间对齐；后续若动作抖动明显，再单独评估插值策略。
  - [x] 暂不修改 `kuavo_data/common/kuavo_dataset.py` 公共 reader，避免影响 ACT/DP 既有转换链路。
  - [x] 新增 `end_effector_profile` 配置，沿用 ACT/DP 的 `dataset.eef_type` 入口选择末端类型：`auto` 可从 `qiangnao` 推导 `qiangnao_tactile`，从 `leju_claw/rq2f85` 推导 `gripper_no_tactile`。
  - [x] 将转换脚本从固定 28D + 必需 tactile schema 放宽为 profile 化 schema：灵巧手仍为 28D + 可选 tactile，二夹爪为 18D + no tactile。
- [x] **1.4 RGB-D 视觉流保存策略**
  - [x] 保存头部 RGB：`observation.images.head_cam_h`。
  - [x] 保存与头部 RGB 对齐的 depth：默认使用当前实际数据配置中的 `/cam_h/depth/image_raw/compressed`。
  - [x] 兼容实采与官方模拟数据的 depth topic 差异：`/cam_h/depth/image_raw/compressed` 绑定普通 `CompressedImage` 直解，`/cam_h/depth/image_raw/compressedDepth` 绑定 ROS compressedDepth PNG payload 解码。
  - [x] 保留 raw depth 候选：`/camera/depth/image_rect_raw`、`encoding=16UC1` 只作为配置化 fallback，必须由 Inspector/validator 复核后才能启用。
  - [x] DECO 转换脚本支持 `depth_encoding: compressed_image`、`compressedDepth_png`、`raw_16uc1` 与候选省略 encoding 时的 `auto` decoder；当前默认优先使用 `compressed_image`。
  - [x] depth 保存为 LeRobot 可识别的 depth feature；第一版按 Kuavo 现有兼容规范存为 3-channel depth image，并在 wrapper/config 中保留 1-channel depth backbone 语义。
  - [x] 若复用旧脚本中“depth clip 后归一化为 uint8 并 repeat 成 3 通道”的逻辑，必须在 DECO wrapper/config 中显式还原或声明 depth backbone 的输入通道语义；主路线仍偏向 1-channel depth backbone。
  - [x] 第一版 depth 转换策略冻结为：`uint16/mm depth -> depth_range clip -> per-frame normalize -> uint8 -> repeat 3 channels`，wrapper 取单通道送入 depth backbone。
  - [ ] 后续备选 depth 转换策略 A：保存 `uint16` 或单通道 metric depth feature，保留毫米尺度，再由 dataset stats / depth normalizer 统一归一化。
  - [ ] 后续备选 depth 转换策略 B：尽量保存 raw 或近 raw depth，在 wrapper 中按统一 depth range / stats 转成 float depth；该方案需要同步改数据转换、validator、policy config 和 wrapper。
  - [x] 转换阶段的 resize 策略与 Kuavo ACT/DP 工具链保持一致，默认 `640x480`。
- [x] **1.5 触觉频率与量纲降维**
  - [x] 提取 Kuavo 话题 `/dexhand/touch_state`。
  - [x] 将 Kuavo 原生触觉流下采样至 30Hz LeRobot 时间轴。
  - [x] 仅提取指尖/指腹的法向力 `normal_force1/2/3`，舍弃切向力和接近觉。
  - [x] 保留空间特征：5 指 × 3 点 × 2 手 = 30 维，并直接除以 100 缩放到牛顿量纲。
  - [x] 将 tactile 从全局必需字段改为 `qiangnao_tactile` 专属字段；`gripper_no_tactile` 数据集不写入 `observation.tactile`，也不允许 validator 要求 tactile。
- [x] **1.6 动作空间 (28 维) 索引重组**
  - [x] 在 `CvtRosbag2Lerobot_DECO.py` 中以 helper functions 建立 Kuavo -> DECO 重映射逻辑。
  - [x] 接收 Kuavo 原生上半身来源：`joint_q[12:19]` 左臂、`joint_q[19:26]` 右臂、`/dexhand/state` 或 `/control_robot_hand_position` 左右手、`joint_q[26:28]` 头部 state。
  - [x] 输出 DECO 强制 28 维排列：`左臂 0-6 -> 左手 7-12 -> 右臂 13-19 -> 右手 20-25 -> 头部 26-27`。
  - [x] arm action 继承现有清洗逻辑：优先使用 `/kuavo_arm_traj_synced`，否则 `/kuavo_arm_traj`，`/joint_cmd` 只作为 fallback 或一致性对照。
  - [x] hand action 继承现有清洗逻辑：使用 `/control_robot_hand_position` 的左右手目标位置；DECO 配置中固定使用左右手各 6 DoF，不使用 ACT/DP 默认的 `dex_dof_needed: 1` 压缩策略。
  - [x] `observation.state[26:28]` 逐帧保留对齐后的 `/sensors_data_raw.joint_q[26:28]`，`action[26:28]` 逐帧保留独立对齐后的 `/joint_cmd.joint_q[26:28]`；两者均不固定填充或相互复制。
  - [x] 新增 `gripper_no_tactile` 18D 顺序：`左臂 0-6 -> 左夹爪 7 -> 右臂 8-14 -> 右夹爪 15 -> 头部 16-17`。
  - [x] `leju_claw` 与 `rq2f85` 在清洗入口保留不同 topic 和归一化尺度，但进入 DECO 后共享 `gripper_no_tactile` 18D schema。
- [ ] **1.7 单 rosbag 转换试跑与数据一致性检查**
  - [x] 用户已提供已转换示例数据集 `data_example/lerobot`，可作为 1.7 validator 的首个检查对象。
  - [x] 静态查看 `data_example/lerobot/meta/info.json`，确认 `fps=30`、`total_episodes=1`、`total_frames=331`，且 metadata 中包含 RGB、depth、state、tactile、action 字段。
  - [x] 新建 `kuavo_data/validate_deco_lerobot_dataset.py`，用于检查转换结果是否符合 Kuavo-DECO 方案；该脚本不依赖 ROS1，也不读取 rosbag。
  - [x] 用户已首次运行 validator 并反馈 `data_example/lerobot/deco_validation_report.md`；当前 metadata、schema 与文件结构检查通过，但 parquet 数值检查被运行环境缺少 `pyarrow/fastparquet` 阻塞，video probe 被 `cv2/protobuf` 环境问题跳过。
  - [ ] **Debug Hook（后续数据状态复查）**：1.7 暂不关闭；训练前必须在具备 parquet engine 的环境中补跑完整 validator，确认 timestamp、state/action/tactile 数值语义与头部处理逻辑。
  - [ ] 在允许执行的环境中补跑完整 validator，并把控制台输出与 `deco_validation_report.md` 反馈回来；当前 Codex 机器遵守 No-Runtime 约束，不直接运行 Python。
  - [ ] 检查数据集字段：`observation.images.head_cam_h`、depth feature、`observation.state`、`action` 必须存在；`observation.tactile` 仅在 `qiangnao_tactile` 且配置要求触觉时必须存在。
  - [ ] 检查维度：RGB `(3,H,W)`、depth `(1,H,W)` 或等价 depth shape；`qiangnao_tactile` 为 state/action `(28,)`、tactile `(30,)`；`gripper_no_tactile` 为 state/action `(18,)` 且无 tactile。
  - [ ] 检查 depth 来源与编码：确认 `/cam_h/depth/image_raw/compressed` 可通过 `compressed_image` decoder 解码为 `uint16` depth，且 `/cam_h/depth/image_raw/compressedDepth` 可通过 `compressedDepth_png` decoder 定位 PNG payload 后解码；若启用 raw `16UC1` fallback，则额外检查对应消息封装和 `height/width/step/is_bigendian/data` 解析正确。
  - [ ] 检查时间轴：episode 目标频率约为 30Hz，RGB/depth/state/action/tactile 时间戳对齐误差在可配置阈值内。
  - [ ] 检查语义：profile 对应 action 重排顺序、头部 state 来自逐帧 `/sensors_data_raw`、头部 action 来自逐帧 `/joint_cmd`；若存在触觉，则检查 normal force 除以 100 后为牛顿量纲。
  - [ ] 输出中文 validation report，列出 pass/fail、异常 episode、缺失字段、shape mismatch 和时间对齐误差。

---

## 阶段二：工具链整合阶段 (Toolchain Integration Phase)

**核心目标**：将 DECO 原生代码库以复制副本方式收编入 Kuavo 工具链生态，统一依赖记录、路径约定与最小侵入策略。

- [x] **2.1 代码库复制归档**
  - [x] 保留现有 `DECO/` 文件夹不变，将其内容复制到 `third_party/deco/`，与现存 `third_party/lerobot/` 并列。
  - [x] 复制阶段不修改 DECO 原始源码，不修改 `third_party/lerobot/`，后续 Kuavo 适配只通过 wrapper 层接入 `third_party/deco/`。
  - [x] 复制时排除 `.DS_Store`，避免 macOS 元数据进入第三方源码副本。
- [x] **2.2 Python 依赖与包路径约定**
  - [x] 采用动态注入法作为后续 wrapper 约定：在 `kuavo_train/wrapper/policy/deco/` 中将 `third_party/deco` 加入 `sys.path`，零侵入兼容 DECO 内部 `from models.xxx` 导入路径。
  - [x] 分析原生 `DECO/requirements.txt`，在根目录 `requirements_DECO.txt` 中记录 DECO 原始依赖与 Kuavo/LeRobot 兼容说明；当前阶段不安装依赖、不执行环境变更。
  - [x] **依赖最小化复查点**：阶段三/四明确实际 wrapper、模型手术、数据转换、validator 与部署入口后，已将 `requirements_DECO.txt` 从说明型记录改为 Linux pip-only requirements；文件只保留 pip 可安装依赖与本仓库 editable 包，ROS Noetic、rosbag、cv_bridge、sensor_msgs、std_msgs、kuavo_msgs 等由系统 ROS 环境提供，避免把不可可靠 pip 安装的 ROS 分发包混入 requirements。
- [x] **2.3 third_party/deco baseline 清理**
  - [x] 删除 `third_party/deco/config/act.yaml` 与 `third_party/deco/config/dp.yaml`，避免 copied DECO 副本继续暴露上游 ACT/DP baseline 配置入口。
  - [x] 删除 `third_party/deco/models/act/` 与 `third_party/deco/models/dp/` 下的 ACT/DP baseline Python 文件，只保留 `third_party/deco/models/deco/` 作为当前 DECO 主体代码路径。
  - [x] 保留原 Kuavo 工具链中的 ACT/DP wrapper、配置与 LeRobot 集成内容不变；本次清理范围仅限 `third_party/deco` 内部 copied baseline 文件。
  - [x] 保留 `third_party/deco/train.py`、`third_party/deco/inference.py`、`third_party/deco/dataset.py`、`third_party/deco/config/deco.yaml` 与 `third_party/deco/models/deco/*`，避免影响原生 DECO 参考链路和 Kuavo-DECO wrapper 链路。
- [x] **2.4 DECO YAML 配置收尾复查**
  - [x] 复查 `configs/data/KuavoRosbag2Lerobot_deco.yaml`，清理容易误导用户的无效配置项，并把固定 schema、固定 RGB-D 路线、depth 存储格式、validator 检查项等信息迁移为注释说明。
  - [x] 复查 `configs/policy/deco_config.yaml`，删除 DECO 训练入口不消费的 `ema_power` 字段和训练层重复 scheduler 字段，保留 policy 层 scheduler 作为唯一学习率调度配置入口。
  - [x] 复查 `configs/deploy/kuavo_deco_env.yaml`，删除 DECO 部署路径不使用的 eef/base limits 与原 ACT/DP `arm_state_keys` 字段，并补充 eef/profile/state layout/inference mode/head state source 的可选项说明。
  - [x] 复查 `third_party/deco/config/deco.yaml`，补充原生 DECO 参考配置中 action_dim、chunk_size、tactile/plugin、backbone、inf_step 等字段语义。
  - [x] 对所有保留的说明型字段标注“固定约束/信息字段”等边界，避免用户误以为它们是可切换到另一条链路的开关。
- [x] **2.5 README_DECO 使用指导重构**
  - [x] 将 `README_DECO.md` 从阶段性记录重构为面向用户的 DECO instruction，开头说明 DECO 是后续合并到 Kuavo 跨端工具链中的策略链路。
  - [x] 补充 Linux 主线安装说明，明确 `pip install -r requirements_DECO.txt` 是唯一 DECO pip requirements 入口，ROS Noetic 与 Kuavo 消息环境仍由系统/ROS workspace 提供。
  - [x] 补充数据清洗说明，明确从仓库根目录运行 `kuavo_data/CvtRosbag2Lerobot_DECO.py`，配置文件为 `configs/data/KuavoRosbag2Lerobot_deco.yaml`，并列明 `eef_type`、profile、depth encoding、tactile、overwrite 等需要人工调整的参数。
  - [x] 补充 DECO 两阶段训练说明，明确 `visual_main` 与 `tactile_adapter` 是两次独立启动，强脑灵巧手可选第二阶段 tactile PI_Adapter，二夹爪 profile 必须关闭 tactile/LoRA。
  - [x] 补充部署说明，明确 `configs/deploy/kuavo_deco_env.yaml` 是 DECO 唯一部署入口，并写清 `qiangnao_no_tactile`、`qiangnao_tactile`、`gripper_no_tactile` 三种部署模式、run-root 权重资产和 server/client pre/postprocessor 归属。

---

## 阶段三：模型适配阶段 (Model Surgery & Visual Frontend Phase)

**核心目标**：保留 DECO action-token Flow Matching 主干，替换其原生视觉前端，并完成 Kuavo 30 维触觉适配。

> [!WARNING]
> **预训练权重兼容性断言**：  
> 触觉分支与视觉前端都会改变部分参数形状或语义。DECO 官方基于 Inspire Hand 触觉或双 RGB 视觉假设训练出的完整权重不能直接严格加载。可优先复用 ImageNet ResNet backbone 权重；DECO 主干权重是否能部分加载，需要后续按参数名和 shape 做静态筛选。

- [x] **3.1 Kuavo RGB-D 视觉前端移植**
  - [x] 新建或改造 DECO 视觉编码模块，替换原生双路视觉编码入口。
  - [x] 默认使用 `vision_backbone: resnet34`，并允许配置切换 `resnet18`。
  - [x] RGB backbone 输出 layer4 feature map，并投影到 DECO hidden dim。
  - [x] depth backbone 使用 1-channel ResNet；第一层卷积权重使用 RGB backbone conv1 权重按通道平均初始化。
  - [x] RGB 与 depth 使用独立 backbone，不共享同一个 ResNet；二者只在 ResNet 后的 token 层做 cross-modal fusion。
  - [x] 复用 ACT 风格 RGB-depth cross attention fusion，并保留两路视觉 token：`fused_rgb_tokens` 与 `fused_depth_tokens`。
  - [x] 将 DECO 原生两路 token 语义改写为 `fused_rgb/fused_depth`；`MMAttention` 中基于 `total_img_len / 2` 的视觉分流逻辑仍可保留，但必须用中文注释说明新语义。
  - [x] 清理 `third_party/deco/models/deco/deco.py`：移除旧双 RGB 兼容分支和相关模型主体命名，forward 接口改为 `forward(rgb, depth, ...)`，使模型主体仅保留 Kuavo RGB-D 路线。
  - [x] 保留空间 token 形式，而不是过早压成单个全局向量，以匹配 DECO 原本 image tokens 与 action tokens 联合 attention 的结构。
  - [x] 第一版暂不采用单路 `visual_tokens: [B, L, D]` 方案；该方案作为后续 ablation 或二期重构候选。
- [x] **3.2 触觉编码器手术 (Tactile Encoder Surgery)**
  - [x] 删除或绕过原版 `init_tac_regions` 的 1062 维 Inspire Hand 区域均值逻辑。
  - [x] 将触觉输入改为 Kuavo 双手 30 维：左手 15 + 右手 15。
  - [x] 洗数据脚本中 `normal_force / 100` 只作为单位换算，表示把 Kuavo 原始触觉量转换为牛顿；模型入口仍需执行 DECO-style 触觉归一化。
  - [x] 新增或配置 `tactile_left_max`、`tactile_right_max`，其数值应基于已经转换成牛顿的 Kuavo 触觉数据，而不是直接复用 DECO Inspire Hand 原始单位下的 `3486/4050`。
  - [x] 在配置注释中写明 `tactile_left_max` 与 `tactile_right_max` 的含义：分别表示左手 15 维、右手 15 维触觉牛顿值的归一化上限，用于执行 DECO-style `tac / tactile_max`。
  - [x] `tactile_left_max` 与 `tactile_right_max` 允许先填写 `null` 作为待统计占位；但当 `use_tactile: true` 进入正式触觉训练时必须填写正数，可来自训练集统计最大值、分位数上限或人工审定安全上限。
  - [x] 清理 `third_party/deco/config/deco.yaml`：移除旧 DECO `tac_left_max/tac_right_max` 别名，只保留 Kuavo 标准字段 `tactile_left_max/tactile_right_max`。
  - [x] 清理 `third_party/deco/config/deco.yaml`：移除重复的 `data.chunk_size`，以 `model.chunk_size` 作为 DECO action chunk 长度的唯一权威配置。
  - [x] 若 `use_tactile: true` 且这两个 tactile max 仍为 `null` 或非正数，DECO wrapper/config 应显式报错，避免静默用错误尺度训练触觉分支。
  - [x] 避免 `observation.tactile` 被 LeRobot 识别为普通 STATE 后走 `MEAN_STD`；若当前 feature type 无法区分 tactile，则通过 `lerobot_patches/custom_patches.py` 或 wrapper/preprocessor 适配把 tactile 从 STATE 归一化路径中隔离出来。
  - [x] 新增 `gripper_no_tactile` 配置保护：二夹爪 profile 下必须保持 `use_tactile: false`、`use_tactile_lora: false`，并禁止 `training_stage: tactile_adapter`。
  - [x] 将 `self.tactile_encoder` 输入维度从 `1062*2` 改为 `15*2`。
  - [x] 将 tactile gating/fusion 维度从 `68` 调整为 `64`：15 + 15 + 34。
  - [x] 前向传播中直接使用 `tac1` 与 `tac2`，不再做 Inspire Hand 区域切片均值。
- [x] **3.3 DECO 主干保留策略**
  - [x] 保留 `action_encoder`、`action_embedd`、`MMAttention`、`linear` 动作头和 `add_noise` 逻辑。
  - [x] 保留训练目标 `F.mse_loss(out, noise - action)`。
  - [x] 第一版 loss 不使用 `action_is_pad` mask，保持 DECO 原生“mask 返回但 diffusion loss 不消费 mask”的行为。
  - [x] 保留推理阶段 Flow Matching 去噪循环；`inf_step` 仍只表示去噪步数，不表示 10Hz 控制频率。
- [x] **3.4 模型静态验证**
  - [x] 做静态 shape 审查：RGB `(B, 3, H, W)`、depth `(B, 1, H, W)`、state `(B, 28)`、tactile `(B, 30)`、action `(B, chunk_size, 28)`。
  - [x] 增补 `gripper_no_tactile` shape 审查：RGB `(B, 3, H, W)`、depth `(B, 1, H, W)`、state `(B, 18)`、无 tactile、action `(B, chunk_size, 18)`。
  - [x] 做 visual token 审查：`fused_rgb_tokens` 与 `fused_depth_tokens` 应具有相同空间长度，拼接后为 `[B, 2L, dim]`，以兼容 DECO 原生两路视觉 token 假设。
  - [x] 明确不在当前机器执行 forward 验证；仅通过代码审查、shape 推导和注释记录完成逻辑验证。
- [x] **3.5 触觉 LoRA / Plugin Adapter 保留策略**
  - [x] 保留 DECO 源码中的 `PI_Adapter` 低秩 adapter 思路：`down: dim -> rank`，`up: rank -> out_dim`，并以 residual delta 形式注入 image/action attention 与 MLP 分支。
  - [x] Kuavo 配置层使用 `use_tactile_lora` 命名，wrapper 内部映射到 DECO 原生 `plugin`；使用 `tactile_lora_rank` 映射到 `plugin_rank`。
  - [x] 默认 `tactile_lora_rank: 32`，与 DECO 源码默认 `plugin_rank=32` 对齐。
  - [x] 默认 `freeze_pretrained_main: true`：加载 `base_policy_path` 或 `deco_init_pth_path` 后，冻结 checkpoint 中已有且 shape 匹配的主干参数。
  - [x] 保持新出现的 Kuavo 触觉编码器、tactile cross-attention、PI_Adapter 和必要的 RGB-D bridge 参数可训练。
  - [x] 若指定 `adapter_model_path`，表示加载已经合并保存的 base + adapter 权重，用于部署或继续 adapter finetune。
  - [x] 保存最终 policy 时清空 `base_policy_path`、`adapter_model_path`、`deco_init_pth_path` 并关闭外部初始化读取，保证 `.safetensors` 权重目录迁移后不再依赖第一阶段目录或原生 `.pth`；完整部署归档仍以 run 根目录为单位。
- [x] **3.6 分阶段训练边界**
  - [x] 严格遵循 DECO 两步式训练法，不在第一版训练中同时解决视觉改造与触觉 adapter。
  - [x] 第一阶段 `training_stage: visual_main` 先关闭触觉：`use_tactile: false`、`use_tactile_lora: false`，验证 RGB-D 前端 + state + DECO Flow Matching 主干是否可独立学习。
  - [x] 第一阶段训练 RGB-D bridge、视觉投影层和 DECO 主支干，使视觉主干先完整适配 Kuavo 数据；第一版不额外引入第一阶段冻结开关。
  - [x] 第一阶段训练结束后保存 LeRobot run 目录与 `.safetensors` epoch 权重；若用户不使用触觉，部署时使用第一阶段 run 根目录并选择对应 epoch。
  - [x] 第二阶段 `training_stage: tactile_adapter` 作为第二次独立训练启动：`use_tactile: true`、`use_tactile_lora: true`，通过 `base_policy_path` 加载第一阶段 `.safetensors` policy，冻结主干，微调 tactile adapter。
  - [x] 第二阶段虽然只更新 tactile/PI_Adapter 参数，但 RGB-D visual branch 与 state branch 仍必须参与 forward，作为触觉 adapter 学习动作修正的条件上下文。
  - [x] 若用户不启动第二阶段，则不训练 tactile/PI_Adapter；部署时使用第一阶段 run 根目录 + 选定 `.safetensors` epoch 权重，并保持 `use_tactile: false`、`use_tactile_lora: false`。
  - [x] 该两阶段流程用于避免新视觉前端尚未稳定时，把误差错误归因到触觉 LoRA。
  - [x] `gripper_no_tactile` 仅允许第一阶段 `visual_main` 从零训练或常规续训；由于没有 tactile feature，后续不得进入 tactile LoRA / PI_Adapter 二阶段训练。
- [x] **3.7 预处理与 state 接入边界**
  - [x] 将 `Resize/Letterbox` 实现为 RGB 与 depth 共享的确定性空间预处理，并放在 DECO 专用 preprocessor 中、LeRobot normalizer 之前执行；默认采用 DECO 原生 `256x256 letterbox`，RGB 使用双线性插值与灰色 padding `fill=128`，depth 使用 nearest 插值与独立 padding 值（默认 `0` 或 invalid depth）。
  - [x] 将 `GaussianBlur` 纳入 Kuavo `RGB_Augmenter` 随机增强池；RGB 随机增强应在确定性 `Resize/Letterbox` 之后、normalizer 之前执行。
  - [x] RGB 增强权重默认参考 Kuavo ACT：Identity/Notransform 权重 `3.0`，其他增强权重 `1.0`，默认 `max_num_transforms: 1`，保证一部分样本保持原图。
  - [x] `observation.state` 只由 LeRobot preprocessor 基于 dataset stats 做一次归一化；DECO wrapper 内不得再按 `config/deco.yaml` 手动归一化。
  - [x] 清理 `third_party/deco/config/deco.yaml` 中 DECO 原生 obs/action 手动归一化统计字段；`third_party/deco/inference.py` 仅在旧统计字段显式存在时才执行兼容归一化。
  - [x] 归一化后的 state 仍走 DECO 路线：`obs_encoder(action_dim -> dim) -> time embedding condition -> MMAttention`，不改成 ACT 的 state token 或 VAE encoder 输入。
  - [x] `observation.tactile` 从 batch 中单独取出并切分为左右手，不混入 `observation.state`，且按 DECO-style tactile max 归一化并默认 clamp 到 `[0, 1]` 后进入 tactile encoder。

---

## 阶段四：核心封装阶段 (Core Wrapper Phase)

**核心目标**：在 `kuavo_train/wrapper/policy/deco/` 下构建 LeRobot Policy wrapper，把 Kuavo RGB-D 数据管道与 DECO 主干接起来，同时不修改 Kuavo 通用训练入口的基本范式。

- [x] **4.0 LeRobot feature 与 DECO preprocessor 接入边界**
  - [x] 在 `lerobot_patches/custom_patches.py` 中新增 `FeatureType.TACTILE`，并将 `observation.tactile` 映射为 TACTILE，而不是普通 STATE。
  - [x] 在 DECO config 的 normalization mapping 中显式设置 `TACTILE: IDENTITY`，确保 LeRobot `NormalizerProcessorStep` 不对 tactile 做 `MEAN_STD`。
  - [x] 将 DECO 专用视觉预处理实现放在 `kuavo_train/wrapper/policy/deco/DECOProcessor.py`；`lerobot_patches/` 只做全局 feature/type 兼容，不承载 DECO 模型专属预处理逻辑。
  - [x] DECO 视觉 preprocessor 负责 RGB/depth 同步 `Resize/Letterbox`，并保证执行顺序为 raw RGB-D -> deterministic letterbox -> RGB-only random augmentation -> LeRobot normalizer -> DECO wrapper/model。
  - [x] 阶段一 `use_tactile: false` 时，tactile feature 可保留在 dataset batch 中，但 DECO tactile normalization 和 tactile branch 不启用；阶段二 `use_tactile: true` 时才执行 tactile max normalization。
  - [x] `kuavo_train/train_policy.py` 只做最小策略注册或轻量 processor/factory 接入，不改 epoch loop、optimizer step、dataloader loop、checkpoint 保存主流程。
- [x] **4.1 开发 DECOConfigWrapper**
  - [x] 注册 DECO 专属超参数：`training_stage`、`chunk_size`、`dim`、`num_attn_blocks`、`inf_step`、`use_tactile`、`vision_backbone`、`depth_backbone` 与 `dataset_hz` 等。
  - [x] 默认 `vision_backbone: resnet34`、`depth_backbone: resnet34`；允许切换 `resnet18`。
  - [x] 不再暴露旧视觉模式开关；DECO 主体固定为 RGB/depth 独立 backbone + cross attention + 两路 visual tokens。
  - [x] 注册确定性空间预处理参数：默认 `resize_shape: [256, 256]`、`use_letterbox: true`、`letterbox_fill_rgb: 128/255`、`letterbox_fill_depth: 0`、RGB/depth 插值策略等。
  - [x] 注册触觉 adapter 参数：`use_tactile_lora`、`tactile_lora_rank`、`freeze_pretrained_main`、`base_policy_path`、`deco_init_pth_path`、`adapter_model_path`。
  - [x] 注册 DECO-style 触觉归一化参数：`tactile_left_max`、`tactile_right_max` 或等价配置；其输入单位为洗数据后已经 `/100` 的牛顿值。
  - [x] 注册 `clip_tactile_to_unit: true` 默认开关，使 `tactile / tactile_max` 后默认 clamp 到 `[0, 1]`，与 DECO 原生 inference 行为一致。
  - [x] 在 `DECOConfigWrapper` 中对 tactile max 做配置保护：允许默认 `null` 表示待统计；当 `use_tactile: true` 时必须为正数，否则抛出清晰错误。
  - [x] 若 `training_stage: tactile_adapter`，必须要求 `use_tactile: true`、`use_tactile_lora: true`，并提供 `base_policy_path` 或明确的初始化路径。
  - [x] 在 wrapper 内部将 `use_tactile_lora` 映射到 DECO 原生 `plugin`，将 `tactile_lora_rank` 映射到 `plugin_rank`，避免用户直接面对源码中的 plugin 命名歧义。
  - [x] normalization mapping 兼容 RGB、DEPTH、STATE、ACTION、TACTILE，其中 TACTILE 默认 `IDENTITY`。
- [x] **4.2 开发 DECOPolicyWrapper.forward**
  - [x] 从 LeRobot batch 中读取 `observation.images.head_cam_h`、depth feature、`observation.state`、`observation.tactile`、`action`。
  - [x] 假设 RGB/depth 已由 DECO preprocessor 完成同步 `Resize/Letterbox`，且 RGB 随机增强与 LeRobot normalizer 已在 forward 前完成；`policy.forward()` 内不得再做 raw-pixel 语义的 resize/padding。
  - [x] RGB 随机增强池包含 Kuavo ACT 既有增强与 GaussianBlur；depth 不做 photometric augmentation，只做同步空间变换和 depth 归一化。
  - [x] 直接使用 LeRobot preprocessor 已归一化的 `observation.state`，传入 DECO `obs_encoder`，不得进行 DECO 原生手动二次归一化。
  - [x] 当 `use_tactile: true` 时，将 `observation.tactile` 从 batch 中单独读取，切分为左手 `0:15` 与右手 `15:30`，按 DECO-style 左/右手 tactile max 归一化并按配置 clamp 后得到 `tac1/tac2`。
  - [x] 当 `use_tactile: false` 时，不读取或不使用 tactile branch，未训练的 tactile/PI_Adapter 不参与 forward，也不会影响推理输出。
  - [x] 调用改造后的 DECO 主干并计算 `loss = F.mse_loss(out, noise - action)`，第一版不额外乘 `action_is_pad` mask。
  - [x] 返回标准 `(loss, {"loss": loss.item()})`，兼容 `kuavo_train/train_policy.py`。
- [x] **4.3 开发 DECOPolicyWrapper.select_action**
  - [x] 接收已通过训练一致 preprocessor 处理的 RGB、depth、state、tactile 观测。
  - [x] 调用 DECO Flow Matching 推理得到 30Hz 语义 action chunk。
  - [x] 部署端仅支持 Receding Horizon 与 Temporal Ensembling，并按 checkpoint 的 `dataset_hz` 消费原始 action chunk。
  - [x] 每次 `popleft()` 返回标准 28 维底层控制指令。
  - [x] 支持两类部署权重：`visual_main` 部署关闭 tactile/LoRA，使用第一阶段选定 epoch；`tactile_adapter` 部署开启 tactile/LoRA，使用第二阶段选定 epoch。二者的完整部署资产均按 Kuavo 旧逻辑保留为 run 根目录 + `epoch<epoch>` 权重子目录。

---

## 阶段五：配置集成阶段 (Configuration Integration Phase)

**核心目标**：编写面向用户的 YAML 配置文件，使 DECO 使用 Kuavo 现有 RGB-D 输入、视觉增强、训练入口和部署参数。

- [x] **5.1 编写 `configs/policy/deco_config.yaml`**
  - [x] 沿用 Kuavo Hydra 配置规范：`batch_size`、`max_epoch`、`device`、`RGB_Augmenter`、optimizer/scheduler 等。
  - [x] 复用 ACT/DP 的 `RGB_Augmenter` 配置，并新增 DECO blur 候选：Identity/Notransform、ColorJitter、SharpnessJitter、RandomMask、RandomBorderCutout、GaussianNoise、GammaCorrection、GaussianBlur。
  - [x] 显式配置增强权重：Identity/Notransform 默认 `3.0`，其他增强默认 `1.0`，`max_num_transforms: 1`，保证部分训练样本保持原始 RGB 分布。
  - [x] 显式配置确定性空间预处理：默认 `resize_shape: [256, 256]`、`use_letterbox: true`、RGB 双线性插值、depth nearest 插值、`letterbox_fill_rgb: 128/255`、`letterbox_fill_depth: 0`；该部分不作为随机增强池的一员，并由 DECO 专用 preprocessor 在 normalizer 前执行。
  - [x] 显式配置 depth，并说明 depth 不做 RGB photometric augmentation。
  - [x] 显式记录第一版 depth 输入语义：当前 depth 来自 3-channel uint8 兼容存储，wrapper/DECO 主干取单通道后进入 1-channel depth backbone；该输入不是严格 metric depth。
  - [x] 在配置注释中保留后续备选策略说明：若要保留绝对毫米尺度，应升级为 `uint16`/单通道 metric depth feature 或 raw depth 保存方案，并同步更新 converter 与 validator。
  - [x] 不再额外暴露视觉 token 模式开关；DECO 主体固定为 RGB/depth 独立 backbone、cross attention 后仍保留两路 visual tokens。
  - [x] 注入 DECO 参数：`policy_name: deco`、`training_stage: visual_main`、`chunk_size`、`dim`、`num_attn_blocks`、`inf_step`。
  - [x] 默认配置面向第一阶段主干训练：`use_tactile: false`、`use_tactile_lora: false`、`tactile_left_max: null`、`tactile_right_max: null`；若用户完全不使用触觉，第一阶段 run 根目录 + 选定 epoch 权重可直接部署。
  - [x] 为第二阶段触觉 adapter 训练提供清晰 override 注释：`training_stage: tactile_adapter`、`use_tactile: true`、`use_tactile_lora: true`、`base_policy_path: /path/to/stage1_policy`、`freeze_pretrained_main: true`。
  - [x] 默认 `vision_backbone: resnet34`、`depth_backbone: resnet34`，保留 `resnet18` 作为低延迟备选。
  - [x] 数据频率由 dataset metadata 注入，部署 `env.ros_rate` 必须与 checkpoint 的 `dataset_hz` 一致。
  - [x] 显式说明 `observation.state` 使用 LeRobot stats 归一化，但模型接入方式遵循 DECO 的 `obs_encoder + time embedding`，不使用 ACT state token / VAE 路线。
  - [x] 显式说明 `observation.tactile` 不跟随 STATE `MEAN_STD`；触觉以洗数据后牛顿值为输入，先作为 `TACTILE: IDENTITY` 通过 LeRobot preprocessor，再按 DECO-style `tactile_left_max` / `tactile_right_max` 做归一化并默认 clamp 到 `[0, 1]` 后进入 tactile encoder。
  - [x] 对 `tactile_left_max`、`tactile_right_max` 写中文注释：可临时填 `null` 表示待统计；正式触觉训练时应填写训练集统计最大值、分位数上限或人工审定上限，单位为牛顿，且必须为正数。
  - [x] 显式配置 loss 行为：`loss = F.mse_loss(out, noise - action)`，第一版不使用 `action_is_pad` mask。
  - [x] 显式配置触觉 adapter：`use_tactile_lora`、`tactile_lora_rank: 32`、`freeze_pretrained_main: true`、`clip_tactile_to_unit: true`。
  - [x] 显式配置权重入口：`load_external_init_weights: true`、`base_policy_path: null`、`adapter_model_path: null`、`deco_init_pth_path: null`，并注释说明 `base_policy_path/adapter_model_path` 面向 Kuavo `.safetensors` policy 目录，`deco_init_pth_path` 仅用于兼容 DECO 原生 `.pth` 初始化，最终保存时外部路径会被清空。
  - [x] 不额外暴露第一阶段视觉冻结开关，避免与 `freeze_pretrained_main` 和两阶段加载冻结语义重复。
  - [x] 新增 `gripper_no_tactile` 训练 override 示例：`action_dim: 18`、`training_stage: visual_main`、`use_tactile: false`、`use_tactile_lora: false`、`load_external_init_weights: false`。
  - [x] 在 `DECOConfigWrapper` 中校验 dataset profile 与 policy `action_dim/use_tactile/training_stage` 一致，避免 18D gripper 数据误配 28D tactile 配置。
- [x] **5.2 编写 `configs/data/KuavoRosbag2Lerobot_deco.yaml`**
  - [x] 默认 `train_hz: 30`、`use_depth: true`。
  - [x] 新增 `deco.end_effector_profile`，允许 `auto`、`qiangnao_tactile` 与 `gripper_no_tactile`；显式 profile 会要求其与 `dataset.eef_type` 一致。
  - [x] 默认 `rgb_topic: /cam_h/color/image_raw/compressed`。
  - [x] 默认 `depth_topic: /cam_h/depth/image_raw/compressed`、`depth_encoding: compressed_image`，并在转换脚本中兼容 `/cam_h/depth/image_raw/compressedDepth` + `compressedDepth_png`。
  - [x] 注释说明 `/camera/depth/image_rect_raw` 目前只作为 raw fallback 候选 topic，不能在未经 Inspector/validator 复核时替代当前实际数据配置。
  - [x] 注释说明原始采集流可能为 100Hz+，转换脚本必须以目标时间轴下采样，而不是假设固定整数跳帧。
  - [x] 注释说明部署控制 10Hz 不在洗数据阶段处理，而在 wrapper/deploy action queue 阶段处理。
  - [x] 将 validation 配置从固定 `expected_state_dim/action_dim/tactile_dim/require_tactile` 改为随 profile 设置：`qiangnao_tactile` 为 28D + 可选 30D tactile，`gripper_no_tactile` 为 18D + no tactile。

---

## 阶段六：部署与演示阶段 (Deployment & Demonstration Phase)

**核心目标**：验证 `.safetensors` 模型资产包能在 Kuavo 部署体系中以 10Hz 控制频率稳定输出 profile 对应动作。阶段六第一轮优先打通本地单进程推理闭环（`real_single_test.py` / `sim_auto_test.py`），server/client 推理放到第二轮；部署范围同时覆盖三种 DECO 推理模式：`qiangnao_tactile`、`qiangnao_no_tactile`、`gripper_no_tactile`。

- [x] **6.1 适配 `kuavo_deploy` 节点**
  - [x] 静态接入 `deco` / `DECO` policy 类型，确保部署脚本会调用 `CustomDECOPolicyWrapper.from_pretrained()` 读取 `.safetensors` 和 `config.json`。
  - [x] 在部署入口导入 `DECOProcessor.py`，确保加载保存的 `policy_preprocessor.json` 时能找到 `deco_rgbd_letterbox_processor`。
  - [x] 新建 `configs/deploy/kuavo_deco_env.yaml`，将 DECO 部署配置从通用 `kuavo_env.yaml` 中拆出，并把 depth topic 固定为当前数据规划的 `/cam_h/depth/image_raw/compressed`。
  - [x] 将 `configs/deploy/kuavo_env.yaml` 还原为 ACT/DP 通用部署配置，不在该文件中承载 DECO 专用 depth/tactile 语义。
  - [x] 文档明确部署资产采用原 Kuavo 三层 run 路径：`outputs/train/<task>/<method>/<timestamp>/`，`epoch` 字段只选择 `epoch<epoch>` 权重子目录；`epochbest/` 单独不是完整部署包。
  - [x] 将 `configs/deploy/kuavo_deco_env.yaml` 扩展为 DECO 唯一部署入口，明确三种 `deco.inference_mode`：
    - `qiangnao_tactile`：28D 灵巧手 + 30D tactile，要求加载已保存 `use_tactile=true`、`use_tactile_lora=true` 的二阶段 tactile adapter checkpoint。
    - `qiangnao_no_tactile`：28D 灵巧手，不订阅、不输入 `observation.tactile`，要求加载已保存 `use_tactile=false`、`use_tactile_lora=false` 的视觉主干 checkpoint。
    - `gripper_no_tactile`：18D 二爪夹，不订阅、不输入 `observation.tactile`，支持 `eef_type=leju_claw` 与 `eef_type=rq2f85`，二者共享 18D state/action schema，仅 topic 与下发缩放不同。
  - [x] 在 `configs/deploy/kuavo_deco_env.yaml` 中新增并注释 `head_state_source`：
    - `live_joint_q`：从 `/sensors_data_raw.joint_data.joint_q[26:28]` 实时读取头部 yaw/pitch，更贴近机器人当前状态，但可能带入传感器微小抖动。
    - `fixed_config`：使用 `head_init` 作为固定头部 state，更稳定，适合部署期间头部固定的任务，但必须确认与训练数据头部姿态一致。
  - [x] `kuavo_deploy/config.py` 支持 `policy_type: deco`、`deco.inference_mode`、`state_layout: deco_28d/deco_18d`、`head_state_source` 与 `depth_encoding: compressed_image/compressedDepth_png`。
  - [x] `ObsBuffer` 静态接入 DECO 默认 depth topic `/cam_h/depth/image_raw/compressed`，并通过 `depth_encoding: compressed_image` 区分普通 compressed image 与旧 `compressedDepth_png`。
  - [x] `ObsBuffer` 仅在 `deco.inference_mode=qiangnao_tactile` 时静态接入 `/dexhand/touch_state`，按左手 15 + 右手 15 的 normal force 顺序构造 30D `observation.tactile`，并保持 `/100` 牛顿换算。
  - [x] 新增部署侧 DECO 映射 helper（建议 `kuavo_deploy/utils/deco_obs_action.py`），集中封装 `build_deco_28d_state`、`build_deco_18d_state`、`decode_deco_28d_action`、`decode_deco_18d_action`，避免把 DECO schema 细节散落在环境类中。
  - [x] `ConfigEnv` 与 `KuavoBaseRosEnv` 新增 `state_layout: deco_28d`，在线拼接 `left arm 7 + left hand 6 + right arm 7 + right hand 6 + head 2` 的 28D `observation.state`；该 layout 同时支持 `qiangnao_tactile` 与 `qiangnao_no_tactile`。
  - [x] `ConfigEnv` 与 `KuavoBaseRosEnv` 新增 `state_layout: deco_18d`，在线拼接 `left arm 7 + left gripper 1 + right arm 7 + right gripper 1 + head 2` 的 18D `observation.state`；该 layout 支持 `leju_claw` 与 `rq2f85`。
  - [x] `KuavoBaseRosEnv.step()` 静态接入 DECO 28D action 解释与下发：双臂 14D 下发到 arm，双手 12D 还原到 0-100 dexhand 指令，head action 当前保留维度但不下发。
  - [x] `KuavoBaseRosEnv.step()` 静态接入 DECO 18D action 解释与下发：双臂 14D 下发到 arm，左右夹爪 2D 按当前工具链既有比例分别下发到 `leju_claw` 或 `rq2f85`，head action 当前保留维度但不下发。
  - [x] 本地单进程推理入口先完成 DECO 闭环：`raw obs -> run-root preprocessor -> CustomDECOPolicyWrapper.select_action -> run-root postprocessor -> env.step()`，优先覆盖 `real_single_test.py` 与 `sim_auto_test.py`。
  - [x] 本地推理入口增加部署配置与 checkpoint config 的一致性校验：`deco.inference_mode`、`state_layout`、`eef_type`、`use_tactile`、`use_tactile_lora`、`action_dim` 必须与已保存 `config.json` 的结构语义一致；部署配置只做选择与校验，不强行覆盖 checkpoint 的模型结构字段。
  - [x] server/client 采用 Kuavo ACT 原版语义完成静态计划冻结：eval/client 侧负责 `run-root preprocessor -> PolicyClient.select_action -> run-root postprocessor`，server 只负责加载 policy 并对已经预处理的 observation 调用 `policy.select_action()`，返回尚未 postprocess 的模型 action，避免 client/server 双重归一化。
  - [x] `kuavo_deploy/kuavo_service/server.py` 支持通过启动参数或 `KUAVO_DEPLOY_CONFIG` 环境变量选择部署配置，避免服务端只能读取旧的硬编码配置。
  - [x] `kuavo_deploy/kuavo_service/server.py` 按 `policy_type` 静态支持 `act`、`diffusion`、`deco` 三类本地 policy 加载；`deco` 路径加载 `CustomDECOPolicyWrapper` 并复用 checkpoint/config 一致性校验。
  - [x] `kuavo_deploy/kuavo_service/client.py` 保持 ACT 原版 `PolicyClient.select_action(obs_dict)` 接口不变，仅补充 timeout、api_token 与 server error 处理；不在 client 内部引入 preprocessor/postprocessor。
  - [x] 部署侧观测已静态接入与训练字段一致的 RGB、depth、state，以及仅在 `qiangnao_tactile` 下存在的 tactile；实际 ROS topic、shape 和时序仍需阶段 6.2/6.3 在允许运行的环境中验证。
- [ ] **6.2 数据频率 / 部署控制频率一致性验证**
  - [ ] 检查 `env.ros_rate == checkpoint.dataset_hz`，避免动作时间语义发生变化。
  - [ ] 检查 action chunk 长度是否足够覆盖部署控制队列需求。
  - [ ] 分别检查 `deco_28d` 的头部 action 维度 26-27 与 `deco_18d` 的头部 action 维度 16-17；当前数据来自逐帧 `/joint_cmd`，但部署端仍只保留预测维度而不下发头部控制。
  - [ ] 检查 `head_state_source=live_joint_q/fixed_config` 对在线 state 分布的影响，确认部署输入与训练数据头部姿态语义一致。
- [ ] **6.3 闭环测试验证**
  - [ ] 第一轮关闭触觉进入仿真或真机 dry-run：覆盖 `qiangnao_no_tactile` 与 `gripper_no_tactile`，只验证 RGB-D + state + action 的 DECO 主干闭环。
  - [ ] 仿真好结果标准：无 NaN/Inf、无关节越界、动作输出平滑、左右手/左右臂映射正确、头部预测维度不干扰现有下发路径、action queue 节奏与数据频率一致。
  - [ ] 任务行为好结果标准：末端运动方向符合示教趋势，抓取或接触前动作不过早抖动，成功率和轨迹平滑度至少接近同数据上的 ACT/DP 基线。
  - [ ] RGB-D 感知验证：检查 RGB 与 depth 是否对齐，depth 是否进入正确 backbone，RGB 增强不会错误作用到 depth。
  - [ ] 第二轮开启触觉 LoRA：使用 `qiangnao_tactile` 加载二阶段 tactile adapter checkpoint，再做仿真、离线 replay 或低风险真机验证。
  - [ ] 触觉反馈验证：在仿真或真机低风险场景中施加指尖压力，观察 30 维法向力变化是否能影响 action chunk，而不是被模型忽略。
- [ ] **6.4 实机上线分级检查**
  - [ ] 上实机前先做 dry-run：真实传感器输入，策略输出只记录不下发，检查 10Hz 输出节奏、action 范围和异常峰值。
  - [ ] 低速限幅闭环：开启更严格的关节速度、关节位置和手指开合限幅，确认动作无方向反转和突发尖峰。
  - [ ] 完整闭环：仅在 dry-run 与低速限幅均通过后，再进入正常速度或正常任务范围。
  - [ ] 任何阶段若出现 RGB-depth 时间错位、触觉全零/饱和、action 越界、左右映射错误或控制频率漂移，应回退到上一检查点。

---

## Done When

- [x] `PLANS.md`、`Content/DECO_Technical_Decisions.md`、`README_DECO.md` 对 RGB-D 新架构描述一致。
- [x] `configs/data/KuavoRosbag2Lerobot_deco.yaml` 准确记录 30Hz 数据采样、RGB-D 保存和触觉/state/action 映射。
- [x] `configs/data/KuavoRosbag2Lerobot_deco.yaml` 明确当前默认 depth topic 为 `/cam_h/depth/image_raw/compressed`，默认 decoder 为 `compressed_image`；`kuavo_data/CvtRosbag2Lerobot_DECO.py` 已兼容 `/cam_h/depth/image_raw/compressedDepth` + `compressedDepth_png`，raw `16UC1` depth 仍仅作为可配置 fallback。
- [x] `configs/policy/deco_config.yaml` 准确记录 ResNet34 默认 backbone、RGB-D 前端、视觉增强、Flow Matching 主干、30Hz/10Hz 解耦参数。
- [x] `configs/policy/deco_config.yaml` 准确记录默认 `256x256 letterbox`、RGB 灰色 padding `128`、depth 独立 padding、确定性 `Resize/Letterbox` 与随机 `RGB_Augmenter` 的边界，并包含 GaussianBlur 及其采样权重。
- [x] `configs/policy/deco_config.yaml` 准确记录第一版 depth 使用 3-channel uint8 兼容存储并由 wrapper 取单通道的事实，同时说明 metric depth / raw depth 是后续备选升级路线。
- [x] DECO wrapper 完成静态审查：RGB/depth 使用独立 backbone，cross attention 后仍以 `fused_rgb/fused_depth` 两路 visual tokens 接入 DECO MMAttention。
- [x] DECO wrapper 完成静态审查：`observation.state` 只经 LeRobot stats 归一化一次，并按 DECO `obs_encoder + time embedding` 路线接入 MMAttention。
- [x] `lerobot_patches/custom_patches.py` 完成静态审查：`observation.tactile` 被识别为 `FeatureType.TACTILE`，并在 DECO normalization mapping 中使用 `IDENTITY`。
- [x] DECO wrapper 完成静态审查：`observation.tactile` 不走 STATE `MEAN_STD`，而是在 `/100` 牛顿单位基础上执行 DECO-style 左/右手 tactile max 归一化并默认 clamp 到 `[0, 1]` 后进入 tactile encoder。
- [x] DECO wrapper 完成静态审查：`use_tactile: true` 时 `tactile_left_max` 与 `tactile_right_max` 必须为正数；`null` 只允许作为配置占位，不允许进入正式触觉训练。
- [x] DECO wrapper 完成静态审查：Flow Matching loss 严格使用 `F.mse_loss(out, noise - action)`，第一版不额外使用 `action_is_pad` mask。
- [x] DECO 专用 preprocessor 完成静态审查：RGB/depth 同步 `256x256 letterbox` 在 normalizer 前执行，RGB 随机增强在 letterbox 后、normalizer 前执行，depth 不做 photometric augmentation。
- [x] DECO 部署路径语义完成文档同步：完整部署资产是 `outputs/train/<task>/<method>/<timestamp>/` run 根目录，`epochbest/` 或任意 `epoch<epoch>/` 只是权重子目录，processor 仍从 run 根目录读取。
- [x] `configs/deploy/kuavo_deco_env.yaml` 已新增，`configs/deploy/kuavo_env.yaml` 已还原为通用 ACT/DP 配置。
- [x] `configs/data/KuavoRosbag2Lerobot_deco.yaml` 与 `kuavo_data/CvtRosbag2Lerobot_DECO.py` 支持 `qiangnao_tactile` 与 `gripper_no_tactile` 两类 end-effector profile。
- [x] `gripper_no_tactile` 完成静态审查：`leju_claw` 与 `rq2f85` 在数据清洗入口 topic/尺度不同，但进入 DECO 后共享 18D state/action schema，且不写入、不要求、不使用 `observation.tactile`。
- [x] DECO 本地在线部署 obs/action 完成静态接入：`deco_28d` / `deco_18d` state/action、`compressed_image` depth decoder、按模式启用的 30D tactile callback、本地 run-root pre/post processor 与 checkpoint config 一致性校验已在代码中连通。
- [x] DECO server/client 部署链路完成静态接入：server 可通过启动参数或 `KUAVO_DEPLOY_CONFIG` 选择 DECO 配置，并按 Kuavo ACT 原版语义明确 pre/post processor 归属在 eval/client 调用侧，server 只执行已预处理 observation 到未 postprocess action 的 policy 推理。
- [x] `configs/policy/deco_config.yaml` 准确记录 `training_stage`、`use_tactile_lora`、`tactile_lora_rank`、`freeze_pretrained_main`、`base_policy_path`、`adapter_model_path`、`deco_init_pth_path` 等两阶段训练和 tactile adapter 参数。
- [x] 两阶段训练逻辑完成静态审查：`visual_main` 与 `tactile_adapter` 是两次独立启动；若关闭触觉，第一阶段 run 根目录 + 选定 epoch 权重可直接部署；若开启触觉，第二阶段加载第一阶段 policy 并冻结主干训练 tactile/PI_Adapter。
- [x] DECO 最终保存的 policy 权重目录完成静态审查：`config.json` 不保留外部初始化路径，迁移时不再依赖第一阶段目录或原生 `.pth` 文件；完整部署包仍必须保留 run 根目录中的 processor 文件。
- [x] `kuavo_data/CvtRosbag2Lerobot_DECO.py` 完成静态审查：不修改公共 reader，不破坏 ACT/DP 数据链路。
- [ ] `kuavo_data/validate_deco_lerobot_dataset.py` 完成静态审查，并能在允许执行的环境中验证单 rosbag 转换结果是否符合 RGB-D、30Hz、profile 对应 action/state 维度、可选 30 维 tactile 方案。
- [x] DECO wrapper 完成静态审查：RGB-D visual frontend 接入 DECO action-token Flow Matching 主干，loss 为 `F.mse_loss(out, noise - action)`。
- [ ] 先完成关闭触觉的仿真验证，再进入触觉 LoRA 验证；通过 dry-run 与低速限幅检查后才进入实机完整闭环。
- [x] `requirements_DECO.txt` 已改为 Linux pip-only requirements，覆盖 DECO 数据转换、训练、validator、本地部署与 server/client 推理服务中 pip 可安装的依赖；ROS/Kuavo 消息环境作为系统前置条件，不再写成 pip 安装条目。
- [x] 阶段三/四完成实际 DECO import 范围确认后，已复查 `requirements_DECO.txt`：保留 Kuavo/LeRobot 当前版本作为主线，不采用 DECO 原生会冲突的 `torch/torchvision/diffusers/huggingface-hub` pin，也不照搬 `requirements_total.txt` 中不可可靠 pip 安装的 ROS 包。
- [x] `third_party/deco` 中上游 ACT/DP baseline 配置与模型文件已删除；原 Kuavo ACT/DP 工具链、`third_party/deco/models/deco/*` 与 Kuavo-DECO wrapper 路线保持不变。
- [x] DECO 相关 YAML 已完成收尾复查：无效字段已清理，固定约束/信息字段已转为明确注释，多选项字段已列明可填值与语义。
- [x] `README_DECO.md` 已重构为完整 DECO 使用指导，覆盖 Linux 安装、数据清洗、两阶段训练、关闭 LoRA 的条件、部署配置和常见误配。
- [x] `AI_Logs.md` 用中文记录每次文档与代码修改。
