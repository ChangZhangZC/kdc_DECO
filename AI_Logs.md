# AI Execution Logs

## 2026-05-13

### 补充 tactile LoRA 训练策略与分阶段验证检查点
- **任务**: 根据用户确认，继续补充 `PLANS.md` 和 `Content/DECO_Technical_Decisions.md`，将 DECO 源码中的触觉低秩 adapter 机制、配置开关、数据转换验证点和仿真/实机分阶段检查流程纳入当前 Kuavo-DECO 总体方案。
- **源码确认**:
  - `DECO/models/deco/deco.py` 中的触觉 LoRA 实际实现名为 `PI_Adapter`，不是外部 PEFT LoRA；其结构为 `Linear(dim -> rank)` 与 `Linear(rank -> out_dim)` 的低秩 residual adapter。
  - 当 `use_tactile=True` 且 `plugin=True` 时，`MMAttention` 会在 image/action 的 QKV、projection 和 MLP 分支中注入 `PI_Adapter`。
  - 当指定 `pretrain_model_path`、开启 `use_tactile` 和 `plugin`、且未指定 `adapter_model_path` 时，DECO 会加载预训练主干并冻结 checkpoint 中已有参数，只训练新出现的 tactile/adapter 相关参数。
  - `DECO/train.py` 中优化器只接收 `requires_grad=True` 的参数，因此该冻结策略会真实影响训练参数集合。
- **修改内容**:
  - 修改 `PLANS.md`：
    - 将核心架构策略补充为包含 `Tactile Plugin/LoRA 低秩微调保留`。
    - 在 `0.2 已确认技术决策` 中明确保留 DECO 原生 `plugin=True` / `PI_Adapter` 触觉 adapter 范式，并说明它是 DECO 自实现低秩 adapter，不是外部 PEFT LoRA。
    - 在 `0.3 关键解释` 中新增 `use_tactile_lora` 与 `freeze_pretrained_main` 的含义。
    - 新增 `1.7 单 rosbag 转换试跑与数据一致性检查`，要求转换脚本完成后在允许执行的环境中转换一个 rosbag，并规划新增 `kuavo_data/validate_deco_lerobot_dataset.py` 检查字段、shape、30Hz 时间轴、RGB-depth 对齐、28 维 action 映射和 30 维 tactile 量纲。
    - 新增 `3.5 触觉 LoRA / Plugin Adapter 保留策略`，记录 `use_tactile_lora -> plugin`、`tactile_lora_rank -> plugin_rank` 的配置映射，默认 `tactile_lora_rank: 32`，并保留主干冻结、adapter 微调策略。
    - 新增 `3.6 分阶段训练边界`，明确先关闭触觉验证 RGB-D + DECO 主干，再加载视觉/RGB-D checkpoint 冻结主干并微调 tactile adapter。
    - 在 `4.1 DECOConfigWrapper` 与 `5.1 configs/policy/deco_config.yaml` 中新增 `use_tactile_lora`、`tactile_lora_rank`、`freeze_pretrained_main`、`pretrain_model_path`、`adapter_model_path`、`freeze_vision_backbone` 等配置项。
    - 扩展 `6.3 闭环测试验证` 并新增 `6.4 实机上线分级检查`，要求先关闭触觉进仿真，满足动作平滑、左右映射正确、10Hz 队列稳定等标准后，再开启触觉 LoRA，最后按 dry-run、低速限幅、完整闭环顺序上实机。
    - 更新 `Done When`，加入 tactile adapter 配置、数据集验证脚本、视觉-only 仿真和触觉 LoRA 验证要求。
  - 修改 `Content/DECO_Technical_Decisions.md`：
    - 在文档定位和当前架构中新增 tactile plugin / low-rank adapter 微调决策。
    - 扩展触觉模型手术说明，明确 Kuavo 触觉 token 仍应进入 DECO tactile cross-attention。
    - 新增 `4.6 Tactile Plugin / LoRA-style Adapter 机制`，记录 `PI_Adapter` 结构、注入位置、冻结逻辑和 Kuavo 配置命名。
    - 新增 `4.7 分阶段训练与验证策略`，记录视觉-only 阶段、触觉 adapter 阶段和部署验证阶段的边界与好结果标准。
    - 更新待实现清单，加入 tactile adapter 配置项和 `validate_deco_lerobot_dataset.py`。
- **目的**:
  - 保留导师要求的 DECO 触觉 adapter 微调方式，同时让 Kuavo 配置层具备清晰开关。
  - 避免 RGB-D 前端替换和触觉 LoRA 同时引入时难以定位问题，因此将验证路线拆成视觉-only、触觉 adapter、实机分级上线三段。
  - 在完整数据转换脚本完成后增加独立 validation checkpoint，防止字段、维度、频率、深度图、触觉量纲或 action 映射错误进入训练阶段。

### 重构 Kuavo-DECO 总体计划为 RGB-D 视觉前端方案
- **任务**: 根据用户与导师的新要求，重构 `PLANS.md` 中的 DECO 集成总体架构，重点将视觉策略从“DECO 原生双 RGB / 单目复制 / 左右切分”调整为“Kuavo/ACT 风格 RGB-D 视觉前端 + DECO Action-Token Flow Matching 主干”。
- **修改内容**:
  - 重写 `PLANS.md`：
    - 将标题更新为 `Kuavo-DECO RGB-D 架构迁移与系统集成宏观计划书`。
    - 将核心架构策略更新为 `Wrapper 融入模式 + Kuavo/ACT 风格 RGB-D 视觉前端移植 + DECO Action-Token Flow Matching 主干保留 + 30Hz 数据 / 10Hz 控制解耦`。
    - 新增 `0. 当前冻结的总体架构`，明确整体链路为：Kuavo rosbag RGB/depth/state/action/tactile → 30Hz LeRobot RGB-D 数据集 → Kuavo RGB_Augmenter/Normalizer → RGB ResNet34 + Depth ResNet34 → ACT 风格 RGB-depth cross attention fusion → DECO action-token Flow Matching transformer → 28D action chunk → 10Hz 控制队列。
    - 明确 `vision_backbone: resnet34` 和 `depth_backbone: resnet34` 作为默认配置，同时允许切换到 `resnet18` 以降低推理延迟和显存压力。
    - 将阶段一重构为 `RGB-D 数据引擎阶段`：
      - 默认 `train_hz: 30`、`use_depth: true`。
      - 要求转换脚本基于真实时间戳生成 30Hz 目标时间轴，兼容原始采集流为 100Hz 或更高频率的情况。
      - 明确不再依赖 `MAIN_TIMELINE_FPS // TRAIN_HZ` 的整数跳帧假设。
      - 明确保存 `observation.images.head_cam_h` 与对齐的 depth feature。
    - 将旧的视觉策略替换为新主路线：
      - 放弃 `/cam_h` 单目复制成 DECO `img1/img2`。
      - 放弃 `/cam_h` 左右裁切成伪双目。
      - 新路线为 Kuavo RGB-D 前端替换 DECO 原生 `img_encoding(img1, img2)`。
    - 将阶段三重构为 `模型适配阶段`：
      - 新增 Kuavo RGB-D 视觉前端移植任务。
      - 保留 DECO action token、MMAttention、Flow Matching loss、denoising loop。
      - 保留并细化触觉 30 维手术任务。
    - 将阶段四 wrapper 任务改为读取 RGB、depth、state、tactile 和 action，并在部署阶段用 `action_stride = dataset_hz // control_hz = 3` 将 30Hz 语义动作转换为 10Hz 控制输出。
    - 将阶段五配置任务更新为 `configs/policy/deco_config.yaml` 中显式配置 `dataset_hz: 30`、`control_hz: 10`、`action_stride: 3`。
    - 更新 `Done When`，要求 `PLANS.md`、`Content/DECO_Technical_Decisions.md`、`README_DECO.md` 对 RGB-D 新架构保持一致。
  - 重写 `Content/DECO_Technical_Decisions.md`：
    - 记录当前冻结总体技术路线：Kuavo RGB-D 数据 → 30Hz LeRobot → Kuavo RGB_Augmenter/Normalizer → RGB/Depth ResNet34 → ACT 风格 RGB-depth cross attention fusion → DECO action-token Flow Matching 主干 → 10Hz 控制队列。
    - 明确被替代的旧方案：`use_depth: false`、只保存 `/cam_h` RGB、不保存 depth、单目复制为双目、左右裁切伪双目、默认 10Hz 数据转换、剥离 depth 以保持 DECO 原生假设。
    - 记录 RGB 增强内容：Identity、ColorJitter、SharpnessJitter、RandomMask、RandomBorderCutout、GaussianNoise、GammaCorrection。
    - 记录 depth 策略：只做同步 crop/resize 与 depth 归一化，不做颜色类增强。
    - 记录 ResNet34 默认值的优缺点与 `resnet18` 备选策略。
    - 记录 30Hz 数据与 10Hz 控制的解耦方式：洗数据阶段 `train_hz=30`，部署阶段 `control_hz=10`，默认 `action_stride=3`。
- **目的**:
  - 让计划文档与导师要求一致：保留 Kuavo 工具链中的 RGB-D 输入方式和视觉增强方案。
  - 避免后续实现继续沿用旧的 DECO 双目 RGB 假设或单目复制 fallback。
  - 明确 DECO 的可保留部分是 action-token Flow Matching 主干，而不是原生视觉入口。
  - 将高频采集、30Hz 训练数据、10Hz 控制输出三个时间尺度分层处理，降低数据转换与部署语义混乱的风险。

### 同步头部自由度结论到技术决策文档
- **任务**: 根据用户要求，将 `PLANS.md` 中已经记录的头部自由度 Inspector 结论同步到 `Content/DECO_Technical_Decisions.md`，不修改视觉策略、不修改整体阶段一方案。
- **修改内容**:
  - 修改 `Content/DECO_Technical_Decisions.md` 的最后更新时间为 `2026-05-13`。
  - 更新 `2.5 28 维 state/action 映射`：
    - 明确当前 rosbag 的 `/sensors_data_raw.joint_data.joint_q` 长度为 28，头部索引使用 V4x/V49 方案 `joint_q[26:28]`。
    - 记录当前样本头部固定姿态均值约为 `[-0.001657, 0.433904]` rad，即 `[-0.09494°, 24.86089°]`，并记录 `/robot_head_motion_data` 的 `[0.0, 25.0]` 作为 pitch 约 25° 的交叉验证。
    - 将 `observation.state[26:28]` 的策略改为使用每个 episode 内 `joint_q[26:28]` 的实测固定均值并广播到所有帧，不逐帧写入微小传感器抖动。
    - 保持 `action[26:28]` 固定补 `[0.0, 0.0]`，表示阶段一暂不控制头部。
  - 更新 state 来源表和缺失 topic 策略：
    - 头部 state 来源改为 `/sensors_data_raw.joint_data.joint_q[26:28]` 的 episode 均值。
    - 明确 `observation.state[26:28]` 默认不补零；只有读不到字段、字段异常或该 episode 无法计算均值时，才进入 review/fallback，不在未 review 时直接写成 `[0.0, 0.0]`。
  - 更新 `3.3 当前等待：1.2 Review 与视觉策略冻结`：
    - 增加头部自由度 Inspector 已完成结论。
    - 明确视觉策略仍待后续讨论与冻结，本文档当前不把 `1.2` 整体标记为完成。
- **目的**: 让技术决策文档与 `PLANS.md` 保持一致，避免后续实现者误用旧的“头部 state 直接补零”策略，同时保留后续继续调整视觉策略和阶段一整体方案的空间。

### 记录头部自由度 Inspector 结论
- **任务**: 根据用户运行 `inspect_deco_stage1_schema.py` 后生成的 `kuavo_data/inspect_outputs/inspector_outputs.txt`，将头部两个自由度的最终处理约束同步写入 `PLANS.md`。
- **Inspector 关键结果**:
  - `/sensors_data_raw.joint_data.joint_q` 长度为 28，说明当前 rosbag 使用 V4x/V49 索引方案，头部自由度位于 `joint_q[26:28]`。
  - `joint_q[26:28]` 在当前样本中的均值约为 `[-0.001657, 0.433904]` rad，即 `[-0.09494°, 24.86089°]`。
  - `/robot_head_motion_data` 第一条消息为 `[0.0, 25.0]`，与 `joint_q[26:28]` 中 pitch 约 25° 的结论一致。
  - Inspector 使用严格阈值 `0.001 rad` 时给出 `stable_under_threshold: False`，主要原因是 yaw 的 range 约 `0.00192 rad`。从工程角度判断，该变化约 `0.11°`，更接近传感器微小抖动，而不是头部真实运动。
- **修改内容**:
  - 修改 `PLANS.md` 的 `1.2 Inspector 结果 Review 与视觉策略冻结`：
    - 增加头部自由度 Inspector 结论，明确当前样本头部索引、均值角度、以及 `/robot_head_motion_data` 的交叉验证。
    - 增加 state/action 初步冻结策略：`observation.state[26:28]` 使用每个 episode 内 `joint_q[26:28]` 的实测固定均值并广播到所有帧；不逐帧写入微小传感器抖动；`action[26:28]` 仍固定补 `[0.0, 0.0]`。
  - 修改 `PLANS.md` 的 `1.5 动作空间 (28 维) 索引重组`：
    - 将原先“读不到或不稳定则回退补零”的笼统描述，细化为当前样本使用 episode 固定均值；若后续 rosbag 读不到该字段，或确认头部存在真实运动，则暂停该策略并重新 review。
- **目的**: 避免把真实头部 pitch 约 25° 的固定姿态误写成零点，同时避免把传感器微抖动逐帧写入训练 state。该记录只冻结头部自由度处理方式；整体阶段一方案和视觉策略后续仍可继续讨论与调整。

## 2026-05-12

### 执行阶段一 1.1：新增 Rosbag Schema Inspector
- **任务**: 根据阶段一计划，新增只读 Inspector，用于确认 Kuavo rosbag 的关键 topic schema、头部相机是否为单目/双目、以及头部两个自由度是否存在稳定锁定角度。
- **修改内容**:
  - 新建 `kuavo_data/inspect_deco_stage1_schema.py`：
    - 默认读取 `data_example/vr_record_2026-04-15-15-57-47.bag`，并以只读方式打开 rosbag；未索引时仅尝试 `allow_unindexed=True`，不执行 reindex 写入。
    - 打印用户当前记录 topic 的存在性、消息类型、消息数量、频率、首尾时间，帮助确认 rosbag schema。
    - 解码 `/cam_h/color/image_raw/compressed` 第一帧，并导出 `data_example/inspect_outputs/cam_h_full.jpg`、`cam_h_left_half.jpg`、`cam_h_right_half.jpg`，用于人工判断头部图像是单目整图还是左右拼接双目图。
    - 打印 `/cam_h`、`/cam_l`、`/cam_r` 的 CameraInfo 宽高、内参矩阵和畸变参数，辅助判断相机流语义。
    - 统计 `/sensors_data_raw.joint_data.joint_q[26:28]` 的前若干样本、均值、角度制均值、最小值、最大值、range 和 std，并额外打印 V52 兼容候选 `joint_q[27:29]` 供 review。
    - 检查 `/dexhand/state`、`/dexhand/touch_state`、`/control_robot_hand_position`、`/kuavo_arm_traj`、`/joint_cmd` 等关键字段长度是否满足后续 28 维 state/action 与 30 维 tactile 构造要求。
  - 更新 `PLANS.md`：
    - 将 `1.1 Rosbag Schema Inspector` 及其三个子项标记为已完成。
    - 保持 `1.2 Inspector 结果 Review 与视觉策略冻结` 为未完成，等待用户运行 Inspector 并反馈结果后再冻结。
  - 更新 `Content/DECO_Technical_Decisions.md`：
    - 将 1.1 Inspector 状态改为已创建。
    - 补充建议运行命令与需要反馈的输出章节。
- **目的**: 在进入完整 DECO 数据转换脚本之前，先用可复现的只读检查脚本拿到视觉源和头部关节锁定方式的证据，避免在 schema 未确认时直接硬编码视觉拆分策略或头部补零策略。

### 更新头部自由度 state/action 决策
- **任务**: 根据用户确认，将头部两个自由度的处理策略从“state/action 直接补零”细化为“state 优先读取稳定实测固定角，失败后回退补零；action 暂时补零”。
- **修改内容**:
  - 修改 `PLANS.md` 阶段一技术决策：
    - 在 `1.0 技术决策记录与方案冻结` 中明确头部 state 优先使用 Inspector 验证后的实测固定角度，action 补零。
    - 在 `1.1 Rosbag Schema Inspector` 中补充对 `/sensors_data_raw.joint_data.joint_q[26:28]` 的前几帧、最小值和最大值统计要求。
    - 在 `1.5 动作空间 (28 维) 索引重组` 中明确 `observation.state[26:28]` 的优先级：先用稳定实测固定角，读不到或不稳定时回退补零；`action[26:28]` 当前始终补零。
  - 修改 `Content/DECO_Technical_Decisions.md`：
    - 更新 28 维 state/action 映射表，头部 state 改为 `/sensors_data_raw` 优先、fallback 补零，头部 action 保持补零。
    - 更新缺失 topic 策略，区分 head state 的条件补零和 head action 的固定补零。
    - 更新 Inspector 要求，要求打印头部关节角切片的前若干帧、最小值、最大值和稳定性判断。
- **目的**: 避免把真实固定姿态误写成数学零点，同时保留在 rosbag 字段不可用或不稳定时的鲁棒回退路径，使阶段一数据转换既尊重真实机器人状态，又不破坏 DECO 固定 28 维接口。

### 执行阶段一 1.0：冻结技术决策与更新计划结构
- **任务**: 根据用户确认，将 DECO 阶段一任务正式拆分为技术决策记录、Rosbag Schema Inspector、Inspector 结果 Review、完整转换流程实现几个检查点，并沉淀可持续维护的技术决策文档。
- **修改内容**:
  - 修改 `PLANS.md` 中的阶段一结构：
    - 新增并勾选 `1.0 技术决策记录与方案冻结`，记录本次文档冻结已完成。
    - 新增 `1.1 Rosbag Schema Inspector`，明确后续需要编写只读 inspector，检查 rosbag topic、消息频率、字段长度，并导出 `/cam_h/color/image_raw/compressed` 的完整图和左右半图。
    - 新增 `1.2 Inspector 结果 Review 与视觉策略冻结`，要求在完整转换实现前先根据用户反馈确认 `/cam_h` 是单目整图还是左右拼接图。
    - 新增 `1.3 完整 DECO 数据转换流程实现`，明确后续需要新建 `_deco.yaml` 和 `CvtRosbag2Lerobot_DECO.py`。
    - 将原有触觉、28 维动作、视觉流任务重排为 `1.4`、`1.5`、`1.6`，并将默认 30Hz 目标调整为阶段一当前确认的默认 10Hz，同时保留未来改回 30Hz 的说明。
  - 修改 `PLANS.md` 阶段四 wrapper loss 描述：
    - 将旧的 `F.mse_loss(act, noise)` 修正为 DECO 源码一致的 Flow Matching 目标 `F.mse_loss(out, noise - action)`。
    - 补充 `out` 与 `noise - action` 的语义说明，避免后续 wrapper 实现误用扩散策略的噪声预测 loss。
  - 新建 `Content/DECO_Technical_Decisions.md`：
    - 记录阶段一已确认方案：新建 `configs/data/KuavoRosbag2Lerobot_deco.yaml`、默认 `train_hz: 10`、`use_depth: false`、固定 28 维 state/action、头部两维补零、转换阶段不 resize、暂不修改公共 reader。
    - 记录视觉源备选方案：`/cam_h` 单目复制、`/cam_h` 左右切分、以及仅在确认 `/cam_l`/`/cam_r` 为头部左右目时才使用它们。
    - 记录缺失 topic 策略、state/action 来源、触觉解析顺序、时间对齐策略和后续 Inspector 输出要求。
- **目的**: 将此前对话中的阶段一技术细节从聊天上下文沉淀到仓库文档中，形成后续代码实现与 review 的明确依据，减少因上下文压缩或多人协作导致的方案漂移。

### 准备 Kuavo 与 DECO 数据格式对照样本
- **任务**: 根据阶段一数据引擎开发前的 schema 对齐需求，准备用于静态分析和后续 inspector 脚本设计的数据样本。
- **修改内容**:
  - 确认用户已在 `data_example/` 下放置 Kuavo rosbag 示例文件 `vr_record_2026-04-15-15-57-47.bag`，大小约 128MB。
  - 从 Hugging Face 数据集 `BAAI-Humanoid/DECO-50` 下载单个 DECO 原生 episode 包 `task1/data-t1-1/episode_0000.tar.gz`，保存为 `data_example/deco_episode_0000.tar.gz`，文件大小为 270,967,842 bytes。
  - 仅完成文件级下载和存在性确认，未解压数据包，未运行 Python 脚本，也未执行任何数据转换或训练相关代码，符合当前项目的 No-Runtime Execution 约束。
- **目的**: 为后续编写 `CvtRosbag2Lerobot_DECO.py` 前的格式对照做准备。Kuavo rosbag 样本用于确认 ROS topic、消息字段、时间戳和多模态对齐方式；DECO episode 样本用于确认原生 `colors/`、`tactiles/`、`data.pkl`/`data.json` 的目录结构、字段命名、动作/状态维度和触觉张量形态。

## 2026-05-08

### 制定 Kuavo-DECO 架构迁移与系统集成宏观计划书
- **任务**: 经过多轮讨论，在 `PLANS.md` 中全面梳理并敲定了将 DECO 模型集成到 Kuavo 工具链的全阶段（阶段一至阶段六）宏观计划书。
- **修改内容**:
  - 确立了采用 Wrapper 融入模式的架构策略，通过**动态注入**（`sys.path.append`）零侵入解决依赖问题。
  - 详细定义了“模型手术”逻辑（阶段三）：切除 `init_tac_regions`，修改底层网络输入维度以完美匹配 30 维触觉特征，并以 Warning 形式记录了放弃原生触觉预训练权重的妥协。
  - 敲定了核心封装拦截器（阶段四）的逻辑：通过编写 `DECOPolicyWrapper`，接管 `forward`（拦截训练）和 `select_action`（拦截推理），从而严格死守“完全不修改官方 `train.py`”的底线。
  - 确认了复用 Kuavo 视觉增强器（`RGB_Augmenter`）的参数配置方案（阶段五）。
- **目的**: 明确未来多轮迭代的代码编写准则、核心修改点和执行顺序。此计划书现已成为后续开发的唯一真理源 (Single Source of Truth)，防止后续实操偏航。

### 完善指导原则与角色定义
- **任务**: 进一步完善 `AGENTS.md` 中的 SOP、约束条件以及身份角色。
- **修改内容**:
  - **角色增强**: 增加了“教学导师 (Instructional Mentor)”身份，负责在对话中讲解算法知识、具身智能（Embodied AI）和 VLA 模型。
  - **SOP 扩展**: 新增了第 5 步“任务后总结 (Post-Task Synthesis)”，要求在任务完成后详细解释逻辑和代码变动。
  - **约束增强**: 新增了“双语文档 (Bilingual Documentation)”要求，规定所有代码修改必须附带详尽的中文批注。
- **目的**: 提升协作的透明度，支持用户的学习需求，并确保代码的可维护性和可评审性。

### 更新 AGENTS.md 指导原则
- **任务**: 在 `AGENTS.md` 的 SOP 章节中补充关于使用 Skills 的指导。
- **修改内容**:
  - 在 `1. Context Discovery` 下新增了 `Skill Empowerment` 子项。
  - 明确要求在执行任务时，主动识别并利用专业技能（如 `ros2-development`, `deep-learning-pytorch`, `robot-perception` 等）以确保实现符合领域最佳实践。
- **目的**: 强化对现有专业技能库的利用，提升代码质量和系统集成的专业性。

## 2026-05-07

### 初始化 DECO 依赖和相关说明文档
- **任务**: 初始化 DECO 依赖和相关说明文档。
  - **修改内容**:
    - 创建了带有标题和简要描述的 `requirements_DECO.txt`，用于记录 DECO 模型专属的依赖项。
    - 创建了带有标题和简要描述的 `README_DECO.md`，用于记录有关新增的 DECO 模型功能的运行和使用说明。
  - **目的**: 满足 `AGENTS.md` 中 `Done When` (完成标准) 部分的预置条件，并为即将进行的整合步骤做准备。

### 分析与文档整理
- **任务**: 分析 DECO 集成的关键技术点，并检查原始 DECO 的视频增强机制。
- **修改内容**:
  - 创建了 `Content/DECO_Integration_Notes.md` 技术备忘录文件。
  - 在文件中总结了对齐30Hz、数据预处理（发现 DECO 使用 ColorJitter 和 GaussianBlur 做光线与降噪鲁棒性训练）、模型转换逻辑、训练约束（60-70M 参数验证、train.py 不可动原则）。
- **目的**: 对导师会议记录进行代码级背书与技术确认，并为后续编写 Wrapper 指明实现路径。

### 架构梳理与链路剖析
- **任务**: 分析整个 kuavo_data_challenge 的数据处理到模型训练与部署的全局链路。
- **修改内容**:
  - 创建了 `Content/Kuavo_Toolchain_Overview.md` 架构说明文件。
  - 详细梳理了 `kuavo_data` (数据清洗转换)、`third_party/lerobot` (核心引擎)、`wrapper` (适配器机制) 以及 `kuavo_train/kuavo_deploy` (训练部署闭环) 的各自职责与联动关系。
- **目的**: 帮助用户在深入 DECO 模型代码前，建立对整个机器人学习平台底座的全局宏观认知，明确 DECO 未来各个模块的嵌入点。
