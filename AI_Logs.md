# AI Execution Logs

## 2026-05-12

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
