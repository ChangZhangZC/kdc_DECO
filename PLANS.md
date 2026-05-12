# Kuavo-DECO 架构迁移与系统集成宏观计划书

> **生成日期**：2026-05-08
> **核心架构策略**：**方案 A (Wrapper 融入模式)** + 局部模型截肢手术
> **使用说明**：本计划书为 Kuavo-DECO 集成的唯一“真理源 (Single Source of Truth)”。后续 AI Agent 应在每次会话开始时读取此文件与 `AI_Logs.md`，并在完成任务后更新 Checkbox 状态。

---

## 阶段一：数据引擎阶段 (Data Engine Phase)
**核心目标**：保持 `CvtRosbag2Lerobot.py`中的基本大逻辑，顺序和一些辅助函数，再此基础上进行修改，完成物理语义向数学张量的精确转换，将 rosbag 统一转存为 LeRobot Parquet 格式，将新的脚本保存在 `kuavo_data/` 文件夹下，命名为 `CvtRosbag2Lerobot_DECO.py`，并对代码进行详细的批注和备注。

- [x] **1.0 技术决策记录与方案冻结**
  - [x] 新建 `Content/DECO_Technical_Decisions.md`，作为 DECO 集成技术决策记录文件，后续阶段二、阶段三的具体技术选型也继续追加到该文件。
  - [x] 记录阶段一已确认方案与备选方案：新建 `_deco.yaml` 数据配置、默认 `train_hz: 10`、`use_depth: false`、输出固定 28 维、头部 state 优先使用 Inspector 验证后的实测固定角度且 action 补零、转换阶段不 resize、暂不修改公共 reader、缺失 topic 策略与视觉源候选。
  - [x] 将 Inspector 作为正式检查点：先通过只读脚本确认 rosbag schema、`/cam_h/color/image_raw/compressed` 是否为单目整图或双目拼接，再冻结最终视觉策略。
- [ ] **1.1 Rosbag Schema Inspector**
  - [ ] 新建只读脚本 `kuavo_data/inspect_deco_stage1_schema.py`，用于检查 `data_example/vr_record_2026-04-15-15-57-47.bag` 中关键 topic 的存在性、消息数量、时间范围、估算频率和字段结构。
  - [ ] 对 `/cam_h/color/image_raw/compressed` 解码第一帧，打印 `height/width/aspect_ratio`，并导出 `cam_h_full.jpg`、`cam_h_left_half.jpg`、`cam_h_right_half.jpg` 三张图片供人工判断单目/双目。
  - [ ] 检查 `/dexhand/touch_state`、`/dexhand/state`、`/sensors_data_raw`、`/kuavo_arm_traj`、`/joint_cmd`、`/control_robot_hand_position` 等 topic 的字段长度是否满足 DECO 28 维 state/action 与 30 维 tactile 的构造要求；同时统计 `/sensors_data_raw.joint_data.joint_q[26:28]` 的前几帧、最小值和最大值，用于判断头部锁定角是否稳定可用。
- [ ] **1.2 Inspector 结果 Review 与视觉策略冻结**
  - [ ] 根据 Inspector 输出和导出的三张图，确认最终视觉输入策略：`/cam_h` 单目复制、`/cam_h` 左右切分，或改用其他真实头部双目 topic。
  - [ ] 将最终视觉策略补写进 `Content/DECO_Technical_Decisions.md`，并同步更新 `README_DECO.md` 的数据转换说明。
- [ ] **1.3 完整 DECO 数据转换流程实现**
  - [ ] 新建 `configs/data/KuavoRosbag2Lerobot_deco.yaml`，默认 `train_hz: 10`、`use_depth: false`，并在注释中标明后续从 10Hz 调整到 30Hz 的位置和影响。
  - [ ] 新建 `kuavo_data/CvtRosbag2Lerobot_DECO.py`，实现 DECO 专用 rosbag reader、最近邻时间对齐、固定 28 维 state/action 映射、30 维触觉提取和 LeRobot dataset 写入。
  - [ ] 转换阶段不做图像 resize；图像尺寸从第一帧自动推断并注册到 LeRobot features，后续在 wrapper/训练预处理阶段统一转换到 DECO 所需的 256×256。
  - [ ] 暂不修改 `kuavo_data/common/kuavo_dataset.py` 公共 reader，避免影响 ACT/DP 既有转换链路。
- [ ] **1.4 触觉频率与量纲降维**
  - [ ] 提取 Kuavo 话题 `/dexhand/touch_state`。
  - [ ] 将 Kuavo 原生约 100Hz 的触觉频率**下采样 (Downsampling)** 至数据配置中的 `train_hz`。阶段一默认使用 10Hz；后续如硬件与数据稳定支持，可在 `_deco.yaml` 中调整为 30Hz。
  - [ ] 仅提取指尖/指腹的法向力 (`normal_force`)，舍弃切向力和接近觉。
  - [ ] 保留空间特征（5 指 × 3 点 = 单手 15 维，双手共 30 维），并直接除以 100 缩放至 0~25 牛顿物理量程，保留原始物理意义，后续的 Mean-Std 归一化留给 `config` 阶段。
- [ ] **1.5 动作空间 (28 维) 索引重组**
  - [ ] 建立 `KuavoDecoMapper` 重映射逻辑。
  - [ ] 接收 Kuavo 原生 28 维排列：`臂(0-6, 7-13) -> 手(14-19, 20-25) -> 头(26-27)`。
  - [ ] 输出 DECO 强制 28 维排列：`左侧全集(左臂 0-6, 左手 7-12) -> 右侧全集(右臂 13-19, 右手 20-25) -> 头(26-27)`。
  - [ ] 头部维度保留索引 26-27。`observation.state[26:28]` 优先使用 Inspector 验证为稳定的 `/sensors_data_raw.joint_data.joint_q[26:28]` 实测固定角度；若读不到或不稳定，则回退补零。`action[26:28]` 当前始终补零，表示 DECO 策略暂不控制头部。
- [ ] **1.6 视觉流预处理剥离**
  - [ ] 仅提取头部 RGB 视频流。若 Inspector 确认 `/cam_h/color/image_raw/compressed` 是单目整图，则数据集只保存单路 `head_cam_h`，后续 wrapper 将其复制为 DECO 的 `img1/img2`；若 Inspector 确认其为左右拼接双目，则在转换脚本中切分为 `head_cam_left/head_cam_right`。
  - [ ] 剥离并忽略深度流 (Depth) 记录，严格保持 DECO 原生假设。
  - [ ] 图像频率与 `_deco.yaml` 中的 `train_hz` 对齐。阶段一默认 10Hz，后续可根据数据质量和硬件能力调整为 30Hz。

---

## 阶段二：工具链整合阶段 (Toolchain Integration Phase)
**核心目标**：将 DECO 原生代码库收编入 Kuavo 工具链生态，统一依赖与路径管理。

- [ ] **2.1 代码库物理迁移**
  - [ ] 将现有的 `DECO/` 文件夹整体移入 `third_party/deco/`，与现存的 ACT、DP 基线并列。
- [ ] **2.2 Python 依赖与包路径修复**
  - [ ] 采用**动态注入法**：在后续包裹层 `DECOModelWrapper.py` 的顶部增加 `sys.path.append("third_party/deco")`，零侵入地修复所有从 `models.xxx` 导入的路径错误，不修改原包逻辑。
  - [ ] 分析原生 `DECO/requirements.txt`，分离出 Kuavo 环境缺少的增量依赖（如 `timm`, `einops`），更新并维护根目录的 `requirements_DECO.txt`，确保环境无缝拉起。

---

## 阶段三：硬件适配与模型验证阶段 (Hardware Adaptation & Model Surgery Phase)
**核心目标**：对 DECO 模型源码（`deco.py` 等）进行极其精准的“截肢手术”，使其硬件感知层与 Kuavo 的实际 30 维触觉传感器完美吻合。

> [!WARNING]
> **预训练权重兼容性断言**：
> 由于本阶段将修改 DECO 底层触觉感知网络（`nn.Linear`）的输入维度，这会导致 DECO 官方提供的“带有触觉融合层”的预训练权重无法直接加载（会报 Tensor Shape Mismatch 错误）。
> （注：纯视觉层 ResNet34 的权重依然可完美加载。）
> 这意味着，由于换了传感器，Kuavo 的触觉融合网络部分必须从头开始训练（Train from scratch）。此为权衡后一致接受的妥协。

- [ ] **3.1 触觉编码器手术 (Tactile Encoder Surgery)**
  - [ ] **切除冗余代码**：删除原版 `deco.py` 中专为 Inspire Hand 写的 `init_tac_regions` 函数（该函数原本用于把 1062 维强行切片均分降为 17 维）。
  - [ ] **改写网络输入形状**：将 `self.tactile_encoder` 中的第一个 `nn.Linear` 输入维度从 `1062*2` 改为 `15*2` (即 30)；将 `self.gated` 的门控维度从 `68` 改为 `64` (15+15+34=64)。
  - [ ] **透传前向传播**：在 `forward` 函数中，删除原来的 `for` 循环求均值代码，直接令 `tac1_avg = tac1` (15维) 和 `tac2_avg = tac2` (15维)，无损直通网络。
- [ ] **3.2 动作及本体感知维度确认**
  - [ ] 确认源码中动作/本体输入没有引发关于 28 维度的硬编码越界错误。
- [ ] **3.3 模型静态验证**
  - [ ] 构造一个 Fake Tensor Batch (包含 256x256 图像, 28维动作, 30维触觉)，空跑一次网络前向传播，验证 Shape 是否完全对齐，确保计算图能正常反向传播。

---

## 阶段四：核心封装阶段 (Core Wrapper Phase)
**核心目标**：在 `kuavo_train/wrapper/policy/deco/` 下构建 LeRobot 框架拦截器，无缝嵌入 Kuavo 生态，同时严格保证 DECO 原生 Loss 计算逻辑不变，把对 DECO 核心代码的修改降到最低。

- [ ] **4.1 开发 DECOConfigWrapper (配置层伪装)**
  - [ ] 新建配置类，注册 DECO 专属超参数（`chunk_size`, `dim`, `num_attn_blocks` 等）。
  - [ ] 统筹管理 `image_features` 和 `tactile_features` 键名，确保兼容 LeRobot 的配置文件解析体系。
- [ ] **4.2 开发 DECOPolicyWrapper.forward (训练拦截器)**
  - [ ] 继承自 `nn.Module`，对外伪装成标准的 LeRobot Policy 接口。
  - [ ] **解包与切片**：从 LeRobot 传入的字典 `batch` 中提取双目图像赋值给 `img1`, `img2`；将 `batch["observation.tactile"]` (30维) 切片拆分为 `tac1` (前15维左手) 和 `tac2` (后15维右手)。
  - [ ] **无损底层调用**：将解析好的数据传入动过手术的 `DECO(..., training=True)`。
  - [ ] **原生 Loss 保护与外层欺骗**：原封不动保留 DECO 源码中的 Flow Matching 训练目标 `F.mse_loss(out, noise - action)`，其中 `out` 是网络预测的速度场/残差方向，`noise - action` 是从专家动作指向噪声样本的目标速度场。将计算结果封装成严格的 `(loss, {"loss": loss.item()})` 格式返回，实现“绝对不修改官方 `train.py`”的终极目标。
- [ ] **4.3 开发 DECOPolicyWrapper.select_action (推理发射器)**
  - [ ] **组装实时观测**：接受真机或仿真环境发来的实时观测字典，转换为网络张量。
  - [ ] **原生去噪循环**：调用 `DECO(..., training=False)` 执行其原生的多步去噪推理，获得 Action Chunk 动作序列。
  - [ ] **标准队列发射**：复用 Kuavo 现存的 `self._queues["action"]` 双端队列调度机制，将动作推入队列，每次 `popleft()` 返回标准的 28 维底层控制指令供硬件执行。

---

## 阶段五：配置集成阶段 (Configuration Integration Phase)
**核心目标**：编写面向用户的 YAML 配置文件，充分利用 Kuavo 的基础设施（如视觉增强）赋能 DECO。

- [ ] **5.1 编写 `configs/policy/deco_config.yaml`**
  - [ ] **继承标准参数**：沿用 Kuavo 的 Hydra 配置规范，继承 `batch_size: 32`, `max_epoch: 500`, `device: "cuda"` 等基础训练参数。
  - [ ] **复用 Kuavo 视觉增强器**：将 `diffusion_config.yaml` 中强大的 `RGB_Augmenter` 模块（包含 ColorJitter, SharpnessJitter, GaussianNoise 等）直接平移。由 LeRobot 数据管道自动完成图像增强，从而彻底抛弃原 DECO `dataset.py` 内部写的图像处理逻辑。
  - [ ] **注入 DECO 超参数**：通过 `policy_name: deco` 和 `_target_` 指向我们在第四阶段写的 `DECOConfigWrapper`，并设定 `chunk_size: 32`, `dim: 512`, `num_attn_blocks: 6`, `use_tactile: true` 等专有网络参数。

---

## 阶段六：部署与演示阶段 (Deployment & Demonstration Phase)
**核心目标**：验证最终生成的 `.safetensors` 模型资产包能否在 Kuavo 现有的部署体系内顺滑流转，实现“黑盒”调用。

- [ ] **6.1 适配 `kuavo_deploy` 节点**
  - [ ] 确保 DECO 训练产出的 `.safetensors` 和 `config.json` 能够被部署脚本（如 `eval_kuavo.py`）无缝读取。
  - [ ] 由于我们在 Phase 4 完成了统一封装，部署团队只需调用 `action = policy.select_action(obs_dict)`，完全不需要知道底层模型是 DECO 还是 ACT。
- [ ] **6.2 闭环测试验证**
  - [ ] **仿真验证**：在 MuJoCo 仿真中启动机器人，验证 28 维动作流输出是否平滑，物理映射是否正确（严防左右手顺拐）。
  - [ ] **触觉反馈验证**：在真机或仿真中施加指尖压力，观察模型是否能根据输入的 30 维法向力张量做出期望的行为调整。
