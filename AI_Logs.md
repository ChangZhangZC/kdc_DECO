# AI Execution Logs

## 2026-05-18

### 同步 PLANS 阶段六当前进度状态
- **任务**: 根据四个部署链路文件已回退到提交 `dd323acfcdb162653acac201e63afa74874ab0a7` 的事实，修正 `PLANS.md` 中仍被误标为完成的阶段六部署项。本次只做文档状态同步，未运行 Python、ROS、训练、forward、validator、仿真、部署或环境变更命令。
- **修改文件**: `PLANS.md`
  - 将 `ObsBuffer` 接入 DECO depth topic 与 `depth_encoding: compressed_image` 的任务从已完成改为未完成。
  - 将 `ObsBuffer` 接入 `/dexhand/touch_state` 并构造 30D `observation.tactile` 的任务从已完成改为未完成。
  - 将 `ConfigEnv` / `KuavoBaseRosEnv` 在线拼接 `state_layout: deco_28d` 的任务从已完成改为未完成。
  - 将 `KuavoBaseRosEnv.step()` 解释并下发 DECO 28D action 的任务从已完成改为未完成。
  - 将 `server.py` 支持启动参数或 `KUAVO_DEPLOY_CONFIG` 选择部署配置的任务从已完成改为未完成。
  - 将服务端统一加载 run 根目录 pre/post processor 并在 `select_action(raw_obs)` 内接管处理链路的任务从已完成改为未完成。
  - 将 Done When 中“DECO 在线部署 obs/action 完成静态接入”改为未完成。
- **当前边界**:
  - `real_single_test.py` 与 `sim_auto_test.py` 中的 DECO policy 类型和 `DECOProcessor.py` 注册仍然保留，因此 `PLANS.md` 中对应两项继续保持已完成。
  - `configs/deploy/kuavo_deco_env.yaml` 新增与 `configs/deploy/kuavo_env.yaml` 通用配置还原仍然成立，因此相关条目继续保持已完成。

### 回退阶段六部署链路四个文件到指定提交状态
- **任务**: 根据用户要求，暂时忽略此前对阶段六部署链路的实操改动，将四个部署链文件恢复到提交 `dd323acfcdb162653acac201e63afa74874ab0a7` 中的状态。本次未运行 Python、训练、forward、validator、仿真、ROS 节点、部署服务或环境变更命令。
- **恢复文件**:
  - `kuavo_deploy/config.py`
  - `kuavo_deploy/utils/obs_buffer.py`
  - `kuavo_deploy/kuavo_env/KuavoBaseRosEnv.py`
  - `kuavo_deploy/kuavo_service/server.py`
- **恢复结果**:
  - 上述四个文件已与 `dd323acfcdb162653acac201e63afa74874ab0a7` 中的版本一致。
  - 此次恢复撤销了此前尚未正式确认手术方案的部署链改动，包括 `state_layout: deco_28d`、在线 28D state/action、DECO depth/tactile callback 和服务端 processor 接管等内容。
- **边界说明**:
  - 本次只按用户要求恢复四个部署链文件，不回退已确认的 wrapper、policy YAML、DECO 主干或训练入口相关文件。
  - 后续阶段六部署链路应先补充手术方案并获得确认，再重新实施代码改动。

### 明确 DECO 部署资产采用 run 根目录并新增专用部署配置
- **任务**: 根据用户确认的方案 A，承认 DECO 部署资产是完整 `outputs/train/<task>/<method>/<timestamp>/` run 目录，而不是单独的 `epochbest/` 或任意 `epoch<epoch>/` 子目录。本次仅做静态配置与文档修改，未运行 Python、训练、forward、validator、仿真、部署或环境变更命令。
- **新增文件**: `configs/deploy/kuavo_deco_env.yaml`
  - 新增 DECO 专用部署配置，避免继续把 DECO 的 RGB-D depth topic、触觉说明和 run-root 部署语义混入通用 `kuavo_env.yaml`。
  - 在文件头部明确完整部署资产为 `outputs/train/<task>/<method>/<timestamp>/`，其中 `epoch` 字段只选择 `epoch<epoch>` 权重子目录；`policy_preprocessor.json` 与 `policy_postprocessor.json` 仍保存在 run 根目录，所以 `epochbest/` 单独不是完整部署包。
  - 将 DECO depth topic 写为 `/cam_h/depth/image_raw/compressed`，与当前数据转换规划中的 `compressed_image` decoder 路线保持一致。
  - 保留 tactile adapter 的配置注释，说明阶段 6.1 完成 `dexhandTouchState -> 30D observation.tactile` 在线接入后再启用。
- **修改文件 1**: `configs/deploy/kuavo_env.yaml`
  - 将此前为了 DECO 审查临时扩展的 `policy_type` 注释还原为原通用配置语义：`Supports diffusion, act`。
  - 未修改原有 ACT/DP topic、depth `compressedDepth` 路线、运行参数或字段结构。
- **修改文件 2**: `README_DECO.md`
  - 新增“部署路径约定”章节，说明 DECO 应使用 `configs/deploy/kuavo_deco_env.yaml`，通用 `kuavo_env.yaml` 保持 ACT/DP 默认语义。
  - 明确部署配置中填写的是 `task/method/timestamp` 三层 run 路径，权重由 `epoch` 字段选择；完整部署、迁移或归档时应保留整个 `run_xxx/` 目录。
  - 补充说明现有 `script.py` / `script_auto_test.py` 需要通过 `--config configs/deploy/kuavo_deco_env.yaml` 显式传入 DECO 配置；`server.py` 的配置路径可配置化仍属于阶段六后续项。
- **修改文件 3**: `PLANS.md`
  - 更新已确认技术决策和阶段 6.1 清单，记录 DECO 部署资产采用原 Kuavo run 根目录方案。
  - 将 `kuavo_deco_env.yaml` 新增和 `kuavo_env.yaml` 还原标记为已完成；保留“严格 28D dexhand state/action 与 30D tactile 在线接入仍待阶段 6.1 后续完成”的未完成项。
- **修改文件 4**: `Content/DECO_Technical_Decisions.md`
  - 更新权重格式与路径语义，区分训练初始化用的 `.safetensors` policy 权重目录、兼容导入用的 `.pth`，以及部署归档用的完整 run 根目录。
  - 明确 `epochbest/` 或任意 `epoch<epoch>/` 只是权重子目录，不包含完整 processor 资产。
- **未执行项与边界**:
  - 未运行任何 Python、YAML loader、ROS 节点、训练、forward、validator、仿真或部署命令。
  - 未删除文件，未修改 `third_party/lerobot/`。

### 处理阶段四 Wrapper 静态审查反馈
- **任务**: 根据静态 code review 反馈修复 DECO wrapper、训练入口和部署入口中的一致性问题。本次仍遵守 No-Runtime 约束，未运行 Python、训练、forward、validator、仿真、部署或环境变更命令。
- **修复 1**: `kuavo_train/wrapper/policy/deco/DECOPolicyWrapper.py`
  - 修正 `deco_init_pth_path` 初始化路径的冻结语义：当 `training_stage: tactile_adapter` 且 `freeze_pretrained_main: true` 时，`.pth` 初始化与 `.safetensors` 初始化一样会进入主干冻结策略。
  - 新增 `_freeze_main_for_tactile_adapter()`：在 tactile adapter 阶段按参数名冻结主干，只保留 `tactile_encoder`、`gated`、`pos_tac_embedd`、tactile cross-attention 和 `PI_Adapter` 相关参数可训练。
  - 当最终保存的 policy 配置关闭外部初始化读取时，加载 `.safetensors` 后不会再访问第一阶段目录或原生 `.pth`，但仍保留 tactile adapter 阶段的主干冻结语义。
- **修复 2**: `kuavo_train/wrapper/policy/deco/DECOConfigWrapper.py`
  - 新增 `load_external_init_weights`，用于区分“训练初始化需要读取外部权重”和“最终 policy 已经自包含”。
  - 在 `_save_pretrained()` 中清空 `base_policy_path`、`adapter_model_path`、`deco_init_pth_path`，并将 `load_external_init_weights` 写为 `false`，保证最终 `.safetensors` policy 目录迁移后不依赖外部初始化路径。
  - 放宽 tactile adapter 的路径校验：只有 `load_external_init_weights: true` 时才要求提供 `base_policy_path`、`deco_init_pth_path` 或 `adapter_model_path`。
- **修复 3**: `configs/policy/deco_config.yaml`
  - 新增并注释 `load_external_init_weights: true`，明确该字段只用于训练初始化，最终保存时会关闭。
- **修复 4**: `kuavo_train/train_policy.py` 与 `kuavo_train/train_policy_with_accelerate.py`
  - 优化器构建改为使用 `policy.get_optim_params()`，使 ACT/DECO wrapper 中的参数组和 `requires_grad` 过滤逻辑真正生效。
  - 未修改训练循环、epoch、dataloader、checkpoint 或 scheduler 语义。
- **修复 5**: `kuavo_deploy/src/eval/real_single_test.py` 与 `kuavo_deploy/src/eval/sim_auto_test.py`
  - 新增 `CustomDECOPolicyWrapper` 导入，并支持 `policy_type: deco` / `DECO`。
  - 显式导入 `DECOProcessor.py`，确保加载保存的 `policy_preprocessor.json` 时能注册并找到 `deco_rgbd_letterbox_processor`。
  - 保留原 `diffusion`、`act`、`client` 分支，并避免对 `client` 分支强制调用模型专属的 `eval()` / `to()` / `reset()`。
- **修复 6**: `kuavo_deploy/kuavo_service/server.py`
  - 服务器侧新增 `act`、`deco` 分支，不再只硬编码 `CustomDiffusionPolicyWrapper`。
  - 改用当前仓库存在的 `load_kuavo_config()` 和 `configs/deploy/kuavo_env.yaml`，避免继续引用不存在的 `configs.deploy.config_inference` 与 `kuavo_real_env.yaml`。
  - 显式导入 `DECOProcessor.py`，为服务端可能加载 DECO processor 预留注册。
- **修复 7**: `kuavo_deploy/config.py` 与 `configs/deploy/kuavo_env.yaml`
  - 部署配置校验支持 `diffusion`、`act`、`deco`、`client`，并更新 YAML 注释。
- **修复 8**: `README_DECO.md`、`PLANS.md`、`Content/DECO_Technical_Decisions.md`
  - 同步 README 当前状态：训练 wrapper、DECO 专用 preprocessor、策略配置和基础部署入口注册已完成静态接入。
  - 在计划和技术决策中记录最终 policy 自包含语义、部署入口 DECO 注册、DECOProcessor 注册和 optimizer 使用 wrapper 参数组的修复。
- **未执行项与边界**:
  - 未运行 Python、import 编译、forward shape test、训练、数据 validator、仿真、部署或服务端联调。
  - 未执行任何安装、删除或环境变更命令。
  - 未修改 `third_party/lerobot/`，未修改根目录 `DECO/` 原生备份。

### 同步阶段四/五 Wrapper 与两阶段训练设计决策
- **任务**: 根据用户关于 DECO wrapper、tactile 独立分支、DECO 专用 preprocessor、两阶段训练和权重格式语义的确认，更新 `PLANS.md` 与 `Content/DECO_Technical_Decisions.md`。本次只做文档级方案同步，未运行 Python、训练、forward、validator、部署脚本或任何环境变更命令。
- **修改文件 1**: `PLANS.md`
  - 将重构日期更新为 `2026-05-18`。
  - 更新总体架构图，将 Kuavo-DECO 输入明确拆成三条平行主枝：
    - `state branch`：28 维 `observation.state` 由 LeRobot stats 归一化后进入 DECO `obs_encoder`，再调制 time embedding。
    - `RGB-D visual branch`：RGB/depth 经 DECO 专用 preprocessor，同步 `256x256 letterbox` 后进入 RGB-D 双流 ResNet 与 cross attention。
    - `tactile branch`：30 维 `observation.tactile` 作为独立 `TACTILE` feature，不走 STATE `MEAN_STD`，由 wrapper 按左右手 tactile max 归一化并默认 clamp 到 `[0, 1]` 后进入 tactile encoder / PI_Adapter。
  - 在已确认技术决策中新增 `FeatureType.TACTILE`、`TACTILE: IDENTITY`、`clip_tactile_to_unit: true`、DECO 专用 preprocessor 放置位置、`.safetensors` 与 `.pth` 权重语义分层，以及 `train_policy.py` 只做最小策略注册、不改核心训练循环的边界。
  - 更新阶段三 `3.6` 分阶段训练边界，明确 `visual_main` 与 `tactile_adapter` 是两次独立训练启动，而不是同一个 epoch loop 内自动交替或自动切换。
  - 更新阶段三 `3.7` 预处理边界，明确 RGB/depth deterministic `Resize/Letterbox` 在 LeRobot normalizer 之前执行，RGB 随机增强在 letterbox 之后、normalizer 之前执行。
  - 新增阶段四 `4.0 LeRobot feature 与 DECO preprocessor 接入边界`，记录 `lerobot_patches/custom_patches.py` 只负责全局 feature/type 兼容，DECO 专用视觉预处理应放在 `kuavo_train/wrapper/policy/deco/`。
  - 更新阶段四 `DECOConfigWrapper`、`DECOPolicyWrapper.forward` 与 `select_action` 任务，补充 `training_stage`、`base_policy_path`、`deco_init_pth_path`、`clip_tactile_to_unit`、两类部署资产以及 tactile 开关行为。
  - 更新阶段五 `configs/policy/deco_config.yaml` 任务，默认面向第一阶段主干训练：`use_tactile: false`、`use_tactile_lora: false`、tactile max 允许 `null`；第二阶段通过 override 开启 tactile adapter 并加载第一阶段 `.safetensors` policy 目录。
  - 更新 Done When，新增 `FeatureType.TACTILE`、DECO 专用 preprocessor、两阶段训练 `.safetensors` policy 目录和 tactile clamp 行为的静态审查要求。
- **修改文件 2**: `Content/DECO_Technical_Decisions.md`
  - 将最后更新时间更新为 `2026-05-18`。
  - 更新总体技术路线，明确三条平行主枝：state、RGB-D visual、tactile。
  - 在核心原则中新增：`observation.tactile` 独立识别为 `TACTILE` 并保持 `IDENTITY`，DECO wrapper 再执行 tactile max normalization；DECO 专用 preprocessor 不放在 `lerobot_patches/`；两阶段训练是两次独立启动；正式训练和部署资产为 LeRobot `.safetensors` policy 目录。
  - 更新 `4.5 obs/state 归一化与模型接入策略`，说明 tactile 不拼入 state，也不跟随 `STATE: MEAN_STD`。
  - 更新 `4.6 触觉模型手术`，记录默认 `clip_tactile_to_unit: true`，与 DECO 原生 inference 的 `clamp(0, 1.0)` 保持一致。
  - 更新 `4.7 Tactile Plugin / LoRA-style Adapter 机制`，将默认配置改为第一阶段关闭 tactile/LoRA，第二阶段通过 override 开启。
  - 重写 `4.8 分阶段训练与验证策略`，明确第一阶段完整训练 RGB-D + state 主干，第二阶段加载第一阶段 `.safetensors` policy 并冻结主干训练 tactile/PI_Adapter；若用户不使用触觉，则第一阶段产物即可部署。
  - 新增 `4.9 DECO 专用 preprocessor 放置边界`，说明 preprocessor 负责 raw RGB-D 上的 letterbox 和 RGB-only augmentation 顺序，`policy.forward()` 不做 raw-pixel padding。
  - 新增 `4.10 权重格式与路径语义`，区分 `base_policy_path`、`adapter_model_path` 和 `deco_init_pth_path`。
  - 更新待实现清单，加入 `FeatureType.TACTILE`、`DECOProcessor.py`、两阶段训练配置与 tactile clamp 的实现要求。
- **未执行项**:
  - 未修改 wrapper 代码、训练入口、配置 YAML 或 LeRobot patch 代码。
  - 未运行 Python、未做 import 编译检查、未训练、未验证数据集、未部署。
  - 未修改 `third_party/lerobot/`。

### 阶段四 DECO Wrapper 与第三方 DECO 主干静态实现
- **任务**: 根据用户确认的阶段四 wrapper 方案，围绕 `third_party/deco` 下的 Kuavo 定制 DECO 主干完成训练 wrapper、DECO 专用 preprocessor、tactile feature 隔离、两阶段训练配置和文档同步。本次遵守 No-Runtime 约束，只做静态代码与文档修改，未运行 Python、训练、forward、validator、部署脚本或任何环境变更命令。
- **修改文件 1**: `third_party/deco/models/deco/deco.py`
  - 删除与当前 Kuavo-DECO RGB-D + tactile 三主枝逻辑无关、容易造成二次审阅混淆的旧字段和加载逻辑，包括 `visual_input_mode`、`img_pretrain`、`freeze_backbone`、`pretrain_model_path`、`adapter_model_path`、`load_visual_pretrain`、`load_encoder_pretrain`、旧 `freeze` 逻辑和底部示例代码。
  - 将 DECO 主体收窄为纯 RGB-D action-token Flow Matching 模型：RGB 与 depth 使用独立 backbone，cross attention 后仍以两路 visual tokens 接入 MMAttention。
  - 将 checkpoint 路径、`.safetensors` 加载、`.pth` 兼容加载和冻结策略全部上移到 Kuavo wrapper，避免第三方 DECO 主干同时承担训练编排职责。
- **修改文件 2**: `third_party/deco/config/deco.yaml`
  - 删除 `visual_input_mode`、`img_pretrain`、`freeze_backbone`、`pretrain_model_path`、`adapter_model_path` 等旧配置字段。
  - 保留模型结构参数、RGB-D backbone、触觉 max 语义和当前 DECO 主干真正需要的字段，使 `third_party/deco` 默认只表达当前 Kuavo RGB-D 主干。
- **修改文件 3**: `third_party/deco/models/deco/train_one_epoch.py`
  - 将旧局部变量 `img1/img2` 重命名为 `rgb/depth`，只做命名级静态清理，不改变 DECO 原生训练辅助函数的 loss、优化或数据读取流程。
- **新增文件 1**: `kuavo_train/wrapper/policy/deco/__init__.py`
  - 新增 `ensure_deco_on_path()`，集中处理 `third_party/deco` 路径注入。
  - 用中文注释明确 wrapper、DECO 主干和 LeRobot patch 的职责边界。
- **新增文件 2**: `kuavo_train/wrapper/policy/deco/DECOConfigWrapper.py`
  - 新增 `CustomDECOConfigWrapper` 并注册为 `custom_deco`。
  - 配置字段只保留当前已确认逻辑：三主枝输入、RGB/depth key、tactile key、`training_stage`、`use_tactile_lora`、`tactile_lora_rank`、`freeze_pretrained_main`、`base_policy_path`、`adapter_model_path`、`deco_init_pth_path`、tactile max、letterbox、30Hz/10Hz stride、optimizer/scheduler 等。
  - 增加配置 guard：`tactile_adapter` 必须开启 tactile 和 tactile LoRA，并提供 `base_policy_path` 或 `deco_init_pth_path`；`use_tactile: true` 时左右 tactile max 必须为正数；`dataset_hz/control_hz/action_stride` 必须一致。
  - 默认 normalization mapping 将 `TACTILE` 设为 `IDENTITY`，避免 tactile 误走 `STATE: MEAN_STD`。
- **新增文件 3**: `kuavo_train/wrapper/policy/deco/DECOProcessor.py`
  - 新增 DECO 专用 RGB-D preprocessor：在 LeRobot normalizer 前对 RGB/depth 同步执行 `256x256 letterbox`。
  - RGB 使用 bilinear 和灰色 padding，depth 使用 nearest 和独立 padding；depth 不做 photometric augmentation。
  - 将 resize 调用封装为 `_interpolate()`，只有 bilinear/bicubic 模式传入 `align_corners=False`，nearest 模式不传该参数，避免 depth resize 分支出现不必要的 PyTorch 参数歧义。
  - 构建 DECO 专用 pre/post processor pipeline，使训练入口中的 RGB 随机增强可以插在 letterbox 后、normalizer 前。
- **新增文件 4**: `kuavo_train/wrapper/policy/deco/DECOPolicyWrapper.py`
  - 新增 `CustomDECOPolicyWrapper`，将 Kuavo batch 映射到 DECO 原生 `forward(rgb, depth, obs, action, task_id, tac1, tac2, training=True/False)`。
  - `forward()` 中不执行 raw-pixel resize/padding，只读取 preprocessor 已处理的 RGB/depth。
  - tactile 作为独立第三主枝读取 `observation.tactile`，拆成左右手各 15 维，按左右 tactile max 做 DECO-style 归一化并默认 clamp 到 `[0, 1]`。
  - loss 严格保持 DECO Flow Matching 目标 `F.mse_loss(out, noise - action)`，第一版不额外使用 `action_is_pad` mask。
  - 支持 `base_policy_path` 的 `.safetensors` 加载与第二阶段主干冻结，支持 `deco_init_pth_path` 作为历史 `.pth` 初始化兼容入口，支持 `adapter_model_path` 作为可选 adapter/full policy 加载入口。
  - `select_action()` 使用 `action_stride` 从 action chunk 中按 30Hz 数据 / 10Hz 控制节奏抽取动作队列。
- **修改文件 4**: `lerobot_patches/custom_patches.py`
  - 新增或扩展 `FeatureType.TACTILE`，并在 `dataset_to_policy_features` 中优先将 `observation.tactile` 识别为 `TACTILE`，避免其被通用 `observation*` 规则错误归入 `STATE`。
- **修改文件 5**: `kuavo_train/train_policy.py`
  - 只做最小策略注册和 processor 分发：新增 `deco` / `DECO -> CustomDECOPolicyWrapper` 注册，并在 policy config 是 DECO wrapper 时使用 `make_deco_pre_post_processors()`。
  - 未修改训练循环、优化器构建、dataloader、epoch 调度、checkpoint 逻辑或 LeRobot 主训练语义。
- **修改文件 6**: `kuavo_train/train_policy_with_accelerate.py`
  - 与单机训练入口保持同样的最小 DECO 注册和 pre/post processor 分发。
  - 未修改 accelerate 训练循环或分布式训练语义。
- **新增文件 5**: `configs/policy/deco_config.yaml`
  - 新增 Kuavo-DECO 默认策略配置，默认面向第一阶段 `visual_main`：`use_tactile: false`、`use_tactile_lora: false`，第一阶段产物可作为无触觉 `.safetensors` policy 直接部署。
  - 记录第二阶段 `tactile_adapter` 的 override 方式：开启 tactile 与 tactile LoRA，加载第一阶段 `base_policy_path`，冻结主干训练 tactile/PI_Adapter。
  - 明确 RGB-D 预处理、RGB-only augmentation、ResNet34 默认 backbone、30Hz 数据 / 10Hz 控制、`action_stride: 3` 和 tactile max 配置约束。
- **修改文件 7**: `README_DECO.md`
  - 更新当前 wrapper 状态，说明 `kuavo_train/wrapper/policy/deco/` 已存在，并记录 DECO 主干已固定为 RGB-D，路径与 checkpoint 加载由 wrapper 负责。
  - 更新当前限制，说明本地只完成静态实现，尚未执行 import、forward、训练或部署验证。
- **修改文件 8**: `third_party/deco/README.MD`
  - 更新示例配置，删除已从 `third_party/deco/config/deco.yaml` 移除的旧路径和冻结字段。
  - 添加说明：路径、冻结与两阶段训练不再由 DECO 原生 YAML 表达，而由 Kuavo wrapper/config 负责。
- **修改文件 9**: `Content/DECO_Technical_Decisions.md`
  - 同步阶段四实现状态，勾选 DECO config、preprocessor、tactile feature、tactile max、两阶段训练和 wrapper 相关清单。
  - 将随机增强描述收窄为实际实现的 `GaussianBlur`，并记录不额外保留 `visual_token_mode` 这类与当前逻辑无关的字段。
  - 清理早期方案中未实现的随机 blur 变体描述，避免与当前 `configs/policy/deco_config.yaml` 的实际增强池不一致。
  - 将早期方案中旧双路视觉接口名改为概念性描述，避免与当前 `forward(rgb, depth, ...)` 接口混淆。
- **修改文件 10**: `PLANS.md`
  - 勾选阶段四 wrapper 相关完成项、阶段五策略配置完成项和 Done When 中已静态完成的 RGB-D/tactile/two-stage 事项。
  - 同步勾选阶段三 `3.5`、`3.6`、`3.7` 中已由本次 wrapper/config 实现完成的 tactile LoRA 映射、两阶段训练边界、DECO preprocessor 与 state/tactile 接入边界。
  - 保留 validator、仿真、部署、requirements 最小化复查等尚未执行事项为未完成。
  - 清理早期计划中未实现的第一阶段冻结开关、触觉阶段自动开关和随机 blur 变体描述，使计划只保留当前已确认的 wrapper/config 字段。
  - 将早期计划中旧双路视觉接口名改为概念性描述，保持阶段四审阅材料只围绕 RGB-D 路线展开。
- **修改文件 11**: `README_DECO.md` 与 `third_party/deco/README.MD`
  - 将旧视觉模式字段名改为概念性描述，避免在当前阶段四审阅中继续出现已删除字段。
- **未执行项与边界**:
  - 未运行 Python、import 编译、forward shape test、训练、数据 validator、仿真或部署。
  - 未执行 `pip install`、`conda install`、`brew install` 或其他环境修改命令。
  - 未执行删除文件命令，未修改 `third_party/lerobot/`，未修改根目录 `DECO/` 原生备份。

## 2026-05-17

### 移除 DECO 模型主体中的 dual_rgb 旧兼容入口
- **任务**: 根据用户明确要求，删除旧模型中对当前 Kuavo-DECO RGB-D 路线无用且容易造成混淆的 `img1/img2`、`dual_rgb_img_encoding`、`dual_rgb` 兼容命名和分支，使 `third_party/deco` 下的 DECO 主体仅服务 RGB-D 输入。全过程遵守 No-Runtime 约束，未运行 Python、训练、forward、validator 或环境变更命令。
- **修改文件 1**: `third_party/deco/models/deco/deco.py`
  - 将 `DECO.__init__` 和 `modeling` 的 `visual_input_mode` 默认值改为 `dual_stream_rgb_depth`。
  - 将配置校验收窄为仅允许 `dual_stream_rgb_depth`，传入其他值会显式报错。
  - 删除 `dual_rgb_img_encoding` 旧双 RGB 编码函数，`img_encoding` 现在直接进入 RGB-D 编码路径。
  - 将 `forward(img1, img2, ...)` 改为 `forward(rgb, depth, ...)`，并同步更新 docstring、推理采样 device/dtype 来源和示例注释。
  - 删除 `uses_rgbd` 兼容标志，depth backbone、depth head 与 RGB-depth fusion 现在始终存在。
  - 更新 `MMAttention` 注释，将视觉 token 前后两半明确写为 RGB stream 与 depth stream。
- **修改文件 2**: `third_party/deco/inference.py`
  - 将 `preprocess(img1, img2, ...)` 改为 `preprocess(rgb, depth, ...)`。
  - 删除根据 `visual_input_mode` 分流的旧双 RGB 推理预处理逻辑，推理入口始终按 RGB-D 处理第二路输入。
  - 将 `predict_action` 参数与内部调用同步改为 `rgb/depth` 命名。
- **修改文件 3**: `third_party/deco/config/deco.yaml`
  - 删除 `dual_rgb` 配置说明，明确 `visual_input_mode` 仅保留 Kuavo RGB-D 路线。
- **修改文件 4**: `README_DECO.md`
  - 更新阶段三视觉前端说明，明确旧 DECO `dual_rgb` 双 RGB 兼容入口已从 Kuavo 定制副本的模型主体中移除。
- **修改文件 5**: `third_party/deco/README.MD`
  - 更新 YAML 示例，将 `visual_input_mode` 注释改为 Kuavo RGB-D only。
- **修改文件 6**: `PLANS.md`
  - 在阶段三视觉前端 checklist 中记录已移除旧 `dual_rgb` 双 RGB 兼容分支，并将模型主体接口收窄为 `forward(rgb, depth, ...)`。
- **静态检查**:
  - 使用 `rg` 检查 `third_party/deco/models/deco/deco.py`、`third_party/deco/inference.py`、`third_party/deco/config/deco.yaml`、`README_DECO.md`、`third_party/deco/README.MD`，确认模型主体和推理入口中不再残留 `dual_rgb_img_encoding`、`uses_rgbd` 或 `img1/img2` 旧接口命名。
- **未执行项**:
  - 未运行 Python、未进行 import 编译检查、未进行 forward shape test、未训练、未验证数据集、未部署。

### 清理 third_party/deco 配置中的旧 DECO 兼容字段
- **任务**: 根据用户要求复查 `deck.yml/deco.yaml` 中是否存在重复的 `chunk_size` 与新旧 tactile max 参数，并删除与当前 Kuavo-DECO RGB-D 模型路线无关的旧字段，避免后续阶段四 wrapper 接入时产生配置歧义。静态检查确认仓库内没有 `deck.yml`，本次实际清理对象为 `third_party/deco/config/deco.yaml`。
- **修改文件 1**: `third_party/deco/config/deco.yaml`
  - 将 `visual_input_mode` 默认值从 `dual_rgb` 调整为 `dual_stream_rgb_depth`，使第三方副本默认服务 Kuavo RGB-D 路线；`dual_rgb` 仍作为源码层面的兼容模式保留。
  - 删除重复的 `data.chunk_size`，明确 `model.chunk_size` 是 DECO action chunk 长度的唯一权威配置。
  - 删除旧 DECO 别名 `tac_left_max/tac_right_max`，只保留 Kuavo 标准字段 `tactile_left_max/tactile_right_max`。
  - 删除 DECO 原生 dataset/inference 路线使用的 `norm_type`、`observation_mean/std/min/max`、`action_mean/std/min/max`，避免与 Kuavo/LeRobot preprocessor 的归一化职责混淆。
  - 删除旧 RGB `img_mean/img_std` 配置，仅保留 `img_size: [256, 256]`；图像标准化由后续 Kuavo wrapper/preprocessor 统一负责。
- **修改文件 2**: `third_party/deco/inference.py`
  - 新增兼容式 `normalize_obs_if_configured`：只有旧统计字段显式存在时才执行 DECO 原生 obs 归一化，否则默认认为 state 已由 Kuavo wrapper/preprocessor 处理。
  - 新增 `build_rgb_transform`：只有旧 `img_mean/img_std` 显式存在时才追加 RGB Normalize，否则只做 resize、tensor 化与缩放。
  - 修改 `postprocess`：只有旧 action 统计字段显式存在时才做反归一化，否则直接返回模型输出，避免依赖已从配置中删除的旧字段。
  - 修改 `get_tactile_max`：不再回退读取 `tac_left_max/tac_right_max`，`use_tactile=True` 时必须使用 Kuavo 标准字段 `tactile_left_max/tactile_right_max`。
- **修改文件 3**: `README_DECO.md`
  - 更新阶段三说明：当前 `third_party/deco/config/deco.yaml` 默认 `dual_stream_rgb_depth`，`dual_rgb` 只作为兼容/对照开关。
  - 删除“保留旧 `tac_left_max/tac_right_max`”的说明，改为说明旧别名已移除。
  - 记录配置中只保留一处 `chunk_size`，并说明 obs/action/image 标准化由 Kuavo wrapper/preprocessor 负责。
- **修改文件 4**: `third_party/deco/README.MD`
  - 更新 YAML 示例，移除 `data.chunk_size`、`tac_left_max/tac_right_max`、obs/action 旧统计字段和 `img_mean/img_std`。
  - 添加 Kuavo note，说明当前 `config/deco.yaml` 已为 Kuavo RGB-D wrapper 路线瘦身，不再保留 DECO-50 原生训练配置字段。
- **修改文件 5**: `PLANS.md`
  - 在阶段三 checklist 中记录 `third_party/deco/config/deco.yaml` 已完成旧 tactile 别名清理、重复 `chunk_size` 清理，以及原生 obs/action 手动统计字段清理。
- **未执行项**:
  - 未运行 Python、未执行训练、forward、validator、部署或任何环境变更命令。
  - 未删除文件，未修改 `third_party/lerobot/`，未修改根目录 `DECO/` 原始参考副本。

## 2026-05-16

### 阶段三 third_party/deco 模型手术静态实现
- **任务**: 根据用户明确 `proceed` 指令，正式进入阶段三代码任务，并按用户要求直接修改 `third_party/deco` 下的 DECO 副本。目标是保留 DECO action-token Flow Matching 主干，同时新增 Kuavo RGB-D 视觉前端和 Kuavo 30 维触觉适配。全过程遵守 No-Runtime 约束，未执行 Python、训练、forward、validator 或环境变更命令。
- **修改文件 1**: `third_party/deco/models/deco/img_encoder.py`
  - 将原先固定 3-channel 的 `ResNet34` 拆成可配置 `ResNetBackbone`。
  - 新增 `ResNet18` 与保留 `ResNet34`，并提供 `build_resnet_backbone(backbone_name, in_channels)`。
  - 支持 `in_channels=1`，用于 Kuavo RGB-D 模式下的 depth backbone。
- **修改文件 2**: `third_party/deco/models/deco/deco.py`
  - 新增 `RGBDepthCrossAttentionFusion`，实现 ACT 风格 RGB-depth 双向 cross attention。
  - 新增 `visual_input_mode`，支持 `dual_rgb` 与 `dual_stream_rgb_depth` 两种模式：
    - `dual_rgb` 保留 DECO 原生双 RGB 兼容路径。
    - `dual_stream_rgb_depth` 将 `img1` 解释为 RGB、`img2` 解释为 depth。
  - 在 RGB-D 模式下新增独立 `depth_encoder`、`depth_head` 与 `rgb_depth_fusion`。
  - RGB 使用 3-channel backbone，depth 使用 1-channel backbone；depth conv1 由 RGB conv1 权重按通道均值初始化。
  - 改造 `img_encoding`：RGB-D 模式下分别编码 RGB/depth，投影后做 cross attention，并输出 `fused_rgb_tokens` 与 `fused_depth_tokens`，再拼接为 `[B, 2L, dim]` 进入 DECO `MMAttention`。
  - 保留 `pos_idx_embedd` 两路视觉流区分语义；中文注释说明当前两半 token 在 Kuavo RGB-D 模式下表示 RGB stream 与 depth stream。
  - 删除原 `init_tac_regions` 的 1062 维 Inspire Hand 区域均值逻辑。
  - 将 tactile 输入改为 `tac1=[B,15]` 与 `tac2=[B,15]`，即 Kuavo 左右手各 15 维 normalized tactile。
  - 将 `tactile_encoder` 改为 `30D -> 34D`，将 tactile gating/fusion 从 `68` 改为 `64`。
  - 新增 `encode_kuavo_tactile` 静态 shape 检查，开启 tactile 时若未传入左右手 15D 张量会显式报错。
  - 保留 DECO 主干的 `action_encoder`、`action_embedd`、`MMAttention`、`linear`、`add_noise`、Flow Matching denoising loop 与训练目标语义。
  - 修正 adapter finetune 冻结策略：只冻结 shape 匹配且实际加载成功的 checkpoint 参数，避免 Kuavo 30D tactile 新参数被错误冻结。
- **修改文件 3**: `third_party/deco/inference.py`
  - `letterbox` 新增 interpolation 参数。
  - 在 `dual_stream_rgb_depth` 模式下，推理预处理将 `img2` 按 depth 单通道处理，不再套 RGB ImageNet mean/std。
  - 对 3-channel repeat depth 取第一通道，恢复 1-channel depth 语义。
  - 新增 `get_tactile_max`，优先读取 Kuavo 语义的 `tactile_left_max` / `tactile_right_max`，并在 `use_tactile=True` 时要求二者为正数。
  - 关闭 tactile 时提供左右手 15D 零占位，避免非 tactile 推理路径被无关 tactile 参数阻塞。
- **修改文件 4**: `third_party/deco/config/deco.yaml`
  - 新增 `visual_input_mode`、`vision_backbone`、`depth_backbone` 配置项。
  - 添加中文注释说明 `dual_rgb` 与 `dual_stream_rgb_depth` 的含义。
  - 新增 Kuavo 语义的 `tactile_left_max` / `tactile_right_max`，默认 `null`，并保留原 DECO 旧字段 `tac_left_max` / `tac_right_max` 作为原生 dataset 兼容项。
  - 添加 tactile max 中文说明，强调 Kuavo 正式触觉训练应使用已转换成牛顿后的训练集统计正数，不应直接复用 Inspire Hand 原始量纲数值。
- **修改文件 5**: `third_party/deco/models/deco/train_one_epoch.py`
  - 仅修正旧注释中的 action 维度，将 `(chunksize, 26)` 改为 `(chunksize, 28)`；未改变训练 loss 或训练逻辑。
- **修改文件 6**: `third_party/deco/ACTION_SPACE_ANATOMY.md`
  - 更新 tactile 章节，说明 Kuavo-DECO 当前输入为左右手各 15 维、合计 30 维。
  - 记录 `init_tac_regions` 已从 Kuavo 定制副本中移除，当前 tactile 条件位置为 `15 + 15 + 34 = 64`。
- **修改文件 7**: `third_party/deco/README.MD`
  - 在 YAML 示例中补充 `visual_input_mode`、`vision_backbone`、`depth_backbone`。
  - 标注原 `tac_left_max/tac_right_max` 是 DECO 原始数值，Kuavo 训练应覆盖为牛顿量纲下的左右手 tactile max。
- **修改文件 8**: `README_DECO.md`
  - 新增阶段三模型手术说明，记录 RGB-D 模式、backbone 策略、触觉 30D 路径和保留的 DECO 主干行为。
  - 更新当前限制，说明训练 wrapper 与部署 wrapper 仍属后续阶段。
- **修改文件 9**: `PLANS.md`
  - 勾选阶段三中已实际完成的 `3.1` 视觉前端移植、`3.3` 主干保留策略、`3.4` 模型静态验证，以及 `3.2/3.5` 中已由 `third_party/deco` 代码完成的子项。
  - 保留 wrapper/config guard、LeRobot tactile feature 隔离、`use_tactile_lora` 到 `plugin` 映射等阶段四任务为未完成。
- **静态检查**:
  - 执行 `git diff --check`，结果通过，无 trailing whitespace。
  - 使用 `rg` 检查 `third_party/deco/models/deco` 与推理入口，确认核心代码中不再残留 `1062*2`、`[B, 68]`、`tactile_data_index`、`def init_tac_regions`、`chunksize, 26` 等旧实现痕迹。
- **未执行项**:
  - 未运行 Python、未进行 import 编译检查、未进行 forward shape test、未训练、未验证数据集、未部署。
  - 未修改根目录 `DECO/` 原始参考副本。
  - 未修改 `third_party/lerobot/`。

## 2026-05-15

### 更新阶段三图像增强、视觉 token、state/tactile 接入与 depth 默认策略决策
- **任务**: 根据用户确认，将阶段三前置讨论中达成的新方案同步到 `PLANS.md` 与 `Content/DECO_Technical_Decisions.md`。本条合并记录两轮讨论内容：上一轮关于 Resize/Letterbox、GaussianBlur、obs/state 归一化、depth 默认策略的结论；本轮关于 RGB/depth visual token 结构、DECO 256x256 letterbox、DECO-style tactile normalization 与原生 Flow Matching loss 的结论。用户已明确接受该方案并要求更新文档。
- **修改文件 1**: `PLANS.md`
  - 在 `0.2 已确认技术决策` 中新增视觉预处理分层决策：
    - `Resize/Letterbox` 作为 RGB 与 depth 共享的确定性空间预处理，不进入随机增强池。
    - 默认采用 DECO 原生 `256x256 letterbox`；RGB padding 使用灰色 `fill=128`，depth padding 单独配置，默认使用 `0` 或 invalid depth。
    - `RGB_Augmenter` 作为训练期随机增强池。
    - RGB resize 使用双线性插值，depth resize 使用 nearest 插值。
  - 在 `0.2 已确认技术决策` 中新增 RGB-D visual token 结构决策：
    - RGB 与 depth 使用独立 ResNet backbone，不共享同一个 ResNet。
    - depth backbone 为 1-channel ResNet，第一层权重参考 Kuavo ACT 用 RGB conv1 权重通道均值初始化。
    - 阶段三第一版保留 DECO 原生“两路视觉 token”结构，但把原生 `img1/img2` 语义替换为 `fused_rgb/fused_depth`。
    - 暂不采用单路 `visual_tokens: [B, L, D]` 重构方案；该方案作为后续 RGB-D 主链路稳定后的 ablation 或二期重构候选。
  - 在 `0.2 已确认技术决策` 中扩展 RGB 增强池：
    - 在 Kuavo ACT 既有 Identity/Notransform、ColorJitter、SharpnessJitter、RandomMask、RandomBorderCutout、GaussianNoise、GammaCorrection 基础上，新增 GaussianBlur/RandomGaussianBlur 候选。
    - 记录默认采样权重参考 Kuavo ACT：Identity/Notransform `3.0`，其他增强 `1.0`，默认每次采样一个增强，保证一部分样本保持原图。
    - 记录若需要贴近 DECO 原生 GaussianBlur 行为，应新增 `RandomGaussianBlur`，被采样后随机选择 `kernel_size in [3,5,7]` 与 `sigma in [0.1,2.0]`。
  - 在 `0.2 已确认技术决策` 中新增 `observation.state` 决策：
    - state 归一化采用 Kuavo/LeRobot preprocessor 与 dataset stats。
    - DECO wrapper 内不再沿用 DECO 原生 `dataset.py` / `inference.py` 的手动二次归一化。
    - 归一化后的 28 维 state 仍通过 DECO `obs_encoder` 后加到 time embedding，用于调制 MMAttention，不改成 ACT state token / VAE encoder 路线。
    - `observation.tactile` 独立进入 tactile encoder / cross-attention / PI_Adapter，不与 state 混拼。
  - 在 `0.2 已确认技术决策` 与阶段三/四任务中新增 tactile 处理边界：
    - 洗数据脚本中的 `normal_force / 100` 只表示单位换算，把 Kuavo 原始 normal force 转成牛顿。
    - 进入 DECO 模型前，触觉按 DECO-style 左/右手 tactile max 归一化，再进入 Kuavo 30 维 tactile encoder。
    - `observation.tactile` 不应被 LeRobot 当作普通 STATE 走 `MEAN_STD`；后续实现需通过 patch、preprocessor 或 wrapper 适配隔离 tactile 归一化路径。
  - 在 `3.1 Kuavo RGB-D 视觉前端移植` 中补充：
    - RGB/depth 独立 backbone。
    - cross attention 后保留 `fused_rgb_tokens` 与 `fused_depth_tokens` 两路 visual tokens。
    - DECO `MMAttention` 中 `total_img_len / 2` 的分流逻辑可保留，但必须注释说明当前两半分别表示 RGB stream 与 depth stream。
  - 在 `3.2 触觉编码器手术` 中补充：
    - `tactile_encoder` 从 `30D -> 34D`。
    - tactile fusion 维度为 `15 + 15 + 34 = 64`，原生 `gated=Linear(68,68)` 需改为 `Linear(64,64)`。
    - `tactile_left_max` / `tactile_right_max` 应基于已转换成牛顿的 Kuavo tactile 数据，不能直接复用 DECO Inspire Hand 原始单位下的 `3486/4050`。
  - 在 `3.3 DECO 主干保留策略`、阶段四 wrapper 与阶段五配置任务中明确：
    - Flow Matching loss 第一版严格保持 `F.mse_loss(out, noise - action)`。
    - 不额外使用 `action_is_pad` mask，以对齐 DECO 原生训练代码中“mask 返回但 diffusion loss 不消费 mask”的行为。
  - 在阶段三新增 `3.7 预处理与 state 接入边界`，细化后续实现 checklist。
  - 在阶段四 `DECOConfigWrapper` 与 `DECOPolicyWrapper.forward` 中补充确定性空间预处理、RGB 增强池、state 单次归一化和 tactile 独立读取要求。
  - 在阶段五 `configs/policy/deco_config.yaml` 任务中补充 `Resize/Letterbox` 配置、GaussianBlur/RandomGaussianBlur、增强权重和 state 接入说明。
  - 在 Done When 中新增针对视觉预处理边界和 state 接入方式的验收项。
- **修改文件 2**: `Content/DECO_Technical_Decisions.md`
  - 将最后更新时间更新为 `2026-05-15`。
  - 在总体原则中新增视觉预处理分层、RGB blur 增强、state 接入方式三项冻结原则。
  - 重写 `3.4 RGB 与 depth 的增强策略`：
    - 明确 `Resize/Letterbox` 是确定性空间预处理。
    - 明确默认采用 DECO 原生 `256x256 letterbox`，RGB 使用灰色 padding `128`，depth 使用独立 padding。
    - 明确 RGB 随机增强池新增 GaussianBlur/RandomGaussianBlur。
    - 明确 Kuavo-DECO 不直接复刻 DECO `p=0.5` blur 触发概率，而是将 blur 纳入 Kuavo 增强池；如果需要保留 DECO 内部参数分布，则新增 `RandomGaussianBlur`。
    - 明确 depth 不做 ColorJitter、Gamma、GaussianBlur、GaussianNoise、RandomMask、RandomBorderCutout 等 RGB photometric 或遮挡增强。
  - 更新 `4.1 视觉前端替换策略`：
    - 冻结阶段三第一版使用 `dual_stream_rgb_depth` visual token 路线。
    - RGB/depth 独立 ResNet，cross attention 后仍输出 `fused_rgb/fused_depth` 两路 visual tokens。
    - 新增两路 visual tokens 与单路 visual token 的 Pros/Contra 对比；单路方案暂不作为第一版。
  - 更新 `4.3 DECO 主干保留策略`：
    - 新增 loss 决策：保持 `F.mse_loss(out, noise - action)`，不额外使用 `action_is_pad` mask。
  - 新增 `4.5 obs/state 归一化与模型接入策略`：
    - 对比 DECO 原生手动归一化与 Kuavo/LeRobot preprocessor 归一化。
    - 对比 DECO `obs_encoder + time embedding` 条件注入与 ACT state token / VAE encoder 路线。
    - 冻结 Kuavo-DECO 方案：数值归一化跟 Kuavo/LeRobot，模型接入方式跟 DECO。
    - 补充 tactile 不走 state 路线：tactile 在牛顿单位基础上按 DECO-style 左/右手 tactile max 归一化后进入 tactile encoder / PI_Adapter。
  - 更新 `4.6 触觉模型手术`：
    - 明确 `/100` 是单位换算，不是 DECO tactile normalization。
    - 明确 Kuavo 30 维 tactile 可视为已经抽取好的触觉区域值。
    - 明确 tactile max 参数应来自已换算为牛顿的 Kuavo tactile 数据统计或配置。
  - 同步修正旧 depth 默认表述：
    - 默认 depth topic 更新为 `/cam_h/depth/image_raw/compressed`。
    - 默认 depth decoder 更新为 `compressed_image`。
    - `compressedDepth_png` 与 raw `16UC1` 保留为待 Inspector/validator 复核的候选路径。
  - 在待实现清单中新增 `configs/policy/deco_config.yaml` 对确定性预处理、RGB 增强池、state 接入策略的记录要求。
- **未执行项**:
  - 未运行 Python、训练、forward、validator、部署脚本或任何环境变更命令。
  - 未修改模型代码、wrapper 代码、第三方 DECO 源码或 `third_party/lerobot/`。
  - 本次仅做文档级方案同步与静态一致性检查。

## 2026-05-14

### 拆分 DECO 与 Kuavo ACT 流程说明文档
- **任务**: 根据用户要求，将 `DECO_ACT_Flow_Explanation.md` 拆分为三个独立文件，分别讲解 DECO 原生流程、Kuavo ACT 流程、两者对比及第三阶段流程，保持内容结构不变。
- **新增文件**:
  - `Content/DECO_Flow_Explanation.md`：包含文件开头说明、静态阅读范围及 DECO 原生整体流程部分。
  - `Content/Kuavo_ACT_Flow_Explanation.md`：包含文件开头说明及 Kuavo ACT RGB-D Wrapper 整体流程部分。
  - `Content/DECO_ACT_Comparison_Stage3_Flow.md`：包含文件开头说明、DECO 与 ACT 的关键差异、第三阶段目标流程及面向实现的结论部分。
- **未执行项**:
  - 根据系统约束要求，禁止使用命令删除原始文件，因此保留了原始的 `Content/DECO_ACT_Flow_Explanation.md`，请用户如需删除自行手动执行 `rm` 命令。

### 整理 DECO 与 Kuavo ACT 流程说明文档及流程图
- **任务**: 根据用户在进入阶段三前的学习需求，将此前对 DECO 原生源码流程和 Kuavo ACT RGB-D wrapper 流程的静态分析整理为可持续参考的 Markdown 文档，并配套生成流程图图片，帮助后续确认第三阶段模型适配边界。
- **新增文件**:
  - 新增 `Content/DECO_ACT_Flow_Explanation.md`：
    - 记录本次静态阅读范围，包括 `third_party/deco/dataset.py`、`third_party/deco/train.py`、`third_party/deco/models/deco/deco.py`、`third_party/deco/models/deco/img_encoder.py`、`third_party/deco/models/deco/train_one_epoch.py`、`third_party/deco/inference.py`、`kuavo_train/train_policy.py`、`kuavo_train/wrapper/policy/act/ACTPolicyWrapper.py`、`kuavo_train/wrapper/policy/act/ACTModelWrapper.py`、`kuavo_train/utils/transforms.py` 和 `configs/policy/act_config.yaml`。
    - 梳理 DECO 原生数据流：自定义 episode 数据结构、双 RGB 输入、触觉 `.npy`、`data.pkl` 中 state/action chunk、图像增强、状态/action/触觉归一化。
    - 说明 DECO 原生视觉前端实际为两个 RGB 输入共享一个 ResNet34，而不是两个独立 ResNet；`img_encoding` 将两个图像在 batch 维拼接，经过共享 ResNet34 后再拆分、展平为空间 token，并加入 camera id embedding 与 RoPE。
    - 说明 DECO 主干保留 action token、MMAttention、Flow Matching 加噪、`F.mse_loss(out, noise - action)` 训练目标，以及推理阶段从随机 action noise 多步去噪得到 action chunk 的逻辑。
    - 梳理 Kuavo ACT RGB-D wrapper 流程：LeRobot batch、preprocessor、RGB_Augmenter、Normalizer、OBS_IMAGES/OBS_DEPTH 组装、RGB ResNet、1-channel depth ResNet、RGB-depth cross attention fusion、ACT transformer encoder/decoder、L1+KL loss 与 `save_pretrained` 保存体系。
    - 对比 DECO 与 ACT 在数据格式、视觉输入、ResNet 数量、depth 支持、增强策略、视觉 token、动作建模、loss、触觉和保存方式上的差异。
    - 总结第三阶段实现边界：替换 DECO 原生 `img_encoding(img1, img2)`，借鉴 ACT RGB-D 前端和 cross-modal fusion，保留 DECO action-token Flow Matching 主干与 tactile PI_Adapter 思路。
  - 新增 `Content/assets/deco_act_flow/deco_original_flow.svg`：
    - 以静态 SVG 图展示 DECO 原生训练与推理数据流。
  - 新增 `Content/assets/deco_act_flow/kuavo_act_rgbd_flow.svg`：
    - 以静态 SVG 图展示 Kuavo ACT RGB-D wrapper 的数据流和融合路径。
  - 新增 `Content/assets/deco_act_flow/stage3_deco_rgbd_target_flow.svg`：
    - 以静态 SVG 图展示阶段三 Kuavo RGB-D 前端接入 DECO 主干的目标结构。
- **实现说明**:
  - 文档中同时保留 Mermaid 源码和 SVG 图片引用；Mermaid 便于后续修改逻辑，SVG 便于普通 Markdown 预览器直接显示。
  - SVG 图片为手写静态矢量图，不依赖外部渲染工具或运行脚本。
- **未执行项**:
  - 未运行 Python、训练、forward、validator 或部署脚本。
  - 未安装或更新任何依赖。
  - 未修改 `third_party/lerobot/`、`third_party/deco/`、训练 wrapper 或模型代码。

## 2026-05-13

### 执行阶段二：复制 DECO 源码到 third_party/deco 并记录依赖边界
- **任务**: 根据用户确认，执行 Kuavo-DECO 阶段二工具链整合。用户明确要求保留根目录 `DECO/` 不变，后续采用复制而不是移动；若 `.DS_Store` 跟随目录复制，则目标副本中不应保留该文件。
- **复制结果**:
  - 新增 `third_party/deco/`，作为 DECO 第三方源码副本，与现有 `third_party/lerobot/` 并列。
  - 源码副本来自根目录 `DECO/`，包括 `models/`、`config/`、`deploy/`、`assets/`、`utils/`、`train.py`、`dataset.py`、`inference.py`、`README.MD`、`LICENSE`、`requirements.txt` 与 `ACTION_SPACE_ANATOMY.md` 等内容。
  - 保留根目录 `DECO/` 原样不动，后续 Kuavo 集成默认以 `third_party/deco/` 作为接入源，根目录 `DECO/` 仅作为原始参考副本。
  - 复制时排除 `.DS_Store`，并静态确认 `third_party/deco/` 下不存在 `.DS_Store`。
- **文档修改**:
  - 修改 `PLANS.md`：
    - 将阶段二从“物理迁移/移动”改为“复制归档”。
    - 将 `2.1` 标记为完成，记录保留 `DECO/`、复制到 `third_party/deco/`、不修改 `third_party/lerobot/`、不保留 `.DS_Store` 的约束。
    - 将 `2.2` 标记为完成，记录后续 wrapper 通过将 `third_party/deco` 加入 `sys.path` 来兼容 DECO 原始 `from models.xxx` 导入方式；阶段二只做路径约定，不写 wrapper。
  - 修改 `README_DECO.md`：
    - 新增“阶段二：源码复制与路径约定”章节。
    - 说明 `DECO/` 与 `third_party/deco/` 的职责边界。
    - 说明 ACT/DP 当前来自 LeRobot submodule，而 Kuavo 的策略适配通过 `kuavo_train/wrapper/policy/...` 完成，DECO 后续也沿用 wrapper 接入模式。
    - 记录后续 wrapper 推荐的 `sys.path` 注入方式。
  - 修改 `requirements_DECO.txt`：
    - 记录 `third_party/deco/requirements.txt` 中的 DECO 上游原始依赖 pin。
    - 对照当前 Kuavo/LeRobot 依赖文件，说明多数核心依赖已由现有环境覆盖。
    - 明确当前阶段不安装依赖，不强制切换 `torch`、`torchvision`、`diffusers`、`huggingface-hub` 等核心库版本。
- **未执行项**:
  - 未运行 Python 脚本、未执行训练、未执行 forward、未运行 validator。
  - 未安装或升级任何依赖。
  - 未修改 `third_party/lerobot/`。
  - 未实现 `kuavo_train/wrapper/policy/deco/`，该部分保留到阶段三/四。

### 新增 DECO LeRobot 数据集验证脚本并推进 1.7
- **任务**: 根据用户提供的已转换示例数据集 `data_example/lerobot`，推进 `PLANS.md` 阶段一 1.7 数据一致性检查。当前机器仍遵守 No-Runtime 约束，不运行 Python validator；用户将在可运行环境执行并反馈结果。
- **静态查看结果**:
  - `data_example/lerobot` 目录包含 LeRobot v3 数据结构：`meta/info.json`、`meta/stats.json`、`meta/tasks.parquet`、`data/chunk-000/file-000.parquet`、RGB/depth video 文件。
  - `meta/info.json` 中 `codebase_version` 为 `v3.0`，`fps` 为 `30`，`total_episodes` 为 `1`，`total_frames` 为 `331`。
  - metadata 中存在 `observation.images.head_cam_h`、`observation.depth_h`、`observation.state`、`observation.tactile`、`action`。
  - metadata shape 符合当前方案：state `(28,)`、action `(28,)`、tactile `(30,)`、RGB/depth video `(3, 480, 640)`。
- **新增文件**:
  - 新建 `kuavo_data/validate_deco_lerobot_dataset.py`：
    - 不依赖 ROS1、rospy、rosbag 或 kuavo_msgs，只验证已经转换好的 LeRobot 数据集。
    - 基础检查读取 `meta/info.json`，验证 codebase、fps、episode/frame/task 数、必需 feature、shape 和 feature names。
    - 文件结构检查验证 data parquet、episode metadata parquet、tasks parquet、stats json、RGB/depth mp4 是否存在。
    - 若环境中存在 `numpy/pandas/pyarrow`，进一步读取 parquet 检查 timestamp 约 30Hz、state/action/tactile 维度与有限值、head action 是否全零、head state 是否近似固定、tactile 是否异常全零。
    - 若环境中存在 `cv2`，进一步打开 RGB/depth video 首帧，检查首帧尺寸是否为 640×480。
    - 输出 Markdown report，默认写入 `<root>/deco_validation_report.md`；存在 FAIL 时进程返回非零退出码。
- **同步文档**:
  - 修改 `README_DECO.md`：
    - 增加 validator 运行命令：`python kuavo_data/validate_deco_lerobot_dataset.py --root data_example/lerobot --report data_example/lerobot/deco_validation_report.md`。
    - 增加 `--metadata-only` 命令，用于缺少 parquet/video 依赖时先做基础结构检查。
  - 修改 `PLANS.md`：
    - 在 1.7 中记录用户已提供 `data_example/lerobot` 作为首个检查对象。
    - 将“新建 validator 脚本”和“静态查看 metadata”标记为完成。
    - 保持 1.7 整体未完成，等待用户运行 validator 并反馈结果后再确认。
- **未执行项**:
  - 未运行 `kuavo_data/validate_deco_lerobot_dataset.py`。
  - 未读取 parquet 数值内容。
  - 未打开 mp4 视频做帧级检查。

### 修复 depth topic 名称与解码器不匹配问题（首次试跑报错修复）

- **问题描述**: 首次在真实 rosbag 上运行 `CvtRosbag2Lerobot_DECO.py` 时报错 `rosbag 缺少 DECO 必需数据：['observation.depth_h']`。经用户提供 rosbag info 截图确认，实际 rosbag 中的 depth topic 为 `/cam_h/depth/image_raw/compressed`（消息类型 `sensor_msgs/CompressedImage`），而非配置中写的 `/cam_h/depth/image_raw/compressedDepth`（`sensor_msgs/CompressedDepth`）。
- **根因分析**: 两种消息格式的二进制布局不同：
  - `CompressedDepth`: `msg.data` 前面有若干字节的配置头（quantization info 等），需要先搜索 PNG magic header (`\x89PNG`) 跳过前缀才能解码。这是原 ACT/DP 脚本 `process_depth_image` 的做法。
  - `CompressedImage`: `msg.data` 就是标准的 PNG/JPEG 缓冲区，直接 `cv2.imdecode` 即可。
- **修改文件 1**: `configs/data/KuavoRosbag2Lerobot_deco.yaml`
  - `deco.depth_topic`: 从 `/cam_h/depth/image_raw/compressedDepth` 改为 `/cam_h/depth/image_raw/compressed`
  - `deco.depth_encoding`: 从 `compressedDepth_png` 改为 `compressed_image`
  - 增加中文注释说明两种格式的区别
- **修改文件 2**: `kuavo_data/CvtRosbag2Lerobot_DECO.py`
  - 在 `DecoRosbagReader` 类中新增方法 `process_compressed_depth_image(self, msg)`：
    - 使用 `np.frombuffer(msg.data, np.uint8)` + `cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)` 直接解码
    - 包含 `dtype != np.uint16` 的警告检查
    - 使用 `cv2.INTER_NEAREST` resize 到 `(kuavo.RESIZE_W, kuavo.RESIZE_H)`
  - 在 `build_topic_process_map` 方法的 `depth_encoding` 判断中新增 `compressed_image` 分支，指向新方法
  - 错误提示文案更新为"当前支持 compressedDepth_png、compressed_image 或 raw_16uc1"
- **未修改文件**: `kuavo_data/common/kuavo_dataset.py`（守住公共 reader 不动的约束）


### 完善 DECO 数据 YAML 的路径语义与验证约定
- **任务**: 根据用户要求，对比原始 `configs/data/KuavoRosbag2Lerobot.yaml`、训练配置和部署配置后，完善 `configs/data/KuavoRosbag2Lerobot_deco.yaml`。用户明确要求不要额外显式展开 topic 字段，因此本次只补充路径语义和后续验证约定，不修改 topic map。
- **修改内容**:
  - 修改 `configs/data/KuavoRosbag2Lerobot_deco.yaml`：
    - 为 `rosbag.rosbag_dir`、`rosbag.num_used`、`rosbag.lerobot_dir` 增加中文注释。
    - 将 `rosbag.lerobot_dir` 示例从 `/your/path/to/your/lerobotdata_deco/` 调整为 `/your/path/to/your/lerobotdata_deco/lerobot`，使其直接对应后续训练配置中的 `root` 目录，减少训练时路径填错。
    - 注释说明：如果使用绝对路径，应直接指向最终 LeRobot dataset root；如果使用相对路径，脚本会按 `<rosbag_dir>/../<lerobot_dir>/lerobot` 生成，以兼容原 ACT/DP 转换脚本目录习惯。
    - 新增 `validation` 配置块，作为后续 `validate_deco_lerobot_dataset.py` 的检查约定，包含 `expected_train_hz: 30`、RGB/depth key、state/action/tactile 维度、是否要求触觉、是否要求 depth、是否要求头部 state 均值与头部 action 补零。
  - 修改 `README_DECO.md`：
    - 将手动运行命令中的 `rosbag.lerobot_dir` 示例同步为 `/path/to/output_lerobot_deco/lerobot`。
    - 补充说明该路径应对应后续训练配置 `root` 读取的最终目录；相对路径仍按旧脚本习惯展开。
- **未修改内容**:
  - 未新增 `state_topic`、`hand_state_topic`、`arm_traj_topic` 等显式 topic 字段，遵守用户“topic 不要显示写”的要求。
  - 未修改 `kuavo_data/CvtRosbag2Lerobot_DECO.py`。
  - 未运行 Python、未执行 rosbag 转换、未执行训练或 validator。

### 执行阶段一：新增 DECO 数据配置与 rosbag 转 LeRobot 转换脚本
- **任务**: 根据用户确认，开始执行阶段一数据引擎任务，新增 DECO 专用 YAML 与 `_DECO` 洗数据脚本。遵守当前机器 No-Runtime 约束，只做静态代码修改与逻辑审查，不运行 Python、不转换 rosbag、不训练模型。
- **新增文件**:
  - 新建 `configs/data/KuavoRosbag2Lerobot_deco.yaml`：
    - 默认 `dataset.train_hz: 30`、`dataset.use_depth: true`、`dataset.eef_type: qiangnao`、`dataset.which_arm: both`、`dataset.dex_dof_needed: 6`。
    - 默认 RGB topic 为 `/cam_h/color/image_raw/compressed`。
    - 默认 depth topic 为 `/cam_h/depth/image_raw/compressedDepth`，默认 `depth_encoding: compressedDepth_png`。
    - 将 `/camera/depth/image_rect_raw` 与 `raw_16uc1` 仅保留为配置化 fallback，并设置 `allow_raw_depth_fallback: false`，防止未经复核时替代现有 ACT/DP depth 链路。
    - 记录 DECO 固定 schema：`state_dim: 28`、`action_dim: 28`、`tactile_dim: 30`。
    - 记录 arm action 优先级：`/kuavo_arm_traj_synced` -> `/kuavo_arm_traj` -> `/joint_cmd`。
    - 记录触觉 topic `/dexhand/touch_state` 与 `tactile_force_scale: 100.0`。
    - 记录头部策略：state 使用 `joint_q[26:28]` episode 均值，action 固定 `[0.0, 0.0]`。
  - 新建 `kuavo_data/CvtRosbag2Lerobot_DECO.py`：
    - 新增 DECO 专用 `DecoRosbagReader`，继承 `KuavoRosbagReader` 的 bag 列表与未索引 bag 打开保护逻辑，但不修改公共 `kuavo_data/common/kuavo_dataset.py`。
    - 保留原脚本关键保护：使用 bag time 覆盖 header stamp、跳过无法解码的 depth 帧后统一做必需字段检查、arm action gap 检测、action range clamp、失败 bag 写入 `error_DECO.txt`。
    - 使用真实时间戳生成 30Hz 目标时间轴，不依赖 `MAIN_TIMELINE_FPS // TRAIN_HZ` 整数跳帧假设，以兼容 100Hz 或更高频采集流。
    - 对 RGB、depth、state、hand state、arm action、hand action、tactile 使用 nearest-neighbor 对齐，并截取所有必需模态共同覆盖的时间段。
    - 构造 `observation.state` 28 维：左臂 `joint_q[12:19]`、左手 `/dexhand/state[:6]`、右臂 `joint_q[19:26]`、右手 `/dexhand/state[6:12]`、头部 episode 均值 `joint_q[26:28]`。
    - 构造 `action` 28 维：左/右臂优先使用 `/kuavo_arm_traj_synced` 或 `/kuavo_arm_traj`，断流或缺失时 fallback 到 `/joint_cmd`；左右手使用 `/control_robot_hand_position`；头部补零。
    - 构造 `observation.tactile` 30 维：左右手各 5 指、每指 `normal_force1/2/3`，除以 100 转为牛顿；触觉字段缺失时显式报错，不静默补零。
    - depth 默认复用 Kuavo compressedDepth PNG decoder；同时实现 `raw_16uc1` decoder 作为配置化候选。
    - 当前 LeRobot 磁盘 schema 中将 depth 保存为 3-channel depth image，以兼容 image/video writer；后续 wrapper 需按 depth 语义还原到 1-channel depth backbone。
- **同步文档**:
  - 修改 `PLANS.md`：
    - 将阶段一 `1.3`、`1.4`、`1.5`、`1.6` 与阶段五 `5.2` 标记为已完成。
    - 将 Done When 中 DECO data yaml 与 converter 静态审查项标记为已完成。
    - 修正动作空间描述，明确 raw `joint_q` 中机械臂来源为 `joint_q[12:19]` 和 `joint_q[19:26]`，手部来自 dexhand topic，而不是把手部误写成 `joint_q` 内部字段。
  - 修改 `Content/DECO_Technical_Decisions.md`：
    - 新增 `3.10 阶段一实现状态`，记录本次新增 YAML、转换脚本、depth 3-channel 兼容策略、raw depth fallback 状态和 No-Runtime 约束。
    - 将待实现清单中的数据 YAML 与 converter 项标记为已完成。
  - 修改 `README_DECO.md`：
    - 新增阶段一 RGB-D 数据转换说明、目标字段、关键配置、手动运行命令、运行后检查项和当前限制。
- **静态复核结论**:
  - 已静态确认当前 LeRobot 本地源码路径为 `lerobot.datasets.*`，脚本同时保留旧版 `lerobot.common.datasets.*` fallback。
  - 已静态确认本地 `KuavoRosbagReader` 的 bag 打开函数为 `load_raw_rosbag()`，bag 列表函数为 `list_bag_files()`，并在 DECO 脚本中对齐。
  - 已静态确认本地 LeRobot `validate_frame(frame, features)` 是工具函数，DECO 脚本使用该函数做 frame schema 检查。
- **未执行项**:
  - 未运行 `kuavo_data/CvtRosbag2Lerobot_DECO.py`。
  - 未转换任何 rosbag。
  - 未运行训练、forward、部署或 validator。

### 修正 depth 决策为优先复用现有 Kuavo 清洗链路
- **任务**: 根据用户反馈，修正此前关于 depth topic 的计划表述，避免把用户补充但可能不准确的 `/camera/depth/image_rect_raw` 直接冻结为默认方案；将 depth 相关策略调整为优先复用现有 ACT/DP 清洗脚本逻辑。
- **静态确认**:
  - 现有 `kuavo_data/common/kuavo_dataset.py` 中，头部 RGB 默认来自 `/cam_h/color/image_raw/compressed`。
  - 现有 depth 默认来自 `/cam_h/depth/image_raw/compressedDepth`，处理函数为 `process_depth_image`，其逻辑是在 compressedDepth 数据中定位 PNG header，再用 `cv2.imdecode(..., cv2.IMREAD_UNCHANGED)` 解码为 `uint16` depth。
  - 现有 `CvtRosbag2Lerobot.py` 中，state 来自 `/sensors_data_raw`，arm action 先读 `/joint_cmd`，随后被 `/kuavo_arm_traj` 或 `/kuavo_arm_traj_synced` 覆盖，hand action 来自 `/control_robot_hand_position`。
- **修改内容**:
  - 修改 `PLANS.md`：
    - 在 `0.2 已确认技术决策` 中明确 depth topic 与 decoder 默认继承 Kuavo 现有 ACT/DP 清洗链路：`/cam_h/depth/image_raw/compressedDepth` 与 compressedDepth PNG 解码逻辑。
    - 将 `/camera/depth/image_rect_raw`、`encoding=16UC1` 降级为待 Inspector/validator 复核的 raw depth fallback，不作为当前冻结默认值。
    - 在阶段一 `1.3` 和 `1.4` 中补充：DECO converter 优先复用现有 `KuavoRosbagReader` 的 topic map、message processor 和 nearest-neighbor 对齐思路；仅在 30Hz 目标时间轴、DECO 固定 schema、30 维触觉、28 维 state/action、depth 单通道语义处做专用适配。
    - 在 `1.4` 中新增两类 depth decoder 计划：默认 `compressedDepth_png`，候选 `raw_16uc1`。
    - 在 `1.6` 中记录 arm action 继承现有清洗逻辑：优先 `/kuavo_arm_traj_synced`，否则 `/kuavo_arm_traj`，`/joint_cmd` 只作为 fallback 或一致性检查；hand action 使用 `/control_robot_hand_position`；DECO 固定使用左右手各 6 DoF，不沿用 `dex_dof_needed: 1` 压缩策略。
    - 在 `1.7` validation checkpoint 中新增 depth 来源、编码和 decoder 检查。
    - 在 `5.2` 数据配置任务中新增默认 `rgb_topic`、`depth_topic`、`depth_encoding`，并说明 raw `16UC1` topic 只是候选。
  - 修改 `Content/DECO_Technical_Decisions.md`：
    - 在总体原则中新增“数据清洗实现向 Kuavo 现有链路靠齐”。
    - 在数据配置与 RGB-D 视觉源策略中明确默认 depth topic 为 `/cam_h/depth/image_raw/compressedDepth`，默认 decoder 为 compressedDepth PNG 解码。
    - 将 `/camera/depth/image_rect_raw` / `raw_16uc1` 记录为未冻结候选，必须由 Inspector/validator 复核后启用。
    - 在 depth 增强策略中说明旧脚本存在将 depth 归一化为 `uint8` 并 repeat 成 3 通道的兼容做法；Kuavo-DECO 主路线仍保留 1-channel depth backbone 语义，除非 wrapper/config 明确声明兼容模式。
    - 在 action 来源表中同步现有脚本逻辑：`/kuavo_arm_traj_synced` 优先，其次 `/kuavo_arm_traj`，`/joint_cmd` fallback；hand action 使用 `/control_robot_hand_position`。
- **目的**:
  - 避免在 depth topic 仍不确定时引入新的 raw depth 路线，降低阶段一数据转换风险。
  - 最大限度复用已经服务 ACT/DP 的 Kuavo 清洗链路，只在 DECO 不同的 schema、频率、触觉和模型输入语义上做必要改造。
  - 保留 raw `16UC1` 支持的扩展口，但通过配置和 validation 明确它不是当前默认路径。

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
