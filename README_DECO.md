# Kuavo-DECO 使用指导

本文档是一份在 `kuavo_data_challenge` 工具链中使用 DECO 的 instruction。当前 DECO 不是原仓库一开始就内置的策略，而是后续合并到 Kuavo 跨端工具链中的模型链路；因此它需要同时遵守 Kuavo/LeRobot 的数据、训练、部署资产组织方式，以及 DECO 自身的 RGB-D、Flow Matching、tactile adapter 约束。

本文只面向 Linux 主线环境，不考虑 Windows。推荐按下面顺序推进：

```text
安装依赖与 ROS 环境
  -> rosbag 清洗为 LeRobot 数据集
  -> 第一阶段 visual_main 训练 RGB-D + state 主干
  -> 可选第二阶段 tactile_adapter 训练 tactile PI_Adapter
  -> 使用 configs/deploy/kuavo_deco_env.yaml 部署
```

## 1. 安装与配置

### 1.1 代码位置

在仓库根目录下操作：

```bash
cd /path/to/kuavo_data_challenge
```

与 DECO 相关的主要路径如下：

| 路径 | 用途 |
| --- | --- |
| `requirements_DECO.txt` | Linux pip-only 依赖入口。 |
| `configs/data/KuavoRosbag2Lerobot_deco.yaml` | DECO 数据清洗配置。 |
| `kuavo_data/CvtRosbag2Lerobot_DECO.py` | DECO rosbag 到 LeRobot 数据集转换脚本。 |
| `configs/policy/deco_config.yaml` | DECO 训练配置。 |
| `kuavo_train/train_policy.py` | Kuavo 单机训练入口。 |
| `configs/deploy/kuavo_deco_env.yaml` | DECO 唯一部署配置入口。 |
| `third_party/deco/` | 已收编的 DECO 源码副本，当前只保留 DECO 主链路。 |
| `DECO/` | 原始 DECO 参考副本，不作为 Kuavo 集成主入口。 |

### 1.2 Python 依赖

DECO 环境安装只使用一份 requirements：

```bash
pip install -r requirements_DECO.txt
```

`requirements_DECO.txt` 的边界是 pip 可安装的 Python 包与本仓库 editable 包。不要直接用 `requirements_total.txt` 作为 DECO pip 安装来源，因为其中包含很多 ROS 分发包或系统包，例如 `rospy`、`rosbag`、`cv_bridge`、`sensor_msgs` 等，这些不应由 pip 负责安装。

### 1.3 ROS 与系统前置条件

只安装 pip 依赖还不够。数据清洗、仿真和实机部署依赖 ROS Noetic 与 Kuavo 消息环境，运行前应由系统或 ROS workspace 提供：

```bash
source /opt/ros/noetic/setup.bash
source /path/to/your/kuavo_ros_ws/devel/setup.bash
```

需要确认的外部能力包括：

- ROS Noetic 可用，并能导入 `rospy`、`rosbag`、`cv_bridge`。
- `sensor_msgs`、`std_msgs`、`geometry_msgs`、`std_srvs`、`kuavo_msgs` 等消息包已由 ROS 环境提供。
- 训练机器具备与 `torch==2.7.1` / `torchvision==0.22.1` 匹配的 CUDA 或 CPU 运行环境。
- 若使用实机部署，Kuavo SDK、机器人 ROS 驱动、控制话题和相机/depth/tactile 话题必须与部署 YAML 一致。

## 2. 数据处理

### 2.1 运行入口

从仓库根目录运行 DECO 专用清洗脚本：

```bash
python kuavo_data/CvtRosbag2Lerobot_DECO.py \
  rosbag.rosbag_dir=/path/to/rosbag_dir \
  rosbag.lerobot_dir=/path/to/output_lerobot_deco/lerobot
```

脚本默认读取的配置文件是：

```text
configs/data/KuavoRosbag2Lerobot_deco.yaml
```

长期使用时建议直接修改这份 YAML；临时测试时可以像上面一样用 Hydra override 覆盖单个字段。

### 2.2 必须人工调整的字段

| 字段 | 什么时候要改 | 说明 |
| --- | --- | --- |
| `rosbag.rosbag_dir` | 每次换数据都要改 | 输入 rosbag 目录，目录内应包含一个或多个 `.bag` 文件。 |
| `rosbag.num_used` | 只想转换部分 bag 时改 | `null` 表示使用目录下全部 bag。 |
| `rosbag.lerobot_dir` | 每次指定输出数据集时改 | 推荐写最终 LeRobot dataset root，例如 `/data/task_x_deco/lerobot`。 |
| `dataset.task_description` | 任务变化时改 | 写入 LeRobot task metadata。 |
| `dataset.eef_type` | 末端执行器类型变化时改 | 可填 `qiangnao`、`leju_claw`、`rq2f85`。 |
| `deco.end_effector_profile` | 一般保持 `auto` | 手动指定时必须与 `dataset.eef_type` 匹配。 |
| `deco.write_tactile` | 只有强脑灵巧手数据需要考虑 | `qiangnao_tactile` 可写入 tactile；二夹爪 profile 会强制忽略 tactile。 |
| `deco.depth_topic` / `deco.depth_encoding` | depth topic 或封装格式与默认不一致时改 | 默认是 `/cam_h/depth/image_raw/compressed` + `compressed_image`。 |
| `deco.overwrite` | 确认覆盖已有输出目录时改 | 默认 `false`，避免误删或覆盖已有数据。 |

### 2.3 末端执行器选项

`dataset.eef_type` 是给用户最主要的选择入口：

- `qiangnao`：强脑灵巧手。`deco.end_effector_profile=auto` 会映射为 `qiangnao_tactile`，输出 28D `observation.state` / `action`，可选写入 30D `observation.tactile`。
- `leju_claw`：Leju 二指夹爪。`auto` 会映射为 `gripper_no_tactile`，输出 18D state/action，不写入 tactile。
- `rq2f85`：Robotiq/RQ2F85 二指夹爪或仿真夹爪。`auto` 同样映射为 `gripper_no_tactile`，输出 18D state/action，不写入 tactile。

`deco.end_effector_profile` 可选：

- `auto`：推荐值，根据 `dataset.eef_type` 自动推导。
- `qiangnao_tactile`：显式指定强脑灵巧手 profile，要求 `dataset.eef_type=qiangnao`。
- `gripper_no_tactile`：显式指定二夹爪无触觉 profile，要求 `dataset.eef_type=leju_claw` 或 `rq2f85`。

输出维度由 profile 决定，不要手动制造新的维度组合：

```text
qiangnao_tactile:
  left arm 7 + left hand 6 + right arm 7 + right hand 6 + head 2 = 28D
  tactile = left hand 15 + right hand 15 = 30D

gripper_no_tactile:
  left arm 7 + left gripper 1 + right arm 7 + right gripper 1 + head 2 = 18D
  no observation.tactile
```

### 2.4 Depth 选项

默认 depth 配置是：

```yaml
deco:
  depth_topic: /cam_h/depth/image_raw/compressed
  depth_encoding: compressed_image
```

`depth_encoding` 可填：

- `compressed_image`：普通 `sensor_msgs/CompressedImage`，`data` 可直接图像解码。当前默认。
- `compressedDepth_png`：ROS compressedDepth 封装，需要定位 PNG payload 后解码。
- `auto`：先尝试 `compressed_image`，失败后尝试 `compressedDepth_png`。
- `raw_16uc1`：raw depth fallback，只应在 `allow_raw_depth_fallback=true` 且对应 topic 已复核时使用。

第一版数据集为了兼容当前 LeRobot image/video writer，会把 depth 保存为 3-channel depth image；DECO wrapper 在训练时再取单通道送入 1-channel depth backbone。这个存储策略不是严格 metric depth 保真方案，后续如果要保留毫米尺度，需要同步升级 converter、validator、训练 config 和部署入口。

### 2.5 不建议随意修改的字段

下面字段在 YAML 中保留主要是为了说明 DECO 固定约束，不是普通用户开关：

- `dataset.only_arm: true`：DECO 当前只训练上半身操作。
- `dataset.which_arm: both`：DECO 28D/18D schema 都是双臂。
- `dataset.use_depth: true`：Kuavo-DECO 当前是 RGB-D 路线，不是纯 RGB 路线。
- `dataset.train_hz: 30`：训练数据统一为 30Hz；部署 10Hz 由 wrapper/action queue 处理。
- `dataset.dex_dof_needed: 6`：强脑灵巧手使用左右手各 6 DoF，不沿用 ACT/DP 单维开合量。
- `dataset.delta_action: false`、`dataset.relative_start: false`：当前 DECO 输出绝对 joint/action schema。

### 2.6 转换后检查

转换完成后，建议在允许执行的环境中运行 validator：

```bash
python kuavo_data/validate_deco_lerobot_dataset.py \
  --root /path/to/output_lerobot_deco/lerobot \
  --report /path/to/output_lerobot_deco/lerobot/deco_validation_report.md
```

二夹爪无触觉数据集建议显式指定：

```bash
python kuavo_data/validate_deco_lerobot_dataset.py \
  --root /path/to/gripper_lerobot/lerobot \
  --end-effector-profile gripper_no_tactile \
  --no-require-tactile
```

如果当前环境缺少 parquet/video 依赖，可以先只检查 metadata 和文件结构：

```bash
python kuavo_data/validate_deco_lerobot_dataset.py \
  --root /path/to/output_lerobot_deco/lerobot \
  --metadata-only
```

## 3. 模型训练

### 3.1 训练入口与配置文件

DECO 训练使用 Kuavo 原有训练入口：

```text
kuavo_train/train_policy.py
```

训练配置文件是：

```text
configs/policy/deco_config.yaml
```

从仓库根目录启动训练的基本形式：

```bash
python kuavo_train/train_policy.py \
  --config-path=../configs/policy/ \
  --config-name=deco_config.yaml \
  task=your_task_name \
  method=deco_visual_main \
  root=/path/to/output_lerobot_deco/lerobot \
  policy_name=deco \
  training.batch_size=16
```

常用字段含义：

| 字段 | 说明 |
| --- | --- |
| `task` | 任务名，会进入 `outputs/train/<task>/...`。 |
| `method` | 方法名，建议区分 `deco_visual_main`、`deco_tactile_adapter`、`deco_gripper` 等训练分支。 |
| `root` | 数据清洗输出的 LeRobot dataset root，必须与 `rosbag.lerobot_dir` 对齐。 |
| `policy_name` | DECO 必须填 `deco`。 |
| `training.batch_size` | 根据显存调整。 |
| `training.max_epoch` | 训练 epoch 数。 |
| `policy.end_effector_profile` | 必须与数据集 schema 一致。 |
| `policy.action_dim` | `qiangnao_tactile` 用 28，`gripper_no_tactile` 用 18。 |

训练输出采用 Kuavo run-root 结构：

```text
outputs/train/<task>/<method>/run_<timestamp>/
  policy_preprocessor.json
  policy_postprocessor.json
  epoch<epoch>/
    config.json
    model.safetensors
```

部署和迁移时要保留整个 `run_<timestamp>/` 目录；单独拷贝 `epochbest/` 或某个 `epoch<epoch>/` 不是完整部署资产。

### 3.2 第一阶段：visual_main 主干训练

第一阶段训练 RGB-D + state + DECO Flow Matching 主干，不使用 tactile 分支：

```yaml
policy:
  training_stage: visual_main
  use_tactile: false
  use_tactile_lora: false
```

强脑灵巧手无触觉主干训练使用 28D：

```bash
python kuavo_train/train_policy.py \
  --config-path=../configs/policy/ \
  --config-name=deco_config.yaml \
  task=your_task_name \
  method=deco_visual_main \
  root=/path/to/qiangnao_lerobot/lerobot \
  policy_name=deco \
  policy.end_effector_profile=qiangnao_tactile \
  policy.action_dim=28 \
  policy.training_stage=visual_main \
  policy.use_tactile=false \
  policy.use_tactile_lora=false
```

二夹爪无触觉训练使用 18D，并且不能进入 tactile adapter：

```bash
python kuavo_train/train_policy.py \
  --config-path=../configs/policy/ \
  --config-name=deco_config.yaml \
  task=your_task_name \
  method=deco_gripper_visual_main \
  root=/path/to/gripper_lerobot/lerobot \
  policy_name=deco \
  policy.end_effector_profile=gripper_no_tactile \
  policy.action_dim=18 \
  policy.training_stage=visual_main \
  policy.use_tactile=false \
  policy.use_tactile_lora=false \
  policy.load_external_init_weights=false
```

如果不使用触觉，训练到这里即可结束。部署时直接使用第一阶段 run 根目录和选定 epoch。

### 3.3 第二阶段：tactile_adapter 训练

第二阶段是可选项，只适用于强脑灵巧手触觉路线：

```yaml
policy:
  end_effector_profile: qiangnao_tactile
  action_dim: 28
  training_stage: tactile_adapter
  use_tactile: true
  use_tactile_lora: true
  freeze_pretrained_main: true
  base_policy_path: /path/to/stage1/run_xxx/epoch<epoch>
  tactile_left_max: 1.0
  tactile_right_max: 1.0
```

启动示例：

```bash
python kuavo_train/train_policy.py \
  --config-path=../configs/policy/ \
  --config-name=deco_config.yaml \
  task=your_task_name \
  method=deco_tactile_adapter \
  root=/path/to/qiangnao_lerobot/lerobot \
  policy_name=deco \
  policy.end_effector_profile=qiangnao_tactile \
  policy.action_dim=28 \
  policy.training_stage=tactile_adapter \
  policy.use_tactile=true \
  policy.use_tactile_lora=true \
  policy.base_policy_path=/path/to/outputs/train/your_task_name/deco_visual_main/run_xxx/epochbest \
  policy.tactile_left_max=1.0 \
  policy.tactile_right_max=1.0
```

这里的 `use_tactile_lora` 对应 DECO 源码中的 `plugin` / `PI_Adapter` 低秩 adapter，不是外部 PEFT LoRA。第二阶段会加载第一阶段选定 epoch 的 policy 权重，冻结已经匹配的主干参数，训练 tactile encoder、tactile cross-attention、PI_Adapter 和必要桥接参数。

`tactile_left_max` 与 `tactile_right_max` 的单位是清洗后 `normal_force / 100` 的牛顿值。正式触觉训练时必须填正数，可以来自训练集最大值、分位数上限或人工审定安全上限；不要在 `use_tactile=true` 时保留 `null`。

### 3.4 什么时候关闭 LoRA / tactile adapter

下面情况应关闭 `use_tactile_lora`，并保持 `training_stage=visual_main`：

- 使用 `gripper_no_tactile`，因为二夹爪数据没有 `observation.tactile`。
- 只想训练或部署 RGB-D + state 主干，不想使用触觉。
- 还没有完成稳定的第一阶段主干训练。
- 触觉数据质量、量纲或 `tactile_left_max/right_max` 还没有确认。
- 做无触觉 ablation，希望与 ACT/DP 或 DECO visual_main 公平对比。
- 部署加载的是第一阶段 `use_tactile=false` 的 checkpoint。

如果不使用二阶段触觉训练，只需要在 `configs/policy/deco_config.yaml` 或命令行 override 中保持：

```yaml
policy:
  training_stage: visual_main
  use_tactile: false
  use_tactile_lora: false
  base_policy_path: null
  adapter_model_path: null
```

二阶段不是 `train_policy.py` 自动从第一阶段切到第二阶段，也不是同一个 epoch loop 里自动切换配置。它是第二次独立启动训练。

### 3.5 不同 profile 的训练配置对齐

| 数据 profile | 训练 profile | `action_dim` | tactile | 是否有第二阶段 |
| --- | --- | --- | --- | --- |
| `qiangnao_tactile` | `qiangnao_tactile` | 28 | 第一阶段关闭，第二阶段可开启 | 可选 |
| `gripper_no_tactile` | `gripper_no_tactile` | 18 | 必须关闭 | 不允许 |

如果数据集是二夹爪 18D，但训练仍使用默认 `action_dim=28`，wrapper 应报错；如果强行绕过这类检查，state/action normalizer、模型输入输出层和部署 action 解码都会不一致。

## 4. 部署

### 4.1 部署配置入口

DECO 部署只使用：

```text
configs/deploy/kuavo_deco_env.yaml
```

通用 `configs/deploy/kuavo_env.yaml` 保持 ACT/DP 语义，不承载 DECO 的 depth、tactile、28D/18D state layout 配置。

必须人工调整的字段：

| 字段 | 说明 |
| --- | --- |
| `env.eef_type` | `qiangnao`、`leju_claw` 或 `rq2f85`，必须与训练数据和 checkpoint 语义一致。 |
| `env.state_layout` | `deco_28d` 用于强脑灵巧手，`deco_18d` 用于二夹爪。 |
| `deco.inference_mode` | `qiangnao_tactile`、`qiangnao_no_tactile` 或 `gripper_no_tactile`。 |
| `deco.head_state_source` | `live_joint_q` 或 `fixed_config`。 |
| `inference.policy_type` | 本地推理填 `deco`；server/client 调用侧填 `client`。 |
| `inference.task` | 对应 `outputs/train/<task>/`。 |
| `inference.method` | 对应 `outputs/train/<task>/<method>/`。 |
| `inference.timestamp` | 对应 run 目录名，例如 `run_20260518_120000`。 |
| `inference.epoch` | 选择 `epoch<epoch>`，例如 `best`、`50`、`100`。 |
| `inference.device` | `cuda` 或 `cpu`。 |
| `obs_key_map.depth_h` | 在线 depth topic 或 encoding 与训练/数据配置不一致时才改。 |

### 4.2 三种部署模式

| `deco.inference_mode` | 需要的 checkpoint | 配套配置 |
| --- | --- | --- |
| `qiangnao_no_tactile` | 第一阶段 visual_main，`use_tactile=false` | `env.eef_type=qiangnao`，`env.state_layout=deco_28d`。 |
| `qiangnao_tactile` | 第二阶段 tactile_adapter，`use_tactile=true`、`use_tactile_lora=true` | `env.eef_type=qiangnao`，`env.state_layout=deco_28d`，订阅 tactile。 |
| `gripper_no_tactile` | 18D visual_main，`end_effector_profile=gripper_no_tactile` | `env.eef_type=leju_claw` 或 `rq2f85`，`env.state_layout=deco_18d`。 |

部署配置只负责选择与校验 checkpoint，不应强行覆盖 checkpoint 中保存的模型结构字段。若 checkpoint 的 `action_dim/use_tactile/use_tactile_lora/end_effector_profile` 与部署 YAML 不一致，应修正 YAML 或换正确权重。

### 4.3 本地真机或 dry-run 入口

本地单进程真机推理沿用 Kuavo 原有脚本：

```bash
python kuavo_deploy/src/scripts/script.py \
  --task run \
  --config configs/deploy/kuavo_deco_env.yaml
```

常用 task：

- `run`：从当前位置直接运行模型。
- `go_run`：先按 go bag 到达工作位置，再运行模型。
- `here_run`：插值到 bag 最后一帧状态后运行。
- `back_to_zero`：中断后倒放 bag 回零。

真正上实机前建议先使用 dry-run 或低速限幅策略检查 action 范围、方向、频率和左右映射。

### 4.4 仿真自动测试入口

仿真自动测试使用：

```bash
python kuavo_deploy/src/scripts/script_auto_test.py \
  --task auto_test \
  --config configs/deploy/kuavo_deco_env.yaml
```

优先验证无触觉主干：

```yaml
deco:
  inference_mode: qiangnao_no_tactile
```

或二夹爪：

```yaml
deco:
  inference_mode: gripper_no_tactile
```

触觉 adapter 建议在 visual_main 闭环稳定后再进入：

```yaml
deco:
  inference_mode: qiangnao_tactile
```

### 4.5 Server / Client 模式

本机调试时建议只绑定 loopback，避免把推理服务暴露到局域网：

```bash
python kuavo_deploy/kuavo_service/server.py \
  --config configs/deploy/kuavo_deco_env.yaml \
  --host 127.0.0.1 \
  --port 5555
```

也可以通过环境变量选择配置；此时 server 默认仍只绑定本机：

```bash
export KUAVO_DEPLOY_CONFIG=configs/deploy/kuavo_deco_env.yaml
python kuavo_deploy/kuavo_service/server.py --port 5555
```

如果确实需要跨机器访问，必须显式提供 token：

```bash
python kuavo_deploy/kuavo_service/server.py \
  --config configs/deploy/kuavo_deco_env.yaml \
  --host 0.0.0.0 \
  --port 5555 \
  --api-token "$KUAVO_INFERENCE_API_TOKEN"
```

client 侧在 `configs/deploy/kuavo_deco_env.yaml` 中设置 `inference.policy_type=client`、
`client_host`、`client_port`、`client_timeout_ms`；若 server 使用 token，则将
`client_api_token_env` 设置为环境变量名，例如 `KUAVO_INFERENCE_API_TOKEN`。

当前 server/client 遵守 Kuavo ACT 原版语义：

```text
eval/client side:
  raw obs -> run-root preprocessor -> PolicyClient.select_action

server side:
  processed observation -> policy.select_action -> raw model action

eval/client side:
  run-root postprocessor -> env.step
```

因此 server 不接管 raw ROS observation，不执行图像 letterbox，不执行 normalizer，也不执行 postprocessor。不要把 preprocessor/postprocessor 同时放到 client 和 server 两侧，否则会产生重复归一化或重复反归一化。

## 5. 推荐工作流

### 5.1 强脑灵巧手，无触觉

```text
数据:
  dataset.eef_type=qiangnao
  deco.end_effector_profile=auto
  deco.write_tactile=false 或 true 均可；训练会关闭 tactile

训练:
  policy.end_effector_profile=qiangnao_tactile
  policy.action_dim=28
  policy.training_stage=visual_main
  policy.use_tactile=false
  policy.use_tactile_lora=false

部署:
  env.eef_type=qiangnao
  env.state_layout=deco_28d
  deco.inference_mode=qiangnao_no_tactile
```

### 5.2 强脑灵巧手，带触觉 adapter

```text
数据:
  dataset.eef_type=qiangnao
  deco.end_effector_profile=auto
  deco.write_tactile=true

第一阶段训练:
  visual_main
  use_tactile=false
  use_tactile_lora=false

第二阶段训练:
  tactile_adapter
  use_tactile=true
  use_tactile_lora=true
  base_policy_path=/path/to/stage1/run_xxx/epoch<epoch>
  tactile_left_max/right_max 填正数

部署:
  env.eef_type=qiangnao
  env.state_layout=deco_28d
  deco.inference_mode=qiangnao_tactile
```

### 5.3 二夹爪无触觉

```text
数据:
  dataset.eef_type=leju_claw 或 rq2f85
  deco.end_effector_profile=auto

训练:
  policy.end_effector_profile=gripper_no_tactile
  policy.action_dim=18
  policy.training_stage=visual_main
  policy.use_tactile=false
  policy.use_tactile_lora=false
  policy.load_external_init_weights=false

部署:
  env.eef_type=leju_claw 或 rq2f85
  env.state_layout=deco_18d
  deco.inference_mode=gripper_no_tactile
```

## 6. 常见误配

- 不要把 `requirements_total.txt` 当作 DECO pip 环境入口；DECO 使用 `requirements_DECO.txt`。
- 不要把 ROS Noetic 包写进 pip requirements；ROS 包由 apt 或 ROS workspace 提供。
- 不要混用 28D 数据和 18D policy 配置。
- 不要在 `gripper_no_tactile` 下开启 `use_tactile` 或 `use_tactile_lora`。
- 不要在没有第一阶段稳定 checkpoint 的情况下直接训练 tactile adapter。
- 不要在 `tactile_left_max/right_max` 仍为 `null` 或没有量纲确认时开启 `use_tactile=true`。
- 不要只拷贝 `epochbest/` 做部署；完整部署资产是整个 `run_<timestamp>/` 目录。
- 不要把部署 `deco.inference_mode` 当作脚本选择器；实际入口仍由运行的 `script.py`、`script_auto_test.py` 或 `server.py` 决定。
- 不要把 `train_hz=30` 改成部署频率；部署 10Hz 由 `control_hz=10` 和 `action_stride=3` 处理。
- 不要让 server/client 两侧都执行 preprocessor 或 postprocessor。

## 7. 当前验证边界

本仓库中的 DECO 数据、训练、部署代码已经完成静态接入与文档同步。实际项目落地时仍应在目标 Linux/ROS 环境中补做：

- rosbag 转换后的 validator 检查。
- 第一阶段 `visual_main` 的训练与 checkpoint 保存检查。
- 无触觉仿真或 dry-run 闭环。
- 若使用触觉，再进行第二阶段 tactile adapter 训练与低风险部署验证。
- 上实机前的 action 范围、左右映射、10Hz 控制节奏、RGB-depth 对齐和 tactile 非零/饱和检查。
