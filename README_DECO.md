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

| 路径                                         | 用途                                             |
| -------------------------------------------- | ------------------------------------------------ |
| `requirements_DECO.txt`                      | Linux pip-only 依赖入口。                        |
| `configs/data/KuavoRosbag2Lerobot_deco.yaml` | DECO 数据清洗配置。                              |
| `kuavo_data/CvtRosbag2Lerobot_DECO.py`       | DECO rosbag 到 LeRobot 数据集转换脚本。          |
| `configs/policy/deco_config.yaml`            | DECO 训练配置。                                  |
| `kuavo_train/train_policy.py`                | Kuavo 单机训练入口。                             |
| `configs/deploy/kuavo_deco_env.yaml`         | DECO 唯一部署配置入口。                          |
| `third_party/deco/`                          | 已收编的 DECO 源码副本，当前只保留 DECO 主链路。 |

### 1.2 Python 依赖

环境初始化:
```bash
conda create -n kuavo_deco python=3.10 -y
conda activate kuavo_deco
conda install -c conda-forge "ffmpeg=7.*" libstdcxx-ng -y

# DECO 环境安装只使用一份 requirements：
python -m pip install -r requirements_DECO.txt
```

环境检查:
```bash
python -c "import torch; print(torch.__version__)"
python -c "import torchcodec; print(torchcodec.__file__)"
python -c "from torchcodec.decoders import VideoDecoder; print('torchcodec ok')"
```

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

### 2.2 必须人工调整的参数

| 字段                        | 参数含义               | 说明                                                                 |
| --------------------------- | ---------------------- | -------------------------------------------------------------------- |
| `rosbag.rosbag_dir`         | rosbag 输入目录        | 目录内应包含一个或多个 `.bag` 文件                                   |
| `rosbag.num_used`           | 转录 rosbag 数目       | `null` 表示使用目录下全部 bag                                        |
| `rosbag.lerobot_dir`        | rosbag 输出目录        | 推荐写最终 LeRobot dataset root，例如 `/data/task_x_deco/lerobot`    |
| `dataset.task_description`  | 任务名称描述           | 写入 LeRobot task metadata                                           |
| `dataset.eef_type`          | 末端执行器类型         | 可填 `qiangnao`、`leju_claw`、`rq2f85`。                             |
| `deco.end_effector_profile` | 一般保持 `auto`        | 手动指定时必须与 `dataset.eef_type` 匹配                             |
| `deco.write_tactile`        | 是否转录触觉信息的开关 | `qiangnao_tactile` 可写入 tactile；二夹爪 profile 会强制忽略 tactile |
| `deco.overwrite`            | 覆盖已有输出目录       | 默认 `false`，避免误删或覆盖已有数据                                 |

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

这里的 depth_topic 是首选 depth topic，不是唯一 topic。转换脚本会根据当前 bag 中实际存在的 topic 构造候选并按顺序选择：
1. 优先使用 depth_topic + depth_encoding。
2. 如果首选 topic 以 /compressed 结尾，脚本会额外尝试对应的 /compressedDepth + compressedDepth_png。
3. 如果首选 topic 以 /compressedDepth 结尾，脚本会额外尝试对应的 /compressed + compressed_image。

为了兼容当前 LeRobot image/video writer，会把 depth 保存为 3-channel depth image；DECO wrapper 在训练时再取单通道送入 1-channel depth backbone。这个存储策略不是严格 metric depth 保真方案，后续如果要保留毫米尺度，需要同步升级 converter、训练 config 和部署入口。

### 2.5 不建议随意修改的字段

下面字段在 YAML 中保留主要是为了说明 DECO 固定约束，不是普通用户开关：

- `dataset.only_arm: true`：DECO 当前只训练上半身操作。
- `dataset.which_arm: both`：DECO 28D/18D schema 都是双臂。
- `dataset.use_depth: true`：Kuavo-DECO 当前是 RGB-D 路线，不是纯 RGB 路线。
- `dataset.train_hz: 30`：训练数据统一为 30Hz；部署 10Hz 由 wrapper/action queue 处理。
- `dataset.dex_dof_needed: 6`：强脑灵巧手使用左右手各 6 DoF，不沿用 ACT/DP 单维开合量。
- `dataset.delta_action: false`、`dataset.relative_start: false`：当前 DECO 输出绝对 joint/action schema。


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

### 3.2 配置文件中的建议可调参数说明

`deco_config.yaml`中:

- save path & training 参数:

| 字段                        | 说明                                                                                     |
| --------------------------- | ---------------------------------------------------------------------------------------- |
| `task`                      | 任务名称，会进入 `outputs/train/<task>/...`。                                            |
| `method`                    | 方法名称，建议区分 `deco_visual_main`、`deco_tactile_adapter`、`deco_gripper` 等训练分支 |
| `root`                      | 数据清洗输出的 LeRobot dataset root，必须与 `rosbag.lerobot_dir` 对齐                    |
| `policy_name`               | 固定值，默认`deco`，也兼容大写，即 `DECO`                                                |
| `training.max_epoch`        | 训练 epoch 数                                                                            |
| `training.save_freq_epoch`  | 每隔多少个 epoch 保存一次额外 checkpoint                                                 |
| `training.batch_size`       | 根据显存调整                                                                             |
| `training.num_workers`      | DataLoader 的 worker 数量                                                                |
| `training.resume`           | 是否启用恢复训练逻辑，默认false，表示从头开始训练                                        |
| `training.resume_timestamp` | 仅当`resume=true`时生效，指向续接训练的权重                                              |
| `training.RGB_Augmenter`    | 调整图像增强的算法池和权重                                                               |

- policy 参数:

| 字段                           | 说明                                                                                                 |
| ------------------------------ | ---------------------------------------------------------------------------------------------------- |
| `policy.training_stage`        | 训练阶段。visual_main 是第一阶段 RGB-D + state 主干训练；tactile_adapter 是第二阶段触觉 adapter 微调 |
| `policy.use_tactile`           | 是否把 tactile 输入模型。`false` 时即使数据集有 tactile，模型 forward 也不会读                       |
| `policy.use_tactile_lora`      | 是否启用 DECO 自实现的 tactile PI_Adapter/plugin                                                     |
| `policy.end_effector_profile`  | 数据 schema 类型。选填 `qiangnao_tactile` ；`gripper_no_tactile`                                     |
| `policy.chunk_size`            | 预测多少步未来 action                                                                                |
| `policy.action_dim`            | action 的维度。`qiangnao_tactile` 必须是 28；`gripper_no_tactile` 必须是 18                          |
| `policy.vision_backbone`       | RGB 分支 backbone，选填 `resnet34`(default)；`resnet18`                                              |
| `policy.depth_backbone`        | Depth 分支 backbone。选填 `resnet34`(default)；`resnet18`                                            |
| `policy.normalization_mapping` | 各模态正则化选项                                                                                     |
| `policy.tactile_left_max`      | 左手 tactile 归一化最大值，匹配灵巧手触觉量程(单位:N)                                                |
| `policy.tactile_right_max`     | 右手 tactile 归一化最大值，匹配灵巧手触觉量程(单位:N)                                                |

`policy.load_external_init_weights=false` 和其之后的三个参数，提供外部初始化权重入口，允许外部 `.pth` 格式权重进来作为训练初始化，但它不是唯一入口，也不是推荐主路径。

### 3.3 第一阶段：visual_main 主干训练

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

### 3.4 第二阶段：tactile_adapter 训练

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
  policy.tactile_left_max=25 \
  policy.tactile_right_max=25
```

这里的 `use_tactile_lora` 对应 DECO 源码中的 `plugin` / `PI_Adapter` 低秩 adapter，不是外部 PEFT LoRA。第二阶段会加载第一阶段选定 epoch 的 policy 权重，冻结已经匹配的主干参数，训练 tactile encoder、tactile cross-attention、PI_Adapter 和必要桥接参数。

`tactile_left_max` 与 `tactile_right_max` 的单位是清洗后 `normal_force / 100` 的牛顿值，正式触觉训练时必须填正数，可以来自训练集最大值、分位数上限、人工审定安全上限和灵巧手自身量程

### 3.5 什么时候关闭 LoRA / tactile adapter

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

### 3.6 不同 profile 的训练配置对齐

| 数据 profile         | 训练 profile         | `action_dim` | tactile                      | 是否有第二阶段 |
| -------------------- | -------------------- | ------------ | ---------------------------- | -------------- |
| `qiangnao_tactile`   | `qiangnao_tactile`   | 28           | 第一阶段关闭，第二阶段可开启 | 可选           |
| `gripper_no_tactile` | `gripper_no_tactile` | 18           | 必须关闭                     | 不允许         |

如果数据集是二夹爪 18D，但训练仍使用默认 `action_dim=28`，wrapper 应报错；如果强行绕过这类检查，state/action normalizer、模型输入输出层和部署 action 解码都会不一致。

## 4. 部署

### 4.1 部署配置入口

DECO 部署只使用：

```text
configs/deploy/kuavo_deco_env.yaml
```

通用 `configs/deploy/kuavo_env.yaml` 保持 ACT/DP 语义，不承载 DECO 的 depth、tactile、28D/18D state layout 配置。

必须人工调整的字段：

| 字段                     | 说明                                                                      |
| ------------------------ | ------------------------------------------------------------------------- |
| `env.eef_type`           | `qiangnao`、`leju_claw` 或 `rq2f85`，必须与训练数据和 checkpoint 语义一致 |
| `env.state_layout`       | `deco_28d` 用于强脑灵巧手，`deco_18d` 用于二夹爪                          |
| `env.depth_h`            | `compressedDepth` 用于模拟，`compressed` 用用于实机部署                   |
| `deco.inference_mode`    | `qiangnao_tactile`、`qiangnao_no_tactile` 或 `gripper_no_tactile`         |
| `deco.runtime_mode`      | `local_real`、`local_sim`、`server` 或 `dry_run`                          |
| `deco.head_state_source` | `live_joint_q` 或 `fixed_config` ，固定头部自由度时，默认`fixed_config`   |
| `inference.policy_type`  | 本地推理填 `deco`；server/client 调用侧填 `client`                        |
| `inference.task`         | 对应 `outputs/train/<task>/`                                              |
| `inference.method`       | 对应 `outputs/train/<task>/<method>/`                                     |
| `inference.timestamp`    | 对应 run 目录名，例如 `run_20260518_120000`                               |
| `inference.epoch`        | 选择 `epoch<epoch>`，例如 `best`、`50`、`100`                             |
| `inference.device`       | `cuda` 或 `cpu`                                                           |




### 4.2 模拟部署

模拟部署沿用 Kuavo 原有交互入口：本仓库运行 `eval_kuavo.py`，另一个 Kuavo Mujoco 仿真器仓库负责启动仿真环境；两端通过 ROS topic、service 和 action command 互通。DECO 在这条链路中只替换 policy 推理部分，整体流程仍然是：

```text
Mujoco simulator / ROS topics
  -> kuavo_deploy 读取 RGB-D、state、gripper/tactile
  -> run-root preprocessor
  -> CustomDECOPolicyWrapper.select_action
  -> run-root postprocessor
  -> KuavoSimEnv.step 下发仿真动作
```

```bash
# to unset realworld kuavo setup in . bashrc
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311
export ROS_IP=127.0.0.1
unset ROS_HOSTNAME

cd kuavo_data_challenge
conda activate kuavo_deco

#启动方式与原 Kuavo 仿真测试一致：
python kuavo_deploy/eval_kuavo.py
```

交互步骤：

1. 先在另一个 Kuavo Mujoco 仿真器仓库中启动 simulator。
2. 在本仓库运行上面的 `eval_kuavo.py`。
3. 菜单中选择 `3. Task Selection Menu`。
4. 配置文件路径输入 `configs/deploy/kuavo_deco_env.yaml`。
5. 选择 `8. auto_test`，进入仿真自动测试。

二夹爪无触觉模型：

```yaml
env:
  env_name: Kuavo-Sim
  eef_type: rq2f85        # 若仿真器使用 Leju 爪，则改为 leju_claw
  state_layout: deco_18d
  which_arm: both
  only_arm: true
  depth_h: ["/cam_h/depth/image_raw/compressedDepth", "CompressedImage", 30, *IMGSIZE, *DEPTHRANGE, "compressedDepth_png"]

deco:
  inference_mode: gripper_no_tactile
  runtime_mode: local_sim

inference:
  policy_type: deco
  task: your_task
  method: your_method
  timestamp: run_YYYYMMDD_HHMMSS
  epoch: best
```

强脑无触觉模型：

```yaml
env:
  env_name: Kuavo-Sim
  eef_type: qiangnao
  state_layout: deco_28d
  which_arm: both
  only_arm: true
  qiangnao_dof_needed: 6
  depth_h: ["/cam_h/depth/image_raw/compressedDepth", "CompressedImage", 30, *IMGSIZE, *DEPTHRANGE, "compressedDepth_png"]

deco:
  inference_mode: qiangnao_no_tactile
  runtime_mode: local_sim

inference:
  policy_type: deco
  task: your_task
  method: your_method
  timestamp: run_YYYYMMDD_HHMMSS
  epoch: best
```

强脑触觉 Adapter：

```yaml
env:
  env_name: Kuavo-Sim
  eef_type: qiangnao
  state_layout: deco_28d
  which_arm: both
  only_arm: true
  qiangnao_dof_needed: 6
  depth_h: ["/cam_h/depth/image_raw/compressedDepth", "CompressedImage", 30, *IMGSIZE, *DEPTHRANGE, "compressedDepth_png"]

deco:
  inference_mode: qiangnao_tactile
  runtime_mode: local_sim

inference:
  policy_type: deco
  task: your_task
  method: your_method
  timestamp: run_YYYYMMDD_HHMMSS
  epoch: best
```

仿真器发布的 RGB、depth、`/sensors_data_raw`、末端执行器状态 topic 必须与 `env.obs_key_map` 中的配置一致。若仿真器 depth 使用 `compressedDepth` 封装，需要同步修改 `obs_key_map.depth_h` 的 topic 和 `depth_encoding`。

### 4.3 实际部署

实际部署使用同一份 `configs/deploy/kuavo_deco_env.yaml`，核心差异是把环境切到真机，并确认真机 ROS topic、末端执行器和 checkpoint profile 一致：

```yaml
env:
  env_name: Kuavo-Real

deco:
  runtime_mode: local_real
```

本地真机推理沿用 Kuavo 原有脚本：

```bash
python kuavo_deploy/src/scripts/script.py \
  --task run \
  --config configs/deploy/kuavo_deco_env.yaml
```

也可以根据任务需要使用：

- `go_run`：先按 `inference.go_bag_path` 到达工作位置，再运行模型。
- `here_run`：插值到 bag 最后一帧状态后运行模型。
- `back_to_zero`：中断后倒放 bag 回零。

二夹爪无触觉模型：

```yaml
env:
  env_name: Kuavo-Real
  eef_type: leju_claw     # 若真机使用 RQ2F85，则改为 rq2f85
  state_layout: deco_18d
  which_arm: both
  only_arm: true
  depth_h: ["/cam_h/depth/image_raw/compressed", "CompressedImage", 30, *IMGSIZE, *DEPTHRANGE, "compressed_image"]

deco:
  inference_mode: gripper_no_tactile
  runtime_mode: local_real

inference:
  policy_type: deco
  go_bag_path: /path/to/your/go.bag
  task: your_task
  method: your_method
  timestamp: run_YYYYMMDD_HHMMSS
  epoch: best
```

强脑无触觉模型：

```yaml
env:
  env_name: Kuavo-Real
  eef_type: qiangnao
  state_layout: deco_28d
  which_arm: both
  only_arm: true
  qiangnao_dof_needed: 6
  depth_h: ["/cam_h/depth/image_raw/compressed", "CompressedImage", 30, *IMGSIZE, *DEPTHRANGE, "compressed_image"]

deco:
  inference_mode: qiangnao_no_tactile
  runtime_mode: local_real

inference:
  policy_type: deco
  go_bag_path: /path/to/your/go.bag
  task: your_task
  method: your_method
  timestamp: run_YYYYMMDD_HHMMSS
  epoch: best
```

强脑触觉 Adapter：

```yaml
env:
  env_name: Kuavo-Real
  eef_type: qiangnao
  state_layout: deco_28d
  which_arm: both
  only_arm: true
  qiangnao_dof_needed: 6
  depth_h: ["/cam_h/depth/image_raw/compressed", "CompressedImage", 30, *IMGSIZE, *DEPTHRANGE, "compressed_image"]

deco:
  inference_mode: qiangnao_tactile
  runtime_mode: local_real

inference:
  policy_type: deco
  go_bag_path: /path/to/your/go.bag
  task: your_task
  method: your_method
  timestamp: run_YYYYMMDD_HHMMSS
  epoch: best
```

实际部署前需要额外核对 `env.obs_key_map`：真机 RGB、depth、`/sensors_data_raw`、夹爪或灵巧手状态、触觉 topic 是否与 YAML 完全一致。Server / Client 模式可以用于边侧机推理：机器人侧作为 client 采集观测并执行动作，边侧机或 GPU 机器作为 server 运行 DECO policy。

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

