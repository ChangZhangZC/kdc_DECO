# Kuavo-DECO 使用说明

本文档说明当前 `kuavo_data_challenge` 仓库中的 DECO 数据转换、训练与部署链路。
当前实现已经收敛为固定三视角 RGB 方案，不再使用历史 RGB-D 前端：

```text
Kuavo rosbag
  -> 30 Hz LeRobot 数据集
  -> head / left wrist / right wrist 三路 RGB + state + 可选 tactile
  -> DECO Visual Main + Action-Token Flow Matching
  -> 28D 灵巧手或 18D 二夹爪 action chunk
  -> Receding Horizon 或 Temporal Ensembling 动作分发
```

当前主线只面向 Linux、ROS Noetic 与 Kuavo ROS 环境，不考虑 Windows。

## 1. 当前架构边界

### 1.1 固定输入与模型结构

- 视觉输入固定为三路 RGB，顺序不可交换：
  1. `observation.images.head_cam_h`
  2. `observation.images.wrist_cam_l`
  3. `observation.images.wrist_cam_r`
- 当前 DECO policy 不读取 depth。已有 LeRobot 数据集即使保留 depth 或其他相机字段，也不会把它们送入 DECO processor、normalizer 或模型。
- 模型保留 DECO 的 Action-Token Flow Matching 主干，视觉 backbone 可选择 `resnet34` 或 `resnet18`，默认 `resnet34`。
- 当前只支持双臂、双末端执行器和头部，不支持单臂、腿部、腰部或底盘 schema。
- 当前不使用 task conditioning。数据集中的 task 只是 LeRobot 格式要求的兼容占位，不进入模型。

### 1.2 固定 state/action profile

| Profile | 末端执行器 | State/Action | Tactile | 训练阶段 |
| --- | --- | ---: | ---: | --- |
| `qiangnao_tactile` | 强脑灵巧手 | 28D | 可选 30D | `visual_main`；可选 `tactile_adapter` |
| `gripper_no_tactile` | Leju Claw 或 RQ2F85 | 18D | 不支持 | 仅 `visual_main` |

28D 顺序：

```text
[0:7]   left arm
[7:13]  left hand
[13:20] right arm
[20:26] right hand
[26:28] head yaw / pitch
```

18D 顺序：

```text
[0:7]   left arm
[7]     left gripper
[8:15]  right arm
[15]    right gripper
[16:18] head yaw / pitch
```

转换脚本输出绝对 joint/action，不支持 delta action 或 relative-start action。头部 observation 与 action 分别从逐帧对齐的 `/sensors_data_raw` 和 `/joint_cmd` 中读取，不使用固定填充或相互复制。

## 2. 代码与环境

### 2.1 主要文件

| 路径 | 用途 |
| --- | --- |
| `requirements_DECO.txt` | DECO 的 Linux pip 依赖入口 |
| `configs/data/KuavoRosbag2Lerobot_deco.yaml` | DECO 数据转换配置 |
| `kuavo_data/CvtRosbag2Lerobot_DECO.py` | Kuavo rosbag 到 LeRobot 数据集的转换入口 |
| `configs/policy/deco_config.yaml` | DECO 训练配置 |
| `kuavo_train/train_policy.py` | Kuavo policy 训练入口 |
| `kuavo_train/wrapper/policy/deco/` | DECO 的 LeRobot config、policy、processor 与动作分发适配层 |
| `configs/deploy/kuavo_deco_env.yaml` | DECO 仿真、真机和 client 部署配置入口 |
| `kuavo_deploy/utils/deco_obs_action.py` | DECO 在线观测拼接、动作解码与兼容性校验 |
| `third_party/deco/` | DECO 模型主体源码 |

不要直接修改 `third_party/lerobot/`。Kuavo 对 LeRobot 的自定义兼容逻辑应继续放在 `lerobot_patches/` 中。

### 2.2 Python 与 ROS 环境

示例环境初始化：

```bash
conda create -n kuavo_deco python=3.10 -y
conda activate kuavo_deco
conda install -c conda-forge "ffmpeg=7.*" libstdcxx-ng -y
python -m pip install -r requirements_DECO.txt
```

数据转换、仿真和实机部署还需要 ROS Noetic 与 Kuavo 消息环境：

```bash
source /opt/ros/noetic/setup.bash
source /path/to/your/kuavo_ros_ws/devel/setup.bash
```

运行前应确认：

- 可以导入 `rospy`、`rosbag` 与 `cv_bridge`；
- `sensor_msgs`、`std_msgs`、`geometry_msgs`、`std_srvs` 与 `kuavo_msgs` 可用；
- 机器人 ROS 驱动、相机、末端执行器和触觉 topic 与配置一致；
- 训练设备满足 `requirements_DECO.txt` 中 PyTorch 版本的运行要求。

## 3. 数据转换

### 3.1 配置

编辑：

```text
configs/data/KuavoRosbag2Lerobot_deco.yaml
```

主要字段：

| 字段 | 说明 |
| --- | --- |
| `rosbag.rosbag_dir` | 包含一个或多个 `.bag` 文件的输入目录 |
| `rosbag.num_used` | 转换 bag 数量；`null` 表示全部 |
| `rosbag.lerobot_dir` | LeRobot dataset root 输出目录 |
| `dataset.eef_type` | `qiangnao`、`leju_claw` 或 `rq2f85` |
| `dataset.train_hz` | 输出数据集帧率，当前默认 30 Hz |
| `dataset.sample_drop` | 从 head RGB 主时间轴首尾丢弃的原始帧数 |
| `dataset.dex_dof_needed` | 固定 schema 信息，当前只允许 `6` |
| `deco.end_effector_profile` | 推荐 `auto`，根据 `eef_type` 推导 28D/18D profile |
| `deco.rgb_keys` | 固定三视角的短 key，顺序必须为 head、left wrist、right wrist |
| `deco.rgb_topics` | 三路 RGB 对应的 rosbag topic |
| `deco.write_tactile` | 强脑数据是否写入 30D tactile；二夹爪会强制忽略 |
| `deco.overwrite` | 是否允许覆盖已有输出目录，默认 `false` |

默认 RGB topic：

```yaml
deco:
  rgb_keys:
    - head_cam_h
    - wrist_cam_l
    - wrist_cam_r
  rgb_topics:
    head_cam_h: /cam_h/color/image_raw/compressed
    wrist_cam_l: /cam_l/color/image_raw/compressed
    wrist_cam_r: /cam_r/color/image_raw/compressed
```

所有模态以 head RGB 为主时间轴进行最近邻对齐。当前转换器不读取、对齐或写入 depth。

### 3.2 运行

从仓库根目录执行：

```bash
python kuavo_data/CvtRosbag2Lerobot_DECO.py \
  rosbag.rosbag_dir=/path/to/rosbag_dir \
  rosbag.lerobot_dir=/path/to/output_lerobot_deco/lerobot
```

长期参数建议写入 YAML；临时测试可以通过 Hydra override 覆盖单个字段。

### 3.3 Profile 选择

强脑灵巧手数据：

```yaml
dataset:
  eef_type: qiangnao

deco:
  end_effector_profile: auto
  write_tactile: true
```

Leju Claw 或 RQ2F85 数据：

```yaml
dataset:
  eef_type: leju_claw  # 或 rq2f85

deco:
  end_effector_profile: auto
  write_tactile: false
```

`auto` 的推导规则如下：

- `qiangnao` -> `qiangnao_tactile` -> 28D state/action；
- `leju_claw` 或 `rq2f85` -> `gripper_no_tactile` -> 18D state/action。

不要自行组合其他 action 维度。转换、训练、normalizer、checkpoint 与部署 action decoder 必须共享同一 profile。

## 4. 模型训练

### 4.1 基本入口

训练配置：

```text
configs/policy/deco_config.yaml
```

启动示例：

```bash
python kuavo_train/train_policy.py \
  --config-path=../configs/policy/ \
  --config-name=deco_config.yaml \
  task=your_task_name \
  method=deco_visual_main \
  root=/path/to/lerobot_dataset \
  policy_name=deco
```

`policy_name` 固定使用 `deco`。训练入口会从数据集 metadata 读取真实 fps，并把它保存为 checkpoint 的 `dataset_hz`。

主要训练参数：

| 字段 | 说明 |
| --- | --- |
| `task` | 输出目录中的任务名 |
| `method` | 输出目录中的方法名 |
| `root` | LeRobot dataset root |
| `training.max_epoch` | 最大训练 epoch |
| `training.save_freq_epoch` | 额外 checkpoint 保存间隔 |
| `training.batch_size` | batch size |
| `training.num_workers` | DataLoader worker 数量 |
| `training.resume` | 是否精确恢复同一次训练 run |
| `training.resume_timestamp` | `resume=true` 时指向原 `run_<timestamp>` 目录名 |
| `training.deco_init_pth_path` | 仅用于 `visual_main` 的历史 DECO `.pth` warm start |
| `training.RGB_Augmenter` | 三路 RGB 共用的增强配置 |

### 4.2 Profile 与训练阶段契约

当前训练配置只要求用户选择：

```yaml
policy:
  end_effector_profile: qiangnao_tactile
  training_stage: visual_main
```

以下字段由 wrapper 固定派生，不应再作为常规命令行参数手工组合：

| `training_stage` | `action_dim` | `use_tactile` | `use_tactile_lora` | `freeze_pretrained_main` |
| --- | ---: | ---: | ---: | ---: |
| `visual_main` + `qiangnao_tactile` | 28 | `false` | `false` | `false` |
| `visual_main` + `gripper_no_tactile` | 18 | `false` | `false` | `false` |
| `tactile_adapter` + `qiangnao_tactile` | 28 | `true` | `true` | `true` |

`gripper_no_tactile` 不能进入 `tactile_adapter`。训练入口会再次强制阶段契约，防止旧 checkpoint 字段或 Hydra override 产生非法组合。

### 4.3 第一阶段：Visual Main

强脑灵巧手无触觉主干：

```bash
python kuavo_train/train_policy.py \
  --config-path=../configs/policy/ \
  --config-name=deco_config.yaml \
  task=your_task_name \
  method=deco_visual_main \
  root=/path/to/qiangnao_lerobot/lerobot \
  policy.end_effector_profile=qiangnao_tactile \
  policy.training_stage=visual_main
```

二夹爪无触觉主干：

```bash
python kuavo_train/train_policy.py \
  --config-path=../configs/policy/ \
  --config-name=deco_config.yaml \
  task=your_task_name \
  method=deco_gripper_visual_main \
  root=/path/to/gripper_lerobot/lerobot \
  policy.end_effector_profile=gripper_no_tactile \
  policy.training_stage=visual_main
```

从头训练时保持：

```yaml
training:
  deco_init_pth_path: null

policy:
  load_external_init_weights: false
  base_policy_path: null
  adapter_model_path: null
```

若必须使用可信的历史 DECO `.pth` 做 Visual Main warm start：

```yaml
training:
  deco_init_pth_path: /path/to/trusted_init.pth

policy:
  load_external_init_weights: true
```

`.pth` 只用于历史兼容；新的 Kuavo-DECO 权重资产优先使用 LeRobot `.safetensors`。

### 4.4 第二阶段：Tactile Adapter

第二阶段只适用于 `qiangnao_tactile`，并且是一次独立训练启动，不会在同一个 epoch loop 中自动从第一阶段切换。

从已训练 Visual Main 新建 tactile adapter：

```yaml
policy:
  end_effector_profile: qiangnao_tactile
  training_stage: tactile_adapter
  load_external_init_weights: true
  base_policy_path: /path/to/stage1/run_xxx/epochbest
  adapter_model_path: null
  tactile_left_max: 25
  tactile_right_max: 25
```

从已有 tactile adapter checkpoint 开启一个新的、仅加载权重的训练 run：

```yaml
policy:
  training_stage: tactile_adapter
  load_external_init_weights: true
  base_policy_path: null
  adapter_model_path: /path/to/adapter/run_xxx/epochbest
```

`base_policy_path` 与 `adapter_model_path` 必须二选一。这里的 adapter 是 DECO 自身的 tactile plugin / PI_Adapter，不是外部 PEFT LoRA。

`tactile_left_max` 和 `tactile_right_max` 必须是正数，单位对应数据转换后 `normal_force / tactile_force_scale` 的值；默认 `tactile_force_scale=100.0`。Tactile 在 LeRobot normalizer 中保持 `IDENTITY`，wrapper 会按左右手最大值单独归一化。

### 4.5 Resume 与新训练的区别

精确继续同一个训练 run：

```yaml
training:
  resume: true
  resume_timestamp: run_YYYYMMDD_HHMMSS
```

Resume 会恢复 policy、optimizer、scheduler、AMP、RNG 与训练进度。它不能与 `load_external_init_weights`、`deco_init_pth_path`、`base_policy_path` 或 `adapter_model_path` 同时使用。

### 4.6 输出资产

训练输出结构：

```text
outputs/train/<task>/<method>/run_<timestamp>/
  config.json
  model.safetensors
  policy_preprocessor.json
  policy_postprocessor.json
  learning_state.pth
  rng_state.pth
  epochbest/
    config.json
    model.safetensors
  epoch<epoch>/
    config.json
    model.safetensors
```

部署和迁移时保留整个 `run_<timestamp>/`。根目录包含 preprocessor、postprocessor 和恢复训练状态；`epochbest/` 或 `epoch<epoch>/` 只保存对应 policy checkpoint。

## 5. 部署

### 5.1 唯一配置入口

DECO 部署使用：

```text
configs/deploy/kuavo_deco_env.yaml
```

当前配置不包含 depth。三路 RGB 的在线 key 必须能与 checkpoint 中的 `policy.rgb_keys` 一一对应。

主要字段：

| 字段 | 说明 |
| --- | --- |
| `env.env_name` | `Kuavo-Sim` 或 `Kuavo-Real` |
| `env.ros_rate` | 必须等于 checkpoint 的 `dataset_hz`，不是独立推理频率旋钮 |
| `env.obs_key_map` | 三路 RGB、state、末端执行器和可选 tactile topic |
| `deco.inference_mode` | 唯一部署 schema 入口，自动派生末端类型、18D/28D layout、双臂约束和 tactile 开关 |
| `deco.inf_step` | `null` 使用 checkpoint 值；正整数仅覆盖在线 denoising 次数 |
| `deco.head_control.mode` | `fixed` 或 `policy` |
| `deco.head_control.initial_value` | 头部初始化 yaw/pitch，单位 rad |
| `deco.head_control.fixed_value` | `fixed` 模式下的头部 state/action；`policy` 模式必须为 `null` |
| `deco.action_dispatch.mode` | `receding_horizon` 或 `temporal_ensemble` |
| `inference.policy_type` | 本地推理为 `deco`，远端调用侧为 `client` |
| `inference.checkpoint.*` | 训练 task、method、run timestamp 与 epoch |

### 5.2 Inference mode

| `deco.inference_mode` | 自动派生的 `env.eef_type` | 自动派生的 `env.state_layout` | Checkpoint profile | Tactile |
| --- | --- | --- | --- | --- |
| `qiangnao_tactile` | `qiangnao` | `deco_28d` | `qiangnao_tactile` | 开启 |
| `qiangnao_no_tactile` | `qiangnao` | `deco_28d` | `qiangnao_tactile` | 关闭 |
| `leju_claw_no_tactile` | `leju_claw` | `deco_18d` | `gripper_no_tactile` | 关闭 |
| `rq2f85_no_tactile` | `rq2f85` | `deco_18d` | `gripper_no_tactile` | 关闭 |

部署加载 checkpoint 后会静态核对 profile、action 维度、tactile 开关、末端类型、state layout 和三路 RGB key，配置不一致时直接报错。

### 5.3 最小配置示例

RQ2F85 仿真部署：

```yaml
env:
  env_name: Kuavo-Sim
  ros_rate: 30

deco:
  inference_mode: rq2f85_no_tactile
  inf_step: null
  head_control:
    mode: fixed
    initial_value: [-0.000115, 0.262111]
    fixed_value: [-0.000115, 0.262111]
  action_dispatch:
    mode: receding_horizon
    receding_horizon:
      n_action_steps: 16
    temporal_ensemble:
      coefficient: 0.1

inference:
  policy_type: deco
  device: cuda
  checkpoint:
    task: your_task
    method: deco_gripper_visual_main
    timestamp: run_YYYYMMDD_HHMMSS
    epoch: best
```

切换末端执行器时只选择新的 `deco.inference_mode` 并加载语义匹配的 checkpoint。配置加载器会从该 mode 自动派生 `env.eef_type`、`env.state_layout`、`env.which_arm=both`、`env.only_arm=true` 与 `env.qiangnao_dof_needed`；如果 YAML 仍显式填写了冲突值，加载时会直接报错。

### 5.4 动作分发

当前支持两种互斥模式：

- `receding_horizon`：推理一个 action chunk，并连续消费前 `n_action_steps` 步；`null` 表示消费完整 chunk。
- `temporal_ensemble`：每个控制周期重新预测完整 chunk，对同一绝对时刻的重叠预测做指数加权。使用该模式时，`receding_horizon.n_action_steps` 必须为 `null`。

两种模式都按照 checkpoint 的 `dataset_hz` 消费动作，因此必须满足：

```text
env.ros_rate == checkpoint.dataset_hz
```

当前不支持部署期 stride 降频。

### 5.5 仿真入口

仿真沿用 Kuavo 现有菜单入口：

```bash
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311
export ROS_IP=127.0.0.1
unset ROS_HOSTNAME

python kuavo_deploy/eval_kuavo.py
```

基本流程：

1. 在 Kuavo Mujoco 仿真器仓库启动 simulator；
2. 在本仓库运行 `kuavo_deploy/eval_kuavo.py`；
3. 进入 `Task Selection Menu`；
4. 选择 `configs/deploy/kuavo_deco_env.yaml`；
5. 选择 `auto_test`。

### 5.6 真机入口

本地真机推理：

```bash
python kuavo_deploy/src/scripts/script.py \
  --task run \
  --config configs/deploy/kuavo_deco_env.yaml
```

真机部署前需要将 `env.env_name` 改为 `Kuavo-Real`，并逐项核对三路 RGB、`/sensors_data_raw`、末端执行器、触觉和控制 topic。

可用任务模式仍由实际启动脚本决定，例如 `run`、`go_run`、`here_run` 和 `back_to_zero`；`deco.inference_mode` 只描述 policy 与机器人 schema，不选择脚本行为。

Server / Client 模式可用于边侧机推理：机器人侧 client 负责采集观测和执行动作，GPU 机器上的 server 负责加载 policy。Preprocessor 与 postprocessor 只能在链路中的一侧执行，不能重复处理。

## 6. 推荐工作流

### 6.1 强脑灵巧手，无触觉

```text
数据:     dataset.eef_type=qiangnao
          deco.write_tactile=false 或 true
训练:     policy.end_effector_profile=qiangnao_tactile
          policy.training_stage=visual_main
部署:     deco.inference_mode=qiangnao_no_tactile
          # loader 自动派生 qiangnao + deco_28d
```

### 6.2 强脑灵巧手，带 Tactile Adapter

```text
数据:     dataset.eef_type=qiangnao
          deco.write_tactile=true
第一阶段: training_stage=visual_main
第二阶段: training_stage=tactile_adapter
          load_external_init_weights=true
          base_policy_path=<第一阶段 checkpoint>
部署:     deco.inference_mode=qiangnao_tactile
          # loader 自动派生 qiangnao + deco_28d
```

### 6.3 Leju Claw 或 RQ2F85

```text
数据:     dataset.eef_type=leju_claw 或 rq2f85
训练:     policy.end_effector_profile=gripper_no_tactile
          policy.training_stage=visual_main
部署:     deco.inference_mode=leju_claw_no_tactile 或 rq2f85_no_tactile
          # loader 自动派生对应 eef_type + deco_18d
```

## 7. 常见误配

- 不要把当前 DECO 描述为 RGB-D 模型；当前 policy 固定消费三路 RGB，不读取 depth。
- 不要交换 head、left wrist、right wrist 的相机顺序，也不要复制单路图像作为缺失相机的降级方案。
- 不要混用 28D 数据、18D policy 或不一致的部署 `state_layout`。
- 不要手工组合 `action_dim`、`use_tactile`、`use_tactile_lora` 与 `freeze_pretrained_main`；它们由 profile 和训练阶段确定。
- 不要让 `gripper_no_tactile` 进入 `tactile_adapter`。
- 不要在没有有效 Visual Main 权重时冻结主干并训练 tactile adapter。
- 不要同时填写 `base_policy_path` 与 `adapter_model_path`。
- 不要将精确 resume 与任何外部初始化权重入口混用。
- 不要把部署 `deco.inference_mode` 当作脚本选择器。
- 不要让 `env.ros_rate` 与 checkpoint 的 `dataset_hz` 不一致。
- 不要尝试使用已经移除的 `stride_action` 部署模式。
- 不要让 server/client 两侧重复执行 preprocessor 或 postprocessor。
- 不要只迁移单个 `epochbest/` 目录而遗漏 run 根目录中的处理器和恢复状态。

## 8. 当前未提供的能力

- Depth / RGB-D 视觉前端；
- 单相机或双相机降级路径；
- 单臂 state/action schema；
- Delta action 或 relative-start action；
- 部署期 stride 降频；
- 自动从 `visual_main` 切换到 `tactile_adapter` 的单次训练流程；
- 独立的数据 Inspector 或 Validator 使用入口。数据检查应以当前转换器输出、LeRobot metadata 和后续专用工具为准。
