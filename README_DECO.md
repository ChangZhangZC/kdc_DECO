# DECO Model Integration Guide

本文档记录 Kuavo-DECO 集成的使用方式。当前已完成阶段一的数据转换静态实现、二夹爪无触觉 profile 适配、阶段二的 DECO 源码复制归档与依赖记录、阶段三的 `third_party/deco` 模型手术静态实现，以及阶段四/五的训练 wrapper、DECO 专用 preprocessor、策略配置、基础部署入口注册和 DECO 专用部署配置。后续仍需在允许运行的环境中补做 validator 实跑、仿真闭环、实机 dry-run 与依赖最小化复查。

## 阶段一：RGB-D 数据转换

### 目标

DECO 阶段一转换链路将 Kuavo rosbag 转成 DECO wrapper 预期的 LeRobot 数据集：

- `observation.images.head_cam_h`：头部 RGB 图像。
- `observation.depth_h`：与头部 RGB 对齐的 depth image。当前为了兼容 LeRobot image/video writer，磁盘中保存为 3-channel depth image；DECO wrapper 会按 depth 语义取单通道输入。
- `observation.state` / `action`：由 `end_effector_profile` 决定维度和顺序。
- `qiangnao_tactile`：28 维，顺序为左臂 7 + 左手 6 + 右臂 7 + 右手 6 + 头部 2；可写入 `observation.tactile`。
- `gripper_no_tactile`：18 维，顺序为左臂 7 + 左夹爪 1 + 右臂 7 + 右夹爪 1 + 头部 2；不写入、不要求、不使用 `observation.tactile`。
- `observation.tactile`：仅 `qiangnao_tactile` 可选写入，30 维，顺序为左手 15 + 右手 15 的 normal force，原始值除以 100 后作为牛顿量纲。
- `action` 的头部两维当前固定为 `[0.0, 0.0]`。

### 配置文件

数据配置位于：

```bash
configs/data/KuavoRosbag2Lerobot_deco.yaml
```

关键默认值：

- `dataset.eef_type: qiangnao`
- `deco.end_effector_profile: auto`
- `dataset.train_hz: 30`
- `dataset.use_depth: true`
- `dataset.dex_dof_needed: 6`
- `deco.rgb_topic: /cam_h/color/image_raw/compressed`
- `deco.depth_topic: /cam_h/depth/image_raw/compressed`
- `deco.depth_encoding: compressed_image`
- `deco.raw_depth_topic: /camera/depth/image_rect_raw`
- `deco.allow_raw_depth_fallback: false`

`deco.end_effector_profile: auto` 会根据 `dataset.eef_type` 推导 schema：

- `dataset.eef_type=qiangnao` -> `qiangnao_tactile`，输出 28D state/action，可按 `deco.write_tactile` 写入 30D tactile。
- `dataset.eef_type=leju_claw` -> `gripper_no_tactile`，读取 `/leju_claw_state` 和 `/leju_claw_command`，输出 18D state/action，无 tactile。
- `dataset.eef_type=rq2f85` -> `gripper_no_tactile`，读取 `/gripper/state` 和 `/gripper/command`，输出 18D state/action，无 tactile。

当前实际 rosbag 中的头部 depth 使用 `/cam_h/depth/image_raw/compressed`，消息语义为 `sensor_msgs/CompressedImage`，因此默认使用 `compressed_image` decoder，直接通过 `cv2.imdecode(..., cv2.IMREAD_UNCHANGED)` 解码图像缓冲区。

`/cam_h/depth/image_raw/compressedDepth`、`compressedDepth_png`、`/camera/depth/image_rect_raw` 与 `raw_16uc1` 目前只是候选 fallback。未经 Inspector 或 validator 复核时，不应替代当前默认 depth topic。

### 手动运行流程

当前 Codex 机器遵守 No-Runtime 约束，不直接执行转换。需要在允许运行 ROS/LeRobot 环境的机器上手动执行：

```bash
python kuavo_data/CvtRosbag2Lerobot_DECO.py \
  rosbag.rosbag_dir=/path/to/rosbag_dir \
  rosbag.lerobot_dir=/path/to/output_lerobot_deco/lerobot
```

`rosbag.lerobot_dir` 建议直接填写最终 LeRobot dataset root，也就是后续训练配置 `root` 会读取的目录。若使用相对路径，脚本会按 `<rosbag_dir>/../<lerobot_dir>/lerobot` 生成，以兼容原 ACT/DP 转换脚本的目录习惯。

如果输出目录已经存在，脚本默认报错，避免误删已有数据。确认需要覆盖时再显式设置：

```bash
python kuavo_data/CvtRosbag2Lerobot_DECO.py \
  rosbag.rosbag_dir=/path/to/rosbag_dir \
  rosbag.lerobot_dir=/path/to/output_lerobot_deco/lerobot \
  deco.overwrite=true
```

### 运行后检查

转换完成后，下一阶段应新增并运行 `kuavo_data/validate_deco_lerobot_dataset.py`，至少检查：

- 字段是否齐全：RGB、depth、state、tactile、action。
- 维度是否符合：`qiangnao_tactile` 为 state/action `(28,)`、tactile `(30,)`；`gripper_no_tactile` 为 state/action `(18,)` 且没有 tactile。
- 目标时间轴是否约为 30Hz。
- depth 是否来自默认 `/cam_h/depth/image_raw/compressed`，并通过 `compressed_image` decoder 正确解码。
- 头部 state 是否为每个 episode 的 `joint_q[26:28]` 均值广播，头部 action 是否补零。
- arm action 是否按 `/kuavo_arm_traj_synced`、`/kuavo_arm_traj`、`/joint_cmd` 优先级选取。

当前 validator 已提供基础 metadata 检查和可选 parquet/video 深度检查。它不依赖 ROS1，也不读取 `.bag`：

```bash
python kuavo_data/validate_deco_lerobot_dataset.py \
  --root data_example/lerobot \
  --report data_example/lerobot/deco_validation_report.md
```

二夹爪无触觉数据集应显式选择 gripper profile：

```bash
python kuavo_data/validate_deco_lerobot_dataset.py \
  --root /path/to/gripper_lerobot \
  --end-effector-profile gripper_no_tactile \
  --no-require-tactile
```

如果当前环境缺少 `pandas/pyarrow/numpy/cv2`，可以先只做 metadata 和文件结构检查：

```bash
python kuavo_data/validate_deco_lerobot_dataset.py \
  --root data_example/lerobot \
  --metadata-only
```

## 阶段二：源码复制与路径约定

### 源码位置

当前保留仓库根目录下的原始 `DECO/` 文件夹不变，并将其复制到：

```bash
third_party/deco
```

后续 Kuavo-DECO 集成应以 `third_party/deco/` 作为第三方源码副本；根目录 `DECO/` 仅作为原始参考副本保留，避免后续适配过程中混淆修改来源。

### 与 LeRobot 的关系

ACT 和 Diffusion Policy 当前来自 LeRobot submodule 内部：

```bash
third_party/lerobot/src/lerobot/policies/act
third_party/lerobot/src/lerobot/policies/diffusion
```

Kuavo 对 ACT/DP 的适配不直接修改 `third_party/lerobot/`，而是在 `kuavo_train/wrapper/policy/act/` 与 `kuavo_train/wrapper/policy/diffusion/` 中继承并封装原始 policy。DECO 沿用该模式：不注册到 LeRobot submodule，不修改 `third_party/lerobot/`，而是在 `kuavo_train/wrapper/policy/deco/` 中接入 `third_party/deco/`。

### Python 路径约定

DECO 原始源码内部使用了类似下面的绝对导入：

```python
from models.deco.deco import DECO
from models.deco.img_encoder import ResNet34
```

wrapper 接入时，会在 `kuavo_train/wrapper/policy/deco/__init__.py` 中把 `third_party/deco` 注入 `sys.path`，使这些导入继续按原始 DECO 结构工作：

```python
import sys
from pathlib import Path

DECO_ROOT = Path(__file__).resolve().parents[4] / "third_party" / "deco"
if str(DECO_ROOT) not in sys.path:
    sys.path.insert(0, str(DECO_ROOT))
```

当前 `DECOPolicyWrapper` 已按该路径约定接入；本仓库仍遵守静态修改约束，未在 Codex 机器上执行训练或 forward 验证。

### 依赖记录

DECO 原始依赖见：

```bash
third_party/deco/requirements.txt
```

根目录 `requirements_DECO.txt` 记录了 DECO 原始依赖与 Kuavo/LeRobot 环境的兼容说明。当前阶段不安装依赖；`torch`、`torchvision`、`diffusers`、`huggingface-hub` 等核心包应以后续实际 Kuavo/LeRobot 训练环境为准，不在本阶段强制切换版本。

## 阶段三：third_party/deco 模型手术

### 视觉输入模式

阶段三已在 `third_party/deco/models/deco/` 中加入 Kuavo RGB-D 视觉前端适配。核心配置项位于：

```yaml
model:
  vision_backbone: resnet34
  depth_backbone: resnet34
```

Kuavo 定制副本中的 DECO 主体已经固定为 RGB-D 路线：`rgb` 为头部 RGB，`depth` 为对齐深度图。旧 DECO 双 RGB 兼容入口已从模型主体中移除，避免与当前 RGB-D 规划混淆。

RGB-D 模式下：

- RGB 使用 3-channel ResNet backbone。
- depth 使用 1-channel ResNet backbone。
- depth conv1 由 RGB conv1 权重按通道均值初始化。
- RGB/depth 在 ResNet layer4 后做双向 cross attention。
- 输出仍保留两路 visual tokens：`fused_rgb_tokens` 与 `fused_depth_tokens`，继续接入 DECO `MMAttention`。

当前 `third_party/deco/config/deco.yaml` 只保留模型结构字段；`.safetensors`、两阶段冻结和原生 `.pth` 兼容加载均由 `kuavo_train/wrapper/policy/deco/` 处理。

### 触觉输入

阶段三已将 DECO tactile 分支从 Inspire Hand 1062D 区域均值逻辑改为 Kuavo 30D 输入：

- `tac1`: 左手 15D。
- `tac2`: 右手 15D。
- `tactile_encoder`: `30D -> 34D`。
- tactile fusion: `15 + 15 + 34 = 64`。

模型 forward 期望收到的 `tac1/tac2` 已经完成 DECO-style tactile max 归一化。数据转换阶段的 `/100` 只表示把 Kuavo normal force 转成牛顿；正式触觉训练前仍需要在 wrapper/config 中填写正数 `tactile_left_max` 与 `tactile_right_max`。

`third_party/deco/config/deco.yaml` 只保留 Kuavo 语义的 `tactile_left_max/tactile_right_max`。旧 DECO 别名 `tac_left_max/tac_right_max` 已移除，避免与 Kuavo 标准化 wrapper 字段混淆。当 `use_tactile: true` 时，这两个字段必须填写为正数。

配置文件中也只保留一处 `chunk_size`：`model.chunk_size`。DECO 原生 `data.chunk_size`、obs/action 手动归一化统计量和旧 RGB `img_mean/img_std` 不再作为 Kuavo RGB-D 路线的权威配置；state/action/image 的标准化由 Kuavo LeRobot wrapper/preprocessor 负责。

### 保留项

阶段三没有改变 DECO 主干的以下行为：

- `action_encoder`
- action token chunk 建模
- `MMAttention`
- Flow Matching `add_noise`
- 训练目标 `F.mse_loss(out, noise - action)`
- 推理阶段 denoising loop

## 训练端 Profile 配置

DECO policy 配置位于：

```bash
configs/policy/deco_config.yaml
```

默认配置仍是 `qiangnao_tactile`：

```yaml
policy:
  end_effector_profile: qiangnao_tactile
  action_dim: 28
  training_stage: visual_main
  use_tactile: false
  use_tactile_lora: false
```

如果训练二夹爪无触觉数据，需要同时切换 profile 和动作维度：

```yaml
policy:
  end_effector_profile: gripper_no_tactile
  action_dim: 18
  training_stage: visual_main
  use_tactile: false
  use_tactile_lora: false
  load_external_init_weights: false
```

`gripper_no_tactile` 没有 `observation.tactile`，因此 wrapper 会拒绝 `use_tactile: true`、`use_tactile_lora: true` 或 `training_stage: tactile_adapter`。这和 DECO 主干并不矛盾：18D state/action 会先经过 `Linear(action_dim -> dim)` 投影到统一 hidden dim，Transformer 主干仍处理固定 hidden token；随 profile 改变的是数据 schema、normalizer stats 以及输入/输出线性层形状。

## 部署路径约定

DECO 部署配置位于：

```bash
configs/deploy/kuavo_deco_env.yaml
```

通用 `configs/deploy/kuavo_env.yaml` 保持 ACT/DP 默认语义，不承载 DECO 专用 depth/tactile 配置。DECO 的头部 depth topic 固定为当前数据规划中的 `/cam_h/depth/image_raw/compressed`，与 `compressed_image` decoder 路线一致；旧 ACT/DP `compressedDepth_png` 路线仍保留给通用配置使用。

部署资产沿用原 Kuavo 三层 run 路径：

```text
outputs/train/<task>/<method>/<timestamp>/
```

在 `kuavo_deco_env.yaml` 中填写：

- `task`：对应 `outputs/train/<task>/`。
- `method`：建议为 `deco` 或实际训练方法名。
- `timestamp`：对应 run 目录名，例如 `run_20260518_120000`。
- `epoch`：选择 run 目录下的 `epoch<epoch>` 权重子目录，例如 `best`、`50`、`100`。

因此，模型权重实际从：

```text
outputs/train/<task>/<method>/<timestamp>/epoch<epoch>
```

读取，但 `policy_preprocessor.json` 与 `policy_postprocessor.json` 保存在 run 根目录。`epochbest/` 或任意 `epoch<epoch>/` 单独拷贝不是完整部署包；部署、迁移或归档时应保留整个 `run_xxx/` 目录。

第一版部署设计仍以 `qiangnao_tactile` / `deco_28d` 为目标：在线 `observation.state` 顺序应与离线 converter 保持一致，即左臂 7 + 左手 6 + 右臂 7 + 右手 6 + 头部 2。当前部署侧若要接入 `gripper_no_tactile`，需要后续单独设计 18D 在线 state/action 拼接和下发逻辑；本轮二夹爪适配不修改 `kuavo_deploy/*`。

使用已有 `kuavo_deploy/src/scripts/script.py` 或 `script_auto_test.py` 时，应通过 `--config configs/deploy/kuavo_deco_env.yaml` 显式传入 DECO 配置。`kuavo_deploy/kuavo_service/server.py` 也支持 `--config configs/deploy/kuavo_deco_env.yaml`，或通过 `KUAVO_DEPLOY_CONFIG` 环境变量选择配置；服务端内部会从 run 根目录加载 pre/post processor，使调用方只需要发送 raw obs。

## 当前限制

- 训练端 DECO wrapper 已完成静态接入，但尚未在本机执行 import、forward 或训练验证。
- 部署入口已完成部分 DECO policy 类型和自定义 preprocessor 注册；`deco_28d` 在线 state/action、30D tactile callback、服务端 processor 接管以及 `gripper_no_tactile` 在线部署适配仍属后续阶段。
- 部署端 10Hz 控制频率不在洗数据阶段处理，由 wrapper/deploy action queue 使用 `action_stride=3` 完成。
- 当前新增转换脚本与 validator 仅完成静态审查，尚未在本机执行 rosbag 转换或 validator。
- 阶段三模型修改仅做静态代码审查，尚未在本机执行 forward、训练或部署验证。
