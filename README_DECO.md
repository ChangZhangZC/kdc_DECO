# DECO Model Integration Guide

本文档记录 Kuavo-DECO 集成的使用方式。当前已完成阶段一的数据转换静态实现，以及阶段二的 DECO 源码复制归档与依赖记录；训练 wrapper、模型手术和部署 wrapper 仍以后续阶段为准。

## 阶段一：RGB-D 数据转换

### 目标

DECO 阶段一转换链路将 Kuavo rosbag 转成 DECO wrapper 预期的 LeRobot 数据集：

- `observation.images.head_cam_h`：头部 RGB 图像。
- `observation.depth_h`：与头部 RGB 对齐的 depth image。当前为了兼容 LeRobot image/video writer，磁盘中保存为 3-channel depth image；后续 wrapper 需按 depth 语义还原为单通道输入。
- `observation.state`：28 维，顺序为左臂 7 + 左手 6 + 右臂 7 + 右手 6 + 头部 2。
- `observation.tactile`：30 维，顺序为左手 15 + 右手 15 的 normal force，原始值除以 100 后作为牛顿量纲。
- `action`：28 维，顺序同 state；头部 action 当前固定为 `[0.0, 0.0]`。

### 配置文件

数据配置位于：

```bash
configs/data/KuavoRosbag2Lerobot_deco.yaml
```

关键默认值：

- `dataset.train_hz: 30`
- `dataset.use_depth: true`
- `dataset.dex_dof_needed: 6`
- `deco.rgb_topic: /cam_h/color/image_raw/compressed`
- `deco.depth_topic: /cam_h/depth/image_raw/compressed`
- `deco.depth_encoding: compressed_image`
- `deco.raw_depth_topic: /camera/depth/image_rect_raw`
- `deco.allow_raw_depth_fallback: false`

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
- 维度是否符合：state/action `(28,)`，tactile `(30,)`。
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

Kuavo 对 ACT/DP 的适配不直接修改 `third_party/lerobot/`，而是在 `kuavo_train/wrapper/policy/act/` 与 `kuavo_train/wrapper/policy/diffusion/` 中继承并封装原始 policy。DECO 后续也沿用该模式：不注册到 LeRobot submodule，不修改 `third_party/lerobot/`，而是在后续 `kuavo_train/wrapper/policy/deco/` 中接入 `third_party/deco/`。

### Python 路径约定

DECO 原始源码内部使用了类似下面的绝对导入：

```python
from models.deco.deco import DECO
from models.deco.img_encoder import ResNet34
```

后续 wrapper 接入时，应在 wrapper 顶部把 `third_party/deco` 注入 `sys.path`，使这些导入继续按原始 DECO 结构工作：

```python
import sys
from pathlib import Path

DECO_ROOT = Path(__file__).resolve().parents[4] / "third_party" / "deco"
if str(DECO_ROOT) not in sys.path:
    sys.path.insert(0, str(DECO_ROOT))
```

该路径约定只说明后续 wrapper 的接入方式；阶段二不实现 `DECOPolicyWrapper`，也不执行训练或 forward 验证。

### 依赖记录

DECO 原始依赖见：

```bash
third_party/deco/requirements.txt
```

根目录 `requirements_DECO.txt` 记录了 DECO 原始依赖与 Kuavo/LeRobot 环境的兼容说明。当前阶段不安装依赖；`torch`、`torchvision`、`diffusers`、`huggingface-hub` 等核心包应以后续实际 Kuavo/LeRobot 训练环境为准，不在本阶段强制切换版本。

## 当前限制

- 训练端 DECO wrapper 尚未完成；不要直接把该数据集喂给未适配 RGB-D/tactile schema 的旧 DECO 训练入口。
- 部署端 10Hz 控制频率不在洗数据阶段处理，后续由 wrapper/deploy action queue 使用 `action_stride=3` 完成。
- 当前新增转换脚本与 validator 仅完成静态审查，尚未在本机执行 rosbag 转换或 validator。
