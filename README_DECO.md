# DECO Model Integration Guide

本文档记录 Kuavo-DECO 集成的使用方式。当前已完成阶段一的静态实现：新增 DECO 专用数据配置与 rosbag -> LeRobot 转换脚本；训练 wrapper、模型手术和部署 wrapper 仍以后续阶段为准。

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
- `deco.depth_topic: /cam_h/depth/image_raw/compressedDepth`
- `deco.depth_encoding: compressedDepth_png`
- `deco.raw_depth_topic: /camera/depth/image_rect_raw`
- `deco.allow_raw_depth_fallback: false`

`/camera/depth/image_rect_raw` 与 `raw_16uc1` 目前只是候选 fallback。未经 Inspector 或 validator 复核时，不应替代默认的 ACT/DP depth topic。

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
- depth 是否来自默认 compressedDepth PNG 解码链路。
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

## 当前限制

- 训练端 DECO wrapper 尚未完成；不要直接把该数据集喂给未适配 RGB-D/tactile schema 的旧 DECO 训练入口。
- 部署端 10Hz 控制频率不在洗数据阶段处理，后续由 wrapper/deploy action queue 使用 `action_stride=3` 完成。
- 当前新增转换脚本与 validator 仅完成静态审查，尚未在本机执行 rosbag 转换或 validator。
