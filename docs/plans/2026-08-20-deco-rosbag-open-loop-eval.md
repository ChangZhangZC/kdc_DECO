# DECO Rosbag Open Loop Evaluation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 `kdc_DECO` 仓库内新增直接读取单个 Rosbag 的 DECO 开环评估入口，将按训练数据规则构造的 Ground Truth action 与模型预测 action 绘制在同一坐标轴并输出可追溯指标。

**Architecture:** 评估入口复用 `DecoRosbagReader` 完成 topic 解码和时间对齐，并在 eval 内保留一份与原数据转换流程等价的单帧 18D/28D schema 构造逻辑，不修改稳定的数据清洗入口；模型、preprocessor 与 postprocessor 均从 `outputs/train/<task>/<method>/<timestamp>` 目录层级加载。评估只调用 `predict_action_chunk()`，按 `action_horizon` 拼接预测结果，不接入 ROS 控制、仿真或真机执行。

**Tech Stack:** Python、Hydra/OmegaConf、PyTorch、LeRobot processor、ROS bag reader、NumPy、Matplotlib。

---

### Task 1: 保护现有 Rosbag 数据转换入口

**Files:**
- Review only: `kuavo_data/CvtRosbag2Lerobot_DECO.py`

1. 保持 `CvtRosbag2Lerobot_DECO.py` 与本功能开始前完全一致，不调整现有 `populate_dataset()`。
2. 只读取并复用其现有 reader 与 state/action 基础转换函数。
3. 在 eval 内复制 `populate_dataset()` 所需的 28D/18D 单帧组装分支并逐项静态核对。

### Task 2: 新增 Open Loop Eval 配置

**Files:**
- Create: `configs/eval/deco_open_loop_eval.yaml`

1. 通过 Hydra defaults 将 `configs/data/KuavoRosbag2Lerobot_deco.yaml` 组合到 `conversion` 命名空间。
2. 定义单 Rosbag 路径、起始帧、最大步数、训练目录层级、device、seed、`inf_step`、`action_horizon` 和输出选项。
3. 保持模型结构、normalization statistics、profile、chunk size 与 RGB key 由 checkpoint 决定，不在评估 YAML 重复定义。

### Task 3: 实现 DECO Open Loop Eval

**Files:**
- Create: `kuavo_eval/__init__.py`
- Create: `kuavo_eval/open_loop_eval.py`

1. 按 `outputs/train/<task>/<method>/<timestamp>/epoch<epoch>` 解析 checkpoint，并从 run root 加载 processor。
2. 加载 checkpoint 后，校验 Rosbag 清洗配置与 checkpoint 的 FPS、profile、触觉、三视角和 action 维度。
3. 使用 `DecoRosbagReader.process_rosbag()` 在内存中读取并对齐 Rosbag，不生成 LeRobot dataset。
4. 使用 eval 本地的 `build_deco_frame_from_aligned_bag()` 副本逐帧构造 Ground Truth action；每隔 `action_horizon` 构造当前 observation、调用保存的 preprocessor、`predict_action_chunk()` 与 postprocessor。
5. 严格裁剪 episode 尾部有效长度，拒绝 NaN/Inf 与 shape 不一致。
6. 输出 MSE/MAE、`summary.json`、`predictions.npz` 和分页 action 对比图。

### Task 4: 文档与追踪

**Files:**
- Modify: `README_DECO.md`
- Modify: `AI_Logs.md`

1. 记录配置字段、启动命令、输出结构、GT action 来源和开环评估边界。
2. 在 `AI_Logs.md` 的 `2026-08-20` 日期下记录所有新增/修改文件、功能目的与静态验证边界。

### Task 5: 静态验证

**Files:**
- Review all files above.

1. 使用文本检索核对旧帧构造逻辑没有残留重复实现。
2. 静态核对 YAML → checkpoint/run-root → Rosbag reader → frame builder → preprocessor → policy → postprocessor → metrics/plot 的完整调用链。
3. 使用 `git diff --check` 与人工语法结构审查，不执行 Python、pytest、模型、Rosbag、ROS、仿真或部署程序。
