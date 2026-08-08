# DECO 训练配置清理实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task.

**Goal:** 将 DECO 训练配置收敛为 `visual_main` 与 `tactile_adapter` 两种严格模式，明确三类权重入口，并保证旧 checkpoint 配置仍可读取。

**Architecture:** YAML 只暴露真正需要选择的训练字段；固定 schema 与阶段布尔开关由 `CustomDECOConfigWrapper` 派生。训练入口注入数据集频率与历史 `.pth` 初始化路径，并在构造模型前统一校验 resume、主干初始化和 adapter 续训的互斥关系。

**Tech Stack:** Python、Hydra/OmegaConf、PyTorch、LeRobot policy wrapper、safetensors。

---

### Task 1: 收敛训练配置

**Files:**
- Modify: `configs/policy/deco_config.yaml`

- [x] 固化两种 `training_stage` 的语义，移除由代码派生的信息字段。
- [x] 将历史 `.pth` 入口移动到 `training.resume` 附近。
- [x] 在配置注释中明确 `adapter_model_path` 与 `training.resume` 的区别。

### Task 2: 固化阶段契约和权重入口

**Files:**
- Modify: `kuavo_train/wrapper/policy/deco/DECOConfigWrapper.py`
- Modify: `kuavo_train/wrapper/policy/deco/DECOPolicyWrapper.py`

- [x] 由阶段派生 tactile、LoRA、冻结开关和 action 维度。
- [x] 校验三类权重入口的阶段适用范围、互斥关系和显式加载开关。
- [x] 保留历史字段用于旧 checkpoint 反序列化，保存新 checkpoint 时继续清空外部路径。

### Task 3: 对齐两条训练入口

**Files:**
- Modify: `kuavo_train/train_policy.py`
- Modify: `kuavo_train/train_policy_with_accelerate.py`

- [x] 从 LeRobot 数据集元数据注入 `dataset_hz`。
- [x] 在模型构造前应用统一的训练初始化校验。
- [x] 让尾帧采样读取派生后的 `drop_n_last_frames`。

### Task 4: 静态验收和记录

**Files:**
- Modify: `AI_Logs.md`

- [x] 静态检查配置到权重加载、冻结、采样和 resume 的完整链路。
- [x] 不运行 Python、训练、测试或环境修改命令。
- [x] 记录变更目的、兼容边界和后续 README 待办。
