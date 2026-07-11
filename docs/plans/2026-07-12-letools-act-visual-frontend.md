# LeTools ACT 六流视觉前端迁移实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task.

**Goal:** 将 LeTools ACT 的六路共享 ResNet18 视觉编码前端迁移到 Kuavo-DECO，同时保持 DECO 的 RoPE、MMAttention、Flow Matching、joint state、tactile 与动作分发逻辑不变。

**Architecture:** 三组 RGB-D 在 256×256 letterbox 后按 `[head RGB, head depth, left RGB, left depth, right RGB, right depth]` 交错为六个三通道视觉流。六路使用同一个 ImageNet 预训练 ResNet18、FrozenBatchNorm2d、layer4 feature map 与共享 1×1 projection，形成 `[B, 384, 512]` 连续视觉 token 后进入现有 DECO 主干。

**Tech Stack:** Python、PyTorch、torchvision、LeRobot processor、Hydra/OmegaConf、DECO Flow Matching。

---

## Task 1：配置与预处理语义

**Files:**
- Modify: `configs/policy/deco_config.yaml`
- Modify: `kuavo_train/wrapper/policy/deco/DECOConfigWrapper.py`
- Modify: `kuavo_train/wrapper/policy/deco/DECOProcessor.py`

1. 注册 `visual_fusion_mode=letools_act`，保留 `act_rgbd` 兼容模式。
2. 增加 ImageNet ResNet18 权重、final stride dilation 与 ImageNet stats 配置字段。
3. 将新模式限制为三组 RGB-D、三通道 depth、ResNet18、关闭 RGB-D cross-attention。
4. 保持 `letterbox -> optional RGB-only augmentation -> normalizer` 顺序；RGB augmenter 默认关闭。
5. 对六个视觉 key 使用 ImageNet mean/std，并让 RGB/DEPTH 均采用 `MEAN_STD`。

## Task 2：六流共享视觉编码器

**Files:**
- Create: `third_party/deco/models/deco/letools_act_visual_encoder.py`
- Modify: `third_party/deco/models/deco/deco.py`
- Modify: `kuavo_train/wrapper/policy/deco/DECOPolicyWrapper.py`

1. 封装 torchvision ResNet18、FrozenBatchNorm2d、layer4 getter 与共享 1×1 projection。
2. 按 RGB-D pair 交错组装六流，不把 depth 压成单通道。
3. 将 `[B, 6, 3, 256, 256]` 编码成 `[B, 6, 64, 512]`，再展平为 `[B, 384, 512]`。
4. 对六个 stream 段分别复用现有二维 RoPE；不添加 camera/modality embedding。
5. 保留 `act_rgbd` 原路径与旧 checkpoint 构造能力。

## Task 3：部署、回归定义与文档

**Files:**
- Modify: `kuavo_deploy/utils/deco_obs_action.py`
- Create: `tests/test_deco_letools_visual_frontend.py`
- Create: `Content/DECO_LeTools_ACT_Visual_Frontend.md`
- Modify: `PLANS.md`
- Modify: `AI_Logs.md`

1. 部署视觉 key 校验同时接受 `act_rgbd` 与 `letools_act`。
2. 静态回归测试定义覆盖六流顺序、三通道 depth、token shape、ImageNet stats、非法配置、旧模式兼容和增强开关。
3. 记录架构边界、输入输出 shape、兼容性与已知计算代价。
4. 按仓库 No-Runtime 规则仅进行文本检索、人工逻辑审查、`git diff --check` 与 Git 状态检查，不运行 Python/pytest/训练/ROS/仿真。

## Task 4：交付

1. 复核最终 diff 不包含 state、tactile、loss、action dispatcher 或数据转换逻辑变化。
2. 更新中文操作日志和计划清单。
3. 创建 `feat(deco): port LeTools ACT six-stream visual frontend` 提交。
4. 推送到 `origin/deco/feature/letools-visual`。
