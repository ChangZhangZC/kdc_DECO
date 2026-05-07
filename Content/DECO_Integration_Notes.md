# DECO 模型与 Kuavo 系统集成技术备忘录

本文档详细记录了在 `kuavo_data_challenge` 工作流中集成 DECO 模型的关键技术考量与实现步骤，以确保代码落地过程严格遵循导师要求及 LeRobot 框架规范。

## 1. 视频流对齐 (Video Stream Alignment: 30Hz)
- **技术点**：DECO 对时序强依赖，必须保证训练集与部署时的数据流频率严格对齐。
- **落地实现**：
  - 在底层 `kuavo_data/` 脚本（如 `CvtRosbag2Lerobot.py`）中，确保从 ROS bag 提取的 RGB 图像和触觉数据按严格的 `fps=30` 进行抽帧、对齐。
  - 在后续 `configs/data/` 的数据配置中，硬编码 `fps: 30` 确保 LeRobot 训练引擎按固定频率取帧。

## 2. 数据预处理与视觉增强 (Video Preprocessing & Augmentation)
- **分辨率下采样**：
  - 必须对相机的原始高分辨率视频在进入模型前进行 `Resize`（通常为 `(224, 224)` 或 `(256, 256)`），这是保持模型显存开销稳定的必要步骤。
- **视频特征增强与抗干扰（消除曝光、光照等）**：
  - 经查阅 `DECO/dataset.py` 源码，DECO 原生包含针对光照和模糊的数据增强逻辑（使用 `ColorJitter` 处理光线影响，使用 `GaussianBlur` 进行降噪/去模糊的逆向鲁棒性训练）。
  - **重要发现：Kuavo 仓库原生支持强大的视频增强！** 
    查阅配置文件 `configs/policy/act_config.yaml` 和 `configs/policy/diffusion_config.yaml` 后，发现在它们的 `training` 配置块下，都已内置了一个非常强大且可配的 `RGB_Augmenter` 模块。
    这包含：
    1. **ColorJitter (颜色扰动)**：分别独立控制 `brightness` (亮度)、`contrast` (对比度)、`saturation` (饱和度) 和 `hue` (色相)。这完美契合了“消除光线影响”的需求。
    2. **SharpnessJitter (锐度扰动)**：随机调整图像锐度。
    3. **GaussianNoise (高斯噪声)**：随机加入特定均值和标准差的噪点（对应降噪鲁棒性训练）。
    4. **GammaCorrection (Gamma 校正)**：动态调整 Gamma 值，模拟不同曝光条件。
    5. **RandomMask / RandomBorderCutout (随机遮挡/边缘裁剪)**：提升对视野局部遮挡的鲁棒性。
  - **落地实现**：我们**不需要**去修改底层的 Dataloader 代码来实现 DECO 的视频增强。在未来创建的 `configs/policy/deco_config.yaml` 中，只需将现有的 `RGB_Augmenter` 配置块“原封不动”地复制过去即可，这样 DECO 就能无缝享受到整个 Kuavo 系统已有的所有去曝光、降噪等高级数据增强福利。

## 3. 模型数据流转换 (Data Flow Bridge)
- **技术点**：连接 LeRobot 的通用 `batch` 数据流与 DECO 定制的输入流。
- **落地实现**：
  - **原则**：绝不修改 `DECO/models/deco/deco.py` 源文件。
  - 在 `kuavo_train/wrapper/policy/deco/` 下实现 `DECOModelWrapper`：
    - 输入：接收 LeRobot `dataloader` 传来的 `batch` 字典（包含 `OBS_IMAGES`, `OBS_STATE`, `ACTION` 等）。
    - 拆箱：从中分离出前向/侧边相机图像（分配为 `img1`, `img2`），拼接本体状态为 `obs`，并加载触觉 `tac1`, `tac2` 等。
    - 前向传递：将拆解后的张量按照 `(img1, img2, obs, act, task_idx, tac1, tac2)` 的签名传入原始 `DECO.forward()`。

## 4. 训练脚本与 Loss 的控制 (Train Script & Loss Handling)
- **算力与参数量约束**：
  - 目前可用算力上限评估为处理 60-70M 参数模型。
  - DECO 基础架构由 ResNet34（约21M）和多层多模态 Transformer Attention Block 组成。默认配置（6层，dim=512）参数量约在 35M-55M 之间，**符合我们的算力约束**。
  - **剪裁策略备选**：若需加速或缩小体积，可在 Wrapper 传参时直接修改 `num_attn_blocks=4` 或 `dim=256`。
- **Loss 计算封装 (train.py 不可动原则)**：
  - LeRobot 的主流程 `train.py` 只期望调用 `loss, loss_dict = policy.forward(batch)`。
  - DECO 属于 Diffusion 架构，原版的扩散去噪 Loss (`F.mse_loss(out, noise - action)`) 是写在自己的 `train_one_epoch.py` 循环中的。
  - **落地实现**：在实现 `DECOPolicyWrapper` 时，必须将其原本的 Diffusion 前向加噪步骤以及 MSE Loss 的计算封装进 Wrapper 的 `forward` 函数内，再将标量 `loss` 传出，从而做到“即插即用”，完全不碰底层 `train.py`。
