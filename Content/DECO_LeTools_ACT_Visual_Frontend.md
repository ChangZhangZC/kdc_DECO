# Kuavo-DECO 对齐 LeTools ACT 六流视觉前端

> **分支**：`deco/feature/letools-visual`
>
> **基线**：`deco/feature/muti-visual@517069b`
>
> **记录日期**：2026-07-12

## 1. 冻结结论

本分支只替换进入 DECO 主干之前的视觉编码前端。数据转换、30Hz 时间轴、joint state、触觉、action token、Flow Matching loss、推理去噪与动作 dispatcher 均不在本次修改范围内。

固定视觉路径如下：

```text
三组 RGB-D
  -> 256x256 letterbox
  -> 可选 RGB-only augmentation（默认关闭；depth 永不增强）
  -> 六路 ImageNet mean/std
  -> [head RGB, head depth, left RGB, left depth, right RGB, right depth]
  -> shared torchvision ImageNet ResNet18 + FrozenBatchNorm2d
  -> layer4 + shared Conv2d(512, 512, kernel_size=1)
  -> 6 x 8 x 8 tokens = [B, 384, 512]
  -> 每个 64-token stream 独立复用 DECO 二维 RoPE
  -> DECO MMAttention / Flow Matching 主干
```

## 2. 与 LeTools ACT 对齐的边界

本机 LeTools ACT 参考实现的关键事实为：

- ACT 配置使用 `vision_backbone: resnet18`、`ResNet18_Weights.IMAGENET1K_V1` 和 `replace_final_stride_with_dilation: false`。
- ACT 模型使用 torchvision ResNet、`FrozenBatchNorm2d`、`IntermediateLayerGetter(layer4)`，并使用 `1x1 Conv` 将 layer4 通道投影到 `dim_model=512`。
- 多个 image feature 逐个通过同一个 `self.backbone` 和同一个 `encoder_img_feat_input_proj`，因此属于共享权重的多流编码。
- LeTools 训练配置的 `dataset.use_imagenet_stats: true` 会把数据集所有 camera key 的 mean/std 覆盖为 ImageNet 数值。
- LeTools ACT 默认配置没有开启图像增强：`dataset.image_transforms` 为空。因此本分支保留 Kuavo 已有 RGB 增强池作为显式开关，但默认关闭；打开时仍只作用于 RGB。

迁移到 DECO 后只复刻上述视觉 backbone 边界，不引入 ACT 的 latent token、VAE encoder、ACT Transformer encoder/decoder 或动作预测头。

## 3. 六流与数值语义

三组 RGB-D 都以三通道图像进入模型。特别是 depth 在 `letools_act` 模式下不再取第一通道，也不使用独立 1-channel depth backbone。

六个输入 key 固定为：

1. `observation.images.head_cam_h`
2. `observation.depth_h`
3. `observation.images.wrist_cam_l`
4. `observation.depth_l`
5. `observation.images.wrist_cam_r`
6. `observation.depth_r`

六路都采用：

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

DECO preprocessor 会复制 dataset stats 后，只覆盖六个视觉 key；state/action 的数据集统计与 tactile 的 `IDENTITY` 语义保持不变。覆盖后的统计随 policy preprocessor 保存，使训练与部署恢复相同数值路径。

## 4. DECO 主干接口

`256x256` 输入经标准、无 dilation 的 ResNet18 到 layer4 后得到 `8x8` feature map。每个 stream 形成 64 个 token，六流按冻结顺序展平为 384 个视觉 token。

DECO 不增加 camera embedding 或 modality embedding。`MMAttention` 按六个连续的 64-token segment 分段应用同一份二维 RoPE，所以 RoPE 只表达 stream 内空间位置；流身份由固定序列位置隐式区分。六流随后与 action token 在原有 MMAttention 中交互。

## 5. 兼容模式

`visual_fusion_mode` 支持：

- `letools_act`：本分支默认。共享三通道 ResNet18 六流，不构建独立 depth backbone、RGB-depth cross-attention 或 concat+Linear fusion。
- `act_rgbd`：保留旧 checkpoint/A-B 对照。继续使用独立 RGB/1-channel depth backbone，可选 RGB-depth cross-attention，并在每个相机内 concat+Linear 得到 fused camera tokens。

两种模式在部署端继续要求相同的三组 RGB/depth observation keys。旧 checkpoint 缺少新增字段时仍落到旧默认值，不会被静默切成 ImageNet 六流数值路径。

## 6. 优缺点与验证边界

优点：视觉结构与 LeTools ACT 的 ImageNet ResNet18 前端一致；RGB 与 depth 不在主干前提前压缩；六路共享权重，避免六套 backbone 参数；新旧前端可显式 A/B 对照。

代价：一次前向需要编码六张三通道图，计算量高于旧三组 fused view；depth 使用 RGB ImageNet 先验，不等价于物理深度专用编码；没有显式 stream embedding，模型只能结合固定顺序与内容学习来源差异。

按照本仓库 No-Runtime 规则，本分支只定义静态回归测试并进行人工静态分析，不在本机执行 Python、pytest、训练、ROS 或仿真。运行环境仍需验证权重下载/缓存、单步 forward、显存、训练吞吐及真实部署时延。
