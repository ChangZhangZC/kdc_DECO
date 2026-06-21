# Kuavo-DECO 多相机 RGB-D 前端对齐计划

> 记录日期：2026-06-17  
> 用途：记录本功能分支中 Kuavo-DECO 视觉前端从单头部 RGB-D / RGB-depth 双流 token，升级到 Kuavo ACT 风格多相机 RGB-D 前端的关键决策。本文是 `Content/DECO_Technical_Decisions.md` 的功能分支补充，不回写旧冻结决策。

## 1. 当前目标

本轮目标是让 DECO 的前端视觉流尽量对齐 Kuavo ACT：同时使用头部 RGB-D、左腕 RGB-D、右腕 RGB-D。每个相机内部先做 RGB/depth token-level fusion，再把三组 fused visual tokens 按固定顺序输入 DECO 主干。

固定相机顺序为：

```text
head_cam_h / depth_h
-> wrist_cam_l / depth_l
-> wrist_cam_r / depth_r
```

最终进入 DECO 的视觉 token 不再是旧版：

```text
[RGB tokens][Depth tokens]
```

而是新版：

```text
[head fused RGB-D tokens][left wrist fused RGB-D tokens][right wrist fused RGB-D tokens]
```

## 2. 数据与预处理决策

- 数据清洗默认写出三组 RGB-D feature：`observation.images.head_cam_h`、`observation.depth_h`、`observation.images.wrist_cam_l`、`observation.depth_l`、`observation.images.wrist_cam_r`、`observation.depth_r`。
- `deco.use_wrist_cameras: true` 为默认值；如果设为 `false`，转换脚本只保留 head-only 兼容路径。
- 视觉时间轴由 `dataset.timeline_strategy` 控制：
  - `deco_fixed_hz`：默认 DECO 路线，生成真实 30Hz 目标时间戳。
  - `act_densest_camera_jump`：ACT 对齐实验路线，选择帧数最多的视觉流并按整数 jump 采样。
- 数据集磁盘图像仍保持 `640x480`；训练/部署 preprocessor 再执行 `256x256 letterbox`。
- `256x256` 不是模型数学硬编码，但它是原生 DECO 和当前 Kuavo-DECO 的默认输入尺寸约定。
- RGB letterbox 对齐原生 DECO：bilinear resize，padding 为灰色 `128/255`。
- depth letterbox 是 Kuavo-DECO depth 扩展：nearest resize，padding 默认 `0.5`。`0.5` 是当前归一化 depth image 的中性补边值，避免 `0` 被模型误解成最近深度或无效深度。
- RGB augmentation 保留当前库中融合 ACT + DECO 的增强集合，包括 `GaussianBlur`；depth 不做颜色、噪声、mask、blur 类增强。

## 3. 模型接口决策

policy config 从单 key 改为列表：

```yaml
rgb_keys:
  - observation.images.head_cam_h
  - observation.images.wrist_cam_l
  - observation.images.wrist_cam_r
depth_keys:
  - observation.depth_h
  - observation.depth_l
  - observation.depth_r
visual_fusion_mode: act_rgbd
```

wrapper 按配置顺序 stack 多视角输入：

```text
rgb_views:   [B, V, 3, H, W]
depth_views: [B, V, 1/3, H, W]
V = 3
```

每个 view 内部执行：

```text
RGB ResNet34 + Depth ResNet34
-> 双向 RGB-depth cross attention
-> concat(fused_rgb, fused_depth)
-> Linear(2C -> C)
-> camera fused tokens [B, L, C]
```

然后按相机顺序串接：

```text
[B, L, C] * 3 -> [B, 3L, C]
```

`MMAttention` 不再按 `total_img_len / 2` 把视觉 token 切成 RGB/depth 两半，而是按 `visual_num_views` 和 `tokens_per_view` 切相机段。每个相机段复用同一套二维 image RoPE；RoPE 只表达图像内空间位置，不表达相机身份。本轮不新增 camera embedding，保持 ACT 风格的固定顺序语义。

## 4. ACT 相似性标注

| 维度 | 结论 |
| --- | --- |
| 三组 RGB-D 输入 | Aligned |
| 相机顺序 head -> left wrist -> right wrist | Aligned |
| depth 兼容存储 + wrapper 转 1-channel | Aligned |
| RGB-D token-level fusion | 视觉前端思想对齐 ACT，融合后接 DECO 主干 |
| RGB augmentation | Partially aligned，保留 ACT 集合并融合 DECO GaussianBlur |
| 时间轴 | Configurable，默认 DECO 30Hz，可选 ACT jump |
| 分辨率 | Intentional deviation from ACT，保留 DECO 256x256 letterbox |
| 模型主干/loss | Not aligned by design，保留 DECO Flow Matching |

## 5. 明确不实施项

- 不把三相机 token 伪装成旧 `[RGB][Depth]` 两半结构。
- 不把 RGB-D 在像素通道上堆成 4/6/12 通道送入单个 ResNet。
- 不因为视觉前端改造重新引入或重构 action_stride 类动作执行算法。
- 不修改 `third_party/lerobot/`。
- 不在本轮引入 depth valid mask；如果需要 mask，应作为后续单独 ablation。
