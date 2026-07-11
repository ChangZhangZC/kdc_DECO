"""LeTools ACT 风格的共享六流视觉编码器。"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torchvision
from torchvision.models import ResNet18_Weights
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d


LETOOLS_RESNET18_WEIGHTS = "ResNet18_Weights.IMAGENET1K_V1"


def interleave_rgb_depth_streams(rgb: Tensor, depth: Tensor) -> Tensor:
    """按 `[head RGB, head depth, left RGB, ...]` 交错三组 RGB-D。

    RGB 与 depth 都必须保持三通道；该函数不会执行均值降维或模态融合。把顺序逻辑
    独立出来后，可用标量标记的轻量测试直接验证顺序，而无需构造完整 ResNet。
    """

    if rgb.ndim == 4:
        rgb = rgb.unsqueeze(1)
    if depth.ndim == 4:
        depth = depth.unsqueeze(1)
    if rgb.ndim != 5 or depth.ndim != 5:
        raise ValueError(
            "LeTools ACT RGB-D inputs must be [B,V,C,H,W], "
            f"got rgb={tuple(rgb.shape)}, depth={tuple(depth.shape)}."
        )
    if rgb.shape[:2] != depth.shape[:2] or rgb.shape[-2:] != depth.shape[-2:]:
        raise ValueError(
            "LeTools ACT RGB/depth must share batch/view/spatial shape, "
            f"got rgb={tuple(rgb.shape)}, depth={tuple(depth.shape)}."
        )
    if rgb.shape[1] != 3:
        raise ValueError(f"letools_act requires exactly three RGB-D pairs, got {rgb.shape[1]}.")
    if rgb.shape[2] != 3 or depth.shape[2] != 3:
        raise ValueError(
            "letools_act treats RGB and compatible depth images as 3-channel visual streams, "
            f"got rgb_channels={rgb.shape[2]}, depth_channels={depth.shape[2]}."
        )
    if rgb.shape[-2:] != (256, 256):
        raise ValueError(f"letools_act expects 256x256 letterboxed inputs, got {tuple(rgb.shape[-2:])}.")

    batch_size, num_views, channels, height, width = rgb.shape
    return torch.stack((rgb, depth), dim=2).reshape(
        batch_size,
        num_views * 2,
        channels,
        height,
        width,
    )


class LeToolsACTVisualEncoder(nn.Module):
    """把多个三通道视觉流编码成连续的二维 feature tokens。

    该模块只复刻 LeTools ACT 的视觉 backbone 边界：所有相机与 depth 兼容图共享
    同一个 torchvision ResNet18，取 layer4 feature map，再使用共享 1×1 Conv 投影到
    DECO hidden dim。它不包含 ACT VAE、Transformer encoder 或 decoder。
    """

    def __init__(
        self,
        *,
        dim: int = 512,
        vision_backbone: str = "resnet18",
        pretrained_backbone_weights: str | None = LETOOLS_RESNET18_WEIGHTS,
        replace_final_stride_with_dilation: bool = False,
    ) -> None:
        super().__init__()
        if dim != 512:
            raise ValueError(f"LeToolsACTVisualEncoder requires dim=512, got {dim}.")
        if not isinstance(vision_backbone, str) or vision_backbone.lower() != "resnet18":
            raise ValueError(f"LeToolsACTVisualEncoder only supports resnet18, got {vision_backbone!r}.")
        if type(replace_final_stride_with_dilation) is not bool:
            raise ValueError("replace_final_stride_with_dilation must be a boolean true/false value.")
        if replace_final_stride_with_dilation:
            raise ValueError("LeTools ACT ResNet18 requires replace_final_stride_with_dilation=False.")
        weights = self._resolve_weights(pretrained_backbone_weights)
        backbone_model = torchvision.models.resnet18(
            replace_stride_with_dilation=[False, False, False],
            weights=weights,
            norm_layer=FrozenBatchNorm2d,
        )
        backbone_channels = backbone_model.fc.in_features
        self.backbone = IntermediateLayerGetter(
            backbone_model,
            return_layers={"layer4": "feature_map"},
        )
        self.input_projection = nn.Conv2d(backbone_channels, dim, kernel_size=1)
        self.dim = dim

    @staticmethod
    def _resolve_weights(weights_name: str | None) -> ResNet18_Weights | None:
        """显式解析允许的权重名，避免使用 eval 解释任意配置字符串。"""

        if weights_name is None:
            return None
        if weights_name == LETOOLS_RESNET18_WEIGHTS:
            return ResNet18_Weights.IMAGENET1K_V1
        raise ValueError(
            "Unsupported LeTools ACT ResNet18 weights. Expected None or "
            f"{LETOOLS_RESNET18_WEIGHTS!r}, got {weights_name!r}."
        )

    def forward(self, visual_streams: Tensor) -> tuple[Tensor, int, int]:
        """编码 `[B,S,3,H,W]`，返回 `[B,S,H'W',D]` 与 feature map 尺寸。"""

        if visual_streams.ndim != 5:
            raise ValueError(
                "LeTools ACT visual streams must be [B,S,3,H,W], "
                f"got {tuple(visual_streams.shape)}."
            )
        batch_size, num_streams, channels, height, width = visual_streams.shape
        if num_streams != 6:
            raise ValueError(f"LeTools ACT visual encoder requires exactly six streams, got {num_streams}.")
        if channels != 3:
            raise ValueError(f"Every LeTools ACT visual stream must have 3 channels, got {channels}.")
        if (height, width) != (256, 256):
            raise ValueError(f"Every LeTools ACT visual stream must be 256x256, got {(height, width)}.")
        flat_streams = visual_streams.reshape(batch_size * num_streams, channels, height, width)
        feature_map = self.backbone(flat_streams)["feature_map"]
        projected = self.input_projection(feature_map)
        feat_h, feat_w = projected.shape[-2:]
        tokens = projected.flatten(start_dim=2).transpose(1, 2)
        tokens = tokens.reshape(batch_size, num_streams, feat_h * feat_w, self.dim)
        return tokens, feat_h, feat_w
