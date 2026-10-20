from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from kuavo_train.wrapper.policy.deco.DECOConfigWrapper import CustomDECOConfigWrapper


@dataclass
@ProcessorStepRegistry.register(name="deco_3view_rgb_letterbox_processor")
class DECO3ViewRGBLetterboxProcessorStep(ProcessorStep):
    """DECO 固定三视角 RGB 空间预处理。

    该 step 必须放在 LeRobot normalizer 之前：此时 RGB 仍保持 raw 图像尺度语义，
    `letterbox_fill_rgb=128/255` 才对应原始灰色 padding。随机 RGB augmentation 由
    `train_policy.py` 的 AugmentationProcessorStep 插入到本 step 之后、normalizer 之前。
    """

    rgb_keys: list[str]
    resize_shape: tuple[int, int] = (256, 256)
    use_letterbox: bool = True
    letterbox_fill_rgb: float = 128.0 / 255.0

    def __post_init__(self) -> None:
        """阻止被手工构造或反序列化的 processor 绕过 policy 配置校验。"""

        if len(self.rgb_keys) != 3 or len(set(self.rgb_keys)) != 3:
            raise ValueError(
                "DECO 3View RGB processor requires exactly three unique rgb_keys "
                "in head/left-wrist/right-wrist order."
            )
        if len(self.resize_shape) != 2 or any(size <= 0 for size in self.resize_shape):
            raise ValueError(f"resize_shape must contain two positive integers, got {self.resize_shape!r}.")

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("DECO 3View RGB processor requires an observation mapping.")

        new_observation = dict(observation)
        source_spatial_shape: tuple[int, int] | None = None
        for key in self.rgb_keys:
            if key not in new_observation:
                raise ValueError(f"Missing 3View RGB input feature before letterbox: {key}")
            source = torch.as_tensor(new_observation[key])
            if source.ndim < 3 or source.shape[-3] != 3:
                raise ValueError(
                    f"{key} must contain RGB tensors with trailing shape [3,H,W], "
                    f"got {tuple(source.shape)}."
                )
            current_spatial_shape = tuple(source.shape[-2:])
            if source_spatial_shape is None:
                source_spatial_shape = current_spatial_shape
            elif current_spatial_shape != source_spatial_shape:
                raise ValueError(
                    "All 3View RGB inputs must share source spatial shape before letterbox; "
                    f"{key} got {current_spatial_shape}, expected {source_spatial_shape}."
                )
            new_observation[key] = self._resize_tensor(
                source,
                fill_value=self.letterbox_fill_rgb,
            )
        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def _resize_tensor(self, value: Any, *, fill_value: float) -> torch.Tensor:
        tensor = torch.as_tensor(value)
        if tensor.ndim < 3:
            raise ValueError(f"DECO 3View RGB input must have channel/height/width dims, got {tuple(tensor.shape)}")
        if not tensor.is_floating_point():
            # LeRobot video loader normally returns float [0, 1]；该分支兼容 image/parquet 中的 uint8。
            tensor = tensor.to(dtype=torch.float32) / 255.0

        prefix_shape = tensor.shape[:-3]
        channels, height, width = tensor.shape[-3:]
        flat = tensor.reshape(-1, channels, height, width)

        if self.use_letterbox:
            resized = self._letterbox(flat, fill_value=fill_value)
        else:
            resized = self._interpolate(flat, size=tuple(self.resize_shape))

        return resized.reshape(*prefix_shape, channels, *self.resize_shape)

    def _letterbox(self, tensor: torch.Tensor, *, fill_value: float) -> torch.Tensor:
        target_h, target_w = self.resize_shape
        _, _, height, width = tensor.shape
        scale = min(target_h / height, target_w / width)
        new_h = max(int(height * scale), 1)
        new_w = max(int(width * scale), 1)
        resized = self._interpolate(tensor, size=(new_h, new_w))
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        return F.pad(resized, (left, right, top, bottom), value=fill_value)

    def _interpolate(self, tensor: torch.Tensor, *, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)

    def get_config(self) -> dict[str, Any]:
        """保存 3View RGB letterbox 构造参数，供部署阶段从 JSON 中完整还原。"""
        return {
            "rgb_keys": list(self.rgb_keys),
            "resize_shape": list(self.resize_shape),
            "use_letterbox": self.use_letterbox,
            "letterbox_fill_rgb": self.letterbox_fill_rgb,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        transformed = {
            feature_type: dict(feature_map)
            for feature_type, feature_map in features.items()
        }
        observation_features = transformed.get(PipelineFeatureType.OBSERVATION, {})
        for key in self.rgb_keys:
            feature = observation_features.get(key)
            if feature is None:
                raise ValueError(f"Missing 3View RGB feature metadata: {key}")
            if len(feature.shape) != 3 or feature.shape[0] != 3:
                raise ValueError(f"{key} must be RGB feature shape (3,H,W), got {tuple(feature.shape)}")
            observation_features[key] = PolicyFeature(
                type=feature.type,
                shape=(feature.shape[0], *self.resize_shape),
            )
        return transformed


def make_deco_pre_post_processors(
    config: CustomDECOConfigWrapper,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """构建 DECO 专用 pre/postprocessor。

    顺序固定为：raw 3View RGB -> batch/device -> DECO letterbox -> RGB augmentation
    （由训练入口插入）-> LeRobot normalizer。tactile 在 normalizer 中为 IDENTITY。
    """

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        DECO3ViewRGBLetterboxProcessorStep(
            rgb_keys=list(config.rgb_keys),
            resize_shape=tuple(config.resize_shape),
            use_letterbox=config.use_letterbox,
            letterbox_fill_rgb=config.letterbox_fill_rgb,
        ),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
