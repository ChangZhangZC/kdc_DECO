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
@ProcessorStepRegistry.register(name="deco_rgbd_letterbox_processor")
class DECORGBDLetterboxProcessorStep(ProcessorStep):
    """DECO 专用 RGB-D 空间预处理。

    该 step 必须放在 LeRobot normalizer 之前：此时 RGB/depth 仍保持 raw 图像尺度语义，
    `letterbox_fill_rgb=128/255` 才对应原始灰色 padding。随机 RGB augmentation 由
    `train_policy.py` 的 AugmentationProcessorStep 插入到本 step 之后、normalizer 之前。
    """

    rgb_keys: list[str]
    depth_keys: list[str]
    resize_shape: tuple[int, int] = (256, 256)
    use_letterbox: bool = True
    letterbox_fill_rgb: float = 128.0 / 255.0
    letterbox_fill_depth: float = 0.0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return new_transition

        new_observation = dict(observation)
        for key in self.rgb_keys:
            if key in new_observation:
                new_observation[key] = self._resize_tensor(
                    new_observation[key],
                    fill_value=self.letterbox_fill_rgb,
                    mode="bilinear",
                )
        for key in self.depth_keys:
            if key in new_observation:
                new_observation[key] = self._resize_tensor(
                    new_observation[key],
                    fill_value=self.letterbox_fill_depth,
                    mode="nearest",
                )
        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def _resize_tensor(self, value: Any, *, fill_value: float, mode: str) -> torch.Tensor:
        tensor = torch.as_tensor(value)
        if tensor.ndim < 3:
            raise ValueError(f"DECO RGB-D input must have channel/height/width dims, got {tuple(tensor.shape)}")
        if not tensor.is_floating_point():
            # LeRobot video loader normally returns float [0, 1]；该分支兼容 image/parquet 中的 uint8。
            tensor = tensor.to(dtype=torch.float32) / 255.0

        prefix_shape = tensor.shape[:-3]
        channels, height, width = tensor.shape[-3:]
        flat = tensor.reshape(-1, channels, height, width)

        if self.use_letterbox:
            resized = self._letterbox(flat, fill_value=fill_value, mode=mode)
        else:
            resized = self._interpolate(flat, size=tuple(self.resize_shape), mode=mode)

        return resized.reshape(*prefix_shape, channels, *self.resize_shape)

    def _letterbox(self, tensor: torch.Tensor, *, fill_value: float, mode: str) -> torch.Tensor:
        target_h, target_w = self.resize_shape
        _, _, height, width = tensor.shape
        scale = min(target_h / height, target_w / width)
        new_h = max(int(height * scale), 1)
        new_w = max(int(width * scale), 1)
        resized = self._interpolate(tensor, size=(new_h, new_w), mode=mode)
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        return F.pad(resized, (left, right, top, bottom), value=fill_value)

    def _interpolate(self, tensor: torch.Tensor, *, size: tuple[int, int], mode: str) -> torch.Tensor:
        kwargs: dict[str, Any] = {"size": size, "mode": mode}
        if mode in {"bilinear", "bicubic"}:
            kwargs["align_corners"] = False
        return F.interpolate(tensor, **kwargs)

    def get_config(self) -> dict[str, Any]:
        """保存 DECO RGB-D letterbox 构造参数，供部署阶段从 JSON 中完整还原。"""
        return {
            "rgb_keys": list(self.rgb_keys),
            "depth_keys": list(self.depth_keys),
            "resize_shape": list(self.resize_shape),
            "use_letterbox": self.use_letterbox,
            "letterbox_fill_rgb": self.letterbox_fill_rgb,
            "letterbox_fill_depth": self.letterbox_fill_depth,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        transformed = {
            feature_type: dict(feature_map)
            for feature_type, feature_map in features.items()
        }
        observation_features = transformed.get(PipelineFeatureType.OBSERVATION, {})
        for key in [*self.rgb_keys, *self.depth_keys]:
            feature = observation_features.get(key)
            if feature is not None and len(feature.shape) == 3:
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

    顺序固定为：raw RGB-D -> batch/device -> DECO letterbox -> RGB augmentation
    （由训练入口插入）-> LeRobot normalizer。tactile 在 normalizer 中为 IDENTITY。
    """

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        DECORGBDLetterboxProcessorStep(
            rgb_keys=[config.rgb_key],
            depth_keys=[config.depth_key],
            resize_shape=tuple(config.resize_shape),
            use_letterbox=config.use_letterbox,
            letterbox_fill_rgb=config.letterbox_fill_rgb,
            letterbox_fill_depth=config.letterbox_fill_depth,
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
