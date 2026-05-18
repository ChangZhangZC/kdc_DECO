from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import draccus
from huggingface_hub.constants import CONFIG_NAME
from omegaconf import DictConfig, ListConfig, OmegaConf

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE


def _is_positive_number(value: float | int | None) -> bool:
    return value is not None and float(value) > 0.0


@PreTrainedConfig.register_subclass("custom_deco")
@dataclass
class CustomDECOConfigWrapper(PreTrainedConfig):
    """Kuavo-DECO 的 LeRobot policy 配置。

    该配置只保留当前 RGB-D + state + 可选 tactile adapter 路线需要的字段。
    `.safetensors` 加载、两阶段冻结和 DECO 原生 `.pth` 兼容均由 wrapper 处理，
    不再塞回 `third_party/deco` 模型主体。
    """

    # 数据时间窗：DECO 当前只消费当前观测，但预测一个 action chunk。
    chunk_size: int = 32
    drop_n_last_frames: int | None = None

    # 模型结构。
    action_dim: int = 28
    obs_state: bool = True
    use_task_condition: bool = False
    num_tasks: int = 10
    num_attn_blocks: int = 6
    inf_step: int = 5
    heads: int = 8
    dim: int = 512
    rope_axes_dim: tuple[int, int] = (256, 256)
    vision_backbone: str = "resnet34"
    depth_backbone: str = "resnet34"

    # 两阶段训练与 tactile adapter。
    training_stage: str = "visual_main"
    use_tactile: bool = False
    use_tactile_lora: bool = False
    tactile_lora_rank: int = 32
    freeze_pretrained_main: bool = True
    tactile_left_max: float | None = None
    tactile_right_max: float | None = None
    clip_tactile_to_unit: bool = True

    # 权重入口语义：
    # base/adapter 面向 LeRobot `.safetensors` policy 目录，deco_init_pth_path 仅兼容原生 `.pth`。
    load_external_init_weights: bool = True
    base_policy_path: str | None = None
    adapter_model_path: str | None = None
    deco_init_pth_path: str | None = None

    # 训练/部署预处理字段。
    rgb_key: str = "observation.images.head_cam_h"
    depth_key: str = "observation.depth_h"
    tactile_key: str = "observation.tactile"
    resize_shape: tuple[int, int] = (256, 256)
    use_letterbox: bool = True
    letterbox_fill_rgb: float = 128.0 / 255.0
    letterbox_fill_depth: float = 0.0
    dataset_hz: int = 30
    control_hz: int = 10
    action_stride: int = 3

    # Normalizer 只负责 LeRobot 统计归一化；tactile 在这里显式保持 IDENTITY。
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "DEPTH": NormalizationMode.MIN_MAX,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
            "TACTILE": NormalizationMode.IDENTITY,
        }
    )

    # 优化器配置保持最小集合，避免把 ACT/DP 无关字段带进 DECO。
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    # 兼容现有 Kuavo config 的 custom 注入；若字段已存在则要求用户直接改顶层字段。
    custom: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._convert_omegaconf_fields()
        self._merge_custom_fields()
        self._merge_default_normalization_mapping()
        self._validate_stage_and_tactile()
        self._validate_frequency()
        if self.drop_n_last_frames is None:
            self.drop_n_last_frames = max(self.chunk_size - 1, 0)

    def _convert_omegaconf_fields(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (DictConfig, ListConfig)):
                setattr(self, f.name, OmegaConf.to_container(value, resolve=True))
        self.resize_shape = tuple(self.resize_shape)
        self.rope_axes_dim = tuple(self.rope_axes_dim)
        self.optimizer_betas = tuple(self.optimizer_betas)

    def _merge_custom_fields(self) -> None:
        if not isinstance(self.custom, dict):
            return
        for key, value in self.custom.items():
            if hasattr(self, key):
                raise ValueError(
                    f"Custom setting '{key}: {value}' conflicts with DECO config. "
                    "请直接修改 policy 顶层字段，避免同一含义出现两处配置。"
                )
            setattr(self, key, value)

    def _merge_default_normalization_mapping(self) -> None:
        default_map = {
            "VISUAL": NormalizationMode.MEAN_STD,
            "DEPTH": NormalizationMode.MIN_MAX,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
            "TACTILE": NormalizationMode.IDENTITY,
        }
        merged = dict(default_map)
        merged.update(self.normalization_mapping)
        self.normalization_mapping = merged

    def _validate_stage_and_tactile(self) -> None:
        if self.training_stage not in {"visual_main", "tactile_adapter"}:
            raise ValueError("training_stage must be 'visual_main' or 'tactile_adapter'.")
        if self.training_stage == "tactile_adapter":
            if not self.use_tactile or not self.use_tactile_lora:
                raise ValueError("tactile_adapter stage requires use_tactile=True and use_tactile_lora=True.")
            if (
                self.load_external_init_weights
                and not self.base_policy_path
                and not self.deco_init_pth_path
                and not self.adapter_model_path
            ):
                raise ValueError(
                    "tactile_adapter stage requires base_policy_path, deco_init_pth_path, or adapter_model_path."
                )
        if self.use_tactile:
            if not _is_positive_number(self.tactile_left_max) or not _is_positive_number(self.tactile_right_max):
                raise ValueError(
                    "use_tactile=True requires positive tactile_left_max and tactile_right_max "
                    "based on Kuavo tactile values after normal_force / 100."
                )

    def _validate_frequency(self) -> None:
        if self.dataset_hz <= 0 or self.control_hz <= 0:
            raise ValueError("dataset_hz and control_hz must be positive.")
        if self.dataset_hz % self.control_hz != 0:
            raise ValueError("dataset_hz must be divisible by control_hz for deterministic action_stride.")
        expected_stride = self.dataset_hz // self.control_hz
        if self.action_stride != expected_stride:
            raise ValueError(f"action_stride must be {expected_stride} for dataset_hz/control_hz.")

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        return {
            key: ft
            for key, ft in self.input_features.items()
            if ft.type in {FeatureType.VISUAL, getattr(FeatureType, "RGB", FeatureType.VISUAL)}
        }

    @property
    def depth_features(self) -> dict[str, PolicyFeature]:
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.DEPTH}

    @property
    def tactile_feature(self) -> PolicyFeature | None:
        tactile_type = getattr(FeatureType, "TACTILE", None)
        for key, ft in self.input_features.items():
            if key == self.tactile_key or (tactile_type is not None and ft.type is tactile_type):
                return ft
        return None

    def validate_features(self) -> None:
        if self.rgb_key not in self.input_features:
            raise ValueError(f"Missing RGB input feature: {self.rgb_key}")
        if self.depth_key not in self.input_features:
            raise ValueError(f"Missing depth input feature: {self.depth_key}")
        if OBS_STATE not in self.input_features:
            raise ValueError(f"Missing state input feature: {OBS_STATE}")
        if ACTION not in self.output_features:
            raise ValueError(f"Missing action output feature: {ACTION}")

        state_shape = tuple(self.input_features[OBS_STATE].shape)
        action_shape = tuple(self.output_features[ACTION].shape)
        if state_shape != (self.action_dim,):
            raise ValueError(f"observation.state must be ({self.action_dim},), got {state_shape}")
        if action_shape != (self.action_dim,):
            raise ValueError(f"action must be ({self.action_dim},), got {action_shape}")

        if self.use_tactile:
            tactile = self.tactile_feature
            if tactile is None:
                raise ValueError(f"Missing tactile input feature: {self.tactile_key}")
            if tuple(tactile.shape) != (30,):
                raise ValueError(f"observation.tactile must be (30,), got {tuple(tactile.shape)}")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        cfg_copy = deepcopy(self)
        # 训练初始化路径不是最终 policy 资产的一部分。保存后的 `.safetensors`
        # 已经包含完整权重，部署或迁移时不应再访问第一阶段目录或原生 `.pth`。
        cfg_copy.load_external_init_weights = False
        cfg_copy.base_policy_path = None
        cfg_copy.adapter_model_path = None
        cfg_copy.deco_init_pth_path = None
        with open(save_directory / CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(cfg_copy, f, indent=4)
