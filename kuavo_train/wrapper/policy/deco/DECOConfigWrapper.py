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


QIANGNAO_TACTILE_PROFILE = "qiangnao_tactile"
GRIPPER_NO_TACTILE_PROFILE = "gripper_no_tactile"
SUPPORTED_END_EFFECTOR_PROFILES = {QIANGNAO_TACTILE_PROFILE, GRIPPER_NO_TACTILE_PROFILE}
ACT_RGBD_FRONTEND = "act_rgbd"
LETOOLS_ACT_FRONTEND = "letools_act"
SUPPORTED_VISUAL_FUSION_MODES = {ACT_RGBD_FRONTEND, LETOOLS_ACT_FRONTEND}
LETOOLS_RESNET18_WEIGHTS = "ResNet18_Weights.IMAGENET1K_V1"
DEFAULT_RGB_KEYS = (
    "observation.images.head_cam_h",
    "observation.images.wrist_cam_l",
    "observation.images.wrist_cam_r",
)
DEFAULT_DEPTH_KEYS = (
    "observation.depth_h",
    "observation.depth_l",
    "observation.depth_r",
)


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
    # end_effector_profile 决定 state/action 的物理 schema；action_dim 必须与 profile 一致。
    end_effector_profile: str = QIANGNAO_TACTILE_PROFILE
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
    # LeTools ACT 使用 torchvision ImageNet ResNet18；旧 act_rgbd checkpoint 缺少这些字段时
    # 继续使用 None/False/False 默认值，避免重建旧 preprocessor 时静默改变视觉数值分布。
    pretrained_backbone_weights: str | None = None
    replace_final_stride_with_dilation: bool = False
    use_imagenet_stats: bool = False
    # 默认值保留 act_rgbd 以兼容旧 checkpoint；新分支 YAML 显式选择严格六流 letools_act。
    visual_fusion_mode: str = ACT_RGBD_FRONTEND
    # 仅旧 act_rgbd 可启用每个相机内部 RGB/depth token 双向 cross attention。
    use_rgbd_cross_attention: bool = False

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
    rgb_keys: tuple[str, ...] = DEFAULT_RGB_KEYS
    depth_keys: tuple[str, ...] = DEFAULT_DEPTH_KEYS
    # legacy fallback：旧配置若只保存单 key，可在 _convert_omegaconf_fields 中提升为列表。
    rgb_key: str | None = None
    depth_key: str | None = None
    tactile_key: str = "observation.tactile"
    resize_shape: tuple[int, int] = (256, 256)
    use_letterbox: bool = True
    letterbox_fill_rgb: float = 128.0 / 255.0
    letterbox_fill_depth: float = 0.5
    dataset_hz: int = 30
    control_hz: int = 10
    action_stride: int = 3
    # Receding Horizon 连续执行步数；None 表示完整消费原始 chunk。
    # action_stride/control_hz 仅为旧 checkpoint 与显式 Stride Action 模式保留。
    n_action_steps: int | None = None

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
        self._set_and_validate_temporal_window()
        self._validate_end_effector_profile()
        self._validate_visual_fusion_mode()
        self._validate_rgbd_cross_attention()
        self._validate_letools_visual_frontend()
        self._validate_stage_and_tactile()
        self._validate_frequency()

    def _convert_omegaconf_fields(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (DictConfig, ListConfig)):
                setattr(self, f.name, OmegaConf.to_container(value, resolve=True))
        self.resize_shape = tuple(self.resize_shape)
        self.rope_axes_dim = tuple(self.rope_axes_dim)
        self.optimizer_betas = tuple(self.optimizer_betas)
        self.rgb_keys = tuple(self.rgb_keys)
        self.depth_keys = tuple(self.depth_keys)
        if self.rgb_key is not None and self.rgb_keys not in {DEFAULT_RGB_KEYS, (self.rgb_key,)}:
            raise ValueError("Do not set both legacy rgb_key and new rgb_keys with different values.")
        if self.depth_key is not None and self.depth_keys not in {DEFAULT_DEPTH_KEYS, (self.depth_key,)}:
            raise ValueError("Do not set both legacy depth_key and new depth_keys with different values.")
        if self.rgb_key is not None:
            self.rgb_keys = (self.rgb_key,)
        if self.depth_key is not None:
            self.depth_keys = (self.depth_key,)

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

    def _set_and_validate_temporal_window(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        required_drop = self.chunk_size - 1
        if self.drop_n_last_frames is None:
            self.drop_n_last_frames = required_drop
        if self.drop_n_last_frames < required_drop:
            raise ValueError(
                "drop_n_last_frames must be >= chunk_size - 1 because DECO loss "
                "does not consume action_is_pad. 请不要让尾部 padded action 进入训练。"
            )

    def _validate_end_effector_profile(self) -> None:
        if self.end_effector_profile not in SUPPORTED_END_EFFECTOR_PROFILES:
            raise ValueError(
                "end_effector_profile must be 'qiangnao_tactile' or 'gripper_no_tactile'."
            )
        expected_action_dim = 28 if self.end_effector_profile == QIANGNAO_TACTILE_PROFILE else 18
        if self.action_dim != expected_action_dim:
            raise ValueError(
                f"{self.end_effector_profile} requires action_dim={expected_action_dim}, "
                f"got action_dim={self.action_dim}."
            )
        if self.end_effector_profile == GRIPPER_NO_TACTILE_PROFILE:
            if self.use_tactile or self.use_tactile_lora:
                raise ValueError(
                    "gripper_no_tactile has no observation.tactile; keep use_tactile=False "
                    "and use_tactile_lora=False."
                )
            if self.training_stage == "tactile_adapter":
                raise ValueError("gripper_no_tactile cannot enter tactile_adapter training_stage.")

    def _validate_visual_fusion_mode(self) -> None:
        if self.visual_fusion_mode not in SUPPORTED_VISUAL_FUSION_MODES:
            raise ValueError(
                "visual_fusion_mode must be 'act_rgbd' or 'letools_act'. "
                f"got {self.visual_fusion_mode!r}."
            )

    def _validate_rgbd_cross_attention(self) -> None:
        if type(self.use_rgbd_cross_attention) is not bool:
            raise ValueError(
                "use_rgbd_cross_attention must be a boolean true/false value, "
                f"got {self.use_rgbd_cross_attention!r}."
            )
        if self.visual_fusion_mode == LETOOLS_ACT_FRONTEND and self.use_rgbd_cross_attention:
            raise ValueError(
                "letools_act uses six independent shared-ResNet visual streams and therefore requires "
                "use_rgbd_cross_attention=False."
            )

    def _validate_letools_visual_frontend(self) -> None:
        """约束严格 LeTools ACT 六流前端，避免配置字段被静默忽略。"""

        if self.visual_fusion_mode != LETOOLS_ACT_FRONTEND:
            return
        if len(self.rgb_keys) != 3 or len(self.depth_keys) != 3:
            raise ValueError(
                "letools_act requires exactly three RGB/depth pairs: head, left wrist, and right wrist."
            )
        if self.rgb_keys != DEFAULT_RGB_KEYS or self.depth_keys != DEFAULT_DEPTH_KEYS:
            raise ValueError(
                "letools_act requires the fixed ordered RGB-D keys: "
                "head, left wrist, then right wrist."
            )
        if type(self.use_letterbox) is not bool:
            raise ValueError("letools_act requires use_letterbox to be a boolean true/false value.")
        if self.resize_shape != (256, 256) or not self.use_letterbox:
            raise ValueError("letools_act requires use_letterbox=True and resize_shape=(256, 256).")
        if not isinstance(self.vision_backbone, str):
            raise ValueError("letools_act requires vision_backbone='resnet18'.")
        if self.vision_backbone.lower() != "resnet18":
            raise ValueError("letools_act requires vision_backbone='resnet18'.")
        if self.pretrained_backbone_weights != LETOOLS_RESNET18_WEIGHTS:
            raise ValueError(
                "letools_act requires pretrained_backbone_weights="
                f"'{LETOOLS_RESNET18_WEIGHTS}'."
            )
        if type(self.replace_final_stride_with_dilation) is not bool:
            raise ValueError(
                "letools_act requires replace_final_stride_with_dilation to be a boolean true/false value."
            )
        if self.replace_final_stride_with_dilation:
            raise ValueError("letools_act requires replace_final_stride_with_dilation=False.")
        if type(self.use_imagenet_stats) is not bool:
            raise ValueError("letools_act requires use_imagenet_stats to be a boolean true/false value.")
        if not self.use_imagenet_stats:
            raise ValueError("letools_act requires use_imagenet_stats=True for all six visual streams.")
        if self.dim != 512:
            raise ValueError("letools_act requires dim=512 to match the LeTools ACT projection width.")
        for feature_type in ("VISUAL", "DEPTH"):
            if self.normalization_mapping.get(feature_type) != NormalizationMode.MEAN_STD:
                raise ValueError(
                    f"letools_act requires normalization_mapping.{feature_type}=MEAN_STD."
                )

    def _validate_stage_and_tactile(self) -> None:
        if self.training_stage not in {"visual_main", "tactile_adapter"}:
            raise ValueError("training_stage must be 'visual_main' or 'tactile_adapter'.")
        if self.training_stage == "tactile_adapter":
            if self.end_effector_profile != QIANGNAO_TACTILE_PROFILE:
                raise ValueError("tactile_adapter stage is only valid for qiangnao_tactile.")
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
        if self.n_action_steps is not None:
            if self.n_action_steps <= 0:
                raise ValueError("n_action_steps must be positive or null.")
            if self.n_action_steps > self.chunk_size:
                raise ValueError(
                    "n_action_steps cannot exceed the original DECO chunk_size "
                    f"({self.chunk_size})."
                )

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
        if len(self.rgb_keys) == 0 or len(self.depth_keys) == 0:
            raise ValueError("rgb_keys/depth_keys must not be empty.")
        if len(self.rgb_keys) != len(self.depth_keys):
            raise ValueError(
                "rgb_keys and depth_keys must have the same number of views, "
                f"got {len(self.rgb_keys)} RGB and {len(self.depth_keys)} depth."
            )
        for key in self.rgb_keys:
            if key not in self.input_features:
                raise ValueError(f"Missing RGB input feature: {key}")
        for key in self.depth_keys:
            if key not in self.input_features:
                raise ValueError(f"Missing depth input feature: {key}")
        if OBS_STATE not in self.input_features:
            raise ValueError(f"Missing state input feature: {OBS_STATE}")
        if ACTION not in self.output_features:
            raise ValueError(f"Missing action output feature: {ACTION}")

        rgb_shapes = [tuple(self.input_features[key].shape) for key in self.rgb_keys]
        depth_shapes = [tuple(self.input_features[key].shape) for key in self.depth_keys]
        state_shape = tuple(self.input_features[OBS_STATE].shape)
        action_shape = tuple(self.output_features[ACTION].shape)
        for key, shape in zip(self.rgb_keys, rgb_shapes):
            if len(shape) != 3 or shape[0] != 3:
                raise ValueError(f"{key} must be RGB image shape (3,H,W), got {shape}")
        for key, shape in zip(self.depth_keys, depth_shapes):
            valid_channels = {3} if self.visual_fusion_mode == LETOOLS_ACT_FRONTEND else {1, 3}
            if len(shape) != 3 or shape[0] not in valid_channels:
                if self.visual_fusion_mode == LETOOLS_ACT_FRONTEND:
                    raise ValueError(
                        f"{key} must be 3-channel depth image shape (3,H,W) for letools_act, got {shape}"
                    )
                raise ValueError(f"{key} must be depth image shape (1,H,W) or (3,H,W), got {shape}")
        spatial_shapes = {shape[-2:] for shape in [*rgb_shapes, *depth_shapes]}
        if len(spatial_shapes) != 1:
            raise ValueError(f"All RGB/depth inputs must share H,W before DECOProcessor, got {spatial_shapes}")
        if state_shape != (self.action_dim,):
            raise ValueError(f"observation.state must be ({self.action_dim},), got {state_shape}")
        if action_shape != (self.action_dim,):
            raise ValueError(f"action must be ({self.action_dim},), got {action_shape}")

        if self.end_effector_profile == GRIPPER_NO_TACTILE_PROFILE and self.tactile_feature is not None:
            raise ValueError("gripper_no_tactile dataset must not include observation.tactile.")
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
