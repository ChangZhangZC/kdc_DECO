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
FIXED_RGB_VIEW_COUNT = 3
DEFAULT_RGB_KEYS = (
    "observation.images.head_cam_h",
    "observation.images.wrist_cam_l",
    "observation.images.wrist_cam_r",
)


def _is_positive_number(value: float | int | None) -> bool:
    return value is not None and float(value) > 0.0


@PreTrainedConfig.register_subclass("custom_deco")
@dataclass
class CustomDECOConfigWrapper(PreTrainedConfig):
    """Kuavo-DECO 的 LeRobot policy 配置。

    该配置只保留固定三视角 RGB + state + 可选 tactile adapter 路线需要的字段。
    三个 RGB key 的顺序固定表示 head、left wrist、right wrist；模型不读取 depth。
    `.safetensors` 加载、两阶段冻结和 DECO 原生 `.pth` 兼容均由 wrapper 处理，
    不再塞回 `third_party/deco` 模型主体。
    """

    # 数据时间窗：DECO 当前只消费当前观测，但预测一个 action chunk。
    chunk_size: int = 32
    drop_n_last_frames: int | None = None

    # 模型结构。action_dim 等历史字段继续保留，以兼容旧 checkpoint 的配置反序列化；
    # 活动训练配置不再要求用户填写，而是由 end_effector_profile 统一派生。
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

    # 两阶段训练与 tactile adapter。三个布尔字段保留给旧 checkpoint 读取；
    # 新配置使用 None 表示按 training_stage 派生，训练入口会再次强制阶段契约。
    training_stage: str = "visual_main"
    use_tactile: bool | None = None
    use_tactile_lora: bool | None = None
    tactile_lora_rank: int = 32
    freeze_pretrained_main: bool | None = None
    tactile_left_max: float | None = None
    tactile_right_max: float | None = None
    clip_tactile_to_unit: bool = True

    # 权重入口语义：
    # base/adapter 面向 LeRobot `.safetensors` policy 目录，deco_init_pth_path 仅兼容原生 `.pth`。
    load_external_init_weights: bool = False
    base_policy_path: str | None = None
    adapter_model_path: str | None = None
    deco_init_pth_path: str | None = None

    # 训练/部署预处理字段。
    # key 名称允许按数据集实际 schema 修改，但三个位置的物理语义和顺序不可改变。
    rgb_keys: tuple[str, str, str] = DEFAULT_RGB_KEYS
    tactile_key: str = "observation.tactile"
    resize_shape: tuple[int, int] = (256, 256)
    use_letterbox: bool = True
    letterbox_fill_rgb: float = 128.0 / 255.0
    dataset_hz: int = 30
    # Receding Horizon 连续执行原始 chunk 的前 N 步；None 表示完整消费 chunk。
    n_action_steps: int | None = 16

    # Normalizer 只负责 LeRobot 统计归一化；tactile 在这里显式保持 IDENTITY。
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
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
        self._apply_training_stage_contract(force=False)
        self._set_and_validate_temporal_window()
        self._validate_end_effector_profile()
        self._validate_rgb_keys()
        self._validate_tactile_limits()
        self._validate_frequency()
        self._select_model_input_features()

    def _convert_omegaconf_fields(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (DictConfig, ListConfig)):
                setattr(self, f.name, OmegaConf.to_container(value, resolve=True))
        self.resize_shape = tuple(self.resize_shape)
        self.rope_axes_dim = tuple(self.rope_axes_dim)
        self.optimizer_betas = tuple(self.optimizer_betas)
        self.rgb_keys = tuple(self.rgb_keys)

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
        # action_dim 是 profile 的固定信息，不再作为用户可选项。即使旧 checkpoint
        # 中带有该字段，也以当前 profile 的 schema 为准，随后由 validate_features
        # 对数据集实际 state/action shape 做最终核验。
        self.action_dim = expected_action_dim
        if self.end_effector_profile == GRIPPER_NO_TACTILE_PROFILE:
            if self.training_stage == "tactile_adapter":
                raise ValueError("gripper_no_tactile cannot enter tactile_adapter training_stage.")

    def _validate_rgb_keys(self) -> None:
        if len(self.rgb_keys) != FIXED_RGB_VIEW_COUNT:
            raise ValueError(
                f"3View RGB requires exactly {FIXED_RGB_VIEW_COUNT} rgb_keys in "
                "head/left-wrist/right-wrist order, "
                f"got {len(self.rgb_keys)}: {self.rgb_keys!r}."
            )
        if any(not isinstance(key, str) or not key.strip() for key in self.rgb_keys):
            raise ValueError(f"rgb_keys must contain three non-empty strings, got {self.rgb_keys!r}.")
        if len(set(self.rgb_keys)) != FIXED_RGB_VIEW_COUNT:
            raise ValueError(f"rgb_keys must be unique, got {self.rgb_keys!r}.")

    def _apply_training_stage_contract(self, *, force: bool) -> None:
        """把两种训练模式固化为唯一合法组合。

        新 YAML 不再暴露三个布尔字段，因此 None 会按阶段派生。直接加载旧
        checkpoint 时保留其显式布尔值以维持模型结构；训练入口使用 force=True，
        让今后的训练严格只有 visual_main 与 tactile_adapter 两种模式。
        """

        if self.training_stage not in {"visual_main", "tactile_adapter"}:
            raise ValueError("training_stage must be 'visual_main' or 'tactile_adapter'.")
        stage_values = {
            "visual_main": (False, False, False),
            "tactile_adapter": (True, True, True),
        }
        expected_tactile, expected_lora, expected_freeze = stage_values[self.training_stage]
        if force or self.use_tactile is None:
            self.use_tactile = expected_tactile
        if force or self.use_tactile_lora is None:
            self.use_tactile_lora = expected_lora
        if force or self.freeze_pretrained_main is None:
            self.freeze_pretrained_main = expected_freeze

    def _validate_tactile_limits(self) -> None:
        if self.training_stage == "tactile_adapter" and self.end_effector_profile != QIANGNAO_TACTILE_PROFILE:
            raise ValueError("tactile_adapter stage is only valid for qiangnao_tactile.")
        if self.use_tactile:
            if not _is_positive_number(self.tactile_left_max) or not _is_positive_number(self.tactile_right_max):
                raise ValueError(
                    "use_tactile=True requires positive tactile_left_max and tactile_right_max "
                    "based on Kuavo tactile values after normal_force / 100."
                )

    @staticmethod
    def _normalize_optional_path(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    def configure_training_initialization(
        self,
        *,
        resume: bool,
        resume_timestamp: str | None,
        deco_init_pth_path: str | None,
    ) -> None:
        """在训练构造模型前校验 resume 与三类权重入口。

        `training.resume` 恢复同一次运行的 policy、优化器、scheduler、AMP、RNG
        和进度；三个外部路径都只做权重初始化，因此必须与 resume 互斥。
        该校验刻意不放进 ``__post_init__``，以便路径已清空的自包含旧
        checkpoint 仍可用于部署，或由 resume 流程完整恢复。
        """

        # 无论旧 YAML 是否还带有历史布尔字段，进入训练后都只允许当前两种
        # 固定模式；随后重新筛选 features，确保 tactile 输入与阶段一致。
        self._apply_training_stage_contract(force=True)
        self._validate_end_effector_profile()
        self._validate_tactile_limits()
        self._select_model_input_features()

        training_pth = self._normalize_optional_path(deco_init_pth_path)
        legacy_policy_pth = self._normalize_optional_path(self.deco_init_pth_path)
        if training_pth and legacy_policy_pth and training_pth != legacy_policy_pth:
            raise ValueError(
                "deco_init_pth_path is set differently under training and policy; "
                "请只保留 training.deco_init_pth_path。"
            )

        # 兼容旧 YAML 曾把 `.pth` 路径放在 policy 下；新配置只展示 training 入口。
        self.deco_init_pth_path = training_pth or legacy_policy_pth
        self.base_policy_path = self._normalize_optional_path(self.base_policy_path)
        self.adapter_model_path = self._normalize_optional_path(self.adapter_model_path)
        active_paths = {
            "deco_init_pth_path": self.deco_init_pth_path,
            "base_policy_path": self.base_policy_path,
            "adapter_model_path": self.adapter_model_path,
        }
        configured_paths = [name for name, path in active_paths.items() if path]

        if resume:
            if not self._normalize_optional_path(resume_timestamp):
                raise ValueError("training.resume=true requires a non-empty resume_timestamp.")
            if self.load_external_init_weights or configured_paths:
                raise ValueError(
                    "training.resume=true performs an exact same-run resume and cannot be combined "
                    "with load_external_init_weights or external weight paths."
                )
            return

        if self.training_stage == "visual_main":
            if self.base_policy_path or self.adapter_model_path:
                raise ValueError(
                    "visual_main only supports training from scratch or legacy `.pth` warm start; "
                    "base_policy_path/adapter_model_path are tactile_adapter entrances."
                )
            if self.deco_init_pth_path and not self.load_external_init_weights:
                raise ValueError(
                    "training.deco_init_pth_path requires policy.load_external_init_weights=true."
                )
            if self.load_external_init_weights and not self.deco_init_pth_path:
                raise ValueError(
                    "visual_main with load_external_init_weights=true requires "
                    "training.deco_init_pth_path."
                )
            return

        if self.deco_init_pth_path:
            raise ValueError(
                "training.deco_init_pth_path is only valid for visual_main; "
                "tactile_adapter must use base_policy_path or adapter_model_path."
            )
        adapter_sources = [path for path in (self.base_policy_path, self.adapter_model_path) if path]
        if not self.load_external_init_weights:
            raise ValueError(
                "tactile_adapter requires policy.load_external_init_weights=true so the frozen "
                "Visual Main + Action Decoder cannot remain randomly initialized."
            )
        if len(adapter_sources) != 1:
            raise ValueError(
                "tactile_adapter requires exactly one of base_policy_path (new adapter training) "
                "or adapter_model_path (weight-only adapter continuation)."
            )

    def _validate_frequency(self) -> None:
        if isinstance(self.dataset_hz, bool) or not isinstance(self.dataset_hz, int) or self.dataset_hz <= 0:
            raise ValueError("dataset_hz must be a positive integer.")
        if self.n_action_steps is not None:
            if (
                isinstance(self.n_action_steps, bool)
                or not isinstance(self.n_action_steps, int)
                or self.n_action_steps <= 0
            ):
                raise ValueError("n_action_steps must be a positive integer or null.")
            if self.n_action_steps > self.chunk_size:
                raise ValueError(
                    "n_action_steps cannot exceed the original DECO chunk_size "
                    f"({self.chunk_size})."
                )

    def _select_model_input_features(self) -> None:
        """把数据集 feature 收窄为 DECO 真正消费的三路 RGB、state 与可选 tactile。

        已有 LeRobot 数据集可以继续保留 depth 或其他相机字段，但它们不会进入 DECO
        processor/normalizer，也不会被序列化为 checkpoint 所需输入。
        """

        # Hydra 在 __post_init__ 期间可能把 PolicyFeature 展平为 dict；这里只需按
        # 数据 key 判断，避免早于训练入口的统一类型恢复逻辑访问 feature.type。
        if (
            self.end_effector_profile == GRIPPER_NO_TACTILE_PROFILE
            and self.tactile_key in self.input_features
        ):
            raise ValueError("gripper_no_tactile dataset must not include observation.tactile.")

        selected_keys = {*self.rgb_keys, OBS_STATE}
        if self.use_tactile:
            selected_keys.add(self.tactile_key)
        self.input_features = {
            key: feature
            for key, feature in self.input_features.items()
            if key in selected_keys
        }

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        return {
            key: ft
            for key, ft in self.input_features.items()
            if ft.type in {FeatureType.VISUAL, getattr(FeatureType, "RGB", FeatureType.VISUAL)}
        }

    @property
    def tactile_feature(self) -> PolicyFeature | None:
        tactile_type = getattr(FeatureType, "TACTILE", None)
        for key, ft in self.input_features.items():
            if key == self.tactile_key or (tactile_type is not None and ft.type is tactile_type):
                return ft
        return None

    def validate_features(self) -> None:
        for key in self.rgb_keys:
            if key not in self.input_features:
                raise ValueError(f"Missing 3View RGB input feature: {key}")
        if OBS_STATE not in self.input_features:
            raise ValueError(f"Missing state input feature: {OBS_STATE}")
        if ACTION not in self.output_features:
            raise ValueError(f"Missing action output feature: {ACTION}")

        rgb_shapes = [tuple(self.input_features[key].shape) for key in self.rgb_keys]
        state_shape = tuple(self.input_features[OBS_STATE].shape)
        action_shape = tuple(self.output_features[ACTION].shape)
        for key, rgb_shape in zip(self.rgb_keys, rgb_shapes):
            if len(rgb_shape) != 3 or rgb_shape[0] != 3:
                raise ValueError(f"{key} must be RGB image shape (3,H,W), got {rgb_shape}")
        if len({shape[-2:] for shape in rgb_shapes}) != 1:
            raise ValueError(
                "All 3View RGB features must share spatial shape, "
                f"got {dict(zip(self.rgb_keys, rgb_shapes))}."
            )
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
