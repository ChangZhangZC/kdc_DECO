from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torchvision.ops.misc import FrozenBatchNorm2d

# 必须先注册 Kuavo 的 DEPTH/TACTILE feature 类型，再导入 LeRobot 类型对象。
import lerobot_patches.custom_patches  # noqa: F401
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import NormalizerProcessorStep
from lerobot.processor.core import TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE

from kuavo_deploy.utils.deco_obs_action import validate_deco_policy_compatibility
from kuavo_train.train_policy import AugmentationProcessorStep, insert_before_normalizer
from kuavo_train.wrapper.policy.deco import ensure_deco_on_path
from kuavo_train.wrapper.policy.deco.DECOConfigWrapper import (
    DEFAULT_DEPTH_KEYS,
    DEFAULT_RGB_KEYS,
    LETOOLS_RESNET18_WEIGHTS,
    CustomDECOConfigWrapper,
)
from kuavo_train.wrapper.policy.deco.DECOProcessor import (
    DECORGBDLetterboxProcessorStep,
    LETOOLS_IMAGENET_STATS,
    _build_deco_normalizer_stats,
    make_deco_pre_post_processors,
)


ensure_deco_on_path()
from models.deco.deco import DECO  # noqa: E402
from models.deco.letools_act_visual_encoder import (  # noqa: E402
    LeToolsACTVisualEncoder,
    interleave_rgb_depth_streams,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _strict_letools_config(**overrides: object) -> CustomDECOConfigWrapper:
    """构造不依赖数据集的严格六流配置，便于逐项验证 fail-fast 约束。"""

    kwargs: dict[str, object] = {
        "device": "cpu",
        "load_external_init_weights": False,
        "visual_fusion_mode": "letools_act",
        "vision_backbone": "resnet18",
        "pretrained_backbone_weights": LETOOLS_RESNET18_WEIGHTS,
        "replace_final_stride_with_dilation": False,
        "use_imagenet_stats": True,
        "use_rgbd_cross_attention": False,
        "normalization_mapping": {
            "VISUAL": NormalizationMode.MEAN_STD,
            "DEPTH": NormalizationMode.MEAN_STD,
        },
    }
    kwargs.update(overrides)
    return CustomDECOConfigWrapper(**kwargs)


def _strict_input_features(*, depth_channels: int = 3) -> dict[str, PolicyFeature]:
    """生成与三组 Kuavo RGB-D 及 28D state 对应的最小 feature schema。"""

    features = {
        key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
        for key in DEFAULT_RGB_KEYS
    }
    features.update(
        {
            key: PolicyFeature(type=FeatureType.DEPTH, shape=(depth_channels, 480, 640))
            for key in DEFAULT_DEPTH_KEYS
        }
    )
    features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(28,))
    return features


def test_six_stream_order_is_rgb_depth_interleaved_per_camera() -> None:
    """顺序必须固定为 head、left、right，且每组内部先 RGB 后 depth。"""

    rgb = torch.zeros(1, 3, 3, 256, 256)
    depth = torch.zeros_like(rgb)
    for view_index in range(3):
        rgb[:, view_index].fill_(view_index + 1)
        depth[:, view_index].fill_(view_index + 11)

    streams = interleave_rgb_depth_streams(rgb, depth)

    assert streams.shape == (1, 6, 3, 256, 256)
    assert streams[0, :, 0, 0, 0].tolist() == [1, 11, 2, 12, 3, 13]


def test_letools_depth_must_remain_three_channel() -> None:
    rgb = torch.zeros(1, 3, 3, 256, 256)
    single_channel_depth = torch.zeros(1, 3, 1, 256, 256)

    with pytest.raises(ValueError, match="3-channel visual streams"):
        interleave_rgb_depth_streams(rgb, single_channel_depth)


def test_shared_resnet18_produces_64_tokens_per_stream() -> None:
    """256x256 经无 dilation 的 ResNet18 layer4 后应固定得到 8x8 feature map。"""

    encoder = LeToolsACTVisualEncoder(
        pretrained_backbone_weights=None,
        replace_final_stride_with_dilation=False,
    )
    visual_streams = torch.zeros(1, 6, 3, 256, 256)

    with torch.no_grad():
        tokens, feature_height, feature_width = encoder(visual_streams)

    assert (feature_height, feature_width) == (8, 8)
    assert tokens.shape == (1, 6, 64, 512)
    assert encoder.input_projection.kernel_size == (1, 1)
    assert any(isinstance(module, FrozenBatchNorm2d) for module in encoder.backbone.modules())


def test_deco_packs_six_rope_segments_without_rgb_depth_fusion() -> None:
    model = DECO(
        act_dim=28,
        chunk_size=2,
        num_attn_blocks=1,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        visual_fusion_mode="letools_act",
        use_rgbd_cross_attention=False,
    )
    per_stream_tokens = torch.zeros(1, 6, 64, 512)

    packed, rotary_embedding, segment_count, tokens_per_segment = model.pack_visual_token_sequences(
        per_stream_tokens,
        8,
        8,
    )

    assert packed.shape == (1, 384, 512)
    assert segment_count == 6
    assert tokens_per_segment == 64
    assert rotary_embedding[0].shape == (64, model.head_dim)
    assert hasattr(model, "letools_visual_encoder")
    assert not hasattr(model, "depth_encoder")
    assert not hasattr(model, "rgb_depth_fusion")
    assert not hasattr(model, "rgb_depth_fusion_proj")


def test_all_six_visual_keys_use_imagenet_stats_without_mutating_dataset_stats() -> None:
    original_rgb_mean = torch.zeros(3, 1, 1)
    dataset_stats = {
        DEFAULT_RGB_KEYS[0]: {
            "mean": original_rgb_mean.clone(),
            "std": torch.ones(3, 1, 1),
        },
        OBS_STATE: {"mean": torch.tensor([7.0]), "std": torch.tensor([2.0])},
    }
    config = SimpleNamespace(
        use_imagenet_stats=True,
        rgb_keys=DEFAULT_RGB_KEYS,
        depth_keys=DEFAULT_DEPTH_KEYS,
    )

    normalizer_stats = _build_deco_normalizer_stats(config, dataset_stats)

    expected_mean = torch.tensor(LETOOLS_IMAGENET_STATS["mean"], dtype=torch.float32)
    expected_std = torch.tensor(LETOOLS_IMAGENET_STATS["std"], dtype=torch.float32)
    assert normalizer_stats is not None
    for key in (*DEFAULT_RGB_KEYS, *DEFAULT_DEPTH_KEYS):
        assert torch.equal(normalizer_stats[key]["mean"], expected_mean)
        assert torch.equal(normalizer_stats[key]["std"], expected_std)
    assert torch.equal(normalizer_stats[OBS_STATE]["mean"], torch.tensor([7.0]))
    assert torch.equal(dataset_stats[DEFAULT_RGB_KEYS[0]]["mean"], original_rgb_mean)


def test_rgb_augmentation_is_disabled_by_default_and_excludes_depth() -> None:
    config = OmegaConf.load(REPO_ROOT / "configs/policy/deco_config.yaml")
    augmentation_step = AugmentationProcessorStep(
        transform=lambda value: value + 1,
        cam_keys=[*DEFAULT_RGB_KEYS, *DEFAULT_DEPTH_KEYS],
    )
    rgb = torch.zeros(3, 8, 8)
    depth = torch.zeros(3, 8, 8)
    transition = {
        TransitionKey.OBSERVATION: {
            DEFAULT_RGB_KEYS[0]: rgb,
            DEFAULT_DEPTH_KEYS[0]: depth,
        }
    }

    augmented = augmentation_step(transition)[TransitionKey.OBSERVATION]

    assert config.training.RGB_Augmenter.enable is False
    assert augmentation_step.cam_keys == list(DEFAULT_RGB_KEYS)
    assert torch.equal(augmented[DEFAULT_RGB_KEYS[0]], rgb + 1)
    assert torch.equal(augmented[DEFAULT_DEPTH_KEYS[0]], depth)


def test_preprocessor_order_is_letterbox_then_rgb_augmentation_then_normalization() -> None:
    """训练入口插入增强 step 后，raw-pixel 空间处理顺序仍必须保持冻结语义。"""

    config = _strict_letools_config(
        input_features=_strict_input_features(),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(28,))},
    )
    dataset_stats = {
        OBS_STATE: {"mean": torch.zeros(28), "std": torch.ones(28)},
        ACTION: {"mean": torch.zeros(28), "std": torch.ones(28)},
    }
    preprocessor, _ = make_deco_pre_post_processors(config, dataset_stats)
    augmentation_step = AugmentationProcessorStep(
        transform=lambda value: value,
        cam_keys=[*DEFAULT_RGB_KEYS, *DEFAULT_DEPTH_KEYS],
    )

    insert_before_normalizer(preprocessor, augmentation_step)
    letterbox_index = next(
        index
        for index, step in enumerate(preprocessor.steps)
        if isinstance(step, DECORGBDLetterboxProcessorStep)
    )
    augmentation_index = preprocessor.steps.index(augmentation_step)
    normalizer_index = next(
        index
        for index, step in enumerate(preprocessor.steps)
        if isinstance(step, NormalizerProcessorStep)
    )

    assert letterbox_index < augmentation_index < normalizer_index


@pytest.mark.parametrize(
    ("overrides", "error_pattern"),
    [
        ({"visual_fusion_mode": "unknown"}, "visual_fusion_mode"),
        ({"vision_backbone": "resnet34"}, "vision_backbone='resnet18'"),
        ({"pretrained_backbone_weights": None}, "pretrained_backbone_weights"),
        ({"replace_final_stride_with_dilation": True}, "replace_final_stride_with_dilation=False"),
        ({"use_imagenet_stats": False}, "use_imagenet_stats=True"),
        ({"use_rgbd_cross_attention": True}, "use_rgbd_cross_attention=False"),
        ({"rgb_keys": DEFAULT_RGB_KEYS[:2]}, "exactly three RGB/depth pairs"),
        (
            {"rgb_keys": (DEFAULT_RGB_KEYS[1], DEFAULT_RGB_KEYS[0], DEFAULT_RGB_KEYS[2])},
            "fixed ordered RGB-D keys",
        ),
        ({"use_letterbox": False}, "use_letterbox=True"),
        (
            {
                "normalization_mapping": {
                    "VISUAL": NormalizationMode.MEAN_STD,
                    "DEPTH": NormalizationMode.MIN_MAX,
                }
            },
            "normalization_mapping.DEPTH=MEAN_STD",
        ),
    ],
)
def test_invalid_letools_frontend_configuration_fails_fast(
    overrides: dict[str, object],
    error_pattern: str,
) -> None:
    with pytest.raises(ValueError, match=error_pattern):
        _strict_letools_config(**overrides)


def test_letools_feature_schema_rejects_single_channel_depth() -> None:
    config = _strict_letools_config(
        input_features=_strict_input_features(depth_channels=1),
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(28,))},
    )

    with pytest.raises(ValueError, match="3-channel depth image"):
        config.validate_features()


def test_legacy_act_rgbd_frontend_still_constructs() -> None:
    model = DECO(
        act_dim=28,
        chunk_size=2,
        num_attn_blocks=1,
        vision_backbone="resnet18",
        depth_backbone="resnet18",
        visual_fusion_mode="act_rgbd",
        use_rgbd_cross_attention=False,
    )

    assert hasattr(model, "img_encoder")
    assert hasattr(model, "depth_encoder")
    assert hasattr(model, "rgb_depth_fusion")
    assert hasattr(model, "rgb_depth_fusion_proj")
    assert not hasattr(model, "letools_visual_encoder")


def test_deployment_visual_key_validation_accepts_letools_mode() -> None:
    policy_config = SimpleNamespace(
        end_effector_profile="gripper_no_tactile",
        action_dim=18,
        use_tactile=False,
        use_tactile_lora=False,
        visual_fusion_mode="letools_act",
        rgb_keys=DEFAULT_RGB_KEYS,
        depth_keys=DEFAULT_DEPTH_KEYS,
    )
    deploy_config = SimpleNamespace(inference_mode="gripper_no_tactile")
    all_short_keys = {
        **{key.removeprefix("observation.images."): object() for key in DEFAULT_RGB_KEYS},
        **{key.removeprefix("observation."): object() for key in DEFAULT_DEPTH_KEYS},
    }
    env_config = SimpleNamespace(
        state_layout="deco_18d",
        eef_type="rq2f85",
        obs_key_map=all_short_keys,
    )

    validate_deco_policy_compatibility(policy_config, deploy_config, env_config)

    env_config.obs_key_map.pop("depth_r")
    with pytest.raises(ValueError, match="missing visual keys"):
        validate_deco_policy_compatibility(policy_config, deploy_config, env_config)
