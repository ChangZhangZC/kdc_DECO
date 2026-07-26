from __future__ import annotations

# 训练入口会先安装 Kuavo 对 FeatureType.DEPTH/TACTILE 的兼容扩展；
# 本测试保持相同导入顺序，覆盖用户实际 Hydra 构造路径。
import lerobot_patches.custom_patches  # noqa: F401
import pytest
from lerobot.configs.types import FeatureType, PolicyFeature

from kuavo_train.wrapper.policy.deco.DECOConfigWrapper import (
    DEFAULT_RGB_KEYS,
    CustomDECOConfigWrapper,
)


def _feature(feature_type: FeatureType | str, shape: list[int]) -> dict[str, object]:
    """模拟 Hydra 将嵌套 PolicyFeature dataclass 展平后的字典。"""

    return {"type": feature_type, "shape": shape}


def test_hydra_feature_dicts_are_restored_before_three_view_selection() -> None:
    """三路 RGB 数据可保留 depth 元数据，但模型配置只选择实际消费的字段。"""

    input_features = {
        "observation.state": _feature("STATE", [18]),
        DEFAULT_RGB_KEYS[0]: _feature(FeatureType.VISUAL, [3, 480, 640]),
        "observation.depth_h": _feature("DEPTH", [3, 480, 640]),
        DEFAULT_RGB_KEYS[1]: _feature(FeatureType.VISUAL, [3, 480, 640]),
        "observation.depth_l": _feature("DEPTH", [3, 480, 640]),
        DEFAULT_RGB_KEYS[2]: _feature(FeatureType.VISUAL, [3, 480, 640]),
        "observation.depth_r": _feature("DEPTH", [3, 480, 640]),
    }
    output_features = {
        "action": _feature("ACTION", [18]),
    }

    config = CustomDECOConfigWrapper(
        end_effector_profile="gripper_no_tactile",
        action_dim=18,
        input_features=input_features,
        output_features=output_features,
        device="cpu",
    )

    assert tuple(config.input_features) == (
        "observation.state",
        *DEFAULT_RGB_KEYS,
    )
    assert all(isinstance(feature, PolicyFeature) for feature in config.input_features.values())
    assert isinstance(config.output_features["action"], PolicyFeature)
    assert config.input_features[DEFAULT_RGB_KEYS[0]].shape == (3, 480, 640)
    assert not any("depth" in key for key in config.input_features)
    config.validate_features()


def test_malformed_hydra_feature_dict_fails_with_field_context() -> None:
    input_features = {
        "observation.state": {"type": "STATE"},
    }

    with pytest.raises(ValueError, match=r"input_features\['observation\.state'\].*shape"):
        CustomDECOConfigWrapper(
            end_effector_profile="gripper_no_tactile",
            action_dim=18,
            input_features=input_features,
            output_features={"action": _feature("ACTION", [18])},
            device="cpu",
        )
