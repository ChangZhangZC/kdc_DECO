from __future__ import annotations

# 训练入口会先安装 Kuavo 对 FeatureType.DEPTH/TACTILE 的兼容扩展；
# 本测试保持相同导入顺序，覆盖用户实际 Hydra 构造路径。
import lerobot_patches.custom_patches  # noqa: F401
from lerobot.configs.types import FeatureType, PolicyFeature

from kuavo_train.wrapper.policy.deco.DECOConfigWrapper import (
    DEFAULT_RGB_KEYS,
    CustomDECOConfigWrapper,
)


def _feature(feature_type: FeatureType | str, shape: list[int]) -> dict[str, object]:
    """模拟 Hydra 将嵌套 PolicyFeature dataclass 展平后的字典。"""

    return {"type": feature_type, "shape": shape}


def test_hydra_feature_dicts_are_filtered_without_early_type_access() -> None:
    """构造期间不访问 dict.type，训练入口恢复类型后可正常校验三路 RGB。"""

    input_features = {
        "observation.state": _feature(FeatureType.STATE, [18]),
        DEFAULT_RGB_KEYS[0]: _feature(FeatureType.VISUAL, [3, 480, 640]),
        "observation.depth_h": _feature(FeatureType.DEPTH, [3, 480, 640]),
        DEFAULT_RGB_KEYS[1]: _feature(FeatureType.VISUAL, [3, 480, 640]),
        "observation.depth_l": _feature(FeatureType.DEPTH, [3, 480, 640]),
        DEFAULT_RGB_KEYS[2]: _feature(FeatureType.VISUAL, [3, 480, 640]),
        "observation.depth_r": _feature(FeatureType.DEPTH, [3, 480, 640]),
    }
    output_features = {
        "action": _feature(FeatureType.ACTION, [18]),
    }

    config = CustomDECOConfigWrapper(
        end_effector_profile="gripper_no_tactile",
        action_dim=18,
        input_features=input_features,
        output_features=output_features,
        device="cpu",
    )

    # __post_init__ 只按 key 完成白名单筛选，不重复实现训练入口已有的类型恢复。
    assert tuple(config.input_features) == (
        "observation.state",
        *DEFAULT_RGB_KEYS,
    )
    assert all(isinstance(feature, dict) for feature in config.input_features.values())
    assert not any("depth" in key for key in config.input_features)

    # 对齐 train_policy.py 与 train_policy_with_accelerate.py 已验证分支的现有逻辑：
    # Hydra instantiate 返回后，再把 feature dict 恢复为 PolicyFeature。
    config.input_features = {
        key: PolicyFeature(**feature)
        for key, feature in config.input_features.items()
    }
    config.output_features = {
        key: PolicyFeature(**feature)
        for key, feature in config.output_features.items()
    }
    assert config.input_features[DEFAULT_RGB_KEYS[0]].shape == [3, 480, 640]
    assert config.input_features[DEFAULT_RGB_KEYS[0]].type is FeatureType.VISUAL
    config.validate_features()
