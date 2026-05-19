from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


QIANGNAO_TACTILE_MODE = "qiangnao_tactile"
QIANGNAO_NO_TACTILE_MODE = "qiangnao_no_tactile"
GRIPPER_NO_TACTILE_MODE = "gripper_no_tactile"

DECO_28D_LAYOUT = "deco_28d"
DECO_18D_LAYOUT = "deco_18d"
HEAD_STATE_LIVE_JOINT_Q = "live_joint_q"
HEAD_STATE_FIXED_CONFIG = "fixed_config"


@dataclass(frozen=True)
class DecodedDECOAction:
    """DECO action 反解后的机器人控制量。

    `arm_joints` 始终是双臂 14 维目标关节角；灵巧手与二爪夹字段互斥使用。
    """

    arm_joints: np.ndarray
    head_action: np.ndarray
    left_hand: np.ndarray | None = None
    right_hand: np.ndarray | None = None
    left_gripper: float | None = None
    right_gripper: float | None = None


def is_deco_layout(state_layout: str) -> bool:
    return state_layout in {DECO_28D_LAYOUT, DECO_18D_LAYOUT}


def _to_1d_float_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} is empty.")
    return array


def _require_dim(value: Any, name: str, expected_dim: int) -> np.ndarray:
    array = _to_1d_float_array(value, name)
    if array.size != expected_dim:
        raise ValueError(f"{name} must be {expected_dim}D for DECO deployment, got {array.size}D.")
    return array


def resolve_deco_head_state(
    *,
    head_state_source: str,
    live_head_q: Any | None,
    head_init: Any | None,
) -> np.ndarray:
    """根据部署配置选择 DECO state 中的头部 2 维。

    live_joint_q: 使用在线 /sensors_data_raw.joint_data.joint_q[26:28]。
    fixed_config: 使用部署配置 env.head_init。
    """

    if head_state_source == HEAD_STATE_LIVE_JOINT_Q:
        if live_head_q is None:
            raise ValueError("head_state_source='live_joint_q' requires online head_q observation.")
        return _require_dim(live_head_q, "head_q", 2)
    if head_state_source == HEAD_STATE_FIXED_CONFIG:
        if head_init is None:
            raise ValueError("head_state_source='fixed_config' requires env.head_init.")
        return _require_dim(head_init, "head_init", 2)
    raise ValueError("head_state_source must be 'live_joint_q' or 'fixed_config'.")


def build_deco_28d_state(
    *,
    arm_joints: Any,
    dexhand_state: Any,
    live_head_q: Any | None,
    head_init: Any | None,
    head_state_source: str,
) -> np.ndarray:
    """构造 DECO 灵巧手 28D state。

    顺序必须与数据转换脚本一致：左臂 7 + 左手 6 + 右臂 7 + 右手 6 + 头部 2。
    """

    arms = _require_dim(arm_joints, "joint_q", 14)
    hands = _require_dim(dexhand_state, "dexhand_state", 12)
    head = resolve_deco_head_state(
        head_state_source=head_state_source,
        live_head_q=live_head_q,
        head_init=head_init,
    )
    return np.concatenate((arms[:7], hands[:6], arms[7:14], hands[6:12], head), axis=0).astype(np.float32)


def build_deco_18d_state(
    *,
    arm_joints: Any,
    gripper_state: Any,
    live_head_q: Any | None,
    head_init: Any | None,
    head_state_source: str,
) -> np.ndarray:
    """构造 DECO 二爪夹 18D state。

    顺序必须与数据转换脚本一致：左臂 7 + 左夹爪 1 + 右臂 7 + 右夹爪 1 + 头部 2。
    """

    arms = _require_dim(arm_joints, "joint_q", 14)
    gripper = _to_1d_float_array(gripper_state, "gripper_state")
    if gripper.size == 1:
        # rq2f85 部分消息可能只给出一个对称夹爪值；部署侧与数据转换侧保持复制语义。
        gripper = np.repeat(gripper, 2)
    if gripper.size != 2:
        raise ValueError(f"gripper_state must be 1D or 2D for deco_18d, got {gripper.size}D.")
    head = resolve_deco_head_state(
        head_state_source=head_state_source,
        live_head_q=live_head_q,
        head_init=head_init,
    )
    return np.concatenate((arms[:7], gripper[:1], arms[7:14], gripper[1:2], head), axis=0).astype(np.float32)


def decode_deco_28d_action(action: Any) -> DecodedDECOAction:
    """反解 DECO 灵巧手 28D action。

    头部 action 仅保留用于记录和检查，第一版不下发头部控制。
    """

    act = _require_dim(action, "deco_28d_action", 28)
    left_arm = act[:7]
    left_hand = act[7:13]
    right_arm = act[13:20]
    right_hand = act[20:26]
    head_action = act[26:28]
    return DecodedDECOAction(
        arm_joints=np.concatenate((left_arm, right_arm), axis=0).astype(np.float32),
        left_hand=left_hand.astype(np.float32),
        right_hand=right_hand.astype(np.float32),
        head_action=head_action.astype(np.float32),
    )


def decode_deco_18d_action(action: Any) -> DecodedDECOAction:
    """反解 DECO 二爪夹 18D action。

    头部 action 仅保留用于记录和检查，第一版不下发头部控制。
    """

    act = _require_dim(action, "deco_18d_action", 18)
    left_arm = act[:7]
    left_gripper = float(act[7])
    right_arm = act[8:15]
    right_gripper = float(act[15])
    head_action = act[16:18]
    return DecodedDECOAction(
        arm_joints=np.concatenate((left_arm, right_arm), axis=0).astype(np.float32),
        left_gripper=left_gripper,
        right_gripper=right_gripper,
        head_action=head_action.astype(np.float32),
    )


def validate_deco_policy_compatibility(policy_config: Any, deploy_config: Any, env_config: Any) -> None:
    """静态校验部署配置与 checkpoint config 的结构语义是否一致。

    部署配置只负责选择权重和在线 schema；模型结构字段必须来自 checkpoint 自身。
    """

    if policy_config is None:
        raise ValueError("DECO policy has no config; cannot validate deployment compatibility.")

    inference_mode = deploy_config.inference_mode
    if inference_mode == QIANGNAO_TACTILE_MODE:
        expected = {
            "state_layout": DECO_28D_LAYOUT,
            "eef_type": "qiangnao",
            "end_effector_profile": "qiangnao_tactile",
            "action_dim": 28,
            "use_tactile": True,
            "use_tactile_lora": True,
        }
    elif inference_mode == QIANGNAO_NO_TACTILE_MODE:
        expected = {
            "state_layout": DECO_28D_LAYOUT,
            "eef_type": "qiangnao",
            "end_effector_profile": "qiangnao_tactile",
            "action_dim": 28,
            "use_tactile": False,
            "use_tactile_lora": False,
        }
    elif inference_mode == GRIPPER_NO_TACTILE_MODE:
        expected = {
            "state_layout": DECO_18D_LAYOUT,
            "eef_type": {"leju_claw", "rq2f85"},
            "end_effector_profile": "gripper_no_tactile",
            "action_dim": 18,
            "use_tactile": False,
            "use_tactile_lora": False,
        }
    else:
        raise ValueError(f"Unsupported DECO inference_mode: {inference_mode}")

    _assert_equal("env.state_layout", env_config.state_layout, expected["state_layout"])
    expected_eef = expected["eef_type"]
    if isinstance(expected_eef, set):
        if env_config.eef_type not in expected_eef:
            raise ValueError(f"env.eef_type must be one of {sorted(expected_eef)}, got {env_config.eef_type}.")
    else:
        _assert_equal("env.eef_type", env_config.eef_type, expected_eef)

    _assert_equal("policy.end_effector_profile", policy_config.end_effector_profile, expected["end_effector_profile"])
    _assert_equal("policy.action_dim", policy_config.action_dim, expected["action_dim"])
    _assert_equal("policy.use_tactile", policy_config.use_tactile, expected["use_tactile"])
    _assert_equal("policy.use_tactile_lora", policy_config.use_tactile_lora, expected["use_tactile_lora"])


def _assert_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{name} must be {expected!r} for DECO deployment, got {actual!r}.")
