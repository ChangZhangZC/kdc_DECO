#!/usr/bin/env python3
import numpy as np

from kuavo_deploy.kuavo_env.KuavoSimEnv import KuavoSimEnv
from kuavo_deploy.utils.deco_obs_action import (
    DECO_18D_LAYOUT,
    DECO_28D_LAYOUT,
    decode_deco_18d_action,
    decode_deco_28d_action,
)
from kuavo_deploy.utils.logging_utils import setup_logger


log_robot = setup_logger("robot")


class ArmSafetyError(RuntimeError):
    """Raised before DECO action clipping/dispatch when arm safety validation fails."""


class KuavoDECOEnv(KuavoSimEnv):
    """DECO-only Kuavo environment with a raw-action arm safety guard.

    Strict safety validation runs before the inherited action-space clipping path.
    The validator never modifies the action; after it passes, KuavoBaseRosEnv.step()
    keeps the historical clipping and dispatch behavior unchanged.
    """

    def __init__(self, config):
        super().__init__(config)
        safety = self.limits.get("deco_arm_safety", {})
        self.deco_arm_safety_mode = safety.get("mode", "normal")
        self.deco_arm_max_step_delta = float(safety.get("arm_max_step_delta", np.pi / 2))
        self._validate_safety_config()

    def _validate_safety_config(self):
        if self.state_layout not in {DECO_18D_LAYOUT, DECO_28D_LAYOUT}:
            raise ValueError("Kuavo-DECO requires state_layout='deco_18d' or 'deco_28d'.")
        if self.deco_arm_safety_mode not in {"normal", "strict"}:
            raise ValueError("deco_arm_safety.mode must be 'normal' or 'strict'.")
        if not np.isfinite(self.deco_arm_max_step_delta) or self.deco_arm_max_step_delta <= 0:
            raise ValueError("deco_arm_safety.arm_max_step_delta must be a finite positive value in radians.")

    def step(self, action):
        # Safety check and clipping are intentionally separate stages:
        # raw DECO action -> strict validation -> inherited check_action()/clip -> dispatch.
        if self.deco_arm_safety_mode == "strict":
            self._validate_raw_deco_arm_action(action)
        return super().step(action)

    def _validate_raw_deco_arm_action(self, action):
        if self.state_layout == DECO_28D_LAYOUT:
            decoded = decode_deco_28d_action(action)
        else:
            decoded = decode_deco_18d_action(action)

        target = np.asarray(decoded.arm_joints, dtype=np.float64).reshape(-1)
        actual = self._get_actual_arm_joint_q()
        joint_min = np.asarray(self.limits["joint_q"]["min"], dtype=np.float64).reshape(-1)
        joint_max = np.asarray(self.limits["joint_q"]["max"], dtype=np.float64).reshape(-1)

        if target.shape != (14,):
            self._raise_safety_error(f"target arm command must be 14D, got shape={target.shape}")
        if joint_min.shape != (14,) or joint_max.shape != (14,):
            self._raise_safety_error(
                f"joint_q limits must both be 14D, got min={joint_min.shape}, max={joint_max.shape}"
            )
        if not np.isfinite(target).all():
            self._raise_safety_error(f"target arm command contains NaN/Inf: {target}")
        if not np.isfinite(joint_min).all() or not np.isfinite(joint_max).all():
            self._raise_safety_error("joint_q limits contain NaN/Inf")

        names = [f"left_j{i}" for i in range(1, 8)] + [f"right_j{i}" for i in range(1, 8)]
        limit_indices = np.flatnonzero((target < joint_min) | (target > joint_max))
        if limit_indices.size:
            details = "; ".join(
                f"{names[i]} target={target[i]:.6f} allowed=[{joint_min[i]:.6f},{joint_max[i]:.6f}]"
                for i in limit_indices
            )
            self._raise_safety_error(f"[absolute_limit] {details}")

        delta = np.abs(target - actual)
        delta_indices = np.flatnonzero(delta > self.deco_arm_max_step_delta)
        if delta_indices.size:
            details = "; ".join(
                f"{names[i]} actual={actual[i]:.6f} target={target[i]:.6f} "
                f"delta={delta[i]:.6f} max_delta={self.deco_arm_max_step_delta:.6f}"
                for i in delta_indices
            )
            self._raise_safety_error(f"[step_delta] {details}")

    def _get_actual_arm_joint_q(self):
        try:
            actual = np.asarray(
                self.robot_state.arm_joint_state().position,
                dtype=np.float64,
            ).reshape(-1)
        except Exception as exc:
            message = f"ARM SAFETY VIOLATION: failed to read actual arm joint_q: {exc}. ARM COMMAND WAS NOT SENT."
            log_robot.critical(message)
            raise ArmSafetyError(message) from exc

        if actual.shape != (14,):
            self._raise_safety_error(f"actual arm joint_q must be 14D, got shape={actual.shape}")
        if not np.isfinite(actual).all():
            self._raise_safety_error(f"actual arm joint_q contains NaN/Inf: {actual}")
        return actual

    @staticmethod
    def _raise_safety_error(details):
        message = f"ARM SAFETY VIOLATION {details}. ARM COMMAND WAS NOT SENT."
        log_robot.critical(message)
        raise ArmSafetyError(message)
