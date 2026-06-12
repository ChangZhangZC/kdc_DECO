from __future__ import annotations

from types import SimpleNamespace

import pytest

from kuavo_deploy.config import (
    ConfigActionDispatch,
    ConfigDeco,
    ConfigEnv,
    ConfigRecedingHorizon,
    ConfigStrideAction,
    ConfigTemporalEnsemble,
)


def test_deploy_action_dispatch_defaults_to_receding_horizon() -> None:
    config = ConfigDeco()

    assert config.action_dispatch.mode == "receding_horizon"
    assert config.action_dispatch.receding_horizon.n_action_steps is None


def test_deploy_inf_step_accepts_null_or_positive_integer() -> None:
    ConfigDeco(inf_step=None).validate_action_dispatch_structure()
    ConfigDeco(inf_step=8).validate_action_dispatch_structure()

    for invalid_value in (True, 0, -1, 1.5, "8"):
        with pytest.raises(ValueError, match="deco.inf_step"):
            ConfigDeco(inf_step=invalid_value).validate_action_dispatch_structure()


def test_deploy_inf_step_override_updates_policy_config_and_model() -> None:
    policy = SimpleNamespace(
        config=SimpleNamespace(dataset_hz=30, chunk_size=32, inf_step=5),
        model=SimpleNamespace(inference_step=5),
    )
    configured_dispatch: dict[str, object] = {}

    def configure_action_dispatch(mode: str, **kwargs: object) -> None:
        configured_dispatch["mode"] = mode
        configured_dispatch.update(kwargs)

    policy.configure_action_dispatch = configure_action_dispatch
    deploy_config = ConfigDeco(inf_step=8)
    env = ConfigEnv(state_layout="deco_18d", eef_type="rq2f85", ros_rate=30)

    from kuavo_deploy.utils.deco_obs_action import configure_deco_action_dispatch

    configure_deco_action_dispatch(policy, deploy_config, env)

    assert policy.config.inf_step == 8
    assert policy.model.inference_step == 8
    assert configured_dispatch["mode"] == "receding_horizon"
    assert configured_dispatch["n_action_steps"] is None


def test_stride_mode_requires_matching_environment_rate() -> None:
    config = ConfigDeco(
        action_dispatch=ConfigActionDispatch(
            mode="stride_action",
            stride_action=ConfigStrideAction(target_hz=10),
        )
    )
    env = ConfigEnv(state_layout="deco_18d", eef_type="rq2f85", ros_rate=30)

    with pytest.raises(ValueError, match="env.ros_rate"):
        config.validate_action_dispatch_timing(dataset_hz=30, chunk_size=32, env=env)


def test_receding_horizon_rejects_steps_larger_than_chunk() -> None:
    config = ConfigDeco(
        action_dispatch=ConfigActionDispatch(
            mode="receding_horizon",
            receding_horizon=ConfigRecedingHorizon(n_action_steps=33),
        )
    )
    env = ConfigEnv(state_layout="deco_18d", eef_type="rq2f85", ros_rate=30)

    with pytest.raises(ValueError, match="chunk_size"):
        config.validate_action_dispatch_timing(dataset_hz=30, chunk_size=32, env=env)


def test_inactive_strategy_queue_parameter_is_rejected() -> None:
    config = ConfigDeco(
        action_dispatch=ConfigActionDispatch(
            mode="temporal_ensemble",
            receding_horizon=ConfigRecedingHorizon(n_action_steps=4),
            temporal_ensemble=ConfigTemporalEnsemble(coefficient=0.1),
        )
    )

    with pytest.raises(ValueError, match="receding_horizon.n_action_steps"):
        config.validate_action_dispatch_structure()
