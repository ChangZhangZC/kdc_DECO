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
    load_kuavo_config,
)
from kuavo_deploy.utils.deco_obs_action import configure_deco_runtime


class FakeDECOPolicy:
    """只实现动作分发配置所需的最小 policy 接口。"""

    def __init__(self) -> None:
        self.config = SimpleNamespace(dataset_hz=30, chunk_size=32, n_action_steps=16, inf_step=5)
        self.model = SimpleNamespace(inference_step=5)
        self.dispatch_mode = "receding_horizon"
        self.dispatch_kwargs: dict[str, object] = {}

    def configure_action_dispatch(self, mode: str, **kwargs: object) -> None:
        self.dispatch_mode = mode
        self.dispatch_kwargs = kwargs

    def get_dispatch_info(self) -> dict[str, object]:
        return {"mode": self.dispatch_mode, "queue_remaining": 0}


def test_deploy_defaults_to_16_step_receding_horizon() -> None:
    config = ConfigDeco()

    assert config.action_dispatch.mode == "receding_horizon"
    assert config.action_dispatch.receding_horizon.n_action_steps == 16


def test_deploy_yaml_loads_nested_action_dispatch_and_null_inf_step(tmp_path) -> None:
    """覆盖真实部署 YAML 的嵌套 dataclass 反序列化边界。"""

    config_path = tmp_path / "deco_deploy.yaml"
    config_path.write_text(
        "env:\n"
        "  eef_type: rq2f85\n"
        "  state_layout: deco_18d\n"
        "  ros_rate: 30\n"
        "inference:\n"
        "  policy_type: deco\n"
        "deco:\n"
        "  inference_mode: gripper_no_tactile\n"
        "  inf_step: null\n"
        "  action_dispatch:\n"
        "    mode: receding_horizon\n"
        "    receding_horizon:\n"
        "      n_action_steps: 16\n",
        encoding="utf-8",
    )

    config = load_kuavo_config(str(config_path))

    assert config.deco.inf_step is None
    assert config.deco.action_dispatch.mode == "receding_horizon"
    assert config.deco.action_dispatch.receding_horizon.n_action_steps == 16


def test_receding_horizon_requires_30hz_environment() -> None:
    config = ConfigDeco()
    env = ConfigEnv(state_layout="deco_18d", eef_type="rq2f85", ros_rate=10)

    with pytest.raises(ValueError, match="env.ros_rate"):
        config.validate_action_dispatch_timing(dataset_hz=30, chunk_size=32, env=env)


def test_runtime_configures_16_step_receding_horizon() -> None:
    policy = FakeDECOPolicy()
    config = ConfigDeco()
    env = ConfigEnv(state_layout="deco_18d", eef_type="rq2f85", ros_rate=30)

    dispatch_info = configure_deco_runtime(policy, config, env)

    assert policy.dispatch_mode == "receding_horizon"
    assert policy.dispatch_kwargs["n_action_steps"] == 16
    assert dispatch_info["mode"] == "receding_horizon"
    assert dispatch_info["inf_step"] == 5
    assert dispatch_info["inf_step_source"] == "checkpoint"


def test_runtime_inf_step_override_updates_config_and_model() -> None:
    policy = FakeDECOPolicy()
    config = ConfigDeco(inf_step=10)
    env = ConfigEnv(state_layout="deco_18d", eef_type="rq2f85", ros_rate=30)

    dispatch_info = configure_deco_runtime(policy, config, env)

    assert policy.config.inf_step == 10
    assert policy.model.inference_step == 10
    assert dispatch_info["inf_step"] == 10
    assert dispatch_info["inf_step_source"] == "deploy_override"


@pytest.mark.parametrize("invalid_inf_step", [True, 0, -1, 1.5, "10"])
def test_deploy_inf_step_rejects_invalid_values(invalid_inf_step: object) -> None:
    config = ConfigDeco(inf_step=invalid_inf_step)

    with pytest.raises(ValueError, match="deco.inf_step"):
        config.validate_action_dispatch_structure()


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


def test_temporal_ensemble_requires_inactive_receding_steps_to_be_null() -> None:
    config = ConfigDeco(
        action_dispatch=ConfigActionDispatch(
            mode="temporal_ensemble",
            receding_horizon=ConfigRecedingHorizon(n_action_steps=None),
            temporal_ensemble=ConfigTemporalEnsemble(coefficient=0.1),
        )
    )
    env = ConfigEnv(state_layout="deco_18d", eef_type="rq2f85", ros_rate=30)

    config.validate_action_dispatch_timing(dataset_hz=30, chunk_size=32, env=env)


def test_stride_mode_requires_matching_environment_rate() -> None:
    config = ConfigDeco(
        action_dispatch=ConfigActionDispatch(
            mode="stride_action",
            receding_horizon=ConfigRecedingHorizon(n_action_steps=None),
            stride_action=ConfigStrideAction(target_hz=10),
        )
    )
    env = ConfigEnv(state_layout="deco_18d", eef_type="rq2f85", ros_rate=30)

    with pytest.raises(ValueError, match="env.ros_rate"):
        config.validate_action_dispatch_timing(dataset_hz=30, chunk_size=32, env=env)
