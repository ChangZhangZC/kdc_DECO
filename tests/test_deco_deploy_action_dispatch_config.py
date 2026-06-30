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
    """只覆盖部署运行时配置需要的最小 DECO policy 接口。"""

    def __init__(self) -> None:
        self.config = SimpleNamespace(inf_step=5, dataset_hz=30, chunk_size=32)
        self.model = SimpleNamespace(inference_step=5)
        self.dispatch_mode = "receding_horizon"
        self.dispatch_kwargs: dict[str, object] = {}

    def configure_action_dispatch(self, mode: str, **kwargs: object) -> None:
        self.dispatch_mode = mode
        self.dispatch_kwargs = kwargs

    def get_dispatch_info(self) -> dict[str, object]:
        return {"mode": self.dispatch_mode, "chunk_size": self.config.chunk_size}


def _write_deco_deploy_yaml(tmp_path, *, inf_step_line: str | None) -> str:
    """生成满足 ConfigDeco schema 的最小部署 YAML，用于覆盖 loader 兼容路径。"""

    inf_step_yaml = "" if inf_step_line is None else f"  inf_step: {inf_step_line}\n"
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
        f"{inf_step_yaml}"
        "  action_dispatch:\n"
        "    mode: receding_horizon\n"
        "    receding_horizon:\n"
        "      n_action_steps: null\n",
        encoding="utf-8",
    )
    return str(config_path)


def test_deploy_action_dispatch_defaults_to_receding_horizon() -> None:
    config = ConfigDeco()

    assert config.action_dispatch.mode == "receding_horizon"
    assert config.action_dispatch.receding_horizon.n_action_steps is None


def test_deploy_inf_step_defaults_to_checkpoint_value() -> None:
    config = ConfigDeco()

    assert config.inf_step is None


def test_deploy_yaml_accepts_null_inf_step(tmp_path) -> None:
    config_path = _write_deco_deploy_yaml(tmp_path, inf_step_line="null")

    config = load_kuavo_config(config_path)

    assert config.deco.inf_step is None


def test_legacy_deploy_yaml_without_inf_step_uses_checkpoint_value(tmp_path) -> None:
    config_path = _write_deco_deploy_yaml(tmp_path, inf_step_line=None)

    config = load_kuavo_config(config_path)

    assert config.deco.inf_step is None


@pytest.mark.parametrize("invalid_inf_step", [True, 0, -1, 1.5, "10"])
def test_deploy_inf_step_rejects_invalid_values(invalid_inf_step) -> None:
    config = ConfigDeco(inf_step=invalid_inf_step)

    with pytest.raises(ValueError, match="deco.inf_step"):
        config.validate_action_dispatch_structure()


def test_runtime_null_inf_step_preserves_checkpoint_and_model_values() -> None:
    policy = FakeDECOPolicy()
    deploy_config = ConfigDeco(inf_step=None)
    env = ConfigEnv(state_layout="deco_18d", eef_type="rq2f85", ros_rate=30)

    runtime_info = configure_deco_runtime(policy, deploy_config, env)

    assert policy.config.inf_step == 5
    assert policy.model.inference_step == 5
    assert runtime_info["inf_step"] == 5
    assert runtime_info["inf_step_source"] == "checkpoint"
    assert runtime_info["action_dispatch"]["mode"] == "receding_horizon"


def test_runtime_inf_step_override_updates_config_and_model_only() -> None:
    policy = FakeDECOPolicy()
    deploy_config = ConfigDeco(inf_step=10)
    env = ConfigEnv(state_layout="deco_18d", eef_type="rq2f85", ros_rate=30)
    original_chunk_size = policy.config.chunk_size

    runtime_info = configure_deco_runtime(policy, deploy_config, env)

    assert policy.config.inf_step == 10
    assert policy.model.inference_step == 10
    assert policy.config.chunk_size == original_chunk_size
    assert policy.dispatch_mode == "receding_horizon"
    assert policy.dispatch_kwargs["n_action_steps"] is None
    assert runtime_info["inf_step"] == 10
    assert runtime_info["inf_step_source"] == "deploy_override"


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
