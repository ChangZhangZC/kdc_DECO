"""
Single merged configuration loader for Kuavo.
Loads ./configs/deploy/kuavo_env.yaml by default (keeps original YAML structure).
Provides:
  - ConfigEnv (environment) with slice properties:
      .slice_robot, .qiangnao_slice, .claw_slice
  - ConfigInference (inference)
  - KuavoConfig (master)
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Any, Dict
import math
import os
import yaml


@dataclass
class Range:
    min: List[float]
    max: List[float]

@dataclass
class LimitsConfig:
    joint_q: Range = field(default_factory=lambda: Range([-3.14]*14, [3.14]*14))
    gripper: Range = field(default_factory=lambda: Range([0, 0], [1, 1]))
    head_q: Range = field(default_factory=lambda: Range([-1.0, -1.0], [1.0, 1.0]))
    eef: Range = field(default_factory=lambda: Range(
        [-1, -1, -1, -3.14, -3.14, -3.14,
         -1, -1, -1, -3.14, -3.14, -3.14],
        [1, 1, 1, 3.14, 3.14, 3.14,
         1, 1, 1, 3.14, 3.14, 3.14]
    ))
    eef_relative: Range = field(default_factory=lambda: Range(
        [-0.005, -0.0075, -0.004, -0.03, -0.03, -0.05,
         -0.005, -0.0075, -0.004, -0.03, -0.03, -0.05],
        [0.005, 0.0075, 0.004, 0.03, 0.03, 0.05,
         0.005, 0.0075, 0.004, 0.03, 0.03, 0.05]
    ))
    base: Range = field(default_factory=lambda: Range([-2.0, -2.0, -3.14, 0],
                                                      [2.0, 2.0, 3.14, 1]))

# -----------------------
# Environment Dataclass
# -----------------------
@dataclass
class ConfigEnv:
    env_name: str = "Kuavo-Sim"
    real: bool = False
    only_arm: bool = True
    eef_type: str = "rq2f85"
    state_layout: str = "standard"
    control_mode: str = "joint"
    which_arm: str = "both"
    head_init: Optional[List[float]] = field(default_factory=lambda: [0.0, 0.0])
    use_delta: bool = False
    delta_type: str = "Tsub"  # "Tsub","Tinv","RPY"
    ros_rate: int = 10
    image_size: List[int] = field(default_factory=lambda: [640, 480])
    depth_range: List[int] = field(default_factory=lambda: [0, 1500])
    obs_key_map: Dict[str, List[Any]] = field(default_factory=dict)
    arm_state_keys: List[str]=field(default_factory=list)
    ratio: float = 0.5
    frame_alignment: bool = True
    qiangnao_dof_needed: int = 1

    fk_joint_angles_for_reset: Optional[List[float]] = None
    rotation_threshold: Optional[float] = None
    
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    is_binary: bool = False

    # -------- Validation ----------
    def validate(self):
        if self.eef_type not in ["rq2f85", "leju_claw", "qiangnao"]:
            raise ValueError(f"Invalid eef_type: {self.eef_type}. Valid: rq2f85, leju_claw, qiangnao")
        if self.which_arm not in ["left", "right", "both"]:
            raise ValueError(f"Invalid which_arm: {self.which_arm}. Valid: left, right, both")
        if self.state_layout not in ["standard", "deco_28d", "deco_18d"]:
            raise ValueError("state_layout must be 'standard', 'deco_28d', or 'deco_18d'")
        if self.state_layout.startswith("deco_") and self.which_arm != "both":
            raise ValueError("DECO deployment layouts currently require which_arm='both'.")
        if self.state_layout.startswith("deco_") and not self.only_arm:
            raise ValueError("DECO deployment layouts currently require only_arm=True; base control is not part of DECO action_dim.")
        if self.state_layout == "deco_28d" and self.eef_type != "qiangnao":
            raise ValueError("state_layout='deco_28d' requires eef_type='qiangnao'.")
        if self.state_layout == "deco_18d" and self.eef_type not in ["leju_claw", "rq2f85"]:
            raise ValueError("state_layout='deco_18d' requires eef_type='leju_claw' or 'rq2f85'.")
        if not isinstance(self.image_size, list) or len(self.image_size) != 2:
            raise ValueError("image_size must be a list [width, height], matching cv2.resize.")
        # ensure lists lengths for arm bounds
        if not (len(self.limits["joint_q"]["max"]) == len(self.limits["joint_q"]["min"]) == 14):
            raise ValueError("Robot arm_min/arm_max must be lists of length 14")
        if self.state_layout == "deco_28d" and not (
            len(self.limits["gripper"]["max"]) >= 12 and len(self.limits["gripper"]["min"]) >= 12
        ):
            raise ValueError("state_layout='deco_28d' requires gripper limits with at least 12 values.")
        if self.state_layout == "deco_18d" and not (
            len(self.limits["gripper"]["max"]) >= 2 and len(self.limits["gripper"]["min"]) >= 2
        ):
            raise ValueError("state_layout='deco_18d' requires gripper limits with at least 2 values.")
        if self.state_layout.startswith("deco_") and not (
            len(self.limits.get("head_q", {}).get("max", [])) >= 2 and len(self.limits.get("head_q", {}).get("min", [])) >= 2
        ):
            raise ValueError("DECO deployment layouts require head_q limits with at least 2 values.")
        if self.qiangnao_dof_needed not in [1, 6]:
            raise ValueError("qiangnao_dof_needed must be 1 for ACT/DP style gripper state or 6 for DECO dexhand state.")

    # -------- Derived properties ----------
    @property
    def joint_q_slice(self):
        return {
            "left": [[12, 19]],
            "right": [[19, 26]],
            "both": [[12, 19], [19, 26]]
        }[self.which_arm]

    @property
    def gripper_slice(self):
        if self.eef_type == "rq2f85" or self.eef_type == "leju_claw":
            return {
                "left": [[0, 1]],
                "right": [[1, 2]],
                "both": [[0, 1], [1, 2]]
            }[self.which_arm]
        elif self.eef_type == "qiangnao" and self.qiangnao_dof_needed == 1:
            return {
                "left": [[0, 1]],
                "right": [[6, 7]],
                "both": [[0, 1], [6, 7]]
            }[self.which_arm]
        elif self.eef_type == "qiangnao" and self.qiangnao_dof_needed == 6:
            # DECO 28D 需要完整左右手各 6 维，而不是 ACT/DP 默认的单维开合量。
            return {
                "left": [[0, 6]],
                "right": [[6, 12]],
                "both": [[0, 6], [6, 12]]
            }[self.which_arm]
        else:
            raise ValueError("Unsupported eef_type or dof config")

    # ---------------- obs_key_map build ----------------
    def build_obs_key_map(self, deco: Optional["ConfigDeco"] = None) -> Dict[str, Any]:
        obs_map = {}
        for key, info in self.obs_key_map.items():
            if key == "head_q" and deco is not None and deco.head_control.mode == "fixed":
                # fixed 模式的头部 observation 与最终 action 都来自 fixed_value；
                # YAML topic 模板保持不变，但运行时不创建无效的头部订阅和对齐依赖。
                continue
            if key == "tactile" and (deco is None or deco.inference_mode != "qiangnao_tactile"):
                # 只有带触觉的 DECO 灵巧手推理才订阅 tactile，避免无触觉 checkpoint 收到多余 feature。
                continue
            if key in ["rq2f85", "qiangnao", "leju_claw"] and key != self.eef_type:
                # 同一个部署配置可以保留多个末端 topic 模板，但实际只启用当前 eef_type。
                continue
            base = {
                "topic": info[0],
                "msg_type": info[1],
                "frequency": info[2],
                "handle": {"params": {}}
            }
            # 统一规则化参数处理
            if len(info) == 4 and isinstance(info[3], list):
                base["handle"]["params"]["resize_wh"] = info[3]
            if len(info) == 5 and isinstance(info[3], list) and isinstance(info[4], list):
                base["handle"]["params"]["resize_wh"] = info[3]
                base["handle"]["params"]["depth_range"] = info[4]
                if "depth" in key:
                    base["handle"]["params"]["depth_encoding"] = "compressedDepth_png"
            if len(info) >= 6 and isinstance(info[3], list) and isinstance(info[4], list):
                base["handle"]["params"]["resize_wh"] = info[3]
                base["handle"]["params"]["depth_range"] = info[4]
                base["handle"]["params"]["depth_encoding"] = info[5]
            if key == "tactile" and len(info) >= 4:
                base["handle"]["params"]["force_scale"] = float(info[3])

            # 特殊键处理
            if key == "joint_q":
                base["handle"]["params"]["slice"] = self.joint_q_slice
            if key == "head_q":
                base["handle"]["params"]["slice"] = [[26, 28]]
            if key in ["rq2f85", "qiangnao", "leju_claw"]:
                base["handle"]["params"]["slice"] = self.gripper_slice
                obs_map["gripper"] = base
                continue
            if key == "eef_pose" and len(info) >= 3 and info[0] == "computed":
                obs_map["eef_pose"] = {
                    "type": "computed",
                    "source": info[1],
                    "frequency": info[2]
                }
                continue
            obs_map[key] = base
        return obs_map



# -----------------------
# Inference Dataclass
# -----------------------
@dataclass
class ConfigInference:
    go_bag_path: str = ""
    policy_type: str = "diffusion"  # 支持 diffusion, act 等
    eval_episodes: int = 1
    seed: int = 42
    start_seed: int = 42
    device: str = "cuda"  # or "cpu"
    task: str = ""
    method: str = ""
    timestamp: str = ""
    epoch: int = 1
    max_episode_steps: int = 1000
    env_name: str = "Kuavo-Sim"
    client_host: str = "localhost"
    client_port: int = 5555
    client_timeout_ms: int = 15000
    client_api_token_env: str = ""

    def validate(self):
        if self.policy_type not in ["diffusion", "act", "deco", "client"]:
            # 若将来支持更多策略，请在此扩展
            # Expansion room for future support for other policies
            raise ValueError(f"Unsupported policy_type '{self.policy_type}'")
        if self.device not in ["cuda", "cpu"]:
            raise ValueError("device must be 'cuda' or 'cpu'")
        if not isinstance(self.client_host, str) or not self.client_host:
            raise ValueError("client_host must be a non-empty string.")
        self.client_port = int(self.client_port)
        self.client_timeout_ms = int(self.client_timeout_ms)
        if self.client_port <= 0:
            raise ValueError("client_port must be a positive integer.")
        if self.client_timeout_ms <= 0:
            raise ValueError("client_timeout_ms must be a positive integer.")
        if self.client_api_token_env is None:
            self.client_api_token_env = ""

    def client_api_token_value(self) -> Optional[str]:
        """从环境变量读取 client token，避免把远端推理 token 明文写入部署 YAML。"""

        token_env = str(self.client_api_token_env or "").strip()
        if not token_env:
            return None
        token = os.environ.get(token_env)
        if not token:
            raise ValueError(
                f"inference.client_api_token_env={token_env!r} is set, "
                "but the environment variable is empty or missing."
            )
        return token


@dataclass
class ConfigRecedingHorizon:
    """连续消费原始 chunk 前 N 步；None 表示完整 chunk。"""

    n_action_steps: Optional[int] = 16


@dataclass
class ConfigTemporalEnsemble:
    """原生在线 Temporal Ensembling 的指数衰减系数。"""

    coefficient: float = 0.1


@dataclass
class ConfigActionDispatch:
    """DECO 两种互斥动作分发策略的部署接口。"""

    mode: str = "receding_horizon"
    receding_horizon: ConfigRecedingHorizon = field(default_factory=ConfigRecedingHorizon)
    temporal_ensemble: ConfigTemporalEnsemble = field(default_factory=ConfigTemporalEnsemble)

    def __post_init__(self) -> None:
        # YAML loader 和 asdict() 会把嵌套 dataclass 变成 dict；在配置边界恢复强类型。
        if isinstance(self.receding_horizon, dict):
            self.receding_horizon = ConfigRecedingHorizon(**self.receding_horizon)
        if isinstance(self.temporal_ensemble, dict):
            self.temporal_ensemble = ConfigTemporalEnsemble(**self.temporal_ensemble)


@dataclass
class ConfigHeadControl:
    """DECO 头部 observation 与最终动作下发的互斥配置。"""

    mode: str = "fixed"
    initial_value: List[float] = field(default_factory=lambda: [0.0, 0.0])
    fixed_value: Optional[List[float]] = field(default_factory=lambda: [0.0, 0.0])

    @staticmethod
    def _validate_head_value(name: str, value: Any, *, required: bool) -> Optional[List[float]]:
        if value is None:
            if required:
                raise ValueError(f"{name} must be a 2D yaw/pitch value in radians.")
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{name} must contain exactly two values: [yaw, pitch] in radians.")
        result = [float(item) for item in value]
        if not all(math.isfinite(item) for item in result):
            raise ValueError(f"{name} must contain finite values.")
        return result

    def validate(self, head_limits: Any) -> None:
        if self.mode not in {"fixed", "policy"}:
            raise ValueError("deco.head_control.mode must be 'fixed' or 'policy'.")

        self.initial_value = self._validate_head_value(
            "deco.head_control.initial_value", self.initial_value, required=True
        )
        self.fixed_value = self._validate_head_value(
            "deco.head_control.fixed_value",
            self.fixed_value,
            required=self.mode == "fixed",
        )
        if self.mode == "policy" and self.fixed_value is not None:
            raise ValueError("deco.head_control.fixed_value must be null when mode='policy'.")

        head_min = list(head_limits.get("min", []))
        head_max = list(head_limits.get("max", []))
        if len(head_min) < 2 or len(head_max) < 2:
            raise ValueError("env.limits.head_q must provide two-dimensional physical limits.")
        for name, value in (
            ("deco.head_control.initial_value", self.initial_value),
            ("deco.head_control.fixed_value", self.fixed_value),
        ):
            if value is None:
                continue
            if any(value[index] < head_min[index] or value[index] > head_max[index] for index in range(2)):
                raise ValueError(
                    f"{name}={value!r} exceeds env.limits.head_q; "
                    "head configuration uses final robot coordinates in radians and is not silently clipped."
                )


@dataclass
class ConfigDeco:
    """DECO 部署专用配置。

    这些字段只描述在线推理如何拼接观测、选择 checkpoint 和解释动作；
    不应覆盖 checkpoint 自己保存的模型结构字段。
    """

    inference_mode: str = "qiangnao_no_tactile"
    # None 保持 checkpoint 保存的 Flow Matching 推理步数；正整数仅覆盖在线 denoising 循环。
    inf_step: Optional[int] = None
    head_control: ConfigHeadControl = field(default_factory=ConfigHeadControl)
    # 仅用于读取旧部署 YAML；加载器会转换为 head_control，新的配置不要再填写。
    head_state_source: Optional[str] = None
    action_dispatch: ConfigActionDispatch = field(default_factory=ConfigActionDispatch)

    def __post_init__(self) -> None:
        if isinstance(self.head_control, dict):
            self.head_control = ConfigHeadControl(**self.head_control)
        if isinstance(self.action_dispatch, dict):
            self.action_dispatch = ConfigActionDispatch(**self.action_dispatch)

    def validate_action_dispatch_structure(self) -> None:
        """校验两种 dispatcher 的互斥配置，不依赖 checkpoint 运行参数。"""

        if self.inf_step is not None and (
            isinstance(self.inf_step, bool)
            or not isinstance(self.inf_step, int)
            or self.inf_step <= 0
        ):
            raise ValueError("deco.inf_step must be a positive integer or null.")

        dispatch = self.action_dispatch
        if not isinstance(dispatch, ConfigActionDispatch):
            raise ValueError("deco.action_dispatch must be a mapping/object.")
        if not isinstance(dispatch.receding_horizon, ConfigRecedingHorizon):
            raise ValueError("deco.action_dispatch.receding_horizon must be a mapping/object.")
        if not isinstance(dispatch.temporal_ensemble, ConfigTemporalEnsemble):
            raise ValueError("deco.action_dispatch.temporal_ensemble must be a mapping/object.")

        supported_modes = {"receding_horizon", "temporal_ensemble"}
        if dispatch.mode not in supported_modes:
            raise ValueError(
                f"deco.action_dispatch.mode must be one of {sorted(supported_modes)}, "
                f"got {dispatch.mode!r}."
            )

        n_action_steps = dispatch.receding_horizon.n_action_steps
        if n_action_steps is not None:
            if isinstance(n_action_steps, bool) or not isinstance(n_action_steps, int) or n_action_steps <= 0:
                raise ValueError("deco.action_dispatch.receding_horizon.n_action_steps must be positive or null.")
            if dispatch.mode != "receding_horizon":
                raise ValueError(
                    "deco.action_dispatch.receding_horizon.n_action_steps must be null "
                    "when another mode is active."
                )

        coefficient = dispatch.temporal_ensemble.coefficient
        if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)) or coefficient <= 0:
            raise ValueError("deco.action_dispatch.temporal_ensemble.coefficient must be positive.")

    def validate_action_dispatch_timing(self, dataset_hz: int, chunk_size: int, env: ConfigEnv) -> None:
        """checkpoint 加载后校验数据频率、控制频率与 chunk 边界。"""

        self.validate_action_dispatch_structure()
        if dataset_hz <= 0 or chunk_size <= 0:
            raise ValueError("DECO checkpoint dataset_hz and chunk_size must be positive.")

        dispatch = self.action_dispatch
        if env.ros_rate != dataset_hz:
            raise ValueError(
                f"{dispatch.mode} requires env.ros_rate == checkpoint dataset_hz "
                f"({dataset_hz}), got {env.ros_rate}."
            )
        n_action_steps = dispatch.receding_horizon.n_action_steps
        if n_action_steps is not None and n_action_steps > chunk_size:
            raise ValueError(
                "deco.action_dispatch.receding_horizon.n_action_steps cannot exceed "
                f"checkpoint chunk_size ({chunk_size})."
            )

    def validate(self, env: ConfigEnv, inference: ConfigInference):
        deco_policy = inference.policy_type == "deco"
        deco_client = inference.policy_type == "client" and env.state_layout.startswith("deco_")
        if not (deco_policy or deco_client):
            return
        # client 模式下真实 policy 在 server 端，但调用侧仍负责构造 DECO observation
        # 并反解 DECO action；因此只要使用 deco_* state_layout，就必须校验 env/deco schema。
        supported_modes = {
            "qiangnao_tactile",
            "qiangnao_no_tactile",
            "leju_claw_no_tactile",
            "rq2f85_no_tactile",
        }
        if self.inference_mode not in supported_modes:
            raise ValueError(
                f"deco.inference_mode must be one of {sorted(supported_modes)}, got {self.inference_mode!r}."
            )
        if not isinstance(self.head_control, ConfigHeadControl):
            raise ValueError("deco.head_control must be a mapping/object.")
        self.head_control.validate(env.limits["head_q"])
        self.validate_action_dispatch_structure()

        if self.inference_mode in ["qiangnao_tactile", "qiangnao_no_tactile"]:
            if env.eef_type != "qiangnao" or env.state_layout != "deco_28d":
                raise ValueError(
                    f"{self.inference_mode} requires env.eef_type='qiangnao' and env.state_layout='deco_28d'."
                )
            if env.qiangnao_dof_needed != 6:
                raise ValueError(f"{self.inference_mode} requires env.qiangnao_dof_needed=6.")
        if self.inference_mode in {"leju_claw_no_tactile", "rq2f85_no_tactile"}:
            expected_eef = "leju_claw" if self.inference_mode == "leju_claw_no_tactile" else "rq2f85"
            if env.eef_type != expected_eef or env.state_layout != "deco_18d":
                raise ValueError(
                    f"{self.inference_mode} requires env.eef_type={expected_eef!r} "
                    "and env.state_layout='deco_18d'."
                )


# -----------------------
# Master config
# -----------------------
@dataclass
class KuavoConfig:
    env: ConfigEnv
    inference: ConfigInference
    deco: ConfigDeco = field(default_factory=ConfigDeco)

    def validate(self):
        self.env.validate()
        self.inference.validate()
        self.deco.validate(self.env, self.inference)


def _merge_inference_section(
    flat_config: Dict[str, Any],
    section_name: str,
    field_mapping: Dict[str, str],
) -> None:
    """把 DECO 新版条件配置区展开为现有内部字段，并保留旧扁平 YAML 兼容。"""

    section = flat_config.pop(section_name, None)
    if section is None:
        return
    if not isinstance(section, dict):
        raise ValueError(f"inference.{section_name} must be a mapping/object.")
    unknown = set(section) - set(field_mapping)
    if unknown:
        raise ValueError(f"Unknown inference.{section_name} fields: {sorted(unknown)}")
    for nested_name, internal_name in field_mapping.items():
        if nested_name not in section:
            continue
        nested_value = section[nested_name]
        if internal_name in flat_config and flat_config[internal_name] != nested_value:
            raise ValueError(
                f"inference.{internal_name} conflicts with inference.{section_name}.{nested_name}."
            )
        flat_config[internal_name] = nested_value


def _normalize_inference_config(inference_config: Dict[str, Any]) -> Dict[str, Any]:
    """展开 checkpoint、真机、client 与评估配置区，不改变下游 dataclass 接口。"""

    normalized = dict(inference_config)
    _merge_inference_section(
        normalized,
        "checkpoint",
        {"task": "task", "method": "method", "timestamp": "timestamp", "epoch": "epoch"},
    )
    _merge_inference_section(normalized, "real_device", {"go_bag_path": "go_bag_path"})
    _merge_inference_section(
        normalized,
        "client",
        {
            "host": "client_host",
            "port": "client_port",
            "timeout_ms": "client_timeout_ms",
            "api_token_env": "client_api_token_env",
        },
    )
    _merge_inference_section(
        normalized,
        "evaluation",
        {
            "eval_episodes": "eval_episodes",
            "seed": "seed",
            "start_seed": "start_seed",
            "max_episode_steps": "max_episode_steps",
        },
    )
    return normalized


def _normalize_deco_config(
    env_config: Dict[str, Any],
    inference_config: Dict[str, Any],
    deco_config: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """兼容旧部署字段，并从唯一 inference_mode 推导 DECO 固定架构。"""

    normalized_env = dict(env_config)
    normalized_deco = dict(deco_config)
    policy_type = str(inference_config.get("policy_type", "diffusion")).lower()
    explicit_deco = bool(normalized_deco) or str(normalized_env.get("state_layout", "")).startswith("deco_")
    if policy_type not in {"deco", "client"} or (policy_type == "client" and not explicit_deco):
        return normalized_env, normalized_deco

    # 旧 head_state_source 只在 head_control 缺失时迁移；新旧字段同时出现时拒绝歧义。
    legacy_head_source = normalized_deco.get("head_state_source")
    if "head_control" not in normalized_deco:
        initial_value = normalized_env.get("head_init", [0.0, 0.0])
        if legacy_head_source in {None, "fixed_config"}:
            normalized_deco["head_control"] = {
                "mode": "fixed",
                "initial_value": initial_value,
                "fixed_value": initial_value,
            }
        elif legacy_head_source == "live_joint_q":
            normalized_deco["head_control"] = {
                "mode": "policy",
                "initial_value": initial_value,
                "fixed_value": None,
            }
        else:
            raise ValueError("deco.head_state_source must be 'live_joint_q' or 'fixed_config'.")
    elif legacy_head_source is not None:
        expected_mode = "policy" if legacy_head_source == "live_joint_q" else "fixed"
        actual_mode = normalized_deco["head_control"].get("mode")
        if actual_mode != expected_mode:
            raise ValueError("Legacy deco.head_state_source conflicts with deco.head_control.mode.")

    inference_mode = normalized_deco.get("inference_mode", "qiangnao_no_tactile")
    if inference_mode == "gripper_no_tactile":
        # 旧配置用 env.eef_type 区分两个 18D 控制器；新配置改为两个无歧义 mode。
        legacy_eef = normalized_env.get("eef_type")
        if legacy_eef == "leju_claw":
            inference_mode = "leju_claw_no_tactile"
        elif legacy_eef == "rq2f85":
            inference_mode = "rq2f85_no_tactile"
        else:
            raise ValueError(
                "Legacy deco.inference_mode='gripper_no_tactile' requires "
                "env.eef_type='leju_claw' or 'rq2f85'."
            )
        normalized_deco["inference_mode"] = inference_mode

    profiles = {
        "qiangnao_tactile": ("qiangnao", "deco_28d", 6),
        "qiangnao_no_tactile": ("qiangnao", "deco_28d", 6),
        "leju_claw_no_tactile": ("leju_claw", "deco_18d", 1),
        "rq2f85_no_tactile": ("rq2f85", "deco_18d", 1),
    }
    if inference_mode not in profiles:
        raise ValueError(f"Unsupported DECO inference_mode: {inference_mode!r}.")

    eef_type, state_layout, qiangnao_dof = profiles[inference_mode]
    derived_fields = {
        "control_mode": "joint",
        "which_arm": "both",
        "only_arm": True,
        "eef_type": eef_type,
        "state_layout": state_layout,
        "qiangnao_dof_needed": qiangnao_dof,
    }
    for field_name, expected_value in derived_fields.items():
        if field_name in normalized_env and normalized_env[field_name] != expected_value:
            raise ValueError(
                f"env.{field_name}={normalized_env[field_name]!r} conflicts with "
                f"deco.inference_mode={inference_mode!r}; expected {expected_value!r}."
            )
        normalized_env[field_name] = expected_value

    return normalized_env, normalized_deco


# -----------------------
# Loader
# -----------------------
def load_kuavo_config(config_path: Optional[str] = None) -> KuavoConfig:
    """
    Load config from YAML.
    Default path: ./configs/deploy/kuavo_env.yaml (same name as your original file)
    """
    if config_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "../configs", "deploy", "kuavo_env.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # The user's original YAML was mostly top-level keys (not nested under env/inference).
    # We'll support both styles:
    #  - top-level flat (as your original): keys like 'real', 'policy_type', ...
    #  - nested style: {env: {...}, inference: {...}}
    if 'env' in cfg and 'inference' in cfg:
        env_cfg: Dict[str, Any] = cfg.get('env', {})
        inf_cfg: Dict[str, Any] = _normalize_inference_config(cfg.get('inference', {}))
        deco_cfg: Dict[str, Any] = cfg.get('deco', {})
    else:
        # 自动根据 dataclass 字段划分 env / inference
        env_fields = set(ConfigEnv.__dataclass_fields__.keys())
        inf_fields = set(ConfigInference.__dataclass_fields__.keys())
        deco_fields = set(ConfigDeco.__dataclass_fields__.keys())

        env_cfg = {}
        inf_cfg = {}
        deco_cfg = {}

        for k, v in cfg.items():
            if k in env_fields:
                env_cfg[k] = v
            elif k in inf_fields:
                inf_cfg[k] = v
            elif k in deco_fields:
                deco_cfg[k] = v
            elif k == "deco" and isinstance(v, dict):
                deco_cfg.update(v)
            elif k == "limits" and isinstance(v, dict):
                def dict_to_range(d):
                    return Range(d.get("min", []), d.get("max", []))

                env_cfg["limits"] = asdict(LimitsConfig(
                    joint_q=dict_to_range(v.get("joint_q", {})),
                    gripper=dict_to_range(v.get("gripper", {})),
                    head_q=dict_to_range(v.get("head_q", {})),
                    eef=dict_to_range(v.get("eef", {})),
                    eef_relative=dict_to_range(v.get("eef_relative", {})),
                    base=dict_to_range(v.get("base", {})),
                ))
            else:
                env_cfg[k] = v
        inf_cfg = _normalize_inference_config(inf_cfg)

    env_cfg, deco_cfg = _normalize_deco_config(env_cfg, inf_cfg, deco_cfg)
    # Merge defaults with provided config
    default_env = ConfigEnv()
    default_inf = ConfigInference()
    default_deco = ConfigDeco()

    merged_env = {**asdict(default_env), **env_cfg}
    merged_inf = {**asdict(default_inf), **inf_cfg}
    merged_deco = {**asdict(default_deco), **deco_cfg}

    env = ConfigEnv(**merged_env)
    inference = ConfigInference(**merged_inf)
    deco = ConfigDeco(**merged_deco)

    config = KuavoConfig(env=env, inference=inference, deco=deco)
    config.env.obs_key_map = config.env.build_obs_key_map(config.deco)
    config.validate()
    return config


# -----------------------
# Quick test when run as script
# -----------------------
if __name__ == "__main__":
    cfg = load_kuavo_config()
    print(isinstance(cfg, KuavoConfig))
    print("=== Env basic ===")
    print("eef_type:", cfg.env.eef_type)
    print("eef_name:", cfg.env.env_name)
    print("which_arm:", cfg.env.which_arm)
    print("cam keys:", cfg.env.obs_key_map)
    print("slice_robot:", cfg.env.gripper_slice)
    print("qiangnao_slice:", cfg.env.joint_q_slice)
    print("=== Inference basic ===")
    print("policy_type:", cfg.inference.policy_type)
    print("device:", cfg.inference.device)
    print("arm_state_keys",cfg.env.arm_state_keys)
