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
class ConfigDeco:
    """DECO 部署专用配置。

    这些字段只描述在线推理如何拼接观测、选择 checkpoint 和解释动作；
    不应覆盖 checkpoint 自己保存的模型结构字段。
    """

    inference_mode: str = "qiangnao_no_tactile"
    runtime_mode: str = "local_real"
    head_state_source: str = "live_joint_q"

    def validate(self, env: ConfigEnv, inference: ConfigInference):
        deco_policy = inference.policy_type == "deco"
        deco_client = inference.policy_type == "client" and env.state_layout.startswith("deco_")
        if not (deco_policy or deco_client):
            return
        # client 模式下真实 policy 在 server 端，但调用侧仍负责构造 DECO observation
        # 并反解 DECO action；因此只要使用 deco_* state_layout，就必须校验 env/deco schema。
        if self.inference_mode not in ["qiangnao_tactile", "qiangnao_no_tactile", "gripper_no_tactile"]:
            raise ValueError(
                "deco.inference_mode must be 'qiangnao_tactile', 'qiangnao_no_tactile', or 'gripper_no_tactile'."
            )
        if self.runtime_mode not in ["local_real", "local_sim", "server", "dry_run"]:
            raise ValueError("deco.runtime_mode must be 'local_real', 'local_sim', 'server', or 'dry_run'.")
        if self.head_state_source not in ["live_joint_q", "fixed_config"]:
            raise ValueError("deco.head_state_source must be 'live_joint_q' or 'fixed_config'.")

        if self.inference_mode in ["qiangnao_tactile", "qiangnao_no_tactile"]:
            if env.eef_type != "qiangnao" or env.state_layout != "deco_28d":
                raise ValueError(
                    f"{self.inference_mode} requires env.eef_type='qiangnao' and env.state_layout='deco_28d'."
                )
            if env.qiangnao_dof_needed != 6:
                raise ValueError(f"{self.inference_mode} requires env.qiangnao_dof_needed=6.")
        if self.inference_mode == "gripper_no_tactile":
            if env.eef_type not in ["leju_claw", "rq2f85"] or env.state_layout != "deco_18d":
                raise ValueError(
                    "gripper_no_tactile requires env.eef_type='leju_claw' or 'rq2f85' and env.state_layout='deco_18d'."
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
        inf_cfg: Dict[str, Any] = cfg.get('inference', {})
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
