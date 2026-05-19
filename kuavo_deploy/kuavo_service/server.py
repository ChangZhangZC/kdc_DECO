# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import lerobot_patches.custom_patches  # noqa: F401 - 保持 Kuavo/LeRobot 自定义补丁在 policy 加载前生效
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
from io import BytesIO
import torch
import zmq

from kuavo_deploy.config import KuavoConfig, load_kuavo_config
from kuavo_deploy.utils.deco_obs_action import validate_deco_policy_compatibility
from kuavo_train.wrapper.policy.act.ACTPolicyWrapper import CustomACTPolicyWrapper
from kuavo_train.wrapper.policy.deco.DECOPolicyWrapper import CustomDECOPolicyWrapper
from kuavo_train.wrapper.policy.deco import DECOProcessor  # noqa: F401 - 注册 DECO processor，保持与本地 eval 入口一致
from kuavo_train.wrapper.policy.diffusion.DiffusionPolicyWrapper import CustomDiffusionPolicyWrapper

class TorchSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        buffer = BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        buffer = BytesIO(data)
        try:
            return torch.load(buffer, weights_only=True)
        except TypeError as exc:
            raise RuntimeError(
                "Inference message loading requires torch.load(weights_only=True). "
                "请升级到支持 weights_only=True 的 PyTorch 版本，避免对网络输入使用 pickle 反序列化。"
            ) from exc


def _host_requires_token(host: str) -> bool:
    """判断 server 是否绑定到非本机地址。

    ZeroMQ 消息仍使用 torch 序列化承载 tensor；即使已启用 weights_only，
    非本机暴露也必须要求 token，避免未授权客户端直接驱动机器人策略。
    """

    local_hosts = {"localhost", "127.0.0.1", "::1", "[::1]"}
    return host.strip().lower() not in local_hosts


def _validate_safe_payload(value: Any, path: str = "payload") -> None:
    """限制 ZMQ 请求/响应只携带 policy 推理需要的安全结构。"""

    if value is None or isinstance(value, (str, int, float, bool, torch.Tensor)):
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_safe_payload(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings, got {type(key).__name__}.")
            _validate_safe_payload(item, f"{path}.{key}")
        return
    raise ValueError(f"{path} contains unsupported type: {type(value).__name__}.")


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class BaseInferenceServer:
    """
    An inference server that spin up a ZeroMQ socket and listen for incoming requests.
    Can add custom endpoints by calling `register_endpoint`.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, api_token: str = None):
        if _host_requires_token(host) and not api_token:
            raise ValueError(
                "Binding inference server to a non-local host requires --api-token. "
                "请仅在可信网络中显式提供 token 后再暴露 ZeroMQ 推理服务。"
            )
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, EndpointHandler] = {}
        self.api_token = api_token

        # Register the ping endpoint by default
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)

    def _kill_server(self):
        """
        Kill the server.
        """
        self.running = False

    def _handle_ping(self) -> dict:
        """
        Simple ping handler that returns a success message.
        """
        return {"status": "ok", "message": "Server is running"}

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True):
        """
        Register a new endpoint to the server.

        Args:
            name: The name of the endpoint.
            handler: The handler function that will be called when the endpoint is hit.
            requires_input: Whether the handler requires input data.
        """
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def _validate_token(self, request: dict) -> bool:
        """
        Validate the API token in the request.
        """
        if self.api_token is None:
            return True  # No token required
        return request.get("api_token") == self.api_token

    def run(self):
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Server is ready and listening on {addr}")
        while self.running:
            try:
                message = self.socket.recv()
                request = TorchSerializer.from_bytes(message)
                if not isinstance(request, dict):
                    raise ValueError(f"Inference request must be a dict, got {type(request).__name__}.")

                # Validate token before processing request
                if not self._validate_token(request):
                    self.socket.send(
                        TorchSerializer.to_bytes({"error": "Unauthorized: Invalid API token"})
                    )
                    continue

                endpoint = request.get("endpoint", "select_action")
                if not isinstance(endpoint, str):
                    raise ValueError("request.endpoint must be a string.")

                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")

                handler = self._endpoints[endpoint]
                data = request.get("data", {})
                if handler.requires_input:
                    _validate_safe_payload(data, "request.data")
                result = (
                    handler.handler(data)
                    if handler.requires_input
                    else handler.handler()
                )
                _validate_safe_payload(result, "response")
                self.socket.send(TorchSerializer.to_bytes(result))
            except Exception as e:
                print(f"Error in server: {e}")
                import traceback

                print(traceback.format_exc())
                self.socket.send(TorchSerializer.to_bytes({"error": str(e)}))

class RobotInferenceServer(BaseInferenceServer):
    """
    Server with three endpoints for real robot policies
    """

    def __init__(self, model, host: str = "127.0.0.1", port: int = 5555, api_token: str = None):
        super().__init__(host, port, api_token)
        self.register_endpoint("select_action", model.select_action)
        self.register_endpoint("reset", model.reset, requires_input=False)

    @staticmethod
    def start_server(policy, port: int, host: str = "127.0.0.1", api_token: str = None):
        server = RobotInferenceServer(policy, host=host, port=port, api_token=api_token)
        server.run()

#####################################################################################

# Convert raw observations into the observations required by the model.
def hardware_obses_to_policy_obs_dict(obs):
    # ACT 原版 server/client 语义：eval/client 侧已经完成 run-root preprocessor。
    # 因此 server 只透传已经处理好的 observation，不在这里做归一化或图像预处理。
    obs_dict = obs
    return obs_dict

def resolve_config_path(config_path: Optional[str]) -> Optional[str]:
    """解析服务端部署配置路径。

    优先级保持简单且可追踪：
    1. 命令行 `--config`
    2. 环境变量 `KUAVO_DEPLOY_CONFIG`
    3. `load_kuavo_config()` 的默认 `configs/deploy/kuavo_env.yaml`
    """

    if config_path:
        return config_path
    return os.environ.get("KUAVO_DEPLOY_CONFIG")


def build_pretrained_path(cfg: KuavoConfig) -> Path:
    """按 Kuavo 原有 run-root 语义定位被服务端加载的 epoch 权重目录。"""

    inference = cfg.inference
    return Path("outputs") / "train" / inference.task / inference.method / inference.timestamp / f"epoch{inference.epoch}"


def load_policy_from_config(cfg: KuavoConfig):
    """根据部署配置加载 ACT/DP/DECO policy。

    server 只负责 policy 推理，不加载 run-root pre/postprocessor；processor 仍由
    real_single_test.py / sim_auto_test.py 或其他调用侧在发送请求前后执行。
    """

    pretrained_path = build_pretrained_path(cfg)
    policy_type = cfg.inference.policy_type.lower()

    if policy_type == "diffusion":
        policy = CustomDiffusionPolicyWrapper.from_pretrained(pretrained_path, strict=True)
    elif policy_type == "act":
        policy = CustomACTPolicyWrapper.from_pretrained(pretrained_path, strict=True)
    elif policy_type == "deco":
        policy = CustomDECOPolicyWrapper.from_pretrained(pretrained_path, strict=True)
        validate_deco_policy_compatibility(policy.config, cfg.deco, cfg.env)
    else:
        raise ValueError(
            "Server policy_type must be 'diffusion', 'act', or 'deco'. "
            f"Got '{cfg.inference.policy_type}'."
        )

    device = torch.device(cfg.inference.device)
    policy.eval()
    policy.to(device)
    if hasattr(policy, "reset"):
        policy.reset()
    return policy


class Policy:
    def __init__(self, config_path: Optional[str] = None):
        # 服务端与本地 eval 共用 `kuavo_deploy.config.load_kuavo_config`，
        # 避免 server 使用另一套旧 config parser 后出现字段语义分叉。
        resolved_config_path = resolve_config_path(config_path)
        self.config = load_kuavo_config(resolved_config_path)
        self.policy = load_policy_from_config(self.config)

    def select_action(self,obs):
        obs = hardware_obses_to_policy_obs_dict(obs)
        return self.policy.select_action(obs)

    def reset(self):
        """重置服务端真实 policy，清空 DECO/ACT 等策略内部 action queue。"""

        if hasattr(self.policy, "reset"):
            self.policy.reset()
        return {"status": "ok", "message": "Policy reset"}


def parse_args():
    parser = argparse.ArgumentParser(description="Kuavo ACT-compatible policy inference server")
    parser.add_argument(
        "--config",
        default=None,
        help="部署配置路径。若省略，则读取 KUAVO_DEPLOY_CONFIG；仍为空时使用 load_kuavo_config 默认配置。",
    )
    parser.add_argument("--host", default="127.0.0.1", help="ZeroMQ bind host，默认仅绑定本机。")
    parser.add_argument("--port", type=int, default=5555, help="ZeroMQ REP 端口。")
    parser.add_argument("--api-token", default=None, help="可选 API token；为空时不启用鉴权。")
    return parser.parse_args()


def main():
    args = parse_args()
    policy = Policy(config_path=args.config)

    # Start the server
    server = RobotInferenceServer(policy, host=args.host, port=args.port, api_token=args.api_token)
    server.run()


if __name__ == "__main__":
    main()
