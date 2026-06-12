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

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Dict

import torch
import zmq


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


def _validate_safe_payload(value: Any, path: str = "payload") -> None:
    """限制 client/server 消息只携带推理需要的安全结构。"""

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


def _host_requires_token(host: str) -> bool:
    """判断保留的 server helper 是否绑定到非本机地址。"""

    local_hosts = {"localhost", "127.0.0.1", "::1", "[::1]"}
    return host.strip().lower() not in local_hosts


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
                "Binding inference server to a non-local host requires api_token. "
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


class BaseInferenceClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str = None,
    ):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        if hasattr(self, "socket"):
            self.socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        # timeout_ms 原来只保存在对象上但没有传给 ZeroMQ；这里显式设置收发超时，
        # 让 ACT/DP/DECO 的远端推理在 server 无响应时能尽早失败，而不是永久阻塞控制循环。
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def kill_server(self):
        """
        Kill the server.
        """
        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self, endpoint, data = None, requires_input = True
    ) -> dict:
        """
        Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        if not isinstance(endpoint, str):
            raise ValueError("endpoint must be a string.")
        request: dict = {"endpoint": endpoint}
        if requires_input:
            _validate_safe_payload(data, "request.data")
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token
        
        self.socket.send(TorchSerializer.to_bytes(request))
        message = self.socket.recv()
        response = TorchSerializer.from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            # server 端已经捕获到异常时，不把错误字典伪装成 action 继续向下游 postprocessor 传递。
            raise RuntimeError(f"Inference server error: {response['error']}")
        _validate_safe_payload(response, "response")

        return response

    def reset(self):
        """远端重置真实 policy，清空服务端 action queue。"""

        self.call_endpoint("reset", requires_input=False)

    def __del__(self):
        """Cleanup resources on destruction"""
        if hasattr(self, "socket"):
            self.socket.close(linger=0)
        if hasattr(self, "context"):
            self.context.term()


class ExternalRobotInferenceClient(BaseInferenceClient):
    """
    Client for communicating with the RealRobotServer
    """

    def select_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the action from the server.
        The exact definition of the observations is defined
        by the policy, which contains the modalities configuration.
        """
        return self.call_endpoint("select_action", observations)

    def reset(self) -> None:
        self.call_endpoint("reset", requires_input=False)
        

# policy client
class PolicyClient:
    def __init__(self, host="localhost", port=5555, timeout_ms=15000, api_token=None):
        # 保持 Kuavo ACT 原版 API：PolicyClient 只负责把已经 preprocessor 处理过的
        # observation 发给 server，并返回尚未 postprocessor 的模型 action。
        self.policy = ExternalRobotInferenceClient(
            host=host,
            port=port,
            timeout_ms=timeout_ms,
            api_token=api_token,
        )

    def select_action(self, obs_dict):
        response = self.policy.select_action(obs_dict)
        if isinstance(response, dict) and "action" in response:
            self.action = response["action"]
            self._dispatch_info = response.get("dispatch_info", {})
        else:
            self.action = response
            self._dispatch_info = {}
        return self.action

    def get_dispatch_info(self) -> Dict[str, Any]:
        """返回服务端随 DECO action 附带的轻量调度元数据。"""

        return dict(getattr(self, "_dispatch_info", {}))

    def reset(self) -> None:
        """保持本地 policy API 一致；真实 reset 通过 server endpoint 执行。"""

        self.policy.reset()
        self._dispatch_info = {}


# # convert hardware observations to policy's observation dict
# def hardware_obses_to_policy_obs_dict(robot_qpos, head_cam_h, wrist_cam_l, wrist_cam_r):
#     obs_dict = {
#             "video.head_cam_h": head_cam_h.reshape(1, 256, 256, 3),
#             "video.wrist_cam_l": wrist_cam_l.reshape(1, 256, 256, 3),
#             "video.wrist_cam_r": wrist_cam_r.reshape(1, 256, 256, 3),
#             "state.state": robot_qpos.reshape(1, -1).astype(np.float64),
#             "annotation.human.action.task_description": ["DEBUG"],
#         }
#     return obs_dict
