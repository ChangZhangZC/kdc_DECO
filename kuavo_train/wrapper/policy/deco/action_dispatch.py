from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor


PredictChunk = Callable[[Any], Tensor]
SUPPORTED_ACTION_DISPATCH_MODES = {
    "receding_horizon",
    "temporal_ensemble",
}


def _validate_action_chunk(action_chunk: Tensor) -> Tensor:
    """将模型输出约束为 dispatcher 统一消费的 `[1, chunk, action_dim]`。"""

    if not isinstance(action_chunk, Tensor):
        raise TypeError(f"DECO predict_action_chunk must return torch.Tensor, got {type(action_chunk).__name__}.")
    if action_chunk.ndim != 3 or action_chunk.shape[0] != 1:
        raise ValueError(
            "DECO action dispatcher requires action chunk shape [1, chunk_size, action_dim], "
            f"got {tuple(action_chunk.shape)}."
        )
    if action_chunk.shape[1] <= 0 or action_chunk.shape[2] <= 0:
        raise ValueError(f"DECO action chunk dimensions must be positive, got {tuple(action_chunk.shape)}.")
    return action_chunk


class ActionDispatcher(ABC):
    """DECO 在线动作分发接口，只处理 chunk 的时间消费方式，不修改模型输出值域。"""

    mode: str

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._chunk_id = -1
        self._last_dispatch_info: dict[str, int | float | str | bool | None] = {
            "mode": self.mode,
            "chunk_id": None,
            "action_index": None,
            "queue_remaining": 0,
            "model_inference": False,
            "inference_time_ns": 0,
            "ensemble_count": None,
        }
        self._reset_state()

    def _predict(self, predict_chunk: PredictChunk, observation: Any) -> Tensor:
        start_ns = time.perf_counter_ns()
        action_chunk = _validate_action_chunk(predict_chunk(observation))
        inference_time_ns = time.perf_counter_ns() - start_ns
        self._chunk_id += 1
        self._last_inference_time_ns = inference_time_ns
        return action_chunk

    def _record_dispatch(
        self,
        *,
        action_index: int,
        queue_remaining: int,
        model_inference: bool,
        ensemble_count: int | None = None,
    ) -> None:
        self._last_dispatch_info = {
            "mode": self.mode,
            "chunk_id": self._chunk_id,
            "action_index": action_index,
            "queue_remaining": queue_remaining,
            "model_inference": model_inference,
            "inference_time_ns": self._last_inference_time_ns if model_inference else 0,
            "ensemble_count": ensemble_count,
        }

    def get_dispatch_info(self) -> dict[str, int | float | str | bool | None]:
        """返回最近一次动作选择的诊断信息，不参与控制决策。"""

        return dict(self._last_dispatch_info)

    @abstractmethod
    def _reset_state(self) -> None:
        """清空该策略跨控制周期保存的状态。"""

    @abstractmethod
    def select_action(self, predict_chunk: PredictChunk, observation: Any) -> Tensor:
        """从当前 observation 和模型 chunk 中选择一个即将下发的 action。"""


class RecedingHorizonDispatcher(ActionDispatcher):
    """连续执行 chunk 前 N 步；队列耗尽后使用最新观测重新规划。"""

    mode = "receding_horizon"

    def __init__(self, n_action_steps: int | None) -> None:
        if n_action_steps is not None and (
            isinstance(n_action_steps, bool)
            or not isinstance(n_action_steps, int)
            or n_action_steps <= 0
        ):
            raise ValueError("receding_horizon.n_action_steps must be a positive integer or null.")
        self.n_action_steps = n_action_steps
        super().__init__()

    def _reset_state(self) -> None:
        self._action_queue: deque[tuple[int, Tensor]] = deque()
        self._last_inference_time_ns = 0

    def select_action(self, predict_chunk: PredictChunk, observation: Any) -> Tensor:
        performed_inference = False
        if not self._action_queue:
            action_chunk = self._predict(predict_chunk, observation)
            chunk_size = int(action_chunk.shape[1])
            action_steps = chunk_size if self.n_action_steps is None else self.n_action_steps
            if action_steps > chunk_size:
                raise ValueError(
                    "receding_horizon.n_action_steps cannot exceed model chunk_size "
                    f"({chunk_size}), got {action_steps}."
                )
            self._action_queue.extend(
                (index, action)
                for index, action in enumerate(action_chunk[0, :action_steps])
            )
            performed_inference = True

        action_index, action = self._action_queue.popleft()
        self._record_dispatch(
            action_index=action_index,
            queue_remaining=len(self._action_queue),
            model_inference=performed_inference,
        )
        return action


class TemporalEnsemblingDispatcher(ActionDispatcher):
    """复用原生 DECO/ACT 在线公式，对多个重叠 chunk 的同一绝对时刻预测做指数加权。"""

    mode = "temporal_ensemble"

    def __init__(self, coefficient: float) -> None:
        if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)) or coefficient <= 0:
            raise ValueError("temporal_ensemble.coefficient must be positive.")
        self.coefficient = float(coefficient)
        super().__init__()

    def _reset_state(self) -> None:
        self._ensembled_actions: Tensor | None = None
        self._ensembled_actions_count: Tensor | None = None
        self._ensemble_weights: Tensor | None = None
        self._ensemble_weights_cumsum: Tensor | None = None
        self._last_inference_time_ns = 0

    def _initialize_weights(self, action_chunk: Tensor) -> None:
        chunk_size = int(action_chunk.shape[1])
        steps = torch.arange(chunk_size, dtype=action_chunk.dtype, device=action_chunk.device)
        self._ensemble_weights = torch.exp(-self.coefficient * steps)
        self._ensemble_weights_cumsum = torch.cumsum(self._ensemble_weights, dim=0)

    def select_action(self, predict_chunk: PredictChunk, observation: Any) -> Tensor:
        action_chunk = self._predict(predict_chunk, observation)
        if self._ensemble_weights is None:
            self._initialize_weights(action_chunk)

        if self._ensembled_actions is None:
            self._ensembled_actions = action_chunk.clone()
            self._ensembled_actions_count = torch.ones(
                (action_chunk.shape[1], 1),
                dtype=torch.long,
                device=action_chunk.device,
            )
        else:
            if action_chunk.shape[1] != self._ensembled_actions.shape[1] + 1:
                raise ValueError(
                    "Temporal Ensembling requires a stable DECO chunk_size across control cycles."
                )
            assert self._ensembled_actions_count is not None
            assert self._ensemble_weights is not None
            assert self._ensemble_weights_cumsum is not None
            counts = self._ensembled_actions_count
            self._ensembled_actions *= self._ensemble_weights_cumsum[counts - 1]
            self._ensembled_actions += action_chunk[:, :-1] * self._ensemble_weights[counts]
            self._ensembled_actions /= self._ensemble_weights_cumsum[counts]
            self._ensembled_actions_count = torch.clamp(counts + 1, max=action_chunk.shape[1])
            self._ensembled_actions = torch.cat(
                (self._ensembled_actions, action_chunk[:, -1:]),
                dim=1,
            )
            self._ensembled_actions_count = torch.cat(
                (
                    self._ensembled_actions_count,
                    torch.ones_like(self._ensembled_actions_count[-1:]),
                ),
                dim=0,
            )

        assert self._ensembled_actions_count is not None
        action = self._ensembled_actions[:, 0]
        ensemble_count = int(self._ensembled_actions_count[0].item())
        self._ensembled_actions = self._ensembled_actions[:, 1:]
        self._ensembled_actions_count = self._ensembled_actions_count[1:]
        self._record_dispatch(
            action_index=0,
            queue_remaining=int(self._ensembled_actions.shape[1]),
            model_inference=True,
            ensemble_count=ensemble_count,
        )
        return action.squeeze(0)


def make_action_dispatcher(
    mode: str,
    *,
    n_action_steps: int | None = None,
    temporal_ensemble_coefficient: float = 0.1,
) -> ActionDispatcher:
    """根据部署配置创建唯一 dispatcher，避免两种策略叠加。"""

    if mode == "receding_horizon":
        return RecedingHorizonDispatcher(n_action_steps=n_action_steps)
    if mode == "temporal_ensemble":
        return TemporalEnsemblingDispatcher(coefficient=temporal_ensemble_coefficient)
    raise ValueError(
        f"Unsupported DECO action dispatch mode {mode!r}; "
        f"expected one of {sorted(SUPPORTED_ACTION_DISPATCH_MODES)}."
    )
