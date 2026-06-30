from __future__ import annotations

import torch
import pytest

from kuavo_train.wrapper.policy.deco.action_dispatch import (
    RecedingHorizonDispatcher,
    StrideActionDispatcher,
    TemporalEnsemblingDispatcher,
    make_action_dispatcher,
)


def _chunk(chunk_size: int = 32, action_dim: int = 2) -> torch.Tensor:
    values = torch.arange(chunk_size, dtype=torch.float32)
    return values.view(1, chunk_size, 1).repeat(1, 1, action_dim)


def test_receding_horizon_defaults_to_full_consecutive_chunk() -> None:
    dispatcher = RecedingHorizonDispatcher(n_action_steps=None)
    inference_calls = 0

    def predict_chunk(_observation: object) -> torch.Tensor:
        nonlocal inference_calls
        inference_calls += 1
        return _chunk()

    outputs = [dispatcher.select_action(predict_chunk, {}) for _ in range(32)]

    assert [int(action[0].item()) for action in outputs] == list(range(32))
    assert inference_calls == 1


def test_receding_horizon_replans_after_configured_steps() -> None:
    dispatcher = RecedingHorizonDispatcher(n_action_steps=4)
    inference_calls = 0

    def predict_chunk(_observation: object) -> torch.Tensor:
        nonlocal inference_calls
        inference_calls += 1
        return _chunk() + inference_calls * 100

    outputs = [dispatcher.select_action(predict_chunk, {}) for _ in range(5)]

    assert [int(action[0].item()) for action in outputs] == [100, 101, 102, 103, 200]
    assert inference_calls == 2


def test_stride_action_selects_30hz_chunk_at_10hz() -> None:
    dispatcher = StrideActionDispatcher(dataset_hz=30, target_hz=10, queue_steps=None)

    outputs = [dispatcher.select_action(lambda _: _chunk(), {}) for _ in range(11)]

    assert [int(action[0].item()) for action in outputs] == list(range(0, 32, 3))


def test_temporal_ensemble_matches_native_online_update() -> None:
    dispatcher = TemporalEnsemblingDispatcher(coefficient=0.1)
    chunks = iter((_chunk(chunk_size=4), _chunk(chunk_size=4) + 10))

    first = dispatcher.select_action(lambda _: next(chunks), {})
    second = dispatcher.select_action(lambda _: next(chunks), {})

    newest_weight = torch.exp(torch.tensor(-0.1))
    expected_second = (torch.tensor(1.0) + torch.tensor(10.0) * newest_weight) / (1.0 + newest_weight)
    assert torch.allclose(first, torch.tensor([0.0, 0.0]))
    assert torch.allclose(second, expected_second.repeat(2))


def test_reset_clears_dispatch_history() -> None:
    dispatcher = RecedingHorizonDispatcher(n_action_steps=4)
    dispatcher.select_action(lambda _: _chunk(), {})

    dispatcher.reset()
    action = dispatcher.select_action(lambda _: _chunk() + 100, {})

    assert int(action[0].item()) == 100
    assert dispatcher.get_dispatch_info()["chunk_id"] == 0


def test_invalid_dispatch_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported DECO action dispatch mode"):
        make_action_dispatcher("unknown", dataset_hz=30)


def test_temporal_ensemble_rejects_non_positive_coefficient() -> None:
    with pytest.raises(ValueError, match="coefficient"):
        TemporalEnsemblingDispatcher(coefficient=0.0)


def test_stride_action_requires_integer_frequency_ratio() -> None:
    with pytest.raises(ValueError, match="divisible"):
        StrideActionDispatcher(dataset_hz=30, target_hz=7, queue_steps=None)
