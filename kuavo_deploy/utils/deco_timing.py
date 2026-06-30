from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _duration_summary(values_ns: list[int]) -> dict[str, float | None]:
    values_ms = [value / 1_000_000.0 for value in values_ns]
    return {
        "mean_ms": sum(values_ms) / len(values_ms) if values_ms else None,
        "p50_ms": _percentile(values_ms, 0.50),
        "p95_ms": _percentile(values_ms, 0.95),
        "p99_ms": _percentile(values_ms, 0.99),
    }


class DecoTimingRecorder:
    """批量记录 DECO 控制链路时间，避免每个控制周期同步写盘。"""

    def __init__(self, output_directory: Path, *, enabled: bool, flush_every_steps: int, target_hz: int) -> None:
        self.enabled = enabled
        self.flush_every_steps = flush_every_steps
        self.target_hz = target_hz
        self.trace_path = output_directory / "deco_timing_trace.jsonl"
        self.summary_path = output_directory / "deco_timing_summary.json"
        self._buffer: list[dict[str, Any]] = []
        self._records: list[dict[str, Any]] = []
        if self.enabled:
            self.trace_path.write_text("", encoding="utf-8")

    def record(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._buffer.append(record)
        self._records.append(record)
        if len(self._buffer) >= self.flush_every_steps:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or not self._buffer:
            return
        with self.trace_path.open("a", encoding="utf-8") as trace_file:
            for record in self._buffer:
                trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._buffer.clear()
        self._write_summary()

    def close(self) -> None:
        if not self.enabled:
            return
        self.flush()
        self._write_summary()

    def _write_summary(self) -> None:
        if not self._records:
            summary = {"steps": 0, "target_hz": self.target_hz}
            self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return

        duration_fields = {
            "preprocess": "preprocess_time_ns",
            "policy_call": "policy_call_time_ns",
            "model_inference": "model_inference_time_ns",
            "postprocess": "postprocess_time_ns",
            "environment_step": "environment_step_time_ns",
            "ros_sleep": "ros_sleep_time_ns",
            "observation": "observation_time_ns",
            "control_loop": "control_loop_time_ns",
        }
        durations = {
            name: _duration_summary(
                [int(record[field]) for record in self._records if record.get(field) is not None]
            )
            for name, field in duration_fields.items()
        }

        send_records = [record for record in self._records if record.get("command_send_wall_ns") is not None]
        send_intervals_ns = [
            int(later["command_send_wall_ns"]) - int(earlier["command_send_wall_ns"])
            for earlier, later in zip(send_records, send_records[1:])
            if earlier.get("episode") == later.get("episode")
        ]
        mean_interval_ns = sum(send_intervals_ns) / len(send_intervals_ns) if send_intervals_ns else None
        deadline_ns = 1_000_000_000 / self.target_hz
        deadline_misses = sum(
            int(record.get("control_loop_time_ns", 0) > deadline_ns) for record in self._records
        )
        clipped_steps = sum(int(record.get("clipped_dimensions", 0) > 0) for record in self._records)
        chunk_boundary_times = [
            int(record["control_loop_time_ns"])
            for record in self._records
            if record.get("model_inference") and record.get("control_loop_time_ns") is not None
        ]

        summary = {
            "steps": len(self._records),
            "target_hz": self.target_hz,
            "actual_command_hz": 1_000_000_000 / mean_interval_ns if mean_interval_ns else None,
            "command_interval": _duration_summary(send_intervals_ns),
            "deadline_miss_count": deadline_misses,
            "deadline_miss_ratio": deadline_misses / len(self._records),
            "clipped_step_count": clipped_steps,
            "clipped_step_ratio": clipped_steps / len(self._records),
            "chunk_boundary_control_loop": _duration_summary(chunk_boundary_times),
            "durations": durations,
        }
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
