#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kuavo-DECO LeRobot 数据集验证脚本。

用途：
1. 验证已经转换好的 LeRobot 数据集是否符合阶段一 Kuavo-DECO schema。
2. 不读取 rosbag，不依赖 ROS1/rospy/rosbag/kuavo_msgs。
3. 基础检查只依赖 Python 标准库；若环境中存在 pyarrow/pandas/numpy/cv2，则自动做更深入的
   parquet 数值检查与视频文件首帧检查。

注意：
当前 Codex 机器遵守 No-Runtime 约束，本脚本由用户在允许运行的环境中执行，并把输出反馈回来。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"


@dataclass
class CheckResult:
    """单条检查结果。"""

    name: str
    status: str
    detail: str


class DecoDatasetValidator:
    """对已转换的 Kuavo-DECO LeRobot 数据集做 schema 与数值检查。"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = Path(args.root).expanduser().resolve()
        self.report_path = Path(args.report).expanduser().resolve() if args.report else self.root / "deco_validation_report.md"
        self.results: list[CheckResult] = []
        self.info: dict[str, Any] = {}

    def add(self, name: str, status: str, detail: str) -> None:
        self.results.append(CheckResult(name=name, status=status, detail=detail))

    def validate(self) -> int:
        """执行全部检查并返回进程退出码。"""

        self.check_root()
        self.check_info_json()
        self.check_metadata_schema()
        self.check_file_layout()
        self.check_parquet_values()
        self.check_video_files()
        self.write_report()
        self.print_summary()
        return 1 if any(result.status == FAIL for result in self.results) else 0

    def check_root(self) -> None:
        if self.root.exists() and self.root.is_dir():
            self.add("dataset root", PASS, f"root exists: {self.root}")
        else:
            self.add("dataset root", FAIL, f"root missing or not a directory: {self.root}")

    def check_info_json(self) -> None:
        info_path = self.root / "meta" / "info.json"
        if not info_path.exists():
            self.add("meta/info.json", FAIL, f"missing: {info_path}")
            return
        try:
            self.info = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - 验证脚本需要把用户环境中的解析错误写进报告
            self.add("meta/info.json", FAIL, f"failed to parse JSON: {exc}")
            return

        self.add("meta/info.json", PASS, "parsed successfully")
        self.expect_equal("codebase_version", self.info.get("codebase_version"), self.args.expected_codebase)
        self.expect_equal("fps", self.info.get("fps"), self.args.expected_fps)
        self.expect_min("total_episodes", self.info.get("total_episodes"), 1)
        self.expect_min("total_frames", self.info.get("total_frames"), 1)
        self.expect_min("total_tasks", self.info.get("total_tasks"), 1)

    def expect_equal(self, name: str, actual: Any, expected: Any) -> None:
        if actual == expected:
            self.add(name, PASS, f"{actual!r}")
        else:
            self.add(name, FAIL, f"expected {expected!r}, got {actual!r}")

    def expect_min(self, name: str, actual: Any, minimum: int) -> None:
        if isinstance(actual, int) and actual >= minimum:
            self.add(name, PASS, f"{actual}")
        else:
            self.add(name, FAIL, f"expected integer >= {minimum}, got {actual!r}")

    def check_metadata_schema(self) -> None:
        if not self.info:
            self.add("metadata schema", SKIP, "info.json unavailable")
            return

        features = self.info.get("features", {})
        required = {
            "observation.state": [self.args.expected_state_dim],
            "observation.tactile": [self.args.expected_tactile_dim],
            "action": [self.args.expected_action_dim],
            self.args.expected_rgb_key: [3, self.args.expected_height, self.args.expected_width],
            self.args.expected_depth_key: [3, self.args.expected_height, self.args.expected_width],
        }

        for key, expected_shape in required.items():
            feature = features.get(key)
            if feature is None:
                self.add(f"feature {key}", FAIL, "missing")
                continue
            actual_shape = feature.get("shape")
            if actual_shape == expected_shape:
                self.add(f"feature {key}", PASS, f"shape={actual_shape}, dtype={feature.get('dtype')}")
            else:
                self.add(f"feature {key}", FAIL, f"expected shape={expected_shape}, got shape={actual_shape}")

        state_names = features.get("observation.state", {}).get("names", {}).get("state", [])
        action_names = features.get("action", {}).get("names", {}).get("action", [])
        tactile_names = features.get("observation.tactile", {}).get("names", {}).get("tactile", [])

        self.check_names("state names", state_names, self.args.expected_state_dim)
        self.check_names("action names", action_names, self.args.expected_action_dim)
        self.check_names("tactile names", tactile_names, self.args.expected_tactile_dim)

        if state_names[-2:] == ["head_head_yaw", "head_head_pitch"]:
            self.add("head state names", PASS, "last two state dims are head yaw/pitch")
        else:
            self.add("head state names", FAIL, f"last two state names: {state_names[-2:]}")

    def check_names(self, name: str, names: list[Any], expected_len: int) -> None:
        if len(names) == expected_len:
            self.add(name, PASS, f"{expected_len} names")
        else:
            self.add(name, FAIL, f"expected {expected_len} names, got {len(names)}")

    def check_file_layout(self) -> None:
        data_files = sorted((self.root / "data").glob("chunk-*/*.parquet"))
        episode_files = sorted((self.root / "meta" / "episodes").glob("chunk-*/*.parquet"))
        tasks_file = self.root / "meta" / "tasks.parquet"
        stats_file = self.root / "meta" / "stats.json"

        self.check_file_count("data parquet files", data_files)
        self.check_file_count("episode metadata parquet files", episode_files)
        self.check_file_exists("meta/tasks.parquet", tasks_file)
        self.check_file_exists("meta/stats.json", stats_file)

        for key in (self.args.expected_rgb_key, self.args.expected_depth_key):
            video_files = sorted((self.root / "videos" / key).glob("chunk-*/*.mp4"))
            self.check_file_count(f"video files for {key}", video_files)

    def check_file_count(self, name: str, files: list[Path]) -> None:
        if files:
            self.add(name, PASS, f"{len(files)} file(s)")
        else:
            self.add(name, FAIL, "no files found")

    def check_file_exists(self, name: str, path: Path) -> None:
        if path.exists():
            self.add(name, PASS, str(path))
        else:
            self.add(name, FAIL, f"missing: {path}")

    def check_parquet_values(self) -> None:
        if self.args.metadata_only:
            self.add("parquet value checks", SKIP, "--metadata-only enabled")
            return

        try:
            import numpy as np  # type: ignore
            import pandas as pd  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self.add("parquet value checks", SKIP, f"numpy/pandas unavailable: {exc}")
            return

        data_files = sorted((self.root / "data").glob("chunk-*/*.parquet"))
        if not data_files:
            self.add("parquet value checks", SKIP, "no data parquet files")
            return

        try:
            frames = [pd.read_parquet(path) for path in data_files]
            table = pd.concat(frames, ignore_index=True)
        except Exception as exc:  # noqa: BLE001
            self.add("parquet read", FAIL, f"failed to read data parquet: {exc}")
            return

        self.add("parquet read", PASS, f"{len(table)} rows from {len(data_files)} file(s)")
        self.check_columns(table)
        self.check_timestamp(table, np)
        state = self.check_vector_column(table, np, "observation.state", self.args.expected_state_dim)
        tactile = self.check_vector_column(table, np, "observation.tactile", self.args.expected_tactile_dim)
        action = self.check_vector_column(table, np, "action", self.args.expected_action_dim)
        self.check_head_semantics(np, state, action)
        self.check_tactile_semantics(np, tactile)

    def check_columns(self, table: Any) -> None:
        required_columns = {
            "observation.state",
            "observation.tactile",
            "action",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        }
        missing = sorted(required_columns - set(table.columns))
        if missing:
            self.add("parquet required columns", FAIL, f"missing columns: {missing}")
        else:
            self.add("parquet required columns", PASS, "all required scalar/vector columns exist")

    def series_to_matrix(self, np: Any, series: Any, dim: int) -> Any | None:
        values = []
        for value in series:
            array = np.asarray(value, dtype=np.float32).reshape(-1)
            if array.shape[0] != dim:
                return None
            values.append(array)
        return np.stack(values, axis=0)

    def check_vector_column(self, table: Any, np: Any, column: str, dim: int) -> Any | None:
        if column not in table.columns:
            self.add(f"{column} values", FAIL, "column missing")
            return None
        matrix = self.series_to_matrix(np, table[column], dim)
        if matrix is None:
            self.add(f"{column} values", FAIL, f"some rows are not dim={dim}")
            return None
        if np.isfinite(matrix).all():
            self.add(f"{column} values", PASS, f"shape={matrix.shape}, finite=True")
        else:
            self.add(f"{column} values", FAIL, "contains NaN or Inf")
        return matrix

    def check_timestamp(self, table: Any, np: Any) -> None:
        if "timestamp" not in table.columns:
            self.add("timestamp hz", FAIL, "timestamp column missing")
            return
        timestamps = np.asarray(table["timestamp"], dtype=np.float64).reshape(-1)
        if timestamps.shape[0] < 2:
            self.add("timestamp hz", WARN, "not enough frames to estimate hz")
            return
        diffs = np.diff(timestamps)
        expected_dt = 1.0 / float(self.args.expected_fps)
        max_abs_error = float(np.max(np.abs(diffs - expected_dt)))
        mean_dt = float(np.mean(diffs))
        tolerance = float(self.args.timestamp_tolerance_s)
        if max_abs_error <= tolerance:
            self.add("timestamp hz", PASS, f"mean_dt={mean_dt:.6f}, max_abs_error={max_abs_error:.6f}")
        else:
            self.add(
                "timestamp hz",
                FAIL,
                f"expected_dt={expected_dt:.6f}, mean_dt={mean_dt:.6f}, max_abs_error={max_abs_error:.6f}, tolerance={tolerance:.6f}",
            )

    def check_head_semantics(self, np: Any, state: Any | None, action: Any | None) -> None:
        if state is not None:
            head_state_std = np.std(state[:, -2:], axis=0)
            if np.max(head_state_std) <= self.args.head_state_std_max:
                self.add("head state stability", PASS, f"std={head_state_std.tolist()}")
            else:
                self.add("head state stability", WARN, f"std={head_state_std.tolist()} exceeds threshold")

        if action is not None:
            head_action_absmax = float(np.max(np.abs(action[:, -2:])))
            if head_action_absmax <= self.args.head_action_atol:
                self.add("head action zero", PASS, f"max_abs={head_action_absmax:.8f}")
            else:
                self.add("head action zero", FAIL, f"max_abs={head_action_absmax:.8f}")

    def check_tactile_semantics(self, np: Any, tactile: Any | None) -> None:
        if tactile is None:
            return
        abs_max = float(np.max(np.abs(tactile)))
        if abs_max == 0.0:
            self.add("tactile nonzero", WARN, "all tactile values are zero")
        else:
            self.add("tactile nonzero", PASS, f"max_abs={abs_max:.6f}")

    def check_video_files(self) -> None:
        if self.args.metadata_only or self.args.skip_video_probe:
            self.add("video probe", SKIP, "video probing disabled")
            return
        try:
            import cv2  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self.add("video probe", SKIP, f"cv2 unavailable: {exc}")
            return

        for key in (self.args.expected_rgb_key, self.args.expected_depth_key):
            files = sorted((self.root / "videos" / key).glob("chunk-*/*.mp4"))
            if not files:
                self.add(f"video probe {key}", FAIL, "no mp4 files")
                continue
            self.probe_one_video(cv2, key, files[0])

    def probe_one_video(self, cv2: Any, key: str, path: Path) -> None:
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                self.add(f"video probe {key}", FAIL, f"failed to open {path}")
                return
            ok, frame = capture.read()
            if not ok or frame is None:
                self.add(f"video probe {key}", FAIL, f"failed to read first frame from {path}")
                return
            height, width = frame.shape[:2]
            if height == self.args.expected_height and width == self.args.expected_width:
                self.add(f"video probe {key}", PASS, f"first_frame={width}x{height}")
            else:
                self.add(f"video probe {key}", FAIL, f"expected {self.args.expected_width}x{self.args.expected_height}, got {width}x{height}")
        finally:
            capture.release()

    def write_report(self) -> None:
        lines = [
            "# Kuavo-DECO LeRobot Validation Report",
            "",
            f"- Dataset root: `{self.root}`",
            f"- Expected FPS: `{self.args.expected_fps}`",
            "",
            "| Check | Status | Detail |",
            "| --- | --- | --- |",
        ]
        for result in self.results:
            detail = str(result.detail).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {result.name} | {result.status} | {detail} |")
        lines.append("")

        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text("\n".join(lines), encoding="utf-8")

    def print_summary(self) -> None:
        counts = {status: 0 for status in (PASS, WARN, FAIL, SKIP)}
        for result in self.results:
            counts[result.status] = counts.get(result.status, 0) + 1
        print("Kuavo-DECO validation summary:")
        print(f"  PASS: {counts[PASS]}")
        print(f"  WARN: {counts[WARN]}")
        print(f"  FAIL: {counts[FAIL]}")
        print(f"  SKIP: {counts[SKIP]}")
        print(f"  report: {self.report_path}")
        if counts[FAIL] > 0:
            print("Validation failed. Please inspect the report above.")
        else:
            print("Validation completed without FAIL items.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a converted Kuavo-DECO LeRobot dataset.")
    parser.add_argument("--root", default="data_example/lerobot", help="LeRobot dataset root directory.")
    parser.add_argument("--report", default=None, help="Markdown report output path. Default: <root>/deco_validation_report.md")
    parser.add_argument("--metadata-only", action="store_true", help="Only check metadata and file layout; skip parquet/video value checks.")
    parser.add_argument("--skip-video-probe", action="store_true", help="Skip cv2 video first-frame probing.")

    parser.add_argument("--expected-codebase", default="v3.0")
    parser.add_argument("--expected-fps", type=int, default=30)
    parser.add_argument("--expected-width", type=int, default=640)
    parser.add_argument("--expected-height", type=int, default=480)
    parser.add_argument("--expected-rgb-key", default="observation.images.head_cam_h")
    parser.add_argument("--expected-depth-key", default="observation.depth_h")
    parser.add_argument("--expected-state-dim", type=int, default=28)
    parser.add_argument("--expected-action-dim", type=int, default=28)
    parser.add_argument("--expected-tactile-dim", type=int, default=30)
    parser.add_argument("--timestamp-tolerance-s", type=float, default=0.01)
    parser.add_argument("--head-action-atol", type=float, default=1e-6)
    parser.add_argument("--head-state-std-max", type=float, default=0.02)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    validator = DecoDatasetValidator(args)
    return validator.validate()


if __name__ == "__main__":
    sys.exit(main())
