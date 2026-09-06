#!/usr/bin/env python3
from __future__ import annotations

from kuavo_eval.no_ros_runtime import ensure_pure_python_rosbag_runtime

ensure_pure_python_rosbag_runtime()

from kuavo_eval.open_loop_eval import main  # noqa: E402


if __name__ == "__main__":
    main()
