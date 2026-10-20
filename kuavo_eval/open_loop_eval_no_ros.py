#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kuavo_eval.no_ros_runtime import ensure_pure_python_rosbag_runtime

ensure_pure_python_rosbag_runtime()

# Make the no-ROS entrypoint independent from Hydra's package-relative config
# resolution. The canonical eval config is always loaded explicitly here.
if "--config-path" not in sys.argv:
    sys.argv.extend(["--config-path", str(PROJECT_ROOT / "configs" / "eval")])
if "--config-name" not in sys.argv:
    sys.argv.extend(["--config-name", "deco_open_loop_eval"])

from kuavo_eval.open_loop_eval import main  # noqa: E402


if __name__ == "__main__":
    main()
