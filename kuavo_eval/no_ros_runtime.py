from __future__ import annotations

import importlib.util

PURE_PYTHON_ROSBAG_DEPS = {
    "rosbag": "pip install --extra-index-url https://rospypi.github.io/simple/ rosbag",
    "genpy": "pip install --extra-index-url https://rospypi.github.io/simple/ genpy",
    "roslz4": "pip install --extra-index-url https://rospypi.github.io/simple/ roslz4",
}


def ensure_pure_python_rosbag_runtime() -> None:
    """Require only the Python rosbag runtime; no system ROS installation is needed."""

    missing = [
        module_name
        for module_name in PURE_PYTHON_ROSBAG_DEPS
        if importlib.util.find_spec(module_name) is None
    ]
    if not missing:
        return

    install_cmds = "\n".join(PURE_PYTHON_ROSBAG_DEPS[name] for name in missing)
    raise ImportError(
        "Offline Eval requires the pure-Python ROS1 bag runtime, but does not require "
        "a system ROS installation, roscore, or a sourced catkin workspace.\n"
        "Install the missing Python modules in the active environment:\n"
        f"{install_cmds}"
    )
