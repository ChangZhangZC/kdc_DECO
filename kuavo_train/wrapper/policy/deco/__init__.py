"""Kuavo-DECO policy wrapper package.

本目录只承载 DECO 接入 Kuavo/LeRobot 训练链路所需的 wrapper、配置和
preprocessor。DECO 模型主体仍位于 `third_party/deco`，LeRobot 全局补丁仍位于
`lerobot_patches/`。
"""

from pathlib import Path
import sys


def ensure_deco_on_path() -> None:
    """将 `third_party/deco` 加入导入路径，兼容 DECO 原生 `models.*` 导入方式。"""

    repo_root = Path(__file__).resolve().parents[4]
    deco_root = repo_root / "third_party" / "deco"
    deco_root_str = str(deco_root)
    if deco_root_str not in sys.path:
        sys.path.insert(0, deco_root_str)
