#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kuavo rosbag -> LeRobot DECO 数据集转换脚本。

本文件是 DECO 阶段一的数据入口，刻意与原始 CvtRosbag2Lerobot.py 并行存在：
1. 保留原 Kuavo/ACT/DP 工具链中的 rosbag 读取、坏时间戳覆盖、compressedDepth 解码、
   动作范围保护、失败 bag 记录等保护性逻辑。
2. 只在 DECO 需要的核心 schema 上做收敛：RGB-D、28 维 state/action、30 维 tactile、
   训练数据 30Hz 时间轴，以及头部自由度的固定策略。
3. 当前版本为了兼容 LeRobot 图像/视频写入器，将 depth 以 3 通道 image/video 形式保存；
   后续 DECO wrapper 再把它还原为单通道 depth tensor 送入 depth backbone。
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler
from tqdm import tqdm

# 保持与原转换脚本一致：先加载 LeRobot patch，避免直接修改 third_party/lerobot 子模块。
try:
    import lerobot_patches.custom_patches  # noqa: F401
except ImportError:
    pass

from kuavo_data.common import kuavo_dataset as kuavo


try:
    # 新版 LeRobot 仓库路径。当前 third_party/lerobot 使用这一套 import。
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import validate_frame
except Exception:
    try:
        # 兼容旧版 LeRobot common.* 路径。
        from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.common.datasets.utils import validate_frame
    except Exception:
        sys.path.append(os.path.join(os.path.dirname(__file__), "../third_party/lerobot/src"))
        from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.utils import validate_frame


logging.basicConfig(level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True)])
log = logging.getLogger(__name__)


DECO_RGB_KEY = "head_cam_h"
DECO_DEPTH_KEY = "depth_h"

# DECO 固定状态顺序：左臂 7 + 左手 6 + 右臂 7 + 右手 6 + 头部 2。
DECO_STATE_NAMES = [
    *[f"left_arm_{name}" for name in kuavo.DEFAULT_ARM_JOINT_NAMES[:7]],
    *[f"left_hand_{name}" for name in kuavo.DEFAULT_DEXHAND_JOINT_NAMES[:6]],
    *[f"right_arm_{name}" for name in kuavo.DEFAULT_ARM_JOINT_NAMES[7:]],
    *[f"right_hand_{name}" for name in kuavo.DEFAULT_DEXHAND_JOINT_NAMES[6:]],
    *[f"head_{name}" for name in kuavo.DEFAULT_HEAD_JOINT_NAMES],
]
DECO_ACTION_NAMES = DECO_STATE_NAMES.copy()

# dexhand/touch_state 每个手 5 指，每指 3 个 normal force。
DECO_TACTILE_NAMES = [
    *[
        f"left_{finger}_normal_force{force_idx}"
        for finger in ("thumb", "index", "middle", "ring", "little")
        for force_idx in (1, 2, 3)
    ],
    *[
        f"right_{finger}_normal_force{force_idx}"
        for finger in ("thumb", "index", "middle", "ring", "little")
        for force_idx in (1, 2, 3)
    ],
]

# 原脚本中的机械臂动作保护范围。这里仍只 clamp 机械臂，不改手部和头部。
# 原数组中第 8/16 项是夹爪占位，DECO 使用 dexhand 6 DoF，所以右臂范围从索引 8 开始取。
DEFAULT_ARM_JOINT_RANGE = np.array(
    [
        [-3.14159, 1.5708],
        [-0.349066, 2.0944],
        [-1.5708, 1.5708],
        [-2.61799, 0],
        [-1.5708, 1.5708],
        [-1.309, 0.698132],
        [-0.698132, 0.698132],
        [-1, 1],
        [-3.14159, 1.5708],
        [-2.0944, 0.349066],
        [-1.5708, 1.5708],
        [-2.61799, 0],
        [-1.5708, 1.5708],
        [-0.698132, 1.309],
        [-0.698132, 0.698132],
        [-1, 1],
    ],
    dtype=np.float32,
)
DECO_LEFT_ARM_RANGE = DEFAULT_ARM_JOINT_RANGE[:7]
DECO_RIGHT_ARM_RANGE = DEFAULT_ARM_JOINT_RANGE[8:15]


@dataclass
class DatasetConfig:
    """数据集创建所需的最小配置，与原脚本保持同名概念。"""

    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


def cfg_select(cfg: DictConfig, key: str, default: Any = None) -> Any:
    """集中读取 Hydra 配置，避免在主逻辑中散落 try/except。"""

    return OmegaConf.select(cfg, key, default=default)


def as_float_array(name: str, value: Any, min_len: int | None = None) -> np.ndarray:
    """把 rosbag 解析结果转成 float32，并对关键维度做显式保护。"""

    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if min_len is not None and array.shape[0] < min_len:
        raise ValueError(f"{name} 维度不足，期望至少 {min_len}，实际 {array.shape[0]}")
    return array


def normalise_hand_position(name: str, value: Any, is_binary: bool) -> np.ndarray:
    """
    将灵巧手位置统一到 [0, 1]。

    原 Kuavo 清洗脚本在非 binary 模式下将 dexhand 原始 0-100 指令除以 100；
    DECO 固定使用左右手完整 12 DoF，因此这里保留同样归一化逻辑但不再裁成 1 DoF。
    """

    hand = as_float_array(name, value, min_len=12)[:12]
    if is_binary:
        hand = np.where(hand > 50.0, 1.0, 0.0)
    else:
        hand = hand / 100.0
    return np.clip(hand, 0.0, 1.0).astype(np.float32)


def depth_to_compatible_image(depth: Any, depth_range: list[float]) -> np.ndarray:
    """
    将 uint16/mm 深度图转成 3 通道 uint8 图像。

    这是第一阶段的兼容策略：LeRobot 的 image/video writer 对多模态图像路径更稳定，
    所以磁盘上保存 3 通道 depth image；训练 wrapper 中再按 depth 语义还原为单通道。
    """

    depth_array = np.asarray(depth)
    if depth_array.ndim == 3:
        depth_array = depth_array[:, :, 0]
    if depth_array.ndim != 2:
        raise ValueError(f"depth_h 期望二维深度图，实际 shape={depth_array.shape}")

    min_depth, max_depth = float(depth_range[0]), float(depth_range[1])
    if max_depth <= min_depth:
        raise ValueError(f"depth_range 非法：{depth_range}")

    # 与原 ACT/DP 清洗脚本保持一致：先按配置深度范围裁剪，再对当前帧局部归一化。
    # 这样磁盘中的 depth_h 与既有 Kuavo depth 数据格式一致，后续 wrapper 再按 depth 语义读取。
    clipped = np.clip(depth_array.astype(np.float32), min_depth, max_depth)
    local_min = float(clipped.min())
    local_max = float(clipped.max())
    normalised = ((clipped - local_min) / (local_max - local_min + 1e-9) * 255.0).astype(np.uint8)
    return np.repeat(normalised[:, :, None], 3, axis=2)


def decode_raw_16uc1_depth(msg: Any) -> np.ndarray:
    """
    备用 raw depth 解码器。

    当前冻结默认仍使用 /cam_h/depth/image_raw/compressedDepth；
    /camera/depth/image_rect_raw 仅作为可配置 fallback，因此此函数不改变默认工具链。
    """

    encoding = getattr(msg, "encoding", "")
    if encoding not in {"16UC1", "mono16"}:
        raise ValueError(f"raw depth encoding 不支持：{encoding}")

    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    dtype = np.dtype(">u2" if int(getattr(msg, "is_bigendian", 0)) else "<u2")
    row_width = step // dtype.itemsize
    expected_count = height * row_width
    depth = np.frombuffer(msg.data, dtype=dtype, count=expected_count)
    depth = depth.reshape(height, row_width)[:, :width].astype(np.uint16, copy=False)
    return cv2.resize(depth, (kuavo.RESIZE_W, kuavo.RESIZE_H), interpolation=cv2.INTER_NEAREST)


def sequence_timestamps(sequence: list[dict[str, Any]], key: str) -> np.ndarray:
    """提取并检查一段消息序列的时间戳。"""

    if not sequence:
        raise ValueError(f"{key} 没有可用消息")
    return np.asarray([float(item["timestamp"]) for item in sequence], dtype=np.float64)


def nearest_align(sequence: list[dict[str, Any]], target_timestamps: np.ndarray) -> list[dict[str, Any]]:
    """
    按目标时间轴做最近邻对齐。

    这里保留原脚本“以主图像流为基准”的思想，但不再依赖整数降采样倍数，
    以支持 100Hz 甚至更高采集频率稳定降到 30Hz。
    """

    ordered = sorted(sequence, key=lambda item: float(item["timestamp"]))
    times = np.asarray([float(item["timestamp"]) for item in ordered], dtype=np.float64)
    insert_positions = np.searchsorted(times, target_timestamps, side="left")

    aligned: list[dict[str, Any]] = []
    for target_time, position in zip(target_timestamps, insert_positions):
        if position <= 0:
            nearest_idx = 0
        elif position >= len(times):
            nearest_idx = len(times) - 1
        else:
            before = position - 1
            after = position
            nearest_idx = before if abs(target_time - times[before]) <= abs(times[after] - target_time) else after
        aligned.append(ordered[nearest_idx])
    return aligned


def build_target_timestamps(
    main_sequence: list[dict[str, Any]],
    required_sequences: dict[str, list[dict[str, Any]]],
    train_hz: int,
    sample_drop: int,
) -> np.ndarray:
    """从主 RGB 时间轴生成 30Hz 目标时间戳，并裁掉各 topic 不共同覆盖的边界。"""

    if train_hz <= 0:
        raise ValueError(f"train_hz 必须为正数，当前为 {train_hz}")

    main_times = sequence_timestamps(main_sequence, "main_timeline")
    if sample_drop > 0:
        if main_times.shape[0] <= sample_drop * 2:
            raise ValueError(
                f"主图像流帧数不足，无法执行 sample_drop={sample_drop}，实际帧数={main_times.shape[0]}"
            )
        main_times = main_times[sample_drop:-sample_drop]

    start_time = main_times[0]
    end_time = main_times[-1]
    for key, sequence in required_sequences.items():
        times = sequence_timestamps(sequence, key)
        start_time = max(start_time, times[0])
        end_time = min(end_time, times[-1])

    if end_time <= start_time:
        raise ValueError(
            f"各 topic 没有共同时间覆盖区间：start={start_time:.6f}, end={end_time:.6f}"
        )

    step = 1.0 / float(train_hz)
    count = int(np.floor((end_time - start_time) / step)) + 1
    if count <= 0:
        raise ValueError(f"目标时间轴为空：start={start_time:.6f}, end={end_time:.6f}, train_hz={train_hz}")
    return start_time + np.arange(count, dtype=np.float64) * step


def has_timestamp_gap(sequence: list[dict[str, Any]], threshold_s: float) -> bool:
    """检测动作 topic 是否存在明显断流。"""

    if len(sequence) < 2:
        return False
    times = sequence_timestamps(sequence, "arm_action")
    gaps = np.diff(times)
    return bool(np.any(gaps > threshold_s))


def compute_head_episode_mean(state_items: list[dict[str, Any]]) -> np.ndarray:
    """
    头部自由度策略：读取 joint_q[26:28]，按 episode 求均值后广播到所有帧。

    原 ACT/DP 最终 LeRobot schema 会丢弃头部；DECO 需要 28 维 state，因此这里显式保留
    头部观测，但不让 action 学习头部控制。
    """

    head_values = []
    for item in state_items:
        joint_q = as_float_array("observation.state/joint_q", item["data"], min_len=28)
        head_values.append(joint_q[26:28])
    if not head_values:
        raise ValueError("无法计算头部均值：observation.state 为空")
    return np.mean(np.stack(head_values, axis=0), axis=0).astype(np.float32)


def build_deco_state(joint_q: Any, hand_state_raw: Any, head_mean: np.ndarray, is_binary: bool) -> np.ndarray:
    """构造 DECO 28 维观测状态。"""

    joint_q_array = as_float_array("observation.state/joint_q", joint_q, min_len=28)
    hand_state = normalise_hand_position("observation.qiangnao", hand_state_raw, is_binary)

    state = np.concatenate(
        [
            joint_q_array[12:19],
            hand_state[:6],
            joint_q_array[19:26],
            hand_state[6:12],
            as_float_array("head_mean", head_mean, min_len=2)[:2],
        ],
        axis=0,
    ).astype(np.float32)
    if state.shape[0] != 28:
        raise ValueError(f"DECO state 维度错误：{state.shape[0]}")
    return state


def extract_arm_action(action_item: dict[str, Any], source_key: str) -> tuple[np.ndarray, np.ndarray]:
    """从优先级选中的 arm action topic 中取左右臂 7 DoF。"""

    action_array = as_float_array(source_key, action_item["data"])
    if source_key in {"action.kuavo_arm_traj", "action.kuavo_arm_traj_alt"}:
        action_array = as_float_array(source_key, action_array, min_len=14)
        return action_array[:7], action_array[7:14]
    if source_key == "action.joint_cmd":
        action_array = as_float_array(source_key, action_array, min_len=28)
        return action_array[12:19], action_array[19:26]
    raise ValueError(f"未知 arm action source：{source_key}")


def build_deco_action(
    arm_action_item: dict[str, Any],
    arm_source_key: str,
    hand_action_raw: Any,
    head_action_fill: list[float],
    is_binary: bool,
) -> np.ndarray:
    """构造 DECO 28 维动作。"""

    left_arm, right_arm = extract_arm_action(arm_action_item, arm_source_key)
    hand_action = normalise_hand_position("action.qiangnao", hand_action_raw, is_binary)
    head_action = as_float_array("head_action_fill", head_action_fill, min_len=2)[:2]

    action = np.concatenate(
        [
            left_arm.astype(np.float32),
            hand_action[:6],
            right_arm.astype(np.float32),
            hand_action[6:12],
            head_action.astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    if action.shape[0] != 28:
        raise ValueError(f"DECO action 维度错误：{action.shape[0]}")
    return action


def clamp_deco_arm_action(action: np.ndarray) -> np.ndarray:
    """沿用原脚本动作保护：只裁剪左右机械臂角度，避免异常轨迹污染训练。"""

    clipped = action.copy()
    clipped[0:7] = np.clip(clipped[0:7], DECO_LEFT_ARM_RANGE[:, 0], DECO_LEFT_ARM_RANGE[:, 1])
    clipped[13:20] = np.clip(clipped[13:20], DECO_RIGHT_ARM_RANGE[:, 0], DECO_RIGHT_ARM_RANGE[:, 1])
    return clipped.astype(np.float32)


class DecoRosbagReader(kuavo.KuavoRosbagReader):
    """DECO 专用 rosbag reader，复用 Kuavo reader 的 bag list/load 基础能力。"""

    def __init__(self, cfg: DictConfig):
        self.logger = logging.getLogger(__name__)
        self._msg_processer = kuavo.KuavoMsgProcesser()
        self.cfg = cfg

        self.rgb_key = f"observation.images.{cfg_select(cfg, 'deco.rgb_key', DECO_RGB_KEY)}"
        self.depth_key = f"observation.{cfg_select(cfg, 'deco.depth_key', DECO_DEPTH_KEY)}"
        self.depth_encoding = str(cfg_select(cfg, "deco.depth_encoding", "compressedDepth_png"))
        self.allow_raw_depth_fallback = bool(cfg_select(cfg, "deco.allow_raw_depth_fallback", False))
        self.force_scale = float(cfg_select(cfg, "deco.tactile_force_scale", 100.0))
        self.train_hz = int(cfg_select(cfg, "dataset.train_hz", 30))
        self.sample_drop = int(cfg_select(cfg, "dataset.sample_drop", 0))
        self.head_action_fill = list(cfg_select(cfg, "deco.head_action_fill", [0.0, 0.0]))

        configured_gap = cfg_select(cfg, "deco.arm_traj_gap_threshold_s", None)
        self.arm_gap_threshold_s = (
            float(configured_gap)
            if configured_gap is not None
            else 0.15 * 10.0 / float(max(self.train_hz, 1))
        )

        self._arm_topic_to_key = {
            str(cfg_select(cfg, "deco.arm_traj_synced_topic", "/kuavo_arm_traj_synced")): "action.kuavo_arm_traj_alt",
            str(cfg_select(cfg, "deco.arm_traj_topic", "/kuavo_arm_traj")): "action.kuavo_arm_traj",
            str(cfg_select(cfg, "deco.joint_cmd_topic", "/joint_cmd")): "action.joint_cmd",
        }
        priority_topics = list(
            cfg_select(
                cfg,
                "deco.arm_action_priority",
                ["/kuavo_arm_traj_synced", "/kuavo_arm_traj", "/joint_cmd"],
            )
        )
        self.arm_action_priority_keys = [
            self._arm_topic_to_key[topic] for topic in priority_topics if topic in self._arm_topic_to_key
        ]
        if "action.joint_cmd" not in self.arm_action_priority_keys:
            self.arm_action_priority_keys.append("action.joint_cmd")

        self._topic_process_map = self.build_topic_process_map(cfg)

    def build_topic_process_map(self, cfg: DictConfig) -> dict[str, tuple[str, Any]]:
        """集中定义 DECO 所需 topic，便于后续和配置文件逐项核对。"""

        rgb_topic = str(cfg_select(cfg, "deco.rgb_topic", "/cam_h/color/image_raw/compressed"))
        depth_topic = str(cfg_select(cfg, "deco.depth_topic", "/cam_h/depth/image_raw/compressedDepth"))
        raw_depth_topic = str(cfg_select(cfg, "deco.raw_depth_topic", "/camera/depth/image_rect_raw"))
        tactile_topic = str(cfg_select(cfg, "deco.tactile_topic", "/dexhand/touch_state"))
        hand_state_topic = str(cfg_select(cfg, "deco.hand_state_topic", "/dexhand/state"))
        hand_action_topic = str(cfg_select(cfg, "deco.hand_action_topic", "/control_robot_hand_position"))

        if self.depth_encoding == "compressedDepth_png":
            # compressedDepth 格式：msg.data 前面有配置头，需要跳过找到 PNG magic header。
            depth_process_fn = self._msg_processer.process_depth_image
        elif self.depth_encoding == "compressed_image":
            # CompressedImage 格式：msg.data 就是标准 PNG/JPEG 缓冲区，直接 cv2.imdecode。
            # 实际 rosbag 中 /cam_h/depth/image_raw/compressed 使用的就是此格式。
            depth_process_fn = self.process_compressed_depth_image
        elif self.depth_encoding == "raw_16uc1":
            depth_process_fn = self.process_raw_depth_image
        else:
            raise ValueError(
                f"不支持的 depth_encoding={self.depth_encoding}；"
                "当前支持 compressedDepth_png、compressed_image 或 raw_16uc1"
            )

        topic_map: dict[str, tuple[str, Any]] = {
            self.rgb_key: (rgb_topic, self._msg_processer.process_color_image),
            self.depth_key: (depth_topic, depth_process_fn),
            "observation.state": ("/sensors_data_raw", self._msg_processer.process_joint_state),
            "observation.qiangnao": (hand_state_topic, self._msg_processer.process_dex_state),
            "observation.tactile_raw": (tactile_topic, self.process_tactile_state),
            "action.joint_cmd": ("/joint_cmd", self._msg_processer.process_joint_cmd),
            "action.kuavo_arm_traj": ("/kuavo_arm_traj", self._msg_processer.process_kuavo_arm_traj),
            "action.kuavo_arm_traj_alt": ("/kuavo_arm_traj_synced", self._msg_processer.process_kuavo_arm_traj),
            "action.qiangnao": (hand_action_topic, self._msg_processer.process_qiangnao_cmd),
        }
        if self.allow_raw_depth_fallback and self.depth_encoding != "raw_16uc1":
            topic_map["observation.depth_h_raw"] = (raw_depth_topic, self.process_raw_depth_image)
        return topic_map

    def process_compressed_depth_image(self, msg: Any) -> dict[str, Any]:
        """
        处理 sensor_msgs/CompressedImage 格式的 depth 图像。

        与 process_depth_image (compressedDepth) 的区别：
        - compressedDepth：msg.data 前面有若干字节的配置头（quantization info 等），
          需要先搜索 PNG magic header (\x89PNG) 跳过前缀才能解码。
        - CompressedImage：msg.data 就是完整的 PNG/JPEG 压缩缓冲区，直接 cv2.imdecode 即可。

        实际 rosbag 中 /cam_h/depth/image_raw/compressed 使用的就是此 CompressedImage 格式，
        消息类型为 sensor_msgs/CompressedImage，编码通常是 16-bit PNG。
        """

        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)  # 直接解码，返回 uint16 深度图
        if image is None:
            self.logger.warning("compressed depth image 解码失败，跳过此帧")
            return None

        if image.dtype != np.uint16:
            self.logger.warning(
                "compressed depth image dtype=%s，期望 uint16；继续处理但精度可能下降",
                image.dtype,
            )

        depth_image = cv2.resize(image, (kuavo.RESIZE_W, kuavo.RESIZE_H), interpolation=cv2.INTER_NEAREST)
        return {"data": depth_image, "timestamp": msg.header.stamp.to_sec()}

    def process_raw_depth_image(self, msg: Any) -> dict[str, Any]:
        """raw 16UC1 depth fallback，默认不开启。"""

        return {"data": decode_raw_16uc1_depth(msg), "timestamp": msg.header.stamp.to_sec()}

    def process_tactile_state(self, msg: Any) -> dict[str, Any]:
        """
        解析 dexhand/touch_state。

        这里选择遇到缺失字段立即报错，而不是补零。触觉 LoRA 的训练对触觉语义非常敏感，
        静默补零会把“传感器坏/缺 topic”和“无接触”混在一起。
        """

        tactile_values: list[float] = []
        for hand_name in ("left_hand", "right_hand"):
            hand = getattr(msg, hand_name, None)
            if hand is None:
                raise ValueError(f"touch_state 缺少字段：{hand_name}")
            fingers = list(hand)
            if len(fingers) < 5:
                raise ValueError(f"touch_state.{hand_name} 手指数不足：{len(fingers)}")
            for finger in fingers[:5]:
                for force_name in ("normal_force1", "normal_force2", "normal_force3"):
                    if not hasattr(finger, force_name):
                        raise ValueError(f"touch_state.{hand_name} 缺少字段：{force_name}")
                    tactile_values.append(float(getattr(finger, force_name)) / self.force_scale)

        tactile = np.asarray(tactile_values, dtype=np.float32)
        if tactile.shape[0] != 30:
            raise ValueError(f"触觉向量维度错误：{tactile.shape[0]}")
        return {"data": tactile, "timestamp": msg.header.stamp.to_sec()}

    def process_rosbag(self, bag_path: str) -> dict[str, Any]:
        """
        读取并对齐单个 rosbag。

        关键保护逻辑：
        - 未索引 bag 由父类 load_raw_rosbag 自动 reindex。
        - 所有消息时间戳统一用 bag_time 覆盖 header stamp，保留原脚本对坏 header 的修正。
        - 处理函数返回 None 的帧会跳过，后续由必需 topic 检查统一报错。
        """

        bag = self.load_raw_rosbag(bag_path)
        process_data: dict[str, list[dict[str, Any]]] = {
            key: [] for key in self._topic_process_map.keys()
        }
        try:
            topic_names = [topic for topic, _ in self._topic_process_map.values()]
            for topic, msg, bag_time in bag.read_messages(topics=topic_names):
                for data_name, (mapped_topic, process_func) in self._topic_process_map.items():
                    if topic != mapped_topic:
                        continue
                    msg_data = process_func(msg)
                    if msg_data is None:
                        continue
                    msg_data["timestamp"] = bag_time.to_sec()
                    process_data[data_name].append(msg_data)
        finally:
            bag.close()

        return self.align_frame_data(process_data)

    def select_arm_action_source(self, process_data: dict[str, list[dict[str, Any]]]) -> str:
        """
        按配置优先级选择 arm action。

        优先使用同步后的 /kuavo_arm_traj_synced；若该 topic 缺失或断流，再退回
        /kuavo_arm_traj，最后退回 /joint_cmd。这样保留原脚本的 action 覆写意图，
        同时避免 DECO 数据中出现 999 这类占位动作。
        """

        for key in self.arm_action_priority_keys:
            sequence = process_data.get(key, [])
            if not sequence:
                self.logger.warning("arm action source %s 为空，尝试下一个候选", key)
                continue
            if key in {"action.kuavo_arm_traj", "action.kuavo_arm_traj_alt"} and has_timestamp_gap(
                sequence, self.arm_gap_threshold_s
            ):
                self.logger.warning(
                    "arm action source %s 存在超过 %.3fs 的断流，尝试下一个候选",
                    key,
                    self.arm_gap_threshold_s,
                )
                continue
            return key

        raise ValueError(
            "没有可用的 arm action source；已尝试 /kuavo_arm_traj_synced, /kuavo_arm_traj, /joint_cmd"
        )

    def align_frame_data(self, process_data: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        """将所有 DECO 必需字段对齐到 30Hz 目标时间轴。"""

        arm_action_key = self.select_arm_action_source(process_data)
        required_keys = [
            self.rgb_key,
            self.depth_key,
            "observation.state",
            "observation.qiangnao",
            "observation.tactile_raw",
            arm_action_key,
            "action.qiangnao",
        ]

        if self.allow_raw_depth_fallback and not process_data.get(self.depth_key):
            raw_depth = process_data.get("observation.depth_h_raw", [])
            if raw_depth:
                process_data[self.depth_key] = raw_depth

        missing_keys = [key for key in required_keys if not process_data.get(key)]
        if missing_keys:
            raise ValueError(f"rosbag 缺少 DECO 必需数据：{missing_keys}")

        required_sequences = {key: process_data[key] for key in required_keys}
        target_timestamps = build_target_timestamps(
            process_data[self.rgb_key],
            required_sequences,
            train_hz=self.train_hz,
            sample_drop=self.sample_drop,
        )

        aligned = {key: nearest_align(process_data[key], target_timestamps) for key in required_keys}
        aligned["__metadata__"] = {
            "target_hz": self.train_hz,
            "num_frames": int(target_timestamps.shape[0]),
            "arm_action_source": arm_action_key,
        }
        return aligned


def create_empty_deco_dataset(
    repo_id: str,
    robot_type: str,
    mode: str,
    dataset_config: DatasetConfig,
    root: str | None,
) -> LeRobotDataset:
    """创建符合 DECO 阶段一 schema 的 LeRobotDataset。"""

    motors_features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (28,),
            "names": {"state": DECO_STATE_NAMES},
        },
        "observation.tactile": {
            "dtype": "float32",
            "shape": (30,),
            "names": {"tactile": DECO_TACTILE_NAMES},
        },
        "action": {
            "dtype": "float32",
            "shape": (28,),
            "names": {"action": DECO_ACTION_NAMES},
        },
    }

    camera_features = {
        "observation.images.head_cam_h": {
            "dtype": mode,
            "shape": (3, kuavo.RESIZE_H, kuavo.RESIZE_W),
            "names": ["channels", "height", "width"],
        },
        "observation.depth_h": {
            "dtype": mode,
            "shape": (3, kuavo.RESIZE_H, kuavo.RESIZE_W),
            "names": ["channels", "height", "width"],
        },
    }

    features = {**motors_features, **camera_features}
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=kuavo.TRAIN_HZ,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
        root=root,
    )


def validate_deco_frame(frame: dict[str, Any], dataset: LeRobotDataset) -> dict[str, Any]:
    """
    调用 LeRobot 原生校验前，先做 DECO schema 的显式维度检查。

    这些检查不依赖运行数据集，只约束每帧字段是否满足我们在 PLANS.md 冻结的结构。
    LeRobot 原生 validate_frame 期望 float feature 是 numpy array，因此这里仅转换校验副本；
    真正写入时仍交给 dataset.add_frame 处理 torch -> numpy 的常规流程。
    """

    if tuple(frame["observation.state"].shape) != (28,):
        raise ValueError(f"observation.state 维度错误：{frame['observation.state'].shape}")
    if tuple(frame["observation.tactile"].shape) != (30,):
        raise ValueError(f"observation.tactile 维度错误：{frame['observation.tactile'].shape}")
    if tuple(frame["action"].shape) != (28,):
        raise ValueError(f"action 维度错误：{frame['action'].shape}")

    validation_frame = {}
    for name, value in frame.items():
        if isinstance(value, torch.Tensor):
            validation_frame[name] = value.detach().cpu().numpy()
        else:
            validation_frame[name] = value
    validate_frame(validation_frame, dataset.features)
    return frame


def populate_dataset(
    dataset: LeRobotDataset,
    bag_files: list[str],
    task: str,
    reader: DecoRosbagReader,
    cfg: DictConfig,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    """将 rosbag 列表写入 LeRobotDataset。"""

    depth_range = list(cfg_select(cfg, "dataset.depth_range", [0, 1500]))
    is_binary = bool(cfg_select(cfg, "dataset.is_binary", False))
    failed_bags: list[tuple[str, str]] = []
    episodes = episodes if episodes is not None else list(range(len(bag_files)))

    for ep_idx, ep_path in tqdm(
        zip(episodes, bag_files),
        total=len(bag_files),
        desc="Processing DECO rosbag",
    ):
        try:
            bag_data = reader.process_rosbag(ep_path)
            metadata = bag_data["__metadata__"]
            arm_action_source = str(metadata["arm_action_source"])
            head_mean = compute_head_episode_mean(bag_data["observation.state"])

            num_frames = int(metadata["num_frames"])
            log.info(
                "Episode %s: %s frames at %sHz, arm_action_source=%s",
                ep_idx,
                num_frames,
                metadata["target_hz"],
                arm_action_source,
            )

            for frame_idx in range(num_frames):
                state = build_deco_state(
                    bag_data["observation.state"][frame_idx]["data"],
                    bag_data["observation.qiangnao"][frame_idx]["data"],
                    head_mean,
                    is_binary=is_binary,
                )
                action = build_deco_action(
                    bag_data[arm_action_source][frame_idx],
                    arm_action_source,
                    bag_data["action.qiangnao"][frame_idx]["data"],
                    reader.head_action_fill,
                    is_binary=is_binary,
                )
                tactile = as_float_array(
                    "observation.tactile",
                    bag_data["observation.tactile_raw"][frame_idx]["data"],
                    min_len=30,
                )[:30]

                frame = {
                    "observation.images.head_cam_h": bag_data[reader.rgb_key][frame_idx]["data"],
                    "observation.depth_h": depth_to_compatible_image(
                        bag_data[reader.depth_key][frame_idx]["data"], depth_range
                    ),
                    "observation.state": torch.from_numpy(state).float(),
                    "observation.tactile": torch.from_numpy(tactile.astype(np.float32)).float(),
                    "action": torch.from_numpy(clamp_deco_arm_action(action)).float(),
                    "task": task,
                }
                dataset.add_frame(validate_deco_frame(frame, dataset))

            dataset.save_episode()
            # 与原脚本一致：每个 episode 后重建 hf_dataset，降低长时间转换时的内存压力。
            dataset.hf_dataset = dataset.create_hf_dataset()
        except Exception as exc:  # noqa: BLE001 - 数据转换脚本需要记录每个失败 bag 的完整路径
            log.error("Failed to process DECO bag %s: %s", ep_path, exc)
            failed_bags.append((ep_path, str(exc)))
            # 如果某个 bag 在写入中途失败，必须清空 episode_buffer，避免下一条 bag 继承残帧。
            if getattr(dataset, "episode_buffer", None) is not None:
                dataset.episode_buffer = dataset.create_episode_buffer()

    if failed_bags:
        with open("error_DECO.txt", "w", encoding="utf-8") as file:
            for bag_path, reason in failed_bags:
                file.write(f"{bag_path}\t{reason}\n")
        log.warning("DECO conversion finished with %d failed bags; see error_DECO.txt", len(failed_bags))

    return dataset


def port_deco_rosbag(
    raw_dir: str,
    repo_id: str,
    raw_format: str = "rosbag",
    root: str | None = None,
    n: int | None = None,
    robot_type: str = "kuavo",
    mode: str = "video",
    task: str = "Pick and Place",
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    overwrite: bool = False,
    cfg: DictConfig | None = None,
) -> LeRobotDataset:
    """DECO 数据转换主入口。"""

    if raw_format != "rosbag":
        raise NotImplementedError("DECO 阶段一仅支持 rosbag 输入")
    if cfg is None:
        raise ValueError("port_deco_rosbag 需要传入 Hydra cfg")
    if push_to_hub:
        raise NotImplementedError("当前 DECO 阶段一不自动 push_to_hub")

    reader = DecoRosbagReader(cfg)
    bag_files = reader.list_bag_files(raw_dir)
    if n is not None and n > 0:
        if n > len(bag_files):
            raise ValueError(f"请求转换 {n} 个 bag，但目录中只有 {len(bag_files)} 个")
        bag_files = list(np.random.choice(bag_files, n, replace=False))

    output_root = Path(root) if root is not None else LEROBOT_HOME / repo_id
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"目标 LeRobot 数据集已存在：{output_root}；如需覆盖，请在配置中设置 deco.overwrite=true"
            )
        shutil.rmtree(output_root)

    dataset = create_empty_deco_dataset(
        repo_id=repo_id,
        robot_type=robot_type,
        mode=mode,
        dataset_config=DatasetConfig(),
        root=str(output_root),
    )
    return populate_dataset(dataset, bag_files, task, reader, cfg, episodes=episodes)


@hydra.main(version_base=None, config_path="../configs/data", config_name="KuavoRosbag2Lerobot_deco")
def main(cfg: DictConfig) -> None:
    """Hydra CLI 入口。"""

    kuavo.init_parameters(cfg)

    raw_dir = str(cfg.rosbag.rosbag_dir)
    version = str(cfg.rosbag.lerobot_dir)
    num_used = cfg.rosbag.num_used if cfg.rosbag.num_used is not None else None
    output_root = Path(version)
    if not output_root.is_absolute():
        output_root = Path(raw_dir).parent / version / "lerobot"

    task_name = os.path.basename(os.path.normpath(raw_dir))
    repo_id = f"lerobot/{task_name}_deco"
    overwrite = bool(cfg_select(cfg, "deco.overwrite", False))

    log.info("Start DECO rosbag conversion: raw_dir=%s, output_root=%s", raw_dir, output_root)
    port_deco_rosbag(
        raw_dir=raw_dir,
        repo_id=repo_id,
        root=str(output_root),
        n=num_used,
        task=str(cfg_select(cfg, "dataset.task_description", "Pick and Place")),
        overwrite=overwrite,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
