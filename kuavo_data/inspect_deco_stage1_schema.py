#!/usr/bin/env python3
"""
DECO 阶段一 Rosbag Schema Inspector。

用途：
1. 只读检查 Kuavo rosbag 中和 DECO 数据转换相关的关键 topic。
2. 导出头部相机 `/cam_h/color/image_raw/compressed` 的完整图、左半图、右半图，
   供人工判断它是单目整图还是左右拼接双目图。
3. 统计 `/sensors_data_raw.joint_data.joint_q` 中头部关节候选切片的稳定性，
   供后续决定 `observation.state[26:28]` 使用实测固定角还是回退补零。

注意：
- 本脚本不写 LeRobot 数据集，不修改 rosbag，不改公共 reader。
- 本脚本会在 `data_example/inspect_outputs/` 下写出 3 张检查图片。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - 用户运行环境检查
    raise SystemExit(
        "缺少 cv2/OpenCV，无法解码压缩图像。请在已有项目环境中运行本脚本。"
    ) from exc

try:
    import rosbag
except ImportError as exc:  # pragma: no cover - 用户运行环境检查
    raise SystemExit(
        "缺少 rosbag Python 包，无法读取 .bag。请在已有 ROS/rosbag 环境中运行本脚本。"
    ) from exc


DEFAULT_BAG_PATH = Path("data_example/vr_record_2026-04-15-15-57-47.bag")
DEFAULT_OUTPUT_DIR = Path("data_example/inspect_outputs")

IMAGE_TOPIC = "/cam_h/color/image_raw/compressed"
SENSORS_TOPIC = "/sensors_data_raw"

# 这里保留用户当前 rosbag 中的完整关键 topic 列表，便于一次性确认存在性、类型、频率和字段形态。
RECORDED_TOPICS = [
    "/cam_h/color/camera_info",
    IMAGE_TOPIC,
    "/cam_h/color/metadata",
    "/cam_h/depth/image_raw/compressed",
    "/cam_l/color/camera_info",
    "/cam_l/color/image_raw/compressed",
    "/cam_l/color/metadata",
    "/cam_l/depth/image_rect_raw/compressed",
    "/cam_r/color/camera_info",
    "/cam_r/color/image_raw/compressed",
    "/cam_r/color/metadata",
    "/cam_r/depth/image_rect_raw/compressed",
    "/robot_head_motion_data",
    "/control_robot_hand_position",
    "/dexhand/state",
    "/dexhand/touch_state",
    "/joint_cmd",
    "/kuavo_arm_traj",
    SENSORS_TOPIC,
    "/tf",
    "/tf_static",
    "/humanoid_controller/wbc_arm_eef_pose",
]

CAMERA_INFO_TOPICS = [
    "/cam_h/color/camera_info",
    "/cam_l/color/camera_info",
    "/cam_r/color/camera_info",
]

FIELD_CHECK_TOPICS = [
    SENSORS_TOPIC,
    "/dexhand/state",
    "/dexhand/touch_state",
    "/control_robot_hand_position",
    "/kuavo_arm_traj",
    "/joint_cmd",
    "/robot_head_motion_data",
]


def stamp_to_sec(msg: Any, bag_time: Any) -> float:
    """优先使用消息 header 时间；不可用时回退到 rosbag 记录时间。"""
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None and hasattr(stamp, "to_sec"):
        sec = float(stamp.to_sec())
        if sec > 0:
            return sec
    return float(bag_time.to_sec())


def is_sequence(value: Any) -> bool:
    """判断 ROS 字段是否像数组，同时排除 bytes/string。"""
    if isinstance(value, (str, bytes, bytearray)):
        return False
    return hasattr(value, "__len__") and hasattr(value, "__iter__")


def safe_len(value: Any) -> int | None:
    """安全获取字段长度；不支持长度的字段返回 None。"""
    try:
        return len(value)
    except TypeError:
        return None


def sample_sequence(value: Any, limit: int = 6) -> list[Any]:
    """取数组字段前几个元素，避免终端输出过长。"""
    try:
        return list(value[:limit])
    except TypeError:
        return list(value)[:limit]


def public_field_names(obj: Any) -> list[str]:
    """读取 ROS message 的公开字段名；优先使用 __slots__，避免 dir() 带来大量方法名。"""
    slots = getattr(obj, "__slots__", None)
    if slots:
        return [name for name in slots if not name.startswith("_")]
    names = []
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if callable(value):
            continue
        names.append(name)
    return names


def describe_scalar_or_sequence(value: Any) -> str:
    """将普通字段压缩成一行描述，重点保留类型、长度和少量样例。"""
    if isinstance(value, bytes):
        return f"bytes[len={len(value)}]"
    if isinstance(value, bytearray):
        return f"bytearray[len={len(value)}]"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return repr(value)
    if is_sequence(value):
        length = safe_len(value)
        sample = sample_sequence(value)
        return f"{type(value).__name__}[len={length}, sample={sample}]"
    return f"<{type(value).__name__}>"


def print_message_schema(msg: Any, *, title: str, max_depth: int = 2) -> None:
    """递归打印第一条消息的字段结构，帮助确认真实 ROS message 长什么样。"""

    def walk(obj: Any, indent: int, depth: int) -> None:
        if depth > max_depth:
            return
        for field_name in public_field_names(obj):
            try:
                value = getattr(obj, field_name)
            except Exception as exc:
                print(f"{'  ' * indent}- {field_name}: <读取失败: {exc}>")
                continue

            prefix = f"{'  ' * indent}- {field_name}: "
            if isinstance(value, (bytes, bytearray, str, int, float, bool)) or value is None:
                print(prefix + describe_scalar_or_sequence(value))
            elif is_sequence(value):
                print(prefix + describe_scalar_or_sequence(value))
                first = sample_sequence(value, 1)
                if first and hasattr(first[0], "__slots__") and depth < max_depth:
                    print(f"{'  ' * (indent + 1)}first_item:")
                    walk(first[0], indent + 2, depth + 1)
            elif hasattr(value, "__slots__") and depth < max_depth:
                print(prefix + f"<{type(value).__name__}>")
                walk(value, indent + 1, depth + 1)
            else:
                print(prefix + describe_scalar_or_sequence(value))

    print(f"\n## {title}")
    print(f"type: {type(msg).__name__}")
    walk(msg, indent=0, depth=0)


def get_nested_attr(obj: Any, attr_path: str) -> Any | None:
    """读取类似 `joint_data.joint_q` 的嵌套字段。"""
    current = obj
    for part in attr_path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def open_bag_readonly(bag_path: Path) -> rosbag.Bag:
    """只读打开 rosbag；未索引时仅尝试 allow_unindexed，不做 reindex 写入。"""
    try:
        return rosbag.Bag(str(bag_path), "r")
    except rosbag.bag.ROSBagUnindexedException:
        print("[WARN] rosbag 未索引：尝试 allow_unindexed=True 只读打开，不执行 reindex。")
        return rosbag.Bag(str(bag_path), "r", allow_unindexed=True)


def get_topic_info_map(bag: rosbag.Bag) -> dict[str, Any]:
    """获取 rosbag 内 topic 元信息。"""
    return bag.get_type_and_topic_info().topics


def first_message_for_topic(bag: rosbag.Bag, topic: str) -> tuple[Any | None, float | None]:
    """读取某个 topic 的第一条消息。"""
    for _, msg, bag_time in bag.read_messages(topics=[topic]):
        return msg, stamp_to_sec(msg, bag_time)
    return None, None


def scan_topic_time_range(
    bag: rosbag.Bag,
    topic: str,
    *,
    max_scan: int,
) -> tuple[float | None, float | None, int, bool]:
    """扫描 topic 的首尾时间。max_scan=0 表示全量扫描。"""
    first_sec: float | None = None
    last_sec: float | None = None
    count = 0
    truncated = False
    for _, msg, bag_time in bag.read_messages(topics=[topic]):
        sec = stamp_to_sec(msg, bag_time)
        if first_sec is None:
            first_sec = sec
        last_sec = sec
        count += 1
        if max_scan > 0 and count >= max_scan:
            truncated = True
            break
    return first_sec, last_sec, count, truncated


def print_topic_summary(bag: rosbag.Bag, *, max_topic_scan: int) -> None:
    """打印关键 topic 的存在性、类型、数量、频率和时间范围。"""
    topic_info = get_topic_info_map(bag)
    print("\n# 1. Topic Summary")
    print("status | topic | type | count | hz | first_time | last_time | note")
    print("--- | --- | --- | ---: | ---: | ---: | ---: | ---")
    for topic in RECORDED_TOPICS:
        info = topic_info.get(topic)
        if info is None:
            print(f"MISSING | {topic} | - | 0 | - | - | - | rosbag 中不存在")
            continue

        first_sec, last_sec, scanned_count, truncated = scan_topic_time_range(
            bag, topic, max_scan=max_topic_scan
        )
        count = getattr(info, "message_count", scanned_count)
        hz = getattr(info, "frequency", None)
        hz_text = "-" if hz is None or math.isnan(float(hz)) else f"{float(hz):.3f}"
        note = "time_range_truncated" if truncated else "ok"
        print(
            f"OK | {topic} | {getattr(info, 'msg_type', '-')} | {count} | "
            f"{hz_text} | {format_optional_float(first_sec)} | "
            f"{format_optional_float(last_sec)} | {note}"
        )


def format_optional_float(value: float | None, precision: int = 6) -> str:
    """统一格式化可空浮点数。"""
    if value is None:
        return "-"
    return f"{value:.{precision}f}"


def decode_image_message(msg: Any) -> np.ndarray:
    """解码 ROS Image 或 CompressedImage，返回 OpenCV BGR 图像。"""
    if hasattr(msg, "format") and hasattr(msg, "data"):
        # sensor_msgs/CompressedImage：直接用 OpenCV 从 JPEG/PNG buffer 解码。
        image_buffer = np.frombuffer(msg.data, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"CompressedImage 解码失败，format={getattr(msg, 'format', '<unknown>')}")
        return image_bgr

    if all(hasattr(msg, name) for name in ("height", "width", "encoding", "data")):
        # sensor_msgs/Image：只支持 Inspector 常见的 rgb8/bgr8/mono8；其他编码先暴露出来给用户确认。
        height = int(msg.height)
        width = int(msg.width)
        encoding = str(msg.encoding).lower()
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        if encoding == "rgb8":
            image_rgb = raw.reshape(height, width, 3)
            return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if encoding == "bgr8":
            return raw.reshape(height, width, 3)
        if encoding == "mono8":
            image_gray = raw.reshape(height, width)
            return cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        raise ValueError(f"暂不支持的 Image encoding: {msg.encoding}")

    raise TypeError("消息既不像 CompressedImage，也不像 Image。")


def inspect_head_camera_image(bag: rosbag.Bag, output_dir: Path) -> None:
    """导出头部相机完整图和左右半图，并打印尺寸与双目启发式判断。"""
    print("\n# 2. Head Camera Image Check")
    msg, sec = first_message_for_topic(bag, IMAGE_TOPIC)
    if msg is None:
        print(f"[ERROR] 找不到图像 topic: {IMAGE_TOPIC}")
        return

    image_bgr = decode_image_message(msg)
    height, width = image_bgr.shape[:2]
    aspect_ratio = width / height if height else float("nan")

    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / "cam_h_full.jpg"
    left_path = output_dir / "cam_h_left_half.jpg"
    right_path = output_dir / "cam_h_right_half.jpg"

    mid_x = width // 2
    cv2.imwrite(str(full_path), image_bgr)
    cv2.imwrite(str(left_path), image_bgr[:, :mid_x])
    cv2.imwrite(str(right_path), image_bgr[:, mid_x:])

    print(f"topic: {IMAGE_TOPIC}")
    print(f"first_frame_time: {format_optional_float(sec)}")
    print(f"height: {height}")
    print(f"width: {width}")
    print(f"aspect_ratio_width_over_height: {aspect_ratio:.6f}")
    print(f"full_image: {full_path.resolve()}")
    print(f"left_half_image: {left_path.resolve()}")
    print(f"right_half_image: {right_path.resolve()}")

    # Gemini-335L 的 RGB 规格常见为 1280x800，宽高比约 1.6。
    # 若图像是左右拼接双目，常见宽高比会接近 3.2，或者宽度明显约等于单目宽度的 2 倍。
    if aspect_ratio >= 2.6:
        guess = "LIKELY_STEREO_SIDE_BY_SIDE"
        reason = "宽高比明显偏宽，接近左右拼接图像特征。"
    elif 1.3 <= aspect_ratio <= 1.9:
        guess = "LIKELY_SINGLE_RGB"
        reason = "宽高比接近 Gemini-335L 单路 RGB 规格 1280x800 的 1.6。"
    else:
        guess = "REVIEW_REQUIRED"
        reason = "宽高比不典型，需要结合导出图片人工判断。"
    print(f"image_layout_guess: {guess}")
    print(f"image_layout_reason: {reason}")


def print_camera_info(bag: rosbag.Bag) -> None:
    """打印相机内参 topic 的宽高和 K/D，辅助判断相机流含义。"""
    print("\n# 3. CameraInfo Check")
    for topic in CAMERA_INFO_TOPICS:
        msg, sec = first_message_for_topic(bag, topic)
        if msg is None:
            print(f"\n## {topic}\nMISSING")
            continue
        print(f"\n## {topic}")
        print(f"first_msg_time: {format_optional_float(sec)}")
        print(f"width: {getattr(msg, 'width', '-')}")
        print(f"height: {getattr(msg, 'height', '-')}")
        print(f"distortion_model: {getattr(msg, 'distortion_model', '-')}")
        print(f"K: {list(getattr(msg, 'K', []))}")
        print(f"D: {list(getattr(msg, 'D', []))}")


def summarize_numeric_matrix(values: list[list[float]]) -> dict[str, np.ndarray]:
    """统计二维数值数组的 mean/std/min/max/range。"""
    matrix = np.asarray(values, dtype=np.float64)
    return {
        "mean": matrix.mean(axis=0),
        "std": matrix.std(axis=0),
        "min": matrix.min(axis=0),
        "max": matrix.max(axis=0),
        "range": matrix.max(axis=0) - matrix.min(axis=0),
    }


def print_head_candidate_summary(
    *,
    name: str,
    values: list[list[float]],
    threshold_rad: float,
) -> None:
    """打印一个头部索引候选切片的稳定性统计。"""
    print(f"\n## {name}")
    if not values:
        print("samples: 0")
        print("decision: FALLBACK_ZERO")
        print("reason: 没有读到足够的 joint_q 数据。")
        return

    stats = summarize_numeric_matrix(values)
    max_range = float(np.max(stats["range"]))
    stable = max_range <= threshold_rad
    mean_deg = np.rad2deg(stats["mean"])
    first_samples = values[:5]

    print(f"samples: {len(values)}")
    print(f"first_5_rad: {first_samples}")
    print(f"mean_rad: {stats['mean'].tolist()}")
    print(f"mean_deg: {mean_deg.tolist()}")
    print(f"min_rad: {stats['min'].tolist()}")
    print(f"max_rad: {stats['max'].tolist()}")
    print(f"range_rad: {stats['range'].tolist()}")
    print(f"std_rad: {stats['std'].tolist()}")
    print(f"stability_threshold_rad: {threshold_rad}")
    print(f"stable_under_threshold: {stable}")
    if stable:
        print("decision: USE_MEASURED_CONSTANT")
        print("reason: 该候选头部切片跨样本稳定，可作为锁定头部的固定 state。")
    else:
        print("decision: FALLBACK_ZERO_OR_REVIEW")
        print("reason: 该候选头部切片跨样本变化超过阈值，阶段一不应直接当作固定锁定角。")


def inspect_head_joint_lock(
    bag: rosbag.Bag,
    *,
    max_samples: int,
    threshold_rad: float,
) -> None:
    """统计头部关节候选切片是否稳定。"""
    print("\n# 4. Head Joint Lock Check")
    print(f"source_topic: {SENSORS_TOPIC}")
    print("candidate_v4x_v49: joint_data.joint_q[26:28] -> head_yaw/head_pitch")
    print("candidate_v52: joint_data.joint_q[27:29] -> 兼容 V52 腰部偏移候选，仅供 review")

    values_26_28: list[list[float]] = []
    values_27_29: list[list[float]] = []
    joint_q_lengths: list[int] = []

    for _, msg, _ in bag.read_messages(topics=[SENSORS_TOPIC]):
        joint_q = get_nested_attr(msg, "joint_data.joint_q")
        if joint_q is None:
            continue
        length = safe_len(joint_q)
        if length is None:
            continue
        joint_q_lengths.append(length)
        if length >= 28:
            values_26_28.append([float(joint_q[26]), float(joint_q[27])])
        if length >= 29:
            values_27_29.append([float(joint_q[27]), float(joint_q[28])])
        if max_samples > 0 and len(values_26_28) >= max_samples:
            break

    if joint_q_lengths:
        unique_lengths = sorted(set(joint_q_lengths))
        print(f"joint_q_lengths_seen: {unique_lengths}")
    else:
        print("joint_q_lengths_seen: []")

    print_head_candidate_summary(
        name="candidate_v4x_v49_joint_q_26_28",
        values=values_26_28,
        threshold_rad=threshold_rad,
    )
    print_head_candidate_summary(
        name="candidate_v52_joint_q_27_29",
        values=values_27_29,
        threshold_rad=threshold_rad,
    )


def check_length(
    msg: Any,
    field_path: str,
    *,
    minimum: int | None = None,
    exact: int | None = None,
) -> str:
    """检查数组字段长度，输出 OK/REVIEW/MISSING。"""
    value = get_nested_attr(msg, field_path)
    if value is None:
        return f"MISSING field={field_path}"
    length = safe_len(value)
    if length is None:
        return f"REVIEW field={field_path} has_no_len type={type(value).__name__}"
    if exact is not None and length != exact:
        return f"REVIEW field={field_path} len={length}, expected_exact={exact}"
    if minimum is not None and length < minimum:
        return f"REVIEW field={field_path} len={length}, expected_min={minimum}"
    return f"OK field={field_path} len={length}"


def inspect_touch_state_shape(msg: Any) -> list[str]:
    """检查 `/dexhand/touch_state` 是否符合双手 5 指 × 3 normal_force 的预期。"""
    lines: list[str] = []
    total_normal_force = 0
    for hand_name in ("left_hand", "right_hand"):
        hand = getattr(msg, hand_name, None)
        if hand is None:
            lines.append(f"MISSING field={hand_name}")
            continue
        hand_len = safe_len(hand)
        lines.append(f"{'OK' if hand_len == 5 else 'REVIEW'} field={hand_name} len={hand_len}, expected_exact=5")
        if hand_len is None:
            continue
        hand_normal_force = 0
        for finger_index, finger in enumerate(list(hand)[:5]):
            missing_fields = []
            for force_name in ("normal_force1", "normal_force2", "normal_force3"):
                if hasattr(finger, force_name):
                    hand_normal_force += 1
                else:
                    missing_fields.append(force_name)
            if missing_fields:
                lines.append(
                    f"REVIEW field={hand_name}[{finger_index}] missing_normal_force={missing_fields}"
                )
        total_normal_force += hand_normal_force
        lines.append(f"{hand_name}_normal_force_count: {hand_normal_force}, expected=15")
    lines.append(f"total_normal_force_count: {total_normal_force}, expected=30")
    return lines


def print_field_checks(bag: rosbag.Bag, *, schema_depth: int) -> None:
    """打印关键 topic 第一条消息的 schema，并做维度检查。"""
    print("\n# 5. Critical Field Length Checks")
    for topic in FIELD_CHECK_TOPICS:
        msg, sec = first_message_for_topic(bag, topic)
        if msg is None:
            print(f"\n## {topic}\nMISSING")
            continue

        print_message_schema(msg, title=f"{topic} first message schema", max_depth=schema_depth)
        print(f"first_msg_time: {format_optional_float(sec)}")

        if topic == SENSORS_TOPIC:
            print(check_length(msg, "joint_data.joint_q", minimum=28))
            joint_q = get_nested_attr(msg, "joint_data.joint_q")
            if joint_q is not None and safe_len(joint_q) is not None:
                print(f"arm_left_candidate_joint_q_12_19: {sample_slice(joint_q, 12, 19)}")
                print(f"arm_right_candidate_joint_q_19_26: {sample_slice(joint_q, 19, 26)}")
                print(f"head_candidate_joint_q_26_28: {sample_slice(joint_q, 26, 28)}")

        elif topic == "/dexhand/state":
            print(check_length(msg, "position", minimum=12))

        elif topic == "/dexhand/touch_state":
            for line in inspect_touch_state_shape(msg):
                print(line)

        elif topic == "/control_robot_hand_position":
            if hasattr(msg, "left_hand_position") or hasattr(msg, "right_hand_position"):
                print(check_length(msg, "left_hand_position", exact=6))
                print(check_length(msg, "right_hand_position", exact=6))
            else:
                print(check_length(msg, "position", minimum=12))

        elif topic == "/kuavo_arm_traj":
            print(check_length(msg, "position", minimum=14))

        elif topic == "/joint_cmd":
            print(check_length(msg, "joint_q", minimum=28))


def sample_slice(value: Iterable[Any], start: int, end: int) -> list[Any] | str:
    """安全打印数组切片样例。"""
    try:
        seq = list(value)
    except TypeError:
        return "<not_sequence>"
    if len(seq) < end:
        return f"<len={len(seq)} shorter_than_{end}>"
    return seq[start:end]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Kuavo rosbag schema for DECO stage 1 decisions."
    )
    parser.add_argument(
        "--bag",
        type=Path,
        default=DEFAULT_BAG_PATH,
        help=f"rosbag 文件路径，默认 {DEFAULT_BAG_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Inspector 图片输出目录，默认 {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--max-topic-scan",
        type=int,
        default=0,
        help="每个 topic 扫描多少条来估计时间范围；0 表示全量扫描。",
    )
    parser.add_argument(
        "--max-head-samples",
        type=int,
        default=10000,
        help="用于头部关节稳定性统计的最大 /sensors_data_raw 样本数；0 表示全量。",
    )
    parser.add_argument(
        "--head-stability-threshold-rad",
        type=float,
        default=1e-3,
        help="判断头部关节锁定稳定性的最大 range 阈值，单位 rad。",
    )
    parser.add_argument(
        "--schema-depth",
        type=int,
        default=2,
        help="第一条消息 schema 的递归打印深度。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bag_path = args.bag
    if not bag_path.exists():
        raise SystemExit(f"找不到 rosbag 文件: {bag_path}")

    print("# DECO Stage 1 Rosbag Schema Inspector")
    print(f"bag: {bag_path.resolve()}")
    print(f"output_dir: {args.output_dir.resolve()}")
    print("mode: read_only_schema_and_image_export")

    bag = open_bag_readonly(bag_path)
    try:
        print_topic_summary(bag, max_topic_scan=args.max_topic_scan)
        inspect_head_camera_image(bag, args.output_dir)
        print_camera_info(bag)
        inspect_head_joint_lock(
            bag,
            max_samples=args.max_head_samples,
            threshold_rad=args.head_stability_threshold_rad,
        )
        print_field_checks(bag, schema_depth=args.schema_depth)
    finally:
        bag.close()

    print("\n# 6. What To Send Back")
    print("请把终端输出全文或至少以下章节反馈给我：")
    print("- `# 2. Head Camera Image Check`")
    print("- `# 3. CameraInfo Check`")
    print("- `# 4. Head Joint Lock Check`")
    print("- `# 5. Critical Field Length Checks` 中 sensors/dexhand/action 相关部分")
    print("同时请反馈这三张图片的人工观察结论：")
    print(f"- {args.output_dir / 'cam_h_full.jpg'}")
    print(f"- {args.output_dir / 'cam_h_left_half.jpg'}")
    print(f"- {args.output_dir / 'cam_h_right_half.jpg'}")


if __name__ == "__main__":
    main()
