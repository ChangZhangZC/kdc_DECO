#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""直接使用单个 Rosbag 对 Kuavo-DECO checkpoint 做 Open Loop Eval。

本脚本以 LeTools/GR00T 的 ``open_loop_eval.py`` 为结构参考，但只使用当前
``kdc_DECO`` 仓库内已有的 Rosbag reader、DECO policy 和 LeRobot processor：

1. 原始 Rosbag 在内存中按 ``KuavoRosbag2Lerobot_deco.yaml`` 解码并对齐；
2. Ground Truth 使用与训练数转完全相同的 18D/28D action 构造规则；
3. checkpoint 与 processor 由 ``outputs/train/task/method/timestamp`` 层级统一定位；
4. 每隔 ``action_horizon`` 帧使用真实观测预测一个 action chunk；
5. 将 Rosbag Ground Truth 与 postprocessor 后的物理 action 绘制在同一坐标轴。

这是离线开环诊断，不会把预测动作发给 ROS、仿真或真机，也不能替代闭环成功率评估。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any


# 直接运行 ``python kuavo_eval/open_loop_eval.py`` 时先补仓库根目录，
# 确保能够导入同级 lerobot_patches、kuavo_data 与 kuavo_train 包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lerobot_patches.custom_patches  # noqa: F401,E402 - 必须先应用 Kuavo LeRobot patch

import hydra  # noqa: E402
from hydra.utils import get_original_cwd  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
import torch  # noqa: E402

from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

from kuavo_data.CvtRosbag2Lerobot_DECO import (  # noqa: E402
    DECO_LEROBOT_TASK,
    GRIPPER_NO_TACTILE_PROFILE,
    QIANGNAO_TACTILE_PROFILE,
    DecoRosbagReader,
    as_float_array,
    build_deco_action,
    build_deco_gripper_action,
    build_deco_gripper_state,
    build_deco_state,
    clamp_deco_arm_action,
    cfg_select,
    initialize_deco_rgb_parameters,
    profile_action_names,
)
from kuavo_train.wrapper.policy.deco import DECOProcessor  # noqa: F401,E402 - 注册自定义 processor step
from kuavo_train.wrapper.policy.deco.DECOConfigWrapper import (  # noqa: E402
    CustomDECOConfigWrapper,
)
from kuavo_train.wrapper.policy.deco.DECOPolicyWrapper import (  # noqa: E402
    CustomDECOPolicyWrapper,
)


LOGGER = logging.getLogger(__name__)
POLICY_CONFIG_FILENAME = "config.json"
POLICY_WEIGHTS_FILENAME = "model.safetensors"
PREPROCESSOR_FILENAME = "policy_preprocessor.json"
POSTPROCESSOR_FILENAME = "policy_postprocessor.json"


def _resolve_path(value: str | Path) -> Path:
    """相对路径始终以 Hydra 启动前的仓库工作目录为基准。"""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(get_original_cwd()) / path
    return path.resolve()


def _checkpoint_label(epoch: Any) -> str | None:
    """把部署配置中的 best/100/epoch100 转换成 checkpoint 目录名。"""

    epoch_text = str(epoch).strip()
    if not epoch_text:
        raise ValueError("inference.checkpoint.epoch 不能为空。")
    if epoch_text in {"latest", "root"}:
        return None
    return epoch_text if epoch_text.startswith("epoch") else f"epoch{epoch_text}"


def resolve_training_assets(cfg: DictConfig) -> tuple[Path, Path, str]:
    """从统一训练目录层级解析 run root、checkpoint 与显示标签。"""

    checkpoint_cfg = cfg.inference.checkpoint
    root = _resolve_path(str(checkpoint_cfg.root))
    task = str(checkpoint_cfg.task).strip()
    method = str(checkpoint_cfg.method).strip()
    timestamp = str(checkpoint_cfg.timestamp).strip()
    if not task or not method or not timestamp:
        raise ValueError("checkpoint.task/method/timestamp 必须是非空字符串。")

    run_root = root / task / method / timestamp
    checkpoint_label = _checkpoint_label(checkpoint_cfg.epoch)
    checkpoint_path = run_root if checkpoint_label is None else run_root / checkpoint_label
    display_label = checkpoint_label or "latest"

    required_paths = {
        "checkpoint config": checkpoint_path / POLICY_CONFIG_FILENAME,
        "checkpoint weights": checkpoint_path / POLICY_WEIGHTS_FILENAME,
        "policy preprocessor": run_root / PREPROCESSOR_FILENAME,
        "policy postprocessor": run_root / POSTPROCESSOR_FILENAME,
    }
    missing = [f"{name}: {path}" for name, path in required_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("DECO Open Loop Eval 缺少训练资产：\n" + "\n".join(missing))
    return run_root, checkpoint_path, display_label


def _validate_eval_config(cfg: DictConfig) -> None:
    """在读取 Rosbag 或加载 GPU 模型前拒绝明显非法的评估配置。"""

    bag_path = _resolve_path(str(cfg.input.rosbag_path))
    if not bag_path.is_file():
        raise FileNotFoundError(f"Open Loop Eval Rosbag 不存在：{bag_path}")
    if bag_path.suffix.lower() != ".bag":
        raise ValueError(f"input.rosbag_path 必须指向单个 .bag 文件：{bag_path}")

    start_frame = cfg.input.start_frame
    if isinstance(start_frame, bool) or not isinstance(start_frame, int) or start_frame < 0:
        raise ValueError("input.start_frame 必须是非负整数。")
    max_steps = cfg.input.max_steps
    if max_steps is not None and (
        isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0
    ):
        raise ValueError("input.max_steps 必须是正整数或 null。")

    configured_horizon = cfg.inference.action_horizon
    if configured_horizon is not None and (
        isinstance(configured_horizon, bool)
        or not isinstance(configured_horizon, int)
        or configured_horizon <= 0
    ):
        raise ValueError("inference.action_horizon 必须是正整数或 null。")
    configured_inf_step = cfg.inference.inf_step
    if configured_inf_step is not None and (
        isinstance(configured_inf_step, bool)
        or not isinstance(configured_inf_step, int)
        or configured_inf_step <= 0
    ):
        raise ValueError("inference.inf_step 必须是正整数或 null。")

    seed = cfg.inference.seed
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("inference.seed 必须是非负整数。")

    dimensions_per_figure = cfg.output.dimensions_per_figure
    if (
        isinstance(dimensions_per_figure, bool)
        or not isinstance(dimensions_per_figure, int)
        or dimensions_per_figure <= 0
    ):
        raise ValueError("output.dimensions_per_figure 必须是正整数。")
    dpi = cfg.output.dpi
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("output.dpi 必须是正整数。")


def _resolve_action_horizon(cfg: DictConfig, policy: CustomDECOPolicyWrapper) -> int:
    """优先使用评估覆盖值，其次 checkpoint.n_action_steps，最后使用完整 chunk。"""

    configured = cfg.inference.action_horizon
    if configured is not None:
        action_horizon = int(configured)
    elif policy.config.n_action_steps is not None:
        action_horizon = int(policy.config.n_action_steps)
    else:
        action_horizon = int(policy.config.chunk_size)

    chunk_size = int(policy.config.chunk_size)
    if action_horizon > chunk_size:
        raise ValueError(
            "Open Loop Eval action_horizon 不能超过 checkpoint chunk_size："
            f"horizon={action_horizon}, chunk_size={chunk_size}"
        )
    return action_horizon


def _configure_conversion_from_checkpoint(
    conversion_cfg: DictConfig,
    policy: CustomDECOPolicyWrapper,
) -> None:
    """把 checkpoint 固定结构注入 Rosbag reader，并保留用户对具体末端类型的选择。"""

    checkpoint_hz = int(policy.config.dataset_hz)
    configured_hz = int(cfg_select(conversion_cfg, "dataset.train_hz", 0))
    if configured_hz != checkpoint_hz:
        raise ValueError(
            "conversion.dataset.train_hz 必须等于 checkpoint.dataset_hz："
            f"conversion={configured_hz}, checkpoint={checkpoint_hz}"
        )

    # profile 与是否需要 tactile 是 checkpoint 模型输入契约，不允许评估 YAML
    # 静默构造另一种 schema。gripper 的 leju/rq2f85 具体类型仍由 eef_type 选择。
    OmegaConf.update(
        conversion_cfg,
        "deco.end_effector_profile",
        str(policy.config.end_effector_profile),
        merge=False,
    )
    OmegaConf.update(
        conversion_cfg,
        "deco.write_tactile",
        bool(policy.config.use_tactile),
        merge=False,
    )


def _validate_policy_reader_compatibility(
    policy: CustomDECOPolicyWrapper,
    reader: DecoRosbagReader,
    bag_metadata: dict[str, Any],
) -> None:
    """校验 checkpoint、Rosbag reader 与对齐结果共享同一 DECO 数据契约。"""

    expected_rgb_keys = tuple(policy.config.rgb_keys)
    actual_rgb_keys = tuple(reader.rgb_keys)
    if actual_rgb_keys != expected_rgb_keys:
        raise ValueError(
            "Rosbag 三视角 key/顺序与 checkpoint 不一致："
            f"reader={actual_rgb_keys}, checkpoint={expected_rgb_keys}"
        )
    if reader.profile != policy.config.end_effector_profile:
        raise ValueError(
            "Rosbag profile 与 checkpoint 不一致："
            f"reader={reader.profile}, checkpoint={policy.config.end_effector_profile}"
        )
    if bool(reader.include_tactile) != bool(policy.config.use_tactile):
        raise ValueError(
            "Rosbag tactile 输入与 checkpoint 不一致："
            f"reader={reader.include_tactile}, checkpoint={policy.config.use_tactile}"
        )

    target_hz = int(bag_metadata.get("target_hz", 0))
    if target_hz != int(policy.config.dataset_hz):
        raise ValueError(
            "Rosbag 对齐频率与 checkpoint 不一致："
            f"rosbag={target_hz}, checkpoint={policy.config.dataset_hz}"
        )
    action_names = profile_action_names(reader.profile)
    if len(action_names) != int(policy.config.action_dim):
        raise ValueError(
            "Rosbag action schema 与 checkpoint.action_dim 不一致："
            f"schema={len(action_names)}, checkpoint={policy.config.action_dim}"
        )


def build_deco_frame_from_aligned_bag(
    bag_data: dict[str, Any],
    frame_idx: int,
    reader: DecoRosbagReader,
    cfg: DictConfig,
) -> dict[str, Any]:
    """在评估侧从已对齐 Rosbag 构造一帧 DECO observation 与 GT action。

    这是数据清洗脚本原有 ``populate_dataset()`` 单帧构造代码的评估侧副本。
    按用户要求，不再修改已经稳定使用的数据清洗入口；这里显式保留相同的
    18D/28D 排列、action source、末端单位转换、头部 action 与机械臂裁剪语义。
    """

    # 以下内容逐项对应数据清洗脚本 populate_dataset() 的单帧构造分支。
    # 这里有意保留本地副本，以保证新增评估不会改变既有数据转换代码。
    is_binary = bool(cfg_select(cfg, "dataset.is_binary", False))
    profile = reader.profile
    arm_action_source = str(bag_data["__metadata__"]["arm_action_source"])

    if profile == QIANGNAO_TACTILE_PROFILE:
        state = build_deco_state(
            bag_data["observation.state"][frame_idx]["data"],
            bag_data["observation.qiangnao"][frame_idx]["data"],
            is_binary=is_binary,
        )
        action = build_deco_action(
            bag_data[arm_action_source][frame_idx],
            arm_action_source,
            bag_data["action.qiangnao"][frame_idx]["data"],
            bag_data["action.joint_cmd"][frame_idx]["data"],
            is_binary=is_binary,
        )
    elif profile == GRIPPER_NO_TACTILE_PROFILE:
        state = build_deco_gripper_state(
            bag_data["observation.state"][frame_idx]["data"],
            bag_data["observation.gripper"][frame_idx]["data"],
            eef_type=reader.eef_type,
            is_binary=is_binary,
        )
        action = build_deco_gripper_action(
            bag_data[arm_action_source][frame_idx],
            arm_action_source,
            bag_data["action.gripper"][frame_idx]["data"],
            bag_data["action.joint_cmd"][frame_idx]["data"],
            eef_type=reader.eef_type,
            is_binary=is_binary,
        )
    else:
        raise ValueError(f"未知 DECO profile：{profile}")

    frame: dict[str, Any] = {
        # 三路键由 reader.rgb_keys 固定排序，必须与 policy.rgb_keys 完全一致。
        rgb_key: bag_data[rgb_key][frame_idx]["data"] for rgb_key in reader.rgb_keys
    }
    frame.update({
        "observation.state": torch.from_numpy(state).float(),
        "action": torch.from_numpy(clamp_deco_arm_action(action, profile)).float(),
        # 当前 DECO 不使用 task conditioning；仍保留 LeRobot 兼容占位值。
        "task": DECO_LEROBOT_TASK,
    })
    if reader.include_tactile:
        tactile = as_float_array(
            "observation.tactile",
            bag_data["observation.tactile_raw"][frame_idx]["data"],
            min_len=30,
        )[:30]
        frame["observation.tactile"] = torch.from_numpy(tactile.astype(np.float32)).float()
    return frame


def _rgb_hwc_uint8_to_chw_float(value: Any, key: str) -> torch.Tensor:
    """复现 LeRobot 视频 loader 的模型输入语义：HWC uint8 -> CHW float [0,1]。"""

    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Rosbag RGB {key} 必须是 HWC 三通道图像，实际为 {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Rosbag RGB {key} 必须是 uint8，实际为 {image.dtype}")
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0


def build_policy_observation(
    frame: dict[str, Any],
    policy: CustomDECOPolicyWrapper,
) -> dict[str, Any]:
    """从 eval 本地构造的 Rosbag frame 中选择 checkpoint 实际消费的 observation。"""

    observation: dict[str, Any] = {
        key: _rgb_hwc_uint8_to_chw_float(frame[key], key)
        for key in policy.config.rgb_keys
    }
    observation["observation.state"] = torch.as_tensor(
        frame["observation.state"], dtype=torch.float32
    ).clone()
    if policy.config.use_tactile:
        tactile_key = str(policy.config.tactile_key)
        if tactile_key not in frame:
            raise ValueError(f"tactile checkpoint 的 Rosbag frame 缺少 {tactile_key}")
        observation[tactile_key] = torch.as_tensor(
            frame[tactile_key], dtype=torch.float32
        ).clone()
    return observation


def _to_action_chunk_numpy(value: Any, *, expected_dim: int) -> np.ndarray:
    """把 postprocessor 输出严格约束成 [chunk_size, action_dim]。"""

    if not isinstance(value, torch.Tensor):
        raise TypeError(
            "DECO postprocessor 必须返回 torch.Tensor，"
            f"实际为 {type(value).__name__}"
        )
    array = value.detach().cpu().float().numpy()
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1] != expected_dim:
        raise ValueError(
            "DECO postprocessor action chunk 必须是 [1,chunk,dim] 或 [chunk,dim]，"
            f"实际为 {array.shape}, expected_dim={expected_dim}"
        )
    if not np.isfinite(array).all():
        raise ValueError("DECO prediction 包含 NaN 或 Inf。")
    return array.astype(np.float32, copy=False)


def evaluate_rosbag(
    *,
    policy: CustomDECOPolicyWrapper,
    preprocessor: Any,
    postprocessor: Any,
    reader: DecoRosbagReader,
    conversion_cfg: DictConfig,
    bag_data: dict[str, Any],
    start_frame: int,
    max_steps: int | None,
    action_horizon: int,
) -> dict[str, Any]:
    """按真实 Rosbag observation 分段预测，并返回可绘图的完整时序。"""

    total_frames = int(bag_data["__metadata__"]["num_frames"])
    if start_frame >= total_frames:
        raise ValueError(
            f"input.start_frame={start_frame} 超出 Rosbag 有效帧数 {total_frames}。"
        )
    stop_frame = total_frames
    if max_steps is not None:
        stop_frame = min(stop_frame, start_frame + max_steps)
    num_steps = stop_frame - start_frame
    if num_steps <= 0:
        raise ValueError("Open Loop Eval 没有可用评估帧。")

    frames = [
        build_deco_frame_from_aligned_bag(bag_data, frame_idx, reader, conversion_cfg)
        for frame_idx in range(start_frame, stop_frame)
    ]
    ground_truth = np.stack(
        [torch.as_tensor(frame["action"]).detach().cpu().float().numpy() for frame in frames],
        axis=0,
    ).astype(np.float32)
    expected_dim = int(policy.config.action_dim)
    if ground_truth.shape != (num_steps, expected_dim):
        raise ValueError(
            "Rosbag Ground Truth shape 与 checkpoint 不一致："
            f"ground_truth={ground_truth.shape}, expected={(num_steps, expected_dim)}"
        )
    if not np.isfinite(ground_truth).all():
        raise ValueError("Rosbag Ground Truth action 包含 NaN 或 Inf。")

    predicted_parts: list[np.ndarray] = []
    inference_frame_indices: list[int] = []
    policy.reset()
    for local_start in range(0, num_steps, action_horizon):
        global_frame_idx = start_frame + local_start
        inference_frame_indices.append(global_frame_idx)
        raw_observation = build_policy_observation(frames[local_start], policy)
        processed_observation = preprocessor(raw_observation)
        with torch.inference_mode():
            normalized_chunk = policy.predict_action_chunk(processed_observation)
        physical_chunk = _to_action_chunk_numpy(
            postprocessor(normalized_chunk),
            expected_dim=expected_dim,
        )

        valid_steps = min(action_horizon, num_steps - local_start)
        if physical_chunk.shape[0] < valid_steps:
            raise ValueError(
                "DECO prediction chunk 短于当前有效 action_horizon："
                f"predicted={physical_chunk.shape[0]}, required={valid_steps}"
            )
        predicted_parts.append(physical_chunk[:valid_steps])

    prediction = np.concatenate(predicted_parts, axis=0)
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            "拼接后的 DECO prediction 与 Ground Truth shape 不一致："
            f"prediction={prediction.shape}, ground_truth={ground_truth.shape}"
        )

    head_key = reader.rgb_keys[0]
    timestamps = np.asarray(
        [bag_data[head_key][idx]["timestamp"] for idx in range(start_frame, stop_frame)],
        dtype=np.float64,
    )
    return {
        "ground_truth": ground_truth,
        "prediction": prediction,
        "timestamps": timestamps,
        "frame_indices": np.arange(start_frame, stop_frame, dtype=np.int64),
        "inference_frame_indices": np.asarray(inference_frame_indices, dtype=np.int64),
    }


def calculate_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    action_names: list[str],
) -> dict[str, Any]:
    """计算物理 action 的总体与逐维 MSE/MAE。"""

    error = prediction - ground_truth
    squared_error = np.square(error)
    absolute_error = np.abs(error)
    per_dimension = []
    for dim_idx, action_name in enumerate(action_names):
        per_dimension.append(
            {
                "index": dim_idx,
                "name": action_name,
                "mse": float(np.mean(squared_error[:, dim_idx])),
                "mae": float(np.mean(absolute_error[:, dim_idx])),
            }
        )
    return {
        "mse": float(np.mean(squared_error)),
        "mae": float(np.mean(absolute_error)),
        "per_dimension": per_dimension,
    }


def plot_action_comparison(
    *,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    frame_indices: np.ndarray,
    inference_frame_indices: np.ndarray,
    action_names: list[str],
    action_horizon: int,
    output_dir: Path,
    paginate_dimensions: bool,
    dimensions_per_figure: int,
    dpi: int,
) -> list[Path]:
    """在每个 subplot 中绘制 Rosbag GT 与 DECO Prediction 两根曲线。"""

    action_dim = ground_truth.shape[1]
    page_size = dimensions_per_figure if paginate_dimensions else action_dim
    saved_paths: list[Path] = []
    for dim_start in range(0, action_dim, page_size):
        dim_stop = min(dim_start + page_size, action_dim)
        figure, axes = plt.subplots(
            nrows=dim_stop - dim_start,
            ncols=1,
            figsize=(12, 3.2 * (dim_stop - dim_start)),
            squeeze=False,
        )
        figure.suptitle(
            "DECO Open Loop Eval - Rosbag Ground Truth vs Prediction "
            f"(action_horizon={action_horizon})",
            fontsize=14,
        )
        for row_idx, dim_idx in enumerate(range(dim_start, dim_stop)):
            axis = axes[row_idx, 0]
            axis.plot(
                frame_indices,
                ground_truth[:, dim_idx],
                label="Rosbag Ground Truth",
                linewidth=1.4,
            )
            axis.plot(
                frame_indices,
                prediction[:, dim_idx],
                label="DECO Prediction",
                linewidth=1.2,
            )
            for marker_idx, inference_frame_idx in enumerate(inference_frame_indices):
                axis.axvline(
                    int(inference_frame_idx),
                    color="red",
                    linestyle=":",
                    alpha=0.35,
                    label="Inference point" if marker_idx == 0 else None,
                )
            axis.set_title(f"Action {dim_idx}: {action_names[dim_idx]}")
            axis.set_xlabel("Aligned Rosbag frame")
            axis.set_ylabel("Physical action")
            axis.grid(alpha=0.2)
            axis.legend(loc="upper right")

        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
        output_path = output_dir / f"action_dims_{dim_start:02d}_{dim_stop - 1:02d}.png"
        figure.savefig(output_path, dpi=dpi)
        plt.close(figure)
        saved_paths.append(output_path)
    return saved_paths


def _build_output_directory(
    cfg: DictConfig,
    checkpoint_label: str,
    bag_path: Path,
) -> Path:
    checkpoint_cfg = cfg.inference.checkpoint
    output_dir = (
        _resolve_path(str(cfg.output.root))
        / str(checkpoint_cfg.task)
        / str(checkpoint_cfg.method)
        / str(checkpoint_cfg.timestamp)
        / checkpoint_label
        / bag_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="eval/deco_open_loop_eval",
)
def main(cfg: DictConfig) -> None:
    """Open Loop Eval Hydra 入口。"""

    logging.basicConfig(level=logging.INFO)
    _validate_eval_config(cfg)
    run_root, checkpoint_path, checkpoint_label = resolve_training_assets(cfg)
    bag_path = _resolve_path(str(cfg.input.rosbag_path))

    device = torch.device(str(cfg.inference.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("inference.device 请求 CUDA，但当前 PyTorch 未检测到可用 CUDA。")
    set_seed(int(cfg.inference.seed))

    # 必须在读取 safetensors 前覆盖 checkpoint 中保存的训练设备；否则当 checkpoint
    # 记录 cuda、评估配置选择 cpu 时，LeRobot 会先按旧设备映射权重并提前失败。
    policy_config = CustomDECOConfigWrapper.from_pretrained(checkpoint_path)
    policy_config.device = str(device)
    policy = CustomDECOPolicyWrapper.from_pretrained(
        checkpoint_path,
        config=policy_config,
        strict=True,
    )
    if cfg.inference.inf_step is not None:
        policy.config.inf_step = int(cfg.inference.inf_step)
        policy.model.inference_step = int(cfg.inference.inf_step)
    action_horizon = _resolve_action_horizon(cfg, policy)

    # 保存的 processor statistics 位于 run root；只覆盖设备，不重建或替换统计量。
    preprocessor, postprocessor = make_pre_post_processors(
        None,
        run_root,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    _configure_conversion_from_checkpoint(cfg.conversion, policy)
    initialize_deco_rgb_parameters(cfg.conversion)
    reader = DecoRosbagReader(cfg.conversion)
    bag_data = reader.process_rosbag(str(bag_path))
    bag_metadata = bag_data["__metadata__"]
    _validate_policy_reader_compatibility(policy, reader, bag_metadata)

    result = evaluate_rosbag(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        reader=reader,
        conversion_cfg=cfg.conversion,
        bag_data=bag_data,
        start_frame=int(cfg.input.start_frame),
        max_steps=None if cfg.input.max_steps is None else int(cfg.input.max_steps),
        action_horizon=action_horizon,
    )
    action_names = profile_action_names(reader.profile)
    metrics = calculate_metrics(
        result["ground_truth"],
        result["prediction"],
        action_names,
    )

    output_dir = _build_output_directory(cfg, checkpoint_label, bag_path)
    plot_paths: list[Path] = []
    if bool(cfg.output.save_plot):
        plot_paths = plot_action_comparison(
            ground_truth=result["ground_truth"],
            prediction=result["prediction"],
            frame_indices=result["frame_indices"],
            inference_frame_indices=result["inference_frame_indices"],
            action_names=action_names,
            action_horizon=action_horizon,
            output_dir=output_dir,
            paginate_dimensions=bool(cfg.output.paginate_dimensions),
            dimensions_per_figure=int(cfg.output.dimensions_per_figure),
            dpi=int(cfg.output.dpi),
        )

    if bool(cfg.output.save_predictions):
        np.savez_compressed(
            output_dir / "predictions.npz",
            timestamps=result["timestamps"],
            frame_indices=result["frame_indices"],
            inference_frame_indices=result["inference_frame_indices"],
            ground_truth_actions=result["ground_truth"],
            predicted_actions=result["prediction"],
            action_names=np.asarray(action_names),
        )

    summary = {
        "rosbag_path": str(bag_path),
        "run_root": str(run_root),
        "checkpoint_path": str(checkpoint_path),
        "processor_root": str(run_root),
        "output_directory": str(output_dir),
        "device": str(device),
        "seed": int(cfg.inference.seed),
        "end_effector_profile": reader.profile,
        "eef_type": reader.eef_type,
        "use_tactile": bool(policy.config.use_tactile),
        "dataset_hz": int(policy.config.dataset_hz),
        "chunk_size": int(policy.config.chunk_size),
        "action_horizon": action_horizon,
        "inf_step": int(policy.model.inference_step),
        "action_dim": int(policy.config.action_dim),
        "start_frame": int(result["frame_indices"][0]),
        "num_steps": int(result["ground_truth"].shape[0]),
        "arm_action_source": str(bag_metadata["arm_action_source"]),
        "metrics": metrics,
        "plots": [str(path) for path in plot_paths],
    }
    if bool(cfg.output.save_summary):
        with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

    LOGGER.info(
        "DECO Open Loop Eval 完成：steps=%d, MSE=%.8f, MAE=%.8f, output=%s",
        summary["num_steps"],
        metrics["mse"],
        metrics["mae"],
        output_dir,
    )


if __name__ == "__main__":
    main()
