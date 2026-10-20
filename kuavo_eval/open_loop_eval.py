#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct Rosbag open-loop evaluation for Kuavo-DECO.

Configuration ownership is intentionally split into three layers:
1. data.config_path: historical Rosbag conversion/GT contract;
2. checkpoint + saved processors: model architecture and normalization contract;
3. runtime: evaluation-only overrides such as inf_step and action_horizon.

The evaluator works fully offline. It never publishes ROS control commands.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lerobot_patches.custom_patches  # noqa: F401,E402

import hydra  # noqa: E402
from hydra.utils import get_original_cwd  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
import torch  # noqa: E402

from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

from kuavo_data.CvtRosbag2Lerobot_DECO import (  # noqa: E402
    GRIPPER_NO_TACTILE_PROFILE,
    QIANGNAO_TACTILE_PROFILE,
    DecoRosbagReader,
    as_float_array,
    build_deco_action,
    build_deco_gripper_action,
    build_deco_gripper_state,
    build_deco_state,
    clamp_deco_arm_action,
    compute_head_episode_mean,
    cfg_select,
    initialize_deco_rgb_parameters,
    profile_action_names,
)
from kuavo_train.wrapper.policy.deco import DECOProcessor  # noqa: F401,E402
from kuavo_train.wrapper.policy.deco.DECOPolicyWrapper import CustomDECOPolicyWrapper  # noqa: E402

LOGGER = logging.getLogger(__name__)
POLICY_CONFIG_FILENAME = "config.json"
POLICY_WEIGHTS_FILENAME = "model.safetensors"
PREPROCESSOR_FILENAME = "policy_preprocessor.json"
POSTPROCESSOR_FILENAME = "policy_postprocessor.json"


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(get_original_cwd()) / path
    return path.resolve()


def _checkpoint_label(epoch: Any) -> str | None:
    text = str(epoch).strip()
    if not text:
        raise ValueError("checkpoint.epoch 不能为空。")
    if text in {"latest", "root"}:
        return None
    return text if text.startswith("epoch") else f"epoch{text}"


def resolve_training_assets(cfg: DictConfig) -> tuple[Path, Path, str]:
    checkpoint_cfg = cfg.checkpoint
    root = _resolve_path(str(checkpoint_cfg.root))
    task = str(checkpoint_cfg.task).strip()
    method = str(checkpoint_cfg.method).strip()
    timestamp = str(checkpoint_cfg.timestamp).strip()
    if not task or not method or not timestamp:
        raise ValueError("checkpoint.task/method/timestamp 必须是非空字符串。")

    run_root = root / task / method / timestamp
    label = _checkpoint_label(checkpoint_cfg.epoch)
    checkpoint_path = run_root if label is None else run_root / label
    display_label = label or "latest"
    required = {
        "checkpoint config": checkpoint_path / POLICY_CONFIG_FILENAME,
        "checkpoint weights": checkpoint_path / POLICY_WEIGHTS_FILENAME,
        "policy preprocessor": run_root / PREPROCESSOR_FILENAME,
        "policy postprocessor": run_root / POSTPROCESSOR_FILENAME,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("DECO Open Loop Eval 缺少训练资产：\n" + "\n".join(missing))
    return run_root, checkpoint_path, display_label


def load_data_conversion_config(cfg: DictConfig) -> tuple[Path, DictConfig]:
    config_path = _resolve_path(str(cfg.data.config_path))
    if not config_path.is_file():
        raise FileNotFoundError(f"data.config_path 不存在：{config_path}")
    loaded = OmegaConf.load(config_path)
    if not isinstance(loaded, DictConfig):
        raise TypeError(f"data.config_path 必须加载为 YAML DictConfig：{config_path}")
    if OmegaConf.select(loaded, "dataset") is None or OmegaConf.select(loaded, "deco") is None:
        raise ValueError("历史数转 YAML 必须包含 dataset 和 deco 两个配置块。")
    return config_path, loaded


def _validate_eval_config(cfg: DictConfig) -> None:
    bag_path = _resolve_path(str(cfg.input.rosbag_path))
    if not bag_path.is_file() or bag_path.suffix.lower() != ".bag":
        raise ValueError(f"input.rosbag_path 必须指向存在的 .bag 文件：{bag_path}")

    start_frame = cfg.input.start_frame
    if isinstance(start_frame, bool) or not isinstance(start_frame, int) or start_frame < 0:
        raise ValueError("input.start_frame 必须是非负整数。")

    max_steps = cfg.input.max_steps
    if max_steps is not None and (
        isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0
    ):
        raise ValueError("input.max_steps 必须是正整数或 null。")

    for key in ("action_horizon", "inf_step"):
        value = cfg.runtime[key]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"runtime.{key} 必须是正整数或 null。")

    seed = cfg.runtime.seed
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("runtime.seed 必须是非负整数。")

    dimensions_per_figure = cfg.output.dimensions_per_figure
    dpi = cfg.output.dpi
    if (
        isinstance(dimensions_per_figure, bool)
        or not isinstance(dimensions_per_figure, int)
        or dimensions_per_figure <= 0
    ):
        raise ValueError("output.dimensions_per_figure 必须是正整数。")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("output.dpi 必须是正整数。")


def _resolve_action_horizon(cfg: DictConfig, policy: CustomDECOPolicyWrapper) -> int:
    if cfg.runtime.action_horizon is not None:
        horizon = int(cfg.runtime.action_horizon)
    elif policy.config.n_action_steps is not None:
        horizon = int(policy.config.n_action_steps)
    else:
        horizon = int(policy.config.chunk_size)

    chunk_size = int(policy.config.chunk_size)
    if horizon > chunk_size:
        raise ValueError(f"action_horizon={horizon} 不能超过 checkpoint.chunk_size={chunk_size}。")
    return horizon


def _validate_data_checkpoint_contract(
    *,
    data_cfg: DictConfig,
    policy: CustomDECOPolicyWrapper,
    reader: DecoRosbagReader,
) -> None:
    data_hz = int(cfg_select(data_cfg, "dataset.train_hz", 0))
    checkpoint_hz = int(policy.config.dataset_hz)
    if data_hz != checkpoint_hz:
        raise ValueError(
            "历史数转 YAML 与 checkpoint 的时间语义不一致："
            f"data.dataset.train_hz={data_hz}, checkpoint.dataset_hz={checkpoint_hz}"
        )

    data_rgb = tuple(reader.rgb_keys)
    checkpoint_rgb = tuple(policy.config.rgb_keys)
    if data_rgb != checkpoint_rgb:
        raise ValueError(
            "历史数转 YAML 与 checkpoint 的 RGB key/顺序不一致："
            f"data={data_rgb}, checkpoint={checkpoint_rgb}"
        )

    if reader.profile != policy.config.end_effector_profile:
        raise ValueError(
            "历史数转 YAML 与 checkpoint 的 end-effector profile 不一致："
            f"data={reader.profile}, checkpoint={policy.config.end_effector_profile}"
        )

    data_action_dim = len(profile_action_names(reader.profile))
    if data_action_dim != int(policy.config.action_dim):
        raise ValueError(
            "历史数转 YAML 与 checkpoint 的 action_dim 不一致："
            f"data={data_action_dim}, checkpoint={policy.config.action_dim}"
        )

    if bool(policy.config.use_tactile) and not bool(reader.include_tactile):
        raise ValueError(
            "checkpoint.use_tactile=true，但历史数转 YAML 未启用 tactile；"
            "无法复现该 checkpoint 的输入契约。"
        )
    if bool(reader.include_tactile) and not bool(policy.config.use_tactile):
        LOGGER.info("历史数据包含 tactile，但该 checkpoint 不消费 tactile；保留历史数转契约。")


def _validate_aligned_metadata(
    policy: CustomDECOPolicyWrapper,
    reader: DecoRosbagReader,
    metadata: dict[str, Any],
) -> None:
    if int(metadata.get("target_hz", 0)) != int(policy.config.dataset_hz):
        raise ValueError("Rosbag 实际对齐频率与 checkpoint.dataset_hz 不一致。")
    if str(metadata.get("end_effector_profile", "")) != str(reader.profile):
        raise ValueError("Rosbag aligned metadata 的 profile 与历史数转配置不一致。")


def _contract_snapshot(
    *,
    data_config_path: Path,
    data_cfg: DictConfig,
    policy: CustomDECOPolicyWrapper,
    cfg: DictConfig,
    action_horizon: int,
    checkpoint_inf_step: int,
) -> dict[str, Any]:
    return {
        "data": {
            "config_path": str(data_config_path),
            "eef_type": str(cfg_select(data_cfg, "dataset.eef_type", "")),
            "train_hz": int(cfg_select(data_cfg, "dataset.train_hz", 0)),
            "sample_drop": int(cfg_select(data_cfg, "dataset.sample_drop", 0)),
            "is_binary": bool(cfg_select(data_cfg, "dataset.is_binary", False)),
            "resize": {
                "width": int(cfg_select(data_cfg, "dataset.resize.width", 0)),
                "height": int(cfg_select(data_cfg, "dataset.resize.height", 0)),
            },
            "rgb_keys": list(cfg_select(data_cfg, "deco.rgb_keys", [])),
            "arm_action_priority": list(cfg_select(data_cfg, "deco.arm_action_priority", [])),
            "head_action_fill": list(cfg_select(data_cfg, "deco.head_action_fill", [])),
            "write_tactile": bool(cfg_select(data_cfg, "deco.write_tactile", False)),
        },
        "checkpoint": {
            "end_effector_profile": str(policy.config.end_effector_profile),
            "action_dim": int(policy.config.action_dim),
            "rgb_keys": list(policy.config.rgb_keys),
            "dataset_hz": int(policy.config.dataset_hz),
            "chunk_size": int(policy.config.chunk_size),
            "n_action_steps": None if policy.config.n_action_steps is None else int(policy.config.n_action_steps),
            "inf_step": int(checkpoint_inf_step),
            "use_tactile": bool(policy.config.use_tactile),
            "resize_shape": list(policy.config.resize_shape),
        },
        "runtime": {
            "device": str(cfg.runtime.device),
            "seed": int(cfg.runtime.seed),
            "inf_step_override": None if cfg.runtime.inf_step is None else int(cfg.runtime.inf_step),
            "effective_inf_step": int(policy.model.inference_step),
            "action_horizon_override": (
                None if cfg.runtime.action_horizon is None else int(cfg.runtime.action_horizon)
            ),
            "effective_action_horizon": int(action_horizon),
        },
    }


def _log_contract_report(snapshot: dict[str, Any]) -> None:
    data = snapshot["data"]
    checkpoint = snapshot["checkpoint"]
    runtime = snapshot["runtime"]
    lines = [
        "================ DECO OFFLINE EVAL CONTRACT ================",
        f"Data config           : {data['config_path']}",
        f"Data eef_type         : {data['eef_type']}",
        f"Data train_hz         : {data['train_hz']}",
        f"Data sample_drop      : {data['sample_drop']}",
        f"Data is_binary        : {data['is_binary']}",
        f"Data head_action_fill : {data['head_action_fill']}",
        f"Checkpoint profile    : {checkpoint['end_effector_profile']}",
        f"Checkpoint action_dim : {checkpoint['action_dim']}",
        f"Checkpoint RGB keys   : {checkpoint['rgb_keys']}",
        f"Checkpoint dataset_hz : {checkpoint['dataset_hz']}",
        f"Checkpoint chunk_size : {checkpoint['chunk_size']}",
        f"Checkpoint n_actions  : {checkpoint['n_action_steps']}",
        f"Checkpoint inf_step   : {checkpoint['inf_step']}",
        f"Saved tactile input   : {checkpoint['use_tactile']}",
        f"Runtime inf_step      : {runtime['effective_inf_step']} "
        f"(override={runtime['inf_step_override']})",
        f"Runtime horizon       : {runtime['effective_action_horizon']} "
        f"(override={runtime['action_horizon_override']})",
        "Compatibility         : PASS",
        "============================================================",
    ]
    LOGGER.info("\n%s", "\n".join(lines))


def validate_head_contract(
    reader: DecoRosbagReader,
    head_mean: np.ndarray,
) -> np.ndarray:
    head_mean = np.asarray(head_mean, dtype=np.float32).reshape(-1)
    if head_mean.shape != (2,) or not np.isfinite(head_mean).all():
        raise ValueError(f"head_mean 必须是有限 2D 向量，实际={head_mean}")

    head_action_fill = as_float_array(
        "head_action_fill",
        reader.head_action_fill,
        min_len=2,
    )[:2]
    if not np.isfinite(head_action_fill).all():
        raise ValueError(f"head_action_fill 包含 NaN/Inf：{head_action_fill}")
    return head_action_fill.astype(np.float32)


def build_deco_frame_from_aligned_bag(
    bag_data: dict[str, Any],
    frame_idx: int,
    reader: DecoRosbagReader,
    cfg: DictConfig,
    head_mean: np.ndarray,
) -> dict[str, Any]:
    is_binary = bool(cfg_select(cfg, "dataset.is_binary", False))
    profile = reader.profile
    arm_action_source = str(bag_data["__metadata__"]["arm_action_source"])

    if profile == QIANGNAO_TACTILE_PROFILE:
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
    elif profile == GRIPPER_NO_TACTILE_PROFILE:
        state = build_deco_gripper_state(
            bag_data["observation.state"][frame_idx]["data"],
            bag_data["observation.gripper"][frame_idx]["data"],
            head_mean,
            eef_type=reader.eef_type,
            is_binary=is_binary,
        )
        action = build_deco_gripper_action(
            bag_data[arm_action_source][frame_idx],
            arm_action_source,
            bag_data["action.gripper"][frame_idx]["data"],
            reader.head_action_fill,
            eef_type=reader.eef_type,
            is_binary=is_binary,
        )
    else:
        raise ValueError(f"未知 DECO profile：{profile}")

    frame: dict[str, Any] = {
        rgb_key: bag_data[rgb_key][frame_idx]["data"]
        for rgb_key in reader.rgb_keys
    }
    frame.update(
        {
            "observation.state": torch.from_numpy(state).float(),
            "action": torch.from_numpy(clamp_deco_arm_action(action, profile)).float(),
        }
    )
    if reader.include_tactile:
        tactile = as_float_array(
            "observation.tactile",
            bag_data["observation.tactile_raw"][frame_idx]["data"],
            min_len=30,
        )[:30]
        frame["observation.tactile"] = torch.from_numpy(
            tactile.astype(np.float32)
        ).float()
    return frame


def _rgb_hwc_uint8_to_chw_float(value: Any, key: str) -> torch.Tensor:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError(
            f"Rosbag RGB {key} 必须是 HWC uint8 三通道，"
            f"实际 shape={image.shape}, dtype={image.dtype}"
        )
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0


def build_policy_observation(
    frame: dict[str, Any],
    policy: CustomDECOPolicyWrapper,
) -> dict[str, Any]:
    observation = {
        key: _rgb_hwc_uint8_to_chw_float(frame[key], key)
        for key in policy.config.rgb_keys
    }
    observation["observation.state"] = torch.as_tensor(
        frame["observation.state"],
        dtype=torch.float32,
    ).clone()

    if policy.config.use_tactile:
        tactile_key = str(policy.config.tactile_key)
        if tactile_key not in frame:
            raise ValueError(f"tactile checkpoint 的 Rosbag frame 缺少 {tactile_key}")
        observation[tactile_key] = torch.as_tensor(
            frame[tactile_key],
            dtype=torch.float32,
        ).clone()
    return observation


def _to_action_chunk_numpy(value: Any, expected_dim: int) -> np.ndarray:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            "DECO postprocessor 必须返回 torch.Tensor，"
            f"实际={type(value).__name__}"
        )

    array = value.detach().cpu().float().numpy()
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1] != expected_dim:
        raise ValueError(
            "postprocessed action chunk shape 非法："
            f"{array.shape}, expected_dim={expected_dim}"
        )
    if not np.isfinite(array).all():
        raise ValueError("DECO prediction 包含 NaN/Inf。")
    return array.astype(np.float32, copy=False)


def evaluate_rosbag(
    *,
    policy: CustomDECOPolicyWrapper,
    preprocessor: Any,
    postprocessor: Any,
    reader: DecoRosbagReader,
    data_cfg: DictConfig,
    bag_data: dict[str, Any],
    start_frame: int,
    max_steps: int | None,
    action_horizon: int,
) -> dict[str, Any]:
    total_frames = int(bag_data["__metadata__"]["num_frames"])
    if start_frame >= total_frames:
        raise ValueError(f"start_frame={start_frame} 超出有效帧数 {total_frames}。")

    head_mean = compute_head_episode_mean(bag_data["observation.state"])
    head_action_fill = validate_head_contract(reader, head_mean)

    stop_frame = total_frames if max_steps is None else min(total_frames, start_frame + max_steps)
    num_steps = stop_frame - start_frame
    if num_steps <= 0:
        raise ValueError("Open Loop Eval 没有可用评估帧。")

    frames = [
        build_deco_frame_from_aligned_bag(
            bag_data,
            frame_idx,
            reader,
            data_cfg,
            head_mean,
        )
        for frame_idx in range(start_frame, stop_frame)
    ]

    ground_truth = np.stack(
        [torch.as_tensor(frame["action"]).cpu().float().numpy() for frame in frames],
        axis=0,
    ).astype(np.float32)

    expected_dim = int(policy.config.action_dim)
    if ground_truth.shape != (num_steps, expected_dim):
        raise ValueError(
            "Ground Truth action shape 非法："
            f"{ground_truth.shape}, expected={(num_steps, expected_dim)}"
        )
    if not np.isfinite(ground_truth).all():
        raise ValueError("Ground Truth action 包含 NaN/Inf。")

    predicted_parts: list[np.ndarray] = []
    inference_frame_indices: list[int] = []
    policy.reset()

    for local_start in range(0, num_steps, action_horizon):
        inference_frame_indices.append(start_frame + local_start)
        raw_observation = build_policy_observation(frames[local_start], policy)
        processed_observation = preprocessor(raw_observation)

        with torch.inference_mode():
            normalized_chunk = policy.predict_action_chunk(processed_observation)

        physical_chunk = _to_action_chunk_numpy(
            postprocessor(normalized_chunk),
            expected_dim,
        )

        valid_steps = min(action_horizon, num_steps - local_start)
        if physical_chunk.shape[0] < valid_steps:
            raise ValueError(
                "prediction chunk 太短："
                f"predicted={physical_chunk.shape[0]}, required={valid_steps}"
            )
        predicted_parts.append(physical_chunk[:valid_steps])

    prediction = np.concatenate(predicted_parts, axis=0)
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"prediction={prediction.shape} 与 ground_truth={ground_truth.shape} 不一致。"
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
        "head_mean": head_mean.astype(np.float32),
        "head_action_fill": head_action_fill,
    }


def action_group_indices(profile: str) -> dict[str, list[int]]:
    if profile == QIANGNAO_TACTILE_PROFILE:
        return {
            "arm": [*range(0, 7), *range(13, 20)],
            "eef": [*range(7, 13), *range(20, 26)],
            "head": [26, 27],
            "deploy_executed": list(range(26)),
        }
    if profile == GRIPPER_NO_TACTILE_PROFILE:
        return {
            "arm": [*range(0, 7), *range(8, 15)],
            "eef": [7, 15],
            "head": [16, 17],
            "deploy_executed": list(range(16)),
        }
    raise ValueError(f"未知 DECO profile：{profile}")


def _metric_block(error: np.ndarray) -> dict[str, float]:
    return {
        "mse": float(np.mean(np.square(error))),
        "mae": float(np.mean(np.abs(error))),
    }


def calculate_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    action_names: list[str],
    profile: str,
) -> dict[str, Any]:
    error = prediction - ground_truth
    groups = action_group_indices(profile)

    per_dimension = [
        {
            "index": index,
            "name": name,
            "mse": float(np.mean(np.square(error[:, index]))),
            "mae": float(np.mean(np.abs(error[:, index]))),
        }
        for index, name in enumerate(action_names)
    ]

    return {
        "overall": _metric_block(error),
        "deploy_executed": _metric_block(error[:, groups["deploy_executed"]]),
        "groups": {
            name: _metric_block(error[:, indices])
            for name, indices in groups.items()
            if name != "deploy_executed"
        },
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
            f"DECO Open Loop Eval - Rosbag GT vs Prediction (action_horizon={action_horizon})",
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
            for marker_idx, inference_idx in enumerate(inference_frame_indices):
                axis.axvline(
                    int(inference_idx),
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
        path = output_dir / f"action_dims_{dim_start:02d}_{dim_stop - 1:02d}.png"
        figure.savefig(path, dpi=dpi)
        plt.close(figure)
        saved_paths.append(path)

    return saved_paths


def _build_output_directory(
    cfg: DictConfig,
    checkpoint_label: str,
    bag_path: Path,
) -> Path:
    checkpoint_cfg = cfg.checkpoint
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
    logging.basicConfig(level=logging.INFO)
    _validate_eval_config(cfg)

    data_config_path, data_cfg = load_data_conversion_config(cfg)
    run_root, checkpoint_path, checkpoint_label = resolve_training_assets(cfg)
    bag_path = _resolve_path(str(cfg.input.rosbag_path))

    device = torch.device(str(cfg.runtime.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device 请求 CUDA，但当前 PyTorch 未检测到可用 CUDA。")
    set_seed(int(cfg.runtime.seed))

    policy = CustomDECOPolicyWrapper.from_pretrained(checkpoint_path, strict=True)
    policy.eval()
    policy.to(device)
    policy.reset()

    checkpoint_inf_step = int(policy.model.inference_step)
    if cfg.runtime.inf_step is not None:
        policy.config.inf_step = int(cfg.runtime.inf_step)
        policy.model.inference_step = int(cfg.runtime.inf_step)

    action_horizon = _resolve_action_horizon(cfg, policy)
    preprocessor, postprocessor = make_pre_post_processors(None, run_root)

    initialize_deco_rgb_parameters(data_cfg)
    reader = DecoRosbagReader(data_cfg)
    _validate_data_checkpoint_contract(
        data_cfg=data_cfg,
        policy=policy,
        reader=reader,
    )

    contract = _contract_snapshot(
        data_config_path=data_config_path,
        data_cfg=data_cfg,
        policy=policy,
        cfg=cfg,
        action_horizon=action_horizon,
        checkpoint_inf_step=checkpoint_inf_step,
    )
    _log_contract_report(contract)

    bag_data = reader.process_rosbag(str(bag_path))
    metadata = bag_data["__metadata__"]
    _validate_aligned_metadata(policy, reader, metadata)

    result = evaluate_rosbag(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        reader=reader,
        data_cfg=data_cfg,
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
        reader.profile,
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
            head_observation_mean=result["head_mean"],
            head_action_fill=result["head_action_fill"],
        )

    summary = {
        "rosbag_path": str(bag_path),
        "data_config_path": str(data_config_path),
        "run_root": str(run_root),
        "checkpoint_path": str(checkpoint_path),
        "processor_root": str(run_root),
        "output_directory": str(output_dir),
        "contract": contract,
        "end_effector_profile": reader.profile,
        "eef_type": reader.eef_type,
        "start_frame": int(result["frame_indices"][0]),
        "num_steps": int(result["ground_truth"].shape[0]),
        "arm_action_source": str(metadata["arm_action_source"]),
        "head_contract": {
            "observation_source": "episode_mean_joint_q_26_28",
            "observation_value": result["head_mean"].tolist(),
            "action_source": "data.config_path -> deco.head_action_fill",
            "action_fill": result["head_action_fill"].tolist(),
            "model_head_action_executed_in_stable_deploy": False,
        },
        "metrics": metrics,
        "plots": [str(path) for path in plot_paths],
    }

    if bool(cfg.output.save_summary):
        with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

    LOGGER.info(
        "DECO Open Loop Eval 完成：steps=%d, executed_MSE=%.8f, executed_MAE=%.8f, "
        "overall_MSE=%.8f, output=%s",
        summary["num_steps"],
        metrics["deploy_executed"]["mse"],
        metrics["deploy_executed"]["mae"],
        metrics["overall"]["mse"],
        output_dir,
    )


if __name__ == "__main__":
    main()
