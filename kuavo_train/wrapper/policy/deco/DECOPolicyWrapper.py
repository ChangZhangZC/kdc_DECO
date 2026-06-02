from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors.torch import load_file as load_safetensors_file
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from kuavo_train.wrapper.policy.deco import ensure_deco_on_path
from kuavo_train.wrapper.policy.deco.DECOConfigWrapper import CustomDECOConfigWrapper

ensure_deco_on_path()
from models.deco.deco import DECO  # noqa: E402


LOGGER = logging.getLogger(__name__)


class CustomDECOPolicyWrapper(PreTrainedPolicy):
    """Kuavo-DECO 的 LeRobot policy wrapper。

    wrapper 负责 LeRobot batch 解包、tactile max normalization、两阶段权重加载
    与 action queue；`third_party/deco` 中的 DECO 类只保留模型主体。
    """

    config_class = CustomDECOConfigWrapper
    name = "custom_deco"

    def __init__(self, config: CustomDECOConfigWrapper):
        super().__init__(config)
        config.validate_features()
        self.model = DECO(
            act_dim=config.action_dim,
            chunk_size=config.chunk_size,
            obs_state=config.obs_state,
            use_tactile=config.use_tactile,
            plugin=config.use_tactile_lora,
            plugin_rank=config.tactile_lora_rank,
            use_task_condition=config.use_task_condition,
            num_tasks=config.num_tasks,
            inf_step=config.inf_step,
            num_attn_blocks=config.num_attn_blocks,
            heads=config.heads,
            dim=config.dim,
            rope_axes_dim=config.rope_axes_dim,
            vision_backbone=config.vision_backbone,
            depth_backbone=config.depth_backbone,
            visual_fusion_mode=config.visual_fusion_mode,
        )
        self._action_queue: deque[Tensor] = deque()
        self._weight_load_reports: list[dict[str, int | str]] = []
        self._load_configured_weights()
        if self._should_freeze_main_for_tactile_adapter():
            self._freeze_main_for_tactile_adapter()

    def _load_configured_weights(self) -> None:
        if not self.config.load_external_init_weights:
            return
        freeze_loaded_init = (
            self.config.training_stage == "tactile_adapter" and self.config.freeze_pretrained_main
        )
        if self.config.deco_init_pth_path:
            self._load_torch_checkpoint(
                self.config.deco_init_pth_path,
                freeze_loaded=freeze_loaded_init,
                require_main_match=self._should_freeze_main_for_tactile_adapter(),
            )
        if self.config.base_policy_path:
            self._load_safetensors_policy(
                self.config.base_policy_path,
                freeze_loaded=freeze_loaded_init,
                require_main_match=self._should_freeze_main_for_tactile_adapter(),
            )
        if self.config.adapter_model_path:
            self._load_safetensors_policy(
                self.config.adapter_model_path,
                freeze_loaded=False,
                require_main_match=self._should_freeze_main_for_tactile_adapter(),
            )

    def _load_safetensors_policy(
        self,
        path: str | Path,
        *,
        freeze_loaded: bool,
        require_main_match: bool,
    ) -> None:
        model_path = Path(path)
        if model_path.is_dir():
            model_path = model_path / SAFETENSORS_SINGLE_FILE
        state_dict = load_safetensors_file(str(model_path), device="cpu")
        self._load_matching_tensors(
            state_dict,
            freeze_loaded=freeze_loaded,
            require_main_match=require_main_match,
            source=str(model_path),
        )

    def _load_torch_checkpoint(
        self,
        path: str | Path,
        *,
        freeze_loaded: bool,
        require_main_match: bool,
    ) -> None:
        checkpoint = self._safe_load_torch_checkpoint(path)
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise ValueError(f"DECO checkpoint must be a state_dict-like mapping: {path}")
        self._load_matching_tensors(
            checkpoint,
            freeze_loaded=freeze_loaded,
            require_main_match=require_main_match,
            source=str(path),
        )

    def _safe_load_torch_checkpoint(self, path: str | Path) -> Any:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise RuntimeError(
                "DECO .pth loading requires torch.load(weights_only=True). "
                "请优先使用 `.safetensors`；若必须导入 `.pth`，请升级到支持 "
                "weights_only=True 的 PyTorch 版本，并仅使用可信本地历史权重。"
            ) from exc

    def _load_matching_tensors(
        self,
        state_dict: dict[str, Any],
        *,
        freeze_loaded: bool,
        require_main_match: bool,
        source: str,
    ) -> None:
        target_state = self.state_dict()
        matched: dict[str, Tensor] = {}
        skipped_non_tensor = 0
        skipped_missing_key = 0
        skipped_shape = 0

        for raw_key, value in state_dict.items():
            if not torch.is_tensor(value):
                skipped_non_tensor += 1
                continue
            key_exists = False
            loaded = False
            for key in self._candidate_state_keys(raw_key):
                if key not in target_state:
                    continue
                key_exists = True
                if target_state[key].shape != value.shape:
                    continue
                matched[key] = value
                loaded = True
                break
            if loaded:
                continue
            if key_exists:
                skipped_shape += 1
            else:
                skipped_missing_key += 1

        parameter_names = {name for name, _ in self.named_parameters()}
        main_matched = sum(
            1
            for key in matched
            if key in parameter_names and not self._is_tactile_adapter_parameter(key)
        )
        # 记录权重匹配统计，便于第二阶段确认被冻结主干确实来自 checkpoint。
        report: dict[str, int | str] = {
            "source": source,
            "matched": len(matched),
            "main_matched": main_matched,
            "target_missing": len(set(target_state) - set(matched)),
            "skipped_non_tensor": skipped_non_tensor,
            "skipped_missing_key": skipped_missing_key,
            "skipped_shape": skipped_shape,
        }
        self._weight_load_reports.append(report)

        if not matched:
            raise ValueError(
                "No compatible tensors were loaded from DECO checkpoint "
                f"{source}. skipped_missing_key={skipped_missing_key}, "
                f"skipped_shape={skipped_shape}, skipped_non_tensor={skipped_non_tensor}"
            )
        if require_main_match and main_matched == 0:
            raise ValueError(
                "tactile_adapter with freeze_pretrained_main=True requires at least one "
                "non-adapter main-branch tensor to match before freezing the main model. "
                f"source={source}, matched={len(matched)}, main_matched={main_matched}, "
                f"skipped_missing_key={skipped_missing_key}, skipped_shape={skipped_shape}"
            )

        LOGGER.info(
            "Loaded DECO checkpoint tensors from %s: matched=%d, main_matched=%d, "
            "target_missing=%d, skipped_missing_key=%d, skipped_shape=%d, skipped_non_tensor=%d",
            source,
            len(matched),
            main_matched,
            report["target_missing"],
            skipped_missing_key,
            skipped_shape,
            skipped_non_tensor,
        )
        self.load_state_dict(matched, strict=False)
        if freeze_loaded:
            loaded_names = set(matched)
            for name, parameter in self.named_parameters():
                if name in loaded_names:
                    parameter.requires_grad = False

    def _candidate_state_keys(self, raw_key: str) -> list[str]:
        key = raw_key[len("module."):] if raw_key.startswith("module.") else raw_key
        candidates = [key]
        if key.startswith("model."):
            candidates.append(key[len("model."):])
        else:
            candidates.append(f"model.{key}")
        return candidates

    def _should_freeze_main_for_tactile_adapter(self) -> bool:
        return self.config.training_stage == "tactile_adapter" and self.config.freeze_pretrained_main

    def _freeze_main_for_tactile_adapter(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad = self._is_tactile_adapter_parameter(name)

    def _is_tactile_adapter_parameter(self, name: str) -> bool:
        return (
            name.startswith("model.tactile_encoder")
            or name.startswith("model.gated")
            or name.startswith("model.pos_tac_embedd")
            or ".tactile_key" in name
            or ".tactile_value" in name
            or "_pi." in name
        )

    def reset(self) -> None:
        self._action_queue.clear()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        rgb, depth, state, action, task_idx = self._unpack_batch(batch, require_action=True)
        tac1, tac2 = self._prepare_tactile(batch)
        out, noise = self.model(
            rgb,
            depth,
            obs=state,
            act=action,
            task_idx=task_idx,
            tac1=tac1,
            tac2=tac2,
            training=True,
        )
        loss = F.mse_loss(out, noise - action)
        return loss, {"loss": float(loss.detach().cpu())}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        rgb, depth, state, _, task_idx = self._unpack_batch(batch, require_action=False)
        tac1, tac2 = self._prepare_tactile(batch)
        return self.model(
            rgb,
            depth,
            obs=state,
            act=None,
            task_idx=task_idx,
            tac1=tac1,
            tac2=tac2,
            training=False,
        )

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        if len(self._action_queue) == 0:
            action_chunk = self.predict_action_chunk(batch, **kwargs)
            # DECO chunk 是 30Hz 语义动作；部署 10Hz 时按 action_stride 降频进入执行队列。
            strided_actions = action_chunk[0, :: self.config.action_stride]
            self._action_queue.extend(strided_actions)
        return self._action_queue.popleft()

    def _unpack_batch(
        self,
        batch: dict[str, Tensor],
        *,
        require_action: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        rgb = self._ensure_batched_tensor(batch[self.config.rgb_key], image=True)
        depth = self._ensure_batched_tensor(batch[self.config.depth_key], image=True)
        state = self._ensure_batched_tensor(batch[OBS_STATE], image=False)

        action = None
        if require_action:
            action = batch[ACTION]
            if action.ndim == 2:
                action = action.unsqueeze(0)

        task_idx = None
        if self.config.use_task_condition:
            task_idx = batch.get("task_index")
            if task_idx is None:
                task_idx = torch.zeros(rgb.shape[0], dtype=torch.long, device=rgb.device)
            elif task_idx.ndim == 0:
                task_idx = task_idx.unsqueeze(0)
        return rgb, depth, state, action, task_idx

    def _ensure_batched_tensor(self, value: Tensor, *, image: bool) -> Tensor:
        tensor = value
        if image and tensor.ndim == 3:
            return tensor.unsqueeze(0)
        if not image and tensor.ndim == 1:
            return tensor.unsqueeze(0)
        return tensor

    def _prepare_tactile(self, batch: dict[str, Tensor]) -> tuple[Tensor | None, Tensor | None]:
        if not self.config.use_tactile:
            return None, None
        tactile = batch.get(self.config.tactile_key)
        if tactile is None:
            raise ValueError(f"use_tactile=True requires batch key {self.config.tactile_key}")
        if tactile.ndim == 1:
            tactile = tactile.unsqueeze(0)
        if tactile.shape[-1] != 30:
            raise ValueError(f"observation.tactile must be 30D, got {tuple(tactile.shape)}")

        tac1 = tactile[..., :15] / float(self.config.tactile_left_max)
        tac2 = tactile[..., 15:] / float(self.config.tactile_right_max)
        if self.config.clip_tactile_to_unit:
            tac1 = tac1.clamp(0.0, 1.0)
            tac2 = tac2.clamp(0.0, 1.0)
        return tac1, tac2

    def get_optim_params(self) -> list[dict[str, Any]]:
        return [{"params": [p for p in self.parameters() if p.requires_grad], "lr": self.config.optimizer_lr}]
