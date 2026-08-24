"""Online prediction-only updates for the feature-level adapter."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from rl.adapter_prediction_heads import (
    DEFAULT_REWARD_TARGET_SPECS,
    ProjectedSixViewFpnSemanticMaskHead,
    RewardPredictionHead,
    RewardPredictionLoss,
    RoachSemanticMaskLoss,
    build_reward_prediction_targets,
    default_roach_channel_loss_weights,
)
from rl.roach_bev_target import render_roach_bev_target


def _module_grad_norm(module: Optional[nn.Module]) -> float:
    if module is None:
        return 0.0
    sq_sum = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        sq_sum += float(torch.sum(grad * grad).cpu().item())
    return float(math.sqrt(max(sq_sum, 0.0)))


def _detach_feature_groups(feature_groups: Sequence[Sequence[torch.Tensor]]) -> Tuple[Tuple[torch.Tensor, ...], ...]:
    return tuple(tuple(tensor.detach() for tensor in group) for group in feature_groups)


def _tensor_to_float_metrics(metrics: Mapping[str, object]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            if value.numel() == 1:
                result[key] = float(value.detach().cpu().item())
            continue
        try:
            result[key] = float(value)
        except (TypeError, ValueError):
            continue
    return result


class AdapterPredictionAgentMixin:
    """Mixin implementing prediction-only adapter updates.

    The action forward stores detached base FPN features.  The prediction step
    consumes that cache once, reruns the adapter with a new autograd graph, and
    updates only the feature adapter and prediction heads.
    """

    def _init_adapter_prediction(self) -> None:
        self._adapter_prediction_reward_head: Optional[nn.Module] = None
        self._adapter_prediction_semantic_head: Optional[nn.Module] = None
        self._adapter_prediction_reward_loss_fn: Optional[nn.Module] = None
        self._adapter_prediction_semantic_loss_fn: Optional[nn.Module] = None
        self._adapter_prediction_optimizer: Optional[torch.optim.Optimizer] = None
        self._adapter_prediction_forward_cache: Optional[Dict[str, object]] = None
        self._adapter_prediction_debug_export_count = 0

        if not bool(getattr(self.config, "adapter_prediction_enabled", False)):
            return
        update_mode = str(getattr(self.config, "adapter_prediction_update_mode", "prediction_only") or "")
        if update_mode != "prediction_only":
            raise ValueError("adapter_prediction_update_mode must be 'prediction_only' for this stage")
        if self._feature_dcnv4_adapter is None:
            return

        train_reward = bool(getattr(self.config, "adapter_prediction_train_reward", False))
        train_semantic = bool(getattr(self.config, "adapter_prediction_train_semantic", False))
        if not train_reward and not train_semantic:
            return

        feature_dim = int(getattr(self.config, "feature_adapter_feature_dim", 256))
        ego_dim = int(getattr(self.config, "feature_adapter_ego_state_dim", self.config.state_dim))
        levels = tuple(int(level) for level in getattr(self.config, "feature_adapter_levels", (0, 1, 2, 3)))
        hidden_dim = int(getattr(self.config, "hidden_dim", 256))
        action_dim = int(getattr(self.config, "adapter_prediction_action_dim", 9))
        visual_state_dim = feature_dim * len(levels) + ego_dim

        if train_reward:
            self._adapter_prediction_reward_head = RewardPredictionHead(
                visual_state_dim=visual_state_dim,
                action_dim=action_dim,
                output_dim=len(DEFAULT_REWARD_TARGET_SPECS),
                hidden_dim=hidden_dim,
                action_hidden_dim=int(getattr(self.config, "adapter_prediction_action_hidden_dim", 64)),
                dropout=float(getattr(self.config, "adapter_prediction_dropout", 0.0)),
            ).to(self.device)
            self._adapter_prediction_reward_loss_fn = RewardPredictionLoss(
                huber_delta=float(getattr(self.config, "adapter_prediction_reward_huber_delta", 1.0))
            ).to(self.device)

        if train_semantic:
            semantic_hidden_dim = int(getattr(self.config, "adapter_prediction_semantic_hidden_dim", 128))
            self._adapter_prediction_semantic_head = ProjectedSixViewFpnSemanticMaskHead(
                feature_dim=feature_dim,
                levels=levels,
                hidden_dim=semantic_hidden_dim,
                bev_width=int(getattr(self.config, "adapter_prediction_bev_width", 192)),
                pixels_per_meter=float(getattr(self.config, "adapter_prediction_pixels_per_meter", 5.0)),
                pixels_ev_to_bottom=int(getattr(self.config, "adapter_prediction_pixels_ev_to_bottom", 40)),
                image_width=int(getattr(self.config, "adapter_prediction_image_width", 1600)),
                image_height=int(getattr(self.config, "adapter_prediction_image_height", 900)),
            ).to(self.device)
            channel_weights = default_roach_channel_loss_weights(
                self._adapter_prediction_semantic_head.channel_names,
                route_weight=float(getattr(self.config, "adapter_prediction_semantic_route_weight", 0.0)),
                road_weight=float(getattr(self.config, "adapter_prediction_semantic_road_weight", 1.0)),
                lane_weight=float(getattr(self.config, "adapter_prediction_semantic_lane_weight", 1.0)),
                latest_vehicle_weight=float(
                    getattr(self.config, "adapter_prediction_semantic_latest_vehicle_weight", 1.0)
                ),
                latest_walker_weight=float(
                    getattr(self.config, "adapter_prediction_semantic_latest_walker_weight", 1.0)
                ),
                latest_traffic_light_stop_weight=float(
                    getattr(self.config, "adapter_prediction_semantic_latest_tl_stop_weight", 1.0)
                ),
            )
            self._adapter_prediction_semantic_loss_fn = RoachSemanticMaskLoss(
                channel_names=self._adapter_prediction_semantic_head.channel_names,
                channel_loss_weights=channel_weights,
                bce_weight=float(getattr(self.config, "adapter_prediction_semantic_bce_weight", 1.0)),
                dice_weight=float(getattr(self.config, "adapter_prediction_semantic_dice_weight", 0.5)),
                positive_weight=float(getattr(self.config, "adapter_prediction_semantic_positive_weight", 2.0)),
            ).to(self.device)

        for param in self._feature_dcnv4_adapter.parameters():
            param.requires_grad = True

        adapter_params = [param for param in self._feature_dcnv4_adapter.parameters() if param.requires_grad]
        param_groups: List[Dict[str, object]] = []
        if adapter_params:
            param_groups.append(
                {
                    "params": adapter_params,
                    "lr": float(getattr(self.config, "adapter_prediction_lr", 3e-5)),
                }
            )
        if self._adapter_prediction_reward_head is not None:
            param_groups.append(
                {
                    "params": list(self._adapter_prediction_reward_head.parameters()),
                    "lr": float(getattr(self.config, "prediction_head_lr", 1e-4)),
                }
            )
        if self._adapter_prediction_semantic_head is not None:
            param_groups.append(
                {
                    "params": list(self._adapter_prediction_semantic_head.parameters()),
                    "lr": float(getattr(self.config, "prediction_head_lr", 1e-4)),
                }
            )
        if param_groups:
            self._adapter_prediction_optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=float(getattr(self.config, "adapter_prediction_weight_decay", 1e-4)),
            )

    @property
    def adapter_prediction_enabled(self) -> bool:
        return self._adapter_prediction_optimizer is not None

    def clear_adapter_prediction_forward_cache(self) -> None:
        self._adapter_prediction_forward_cache = None

    def cache_adapter_prediction_forward_base(self, feature_maps, observation: Optional[Mapping]) -> None:
        if not self.adapter_prediction_enabled:
            return
        if not bool(getattr(self.config, "adapter_prediction_reuse_forward_cache", True)):
            return
        if self._feature_dcnv4_adapter is None or observation is None:
            return

        base_groups = self._feature_maps_format(feature_maps, inverse=True)
        state_tensor = torch.from_numpy(self._state_np(observation)).unsqueeze(0).to(self.device)
        ego_state = self._ego_state_from_state_tensor(
            state_tensor,
            ego_dim=int(getattr(self.config, "feature_adapter_ego_state_dim", self.config.state_dim)),
        )
        sensor_frame = observation.get("sensor_frame")
        try:
            sensor_frame_int = int(sensor_frame)
        except (TypeError, ValueError):
            sensor_frame_int = -1
        self._adapter_prediction_forward_cache = {
            "base_groups": _detach_feature_groups(base_groups),
            "ego_state": ego_state.detach(),
            "sensor_frame": sensor_frame_int,
        }

    def build_adapter_prediction_action_vector(self, policy_output, control_action: object) -> np.ndarray:
        action_dim = int(getattr(self.config, "adapter_prediction_action_dim", 9))
        action = np.asarray(control_action, dtype=np.float32).reshape(-1)
        steer = float(action[0]) if action.size >= 1 else 0.0
        throttle_brake = float(action[1]) if action.size >= 2 else 0.0
        throttle = max(throttle_brake, 0.0)
        brake = max(-throttle_brake, 0.0)

        trajectory_tensor = getattr(policy_output, "selected_trajectory", None)
        if torch.is_tensor(trajectory_tensor):
            trajectory = trajectory_tensor.detach().float().cpu().numpy().reshape(-1, 2)
        else:
            trajectory = np.zeros((0, 2), dtype=np.float32)
        if trajectory.size:
            endpoint = trajectory[-1]
            path_points = np.concatenate([np.zeros((1, 2), dtype=np.float32), trajectory.astype(np.float32)], axis=0)
            deltas = np.diff(path_points, axis=0)
            path_length = float(np.linalg.norm(deltas, axis=1).sum())
            if len(deltas) >= 2:
                heading0 = math.atan2(float(deltas[0, 1]), float(deltas[0, 0]))
                heading1 = math.atan2(float(deltas[-1, 1]), float(deltas[-1, 0]))
                heading_delta = math.atan2(math.sin(heading1 - heading0), math.cos(heading1 - heading0))
            elif len(deltas) == 1:
                heading_delta = math.atan2(float(deltas[0, 1]), float(deltas[0, 0]))
            else:
                heading_delta = 0.0
            endpoint_x = float(endpoint[0])
            endpoint_y = float(endpoint[1])
        else:
            endpoint_x = 0.0
            endpoint_y = 0.0
            path_length = 0.0
            heading_delta = 0.0

        selected_prob = float(getattr(policy_output, "selected_prob", 0.0) or 0.0)
        entropy_tensor = getattr(policy_output, "entropy", None)
        if torch.is_tensor(entropy_tensor):
            entropy = float(entropy_tensor.detach().float().mean().cpu().item())
        else:
            entropy = 0.0

        vec = np.asarray(
            [
                steer,
                throttle,
                brake,
                endpoint_x,
                endpoint_y,
                path_length,
                heading_delta,
                selected_prob,
                entropy,
            ],
            dtype=np.float32,
        )
        if vec.shape[0] < action_dim:
            vec = np.pad(vec, (0, action_dim - vec.shape[0]), mode="constant")
        return vec[:action_dim].astype(np.float32, copy=False)

    def _pool_adapter_prediction_visual_state(
        self,
        base_groups: Sequence[Sequence[torch.Tensor]],
        adapted_groups: Sequence[Sequence[torch.Tensor]],
        ego_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        levels = tuple(int(level) for level in getattr(self.config, "feature_adapter_levels", (0, 1, 2, 3)))
        pooled_by_group: List[torch.Tensor] = []
        residual_losses: List[torch.Tensor] = []
        for base_group, adapted_group in zip(base_groups, adapted_groups):
            pooled_levels: List[torch.Tensor] = []
            for level in levels:
                base = base_group[level].detach()
                adapted = adapted_group[level]
                if adapted.dim() != 5:
                    raise ValueError(f"adapted L{level} must be [B,Ncam,C,H,W], got {tuple(adapted.shape)}")
                pooled_levels.append(adapted.float().mean(dim=(1, 3, 4)))
                residual_losses.append((adapted.float() - base.float()).pow(2).mean())
            pooled_by_group.append(torch.cat(pooled_levels, dim=-1))
        pooled = torch.stack(pooled_by_group, dim=0).mean(dim=0)
        visual_state = torch.cat([pooled, ego_state.float()], dim=-1)
        if residual_losses:
            residual_loss = torch.stack(residual_losses).mean()
        else:
            residual_loss = torch.zeros((), device=visual_state.device)
        return visual_state, residual_loss

    def _adapter_prediction_semantic_target_status(
        self,
        semantic_target: Optional[Mapping[str, object]],
        expected_frame: int,
    ) -> Tuple[bool, Optional[torch.Tensor], Dict[str, float]]:
        metrics = {
            "adapter_pred_semantic_skip_no_target": 0.0,
            "adapter_pred_semantic_skip_frame_mismatch": 0.0,
            "adapter_pred_semantic_skip_sensor_not_exact": 0.0,
            "adapter_pred_semantic_valid_rate": 0.0,
            "semantic_target_available_rate": 0.0,
            "semantic_target_frame_mismatch_count": 0.0,
        }
        if semantic_target is None:
            metrics["adapter_pred_semantic_skip_no_target"] = 1.0
            return False, None, metrics
        masks = semantic_target.get("masks")
        if masks is not None:
            metrics["semantic_target_available_rate"] = 1.0
        error = str(semantic_target.get("error", "") or "")
        if not bool(semantic_target.get("sensor_frame_exact", True)):
            metrics["adapter_pred_semantic_skip_sensor_not_exact"] = 1.0
            return False, None, metrics
        try:
            frame = int(semantic_target.get("frame", -1))
        except (TypeError, ValueError):
            frame = -1
        if frame != int(expected_frame) or error == "frame_mismatch":
            metrics["adapter_pred_semantic_skip_frame_mismatch"] = 1.0
            metrics["semantic_target_frame_mismatch_count"] = 1.0
            return False, None, metrics
        if masks is None or error:
            metrics["adapter_pred_semantic_skip_no_target"] = 1.0
            return False, None, metrics
        target = torch.as_tensor(masks, device=self.device)
        if target.dim() == 3:
            target = target.unsqueeze(0)
        metrics["adapter_pred_semantic_valid_rate"] = 1.0
        return True, target, metrics

    def _adapter_prediction_debug_dir(self, total_step: int) -> Optional[Path]:
        debug_dir = str(getattr(self.config, "roach_bev_target_debug_dir", "") or "").strip()
        interval = int(getattr(self.config, "roach_bev_target_debug_interval", 0) or 0)
        max_frames = int(getattr(self.config, "roach_bev_target_debug_max_frames", 100) or 100)
        if not debug_dir or interval <= 0:
            return None
        if int(total_step) % interval != 0:
            return None
        if self._adapter_prediction_debug_export_count >= max(0, max_frames):
            return None
        out_dir = Path(debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    @staticmethod
    def _write_rgb_png(path: Path, image: np.ndarray) -> None:
        import cv2

        image = np.asarray(image)
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        cv2.imwrite(str(path), image[:, :, ::-1])

    @staticmethod
    def _render_prediction_probabilities(probabilities: np.ndarray) -> np.ndarray:
        pred_masks = (np.asarray(probabilities) >= 0.5).astype(np.uint8) * 255
        return render_roach_bev_target(pred_masks)

    @staticmethod
    def _render_active_overlay(
        *,
        probabilities: np.ndarray,
        target: np.ndarray,
        active_indices: Sequence[int],
        visibility: Optional[np.ndarray],
    ) -> np.ndarray:
        if not active_indices:
            active_indices = tuple(range(min(probabilities.shape[0], target.shape[0])))
        pred_active = (probabilities[list(active_indices)] >= 0.5).any(axis=0)
        target_active = (target[list(active_indices)] > 0).any(axis=0)
        overlay = np.zeros((target.shape[-2], target.shape[-1], 3), dtype=np.uint8)
        overlay[target_active] = (255, 64, 64)
        overlay[pred_active] = (64, 220, 64)
        overlay[target_active & pred_active] = (255, 220, 64)
        if visibility is not None:
            visible = np.asarray(visibility).astype(bool)
            overlay[~visible] = (32, 32, 32)
        return overlay

    def _maybe_export_adapter_prediction_debug(
        self,
        *,
        total_step: int,
        expected_frame: int,
        semantic_frame: int,
        semantic_logits: Optional[torch.Tensor],
        semantic_target: Optional[torch.Tensor],
        visibility_mask: Optional[torch.Tensor],
        metrics: Mapping[str, float],
    ) -> None:
        out_dir = self._adapter_prediction_debug_dir(total_step)
        if out_dir is None:
            return

        metric_path = out_dir / "adapter_prediction_metrics.jsonl"
        metric_payload = {
            "total_step": int(total_step),
            "expected_frame": int(expected_frame),
            "semantic_frame": int(semantic_frame),
            "metrics": {
                str(key): float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            },
        }
        with metric_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric_payload, sort_keys=True) + "\n")

        if semantic_logits is None or semantic_target is None:
            self._adapter_prediction_debug_export_count += 1
            return

        with torch.no_grad():
            probs = torch.sigmoid(semantic_logits.detach().float()[0]).cpu().numpy()
            target = semantic_target.detach().float()[0].cpu().numpy()
            if (float(target.max()) if target.size else 0.0) > 1.0:
                target = np.clip(target / 255.0, 0.0, 1.0)
            visibility_np = None
            if visibility_mask is not None:
                vis = visibility_mask.detach().float()
                if vis.dim() == 4:
                    vis = vis[0, 0]
                elif vis.dim() == 3:
                    vis = vis[0]
                visibility_np = (vis.cpu().numpy() > 0.5)

        weights = None
        names: Tuple[str, ...] = ()
        if self._adapter_prediction_semantic_head is not None:
            names = tuple(self._adapter_prediction_semantic_head.channel_names)
        if self._adapter_prediction_semantic_loss_fn is not None:
            weights = self._adapter_prediction_semantic_loss_fn.channel_loss_weights.detach().cpu().view(-1).numpy()
        active_indices = [idx for idx, weight in enumerate(weights if weights is not None else []) if float(weight) > 0.0]

        target_uint8 = (np.clip(target, 0.0, 1.0) * 255.0).astype(np.uint8)
        self._write_rgb_png(
            out_dir / f"adapter_pred_target_frame_{int(expected_frame):08d}_step_{int(total_step):08d}.png",
            render_roach_bev_target(target_uint8),
        )
        self._write_rgb_png(
            out_dir / f"adapter_pred_sigmoid_frame_{int(expected_frame):08d}_step_{int(total_step):08d}.png",
            self._render_prediction_probabilities(probs),
        )
        if visibility_np is not None:
            self._write_rgb_png(
                out_dir / f"adapter_pred_projector_valid_frame_{int(expected_frame):08d}_step_{int(total_step):08d}.png",
                (visibility_np.astype(np.uint8) * 255),
            )
        self._write_rgb_png(
            out_dir / f"adapter_pred_overlay_frame_{int(expected_frame):08d}_step_{int(total_step):08d}.png",
            self._render_active_overlay(
                probabilities=probs,
                target=target_uint8,
                active_indices=active_indices,
                visibility=visibility_np,
            ),
        )
        if names:
            channel_payload = {
                "total_step": int(total_step),
                "expected_frame": int(expected_frame),
                "semantic_frame": int(semantic_frame),
                "channel_names": list(names),
                "channel_loss_weights": [
                    float(weights[idx]) if weights is not None and idx < len(weights) else 0.0
                    for idx in range(len(names))
                ],
            }
            with (out_dir / "adapter_prediction_channels.json").open("w", encoding="utf-8") as handle:
                json.dump(channel_payload, handle, indent=2, sort_keys=True)
        self._adapter_prediction_debug_export_count += 1

    def update_adapter_prediction_from_step(
        self,
        *,
        reward: float,
        reward_info: Optional[Mapping[str, object]],
        semantic_target: Optional[Mapping[str, object]],
        action_summary: Optional[object],
        total_step: int,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            "adapter_prediction_enabled": float(self.adapter_prediction_enabled),
            "adapter_prediction_skipped": 1.0,
            "adapter_prediction_reward_loss": 0.0,
            "adapter_prediction_semantic_loss": 0.0,
            "adapter_prediction_total_loss": 0.0,
            "adapter_prediction_residual_loss": 0.0,
            "adapter_prediction_grad_norm_adapter": 0.0,
            "adapter_prediction_grad_norm_reward_head": 0.0,
            "adapter_prediction_grad_norm_semantic_head": 0.0,
        }
        for level in tuple(int(level) for level in getattr(self.config, "feature_adapter_levels", ())):
            metrics[f"feature_adapter_alpha_grad_L{level}"] = 0.0
        try:
            expected_frame = -1
            semantic_frame = -1
            semantic_logits_for_debug: Optional[torch.Tensor] = None
            semantic_target_for_debug: Optional[torch.Tensor] = None
            visibility_mask_for_debug: Optional[torch.Tensor] = None
            if not self.adapter_prediction_enabled:
                return metrics
            every_n = max(1, int(getattr(self.config, "adapter_prediction_every_n_steps", 1)))
            if int(total_step) % every_n != 0:
                return metrics
            cache = self._adapter_prediction_forward_cache
            if cache is None:
                metrics["adapter_prediction_skip_no_cache"] = 1.0
                return metrics
            if self._feature_dcnv4_adapter is None or self._adapter_prediction_optimizer is None:
                return metrics

            base_groups = cache["base_groups"]
            ego_state = cache["ego_state"]
            expected_frame = int(cache.get("sensor_frame", -1))
            if semantic_target is not None:
                try:
                    semantic_frame = int(semantic_target.get("frame", -1))
                except (TypeError, ValueError):
                    semantic_frame = -1
            semantic_valid, semantic_tensor, semantic_status = self._adapter_prediction_semantic_target_status(
                semantic_target,
                expected_frame,
            )
            metrics.update(semantic_status)

            train_reward = self._adapter_prediction_reward_head is not None
            train_semantic = self._adapter_prediction_semantic_head is not None and semantic_valid
            if not train_reward and not train_semantic:
                return metrics

            self._model.eval()
            self._feature_dcnv4_adapter.train()
            if self._adapter_prediction_reward_head is not None:
                self._adapter_prediction_reward_head.train()
            if self._adapter_prediction_semantic_head is not None:
                self._adapter_prediction_semantic_head.train()

            adapted_groups, adapter_metrics = self._feature_dcnv4_adapter(
                base_groups,
                ego_state,
                return_metrics=True,
            )
            visual_state, residual_loss = self._pool_adapter_prediction_visual_state(
                base_groups,
                adapted_groups,
                ego_state,
            )
            residual_weight = float(getattr(self.config, "adapter_prediction_residual_weight", 1e-3))
            total_loss = residual_loss * residual_weight
            for key, value in adapter_metrics.items():
                float_value = float(value)
                metrics[key] = float_value
                metrics[key.replace("feature_adapter_", "adapter_feature_")] = float_value
            metrics["adapter_prediction_residual_loss"] = float(residual_loss.detach().cpu().item())
            metrics["adapter_prediction_residual_weight"] = float(residual_weight)

            if train_reward:
                action_dim = int(getattr(self.config, "adapter_prediction_action_dim", 9))
                if action_summary is None:
                    action_tensor = torch.zeros((visual_state.shape[0], action_dim), device=self.device)
                else:
                    action_tensor = torch.as_tensor(action_summary, device=self.device, dtype=torch.float32)
                    if action_tensor.dim() == 1:
                        action_tensor = action_tensor.unsqueeze(0)
                cache["action_summary"] = action_tensor.detach()
                reward_targets = build_reward_prediction_targets(
                    reward=float(reward),
                    reward_info=reward_info,
                    device=self.device,
                )
                reward_pred = self._adapter_prediction_reward_head(visual_state, action_tensor)
                reward_loss, reward_metrics = self._adapter_prediction_reward_loss_fn(
                    reward_pred,
                    reward_targets,
                    prefix="reward_pred",
                )
                total_loss = total_loss + float(
                    getattr(self.config, "adapter_prediction_reward_weight", 1.0)
                ) * reward_loss
                metrics["adapter_prediction_reward_loss"] = float(reward_loss.detach().cpu().item())
                metrics["reward_pred_loss_scalar"] = float(reward_metrics["reward_pred_loss"].detach().cpu().item())
                metrics.update(_tensor_to_float_metrics(reward_metrics))

            if train_semantic and semantic_tensor is not None:
                semantic_output = self._adapter_prediction_semantic_head(adapted_groups, return_aux=True)
                semantic_logits = semantic_output["logits"]
                visibility_mask = semantic_output.get("visibility_mask")
                semantic_loss, semantic_metrics = self._adapter_prediction_semantic_loss_fn(
                    semantic_logits,
                    semantic_tensor,
                    visibility_mask=visibility_mask,
                    prefix="adapter_prediction_semantic",
                )
                total_loss = total_loss + float(
                    getattr(self.config, "adapter_prediction_semantic_weight", 1.0)
                ) * semantic_loss
                metrics["adapter_prediction_semantic_loss"] = float(semantic_loss.detach().cpu().item())
                metrics.update(_tensor_to_float_metrics(semantic_output.get("metrics", {})))
                metrics.update(_tensor_to_float_metrics(semantic_metrics))
                semantic_logits_for_debug = semantic_logits.detach()
                semantic_target_for_debug = semantic_tensor.detach()
                visibility_mask_for_debug = None if visibility_mask is None else visibility_mask.detach()

            if not bool(torch.isfinite(total_loss.detach()).all().item()):
                metrics["adapter_prediction_nonfinite_loss"] = 1.0
                raise FloatingPointError(
                    "non-finite adapter prediction loss: "
                    f"total={float(total_loss.detach().float().cpu().item())} "
                    f"reward={metrics.get('adapter_prediction_reward_loss', 0.0)} "
                    f"semantic={metrics.get('adapter_prediction_semantic_loss', 0.0)} "
                    f"residual={metrics.get('adapter_prediction_residual_loss', 0.0)}"
                )

            self._adapter_prediction_optimizer.zero_grad()
            total_loss.backward()
            metrics["adapter_prediction_grad_norm_adapter"] = _module_grad_norm(self._feature_dcnv4_adapter)
            metrics["adapter_prediction_grad_norm_reward_head"] = _module_grad_norm(
                self._adapter_prediction_reward_head
            )
            metrics["adapter_prediction_grad_norm_semantic_head"] = _module_grad_norm(
                self._adapter_prediction_semantic_head
            )
            alpha_by_level = getattr(self._feature_dcnv4_adapter, "residual_alpha_by_level", None)
            if alpha_by_level is not None:
                for level in tuple(int(level) for level in getattr(self.config, "feature_adapter_levels", ())):
                    level_key = str(level)
                    if level_key not in alpha_by_level:
                        continue
                    alpha_grad = alpha_by_level[level_key].grad
                    if alpha_grad is not None:
                        metrics[f"feature_adapter_alpha_grad_L{level}"] = float(
                            alpha_grad.detach().float().cpu().item()
                        )
            params = []
            for group in self._adapter_prediction_optimizer.param_groups:
                params.extend(group["params"])
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                params,
                float(getattr(self.config, "adapter_prediction_max_grad_norm", 1.0)),
            )
            total_grad_norm_value = float(
                total_grad_norm.detach().float().cpu().item()
                if torch.is_tensor(total_grad_norm)
                else total_grad_norm
            )
            if not math.isfinite(total_grad_norm_value):
                metrics["adapter_prediction_nonfinite_grad"] = 1.0
                self._adapter_prediction_optimizer.zero_grad()
                raise FloatingPointError(
                    f"non-finite adapter prediction gradient norm: total_grad_norm={total_grad_norm_value}"
                )
            self._adapter_prediction_optimizer.step()

            metrics["adapter_prediction_total_loss"] = float(total_loss.detach().cpu().item())
            metrics["adapter_prediction_total_grad_norm"] = total_grad_norm_value
            metrics["adapter_prediction_skipped"] = 0.0
            self._maybe_export_adapter_prediction_debug(
                total_step=total_step,
                expected_frame=expected_frame,
                semantic_frame=semantic_frame,
                semantic_logits=semantic_logits_for_debug,
                semantic_target=semantic_target_for_debug,
                visibility_mask=visibility_mask_for_debug,
                metrics=metrics,
            )
            return metrics
        finally:
            self.clear_adapter_prediction_forward_cache()

    def _adapter_prediction_state_dict(self) -> Dict[str, object]:
        semantic_channel_names = None
        semantic_channel_weights = None
        if self._adapter_prediction_semantic_head is not None:
            semantic_channel_names = tuple(self._adapter_prediction_semantic_head.channel_names)
        if self._adapter_prediction_semantic_loss_fn is not None:
            semantic_channel_weights = (
                self._adapter_prediction_semantic_loss_fn.channel_loss_weights.detach().cpu()
            )
        return {
            "adapter_prediction_enabled": bool(self.adapter_prediction_enabled),
            "reward_head": (
                self._adapter_prediction_reward_head.state_dict()
                if self._adapter_prediction_reward_head is not None
                else None
            ),
            "semantic_head": (
                self._adapter_prediction_semantic_head.state_dict()
                if self._adapter_prediction_semantic_head is not None
                else None
            ),
            "optimizer": (
                self._adapter_prediction_optimizer.state_dict()
                if self._adapter_prediction_optimizer is not None
                else None
            ),
            "reward_target_names": tuple(spec.name for spec in DEFAULT_REWARD_TARGET_SPECS),
            "semantic_channel_names": semantic_channel_names,
            "semantic_channel_loss_weights": semantic_channel_weights,
        }

    def _load_adapter_prediction_state(self, state_dict: Optional[Mapping], load_optimizers: bool = True) -> None:
        if not state_dict:
            return
        if self._adapter_prediction_reward_head is not None and state_dict.get("reward_head") is not None:
            self._load_adapter_prediction_module_strict(
                self._adapter_prediction_reward_head, state_dict.get("reward_head"), "reward_head"
            )
        if self._adapter_prediction_semantic_head is not None and state_dict.get("semantic_head") is not None:
            self._load_adapter_prediction_module_strict(
                self._adapter_prediction_semantic_head, state_dict.get("semantic_head"), "semantic_head"
            )
        if (
            load_optimizers
            and self._adapter_prediction_optimizer is not None
            and state_dict.get("optimizer") is not None
        ):
            self._adapter_prediction_optimizer.load_state_dict(state_dict["optimizer"])
            self._optimizer_to_device(self._adapter_prediction_optimizer)

    @staticmethod
    def _load_adapter_prediction_module_strict(module: nn.Module, state_dict: Mapping, label: str) -> None:
        if not isinstance(state_dict, Mapping):
            raise RuntimeError(f"adapter_prediction.{label} must be a mapping")
        expected = module.state_dict()
        missing = sorted(set(expected) - set(state_dict))
        unexpected = sorted(set(state_dict) - set(expected))
        shape_mismatch = sorted(
            name
            for name in set(expected) & set(state_dict)
            if tuple(expected[name].shape) != tuple(state_dict[name].shape)
        )
        if missing or unexpected or shape_mismatch:
            raise RuntimeError(
                f"adapter_prediction.{label} strict state mismatch: "
                f"missing={missing}, unexpected={unexpected}, shape_mismatch={shape_mismatch}"
            )
        module.load_state_dict(state_dict, strict=True)
