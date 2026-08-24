"""Prediction heads for online feature-adapter representation learning.

The heads in this module are attached after EgoStateDCNv4FeatureAdapter.  They
are trained by the rollout step that just used the adapted features, but they
do not change replay contents or the SAC actor/critic optimizers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from rl.roach_bev_target import ROACH_DEFAULT_HISTORY_IDX, roach_bev_channel_names


EPS = 1e-6


@dataclass(frozen=True)
class RewardTargetSpec:
    name: str
    key: str
    clip_min: float
    clip_max: float
    scale: float = 1.0
    weight: float = 1.0


DEFAULT_REWARD_TARGET_SPECS: Tuple[RewardTargetSpec, ...] = (
    RewardTargetSpec("reward", "__reward__", -10.0, 10.0, 10.0, 1.0),
    RewardTargetSpec("r_speed", "r_speed", -1.0, 1.0, 1.0, 1.0),
    RewardTargetSpec("r_position", "r_position", -1.0, 1.0, 1.0, 1.0),
    RewardTargetSpec("r_rotation", "r_rotation", -1.0, 1.0, 1.0, 1.0),
    RewardTargetSpec("r_progress", "r_progress", -1.0, 1.0, 1.0, 1.0),
    RewardTargetSpec("route_progress_delta", "route_progress_delta", 0.0, 0.01, 0.01, 1.0),
    RewardTargetSpec("r_dense_safety_direct", "r_dense_safety_direct", -1.0, 1.0, 1.0, 0.5),
    RewardTargetSpec("r_dense_ttc", "r_dense_ttc", -1.0, 1.0, 1.0, 0.5),
    RewardTargetSpec("r_dense_headway", "r_dense_headway", -1.0, 1.0, 1.0, 0.5),
    RewardTargetSpec("r_dense_min_distance", "r_dense_min_distance", -1.0, 1.0, 1.0, 0.5),
)


@dataclass
class RewardPredictionTargets:
    values: torch.Tensor
    valid_mask: torch.Tensor
    names: Tuple[str, ...]
    weights: torch.Tensor


@dataclass(frozen=True)
class CameraSpec:
    name: str
    x: float
    y: float
    z: float
    yaw: float
    pitch: float = 0.0
    roll: float = 0.0
    fov: float = 70.0


DEFAULT_HIPAD_CAMERA_SPECS: Tuple[CameraSpec, ...] = (
    CameraSpec("front", 0.80, 0.0, 1.60, 0.0, fov=70.0),
    CameraSpec("front_left", 0.27, -0.55, 1.60, -55.0, fov=70.0),
    CameraSpec("front_right", 0.27, 0.55, 1.60, 55.0, fov=70.0),
    CameraSpec("rear", -2.0, 0.0, 1.60, 180.0, fov=110.0),
    CameraSpec("rear_left", -0.32, -0.55, 1.60, -110.0, fov=70.0),
    CameraSpec("rear_right", -0.32, 0.55, 1.60, 110.0, fov=70.0),
)


def _safe_float(value: object) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _sanitize_metric_name(name: str) -> str:
    return str(name).replace("/", "_").replace(" ", "_")


def _choose_norm_groups(channels: int) -> int:
    for group in (8, 4, 2, 1):
        if channels % group == 0:
            return group
    return 1


def _rotation_matrix_yaw_pitch_roll(yaw: float, pitch: float, roll: float) -> torch.Tensor:
    """CARLA-style local camera axes to ego axes rotation matrix."""

    yaw_r = math.radians(float(yaw))
    pitch_r = math.radians(float(pitch))
    roll_r = math.radians(float(roll))

    cy, sy = math.cos(yaw_r), math.sin(yaw_r)
    cp, sp = math.cos(pitch_r), math.sin(pitch_r)
    cr, sr = math.cos(roll_r), math.sin(roll_r)

    rz = torch.tensor(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    ry = torch.tensor(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
        dtype=torch.float32,
    )
    rx = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
        dtype=torch.float32,
    )
    return rz @ ry @ rx


def _normalize_semantic_target(target: torch.Tensor, scale: float = 255.0) -> torch.Tensor:
    target = target.float()
    if target.numel() > 0 and float(scale) > 0.0 and target.detach().max().item() > 1.0:
        target = target / float(scale)
    return target.clamp(0.0, 1.0)


def default_roach_channel_loss_weights(
    channel_names: Sequence[str],
    *,
    route_weight: float = 0.0,
    road_weight: float = 1.0,
    lane_weight: float = 1.0,
    latest_vehicle_weight: float = 1.0,
    latest_walker_weight: float = 1.0,
    latest_traffic_light_stop_weight: float = 1.0,
) -> torch.Tensor:
    """Channel weights for the first production BEV-mask supervision stage."""

    weights: List[float] = []
    for name in channel_names:
        weight = 0.0
        if name == "road":
            weight = float(road_weight)
        elif name == "route":
            weight = float(route_weight)
        elif name == "lane":
            weight = float(lane_weight)
        elif name == "vehicle_h-1":
            weight = float(latest_vehicle_weight)
        elif name == "walker_h-1":
            weight = float(latest_walker_weight)
        elif name == "traffic_light_stop_h-1":
            weight = float(latest_traffic_light_stop_weight)
        weights.append(float(weight))
    return torch.tensor(weights, dtype=torch.float32)


def build_reward_prediction_targets(
    *,
    reward: float,
    reward_info: Optional[Mapping[str, object]],
    device: torch.device,
    specs: Sequence[RewardTargetSpec] = DEFAULT_REWARD_TARGET_SPECS,
) -> RewardPredictionTargets:
    reward_info = reward_info or {}
    values: List[float] = []
    masks: List[float] = []
    weights: List[float] = []
    names: List[str] = []
    for spec in specs:
        raw_value = reward if spec.key == "__reward__" else reward_info.get(spec.key)
        value = _safe_float(raw_value)
        if value is None:
            values.append(0.0)
            masks.append(0.0)
        else:
            clipped = min(max(value, float(spec.clip_min)), float(spec.clip_max))
            scale = float(spec.scale) if abs(float(spec.scale)) > EPS else 1.0
            values.append(float(clipped) / scale)
            masks.append(1.0)
        weights.append(float(spec.weight))
        names.append(str(spec.name))

    return RewardPredictionTargets(
        values=torch.tensor(values, device=device, dtype=torch.float32).view(1, -1),
        valid_mask=torch.tensor(masks, device=device, dtype=torch.float32).view(1, -1),
        names=tuple(names),
        weights=torch.tensor(weights, device=device, dtype=torch.float32).view(1, -1),
    )


class RewardPredictionHead(nn.Module):
    """Action-conditioned reward/component predictor.

    The action branch is intentionally narrower than the visual-state branch so
    the head cannot cheaply ignore adapted scene features and rely only on the
    selected action prior.
    """

    def __init__(
        self,
        *,
        visual_state_dim: int,
        action_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        action_hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.visual_state_dim = int(visual_state_dim)
        self.action_dim = int(action_dim)
        self.output_dim = int(output_dim)
        self.visual_state_embed = nn.Sequential(
            nn.LayerNorm(self.visual_state_dim),
            nn.Linear(self.visual_state_dim, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
        )
        self.action_embed = nn.Sequential(
            nn.LayerNorm(self.action_dim),
            nn.Linear(self.action_dim, int(action_hidden_dim)),
            nn.ReLU(inplace=True),
        )
        self.predictor = nn.Sequential(
            nn.Linear(int(hidden_dim) + int(action_hidden_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), self.output_dim),
        )

    def forward(self, visual_state: torch.Tensor, action_vec: torch.Tensor) -> torch.Tensor:
        if visual_state.dim() != 2:
            raise ValueError(f"visual_state must be [B,D], got {tuple(visual_state.shape)}")
        if action_vec.dim() == 1:
            action_vec = action_vec.unsqueeze(0)
        if action_vec.dim() != 2:
            raise ValueError(f"action_vec must be [B,A], got {tuple(action_vec.shape)}")
        if visual_state.shape[0] != action_vec.shape[0]:
            raise ValueError(
                f"visual/action batch mismatch: {visual_state.shape[0]} vs {action_vec.shape[0]}"
            )
        return self.predictor(
            torch.cat(
                [
                    self.visual_state_embed(visual_state.float()),
                    self.action_embed(action_vec.float()),
                ],
                dim=-1,
            )
        )


class RewardPredictionLoss(nn.Module):
    def __init__(self, huber_delta: float = 1.0) -> None:
        super().__init__()
        self.huber_delta = float(huber_delta)

    def forward(
        self,
        pred: torch.Tensor,
        targets: RewardPredictionTargets,
        *,
        prefix: str = "reward_pred",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if pred.shape != targets.values.shape:
            raise ValueError(f"reward pred shape {tuple(pred.shape)} != target {tuple(targets.values.shape)}")
        valid = targets.valid_mask.float()
        weights = targets.weights.float()
        raw = F.huber_loss(pred.float(), targets.values.float(), delta=self.huber_delta, reduction="none")
        weighted = raw * valid * weights
        denom = (valid * weights).sum().clamp_min(EPS)
        loss = weighted.sum() / denom

        metrics: Dict[str, torch.Tensor] = {
            f"{prefix}_loss": loss.detach(),
            f"{prefix}_target_valid_rate": valid.mean().detach(),
        }
        for idx, name in enumerate(targets.names):
            key = _sanitize_metric_name(name)
            valid_i = valid[:, idx]
            denom_i = valid_i.sum().clamp_min(EPS)
            metrics[f"{prefix}_loss_{key}"] = ((raw[:, idx] * valid_i).sum() / denom_i).detach()
            metrics[f"reward_target_valid_rate/{key}"] = valid_i.mean().detach()
            metrics[f"reward_target_mean/{key}"] = (
                (targets.values[:, idx] * valid_i).sum() / denom_i
            ).detach()
            metrics[f"reward_pred_mean/{key}"] = ((pred[:, idx] * valid_i).sum() / denom_i).detach()
        return loss, metrics


class GroundPlaneProjector(nn.Module):
    """Fixed ego-ground-plane to multi-camera feature-grid projector."""

    def __init__(
        self,
        *,
        bev_width: int = 192,
        pixels_per_meter: float = 5.0,
        pixels_ev_to_bottom: int = 40,
        image_width: int = 1600,
        image_height: int = 900,
        camera_specs: Sequence[CameraSpec] = DEFAULT_HIPAD_CAMERA_SPECS,
        min_forward_depth: float = 0.05,
    ) -> None:
        super().__init__()
        self.bev_width = int(bev_width)
        self.pixels_per_meter = float(pixels_per_meter)
        self.pixels_ev_to_bottom = int(pixels_ev_to_bottom)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.min_forward_depth = float(min_forward_depth)
        self.camera_names = tuple(spec.name for spec in camera_specs)

        locs = []
        rots = []
        focal_lengths = []
        for spec in camera_specs:
            locs.append([float(spec.x), float(spec.y), float(spec.z)])
            rots.append(_rotation_matrix_yaw_pitch_roll(spec.yaw, spec.pitch, spec.roll))
            focal_lengths.append(
                0.5 * float(self.image_width) / math.tan(0.5 * math.radians(float(spec.fov)))
            )
        self.register_buffer("camera_locs", torch.tensor(locs, dtype=torch.float32), persistent=False)
        self.register_buffer("camera_rot_c2v", torch.stack(rots, dim=0).float(), persistent=False)
        self.register_buffer("camera_focal", torch.tensor(focal_lengths, dtype=torch.float32), persistent=False)

        rows = torch.arange(self.bev_width, dtype=torch.float32)
        cols = torch.arange(self.bev_width, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
        forward = (float(self.bev_width - self.pixels_ev_to_bottom) - 0.5 - grid_y) / self.pixels_per_meter
        right = (grid_x + 0.5 - 0.5 * float(self.bev_width)) / self.pixels_per_meter
        up = torch.zeros_like(forward)
        self.register_buffer(
            "bev_points_vehicle",
            torch.stack([forward, right, up], dim=-1).float(),
            persistent=False,
        )

    @property
    def num_cameras(self) -> int:
        return int(self.camera_locs.shape[0])

    def projection_grid(self, *, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        points = self.bev_points_vehicle.to(device=device, dtype=torch.float32).view(1, -1, 3)
        locs = self.camera_locs.to(device=device, dtype=torch.float32).view(self.num_cameras, 1, 3)
        rot_v2c = self.camera_rot_c2v.to(device=device, dtype=torch.float32).transpose(1, 2)
        rel = points - locs
        cam = torch.einsum("nmc,nkc->nkm", rot_v2c, rel)
        x_forward = cam[..., 0].clamp_min(EPS)
        y_right = cam[..., 1]
        z_up = cam[..., 2]

        focal = self.camera_focal.to(device=device, dtype=torch.float32).view(self.num_cameras, 1)
        u = focal * (y_right / x_forward) + 0.5 * float(self.image_width)
        v = -focal * (z_up / x_forward) + 0.5 * float(self.image_height)
        valid = (
            (cam[..., 0] > float(self.min_forward_depth))
            & (u >= 0.0)
            & (u <= float(self.image_width - 1))
            & (v >= 0.0)
            & (v <= float(self.image_height - 1))
        )
        grid_x = (u + 0.5) / float(self.image_width) * 2.0 - 1.0
        grid_y = (v + 0.5) / float(self.image_height) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).view(
            self.num_cameras,
            self.bev_width,
            self.bev_width,
            2,
        )
        valid = valid.view(self.num_cameras, self.bev_width, self.bev_width)
        return grid.to(dtype=dtype), valid


class ProjectedSixViewFpnSemanticMaskHead(nn.Module):
    """Project six-view FPN features onto ego BEV and decode Roach masks."""

    def __init__(
        self,
        *,
        feature_dim: int = 256,
        levels: Sequence[int] = (0, 1, 2, 3),
        hidden_dim: int = 128,
        bev_width: int = 192,
        pixels_per_meter: float = 5.0,
        pixels_ev_to_bottom: int = 40,
        image_width: int = 1600,
        image_height: int = 900,
        history_idx: Sequence[int] = ROACH_DEFAULT_HISTORY_IDX,
        camera_specs: Sequence[CameraSpec] = DEFAULT_HIPAD_CAMERA_SPECS,
        output_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.levels = tuple(int(level) for level in levels)
        if not self.levels:
            raise ValueError("semantic head levels must not be empty")
        self.hidden_dim = int(hidden_dim)
        self.channel_names = roach_bev_channel_names(tuple(int(idx) for idx in history_idx))
        self.output_channels = int(output_channels or len(self.channel_names))
        if self.output_channels != len(self.channel_names):
            raise ValueError(
                f"output_channels={self.output_channels} does not match Roach channel count "
                f"{len(self.channel_names)}"
            )
        self.projector = GroundPlaneProjector(
            bev_width=int(bev_width),
            pixels_per_meter=float(pixels_per_meter),
            pixels_ev_to_bottom=int(pixels_ev_to_bottom),
            image_width=int(image_width),
            image_height=int(image_height),
            camera_specs=camera_specs,
        )
        self.level_projections = nn.ModuleDict(
            {
                str(level): nn.Sequential(
                    nn.Conv2d(self.feature_dim, self.hidden_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(_choose_norm_groups(self.hidden_dim), self.hidden_dim),
                    nn.ReLU(inplace=True),
                )
                for level in self.levels
            }
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_choose_norm_groups(self.hidden_dim), self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_choose_norm_groups(self.hidden_dim), self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, self.output_channels, kernel_size=1),
        )

    def _project_level(
        self,
        level_feature: torch.Tensor,
        *,
        level: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if level_feature.dim() != 5:
            raise ValueError(f"level feature must be [B,Ncam,C,H,W], got {tuple(level_feature.shape)}")
        batch, ncam, channels, _, _ = level_feature.shape
        if ncam != self.projector.num_cameras:
            raise ValueError(f"expected {self.projector.num_cameras} cameras, got {ncam}")
        if channels != self.feature_dim:
            raise ValueError(f"expected feature_dim={self.feature_dim}, got {channels}")

        grid, valid = self.projector.projection_grid(device=level_feature.device, dtype=level_feature.dtype)
        grid = grid.unsqueeze(0).expand(batch, -1, -1, -1, -1).reshape(
            batch * ncam,
            self.projector.bev_width,
            self.projector.bev_width,
            2,
        )
        flat = level_feature.float().reshape(batch * ncam, channels, level_feature.shape[-2], level_feature.shape[-1])
        sampled = F.grid_sample(flat, grid.float(), mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled = sampled.view(batch, ncam, channels, self.projector.bev_width, self.projector.bev_width)
        valid_bool = valid.to(device=level_feature.device).view(
            1,
            ncam,
            1,
            self.projector.bev_width,
            self.projector.bev_width,
        )
        valid_f = valid_bool.to(dtype=sampled.dtype)
        fused = (sampled * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp_min(1.0)
        visibility = valid_bool.any(dim=1).float()
        camera_ratio = valid.to(device=level_feature.device, dtype=torch.float32).mean(dim=(1, 2)).detach()
        camera_visibility = valid_f.mean().detach()
        out_of_bounds = (1.0 - valid_f.mean()).detach()
        bev_ratio = visibility.mean().detach()
        metrics = {
            "projector_camera_visibility_ratio": camera_visibility,
            "projector_grid_out_of_bounds_ratio": out_of_bounds,
            "projector_valid_bev_ratio": bev_ratio,
            f"projector_camera_visibility_ratio_L{int(level)}": camera_visibility,
            f"projector_grid_out_of_bounds_ratio_L{int(level)}": out_of_bounds,
            f"projector_valid_bev_ratio_L{int(level)}": bev_ratio,
        }
        for camera_idx, camera_name in enumerate(self.projector.camera_names):
            key = _sanitize_metric_name(camera_name)
            metrics[f"projector_camera_visibility_ratio/{key}"] = camera_ratio[camera_idx]
            metrics[f"projector_camera_visibility_ratio_L{int(level)}/{key}"] = camera_ratio[camera_idx]
        return fused, visibility, metrics

    def forward(self, adapted_groups: Sequence[Sequence[torch.Tensor]], *, return_aux: bool = False):
        projected: List[torch.Tensor] = []
        visibility_masks: List[torch.Tensor] = []
        metric_values: Dict[str, List[torch.Tensor]] = {}
        for group in adapted_groups:
            for level in self.levels:
                if level >= len(group):
                    raise ValueError(f"semantic head requested L{level}, but group has only {len(group)} levels")
                fused, visibility, metrics = self._project_level(group[level], level=level)
                projected.append(self.level_projections[str(level)](fused))
                visibility_masks.append(visibility)
                for key, value in metrics.items():
                    metric_values.setdefault(key, []).append(value)
        if not projected:
            raise ValueError("no projected features were produced")
        bev_feature = torch.stack(projected, dim=0).mean(dim=0)
        visibility_mask = torch.stack(visibility_masks, dim=0).amax(dim=0)
        logits = self.decoder(bev_feature)
        metrics = {
            key: torch.stack(values).mean().detach()
            for key, values in metric_values.items()
            if values
        }
        if return_aux:
            return {
                "logits": logits,
                "visibility_mask": visibility_mask,
                "metrics": metrics,
            }
        return logits


class RoachSemanticMaskLoss(nn.Module):
    def __init__(
        self,
        *,
        channel_names: Sequence[str],
        channel_loss_weights: torch.Tensor,
        bce_weight: float = 1.0,
        dice_weight: float = 0.5,
        positive_weight: float = 2.0,
        target_scale: float = 255.0,
    ) -> None:
        super().__init__()
        self.channel_names = tuple(str(name) for name in channel_names)
        if int(channel_loss_weights.numel()) != len(self.channel_names):
            raise ValueError("channel_loss_weights length must match channel_names")
        self.register_buffer("channel_loss_weights", channel_loss_weights.float().view(1, -1), persistent=True)
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.positive_weight = float(positive_weight)
        self.target_scale = float(target_scale)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        *,
        visibility_mask: Optional[torch.Tensor] = None,
        prefix: str = "adapter_prediction_semantic",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if target.dim() == 3:
            target = target.unsqueeze(0)
        if target.dim() != 4:
            raise ValueError(f"semantic target must be [B,C,H,W], got {tuple(target.shape)}")
        if target.shape[1] != logits.shape[1]:
            raise ValueError(f"semantic channel mismatch: logits={logits.shape[1]}, target={target.shape[1]}")
        target = _normalize_semantic_target(target.to(device=logits.device), scale=self.target_scale)
        if tuple(target.shape[-2:]) != tuple(logits.shape[-2:]):
            logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)

        if visibility_mask is None:
            mask = torch.ones((target.shape[0], 1, target.shape[2], target.shape[3]), device=target.device)
        else:
            mask = visibility_mask.to(device=target.device, dtype=target.dtype)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            if tuple(mask.shape[-2:]) != tuple(target.shape[-2:]):
                mask = F.interpolate(mask, size=target.shape[-2:], mode="nearest")
            if mask.shape[1] != 1:
                mask = mask[:, :1]
        mask = mask.float().clamp(0.0, 1.0)
        mask_channels = mask.expand(-1, target.shape[1], -1, -1)

        positive_factor = 1.0 + target.float().gt(0.0).float() * (float(self.positive_weight) - 1.0)
        bce = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
        bce = bce * positive_factor
        raw_channel_mask_sum = mask_channels.sum(dim=(0, 2, 3))
        channel_has_visibility = raw_channel_mask_sum > EPS
        per_channel_mask_sum = raw_channel_mask_sum.clamp_min(EPS)
        bce_channel = (bce * mask_channels).sum(dim=(0, 2, 3)) / per_channel_mask_sum

        probs = torch.sigmoid(logits.float())
        intersection = (probs * target.float() * mask_channels).sum(dim=(0, 2, 3))
        denominator = ((probs + target.float()) * mask_channels).sum(dim=(0, 2, 3)).clamp_min(EPS)
        dice_channel = 1.0 - (2.0 * intersection + EPS) / (denominator + EPS)
        channel_loss = float(self.bce_weight) * bce_channel + float(self.dice_weight) * dice_channel
        channel_loss = torch.where(channel_has_visibility, channel_loss, torch.zeros_like(channel_loss))

        weights = self.channel_loss_weights.to(device=logits.device).view(-1)
        active = (weights > 0.0) & channel_has_visibility
        if bool(active.any()):
            loss = (channel_loss * weights).sum() / weights[active].sum().clamp_min(EPS)
        else:
            loss = logits.float().sum() * 0.0

        target_positive = ((target.float() > 0.0).float() * mask_channels).sum(dim=(0, 2, 3)) / per_channel_mask_sum
        metrics: Dict[str, torch.Tensor] = {
            f"{prefix}_loss": loss.detach(),
            "semantic_visibility_valid_rate": mask.mean().detach(),
        }
        for idx, name in enumerate(self.channel_names):
            key = _sanitize_metric_name(name)
            metrics[f"semantic_channel_pos_rate/{key}"] = target_positive[idx].detach()
            metrics[f"semantic_channel_loss/{key}"] = channel_loss[idx].detach()
            metrics[f"semantic_channel_weight/{key}"] = weights[idx].detach()
        return loss, metrics
