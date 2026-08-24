"""SAC value and critic networks for AdaptDrive's clean dual-plan action."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


DUAL_PID_SUMMARY_DIM = 32
PID_PLAN_SUMMARY_DIM = 5
PID_BRAKE_SPEED = 0.4
PID_BRAKE_RATIO = 1.1
PID_CLIP_DELTA = 0.25


def build_prev_pid_input(
    prev_pid_summary: Optional[torch.Tensor],
    prev_pid_mask: Optional[torch.Tensor],
    batch_size: int,
    device: torch.device,
    summary_dim: int = DUAL_PID_SUMMARY_DIM,
) -> torch.Tensor:
    """Pack the previous PID summary together with its validity mask."""

    if prev_pid_summary is None:
        summary = torch.zeros(batch_size, summary_dim, device=device)
    else:
        summary = prev_pid_summary.float().to(device)
        if summary.dim() == 1:
            summary = summary.unsqueeze(0)

    if prev_pid_mask is None:
        mask = torch.zeros(batch_size, 1, device=device)
    else:
        mask = prev_pid_mask.float().to(device)
        if mask.dim() == 0:
            mask = mask.view(1, 1).expand(batch_size, 1)
        elif mask.dim() == 1:
            mask = mask.unsqueeze(1)
        elif mask.dim() > 2:
            mask = mask.view(batch_size, 1)

    return torch.cat([summary, mask], dim=-1)


def extract_dual_pid_plan_summary(
    longitudinal_trajectory: torch.Tensor,
    lateral_trajectory: torch.Tensor,
    speed: torch.Tensor,
) -> torch.Tensor:
    """Differentiable summary matching clean dual-trajectory PID semantics."""

    if lateral_trajectory.dim() not in (3, 4):
        raise ValueError(f"Unsupported lateral trajectory shape: {lateral_trajectory.shape}")
    if longitudinal_trajectory.dim() not in (3, 4):
        raise ValueError(f"Unsupported longitudinal trajectory shape: {longitudinal_trajectory.shape}")

    reshape_prefix = lateral_trajectory.shape[:-2]
    if lateral_trajectory.dim() == 4:
        batch_size, num_modes = lateral_trajectory.shape[:2]
        lateral = lateral_trajectory.reshape(-1, lateral_trajectory.shape[-2], 2)
        if longitudinal_trajectory.dim() == 3:
            longitudinal = longitudinal_trajectory.unsqueeze(1).expand(-1, num_modes, -1, -1)
        else:
            longitudinal = longitudinal_trajectory
        longitudinal = longitudinal.reshape(-1, longitudinal.shape[-2], 2)
        speed_flat = speed.reshape(batch_size, -1)[:, :1].expand(-1, num_modes).reshape(-1)
    else:
        lateral = lateral_trajectory
        longitudinal = longitudinal_trajectory
        speed_flat = speed.reshape(speed.shape[0], -1)[:, 0]

    longitudinal_deltas = longitudinal[:, 1:] - longitudinal[:, :-1]
    desired_speed = torch.linalg.norm(longitudinal_deltas, dim=-1).mean(dim=-1) / 0.2

    lateral_midpoints = (lateral[:, 1:] + lateral[:, :-1]) * 0.5
    midpoint_norms = torch.linalg.norm(lateral_midpoints, dim=-1)
    aim_indices = torch.argmin(torch.abs(midpoint_norms - speed_flat.unsqueeze(-1)), dim=-1)
    batch_indices = torch.arange(lateral.shape[0], device=lateral.device)
    aim = lateral[batch_indices, aim_indices]
    angle = torch.rad2deg(torch.pi / 2 - torch.atan2(aim[:, 1], aim[:, 0])) / 90.0

    safe_desired_speed = torch.clamp(desired_speed, min=1e-6)
    brake = (desired_speed < PID_BRAKE_SPEED) | ((speed_flat / safe_desired_speed) > PID_BRAKE_RATIO)
    delta = torch.clamp(desired_speed - speed_flat, 0.0, PID_CLIP_DELTA)
    summary = torch.stack([desired_speed, angle, angle, delta, brake.float()], dim=-1)
    return summary.view(*reshape_prefix, PID_PLAN_SUMMARY_DIM)


def _make_q(input_dim: int, hidden_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, 1),
    )


class HiPADValue(nn.Module):
    """State value network conditioned on the previous clean PID summary."""

    def __init__(self, feature_dim: int, hidden_dim: int, pid_summary_dim: int = DUAL_PID_SUMMARY_DIM):
        super().__init__()
        self.pid_summary_dim = int(pid_summary_dim)
        self.net = nn.Sequential(
            nn.Linear(feature_dim + self.pid_summary_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features, prev_pid_summary=None, prev_pid_mask=None):
        pid = build_prev_pid_input(
            prev_pid_summary,
            prev_pid_mask,
            features.shape[0],
            features.device,
            summary_dim=self.pid_summary_dim,
        )
        return self.net(torch.cat([features.float(), pid], dim=-1))


class HiPADDualTrajectoryCritic(nn.Module):
    """Twin-Q critic for clean longitudinal-plan plus lateral-plan actions."""

    def __init__(self, feature_dim: int, hidden_dim: int, fut_ts: int):
        super().__init__()
        self.fut_ts = int(fut_ts)
        input_dim = feature_dim + DUAL_PID_SUMMARY_DIM + 1 + fut_ts * 4 + PID_PLAN_SUMMARY_DIM
        self.q1 = _make_q(input_dim, hidden_dim)
        self.q2 = _make_q(input_dim, hidden_dim)

    def _combined(
        self,
        features,
        state,
        longitudinal_trajectory,
        lateral_trajectory,
        prev_pid_summary=None,
        prev_pid_mask=None,
    ):
        pid = build_prev_pid_input(
            prev_pid_summary,
            prev_pid_mask,
            features.shape[0],
            features.device,
            summary_dim=DUAL_PID_SUMMARY_DIM,
        )
        longitudinal_flat = longitudinal_trajectory.reshape(longitudinal_trajectory.shape[0], -1).float()
        lateral_flat = lateral_trajectory.reshape(lateral_trajectory.shape[0], -1).float()
        plan_summary = extract_dual_pid_plan_summary(
            longitudinal_trajectory,
            lateral_trajectory,
            state[:, 0:1],
        ).float()
        return torch.cat(
            [features.float(), pid, longitudinal_flat, lateral_flat, plan_summary],
            dim=-1,
        )

    def forward(
        self,
        features,
        state,
        longitudinal_trajectory,
        lateral_trajectory,
        prev_pid_summary=None,
        prev_pid_mask=None,
    ):
        combined = self._combined(
            features,
            state,
            longitudinal_trajectory,
            lateral_trajectory,
            prev_pid_summary,
            prev_pid_mask,
        )
        return self.q1(combined), self.q2(combined)

    def evaluate_candidates(
        self,
        features,
        state,
        longitudinal_trajectory,
        lateral_candidates,
        prev_pid_summary=None,
        prev_pid_mask=None,
    ):
        batch_size, num_modes = lateral_candidates.shape[:2]
        features_exp = features.unsqueeze(1).expand(-1, num_modes, -1).reshape(batch_size * num_modes, -1)
        state_exp = state.unsqueeze(1).expand(-1, num_modes, -1).reshape(batch_size * num_modes, -1)
        if longitudinal_trajectory.dim() == 3:
            longitudinal_exp = longitudinal_trajectory.unsqueeze(1).expand(-1, num_modes, -1, -1)
        elif longitudinal_trajectory.dim() == 4:
            expected_shape = (batch_size, num_modes, self.fut_ts, 2)
            if tuple(longitudinal_trajectory.shape) != expected_shape:
                raise ValueError(
                    f"mode-aligned longitudinal candidates have shape {tuple(longitudinal_trajectory.shape)}, "
                    f"expected {expected_shape}"
                )
            longitudinal_exp = longitudinal_trajectory
        else:
            raise ValueError(
                f"unsupported longitudinal candidate shape: {tuple(longitudinal_trajectory.shape)}"
            )
        longitudinal_exp = longitudinal_exp.reshape(batch_size * num_modes, self.fut_ts, 2)
        lateral_exp = lateral_candidates.reshape(batch_size * num_modes, self.fut_ts, 2)
        prev_pid_exp = None if prev_pid_summary is None else (
            prev_pid_summary.unsqueeze(1).expand(-1, num_modes, -1).reshape(batch_size * num_modes, -1)
        )
        prev_mask_exp = None if prev_pid_mask is None else (
            prev_pid_mask.unsqueeze(1).expand(-1, num_modes).reshape(batch_size * num_modes)
        )
        q1, q2 = self.forward(
            features_exp,
            state_exp,
            longitudinal_exp,
            lateral_exp,
            prev_pid_exp,
            prev_mask_exp,
        )
        return q1.view(batch_size, num_modes), q2.view(batch_size, num_modes)
