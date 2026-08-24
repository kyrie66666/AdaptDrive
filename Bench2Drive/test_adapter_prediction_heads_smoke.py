#!/usr/bin/env python3
"""CPU smoke checks for adapter prediction heads."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCH2DRIVE_ROOT = PROJECT_ROOT / "Bench2Drive"
LEADERBOARD_ROOT = BENCH2DRIVE_ROOT / "leaderboard"
for path in (str(BENCH2DRIVE_ROOT), str(LEADERBOARD_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from rl.adapter_prediction_heads import (  # noqa: E402
    DEFAULT_REWARD_TARGET_SPECS,
    ProjectedSixViewFpnSemanticMaskHead,
    RewardPredictionHead,
    RewardPredictionLoss,
    RoachSemanticMaskLoss,
    build_reward_prediction_targets,
    default_roach_channel_loss_weights,
)


def main() -> None:
    torch.manual_seed(7)
    batch = 2
    feature_dim = 32
    level0 = torch.randn(batch, 6, feature_dim, 24, 40, requires_grad=True)
    level1 = torch.randn(batch, 6, feature_dim, 12, 20, requires_grad=True)
    level2 = torch.randn(batch, 6, feature_dim, 6, 10, requires_grad=True)
    level3 = torch.randn(batch, 6, feature_dim, 3, 5, requires_grad=True)
    semantic_head = ProjectedSixViewFpnSemanticMaskHead(
        feature_dim=feature_dim,
        levels=(0, 1, 2, 3),
        hidden_dim=16,
        bev_width=32,
        image_width=160,
        image_height=90,
    )
    semantic_out = semantic_head([[level0, level1, level2, level3]], return_aux=True)
    logits = semantic_out["logits"]
    assert tuple(logits.shape) == (batch, 15, 32, 32), tuple(logits.shape)
    assert "projector_valid_bev_ratio" in semantic_out["metrics"]

    target = torch.zeros(batch, 15, 32, 32, dtype=torch.uint8)
    target[:, 0, 6:26, 4:28] = 255
    target[:, 2, 15:17, 3:29] = 120
    target[:, 6, 14:19, 16:21] = 255
    target[:, 10, 10:14, 10:12] = 255
    target[:, 14, 4:6, 14:24] = 255
    channel_weights = default_roach_channel_loss_weights(semantic_head.channel_names)
    semantic_loss_fn = RoachSemanticMaskLoss(
        channel_names=semantic_head.channel_names,
        channel_loss_weights=channel_weights,
    )
    semantic_loss, semantic_metrics = semantic_loss_fn(
        logits,
        target,
        visibility_mask=semantic_out["visibility_mask"],
    )
    assert torch.isfinite(semantic_loss).item(), "semantic loss should be finite"
    assert "semantic_channel_pos_rate/road" in semantic_metrics

    reward_head = RewardPredictionHead(
        visual_state_dim=80,
        action_dim=9,
        output_dim=len(DEFAULT_REWARD_TARGET_SPECS),
        hidden_dim=64,
        action_hidden_dim=16,
    )
    visual_state = torch.randn(1, 80, requires_grad=True)
    action_vec = torch.randn(1, 9)
    reward_pred = reward_head(visual_state, action_vec)
    reward_targets = build_reward_prediction_targets(
        reward=1.25,
        reward_info={
            "r_speed": 0.8,
            "r_position": -0.1,
            "r_rotation": -0.05,
            "r_progress": 0.02,
            "route_progress_delta": 0.001,
        },
        device=torch.device("cpu"),
    )
    reward_loss, reward_metrics = RewardPredictionLoss()(reward_pred, reward_targets)
    assert torch.isfinite(reward_loss).item(), "reward loss should be finite"
    assert "reward_target_valid_rate/r_speed" in reward_metrics

    total = semantic_loss + reward_loss
    total.backward()
    semantic_grad_ok = any(
        param.grad is not None and float(param.grad.detach().abs().max().item()) > 0.0
        for param in semantic_head.parameters()
    )
    reward_grad_ok = any(
        param.grad is not None and float(param.grad.detach().abs().max().item()) > 0.0
        for param in reward_head.parameters()
    )
    assert semantic_grad_ok, "semantic head should receive gradients"
    assert reward_grad_ok, "reward head should receive gradients"
    print("adapter_prediction_heads_smoke: PASS", flush=True)


if __name__ == "__main__":
    main()
