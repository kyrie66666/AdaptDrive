#!/usr/bin/env python3
"""CPU one-batch gradient smoke for the clean dual-trajectory critic."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "leaderboard"))
REPO_ROOT = PROJECT_ROOT.parent

from rl.hipad_project_runtime import activate_hipad_project_root

activate_hipad_project_root(REPO_ROOT / "HiP-AD", repo_root=REPO_ROOT)

from rl.adaptdrive_sac import (
    DUAL_PID_SUMMARY_DIM,
    HiPADDualTrajectoryCritic,
    HiPADValue,
    extract_dual_pid_plan_summary,
)
from rl.hipad_clean_control import clean_dual_pid_step, create_clean_dual_pid_controller


def main() -> None:
    torch.manual_seed(7)
    batch_size, num_modes, feature_dim, hidden_dim, fut_ts = 3, 5, 16, 32, 6
    critic = HiPADDualTrajectoryCritic(feature_dim, hidden_dim, fut_ts)
    value = HiPADValue(feature_dim, hidden_dim, pid_summary_dim=DUAL_PID_SUMMARY_DIM)

    features = torch.randn(batch_size, feature_dim)
    state = torch.randn(batch_size, 21)
    state[:, 0] = state[:, 0].abs()
    longitudinal = torch.randn(batch_size, num_modes, fut_ts, 2)
    candidates = torch.randn(batch_size, num_modes, fut_ts, 2, requires_grad=True)
    prev_pid = torch.randn(batch_size, DUAL_PID_SUMMARY_DIM)
    prev_mask = torch.ones(batch_size)

    captured = {}

    def capture_combined(_module, args):
        captured["combined"] = args[0].detach().clone()

    hook = critic.q1[0].register_forward_pre_hook(capture_combined)

    q1, q2 = critic.evaluate_candidates(
        features,
        state,
        longitudinal,
        candidates,
        prev_pid,
        prev_mask,
    )
    hook.remove()
    assert q1.shape == (batch_size, num_modes)
    assert q2.shape == (batch_size, num_modes)
    longitudinal_start = feature_dim + DUAL_PID_SUMMARY_DIM + 1
    longitudinal_end = longitudinal_start + fut_ts * 2
    torch.testing.assert_close(
        captured["combined"][:, longitudinal_start:longitudinal_end],
        longitudinal.reshape(batch_size * num_modes, fut_ts * 2),
    )
    actor_loss = -torch.minimum(q1, q2).mean()
    actor_loss.backward()
    assert candidates.grad is not None
    assert torch.isfinite(candidates.grad).all()
    assert float(candidates.grad.abs().sum()) > 0.0

    v = value(features, prev_pid, prev_mask)
    assert v.shape == (batch_size, 1)
    assert torch.isfinite(v).all()

    longitudinal_np = longitudinal[0, 0].numpy().astype("float32")
    lateral_np = candidates.detach()[0, 0].numpy().astype("float32")
    speed_np = state.detach()[0, 0].numpy().astype("float32")
    _, metadata = clean_dual_pid_step(
        create_clean_dual_pid_controller(),
        longitudinal_np,
        lateral_np,
        speed_np,
        torch.zeros(2).numpy().astype("float32"),
    )
    differentiable = extract_dual_pid_plan_summary(
        longitudinal[0:1, 0],
        candidates.detach()[0:1, 0],
        state[0:1, 0:1],
    )[0]
    expected = torch.tensor([
        metadata["desired_speed"],
        metadata["angle_final"],
        metadata["angle_target"],
        metadata["delta"],
        metadata["brake"],
    ])
    # Clean control hard-disables target-point steering, so angle_final is the
    # lateral-plan angle.  The critic deliberately stores that value in both
    # angle slots; target angle is already represented by observation context.
    expected[2] = expected[1]
    torch.testing.assert_close(differentiable, expected, rtol=1e-5, atol=1e-5)
    print("HiP-AD dual critic one-batch gradient smoke passed")


if __name__ == "__main__":
    main()
