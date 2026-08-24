#!/usr/bin/env python3
"""Fast checks for critic-warmup and actor-update gates."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "leaderboard"))

from rl.hipad_training_gates import checkpoint_resume_decision, policy_update_decision


DEFAULTS = {
    "policy_learning_starts": 5000,
    "policy_update_every_n_steps": 2,
    "min_critic_updates_before_policy": 10,
    "max_policy_q_loss_ema": 5.0,
}


def decide(**overrides):
    args = {
        "total_step": 5000,
        "critic_update_count": 10,
        "critic_q_loss_ema": 1.0,
        **DEFAULTS,
        **overrides,
    }
    return policy_update_decision(**args)


def main() -> None:
    assert decide(total_step=4999) == (False, "before_policy_learning_starts")
    assert decide(critic_update_count=8) == (False, "insufficient_critic_updates")
    assert decide(critic_q_loss_ema=None) == (False, "critic_q_loss_ema_unavailable")
    assert decide(critic_q_loss_ema=5.1) == (False, "critic_q_loss_ema_above_threshold")
    assert decide(critic_update_count=11) == (False, "policy_update_interval")
    assert decide(critic_update_count=12) == (True, "")
    assert decide(critic_q_loss_ema=500.0, max_policy_q_loss_ema=0.0) == (True, "")
    resume_defaults = {
        "checkpoint_signature": "old",
        "current_signature": "new",
        "checkpoint_signature_version": 3,
        "current_signature_version": 4,
        "strict_signature": False,
        "allow_signature_mismatch_full_resume": False,
        "has_full_state": True,
    }
    assert checkpoint_resume_decision(**resume_defaults) == (
        False,
        "legacy_signature_requires_explicit_override",
    )
    assert checkpoint_resume_decision(
        **{**resume_defaults, "allow_signature_mismatch_full_resume": True}
    ) == (True, "explicit_signature_mismatch_override")
    assert checkpoint_resume_decision(
        **{
            **resume_defaults,
            "checkpoint_signature": "new",
            "checkpoint_signature_version": 4,
        }
    ) == (True, "signature_match")
    v8_defaults = {
        **resume_defaults,
        "checkpoint_signature_version": 7,
        "current_signature_version": 8,
        "allow_signature_mismatch_full_resume": False,
    }
    assert checkpoint_resume_decision(**v8_defaults) == (
        False,
        "legacy_signature_requires_explicit_override",
    )
    assert checkpoint_resume_decision(
        **{
            **resume_defaults,
            "allow_signature_mismatch_full_resume": True,
            "strict_signature": True,
        }
    ) == (False, "strict_signature_mismatch")
    print("HiP-AD policy update gates smoke passed")


if __name__ == "__main__":
    main()
