"""Pure update-gating logic for HiP-AD clean SAC training."""

from __future__ import annotations

from typing import Optional, Tuple


def policy_update_decision(
    *,
    total_step: int,
    critic_update_count: int,
    critic_q_loss_ema: Optional[float],
    policy_learning_starts: int,
    policy_update_every_n_steps: int,
    min_critic_updates_before_policy: int,
    max_policy_q_loss_ema: float,
) -> Tuple[bool, str]:
    """Return whether an actor update is safe and the explicit skip reason."""

    if int(total_step) < int(policy_learning_starts):
        return False, "before_policy_learning_starts"
    if int(critic_update_count) < int(min_critic_updates_before_policy):
        return False, "insufficient_critic_updates"
    if critic_q_loss_ema is None:
        return False, "critic_q_loss_ema_unavailable"
    max_q_loss = float(max_policy_q_loss_ema)
    if max_q_loss > 0.0 and float(critic_q_loss_ema) > max_q_loss:
        return False, "critic_q_loss_ema_above_threshold"
    update_every = max(1, int(policy_update_every_n_steps))
    if int(critic_update_count) % update_every != 0:
        return False, "policy_update_interval"
    return True, ""


def checkpoint_resume_decision(
    *,
    checkpoint_signature: Optional[str],
    current_signature: str,
    checkpoint_signature_version: int,
    current_signature_version: int,
    strict_signature: bool,
    allow_signature_mismatch_full_resume: bool,
    has_full_state: bool,
) -> Tuple[bool, str]:
    """Decide whether replay/optimizers may be restored from a checkpoint.

    Older signature versions are never treated as implicitly compatible. A
    mismatched checkpoint needs the same explicit override regardless of its
    version; otherwise only model weights may be loaded.
    """

    if checkpoint_signature == current_signature:
        if not has_full_state:
            return False, "matching_signature_missing_full_state"
        return True, "signature_match"
    if strict_signature:
        return False, "strict_signature_mismatch"
    if not allow_signature_mismatch_full_resume:
        if int(checkpoint_signature_version) < int(current_signature_version):
            return False, "legacy_signature_requires_explicit_override"
        return False, "signature_mismatch_requires_explicit_override"
    if not has_full_state:
        return False, "signature_override_missing_full_state"
    return True, "explicit_signature_mismatch_override"
