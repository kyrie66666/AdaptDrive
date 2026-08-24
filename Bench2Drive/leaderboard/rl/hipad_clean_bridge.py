"""Non-invasive runtime bridge for the unmodified HiP-AD model."""

from __future__ import annotations

from typing import Optional

import torch


class HiPADCleanBridgeError(RuntimeError):
    """Raised when clean model context capture/reset invariants are violated."""


class HiPADCleanPlanningBridge:
    """Capture the final spatial align query only during a rollout head forward.

    The final spatial regression branch is also called by replay-time policy
    updates.  Capture is explicitly scoped so those calls cannot overwrite the
    rollout context stored with a transition.
    """

    def __init__(self, onedecoder, *, num_modes: int = 48, feature_dim: int = 256):
        self.onedecoder = onedecoder
        self.num_modes = int(num_modes)
        self.feature_dim = int(feature_dim)
        self._capture_active = False
        self._frame_token = None
        self._capture_count = 0
        self._captured_context: Optional[torch.Tensor] = None

        plan_refine = getattr(onedecoder, "plan_refine", None)
        if not plan_refine:
            raise HiPADCleanBridgeError("clean onedecoder has no plan_refine layers")
        final_refine = plan_refine[-1]
        branch = getattr(final_refine, "plan_reg_branch_spat_2m", None)
        if branch is None:
            raise HiPADCleanBridgeError("final plan_refine has no plan_reg_branch_spat_2m")
        self._hook_handle = branch.register_forward_pre_hook(self._capture_spatial_context)

    def _capture_spatial_context(self, _module, inputs) -> None:
        if not self._capture_active:
            return
        if not inputs or not torch.is_tensor(inputs[0]):
            raise HiPADCleanBridgeError("spat-2m hook did not receive a tensor context")
        self._capture_count += 1
        if self._capture_count > 1:
            raise HiPADCleanBridgeError(
                f"spat-2m branch fired more than once for frame_token={self._frame_token!r}"
            )
        self._captured_context = inputs[0]

    def begin_rollout_capture(self, frame_token) -> None:
        if self._capture_active:
            raise HiPADCleanBridgeError(
                f"nested rollout capture: active={self._frame_token!r}, new={frame_token!r}"
            )
        self._capture_active = True
        self._frame_token = frame_token
        self._capture_count = 0
        self._captured_context = None

    def end_rollout_capture(self) -> torch.Tensor:
        if not self._capture_active:
            raise HiPADCleanBridgeError("end_rollout_capture called without begin_rollout_capture")
        token = self._frame_token
        count = self._capture_count
        context = self._captured_context
        self._capture_active = False
        self._frame_token = None
        self._capture_count = 0
        self._captured_context = None

        if count != 1 or context is None:
            raise HiPADCleanBridgeError(
                f"expected one spat-2m context for frame_token={token!r}, captured {count}"
            )
        if context.ndim != 3:
            raise HiPADCleanBridgeError(f"plan align context must be rank 3, got {tuple(context.shape)}")
        if tuple(context.shape[1:]) != (self.num_modes, self.feature_dim):
            raise HiPADCleanBridgeError(
                "unexpected plan align context shape: "
                f"got {tuple(context.shape)}, expected [B, {self.num_modes}, {self.feature_dim}]"
            )
        if not torch.isfinite(context).all():
            raise HiPADCleanBridgeError("plan align context contains non-finite values")
        return context

    def abort_rollout_capture(self) -> None:
        self._capture_active = False
        self._frame_token = None
        self._capture_count = 0
        self._captured_context = None

    def reset_temporal_state(self) -> int:
        """Reset every close-loop bank and return the number of reset banks."""

        reset_count = 0
        for name in (
            "det_instance_bank_list",
            "map_instance_bank_list",
            "ego_instance_bank_list",
            "plan_instance_bank_list",
            "scenes_instance_bank_list",
        ):
            for bank in getattr(self.onedecoder, name, ()):
                reset = getattr(bank, "reset", None)
                if callable(reset):
                    reset()
                    reset_count += 1
        self.onedecoder.run_step = 0
        self.abort_rollout_capture()
        if reset_count == 0:
            raise HiPADCleanBridgeError("no clean temporal instance banks were reset")
        return reset_count

    def close(self) -> None:
        self.abort_rollout_capture()
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
