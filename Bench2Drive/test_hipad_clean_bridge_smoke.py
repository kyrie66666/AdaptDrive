#!/usr/bin/env python3
"""CPU-only smoke for clean planning context capture and temporal reset."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "leaderboard"))

from rl.hipad_clean_bridge import HiPADCleanBridgeError, HiPADCleanPlanningBridge


class _Bank:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


class _Refine(nn.Module):
    def __init__(self):
        super().__init__()
        self.plan_reg_branch_spat_2m = nn.Linear(256, 12)


class _Decoder:
    def __init__(self):
        self.plan_refine = [_Refine()]
        self.det_instance_bank_list = [_Bank(), _Bank()]
        self.plan_instance_bank_list = [_Bank()]
        self.run_step = 9


def main() -> None:
    decoder = _Decoder()
    bridge = HiPADCleanPlanningBridge(decoder)
    context = torch.randn(1, 48, 256)

    bridge.begin_rollout_capture("frame-1")
    decoder.plan_refine[-1].plan_reg_branch_spat_2m(context)
    captured = bridge.end_rollout_capture()
    assert captured is context

    # A replay-time direct branch call must not create a fresh capture.
    decoder.plan_refine[-1].plan_reg_branch_spat_2m(torch.randn_like(context))
    try:
        bridge.end_rollout_capture()
    except HiPADCleanBridgeError:
        pass
    else:
        raise AssertionError("out-of-scope branch call was captured")

    assert bridge.reset_temporal_state() == 3
    assert decoder.run_step == 0
    assert all(bank.reset_count == 1 for bank in decoder.det_instance_bank_list)
    assert decoder.plan_instance_bank_list[0].reset_count == 1
    bridge.close()
    print("HiP-AD planning bridge smoke passed")


if __name__ == "__main__":
    main()
