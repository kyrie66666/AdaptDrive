#!/usr/bin/env python3
"""Fast synthetic smoke for mode-aligned frozen clean speed decoding."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "leaderboard"))

from rl.hipad_clean_speed_decode import decode_mode_aligned_clean_speed


class FakePlanDecoder:
    ego_fut_cmd = 1
    speed_refer = ("temp", "5hz")
    with_rescore = True
    anchor_types = [
        ("speed", "5hz", (0.0, 0.4)),
        ("speed", "5hz", (0.4, 3.0)),
        ("speed", "5hz", (3.0, 999.0)),
    ]

    def rescore(
        self,
        plan_cls,
        plan_reg,
        motion_cls,
        motion_reg,
        det_anchors,
        det_confidence,
        **kwargs,
    ):
        del motion_cls, motion_reg, det_anchors, det_confidence, kwargs
        rescored = plan_cls.clone()
        # The synthetic y displacement stores the lateral-mode index.
        modes = torch.round(plan_reg[:, 0, 0, 1] / 3.0).long()
        partial_collision = modes == 1
        rescored[partial_collision, 2] -= 999.0
        all_collision = modes == 2
        return rescored, all_collision


def _model_outs(num_modes: int = 4, fut_ts: int = 6):
    reg_groups = []
    cls_groups = []
    for area in range(3):
        reg = torch.zeros(1, 1, num_modes, fut_ts, 2)
        for mode in range(num_modes):
            reg[0, 0, mode, :, 0] = float(10 * mode + area + 1)
            reg[0, 0, mode, :, 1] = float(mode)
        reg_groups.append(reg.reshape(1, 1, -1))
        cls = torch.full((1, 1, num_modes), float(area))
        cls_groups.append(cls)

    plan_output = {
        "prediction": [torch.cat(reg_groups, dim=2)],
        "classification": [torch.cat(cls_groups, dim=2)],
    }
    det_output = {
        "prediction": [torch.zeros(1, 1, 11)],
        "classification": [torch.zeros(1, 1, 1)],
    }
    motion_output = {
        "prediction": [torch.zeros(1, 1, 1, fut_ts, 2)],
        "classification": [torch.zeros(1, 1, 1)],
    }
    return det_output, {}, {}, plan_output, motion_output, {}


def main() -> None:
    decoded = decode_mode_aligned_clean_speed(
        FakePlanDecoder(),
        {"gt_ego_fut_cmd": torch.ones(1, 1)},
        _model_outs(),
        num_lateral_modes=4,
        fut_ts=6,
        rescore_chunk_size=4,
    )
    assert decoded.trajectories.shape == (1, 4, 6, 2)
    assert decoded.raw_speed_area_indices.tolist() == [[2, 2, 2, 2]]
    assert decoded.speed_area_indices.tolist() == [[2, 1, 2, 2]]
    assert decoded.rescore_changed.tolist() == [[False, True, False, False]]
    assert decoded.suppressed_area_count.tolist() == [[0, 1, 0, 0]]
    assert decoded.all_collision.tolist() == [[False, False, True, False]]
    assert torch.count_nonzero(decoded.trajectories[0, 2]) == 0
    # Mode 1 has its highest-speed area suppressed, so area 1 must be used.
    expected_x = torch.arange(1, 7, dtype=torch.float32) * 12.0
    expected_y = torch.arange(1, 7, dtype=torch.float32)
    torch.testing.assert_close(decoded.trajectories[0, 1, :, 0], expected_x)
    torch.testing.assert_close(decoded.trajectories[0, 1, :, 1], expected_y)
    print("HiP-AD mode-aligned speed decode synthetic smoke passed")


if __name__ == "__main__":
    main()
