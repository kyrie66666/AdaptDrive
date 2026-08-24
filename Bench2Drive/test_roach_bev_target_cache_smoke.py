#!/usr/bin/env python3
"""Smoke checks for frame-keyed Roach BEV target cache semantics."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCH2DRIVE_ROOT = PROJECT_ROOT / "Bench2Drive"
LEADERBOARD_ROOT = BENCH2DRIVE_ROOT / "leaderboard"
for path in (str(BENCH2DRIVE_ROOT), str(LEADERBOARD_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from rl.roach_bev_target_cache import FrameKeyedRoachBevTargetCache, RoachBevTransientTarget  # noqa: E402


def main() -> None:
    cache = FrameKeyedRoachBevTargetCache(max_entries=2)
    masks = np.zeros((15, 192, 192), dtype=np.uint8)
    masks[0, 10:20, 10:20] = 255
    cache.put(
        RoachBevTransientTarget(
            frame=100,
            masks=masks,
            channel_names=tuple(f"c{i}" for i in range(15)),
            sensor_frame_exact=True,
            town_name="Town01",
        )
    )
    item = cache.pop(100)
    assert item is not None
    assert item["valid"] is True
    assert int(item["frame"]) == 100
    assert cache.pop(100) is None, "cache must be one-shot"

    cache.put(
        RoachBevTransientTarget(
            frame=101,
            masks=masks,
            channel_names=tuple(f"c{i}" for i in range(15)),
            sensor_frame_exact=True,
        )
    )
    mismatch = cache.pop(102)
    assert mismatch is not None
    assert mismatch["valid"] is False
    assert mismatch["error"] == "frame_mismatch"
    assert int(mismatch["expected_frame"]) == 102
    assert int(mismatch["actual_frame"]) == 101

    cache.put(
        RoachBevTransientTarget(
            frame=103,
            masks=None,
            channel_names=tuple(f"c{i}" for i in range(15)),
            sensor_frame_exact=False,
            error="sensor_not_exact",
        )
    )
    invalid = cache.pop(103)
    assert invalid is not None
    assert invalid["valid"] is False
    assert invalid["sensor_frame_exact"] is False
    print("roach_bev_target_cache_smoke: PASS", flush=True)


if __name__ == "__main__":
    main()

