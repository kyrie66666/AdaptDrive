"""Frame-keyed transient cache for Roach BEV semantic targets."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np


@dataclass
class RoachBevTransientTarget:
    frame: int
    masks: Optional[np.ndarray]
    channel_names: Tuple[str, ...]
    sensor_frame_exact: bool
    town_name: str = ""
    error: str = ""
    pixels_per_meter: float = 5.0
    pixels_ev_to_bottom: int = 40
    width_in_pixels: int = 192

    @property
    def valid(self) -> bool:
        return self.masks is not None and not self.error and bool(self.sensor_frame_exact)

    def as_dict(self) -> Dict[str, object]:
        return {
            "frame": int(self.frame),
            "masks": self.masks,
            "channel_names": tuple(self.channel_names),
            "sensor_frame_exact": bool(self.sensor_frame_exact),
            "town_name": str(self.town_name),
            "error": str(self.error),
            "pixels_per_meter": float(self.pixels_per_meter),
            "pixels_ev_to_bottom": int(self.pixels_ev_to_bottom),
            "width_in_pixels": int(self.width_in_pixels),
            "valid": bool(self.valid),
        }


class FrameKeyedRoachBevTargetCache:
    """Small one-shot cache keyed by CARLA sensor_frame."""

    def __init__(self, max_entries: int = 8) -> None:
        self.max_entries = max(1, int(max_entries))
        self._items: "OrderedDict[int, RoachBevTransientTarget]" = OrderedDict()

    def clear(self) -> None:
        self._items.clear()

    def put(self, target: RoachBevTransientTarget) -> None:
        frame = int(target.frame)
        self._items[frame] = target
        self._items.move_to_end(frame)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)

    def pop(self, expected_frame: int) -> Optional[Dict[str, object]]:
        expected = int(expected_frame)
        target = self._items.pop(expected, None)
        if target is not None:
            return target.as_dict()
        if not self._items:
            return None
        frame, target = self._items.popitem(last=False)
        payload = target.as_dict()
        payload["expected_frame"] = expected
        payload["actual_frame"] = int(frame)
        payload["error"] = "frame_mismatch"
        payload["valid"] = False
        return payload

    def latest(self) -> Optional[Mapping[str, object]]:
        if not self._items:
            return None
        return next(reversed(self._items.values())).as_dict()

    def __len__(self) -> int:
        return len(self._items)

