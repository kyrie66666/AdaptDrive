#!/usr/bin/env python3
"""CARLA-server-free smoke for the minimal Roach global rasterizer."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from roach_map.asset_schema import REQUIRED_LAYERS  # noqa: E402
from roach_map.map_rasterizer import GlobalRoachMapRasterizer, MapBounds  # noqa: E402


class Vector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __rmul__(self, scale):
        return Vector(scale * self.x, scale * self.y, scale * self.z)


class Location(Vector):
    def __add__(self, other):
        return Location(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Location(self.x - other.x, self.y - other.y, self.z - other.z)


class Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch = float(pitch)
        self.yaw = float(yaw)
        self.roll = float(roll)


class Transform:
    def __init__(self, location, rotation):
        self.location = location
        self.rotation = rotation

    def get_forward_vector(self):
        yaw = math.radians(self.rotation.yaw)
        return Vector(math.cos(yaw), math.sin(yaw), 0.0)


class SideWaypoint:
    def __init__(self, x, y, lane_type):
        self.transform = Transform(Location(x, y, 0.0), Rotation(yaw=0.0))
        self.lane_width = 1.5
        self.lane_type = lane_type
        self._left = None
        self._right = None

    def get_left_lane(self):
        return self._left

    def get_right_lane(self):
        return self._right


class Waypoint:
    def __init__(self, x, y, s, marking_type, marking_color, lane_type):
        self.transform = Transform(Location(x, y, 0.0), Rotation(yaw=0.0))
        self.s = float(s)
        self.road_id = 1
        self.section_id = 0
        self.lane_id = -1
        self.lane_width = 3.5
        self.lane_type = lane_type.Driving
        self.is_junction = False
        marking = SimpleNamespace(type=marking_type, color=marking_color)
        self.left_lane_marking = marking
        self.right_lane_marking = marking
        self._next = None
        self._left = SideWaypoint(x, y - 2.5, lane_type.Shoulder)
        self._right = SideWaypoint(x, y + 2.5, lane_type.Parking)
        self._right._right = SideWaypoint(x, y + 4.0, lane_type.Sidewalk)

    def next(self, _distance):
        return [] if self._next is None else [self._next]

    def get_left_lane(self):
        return self._left

    def get_right_lane(self):
        return self._right


class FakeMap:
    def __init__(self, waypoints, extra_topologies=()):
        self._waypoints = waypoints
        self._extra_topologies = list(extra_topologies)

    def get_topology(self):
        return [(self._waypoints[0], self._waypoints[-1])] + self._extra_topologies


class CountingRasterizer(GlobalRoachMapRasterizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.follow_calls = 0

    def _follow_road(self, start_waypoint):
        self.follow_calls += 1
        return super()._follow_road(start_waypoint)


def main() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame

    lane_type = SimpleNamespace(
        NONE=0,
        Broken=1,
        Solid=2,
        SolidBroken=3,
        BrokenSolid=4,
        BrokenBroken=5,
        SolidSolid=6,
    )
    lane_color = SimpleNamespace(Other=0, White=1)
    lane_color.Yellow = 2
    lane_type_kind = SimpleNamespace(Driving=1, Shoulder=2, Parking=3, Sidewalk=4)
    fake_carla = SimpleNamespace(
        LaneMarkingType=lane_type,
        LaneMarkingColor=lane_color,
        LaneType=lane_type_kind,
        Rotation=Rotation,
        Transform=Transform,
    )
    marking_specs = (
        [(lane_type.Broken, lane_color.White)] * 4
        + [(lane_type.Solid, lane_color.White)] * 4
        + [(lane_type.Broken, lane_color.Yellow)] * 4
        + [(lane_type.Solid, lane_color.Yellow)] * 5
    )
    waypoints = [
        Waypoint(x, 0.0, x + 8.0, marking_type, marking_color, lane_type_kind)
        for x, (marking_type, marking_color) in zip(range(-8, 9), marking_specs)
    ]
    for current, following in zip(waypoints, waypoints[1:]):
        current._next = following

    bounds = MapBounds(
        min_x_meters=-16.0,
        min_y_meters=-16.0,
        max_x_meters=16.0,
        max_y_meters=16.0,
        width_in_meters=32.0,
        width_in_pixels=128,
        margin_meters=0.0,
        pixels_per_meter=4.0,
    )
    pygame.init()
    pygame.display.set_mode((1, 1), 0, 32)
    try:
        rasterizer = GlobalRoachMapRasterizer(
            carla_module=fake_carla,
            pygame_module=pygame,
            bounds=bounds,
            lane_precision_meters=1.0,
        )
        masks = rasterizer.rasterize(
            FakeMap(waypoints),
            stopline_vertices=[(Location(-1.0, -3.0), Location(-1.0, 3.0))],
        )
        far_waypoints = [
            Waypoint(x, 1000.0, x - 1000.0, lane_type.Broken, lane_color.White, lane_type_kind)
            for x in range(1000, 1004)
        ]
        for current, following in zip(far_waypoints, far_waypoints[1:]):
            current._next = following
        clipped = CountingRasterizer(
            carla_module=fake_carla,
            pygame_module=pygame,
            bounds=bounds,
            lane_precision_meters=1.0,
            clip_padding_meters=0.0,
        )
        clipped_masks = clipped.rasterize(
            FakeMap(waypoints, extra_topologies=[(far_waypoints[0], far_waypoints[-1])])
        )
    finally:
        pygame.display.quit()
        pygame.quit()

    assert set(masks) == set(REQUIRED_LAYERS)
    for name, mask in masks.items():
        assert mask.shape == (128, 128), (name, mask.shape)
        assert mask.dtype == np.uint8, (name, mask.dtype)
        assert np.count_nonzero(mask) > 0, name
        assert set(np.unique(mask)).issubset({0, 255}), (name, np.unique(mask))
    assert np.count_nonzero(masks["lane_marking_white_broken"]) <= np.count_nonzero(
        masks["lane_marking_all"]
    )
    assert clipped.follow_calls == 1
    assert np.count_nonzero(clipped_masks["road"]) > 0
    print("Roach map rasterizer smoke passed")


if __name__ == "__main__":
    main()
