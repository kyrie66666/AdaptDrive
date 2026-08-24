#!/usr/bin/env python3
"""CPU smoke checks for Roach-style BEV semantic target generation."""

from __future__ import annotations

from dataclasses import dataclass
import sys
import tempfile
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCH2DRIVE_ROOT = PROJECT_ROOT / "Bench2Drive"
TOOLS_DIR = BENCH2DRIVE_ROOT / "tools"
LEADERBOARD_ROOT = BENCH2DRIVE_ROOT / "leaderboard"
for path in (str(TOOLS_DIR), str(LEADERBOARD_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from roach_map.asset_schema import GlobalMapMetadata, REQUIRED_LAYERS, write_global_asset  # noqa: E402
from rl.roach_bev_target import (  # noqa: E402
    RoachActorBox,
    RoachBevTargetConfig,
    RoachBevTargetGenerator,
)


@dataclass(frozen=True)
class Location:
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class Rotation:
    yaw: float = 0.0


@dataclass(frozen=True)
class Transform:
    location: Location
    rotation: Rotation


@dataclass(frozen=True)
class Extent:
    x: float
    y: float
    z: float = 1.0


@dataclass(frozen=True)
class Waypoint:
    transform: Transform


def _make_asset(root: Path) -> Path:
    width = 512
    masks = {name: np.zeros((width, width), dtype=np.uint8) for name in REQUIRED_LAYERS}
    masks["road"][190:320, 190:320] = 255
    masks["lane_marking_all"][248:252, 190:320] = 255
    masks["lane_marking_white_broken"][248:252, 235:265] = 255
    metadata = GlobalMapMetadata(
        town_name="TownUnitTarget",
        carla_server_version="0.9.15-test",
        opendrive_sha256="2" * 64,
        pixels_per_meter=5.0,
        margin_meters=100.0,
        waypoint_spacing_meters=2.0,
        lane_precision_meters=0.05,
        world_offset_x_meters=-50.0,
        world_offset_y_meters=-50.0,
        width_in_meters=float(width) / 5.0,
        width_in_pixels=width,
    )
    asset_path = root / "TownUnitTarget.h5"
    write_global_asset(asset_path, masks, metadata, chunk_size_pixels=64)
    return asset_path


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="roach_bev_target_") as tmp_dir:
        root = Path(tmp_dir)
        _make_asset(root)
        cfg = RoachBevTargetConfig(width_in_pixels=64, pixels_ev_to_bottom=16)
        generator = RoachBevTargetGenerator(asset_root=root, config=cfg)

        ego = Transform(Location(0.0, 0.0), Rotation(yaw=0.0))
        route = [Waypoint(Transform(Location(float(x), 0.0), Rotation())) for x in range(0, 35, 5)]
        vehicle = RoachActorBox(
            transform=Transform(Location(10.0, 0.0), Rotation(yaw=0.0)),
            bbox_location=Location(0.0, 0.0),
            bbox_extent=Extent(2.0, 1.0, 1.0),
        )
        red_stopline = [(Location(8.0, -2.0), Location(8.0, 2.0))]
        result = generator.build(
            town_name="TownUnitTarget",
            ego_transform=ego,
            route_waypoints=route,
            vehicle_boxes=[vehicle],
            traffic_light_stopline_segments={"red": red_stopline},
        )
        masks = result["masks"]
        assert masks.shape == (15, 64, 64), masks.shape
        assert masks.dtype == np.uint8
        assert result["channel_names"][0:3] == ("road", "route", "lane")
        assert int(np.count_nonzero(masks[0])) > 0, "road channel should be non-empty"
        assert int(np.count_nonzero(masks[1])) > 0, "route channel should be non-empty"
        assert int(np.count_nonzero(masks[2])) > 0, "lane channel should be non-empty"
        assert 120 in set(np.unique(masks[2]).tolist()), "broken lane should use Roach value 120"
        assert int(np.count_nonzero(masks[3])) > 0, "vehicle history should be non-empty"
        assert int((masks[11:] == 255).sum()) > 0, "red traffic-light history should be encoded"

        result2 = generator.build(
            town_name="TownUnitTarget",
            ego_transform=ego,
            route_waypoints=route,
            vehicle_boxes=[],
            traffic_light_stopline_segments={},
        )
        masks2 = result2["masks"]
        assert int(np.count_nonzero(masks2[3])) > 0, "early history slots should retain previous frame"
        assert int(np.count_nonzero(masks2[6])) == 0, "latest vehicle history slot should reflect empty frame"
        generator.close()

    print("roach_bev_target_smoke: PASS", flush=True)


if __name__ == "__main__":
    main()
