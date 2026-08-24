#!/usr/bin/env python3
"""Fast round-trip smoke for the Roach static-map HDF5 schema."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from roach_map.asset_schema import (  # noqa: E402
    GlobalMapMetadata,
    REQUIRED_LAYERS,
    TiledMapMetadata,
    validate_global_asset,
    validate_static_map_asset,
    validate_tiled_asset,
    write_global_asset,
    write_tiled_asset,
)
from roach_map.map_rasterizer import estimate_global_raster_memory  # noqa: E402


def main() -> None:
    width = 64
    masks = {name: np.zeros((width, width), dtype=np.uint8) for name in REQUIRED_LAYERS}
    for index, name in enumerate(REQUIRED_LAYERS):
        masks[name][index + 1, index + 2] = 255
    masks["road"][8:56, 12:52] = 255
    masks["lane_marking_all"][16:48, 31:33] = 255
    masks["lane_marking_white_broken"][20:24, 31:33] = 255
    metadata = GlobalMapMetadata(
        town_name="TownUnitTest",
        carla_server_version="0.9.15-test",
        opendrive_sha256="0" * 64,
        pixels_per_meter=5.0,
        margin_meters=100.0,
        waypoint_spacing_meters=2.0,
        lane_precision_meters=0.05,
        world_offset_x_meters=-100.0,
        world_offset_y_meters=-100.0,
        width_in_meters=12.8,
        width_in_pixels=width,
    )

    with tempfile.TemporaryDirectory(prefix="roach_map_schema_") as tmp_dir:
        asset_path = Path(tmp_dir) / "TownUnitTest.h5"
        manifest = write_global_asset(asset_path, masks, metadata, chunk_size_pixels=16)
        assert manifest["asset_sha256"]
        report = validate_global_asset(asset_path)
        assert report["valid"], report["errors"]
        assert report["town_name"] == "TownUnitTest"
        assert report["width_in_pixels"] == width
        assert report["layers"]["road"]["nonzero_pixels"] > 0

        tiled_path = Path(tmp_dir) / "TownUnitTestTiled.h5"
        tiled_metadata = TiledMapMetadata(
            town_name="TownUnitTestTiled",
            carla_server_version="0.9.15-test",
            opendrive_sha256="1" * 64,
            pixels_per_meter=5.0,
            margin_meters=100.0,
            waypoint_spacing_meters=2.0,
            lane_precision_meters=0.05,
            world_offset_x_meters=-100.0,
            world_offset_y_meters=-100.0,
            width_in_meters=12.8,
            width_in_pixels=width,
            tile_size_pixels=32,
            tile_count_x=2,
            tile_count_y=2,
        )

        def iter_tiles():
            tile_id = 0
            for row_start in range(0, width, 32):
                row_end = min(width, row_start + 32)
                for col_start in range(0, width, 32):
                    col_end = min(width, col_start + 32)
                    yield {
                        "tile_id": f"tile_{tile_id}",
                        "row_start": row_start,
                        "row_end": row_end,
                        "col_start": col_start,
                        "col_end": col_end,
                        "masks": {
                            name: array[row_start:row_end, col_start:col_end]
                            for name, array in masks.items()
                        },
                    }
                    tile_id += 1

        tiled_manifest = write_tiled_asset(
            tiled_path,
            tiled_metadata,
            iter_tiles(),
            chunk_size_pixels=16,
        )
        assert tiled_manifest["asset_sha256"]
        tiled_report = validate_tiled_asset(tiled_path)
        assert tiled_report["valid"], tiled_report["errors"]
        assert tiled_report["storage_mode"] == "tiled"
        assert tiled_report["tile_count_written"] == 4
        auto_report = validate_static_map_asset(tiled_path)
        assert auto_report["valid"], auto_report["errors"]

    estimate = estimate_global_raster_memory(3255, layer_count=3)
    assert estimate.raw_surface_bytes == 3255 * 3255 * 3 * 4
    assert estimate.estimated_peak_bytes > estimate.raw_surface_bytes
    print("Roach static-map schema smoke passed")


if __name__ == "__main__":
    main()
