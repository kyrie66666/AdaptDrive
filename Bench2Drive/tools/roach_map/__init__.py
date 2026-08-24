"""Roach-compatible static-map generation utilities.

The package preserves the complete static-map asset contract while remaining
independent of the original carla-roach RL training stack.
"""

from .asset_schema import (
    ASSET_FORMAT_VERSION,
    GENERATOR_VERSION,
    REQUIRED_LAYERS,
    RUNTIME_REQUIRED_LAYERS,
    GlobalMapMetadata,
    TiledMapMetadata,
    validate_global_asset,
    validate_static_map_asset,
    validate_tiled_asset,
    write_global_asset,
    write_tiled_asset,
)
from .map_rasterizer import (
    GlobalRasterEstimate,
    MapBounds,
    estimate_global_raster_memory,
)

__all__ = [
    "ASSET_FORMAT_VERSION",
    "GENERATOR_VERSION",
    "REQUIRED_LAYERS",
    "RUNTIME_REQUIRED_LAYERS",
    "GlobalMapMetadata",
    "TiledMapMetadata",
    "validate_global_asset",
    "validate_static_map_asset",
    "validate_tiled_asset",
    "write_global_asset",
    "write_tiled_asset",
    "GlobalRasterEstimate",
    "MapBounds",
    "estimate_global_raster_memory",
]
