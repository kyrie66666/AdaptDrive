#!/usr/bin/env python3
"""Generate complete Roach-compatible static Town map assets.

By default the tool owns one CARLA server for the duration of the command,
matching the original project's self-contained workflow while avoiding global
``killall``. ``--connect-existing`` is available for debugging. Normal-size
Towns use the reference global raster. Oversized Towns can use tiled generation
while preserving the same global-shaped HDF5 layer datasets exposed to runtime
readers.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from roach_map.asset_schema import (  # noqa: E402
    GlobalMapMetadata,
    REQUIRED_LAYERS,
    TiledMapMetadata,
    sha256_bytes,
    validate_static_map_asset,
    validate_global_asset,
    write_global_asset,
    write_tiled_asset,
)
from roach_map.carla_server import OwnedCarlaServer  # noqa: E402
from roach_map.map_rasterizer import (  # noqa: E402
    GlobalRoachMapRasterizer,
    compute_map_bounds,
    estimate_global_raster_memory,
)
from roach_map.traffic_light_geometry import collect_stopline_vertices  # noqa: E402


BENCH2DRIVE_TOWNS = (
    "Town01",
    "Town02",
    "Town03",
    "Town04",
    "Town05",
    "Town06",
    "Town07",
    "Town10HD",
    "Town11",
    "Town12",
    "Town13",
    "Town15",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--towns", nargs="+", required=True, help="CARLA Town names to generate")
    parser.add_argument("--host", default="127.0.0.1", help="Owned or existing CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="Owned or existing CARLA server RPC port")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--world-load-attempts", type=int, default=3)
    parser.add_argument("--world-load-settle-seconds", type=float, default=20.0)
    parser.add_argument("--carla-root", default=os.environ.get("CARLA_ROOT", ""))
    parser.add_argument(
        "--connect-existing",
        action="store_true",
        help="Do not launch CARLA; connect to the supplied host/port instead",
    )
    parser.add_argument("--server-startup-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--server-shutdown-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--server-warmup-seconds", type=float, default=30.0)
    parser.add_argument("--server-log", default="")
    parser.add_argument(
        "--cuda-visible-devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        help="CUDA_VISIBLE_DEVICES for the owned CARLA process",
    )
    parser.add_argument("--graphics-adapter", type=int, default=None)
    parser.add_argument(
        "--carla-launch-user",
        default=os.environ.get("CARLA_LAUNCH_USER", "carla" if os.getuid() == 0 else ""),
    )
    parser.add_argument("--xdg-runtime-dir", default=os.environ.get("XDG_RUNTIME_DIR", ""))
    parser.add_argument("--vk-icd-filenames", default=os.environ.get("VK_ICD_FILENAMES", ""))
    parser.add_argument("--display", default=os.environ.get("DISPLAY", ""))
    parser.add_argument(
        "--server-extra-arg",
        action="append",
        default=[],
        help="Additional CARLA argument; repeat the flag for multiple arguments",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("ROACH_BEV_MAP_ROOT", ""),
        help="Asset directory; defaults to ROACH_BEV_MAP_ROOT",
    )
    parser.add_argument("--pixels-per-meter", type=float, default=5.0)
    parser.add_argument("--margin-meters", type=float, default=100.0)
    parser.add_argument("--waypoint-spacing-meters", type=float, default=2.0)
    parser.add_argument("--lane-precision-meters", type=float, default=0.05)
    parser.add_argument("--max-estimated-memory-gb", type=float, default=32.0)
    parser.add_argument("--chunk-size-pixels", type=int, default=1024)
    parser.add_argument("--compression-level", type=int, default=4)
    parser.add_argument(
        "--storage-mode",
        choices=("auto", "global", "tiled"),
        default="auto",
        help="auto uses global when it passes the memory gate, tiled otherwise",
    )
    parser.add_argument(
        "--tile-size-pixels",
        type=int,
        default=8192,
        help="Generation tile edge for storage_mode=tiled/auto fallback",
    )
    parser.add_argument(
        "--tile-overlap-pixels",
        type=int,
        default=0,
        help="Reserved in metadata; current writer exposes seamless global-shaped datasets",
    )
    parser.add_argument(
        "--tile-clip-padding-meters",
        type=float,
        default=100.0,
        help="Expanded world-space margin used to cull topology while rasterizing each tile",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip an existing Town asset instead of failing; incompatible with --overwrite",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load each Town and report bounds/memory without allocating surfaces",
    )
    parser.add_argument(
        "--allow-unsafe-global",
        action="store_true",
        help="Override memory gate; not recommended and never implied by --overwrite",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print topology progress every N entries; 0 disables periodic progress",
    )
    return parser.parse_args(argv)


def _setup_carla_python_api(carla_root: str) -> None:
    try:
        import carla  # noqa: F401

        return
    except ImportError:
        pass

    if not carla_root:
        raise RuntimeError(
            "CARLA Python API is not importable. Set CARLA_ROOT or pass --carla-root."
        )
    root = Path(carla_root).expanduser().resolve()
    candidates = sorted(glob.glob(str(root / "PythonAPI/carla/dist/carla-*.egg")))
    python_api_paths = [root / "PythonAPI", root / "PythonAPI/carla"]
    for path in [Path(item) for item in candidates] + python_api_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        import carla  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"Failed to import CARLA Python API from CARLA_ROOT={root}; "
            f"examined eggs={candidates}"
        ) from exc


def _peak_rss_gib() -> float:
    # Linux ru_maxrss is KiB. This tool is Linux/CARLA-specific.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2))


def _current_rss_gib() -> Optional[float]:
    try:
        import psutil
    except ImportError:
        return None
    return float(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3))


def _normalize_town_name(map_name: str) -> str:
    return str(map_name).rstrip("/").split("/")[-1]


def _load_world_verified(client, args: argparse.Namespace, requested_town: str):
    """Load a Town and wait until CARLA reports the requested map."""

    last_map_name = ""
    last_error = None
    for attempt in range(1, int(args.world_load_attempts) + 1):
        print(
            json.dumps(
                {
                    "event": "load_world_attempt",
                    "town": requested_town,
                    "attempt": attempt,
                    "max_attempts": int(args.world_load_attempts),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        try:
            loaded_world = client.load_world(requested_town, reset_settings=False)
        except Exception as exc:
            last_error = exc
            print(
                json.dumps(
                    {
                        "event": "load_world_error",
                        "town": requested_town,
                        "attempt": attempt,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            continue

        last_map_name = _normalize_town_name(loaded_world.get_map().name)
        if last_map_name == requested_town:
            return loaded_world

        deadline = time.monotonic() + float(args.world_load_settle_seconds)
        while time.monotonic() < deadline:
            try:
                current_world = client.get_world()
                last_map_name = _normalize_town_name(current_world.get_map().name)
                if last_map_name == requested_town:
                    return current_world
            except Exception as exc:
                last_error = exc
            time.sleep(1.0)
        print(
            json.dumps(
                {
                    "event": "load_world_mismatch",
                    "town": requested_town,
                    "reported_map": last_map_name,
                    "attempt": attempt,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    try:
        available_maps = sorted(_normalize_town_name(value) for value in client.get_available_maps())
    except Exception as exc:
        available_maps = [f"unavailable: {type(exc).__name__}: {exc}"]
    raise RuntimeError(
        f"CARLA did not switch to {requested_town!r} after {args.world_load_attempts} attempts; "
        f"last reported map={last_map_name!r}; last_error={last_error}; "
        f"available_maps={available_maps}"
    )


def _progress_reporter(town: str, every: int):
    last_reported = {"index": 0}

    def report(index: int, total: int) -> None:
        should_report = index == total or (every > 0 and index - last_reported["index"] >= every)
        if should_report:
            print(
                json.dumps(
                    {
                        "event": "raster_progress",
                        "town": town,
                        "topology_index": index,
                        "topology_total": total,
                        "current_rss_gib": (
                            round(_current_rss_gib(), 3) if _current_rss_gib() is not None else None
                        ),
                        "peak_rss_gib": round(_peak_rss_gib(), 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            last_reported["index"] = index

    return report


def _tile_windows(width: int, tile_size: int):
    width = int(width)
    tile_size = int(tile_size)
    tile_index = 0
    for row_start in range(0, width, tile_size):
        row_end = min(width, row_start + tile_size)
        for col_start in range(0, width, tile_size):
            col_end = min(width, col_start + tile_size)
            yield tile_index, row_start, row_end, col_start, col_end
            tile_index += 1


def _tile_bounds_from_global(bounds, row_start: int, col_start: int, tile_size: int):
    ppm = float(bounds.pixels_per_meter)
    tile_width_m = float(tile_size) / ppm
    min_x = float(bounds.min_x_meters) + float(col_start) / ppm
    min_y = float(bounds.min_y_meters) + float(row_start) / ppm
    return type(bounds)(
        min_x_meters=min_x,
        min_y_meters=min_y,
        max_x_meters=min_x + tile_width_m,
        max_y_meters=min_y + tile_width_m,
        width_in_meters=tile_width_m,
        width_in_pixels=int(tile_size),
        margin_meters=float(bounds.margin_meters),
        pixels_per_meter=ppm,
    )


def _tile_progress_reporter(town: str, tile_id: str, every: int):
    base_reporter = _progress_reporter(town, every)

    def report(index: int, total: int) -> None:
        base_reporter(index, total)
        if index == total:
            print(
                json.dumps(
                    {
                        "event": "tile_raster_complete",
                        "town": town,
                        "tile_id": tile_id,
                        "topology_total": total,
                        "current_rss_gib": (
                            round(_current_rss_gib(), 3) if _current_rss_gib() is not None else None
                        ),
                        "peak_rss_gib": round(_peak_rss_gib(), 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    return report


def _make_metadata(client, carla_map, args: argparse.Namespace, town: str, bounds, stopline_count: int):
    opendrive = carla_map.to_opendrive()
    return dict(
        town_name=town,
        carla_server_version=str(client.get_server_version()),
        opendrive_sha256=sha256_bytes(opendrive.encode("utf-8")),
        pixels_per_meter=float(args.pixels_per_meter),
        margin_meters=float(args.margin_meters),
        waypoint_spacing_meters=float(args.waypoint_spacing_meters),
        lane_precision_meters=float(args.lane_precision_meters),
        world_offset_x_meters=float(bounds.min_x_meters),
        world_offset_y_meters=float(bounds.min_y_meters),
        width_in_meters=float(bounds.width_in_meters),
        width_in_pixels=int(bounds.width_in_pixels),
        stopline_segment_count=int(stopline_count),
    )


def _select_storage_mode(args: argparse.Namespace, estimate) -> str:
    if args.storage_mode == "global":
        return "global"
    if args.storage_mode == "tiled":
        return "tiled"
    if estimate.estimated_peak_gib <= float(args.max_estimated_memory_gb):
        return "global"
    if bool(args.allow_unsafe_global):
        return "global"
    return "tiled"


def _init_pygame():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame

    pygame.init()
    pygame.display.set_mode((1, 1), 0, 32)
    return pygame


def _log_pygame_surface_format(pygame, town: str) -> None:
    surface_probe = pygame.Surface((1, 1), depth=32)
    print(
        json.dumps(
            {
                "event": "pygame_surface_format",
                "town": town,
                "bitsize": surface_probe.get_bitsize(),
                "bytesize": surface_probe.get_bytesize(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _write_global_map_asset(
    client,
    carla,
    carla_map,
    world,
    args: argparse.Namespace,
    requested_town: str,
    bounds,
    asset_path: Path,
) -> None:
    pygame = _init_pygame()
    _log_pygame_surface_format(pygame, requested_town)

    started = time.monotonic()
    try:
        stopline_vertices = collect_stopline_vertices(world, carla)
        rasterizer = GlobalRoachMapRasterizer(
            carla_module=carla,
            pygame_module=pygame,
            bounds=bounds,
            lane_precision_meters=args.lane_precision_meters,
            progress_callback=_progress_reporter(requested_town, args.progress_every),
        )
        masks = rasterizer.rasterize(carla_map, stopline_vertices=stopline_vertices)
    finally:
        pygame.display.quit()
        pygame.quit()
    raster_seconds = time.monotonic() - started

    metadata = GlobalMapMetadata(
        **_make_metadata(client, carla_map, args, requested_town, bounds, len(stopline_vertices))
    )
    write_started = time.monotonic()
    manifest = write_global_asset(
        asset_path,
        masks,
        metadata,
        overwrite=args.overwrite,
        chunk_size_pixels=args.chunk_size_pixels,
        compression_level=args.compression_level,
    )
    write_seconds = time.monotonic() - write_started
    print(
        json.dumps(
            {
                "event": "asset_complete",
                "town": requested_town,
                "storage_mode": "global",
                "asset_path": str(asset_path),
                "asset_sha256": manifest["asset_sha256"],
                "asset_size_bytes": manifest["asset_size_bytes"],
                "raster_seconds": round(raster_seconds, 3),
                "write_seconds": round(write_seconds, 3),
                "current_rss_gib": (
                    round(_current_rss_gib(), 3) if _current_rss_gib() is not None else None
                ),
                "peak_rss_gib": round(_peak_rss_gib(), 3),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _write_tiled_map_asset(
    client,
    carla,
    carla_map,
    world,
    args: argparse.Namespace,
    requested_town: str,
    bounds,
    asset_path: Path,
) -> None:
    if int(args.tile_size_pixels) <= 0:
        raise ValueError("--tile-size-pixels must be positive")
    if int(args.tile_overlap_pixels) != 0:
        raise NotImplementedError(
            "--tile-overlap-pixels must remain 0 for roach_static_map_v1 tiled generation; "
            "runtime-facing datasets already expose seamless global coordinates"
        )

    width = int(bounds.width_in_pixels)
    tile_size = int(args.tile_size_pixels)
    tile_count_x = (width + tile_size - 1) // tile_size
    tile_count_y = tile_count_x
    tile_estimate = estimate_global_raster_memory(
        min(tile_size, width),
        layer_count=len(REQUIRED_LAYERS),
    )
    if tile_estimate.estimated_peak_gib > float(args.max_estimated_memory_gb) and not args.allow_unsafe_global:
        raise MemoryError(
            f"Refusing tiled raster for {requested_town}: per-tile estimated peak "
            f"{tile_estimate.estimated_peak_gib:.2f} GiB exceeds limit "
            f"{args.max_estimated_memory_gb:.2f} GiB. Reduce --tile-size-pixels or raise the "
            "explicit memory limit."
        )

    print(
        json.dumps(
            {
                "event": "tiled_preflight",
                "town": requested_town,
                "tile_size_pixels": tile_size,
                "tile_overlap_pixels": int(args.tile_overlap_pixels),
                "tile_count_x": tile_count_x,
                "tile_count_y": tile_count_y,
                "tile_count_total": tile_count_x * tile_count_y,
                "per_tile_memory_estimate": tile_estimate.as_dict(),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    pygame = _init_pygame()
    _log_pygame_surface_format(pygame, requested_town)
    stopline_vertices = collect_stopline_vertices(world, carla)
    metadata = TiledMapMetadata(
        **_make_metadata(client, carla_map, args, requested_town, bounds, len(stopline_vertices)),
        tile_size_pixels=tile_size,
        tile_overlap_pixels=int(args.tile_overlap_pixels),
        tile_count_x=tile_count_x,
        tile_count_y=tile_count_y,
    )

    def tile_iterator():
        for tile_index, row_start, row_end, col_start, col_end in _tile_windows(width, tile_size):
            tile_id = f"r{row_start}_c{col_start}"
            tile_bounds = _tile_bounds_from_global(bounds, row_start, col_start, tile_size)
            started = time.monotonic()
            rasterizer = GlobalRoachMapRasterizer(
                carla_module=carla,
                pygame_module=pygame,
                bounds=tile_bounds,
                lane_precision_meters=args.lane_precision_meters,
                clip_padding_meters=args.tile_clip_padding_meters,
                progress_callback=_tile_progress_reporter(requested_town, tile_id, args.progress_every),
            )
            masks = rasterizer.rasterize(carla_map, stopline_vertices=stopline_vertices)
            raster_seconds = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "event": "tile_ready",
                        "town": requested_town,
                        "tile_index": tile_index,
                        "tile_id": tile_id,
                        "row_start": row_start,
                        "row_end": row_end,
                        "col_start": col_start,
                        "col_end": col_end,
                        "raster_seconds": round(raster_seconds, 3),
                        "current_rss_gib": (
                            round(_current_rss_gib(), 3) if _current_rss_gib() is not None else None
                        ),
                        "peak_rss_gib": round(_peak_rss_gib(), 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            yield {
                "tile_id": tile_id,
                "row_start": row_start,
                "row_end": row_end,
                "col_start": col_start,
                "col_end": col_end,
                "masks": masks,
            }

    started = time.monotonic()
    try:
        manifest = write_tiled_asset(
            asset_path,
            metadata,
            tile_iterator(),
            overwrite=args.overwrite,
            chunk_size_pixels=args.chunk_size_pixels,
            compression_level=args.compression_level,
        )
    finally:
        pygame.display.quit()
        pygame.quit()
    total_seconds = time.monotonic() - started
    print(
        json.dumps(
            {
                "event": "asset_complete",
                "town": requested_town,
                "storage_mode": "tiled",
                "asset_path": str(asset_path),
                "asset_sha256": manifest["asset_sha256"],
                "asset_size_bytes": manifest["asset_size_bytes"],
                "tile_count_written": len(manifest.get("tiles", [])),
                "total_seconds": round(total_seconds, 3),
                "current_rss_gib": (
                    round(_current_rss_gib(), 3) if _current_rss_gib() is not None else None
                ),
                "peak_rss_gib": round(_peak_rss_gib(), 3),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def generate_one(client, carla, args: argparse.Namespace, town: str) -> None:
    requested_town = str(town)
    if requested_town not in BENCH2DRIVE_TOWNS:
        raise ValueError(
            f"Unsupported/non-canonical Town {requested_town!r}; expected one of {BENCH2DRIVE_TOWNS}"
        )

    if not args.dry_run:
        output_dir = Path(args.output_dir).expanduser().resolve()
        asset_path = output_dir / f"{requested_town}.h5"
        if asset_path.exists():
            if args.skip_existing:
                report = validate_static_map_asset(asset_path, scan_values=False)
                if not report["valid"]:
                    raise RuntimeError(
                        f"Refusing to skip invalid existing asset {asset_path}: {report['errors']}"
                    )
                if report["town_name"] != requested_town:
                    raise RuntimeError(
                        f"Refusing to skip {asset_path}: embedded town_name="
                        f"{report['town_name']!r}, expected {requested_town!r}"
                    )
                print(
                    json.dumps(
                        {
                            "event": "skip_existing_asset",
                            "town": requested_town,
                            "asset_path": str(asset_path),
                            "asset_sha256": report.get("asset_sha256"),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                return
            if not args.overwrite:
                raise FileExistsError(
                    f"Asset already exists; pass --skip-existing or --overwrite: {asset_path}"
                )

    print(json.dumps({"event": "load_world", "town": requested_town}), flush=True)
    world = _load_world_verified(client, args, requested_town)
    carla_map = world.get_map()
    loaded_town = _normalize_town_name(carla_map.name)
    if loaded_town != requested_town:
        raise RuntimeError(
            f"CARLA loaded map {carla_map.name!r}, expected canonical Town {requested_town!r}"
        )

    bounds = compute_map_bounds(
        carla_map,
        pixels_per_meter=args.pixels_per_meter,
        margin_meters=args.margin_meters,
        waypoint_spacing_meters=args.waypoint_spacing_meters,
    )
    estimate = estimate_global_raster_memory(bounds.width_in_pixels, layer_count=len(REQUIRED_LAYERS))
    preflight = {
        "event": "global_preflight",
        "town": requested_town,
        "bounds": bounds.as_dict(),
        "memory_estimate": estimate.as_dict(),
        "max_estimated_memory_gb": float(args.max_estimated_memory_gb),
        "layers": list(REQUIRED_LAYERS),
    }
    print(json.dumps(preflight, sort_keys=True), flush=True)
    if args.dry_run:
        return

    storage_mode = _select_storage_mode(args, estimate)
    print(
        json.dumps(
            {
                "event": "storage_decision",
                "town": requested_town,
                "requested_storage_mode": args.storage_mode,
                "selected_storage_mode": storage_mode,
                "allow_unsafe_global": bool(args.allow_unsafe_global),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if (
        storage_mode == "global"
        and estimate.estimated_peak_gib > float(args.max_estimated_memory_gb)
        and not args.allow_unsafe_global
    ):
        raise MemoryError(
            f"Refusing global raster for {requested_town}: estimated peak "
            f"{estimate.estimated_peak_gib:.2f} GiB exceeds limit "
            f"{args.max_estimated_memory_gb:.2f} GiB. Use --storage-mode tiled or "
            "--allow-unsafe-global is an explicit last-resort override."
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    asset_path = output_dir / f"{requested_town}.h5"
    if storage_mode == "global":
        _write_global_map_asset(client, carla, carla_map, world, args, requested_town, bounds, asset_path)
        return
    if storage_mode == "tiled":
        _write_tiled_map_asset(client, carla, carla_map, world, args, requested_town, bounds, asset_path)
        return
    raise AssertionError(f"Unhandled storage mode {storage_mode!r}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if not args.dry_run and not args.output_dir:
        raise RuntimeError("--output-dir or ROACH_BEV_MAP_ROOT is required unless --dry-run is used")
    if args.max_estimated_memory_gb <= 0:
        raise ValueError("--max-estimated-memory-gb must be positive")
    if args.chunk_size_pixels <= 0:
        raise ValueError("--chunk-size-pixels must be positive")
    if args.tile_size_pixels <= 0:
        raise ValueError("--tile-size-pixels must be positive")
    if args.tile_overlap_pixels < 0:
        raise ValueError("--tile-overlap-pixels must be non-negative")
    if args.tile_clip_padding_meters < 0:
        raise ValueError("--tile-clip-padding-meters must be non-negative")
    if not 0 <= args.compression_level <= 9:
        raise ValueError("--compression-level must be in [0, 9]")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    if args.world_load_attempts <= 0:
        raise ValueError("--world-load-attempts must be positive")
    if args.world_load_settle_seconds <= 0:
        raise ValueError("--world-load-settle-seconds must be positive")
    if args.server_startup_timeout_seconds <= 0:
        raise ValueError("--server-startup-timeout-seconds must be positive")
    if args.server_shutdown_timeout_seconds <= 0:
        raise ValueError("--server-shutdown-timeout-seconds must be positive")
    if args.server_warmup_seconds < 0:
        raise ValueError("--server-warmup-seconds must be non-negative")
    if args.overwrite and args.skip_existing:
        raise ValueError("--overwrite and --skip-existing are mutually exclusive")

    _setup_carla_python_api(args.carla_root)
    import carla

    def run_with_client(client) -> None:
        client.set_timeout(float(args.timeout_seconds))
        print(
            json.dumps(
                {
                    "event": "carla_connected",
                    "host": args.host,
                    "port": int(args.port),
                    "server_version": str(client.get_server_version()),
                    "client_version": str(client.get_client_version()),
                    "server_mode": "existing" if args.connect_existing else "owned",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        for town in args.towns:
            generate_one(client, carla, args, town)

    if args.connect_existing:
        run_with_client(carla.Client(args.host, int(args.port)))
        return

    if not args.carla_root:
        raise RuntimeError("CARLA_ROOT or --carla-root is required unless --connect-existing is used")
    server_log = Path(args.server_log) if args.server_log else None
    server = OwnedCarlaServer(
        carla_module=carla,
        carla_root=Path(args.carla_root),
        host=args.host,
        port=args.port,
        startup_timeout_seconds=args.server_startup_timeout_seconds,
        shutdown_timeout_seconds=args.server_shutdown_timeout_seconds,
        server_log=server_log,
        cuda_visible_devices=args.cuda_visible_devices,
        graphics_adapter=args.graphics_adapter,
        launch_user=args.carla_launch_user,
        runtime_dir=Path(args.xdg_runtime_dir) if args.xdg_runtime_dir else None,
        vk_icd_filenames=args.vk_icd_filenames,
        display=args.display,
        server_warmup_seconds=args.server_warmup_seconds,
        extra_args=args.server_extra_arg,
    )
    print(
        json.dumps(
            {
                "event": "launch_owned_carla",
                "command": server.command(),
                "server_log": str(server.server_log),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    with server as client:
        run_with_client(client)


if __name__ == "__main__":
    main()
