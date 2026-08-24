#!/usr/bin/env python3
"""Render quick PNG previews for generated Roach static-map assets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from roach_map.asset_schema import REQUIRED_LAYERS, manifest_path_for  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", nargs="+", help="HDF5 asset files or directories")
    parser.add_argument("--output-dir", required=True, help="Directory for PNG previews")
    parser.add_argument(
        "--mode",
        choices=("auto", "overview", "crop"),
        default="auto",
        help="auto uses crop for tiled assets and overview for global assets",
    )
    parser.add_argument(
        "--max-overview-pixels",
        type=int,
        default=2048,
        help="Maximum side length for overview previews",
    )
    parser.add_argument("--crop-size-pixels", type=int, default=2048)
    parser.add_argument("--crop-row", type=int, default=None)
    parser.add_argument("--crop-col", type=int, default=None)
    return parser.parse_args(argv)


def _expand_assets(values: Sequence[str]):
    paths = []
    for value in values:
        path = Path(value).expanduser()
        if path.is_dir():
            paths.extend(sorted(path.glob("*.h5")))
        else:
            paths.append(path)
    deduplicated = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduplicated.append(resolved)
    return deduplicated


def _save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(image).save(path)
        return
    except ImportError:
        pass
    try:
        import imageio.v2 as imageio

        imageio.imwrite(path, image)
        return
    except ImportError:
        pass
    try:
        import cv2

        cv2.imwrite(str(path), image[:, :, ::-1])
        return
    except ImportError as exc:
        raise RuntimeError("Saving PNG previews requires pillow, imageio or opencv-python") from exc


def _layer(handle, name: str, selection) -> np.ndarray:
    if name not in handle:
        return np.zeros(handle["road"][selection].shape, dtype=np.uint8)
    return np.asarray(handle[name][selection], dtype=np.uint8)


def _compose_rgb(handle, selection) -> np.ndarray:
    road = _layer(handle, "road", selection) > 0
    shoulder = _layer(handle, "shoulder", selection) > 0
    parking = _layer(handle, "parking", selection) > 0
    sidewalk = _layer(handle, "sidewalk", selection) > 0
    lane_all = _layer(handle, "lane_marking_all", selection) > 0
    stopline = _layer(handle, "stopline", selection) > 0

    image = np.zeros((*road.shape, 3), dtype=np.uint8)
    image[road] = np.maximum(image[road], np.asarray([70, 70, 70], dtype=np.uint8))
    image[shoulder] = np.maximum(image[shoulder], np.asarray([40, 120, 65], dtype=np.uint8))
    image[parking] = np.maximum(image[parking], np.asarray([145, 120, 40], dtype=np.uint8))
    image[sidewalk] = np.maximum(image[sidewalk], np.asarray([60, 80, 130], dtype=np.uint8))
    image[lane_all] = np.asarray([240, 240, 210], dtype=np.uint8)
    image[stopline] = np.asarray([240, 40, 40], dtype=np.uint8)
    return image


def _read_manifest(asset_path: Path):
    manifest_path = manifest_path_for(asset_path)
    if not manifest_path.is_file():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _best_tiled_crop(asset_path: Path, width: int, crop_size: int) -> Tuple[int, int, int, int]:
    manifest = _read_manifest(asset_path)
    tiles = [] if manifest is None else list(manifest.get("tiles", []))
    if not tiles:
        center = width // 2
        half = crop_size // 2
        row_start = max(0, min(width - crop_size, center - half))
        col_start = row_start
        return row_start, min(width, row_start + crop_size), col_start, min(width, col_start + crop_size)
    best = max(
        tiles,
        key=lambda tile: int(tile.get("nonzero_by_layer", {}).get("road", 0)),
    )
    row_center = (int(best["row_start"]) + int(best["row_end"])) // 2
    col_center = (int(best["col_start"]) + int(best["col_end"])) // 2
    half = int(crop_size) // 2
    row_start = max(0, min(max(0, width - crop_size), row_center - half))
    col_start = max(0, min(max(0, width - crop_size), col_center - half))
    return row_start, min(width, row_start + crop_size), col_start, min(width, col_start + crop_size)


def _crop_from_args(args: argparse.Namespace, width: int) -> Optional[Tuple[int, int, int, int]]:
    if args.crop_row is None and args.crop_col is None:
        return None
    if args.crop_row is None or args.crop_col is None:
        raise ValueError("--crop-row and --crop-col must be supplied together")
    crop_size = min(int(args.crop_size_pixels), width)
    row_start = max(0, min(max(0, width - crop_size), int(args.crop_row)))
    col_start = max(0, min(max(0, width - crop_size), int(args.crop_col)))
    return row_start, min(width, row_start + crop_size), col_start, min(width, col_start + crop_size)


def render_asset(asset_path: Path, args: argparse.Namespace) -> dict:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to visualize Roach static-map assets") from exc

    with h5py.File(str(asset_path), "r", libver="latest", swmr=True) as handle:
        missing = [name for name in REQUIRED_LAYERS if name not in handle]
        if missing:
            raise RuntimeError(f"{asset_path} is missing required layers: {missing}")
        attrs = dict(handle.attrs)
        town = str(attrs.get("town_name", asset_path.stem))
        storage_mode = str(attrs.get("storage_mode", ""))
        width = int(attrs.get("width_in_pixels", handle["road"].shape[0]))
        explicit_crop = _crop_from_args(args, width)
        mode = args.mode
        if mode == "auto":
            mode = "crop" if storage_mode == "tiled" else "overview"

        if mode == "overview":
            stride = max(1, int(np.ceil(width / float(args.max_overview_pixels))))
            selection = (slice(0, width, stride), slice(0, width, stride))
            suffix = f"overview_s{stride}"
            crop_window = None
        else:
            crop_size = min(int(args.crop_size_pixels), width)
            if explicit_crop is None:
                crop_window = _best_tiled_crop(asset_path, width, crop_size)
            else:
                crop_window = explicit_crop
            row_start, row_end, col_start, col_end = crop_window
            selection = (slice(row_start, row_end), slice(col_start, col_end))
            suffix = f"crop_r{row_start}_c{col_start}_{row_end - row_start}x{col_end - col_start}"

        image = _compose_rgb(handle, selection)
        output_path = Path(args.output_dir).expanduser().resolve() / f"{town}_{suffix}.png"
        _save_png(output_path, image)
        return {
            "asset_path": str(asset_path),
            "town_name": town,
            "storage_mode": storage_mode,
            "width_in_pixels": width,
            "mode": mode,
            "crop_window": list(crop_window) if mode == "crop" else None,
            "output_path": str(output_path),
            "output_shape": list(image.shape),
        }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.max_overview_pixels <= 0:
        raise ValueError("--max-overview-pixels must be positive")
    if args.crop_size_pixels <= 0:
        raise ValueError("--crop-size-pixels must be positive")
    asset_paths = _expand_assets(args.assets)
    if not asset_paths:
        raise RuntimeError("No .h5 assets matched the supplied paths")

    failed = False
    for asset_path in asset_paths:
        try:
            report = render_asset(asset_path, args)
        except Exception as exc:
            report = {
                "asset_path": str(asset_path),
                "valid": False,
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
            failed = True
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
