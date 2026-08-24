"""Versioned HDF5 schema for Roach-compatible static Town map assets."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np


ASSET_FORMAT_VERSION = "roach_static_map_v1"
GENERATOR_VERSION = "i2rad_roach_map_generator_v1"
REQUIRED_LAYERS: Tuple[str, ...] = (
    "road",
    "shoulder",
    "parking",
    "sidewalk",
    "stopline",
    "lane_marking_all",
    "lane_marking_yellow_broken",
    "lane_marking_yellow_solid",
    "lane_marking_white_broken",
    "lane_marking_white_solid",
)
RUNTIME_REQUIRED_LAYERS: Tuple[str, ...] = (
    "road",
    "lane_marking_all",
    "lane_marking_white_broken",
)
MANIFEST_SUFFIX = ".manifest.json"


@dataclass(frozen=True)
class GlobalMapMetadata:
    """Metadata required to reproduce and safely load a global raster asset."""

    town_name: str
    carla_server_version: str
    opendrive_sha256: str
    pixels_per_meter: float
    margin_meters: float
    waypoint_spacing_meters: float
    lane_precision_meters: float
    world_offset_x_meters: float
    world_offset_y_meters: float
    width_in_meters: float
    width_in_pixels: int
    stopline_segment_count: int = 0
    storage_mode: str = "global"
    asset_format_version: str = ASSET_FORMAT_VERSION
    generator_version: str = GENERATOR_VERSION
    generated_utc: str = ""

    def normalized(self) -> "GlobalMapMetadata":
        if self.generated_utc:
            return self
        values = asdict(self)
        values["generated_utc"] = datetime.now(timezone.utc).isoformat()
        return GlobalMapMetadata(**values)


@dataclass(frozen=True)
class TiledMapMetadata(GlobalMapMetadata):
    """Metadata for a tiled writer whose logical layer datasets stay global-shaped."""

    storage_mode: str = "tiled"
    tile_size_pixels: int = 4096
    tile_overlap_pixels: int = 0
    tile_count_x: int = 0
    tile_count_y: int = 0

    def normalized(self) -> "TiledMapMetadata":
        if self.generated_utc:
            return self
        values = asdict(self)
        values["generated_utc"] = datetime.now(timezone.utc).isoformat()
        return TiledMapMetadata(**values)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path_for(asset_path: Path) -> Path:
    asset_path = Path(asset_path)
    return asset_path.with_name(asset_path.name + MANIFEST_SUFFIX)


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(str(tmp_path), str(path))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _validate_masks_before_write(
    masks: Mapping[str, np.ndarray], metadata: GlobalMapMetadata
) -> Dict[str, np.ndarray]:
    missing = [name for name in REQUIRED_LAYERS if name not in masks]
    if missing:
        raise ValueError("Static-map masks are missing required layers: " + ", ".join(missing))

    expected_shape = (int(metadata.width_in_pixels), int(metadata.width_in_pixels))
    normalized: Dict[str, np.ndarray] = {}
    for name in REQUIRED_LAYERS:
        array = np.asarray(masks[name])
        if array.shape != expected_shape:
            raise ValueError(f"Layer {name!r} has shape {array.shape}, expected {expected_shape}")
        if array.dtype != np.uint8:
            raise ValueError(f"Layer {name!r} has dtype {array.dtype}, expected uint8")
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        normalized[name] = array
    return normalized


def _validate_tile_masks_before_write(
    masks: Mapping[str, np.ndarray],
    *,
    expected_shape: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    missing = [name for name in REQUIRED_LAYERS if name not in masks]
    if missing:
        raise ValueError("Static-map tile masks are missing required layers: " + ", ".join(missing))

    normalized: Dict[str, np.ndarray] = {}
    for name in REQUIRED_LAYERS:
        array = np.asarray(masks[name])
        if array.shape[0] < expected_shape[0] or array.shape[1] < expected_shape[1]:
            raise ValueError(
                f"Tile layer {name!r} has shape {array.shape}, expected at least {expected_shape}"
            )
        array = array[: expected_shape[0], : expected_shape[1]]
        if array.dtype != np.uint8:
            raise ValueError(f"Tile layer {name!r} has dtype {array.dtype}, expected uint8")
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        normalized[name] = array
    return normalized


def write_global_asset(
    asset_path: Path,
    masks: Mapping[str, np.ndarray],
    metadata: GlobalMapMetadata,
    *,
    overwrite: bool = False,
    chunk_size_pixels: int = 1024,
    compression_level: int = 4,
) -> Dict[str, object]:
    """Atomically write one global HDF5 asset and its hash manifest."""

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised in the dedicated map environment
        raise RuntimeError("h5py is required to write Roach static-map assets") from exc

    asset_path = Path(asset_path)
    metadata = metadata.normalized()
    normalized_masks = _validate_masks_before_write(masks, metadata)
    if asset_path.exists() and not overwrite:
        raise FileExistsError(f"Asset already exists; pass --overwrite to replace it: {asset_path}")

    asset_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = asset_path.with_name(f".{asset_path.name}.{uuid.uuid4().hex}.tmp")
    manifest_path = manifest_path_for(asset_path)
    width = int(metadata.width_in_pixels)
    chunk_edge = max(1, min(int(chunk_size_pixels), width))

    try:
        with h5py.File(str(tmp_path), "w", libver="latest") as handle:
            for key, value in asdict(metadata).items():
                handle.attrs[key] = value
            handle.attrs["world_offset_in_meters"] = np.asarray(
                [metadata.world_offset_x_meters, metadata.world_offset_y_meters],
                dtype=np.float32,
            )
            handle.attrs["layer_names_json"] = json.dumps(list(REQUIRED_LAYERS))

            for name in REQUIRED_LAYERS:
                handle.create_dataset(
                    name,
                    data=normalized_masks[name],
                    dtype=np.uint8,
                    chunks=(chunk_edge, chunk_edge),
                    compression="gzip",
                    compression_opts=int(compression_level),
                    shuffle=True,
                )
            handle.flush()

        os.replace(str(tmp_path), str(asset_path))
        asset_sha256 = sha256_file(asset_path)
        manifest = {
            "asset_path": asset_path.name,
            "asset_sha256": asset_sha256,
            "asset_size_bytes": asset_path.stat().st_size,
            "metadata": asdict(metadata),
            "required_layers": list(REQUIRED_LAYERS),
        }
        _atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _iter_dataset_chunks(dataset) -> Iterable[np.ndarray]:
    if dataset is None:
        return
    if dataset.chunks is None:
        yield np.asarray(dataset[...])
        return
    for selection in dataset.iter_chunks():
        yield np.asarray(dataset[selection])


def write_tiled_asset(
    asset_path: Path,
    metadata: TiledMapMetadata,
    tile_iterator,
    *,
    overwrite: bool = False,
    chunk_size_pixels: int = 1024,
    compression_level: int = 4,
) -> Dict[str, object]:
    """Write one tiled asset while preserving global-shaped layer datasets."""

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised in the dedicated map environment
        raise RuntimeError("h5py is required to write Roach tiled static-map assets") from exc

    asset_path = Path(asset_path)
    metadata = metadata.normalized()
    if metadata.storage_mode != "tiled":
        raise ValueError(f"Tiled asset metadata must use storage_mode=\'tiled\', got {metadata.storage_mode!r}")
    if int(metadata.tile_size_pixels) <= 0:
        raise ValueError("tile_size_pixels must be positive")
    if int(metadata.tile_overlap_pixels) < 0:
        raise ValueError("tile_overlap_pixels must be non-negative")
    if asset_path.exists() and not overwrite:
        raise FileExistsError(f"Asset already exists; pass --overwrite to replace it: {asset_path}")

    asset_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = asset_path.with_name(f".{asset_path.name}.{uuid.uuid4().hex}.tmp")
    manifest_path = manifest_path_for(asset_path)
    width = int(metadata.width_in_pixels)
    chunk_edge = max(1, min(int(chunk_size_pixels), int(metadata.tile_size_pixels), width))
    tile_reports = []

    try:
        with h5py.File(str(tmp_path), "w", libver="latest") as handle:
            for key, value in asdict(metadata).items():
                handle.attrs[key] = value
            handle.attrs["world_offset_in_meters"] = np.asarray(
                [metadata.world_offset_x_meters, metadata.world_offset_y_meters],
                dtype=np.float32,
            )
            handle.attrs["layer_names_json"] = json.dumps(list(REQUIRED_LAYERS))
            datasets = {}
            for name in REQUIRED_LAYERS:
                datasets[name] = handle.create_dataset(
                    name,
                    shape=(width, width),
                    dtype=np.uint8,
                    chunks=(chunk_edge, chunk_edge),
                    compression="gzip",
                    compression_opts=int(compression_level),
                    shuffle=True,
                    fillvalue=0,
                )
            tiles_group = handle.create_group("tiles")

            for tile in tile_iterator:
                row_start = int(tile["row_start"])
                row_end = int(tile["row_end"])
                col_start = int(tile["col_start"])
                col_end = int(tile["col_end"])
                if row_start < 0 or col_start < 0 or row_end > width or col_end > width:
                    raise ValueError(
                        "Tile write window escapes asset bounds: "
                        f"rows=[{row_start},{row_end}) cols=[{col_start},{col_end}) width={width}"
                    )
                if row_start >= row_end or col_start >= col_end:
                    raise ValueError(
                        f"Invalid tile write window rows=[{row_start},{row_end}) cols=[{col_start},{col_end})"
                    )
                expected_shape = (row_end - row_start, col_end - col_start)
                masks = _validate_tile_masks_before_write(tile["masks"], expected_shape=expected_shape)
                nonzero_by_layer = {}
                for name in REQUIRED_LAYERS:
                    datasets[name][row_start:row_end, col_start:col_end] = masks[name]
                    nonzero_by_layer[name] = int(np.count_nonzero(masks[name]))
                tile_id = str(tile.get("tile_id", f"r{row_start}_c{col_start}"))
                tile_group = tiles_group.create_group(tile_id)
                tile_group.attrs["row_start"] = row_start
                tile_group.attrs["row_end"] = row_end
                tile_group.attrs["col_start"] = col_start
                tile_group.attrs["col_end"] = col_end
                tile_group.attrs["nonzero_by_layer_json"] = json.dumps(nonzero_by_layer, sort_keys=True)
                tile_reports.append(
                    {
                        "tile_id": tile_id,
                        "row_start": row_start,
                        "row_end": row_end,
                        "col_start": col_start,
                        "col_end": col_end,
                        "nonzero_by_layer": nonzero_by_layer,
                    }
                )
            handle.attrs["tile_count_written"] = len(tile_reports)
            handle.flush()

        os.replace(str(tmp_path), str(asset_path))
        asset_sha256 = sha256_file(asset_path)
        manifest = {
            "asset_path": asset_path.name,
            "asset_sha256": asset_sha256,
            "asset_size_bytes": asset_path.stat().st_size,
            "metadata": asdict(metadata),
            "required_layers": list(REQUIRED_LAYERS),
            "tiles": tile_reports,
        }
        _atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def validate_global_asset(
    asset_path: Path,
    *,
    verify_manifest_hash: bool = True,
    require_manifest: bool = True,
    scan_values: bool = True,
    expected_storage_mode: Optional[str] = "global",
) -> Dict[str, object]:
    """Validate schema, metadata, datasets and optional sidecar hash."""

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised in the dedicated map environment
        raise RuntimeError("h5py is required to validate Roach static-map assets") from exc

    asset_path = Path(asset_path)
    if not asset_path.is_file():
        raise FileNotFoundError(f"Static-map asset does not exist: {asset_path}")

    manifest_path = manifest_path_for(asset_path)
    manifest: Optional[Dict[str, object]] = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif require_manifest:
        raise FileNotFoundError(f"Static-map manifest does not exist: {manifest_path}")

    errors = []
    layers: Dict[str, object] = {}
    with h5py.File(str(asset_path), "r", libver="latest", swmr=True) as handle:
        attrs = dict(handle.attrs)
        format_version = str(attrs.get("asset_format_version", ""))
        storage_mode = str(attrs.get("storage_mode", ""))
        width = int(attrs.get("width_in_pixels", 0))
        ppm = float(attrs.get("pixels_per_meter", 0.0))
        world_offset = np.asarray(attrs.get("world_offset_in_meters", []), dtype=np.float32)

        if format_version != ASSET_FORMAT_VERSION:
            errors.append(
                f"asset_format_version={format_version!r}, expected {ASSET_FORMAT_VERSION!r}"
            )
        if expected_storage_mode is not None and storage_mode != expected_storage_mode:
            errors.append(
                f"storage_mode={storage_mode!r}, expected {expected_storage_mode!r} for this validator"
            )
        if expected_storage_mode is None and storage_mode not in {"global", "tiled"}:
            errors.append(f"storage_mode={storage_mode!r}, expected 'global' or 'tiled'")
        if width <= 0:
            errors.append(f"width_in_pixels must be positive, got {width}")
        if not np.isfinite(ppm) or ppm <= 0:
            errors.append(f"pixels_per_meter must be finite and positive, got {ppm}")
        if world_offset.shape != (2,) or not np.isfinite(world_offset).all():
            errors.append(f"world_offset_in_meters must be finite shape (2,), got {world_offset}")

        expected_shape = (width, width)
        for name in REQUIRED_LAYERS:
            if name not in handle:
                errors.append(f"missing required dataset {name!r}")
                continue
            dataset = handle[name]
            if tuple(dataset.shape) != expected_shape:
                errors.append(f"dataset {name!r} shape={dataset.shape}, expected={expected_shape}")
            if dataset.dtype != np.dtype(np.uint8):
                errors.append(f"dataset {name!r} dtype={dataset.dtype}, expected=uint8")
            if dataset.chunks is None:
                errors.append(f"dataset {name!r} must be chunked")

            layer_report: Dict[str, object] = {
                "shape": list(dataset.shape),
                "dtype": str(dataset.dtype),
                "chunks": list(dataset.chunks) if dataset.chunks is not None else None,
                "compression": dataset.compression,
            }
            if scan_values:
                value_min = 255
                value_max = 0
                nonzero = 0
                total = 0
                unique_values = set()
                for chunk in _iter_dataset_chunks(dataset):
                    if chunk.size == 0:
                        continue
                    value_min = min(value_min, int(chunk.min()))
                    value_max = max(value_max, int(chunk.max()))
                    nonzero += int(np.count_nonzero(chunk))
                    total += int(chunk.size)
                    if len(unique_values) <= 32:
                        unique_values.update(int(value) for value in np.unique(chunk))
                layer_report.update(
                    {
                        "min": value_min,
                        "max": value_max,
                        "nonzero_pixels": nonzero,
                        "nonzero_ratio": float(nonzero / total) if total else 0.0,
                        "unique_values": sorted(unique_values) if len(unique_values) <= 32 else ">32",
                    }
                )
                if name in RUNTIME_REQUIRED_LAYERS and name != "lane_marking_white_broken" and nonzero == 0:
                    errors.append(f"dataset {name!r} is unexpectedly empty")
            layers[name] = layer_report

        report = {
            "asset_path": str(asset_path.resolve()),
            "asset_size_bytes": asset_path.stat().st_size,
            "asset_format_version": format_version,
            "generator_version": str(attrs.get("generator_version", "")),
            "town_name": str(attrs.get("town_name", "")),
            "carla_server_version": str(attrs.get("carla_server_version", "")),
            "opendrive_sha256": str(attrs.get("opendrive_sha256", "")),
            "storage_mode": storage_mode,
            "pixels_per_meter": ppm,
            "width_in_pixels": width,
            "world_offset_in_meters": world_offset.tolist(),
            "layers": layers,
            "manifest_path": str(manifest_path.resolve()) if manifest_path.exists() else None,
        }
        if storage_mode == "tiled":
            tile_size = int(attrs.get("tile_size_pixels", 0))
            tile_overlap = int(attrs.get("tile_overlap_pixels", -1))
            tile_count_x = int(attrs.get("tile_count_x", 0))
            tile_count_y = int(attrs.get("tile_count_y", 0))
            tile_count_written = int(attrs.get("tile_count_written", 0))
            if tile_size <= 0:
                errors.append(f"tile_size_pixels must be positive for tiled assets, got {tile_size}")
            if tile_overlap < 0:
                errors.append(f"tile_overlap_pixels must be non-negative, got {tile_overlap}")
            expected_tile_count = tile_count_x * tile_count_y
            if tile_count_x <= 0 or tile_count_y <= 0:
                errors.append(
                    f"tile_count_x/y must be positive for tiled assets, got {tile_count_x}/{tile_count_y}"
                )
            if tile_count_written != expected_tile_count:
                errors.append(
                    "tile_count_written mismatch: "
                    f"written={tile_count_written}, expected={expected_tile_count}"
                )
            if "tiles" not in handle:
                errors.append("missing 'tiles' group for tiled asset")
            elif len(handle["tiles"]) != tile_count_written:
                errors.append(
                    f"tiles group contains {len(handle['tiles'])} entries, expected {tile_count_written}"
                )
            report.update(
                {
                    "tile_size_pixels": tile_size,
                    "tile_overlap_pixels": tile_overlap,
                    "tile_count_x": tile_count_x,
                    "tile_count_y": tile_count_y,
                    "tile_count_written": tile_count_written,
                }
            )

    if manifest is not None:
        manifest_name = str(manifest.get("asset_path", ""))
        if manifest_name != asset_path.name:
            errors.append(f"manifest asset_path={manifest_name!r}, expected {asset_path.name!r}")
        if verify_manifest_hash:
            actual_sha256 = sha256_file(asset_path)
            expected_sha256 = str(manifest.get("asset_sha256", ""))
            report["asset_sha256"] = actual_sha256
            if actual_sha256 != expected_sha256:
                errors.append(
                    f"asset sha256 mismatch: manifest={expected_sha256!r}, actual={actual_sha256!r}"
                )

    report["errors"] = errors
    report["valid"] = not errors
    return report


def validate_static_map_asset(
    asset_path: Path,
    *,
    verify_manifest_hash: bool = True,
    require_manifest: bool = True,
    scan_values: bool = True,
) -> Dict[str, object]:
    """Validate either global or tiled static-map storage."""

    return validate_global_asset(
        asset_path,
        verify_manifest_hash=verify_manifest_hash,
        require_manifest=require_manifest,
        scan_values=scan_values,
        expected_storage_mode=None,
    )


def validate_tiled_asset(
    asset_path: Path,
    *,
    verify_manifest_hash: bool = True,
    require_manifest: bool = True,
    scan_values: bool = True,
) -> Dict[str, object]:
    """Validate schema, metadata and datasets for tiled storage."""

    return validate_global_asset(
        asset_path,
        verify_manifest_hash=verify_manifest_hash,
        require_manifest=require_manifest,
        scan_values=scan_values,
        expected_storage_mode="tiled",
    )
