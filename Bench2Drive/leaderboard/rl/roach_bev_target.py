"""Roach-style ego-centric BEV semantic-mask target generation.

The generator mirrors the CARLA-Roach ChauffeurNet birdview contract without
importing CARLA at module import time.  It consumes the static Town HDF5 assets
generated under ROACH_BEV_MAP_ROOT plus per-frame route/actor geometry from the
closed-loop environment.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


ROACH_STATIC_CHANNELS: Tuple[str, ...] = ("road", "route", "lane")
ROACH_DYNAMIC_GROUPS: Tuple[str, ...] = ("vehicle", "walker", "traffic_light_stop")
ROACH_DEFAULT_HISTORY_IDX: Tuple[int, ...] = (-16, -11, -6, -1)


@dataclass(frozen=True)
class RoachBevTargetConfig:
    width_in_pixels: int = 192
    pixels_ev_to_bottom: int = 40
    pixels_per_meter: float = 5.0
    history_idx: Tuple[int, ...] = ROACH_DEFAULT_HISTORY_IDX
    route_max_waypoints: int = 80
    route_thickness_pixels: int = 16
    stopline_thickness_pixels: int = 6
    static_crop_padding_pixels: int = 2
    scale_vehicle_bbox: float = 1.0
    scale_walker_bbox: float = 2.0
    scale_collision_bbox: float = 1.0

    def __post_init__(self) -> None:
        if int(self.width_in_pixels) <= 0:
            raise ValueError("width_in_pixels must be positive")
        if int(self.pixels_ev_to_bottom) < 0 or int(self.pixels_ev_to_bottom) > int(self.width_in_pixels):
            raise ValueError("pixels_ev_to_bottom must be in [0, width_in_pixels]")
        if not np.isfinite(float(self.pixels_per_meter)) or float(self.pixels_per_meter) <= 0.0:
            raise ValueError("pixels_per_meter must be finite and positive")
        if not self.history_idx:
            raise ValueError("history_idx must contain at least one index")
        if any(int(idx) >= 0 for idx in self.history_idx):
            raise ValueError("history_idx must use negative history indices, matching CARLA-Roach")

    @property
    def channel_names(self) -> Tuple[str, ...]:
        names: List[str] = list(ROACH_STATIC_CHANNELS)
        for group in ROACH_DYNAMIC_GROUPS:
            for idx in self.history_idx:
                names.append(f"{group}_h{idx}")
        return tuple(names)


@dataclass(frozen=True)
class RoachActorBox:
    """A CARLA-like actor box descriptor.

    transform is the actor/world transform.  bbox_location and bbox_extent are
    CARLA-style local bounding-box location/extent objects or any duck-typed
    object exposing x/y(/z).
    """

    transform: object
    bbox_location: object
    bbox_extent: object


@dataclass(frozen=True)
class RoachDynamicFrame:
    vehicles: Tuple[RoachActorBox, ...]
    walkers: Tuple[RoachActorBox, ...]
    tl_green: Tuple[Tuple[object, object], ...]
    tl_yellow: Tuple[Tuple[object, object], ...]
    tl_red: Tuple[Tuple[object, object], ...]
    stops: Tuple[RoachActorBox, ...]


def roach_bev_channel_names(
    history_idx: Sequence[int] = ROACH_DEFAULT_HISTORY_IDX,
) -> Tuple[str, ...]:
    return RoachBevTargetConfig(history_idx=tuple(int(idx) for idx in history_idx)).channel_names


def _coord(value: object, name: str, default: float = 0.0) -> float:
    return float(getattr(value, name, default))


def _location_xy(location: object) -> np.ndarray:
    return np.asarray([_coord(location, "x"), _coord(location, "y")], dtype=np.float32)


def _transform_location(transform: object) -> object:
    return getattr(transform, "location")


def _transform_yaw_degrees(transform: object) -> float:
    rotation = getattr(transform, "rotation")
    return float(getattr(rotation, "yaw", 0.0))


def _copy_extent_scaled(extent: object, scale: float) -> np.ndarray:
    scale = float(scale)
    return np.asarray(
        [
            max(0.0, _coord(extent, "x") * scale),
            max(0.0, _coord(extent, "y") * scale),
            max(0.0, _coord(extent, "z") * scale),
        ],
        dtype=np.float32,
    )


def _rotate_xy(points: np.ndarray, yaw_degrees: float) -> np.ndarray:
    yaw = np.deg2rad(float(yaw_degrees))
    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))
    rot = np.asarray([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float32)
    return points @ rot.T


def _local_to_world_xy(transform: object, local_xy: np.ndarray) -> np.ndarray:
    origin = _location_xy(_transform_location(transform))
    return origin + _rotate_xy(local_xy, _transform_yaw_degrees(transform))


def _actor_polygon_world_xy(actor_box: RoachActorBox, *, scale: float = 1.0) -> np.ndarray:
    bbox_loc = _location_xy(actor_box.bbox_location)
    extent = _copy_extent_scaled(actor_box.bbox_extent, scale)
    # Match CARLA-Roach's five-point vehicle footprint.  The middle front point
    # makes the filled polygon less sensitive to the vehicle's nose shape.
    local = np.asarray(
        [
            [-extent[0], -extent[1]],
            [extent[0], -extent[1]],
            [extent[0], 0.0],
            [extent[0], extent[1]],
            [-extent[0], extent[1]],
        ],
        dtype=np.float32,
    )
    return _local_to_world_xy(actor_box.transform, local + bbox_loc.reshape(1, 2))


def _route_location(item: object) -> Optional[object]:
    if isinstance(item, (list, tuple)) and item:
        item = item[0]
    if hasattr(item, "transform"):
        return getattr(item.transform, "location", None)
    if hasattr(item, "location"):
        return getattr(item, "location")
    if hasattr(item, "x") and hasattr(item, "y"):
        return item
    return None


class RoachStaticMapAsset:
    """Lazy HDF5 reader for one global-shaped Roach static-map asset."""

    def __init__(self, asset_path: Path) -> None:
        self.asset_path = Path(asset_path)
        if not self.asset_path.is_file():
            raise FileNotFoundError(f"Roach static-map asset does not exist: {self.asset_path}")
        self._handle = None
        self._attrs: Optional[Dict[str, object]] = None

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._attrs = None

    def __enter__(self) -> "RoachStaticMapAsset":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _ensure_open(self):
        if self._handle is not None:
            return self._handle
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - exercised in roach-map env
            fallback_site_packages = str(os.environ.get("ROACH_H5PY_SITE_PACKAGES", "") or "").strip()
            if fallback_site_packages:
                import sys

                if fallback_site_packages not in sys.path:
                    sys.path.append(fallback_site_packages)
                try:
                    import h5py
                except ImportError as fallback_exc:
                    raise RuntimeError(
                        "h5py is required to read Roach BEV static-map assets; "
                        "ROACH_H5PY_SITE_PACKAGES was set but import still failed"
                    ) from fallback_exc
            else:
                raise RuntimeError(
                    "h5py is required to read Roach BEV static-map assets; install h5py in the runtime env "
                    "or set ROACH_H5PY_SITE_PACKAGES plus the matching LD_LIBRARY_PATH"
                ) from exc
        self._handle = h5py.File(str(self.asset_path), "r", libver="latest", swmr=True)
        self._attrs = dict(self._handle.attrs)
        return self._handle

    @property
    def attrs(self) -> Mapping[str, object]:
        self._ensure_open()
        assert self._attrs is not None
        return self._attrs

    @property
    def pixels_per_meter(self) -> float:
        return float(self.attrs["pixels_per_meter"])

    @property
    def world_offset(self) -> np.ndarray:
        return np.asarray(self.attrs["world_offset_in_meters"], dtype=np.float32)

    @property
    def width_in_pixels(self) -> int:
        return int(self.attrs["width_in_pixels"])

    def world_to_cv_pixel(self, location: object) -> np.ndarray:
        xy = _location_xy(location)
        shifted = (xy - self.world_offset) * float(self.pixels_per_meter)
        # OpenCV uses source coordinates as (x=column, y=row).
        return np.asarray([shifted[0], shifted[1]], dtype=np.float32)

    def warp_layer(
        self,
        layer_name: str,
        warp_matrix: np.ndarray,
        source_polygon_xy: np.ndarray,
        *,
        output_width: int,
        padding_pixels: int = 2,
    ) -> np.ndarray:
        handle = self._ensure_open()
        if layer_name not in handle:
            raise KeyError(f"Static-map asset {self.asset_path} is missing layer {layer_name!r}")
        dataset = handle[layer_name]
        width = int(self.width_in_pixels)
        polygon = np.asarray(source_polygon_xy, dtype=np.float32).reshape(-1, 2)
        min_col = int(np.floor(float(polygon[:, 0].min()))) - int(padding_pixels)
        max_col = int(np.ceil(float(polygon[:, 0].max()))) + int(padding_pixels)
        min_row = int(np.floor(float(polygon[:, 1].min()))) - int(padding_pixels)
        max_row = int(np.ceil(float(polygon[:, 1].max()))) + int(padding_pixels)
        col_start = max(0, min(width, min_col))
        col_end = max(0, min(width, max_col + 1))
        row_start = max(0, min(width, min_row))
        row_end = max(0, min(width, max_row + 1))
        if row_start >= row_end or col_start >= col_end:
            return np.zeros((int(output_width), int(output_width)), dtype=np.uint8)

        crop = np.asarray(dataset[row_start:row_end, col_start:col_end], dtype=np.uint8)
        local_matrix = np.asarray(warp_matrix, dtype=np.float32).copy()
        local_matrix[:, 2] += local_matrix[:, 0] * float(col_start) + local_matrix[:, 1] * float(row_start)
        return cv2.warpAffine(
            crop,
            local_matrix,
            (int(output_width), int(output_width)),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.uint8)


class RoachStaticMapAssetCache:
    def __init__(self, asset_root: Optional[Path] = None) -> None:
        root = asset_root or os.environ.get("ROACH_BEV_MAP_ROOT")
        if not root:
            raise ValueError("asset_root or ROACH_BEV_MAP_ROOT is required for Roach BEV target generation")
        self.asset_root = Path(root)
        self._assets: Dict[str, RoachStaticMapAsset] = {}

    def get(self, town_name: str) -> RoachStaticMapAsset:
        town = str(town_name)
        if town not in self._assets:
            self._assets[town] = RoachStaticMapAsset(self.asset_root / f"{town}.h5")
        return self._assets[town]

    def close(self) -> None:
        for asset in self._assets.values():
            asset.close()
        self._assets.clear()


class RoachBevTargetGenerator:
    """Generate Roach-compatible BEV semantic masks for one ego stream."""

    def __init__(
        self,
        *,
        asset_root: Optional[Path] = None,
        config: Optional[RoachBevTargetConfig] = None,
    ) -> None:
        self.config = config or RoachBevTargetConfig()
        self.asset_cache = RoachStaticMapAssetCache(asset_root=asset_root)
        self._history: deque[RoachDynamicFrame] = deque(maxlen=max(abs(int(i)) for i in self.config.history_idx))

    @property
    def channel_names(self) -> Tuple[str, ...]:
        return self.config.channel_names

    def close(self) -> None:
        self.asset_cache.close()
        self._history.clear()

    def reset(self) -> None:
        self._history.clear()

    def _warp_transform(
        self,
        asset: RoachStaticMapAsset,
        ego_transform: object,
    ) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        width = float(cfg.width_in_pixels)
        ego_px = asset.world_to_cv_pixel(_transform_location(ego_transform))
        yaw = np.deg2rad(_transform_yaw_degrees(ego_transform))
        forward_vec = np.asarray([np.cos(yaw), np.sin(yaw)], dtype=np.float32)
        right_vec = np.asarray([np.cos(yaw + 0.5 * np.pi), np.sin(yaw + 0.5 * np.pi)], dtype=np.float32)

        bottom_left = ego_px - float(cfg.pixels_ev_to_bottom) * forward_vec - 0.5 * width * right_vec
        top_left = ego_px + (width - float(cfg.pixels_ev_to_bottom)) * forward_vec - 0.5 * width * right_vec
        top_right = ego_px + (width - float(cfg.pixels_ev_to_bottom)) * forward_vec + 0.5 * width * right_vec
        bottom_right = ego_px - float(cfg.pixels_ev_to_bottom) * forward_vec + 0.5 * width * right_vec

        src_pts = np.stack((bottom_left, top_left, top_right), axis=0).astype(np.float32)
        dst_pts = np.asarray(
            [[0.0, width - 1.0], [0.0, 0.0], [width - 1.0, 0.0]],
            dtype=np.float32,
        )
        warp_matrix = cv2.getAffineTransform(src_pts, dst_pts)
        source_polygon = np.stack((bottom_left, top_left, top_right, bottom_right), axis=0).astype(np.float32)
        return warp_matrix, source_polygon

    def _mask_from_actor_boxes(
        self,
        boxes: Sequence[RoachActorBox],
        asset: RoachStaticMapAsset,
        warp_matrix: np.ndarray,
        *,
        scale: float,
    ) -> np.ndarray:
        width = int(self.config.width_in_pixels)
        mask = np.zeros((width, width), dtype=np.uint8)
        for box in boxes:
            polygon_world = _actor_polygon_world_xy(box, scale=scale)
            polygon_source = (polygon_world - asset.world_offset.reshape(1, 2)) * float(asset.pixels_per_meter)
            polygon_warped = cv2.transform(polygon_source.reshape(-1, 1, 2).astype(np.float32), warp_matrix)
            polygon_int = np.round(polygon_warped[:, 0, :]).astype(np.int32)
            if polygon_int.shape[0] >= 3:
                cv2.fillConvexPoly(mask, polygon_int, 1)
        return mask.astype(bool)

    def _mask_from_stopline_segments(
        self,
        segments: Sequence[Tuple[object, object]],
        asset: RoachStaticMapAsset,
        warp_matrix: np.ndarray,
    ) -> np.ndarray:
        width = int(self.config.width_in_pixels)
        mask = np.zeros((width, width), dtype=np.uint8)
        for left, right in segments:
            source = np.stack((asset.world_to_cv_pixel(left), asset.world_to_cv_pixel(right)), axis=0)
            warped = cv2.transform(source.reshape(-1, 1, 2).astype(np.float32), warp_matrix)
            p0 = tuple(np.round(warped[0, 0]).astype(np.int32).tolist())
            p1 = tuple(np.round(warped[1, 0]).astype(np.int32).tolist())
            cv2.line(mask, p0, p1, color=1, thickness=int(self.config.stopline_thickness_pixels))
        return mask.astype(bool)

    def _route_mask(
        self,
        route_waypoints: Sequence[object],
        asset: RoachStaticMapAsset,
        warp_matrix: np.ndarray,
    ) -> np.ndarray:
        width = int(self.config.width_in_pixels)
        mask = np.zeros((width, width), dtype=np.uint8)
        points = []
        for item in list(route_waypoints)[: int(self.config.route_max_waypoints)]:
            location = _route_location(item)
            if location is None:
                continue
            points.append(asset.world_to_cv_pixel(location))
        if len(points) >= 2:
            source = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
            warped = cv2.transform(source, warp_matrix)
            cv2.polylines(
                mask,
                [np.round(warped[:, 0, :]).astype(np.int32)],
                False,
                color=1,
                thickness=int(self.config.route_thickness_pixels),
            )
        return mask.astype(bool)

    def _select_history(self) -> List[RoachDynamicFrame]:
        if not self._history:
            empty = RoachDynamicFrame((), (), (), (), (), ())
            return [empty for _ in self.config.history_idx]
        frames = list(self._history)
        qsize = len(frames)
        selected = []
        for idx in self.config.history_idx:
            safe_idx = max(int(idx), -qsize)
            selected.append(frames[safe_idx])
        return selected

    def build(
        self,
        *,
        town_name: str,
        ego_transform: object,
        route_waypoints: Sequence[object],
        vehicle_boxes: Sequence[RoachActorBox] = (),
        walker_boxes: Sequence[RoachActorBox] = (),
        traffic_light_stopline_segments: Optional[Mapping[str, Sequence[Tuple[object, object]]]] = None,
        stop_boxes: Sequence[RoachActorBox] = (),
    ) -> Dict[str, object]:
        """Generate a Roach BEV target for the current frame.

        traffic_light_stopline_segments accepts keys green/yellow/red.  Values
        are stopline endpoint pairs in world coordinates.
        """

        asset = self.asset_cache.get(town_name)
        cfg = self.config
        if not np.isclose(float(asset.pixels_per_meter), float(cfg.pixels_per_meter)):
            raise ValueError(
                f"Target ppm={cfg.pixels_per_meter} does not match asset ppm={asset.pixels_per_meter}"
            )

        tl = traffic_light_stopline_segments or {}
        frame = RoachDynamicFrame(
            vehicles=tuple(vehicle_boxes),
            walkers=tuple(walker_boxes),
            tl_green=tuple(tl.get("green", ())),
            tl_yellow=tuple(tl.get("yellow", ())),
            tl_red=tuple(tl.get("red", ())),
            stops=tuple(stop_boxes),
        )
        self._history.append(frame)

        warp_matrix, source_polygon = self._warp_transform(asset, ego_transform)

        road_raw = asset.warp_layer(
            "road",
            warp_matrix,
            source_polygon,
            output_width=int(cfg.width_in_pixels),
            padding_pixels=int(cfg.static_crop_padding_pixels),
        )
        lane_all = asset.warp_layer(
            "lane_marking_all",
            warp_matrix,
            source_polygon,
            output_width=int(cfg.width_in_pixels),
            padding_pixels=int(cfg.static_crop_padding_pixels),
        )
        lane_broken = asset.warp_layer(
            "lane_marking_white_broken",
            warp_matrix,
            source_polygon,
            output_width=int(cfg.width_in_pixels),
            padding_pixels=int(cfg.static_crop_padding_pixels),
        )

        road = (road_raw > 0).astype(np.uint8) * 255
        route = self._route_mask(route_waypoints, asset, warp_matrix).astype(np.uint8) * 255
        lane = (lane_all > 0).astype(np.uint8) * 255
        lane[lane_broken > 0] = 120

        vehicle_history: List[np.ndarray] = []
        walker_history: List[np.ndarray] = []
        tl_history: List[np.ndarray] = []
        for item in self._select_history():
            vehicle_history.append(
                self._mask_from_actor_boxes(
                    item.vehicles,
                    asset,
                    warp_matrix,
                    scale=float(cfg.scale_vehicle_bbox),
                ).astype(np.uint8)
                * 255
            )
            walker_history.append(
                self._mask_from_actor_boxes(
                    item.walkers,
                    asset,
                    warp_matrix,
                    scale=float(cfg.scale_walker_bbox),
                ).astype(np.uint8)
                * 255
            )
            c_tl = np.zeros((int(cfg.width_in_pixels), int(cfg.width_in_pixels)), dtype=np.uint8)
            c_tl[self._mask_from_stopline_segments(item.tl_green, asset, warp_matrix)] = 80
            c_tl[self._mask_from_stopline_segments(item.tl_yellow, asset, warp_matrix)] = 170
            c_tl[self._mask_from_stopline_segments(item.tl_red, asset, warp_matrix)] = 255
            c_tl[
                self._mask_from_actor_boxes(
                    item.stops,
                    asset,
                    warp_matrix,
                    scale=1.0,
                )
            ] = 255
            tl_history.append(c_tl)

        masks = np.stack(
            (road, route, lane, *vehicle_history, *walker_history, *tl_history),
            axis=0,
        ).astype(np.uint8)

        return {
            "masks": masks,
            "channel_names": self.channel_names,
            "town_name": str(town_name),
            "warp_matrix": warp_matrix.astype(np.float32),
            "source_polygon": source_polygon.astype(np.float32),
            "pixels_per_meter": float(cfg.pixels_per_meter),
            "width_in_pixels": int(cfg.width_in_pixels),
        }


def actor_box_from_carla_actor(actor: object) -> RoachActorBox:
    bbox = getattr(actor, "bounding_box")
    return RoachActorBox(
        transform=actor.get_transform(),
        bbox_location=getattr(bbox, "location"),
        bbox_extent=getattr(bbox, "extent"),
    )


def actor_box_from_level_bbox(bbox: object, transform_factory=None, location_factory=None) -> RoachActorBox:
    """Convert a CARLA level bounding box to a RoachActorBox.

    transform_factory/location_factory are optional CARLA constructors.  Tests
    can pass simple objects directly by providing bbox.transform-like fields.
    """

    if transform_factory is None:
        transform = getattr(bbox, "transform", None)
        if transform is None:
            transform = type("TransformLike", (), {"location": bbox.location, "rotation": bbox.rotation})()
    else:
        transform = transform_factory(bbox.location, bbox.rotation)
    if location_factory is None:
        bbox_location = type("LocationLike", (), {"x": 0.0, "y": 0.0, "z": 0.0})()
    else:
        bbox_location = location_factory()
    return RoachActorBox(transform=transform, bbox_location=bbox_location, bbox_extent=bbox.extent)


def render_roach_bev_target(masks: np.ndarray) -> np.ndarray:
    """RGB debug rendering for a Roach BEV target tensor."""

    masks = np.asarray(masks)
    if masks.ndim != 3 or masks.shape[0] < 3:
        raise ValueError(f"masks must be [C,H,W] with at least 3 channels, got {masks.shape}")
    height, width = masks.shape[-2:]
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[masks[0] > 0] = (46, 52, 54)
    image[masks[1] > 0] = (136, 138, 133)
    image[masks[2] > 0] = (255, 0, 255)
    history_count = (masks.shape[0] - 3) // 3
    vehicle_start = 3
    walker_start = vehicle_start + history_count
    tl_start = walker_start + history_count
    for idx in range(history_count):
        factor = float(history_count - 1 - idx) * 0.2
        vehicle_color = np.asarray([0, 0, min(255, int(255 + (255 - 255) * factor))], dtype=np.uint8)
        walker_color = np.asarray([0, 255, 255], dtype=np.uint8)
        image[masks[vehicle_start + idx] > 0] = vehicle_color
        image[masks[walker_start + idx] > 0] = walker_color
        tl_mask = masks[tl_start + idx]
        image[tl_mask == 80] = (0, 255, 0)
        image[tl_mask == 170] = (255, 255, 0)
        image[tl_mask == 255] = (255, 0, 0)
    return image
