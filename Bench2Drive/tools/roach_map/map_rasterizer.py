"""Complete Roach static Town-map rasterization, isolated from RL training."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .asset_schema import REQUIRED_LAYERS


COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)


@dataclass(frozen=True)
class MapBounds:
    min_x_meters: float
    min_y_meters: float
    max_x_meters: float
    max_y_meters: float
    width_in_meters: float
    width_in_pixels: int
    margin_meters: float
    pixels_per_meter: float

    @property
    def world_offset(self) -> np.ndarray:
        return np.asarray([self.min_x_meters, self.min_y_meters], dtype=np.float32)

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GlobalRasterEstimate:
    width_in_pixels: int
    pixel_count: int
    layer_count: int
    surface_bytes_per_pixel: int
    output_bytes_per_pixel: int
    raw_surface_bytes: int
    output_mask_bytes: int
    fixed_overhead_bytes: int
    safety_factor: float
    estimated_peak_bytes: int

    @property
    def estimated_peak_gib(self) -> float:
        return float(self.estimated_peak_bytes / (1024 ** 3))

    def as_dict(self) -> Dict[str, object]:
        values = asdict(self)
        values["estimated_peak_gib"] = self.estimated_peak_gib
        return values


def estimate_global_raster_memory(
    width_in_pixels: int,
    *,
    layer_count: int = 3,
    surface_bytes_per_pixel: int = 4,
    output_bytes_per_pixel: int = 1,
    fixed_overhead_bytes: int = 512 * 1024 * 1024,
    safety_factor: float = 1.35,
) -> GlobalRasterEstimate:
    """Conservative preflight estimate for the in-memory global renderer."""

    width = int(width_in_pixels)
    if width <= 0:
        raise ValueError(f"width_in_pixels must be positive, got {width}")
    if layer_count <= 0:
        raise ValueError(f"layer_count must be positive, got {layer_count}")
    pixel_count = width * width
    raw_surface_bytes = pixel_count * int(layer_count) * int(surface_bytes_per_pixel)
    output_mask_bytes = pixel_count * int(layer_count) * int(output_bytes_per_pixel)
    variable_bytes = raw_surface_bytes + output_mask_bytes
    estimated_peak = math.ceil(variable_bytes * float(safety_factor) + int(fixed_overhead_bytes))
    return GlobalRasterEstimate(
        width_in_pixels=width,
        pixel_count=pixel_count,
        layer_count=int(layer_count),
        surface_bytes_per_pixel=int(surface_bytes_per_pixel),
        output_bytes_per_pixel=int(output_bytes_per_pixel),
        raw_surface_bytes=raw_surface_bytes,
        output_mask_bytes=output_mask_bytes,
        fixed_overhead_bytes=int(fixed_overhead_bytes),
        safety_factor=float(safety_factor),
        estimated_peak_bytes=estimated_peak,
    )


def compute_map_bounds(
    carla_map,
    *,
    pixels_per_meter: float,
    margin_meters: float = 100.0,
    waypoint_spacing_meters: float = 2.0,
) -> MapBounds:
    """Match Roach's square global-map extent calculation."""

    ppm = float(pixels_per_meter)
    if not np.isfinite(ppm) or ppm <= 0:
        raise ValueError(f"pixels_per_meter must be finite and positive, got {ppm}")
    margin = float(margin_meters)
    spacing = float(waypoint_spacing_meters)
    waypoints = list(carla_map.generate_waypoints(spacing))
    if not waypoints:
        raise RuntimeError("CARLA map returned no waypoints; cannot determine raster bounds")

    min_x = min(float(w.transform.location.x) for w in waypoints) - margin
    min_y = min(float(w.transform.location.y) for w in waypoints) - margin
    max_x = max(float(w.transform.location.x) for w in waypoints) + margin
    max_y = max(float(w.transform.location.y) for w in waypoints) + margin
    width_m = max(max_x - min_x, max_y - min_y)
    width_px = int(round(ppm * width_m))
    if width_px <= 0:
        raise RuntimeError(f"Computed invalid raster width {width_px} for map width {width_m}")
    return MapBounds(
        min_x_meters=min_x,
        min_y_meters=min_y,
        max_x_meters=max_x,
        max_y_meters=max_y,
        width_in_meters=width_m,
        width_in_pixels=width_px,
        margin_meters=margin,
        pixels_per_meter=ppm,
    )


class GlobalRoachMapRasterizer:
    """Draw Roach-compatible global road and lane-marking masks."""

    def __init__(
        self,
        *,
        carla_module,
        pygame_module,
        bounds: MapBounds,
        lane_precision_meters: float = 0.05,
        max_samples_per_topology_entry: int = 2_000_000,
        clip_padding_meters: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        self.carla = carla_module
        self.pygame = pygame_module
        self.bounds = bounds
        self.lane_precision_meters = float(lane_precision_meters)
        self.max_samples_per_topology_entry = int(max_samples_per_topology_entry)
        self.clip_padding_meters = None if clip_padding_meters is None else float(clip_padding_meters)
        self.progress_callback = progress_callback
        if self.lane_precision_meters <= 0:
            raise ValueError("lane_precision_meters must be positive")
        if self.max_samples_per_topology_entry <= 0:
            raise ValueError("max_samples_per_topology_entry must be positive")
        if self.clip_padding_meters is not None and self.clip_padding_meters < 0:
            raise ValueError("clip_padding_meters must be non-negative")

    def rasterize(self, carla_map, stopline_vertices: Sequence[Tuple[object, object]] = ()) -> Dict[str, np.ndarray]:
        width = int(self.bounds.width_in_pixels)
        surfaces = {name: self._new_surface(width) for name in REQUIRED_LAYERS}

        topology = sorted(
            carla_map.get_topology(),
            key=lambda pair: float(pair[0].transform.location.z),
        )
        total = len(topology)
        for index, (waypoint, end_waypoint) in enumerate(topology, start=1):
            if not self._topology_pair_may_intersect(waypoint, end_waypoint):
                if self.progress_callback is not None:
                    self.progress_callback(index, total)
                continue

            waypoints = self._follow_road(waypoint)
            waypoints = self._filter_waypoints_for_bounds(waypoints)
            if len(waypoints) < 2:
                if self.progress_callback is not None:
                    self.progress_callback(index, total)
                continue
            shoulder = [[], []]
            parking = [[], []]
            sidewalk = [[], []]
            for sample in waypoints:
                left = sample.get_left_lane()
                while left is not None and left.lane_type != self.carla.LaneType.Driving:
                    if left.lane_type == self.carla.LaneType.Shoulder:
                        shoulder[0].append(left)
                    elif left.lane_type == self.carla.LaneType.Parking:
                        parking[0].append(left)
                    elif left.lane_type == self.carla.LaneType.Sidewalk:
                        sidewalk[0].append(left)
                    left = left.get_left_lane()

                right = sample.get_right_lane()
                while right is not None and right.lane_type != self.carla.LaneType.Driving:
                    if right.lane_type == self.carla.LaneType.Shoulder:
                        shoulder[1].append(right)
                    elif right.lane_type == self.carla.LaneType.Parking:
                        parking[1].append(right)
                    elif right.lane_type == self.carla.LaneType.Sidewalk:
                        sidewalk[1].append(right)
                    right = right.get_right_lane()

            self._draw_lane(surfaces["road"], waypoints, COLOR_WHITE)
            for side in (0, 1):
                self._draw_lane(surfaces["shoulder"], shoulder[side], COLOR_WHITE)
                self._draw_lane(surfaces["parking"], parking[side], COLOR_WHITE)
                self._draw_lane(surfaces["sidewalk"], sidewalk[side], COLOR_WHITE)
            if not waypoint.is_junction:
                self._draw_lane_marking_single_side(surfaces, waypoints, -1)
                self._draw_lane_marking_single_side(surfaces, waypoints, 1)
            if self.progress_callback is not None:
                self.progress_callback(index, total)

        for left_location, right_location in stopline_vertices:
            points = [
                self._world_to_surface_pixel(left_location),
                self._world_to_surface_pixel(right_location),
            ]
            self._draw_line(surfaces["stopline"], points, width=2)

        masks = {name: self._surface_to_uint8(surface) for name, surface in surfaces.items()}
        del surfaces
        return masks

    def _expanded_bounds(self) -> Tuple[float, float, float, float]:
        padding = 0.0 if self.clip_padding_meters is None else float(self.clip_padding_meters)
        return (
            float(self.bounds.min_x_meters) - padding,
            float(self.bounds.max_x_meters) + padding,
            float(self.bounds.min_y_meters) - padding,
            float(self.bounds.max_y_meters) + padding,
        )

    def _location_in_expanded_bounds(self, location) -> bool:
        min_x, max_x, min_y, max_y = self._expanded_bounds()
        x = float(location.x)
        y = float(location.y)
        return min_x <= x <= max_x and min_y <= y <= max_y

    def _topology_pair_may_intersect(self, start_waypoint, end_waypoint) -> bool:
        if self.clip_padding_meters is None:
            return True
        min_x, max_x, min_y, max_y = self._expanded_bounds()
        start = start_waypoint.transform.location
        end = end_waypoint.transform.location
        pair_min_x = min(float(start.x), float(end.x))
        pair_max_x = max(float(start.x), float(end.x))
        pair_min_y = min(float(start.y), float(end.y))
        pair_max_y = max(float(start.y), float(end.y))
        return not (pair_max_x < min_x or pair_min_x > max_x or pair_max_y < min_y or pair_min_y > max_y)

    def _filter_waypoints_for_bounds(self, waypoints: Sequence[object]) -> List[object]:
        if self.clip_padding_meters is None:
            return list(waypoints)
        return [
            waypoint
            for waypoint in waypoints
            if self._location_in_expanded_bounds(waypoint.transform.location)
        ]

    def _new_surface(self, width: int):
        surface = self.pygame.Surface((width, width), depth=32)
        surface.fill(COLOR_BLACK)
        return surface

    def _follow_road(self, start_waypoint) -> List[object]:
        waypoints = [start_waypoint]
        current = start_waypoint
        seen = set()
        for _ in range(self.max_samples_per_topology_entry):
            next_waypoints = current.next(self.lane_precision_meters)
            if not next_waypoints:
                break
            candidate = next_waypoints[0]
            if candidate.road_id != start_waypoint.road_id:
                break
            key = (
                int(candidate.road_id),
                int(candidate.section_id),
                int(candidate.lane_id),
                round(float(candidate.s), 4),
            )
            if key in seen:
                break
            seen.add(key)
            waypoints.append(candidate)
            current = candidate
        else:
            raise RuntimeError(
                "Exceeded max_samples_per_topology_entry while following "
                f"road_id={start_waypoint.road_id}; possible cyclic topology"
            )
        return waypoints

    def _draw_lane(self, surface, waypoints: Sequence[object], color) -> None:
        lane_left = [
            self._lateral_shift(waypoint.transform, -waypoint.lane_width * 0.5)
            for waypoint in waypoints
        ]
        lane_right = [
            self._lateral_shift(waypoint.transform, waypoint.lane_width * 0.5)
            for waypoint in waypoints
        ]
        polygon = lane_left + list(reversed(lane_right))
        points = [self._world_to_surface_pixel(location) for location in polygon]
        if len(points) > 2:
            self.pygame.draw.polygon(surface, color, points, 5)
            self.pygame.draw.polygon(surface, color, points)

    def _draw_lane_marking_single_side(
        self,
        surfaces: Dict[str, object],
        waypoints: Sequence[object],
        sign: int,
    ) -> None:
        carla = self.carla
        previous_type = carla.LaneMarkingType.NONE
        previous_color = carla.LaneMarkingColor.Other
        current_type = carla.LaneMarkingType.NONE
        markings_list: List[Tuple[object, object, List[Tuple[int, int]]]] = []
        temp_waypoints: List[object] = []

        for sample in waypoints:
            marking = sample.left_lane_marking if sign < 0 else sample.right_lane_marking
            if marking is None:
                continue
            if current_type != marking.type:
                markings_list.extend(
                    self._get_lane_markings(
                        previous_type,
                        previous_color,
                        temp_waypoints,
                        sign,
                    )
                )
                current_type = marking.type
                temp_waypoints = temp_waypoints[-1:]
            else:
                temp_waypoints.append(sample)
                previous_type = marking.type
                previous_color = marking.color

        markings_list.extend(
            self._get_lane_markings(previous_type, previous_color, temp_waypoints, sign)
        )
        for marking_type, marking_color, points in markings_list:
            if marking_color == carla.LaneMarkingColor.White:
                if marking_type == carla.LaneMarkingType.Broken:
                    self._draw_line(surfaces["lane_marking_white_broken"], points, width=1)
                elif marking_type == carla.LaneMarkingType.Solid:
                    self._draw_line(surfaces["lane_marking_white_solid"], points, width=1)
            elif marking_color == carla.LaneMarkingColor.Yellow:
                if marking_type == carla.LaneMarkingType.Broken:
                    self._draw_line(surfaces["lane_marking_yellow_broken"], points, width=1)
                elif marking_type == carla.LaneMarkingType.Solid:
                    self._draw_line(surfaces["lane_marking_yellow_solid"], points, width=1)
            self._draw_line(surfaces["lane_marking_all"], points, width=1)

    def _get_lane_markings(
        self,
        marking_type,
        marking_color,
        waypoints: Sequence[object],
        sign: int,
    ) -> List[Tuple[object, object, List[Tuple[int, int]]]]:
        carla = self.carla
        if not waypoints:
            return []
        margin = 0.25
        marking_1 = [
            self._world_to_surface_pixel(
                self._lateral_shift(sample.transform, sign * sample.lane_width * 0.5)
            )
            for sample in waypoints
        ]
        if marking_type in {carla.LaneMarkingType.Broken, carla.LaneMarkingType.Solid}:
            return [(marking_type, marking_color, marking_1)]

        marking_2 = [
            self._world_to_surface_pixel(
                self._lateral_shift(
                    sample.transform,
                    sign * (sample.lane_width * 0.5 + margin * 2),
                )
            )
            for sample in waypoints
        ]
        if marking_type == carla.LaneMarkingType.SolidBroken:
            return [
                (carla.LaneMarkingType.Broken, marking_color, marking_1),
                (carla.LaneMarkingType.Solid, marking_color, marking_2),
            ]
        if marking_type == carla.LaneMarkingType.BrokenSolid:
            return [
                (carla.LaneMarkingType.Solid, marking_color, marking_1),
                (carla.LaneMarkingType.Broken, marking_color, marking_2),
            ]
        if marking_type == carla.LaneMarkingType.BrokenBroken:
            return [
                (carla.LaneMarkingType.Broken, marking_color, marking_1),
                (carla.LaneMarkingType.Broken, marking_color, marking_2),
            ]
        if marking_type == carla.LaneMarkingType.SolidSolid:
            return [
                (carla.LaneMarkingType.Solid, marking_color, marking_1),
                (carla.LaneMarkingType.Solid, marking_color, marking_2),
            ]
        return [(carla.LaneMarkingType.NONE, marking_color, marking_1)]

    def _draw_line(self, surface, points: Sequence[Tuple[int, int]], width: int) -> None:
        if len(points) >= 2:
            self.pygame.draw.lines(surface, COLOR_WHITE, False, points, int(width))

    def _lateral_shift(self, transform, shift: float):
        rotation = self.carla.Rotation(
            pitch=float(transform.rotation.pitch),
            yaw=float(transform.rotation.yaw) + 90.0,
            roll=float(transform.rotation.roll),
        )
        shifted_transform = self.carla.Transform(transform.location, rotation)
        return transform.location + float(shift) * shifted_transform.get_forward_vector()

    def _world_to_surface_pixel(self, location) -> Tuple[int, int]:
        x = self.bounds.pixels_per_meter * (float(location.x) - self.bounds.min_x_meters)
        y = self.bounds.pixels_per_meter * (float(location.y) - self.bounds.min_y_meters)
        # Roach swaps world x/y when drawing into pygame. pygame array3d then
        # stores [surface_x, surface_y], which matches OpenCV [row, column].
        return int(round(y)), int(round(x))

    def _surface_to_uint8(self, surface) -> np.ndarray:
        # pixels3d is a view and avoids array3d's full RGB temporary copy.
        pixels = self.pygame.surfarray.pixels3d(surface)
        try:
            mask = np.array(pixels[..., 0], dtype=np.uint8, copy=True, order="C")
        finally:
            del pixels
        return mask
