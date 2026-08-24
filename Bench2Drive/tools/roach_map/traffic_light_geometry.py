"""Traffic-light stop-line geometry used by the full Roach map asset."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _is_junction(waypoint) -> bool:
    if hasattr(waypoint, "is_junction"):
        return bool(waypoint.is_junction)
    return bool(waypoint.is_intersection)


def _traffic_light_stopline_vertices(traffic_light, carla_map, carla_module):
    """Port the intended Roach traffic-light trigger-to-stopline conversion."""

    base_transform = traffic_light.get_transform()
    trigger_location = traffic_light.trigger_volume.location
    trigger_extent = traffic_light.trigger_volume.extent
    x_values = np.arange(-0.9 * trigger_extent.x, 0.9 * trigger_extent.x, 1.0)

    initial_waypoints = []
    for x_value in x_values:
        point = base_transform.transform(trigger_location + carla_module.Location(x=float(x_value)))
        waypoint = carla_map.get_waypoint(point)
        if waypoint is None:
            continue
        if (
            not initial_waypoints
            or initial_waypoints[-1].road_id != waypoint.road_id
            or initial_waypoints[-1].lane_id != waypoint.lane_id
        ):
            initial_waypoints.append(waypoint)

    stopline_vertices = []
    for initial_waypoint in initial_waypoints:
        waypoint = initial_waypoint
        seen = set()
        while not _is_junction(waypoint):
            key = (
                int(waypoint.road_id),
                int(waypoint.section_id),
                int(waypoint.lane_id),
                round(float(waypoint.s), 3),
            )
            if key in seen:
                break
            seen.add(key)
            next_waypoints = waypoint.next(0.5)
            if not next_waypoints or _is_junction(next_waypoints[0]):
                break
            waypoint = next_waypoints[0]

        forward = waypoint.transform.get_forward_vector()
        right = carla_module.Vector3D(x=-forward.y, y=forward.x, z=0.0)
        left_location = waypoint.transform.location - 0.4 * waypoint.lane_width * right
        right_location = waypoint.transform.location + 0.4 * waypoint.lane_width * right
        stopline_vertices.append((left_location, right_location))
    return stopline_vertices


def collect_stopline_vertices(world, carla_module) -> List[Tuple[object, object]]:
    """Collect every traffic-light stop line in the currently loaded Town."""

    carla_map = world.get_map()
    vertices: List[Tuple[object, object]] = []
    actors = world.get_actors()
    traffic_lights = actors.filter("*traffic_light*") if hasattr(actors, "filter") else actors
    for actor in traffic_lights:
        if "traffic_light" not in str(actor.type_id):
            continue
        vertices.extend(_traffic_light_stopline_vertices(actor, carla_map, carla_module))
    return vertices
