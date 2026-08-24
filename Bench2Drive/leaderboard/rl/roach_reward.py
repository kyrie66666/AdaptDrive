"""
Supplementary carla-roach reward diagnostics for Bench2Drive training.

This module mirrors the core shaping/terminal logic from:
- carla-roach/carla_gym/core/task_actor/ego_vehicle/reward/valeo_action.py
- carla-roach/carla_gym/core/task_actor/ego_vehicle/terminal/valeo.py

It is intentionally logging-only:
- does not modify the environment reward
- does not change replay contents

Important:
- this file is self-contained on purpose
- it does not import the `carla_gym` package directly
- that avoids the `gym` dependency path inside `carla-roach/carla_gym/__init__.py`
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional

import carla
import numpy as np

def cast_angle(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0


def carla_rot_to_mat(carla_rotation: carla.Rotation) -> np.ndarray:
    roll = np.deg2rad(carla_rotation.roll)
    pitch = np.deg2rad(carla_rotation.pitch)
    yaw = np.deg2rad(carla_rotation.yaw)

    yaw_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])
    pitch_matrix = np.array([
        [np.cos(pitch), 0, -np.sin(pitch)],
        [0, 1, 0],
        [np.sin(pitch), 0, np.cos(pitch)],
    ])
    roll_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(roll), np.sin(roll)],
        [0, -np.sin(roll), np.cos(roll)],
    ])
    return yaw_matrix.dot(pitch_matrix).dot(roll_matrix)


def vec_global_to_ref(target_vec_in_global: carla.Vector3D, ref_rot_in_global: carla.Rotation) -> carla.Vector3D:
    rotation = carla_rot_to_mat(ref_rot_in_global)
    np_vec_in_global = np.array([[target_vec_in_global.x], [target_vec_in_global.y], [target_vec_in_global.z]])
    np_vec_in_ref = rotation.T.dot(np_vec_in_global)
    return carla.Vector3D(x=np_vec_in_ref[0, 0], y=np_vec_in_ref[1, 0], z=np_vec_in_ref[2, 0])


def loc_global_to_ref(target_loc_in_global: carla.Location, ref_trans_in_global: carla.Transform) -> carla.Location:
    x = target_loc_in_global.x - ref_trans_in_global.location.x
    y = target_loc_in_global.y - ref_trans_in_global.location.y
    z = target_loc_in_global.z - ref_trans_in_global.location.z
    vec_in_ref = vec_global_to_ref(carla.Vector3D(x=x, y=y, z=z), ref_trans_in_global.rotation)
    return carla.Location(x=vec_in_ref.x, y=vec_in_ref.y, z=vec_in_ref.z)


def rot_global_to_ref(target_rot_in_global: carla.Rotation, ref_rot_in_global: carla.Rotation) -> carla.Rotation:
    return carla.Rotation(
        roll=cast_angle(target_rot_in_global.roll - ref_rot_in_global.roll),
        pitch=cast_angle(target_rot_in_global.pitch - ref_rot_in_global.pitch),
        yaw=cast_angle(target_rot_in_global.yaw - ref_rot_in_global.yaw),
    )


def lbc_hazard_vehicle(obs_surrounding_vehicles, ev_speed=None, proximity_threshold=9.5):
    del ev_speed
    for i, is_valid in enumerate(obs_surrounding_vehicles['binary_mask']):
        if not is_valid:
            continue
        sv_yaw = obs_surrounding_vehicles['rotation'][i][2]
        same_heading = abs(sv_yaw) <= 150
        sv_loc = obs_surrounding_vehicles['location'][i]
        distance = np.linalg.norm(sv_loc[0:2])
        if distance < 0.001:
            return sv_loc
        if distance <= proximity_threshold:
            angle = np.rad2deg(np.arctan2(sv_loc[1], sv_loc[0]))
            if same_heading and abs(angle) < 45:
                return sv_loc
    return None


def lbc_hazard_walker(obs_surrounding_pedestrians, ev_speed=None, proximity_threshold=9.5):
    del ev_speed
    for i, is_valid in enumerate(obs_surrounding_pedestrians['binary_mask']):
        if not is_valid:
            continue
        if int(obs_surrounding_pedestrians['on_sidewalk'][i]) == 1:
            continue

        ped_loc = obs_surrounding_pedestrians['location'][i]
        dist = np.linalg.norm(ped_loc)
        degree = 162 / (np.clip(dist, 1.5, 10.5) + 0.3)
        if dist < 0.001:
            return ped_loc
        if dist <= proximity_threshold:
            angle = np.rad2deg(np.arctan2(ped_loc[1], ped_loc[0]))
            if abs(angle) < degree:
                return ped_loc
    return None


class RunStopSign:
    def __init__(self, carla_world, proximity_threshold=50.0, speed_threshold=0.1, waypoint_step=1.0):
        self._map = carla_world.get_map()
        self._proximity_threshold = proximity_threshold
        self._speed_threshold = speed_threshold
        self._waypoint_step = waypoint_step
        self._list_stop_signs = [_actor for _actor in carla_world.get_actors() if 'traffic.stop' in _actor.type_id]
        self._target_stop_sign = None
        self._stop_completed = False
        self._affected_by_stop = False

    def tick(self, vehicle, timestamp):
        del timestamp
        ev_loc = vehicle.get_location()

        if self._target_stop_sign is None:
            self._target_stop_sign = self._scan_for_stop_sign(vehicle.get_transform())
        else:
            if not self._stop_completed:
                current_speed = np.linalg.norm([vehicle.get_velocity().x, vehicle.get_velocity().y])
                if current_speed < self._speed_threshold:
                    self._stop_completed = True

            if not self._affected_by_stop:
                stop_t = self._target_stop_sign.get_transform()
                transformed_tv = stop_t.transform(self._target_stop_sign.trigger_volume.location)
                stop_extent = self._target_stop_sign.trigger_volume.extent
                if self.point_inside_boundingbox(ev_loc, transformed_tv, stop_extent):
                    self._affected_by_stop = True

            if not self.is_affected_by_stop(ev_loc, self._target_stop_sign):
                self._target_stop_sign = None
                self._stop_completed = False
                self._affected_by_stop = False

    def _scan_for_stop_sign(self, vehicle_transform):
        ve_dir = vehicle_transform.get_forward_vector()
        wp = self._map.get_waypoint(vehicle_transform.location)
        wp_dir = wp.transform.get_forward_vector()
        dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z
        if dot_ve_wp > 0:
            for stop_sign in self._list_stop_signs:
                if self.is_affected_by_stop(vehicle_transform.location, stop_sign):
                    return stop_sign
        return None

    def is_affected_by_stop(self, vehicle_loc, stop, multi_step=20):
        stop_t = stop.get_transform()
        stop_location = stop_t.location
        if stop_location.distance(vehicle_loc) > self._proximity_threshold:
            return False
        transformed_tv = stop_t.transform(stop.trigger_volume.location)

        list_locations = [vehicle_loc]
        waypoint = self._map.get_waypoint(vehicle_loc)
        for _ in range(multi_step):
            if waypoint is None:
                break
            next_wps = waypoint.next(self._waypoint_step)
            if not next_wps:
                break
            waypoint = next_wps[0]
            if waypoint is None:
                break
            list_locations.append(waypoint.transform.location)

        for actor_location in list_locations:
            if self.point_inside_boundingbox(actor_location, transformed_tv, stop.trigger_volume.extent):
                return True
        return False

    @staticmethod
    def point_inside_boundingbox(point, bb_center, bb_extent):
        bb_extent.x = max(bb_extent.x, bb_extent.y)
        bb_extent.y = max(bb_extent.x, bb_extent.y)
        a = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y - bb_extent.y)
        b = carla.Vector2D(bb_center.x + bb_extent.x, bb_center.y - bb_extent.y)
        d = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y + bb_extent.y)
        m = carla.Vector2D(point.x, point.y)
        ab = b - a
        ad = d - a
        am = m - a
        am_ab = am.x * ab.x + am.y * ab.y
        ab_ab = ab.x * ab.x + ab.y * ab.y
        am_ad = am.x * ad.x + am.y * ad.y
        ad_ad = ad.x * ad.x + ad.y * ad.y
        return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad


def _get_traffic_light_waypoints(traffic_light, carla_map):
    base_transform = traffic_light.get_transform()
    tv_loc = traffic_light.trigger_volume.location
    tv_ext = traffic_light.trigger_volume.extent
    x_values = np.arange(-0.9 * tv_ext.x, 0.9 * tv_ext.x, 1.0)
    area = []
    for x in x_values:
        point_location = base_transform.transform(tv_loc + carla.Location(x=x))
        area.append(point_location)

    ini_wps = []
    for pt in area:
        wpx = carla_map.get_waypoint(pt)
        if not ini_wps or ini_wps[-1].road_id != wpx.road_id or ini_wps[-1].lane_id != wpx.lane_id:
            ini_wps.append(wpx)

    stopline_wps = []
    stopline_vertices = []
    junction_paths = []
    junction_wps = []
    for wpx in ini_wps:
        while not wpx.is_intersection:
            next_wp = wpx.next(0.5)[0]
            if next_wp and not next_wp.is_intersection:
                wpx = next_wp
            else:
                break
        junction_wps.append(wpx)
        stopline_wps.append(wpx)
        vec_forward = wpx.transform.get_forward_vector()
        vec_right = carla.Vector3D(x=-vec_forward.y, y=vec_forward.x, z=0)
        loc_left = wpx.transform.location - 0.4 * wpx.lane_width * vec_right
        loc_right = wpx.transform.location + 0.4 * wpx.lane_width * vec_right
        stopline_vertices.append([loc_left, loc_right])

    queue = list(junction_wps)
    path_wps = []
    while queue:
        current_wp = queue.pop()
        path_wps.append(current_wp)
        next_wps = current_wp.next(1.0)
        for next_wp in next_wps:
            if next_wp.is_junction:
                queue.append(next_wp)
            else:
                junction_paths.append(path_wps)
                path_wps = []

    return carla.Location(base_transform.transform(tv_loc)), stopline_wps, stopline_vertices, junction_paths


class TrafficLightHandler:
    num_tl = 0
    list_tl_actor = []
    list_tv_loc = []
    list_stopline_wps = []
    list_stopline_vtx = []
    list_junction_paths = []
    carla_map = None

    @staticmethod
    def reset(world):
        TrafficLightHandler.carla_map = world.get_map()
        TrafficLightHandler.num_tl = 0
        TrafficLightHandler.list_tl_actor = []
        TrafficLightHandler.list_tv_loc = []
        TrafficLightHandler.list_stopline_wps = []
        TrafficLightHandler.list_stopline_vtx = []
        TrafficLightHandler.list_junction_paths = []
        for actor in world.get_actors():
            if 'traffic_light' in actor.type_id:
                tv_loc, stopline_wps, stopline_vtx, junction_paths = _get_traffic_light_waypoints(actor, TrafficLightHandler.carla_map)
                TrafficLightHandler.list_tl_actor.append(actor)
                TrafficLightHandler.list_tv_loc.append(tv_loc)
                TrafficLightHandler.list_stopline_wps.append(stopline_wps)
                TrafficLightHandler.list_stopline_vtx.append(stopline_vtx)
                TrafficLightHandler.list_junction_paths.append(junction_paths)
                TrafficLightHandler.num_tl += 1

    @staticmethod
    def get_light_state(vehicle, offset=0.0, dist_threshold=15.0):
        vec_tra = vehicle.get_transform()
        veh_dir = vec_tra.get_forward_vector()
        hit_loc = vec_tra.transform(carla.Location(x=offset))
        hit_wp = TrafficLightHandler.carla_map.get_waypoint(hit_loc)

        light_loc = None
        light_state = None
        light_id = None
        for i in range(TrafficLightHandler.num_tl):
            traffic_light = TrafficLightHandler.list_tl_actor[i]
            tv_loc = 0.5 * TrafficLightHandler.list_stopline_wps[i][0].transform.location + 0.5 * TrafficLightHandler.list_stopline_wps[i][-1].transform.location
            distance = np.sqrt((tv_loc.x - hit_loc.x) ** 2 + (tv_loc.y - hit_loc.y) ** 2)
            if distance > dist_threshold:
                continue
            for wp in TrafficLightHandler.list_stopline_wps[i]:
                wp_dir = wp.transform.get_forward_vector()
                dot_ve_wp = veh_dir.x * wp_dir.x + veh_dir.y * wp_dir.y + veh_dir.z * wp_dir.z
                wp_1 = wp.previous(4.0)[0]
                same_road = (hit_wp.road_id == wp.road_id) and (hit_wp.lane_id == wp.lane_id)
                same_road_1 = (hit_wp.road_id == wp_1.road_id) and (hit_wp.lane_id == wp_1.lane_id)
                if (same_road or same_road_1) and dot_ve_wp > 0:
                    loc_in_ev = loc_global_to_ref(wp.transform.location, vec_tra)
                    light_loc = np.array([loc_in_ev.x, loc_in_ev.y, loc_in_ev.z], dtype=np.float32)
                    light_state = traffic_light.state
                    light_id = traffic_light.id
                    break
        return light_state, light_loc, light_id


class _EnvAdapter:
    """Minimal TaskVehicle-like adapter for the logging-only roach reward."""

    def __init__(self, env, criteria_stop):
        self._env = env
        self.vehicle = env.vehicle
        self.criteria_stop = criteria_stop
        self.info_criteria = {
            "run_red_light": None,
            "collision": None,
            "run_stop_sign": None,
            "blocked": None,
        }

    def update(self, env, criteria_stop, info_criteria: Dict[str, object]) -> None:
        self._env = env
        self.vehicle = env.vehicle
        self.criteria_stop = criteria_stop
        self.info_criteria = info_criteria

    def get_route_transform(self):
        if self.vehicle is None:
            return carla.Transform()

        route_waypoints = getattr(self._env, "route_waypoints", None) or []
        if not route_waypoints:
            return self.vehicle.get_transform()

        idx = int(min(max(getattr(self._env, "current_waypoint_idx", 0), 0), len(route_waypoints) - 1))
        loc1 = route_waypoints[idx].transform.location
        if idx > 0:
            loc0 = route_waypoints[idx - 1].transform.location
        else:
            loc0 = self.vehicle.get_location()

        if loc1.distance(loc0) < 0.1:
            yaw = route_waypoints[idx].transform.rotation.yaw
        else:
            forward = loc1 - loc0
            yaw = np.rad2deg(np.arctan2(forward.y, forward.x))
        return carla.Transform(location=loc0, rotation=carla.Rotation(yaw=float(yaw)))


class RoachRewardMonitor:
    """Compute logging-only carla-roach reward diagnostics from the current env."""

    def __init__(self, eval_mode: bool = False, eval_time: float = 1200.0):
        self.eval_mode = bool(eval_mode)
        self.eval_time = float(eval_time)
        self._max_speed = 6.0
        self._min_thresh_lat_dist = 3.5
        self._vehicle_stuck_step = 100
        self._last_steer = 0.0
        self._vehicle_stuck_counter = 0
        self._speed_queue = deque(maxlen=10)
        self._last_lat_dist = 0.0
        self._tl_offset = 0.0
        self._env = None
        self._adapter = None
        self._criteria_stop = None

    def reset(self, env) -> None:
        self._env = env
        self._last_steer = 0.0
        self._vehicle_stuck_counter = 0
        self._speed_queue.clear()
        self._last_lat_dist = 0.0

        if env is None or env.vehicle is None or env.world is None:
            self._adapter = None
            self._criteria_stop = None
            return

        self._criteria_stop = RunStopSign(env.world)
        self._adapter = _EnvAdapter(env, self._criteria_stop)
        self._tl_offset = -0.8 * env.vehicle.bounding_box.extent.x
        TrafficLightHandler.reset(env.world)

    def _observe_surrounding_vehicles(self) -> Dict[str, np.ndarray]:
        vehicle = self._env.vehicle
        world = self._env.world
        ev_transform = vehicle.get_transform()
        ev_location = ev_transform.location
        route_map = world.get_map()

        surrounding = []
        for other in world.get_actors().filter("vehicle.*"):
            if other.id == vehicle.id:
                continue
            if other.get_location().distance(ev_location) > 15.0:
                continue
            surrounding.append(other)
        surrounding.sort(key=lambda actor: actor.get_location().distance(ev_location))

        max_count = 10
        binary_mask, location, rotation, absolute_velocity, road_id, lane_id = [], [], [], [], [], []
        for other in surrounding[:max_count]:
            binary_mask.append(1)
            loc = loc_global_to_ref(other.get_transform().location, ev_transform)
            rot = rot_global_to_ref(other.get_transform().rotation, ev_transform.rotation)
            vel = vec_global_to_ref(other.get_velocity(), ev_transform.rotation)
            location.append([loc.x, loc.y, loc.z])
            rotation.append([rot.roll, rot.pitch, rot.yaw])
            absolute_velocity.append([vel.x, vel.y, vel.z])
            wp = route_map.get_waypoint(other.get_location())
            road_id.append(wp.road_id)
            lane_id.append(wp.lane_id)

        for _ in range(max_count - len(binary_mask)):
            binary_mask.append(0)
            location.append([0.0, 0.0, 0.0])
            rotation.append([0.0, 0.0, 0.0])
            absolute_velocity.append([0.0, 0.0, 0.0])
            road_id.append(0)
            lane_id.append(0)

        return {
            "binary_mask": np.asarray(binary_mask, dtype=np.int8),
            "location": np.asarray(location, dtype=np.float32),
            "rotation": np.asarray(rotation, dtype=np.float32),
            "absolute_velocity": np.asarray(absolute_velocity, dtype=np.float32),
            "road_id": np.asarray(road_id, dtype=np.int8),
            "lane_id": np.asarray(lane_id, dtype=np.int8),
        }

    def _observe_surrounding_pedestrians(self) -> Dict[str, np.ndarray]:
        vehicle = self._env.vehicle
        world = self._env.world
        ev_transform = vehicle.get_transform()
        ev_location = ev_transform.location
        route_map = world.get_map()

        surrounding = []
        for walker in world.get_actors().filter("walker.pedestrian*"):
            if walker.get_location().distance(ev_location) > 15.0:
                continue
            surrounding.append(walker)
        surrounding.sort(key=lambda actor: actor.get_location().distance(ev_location))

        max_count = 10
        binary_mask, location, rotation, absolute_velocity, on_sidewalk, road_id, lane_id = [], [], [], [], [], [], []
        for walker in surrounding[:max_count]:
            binary_mask.append(1)
            loc = loc_global_to_ref(walker.get_transform().location, ev_transform)
            rot = rot_global_to_ref(walker.get_transform().rotation, ev_transform.rotation)
            vel = vec_global_to_ref(walker.get_velocity(), ev_transform.rotation)
            location.append([loc.x, loc.y, loc.z])
            rotation.append([rot.roll, rot.pitch, rot.yaw])
            absolute_velocity.append([vel.x, vel.y, vel.z])
            wp_driving = route_map.get_waypoint(
                walker.get_location(),
                project_to_road=False,
                lane_type=carla.LaneType.Driving,
            )
            on_sidewalk.append(1 if wp_driving is None else 0)
            wp = route_map.get_waypoint(walker.get_location())
            road_id.append(wp.road_id)
            lane_id.append(wp.lane_id)

        for _ in range(max_count - len(binary_mask)):
            binary_mask.append(0)
            location.append([0.0, 0.0, 0.0])
            rotation.append([0.0, 0.0, 0.0])
            absolute_velocity.append([0.0, 0.0, 0.0])
            on_sidewalk.append(0)
            road_id.append(0)
            lane_id.append(0)

        return {
            "binary_mask": np.asarray(binary_mask, dtype=np.int8),
            "location": np.asarray(location, dtype=np.float32),
            "rotation": np.asarray(rotation, dtype=np.float32),
            "absolute_velocity": np.asarray(absolute_velocity, dtype=np.float32),
            "on_sidewalk": np.asarray(on_sidewalk, dtype=np.int8),
            "road_id": np.asarray(road_id, dtype=np.int8),
            "lane_id": np.asarray(lane_id, dtype=np.int8),
        }

    def compute(self, info: Optional[Dict[str, object]]) -> Dict[str, object]:
        if self._env is None or self._env.vehicle is None or self._adapter is None:
            return {}

        timestamp = {
            "step": int((info or {}).get("step", 0)),
            "relative_simulation_time": float((info or {}).get("elapsed_time", 0.0)),
        }
        self._criteria_stop.tick(self._env.vehicle, timestamp)

        termination_reasons = set()
        for reason in (info or {}).get("termination_reasons", []) or []:
            termination_reasons.add(str(reason))
        termination = (info or {}).get("termination")
        if termination:
            termination_reasons.add(str(termination))

        info_criteria = {
            "run_red_light": {"event": "run"} if "red_light" in termination_reasons else None,
            "collision": {"event": "collision"} if "collision" in termination_reasons else None,
            "run_stop_sign": {"event": "run"} if "run_stop_sign" in termination_reasons else None,
            "blocked": {"event": "blocked"} if "vehicle_blocked" in termination_reasons else None,
        }
        self._adapter.update(self._env, self._criteria_stop, info_criteria)

        ev_transform = self._env.vehicle.get_transform()
        ev_control = self._env.vehicle.get_control()
        ev_vel = self._env.vehicle.get_velocity()
        ev_speed = float(np.linalg.norm(np.array([ev_vel.x, ev_vel.y], dtype=np.float32)))

        if abs(ev_control.steer - self._last_steer) > 0.01:
            r_action = -0.1
        else:
            r_action = 0.0
        self._last_steer = ev_control.steer

        obs_vehicle = self._observe_surrounding_vehicles()
        obs_pedestrian = self._observe_surrounding_pedestrians()
        hazard_vehicle_loc = lbc_hazard_vehicle(obs_vehicle, proximity_threshold=9.5)
        hazard_ped_loc = lbc_hazard_walker(obs_pedestrian, proximity_threshold=9.5)
        light_state, light_loc, _ = TrafficLightHandler.get_light_state(
            self._env.vehicle, offset=self._tl_offset, dist_threshold=18.0
        )

        desired_spd_veh = desired_spd_ped = desired_spd_rl = desired_spd_stop = self._max_speed

        if hazard_vehicle_loc is not None:
            dist_veh = max(0.0, np.linalg.norm(hazard_vehicle_loc[0:2]) - 8.0)
            desired_spd_veh = self._max_speed * np.clip(dist_veh, 0.0, 5.0) / 5.0

        if hazard_ped_loc is not None:
            dist_ped = max(0.0, np.linalg.norm(hazard_ped_loc[0:2]) - 6.0)
            desired_spd_ped = self._max_speed * np.clip(dist_ped, 0.0, 5.0) / 5.0

        if light_state in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow):
            dist_rl = max(0.0, np.linalg.norm(light_loc[0:2]) - 5.0)
            desired_spd_rl = self._max_speed * np.clip(dist_rl, 0.0, 5.0) / 5.0

        stop_sign = self._criteria_stop._target_stop_sign
        stop_loc = None
        if (stop_sign is not None) and (not self._criteria_stop._stop_completed):
            trans = stop_sign.get_transform()
            tv_loc = stop_sign.trigger_volume.location
            loc_in_world = trans.transform(tv_loc)
            loc_in_ev = loc_global_to_ref(loc_in_world, ev_transform)
            stop_loc = np.array([loc_in_ev.x, loc_in_ev.y, loc_in_ev.z], dtype=np.float32)
            dist_stop = max(0.0, np.linalg.norm(stop_loc[0:2]) - 5.0)
            desired_spd_stop = self._max_speed * np.clip(dist_stop, 0.0, 5.0) / 5.0

        desired_speed = float(min(self._max_speed, desired_spd_veh, desired_spd_ped, desired_spd_rl, desired_spd_stop))
        r_speed = 1.0 - abs(ev_speed - desired_speed) / self._max_speed

        route_transform = self._adapter.get_route_transform()
        d_vec = ev_transform.location - route_transform.location
        np_d_vec = np.array([d_vec.x, d_vec.y], dtype=np.float32)
        wp_unit_forward = route_transform.rotation.get_forward_vector()
        np_wp_unit_right = np.array([-wp_unit_forward.y, wp_unit_forward.x], dtype=np.float32)
        lateral_distance = float(abs(np.dot(np_wp_unit_right, np_d_vec)))
        r_position = -1.0 * (lateral_distance / 2.0)

        angle_difference = float(np.deg2rad(abs(cast_angle(ev_transform.rotation.yaw - route_transform.rotation.yaw))))
        r_rotation = -1.0 * angle_difference

        self._speed_queue.append(ev_speed)
        is_free_road = (
            hazard_vehicle_loc is None
            and hazard_ped_loc is None
            and (light_state is None or light_state == carla.TrafficLightState.Green)
        )
        if is_free_road and float(np.mean(self._speed_queue)) < 1.0:
            self._vehicle_stuck_counter += 1
        if float(np.mean(self._speed_queue)) >= 1.0:
            self._vehicle_stuck_counter = 0
        c_vehicle_stuck = self._vehicle_stuck_counter >= self._vehicle_stuck_step

        if lateral_distance - self._last_lat_dist > 0.8:
            thresh_lat_dist = lateral_distance + 0.5
        else:
            thresh_lat_dist = max(self._min_thresh_lat_dist, self._last_lat_dist)
        c_lat_dist = lateral_distance > thresh_lat_dist + 1e-2
        self._last_lat_dist = lateral_distance

        c_run_rl = info_criteria["run_red_light"] is not None
        c_collision = info_criteria["collision"] is not None
        c_run_stop = info_criteria["run_stop_sign"] is not None
        c_blocked = info_criteria["blocked"] is not None
        timeout = self.eval_mode and timestamp["relative_simulation_time"] > self.eval_time
        done = bool(c_vehicle_stuck or c_lat_dist or c_run_rl or c_collision or c_run_stop or c_blocked or timeout)

        terminal_reward = 0.0
        if done:
            terminal_reward = -1.0
        if c_run_rl or c_collision or c_run_stop:
            terminal_reward -= ev_speed

        roach_reward = float(r_speed + r_position + r_rotation + terminal_reward + r_action)

        if hazard_vehicle_loc is None:
            txt_hazard_veh = "[]"
        else:
            txt_hazard_veh = np.array2string(hazard_vehicle_loc[0:2], precision=1, separator=",", suppress_small=True)
        if hazard_ped_loc is None:
            txt_hazard_ped = "[]"
        else:
            txt_hazard_ped = np.array2string(hazard_ped_loc[0:2], precision=1, separator=",", suppress_small=True)
        if light_loc is None:
            txt_light = "[]"
        else:
            txt_light = np.array2string(light_loc[0:2], precision=1, separator=",", suppress_small=True)
        if stop_loc is None:
            txt_stop = "[]"
        else:
            txt_stop = np.array2string(stop_loc[0:2], precision=1, separator=",", suppress_small=True)

        debug_texts = [
            f'roach_r:{roach_reward:5.2f} rp:{r_position:5.2f} rr:{r_rotation:5.2f}',
            f'roach_ds:{desired_speed:5.2f} rs:{r_speed:5.2f} ra:{r_action:5.2f}',
            f'roach_veh_ds:{desired_spd_veh:5.2f} {txt_hazard_veh}',
            f'roach_ped_ds:{desired_spd_ped:5.2f} {txt_hazard_ped}',
            f'roach_tl_ds:{desired_spd_rl:5.2f} {light_state}{txt_light}',
            f'roach_st_ds:{desired_spd_stop:5.2f} {txt_stop}',
            f'roach_term:{terminal_reward:5.2f}',
        ]

        return {
            "roach_reward": roach_reward,
            "roach_r_speed": float(r_speed),
            "roach_r_position": float(r_position),
            "roach_r_rotation": float(r_rotation),
            "roach_r_action": float(r_action),
            "roach_r_terminal": float(terminal_reward),
            "roach_desired_speed": float(desired_speed),
            "roach_lateral_distance": float(lateral_distance),
            "roach_angle_difference": float(angle_difference),
            "roach_vehicle_stuck_counter": int(self._vehicle_stuck_counter),
            "roach_is_free_road": bool(is_free_road),
            "roach_done": bool(done),
            "roach_timeout": bool(timeout),
            "roach_debug_texts": debug_texts,
        }
