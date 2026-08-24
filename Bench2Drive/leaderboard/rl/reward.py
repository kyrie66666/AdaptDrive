"""
Reward Function for Bench2Drive RL (carla-roach Valeo-style)
============================================================

Ports the local carla-roach reward structure into Bench2Drive:
- shaping from `valeo_action.py`
- terminal logic from `terminal/valeo.py`

Bench2Drive keeps route progress as diagnostic info in env.py, but it is no
longer part of the RL reward itself.
"""

from collections import deque
from dataclasses import dataclass
import math
from typing import Dict, Optional, Set, Tuple

import carla
import numpy as np
from srunner.scenariomanager.traffic_events import TrafficEventType


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""

    # carla-roach Valeo reward parameters
    max_speed: float = 6.0
    steer_change_penalty: float = 0.02
    hazard_vehicle_dist: float = 9.5
    hazard_pedestrian_dist: float = 9.5
    traffic_light_dist: float = 18.0
    stop_sign_dist: float = 10.0
    tl_offset_factor: float = 0.8
    route_reference_lookahead: int = 25
    route_membership_lookahead: int = 20
    vehicle_stuck_step: int = 100
    hard_stuck_step: int = 200
    speed_queue_size: int = 10
    stuck_speed_threshold: float = 1.0
    same_direction_heading_threshold_deg: float = 45.0
    rotation_ok_deg: float = 20.0
    rotation_warn_deg: float = 45.0
    rotation_bad_deg: float = 90.0
    recoverable_max_distance: float = 5.0
    bad_off_route_distance: float = 12.0
    route_recoverable_penalty: float = -0.05
    route_bad_penalty: float = -0.30
    route_progress_reward_scale: float = 20.0
    terminal_penalty_collision: float = -20.0
    terminal_penalty_off_route: float = -20.0
    terminal_penalty_red_light: float = -8.0
    terminal_penalty_run_stop: float = -8.0
    eval_mode: bool = False
    eval_time: float = 1200.0

    # Line E is opt-in. Defaults preserve the legacy C/D reward exactly.
    reward_variant: str = "legacy"
    enable_line_e_reward: bool = False
    terminal_penalty_blocked: float = -10.0
    terminal_penalty_timeout: float = -10.0
    line_e_hard_stuck_terminal_penalty: float = -10.0
    # Line-E-only safe-wait classification.  The legacy Roach reward keeps its
    # original actor-centre cutoff; this extra check aligns Line E with the
    # bbox geometry used by the frozen clean collision rescore.
    line_e_bbox_safety_wait: bool = True
    line_e_safety_wait_clearance: float = 9.5
    line_e_safety_wait_lateral_margin: float = 0.5
    line_e_safe_wait_stop_speed: float = 0.2
    line_e_safe_wait_reward_grace_steps: int = 20
    line_e_blocked_timeout_steps: int = 200
    free_road_efficiency_scale: float = 0.20
    low_speed_threshold: float = 1.0
    low_speed_grace_steps: int = 20
    low_speed_penalty_per_step: float = -0.03
    low_speed_episode_cap: float = -4.0
    no_progress_threshold: float = 1e-4
    no_progress_grace_steps: int = 30
    no_progress_penalty_per_step: float = -0.02
    no_progress_episode_cap: float = -4.0
    enable_comfort_penalty: bool = False
    comfort_penalty_max: float = -0.03
    enable_direct_dense_safety: bool = False
    enable_dense_ttc: bool = True
    enable_dense_headway: bool = True
    enable_dense_min_distance: bool = True
    direct_dense_ttc_weight: float = 1.0
    direct_dense_headway_weight: float = 1.0
    direct_dense_min_distance_weight: float = 1.0
    dense_safety_forward_distance: float = 35.0
    dense_safety_lateral_distance: float = 2.7
    dense_safety_same_direction_dot: float = 0.3
    dense_safety_min_closing_speed: float = 0.1
    ttc_safe_time: float = 4.0
    ttc_min_time: float = 0.5
    dense_ttc_penalty_max: float = -0.40
    headway_time: float = 1.5
    headway_min_distance: float = 4.0
    dense_headway_penalty_max: float = -0.20
    min_distance_safe: float = 3.0
    dense_min_distance_penalty_max: float = -0.10
    dense_safety_penalty_cap: float = -0.60

    # Legacy fields kept for compatibility with existing config dumps/helpers.
    position_factor: float = 4.0
    speed_reward_scale: float = 0.05
    action_reward_scale: float = 0.2
    position_reward_scale: float = 0.05
    rotation_reward_scale: float = 0.05
    terminal_penalty: float = -5.0
    progress_reward_scale: float = 100.0
    success_reward: float = 15.0
    corridor_penalty_recoverable: float = -0.05
    corridor_penalty_unrecoverable: float = -0.30
    collision_event_penalty_base: float = 20.0
    collision_event_penalty_progress_scale: float = 20.0
    off_route_event_penalty_base: float = 15.0
    off_route_event_penalty_progress_scale: float = 15.0
    stuck_event_penalty_base: float = 10.0
    stuck_event_penalty_progress_scale: float = 10.0
    wrong_world_event_penalty: float = 50.0
    max_distance_from_route: float = 10.0
    max_stuck_time: float = 10.0
    spawn_grace_steps: int = 10
    soft_stuck_time: float = 2.0
    soft_stuck_penalty_per_step: float = 0.005
    soft_stuck_penalty_cap: float = 0.03
    hard_stuck_time_no_adjacent_lane: float = 6.0
    hard_stuck_time_with_adjacent_lane: float = 9.0
    hard_stuck_terminal_penalty: float = -5.0
    stuck_min_displacement_m: float = 1.0
    stuck_min_progress_delta: float = 0.002


class RewardCalculator:
    """Calculate carla-roach style reward for Bench2Drive."""

    def __init__(self, config: RewardConfig):
        self.config = config
        self._collision_detected = False
        self._lane_invasion_detected = False
        self._last_steer = 0.0
        self._last_throttle = 0.0
        self._last_brake = 0.0
        self._vehicle_stuck_counter = 0
        self._safe_wait_stopped_counter = 0
        self._blocked_wait_counter = 0
        self._blocked_wait_actor_id = -1
        self._low_speed_counter = 0
        self._no_progress_counter = 0
        self._low_speed_penalty_total = 0.0
        self._no_progress_penalty_total = 0.0
        self._speed_queue = deque(maxlen=max(1, int(config.speed_queue_size)))
        self._last_lat_dist = 0.0
        self._seen_event_keys: Set[Tuple[int, str, str]] = set()

    def reset(self, route_scenario=None):
        """Reset reward state and seed existing scenario events."""
        self._collision_detected = False
        self._lane_invasion_detected = False
        self._last_steer = 0.0
        self._last_throttle = 0.0
        self._last_brake = 0.0
        self._vehicle_stuck_counter = 0
        self._safe_wait_stopped_counter = 0
        self._blocked_wait_counter = 0
        self._blocked_wait_actor_id = -1
        self._low_speed_counter = 0
        self._no_progress_counter = 0
        self._low_speed_penalty_total = 0.0
        self._no_progress_penalty_total = 0.0
        self._speed_queue.clear()
        self._last_lat_dist = 0.0
        self._seen_event_keys = set()
        self._seed_seen_events(route_scenario)

    def _seed_seen_events(self, route_scenario) -> None:
        for criterion in self._iter_criteria(route_scenario):
            for event in getattr(criterion, "events", []):
                self._seen_event_keys.add(self._event_key(event))

    @staticmethod
    def _iter_criteria(route_scenario):
        get_criteria = getattr(route_scenario, "get_criteria", None)
        if get_criteria is None:
            return []
        try:
            return get_criteria() or []
        except Exception:
            return []

    def _collect_new_events(self, route_scenario):
        new_events = []
        for criterion in self._iter_criteria(route_scenario):
            for event in getattr(criterion, "events", []):
                key = self._event_key(event)
                if key in self._seen_event_keys:
                    continue
                self._seen_event_keys.add(key)
                new_events.append(event)
        return new_events

    @staticmethod
    def _event_key(event) -> Tuple[int, str, str]:
        return (int(event.get_frame()), str(event.get_type()), event.get_message())

    def _select_reference_waypoint(
        self,
        vehicle_location: carla.Location,
        route_waypoints: list,
        lookahead: Optional[int] = None,
    ) -> Optional[carla.Waypoint]:
        """Select a roach-style current route waypoint from the route front."""
        if not route_waypoints:
            return None

        if lookahead is None:
            lookahead = self.config.route_reference_lookahead
        candidate_count = min(len(route_waypoints), int(lookahead))
        candidates = route_waypoints[:candidate_count]
        return min(candidates, key=lambda wp: vehicle_location.distance(wp.transform.location))

    def _same_direction(self, current_wp: carla.Waypoint, route_wp: carla.Waypoint) -> bool:
        heading_diff = abs(
            self._normalize_angle(current_wp.transform.rotation.yaw - route_wp.transform.rotation.yaw)
        )
        return heading_diff <= float(self.config.same_direction_heading_threshold_deg)

    def _is_route_acceptable(self, current_wp: Optional[carla.Waypoint], route_waypoints: list) -> bool:
        if current_wp is None or not route_waypoints:
            return False

        lookahead = max(1, int(self.config.route_membership_lookahead))
        for route_wp in route_waypoints[:lookahead]:
            if route_wp.lane_type != carla.LaneType.Driving:
                continue
            if current_wp.road_id == route_wp.road_id and current_wp.lane_id == route_wp.lane_id:
                return True

            for neighbor in (route_wp.get_left_lane(), route_wp.get_right_lane()):
                if neighbor is None:
                    continue
                if neighbor.lane_type != carla.LaneType.Driving:
                    continue
                if neighbor.road_id != route_wp.road_id:
                    continue
                if not self._same_direction(neighbor, route_wp):
                    continue
                if current_wp.road_id == neighbor.road_id and current_wp.lane_id == neighbor.lane_id:
                    return True

        return False

    def _get_hazard_vehicle_speed(self, vehicle: carla.Vehicle) -> float:
        """Check for hazard vehicles and return desired speed."""
        world = vehicle.get_world()
        ev_transform = vehicle.get_transform()
        ev_location = ev_transform.location
        ev_forward = ev_transform.rotation.get_forward_vector()

        vehicles = world.get_actors().filter("vehicle.*")
        min_desired_speed = self.config.max_speed

        for other in vehicles:
            if other.id == vehicle.id:
                continue

            other_location = other.get_transform().location
            distance = ev_location.distance(other_location)
            if distance > self.config.hazard_vehicle_dist:
                continue

            direction = np.array(
                [other_location.x - ev_location.x, other_location.y - ev_location.y],
                dtype=np.float32,
            )
            forward = np.array([ev_forward.x, ev_forward.y], dtype=np.float32)
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 0.1:
                continue

            direction = direction / direction_norm
            forward = forward / (np.linalg.norm(forward) + 1e-8)

            if np.dot(forward, direction) > 0.5:
                effective_dist = max(0.0, distance - 8.0)
                desired_spd = self.config.max_speed * np.clip(effective_dist / 5.0, 0.0, 1.0)
                min_desired_speed = min(min_desired_speed, desired_spd)

        return float(min_desired_speed)

    def _get_hazard_pedestrian_speed(self, vehicle: carla.Vehicle) -> float:
        """Check for hazard pedestrians and return desired speed."""
        world = vehicle.get_world()
        ev_transform = vehicle.get_transform()
        ev_location = ev_transform.location
        ev_forward = ev_transform.rotation.get_forward_vector()

        walkers = world.get_actors().filter("walker.*")
        min_desired_speed = self.config.max_speed

        for walker in walkers:
            walker_location = walker.get_transform().location
            distance = ev_location.distance(walker_location)
            if distance > self.config.hazard_pedestrian_dist:
                continue

            direction = np.array(
                [walker_location.x - ev_location.x, walker_location.y - ev_location.y],
                dtype=np.float32,
            )
            forward = np.array([ev_forward.x, ev_forward.y], dtype=np.float32)
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 0.1:
                continue

            direction = direction / direction_norm
            forward = forward / (np.linalg.norm(forward) + 1e-8)

            if np.dot(forward, direction) > 0.3:
                effective_dist = max(0.0, distance - 6.0)
                desired_spd = self.config.max_speed * np.clip(effective_dist / 5.0, 0.0, 1.0)
                min_desired_speed = min(min_desired_speed, desired_spd)

        return float(min_desired_speed)

    def _get_traffic_light_speed(self, vehicle: carla.Vehicle) -> Tuple[float, Optional[carla.TrafficLightState]]:
        """Check for traffic lights and return desired speed and current state."""
        world = vehicle.get_world()
        ev_transform = vehicle.get_transform()
        ev_location = ev_transform.location
        ev_forward = ev_transform.rotation.get_forward_vector()

        traffic_lights = world.get_actors().filter("traffic.traffic_light")
        min_desired_speed = self.config.max_speed
        light_state = None

        for tl in traffic_lights:
            tl_location = tl.get_transform().location
            distance = ev_location.distance(tl_location)
            if distance > self.config.traffic_light_dist:
                continue

            direction = np.array(
                [tl_location.x - ev_location.x, tl_location.y - ev_location.y],
                dtype=np.float32,
            )
            forward = np.array([ev_forward.x, ev_forward.y], dtype=np.float32)
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 0.1:
                continue

            direction = direction / direction_norm
            forward = forward / (np.linalg.norm(forward) + 1e-8)

            if np.dot(forward, direction) > 0.7:
                state = tl.state
                light_state = state
                if state in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow):
                    effective_dist = max(0.0, distance - 5.0)
                    desired_spd = self.config.max_speed * np.clip(effective_dist / 5.0, 0.0, 1.0)
                    min_desired_speed = min(min_desired_speed, desired_spd)

        return float(min_desired_speed), light_state

    def _get_stop_sign_speed(self, vehicle: carla.Vehicle) -> float:
        """Check for stop signs and return desired speed."""
        world = vehicle.get_world()
        ev_transform = vehicle.get_transform()
        ev_location = ev_transform.location
        ev_forward = ev_transform.rotation.get_forward_vector()

        stop_signs = world.get_actors().filter("traffic.stop")
        min_desired_speed = self.config.max_speed

        for ss in stop_signs:
            ss_transform = ss.get_transform()
            if hasattr(ss, "trigger_volume"):
                tv_location = ss.trigger_volume.location
                ss_location = ss_transform.transform(tv_location)
            else:
                ss_location = ss_transform.location

            distance = ev_location.distance(ss_location)
            if distance > self.config.stop_sign_dist:
                continue

            direction = np.array(
                [ss_location.x - ev_location.x, ss_location.y - ev_location.y],
                dtype=np.float32,
            )
            forward = np.array([ev_forward.x, ev_forward.y], dtype=np.float32)
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 0.1:
                continue

            direction = direction / direction_norm
            forward = forward / (np.linalg.norm(forward) + 1e-8)

            if np.dot(forward, direction) > 0.6:
                effective_dist = max(0.0, distance - 5.0)
                desired_spd = self.config.max_speed * np.clip(effective_dist / 5.0, 0.0, 1.0)
                min_desired_speed = min(min_desired_speed, desired_spd)

        return float(min_desired_speed)

    def _line_e_enabled(self) -> bool:
        return bool(self.config.enable_line_e_reward or str(self.config.reward_variant).lower() == "line_e")

    @staticmethod
    def is_failure_truncation_reason(reason: Optional[str]) -> bool:
        if reason is None:
            return False
        normalized = str(reason).strip().lower()
        return normalized in {"timeout", "max_tick_count", "max_episode_steps", "tickruntime", "tick_runtime"}

    @staticmethod
    def _xy_unit(vector: carla.Vector3D) -> np.ndarray:
        arr = np.array([float(vector.x), float(vector.y)], dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm < 1e-6:
            return np.array([1.0, 0.0], dtype=np.float32)
        return arr / norm

    def _actor_bbox_projection(
        self,
        actor,
        reference_forward: np.ndarray,
        reference_right: np.ndarray,
    ) -> Dict[str, object]:
        """Project one actor's oriented bbox onto ego longitudinal/lateral axes."""

        transform = actor.get_transform()
        actor_forward = self._xy_unit(transform.rotation.get_forward_vector())
        actor_right = np.array([-actor_forward[1], actor_forward[0]], dtype=np.float32)
        bbox = getattr(actor, "bounding_box", None)

        local_center_x = float(getattr(getattr(bbox, "location", None), "x", 0.0))
        local_center_y = float(getattr(getattr(bbox, "location", None), "y", 0.0))
        center = np.array(
            [float(transform.location.x), float(transform.location.y)],
            dtype=np.float32,
        )
        center = center + local_center_x * actor_forward + local_center_y * actor_right

        bbox_yaw = math.radians(float(getattr(getattr(bbox, "rotation", None), "yaw", 0.0)))
        bbox_forward = math.cos(bbox_yaw) * actor_forward + math.sin(bbox_yaw) * actor_right
        bbox_right = -math.sin(bbox_yaw) * actor_forward + math.cos(bbox_yaw) * actor_right
        extent_x = max(0.0, float(getattr(getattr(bbox, "extent", None), "x", 0.0)))
        extent_y = max(0.0, float(getattr(getattr(bbox, "extent", None), "y", 0.0)))

        half_longitudinal = (
            abs(float(np.dot(bbox_forward, reference_forward))) * extent_x
            + abs(float(np.dot(bbox_right, reference_forward))) * extent_y
        )
        half_lateral = (
            abs(float(np.dot(bbox_forward, reference_right))) * extent_x
            + abs(float(np.dot(bbox_right, reference_right))) * extent_y
        )
        return {
            "center": center,
            "half_longitudinal": float(half_longitudinal),
            "half_lateral": float(half_lateral),
            "forward": actor_forward,
        }

    @staticmethod
    def _empty_line_e_blocker_info() -> Dict[str, object]:
        return {
            "active": False,
            "actor_id": -1,
            "actor_type": "",
            "actor_speed": 0.0,
            "center_distance": float("inf"),
            "longitudinal_center_distance": float("inf"),
            "lateral_center_offset": float("inf"),
            "longitudinal_clearance": float("inf"),
            "lateral_clearance": float("inf"),
            "lateral_overlap": 0.0,
            "heading_dot": 0.0,
        }

    def _find_line_e_vehicle_blocker(self, vehicle: carla.Vehicle) -> Dict[str, object]:
        """Find the nearest same-direction front vehicle by oriented bbox clearance.

        This is deliberately a Line-E reward/termination diagnostic.  It does
        not alter the frozen clean speed decoder, collision rescore, SAC action,
        or PID command.
        """

        result = self._empty_line_e_blocker_info()
        if not bool(self.config.line_e_bbox_safety_wait):
            return result

        try:
            transform = vehicle.get_transform()
            ego_forward = self._xy_unit(transform.rotation.get_forward_vector())
            ego_right = np.array([-ego_forward[1], ego_forward[0]], dtype=np.float32)
            ego_bbox = self._actor_bbox_projection(vehicle, ego_forward, ego_right)
            vehicles = vehicle.get_world().get_actors().filter("vehicle.*")
        except Exception:
            return result

        ego_center = ego_bbox["center"]
        max_clearance = float(self.config.line_e_safety_wait_clearance)
        lateral_margin = float(self.config.line_e_safety_wait_lateral_margin)
        min_heading_dot = math.cos(math.radians(float(self.config.same_direction_heading_threshold_deg)))

        for actor in vehicles:
            try:
                if int(actor.id) == int(vehicle.id):
                    continue
                actor_bbox = self._actor_bbox_projection(actor, ego_forward, ego_right)
                delta = actor_bbox["center"] - ego_center
                longitudinal_center = float(np.dot(delta, ego_forward))
                lateral_center = float(np.dot(delta, ego_right))
                if longitudinal_center <= 0.0:
                    continue

                heading_dot = float(np.dot(actor_bbox["forward"], ego_forward))
                if heading_dot < min_heading_dot:
                    continue

                longitudinal_clearance = (
                    longitudinal_center
                    - float(ego_bbox["half_longitudinal"])
                    - float(actor_bbox["half_longitudinal"])
                )
                if longitudinal_clearance > max_clearance:
                    continue

                combined_half_width = float(ego_bbox["half_lateral"]) + float(actor_bbox["half_lateral"])
                lateral_clearance = abs(lateral_center) - combined_half_width
                if lateral_clearance > lateral_margin:
                    continue

                center_distance = float(np.linalg.norm(delta))
                if result["active"] and longitudinal_clearance >= float(result["longitudinal_clearance"]):
                    continue

                actor_velocity = actor.get_velocity()
                actor_speed = float(np.linalg.norm([float(actor_velocity.x), float(actor_velocity.y)]))
                result = {
                    "active": True,
                    "actor_id": int(actor.id),
                    "actor_type": str(getattr(actor, "type_id", "")),
                    "actor_speed": actor_speed,
                    "center_distance": center_distance,
                    "longitudinal_center_distance": longitudinal_center,
                    "lateral_center_offset": lateral_center,
                    "longitudinal_clearance": longitudinal_clearance,
                    "lateral_clearance": lateral_clearance,
                    "lateral_overlap": max(0.0, -lateral_clearance),
                    "heading_dot": heading_dot,
                }
            except Exception:
                continue

        return result

    def _compute_direct_dense_safety(
        self,
        vehicle: carla.Vehicle,
        speed: float,
        line_e_free_road: bool,
        legal_wait: bool,
    ) -> Dict[str, object]:
        """Lightweight front-corridor TTC/headway/min-distance shaping."""
        defaults: Dict[str, object] = {
            "r_dense_ttc": 0.0,
            "r_dense_headway": 0.0,
            "r_dense_min_distance": 0.0,
            "r_dense_safety_direct": 0.0,
            "min_ttc": float("inf"),
            "min_headway_distance": float("inf"),
            "nearest_actor_type": "",
            "nearest_actor_distance": float("inf"),
            "nearest_actor_longitudinal_gap": float("inf"),
            "nearest_actor_lateral_gap": float("inf"),
            "nearest_actor_relative_speed": 0.0,
        }
        if not bool(self.config.enable_direct_dense_safety):
            return defaults
        if legal_wait or not line_e_free_road:
            return defaults

        try:
            world = vehicle.get_world()
            transform = vehicle.get_transform()
            ego_location = transform.location
            ego_forward = self._xy_unit(transform.rotation.get_forward_vector())
            ego_right = np.array([-ego_forward[1], ego_forward[0]], dtype=np.float32)
            ego_velocity = vehicle.get_velocity()
            ego_velocity_xy = np.array([float(ego_velocity.x), float(ego_velocity.y)], dtype=np.float32)
            ego_along = float(np.dot(ego_velocity_xy, ego_forward))
            actors = list(world.get_actors().filter("vehicle.*")) + list(world.get_actors().filter("walker.*"))
        except Exception:
            return defaults

        forward_limit = float(self.config.dense_safety_forward_distance)
        lateral_limit = float(self.config.dense_safety_lateral_distance)
        same_direction_dot = float(self.config.dense_safety_same_direction_dot)
        min_closing_speed = float(self.config.dense_safety_min_closing_speed)

        min_ttc = float("inf")
        min_headway = float("inf")
        nearest_distance = float("inf")
        nearest_actor_type = ""
        nearest_longitudinal = float("inf")
        nearest_lateral = float("inf")
        nearest_relative_speed = 0.0

        for actor in actors:
            try:
                if actor.id == vehicle.id:
                    continue
                type_id = str(getattr(actor, "type_id", ""))
                actor_location = actor.get_transform().location
                delta = np.array(
                    [
                        float(actor_location.x - ego_location.x),
                        float(actor_location.y - ego_location.y),
                    ],
                    dtype=np.float32,
                )
                longitudinal_gap = float(np.dot(delta, ego_forward))
                lateral_gap = float(np.dot(delta, ego_right))
                if longitudinal_gap <= 0.5 or longitudinal_gap > forward_limit:
                    continue
                if abs(lateral_gap) > lateral_limit:
                    continue

                if type_id.startswith("vehicle."):
                    actor_forward = self._xy_unit(actor.get_transform().rotation.get_forward_vector())
                    if float(np.dot(actor_forward, ego_forward)) < same_direction_dot:
                        continue

                distance = float(np.linalg.norm(delta))
                actor_velocity = actor.get_velocity()
                actor_velocity_xy = np.array([float(actor_velocity.x), float(actor_velocity.y)], dtype=np.float32)
                actor_along = float(np.dot(actor_velocity_xy, ego_forward))
                closing_speed = max(0.0, ego_along - actor_along)

                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_actor_type = type_id
                    nearest_longitudinal = longitudinal_gap
                    nearest_lateral = lateral_gap
                    nearest_relative_speed = closing_speed

                min_headway = min(min_headway, longitudinal_gap)
                if closing_speed > min_closing_speed:
                    effective_gap = max(0.1, longitudinal_gap - 4.0)
                    min_ttc = min(min_ttc, float(effective_gap / closing_speed))
            except Exception:
                continue

        r_dense_ttc = 0.0
        if bool(self.config.enable_dense_ttc) and math.isfinite(min_ttc):
            safe_time = max(float(self.config.ttc_safe_time), float(self.config.ttc_min_time) + 1e-3)
            if min_ttc < safe_time:
                denom = max(1e-3, safe_time - float(self.config.ttc_min_time))
                risk = float(np.clip((safe_time - min_ttc) / denom, 0.0, 1.0))
                r_dense_ttc = -abs(float(self.config.dense_ttc_penalty_max)) * float(self.config.direct_dense_ttc_weight) * (risk ** 2)

        r_dense_headway = 0.0
        if bool(self.config.enable_dense_headway) and math.isfinite(min_headway):
            safe_headway = max(float(self.config.headway_min_distance), float(self.config.headway_time) * max(0.0, speed))
            if safe_headway > 1e-3 and min_headway < safe_headway:
                risk = float(np.clip((safe_headway - min_headway) / safe_headway, 0.0, 1.0))
                r_dense_headway = -abs(float(self.config.dense_headway_penalty_max)) * float(self.config.direct_dense_headway_weight) * (risk ** 2)

        r_dense_min_distance = 0.0
        if bool(self.config.enable_dense_min_distance) and math.isfinite(nearest_distance):
            safe_distance = max(1e-3, float(self.config.min_distance_safe))
            if nearest_distance < safe_distance:
                risk = float(np.clip((safe_distance - nearest_distance) / safe_distance, 0.0, 1.0))
                r_dense_min_distance = -abs(float(self.config.dense_min_distance_penalty_max)) * float(self.config.direct_dense_min_distance_weight) * (risk ** 2)

        r_dense_safety_direct = r_dense_ttc + r_dense_headway + r_dense_min_distance
        r_dense_safety_direct = max(float(self.config.dense_safety_penalty_cap), float(r_dense_safety_direct))

        defaults.update(
            {
                "r_dense_ttc": float(r_dense_ttc),
                "r_dense_headway": float(r_dense_headway),
                "r_dense_min_distance": float(r_dense_min_distance),
                "r_dense_safety_direct": float(r_dense_safety_direct),
                "min_ttc": float(min_ttc),
                "min_headway_distance": float(min_headway),
                "nearest_actor_type": nearest_actor_type,
                "nearest_actor_distance": float(nearest_distance),
                "nearest_actor_longitudinal_gap": float(nearest_longitudinal),
                "nearest_actor_lateral_gap": float(nearest_lateral),
                "nearest_actor_relative_speed": float(nearest_relative_speed),
            }
        )
        return defaults

    def compute_reward(
        self,
        vehicle: carla.Vehicle,
        route_waypoints: list,
        elapsed_time: float,
        ev_control: carla.VehicleControl = None,
        route_scenario=None,
        scenario_running: bool = True,
        route_complete: bool = False,
        route_progress_delta: float = 0.0,
        truncation_reason: Optional[str] = None,
    ) -> Tuple[float, bool, Dict]:
        """
        Compute reward for current step using carla-roach Valeo-style reward.

        Returns:
            reward: float
            done: bool
            info: dict with reward components and terminal diagnostics
        """
        del scenario_running
        info: Dict[str, object] = {}

        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        speed = float(np.linalg.norm([velocity.x, velocity.y]))
        line_e_enabled = self._line_e_enabled()
        failure_timeout = bool(line_e_enabled and self.is_failure_truncation_reason(truncation_reason))

        if ev_control is None:
            ev_control = vehicle.get_control()

        desired_spd_veh = self._get_hazard_vehicle_speed(vehicle)
        desired_spd_ped = self._get_hazard_pedestrian_speed(vehicle)
        desired_spd_rl, light_state = self._get_traffic_light_speed(vehicle)
        desired_spd_stop = self._get_stop_sign_speed(vehicle)
        desired_speed = min(
            self.config.max_speed,
            desired_spd_veh,
            desired_spd_ped,
            desired_spd_rl,
            desired_spd_stop,
        )

        r_speed = 0.1 * (1.0 - abs(speed - desired_speed) / self.config.max_speed)
        r_speed_pre_safe_wait = float(r_speed)
        r_speed_safe_wait_adjustment = 0.0

        world_map = vehicle.get_world().get_map()
        current_wp = world_map.get_waypoint(
            transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        nearest_wp = self._select_reference_waypoint(
            transform.location,
            route_waypoints,
            lookahead=self.config.route_membership_lookahead,
        )
        lateral_distance = 0.0
        route_distance = 0.0
        angle_diff = 0.0
        angle_diff_deg = 0.0
        wrong_world = False
        if nearest_wp is not None:
            wp_transform = nearest_wp.transform
            d_vec = transform.location - wp_transform.location
            np_d_vec = np.array([d_vec.x, d_vec.y], dtype=np.float32)
            wp_unit_forward = wp_transform.rotation.get_forward_vector()
            np_wp_unit_right = np.array([-wp_unit_forward.y, wp_unit_forward.x], dtype=np.float32)
            lateral_distance = float(abs(np.dot(np_wp_unit_right, np_d_vec)))
            route_distance = float(transform.location.distance(wp_transform.location))

            if lateral_distance > 1000.0:
                wrong_world = True

            angle_diff_deg = abs(self._normalize_angle(transform.rotation.yaw - wp_transform.rotation.yaw))
            angle_diff = math.radians(angle_diff_deg)

        route_acceptable = self._is_route_acceptable(current_wp, route_waypoints)
        if route_acceptable:
            route_membership_state = 'acceptable'
            r_position = 0.0
        elif route_distance <= float(self.config.recoverable_max_distance):
            route_membership_state = 'recoverable'
            r_position = float(self.config.route_recoverable_penalty)
        else:
            route_membership_state = 'bad'
            r_position = float(self.config.route_bad_penalty)

        if angle_diff_deg <= float(self.config.rotation_ok_deg):
            r_rotation = 0.0
        elif angle_diff_deg <= float(self.config.rotation_warn_deg):
            r_rotation = -0.02
        elif angle_diff_deg <= float(self.config.rotation_bad_deg):
            r_rotation = -0.10
        else:
            r_rotation = -0.30

        prev_steer = float(self._last_steer)
        prev_throttle = float(self._last_throttle)
        prev_brake = float(self._last_brake)
        r_action = -self.config.steer_change_penalty if abs(float(ev_control.steer) - prev_steer) > 0.01 else 0.0
        self._last_steer = float(ev_control.steer)

        c_lat_dist = bool((current_wp is None and route_distance > float(self.config.recoverable_max_distance))
                          or route_distance > float(self.config.bad_off_route_distance))
        self._last_lat_dist = lateral_distance

        self._speed_queue.append(speed)
        mean_speed = float(np.mean(self._speed_queue)) if self._speed_queue else speed
        legacy_is_free_road = (
            desired_spd_veh >= self.config.max_speed - 1e-6
            and desired_spd_ped >= self.config.max_speed - 1e-6
            and (light_state is None or light_state == carla.TrafficLightState.Green)
        )
        line_e_blocker_info = self._empty_line_e_blocker_info()
        legal_wait = False
        line_e_is_free_road = False
        if line_e_enabled:
            line_e_blocker_info = self._find_line_e_vehicle_blocker(vehicle)
            legal_wait = bool(
                desired_spd_veh < self.config.max_speed - 1e-6
                or desired_spd_ped < self.config.max_speed - 1e-6
                or desired_spd_rl < self.config.max_speed - 1e-6
                or desired_spd_stop < self.config.max_speed - 1e-6
                or (light_state is not None and light_state != carla.TrafficLightState.Green)
                or line_e_blocker_info["active"]
            )
            line_e_is_free_road = bool(
                legacy_is_free_road
                and route_acceptable
                and not c_lat_dist
                and not legal_wait
                and desired_speed >= self.config.max_speed - 1e-6
            )

        is_free_road = line_e_is_free_road if line_e_enabled else legacy_is_free_road
        if line_e_enabled:
            # A legal safety wait is not evidence that the policy is stuck on
            # free road.  Reset instead of preserving a nearly-terminal stale
            # counter across the wait.
            if is_free_road and mean_speed < self.config.stuck_speed_threshold:
                self._vehicle_stuck_counter += 1
            else:
                self._vehicle_stuck_counter = 0
        else:
            # Preserve the legacy Roach counter semantics exactly.
            if is_free_road and mean_speed < self.config.stuck_speed_threshold:
                self._vehicle_stuck_counter += 1
            if mean_speed >= self.config.stuck_speed_threshold:
                self._vehicle_stuck_counter = 0

        if line_e_enabled and legal_wait and speed < float(self.config.line_e_safe_wait_stop_speed):
            self._safe_wait_stopped_counter += 1
        else:
            self._safe_wait_stopped_counter = 0

        blocker_id = int(line_e_blocker_info["actor_id"])
        blocker_waiting_without_progress = bool(
            line_e_enabled
            and line_e_blocker_info["active"]
            and mean_speed < float(self.config.stuck_speed_threshold)
            and float(route_progress_delta) < float(self.config.no_progress_threshold)
        )
        if blocker_waiting_without_progress:
            if blocker_id == self._blocked_wait_actor_id:
                self._blocked_wait_counter += 1
            else:
                self._blocked_wait_actor_id = blocker_id
                self._blocked_wait_counter = 1
        else:
            self._blocked_wait_actor_id = -1
            self._blocked_wait_counter = 0
        blocked_timeout_steps = max(0, int(self.config.line_e_blocked_timeout_steps))
        blocked_timeout_active = bool(
            line_e_enabled
            and blocked_timeout_steps > 0
            and self._blocked_wait_counter >= blocked_timeout_steps
        )

        soft_stuck_active = self._vehicle_stuck_counter >= self.config.vehicle_stuck_step
        hard_stuck_active = self._vehicle_stuck_counter >= max(
            int(self.config.hard_stuck_step),
            int(self.config.vehicle_stuck_step) + 1,
        )

        r_stuck_soft = 0.0
        if soft_stuck_active and not hard_stuck_active:
            soft_steps = max(1, int(self.config.hard_stuck_step) - int(self.config.vehicle_stuck_step))
            soft_progress = float(self._vehicle_stuck_counter - int(self.config.vehicle_stuck_step)) / float(soft_steps)
            soft_progress = float(np.clip(soft_progress, 0.0, 1.0))
            capped_soft_penalty = min(
                float(self.config.soft_stuck_penalty_cap),
                float(self.config.soft_stuck_penalty_per_step) * float(soft_steps),
            )
            r_stuck_soft = -capped_soft_penalty * soft_progress

        c_vehicle_stuck = hard_stuck_active

        new_events = self._collect_new_events(route_scenario)
        event_names = []
        c_run_rl = False
        c_run_stop = False
        c_blocked = False
        for event in new_events:
            event_type = event.get_type()
            event_names.append(str(event_type).replace("TrafficEventType.", ""))
            if event_type == TrafficEventType.TRAFFIC_LIGHT_INFRACTION:
                c_run_rl = True
            elif event_type == TrafficEventType.STOP_INFRACTION:
                c_run_stop = True
            elif event_type == TrafficEventType.VEHICLE_BLOCKED:
                c_blocked = True

        c_collision = bool(self._collision_detected)
        self._collision_detected = False
        timeout = bool(self.config.eval_mode and elapsed_time > self.config.eval_time)
        timeout = bool(timeout or failure_timeout)

        done = bool(
            route_complete
            or wrong_world
            or hard_stuck_active
            or blocked_timeout_active
            or c_lat_dist
            or c_run_rl
            or c_collision
            or c_run_stop
            or c_blocked
            or timeout
        )

        r_terminal = 0.0
        r_stuck_terminal = 0.0
        if not route_complete:
            if c_collision:
                r_terminal = float(self.config.terminal_penalty_collision)
            elif c_lat_dist:
                r_terminal = float(self.config.terminal_penalty_off_route)
            elif c_run_rl:
                r_terminal = float(self.config.terminal_penalty_red_light) - speed
            elif c_run_stop:
                r_terminal = float(self.config.terminal_penalty_run_stop) - speed
            elif hard_stuck_active:
                r_stuck_terminal = float(self.config.hard_stuck_terminal_penalty)

        r_success = float(self.config.success_reward) if route_complete else 0.0
        reward = r_speed + r_position + r_rotation + r_action + r_stuck_soft + r_terminal + r_stuck_terminal + r_success
        r_original_base = float(reward)

        r_blocked = 0.0
        r_timeout = 0.0
        r_hard_stuck = 0.0
        r_free_road_efficiency = 0.0
        r_low_speed = 0.0
        r_no_progress = 0.0
        r_comfort = 0.0
        dense_safety_info = self._compute_direct_dense_safety(vehicle, speed, False, True)
        terminal_for_replay = False
        timeout_kind = str(truncation_reason or "")

        if line_e_enabled:
            terminal_for_replay = bool(failure_timeout)

            # Once a legal wait has settled, neither reward nor punish merely
            # matching a near-zero target speed.  The short grace preserves the
            # original deceleration feedback, while preventing stationary
            # reward accumulation during long red lights or front blockers.
            if (
                self._safe_wait_stopped_counter > int(self.config.line_e_safe_wait_reward_grace_steps)
                and speed < float(self.config.line_e_safe_wait_stop_speed)
            ):
                r_speed_safe_wait_adjustment = -float(r_speed)
                reward += r_speed_safe_wait_adjustment
                r_speed = 0.0

            # Keep existing high-priority terminal penalties exclusive.
            has_high_priority_terminal = bool(c_collision or c_lat_dist or c_run_rl or c_run_stop or route_complete)
            if not has_high_priority_terminal:
                if c_blocked:
                    r_blocked = float(self.config.terminal_penalty_blocked)
                elif timeout:
                    r_timeout = float(self.config.terminal_penalty_timeout)
                elif hard_stuck_active:
                    r_hard_stuck = float(self.config.line_e_hard_stuck_terminal_penalty) - float(r_stuck_terminal)

            if line_e_is_free_road:
                r_free_road_efficiency = float(self.config.free_road_efficiency_scale) * float(
                    np.clip(speed / max(1e-6, float(self.config.max_speed)), 0.0, 1.0)
                )

                if mean_speed < float(self.config.low_speed_threshold):
                    self._low_speed_counter += 1
                else:
                    self._low_speed_counter = 0
                if float(route_progress_delta) < float(self.config.no_progress_threshold):
                    self._no_progress_counter += 1
                else:
                    self._no_progress_counter = 0
            else:
                self._low_speed_counter = 0
                self._no_progress_counter = 0

            if self._low_speed_counter > int(self.config.low_speed_grace_steps):
                remaining = float(self.config.low_speed_episode_cap) - self._low_speed_penalty_total
                if remaining < 0.0:
                    r_low_speed = max(float(self.config.low_speed_penalty_per_step), remaining)
                    self._low_speed_penalty_total += r_low_speed
            if self._no_progress_counter > int(self.config.no_progress_grace_steps):
                remaining = float(self.config.no_progress_episode_cap) - self._no_progress_penalty_total
                if remaining < 0.0:
                    r_no_progress = max(float(self.config.no_progress_penalty_per_step), remaining)
                    self._no_progress_penalty_total += r_no_progress

            if bool(self.config.enable_comfort_penalty):
                control_change = (
                    abs(float(ev_control.steer) - prev_steer)
                    + 0.5 * abs(float(ev_control.throttle) - prev_throttle)
                    + 0.5 * abs(float(ev_control.brake) - prev_brake)
                )
                r_comfort = -min(abs(float(self.config.comfort_penalty_max)), 0.02 * control_change)

            dense_safety_info = self._compute_direct_dense_safety(vehicle, speed, line_e_is_free_road, legal_wait)
            reward += (
                r_blocked
                + r_timeout
                + r_hard_stuck
                + r_free_road_efficiency
                + r_low_speed
                + r_no_progress
                + float(dense_safety_info.get("r_dense_safety_direct", 0.0))
                + r_comfort
            )

        self._last_throttle = float(ev_control.throttle)
        self._last_brake = float(ev_control.brake)

        termination_reasons = []
        if c_collision:
            termination_reasons.append("collision")
        if c_run_rl:
            termination_reasons.append("red_light")
        if c_run_stop:
            termination_reasons.append("run_stop_sign")
        if c_blocked:
            termination_reasons.append("vehicle_blocked")
        if c_lat_dist:
            termination_reasons.append("off_route")
        if hard_stuck_active:
            termination_reasons.append("vehicle_stuck")
        if blocked_timeout_active:
            termination_reasons.append("blocked_timeout")
        if timeout:
            termination_reasons.append("timeout")
        if wrong_world:
            termination_reasons.append("wrong_world")
        if route_complete:
            termination_reasons.append("route_complete")

        info.update(
            {
                "r_speed": float(r_speed),
                "r_position": float(r_position),
                "r_rotation": float(r_rotation),
                "r_action": float(r_action),
                "r_terminal": float(r_terminal),
                "r_progress": 0.0,
                "r_event": 0.0,
                "r_corridor": 0.0,
                "r_success": float(r_success),
                "r_stuck_soft": float(r_stuck_soft),
                "r_stuck_terminal": float(r_stuck_terminal),
                "r_original_base": float(r_original_base),
                "r_speed_pre_safe_wait": float(r_speed_pre_safe_wait),
                "r_line_e_safe_wait_speed_adjustment": float(r_speed_safe_wait_adjustment),
                "r_blocked": float(r_blocked),
                "r_timeout": float(r_timeout),
                "r_hard_stuck": float(r_hard_stuck),
                "r_free_road_efficiency": float(r_free_road_efficiency),
                "r_low_speed": float(r_low_speed),
                "r_no_progress": float(r_no_progress),
                "r_dense_ttc": float(dense_safety_info.get("r_dense_ttc", 0.0)),
                "r_dense_headway": float(dense_safety_info.get("r_dense_headway", 0.0)),
                "r_dense_min_distance": float(dense_safety_info.get("r_dense_min_distance", 0.0)),
                "r_dense_safety_direct": float(dense_safety_info.get("r_dense_safety_direct", 0.0)),
                "r_comfort": float(r_comfort),
                "speed": float(speed),
                "desired_speed": float(desired_speed),
                "desired_speed_vehicle": float(desired_spd_veh),
                "desired_speed_pedestrian": float(desired_spd_ped),
                "desired_speed_traffic_light": float(desired_spd_rl),
                "desired_speed_stop_sign": float(desired_spd_stop),
                "lateral_distance": float(lateral_distance),
                "route_distance": float(route_distance),
                "angle_diff_rad": float(angle_diff),
                "angle_diff_deg": float(angle_diff_deg),
                "route_corridor_threshold": float(self.config.recoverable_max_distance),
                "route_corridor_excess": float(max(0.0, route_distance - float(self.config.recoverable_max_distance))),
                "route_membership_state": route_membership_state,
                "route_acceptable": bool(route_acceptable),
                "mean_speed": float(mean_speed),
                "vehicle_stuck_counter": int(self._vehicle_stuck_counter),
                "soft_stuck_active": bool(soft_stuck_active),
                "hard_stuck_active": bool(hard_stuck_active),
                "is_blocked": bool(c_blocked),
                "is_blocked_timeout": bool(blocked_timeout_active),
                "is_timeout": bool(timeout),
                "timeout_kind": timeout_kind,
                "is_hard_stuck": bool(hard_stuck_active),
                "is_free_road": bool(is_free_road),
                "legacy_is_free_road": bool(legacy_is_free_road),
                "is_legal_wait": bool(legal_wait),
                "legal_wait": bool(legal_wait),
                "safe_wait_active": bool(line_e_enabled and legal_wait),
                "bbox_safety_wait_active": bool(line_e_blocker_info["active"]),
                "safe_wait_stopped_counter": int(self._safe_wait_stopped_counter),
                "blocked_wait_counter": int(self._blocked_wait_counter),
                "blocked_wait_actor_id": int(self._blocked_wait_actor_id),
                "line_e_enabled": bool(line_e_enabled),
                "enable_direct_dense_safety": bool(self.config.enable_direct_dense_safety),
                "low_speed_counter": int(self._low_speed_counter),
                "no_progress_counter": int(self._no_progress_counter),
                "route_progress_delta": float(route_progress_delta),
                "terminal_for_replay": bool(terminal_for_replay),
                "min_ttc": float(dense_safety_info.get("min_ttc", float("inf"))),
                "min_headway_distance": float(dense_safety_info.get("min_headway_distance", float("inf"))),
                "nearest_actor_type": str(dense_safety_info.get("nearest_actor_type", "")),
                "nearest_actor_distance": float(dense_safety_info.get("nearest_actor_distance", float("inf"))),
                "nearest_actor_longitudinal_gap": float(
                    dense_safety_info.get("nearest_actor_longitudinal_gap", float("inf"))
                ),
                "nearest_actor_lateral_gap": float(dense_safety_info.get("nearest_actor_lateral_gap", float("inf"))),
                "nearest_actor_relative_speed": float(dense_safety_info.get("nearest_actor_relative_speed", 0.0)),
                "safety_wait_blocker_id": int(line_e_blocker_info["actor_id"]),
                "safety_wait_blocker_type": str(line_e_blocker_info["actor_type"]),
                "safety_wait_blocker_speed": float(line_e_blocker_info["actor_speed"]),
                "safety_wait_blocker_center_distance": float(line_e_blocker_info["center_distance"]),
                "safety_wait_blocker_longitudinal_center_distance": float(
                    line_e_blocker_info["longitudinal_center_distance"]
                ),
                "safety_wait_blocker_lateral_center_offset": float(
                    line_e_blocker_info["lateral_center_offset"]
                ),
                "safety_wait_blocker_longitudinal_clearance": float(
                    line_e_blocker_info["longitudinal_clearance"]
                ),
                "safety_wait_blocker_lateral_clearance": float(line_e_blocker_info["lateral_clearance"]),
                "safety_wait_blocker_lateral_overlap": float(line_e_blocker_info["lateral_overlap"]),
                "safety_wait_blocker_heading_dot": float(line_e_blocker_info["heading_dot"]),
                "light_state": str(light_state).split(".")[-1] if light_state is not None else "None",
                "event_names": event_names,
                "route_complete": bool(route_complete),
                "termination_reasons": termination_reasons,
                "termination": termination_reasons[0] if termination_reasons else None,
                "off_route_candidate": bool(c_lat_dist),
            }
        )

        return float(reward), done, info

    @staticmethod
    def _normalize_angle(angle):
        """Normalize angle to [-180, 180]."""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def on_collision(self, event: carla.CollisionEvent):
        """Collision callback."""
        self._collision_detected = True

    def on_lane_invasion(self, event: carla.LaneInvasionEvent):
        """Lane invasion callback (kept for sensor compatibility)."""
        self._lane_invasion_detected = True
