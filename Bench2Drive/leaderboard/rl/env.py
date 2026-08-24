"""
Bench2Drive RL Environment
==========================

Gym-like environment for Bench2Drive SAC training.
Aligned with leaderboard_eval scene loading approach.
"""

import os
import random
import sys
import time
import copy
import math
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import carla
import py_trees
from scipy.optimize import fsolve

# Add the CARLA PythonAPI path for the navigation agents module. Launchers are
# responsible for requiring CARLA_ROOT before this runtime is imported.
_CARLA_ROOT = os.environ.get('CARLA_ROOT', '')
if _CARLA_ROOT:
    sys.path.insert(0, os.path.join(_CARLA_ROOT, 'PythonAPI/carla'))

# Import srunner for CarlaDataProvider
if _CARLA_ROOT:
    sys.path.insert(0, os.path.join(_CARLA_ROOT, 'PythonAPI'))
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

from rl.obs_builder import ObservationBuilder, ObservationConfig
from rl.reward import RewardCalculator, RewardConfig
from rl.roach_bev_target import (
    RoachActorBox,
    RoachBevTargetConfig,
    RoachBevTargetGenerator,
    actor_box_from_carla_actor,
    actor_box_from_level_bbox,
    render_roach_bev_target,
)
from rl.roach_bev_target_cache import FrameKeyedRoachBevTargetCache, RoachBevTransientTarget
from rl.step_manager import StepManager, StepManagerConfig

# Import leaderboard's SimulationBackend (same as leaderboard_eval)
from rl.sim_backend import SimulationBackend, SimulationConfig, build_simulation_config_from_args
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.utils.route_parser import RouteParser

# Import for route interpolation (same as leaderboard)
try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    from agents.navigation.local_planner import RoadOption
except ImportError:
    print("[Bench2DriveSACEnv] Warning: Could not import agents module, route interpolation disabled")
    GlobalRoutePlanner = None
    RoadOption = None

from rl.navigation_route_planner import RoutePlanner as NavigationRoutePlanner
try:
    from srunner.tools.route_manipulation import location_route_to_gps
except ImportError:
    location_route_to_gps = None


@dataclass
class RLEnvConfig:
    """Configuration for RL environment."""
    routes: str
    simulation: SimulationConfig
    observation: ObservationConfig
    reward: RewardConfig
    step_manager: StepManagerConfig
    max_episode_steps: int = 4000
    random_routes: bool = True
    fixed_route_idx: int = -1
    fixed_route_name: str = ""
    sensor_packet_timeout: float = 30.0
    sensor_packet_log_interval: float = 5.0
    sensor_packet_grace_seconds: float = 0.5
    sensor_packet_max_lag_frames: int = 1
    roach_bev_target_enabled: bool = False
    roach_bev_map_root: str = ""
    roach_bev_target_cache_size: int = 8
    roach_bev_target_debug_dir: str = ""
    roach_bev_target_debug_interval: int = 0
    roach_bev_target_debug_max_frames: int = 100


class Bench2DriveSACEnv:
    """
    Bench2Drive Environment for SAC training.

    Follows OpenAI Gym interface:
    - reset() -> observation, info
    - step(action) -> observation, reward, terminated, truncated, info
    - close()

    Scene loading aligned with leaderboard_eval approach:
    - Uses SimulationBackend for world management
    - Uses CarlaDataProvider for actor tracking
    - Loads world per route with proper sync settings
    """

    def __init__(self, config: RLEnvConfig):
        self.config = config

        # Initialize simulation using SimulationBackend (same as leaderboard_eval)
        self._sim_backend = SimulationBackend(config.simulation)
        self.client, self.client_timeout, self.traffic_manager = self._sim_backend.start()
        self.world = self._sim_backend.world

        # Initialize observation builder
        self.obs_builder = ObservationBuilder(config.observation)

        # Initialize reward calculator
        self.reward_calculator = RewardCalculator(config.reward)

        # Initialize step manager
        self.step_manager = StepManager(config.step_manager)

        # Load routes with full config (including town info)
        self.route_configs = self._load_route_configs(config.routes)
        self.route_scenario_configs = {
            route_config.name: route_config
            for route_config in RouteParser.parse_routes_file(config.routes)
        }
        self.current_route_idx = 0

        # Episode state
        self.vehicle: Optional[carla.Vehicle] = None
        self.route_scenario: Optional[RouteScenario] = None
        self.route_waypoints: List[carla.Waypoint] = []
        self.current_waypoint_idx = 0
        self.route_progress = 0.0
        self.prev_route_progress = 0.0
        self.episode_step = 0
        self._last_scenario_build_time = 0.0
        self._scenario_build_interval = 1.0

        # Generic observation navigation. Clean HiP-AD policy navigation is
        # bound separately by the clean runtime.
        self._route_planner: Optional[NavigationRoutePlanner] = None
        self.lat_ref: float = 42.0
        self.lon_ref: float = 2.0

        # Route switching optimization: stay on same route for N episodes
        self.route_switch_interval = 10  # Switch route every 10 episodes for better learning stability
        self.episodes_on_current_route = 0
        self._current_route_config = None
        self._current_interpolated_route = None
        self._scene_instance_id = 0
        self._current_scene_token = None
        self._reset_warmup_observations = []
        self._recent_positions = deque(maxlen=256)
        self._recent_progress = deque(maxlen=256)
        self._recent_controls = deque(maxlen=256)
        self._roach_bev_target_generator: Optional[RoachBevTargetGenerator] = None
        self._roach_bev_target_cache = FrameKeyedRoachBevTargetCache(config.roach_bev_target_cache_size)
        self._roach_tl_cache_world_id = None
        self._roach_tl_entries: List[Dict[str, object]] = []
        self._roach_bev_debug_count = 0
        self._init_roach_bev_target_generator()

    def _init_roach_bev_target_generator(self) -> None:
        if not bool(getattr(self.config, 'roach_bev_target_enabled', False)):
            return
        asset_root = str(getattr(self.config, 'roach_bev_map_root', '') or '').strip()
        target_config = RoachBevTargetConfig()
        self._roach_bev_target_generator = RoachBevTargetGenerator(
            asset_root=Path(asset_root) if asset_root else None,
            config=target_config,
        )

    def pop_roach_bev_target(self, expected_frame: int) -> Optional[Dict[str, object]]:
        """One-shot read for current-frame semantic supervision."""
        if self._roach_bev_target_generator is None:
            return None
        return self._roach_bev_target_cache.pop(int(expected_frame))

    def _reset_roach_bev_target_state(self) -> None:
        self._roach_bev_target_cache.clear()
        self._roach_tl_cache_world_id = None
        self._roach_tl_entries.clear()
        if self._roach_bev_target_generator is not None:
            self._roach_bev_target_generator.reset()

    def _roach_world_town_name(self) -> str:
        if self.world is None:
            return ""
        return self.world.get_map().name.split("/")[-1]

    @staticmethod
    def _roach_get_traffic_light_waypoints(traffic_light, carla_map):
        base_transform = traffic_light.get_transform()
        tv_loc = traffic_light.trigger_volume.location
        tv_ext = traffic_light.trigger_volume.extent

        x_values = np.arange(-0.9 * tv_ext.x, 0.9 * tv_ext.x, 1.0)
        area = [base_transform.transform(tv_loc + carla.Location(x=float(x))) for x in x_values]

        ini_wps = []
        for point in area:
            waypoint = carla_map.get_waypoint(point)
            if waypoint is None:
                continue
            if not ini_wps or ini_wps[-1].road_id != waypoint.road_id or ini_wps[-1].lane_id != waypoint.lane_id:
                ini_wps.append(waypoint)

        stopline_wps = []
        stopline_vertices = []
        for waypoint in ini_wps:
            current = waypoint
            while not current.is_intersection:
                next_wps = current.next(0.5)
                if next_wps and not next_wps[0].is_intersection:
                    current = next_wps[0]
                else:
                    break
            stopline_wps.append(current)
            forward = current.transform.get_forward_vector()
            right = carla.Vector3D(x=-forward.y, y=forward.x, z=0.0)
            loc_left = current.transform.location - 0.4 * current.lane_width * right
            loc_right = current.transform.location + 0.4 * current.lane_width * right
            stopline_vertices.append((loc_left, loc_right))
        tv_world_loc = base_transform.transform(tv_loc)
        return tv_world_loc, stopline_wps, stopline_vertices

    def _refresh_roach_traffic_light_cache(self) -> None:
        if self.world is None:
            self._roach_tl_entries.clear()
            self._roach_tl_cache_world_id = None
            return
        world_id = id(self.world)
        if self._roach_tl_cache_world_id == world_id and self._roach_tl_entries:
            return
        self._roach_tl_entries = []
        carla_map = self.world.get_map()
        for actor in self.world.get_actors():
            if 'traffic_light' not in actor.type_id:
                continue
            try:
                tv_loc, stopline_wps, stopline_vtx = self._roach_get_traffic_light_waypoints(actor, carla_map)
            except Exception:
                continue
            if stopline_wps and stopline_vtx:
                self._roach_tl_entries.append(
                    {
                        "actor": actor,
                        "tv_loc": tv_loc,
                        "stopline_wps": stopline_wps,
                        "stopline_vtx": tuple(stopline_vtx),
                    }
                )
        self._roach_tl_cache_world_id = world_id

    def _roach_stopline_segments_by_state(self, ego_loc, dist_threshold: float = 50.0) -> Dict[str, Tuple]:
        self._refresh_roach_traffic_light_cache()
        result: Dict[str, List[Tuple[object, object]]] = {"green": [], "yellow": [], "red": []}
        state_to_key = {
            carla.TrafficLightState.Green: "green",
            carla.TrafficLightState.Yellow: "yellow",
            carla.TrafficLightState.Red: "red",
        }
        for entry in self._roach_tl_entries:
            actor = entry["actor"]
            tv_loc = entry["tv_loc"]
            if tv_loc.distance(ego_loc) > float(dist_threshold):
                continue
            key = state_to_key.get(actor.state)
            if key is None:
                continue
            result[key].extend(entry["stopline_vtx"])
        return {key: tuple(value) for key, value in result.items()}

    @staticmethod
    def _roach_city_object_labels(*names: str) -> Tuple[object, ...]:
        labels: List[object] = []
        seen = set()
        for name in names:
            if not hasattr(carla.CityObjectLabel, name):
                continue
            label = getattr(carla.CityObjectLabel, name)
            key = int(label) if isinstance(label, int) else repr(label)
            if key in seen:
                continue
            seen.add(key)
            labels.append(label)
        return tuple(labels)

    def _roach_level_bbox_boxes(self, labels, ego_loc) -> Tuple[RoachActorBox, ...]:
        if self._roach_bev_target_generator is None or self.world is None:
            return ()
        cfg = self._roach_bev_target_generator.config
        distance_threshold = float(np.ceil(float(cfg.width_in_pixels) / float(cfg.pixels_per_meter)))

        def is_within_distance(bbox) -> bool:
            loc = getattr(bbox, "location", None)
            if loc is None:
                return False
            near = (
                abs(float(ego_loc.x) - float(loc.x)) < distance_threshold
                and abs(float(ego_loc.y) - float(loc.y)) < distance_threshold
                and abs(float(ego_loc.z) - float(loc.z)) < 8.0
            )
            is_ego = abs(float(ego_loc.x) - float(loc.x)) < 1.0 and abs(float(ego_loc.y) - float(loc.y)) < 1.0
            return near and not is_ego

        if not isinstance(labels, (list, tuple)):
            labels = (labels,)
        result: List[RoachActorBox] = []
        for label in labels:
            try:
                result.extend(actor_box_from_level_bbox(bbox) for bbox in self.world.get_level_bbs(label)
                              if is_within_distance(bbox))
            except Exception:
                continue
        return tuple(result)

    def _roach_actor_boxes_by_filter(self, actor_filter: str) -> Tuple[RoachActorBox, ...]:
        if self.world is None or self.vehicle is None:
            return ()
        boxes: List[RoachActorBox] = []
        ego_id = int(getattr(self.vehicle, "id", -1))
        ego_loc = self.vehicle.get_location()
        distance_threshold = 45.0
        for actor in self.world.get_actors().filter(actor_filter):
            if int(getattr(actor, "id", -2)) == ego_id:
                continue
            try:
                if actor.get_location().distance(ego_loc) > distance_threshold:
                    continue
                boxes.append(actor_box_from_carla_actor(actor))
            except Exception:
                continue
        return tuple(boxes)

    def _roach_stop_sign_boxes(self, ego_loc, dist_threshold: float = 50.0) -> Tuple[RoachActorBox, ...]:
        if self.world is None:
            return ()
        boxes: List[RoachActorBox] = []
        try:
            stop_signs = self.world.get_actors().filter("traffic.stop")
        except Exception:
            return ()
        for stop_sign in stop_signs:
            try:
                if stop_sign.get_location().distance(ego_loc) > float(dist_threshold):
                    continue
                bb_loc = carla.Location(stop_sign.trigger_volume.location)
                bb_ext = carla.Vector3D(stop_sign.trigger_volume.extent)
                max_extent = max(float(bb_ext.x), float(bb_ext.y))
                bb_ext.x = max_extent
                bb_ext.y = max_extent
                boxes.append(
                    RoachActorBox(
                        transform=stop_sign.get_transform(),
                        bbox_location=bb_loc,
                        bbox_extent=bb_ext,
                    )
                )
            except Exception:
                continue
        return tuple(boxes)

    def _maybe_export_roach_bev_target_debug(self, frame: int, payload: Mapping[str, object]) -> None:
        debug_dir = str(getattr(self.config, 'roach_bev_target_debug_dir', '') or '').strip()
        interval = int(getattr(self.config, 'roach_bev_target_debug_interval', 0) or 0)
        max_frames = int(getattr(self.config, 'roach_bev_target_debug_max_frames', 100) or 100)
        if not debug_dir or interval <= 0:
            return
        if self._roach_bev_debug_count >= max_frames:
            return
        if int(frame) % interval != 0:
            return
        out_dir = Path(debug_dir)
        if not out_dir.is_absolute():
            out_dir = Path.cwd() / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        if payload.get("error"):
            import json

            error_payload = {
                "frame": int(frame),
                "town_name": str(payload.get("town_name", "")),
                "sensor_frame_exact": bool(payload.get("sensor_frame_exact", False)),
                "error": str(payload.get("error", "")),
                "valid": bool(payload.get("valid", False)),
            }
            with (out_dir / "roach_bev_target_errors.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(error_payload, sort_keys=True) + "\n")
            self._roach_bev_debug_count += 1
            return
        masks = payload.get("masks")
        if masks is None:
            return
        rendered = render_roach_bev_target(np.asarray(masks))
        import cv2
        cv2.imwrite(str(out_dir / f"roach_bev_target_frame_{int(frame):08d}.png"), rendered[:, :, ::-1])
        self._roach_bev_debug_count += 1

    def _maybe_cache_roach_bev_target(self, observation: Mapping[str, np.ndarray], info: Mapping[str, object]) -> None:
        generator = self._roach_bev_target_generator
        if generator is None:
            return
        try:
            frame = int(observation.get('sensor_frame', info.get('sensor_frame_used', -1)))
        except (TypeError, ValueError):
            frame = -1
        exact = bool(info.get('sensor_frame_exact', True))
        channel_names = generator.channel_names
        if not exact:
            self._roach_bev_target_cache.put(
                RoachBevTransientTarget(
                    frame=frame,
                    masks=None,
                    channel_names=channel_names,
                    sensor_frame_exact=False,
                    town_name=self._roach_world_town_name(),
                    error="sensor_not_exact",
                )
            )
            return
        try:
            if self.vehicle is None or self.world is None:
                raise RuntimeError("vehicle/world is not ready")
            ego_transform = self.vehicle.get_transform()
            ego_loc = ego_transform.location
            vehicle_labels = self._roach_city_object_labels(
                "Vehicles", "Vehicle", "Car", "Truck", "Bus", "Motorcycle", "Bicycle", "Train",
            )
            vehicle_boxes = self._roach_level_bbox_boxes(vehicle_labels, ego_loc)
            if not vehicle_boxes:
                vehicle_boxes = self._roach_actor_boxes_by_filter("vehicle*")
            walker_labels = self._roach_city_object_labels("Pedestrians", "Pedestrian", "Walkers", "Walker")
            walker_boxes = self._roach_level_bbox_boxes(walker_labels, ego_loc)
            if not walker_boxes:
                walker_boxes = self._roach_actor_boxes_by_filter("walker.pedestrian*")
            target = generator.build(
                town_name=str(info.get('town', self._roach_world_town_name()) or self._roach_world_town_name()),
                ego_transform=ego_transform,
                route_waypoints=self.route_waypoints[self.current_waypoint_idx:],
                vehicle_boxes=vehicle_boxes,
                walker_boxes=walker_boxes,
                traffic_light_stopline_segments=self._roach_stopline_segments_by_state(ego_loc),
                stop_boxes=self._roach_stop_sign_boxes(ego_loc),
            )
            masks = np.asarray(target["masks"], dtype=np.uint8)
            transient = RoachBevTransientTarget(
                frame=frame,
                masks=masks,
                channel_names=tuple(target["channel_names"]),
                sensor_frame_exact=True,
                town_name=str(target["town_name"]),
                error="",
                pixels_per_meter=float(target["pixels_per_meter"]),
                pixels_ev_to_bottom=int(generator.config.pixels_ev_to_bottom),
                width_in_pixels=int(target["width_in_pixels"]),
            )
            self._roach_bev_target_cache.put(transient)
            self._maybe_export_roach_bev_target_debug(frame, transient.as_dict())
        except Exception as exc:
            transient = RoachBevTransientTarget(
                frame=frame,
                masks=None,
                channel_names=channel_names,
                sensor_frame_exact=exact,
                town_name=self._roach_world_town_name(),
                error=f"target_generation_failed:{type(exc).__name__}:{exc}",
            )
            self._roach_bev_target_cache.put(transient)
            self._maybe_export_roach_bev_target_debug(frame, transient.as_dict())

    def _disable_autopilot(self, vehicle: Optional[carla.Vehicle]) -> None:
        """Disable autopilot against the active TrafficManager port explicitly."""
        if vehicle is None:
            return
        tm_port = int(getattr(self.config.simulation, 'traffic_manager_port', 0) or CarlaDataProvider.get_traffic_manager_port())
        vehicle.set_autopilot(False, tm_port)

    def _wait_for_sensor_frame(self, frame: int, timeout_seconds: Optional[float] = None) -> int:
        """Wait for an exact packet, then fall back to the latest complete frame within a small lag budget."""
        if timeout_seconds is None:
            timeout_seconds = float(self.config.sensor_packet_timeout)
        log_interval = max(0.5, float(self.config.sensor_packet_log_interval))
        grace_seconds = max(0.0, float(self.config.sensor_packet_grace_seconds))
        max_lag_frames = max(0, int(self.config.sensor_packet_max_lag_frames))
        deadline = time.time() + timeout_seconds
        next_log_time = time.time() + log_interval
        fallback_allowed_at = time.time() + grace_seconds
        while time.time() < deadline:
            if self.obs_builder.all_sensors_ready(frame):
                return int(frame)

            now = time.time()
            if max_lag_frames > 0 and now >= fallback_allowed_at:
                fallback_frame = self.obs_builder.get_latest_complete_frame(max_frame=frame)
                if fallback_frame is not None and (frame - fallback_frame) <= max_lag_frames:
                    if fallback_frame != frame:
                        print(
                            f"[Bench2DriveSACEnv] Using fallback sensor frame {fallback_frame} "
                            f"for requested frame {frame}"
                        )
                    return int(fallback_frame)
            if now >= next_log_time:
                debug = self.obs_builder.get_sensor_packet_debug_info(frame)
                print(
                    f"[Bench2DriveSACEnv] Waiting for sensor frame {frame}: "
                    f"missing={debug['missing_sensors']} latest_frames={debug['latest_sensor_frames']}"
                )
                next_log_time = now + log_interval
            time.sleep(0.002)
        debug = self.obs_builder.get_sensor_packet_debug_info(frame)
        if max_lag_frames > 0:
            fallback_frame = self.obs_builder.get_latest_complete_frame(max_frame=frame)
            if fallback_frame is not None and (frame - fallback_frame) <= max_lag_frames:
                print(
                    f"[Bench2DriveSACEnv] Timed out on frame {frame}, falling back to latest complete "
                    f"frame {fallback_frame}"
                )
                return int(fallback_frame)
        raise RuntimeError(
            f"Timed out waiting for complete sensor packet for frame {frame}; "
            f"missing={debug['missing_sensors']} latest_frames={debug['latest_sensor_frames']} "
            f"complete_frames_tail={debug['complete_frames_tail']}"
        )

    def _reset_stuck_state(self) -> None:
        self._recent_positions.clear()
        self._recent_progress.clear()
        self._recent_controls.clear()

    def _record_motion_state(self, control: carla.VehicleControl, route_progress: float) -> None:
        if self.vehicle is None:
            return
        location = self.vehicle.get_transform().location
        self._recent_positions.append((float(location.x), float(location.y)))
        self._recent_progress.append(float(route_progress))
        self._recent_controls.append(
            (
                float(control.throttle),
                float(control.brake),
                float(control.steer),
            )
        )

    def _has_adjacent_driving_lane(self) -> bool:
        if self.vehicle is None or self.world is None:
            return False
        current_wp = self.world.get_map().get_waypoint(self.vehicle.get_transform().location)
        if current_wp is None:
            return False

        current_sign = 1 if current_wp.lane_id >= 0 else -1
        for neighbor in (current_wp.get_left_lane(), current_wp.get_right_lane()):
            if neighbor is None:
                continue
            if neighbor.lane_type != carla.LaneType.Driving:
                continue
            if neighbor.road_id != current_wp.road_id:
                continue
            neighbor_sign = 1 if neighbor.lane_id >= 0 else -1
            if neighbor_sign != current_sign:
                continue
            return True
        return False

    @staticmethod
    def _normalize_angle_deg(angle: float) -> float:
        while angle > 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return angle

    def _is_recoverable_route_deviation(self, reward_info: Dict) -> bool:
        """Allow temporary adjacent-lane deviation when it is still recoverable."""
        if not reward_info.get('off_route_candidate', False):
            return False
        if self.vehicle is None or self.world is None or not self.route_waypoints:
            return False
        if not self._has_adjacent_driving_lane():
            return False

        current_wp = self.world.get_map().get_waypoint(
            self.vehicle.get_transform().location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if current_wp is None:
            return False

        route_idx = min(self.current_waypoint_idx, len(self.route_waypoints) - 1)
        route_wp = self.route_waypoints[route_idx]
        heading_diff = abs(
            self._normalize_angle_deg(
                current_wp.transform.rotation.yaw - route_wp.transform.rotation.yaw
            )
        )
        return heading_diff < 45.0

    def _compute_stuck_status(self, control: carla.VehicleControl, reward_info: Dict, route_progress: float) -> Dict:
        """Classify legal waiting vs soft/hard stuck in env space."""
        desired_speed = float(reward_info.get('desired_speed', self.config.reward.max_speed))
        legal_wait = bool(reward_info.get('legal_wait', desired_speed <= 0.5))
        adjacent_lane_exists = self._has_adjacent_driving_lane()
        tick_interval = float(self.config.step_manager.tick_interval)
        soft_window_steps = max(1, int(round(self.config.reward.soft_stuck_time / tick_interval)))
        hard_seconds = (
            self.config.reward.hard_stuck_time_with_adjacent_lane
            if adjacent_lane_exists
            else self.config.reward.hard_stuck_time_no_adjacent_lane
        )
        hard_window_steps = max(soft_window_steps, int(round(hard_seconds / tick_interval)))

        recent_displacement = 0.0
        recent_progress_delta = 0.0
        if len(self._recent_positions) >= soft_window_steps:
            start_x, start_y = self._recent_positions[-soft_window_steps]
            end_x, end_y = self._recent_positions[-1]
            recent_displacement = float(np.hypot(end_x - start_x, end_y - start_y))
            recent_progress_delta = float(self._recent_progress[-1] - self._recent_progress[-soft_window_steps])
        elif len(self._recent_positions) >= 2:
            start_x, start_y = self._recent_positions[0]
            end_x, end_y = self._recent_positions[-1]
            recent_displacement = float(np.hypot(end_x - start_x, end_y - start_y))
            recent_progress_delta = float(self._recent_progress[-1] - self._recent_progress[0])

        low_motion = (
            recent_displacement < self.config.reward.stuck_min_displacement_m and
            recent_progress_delta < self.config.reward.stuck_min_progress_delta
        )

        soft_stuck = (
            not legal_wait and
            len(self._recent_positions) >= soft_window_steps and
            low_motion and
            float(reward_info.get('speed', 0.0)) < 0.5
        )

        control_effort = control.throttle > 0.2 and control.brake < 0.1
        hard_stuck = (
            soft_stuck and
            len(self._recent_positions) >= hard_window_steps and
            (control_effort or not adjacent_lane_exists)
        )

        soft_penalty = 0.0
        if soft_stuck and not hard_stuck:
            soft_duration_steps = max(0, len(self._recent_positions) - soft_window_steps + 1)
            soft_penalty = -min(
                self.config.reward.soft_stuck_penalty_cap,
                self.config.reward.soft_stuck_penalty_per_step * soft_duration_steps,
            )

        return {
            'legal_wait': legal_wait,
            'adjacent_lane_exists': adjacent_lane_exists,
            'recent_displacement': recent_displacement,
            'recent_progress_delta': recent_progress_delta,
            'soft_stuck': soft_stuck and not hard_stuck,
            'hard_stuck': hard_stuck,
            'soft_stuck_penalty': soft_penalty,
        }

    def _update_geo_reference(self):
        """Parse map geo-reference like the leaderboard agent."""
        xodr = self.world.get_map().to_opendrive()
        tree = ET.ElementTree(ET.fromstring(xodr))

        lat_ref = 42.0
        lon_ref = 2.0
        for opendrive in tree.iter("OpenDRIVE"):
            for header in opendrive.iter("header"):
                for georef in header.iter("geoReference"):
                    if georef.text:
                        for item in georef.text.split(' '):
                            if '+lat_0' in item:
                                lat_ref = float(item.split('=')[1])
                            if '+lon_0' in item:
                                lon_ref = float(item.split('=')[1])

        self.lat_ref = lat_ref
        self.lon_ref = lon_ref

    def _gps_to_location(self, gps: np.ndarray) -> np.ndarray:
        """Convert GNSS readings to leaderboard-style planar coordinates."""
        EARTH_RADIUS_EQUA = 6378137.0
        lat, lon = gps[:2]
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat + 90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y], dtype=np.float32)

    def _update_geo_reference_from_route(self, gps_route, world_route) -> None:
        """Match vad_b2d_agent / hipad_b2d_agent route-based geo-reference solving."""
        if not gps_route or not world_route:
            return
        try:
            locx = world_route[0][0].location.x
            locy = world_route[0][0].location.y
            lon = gps_route[0][0]['lon']
            lat = gps_route[0][0]['lat']
            earth_radius = 6378137.0

            def equations(vars_):
                x, y = vars_
                eq1 = (
                    (lon * math.cos(x * math.pi / 180.0) - (locx * x * 180.0) / (math.pi * earth_radius))
                    - math.cos(x * math.pi / 180.0) * y
                )
                eq2 = (
                    math.log(math.tan((lat + 90.0) * math.pi / 360.0)) * earth_radius * math.cos(x * math.pi / 180.0)
                    + locy
                    - math.cos(x * math.pi / 180.0) * earth_radius * math.log(math.tan((90.0 + x) * math.pi / 360.0))
                )
                return [eq1, eq2]

            solution = fsolve(equations, [0, 0])
            self.lat_ref = float(solution[0])
            self.lon_ref = float(solution[1])
        except Exception as exc:
            print(f"[Bench2DriveSACEnv] Route-based geo reference solve failed: {exc}")

    def _augment_observation_with_geo(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Overwrite position-dependent fields with GNSS-derived coordinates when available."""
        gps = observation.get('gps')
        if gps is None or len(gps) < 2:
            return observation

        pos = self._gps_to_location(gps)
        observation['pos'] = pos

        if 'can_bus' in observation:
            observation['can_bus'][0] = pos[0]
            observation['can_bus'][1] = -pos[1]

        if 'state' in observation and observation['state'].shape[0] >= 15:
            observation['state'][13] = pos[0]
            observation['state'][14] = pos[1]

        return observation

        print(f"[Bench2DriveSACEnv] Initialized with {len(self.route_configs)} routes")

    def _resolve_fixed_route_idx(self) -> Optional[int]:
        route_name = str(getattr(self.config, "fixed_route_name", "") or "").strip()
        if route_name:
            for idx, route_config in enumerate(self.route_configs):
                if route_config.get("name") == route_name:
                    return int(idx)
            raise ValueError(f"fixed_route_name={route_name} not found in route configs")

        route_idx = int(getattr(self.config, "fixed_route_idx", -1))
        if route_idx >= 0:
            if route_idx >= len(self.route_configs):
                raise ValueError(f"fixed_route_idx={route_idx} out of range [0, {len(self.route_configs) - 1}]")
            return route_idx
        return None

    def _load_route_configs(self, routes_file: str) -> List[dict]:
        """Load route configs from XML file (same format as leaderboard)."""
        route_configs = []

        if not os.path.exists(routes_file):
            print(f"[Bench2DriveSACEnv] Warning: Routes file not found: {routes_file}")
            return []

        tree = ET.parse(routes_file)
        root = tree.getroot()

        for route_elem in root.findall('route'):
            route_id = route_elem.get('id', '0')
            town = route_elem.get('town', 'Town12')

            # Skip problematic towns (Town15 causes CARLA to hang)
            # if town == 'Town15':
            #     continue

            # Load waypoints
            raw_waypoints = []
            waypoints_elem = route_elem.find('waypoints')
            if waypoints_elem is not None:
                positions = waypoints_elem.findall('position')
                for i, pos_elem in enumerate(positions):
                    x = float(pos_elem.get('x', 0))
                    y = float(pos_elem.get('y', 0))
                    z = float(pos_elem.get('z', 0))

                    # Calculate yaw from next position
                    if i < len(positions) - 1:
                        next_x = float(positions[i + 1].get('x', x))
                        next_y = float(positions[i + 1].get('y', y))
                        yaw = np.degrees(np.arctan2(next_y - y, next_x - x))
                    else:
                        yaw = 0.0

                    transform = carla.Transform(
                        carla.Location(x=x, y=y, z=z),
                        carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0)
                    )
                    raw_waypoints.append(transform)

            if raw_waypoints:
                route_configs.append({
                    'id': route_id,
                    'town': town,
                    'waypoints': raw_waypoints,
                    'name': f'RouteScenario_{route_id}'
                })

        print(f"[Bench2DriveSACEnv] Loaded {len(route_configs)} routes from {routes_file}")
        return route_configs

    def _load_routes(self, routes_file: str) -> List[List[carla.Transform]]:
        """Load routes from XML file and interpolate using GlobalRoutePlanner (same as leaderboard)."""
        routes = []

        if not os.path.exists(routes_file):
            print(f"[Bench2DriveSACEnv] Warning: Routes file not found: {routes_file}")
            return [[]]

        tree = ET.parse(routes_file)
        root = tree.getroot()

        for route_elem in root.findall('route'):
            # Load raw waypoints from XML
            raw_waypoints = []
            waypoints_elem = route_elem.find('waypoints')
            if waypoints_elem is not None:
                positions = waypoints_elem.findall('position')
                for i, pos_elem in enumerate(positions):
                    x = float(pos_elem.get('x', 0))
                    y = float(pos_elem.get('y', 0))
                    z = float(pos_elem.get('z', 0))

                    # Calculate yaw from next position
                    if i < len(positions) - 1:
                        next_x = float(positions[i + 1].get('x', x))
                        next_y = float(positions[i + 1].get('y', y))
                        yaw = np.degrees(np.arctan2(next_y - y, next_x - x))
                    else:
                        yaw = 0.0

                    transform = carla.Transform(
                        carla.Location(x=x, y=y, z=z),
                        carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0)
                    )
                    raw_waypoints.append(transform)

            if raw_waypoints:
                # Store raw waypoints for later interpolation
                routes.append(raw_waypoints)

        print(f"[Bench2DriveSACEnv] Loaded {len(routes)} routes from {routes_file}")
        return routes if routes else [[]]

    def _clone_route_scenario_config(self, route_config):
        """Clone a RouteScenarioConfiguration without deepcopying CARLA objects."""
        if route_config is None:
            return None
        cloned = copy.copy(route_config)
        cloned.keypoints = list(route_config.keypoints)
        cloned.scenario_configs = list(route_config.scenario_configs)
        cloned.weather = list(route_config.weather)
        return cloned

    def _interpolate_route(self, raw_waypoints: List[carla.Transform]) -> List[carla.Transform]:
        """
        Interpolate route using GlobalRoutePlanner (same as leaderboard).
        This ensures waypoints are on the road and properly spaced.
        """
        if GlobalRoutePlanner is None:
            # Fallback: return raw waypoints if agents module not available
            return raw_waypoints

        if len(raw_waypoints) < 2:
            return raw_waypoints

        # Get world and map from simulation (via SimulationBackend)
        world = self.world
        carla_map = world.get_map()

        # Create GlobalRoutePlanner with 1.0m resolution (same as leaderboard)
        grp = GlobalRoutePlanner(carla_map, 1.0)

        # Get start/end locations from raw waypoints
        interpolated_route = []

        # Trace route between consecutive waypoints
        for i in range(len(raw_waypoints) - 1):
            start_loc = raw_waypoints[i].location
            end_loc = raw_waypoints[i + 1].location

            # Use trace_route to get interpolated waypoints on the road
            try:
                trace = grp.trace_route(start_loc, end_loc)
                for wp, connection in trace:
                    interpolated_route.append(wp.transform)
            except Exception as e:
                print(f"[Bench2DriveSACEnv] Warning: Failed to trace route segment {i}: {e}")
                # Fallback: use raw waypoint
                interpolated_route.append(raw_waypoints[i])

        # Always add the last waypoint
        interpolated_route.append(raw_waypoints[-1])

        return interpolated_route

    def reset(self, seed: Optional[int] = None, force_new_route: bool = False) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Reset environment and start new episode.

        Scene loading aligned with leaderboard_eval:
        1. Randomly select route (or keep same route based on route_switch_interval)
        2. Load world for route's town using SimulationBackend.load_world()
        3. Set up CarlaDataProvider
        4. Spawn ego vehicle at route start (with elevation)
        5. Initialize sensors

        Args:
            seed: Random seed
            force_new_route: If True, force switching to a new route regardless of interval

        Returns:
            observation: dict with 'rgb' and 'state'
            info: dict with episode info
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Clean up previous episode
        self._cleanup_actors()
        self._reset_roach_bev_target_state()

        fixed_route_idx = self._resolve_fixed_route_idx()

        # Determine if we should switch route
        self.episodes_on_current_route += 1
        if fixed_route_idx is not None:
            should_switch_route = force_new_route or self._current_route_config is None
        else:
            should_switch_route = force_new_route or self.episodes_on_current_route >= self.route_switch_interval

        if should_switch_route or self._current_route_config is None:
            # Select new route, optionally pinned for controlled comparisons.
            if fixed_route_idx is not None:
                self.current_route_idx = fixed_route_idx
            elif self.config.random_routes:
                self.current_route_idx = random.randint(0, len(self.route_configs) - 1)
            else:
                self.current_route_idx = (self.current_route_idx + 1) % len(self.route_configs)
            self._current_route_config = self.route_configs[self.current_route_idx]
            self.episodes_on_current_route = 1  # Reset counter
            route_config = self._current_route_config
            town = route_config['town']
            raw_waypoints = route_config['waypoints']

            print(f"[DEBUG] Selected route {self.current_route_idx} ({route_config['name']}) in {town}")

            # Load world for this route (same as leaderboard_eval._load_and_wait_for_world)
            print(f"[DEBUG] About to load world for {town}...")
            try:
                self._load_world_for_route(town)
            except TimeoutError as e:
                print(f"[ERROR] Timeout loading world for {town}: {e}")
                raise
            except Exception as e:
                print(f"[ERROR] Failed to load world for {town}: {type(e).__name__}: {e}")
                raise

            # Interpolate route using GlobalRoutePlanner (same as leaderboard)
            if raw_waypoints and len(raw_waypoints) > 0:
                self._current_interpolated_route = self._interpolate_route(raw_waypoints)
                print(f"[DEBUG] Interpolated route: {len(raw_waypoints)} raw waypoints -> {len(self._current_interpolated_route)} interpolated waypoints")
            else:
                self._current_interpolated_route = raw_waypoints
        else:
            # Keep same route, just select new starting point
            route_config = self._current_route_config
            town = route_config['town']
            print(f"[DEBUG] Reusing route {self.current_route_idx} ({route_config['name']}) in {town} (episode {self.episodes_on_current_route}/{self.route_switch_interval})")

            # CRITICAL: Verify current world matches the route's town
            # If world loading previously failed, we may be in the wrong town
            current_map_name = self.world.get_map().name.split("/")[-1] if self.world else None
            if current_map_name != town:
                print(f"[WARNING] World town mismatch! Current: {current_map_name}, Required: {town}. Reloading world...")
                try:
                    self._load_world_for_route(town)
                except TimeoutError as e:
                    print(f"[ERROR] Timeout loading world for {town}: {e}")
                    raise
                except Exception as e:
                    print(f"[ERROR] Failed to load world for {town}: {type(e).__name__}: {e}")
                    raise

            # Re-initialize CarlaDataProvider with the current world (it was cleaned up)
            CarlaDataProvider.set_client(self.client)
            CarlaDataProvider.set_world(self.world)
            CarlaDataProvider.set_traffic_manager_port(self.config.simulation.traffic_manager_port)
            CarlaDataProvider.set_runtime_init_mode(False)
            self.world.reset_all_traffic_lights()
            if self.traffic_manager is not None:
                self.traffic_manager.set_random_device_seed(getattr(self.config.simulation, 'traffic_manager_seed', 0))
            self.world.tick()

        route_name = route_config['name']
        town = route_config['town']
        full_route_config = self._clone_route_scenario_config(self.route_scenario_configs.get(route_name))
        if full_route_config is None:
            raise RuntimeError(f"Missing full RouteScenario configuration for {route_name}")

        # Spawn ego and background actors using the same RouteScenario path as
        # leaderboard evaluation. This is what actually creates runtime traffic
        # and parked vehicles for the route.
        self.route_scenario = RouteScenario(world=self.world, config=full_route_config, debug_mode=0)
        self.vehicle = self.route_scenario.ego_vehicles[0]
        self._disable_autopilot(self.vehicle)
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False))
        route = list(self.route_scenario.route)
        route_transforms = [transform for transform, _ in route]
        self._current_interpolated_route = route_transforms
        self.current_waypoint_idx = 0
        self._last_scenario_build_time = 0.0
        GameTime.restart()

        # Setup sensors
        self.obs_builder.setup_sensors(self.world, self.vehicle)

        # Setup collision and lane invasion sensors
        self._setup_event_sensors()

        # Build route waypoints for navigation
        self.route_waypoints = self._build_route_waypoints(route_transforms)
        self.current_waypoint_idx = 0
        self.route_progress = 0.0
        self.prev_route_progress = 0.0

        self._update_geo_reference_from_route(self.route_scenario.gps_route, route)

        # Initialize RoutePlanner for navigation (from vad_b2d_agent.py)
        self._setup_route_planner(route)

        self.episode_step = 0

        # Reset managers
        self.reward_calculator.reset(self.route_scenario)
        self.step_manager.reset()
        self._reset_stuck_state()
        self._reset_warmup_observations = []

        # Wait for sensors to initialize and vehicle to settle
        # Also clear any spawn-time collision events
        for _ in range(20):
            self.world.tick()

        # Ensure we are using fresh frames from the current sensor generation.
        # Without this, late callbacks from the previous episode can leak stale
        # images into the next reset, which shows the old ego vehicle and
        # missing current traffic in the first BEV/RGB frames.
        requested_frame = None
        for _ in range(40):
            requested_frame = self.world.tick()
            try:
                frame = self._wait_for_sensor_frame(requested_frame)
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError('Timed out waiting for a complete sensor packet during reset')

        # Clear any collision events that occurred during spawn/settling
        self.reward_calculator.reset(self.route_scenario)
        self.step_manager.reset()

        self._scene_instance_id += 1
        self._current_scene_token = f"{route_name}__ep{self._scene_instance_id:06d}"

        # Rear cameras can still lag for the first complete packet(s) after a
        # reset. Keep the first few packets as hidden warmup observations for
        # temporal models, then return the next packet to the caller.
        warmup_frames = [int(frame)]
        warmup_packet_count = 3
        for _ in range(max(0, warmup_packet_count - 1)):
            requested_frame = None
            for _ in range(40):
                requested_frame = self.world.tick()
                try:
                    frame = self._wait_for_sensor_frame(requested_frame)
                    warmup_frames.append(int(frame))
                    break
                except RuntimeError:
                    continue
            else:
                raise RuntimeError('Timed out waiting for additional warmup sensor packets during reset')

        self._reset_warmup_observations = []
        for idx, warmup_frame in enumerate(warmup_frames):
            warmup_observation = self.obs_builder.get_observation(self.vehicle, frame=warmup_frame)
            warmup_observation = self._augment_observation_with_geo(warmup_observation)
            warmup_target_point, warmup_command = self._get_navigation_info(warmup_observation)
            warmup_observation['target_point'] = warmup_target_point
            warmup_observation['command'] = warmup_command
            warmup_observation['scene_token'] = self._current_scene_token
            warmup_observation['timestamp'] = float(idx - len(warmup_frames)) / 20.0
            self._reset_warmup_observations.append(
                {
                    key: (value.copy() if isinstance(value, np.ndarray) else value)
                    for key, value in warmup_observation.items()
                }
            )

        requested_frame = None
        for _ in range(40):
            requested_frame = self.world.tick()
            try:
                frame = self._wait_for_sensor_frame(requested_frame)
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError('Timed out waiting for a stable sensor packet during reset')

        # Get initial observation
        observation = self.obs_builder.get_observation(self.vehicle, frame=frame)
        observation = self._augment_observation_with_geo(observation)

        # Get navigation information
        target_point, command = self._get_navigation_info(observation)
        observation['target_point'] = target_point
        observation['command'] = command
        observation['scene_token'] = self._current_scene_token
        observation['timestamp'] = self.episode_step / 20.0

        info = {
            'route_name': route_name,
            'scene_token': self._current_scene_token,
            'route_idx': self.current_route_idx,
            'town': town,
            'start_idx': 0,
            'route_progress': 0.0,
            'sensor_frame_discarded': int(warmup_frames[-1]),
            'sensor_frames_discarded': [int(x) for x in warmup_frames],
            'sensor_frame_requested': int(requested_frame if requested_frame is not None else frame),
            'sensor_frame_used': int(frame),
            'sensor_frame_exact': bool(requested_frame == frame if requested_frame is not None else True),
            'ego_x': float(self.vehicle.get_transform().location.x),
            'ego_y': float(self.vehicle.get_transform().location.y),
            'ego_yaw': float(self.vehicle.get_transform().rotation.yaw),
        }
        self._record_motion_state(carla.VehicleControl(), 0.0)
        self._maybe_cache_roach_bev_target(observation, info)

        return observation, info

    def get_reset_warmup_observation(self) -> Optional[Dict[str, np.ndarray]]:
        if not self._reset_warmup_observations:
            return None
        return {
            key: (value.copy() if isinstance(value, np.ndarray) else value)
            for key, value in self._reset_warmup_observations[-1].items()
        }

    def get_reset_warmup_observations(self) -> List[Dict[str, np.ndarray]]:
        return [
            {
                key: (value.copy() if isinstance(value, np.ndarray) else value)
                for key, value in observation.items()
            }
            for observation in self._reset_warmup_observations
        ]

    def _load_world_for_route(self, town: str, max_retries: int = 2):
        """
        Load world for the given town using SimulationBackend.
        Same as leaderboard_eval._load_and_wait_for_world()
        With retry logic for handling CARLA timeouts.
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                # Load world using SimulationBackend (sets up sync mode, CarlaDataProvider, etc.)
                world_result = self._sim_backend.load_world(town)
                self.world = world_result.world
                self._update_geo_reference()
                self._reset_roach_bev_target_state()
                print(f"[DEBUG] Loaded world for {town}")
                return
            except TimeoutError as e:
                last_error = e
                print(f"[WARNING] World load attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"[WARNING] Retrying world load in 3 seconds...")
                    import time
                    time.sleep(3)
                else:
                    print(f"[ERROR] Failed to load world {town} after {max_retries} attempts")
                    raise last_error
            except Exception as e:
                print(f"[ERROR] Unexpected error loading world {town}: {type(e).__name__}: {e}")
                raise

    def _select_route_start(self, route: List[carla.Transform], route_idx: int = 0) -> Tuple[carla.Transform, int]:
        """
        Select a fixed starting point from the route.
        The returned transform is snapped to the current road centerline when
        possible, and the logical route index is preserved.
        """
        if not route or len(route) == 0:
            raise RuntimeError("Empty route, cannot select start point")

        current_map = self.world.get_map()
        current_map_name = current_map.name.split("/")[-1] if current_map else "Unknown"
        route_idx = min(max(route_idx, 0), len(route) - 1)
        route_start = route[route_idx]
        sample_loc = route_start.location
        sample_waypoint = current_map.get_waypoint(sample_loc)
        if sample_waypoint is None:
            print(
                f"[WARNING] Route validation failed: route[{route_idx}] {sample_loc} has no corresponding waypoint in {current_map_name}"
            )
            print(
                f"[WARNING] This usually means the route belongs to a different town than the current world"
            )
            fallback_transform = carla.Transform(route_start.location, route_start.rotation)
            fallback_transform.location.z += 0.5
            return fallback_transform, route_idx

        dist = sample_waypoint.transform.location.distance(sample_loc)
        if dist > 100:
            print(
                f"[WARNING] Route validation warning: route[{route_idx}] {sample_loc} is {dist:.1f}m from nearest waypoint {sample_waypoint.transform.location}"
            )
            print(f"[WARNING] Current map: {current_map_name}, route may belong to different town")

        road_transform = sample_waypoint.transform
        road_transform.location.z += 0.5
        print(f"[DEBUG] Spawning at route[{route_idx}] (fixed start candidate, {len(route)} total waypoints)")
        return road_transform, route_idx

    def _select_route_start_candidates(self, route: List[carla.Transform], max_candidates: int = 10) -> List[Tuple[carla.Transform, int]]:
        """
        Build deterministic spawn candidates from the beginning of the route.
        This keeps initialization aligned with Bench2Drive route starts while
        avoiding random spawn points.
        """
        if not route:
            raise RuntimeError("Empty route, cannot select start candidates")

        candidate_count = min(max_candidates, len(route))
        return [self._select_route_start(route, route_idx=i) for i in range(candidate_count)]

    def _is_spawn_location_free(self, transform: carla.Transform, radius: float = 2.0) -> bool:
        """Check if a spawn location is free of other actors."""
        # Get all actors in the world
        actors = self.world.get_actors()

        # Check distance to all vehicles and walkers
        for actor in actors:
            if actor.type_id.startswith('vehicle') or actor.type_id.startswith('walker'):
                dist = actor.get_location().distance(transform.location)
                if dist < radius:
                    return False

        return True

    def _build_route_waypoints(self, route: List[carla.Transform]) -> List[carla.Waypoint]:
        """Build a dense, de-duplicated route centerline from route transforms."""
        route_waypoints = []
        if not route:
            return route_waypoints

        carla_map = self.world.get_map()
        last_wp = None
        for transform in route:
            waypoint = carla_map.get_waypoint(transform.location)
            if waypoint is None:
                continue

            if last_wp is not None:
                same_lane = (
                    waypoint.road_id == last_wp.road_id and
                    waypoint.lane_id == last_wp.lane_id
                )
                same_position = waypoint.transform.location.distance(last_wp.transform.location) < 0.75
                if same_lane and same_position:
                    continue

            route_waypoints.append(waypoint)
            last_wp = waypoint
        return route_waypoints

    def _update_route_progress(self):
        """
        Advance the current route waypoint index monotonically.

        Reward and off-route checks should reference the ego vehicle's current
        route transform, similar to carla-roach, not the route spawn point.
        Search a bounded forward window to keep the index stable and cheap.
        """
        if self.vehicle is None or not self.route_waypoints:
            return

        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_forward = vehicle_transform.rotation.get_forward_vector()
        start_idx = max(0, self.current_waypoint_idx)
        end_idx = min(len(self.route_waypoints), start_idx + 80)
        if start_idx >= end_idx:
            return

        best_idx = start_idx
        best_score = float("inf")
        fallback_idx = start_idx
        fallback_distance = float("inf")
        for idx in range(start_idx, end_idx):
            waypoint = self.route_waypoints[idx]
            wp_transform = waypoint.transform
            distance = vehicle_location.distance(wp_transform.location)

            if distance < fallback_distance:
                fallback_distance = distance
                fallback_idx = idx

            wp_forward = wp_transform.rotation.get_forward_vector()
            heading_alignment = float(
                vehicle_forward.x * wp_forward.x + vehicle_forward.y * wp_forward.y
            )
            heading_alignment = float(np.clip(heading_alignment, -1.0, 1.0))
            heading_error_deg = float(np.degrees(np.arccos(heading_alignment)))

            delta = wp_transform.location - vehicle_location
            forward_projection = float(delta.x * vehicle_forward.x + delta.y * vehicle_forward.y)

            if heading_error_deg > 75.0 and distance > 3.0:
                continue
            if forward_projection < -4.0 and distance > 3.0:
                continue

            score = distance + 0.05 * heading_error_deg
            if score < best_score:
                best_score = score
                best_idx = idx

        if best_score == float("inf"):
            best_idx = fallback_idx

        self.current_waypoint_idx = best_idx
        self.route_progress = float(
            self.current_waypoint_idx / max(1, len(self.route_waypoints) - 1)
        ) if self.route_waypoints else 0.0

    def _spawn_ego_vehicle(self, transform: carla.Transform) -> carla.Vehicle:
        """
        Spawn ego vehicle using CarlaDataProvider (same as RouteScenario._spawn_ego_vehicle).

        This ensures proper actor registration and follows leaderboard conventions.
        """
        # Elevate transform to avoid ground collision (same as leaderboard route_scenario.py)
        elevate_transform = carla.Transform(
            carla.Location(
                x=transform.location.x,
                y=transform.location.y,
                z=transform.location.z + 0.5  # Elevate by 0.5m
            ),
            transform.rotation
        )

        # Use CarlaDataProvider to spawn actor (same as RouteScenario)
        ego_vehicle = CarlaDataProvider.request_new_actor(
            'vehicle.lincoln.mkz_2020',
            elevate_transform,
            rolename='hero'
        )

        if ego_vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle at the start of the route")

        # Set spectator to follow vehicle (same as RouteScenario)
        spectator = self.world.get_spectator()
        spectator.set_transform(carla.Transform(
            elevate_transform.location + carla.Location(z=50),
            carla.Rotation(pitch=-90)
        ))

        # Tick world to ensure vehicle is properly spawned
        self.world.tick()

        # Disable autopilot (RL will control)
        self._disable_autopilot(ego_vehicle)

        print(f"[DEBUG] Spawned ego vehicle at {elevate_transform.location}")

        return ego_vehicle

    def _setup_event_sensors(self):
        """Setup collision and lane invasion sensors."""
        blueprint_library = self.world.get_blueprint_library()

        # Collision sensor
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.collision_sensor.listen(self.reward_calculator.on_collision)

        # Lane invasion sensor
        lane_bp = blueprint_library.find('sensor.other.lane_invasion')
        self.lane_sensor = self.world.spawn_actor(
            lane_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.lane_sensor.listen(self.reward_calculator.on_lane_invasion)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: [steer, throttle/brake] in [-1, 1]

        Returns:
            observation: dict with 'rgb' and 'state'
            reward: float
            terminated: bool (episode ended due to failure/success)
            truncated: bool (episode ended due to time limit)
            info: dict with step info
        """
        # Apply action
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle_brake = float(np.clip(action[1], -1.0, 1.0))

        if throttle_brake >= 0:
            throttle = throttle_brake
            brake = 0.0
        else:
            throttle = 0.0
            brake = -throttle_brake

        control = carla.VehicleControl(
            steer=steer,
            throttle=throttle,
            brake=brake,
            hand_brake=False,
            manual_gear_shift=False
        )
        self._disable_autopilot(self.vehicle)
        self.vehicle.apply_control(control)

        # Tick world
        requested_frame = self.world.tick()
        snapshot = self.world.get_snapshot()
        GameTime.on_carla_tick(snapshot.timestamp)
        CarlaDataProvider.on_carla_tick()
        self._maybe_build_runtime_scenarios(snapshot.timestamp.elapsed_seconds)
        scenario_running = True
        if self.route_scenario is not None and self.route_scenario.scenario_tree is not None:
            py_trees.blackboard.Blackboard().set("AV_control", control, overwrite=True)
            self.route_scenario.scenario_tree.tick_once()
            scenario_running = self.route_scenario.scenario_tree.status == py_trees.common.Status.RUNNING
        frame = self._wait_for_sensor_frame(requested_frame)
        self.episode_step += 1

        # Get observation
        observation = self.obs_builder.get_observation(self.vehicle, frame=frame)
        observation = self._augment_observation_with_geo(observation)

        # Get navigation information
        target_point, command = self._get_navigation_info(observation)
        observation['target_point'] = target_point
        observation['command'] = command
        observation['scene_token'] = self._current_scene_token if self._current_scene_token is not None else (
            self._current_route_config['name'] if self._current_route_config else f"route_{self.current_route_idx}"
        )
        observation['timestamp'] = self.episode_step / 20.0

        self._update_route_progress()

        current_route_progress = self.route_progress
        progress_delta = max(0.0, current_route_progress - self.prev_route_progress)
        route_complete = bool(self.route_waypoints) and self.current_waypoint_idx >= len(self.route_waypoints) - 1

        # Compute reward. Failure truncation reasons are resolved before reward
        # computation so opt-in reward variants can assign terminal penalties
        # from the real StepManager/env source instead of guessing.
        time_limit_reached, elapsed_time, time_limit_reason = self.step_manager.step(self.world)
        max_episode_reached = self.episode_step >= self.config.max_episode_steps
        truncation_reason = time_limit_reason if time_limit_reached else None
        if max_episode_reached and truncation_reason is None:
            truncation_reason = "max_episode_steps"
        reward, done_reward, reward_info = self.reward_calculator.compute_reward(
            self.vehicle,
            self.route_waypoints[self.current_waypoint_idx:],
            elapsed_time,
            ev_control=control,
            route_scenario=self.route_scenario,
            scenario_running=scenario_running,
            route_complete=route_complete,
            route_progress_delta=progress_delta,
            truncation_reason=truncation_reason,
        )
        progress_reward = float(self.config.reward.route_progress_reward_scale) * float(progress_delta)
        reward += progress_reward
        reward_info['r_progress'] = float(progress_reward)
        reward_info['route_progress'] = current_route_progress
        reward_info['route_progress_delta'] = progress_delta
        reward_info['scenario_running'] = bool(scenario_running)

        # The legacy waypoint-completion heuristic was designed for the
        # simplified RL-only environment. Once RouteScenario is active, that
        # heuristic can mark a route as completed within a handful of ticks
        # even when the ego vehicle has barely moved, because it is comparing
        # against dense scenario route waypoints near the spawn point. Keep the
        # background traffic/scenario pipeline, but defer episode termination to
        # the reward logic (collision/off-route/stuck) and the overall episode
        # step limit instead of this legacy completion shortcut.

        # Determine termination
        # Keep ticking RouteScenario so background traffic and parked actors are
        # maintained, but do not terminate the RL episode merely because the
        # scenario tree stopped running. In practice the tree often reaches a
        # terminal status long before the ego vehicle has had a chance to
        # accelerate, which was causing repeated 3-11 step episodes with near
        # zero speed.
        terminated = bool(done_reward)
        truncated = bool(time_limit_reached or max_episode_reached)
        if truncated and 'truncation_reason' not in reward_info:
            reward_info['truncation_reason'] = truncation_reason
        if terminated and 'termination' not in reward_info:
            reward_info['termination'] = 'env_terminal'

        # Get vehicle speed for info
        velocity = self.vehicle.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
        traffic_debug = self._summarize_nearby_vehicles(radius=80.0)
        self.prev_route_progress = current_route_progress
        self._record_motion_state(control, current_route_progress)

        info = {
            'step': self.episode_step,
            'elapsed_time': elapsed_time,
            'route_name': self._current_route_config['name'] if self._current_route_config else f"route_{self.current_route_idx}",
            'scene_token': self._current_scene_token if self._current_scene_token is not None else f"route_{self.current_route_idx}",
            'route_idx': self.current_route_idx,
            'current_waypoint_idx': int(self.current_waypoint_idx),
            'route_progress': current_route_progress,
            'sensor_frame_requested': int(requested_frame),
            'sensor_frame_used': int(frame),
            'sensor_frame_exact': bool(requested_frame == frame),
            'speed': float(speed),
            'ego_x': float(self.vehicle.get_transform().location.x),
            'ego_y': float(self.vehicle.get_transform().location.y),
            'ego_yaw': float(self.vehicle.get_transform().rotation.yaw),
            'scenario_running': bool(scenario_running),
            'termination': reward_info.get('termination'),
            'termination_reasons': reward_info.get('termination'),
            'truncation_reason': reward_info.get('truncation_reason'),
            **traffic_debug,
            **reward_info
        }
        self._maybe_cache_roach_bev_target(observation, info)

        return observation, reward, terminated, truncated, info

    def _setup_route_planner(self, route: List[carla.Transform]):
        """Initialize the project-owned planner for generic observations."""
        self._route_planner = NavigationRoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        if location_route_to_gps is not None:
            gps_route = location_route_to_gps(route, self.lat_ref, self.lon_ref)
            self._route_planner.set_route(gps_route, True)
        else:
            self._route_planner.set_route(route, False)

    def _get_navigation_info(self, observation: Optional[Dict[str, np.ndarray]] = None) -> Tuple[np.ndarray, int]:
        """
        Get navigation information from RoutePlanner.

        Returns:
            target_point: [2] (x, y) in ego coordinates
            command: int (1-6) navigation command
        """
        if self._route_planner is None or self.vehicle is None:
            # Default: straight ahead
            return np.array([10.0, 0.0], dtype=np.float32), 3  # STRAIGHT

        transform = self.vehicle.get_transform()
        gps = None if observation is None else observation.get('gps')
        if gps is not None:
            pos = self._gps_to_location(gps)
        else:
            pos = np.array([transform.location.x, transform.location.y], dtype=np.float32)

        # Get navigation from RoutePlanner (same as vad_b2d_agent.py line 292)
        near_node, near_command = self._route_planner.run_step(pos)

        # Convert near_node from GPS/world coordinates to the HiP-AD/VAD local
        # convention, matching vad_b2d_agent.py:
        #   R(compass) @ [target_x - ego_x, -(target_y - ego_y)]
        if observation is not None and 'compass' in observation:
            compass = float(observation['compass'])
        else:
            imu = self.obs_builder._sensor_data.get('imu')
            compass = float(imu[6]) if imu is not None else 0.0
        if np.isnan(compass):
            compass = 0.0
        rotation_matrix = np.array([
            [np.cos(compass), -np.sin(compass)],
            [np.sin(compass), np.cos(compass)]
        ], dtype=np.float32)

        target_delta = np.array(
            [near_node[0] - pos[0], -near_node[1] + pos[1]],
            dtype=np.float32,
        )
        target_point_local = rotation_matrix @ target_delta

        # Command mapping: VAD uses 1-6 (LEFT=1, RIGHT=2, STRAIGHT=3, LANEFOLLOW=4, CHANGELANELEFT=5, CHANGELANERIGHT=6)
        # near_command values from RoutePlanner: need to check what values it returns
        # Assuming it returns similar to RoadOption values
        command_value = getattr(near_command, 'value', near_command)
        try:
            command = int(command_value)
        except (TypeError, ValueError):
            command = 3  # Default to STRAIGHT only for an actually invalid command.

        # Clamp to valid range [1, 6]
        command = max(1, min(6, command))

        return target_point_local.astype(np.float32), int(command)

    def _cleanup_actors(self):
        """Clean up actors from previous episode using CarlaDataProvider."""
        # Stop sensor listeners first to prevent callbacks during destruction
        try:
            if hasattr(self, 'collision_sensor') and self.collision_sensor:
                if self.collision_sensor.is_alive:
                    self.collision_sensor.stop()
        except Exception:
            pass

        try:
            if hasattr(self, 'lane_sensor') and self.lane_sensor:
                if self.lane_sensor.is_alive:
                    self.lane_sensor.stop()
        except Exception:
            pass

        # Wait a tick for sensor callbacks to complete
        # Skip if world is None or may be disconnected
        try:
            if self.world and self.world.get_settings():
                self.world.tick()
        except Exception:
            # World may be disconnected or invalid
            pass

        # Clean up observation builder sensors
        self.obs_builder.cleanup()

        if self.route_scenario is not None:
            try:
                if getattr(self.route_scenario, "_parked_ids", None):
                    self.route_scenario.client.apply_batch(
                        [carla.command.DestroyActor(actor_id) for actor_id in self.route_scenario._parked_ids]
                    )
            except Exception as e:
                print(f"[DEBUG] RouteScenario parked actor cleanup error: {e}")
            try:
                self.route_scenario.remove_all_actors()
            except Exception as e:
                print(f"[DEBUG] RouteScenario actor cleanup error: {e}")
            self.route_scenario = None

        # Use CarlaDataProvider to clean up actors (same as leaderboard_eval._cleanup)
        # This properly removes all scenario-related actors
        try:
            CarlaDataProvider.cleanup()
        except Exception as e:
            print(f"[DEBUG] CarlaDataProvider cleanup error: {e}")

        # Reset attributes
        self.collision_sensor = None
        self.lane_sensor = None
        self.vehicle = None
        self._roach_bev_target_cache.clear()

    def _summarize_nearby_vehicles(self, radius: float = 80.0) -> Dict:
        if self.world is None or self.vehicle is None:
            return {
                'world_vehicle_count': 0,
                'nearby_vehicle_count': 0,
                'nearby_vehicle_speeds': [],
            }

        ego_location = self.vehicle.get_transform().location
        nearby = []
        world_count = 0
        for actor in self.world.get_actors().filter('vehicle.*'):
            if actor.id == self.vehicle.id:
                continue
            world_count += 1
            try:
                actor_loc = actor.get_transform().location
                distance = ego_location.distance(actor_loc)
                if distance <= radius:
                    vel = actor.get_velocity()
                    speed = float(np.linalg.norm([vel.x, vel.y, vel.z]))
                    nearby.append((distance, speed, actor.id, actor.attributes.get('role_name')))
            except Exception:
                continue

        nearby.sort(key=lambda item: item[0])
        return {
            'world_vehicle_count': int(world_count),
            'nearby_vehicle_count': int(len(nearby)),
            'nearby_vehicle_speeds': [
                {
                    'distance': float(round(distance, 3)),
                    'speed': float(round(speed, 4)),
                    'id': int(actor_id),
                    'role_name': role_name,
                }
                for distance, speed, actor_id, role_name in nearby[:8]
            ],
        }

    def _maybe_build_runtime_scenarios(self, elapsed_seconds: float):
        if self.route_scenario is None or self.vehicle is None:
            return
        if elapsed_seconds - self._last_scenario_build_time < self._scenario_build_interval:
            return
        self.route_scenario.build_scenarios(self.vehicle, debug=False)
        self.route_scenario.spawn_parked_vehicles(self.vehicle)
        self._last_scenario_build_time = elapsed_seconds

    def set_route_switch_interval(self, interval: int):
        """
        Set the interval for switching routes.
        Args:
            interval: Number of episodes to run on the same route before switching.
                      Set to 1 to switch every episode (default).
        """
        self.route_switch_interval = max(1, interval)
        print(f"[Bench2DriveSACEnv] Route switch interval set to {self.route_switch_interval}")

    def close(self):
        """Close environment and cleanup using SimulationBackend."""
        self._cleanup_actors()
        self.obs_builder.cleanup()
        if self._roach_bev_target_generator is not None:
            self._roach_bev_target_generator.close()
        self._roach_bev_target_cache.clear()

        # Use SimulationBackend to properly reset world settings and close
        if hasattr(self, '_sim_backend') and self._sim_backend:
            self._sim_backend.close(reset_world_settings=True, stop_server=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
