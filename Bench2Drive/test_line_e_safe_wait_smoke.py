#!/usr/bin/env python3
"""No-server smoke checks for Line-E bbox safety-wait and bounded reset."""

from __future__ import annotations

from enum import Enum
import math
import os
from pathlib import Path
import sys
import types


BENCH2DRIVE_ROOT = Path(__file__).resolve().parent
for path in (
    BENCH2DRIVE_ROOT / "leaderboard",
    BENCH2DRIVE_ROOT / "scenario_runner",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


try:
    import carla  # type: ignore
except ImportError:
    carla = types.ModuleType("carla")

    class LaneType:
        Driving = "Driving"

    class TrafficLightState(Enum):
        Red = 0
        Yellow = 1
        Green = 2

    class VehicleControl:
        def __init__(self, steer=0.0, throttle=0.0, brake=0.0, **kwargs):
            del kwargs
            self.steer = float(steer)
            self.throttle = float(throttle)
            self.brake = float(brake)

    carla.LaneType = LaneType
    carla.TrafficLightState = TrafficLightState
    carla.VehicleControl = VehicleControl
    carla.Vector3D = object
    carla.Vehicle = object
    carla.CollisionEvent = object
    carla.LaneInvasionEvent = object
    sys.modules["carla"] = carla


from rl.reward import RewardCalculator, RewardConfig


class Vector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def distance(self, other) -> float:
        return float(math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2))

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)


class Rotation:
    def __init__(self, yaw=0.0):
        self.yaw = float(yaw)

    def get_forward_vector(self):
        radians = math.radians(self.yaw)
        return Vector(math.cos(radians), math.sin(radians), 0.0)


class Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location or Vector()
        self.rotation = rotation or Rotation()


class BoundingBox:
    def __init__(self, extent_x=1.4, extent_y=0.9):
        self.location = Vector()
        self.rotation = Rotation()
        self.extent = Vector(extent_x, extent_y, 0.8)


class Waypoint:
    def __init__(self, location=None, road_id=1, lane_id=1):
        self.transform = Transform(location or Vector(), Rotation())
        self.road_id = int(road_id)
        self.lane_id = int(lane_id)
        self.lane_type = carla.LaneType.Driving

    def get_left_lane(self):
        return None

    def get_right_lane(self):
        return None


class ActorCollection(list):
    def filter(self, pattern):
        if pattern == "vehicle.*":
            return ActorCollection(actor for actor in self if actor.type_id.startswith("vehicle."))
        if pattern == "walker.*":
            return ActorCollection(actor for actor in self if actor.type_id.startswith("walker."))
        return ActorCollection(actor for actor in self if actor.type_id == pattern)


class FakeMap:
    def __init__(self, waypoint):
        self.waypoint = waypoint

    def get_waypoint(self, *args, **kwargs):
        del args, kwargs
        return self.waypoint


class FakeWorld:
    def __init__(self, waypoint):
        self.actors = ActorCollection()
        self.map = FakeMap(waypoint)

    def get_actors(self):
        return self.actors

    def get_map(self):
        return self.map


class FakeActor:
    def __init__(
        self,
        actor_id,
        type_id,
        x,
        y=0.0,
        yaw=0.0,
        speed=0.0,
        extent_x=1.4,
        extent_y=0.9,
    ):
        self.id = int(actor_id)
        self.type_id = str(type_id)
        self._transform = Transform(Vector(x, y, 0.0), Rotation(yaw))
        self._velocity = Vector(speed * math.cos(math.radians(yaw)), speed * math.sin(math.radians(yaw)), 0.0)
        self.bounding_box = BoundingBox(extent_x, extent_y)
        self._world = None
        self.state = carla.TrafficLightState.Green

    def get_transform(self):
        return self._transform

    def get_velocity(self):
        return self._velocity

    def get_world(self):
        return self._world

    def get_control(self):
        return carla.VehicleControl()


def make_scene(blocker_x=None, blocker_y=0.0, blocker_speed=0.0):
    waypoint = Waypoint(Vector())
    world = FakeWorld(waypoint)
    ego = FakeActor(1, "vehicle.ego", 0.0, speed=0.0)
    ego._world = world
    world.actors.append(ego)
    blocker = None
    if blocker_x is not None:
        blocker = FakeActor(2, "vehicle.blocker", blocker_x, blocker_y, speed=blocker_speed)
        blocker._world = world
        world.actors.append(blocker)
    return world, ego, blocker, [waypoint]


def step(calculator, ego, route, progress_delta=0.0):
    return calculator.compute_reward(
        ego,
        route,
        elapsed_time=0.0,
        ev_control=carla.VehicleControl(),
        route_progress_delta=progress_delta,
    )


def assert_close(value, expected, tolerance=1e-6):
    if abs(float(value) - float(expected)) > tolerance:
        raise AssertionError(f"expected {expected}, got {value}")


def test_bbox_gap_catches_center_distance_mismatch():
    _, ego, _, route = make_scene(blocker_x=10.0)
    calculator = RewardCalculator(RewardConfig(reward_variant="line_e", line_e_blocked_timeout_steps=1000))
    _, done, info = step(calculator, ego, route)
    assert not done
    assert info["bbox_safety_wait_active"]
    assert info["is_legal_wait"]
    assert not info["is_free_road"]
    assert info["vehicle_stuck_counter"] == 0
    assert info["safety_wait_blocker_id"] == 2
    assert_close(info["safety_wait_blocker_center_distance"], 10.0)
    assert_close(info["safety_wait_blocker_longitudinal_clearance"], 7.2)


def test_adjacent_lane_does_not_trigger_bbox_wait():
    _, ego, _, route = make_scene(blocker_x=10.0, blocker_y=3.5)
    calculator = RewardCalculator(RewardConfig(reward_variant="line_e", line_e_blocked_timeout_steps=1000))
    _, done, info = step(calculator, ego, route)
    assert not done
    assert not info["bbox_safety_wait_active"]
    assert not info["is_legal_wait"]
    assert info["is_free_road"]
    assert info["vehicle_stuck_counter"] == 1


def test_blocker_departure_clears_wait_immediately():
    world, ego, blocker, route = make_scene(blocker_x=10.0)
    calculator = RewardCalculator(RewardConfig(reward_variant="line_e", line_e_blocked_timeout_steps=1000))
    _, _, waiting = step(calculator, ego, route)
    assert waiting["bbox_safety_wait_active"]
    world.actors.remove(blocker)
    _, done, cleared = step(calculator, ego, route)
    assert not done
    assert not cleared["bbox_safety_wait_active"]
    assert not cleared["is_legal_wait"]
    assert cleared["blocked_wait_counter"] == 0


def test_red_light_wait_and_green_resume():
    world, ego, _, route = make_scene()
    light = FakeActor(3, "traffic.traffic_light", 4.0)
    light._world = world
    light.state = carla.TrafficLightState.Red
    world.actors.append(light)
    calculator = RewardCalculator(RewardConfig(reward_variant="line_e", line_e_blocked_timeout_steps=2))
    _, done_red, red = step(calculator, ego, route)
    assert not done_red
    assert red["is_legal_wait"]
    assert red["vehicle_stuck_counter"] == 0
    assert red["blocked_wait_counter"] == 0
    light.state = carla.TrafficLightState.Green
    _, done_green, green = step(calculator, ego, route)
    assert not done_green
    assert not green["is_legal_wait"]
    assert green["is_free_road"]


def test_true_free_road_stop_still_terminates_as_vehicle_stuck():
    _, ego, _, route = make_scene()
    config = RewardConfig(
        reward_variant="line_e",
        vehicle_stuck_step=1,
        hard_stuck_step=2,
        line_e_blocked_timeout_steps=1000,
    )
    calculator = RewardCalculator(config)
    _, done_first, _ = step(calculator, ego, route)
    _, done_second, info = step(calculator, ego, route)
    assert not done_first
    assert done_second
    assert "vehicle_stuck" in info["termination_reasons"]
    assert "blocked_timeout" not in info["termination_reasons"]


def test_settled_safe_wait_cannot_accumulate_positive_speed_reward():
    _, ego, _, route = make_scene(blocker_x=8.0)
    config = RewardConfig(
        reward_variant="line_e",
        line_e_safe_wait_reward_grace_steps=1,
        line_e_blocked_timeout_steps=1000,
    )
    calculator = RewardCalculator(config)
    _, _, first = step(calculator, ego, route)
    _, _, second = step(calculator, ego, route)
    assert first["r_speed"] > 0.0
    assert_close(second["r_speed"], 0.0)
    assert second["r_line_e_safe_wait_speed_adjustment"] < 0.0


def test_persistent_same_blocker_uses_bounded_timeout_without_stuck_penalty():
    _, ego, _, route = make_scene(blocker_x=10.0)
    config = RewardConfig(
        reward_variant="line_e",
        vehicle_stuck_step=1,
        hard_stuck_step=2,
        line_e_blocked_timeout_steps=2,
    )
    calculator = RewardCalculator(config)
    _, done_first, _ = step(calculator, ego, route)
    _, done_second, info = step(calculator, ego, route)
    assert not done_first
    assert done_second
    assert info["termination_reasons"] == ["blocked_timeout"]
    assert info["vehicle_stuck_counter"] == 0
    assert_close(info["r_hard_stuck"], 0.0)
    assert_close(info["r_stuck_terminal"], 0.0)


def test_legacy_reward_ignores_line_e_bbox_classifier():
    _, ego, _, route = make_scene(blocker_x=10.0)
    calculator = RewardCalculator(RewardConfig(reward_variant="legacy", enable_line_e_reward=False))
    reward, done, info = step(calculator, ego, route)
    assert not done
    assert not info["bbox_safety_wait_active"]
    assert not info["is_legal_wait"]
    assert info["legacy_is_free_road"]
    assert info["vehicle_stuck_counter"] == 1
    assert_close(reward, 0.0)


def test_blocked_timeout_terminal_frame_is_not_written_to_replay():
    from rl.hipad_policy_finetune_config import HiPADPolicyFinetuneConfig
    from stable_train_hipad_policy_finetune import replay_store_decision

    config = HiPADPolicyFinetuneConfig()
    store, reason = replay_store_decision(
        {"sensor_frame_exact": True},
        {
            "sensor_frame_exact": True,
            "termination_reasons": ["blocked_timeout"],
            "terminal_for_replay": False,
        },
        transition_terminal=True,
        config=config,
    )
    assert not store
    assert reason == "invalid_terminal:blocked_timeout"


def main():
    tests = (
        test_bbox_gap_catches_center_distance_mismatch,
        test_adjacent_lane_does_not_trigger_bbox_wait,
        test_blocker_departure_clears_wait_immediately,
        test_red_light_wait_and_green_resume,
        test_true_free_road_stop_still_terminates_as_vehicle_stuck,
        test_settled_safe_wait_cannot_accumulate_positive_speed_reward,
        test_persistent_same_blocker_uses_bounded_timeout_without_stuck_penalty,
        test_legacy_reward_ignores_line_e_bbox_classifier,
        test_blocked_timeout_terminal_frame_is_not_written_to_replay,
    )
    for test in tests:
        test()
    print(f"line-e safe-wait smoke passed ({len(tests)} checks)")


if __name__ == "__main__":
    main()
