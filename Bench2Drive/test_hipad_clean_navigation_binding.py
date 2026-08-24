#!/usr/bin/env python3
"""Fast clean navigation binding/control-source smoke; no CARLA process."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "leaderboard"))

from rl.hipad_clean_navigation import (
    CLEAN_NAVIGATION_SOURCE,
    bind_clean_global_plan,
    clean_control_target_from_policy_output,
    clean_replay_navigation,
)


class FakeAgent:
    def __init__(self):
        self._input_adapter = SimpleNamespace(_global_plan=None, _global_plan_world_coord=None)
        self.received = None

    def set_global_plan(self, global_plan_gps, global_plan_world):
        self.received = (global_plan_gps, global_plan_world)
        # Simulate the clean wrapper's route downsampling while preserving
        # matching GPS/world lengths.
        self._input_adapter._global_plan = list(global_plan_gps[::2])
        self._input_adapter._global_plan_world_coord = list(global_plan_world[::2])


def expect_runtime_error(fn, expected_text: str) -> None:
    try:
        fn()
    except RuntimeError as exc:
        assert expected_text in str(exc), str(exc)
    else:
        raise AssertionError(f"Expected RuntimeError containing {expected_text!r}")


def main() -> None:
    gps_route = [({"lat": float(i), "lon": float(i)}, i % 6 + 1) for i in range(6)]
    world_route = [(f"transform-{i}", i % 6 + 1) for i in range(6)]
    env = SimpleNamespace(
        route_scenario=SimpleNamespace(
            name="RouteScenario_navigation_smoke",
            gps_route=gps_route,
            route=world_route,
        )
    )
    agent = FakeAgent()
    binding = bind_clean_global_plan(env, agent)
    assert agent.received == (gps_route, world_route)
    assert binding == {
        "source": CLEAN_NAVIGATION_SOURCE,
        "route_name": "RouteScenario_navigation_smoke",
        "raw_gps_points": 6,
        "raw_world_points": 6,
        "downsampled_points": 3,
    }

    output = SimpleNamespace(
        valid=True,
        error="",
        navigation_command=2,
        target_point_np=np.array([7.5, -1.25], dtype=np.float32),
    )
    target = clean_control_target_from_policy_output(output)
    assert np.array_equal(target, np.array([7.5, -1.25], dtype=np.float32))
    replay_navigation = clean_replay_navigation(output)
    assert replay_navigation["command"] == 2
    assert np.array_equal(replay_navigation["target_point"], target)

    expect_runtime_error(
        lambda: bind_clean_global_plan(SimpleNamespace(route_scenario=None), FakeAgent()),
        "env.route_scenario",
    )
    mismatch_env = SimpleNamespace(
        route_scenario=SimpleNamespace(gps_route=gps_route, route=world_route[:-1])
    )
    expect_runtime_error(
        lambda: bind_clean_global_plan(mismatch_env, FakeAgent()),
        "length mismatch",
    )
    expect_runtime_error(
        lambda: clean_control_target_from_policy_output(
            SimpleNamespace(valid=True, target_point_np=np.zeros(3, dtype=np.float32))
        ),
        "expected (2,)",
    )
    expect_runtime_error(
        lambda: clean_replay_navigation(
            SimpleNamespace(
                valid=True,
                target_point_np=np.zeros(2, dtype=np.float32),
                navigation_command=0,
            )
        ),
        "outside [1, 6]",
    )
    print("HiP-AD global-plan/navigation-source smoke passed")


if __name__ == "__main__":
    main()
