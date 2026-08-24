"""Fail-fast helpers for HiP-AD route and control navigation.

The generic RL environment may expose navigation produced by a legacy VAD
planner.  Clean HiP-AD rollout must instead bind the current Bench2Drive route
to HiP-AD's own RoutePlanner and reuse the exact target point consumed by
the model for dual-PID control and replay.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


CLEAN_NAVIGATION_SOURCE = "hipad_clean_global_plan_v1"


def bind_clean_global_plan(env, agent) -> Dict[str, object]:
    """Bind the active route to the clean wrapper and validate the result."""

    route_scenario = getattr(env, "route_scenario", None)
    if route_scenario is None:
        raise RuntimeError("HiP-AD clean navigation requires env.route_scenario")

    global_plan_gps = getattr(route_scenario, "gps_route", None)
    global_plan_world = getattr(route_scenario, "route", None)
    if not global_plan_gps or not global_plan_world:
        raise RuntimeError("HiP-AD clean navigation requires non-empty GPS and world route plans")
    if len(global_plan_gps) != len(global_plan_world):
        raise RuntimeError(
            "HiP-AD clean route-plan length mismatch: "
            f"gps={len(global_plan_gps)}, world={len(global_plan_world)}"
        )

    set_global_plan = getattr(agent, "set_global_plan", None)
    if not callable(set_global_plan):
        raise RuntimeError("HiP-AD clean agent does not expose set_global_plan")
    set_global_plan(global_plan_gps, global_plan_world)

    input_adapter = getattr(agent, "_input_adapter", None)
    bound_gps = getattr(input_adapter, "_global_plan", None)
    bound_world = getattr(input_adapter, "_global_plan_world_coord", None)
    if not bound_gps or not bound_world:
        raise RuntimeError("HiP-AD clean wrapper rejected or lost the bound global plan")
    if len(bound_gps) != len(bound_world):
        raise RuntimeError(
            "HiP-AD clean downsampled route-plan length mismatch: "
            f"gps={len(bound_gps)}, world={len(bound_world)}"
        )

    route_name = getattr(route_scenario, "name", None)
    if route_name is None:
        route_name = getattr(getattr(route_scenario, "config", None), "name", "<unknown>")
    return {
        "source": CLEAN_NAVIGATION_SOURCE,
        "route_name": str(route_name),
        "raw_gps_points": int(len(global_plan_gps)),
        "raw_world_points": int(len(global_plan_world)),
        "downsampled_points": int(len(bound_gps)),
    }


def clean_control_target_from_policy_output(policy_output) -> np.ndarray:
    """Return the exact target point used by the current clean model forward."""

    if policy_output is None or not bool(getattr(policy_output, "valid", False)):
        error = "missing policy output" if policy_output is None else getattr(policy_output, "error", "invalid")
        raise RuntimeError(f"Cannot build clean PID control target from invalid policy output: {error}")

    target_point = getattr(policy_output, "target_point_np", None)
    if target_point is None:
        raise RuntimeError("Clean policy output is missing the model-consumed target_point")
    target_point = np.asarray(target_point, dtype=np.float32)
    if target_point.shape != (2,):
        raise RuntimeError(f"Clean policy target_point has shape {target_point.shape}, expected (2,)")
    if not np.isfinite(target_point).all():
        raise RuntimeError("Clean policy target_point contains non-finite values")
    return target_point.copy()


def clean_replay_navigation(policy_output) -> Dict[str, object]:
    """Build replay navigation fields from the clean model's actual inputs."""

    target_point = clean_control_target_from_policy_output(policy_output)
    command = getattr(policy_output, "navigation_command", None)
    if command is None:
        raise RuntimeError("Clean policy output is missing navigation_command")
    command = int(command)
    if command < 1 or command > 6:
        raise RuntimeError(f"Clean navigation command {command} is outside [1, 6]")
    return {
        "target_point": target_point,
        "command": command,
    }
