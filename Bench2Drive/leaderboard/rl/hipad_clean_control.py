"""Shared clean dual-trajectory PID construction and replay metadata."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from rl.adaptdrive_sac import DUAL_PID_SUMMARY_DIM


CLEAN_DUAL_PID_KWARGS = {
    "turn_KP": 1.0,
    "turn_KI": 0.75,
    "turn_KD": 0.0,
    "turn_n": 10,
    "speed_KP": 5.0,
    "speed_KI": 0.5,
    "speed_KD": 1.0,
    "speed_n": 10,
    "waypoint_time": 0.2,
}


def create_clean_dual_pid_controller():
    from bench2drive.leaderboard.team_code.pid_controller import PIDController

    return PIDController(**CLEAN_DUAL_PID_KWARGS)


def clean_dual_pid_step(
    pid_controller,
    longitudinal_trajectory: np.ndarray,
    lateral_trajectory: np.ndarray,
    ego_speed,
    target_point: np.ndarray,
) -> Tuple[np.ndarray, Dict]:
    """Run clean PID and convert its output to Bench2Drive's 2-D action."""

    steer, throttle, brake, metadata = pid_controller.control_pid(
        np.asarray(longitudinal_trajectory, dtype=np.float32),
        np.asarray(lateral_trajectory, dtype=np.float32),
        np.asarray(ego_speed, dtype=np.float32),
        np.asarray(target_point, dtype=np.float32),
    )
    if brake < 0.05:
        brake = 0.0
    if throttle > brake:
        brake = 0.0
    action = np.array([steer, -float(brake) if brake else throttle], dtype=np.float32)
    return action, metadata


def pack_clean_dual_pid_summary(action: np.ndarray, pid_metadata: Dict, pid_controller) -> np.ndarray:
    """Pack control outputs, intermediates, and both 10-step PID windows."""

    action = np.asarray(action, dtype=np.float32)
    aim = np.asarray(pid_metadata.get("aim", (0.0, 0.0)), dtype=np.float32).reshape(-1)
    base = [
        float(action[0]),
        float(action[1]),
        float(pid_metadata.get("speed", 0.0)),
        float(pid_metadata.get("desired_speed", 0.0)),
        float(aim[0]) if aim.size > 0 else 0.0,
        float(aim[1]) if aim.size > 1 else 0.0,
        float(pid_metadata.get("angle", 0.0)),
        float(pid_metadata.get("angle_last", 0.0)),
        float(pid_metadata.get("angle_target", 0.0)),
        float(pid_metadata.get("angle_final", 0.0)),
        float(pid_metadata.get("delta", 0.0)),
        float(pid_metadata.get("brake", 0.0)),
    ]
    turn_window = list(pid_controller.turn_controller._window)
    speed_window = list(pid_controller.speed_controller._window)
    summary = np.asarray(base + turn_window + speed_window, dtype=np.float32)
    if summary.shape != (DUAL_PID_SUMMARY_DIM,):
        raise RuntimeError(
            f"dual PID summary has shape {summary.shape}, expected {(DUAL_PID_SUMMARY_DIM,)}"
        )
    if not np.isfinite(summary).all():
        raise RuntimeError("dual PID summary contains non-finite values")
    return summary
