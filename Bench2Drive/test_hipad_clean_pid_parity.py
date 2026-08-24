#!/usr/bin/env python3
"""Sequence parity test against HiP-AD's native dual PID class."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "leaderboard"))

from rl.hipad_project_runtime import activate_hipad_project_root

activate_hipad_project_root(REPO_ROOT / "HiP-AD", repo_root=REPO_ROOT)

from bench2drive.leaderboard.team_code.pid_controller import PIDController
from rl.hipad_clean_control import (
    CLEAN_DUAL_PID_KWARGS,
    clean_dual_pid_step,
    create_clean_dual_pid_controller,
    pack_clean_dual_pid_summary,
)


def _direct_step(controller, longitudinal, lateral, speed, target):
    steer, throttle, brake, metadata = controller.control_pid(longitudinal, lateral, speed, target)
    if brake < 0.05:
        brake = 0.0
    if throttle > brake:
        brake = 0.0
    return np.array([steer, -float(brake) if brake else throttle], dtype=np.float32), metadata


def main() -> None:
    native = PIDController(**CLEAN_DUAL_PID_KWARGS)
    migrated = create_clean_dual_pid_controller()
    base_long = np.array(
        [[0.2, 0.0], [0.7, 0.0], [1.4, 0.05], [2.3, 0.1], [3.3, 0.2], [4.4, 0.3]],
        dtype=np.float32,
    )
    base_lat = np.array(
        [[0.2, 0.0], [0.8, 0.1], [1.6, 0.25], [2.5, 0.5], [3.5, 0.9], [4.6, 1.4]],
        dtype=np.float32,
    )

    metadata_fields = (
        "speed", "steer", "throttle", "brake", "aim", "target", "desired_speed",
        "angle", "angle_last", "angle_target", "angle_final", "delta",
    )
    for index, speed_value in enumerate((0.0, 0.4, 1.2, 2.0, 1.6)):
        longitudinal = base_long * (1.0 + 0.03 * index)
        lateral = base_lat.copy()
        lateral[:, 1] *= 1.0 - 0.08 * index
        speed = np.asarray(speed_value, dtype=np.float32)
        target = np.array([3.0, 0.25 * index], dtype=np.float32)

        expected_action, expected_metadata = _direct_step(native, longitudinal, lateral, speed, target)
        actual_action, actual_metadata = clean_dual_pid_step(
            migrated, longitudinal, lateral, speed, target,
        )
        np.testing.assert_allclose(actual_action, expected_action, rtol=0.0, atol=1e-7)
        for field in metadata_fields:
            np.testing.assert_allclose(actual_metadata[field], expected_metadata[field], rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(
            list(migrated.turn_controller._window),
            list(native.turn_controller._window),
            rtol=0.0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            list(migrated.speed_controller._window),
            list(native.speed_controller._window),
            rtol=0.0,
            atol=1e-7,
        )
        summary = pack_clean_dual_pid_summary(actual_action, actual_metadata, migrated)
        assert summary.shape == (32,)
        assert np.isfinite(summary).all()

    print("HiP-AD dual PID sequence parity passed")


if __name__ == "__main__":
    main()
