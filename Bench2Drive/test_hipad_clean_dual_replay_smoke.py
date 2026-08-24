#!/usr/bin/env python3
"""Fast CPU smoke for the clean dual-trajectory replay schema."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "leaderboard"))

from rl.adaptdrive_sac import DUAL_PID_SUMMARY_DIM
from rl.replay import (
    CLEAN_DUAL_TRAJECTORY_CONTROL,
    CLEAN_DUAL_TRAJECTORY_SCHEMA_VERSION,
    FeatureReplayBuffer,
)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="hipad_clean_dual_replay_") as tmpdir:
        replay = FeatureReplayBuffer(
            capacity=4,
            state_shape=(21,),
            actor_base_shape=(8,),
            critic_bev_shape=(8,),
            trajectory_shape=(6, 2),
            mmap_dir=tmpdir,
            pid_summary_dim=DUAL_PID_SUMMARY_DIM,
            control_semantics=CLEAN_DUAL_TRAJECTORY_CONTROL,
        )
        observation = {
            "state": np.arange(21, dtype=np.float32),
            "target_point": np.array([1.0, 2.0], dtype=np.float32),
            "command": 3,
        }
        lateral = np.arange(12, dtype=np.float32).reshape(6, 2)
        longitudinal = lateral + 100.0
        candidate_longitudinal = np.stack(
            [longitudinal + float(mode) for mode in range(48)],
            axis=0,
        ).astype(np.float16)
        replay.add(
            observation,
            np.ones(8, dtype=np.float32),
            np.ones(8, dtype=np.float32) * 2,
            lateral,
            np.arange(DUAL_PID_SUMMARY_DIM, dtype=np.float32),
            1.25,
            observation,
            np.ones(8, dtype=np.float32) * 3,
            False,
            plan_cls_context=np.ones((48, 256), dtype=np.float16),
            all_candidates=np.ones((48, 6, 2), dtype=np.float16),
            longitudinal_trajectory=longitudinal,
            candidate_longitudinal_trajectories=candidate_longitudinal,
            selected_lateral_mode=7,
            longitudinal_mode=2,
        )
        batch = replay.sample(1, device="cpu")
        assert batch.trajectories.shape == (1, 6, 2)
        assert batch.longitudinal_trajectories.shape == (1, 6, 2)
        assert batch.candidate_longitudinal_trajectories.shape == (1, 48, 6, 2)
        np.testing.assert_allclose(
            batch.candidate_longitudinal_trajectories[0, 7].numpy(),
            candidate_longitudinal[7].astype(np.float32),
        )
        assert batch.pid_summaries.shape == (1, DUAL_PID_SUMMARY_DIM)
        assert int(batch.selected_lateral_modes[0]) == 7
        assert int(batch.longitudinal_modes[0]) == 2
        assert replay.state_dict()["schema_version"] == CLEAN_DUAL_TRAJECTORY_SCHEMA_VERSION
        assert replay.state_dict()["control_semantics"] == CLEAN_DUAL_TRAJECTORY_CONTROL
        assert replay.state_dict()["longitudinal_source"].endswith("per_lateral_mode")

        legacy_state = dict(replay.state_dict())
        legacy_state["schema_version"] = 4
        legacy_state["control_semantics"] = "single_trajectory_legacy"
        assert not replay.load_state_dict(legacy_state)
        replay.close()
    print("HiP-AD dual replay smoke passed")


if __name__ == "__main__":
    main()
