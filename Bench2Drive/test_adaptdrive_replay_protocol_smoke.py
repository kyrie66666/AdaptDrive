#!/usr/bin/env python3
"""Offline save/resume/corruption smoke for the AdaptDrive replay protocol."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np


BENCH2DRIVE_ROOT = Path(__file__).resolve().parent
LEADERBOARD_ROOT = BENCH2DRIVE_ROOT / "leaderboard"
for path in (BENCH2DRIVE_ROOT, LEADERBOARD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rl.adaptdrive_replay import (  # noqa: E402
    create_feature_replay,
    resume_feature_replay,
    write_replay_state_snapshot,
)
from rl.adaptdrive_training_signature import TRAINING_SIGNATURE_VERSION  # noqa: E402
from rl.replay import CLEAN_DUAL_TRAJECTORY_CONTROL  # noqa: E402


EXPERIMENT_ID = "replay_protocol_smoke"
SIGNATURE = "a" * 64
CAPACITY = 4
SHAPES = {
    "state_shape": (3,),
    "actor_base_shape": (5,),
    "critic_bev_shape": (5,),
    "trajectory_shape": (2, 2),
    "pid_summary_dim": 4,
    "control_semantics": CLEAN_DUAL_TRAJECTORY_CONTROL,
}


def _add_transition(replay, index: int) -> None:
    scalar = float(index + 1)
    state = np.full((3,), scalar, dtype=np.float32)
    next_state = state + 0.5
    trajectory = np.full((2, 2), scalar, dtype=np.float32)
    candidates = np.full((48, 2, 2), scalar, dtype=np.float32)
    replay.add(
        {"state": state, "target_point": np.array([scalar, -scalar], dtype=np.float32), "command": 3},
        actor_base_features=np.full((5,), scalar, dtype=np.float32),
        critic_bev_features=np.full((5,), scalar + 0.1, dtype=np.float32),
        trajectory=trajectory,
        pid_summary=np.full((4,), scalar, dtype=np.float32),
        reward=scalar,
        next_observation={
            "state": next_state,
            "target_point": np.array([scalar + 0.5, -scalar], dtype=np.float32),
            "command": 4,
        },
        next_critic_bev_features=np.full((5,), scalar + 0.2, dtype=np.float32),
        done=index == 2,
        prev_pid_summary=np.full((4,), scalar - 0.5, dtype=np.float32),
        plan_cls_context=np.full((48, 256), scalar, dtype=np.float32),
        all_candidates=candidates,
        reference_logits=np.linspace(-1.0, 1.0, 48, dtype=np.float32),
        longitudinal_trajectory=trajectory + 0.25,
        candidate_longitudinal_trajectories=candidates + 0.25,
        selected_lateral_mode=index,
        longitudinal_mode=index + 1,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="adaptdrive_replay_protocol_") as temp_dir:
        replay_root = Path(temp_dir) / "replay"
        replay, context = create_feature_replay(
            replay_root=str(replay_root),
            experiment_id=EXPERIMENT_ID,
            training_signature=SIGNATURE,
            capacity=CAPACITY,
            **SHAPES,
        )
        for index in range(3):
            _add_transition(replay, index)
        replay_ref = write_replay_state_snapshot(replay, context)
        checkpoint = {
            "checkpoint_version": 2,
            "checkpoint_uuid": replay_ref["checkpoint_uuid"],
            "experiment_id": EXPERIMENT_ID,
            "training_signature_version": TRAINING_SIGNATURE_VERSION,
            "training_signature": SIGNATURE,
            "replay_ref": replay_ref,
        }
        replay.close()

        resumed, resumed_context = resume_feature_replay(
            checkpoint,
            replay_root=str(replay_root),
            experiment_id=EXPERIMENT_ID,
            training_signature=SIGNATURE,
            capacity=CAPACITY,
            **SHAPES,
        )
        assert resumed_context.replay_uuid == context.replay_uuid
        assert len(resumed) == 3
        assert resumed.ptr == 3
        assert np.allclose(resumed.states[1], np.full((3,), 2.0, dtype=np.float32))
        resumed.close()

        states_path = context.payload_dir / "states.dat"
        with states_path.open("r+b") as stream:
            original = stream.read(1)
            stream.seek(0)
            stream.write(bytes([original[0] ^ 0x01]))
            stream.flush()
        try:
            resume_feature_replay(
                checkpoint,
                replay_root=str(replay_root),
                experiment_id=EXPERIMENT_ID,
                training_signature=SIGNATURE,
                capacity=CAPACITY,
                **SHAPES,
            )
        except RuntimeError as exc:
            assert "slot hash mismatch" in str(exc)
        else:
            raise AssertionError("same-size replay payload corruption was accepted")

    print("adaptdrive_replay_protocol_smoke: PASS uuid_manifest=1 corruption_rejected=1", flush=True)


if __name__ == "__main__":
    main()
