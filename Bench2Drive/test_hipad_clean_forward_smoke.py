#!/usr/bin/env python3
"""Multi-frame clean model/bridge/reset smoke; no CARLA process is started."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
CLEAN_ROOT = REPO_ROOT / "HiP-AD"
IMPORT_PATHS = [PROJECT_ROOT, PROJECT_ROOT / "leaderboard", PROJECT_ROOT / "scenario_runner", REPO_ROOT]
if os.environ.get("CARLA_ROOT"):
    IMPORT_PATHS.append(Path(os.environ["CARLA_ROOT"]) / "PythonAPI" / "carla")
for path in IMPORT_PATHS:
    sys.path.insert(0, str(path))

from rl.hipad_project_runtime import activate_hipad_project_root, validate_hipad_checkpoint_asset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-role", default="clean_base", choices=("clean_base", "clean_finetuned"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("SKIPPED: HiP-AD FlashAttention requires a visible CUDA device")
        return
    checkpoint = validate_hipad_checkpoint_asset(
        args.checkpoint,
        label="smoke checkpoint",
        reject_symlink=True,
        checkpoint_role=args.checkpoint_role,
        repo_root=REPO_ROOT,
    )
    activate_hipad_project_root(CLEAN_ROOT, repo_root=REPO_ROOT)

    from rl.hipad_policy_finetune_agent import HiPADPolicyFinetuneAgent
    from rl.hipad_policy_finetune_config import HiPADPolicyFinetuneConfig

    config = HiPADPolicyFinetuneConfig(
        hipad_project_root=str(CLEAN_ROOT),
        hipad_config_path=str(CLEAN_ROOT / "local_runtime" / "hipad_b2d_stage2_clean_local.py"),
        hipad_checkpoint_path=str(checkpoint),
        adapter_mode="none",
        strict_policy=True,
    )
    agent = HiPADPolicyFinetuneAgent(config, device=torch.device("cuda"))
    observation = {
        "rgb": np.zeros((6, 900, 1600, 3), dtype=np.uint8),
        "state": np.zeros((21,), dtype=np.float32),
        "can_bus": np.zeros((18,), dtype=np.float32),
        "gps": np.zeros((3,), dtype=np.float32),
        "compass": 0.0,
        "target_point": np.array([10.0, 0.0], dtype=np.float32),
        "target_point_next": np.array([20.0, 0.0], dtype=np.float32),
        "command": 4,
        "scene_token": "clean-forward-smoke",
    }
    output = agent.forward_policy(observation, deterministic=True)
    assert output.valid
    assert tuple(output.plan_cls_context_np.shape) == (1, 48, 256)
    assert tuple(output.all_candidates_np.shape) == (1, 48, 6, 2)
    assert tuple(output.speed_trajectory_np.shape) == (6, 2)
    assert tuple(output.longitudinal_candidates_np.shape) == (48, 6, 2)
    np.testing.assert_allclose(
        output.speed_trajectory_np,
        output.longitudinal_candidates_np[output.mode_idx].astype(np.float32),
        rtol=1e-3,
        atol=1e-3,
    )
    assert 0 <= output.mode_idx < 48
    assert 0 <= output.speed_mode_idx < 3
    assert np.isfinite(output.plan_cls_context_np).all()
    assert np.isfinite(output.all_candidates_np).all()
    assert np.isfinite(output.speed_trajectory_np).all()
    assert np.isfinite(output.longitudinal_candidates_np).all()
    assert output.navigation_command == 4
    assert np.array_equal(output.target_point_np, observation["target_point"])
    assert np.array_equal(output.target_point_next_np, observation["target_point_next"])
    assert agent._onedecoder.run_step == 1

    second_output = agent.forward_policy(observation, deterministic=True)
    assert second_output.valid
    assert agent._onedecoder.run_step == 2

    agent.reset_temporal_state()
    assert agent._onedecoder.run_step == 0
    reset_output = agent.forward_policy(observation, deterministic=True)
    assert reset_output.valid
    assert agent._onedecoder.run_step == 1
    # FlashAttention/FP16 is not bitwise deterministic. Reset parity is bounded
    # by a small absolute tolerance rather than exact equality.
    context_reset_error = float(np.max(np.abs(
        reset_output.plan_cls_context_np.astype(np.float32) - output.plan_cls_context_np.astype(np.float32)
    )))
    candidate_reset_error = float(np.max(np.abs(
        reset_output.all_candidates_np.astype(np.float32) - output.all_candidates_np.astype(np.float32)
    )))
    speed_reset_error = float(np.max(np.abs(
        reset_output.speed_trajectory_np.astype(np.float32) - output.speed_trajectory_np.astype(np.float32)
    )))
    assert context_reset_error < 0.02, context_reset_error
    assert candidate_reset_error < 0.02, candidate_reset_error
    assert speed_reset_error < 0.02, speed_reset_error
    agent.reset_temporal_state()
    assert agent._onedecoder.run_step == 0
    print(
        f"context={output.plan_cls_context_np.shape} candidates={output.all_candidates_np.shape} "
        f"speed={output.speed_trajectory_np.shape} lateral_mode={output.mode_idx} "
        f"speed_area_mode={output.speed_mode_idx} temporal_steps=2 "
        f"reset_max_errors={context_reset_error:.5f}/{candidate_reset_error:.5f}/{speed_reset_error:.5f}"
    )
    print("HiP-AD multi-frame forward/reset smoke passed")


if __name__ == "__main__":
    main()
