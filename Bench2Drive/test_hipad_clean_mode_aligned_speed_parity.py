#!/usr/bin/env python3
"""Real-checkpoint parity between stock and mode-aligned clean speed decode."""

from __future__ import annotations

import argparse
import os
import sys
import time
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
    parser.add_argument("--benchmark-iters", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("SKIPPED: HiP-AD FlashAttention requires a visible CUDA device")
        return

    checkpoint = validate_hipad_checkpoint_asset(
        args.checkpoint,
        label="mode-aligned parity checkpoint",
        reject_symlink=True,
        checkpoint_role="clean_base",
        repo_root=REPO_ROOT,
    )
    activate_hipad_project_root(CLEAN_ROOT, repo_root=REPO_ROOT)
    from rl.hipad_clean_speed_decode import decode_mode_aligned_clean_speed
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
        "scene_token": "mode-aligned-speed-parity",
    }

    with torch.no_grad():
        inputs, _, model_outs = agent._raw_forward(observation)
        logits, _ = agent._policy_tensors_from_model_outs(inputs, model_outs)
        greedy_mode = int(logits[0].argmax().item())
        aligned = decode_mode_aligned_clean_speed(
            agent._onedecoder.plan_decoder,
            inputs,
            model_outs,
            num_lateral_modes=config.num_policy_modes,
            fut_ts=config.fut_ts,
        )
        det_output, _, ego_output, plan_output, motion_output, _ = model_outs
        stock = agent._onedecoder.plan_decoder.decode(
            ego_output,
            det_output,
            motion_output,
            plan_output,
            inputs,
        )[0]["plan_speed_5hz"].to(aligned.trajectories.device).float()

        benchmark_iters = max(1, int(args.benchmark_iters))
        torch.cuda.synchronize()
        stock_start = time.perf_counter()
        for _ in range(benchmark_iters):
            agent._onedecoder.plan_decoder.decode(
                ego_output,
                det_output,
                motion_output,
                plan_output,
                inputs,
            )
        torch.cuda.synchronize()
        stock_decoder_ms = (time.perf_counter() - stock_start) * 1000.0 / benchmark_iters

        torch.cuda.reset_peak_memory_stats()
        baseline_bytes = torch.cuda.memory_allocated()
        aligned_start = time.perf_counter()
        for _ in range(benchmark_iters):
            aligned_benchmark = decode_mode_aligned_clean_speed(
                agent._onedecoder.plan_decoder,
                inputs,
                model_outs,
                num_lateral_modes=config.num_policy_modes,
                fut_ts=config.fut_ts,
            )
        torch.cuda.synchronize()
        mode_aligned_ms = (time.perf_counter() - aligned_start) * 1000.0 / benchmark_iters
        mode_aligned_peak_mib = (
            torch.cuda.max_memory_allocated() - baseline_bytes
        ) / (1024.0 ** 2)
        assert torch.isfinite(aligned_benchmark.trajectories).all()

    parity_error = float((aligned.trajectories[0, greedy_mode] - stock).abs().max().item())
    assert parity_error <= 1e-6, parity_error
    assert torch.isfinite(aligned.trajectories).all()
    assert aligned.speed_area_indices.shape == (1, config.num_policy_modes)
    assert aligned.all_collision.shape == (1, config.num_policy_modes)
    print(
        f"greedy_mode={greedy_mode} parity_max_abs_error={parity_error:.8g} "
        f"all_collision_rate={float(aligned.all_collision.float().mean().item()):.6f} "
        f"stock_decoder_ms={stock_decoder_ms:.3f} mode_aligned_ms={mode_aligned_ms:.3f} "
        f"mode_aligned_peak_increment_mib={mode_aligned_peak_mib:.3f}"
    )
    print("HiP-AD mode-aligned speed real-checkpoint parity passed")


if __name__ == "__main__":
    main()
