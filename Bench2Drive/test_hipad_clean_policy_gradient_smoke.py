#!/usr/bin/env python3
"""CPU one-batch gradient smoke for the migrated final HiP-AD planning branches."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

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


def _parameter_delta(module, before) -> float:
    return float(sum(
        (param.detach() - old).abs().sum().item()
        for param, old in zip(module.parameters(), before)
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-role", default="clean_base", choices=("clean_base", "clean_finetuned"))
    args = parser.parse_args()

    torch.manual_seed(17)
    activate_hipad_project_root(CLEAN_ROOT, repo_root=REPO_ROOT)
    checkpoint = validate_hipad_checkpoint_asset(
        args.checkpoint,
        label="policy-gradient smoke checkpoint",
        reject_symlink=True,
        checkpoint_role=args.checkpoint_role,
        repo_root=REPO_ROOT,
    )

    from rl.hipad_policy_finetune_agent import HiPADPolicyFinetuneAgent
    from rl.hipad_policy_finetune_config import HiPADPolicyFinetuneConfig
    from rl.adaptdrive_sac import DUAL_PID_SUMMARY_DIM

    config = HiPADPolicyFinetuneConfig(
        hipad_project_root=str(CLEAN_ROOT),
        hipad_config_path=str(CLEAN_ROOT / "local_runtime" / "hipad_b2d_stage2_clean_local.py"),
        hipad_checkpoint_path=str(checkpoint),
        hipad_checkpoint_role=args.checkpoint_role,
        reference_kl_weight=0.3,
        reference_kl_final_weight=0.1,
        reference_decay_steps=100,
        trajectory_trust_region_weight=1.0,
        detach_policy_q_candidates=False,
    )
    agent = HiPADPolicyFinetuneAgent(config, device=torch.device("cpu"))
    final_refine = agent._onedecoder.plan_refine[-1]
    first_refine = agent._onedecoder.plan_refine[0]
    cls_before = [param.detach().clone() for param in final_refine.plan_cls_branch.parameters()]
    reg_before = [param.detach().clone() for param in final_refine.plan_reg_branch_spat_2m.parameters()]
    first_before = [param.detach().clone() for param in first_refine.plan_cls_branch.parameters()]

    batch_size = 2
    batch = SimpleNamespace(
        plan_cls_context=torch.randn(batch_size, config.num_policy_modes, config.critic_plan_dim),
        critic_bev_features=torch.randn(batch_size, config.feature_dim),
        observations={"state": torch.randn(batch_size, config.state_dim)},
        all_candidates=torch.randn(batch_size, config.num_policy_modes, config.fut_ts, 2),
        candidate_longitudinal_trajectories=torch.randn(
            batch_size,
            config.num_policy_modes,
            config.fut_ts,
            2,
        ),
        prev_pid_summaries=torch.randn(batch_size, DUAL_PID_SUMMARY_DIM),
        prev_pid_summary_masks=torch.ones(batch_size),
    )
    metrics = agent.update_policy_from_feature_batch(batch, total_step=50)

    cls_delta = _parameter_delta(final_refine.plan_cls_branch, cls_before)
    reg_delta = _parameter_delta(final_refine.plan_reg_branch_spat_2m, reg_before)
    first_delta = _parameter_delta(first_refine.plan_cls_branch, first_before)
    assert cls_delta > 0.0, "final plan_cls_branch did not update"
    assert reg_delta > 0.0, "final plan_reg_branch_spat_2m did not receive candidate Q gradients"
    assert first_delta == 0.0, "an earlier refine classification branch changed"
    assert metrics["policy_q_candidate_grad"] == 1.0
    assert metrics["policy_q_candidate_grad_enabled"] == 1.0
    assert metrics["policy_q_candidate_grad_norm"] > 0.0
    assert metrics["policy_plan_cls_branch_grad_norm"] > 0.0
    assert metrics["policy_plan_spat_reg_branch_grad_norm"] > 0.0
    assert abs(metrics["reference_kl_weight"] - 0.2) < 1e-6
    print(
        f"final_cls_delta={cls_delta:.6g} final_spat_reg_delta={reg_delta:.6g} "
        f"earlier_cls_delta={first_delta:.6g} kl_weight={metrics['reference_kl_weight']:.3f} "
        f"q_candidate_grad_norm={metrics['policy_q_candidate_grad_norm']:.6g} "
        f"cls_grad_norm={metrics['policy_plan_cls_branch_grad_norm']:.6g} "
        f"spat_grad_norm={metrics['policy_plan_spat_reg_branch_grad_norm']:.6g}"
    )
    print("HiP-AD final planning-branch one-batch gradient smoke passed")


if __name__ == "__main__":
    main()
