#!/usr/bin/env python3
"""Build HiP-AD on CPU and verify full-checkpoint parameter coverage."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
CLEAN_ROOT = REPO_ROOT / "HiP-AD"
CARLA_ROOT = os.environ.get("CARLA_ROOT", "")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "leaderboard"))
sys.path.insert(0, str(PROJECT_ROOT / "scenario_runner"))
sys.path.insert(0, str(REPO_ROOT))
if CARLA_ROOT:
    carla_root = Path(CARLA_ROOT)
    carla_egg = carla_root / "PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
    for path in (carla_egg, carla_root / "PythonAPI", carla_root / "PythonAPI/carla"):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

from rl.hipad_project_runtime import (
    activate_hipad_project_root,
    collect_hipad_provenance,
    validate_hipad_checkpoint_asset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-role", default="clean_base", choices=("clean_base", "clean_finetuned"))
    args = parser.parse_args()

    activate_hipad_project_root(CLEAN_ROOT, repo_root=REPO_ROOT)
    checkpoint = validate_hipad_checkpoint_asset(
        args.checkpoint,
        label="smoke checkpoint",
        reject_symlink=True,
        checkpoint_role=args.checkpoint_role,
        repo_root=REPO_ROOT,
    )
    from rl.hipad_policy_finetune_agent import HiPADPolicyFinetuneAgent
    from rl.hipad_policy_finetune_config import HiPADPolicyFinetuneConfig

    config = HiPADPolicyFinetuneConfig(
        hipad_project_root=str(CLEAN_ROOT),
        hipad_config_path=str(CLEAN_ROOT / "local_runtime" / "hipad_b2d_stage2_clean_local.py"),
        hipad_checkpoint_path=str(checkpoint),
        hipad_checkpoint_role=args.checkpoint_role,
    )
    agent = HiPADPolicyFinetuneAgent(config, device=torch.device("cpu"))
    provenance = collect_hipad_provenance(CLEAN_ROOT)
    coverage = agent._input_adapter.runtime_asset_provenance["checkpoint.parameter_coverage"]
    assert coverage.split("/")[0] == coverage.split("/")[1]
    assert provenance["SparseOneDecoder"].startswith(str(CLEAN_ROOT))
    trainable_names = agent._trainable_param_names
    assert trainable_names
    assert all(
        "plan_refine.5.plan_cls_branch" in name or "plan_refine.5.plan_reg_branch_spat_2m" in name
        for name in trainable_names
    )
    assert not any("plan_refine.0.plan_cls_branch" in name for name in trainable_names)
    eval_state = agent.state_dict()
    agent.load_policy_state_dict_for_eval(eval_state)
    early_name, early_param = next(
        (name, param)
        for name, param in agent._model.named_parameters()
        if "plan_refine.0.plan_cls_branch" in name
    )
    eval_state["hipad_trainable"][early_name] = early_param.detach().cpu().clone()
    try:
        agent.load_policy_state_dict_for_eval(eval_state)
    except RuntimeError as exc:
        assert "unexpected" in str(exc)
    else:
        raise AssertionError("evaluation accepted a frozen early-refine branch from a legacy checkpoint")
    print(coverage)
    print(f"trainable_scope=final_refine_only parameters={agent.trainable_parameter_count}")
    print("HiP-AD model load smoke passed")


if __name__ == "__main__":
    main()
