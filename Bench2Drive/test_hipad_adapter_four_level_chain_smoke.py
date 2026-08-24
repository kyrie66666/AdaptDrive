#!/usr/bin/env python3
"""Structure-only gate for the HiP-AD four-level feature-adapter chain.

This smoke does not launch CARLA, create an env, or run training.  It builds the
real HiPADPolicyFinetuneAgent and checks trainable parameter ownership,
four-level feature-adapter state, and optimizer separation.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from typing import Iterable, Set

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
CLEAN_ROOT = REPO_ROOT / "HiP-AD"
CARLA_ROOT = os.environ.get("CARLA_ROOT", "")

for path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "leaderboard",
    PROJECT_ROOT / "scenario_runner",
    REPO_ROOT,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
if CARLA_ROOT:
    carla_root = Path(CARLA_ROOT)
    carla_egg = carla_root / "PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
    for path in (carla_egg, carla_root / "PythonAPI", carla_root / "PythonAPI/carla"):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

from rl.hipad_policy_finetune_config import HiPADPolicyFinetuneConfig  # noqa: E402
from rl.hipad_project_runtime import activate_hipad_project_root  # noqa: E402


FOUR_LEVELS = (0, 1, 2, 3)


def _param_ids(params: Iterable[torch.nn.Parameter]) -> Set[int]:
    return {id(param) for param in params}


def _optimizer_param_ids(optimizer: torch.optim.Optimizer) -> Set[int]:
    ids: Set[int] = set()
    for group in optimizer.param_groups:
        ids.update(id(param) for param in group["params"])
    return ids


def _module_param_ids(module: torch.nn.Module) -> Set[int]:
    return _param_ids(module.parameters())


def _assert_disjoint(left: Set[int], right: Set[int], message: str) -> None:
    overlap = left & right
    if overlap:
        raise AssertionError(f"{message}: {len(overlap)} overlapping parameter objects")


def _make_config(checkpoint: Path) -> HiPADPolicyFinetuneConfig:
    return HiPADPolicyFinetuneConfig(
        hipad_project_root=str(CLEAN_ROOT),
        hipad_config_path=str(CLEAN_ROOT / "local_runtime" / "hipad_b2d_stage2_clean_local.py"),
        hipad_checkpoint_path=str(checkpoint),
        hipad_checkpoint_role="clean_base",
        adapter_mode="dcnv4_feature",
        enable_feature_dcnv4_adapter=True,
        enable_ego_state_adapter=False,
        feature_adapter_levels=FOUR_LEVELS,
        adapter_prediction_enabled=True,
        adapter_prediction_train_reward=True,
        adapter_prediction_train_semantic=True,
        adapter_prediction_update_mode="prediction_only",
    )


def _prepare_dcnv4_for_structure_probe(require_real: bool) -> str:
    try:
        import DCNv4  # noqa: F401
    except Exception:
        if require_real:
            raise
        import torch.nn as nn
        import rl.ego_state_adapter as ego_state_adapter

        class ParameterCompatibleDCNv4(nn.Module):
            """CPU structure double with the exact default DCNv4 parameter schema."""

            def __init__(
                self,
                channels=64,
                kernel_size=3,
                group=4,
                remove_center=False,
                output_bias=True,
                without_pointwise=False,
                **kwargs,
            ) -> None:
                super().__init__()
                del kwargs
                points = int(group) * (int(kernel_size) ** 2 - int(remove_center))
                offset_dim = int(math.ceil((points * 3) / 8) * 8)
                self.offset_mask = nn.Linear(int(channels), offset_dim)
                if not without_pointwise:
                    self.value_proj = nn.Linear(int(channels), int(channels))
                    self.output_proj = nn.Linear(int(channels), int(channels), bias=bool(output_bias))

            def forward(self, tokens, shape=None):
                del shape
                return tokens

        ego_state_adapter._load_dcnv4_class = lambda: ParameterCompatibleDCNv4
        return "fake_parameter_compatible_structure_only"
    return "real"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "HiP-AD" / "work_dirs" / "hipad_b2d_stage2" / "latest.pth"),
    )
    parser.add_argument(
        "--require-real-dcnv4",
        action="store_true",
        help="Fail if the DCNv4 package cannot be imported. By default this structure smoke falls back to an identity DCNv4 block.",
    )
    parser.add_argument(
        "--init-from",
        default="",
        help="Optionally apply the registered legacy parent and verify fresh optimizer state.",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"HiP-AD checkpoint does not exist: {checkpoint}")

    activate_hipad_project_root(CLEAN_ROOT, repo_root=REPO_ROOT)
    from projects.mmdet3d_plugin.ops import feature_maps_format

    del feature_maps_format
    dcnv4_mode = _prepare_dcnv4_for_structure_probe(require_real=bool(args.require_real_dcnv4))
    from rl.hipad_policy_finetune_agent import HiPADPolicyFinetuneAgent

    agent = HiPADPolicyFinetuneAgent(_make_config(checkpoint), device=torch.device("cpu"))

    initialization_mode = "fresh_base"
    initialization = None
    if args.init_from:
        from rl.adaptdrive_init import (
            apply_registered_legacy_parent,
            import_registered_legacy_parent,
        )

        test_signature = "f" * 64
        initialization = import_registered_legacy_parent(
            args.init_from,
            base_checkpoint_path=str(checkpoint),
            route_path=str(PROJECT_ROOT / "leaderboard/data/bench2drive220.xml"),
            hipad_root=str(CLEAN_ROOT),
            target_training_signature=test_signature,
        )
        apply_registered_legacy_parent(
            agent,
            initialization,
            current_training_signature=test_signature,
        )
        assert initialization.provenance()["new_step"] == 0
        assert initialization.provenance()["new_episode"] == 0
        initialization_mode = "registered_v7_weights"

    assert agent.adapter_mode == "dcnv4_feature"
    assert agent.feature_dcnv4_adapter_enabled
    assert agent.adapter_prediction_enabled
    assert tuple(agent.config.feature_adapter_levels) == FOUR_LEVELS
    assert agent._feature_dcnv4_adapter is not None
    assert tuple(agent._feature_dcnv4_adapter.levels) == FOUR_LEVELS

    adapter_state = agent._feature_dcnv4_adapter.state_dict()
    for level in FOUR_LEVELS:
        alpha_key = f"residual_alpha_by_level.{level}"
        assert alpha_key in adapter_state, f"missing {alpha_key}"
        alpha = agent._feature_dcnv4_adapter.residual_alpha_by_level[str(level)]
        assert alpha.requires_grad, f"L{level} alpha should train through adapter prediction"
        expected_alpha = (
            initialization.imported_agent["feature_dcnv4_adapter"][alpha_key]
            if initialization is not None
            else torch.ones(())
        )
        assert torch.allclose(alpha.detach().cpu(), expected_alpha), f"L{level} alpha state mismatch"

    trainable_names = list(agent._trainable_param_names)
    assert trainable_names, "HiP-AD final planning heads should be trainable"
    allowed_trainable = ("plan_refine.5.plan_cls_branch", "plan_refine.5.plan_reg_branch_spat_2m")
    assert all(any(token in name for token in allowed_trainable) for name in trainable_names), trainable_names[:8]
    assert not any("plan_cls_branch_speed" in name for name in trainable_names), trainable_names
    assert not any("plan_refine.0" in name for name in trainable_names), trainable_names[:8]

    model_requires_grad_names = [
        name for name, param in agent._model.named_parameters() if param.requires_grad
    ]
    assert model_requires_grad_names == trainable_names

    policy_ids = _optimizer_param_ids(agent.policy_optimizer)
    critic_ids = _optimizer_param_ids(agent.critic_optimizer)
    vf_ids = _optimizer_param_ids(agent.vf_optimizer)
    alpha_ids = _optimizer_param_ids(agent.alpha_optimizer) if agent.alpha_optimizer is not None else set()
    sac_ids = set().union(policy_ids, critic_ids, vf_ids, alpha_ids)

    adapter_ids = _module_param_ids(agent._feature_dcnv4_adapter)
    reward_head_ids = _module_param_ids(agent._adapter_prediction_reward_head)
    semantic_head_ids = _module_param_ids(agent._adapter_prediction_semantic_head)
    prediction_ids = _optimizer_param_ids(agent._adapter_prediction_optimizer)

    assert adapter_ids <= prediction_ids, "adapter params must be owned by adapter prediction optimizer"
    assert reward_head_ids <= prediction_ids, "reward head params must be owned by adapter prediction optimizer"
    assert semantic_head_ids <= prediction_ids, "semantic head params must be owned by adapter prediction optimizer"
    _assert_disjoint(adapter_ids, policy_ids, "feature adapter leaked into policy optimizer")
    _assert_disjoint(reward_head_ids | semantic_head_ids, sac_ids, "prediction heads leaked into SAC optimizers")
    _assert_disjoint(prediction_ids, sac_ids, "adapter prediction optimizer overlaps SAC optimizers")

    pred_groups = agent._adapter_prediction_optimizer.param_groups
    assert len(pred_groups) == 3, f"expected adapter/reward/semantic param groups, got {len(pred_groups)}"
    assert _param_ids(pred_groups[0]["params"]) == adapter_ids
    assert _param_ids(pred_groups[1]["params"]) == reward_head_ids
    assert _param_ids(pred_groups[2]["params"]) == semantic_head_ids
    for optimizer in (
        agent.policy_optimizer,
        agent.critic_optimizer,
        agent.vf_optimizer,
        agent.alpha_optimizer,
        agent._adapter_prediction_optimizer,
    ):
        if optimizer is not None:
            assert not optimizer.state, "structure/init smoke must preserve fresh optimizer state"

    state = agent.state_dict()
    assert tuple(state["feature_adapter_levels"]) == FOUR_LEVELS
    assert "feature_adapter_aux" not in state
    assert "adapter_prediction" in state
    for level in FOUR_LEVELS:
        assert f"residual_alpha_by_level.{level}" in state["feature_dcnv4_adapter"]
    agent.load_state_dict(state, load_optimizers=True, strict=True)

    broken_state = dict(state)
    broken_critic = dict(state["critic"])
    broken_critic.pop(next(iter(broken_critic)))
    broken_state["critic"] = broken_critic
    try:
        agent.load_state_dict(broken_state, load_optimizers=True, strict=True)
    except RuntimeError as exc:
        assert "critic strict state mismatch" in str(exc)
    else:
        raise AssertionError("strict full-state restore accepted an incomplete critic")

    print(
        "hipad_adapter_four_level_chain_smoke: PASS",
        f"trainable_params={agent.trainable_parameter_count}",
        f"feature_adapter_params={agent.feature_dcnv4_adapter_parameter_count}",
        f"prediction_param_groups={len(pred_groups)}",
        f"dcnv4_mode={dcnv4_mode}",
        f"initialization={initialization_mode}",
        f"levels={FOUR_LEVELS}",
        flush=True,
    )


if __name__ == "__main__":
    main()
