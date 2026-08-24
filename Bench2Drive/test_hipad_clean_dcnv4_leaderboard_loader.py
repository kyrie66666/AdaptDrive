#!/usr/bin/env python3
"""Offline gate for the adapter-aware HiP-AD leaderboard checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEADERBOARD_ROOT = PROJECT_ROOT / "Bench2Drive" / "leaderboard"
for path in (PROJECT_ROOT / "Bench2Drive", LEADERBOARD_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rl.hipad_clean_adapter_checkpoint import (  # noqa: E402
    FOUR_LEVELS,
    load_adapter_checkpoint,
)


DEFAULT_BASE = Path(
    os.environ.get(
        "HIPAD_BASE_CKPT",
        PROJECT_ROOT.parent / "AdaptDrive-assets/hipad/checkpoints/hipad_b2d_stage2_base.pth",
    )
)
DEFAULT_FINETUNE = Path(
    os.environ.get(
        "FINETUNE_CKPT",
        PROJECT_ROOT.parent
        / "AdaptDrive-assets/adaptdrive/checkpoints/adaptdrive_sig7_step140906.pt",
    )
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", default=str(DEFAULT_BASE))
    parser.add_argument("--finetune-checkpoint", default=str(DEFAULT_FINETUNE))
    args = parser.parse_args()

    clean_root = Path(os.environ.get("HIPAD_ROOT", PROJECT_ROOT / "HiP-AD")).expanduser().resolve()
    bundle = load_adapter_checkpoint(
        args.finetune_checkpoint,
        base_checkpoint_path=args.base_checkpoint,
        expected_project_root=str(clean_root),
    )
    assert bundle.training_signature_version == 7
    assert bundle.replay_schema_version == 5
    assert bundle.adapter_mode == "dcnv4_feature"
    assert bundle.feature_adapter_levels == FOUR_LEVELS
    assert bundle.feature_adapter_feature_dim == 256
    assert bundle.feature_adapter_ego_state_dim == 21
    assert len(bundle.hipad_trainable) == 25
    assert len(bundle.feature_adapter_state) == 132
    assert bundle.adapter_prediction_present

    with tempfile.TemporaryDirectory(prefix="adaptdrive_v8_deployment_") as temp_dir:
        v8_path = Path(temp_dir) / "checkpoint_v8.pt"
        torch.save(
            {
                "checkpoint_version": 2,
                "training_signature_version": 8,
                "training_signature": "8" * 64,
                "agent": {
                    "adapter_mode": bundle.adapter_mode,
                    "feature_adapter_levels": bundle.feature_adapter_levels,
                    "hipad_trainable": bundle.hipad_trainable,
                    "feature_dcnv4_adapter": bundle.feature_adapter_state,
                    "adapter_prediction": {"deployment_validation": True},
                },
                "finetune_config": {
                    "control_semantics": bundle.control_semantics,
                    "replay_schema_version": bundle.replay_schema_version,
                    "adapter_mode": bundle.adapter_mode,
                    "feature_adapter_levels": bundle.feature_adapter_levels,
                    "feature_adapter_feature_dim": bundle.feature_adapter_feature_dim,
                    "feature_adapter_ego_state_dim": bundle.feature_adapter_ego_state_dim,
                    "feature_adapter_ego_hidden_dim": bundle.feature_adapter_ego_hidden_dim,
                    "feature_adapter_bottleneck_reduction": bundle.feature_adapter_bottleneck_reduction,
                    "feature_adapter_dcn_group": bundle.feature_adapter_dcn_group,
                    "feature_adapter_residual_scale": bundle.feature_adapter_residual_scale,
                    "feature_adapter_zero_init": bundle.feature_adapter_zero_init,
                    "feature_adapter_norm_type": bundle.feature_adapter_norm_type,
                    "feature_adapter_norm_groups": bundle.feature_adapter_norm_groups,
                    "hipad_project_root": "/historical/training/HiP-AD",
                },
                "runtime_provenance": {
                    "checkpoint.sha256": bundle.base_checkpoint_sha256,
                    "hipad_project_root": "/historical/training/HiP-AD",
                },
            },
            v8_path,
        )
        v8_bundle = load_adapter_checkpoint(
            str(v8_path),
            base_checkpoint_path=args.base_checkpoint,
            expected_project_root=str(clean_root),
        )
        assert v8_bundle.training_signature_version == 8
        assert v8_bundle.checkpoint_sha256 != bundle.checkpoint_sha256
        assert len(v8_bundle.hipad_trainable) == 25
        assert len(v8_bundle.feature_adapter_state) == 132
    print(
        "hipad_clean_dcnv4_leaderboard_loader: PASS "
        f"step_signatures=({bundle.training_signature_version},{v8_bundle.training_signature_version}) "
        f"hipad_trainable={len(bundle.hipad_trainable)} "
        f"feature_adapter={len(bundle.feature_adapter_state)} "
        f"levels={bundle.feature_adapter_levels}",
        flush=True,
    )


if __name__ == "__main__":
    main()
