#!/usr/bin/env python3
"""Fast provenance gate checks for adapter-aware clean evaluation checkpoints."""

from __future__ import annotations

import copy
import hashlib
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, PROJECT_ROOT / "leaderboard", PROJECT_ROOT / "scenario_runner", REPO_ROOT):
    sys.path.insert(0, str(path))

from rl.hipad_policy_finetune_config import HiPADPolicyFinetuneConfig
from stable_eval_hipad_policy import _normalize_eval_paths, apply_finetune_checkpoint_config


def _must_reject(config, checkpoint, message: str) -> None:
    try:
        apply_finetune_checkpoint_config(config, checkpoint)
    except RuntimeError:
        return
    raise AssertionError(message)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="hipad_clean_eval_gate_") as tmpdir:
        base_path = Path(tmpdir) / "base.pth"
        base_path.write_bytes(b"small deterministic base asset for provenance gate")
        base_hash = hashlib.sha256(base_path.read_bytes()).hexdigest()
        clean_root = REPO_ROOT / "HiP-AD"
        config = HiPADPolicyFinetuneConfig(
            hipad_project_root=str(clean_root),
            hipad_checkpoint_path=str(base_path),
        )
        checkpoint = {
            "checkpoint_version": 1,
            "training_signature_version": 7,
            "training_signature": "1" * 64,
            "agent": {
                "adapter_mode": "dcnv4_feature",
                "feature_adapter_levels": (0, 1, 2, 3),
                "hipad_trainable": {"planning.weight": object()},
                "feature_dcnv4_adapter": {"adapter.weight": object()},
                "adapter_prediction": {"reward_head": object()},
            },
            "finetune_config": {
                "adapter_mode": "dcnv4_feature",
                "feature_adapter_levels": (0, 1, 2, 3),
                "control_semantics": config.control_semantics,
                "replay_schema_version": config.replay_schema_version,
                "hipad_project_root": "/historical/source/HiP-AD",
            },
            "runtime_provenance": {
                "hipad_project_root": "/historical/source/HiP-AD",
                "checkpoint.sha256": base_hash,
            },
        }
        assert apply_finetune_checkpoint_config(config, checkpoint) is checkpoint["agent"]
        assert config.adapter_mode == "dcnv4_feature"
        assert config.feature_adapter_levels == (0, 1, 2, 3)

        current = copy.deepcopy(checkpoint)
        current["checkpoint_version"] = 2
        current["training_signature_version"] = 8
        current["training_signature"] = "2" * 64
        assert apply_finetune_checkpoint_config(config, current) is current["agent"]

        legacy = copy.deepcopy(checkpoint)
        legacy["training_signature_version"] = 6
        _must_reject(config, legacy, "legacy signature was accepted")

        wrong_version = copy.deepcopy(current)
        wrong_version["checkpoint_version"] = 1
        _must_reject(config, wrong_version, "v8 checkpoint_version=1 was accepted")

        malformed_signature = copy.deepcopy(current)
        malformed_signature["training_signature"] = "not-a-sha256"
        _must_reject(config, malformed_signature, "malformed training signature was accepted")

        wrong_schema = copy.deepcopy(checkpoint)
        wrong_schema["finetune_config"]["replay_schema_version"] = 3
        _must_reject(config, wrong_schema, "legacy replay schema was accepted")

        wrong_control = copy.deepcopy(checkpoint)
        wrong_control["finetune_config"]["control_semantics"] = "single_trajectory_legacy"
        _must_reject(config, wrong_control, "single-trajectory control checkpoint was accepted")

        wrong_hash = copy.deepcopy(checkpoint)
        wrong_hash["runtime_provenance"]["checkpoint.sha256"] = "0" * 64
        _must_reject(config, wrong_hash, "mismatched base checkpoint hash was accepted")

        wrong_levels = copy.deepcopy(checkpoint)
        wrong_levels["agent"]["feature_adapter_levels"] = (1, 2, 3)
        _must_reject(config, wrong_levels, "non-four-level adapter checkpoint was accepted")

        wrong_mode = copy.deepcopy(checkpoint)
        wrong_mode["finetune_config"]["adapter_mode"] = "plan_query"
        _must_reject(config, wrong_mode, "non-AdaptDrive adapter mode was accepted")

        run_root = Path(tmpdir) / "external-runs"
        args = SimpleNamespace(
            experiment_id="portable-eval",
            run_root=str(run_root),
            runtime_dir="",
            log_dir="",
        )
        normalized = _normalize_eval_paths(args)
        assert normalized.runtime_dir == str(run_root / "runtime" / "portable-eval")
        assert normalized.log_dir == str(run_root / "evaluations" / "portable-eval")
    print("HiP-AD evaluation checkpoint provenance gate passed")


if __name__ == "__main__":
    main()
