#!/usr/bin/env python3
"""Fast, dependency-light gate for the frozen AdaptDrive research contract."""

from __future__ import annotations

from pathlib import Path
import sys


BENCH2DRIVE_ROOT = Path(__file__).resolve().parent
LEADERBOARD_ROOT = BENCH2DRIVE_ROOT / "leaderboard"
for path in (BENCH2DRIVE_ROOT, LEADERBOARD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rl.hipad_policy_finetune_config import HiPADPolicyFinetuneConfig  # noqa: E402


FORBIDDEN_CANONICAL_TOKENS = (
    "FeatureAdapterAuxAgentMixin",
    "feature_adapter_aux",
    "feature_adapter_train_aux",
    "feature_adapter_train_actor_loss",
    "feature_adapter_update_mode",
    "dense_safety_reward",
    "use_dense_safety_reward",
    "--use-dense-safety-reward",
    "allow-signature-mismatch-full-resume",
    "--strict-resume-signature",
    "loading agent weights only",
    "buffer_state.npy",
    "/home/tmp2",
)


def main() -> None:
    config = HiPADPolicyFinetuneConfig()
    assert tuple(config.feature_adapter_levels) == (0, 1, 2, 3)
    assert config.adapter_prediction_update_mode == "prediction_only"
    for legacy_attribute in (
        "feature_adapter_train_aux",
        "feature_adapter_aux_lr",
        "feature_adapter_update_mode",
        "use_dense_safety_reward",
        "dense_voxel_size",
    ):
        assert not hasattr(config, legacy_attribute), legacy_attribute

    canonical_files = (
        BENCH2DRIVE_ROOT / "stable_train_hipad_policy_finetune.py",
        LEADERBOARD_ROOT / "rl" / "hipad_policy_finetune_config.py",
        LEADERBOARD_ROOT / "rl" / "hipad_policy_finetune_agent.py",
    )
    canonical_source = "\n".join(path.read_text(encoding="utf-8") for path in canonical_files)
    for token in FORBIDDEN_CANONICAL_TOKENS:
        assert token not in canonical_source, token

    trainer_source = canonical_files[0].read_text(encoding="utf-8")
    signature_source = (
        LEADERBOARD_ROOT / "rl" / "adaptdrive_training_signature.py"
    ).read_text(encoding="utf-8")
    assert "TRAINING_SIGNATURE_VERSION = 8" in signature_source
    assert 'default="dcnv4_feature"' in trainer_source
    assert 'default="line_e"' in trainer_source
    assert "enable_direct_dense_safety=True" in trainer_source
    assert "adapter_prediction_enabled=True" in trainer_source
    assert '"routes_path"' not in signature_source
    assert "PATH_CONFIG_SIGNATURE_KEYS" in signature_source
    assert '"hipad_content_manifest"' in signature_source
    assert '"roach_bev_map_manifest"' in signature_source

    project_root = BENCH2DRIVE_ROOT.parent
    vendored_dcnv4 = project_root / "third_party" / "DCNv4"
    assert (vendored_dcnv4 / "DCNv4_op" / "setup.py").is_file()
    assert (vendored_dcnv4 / "LICENSE").is_file()
    assert (vendored_dcnv4 / "ADAPTDRIVE_UPSTREAM.md").is_file()
    assert (vendored_dcnv4 / "ADAPTDRIVE_BUILD.md").is_file()
    assert (vendored_dcnv4 / "UPSTREAM_SHA256SUMS").is_file()
    for launcher_name in (
        "run_adaptdrive_train.sh",
        "run_adaptdrive_eval.sh",
        "run_adaptdrive_leaderboard.sh",
        "run_hipad_clean_dcnv4_adapter_eval.sh",
    ):
        launcher_source = (project_root / "Bench2Drive" / launcher_name).read_text(encoding="utf-8")
        assert "third_party/DCNv4" in launcher_source
        assert 'DCNV4_ROOT="${DCNV4_ROOT:-}"' in launcher_source
    assert 'PROJECT_ROOT / "leaderboard/rl/reward.py"' in trainer_source
    assert 'PROJECT_ROOT / "leaderboard/rl/adaptdrive_init.py"' in trainer_source
    assert 'PROJECT_ROOT / "leaderboard/rl/adaptdrive_replay.py"' in trainer_source
    assert 'PROJECT_ROOT / "leaderboard/rl/hipad_clean_control.py"' in trainer_source
    assert 'PROJECT_ROOT / "leaderboard/rl/sim_backend.py"' in trainer_source
    assert 'PROJECT_ROOT / "leaderboard/rl/navigation_route_planner.py"' in trainer_source
    assert '"--init-from"' in trainer_source
    assert "create_feature_replay(" in trainer_source
    assert "resume_feature_replay(" in trainer_source
    assert '"replay_ref": replay_ref' in trainer_source
    assert '"experiment_id": experiment_id' in trainer_source
    assert "load_optimizers=True, strict=True" in trainer_source
    assert "checkpoint_resume_decision" not in trainer_source

    eval_source = (BENCH2DRIVE_ROOT / "stable_eval_hipad_policy.py").read_text(encoding="utf-8")
    deployment_source = (
        LEADERBOARD_ROOT / "rl" / "hipad_clean_adapter_checkpoint.py"
    ).read_text(encoding="utf-8")
    assert "SUPPORTED_DEPLOYMENT_SIGNATURE_VERSIONS = (7, TRAINING_SIGNATURE_VERSION)" in eval_source
    assert "saved_root != expected_root" not in eval_source
    assert "provenance_root != expected_root" not in eval_source
    assert "SUPPORTED_DEPLOYMENT_SIGNATURE_VERSIONS = (7, CURRENT_TRAINING_SIGNATURE_VERSION)" in deployment_source
    assert "expected one of {SUPPORTED_DEPLOYMENT_SIGNATURE_VERSIONS}" in deployment_source

    for launcher_name in (
        "run_adaptdrive_train.sh",
        "run_adaptdrive_eval.sh",
        "run_adaptdrive_leaderboard.sh",
    ):
        launcher_source = (BENCH2DRIVE_ROOT / launcher_name).read_text(encoding="utf-8")
        assert "EXPERIMENT_ID" in launcher_source
        assert "ADAPTDRIVE_RUN_ROOT" in launcher_source
        assert "ADAPTDRIVE_VALIDATE_ONLY" in launcher_source
        for token in ("/home/tmp2", "/opt/data/private/project", "/home/deeplearning", "small_paper"):
            assert token not in launcher_source, (launcher_name, token)

    agent_source = canonical_files[2].read_text(encoding="utf-8")
    assert "class HiPADPolicyFinetuneAgent(AdapterPredictionAgentMixin):" in agent_source
    assert '"adapter_prediction": self._adapter_prediction_state_dict()' in agent_source
    assert "self._load_adapter_prediction_state(" in agent_source

    reward_source = (LEADERBOARD_ROOT / "rl" / "reward.py").read_text(encoding="utf-8")
    assert "def _compute_direct_dense_safety(" in reward_source
    assert '"r_dense_safety_direct"' in reward_source

    print(
        "adaptdrive_contract_smoke: PASS "
        "adapter=dcnv4_feature levels=(0,1,2,3) "
        "update=prediction_only reward=line_e+direct_dense_safety",
        flush=True,
    )


if __name__ == "__main__":
    main()
