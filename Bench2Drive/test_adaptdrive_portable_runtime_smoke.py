#!/usr/bin/env python3
"""Fast offline checks for AdaptDrive relocation and v7 deployment loading."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "HiP-AD" / "local_runtime" / "hipad_b2d_stage2_clean_local.py"
OFFLINE_CONFIG_PATHS = (
    PROJECT_ROOT / "HiP-AD" / "projects" / "configs" / "hipad_b2d_stage1.py",
    PROJECT_ROOT / "HiP-AD" / "projects" / "configs" / "hipad_b2d_stage2.py",
)
LEADERBOARD_ROOT = PROJECT_ROOT / "Bench2Drive" / "leaderboard"
if str(LEADERBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(LEADERBOARD_ROOT))

from rl.hipad_clean_adapter_checkpoint import load_adapter_checkpoint  # noqa: E402


@contextmanager
def _environment(name: str, value):
    previous = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = str(value)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _load_config_as_mmcv(config_path: Path, file_dirname: Path):
    """Execute this pure-Python config after MMCV predefined-var substitution."""

    source = config_path.read_text(encoding="utf-8")
    source = source.replace("{{ fileDirname }}", str(file_dirname).replace("\\", "/"))
    namespace = {"__file__": str(config_path)}
    exec(compile(source, str(config_path), "exec"), namespace)
    return namespace


def _expect_runtime_error(fragment: str, operation) -> None:
    try:
        operation()
    except RuntimeError as exc:
        if fragment not in str(exc):
            raise AssertionError(f"expected {fragment!r} in error, got: {exc}") from exc
    else:
        raise AssertionError(f"expected RuntimeError containing {fragment!r}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint(
    base_hash: str,
    saved_project_root: str,
    *,
    hipad_count: int = 25,
    adapter_count: int = 132,
):
    return {
        "checkpoint_version": 1,
        "training_signature_version": 7,
        "training_signature": "7" * 64,
        "runtime_provenance": {"checkpoint.sha256": base_hash},
        "finetune_config": {
            "replay_schema_version": 5,
            "control_semantics": "hipad_clean_dual_pid_v2_mode_aligned",
            "adapter_mode": "dcnv4_feature",
            "feature_adapter_levels": [0, 1, 2, 3],
            "feature_adapter_feature_dim": 256,
            "feature_adapter_ego_state_dim": 21,
            # Deliberately point at a different pre-migration workspace. A v7
            # root is metadata, while the immutable base hash proves lineage.
            "hipad_project_root": saved_project_root,
        },
        "agent": {
            "adapter_mode": "dcnv4_feature",
            "feature_adapter_levels": [0, 1, 2, 3],
            "hipad_trainable": {
                f"planning.{index}": torch.zeros(1) for index in range(hipad_count)
            },
            "feature_dcnv4_adapter": {
                f"adapter.{index}": torch.zeros(1) for index in range(adapter_count)
            },
        },
    }


def test_config_paths_are_relocation_safe() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        parent = Path(temp_dir)
        runtime_dir = parent / "AdaptDrive" / "HiP-AD" / "local_runtime"
        runtime_dir.mkdir(parents=True)

        with _environment("ADAPTDRIVE_ASSET_ROOT", None):
            config = _load_config_as_mmcv(CONFIG_PATH, runtime_dir)
        clean_root = runtime_dir.parent.resolve()
        default_asset_root = (parent / "AdaptDrive-assets").resolve()
        assert Path(config["project_dir"]) == clean_root
        assert Path(config["anchor_paths"]["det"]).parent == clean_root / "data" / "kmeans"
        assert Path(config["model"]["img_backbone"]["pretrained"]) == (
            default_asset_root / "hipad" / "pretrained" / "resnet50-19c8e357.pth"
        )

        explicit_asset_root = (parent / "immutable-assets").resolve()
        with _environment("ADAPTDRIVE_ASSET_ROOT", explicit_asset_root):
            config = _load_config_as_mmcv(CONFIG_PATH, runtime_dir)
        assert Path(config["model"]["img_backbone"]["pretrained"]) == (
            explicit_asset_root / "hipad" / "pretrained" / "resnet50-19c8e357.pth"
        )

        with _environment("ADAPTDRIVE_ASSET_ROOT", "relative-assets"):
            try:
                _load_config_as_mmcv(CONFIG_PATH, runtime_dir)
            except ValueError as exc:
                assert "must be an absolute path" in str(exc)
            else:
                raise AssertionError("relative ADAPTDRIVE_ASSET_ROOT was accepted")


def test_offline_config_paths_are_relocation_safe() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        parent = Path(temp_dir)
        clean_root = parent / "AdaptDrive" / "HiP-AD"
        config_dir = clean_root / "projects" / "configs"
        config_dir.mkdir(parents=True)
        default_asset_root = parent / "AdaptDrive-assets"

        with _environment("ADAPTDRIVE_ASSET_ROOT", None), _environment("HIPAD_STAGE1_CKPT", None):
            configs = [
                _load_config_as_mmcv(config_path, config_dir)
                for config_path in OFFLINE_CONFIG_PATHS
            ]
        for config in configs:
            assert Path(config["project_dir"]) == clean_root
            assert Path(config["anchor_paths"]["det"]).parent == clean_root / "data" / "kmeans"
            assert Path(config["model"]["img_backbone"]["pretrained"]) == (
                default_asset_root / "hipad" / "pretrained" / "resnet50-19c8e357.pth"
            )
        assert configs[1]["load_from"] is None

        explicit_asset_root = parent / "immutable-assets"
        stage1_checkpoint = parent / "checkpoints" / "stage1.pth"
        with _environment("ADAPTDRIVE_ASSET_ROOT", explicit_asset_root), _environment(
            "HIPAD_STAGE1_CKPT", stage1_checkpoint
        ):
            stage2 = _load_config_as_mmcv(OFFLINE_CONFIG_PATHS[1], config_dir)
        assert Path(stage2["model"]["img_backbone"]["pretrained"]) == (
            explicit_asset_root / "hipad" / "pretrained" / "resnet50-19c8e357.pth"
        )
        assert Path(stage2["load_from"]) == stage1_checkpoint

        with _environment("ADAPTDRIVE_ASSET_ROOT", "relative-assets"):
            try:
                _load_config_as_mmcv(OFFLINE_CONFIG_PATHS[0], config_dir)
            except ValueError as exc:
                assert "ADAPTDRIVE_ASSET_ROOT must be an absolute path" in str(exc)
            else:
                raise AssertionError("offline config accepted a relative ADAPTDRIVE_ASSET_ROOT")

        with _environment("HIPAD_STAGE1_CKPT", "relative-stage1.pth"):
            try:
                _load_config_as_mmcv(OFFLINE_CONFIG_PATHS[1], config_dir)
            except ValueError as exc:
                assert "HIPAD_STAGE1_CKPT must be an absolute path" in str(exc)
            else:
                raise AssertionError("stage-2 config accepted a relative HIPAD_STAGE1_CKPT")


def test_legacy_v7_root_uses_content_hash_compatibility() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        runtime_root = root / "moved" / "HiP-AD"
        plugin_dir = runtime_root / "projects" / "mmdet3d_plugin"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "__init__.py").write_text("", encoding="utf-8")

        base = root / "base.pth"
        base.write_bytes(b"audited base checkpoint bytes")
        saved_project_root = str(root / "source-workspace" / "HiP-AD")
        checkpoint_path = root / "legacy_v7.pt"
        torch.save(_checkpoint(_sha256(base), saved_project_root), checkpoint_path)

        bundle = load_adapter_checkpoint(
            str(checkpoint_path),
            base_checkpoint_path=str(base),
            expected_project_root=str(runtime_root),
        )
        assert bundle.training_signature_version == 7
        assert bundle.base_checkpoint_sha256 == _sha256(base)
        assert len(bundle.hipad_trainable) == 25
        assert len(bundle.feature_adapter_state) == 132

        _expect_runtime_error(
            "base_checkpoint_path is required",
            lambda: load_adapter_checkpoint(
                str(checkpoint_path),
                expected_project_root=str(runtime_root),
            ),
        )

        wrong_base = root / "wrong_base.pth"
        wrong_base.write_bytes(b"different checkpoint bytes")
        _expect_runtime_error(
            "base checkpoint hash mismatch",
            lambda: load_adapter_checkpoint(
                str(checkpoint_path),
                base_checkpoint_path=str(wrong_base),
                expected_project_root=str(runtime_root),
            ),
        )

        incomplete_path = root / "incomplete_v7.pt"
        torch.save(
            _checkpoint(_sha256(base), saved_project_root, hipad_count=24),
            incomplete_path,
        )
        _expect_runtime_error(
            "hipad_trainable tensor count mismatch",
            lambda: load_adapter_checkpoint(
                str(incomplete_path),
                base_checkpoint_path=str(base),
                expected_project_root=str(runtime_root),
            ),
        )

        incomplete_adapter_path = root / "incomplete_adapter_v7.pt"
        torch.save(
            _checkpoint(_sha256(base), saved_project_root, adapter_count=131),
            incomplete_adapter_path,
        )
        _expect_runtime_error(
            "feature_dcnv4_adapter tensor count mismatch",
            lambda: load_adapter_checkpoint(
                str(incomplete_adapter_path),
                base_checkpoint_path=str(base),
                expected_project_root=str(runtime_root),
            ),
        )


def main() -> None:
    test_config_paths_are_relocation_safe()
    test_offline_config_paths_are_relocation_safe()
    test_legacy_v7_root_uses_content_hash_compatibility()
    print("AdaptDrive portable runtime smoke passed")


if __name__ == "__main__":
    main()
