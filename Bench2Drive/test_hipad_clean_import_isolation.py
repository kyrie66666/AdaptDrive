#!/usr/bin/env python3
"""Fast process-isolation smoke for the canonical AdaptDrive HiP-AD tree."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
RL_ROOT = PROJECT_ROOT / "leaderboard"
CLEAN_ROOT = REPO_ROOT / "HiP-AD"

if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))

from rl.hipad_project_runtime import (
    HiPADProjectIsolationError,
    activate_hipad_project_root,
    validate_hipad_checkpoint_asset,
    validate_hipad_checkpoint_role,
    validate_runtime_asset,
)


def _assert_clean_activation() -> None:
    activated = activate_hipad_project_root(CLEAN_ROOT, repo_root=REPO_ROOT)
    assert activated == CLEAN_ROOT.resolve()
    assert Path(sys.path[0]).resolve() == CLEAN_ROOT.resolve()
    assert Path(sys.path[1]).resolve() == (CLEAN_ROOT / "bench2drive").resolve()


def _assert_competing_project_rejected_in_fresh_process() -> None:
    with tempfile.TemporaryDirectory(prefix="hipad_competing_root_") as tmpdir:
        competing_root = Path(tmpdir) / "HiP-AD"
        plugin_dir = competing_root / "projects" / "mmdet3d_plugin"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "__init__.py").write_text("", encoding="utf-8")
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(RL_ROOT)!r}); "
            "from rl.hipad_project_runtime import activate_hipad_project_root, HiPADProjectIsolationError; "
            f"sys.path.insert(0, {str(competing_root)!r}); "
            "\ntry:\n"
            f" activate_hipad_project_root({str(CLEAN_ROOT)!r}, repo_root={str(REPO_ROOT)!r})\n"
            "except HiPADProjectIsolationError:\n sys.exit(0)\n"
            "sys.exit(3)\n"
        )
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        result = subprocess.run([sys.executable, "-c", code], env=env, check=False)
        assert result.returncode == 0, result.returncode


def main() -> None:
    _assert_competing_project_rejected_in_fresh_process()
    _assert_clean_activation()
    finetuned = CLEAN_ROOT / "work_dirs" / "hipad_b2d_stage2_finetuned" / "finetuned_latest.pth"
    try:
        validate_hipad_checkpoint_role(finetuned, "clean_base")
    except (HiPADProjectIsolationError, FileNotFoundError):
        pass
    else:
        raise AssertionError("historical finetuned checkpoint was accepted as clean_base")
    validate_hipad_checkpoint_role(finetuned, "clean_finetuned")
    runtime_config = CLEAN_ROOT / "local_runtime" / "hipad_b2d_stage2_clean_local.py"
    assert validate_runtime_asset(runtime_config, label="canonical config") == runtime_config.resolve()
    source_tree_asset = CLEAN_ROOT / "data" / "kmeans" / "b2d_det_900.npy"
    try:
        validate_hipad_checkpoint_asset(
            source_tree_asset,
            label="source-tree checkpoint gate",
            reject_symlink=True,
            checkpoint_role="clean_base",
            repo_root=REPO_ROOT,
        )
    except HiPADProjectIsolationError:
        pass
    else:
        raise AssertionError("a checkpoint-like asset inside the HiP-AD source tree was accepted")
    with tempfile.TemporaryDirectory(prefix="hipad_wrong_base_") as tmpdir:
        wrong_base = Path(tmpdir) / "base.pth"
        wrong_base.write_bytes(b"not the canonical HiP-AD base checkpoint")
        try:
            validate_hipad_checkpoint_asset(
                wrong_base,
                label="wrong external base",
                reject_symlink=True,
                checkpoint_role="clean_base",
                repo_root=REPO_ROOT,
            )
        except HiPADProjectIsolationError:
            pass
        else:
            raise AssertionError("an external checkpoint with the wrong hash was accepted as clean_base")
    print("HiP-AD import isolation smoke passed")


if __name__ == "__main__":
    main()
