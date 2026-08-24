#!/usr/bin/env python3
"""Config-only smoke for clean anchor normalization and backbone isolation."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
CLEAN_ROOT = REPO_ROOT / "HiP-AD"
sys.path.insert(0, str(PROJECT_ROOT / "leaderboard"))

from rl.hipad_project_runtime import activate_hipad_project_root, configure_and_audit_hipad_assets


def main() -> None:
    activate_hipad_project_root(CLEAN_ROOT, repo_root=REPO_ROOT)
    from mmcv import Config

    cfg = Config.fromfile(str(CLEAN_ROOT / "local_runtime" / "hipad_b2d_stage2_clean_local.py"))
    provenance = configure_and_audit_hipad_assets(cfg, CLEAN_ROOT)
    assert cfg.model.img_backbone.pretrained is None
    asset_paths = [value for key, value in provenance.items() if key.endswith(".path")]
    assert asset_paths
    for path in asset_paths:
        Path(path).relative_to(CLEAN_ROOT)
    print(json.dumps(provenance, indent=2, sort_keys=True))
    print("HiP-AD config asset provenance smoke passed")


if __name__ == "__main__":
    main()
