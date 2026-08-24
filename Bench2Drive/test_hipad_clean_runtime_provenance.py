#!/usr/bin/env python3
"""Import the real clean plugin stack and print validated source provenance."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "leaderboard"))

from rl.hipad_project_runtime import activate_hipad_project_root, collect_hipad_provenance


def main() -> None:
    clean_root = REPO_ROOT / "HiP-AD"
    activate_hipad_project_root(clean_root, repo_root=REPO_ROOT)
    provenance = collect_hipad_provenance(clean_root)
    print(json.dumps(provenance, indent=2, sort_keys=True))
    print("HiP-AD runtime provenance smoke passed")


if __name__ == "__main__":
    main()
