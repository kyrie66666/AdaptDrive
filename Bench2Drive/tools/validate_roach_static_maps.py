#!/usr/bin/env python3
"""Offline validator for generated Roach static Town map assets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from roach_map.asset_schema import (  # noqa: E402
    validate_global_asset,
    validate_static_map_asset,
    validate_tiled_asset,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", nargs="+", help="HDF5 asset files or directories")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip streaming value/nonzero scan; schema and shape are still checked",
    )
    parser.add_argument("--no-verify-hash", action="store_true")
    parser.add_argument("--allow-missing-manifest", action="store_true")
    parser.add_argument(
        "--storage-mode",
        choices=("auto", "global", "tiled"),
        default="auto",
        help="Expected storage mode; auto accepts either global or tiled assets",
    )
    return parser.parse_args(argv)


def _expand_assets(values: Sequence[str]):
    paths = []
    for value in values:
        path = Path(value).expanduser()
        if path.is_dir():
            paths.extend(sorted(path.glob("*.h5")))
        else:
            paths.append(path)
    deduplicated = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduplicated.append(resolved)
    return deduplicated


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    asset_paths = _expand_assets(args.assets)
    if not asset_paths:
        raise RuntimeError("No .h5 assets matched the supplied paths")

    failed = False
    for asset_path in asset_paths:
        try:
            validator = validate_static_map_asset
            if args.storage_mode == "global":
                validator = validate_global_asset
            elif args.storage_mode == "tiled":
                validator = validate_tiled_asset
            report = validator(
                asset_path,
                verify_manifest_hash=not args.no_verify_hash,
                require_manifest=not args.allow_missing_manifest,
                scan_values=not args.metadata_only,
            )
        except Exception as exc:
            report = {
                "asset_path": str(asset_path),
                "valid": False,
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        failed = failed or not bool(report.get("valid", False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
