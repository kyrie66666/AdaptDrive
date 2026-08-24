"""Path-independent semantic signature for new AdaptDrive training runs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


TRAINING_SIGNATURE_VERSION = 8

RUNTIME_CONFIG_SIGNATURE_KEYS = {
    "max_train_steps",
    "replay_mmap_dir",
    "checkpoint_dir",
    "checkpoint_every",
    "checkpoint_on_episode_end",
    "checkpoint_latest_before_reset",
    "log_dir",
    "replay_skip_log_every",
    "roach_bev_target_debug_dir",
    "roach_bev_target_debug_interval",
    "roach_bev_target_debug_max_frames",
}

PATH_CONFIG_SIGNATURE_KEYS = {
    "hipad_project_root",
    "hipad_config_path",
    "hipad_checkpoint_path",
    "hipad_checkpoint_asset_origin",
    "roach_bev_map_root",
    "routes",
}

HIPAD_CONTENT_PATHS = (
    "local_runtime/hipad_b2d_stage2_clean_local.py",
    "data/kmeans/b2d_det_900.npy",
    "data/kmeans/b2d_map_100.npy",
    "data/kmeans/b2d_motion_6.npy",
    "data/kmeans/b2d_plan_spat_6x8_2m.npy",
    "data/kmeans/b2d_plan_spat_6x8_5m.npy",
    "projects/mmdet3d_plugin/__init__.py",
    "bench2drive/leaderboard/team_code/hipad_b2d_agent.py",
    "bench2drive/leaderboard/team_code/planner.py",
    "bench2drive/leaderboard/team_code/pid_controller.py",
)

HIPAD_CONTENT_DIRS = (
    "projects/mmdet3d_plugin/models",
    "projects/mmdet3d_plugin/ops",
)

SEMANTIC_SOURCE_SUFFIXES = frozenset({".py", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".so"})
EXPECTED_ROACH_BEV_ASSETS = frozenset({
    "Town01.h5",
    "Town02.h5",
    "Town03.h5",
    "Town04.h5",
    "Town05.h5",
    "Town06.h5",
    "Town07.h5",
    "Town10HD.h5",
    "Town11.h5",
    "Town12.h5",
    "Town13.h5",
    "Town15.h5",
})


def canonical_json_bytes(value) -> bytes:
    """Encode structured protocol data deterministically."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def hash_file(path) -> str:
    path = Path(path)
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, label: str) -> str:
    digest = hash_file(path)
    if not digest:
        raise FileNotFoundError(f"{label} is missing: {path}")
    return digest


def content_manifest(root: Path, relative_paths: Iterable[str]) -> Mapping[str, str]:
    root = Path(root).expanduser().resolve()
    return {
        relative_path: _require_hash(root / relative_path, f"signature content {relative_path}")
        for relative_path in sorted(str(path).replace("\\", "/") for path in relative_paths)
    }


def directory_content_manifest(
    root: Path,
    relative_dirs: Sequence[str],
    *,
    suffixes=SEMANTIC_SOURCE_SUFFIXES,
) -> Mapping[str, str]:
    root = Path(root).expanduser().resolve()
    relative_paths = set()
    for relative_dir in relative_dirs:
        directory = (root / relative_dir).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"signature content directory is missing: {directory}")
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in suffixes and "__pycache__" not in path.parts:
                relative_paths.add(path.relative_to(root).as_posix())
    if not relative_paths:
        raise RuntimeError("signature content directories contain no semantic source or binary files")
    return content_manifest(root, relative_paths)


def dcnv4_content_manifest(*, required: bool) -> Mapping[str, str]:
    """Fingerprint the external DCNv4 package without embedding its location."""

    spec = importlib.util.find_spec("DCNv4")
    if spec is None:
        if required:
            raise RuntimeError(
                "DCNv4 is not importable. Install it in the active environment or add DCNV4_ROOT "
                "to PYTHONPATH before computing an AdaptDrive training signature."
            )
        return {}

    roots = [Path(path).resolve() for path in (spec.submodule_search_locations or ())]
    if not roots and spec.origin:
        roots = [Path(spec.origin).resolve().parent]
    manifest = {}
    for root_index, root in enumerate(roots):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in SEMANTIC_SOURCE_SUFFIXES:
                continue
            logical_name = f"package{root_index}/{path.relative_to(root).as_posix()}"
            manifest[logical_name] = _require_hash(path, f"DCNv4 content {logical_name}")
    if required and not manifest:
        raise RuntimeError("DCNv4 was found but no Python/CUDA/shared-library content could be fingerprinted")
    return manifest


def roach_bev_map_manifest(root: str) -> Mapping[str, Mapping[str, str]]:
    if not root:
        return {}
    map_root = Path(root).expanduser().resolve()
    if not map_root.is_dir():
        raise FileNotFoundError(f"ROACH_BEV_MAP_ROOT is missing: {map_root}")
    manifest = {}
    for metadata_path in sorted(map_root.glob("*.h5.manifest.json")):
        if metadata_path.is_symlink():
            raise RuntimeError(f"Roach BEV manifest must not be a symlink: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        asset_name = str(metadata.get("asset_path") or metadata_path.name.removesuffix(".manifest.json"))
        if Path(asset_name).name != asset_name or asset_name not in EXPECTED_ROACH_BEV_ASSETS:
            raise RuntimeError(f"unexpected or unsafe Roach BEV asset name: {asset_name!r}")
        asset_path = map_root / asset_name
        if asset_path.is_symlink():
            raise RuntimeError(f"Roach BEV asset must not be a symlink: {asset_path}")
        actual_hash = _require_hash(asset_path, f"Roach BEV asset {asset_name}")
        declared_hash = str(metadata.get("asset_sha256", ""))
        if declared_hash and declared_hash != actual_hash:
            raise RuntimeError(
                f"Roach BEV asset hash mismatch for {asset_name}: "
                f"manifest={declared_hash}, actual={actual_hash}"
            )
        declared_size = int(metadata.get("asset_size_bytes", -1))
        actual_size = int(asset_path.stat().st_size)
        if declared_size != actual_size:
            raise RuntimeError(
                f"Roach BEV asset size mismatch for {asset_name}: "
                f"manifest={declared_size}, actual={actual_size}"
            )
        manifest[asset_name] = {
            "asset_sha256": actual_hash,
            "manifest_sha256": _require_hash(metadata_path, f"Roach BEV manifest {metadata_path.name}"),
        }
    if set(manifest) != EXPECTED_ROACH_BEV_ASSETS:
        raise RuntimeError(
            "ROACH_BEV_MAP_ROOT asset set mismatch: "
            f"missing={sorted(EXPECTED_ROACH_BEV_ASSETS - set(manifest))}, "
            f"unexpected={sorted(set(manifest) - EXPECTED_ROACH_BEV_ASSETS)}"
        )
    return manifest


def build_training_signature_payload(
    *,
    env_config,
    config,
    project_root: Path,
    semantic_files: Iterable[Path],
    require_dcnv4: bool = True,
) -> dict:
    """Build the v8 payload without embedding any physical filesystem root."""

    config_payload = asdict(config)
    for key in RUNTIME_CONFIG_SIGNATURE_KEYS | PATH_CONFIG_SIGNATURE_KEYS:
        config_payload.pop(key, None)

    project_root = Path(project_root).expanduser().resolve()
    semantic_manifest = {}
    for path in semantic_files:
        path = Path(path).expanduser().resolve()
        try:
            logical_name = path.relative_to(project_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"semantic signature file escaped project root: {path}") from exc
        semantic_manifest[logical_name] = _require_hash(path, f"semantic file {logical_name}")

    return {
        "training_signature_version": TRAINING_SIGNATURE_VERSION,
        "routes_sha256": _require_hash(Path(env_config.routes), "route file"),
        "env": {
            "observation": asdict(env_config.observation),
            "reward": asdict(env_config.reward),
            "step_manager": asdict(env_config.step_manager),
            "simulation": {
                "traffic_manager_seed": int(env_config.simulation.traffic_manager_seed),
                "frame_rate": float(env_config.simulation.frame_rate),
                "render_offscreen": bool(env_config.simulation.render_offscreen),
                "tile_stream_distance": float(env_config.simulation.tile_stream_distance),
                "actor_active_distance": float(env_config.simulation.actor_active_distance),
                "deterministic_ragdolls": bool(env_config.simulation.deterministic_ragdolls),
                "spectator_as_ego": bool(env_config.simulation.spectator_as_ego),
            },
            "max_episode_steps": env_config.max_episode_steps,
            "random_routes": bool(env_config.random_routes),
            "fixed_route_idx": int(env_config.fixed_route_idx),
            "fixed_route_name": str(env_config.fixed_route_name),
            "sensor_packet_timeout": float(env_config.sensor_packet_timeout),
            "sensor_packet_grace_seconds": float(env_config.sensor_packet_grace_seconds),
            "sensor_packet_max_lag_frames": int(env_config.sensor_packet_max_lag_frames),
            "roach_bev_target_enabled": bool(env_config.roach_bev_target_enabled),
            "roach_bev_target_cache_size": int(env_config.roach_bev_target_cache_size),
        },
        "hipad_policy_finetune_config": config_payload,
        "hipad_checkpoint_sha256": _require_hash(Path(config.hipad_checkpoint_path), "HiP-AD checkpoint"),
        "hipad_content_manifest": content_manifest(Path(config.hipad_project_root), HIPAD_CONTENT_PATHS),
        "hipad_source_manifest": directory_content_manifest(
            Path(config.hipad_project_root), HIPAD_CONTENT_DIRS
        ),
        "dcnv4_content_manifest": dcnv4_content_manifest(required=require_dcnv4),
        "roach_bev_map_manifest": roach_bev_map_manifest(config.roach_bev_map_root),
        "semantic_file_hashes": dict(sorted(semantic_manifest.items())),
    }


def compute_training_signature_v8(**kwargs) -> str:
    payload = build_training_signature_payload(**kwargs)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
