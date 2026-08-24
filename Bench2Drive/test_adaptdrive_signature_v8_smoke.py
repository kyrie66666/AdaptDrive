#!/usr/bin/env python3
"""Offline relocation and content-sensitivity smoke for training signature v8."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


BENCH2DRIVE_ROOT = Path(__file__).resolve().parent
LEADERBOARD_ROOT = BENCH2DRIVE_ROOT / "leaderboard"
for path in (BENCH2DRIVE_ROOT, LEADERBOARD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rl.adaptdrive_training_signature import (  # noqa: E402
    HIPAD_CONTENT_DIRS,
    HIPAD_CONTENT_PATHS,
    EXPECTED_ROACH_BEV_ASSETS,
    TRAINING_SIGNATURE_VERSION,
    build_training_signature_payload,
    compute_training_signature_v8,
    hash_file,
)
from rl.hipad_policy_finetune_config import HiPADPolicyFinetuneConfig  # noqa: E402


@dataclass
class _ObservationConfig:
    image_width: int = 1600
    image_height: int = 900


@dataclass
class _RewardConfig:
    reward_variant: str = "line_e"
    enable_line_e_reward: bool = True
    enable_direct_dense_safety: bool = True


@dataclass
class _StepManagerConfig:
    timeout: float = 600.0
    max_tick_count: int = 4000


@dataclass
class _SimulationConfig:
    traffic_manager_seed: int = 0
    frame_rate: float = 20.0
    render_offscreen: bool = True
    tile_stream_distance: float = 650.0
    actor_active_distance: float = 650.0
    deterministic_ragdolls: bool = True
    spectator_as_ego: bool = False


def _make_env(routes: Path):
    return SimpleNamespace(
        routes=str(routes),
        observation=_ObservationConfig(),
        reward=_RewardConfig(),
        step_manager=_StepManagerConfig(),
        simulation=_SimulationConfig(),
        max_episode_steps=4000,
        random_routes=True,
        fixed_route_idx=-1,
        fixed_route_name="",
        sensor_packet_timeout=30.0,
        sensor_packet_log_interval=2.0,
        sensor_packet_grace_seconds=0.5,
        sensor_packet_max_lag_frames=0,
        roach_bev_target_enabled=True,
        roach_bev_target_cache_size=8,
    )


def _make_tree(root: Path, content_tag: str) -> tuple[HiPADPolicyFinetuneConfig, object, list[Path]]:
    project = root / "AdaptDrive"
    bench = project / "Bench2Drive"
    hipad = project / "HiP-AD"
    assets = root / "AdaptDrive-assets"
    maps = assets / "roach_bev_maps"
    maps.mkdir(parents=True)

    for relative_path in HIPAD_CONTENT_PATHS:
        path = hipad / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{relative_path}:{content_tag}".encode("utf-8"))

    for relative_dir in HIPAD_CONTENT_DIRS:
        path = hipad / relative_dir / "semantic_stub.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative_dir}:{content_tag}", encoding="utf-8")

    routes = bench / "leaderboard/data/bench2drive220.xml"
    routes.parent.mkdir(parents=True, exist_ok=True)
    routes.write_text(f"<routes>{content_tag}</routes>", encoding="utf-8")

    base = assets / "hipad/checkpoints/base.pth"
    base.parent.mkdir(parents=True)
    base.write_bytes(f"base:{content_tag}".encode("utf-8"))

    for asset_name in sorted(EXPECTED_ROACH_BEV_ASSETS):
        town = maps / asset_name
        town.write_bytes(f"map:{asset_name}:{content_tag}".encode("utf-8"))
        (maps / f"{asset_name}.manifest.json").write_text(
            json.dumps(
                {
                    "asset_path": town.name,
                    "asset_sha256": hash_file(town),
                    "asset_size_bytes": town.stat().st_size,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    semantic_files = []
    for relative_path in ("leaderboard/rl/reward.py", "stable_train_hipad_policy_finetune.py"):
        path = bench / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative_path}:{content_tag}", encoding="utf-8")
        semantic_files.append(path)

    config = HiPADPolicyFinetuneConfig(
        hipad_project_root=str(hipad),
        hipad_config_path=str(hipad / HIPAD_CONTENT_PATHS[0]),
        hipad_checkpoint_path=str(base),
        roach_bev_map_root=str(maps),
        routes=str(routes),
        adapter_mode="dcnv4_feature",
        enable_feature_dcnv4_adapter=True,
        feature_adapter_levels=(0, 1, 2, 3),
        adapter_prediction_enabled=True,
        adapter_prediction_train_reward=True,
        adapter_prediction_train_semantic=True,
    )
    return config, _make_env(routes), semantic_files


def main() -> None:
    assert TRAINING_SIGNATURE_VERSION == 8
    with tempfile.TemporaryDirectory() as temp_dir:
        first_root = Path(temp_dir) / "physical-a"
        first_config, first_env, first_files = _make_tree(first_root, "same-content")
        first_signature = compute_training_signature_v8(
            env_config=first_env,
            config=first_config,
            project_root=first_root / "AdaptDrive/Bench2Drive",
            semantic_files=first_files,
            require_dcnv4=False,
        )

        second_root = Path(temp_dir) / "physical-b"
        shutil.copytree(first_root, second_root)
        second_project = second_root / "AdaptDrive"
        second_config = HiPADPolicyFinetuneConfig(**{
            **first_config.__dict__,
            "hipad_project_root": str(second_project / "HiP-AD"),
            "hipad_config_path": str(second_project / "HiP-AD" / HIPAD_CONTENT_PATHS[0]),
            "hipad_checkpoint_path": str(second_root / "AdaptDrive-assets/hipad/checkpoints/base.pth"),
            "roach_bev_map_root": str(second_root / "AdaptDrive-assets/roach_bev_maps"),
            "routes": str(second_project / "Bench2Drive/leaderboard/data/bench2drive220.xml"),
        })
        second_env = _make_env(Path(second_config.routes))
        second_files = [
            second_project / "Bench2Drive/leaderboard/rl/reward.py",
            second_project / "Bench2Drive/stable_train_hipad_policy_finetune.py",
        ]
        second_signature = compute_training_signature_v8(
            env_config=second_env,
            config=second_config,
            project_root=second_project / "Bench2Drive",
            semantic_files=second_files,
            require_dcnv4=False,
        )
        assert first_signature == second_signature, "physical relocation changed signature v8"

        second_env.sensor_packet_log_interval = 99.0
        logging_only_signature = compute_training_signature_v8(
            env_config=second_env,
            config=second_config,
            project_root=second_project / "Bench2Drive",
            semantic_files=second_files,
            require_dcnv4=False,
        )
        assert logging_only_signature == second_signature, "logging cadence changed semantic signature"
        second_env.sensor_packet_log_interval = 2.0

        second_env.simulation.frame_rate = 10.0
        changed_simulation_signature = compute_training_signature_v8(
            env_config=second_env,
            config=second_config,
            project_root=second_project / "Bench2Drive",
            semantic_files=second_files,
            require_dcnv4=False,
        )
        assert changed_simulation_signature != second_signature, "simulation semantics did not change signature"
        second_env.simulation.frame_rate = 20.0

        payload = build_training_signature_payload(
            env_config=second_env,
            config=second_config,
            project_root=second_project / "Bench2Drive",
            semantic_files=second_files,
            require_dcnv4=False,
        )
        serialized = json.dumps(payload, sort_keys=True)
        assert str(first_root) not in serialized
        assert str(second_root) not in serialized

        second_files[0].write_text("semantic code changed", encoding="utf-8")
        changed_signature = compute_training_signature_v8(
            env_config=second_env,
            config=second_config,
            project_root=second_project / "Bench2Drive",
            semantic_files=second_files,
            require_dcnv4=False,
        )
        assert changed_signature != second_signature, "semantic code change did not change signature v8"

    print("adaptdrive_signature_v8_smoke: PASS relocation_stable=1 content_sensitive=1", flush=True)


if __name__ == "__main__":
    main()
