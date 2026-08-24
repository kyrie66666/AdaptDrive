#!/usr/bin/env python3
from __future__ import annotations

"""Standalone trainer for closed-loop RL finetuning of HiP-AD planning policy."""

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
HIPAD_ROOT = REPO_ROOT / "HiP-AD"

for path in (
    REPO_ROOT,
    PROJECT_ROOT,
    PROJECT_ROOT / "leaderboard",
    PROJECT_ROOT / "scenario_runner",
):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

CARLA_ROOT = os.environ.get("CARLA_ROOT", "")


def ensure_carla_python_paths(carla_root: str) -> None:
    root = Path(carla_root)
    carla_egg = root / "PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
    for path in (
        carla_egg,
        root / "PythonAPI",
        root / "PythonAPI/carla",
    ):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


ensure_carla_python_paths(CARLA_ROOT)

from rl.hipad_clean_control import (
    CLEAN_DUAL_PID_KWARGS,
    clean_dual_pid_step,
    pack_clean_dual_pid_summary,
)
from rl.hipad_clean_navigation import (
    bind_clean_global_plan,
    clean_control_target_from_policy_output,
    clean_replay_navigation,
)
from rl.hipad_project_runtime import (
    activate_hipad_project_root,
    collect_hipad_provenance,
    hipad_checkpoint_asset_origin,
    validate_hipad_checkpoint_asset,
    validate_hipad_checkpoint_role,
    validate_runtime_asset,
)
from rl.adaptdrive_sac import DUAL_PID_SUMMARY_DIM
from rl.adaptdrive_init import (
    apply_registered_legacy_parent,
    import_registered_legacy_parent,
)
from rl.adaptdrive_replay import (
    StrictFeatureReplayBuffer,
    create_feature_replay,
    resume_feature_replay,
    validate_experiment_id,
    write_replay_state_snapshot,
)
from rl.adaptdrive_training_signature import (
    TRAINING_SIGNATURE_VERSION,
    compute_training_signature_v8,
    hash_file,
)
from rl.hipad_training_gates import policy_update_decision
from rl.hipad_policy_finetune_config import HiPADPolicyFinetuneConfig
from rl.replay import (
    CLEAN_DUAL_TRAJECTORY_CONTROL,
    CLEAN_DUAL_TRAJECTORY_SCHEMA_VERSION,
)


REWARD_LOG_KEYS = (
    "r_progress",
    "r_speed",
    "r_position",
    "r_rotation",
    "r_action",
    "r_terminal",
    "r_success",
    "r_stuck_soft",
    "r_stuck_terminal",
    "r_blocked",
    "r_timeout",
    "r_hard_stuck",
    "r_free_road_efficiency",
    "r_low_speed",
    "r_no_progress",
    "r_dense_ttc",
    "r_dense_headway",
    "r_dense_min_distance",
    "r_dense_safety_direct",
    "r_comfort",
)

def format_feature_adapter_level_metrics(metrics, levels) -> str:
    metrics = metrics or {}
    pieces = []
    for level in tuple(int(level) for level in levels):
        pieces.extend(
            [
                f"feature_adapter_alpha_L{level}={metrics.get(f'feature_adapter_alpha_L{level}', 0.0):.6f}",
                (
                    f"feature_adapter_raw_residual_l2_L{level}="
                    f"{metrics.get(f'feature_adapter_raw_residual_l2_L{level}', 0.0):.6f}"
                ),
                (
                    f"feature_adapter_effective_delta_l2_L{level}="
                    f"{metrics.get(f'feature_adapter_effective_delta_l2_L{level}', 0.0):.6f}"
                ),
                f"feature_adapter_l2_L{level}={metrics.get(f'feature_adapter_residual_l2_L{level}', 0.0):.6f}",
                (
                    f"feature_adapter_base_rms_L{level}="
                    f"{metrics.get(f'feature_adapter_base_rms_L{level}', 0.0):.8f}"
                ),
                (
                    f"feature_adapter_raw_residual_rms_L{level}="
                    f"{metrics.get(f'feature_adapter_raw_residual_rms_L{level}', 0.0):.8f}"
                ),
                (
                    f"feature_adapter_effective_delta_rms_L{level}="
                    f"{metrics.get(f'feature_adapter_effective_delta_rms_L{level}', 0.0):.8f}"
                ),
                (
                    f"feature_adapter_effective_to_base_ratio_L{level}="
                    f"{metrics.get(f'feature_adapter_effective_to_base_ratio_L{level}', 0.0):.8f}"
                ),
                (
                    f"feature_adapter_adapted_to_base_ratio_L{level}="
                    f"{metrics.get(f'feature_adapter_adapted_to_base_ratio_L{level}', 0.0):.8f}"
                ),
            ]
        )
    return " ".join(pieces)


def format_feature_adapter_alpha_grad_metrics(metrics, levels) -> str:
    metrics = metrics or {}
    return " ".join(
        f"feature_adapter_alpha_grad_L{level}="
        f"{metrics.get(f'feature_adapter_alpha_grad_L{level}', 0.0):.8e}"
        for level in tuple(int(level) for level in levels)
    )


class StableLogger:
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"hipad_policy_finetune_{timestamp}.log"

    def log(self, message: str, level: str = "INFO") -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] [{level}] {message}"
        print(line, flush=True)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def format_optional_float(value, digits: int = 4) -> str:
    if value is None:
        return "nan"
    return f"{float(value):.{digits}f}"


def format_reward_parts(parts: dict) -> str:
    return " ".join(f"{key}={parts.get(key, 0.0):.3f}" for key in REWARD_LOG_KEYS)


def make_reward_config(args):
    from rl.reward import RewardConfig

    reward_variant = str(args.reward_variant or "legacy").lower()
    enable_line_e_reward = bool(args.enable_line_e_reward or reward_variant == "line_e")
    return RewardConfig(
        reward_variant=reward_variant,
        enable_line_e_reward=enable_line_e_reward,
        terminal_penalty_blocked=args.terminal_penalty_blocked,
        terminal_penalty_timeout=args.terminal_penalty_timeout,
        line_e_hard_stuck_terminal_penalty=args.line_e_hard_stuck_terminal_penalty,
        line_e_bbox_safety_wait=bool(args.line_e_bbox_safety_wait),
        line_e_safety_wait_clearance=args.line_e_safety_wait_clearance,
        line_e_safety_wait_lateral_margin=args.line_e_safety_wait_lateral_margin,
        line_e_safe_wait_stop_speed=args.line_e_safe_wait_stop_speed,
        line_e_safe_wait_reward_grace_steps=args.line_e_safe_wait_reward_grace_steps,
        line_e_blocked_timeout_steps=args.line_e_blocked_timeout_steps,
        free_road_efficiency_scale=args.free_road_efficiency_scale,
        low_speed_threshold=args.low_speed_threshold,
        low_speed_grace_steps=args.low_speed_grace_steps,
        low_speed_penalty_per_step=args.low_speed_penalty_per_step,
        low_speed_episode_cap=args.low_speed_episode_cap,
        no_progress_threshold=args.no_progress_threshold,
        no_progress_grace_steps=args.no_progress_grace_steps,
        no_progress_penalty_per_step=args.no_progress_penalty_per_step,
        no_progress_episode_cap=args.no_progress_episode_cap,
        enable_comfort_penalty=bool(args.enable_comfort_penalty),
        comfort_penalty_max=args.comfort_penalty_max,
        enable_direct_dense_safety=bool(args.enable_direct_dense_safety),
        enable_dense_ttc=bool(args.enable_dense_ttc),
        enable_dense_headway=bool(args.enable_dense_headway),
        enable_dense_min_distance=bool(args.enable_dense_min_distance),
        direct_dense_ttc_weight=args.direct_dense_ttc_weight,
        direct_dense_headway_weight=args.direct_dense_headway_weight,
        direct_dense_min_distance_weight=args.direct_dense_min_distance_weight,
        dense_safety_forward_distance=args.direct_dense_forward_distance,
        dense_safety_lateral_distance=args.direct_dense_lateral_distance,
        dense_safety_same_direction_dot=args.direct_dense_same_direction_dot,
        dense_safety_min_closing_speed=args.direct_dense_min_closing_speed,
        ttc_safe_time=args.ttc_safe_time,
        ttc_min_time=args.ttc_min_time,
        dense_ttc_penalty_max=args.dense_ttc_penalty_max,
        headway_time=args.headway_time,
        headway_min_distance=args.headway_min_distance,
        dense_headway_penalty_max=args.dense_headway_penalty_max,
        min_distance_safe=args.min_distance_safe,
        dense_min_distance_penalty_max=args.dense_min_distance_penalty_max,
        dense_safety_penalty_cap=args.direct_dense_safety_penalty_cap,
    )


def make_env_config(args, config: HiPADPolicyFinetuneConfig):
    from rl.env import RLEnvConfig
    from rl.obs_builder import ObservationConfig
    from rl.sim_backend import SimulationConfig
    from rl.step_manager import StepManagerConfig

    simulation = SimulationConfig(
        host=args.host,
        port=args.port,
        traffic_manager_port=args.traffic_manager_port,
        timeout=args.timeout,
        gpu_rank=args.gpu_id,
        carla_cuda_visible_devices=args.carla_cuda_visible_devices,
        carla_graphicsadapter=args.carla_graphicsadapter,
        launch_server=not args.no_launch_server,
        carla_root=args.carla_root,
        save_path=args.runtime_dir,
        runtime_dir=args.xdg_runtime_dir,
        vk_icd_filenames=args.vk_icd_filenames,
        launch_user=args.carla_launch_user,
        server_warmup_seconds=args.server_warmup_seconds,
    )
    return RLEnvConfig(
        routes=args.routes or config.routes,
        simulation=simulation,
        observation=ObservationConfig(image_width=1600, image_height=900),
        reward=make_reward_config(args),
        step_manager=StepManagerConfig(timeout=args.timeout, max_tick_count=config.max_episode_steps),
        max_episode_steps=config.max_episode_steps,
        random_routes=True,
        fixed_route_idx=args.fixed_route_idx,
        fixed_route_name=args.fixed_route_name,
        sensor_packet_timeout=args.sensor_packet_timeout,
        sensor_packet_log_interval=args.sensor_packet_log_interval,
        sensor_packet_grace_seconds=args.sensor_packet_grace_seconds,
        sensor_packet_max_lag_frames=args.sensor_packet_max_lag_frames,
        roach_bev_target_enabled=bool(
            config.adapter_prediction_enabled and config.adapter_prediction_train_semantic
        ),
        roach_bev_map_root=str(config.roach_bev_map_root),
        roach_bev_target_debug_dir=str(config.roach_bev_target_debug_dir),
        roach_bev_target_debug_interval=int(config.roach_bev_target_debug_interval),
        roach_bev_target_debug_max_frames=int(config.roach_bev_target_debug_max_frames),
    )


def make_config(args) -> HiPADPolicyFinetuneConfig:
    config = HiPADPolicyFinetuneConfig()
    config.hipad_project_root = args.hipad_root
    config.hipad_config_path = args.hipad_config
    config.hipad_checkpoint_path = args.hipad_checkpoint
    config.hipad_checkpoint_role = args.hipad_checkpoint_role
    config.strict_policy = bool(args.strict_policy)
    config.deterministic_rollout = bool(args.deterministic_rollout)
    config.max_train_steps = args.max_train_steps
    config.batch_size = args.batch_size
    config.learning_starts = args.learning_starts
    config.train_every_n_steps = args.train_every_n_steps
    config.gradient_steps = args.gradient_steps
    config.policy_learning_starts = args.policy_learning_starts
    config.policy_update_every_n_steps = args.policy_update_every_n_steps
    config.min_critic_updates_before_policy = args.min_critic_updates_before_policy
    config.max_policy_q_loss_ema = args.max_policy_q_loss_ema
    config.q_loss_ema_beta = args.q_loss_ema_beta
    config.replay_capacity = args.replay_capacity
    config.replay_max_storage_gb = args.replay_max_storage_gb
    config.replay_mmap_dir = args.replay_mmap_dir
    config.checkpoint_dir = args.checkpoint_dir
    config.checkpoint_every = args.checkpoint_every
    config.log_dir = args.log_dir
    config.hidden_dim = args.hidden_dim
    config.policy_lr = args.policy_lr
    config.critic_lr = args.critic_lr
    config.alpha_lr = args.alpha_lr
    config.target_entropy = args.target_entropy
    config.max_grad_norm = args.max_grad_norm
    config.critic_loss_type = args.critic_loss_type
    config.critic_huber_delta = args.critic_huber_delta
    config.adapter_mode = str(args.adapter_mode)
    config.enable_ego_state_adapter = False
    config.enable_feature_dcnv4_adapter = True
    config.ego_adapter_feature_dim = int(args.ego_adapter_feature_dim)
    config.ego_adapter_ego_state_dim = int(args.ego_adapter_ego_state_dim)
    config.ego_adapter_hidden_dim = int(args.ego_adapter_hidden_dim)
    config.ego_adapter_ego_hidden_dim = int(args.ego_adapter_ego_hidden_dim)
    config.ego_adapter_residual_scale = float(args.ego_adapter_residual_scale)
    config.ego_adapter_dropout = float(args.ego_adapter_dropout)
    config.ego_adapter_use_layer_norm = bool(args.ego_adapter_use_layer_norm)
    config.feature_adapter_levels = tuple(int(level) for level in args.feature_adapter_levels)
    if config.feature_adapter_levels != (0, 1, 2, 3):
        raise ValueError(
            "AdaptDrive requires the four-level feature adapter: --feature-adapter-levels 0 1 2 3"
        )
    config.feature_adapter_residual_scale = float(args.feature_adapter_residual_scale)
    config.feature_adapter_zero_init = bool(args.feature_adapter_zero_init)
    config.feature_adapter_feature_dim = int(args.feature_adapter_feature_dim)
    config.feature_adapter_ego_state_dim = int(args.feature_adapter_ego_state_dim)
    config.feature_adapter_ego_hidden_dim = int(args.feature_adapter_ego_hidden_dim)
    config.feature_adapter_bottleneck_reduction = int(args.feature_adapter_bottleneck_reduction)
    config.feature_adapter_dcn_group = int(args.feature_adapter_dcn_group)
    config.feature_adapter_norm_type = str(args.feature_adapter_norm_type)
    config.feature_adapter_norm_groups = int(args.feature_adapter_norm_groups)
    config.adapter_prediction_enabled = bool(args.adapter_prediction_enabled)
    config.adapter_prediction_train_reward = bool(args.adapter_prediction_train_reward)
    config.adapter_prediction_train_semantic = bool(args.adapter_prediction_train_semantic)
    config.adapter_prediction_every_n_steps = int(args.adapter_prediction_every_n_steps)
    config.adapter_prediction_reuse_forward_cache = bool(args.adapter_prediction_reuse_forward_cache)
    config.adapter_prediction_update_mode = str(args.adapter_prediction_update_mode)
    config.adapter_prediction_lr = float(args.adapter_prediction_lr)
    config.prediction_head_lr = float(args.prediction_head_lr)
    config.adapter_prediction_weight_decay = float(args.adapter_prediction_weight_decay)
    config.adapter_prediction_max_grad_norm = float(args.adapter_prediction_max_grad_norm)
    config.adapter_prediction_reward_weight = float(args.adapter_prediction_reward_weight)
    config.adapter_prediction_semantic_weight = float(args.adapter_prediction_semantic_weight)
    config.adapter_prediction_residual_weight = float(args.adapter_prediction_residual_weight)
    config.adapter_prediction_semantic_route_weight = float(args.adapter_prediction_semantic_route_weight)
    config.roach_bev_map_root = str(args.roach_bev_map_root or config.roach_bev_map_root)
    config.roach_bev_target_debug_dir = str(args.roach_bev_target_debug_dir)
    config.roach_bev_target_debug_interval = int(args.roach_bev_target_debug_interval)
    config.roach_bev_target_debug_max_frames = int(args.roach_bev_target_debug_max_frames)
    config.reference_kl_weight = args.reference_kl_weight
    config.reference_kl_final_weight = args.reference_kl_final_weight
    config.trajectory_trust_region_weight = args.trajectory_trust_region_weight
    config.reference_decay_steps = args.reference_decay_steps
    config.detach_policy_q_candidates = bool(args.detach_policy_q_candidates)
    config.skip_sensor_mismatch_replay = bool(args.skip_sensor_mismatch_replay)
    config.skip_invalid_terminal_replay = bool(args.skip_invalid_terminal_replay)
    config.replay_skip_log_every = args.replay_skip_log_every
    config.checkpoint_on_episode_end = bool(args.checkpoint_on_episode_end)
    config.checkpoint_latest_before_reset = bool(args.checkpoint_latest_before_reset)
    config.routes = args.routes or config.routes
    return config


def compute_training_signature(env_config, config: HiPADPolicyFinetuneConfig) -> str:
    semantic_files = [
        PROJECT_ROOT / "leaderboard/rl/env.py",
        PROJECT_ROOT / "leaderboard/rl/adaptdrive_env.py",
        PROJECT_ROOT / "leaderboard/rl/sim_backend.py",
        PROJECT_ROOT / "leaderboard/rl/obs_builder.py",
        PROJECT_ROOT / "leaderboard/rl/replay.py",
        PROJECT_ROOT / "leaderboard/rl/reward.py",
        PROJECT_ROOT / "leaderboard/rl/roach_reward.py",
        PROJECT_ROOT / "leaderboard/rl/step_manager.py",
        PROJECT_ROOT / "leaderboard/rl/navigation_route_planner.py",
        PROJECT_ROOT / "leaderboard/rl/adaptdrive_calibration.py",
        PROJECT_ROOT / "leaderboard/rl/hipad_clean_runtime.py",
        PROJECT_ROOT / "leaderboard/rl/hipad_project_runtime.py",
        PROJECT_ROOT / "leaderboard/rl/hipad_clean_bridge.py",
        PROJECT_ROOT / "leaderboard/rl/hipad_clean_navigation.py",
        PROJECT_ROOT / "leaderboard/rl/hipad_clean_control.py",
        PROJECT_ROOT / "leaderboard/rl/hipad_clean_speed_decode.py",
        PROJECT_ROOT / "leaderboard/rl/adaptdrive_sac.py",
        PROJECT_ROOT / "leaderboard/rl/adaptdrive_init.py",
        PROJECT_ROOT / "leaderboard/rl/adaptdrive_replay.py",
        PROJECT_ROOT / "leaderboard/rl/hipad_training_gates.py",
        PROJECT_ROOT / "leaderboard/rl/ego_state_adapter.py",
        PROJECT_ROOT / "leaderboard/rl/adapter_prediction_heads.py",
        PROJECT_ROOT / "leaderboard/rl/adapter_prediction_update.py",
        PROJECT_ROOT / "leaderboard/rl/adaptdrive_training_signature.py",
        PROJECT_ROOT / "leaderboard/rl/hipad_policy_finetune_agent.py",
        PROJECT_ROOT / "leaderboard/rl/hipad_policy_finetune_config.py",
        PROJECT_ROOT / "leaderboard/rl/roach_bev_target.py",
        PROJECT_ROOT / "leaderboard/rl/roach_bev_target_cache.py",
        PROJECT_ROOT / "leaderboard/leaderboard/utils/route_manipulation.py",
        PROJECT_ROOT / "stable_train_hipad_policy_finetune.py",
    ]
    return compute_training_signature_v8(
        env_config=env_config,
        config=config,
        project_root=PROJECT_ROOT,
        semantic_files=semantic_files,
    )


def save_checkpoint(
    agent,
    replay,
    replay_context,
    step: int,
    episode: int,
    path: Path,
    latest_path: Path,
    signature: str,
    experiment_id: str,
    logger: StableLogger,
    trainer_state: Optional[dict] = None,
    initialization: Optional[dict] = None,
):
    def atomic_torch_save(obj, target_path: Path) -> None:
        tmp_path = target_path.with_name(f"{target_path.name}.tmp")
        torch.save(obj, tmp_path)
        os.replace(tmp_path, target_path)

    replay_ref = write_replay_state_snapshot(replay, replay_context)
    checkpoint = {
        "checkpoint_version": 2,
        "checkpoint_uuid": replay_ref["checkpoint_uuid"],
        "experiment_id": experiment_id,
        "training_signature_version": TRAINING_SIGNATURE_VERSION,
        "training_signature": signature,
        "step": int(step),
        "episode": int(episode),
        "agent": agent.state_dict(),
        "trainer_state": dict(trainer_state or {}),
        "replay_size": len(replay),
        "replay_ref": replay_ref,
        "initialization": dict(initialization or {}),
        "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "runtime_provenance": dict(getattr(agent, "runtime_provenance", {})),
        "finetune_config": asdict(agent.config),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(checkpoint, path)
    if latest_path != path:
        atomic_torch_save(checkpoint, latest_path)
    logger.log(
        f"Checkpoint saved: {path} checkpoint_uuid={replay_ref['checkpoint_uuid']} "
        f"replay_uuid={replay_ref['replay_uuid']} replay_size={replay_ref['replay_size']}"
    )


def termination_items(info: dict):
    reasons = info.get("termination_reasons", info.get("termination"))
    if reasons is None:
        return []
    if isinstance(reasons, (list, tuple, set)):
        return [str(item) for item in reasons]
    return [str(reasons)]


def replay_store_decision(current_info: dict, next_info: dict, transition_terminal: bool, config: HiPADPolicyFinetuneConfig):
    if config.skip_invalid_terminal_replay and transition_terminal:
        # ``blocked_timeout`` is a bounded simulator reset after the frozen
        # clean policy has safely waited behind one persistent blocker.  It is
        # not evidence that the sampled SAC mode should receive a terminal
        # failure target, so do not store that one terminal transition.
        invalid_terminal_reasons = {"wrong_world", "blocked_timeout"}
        if not bool(next_info.get("terminal_for_replay", False)):
            invalid_terminal_reasons.add("timeout")
        matched = invalid_terminal_reasons.intersection(set(termination_items(next_info)))
        if matched:
            return False, "invalid_terminal:" + ",".join(sorted(matched))

    if config.skip_sensor_mismatch_replay:
        current_exact = bool(current_info.get("sensor_frame_exact", True))
        next_exact = bool(next_info.get("sensor_frame_exact", True))
        if not current_exact or not next_exact:
            return False, "sensor_packet_mismatch"

    return True, ""


def prime_hipad_after_reset(env, agent, logger: Optional[StableLogger] = None) -> None:
    policy = getattr(agent, "policy", None)
    if policy is None or not getattr(policy, "enabled", False):
        return

    warmup_observations = None
    if hasattr(env, "get_reset_warmup_observations"):
        warmup_observations = env.get_reset_warmup_observations()
    elif hasattr(env, "get_reset_warmup_observation"):
        warmup_observation = env.get_reset_warmup_observation()
        warmup_observations = [] if warmup_observation is None else [warmup_observation]

    if not warmup_observations:
        return

    for warmup_observation in warmup_observations:
        prediction = policy.prime(warmup_observation, fut_ts=agent.config.fut_ts)
        if logger is not None and not prediction.valid and prediction.error not in {"no_observation", ""}:
            logger.log(f"[HiP-AD Warmup] non-fatal prime failure: {prediction.error}", level="WARNING")


def bind_hipad_after_reset(env, agent, logger: Optional[StableLogger] = None) -> dict:
    """Bind the active Bench2Drive route before any clean temporal warmup."""

    binding = bind_clean_global_plan(env, agent)
    if logger is not None:
        logger.log(
            "[HiP-AD Navigation] "
            f"source={binding['source']} route={binding['route_name']} "
            f"raw_gps={binding['raw_gps_points']} raw_world={binding['raw_world_points']} "
            f"downsampled={binding['downsampled_points']}"
        )
    return binding


def log_clean_navigation_context(policy_output, logger: StableLogger, episode_index: int) -> None:
    """Log and validate the first model-consumed navigation context per episode."""

    target_point = clean_control_target_from_policy_output(policy_output)
    target_point_next = np.asarray(policy_output.target_point_next_np, dtype=np.float32)
    if target_point_next.shape != (2,) or not np.isfinite(target_point_next).all():
        raise RuntimeError(
            "Clean policy target_point_next must be a finite (2,) vector, got "
            f"shape={target_point_next.shape}"
        )
    command = int(policy_output.navigation_command)
    if command < 1 or command > 6:
        raise RuntimeError(f"Clean navigation command {command} is outside [1, 6]")
    logger.log(
        f"[HiP-AD Navigation Context] episode={episode_index} command={command} "
        f"target_point=[{target_point[0]:.4f}, {target_point[1]:.4f}] "
        f"target_point_next=[{target_point_next[0]:.4f}, {target_point_next[1]:.4f}]"
    )


def _normalize_run_paths(args):
    experiment_id = validate_experiment_id(args.experiment_id)
    run_root = Path(args.run_root).expanduser().resolve()
    expected_paths = {
        "runtime_dir": run_root / "runtime" / experiment_id,
        "replay_mmap_dir": run_root / "replay" / experiment_id,
        "checkpoint_dir": run_root / "checkpoints" / experiment_id,
        "log_dir": run_root / "logs" / experiment_id,
    }
    for attribute, expected in expected_paths.items():
        configured = str(getattr(args, attribute, "") or "").strip()
        if configured and Path(configured).expanduser().resolve() != expected:
            option = "--" + attribute.replace("_", "-")
            raise ValueError(f"{option} must be {expected} for experiment_id={experiment_id}")
        setattr(args, attribute, str(expected))
    args.experiment_id = experiment_id
    args.run_root = str(run_root)
    return args


def parse_args():
    parser = argparse.ArgumentParser(description="HiP-AD planning cls + final spat-2m reg closed-loop finetune trainer")
    parser.add_argument("--experiment-id", default=os.environ.get("EXPERIMENT_ID", ""))
    parser.add_argument(
        "--run-root",
        default=os.environ.get("ADAPTDRIVE_RUN_ROOT", str(REPO_ROOT.parent / "AdaptDrive-runs")),
    )
    parser.add_argument("--hipad-root", default=str(HIPAD_ROOT))
    parser.add_argument(
        "--hipad-config",
        default=str(HIPAD_ROOT / "local_runtime/hipad_b2d_stage2_clean_local.py"),
    )
    parser.add_argument("--hipad-checkpoint", default=os.environ.get("HIPAD_CKPT", ""))
    parser.add_argument(
        "--hipad-checkpoint-role",
        default=os.environ.get("HIPAD_CHECKPOINT_ROLE", "clean_base"),
        choices=("clean_base", "clean_finetuned"),
    )
    parser.add_argument("--allow-invalid-hipad-plan", action="store_false", dest="strict_policy")
    parser.set_defaults(strict_policy=True)

    parser.add_argument("--carla-root", default=CARLA_ROOT)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=30300, type=int)
    parser.add_argument("--traffic-manager-port", default=52300, type=int)
    parser.add_argument("--timeout", default=600.0, type=float)
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--carla-cuda-visible-devices", default=os.environ.get("CARLA_CUDA_VISIBLE_DEVICES", ""))
    parser.add_argument("--carla-graphicsadapter", default=-1, type=int)
    parser.add_argument("--no-launch-server", action="store_true")
    parser.add_argument("--server-warmup-seconds", default=30.0, type=float)
    parser.add_argument("--runtime-dir", default="")
    parser.add_argument("--xdg-runtime-dir", default="")
    parser.add_argument("--vk-icd-filenames", default="")
    parser.add_argument("--carla-launch-user", default=os.environ.get("CARLA_LAUNCH_USER", ""))
    parser.add_argument("--routes", default=str(PROJECT_ROOT / "leaderboard/data/bench2drive220.xml"))
    parser.add_argument("--fixed-route-idx", default=-1, type=int)
    parser.add_argument("--fixed-route-name", default="")
    parser.add_argument(
        "--deterministic-rollout",
        action="store_true",
        help="Use argmax planning modes; intended for no-update parity, not exploration training.",
    )
    parser.add_argument("--sensor-packet-timeout", default=30.0, type=float)
    parser.add_argument("--sensor-packet-log-interval", default=2.0, type=float)
    parser.add_argument("--sensor-packet-grace-seconds", default=0.5, type=float)
    parser.add_argument(
        "--sensor-packet-max-lag-frames",
        default=0,
        type=int,
        help="Reject stale sensor packets older than this many frames; 0 enforces exact current-frame packets.",
    )

    parser.add_argument("--max-train-steps", default=1000000, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--learning-starts", default=5000, type=int)
    parser.add_argument("--train-every-n-steps", default=100, type=int)
    parser.add_argument("--gradient-steps", default=50, type=int)
    parser.add_argument("--policy-learning-starts", default=5000, type=int)
    parser.add_argument("--policy-update-every-n-steps", default=1, type=int)
    parser.add_argument("--min-critic-updates-before-policy", default=10, type=int)
    parser.add_argument("--max-policy-q-loss-ema", default=5.0, type=float)
    parser.add_argument("--q-loss-ema-beta", default=0.95, type=float)
    parser.add_argument("--replay-capacity", default=200000, type=int)
    parser.add_argument("--replay-max-storage-gb", default=200.0, type=float)
    parser.add_argument(
        "--replay-mmap-dir",
        default="",
        help="Must resolve to ADAPTDRIVE_RUN_ROOT/replay/EXPERIMENT_ID.",
    )
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--checkpoint-every", default=1000, type=int)
    parser.add_argument("--log-dir", default="")
    parser.add_argument(
        "--init-from",
        default="",
        help="Start a fresh v8 run from the one registered legacy-v7 parent checkpoint.",
    )
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--hidden-dim", default=256, type=int)
    parser.add_argument("--policy-lr", default=1e-5, type=float)
    parser.add_argument("--critic-lr", default=3e-4, type=float)
    parser.add_argument("--alpha-lr", default=3e-4, type=float)
    parser.add_argument(
        "--target-entropy",
        default=-1.0,
        type=float,
        help=(
            "SAC entropy target over HiP-AD planning modes. "
            "Ignored when config.learnable_temperature=False; a negative value keeps the config default path."
        ),
    )
    parser.add_argument("--max-grad-norm", default=5.0, type=float)
    parser.add_argument("--critic-loss-type", default="huber")
    parser.add_argument("--critic-huber-delta", default=1.0, type=float)
    parser.add_argument(
        "--adapter-mode",
        default="dcnv4_feature",
        choices=("dcnv4_feature",),
        help="Frozen AdaptDrive adapter mode.",
    )
    parser.add_argument(
        "--enable-ego-state-adapter",
        action="store_true",
        help="Deprecated alias for --adapter-mode plan_query.",
    )
    parser.add_argument("--ego-adapter-feature-dim", default=256, type=int)
    parser.add_argument("--ego-adapter-ego-state-dim", default=21, type=int)
    parser.add_argument("--ego-adapter-hidden-dim", default=256, type=int)
    parser.add_argument("--ego-adapter-ego-hidden-dim", default=0, type=int)
    parser.add_argument("--ego-adapter-residual-scale", default=1.0, type=float)
    parser.add_argument("--ego-adapter-dropout", default=0.0, type=float)
    parser.add_argument("--disable-ego-adapter-layer-norm", action="store_false", dest="ego_adapter_use_layer_norm")
    parser.set_defaults(ego_adapter_use_layer_norm=True)
    parser.add_argument(
        "--enable-feature-dcnv4-adapter",
        action="store_true",
        help="Alias for --adapter-mode dcnv4_feature unless --adapter-mode is explicitly set.",
    )
    parser.add_argument("--feature-adapter-levels", nargs="+", default=(0, 1, 2, 3), type=int)
    parser.add_argument("--feature-adapter-residual-scale", default=1.0, type=float)
    parser.add_argument("--disable-feature-adapter-zero-init", action="store_false", dest="feature_adapter_zero_init")
    parser.set_defaults(feature_adapter_zero_init=True)
    parser.add_argument("--feature-adapter-feature-dim", default=256, type=int)
    parser.add_argument("--feature-adapter-ego-state-dim", default=21, type=int)
    parser.add_argument("--feature-adapter-ego-hidden-dim", default=0, type=int)
    parser.add_argument("--feature-adapter-bottleneck-reduction", default=4, type=int)
    parser.add_argument("--feature-adapter-dcn-group", default=0, type=int)
    parser.add_argument(
        "--feature-adapter-norm-type",
        default="group",
        choices=("batch", "group", "layer", "identity"),
        help="Normalization inside the DCNv4 bottleneck; group avoids unstable rollout-time BN.",
    )
    parser.add_argument("--feature-adapter-norm-groups", default=8, type=int)
    parser.add_argument("--adapter-prediction-enabled", action="store_true")
    parser.add_argument("--adapter-prediction-train-reward", action="store_true")
    parser.add_argument("--adapter-prediction-train-semantic", action="store_true")
    parser.add_argument("--adapter-prediction-every-n-steps", default=1, type=int)
    parser.add_argument(
        "--disable-adapter-prediction-forward-cache",
        action="store_false",
        dest="adapter_prediction_reuse_forward_cache",
    )
    parser.set_defaults(adapter_prediction_reuse_forward_cache=True)
    parser.add_argument(
        "--adapter-prediction-update-mode",
        default="prediction_only",
        choices=("prediction_only",),
    )
    parser.add_argument("--adapter-prediction-lr", default=3e-5, type=float)
    parser.add_argument("--prediction-head-lr", default=1e-4, type=float)
    parser.add_argument("--adapter-prediction-weight-decay", default=1e-4, type=float)
    parser.add_argument("--adapter-prediction-max-grad-norm", default=1.0, type=float)
    parser.add_argument("--adapter-prediction-reward-weight", default=1.0, type=float)
    parser.add_argument("--adapter-prediction-semantic-weight", default=1.0, type=float)
    parser.add_argument("--adapter-prediction-residual-weight", default=1e-3, type=float)
    parser.add_argument(
        "--adapter-prediction-semantic-route-weight",
        default=0.0,
        type=float,
        help="Keep route supervision off unless route conditioning is explicitly added.",
    )
    parser.add_argument("--roach-bev-map-root", default=os.environ.get("ROACH_BEV_MAP_ROOT", ""))
    parser.add_argument("--roach-bev-target-debug-dir", default="")
    parser.add_argument("--roach-bev-target-debug-interval", default=0, type=int)
    parser.add_argument("--roach-bev-target-debug-max-frames", default=100, type=int)
    parser.add_argument("--reference-kl-weight", default=0.3, type=float)
    parser.add_argument("--reference-kl-final-weight", default=0.3, type=float)
    parser.add_argument(
        "--trajectory-trust-region-weight",
        default=1.0,
        type=float,
        help="Active Huber trust-region weight on regressed spat-2m candidates; do not decay to zero.",
    )
    parser.add_argument(
        "--reference-traj-weight",
        default=0.0,
        type=float,
        help="Deprecated compatibility option; ignored. Use --trajectory-trust-region-weight.",
    )
    parser.add_argument(
        "--reference-traj-final-weight",
        default=0.0,
        type=float,
        help="Deprecated compatibility option; ignored. The active trajectory trust region does not decay.",
    )
    parser.add_argument("--reference-decay-steps", default=1, type=int)
    parser.add_argument(
        "--detach-policy-q-candidates",
        action="store_true",
        dest="detach_policy_q_candidates",
        help="Block SAC Q gradients from updating the spat-2m regression candidates.",
    )
    parser.add_argument(
        "--allow-policy-q-candidate-grad",
        action="store_false",
        dest="detach_policy_q_candidates",
        help="Deprecated compatibility alias; candidate Q gradients are enabled by default.",
    )
    parser.set_defaults(detach_policy_q_candidates=False)
    parser.add_argument("--store-sensor-mismatch-replay", action="store_false", dest="skip_sensor_mismatch_replay")
    parser.add_argument("--store-invalid-terminal-replay", action="store_false", dest="skip_invalid_terminal_replay")
    parser.set_defaults(skip_sensor_mismatch_replay=True, skip_invalid_terminal_replay=True)
    parser.add_argument("--replay-skip-log-every", default=100, type=int)
    parser.add_argument("--disable-episode-checkpoint", action="store_false", dest="checkpoint_on_episode_end")
    parser.set_defaults(checkpoint_on_episode_end=True)
    parser.add_argument(
        "--checkpoint-latest-before-reset",
        action="store_true",
        help=(
            "Overwrite checkpoint_latest.pt after each completed episode before env.reset(). "
            "This limits progress loss when CARLA aborts during route/scenario reset without "
            "creating a separate checkpoint file per episode."
        ),
    )
    parser.set_defaults(checkpoint_latest_before_reset=False)
    parser.add_argument("--reward-variant", default="line_e", choices=("line_e",))
    parser.add_argument("--enable-line-e-reward", action="store_true")
    parser.add_argument("--terminal-penalty-blocked", default=-10.0, type=float)
    parser.add_argument("--terminal-penalty-timeout", default=-10.0, type=float)
    parser.add_argument("--line-e-hard-stuck-terminal-penalty", default=-10.0, type=float)
    parser.add_argument(
        "--disable-line-e-bbox-safety-wait",
        action="store_false",
        dest="line_e_bbox_safety_wait",
        help="Disable the Line-E-only bbox-aware front-vehicle safety-wait classifier.",
    )
    parser.set_defaults(line_e_bbox_safety_wait=True)
    parser.add_argument("--line-e-safety-wait-clearance", default=9.5, type=float)
    parser.add_argument("--line-e-safety-wait-lateral-margin", default=0.5, type=float)
    parser.add_argument("--line-e-safe-wait-stop-speed", default=0.2, type=float)
    parser.add_argument("--line-e-safe-wait-reward-grace-steps", default=20, type=int)
    parser.add_argument("--line-e-blocked-timeout-steps", default=200, type=int)
    parser.add_argument("--free-road-efficiency-scale", default=0.20, type=float)
    parser.add_argument("--low-speed-threshold", default=1.0, type=float)
    parser.add_argument("--low-speed-grace-steps", default=20, type=int)
    parser.add_argument("--low-speed-penalty-per-step", default=-0.03, type=float)
    parser.add_argument("--low-speed-episode-cap", default=-4.0, type=float)
    parser.add_argument("--no-progress-threshold", default=1e-4, type=float)
    parser.add_argument("--no-progress-grace-steps", default=30, type=int)
    parser.add_argument("--no-progress-penalty-per-step", default=-0.02, type=float)
    parser.add_argument("--no-progress-episode-cap", default=-4.0, type=float)
    parser.add_argument("--enable-comfort-penalty", action="store_true")
    parser.add_argument("--comfort-penalty-max", default=-0.03, type=float)
    parser.add_argument("--enable-direct-dense-safety", action="store_true")
    parser.add_argument("--disable-dense-ttc", action="store_false", dest="enable_dense_ttc")
    parser.add_argument("--disable-dense-headway", action="store_false", dest="enable_dense_headway")
    parser.add_argument("--disable-dense-min-distance", action="store_false", dest="enable_dense_min_distance")
    parser.set_defaults(
        enable_line_e_reward=True,
        enable_direct_dense_safety=True,
        adapter_prediction_enabled=True,
        adapter_prediction_train_reward=True,
        adapter_prediction_train_semantic=True,
        enable_dense_ttc=True,
        enable_dense_headway=True,
        enable_dense_min_distance=True,
    )
    parser.add_argument("--direct-dense-ttc-weight", default=1.0, type=float)
    parser.add_argument("--direct-dense-headway-weight", default=1.0, type=float)
    parser.add_argument("--direct-dense-min-distance-weight", default=1.0, type=float)
    parser.add_argument("--direct-dense-forward-distance", default=35.0, type=float)
    parser.add_argument("--direct-dense-lateral-distance", default=2.7, type=float)
    parser.add_argument("--direct-dense-same-direction-dot", default=0.3, type=float)
    parser.add_argument("--direct-dense-min-closing-speed", default=0.1, type=float)
    parser.add_argument("--ttc-safe-time", default=4.0, type=float)
    parser.add_argument("--ttc-min-time", default=0.5, type=float)
    parser.add_argument("--dense-ttc-penalty-max", default=-0.40, type=float)
    parser.add_argument("--headway-time", default=1.5, type=float)
    parser.add_argument("--headway-min-distance", default=4.0, type=float)
    parser.add_argument("--dense-headway-penalty-max", default=-0.20, type=float)
    parser.add_argument("--min-distance-safe", default=3.0, type=float)
    parser.add_argument("--dense-min-distance-penalty-max", default=-0.10, type=float)
    parser.add_argument("--direct-dense-safety-penalty-cap", default=-0.60, type=float)
    parser.add_argument("--route-switch-interval", default=5, type=int)
    return _normalize_run_paths(parser.parse_args())


def train(args) -> None:
    if not args.carla_root:
        raise ValueError("CARLA_ROOT or --carla-root must point to the CARLA 0.9.15 installation")
    if args.init_from and args.resume_from:
        raise ValueError("--init-from and --resume-from are mutually exclusive")
    ensure_carla_python_paths(args.carla_root)

    if not args.hipad_checkpoint:
        raise ValueError("--hipad-checkpoint or HIPAD_CKPT must point to a HiP-AD checkpoint")
    clean_root = activate_hipad_project_root(args.hipad_root, repo_root=REPO_ROOT)
    config_path = validate_runtime_asset(args.hipad_config, label="HiP-AD config")
    checkpoint_path = validate_hipad_checkpoint_asset(
        args.hipad_checkpoint,
        label="HiP-AD checkpoint",
        reject_symlink=True,
        checkpoint_role=args.hipad_checkpoint_role,
        repo_root=REPO_ROOT,
    )
    validate_hipad_checkpoint_role(checkpoint_path, args.hipad_checkpoint_role)
    try:
        config_path.relative_to(clean_root)
    except ValueError:
        raise ValueError(f"HiP-AD config must be inside {clean_root}, got {config_path}")
    args.hipad_root = str(clean_root)
    args.hipad_config = str(config_path)
    args.hipad_checkpoint = str(checkpoint_path)
    checkpoint_origin = hipad_checkpoint_asset_origin(checkpoint_path, repo_root=REPO_ROOT)

    config = make_config(args)
    config.hipad_checkpoint_asset_origin = checkpoint_origin
    if config.control_semantics != CLEAN_DUAL_TRAJECTORY_CONTROL:
        raise RuntimeError(
            f"Line-C control semantics mismatch: config={config.control_semantics!r}, "
            f"runtime={CLEAN_DUAL_TRAJECTORY_CONTROL!r}"
        )
    if int(config.replay_schema_version) != CLEAN_DUAL_TRAJECTORY_SCHEMA_VERSION:
        raise RuntimeError(
            f"Line-C replay schema mismatch: config={config.replay_schema_version}, "
            f"runtime={CLEAN_DUAL_TRAJECTORY_SCHEMA_VERSION}"
        )
    env_config = make_env_config(args, config)
    signature = compute_training_signature(env_config, config)
    legacy_initialization = None
    if args.init_from:
        legacy_initialization = import_registered_legacy_parent(
            args.init_from,
            base_checkpoint_path=config.hipad_checkpoint_path,
            route_path=env_config.routes,
            hipad_root=config.hipad_project_root,
            target_training_signature=signature,
        )
    from rl.adaptdrive_env import AdaptDriveBench2DriveSACEnv as Bench2DriveSACEnv
    from rl.hipad_policy_finetune_agent import HiPADPolicyFinetuneAgent
    from rl.roach_reward import RoachRewardMonitor

    startup_provenance = collect_hipad_provenance(config.hipad_project_root)
    startup_provenance["checkpoint.path"] = str(checkpoint_path)
    startup_provenance["checkpoint.role"] = config.hipad_checkpoint_role
    startup_provenance["checkpoint.asset_origin"] = config.hipad_checkpoint_asset_origin
    startup_provenance["dual_pid_constructor_args"] = json.dumps(CLEAN_DUAL_PID_KWARGS, sort_keys=True)

    logger = StableLogger(config.log_dir)
    logger.log("=" * 70)
    logger.log("HiP-AD Planning Policy Closed-Loop Finetuning")
    logger.log(f"HiP-AD root: {config.hipad_project_root}")
    logger.log(f"HiP-AD config: {config.hipad_config_path}")
    logger.log(f"HiP-AD checkpoint: {config.hipad_checkpoint_path}")
    logger.log(f"Policy anchor type: {config.policy_anchor_type}")
    logger.log(f"Rollout mode selection: {'argmax' if config.deterministic_rollout else 'sampling'}")
    logger.log(
        f"Control semantics: {config.control_semantics} "
        f"replay_schema={config.replay_schema_version} "
        f"training_signature_version={TRAINING_SIGNATURE_VERSION}"
    )
    logger.log(f"Trajectory trust-region weight: {config.trajectory_trust_region_weight}")
    logger.log(
        "Actor update gates: "
        f"learning_starts={config.policy_learning_starts} "
        f"critic_updates_before_policy={config.min_critic_updates_before_policy} "
        f"update_every_critic_steps={config.policy_update_every_n_steps} "
        f"max_q_loss_ema={config.max_policy_q_loss_ema} "
        f"q_loss_ema_beta={config.q_loss_ema_beta}"
    )
    logger.log(
        "Actor candidate gradient: "
        f"{'detached' if config.detach_policy_q_candidates else 'enabled_to_final_spat_2m_reg'}"
    )
    logger.log(
        "Plan-query adapter: "
        f"adapter_mode={config.adapter_mode} "
        f"enabled={config.enable_ego_state_adapter} "
        f"feature_dim={config.ego_adapter_feature_dim} "
        f"ego_dim={config.ego_adapter_ego_state_dim} "
        f"hidden_dim={config.ego_adapter_hidden_dim} "
        f"residual_scale={config.ego_adapter_residual_scale}"
    )
    logger.log(
        "Feature DCNv4 adapter: "
        f"enabled={config.enable_feature_dcnv4_adapter} "
        f"levels={config.feature_adapter_levels} "
        f"residual_scale={config.feature_adapter_residual_scale} "
        f"zero_init={config.feature_adapter_zero_init} "
        f"norm={config.feature_adapter_norm_type} "
        "update_owner=adapter_prediction_only"
    )
    logger.log(
        "Adapter prediction: "
        f"enabled={config.adapter_prediction_enabled} "
        f"reward={config.adapter_prediction_train_reward} "
        f"semantic={config.adapter_prediction_train_semantic} "
        f"mode={config.adapter_prediction_update_mode} "
        f"every={config.adapter_prediction_every_n_steps} "
        f"reuse_cache={config.adapter_prediction_reuse_forward_cache} "
        f"adapter_lr={config.adapter_prediction_lr} "
        f"head_lr={config.prediction_head_lr} "
        f"semantic_route_w={config.adapter_prediction_semantic_route_weight} "
        f"roach_root={config.roach_bev_map_root or '<env/default>'}"
    )
    logger.log(f"Routes: {env_config.routes}")
    logger.log(f"Fixed route idx/name: {env_config.fixed_route_idx}/{env_config.fixed_route_name or '<none>'}")
    logger.log(
        "Reward: "
        f"variant={env_config.reward.reward_variant} "
        f"line_e={env_config.reward.enable_line_e_reward} "
        f"bbox_safe_wait={env_config.reward.line_e_bbox_safety_wait} "
        f"safe_wait_clearance={env_config.reward.line_e_safety_wait_clearance} "
        f"safe_wait_lateral_margin={env_config.reward.line_e_safety_wait_lateral_margin} "
        f"safe_wait_reward_grace={env_config.reward.line_e_safe_wait_reward_grace_steps} "
        f"blocked_timeout_steps={env_config.reward.line_e_blocked_timeout_steps} "
        f"direct_dense_safety={env_config.reward.enable_direct_dense_safety} "
        f"dense_cap={env_config.reward.dense_safety_penalty_cap}"
    )
    logger.log(f"Launch CARLA: {env_config.simulation.launch_server}")
    logger.log(
        "Sensor packet wait: "
        f"timeout={env_config.sensor_packet_timeout}s "
        f"log_interval={env_config.sensor_packet_log_interval}s "
        f"grace={env_config.sensor_packet_grace_seconds}s "
        f"max_lag_frames={env_config.sensor_packet_max_lag_frames}"
    )
    for name, source in startup_provenance.items():
        logger.log(f"Startup HiP-AD provenance {name}: {source}")
    logger.log("=" * 70)

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint = checkpoint_dir / "checkpoint_latest.pt"
    resume_path = Path(args.resume_from) if args.resume_from else None
    checkpoint = None
    total_step = 0
    total_episode = 0
    if resume_path is not None:
        resume_path = resume_path.expanduser().resolve()
        if not resume_path.is_file() or resume_path.is_symlink():
            raise FileNotFoundError(f"v8 resume checkpoint must be a non-symlink regular file: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise RuntimeError("v8 resume checkpoint must contain a mapping")
        required_checkpoint_keys = {
            "agent",
            "trainer_state",
            "replay_ref",
            "initialization",
            "checkpoint_uuid",
        }
        missing_checkpoint_keys = sorted(required_checkpoint_keys - set(checkpoint))
        if missing_checkpoint_keys:
            raise RuntimeError(f"v8 resume checkpoint is incomplete: missing={missing_checkpoint_keys}")
        if int(checkpoint.get("checkpoint_version", 0)) != 2:
            raise RuntimeError("full resume requires AdaptDrive checkpoint_version=2")
        if int(checkpoint.get("training_signature_version", 0)) != TRAINING_SIGNATURE_VERSION:
            raise RuntimeError("full resume requires an AdaptDrive signature-v8 checkpoint")
        if str(checkpoint.get("training_signature", "")) != signature:
            raise RuntimeError(
                "full resume training signature mismatch; "
                "use --init-from only for the registered v7 parent"
            )
        if str(checkpoint.get("experiment_id", "")) != args.experiment_id:
            raise RuntimeError("full resume experiment_id mismatch")
        if "replay_state" in checkpoint:
            raise RuntimeError("v8 full resume checkpoint must use replay_ref, not inline replay_state metadata")
        total_step = int(checkpoint.get("step", -1))
        total_episode = int(checkpoint.get("episode", -1))
        if total_step < 0 or total_episode < 0:
            raise RuntimeError("v8 resume checkpoint has invalid trainer counters")
        logger.log(
            f"Strict v8 full resume: step={total_step} episode={total_episode} "
            f"checkpoint_uuid={checkpoint['checkpoint_uuid']}"
        )

    if checkpoint is not None:
        initialization_provenance = checkpoint["initialization"]
        if not isinstance(initialization_provenance, dict):
            raise RuntimeError("v8 resume checkpoint initialization provenance must be a mapping")
    elif legacy_initialization is not None:
        initialization_provenance = legacy_initialization.provenance()
        logger.log(
            "Fresh v8 initialization from registered legacy parent: "
            f"parent_step={legacy_initialization.parent_step} parent_episode={legacy_initialization.parent_episode} "
            "new_step=0 new_episode=0 fresh_optimizers=1 fresh_replay=1"
        )
    else:
        initialization_provenance = {
            "profile": "fresh_hipad_base_v8",
            "target_training_signature_version": TRAINING_SIGNATURE_VERSION,
            "target_training_signature": signature,
            "new_step": 0,
            "new_episode": 0,
            "fresh_optimizers": True,
            "fresh_replay": True,
        }
    expected_initialization = {
        "target_training_signature_version": TRAINING_SIGNATURE_VERSION,
        "target_training_signature": signature,
        "new_step": 0,
        "new_episode": 0,
    }
    initialization_mismatch = {
        key: (initialization_provenance.get(key), expected)
        for key, expected in expected_initialization.items()
        if initialization_provenance.get(key) != expected
    }
    if initialization_mismatch or not str(initialization_provenance.get("profile", "")):
        raise RuntimeError(
            f"checkpoint initialization provenance mismatch: {initialization_mismatch}"
        )

    env = None
    replay = None
    replay_context = None
    try:
        logger.log("Creating HiP-AD policy-finetune agent")
        agent = HiPADPolicyFinetuneAgent(config)
        provenance = dict(startup_provenance)
        provenance.update(agent._input_adapter.runtime_asset_provenance)
        provenance["checkpoint.sha256"] = hash_file(config.hipad_checkpoint_path)
        agent.runtime_provenance = dict(provenance)
        for name, source in provenance.items():
            logger.log(f"HiP-AD provenance {name}: {source}")
        logger.log(f"Trainable HiP-AD planning params: {agent.trainable_parameter_count:,}")
        logger.log(
            "Plan-query adapter params: "
            f"{agent.plan_query_adapter_parameter_count:,} "
            f"(enabled={agent.plan_query_adapter_enabled})"
        )
        logger.log(
            "Feature DCNv4 adapter params: "
            f"{agent.feature_dcnv4_adapter_parameter_count:,} "
            f"(enabled={agent.feature_dcnv4_adapter_enabled}, levels={config.feature_adapter_levels})"
        )
        logger.log(f"SAC target entropy: {agent.target_entropy:.4f}")
        if legacy_initialization is not None:
            apply_registered_legacy_parent(
                agent,
                legacy_initialization,
                current_training_signature=signature,
            )
            del legacy_initialization
        elif checkpoint is not None:
            try:
                agent.load_state_dict(checkpoint["agent"], load_optimizers=True, strict=True)
            except Exception as exc:
                raise RuntimeError(
                    "Strict v8 agent/optimizer restore failed; refusing partial resume."
                ) from exc

        replay_budget_bytes = int(config.replay_max_storage_gb * (1024 ** 3))
        replay_shapes = {
            "state_shape": (config.state_dim,),
            "actor_base_shape": (config.feature_dim,),
            "critic_bev_shape": (config.feature_dim,),
            "trajectory_shape": (config.fut_ts, 2),
            "pid_summary_dim": DUAL_PID_SUMMARY_DIM,
            "control_semantics": CLEAN_DUAL_TRAJECTORY_CONTROL,
        }
        replay_capacity_by_budget = StrictFeatureReplayBuffer.capacity_for_storage_budget(
            replay_budget_bytes,
            **replay_shapes,
        )
        replay_capacity = min(config.replay_capacity, replay_capacity_by_budget)
        if checkpoint is not None:
            replay, replay_context = resume_feature_replay(
                checkpoint,
                replay_root=config.replay_mmap_dir,
                experiment_id=args.experiment_id,
                training_signature=signature,
                capacity=replay_capacity,
                **replay_shapes,
            )
            logger.log(
                f"Strict replay restored: replay_uuid={replay_context.replay_uuid} "
                f"size={len(replay)} capacity={replay.capacity}"
            )

        logger.log("Creating environment")
        env = Bench2DriveSACEnv(env_config)
        if hasattr(env, "set_route_switch_interval"):
            env.set_route_switch_interval(args.route_switch_interval)
        reward_monitor = RoachRewardMonitor(eval_mode=False)
        pid_controller = agent.create_rollout_pid_controller()
        observation, info = env.reset()
        if tuple(observation["state"].shape) != tuple(replay_shapes["state_shape"]):
            raise RuntimeError(
                f"runtime observation state shape changed: {observation['state'].shape} "
                f"!= {replay_shapes['state_shape']}"
            )
        reward_monitor.reset(env)
        bind_hipad_after_reset(env, agent, logger)
        prime_hipad_after_reset(env, agent, logger)
        current_policy_output = agent.forward_policy(
            observation,
            deterministic=config.deterministic_rollout,
            include_reference=False,
        )
        log_clean_navigation_context(current_policy_output, logger, total_episode + 1)
        agent.begin_rollout_episode(observation, total_step, total_episode + 1)
        if replay is None:
            logger.log(f"Creating fresh UUID replay: capacity={replay_capacity}")
            replay, replay_context = create_feature_replay(
                replay_root=config.replay_mmap_dir,
                experiment_id=args.experiment_id,
                training_signature=signature,
                capacity=replay_capacity,
                **replay_shapes,
            )
            logger.log(f"Fresh replay UUID: {replay_context.replay_uuid}")

        current_prev_pid_summary = None
        current_prev_pid_mask = 0.0
        episode_reward = 0.0
        episode_length = 0
        episode_reward_parts = {key: 0.0 for key in REWARD_LOG_KEYS}
        restored_trainer_state = (
            checkpoint.get("trainer_state", {}) if checkpoint is not None else {}
        )
        critic_update_count = int(restored_trainer_state.get("critic_update_count", 0))
        policy_update_count = int(restored_trainer_state.get("policy_update_count", 0))
        critic_q_loss_ema = restored_trainer_state.get("critic_q_loss_ema")
        critic_q_loss_ema = None if critic_q_loss_ema is None else float(critic_q_loss_ema)
        replay_skipped = int(restored_trainer_state.get("replay_skipped", 0))
        last_policy_skip_reason = str(restored_trainer_state.get("last_policy_skip_reason", ""))

        def trainer_state_payload() -> dict:
            return {
                "critic_update_count": int(critic_update_count),
                "policy_update_count": int(policy_update_count),
                "critic_q_loss_ema": None if critic_q_loss_ema is None else float(critic_q_loss_ema),
                "replay_skipped": int(replay_skipped),
                "last_policy_skip_reason": last_policy_skip_reason,
            }

        while total_step < config.max_train_steps:
            observation["scene_token"] = observation.get(
                "scene_token",
                info.get("scene_token", info.get("route_name", "hipad_policy_finetune_scene")),
            )
            semantic_target_t = None
            if bool(config.adapter_prediction_enabled and config.adapter_prediction_train_semantic):
                try:
                    semantic_target_t = env.pop_roach_bev_target(
                        expected_frame=int(observation.get("sensor_frame", -1))
                    )
                except Exception as exc:
                    logger.log(f"Roach BEV target pop failed at step={total_step}: {exc}", level="WARNING")
                    semantic_target_t = None
            trajectory = np.clip(
                agent.trajectory_numpy(current_policy_output),
                -config.base_plan_clip,
                config.base_plan_clip,
            )
            speed_trajectory = np.clip(
                agent.speed_trajectory_numpy(current_policy_output),
                -config.base_plan_clip,
                config.base_plan_clip,
            )
            speed = np.asarray(info.get("speed", 0.0), dtype=np.float32)
            target = clean_control_target_from_policy_output(current_policy_output)
            action, pid_metadata = clean_dual_pid_step(
                pid_controller,
                speed_trajectory,
                trajectory,
                speed,
                target,
            )
            current_pid_summary = pack_clean_dual_pid_summary(action, pid_metadata, pid_controller)
            adapter_prediction_action_summary = agent.build_adapter_prediction_action_vector(
                current_policy_output,
                action,
            )

            next_observation, reward, terminated, truncated, next_info = env.step(action)
            next_info.update(reward_monitor.compute(next_info))
            reward = float(reward)
            done = bool(terminated or truncated)
            transition_terminal = bool(terminated or next_info.get("terminal_for_replay", False))

            total_step += 1
            episode_length += 1
            episode_reward += float(reward)
            for key in REWARD_LOG_KEYS:
                episode_reward_parts[key] += safe_float(next_info.get(key, 0.0))

            critic_metrics = None
            policy_metrics = None
            adapter_prediction_metrics = agent.update_adapter_prediction_from_step(
                reward=reward,
                reward_info=next_info,
                semantic_target=semantic_target_t,
                action_summary=adapter_prediction_action_summary,
                total_step=total_step,
            )
            if len(replay) >= max(config.learning_starts, config.batch_size):
                if total_step % config.train_every_n_steps == 0:
                    for _ in range(config.gradient_steps):
                        batch = replay.sample(config.batch_size, device=agent.device)
                        critic_metrics = agent.update_critic_value_from_feature_batch(batch)
                        critic_update_count += 1
                        q_loss = safe_float(critic_metrics.get("critic_q_loss", 0.0))
                        if critic_q_loss_ema is None:
                            critic_q_loss_ema = q_loss
                        else:
                            beta = min(0.9999, max(0.0, float(config.q_loss_ema_beta)))
                            critic_q_loss_ema = beta * critic_q_loss_ema + (1.0 - beta) * q_loss
                        update_policy, skip_reason = policy_update_decision(
                            total_step=total_step,
                            critic_update_count=critic_update_count,
                            critic_q_loss_ema=critic_q_loss_ema,
                            policy_learning_starts=config.policy_learning_starts,
                            policy_update_every_n_steps=config.policy_update_every_n_steps,
                            min_critic_updates_before_policy=config.min_critic_updates_before_policy,
                            max_policy_q_loss_ema=config.max_policy_q_loss_ema,
                        )
                        if update_policy:
                            policy_metrics = agent.update_policy_from_feature_batch(batch, total_step=total_step)
                            policy_update_count += 1
                            last_policy_skip_reason = ""
                        else:
                            last_policy_skip_reason = skip_reason

                # Freeze verification: every 1000 steps, confirm that frozen
                # parameters have zero gradient.
                if total_step > 0 and total_step % 1000 == 0:
                    leak_count = 0
                    for name, param in agent._model.named_parameters():
                        if param.requires_grad:
                            continue
                        if param.grad is not None and param.grad.norm() > 1e-8:
                            leak_count += 1
                            logger.log(
                                f"FREEZE_LEAK: {name} grad_norm={param.grad.norm():.6f}",
                                level="ERROR",
                            )
                    if leak_count == 0:
                        logger.log(
                            f"[FreezeCheck] step={total_step}: all frozen params have zero grad ✓",
                        )

            if transition_terminal:
                next_policy_output = None
                next_critic_features = np.zeros((config.feature_dim,), dtype=np.float32)
            else:
                next_policy_output = agent.forward_policy(
                    next_observation,
                    deterministic=config.deterministic_rollout,
                )
                next_critic_features = next_policy_output.feature_np.copy()

            should_store_transition, replay_skip_reason = replay_store_decision(
                info,
                next_info,
                transition_terminal,
                config,
            )
            next_info["replay_transition_stored"] = should_store_transition
            next_info["replay_transition_skipped"] = not should_store_transition
            next_info["replay_skip_reason"] = replay_skip_reason
            if should_store_transition:
                replay_navigation = clean_replay_navigation(current_policy_output)
                next_replay_navigation = (
                    clean_replay_navigation(next_policy_output)
                    if next_policy_output is not None
                    else {
                        "target_point": np.zeros(2, dtype=np.float32),
                        "command": int(replay_navigation["command"]),
                    }
                )
                replay.add(
                    {
                        "state": np.asarray(observation["state"], dtype=np.float32).copy(),
                        "target_point": replay_navigation["target_point"],
                        "command": replay_navigation["command"],
                    },
                    current_policy_output.feature_np.copy(),
                    current_policy_output.feature_np.copy(),
                    trajectory.copy(),
                    current_pid_summary.copy(),
                    float(reward),
                    {
                        "state": np.asarray(next_observation["state"], dtype=np.float32).copy(),
                        "target_point": next_replay_navigation["target_point"],
                        "command": next_replay_navigation["command"],
                    },
                    next_critic_features,
                    transition_terminal,
                    prev_pid_summary=None if current_prev_pid_summary is None else current_prev_pid_summary.copy(),
                    plan_cls_context=current_policy_output.plan_cls_context_np,
                    all_candidates=current_policy_output.all_candidates_np,
                    longitudinal_trajectory=speed_trajectory.copy(),
                    candidate_longitudinal_trajectories=(
                        current_policy_output.longitudinal_candidates_np.copy()
                    ),
                    selected_lateral_mode=current_policy_output.mode_idx,
                    longitudinal_mode=current_policy_output.speed_mode_idx,
                )
            else:
                replay_skipped += 1
                log_every = max(1, int(config.replay_skip_log_every))
                if replay_skipped == 1 or replay_skipped % log_every == 0:
                    logger.log(
                        f"[ReplaySkip] step={total_step} skipped={replay_skipped} "
                        f"reason={replay_skip_reason} replay={len(replay)}"
                    )

            if total_step % 100 == 0:
                logger.log(
                    f"[Heartbeat] step={total_step} episode={total_episode + 1} "
                    f"return={episode_reward:.3f} length={episode_length} "
                    f"route={next_info.get('route_name')} progress={next_info.get('route_progress', 0.0):.4f} "
                    f"speed={next_info.get('speed', 0.0):.3f} replay={len(replay)} "
                    f"mean_speed_10={next_info.get('mean_speed', 0.0):.3f} "
                    f"reward_desired={next_info.get('desired_speed', 0.0):.3f} "
                    f"reward_desired_veh={next_info.get('desired_speed_vehicle', 0.0):.3f} "
                    f"legal_wait={int(bool(next_info.get('is_legal_wait', False)))} "
                    f"bbox_safe_wait={int(bool(next_info.get('bbox_safety_wait_active', False)))} "
                    f"free_road={int(bool(next_info.get('is_free_road', False)))} "
                    f"stuck_counter={int(next_info.get('vehicle_stuck_counter', 0))} "
                    f"blocked_wait_counter={int(next_info.get('blocked_wait_counter', 0))} "
                    f"blocker_id={int(next_info.get('safety_wait_blocker_id', -1))} "
                    f"blocker_center={next_info.get('safety_wait_blocker_center_distance', float('inf')):.3f} "
                    f"blocker_bbox_gap="
                    f"{next_info.get('safety_wait_blocker_longitudinal_clearance', float('inf')):.3f} "
                    f"blocker_lat_clearance="
                    f"{next_info.get('safety_wait_blocker_lateral_clearance', float('inf')):.3f} "
                    f"blocker_speed={next_info.get('safety_wait_blocker_speed', 0.0):.3f} "
                    f"pid_desired={safe_float(pid_metadata.get('desired_speed', 0.0)):.3f} "
                    f"pid_throttle={max(0.0, float(action[1])):.3f} "
                    f"pid_brake={max(0.0, -float(action[1])):.3f} "
                    f"speed_area_raw={int(current_policy_output.speed_raw_mode_idx)} "
                    f"speed_area_rescored={int(current_policy_output.speed_mode_idx)} "
                    f"speed_rescore_changed="
                    f"{int(current_policy_output.speed_rescore_changed_selected)} "
                    f"speed_suppressed_areas="
                    f"{int(current_policy_output.speed_suppressed_area_count_selected)} "
                    f"speed_feasible_modes={int(current_policy_output.speed_feasible_mode_count)} "
                    f"speed_modes_gt_0_4={int(current_policy_output.speed_modes_desired_gt_0_4)} "
                    f"speed_modes_gt_1_0={int(current_policy_output.speed_modes_desired_gt_1_0)} "
                    f"replay_skipped={replay_skipped} "
                    f"critic_updates={critic_update_count} "
                    f"policy_updates={policy_update_count} "
                    f"policy_skip={last_policy_skip_reason or '<none>'} "
                    f"q_loss_ema={format_optional_float(critic_q_loss_ema)} "
                    f"r_progress={next_info.get('r_progress', 0.0):.3f} "
                    f"r_terminal={next_info.get('r_terminal', 0.0):.3f} "
                    f"r_success={next_info.get('r_success', 0.0):.3f} "
                    f"r_dense_safety_direct={next_info.get('r_dense_safety_direct', 0.0):.3f}"
                )

            adapter_prediction_log_now = bool(
                adapter_prediction_metrics is not None
                and adapter_prediction_metrics.get("adapter_prediction_skipped", 1.0) == 0.0
                and (total_step <= 10 or total_step % 100 == 0)
            )
            if (
                critic_metrics is not None
                or policy_metrics is not None
                or adapter_prediction_log_now
            ):
                critic_metrics = critic_metrics or {}
                policy_metrics = policy_metrics or {}
                adapter_prediction_metrics = adapter_prediction_metrics or {}
                feature_adapter_level_text = format_feature_adapter_level_metrics(
                    current_policy_output.feature_adapter_metrics,
                    config.feature_adapter_levels,
                )
                feature_adapter_alpha_grad_text = format_feature_adapter_alpha_grad_metrics(
                    adapter_prediction_metrics,
                    config.feature_adapter_levels,
                )
                logger.log(
                    f"[Step {total_step}] replay={len(replay)} "
                    f"q_loss={critic_metrics.get('critic_q_loss', 0.0):.4f} "
                    f"v_loss={critic_metrics.get('critic_v_loss', 0.0):.4f} "
                    f"critic_updates={critic_update_count} "
                    f"q_loss_ema={format_optional_float(critic_q_loss_ema)} "
                    f"policy_loss={policy_metrics.get('policy_loss', 0.0):.4f} "
                    f"sac_loss={policy_metrics.get('sac_policy_loss', 0.0):.4f} "
                    f"policy_q={policy_metrics.get('policy_q', 0.0):.4f} "
                    f"alpha={policy_metrics.get('alpha', float(agent.alpha.item())):.4f} "
                    f"entropy={policy_metrics.get('traj_entropy', 0.0):.4f} "
                    f"ref_kl={policy_metrics.get('reference_kl_loss', 0.0):.4f} "
                    f"ref_kl_w={policy_metrics.get('reference_kl_weight', 0.0):.3f} "
                    f"trust_loss={policy_metrics.get('trajectory_trust_region_loss', 0.0):.6f} "
                    f"trust_w={policy_metrics.get('trajectory_trust_region_weight', 0.0):.3f} "
                    f"candidate_l2_mean={policy_metrics.get('candidate_l2_mean', 0.0):.4f} "
                    f"candidate_l2_max={policy_metrics.get('candidate_l2_max', 0.0):.4f} "
                    f"candidate_delta_h={policy_metrics.get('candidate_delta_per_horizon', '')} "
                    f"candidate_dx={policy_metrics.get('candidate_delta_x_mean', 0.0):.4f} "
                    f"candidate_dy={policy_metrics.get('candidate_delta_y_mean', 0.0):.4f} "
                    f"mode={current_policy_output.mode_idx} "
                    f"max_prob={policy_metrics.get('max_prob', current_policy_output.max_prob):.4f} "
                    f"selected_prob={policy_metrics.get('selected_prob', current_policy_output.selected_prob):.4f} "
                    f"logit_std={policy_metrics.get('logit_std', current_policy_output.logit_std):.4f} "
                    f"q_std={policy_metrics.get('q_std_modes', 0.0):.4f} "
                    f"q_gap={policy_metrics.get('q_gap_modes', 0.0):.4f} "
                    f"q_candidate_grad={policy_metrics.get('policy_q_candidate_grad', 0.0):.0f} "
                    f"policy_q_candidate_grad_enabled="
                    f"{policy_metrics.get('policy_q_candidate_grad_enabled', 0.0):.0f} "
                    f"policy_q_candidate_grad_norm="
                    f"{policy_metrics.get('policy_q_candidate_grad_norm', 0.0):.8e} "
                    f"policy_plan_cls_branch_grad_norm="
                    f"{policy_metrics.get('policy_plan_cls_branch_grad_norm', 0.0):.8e} "
                    f"policy_plan_spat_reg_branch_grad_norm="
                    f"{policy_metrics.get('policy_plan_spat_reg_branch_grad_norm', 0.0):.8e} "
                    f"decoder_mode={int(current_policy_output.decoder_lateral_mode)} "
                    f"selected_mode={int(current_policy_output.mode_idx)} "
                    f"speed_area_raw={int(current_policy_output.speed_raw_mode_idx)} "
                    f"speed_area_mode={int(current_policy_output.speed_mode_idx)} "
                    f"speed_rescore_changed_selected="
                    f"{int(current_policy_output.speed_rescore_changed_selected)} "
                    f"speed_rescore_changed_rate="
                    f"{float(current_policy_output.speed_rescore_changed_rate):.6f} "
                    f"speed_suppressed_areas_selected="
                    f"{int(current_policy_output.speed_suppressed_area_count_selected)} "
                    f"speed_suppressed_areas_mean="
                    f"{float(current_policy_output.speed_suppressed_area_count_mean):.3f} "
                    f"speed_feasible_modes={int(current_policy_output.speed_feasible_mode_count)} "
                    f"speed_modes_gt_0_4={int(current_policy_output.speed_modes_desired_gt_0_4)} "
                    f"speed_modes_gt_1_0={int(current_policy_output.speed_modes_desired_gt_1_0)} "
                    f"speed_all_col_selected={int(current_policy_output.speed_all_collision_selected)} "
                    f"speed_all_col_rate={float(current_policy_output.speed_all_collision_rate):.6f} "
                    f"plan_query_adapter={policy_metrics.get('plan_query_adapter_enabled', 0.0):.0f} "
                    f"plan_query_adapter_delta={policy_metrics.get('plan_query_adapter_delta_l2', 0.0):.6f} "
                    f"feature_dcnv4_adapter={float(agent.feature_dcnv4_adapter_enabled):.0f} "
                    f"{feature_adapter_level_text} "
                    f"{feature_adapter_alpha_grad_text} "
                    f"adapter_pred={adapter_prediction_metrics.get('adapter_prediction_enabled', 0.0):.0f} "
                    f"adapter_pred_loss={adapter_prediction_metrics.get('adapter_prediction_total_loss', 0.0):.4f} "
                    f"adapter_pred_reward={adapter_prediction_metrics.get('adapter_prediction_reward_loss', 0.0):.4f} "
                    f"adapter_pred_semantic={adapter_prediction_metrics.get('adapter_prediction_semantic_loss', 0.0):.4f} "
                    f"adapter_pred_residual={adapter_prediction_metrics.get('adapter_prediction_residual_loss', 0.0):.6f} "
                    f"adapter_pred_grad={adapter_prediction_metrics.get('adapter_prediction_grad_norm_adapter', 0.0):.4f} "
                    f"adapter_pred_reward_grad={adapter_prediction_metrics.get('adapter_prediction_grad_norm_reward_head', 0.0):.4f} "
                    f"adapter_pred_semantic_grad={adapter_prediction_metrics.get('adapter_prediction_grad_norm_semantic_head', 0.0):.4f} "
                    f"adapter_pred_sem_valid={adapter_prediction_metrics.get('adapter_pred_semantic_valid_rate', 0.0):.3f} "
                    f"adapter_pred_sem_no_target={adapter_prediction_metrics.get('adapter_pred_semantic_skip_no_target', 0.0):.0f} "
                    f"adapter_pred_sem_mismatch={adapter_prediction_metrics.get('adapter_pred_semantic_skip_frame_mismatch', 0.0):.0f} "
                    f"adapter_pred_sem_not_exact={adapter_prediction_metrics.get('adapter_pred_semantic_skip_sensor_not_exact', 0.0):.0f} "
                    f"projector_valid={adapter_prediction_metrics.get('projector_valid_bev_ratio', 0.0):.3f} "
                    f"grad_norm={policy_metrics.get('grad_norm', 0.0):.4f} "
                    f"r_dense_safety_direct={next_info.get('r_dense_safety_direct', 0.0):.4f} "
                    f"replay_skipped={replay_skipped}"
                )

            if total_step % config.checkpoint_every == 0:
                save_checkpoint(
                    agent,
                    replay,
                    replay_context,
                    total_step,
                    total_episode,
                    checkpoint_dir / f"checkpoint_{total_step:08d}.pt",
                    latest_checkpoint,
                    signature,
                    args.experiment_id,
                    logger,
                    trainer_state=trainer_state_payload(),
                    initialization=initialization_provenance,
                )

            if done:
                total_episode += 1
                logger.log(
                    f"[Episode {total_episode}] step={total_step} return={episode_reward:.3f} "
                    f"length={episode_length} route={next_info.get('route_name')} "
                    f"progress={next_info.get('route_progress', 0.0):.4f} "
                    f"termination={next_info.get('termination_reasons', next_info.get('termination'))} "
                    f"truncation={next_info.get('truncation_reason')}"
                )
                logger.log(
                    f"[Episode {total_episode} Safety] "
                    f"speed={next_info.get('speed', 0.0):.3f} "
                    f"mean_speed_10={next_info.get('mean_speed', 0.0):.3f} "
                    f"legal_wait={int(bool(next_info.get('is_legal_wait', False)))} "
                    f"bbox_safe_wait={int(bool(next_info.get('bbox_safety_wait_active', False)))} "
                    f"stuck_counter={int(next_info.get('vehicle_stuck_counter', 0))} "
                    f"blocked_wait_counter={int(next_info.get('blocked_wait_counter', 0))} "
                    f"blocker_id={int(next_info.get('safety_wait_blocker_id', -1))} "
                    f"blocker_center={next_info.get('safety_wait_blocker_center_distance', float('inf')):.3f} "
                    f"blocker_bbox_gap="
                    f"{next_info.get('safety_wait_blocker_longitudinal_clearance', float('inf')):.3f} "
                    f"blocker_lateral_clearance="
                    f"{next_info.get('safety_wait_blocker_lateral_clearance', float('inf')):.3f} "
                    f"blocker_speed={next_info.get('safety_wait_blocker_speed', 0.0):.3f} "
                    f"replay_stored={int(bool(next_info.get('replay_transition_stored', False)))} "
                    f"replay_skip_reason={next_info.get('replay_skip_reason', '') or '<none>'}"
                )
                logger.log(f"[Episode {total_episode} Reward] {format_reward_parts(episode_reward_parts)}")
                if config.checkpoint_latest_before_reset:
                    save_checkpoint(
                        agent,
                        replay,
                        replay_context,
                        total_step,
                        total_episode,
                        latest_checkpoint,
                        latest_checkpoint,
                        signature,
                        args.experiment_id,
                        logger,
                        trainer_state=trainer_state_payload(),
                        initialization=initialization_provenance,
                    )
                if config.checkpoint_on_episode_end:
                    save_checkpoint(
                        agent,
                        replay,
                        replay_context,
                        total_step,
                        total_episode,
                        checkpoint_dir / f"checkpoint_episode_{total_episode:06d}_step_{total_step:08d}.pt",
                        latest_checkpoint,
                        signature,
                        args.experiment_id,
                        logger,
                        trainer_state=trainer_state_payload(),
                        initialization=initialization_provenance,
                    )
                agent.reset_temporal_state()
                pid_controller = agent.create_rollout_pid_controller()
                observation, info = env.reset()
                reward_monitor.reset(env)
                bind_hipad_after_reset(env, agent, logger)
                prime_hipad_after_reset(env, agent, logger)
                current_policy_output = agent.forward_policy(
                    observation,
                    deterministic=config.deterministic_rollout,
                    include_reference=False,
                )
                log_clean_navigation_context(current_policy_output, logger, total_episode + 1)
                current_prev_pid_summary = None
                current_prev_pid_mask = 0.0
                episode_reward = 0.0
                episode_length = 0
                episode_reward_parts = {key: 0.0 for key in REWARD_LOG_KEYS}
                agent.begin_rollout_episode(observation, total_step, total_episode + 1)
            else:
                observation = next_observation
                info = next_info
                current_policy_output = next_policy_output
                current_prev_pid_summary = current_pid_summary
                current_prev_pid_mask = 1.0

        save_checkpoint(
            agent,
            replay,
            replay_context,
            total_step,
            total_episode,
            checkpoint_dir / f"checkpoint_final_{total_step:08d}.pt",
            latest_checkpoint,
            signature,
            args.experiment_id,
            logger,
            trainer_state=trainer_state_payload(),
            initialization=initialization_provenance,
        )
    finally:
        if replay is not None:
            replay.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    train(parse_args())
