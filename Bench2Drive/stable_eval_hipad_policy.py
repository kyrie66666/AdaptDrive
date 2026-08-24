#!/usr/bin/env python3
from __future__ import annotations

"""Frozen closed-loop evaluation for HiP-AD's native planning policy."""

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import torch

from stable_train_hipad_policy_finetune import (
    HIPAD_ROOT,
    PROJECT_ROOT,
    REPO_ROOT,
    StableLogger,
    bind_hipad_after_reset,
    ensure_carla_python_paths,
    hash_file,
    log_clean_navigation_context,
    prime_hipad_after_reset,
)

from rl.reward import RewardConfig

from rl.adaptdrive_replay import validate_experiment_id
from rl.adaptdrive_training_signature import TRAINING_SIGNATURE_VERSION
from rl.hipad_clean_adapter_checkpoint import EXPECTED_ADAPTER_MODE, FOUR_LEVELS
from rl.hipad_clean_control import clean_dual_pid_step
from rl.hipad_clean_navigation import clean_control_target_from_policy_output
from rl.hipad_project_runtime import (
    activate_hipad_project_root,
    collect_hipad_provenance,
    hipad_checkpoint_asset_origin,
    validate_hipad_checkpoint_asset,
    validate_hipad_checkpoint_role,
    validate_runtime_asset,
)
from rl.hipad_policy_finetune_config import HiPADPolicyFinetuneConfig


REWARD_KEYS = (
    "r_progress",
    "r_speed",
    "r_position",
    "r_rotation",
    "r_action",
    "r_terminal",
    "r_success",
    "r_stuck_soft",
    "r_stuck_terminal",
)
SUPPORTED_DEPLOYMENT_SIGNATURE_VERSIONS = (7, TRAINING_SIGNATURE_VERSION)


def _float_value(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _termination_items(info: Dict) -> Iterable[str]:
    reasons = info.get("termination_reasons", info.get("termination"))
    if reasons is None:
        return []
    if isinstance(reasons, (list, tuple, set)):
        return [str(item) for item in reasons]
    return [str(reasons)]


def _format_sums(sums: Dict[str, float]) -> str:
    return " ".join(f"{key}={sums.get(key, 0.0):.3f}" for key in REWARD_KEYS)


def _mean(values) -> float:
    return float(np.mean(values)) if values else 0.0


def make_config(args) -> HiPADPolicyFinetuneConfig:
    config = HiPADPolicyFinetuneConfig()
    config.hipad_project_root = args.hipad_root
    config.hipad_config_path = args.hipad_config
    config.hipad_checkpoint_path = args.hipad_checkpoint
    config.hipad_checkpoint_role = args.hipad_checkpoint_role
    config.strict_policy = bool(args.strict_policy)
    config.max_train_steps = args.max_eval_steps
    config.routes = args.routes or config.routes
    config.log_dir = args.log_dir
    return config


def make_eval_env_config(args, config: HiPADPolicyFinetuneConfig):
    """Frozen-eval RLEnvConfig builder.

    Deliberately independent of stable_train_hipad_policy_finetune.make_env_config:
    that helper reads ~100+ reward/adapter/optimizer CLI fields this eval script
    doesn't (and shouldn't) expose. Reward always uses RewardConfig()'s built-in
    legacy defaults so this can never diverge from the reward code as written.
    """
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
        reward=RewardConfig(),
        step_manager=StepManagerConfig(timeout=args.timeout, max_tick_count=config.max_episode_steps),
        max_episode_steps=config.max_episode_steps,
        random_routes=True,
        fixed_route_idx=args.fixed_route_idx,
        fixed_route_name=args.fixed_route_name,
        # Not exposed on this script's parser. Hardcoded to match the training
        # entry's own CLI defaults (strict current-frame sensor policy) rather
        # than RLEnvConfig's looser built-in defaults, so eval stays comparable
        # to a no-update SAC-wrapper rollout.
        sensor_packet_timeout=30.0,
        sensor_packet_log_interval=2.0,
        sensor_packet_grace_seconds=0.5,
        sensor_packet_max_lag_frames=0,
    )


def apply_finetune_checkpoint_config(config: HiPADPolicyFinetuneConfig, checkpoint: dict) -> dict:
    signature_version = int(checkpoint.get("training_signature_version", 0))
    if signature_version not in SUPPORTED_DEPLOYMENT_SIGNATURE_VERSIONS:
        raise RuntimeError(
            "finetune checkpoint signature version mismatch: "
            f"found {signature_version}, expected one of {SUPPORTED_DEPLOYMENT_SIGNATURE_VERSIONS}"
        )
    expected_checkpoint_version = 1 if signature_version == 7 else 2
    if int(checkpoint.get("checkpoint_version", 0)) != expected_checkpoint_version:
        raise RuntimeError(
            "finetune checkpoint version mismatch: "
            f"found {checkpoint.get('checkpoint_version')!r}, expected {expected_checkpoint_version}"
        )
    training_signature = str(checkpoint.get("training_signature", ""))
    if len(training_signature) != 64 or any(character not in "0123456789abcdef" for character in training_signature):
        raise RuntimeError("finetune checkpoint training_signature must be a 64-character lowercase SHA-256")
    agent_state = checkpoint.get("agent")
    if not isinstance(agent_state, Mapping):
        raise RuntimeError("finetune checkpoint must contain an agent state")
    saved_config = checkpoint.get("finetune_config", {})
    if not isinstance(saved_config, Mapping):
        raise RuntimeError("finetune checkpoint has no structured finetune_config")
    if saved_config.get("control_semantics") != config.control_semantics:
        raise RuntimeError(
            "finetune checkpoint control semantics mismatch: "
            f"found {saved_config.get('control_semantics')!r}, expected {config.control_semantics!r}"
        )
    if int(saved_config.get("replay_schema_version", 0)) != int(config.replay_schema_version):
        raise RuntimeError(
            "finetune checkpoint replay schema mismatch: "
            f"found {saved_config.get('replay_schema_version')!r}, expected {config.replay_schema_version}"
        )
    runtime_provenance = checkpoint.get("runtime_provenance", {})
    if not isinstance(runtime_provenance, Mapping):
        raise RuntimeError("finetune checkpoint has no runtime provenance")
    expected_base_hash = hash_file(config.hipad_checkpoint_path)
    saved_base_hash = str(runtime_provenance.get("checkpoint.sha256", ""))
    if saved_base_hash != expected_base_hash:
        raise RuntimeError(
            f"finetune checkpoint base hash mismatch: found {saved_base_hash or '<missing>'}, "
            f"expected {expected_base_hash}"
        )
    adapter_mode = str(saved_config.get("adapter_mode", ""))
    agent_adapter_mode = str(agent_state.get("adapter_mode", ""))
    if adapter_mode != EXPECTED_ADAPTER_MODE or agent_adapter_mode != EXPECTED_ADAPTER_MODE:
        raise RuntimeError(
            "finetune checkpoint adapter mode mismatch: "
            f"config={adapter_mode!r}, agent={agent_adapter_mode!r}, expected={EXPECTED_ADAPTER_MODE!r}"
        )
    saved_levels = tuple(int(level) for level in saved_config.get("feature_adapter_levels", ()))
    agent_levels = tuple(int(level) for level in agent_state.get("feature_adapter_levels", ()))
    if saved_levels != FOUR_LEVELS or agent_levels != FOUR_LEVELS:
        raise RuntimeError(
            "finetune checkpoint feature adapter levels mismatch: "
            f"config={saved_levels}, agent={agent_levels}, expected={FOUR_LEVELS}"
        )
    for key in ("hipad_trainable", "feature_dcnv4_adapter", "adapter_prediction"):
        value = agent_state.get(key)
        if not isinstance(value, Mapping) or not value:
            raise RuntimeError(f"finetune checkpoint agent.{key} must be a non-empty mapping")
    adapter_keys = (
        "ego_adapter_feature_dim",
        "ego_adapter_ego_state_dim",
        "ego_adapter_hidden_dim",
        "ego_adapter_ego_hidden_dim",
        "ego_adapter_residual_scale",
        "ego_adapter_dropout",
        "ego_adapter_use_layer_norm",
        "feature_adapter_levels",
        "feature_adapter_residual_scale",
        "feature_adapter_zero_init",
        "feature_adapter_feature_dim",
        "feature_adapter_ego_state_dim",
        "feature_adapter_ego_hidden_dim",
        "feature_adapter_bottleneck_reduction",
        "feature_adapter_dcn_group",
        "feature_adapter_norm_type",
        "feature_adapter_norm_groups",
    )
    config.adapter_mode = adapter_mode
    config.enable_ego_state_adapter = False
    config.enable_feature_dcnv4_adapter = True
    for key in adapter_keys:
        if key in saved_config:
            value = saved_config[key]
            if key == "feature_adapter_levels":
                value = tuple(int(level) for level in value)
            setattr(config, key, value)
    return agent_state


def _normalize_eval_paths(args):
    experiment_id = validate_experiment_id(args.experiment_id)
    run_root = Path(args.run_root).expanduser().resolve()
    expected_paths = {
        "runtime_dir": run_root / "runtime" / experiment_id,
        "log_dir": run_root / "evaluations" / experiment_id,
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
    parser = argparse.ArgumentParser(description="Frozen deterministic HiP-AD closed-loop evaluator")
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
    parser.add_argument(
        "--finetune-checkpoint",
        default="",
        help="Optional SAC .pt checkpoint; strictly restores HiP branches and external adapters.",
    )
    parser.add_argument("--allow-invalid-hipad-plan", action="store_false", dest="strict_policy")
    parser.set_defaults(strict_policy=True)

    parser.add_argument("--carla-root", default=os.environ.get("CARLA_ROOT", ""))
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=30350, type=int)
    parser.add_argument("--traffic-manager-port", default=52350, type=int)
    parser.add_argument("--timeout", default=600.0, type=float)
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--no-launch-server", action="store_true")
    parser.add_argument("--server-warmup-seconds", default=30.0, type=float)
    parser.add_argument("--runtime-dir", default="")
    parser.add_argument("--xdg-runtime-dir", default="")
    parser.add_argument("--vk-icd-filenames", default="")
    parser.add_argument("--carla-launch-user", default=os.environ.get("CARLA_LAUNCH_USER", ""))
    parser.add_argument("--routes", default=str(PROJECT_ROOT / "leaderboard/data/bench2drive220.xml"))
    parser.add_argument("--fixed-route-idx", default=-1, type=int)
    parser.add_argument("--fixed-route-name", default="")

    parser.add_argument("--max-eval-steps", default=1000, type=int)
    parser.add_argument("--max-episodes", default=0, type=int, help="0 means no episode-count limit")
    parser.add_argument("--heartbeat-every", default=100, type=int)
    parser.add_argument("--route-switch-interval", default=5, type=int)
    parser.add_argument("--stochastic", action="store_true", help="sample HiP-AD modes instead of argmax")
    parser.add_argument("--log-dir", default="")
    return _normalize_eval_paths(parser.parse_args())


def _log_episode(
    logger: StableLogger,
    episode_idx: int,
    total_step: int,
    episode_reward: float,
    episode_length: int,
    route_name: str,
    progress: float,
    termination,
    truncation,
    reward_sums: Dict[str, float],
    entropy_values,
    max_prob_values,
    selected_prob_values,
    logit_std_values,
    modes,
    prefix: str = "Episode",
) -> None:
    logger.log(
        f"[{prefix} {episode_idx}] step={total_step} return={episode_reward:.3f} "
        f"length={episode_length} route={route_name} progress={progress:.4f} "
        f"termination={termination} truncation={truncation}"
    )
    logger.log(
        f"[{prefix} {episode_idx} Reward] {_format_sums(reward_sums)}"
    )
    logger.log(
        f"[{prefix} {episode_idx} Policy] entropy_mean={_mean(entropy_values):.4f} "
        f"max_prob_mean={_mean(max_prob_values):.4f} "
        f"selected_prob_mean={_mean(selected_prob_values):.4f} "
        f"logit_std_mean={_mean(logit_std_values):.4f} "
        f"unique_modes={len(set(modes))}"
    )


def evaluate(args) -> None:
    if not args.carla_root:
        raise ValueError("--carla-root or CARLA_ROOT must point to a CARLA installation")
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

    from rl.adaptdrive_env import AdaptDriveBench2DriveSACEnv as Bench2DriveSACEnv
    from rl.hipad_policy_finetune_agent import HiPADPolicyFinetuneAgent
    from rl.roach_reward import RoachRewardMonitor

    finetune_checkpoint = None
    finetune_path = None
    finetune_agent_state = None
    if args.finetune_checkpoint:
        finetune_path = validate_runtime_asset(
            args.finetune_checkpoint,
            label="SAC finetune checkpoint",
            reject_symlink=True,
        )
        finetune_checkpoint = torch.load(finetune_path, map_location="cpu")

    config = make_config(args)
    config.hipad_checkpoint_asset_origin = checkpoint_origin
    if finetune_checkpoint is not None:
        finetune_agent_state = apply_finetune_checkpoint_config(config, finetune_checkpoint)
    env_config = make_eval_env_config(args, config)
    deterministic = not bool(args.stochastic)
    startup_provenance = collect_hipad_provenance(config.hipad_project_root)
    startup_provenance["checkpoint.path"] = str(checkpoint_path)
    startup_provenance["checkpoint.role"] = config.hipad_checkpoint_role
    startup_provenance["checkpoint.asset_origin"] = config.hipad_checkpoint_asset_origin
    logger = StableLogger(config.log_dir)

    logger.log("=" * 70)
    logger.log("Frozen HiP-AD Planning Policy Closed-Loop Evaluation")
    logger.log(f"HiP-AD root: {config.hipad_project_root}")
    logger.log(f"HiP-AD config: {config.hipad_config_path}")
    logger.log(f"HiP-AD checkpoint: {config.hipad_checkpoint_path}")
    logger.log(f"Policy anchor type: {config.policy_anchor_type}")
    logger.log(f"Mode selection: {'argmax' if deterministic else 'sampling'}")
    logger.log(f"Routes: {env_config.routes}")
    logger.log(f"Fixed route idx/name: {env_config.fixed_route_idx}/{env_config.fixed_route_name or '<none>'}")
    logger.log(f"Launch CARLA: {env_config.simulation.launch_server}")
    for name, source in startup_provenance.items():
        logger.log(f"Startup HiP-AD provenance {name}: {source}")
    logger.log("=" * 70)

    env = None
    episode_records = []
    termination_counts = Counter()
    try:
        logger.log("Creating frozen HiP-AD policy agent")
        agent = HiPADPolicyFinetuneAgent(config)
        if finetune_agent_state is not None:
            agent.load_policy_state_dict_for_eval(finetune_agent_state)
            logger.log(
                f"Strictly restored finetuned policy + adapter from {args.finetune_checkpoint} "
                f"(adapter_mode={config.adapter_mode})"
            )
        provenance = dict(startup_provenance)
        provenance.update(agent._input_adapter.runtime_asset_provenance)
        if finetune_path is not None:
            provenance["finetune_checkpoint.path"] = str(finetune_path)
            provenance["finetune_checkpoint.sha256"] = hash_file(str(finetune_path))
        for name, source in provenance.items():
            logger.log(f"HiP-AD provenance {name}: {source}")
        planning_param_count = agent.trainable_parameter_count
        agent.freeze_all_for_eval()
        logger.log(f"HiP-AD planning params frozen for eval: {planning_param_count:,}")

        # Reject incompatible checkpoint tensors before starting a CARLA world.
        logger.log("Creating environment")
        env = Bench2DriveSACEnv(env_config)
        if hasattr(env, "set_route_switch_interval"):
            env.set_route_switch_interval(args.route_switch_interval)
        reward_monitor = RoachRewardMonitor(eval_mode=False)

        pid_controller = agent.create_rollout_pid_controller()
        observation, info = env.reset()
        reward_monitor.reset(env)
        bind_hipad_after_reset(env, agent, logger)
        prime_hipad_after_reset(env, agent, logger)
        with torch.no_grad():
            current_policy_output = agent.forward_policy(observation, deterministic=deterministic)
        log_clean_navigation_context(current_policy_output, logger, 1)
        agent.begin_rollout_episode(observation, 0, 1)

        total_step = 0
        total_episode = 0
        episode_reward = 0.0
        episode_length = 0
        reward_sums = {key: 0.0 for key in REWARD_KEYS}
        entropy_values = []
        max_prob_values = []
        selected_prob_values = []
        logit_std_values = []
        modes = []
        last_info = info

        while total_step < args.max_eval_steps:
            if args.max_episodes > 0 and total_episode >= args.max_episodes:
                break

            observation["scene_token"] = observation.get(
                "scene_token",
                info.get("scene_token", info.get("route_name", "hipad_policy_eval_scene")),
            )
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
            action, _ = clean_dual_pid_step(
                pid_controller,
                speed_trajectory,
                trajectory,
                speed,
                target,
            )

            next_observation, reward, terminated, truncated, next_info = env.step(action)
            next_info.update(reward_monitor.compute(next_info))
            done = bool(terminated or truncated)

            total_step += 1
            episode_length += 1
            episode_reward += float(reward)
            last_info = next_info

            for key in REWARD_KEYS:
                reward_sums[key] += _float_value(next_info.get(key, 0.0))
            entropy_values.append(float(current_policy_output.entropy.mean().detach().cpu().item()))
            max_prob_values.append(float(current_policy_output.max_prob))
            selected_prob_values.append(float(current_policy_output.selected_prob))
            logit_std_values.append(float(current_policy_output.logit_std))
            modes.append(int(current_policy_output.mode_idx))

            if args.heartbeat_every > 0 and total_step % args.heartbeat_every == 0:
                logger.log(
                    f"[Heartbeat] step={total_step} episode={total_episode + 1} "
                    f"return={episode_reward:.3f} length={episode_length} "
                    f"route={next_info.get('route_name')} "
                    f"progress={next_info.get('route_progress', 0.0):.4f} "
                    f"speed={next_info.get('speed', 0.0):.3f} "
                    f"mode={current_policy_output.mode_idx} "
                    f"entropy={float(current_policy_output.entropy.mean().detach().cpu().item()):.4f} "
                    f"max_prob={current_policy_output.max_prob:.4f} "
                    f"r_progress={next_info.get('r_progress', 0.0):.3f} "
                    f"r_terminal={next_info.get('r_terminal', 0.0):.3f}"
                )

            if done:
                total_episode += 1
                term_items = list(_termination_items(next_info))
                termination_counts.update(term_items or ["unknown"])
                route_name = str(next_info.get("route_name", ""))
                progress = _float_value(next_info.get("route_progress", 0.0))
                episode_records.append(
                    {
                        "return": episode_reward,
                        "length": episode_length,
                        "route": route_name,
                        "progress": progress,
                        "termination": term_items,
                    }
                )
                _log_episode(
                    logger,
                    total_episode,
                    total_step,
                    episode_reward,
                    episode_length,
                    route_name,
                    progress,
                    term_items,
                    next_info.get("truncation_reason"),
                    reward_sums,
                    entropy_values,
                    max_prob_values,
                    selected_prob_values,
                    logit_std_values,
                    modes,
                )

                agent.reset_temporal_state()
                pid_controller = agent.create_rollout_pid_controller()
                observation, info = env.reset()
                reward_monitor.reset(env)
                bind_hipad_after_reset(env, agent, logger)
                prime_hipad_after_reset(env, agent, logger)
                with torch.no_grad():
                    current_policy_output = agent.forward_policy(observation, deterministic=deterministic)
                log_clean_navigation_context(current_policy_output, logger, total_episode + 1)
                agent.begin_rollout_episode(observation, total_step, total_episode + 1)
                episode_reward = 0.0
                episode_length = 0
                reward_sums = {key: 0.0 for key in REWARD_KEYS}
                entropy_values = []
                max_prob_values = []
                selected_prob_values = []
                logit_std_values = []
                modes = []
            else:
                observation = next_observation
                info = next_info
                with torch.no_grad():
                    current_policy_output = agent.forward_policy(observation, deterministic=deterministic)

        if episode_length > 0:
            _log_episode(
                logger,
                total_episode + 1,
                total_step,
                episode_reward,
                episode_length,
                str(last_info.get("route_name", "")),
                _float_value(last_info.get("route_progress", 0.0)),
                "partial",
                last_info.get("truncation_reason"),
                reward_sums,
                entropy_values,
                max_prob_values,
                selected_prob_values,
                logit_std_values,
                modes,
                prefix="Partial",
            )

        logger.log("=" * 70)
        logger.log(f"Eval finished: steps={total_step} completed_episodes={len(episode_records)}")
        if episode_records:
            returns = [item["return"] for item in episode_records]
            lengths = [item["length"] for item in episode_records]
            progresses = [item["progress"] for item in episode_records]
            logger.log(
                f"Completed episode stats: return_mean={_mean(returns):.3f} "
                f"length_mean={_mean(lengths):.1f} progress_mean={_mean(progresses):.4f}"
            )
            logger.log(f"Termination counts: {dict(termination_counts)}")
        logger.log("=" * 70)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    evaluate(parse_args())
