"""Configuration for closed-loop RL finetuning of HiP-AD planning policy."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_hipad_root() -> str:
    return str(_project_root() / "HiP-AD")


def _default_hipad_config() -> str:
    return str(_project_root() / "HiP-AD" / "local_runtime" / "hipad_b2d_stage2_clean_local.py")


def _default_routes() -> str:
    return str(_project_root() / "Bench2Drive" / "leaderboard" / "data" / "bench2drive220.xml")


@dataclass
class HiPADPolicyFinetuneConfig:
    """Settings for Line C: finetune HiP-AD cls plus final spat-2m reg branch."""

    hipad_project_root: str = field(default_factory=_default_hipad_root)
    hipad_config_path: str = field(default_factory=_default_hipad_config)
    hipad_checkpoint_path: str = field(default_factory=lambda: os.environ.get("HIPAD_CKPT", ""))
    hipad_checkpoint_role: str = "clean_base"
    hipad_checkpoint_asset_origin: str = ""
    strict_policy: bool = True
    deterministic_rollout: bool = False
    control_semantics: str = "hipad_clean_dual_pid_v2_mode_aligned"
    replay_schema_version: int = 5

    policy_anchor_type: Tuple[str, str] = ("spat", "2m")
    fut_ts: int = 6
    state_dim: int = 21
    command_dim: int = 6
    num_policy_modes: int = 48
    base_plan_clip: float = 30.0
    critic_perception_dim: int = 256
    critic_decoder_dim: int = 256
    critic_decoder_sources: Tuple[str, ...] = ("det", "map")
    use_critic_perception_context: bool = True
    use_critic_decoder_context: bool = True
    use_critic_plan_context: bool = False
    critic_plan_dim: int = 256
    critic_plan_cls_dim: int = 48

    # Adapter mode is explicit so the plan-query baseline cannot be confused
    # with the future feature-level DCNv4 adapter. The deprecated
    # enable_ego_state_adapter flag below is kept only as a plan_query alias.
    adapter_mode: str = "none"  # none | plan_query | dcnv4_feature | hybrid
    # Optional ego-state adapter over replayed HiP-AD plan-align query tokens.
    # Default is off so Line C/D legacy behavior stays unchanged.
    enable_ego_state_adapter: bool = False
    ego_adapter_feature_dim: int = 256
    ego_adapter_ego_state_dim: int = 21
    ego_adapter_hidden_dim: int = 256
    ego_adapter_ego_hidden_dim: int = 0
    ego_adapter_residual_scale: float = 1.0
    ego_adapter_dropout: float = 0.0
    ego_adapter_use_layer_norm: bool = True

    # Optional ego-state DCNv4 adapter over realtime HiP-AD FPN feature maps.
    # This is default-off and does not change replay/off-policy update schema.
    enable_feature_dcnv4_adapter: bool = False
    feature_adapter_levels: Tuple[int, ...] = (0, 1, 2, 3)
    feature_adapter_residual_scale: float = 1.0
    feature_adapter_zero_init: bool = True
    feature_adapter_feature_dim: int = 256
    feature_adapter_ego_state_dim: int = 21
    feature_adapter_ego_hidden_dim: int = 0
    feature_adapter_bottleneck_reduction: int = 4
    feature_adapter_dcn_group: int = 0
    feature_adapter_norm_type: str = "group"
    feature_adapter_norm_groups: int = 8
    # AdaptDrive adapter training path. It is separate from SAC replay updates:
    # SAC never owns feature-adapter parameters.
    adapter_prediction_enabled: bool = False
    adapter_prediction_train_reward: bool = False
    adapter_prediction_train_semantic: bool = False
    adapter_prediction_every_n_steps: int = 1
    adapter_prediction_reuse_forward_cache: bool = True
    adapter_prediction_update_mode: str = "prediction_only"
    adapter_prediction_lr: float = 3e-5
    prediction_head_lr: float = 1e-4
    adapter_prediction_weight_decay: float = 1e-4
    adapter_prediction_max_grad_norm: float = 1.0
    adapter_prediction_reward_weight: float = 1.0
    adapter_prediction_semantic_weight: float = 1.0
    adapter_prediction_residual_weight: float = 1e-3
    adapter_prediction_action_dim: int = 9
    adapter_prediction_action_hidden_dim: int = 64
    adapter_prediction_dropout: float = 0.0
    adapter_prediction_reward_huber_delta: float = 1.0
    adapter_prediction_bev_width: int = 192
    adapter_prediction_pixels_ev_to_bottom: int = 40
    adapter_prediction_pixels_per_meter: float = 5.0
    adapter_prediction_image_width: int = 1600
    adapter_prediction_image_height: int = 900
    adapter_prediction_semantic_hidden_dim: int = 128
    adapter_prediction_semantic_bce_weight: float = 1.0
    adapter_prediction_semantic_dice_weight: float = 0.5
    adapter_prediction_semantic_positive_weight: float = 2.0
    adapter_prediction_semantic_road_weight: float = 1.0
    adapter_prediction_semantic_lane_weight: float = 1.0
    adapter_prediction_semantic_route_weight: float = 0.0
    adapter_prediction_semantic_latest_vehicle_weight: float = 1.0
    adapter_prediction_semantic_latest_walker_weight: float = 1.0
    adapter_prediction_semantic_latest_tl_stop_weight: float = 1.0
    roach_bev_map_root: str = field(default_factory=lambda: os.environ.get("ROACH_BEV_MAP_ROOT", ""))
    roach_bev_target_debug_dir: str = ""
    roach_bev_target_debug_interval: int = 0
    roach_bev_target_debug_max_frames: int = 100

    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.05
    learnable_temperature: bool = False
    # Used only when learnable_temperature is enabled.
    target_entropy: float = -1.0
    policy_lr: float = 1e-5
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    max_grad_norm: float = 5.0
    critic_loss_type: str = "huber"
    critic_huber_delta: float = 1.0

    # Soft teacher via KL(ref || cur), linearly scheduled over reference_decay_steps.
    reference_kl_weight: float = 0.3
    reference_kl_final_weight: float = 0.3
    # Trust-region loss on regressed spat-2m candidates. This is an active
    # constraint for Line C and must not decay to zero.
    trajectory_trust_region_weight: float = 1.0
    reference_decay_steps: int = 1
    detach_policy_q_candidates: bool = False

    max_train_steps: int = 1000000
    batch_size: int = 64
    replay_capacity: int = 200000
    replay_max_storage_gb: float = 200.0
    replay_mmap_dir: str = "./hipad_clean_policy_finetune_feature_replay_v4"
    learning_starts: int = 5000
    train_every_n_steps: int = 100
    gradient_steps: int = 50
    policy_learning_starts: int = 5000
    policy_update_every_n_steps: int = 1
    min_critic_updates_before_policy: int = 10
    max_policy_q_loss_ema: float = 5.0
    q_loss_ema_beta: float = 0.95
    skip_sensor_mismatch_replay: bool = True
    skip_invalid_terminal_replay: bool = True
    replay_skip_log_every: int = 100
    checkpoint_on_episode_end: bool = True
    checkpoint_latest_before_reset: bool = False
    checkpoint_dir: str = "./hipad_clean_policy_finetune_checkpoints"
    checkpoint_every: int = 1000
    log_dir: str = "./hipad_clean_policy_finetune_logs"

    routes: str = field(default_factory=_default_routes)
    max_episode_steps: int = 4000

    @property
    def handcrafted_feature_dim(self) -> int:
        # Teacher-aligned state context only: no plan/action information here.
        return int(self.state_dim + 2 + self.command_dim)

    @property
    def critic_context_dim(self) -> int:
        perception_dim = self.critic_perception_dim if self.use_critic_perception_context else 0
        decoder_dim = (
            len(self.critic_decoder_sources) * self.critic_decoder_dim
            if self.use_critic_decoder_context
            else 0
        )
        return int(perception_dim + decoder_dim)

    @property
    def feature_dim(self) -> int:
        plan_extra = 0
        if self.use_critic_plan_context:
            plan_extra = self.critic_plan_dim + self.critic_plan_cls_dim
        return int(self.handcrafted_feature_dim + self.critic_context_dim + plan_extra)
