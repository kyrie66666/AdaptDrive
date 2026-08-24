"""Closed-loop RL finetuning agent for HiP-AD cls plus final spat-2m reg branch."""

import copy
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from rl.ego_state_adapter import (
    build_ego_state_dcnv4_feature_adapter,
    build_ego_state_plan_query_adapter,
    load_adapter_state_strict_alpha_compat,
)
from rl.adapter_prediction_update import AdapterPredictionAgentMixin
from rl.hipad_clean_bridge import HiPADCleanPlanningBridge
from rl.hipad_clean_speed_decode import decode_mode_aligned_clean_speed
from rl.hipad_policy_finetune_config import HiPADPolicyFinetuneConfig
from rl.adaptdrive_sac import DUAL_PID_SUMMARY_DIM, HiPADDualTrajectoryCritic, HiPADValue
from rl.hipad_clean_runtime import HiPADCleanRuntime, HiPADRuntimePrediction


def _module_grad_norm(module: nn.Module) -> float:
    """Return the pre-clipping L2 norm of gradients owned by one module."""

    squared_norms = []
    for param in module.parameters():
        if param.grad is None:
            continue
        squared_norms.append(param.grad.detach().float().square().sum())
    if not squared_norms:
        return 0.0
    return float(torch.stack(squared_norms).sum().sqrt().cpu().item())


@dataclass
class HiPADPolicyForwardOutput:
    valid: bool
    source: str
    error: str
    logits: torch.Tensor
    log_probs: torch.Tensor
    probs: torch.Tensor
    entropy: torch.Tensor
    candidates: torch.Tensor
    selected_index: torch.Tensor
    selected_trajectory: torch.Tensor
    speed_trajectory: torch.Tensor
    feature_np: np.ndarray
    feature_tensor: torch.Tensor
    state_tensor: torch.Tensor
    mode_idx: int
    max_prob: float
    selected_prob: float
    logit_std: float
    speed_mode_idx: int
    reference_logits: Optional[torch.Tensor] = None
    reference_log_probs: Optional[torch.Tensor] = None
    reference_probs: Optional[torch.Tensor] = None
    reference_candidates: Optional[torch.Tensor] = None
    # Fields stored in replay for off-policy actor updates.
    plan_cls_context_np: Optional[np.ndarray] = None     # align_query [48, 256]
    all_candidates_np: Optional[np.ndarray] = None       # all 48 candidates [48, 6, 2]
    speed_trajectory_np: Optional[np.ndarray] = None     # frozen clean plan_speed_5hz [6, 2]
    reference_logits_np: Optional[np.ndarray] = None     # ref_logits [48]
    feature_adapter_metrics: Optional[Dict[str, float]] = None
    # Navigation actually consumed by this model forward. Rollout PID and
    # replay must use these fields instead of the generic RL-env fallback.
    navigation_command: Optional[int] = None
    target_point_np: Optional[np.ndarray] = None
    target_point_next_np: Optional[np.ndarray] = None
    # Frozen clean longitudinal action mapped one-to-one to the 48 lateral
    # modes.  The selected row is executed; all rows are replayed so discrete
    # SAC can evaluate the same composite actions the environment would run.
    longitudinal_candidates: Optional[torch.Tensor] = None  # [B, 48, 6, 2]
    longitudinal_candidates_np: Optional[np.ndarray] = None  # [48, 6, 2]
    speed_raw_area_indices: Optional[torch.Tensor] = None  # [B, 48], before rescore
    speed_area_indices: Optional[torch.Tensor] = None  # [B, 48]
    speed_all_collision: Optional[torch.Tensor] = None  # [B, 48]
    speed_raw_mode_idx: int = -1
    speed_rescore_changed_selected: bool = False
    speed_rescore_changed_rate: float = 0.0
    speed_suppressed_area_count_selected: int = 0
    speed_suppressed_area_count_mean: float = 0.0
    speed_feasible_mode_count: int = 0
    speed_modes_desired_gt_0_4: int = 0
    speed_modes_desired_gt_1_0: int = 0
    speed_all_collision_selected: bool = False
    speed_all_collision_rate: float = 0.0
    decoder_lateral_mode: int = -1


class HiPADPolicyFinetuneAgent(AdapterPredictionAgentMixin):
    """Use HiP-AD plan logits and final spat-2m reg branch as the trainable policy."""

    def __init__(self, config: HiPADPolicyFinetuneConfig, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._input_adapter = HiPADCleanRuntime(
            project_root=config.hipad_project_root,
            config_path=config.hipad_config_path,
            checkpoint_path=config.hipad_checkpoint_path,
            device=self.device,
            enabled=True,
            require_clean_tree=True,
        )
        self.policy = self
        self._model = None
        self._onedecoder = None
        self._clean_bridge = None
        self._rollout_forward_index = 0
        self._policy_anchor_index = -1
        self._trainable_param_names = []
        self._reference_plan_cls_branch = None
        self._reference_plan_reg_branch_spat_2m = None
        self.adapter_mode = self._resolve_adapter_mode()
        self._plan_query_adapter: Optional[nn.Module] = None
        self._feature_dcnv4_adapter: Optional[nn.Module] = None
        self._last_feature_adapter_metrics: Dict[str, float] = {}

        self._ensure_model()
        self._freeze_and_select_trainable_modules()
        self._init_plan_query_adapter()
        self._init_feature_dcnv4_adapter()
        self._init_adapter_prediction()
        self._init_reference_branches()

        self.critic = HiPADDualTrajectoryCritic(config.feature_dim, config.hidden_dim, config.fut_ts).to(self.device)
        self.vf = HiPADValue(
            config.feature_dim,
            config.hidden_dim,
            pid_summary_dim=DUAL_PID_SUMMARY_DIM,
        ).to(self.device)
        self.vf_target = HiPADValue(
            config.feature_dim,
            config.hidden_dim,
            pid_summary_dim=DUAL_PID_SUMMARY_DIM,
        ).to(self.device)
        self.vf_target.load_state_dict(self.vf.state_dict())
        for param in self.vf_target.parameters():
            param.requires_grad = False

        policy_params = self._policy_trainable_parameters()
        if not policy_params:
            raise RuntimeError("No HiP-AD planning parameters were marked trainable")
        self.policy_optimizer = torch.optim.Adam(policy_params, lr=config.policy_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.vf_optimizer = torch.optim.Adam(self.vf.parameters(), lr=config.critic_lr)
        self.log_alpha = torch.tensor(
            [np.log(config.alpha)],
            dtype=torch.float32,
            device=self.device,
            requires_grad=config.learnable_temperature,
        )
        self.alpha_optimizer = (
            torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
            if config.learnable_temperature
            else None
        )
        self.target_entropy = (
            float(config.target_entropy)
            if config.target_entropy >= 0.0
            else float(0.98 * np.log(max(2, config.num_policy_modes)))
        )
        self._set_eval_mode()

    @property
    def enabled(self) -> bool:
        return bool(self._input_adapter.enabled)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def trainable_parameter_count(self) -> int:
        return sum(param.numel() for param in self._policy_trainable_parameters())

    @property
    def plan_query_adapter_enabled(self) -> bool:
        return self._plan_query_adapter is not None

    @property
    def plan_query_adapter_parameter_count(self) -> int:
        if self._plan_query_adapter is None:
            return 0
        return sum(param.numel() for param in self._plan_query_adapter.parameters() if param.requires_grad)

    @property
    def ego_state_adapter_enabled(self) -> bool:
        """Deprecated alias for the plan-query adapter baseline."""
        return self.plan_query_adapter_enabled

    @property
    def ego_state_adapter_parameter_count(self) -> int:
        """Deprecated alias for the plan-query adapter baseline."""
        return self.plan_query_adapter_parameter_count

    @property
    def feature_dcnv4_adapter_enabled(self) -> bool:
        return self._feature_dcnv4_adapter is not None

    @property
    def feature_dcnv4_adapter_parameter_count(self) -> int:
        if self._feature_dcnv4_adapter is None:
            return 0
        return sum(param.numel() for param in self._feature_dcnv4_adapter.parameters())

    def freeze_all_for_eval(self) -> None:
        """Disable gradients for frozen closed-loop evaluation."""
        for module in (
            self._model,
            self.critic,
            self.vf,
            self.vf_target,
            self._reference_plan_cls_branch,
            self._reference_plan_reg_branch_spat_2m,
            self._plan_query_adapter,
            self._feature_dcnv4_adapter,
            self._adapter_prediction_reward_head,
            self._adapter_prediction_semantic_head,
        ):
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = False
        self.log_alpha.requires_grad_(False)
        self._set_eval_mode()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        self._input_adapter._lazy_init()
        if not self._input_adapter._initialized or self._input_adapter._model is None:
            raise RuntimeError(self._input_adapter._init_error or "HiP-AD model initialization failed")

        self._model = self._input_adapter._model
        self._model.to(self.device)
        self._model.eval()
        self._onedecoder = self._model.head.onedecoder_head
        self._onedecoder.with_close_loop = True
        if not getattr(self._onedecoder, "is_init_bank_list", False):
            self._onedecoder.init_instance_bank_list()
        self._clean_bridge = HiPADCleanPlanningBridge(
            self._onedecoder,
            num_modes=self.config.num_policy_modes,
            feature_dim=self.config.critic_plan_dim,
        )

        policy_anchor = tuple(self.config.policy_anchor_type)
        anchor_types = [tuple(item) for item in self._onedecoder.plan_anchor_types]
        if policy_anchor not in anchor_types:
            raise ValueError(f"policy_anchor_type={policy_anchor} not found in {anchor_types}")
        self._policy_anchor_index = anchor_types.index(policy_anchor)

    def _modules_to_train(self):
        """Train plan cls branches plus the final spat-2m regression branch.

        The shared planning trunk stays frozen; off-policy actor updates use
        the replayed final align_query, so only the final refine layer's
        spat-2m reg branch has a valid replay gradient path.
        """
        final_refine = self._onedecoder.plan_refine[-1]
        if not hasattr(final_refine, 'plan_cls_branch'):
            raise RuntimeError("Final plan_refine layer does not expose plan_cls_branch")
        if not hasattr(final_refine, 'plan_reg_branch_spat_2m'):
            raise RuntimeError("Final plan_refine layer does not expose plan_reg_branch_spat_2m")
        modules = [final_refine.plan_cls_branch, final_refine.plan_reg_branch_spat_2m]
        # plan_cls_branch_speed is NOT trained: agent execution uses only
        # spat_2m cls (see _select_policy_tensors), so speed cls has no SAC
        # action alignment.
        return modules

    def _freeze_and_select_trainable_modules(self) -> None:
        for param in self._model.parameters():
            param.requires_grad = False

        for module in self._modules_to_train():
            for param in module.parameters():
                param.requires_grad = True

        self._trainable_param_names = [
            name for name, param in self._model.named_parameters() if param.requires_grad
        ]

    def _resolve_adapter_mode(self) -> str:
        mode = str(getattr(self.config, "adapter_mode", "none") or "none").lower()
        if mode == "none":
            wants_plan_query = bool(getattr(self.config, "enable_ego_state_adapter", False))
            wants_feature_dcnv4 = bool(getattr(self.config, "enable_feature_dcnv4_adapter", False))
            if wants_plan_query and wants_feature_dcnv4:
                mode = "hybrid"
            elif wants_feature_dcnv4:
                mode = "dcnv4_feature"
            elif wants_plan_query:
                mode = "plan_query"
        valid_modes = {"none", "plan_query", "dcnv4_feature", "hybrid"}
        if mode not in valid_modes:
            raise ValueError(f"Unsupported adapter_mode={mode!r}; expected one of {sorted(valid_modes)}")
        self.config.adapter_mode = mode
        self.config.enable_ego_state_adapter = bool(mode in ("plan_query", "hybrid"))
        self.config.enable_feature_dcnv4_adapter = bool(mode in ("dcnv4_feature", "hybrid"))
        return mode

    def _init_plan_query_adapter(self) -> None:
        if self.adapter_mode not in ("plan_query", "hybrid"):
            self._plan_query_adapter = None
            return
        ego_hidden_dim = int(getattr(self.config, "ego_adapter_ego_hidden_dim", 0) or 0)
        self._plan_query_adapter = build_ego_state_plan_query_adapter(
            feature_dim=int(getattr(self.config, "ego_adapter_feature_dim", 256)),
            ego_state_dim=int(getattr(self.config, "ego_adapter_ego_state_dim", self.config.state_dim)),
            ego_hidden_dim=ego_hidden_dim if ego_hidden_dim > 0 else None,
            hidden_dim=int(getattr(self.config, "ego_adapter_hidden_dim", 256)),
            residual_scale=float(getattr(self.config, "ego_adapter_residual_scale", 1.0)),
            dropout=float(getattr(self.config, "ego_adapter_dropout", 0.0)),
            use_layer_norm=bool(getattr(self.config, "ego_adapter_use_layer_norm", True)),
        ).to(self.device)

    def _init_feature_dcnv4_adapter(self) -> None:
        if self.adapter_mode not in ("dcnv4_feature", "hybrid"):
            self._feature_dcnv4_adapter = None
            return
        ego_hidden_dim = int(getattr(self.config, "feature_adapter_ego_hidden_dim", 0) or 0)
        dcn_group = int(getattr(self.config, "feature_adapter_dcn_group", 0) or 0)
        self._feature_dcnv4_adapter = build_ego_state_dcnv4_feature_adapter(
            feature_dim=int(getattr(self.config, "feature_adapter_feature_dim", 256)),
            ego_state_dim=int(getattr(self.config, "feature_adapter_ego_state_dim", self.config.state_dim)),
            levels=tuple(int(level) for level in getattr(self.config, "feature_adapter_levels", (0, 1, 2, 3))),
            ego_hidden_dim=ego_hidden_dim if ego_hidden_dim > 0 else None,
            bottleneck_reduction=int(getattr(self.config, "feature_adapter_bottleneck_reduction", 4)),
            dcn_group=dcn_group if dcn_group > 0 else None,
            zero_init_residual=bool(getattr(self.config, "feature_adapter_zero_init", True)),
            residual_scale=float(getattr(self.config, "feature_adapter_residual_scale", 1.0)),
            norm_type=str(getattr(self.config, "feature_adapter_norm_type", "group")),
            norm_groups=int(getattr(self.config, "feature_adapter_norm_groups", 8)),
        ).to(self.device)
        train_feature_adapter = bool(getattr(self.config, "adapter_prediction_enabled", False))
        for param in self._feature_dcnv4_adapter.parameters():
            param.requires_grad = train_feature_adapter

    def _policy_trainable_parameters(self):
        params = [param for param in self._model.parameters() if param.requires_grad]
        if self._plan_query_adapter is not None:
            params.extend(param for param in self._plan_query_adapter.parameters() if param.requires_grad)
        return params

    def _init_reference_branches(self) -> None:
        """Keep frozen pretrained output branches on-device for cheap reference recomputation."""
        final_refine = self._onedecoder.plan_refine[-1]
        self._reference_plan_cls_branch = copy.deepcopy(final_refine.plan_cls_branch).to(self.device).eval()
        self._reference_plan_reg_branch_spat_2m = copy.deepcopy(
            final_refine.plan_reg_branch_spat_2m
        ).to(self.device).eval()
        for module in (self._reference_plan_cls_branch, self._reference_plan_reg_branch_spat_2m):
            for param in module.parameters():
                param.requires_grad = False

    def _named_trainable_params(self) -> Dict[str, torch.nn.Parameter]:
        named_params = dict(self._model.named_parameters())
        return {
            name: named_params[name]
            for name in self._trainable_param_names
            if name in named_params
        }

    def _capture_trainable_state(self) -> Dict[str, torch.Tensor]:
        return {
            name: param.detach().cpu().clone()
            for name, param in self._named_trainable_params().items()
        }

    def _load_trainable_state(self, state: Dict[str, torch.Tensor]) -> None:
        for name, param in self._named_trainable_params().items():
            value = state.get(name)
            if value is not None:
                param.data.copy_(value.to(device=param.device, dtype=param.dtype))

    def _snapshot_runtime_state(self):
        module_states = []
        for module in self._model.modules():
            attrs = {}
            for attr in ("cached_feature", "cached_anchor", "confidence",
                         "metas", "mask"):
                if hasattr(module, attr):
                    value = getattr(module, attr)
                    if value is None:
                        attrs[attr] = None
                    elif torch.is_tensor(value):
                        attrs[attr] = value.detach().clone()
                    else:
                        # e.g. metas is a dict or list — shallow-copy-nested
                        attrs[attr] = value
            if attrs:
                module_states.append((module, attrs))
        run_step = getattr(self._onedecoder, "run_step", None)
        return run_step, module_states

    def _restore_runtime_state(self, snapshot) -> None:
        if snapshot is None:
            return
        run_step, module_states = snapshot
        if run_step is not None and hasattr(self._onedecoder, "run_step"):
            self._onedecoder.run_step = run_step
        for module, attrs in module_states:
            for attr, value in attrs.items():
                if value is None:
                    setattr(module, attr, None)
                elif torch.is_tensor(value):
                    setattr(module, attr, value.detach().clone())
                else:
                    setattr(module, attr, value)

    def _set_eval_mode(self) -> None:
        self._model.eval()
        self.critic.eval()
        self.vf.eval()
        self.vf_target.eval()
        self._reference_plan_cls_branch.eval()
        self._reference_plan_reg_branch_spat_2m.eval()
        if self._plan_query_adapter is not None:
            self._plan_query_adapter.eval()
        if self._feature_dcnv4_adapter is not None:
            self._feature_dcnv4_adapter.eval()
        if self._adapter_prediction_reward_head is not None:
            self._adapter_prediction_reward_head.eval()
        if self._adapter_prediction_semantic_head is not None:
            self._adapter_prediction_semantic_head.eval()

    def _set_train_mode(self) -> None:
        self._model.eval()
        for module in self._modules_to_train():
            module.train()
        if self._plan_query_adapter is not None:
            self._plan_query_adapter.train()
        if self._feature_dcnv4_adapter is not None:
            if bool(getattr(self.config, "adapter_prediction_enabled", False)):
                self._feature_dcnv4_adapter.train()
            else:
                self._feature_dcnv4_adapter.eval()
        if self._adapter_prediction_reward_head is not None:
            self._adapter_prediction_reward_head.train()
        if self._adapter_prediction_semantic_head is not None:
            self._adapter_prediction_semantic_head.train()
        self.critic.train()
        self.vf.train()
        self.vf_target.eval()
        self._reference_plan_cls_branch.eval()
        self._reference_plan_reg_branch_spat_2m.eval()

    def set_global_plan(self, global_plan_gps, global_plan_world_coord) -> None:
        self._input_adapter.set_global_plan(global_plan_gps, global_plan_world_coord)

    def reset_temporal_state(self) -> None:
        self._input_adapter._step = -1
        self._input_adapter._route_planner = None
        self._clean_bridge.reset_temporal_state()

    @staticmethod
    def create_rollout_pid_controller():
        """Build the exact dual-trajectory controller used by clean evaluation."""
        from rl.hipad_clean_control import create_clean_dual_pid_controller

        return create_clean_dual_pid_controller()

    def prime(self, observation: Optional[Dict], fut_ts: int = 6) -> HiPADRuntimePrediction:
        del fut_ts
        if observation is None:
            return HiPADRuntimePrediction(valid=False, plan_temp=None, plan_spat=None, error="no_observation")
        return self._input_adapter.predict(observation)

    def begin_rollout_episode(self, observation: Dict, total_step: int, episode_index: int) -> str:
        del observation, total_step, episode_index
        return "hipad_policy_finetune"

    def current_rollout_source_label(self) -> str:
        return "hipad_policy_finetune"

    def _state_np(self, observation: Dict) -> np.ndarray:
        state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
        if state.shape[0] < self.config.state_dim:
            state = np.pad(state, (0, self.config.state_dim - state.shape[0]), mode="constant")
        return state[: self.config.state_dim].astype(np.float32, copy=False)

    def _command_onehot(self, command_value: int) -> np.ndarray:
        command_value = max(1, min(self.config.command_dim, int(command_value)))
        out = np.zeros(self.config.command_dim, dtype=np.float32)
        out[command_value - 1] = 1.0
        return out

    def _handcrafted_feature_from_observation_and_plan(self, observation: Dict, plan: np.ndarray) -> np.ndarray:
        state = self._state_np(observation)
        target = np.asarray(
            observation.get("target_point", np.zeros(2, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        if target.shape[0] < 2:
            target = np.pad(target, (0, 2 - target.shape[0]), mode="constant")
        target = target[:2]

        return np.concatenate(
            [
                state,
                target.astype(np.float32, copy=False),
                self._command_onehot(int(observation.get("command", 3))),
            ],
            axis=0,
        ).astype(np.float32)

    def _pad_or_trim_np(self, values: np.ndarray, size: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.shape[0] > size:
            return values[:size].astype(np.float32, copy=False)
        if values.shape[0] < size:
            values = np.pad(values, (0, size - values.shape[0]), mode="constant")
        return values.astype(np.float32, copy=False)

    def _pad_or_trim_tensor(self, values: torch.Tensor, size: int) -> torch.Tensor:
        values = values.float().reshape(values.shape[0], -1)
        if values.shape[1] > size:
            return values[:, :size]
        if values.shape[1] < size:
            values = F.pad(values, (0, size - values.shape[1]))
        return values

    def _ego_state_from_state_tensor(self, state: torch.Tensor, ego_dim: Optional[int] = None) -> torch.Tensor:
        ego_dim = int(ego_dim or getattr(self.config, "ego_adapter_ego_state_dim", self.config.state_dim))
        state = state.float().reshape(state.shape[0], -1)
        if state.shape[1] > ego_dim:
            return state[:, :ego_dim]
        if state.shape[1] < ego_dim:
            state = F.pad(state, (0, ego_dim - state.shape[1]))
        return state

    def _apply_plan_query_adapter(
        self,
        plan_context: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        if self._plan_query_adapter is None:
            return plan_context
        ego_state = self._ego_state_from_state_tensor(state).to(
            device=plan_context.device,
            dtype=plan_context.dtype,
        )
        return self._plan_query_adapter(plan_context, ego_state)

    def _adapt_plan_context_with_plan_query_if_enabled(
        self,
        plan_context: Optional[torch.Tensor],
        state: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if plan_context is None or not torch.is_tensor(plan_context):
            return plan_context
        return self._apply_plan_query_adapter(plan_context, state)

    def _pool_tensor_feature(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach().float()
        if tensor.dim() == 5:
            return tensor.mean(dim=(1, 3, 4))
        if tensor.dim() == 4:
            return tensor.mean(dim=(2, 3))
        if tensor.dim() == 3:
            return tensor.mean(dim=1)
        if tensor.dim() == 2:
            return tensor
        return tensor.reshape(tensor.shape[0], -1)

    def _zero_context(self, size: int) -> torch.Tensor:
        return torch.zeros(1, size, dtype=torch.float32, device=self.device)

    def _critic_regression_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_type = str(self.config.critic_loss_type).lower()
        if loss_type in {"huber", "smooth_l1", "smoothl1"}:
            return F.smooth_l1_loss(
                prediction.float(),
                target.float(),
                beta=max(float(self.config.critic_huber_delta), 1e-6),
            )
        return F.mse_loss(prediction.float(), target.float())

    def _scheduled_weight(self, initial: float, final: float, total_step: int) -> float:
        initial = float(initial)
        final = float(final)
        decay_steps = int(self.config.reference_decay_steps)
        if decay_steps <= 0:
            return initial
        ratio = min(1.0, max(0.0, float(total_step) / float(decay_steps)))
        return final + (initial - final) * (1.0 - ratio)

    def _pool_perception_context(self, feature_maps) -> torch.Tensor:
        size = self.config.critic_perception_dim
        if not self.config.use_critic_perception_context:
            return self._zero_context(0)
        if feature_maps is None:
            return self._zero_context(size)

        if (
            isinstance(feature_maps, (list, tuple))
            and len(feature_maps) == 3
            and torch.is_tensor(feature_maps[0])
            and feature_maps[0].dim() == 3
        ):
            return self._pad_or_trim_tensor(feature_maps[0].detach().float().mean(dim=1), size)

        if torch.is_tensor(feature_maps):
            return self._pad_or_trim_tensor(self._pool_tensor_feature(feature_maps), size)

        pooled_levels = []
        for feature_map in feature_maps:
            if torch.is_tensor(feature_map):
                pooled_levels.append(self._pad_or_trim_tensor(self._pool_tensor_feature(feature_map), size))
        if not pooled_levels:
            return self._zero_context(size)
        return torch.stack(pooled_levels, dim=0).mean(dim=0)

    def _pool_decoder_context(self, model_outs) -> torch.Tensor:
        if not self.config.use_critic_decoder_context:
            return self._zero_context(0)
        if model_outs is None:
            return self._zero_context(len(self.config.critic_decoder_sources) * self.config.critic_decoder_dim)

        outputs_by_name = {
            "det": model_outs[0],
            "map": model_outs[1],
            "ego": model_outs[2],
            "plan": model_outs[3],
            "motion": model_outs[4],
            "scenes": model_outs[5],
        }
        pooled = []
        for source in self.config.critic_decoder_sources:
            output = outputs_by_name.get(source)
            feature = output.get("instance_feature") if isinstance(output, dict) else None
            if torch.is_tensor(feature):
                pooled.append(
                    self._pad_or_trim_tensor(
                        self._pool_tensor_feature(feature),
                        self.config.critic_decoder_dim,
                    )
                )
            else:
                pooled.append(self._zero_context(self.config.critic_decoder_dim))
        if not pooled:
            return self._zero_context(0)
        return torch.cat(pooled, dim=-1)

    def _critic_feature_from_context(
        self,
        observation: Dict,
        plan: np.ndarray,
        feature_maps=None,
        model_outs=None,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        handcrafted = self._handcrafted_feature_from_observation_and_plan(observation, plan)
        handcrafted = self._pad_or_trim_np(handcrafted, self.config.handcrafted_feature_dim)
        handcrafted_tensor = torch.from_numpy(handcrafted).unsqueeze(0).to(self.device)

        extras = [handcrafted_tensor]
        extras.append(self._pool_perception_context(feature_maps))
        extras.append(self._pool_decoder_context(model_outs))

        context = torch.cat(extras, dim=-1)
        context = self._pad_or_trim_tensor(context, self.config.feature_dim)
        context_np = context.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return context_np, context.detach()

    def _fallback_output(self, observation: Dict, error: str) -> HiPADPolicyForwardOutput:
        zeros_logits = torch.zeros(1, self.config.num_policy_modes, device=self.device)
        zeros_candidates = torch.zeros(1, self.config.num_policy_modes, self.config.fut_ts, 2, device=self.device)
        log_probs = F.log_softmax(zeros_logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
        selected_index = torch.zeros(1, dtype=torch.long, device=self.device)
        selected_trajectory = zeros_candidates[:, 0]
        feature_np, feature_tensor = self._critic_feature_from_context(
            observation,
            np.zeros((self.config.fut_ts, 2), dtype=np.float32),
        )
        state_tensor = torch.from_numpy(self._state_np(observation)).unsqueeze(0).to(self.device)
        return HiPADPolicyForwardOutput(
            valid=False,
            source="invalid",
            error=error,
            logits=zeros_logits,
            log_probs=log_probs,
            probs=probs,
            entropy=entropy,
            candidates=zeros_candidates,
            selected_index=selected_index,
            selected_trajectory=selected_trajectory,
            speed_trajectory=torch.zeros(1, self.config.fut_ts, 2, device=self.device),
            feature_np=feature_np,
            feature_tensor=feature_tensor,
            state_tensor=state_tensor,
            mode_idx=0,
            max_prob=float(probs.max().detach().item()),
            selected_prob=float(probs[0, 0].detach().item()),
            logit_std=0.0,
            speed_mode_idx=-1,
            speed_trajectory_np=np.zeros((self.config.fut_ts, 2), dtype=np.float32),
            longitudinal_candidates=zeros_candidates,
            longitudinal_candidates_np=np.zeros(
                (self.config.num_policy_modes, self.config.fut_ts, 2),
                dtype=np.float32,
            ),
            speed_area_indices=torch.zeros(
                1,
                self.config.num_policy_modes,
                dtype=torch.long,
                device=self.device,
            ),
            speed_all_collision=torch.zeros(
                1,
                self.config.num_policy_modes,
                dtype=torch.bool,
                device=self.device,
            ),
        )

    def _prepare_inputs(self, observation: Dict, return_navigation: bool = False):
        raw_inputs = self._input_adapter._prepare_inputs(observation)
        navigation = self._navigation_context_from_inputs(raw_inputs)
        inputs = self._input_adapter._batchify(raw_inputs)
        if return_navigation:
            return inputs, navigation
        return inputs

    @staticmethod
    def _navigation_context_from_inputs(inputs: Dict) -> Tuple[int, np.ndarray, np.ndarray]:
        required = ("gt_ego_fut_cmd", "target_point", "target_point_next")
        missing = [name for name in required if name not in inputs]
        if missing:
            raise RuntimeError("clean model inputs are missing navigation fields: " + ", ".join(missing))

        def as_numpy(value) -> np.ndarray:
            if torch.is_tensor(value):
                return value.detach().float().cpu().numpy()
            return np.asarray(value, dtype=np.float32)

        command_onehot = as_numpy(inputs["gt_ego_fut_cmd"])
        target_point = as_numpy(inputs["target_point"])
        target_point_next = as_numpy(inputs["target_point_next"])
        command_onehot = command_onehot.reshape(-1, command_onehot.shape[-1])
        target_point = target_point.reshape(-1, 2)
        target_point_next = target_point_next.reshape(-1, 2)
        if command_onehot.shape[0] != 1 or target_point.shape[0] != 1 or target_point_next.shape[0] != 1:
            raise RuntimeError(
                "clean rollout expects batch size 1 for navigation, got "
                f"command={tuple(command_onehot.shape)}, target={tuple(target_point.shape)}, "
                f"target_next={tuple(target_point_next.shape)}"
            )

        command = int(command_onehot.argmax(axis=-1)[0]) + 1
        target_np = target_point[0].astype(np.float32, copy=True)
        target_next_np = target_point_next[0].astype(np.float32, copy=True)
        if command < 1 or command > 6:
            raise RuntimeError(f"clean navigation command {command} is outside [1, 6]")
        if not np.isfinite(target_np).all() or not np.isfinite(target_next_np).all():
            raise RuntimeError("clean navigation target point contains non-finite values")
        return command, target_np, target_next_np

    def _feature_maps_format(self, feature_maps, inverse: bool = False):
        extract_feat_fn = getattr(self._model.extract_feat, "__func__", self._model.extract_feat)
        feature_maps_format = getattr(extract_feat_fn, "__globals__", {}).get("feature_maps_format")
        if feature_maps_format is None:
            from projects.mmdet3d_plugin.ops import feature_maps_format

        return feature_maps_format(feature_maps, inverse=inverse)

    def _adapt_feature_maps_with_dcnv4_if_enabled(self, feature_maps, observation: Optional[Dict]):
        self._last_feature_adapter_metrics = {}
        if self._feature_dcnv4_adapter is None:
            return feature_maps
        if observation is None:
            raise RuntimeError("feature DCNv4 adapter requires the current observation to build ego_state")
        state_tensor = torch.from_numpy(self._state_np(observation)).unsqueeze(0).to(self.device)
        ego_state = self._ego_state_from_state_tensor(
            state_tensor,
            ego_dim=int(getattr(self.config, "feature_adapter_ego_state_dim", self.config.state_dim)),
        )
        inverse_feature_maps = self._feature_maps_format(feature_maps, inverse=True)
        adapted_feature_maps, metrics = self._feature_dcnv4_adapter(
            inverse_feature_maps,
            ego_state,
            return_metrics=True,
        )
        self._last_feature_adapter_metrics = dict(metrics)
        return self._feature_maps_format(adapted_feature_maps)

    def _extract_feature_maps(self, inputs: Dict, observation: Optional[Dict] = None):
        with torch.no_grad():
            feature_maps = self._model.extract_feat(inputs["img"])
        self.cache_adapter_prediction_forward_base(feature_maps, observation)
        return self._adapt_feature_maps_with_dcnv4_if_enabled(feature_maps, observation)

    def _head_forward(self, inputs: Dict, feature_maps):
        frame_token = (self._input_adapter._step, self._rollout_forward_index)
        self._rollout_forward_index += 1
        self._clean_bridge.begin_rollout_capture(frame_token)
        try:
            model_outs = self._model.head(inputs["img"], feature_maps, inputs)
            plan_align_query = self._clean_bridge.end_rollout_capture()
        except Exception:
            self._clean_bridge.abort_rollout_capture()
            raise
        model_outs[3]["plan_align_query"] = plan_align_query
        return model_outs

    def _decode_clean_speed_plan(
        self,
        inputs: Dict,
        model_outs,
        decoder_lateral_mode: int,
        selected_lateral_mode: int,
    ):
        """Decode frozen clean longitudinal actions in every lateral context."""

        decoder_lateral_mode = int(decoder_lateral_mode)
        selected_lateral_mode = int(selected_lateral_mode)
        if not (0 <= decoder_lateral_mode < self.config.num_policy_modes):
            raise RuntimeError(f"decoder lateral mode {decoder_lateral_mode} is outside [0, {self.config.num_policy_modes})")
        if not (0 <= selected_lateral_mode < self.config.num_policy_modes):
            raise RuntimeError(f"selected lateral mode {selected_lateral_mode} is outside [0, {self.config.num_policy_modes})")
        decoded = decode_mode_aligned_clean_speed(
            self._onedecoder.plan_decoder,
            inputs,
            model_outs,
            num_lateral_modes=self.config.num_policy_modes,
            fut_ts=self.config.fut_ts,
            output_frequency="5hz",
        )
        selected_speed_plan = decoded.trajectories[:, selected_lateral_mode]
        speed_mode_idx = int(decoded.speed_area_indices[0, selected_lateral_mode].item())
        selected_all_collision = bool(decoded.all_collision[0, selected_lateral_mode].item())
        return (
            selected_speed_plan,
            decoded.trajectories,
            decoded.raw_speed_area_indices,
            decoded.speed_area_indices,
            decoded.rescore_changed,
            decoded.suppressed_area_count,
            decoded.all_collision,
            speed_mode_idx,
            selected_all_collision,
        )

    def _raw_forward(self, observation: Dict):
        inputs = self._prepare_inputs(observation)
        feature_maps = self._extract_feature_maps(inputs, observation)
        model_outs = self._head_forward(inputs, feature_maps)
        return inputs, feature_maps, model_outs

    def _select_policy_tensors(self, inputs: Dict, plan_output: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        prediction = plan_output["prediction"][-1].float()
        classification = plan_output["classification"][-1].float()
        num_group = len(self._onedecoder.plan_anchor_types)
        reg_preds = list(prediction.chunk(chunks=num_group, dim=2))
        cls_preds = list(classification.chunk(chunks=num_group, dim=2))

        batch_size = classification.shape[0]
        cls = cls_preds[self._policy_anchor_index].reshape(batch_size, self._onedecoder.ego_fut_cmd, -1)
        reg = reg_preds[self._policy_anchor_index].reshape(
            batch_size,
            self._onedecoder.ego_fut_cmd,
            -1,
            self.config.fut_ts,
            2,
        ).cumsum(dim=-2)

        if self._onedecoder.ego_fut_cmd > 1:
            cmd = inputs["gt_ego_fut_cmd"].argmax(dim=-1).long().to(cls.device)
        else:
            cmd = torch.zeros(batch_size, dtype=torch.long, device=cls.device)
        batch_indices = torch.arange(batch_size, device=cls.device)
        return cls[batch_indices, cmd], reg[batch_indices, cmd]

    def _policy_tensors_from_model_outs(self, inputs: Dict, model_outs) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, _, plan_output, _, _ = model_outs
        logits, candidates = self._select_policy_tensors(inputs, plan_output)
        logits = logits[:, : self.config.num_policy_modes]
        candidates = candidates[:, : self.config.num_policy_modes]
        candidates = torch.clamp(candidates, -self.config.base_plan_clip, self.config.base_plan_clip)
        return logits, candidates

    def _adapted_policy_tensors_from_align_query(
        self,
        plan_align_query: Optional[torch.Tensor],
        state: torch.Tensor,
        base_candidates: torch.Tensor,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        if self._plan_query_adapter is None or plan_align_query is None or not torch.is_tensor(plan_align_query):
            return None, None, plan_align_query, None

        final_refine = self._onedecoder.plan_refine[-1]
        batch_size = plan_align_query.shape[0]
        adapted_query = self._apply_plan_query_adapter(plan_align_query, state)

        logits_flat = final_refine.plan_cls_branch(adapted_query)
        logits = logits_flat.reshape(batch_size, self.config.num_policy_modes)
        reg_cur = final_refine.plan_reg_branch_spat_2m(adapted_query)
        with torch.no_grad():
            base_reg_cur = final_refine.plan_reg_branch_spat_2m(plan_align_query)
            ref_reg = self._reference_plan_reg_branch_spat_2m(plan_align_query)
            base_delta = (ref_reg - base_reg_cur).reshape(
                batch_size,
                self.config.num_policy_modes,
                self.config.fut_ts,
                2,
            )
            candidate_ref = torch.clamp(
                base_candidates.detach() + base_delta,
                -self.config.base_plan_clip,
                self.config.base_plan_clip,
            )
        candidate_delta = (reg_cur - ref_reg.detach()).reshape(
            batch_size,
            self.config.num_policy_modes,
            self.config.fut_ts,
            2,
        )
        candidates = torch.clamp(
            candidate_ref + candidate_delta,
            -self.config.base_plan_clip,
            self.config.base_plan_clip,
        )
        return logits, candidates, adapted_query, candidate_ref

    def _reference_policy_tensors_from_align_query(
        self,
        plan_align_query: Optional[torch.Tensor],
        current_candidates: torch.Tensor,
        current_plan_context: Optional[torch.Tensor] = None,
    ):
        if self.config.reference_kl_weight <= 0.0:
            return None, None, None, None
        if plan_align_query is None or not torch.is_tensor(plan_align_query):
            return None, None, None, None

        final_refine = self._onedecoder.plan_refine[-1]
        current_plan_context = current_plan_context if torch.is_tensor(current_plan_context) else plan_align_query
        batch_size = current_candidates.shape[0]
        with torch.no_grad():
            logits_flat = self._reference_plan_cls_branch(plan_align_query)
            logits = logits_flat.reshape(batch_size, self.config.num_policy_modes)
            cur_reg = final_refine.plan_reg_branch_spat_2m(current_plan_context)
            ref_reg = self._reference_plan_reg_branch_spat_2m(plan_align_query)
            ref_delta = (ref_reg - cur_reg).reshape(
                batch_size,
                self.config.num_policy_modes,
                self.config.fut_ts,
                2,
            )
            candidates = torch.clamp(
                current_candidates.detach() + ref_delta,
                -self.config.base_plan_clip,
                self.config.base_plan_clip,
            )
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
        return logits.detach(), log_probs.detach(), probs.detach(), candidates.detach()

    def _forward_policy_impl(
        self,
        observation: Dict,
        deterministic: bool,
        include_reference: bool,
    ) -> HiPADPolicyForwardOutput:
        self.clear_adapter_prediction_forward_cache()
        inputs, navigation = self._prepare_inputs(observation, return_navigation=True)
        navigation_command, target_point_np, target_point_next_np = navigation
        navigation_observation = dict(observation)
        navigation_observation["command"] = navigation_command
        navigation_observation["target_point"] = target_point_np
        navigation_observation["target_point_next"] = target_point_next_np
        feature_maps = self._extract_feature_maps(inputs, observation)
        feature_adapter_metrics = dict(self._last_feature_adapter_metrics)
        reference_logits = None
        reference_log_probs = None
        reference_probs = None
        reference_candidates = None

        model_outs = self._head_forward(inputs, feature_maps)
        logits, candidates = self._policy_tensors_from_model_outs(inputs, model_outs)
        decoder_logits = logits
        state_tensor = torch.from_numpy(self._state_np(observation)).unsqueeze(0).to(self.device)
        current_plan_context = None
        replay_candidate_ref = None

        # Extract HiP-AD plan features for critic and replay.
        _, _, _, plan_output, _, _ = model_outs
        plan_align_query = plan_output.get("plan_align_query")
        if self._plan_query_adapter is not None:
            adapted_logits, adapted_candidates, current_plan_context, replay_candidate_ref = (
                self._adapted_policy_tensors_from_align_query(plan_align_query, state_tensor, candidates)
            )
            if adapted_logits is not None and adapted_candidates is not None:
                logits = adapted_logits
                candidates = adapted_candidates
        else:
            current_plan_context = plan_align_query
        if include_reference:
            (
                reference_logits,
                reference_log_probs,
                reference_probs,
                reference_candidates,
            ) = self._reference_policy_tensors_from_align_query(
                plan_align_query,
                candidates,
                current_plan_context=current_plan_context,
            )

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
        if deterministic:
            selected_index = probs.argmax(dim=-1)
        else:
            selected_index = Categorical(probs=probs).sample()
        selected_prob = probs[
            torch.arange(probs.shape[0], device=probs.device),
            selected_index,
        ]
        selected_trajectory = candidates[
            torch.arange(candidates.shape[0], device=candidates.device),
            selected_index,
        ]
        decoder_lateral_mode = int(decoder_logits[0].argmax().item())
        (
            speed_trajectory,
            longitudinal_candidates,
            speed_raw_area_indices,
            speed_area_indices,
            speed_rescore_changed,
            speed_suppressed_area_count,
            speed_all_collision,
            speed_mode_idx,
            speed_all_collision_selected,
        ) = self._decode_clean_speed_plan(
            inputs,
            model_outs,
            decoder_lateral_mode=decoder_lateral_mode,
            selected_lateral_mode=int(selected_index[0].item()),
        )
        longitudinal_deltas = longitudinal_candidates[:, :, 1:] - longitudinal_candidates[:, :, :-1]
        longitudinal_desired_speeds = torch.linalg.norm(longitudinal_deltas, dim=-1).mean(dim=-1) / 0.2
        speed_modes_desired_gt_0_4 = int((longitudinal_desired_speeds[0] > 0.4).sum().item())
        speed_modes_desired_gt_1_0 = int((longitudinal_desired_speeds[0] > 1.0).sum().item())
        speed_feasible_mode_count = int(
            ((longitudinal_desired_speeds[0] >= 0.4) & (~speed_all_collision[0])).sum().item()
        )
        selected_lateral_mode = int(selected_index[0].item())
        greedy_trajectory = candidates[0, int(probs[0].argmax().item())].detach().cpu().numpy()
        feature_np, feature_tensor = self._critic_feature_from_context(
            navigation_observation,
            greedy_trajectory,
            feature_maps,
            model_outs,
        )

        # Build numpy copies for replay storage.
        # reference_logits are recomputed on-the-fly in update_policy_from_feature_batch
        # by frozen reference branches; they are not stored in replay.
        align_np = None
        replay_candidates = candidates.detach()
        if plan_align_query is not None and torch.is_tensor(plan_align_query):
            align_np = plan_align_query.detach().cpu().numpy().astype(np.float16)
            if replay_candidate_ref is not None:
                replay_candidates = replay_candidate_ref.detach()
            else:
                final_refine = self._onedecoder.plan_refine[-1]
                with torch.no_grad():
                    cur_reg = final_refine.plan_reg_branch_spat_2m(plan_align_query)
                    ref_reg = self._reference_plan_reg_branch_spat_2m(plan_align_query)
                    ref_delta = (ref_reg - cur_reg).reshape(
                        candidates.shape[0],
                        self.config.num_policy_modes,
                        self.config.fut_ts,
                        2,
                    )
                    replay_candidates = torch.clamp(
                        candidates.detach() + ref_delta,
                        -self.config.base_plan_clip,
                        self.config.base_plan_clip,
                    )
        cand_np = replay_candidates.detach().cpu().numpy().astype(np.float16)

        return HiPADPolicyForwardOutput(
            valid=True,
            source="plan_{}_{}".format(*self.config.policy_anchor_type),
            error="",
            logits=logits,
            log_probs=log_probs,
            probs=probs,
            entropy=entropy,
            candidates=candidates,
            selected_index=selected_index,
            selected_trajectory=selected_trajectory,
            speed_trajectory=speed_trajectory,
            feature_np=feature_np,
            feature_tensor=feature_tensor,
            state_tensor=state_tensor,
            mode_idx=int(selected_index[0].item()),
            max_prob=float(probs.max(dim=-1).values.mean().detach().item()),
            selected_prob=float(selected_prob.mean().detach().item()),
            logit_std=float(logits.std(dim=-1, unbiased=False).mean().detach().item()),
            speed_mode_idx=speed_mode_idx,
            reference_logits=reference_logits,
            reference_log_probs=reference_log_probs,
            reference_probs=reference_probs,
            reference_candidates=reference_candidates,
            plan_cls_context_np=align_np,
            all_candidates_np=cand_np,
            speed_trajectory_np=speed_trajectory[0].detach().cpu().numpy().astype(np.float32),
            longitudinal_candidates=longitudinal_candidates,
            longitudinal_candidates_np=longitudinal_candidates[0].detach().cpu().numpy().astype(np.float16),
            speed_raw_area_indices=speed_raw_area_indices,
            speed_area_indices=speed_area_indices,
            speed_all_collision=speed_all_collision,
            speed_raw_mode_idx=int(speed_raw_area_indices[0, selected_lateral_mode].item()),
            speed_rescore_changed_selected=bool(speed_rescore_changed[0, selected_lateral_mode].item()),
            speed_rescore_changed_rate=float(speed_rescore_changed.float().mean().detach().item()),
            speed_suppressed_area_count_selected=int(
                speed_suppressed_area_count[0, selected_lateral_mode].item()
            ),
            speed_suppressed_area_count_mean=float(
                speed_suppressed_area_count.float().mean().detach().item()
            ),
            speed_feasible_mode_count=speed_feasible_mode_count,
            speed_modes_desired_gt_0_4=speed_modes_desired_gt_0_4,
            speed_modes_desired_gt_1_0=speed_modes_desired_gt_1_0,
            speed_all_collision_selected=speed_all_collision_selected,
            speed_all_collision_rate=float(speed_all_collision.float().mean().detach().item()),
            decoder_lateral_mode=decoder_lateral_mode,
            reference_logits_np=None,  # computed on-the-fly in update_policy_from_feature_batch
            feature_adapter_metrics=feature_adapter_metrics,
            navigation_command=navigation_command,
            target_point_np=target_point_np,
            target_point_next_np=target_point_next_np,
        )

    def forward_policy(
        self,
        observation: Dict,
        deterministic: bool = False,
        track_grad: bool = False,
        include_reference: bool = False,
        preserve_runtime_state: bool = False,
    ) -> HiPADPolicyForwardOutput:
        self._set_eval_mode()
        runtime_snapshot = self._snapshot_runtime_state() if preserve_runtime_state else None
        try:
            if track_grad:
                return self._forward_policy_impl(observation, deterministic, include_reference)
            else:
                with torch.no_grad():
                    return self._forward_policy_impl(observation, deterministic, include_reference)
        except Exception as exc:
            if self.config.strict_policy:
                raise
            return self._fallback_output(observation, str(exc))
        finally:
            if preserve_runtime_state:
                self._restore_runtime_state(runtime_snapshot)

    @torch.no_grad()
    def trajectory_numpy(self, policy_output: HiPADPolicyForwardOutput) -> np.ndarray:
        return policy_output.selected_trajectory[0].detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def speed_trajectory_numpy(self, policy_output: HiPADPolicyForwardOutput) -> np.ndarray:
        return policy_output.speed_trajectory[0].detach().cpu().numpy().astype(np.float32)

    def update_critic_value_from_feature_batch(self, batch) -> Dict[str, float]:
        self._set_train_mode()
        state = batch.observations["state"].float().to(self.device)
        critic_features = batch.critic_bev_features.float().to(self.device)
        next_critic_features = batch.next_critic_bev_features.float().to(self.device)
        prev_pid_summaries = batch.prev_pid_summaries.float().to(self.device)
        prev_pid_masks = batch.prev_pid_summary_masks.float().to(self.device)
        pid_summaries = batch.pid_summaries.float().to(self.device)
        lateral_trajectories = batch.trajectories.float().to(self.device)
        longitudinal_trajectories = batch.longitudinal_trajectories.float().to(self.device)
        candidate_longitudinal_trajectories = batch.candidate_longitudinal_trajectories.float().to(self.device)
        plan_cls_context = batch.plan_cls_context.float().to(self.device)
        candidate_ref = batch.all_candidates.float().to(self.device)
        rewards = batch.rewards.float().to(self.device)
        dones = batch.dones.float().to(self.device)
        batch_size = plan_cls_context.shape[0]
        final_refine = self._onedecoder.plan_refine[-1]

        with torch.no_grad():
            policy_plan_context = self._adapt_plan_context_with_plan_query_if_enabled(plan_cls_context, state)
            next_v = self.vf_target(next_critic_features, pid_summaries, torch.ones_like(dones)).float()
            target_q = rewards + (1.0 - dones) * self.config.gamma * next_v
            target_q = torch.clamp(target_q, min=-100.0, max=100.0)

        q1, q2 = self.critic(
            critic_features,
            state,
            longitudinal_trajectories,
            lateral_trajectories,
            prev_pid_summaries,
            prev_pid_masks,
        )
        critic_loss = self._critic_regression_loss(q1, target_q) + self._critic_regression_loss(q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            logits_flat = final_refine.plan_cls_branch(policy_plan_context)
            logits = logits_flat.reshape(batch_size, self.config.num_policy_modes)
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            reg_cur = final_refine.plan_reg_branch_spat_2m(policy_plan_context)
            ref_reg = self._reference_plan_reg_branch_spat_2m(plan_cls_context)
            candidate_delta = (reg_cur - ref_reg).reshape(
                batch_size,
                self.config.num_policy_modes,
                self.config.fut_ts,
                2,
            )
            candidate_cur = torch.clamp(
                candidate_ref + candidate_delta,
                -self.config.base_plan_clip,
                self.config.base_plan_clip,
            )
            q1_policy, q2_policy = self.critic.evaluate_candidates(
                critic_features,
                state,
                candidate_longitudinal_trajectories,
                candidate_cur,
                prev_pid_summaries,
                prev_pid_masks,
            )
            min_q_policy = torch.min(q1_policy, q2_policy)
            target_v = (
                probs.detach() *
                (min_q_policy - self.alpha.detach() * log_probs.detach())
            ).sum(dim=-1, keepdim=True)
        current_v = self.vf(critic_features, prev_pid_summaries, prev_pid_masks).float()
        vf_loss = self._critic_regression_loss(current_v, target_v)
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._soft_update_target()

        return {
            "critic_q_loss": float(critic_loss.item()),
            "critic_v_loss": float(vf_loss.item()),
            "q_value": float(torch.min(q1, q2).mean().detach().item()),
            "v_value": float(current_v.mean().detach().item()),
        }

    def update_policy_from_feature_batch(self, batch, total_step: int = 0) -> Dict[str, float]:
        """Off-policy actor update using replayed align_query.

        The policy logits and final spat-2m candidate residuals are recomputed
        from plan_cls_context. Candidate reconstruction uses
        candidate_ref + (reg_cur - reg_ref), so replay does not need anchors.
        """
        self._set_train_mode()

        plan_cls_context = batch.plan_cls_context.float().to(self.device)  # [B, 48, 256]
        critic_features = batch.critic_bev_features.float().to(self.device)
        state = batch.observations["state"].float().to(self.device)
        candidate_ref = batch.all_candidates.float().to(self.device)  # [B, 48, 6, 2]
        candidate_longitudinal_trajectories = batch.candidate_longitudinal_trajectories.float().to(self.device)
        prev_pid_summaries = batch.prev_pid_summaries.float().to(self.device)
        prev_pid_masks = batch.prev_pid_summary_masks.float().to(self.device)
        batch_size = plan_cls_context.shape[0]
        final_refine = self._onedecoder.plan_refine[-1]
        policy_plan_context = self._adapt_plan_context_with_plan_query_if_enabled(plan_cls_context, state)

        # Reference logits/reg are recomputed by frozen pretrained output branches.
        ref_logits = None
        with torch.no_grad():
            if self.config.reference_kl_weight > 0.0:
                ref_logits_flat = self._reference_plan_cls_branch(plan_cls_context)
                ref_logits = ref_logits_flat.reshape(batch_size, self.config.num_policy_modes)
            ref_reg = self._reference_plan_reg_branch_spat_2m(plan_cls_context)

        # Current logits/reg from trained output branches.
        logits_flat = final_refine.plan_cls_branch(policy_plan_context)  # [B, 48, 1]
        logits = logits_flat.reshape(batch_size, self.config.num_policy_modes)
        reg_cur = final_refine.plan_reg_branch_spat_2m(policy_plan_context)  # [B, 48, 12]

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)

        candidate_delta = (reg_cur - ref_reg).reshape(batch_size, self.config.num_policy_modes, self.config.fut_ts, 2)
        candidates = candidate_ref + candidate_delta
        candidates = torch.clamp(candidates, -self.config.base_plan_clip, self.config.base_plan_clip)

        # Q values for all 48 modes. Freeze critic parameters while keeping
        # candidate gradients so SAC can update the spat-2m reg branch.
        critic_requires_grad = [param.requires_grad for param in self.critic.parameters()]
        for param in self.critic.parameters():
            param.requires_grad = False
        q_candidate_grad_enabled = not bool(self.config.detach_policy_q_candidates)
        q_candidates = candidates.detach() if not q_candidate_grad_enabled else candidates.clone()
        if q_candidate_grad_enabled and q_candidates.requires_grad:
            q_candidates.retain_grad()
        try:
            q1, q2 = self.critic.evaluate_candidates(
                critic_features,
                state,
                candidate_longitudinal_trajectories,
                q_candidates,
                prev_pid_summaries,
                prev_pid_masks,
            )
            min_q = torch.min(q1, q2)
        finally:
            for param, requires_grad in zip(self.critic.parameters(), critic_requires_grad):
                param.requires_grad = requires_grad

        sac_policy_loss = (
            probs * (self.alpha.detach() * log_probs - min_q)
        ).sum(dim=-1).mean()

        # Soft teacher: KL(ref || cur), computed on-the-fly (no replay storage needed).
        reference_kl_loss = torch.zeros((), device=self.device)
        kl_weight = 0.0
        if ref_logits is not None:
            with torch.no_grad():
                ref_probs = F.softmax(ref_logits, dim=-1)
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            reference_kl_loss = (
                ref_probs * (ref_log_probs - log_probs)
            ).sum(dim=-1).mean()
            kl_weight = self._scheduled_weight(
                self.config.reference_kl_weight,
                self.config.reference_kl_final_weight,
                total_step,
            )

        candidate_trust_region_loss = F.smooth_l1_loss(
            candidates.float(),
            candidate_ref.float(),
            beta=max(float(self.config.critic_huber_delta), 1e-6),
        )
        traj_weight = float(getattr(self.config, "trajectory_trust_region_weight", 1.0))
        policy_loss = sac_policy_loss + kl_weight * reference_kl_loss + traj_weight * candidate_trust_region_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        q_candidate_grad_norm = 0.0
        if q_candidate_grad_enabled and q_candidates.grad is not None:
            q_candidate_grad_norm = float(q_candidates.grad.detach().float().norm().cpu().item())
        plan_cls_branch_grad_norm = _module_grad_norm(final_refine.plan_cls_branch)
        plan_spat_reg_branch_grad_norm = _module_grad_norm(final_refine.plan_reg_branch_spat_2m)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._policy_trainable_parameters(),
            self.config.max_grad_norm,
        )
        self.policy_optimizer.step()

        alpha_loss_value = 0.0
        if self.config.learnable_temperature:
            alpha_loss = (self.log_alpha * (entropy.detach() - self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_value = float(alpha_loss.item())

        expected_q = (probs.detach() * min_q.detach()).sum(dim=-1, keepdim=True)
        q_gap_modes = min_q.detach().max(dim=-1).values - min_q.detach().min(dim=-1).values
        candidate_l2 = torch.linalg.norm((candidates - candidate_ref).detach().float(), dim=-1)
        candidate_delta_per_horizon = candidate_l2.mean(dim=(0, 1))
        candidate_delta_x_mean = (candidates[..., 0] - candidate_ref[..., 0]).detach().abs().mean()
        candidate_delta_y_mean = (candidates[..., 1] - candidate_ref[..., 1]).detach().abs().mean()
        plan_query_adapter_delta = torch.zeros((), device=self.device)
        if self._plan_query_adapter is not None:
            plan_query_adapter_delta = torch.linalg.norm(
                (policy_plan_context - plan_cls_context).detach().float(),
                dim=-1,
            ).mean()

        return {
            "policy_loss": float(policy_loss.item()),
            "sac_policy_loss": float(sac_policy_loss.item()),
            "reference_kl_loss": float(reference_kl_loss.item()),
            "reference_kl_weight": float(kl_weight),
            "trajectory_trust_region_loss": float(candidate_trust_region_loss.item()),
            "trajectory_trust_region_weight": float(traj_weight),
            "reference_traj_drift": float(candidate_trust_region_loss.detach().item()),
            "candidate_l2_mean": float(candidate_l2.mean().item()),
            "candidate_l2_max": float(candidate_l2.max().item()),
            "candidate_delta_per_horizon": ",".join(
                f"{value:.4f}" for value in candidate_delta_per_horizon.detach().cpu().tolist()
            ),
            "candidate_delta_x_mean": float(candidate_delta_x_mean.item()),
            "candidate_delta_y_mean": float(candidate_delta_y_mean.item()),
            "traj_entropy": float(entropy.mean().detach().item()),
            "alpha": float(self.alpha.item()),
            "alpha_loss": alpha_loss_value,
            "policy_q": float(expected_q.mean().item()),
            "selected_log_prob": 0.0,
            "max_prob": float(probs.max(dim=-1).values.mean().detach().item()),
            "selected_prob": 0.0,
            "logit_std": float(logits.std(dim=-1, unbiased=False).mean().detach().item()),
            "q_std_modes": float(min_q.detach().std(dim=-1, unbiased=False).mean().item()),
            "q_gap_modes": float(q_gap_modes.mean().item()),
            "plan_query_adapter_enabled": float(self.plan_query_adapter_enabled),
            "plan_query_adapter_delta_l2": float(plan_query_adapter_delta.item()),
            "grad_norm": float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm),
            "policy_q_candidate_grad": float(q_candidate_grad_enabled),
            "policy_q_candidate_grad_enabled": float(q_candidate_grad_enabled),
            "policy_q_candidate_grad_norm": float(q_candidate_grad_norm),
            "policy_plan_cls_branch_grad_norm": float(plan_cls_branch_grad_norm),
            "policy_plan_spat_reg_branch_grad_norm": float(plan_spat_reg_branch_grad_norm),
            "policy_skipped": 0.0,
        }

    def update_policy_from_rollout(
        self,
        policy_output: HiPADPolicyForwardOutput,
        prev_pid_summary: Optional[np.ndarray] = None,
        prev_pid_mask: float = 0.0,
        total_step: int = 0,
    ) -> Dict[str, float]:
        if not policy_output.valid:
            return {"policy_loss": 0.0, "traj_entropy": 0.0, "alpha": float(self.alpha.item()), "policy_skipped": 1.0}

        self._set_train_mode()
        prev_pid = None
        if prev_pid_summary is not None:
            prev_pid = torch.from_numpy(np.asarray(prev_pid_summary, dtype=np.float32)).unsqueeze(0).to(self.device)
        prev_mask = torch.tensor([prev_pid_mask], dtype=torch.float32, device=self.device)

        critic_requires_grad = [param.requires_grad for param in self.critic.parameters()]
        for param in self.critic.parameters():
            param.requires_grad = False
        q_candidate_grad_enabled = not bool(self.config.detach_policy_q_candidates)
        q_candidates = (
            policy_output.candidates.detach()
            if not q_candidate_grad_enabled
            else policy_output.candidates.clone()
        )
        if q_candidate_grad_enabled and q_candidates.requires_grad:
            q_candidates.retain_grad()
        try:
            q1, q2 = self.critic.evaluate_candidates(
                policy_output.feature_tensor.float(),
                policy_output.state_tensor.float(),
                policy_output.longitudinal_candidates.float(),
                q_candidates.float(),
                prev_pid,
                prev_mask,
            )
            min_q = torch.min(q1, q2)
            sac_policy_loss = (
                policy_output.probs * (self.alpha.detach() * policy_output.log_probs - min_q)
            ).sum(dim=-1).mean()
            reference_kl_loss = torch.zeros((), device=self.device)
            kl_weight = 0.0
            if policy_output.reference_probs is not None and policy_output.reference_log_probs is not None:
                reference_kl_loss = (
                    policy_output.reference_probs *
                    (policy_output.reference_log_probs - policy_output.log_probs)
                ).sum(dim=-1).mean()
                kl_weight = self._scheduled_weight(
                    self.config.reference_kl_weight,
                    self.config.reference_kl_final_weight,
                    total_step,
                )
            # reference_traj_loss removed from training: with frozen trunk,
            # candidates are deterministic → no gradient path to plan_cls_branch.
            # Computed for drift monitoring only.
            reference_traj_drift = torch.zeros((), device=self.device)
            if policy_output.reference_candidates is not None:
                reference_traj_drift = F.mse_loss(
                    policy_output.candidates.detach().float(),
                    policy_output.reference_candidates.detach().float(),
                )
            policy_loss = sac_policy_loss + kl_weight * reference_kl_loss
        finally:
            for param, requires_grad in zip(self.critic.parameters(), critic_requires_grad):
                param.requires_grad = requires_grad

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        q_candidate_grad_norm = 0.0
        if q_candidate_grad_enabled and q_candidates.grad is not None:
            q_candidate_grad_norm = float(q_candidates.grad.detach().float().norm().cpu().item())
        final_refine = self._onedecoder.plan_refine[-1]
        plan_cls_branch_grad_norm = _module_grad_norm(final_refine.plan_cls_branch)
        plan_spat_reg_branch_grad_norm = _module_grad_norm(final_refine.plan_reg_branch_spat_2m)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._policy_trainable_parameters(),
            self.config.max_grad_norm,
        )
        self.policy_optimizer.step()

        alpha_loss_value = 0.0
        if self.config.learnable_temperature:
            alpha_loss = (self.log_alpha * (policy_output.entropy.detach() - self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_value = float(alpha_loss.item())

        expected_q = (policy_output.probs.detach() * min_q.detach()).sum(dim=-1, keepdim=True)
        selected_log_prob = policy_output.log_probs[
            torch.arange(policy_output.log_probs.shape[0], device=self.device),
            policy_output.selected_index,
        ]
        q_std_modes = min_q.detach().std(dim=-1, unbiased=False)
        q_gap_modes = min_q.detach().max(dim=-1).values - min_q.detach().min(dim=-1).values
        return {
            "policy_loss": float(policy_loss.item()),
            "sac_policy_loss": float(sac_policy_loss.item()),
            "reference_kl_loss": float(reference_kl_loss.item()),
            "reference_traj_drift": float(reference_traj_drift.item()),
            "reference_kl_weight": float(kl_weight),
            "traj_entropy": float(policy_output.entropy.mean().detach().item()),
            "alpha": float(self.alpha.item()),
            "alpha_loss": alpha_loss_value,
            "policy_q": float(expected_q.mean().item()),
            "selected_log_prob": float(selected_log_prob.mean().detach().item()),
            "max_prob": float(policy_output.max_prob),
            "selected_prob": float(policy_output.selected_prob),
            "logit_std": float(policy_output.logit_std),
            "q_std_modes": float(q_std_modes.mean().item()),
            "q_gap_modes": float(q_gap_modes.mean().item()),
            "plan_query_adapter_enabled": float(self.plan_query_adapter_enabled),
            "plan_query_adapter_delta_l2": 0.0,
            "grad_norm": float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm),
            "policy_q_candidate_grad": float(q_candidate_grad_enabled),
            "policy_q_candidate_grad_enabled": float(q_candidate_grad_enabled),
            "policy_q_candidate_grad_norm": float(q_candidate_grad_norm),
            "policy_plan_cls_branch_grad_norm": float(plan_cls_branch_grad_norm),
            "policy_plan_spat_reg_branch_grad_norm": float(plan_spat_reg_branch_grad_norm),
            "policy_skipped": 0.0,
        }

    def update_policy_from_observation(
        self,
        observation: Dict,
        prev_pid_summary: Optional[np.ndarray] = None,
        prev_pid_mask: float = 0.0,
        total_step: int = 0,
    ) -> Dict[str, float]:
        policy_output = self.forward_policy(
            observation,
            deterministic=False,
            track_grad=True,
            include_reference=True,
            preserve_runtime_state=True,
        )
        return self.update_policy_from_rollout(
            policy_output,
            prev_pid_summary=prev_pid_summary,
            prev_pid_mask=prev_pid_mask,
            total_step=total_step,
        )

    def _soft_update_target(self) -> None:
        tau = self.config.tau
        for param, target_param in zip(self.vf.parameters(), self.vf_target.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def _optimizer_to_device(self, optimizer: Optional[torch.optim.Optimizer]) -> None:
        if optimizer is None:
            return
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def _hipad_trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        named_params = dict(self._model.named_parameters())
        return {
            name: named_params[name].detach().cpu()
            for name in self._trainable_param_names
            if name in named_params
        }

    def _load_shape_matched_module_state(self, module, module_state) -> None:
        if not module_state:
            return
        current_state = module.state_dict()
        compatible_state = {
            name: value
            for name, value in module_state.items()
            if name in current_state and tuple(current_state[name].shape) == tuple(value.shape)
        }
        module.load_state_dict(compatible_state, strict=False)

    def state_dict(self):
        plan_query_adapter_state = (
            self._plan_query_adapter.state_dict()
            if self._plan_query_adapter is not None
            else None
        )
        feature_dcnv4_adapter_state = (
            self._feature_dcnv4_adapter.state_dict()
            if self._feature_dcnv4_adapter is not None
            else None
        )
        return {
            "adapter_mode": str(self.adapter_mode),
            "hipad_trainable": self._hipad_trainable_state_dict(),
            "hipad_trainable_names": list(self._trainable_param_names),
            "plan_query_adapter_enabled": bool(self.plan_query_adapter_enabled),
            "plan_query_adapter": plan_query_adapter_state,
            "feature_dcnv4_adapter_enabled": bool(self.feature_dcnv4_adapter_enabled),
            "feature_dcnv4_adapter_parameter_count": int(self.feature_dcnv4_adapter_parameter_count),
            "feature_adapter_levels": tuple(int(level) for level in getattr(self.config, "feature_adapter_levels", ())),
            "feature_dcnv4_adapter": feature_dcnv4_adapter_state,
            "adapter_prediction": self._adapter_prediction_state_dict(),
            # Deprecated checkpoint keys kept for compatibility with earlier
            # plan-query adapter checkpoints.
            "ego_state_adapter_enabled": bool(self.plan_query_adapter_enabled),
            "ego_state_adapter": plan_query_adapter_state,
            "critic": self.critic.state_dict(),
            "vf": self.vf.state_dict(),
            "vf_target": self.vf_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "vf_optimizer": self.vf_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict() if self.alpha_optimizer is not None else None,
        }

    @staticmethod
    def _strict_module_state(module: nn.Module, state, label: str) -> Mapping:
        if not isinstance(state, Mapping):
            raise RuntimeError(f"{label} must be a mapping")
        current = module.state_dict()
        missing = sorted(set(current) - set(state))
        unexpected = sorted(set(state) - set(current))
        shape_mismatch = sorted(
            name
            for name in set(current) & set(state)
            if not torch.is_tensor(state[name])
            or tuple(current[name].shape) != tuple(state[name].shape)
        )
        if missing or unexpected or shape_mismatch:
            raise RuntimeError(
                f"{label} strict state mismatch: missing={missing}, unexpected={unexpected}, "
                f"shape_mismatch={shape_mismatch}"
            )
        return state

    def _load_full_state_dict_strict(self, state_dict: Mapping, load_optimizers: bool) -> None:
        current_state = self.state_dict()
        missing_top = sorted(set(current_state) - set(state_dict))
        unexpected_top = sorted(set(state_dict) - set(current_state))
        if missing_top or unexpected_top:
            raise RuntimeError(
                f"AdaptDrive agent state mismatch: missing={missing_top}, unexpected={unexpected_top}"
            )
        expected_metadata = {
            "adapter_mode": self.adapter_mode,
            "hipad_trainable_names": list(self._trainable_param_names),
            "plan_query_adapter_enabled": self.plan_query_adapter_enabled,
            "feature_dcnv4_adapter_enabled": self.feature_dcnv4_adapter_enabled,
            "feature_dcnv4_adapter_parameter_count": self.feature_dcnv4_adapter_parameter_count,
            "feature_adapter_levels": tuple(int(level) for level in self.config.feature_adapter_levels),
            "ego_state_adapter_enabled": self.plan_query_adapter_enabled,
        }
        metadata_mismatch = {
            key: (state_dict.get(key), expected)
            for key, expected in expected_metadata.items()
            if state_dict.get(key) != expected
        }
        if metadata_mismatch:
            raise RuntimeError(f"AdaptDrive agent metadata mismatch: {metadata_mismatch}")

        named_params = dict(self._model.named_parameters())
        planning_state = state_dict.get("hipad_trainable")
        if not isinstance(planning_state, Mapping):
            raise RuntimeError("agent.hipad_trainable must be a mapping")
        expected_planning = {name: named_params[name] for name in self._trainable_param_names}
        self._strict_module_state_dict_compatible(planning_state, expected_planning, "agent.hipad_trainable")

        if self._plan_query_adapter is None:
            if state_dict.get("plan_query_adapter") is not None or state_dict.get("ego_state_adapter") is not None:
                raise RuntimeError("dcnv4_feature resume checkpoint contains a plan-query adapter")
        else:
            self._strict_module_state(
                self._plan_query_adapter,
                state_dict.get("plan_query_adapter"),
                "plan_query_adapter",
            )
        if self._feature_dcnv4_adapter is None:
            raise RuntimeError("AdaptDrive strict resume requires a feature DCNv4 adapter")
        self._strict_module_state(
            self._feature_dcnv4_adapter,
            state_dict.get("feature_dcnv4_adapter"),
            "feature_dcnv4_adapter",
        )
        self._strict_module_state(self.critic, state_dict.get("critic"), "critic")
        self._strict_module_state(self.vf, state_dict.get("vf"), "vf")
        self._strict_module_state(self.vf_target, state_dict.get("vf_target"), "vf_target")

        prediction = state_dict.get("adapter_prediction")
        if not isinstance(prediction, Mapping):
            raise RuntimeError("agent.adapter_prediction must be a mapping")
        if self._adapter_prediction_reward_head is None or self._adapter_prediction_semantic_head is None:
            raise RuntimeError("AdaptDrive strict resume requires both prediction heads")
        self._strict_module_state(
            self._adapter_prediction_reward_head,
            prediction.get("reward_head"),
            "adapter_prediction.reward_head",
        )
        self._strict_module_state(
            self._adapter_prediction_semantic_head,
            prediction.get("semantic_head"),
            "adapter_prediction.semantic_head",
        )
        current_prediction = self._adapter_prediction_state_dict()
        for key in ("adapter_prediction_enabled", "reward_target_names", "semantic_channel_names"):
            if prediction.get(key) != current_prediction.get(key):
                raise RuntimeError(f"agent.adapter_prediction.{key} mismatch")
        current_weights = current_prediction.get("semantic_channel_loss_weights")
        saved_weights = prediction.get("semantic_channel_loss_weights")
        if not torch.is_tensor(saved_weights) or not torch.equal(current_weights, saved_weights.cpu()):
            raise RuntimeError("agent.adapter_prediction semantic channel weights mismatch")
        if self._adapter_prediction_optimizer is not None and not isinstance(prediction.get("optimizer"), Mapping):
            raise RuntimeError("strict resume requires adapter prediction optimizer state")

        log_alpha = state_dict.get("log_alpha")
        if not torch.is_tensor(log_alpha) or tuple(log_alpha.shape) != tuple(self.log_alpha.shape):
            raise RuntimeError("agent.log_alpha shape mismatch")
        if load_optimizers:
            for key in ("policy_optimizer", "critic_optimizer", "vf_optimizer"):
                if not isinstance(state_dict.get(key), Mapping):
                    raise RuntimeError(f"strict resume requires {key}")
            if self.alpha_optimizer is None and state_dict.get("alpha_optimizer") is not None:
                raise RuntimeError("checkpoint contains alpha optimizer while target temperature is fixed")
            if self.alpha_optimizer is not None and not isinstance(state_dict.get("alpha_optimizer"), Mapping):
                raise RuntimeError("strict resume requires alpha optimizer state")

        for name, value in planning_state.items():
            parameter = named_params[name]
            parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        if self._plan_query_adapter is not None:
            self._plan_query_adapter.load_state_dict(state_dict["plan_query_adapter"], strict=True)
        self._feature_dcnv4_adapter.load_state_dict(state_dict["feature_dcnv4_adapter"], strict=True)
        self._load_adapter_prediction_state(prediction, load_optimizers=load_optimizers)
        self.critic.load_state_dict(state_dict["critic"], strict=True)
        self.vf.load_state_dict(state_dict["vf"], strict=True)
        self.vf_target.load_state_dict(state_dict["vf_target"], strict=True)
        self.log_alpha.data.copy_(log_alpha.to(device=self.device, dtype=self.log_alpha.dtype))
        if load_optimizers:
            self.policy_optimizer.load_state_dict(state_dict["policy_optimizer"])
            self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
            self.vf_optimizer.load_state_dict(state_dict["vf_optimizer"])
            for optimizer in (self.policy_optimizer, self.critic_optimizer, self.vf_optimizer):
                self._optimizer_to_device(optimizer)
            if self.alpha_optimizer is not None:
                self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])
                self._optimizer_to_device(self.alpha_optimizer)
        self._set_eval_mode()

    @staticmethod
    def _strict_module_state_dict_compatible(state: Mapping, expected: Mapping, label: str) -> None:
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        shape_mismatch = sorted(
            name
            for name in set(expected) & set(state)
            if not torch.is_tensor(state[name])
            or tuple(state[name].shape) != tuple(expected[name].shape)
        )
        if missing or unexpected or shape_mismatch:
            raise RuntimeError(
                f"{label} strict state mismatch: missing={missing}, unexpected={unexpected}, "
                f"shape_mismatch={shape_mismatch}"
            )

    def load_state_dict(self, state_dict, load_optimizers: bool = True, strict: bool = False):
        if strict:
            if not isinstance(state_dict, Mapping):
                raise RuntimeError("AdaptDrive agent state must be a mapping")
            self._load_full_state_dict_strict(state_dict, load_optimizers=load_optimizers)
            return
        named_params = dict(self._model.named_parameters())
        for name, value in state_dict.get("hipad_trainable", {}).items():
            if name in named_params:
                param = named_params[name]
                param.data.copy_(value.to(device=param.device, dtype=param.dtype))

        plan_query_state = state_dict.get("plan_query_adapter")
        if plan_query_state is None:
            plan_query_state = state_dict.get("ego_state_adapter")
        if self._plan_query_adapter is not None and plan_query_state is not None:
            self._load_shape_matched_module_state(
                self._plan_query_adapter,
                plan_query_state,
            )
        elif self._plan_query_adapter is None and plan_query_state is not None:
            print(
                "[HiPADPolicyFinetuneAgent] Ignoring plan-query adapter checkpoint state "
                f"because adapter_mode={self.adapter_mode!r}"
            )
        feature_dcnv4_state = state_dict.get("feature_dcnv4_adapter")
        if self._feature_dcnv4_adapter is not None and feature_dcnv4_state is not None:
            self._load_shape_matched_module_state(
                self._feature_dcnv4_adapter,
                feature_dcnv4_state,
            )
        elif self._feature_dcnv4_adapter is None and feature_dcnv4_state is not None:
            print(
                "[HiPADPolicyFinetuneAgent] Ignoring feature DCNv4 adapter checkpoint state "
                f"because adapter_mode={self.adapter_mode!r}"
            )
        self._load_adapter_prediction_state(state_dict.get("adapter_prediction", {}), load_optimizers=load_optimizers)
        self._load_shape_matched_module_state(self.critic, state_dict.get("critic"))
        self._load_shape_matched_module_state(self.vf, state_dict.get("vf"))
        self._load_shape_matched_module_state(self.vf_target, state_dict.get("vf_target", state_dict.get("vf")))
        if "log_alpha" in state_dict:
            self.log_alpha.data.copy_(state_dict["log_alpha"].to(self.device))
        if load_optimizers:
            if "policy_optimizer" in state_dict:
                self.policy_optimizer.load_state_dict(state_dict["policy_optimizer"])
                self._optimizer_to_device(self.policy_optimizer)
            if "critic_optimizer" in state_dict:
                self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
                self._optimizer_to_device(self.critic_optimizer)
            if "vf_optimizer" in state_dict:
                self.vf_optimizer.load_state_dict(state_dict["vf_optimizer"])
                self._optimizer_to_device(self.vf_optimizer)
            if self.alpha_optimizer is not None and state_dict.get("alpha_optimizer") is not None:
                self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])
                self._optimizer_to_device(self.alpha_optimizer)
        self._set_eval_mode()

    def load_policy_state_dict_for_eval(self, state_dict) -> None:
        """Strictly restore model branches and external adapters for evaluation."""

        named_params = dict(self._model.named_parameters())
        trainable_state = state_dict.get("hipad_trainable")
        if not isinstance(trainable_state, dict) or not trainable_state:
            raise RuntimeError("finetune checkpoint has no agent.hipad_trainable state")
        missing_model = [name for name in self._trainable_param_names if name not in trainable_state]
        expected_trainable = set(self._trainable_param_names)
        unexpected_model = [name for name in trainable_state if name not in expected_trainable]
        mismatched_model = [
            name for name, value in trainable_state.items()
            if name in named_params and tuple(value.shape) != tuple(named_params[name].shape)
        ]
        if missing_model or unexpected_model or mismatched_model:
            raise RuntimeError(
                "finetuned HiP-AD policy state mismatch: "
                f"missing={missing_model}, unexpected={unexpected_model}, shape_mismatch={mismatched_model}"
            )
        for name, value in trainable_state.items():
            param = named_params[name]
            param.data.copy_(value.to(device=param.device, dtype=param.dtype))

        adapter_specs = (
            ("plan_query_adapter", self._plan_query_adapter),
            ("feature_dcnv4_adapter", self._feature_dcnv4_adapter),
        )
        for key, module in adapter_specs:
            module_state = state_dict.get(key)
            if module is None:
                if module_state:
                    raise RuntimeError(f"checkpoint contains {key}, but evaluation adapter_mode={self.adapter_mode!r}")
                continue
            load_adapter_state_strict_alpha_compat(
                module,
                module_state,
                key=key,
                adapter_mode=self.adapter_mode,
            )
        self._set_eval_mode()
