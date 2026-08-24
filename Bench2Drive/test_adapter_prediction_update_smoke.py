#!/usr/bin/env python3
"""Smoke checks for adapter prediction update cache lifecycle and gradients."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCH2DRIVE_ROOT = PROJECT_ROOT / "Bench2Drive"
LEADERBOARD_ROOT = BENCH2DRIVE_ROOT / "leaderboard"
for path in (str(BENCH2DRIVE_ROOT), str(LEADERBOARD_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from rl.adapter_prediction_update import AdapterPredictionAgentMixin  # noqa: E402


class TinyFeatureAdapter(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.delta = nn.Parameter(torch.full((1, 1, channels, 1, 1), 1e-3))
        self.residual_alpha_by_level = nn.ParameterDict(
            {str(level): nn.Parameter(torch.ones(())) for level in range(4)}
        )

    def forward(self, feature_groups, ego_state, return_metrics: bool = False):
        del ego_state
        adapted_groups = []
        metrics = {}
        for group in feature_groups:
            adapted_group = []
            for level, tensor in enumerate(group):
                alpha = self.residual_alpha_by_level[str(level)].to(device=tensor.device, dtype=tensor.dtype)
                residual = self.delta.to(device=tensor.device, dtype=tensor.dtype).expand_as(tensor) * alpha
                adapted_group.append(tensor + residual)
                metrics[f"feature_adapter_residual_l2_L{level}"] = residual.float().pow(2).mean().detach()
            adapted_groups.append(adapted_group)
        if return_metrics:
            return adapted_groups, metrics
        return adapted_groups


class FrozenTinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1), requires_grad=False)


class FakeAgent(AdapterPredictionAgentMixin):
    def __init__(self, *, with_adapter: bool = True) -> None:
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(
            adapter_prediction_enabled=True,
            adapter_prediction_train_reward=True,
            adapter_prediction_train_semantic=True,
            adapter_prediction_every_n_steps=1,
            adapter_prediction_reuse_forward_cache=True,
            adapter_prediction_update_mode="prediction_only",
            adapter_prediction_lr=1e-3,
            prediction_head_lr=1e-3,
            adapter_prediction_weight_decay=0.0,
            adapter_prediction_max_grad_norm=10.0,
            adapter_prediction_reward_weight=1.0,
            adapter_prediction_semantic_weight=1.0,
            adapter_prediction_residual_weight=1e-3,
            adapter_prediction_action_dim=9,
            adapter_prediction_action_hidden_dim=8,
            adapter_prediction_semantic_hidden_dim=8,
            adapter_prediction_bev_width=32,
            adapter_prediction_image_width=160,
            adapter_prediction_image_height=90,
            adapter_prediction_semantic_route_weight=0.0,
            feature_adapter_feature_dim=8,
            feature_adapter_ego_state_dim=21,
            feature_adapter_levels=(0, 1, 2, 3),
            hidden_dim=32,
            state_dim=21,
        )
        self._feature_dcnv4_adapter = TinyFeatureAdapter(8) if with_adapter else None
        self._model = FrozenTinyModel()
        self._init_adapter_prediction()

    def _feature_maps_format(self, feature_maps, inverse: bool = False):
        del inverse
        return feature_maps

    def _state_np(self, observation):
        return np.asarray(observation["state"], dtype=np.float32)

    def _ego_state_from_state_tensor(self, state_tensor, ego_dim: int):
        return state_tensor[:, :ego_dim]

    def _load_shape_matched_module_state(self, module, module_state) -> None:
        if module_state:
            module.load_state_dict(module_state, strict=False)

    def _optimizer_to_device(self, optimizer) -> None:
        del optimizer


def main() -> None:
    torch.manual_seed(3)
    disabled_agent = FakeAgent(with_adapter=False)
    assert disabled_agent._adapter_prediction_optimizer is None

    agent = FakeAgent(with_adapter=True)
    assert agent.adapter_prediction_enabled

    base_groups = [[
        torch.randn(1, 6, 8, 16, 28),
        torch.randn(1, 6, 8, 8, 14),
        torch.randn(1, 6, 8, 4, 7),
        torch.randn(1, 6, 8, 2, 4),
    ]]
    observation = {
        "state": np.zeros(21, dtype=np.float32),
        "sensor_frame": np.int64(5),
    }
    agent.cache_adapter_prediction_forward_base(base_groups, observation)
    assert agent._adapter_prediction_forward_cache is not None
    for level in range(4):
        assert not agent._adapter_prediction_forward_cache["base_groups"][0][level].requires_grad

    masks = np.zeros((15, 32, 32), dtype=np.uint8)
    masks[0, 4:28, 4:28] = 255
    masks[2, 14:16, 5:27] = 120
    masks[6, 12:17, 15:21] = 255
    masks[10, 9:12, 9:12] = 255
    masks[14, 3:5, 12:25] = 255
    metrics = agent.update_adapter_prediction_from_step(
        reward=1.0,
        reward_info={
            "r_speed": 0.5,
            "r_position": -0.1,
            "r_rotation": -0.05,
            "r_progress": 0.01,
            "route_progress_delta": 0.001,
        },
        semantic_target={
            "frame": 5,
            "masks": masks,
            "sensor_frame_exact": True,
            "channel_names": tuple(f"c{i}" for i in range(15)),
        },
        action_summary=np.zeros(9, dtype=np.float32),
        total_step=1,
    )
    assert agent._adapter_prediction_forward_cache is None, "cache must be cleared after update"
    assert metrics["adapter_prediction_skipped"] == 0.0
    assert metrics["adapter_prediction_grad_norm_adapter"] > 0.0
    assert metrics["adapter_prediction_grad_norm_reward_head"] > 0.0
    assert metrics["adapter_prediction_grad_norm_semantic_head"] > 0.0
    alpha_grads = [metrics[f"feature_adapter_alpha_grad_L{level}"] for level in range(4)]
    assert all(np.isfinite(value) for value in alpha_grads)
    assert any(abs(value) > 0.0 for value in alpha_grads)
    assert agent._model.weight.grad is None
    assert "projector_valid_bev_ratio" in metrics

    state = agent._adapter_prediction_state_dict()
    assert state["reward_head"] is not None
    assert state["semantic_head"] is not None
    assert state["optimizer"] is not None
    assert state["semantic_channel_names"] is not None
    assert state["semantic_channel_loss_weights"] is not None

    restored_agent = FakeAgent(with_adapter=True)
    restored_agent._load_adapter_prediction_state(state, load_optimizers=True)
    for key, value in agent._adapter_prediction_reward_head.state_dict().items():
        restored = restored_agent._adapter_prediction_reward_head.state_dict()[key]
        assert torch.allclose(value, restored), f"reward head state mismatch for {key}"
    assert restored_agent._adapter_prediction_optimizer.state_dict()["state"], (
        "adapter prediction optimizer state should restore after one update"
    )

    broken_state = dict(state)
    broken_reward_head = dict(state["reward_head"])
    broken_reward_head.pop(next(iter(broken_reward_head)))
    broken_state["reward_head"] = broken_reward_head
    try:
        FakeAgent(with_adapter=True)._load_adapter_prediction_state(
            broken_state,
            load_optimizers=False,
        )
    except RuntimeError as exc:
        assert "reward_head strict state mismatch" in str(exc)
    else:
        raise AssertionError("incomplete legacy parent reward_head was accepted")
    print("adapter_prediction_update_smoke: PASS", flush=True)


if __name__ == "__main__":
    main()
