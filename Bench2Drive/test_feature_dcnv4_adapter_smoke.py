"""Smoke checks for the HiP-AD feature-level DCNv4 adapter.

This test does not start CARLA or instantiate a HiP-AD checkpoint. It only
checks the formatted FPN feature contract and the standalone adapter module.
"""

from pathlib import Path
import sys

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HIPAD_ROOT = PROJECT_ROOT / "HiP-AD"
BENCH2DRIVE_ROOT = PROJECT_ROOT / "Bench2Drive"
RL_ROOT = BENCH2DRIVE_ROOT / "leaderboard"

sys.path.insert(0, str(BENCH2DRIVE_ROOT))
sys.path.insert(0, str(RL_ROOT))
sys.path.insert(0, str(BENCH2DRIVE_ROOT / "scenario_runner"))

from rl.hipad_project_runtime import activate_hipad_project_root  # noqa: E402

activate_hipad_project_root(HIPAD_ROOT, repo_root=PROJECT_ROOT)

from projects.mmdet3d_plugin.ops import feature_maps_format  # noqa: E402
from rl.ego_state_adapter import (  # noqa: E402
    build_ego_state_dcnv4_adapter,
    build_ego_state_dcnv4_feature_adapter,
    load_adapter_state_strict_alpha_compat,
)
from rl.hipad_policy_finetune_config import HiPADPolicyFinetuneConfig  # noqa: E402


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_fake_fpn_features(
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    shapes=((88, 160), (44, 80), (22, 40), (11, 20)),
):
    batch_size = 1
    num_cams = 6
    channels = 256
    generator = torch.Generator(device=device)
    generator.manual_seed(7)
    return [
        torch.randn(
            batch_size,
            num_cams,
            channels,
            height,
            width,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        for height, width in shapes
    ]


def check_config_defaults_and_static_paths() -> None:
    config = HiPADPolicyFinetuneConfig()
    assert config.adapter_mode == "none"
    assert config.enable_ego_state_adapter is False
    assert config.enable_feature_dcnv4_adapter is False
    assert tuple(config.feature_adapter_levels) == (0, 1, 2, 3)
    assert config.adapter_prediction_enabled is False
    assert config.adapter_prediction_update_mode == "prediction_only"

    source = (PROJECT_ROOT / "Bench2Drive/leaderboard/rl/hipad_policy_finetune_agent.py").read_text(
        encoding="utf-8"
    )
    policy_params_start = source.index("    def _policy_trainable_parameters(self):")
    policy_params_end = source.index("    def _init_reference_branches", policy_params_start)
    policy_params_source = source[policy_params_start:policy_params_end]
    assert "_feature_dcnv4_adapter" not in policy_params_source


@torch.no_grad()
def check_feature_format_and_zero_init(device: torch.device) -> None:
    levels = _make_fake_fpn_features(device)
    formatted = feature_maps_format(levels)
    assert tuple(formatted[0].shape) == (1, 112200, 256)
    assert tuple(formatted[1].shape) == (6, 4, 2)
    assert tuple(formatted[2].shape) == (6, 4)

    inverse = feature_maps_format(formatted, inverse=True)
    assert len(inverse) == 1
    assert [tuple(level.shape) for level in inverse[0]] == [
        (1, 6, 256, 88, 160),
        (1, 6, 256, 44, 80),
        (1, 6, 256, 22, 40),
        (1, 6, 256, 11, 20),
    ]

    small_inverse = [
        _make_fake_fpn_features(
            device,
            shapes=((8, 16), (4, 8), (2, 4), (1, 2)),
        )
    ]
    adapter = build_ego_state_dcnv4_feature_adapter(
        feature_dim=256,
        ego_state_dim=21,
        levels=(0, 1, 2, 3),
        zero_init_residual=True,
        norm_type="group",
    ).to(device).eval()
    ego_state = torch.randn(1, 21, device=device)
    adapted_inverse, metrics = adapter(small_inverse, ego_state, return_metrics=True)
    max_abs_diff = max(
        (after - before).detach().float().abs().max().item()
        for before, after in zip(small_inverse[0], adapted_inverse[0])
    )
    assert max_abs_diff <= 1e-6, f"zero-init adapter changed formatted features by {max_abs_diff}"
    for level in (0, 1, 2, 3):
        assert metrics[f"feature_adapter_residual_l2_L{level}"] <= 1e-6
        assert metrics[f"feature_adapter_raw_residual_l2_L{level}"] <= 1e-6
        assert metrics[f"feature_adapter_effective_delta_l2_L{level}"] <= 1e-6
        assert metrics[f"feature_adapter_base_rms_L{level}"] > 0.0
        assert metrics[f"feature_adapter_raw_residual_rms_L{level}"] <= 1e-8
        assert metrics[f"feature_adapter_effective_delta_rms_L{level}"] <= 1e-8
        assert metrics[f"feature_adapter_effective_to_base_ratio_L{level}"] <= 1e-8
        assert abs(metrics[f"feature_adapter_adapted_to_base_ratio_L{level}"] - 1.0) <= 1e-6
        assert abs(metrics[f"feature_adapter_alpha_L{level}"] - 1.0) <= 1e-6
        assert f"residual_alpha_by_level.{level}" in adapter.state_dict()

    legacy_state = {
        key: value.detach().clone()
        for key, value in adapter.state_dict().items()
        if not key.startswith("residual_alpha_by_level.")
    }
    restored_adapter = build_ego_state_dcnv4_feature_adapter(
        feature_dim=256,
        ego_state_dim=21,
        levels=(0, 1, 2, 3),
        zero_init_residual=True,
        norm_type="group",
    ).to(device).eval()
    missing_keys, unexpected_keys = restored_adapter.load_state_dict(legacy_state, strict=False)
    assert not unexpected_keys
    assert set(missing_keys) == {
        "residual_alpha_by_level.0",
        "residual_alpha_by_level.1",
        "residual_alpha_by_level.2",
        "residual_alpha_by_level.3",
    }
    for level in (0, 1, 2, 3):
        assert torch.allclose(restored_adapter.residual_alpha_by_level[str(level)].detach().cpu(), torch.ones(()))

    eval_restored_adapter = build_ego_state_dcnv4_feature_adapter(
        feature_dim=256,
        ego_state_dim=21,
        levels=(0, 1, 2, 3),
        zero_init_residual=True,
        norm_type="group",
    ).to(device).eval()
    load_adapter_state_strict_alpha_compat(
        eval_restored_adapter,
        legacy_state,
        key="feature_dcnv4_adapter",
        adapter_mode="dcnv4_feature",
    )
    for level in (0, 1, 2, 3):
        assert torch.allclose(eval_restored_adapter.residual_alpha_by_level[str(level)].detach().cpu(), torch.ones(()))
    print(
        "feature_shape_zero_init_ok:",
        tuple(formatted[0].shape),
        f"max_abs_diff={max_abs_diff:.3e}",
        metrics,
        flush=True,
    )


def check_dcnv4_forward_backward(device: torch.device) -> None:
    adapter = build_ego_state_dcnv4_adapter(
        feature_dim=64,
        ego_state_dim=21,
        bottleneck_reduction=4,
        zero_init_residual=False,
        norm_type="group",
    ).to(device).train()
    feature = torch.randn(2, 64, 8, 8, device=device, requires_grad=True)
    ego_state = torch.randn(2, 21, device=device)
    output = adapter(feature, ego_state)
    loss = output.float().pow(2).mean()
    loss.backward()
    feature_grad = float(feature.grad.detach().abs().mean().item())
    param_grad = 0.0
    for param in adapter.parameters():
        if param.grad is not None:
            param_grad += float(param.grad.detach().abs().sum().item())
    assert tuple(output.shape) == tuple(feature.shape)
    assert feature_grad > 0.0, "DCNv4 adapter backward produced zero feature gradient"
    assert param_grad > 0.0, "DCNv4 adapter backward produced zero parameter gradient"
    print(
        "dcnv4_forward_backward_ok:",
        tuple(output.shape),
        f"loss={float(loss.detach().item()):.6f}",
        f"feature_grad={feature_grad:.6e}",
        f"param_grad_sum={param_grad:.6e}",
        flush=True,
    )


def check_feature_adapter_alpha_backward(device: torch.device) -> None:
    adapter = build_ego_state_dcnv4_feature_adapter(
        feature_dim=64,
        ego_state_dim=21,
        levels=(0, 1, 2, 3),
        bottleneck_reduction=4,
        zero_init_residual=False,
        residual_scale=0.5,
        norm_type="group",
    ).to(device).train()
    feature_groups = [
        [
            torch.randn(1, 2, 64, 8, 8, device=device),
            torch.randn(1, 2, 64, 4, 4, device=device),
            torch.randn(1, 2, 64, 2, 2, device=device),
            torch.randn(1, 2, 64, 1, 1, device=device),
        ]
    ]
    ego_state = torch.randn(1, 21, device=device)
    adapted, metrics = adapter(feature_groups, ego_state, return_metrics=True)
    loss = sum(level_feature.float().pow(2).mean() for level_feature in adapted[0])
    loss.backward()

    for level in (0, 1, 2, 3):
        alpha = adapter.residual_alpha_by_level[str(level)]
        assert alpha.requires_grad
        assert alpha.grad is not None
        assert float(alpha.grad.detach().abs().item()) > 0.0
        assert abs(metrics[f"feature_adapter_alpha_L{level}"] - 1.0) <= 1e-6
        assert metrics[f"feature_adapter_raw_residual_l2_L{level}"] > 0.0
        assert metrics[f"feature_adapter_effective_delta_l2_L{level}"] > 0.0
        assert metrics[f"feature_adapter_base_rms_L{level}"] > 0.0
        assert metrics[f"feature_adapter_raw_residual_rms_L{level}"] > 0.0
        assert metrics[f"feature_adapter_effective_delta_rms_L{level}"] > 0.0
        assert metrics[f"feature_adapter_effective_to_base_ratio_L{level}"] > 0.0
        assert metrics[f"feature_adapter_adapted_to_base_ratio_L{level}"] > 0.0

    print(
        "feature_adapter_alpha_backward_ok:",
        " ".join(
            f"alpha_grad_L{level}={float(adapter.residual_alpha_by_level[str(level)].grad.detach().item()):.6e}"
            for level in (0, 1, 2, 3)
        ),
        flush=True,
    )


def main() -> None:
    check_config_defaults_and_static_paths()
    device = _device()
    if device.type != "cuda":
        raise RuntimeError("DCNv4 smoke requires CUDA; set CUDA_VISIBLE_DEVICES to a usable nonzero GPU")
    check_feature_format_and_zero_init(device)
    check_dcnv4_forward_backward(device)
    check_feature_adapter_alpha_backward(device)
    print("feature_dcnv4_adapter_smoke: PASS", flush=True)


if __name__ == "__main__":
    main()
