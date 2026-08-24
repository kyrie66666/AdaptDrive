#!/usr/bin/env python3
"""Smoke tests for ego-state conditioned feature adapter."""

import sys
from pathlib import Path

import torch


BENCH2DRIVE_ROOT = Path(__file__).resolve().parent
LEADERBOARD_ROOT = BENCH2DRIVE_ROOT / "leaderboard"
for path in (str(BENCH2DRIVE_ROOT), str(LEADERBOARD_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from torch import nn

from rl.ego_state_adapter import DCNv4BottleneckBlock, EgoStateDCNv4Adapter, EgoStatePlanQueryAdapter
from rl.ego_state_adapter import EgoStateEncoder, EgoStateFeatureFusion, SELayer


class FakeDCNv4(nn.Module):
    def __init__(self, channels, **kwargs):
        super().__init__()
        self.proj = nn.Linear(channels, channels)

    def forward(self, input, shape=None):
        return self.proj(input)


def assert_equal(lhs, rhs, message: str) -> None:
    if lhs != rhs:
        raise AssertionError(f"{message}: expected {rhs!r}, found {lhs!r}")


def assert_close(lhs: torch.Tensor, rhs: torch.Tensor, message: str, atol: float = 0.0) -> None:
    if not torch.allclose(lhs, rhs, atol=atol, rtol=0.0):
        raise AssertionError(message)


def exercise_vector_feature() -> None:
    feature = torch.randn(4, 560)
    ego_state = torch.randn(4, 21)
    fusion = EgoStateFeatureFusion(feature_dim=560, ego_state_dim=21)

    fused, gate = fusion(feature, ego_state, return_gate=True)

    assert_equal(tuple(fused.shape), tuple(feature.shape), "vector fused shape should match feature")
    assert_equal(tuple(gate.shape), tuple(feature.shape), "vector gate shape should match feature")
    assert_close(fused, feature * gate, "vector fusion should match SSR-style feature * gate")
    if gate.min().item() < 0.0 or gate.max().item() > 1.0:
        raise AssertionError("SE gate should be in [0, 1]")


def exercise_batch_first_tokens() -> None:
    feature = torch.randn(3, 128, 256)
    ego_state = torch.randn(3, 21)
    fusion = EgoStateFeatureFusion(feature_dim=256, ego_state_dim=21, batch_first=True)

    fused, gate = fusion(feature, ego_state, return_gate=True)
    assert_equal(tuple(fused.shape), tuple(feature.shape), "batch-first token fused shape should match feature")
    assert_equal(tuple(gate.shape), (3, 1, 256), "batch-first token gate should broadcast on token dim")
    assert_close(fused, feature * gate, "batch-first token fusion should match feature * gate")


def exercise_sequence_first_tokens() -> None:
    feature = torch.randn(128, 3, 256)
    ego_state = torch.randn(3, 21)
    fusion = EgoStateFeatureFusion(feature_dim=256, ego_state_dim=21, batch_first=False)

    fused, gate = fusion(feature, ego_state, return_gate=True)
    assert_equal(tuple(fused.shape), tuple(feature.shape), "sequence-first fused shape should match feature")
    assert_equal(tuple(gate.shape), (1, 3, 256), "sequence-first gate should broadcast on token dim")
    assert_close(fused, feature * gate, "sequence-first fusion should match feature * gate")


def exercise_channel_first_feature() -> None:
    feature = torch.randn(2, 64, 8, 8)
    ego_state = torch.randn(2, 21)
    fusion = EgoStateFeatureFusion(feature_dim=64, ego_state_dim=21)

    fused, gate = fusion(feature, ego_state, return_gate=True)
    assert_equal(tuple(fused.shape), tuple(feature.shape), "channel-first fused shape should match feature")
    assert_equal(tuple(gate.shape), (2, 64, 1, 1), "channel-first gate should broadcast spatial dims")
    assert_close(fused, feature * gate, "channel-first fusion should match feature * gate")


def exercise_se_layer_gate() -> None:
    feature = torch.ones(2, 5, 16)
    condition = torch.randn(2, 16)
    se_layer = SELayer(16)
    gated = se_layer(feature, condition)
    assert_equal(tuple(gated.shape), tuple(feature.shape), "SELayer should preserve feature shape")


def exercise_encoder_and_fusion_gradients() -> None:
    feature = torch.randn(2, 8, requires_grad=True)
    ego_state = torch.randn(2, 4)
    target = torch.randn_like(feature)
    fusion = EgoStateFeatureFusion(feature_dim=8, ego_state_dim=4)

    loss = (fusion(feature, ego_state) - target).pow(2).mean()
    loss.backward()
    grad_norm = fusion.ego_encoder.net[0].weight.grad.abs().sum().item()
    if grad_norm <= 0:
        raise AssertionError("ego-state encoder should receive gradients from fused feature loss")


def exercise_single_ego_state() -> None:
    encoder = EgoStateEncoder(ego_state_dim=4, embed_dim=8)
    encoded = encoder(torch.randn(4))
    assert_equal(tuple(encoded.shape), (1, 8), "single ego state should be promoted to batch size 1")


def exercise_dcnv4_bottleneck_block_with_fake_dcn() -> None:
    feature = torch.randn(2, 64, 6, 5)
    block = DCNv4BottleneckBlock(
        channels=64,
        bottleneck_channels=16,
        dcn_group=1,
        dcn_cls=FakeDCNv4,
    )
    output = block(feature)
    assert_equal(tuple(output.shape), tuple(feature.shape), "DCNv4 bottleneck should preserve NCHW shape")


def exercise_full_adapter_nchw_with_fake_dcn() -> None:
    feature = torch.randn(2, 64, 6, 5)
    ego_state = torch.randn(2, 21)
    adapter = EgoStateDCNv4Adapter(
        feature_dim=64,
        ego_state_dim=21,
        bottleneck_channels=16,
        dcn_group=1,
        dcn_cls=FakeDCNv4,
    )
    residual = adapter(feature, ego_state, return_residual=True)
    adapted = adapter(feature, ego_state)
    assert_equal(tuple(residual.shape), tuple(feature.shape), "adapter residual should preserve NCHW shape")
    assert_close(residual, torch.zeros_like(residual), "zero-init adapter residual should start as zero")
    assert_close(adapted, feature, "zero-init full adapter should initially be identity")


def exercise_full_adapter_tokens_with_fake_dcn() -> None:
    feature = torch.randn(2, 30, 64)
    ego_state = torch.randn(2, 21)
    adapter = EgoStateDCNv4Adapter(
        feature_dim=64,
        ego_state_dim=21,
        bottleneck_channels=16,
        dcn_group=1,
        dcn_cls=FakeDCNv4,
    )
    residual = adapter.residual(feature, ego_state, spatial_shape=(6, 5))
    adapted = adapter(feature, ego_state, spatial_shape=(6, 5))
    assert_equal(tuple(residual.shape), tuple(feature.shape), "adapter residual should preserve token shape")
    assert_close(residual, torch.zeros_like(residual), "zero-init token residual should start as zero")
    assert_close(adapted, feature, "zero-init token adapter should initially be identity")


def exercise_full_adapter_zero_init_still_learns() -> None:
    feature = torch.randn(2, 64, 4, 4, requires_grad=True)
    ego_state = torch.randn(2, 21)
    target = torch.randn_like(feature)
    adapter = EgoStateDCNv4Adapter(
        feature_dim=64,
        ego_state_dim=21,
        bottleneck_channels=16,
        dcn_group=1,
        dcn_cls=FakeDCNv4,
    )
    loss = (adapter(feature, ego_state) - target).pow(2).mean()
    loss.backward()
    grad_norm = adapter.bottlenecks[-1].up.weight.grad.abs().sum().item()
    if grad_norm <= 0:
        raise AssertionError("zero-init final 1x1 conv should still receive gradients")


def exercise_plan_query_adapter_identity_and_gradients() -> None:
    feature = torch.randn(3, 48, 256, requires_grad=True)
    ego_state = torch.randn(3, 21)
    target = torch.randn_like(feature)
    adapter = EgoStatePlanQueryAdapter(feature_dim=256, ego_state_dim=21, hidden_dim=128)

    residual = adapter(feature, ego_state, return_residual=True)
    adapted = adapter(feature, ego_state)
    assert_equal(tuple(residual.shape), tuple(feature.shape), "plan-query residual should preserve token shape")
    assert_close(residual, torch.zeros_like(residual), "zero-init plan-query residual should start as zero")
    assert_close(adapted, feature, "zero-init plan-query adapter should initially be identity")

    loss = (adapter(feature, ego_state) - target).pow(2).mean()
    loss.backward()
    grad_norm = adapter.residual_mlp[-1].weight.grad.abs().sum().item()
    if grad_norm <= 0:
        raise AssertionError("zero-init plan-query output projection should still receive gradients")


def main() -> None:
    torch.manual_seed(0)
    exercise_vector_feature()
    exercise_batch_first_tokens()
    exercise_sequence_first_tokens()
    exercise_channel_first_feature()
    exercise_se_layer_gate()
    exercise_encoder_and_fusion_gradients()
    exercise_single_ego_state()
    exercise_dcnv4_bottleneck_block_with_fake_dcn()
    exercise_full_adapter_nchw_with_fake_dcn()
    exercise_full_adapter_tokens_with_fake_dcn()
    exercise_full_adapter_zero_init_still_learns()
    exercise_plan_query_adapter_identity_and_gradients()
    print("ego-state adapter smoke tests passed")


if __name__ == "__main__":
    main()
