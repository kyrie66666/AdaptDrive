"""Ego-state conditioned feature adapters.

This module is intentionally standalone. Importing it does not modify any
existing HiP-AD SAC finetuning path.

Fusion stage:
    ego_state -> MLP -> SSR-style SE gate -> ego-conditioned feature

Residual adapter:
    fused feature -> 2 * (1x1 conv down -> BN -> DCNv4 -> 1x1 conv up)
"""

import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn


def load_adapter_state_strict_alpha_compat(
    module: nn.Module,
    module_state: Dict[str, torch.Tensor],
    *,
    key: str,
    adapter_mode: str,
) -> None:
    """Strict adapter restore with compatibility for newly added feature alpha gates."""

    if not isinstance(module_state, dict) or not module_state:
        raise RuntimeError(f"evaluation adapter_mode={adapter_mode!r} requires checkpoint state {key}")
    incompatible = module.load_state_dict(module_state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    allowed_missing = []
    if key == "feature_dcnv4_adapter":
        allowed_missing = [name for name in missing if name.startswith("residual_alpha_by_level.")]
    allowed_missing_set = set(allowed_missing)
    disallowed_missing = [name for name in missing if name not in allowed_missing_set]
    if disallowed_missing or unexpected:
        raise RuntimeError(
            f"adapter state mismatch for {key}: missing={disallowed_missing}, unexpected={unexpected}"
        )


def _validate_positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _ensure_batched_ego_state(ego_state: torch.Tensor, ego_state_dim: int) -> torch.Tensor:
    if ego_state.dim() == 1:
        ego_state = ego_state.unsqueeze(0)
    if ego_state.dim() != 2:
        raise ValueError(f"ego_state must have shape [D] or [B, D], got {tuple(ego_state.shape)}")
    if ego_state.shape[-1] != ego_state_dim:
        raise ValueError(
            f"ego_state last dim must be {ego_state_dim}, got shape {tuple(ego_state.shape)}"
        )
    return ego_state


def _default_bottleneck_channels(feature_dim: int, reduction: int) -> int:
    reduction = _validate_positive_int("reduction", reduction)
    raw_channels = max(16, int(math.ceil(feature_dim / reduction)))
    return int(math.ceil(raw_channels / 16) * 16)


def _choose_dcn_group(channels: int, preferred_group: int) -> int:
    preferred_group = _validate_positive_int("preferred_group", preferred_group)
    for group in range(min(preferred_group, channels), 0, -1):
        if channels % group == 0 and (channels // group) % 16 == 0:
            return group
    raise ValueError(
        f"DCNv4 requires channels/group to be a multiple of 16, "
        f"but no valid group was found for channels={channels}"
    )


def _choose_norm_group(channels: int, preferred_group: int) -> int:
    preferred_group = _validate_positive_int("preferred_group", preferred_group)
    for group in range(min(preferred_group, channels), 0, -1):
        if channels % group == 0:
            return group
    return 1


def _build_spatial_norm(channels: int, norm_type: str = "group", norm_groups: int = 8) -> nn.Module:
    norm_type = str(norm_type or "group").lower()
    if norm_type in {"none", "identity"}:
        return nn.Identity()
    if norm_type in {"batch", "bn", "batch_norm"}:
        return nn.BatchNorm2d(channels)
    if norm_type in {"group", "gn", "group_norm"}:
        return nn.GroupNorm(_choose_norm_group(channels, norm_groups), channels)
    if norm_type in {"layer", "ln", "layer_norm"}:
        # Channel-first feature maps use GroupNorm(1, C) as a batch-stat-free
        # layer-style normalization over each sample.
        return nn.GroupNorm(1, channels)
    raise ValueError(f"Unsupported norm_type={norm_type!r}; expected batch, group, layer, or identity")


def _normalize_levels(levels: Iterable[int]) -> Tuple[int, ...]:
    normalized: List[int] = []
    for level in levels:
        level = int(level)
        if level < 0:
            raise ValueError(f"feature adapter levels must be non-negative, got {level}")
        if level not in normalized:
            normalized.append(level)
    if not normalized:
        raise ValueError("feature adapter levels must contain at least one level")
    return tuple(normalized)


def _load_dcnv4_class() -> Callable[..., nn.Module]:
    try:
        import DCNv4  # type: ignore
    except Exception as exc:
        raise ImportError(
            "DCNv4 is not importable. Install the DCNv4 operator in the active environment "
            "or add its package root through DCNV4_ROOT/PYTHONPATH before enabling the adapter."
        ) from exc
    return DCNv4.DCNv4


def _feature_to_nchw(
    feature: torch.Tensor,
    feature_dim: int,
    spatial_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, str]:
    if feature.dim() == 4 and feature.shape[1] == feature_dim:
        return feature, "nchw"
    if feature.dim() == 4 and feature.shape[-1] == feature_dim:
        return feature.permute(0, 3, 1, 2).contiguous(), "nhwc"
    if feature.dim() == 3 and feature.shape[-1] == feature_dim:
        if spatial_shape is None:
            raise ValueError("spatial_shape=(H, W) is required for [B, H*W, C] token features")
        height, width = spatial_shape
        if feature.shape[1] != height * width:
            raise ValueError(
                f"token length {feature.shape[1]} does not match spatial_shape={spatial_shape}"
            )
        batch_size = feature.shape[0]
        nchw = feature.view(batch_size, height, width, feature_dim).permute(0, 3, 1, 2).contiguous()
        return nchw, "blc"
    raise ValueError(
        f"feature must be [B, C, H, W], [B, H, W, C], or [B, H*W, C] with C={feature_dim}; "
        f"got {tuple(feature.shape)}"
    )


def _nchw_to_feature(feature: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "nchw":
        return feature
    if layout == "nhwc":
        return feature.permute(0, 2, 3, 1).contiguous()
    if layout == "blc":
        batch_size, channels, height, width = feature.shape
        return feature.permute(0, 2, 3, 1).contiguous().view(batch_size, height * width, channels)
    raise ValueError(f"Unsupported feature layout: {layout}")


def _expand_channel_condition(
    condition: torch.Tensor,
    feature: torch.Tensor,
    feature_dim: int,
    batch_first: bool,
) -> torch.Tensor:
    """Expand a [B, C] condition tensor to a feature-broadcastable shape."""

    if condition.dim() != 2:
        raise ValueError(f"condition must have shape [B, C], got {tuple(condition.shape)}")
    if condition.shape[-1] != feature_dim:
        raise ValueError(
            f"condition last dim must be {feature_dim}, got shape {tuple(condition.shape)}"
        )

    batch_size = condition.shape[0]
    if feature.dim() < 2:
        raise ValueError(f"feature must have at least 2 dims, got {tuple(feature.shape)}")

    if feature.dim() == 2:
        if feature.shape != condition.shape:
            raise ValueError(
                f"feature shape {tuple(feature.shape)} is incompatible with condition {tuple(condition.shape)}"
            )
        return condition

    if feature.shape[-1] == feature_dim:
        if batch_first:
            if feature.shape[0] != batch_size:
                raise ValueError(
                    f"batch-first feature batch {feature.shape[0]} != condition batch {batch_size}"
                )
            return condition.view(batch_size, *([1] * (feature.dim() - 2)), feature_dim)

        if feature.shape[1] != batch_size:
            raise ValueError(
                f"sequence-first feature batch {feature.shape[1]} != condition batch {batch_size}"
            )
        return condition.view(1, batch_size, *([1] * (feature.dim() - 3)), feature_dim)

    if feature.shape[1] == feature_dim:
        if feature.shape[0] != batch_size:
            raise ValueError(f"channel-first feature batch {feature.shape[0]} != condition batch {batch_size}")
        return condition.view(batch_size, feature_dim, *([1] * (feature.dim() - 2)))

    raise ValueError(
        f"feature shape {tuple(feature.shape)} does not expose feature_dim={feature_dim} "
        "as the last dim or channel-first dim"
    )


class SELayer(nn.Module):
    """SSR-style channel gate: condition -> Linear -> ReLU -> Linear -> Sigmoid."""

    def __init__(
        self,
        channels: int,
        act_layer: nn.Module = nn.ReLU,
        gate_layer: nn.Module = nn.Sigmoid,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.channels = _validate_positive_int("channels", channels)
        self.batch_first = bool(batch_first)
        self.mlp_reduce = nn.Linear(self.channels, self.channels)
        self.act1 = act_layer()
        self.mlp_expand = nn.Linear(self.channels, self.channels)
        self.gate = gate_layer()

    def gate_from_condition(self, condition: torch.Tensor) -> torch.Tensor:
        condition = self.mlp_reduce(condition)
        condition = self.act1(condition)
        condition = self.mlp_expand(condition)
        return self.gate(condition)

    def expand_gate(self, gate: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        if gate.dim() == 2:
            gate = _expand_channel_condition(gate, feature, self.channels, self.batch_first)
        return gate.to(device=feature.device, dtype=feature.dtype)

    def forward(
        self,
        feature: torch.Tensor,
        condition: torch.Tensor,
        return_gate: bool = False,
    ) -> torch.Tensor:
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)
        gate = self.gate_from_condition(condition)
        gate = self.expand_gate(gate, feature)
        fused = feature * gate
        if return_gate:
            return fused, gate
        return fused


class EgoStateEncoder(nn.Module):
    """Encode continuous ego state into the same channel width as a feature."""

    def __init__(
        self,
        ego_state_dim: int = 21,
        embed_dim: int = 256,
        hidden_dim: Optional[int] = None,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.ego_state_dim = _validate_positive_int("ego_state_dim", ego_state_dim)
        self.embed_dim = _validate_positive_int("embed_dim", embed_dim)
        hidden_dim = _validate_positive_int("hidden_dim", hidden_dim or max(self.embed_dim // 2, self.ego_state_dim))

        layers = [
            nn.Linear(self.ego_state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.embed_dim),
            nn.ReLU(inplace=True),
        ]
        if use_layer_norm:
            layers.append(nn.LayerNorm(self.embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, ego_state: torch.Tensor) -> torch.Tensor:
        ego_state = _ensure_batched_ego_state(ego_state, self.ego_state_dim)
        first_linear = self.net[0]
        ego_state = ego_state.to(device=first_linear.weight.device, dtype=first_linear.weight.dtype)
        return self.net(ego_state)


class EgoStateFeatureFusion(nn.Module):
    """Fuse ego state into feature using the same SE idea as SSR navigation fusion.

    This is only the first part of the adapter structure:

        ego_state -> MLP -> SSR-style SE gate -> ego-conditioned feature

    The two deformable-conv bottlenecks that produce the final residual feature
    should be added after the deformable convolution source is fixed.
    """

    def __init__(
        self,
        feature_dim: int,
        ego_state_dim: int = 21,
        ego_hidden_dim: Optional[int] = None,
        batch_first: bool = True,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.feature_dim = _validate_positive_int("feature_dim", feature_dim)
        self.ego_state_dim = _validate_positive_int("ego_state_dim", ego_state_dim)
        self.ego_encoder = EgoStateEncoder(
            ego_state_dim=self.ego_state_dim,
            embed_dim=self.feature_dim,
            hidden_dim=ego_hidden_dim,
            use_layer_norm=use_layer_norm,
        )
        self.se = SELayer(self.feature_dim, batch_first=batch_first)

    def condition(self, ego_state: torch.Tensor) -> torch.Tensor:
        return self.ego_encoder(ego_state)

    def gate(self, feature: torch.Tensor, ego_state: torch.Tensor) -> torch.Tensor:
        param = next(self.parameters())
        if feature.device != param.device:
            raise ValueError(
                f"feature is on {feature.device}, but fusion module is on {param.device}; "
                "move the module to the feature device before calling it"
            )
        ego_embed = self.ego_encoder(ego_state)
        gate = self.se.gate_from_condition(ego_embed)
        return self.se.expand_gate(gate, feature)

    def forward(
        self,
        feature: torch.Tensor,
        ego_state: torch.Tensor,
        return_gate: bool = False,
    ) -> torch.Tensor:
        param = next(self.parameters())
        if feature.device != param.device:
            raise ValueError(
                f"feature is on {feature.device}, but fusion module is on {param.device}; "
                "move the module to the feature device before calling it"
            )
        ego_embed = self.ego_encoder(ego_state)
        return self.se(feature, ego_embed, return_gate=return_gate)


class DCNv4BottleneckBlock(nn.Module):
    """1x1 down -> norm -> DCNv4 -> 1x1 up bottleneck for spatial feature maps."""

    def __init__(
        self,
        channels: int,
        bottleneck_channels: int,
        dcn_group: Optional[int] = None,
        dcn_kernel_size: int = 3,
        dcn_offset_scale: float = 1.0,
        dcn_cls: Optional[Callable[..., nn.Module]] = None,
        zero_init_output: bool = False,
        norm_type: str = "group",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.channels = _validate_positive_int("channels", channels)
        self.bottleneck_channels = _validate_positive_int("bottleneck_channels", bottleneck_channels)
        self.dcn_group = dcn_group or _choose_dcn_group(self.bottleneck_channels, preferred_group=4)
        if dcn_kernel_size % 2 != 1:
            raise ValueError(f"dcn_kernel_size must be odd, got {dcn_kernel_size}")

        dcn_cls = dcn_cls or _load_dcnv4_class()
        self.down = nn.Conv2d(self.channels, self.bottleneck_channels, kernel_size=1, bias=False)
        self.norm = _build_spatial_norm(self.bottleneck_channels, norm_type=norm_type, norm_groups=norm_groups)
        self.dcn = dcn_cls(
            channels=self.bottleneck_channels,
            kernel_size=dcn_kernel_size,
            pad=dcn_kernel_size // 2,
            group=self.dcn_group,
            offset_scale=dcn_offset_scale,
        )
        self.up = nn.Conv2d(self.bottleneck_channels, self.channels, kernel_size=1, bias=True)
        if zero_init_output:
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.dim() != 4 or feature.shape[1] != self.channels:
            raise ValueError(
                f"DCNv4BottleneckBlock expects [B, {self.channels}, H, W], got {tuple(feature.shape)}"
            )
        x = self.down(feature)
        x = self.norm(x)
        batch_size, channels, height, width = x.shape
        tokens = x.permute(0, 2, 3, 1).contiguous().view(batch_size, height * width, channels)
        tokens = self.dcn(tokens, shape=(height, width))
        x = tokens.view(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()
        return self.up(x)


class EgoStateDCNv4Adapter(nn.Module):
    """Full adapter: ego-state feature fusion followed by two DCNv4 bottlenecks."""

    def __init__(
        self,
        feature_dim: int,
        ego_state_dim: int = 21,
        ego_hidden_dim: Optional[int] = None,
        bottleneck_channels: Optional[int] = None,
        bottleneck_reduction: int = 4,
        dcn_group: Optional[int] = None,
        dcn_kernel_size: int = 3,
        dcn_offset_scale: float = 1.0,
        use_layer_norm: bool = True,
        zero_init_residual: bool = True,
        residual_scale: float = 1.0,
        norm_type: str = "group",
        norm_groups: int = 8,
        dcn_cls: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.feature_dim = _validate_positive_int("feature_dim", feature_dim)
        bottleneck_channels = bottleneck_channels or _default_bottleneck_channels(
            self.feature_dim,
            bottleneck_reduction,
        )
        bottleneck_channels = _validate_positive_int("bottleneck_channels", bottleneck_channels)
        dcn_group = dcn_group or _choose_dcn_group(bottleneck_channels, preferred_group=4)
        self.residual_scale = float(residual_scale)

        self.fusion = EgoStateFeatureFusion(
            feature_dim=self.feature_dim,
            ego_state_dim=ego_state_dim,
            ego_hidden_dim=ego_hidden_dim,
            batch_first=True,
            use_layer_norm=use_layer_norm,
        )
        dcn_cls = dcn_cls or _load_dcnv4_class()
        self.bottlenecks = nn.Sequential(
            DCNv4BottleneckBlock(
                channels=self.feature_dim,
                bottleneck_channels=bottleneck_channels,
                dcn_group=dcn_group,
                dcn_kernel_size=dcn_kernel_size,
                dcn_offset_scale=dcn_offset_scale,
                dcn_cls=dcn_cls,
                zero_init_output=False,
                norm_type=norm_type,
                norm_groups=norm_groups,
            ),
            DCNv4BottleneckBlock(
                channels=self.feature_dim,
                bottleneck_channels=bottleneck_channels,
                dcn_group=dcn_group,
                dcn_kernel_size=dcn_kernel_size,
                dcn_offset_scale=dcn_offset_scale,
                dcn_cls=dcn_cls,
                zero_init_output=zero_init_residual,
                norm_type=norm_type,
                norm_groups=norm_groups,
            ),
        )

    def residual(
        self,
        feature: torch.Tensor,
        ego_state: torch.Tensor,
        spatial_shape: Optional[Tuple[int, int]] = None,
        apply_residual_scale: bool = True,
    ) -> torch.Tensor:
        nchw, layout = _feature_to_nchw(feature, self.feature_dim, spatial_shape=spatial_shape)
        fused = self.fusion(nchw, ego_state)
        residual = self.bottlenecks(fused)
        if apply_residual_scale:
            residual = residual * self.residual_scale
        return _nchw_to_feature(residual, layout)

    def forward(
        self,
        feature: torch.Tensor,
        ego_state: torch.Tensor,
        spatial_shape: Optional[Tuple[int, int]] = None,
        return_residual: bool = False,
    ) -> torch.Tensor:
        residual = self.residual(feature, ego_state, spatial_shape=spatial_shape)
        if return_residual:
            return residual
        return feature + residual


class EgoStateDCNv4FeatureAdapter(nn.Module):
    """Apply ego-state DCNv4 adapters to selected formatted HiP-AD FPN levels."""

    def __init__(
        self,
        feature_dim: int = 256,
        ego_state_dim: int = 21,
        levels: Sequence[int] = (0, 1, 2, 3),
        ego_hidden_dim: Optional[int] = None,
        bottleneck_channels: Optional[int] = None,
        bottleneck_reduction: int = 4,
        dcn_group: Optional[int] = None,
        dcn_kernel_size: int = 3,
        dcn_offset_scale: float = 1.0,
        use_layer_norm: bool = True,
        zero_init_residual: bool = True,
        residual_scale: float = 1.0,
        norm_type: str = "group",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.feature_dim = _validate_positive_int("feature_dim", feature_dim)
        self.ego_state_dim = _validate_positive_int("ego_state_dim", ego_state_dim)
        self.levels = _normalize_levels(levels)
        self.adapters = nn.ModuleDict()
        self.residual_alpha_by_level = nn.ParameterDict()
        for level in self.levels:
            self.adapters[str(level)] = EgoStateDCNv4Adapter(
                feature_dim=self.feature_dim,
                ego_state_dim=self.ego_state_dim,
                ego_hidden_dim=ego_hidden_dim,
                bottleneck_channels=bottleneck_channels,
                bottleneck_reduction=bottleneck_reduction,
                dcn_group=dcn_group,
                dcn_kernel_size=dcn_kernel_size,
                dcn_offset_scale=dcn_offset_scale,
                use_layer_norm=use_layer_norm,
                zero_init_residual=zero_init_residual,
                residual_scale=residual_scale,
                norm_type=norm_type,
                norm_groups=norm_groups,
            )
            self.residual_alpha_by_level[str(level)] = nn.Parameter(torch.ones(()))
        self.last_residual_l2_by_level: Dict[int, float] = {}
        self.last_raw_residual_l2_by_level: Dict[int, float] = {}
        self.last_effective_delta_l2_by_level: Dict[int, float] = {}
        self.last_alpha_by_level: Dict[int, float] = {}

    def _adapt_level(
        self,
        level_feature: torch.Tensor,
        ego_state: torch.Tensor,
        level: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if level_feature.dim() != 5 or level_feature.shape[2] != self.feature_dim:
            raise ValueError(
                f"feature level L{level} must have shape [B, Ncam, {self.feature_dim}, H, W], "
                f"got {tuple(level_feature.shape)}"
            )
        batch_size, num_cams, channels, height, width = level_feature.shape
        flat_feature = level_feature.reshape(batch_size * num_cams, channels, height, width)
        flat_ego_state = ego_state.repeat_interleave(num_cams, dim=0).to(
            device=flat_feature.device,
            dtype=flat_feature.dtype,
        )
        adapter = self.adapters[str(level)]
        raw_residual = adapter.residual(
            flat_feature,
            flat_ego_state,
            apply_residual_scale=False,
        )
        alpha = self.residual_alpha_by_level[str(level)].to(device=raw_residual.device, dtype=raw_residual.dtype)
        effective_delta = raw_residual * (float(adapter.residual_scale) * alpha)
        adapted = flat_feature + effective_delta
        base_detached = flat_feature.detach().float()
        raw_residual_detached = raw_residual.detach().float()
        effective_delta_detached = effective_delta.detach().float()
        adapted_detached = adapted.detach().float()
        raw_residual_l2 = torch.linalg.vector_norm(raw_residual_detached.flatten(1), dim=1).mean()
        effective_delta_l2 = torch.linalg.vector_norm(effective_delta_detached.flatten(1), dim=1).mean()
        base_rms = base_detached.square().mean().sqrt()
        raw_residual_rms = raw_residual_detached.square().mean().sqrt()
        effective_delta_rms = effective_delta_detached.square().mean().sqrt()
        adapted_rms = adapted_detached.square().mean().sqrt()
        rms_epsilon = torch.tensor(1e-8, device=base_rms.device, dtype=base_rms.dtype)
        safe_base_rms = base_rms.clamp_min(rms_epsilon)
        effective_to_base_ratio = effective_delta_rms / safe_base_rms
        adapted_to_base_ratio = adapted_rms / safe_base_rms
        adapted = adapted.reshape(batch_size, num_cams, channels, height, width)
        alpha_value = alpha.detach().float()
        return (
            adapted,
            raw_residual_l2,
            effective_delta_l2,
            alpha_value,
            base_rms,
            raw_residual_rms,
            effective_delta_rms,
            effective_to_base_ratio,
            adapted_to_base_ratio,
        )

    def forward(
        self,
        feature_groups: Sequence[Sequence[torch.Tensor]],
        ego_state: torch.Tensor,
        return_metrics: bool = False,
    ):
        ego_state = _ensure_batched_ego_state(ego_state, self.ego_state_dim)
        adapted_groups = []
        raw_residual_l2_by_level: Dict[int, List[torch.Tensor]] = {level: [] for level in self.levels}
        effective_delta_l2_by_level: Dict[int, List[torch.Tensor]] = {level: [] for level in self.levels}
        alpha_by_level: Dict[int, List[torch.Tensor]] = {level: [] for level in self.levels}
        base_rms_by_level: Dict[int, List[torch.Tensor]] = {level: [] for level in self.levels}
        raw_residual_rms_by_level: Dict[int, List[torch.Tensor]] = {level: [] for level in self.levels}
        effective_delta_rms_by_level: Dict[int, List[torch.Tensor]] = {level: [] for level in self.levels}
        effective_to_base_ratio_by_level: Dict[int, List[torch.Tensor]] = {level: [] for level in self.levels}
        adapted_to_base_ratio_by_level: Dict[int, List[torch.Tensor]] = {level: [] for level in self.levels}
        for group in feature_groups:
            adapted_levels = list(group)
            for level in self.levels:
                if level >= len(adapted_levels):
                    raise ValueError(
                        f"feature adapter requested L{level}, but inverse feature map group has "
                        f"only {len(adapted_levels)} levels"
                    )
                (
                    adapted_levels[level],
                    raw_residual_l2,
                    effective_delta_l2,
                    alpha_value,
                    base_rms,
                    raw_residual_rms,
                    effective_delta_rms,
                    effective_to_base_ratio,
                    adapted_to_base_ratio,
                ) = self._adapt_level(adapted_levels[level], ego_state, level)
                raw_residual_l2_by_level[level].append(raw_residual_l2)
                effective_delta_l2_by_level[level].append(effective_delta_l2)
                alpha_by_level[level].append(alpha_value)
                base_rms_by_level[level].append(base_rms)
                raw_residual_rms_by_level[level].append(raw_residual_rms)
                effective_delta_rms_by_level[level].append(effective_delta_rms)
                effective_to_base_ratio_by_level[level].append(effective_to_base_ratio)
                adapted_to_base_ratio_by_level[level].append(adapted_to_base_ratio)
            adapted_groups.append(adapted_levels)

        metrics = {
            f"feature_adapter_raw_residual_l2_L{level}": float(torch.stack(values).mean().detach().cpu().item())
            for level, values in raw_residual_l2_by_level.items()
            if values
        }
        metrics.update(
            {
                f"feature_adapter_effective_delta_l2_L{level}": float(
                    torch.stack(values).mean().detach().cpu().item()
                )
                for level, values in effective_delta_l2_by_level.items()
                if values
            }
        )
        metrics.update(
            {
                f"feature_adapter_alpha_L{level}": float(torch.stack(values).mean().detach().cpu().item())
                for level, values in alpha_by_level.items()
                if values
            }
        )
        metrics.update(
            {
                f"feature_adapter_residual_l2_L{level}": float(torch.stack(values).mean().detach().cpu().item())
                for level, values in effective_delta_l2_by_level.items()
                if values
            }
        )
        for metric_name, values_by_level in (
            ("feature_adapter_base_rms", base_rms_by_level),
            ("feature_adapter_raw_residual_rms", raw_residual_rms_by_level),
            ("feature_adapter_effective_delta_rms", effective_delta_rms_by_level),
            ("feature_adapter_effective_to_base_ratio", effective_to_base_ratio_by_level),
            ("feature_adapter_adapted_to_base_ratio", adapted_to_base_ratio_by_level),
        ):
            metrics.update(
                {
                    f"{metric_name}_L{level}": float(torch.stack(values).mean().detach().cpu().item())
                    for level, values in values_by_level.items()
                    if values
                }
            )
        self.last_raw_residual_l2_by_level = {
            level: metrics[f"feature_adapter_raw_residual_l2_L{level}"]
            for level in self.levels
            if f"feature_adapter_raw_residual_l2_L{level}" in metrics
        }
        self.last_effective_delta_l2_by_level = {
            level: metrics[f"feature_adapter_effective_delta_l2_L{level}"]
            for level in self.levels
            if f"feature_adapter_effective_delta_l2_L{level}" in metrics
        }
        self.last_alpha_by_level = {
            level: metrics[f"feature_adapter_alpha_L{level}"]
            for level in self.levels
            if f"feature_adapter_alpha_L{level}" in metrics
        }
        self.last_residual_l2_by_level = {
            level: metrics[f"feature_adapter_residual_l2_L{level}"]
            for level in self.levels
            if f"feature_adapter_residual_l2_L{level}" in metrics
        }
        if return_metrics:
            return adapted_groups, metrics
        return adapted_groups


class EgoStatePlanQueryAdapter(nn.Module):
    """Zero-init residual adapter for HiP-AD plan-align query tokens.

    The SAC finetuning path stores and reuses `plan_align_query` with shape
    [B, num_modes, C]. A pure SE gate would immediately rescale those tokens,
    so this wrapper uses the existing ego-state fusion only as the conditioned
    input to a zero-initialized residual MLP:

        output = plan_query + residual_scale * zero_init_mlp(fusion(plan_query, ego_state))

    At initialization the residual is exactly zero, which keeps legacy policy
    behavior unchanged when the adapter is first enabled.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        ego_state_dim: int = 21,
        ego_hidden_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        residual_scale: float = 1.0,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.feature_dim = _validate_positive_int("feature_dim", feature_dim)
        self.ego_state_dim = _validate_positive_int("ego_state_dim", ego_state_dim)
        hidden_dim = _validate_positive_int("hidden_dim", hidden_dim or self.feature_dim)
        self.residual_scale = float(residual_scale)

        self.fusion = EgoStateFeatureFusion(
            feature_dim=self.feature_dim,
            ego_state_dim=self.ego_state_dim,
            ego_hidden_dim=ego_hidden_dim,
            batch_first=True,
            use_layer_norm=use_layer_norm,
        )
        self.norm = nn.LayerNorm(self.feature_dim) if use_layer_norm else nn.Identity()
        layers = [
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        out_proj = nn.Linear(hidden_dim, self.feature_dim)
        nn.init.zeros_(out_proj.weight)
        nn.init.zeros_(out_proj.bias)
        layers.append(out_proj)
        self.residual_mlp = nn.Sequential(*layers)

    def residual(
        self,
        feature: torch.Tensor,
        ego_state: torch.Tensor,
        return_gate: bool = False,
    ) -> torch.Tensor:
        if feature.dim() != 3 or feature.shape[-1] != self.feature_dim:
            raise ValueError(
                f"EgoStatePlanQueryAdapter expects [B, L, {self.feature_dim}], "
                f"got {tuple(feature.shape)}"
            )
        if return_gate:
            fused, gate = self.fusion(feature, ego_state, return_gate=True)
        else:
            fused = self.fusion(feature, ego_state)
            gate = None
        residual = self.residual_mlp(self.norm(fused)) * self.residual_scale
        if return_gate:
            return residual, gate
        return residual

    def forward(
        self,
        feature: torch.Tensor,
        ego_state: torch.Tensor,
        return_residual: bool = False,
        return_gate: bool = False,
    ) -> torch.Tensor:
        if return_gate:
            residual, gate = self.residual(feature, ego_state, return_gate=True)
        else:
            residual = self.residual(feature, ego_state)
            gate = None
        output = residual if return_residual else feature + residual
        if return_gate:
            return output, gate
        return output


def build_ego_state_feature_fusion(
    feature_dim: int,
    ego_state_dim: int = 21,
    ego_hidden_dim: Optional[int] = None,
    batch_first: bool = True,
    use_layer_norm: bool = True,
) -> EgoStateFeatureFusion:
    return EgoStateFeatureFusion(
        feature_dim=feature_dim,
        ego_state_dim=ego_state_dim,
        ego_hidden_dim=ego_hidden_dim,
        batch_first=batch_first,
        use_layer_norm=use_layer_norm,
    )


def build_ego_state_dcnv4_adapter(
    feature_dim: int,
    ego_state_dim: int = 21,
    ego_hidden_dim: Optional[int] = None,
    bottleneck_channels: Optional[int] = None,
    bottleneck_reduction: int = 4,
    dcn_group: Optional[int] = None,
    dcn_kernel_size: int = 3,
    dcn_offset_scale: float = 1.0,
    use_layer_norm: bool = True,
    zero_init_residual: bool = True,
    residual_scale: float = 1.0,
    norm_type: str = "group",
    norm_groups: int = 8,
) -> EgoStateDCNv4Adapter:
    return EgoStateDCNv4Adapter(
        feature_dim=feature_dim,
        ego_state_dim=ego_state_dim,
        ego_hidden_dim=ego_hidden_dim,
        bottleneck_channels=bottleneck_channels,
        bottleneck_reduction=bottleneck_reduction,
        dcn_group=dcn_group,
        dcn_kernel_size=dcn_kernel_size,
        dcn_offset_scale=dcn_offset_scale,
        use_layer_norm=use_layer_norm,
        zero_init_residual=zero_init_residual,
        residual_scale=residual_scale,
        norm_type=norm_type,
        norm_groups=norm_groups,
    )


def build_ego_state_dcnv4_feature_adapter(
    feature_dim: int = 256,
    ego_state_dim: int = 21,
    levels: Sequence[int] = (0, 1, 2, 3),
    ego_hidden_dim: Optional[int] = None,
    bottleneck_channels: Optional[int] = None,
    bottleneck_reduction: int = 4,
    dcn_group: Optional[int] = None,
    dcn_kernel_size: int = 3,
    dcn_offset_scale: float = 1.0,
    use_layer_norm: bool = True,
    zero_init_residual: bool = True,
    residual_scale: float = 1.0,
    norm_type: str = "group",
    norm_groups: int = 8,
) -> EgoStateDCNv4FeatureAdapter:
    return EgoStateDCNv4FeatureAdapter(
        feature_dim=feature_dim,
        ego_state_dim=ego_state_dim,
        levels=levels,
        ego_hidden_dim=ego_hidden_dim,
        bottleneck_channels=bottleneck_channels,
        bottleneck_reduction=bottleneck_reduction,
        dcn_group=dcn_group,
        dcn_kernel_size=dcn_kernel_size,
        dcn_offset_scale=dcn_offset_scale,
        use_layer_norm=use_layer_norm,
        zero_init_residual=zero_init_residual,
        residual_scale=residual_scale,
        norm_type=norm_type,
        norm_groups=norm_groups,
    )


def build_ego_state_plan_query_adapter(
    feature_dim: int = 256,
    ego_state_dim: int = 21,
    ego_hidden_dim: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    residual_scale: float = 1.0,
    dropout: float = 0.0,
    use_layer_norm: bool = True,
) -> EgoStatePlanQueryAdapter:
    return EgoStatePlanQueryAdapter(
        feature_dim=feature_dim,
        ego_state_dim=ego_state_dim,
        ego_hidden_dim=ego_hidden_dim,
        hidden_dim=hidden_dim,
        residual_scale=residual_scale,
        dropout=dropout,
        use_layer_norm=use_layer_norm,
    )
