#!/usr/bin/env python3
"""Strict checkpoint loading for the HiP-AD leaderboard adapter bridge.

This module is deliberately separate from the existing clean leaderboard agent
and from the SAC trainer.  It only validates and restores the deployment parts
of a SAC checkpoint:

* ``agent["hipad_trainable"]``;
* ``agent["feature_dcnv4_adapter"]``.

Critic/value/replay/optimizer/prediction-head state is intentionally not loaded
for leaderboard inference.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import torch


FOUR_LEVELS: Tuple[int, ...] = (0, 1, 2, 3)
EXPECTED_ADAPTER_MODE = "dcnv4_feature"
EXPECTED_EGO_STATE_DIM = 21
EXPECTED_FEATURE_DIM = 256
CURRENT_TRAINING_SIGNATURE_VERSION = 8
SUPPORTED_DEPLOYMENT_SIGNATURE_VERSIONS = (7, CURRENT_TRAINING_SIGNATURE_VERSION)


@dataclass(frozen=True)
class AdapterCheckpointBundle:
    """Validated deployment state extracted from a SAC finetune checkpoint."""

    checkpoint_path: Path
    checkpoint_sha256: str
    hipad_trainable: Dict[str, torch.Tensor]
    feature_adapter_state: Dict[str, torch.Tensor]
    adapter_mode: str
    feature_adapter_levels: Tuple[int, ...]
    feature_adapter_feature_dim: int
    feature_adapter_ego_state_dim: int
    feature_adapter_ego_hidden_dim: int
    feature_adapter_bottleneck_reduction: int
    feature_adapter_dcn_group: int
    feature_adapter_residual_scale: float
    feature_adapter_zero_init: bool
    feature_adapter_norm_type: str
    feature_adapter_norm_groups: int
    training_signature_version: int
    replay_schema_version: int
    control_semantics: str
    base_checkpoint_sha256: str
    adapter_prediction_present: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping, got {type(value).__name__}")
    return value


def _tensor_state(value, label: str) -> Dict[str, torch.Tensor]:
    mapping = _require_mapping(value, label)
    state: Dict[str, torch.Tensor] = {}
    for name, tensor in mapping.items():
        if not isinstance(name, str):
            raise RuntimeError(f"{label} contains a non-string key: {name!r}")
        if not torch.is_tensor(tensor):
            raise RuntimeError(f"{label}[{name!r}] is not a tensor")
        state[name] = tensor.detach().cpu()
    if not state:
        raise RuntimeError(f"{label} is empty")
    return state


def _as_int(config: Mapping, key: str, default: Optional[int] = None) -> int:
    value = config.get(key, default)
    if value is None:
        raise RuntimeError(f"finetune_config is missing {key}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"finetune_config.{key} is not an integer: {value!r}") from exc


def _validate_state_against_module(
    state: Mapping[str, torch.Tensor],
    module: torch.nn.Module,
    label: str,
    *,
    exact_keys: bool = True,
) -> None:
    module_state = module.state_dict()
    expected = set(module_state)
    received = set(state)
    missing = sorted(expected - received)
    unexpected = sorted(received - expected)
    shape_mismatch = sorted(
        name
        for name in expected & received
        if tuple(module_state[name].shape) != tuple(state[name].shape)
    )
    if exact_keys and (missing or unexpected or shape_mismatch):
        raise RuntimeError(
            f"{label} state mismatch: missing={missing}, unexpected={unexpected}, "
            f"shape_mismatch={shape_mismatch}"
        )
    if shape_mismatch:
        raise RuntimeError(f"{label} shape mismatch: {shape_mismatch}")


def restore_hipad_trainable(
    model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    *,
    expected_names=None,
) -> int:
    """Strictly copy the planning branch state into a clean model."""

    named_parameters = dict(model.named_parameters())
    if expected_names is not None:
        expected = set(expected_names)
        received = set(state)
        missing_expected = sorted(expected - received)
        unexpected = sorted(received - expected)
        if missing_expected or unexpected:
            raise RuntimeError(
                "HiP-AD trainable key mismatch: "
                f"missing={missing_expected}, unexpected={unexpected}"
            )
    missing = sorted(name for name in state if name not in named_parameters)
    shape_mismatch = sorted(
        name
        for name, tensor in state.items()
        if name in named_parameters and tuple(named_parameters[name].shape) != tuple(tensor.shape)
    )
    if missing or shape_mismatch:
        raise RuntimeError(
            "HiP-AD trainable state mismatch: "
            f"missing={missing}, shape_mismatch={shape_mismatch}"
        )
    for name, tensor in state.items():
        parameter = named_parameters[name]
        parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
    return len(state)


def build_feature_adapter_from_bundle(bundle: AdapterCheckpointBundle, device: torch.device):
    """Construct the exact four-level adapter architecture used during SAC."""

    # The leaderboard process already has the clean project on sys.path.  Keep
    # this import local so importing the checkpoint gate itself remains cheap.
    try:
        from rl.ego_state_adapter import build_ego_state_dcnv4_feature_adapter
    except ImportError:
        from ego_state_adapter import build_ego_state_dcnv4_feature_adapter

    adapter = build_ego_state_dcnv4_feature_adapter(
        feature_dim=bundle.feature_adapter_feature_dim,
        ego_state_dim=bundle.feature_adapter_ego_state_dim,
        levels=bundle.feature_adapter_levels,
        ego_hidden_dim=(bundle.feature_adapter_ego_hidden_dim or None),
        bottleneck_reduction=bundle.feature_adapter_bottleneck_reduction,
        dcn_group=(bundle.feature_adapter_dcn_group or None),
        zero_init_residual=bundle.feature_adapter_zero_init,
        residual_scale=bundle.feature_adapter_residual_scale,
        norm_type=bundle.feature_adapter_norm_type,
        norm_groups=bundle.feature_adapter_norm_groups,
    ).to(device)
    _validate_state_against_module(
        bundle.feature_adapter_state,
        adapter,
        "feature_dcnv4_adapter",
    )
    try:
        adapter.load_state_dict(bundle.feature_adapter_state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"feature_dcnv4_adapter strict restore failed: {exc}") from exc
    adapter.eval()
    return adapter


def load_adapter_checkpoint(
    checkpoint_path: str,
    *,
    base_checkpoint_path: Optional[str] = None,
    expected_project_root: Optional[str] = None,
) -> AdapterCheckpointBundle:
    """Load and validate the deployment state from a SAC checkpoint.

    The gate is intentionally strict.  A leaderboard run must never silently
    continue with the adapter disabled after a partial restore.  Legacy v7 and
    current v8 checkpoints may record the source workspace in
    ``hipad_project_root``.  That absolute path is historical metadata only:
    relocation compatibility is proven by the supplied base-checkpoint content
    hash and the exact 25/132 deployment tensor contract, not by path equality.
    """

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"finetune checkpoint not found: {path}")
    checkpoint_sha256 = _sha256(path)
    checkpoint = torch.load(str(path), map_location="cpu")
    root = _require_mapping(checkpoint, "finetune checkpoint")

    signature = int(root.get("training_signature_version", 0))
    if signature not in SUPPORTED_DEPLOYMENT_SIGNATURE_VERSIONS:
        raise RuntimeError(
            "training_signature_version mismatch: "
            f"found {signature}, expected one of {SUPPORTED_DEPLOYMENT_SIGNATURE_VERSIONS}"
        )
    expected_checkpoint_version = 1 if signature == 7 else 2
    if int(root.get("checkpoint_version", 0)) != expected_checkpoint_version:
        raise RuntimeError(
            "checkpoint_version mismatch: "
            f"found {root.get('checkpoint_version')!r}, expected {expected_checkpoint_version}"
        )
    training_signature = str(root.get("training_signature", ""))
    if len(training_signature) != 64 or any(character not in "0123456789abcdef" for character in training_signature):
        raise RuntimeError("training_signature must be a 64-character lowercase SHA-256")

    agent = _require_mapping(root.get("agent"), "checkpoint.agent")
    saved_config = _require_mapping(root.get("finetune_config"), "checkpoint.finetune_config")

    replay_schema = _as_int(saved_config, "replay_schema_version", 0)
    if replay_schema != 5:
        raise RuntimeError(f"replay_schema_version mismatch: found {replay_schema}, expected 5")
    control_semantics = str(saved_config.get("control_semantics", ""))
    if control_semantics != "hipad_clean_dual_pid_v2_mode_aligned":
        raise RuntimeError(
            "control_semantics mismatch: "
            f"found {control_semantics!r}, expected 'hipad_clean_dual_pid_v2_mode_aligned'"
        )

    adapter_mode = str(saved_config.get("adapter_mode", ""))
    agent_adapter_mode = str(agent.get("adapter_mode", ""))
    if adapter_mode != EXPECTED_ADAPTER_MODE or agent_adapter_mode != EXPECTED_ADAPTER_MODE:
        raise RuntimeError(
            "adapter_mode mismatch: "
            f"config={adapter_mode!r}, agent={agent_adapter_mode!r}, expected={EXPECTED_ADAPTER_MODE!r}"
        )

    levels = tuple(int(level) for level in saved_config.get("feature_adapter_levels", ()))
    agent_levels = tuple(int(level) for level in agent.get("feature_adapter_levels", ()))
    if levels != FOUR_LEVELS or agent_levels != FOUR_LEVELS:
        raise RuntimeError(
            "feature_adapter_levels mismatch: "
            f"config={levels}, agent={agent_levels}, expected={FOUR_LEVELS}"
        )

    feature_dim = _as_int(saved_config, "feature_adapter_feature_dim", EXPECTED_FEATURE_DIM)
    ego_state_dim = _as_int(saved_config, "feature_adapter_ego_state_dim", EXPECTED_EGO_STATE_DIM)
    ego_hidden_dim = _as_int(saved_config, "feature_adapter_ego_hidden_dim", 0)
    bottleneck_reduction = _as_int(saved_config, "feature_adapter_bottleneck_reduction", 4)
    dcn_group = _as_int(saved_config, "feature_adapter_dcn_group", 0)
    residual_scale = float(saved_config.get("feature_adapter_residual_scale", 1.0))
    zero_init = bool(saved_config.get("feature_adapter_zero_init", True))
    norm_type = str(saved_config.get("feature_adapter_norm_type", "group"))
    norm_groups = _as_int(saved_config, "feature_adapter_norm_groups", 8)
    if feature_dim != EXPECTED_FEATURE_DIM:
        raise RuntimeError(f"feature_adapter_feature_dim mismatch: found {feature_dim}, expected 256")
    if ego_state_dim != EXPECTED_EGO_STATE_DIM:
        raise RuntimeError(f"feature_adapter_ego_state_dim mismatch: found {ego_state_dim}, expected 21")

    # ``saved_config.hipad_project_root`` is intentionally not compared with
    # the runtime root. Checkpoint locations are deployment metadata, while the
    # immutable base hash and exact tensor contract prove compatibility.
    if expected_project_root is not None:
        runtime_root = Path(expected_project_root).expanduser().resolve()
        plugin_init = runtime_root / "projects" / "mmdet3d_plugin" / "__init__.py"
        if not runtime_root.is_dir() or not plugin_init.is_file():
            raise RuntimeError(
                "expected_project_root is not a HiP-AD project tree: "
                f"{runtime_root}"
            )

    provenance = _require_mapping(root.get("runtime_provenance"), "checkpoint.runtime_provenance")
    saved_base_hash = str(provenance.get("checkpoint.sha256", ""))
    if not saved_base_hash:
        raise RuntimeError("checkpoint.runtime_provenance.checkpoint.sha256 is missing")
    if not base_checkpoint_path:
        raise RuntimeError(
            "base_checkpoint_path is required to verify checkpoint relocation by content hash"
        )
    base_path = Path(base_checkpoint_path).expanduser().resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"base checkpoint not found: {base_path}")
    actual_base_hash = _sha256(base_path)
    if actual_base_hash != saved_base_hash:
        raise RuntimeError(
            "base checkpoint hash mismatch: "
            f"finetune={saved_base_hash}, evaluation={actual_base_hash}"
        )

    hipad_trainable = _tensor_state(agent.get("hipad_trainable"), "checkpoint.agent.hipad_trainable")
    feature_adapter_state = _tensor_state(
        agent.get("feature_dcnv4_adapter"),
        "checkpoint.agent.feature_dcnv4_adapter",
    )
    if len(hipad_trainable) != 25:
        raise RuntimeError(f"hipad_trainable tensor count mismatch: found {len(hipad_trainable)}, expected 25")
    if len(feature_adapter_state) != 132:
        raise RuntimeError(
            "feature_dcnv4_adapter tensor count mismatch: "
            f"found {len(feature_adapter_state)}, expected 132"
        )

    return AdapterCheckpointBundle(
        checkpoint_path=path,
        checkpoint_sha256=checkpoint_sha256,
        hipad_trainable=hipad_trainable,
        feature_adapter_state=feature_adapter_state,
        adapter_mode=adapter_mode,
        feature_adapter_levels=levels,
        feature_adapter_feature_dim=feature_dim,
        feature_adapter_ego_state_dim=ego_state_dim,
        feature_adapter_ego_hidden_dim=ego_hidden_dim,
        feature_adapter_bottleneck_reduction=bottleneck_reduction,
        feature_adapter_dcn_group=dcn_group,
        feature_adapter_residual_scale=residual_scale,
        feature_adapter_zero_init=zero_init,
        feature_adapter_norm_type=norm_type,
        feature_adapter_norm_groups=norm_groups,
        training_signature_version=signature,
        replay_schema_version=replay_schema,
        control_semantics=control_semantics,
        base_checkpoint_sha256=saved_base_hash,
        adapter_prediction_present=(
            isinstance(agent.get("adapter_prediction"), Mapping)
            and bool(agent.get("adapter_prediction"))
        ),
    )
