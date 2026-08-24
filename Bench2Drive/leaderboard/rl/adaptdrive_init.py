"""Strict legacy-parent initialization for fresh AdaptDrive v8 runs.

The registered signature-v7 checkpoint is a weight source, never a resume
checkpoint. Only audited tensors are imported; counters, optimizers, replay
metadata, legacy auxiliary state, and machine-specific paths are discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

import torch

from rl.adaptdrive_training_signature import TRAINING_SIGNATURE_VERSION, hash_file


TRAINING_INIT_PROFILE = "legacy_v7_step140906_adaptdrive_weights_v1"
REGISTERED_PARENT_SHA256 = "481e0c1b7217351f24e5584bbb5b2ef5b2bfeeb66e45272b6e818d1b216a8fc2"
REGISTERED_BASE_SHA256 = "7711b693293533463732d8a3efa8d5148d203344aad727a4661cb84263613956"
REGISTERED_PARENT_STEP = 140906
REGISTERED_PARENT_EPISODE = 379
REGISTERED_PARENT_SIGNATURE_VERSION = 7
REGISTERED_PARENT_TRAINING_SIGNATURE = "9b6e71a3c2f84cf9d4ba5d7ceb854a690e79a725463e4df672012c44b8960331"
REGISTERED_REPLAY_SCHEMA_VERSION = 5
REGISTERED_ROUTE_SHA256 = "ffff24b3a2b8584632730fb42de7cb6036af780c3952cfef08ee8cf96a712f88"

REGISTERED_ANCHOR_SHA256 = {
    "data/kmeans/b2d_det_900.npy":
        "82eb36c4a00932829b24a53883a40695e0b7729b56b62094c54f52dfdbada748",
    "data/kmeans/b2d_map_100.npy":
        "a16ffdbfd6ecf9e73e10a775c8a0e11db4a628527a26c51fbfe23e27ac0d10e7",
    "data/kmeans/b2d_motion_6.npy":
        "12e049d5fcec6d1257e3b0635443329f590a8647ce82a692051fb9e84c0f298a",
    "data/kmeans/b2d_plan_spat_6x8_2m.npy":
        "a6bf63bfb5ea9d2afecc0a94db3d0efac325ceb155b3a10637a98d15d65b640c",
    "data/kmeans/b2d_plan_spat_6x8_5m.npy":
        "94ae8c6335698dadf3a79b52dba38cbad71fd61f9ab5665c9ce2477b6283de13",
}

EXPECTED_CONTROL = "hipad_clean_dual_pid_v2_mode_aligned"
EXPECTED_ADAPTER_MODE = "dcnv4_feature"
EXPECTED_ADAPTER_LEVELS = (0, 1, 2, 3)
EXPECTED_ADAPTER_UPDATE = "prediction_only"

IMPORTED_AGENT_COMPONENTS: Tuple[str, ...] = (
    "hipad_trainable",
    "feature_dcnv4_adapter",
    "adapter_prediction",
    "critic",
    "vf",
    "vf_target",
    "log_alpha",
)

EXCLUDED_PARENT_STATE: Tuple[str, ...] = (
    "policy_optimizer",
    "critic_optimizer",
    "vf_optimizer",
    "alpha_optimizer",
    "adapter_prediction.optimizer",
    "feature_adapter_aux",
    "trainer_state",
    "step",
    "episode",
    "replay_state",
    "replay_size",
    "runtime_provenance.paths",
    "training_signature_v7",
)


@dataclass(frozen=True)
class LegacyParentInitialization:
    parent_checkpoint_sha256: str
    parent_training_signature: str
    parent_step: int
    parent_episode: int
    parent_replay_size: int
    base_checkpoint_sha256: str
    target_training_signature: str
    imported_agent: Dict[str, object]

    def provenance(self) -> Dict[str, object]:
        return {
            "profile": TRAINING_INIT_PROFILE,
            "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
            "parent_training_signature_version": REGISTERED_PARENT_SIGNATURE_VERSION,
            "parent_training_signature": self.parent_training_signature,
            "parent_step": int(self.parent_step),
            "parent_episode": int(self.parent_episode),
            "parent_replay_size_metadata_only": int(self.parent_replay_size),
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "target_training_signature_version": TRAINING_SIGNATURE_VERSION,
            "target_training_signature": self.target_training_signature,
            "imported_components": list(IMPORTED_AGENT_COMPONENTS),
            "excluded_parent_state": list(EXCLUDED_PARENT_STATE),
            "new_step": 0,
            "new_episode": 0,
            "fresh_optimizers": True,
            "fresh_replay": True,
        }


def _require_hash(path: Path, label: str) -> str:
    digest = hash_file(path)
    if not digest:
        raise FileNotFoundError(f"{label} is missing: {path}")
    return digest


def _require_mapping(value, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping, got {type(value).__name__}")
    return value


def _cpu_tensor(value, label: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise RuntimeError(f"{label} must be a tensor")
    return value.detach().cpu()


def _cpu_tensor_mapping(value, label: str) -> Dict[str, torch.Tensor]:
    source = _require_mapping(value, label)
    result: Dict[str, torch.Tensor] = {}
    for name, tensor in source.items():
        if not isinstance(name, str):
            raise RuntimeError(f"{label} contains a non-string key")
        result[name] = _cpu_tensor(tensor, f"{label}.{name}")
    if not result:
        raise RuntimeError(f"{label} is empty")
    return result


def _strict_prediction_state(value) -> Dict[str, object]:
    state = _require_mapping(value, "checkpoint.agent.adapter_prediction")
    reward_head = _cpu_tensor_mapping(state.get("reward_head"), "adapter_prediction.reward_head")
    semantic_head = _cpu_tensor_mapping(state.get("semantic_head"), "adapter_prediction.semantic_head")
    reward_target_names = tuple(str(name) for name in state.get("reward_target_names", ()))
    semantic_channel_names = tuple(str(name) for name in state.get("semantic_channel_names", ()))
    semantic_weights = _cpu_tensor(
        state.get("semantic_channel_loss_weights"),
        "adapter_prediction.semantic_channel_loss_weights",
    )
    if len(reward_head) != 14:
        raise RuntimeError(f"reward prediction head tensor count mismatch: {len(reward_head)} != 14")
    if len(semantic_head) != 20:
        raise RuntimeError(f"semantic prediction head tensor count mismatch: {len(semantic_head)} != 20")
    if not reward_target_names or not semantic_channel_names:
        raise RuntimeError("adapter prediction target/channel names are missing")
    return {
        "adapter_prediction_enabled": True,
        "reward_head": reward_head,
        "semantic_head": semantic_head,
        "optimizer": None,
        "reward_target_names": reward_target_names,
        "semantic_channel_names": semantic_channel_names,
        "semantic_channel_loss_weights": semantic_weights,
    }


def import_registered_legacy_parent(
    checkpoint_path: str,
    *,
    base_checkpoint_path: str,
    route_path: str,
    hipad_root: str,
    target_training_signature: str,
) -> LegacyParentInitialization:
    """Extract the one audited v7 parent into a fresh-run weight bundle."""

    try:
        signature_is_hex = len(str(target_training_signature)) == 64 and int(str(target_training_signature), 16) >= 0
    except ValueError:
        signature_is_hex = False
    if not signature_is_hex:
        raise RuntimeError("target AdaptDrive v8 training signature is invalid")
    parent_path = Path(checkpoint_path).expanduser().resolve()
    base_path = Path(base_checkpoint_path).expanduser().resolve()
    route = Path(route_path).expanduser().resolve()
    clean_root = Path(hipad_root).expanduser().resolve()

    parent_hash = _require_hash(parent_path, "registered parent checkpoint")
    if parent_hash != REGISTERED_PARENT_SHA256:
        raise RuntimeError(
            f"unregistered legacy parent SHA-256: found {parent_hash}, expected {REGISTERED_PARENT_SHA256}"
        )
    base_hash = _require_hash(base_path, "HiP-AD base checkpoint")
    if base_hash != REGISTERED_BASE_SHA256:
        raise RuntimeError(f"base checkpoint SHA-256 mismatch: found {base_hash}, expected {REGISTERED_BASE_SHA256}")
    route_hash = _require_hash(route, "Bench2Drive route file")
    if route_hash != REGISTERED_ROUTE_SHA256:
        raise RuntimeError(f"registered parent route SHA-256 mismatch: {route_hash}")
    for relative_path, expected_hash in REGISTERED_ANCHOR_SHA256.items():
        actual_hash = _require_hash(clean_root / relative_path, f"HiP-AD anchor {relative_path}")
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"HiP-AD anchor SHA-256 mismatch for {relative_path}: "
                f"found {actual_hash}, expected {expected_hash}"
            )

    checkpoint = _require_mapping(torch.load(str(parent_path), map_location="cpu"), "legacy checkpoint")
    expected_top_level = {
        "training_signature_version": REGISTERED_PARENT_SIGNATURE_VERSION,
        "training_signature": REGISTERED_PARENT_TRAINING_SIGNATURE,
        "step": REGISTERED_PARENT_STEP,
        "episode": REGISTERED_PARENT_EPISODE,
    }
    mismatched = {
        key: (checkpoint.get(key), expected)
        for key, expected in expected_top_level.items()
        if checkpoint.get(key) != expected
    }
    if mismatched:
        raise RuntimeError(f"registered parent metadata mismatch: {mismatched}")

    config = _require_mapping(checkpoint.get("finetune_config"), "checkpoint.finetune_config")
    required_config = {
        "control_semantics": EXPECTED_CONTROL,
        "replay_schema_version": REGISTERED_REPLAY_SCHEMA_VERSION,
        "adapter_mode": EXPECTED_ADAPTER_MODE,
        "feature_adapter_levels": EXPECTED_ADAPTER_LEVELS,
        "adapter_prediction_enabled": True,
        "adapter_prediction_train_reward": True,
        "adapter_prediction_train_semantic": True,
        "adapter_prediction_update_mode": EXPECTED_ADAPTER_UPDATE,
        "feature_adapter_train_actor_loss": False,
    }
    config_mismatch = {}
    for key, expected in required_config.items():
        found = config.get(key)
        if key == "feature_adapter_levels" and found is not None:
            found = tuple(int(level) for level in found)
        if found != expected:
            config_mismatch[key] = (found, expected)
    if config_mismatch:
        raise RuntimeError(f"registered parent training contract mismatch: {config_mismatch}")

    provenance = _require_mapping(checkpoint.get("runtime_provenance"), "checkpoint.runtime_provenance")
    if str(provenance.get("checkpoint.sha256", "")) != base_hash:
        raise RuntimeError("registered parent provenance has the wrong base checkpoint SHA-256")

    replay = _require_mapping(checkpoint.get("replay_state"), "checkpoint.replay_state")
    if int(replay.get("schema_version", 0)) != REGISTERED_REPLAY_SCHEMA_VERSION:
        raise RuntimeError("registered parent replay schema mismatch")
    if str(replay.get("control_semantics", "")) != EXPECTED_CONTROL:
        raise RuntimeError("registered parent replay control semantics mismatch")
    replay_size = int(replay.get("size", -1))
    if replay_size < 0 or replay_size != int(checkpoint.get("replay_size", -2)):
        raise RuntimeError("registered parent replay metadata is inconsistent")

    agent = _require_mapping(checkpoint.get("agent"), "checkpoint.agent")
    hipad_trainable = _cpu_tensor_mapping(agent.get("hipad_trainable"), "checkpoint.agent.hipad_trainable")
    saved_trainable_names = tuple(str(name) for name in agent.get("hipad_trainable_names", ()))
    if set(saved_trainable_names) != set(hipad_trainable):
        raise RuntimeError("registered parent HiP-AD trainable names do not match its tensor state")
    feature_adapter = _cpu_tensor_mapping(
        agent.get("feature_dcnv4_adapter"), "checkpoint.agent.feature_dcnv4_adapter"
    )
    critic = _cpu_tensor_mapping(agent.get("critic"), "checkpoint.agent.critic")
    vf = _cpu_tensor_mapping(agent.get("vf"), "checkpoint.agent.vf")
    vf_target = _cpu_tensor_mapping(agent.get("vf_target"), "checkpoint.agent.vf_target")
    expected_counts = {
        "hipad_trainable": (len(hipad_trainable), 25),
        "feature_dcnv4_adapter": (len(feature_adapter), 132),
        "critic": (len(critic), 20),
        "vf": (len(vf), 10),
        "vf_target": (len(vf_target), 10),
    }
    wrong_counts = {name: values for name, values in expected_counts.items() if values[0] != values[1]}
    if wrong_counts:
        raise RuntimeError(f"registered parent tensor-count mismatch: {wrong_counts}")

    imported_agent: Dict[str, object] = {
        "adapter_mode": EXPECTED_ADAPTER_MODE,
        "hipad_trainable": hipad_trainable,
        "feature_dcnv4_adapter": feature_adapter,
        "adapter_prediction": _strict_prediction_state(agent.get("adapter_prediction")),
        "critic": critic,
        "vf": vf,
        "vf_target": vf_target,
        "log_alpha": _cpu_tensor(agent.get("log_alpha"), "checkpoint.agent.log_alpha"),
    }
    return LegacyParentInitialization(
        parent_checkpoint_sha256=parent_hash,
        parent_training_signature=REGISTERED_PARENT_TRAINING_SIGNATURE,
        parent_step=REGISTERED_PARENT_STEP,
        parent_episode=REGISTERED_PARENT_EPISODE,
        parent_replay_size=replay_size,
        base_checkpoint_sha256=base_hash,
        target_training_signature=str(target_training_signature),
        imported_agent=imported_agent,
    )


def _validate_named_tensor_state(
    state: Mapping[str, object],
    expected_state: Mapping[str, torch.Tensor],
    label: str,
) -> None:
    received = set(state)
    expected = set(expected_state)
    missing = sorted(expected - received)
    unexpected = sorted(received - expected)
    shape_mismatch = sorted(
        name
        for name in expected & received
        if not torch.is_tensor(state[name])
        or tuple(state[name].shape) != tuple(expected_state[name].shape)
    )
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            f"{label} state mismatch: missing={missing}, unexpected={unexpected}, "
            f"shape_mismatch={shape_mismatch}"
        )


def _strict_load_module(module, state, label: str) -> None:
    source = _require_mapping(state, label)
    current = module.state_dict()
    _validate_named_tensor_state(source, current, label)
    module.load_state_dict(source, strict=True)


def _optimizer_state_owners(agent) -> Tuple[str, ...]:
    names = (
        "policy_optimizer",
        "critic_optimizer",
        "vf_optimizer",
        "alpha_optimizer",
        "_adapter_prediction_optimizer",
    )
    return tuple(
        name
        for name in names
        if getattr(agent, name, None) is not None and getattr(agent, name).state
    )


def apply_registered_legacy_parent(
    agent,
    initialization: LegacyParentInitialization,
    *,
    current_training_signature: str,
) -> None:
    """Apply audited weights while preserving fresh optimizer state."""

    if initialization.parent_checkpoint_sha256 != REGISTERED_PARENT_SHA256:
        raise RuntimeError("legacy initialization parent SHA-256 is not registered")
    if initialization.base_checkpoint_sha256 != REGISTERED_BASE_SHA256:
        raise RuntimeError("legacy initialization base SHA-256 is not registered")
    if initialization.target_training_signature != str(current_training_signature):
        raise RuntimeError("legacy initialization target v8 signature mismatch")

    config = agent.config
    target_contract = {
        "adapter_mode": str(getattr(agent, "adapter_mode", "")),
        "feature_adapter_levels": tuple(int(level) for level in config.feature_adapter_levels),
        "adapter_prediction_enabled": bool(config.adapter_prediction_enabled),
        "adapter_prediction_train_reward": bool(config.adapter_prediction_train_reward),
        "adapter_prediction_train_semantic": bool(config.adapter_prediction_train_semantic),
        "adapter_prediction_update_mode": str(config.adapter_prediction_update_mode),
    }
    expected_contract = {
        "adapter_mode": EXPECTED_ADAPTER_MODE,
        "feature_adapter_levels": EXPECTED_ADAPTER_LEVELS,
        "adapter_prediction_enabled": True,
        "adapter_prediction_train_reward": True,
        "adapter_prediction_train_semantic": True,
        "adapter_prediction_update_mode": EXPECTED_ADAPTER_UPDATE,
    }
    mismatched = {
        key: (target_contract.get(key), expected)
        for key, expected in expected_contract.items()
        if target_contract.get(key) != expected
    }
    if mismatched:
        raise RuntimeError(f"target AdaptDrive initialization contract mismatch: {mismatched}")
    dirty_optimizers = _optimizer_state_owners(agent)
    if dirty_optimizers:
        raise RuntimeError(f"legacy initialization requires fresh optimizers: {dirty_optimizers}")

    imported = initialization.imported_agent
    named_parameters = dict(agent._model.named_parameters())
    trainable_names = tuple(agent._trainable_param_names)
    planning_state = _require_mapping(imported.get("hipad_trainable"), "initialization hipad_trainable")
    expected_planning = {name: named_parameters[name] for name in trainable_names}
    _validate_named_tensor_state(planning_state, expected_planning, "initialization HiP-AD planning")
    for name, value in planning_state.items():
        parameter = named_parameters[name]
        parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))

    feature_adapter = getattr(agent, "_feature_dcnv4_adapter", None)
    if feature_adapter is None:
        raise RuntimeError("target agent has no feature DCNv4 adapter")
    _strict_load_module(feature_adapter, imported.get("feature_dcnv4_adapter"), "initialization feature adapter")
    _strict_load_module(agent.critic, imported.get("critic"), "initialization critic")
    _strict_load_module(agent.vf, imported.get("vf"), "initialization value function")
    _strict_load_module(agent.vf_target, imported.get("vf_target"), "initialization target value function")

    prediction = _require_mapping(imported.get("adapter_prediction"), "initialization adapter prediction")
    if prediction.get("optimizer") is not None:
        raise RuntimeError("legacy initialization must not contain adapter prediction optimizer state")
    reward_head = getattr(agent, "_adapter_prediction_reward_head", None)
    semantic_head = getattr(agent, "_adapter_prediction_semantic_head", None)
    if reward_head is None or semantic_head is None:
        raise RuntimeError("target agent must build both AdaptDrive prediction heads")
    _strict_load_module(reward_head, prediction.get("reward_head"), "initialization reward head")
    _strict_load_module(semantic_head, prediction.get("semantic_head"), "initialization semantic head")

    from rl.adapter_prediction_heads import DEFAULT_REWARD_TARGET_SPECS

    current_reward_names = tuple(spec.name for spec in DEFAULT_REWARD_TARGET_SPECS)
    saved_reward_names = tuple(str(name) for name in prediction.get("reward_target_names", ()))
    if current_reward_names != saved_reward_names:
        raise RuntimeError("legacy initialization reward target names mismatch")
    current_semantic_names = tuple(str(name) for name in semantic_head.channel_names)
    saved_semantic_names = tuple(str(name) for name in prediction.get("semantic_channel_names", ()))
    if current_semantic_names != saved_semantic_names:
        raise RuntimeError("legacy initialization semantic channel names mismatch")
    current_weights = agent._adapter_prediction_semantic_loss_fn.channel_loss_weights.detach().cpu()
    saved_weights = _cpu_tensor(
        prediction.get("semantic_channel_loss_weights"),
        "initialization semantic channel weights",
    )
    if tuple(current_weights.shape) != tuple(saved_weights.shape) or not torch.equal(current_weights, saved_weights):
        raise RuntimeError("legacy initialization semantic channel weights mismatch")

    log_alpha = _cpu_tensor(imported.get("log_alpha"), "initialization log_alpha")
    if tuple(log_alpha.shape) != tuple(agent.log_alpha.shape):
        raise RuntimeError("legacy initialization log_alpha shape mismatch")
    agent.log_alpha.data.copy_(log_alpha.to(device=agent.log_alpha.device, dtype=agent.log_alpha.dtype))
    if _optimizer_state_owners(agent):
        raise RuntimeError("legacy initialization unexpectedly populated optimizer state")
    agent._set_eval_mode()
