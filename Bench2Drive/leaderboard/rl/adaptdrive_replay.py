"""Strict mmap replay pairing for AdaptDrive full resume.

The legacy replay API can reconstruct ``ptr``/``size`` from checkpoint metadata
even when the mmap payload was not copied.  This module removes that fallback:
full resume requires one immutable checkpoint reference, one replay manifest,
one immutable state snapshot and every declared ``.dat`` payload file.

Each replay lives in a UUID-named directory.  The UUID, experiment ID and v8
training signature must agree in the checkpoint reference, manifest and state
snapshot.  Exact byte sizes are checked for every mmap file.  A bounded
sampled-slot SHA-256 witness detects common truncation, all-zero replacement,
and accidental payload mixing without hashing a potentially 200 GB replay at
every checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from rl.adaptdrive_training_signature import TRAINING_SIGNATURE_VERSION, canonical_json_bytes
from rl.replay import CLEAN_DUAL_TRAJECTORY_CONTROL, CLEAN_DUAL_TRAJECTORY_SCHEMA_VERSION, FeatureReplayBuffer


REPLAY_PROTOCOL_VERSION = 1
REPLAY_MANIFEST_VERSION = 1
PAYLOAD_WITNESS_ALGORITHM = "per_slot_payload_sha256_v1"
REPLAY_MANIFEST_FILENAME = "replay_manifest.json"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class ReplayContext:
    replay_root: Path
    payload_dir: Path
    replay_uuid: str
    experiment_id: str
    training_signature: str
    manifest: Dict[str, Any]
    manifest_sha256: str


def validate_experiment_id(experiment_id: str) -> str:
    value = str(experiment_id or "").strip()
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(
            "experiment_id must be 1-128 characters using only letters, digits, '.', '_' or '-'"
        )
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = canonical_json_bytes(_jsonable(value)) + b"\n"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return _sha256_bytes(payload)


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} must be a non-symlink regular file: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to load {label}: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must contain a JSON object")
    return value


def _safe_child(root: Path, filename: str, *, label: str) -> Path:
    name = str(filename or "")
    if not name or Path(name).name != name or name in {".", ".."}:
        raise RuntimeError(f"unsafe {label} filename: {name!r}")
    raw_path = root / name
    if raw_path.is_symlink():
        raise RuntimeError(f"{label} must not be a symlink: {raw_path}")
    path = raw_path.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes replay directory: {name!r}") from exc
    return path


def _payload_data_arrays(replay: FeatureReplayBuffer) -> Dict[str, np.memmap]:
    names = (
        "states",
        "target_points",
        "commands",
        "actor_base_features",
        "critic_bev_features",
        "prev_pid_summaries",
        "prev_pid_summary_masks",
        "trajectories",
        "longitudinal_trajectories",
        "pid_summaries",
        "selected_lateral_modes",
        "longitudinal_modes",
        "rewards",
        "next_states",
        "next_target_points",
        "next_commands",
        "next_critic_bev_features",
        "dones",
        "plan_cls_context",
        "all_candidates",
        "candidate_longitudinal_trajectories",
        "reference_logits",
    )
    result: Dict[str, np.memmap] = {}
    for name in names:
        value = getattr(replay, name, None)
        if value is not None:
            if not isinstance(value, np.memmap):
                raise RuntimeError(f"feature replay array {name} is not a numpy memmap")
            result[name] = value
    return result


class StrictFeatureReplayBuffer(FeatureReplayBuffer):
    """Feature replay with an incremental content hash for every slot.

    Updating one transition hashes only that transition.  Checkpoint saves hash
    the small slot-hash table, while strict resume recomputes every valid slot
    once and compares it with the table.  This avoids rehashing a multi-GB
    payload at every checkpoint without trusting metadata alone.
    """

    SLOT_HASH_BYTES = hashlib.sha256().digest_size

    @staticmethod
    def estimate_storage_bytes(
        capacity: int,
        state_shape: tuple,
        actor_base_shape: tuple,
        critic_bev_shape: tuple,
        trajectory_shape: tuple,
        pid_summary_dim: int = FeatureReplayBuffer.PID_SUMMARY_DIM,
        control_semantics: str = CLEAN_DUAL_TRAJECTORY_CONTROL,
    ) -> int:
        payload_bytes = FeatureReplayBuffer.estimate_storage_bytes(
            capacity,
            state_shape,
            actor_base_shape,
            critic_bev_shape,
            trajectory_shape,
            pid_summary_dim=pid_summary_dim,
            control_semantics=control_semantics,
        )
        return int(payload_bytes + int(capacity) * StrictFeatureReplayBuffer.SLOT_HASH_BYTES)

    @staticmethod
    def capacity_for_storage_budget(
        max_storage_bytes: int,
        state_shape: tuple,
        actor_base_shape: tuple,
        critic_bev_shape: tuple,
        trajectory_shape: tuple,
        pid_summary_dim: int = FeatureReplayBuffer.PID_SUMMARY_DIM,
        control_semantics: str = CLEAN_DUAL_TRAJECTORY_CONTROL,
    ) -> int:
        if int(max_storage_bytes) <= 0:
            return 1
        per_transition = StrictFeatureReplayBuffer.estimate_storage_bytes(
            1,
            state_shape,
            actor_base_shape,
            critic_bev_shape,
            trajectory_shape,
            pid_summary_dim=pid_summary_dim,
            control_semantics=control_semantics,
        )
        return max(1, int(int(max_storage_bytes) // per_transition))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slot_hashes = self._create_memmap(
            "slot_hashes.dat",
            (self.capacity, self.SLOT_HASH_BYTES),
            np.uint8,
        )

    def _flush(self):
        super()._flush()
        slot_hashes = getattr(self, "slot_hashes", None)
        if slot_hashes is not None:
            slot_hashes.flush()

    def _slot_digest(self, index: int) -> bytes:
        digest = hashlib.sha256()
        digest.update(
            canonical_json_bytes(
                {
                    "slot": int(index),
                    "arrays": [
                        {
                            "logical_name": logical_name,
                            "dtype": np.dtype(array.dtype).str,
                            "item_shape": list(array.shape[1:]),
                        }
                        for logical_name, array in sorted(_payload_data_arrays(self).items())
                    ],
                }
            )
        )
        for logical_name, array in sorted(_payload_data_arrays(self).items()):
            digest.update(logical_name.encode("utf-8"))
            digest.update(np.ascontiguousarray(array[index]).tobytes(order="C"))
        return digest.digest()

    def _update_slot_hash(self, index: int) -> None:
        self.slot_hashes[int(index)] = np.frombuffer(self._slot_digest(index), dtype=np.uint8)

    def add(self, *args, **kwargs):
        write_index = int(self.ptr)
        super().add(*args, **kwargs)
        self._update_slot_hash(write_index)

    def valid_slot_hash_table_sha256(self, size: Optional[int] = None) -> str:
        valid_size = int(self.size if size is None else size)
        if valid_size < 0 or valid_size > self.capacity:
            raise RuntimeError(f"invalid replay size for slot hash table: {valid_size}")
        digest = hashlib.sha256()
        digest.update(canonical_json_bytes({"algorithm": PAYLOAD_WITNESS_ALGORITHM, "size": valid_size}))
        if valid_size:
            digest.update(np.ascontiguousarray(self.slot_hashes[:valid_size]).tobytes(order="C"))
        return digest.hexdigest()

    def verify_valid_slot_hashes(self, size: Optional[int] = None) -> None:
        valid_size = int(self.size if size is None else size)
        if valid_size < 0 or valid_size > self.capacity:
            raise RuntimeError(f"invalid replay size for payload verification: {valid_size}")
        for index in range(valid_size):
            recorded = np.ascontiguousarray(self.slot_hashes[index]).tobytes(order="C")
            actual = self._slot_digest(index)
            if actual != recorded:
                raise RuntimeError(
                    f"replay payload slot hash mismatch at index {index}; refusing full resume"
                )


def _replay_arrays(replay: FeatureReplayBuffer) -> Dict[str, np.memmap]:
    result = _payload_data_arrays(replay)
    slot_hashes = getattr(replay, "slot_hashes", None)
    if slot_hashes is not None:
        if not isinstance(slot_hashes, np.memmap):
            raise RuntimeError("feature replay slot_hashes is not a numpy memmap")
        result["slot_hashes"] = slot_hashes
    return result


def _file_inventory(replay: FeatureReplayBuffer, payload_dir: Path) -> Dict[str, Dict[str, Any]]:
    inventory: Dict[str, Dict[str, Any]] = {}
    for logical_name, array in sorted(_replay_arrays(replay).items()):
        path = Path(str(array.filename)).resolve()
        try:
            path.relative_to(payload_dir.resolve())
        except ValueError as exc:
            raise RuntimeError(f"replay mmap escaped UUID payload directory: {path}") from exc
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"replay mmap must be a non-symlink regular file: {path}")
        shape = tuple(int(dim) for dim in array.shape)
        dtype = np.dtype(array.dtype)
        size_bytes = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
        if int(path.stat().st_size) != size_bytes:
            raise RuntimeError(f"replay mmap size mismatch while building manifest: {path}")
        inventory[logical_name] = {
            "filename": path.name,
            "dtype": dtype.str,
            "shape": list(shape),
            "size_bytes": size_bytes,
        }
    if not inventory:
        raise RuntimeError("feature replay has no mmap payload files")
    actual_dat = {path.name for path in payload_dir.glob("*.dat") if path.is_file()}
    declared_dat = {entry["filename"] for entry in inventory.values()}
    if actual_dat != declared_dat:
        raise RuntimeError(
            f"replay payload inventory mismatch: undeclared={sorted(actual_dat - declared_dat)}, "
            f"missing={sorted(declared_dat - actual_dat)}"
        )
    return inventory


def _manifest_payload(
    replay: FeatureReplayBuffer,
    *,
    payload_dir: Path,
    replay_uuid: str,
    experiment_id: str,
    training_signature: str,
) -> Dict[str, Any]:
    return {
        "manifest_version": REPLAY_MANIFEST_VERSION,
        "protocol_version": REPLAY_PROTOCOL_VERSION,
        "created_utc": _utc_now(),
        "replay_uuid": replay_uuid,
        "experiment_id": experiment_id,
        "training_signature_version": TRAINING_SIGNATURE_VERSION,
        "training_signature": training_signature,
        "buffer_kind": "feature_replay",
        "schema_version": int(replay.schema_version),
        "control_semantics": str(replay.control_semantics),
        "capacity": int(replay.capacity),
        "state_shape": list(replay.state_shape),
        "actor_base_shape": list(replay.actor_base_shape),
        "critic_bev_shape": list(replay.critic_bev_shape),
        "trajectory_shape": list(replay.trajectory_shape),
        "pid_summary_dim": int(replay.pid_summary_dim),
        "payload_witness_algorithm": PAYLOAD_WITNESS_ALGORITHM,
        "files": _file_inventory(replay, payload_dir),
    }


def create_feature_replay(
    *,
    replay_root: str,
    experiment_id: str,
    training_signature: str,
    capacity: int,
    state_shape: tuple,
    actor_base_shape: tuple,
    critic_bev_shape: tuple,
    trajectory_shape: tuple,
    pid_summary_dim: int,
    control_semantics: str = CLEAN_DUAL_TRAJECTORY_CONTROL,
) -> Tuple[FeatureReplayBuffer, ReplayContext]:
    """Create a fresh replay in a never-before-used UUID directory."""

    experiment = validate_experiment_id(experiment_id)
    if not training_signature:
        raise ValueError("training_signature is required for replay creation")
    if control_semantics != CLEAN_DUAL_TRAJECTORY_CONTROL:
        raise ValueError("AdaptDrive replay requires clean dual-trajectory control")
    root = Path(replay_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    replay_uuid = str(uuid.uuid4())
    payload_dir = root / replay_uuid
    payload_dir.mkdir(parents=False, exist_ok=False)
    replay = StrictFeatureReplayBuffer(
        capacity,
        state_shape=state_shape,
        actor_base_shape=actor_base_shape,
        critic_bev_shape=critic_bev_shape,
        trajectory_shape=trajectory_shape,
        mmap_dir=str(payload_dir),
        pid_summary_dim=pid_summary_dim,
        control_semantics=control_semantics,
    )
    replay.training_signature = str(training_signature)
    manifest = _manifest_payload(
        replay,
        payload_dir=payload_dir,
        replay_uuid=replay_uuid,
        experiment_id=experiment,
        training_signature=str(training_signature),
    )
    manifest_path = payload_dir / REPLAY_MANIFEST_FILENAME
    manifest_hash = _atomic_write_json(manifest_path, manifest)
    return replay, ReplayContext(
        replay_root=root,
        payload_dir=payload_dir,
        replay_uuid=replay_uuid,
        experiment_id=experiment,
        training_signature=str(training_signature),
        manifest=dict(manifest),
        manifest_sha256=manifest_hash,
    )


def _payload_witness(
    replay: FeatureReplayBuffer,
    *,
    size: int,
    ptr: int,
) -> Dict[str, Any]:
    if not isinstance(replay, StrictFeatureReplayBuffer):
        raise RuntimeError("strict replay snapshots require StrictFeatureReplayBuffer")
    return {
        "algorithm": PAYLOAD_WITNESS_ALGORITHM,
        "valid_slots": int(size),
        "ptr": int(ptr),
        "slot_hash_table_sha256": replay.valid_slot_hash_table_sha256(size),
    }


def write_replay_state_snapshot(
    replay: FeatureReplayBuffer,
    context: ReplayContext,
) -> Dict[str, Any]:
    """Flush payload and write one immutable checkpoint-specific state file."""

    if str(replay.training_signature) != context.training_signature:
        raise RuntimeError("replay object and context training signatures differ")
    replay._flush()
    current_manifest_hash = _sha256_file(context.payload_dir / REPLAY_MANIFEST_FILENAME)
    if current_manifest_hash != context.manifest_sha256:
        raise RuntimeError("replay manifest changed after creation")
    # Re-check all files before any checkpoint points at this replay.
    _validate_payload_files(context.payload_dir, context.manifest)

    checkpoint_uuid = str(uuid.uuid4())
    buffer_state = _jsonable(replay.state_dict())
    state_snapshot = {
        "protocol_version": REPLAY_PROTOCOL_VERSION,
        "checkpoint_uuid": checkpoint_uuid,
        "saved_utc": _utc_now(),
        "replay_uuid": context.replay_uuid,
        "experiment_id": context.experiment_id,
        "training_signature_version": TRAINING_SIGNATURE_VERSION,
        "training_signature": context.training_signature,
        "manifest_sha256": context.manifest_sha256,
        "buffer_state": buffer_state,
        "payload_witness": _payload_witness(
            replay,
            size=int(buffer_state["size"]),
            ptr=int(buffer_state["ptr"]),
        ),
    }
    state_filename = f"replay_state_{checkpoint_uuid}.json"
    state_path = context.payload_dir / state_filename
    state_hash = _atomic_write_json(state_path, state_snapshot)
    return {
        "protocol_version": REPLAY_PROTOCOL_VERSION,
        "checkpoint_uuid": checkpoint_uuid,
        "replay_uuid": context.replay_uuid,
        "experiment_id": context.experiment_id,
        "training_signature_version": TRAINING_SIGNATURE_VERSION,
        "training_signature": context.training_signature,
        "payload_dir": context.replay_uuid,
        "manifest_filename": REPLAY_MANIFEST_FILENAME,
        "manifest_sha256": context.manifest_sha256,
        "state_filename": state_filename,
        "state_sha256": state_hash,
        "replay_size": int(buffer_state["size"]),
    }


def _validate_payload_files(payload_dir: Path, manifest: Mapping[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise RuntimeError("replay manifest has no payload inventory")
    declared = set()
    for logical_name, raw_entry in files.items():
        if not isinstance(raw_entry, Mapping):
            raise RuntimeError(f"invalid replay manifest file entry: {logical_name}")
        path = _safe_child(payload_dir, str(raw_entry.get("filename", "")), label="replay payload")
        declared.add(path.name)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"required replay payload is missing or a symlink: {path}")
        dtype = np.dtype(str(raw_entry.get("dtype", "")))
        shape = tuple(int(dim) for dim in raw_entry.get("shape", ()))
        expected_size = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
        recorded_size = int(raw_entry.get("size_bytes", -1))
        actual_size = int(path.stat().st_size)
        if expected_size != recorded_size or actual_size != recorded_size:
            raise RuntimeError(
                f"replay payload size mismatch for {logical_name}: "
                f"shape/dtype={expected_size}, manifest={recorded_size}, actual={actual_size}"
            )
    actual = {path.name for path in payload_dir.glob("*.dat") if path.is_file()}
    if actual != declared:
        raise RuntimeError(
            f"replay payload file set mismatch: undeclared={sorted(actual - declared)}, "
            f"missing={sorted(declared - actual)}"
        )


def _expected_inventory(
    *,
    capacity: int,
    state_shape: tuple,
    actor_base_shape: tuple,
    critic_bev_shape: tuple,
    trajectory_shape: tuple,
    pid_summary_dim: int,
    control_semantics: str,
) -> Dict[str, Dict[str, Any]]:
    if control_semantics != CLEAN_DUAL_TRAJECTORY_CONTROL:
        raise RuntimeError("strict AdaptDrive replay only supports clean dual-trajectory control")

    def entry(filename: str, dtype, shape: tuple) -> Dict[str, Any]:
        normalized_shape = tuple(int(dim) for dim in shape)
        normalized_dtype = np.dtype(dtype)
        return {
            "filename": filename,
            "dtype": normalized_dtype.str,
            "shape": list(normalized_shape),
            "size_bytes": int(np.prod(normalized_shape, dtype=np.int64)) * int(normalized_dtype.itemsize),
        }

    cap = int(capacity)
    traj = tuple(int(dim) for dim in trajectory_shape)
    return {
        "actor_base_features": entry("actor_base_features.dat", np.float32, (cap, *actor_base_shape)),
        "all_candidates": entry("candidate_lateral_trajectories.dat", np.float16, (cap, 48, *traj)),
        "candidate_longitudinal_trajectories": entry(
            "candidate_longitudinal_trajectories.dat", np.float16, (cap, 48, *traj)
        ),
        "commands": entry("commands.dat", np.int32, (cap,)),
        "critic_bev_features": entry("critic_bev_features.dat", np.float32, (cap, *critic_bev_shape)),
        "dones": entry("dones.dat", np.float32, (cap,)),
        "longitudinal_modes": entry("longitudinal_speed_area_modes.dat", np.int16, (cap,)),
        "longitudinal_trajectories": entry("executed_longitudinal_trajectories.dat", np.float32, (cap, *traj)),
        "next_commands": entry("next_commands.dat", np.int32, (cap,)),
        "next_critic_bev_features": entry(
            "next_critic_bev_features.dat", np.float32, (cap, *critic_bev_shape)
        ),
        "next_states": entry("next_states.dat", np.float32, (cap, *state_shape)),
        "next_target_points": entry("next_target_points.dat", np.float32, (cap, 2)),
        "pid_summaries": entry("dual_pid_summaries.dat", np.float32, (cap, int(pid_summary_dim))),
        "plan_cls_context": entry("plan_cls_context.dat", np.float16, (cap, 48, 256)),
        "prev_pid_summaries": entry(
            "prev_dual_pid_summaries.dat", np.float32, (cap, int(pid_summary_dim))
        ),
        "prev_pid_summary_masks": entry("prev_pid_summary_masks.dat", np.float32, (cap,)),
        "reference_logits": entry("reference_logits.dat", np.float32, (cap, 48)),
        "rewards": entry("rewards.dat", np.float32, (cap,)),
        "selected_lateral_modes": entry("selected_lateral_modes.dat", np.int16, (cap,)),
        "slot_hashes": entry("slot_hashes.dat", np.uint8, (cap, StrictFeatureReplayBuffer.SLOT_HASH_BYTES)),
        "states": entry("states.dat", np.float32, (cap, *state_shape)),
        "target_points": entry("target_points.dat", np.float32, (cap, 2)),
        "trajectories": entry("executed_lateral_trajectories.dat", np.float32, (cap, *traj)),
    }


def _validate_manifest_contract(
    manifest: Mapping[str, Any],
    *,
    replay_uuid: str,
    experiment_id: str,
    training_signature: str,
    capacity: int,
    state_shape: tuple,
    actor_base_shape: tuple,
    critic_bev_shape: tuple,
    trajectory_shape: tuple,
    pid_summary_dim: int,
    control_semantics: str,
) -> None:
    expected = {
        "manifest_version": REPLAY_MANIFEST_VERSION,
        "protocol_version": REPLAY_PROTOCOL_VERSION,
        "replay_uuid": replay_uuid,
        "experiment_id": experiment_id,
        "training_signature_version": TRAINING_SIGNATURE_VERSION,
        "training_signature": training_signature,
        "buffer_kind": "feature_replay",
        "schema_version": CLEAN_DUAL_TRAJECTORY_SCHEMA_VERSION,
        "control_semantics": control_semantics,
        "capacity": int(capacity),
        "state_shape": list(state_shape),
        "actor_base_shape": list(actor_base_shape),
        "critic_bev_shape": list(critic_bev_shape),
        "trajectory_shape": list(trajectory_shape),
        "pid_summary_dim": int(pid_summary_dim),
        "payload_witness_algorithm": PAYLOAD_WITNESS_ALGORITHM,
    }
    mismatched = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatched:
        raise RuntimeError(f"replay manifest contract mismatch: {mismatched}")
    inventory = manifest.get("files")
    expected_inventory = _expected_inventory(
        capacity=capacity,
        state_shape=state_shape,
        actor_base_shape=actor_base_shape,
        critic_bev_shape=critic_bev_shape,
        trajectory_shape=trajectory_shape,
        pid_summary_dim=pid_summary_dim,
        control_semantics=control_semantics,
    )
    if not isinstance(inventory, Mapping) or dict(inventory) != expected_inventory:
        raise RuntimeError("replay manifest payload inventory does not match the canonical schema")


def _validate_state_snapshot(
    snapshot: Mapping[str, Any],
    *,
    replay_ref: Mapping[str, Any],
    context: ReplayContext,
) -> Mapping[str, Any]:
    expected = {
        "protocol_version": REPLAY_PROTOCOL_VERSION,
        "checkpoint_uuid": replay_ref.get("checkpoint_uuid"),
        "replay_uuid": context.replay_uuid,
        "experiment_id": context.experiment_id,
        "training_signature_version": TRAINING_SIGNATURE_VERSION,
        "training_signature": context.training_signature,
        "manifest_sha256": context.manifest_sha256,
    }
    mismatched = {
        key: (snapshot.get(key), value)
        for key, value in expected.items()
        if snapshot.get(key) != value
    }
    if mismatched:
        raise RuntimeError(f"replay state snapshot mismatch: {mismatched}")
    state = snapshot.get("buffer_state")
    if not isinstance(state, Mapping):
        raise RuntimeError("replay state snapshot has no buffer_state")
    if int(state.get("size", -1)) != int(replay_ref.get("replay_size", -2)):
        raise RuntimeError("checkpoint replay size and state snapshot size differ")
    size = int(state.get("size", -1))
    capacity = int(state.get("capacity", -1))
    ptr = int(state.get("ptr", -1))
    if size < 0 or capacity <= 0 or size > capacity or ptr < 0 or ptr >= capacity:
        raise RuntimeError("replay state snapshot has invalid ptr/size/capacity")
    if size < capacity and ptr != size:
        raise RuntimeError(
            f"non-full replay pointer invariant violated: ptr={ptr}, size={size}, capacity={capacity}"
        )
    witness = snapshot.get("payload_witness")
    if not isinstance(witness, Mapping) or witness.get("algorithm") != PAYLOAD_WITNESS_ALGORITHM:
        raise RuntimeError("replay state snapshot has no supported payload witness")
    return state


def _verify_payload_witness(
    replay: FeatureReplayBuffer,
    *,
    snapshot: Mapping[str, Any],
    state: Mapping[str, Any],
) -> None:
    expected = snapshot.get("payload_witness")
    actual = _payload_witness(replay, size=int(state["size"]), ptr=int(state["ptr"]))
    if actual != expected:
        raise RuntimeError("replay slot-hash table witness mismatch; refusing to expose mmap samples")
    if not isinstance(replay, StrictFeatureReplayBuffer):
        raise RuntimeError("strict replay resume requires StrictFeatureReplayBuffer")
    replay.verify_valid_slot_hashes(int(state["size"]))


def resume_feature_replay(
    checkpoint: Mapping[str, Any],
    *,
    replay_root: str,
    experiment_id: str,
    training_signature: str,
    capacity: int,
    state_shape: tuple,
    actor_base_shape: tuple,
    critic_bev_shape: tuple,
    trajectory_shape: tuple,
    pid_summary_dim: int,
    control_semantics: str = CLEAN_DUAL_TRAJECTORY_CONTROL,
) -> Tuple[FeatureReplayBuffer, ReplayContext]:
    """Strictly reopen a v8 replay; no metadata-only fallback exists."""

    experiment = validate_experiment_id(experiment_id)
    if int(checkpoint.get("training_signature_version", 0)) != TRAINING_SIGNATURE_VERSION:
        raise RuntimeError("full replay resume requires a signature-v8 checkpoint")
    if str(checkpoint.get("training_signature", "")) != str(training_signature):
        raise RuntimeError("full replay resume requires an exact training signature match")
    if str(checkpoint.get("experiment_id", "")) != experiment:
        raise RuntimeError("full replay resume requires the same experiment_id")
    replay_ref = checkpoint.get("replay_ref")
    if not isinstance(replay_ref, Mapping):
        raise RuntimeError("v8 full resume checkpoint has no replay_ref")
    if str(checkpoint.get("checkpoint_uuid", "")) != str(replay_ref.get("checkpoint_uuid", "")):
        raise RuntimeError("checkpoint UUID and replay reference UUID differ")
    expected_ref = {
        "protocol_version": REPLAY_PROTOCOL_VERSION,
        "experiment_id": experiment,
        "training_signature_version": TRAINING_SIGNATURE_VERSION,
        "training_signature": str(training_signature),
    }
    ref_mismatch = {
        key: (replay_ref.get(key), value)
        for key, value in expected_ref.items()
        if replay_ref.get(key) != value
    }
    if ref_mismatch:
        raise RuntimeError(f"checkpoint replay reference mismatch: {ref_mismatch}")
    replay_uuid = str(replay_ref.get("replay_uuid", ""))
    try:
        uuid.UUID(replay_uuid)
    except (ValueError, AttributeError) as exc:
        raise RuntimeError(f"invalid replay UUID in checkpoint: {replay_uuid!r}") from exc
    if str(replay_ref.get("payload_dir", "")) != replay_uuid:
        raise RuntimeError("checkpoint replay payload_dir must equal replay_uuid")

    root = Path(replay_root).expanduser().resolve()
    payload_dir = _safe_child(root, replay_uuid, label="replay UUID directory")
    if not payload_dir.is_dir() or payload_dir.is_symlink():
        raise RuntimeError(f"replay UUID payload directory is missing or a symlink: {payload_dir}")
    manifest_path = _safe_child(
        payload_dir,
        str(replay_ref.get("manifest_filename", "")),
        label="replay manifest",
    )
    manifest_hash = _sha256_file(manifest_path)
    if manifest_hash != str(replay_ref.get("manifest_sha256", "")):
        raise RuntimeError("replay manifest SHA-256 mismatch")
    manifest = _load_json(manifest_path, label="replay manifest")
    _validate_manifest_contract(
        manifest,
        replay_uuid=replay_uuid,
        experiment_id=experiment,
        training_signature=str(training_signature),
        capacity=capacity,
        state_shape=state_shape,
        actor_base_shape=actor_base_shape,
        critic_bev_shape=critic_bev_shape,
        trajectory_shape=trajectory_shape,
        pid_summary_dim=pid_summary_dim,
        control_semantics=control_semantics,
    )
    _validate_payload_files(payload_dir, manifest)
    state_path = _safe_child(
        payload_dir,
        str(replay_ref.get("state_filename", "")),
        label="replay state snapshot",
    )
    state_hash = _sha256_file(state_path)
    if state_hash != str(replay_ref.get("state_sha256", "")):
        raise RuntimeError("replay state snapshot SHA-256 mismatch")
    snapshot = _load_json(state_path, label="replay state snapshot")
    context = ReplayContext(
        replay_root=root,
        payload_dir=payload_dir,
        replay_uuid=replay_uuid,
        experiment_id=experiment,
        training_signature=str(training_signature),
        manifest=dict(manifest),
        manifest_sha256=manifest_hash,
    )
    state = _validate_state_snapshot(snapshot, replay_ref=replay_ref, context=context)

    # Construction occurs only after all payload files and byte sizes passed;
    # therefore the legacy ``_create_memmap`` helper can only take its reuse
    # branch and cannot create an empty replacement during resume.
    replay = StrictFeatureReplayBuffer(
        capacity,
        state_shape=state_shape,
        actor_base_shape=actor_base_shape,
        critic_bev_shape=critic_bev_shape,
        trajectory_shape=trajectory_shape,
        mmap_dir=str(payload_dir),
        pid_summary_dim=pid_summary_dim,
        control_semantics=control_semantics,
    )
    replay.training_signature = str(training_signature)
    try:
        _verify_payload_witness(replay, snapshot=snapshot, state=state)
        if not replay.load_state_dict(dict(state)):
            raise RuntimeError("validated replay state was rejected by FeatureReplayBuffer")
        if int(state.get("size", 0)) > 0 and not manifest.get("files"):
            raise RuntimeError("non-empty replay has no mmap payload inventory")
    except Exception:
        replay.close()
        raise
    return replay, context
