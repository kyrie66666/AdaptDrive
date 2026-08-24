"""
Replay Buffer for Bench2Drive RL (Memory-Mapped Version)
========================================================

Memory-mapped replay buffer for SAC training with prev_bev support.
Stores data on SSD to support large capacity with limited RAM.
"""

from typing import Dict, Optional, NamedTuple
import numpy as np
import torch
import os
from pathlib import Path

SCHEMA_VERSION = 3
CLEAN_DUAL_TRAJECTORY_SCHEMA_VERSION = 5
CLEAN_DUAL_TRAJECTORY_CONTROL = 'hipad_clean_dual_pid_v2_mode_aligned'


def _index_numpy(storage, indices: np.ndarray) -> np.ndarray:
    """Materialize one indexed slice without forcing an extra copy."""
    return np.asarray(storage[indices])


def _tensor_from_storage(storage, indices: np.ndarray, device: str, dtype: torch.dtype) -> torch.Tensor:
    array = _index_numpy(storage, indices)
    tensor = torch.from_numpy(array)
    return tensor.to(device=device, dtype=dtype, non_blocking=True)


def _column_tensor_from_storage(storage, indices: np.ndarray, device: str, dtype: torch.dtype) -> torch.Tensor:
    return _tensor_from_storage(storage, indices, device, dtype).unsqueeze(1)


class Transition(NamedTuple):
    """Single transition in replay buffer."""
    observation: Dict[str, np.ndarray]
    action: np.ndarray
    reward: float
    next_observation: Dict[str, np.ndarray]
    done: bool


class ReplayBuffer:
    """
    Memory-mapped replay buffer for off-policy RL with temporal modeling support.

    Stores observations, trajectories, prev_bev, rewards, next observations, and done flags
    on disk using numpy memmap, allowing large capacity with limited RAM.
    """

    PID_SUMMARY_DIM = 7

    @staticmethod
    def estimate_storage_bytes(
        capacity: int,
        observation_example: Dict[str, np.ndarray],
        trajectory_shape: tuple,
        prev_bev_shape: tuple = (40000, 256),
    ) -> int:
        """Estimate total on-disk bytes for a given replay capacity."""
        rgb_shape = (capacity, *observation_example['rgb'].shape)
        state_shape = (capacity, *observation_example['state'].shape)
        traj_shape = (capacity, *trajectory_shape)
        prev_bev_full_shape = (capacity, *prev_bev_shape)

        total_bytes = (
            np.prod(rgb_shape) * 2 +  # obs_rgb + next_obs_rgb
            np.prod(state_shape) * 3 * 4 +  # obs_state + prev_state + next_obs_state
            capacity * 18 * 4 * 3 +  # obs_can_bus + prev_can_bus + next_obs_can_bus
            np.prod(traj_shape) * 4 +  # trajectories
            np.prod(prev_bev_full_shape) * 4 +  # prev_bevs
            capacity * ReplayBuffer.PID_SUMMARY_DIM * 4 * 2 +  # pid summaries + prev pid summaries
            capacity * 4 * 9  # rewards + dones + masks/nav arrays
        )
        return int(total_bytes)

    @staticmethod
    def capacity_for_storage_budget(
        max_storage_bytes: int,
        observation_example: Dict[str, np.ndarray],
        trajectory_shape: tuple,
        prev_bev_shape: tuple = (40000, 256),
    ) -> int:
        """Compute the maximum replay capacity that fits inside a storage budget."""
        if max_storage_bytes <= 0:
            return 1

        bytes_per_transition = ReplayBuffer.estimate_storage_bytes(
            1,
            observation_example,
            trajectory_shape,
            prev_bev_shape=prev_bev_shape,
        )
        if bytes_per_transition <= 0:
            return 1
        return max(1, int(max_storage_bytes // bytes_per_transition))

    def __init__(
        self,
        capacity: int,
        observation_example: Dict[str, np.ndarray],
        trajectory_shape: tuple,  # (fut_ts, 2) for trajectory waypoints
        prev_bev_shape: tuple = (40000, 256),  # Default BEV feature shape from VAD
        mmap_dir: str = './replay_buffer_mmap',  # Directory for memory-mapped files
    ):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        self.prev_bev_shape = prev_bev_shape
        self.training_signature: Optional[str] = None
        self.mmap_dir = Path(mmap_dir)
        self.mmap_dir.mkdir(parents=True, exist_ok=True)

        print(f"[ReplayBuffer] Initializing memory-mapped buffer:")
        print(f"  Capacity: {capacity}")
        print(f"  Storage: {mmap_dir}")

        # Calculate file sizes
        rgb_shape = (capacity, *observation_example['rgb'].shape)
        state_shape = (capacity, *observation_example['state'].shape)
        traj_shape = (capacity, *trajectory_shape)
        prev_bev_full_shape = (capacity, *prev_bev_shape)

        # Create memory-mapped arrays on disk
        self.observations = {
            'rgb': self._create_memmap('obs_rgb.dat', rgb_shape, np.uint8),
            'state': self._create_memmap('obs_state.dat', state_shape, np.float32),
            'can_bus': self._create_memmap('obs_can_bus.dat', (capacity, 18), np.float32),
        }
        self.prev_states = self._create_memmap('prev_states.dat', state_shape, np.float32)
        self.prev_state_masks = self._create_memmap('prev_state_masks.dat', (capacity,), np.float32)
        self.prev_can_buses = self._create_memmap('prev_can_buses.dat', (capacity, 18), np.float32)
        self.prev_can_bus_masks = self._create_memmap('prev_can_bus_masks.dat', (capacity,), np.float32)
        # Navigation information (target_point: [2], command: scalar)
        self.target_points = self._create_memmap('target_points.dat', (capacity, 2), np.float32)
        self.commands = self._create_memmap('commands.dat', (capacity,), np.int32)
        self.trajectories = self._create_memmap('trajectories.dat', traj_shape, np.float32)
        self.pid_summaries = self._create_memmap('pid_summaries.dat', (capacity, self.PID_SUMMARY_DIM), np.float32)
        self.prev_pid_summaries = self._create_memmap('prev_pid_summaries.dat', (capacity, self.PID_SUMMARY_DIM), np.float32)
        self.prev_pid_summary_masks = self._create_memmap('prev_pid_summary_masks.dat', (capacity,), np.float32)
        self.prev_bevs = self._create_memmap('prev_bevs.dat', prev_bev_full_shape, np.float32)
        self.prev_bev_masks = self._create_memmap('prev_bev_masks.dat', (capacity,), np.float32)
        self.rewards = self._create_memmap('rewards.dat', (capacity,), np.float32)
        self.next_observations = {
            'rgb': self._create_memmap('next_obs_rgb.dat', rgb_shape, np.uint8),
            'state': self._create_memmap('next_obs_state.dat', state_shape, np.float32),
            'can_bus': self._create_memmap('next_obs_can_bus.dat', (capacity, 18), np.float32),
        }
        self.next_target_points = self._create_memmap('next_target_points.dat', (capacity, 2), np.float32)
        self.next_commands = self._create_memmap('next_commands.dat', (capacity,), np.int32)
        self.dones = self._create_memmap('dones.dat', (capacity,), np.float32)

        # Estimate storage usage
        total_bytes = self.estimate_storage_bytes(
            capacity,
            observation_example,
            trajectory_shape,
            prev_bev_shape=prev_bev_shape,
        )
        print(f"  Estimated storage: {total_bytes / 1024**3:.2f} GB")

    def _create_memmap(self, filename: str, shape: tuple, dtype: np.dtype) -> np.memmap:
        """Create or reuse a memory-mapped numpy array on disk."""
        filepath = self.mmap_dir / filename
        expected_bytes = np.prod(shape) * np.dtype(dtype).itemsize

        if filepath.exists():
            actual_bytes = filepath.stat().st_size
            if actual_bytes == expected_bytes:
                print(f"[ReplayBuffer] Reusing existing: {filename}")
                return np.memmap(filepath, dtype=dtype, mode='r+', shape=shape)
            else:
                print(f"[ReplayBuffer] Size mismatch, recreating: {filename}")
                return np.memmap(filepath, dtype=dtype, mode='w+', shape=shape)
        else:
            print(f"[ReplayBuffer] Creating new: {filename}")
            return np.memmap(filepath, dtype=dtype, mode='w+', shape=shape)

    def state_dict(self) -> Dict[str, object]:
        """Serialize replay bookkeeping state (the mmap payload lives on disk already)."""
        state = {
            'ptr': int(self.ptr),
            'size': int(self.size),
            'capacity': int(self.capacity),
            'schema_version': SCHEMA_VERSION,
            'buffer_kind': 'raw_replay',
        }
        if self.training_signature is not None:
            state['training_signature'] = self.training_signature
        return state

    def load_state_dict(self, state: Dict[str, object]) -> bool:
        """Restore replay bookkeeping state from an in-memory dict."""
        if not state:
            return False

        schema_version = int(state.get('schema_version', 0))
        if schema_version != SCHEMA_VERSION:
            print(f"[ReplayBuffer] Schema mismatch: found {schema_version}, expected {SCHEMA_VERSION}")
            return False
        if state.get('buffer_kind') != 'raw_replay':
            print(f"[ReplayBuffer] Buffer-kind mismatch: found {state.get('buffer_kind')}, expected raw_replay")
            return False

        saved_capacity = int(state.get('capacity', self.capacity))
        if saved_capacity != self.capacity:
            print(f"[ReplayBuffer] Capacity mismatch: found {saved_capacity}, expected {self.capacity}")
            return False

        saved_signature = state.get('training_signature')
        if self.training_signature is not None and saved_signature != self.training_signature:
            print(
                f"[ReplayBuffer] Training-signature mismatch: found {saved_signature}, "
                f"expected {self.training_signature}"
            )
            return False

        ptr = int(state.get('ptr', 0))
        size = int(state.get('size', 0))
        if size < 0 or size > self.capacity:
            print(f"[ReplayBuffer] Invalid size in state: {size}")
            return False
        if ptr < 0 or ptr >= self.capacity:
            print(f"[ReplayBuffer] Invalid ptr in state: {ptr}")
            return False

        self.ptr = ptr
        self.size = size
        print(f"[ReplayBuffer] State loaded: ptr={self.ptr}, size={self.size}")
        return True

    def save_state(self, filepath: str):
        """Save buffer state (ptr, size) to disk."""
        state = self.state_dict()
        np.save(filepath, state)
        print(f"[ReplayBuffer] State saved: ptr={self.ptr}, size={self.size}")

    def load_state(self, filepath: str) -> bool:
        """Load buffer state from disk."""
        if not os.path.exists(filepath):
            return False
        try:
            state = np.load(filepath, allow_pickle=True).item()
            return self.load_state_dict(state)
        except Exception as e:
            print(f"[ReplayBuffer] Failed to load state: {e}")
            return False

    def add(
        self,
        observation: Dict[str, np.ndarray],
        trajectory: np.ndarray,  # [fut_ts, 2] trajectory waypoints
        reward: float,
        next_observation: Dict[str, np.ndarray],
        done: bool,
        prev_bev: Optional[np.ndarray] = None,  # [prev_bev_shape] BEV features
        prev_state: Optional[np.ndarray] = None,
        prev_can_bus: Optional[np.ndarray] = None,
        pid_summary: Optional[np.ndarray] = None,
        prev_pid_summary: Optional[np.ndarray] = None,
        info: Optional[Dict] = None,
    ):
        """Add a transition to the buffer."""
        self.observations['rgb'][self.ptr] = observation['rgb']
        self.observations['state'][self.ptr] = observation['state']
        if 'can_bus' in observation:
            self.observations['can_bus'][self.ptr] = observation['can_bus']
        else:
            self.observations['can_bus'][self.ptr] = np.zeros(18, dtype=np.float32)
        if prev_state is not None:
            self.prev_states[self.ptr] = prev_state
            self.prev_state_masks[self.ptr] = 1.0
        else:
            self.prev_states[self.ptr] = np.zeros_like(observation['state'], dtype=np.float32)
            self.prev_state_masks[self.ptr] = 0.0
        if prev_can_bus is not None:
            self.prev_can_buses[self.ptr] = prev_can_bus
            self.prev_can_bus_masks[self.ptr] = 1.0
        else:
            self.prev_can_buses[self.ptr] = np.zeros(18, dtype=np.float32)
            self.prev_can_bus_masks[self.ptr] = 0.0
        if prev_pid_summary is not None:
            self.prev_pid_summaries[self.ptr] = prev_pid_summary
            self.prev_pid_summary_masks[self.ptr] = 1.0
        else:
            self.prev_pid_summaries[self.ptr] = np.zeros(self.PID_SUMMARY_DIM, dtype=np.float32)
            self.prev_pid_summary_masks[self.ptr] = 0.0

        # Store navigation information (with defaults if not present)
        if 'target_point' in observation:
            self.target_points[self.ptr] = observation['target_point']
        else:
            self.target_points[self.ptr] = np.zeros(2, dtype=np.float32)

        if 'command' in observation:
            self.commands[self.ptr] = int(observation['command'])
        else:
            self.commands[self.ptr] = 3  # Default: STRAIGHT

        self.trajectories[self.ptr] = trajectory
        if pid_summary is not None:
            self.pid_summaries[self.ptr] = pid_summary
        else:
            self.pid_summaries[self.ptr] = np.zeros(self.PID_SUMMARY_DIM, dtype=np.float32)

        # Store prev_bev (or zeros if None)
        if prev_bev is not None:
            self.prev_bevs[self.ptr] = prev_bev
            self.prev_bev_masks[self.ptr] = 1.0
        else:
            self.prev_bevs[self.ptr] = np.zeros(self.prev_bev_shape, dtype=np.float32)
            self.prev_bev_masks[self.ptr] = 0.0

        self.rewards[self.ptr] = reward
        self.next_observations['rgb'][self.ptr] = next_observation['rgb']
        self.next_observations['state'][self.ptr] = next_observation['state']
        if 'can_bus' in next_observation:
            self.next_observations['can_bus'][self.ptr] = next_observation['can_bus']
        else:
            self.next_observations['can_bus'][self.ptr] = np.zeros(18, dtype=np.float32)
        if 'target_point' in next_observation:
            self.next_target_points[self.ptr] = next_observation['target_point']
        else:
            self.next_target_points[self.ptr] = np.zeros(2, dtype=np.float32)

        if 'command' in next_observation:
            self.next_commands[self.ptr] = int(next_observation['command'])
        else:
            self.next_commands[self.ptr] = 3  # Default: STRAIGHT
        self.dones[self.ptr] = float(done)

        # Flush to disk periodically (every 100 writes to amortize cost)
        if self.ptr % 100 == 0:
            self._flush()

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _flush(self):
        """Flush memory-mapped arrays to disk."""
        self.observations['rgb'].flush()
        self.observations['state'].flush()
        self.observations['can_bus'].flush()
        self.prev_states.flush()
        self.prev_state_masks.flush()
        self.prev_can_buses.flush()
        self.prev_can_bus_masks.flush()
        self.pid_summaries.flush()
        self.prev_pid_summaries.flush()
        self.prev_pid_summary_masks.flush()
        self.target_points.flush()
        self.commands.flush()
        self.trajectories.flush()
        self.prev_bevs.flush()
        self.prev_bev_masks.flush()
        self.rewards.flush()
        self.next_observations['rgb'].flush()
        self.next_observations['state'].flush()
        self.next_observations['can_bus'].flush()
        self.next_target_points.flush()
        self.next_commands.flush()
        self.dones.flush()

    def sample(self, batch_size: int, device: str = 'cuda') -> 'Batch':
        """Sample a batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)

        batch = Batch(
            observations={
                'rgb': _tensor_from_storage(self.observations['rgb'], indices, device, torch.float32),
                'state': _tensor_from_storage(self.observations['state'], indices, device, torch.float32),
                'can_bus': _tensor_from_storage(self.observations['can_bus'], indices, device, torch.float32),
                'target_point': _tensor_from_storage(self.target_points, indices, device, torch.float32),
                'command': _tensor_from_storage(self.commands, indices, device, torch.long),
            },
            prev_states=_tensor_from_storage(self.prev_states, indices, device, torch.float32),
            prev_state_masks=_tensor_from_storage(self.prev_state_masks, indices, device, torch.float32),
            prev_can_buses=_tensor_from_storage(self.prev_can_buses, indices, device, torch.float32),
            prev_can_bus_masks=_tensor_from_storage(self.prev_can_bus_masks, indices, device, torch.float32),
            pid_summaries=_tensor_from_storage(self.pid_summaries, indices, device, torch.float32),
            prev_pid_summaries=_tensor_from_storage(self.prev_pid_summaries, indices, device, torch.float32),
            prev_pid_summary_masks=_tensor_from_storage(self.prev_pid_summary_masks, indices, device, torch.float32),
            trajectories=_tensor_from_storage(self.trajectories, indices, device, torch.float32),
            prev_bevs=_tensor_from_storage(self.prev_bevs, indices, device, torch.float32),
            prev_bev_masks=_tensor_from_storage(self.prev_bev_masks, indices, device, torch.float32),
            rewards=_column_tensor_from_storage(self.rewards, indices, device, torch.float32),
            next_observations={
                'rgb': _tensor_from_storage(self.next_observations['rgb'], indices, device, torch.float32),
                'state': _tensor_from_storage(self.next_observations['state'], indices, device, torch.float32),
                'can_bus': _tensor_from_storage(self.next_observations['can_bus'], indices, device, torch.float32),
                'target_point': _tensor_from_storage(self.next_target_points, indices, device, torch.float32),
                'command': _tensor_from_storage(self.next_commands, indices, device, torch.long),
            },
            dones=_column_tensor_from_storage(self.dones, indices, device, torch.float32),
        )

        return batch

    def __len__(self) -> int:
        return self.size

    def close(self):
        """Close memory-mapped files and cleanup."""
        self._flush()
        # Files will be closed when memmap objects are garbage collected
        print(f"[ReplayBuffer] Closed. Data remains at: {self.mmap_dir}")

    def cleanup(self):
        """Delete all memory-mapped files."""
        self.close()
        import shutil
        if self.mmap_dir.exists():
            shutil.rmtree(self.mmap_dir)
            print(f"[ReplayBuffer] Cleaned up: {self.mmap_dir}")


class FeatureReplayBuffer:
    """Replay buffer that stores frozen features instead of raw images."""

    PID_SUMMARY_DIM = ReplayBuffer.PID_SUMMARY_DIM

    @staticmethod
    def estimate_storage_bytes(
        capacity: int,
        state_shape: tuple,
        actor_base_shape: tuple,
        critic_bev_shape: tuple,
        trajectory_shape: tuple,
        pid_summary_dim: int = ReplayBuffer.PID_SUMMARY_DIM,
        control_semantics: str = 'single_trajectory_legacy',
    ) -> int:
        trajectory_copies = 2 if control_semantics == CLEAN_DUAL_TRAJECTORY_CONTROL else 1
        total_bytes = (
            np.prod((capacity, *state_shape)) * 4 * 2 +  # state + next_state
            capacity * 2 * 4 * 2 +  # target_point + next_target_point
            capacity * 4 * 2 +  # command + next_command
            np.prod((capacity, *actor_base_shape)) * 4 +
            np.prod((capacity, *critic_bev_shape)) * 4 * 2 +  # critic + next_critic
            capacity * pid_summary_dim * 4 * 2 +  # pid + prev_pid
            capacity * 4 +  # prev pid mask
            np.prod((capacity, *trajectory_shape)) * 4 * trajectory_copies +
            capacity * 4 * 2 +  # reward + done
            capacity * 2 * 2 +  # selected lateral mode + longitudinal speed-area mode
            capacity * 48 * 256 * 2 +  # plan_cls_context (FP16)
            capacity * 48 * trajectory_shape[0] * trajectory_shape[1] * 2 +  # all_candidates (FP16)
            (
                capacity * 48 * trajectory_shape[0] * trajectory_shape[1] * 2
                if control_semantics == CLEAN_DUAL_TRAJECTORY_CONTROL
                else 0
            ) +  # mode-aligned longitudinal candidates (FP16)
            capacity * 48 * 4  # reference_logits (FP32)
        )
        return int(total_bytes)

    @staticmethod
    def capacity_for_storage_budget(
        max_storage_bytes: int,
        state_shape: tuple,
        actor_base_shape: tuple,
        critic_bev_shape: tuple,
        trajectory_shape: tuple,
        pid_summary_dim: int = ReplayBuffer.PID_SUMMARY_DIM,
        control_semantics: str = 'single_trajectory_legacy',
    ) -> int:
        if max_storage_bytes <= 0:
            return 1
        bytes_per_transition = FeatureReplayBuffer.estimate_storage_bytes(
            1,
            state_shape,
            actor_base_shape,
            critic_bev_shape,
            trajectory_shape,
            pid_summary_dim=pid_summary_dim,
            control_semantics=control_semantics,
        )
        if bytes_per_transition <= 0:
            return 1
        return max(1, int(max_storage_bytes // bytes_per_transition))

    def __init__(
        self,
        capacity: int,
        state_shape: tuple,
        actor_base_shape: tuple,
        critic_bev_shape: tuple,
        trajectory_shape: tuple,
        mmap_dir: str = './feature_replay_buffer_mmap',
        pid_summary_dim: int = ReplayBuffer.PID_SUMMARY_DIM,
        control_semantics: str = 'single_trajectory_legacy',
    ):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        self.training_signature: Optional[str] = None
        self.mmap_dir = Path(mmap_dir)
        self.mmap_dir.mkdir(parents=True, exist_ok=True)
        self.state_shape = state_shape
        self.actor_base_shape = actor_base_shape
        self.critic_bev_shape = critic_bev_shape
        self.trajectory_shape = trajectory_shape
        self.pid_summary_dim = int(pid_summary_dim)
        self.control_semantics = str(control_semantics)
        self.schema_version = (
            CLEAN_DUAL_TRAJECTORY_SCHEMA_VERSION
            if self.control_semantics == CLEAN_DUAL_TRAJECTORY_CONTROL
            else SCHEMA_VERSION
        )

        print(f"[FeatureReplayBuffer] Initializing memory-mapped buffer:")
        print(f"  Capacity: {capacity}")
        print(f"  Storage: {mmap_dir}")

        self.states = self._create_memmap('states.dat', (capacity, *state_shape), np.float32)
        self.target_points = self._create_memmap('target_points.dat', (capacity, 2), np.float32)
        self.commands = self._create_memmap('commands.dat', (capacity,), np.int32)
        self.actor_base_features = self._create_memmap('actor_base_features.dat', (capacity, *actor_base_shape), np.float32)
        self.critic_bev_features = self._create_memmap('critic_bev_features.dat', (capacity, *critic_bev_shape), np.float32)
        dual_control = self.control_semantics == CLEAN_DUAL_TRAJECTORY_CONTROL
        pid_name = 'dual_pid_summaries.dat' if dual_control else 'pid_summaries.dat'
        prev_pid_name = 'prev_dual_pid_summaries.dat' if dual_control else 'prev_pid_summaries.dat'
        lateral_name = 'executed_lateral_trajectories.dat' if dual_control else 'trajectories.dat'
        candidate_name = 'candidate_lateral_trajectories.dat' if dual_control else 'all_candidates.dat'
        self.prev_pid_summaries = self._create_memmap(
            prev_pid_name,
            (capacity, self.pid_summary_dim),
            np.float32,
        )
        self.prev_pid_summary_masks = self._create_memmap('prev_pid_summary_masks.dat', (capacity,), np.float32)
        self.trajectories = self._create_memmap(lateral_name, (capacity, *trajectory_shape), np.float32)
        self.longitudinal_trajectories = (
            self._create_memmap(
                'executed_longitudinal_trajectories.dat',
                (capacity, *trajectory_shape),
                np.float32,
            )
            if dual_control
            else None
        )
        self.pid_summaries = self._create_memmap(
            pid_name,
            (capacity, self.pid_summary_dim),
            np.float32,
        )
        self.selected_lateral_modes = (
            self._create_memmap('selected_lateral_modes.dat', (capacity,), np.int16)
            if dual_control
            else None
        )
        self.longitudinal_modes = (
            self._create_memmap('longitudinal_speed_area_modes.dat', (capacity,), np.int16)
            if dual_control
            else None
        )
        self.rewards = self._create_memmap('rewards.dat', (capacity,), np.float32)
        self.next_states = self._create_memmap('next_states.dat', (capacity, *state_shape), np.float32)
        self.next_target_points = self._create_memmap('next_target_points.dat', (capacity, 2), np.float32)
        self.next_commands = self._create_memmap('next_commands.dat', (capacity,), np.int32)
        self.next_critic_bev_features = self._create_memmap('next_critic_bev_features.dat', (capacity, *critic_bev_shape), np.float32)
        self.dones = self._create_memmap('dones.dat', (capacity,), np.float32)
        # Off-policy actor replay fields.
        self.plan_cls_context = self._create_memmap(
            'plan_cls_context.dat', (capacity, 48, 256), np.float16,
        )
        self.all_candidates = self._create_memmap(
            candidate_name, (capacity, 48, *trajectory_shape), np.float16,
        )
        self.candidate_longitudinal_trajectories = (
            self._create_memmap(
                'candidate_longitudinal_trajectories.dat',
                (capacity, 48, *trajectory_shape),
                np.float16,
            )
            if dual_control
            else None
        )
        self.reference_logits = self._create_memmap(
            'reference_logits.dat', (capacity, 48), np.float32,
        )

        total_bytes = FeatureReplayBuffer.estimate_storage_bytes(
            capacity,
            state_shape,
            actor_base_shape,
            critic_bev_shape,
            trajectory_shape,
            pid_summary_dim=self.pid_summary_dim,
            control_semantics=self.control_semantics,
        )
        print(f"  Estimated storage: {total_bytes / 1024**3:.4f} GB")

    def _create_memmap(self, filename: str, shape: tuple, dtype: np.dtype) -> np.memmap:
        filepath = self.mmap_dir / filename
        expected_bytes = np.prod(shape) * np.dtype(dtype).itemsize
        if filepath.exists():
            actual_bytes = filepath.stat().st_size
            if actual_bytes == expected_bytes:
                print(f"[FeatureReplayBuffer] Reusing existing: {filename}")
                return np.memmap(filepath, dtype=dtype, mode='r+', shape=shape)
            print(f"[FeatureReplayBuffer] Size mismatch, recreating: {filename}")
        else:
            print(f"[FeatureReplayBuffer] Creating new: {filename}")
        return np.memmap(filepath, dtype=dtype, mode='w+', shape=shape)

    def state_dict(self) -> Dict[str, object]:
        state = {
            'ptr': int(self.ptr),
            'size': int(self.size),
            'capacity': int(self.capacity),
            'schema_version': self.schema_version,
            'buffer_kind': 'feature_replay',
            'control_semantics': self.control_semantics,
            'pid_summary_dim': self.pid_summary_dim,
            'longitudinal_source': (
                'plan_speed_5hz_frozen_clean_decoder_per_lateral_mode'
                if self.control_semantics == CLEAN_DUAL_TRAJECTORY_CONTROL
                else 'legacy_selected_trajectory'
            ),
            'lateral_source': (
                'sac_selected_plan_spat_2m'
                if self.control_semantics == CLEAN_DUAL_TRAJECTORY_CONTROL
                else 'legacy_selected_trajectory'
            ),
            'state_shape': tuple(self.state_shape),
            'actor_base_shape': tuple(self.actor_base_shape),
            'critic_bev_shape': tuple(self.critic_bev_shape),
            'trajectory_shape': tuple(self.trajectory_shape),
        }
        if self.training_signature is not None:
            state['training_signature'] = self.training_signature
        return state

    def load_state_dict(self, state: Dict[str, object]) -> bool:
        if not state:
            return False
        schema_version = int(state.get('schema_version', 0))
        if schema_version != self.schema_version:
            print(f"[FeatureReplayBuffer] Schema mismatch: found {schema_version}, expected {self.schema_version}")
            return False
        if state.get('buffer_kind') != 'feature_replay':
            print(
                f"[FeatureReplayBuffer] Buffer-kind mismatch: found {state.get('buffer_kind')}, "
                f"expected feature_replay"
            )
            return False
        if state.get('control_semantics', 'single_trajectory_legacy') != self.control_semantics:
            print(
                f"[FeatureReplayBuffer] Control-semantics mismatch: found {state.get('control_semantics')}, "
                f"expected {self.control_semantics}"
            )
            return False
        if int(state.get('pid_summary_dim', ReplayBuffer.PID_SUMMARY_DIM)) != self.pid_summary_dim:
            print("[FeatureReplayBuffer] PID-summary dimension mismatch")
            return False
        if int(state.get('capacity', self.capacity)) != self.capacity:
            print(f"[FeatureReplayBuffer] Capacity mismatch: found {state.get('capacity')}, expected {self.capacity}")
            return False
        if tuple(state.get('state_shape', self.state_shape)) != tuple(self.state_shape):
            print("[FeatureReplayBuffer] State-shape mismatch")
            return False
        if tuple(state.get('actor_base_shape', self.actor_base_shape)) != tuple(self.actor_base_shape):
            print("[FeatureReplayBuffer] Actor-base shape mismatch")
            return False
        if tuple(state.get('critic_bev_shape', self.critic_bev_shape)) != tuple(self.critic_bev_shape):
            print("[FeatureReplayBuffer] Critic-BEV shape mismatch")
            return False
        if tuple(state.get('trajectory_shape', self.trajectory_shape)) != tuple(self.trajectory_shape):
            print("[FeatureReplayBuffer] Trajectory-shape mismatch")
            return False
        saved_signature = state.get('training_signature')
        if self.training_signature is not None and saved_signature != self.training_signature:
            print(
                f"[FeatureReplayBuffer] Training-signature mismatch: found {saved_signature}, "
                f"expected {self.training_signature}"
            )
            return False
        ptr = int(state.get('ptr', 0))
        size = int(state.get('size', 0))
        if size < 0 or size > self.capacity:
            return False
        if ptr < 0 or ptr >= self.capacity:
            return False
        self.ptr = ptr
        self.size = size
        print(f"[FeatureReplayBuffer] State loaded: ptr={self.ptr}, size={self.size}")
        return True

    def save_state(self, filepath: str):
        np.save(filepath, self.state_dict())
        print(f"[FeatureReplayBuffer] State saved: ptr={self.ptr}, size={self.size}")

    def load_state(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
            return False
        try:
            state = np.load(filepath, allow_pickle=True).item()
            return self.load_state_dict(state)
        except Exception as e:
            print(f"[FeatureReplayBuffer] Failed to load state: {e}")
            return False

    def add(
        self,
        observation: Dict[str, np.ndarray],
        actor_base_features: np.ndarray,
        critic_bev_features: np.ndarray,
        trajectory: np.ndarray,
        pid_summary: np.ndarray,
        reward: float,
        next_observation: Dict[str, np.ndarray],
        next_critic_bev_features: np.ndarray,
        done: bool,
        prev_pid_summary: Optional[np.ndarray] = None,
        plan_cls_context: Optional[np.ndarray] = None,
        all_candidates: Optional[np.ndarray] = None,
        reference_logits: Optional[np.ndarray] = None,
        longitudinal_trajectory: Optional[np.ndarray] = None,
        candidate_longitudinal_trajectories: Optional[np.ndarray] = None,
        selected_lateral_mode: Optional[int] = None,
        longitudinal_mode: Optional[int] = None,
    ):
        self.states[self.ptr] = observation['state']
        self.target_points[self.ptr] = observation.get('target_point', np.zeros(2, dtype=np.float32))
        self.commands[self.ptr] = int(observation.get('command', 3))
        self.actor_base_features[self.ptr] = actor_base_features
        self.critic_bev_features[self.ptr] = critic_bev_features
        self.trajectories[self.ptr] = trajectory
        if self.longitudinal_trajectories is not None:
            if longitudinal_trajectory is None:
                raise ValueError("clean dual-trajectory replay requires longitudinal_trajectory")
            if candidate_longitudinal_trajectories is None:
                raise ValueError("clean dual-trajectory replay requires mode-aligned longitudinal candidates")
            self.longitudinal_trajectories[self.ptr] = longitudinal_trajectory
            self.candidate_longitudinal_trajectories[self.ptr] = np.asarray(
                candidate_longitudinal_trajectories,
                dtype=np.float16,
            )
            if selected_lateral_mode is None or longitudinal_mode is None:
                raise ValueError("clean dual-trajectory replay requires both planning mode indices")
            self.selected_lateral_modes[self.ptr] = int(selected_lateral_mode)
            self.longitudinal_modes[self.ptr] = int(longitudinal_mode)
        self.pid_summaries[self.ptr] = pid_summary
        if prev_pid_summary is not None:
            self.prev_pid_summaries[self.ptr] = prev_pid_summary
            self.prev_pid_summary_masks[self.ptr] = 1.0
        else:
            self.prev_pid_summaries[self.ptr] = np.zeros(self.pid_summary_dim, dtype=np.float32)
            self.prev_pid_summary_masks[self.ptr] = 0.0
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_observation['state']
        self.next_target_points[self.ptr] = next_observation.get('target_point', np.zeros(2, dtype=np.float32))
        self.next_commands[self.ptr] = int(next_observation.get('command', 3))
        self.next_critic_bev_features[self.ptr] = next_critic_bev_features
        self.dones[self.ptr] = float(done)
        if plan_cls_context is not None:
            self.plan_cls_context[self.ptr] = plan_cls_context.astype(np.float16)
        if all_candidates is not None:
            self.all_candidates[self.ptr] = all_candidates.astype(np.float16)
        if reference_logits is not None:
            self.reference_logits[self.ptr] = reference_logits.astype(np.float32)

        if self.ptr % 100 == 0:
            self._flush()
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _flush(self):
        self.states.flush()
        self.target_points.flush()
        self.commands.flush()
        self.actor_base_features.flush()
        self.critic_bev_features.flush()
        self.prev_pid_summaries.flush()
        self.prev_pid_summary_masks.flush()
        self.trajectories.flush()
        if self.longitudinal_trajectories is not None:
            self.longitudinal_trajectories.flush()
        self.pid_summaries.flush()
        if self.selected_lateral_modes is not None:
            self.selected_lateral_modes.flush()
            self.longitudinal_modes.flush()
        self.rewards.flush()
        self.next_states.flush()
        self.next_target_points.flush()
        self.next_commands.flush()
        self.next_critic_bev_features.flush()
        self.dones.flush()
        self.plan_cls_context.flush()
        self.all_candidates.flush()
        if self.candidate_longitudinal_trajectories is not None:
            self.candidate_longitudinal_trajectories.flush()
        self.reference_logits.flush()

    def sample(self, batch_size: int, device: str = 'cuda') -> 'FeatureBatch':
        indices = np.random.randint(0, self.size, size=batch_size)
        return FeatureBatch(
            observations={
                'state': _tensor_from_storage(self.states, indices, device, torch.float32),
                'target_point': _tensor_from_storage(self.target_points, indices, device, torch.float32),
                'command': _tensor_from_storage(self.commands, indices, device, torch.long),
            },
            actor_base_features=_tensor_from_storage(self.actor_base_features, indices, device, torch.float32),
            critic_bev_features=_tensor_from_storage(self.critic_bev_features, indices, device, torch.float32),
            prev_pid_summaries=_tensor_from_storage(self.prev_pid_summaries, indices, device, torch.float32),
            prev_pid_summary_masks=_tensor_from_storage(self.prev_pid_summary_masks, indices, device, torch.float32),
            trajectories=_tensor_from_storage(self.trajectories, indices, device, torch.float32),
            pid_summaries=_tensor_from_storage(self.pid_summaries, indices, device, torch.float32),
            rewards=_column_tensor_from_storage(self.rewards, indices, device, torch.float32),
            next_observations={
                'state': _tensor_from_storage(self.next_states, indices, device, torch.float32),
                'target_point': _tensor_from_storage(self.next_target_points, indices, device, torch.float32),
                'command': _tensor_from_storage(self.next_commands, indices, device, torch.long),
            },
            next_critic_bev_features=_tensor_from_storage(self.next_critic_bev_features, indices, device, torch.float32),
            dones=_column_tensor_from_storage(self.dones, indices, device, torch.float32),
            plan_cls_context=_tensor_from_storage(self.plan_cls_context, indices, device, torch.float32),
            all_candidates=_tensor_from_storage(self.all_candidates, indices, device, torch.float32),
            reference_logits=_tensor_from_storage(self.reference_logits, indices, device, torch.float32),
            longitudinal_trajectories=(
                _tensor_from_storage(self.longitudinal_trajectories, indices, device, torch.float32)
                if self.longitudinal_trajectories is not None
                else None
            ),
            candidate_longitudinal_trajectories=(
                _tensor_from_storage(
                    self.candidate_longitudinal_trajectories,
                    indices,
                    device,
                    torch.float32,
                )
                if self.candidate_longitudinal_trajectories is not None
                else None
            ),
            selected_lateral_modes=(
                _tensor_from_storage(self.selected_lateral_modes, indices, device, torch.long)
                if self.selected_lateral_modes is not None
                else None
            ),
            longitudinal_modes=(
                _tensor_from_storage(self.longitudinal_modes, indices, device, torch.long)
                if self.longitudinal_modes is not None
                else None
            ),
        )

    def __len__(self) -> int:
        return self.size

    def close(self):
        self._flush()
        print(f"[FeatureReplayBuffer] Closed. Data remains at: {self.mmap_dir}")

    def cleanup(self):
        self.close()
        import shutil
        if self.mmap_dir.exists():
            shutil.rmtree(self.mmap_dir)
            print(f"[FeatureReplayBuffer] Cleaned up: {self.mmap_dir}")


class Batch(NamedTuple):
    """Batch of transitions."""
    observations: Dict[str, torch.Tensor]
    prev_states: torch.Tensor
    prev_state_masks: torch.Tensor
    prev_can_buses: torch.Tensor
    prev_can_bus_masks: torch.Tensor
    pid_summaries: torch.Tensor
    prev_pid_summaries: torch.Tensor
    prev_pid_summary_masks: torch.Tensor
    trajectories: torch.Tensor  # [B, fut_ts, 2] trajectory waypoints
    prev_bevs: torch.Tensor  # [B, prev_bev_shape] BEV features for temporal modeling
    prev_bev_masks: torch.Tensor
    rewards: torch.Tensor
    next_observations: Dict[str, torch.Tensor]
    dones: torch.Tensor


class FeatureBatch(NamedTuple):
    """Batch of cached frozen-feature transitions (with off-policy actor fields)."""
    observations: Dict[str, torch.Tensor]
    actor_base_features: torch.Tensor
    critic_bev_features: torch.Tensor
    prev_pid_summaries: torch.Tensor
    prev_pid_summary_masks: torch.Tensor
    trajectories: torch.Tensor
    pid_summaries: torch.Tensor
    rewards: torch.Tensor
    next_observations: Dict[str, torch.Tensor]
    next_critic_bev_features: torch.Tensor
    dones: torch.Tensor
    plan_cls_context: Optional[torch.Tensor] = None   # [B, 48, 256]
    all_candidates: Optional[torch.Tensor] = None      # [B, 48, 6, 2]
    reference_logits: Optional[torch.Tensor] = None    # [B, 48]
    longitudinal_trajectories: Optional[torch.Tensor] = None  # [B, 6, 2], clean dual PID only
    candidate_longitudinal_trajectories: Optional[torch.Tensor] = None  # [B, 48, 6, 2]
    selected_lateral_modes: Optional[torch.Tensor] = None
    longitudinal_modes: Optional[torch.Tensor] = None
