"""
Step Manager for Bench2Drive RL
===============================

Manages episode step loop and termination conditions.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import carla


@dataclass
class StepManagerConfig:
    """Configuration for step manager."""
    timeout: float = 600.0  # Episode timeout in seconds
    max_tick_count: int = 4000  # Maximum ticks per episode
    tick_interval: float = 0.05  # 20 Hz (0.05s per tick)


class StepManager:
    """Manages episode execution."""

    def __init__(self, config: StepManagerConfig):
        self.config = config
        self._tick_count = 0
        self._start_sim_time = None
        self._elapsed_time = 0.0

    def reset(self):
        """Reset step manager."""
        self._tick_count = 0
        self._start_sim_time = None
        self._elapsed_time = 0.0

    def step(self, world: carla.World) -> Tuple[bool, float, Optional[str]]:
        """
        Execute one step.

        Returns:
            truncated: bool
            elapsed_time: float
            truncation_reason: optional string reason when truncated
        """
        # Tick world
        # world.tick()  # This should be called by the environment

        self._tick_count += 1

        try:
            snapshot = world.get_snapshot()
            current_sim_time = float(snapshot.timestamp.elapsed_seconds)
        except Exception:
            snapshot = None
            current_sim_time = None

        if current_sim_time is not None:
            if self._start_sim_time is None:
                self._start_sim_time = current_sim_time
            self._elapsed_time = max(0.0, current_sim_time - self._start_sim_time)
        else:
            # Fallback to fixed delta if snapshots are temporarily unavailable.
            self._elapsed_time += float(self.config.tick_interval)

        # Check termination conditions
        if self._tick_count >= self.config.max_tick_count:
            return True, self._elapsed_time, "max_tick_count"

        if self._elapsed_time >= self.config.timeout:
            return True, self._elapsed_time, "timeout"

        return False, self._elapsed_time, None

    def should_terminate(self) -> bool:
        """Check if episode should terminate."""
        if self._tick_count >= self.config.max_tick_count:
            return True
        if self._elapsed_time >= self.config.timeout:
            return True
        return False

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def elapsed_time(self) -> float:
        return self._elapsed_time
