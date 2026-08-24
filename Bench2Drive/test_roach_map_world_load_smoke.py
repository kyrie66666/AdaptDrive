#!/usr/bin/env python3
"""CARLA-free smoke for verified sequential Town switching."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = PROJECT_ROOT / "Bench2Drive/tools/generate_roach_static_maps.py"


class FakeMap:
    def __init__(self, name: str):
        self.name = f"Carla/Maps/{name}"


class FakeWorld:
    def __init__(self, town: str):
        self._map = FakeMap(town)

    def get_map(self):
        return self._map


class FakeClient:
    def __init__(self):
        self.load_attempts = 0
        self.current_world = FakeWorld("Town07")

    def load_world(self, town: str, reset_settings: bool = False):
        assert reset_settings is False
        self.load_attempts += 1
        if self.load_attempts == 1:
            return self.current_world
        self.current_world = FakeWorld(town)
        return self.current_world

    def get_world(self):
        return self.current_world

    @staticmethod
    def get_available_maps():
        return ["/Game/Carla/Maps/Town07", "/Game/Carla/Maps/Town10HD"]


def _load_generator_module():
    spec = importlib.util.spec_from_file_location("roach_static_map_generator", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load generator module from {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    generator = _load_generator_module()
    args = SimpleNamespace(world_load_attempts=2, world_load_settle_seconds=1e-9)
    client = FakeClient()
    world = generator._load_world_verified(client, args, "Town10HD")
    assert client.load_attempts == 2
    assert generator._normalize_town_name(world.get_map().name) == "Town10HD"
    print("Roach sequential Town load smoke passed")


if __name__ == "__main__":
    main()
