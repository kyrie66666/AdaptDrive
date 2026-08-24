#!/usr/bin/env python3
"""CARLA-free ownership/cleanup smoke for the map tool's server manager."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from roach_map.carla_server import OwnedCarlaServer  # noqa: E402


class FakeClient:
    def __init__(self, _host, _port):
        self.timeout = None

    def set_timeout(self, timeout):
        self.timeout = float(timeout)

    @staticmethod
    def get_server_version():
        return "fake-0.9.15"


class FakeCarla:
    Client = FakeClient


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="roach_owned_server_") as tmp_dir:
        root = Path(tmp_dir)
        launcher = root / "CarlaUE4.sh"
        launcher.write_text(
            "#!/bin/sh\n"
            "trap 'exit 0' TERM INT\n"
            "while true; do sleep 1; done\n",
            encoding="utf-8",
        )
        launcher.chmod(0o755)
        manager = OwnedCarlaServer(
            carla_module=FakeCarla,
            carla_root=root,
            host="127.0.0.1",
            port=29876,
            startup_timeout_seconds=2.0,
            shutdown_timeout_seconds=2.0,
            server_log=root / "fake_server.log",
            cuda_visible_devices="7",
            graphics_adapter=3,
            runtime_dir=root / "runtime",
            server_warmup_seconds=0.0,
            extra_args=("-fake-map-test",),
        )
        command = manager.command()
        assert command[0] == str(launcher)
        assert "-carla-rpc-port=29876" in command
        assert "-graphicsadapter=3" in command
        assert "-fake-map-test" in command

        container_manager = OwnedCarlaServer(
            carla_module=FakeCarla,
            carla_root=root,
            host="127.0.0.1",
            port=29877,
            startup_timeout_seconds=2.0,
            shutdown_timeout_seconds=2.0,
            server_log=root / "container_server.log",
            cuda_visible_devices="1",
            graphics_adapter=2,
            launch_user="carla",
            runtime_dir=root / "container-runtime",
            display=":99",
            vk_icd_filenames="",
            server_warmup_seconds=0.0,
        )
        container_command = container_manager.command()
        assert container_command[0] == "su"
        assert "carla" in container_command
        inner_command = container_command[-1]
        assert "XDG_RUNTIME_DIR=" in inner_command
        assert "CUDA_VISIBLE_DEVICES=1" in inner_command
        assert "DISPLAY=:99" in inner_command
        assert "-graphicsadapter=2" in inner_command
        assert "VK_ICD_FILENAMES" not in inner_command

        client = manager.start()
        process = manager.process
        assert client.get_server_version() == "fake-0.9.15"
        assert process is not None and process.poll() is None
        manager.stop()
        assert process.poll() is not None
        assert manager.process is None
    print("Roach owned CARLA server smoke passed")


if __name__ == "__main__":
    main()
