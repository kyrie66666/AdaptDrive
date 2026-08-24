"""Owned CARLA server lifecycle for the standalone Town-map generator."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Sequence

try:
    import pwd
except ImportError:  # pragma: no cover - Linux CARLA runtime only
    pwd = None


class OwnedCarlaServer:
    """Launch one CARLA process and stop only the process group we own."""

    def __init__(
        self,
        *,
        carla_module,
        carla_root: Path,
        host: str,
        port: int,
        startup_timeout_seconds: float,
        shutdown_timeout_seconds: float,
        server_log: Optional[Path] = None,
        cuda_visible_devices: str = "",
        graphics_adapter: Optional[int] = None,
        launch_user: str = "",
        runtime_dir: Optional[Path] = None,
        vk_icd_filenames: str = "",
        display: str = "",
        server_warmup_seconds: float = 30.0,
        extra_args: Sequence[str] = (),
    ) -> None:
        self.carla = carla_module
        self.carla_root = Path(carla_root).expanduser().resolve()
        self.host = str(host)
        self.port = int(port)
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.shutdown_timeout_seconds = float(shutdown_timeout_seconds)
        self.cuda_visible_devices = str(cuda_visible_devices)
        self.graphics_adapter = graphics_adapter
        self.launch_user = str(launch_user)
        self.runtime_dir = (
            Path(runtime_dir).expanduser().resolve()
            if runtime_dir is not None
            else Path(tempfile.gettempdir()) / f"carla-runtime-{self.port}"
        )
        self.vk_icd_filenames = str(vk_icd_filenames)
        self.display = str(display)
        self.server_warmup_seconds = float(server_warmup_seconds)
        self.extra_args = tuple(str(value) for value in extra_args)
        self.server_log = (
            Path(server_log).expanduser().resolve()
            if server_log is not None
            else Path(tempfile.gettempdir()) / f"roach_map_carla_{self.port}.log"
        )
        self.process: Optional[subprocess.Popen] = None
        self._log_handle = None
        self.client = None

    def _carla_command(self):
        script = self.carla_root / "CarlaUE4.sh"
        if not script.is_file():
            raise FileNotFoundError(f"CARLA launcher does not exist: {script}")
        command = [
            str(script),
            "-RenderOffScreen",
            "-nosound",
            f"-carla-rpc-port={self.port}",
        ]
        if self.graphics_adapter is not None:
            command.append(f"-graphicsadapter={int(self.graphics_adapter)}")
        command.extend(self.extra_args)
        return command

    def command(self):
        command = self._carla_command()
        if not self.launch_user:
            return command
        exports = [
            f"export XDG_RUNTIME_DIR={shlex.quote(str(self.runtime_dir))}",
            "export SDL_AUDIODRIVER=dummy",
        ]
        if self.cuda_visible_devices:
            exports.append(f"export CUDA_VISIBLE_DEVICES={shlex.quote(self.cuda_visible_devices)}")
        if self.display:
            exports.append(f"export DISPLAY={shlex.quote(self.display)}")
        # Do not auto-detect/inject an ICD. This container has a documented
        # VK_ERROR_DEVICE_LOST failure when the wrong ICD is forced.
        if self.vk_icd_filenames:
            exports.append(f"export VK_ICD_FILENAMES={shlex.quote(self.vk_icd_filenames)}")
        inner_command = "; ".join(exports + ["exec " + shlex.join(command)])
        use_login_su = os.environ.get("CARLA_SU_LOGIN", "1").lower() not in {"0", "false", "no"}
        return ["su", "-", self.launch_user, "-c", inner_command] if use_login_su else [
            "su",
            self.launch_user,
            "-c",
            inner_command,
        ]

    def _prepare_runtime_dir(self) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        if self.launch_user:
            if pwd is None:
                raise RuntimeError("pwd module is required for --carla-launch-user")
            try:
                user_info = pwd.getpwnam(self.launch_user)
            except KeyError as exc:
                raise RuntimeError(f"CARLA launch user does not exist: {self.launch_user}") from exc
            if os.getuid() == 0:
                os.chown(self.runtime_dir, user_info.pw_uid, user_info.pw_gid)
        self.runtime_dir.chmod(0o700)

    def start(self):
        if self.process is not None:
            raise RuntimeError("Owned CARLA server has already been started")
        if self.host not in {"127.0.0.1", "localhost"}:
            raise ValueError("Owned CARLA server requires --host 127.0.0.1 or localhost")

        environment = os.environ.copy()
        if self.cuda_visible_devices:
            environment["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        environment["XDG_RUNTIME_DIR"] = str(self.runtime_dir)
        environment["SDL_AUDIODRIVER"] = "dummy"
        if self.display:
            environment["DISPLAY"] = self.display
        if self.vk_icd_filenames:
            environment["VK_ICD_FILENAMES"] = self.vk_icd_filenames
        self._prepare_runtime_dir()
        self.server_log.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.server_log.open("w", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                self.command(),
                cwd=str(self.carla_root),
                env=environment,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            self._log_handle.close()
            self._log_handle = None
            raise
        try:
            self.client = self._wait_until_ready()
            if self.server_warmup_seconds > 0:
                time.sleep(self.server_warmup_seconds)
            return self.client
        except Exception:
            self.stop()
            raise

    def _wait_until_ready(self):
        deadline = time.monotonic() + self.startup_timeout_seconds
        last_error = None
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"Owned CARLA server exited with code {self.process.returncode}; "
                    f"inspect {self.server_log}"
                )
            try:
                client = self.carla.Client(self.host, self.port)
                client.set_timeout(2.0)
                client.get_server_version()
                client.set_timeout(max(10.0, self.startup_timeout_seconds))
                return client
            except Exception as exc:  # CARLA raises RuntimeError/TimeoutException variants
                last_error = exc
                time.sleep(1.0)
        raise TimeoutError(
            f"Owned CARLA server did not become ready within {self.startup_timeout_seconds:.1f}s; "
            f"last_error={last_error}; inspect {self.server_log}"
        )

    def stop(self) -> None:
        process = self.process
        self.process = None
        self.client = None
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=self.shutdown_timeout_seconds)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=max(1.0, self.shutdown_timeout_seconds))
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
