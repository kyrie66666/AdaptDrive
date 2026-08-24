"""
Simulation Backend for Bench2Drive RL
=====================================

Shared CARLA bootstrap/runtime helpers for the current RL trainers and
evaluation entrypoints.
"""

import atexit
import os
import pwd
import shlex
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


DEFAULT_FRAME_RATE = 20.0
DEFAULT_CLIENT_TIMEOUT = 300.0


def find_free_port(starting_port: int) -> int:
    port = int(starting_port)
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("localhost", port))
                return port
        except OSError:
            port += 1


@dataclass
class SimulationConfig:
    host: str = "localhost"
    port: int = 2000
    traffic_manager_port: int = 8000
    traffic_manager_seed: int = 0
    timeout: float = DEFAULT_CLIENT_TIMEOUT
    frame_rate: float = DEFAULT_FRAME_RATE
    gpu_rank: int = 0
    launch_server: bool = True
    carla_root: str = ""
    save_path: str = "."
    runtime_dir: str = ""
    vk_icd_filenames: str = ""
    egl_vendor_library_filenames: str = ""
    launch_user: str = ""
    render_offscreen: bool = True
    server_warmup_seconds: float = 8.0
    client_retries: int = 20
    traffic_manager_retries: int = 40
    retry_interval_seconds: float = 5.0
    bootstrap_timeout_cap: float = 60.0
    tile_stream_distance: float = 650.0
    actor_active_distance: float = 650.0
    deterministic_ragdolls: bool = True
    spectator_as_ego: bool = False

    def resolve_carla_root(self) -> str:
        root = self.carla_root or os.environ.get("CARLA_ROOT", "")
        if not root:
            raise RuntimeError("CARLA_ROOT must be set to launch the CARLA server")
        return root

    def resolve_save_path(self) -> str:
        return self.save_path or os.environ.get("SAVE_PATH", ".")

    def resolve_runtime_dir(self) -> str:
        return self.runtime_dir or os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/carla-runtime-{self.port}"

    def resolve_vk_icd_filenames(self) -> str:
        return self.vk_icd_filenames

    def resolve_egl_vendor_library_filenames(self) -> str:
        return (
            self.egl_vendor_library_filenames
            or os.environ.get("CARLA_EGL_VENDOR_LIBRARY_FILENAMES", "")
            or os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES", "")
        )

    def resolve_launch_user(self) -> str:
        return self.launch_user or os.environ.get("CARLA_LAUNCH_USER", "")


@dataclass
class WorldLoadResult:
    world: carla.World
    town: str


class SimulationBackend:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.client: Optional[carla.Client] = None
        self.client_timeout: float = config.timeout
        self.traffic_manager = None
        self.world: Optional[carla.World] = None
        self.server_process: Optional[subprocess.Popen] = None
        self._carla_log_path: Optional[str] = None
        self._registered_atexit = False

    def start(self):
        if self.client is not None and self.traffic_manager is not None:
            return self.client, self.client_timeout, self.traffic_manager

        if self.config.launch_server:
            self._launch_server()
            self._raise_if_server_exited("CARLA server exited during startup")

        self.client_timeout = self.config.timeout or DEFAULT_CLIENT_TIMEOUT
        bootstrap_timeout = min(self.client_timeout, self.config.bootstrap_timeout_cap)

        client = None
        for attempt in range(self.config.client_retries):
            try:
                client = carla.Client(self.config.host, self.config.port)
                client.set_timeout(bootstrap_timeout)
                world = client.get_world()
                self._apply_sync_settings(world)
                client.set_timeout(self.client_timeout)
                self.world = world
                break
            except Exception as exc:
                self._raise_if_server_exited("CARLA server exited while waiting for the client")
                print(
                    f"[SimulationBackend] Client connection failed "
                    f"(attempt {attempt + 1}/{self.config.client_retries}): {exc}",
                    flush=True,
                )
                if attempt + 1 >= self.config.client_retries:
                    raise
                time.sleep(self.config.retry_interval_seconds)

        self.client = client
        self._connect_traffic_manager(self.config.traffic_manager_port)
        return self.client, self.client_timeout, self.traffic_manager

    def _connect_traffic_manager(self, starting_port: Optional[int] = None):
        """(Re)connect to a usable TrafficManager, bumping the port if needed."""
        if self.client is None:
            raise RuntimeError("CARLA client is not initialized")

        port = int(starting_port if starting_port is not None else self.config.traffic_manager_port)
        last_error = None

        for attempt in range(self.config.traffic_manager_retries):
            try:
                candidate_port = find_free_port(port)
                traffic_manager = self.client.get_trafficmanager(candidate_port)
                traffic_manager.set_synchronous_mode(True)
                traffic_manager.set_hybrid_physics_mode(True)
                self.config.traffic_manager_port = candidate_port
                self.traffic_manager = traffic_manager
                return traffic_manager
            except Exception as exc:
                last_error = exc
                print(
                    f"[SimulationBackend] TrafficManager init failed "
                    f"(attempt {attempt + 1}/{self.config.traffic_manager_retries}, "
                    f"port_start={port}): {exc}",
                    flush=True,
                )
                port += 1
                if attempt + 1 >= self.config.traffic_manager_retries:
                    raise
                time.sleep(self.config.retry_interval_seconds)

        if last_error is not None:
            raise last_error

    def load_world(self, town: str, max_retries: int = 3) -> WorldLoadResult:
        if self.client is None:
            self.start()

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Timeout loading world {town}")

        load_timeout = max(1, int(float(os.environ.get("CARLA_WORLD_LOAD_TIMEOUT", "600"))))

        for attempt in range(max_retries):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(load_timeout)

            try:
                print(f"[SimulationBackend] Loading world for {town}... (attempt {attempt + 1}/{max_retries})")
                self.world = self.client.load_world(town, reset_settings=False)
                print("[SimulationBackend] World loaded successfully")
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                break
            except TimeoutError as exc:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                print(f"[SimulationBackend] WARNING: {exc}")
                if attempt + 1 >= max_retries:
                    raise TimeoutError(f"Failed to load world {town} after {max_retries} attempts")
                print("[SimulationBackend] Retrying in 5 seconds...")
                time.sleep(5)
                try:
                    self.client = carla.Client(self.config.host, self.config.port)
                    self.client.set_timeout(self.client_timeout)
                except Exception as reconnect_err:
                    print(f"[SimulationBackend] WARNING: Reconnect failed: {reconnect_err}")
            except Exception:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                raise

        settings = self.world.get_settings()
        settings.tile_stream_distance = self.config.tile_stream_distance
        settings.actor_active_distance = self.config.actor_active_distance
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.config.frame_rate
        settings.deterministic_ragdolls = self.config.deterministic_ragdolls
        settings.spectator_as_ego = self.config.spectator_as_ego
        self.world.apply_settings(settings)

        self._connect_traffic_manager(self.config.traffic_manager_port)
        self.world.reset_all_traffic_lights()
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(self.config.traffic_manager_port)
        CarlaDataProvider.set_runtime_init_mode(False)

        self.traffic_manager.set_random_device_seed(self.config.traffic_manager_seed)
        self.world.tick()

        map_name = CarlaDataProvider.get_map().name.split("/")[-1]
        if map_name != town:
            raise RuntimeError(
                f"The CARLA server uses the wrong map. This scenario requires the use of map {town}"
            )

        return WorldLoadResult(world=self.world, town=town)

    def reset_world_settings(self, client_timed_out: bool = False):
        if self.world is None or client_timed_out:
            return

        try:
            self.world.tick()
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            settings.deterministic_ragdolls = False
            settings.spectator_as_ego = True
            self.world.apply_settings(settings)
        except Exception:
            pass

        if self.traffic_manager is not None:
            try:
                self.traffic_manager.set_synchronous_mode(False)
                self.traffic_manager.set_hybrid_physics_mode(False)
            except Exception:
                pass

    def stop_server(self):
        if self.server_process is None:
            return

        if self.server_process.poll() is None:
            try:
                os.killpg(self.server_process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.server_process = None

    def _read_carla_log_tail(self, max_lines: int = 80) -> str:
        if not self._carla_log_path or not os.path.isfile(self._carla_log_path):
            return ""
        try:
            with open(self._carla_log_path, "r", errors="replace") as log_file:
                lines = log_file.readlines()
        except OSError:
            return ""
        return "".join(lines[-max_lines:]).rstrip()

    def _raise_if_server_exited(self, message: str):
        if self.server_process is None:
            return
        return_code = self.server_process.poll()
        if return_code is None:
            return

        log_tail = self._read_carla_log_tail()
        detail = f"{message}; returncode={return_code}"
        if self._carla_log_path:
            detail += f"; log={self._carla_log_path}"
        if log_tail:
            detail += f"\n--- CARLA log tail ---\n{log_tail}\n--- end CARLA log tail ---"
        raise RuntimeError(detail)

    def close(self, reset_world_settings: bool = True, client_timed_out: bool = False, stop_server: bool = False):
        if reset_world_settings:
            self.reset_world_settings(client_timed_out=client_timed_out)

        self.world = None
        self.traffic_manager = None
        self.client = None

        if stop_server:
            self.stop_server()

    def _launch_server(self):
        carla_root = self.config.resolve_carla_root()
        self.config.port = find_free_port(self.config.port)
        save_path = self.config.resolve_save_path()
        runtime_dir = self.config.resolve_runtime_dir()
        vk_icd_filenames = self.config.resolve_vk_icd_filenames()
        egl_vendor_library_filenames = self.config.resolve_egl_vendor_library_filenames()
        launch_user = self.config.resolve_launch_user()
        fake_uid_path = os.path.join(carla_root, "fake_uid.so")

        os.makedirs(save_path, exist_ok=True)
        os.makedirs(runtime_dir, mode=0o700, exist_ok=True)
        if launch_user:
            try:
                user_info = pwd.getpwnam(launch_user)
                if os.getuid() == 0:
                    os.chown(runtime_dir, user_info.pw_uid, user_info.pw_gid)
            except KeyError:
                raise RuntimeError(f"CARLA launch user does not exist: {launch_user}")
        os.chmod(runtime_dir, 0o700)

        carla_log = os.path.join(save_path, f"carla_server_{self.config.port}.log")
        self._carla_log_path = carla_log
        launch_env = {
            "XDG_RUNTIME_DIR": runtime_dir,
            "SDL_AUDIODRIVER": "dummy",
            "CUDA_VISIBLE_DEVICES": str(self.config.gpu_rank),
            "DISPLAY": os.environ.get("DISPLAY", ":99"),
        }
        if vk_icd_filenames:
            launch_env["VK_ICD_FILENAMES"] = vk_icd_filenames
        if egl_vendor_library_filenames:
            launch_env["__EGL_VENDOR_LIBRARY_FILENAMES"] = egl_vendor_library_filenames
        existing_ld_preload = os.environ.get("LD_PRELOAD", "")
        if os.path.isfile(fake_uid_path):
            launch_env["LD_PRELOAD"] = ":".join(
                value for value in (fake_uid_path, existing_ld_preload) if value
            )
        elif existing_ld_preload:
            launch_env["LD_PRELOAD"] = existing_ld_preload
        render_flag = "-RenderOffScreen" if self.config.render_offscreen else ""

        carla_args = (
            f"{render_flag} -nosound -carla-rpc-port={self.config.port} "
            f"-graphicsadapter={self.config.gpu_rank}"
        ).strip()
        exports = "; ".join(
            f"export {key}={shlex.quote(value)}" for key, value in launch_env.items()
        )
        inner_command = (
            f"{exports}; "
            f"mkdir -p {shlex.quote(runtime_dir)}; "
            f"chmod 700 {shlex.quote(runtime_dir)}; "
            f"{shlex.quote(os.path.join(carla_root, 'CarlaUE4.sh'))} {carla_args}"
        )
        if launch_user:
            command = (
                f"su {shlex.quote(launch_user)} -c {shlex.quote(inner_command)} "
                f"> {shlex.quote(carla_log)} 2>&1"
            )
        else:
            command = f"{inner_command} > {shlex.quote(carla_log)} 2>&1"

        print(
            f"[SimulationBackend] Launching CARLA on port {self.config.port}, "
            f"gpu={self.config.gpu_rank}, user={launch_user or '<current>'}",
            flush=True,
        )
        print(f"[SimulationBackend] CARLA command: {command}", flush=True)
        print(f"[SimulationBackend] CARLA log: {carla_log}", flush=True)
        self.server_process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
        if not self._registered_atexit:
            atexit.register(self.stop_server)
            self._registered_atexit = True
        time.sleep(self.config.server_warmup_seconds)
        self._raise_if_server_exited("CARLA server exited during startup")

    def _apply_sync_settings(self, world: carla.World):
        settings = carla.WorldSettings(
            synchronous_mode=True,
            fixed_delta_seconds=1.0 / self.config.frame_rate,
            deterministic_ragdolls=self.config.deterministic_ragdolls,
            spectator_as_ego=self.config.spectator_as_ego,
        )
        world.apply_settings(settings)


def build_simulation_config_from_args(
    args,
    frame_rate: float = DEFAULT_FRAME_RATE,
    default_timeout: float = DEFAULT_CLIENT_TIMEOUT,
):
    return SimulationConfig(
        host=getattr(args, "host", "localhost"),
        port=int(getattr(args, "port", 2000)),
        traffic_manager_port=int(getattr(args, "traffic_manager_port", 8000)),
        traffic_manager_seed=int(getattr(args, "traffic_manager_seed", 0)),
        timeout=float(getattr(args, "timeout", default_timeout) or default_timeout),
        frame_rate=float(frame_rate),
        gpu_rank=int(getattr(args, "gpu_rank", 0)),
        launch_server=True,
        carla_root=os.environ.get("CARLA_ROOT", ""),
        save_path=os.environ.get("SAVE_PATH", "."),
        runtime_dir=os.environ.get("XDG_RUNTIME_DIR", ""),
        vk_icd_filenames=os.environ.get("VK_ICD_FILENAMES", ""),
        egl_vendor_library_filenames=(
            os.environ.get("CARLA_EGL_VENDOR_LIBRARY_FILENAMES", "")
            or os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES", "")
        ),
        launch_user=os.environ.get("CARLA_LAUNCH_USER", ""),
        server_warmup_seconds=float(getattr(args, "server_warmup_seconds", 8.0) or 8.0),
        bootstrap_timeout_cap=float(getattr(args, "bootstrap_timeout_cap", 60.0) or 60.0),
    )
