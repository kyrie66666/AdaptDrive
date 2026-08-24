"""Stable Bench2Drive environment used by AdaptDrive training and evaluation."""

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from rl.env import Bench2DriveSACEnv
from rl.sim_backend import find_free_port


class _TrafficManagerClientProxy:
    """Always reuse the active experiment TrafficManager instead of spawning new ones."""

    def __init__(self, client, tm_port: int, traffic_manager):
        self._client = client
        self._tm_port = int(tm_port)
        self._traffic_manager = traffic_manager

    def update(self, tm_port: int, traffic_manager) -> None:
        self._tm_port = int(tm_port)
        self._traffic_manager = traffic_manager

    def get_trafficmanager(self, port):
        if self._traffic_manager is not None:
            return self._traffic_manager
        return self._client.get_trafficmanager(port)

    def __getattr__(self, name):
        return getattr(self._client, name)


class AdaptDriveBench2DriveSACEnv(Bench2DriveSACEnv):
    """Bench2Drive env variant that rotates the TrafficManager port per reset."""

    def __init__(self, config):
        self._client_proxy = None
        self._install_carla_data_provider_proxy()
        super().__init__(config)

    def _install_carla_data_provider_proxy(self) -> None:
        if getattr(CarlaDataProvider, '_adaptdrive_proxy_installed', False):
            return

        original_set_client = CarlaDataProvider.set_client
        original_get_client = CarlaDataProvider.get_client

        def set_client_proxy(client, _orig=original_set_client):
            proxy = getattr(AdaptDriveBench2DriveSACEnv, '_active_client_proxy', None)
            if proxy is not None and client is getattr(proxy, '_client', None):
                return _orig(proxy)
            return _orig(client)

        def get_client_proxy(_orig=original_get_client):
            proxy = getattr(AdaptDriveBench2DriveSACEnv, '_active_client_proxy', None)
            client = _orig()
            if proxy is not None and client is getattr(proxy, '_client', None):
                return proxy
            return client

        CarlaDataProvider.set_client = staticmethod(set_client_proxy)
        CarlaDataProvider.get_client = staticmethod(get_client_proxy)
        CarlaDataProvider._adaptdrive_proxy_installed = True

    def _refresh_traffic_manager(self) -> None:
        if self.client is None:
            return

        current_port = int(getattr(self.config.simulation, 'traffic_manager_port', 8000))
        next_port = find_free_port(max(10000, current_port + 1))
        self.config.simulation.traffic_manager_port = next_port
        self._sim_backend.config.traffic_manager_port = next_port

        traffic_manager = self.client.get_trafficmanager(next_port)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_hybrid_physics_mode(True)
        seed = getattr(self.config.simulation, 'traffic_manager_seed', 0)
        traffic_manager.set_random_device_seed(seed)
        self.traffic_manager = traffic_manager

        self._sync_traffic_manager_proxy(next_port, traffic_manager)
        print(f"[AdaptDriveBench2DriveSACEnv] Using TrafficManager port {next_port}")

    def _sync_traffic_manager_proxy(self, tm_port: int, traffic_manager) -> None:
        if self._client_proxy is None:
            self._client_proxy = _TrafficManagerClientProxy(self.client, tm_port, traffic_manager)
        else:
            self._client_proxy.update(tm_port, traffic_manager)
        AdaptDriveBench2DriveSACEnv._active_client_proxy = self._client_proxy

        CarlaDataProvider.set_client(self.client)
        if self.world is not None:
            CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(tm_port)
        CarlaDataProvider.set_runtime_init_mode(False)

    def _load_world_for_route(self, town: str, max_retries: int = 2):
        super()._load_world_for_route(town, max_retries=max_retries)
        self.client = self._sim_backend.client
        self.traffic_manager = self._sim_backend.traffic_manager
        tm_port = int(self._sim_backend.config.traffic_manager_port)
        if self.traffic_manager is not None:
            self._sync_traffic_manager_proxy(tm_port, self.traffic_manager)
            print(f"[AdaptDriveBench2DriveSACEnv] Synced TrafficManager port {tm_port} after world load")

    def reset(self, seed=None, force_new_route: bool = False):
        self._refresh_traffic_manager()
        return super().reset(seed=seed, force_new_route=force_new_route)
