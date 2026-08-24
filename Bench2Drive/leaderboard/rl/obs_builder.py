"""
Observation Builder for Bench2Drive RL
======================================

Builds observations from CARLA world state.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import carla
import cv2


@dataclass
class ObservationConfig:
    """Configuration for observation building."""
    # Match VAD config: img_scale=(1600, 900)
    image_width: int = 1600
    image_height: int = 900
    num_cameras: int = 6
    use_depth: bool = False
    use_semantic: bool = False


class ObservationBuilder:
    """Builds observations from CARLA world state."""

    def __init__(self, config: ObservationConfig):
        self.config = config
        self._camera_sensors: List[carla.Sensor] = []
        self._sensor_data: Dict[str, np.ndarray] = {}
        self._sensor_packets: Dict[int, Dict[str, np.ndarray]] = {}
        self._sensor_generation = 0
        self._expected_sensor_names = set()
        self._required_sensor_names = set()
        self._optional_sensor_names = set()
        self._fresh_sensor_names = set()
        self._sensor_frames: Dict[str, int] = {}
        self._max_packet_history = 4

    def setup_sensors(self, world: carla.World, vehicle: carla.Vehicle):
        """Setup camera sensors on vehicle."""
        self._sensor_generation += 1
        generation = self._sensor_generation
        self._sensor_data.clear()
        self._sensor_packets.clear()
        self._fresh_sensor_names.clear()
        self._sensor_frames.clear()

        # Camera blueprint
        blueprint_library = world.get_blueprint_library()

        # Match vad_b2d_agent.py camera setup exactly.
        camera_configs = [
            ('front', carla.Transform(carla.Location(x=0.80, y=0.0, z=1.60), carla.Rotation(yaw=0.0)), 70.0),
            ('front_left', carla.Transform(carla.Location(x=0.27, y=-0.55, z=1.60), carla.Rotation(yaw=-55.0)), 70.0),
            ('front_right', carla.Transform(carla.Location(x=0.27, y=0.55, z=1.60), carla.Rotation(yaw=55.0)), 70.0),
            ('rear', carla.Transform(carla.Location(x=-2.0, y=0.0, z=1.60), carla.Rotation(yaw=180.0)), 110.0),
            ('rear_left', carla.Transform(carla.Location(x=-0.32, y=-0.55, z=1.60), carla.Rotation(yaw=-110.0)), 70.0),
            ('rear_right', carla.Transform(carla.Location(x=-0.32, y=0.55, z=1.60), carla.Rotation(yaw=110.0)), 70.0),
        ]
        self._required_sensor_names = {name for name, _, _ in camera_configs} | {'bev', 'gps'}
        self._optional_sensor_names = {'imu'}
        self._expected_sensor_names = self._required_sensor_names | self._optional_sensor_names

        # Add BEV camera (top-down view like vad_b2d_agent_visualize.py)
        bev_bp = blueprint_library.find('sensor.camera.rgb')
        bev_bp.set_attribute('image_size_x', '512')
        bev_bp.set_attribute('image_size_y', '512')
        bev_bp.set_attribute('fov', '50')  # 5 * 10.0
        bev_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=50.0), carla.Rotation(pitch=-90.0))
        bev_camera = world.spawn_actor(bev_bp, bev_transform, attach_to=vehicle)
        bev_camera.listen(lambda image, generation=generation: self._on_image(image, 'bev', generation))
        self._camera_sensors.append(bev_camera)

        for name, transform, fov in camera_configs:
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(self.config.image_width))
            camera_bp.set_attribute('image_size_y', str(self.config.image_height))
            camera_bp.set_attribute('fov', str(fov))
            camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
            camera.listen(lambda image, name=name, generation=generation: self._on_image(image, name, generation))
            self._camera_sensors.append(camera)

        imu_bp = blueprint_library.find('sensor.other.imu')
        imu_transform = carla.Transform(carla.Location(x=-1.4, y=0.0, z=0.0))
        imu_bp.set_attribute('sensor_tick', '0.05')
        imu_sensor = world.spawn_actor(imu_bp, imu_transform, attach_to=vehicle)
        imu_sensor.listen(lambda imu_data, generation=generation: self._on_imu(imu_data, generation))
        self._camera_sensors.append(imu_sensor)

        gps_bp = blueprint_library.find('sensor.other.gnss')
        gps_bp.set_attribute('sensor_tick', '0.01')
        gps_transform = carla.Transform(carla.Location(x=-1.4, y=0.0, z=0.0))
        gps_sensor = world.spawn_actor(gps_bp, gps_transform, attach_to=vehicle)
        gps_sensor.listen(lambda gps_data, generation=generation: self._on_gnss(gps_data, generation))
        self._camera_sensors.append(gps_sensor)

    def _on_image(self, image: carla.Image, name: str, generation: int):
        """Callback for camera images."""
        if generation != self._sensor_generation:
            return
        frame = int(getattr(image, 'frame', -1))
        # Convert to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # BGRA
        array = array[:, :, :3]  # BGR
        self._store_sensor_packet(frame, f'raw_{name}', array.copy())
        if name != 'bev':
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
            ok, encoded = cv2.imencode('.jpg', array, encode_param)
            if ok:
                array = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        else:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        self._sensor_data[name] = array
        self._fresh_sensor_names.add(name)
        self._sensor_frames[name] = frame
        self._store_sensor_packet(frame, name, array)

    def _on_imu(self, imu_data: carla.IMUMeasurement, generation: int):
        """Callback for IMU readings."""
        if generation != self._sensor_generation:
            return
        frame = int(getattr(imu_data, 'frame', -1))
        imu_array = np.array([
            imu_data.accelerometer.x,
            imu_data.accelerometer.y,
            imu_data.accelerometer.z,
            imu_data.gyroscope.x,
            imu_data.gyroscope.y,
            imu_data.gyroscope.z,
            imu_data.compass,
        ], dtype=np.float32)
        self._sensor_data['imu'] = imu_array
        self._fresh_sensor_names.add('imu')
        self._sensor_frames['imu'] = frame
        self._store_sensor_packet(frame, 'imu', imu_array)

    def _on_gnss(self, gps_data: carla.GnssMeasurement, generation: int):
        """Callback for GNSS readings."""
        if generation != self._sensor_generation:
            return
        frame = int(getattr(gps_data, 'frame', -1))
        gps_array = np.array([
            gps_data.latitude,
            gps_data.longitude,
            gps_data.altitude,
        ], dtype=np.float32)
        self._sensor_data['gps'] = gps_array
        self._fresh_sensor_names.add('gps')
        self._sensor_frames['gps'] = frame
        self._store_sensor_packet(frame, 'gps', gps_array)

    def _store_sensor_packet(self, frame: int, name: str, value: np.ndarray):
        """Store one sensor payload inside the per-frame packet cache."""
        if frame < 0:
            return
        packet = self._sensor_packets.setdefault(frame, {})
        packet[name] = value
        self._prune_sensor_packets(frame)

    def _prune_sensor_packets(self, latest_frame: int):
        """Keep only a short history of recent frame packets."""
        min_frame = latest_frame - self._max_packet_history
        stale_frames = [frame for frame in self._sensor_packets.keys() if frame < min_frame]
        for frame in stale_frames:
            self._sensor_packets.pop(frame, None)

    def all_sensors_ready(self, frame: Optional[int] = None) -> bool:
        """Whether all required sensors have produced a frame for the current generation."""
        if frame is None:
            return any(
                self._required_sensor_names.issubset(packet.keys())
                for packet in self._sensor_packets.values()
            )
        packet = self._sensor_packets.get(frame)
        return packet is not None and self._required_sensor_names.issubset(packet.keys())

    def get_latest_complete_frame(self, max_frame: Optional[int] = None) -> Optional[int]:
        """Return the newest complete required-sensor packet, optionally capped by frame."""
        complete_frames = [
            sensor_frame
            for sensor_frame, packet in self._sensor_packets.items()
            if self._required_sensor_names.issubset(packet.keys()) and
            (max_frame is None or sensor_frame <= max_frame)
        ]
        if not complete_frames:
            return None
        return int(max(complete_frames))

    def get_sensor_packet_debug_info(self, frame: int) -> Dict[str, object]:
        """Debug information for an expected packet at a specific frame."""
        packet = self._sensor_packets.get(frame, {})
        available = sorted(packet.keys())
        missing_required = sorted(self._required_sensor_names.difference(packet.keys()))
        missing_optional = sorted(self._optional_sensor_names.difference(packet.keys()))
        complete_frames = sorted(
            sensor_frame
            for sensor_frame, frame_packet in self._sensor_packets.items()
            if self._required_sensor_names.issubset(frame_packet.keys())
        )
        return {
            'frame': int(frame),
            'available_sensors': available,
            'missing_sensors': missing_required + missing_optional,
            'missing_required_sensors': missing_required,
            'missing_optional_sensors': missing_optional,
            'latest_sensor_frames': dict(self._sensor_frames),
            'complete_frames_tail': complete_frames[-5:],
        }

    def get_sensor_packet(self, frame: int) -> Dict[str, np.ndarray]:
        """Return one cached frame packet, including raw image payloads when present."""
        packet = self._sensor_packets.get(frame)
        if packet is None:
            raise RuntimeError(f'No sensor packet cached for frame {frame}')
        return packet

    def get_observation(self, vehicle: carla.Vehicle, frame: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Get current observation."""
        if frame is None:
            ready_frames = [
                sensor_frame
                for sensor_frame, packet in self._sensor_packets.items()
                if self._required_sensor_names.issubset(packet.keys())
            ]
            if not ready_frames:
                raise RuntimeError('No complete sensor packet available')
            frame = max(ready_frames)

        packet = self._sensor_packets.get(frame)
        if packet is None or not self._required_sensor_names.issubset(packet.keys()):
            raise RuntimeError(f'Incomplete sensor packet for frame {frame}')

        # Get vehicle state
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        control = vehicle.get_control()
        imu = packet.get('imu')

        if imu is not None:
            acceleration = imu[:3].copy()
            angular_velocity = imu[3:6].copy()
            compass = float(imu[6])
        else:
            veh_ang = vehicle.get_angular_velocity()
            veh_acc = vehicle.get_acceleration()
            acceleration = np.array([veh_acc.x, veh_acc.y, veh_acc.z], dtype=np.float32)
            angular_velocity = np.array([veh_ang.x, veh_ang.y, veh_ang.z], dtype=np.float32)
            compass = float(np.pi / 2 - np.radians(transform.rotation.yaw))

        if np.isnan(compass):
            compass = 0.0
            acceleration = np.zeros(3, dtype=np.float32)
            angular_velocity = np.zeros(3, dtype=np.float32)

        # State vector: [speed, steer, throttle, brake, vx, vy, vz, ax, ay, az, ...]
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
        state = np.array([
            speed,
            control.steer,
            control.throttle,
            control.brake,
            velocity.x,
            velocity.y,
            velocity.z,
            angular_velocity[0],
            angular_velocity[1],
            angular_velocity[2],
            acceleration[0],
            acceleration[1],
            acceleration[2],
            transform.location.x,
            transform.location.y,
            transform.location.z,
            transform.rotation.roll,
            transform.rotation.pitch,
            transform.rotation.yaw,
        ], dtype=np.float32)

        # Pad to 21 dimensions if needed
        if len(state) < 21:
            state = np.pad(state, (0, 21 - len(state)), mode='constant')
        else:
            state = state[:21]

        # Stack camera images
        rgb_images = []
        for name in ['front', 'front_left', 'front_right', 'rear', 'rear_left', 'rear_right']:
            rgb_images.append(packet[name])

        rgb = np.stack(rgb_images, axis=0)  # [6, H, W, 3]

        bev = packet['bev']
        gps = packet['gps']

        ego_theta = -compass + np.pi / 2
        can_bus = np.zeros(18, dtype=np.float32)
        can_bus[0] = transform.location.x
        can_bus[1] = -transform.location.y
        # Match vad_b2d_agent / hipad_b2d_agent raw can_bus semantics:
        # the z slot is left at 0 and is not used for lidar2global translation.
        can_bus[2] = 0.0
        can_bus[3] = np.cos(ego_theta / 2)
        can_bus[4] = 0.0
        can_bus[5] = 0.0
        can_bus[6] = np.sin(ego_theta / 2)
        can_bus[7] = speed
        can_bus[10:13] = acceleration
        can_bus[11] *= -1
        can_bus[13:16] = -angular_velocity
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180

        return {
            'rgb': rgb,
            'state': state,
            'bev': bev,
            'gps': gps,
            'compass': np.float32(compass),
            'can_bus': can_bus,
            'sensor_frame': np.int64(frame),
        }

    def cleanup(self):
        """Cleanup sensors."""
        self._sensor_generation += 1
        for sensor in self._camera_sensors:
            try:
                if sensor.is_alive:
                    sensor.stop()
                    sensor.destroy()
            except Exception:
                # Sensor may already be destroyed or world disconnected
                pass
        self._camera_sensors.clear()
        self._sensor_data.clear()
        self._sensor_packets.clear()
        self._expected_sensor_names.clear()
        self._required_sensor_names.clear()
        self._optional_sensor_names.clear()
        self._fresh_sensor_names.clear()
