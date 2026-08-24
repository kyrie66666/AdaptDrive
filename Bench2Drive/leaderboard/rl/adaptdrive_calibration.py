"""Bench2Drive camera calibration used by the AdaptDrive HiP-AD runtime.

The module is deliberately lightweight: importing it does not initialize
CARLA, OpenMMLab, or any legacy VAD code.
"""

from __future__ import annotations

import numpy as np


CAMERA_NAMES = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

LIDAR2EGO = np.array(
    [[0.0, 1.0, 0.0, -0.39],
     [-1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 1.84],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float32,
)

LIDAR2IMG = {
    "CAM_FRONT": np.array(
        [[1.14251841e3, 8.0e2, 0.0, -9.52e2],
         [0.0, 4.5e2, -1.14251841e3, -8.09704417e2],
         [0.0, 1.0, 0.0, -1.19],
         [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    "CAM_FRONT_LEFT": np.array(
        [[6.03961325e-14, 1.39475744e3, 0.0, -9.20539908e2],
         [-3.68618420e2, 2.58109396e2, -1.14251841e3, -6.47296750e2],
         [-8.19152044e-1, 5.73576436e-1, 0.0, -8.29094072e-1],
         [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    "CAM_FRONT_RIGHT": np.array(
        [[1.31064327e3, -4.77035138e2, 0.0, -4.06010608e2],
         [3.68618420e2, 2.58109396e2, -1.14251841e3, -6.47296750e2],
         [8.19152044e-1, 5.73576436e-1, 0.0, -8.29094072e-1],
         [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    "CAM_BACK": np.array(
        [[-5.60166031e2, -8.0e2, 0.0, -1.288e3],
         [5.51091060e-14, -4.5e2, -5.60166031e2, -8.58939847e2],
         [1.22464680e-16, -1.0, 0.0, -1.61],
         [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    "CAM_BACK_LEFT": np.array(
        [[-1.14251841e3, 8.0e2, 0.0, -6.84385123e2],
         [-4.22861679e2, -1.53909064e2, -1.14251841e3, -4.96004706e2],
         [-9.39692621e-1, -3.42020143e-1, 0.0, -4.92889531e-1],
         [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    "CAM_BACK_RIGHT": np.array(
        [[3.60989788e2, -1.34723223e3, 0.0, -1.04238127e2],
         [4.22861679e2, -1.53909064e2, -1.14251841e3, -4.96004706e2],
         [9.39692621e-1, -3.42020143e-1, 0.0, -4.92889531e-1],
         [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
}

LIDAR2CAM = {
    "CAM_FRONT": np.array(
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, -1.0, -0.24],
         [0.0, 1.0, 0.0, -1.19],
         [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    "CAM_FRONT_LEFT": np.array(
        [[0.57357644, 0.81915204, 0.0, -0.22517331],
         [0.0, 0.0, -1.0, -0.24],
         [-0.81915204, 0.57357644, 0.0, -0.82909407],
         [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    "CAM_FRONT_RIGHT": np.array(
        [[0.57357644, -0.81915204, 0.0, 0.22517331],
         [0.0, 0.0, -1.0, -0.24],
         [0.81915204, 0.57357644, 0.0, -0.82909407],
         [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    "CAM_BACK": np.array(
        [[-1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, -1.0, -0.24],
         [0.0, -1.0, 0.0, -1.61],
         [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    "CAM_BACK_LEFT": np.array(
        [[-0.34202014, 0.93969262, 0.0, -0.25388956],
         [0.0, 0.0, -1.0, -0.24],
         [-0.93969262, -0.34202014, 0.0, -0.49288953],
         [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    "CAM_BACK_RIGHT": np.array(
        [[-0.34202014, -0.93969262, 0.0, 0.25388956],
         [0.0, 0.0, -1.0, -0.24],
         [0.93969262, -0.34202014, 0.0, -0.49288953],
         [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
}


def get_lidar2img_matrix(camera_name: str) -> np.ndarray:
    """Return a defensive copy of a lidar-to-image matrix."""

    if camera_name not in LIDAR2IMG:
        raise KeyError(f"Unknown Bench2Drive camera: {camera_name}")
    return LIDAR2IMG[camera_name].copy()


def get_lidar2cam_matrix(camera_name: str) -> np.ndarray:
    """Return a defensive copy of a lidar-to-camera matrix."""

    if camera_name not in LIDAR2CAM:
        raise KeyError(f"Unknown Bench2Drive camera: {camera_name}")
    return LIDAR2CAM[camera_name].copy()
