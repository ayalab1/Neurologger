"""Legacy WILD motion post-processing after MATLAB-compatible AHRS fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt


@dataclass(frozen=True)
class ImuMotion:
    """Numeric fields written into legacy MATLAB ``fusionData``."""

    orientation: np.ndarray
    acceleration: np.ndarray
    speed: np.ndarray


def quaternion_to_rotation_matrices(quaternions: np.ndarray) -> np.ndarray:
    """Match MATLAB ``quat2rotm`` for scalar-first N-by-4 quaternions."""

    values = np.asarray(quaternions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] == 0:
        raise ValueError("quaternions must be a non-empty N-by-4 array")
    if not np.all(np.isfinite(values)):
        raise ValueError("quaternions must be finite")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("quaternions must have nonzero norm")
    q = values / norms[:, None]
    w, x, y, z = q.T
    matrices = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrices[:, 0, 1] = 2.0 * (x * y - w * z)
    matrices[:, 0, 2] = 2.0 * (x * z + w * y)
    matrices[:, 1, 0] = 2.0 * (x * y + w * z)
    matrices[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrices[:, 1, 2] = 2.0 * (y * z - w * x)
    matrices[:, 2, 0] = 2.0 * (x * z - w * y)
    matrices[:, 2, 1] = 2.0 * (y * z + w * x)
    matrices[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrices


def compute_imu_motion(
    quaternions: np.ndarray,
    calibrated_acceleration: np.ndarray,
    *,
    sample_rate_hz: float = 100.0,
) -> ImuMotion:
    """Match the post-``ahrsfilter`` calculations in ``WILD_scaleIMU``.

    This intentionally preserves the existing MATLAB multiplication
    ``quat2rotm(q) * accel_body`` even though the ``ahrsfilter`` documentation
    describes its quaternion as a navigation-to-body frame rotation.  Changing
    that convention would be a scientific semantic change, not a parity fix.
    """

    acceleration = np.asarray(calibrated_acceleration, dtype=np.float64)
    if acceleration.ndim != 2 or acceleration.shape[1] != 3 or acceleration.shape[0] == 0:
        raise ValueError("calibrated_acceleration must be a non-empty N-by-3 array")
    if not np.all(np.isfinite(acceleration)):
        raise ValueError("calibrated_acceleration must be finite")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be finite and positive")
    orientation = quaternion_to_rotation_matrices(quaternions)
    if orientation.shape[0] != acceleration.shape[0]:
        raise ValueError("quaternion and acceleration row counts must match")
    world = np.einsum("nij,nj->ni", orientation, acceleration)
    world -= np.median(world, axis=0)
    integrated = np.cumsum(world / float(sample_rate_hz), axis=0)
    numerator, denominator = butter(
        2,
        0.1 * 2.0 / float(sample_rate_hz),
        btype="highpass",
    )
    # MATLAB R2024b filtfilt uses an odd reflection of length 3*filter_order
    # for transfer-function input.  scipy's default is 3*max(len(a),len(b)),
    # so the MATLAB length is passed explicitly.
    speed = filtfilt(
        numerator,
        denominator,
        integrated,
        axis=0,
        padtype="odd",
        padlen=3 * (max(len(numerator), len(denominator)) - 1),
        method="pad",
    )
    return ImuMotion(
        orientation=orientation,
        acceleration=world,
        speed=np.asarray(speed, dtype=np.float64),
    )
