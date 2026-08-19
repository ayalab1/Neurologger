"""Numerical reference for MATLAB R2024b ``ahrsfilter`` defaults.

This is a clean Python transcription of the readable 12-error-state indirect
Kalman equations installed with MATLAB R2024b.  It exists solely to reproduce
the legacy WILD MATLAB orientation path; it is not a generic or replaceable
AHRS choice.  Quaternion arrays use MATLAB's scalar-first ``[w, x, y, z]``
layout and describe frame rotations from NED global coordinates to sensor
coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


# R2024b fusion.internal.UnitConversions uses the toolbox's legacy 9.81 m/s²
# conversion, not the conventional exact standard-gravity value 9.80665.
STANDARD_GRAVITY = 9.81


@dataclass(frozen=True)
class AhrsResult:
    quaternions: np.ndarray
    angular_velocity: np.ndarray
    residuals: np.ndarray | None
    residual_covariances: np.ndarray | None


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def _quat_from_rotvec(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle == 0.0:
        return np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    half = 0.5 * angle
    scale = math.sin(half) / angle
    return np.concatenate(([math.cos(half)], np.asarray(vector, dtype=np.float64) * scale))


def _frame_matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """Match MATLAB ``rotmat(q, 'frame')``."""

    w, x, y, z = q
    point = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    return point.T


def _quat_from_point_matrix(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        q = np.asarray(
            (
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            )
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            q = np.asarray(
                ((matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                 (matrix[0, 1] + matrix[1, 0]) / scale,
                 (matrix[0, 2] + matrix[2, 0]) / scale)
            )
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            q = np.asarray(
                ((matrix[0, 2] - matrix[2, 0]) / scale,
                 (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale)
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            q = np.asarray(
                ((matrix[1, 0] - matrix[0, 1]) / scale,
                 (matrix[0, 2] + matrix[2, 0]) / scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale)
            )
    q /= np.linalg.norm(q)
    return -q if q[0] < 0 else q


def _quat_from_frame_matrix(matrix: np.ndarray) -> np.ndarray:
    return _quat_from_point_matrix(np.asarray(matrix, dtype=np.float64).T)


def _ned_ecompass(accel: np.ndarray, magnetic: np.ndarray) -> np.ndarray:
    down = np.asarray(accel, dtype=np.float64)
    east = np.cross(down, magnetic)
    north = np.cross(east, down)
    matrix = np.column_stack((north, east, down))
    norms = np.linalg.norm(matrix, axis=0)
    if np.any(norms == 0.0) or not np.all(np.isfinite(norms)):
        return np.eye(3, dtype=np.float64)
    matrix /= norms
    return matrix


def _negative_skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray(((0.0, z, -y), (-z, 0.0, x), (y, -x, 0.0)))


class AhrsFilter:
    """Stateful R2024b-compatible AHRS filter for NED and decimation one."""

    def __init__(self, *, sample_rate_hz: float = 100.0) -> None:
        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be finite and positive")
        self.sample_rate_hz = float(sample_rate_hz)
        self.dt = 1.0 / self.sample_rate_hz
        self.accelerometer_noise = 2e-6 * STANDARD_GRAVITY**2
        self.gyroscope_noise = 9.1385e-5
        self.gyroscope_drift_noise = 3.0462e-13
        self.linear_acceleration_noise = 1e-4 * STANDARD_GRAVITY**2
        self.linear_acceleration_decay = 0.5
        self.magnetometer_noise = 0.1
        self.magnetic_disturbance_noise = 0.5
        self.magnetic_disturbance_decay = 0.5
        self.expected_magnetic_field_strength = 50.0
        self.reset()

    def reset(self) -> None:
        self.orientation = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
        self.gyroscope_offset = np.zeros(3, dtype=np.float64)
        self.linear_acceleration = np.zeros(3, dtype=np.float64)
        self.magnetic_vector = np.asarray(
            (self.expected_magnetic_field_strength, 0.0, 0.0), dtype=np.float64
        )
        self.covariance = np.zeros((12, 12), dtype=np.float64)
        orient_variance = math.radians(1.0) ** 2 * 2000e-5
        gyro_bias_variance = math.radians(1.0) ** 2 * 250e-3
        accel_variance = 10e-5 * STANDARD_GRAVITY**2
        self.covariance[0:3, 0:3] = orient_variance * np.eye(3)
        self.covariance[3:6, 3:6] = gyro_bias_variance * np.eye(3)
        self.covariance[6:9, 6:9] = accel_variance * np.eye(3)
        self.covariance[9:12, 9:12] = 0.6 * np.eye(3)
        self.first_sample = True

    def _measurement_covariance(self) -> np.ndarray:
        gyro_term = self.dt**2 * (self.gyroscope_drift_noise + self.gyroscope_noise)
        accel = self.accelerometer_noise + self.linear_acceleration_noise + gyro_term
        magnetic = self.magnetometer_noise + self.magnetic_disturbance_noise + gyro_term
        return np.diag(np.repeat((accel, magnetic), 3)).astype(np.float64)

    def process(
        self,
        accelerometer: np.ndarray,
        gyroscope: np.ndarray,
        magnetometer: np.ndarray,
        *,
        include_diagnostics: bool = True,
    ) -> AhrsResult:
        accel = np.asarray(accelerometer, dtype=np.float64)
        gyro = np.asarray(gyroscope, dtype=np.float64)
        magnetic = np.asarray(magnetometer, dtype=np.float64)
        if accel.ndim != 2 or accel.shape[1] != 3 or accel.shape[0] == 0:
            raise ValueError("accelerometer must be a non-empty N-by-3 array")
        if gyro.shape != accel.shape or magnetic.shape != accel.shape:
            raise ValueError("gyroscope and magnetometer must match accelerometer shape")
        if not (np.all(np.isfinite(accel)) and np.all(np.isfinite(gyro)) and np.all(np.isfinite(magnetic))):
            raise ValueError("AHRS inputs must be finite")
        count = accel.shape[0]
        quaternions = np.empty((count, 4), dtype=np.float64)
        angular_velocity = np.empty((count, 3), dtype=np.float64)
        residuals = np.empty((count, 6), dtype=np.float64) if include_diagnostics else None
        residual_covariances = (
            np.empty((count, 6, 6), dtype=np.float64) if include_diagnostics else None
        )
        measurement_covariance = self._measurement_covariance()
        for index in range(count):
            angular_velocity[index] = gyro[index] - self.gyroscope_offset
            if self.first_sample:
                self.orientation = _quat_from_frame_matrix(
                    _ned_ecompass(accel[index], magnetic[index])
                )
                self.first_sample = False

            delta = _quat_from_rotvec((gyro[index] - self.gyroscope_offset) * self.dt)
            prior_orientation = _quat_multiply(self.orientation, delta)
            if prior_orientation[0] < 0:
                prior_orientation = -prior_orientation
            prior_rotation = _frame_matrix_from_quat(prior_orientation)
            gravity_gyro = prior_rotation[:, 2] * STANDARD_GRAVITY
            linear_prior = self.linear_acceleration_decay * self.linear_acceleration
            gravity_accel = accel[index] + linear_prior
            magnetic_gyro = prior_rotation @ self.magnetic_vector

            gravity_difference = gravity_accel - gravity_gyro
            magnetic_difference = magnetic[index] - magnetic_gyro
            h_gravity = _negative_skew(gravity_gyro)
            h_magnetic = _negative_skew(magnetic_gyro)
            observation = np.block(
                [
                    [h_gravity, -h_gravity * self.dt, np.eye(3), np.zeros((3, 3))],
                    [h_magnetic, -h_magnetic * self.dt, np.zeros((3, 3)), -np.eye(3)],
                ]
            )
            residual_covariance = (
                observation @ self.covariance @ observation.T + measurement_covariance
            ).T
            gain_numerator = self.covariance @ observation.T
            gain = np.linalg.solve(residual_covariance.T, gain_numerator.T).T
            residual = np.concatenate((gravity_difference, magnetic_difference))
            magnetic_error = gain[9:12] @ residual
            jammed = float(magnetic_error @ magnetic_error) > (
                4.0 * self.expected_magnetic_field_strength**2
            )
            if jammed:
                posterior_error = gain[0:9, 0:3] @ gravity_difference
                orientation_error = posterior_error[0:3]
                gyro_offset_error = posterior_error[3:6]
                linear_error = posterior_error[6:9]
            else:
                posterior_error = gain @ residual
                orientation_error = posterior_error[0:3]
                gyro_offset_error = posterior_error[3:6]
                linear_error = posterior_error[6:9]

            correction = _quat_from_rotvec(-orientation_error)
            self.orientation = _quat_multiply(prior_orientation, correction)
            if self.orientation[0] < 0:
                self.orientation = -self.orientation
            self.orientation /= np.linalg.norm(self.orientation)
            posterior_rotation = _frame_matrix_from_quat(self.orientation)
            self.gyroscope_offset -= gyro_offset_error
            self.linear_acceleration = linear_prior - linear_error
            if not jammed:
                global_magnetic_error = posterior_rotation.T @ magnetic_error
                disturbed = self.magnetic_vector - global_magnetic_error
                inclination = float(np.clip(math.atan2(disturbed[2], disturbed[0]), -math.pi / 2, math.pi / 2))
                self.magnetic_vector = self.expected_magnetic_field_strength * np.asarray(
                    (math.cos(inclination), 0.0, math.sin(inclination))
                )

            posterior_covariance = self.covariance - gain @ (observation @ self.covariance)
            updated = np.zeros((12, 12), dtype=np.float64)
            diagonal = np.diag(posterior_covariance)
            updated[np.arange(3), np.arange(3)] = diagonal[0:3] + self.dt**2 * (
                diagonal[3:6] + self.gyroscope_drift_noise + self.gyroscope_noise
            )
            updated[np.arange(3, 6), np.arange(3, 6)] = (
                diagonal[3:6] + self.gyroscope_drift_noise
            )
            cross_covariance = -self.dt * updated[np.arange(3, 6), np.arange(3, 6)]
            updated[np.arange(3, 6), np.arange(3)] = cross_covariance
            updated[np.arange(3), np.arange(3, 6)] = cross_covariance
            updated[np.arange(6, 9), np.arange(6, 9)] = (
                self.linear_acceleration_decay**2 * diagonal[6:9]
                + self.linear_acceleration_noise
            )
            updated[np.arange(9, 12), np.arange(9, 12)] = (
                self.magnetic_disturbance_decay**2 * diagonal[9:12]
                + self.magnetic_disturbance_noise
            )
            self.covariance = updated
            quaternions[index] = self.orientation
            if residuals is not None and residual_covariances is not None:
                residuals[index] = residual
                residual_covariances[index] = residual_covariance
        return AhrsResult(
            quaternions=quaternions,
            angular_velocity=angular_velocity,
            residuals=residuals,
            residual_covariances=residual_covariances,
        )


def fuse_imu_ahrs(
    accelerometer: np.ndarray,
    gyroscope: np.ndarray,
    magnetometer: np.ndarray,
    *,
    sample_rate_hz: float = 100.0,
    include_diagnostics: bool = True,
) -> AhrsResult:
    """Filter one complete array using MATLAB R2024b default properties."""

    return AhrsFilter(sample_rate_hz=sample_rate_hz).process(
        accelerometer,
        gyroscope,
        magnetometer,
        include_diagnostics=include_diagnostics,
    )
