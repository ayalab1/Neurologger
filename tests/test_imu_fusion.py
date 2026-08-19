from __future__ import annotations

import numpy as np

from Code.wild_preprocess.analog.imu_fusion import AhrsFilter, fuse_imu_ahrs


MATLAB_R2024B_GOLDEN = np.asarray(
    [
        [0.999223753751430, -0.007631716349316, 0.002472596558212, -0.038568550798865],
        [0.999222224111001, -0.007577954026455, 0.003502975479133, -0.038538949365923],
        [0.999222969987875, -0.007494221263781, 0.003937116748190, -0.038494051592057],
        [0.999225619624879, -0.007387733648754, 0.004063073745855, -0.038432719244772],
        [0.999229660624782, -0.007261620348578, 0.004028261923967, -0.038355277387447],
        [0.999234714848318, -0.007117020406527, 0.003908283832368, -0.038263010597951],
        [0.999240516436353, -0.006954305095565, 0.003742460879860, -0.038157855537383],
        [0.999246862237584, -0.006773730577429, 0.003551787986374, -0.038042209248517],
        [0.999253578998499, -0.006575766863212, 0.003347973394153, -0.037918797752599],
        [0.999260506585885, -0.006361248543889, 0.003137990189999, -0.037790574383720],
        [0.999267491095153, -0.006131426933209, 0.002926366964790, -0.037660632237784],
        [0.999274383281324, -0.005887966437761, 0.002716337582202, -0.037532123295879],
    ],
    dtype=np.float64,
)


def _deterministic_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = np.arange(12, dtype=np.float64)
    acceleration = np.column_stack(
        (
            0.2 * np.sin(0.13 * index),
            -0.15 * np.cos(0.17 * index),
            9.80665 + 0.1 * np.sin(0.07 * index),
        )
    )
    gyroscope = np.column_stack(
        (
            0.02 + 0.003 * np.sin(0.11 * index),
            -0.01 + 0.002 * np.cos(0.09 * index),
            0.03 * np.sin(0.15 * index),
        )
    )
    magnetic = np.column_stack(
        (
            42 + 0.4 * np.cos(0.05 * index),
            3 + 0.2 * np.sin(0.08 * index),
            18 + 0.3 * np.cos(0.06 * index),
        )
    )
    return acceleration, gyroscope, magnetic


def _sign_invariant_error(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return np.minimum(
        np.linalg.norm(actual - expected, axis=1),
        np.linalg.norm(actual + expected, axis=1),
    )


def test_r2024b_default_golden_quaternions() -> None:
    result = fuse_imu_ahrs(*_deterministic_inputs(), sample_rate_hz=100.0)
    error = _sign_invariant_error(result.quaternions, MATLAB_R2024B_GOLDEN)
    assert np.max(error) < 2e-12


def test_stationary_ned_is_identity() -> None:
    count = 20
    acceleration = np.tile((0.0, 0.0, 9.81), (count, 1))
    gyroscope = np.zeros((count, 3), dtype=np.float64)
    magnetic = np.tile((50.0, 0.0, 0.0), (count, 1))
    result = fuse_imu_ahrs(acceleration, gyroscope, magnetic)
    expected = np.tile((1.0, 0.0, 0.0, 0.0), (count, 1))
    assert np.max(_sign_invariant_error(result.quaternions, expected)) < 1e-14


def test_stateful_chunks_equal_one_shot() -> None:
    acceleration, gyroscope, magnetic = _deterministic_inputs()
    expected = fuse_imu_ahrs(acceleration, gyroscope, magnetic)
    filter_object = AhrsFilter(sample_rate_hz=100.0)
    first = filter_object.process(acceleration[:5], gyroscope[:5], magnetic[:5])
    second = filter_object.process(acceleration[5:], gyroscope[5:], magnetic[5:])
    actual = np.vstack((first.quaternions, second.quaternions))
    assert np.array_equal(actual, expected.quaternions)


def test_rejects_invalid_shapes_and_values() -> None:
    acceleration, gyroscope, magnetic = _deterministic_inputs()
    try:
        fuse_imu_ahrs(acceleration[:, :2], gyroscope, magnetic)
    except ValueError as error:
        assert "N-by-3" in str(error)
    else:
        raise AssertionError("expected shape validation")
    acceleration[3, 0] = np.nan
    try:
        fuse_imu_ahrs(acceleration, gyroscope, magnetic)
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("expected finite-value validation")
