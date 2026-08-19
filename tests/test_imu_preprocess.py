from __future__ import annotations

import numpy as np
import pytest

from Code.wild_preprocess.analog.imu_preprocess import (
    IMU_SAMPLE_RATE_HZ,
    scale_imu_nominal,
    resample_imu_1250_to_100,
    design_imu_resample_filter,
    calibrate_imu,
    prepare_imu_prefusion,
)


@pytest.mark.parametrize(
    ("input_rows", "output_rows"),
    ((1, 1), (12, 1), (13, 2), (24, 2), (25, 2), (26, 3), (251, 21), (3127, 251)),
)
def test_matlab_resample_uses_ceil_2n_over_25_length(input_rows: int, output_rows: int) -> None:
    output = resample_imu_1250_to_100(np.ones((input_rows, 3), dtype=np.float64))
    assert output.shape == (output_rows, 3)


def test_matlab_default_filter_has_r2024b_design_shape_and_normalization() -> None:
    coefficients = design_imu_resample_filter()
    assert coefficients.shape == (501,)
    np.testing.assert_allclose(coefficients, coefficients[::-1], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(coefficients.sum(), 2.0, rtol=0.0, atol=1e-14)


def test_matlab_resample_phase_is_delay_compensated_and_constant_is_preserved_interior() -> None:
    impulse = np.zeros((1_000, 1), dtype=np.float64)
    impulse[125, 0] = 1.0  # exactly 0.1 seconds at 1250 Hz
    output = resample_imu_1250_to_100(impulse)
    assert output.shape == (80, 1)
    assert int(np.argmax(np.abs(output[:, 0]))) == 10

    constant = resample_imu_1250_to_100(np.ones((1_250, 1), dtype=np.float64))
    np.testing.assert_allclose(constant[20:-20], 1.0, rtol=0.0, atol=5e-5)


def test_nominal_scaling_and_wild_calibration_match_matlab_equations() -> None:
    adc = np.array(
        [
            [32768, -16384, 8192, 16384, -8192, 4096, 16384, -8192, 4096],
            [16384, -8192, 4096, 8192, -4096, 2048, 8192, -4096, 2048],
            [8192, -4096, 2048, 4096, -2048, 1024, 4096, -2048, 1024],
        ],
        dtype=np.float64,
    )
    nominal = scale_imu_nominal(adc)
    np.testing.assert_allclose(nominal.acc[0], (8.0 * 9.8, -4.0 * 9.8, 2.0 * 9.8))
    np.testing.assert_allclose(
        nominal.gyr[0], np.array((1000.0, -500.0, 250.0)) * np.pi / 180.0
    )
    np.testing.assert_allclose(nominal.mag[0], (575.0, -287.5, 312.5))

    calibrated = calibrate_imu(nominal)
    median_norm = np.median(np.sqrt(np.sum(nominal.acc * nominal.acc, axis=1)))
    np.testing.assert_allclose(calibrated.acc, nominal.acc / median_norm * 9.81)
    np.testing.assert_allclose(calibrated.gyr, nominal.gyr - np.median(nominal.gyr, axis=0))
    np.testing.assert_array_equal(calibrated.mag, nominal.mag)


def test_prefusion_output_has_matlab_time_and_no_sensor_fusion() -> None:
    raw = np.tile(np.arange(1, 10, dtype=np.float64), (1_250, 1))
    result = prepare_imu_prefusion(raw)
    assert result.resampled_adc.shape == (100, 9)
    np.testing.assert_allclose(
        result.timestamp_seconds,
        np.arange(100, dtype=np.float64) / IMU_SAMPLE_RATE_HZ,
    )
    assert result.nominal.acc.shape == result.calibrated.acc.shape == (100, 3)
