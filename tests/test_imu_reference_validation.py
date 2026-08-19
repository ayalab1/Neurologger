from __future__ import annotations

import h5py
import numpy as np

from Code.wild_preprocess.analog.imu_preprocess import prepare_imu_prefusion
from Code.wild_preprocess.analog.imu_fusion import fuse_imu_ahrs
from Code.wild_preprocess.analog.imu_motion import compute_imu_motion
from Code.wild_preprocess.validation.imu_reference import (
    _quaternion_metrics,
    compare_matlab_fusion_reference,
    compare_matlab_prefusion_reference,
    write_matlab_parity_report,
)


def _reference_file(path, source: np.ndarray) -> None:
    result = prepare_imu_prefusion(source[:, 1:10])
    ahrs = fuse_imu_ahrs(
        result.calibrated.acc,
        result.calibrated.gyr,
        result.calibrated.mag,
        include_diagnostics=False,
    )
    motion = compute_imu_motion(ahrs.quaternions, result.calibrated.acc)
    fields = {
        "resampledAdc": result.resampled_adc,
        "scaledData": result.nominal.as_matrix(),
        "acc": result.calibrated.acc,
        "gyr": result.calibrated.gyr,
        "mag": result.calibrated.mag,
        "quaternion": ahrs.quaternions,
        "orientation": motion.orientation.transpose(0, 2, 1),
        "accel": motion.acceleration,
        "speed": motion.speed,
    }
    with h5py.File(path, "w") as handle:
        reference = handle.create_group("reference")
        reference.create_dataset("sourceRows", data=np.asarray([[source.shape[0]]], dtype=float))
        reference.create_dataset("totalDeviceCount", data=np.asarray([[1]], dtype=float))
        reference.create_dataset("deviceIndices", data=np.asarray([[1]], dtype=float))
        device = reference.create_group("device")
        for name, value in fields.items():
            stored_value = value if name == "orientation" else value.T
            stored = handle.create_dataset(f"value_{name}", data=stored_value)
            links = device.create_dataset(name, shape=(1, 1), dtype=h5py.ref_dtype)
            links[0, 0] = stored.ref


def test_quaternion_metrics_is_exact_for_identical_inputs_without_mutation() -> None:
    actual = np.asarray(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.5, -0.5, 0.5, -0.5],
        ]
    )
    expected = actual.copy()
    actual_before = actual.copy()
    expected_before = expected.copy()

    metrics = _quaternion_metrics(actual, expected)

    assert metrics.max_angle_degrees == 0.0
    assert metrics.rms_angle_degrees == 0.0
    assert metrics.mean_angle_degrees == 0.0
    assert metrics.p99_angle_degrees == 0.0
    np.testing.assert_array_equal(actual, actual_before)
    np.testing.assert_array_equal(expected, expected_before)


def test_quaternion_metrics_is_sign_invariant() -> None:
    actual = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, -0.5, 0.5, -0.5],
        ]
    )

    metrics = _quaternion_metrics(actual, -actual)

    assert metrics.max_angle_degrees == 0.0
    assert metrics.rms_angle_degrees == 0.0


def test_quaternion_metrics_resolves_a_sub_microdegree_rotation() -> None:
    angle_degrees = 1e-7
    half_angle = np.deg2rad(angle_degrees) / 2.0
    actual = np.asarray([[1.0, 0.0, 0.0, 0.0]])
    expected = np.asarray([[np.cos(half_angle), np.sin(half_angle), 0.0, 0.0]])

    metrics = _quaternion_metrics(actual, expected)

    np.testing.assert_allclose(
        metrics.max_angle_degrees,
        angle_degrees,
        rtol=1e-12,
        atol=1e-15,
    )


def test_quaternion_metrics_handles_the_180_degree_boundary() -> None:
    actual = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    expected = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
        ]
    )

    metrics = _quaternion_metrics(actual, expected)

    assert metrics.max_angle_degrees == 180.0
    assert metrics.rms_angle_degrees == 180.0
    assert metrics.mean_angle_degrees == 180.0
    assert metrics.p99_angle_degrees == 180.0


def test_full_prefusion_validation_reads_matlab_v73_layout(tmp_path) -> None:
    rows = 313
    frames = np.zeros((rows, 16), dtype="<i2")
    frames[:, 1:10] = (
        np.arange(rows * 9, dtype=np.int64).reshape(rows, 9) % 20_001 - 10_000
    ).astype(np.int16)
    analog = tmp_path / "analogin.dat"
    frames.tofile(analog)
    reference = tmp_path / "reference.mat"
    _reference_file(reference, frames)

    report = compare_matlab_prefusion_reference(analog, reference)
    assert report.source_rows == rows
    assert len(report.devices) == 1
    for metric in report.devices[0].metrics:
        assert metric.max_abs == 0.0
        assert metric.rms == 0.0

    destination = write_matlab_parity_report(report, tmp_path / "report.json")
    assert '"stage": "resampled_adc"' in destination.read_text(encoding="utf-8")

    fusion = compare_matlab_fusion_reference(analog, reference)
    assert fusion.devices[0].quaternion.max_angle_degrees < 2e-6
    for metric in fusion.devices[0].metrics:
        assert metric.max_abs == 0.0
