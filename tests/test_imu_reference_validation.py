from __future__ import annotations

import h5py
import numpy as np

from Code.wild_preprocess.analog.imu_preprocess import prepare_imu_prefusion
from Code.wild_preprocess.analog.imu_fusion import fuse_imu_ahrs
from Code.wild_preprocess.analog.imu_motion import compute_imu_motion
from Code.wild_preprocess.validation.imu_reference import (
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
