"""Stagewise validation against ``WILD_generateIMUReference`` output.

The reference MAT file is MATLAB v7.3/HDF5 and intentionally lives outside
the source recording.  This module reads it without changing either input and
compares the deterministic pre-fusion stages before any AHRS comparison is
attempted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from ..analog.imu_preprocess import prepare_imu_prefusion
from ..analog.imu_fusion import fuse_imu_ahrs
from ..analog.imu_motion import compute_imu_motion
from ..analog.imu import (
    SynchronizedImuResult,
    build_imu_from_merged,
    project_raw_imu_intervals_to_canonical,
    write_synchronized_imu_mat,
)
from ..analog.integrity import IMU_MODALITY_INVALID_KINDS
from ..analog.models import AnalogSyncAnchor, AnalogSyncSegment
from ..binary_io import recordings_from_folders


@dataclass(frozen=True)
class MatlabParityMetrics:
    """Numerical error summary for one named N-by-M stage."""

    stage: str
    rows: int
    columns: int
    max_abs: float
    rms: float
    mean_abs: float
    p99_abs: float


@dataclass(frozen=True)
class MatlabPrefusionDeviceReport:
    """Pre-fusion parity report for one one-based merged device block."""

    device_index: int
    metrics: tuple[MatlabParityMetrics, ...]


@dataclass(frozen=True)
class MatlabPrefusionParityReport:
    """Complete deterministic pre-fusion comparison."""

    source_analog_file: str
    matlab_reference_file: str
    source_rows: int
    total_device_count: int
    devices: tuple[MatlabPrefusionDeviceReport, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MatlabQuaternionMetrics:
    """Sign-invariant orientation error for scalar-first quaternions."""

    rows: int
    max_angle_degrees: float
    rms_angle_degrees: float
    mean_angle_degrees: float
    p99_angle_degrees: float


@dataclass(frozen=True)
class MatlabFusionDeviceReport:
    """AHRS and derived-motion parity for one device."""

    device_index: int
    quaternion: MatlabQuaternionMetrics
    metrics: tuple[MatlabParityMetrics, ...]


@dataclass(frozen=True)
class MatlabFusionParityReport:
    """Complete MATLAB/Python fusion comparison."""

    source_analog_file: str
    matlab_reference_file: str
    source_rows: int
    total_device_count: int
    devices: tuple[MatlabFusionDeviceReport, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _metrics(stage: str, actual: np.ndarray, expected: np.ndarray) -> MatlabParityMetrics:
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    if actual_array.shape != expected_array.shape or actual_array.ndim != 2:
        raise ValueError(
            f"MATLAB parity shape mismatch for {stage}: "
            f"Python {actual_array.shape}, MATLAB {expected_array.shape}"
        )
    difference = actual_array - expected_array
    if not np.all(np.isfinite(difference)):
        raise ValueError(f"MATLAB parity comparison for {stage} contains non-finite values")
    absolute = np.abs(difference)
    return MatlabParityMetrics(
        stage=stage,
        rows=int(actual_array.shape[0]),
        columns=int(actual_array.shape[1]),
        max_abs=float(np.max(absolute, initial=0.0)),
        rms=float(np.sqrt(np.mean(difference * difference))) if difference.size else 0.0,
        mean_abs=float(np.mean(absolute)) if difference.size else 0.0,
        p99_abs=float(np.percentile(absolute, 99.0)) if difference.size else 0.0,
    )


def _matlab_device_matrix(
    handle: h5py.File,
    field: str,
    reference_device_column: int,
) -> np.ndarray:
    references = handle["reference/device"][field]
    matlab_array = np.asarray(handle[references[0, reference_device_column]], dtype=np.float64)
    # MATLAB v7.3/HDF5 stores the dimensions reversed for these numeric
    # matrices.  The harness writes N-by-M and h5py observes M-by-N.
    if matlab_array.ndim != 2:
        raise ValueError(f"unexpected MATLAB reference rank for device.{field}")
    return matlab_array.T


def _matlab_device_orientation(
    handle: h5py.File,
    reference_device_column: int,
) -> np.ndarray:
    references = handle["reference/device"]["orientation"]
    stored = np.asarray(handle[references[0, reference_device_column]], dtype=np.float64)
    if stored.ndim != 3 or stored.shape[1:] != (3, 3):
        raise ValueError("unexpected MATLAB reference rank for device.orientation")
    # A MATLAB 3-by-3-by-N array is observed as N-by-3-by-3 with the two
    # matrix axes reversed through the v7.3/HDF5 dimension convention.
    return stored.transpose(0, 2, 1)


def _quaternion_metrics(actual: np.ndarray, expected: np.ndarray) -> MatlabQuaternionMetrics:
    actual_q = np.array(actual, dtype=np.float64, copy=True)
    expected_q = np.array(expected, dtype=np.float64, copy=True)
    if actual_q.shape != expected_q.shape or actual_q.ndim != 2 or actual_q.shape[1] != 4:
        raise ValueError("MATLAB/Python quaternion shapes must match N-by-4")
    actual_q /= np.linalg.norm(actual_q, axis=1)[:, None]
    expected_q /= np.linalg.norm(expected_q, axis=1)[:, None]
    # q and -q encode the same orientation, so align both inputs to the same
    # hemisphere.  The difference/sum atan2 form stays well-conditioned near
    # zero and returns exact zero for identical normalized floating-point
    # vectors, unlike arccos(dot), which amplifies a one-ULP dot-product error.
    dots = np.sum(actual_q * expected_q, axis=1)
    aligned_expected = np.where(dots[:, None] < 0.0, -expected_q, expected_q)
    difference_norm = np.linalg.norm(actual_q - aligned_expected, axis=1)
    sum_norm = np.linalg.norm(actual_q + aligned_expected, axis=1)
    angles = np.degrees(4.0 * np.arctan2(difference_norm, sum_norm))
    return MatlabQuaternionMetrics(
        rows=int(actual_q.shape[0]),
        max_angle_degrees=float(np.max(angles, initial=0.0)),
        rms_angle_degrees=float(np.sqrt(np.mean(angles * angles))) if angles.size else 0.0,
        mean_angle_degrees=float(np.mean(angles)) if angles.size else 0.0,
        p99_angle_degrees=float(np.percentile(angles, 99.0)) if angles.size else 0.0,
    )


def _matlab_scalar(handle: h5py.File, name: str) -> int:
    value = np.asarray(handle[f"reference/{name}"])
    if value.size != 1:
        raise ValueError(f"MATLAB reference field {name} must be scalar")
    return int(value.reshape(-1)[0])


def compare_matlab_prefusion_reference(
    merged_analog_path: str | Path,
    matlab_reference_path: str | Path,
    *,
    device_indices: Iterable[int] | None = None,
) -> MatlabPrefusionParityReport:
    """Compare full merged analog device blocks to a MATLAB reference.

    Inputs are opened read-only.  Device blocks contain 16 interleaved int16
    lanes in input order, and IMU lanes are one-based channels 2:10.
    """

    analog_path = Path(merged_analog_path)
    reference_path = Path(matlab_reference_path)
    if not analog_path.is_file() or not reference_path.is_file():
        raise FileNotFoundError("merged analog and MATLAB reference files must exist")

    with h5py.File(reference_path, "r") as reference:
        source_rows = _matlab_scalar(reference, "sourceRows")
        total_devices = _matlab_scalar(reference, "totalDeviceCount")
        reference_indices = np.asarray(reference["reference/deviceIndices"]).reshape(-1).astype(int)
        requested = tuple(int(value) for value in (reference_indices if device_indices is None else device_indices))
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("device_indices must be non-empty and unique")
        column_by_device = {int(value): index for index, value in enumerate(reference_indices)}
        if any(index not in column_by_device for index in requested):
            raise ValueError("requested device is absent from the MATLAB reference")

        expected_bytes = source_rows * total_devices * 16 * np.dtype("<i2").itemsize
        if analog_path.stat().st_size != expected_bytes:
            raise ValueError("merged analog byte length does not match MATLAB reference metadata")
        analog = np.memmap(
            analog_path,
            dtype="<i2",
            mode="r",
            shape=(source_rows, total_devices * 16),
        )
        reports: list[MatlabPrefusionDeviceReport] = []
        try:
            for device_index in requested:
                lane_start = (device_index - 1) * 16 + 1
                raw_adc = np.asarray(analog[:, lane_start : lane_start + 9])
                python = prepare_imu_prefusion(raw_adc)
                reference_column = column_by_device[device_index]
                stage_values = (
                    ("resampled_adc", python.resampled_adc, "resampledAdc"),
                    ("nominal_scaled", python.nominal.as_matrix(), "scaledData"),
                    ("calibrated_acc", python.calibrated.acc, "acc"),
                    ("calibrated_gyr", python.calibrated.gyr, "gyr"),
                    ("calibrated_mag", python.calibrated.mag, "mag"),
                )
                reports.append(
                    MatlabPrefusionDeviceReport(
                        device_index=device_index,
                        metrics=tuple(
                            _metrics(
                                stage,
                                actual,
                                _matlab_device_matrix(reference, matlab_field, reference_column),
                            )
                            for stage, actual, matlab_field in stage_values
                        ),
                    )
                )
        finally:
            del analog

    return MatlabPrefusionParityReport(
        source_analog_file=str(analog_path.resolve()),
        matlab_reference_file=str(reference_path.resolve()),
        source_rows=source_rows,
        total_device_count=total_devices,
        devices=tuple(reports),
    )


def write_matlab_parity_report(
    report: MatlabPrefusionParityReport | MatlabFusionParityReport,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a compact JSON validation report without embedding sample data."""

    if not isinstance(report, (MatlabPrefusionParityReport, MatlabFusionParityReport)):
        raise ValueError("report must be a MATLAB parity report")
    path = Path(destination)
    if path.exists() and not overwrite:
        raise FileExistsError(f"MATLAB parity report exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def compare_matlab_fusion_reference(
    merged_analog_path: str | Path,
    matlab_reference_path: str | Path,
    *,
    device_indices: Iterable[int] | None = None,
) -> MatlabFusionParityReport:
    """Compare MATLAB-compatible AHRS and derived motion on full devices."""

    analog_path = Path(merged_analog_path)
    reference_path = Path(matlab_reference_path)
    if not analog_path.is_file() or not reference_path.is_file():
        raise FileNotFoundError("merged analog and MATLAB reference files must exist")
    with h5py.File(reference_path, "r") as reference:
        source_rows = _matlab_scalar(reference, "sourceRows")
        total_devices = _matlab_scalar(reference, "totalDeviceCount")
        reference_indices = np.asarray(reference["reference/deviceIndices"]).reshape(-1).astype(int)
        requested = tuple(int(value) for value in (reference_indices if device_indices is None else device_indices))
        column_by_device = {int(value): index for index, value in enumerate(reference_indices)}
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("device_indices must be non-empty and unique")
        if any(index not in column_by_device for index in requested):
            raise ValueError("requested device is absent from the MATLAB reference")
        expected_bytes = source_rows * total_devices * 16 * np.dtype("<i2").itemsize
        if analog_path.stat().st_size != expected_bytes:
            raise ValueError("merged analog byte length does not match MATLAB reference metadata")
        analog = np.memmap(
            analog_path,
            dtype="<i2",
            mode="r",
            shape=(source_rows, total_devices * 16),
        )
        reports: list[MatlabFusionDeviceReport] = []
        try:
            for device_index in requested:
                lane_start = (device_index - 1) * 16 + 1
                prefusion = prepare_imu_prefusion(
                    np.asarray(analog[:, lane_start : lane_start + 9])
                )
                ahrs = fuse_imu_ahrs(
                    prefusion.calibrated.acc,
                    prefusion.calibrated.gyr,
                    prefusion.calibrated.mag,
                    sample_rate_hz=prefusion.sample_rate_hz,
                    include_diagnostics=False,
                )
                motion = compute_imu_motion(
                    ahrs.quaternions,
                    prefusion.calibrated.acc,
                    sample_rate_hz=prefusion.sample_rate_hz,
                )
                column = column_by_device[device_index]
                matlab_quaternion = _matlab_device_matrix(reference, "quaternion", column)
                reports.append(
                    MatlabFusionDeviceReport(
                        device_index=device_index,
                        quaternion=_quaternion_metrics(ahrs.quaternions, matlab_quaternion),
                        metrics=(
                            _metrics(
                                "orientation",
                                motion.orientation.reshape(-1, 9),
                                _matlab_device_orientation(reference, column).reshape(-1, 9),
                            ),
                            _metrics(
                                "world_acceleration",
                                motion.acceleration,
                                _matlab_device_matrix(reference, "accel", column),
                            ),
                            _metrics(
                                "speed",
                                motion.speed,
                                _matlab_device_matrix(reference, "speed", column),
                            ),
                        ),
                    )
                )
        finally:
            del analog
    return MatlabFusionParityReport(
        source_analog_file=str(analog_path.resolve()),
        matlab_reference_file=str(reference_path.resolve()),
        source_rows=source_rows,
        total_device_count=total_devices,
        devices=tuple(reports),
    )


def _segment_from_manifest(payload: dict[str, object]) -> AnalogSyncSegment:
    values = dict(payload)
    anchors = tuple(
        AnalogSyncAnchor(**dict(anchor)) for anchor in values.pop("anchors", [])
    )
    return AnalogSyncSegment(anchors=anchors, **values)


def regenerate_imu_from_published_session(
    session_folder: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[SynchronizedImuResult, Path]:
    """Regenerate IMU from an existing canonical analog publication.

    This validation helper does not rerun neural synchronization and never
    writes inside the published session.  It reconstructs the exact analog
    mapping and IMU modality exclusions from ``wild_preprocess_run.json``.
    """

    session = Path(session_folder).resolve()
    output = Path(destination).resolve()
    try:
        output.relative_to(session)
    except ValueError:
        pass
    else:
        raise ValueError("validation IMU output must be outside the published session")
    manifest_path = session / "wild_preprocess_run.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"canonical manifest is unavailable: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    merge = manifest.get("merge")
    if not isinstance(merge, dict):
        raise ValueError("canonical manifest has no merge metadata")
    devices = merge.get("devices")
    timelines = merge.get("analog_timelines")
    if not isinstance(devices, list) or not isinstance(timelines, list):
        raise ValueError("canonical manifest lacks device/analog timeline metadata")
    folders = [Path(str(device["folder"])) for device in devices]
    recordings = recordings_from_folders(folders)
    if len(recordings) != len(timelines):
        raise ValueError("manifest device and analog timeline counts differ")
    segment_sets: dict[int, tuple[AnalogSyncSegment, ...]] = {}
    invalid_intervals: dict[int, tuple[tuple[int, int], ...]] = {}
    mapping_hashes: dict[int, str] = {}
    for timeline in timelines:
        device_index = int(timeline["device_index"])
        segments = tuple(
            _segment_from_manifest(dict(segment)) for segment in timeline["segments"]
        )
        segment_sets[device_index] = segments
        raw_invalid = tuple(
            (int(event["raw_start_row"]), int(event["raw_end_row"]))
            for event in timeline.get("integrity_events", [])
            if event.get("kind") in IMU_MODALITY_INVALID_KINDS
        )
        invalid_intervals[device_index] = project_raw_imu_intervals_to_canonical(
            segments,
            raw_invalid,
            device_index=device_index,
        )
        mapping_hashes[device_index] = str(timeline["mapping_hash"])
    master_index = int(manifest["master_index"])
    result = build_imu_from_merged(
        recordings,
        session / "analogin.dat",
        session / "valid_analog_samples.dat",
        segments_by_device=segment_sets,
        canonical_rows=int(merge["analog_samples"]),
        invalid_canonical_intervals_by_device=invalid_intervals,
        mapping_hashes_by_device=mapping_hashes,
        master_index=master_index,
        master_start_sample=int(merge["common_start_master_sample"]),
        master_start_sec=(
            int(merge["common_start_master_sample"])
            / float(recordings[master_index - 1].fs)
        ),
        perform_sensor_fusion=True,
    )
    written = write_synchronized_imu_mat(result, output, overwrite=overwrite)
    return result, written
