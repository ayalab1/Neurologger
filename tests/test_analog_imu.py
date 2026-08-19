from __future__ import annotations

from pathlib import Path
import inspect
import tempfile

import numpy as np
from scipy.io import loadmat
from scipy.signal import firwin

from Code.wild_preprocess.analog.imu import (
    build_imu_from_merged,
    build_synchronized_imu,
    build_filtered_imu_from_merged,
    project_raw_imu_intervals_to_canonical,
    write_synchronized_imu_mat,
)
from Code.wild_preprocess.analog.models import AnalogSyncAnchor, AnalogSyncSegment
from Code.wild_preprocess.analog.write import write_canonical_analog
from Code.wild_preprocess.models import Recording


def _recording(folder: Path, rows: np.ndarray, *, name: str = "device") -> Recording:
    rows.astype("<i2", copy=False).tofile(folder / "analogin.dat")
    return Recording(
        folder=folder,
        amplifier_file=folder / "amplifier.dat",
        analog_file=folder / "analogin.dat",
        ce_params_file=folder / "cerebus.dat",
        device_name=name,
        recording_name=name + "_recording",
        fs=25_000,
        n_channels=64,
        n_samples=rows.shape[0] * 20,
        analog_channels=16,
        analog_samples=rows.shape[0],
    )


def _segment(start: int, end: int, *, device: int = 1) -> AnalogSyncSegment:
    anchors = tuple(
        AnalogSyncAnchor(
            canonical_row=row,
            raw_row=float(row),
            verified=True,
            confidence="high",
            verification_source="test",
        )
        for row in (start, end - 1)
    )
    return AnalogSyncSegment(
        device_index=device,
        canonical_start_row=start,
        canonical_end_row=end,
        raw_start_row=start,
        raw_end_row=end,
        raw_scale=1.0,
        raw_intercept_rows=0.0,
        anchors=anchors,
        confidence="high",
        publishable=True,
    )


def _affine_segment(
    start: int,
    end: int,
    *,
    scale: float,
    intercept: float,
    raw_end: int,
    device: int = 1,
) -> AnalogSyncSegment:
    anchors = tuple(
        AnalogSyncAnchor(
            canonical_row=row,
            raw_row=scale * row + intercept,
            verified=True,
            confidence="high",
            verification_source="test_fractional_affine",
        )
        for row in (start, end - 1)
    )
    return AnalogSyncSegment(
        device_index=device,
        canonical_start_row=start,
        canonical_end_row=end,
        raw_start_row=int(np.floor(scale * start + intercept)),
        raw_end_row=raw_end,
        raw_scale=scale,
        raw_intercept_rows=intercept,
        anchors=anchors,
        confidence="high",
        publishable=True,
    )


def _rows(count: int, values: tuple[int, ...] = (1000,) * 9) -> np.ndarray:
    rows = np.zeros((count, 16), dtype=np.int16)
    rows[:, 1:10] = np.asarray(values, dtype=np.int16)
    return rows


def test_units_shapes_invalid_zero_and_global_phase() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        recording = _recording(folder, _rows(1_500, (32767,) * 9))
        result = build_synchronized_imu(
            [recording], {1: (_segment(0, 1_500),)}, canonical_rows=1_500, filter_taps=11
        )
    device = result.devices[0]
    assert result.time_seconds.shape == (120,)
    assert np.allclose(result.canonical_rows[:4], [0.0, 12.5, 25.0, 37.5])
    assert device.raw_resampled.shape == (120, 9)
    assert device.imu.acc.shape == device.imu.gyr.shape == device.imu.mag.shape == (120, 3)
    first = np.flatnonzero(device.valid)[0]
    assert np.isclose(device.raw_resampled[first, 0], 32767 / 32768 * 8 * 9.8)
    assert np.isclose(device.raw_resampled[first, 3], 32767 / 32768 * 2000 * np.pi / 180)
    assert np.isclose(device.raw_resampled[first, 8], 32767 / 32768 * 2500)
    assert np.all(device.raw_resampled[~device.valid, :] == 0)
    assert np.all(device.source_rows[~device.valid] == -1)
    assert device.status == "OK"
    assert device.valid_count == np.count_nonzero(device.valid)
    assert np.isclose(device.valid_fraction, device.valid_count / 120)
    assert result.status == "OK"


def test_filter_never_leaks_across_mapping_gap() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        rows = _rows(3_000, (0,) * 9)
        # Large impulse before the gap.  The second valid segment is zero and
        # must remain zero even after antialias filtering.
        rows[1_100, 1] = 30_000
        recording = _recording(folder, rows)
        result = build_synchronized_imu(
            [recording],
            {1: (_segment(0, 1_250), _segment(1_750, 3_000))},
            canonical_rows=3_000,
            filter_taps=101,
        )
    device = result.devices[0]
    after_gap = device.canonical_rows >= 1_750
    assert np.any(device.valid & after_gap)
    assert np.allclose(device.raw_resampled[device.valid & after_gap, 0], 0.0)


def test_phase_is_not_restarted_by_later_segment_and_result_is_deterministic() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        recording = _recording(folder, _rows(3_000))
        segments = {1: (_segment(0, 1_200), _segment(1_600, 3_000))}
        first = build_synchronized_imu(
            [recording], segments, canonical_rows=3_000, filter_taps=11, chunk_rows=600
        )
        second = build_synchronized_imu(
            [recording], segments, canonical_rows=3_000, filter_taps=11, chunk_rows=600
        )
    device = first.devices[0]
    positions = np.flatnonzero(device.valid & (device.canonical_rows >= 1_600))
    assert positions.size
    assert np.isclose(device.canonical_rows[positions[0]] % 12.5, 0.0)
    assert np.array_equal(first.devices[0].raw_resampled, second.devices[0].raw_resampled)
    assert np.array_equal(first.devices[0].valid, second.devices[0].valid)


def test_mat_writer_preserves_top_level_imu_and_device_fields() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        recording = _recording(folder, _rows(1_500))
        result = build_synchronized_imu(
            [recording], {1: (_segment(0, 1_500),)}, canonical_rows=1_500, filter_taps=11
        )
        destination = write_synchronized_imu_mat(result, folder / "IMU.mat")
        loaded = loadmat(destination, struct_as_record=False, squeeze_me=True)
    assert "IMU" in loaded
    imu = loaded["IMU"]
    assert imu.fs == 100.0
    assert imu.masterIndex == 1
    assert imu.masterStartSample == 0
    assert imu.masterStartSec == 0.0
    assert np.asarray(imu.sourceAnalogFile).size == 0
    assert imu.fusionStatus == "NOT_RUN"
    assert imu.device.rawResampled.shape[1] == 9
    assert imu.device.imu.acc.shape[1] == 3


def test_matlab_compatible_merged_builder_calibrates_and_fuses_per_valid_run() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        rows = _rows(3_000, (0, 0, 4096, 0, 0, 0, 1000, 0, 500))
        recording = _recording(folder, rows)
        validity = folder / "valid_analog_samples.dat"
        np.ones(rows.shape[0], dtype=np.uint8).tofile(validity)
        result = build_imu_from_merged(
            [recording],
            recording.analog_file,
            validity,
            segments_by_device={1: (_segment(0, rows.shape[0]),)},
            canonical_rows=rows.shape[0],
            perform_sensor_fusion=True,
        )
        destination = write_synchronized_imu_mat(result, folder / "matlab_IMU.mat")
        loaded = loadmat(destination, struct_as_record=False, squeeze_me=True)["IMU"]

    device = result.devices[0]
    assert result.time_seconds.shape == (240,)
    assert np.count_nonzero(device.valid) == 220
    assert device.fusion is not None
    valid_run_start = int(np.flatnonzero(device.valid)[0])
    assert not device.fusion.valid[valid_run_start]
    assert np.array_equal(
        device.fusion.valid[valid_run_start + 1 :],
        device.valid[valid_run_start + 1 :],
    )
    np.testing.assert_allclose(
        np.median(np.linalg.norm(device.imu.acc[device.valid], axis=1)),
        9.81,
        rtol=0.0,
        atol=1e-12,
    )
    assert np.all(np.isfinite(device.fusion.quaternion[device.fusion.valid]))
    assert np.all(np.isnan(device.fusion.quaternion[~device.fusion.valid]))
    assert loaded.fusionStatus == "OK"
    assert loaded.fusionMethod == "matlab_r2024b_ahrsfilter_defaults_per_valid_run"
    assert loaded.device.fusionStatus == "OK"
    assert loaded.device.fusionMethod == "matlab_r2024b_ahrsfilter_defaults_per_valid_run"
    assert loaded.device.fusionData.quaternion.shape[1] == 4
    assert loaded.device.fusionData.orientation.shape[:2] == (3, 3)


def test_matlab_fusion_failure_preserves_scaled_imu_as_warn(tmp_path, monkeypatch) -> None:
    rows = _rows(3_000, (0, 0, 4096, 0, 0, 0, 1000, 0, 500))
    recording = _recording(tmp_path, rows)
    validity = tmp_path / "valid_analog_samples.dat"
    np.ones(rows.shape[0], dtype=np.uint8).tofile(validity)

    def unavailable(*args, **kwargs):
        raise RuntimeError("synthetic AHRS unavailable")

    monkeypatch.setattr("Code.wild_preprocess.analog.imu.fuse_imu_ahrs", unavailable)
    result = build_imu_from_merged(
        [recording],
        recording.analog_file,
        validity,
        segments_by_device={1: (_segment(0, rows.shape[0]),)},
        canonical_rows=rows.shape[0],
    )
    device = result.devices[0]
    assert result.status == "WARN"
    assert device.status == "WARN"
    assert np.any(device.valid)
    assert np.any(device.raw_resampled[device.valid] != 0)
    assert device.fusion is not None
    assert device.fusion.status == "WARN"
    assert not np.any(device.fusion.valid)
    assert np.all(np.isnan(device.fusion.quaternion))
    assert "synthetic AHRS unavailable" in device.fusion.warning


def test_matlab_fusion_warmup_is_invalid_after_each_segment_reset(tmp_path) -> None:
    rows = _rows(5_000, (0, 0, 4096, 0, 0, 0, 1000, 0, 500))
    recording = _recording(tmp_path, rows)
    validity_values = np.ones(rows.shape[0], dtype=np.uint8)
    validity_values[2_000:2_500] = 0
    validity = tmp_path / "valid_analog_samples.dat"
    validity_values.tofile(validity)
    result = build_imu_from_merged(
        [recording],
        recording.analog_file,
        validity,
        segments_by_device={1: (_segment(0, 2_000), _segment(2_500, 5_000))},
        canonical_rows=rows.shape[0],
    )
    device = result.devices[0]
    assert device.fusion is not None
    starts = np.flatnonzero(device.valid & np.r_[True, ~device.valid[:-1]])
    assert starts.size == 2
    assert np.all(~device.fusion.valid[starts])
    assert np.all(np.isnan(device.fusion.quaternion[starts]))
    assert np.all(device.fusion.valid[starts + 1])


def test_matlab_fusion_memory_guard_precedes_large_allocation(tmp_path) -> None:
    rows = _rows(3_000)
    recording = _recording(tmp_path, rows)
    validity = tmp_path / "valid_analog_samples.dat"
    np.ones(rows.shape[0], dtype=np.uint8).tofile(validity)
    try:
        build_imu_from_merged(
            [recording],
            recording.analog_file,
            validity,
            segments_by_device={1: (_segment(0, rows.shape[0]),)},
            canonical_rows=rows.shape[0],
            max_peak_bytes=1,
        )
    except ValueError as error:
        assert "estimated peak memory" in str(error)
    else:
        raise AssertionError("expected explicit peak-memory rejection")


def test_master_metadata_and_warn_status_without_valid_mapping() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        recording = _recording(folder, _rows(1_500))
        # A non-publishable segment exists but cannot be used as temporal
        # support, so the public status is WARN rather than a hard failure.
        unsupported = AnalogSyncSegment(
            device_index=1,
            canonical_start_row=0,
            canonical_end_row=1_500,
            raw_start_row=0,
            raw_end_row=1_500,
            raw_scale=1.0,
            raw_intercept_rows=0.0,
            confidence="unresolved",
            publishable=False,
        )
        result = build_synchronized_imu(
            [recording],
            {1: (unsupported,)},
            canonical_rows=1_500,
            filter_taps=11,
            master_index=1,
            master_start_sample=12_345,
            master_start_sec=0.4938,
        )
    assert result.status == "WARN"
    assert result.master_start_sample == 12_345
    assert np.isclose(result.master_start_sec, 0.4938)
    assert result.devices[0].status == "WARN"
    assert result.devices[0].valid_count == 0


def test_modality_invalid_raw_interval_masks_only_local_imu_support() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        recording = _recording(folder, _rows(3_000, (1_000,) * 9))
        result = build_synchronized_imu(
            [recording],
            {1: (_segment(0, 3_000),)},
            canonical_rows=3_000,
            filter_taps=101,
            invalid_raw_intervals_by_device={1: ((1_250, 1_300),)},
        )
    device = result.devices[0]
    # The 50 raw rows and the 50-row FIR support either side are invalid,
    # but valid acquisition/mapping resumes afterward.  This does not alter
    # the analog temporal mask; it is an IMU-only modality decision.
    local = (device.canonical_rows >= 1_150) & (device.canonical_rows < 1_400)
    assert np.any(~device.valid & local)
    assert np.all(device.raw_resampled[~device.valid & local, :] == 0)
    later = device.canonical_rows >= 1_500
    assert np.any(device.valid & later)
    assert np.all(device.raw_resampled[device.valid & later, :] != 0)


def test_merged_modality_exclusion_sets_warn_with_local_valid_support() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        recording = _recording(folder, _rows(3_000, (1_000,) * 9))
        merged = folder / "merged_analogin.dat"
        validity = folder / "valid_analog_samples.dat"
        _rows(3_000, (1_000,) * 9).astype("<i2").tofile(merged)
        np.ones(3_000, dtype=np.uint8).tofile(validity)
        result = build_filtered_imu_from_merged(
            [recording],
            merged,
            validity,
            segments_by_device={1: (_segment(0, 3_000),)},
            canonical_rows=3_000,
            filter_taps=101,
            invalid_canonical_intervals_by_device={1: ((1_250, 1_300),)},
        )
    device = result.devices[0]
    assert result.status == "WARN"
    assert device.status == "WARN"
    assert 0 < device.valid_count < device.valid.size
    local = (device.canonical_rows >= 1_150) & (device.canonical_rows < 1_400)
    assert np.any(~device.valid & local)
    assert np.all(device.raw_resampled[~device.valid & local, :] == 0)


def test_explicit_memory_limit_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        recording = _recording(folder, _rows(1_500))
        try:
            build_synchronized_imu(
                [recording],
                {1: (_segment(0, 1_500),)},
                canonical_rows=1_500,
                filter_taps=11,
                max_output_samples=2,
            )
        except ValueError as error:
            assert "in-memory limit" in str(error)
        else:
            raise AssertionError("expected explicit in-memory limit rejection")


def _write_merged_pair(
    folder: Path, device_rows: list[np.ndarray], validity_master_first: np.ndarray
) -> tuple[Path, Path]:
    analog_path = folder / "merged_analogin.dat"
    validity_path = folder / "valid_analog_samples.dat"
    np.concatenate(device_rows, axis=1).astype("<i2").tofile(analog_path)
    validity_master_first.astype(np.uint8).tofile(validity_path)
    return analog_path, validity_path


def test_merged_clean_identity_matches_raw_direct_on_common_valid_support() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        count = 3_000
        time = np.arange(count, dtype=np.float64) / 1_250.0
        rows = np.zeros((count, 16), dtype=np.int16)
        for lane in range(9):
            rows[:, lane + 1] = np.rint(2_000 * np.sin(2 * np.pi * (lane + 1) * time)).astype(np.int16)
        recording = _recording(folder, rows)
        analog_path, validity_path = _write_merged_pair(
            folder, [rows], np.ones((count, 1), dtype=np.uint8)
        )
        raw = build_synchronized_imu(
            [recording], {1: (_segment(0, count),)}, canonical_rows=count, filter_taps=51
        )
        merged = build_filtered_imu_from_merged(
            [recording],
            analog_path,
            validity_path,
            segments_by_device={1: (_segment(0, count),)},
            canonical_rows=count,
            filter_taps=51,
        )
    common = raw.devices[0].valid & merged.devices[0].valid
    assert np.any(common)
    # Both paths implement the same symmetric FIR and interpolation. Any
    # difference is floating-point convolution order only.
    assert np.allclose(
        raw.devices[0].raw_resampled[common],
        merged.devices[0].raw_resampled[common],
        rtol=1e-12,
        atol=1e-12,
    )


def test_merged_fractional_affine_matches_raw_direct_with_quantization_bound() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        raw_count = 3_200
        canonical_count = 3_000
        raw_time = np.arange(raw_count, dtype=np.float64) / 1_250.0
        rows = np.zeros((raw_count, 16), dtype=np.int16)
        for lane in range(9):
            signal = (
                5_000 * np.sin(2 * np.pi * (lane + 1) * raw_time)
                + 700 * np.cos(2 * np.pi * 0.37 * raw_time)
            )
            rows[:, lane + 1] = np.rint(signal).astype(np.int16)
        recording = _recording(folder, rows)
        segment = _affine_segment(
            0,
            canonical_count,
            scale=1.0003,
            intercept=10.25,
            raw_end=raw_count,
        )
        merged_analog = folder / "canonical_analogin.dat"
        merged_validity = folder / "valid_analog_samples.dat"
        write_canonical_analog(
            [recording],
            {1: (segment,)},
            master_index=0,
            canonical_rows=canonical_count,
            analog_path=merged_analog,
            validity_path=merged_validity,
        )
        filter_taps = 51
        raw = build_synchronized_imu(
            [recording],
            {1: (segment,)},
            canonical_rows=canonical_count,
            filter_taps=filter_taps,
        )
        merged = build_filtered_imu_from_merged(
            [recording],
            merged_analog,
            merged_validity,
            segments_by_device={1: (segment,)},
            canonical_rows=canonical_count,
            filter_taps=filter_taps,
        )
    raw_device = raw.devices[0]
    merged_device = merged.devices[0]
    common = raw_device.valid & merged_device.valid
    assert np.any(common)
    assert np.allclose(
        raw_device.source_rows[common],
        merged_device.source_rows[common],
        rtol=0.0,
        atol=32 * np.finfo(np.float64).eps * raw_count,
    )
    coefficients = firwin(filter_taps, cutoff=45.0, fs=1_250.0)
    adc_error_bound = 0.5 * np.sum(np.abs(coefficients))
    axis_scales = np.asarray(
        [
            *((8.0 * 9.8) / 32768.0,) * 3,
            *((2000.0 * np.pi / 180.0) / 32768.0,) * 3,
            1150.0 / 32768.0,
            1150.0 / 32768.0,
            2500.0 / 32768.0,
        ]
    )
    numerical_epsilon = 64 * np.finfo(np.float64).eps
    absolute_bound = adc_error_bound * axis_scales + numerical_epsilon
    absolute_error = np.abs(
        raw_device.raw_resampled[common] - merged_device.raw_resampled[common]
    )
    assert np.all(np.max(absolute_error, axis=0) <= absolute_bound)


def test_merged_gap_has_exact_zero_and_no_filter_leak() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        rows = _rows(3_000, (0,) * 9)
        rows[1_100, 1] = 30_000
        recording = _recording(folder, rows)
        validity = np.ones((3_000, 1), dtype=np.uint8)
        validity[1_250:1_750, 0] = 0
        analog_path, validity_path = _write_merged_pair(folder, [rows], validity)
        result = build_filtered_imu_from_merged(
            [recording],
            analog_path,
            validity_path,
            segments_by_device={1: (_segment(0, 1_250), _segment(1_750, 3_000))},
            canonical_rows=3_000,
            filter_taps=101,
        )
    device = result.devices[0]
    invalid = ~device.valid
    assert np.all(device.raw_resampled[invalid] == 0)
    after_gap = device.valid & (device.canonical_rows >= 1_750)
    assert np.any(after_gap)
    assert np.all(device.raw_resampled[after_gap, 0] == 0)


def test_merged_master_not_first_decodes_validity_order_separately_from_blocks() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        first_folder = folder / "first"
        master_folder = folder / "master"
        first_folder.mkdir()
        master_folder.mkdir()
        first_rows = _rows(1_500, (1_000,) * 9)
        master_rows = _rows(1_500, (2_000,) * 9)
        first = _recording(first_folder, first_rows, name="first")
        master = _recording(master_folder, master_rows, name="master")
        # Validity columns are master-first. Master is invalid; first is valid.
        validity = np.column_stack(
            (np.zeros(1_500, dtype=np.uint8), np.ones(1_500, dtype=np.uint8))
        )
        analog_path, validity_path = _write_merged_pair(
            folder, [first_rows, master_rows], validity
        )
        result = build_filtered_imu_from_merged(
            [first, master],
            analog_path,
            validity_path,
            segments_by_device={
                1: (_segment(0, 1_500, device=1),),
                2: (_segment(0, 1_500, device=2),),
            },
            canonical_rows=1_500,
            filter_taps=11,
            master_index=2,
        )
    assert result.devices[0].valid_count > 0
    assert result.devices[1].valid_count == 0
    first_valid = result.devices[0].valid
    assert np.all(result.devices[0].raw_resampled[first_valid] != 0)
    assert np.all(result.devices[1].raw_resampled == 0)
    assert result.provenance["validity_device_order"] == [2, 1]


def test_merged_short_valid_run_has_no_fir_support() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        rows = _rows(1_500)
        recording = _recording(folder, rows)
        validity = np.zeros((1_500, 1), dtype=np.uint8)
        validity[500:550, 0] = 1
        analog_path, validity_path = _write_merged_pair(folder, [rows], validity)
        result = build_filtered_imu_from_merged(
            [recording],
            analog_path,
            validity_path,
            segments_by_device={1: (_segment(0, 1_500),)},
            canonical_rows=1_500,
            filter_taps=101,
        )
    assert result.devices[0].valid_count == 0
    assert np.all(result.devices[0].raw_resampled == 0)
    assert np.all(result.devices[0].source_rows == -1)


def test_merged_provenance_uses_logical_names_and_mat_merged_columns() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        rows = _rows(1_500)
        recording = _recording(folder, rows)
        analog_path, validity_path = _write_merged_pair(
            folder, [rows], np.ones((1_500, 1), dtype=np.uint8)
        )
        result = build_filtered_imu_from_merged(
            [recording],
            analog_path,
            validity_path,
            segments_by_device={1: (_segment(0, 1_500),)},
            canonical_rows=1_500,
            filter_taps=11,
            mapping_hashes_by_device={1: "authoritative-hash"},
        )
        destination = write_synchronized_imu_mat(result, folder / "merged_IMU.mat")
        loaded = loadmat(destination, struct_as_record=False, squeeze_me=True)["IMU"]
    assert result.devices[0].mapping_hash == "authoritative-hash"
    assert loaded.sourceAnalogFile == "analogin.dat"
    assert np.asarray(loaded.sourceAnalogFiles).size == 0
    assert loaded.device.sourceFolder == str(recording.folder)
    assert np.array_equal(np.atleast_1d(loaded.device.analogChannelsMerged), np.arange(2, 11))
    assert str(analog_path) not in str(result.provenance)
    assert str(validity_path) not in str(result.provenance)


def test_merged_renderer_never_scans_whole_grid_per_segment_core() -> None:
    from Code.wild_preprocess.analog import imu as imu_module

    source = inspect.getsource(imu_module._render_merged_device)
    assert "np.searchsorted(canonical_grid" in source
    assert "np.flatnonzero" not in source
    assert "np.floor(canonical_grid).astype" not in source


def test_project_raw_imu_interval_covers_both_linear_endpoints_and_segment_bounds() -> None:
    segments = (_segment(0, 5), _segment(7, 12))
    # Identity mapping: raw interval [2, 4) affects canonical integer rows
    # 2 and 3. It does not manufacture support in the canonical segment gap.
    assert project_raw_imu_intervals_to_canonical(
        segments, ((2, 4),), device_index=1
    ) == ((2, 4),)


def test_project_raw_imu_interval_fractional_scale_is_conservative_and_merged() -> None:
    segment = AnalogSyncSegment(
        device_index=1,
        canonical_start_row=0,
        canonical_end_row=8,
        raw_start_row=0,
        raw_end_row=12,
        raw_scale=1.4,
        raw_intercept_rows=0.0,
        anchors=(
            AnalogSyncAnchor(0, 0.0, True, "high", "test"),
            AnalogSyncAnchor(7, 9.8, True, "high", "test"),
        ),
        confidence="high",
        publishable=True,
    )
    # Raw [2,3) is touched by c=1 (raw 1.4 -> ceil 2) and c=2
    # (raw 2.8 -> floor 2). Adjacent raw [3,4) adds no gap in canonical.
    assert project_raw_imu_intervals_to_canonical(
        (segment,), ((2, 3), (3, 4)), device_index=1
    ) == ((1, 3),)
