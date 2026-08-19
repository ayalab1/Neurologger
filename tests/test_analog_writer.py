from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from Code.wild_preprocess.analog.models import AnalogSyncAnchor, AnalogSyncSegment
from Code.wild_preprocess.analog.write import write_canonical_analog
from Code.wild_preprocess.models import Recording


def _recording(tmp_path: Path, name: str, rows: np.ndarray) -> Recording:
    folder = tmp_path / name
    folder.mkdir()
    analog = folder / "analogin.dat"
    np.asarray(rows, dtype="<i2").tofile(analog)
    return Recording(
        folder=folder,
        amplifier_file=folder / "amplifier.dat",
        analog_file=analog,
        ce_params_file=folder / "CE_params.bin",
        device_name=name,
        recording_name="session",
        fs=20_000,
        n_channels=64,
        n_samples=rows.shape[0] * 16,
        analog_channels=16,
        analog_samples=rows.shape[0],
    )


def _segment(
    device_index: int,
    canonical_start: int,
    canonical_end: int,
    raw_start: int,
    raw_end: int,
    *,
    scale: float = 1.0,
    intercept: float = 0.0,
) -> AnalogSyncSegment:
    anchors = tuple(
        AnalogSyncAnchor(
            canonical_row=row,
            raw_row=scale * row + intercept,
            verified=True,
            confidence="high",
            verification_source="test",
        )
        for row in (canonical_start, canonical_end - 1)
    )
    return AnalogSyncSegment(
        device_index=device_index,
        canonical_start_row=canonical_start,
        canonical_end_row=canonical_end,
        raw_start_row=raw_start,
        raw_end_row=raw_end,
        raw_scale=scale,
        raw_intercept_rows=intercept,
        anchors=anchors,
        confidence="high",
        publishable=True,
    )


def _rows(n_rows: int, offset: int = 0) -> np.ndarray:
    result = np.zeros((n_rows, 16), dtype=np.int16)
    for lane in range(16):
        result[:, lane] = offset + lane * 100 + np.arange(n_rows, dtype=np.int16)
    return result


def _read(path: Path, rows: int, columns: int, dtype: str = "<i2") -> np.ndarray:
    return np.fromfile(path, dtype=dtype).reshape(rows, columns)


def test_clean_identity_writes_master_first_blocks_and_validity(tmp_path: Path) -> None:
    first = _recording(tmp_path, "first", _rows(5, 10))
    master = _recording(tmp_path, "master", _rows(5, 1000))
    result = write_canonical_analog(
        [first, master],
        {1: (_segment(1, 0, 5, 0, 5),), 2: (_segment(2, 0, 5, 0, 5),)},
        master_index=1,
        canonical_rows=5,
        analog_path=tmp_path / "merged_analogin.dat",
        validity_path=tmp_path / "valid_analog_samples.dat",
        chunk_rows=2,
    )
    rendered = _read(result.analog_path, 5, 32)
    assert np.array_equal(rendered[:, :16], _rows(5, 10))
    assert np.array_equal(rendered[:, 16:], _rows(5, 1000))
    assert np.array_equal(_read(result.validity_path, 5, 2, "u1"), np.ones((5, 2), dtype=np.uint8))
    assert result.channel_device_order == (0, 1)
    assert result.validity_device_order == (1, 0)
    assert result.valid_rows_by_device == (5, 5)
    assert result.analog_bytes == 5 * 32 * 2
    assert result.validity_bytes == 10


def test_fractional_continuous_lanes_interpolate_but_discrete_lanes_are_nearest(tmp_path: Path) -> None:
    source = _rows(6)
    source[:, 0] = np.arange(6, dtype=np.int16) * 100
    source[:, 1] = np.arange(6, dtype=np.int16) * 10
    recording = _recording(tmp_path, "one", source)
    result = write_canonical_analog(
        [recording],
        {1: (_segment(1, 0, 4, 0, 6, scale=1.4),)},
        master_index=0,
        canonical_rows=4,
        analog_path=tmp_path / "analogin.dat",
        validity_path=tmp_path / "valid_analog_samples.dat",
    )
    rendered = _read(result.analog_path, 4, 16)
    assert rendered[:, 1].tolist() == [0, 14, 28, 42]
    assert rendered[:, 0].tolist() == [0, 100, 300, 400]
    assert np.all(_read(result.validity_path, 4, 1, "u1") == 1)


def test_gap_is_zeroed_and_not_interpolated_across_segment_boundary(tmp_path: Path) -> None:
    source = _rows(5)
    source[:, 1] = np.array([10, 20, 30, 40, 50], dtype=np.int16)
    recording = _recording(tmp_path, "one", source)
    segments = (
        _segment(1, 0, 2, 0, 2),
        _segment(1, 3, 5, 3, 5),
    )
    result = write_canonical_analog(
        [recording],
        {1: segments},
        master_index=0,
        canonical_rows=5,
        analog_path=tmp_path / "analogin.dat",
        validity_path=tmp_path / "valid_analog_samples.dat",
        chunk_rows=1,
    )
    rendered = _read(result.analog_path, 5, 16)
    validity = _read(result.validity_path, 5, 1, "u1").ravel()
    assert validity.tolist() == [1, 1, 0, 1, 1]
    assert np.array_equal(rendered[2], np.zeros(16, dtype=np.int16))
    assert rendered[:, 1].tolist() == [10, 20, 0, 40, 50]


def test_fractional_segment_endpoint_without_two_row_support_is_zeroed(tmp_path: Path) -> None:
    recording = _recording(tmp_path, "one", _rows(2))
    result = write_canonical_analog(
        [recording],
        {1: (_segment(1, 0, 2, 0, 1, scale=0.5),)},
        master_index=0,
        canonical_rows=2,
        analog_path=tmp_path / "analogin.dat",
        validity_path=tmp_path / "valid_analog_samples.dat",
    )
    rendered = _read(result.analog_path, 2, 16)
    validity = _read(result.validity_path, 2, 1, "u1").ravel()
    assert validity.tolist() == [1, 0]
    assert np.array_equal(rendered[1], np.zeros(16, dtype=np.int16))


def test_exact_last_source_row_needs_no_unused_upper_interpolation_row(tmp_path: Path) -> None:
    recording = _recording(tmp_path, "one", _rows(2))
    result = write_canonical_analog(
        [recording],
        {1: (_segment(1, 0, 2, 0, 2),)},
        master_index=0,
        canonical_rows=2,
        analog_path=tmp_path / "analogin.dat",
        validity_path=tmp_path / "valid_analog_samples.dat",
    )
    assert np.array_equal(_read(result.validity_path, 2, 1, "u1"), np.ones((2, 1), dtype=np.uint8))
    assert np.array_equal(_read(result.analog_path, 2, 16), _rows(2))


def test_byte_counts_and_rendering_are_deterministic(tmp_path: Path) -> None:
    recording = _recording(tmp_path, "one", _rows(6))
    kwargs = dict(
        recordings=[recording],
        segments_by_device={1: (_segment(1, 0, 6, 0, 6),)},
        master_index=0,
        canonical_rows=6,
        validity_path=tmp_path / "valid_a.dat",
        chunk_rows=2,
    )
    first = write_canonical_analog(analog_path=tmp_path / "a.dat", **kwargs)
    second = write_canonical_analog(
        analog_path=tmp_path / "b.dat",
        validity_path=tmp_path / "valid_b.dat",
        recordings=[recording],
        segments_by_device={1: (_segment(1, 0, 6, 0, 6),)},
        master_index=0,
        canonical_rows=6,
        chunk_rows=5,
    )
    assert first.analog_path.read_bytes() == second.analog_path.read_bytes()
    assert first.validity_path.read_bytes() == second.validity_path.read_bytes()
    assert first.analog_path.stat().st_size == first.analog_bytes
    assert first.validity_path.stat().st_size == first.validity_bytes


def test_lane_local_corruption_zeroes_only_that_lane_without_changing_validity(
    tmp_path: Path,
) -> None:
    source = _rows(6, 10)
    recording = _recording(tmp_path, "one", source)
    result = write_canonical_analog(
        [recording],
        {1: (_segment(1, 0, 6, 0, 6),)},
        master_index=0,
        canonical_rows=6,
        analog_path=tmp_path / "analogin.dat",
        validity_path=tmp_path / "valid_analog_samples.dat",
        chunk_rows=2,
        invalid_lane_intervals_by_device={1: {11: ((2, 4),)}},
    )
    rendered = _read(result.analog_path, 6, 16)
    assert np.array_equal(_read(result.validity_path, 6, 1, "u1"), np.ones((6, 1), dtype=np.uint8))
    assert np.array_equal(rendered[:, :11], source[:, :11])
    assert np.array_equal(rendered[:, 12:], source[:, 12:])
    assert rendered[:, 11].tolist() == [1110, 1111, 0, 0, 1114, 1115]


def test_continuous_lane_exclusion_checks_both_interpolation_endpoints(tmp_path: Path) -> None:
    source = _rows(6)
    source[:, 1] = np.arange(6, dtype=np.int16) * 10 + 10
    recording = _recording(tmp_path, "one", source)
    result = write_canonical_analog(
        [recording],
        {1: (_segment(1, 0, 4, 0, 6, scale=1.4),)},
        master_index=0,
        canonical_rows=4,
        analog_path=tmp_path / "analogin.dat",
        validity_path=tmp_path / "valid_analog_samples.dat",
        invalid_lane_intervals_by_device={1: {1: ((2, 3),)}},
    )
    rendered = _read(result.analog_path, 4, 16)
    # Raw coordinates are 0, 1.4, 2.8, 4.2. The interval touches the upper
    # endpoint of 1.4 and the lower endpoint of 2.8, but neither neighbour.
    assert rendered[:, 1].tolist() == [10, 0, 0, 52]
    assert np.array_equal(_read(result.validity_path, 4, 1, "u1"), np.ones((4, 1), dtype=np.uint8))


def test_invalid_device_mapping_or_mapper_output_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recording = _recording(tmp_path, "one", _rows(2))
    with pytest.raises(ValueError, match="same device"):
        write_canonical_analog(
            [recording],
            {1: (_segment(2, 0, 2, 0, 2),)},
            master_index=0,
            canonical_rows=2,
            analog_path=tmp_path / "bad.dat",
            validity_path=tmp_path / "bad_valid.dat",
        )

    def malformed(*args: object, **kwargs: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.array([np.nan, 1.0]), np.array([True, True]), np.array([0, 0])

    monkeypatch.setattr("Code.wild_preprocess.analog.write.map_canonical_rows", malformed)
    with pytest.raises(ValueError, match="non-finite"):
        write_canonical_analog(
            [recording],
            {1: (_segment(1, 0, 2, 0, 2),)},
            master_index=0,
            canonical_rows=2,
            analog_path=tmp_path / "bad2.dat",
            validity_path=tmp_path / "bad2_valid.dat",
        )


def test_second_direct_promotion_failure_restores_existing_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = _recording(tmp_path, "one", _rows(2))
    analog_path = tmp_path / "analogin.dat"
    validity_path = tmp_path / "valid_analog_samples.dat"
    old_analog = b"old analog pair"
    old_validity = b"old validity pair"
    analog_path.write_bytes(old_analog)
    validity_path.write_bytes(old_validity)

    from Code.wild_preprocess.analog import write as writer_module

    original_replace = writer_module.replace_atomic

    def fail_validity_promotion(partial: Path, output: Path) -> None:
        if output == validity_path:
            raise OSError("simulated second promotion failure")
        original_replace(partial, output)

    monkeypatch.setattr(writer_module, "replace_atomic", fail_validity_promotion)
    with pytest.raises(OSError, match="second promotion"):
        write_canonical_analog(
            [recording],
            {1: (_segment(1, 0, 2, 0, 2),)},
            master_index=0,
            canonical_rows=2,
            analog_path=analog_path,
            validity_path=validity_path,
            overwrite=True,
        )
    assert analog_path.read_bytes() == old_analog
    assert validity_path.read_bytes() == old_validity
    assert not (tmp_path / "analogin.dat.partial").exists()
    assert not (tmp_path / "valid_analog_samples.dat.partial").exists()
    assert not (tmp_path / "analogin.dat.previous").exists()
    assert not (tmp_path / "valid_analog_samples.dat.previous").exists()
