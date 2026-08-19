"""Bounded writer for the canonical 1250 Hz analog timeline.

This module intentionally only writes staged files.  Transaction promotion,
manifest updates, and GUI orchestration belong to the pipeline layer.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from ..binary_io import atomic_output_path, close_memmap, interleaved_memmap, replace_atomic
from ..models import Recording
from .models import AnalogSyncSegment
from .segments import _map_validated_canonical_rows, validate_analog_segment_collection

# Local indirection preserves the writer's structural test seam while using a
# collection validated once by ``_coerce_segment_sets`` rather than once per
# output chunk.
map_canonical_rows = _map_validated_canonical_rows


ProgressCallback = Callable[[str, float], None]
DEFAULT_DISCRETE_LANES = frozenset({0, 11, 14, 15})
DEFAULT_CONTINUOUS_LANES = frozenset(range(1, 10))

LaneInvalidIntervals = Mapping[int, Iterable[tuple[int, int]]]
DeviceLaneInvalidIntervals = Mapping[int, LaneInvalidIntervals]


@dataclass(frozen=True)
class CanonicalAnalogWriteResult:
    """Small, transaction-ready summary of a staged analog render."""

    analog_path: Path
    validity_path: Path
    canonical_rows: int
    device_count: int
    channels_per_device: int
    channel_device_order: tuple[int, ...]
    validity_device_order: tuple[int, ...]
    analog_bytes: int
    validity_bytes: int
    valid_rows_by_device: tuple[int, ...]
    common_valid_rows: int


def _output_device_order(device_count: int, master_index: int) -> tuple[int, ...]:
    if not 0 <= master_index < device_count:
        raise ValueError("master_index is outside recordings")
    return (master_index,) + tuple(index for index in range(device_count) if index != master_index)


def _coerce_segment_sets(
    segments_by_device: Mapping[int, Iterable[AnalogSyncSegment]]
    | Sequence[Iterable[AnalogSyncSegment]],
    *,
    device_count: int,
) -> tuple[tuple[AnalogSyncSegment, ...], ...]:
    """Return source-order segment sets, keyed by their one-based device id."""

    if isinstance(segments_by_device, Mapping):
        expected = set(range(1, device_count + 1))
        actual = set(segments_by_device)
        if actual != expected:
            raise ValueError(
                "segments_by_device mapping keys must be exactly the one-based "
                f"recording device ids {sorted(expected)}"
            )
        source_order = tuple(segments_by_device[index] for index in range(1, device_count + 1))
    else:
        source_order = tuple(segments_by_device)
        if len(source_order) != device_count:
            raise ValueError("segments_by_device sequence must align with recordings")
    return tuple(
        validate_analog_segment_collection(segments, device_index=device_index)
        for device_index, segments in enumerate(source_order, start=1)
    )


def _coerce_invalid_lane_intervals(
    intervals_by_device: DeviceLaneInvalidIntervals | None,
    *,
    recordings: Sequence[Recording],
) -> tuple[dict[int, tuple[tuple[int, int], ...]], ...]:
    """Validate sparse one-based device/lane raw-row exclusions."""

    normalized: list[dict[int, tuple[tuple[int, int], ...]]] = [
        {} for _ in recordings
    ]
    if intervals_by_device is None:
        return tuple(normalized)
    valid_devices = set(range(1, len(recordings) + 1))
    unknown_devices = set(intervals_by_device) - valid_devices
    if unknown_devices:
        raise ValueError(
            "invalid_lane_intervals_by_device contains unknown one-based device ids: "
            f"{sorted(unknown_devices)}"
        )
    for device_id, by_lane in intervals_by_device.items():
        raw_row_count = recordings[device_id - 1].analog_samples
        for lane, intervals in by_lane.items():
            if not isinstance(lane, int) or lane < 0 or lane >= 16:
                raise ValueError("invalid analog lane must be a zero-based integer in [0, 16)")
            runs = tuple((int(start), int(end)) for start, end in intervals)
            previous_end = -1
            for start, end in runs:
                if start < 0 or end <= start or end > raw_row_count:
                    raise ValueError(
                        "invalid lane intervals must be non-empty half-open raw-row intervals "
                        "within the source file"
                    )
                if start < previous_end:
                    raise ValueError("invalid lane intervals must be sorted and non-overlapping")
                previous_end = end
            normalized[device_id - 1][lane] = runs
    return tuple(normalized)


def _validate_arguments(
    recordings: Sequence[Recording],
    *,
    master_index: int,
    canonical_rows: int,
    analog_path: Path,
    validity_path: Path,
    chunk_rows: int,
    overwrite: bool,
    continuous_lanes: Iterable[int],
) -> tuple[tuple[Recording, ...], tuple[int, ...]]:
    normalized = tuple(recordings)
    if not normalized:
        raise ValueError("at least one recording is required")
    if not isinstance(canonical_rows, int) or canonical_rows <= 0:
        raise ValueError("canonical_rows must be a positive integer")
    if not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    if analog_path == validity_path:
        raise ValueError("analog_path and validity_path must be distinct")
    if not overwrite and (analog_path.exists() or validity_path.exists()):
        existing = analog_path if analog_path.exists() else validity_path
        raise FileExistsError(f"Output exists; enable overwrite to regenerate: {existing}")
    for recording in normalized:
        if recording.analog_channels != 16:
            raise ValueError(
                "canonical analog writer requires the WILD 16-channel analog frame contract; "
                f"{recording.folder} has {recording.analog_channels} channels"
            )
        if recording.analog_samples <= 0:
            raise ValueError(f"analog source contains no rows: {recording.analog_file}")
    lanes = tuple(int(lane) for lane in continuous_lanes)
    if len(set(lanes)) != len(lanes) or any(lane < 0 or lane >= 16 for lane in lanes):
        raise ValueError("continuous_lanes must contain unique 0-based WILD analog lanes")
    return normalized, _output_device_order(len(normalized), master_index)


def _verify_mapping(
    raw_rows: np.ndarray,
    valid: np.ndarray,
    segment_ids: np.ndarray,
    *,
    last_valid_raw_row: float | None,
) -> float | None:
    """Reject malformed mapping output before it can be published as valid."""

    if raw_rows.ndim != 1 or valid.ndim != 1 or segment_ids.ndim != 1:
        raise ValueError("analog mapping must return one-dimensional arrays")
    if not (raw_rows.size == valid.size == segment_ids.size):
        raise ValueError("analog mapping result arrays have inconsistent lengths")
    if np.any((valid & ~np.isfinite(raw_rows)) | (valid & (segment_ids < 0))):
        raise ValueError("valid analog mapping contains non-finite or unassigned source coordinates")
    valid_rows = raw_rows[valid]
    if valid_rows.size and np.any(np.diff(valid_rows) <= 0):
        raise ValueError("valid analog mapping is not strictly monotone")
    if valid_rows.size and last_valid_raw_row is not None and valid_rows[0] <= last_valid_raw_row:
        raise ValueError("valid analog mapping is not strictly monotone across chunks")
    return float(valid_rows[-1]) if valid_rows.size else last_valid_raw_row


def _render_device_chunk(
    source: np.memmap,
    raw_rows: np.ndarray,
    valid: np.ndarray,
    *,
    continuous_lanes: tuple[int, ...],
    invalid_lane_intervals: Mapping[int, tuple[tuple[int, int], ...]],
) -> np.ndarray:
    """Render one device block without ever sampling an invalid coordinate."""

    result = np.zeros((raw_rows.size, 16), dtype=np.int16)
    positions = np.flatnonzero(valid)
    if not positions.size:
        return result
    coordinates = raw_rows[positions]
    lower = np.floor(coordinates).astype(np.int64)
    nearest = np.rint(coordinates).astype(np.int64)
    # Forward mapping requested one-row interpolation support.  These checks
    # are duplicated here as a structural guard against a malformed mapper.
    upper = np.ceil(coordinates).astype(np.int64)
    if (
        np.any(lower < 0)
        or np.any(upper >= source.shape[0])
        or np.any(nearest < 0)
        or np.any(nearest >= source.shape[0])
    ):
        raise ValueError("valid analog mapping lacks complete in-range source support")

    weights = coordinates - lower
    if continuous_lanes:
        lower_values = np.asarray(source[lower][:, continuous_lanes], dtype=np.float64)
        upper_values = np.asarray(source[upper][:, continuous_lanes], dtype=np.float64)
        interpolated = lower_values + (upper_values - lower_values) * weights[:, None]
        result[np.ix_(positions, continuous_lanes)] = np.clip(
            np.rint(interpolated), np.iinfo(np.int16).min, np.iinfo(np.int16).max
        ).astype(np.int16)
    discrete_lanes = tuple(lane for lane in range(16) if lane not in continuous_lanes)
    if discrete_lanes:
        result[np.ix_(positions, discrete_lanes)] = np.asarray(
            source[nearest][:, discrete_lanes], dtype=np.int16
        )
    for lane, intervals in invalid_lane_intervals.items():
        touched = np.zeros(coordinates.size, dtype=bool)
        support_start = lower if lane in continuous_lanes else nearest
        support_end = upper if lane in continuous_lanes else nearest
        for interval_start, interval_end in intervals:
            touched |= (
                ((support_start >= interval_start) & (support_start < interval_end))
                | ((support_end >= interval_start) & (support_end < interval_end))
            )
        if np.any(touched):
            result[positions[touched], lane] = 0
    return result


def _backup_path(path: Path) -> Path:
    return path.with_name(path.name + ".previous")


def _promote_pair(
    analog_partial: Path,
    validity_partial: Path,
    analog_path: Path,
    validity_path: Path,
) -> None:
    """Promote two direct-call outputs, restoring the old pair on failure."""

    backups = ((analog_path, _backup_path(analog_path)), (validity_path, _backup_path(validity_path)))
    moved_backups: list[tuple[Path, Path]] = []
    promoted: list[Path] = []
    try:
        for output, backup in backups:
            if backup.exists():
                backup.unlink()
            if output.exists():
                os.replace(output, backup)
                moved_backups.append((output, backup))
        replace_atomic(analog_partial, analog_path)
        promoted.append(analog_path)
        replace_atomic(validity_partial, validity_path)
        promoted.append(validity_path)
    except Exception:
        for output in promoted:
            if output.exists():
                output.unlink()
        for output, backup in moved_backups:
            if backup.exists():
                os.replace(backup, output)
        raise
    else:
        for _, backup in moved_backups:
            if backup.exists():
                backup.unlink()


def write_canonical_analog(
    recordings: Sequence[Recording],
    segments_by_device: Mapping[int, Iterable[AnalogSyncSegment]]
    | Sequence[Iterable[AnalogSyncSegment]],
    *,
    master_index: int,
    canonical_rows: int,
    analog_path: str | Path,
    validity_path: str | Path,
    chunk_rows: int = 62_500,
    overwrite: bool = False,
    progress: ProgressCallback | None = None,
    continuous_lanes: Iterable[int] = DEFAULT_CONTINUOUS_LANES,
    invalid_lane_intervals_by_device: DeviceLaneInvalidIntervals | None = None,
    staged: bool = False,
) -> CanonicalAnalogWriteResult:
    """Write canonical ``analogin.dat`` and device validity sidecar atomically.

    The output is sample-major, 16 ``int16`` columns per device.  Device
    blocks retain the historical input recording order.  Sidecar columns are
    master-first, followed by the remaining recording order.  Only known IMU
    lanes 1--9 (zero-based) are linearly interpolated by default; every other
    lane is nearest-neighbour unless explicitly included in
    ``continuous_lanes``.  A zero sidecar bit means the complete corresponding
    16-channel output block is zero; a one is emitted only for finite,
    strictly monotone, in-range mapping coordinates with same-segment linear
    interpolation support.  Lane-local raw corruption may be supplied through
    ``invalid_lane_intervals_by_device``; it zeros only the affected lane and
    deliberately does not change the device-level mapping-valid sidecar.
    """

    analog_path, validity_path = Path(analog_path), Path(validity_path)
    normalized, device_order = _validate_arguments(
        recordings,
        master_index=master_index,
        canonical_rows=canonical_rows,
        analog_path=analog_path,
        validity_path=validity_path,
        chunk_rows=chunk_rows,
        overwrite=overwrite,
        continuous_lanes=continuous_lanes,
    )
    continuous = tuple(sorted(int(lane) for lane in continuous_lanes))
    segment_sets = _coerce_segment_sets(segments_by_device, device_count=len(normalized))
    invalid_lane_intervals = _coerce_invalid_lane_intervals(
        invalid_lane_intervals_by_device, recordings=normalized
    )
    analog_path.parent.mkdir(parents=True, exist_ok=True)
    validity_path.parent.mkdir(parents=True, exist_ok=True)
    analog_partial = atomic_output_path(analog_path)
    validity_partial = atomic_output_path(validity_path)
    if not staged:
        for partial in (analog_partial, validity_partial):
            if partial.exists():
                partial.unlink()

    valid_counts = np.zeros(len(normalized), dtype=np.int64)
    common_valid_rows = 0
    last_valid_positions: list[float | None] = [None] * len(normalized)
    sources: list[np.memmap] = []
    succeeded = False
    try:
        with ExitStack() as stack:
            for recording in normalized:
                sources.append(
                    interleaved_memmap(
                        recording.analog_file,
                        recording.analog_channels,
                        recording.analog_samples,
                    )
                )
            analog_stream = stack.enter_context((analog_path if staged else analog_partial).open("wb"))
            validity_stream = stack.enter_context((validity_path if staged else validity_partial).open("wb"))
            for start in range(0, canonical_rows, chunk_rows):
                count = min(chunk_rows, canonical_rows - start)
                canonical = start + np.arange(count, dtype=np.int64)
                rendered: dict[int, np.ndarray] = {}
                validity = np.zeros((count, len(normalized)), dtype=np.uint8)
                for device_index, (recording, source, segments) in enumerate(
                    zip(normalized, sources, segment_sets)
                ):
                    raw_rows, valid, segment_ids = map_canonical_rows(
                        segments,
                        canonical,
                        raw_row_count=recording.analog_samples,
                        interpolation_half_width=1,
                    )
                    last_valid_positions[device_index] = _verify_mapping(
                        raw_rows,
                        valid,
                        segment_ids,
                        last_valid_raw_row=last_valid_positions[device_index],
                    )
                    rendered[device_index] = _render_device_chunk(
                        source,
                        raw_rows,
                        valid,
                        continuous_lanes=continuous,
                        invalid_lane_intervals=invalid_lane_intervals[device_index],
                    )
                    validity[:, device_index] = valid.astype(np.uint8, copy=False)
                    valid_counts[device_index] += int(np.count_nonzero(valid))
                combined = np.concatenate([rendered[index] for index in range(len(normalized))], axis=1)
                combined.astype("<i2", copy=False).tofile(analog_stream)
                ordered_validity = validity[:, device_order]
                ordered_validity.tofile(validity_stream)
                common_valid_rows += int(np.count_nonzero(np.all(ordered_validity != 0, axis=1)))
                if progress is not None:
                    progress("write_analog", 100.0 * (start + count) / canonical_rows)
        if not staged:
            _promote_pair(analog_partial, validity_partial, analog_path, validity_path)
        succeeded = True
    finally:
        for source in sources:
            close_memmap(source)
        if staged and not succeeded:
            for output in (analog_path, validity_path):
                if output.exists():
                    output.unlink()
        for partial in (analog_partial, validity_partial):
            if partial.exists():
                partial.unlink()

    return CanonicalAnalogWriteResult(
        analog_path=analog_path,
        validity_path=validity_path,
        canonical_rows=canonical_rows,
        device_count=len(normalized),
        channels_per_device=16,
        channel_device_order=tuple(range(len(normalized))),
        validity_device_order=device_order,
        analog_bytes=canonical_rows * len(normalized) * 16 * np.dtype("<i2").itemsize,
        validity_bytes=canonical_rows * len(normalized),
        valid_rows_by_device=tuple(int(valid_counts[index]) for index in device_order),
        common_valid_rows=int(common_valid_rows),
    )
