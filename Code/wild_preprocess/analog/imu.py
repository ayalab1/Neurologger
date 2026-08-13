"""Gap-safe IMU extraction on the canonical analog timeline.

The preferred path consumes the staged canonical ``analogin.dat`` and its
``valid_analog_samples.dat`` sidecar, ensuring IMU generation uses exactly the
mapping that will be published.  The original raw-device path remains for
backward compatibility and numerical reference comparisons.  Both paths
filter only within contiguous valid support and never bridge a gap.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.io import savemat
from scipy.signal import convolve, firwin

from ..binary_io import close_memmap, interleaved_memmap
from ..models import Recording
from .models import AnalogSyncSegment
from .segments import map_canonical_rows, validate_analog_segment_collection
from .imu_fusion import fuse_imu_ahrs
from .imu_preprocess import (
    IMU_RESAMPLE_DOWN,
    IMU_RESAMPLE_UP,
    scale_imu_nominal,
    resample_imu_1250_to_100,
)
from .imu_motion import compute_imu_motion


RAW_ANALOG_SAMPLE_RATE_HZ = 1_250.0
DEFAULT_IMU_SAMPLE_RATE_HZ = 100.0
IMU_CHANNELS_ONE_BASED = tuple(range(2, 11))
IMU_LANES_ZERO_BASED = tuple(channel - 1 for channel in IMU_CHANNELS_ONE_BASED)
DEFAULT_FILTER_TAPS = 251
DEFAULT_FILTER_CUTOFF_HZ = 45.0
DEFAULT_CHUNK_ROWS = 125_000
DEFAULT_MAX_OUTPUT_SAMPLES = 2_000_000
DEFAULT_MAX_PEAK_BYTES = 4 * 1024**3
AHRS_WARMUP_SAMPLES = 1


@dataclass(frozen=True)
class ImuSensorData:
    """Scaled, temporally synchronized IMU axes, each shaped ``(N, 3)``."""

    acc: np.ndarray
    gyr: np.ndarray
    mag: np.ndarray


@dataclass(frozen=True)
class ImuFusionData:
    """MATLAB-compatible AHRS and derived motion fields."""

    quaternion: np.ndarray
    orientation: np.ndarray
    acceleration: np.ndarray
    speed: np.ndarray
    valid: np.ndarray
    status: str
    method: str
    warning: str = ""


@dataclass(frozen=True)
class SynchronizedImuDevice:
    """One device on the common, never-restarted 100 Hz canonical grid."""

    device_index: int
    device_name: str
    recording_name: str
    source_folder: Path
    raw_resampled: np.ndarray
    imu: ImuSensorData
    valid: np.ndarray
    source_rows: np.ndarray
    canonical_rows: np.ndarray
    mapping_hash: str
    valid_count: int
    valid_fraction: float
    status: str
    provenance: dict[str, object]
    fusion: ImuFusionData | None = None


@dataclass(frozen=True)
class SynchronizedImuResult:
    """Canonical 100 Hz IMU result.

    The result intentionally has an explicit in-memory bound.  At the default
    maximum of two million 100 Hz samples, one device's 9-axis float64 output
    consumes about 137 MiB before MAT serialization; requested fusion adds
    quaternion, rotation, acceleration, speed, and validity arrays. Longer
    recordings must use a future streaming output interface rather than
    silently allocating an unbounded array.
    """

    sample_rate_hz: float
    raw_sample_rate_hz: float
    canonical_rows: np.ndarray
    time_seconds: np.ndarray
    devices: tuple[SynchronizedImuDevice, ...]
    master_index: int
    master_start_sample: int
    master_start_sec: float
    status: str
    provenance: dict[str, object]


def _normalise_segment_sets(
    segments_by_device: Mapping[int, Iterable[AnalogSyncSegment]]
    | Sequence[Iterable[AnalogSyncSegment]],
    *,
    device_count: int,
) -> tuple[tuple[AnalogSyncSegment, ...], ...]:
    if isinstance(segments_by_device, Mapping):
        expected = set(range(1, device_count + 1))
        if set(segments_by_device) != expected:
            raise ValueError("segments_by_device must use exactly one-based recording device ids")
        values = tuple(segments_by_device[index] for index in range(1, device_count + 1))
    else:
        values = tuple(segments_by_device)
        if len(values) != device_count:
            raise ValueError("segments_by_device must align with recordings")
    return tuple(
        validate_analog_segment_collection(entries, device_index=index)
        for index, entries in enumerate(values, start=1)
    )


def _interval_bounds(interval: object) -> tuple[int, int]:
    """Read one half-open raw interval without depending on integrity types."""

    if isinstance(interval, tuple) and len(interval) == 2:
        start, end = interval
    elif isinstance(interval, Mapping):
        start, end = interval.get("raw_start_row", interval.get("raw_start")), interval.get(
            "raw_end_row", interval.get("raw_end")
        )
    else:
        start = getattr(interval, "raw_start_row", getattr(interval, "raw_start", None))
        end = getattr(interval, "raw_end_row", getattr(interval, "raw_end", None))
    if not isinstance(start, (int, np.integer)) or not isinstance(end, (int, np.integer)):
        raise ValueError("IMU modality intervals require integer raw_start/raw_end bounds")
    if start < 0 or end <= start:
        raise ValueError("IMU modality intervals must be non-empty half-open raw intervals")
    return int(start), int(end)


def _normalise_invalid_raw_intervals(
    invalid_raw_intervals_by_device: Mapping[int, Iterable[object]] | Sequence[Iterable[object]] | None,
    *,
    recordings: Sequence[Recording],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Normalize modality-only exclusions, merging overlaps deterministically.

    These intervals represent sensor quality (for example ``sensor_stall``),
    not temporal mapping authority.  They are deliberately not fed back into
    the analog writer's ``valid_analog_samples.dat`` mask.
    """

    device_count = len(recordings)
    if invalid_raw_intervals_by_device is None:
        entries: tuple[Iterable[object], ...] = tuple(() for _ in recordings)
    elif isinstance(invalid_raw_intervals_by_device, Mapping):
        expected = set(range(1, device_count + 1))
        if set(invalid_raw_intervals_by_device) != expected:
            raise ValueError("invalid_raw_intervals_by_device must use exactly one-based recording ids")
        entries = tuple(invalid_raw_intervals_by_device[index] for index in range(1, device_count + 1))
    else:
        entries = tuple(invalid_raw_intervals_by_device)
        if len(entries) != device_count:
            raise ValueError("invalid_raw_intervals_by_device must align with recordings")
    normalized: list[tuple[tuple[int, int], ...]] = []
    for recording, device_entries in zip(recordings, entries):
        bounds = sorted(_interval_bounds(entry) for entry in device_entries)
        merged: list[tuple[int, int]] = []
        for start, end in bounds:
            if end > recording.analog_samples:
                raise ValueError("IMU modality interval exceeds raw analog support")
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        normalized.append(tuple(merged))
    return tuple(normalized)


def _validate_parameters(
    recordings: Sequence[Recording],
    *,
    canonical_rows: int,
    output_sample_rate_hz: float,
    filter_taps: int,
    filter_cutoff_hz: float,
    chunk_rows: int,
    max_output_samples: int,
    master_index: int,
    master_start_sample: int,
    master_start_sec: float,
) -> tuple[Recording, ...]:
    normalized = tuple(recordings)
    if not normalized:
        raise ValueError("at least one recording is required")
    if not isinstance(canonical_rows, int) or canonical_rows <= 0:
        raise ValueError("canonical_rows must be a positive integer")
    if not np.isfinite(output_sample_rate_hz) or not 0 < output_sample_rate_hz < RAW_ANALOG_SAMPLE_RATE_HZ:
        raise ValueError("output_sample_rate_hz must be in (0, 1250)")
    if not isinstance(filter_taps, int) or filter_taps < 3 or filter_taps % 2 == 0:
        raise ValueError("filter_taps must be an odd integer of at least three")
    if not np.isfinite(filter_cutoff_hz) or not 0 < filter_cutoff_hz < output_sample_rate_hz / 2:
        raise ValueError("filter_cutoff_hz must be below the output Nyquist frequency")
    if not isinstance(chunk_rows, int) or chunk_rows <= filter_taps:
        raise ValueError("chunk_rows must exceed filter_taps")
    if not isinstance(max_output_samples, int) or max_output_samples <= 0:
        raise ValueError("max_output_samples must be a positive integer")
    if not isinstance(master_index, int) or not 1 <= master_index <= len(normalized):
        raise ValueError("master_index must be a one-based index into recordings")
    if not isinstance(master_start_sample, int) or master_start_sample < 0:
        raise ValueError("master_start_sample must be a non-negative integer")
    if not np.isfinite(master_start_sec) or master_start_sec < 0:
        raise ValueError("master_start_sec must be finite and non-negative")
    for recording in normalized:
        if recording.analog_channels < 10:
            raise ValueError(f"analogin.dat has fewer than one-based channel 10: {recording.analog_file}")
        if recording.analog_samples <= 0:
            raise ValueError(f"analogin.dat contains no rows: {recording.analog_file}")
    return normalized


def _output_grid(canonical_rows: int, output_sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Return global output times and their fractional 1250 Hz row positions."""

    step_rows = RAW_ANALOG_SAMPLE_RATE_HZ / float(output_sample_rate_hz)
    # Include exactly those samples whose canonical coordinate is inside the
    # half-open analog recording support.  The phase is always tied to row 0.
    count = int(np.floor((canonical_rows - 1) / step_rows)) + 1
    canonical = np.arange(count, dtype=np.float64) * step_rows
    time_seconds = np.arange(count, dtype=np.float64) / float(output_sample_rate_hz)
    return canonical, time_seconds


def _scaled_imu(values: np.ndarray) -> np.ndarray:
    """Apply the legacy MATLAB axis scales without calibration or fusion."""

    result = np.asarray(values, dtype=np.float64).copy()
    result[0:3] *= (8.0 * 9.8) / 32768.0
    result[3:6] *= (2000.0 * np.pi / 180.0) / 32768.0
    result[6:9] *= np.asarray((1150.0, 1150.0, 2500.0), dtype=np.float64)[:, None] / 32768.0
    return result


def _mapping_digest(segments: Sequence[AnalogSyncSegment]) -> str:
    encoded = json.dumps(
        [segment.to_dict() for segment in segments], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def project_raw_imu_intervals_to_canonical(
    segments: Iterable[AnalogSyncSegment],
    invalid_raw_intervals: Iterable[object],
    *,
    device_index: int | None = None,
) -> tuple[tuple[int, int], ...]:
    """Project raw IMU exclusions onto integer canonical analog rows.

    A canonical row is excluded when either endpoint used by linear sampling
    (``floor(raw)`` or ``ceil(raw)``) belongs to a raw half-open interval.
    For raw interval ``[s, e)`` this is exactly ``s - 1 < raw < e``.  The
    result is split at publishable segment boundaries and overlapping or
    adjacent projected intervals are merged deterministically.
    """

    ordered = validate_analog_segment_collection(segments, device_index=device_index)
    raw_intervals = sorted(_interval_bounds(interval) for interval in invalid_raw_intervals)
    projected: list[tuple[int, int]] = []
    for segment in ordered:
        if not segment.is_publishable:
            continue
        scale = segment.raw_scale
        intercept = segment.raw_intercept_rows
        for raw_start, raw_end in raw_intervals:
            # Integer c must satisfy:
            #   scale*c + intercept > raw_start - 1
            #   scale*c + intercept < raw_end
            start = math.floor((raw_start - 1 - intercept) / scale) + 1
            end = math.ceil((raw_end - intercept) / scale)
            start = max(int(start), segment.canonical_start_row)
            end = min(int(end), segment.canonical_end_row)
            touches = lambda row: (
                raw_start - 1 < scale * row + intercept < raw_end
            )
            # The affine bound is exact over reals; these local checks remove
            # any binary-float off-by-one at an integer boundary.
            while start < end and not touches(start):
                start += 1
            while start > segment.canonical_start_row and touches(start - 1):
                start -= 1
            while end > start and not touches(end - 1):
                end -= 1
            while end < segment.canonical_end_row and touches(end):
                end += 1
            if end > start:
                projected.append((start, end))
    projected.sort()
    merged: list[tuple[int, int]] = []
    for start, end in projected:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _iter_valid_runs(
    temporal_valid: np.ndarray,
    modality_mask: np.ndarray | None,
    invalid_intervals: Sequence[tuple[int, int]],
    *,
    chunk_rows: int,
) -> Iterable[tuple[int, int]]:
    """Yield contiguous combined-valid runs with bounded working memory."""

    if temporal_valid.ndim != 1:
        raise ValueError("IMU temporal validity must be one-dimensional")
    if modality_mask is not None and modality_mask.shape != temporal_valid.shape:
        raise ValueError("IMU modality mask must align with canonical analog rows")
    open_start: int | None = None
    interval_index = 0
    for chunk_start in range(0, temporal_valid.size, chunk_rows):
        chunk_end = min(temporal_valid.size, chunk_start + chunk_rows)
        combined = np.asarray(temporal_valid[chunk_start:chunk_end], dtype=bool).copy()
        if modality_mask is not None:
            combined &= np.asarray(modality_mask[chunk_start:chunk_end], dtype=bool)
        while interval_index < len(invalid_intervals) and invalid_intervals[interval_index][1] <= chunk_start:
            interval_index += 1
        cursor = interval_index
        while cursor < len(invalid_intervals) and invalid_intervals[cursor][0] < chunk_end:
            start, end = invalid_intervals[cursor]
            combined[max(start, chunk_start) - chunk_start : min(end, chunk_end) - chunk_start] = False
            cursor += 1
        if combined.size and combined[0] and open_start is None:
            open_start = chunk_start
        if combined.size and not combined[0] and open_start is not None:
            yield open_start, chunk_start
            open_start = None
        changes = np.diff(combined.astype(np.int8))
        for offset in np.flatnonzero(changes != 0):
            boundary = chunk_start + int(offset) + 1
            if changes[offset] < 0:
                assert open_start is not None
                yield open_start, boundary
                open_start = None
            else:
                assert open_start is None
                open_start = boundary
    if open_start is not None:
        yield open_start, int(temporal_valid.size)


def _merged_file_shapes(
    merged_analog_path: Path,
    valid_analog_samples_path: Path,
    *,
    canonical_rows: int,
    device_count: int,
) -> None:
    analog_bytes = canonical_rows * device_count * 16 * np.dtype("<i2").itemsize
    validity_bytes = canonical_rows * device_count
    if not merged_analog_path.is_file() or merged_analog_path.stat().st_size != analog_bytes:
        raise ValueError(
            "merged analogin.dat size does not match canonical_rows, device count, and 16 lanes"
        )
    if (
        not valid_analog_samples_path.is_file()
        or valid_analog_samples_path.stat().st_size != validity_bytes
    ):
        raise ValueError(
            "valid_analog_samples.dat size does not match canonical_rows and device count"
        )


def _canonical_modality_support(
    *,
    canonical_rows: int,
    device_count: int,
    canonical_modality_valid_by_device: Mapping[int, np.ndarray]
    | Sequence[np.ndarray]
    | None,
    invalid_canonical_intervals_by_device: Mapping[int, Iterable[object]]
    | Sequence[Iterable[object]]
    | None,
) -> tuple[tuple[np.ndarray | None, tuple[tuple[int, int], ...]], ...]:
    """Normalize optional IMU-only canonical validity decisions."""

    masks: list[np.ndarray | None] = [None] * device_count
    if canonical_modality_valid_by_device is not None:
        if isinstance(canonical_modality_valid_by_device, Mapping):
            unknown = set(canonical_modality_valid_by_device) - set(range(1, device_count + 1))
            if unknown:
                raise ValueError(f"unknown one-based modality-mask device ids: {sorted(unknown)}")
            entries = canonical_modality_valid_by_device.items()
        else:
            values = tuple(canonical_modality_valid_by_device)
            if len(values) != device_count:
                raise ValueError("canonical modality masks must align with input recordings")
            entries = enumerate(values, start=1)
        for device_id, value in entries:
            mask = np.asarray(value)
            if mask.shape != (canonical_rows,):
                raise ValueError("canonical modality mask must have one value per canonical row")
            if not (
                np.issubdtype(mask.dtype, np.bool_)
                or np.all((mask == 0) | (mask == 1))
            ):
                raise ValueError("canonical modality mask must contain only boolean/0/1 values")
            masks[int(device_id) - 1] = mask.astype(bool, copy=False)
    interval_sets: list[list[tuple[int, int]]] = [[] for _ in range(device_count)]
    if invalid_canonical_intervals_by_device is not None:
        if isinstance(invalid_canonical_intervals_by_device, Mapping):
            unknown = set(invalid_canonical_intervals_by_device) - set(range(1, device_count + 1))
            if unknown:
                raise ValueError(f"unknown one-based modality-interval device ids: {sorted(unknown)}")
            interval_entries = invalid_canonical_intervals_by_device.items()
        else:
            values = tuple(invalid_canonical_intervals_by_device)
            if len(values) != device_count:
                raise ValueError("canonical modality intervals must align with input recordings")
            interval_entries = enumerate(values, start=1)
        for device_id, intervals in interval_entries:
            for interval in intervals:
                start, end = _interval_bounds(interval)
                if end > canonical_rows:
                    raise ValueError("IMU modality interval exceeds canonical analog support")
                interval_sets[int(device_id) - 1].append((start, end))
    normalized_intervals: list[tuple[tuple[int, int], ...]] = []
    for intervals in interval_sets:
        intervals.sort()
        merged: list[tuple[int, int]] = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        normalized_intervals.append(tuple(merged))
    return tuple(zip(masks, normalized_intervals))


def _render_merged_device(
    merged: np.memmap,
    temporal_valid: np.ndarray,
    modality_mask: np.ndarray | None,
    invalid_modality_intervals: Sequence[tuple[int, int]],
    segments: Sequence[AnalogSyncSegment],
    *,
    device_zero_based: int,
    canonical_grid: np.ndarray,
    filter_coefficients: np.ndarray,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter one merged device independently within each complete valid run."""

    output = np.zeros((9, canonical_grid.size), dtype=np.float64)
    valid_output = np.zeros(canonical_grid.size, dtype=bool)
    source_rows = np.full(canonical_grid.size, -1.0, dtype=np.float64)
    half = filter_coefficients.size // 2
    lane_start = device_zero_based * 16 + IMU_LANES_ZERO_BASED[0]
    lane_end = lane_start + len(IMU_LANES_ZERO_BASED)
    segment_index = 0
    publishable = tuple(segment for segment in segments if segment.is_publishable)
    combined_runs = _iter_valid_runs(
        temporal_valid, modality_mask, invalid_modality_intervals, chunk_rows=chunk_rows
    )
    split_runs: list[tuple[int, int, AnalogSyncSegment]] = []
    # At most one small entry per validity-run/segment intersection. This is
    # compact segment metadata, never sample-sized working memory.
    for valid_start, valid_end in combined_runs:
        while segment_index < len(publishable) and publishable[segment_index].canonical_end_row <= valid_start:
            segment_index += 1
        cursor = segment_index
        while cursor < len(publishable) and publishable[cursor].canonical_start_row < valid_end:
            segment = publishable[cursor]
            start = max(valid_start, segment.canonical_start_row)
            end = min(valid_end, segment.canonical_end_row)
            if end > start:
                split_runs.append((start, end, segment))
            cursor += 1
    for run_start, run_end, segment in split_runs:
        if run_end - run_start < filter_coefficients.size:
            continue
        for core_start in range(run_start + half, run_end - half, chunk_rows):
            core_end = min(run_end - half, core_start + chunk_rows)
            # canonical_grid is globally sorted. Integer floor(grid) in
            # [core_start, core_end) is equivalent to grid in the same
            # half-open interval, so two binary searches replace a full-grid
            # mask per segment/core.
            target_start = int(np.searchsorted(canonical_grid, core_start, side="left"))
            target_end = int(np.searchsorted(canonical_grid, core_end, side="left"))
            if target_end <= target_start:
                continue
            targets = np.arange(target_start, target_end, dtype=np.int64)
            lower = np.floor(canonical_grid[targets]).astype(np.int64)
            upper = np.ceil(canonical_grid[targets]).astype(np.int64)
            usable = (lower >= run_start + half) & (upper < run_end - half)
            if not np.any(usable):
                continue
            targets = targets[usable]
            lower = lower[usable]
            upper = upper[usable]
            read_start = int(lower.min()) - half
            read_end = int(upper.max()) + half + 1
            values = np.asarray(
                merged[read_start:read_end, lane_start:lane_end], dtype=np.float64
            )
            filtered = convolve(values, filter_coefficients[:, None], mode="same")
            local_lower = lower - read_start
            local_upper = upper - read_start
            weights = canonical_grid[targets] - lower
            sampled = filtered[local_lower] + (
                filtered[local_upper] - filtered[local_lower]
            ) * weights[:, None]
            output[:, targets] = sampled.T
            valid_output[targets] = True
            source_rows[targets] = (
                segment.raw_scale * canonical_grid[targets] + segment.raw_intercept_rows
            )
    return _scaled_imu(output), valid_output, source_rows


def build_filtered_imu_from_merged(
    recordings: Sequence[Recording],
    merged_analog_path: str | Path,
    valid_analog_samples_path: str | Path,
    *,
    segments_by_device: Mapping[int, Iterable[AnalogSyncSegment]]
    | Sequence[Iterable[AnalogSyncSegment]],
    canonical_rows: int,
    output_sample_rate_hz: float = DEFAULT_IMU_SAMPLE_RATE_HZ,
    filter_taps: int = DEFAULT_FILTER_TAPS,
    filter_cutoff_hz: float = DEFAULT_FILTER_CUTOFF_HZ,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    max_output_samples: int = DEFAULT_MAX_OUTPUT_SAMPLES,
    master_index: int = 1,
    master_start_sample: int = 0,
    master_start_sec: float = 0.0,
    canonical_modality_valid_by_device: Mapping[int, np.ndarray]
    | Sequence[np.ndarray]
    | None = None,
    invalid_canonical_intervals_by_device: Mapping[int, Iterable[object]]
    | Sequence[Iterable[object]]
    | None = None,
    mapping_hashes_by_device: Mapping[int, str] | None = None,
    source_analog_label: str = "analogin.dat",
    source_validity_label: str = "valid_analog_samples.dat",
) -> SynchronizedImuResult:
    """Generate gap-safe 100 Hz IMU from the staged canonical analog pair.

    Analog device blocks are interpreted in input-recording order.  Validity
    columns are explicitly decoded in master-first order.  IMU-only modality
    exclusions are combined with mapping validity for filtering, without
    changing the published analog validity sidecar.
    """

    normalized = _validate_parameters(
        recordings,
        canonical_rows=canonical_rows,
        output_sample_rate_hz=output_sample_rate_hz,
        filter_taps=filter_taps,
        filter_cutoff_hz=filter_cutoff_hz,
        chunk_rows=chunk_rows,
        max_output_samples=max_output_samples,
        master_index=master_index,
        master_start_sample=master_start_sample,
        master_start_sec=master_start_sec,
    )
    merged_path = Path(merged_analog_path)
    validity_path = Path(valid_analog_samples_path)
    _merged_file_shapes(
        merged_path,
        validity_path,
        canonical_rows=canonical_rows,
        device_count=len(normalized),
    )
    modality_support = _canonical_modality_support(
        canonical_rows=canonical_rows,
        device_count=len(normalized),
        canonical_modality_valid_by_device=canonical_modality_valid_by_device,
        invalid_canonical_intervals_by_device=invalid_canonical_intervals_by_device,
    )
    segment_sets = _normalise_segment_sets(
        segments_by_device, device_count=len(normalized)
    )
    if mapping_hashes_by_device is not None:
        expected_hash_keys = set(range(1, len(normalized) + 1))
        if set(mapping_hashes_by_device) != expected_hash_keys:
            raise ValueError("mapping_hashes_by_device must use exactly one-based recording ids")
    grid, time_seconds = _output_grid(canonical_rows, output_sample_rate_hz)
    if grid.size > max_output_samples:
        raise ValueError(
            f"IMU result would contain {grid.size} samples, above the explicit in-memory limit "
            f"of {max_output_samples}; use a streaming output implementation"
        )
    coefficients = firwin(
        filter_taps,
        cutoff=float(filter_cutoff_hz),
        fs=RAW_ANALOG_SAMPLE_RATE_HZ,
        pass_zero="lowpass",
    ).astype(np.float64)
    merged = np.memmap(
        merged_path,
        dtype="<i2",
        mode="r",
        shape=(canonical_rows, len(normalized) * 16),
    )
    validity = np.memmap(
        validity_path,
        dtype=np.uint8,
        mode="r",
        shape=(canonical_rows, len(normalized)),
    )
    validity_order = (master_index - 1,) + tuple(
        index for index in range(len(normalized)) if index != master_index - 1
    )
    validity_column_by_device = {
        device_index: column for column, device_index in enumerate(validity_order)
    }
    devices: list[SynchronizedImuDevice] = []
    try:
        for start in range(0, canonical_rows, chunk_rows):
            block = np.asarray(validity[start : min(canonical_rows, start + chunk_rows)])
            if np.any((block != 0) & (block != 1)):
                raise ValueError("valid_analog_samples.dat must contain only 0 and 1")
        for device_zero_based, recording in enumerate(normalized):
            temporal = validity[:, validity_column_by_device[device_zero_based]]
            modality_mask, invalid_modality_intervals = modality_support[device_zero_based]
            scaled, device_valid, source_rows = _render_merged_device(
                merged,
                temporal,
                modality_mask,
                invalid_modality_intervals,
                segments=segment_sets[device_zero_based],
                device_zero_based=device_zero_based,
                canonical_grid=grid,
                filter_coefficients=coefficients,
                chunk_rows=chunk_rows,
            )
            raw_resampled = scaled.T
            sensor = ImuSensorData(
                acc=raw_resampled[:, 0:3],
                gyr=raw_resampled[:, 3:6],
                mag=raw_resampled[:, 6:9],
            )
            valid_count = int(np.count_nonzero(device_valid))
            valid_fraction = float(valid_count / device_valid.size) if device_valid.size else 0.0
            digest_payload = {
                "merged_analog_file": source_analog_label,
                "valid_analog_samples_file": source_validity_label,
                "device_index": device_zero_based + 1,
                "validity_column": validity_column_by_device[device_zero_based],
            }
            digest = (
                str(mapping_hashes_by_device[device_zero_based + 1])
                if mapping_hashes_by_device is not None
                else _mapping_digest(segment_sets[device_zero_based])
            )
            devices.append(
                SynchronizedImuDevice(
                    device_index=device_zero_based + 1,
                    device_name=recording.device_name,
                    recording_name=recording.recording_name,
                    source_folder=recording.folder,
                    raw_resampled=raw_resampled,
                    imu=sensor,
                    valid=device_valid,
                    source_rows=source_rows,
                    canonical_rows=grid.copy(),
                    mapping_hash=digest,
                    valid_count=valid_count,
                    valid_fraction=valid_fraction,
                    status=(
                        "OK"
                        if valid_count
                        and not invalid_modality_intervals
                        and not (
                            modality_mask is not None
                            and np.any(~modality_mask)
                        )
                        else "WARN"
                    ),
                    provenance={
                        **digest_payload,
                        "source_domain": "canonical_merged_analog",
                        "source_rows_semantics": "raw device analog-row coordinate from accepted segment",
                        "imu_channels_one_based_within_device_block": IMU_CHANNELS_ONE_BASED,
                        "analog_channels_merged_one_based": tuple(
                            device_zero_based * 16 + channel for channel in IMU_CHANNELS_ONE_BASED
                        ),
                        "filter": {
                            "type": "symmetric_fir_per_contiguous_valid_run",
                            "taps": int(filter_taps),
                            "cutoff_hz": float(filter_cutoff_hz),
                        },
                        "valid_count": valid_count,
                        "valid_fraction": valid_fraction,
                    },
                )
            )
    finally:
        close_memmap(merged)
        close_memmap(validity)
    all_valid = bool(devices) and all(device.status == "OK" for device in devices)
    return SynchronizedImuResult(
        sample_rate_hz=float(output_sample_rate_hz),
        raw_sample_rate_hz=RAW_ANALOG_SAMPLE_RATE_HZ,
        canonical_rows=grid,
        time_seconds=time_seconds,
        devices=tuple(devices),
        master_index=master_index,
        master_start_sample=master_start_sample,
        master_start_sec=float(master_start_sec),
        status="OK" if all_valid else "WARN",
        provenance={
            "source_domain": "canonical_merged_analog",
            "merged_analog_file": source_analog_label,
            "valid_analog_samples_file": source_validity_label,
            "source_analog_label": source_analog_label,
            "source_validity_label": source_validity_label,
            "analog_device_block_order": "input_recording_order",
            "validity_device_order": [index + 1 for index in validity_order],
            "canonical_grid": "k * 1250 / output_sample_rate_hz, k >= 0",
            "filter_taps": int(filter_taps),
            "filter_cutoff_hz": float(filter_cutoff_hz),
            "fusion": "not performed",
        },
    )


def _production_output_grid(canonical_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Return MATLAB ``resample(...,100,1250)`` length and phase."""

    count = int(math.ceil(canonical_rows * IMU_RESAMPLE_UP / IMU_RESAMPLE_DOWN))
    canonical = np.arange(count, dtype=np.float64) * (
        IMU_RESAMPLE_DOWN / IMU_RESAMPLE_UP
    )
    return canonical, np.arange(count, dtype=np.float64) / DEFAULT_IMU_SAMPLE_RATE_HZ


def _gap_safe_resample_support(
    temporal_valid: np.ndarray,
    modality_mask: np.ndarray | None,
    invalid_modality_intervals: Sequence[tuple[int, int]],
    segments: Sequence[AnalogSyncSegment],
    canonical_grid: np.ndarray,
    *,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int], ...]]:
    """Find outputs whose complete 501-tap polyphase support is verified."""

    valid_output = np.zeros(canonical_grid.size, dtype=bool)
    source_rows = np.full(canonical_grid.size, -1.0, dtype=np.float64)
    fusion_runs: list[tuple[int, int]] = []
    # MATLAB's filter is 501 taps at the 2x upsampled rate.  Relative to
    # 1250 Hz source rows its exact nonzero support is center +/- 125 rows.
    support_half_rows = 125.0
    publishable = tuple(segment for segment in segments if segment.is_publishable)
    segment_index = 0
    for valid_start, valid_end in _iter_valid_runs(
        temporal_valid,
        modality_mask,
        invalid_modality_intervals,
        chunk_rows=chunk_rows,
    ):
        while (
            segment_index < len(publishable)
            and publishable[segment_index].canonical_end_row <= valid_start
        ):
            segment_index += 1
        cursor = segment_index
        while cursor < len(publishable) and publishable[cursor].canonical_start_row < valid_end:
            segment = publishable[cursor]
            run_start = max(valid_start, segment.canonical_start_row)
            run_end = min(valid_end, segment.canonical_end_row)
            if run_end > run_start:
                candidate_start = int(
                    np.searchsorted(
                        canonical_grid,
                        run_start - support_half_rows - 1.0,
                        side="left",
                    )
                )
                candidate_end = int(
                    np.searchsorted(
                        canonical_grid,
                        run_end + support_half_rows,
                        side="left",
                    )
                )
                candidates = np.arange(candidate_start, candidate_end, dtype=np.int64)
                if candidates.size:
                    centers = canonical_grid[candidates]
                    first_source = np.ceil(centers - support_half_rows).astype(np.int64)
                    last_source = np.floor(centers + support_half_rows).astype(np.int64)
                    usable = (first_source >= run_start) & (last_source < run_end)
                    positions = candidates[usable]
                    if positions.size:
                        if np.any(valid_output[positions]):
                            raise ValueError("MATLAB IMU support maps to more than one segment/run")
                        valid_output[positions] = True
                        source_rows[positions] = (
                            segment.raw_scale * canonical_grid[positions]
                            + segment.raw_intercept_rows
                        )
                        changes = np.flatnonzero(np.diff(positions) != 1)
                        starts = np.concatenate(([0], changes + 1))
                        ends = np.concatenate((changes + 1, [positions.size]))
                        fusion_runs.extend(
                            (int(positions[start]), int(positions[end - 1]) + 1)
                            for start, end in zip(starts, ends)
                        )
            cursor += 1
    return valid_output, source_rows, tuple(fusion_runs)


def _calibrate_valid_imu(
    resampled_adc: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, ImuSensorData, np.ndarray]:
    """Apply legacy scale/calibration using verified output rows only."""

    nominal = scale_imu_nominal(resampled_adc)
    nominal_matrix = nominal.as_matrix()
    calibrated_acc = np.zeros_like(nominal.acc)
    calibrated_gyr = np.zeros_like(nominal.gyr)
    calibrated_mag = np.zeros_like(nominal.mag)
    final_valid = np.asarray(valid, dtype=bool).copy()
    if np.any(final_valid):
        norms = np.linalg.norm(nominal.acc[final_valid], axis=1)
        median_norm = float(np.median(norms))
        if not np.isfinite(median_norm) or median_norm <= 0.0:
            final_valid[:] = False
        else:
            calibrated_acc[final_valid] = nominal.acc[final_valid] / median_norm * 9.81
            gyro_bias = np.median(nominal.gyr[final_valid], axis=0)
            calibrated_gyr[final_valid] = nominal.gyr[final_valid] - gyro_bias
            calibrated_mag[final_valid] = nominal.mag[final_valid]
    nominal_matrix[~final_valid] = 0.0
    return (
        nominal_matrix,
        ImuSensorData(
            acc=calibrated_acc,
            gyr=calibrated_gyr,
            mag=calibrated_mag,
        ),
        final_valid,
    )


def _fuse_per_valid_run(
    sensor: ImuSensorData,
    valid: np.ndarray,
    runs: Sequence[tuple[int, int]],
) -> ImuFusionData:
    count = int(valid.size)
    quaternion = np.full((count, 4), np.nan, dtype=np.float64)
    orientation = np.full((count, 3, 3), np.nan, dtype=np.float64)
    acceleration = np.full((count, 3), np.nan, dtype=np.float64)
    speed = np.full((count, 3), np.nan, dtype=np.float64)
    fusion_valid = np.zeros(count, dtype=bool)
    unsupported_run_count = 0
    for start, end in runs:
        positions = np.arange(start, end, dtype=np.int64)
        positions = positions[valid[positions]]
        if positions.size <= 6 or np.any(np.diff(positions) != 1):
            unsupported_run_count += 1
            continue
        run_start = int(positions[0])
        run_end = int(positions[-1]) + 1
        ahrs = fuse_imu_ahrs(
            sensor.acc[run_start:run_end],
            sensor.gyr[run_start:run_end],
            sensor.mag[run_start:run_end],
            sample_rate_hz=DEFAULT_IMU_SAMPLE_RATE_HZ,
            include_diagnostics=False,
        )
        motion = compute_imu_motion(
            ahrs.quaternions,
            sensor.acc[run_start:run_end],
            sample_rate_hz=DEFAULT_IMU_SAMPLE_RATE_HZ,
        )
        publish_start = run_start + AHRS_WARMUP_SAMPLES
        if publish_start >= run_end:
            unsupported_run_count += 1
            continue
        quaternion[publish_start:run_end] = ahrs.quaternions[AHRS_WARMUP_SAMPLES:]
        orientation[publish_start:run_end] = motion.orientation[AHRS_WARMUP_SAMPLES:]
        acceleration[publish_start:run_end] = motion.acceleration[AHRS_WARMUP_SAMPLES:]
        speed[publish_start:run_end] = motion.speed[AHRS_WARMUP_SAMPLES:]
        fusion_valid[publish_start:run_end] = True
    complete = bool(np.any(fusion_valid)) and unsupported_run_count == 0
    return ImuFusionData(
        quaternion=quaternion,
        orientation=orientation,
        acceleration=acceleration,
        speed=speed,
        valid=fusion_valid,
        status="OK" if complete else "WARN",
        method="matlab_r2024b_ahrsfilter_defaults_per_valid_run",
        warning=(
            ""
            if complete
            else f"{unsupported_run_count} valid run(s) were too short for fusion"
        ),
    )


def _unavailable_fusion(valid: np.ndarray, error: Exception) -> ImuFusionData:
    """Represent optional fusion failure without discarding synchronized IMU."""

    count = int(valid.size)
    return ImuFusionData(
        quaternion=np.full((count, 4), np.nan, dtype=np.float64),
        orientation=np.full((count, 3, 3), np.nan, dtype=np.float64),
        acceleration=np.full((count, 3), np.nan, dtype=np.float64),
        speed=np.full((count, 3), np.nan, dtype=np.float64),
        valid=np.zeros(count, dtype=bool),
        status="WARN",
        method="matlab_r2024b_ahrsfilter_unavailable",
        warning=f"sensor fusion unavailable: {error}",
    )


def build_imu_from_merged(
    recordings: Sequence[Recording],
    merged_analog_path: str | Path,
    valid_analog_samples_path: str | Path,
    *,
    segments_by_device: Mapping[int, Iterable[AnalogSyncSegment]]
    | Sequence[Iterable[AnalogSyncSegment]],
    canonical_rows: int,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    max_output_samples: int = DEFAULT_MAX_OUTPUT_SAMPLES,
    max_peak_bytes: int = DEFAULT_MAX_PEAK_BYTES,
    master_index: int = 1,
    master_start_sample: int = 0,
    master_start_sec: float = 0.0,
    canonical_modality_valid_by_device: Mapping[int, np.ndarray]
    | Sequence[np.ndarray]
    | None = None,
    invalid_canonical_intervals_by_device: Mapping[int, Iterable[object]]
    | Sequence[Iterable[object]]
    | None = None,
    mapping_hashes_by_device: Mapping[int, str] | None = None,
    source_analog_label: str = "analogin.dat",
    source_validity_label: str = "valid_analog_samples.dat",
    perform_sensor_fusion: bool = True,
) -> SynchronizedImuResult:
    """Build gap-safe IMU with MATLAB R2024b-compatible numerical stages.

    MATLAB's full-session resampler is evaluated once per device, but an
    output is publishable only when every contributing source row belongs to
    one verified temporal/modality run and one analog mapping segment.  AHRS,
    acceleration centering, integration, and high-pass state restart at every
    such run, so invalid data cannot influence a published fusion value.
    """

    normalized = _validate_parameters(
        recordings,
        canonical_rows=canonical_rows,
        output_sample_rate_hz=DEFAULT_IMU_SAMPLE_RATE_HZ,
        filter_taps=501,
        filter_cutoff_hz=40.0,
        chunk_rows=chunk_rows,
        max_output_samples=max_output_samples,
        master_index=master_index,
        master_start_sample=master_start_sample,
        master_start_sec=master_start_sec,
    )
    merged_path = Path(merged_analog_path)
    validity_path = Path(valid_analog_samples_path)
    _merged_file_shapes(
        merged_path,
        validity_path,
        canonical_rows=canonical_rows,
        device_count=len(normalized),
    )
    modality_support = _canonical_modality_support(
        canonical_rows=canonical_rows,
        device_count=len(normalized),
        canonical_modality_valid_by_device=canonical_modality_valid_by_device,
        invalid_canonical_intervals_by_device=invalid_canonical_intervals_by_device,
    )
    segment_sets = _normalise_segment_sets(segments_by_device, device_count=len(normalized))
    if mapping_hashes_by_device is not None and set(mapping_hashes_by_device) != set(
        range(1, len(normalized) + 1)
    ):
        raise ValueError("mapping_hashes_by_device must use exactly one-based recording ids")
    grid, time_seconds = _production_output_grid(canonical_rows)
    if grid.size > max_output_samples:
        raise ValueError(
            f"IMU result would contain {grid.size} samples, above the explicit in-memory "
            f"limit of {max_output_samples}"
        )
    if not isinstance(max_peak_bytes, int) or max_peak_bytes <= 0:
        raise ValueError("max_peak_bytes must be a positive integer")
    # Conservative allocation guard.  One current device needs a float64
    # N-by-9 source workspace; retained per-output device state includes raw,
    # calibrated, quaternion, rotation, derived motion, coordinates, and masks.
    estimated_peak_bytes = (
        canonical_rows * 9 * np.dtype(np.float64).itemsize
        + grid.size * len(normalized) * 64 * np.dtype(np.float64).itemsize
    )
    if estimated_peak_bytes > max_peak_bytes:
        raise ValueError(
            "MATLAB-compatible IMU estimated peak memory "
            f"{estimated_peak_bytes} bytes exceeds explicit limit {max_peak_bytes}; "
            "use a streaming implementation or a smaller recording"
        )
    merged = np.memmap(
        merged_path,
        dtype="<i2",
        mode="r",
        shape=(canonical_rows, len(normalized) * 16),
    )
    validity = np.memmap(
        validity_path,
        dtype=np.uint8,
        mode="r",
        shape=(canonical_rows, len(normalized)),
    )
    validity_order = (master_index - 1,) + tuple(
        index for index in range(len(normalized)) if index != master_index - 1
    )
    validity_column_by_device = {
        device_index: column for column, device_index in enumerate(validity_order)
    }
    expected_edge_supported = (
        (np.ceil(grid - 125.0).astype(np.int64) >= 0)
        & (np.floor(grid + 125.0).astype(np.int64) < canonical_rows)
    )
    devices: list[SynchronizedImuDevice] = []
    try:
        for start in range(0, canonical_rows, chunk_rows):
            block = np.asarray(validity[start : min(canonical_rows, start + chunk_rows)])
            if np.any((block != 0) & (block != 1)):
                raise ValueError("valid_analog_samples.dat must contain only 0 and 1")
        for device_zero_based, recording in enumerate(normalized):
            lane_start = device_zero_based * 16 + IMU_LANES_ZERO_BASED[0]
            lane_end = lane_start + len(IMU_LANES_ZERO_BASED)
            resampled_adc = resample_imu_1250_to_100(
                np.asarray(merged[:, lane_start:lane_end])
            )
            if resampled_adc.shape[0] != grid.size:
                raise RuntimeError("MATLAB-compatible IMU grid length mismatch")
            temporal = validity[:, validity_column_by_device[device_zero_based]]
            modality_mask, invalid_modality_intervals = modality_support[device_zero_based]
            device_valid, source_rows, fusion_runs = _gap_safe_resample_support(
                temporal,
                modality_mask,
                invalid_modality_intervals,
                segment_sets[device_zero_based],
                grid,
                chunk_rows=chunk_rows,
            )
            raw_resampled, sensor, device_valid = _calibrate_valid_imu(
                resampled_adc,
                device_valid,
            )
            if perform_sensor_fusion:
                try:
                    fusion = _fuse_per_valid_run(
                        sensor, device_valid, fusion_runs
                    )
                except Exception as error:
                    fusion = _unavailable_fusion(device_valid, error)
            else:
                fusion = None
            valid_count = int(np.count_nonzero(device_valid))
            valid_fraction = float(valid_count / device_valid.size) if device_valid.size else 0.0
            has_modality_exclusion = bool(invalid_modality_intervals) or bool(
                modality_mask is not None and np.any(~modality_mask)
            )
            # The fixed polyphase filter necessarily lacks complete source
            # support at the recording edges.  Those deterministic edge rows
            # are not a QC warning; any additional loss is.
            complete_support = np.array_equal(device_valid, expected_edge_supported)
            device_status = (
                "OK"
                if complete_support
                and not has_modality_exclusion
                and (fusion is None or fusion.status == "OK")
                else "WARN"
            )
            digest = (
                str(mapping_hashes_by_device[device_zero_based + 1])
                if mapping_hashes_by_device is not None
                else _mapping_digest(segment_sets[device_zero_based])
            )
            devices.append(
                SynchronizedImuDevice(
                    device_index=device_zero_based + 1,
                    device_name=recording.device_name,
                    recording_name=recording.recording_name,
                    source_folder=recording.folder,
                    raw_resampled=raw_resampled,
                    imu=sensor,
                    valid=device_valid,
                    source_rows=source_rows,
                    canonical_rows=grid.copy(),
                    mapping_hash=digest,
                    valid_count=valid_count,
                    valid_fraction=valid_fraction,
                    status=device_status,
                    provenance={
                        "source_domain": "canonical_merged_analog",
                        "source_rows_semantics": (
                            "raw device analog-row coordinate from accepted segment"
                        ),
                        "merged_analog_file": source_analog_label,
                        "valid_analog_samples_file": source_validity_label,
                        "analog_channels_merged_one_based": tuple(
                            device_zero_based * 16 + channel
                            for channel in IMU_CHANNELS_ONE_BASED
                        ),
                        "resampling": (
                            "matlab_r2024b_resample_100_1250_501tap_firls_kaiser5"
                        ),
                        "calibration": (
                            "WILD_scaleIMU acceleration median norm and gyroscope "
                            "median bias on verified rows"
                        ),
                        "valid_count": valid_count,
                        "valid_fraction": valid_fraction,
                    },
                    fusion=fusion,
                )
            )
    finally:
        close_memmap(merged)
        close_memmap(validity)
    result_status = "OK" if devices and all(device.status == "OK" for device in devices) else "WARN"
    fusion_methods = {
        device.fusion.method for device in devices if device.fusion is not None
    }
    fusion_label = (
        next(iter(fusion_methods))
        if perform_sensor_fusion and len(fusion_methods) == 1
        else "mixed_or_unavailable_per_device"
        if perform_sensor_fusion
        else "not performed"
    )
    return SynchronizedImuResult(
        sample_rate_hz=DEFAULT_IMU_SAMPLE_RATE_HZ,
        raw_sample_rate_hz=RAW_ANALOG_SAMPLE_RATE_HZ,
        canonical_rows=grid,
        time_seconds=time_seconds,
        devices=tuple(devices),
        master_index=master_index,
        master_start_sample=master_start_sample,
        master_start_sec=float(master_start_sec),
        status=result_status,
        provenance={
            "source_domain": "canonical_merged_analog",
            "source_analog_label": source_analog_label,
            "source_validity_label": source_validity_label,
            "analog_device_block_order": "input_recording_order",
            "validity_device_order": [index + 1 for index in validity_order],
            "canonical_grid": "MATLAB ceil(2*N/25), k*12.5 analog rows",
            "resampling": "matlab_r2024b_resample_defaults_gap_safe_support",
            "calibration_support": "verified_output_rows_only",
            "fusion_warmup_samples_per_run": AHRS_WARMUP_SAMPLES,
            "estimated_peak_bytes": int(estimated_peak_bytes),
            "max_peak_bytes": int(max_peak_bytes),
            "fusion": fusion_label,
        },
    )


def _sample_raw_chunk(
    source: np.memmap,
    segments: Sequence[AnalogSyncSegment],
    canonical_integer_rows: np.ndarray,
    *,
    device_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly sample raw IMU lanes only where forward mapping is valid."""

    raw_rows, valid, _ = map_canonical_rows(
        segments,
        canonical_integer_rows,
        raw_row_count=source.shape[0],
        interpolation_half_width=1,
        device_index=device_index,
    )
    samples = np.zeros((canonical_integer_rows.size, 9), dtype=np.float64)
    if not np.any(valid):
        return samples, raw_rows, valid
    positions = np.flatnonzero(valid)
    coordinates = raw_rows[positions]
    lower = np.floor(coordinates).astype(np.int64)
    upper = np.ceil(coordinates).astype(np.int64)
    weights = coordinates - lower
    lower_values = np.asarray(source[lower][:, IMU_LANES_ZERO_BASED], dtype=np.float64)
    upper_values = np.asarray(source[upper][:, IMU_LANES_ZERO_BASED], dtype=np.float64)
    samples[positions] = lower_values + (upper_values - lower_values) * weights[:, None]
    return samples, raw_rows, valid


def _modality_supported_raw_rows(
    raw_rows: np.ndarray, *, invalid_raw_intervals: Sequence[tuple[int, int]]
) -> np.ndarray:
    """Require both source rows of a possible linear interpolation to be clean."""

    supported = np.isfinite(raw_rows)
    if not invalid_raw_intervals:
        return supported
    lower = np.floor(raw_rows).astype(np.int64, copy=False)
    upper = np.ceil(raw_rows).astype(np.int64, copy=False)
    for start, end in invalid_raw_intervals:
        # A fractional sample interpolates [lower, upper], so it is unsafe if
        # either endpoint is part of an IMU-only bad interval.
        supported &= ~((lower < end) & (upper >= start))
    return supported


def _render_device(
    recording: Recording,
    segments: Sequence[AnalogSyncSegment],
    *,
    canonical_grid: np.ndarray,
    filter_coefficients: np.ndarray,
    chunk_rows: int,
    invalid_raw_intervals: Sequence[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render one device without crossing mapping or filter support boundaries."""

    count = canonical_grid.size
    raw = np.zeros((9, count), dtype=np.float64)
    valid = np.zeros(count, dtype=bool)
    source_rows = np.full(count, -1.0, dtype=np.float64)
    half = filter_coefficients.size // 2
    # The +2 covers fractional target interpolation in addition to symmetric
    # FIR support.  Points inside this guard intentionally remain invalid.
    edge_guard = half + 2
    source = interleaved_memmap(recording.analog_file, recording.analog_channels, recording.analog_samples)
    try:
        for segment in segments:
            if not segment.is_publishable:
                continue
            inner_start = segment.canonical_start_row + edge_guard
            inner_end = segment.canonical_end_row - edge_guard
            if inner_end <= inner_start:
                continue
            for core_start in range(inner_start, inner_end, chunk_rows):
                core_end = min(inner_end, core_start + chunk_rows)
                targets = np.flatnonzero((canonical_grid >= core_start) & (canonical_grid < core_end))
                if not targets.size:
                    continue
                sample_start = max(segment.canonical_start_row, core_start - half - 1)
                sample_end = min(segment.canonical_end_row, core_end + half + 2)
                high_rows = np.arange(sample_start, sample_end, dtype=np.int64)
                high_values, mapped_rows, mapped_valid = _sample_raw_chunk(
                    source, segments, high_rows, device_index=segment.device_index
                )
                modality_valid = mapped_valid & _modality_supported_raw_rows(
                    mapped_rows, invalid_raw_intervals=invalid_raw_intervals
                )
                # Mapping validity is part of the FIR support test.  Thus a
                # malformed/local invalid row cannot be blended into a nearby
                # published output sample.
                support_count = convolve(
                    modality_valid.astype(np.int16),
                    np.ones(filter_coefficients.size, dtype=np.int16),
                    mode="same",
                )
                complete_support = support_count == filter_coefficients.size
                filtered = convolve(high_values, filter_coefficients[:, None], mode="same")
                local = canonical_grid[targets] - sample_start
                lower = np.floor(local).astype(np.int64)
                upper = np.ceil(local).astype(np.int64)
                usable = (
                    (lower >= 0)
                    & (upper < high_rows.size)
                    & modality_valid[lower]
                    & modality_valid[upper]
                    & complete_support[lower]
                    & complete_support[upper]
                )
                if not np.any(usable):
                    continue
                destination = targets[usable]
                weights = local[usable] - lower[usable]
                values = filtered[lower[usable]] + (
                    filtered[upper[usable]] - filtered[lower[usable]]
                ) * weights[:, None]
                raw[:, destination] = values.T
                source_rows[destination] = mapped_rows[lower[usable]] + (
                    mapped_rows[upper[usable]] - mapped_rows[lower[usable]]
                ) * weights
                valid[destination] = True
    finally:
        close_memmap(source)
    return _scaled_imu(raw), valid, source_rows


def build_synchronized_imu(
    recordings: Sequence[Recording],
    segments_by_device: Mapping[int, Iterable[AnalogSyncSegment]]
    | Sequence[Iterable[AnalogSyncSegment]],
    *,
    canonical_rows: int,
    output_sample_rate_hz: float = DEFAULT_IMU_SAMPLE_RATE_HZ,
    filter_taps: int = DEFAULT_FILTER_TAPS,
    filter_cutoff_hz: float = DEFAULT_FILTER_CUTOFF_HZ,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    max_output_samples: int = DEFAULT_MAX_OUTPUT_SAMPLES,
    master_index: int = 1,
    master_start_sample: int = 0,
    master_start_sec: float = 0.0,
    invalid_raw_intervals_by_device: Mapping[int, Iterable[object]]
    | Sequence[Iterable[object]]
    | None = None,
) -> SynchronizedImuResult:
    """Extract and antialias raw device IMU channels onto a common 100 Hz grid.

    No spatial/sensor calibration or AHRS fusion is performed here.  This is
    intentional: calibration and fusion policies must be explicit downstream,
    while temporal validity remains auditable per device.  Invalid values are
    exact zero and the corresponding source row is ``-1``.
    """

    normalized = _validate_parameters(
        recordings,
        canonical_rows=canonical_rows,
        output_sample_rate_hz=output_sample_rate_hz,
        filter_taps=filter_taps,
        filter_cutoff_hz=filter_cutoff_hz,
        chunk_rows=chunk_rows,
        max_output_samples=max_output_samples,
        master_index=master_index,
        master_start_sample=master_start_sample,
        master_start_sec=master_start_sec,
    )
    segment_sets = _normalise_segment_sets(segments_by_device, device_count=len(normalized))
    invalid_interval_sets = _normalise_invalid_raw_intervals(
        invalid_raw_intervals_by_device, recordings=normalized
    )
    grid, time_seconds = _output_grid(canonical_rows, output_sample_rate_hz)
    if grid.size > max_output_samples:
        raise ValueError(
            f"IMU result would contain {grid.size} samples, above the explicit in-memory limit "
            f"of {max_output_samples}; use a streaming output implementation"
        )
    coefficients = firwin(
        filter_taps,
        cutoff=float(filter_cutoff_hz),
        fs=RAW_ANALOG_SAMPLE_RATE_HZ,
        pass_zero="lowpass",
    ).astype(np.float64)
    devices: list[SynchronizedImuDevice] = []
    for index, (recording, segments, invalid_intervals) in enumerate(
        zip(normalized, segment_sets, invalid_interval_sets), start=1
    ):
        scaled, valid, source_rows = _render_device(
            recording,
            segments,
            canonical_grid=grid,
            filter_coefficients=coefficients,
            chunk_rows=chunk_rows,
            invalid_raw_intervals=invalid_intervals,
        )
        # Match the MATLAB multi-device contract: rawResampled is N-by-9 and
        # each derived sensor matrix is an N-by-3 view of the same storage.
        raw_resampled = scaled.T
        sensor = ImuSensorData(
            acc=raw_resampled[:, 0:3],
            gyr=raw_resampled[:, 3:6],
            mag=raw_resampled[:, 6:9],
        )
        valid_count = int(np.count_nonzero(valid))
        valid_fraction = float(valid_count / valid.size) if valid.size else 0.0
        device_status = "OK" if valid_count else "WARN"
        digest = _mapping_digest(segments)
        devices.append(
            SynchronizedImuDevice(
                device_index=index,
                device_name=recording.device_name,
                recording_name=recording.recording_name,
                source_folder=recording.folder,
                raw_resampled=raw_resampled,
                imu=sensor,
                valid=valid,
                source_rows=source_rows,
                canonical_rows=grid.copy(),
                mapping_hash=digest,
                valid_count=valid_count,
                valid_fraction=valid_fraction,
                status=device_status,
                provenance={
                    "mapping_hash": digest,
                    "source_analog_file": str(recording.analog_file),
                    "imu_channels_one_based": IMU_CHANNELS_ONE_BASED,
                    "filter": {
                        "type": "symmetric_fir",
                        "taps": int(filter_taps),
                        "cutoff_hz": float(filter_cutoff_hz),
                        "edge_guard_rows": int(filter_taps // 2 + 2),
                    },
                    "temporal_status": "temporal_sync_only",
                    "modality_invalid_raw_intervals": [list(interval) for interval in invalid_intervals],
                    "valid_count": valid_count,
                    "valid_fraction": valid_fraction,
                    "native_odr_reconstruction": "not performed; analog int16 holds are resampled as a 1250 Hz stream, matching legacy MATLAB semantics",
                },
            )
        )
    all_valid = bool(devices) and all(np.any(device.valid) for device in devices)
    return SynchronizedImuResult(
        sample_rate_hz=float(output_sample_rate_hz),
        raw_sample_rate_hz=RAW_ANALOG_SAMPLE_RATE_HZ,
        canonical_rows=grid,
        time_seconds=time_seconds,
        devices=tuple(devices),
        master_index=master_index,
        master_start_sample=master_start_sample,
        master_start_sec=float(master_start_sec),
        status="OK" if all_valid else "WARN",
        provenance={
            "canonical_grid": "k * 1250 / output_sample_rate_hz, k >= 0",
            "filter_taps": int(filter_taps),
            "filter_cutoff_hz": float(filter_cutoff_hz),
            "fusion": "not performed",
            "raw_analog_immutable": True,
            "mapping_hashes_by_device": {str(device.device_index): device.mapping_hash for device in devices},
            "validity_by_device": {
                str(device.device_index): {
                    "valid_count": device.valid_count,
                    "valid_fraction": device.valid_fraction,
                }
                for device in devices
            },
            "native_odr_reconstruction": "not performed; analog int16 holds are resampled as a 1250 Hz stream, matching legacy MATLAB semantics",
        },
    )


def _mat_device(device: SynchronizedImuDevice, time_seconds: np.ndarray) -> dict[str, object]:
    """Return a scipy.io-compatible MATLAB struct payload for one device."""

    if device.fusion is None:
        fusion_payload: object = np.empty((0, 0), dtype=np.float64)
        fusion_status = "NOT_RUN"
        fusion_method = ""
        fusion_valid = np.zeros(device.valid.shape, dtype=np.uint8)
    else:
        fusion_payload = {
            "imu": {
                "acc": device.imu.acc,
                "gyr": device.imu.gyr,
                "mag": device.imu.mag,
            },
            "fs": DEFAULT_IMU_SAMPLE_RATE_HZ,
            "quaternion": device.fusion.quaternion,
            # Legacy MATLAB stores this field as 3-by-3-by-N.
            "orientation": np.moveaxis(device.fusion.orientation, 0, 2),
            "accel": device.fusion.acceleration,
            "speed": device.fusion.speed,
            "timestamp": time_seconds,
            "deviceIndex": int(device.device_index),
            "deviceName": device.device_name,
            "recordingName": device.recording_name,
            "sourceFolder": str(device.source_folder),
            "analogChannelsOriginal": np.asarray(IMU_CHANNELS_ONE_BASED, dtype=np.int16),
            "analogChannelsMerged": np.asarray(
                device.provenance.get("analog_channels_merged_one_based", ()), dtype=np.int16
            ),
            "valid": device.fusion.valid.astype(np.uint8),
            "method": device.fusion.method,
            "status": device.fusion.status,
        }
        fusion_status = device.fusion.status
        fusion_method = device.fusion.method
        fusion_valid = device.fusion.valid.astype(np.uint8)
    return {
        "deviceIndex": int(device.device_index),
        "deviceName": device.device_name,
        "recordingName": device.recording_name,
        "sourceFolder": str(device.provenance.get("source_folder_label", device.source_folder)),
        "analogChannelsOriginal": np.asarray(IMU_CHANNELS_ONE_BASED, dtype=np.int16),
        "analogChannelsMerged": np.asarray(
            device.provenance.get("analog_channels_merged_one_based", ()), dtype=np.int16
        ),
        "timestamp": time_seconds,
        "rawResampled": device.raw_resampled,
        "imu": {"acc": device.imu.acc, "gyr": device.imu.gyr, "mag": device.imu.mag},
        "fusionData": fusion_payload,
        "valid": device.valid.astype(np.uint8),
        "fusionValid": fusion_valid,
        "sourceRows": device.source_rows,
        "canonicalRows": device.canonical_rows,
        "mappingHash": device.mapping_hash,
        "validCount": int(device.valid_count),
        "validFraction": float(device.valid_fraction),
        "status": device.status,
        "fusionStatus": fusion_status,
        "fusionMethod": fusion_method,
        "fusionWarning": "" if device.fusion is None else device.fusion.warning,
    }


def write_synchronized_imu_mat(
    result: SynchronizedImuResult,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a MATLAB v5 ``IMU.mat`` readable through ``scipy.io.loadmat``.

    The v5 container is deliberately selected for broad MATLAB/Python reader
    compatibility.  It preserves the established top-level ``IMU`` variable
    and device fields and includes numeric MATLAB-compatible fusion fields when
    they were requested.
    """

    if not isinstance(result, SynchronizedImuResult):
        raise ValueError("result must be a SynchronizedImuResult")
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"IMU.mat exists; enable overwrite to regenerate: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    merged_source = result.provenance.get("source_domain") == "canonical_merged_analog"
    performed_fusion = bool(result.devices) and all(
        device.fusion is not None for device in result.devices
    )
    top_fusion_status = (
        "OK"
        if performed_fusion and all(device.fusion.status == "OK" for device in result.devices)
        else "WARN"
        if performed_fusion
        else "NOT_RUN"
    )
    fusion_methods = {
        device.fusion.method
        for device in result.devices
        if device.fusion is not None
    }
    top_fusion_method = (
        next(iter(fusion_methods))
        if performed_fusion and len(fusion_methods) == 1
        else "mixed_or_unavailable_per_device"
        if performed_fusion
        else ""
    )
    payload = {
        "createdAt": "",
        # The production path names the single canonical merged analog source;
        # the retained raw-reference path lists its device-local source files.
        "sourceAnalogFile": str(result.provenance.get("source_analog_label", "")),
        "sourceAnalogFiles": (
            np.asarray([], dtype=object)
            if merged_source
            else np.asarray(
                [str(device.provenance["source_analog_file"]) for device in result.devices],
                dtype=object,
            )
        ),
        "fs_raw": float(result.raw_sample_rate_hz),
        "fs": float(result.sample_rate_hz),
        "masterIndex": int(result.master_index),
        "masterStartSample": int(result.master_start_sample),
        "masterStartSec": float(result.master_start_sec),
        "time": result.time_seconds,
        "canonicalRows": result.canonical_rows,
        "status": result.status,
        "fusionStatus": top_fusion_status,
        "fusionMethod": top_fusion_method,
        "device": [_mat_device(device, result.time_seconds) for device in result.devices],
    }
    savemat(destination, {"IMU": payload}, do_compression=True, long_field_names=True)
    return destination
