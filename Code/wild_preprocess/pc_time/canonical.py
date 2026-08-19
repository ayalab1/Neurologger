"""Canonical PC-time fitting and camera timestamp mapping helpers.

The post-hoc merge timeline can contain synthetic master samples for confirmed
master gaps.  Packed PC-clock updates originate on the compressed raw-master
timeline, so their indices must be lifted onto that canonical timeline before
fitting or writing ``pc_time.dat``.  This module keeps that coordinate change
explicit and does not modify the input gap or update arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ..models import DeviceGap
from .decode import DAY_MS
from .infer import PcTimeModel, fit_robust_pc_time_model
from .validate import PcTimeOptions, PcTimeValidation, validate_pc_time_interval
from .write import write_interval_pc_time


@dataclass(frozen=True)
class CanonicalPcTimeFit:
    """PC-clock fit with the raw and canonical update coordinates retained."""

    raw_update_indices: np.ndarray
    canonical_update_indices: np.ndarray
    model: PcTimeModel


@dataclass(frozen=True)
class CameraTimestampMapping:
    """Nearest canonical samples for camera timestamps.

    ``canonical_sample_indices`` is ``-1`` for non-finite, out-of-range, or
    distance-rejected camera timestamps.  ``residual_ms`` is the signed
    canonical-PC-time minus camera-time difference; it is NaN when no sample
    is assigned.  ``selected_device_valid`` is false for unassigned samples
    and whenever any requested validity channel is zero.
    """

    canonical_sample_indices: np.ndarray
    residual_ms: np.ndarray
    distance_ms: np.ndarray
    selected_device_valid: np.ndarray
    in_range: np.ndarray
    camera_unwrapped_ms: np.ndarray
    canonical_pc_unwrapped_ms: np.ndarray


def _confirmed_master_gaps(
    device_gaps: Sequence[DeviceGap],
    *,
    master_device_index: int,
) -> tuple[DeviceGap, ...]:
    """Return sorted, non-overlapping confirmed master insertions.

    ``DeviceGap.device_index`` is one-based, matching merge metadata.  A
    caller should pass only confidently attributed gaps; this helper enforces
    the documented high/medium confidence contract rather than silently
    folding unresolved intervals into the PC-clock coordinate system.
    """

    master_index = int(master_device_index)
    if master_index < 1:
        raise ValueError("master_device_index must be one-based and positive")
    selected = tuple(
        sorted(
            (gap for gap in device_gaps if gap.device_index == master_index),
            key=lambda gap: gap.canonical_start_sample,
        )
    )
    previous_end = 0
    inserted_before = 0
    for gap in selected:
        if gap.missing_samples <= 0:
            raise ValueError("master gaps must contain a positive sample count")
        if gap.confidence not in {"high", "medium"}:
            raise ValueError("canonical PC time accepts only confirmed high/medium master gaps")
        raw_boundary = gap.canonical_start_sample - inserted_before
        if raw_boundary < 0 or gap.canonical_start_sample < previous_end:
            raise ValueError("master gaps must be sorted, non-overlapping canonical insertions")
        previous_end = gap.canonical_end_sample
        inserted_before += gap.missing_samples
    return selected


def map_raw_master_indices_to_canonical(
    raw_update_indices: np.ndarray | Sequence[int],
    device_gaps: Sequence[DeviceGap],
    *,
    master_device_index: int,
) -> np.ndarray:
    """Map raw-master update positions to canonical sample positions.

    Each confirmed master gap occupies ``[canonical_start, canonical_end)``.
    The first raw sample after that insertion maps to ``canonical_end``.  The
    input positions are treated as sample positions (not intervals), and are
    never mutated.
    """

    raw = np.asarray(raw_update_indices, dtype=np.int64)
    if raw.ndim != 1:
        raise ValueError("raw_update_indices must be one-dimensional")
    if np.any(raw < 0):
        raise ValueError("raw_update_indices must be non-negative")
    gaps = _confirmed_master_gaps(device_gaps, master_device_index=master_device_index)
    if not gaps:
        return raw.copy()

    raw_boundaries: list[int] = []
    insertions: list[int] = []
    inserted_before = 0
    for gap in gaps:
        raw_boundaries.append(gap.canonical_start_sample - inserted_before)
        inserted_before += gap.missing_samples
        insertions.append(inserted_before)
    positions = np.searchsorted(np.asarray(raw_boundaries, dtype=np.int64), raw, side="right")
    cumulative = np.r_[0, np.asarray(insertions, dtype=np.int64)]
    canonical = raw + cumulative[positions]
    if canonical.size > 1 and np.any(np.diff(canonical) <= 0):
        raise ValueError("canonical packed-update indices must remain strictly increasing")
    return canonical.astype(np.int64, copy=False)


def fit_gap_aware_pc_time_model(
    raw_update_indices: np.ndarray | Sequence[int],
    packed_values: np.ndarray | Sequence[int],
    sample_rate_hz: float,
    recording_start_ms: int,
    *,
    device_gaps: Sequence[DeviceGap] = (),
    master_device_index: int = 1,
) -> CanonicalPcTimeFit:
    """Fit the existing robust PC model using canonical update indices."""

    raw = np.asarray(raw_update_indices, dtype=np.int64)
    canonical = map_raw_master_indices_to_canonical(
        raw,
        device_gaps,
        master_device_index=master_device_index,
    )
    model = fit_robust_pc_time_model(canonical, np.asarray(packed_values, dtype=np.uint32), sample_rate_hz, recording_start_ms)
    return CanonicalPcTimeFit(raw.copy(), canonical, model)


def validate_canonical_pc_time_interval(
    fit: CanonicalPcTimeFit | PcTimeModel,
    *,
    sample_rate_hz: float,
    canonical_start_sample: int,
    n_samples: int,
    options: PcTimeOptions | None = None,
) -> PcTimeValidation:
    """Validate a fitted PC clock over a final canonical output interval."""

    model = fit.model if isinstance(fit, CanonicalPcTimeFit) else fit
    resolved_options = PcTimeOptions() if options is None else options
    return validate_pc_time_interval(
        model,
        sample_rate_hz=sample_rate_hz,
        common_start_master_sample=int(canonical_start_sample),
        n_samples=int(n_samples),
        options=resolved_options,
    )


def write_canonical_interval_pc_time(
    output_path: Path,
    fit: CanonicalPcTimeFit | PcTimeModel,
    *,
    sample_rate_hz: float,
    canonical_start_sample: int,
    n_samples: int,
    chunk_samples: int = 1_000_000,
    progress: Callable[[float], None] | None = None,
) -> Path:
    """Write daily uint32 PC timestamps on final canonical coordinates.

    The existing writer deliberately retains its modulo-``DAY_MS`` binary
    format.  Consumers requiring a monotonic clock must use
    :func:`unwrap_daily_ms` before comparing timestamps across midnight.
    """

    model = fit.model if isinstance(fit, CanonicalPcTimeFit) else fit
    return write_interval_pc_time(
        Path(output_path),
        model,
        sample_rate_hz=sample_rate_hz,
        common_start_master_sample=int(canonical_start_sample),
        n_samples=int(n_samples),
        chunk_samples=chunk_samples,
        progress=progress,
    )


def _read_uint8_valid_samples(
    path: str | Path,
    *,
    n_samples: int,
    device_count: int,
) -> np.ndarray:
    source = Path(path)
    if device_count <= 0:
        raise ValueError("device_count must be positive with valid_samples_path")
    expected_bytes = int(n_samples) * int(device_count)
    if source.stat().st_size != expected_bytes:
        raise ValueError("valid_samples.dat byte length does not match pc_time and device_count")
    return np.memmap(source, dtype=np.uint8, mode="r", shape=(n_samples, device_count))


def _coerce_valid_samples(
    valid_samples: np.ndarray | None,
    valid_samples_path: str | Path | None,
    *,
    n_samples: int,
    device_count: int | None,
) -> np.ndarray | None:
    if valid_samples is not None and valid_samples_path is not None:
        raise ValueError("pass valid_samples or valid_samples_path, not both")
    if valid_samples is None and valid_samples_path is None:
        return None
    if valid_samples is not None:
        mask = np.asarray(valid_samples)
        if mask.ndim != 2 or mask.shape[0] != n_samples:
            raise ValueError("valid_samples must have shape (canonical_samples, devices)")
        if device_count is not None and mask.shape[1] != int(device_count):
            raise ValueError("device_count does not match valid_samples")
    else:
        if device_count is None:
            raise ValueError("device_count is required with valid_samples_path")
        mask = _read_uint8_valid_samples(valid_samples_path, n_samples=n_samples, device_count=int(device_count))
    if not np.issubdtype(mask.dtype, np.number):
        raise ValueError("valid_samples must contain only 0 and 1")
    try:
        for start in range(0, mask.shape[0], 1_000_000):
            block = np.asarray(mask[start : min(mask.shape[0], start + 1_000_000)])
            if np.any((block != 0) & (block != 1)):
                raise ValueError("valid_samples must contain only 0 and 1")
    except BaseException:
        if isinstance(mask, np.memmap):
            mmap = getattr(mask, "_mmap", None)
            if mmap is not None:
                mmap.close()
        raise
    return mask


def unwrap_daily_ms(values: np.ndarray | Sequence[float]) -> np.ndarray:
    """Unwrap ordered milliseconds-since-midnight values across midnight.

    A decrease larger than half a day starts the next day.  Smaller decreases
    are retained: they are data quality evidence rather than a midnight wrap.
    Non-finite values remain NaN and do not change the day counter.
    """

    daily = np.asarray(values, dtype=float)
    if daily.ndim != 1:
        raise ValueError("daily timestamps must be one-dimensional")
    unwrapped = np.full(daily.shape, np.nan, dtype=float)
    offset = 0.0
    previous: float | None = None
    for index, value in enumerate(daily):
        if not np.isfinite(value):
            continue
        if not 0.0 <= value < DAY_MS:
            raise ValueError("daily timestamps must be milliseconds in [0, 86400000)")
        if previous is not None and value - previous < -(DAY_MS / 2.0):
            offset += DAY_MS
        unwrapped[index] = value + offset
        previous = float(value)
    return unwrapped


def _unwrap_pc_daily_ms(values: np.ndarray, chunk_size: int = 1_000_000) -> np.ndarray:
    unwrapped = np.empty(values.size, dtype=np.float64)
    day_count = 0
    previous_daily: int | None = None
    previous_unwrapped: float | None = None
    for start in range(0, values.size, chunk_size):
        stop = min(values.size, start + chunk_size)
        daily = np.asarray(values[start:stop], dtype=np.int64)
        if np.any((daily < 0) | (daily >= DAY_MS)):
            raise ValueError("pc_time must contain uint32 milliseconds since midnight")
        if daily.size == 0:
            continue
        preceding = daily[0] if previous_daily is None else previous_daily
        differences = np.diff(np.concatenate(([preceding], daily)))
        wraps = differences < -(DAY_MS / 2)
        offsets = day_count + np.cumsum(wraps, dtype=np.int64)
        block = daily.astype(np.float64) + offsets * DAY_MS
        if previous_unwrapped is not None and block[0] < previous_unwrapped:
            raise ValueError("pc_time must be monotonic after daily-wrap unwrapping")
        if np.any(np.diff(block) < 0):
            raise ValueError("pc_time must be monotonic after daily-wrap unwrapping")
        unwrapped[start:stop] = block
        day_count = int(offsets[-1])
        previous_daily = int(daily[-1])
        previous_unwrapped = float(block[-1])
    return unwrapped


def _scan_pc_time_index(
    values: np.ndarray, chunk_size: int = 1_000_000
) -> tuple[np.ndarray, float, float]:
    wraps: list[int] = []
    day_count = 0
    previous_daily: int | None = None
    previous_unwrapped: float | None = None
    for start in range(0, values.size, chunk_size):
        stop = min(values.size, start + chunk_size)
        daily = np.asarray(values[start:stop], dtype=np.int64)
        if np.any((daily < 0) | (daily >= DAY_MS)):
            raise ValueError("pc_time must contain uint32 milliseconds since midnight")
        preceding = daily[0] if previous_daily is None else previous_daily
        differences = np.diff(np.concatenate(([preceding], daily)))
        local_wraps = differences < -(DAY_MS / 2)
        invalid_decreases = (differences < 0) & ~local_wraps
        if np.any(invalid_decreases):
            raise ValueError("pc_time must be monotonic after daily-wrap unwrapping")
        wraps.extend((start + np.flatnonzero(local_wraps)).tolist())
        day_count += int(np.count_nonzero(local_wraps))
        block_last = float(daily[-1] + day_count * DAY_MS)
        if previous_unwrapped is not None and block_last < previous_unwrapped:
            raise ValueError("pc_time must be monotonic after daily-wrap unwrapping")
        previous_daily = int(daily[-1])
        previous_unwrapped = block_last
    wrap_indices = np.asarray(wraps, dtype=np.int64)
    first = float(values[0])
    last = float(values[-1]) + DAY_MS * wrap_indices.size
    return wrap_indices, first, last


def _pc_values_at(
    values: np.ndarray, indices: np.ndarray, wrap_indices: np.ndarray
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    daily = np.asarray(values[indices], dtype=np.float64)
    days = np.searchsorted(wrap_indices, indices, side="right")
    return daily + days * DAY_MS


def _nearest_indexed_pc_values(
    values: np.ndarray,
    wrap_indices: np.ndarray,
    queries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.empty(queries.size, dtype=np.int64)
    residuals = np.empty(queries.size, dtype=np.float64)
    for query_index, query in enumerate(queries):
        lower = 0
        upper = values.size
        while lower < upper:
            middle = (lower + upper) // 2
            middle_value = float(_pc_values_at(values, np.asarray([middle]), wrap_indices)[0])
            if middle_value < query:
                lower = middle + 1
            else:
                upper = middle
        right = min(values.size - 1, lower)
        left = max(0, right - 1)
        candidates = np.asarray([left, right], dtype=np.int64)
        candidate_values = _pc_values_at(values, candidates, wrap_indices)
        selected = int(np.argmin(np.abs(candidate_values - query)))
        indices[query_index] = candidates[selected]
        residuals[query_index] = candidate_values[selected] - query
    return indices, residuals


def _map_loaded_camera_timestamps(
    pc_daily: np.ndarray,
    wrap_indices: np.ndarray,
    pc_first: float,
    pc_last: float,
    camera_daily: np.ndarray,
    camera_unwrapped: np.ndarray,
    mask: np.ndarray | None,
    channels: tuple[int, ...],
    max_distance_ms: float | None,
    retain_canonical_pc_unwrapped: bool,
) -> CameraTimestampMapping:
    assigned = np.full(camera_daily.shape, -1, dtype=np.int64)
    residual = np.full(camera_daily.shape, np.nan, dtype=float)
    in_range = np.zeros(camera_daily.shape, dtype=bool)
    finite = np.isfinite(camera_unwrapped)
    if np.any(finite):
        values = camera_unwrapped[finite]
        within = (values >= pc_first) & (values <= pc_last)
        finite_positions = np.flatnonzero(finite)
        in_range[finite_positions[within]] = True
        if np.any(within):
            candidates, candidate_residual = _nearest_indexed_pc_values(
                pc_daily,
                wrap_indices,
                values[within],
            )
            accepted = (
                np.abs(candidate_residual) <= float(max_distance_ms)
                if max_distance_ms is not None
                else np.ones(candidates.shape, dtype=bool)
            )
            positions = finite_positions[within]
            assigned[positions[accepted]] = candidates[accepted]
            residual[positions[accepted]] = candidate_residual[accepted]
            in_range[positions[~accepted]] = False
    selected_valid = np.zeros(camera_daily.shape, dtype=bool)
    usable = assigned >= 0
    if not channels:
        selected_valid[usable] = True
    elif np.any(usable):
        selected_valid[usable] = np.all(
            np.asarray(mask)[assigned[usable]][:, channels] == 1,
            axis=1,
        )
    retained_pc = (
        _unwrap_pc_daily_ms(pc_daily)
        if retain_canonical_pc_unwrapped
        else np.empty(0, dtype=np.float64)
    )
    return CameraTimestampMapping(
        canonical_sample_indices=assigned,
        residual_ms=residual,
        distance_ms=np.abs(residual),
        selected_device_valid=selected_valid,
        in_range=in_range,
        camera_unwrapped_ms=camera_unwrapped,
        canonical_pc_unwrapped_ms=retained_pc,
    )


def map_camera_timestamps_to_canonical(
    pc_time: np.ndarray | Sequence[int] | str | Path,
    camera_timestamps_ms: np.ndarray | Sequence[float],
    *,
    requested_validity_channels: Sequence[int] = (),
    valid_samples: np.ndarray | None = None,
    valid_samples_path: str | Path | None = None,
    device_count: int | None = None,
    max_distance_ms: float | None = None,
    retain_canonical_pc_unwrapped: bool = False,
) -> CameraTimestampMapping:
    """Map camera daily timestamps to final canonical samples.

    ``pc_time`` is either final ``pc_time.dat`` (little-endian uint32) or its
    in-memory values.  Camera values are milliseconds since midnight.  Both
    ordered series are explicitly unwrapped across midnight before nearest
    sample matching.  Requested validity channels use zero-based file-channel
    indices (ch 0 master, ch 1 slave 1, and so on).  The full unwrapped PC
    vector is omitted from the result by default so long recordings can be
    mapped with bounded memory; request it explicitly only when needed.
    """

    camera_daily = np.asarray(camera_timestamps_ms, dtype=float)
    if camera_daily.ndim != 1:
        raise ValueError("camera_timestamps_ms must be one-dimensional")
    channels = tuple(int(channel) for channel in requested_validity_channels)
    if len(set(channels)) != len(channels) or any(channel < 0 for channel in channels):
        raise ValueError("requested_validity_channels must be unique non-negative indices")
    if max_distance_ms is not None and max_distance_ms < 0:
        raise ValueError("max_distance_ms must be non-negative")

    file_backed_pc: np.memmap | None = None
    if isinstance(pc_time, (str, Path)):
        path = Path(pc_time)
        if path.stat().st_size % np.dtype("<u4").itemsize:
            raise ValueError("pc_time.dat byte length is not uint32-aligned")
        file_backed_pc = np.memmap(path, dtype="<u4", mode="r")
        pc_daily = file_backed_pc
    else:
        pc_daily = np.asarray(pc_time)
    if pc_daily.ndim != 1 or pc_daily.size == 0:
        raise ValueError("pc_time must be a non-empty one-dimensional uint32 series")
    if not np.issubdtype(pc_daily.dtype, np.integer):
        raise ValueError("pc_time must contain uint32 milliseconds since midnight")
    try:
        wrap_indices, pc_first, pc_last = _scan_pc_time_index(pc_daily)
    except BaseException:
        if file_backed_pc is not None:
            mmap = getattr(file_backed_pc, "_mmap", None)
            if mmap is not None:
                mmap.close()
        raise
    camera_unwrapped = unwrap_daily_ms(camera_daily)
    finite_camera = np.isfinite(camera_unwrapped)
    if np.any(finite_camera):
        camera_center = float(np.median(camera_unwrapped[finite_camera]))
        pc_center = (pc_first + pc_last) / 2.0
        camera_unwrapped[finite_camera] += (
            round((pc_center - camera_center) / DAY_MS) * DAY_MS
        )
    try:
        mask = _coerce_valid_samples(
            valid_samples,
            valid_samples_path,
            n_samples=pc_daily.size,
            device_count=device_count,
        )
    except BaseException:
        if file_backed_pc is not None:
            mmap = getattr(file_backed_pc, "_mmap", None)
            if mmap is not None:
                mmap.close()
        raise
    if channels and mask is None:
        if file_backed_pc is not None:
            mmap = getattr(file_backed_pc, "_mmap", None)
            if mmap is not None:
                mmap.close()
        raise ValueError("requested validity channels require valid_samples")
    if mask is not None and channels and max(channels) >= mask.shape[1]:
        if isinstance(mask, np.memmap):
            mmap = getattr(mask, "_mmap", None)
            if mmap is not None:
                mmap.close()
        if file_backed_pc is not None:
            mmap = getattr(file_backed_pc, "_mmap", None)
            if mmap is not None:
                mmap.close()
        raise ValueError("requested validity channel exceeds valid_samples device count")

    try:
        return _map_loaded_camera_timestamps(
            pc_daily,
            wrap_indices,
            pc_first,
            pc_last,
            camera_daily,
            camera_unwrapped,
            mask,
            channels,
            max_distance_ms,
            retain_canonical_pc_unwrapped,
        )
    finally:
        if isinstance(mask, np.memmap):
            mmap = getattr(mask, "_mmap", None)
            if mmap is not None:
                mmap.close()
        if file_backed_pc is not None:
            mmap = getattr(file_backed_pc, "_mmap", None)
            if mmap is not None:
                mmap.close()
