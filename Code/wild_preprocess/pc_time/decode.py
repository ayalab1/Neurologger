"""Decode CE packed PC-clock updates without using legacy preprocessing code."""

from __future__ import annotations

import re
import struct
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Sequence, overload

import numpy as np


DAY_MS = 24 * 60 * 60 * 1000
PACKED_PC_MOD_MS = 1 << 20
PACKED_PC_MS_MASK = PACKED_PC_MOD_MS - 1
PACKED_PC_DELAY_SHIFT = 20
PACKED_PC_DELAY_MASK = (1 << 12) - 1
RAW_MISC_BLOCK_BYTES = 512
EXPANDED_SAMPLES_PER_RAW_CYCLE = 16
CE_PARAMS_DATE_OFFSET = 332
CE_PARAMS_TIME_OFFSET = 336
CE_PARAMS_DATE_SIZE = 4
CE_PARAMS_TIME_SIZE = 20


@dataclass(frozen=True)
class PcTimeLayout:
    """Description of the two packed-PC layouts used by supported recordings."""

    name: str
    raw_misc: bool
    raw_words_per_cycle: int = 16
    raw_low_word_index: int = 14
    raw_high_word_index: int = 15
    expanded_channel_count: int = 6
    expanded_low_channel_index: int = 3
    expanded_high_channel_index: int = 4
    expand_factor: int = EXPANDED_SAMPLES_PER_RAW_CYCLE


@dataclass(frozen=True)
class PackedUpdateDiagnostics:
    raw_candidate_run_count: int
    accepted_update_count: int
    rejected_unstable_run_count: int
    minimum_stable_cycles: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class PackedUpdates:
    """Stable packed-clock updates in raw analog-row coordinates.

    ``raw_row_indices`` are zero-based source rows: 16-word CE64 raw-misc
    cycles for :data:`CE64_RAW_MISC_LAYOUT`, or expanded analog frames for
    :data:`EXPANDED_ANALOG_LAYOUT`.  They are deliberately not multiplied by
    an ephys expansion factor.  Both arrays are copied and made read-only so
    the decoded evidence can safely be shared with a later analog mapping.
    """

    raw_row_indices: np.ndarray
    values: np.ndarray
    diagnostics: PackedUpdateDiagnostics

    def __post_init__(self) -> None:
        rows = np.array(self.raw_row_indices, dtype=np.int64, copy=True)
        values = np.array(self.values, dtype=np.uint32, copy=True)
        if rows.ndim != 1 or values.ndim != 1 or rows.size != values.size:
            raise ValueError("packed raw-row indices and values must be equal-length one-dimensional arrays")
        if np.any(rows < 0) or (rows.size > 1 and np.any(np.diff(rows) <= 0)):
            raise ValueError("packed raw-row indices must be non-negative and strictly increasing")
        rows.flags.writeable = False
        values.flags.writeable = False
        object.__setattr__(self, "raw_row_indices", rows)
        object.__setattr__(self, "values", values)


CE64_RAW_MISC_LAYOUT = PcTimeLayout(name="ce64-raw-misc", raw_misc=True)
EXPANDED_ANALOG_LAYOUT = PcTimeLayout(name="expanded-analog", raw_misc=False)


@dataclass(frozen=True)
class CeParamsHint:
    ephys_sample_rate_hz: int | None = None
    misc_sample_rate_hz: int | None = None
    recording_start_ms: int | None = None
    recording_date: str | None = None


def infer_recording_start_from_name(path: Path) -> int | None:
    """Return milliseconds since midnight from a WILD recording folder name."""

    name = Path(path).name if Path(path).is_dir() else Path(path).parent.name
    match = re.search(r"(?:^|_)(?P<hms>\d{6})(?:\.(?P<ms>\d{1,3}))?$", name)
    if match is None:
        return None
    hms = match.group("hms")
    hours, minutes, seconds = int(hms[:2]), int(hms[2:4]), int(hms[4:])
    millis = int((match.group("ms") or "0").ljust(3, "0"))
    if hours > 23 or minutes > 59 or seconds > 59:
        return None
    return ((hours * 3600 + minutes * 60 + seconds) * 1000) + millis


def _decode_ce_params_recording_start(data: bytes) -> tuple[int, str] | None:
    """Decode the CE RTC date and daily start time from the system header."""

    minimum_length = max(
        CE_PARAMS_DATE_OFFSET + CE_PARAMS_DATE_SIZE,
        CE_PARAMS_TIME_OFFSET + CE_PARAMS_TIME_SIZE,
    )
    if len(data) < minimum_length:
        return None
    _weekday, month, day, year = struct.unpack_from("<BBBB", data, CE_PARAMS_DATE_OFFSET)
    hours, minutes, seconds, _time_format = struct.unpack_from("<BBBB", data, CE_PARAMS_TIME_OFFSET)
    sub_seconds, second_fraction, _daylight_saving, _store_operation = struct.unpack_from(
        "<IIII", data, CE_PARAMS_TIME_OFFSET + 4
    )
    if year > 99:
        return None
    try:
        recording_date = date(2000 + year, month, day)
    except ValueError:
        return None
    if hours > 23 or minutes > 59 or seconds > 59:
        return None
    millis = 0
    if second_fraction > 0 and sub_seconds <= second_fraction:
        millis = ((second_fraction - sub_seconds) * 1000) // (second_fraction + 1)
    start_ms = ((hours * 3600 + minutes * 60 + seconds) * 1000) + millis
    return start_ms, recording_date.isoformat()


def decode_ce_params_recording_start_ms(data: bytes) -> int | None:
    """Decode milliseconds since midnight from the CE RTC metadata."""

    decoded = _decode_ce_params_recording_start(data)
    return decoded[0] if decoded is not None else None


def read_ce_params_hint(recording_folder: Path) -> CeParamsHint:
    path = Path(recording_folder) / "CE_params.bin"
    if not path.is_file():
        return CeParamsHint()
    data = path.read_bytes()[:512]
    if len(data) < 56:
        return CeParamsHint()
    ephys_rate = struct.unpack_from("<I", data, 0)[0]
    sampling_rate_0 = struct.unpack_from("<I", data, 40)[0]
    misc_rate = struct.unpack_from("<I", data, 52)[0]
    decoded_start = _decode_ce_params_recording_start(data)
    return CeParamsHint(
        ephys_sample_rate_hz=(ephys_rate or sampling_rate_0) or None,
        misc_sample_rate_hz=misc_rate or None,
        recording_start_ms=(decoded_start[0] if decoded_start is not None else None),
        recording_date=(decoded_start[1] if decoded_start is not None else None),
    )


def resolve_recording_start_ms(
    recording_folder: Path,
    *,
    explicit_recording_start_ms: int | None = None,
    allow_folder_name_fallback: bool = False,
) -> tuple[int, str]:
    """Resolve an absolute daily anchor without trusting folder names by default."""

    if explicit_recording_start_ms is not None:
        if not 0 <= int(explicit_recording_start_ms) < DAY_MS:
            raise ValueError("explicit recording start must be milliseconds in [0, 86400000)")
        return int(explicit_recording_start_ms), "explicit"
    hint = read_ce_params_hint(recording_folder)
    if hint.recording_start_ms is not None:
        return hint.recording_start_ms, "CE_params.bin"
    if allow_folder_name_fallback:
        from_name = infer_recording_start_from_name(recording_folder)
        if from_name is not None:
            return from_name, "recording folder name (explicit fallback)"
        raise ValueError(
            "No absolute recording-start anchor in CE_params.bin or the explicit worker job, and "
            "the recording folder name is not a supported HHMMSS[.mmm] fallback."
        )
    raise ValueError(
        "No absolute recording-start anchor in CE_params.bin or the explicit worker job; "
        "recording-folder fallback is disabled."
    )


def validate_recording_start_compatibility(
    anchors: Sequence[tuple[int, str | None]],
    *,
    maximum_difference_ms: int = 30_000,
) -> None:
    """Require selected recording starts to describe one acquisition session.

    Full CE dates are authoritative when every anchor has one. Legacy anchors
    without dates retain the prior circular time-of-day comparison so an
    explicit folder-name recovery remains possible.
    """

    if maximum_difference_ms < 0:
        raise ValueError("maximum_difference_ms must be non-negative")
    normalized: list[tuple[int, str | None]] = []
    for start_ms, recording_date in anchors:
        value = int(start_ms)
        if not 0 <= value < DAY_MS:
            raise ValueError("recording start anchor is outside one day")
        normalized.append((value, recording_date))
    if len(normalized) < 2:
        return
    if all(recording_date is not None for _, recording_date in normalized):
        absolute: list[datetime] = []
        for start_ms, recording_date in normalized:
            try:
                day = date.fromisoformat(str(recording_date))
            except ValueError as exc:
                raise ValueError(f"invalid recording date {recording_date!r}") from exc
            absolute.append(
                datetime.combine(day, datetime.min.time())
                + timedelta(milliseconds=start_ms)
            )
        for left in range(len(absolute)):
            for right in range(left + 1, len(absolute)):
                difference = abs((absolute[right] - absolute[left]).total_seconds() * 1000.0)
                if difference > maximum_difference_ms:
                    raise ValueError(
                        f"selected recording starts differ by {difference / 1000:.3f}s "
                        f"(devices {left + 1} and {right + 1}; limit "
                        f"{maximum_difference_ms / 1000:g}s)"
                    )
        return
    for left in range(len(normalized)):
        for right in range(left + 1, len(normalized)):
            first = normalized[left][0]
            second = normalized[right][0]
            difference = abs((second - first + DAY_MS // 2) % DAY_MS - DAY_MS // 2)
            if difference > maximum_difference_ms:
                raise ValueError(
                    f"selected recording starts differ by {difference / 1000:.3f}s "
                    f"(devices {left + 1} and {right + 1}; limit "
                    f"{maximum_difference_ms / 1000:g}s)"
                )


def _packed_updates_from_words(
    words: np.ndarray,
    indices: np.ndarray,
    *,
    minimum_stable_cycles: int = 2,
) -> tuple[np.ndarray, np.ndarray, PackedUpdateDiagnostics]:
    packed = words[:, 0].astype(np.uint32) | (words[:, 1].astype(np.uint32) << 16)
    if packed.size == 0:
        diagnostics = PackedUpdateDiagnostics(0, 0, 0, minimum_stable_cycles)
        return indices.astype(np.int64), packed, diagnostics
    run_starts = np.r_[0, np.flatnonzero(packed[1:] != packed[:-1]) + 1]
    run_ends = np.r_[run_starts[1:], packed.size]
    run_values = packed[run_starts]
    run_lengths = run_ends - run_starts
    candidates = run_values != 0
    stable = candidates & (run_lengths >= minimum_stable_cycles)
    accepted_values = run_values[stable]
    accepted_indices = indices[run_starts[stable]]
    if accepted_values.size:
        changed = np.r_[True, accepted_values[1:] != accepted_values[:-1]]
        accepted_values = accepted_values[changed]
        accepted_indices = accepted_indices[changed]
    diagnostics = PackedUpdateDiagnostics(
        raw_candidate_run_count=int(np.count_nonzero(candidates)),
        accepted_update_count=int(accepted_values.size),
        rejected_unstable_run_count=int(np.count_nonzero(candidates & ~stable)),
        minimum_stable_cycles=minimum_stable_cycles,
    )
    return (
        accepted_indices.astype(np.int64),
        accepted_values.astype(np.uint32),
        diagnostics,
    )


def _normalise_valid_raw_runs(
    n_rows: int,
    *,
    valid_raw_runs: Sequence[tuple[int, int]] | None,
    raw_valid_mask: np.ndarray | Sequence[bool] | None,
) -> tuple[tuple[int, int], ...]:
    """Return deterministic, merged half-open valid source-row runs."""

    if valid_raw_runs is not None and raw_valid_mask is not None:
        raise ValueError("specify either valid_raw_runs or raw_valid_mask, not both")
    if raw_valid_mask is not None:
        mask = np.asarray(raw_valid_mask)
        if mask.ndim != 1 or mask.size != n_rows or mask.dtype != np.bool_:
            raise ValueError("raw_valid_mask must be a one-dimensional boolean array matching raw rows")
        starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
        ends = np.flatnonzero(mask & np.r_[~mask[1:], True]) + 1
        return tuple((int(start), int(end)) for start, end in zip(starts, ends))
    if valid_raw_runs is None:
        return ((0, n_rows),)

    normalized: list[tuple[int, int]] = []
    for item in valid_raw_runs:
        if len(item) != 2:
            raise ValueError("valid_raw_runs entries must be (start, end) half-open pairs")
        start, end = (int(item[0]), int(item[1]))
        if start < 0 or end > n_rows or start >= end:
            raise ValueError("valid_raw_runs must be non-empty half-open intervals within raw rows")
        normalized.append((start, end))
    normalized.sort()
    merged: list[tuple[int, int]] = []
    for start, end in normalized:
        if merged and start < merged[-1][1]:
            raise ValueError("valid_raw_runs must not overlap")
        if merged and start == merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return tuple(merged)


def _collect_packed_update_rows_streaming(
    n_rows: int,
    read_packed_rows: Callable[[int, int], np.ndarray],
    *,
    valid_raw_runs: Sequence[tuple[int, int]] | None,
    raw_valid_mask: np.ndarray | Sequence[bool] | None,
    minimum_stable_cycles: int,
    chunk_rows: int,
) -> PackedUpdates:
    """Decode valid source runs with bounded memory and no gap bridging."""

    runs = _normalise_valid_raw_runs(
        n_rows,
        valid_raw_runs=valid_raw_runs,
        raw_valid_mask=raw_valid_mask,
    )
    row_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    candidate_count = 0
    accepted_count = 0
    rejected_count = 0
    # A validity gap does not create a new packed-clock update. Keep the last
    # accepted word across support-run boundaries so a value held on both
    # sides is emitted once at its original transition, not once per fragment.
    previous_accepted_value: int | None = None
    for start, end in runs:
        run_value: int | None = None
        run_start = start
        run_length = 0

        def finish_run() -> None:
            nonlocal candidate_count, accepted_count, previous_accepted_value, rejected_count
            if run_value is None or run_value == 0:
                return
            candidate_count += 1
            if run_length < minimum_stable_cycles:
                rejected_count += 1
                return
            if previous_accepted_value == run_value:
                return
            row_parts.append(np.asarray([run_start], dtype=np.int64))
            value_parts.append(np.asarray([run_value], dtype=np.uint32))
            accepted_count += 1
            previous_accepted_value = run_value

        for chunk_start in range(start, end, chunk_rows):
            chunk_end = min(chunk_start + chunk_rows, end)
            packed = np.asarray(read_packed_rows(chunk_start, chunk_end), dtype=np.uint32)
            if packed.ndim != 1 or packed.size != chunk_end - chunk_start:
                raise ValueError("packed-row reader returned an unexpected shape")
            change_starts = np.r_[0, np.flatnonzero(packed[1:] != packed[:-1]) + 1]
            change_ends = np.r_[change_starts[1:], packed.size]
            for local_start, local_end in zip(change_starts, change_ends):
                value = int(packed[local_start])
                length = int(local_end - local_start)
                absolute_start = chunk_start + int(local_start)
                if run_value == value:
                    run_length += length
                    continue
                finish_run()
                run_value = value
                run_start = absolute_start
                run_length = length
            # A valid run is deliberately not finalized at a chunk boundary.
        finish_run()
    raw_rows = np.concatenate(row_parts) if row_parts else np.empty(0, dtype=np.int64)
    values = np.concatenate(value_parts) if value_parts else np.empty(0, dtype=np.uint32)
    return PackedUpdates(
        raw_rows,
        values,
        PackedUpdateDiagnostics(
            raw_candidate_run_count=candidate_count,
            accepted_update_count=accepted_count,
            rejected_unstable_run_count=rejected_count,
            minimum_stable_cycles=minimum_stable_cycles,
        ),
    )


@overload
def collect_packed_updates(
    analog_path: Path,
    layout: PcTimeLayout = CE64_RAW_MISC_LAYOUT,
    *,
    return_diagnostics: bool = False,
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def collect_packed_updates(
    analog_path: Path,
    layout: PcTimeLayout,
    *,
    return_diagnostics: bool,
) -> tuple[np.ndarray, np.ndarray, PackedUpdateDiagnostics]: ...


def collect_packed_updates(
    analog_path: Path,
    layout: PcTimeLayout = CE64_RAW_MISC_LAYOUT,
    *,
    return_diagnostics: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, PackedUpdateDiagnostics]:
    """Return stable unique packed updates and their ephys-sample coordinates.

    CE64 raw-misc cycles are expanded to ephys coordinates using ``expand_factor``.
    The expanded-analog layout is kept for legacy recordings with six int16 lanes.
    A nonzero packed value must persist for at least two source cycles; shorter
    runs are retained only in optional decoder diagnostics.
    """

    analog_path = Path(analog_path)
    size = analog_path.stat().st_size
    if layout.raw_misc:
        cycle_bytes = layout.raw_words_per_cycle * 2
        if cycle_bytes <= 0 or RAW_MISC_BLOCK_BYTES % cycle_bytes:
            raise ValueError("raw words per cycle must divide 512-byte raw-misc blocks")
        if size == 0 or size % RAW_MISC_BLOCK_BYTES:
            raise ValueError("raw-misc analogin.dat length must be a non-zero multiple of 512 bytes")
        words = np.memmap(analog_path, dtype="<u2", mode="r")
        try:
            cycles = words.reshape(-1, layout.raw_words_per_cycle)
            lanes = cycles[:, [layout.raw_low_word_index, layout.raw_high_word_index]]
            indices = np.arange(cycles.shape[0], dtype=np.int64) * layout.expand_factor
            result = _packed_updates_from_words(np.asarray(lanes), indices)
            return result if return_diagnostics else result[:2]
        finally:
            mmap = getattr(words, "_mmap", None)
            if mmap is not None:
                mmap.close()
    frame_bytes = layout.expanded_channel_count * 2
    if size == 0 or size % frame_bytes:
        raise ValueError("expanded analogin.dat length must be a non-zero whole number of frames")
    words = np.memmap(analog_path, dtype="<u2", mode="r").reshape(-1, layout.expanded_channel_count)
    try:
        lanes = words[:, [layout.expanded_low_channel_index, layout.expanded_high_channel_index]]
        indices = np.arange(words.shape[0], dtype=np.int64)
        result = _packed_updates_from_words(np.asarray(lanes), indices)
        return result if return_diagnostics else result[:2]
    finally:
        mmap = getattr(words, "_mmap", None)
        if mmap is not None:
            mmap.close()


def collect_packed_update_rows(
    analog_path: Path,
    layout: PcTimeLayout = CE64_RAW_MISC_LAYOUT,
    *,
    valid_raw_runs: Sequence[tuple[int, int]] | None = None,
    raw_valid_mask: np.ndarray | Sequence[bool] | None = None,
    minimum_stable_cycles: int = 2,
    chunk_rows: int = 1_000_000,
) -> PackedUpdates:
    """Return immutable stable updates on raw analog-row coordinates.

    Valid source intervals are half-open ``[start, end)`` raw-row intervals.
    Alternatively, ``raw_valid_mask`` supplies one boolean per raw source row.
    Each valid run is decoded independently: a packed value held on both sides
    of an invalid overwrite/reorder region never becomes one stable update.
    ``chunk_rows`` bounds decoder working memory and exists mainly for testing;
    it does not change the resulting updates or diagnostics.

    The existing :func:`collect_packed_updates` remains the legacy interface
    for ephys-expanded CE64 coordinates and is intentionally unchanged.
    """

    if int(minimum_stable_cycles) < 1:
        raise ValueError("minimum_stable_cycles must be positive")
    if int(chunk_rows) < 1:
        raise ValueError("chunk_rows must be positive")
    analog_path = Path(analog_path)
    size = analog_path.stat().st_size
    if layout.raw_misc:
        cycle_bytes = layout.raw_words_per_cycle * 2
        if cycle_bytes <= 0 or RAW_MISC_BLOCK_BYTES % cycle_bytes:
            raise ValueError("raw words per cycle must divide 512-byte raw-misc blocks")
        if size == 0 or size % RAW_MISC_BLOCK_BYTES:
            raise ValueError("raw-misc analogin.dat length must be a non-zero multiple of 512 bytes")
        words = np.memmap(analog_path, dtype="<u2", mode="r")
        try:
            cycles = words.reshape(-1, layout.raw_words_per_cycle)
            def read_packed_rows(start: int, end: int) -> np.ndarray:
                low = cycles[start:end, layout.raw_low_word_index].astype(np.uint32)
                high = cycles[start:end, layout.raw_high_word_index].astype(np.uint32)
                return low | (high << 16)

            return _collect_packed_update_rows_streaming(
                int(cycles.shape[0]),
                read_packed_rows,
                valid_raw_runs=valid_raw_runs,
                raw_valid_mask=raw_valid_mask,
                minimum_stable_cycles=int(minimum_stable_cycles),
                chunk_rows=int(chunk_rows),
            )
        finally:
            mmap = getattr(words, "_mmap", None)
            if mmap is not None:
                mmap.close()
    frame_bytes = layout.expanded_channel_count * 2
    if size == 0 or size % frame_bytes:
        raise ValueError("expanded analogin.dat length must be a non-zero whole number of frames")
    words = np.memmap(analog_path, dtype="<u2", mode="r")
    try:
        frames = words.reshape(-1, layout.expanded_channel_count)
        def read_packed_rows(start: int, end: int) -> np.ndarray:
            low = frames[start:end, layout.expanded_low_channel_index].astype(np.uint32)
            high = frames[start:end, layout.expanded_high_channel_index].astype(np.uint32)
            return low | (high << 16)

        return _collect_packed_update_rows_streaming(
            int(frames.shape[0]),
            read_packed_rows,
            valid_raw_runs=valid_raw_runs,
            raw_valid_mask=raw_valid_mask,
            minimum_stable_cycles=int(minimum_stable_cycles),
            chunk_rows=int(chunk_rows),
        )
    finally:
        mmap = getattr(words, "_mmap", None)
        if mmap is not None:
            mmap.close()


def unpack_packed_updates(packed_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    packed = np.asarray(packed_values, dtype=np.uint32)
    pc_ms20 = (packed & PACKED_PC_MS_MASK).astype(float)
    delay_ms = ((packed >> PACKED_PC_DELAY_SHIFT) & PACKED_PC_DELAY_MASK).astype(float)
    corrected_ms20 = np.mod(pc_ms20 + delay_ms, PACKED_PC_MOD_MS)
    return pc_ms20, delay_ms, corrected_ms20
