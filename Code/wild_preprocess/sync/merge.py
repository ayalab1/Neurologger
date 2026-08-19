from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ..binary_io import atomic_output_path, close_memmap, interleaved_memmap, replace_atomic
from ..analog import (
    IMU_MODALITY_INVALID_KINDS,
    AnalogIntegrityResult,
    AnalogTimelineResult,
    DeviceClockPrior,
    build_event_driven_analog_segments,
    write_canonical_analog,
)
from ..models import (
    ClassifiedInterval,
    DeviceGap,
    DeviceSourceStep,
    DeviceSyncSegment,
    DeviceTerminalSupport,
    Recording,
    SyncModel,
)
from .segments import map_canonical_positions, validate_segment_collection
from ..version import SYNC_ALGORITHM_VERSION


ProgressCallback = Callable[[str, float], None]
StageCallback = Callable[[Path, dict[str, str]], None]
TimingCallback = Callable[[str, str, int, int], None]
ANALOG_FS = 1250.0
EPHYS_SINC_HALF_WIDTH = 16
EPHYS_SINC_KAISER_BETA = 8.6
INTEGER_MAPPING_TOLERANCE_SAMPLES = 1e-6
MAX_TIME_DAT_SAMPLES = int(np.iinfo(np.int32).max) + 1


def validate_time_dat_length(n_samples: int) -> None:
    """Reject output lengths whose relative ``int32`` time axis would wrap."""

    if n_samples <= 0:
        raise ValueError("time.dat requires at least one output sample")
    if n_samples > MAX_TIME_DAT_SAMPLES:
        raise ValueError(
            f"time.dat cannot represent {n_samples} samples as signed int32 relative indices "
            f"(maximum {MAX_TIME_DAT_SAMPLES})"
        )


def _safe_managed_names(names: object) -> set[str]:
    """Accept only flat filenames from a prior run manifest."""

    if not isinstance(names, list):
        return set()
    safe: set[str] = set()
    for name in names:
        if not isinstance(name, str):
            continue
        candidate = Path(name)
        if candidate.name == name and name not in {"", ".", ".."}:
            safe.add(name)
    return safe


def _previous_managed_names(output_folder: Path) -> set[str]:
    manifest = output_folder / "wild_preprocess_run.json"
    if not manifest.is_file():
        return set()
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    return _safe_managed_names(payload.get("managed_files"))


def _device_models(
    recordings: list[Recording], master_index: int, pair_models: dict[int, SyncModel]
) -> list[SyncModel]:
    models: list[SyncModel] = []
    for index, _recording in enumerate(recordings):
        if index == master_index:
            models.append(
                SyncModel(
                    intercept_samples=0.0,
                    slope_samples_per_second=0.0,
                    drift_ppm=0.0,
                    residual_rms_samples=0.0,
                    residual_max_abs_samples=0.0,
                    accepted_count=0,
                    observation_count=0,
                )
            )
        else:
            models.append(pair_models[index])
    return models


def _device_gap_map(device_gaps: list[DeviceGap]) -> dict[int, tuple[DeviceGap, ...]]:
    mapped: dict[int, list[DeviceGap]] = {}
    for gap in sorted(device_gaps, key=lambda item: (item.device_index, item.canonical_start_sample)):
        mapped.setdefault(gap.device_index - 1, []).append(gap)
    return {index: tuple(values) for index, values in mapped.items()}


def _device_source_step_map(
    source_steps: list[DeviceSourceStep],
) -> dict[int, tuple[DeviceSourceStep, ...]]:
    mapped: dict[int, list[DeviceSourceStep]] = {}
    for step in sorted(source_steps, key=lambda item: (item.device_index, item.canonical_sample)):
        mapped.setdefault(step.device_index - 1, []).append(step)
    return {index: tuple(values) for index, values in mapped.items()}


def _device_interval_map(
    intervals: list[ClassifiedInterval],
) -> dict[int, tuple[ClassifiedInterval, ...]]:
    mapped: dict[int, list[ClassifiedInterval]] = {}
    for interval in intervals:
        if interval.action != "zero_fill":
            continue
        for device_index in interval.affected_device_indices:
            mapped.setdefault(device_index - 1, []).append(interval)
    return {
        index: tuple(sorted(values, key=lambda item: item.canonical_start_sample))
        for index, values in mapped.items()
    }


def _device_terminal_map(
    supports: list[DeviceTerminalSupport],
) -> dict[int, DeviceTerminalSupport]:
    mapped: dict[int, DeviceTerminalSupport] = {}
    for support in supports:
        index = support.device_index - 1
        previous = mapped.get(index)
        if previous is None or support.supported_canonical_end_sample < previous.supported_canonical_end_sample:
            mapped[index] = support
    return mapped


def _device_segment_map(
    segments: list[DeviceSyncSegment],
    *,
    device_count: int,
) -> dict[int, tuple[DeviceSyncSegment, ...]]:
    grouped: dict[int, list[DeviceSyncSegment]] = {}
    for segment in segments:
        if segment.device_index > device_count:
            raise ValueError(f"segment device index {segment.device_index} is outside recordings")
        grouped.setdefault(segment.device_index - 1, []).append(segment)
    return {
        device_index: validate_segment_collection(
            grouped.get(device_index, ()), device_index=device_index + 1
        )
        for device_index in range(device_count)
    }


def _validity_device_order(recordings: list[Recording], master_index: int) -> list[int]:
    """Return source-device indices in the stable validity-channel order."""

    if not 0 <= master_index < len(recordings):
        raise ValueError(f"Invalid master index {master_index} for {len(recordings)} recordings")
    return [master_index, *(index for index in range(len(recordings)) if index != master_index)]


def _source_coordinates(
    model: SyncModel,
    canonical_positions: np.ndarray,
    gaps: tuple[DeviceGap, ...],
    *,
    fs: float,
    guard_samples: int = 0,
    source_steps: tuple[DeviceSourceStep, ...] = (),
    invalid_intervals: tuple[ClassifiedInterval, ...] = (),
    terminal_support: DeviceTerminalSupport | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map canonical positions to compressed device coordinates and validity."""

    source = model.source_scale(fs) * canonical_positions + model.intercept_samples
    valid = np.ones(canonical_positions.size, dtype=bool)
    for gap in gaps:
        start = float(gap.canonical_start_sample)
        end = float(gap.canonical_end_sample)
        source -= gap.missing_samples * (canonical_positions >= end)
        valid &= ~(
            (canonical_positions >= start - guard_samples)
            & (canonical_positions < end + guard_samples)
        )
    for step in source_steps:
        source += step.source_step_samples * (
            canonical_positions >= step.canonical_sample
        )
    for interval in invalid_intervals:
        expansion = guard_samples if interval.kind in {"missing", "duplicate_destination"} else 0
        valid &= ~(
            (canonical_positions >= interval.canonical_start_sample - expansion)
            & (canonical_positions < interval.canonical_end_sample + expansion)
        )
    if terminal_support is not None:
        valid &= canonical_positions < terminal_support.supported_canonical_end_sample
    return source, valid


def _apply_neural_validity_exclusions(
    valid: np.ndarray,
    canonical_positions: np.ndarray,
    *,
    gaps: tuple[DeviceGap, ...],
    guard_samples: int,
    invalid_intervals: tuple[ClassifiedInterval, ...],
    terminal_support: DeviceTerminalSupport | None,
) -> np.ndarray:
    """Apply explicit invalidity decisions without changing a segment mapping."""

    result = np.asarray(valid, dtype=bool).copy()
    for gap in gaps:
        result &= ~(
            (canonical_positions >= gap.canonical_start_sample - guard_samples)
            & (canonical_positions < gap.canonical_end_sample + guard_samples)
        )
    for interval in invalid_intervals:
        expansion = guard_samples if interval.kind in {"missing", "duplicate_destination"} else 0
        result &= ~(
            (canonical_positions >= interval.canonical_start_sample - expansion)
            & (canonical_positions < interval.canonical_end_sample + expansion)
        )
    if terminal_support is not None:
        result &= canonical_positions < terminal_support.supported_canonical_end_sample
    return result


def _common_master_interval(
    recordings: list[Recording],
    models: list[SyncModel],
    master_index: int,
    device_gaps: list[DeviceGap] | None = None,
    minimum_common_start: int = 0,
    maximum_common_end: int | None = None,
    maximum_common_end_device_index: int | None = None,
    maximum_common_end_reason: str = "",
    preserve_device_tails: bool = False,
    device_sync_segments_supplied: bool = False,
) -> tuple[int, int, dict[str, object]]:
    fs = recordings[master_index].fs
    lower_candidates: list[dict[str, object]] = [
        {
            "device_index": master_index + 1,
            "device_name": recordings[master_index].device_name,
            "stream": "master_output_coordinate",
            "candidate_master_sample": 0.0,
        }
    ]
    if minimum_common_start > 0:
        lower_candidates.append(
            {
                "device_index": master_index + 1,
                "device_name": recordings[master_index].device_name,
                "stream": "validated_endpoint_probe",
                "candidate_master_sample": float(minimum_common_start),
            }
        )
    upper_candidates: list[dict[str, object]] = []
    if maximum_common_end is not None:
        limiter_device_index = maximum_common_end_device_index or master_index + 1
        upper_candidates.append(
            {
                "device_index": limiter_device_index,
                "device_name": recordings[limiter_device_index - 1].device_name,
                "stream": "validated_terminal_sync",
                "candidate_master_sample": float(maximum_common_end),
                "reason": maximum_common_end_reason,
            }
        )
    gaps_by_device = _device_gap_map(list(device_gaps or ()))
    for device_index, (recording, model) in enumerate(zip(recordings, models), start=1):
        scale = model.source_scale(fs)
        if scale <= 0:
            raise ValueError(f"Invalid source scale {scale} for {recording.folder}")
        total_missing = sum(
            gap.missing_samples for gap in gaps_by_device.get(device_index - 1, ())
        )
        # Windowed-sinc ephys interpolation needs symmetric source support.
        candidates_enabled = (
            device_index - 1 == master_index
            or (not preserve_device_tails and not device_sync_segments_supplied)
        )
        if candidates_enabled:
            lower_candidates.append(
            {
                "device_index": device_index,
                "device_name": recording.device_name,
                "stream": "ephys",
                "candidate_master_sample": (EPHYS_SINC_HALF_WIDTH - model.intercept_samples) / scale,
            }
            )
            upper_candidates.append(
            {
                "device_index": device_index,
                "device_name": recording.device_name,
                "stream": "ephys",
                "candidate_master_sample": (
                    recording.n_samples - 1 - EPHYS_SINC_HALF_WIDTH
                    - model.intercept_samples + total_missing
                )
                / scale,
            }
            )
        res_rate = fs / ANALOG_FS
        if candidates_enabled:
            upper_candidates.append(
            {
                "device_index": device_index,
                "device_name": recording.device_name,
                "stream": "analog",
                "candidate_master_sample": (
                    (recording.analog_samples - 1) * res_rate
                    - model.intercept_samples
                )
                / scale,
            }
            )
    start_limiter = max(lower_candidates, key=lambda item: float(item["candidate_master_sample"]))
    end_limiter = min(upper_candidates, key=lambda item: float(item["candidate_master_sample"]))
    lower = float(start_limiter["candidate_master_sample"])
    upper = float(end_limiter["candidate_master_sample"])
    start = int(np.ceil(lower))
    end = int(np.floor(upper))
    if end <= start:
        raise ValueError("No common valid interval remains across all recordings.")
    return start, end, {
        "start_limiter": start_limiter,
        "end_limiter": end_limiter,
    }


def _linear_resampled_chunk(
    mapped: np.memmap,
    source_positions: np.ndarray,
) -> np.ndarray:
    lower = np.floor(source_positions).astype(np.int64)
    upper = np.minimum(lower + 1, mapped.shape[0] - 1)
    if lower.min(initial=0) < 0 or upper.max(initial=0) >= mapped.shape[0]:
        raise ValueError("Source mapping left the valid recording interval.")
    fraction = (source_positions - lower).astype(np.float32)
    low_values = np.asarray(mapped[lower], dtype=np.float32)
    if np.all(fraction == 0):
        return low_values.astype(np.int16)
    high_values = np.asarray(mapped[upper], dtype=np.float32)
    values = low_values + (high_values - low_values) * fraction[:, None]
    return np.clip(np.rint(values), np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)


def _windowed_sinc_resampled_chunk(
    mapped: np.memmap,
    source_positions: np.ndarray,
    *,
    half_width: int = EPHYS_SINC_HALF_WIDTH,
    cutoff: float = 1.0,
) -> np.ndarray:
    """Band-limited fractional-delay interpolation for raw neural samples."""

    if not 0 < cutoff <= 1:
        raise ValueError(f"Invalid normalized sinc cutoff: {cutoff}")
    output = np.empty((source_positions.size, mapped.shape[1]), dtype=np.int16)
    tap_offsets = np.arange(-half_width + 1, half_width + 1, dtype=np.int64)
    window = np.kaiser(tap_offsets.size, EPHYS_SINC_KAISER_BETA)
    for block_start in range(0, source_positions.size, 4096):
        block_end = min(source_positions.size, block_start + 4096)
        positions = source_positions[block_start:block_end]
        nearest = np.rint(positions).astype(np.int64)
        if cutoff >= 1.0 - 1e-12 and np.all(
            np.abs(positions - nearest) <= INTEGER_MAPPING_TOLERANCE_SAMPLES
        ):
            if nearest.min(initial=0) < 0 or nearest.max(initial=0) >= mapped.shape[0]:
                raise ValueError("Source mapping left the valid recording interval.")
            output[block_start:block_end] = np.asarray(mapped[nearest], dtype=np.int16)
            continue
        centers = np.floor(positions).astype(np.int64)
        indices = centers[:, None] + tap_offsets[None, :]
        if indices.min(initial=0) < 0 or indices.max(initial=0) >= mapped.shape[0]:
            raise ValueError("Windowed-sinc support left the valid recording interval.")
        distances = positions[:, None] - indices
        weights = cutoff * np.sinc(cutoff * distances) * window[None, :]
        weights /= np.sum(weights, axis=1, keepdims=True)
        values = np.einsum(
            "st,stc->sc",
            weights,
            np.asarray(mapped[indices], dtype=np.float32),
            optimize=True,
        )
        output[block_start:block_end] = np.clip(
            np.rint(values), np.iinfo(np.int16).min, np.iinfo(np.int16).max
        ).astype(np.int16)
    return output


def _write_interleaved_stream(
    output_path: Path,
    recordings: list[Recording],
    models: list[SyncModel],
    master_index: int,
    common_start: int,
    common_end: int,
    *,
    stream: str,
    chunk_seconds: float,
    overwrite: bool,
    progress: ProgressCallback | None,
    device_gaps: list[DeviceGap] | None = None,
    validity_path: Path | None = None,
    classified_intervals: list[ClassifiedInterval] | None = None,
    device_source_steps: list[DeviceSourceStep] | None = None,
    device_terminal_support: list[DeviceTerminalSupport] | None = None,
    device_sync_segments: list[DeviceSyncSegment] | None = None,
) -> tuple[int, int]:
    if validity_path is not None and stream != "ephys":
        raise ValueError("validity output is defined only for the ephys stream")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; enable overwrite to regenerate: {output_path}")
    if validity_path is not None and validity_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists; enable overwrite to regenerate: {validity_path}"
        )
    partial = atomic_output_path(output_path)
    if partial.exists():
        partial.unlink()
    validity_partial = atomic_output_path(validity_path) if validity_path is not None else None
    if validity_partial is not None and validity_partial.exists():
        validity_partial.unlink()
    fs = recordings[master_index].fs
    if stream == "ephys":
        out_fs = float(fs)
        channels = [recording.n_channels for recording in recordings]
        mapped = [
            interleaved_memmap(recording.amplifier_file, recording.n_channels, recording.n_samples)
            for recording in recordings
        ]
        output_count = common_end - common_start + 1
        chunk_samples = max(1, round(chunk_seconds * fs))
    elif stream == "analog":
        out_fs = ANALOG_FS
        channels = [recording.analog_channels for recording in recordings]
        mapped = [
            interleaved_memmap(recording.analog_file, recording.analog_channels, recording.analog_samples)
            for recording in recordings
        ]
        res_rate = fs / ANALOG_FS
        output_count = int(np.floor((common_end - common_start) / res_rate)) + 1
        chunk_samples = max(1, round(chunk_seconds * ANALOG_FS))
    else:
        raise ValueError(f"Unknown stream: {stream}")
    total_channels = sum(channels)
    gaps_by_device = _device_gap_map(list(device_gaps or ()))
    steps_by_device = _device_source_step_map(list(device_source_steps or ()))
    intervals_by_device = _device_interval_map(list(classified_intervals or ()))
    terminal_by_device = _device_terminal_map(list(device_terminal_support or ()))
    segments_by_device = (
        _device_segment_map(device_sync_segments, device_count=len(recordings))
        if device_sync_segments is not None
        else None
    )
    validity_order = _validity_device_order(recordings, master_index) if validity_path is not None else []
    try:
        with ExitStack() as stack:
            output = stack.enter_context(partial.open("wb"))
            validity_output = (
                stack.enter_context(validity_partial.open("wb"))
                if validity_partial is not None
                else None
            )
            for start in range(0, output_count, chunk_samples):
                count = min(chunk_samples, output_count - start)
                if stream == "ephys":
                    master_positions = common_start + start + np.arange(count, dtype=np.float64)
                else:
                    master_positions = common_start + (start + np.arange(count, dtype=np.float64)) * (fs / ANALOG_FS)
                combined = np.empty((count, total_channels), dtype=np.int16)
                validity = (
                    np.empty((count, len(recordings)), dtype=np.uint8)
                    if validity_partial is not None
                    else None
                )
                channel_start = 0
                for device_index, (recording, model, source, n_channels) in enumerate(
                    zip(recordings, models, mapped, channels)
                ):
                    guard = EPHYS_SINC_HALF_WIDTH if stream == "ephys" else int(np.ceil(fs / ANALOG_FS))
                    apply_neural_integrity = stream == "ephys"
                    if apply_neural_integrity and segments_by_device is not None:
                        source_ephys, valid = map_canonical_positions(
                            segments_by_device.get(device_index, ()),
                            master_positions,
                            source_sample_count=recording.n_samples,
                            interpolation_half_width=EPHYS_SINC_HALF_WIDTH,
                            device_index=device_index + 1,
                        )
                        valid = _apply_neural_validity_exclusions(
                            valid,
                            master_positions,
                            gaps=gaps_by_device.get(device_index, ()),
                            guard_samples=guard,
                            invalid_intervals=intervals_by_device.get(device_index, ()),
                            terminal_support=terminal_by_device.get(device_index),
                        )
                    else:
                        source_ephys, valid = _source_coordinates(
                            model,
                            master_positions,
                            gaps_by_device.get(device_index, ()) if apply_neural_integrity else (),
                            fs=fs,
                            guard_samples=guard,
                            source_steps=(
                                steps_by_device.get(device_index, ())
                                if apply_neural_integrity
                                else ()
                            ),
                            invalid_intervals=(
                                intervals_by_device.get(device_index, ())
                                if apply_neural_integrity
                                else ()
                            ),
                            terminal_support=(
                                terminal_by_device.get(device_index)
                                if apply_neural_integrity
                                else None
                            ),
                    )
                    source_positions = source_ephys if stream == "ephys" else source_ephys / (fs / ANALOG_FS)
                    if stream == "ephys":
                        support_positions = np.where(valid, source_positions, 0.0)
                        nearest = np.rint(support_positions)
                        integer_mapping = (
                            np.abs(support_positions - nearest)
                            <= INTEGER_MAPPING_TOLERANCE_SAMPLES
                        )
                        supported = np.where(
                            integer_mapping,
                            (nearest >= 0) & (nearest < source.shape[0]),
                            (support_positions >= EPHYS_SINC_HALF_WIDTH - 1)
                            & (support_positions <= source.shape[0] - 1 - EPHYS_SINC_HALF_WIDTH),
                        )
                    else:
                        supported = (
                            (source_positions >= 0)
                            & (source_positions <= source.shape[0] - 1)
                        )
                    valid &= supported
                    data = np.zeros((count, n_channels), dtype=np.int16)
                    valid_indices = np.flatnonzero(valid)
                    runs: list[np.ndarray] = []
                    if valid_indices.size:
                        boundaries = np.flatnonzero(np.diff(valid_indices) > 1) + 1
                        runs = list(np.split(valid_indices, boundaries))
                    if stream == "ephys":
                        if segments_by_device is not None:
                            # A chunk can contain adjacent independently fitted
                            # segments with different sample-rate scales.  Split
                            # renderer runs at the authoritative segment bounds
                            # so each uses its own anti-alias cutoff.
                            for segment in segments_by_device.get(device_index, ()):
                                if not segment.is_publishable:
                                    continue
                                segment_indices = np.flatnonzero(
                                    valid
                                    & (master_positions >= segment.canonical_start_sample)
                                    & (master_positions < segment.canonical_end_sample)
                                )
                                if not segment_indices.size:
                                    continue
                                boundaries = np.flatnonzero(np.diff(segment_indices) > 1) + 1
                                for run in np.split(segment_indices, boundaries):
                                    data[run] = _windowed_sinc_resampled_chunk(
                                        source,
                                        source_positions[run],
                                        cutoff=min(1.0, 1.0 / segment.source_scale),
                                    )
                        else:
                            for run in runs:
                                data[run] = _windowed_sinc_resampled_chunk(
                                    source,
                                    source_positions[run],
                                    cutoff=min(1.0, 1.0 / model.source_scale(fs)),
                                )
                    else:
                        for run in runs:
                            data[run] = _linear_resampled_chunk(source, source_positions[run])
                            nearest = np.clip(
                                np.rint(source_positions[run]).astype(np.int64),
                                0,
                                source.shape[0] - 1,
                            )
                            data[run, 0] = np.asarray(source[nearest, 0], dtype=np.int16)
                    combined[:, channel_start : channel_start + n_channels] = data
                    if validity is not None:
                        validity[:, device_index] = valid
                    channel_start += n_channels
                combined.astype("<i2", copy=False).tofile(output)
                if validity is not None:
                    assert validity_output is not None
                    validity[:, validity_order].tofile(validity_output)
                if progress is not None:
                    progress(f"write_{stream}", 100.0 * (start + count) / output_count)
        replace_atomic(partial, output_path)
        if validity_partial is not None and validity_path is not None:
            replace_atomic(validity_partial, validity_path)
    finally:
        for source in mapped:
            close_memmap(source)
        if partial.exists():
            partial.unlink()
        if validity_partial is not None and validity_partial.exists():
            validity_partial.unlink()
    return output_count, total_channels


def _write_time_dat(path: Path, n_samples: int, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; enable overwrite to regenerate: {path}")
    partial = atomic_output_path(path)
    with partial.open("wb") as stream:
        for start in range(0, n_samples, 1_000_000):
            end = min(n_samples, start + 1_000_000)
            np.arange(start, end, dtype="<i4").tofile(stream)
    replace_atomic(partial, path)


def _validity_summary(path: Path, n_samples: int, device_count: int) -> dict[str, object]:
    mapped = np.memmap(path, dtype=np.uint8, mode="r", shape=(n_samples, device_count))
    counts = np.zeros(device_count, dtype=np.int64)
    common_count = 0
    try:
        for start in range(0, n_samples, 1_000_000):
            block = np.asarray(mapped[start : min(n_samples, start + 1_000_000)]) != 0
            counts += np.count_nonzero(block, axis=0)
            common_count += int(np.count_nonzero(np.all(block, axis=1)))
    finally:
        close_memmap(mapped)
    return {
        "valid_samples_by_channel": counts.tolist(),
        "valid_fraction_by_channel": (counts / max(1, n_samples)).tolist(),
        "common_valid_samples": common_count,
        "common_valid_fraction": common_count / max(1, n_samples),
    }


def rewrite_staged_ephys_from_segments(
    amplifier_path: Path,
    validity_path: Path,
    recordings: list[Recording],
    master_index: int,
    pair_models: dict[int, SyncModel],
    *,
    common_start: int,
    common_end: int,
    chunk_seconds: float,
    device_gaps: list[DeviceGap] | None = None,
    classified_intervals: list[ClassifiedInterval] | None = None,
    device_source_steps: list[DeviceSourceStep] | None = None,
    device_terminal_support: list[DeviceTerminalSupport] | None = None,
    device_sync_segments: list[DeviceSyncSegment] | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Render a corrected private-stage ephys generation once from raw input."""

    models = _device_models(recordings, master_index, pair_models)
    n_samples, _ = _write_interleaved_stream(
        Path(amplifier_path),
        recordings,
        models,
        master_index,
        common_start,
        common_end,
        stream="ephys",
        chunk_seconds=chunk_seconds,
        overwrite=True,
        progress=progress,
        device_gaps=device_gaps,
        validity_path=Path(validity_path),
        classified_intervals=classified_intervals,
        device_source_steps=device_source_steps,
        device_terminal_support=device_terminal_support,
        device_sync_segments=device_sync_segments,
    )
    return _validity_summary(Path(validity_path), n_samples, len(recordings))


def write_alignment_quality(
    validity_path: Path,
    output_path: Path,
    *,
    n_samples: int,
    device_count: int,
    master_index: int,
    canonical_start_sample: int,
    warning_intervals: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Write a strict alignment mask without deleting measured ephys values."""

    validity_path = Path(validity_path)
    output_path = Path(output_path)
    expected = n_samples * device_count
    if not 0 <= master_index < device_count:
        raise ValueError("master index is outside the device range")
    if not validity_path.is_file() or validity_path.stat().st_size != expected:
        raise ValueError("validity input is missing or has an unexpected size")
    validity_device_order = [master_index] + [
        index for index in range(device_count) if index != master_index
    ]
    validity_column_by_device = {
        device_index + 1: column
        for column, device_index in enumerate(validity_device_order)
    }
    partial = atomic_output_path(output_path)
    partial.unlink(missing_ok=True)
    intervals_by_device: dict[int, list[tuple[int, int]]] = {
        index: [] for index in range(1, device_count + 1)
    }
    for interval in warning_intervals:
        start = max(
            0,
            int(interval["canonical_start_sample"]) - canonical_start_sample,
        )
        end = min(
            n_samples,
            int(interval["canonical_end_sample"]) - canonical_start_sample,
        )
        if end <= start:
            continue
        for device_index in interval["affected_device_indices"]:
            device = int(device_index)
            if device not in intervals_by_device:
                raise ValueError("alignment warning uses an unknown device index")
            intervals_by_device[device].append((start, end))
    source = np.memmap(
        validity_path,
        dtype=np.uint8,
        mode="r",
        shape=(n_samples, device_count),
    )
    try:
        with partial.open("wb") as stream:
            for chunk_start in range(0, n_samples, 1_000_000):
                chunk_end = min(n_samples, chunk_start + 1_000_000)
                block = np.asarray(source[chunk_start:chunk_end], dtype=np.uint8).copy()
                for device, intervals in intervals_by_device.items():
                    for start, end in intervals:
                        overlap_start = max(chunk_start, start)
                        overlap_end = min(chunk_end, end)
                        if overlap_end > overlap_start:
                            block[
                                overlap_start - chunk_start : overlap_end - chunk_start,
                                validity_column_by_device[device],
                            ] = 0
                block.tofile(stream)
        replace_atomic(partial, output_path)
    finally:
        close_memmap(source)
        partial.unlink(missing_ok=True)
    return _validity_summary(output_path, n_samples, device_count)


def apply_staged_zero_fill(
    amplifier_path: Path,
    validity_path: Path,
    recordings: list[Recording],
    master_index: int,
    *,
    canonical_start_sample: int,
    n_output_samples: int,
    intervals: list[ClassifiedInterval],
) -> dict[str, object]:
    """Apply QC-derived exclusions to staged neural data and validity in place."""

    total_channels = sum(recording.n_channels for recording in recordings)
    expected_amplifier_bytes = n_output_samples * total_channels * np.dtype("<i2").itemsize
    expected_validity_bytes = n_output_samples * len(recordings)
    amplifier_path = Path(amplifier_path)
    validity_path = Path(validity_path)
    if amplifier_path.stat().st_size != expected_amplifier_bytes:
        raise ValueError("staged amplifier size changed before post-merge exclusion")
    if validity_path.stat().st_size != expected_validity_bytes:
        raise ValueError("staged validity size changed before post-merge exclusion")

    channel_bounds: list[tuple[int, int]] = []
    channel_start = 0
    for recording in recordings:
        channel_end = channel_start + recording.n_channels
        channel_bounds.append((channel_start, channel_end))
        channel_start = channel_end
    validity_channels = {
        device_index: validity_channel
        for validity_channel, device_index in enumerate(
            _validity_device_order(recordings, master_index)
        )
    }
    amplifier = np.memmap(
        amplifier_path,
        dtype="<i2",
        mode="r+",
        shape=(n_output_samples, total_channels),
        order="C",
    )
    validity = np.memmap(
        validity_path,
        dtype=np.uint8,
        mode="r+",
        shape=(n_output_samples, len(recordings)),
        order="C",
    )
    try:
        for interval in intervals:
            if interval.action != "zero_fill":
                continue
            start = max(0, interval.canonical_start_sample - canonical_start_sample)
            end = min(
                n_output_samples,
                interval.canonical_end_sample - canonical_start_sample,
            )
            if end <= start:
                continue
            for device_index in interval.affected_device_indices:
                source_index = device_index - 1
                if source_index < 0 or source_index >= len(recordings):
                    raise ValueError(f"invalid post-merge exclusion device index {device_index}")
                first_channel, last_channel = channel_bounds[source_index]
                amplifier[start:end, first_channel:last_channel] = 0
                validity[start:end, validity_channels[source_index]] = 0
        amplifier.flush()
        validity.flush()
    finally:
        close_memmap(amplifier)
        close_memmap(validity)
    return _validity_summary(validity_path, n_output_samples, len(recordings))


def _channel_layout_records(
    recordings: list[Recording], master_index: int
) -> list[dict[str, object]]:
    validity_channels = {
        device_index: validity_channel
        for validity_channel, device_index in enumerate(
            _validity_device_order(recordings, master_index)
        )
    }
    rows: list[dict[str, object]] = []
    merged_channel = 1
    for device_index, recording in enumerate(recordings, start=1):
        for channel in range(1, recording.n_channels + 1):
            rows.append(
                {
                    "merged_channel": merged_channel,
                    "device_index": device_index,
                    "device_name": recording.device_name,
                    "recording_name": recording.recording_name,
                    "device_channel": channel,
                    "device_folder": str(recording.folder),
                    "validity_channel": validity_channels[device_index - 1],
                }
            )
            merged_channel += 1
    return rows


def _collect_events(
    analog_path: Path,
    output_folder: Path,
    recordings: list[Recording],
    analog_samples: int,
    total_analog_channels: int,
    overwrite: bool,
    published_output_folder: Path | None = None,
    device_gaps: list[DeviceGap] | None = None,
    common_start: int = 0,
    fs: float = 20_000.0,
    classified_intervals: list[ClassifiedInterval] | None = None,
    device_terminal_support: list[DeviceTerminalSupport] | None = None,
    write_event_files: bool = False,
    analog_validity_path: Path | None = None,
    master_index: int = 0,
) -> list[dict[str, object]]:
    mapped = interleaved_memmap(analog_path, total_analog_channels, analog_samples)
    published_output_folder = output_folder if published_output_folder is None else published_output_folder
    rows: list[dict[str, object]] = []
    gaps_by_device = _device_gap_map(list(device_gaps or ()))
    intervals_by_device = _device_interval_map(list(classified_intervals or ()))
    terminal_by_device = _device_terminal_map(list(device_terminal_support or ()))
    block_start = 0
    analog_validity = (
        np.memmap(
            analog_validity_path,
            dtype=np.uint8,
            mode="r",
            shape=(analog_samples, len(recordings)),
        )
        if analog_validity_path is not None and analog_validity_path.exists()
        else None
    )
    validity_channel_by_device = {
        device: channel
        for channel, device in enumerate(_validity_device_order(recordings, master_index))
    }
    try:
        for device_index, recording in enumerate(recordings, start=1):
            digital = np.asarray(mapped[:, block_start], dtype=np.uint16)
            device_valid = (
                np.asarray(
                    analog_validity[:, validity_channel_by_device[device_index - 1]] != 0
                )
                if analog_validity is not None
                else np.ones(analog_samples, dtype=bool)
            )
            gap_edges: list[int] = []
            for gap in gaps_by_device.get(device_index - 1, ()):
                gap_edges.extend(
                    [
                        int(np.floor((gap.canonical_start_sample - common_start) * ANALOG_FS / fs)),
                        int(np.ceil((gap.canonical_end_sample - common_start) * ANALOG_FS / fs)),
                    ]
                )
            for interval in intervals_by_device.get(device_index - 1, ()):
                gap_edges.extend(
                    [
                        int(np.floor((interval.canonical_start_sample - common_start) * ANALOG_FS / fs)),
                        int(np.ceil((interval.canonical_end_sample - common_start) * ANALOG_FS / fs)),
                    ]
                )
            terminal = terminal_by_device.get(device_index - 1)
            if terminal is not None:
                gap_edges.append(
                    int(
                        np.floor(
                            (terminal.supported_canonical_end_sample - common_start)
                            * ANALOG_FS
                            / fs
                        )
                    )
                )
            for bit_index in range(16):
                state = ((digital >> bit_index) & 1).astype(np.int8)
                valid_starts = np.flatnonzero(device_valid & np.r_[True, ~device_valid[:-1]])
                valid_ends = np.flatnonzero(device_valid & np.r_[~device_valid[1:], True]) + 1
                candidate_pairs: list[tuple[int, int]] = []
                for valid_start, valid_end in zip(valid_starts, valid_ends):
                    # Reset the digital state at every invalid boundary.  Edge
                    # states themselves are not emitted, preventing zero-fill
                    # from creating artificial events.
                    run = state[valid_start:valid_end]
                    if run.size < 2:
                        continue
                    transitions = np.diff(run)
                    starts = valid_start + np.flatnonzero(transitions == 1) + 1
                    ends = valid_start + np.flatnonzero(transitions == -1) + 1
                    end_index = 0
                    for start in starts:
                        while end_index < ends.size and ends[end_index] <= start:
                            end_index += 1
                        if end_index >= ends.size:
                            break
                        candidate_pairs.append((int(start), int(ends[end_index])))
                        end_index += 1
                paired = [
                    (start, end)
                    for start, end in candidate_pairs
                    if not any(abs(start - edge) <= 2 or abs(end - edge) <= 2 for edge in gap_edges)
                ]
                gap_affected_count = len(candidate_pairs) - len(paired)
                starts = np.asarray([item[0] for item in paired], dtype=np.int64)
                ends = np.asarray([item[1] for item in paired], dtype=np.int64)
                count = len(paired)
                event_path = output_folder / f"device_event.dev{device_index:02d}.d{bit_index + 1:02d}.evt"
                if count and write_event_files:
                    if event_path.exists() and not overwrite:
                        raise FileExistsError(f"Output exists; enable overwrite to regenerate: {event_path}")
                    with event_path.open("w", encoding="utf-8") as stream:
                        for start, end in zip(starts[:count], ends[:count]):
                            stream.write(f"{start / ANALOG_FS * 1000:.6f}\t{recording.device_name} DigitIn start {bit_index + 1}\n")
                            stream.write(f"{end / ANALOG_FS * 1000:.6f}\t{recording.device_name} DigitIn end {bit_index + 1}\n")
                    event_text = str(published_output_folder / event_path.name)
                else:
                    if event_path.exists() and overwrite:
                        event_path.unlink()
                    event_text = ""
                rows.append(
                    {
                        "device_index": device_index,
                        "device_name": recording.device_name,
                        "recording_name": recording.recording_name,
                        "digital_bit": bit_index + 1,
                        "merged_analog_channel": block_start + 1,
                        "event_count": count,
                        "gap_affected_event_count_excluded": gap_affected_count,
                        "event_file": event_text,
                    }
                )
            block_start += recording.analog_channels
    finally:
        close_memmap(mapped)
        if analog_validity is not None:
            close_memmap(analog_validity)
    return rows


def _recover_interrupted_transactions(output_folder: Path) -> None:
    for staging in output_folder.glob(".wild_merge_stage_*"):
        if staging.is_dir():
            shutil.rmtree(staging, ignore_errors=True)
    for backup in output_folder.glob(".wild_merge_backup_*"):
        if not backup.is_dir():
            continue
        manifest_path = backup / "transaction.json"
        if (backup / "COMMITTED").exists() or not manifest_path.exists():
            shutil.rmtree(backup, ignore_errors=True)
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_names = set(manifest["old_names"])
        new_names = set(manifest["new_names"])
        # New-only destinations can only have been promoted by this transaction.
        for name in new_names - old_names:
            destination = output_folder / name
            if destination.exists():
                destination.unlink()
        # For replaced outputs, a backup proves that the old destination was
        # moved. Without that proof, preserve the destination: force-kill may
        # have occurred part-way through the backup loop.
        for name in old_names:
            backup_path = backup / name
            if backup_path.exists():
                destination = output_folder / name
                if destination.exists():
                    destination.unlink()
                os.replace(backup_path, destination)
        shutil.rmtree(backup, ignore_errors=True)


def _validate_core_staged_sizes(
    staging: Path,
    staged_outputs: dict[str, str],
    recordings: list[Recording],
) -> None:
    """Reject a staged core stream whose byte length contradicts merge metadata."""

    internal_path = Path(staged_outputs["_merge_internal"])
    merge_info = json.loads(internal_path.read_text(encoding="utf-8"))
    n_samples = int(merge_info["n_samples"])
    analog_samples = int(merge_info["analog_samples"])
    expected = {
        "amplifier.dat": n_samples * int(merge_info["n_channels"]) * 2,
        "analogin.dat": analog_samples * int(merge_info["analog_channels"]) * 2,
        "time.dat": n_samples * 4,
        "valid_samples.dat": n_samples * len(recordings),
    }
    if (staging / "valid_analog_samples.dat").is_file():
        expected["valid_analog_samples.dat"] = analog_samples * len(recordings)
    mismatches = {
        name: (size, (staging / name).stat().st_size if (staging / name).is_file() else None)
        for name, size in expected.items()
        if not (staging / name).is_file() or (staging / name).stat().st_size != size
    }
    if mismatches:
        details = ", ".join(
            f"{name}: expected {size}, actual {actual}"
            for name, (size, actual) in sorted(mismatches.items())
        )
        raise RuntimeError(f"Staged merge byte-length validation failed: {details}")


def _merge_recordings_into_folder(
    recordings: list[Recording],
    master_index: int,
    pair_models: dict[int, SyncModel],
    output_folder: Path,
    *,
    chunk_seconds: float,
    overwrite: bool,
    progress: ProgressCallback | None = None,
    published_output_folder: Path | None = None,
    run_id: str = "",
    device_gaps: list[DeviceGap] | None = None,
    minimum_common_start: int = 0,
    maximum_common_end: int | None = None,
    maximum_common_end_device_index: int | None = None,
    maximum_common_end_reason: str = "",
    classified_intervals: list[ClassifiedInterval] | None = None,
    device_source_steps: list[DeviceSourceStep] | None = None,
    device_terminal_support: list[DeviceTerminalSupport] | None = None,
    preserve_device_tails: bool = False,
    write_event_files: bool = False,
    device_sync_segments: list[DeviceSyncSegment] | None = None,
    analog_integrity_results: list[AnalogIntegrityResult] | None = None,
    analog_clock_priors: list[DeviceClockPrior] | None = None,
    timing: TimingCallback | None = None,
) -> dict[str, str]:
    output_folder.mkdir(parents=True, exist_ok=True)
    models = _device_models(recordings, master_index, pair_models)
    device_gaps = list(device_gaps or ())
    classified_intervals = list(classified_intervals or ())
    device_source_steps = list(device_source_steps or ())
    device_terminal_support = list(device_terminal_support or ())
    segments_by_device = (
        _device_segment_map(device_sync_segments, device_count=len(recordings))
        if device_sync_segments is not None
        else None
    )
    common_start, common_end, common_interval_limits = _common_master_interval(
        recordings,
        models,
        master_index,
        device_gaps,
        minimum_common_start,
        maximum_common_end,
        maximum_common_end_device_index,
        maximum_common_end_reason,
        preserve_device_tails,
        device_sync_segments is not None,
    )
    def output_gap_action(gap: DeviceGap) -> str:
        if gap.canonical_end_sample + EPHYS_SINC_HALF_WIDTH <= common_start:
            return "cropped_before_output"
        if gap.canonical_start_sample - EPHYS_SINC_HALF_WIDTH > common_end:
            return "outside_output"
        return "zero_filled_with_guard"

    validate_time_dat_length(common_end - common_start + 1)
    amplifier_path = output_folder / "amplifier.dat"
    analog_path = output_folder / "analogin.dat"
    analog_validity_path = output_folder / "valid_analog_samples.dat"
    time_path = output_folder / "time.dat"
    validity_path = output_folder / "valid_samples.dat"
    internal_merge_path = output_folder / ".wild_internal_merge.json"
    published_output_folder = output_folder if published_output_folder is None else published_output_folder
    if timing is not None:
        timing("ephys_merge", "start", 0, 0)
    if progress is not None:
        progress("write_ephys", 0.0)
    try:
        ephys_samples, total_channels = _write_interleaved_stream(
            amplifier_path,
            recordings,
            models,
            master_index,
            common_start,
            common_end,
            stream="ephys",
            chunk_seconds=chunk_seconds,
            overwrite=overwrite,
            progress=progress,
            device_gaps=device_gaps,
            validity_path=validity_path,
            classified_intervals=classified_intervals,
            device_source_steps=device_source_steps,
            device_terminal_support=device_terminal_support,
            device_sync_segments=device_sync_segments,
        )
    except BaseException:
        if timing is not None:
            timing("ephys_merge", "failed", 0, 0)
        raise
    else:
        if timing is not None:
            timing(
                "ephys_merge",
                "complete",
                0,
                amplifier_path.stat().st_size + validity_path.stat().st_size,
            )
    analog_timelines: list[AnalogTimelineResult] = []
    if timing is not None:
        timing("analog_merge", "start", 0, 0)
    if progress is not None:
        progress("write_analog", 0.0)
    try:
        if analog_integrity_results is not None or analog_clock_priors is not None:
            if analog_integrity_results is None or analog_clock_priors is None:
                raise ValueError("analog integrity results and clock priors must be supplied together")
            if len(analog_integrity_results) != len(recordings) or len(analog_clock_priors) != len(recordings):
                raise ValueError("analog integrity/prior collections must align with recordings")
            master_fs = float(recordings[master_index].fs)
            analog_samples = int(
                np.floor((common_end - common_start) / (master_fs / ANALOG_FS))
            ) + 1
            analog_segments_by_device: dict[int, tuple] = {}
            for device_index, (recording, integrity, prior) in enumerate(
                zip(recordings, analog_integrity_results, analog_clock_priors), start=1
            ):
                if integrity.device_index != device_index or prior.device_index != device_index:
                    raise ValueError("analog evidence device order does not match recordings")
                prior = replace(prior, canonical_ephys_start_sample=float(common_start))
                temporal_decisions = tuple(
                    event
                    for event in integrity.timeline_events
                    if event.kind != "counter_corruption"
                )
                segments = build_event_driven_analog_segments(
                    prior,
                    canonical_start_row=0,
                    canonical_end_row=analog_samples,
                    raw_row_count=recording.analog_samples,
                    decisions=temporal_decisions,
                )
                analog_segments_by_device[device_index] = segments
                integrity_warning_count = len(integrity.events)
                has_publishable_support = any(
                    segment.publishable for segment in segments
                )
                publishable_coverage = sum(
                    segment.canonical_end_row - segment.canonical_start_row
                    for segment in segments
                    if segment.publishable
                )
                has_complete_publishable_coverage = (
                    has_publishable_support
                    and publishable_coverage == analog_samples
                )
                analog_timelines.append(
                    AnalogTimelineResult(
                        device_index=device_index,
                        segments=segments,
                        integrity_events=integrity.events,
                        status=(
                            "OK"
                            if integrity_warning_count == 0
                            and has_complete_publishable_coverage
                            else "WARN"
                        ),
                        warnings=(
                            ()
                            if integrity_warning_count == 0
                            and has_complete_publishable_coverage
                            else (
                                (
                                    f"{integrity_warning_count} analog integrity event(s)"
                                    if integrity_warning_count
                                    else (
                                        "no publishable analog source support"
                                        if not has_publishable_support
                                        else (
                                            "publishable analog source support covers "
                                            f"{publishable_coverage}/{analog_samples} canonical rows"
                                        )
                                    )
                                ),
                            )
                        ),
                        phase_ephys_samples=prior.phase_ephys_samples,
                        source_raw_row_count=recording.analog_samples,
                        clock_prior=prior,
                    )
                )
            master_segments = analog_segments_by_device[master_index + 1]
            if not any(segment.publishable for segment in master_segments):
                raise ValueError(
                    "canonical master analog mapping has no publishable source support"
                )
            analog_result = write_canonical_analog(
                recordings,
                analog_segments_by_device,
                master_index=master_index,
                canonical_rows=analog_samples,
                analog_path=analog_path,
                validity_path=analog_validity_path,
                chunk_rows=max(1, round(chunk_seconds * ANALOG_FS)),
                overwrite=overwrite,
                progress=progress,
                invalid_lane_intervals_by_device={
                    device_index: {
                        lane: tuple(
                            (event.raw_start_row, event.raw_end_row)
                            for event in integrity.events
                            if event.kind
                            in ({"counter_corruption"} | IMU_MODALITY_INVALID_KINDS)
                            and lane in event.affected_lanes
                        )
                        for lane in range(recording.analog_channels)
                        if any(
                            event.kind
                            in ({"counter_corruption"} | IMU_MODALITY_INVALID_KINDS)
                            and lane in event.affected_lanes
                            for event in integrity.events
                        )
                    }
                    for device_index, (recording, integrity) in enumerate(
                        zip(recordings, analog_integrity_results), start=1
                    )
                },
                staged=True,
            )
            total_analog_channels = analog_result.channels_per_device * len(recordings)
        else:
            analog_samples, total_analog_channels = _write_interleaved_stream(
                analog_path,
                recordings,
                models,
                master_index,
                common_start,
                common_end,
                stream="analog",
                chunk_seconds=chunk_seconds,
                overwrite=overwrite,
                progress=progress,
                device_gaps=device_gaps,
                classified_intervals=classified_intervals,
                device_source_steps=device_source_steps,
                device_terminal_support=device_terminal_support,
                device_sync_segments=device_sync_segments,
            )
    except BaseException:
        if timing is not None:
            timing("analog_merge", "failed", 0, 0)
        raise
    else:
        if timing is not None:
            timing(
                "analog_merge",
                "complete",
                0,
                analog_path.stat().st_size
                + (analog_validity_path.stat().st_size if analog_validity_path.exists() else 0),
            )
    validity_summary = _validity_summary(validity_path, ephys_samples, len(recordings))
    _write_time_dat(time_path, ephys_samples, overwrite)
    channel_layout = _channel_layout_records(recordings, master_index)
    event_summary = _collect_events(
        analog_path,
        output_folder,
        recordings,
        analog_samples,
        total_analog_channels,
        overwrite,
        published_output_folder=published_output_folder,
        device_gaps=([] if analog_timelines else device_gaps),
        common_start=common_start,
        fs=recordings[master_index].fs,
        classified_intervals=([] if analog_timelines else classified_intervals),
        device_terminal_support=([] if analog_timelines else device_terminal_support),
        write_event_files=write_event_files,
        analog_validity_path=(analog_validity_path if analog_validity_path.exists() else None),
        master_index=master_index,
    )

    def mapped_ephys_sample(
        device_index: int, model: SyncModel, canonical_sample: int
    ) -> float | None:
        if segments_by_device is not None:
            source, valid = map_canonical_positions(
                segments_by_device[device_index - 1],
                np.asarray([canonical_sample], dtype=np.float64),
                source_sample_count=recordings[device_index - 1].n_samples,
                interpolation_half_width=0,
                device_index=device_index,
            )
            return float(source[0]) if bool(valid[0]) else None
        return (
            model.source_scale(recordings[master_index].fs) * canonical_sample
            + model.intercept_samples
            - sum(
                gap.missing_samples
                for gap in device_gaps
                if gap.device_index == device_index
                and gap.canonical_end_sample <= canonical_sample
            )
            + (
                0.0
                if device_sync_segments is not None
                else sum(
                    step.source_step_samples
                    for step in device_source_steps
                    if step.device_index == device_index
                    and step.canonical_sample <= canonical_sample
                )
            )
        )

    merge_info = {
        "backend": SYNC_ALGORITHM_VERSION,
        "run_id": run_id,
        "complete": True,
        "master_index": master_index + 1,
        "fs": recordings[master_index].fs,
        "common_start_master_sample": common_start,
        "common_end_master_sample": common_end,
        "n_samples": ephys_samples,
        "n_channels": total_channels,
        "validity_samples_file": validity_path.name,
        "validity_samples_dtype": "uint8",
        "validity_samples_shape": [ephys_samples, len(recordings)],
        "validity_samples_layout": "sample_major_interleaved",
        "validity_summary": validity_summary,
        "validity_samples_channels": [
            {
                "channel": validity_channel,
                "role": "master" if device_index == master_index else f"slave {validity_channel}",
                "device_index": device_index + 1,
                "device_name": recordings[device_index].device_name,
                "device_folder": str(recordings[device_index].folder),
                "amplifier_channel_start": sum(
                    recording.n_channels for recording in recordings[:device_index]
                ),
                "amplifier_channel_count": recordings[device_index].n_channels,
            }
            for validity_channel, device_index in enumerate(
                _validity_device_order(recordings, master_index)
            )
        ],
        "analog_samples": analog_samples,
        "analog_channels": total_analog_channels,
        "analog_status": (
            "NOT_RUN"
            if not analog_timelines
            else ("WARN" if any(item.status == "WARN" for item in analog_timelines) else "OK")
        ),
        "analog_timelines": [item.to_dict() for item in analog_timelines],
        "analog_validity_samples_file": (
            analog_validity_path.name if analog_validity_path.exists() else None
        ),
        "analog_validity_samples_dtype": (
            "uint8" if analog_validity_path.exists() else None
        ),
        "analog_validity_samples_shape": (
            [analog_samples, len(recordings)] if analog_validity_path.exists() else None
        ),
        "analog_channel_device_order": [
            {
                "block": block,
                "device_index": device_index + 1,
                "device_name": recordings[device_index].device_name,
                "channel_start_zero_based": sum(
                    item.analog_channels for item in recordings[:device_index]
                ),
                "channel_count": recordings[device_index].analog_channels,
            }
            for block, device_index in enumerate(range(len(recordings)))
        ],
        "analog_validity_device_order": [
            {
                "channel": validity_channel,
                "device_index": device_index + 1,
                "device_name": recordings[device_index].device_name,
            }
            for validity_channel, device_index in enumerate(
                _validity_device_order(recordings, master_index)
            )
        ],
        "event_files_written": write_event_files,
        "digital_events": event_summary,
        "channel_layout": channel_layout,
        "output_files": {
            "amplifier": amplifier_path.name,
            "analog": analog_path.name,
            "time": time_path.name,
            "validity": validity_path.name,
            **(
                {"analog_validity": analog_validity_path.name}
                if analog_validity_path.exists()
                else {}
            ),
        },
        "common_interval_limits": common_interval_limits,
        "coordinate_system": "canonical_gap_aware_ephys_samples",
        "device_gaps": [
            {
                "device_index": gap.device_index,
                "canonical_start_sample": gap.canonical_start_sample,
                "canonical_end_sample": gap.canonical_end_sample,
                "missing_samples": gap.missing_samples,
                "duration_ms": gap.duration_ms,
                "confidence": gap.confidence,
                "action": output_gap_action(gap),
                "evidence": gap.evidence,
                "intersects_output": output_gap_action(gap) == "zero_filled_with_guard",
                "interpolation_guard_samples": EPHYS_SINC_HALF_WIDTH,
                "zero_fill_start_sample": gap.canonical_start_sample - EPHYS_SINC_HALF_WIDTH,
                "zero_fill_end_sample": gap.canonical_end_sample + EPHYS_SINC_HALF_WIDTH,
            }
            for gap in device_gaps
        ],
        "classified_intervals": [interval.to_dict() for interval in classified_intervals],
        "device_source_steps": [step.to_dict() for step in device_source_steps],
        "device_terminal_support": [support.to_dict() for support in device_terminal_support],
        "device_sync_segments": (
            [segment.to_dict() for segment in device_sync_segments]
            if device_sync_segments is not None
            else []
        ),
        "segment_mapping_authoritative": device_sync_segments is not None,
        "imu_status": "NOT_RUN",
        "devices": [
            {
                "folder": str(recording.folder),
                "device_name": recording.device_name,
                "recording_name": recording.recording_name,
                "scale": model.source_scale(recordings[master_index].fs),
                "intercept_samples": model.intercept_samples,
                "drift_ppm": model.drift_ppm,
                "mapped_ephys_start_sample": (
                    mapped_ephys_sample(device_index, model, common_start)
                ),
                "mapped_ephys_end_sample": (
                    mapped_ephys_sample(device_index, model, common_end)
                ),
                "mapped_analog_start_sample": (
                    model.source_scale(recordings[master_index].fs) * common_start
                    + model.intercept_samples
                )
                / (recordings[master_index].fs / ANALOG_FS),
                "mapped_analog_end_sample": (
                    model.source_scale(recordings[master_index].fs) * common_end
                    + model.intercept_samples
                )
                / (recordings[master_index].fs / ANALOG_FS),
            }
            for device_index, (recording, model) in enumerate(zip(recordings, models), start=1)
        ],
    }
    internal_merge_path.write_text(json.dumps(merge_info, indent=2), encoding="utf-8")
    return {
        "amplifier": str(amplifier_path),
        "analog": str(analog_path),
        "time": str(time_path),
        "validity": str(validity_path),
        **(
            {"analog_validity": str(analog_validity_path)}
            if analog_validity_path.exists()
            else {}
        ),
        "_merge_internal": str(internal_merge_path),
    }


def merge_recordings(
    recordings: list[Recording],
    master_index: int,
    pair_models: dict[int, SyncModel],
    output_folder: Path,
    *,
    chunk_seconds: float,
    overwrite: bool,
    progress: ProgressCallback | None = None,
    run_id: str = "",
    stage_callback: StageCallback | None = None,
    additional_managed_names: set[str] | None = None,
    device_gaps: list[DeviceGap] | None = None,
    minimum_common_start: int = 0,
    maximum_common_end: int | None = None,
    maximum_common_end_device_index: int | None = None,
    maximum_common_end_reason: str = "",
    classified_intervals: list[ClassifiedInterval] | None = None,
    device_source_steps: list[DeviceSourceStep] | None = None,
    device_terminal_support: list[DeviceTerminalSupport] | None = None,
    preserve_device_tails: bool = False,
    write_event_files: bool = False,
    device_sync_segments: list[DeviceSyncSegment] | None = None,
    analog_integrity_results: list[AnalogIntegrityResult] | None = None,
    analog_clock_priors: list[DeviceClockPrior] | None = None,
    timing: TimingCallback | None = None,
) -> dict[str, str]:
    """Stage a complete merged dataset before replacing session-level files.

    ``stage_callback`` runs after the core merge has been written to the
    private staging directory and before any canonical session output is
    replaced.  It may add reports or additional DAT files to the staging
    directory and raise to reject the staged merge.  This lets the pipeline
    gate publication on independent post-merge checks without exposing a
    partially updated session.
    """

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    _recover_interrupted_transactions(output_folder)
    fixed_names = {
        "amplifier.dat",
        "analogin.dat",
        "time.dat",
        "valid_samples.dat",
    }
    if analog_integrity_results is not None or analog_clock_priors is not None:
        fixed_names.add("valid_analog_samples.dat")
    legacy_python_metadata_names = {
        "wild_preprocess_channel_layout.tsv",
        "wild_multilogger_events.tsv",
        "wild_multilogger_mergeInfo.mat",
        "wild_multilogger_mergeInfo.json",
        "wild_multilogger_sync_qc.tsv",
        "wild_multilogger_sync_qc.json",
        "wild_multilogger_sync_qc.mat",
        "wild_multilogger_postmerge_qc.json",
        "pc_time_qc.json",
    }
    legacy_pair_figures = {
        path.name
        for path in output_folder.glob("wild_multilogger_sync_master_vs_*_qc.png")
        if path.is_file()
    }
    event_names = {
        f"device_event.dev{device_index:02d}.d{bit_index:02d}.evt"
        for device_index in range(1, len(recordings) + 1)
        for bit_index in range(1, 17)
    }
    # A later run may use fewer devices.  The previous manifest is therefore
    # part of this transaction, not merely an informational record: names no
    # longer staged (for example device 03 events/figures) are backed up and
    # intentionally omitted from the promoted generation.
    managed_names = (
        fixed_names
        | event_names
        | legacy_python_metadata_names
        | legacy_pair_figures
        | set(additional_managed_names or ())
        | _previous_managed_names(output_folder)
    )
    existing = [output_folder / name for name in managed_names if (output_folder / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output exists; enable overwrite to regenerate: " + ", ".join(str(path) for path in existing)
        )

    staging = Path(tempfile.mkdtemp(prefix=".wild_merge_stage_", dir=output_folder))
    backup = Path(tempfile.mkdtemp(prefix=".wild_merge_backup_", dir=output_folder))
    promoted: list[Path] = []
    backed_up: list[tuple[Path, Path]] = []
    transaction_committed = False
    rollback_complete = False
    try:
        staged_outputs = _merge_recordings_into_folder(
            recordings,
            master_index,
            pair_models,
            staging,
            chunk_seconds=chunk_seconds,
            overwrite=False,
            progress=progress,
            published_output_folder=output_folder,
            run_id=run_id,
            device_gaps=device_gaps,
            minimum_common_start=minimum_common_start,
            maximum_common_end=maximum_common_end,
            maximum_common_end_device_index=maximum_common_end_device_index,
            maximum_common_end_reason=maximum_common_end_reason,
            classified_intervals=classified_intervals,
            device_source_steps=device_source_steps,
            device_terminal_support=device_terminal_support,
            preserve_device_tails=preserve_device_tails,
            write_event_files=write_event_files,
            device_sync_segments=device_sync_segments,
            analog_integrity_results=analog_integrity_results,
            analog_clock_priors=analog_clock_priors,
            timing=timing,
        )
        if stage_callback is not None:
            stage_callback(staging, staged_outputs)
        _validate_core_staged_sizes(staging, staged_outputs, recordings)
        internal_names = {
            Path(value).name for key, value in staged_outputs.items() if key.startswith("_")
        }
        for name in internal_names:
            (staging / name).unlink(missing_ok=True)
        staged_files = {path.name: path for path in staging.iterdir() if path.is_file()}
        missing = fixed_names - set(staged_files)
        if missing:
            raise RuntimeError(f"Staged merge is incomplete: {sorted(missing)}")
        (backup / "transaction.json").write_text(
            json.dumps(
                {
                    "old_names": sorted(path.name for path in existing),
                    "new_names": sorted(staged_files),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if progress is not None:
            progress("publish", 0.0)
        for name in managed_names:
            destination = output_folder / name
            if destination.exists():
                backup_path = backup / name
                os.replace(destination, backup_path)
                backed_up.append((destination, backup_path))
        promotion_names = sorted(
            name for name in staged_files if name != "wild_preprocess_run.json"
        )
        if "wild_preprocess_run.json" in staged_files:
            promotion_names.append("wild_preprocess_run.json")
        for name in promotion_names:
            staged_path = staged_files[name]
            destination = output_folder / name
            os.replace(staged_path, destination)
            promoted.append(destination)
        (backup / "COMMITTED").write_text("complete\n", encoding="utf-8")
        transaction_committed = True
        return {
            key: str(output_folder / Path(value).name)
            for key, value in staged_outputs.items()
            if not key.startswith("_")
        }
    except BaseException:
        try:
            for destination in promoted:
                if destination.exists():
                    destination.unlink()
            for destination, backup_path in reversed(backed_up):
                if backup_path.exists():
                    os.replace(backup_path, destination)
        except BaseException as rollback_error:
            raise RuntimeError(
                "Merge publication rollback was incomplete; recovery backup was preserved at "
                f"{backup}"
            ) from rollback_error
        rollback_complete = True
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if transaction_committed or rollback_complete:
            shutil.rmtree(backup, ignore_errors=True)
