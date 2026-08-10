from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.io import loadmat, savemat

from ..binary_io import atomic_output_path, close_memmap, interleaved_memmap, replace_atomic
from ..models import DeviceGap, Recording, SyncModel
from ..version import SYNC_ALGORITHM_VERSION


ProgressCallback = Callable[[str, float], None]
StageCallback = Callable[[Path, dict[str, str]], None]
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


def _mat_safe(value: object) -> object:
    if value is None:
        return float("nan")
    if isinstance(value, dict):
        return {str(key): _mat_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mat_safe(item) for item in value]
    return value


def add_postmerge_validation_to_merge_mat(path: Path, validation: dict[str, object]) -> None:
    """Add staged post-merge QC to the legacy-compatible merge MAT file."""

    loaded = loadmat(path, simplify_cells=True)
    payload = {key: value for key, value in loaded.items() if not key.startswith("__")}
    merge_info = payload.get("mergeInfo", {})
    if not isinstance(merge_info, dict):
        merge_info = {"legacyMergeInfo": merge_info}
    matlab_validation = _mat_safe(validation)
    merge_info["postmergeValidation"] = matlab_validation
    payload["mergeInfo"] = merge_info
    payload["postmerge_validation"] = matlab_validation
    savemat(path, payload, do_compression=True, long_field_names=True)


def add_traceability_to_merge_mat(
    path: Path,
    *,
    probe_indices: list[int],
    recording_start_anchors: list[dict[str, object]],
) -> None:
    """Persist worker selection/anchor provenance in staged merge metadata."""

    loaded = loadmat(path, simplify_cells=True)
    payload = {key: value for key, value in loaded.items() if not key.startswith("__")}
    merge_info = payload.get("mergeInfo", {})
    if not isinstance(merge_info, dict):
        merge_info = {"legacyMergeInfo": merge_info}
    safe_anchors = _mat_safe(recording_start_anchors)
    merge_info["probeIndices"] = np.asarray(probe_indices, dtype=np.int32)
    merge_info["recordingStartAnchors"] = safe_anchors
    payload["mergeInfo"] = merge_info
    payload["probe_indices"] = np.asarray(probe_indices, dtype=np.int32)
    payload["recording_start_anchors"] = safe_anchors
    savemat(path, payload, do_compression=True, long_field_names=True)


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


def _source_coordinates(
    model: SyncModel,
    canonical_positions: np.ndarray,
    gaps: tuple[DeviceGap, ...],
    *,
    fs: float,
    guard_samples: int = 0,
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
    return source, valid


def _common_master_interval(
    recordings: list[Recording],
    models: list[SyncModel],
    master_index: int,
    device_gaps: list[DeviceGap] | None = None,
    minimum_common_start: int = 0,
    maximum_common_end: int | None = None,
    maximum_common_end_device_index: int | None = None,
    maximum_common_end_reason: str = "",
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
        upper_candidates.append(
            {
                "device_index": device_index,
                "device_name": recording.device_name,
                "stream": "analog",
                "candidate_master_sample": (
                    (recording.analog_samples - 1) * res_rate
                    - model.intercept_samples + total_missing
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
        if np.all(np.abs(positions - nearest) <= INTEGER_MAPPING_TOLERANCE_SAMPLES):
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
) -> tuple[int, int]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; enable overwrite to regenerate: {output_path}")
    partial = atomic_output_path(output_path)
    if partial.exists():
        partial.unlink()
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
    try:
        with partial.open("wb") as output:
            for start in range(0, output_count, chunk_samples):
                count = min(chunk_samples, output_count - start)
                if stream == "ephys":
                    master_positions = common_start + start + np.arange(count, dtype=np.float64)
                else:
                    master_positions = common_start + (start + np.arange(count, dtype=np.float64)) * (fs / ANALOG_FS)
                combined = np.empty((count, total_channels), dtype=np.int16)
                channel_start = 0
                for device_index, (recording, model, source, n_channels) in enumerate(
                    zip(recordings, models, mapped, channels)
                ):
                    guard = EPHYS_SINC_HALF_WIDTH if stream == "ephys" else int(np.ceil(fs / ANALOG_FS))
                    source_ephys, valid = _source_coordinates(
                        model,
                        master_positions,
                        gaps_by_device.get(device_index, ()),
                        fs=fs,
                        guard_samples=guard,
                    )
                    source_positions = source_ephys if stream == "ephys" else source_ephys / (fs / ANALOG_FS)
                    data = np.zeros((count, n_channels), dtype=np.int16)
                    valid_indices = np.flatnonzero(valid)
                    runs: list[np.ndarray] = []
                    if valid_indices.size:
                        boundaries = np.flatnonzero(np.diff(valid_indices) > 1) + 1
                        runs = list(np.split(valid_indices, boundaries))
                    if stream == "ephys":
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
                    channel_start += n_channels
                combined.astype("<i2", copy=False).tofile(output)
                if progress is not None:
                    progress(f"write_{stream}", 100.0 * (start + count) / output_count)
        replace_atomic(partial, output_path)
    finally:
        for source in mapped:
            close_memmap(source)
        if partial.exists():
            partial.unlink()
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


def _write_layout(path: Path, recordings: list[Recording]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(
            ["merged_channel", "device_index", "device_name", "recording_name", "device_channel", "device_folder"]
        )
        merged_channel = 1
        for device_index, recording in enumerate(recordings, start=1):
            for channel in range(1, recording.n_channels + 1):
                writer.writerow(
                    [
                        merged_channel,
                        device_index,
                        recording.device_name,
                        recording.recording_name,
                        channel,
                        str(recording.folder),
                    ]
                )
                merged_channel += 1


def _write_events(
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
) -> Path:
    mapped = interleaved_memmap(analog_path, total_analog_channels, analog_samples)
    summary_path = output_folder / "wild_multilogger_events.tsv"
    published_output_folder = output_folder if published_output_folder is None else published_output_folder
    rows: list[list[object]] = []
    gaps_by_device = _device_gap_map(list(device_gaps or ()))
    block_start = 0
    try:
        for device_index, recording in enumerate(recordings, start=1):
            digital = np.asarray(mapped[:, block_start], dtype=np.uint16)
            gap_edges: list[int] = []
            for gap in gaps_by_device.get(device_index - 1, ()):
                gap_edges.extend(
                    [
                        int(np.floor((gap.canonical_start_sample - common_start) * ANALOG_FS / fs)),
                        int(np.ceil((gap.canonical_end_sample - common_start) * ANALOG_FS / fs)),
                    ]
                )
            for bit_index in range(16):
                state = ((digital >> bit_index) & 1).astype(np.int8)
                transitions = np.diff(np.concatenate(([0], state, [0])))
                starts = np.flatnonzero(transitions == 1)
                ends = np.flatnonzero(transitions == -1)
                count = min(starts.size, ends.size)
                candidate_pairs = [
                    (int(start), int(end)) for start, end in zip(starts[:count], ends[:count])
                ]
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
                if count:
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
                [
                    device_index,
                    recording.device_name,
                    recording.recording_name,
                    bit_index + 1,
                    block_start + 1,
                    count,
                    gap_affected_count,
                    event_text,
                ]
                )
            block_start += recording.analog_channels
    finally:
        close_memmap(mapped)
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(
            [
                "device_index",
                "device_name",
                "recording_name",
                "digital_bit",
                "merged_analog_channel",
                "n_events",
                "n_gap_affected_events_excluded",
                "event_file",
            ]
        )
        writer.writerows(rows)
    return summary_path


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
) -> dict[str, str]:
    output_folder.mkdir(parents=True, exist_ok=True)
    models = _device_models(recordings, master_index, pair_models)
    device_gaps = list(device_gaps or ())
    common_start, common_end, common_interval_limits = _common_master_interval(
        recordings,
        models,
        master_index,
        device_gaps,
        minimum_common_start,
        maximum_common_end,
        maximum_common_end_device_index,
        maximum_common_end_reason,
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
    time_path = output_folder / "time.dat"
    layout_path = output_folder / "wild_preprocess_channel_layout.tsv"
    merge_mat_path = output_folder / "wild_multilogger_mergeInfo.mat"
    merge_json_path = output_folder / "wild_multilogger_mergeInfo.json"
    published_output_folder = output_folder if published_output_folder is None else published_output_folder
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
    )
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
    )
    _write_time_dat(time_path, ephys_samples, overwrite)
    _write_layout(layout_path, recordings)
    events_path = _write_events(
        analog_path,
        output_folder,
        recordings,
        analog_samples,
        total_analog_channels,
        overwrite,
        published_output_folder,
        device_gaps,
        common_start,
        recordings[master_index].fs,
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
        "analog_samples": analog_samples,
        "analog_channels": total_analog_channels,
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
        "imu_status": "not generated by the Python multi-device backend",
        "devices": [
            {
                "folder": str(recording.folder),
                "device_name": recording.device_name,
                "recording_name": recording.recording_name,
                "scale": model.source_scale(recordings[master_index].fs),
                "intercept_samples": model.intercept_samples,
                "drift_ppm": model.drift_ppm,
                "mapped_ephys_start_sample": (
                    model.source_scale(recordings[master_index].fs) * common_start
                    + model.intercept_samples
                    - sum(
                        gap.missing_samples
                        for gap in device_gaps
                        if gap.device_index == device_index
                        and gap.canonical_end_sample <= common_start
                    )
                ),
                "mapped_ephys_end_sample": (
                    model.source_scale(recordings[master_index].fs) * common_end
                    + model.intercept_samples
                    - sum(
                        gap.missing_samples
                        for gap in device_gaps
                        if gap.device_index == device_index
                        and gap.canonical_end_sample <= common_end
                    )
                ),
                "mapped_analog_start_sample": (
                    model.source_scale(recordings[master_index].fs) * common_start
                    + model.intercept_samples
                    - sum(
                        gap.missing_samples
                        for gap in device_gaps
                        if gap.device_index == device_index
                        and gap.canonical_end_sample <= common_start
                    )
                )
                / (recordings[master_index].fs / ANALOG_FS),
                "mapped_analog_end_sample": (
                    model.source_scale(recordings[master_index].fs) * common_end
                    + model.intercept_samples
                    - sum(
                        gap.missing_samples
                        for gap in device_gaps
                        if gap.device_index == device_index
                        and gap.canonical_end_sample <= common_end
                    )
                )
                / (recordings[master_index].fs / ANALOG_FS),
            }
            for device_index, (recording, model) in enumerate(zip(recordings, models), start=1)
        ],
    }
    merge_json_path.write_text(json.dumps(merge_info, indent=2), encoding="utf-8")
    savemat(
        merge_mat_path,
        {
            "mergeInfo": {
                "mode": "multiMerge",
                "runId": run_id,
                "files": np.asarray([str(recording.amplifier_file) for recording in recordings], dtype=object),
                "folders": np.asarray([str(recording.folder) for recording in recordings], dtype=object),
                "masterIndex": master_index + 1,
                "fs": recordings[master_index].fs,
                "nChannels": total_channels,
                "nSamples": ephys_samples,
                "commonStartSample": common_start,
                "commonEndSample": common_end,
                "amplifierFile": str(published_output_folder / amplifier_path.name),
                "analogFile": str(published_output_folder / analog_path.name),
                "timeFile": str(published_output_folder / time_path.name),
                "layoutFile": str(published_output_folder / layout_path.name),
                "backend": SYNC_ALGORITHM_VERSION,
                "commonIntervalLimits": common_interval_limits,
                "deviceGapDeviceIndex": np.asarray(
                    [gap.device_index for gap in device_gaps], dtype=np.int32
                ),
                "deviceGapCanonicalStartSample": np.asarray(
                    [gap.canonical_start_sample for gap in device_gaps], dtype=np.int64
                ),
                "deviceGapMissingSamples": np.asarray(
                    [gap.missing_samples for gap in device_gaps], dtype=np.int64
                ),
                "deviceGapInterpolationGuardSamples": EPHYS_SINC_HALF_WIDTH,
                "deviceGapZeroFillStartSample": np.asarray(
                    [gap.canonical_start_sample - EPHYS_SINC_HALF_WIDTH for gap in device_gaps],
                    dtype=np.int64,
                ),
                "deviceGapZeroFillEndSample": np.asarray(
                    [gap.canonical_end_sample + EPHYS_SINC_HALF_WIDTH for gap in device_gaps],
                    dtype=np.int64,
                ),
                "deviceGapOutputAction": np.asarray(
                    [output_gap_action(gap) for gap in device_gaps], dtype=object
                ),
                "deviceMappedEphysStartSample": np.asarray(
                    [item["mapped_ephys_start_sample"] for item in merge_info["devices"]]
                ),
                "deviceMappedEphysEndSample": np.asarray(
                    [item["mapped_ephys_end_sample"] for item in merge_info["devices"]]
                ),
                "deviceMappedAnalogStartSample": np.asarray(
                    [item["mapped_analog_start_sample"] for item in merge_info["devices"]]
                ),
                "deviceMappedAnalogEndSample": np.asarray(
                    [item["mapped_analog_end_sample"] for item in merge_info["devices"]]
                ),
            },
            "master_index": master_index + 1,
            "fs": recordings[master_index].fs,
            "common_start_master_sample": common_start,
            "common_end_master_sample": common_end,
            "n_samples": ephys_samples,
            "n_channels": total_channels,
            "device_scale": np.asarray([item["scale"] for item in merge_info["devices"]]),
            "device_intercept_samples": np.asarray(
                [item["intercept_samples"] for item in merge_info["devices"]]
            ),
            "device_drift_ppm": np.asarray([item["drift_ppm"] for item in merge_info["devices"]]),
            "common_interval_limits": common_interval_limits,
            "device_gap_device_index": np.asarray(
                [gap.device_index for gap in device_gaps], dtype=np.int32
            ),
            "device_gap_canonical_start_sample": np.asarray(
                [gap.canonical_start_sample for gap in device_gaps], dtype=np.int64
            ),
            "device_gap_missing_samples": np.asarray(
                [gap.missing_samples for gap in device_gaps], dtype=np.int64
            ),
            "device_gap_interpolation_guard_samples": EPHYS_SINC_HALF_WIDTH,
            "device_gap_zero_fill_start_sample": np.asarray(
                [gap.canonical_start_sample - EPHYS_SINC_HALF_WIDTH for gap in device_gaps],
                dtype=np.int64,
            ),
            "device_gap_zero_fill_end_sample": np.asarray(
                [gap.canonical_end_sample + EPHYS_SINC_HALF_WIDTH for gap in device_gaps],
                dtype=np.int64,
            ),
            "device_gap_output_action": np.asarray(
                [output_gap_action(gap) for gap in device_gaps], dtype=object
            ),
            "device_source_ephys_start": np.asarray(
                [item["mapped_ephys_start_sample"] for item in merge_info["devices"]]
            ),
            "device_source_ephys_end": np.asarray(
                [item["mapped_ephys_end_sample"] for item in merge_info["devices"]]
            ),
            "device_source_analog_start": np.asarray(
                [item["mapped_analog_start_sample"] for item in merge_info["devices"]]
            ),
            "device_source_analog_end": np.asarray(
                [item["mapped_analog_end_sample"] for item in merge_info["devices"]]
            ),
            "device_folders": np.asarray([str(recording.folder) for recording in recordings], dtype=object),
        },
        do_compression=True,
        long_field_names=True,
    )
    return {
        "amplifier": str(amplifier_path),
        "analog": str(analog_path),
        "time": str(time_path),
        "layout": str(layout_path),
        "events": str(events_path),
        "merge_mat": str(merge_mat_path),
        "merge_json": str(merge_json_path),
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
        "wild_preprocess_channel_layout.tsv",
        "wild_multilogger_events.tsv",
        "wild_multilogger_mergeInfo.mat",
        "wild_multilogger_mergeInfo.json",
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
        )
        if stage_callback is not None:
            stage_callback(staging, staged_outputs)
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
        for name in managed_names:
            destination = output_folder / name
            if destination.exists():
                backup_path = backup / name
                os.replace(destination, backup_path)
                backed_up.append((destination, backup_path))
        for name, staged_path in staged_files.items():
            destination = output_folder / name
            os.replace(staged_path, destination)
            promoted.append(destination)
        (backup / "COMMITTED").write_text("complete\n", encoding="utf-8")
        return {key: str(output_folder / Path(value).name) for key, value in staged_outputs.items()}
    except BaseException:
        for destination in promoted:
            if destination.exists():
                destination.unlink()
        for destination, backup_path in reversed(backed_up):
            if backup_path.exists():
                os.replace(backup_path, destination)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)
