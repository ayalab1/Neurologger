"""Read-only integrity audit for raw WILD multi-device recordings."""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .binary_io import ANALOG_SAMPLE_RATE_HZ, close_memmap, interleaved_memmap
from .models import Recording
from .sync.features import frame_hash_memmap
from .pc_time.decode import collect_packed_updates
from .pc_time.infer import fit_robust_pc_time_model
from .pc_time.validate import validate_pc_time_interval


RAW_AUDIT_SCHEMA = "wild_preprocess.raw-audit.v2"


@dataclass(frozen=True)
class RawAuditOptions:
    max_duplication_lag_seconds: float = 5.0
    merge_gap_samples: int = 16
    chunk_samples: int = 1_000_000
    validation_batch_samples: int = 100_000
    max_parallel_workers: int = 2

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.max_duplication_lag_seconds)
            or self.max_duplication_lag_seconds <= 0
        ):
            raise ValueError("max_duplication_lag_seconds must be finite and positive")
        if self.merge_gap_samples < 0:
            raise ValueError("merge_gap_samples must be non-negative")
        if self.chunk_samples <= 0 or self.validation_batch_samples <= 0:
            raise ValueError("chunk and validation batch sizes must be positive")
        if self.max_parallel_workers <= 0:
            raise ValueError("max_parallel_workers must be positive")


def _screening_channels(n_channels: int) -> np.ndarray:
    if n_channels < 8:
        return np.arange(n_channels, dtype=int)
    return np.linspace(0, n_channels - 1, 8, dtype=int)


def _screening_keys(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two deterministic keys; every full-frame equality preserves them."""

    contiguous = np.ascontiguousarray(values, dtype="<i2")
    if contiguous.shape[1] >= 8:
        return contiguous[:, :4].view("<u8").ravel(), contiguous[:, 4:8].view("<u8").ravel()
    padded = np.zeros((contiguous.shape[0], 8), dtype="<i2")
    padded[:, : contiguous.shape[1]] = contiguous
    return padded[:, :4].view("<u8").ravel(), padded[:, 4:].view("<u8").ravel()


def _validated_exact_pairs(
    mapped: np.memmap,
    earlier: np.ndarray,
    later: np.ndarray,
    batch_samples: int,
) -> np.ndarray:
    valid = np.zeros(later.size, dtype=bool)
    for start in range(0, later.size, batch_samples):
        stop = min(later.size, start + batch_samples)
        left = earlier[start:stop]
        right = later[start:stop]
        valid[start:stop] = np.all(
            np.asarray(mapped[left, :]) == np.asarray(mapped[right, :]), axis=1
        )
    return valid


def _full_frame_hashes(
    mapped: np.memmap, positions: np.ndarray, batch_samples: int
) -> np.ndarray:
    """Hash full frames in batches; raw equality remains the final authority."""

    hashes = np.empty(positions.size, dtype=np.uint64)
    channel_index = np.arange(1, mapped.shape[1] + 1, dtype=np.uint64)
    weights = channel_index * np.uint64(0x9E3779B185EBCA87)
    salts = channel_index * np.uint64(0xC2B2AE3D27D4EB4F)
    for start in range(0, positions.size, batch_samples):
        stop = min(positions.size, start + batch_samples)
        rows = np.asarray(mapped[positions[start:stop], :]).view("<u2").astype(np.uint64)
        mixed = (rows + salts) * weights
        hashes[start:stop] = np.bitwise_xor.reduce(mixed, axis=1)
    return hashes


def _exact_duplication_pairs(
    recording: Recording,
    options: RawAuditOptions,
    *,
    frame_hash_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    max_lag = min(
        recording.n_samples - 1,
        max(1, int(round(options.max_duplication_lag_seconds * recording.fs))),
    )
    mapped = interleaved_memmap(
        recording.amplifier_file, recording.n_channels, recording.n_samples
    )
    precomputed_hashes = (
        frame_hash_memmap(Path(frame_hash_path), recording.n_samples)
        if frame_hash_path is not None
        else None
    )
    channels = _screening_channels(recording.n_channels)
    later_parts: list[np.ndarray] = []
    lag_parts: list[np.ndarray] = []
    try:
        for base in range(0, recording.n_samples, options.chunk_samples):
            end = min(recording.n_samples, base + options.chunk_samples)
            lookback = max(0, base - max_lag)
            positions = np.arange(lookback, end, dtype=np.int64)
            if precomputed_hashes is not None:
                full_hash = np.asarray(precomputed_hashes[lookback:end], dtype=np.uint64)
                order = np.lexsort((positions, full_hash))
                sorted_hash = full_hash[order]
                sorted_positions = positions[order]
                equal = sorted_hash[1:] == sorted_hash[:-1]
                duplicate = np.r_[equal, False] | np.r_[False, equal]
                if not np.any(duplicate):
                    continue
                candidate_positions = sorted_positions[duplicate]
                candidate_hash = sorted_hash[duplicate]
                full_order = np.lexsort((candidate_positions, candidate_hash))
                candidate_positions = candidate_positions[full_order]
                candidate_hash = candidate_hash[full_order]
                equal_full_key = candidate_hash[1:] == candidate_hash[:-1]
                earlier = candidate_positions[:-1][equal_full_key]
                later = candidate_positions[1:][equal_full_key]
                lags = later - earlier
                keep = (lags > 0) & (lags <= max_lag) & (later >= base)
                earlier = earlier[keep]
                later = later[keep]
                lags = lags[keep]
                if later.size == 0:
                    continue
                exact = _validated_exact_pairs(
                    mapped, earlier, later, options.validation_batch_samples
                )
                if np.any(exact):
                    later_parts.append(later[exact])
                    lag_parts.append(lags[exact])
                continue
            selected = np.asarray(mapped[lookback:end, channels])
            key_a, key_b = _screening_keys(selected)
            order = np.lexsort((positions, key_b, key_a))
            sorted_a = key_a[order]
            sorted_b = key_b[order]
            sorted_positions = positions[order]
            equal = (sorted_a[1:] == sorted_a[:-1]) & (sorted_b[1:] == sorted_b[:-1])
            duplicate = np.r_[equal, False] | np.r_[False, equal]
            if not np.any(duplicate):
                continue
            candidate_positions = sorted_positions[duplicate]
            candidate_a = sorted_a[duplicate]
            candidate_b = sorted_b[duplicate]
            full_hash = _full_frame_hashes(
                mapped, candidate_positions, options.validation_batch_samples
            )
            full_order = np.lexsort(
                (candidate_positions, full_hash, candidate_b, candidate_a)
            )
            candidate_positions = candidate_positions[full_order]
            candidate_a = candidate_a[full_order]
            candidate_b = candidate_b[full_order]
            full_hash = full_hash[full_order]
            equal_full_key = (
                (candidate_a[1:] == candidate_a[:-1])
                & (candidate_b[1:] == candidate_b[:-1])
                & (full_hash[1:] == full_hash[:-1])
            )
            earlier = candidate_positions[:-1][equal_full_key]
            later = candidate_positions[1:][equal_full_key]
            lags = later - earlier
            keep = (lags > 0) & (lags <= max_lag) & (later >= base)
            earlier = earlier[keep]
            later = later[keep]
            lags = lags[keep]
            if later.size == 0:
                continue
            exact = _validated_exact_pairs(
                mapped, earlier, later, options.validation_batch_samples
            )
            if np.any(exact):
                later_parts.append(later[exact])
                lag_parts.append(lags[exact])
    finally:
        if precomputed_hashes is not None:
            close_memmap(precomputed_hashes)
        close_memmap(mapped)
    if not later_parts:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    later = np.concatenate(later_parts)
    lags = np.concatenate(lag_parts)
    order = np.lexsort((later, lags))
    later = later[order]
    lags = lags[order]
    unique = np.r_[True, (later[1:] != later[:-1]) | (lags[1:] != lags[:-1])]
    return later[unique], lags[unique]


def _runs_and_episodes(
    later: np.ndarray,
    lags: np.ndarray,
    merge_gap_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs: list[dict[str, Any]] = []
    for lag in np.unique(lags):
        positions = later[lags == lag]
        split = np.flatnonzero(np.diff(positions) != 1) + 1
        for group in np.split(positions, split):
            start = int(group[0])
            end = int(group[-1]) + 1
            runs.append(
                {
                    "duplicate_start_sample": start,
                    "duplicate_end_sample": end,
                    "source_start_sample": start - int(lag),
                    "source_end_sample": end - int(lag),
                    "lag_samples": int(lag),
                    "exact_match_samples": end - start,
                }
            )
    runs.sort(key=lambda item: (item["lag_samples"], item["duplicate_start_sample"]))
    episodes: list[dict[str, Any]] = []
    for run in runs:
        if (
            episodes
            and episodes[-1]["lag_samples"] == run["lag_samples"]
            and run["duplicate_start_sample"] - episodes[-1]["duplicate_end_sample"]
            <= merge_gap_samples
        ):
            episode = episodes[-1]
            episode["duplicate_end_sample"] = run["duplicate_end_sample"]
            episode["source_end_sample"] = run["source_end_sample"]
            episode["exact_match_samples"] += run["exact_match_samples"]
            episode["exact_run_count"] += 1
            episode["exact_duplicate_fragments"].append(
                [run["duplicate_start_sample"], run["duplicate_end_sample"]]
            )
        else:
            episodes.append(
                {
                    **run,
                    "exact_run_count": 1,
                    "exact_duplicate_fragments": [
                        [run["duplicate_start_sample"], run["duplicate_end_sample"]]
                    ],
                }
            )
    for episode in episodes:
        span = episode["duplicate_end_sample"] - episode["duplicate_start_sample"]
        episode["span_samples"] = span
        episode["match_fraction"] = episode["exact_match_samples"] / span
    episodes.sort(key=lambda item: item["duplicate_start_sample"])
    return runs, episodes


def _union_length(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted((int(start), int(end)) for start, end in intervals if end > start)
    if not ordered:
        return 0
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def scan_exact_duplications(
    recording: Recording,
    options: RawAuditOptions = RawAuditOptions(),
    *,
    frame_hash_path: Path | None = None,
) -> dict[str, Any]:
    later, lags = _exact_duplication_pairs(
        recording, options, frame_hash_path=frame_hash_path
    )
    runs, episodes = _runs_and_episodes(later, lags, options.merge_gap_samples)
    exact_union_samples = _union_length(
        (item["duplicate_start_sample"], item["duplicate_end_sample"]) for item in runs
    )
    envelope_union_samples = _union_length(
        (item["duplicate_start_sample"], item["duplicate_end_sample"]) for item in episodes
    )
    lag_summary = []
    for lag in sorted({int(item["lag_samples"]) for item in episodes}):
        selected = [item for item in episodes if item["lag_samples"] == lag]
        lag_summary.append(
            {
                "lag_samples": lag,
                "lag_ms": 1000.0 * lag / recording.fs,
                "episode_count": len(selected),
                "span_samples": sum(item["span_samples"] for item in selected),
                "exact_match_samples": sum(item["exact_match_samples"] for item in selected),
            }
        )
    for item in runs:
        item["duplicate_start_sec"] = item["duplicate_start_sample"] / recording.fs
        item["duplicate_end_sec"] = item["duplicate_end_sample"] / recording.fs
    for item in episodes:
        item["duplicate_start_sec"] = item["duplicate_start_sample"] / recording.fs
        item["duplicate_end_sec"] = item["duplicate_end_sample"] / recording.fs
        item["duration_ms"] = 1000.0 * item["span_samples"] / recording.fs
    return {
        "searched_max_lag_samples": min(
            recording.n_samples - 1,
            max(1, int(round(options.max_duplication_lag_seconds * recording.fs))),
        ),
        "searched_max_lag_seconds": options.max_duplication_lag_seconds,
        "merge_gap_samples": options.merge_gap_samples,
        "exact_pair_samples": int(later.size),
        "exact_run_count": len(runs),
        "episode_count": len(episodes),
        "short_duplication_episode_count_lt_100_samples": sum(
            item["span_samples"] < 100 for item in episodes
        ),
        "exact_duplication_union_samples": exact_union_samples,
        "exact_duplication_duration_sec": exact_union_samples / recording.fs,
        "exact_duplication_fraction": exact_union_samples / max(1, recording.n_samples),
        "episode_envelope_union_samples": envelope_union_samples,
        "episode_envelope_duration_sec": envelope_union_samples / recording.fs,
        "episode_envelope_fraction": envelope_union_samples / max(1, recording.n_samples),
        "lag_summary": lag_summary,
        "episodes": episodes,
    }


def _terminal_true_count(values: np.ndarray) -> int:
    if values.size == 0 or not bool(values[-1]):
        return 0
    false_positions = np.flatnonzero(~values)
    return values.size if false_positions.size == 0 else values.size - int(false_positions[-1]) - 1


def audit_analog_clock(recording: Recording) -> dict[str, Any]:
    mapped = np.memmap(
        recording.analog_file,
        dtype="<u2",
        mode="r",
        shape=(recording.analog_samples, recording.analog_channels),
    )
    try:
        duration_difference = (
            recording.analog_samples / ANALOG_SAMPLE_RATE_HZ - recording.duration_sec
        )
        counter: dict[str, Any]
        if recording.analog_channels <= 11 or recording.analog_samples < 2:
            counter = {"status": "NO_DATA"}
        else:
            values = np.asarray(mapped[:, 11])
            deltas = (values[1:].astype(np.int64) - values[:-1].astype(np.int64)) % 65536
            normal = deltas == 1
            anomaly_count = int(np.count_nonzero(~normal))
            terminal_transitions = _terminal_true_count(~normal)
            anomaly_values, anomaly_counts = np.unique(deltas[~normal], return_counts=True)
            order = np.argsort(anomaly_counts)[::-1][:10]
            counter = {
                "status": "OK" if anomaly_count == 0 else "WARN",
                "channel_zero_based": 11,
                "transition_count": int(deltas.size),
                "expected_increment_count": int(np.count_nonzero(normal)),
                "expected_increment_fraction": float(np.mean(normal)),
                "anomaly_transition_count": anomaly_count,
                "anomaly_transition_fraction": float(np.mean(~normal)),
                "terminal_consecutive_anomaly_cycles": terminal_transitions + 1
                if terminal_transitions
                else 0,
                "common_anomalous_modulo_deltas": [
                    {"delta": int(anomaly_values[index]), "count": int(anomaly_counts[index])}
                    for index in order
                ],
            }
    finally:
        close_memmap(mapped)
    packed: dict[str, Any]
    if recording.analog_channels <= 15:
        packed = {"status": "NO_DATA", "update_count": 0}
    else:
        packed = {}
        try:
            indices, values, packed_diagnostics = collect_packed_updates(
                recording.analog_file,
                return_diagnostics=True,
            )
        except (ValueError, OSError) as error:
            packed = {"status": "FAIL", "update_count": 0, "message": str(error)}
            indices = np.empty(0, dtype=np.int64)
            values = np.empty(0, dtype=np.uint32)
        if packed.get("status") == "FAIL":
            return {
                "analog_duration_sec": recording.analog_samples / ANALOG_SAMPLE_RATE_HZ,
                "ephys_duration_sec": recording.duration_sec,
                "duration_difference_sec": duration_difference,
                "cycle_counter": counter,
                "packed_pc_time": packed,
            }
        if indices.size == 0:
            packed = {"status": "NO_DATA", "update_count": 0}
        else:
            packed = {
                "status": "UNVALIDATED",
                "update_count": packed_diagnostics.raw_candidate_run_count,
                "accepted_update_count": packed_diagnostics.accepted_update_count,
                "rejected_unstable_update_count": packed_diagnostics.rejected_unstable_run_count,
                "decode": packed_diagnostics.to_dict(),
                "first_update_sample": int(indices[0]),
                "last_update_sample": int(indices[-1]),
                "coverage_fraction": float((indices[-1] - indices[0]) / max(1, recording.n_samples - 1)),
            }
            try:
                from .pc_time.decode import resolve_recording_start_ms

                recording_start_ms, source = resolve_recording_start_ms(recording.folder)
                model = fit_robust_pc_time_model(indices, values, recording.fs, recording_start_ms)
                validation = validate_pc_time_interval(
                    model,
                    sample_rate_hz=recording.fs,
                    common_start_master_sample=0,
                    n_samples=recording.n_samples,
                )
                packed.update(
                    {
                        "status": validation.status,
                        "status_basis": (
                            "packed values stable for at least two source cycles, followed by "
                            "retained-anchor support and ordered clock-regime checks"
                        ),
                        "message": validation.message,
                        "recording_start_source": source,
                        "retained_update_count": model.kept_count,
                        "drift_ppm": model.drift_ppm,
                        "residual_rms_ms": model.residual_rms_ms,
                        "validation": asdict(validation),
                    }
                )
            except (ValueError, OSError) as error:
                packed.update({"status": "FAIL", "message": str(error)})
    return {
        "analog_duration_sec": recording.analog_samples / ANALOG_SAMPLE_RATE_HZ,
        "ephys_duration_sec": recording.duration_sec,
        "duration_difference_sec": duration_difference,
        "cycle_counter": counter,
        "packed_pc_time": packed,
    }


def _recording_audit(recording: Recording, options: RawAuditOptions) -> dict[str, Any]:
    return {
        "device_name": recording.device_name,
        "recording_name": recording.recording_name,
        "folder": str(recording.folder),
        "sample_rate_hz": recording.fs,
        "channel_count": recording.n_channels,
        "sample_count": recording.n_samples,
        "duration_sec": recording.duration_sec,
        "files": {
            "amplifier.dat": {
                "path": str(recording.amplifier_file),
                "bytes": recording.amplifier_file.stat().st_size,
            },
            "analogin.dat": {
                "path": str(recording.analog_file),
                "bytes": recording.analog_file.stat().st_size,
            },
        },
        "exact_duplication": scan_exact_duplications(recording, options),
        "analog_clock": audit_analog_clock(recording),
    }


def _load_sync_evidence(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs = []
    for pair in data.get("pairs", []):
        pairs.append(
            {
                "master_index": pair.get("master_index"),
                "slave_index": pair.get("slave_index"),
                "status": pair.get("status"),
                "offset_steps": pair.get("model", {}).get("offset_steps", []),
                "validated_start_master_sample": pair.get("validated_start_master_sample", 0),
                "terminal_crop_master_sample": pair.get("terminal_crop_master_sample"),
                "terminal_crop_reason": pair.get("terminal_crop_reason", ""),
            }
        )
    return {
        "source_file": str(Path(path).resolve()),
        "status": data.get("status"),
        "master_index": data.get("master_index"),
        "recording_devices": [item.get("device_name") for item in data.get("recordings", [])],
        "device_gap_summary": data.get("device_gap_summary", []),
        "attributed_device_gaps": data.get("device_gaps", []),
        "unresolved_gap_messages": data.get("unresolved_gap_messages", []),
        "pairs": pairs,
        "note": (
            "These are inferred inter-device discontinuities, not direct raw duplication observations; "
            "their fractions must not be added to exact_duplication.exact_duplication_fraction."
        ),
    }


def audit_session(
    recordings: Iterable[Recording],
    *,
    output_path: Path,
    sync_qc_path: Path | None = None,
    options: RawAuditOptions = RawAuditOptions(),
) -> Path:
    ordered = list(recordings)
    if not ordered:
        raise ValueError("At least one recording is required for a raw audit")
    workers = max(1, min(options.max_parallel_workers, len(ordered)))
    if workers == 1:
        device_results = [_recording_audit(recording, options) for recording in ordered]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            device_results = list(
                executor.map(lambda recording: _recording_audit(recording, options), ordered)
            )
    sync_evidence = _load_sync_evidence(sync_qc_path)
    device_index_by_name: dict[str, int] = {}
    if sync_evidence is not None:
        device_index_by_name = {
            name: index
            for index, name in enumerate(sync_evidence.get("recording_devices", []), start=1)
            if name
        }
    gap_summary_by_index = {
        int(item["device_index"]): item
        for item in (sync_evidence or {}).get("device_gap_summary", [])
    }
    master_index = (sync_evidence or {}).get("master_index")
    summary_devices = []
    for fallback_index, device in enumerate(device_results, start=1):
        device_index = device_index_by_name.get(device["device_name"], fallback_index)
        device["device_index"] = device_index
        device["sync_role"] = "master" if device_index == master_index else "slave"
        device["analog_clock"]["packed_pc_time"]["expected_for_sync"] = device_index == master_index
        inferred_missing = gap_summary_by_index.get(
            device_index,
            {
                "device_index": device_index,
                "gap_count": 0,
                "missing_samples": 0,
                "missing_duration_ms": 0.0,
                "missing_fraction": 0.0,
                "longest_gap_samples": 0,
            },
        )
        device["inferred_missing_from_sync"] = inferred_missing
        duplication = device["exact_duplication"]
        summary_devices.append(
            {
                "device_index": device_index,
                "device_name": device["device_name"],
                "sync_role": device["sync_role"],
                "duplication_episode_count": duplication["episode_count"],
                "exact_duplication_samples": duplication["exact_duplication_union_samples"],
                "exact_duplication_fraction": duplication["exact_duplication_fraction"],
                "inferred_gap_count": inferred_missing["gap_count"],
                "inferred_missing_samples": inferred_missing["missing_samples"],
                "inferred_missing_fraction": inferred_missing["missing_fraction"],
            }
        )
    terminal_crops = [
        pair["terminal_crop_master_sample"]
        for pair in (sync_evidence or {}).get("pairs", [])
        if pair.get("terminal_crop_master_sample") is not None
    ]
    validated_starts = [
        int(pair.get("validated_start_master_sample", 0))
        for pair in (sync_evidence or {}).get("pairs", [])
    ]
    validated_start = max(validated_starts, default=0)
    unresolved_count = len((sync_evidence or {}).get("unresolved_gap_messages", []))
    sync_boundary_evidence = {
        "latest_pair_validated_start_master_sample": validated_start,
        "earliest_terminal_crop_master_sample": min(terminal_crops) if terminal_crops else None,
        "unresolved_interior_event_count": unresolved_count,
        "continuous_interval_status": "NOT_ASSESSED",
        "note": (
            "Pair start and terminal-crop bounds do not establish one continuously valid common "
            "interval. Interior unresolved events and the mapped common endpoint must be handled "
            "separately."
        ),
    }
    payload = {
        "schema": RAW_AUDIT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "options": asdict(options),
        "summary": {
            "devices": summary_devices,
            "unresolved_sync_event_count": unresolved_count,
            "earliest_terminal_crop_master_sample": min(terminal_crops) if terminal_crops else None,
            "sync_boundary_evidence": sync_boundary_evidence,
            "fraction_note": (
                "exact_duplication_fraction and inferred_missing_fraction describe different evidence "
                "and may overlap; do not add them."
            ),
        },
        "devices": device_results,
        "sync_qc_evidence": sync_evidence,
    }
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    partial.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(partial, output_path)
    return output_path
