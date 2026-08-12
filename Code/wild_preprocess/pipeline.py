from __future__ import annotations

import os
import json
import math
import shutil
import tempfile
import time
import uuid
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from threading import Lock
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Iterable

from .binary_io import close_memmap, read_ce_params_metadata, recordings_from_folders
from .audit import RawAuditOptions, scan_exact_duplications
from .inspection import write_session_inspection_png
from .integrity import (
    device_gaps_to_intervals,
    merge_compatible_intervals,
    duplicate_destination_intervals,
    SourceToCanonicalMapper,
    terminal_support_from_pair,
    terminal_support_to_interval,
    unresolved_boundaries_from_offset_clusters,
    unresolved_boundary_to_interval,
)
from .models import (
    DeviceSyncAnchor,
    DeviceSyncSegment,
    DeviceGap,
    PipelineResult,
    SyncModel,
    SyncOptions,
    SyncPairResult,
    UnresolvedBoundary,
    validate_device_sync_segments,
)
from .report import save_pair_figure
from .sync.features import build_raw_evidence_scan, feature_memmap
from .sync.gaps import (
    canonicalize_master_sample,
    detect_isolated_offset_crop,
    detect_unconfirmed_terminal_crop,
    detect_adaptive_change_points,
    gap_summary,
    infer_device_gaps,
    localize_relative_offset_step,
    verify_isolated_offset_alias,
)
from .sync.infer import (
    anchors_from_accepted_observations,
    fit_affine_sync_model,
    fit_independent_device_segments,
)
from .sync.merge import apply_staged_zero_fill, merge_recordings
from .sync.observe import observe_pair
from .sync.observe import estimate_lag, _tracking_rejection_reasons
from .sync.attribution import (
    SlaveSlaveEvidence,
    VerifiedPairChange,
    attribute_targeted_events,
)
from .sync.postmerge import (
    PostMergeValidationResult,
    postmerge_exclusion_intervals,
    validate_segment_staged_merge,
)
from .sync.validate import validate_pair
from .version import PIPELINE_ALGORITHM_VERSION, RUN_MANIFEST_SCHEMA_VERSION, SYNC_ALGORITHM_VERSION
from .pc_time import (
    CE64_RAW_MISC_LAYOUT,
    PcTimeOptions,
    collect_packed_updates,
    fit_gap_aware_pc_time_model,
    pc_time_qc_payload,
    resolve_recording_start_ms,
    validate_canonical_pc_time_interval,
    write_canonical_interval_pc_time,
    write_pc_time_summary_png,
    write_pc_time_warning_png,
)


ProgressCallback = Callable[[str, float], None]


class _PerformanceTracker:
    """Accumulate manifest-only wall-clock timing without changing pipeline work."""

    def __init__(self) -> None:
        self._started_at = time.perf_counter()
        self._open: dict[str, float] = {}
        self._stages: dict[str, dict[str, object]] = {}
        self._input_bytes: dict[str, int] = {}
        self._output_bytes: dict[str, dict[str, int]] = {
            "expected": {},
            "actual": {},
        }
        self._storage: dict[str, object] = {}
        self._workers: dict[str, int] = {}

    def configure(
        self,
        *,
        input_bytes: dict[str, int],
        storage: dict[str, object],
        workers: dict[str, int],
    ) -> None:
        self._input_bytes = {key: int(value) for key, value in input_bytes.items()}
        self._storage = dict(storage)
        self._workers = {key: int(value) for key, value in workers.items()}

    def set_workers(self, **workers: int) -> None:
        self._workers.update({key: int(value) for key, value in workers.items()})

    def set_output_bytes(
        self, *, expected: dict[str, int], actual: dict[str, int]
    ) -> None:
        self._output_bytes = {
            "expected": {key: int(value) for key, value in expected.items()},
            "actual": {key: int(value) for key, value in actual.items()},
        }

    def begin(self, name: str) -> None:
        if name in self._open:
            raise RuntimeError(f"performance stage is already active: {name}")
        self._open[name] = time.perf_counter()

    def end(
        self,
        name: str,
        *,
        bytes_read: int = 0,
        bytes_written: int = 0,
        status: str = "complete",
        byte_accounting: str | None = None,
    ) -> None:
        started_at = self._open.pop(name, None)
        if started_at is None:
            return
        entry = self._stages.setdefault(
            name,
            {
                "name": name,
                "wall_seconds": 0.0,
                "invocations": 0,
                "bytes_read": 0,
                "bytes_written": 0,
                "status": "complete",
            },
        )
        entry["wall_seconds"] = float(entry["wall_seconds"]) + max(
            0.0, time.perf_counter() - started_at
        )
        entry["invocations"] = int(entry["invocations"]) + 1
        entry["bytes_read"] = int(entry["bytes_read"]) + int(bytes_read)
        entry["bytes_written"] = int(entry["bytes_written"]) + int(bytes_written)
        if status != "complete":
            entry["status"] = status
        if byte_accounting is not None:
            entry["byte_accounting"] = byte_accounting

    @contextmanager
    def measure(
        self,
        name: str,
        *,
        bytes_read: int = 0,
        bytes_written: int = 0,
        byte_accounting: str | None = None,
    ):
        self.begin(name)
        try:
            yield
        except BaseException:
            self.end(
                name,
                bytes_read=bytes_read,
                bytes_written=bytes_written,
                status="failed",
                byte_accounting=byte_accounting,
            )
            raise
        else:
            self.end(
                name,
                bytes_read=bytes_read,
                bytes_written=bytes_written,
                byte_accounting=byte_accounting,
            )

    def payload(self) -> dict[str, object]:
        stages: list[dict[str, object]] = []
        for entry in self._stages.values():
            item = dict(entry)
            elapsed = float(item["wall_seconds"])
            for field in ("bytes_read", "bytes_written"):
                byte_count = int(item[field])
                if byte_count:
                    item[f"{field}_per_second"] = byte_count / max(elapsed, 1e-12)
            stages.append(item)
        return {
            "schema_version": 1,
            "clock": "time.perf_counter",
            "total_wall_seconds": max(0.0, time.perf_counter() - self._started_at),
            "input_bytes": dict(self._input_bytes),
            "output_bytes": {
                key: dict(value) for key, value in self._output_bytes.items()
            },
            "storage": dict(self._storage),
            "workers": dict(self._workers),
            "stages": stages,
        }


class _StagedMergeRejected(RuntimeError):
    """A validation gate rejected a private staging directory."""

    def __init__(self, validation: PostMergeValidationResult):
        super().__init__(validation.message)
        self.validation = validation


def _overall_status(pairs: list[SyncPairResult]) -> str:
    statuses = {pair.status for pair in pairs}
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "OK"


def _manifest_warnings(pairs: list[SyncPairResult]) -> list[dict[str, object]]:
    """Return only component warnings; normal OK diagnostics stay in sync QC."""

    return [
        {"slave_index": pair.slave_index, "status": pair.status, "message": pair.message}
        for pair in pairs
        if pair.status != "OK"
    ]


def _classified_interval_summary(
    intervals: list,
    *,
    canonical_start_sample: int,
    n_samples: int,
    device_count: int,
) -> list[dict[str, object]]:
    canonical_end = canonical_start_sample + n_samples
    grouped: dict[tuple[int, str], list[tuple[int, int]]] = {}
    counts: dict[tuple[int, str], int] = {}
    for interval in intervals:
        start = max(canonical_start_sample, interval.canonical_start_sample)
        end = min(canonical_end, interval.canonical_end_sample)
        if end <= start:
            continue
        for device_index in interval.affected_device_indices:
            key = (device_index, interval.kind)
            grouped.setdefault(key, []).append((start, end))
            counts[key] = counts.get(key, 0) + 1
    rows: list[dict[str, object]] = []
    for device_index in range(1, device_count + 1):
        for key in sorted(key for key in grouped if key[0] == device_index):
            total = 0
            current_start = current_end = None
            for start, end in sorted(grouped[key]):
                if current_end is not None and start <= current_end:
                    current_end = max(current_end, end)
                else:
                    if current_end is not None and current_start is not None:
                        total += current_end - current_start
                    current_start, current_end = start, end
            if current_end is not None and current_start is not None:
                total += current_end - current_start
            rows.append(
                {
                    "device_index": device_index,
                    "kind": key[1],
                    "interval_count": counts[key],
                    "excluded_samples": total,
                    "excluded_fraction": total / max(1, n_samples),
                }
            )
    return rows


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def _sync_metadata(result: PipelineResult) -> dict[str, object]:
    payload = result.to_dict()
    payload.pop("outputs", None)
    payload.pop("output_folder", None)
    payload["status"] = result.outputs.get("sync_status", result.status)
    canonical_samples = result.recordings[result.master_index - 1].n_samples + sum(
        gap.missing_samples
        for gap in result.device_gaps
        if gap.device_index == result.master_index
    )
    payload["device_gap_summary"] = gap_summary(
        result.device_gaps,
        device_count=len(result.recordings),
        canonical_samples=canonical_samples,
    )
    return payload


def _master_device_segments(
    recording,
    *,
    device_index: int,
    canonical_end_sample: int,
    device_gaps: list,
    unresolved_boundaries: list[UnresolvedBoundary],
) -> tuple[DeviceSyncSegment, ...]:
    """Build the structural canonical-master mapping around explicit gaps."""

    invalid = sorted(
        [
            (gap.canonical_start_sample, gap.canonical_end_sample)
            for gap in device_gaps
            if gap.device_index == device_index
        ]
        + [
            (boundary.canonical_start_sample, boundary.canonical_end_sample)
            for boundary in unresolved_boundaries
        ]
    )
    merged: list[tuple[int, int]] = []
    for start, end in invalid:
        start = max(0, int(start))
        end = min(int(canonical_end_sample), int(end))
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    supported: list[tuple[int, int]] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            supported.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < canonical_end_sample:
        supported.append((cursor, canonical_end_sample))

    segments: list[DeviceSyncSegment] = []
    master_gaps = sorted(
        (gap for gap in device_gaps if gap.device_index == device_index),
        key=lambda item: item.canonical_start_sample,
    )
    for start, end in supported:
        if end - start < 2:
            continue
        missing_before = sum(
            gap.missing_samples for gap in master_gaps if gap.canonical_end_sample <= start
        )
        intercept = float(-missing_before)
        source_start = start + int(intercept)
        source_end = end + int(intercept)
        if source_start < 0 or source_end > recording.n_samples:
            raise ValueError("canonical master segment maps outside raw source support")
        anchors = (
            DeviceSyncAnchor(
                start,
                float(source_start),
                True,
                "high",
                "canonical master structural start support",
            ),
            DeviceSyncAnchor(
                end - 1,
                float(source_end - 1),
                True,
                "high",
                "canonical master structural end support",
            ),
        )
        segments.append(
            DeviceSyncSegment(
                device_index=device_index,
                canonical_start_sample=start,
                canonical_end_sample=end,
                source_start_sample=source_start,
                source_end_sample=source_end,
                source_scale=1.0,
                source_intercept_samples=intercept,
                anchors=anchors,
                residual_rms_samples=0.0,
                residual_max_abs_samples=0.0,
                confidence="high",
                start_transition="recording_start" if start == 0 else "verified_reacquisition",
                end_transition="recording_end" if end == canonical_end_sample else "invalid_boundary",
                publishable=True,
                evidence="canonical master structural mapping",
            )
        )
    return validate_device_sync_segments(segments, device_index=device_index)


def _pair_device_segments(
    pair: SyncPairResult,
    recording,
    *,
    fs: float,
    options: SyncOptions,
    canonicalize: Callable[[int], int],
    canonical_end_sample: int,
    invalid_ranges: list[tuple[int, int]],
) -> tuple[DeviceSyncSegment, ...]:
    """Fit each supported slave interval from its own verified anchors."""

    if pair.status == "FAIL":
        return ()
    raw_anchors = anchors_from_accepted_observations(pair.observations, fs, options)
    anchors = tuple(
        DeviceSyncAnchor(
            canonical_sample=canonicalize(anchor.canonical_sample),
            source_sample=anchor.source_sample,
            verified=anchor.verified,
            confidence=anchor.confidence,
            evidence=anchor.evidence,
        )
        for anchor in raw_anchors
    )
    adaptive = detect_adaptive_change_points(pair.observations, fs, options)
    boundaries = tuple(canonicalize(point.canonical_boundary_sample) for point in adaptive)
    start = max(0, canonicalize(int(pair.validated_start_master_sample)))
    reliable = [item for item in pair.observations if item.accepted]
    supported_end = (
        canonicalize(
            int(
                round(
                    max(item.center_time_sec for item in reliable) * fs
                    + options.window_seconds * fs / 2.0
                )
            )
        )
        if reliable
        else start
    )
    if pair.terminal_crop_master_sample is not None:
        supported_end = min(supported_end, canonicalize(pair.terminal_crop_master_sample))
    end = min(canonical_end_sample, supported_end)
    if end <= start:
        return ()
    adaptive_guard = max(1, int(round(options.window_seconds * fs / 2.0)))
    guarded_changes = [
        (
            canonicalize(max(0, point.canonical_boundary_sample - adaptive_guard)),
            canonicalize(point.canonical_boundary_sample + adaptive_guard),
        )
        for point in adaptive
    ]
    clipped_ranges = []
    for range_start, range_end in sorted([*invalid_ranges, *guarded_changes]):
        clipped_start = max(start, range_start)
        clipped_end = min(end, range_end)
        if clipped_end > clipped_start:
            if clipped_ranges and clipped_start <= clipped_ranges[-1][1]:
                clipped_ranges[-1] = (
                    clipped_ranges[-1][0],
                    max(clipped_ranges[-1][1], clipped_end),
                )
            else:
                clipped_ranges.append((clipped_start, clipped_end))
    return fit_independent_device_segments(
        anchors,
        boundaries,
        device_index=pair.slave_index,
        canonical_start_sample=start,
        canonical_end_sample=end,
        source_sample_count=recording.n_samples,
        unresolved_ranges=clipped_ranges,
    )


def _targeted_slave_slave_evidence(
    recordings,
    pairs: list[SyncPairResult],
    feature_paths: list[Path],
    changes: list[VerifiedPairChange],
    *,
    master_index: int,
    options: SyncOptions,
) -> tuple[SlaveSlaveEvidence, ...]:
    """Confirm local slave/slave relationship changes only near candidates."""

    if len(pairs) < 2 or not changes:
        return ()
    fs = recordings[master_index].fs
    pair_by_device = {pair.slave_index: pair for pair in pairs}
    slave_devices = sorted(pair_by_device)
    candidate_samples = sorted({change.canonical_sample for change in changes})
    window = max(4, int(round(options.endpoint_probe_seconds * fs)))
    guard = max(window, int(round(options.window_seconds * fs / 2.0)))
    results: list[SlaveSlaveEvidence] = []

    for first_position, first_device in enumerate(slave_devices):
        for second_device in slave_devices[first_position + 1 :]:
            first_recording = recordings[first_device - 1]
            second_recording = recordings[second_device - 1]
            first_feature = feature_memmap(
                feature_paths[first_device - 1], first_recording.n_samples
            )
            second_feature = feature_memmap(
                feature_paths[second_device - 1], second_recording.n_samples
            )
            try:
                first_pair = pair_by_device[first_device]
                second_pair = pair_by_device[second_device]

                def relation(master_start: int) -> int | None:
                    center_time = (master_start + window / 2.0) / fs
                    first_offset = first_pair.model.offset_at_seconds(center_time)
                    second_offset = second_pair.model.offset_at_seconds(center_time)
                    first_start = master_start + round(first_offset)
                    second_start = master_start + round(second_offset)
                    if (
                        master_start < 0
                        or first_start < 0
                        or second_start < 0
                        or first_start + window > first_recording.n_samples
                        or second_start + window > second_recording.n_samples
                    ):
                        return None
                    estimate = estimate_lag(
                        np.asarray(first_feature[first_start : first_start + window]),
                        np.asarray(second_feature[second_start : second_start + window]),
                        options.tracking_max_lag_samples,
                        peak_exclusion_samples=options.peak_exclusion_samples,
                    )
                    if _tracking_rejection_reasons(
                        estimate, options, options.tracking_max_lag_samples
                    ):
                        return None
                    return int(round(second_start - first_start + estimate.lag_samples))

                for sample in candidate_samples:
                    before = relation(sample - guard - window)
                    after = relation(sample + guard)
                    if before is None or after is None:
                        continue
                    results.append(
                        SlaveSlaveEvidence(
                            first_device_index=first_device,
                            second_device_index=second_device,
                            canonical_sample=sample,
                            second_minus_first_delta_samples=after - before,
                            verified=True,
                            evidence="targeted full-rate before/after slave correlation",
                        )
                    )
            finally:
                close_memmap(first_feature)
                close_memmap(second_feature)
    return tuple(results)


def _write_manifest(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )
    return path


def _validate_selected_inputs(
    recordings: list,
    *,
    probe_indices: list[int] | None,
    recording_start_anchors: list[dict[str, object]] | None,
) -> list[int]:
    folders = [recording.folder.resolve() for recording in recordings]
    if len(set(folders)) != len(folders):
        raise ValueError("Selected recording folders must be unique.")
    probes = list(range(1, len(recordings) + 1)) if probe_indices is None else [int(value) for value in probe_indices]
    if sorted(probes) != list(range(1, len(recordings) + 1)):
        raise ValueError("Selected probe indices must be unique and exactly 1..N.")
    if recording_start_anchors is not None:
        if len(recording_start_anchors) != len(recordings):
            raise ValueError("recording_start_anchors must provide one anchor for each selected recording.")
        values: list[int] = []
        for index, anchor in enumerate(recording_start_anchors, start=1):
            if not isinstance(anchor, dict) or "milliseconds_since_midnight" not in anchor:
                raise ValueError(f"recording start anchor {index} is missing milliseconds_since_midnight")
            value = int(anchor["milliseconds_since_midnight"])
            if not 0 <= value < 86_400_000:
                raise ValueError(f"recording start anchor {index} is outside one day")
            values.append(value)
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                circular_difference = abs(
                    (values[right] - values[left] + 43_200_000) % 86_400_000 - 43_200_000
                )
                if circular_difference > 30_000:
                    raise ValueError(
                        f"Selected recording start anchors differ by {circular_difference / 1000:.3f}s "
                        f"(devices {left + 1} and {right + 1}; limit 30s)."
                    )
    return probes


def _input_provenance(
    recordings: list,
    probe_indices: list[int],
    recording_start_anchors: list[dict[str, object]] | None,
) -> list[dict[str, object]]:
    anchors = recording_start_anchors or [
        {
            "milliseconds_since_midnight": None,
            "source": None,
            "recording_date": None,
            "recording_date_source": None,
        }
        for _ in recordings
    ]
    rows: list[dict[str, object]] = []
    for index, (recording, probe, anchor) in enumerate(zip(recordings, probe_indices, anchors), start=1):
        rows.append(
            {
                "device_order": index,
                "probe_index": probe,
                "folder": str(recording.folder),
                "device_name": recording.device_name,
                "recording_name": recording.recording_name,
                "files": {
                    "amplifier.dat": {"path": str(recording.amplifier_file), "bytes": recording.amplifier_file.stat().st_size},
                    "analogin.dat": {"path": str(recording.analog_file), "bytes": recording.analog_file.stat().st_size},
                    "CE_params.bin": {"path": str(recording.ce_params_file), "bytes": recording.ce_params_file.stat().st_size},
                },
                "ce_header": read_ce_params_metadata(recording.ce_params_file),
                "recording_start_anchor": anchor,
            }
        )
    return rows


def run_multidevice_sync(
    device_folders: Iterable[Path],
    *,
    master_index: int,
    output_folder: Path,
    overwrite: bool = False,
    merge: bool = True,
    options: SyncOptions | None = None,
    progress: ProgressCallback | None = None,
    native_pc_time: bool = False,
    recording_start_ms: int | None = None,
    validate_postmerge: bool | None = None,
    pc_time_options: PcTimeOptions | None = None,
    probe_indices: list[int] | None = None,
    recording_start_anchors: list[dict[str, object]] | None = None,
    integrity_duplication_scan: bool = False,
    write_event_files: bool = False,
) -> PipelineResult:
    """Estimate multi-device timing and optionally write merged streams.

    `master_index` is zero-based in the Python API.  When ``native_pc_time`` is
    enabled this is the complete worker path: sync, private merge staging,
    post-merge validation, native PC-time generation, and atomic publication.
    The default remains false for callers that previously used this as a
    sync/merge-only library function.
    """

    options = options or SyncOptions()
    pc_time_options = pc_time_options or PcTimeOptions()
    performance = _PerformanceTracker()
    if validate_postmerge is None:
        # Preserve the previous library-level sync/merge behaviour.  The
        # production worker always enables native PC time and therefore the
        # publication gate; direct legacy callers can opt in explicitly.
        validate_postmerge = native_pc_time
    with performance.measure("input_inspection"):
        recordings = recordings_from_folders(device_folders)
        probes = _validate_selected_inputs(
            recordings,
            probe_indices=probe_indices,
            recording_start_anchors=recording_start_anchors,
        )
        input_provenance = _input_provenance(recordings, probes, recording_start_anchors)
    if master_index < 0 or master_index >= len(recordings):
        raise ValueError(f"master_index {master_index} is outside {len(recordings)} recordings")
    output_folder = Path(output_folder).resolve()
    if any(output_folder == recording.folder for recording in recordings):
        raise ValueError("The merged output folder must not be one of the raw recording folders.")
    output_folder.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    attempt_folder = output_folder / f".wild_sync_attempt_{run_id}"
    protected_outputs = [output_folder / "wild_preprocess_run.json"]
    if merge:
        protected_outputs.extend(
            [
                output_folder / "amplifier.dat",
                output_folder / "analogin.dat",
                output_folder / "time.dat",
                output_folder / "valid_samples.dat",
                output_folder / "wild_multilogger_session_inspection.png",
            ]
        )
    existing = [path for path in protected_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output exists; enable overwrite to regenerate: " + ", ".join(str(path) for path in existing)
        )
    attempt_folder.mkdir(parents=True, exist_ok=False)

    pairs: list[SyncPairResult] = []
    cache_root_text = os.environ.get("WILD_SYNC_CACHE_DIR", "").strip()
    cache_root = Path(cache_root_text).expanduser().resolve() if cache_root_text else None
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
    input_bytes = {
        "amplifier.dat": sum(recording.amplifier_file.stat().st_size for recording in recordings),
        "analogin.dat": sum(recording.analog_file.stat().st_size for recording in recordings),
        "CE_params.bin": sum(recording.ce_params_file.stat().st_size for recording in recordings),
    }
    input_bytes["total"] = sum(input_bytes.values())
    performance.configure(
        input_bytes=input_bytes,
        storage={
            "input_locations": sorted({str(recording.folder.anchor) for recording in recordings}),
            "output_location": str(output_folder.anchor),
            "feature_cache": {
                "enabled": cache_root is not None,
                "location": str(cache_root) if cache_root is not None else None,
                "reuse": "none; temporary features are deleted after each run",
            },
        },
        workers={"requested_max_parallel_workers": int(options.max_parallel_workers)},
    )
    # Keep the evidence directory alive through exact-duplication confirmation;
    # its frame hashes are consumed after segment construction.  The owner also
    # cleans itself during exception unwinding, while the explicit cleanup below
    # releases large caches as soon as their last consumer finishes.
    temporary_owner = tempfile.TemporaryDirectory(prefix="wild_sync_", dir=cache_root)
    temporary = temporary_owner.name
    if True:
        temporary_path = Path(temporary)
        feature_paths = [
            temporary_path / f"common_mode_{index:02d}.f32"
            for index in range(len(recordings))
        ]
        coarse_feature_paths = [
            temporary_path / f"common_mode_coarse_{index:02d}.f32"
            for index in range(len(recordings))
        ]
        frame_hash_paths = [
            temporary_path / f"frame_hash_{index:02d}.u64"
            for index in range(len(recordings))
        ]
        evidence_scans: dict[int, object] = {}
        feature_percent = [0.0] * len(recordings)
        feature_lock = Lock()

        def build_feature(index: int) -> Path:
            def feature_progress(stage: str, percent: float) -> None:
                if progress is None:
                    return
                with feature_lock:
                    feature_percent[index] = percent
                    progress("build_features", sum(feature_percent) / len(feature_percent))

            scan = build_raw_evidence_scan(
                recordings[index],
                feature_paths[index],
                coarse_feature_paths[index],
                frame_hash_paths[index],
                highpass_hz=options.highpass_hz,
                coarse_target_rate_hz=options.coarse_feature_rate_hz,
                chunk_seconds=options.chunk_seconds,
                progress=feature_progress,
            )
            with feature_lock:
                evidence_scans[index] = scan
            return scan.feature_path

        feature_worker_count = max(
            1, min(int(options.max_parallel_workers), len(recordings))
        )
        performance.set_workers(feature_workers=feature_worker_count)
        performance.begin("raw_evidence_feature_scan")
        try:
            if feature_worker_count == 1:
                for index in range(len(recordings)):
                    build_feature(index)
            else:
                with ThreadPoolExecutor(
                    max_workers=feature_worker_count,
                    thread_name_prefix="wild-sync-feature",
                ) as executor:
                    futures = [executor.submit(build_feature, index) for index in range(len(recordings))]
                    for future in as_completed(futures):
                        future.result()
        except BaseException:
            performance.end(
                "raw_evidence_feature_scan",
                bytes_read=input_bytes["amplifier.dat"],
                status="failed",
            )
            raise
        else:
            performance.end(
                "raw_evidence_feature_scan",
                bytes_read=input_bytes["amplifier.dat"],
                bytes_written=sum(
                    path.stat().st_size
                    for path in [*feature_paths, *coarse_feature_paths, *frame_hash_paths]
                ),
            )

        slave_indices = [index for index in range(len(recordings)) if index != master_index]
        master = recordings[master_index]

        def observe_slave(slave_index: int):
            return observe_pair(
                master,
                recordings[slave_index],
                feature_paths[master_index],
                feature_paths[slave_index],
                options,
                progress=None,
                master_coarse_feature_path=coarse_feature_paths[master_index],
                slave_coarse_feature_path=coarse_feature_paths[slave_index],
                coarse_downsample_factor=evidence_scans[master_index].coarse_downsample_factor,
            )

        observed_by_slave: dict[int, object] = {}
        isolated_boundary_candidates: list[tuple[int, int, int, str]] = []
        worker_count = max(1, min(int(options.max_parallel_workers), len(slave_indices)))
        performance.set_workers(pair_workers=worker_count)
        with performance.measure("coarse_correlation"):
            if worker_count == 1:
                for completed, slave_index in enumerate(slave_indices, start=1):
                    observed_by_slave[slave_index] = observe_slave(slave_index)
                    if progress is not None:
                        progress("sync_pairs", 100.0 * completed / len(slave_indices))
            else:
                with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="wild-sync-pair") as executor:
                    futures = {executor.submit(observe_slave, slave_index): slave_index for slave_index in slave_indices}
                    for completed, future in enumerate(as_completed(futures), start=1):
                        slave_index = futures[future]
                        observed_by_slave[slave_index] = future.result()
                        if progress is not None:
                            progress("sync_pairs", 100.0 * completed / len(slave_indices))

        performance.begin("full_rate_refinement")
        for slave_index in slave_indices:
            slave = recordings[slave_index]
            observed = observed_by_slave[slave_index]
            model = fit_affine_sync_model(observed.observations, master.fs, options=options)
            if model.offset_steps:
                master_feature = feature_memmap(feature_paths[master_index], master.n_samples)
                slave_feature = feature_memmap(feature_paths[slave_index], slave.n_samples)
                try:
                    localized_steps = tuple(
                        localize_relative_offset_step(
                            master_feature,
                            slave_feature,
                            step,
                            fs=master.fs,
                            options=options,
                        )
                        for step in model.offset_steps
                    )
                finally:
                    close_memmap(master_feature)
                    close_memmap(slave_feature)
                model = fit_affine_sync_model(
                    observed.observations,
                    master.fs,
                    options=options,
                    offset_steps=localized_steps,
                )
            isolated_crop = detect_isolated_offset_crop(
                observed.observations,
                model,
                master.fs,
                options,
            )
            isolated_alias_message = ""
            isolated_recheck_evidence = ""
            if isolated_crop is not None:
                master_feature = feature_memmap(feature_paths[master_index], master.n_samples)
                slave_feature = feature_memmap(feature_paths[slave_index], slave.n_samples)
                try:
                    isolated_alias, isolated_recheck_evidence = verify_isolated_offset_alias(
                        master_feature,
                        slave_feature,
                        observed.observations,
                        model,
                        isolated_crop,
                        master.fs,
                        options,
                    )
                finally:
                    close_memmap(master_feature)
                    close_memmap(slave_feature)
                if isolated_alias:
                    isolated_alias_message = (
                        "rejected 1 isolated offset alias after raw recheck; "
                        + isolated_recheck_evidence
                    )
                    for observation_index in isolated_crop[2]:
                        observation = observed.observations[observation_index]
                        observation.accepted = False
                        observation.model_inlier = False
                        observation.rejection_reason = isolated_alias_message
                    model = fit_affine_sync_model(
                        observed.observations,
                        master.fs,
                        options=options,
                        offset_steps=model.offset_steps,
                    )
                    isolated_crop = None
                else:
                    boundary_start, shift, isolated_indices = isolated_crop
                    boundary_end = max(
                        boundary_start + 1,
                        int(
                            math.ceil(
                                max(
                                    observed.observations[index].center_time_sec
                                    + options.window_seconds / 2.0
                                    for index in isolated_indices
                                )
                                * master.fs
                            )
                        ),
                    )
                    isolated_reason = (
                        f"unresolved isolated interior excursion {shift:+.1f} samples; "
                        + isolated_recheck_evidence
                    )
                    isolated_boundary_candidates.append(
                        (boundary_start, boundary_end, slave_index + 1, isolated_reason)
                    )
                    for observation_index in isolated_indices:
                        observation = observed.observations[observation_index]
                        observation.accepted = False
                        observation.model_inlier = False
                        observation.rejection_reason = isolated_reason
                    model = fit_affine_sync_model(
                        observed.observations,
                        master.fs,
                        options=options,
                        offset_steps=model.offset_steps,
                    )
                    isolated_alias_message = isolated_reason
                    isolated_crop = None
            terminal_crop = detect_unconfirmed_terminal_crop(
                observed.observations,
                model,
                master.fs,
                options,
            )
            terminal_crop_master_sample: int | None = None
            terminal_crop_reason = ""
            validation_observations = observed.observations
            crop_candidates: list[tuple[int, float, tuple[int, ...], str]] = []
            if terminal_crop is not None:
                crop_candidates.append((*terminal_crop, "terminal offset shift"))
            if crop_candidates:
                (
                    terminal_crop_master_sample,
                    terminal_shift,
                    terminal_indices,
                    crop_kind,
                ) = min(crop_candidates, key=lambda item: item[0])
                terminal_crop_reason = (
                    f"unconfirmed {crop_kind} {terminal_shift:+.1f} samples; "
                    f"output cropped before master sample {terminal_crop_master_sample}"
                )
                crop_time_sec = terminal_crop_master_sample / master.fs
                terminal_index_set = set(terminal_indices)
                for observation_index, observation in enumerate(observed.observations):
                    if (
                        observation_index in terminal_index_set
                        or observation.center_time_sec - options.window_seconds / 2.0
                        >= crop_time_sec
                    ):
                        observation.accepted = False
                        observation.model_inlier = False
                        observation.rejection_reason = terminal_crop_reason
                validation_observations = [
                    observation
                    for observation in observed.observations
                    if observation.center_time_sec + options.window_seconds / 2.0
                    <= crop_time_sec
                ]
                validated_steps = tuple(
                    step
                    for step in model.offset_steps
                    if step.master_sample < terminal_crop_master_sample
                )
                model = fit_affine_sync_model(
                    validation_observations,
                    master.fs,
                    options=options,
                    offset_steps=validated_steps,
                )
            status, message = validate_pair(
                observed.initial,
                validation_observations,
                model,
                options,
            )
            operational_warnings = [
                item for item in (isolated_alias_message, terminal_crop_reason) if item
            ]
            if operational_warnings and status != "FAIL":
                status = "WARN"
                message = "; ".join([message] + operational_warnings)
            validated_start = observed.validated_start_master_sample
            early_horizon = validated_start + round(
                (2.0 * options.window_seconds + options.step_seconds) * master.fs
            )
            for step in model.offset_steps:
                if step.master_sample > early_horizon:
                    continue
                stable_after = [
                    item
                    for item in observed.observations
                    if item.accepted and item.center_time_sec >= step.time_sec
                ]
                persistence = max(1, int(options.gap_persistence_observations))
                if len(stable_after) >= persistence:
                    validated_start = max(
                        validated_start,
                        round(
                            (
                                stable_after[persistence - 1].center_time_sec
                                + options.window_seconds / 2
                            )
                            * master.fs
                        ),
                    )
            safe_slave = "".join(
                character if character.isalnum() or character in "_.-" else "_"
                for character in slave.device_name
            )
            figure_path = attempt_folder / (
                f"wild_multilogger_sync_master_vs_{safe_slave}_"
                f"{slave.recording_name}_qc.png"
            )
            pair = SyncPairResult(
                master_index=master_index + 1,
                slave_index=slave_index + 1,
                master_folder=str(master.folder),
                slave_folder=str(slave.folder),
                initial_offset_samples=float(observed.initial.lag_samples),
                initial_peak_to_background=observed.initial.peak_to_background,
                initial_peak_margin_fraction=observed.initial.peak_margin_fraction,
                model=model,
                observations=observed.observations,
                status=status,
                message=message,
                figure_file=str(figure_path),
                validated_start_master_sample=validated_start,
                terminal_crop_master_sample=terminal_crop_master_sample,
                terminal_crop_reason=terminal_crop_reason,
            )
            with performance.measure("pair_figure_generation"):
                save_pair_figure(pair, observed, figure_path)
            pairs.append(pair)

        performance.end("full_rate_refinement")
        with performance.measure("attribution_segment_construction"):
            device_gaps, unresolved_gap_messages = infer_device_gaps(
                pairs,
                device_count=len(recordings),
                master_index=master_index,
                fs=master.fs,
                options=options,
            )
            targeted_changes = [
                VerifiedPairChange(
                    slave_device_index=pair.slave_index,
                    canonical_sample=point.canonical_boundary_sample,
                    delta_samples=int(round(point.delta_samples)),
                    evidence=point.evidence,
                )
                for pair in pairs
                for point in detect_adaptive_change_points(
                    pair.observations, master.fs, options
                )
                if int(round(point.delta_samples)) != 0
            ]
            # Sliding observation windows can report the same already-localized
            # gap at several later centers.  Targeted attribution is only for
            # unexplained transitions; do not turn those window echoes into a
            # new all-device unresolved interval.
            explained_radius = max(
                1,
                int(
                    round(
                        (
                            options.gap_event_time_tolerance_seconds
                            + options.window_seconds
                        )
                        * master.fs
                    )
                ),
            )
            targeted_changes = [
                change
                for change in targeted_changes
                if not any(
                    (
                        gap.device_index == change.slave_device_index
                        or gap.device_index == master_index + 1
                    )
                    and abs(gap.canonical_start_sample - change.canonical_sample)
                    <= explained_radius
                    for gap in device_gaps
                )
            ]
            targeted_slave_evidence = _targeted_slave_slave_evidence(
                recordings,
                pairs,
                feature_paths,
                targeted_changes,
                master_index=master_index,
                options=options,
            )
            targeted_decisions = list(
                attribute_targeted_events(
                    targeted_changes,
                    device_count=len(recordings),
                    master_device_index=master_index + 1,
                    observed_slave_indices=(pair.slave_index for pair in pairs),
                    slave_slave_evidence=targeted_slave_evidence,
                    event_tolerance_samples=max(
                        0,
                        int(round(options.gap_event_time_tolerance_seconds * master.fs)),
                    ),
                    magnitude_tolerance_samples=max(
                        1, int(round(options.gap_level_tolerance_samples))
                    ),
                )
            )

            for decision in targeted_decisions:
                if decision.kind not in {"slave", "master"}:
                    continue
                device_index = decision.device_indices[0]
                size = abs(decision.delta_samples)
                decision_canonical_sample = decision.canonical_sample + sum(
                    gap.missing_samples
                    for gap in device_gaps
                    if gap.device_index == master_index + 1
                    and gap.canonical_start_sample <= decision.canonical_sample
                )
                if any(
                    gap.device_index == device_index
                    and abs(gap.canonical_start_sample - decision_canonical_sample)
                    <= max(1, int(round(options.gap_event_time_tolerance_seconds * master.fs)))
                    for gap in device_gaps
                ):
                    continue
                device_gaps.append(
                    DeviceGap(
                        device_index=device_index,
                        canonical_start_sample=decision_canonical_sample,
                        missing_samples=size,
                        duration_ms=1000.0 * size / master.fs,
                        confidence="high",
                        evidence=decision.evidence,
                    )
                )
            device_gaps.sort(
                key=lambda gap: (gap.canonical_start_sample, gap.device_index)
            )

    performance.begin("attribution_segment_construction")
    canonicalize = lambda sample: canonicalize_master_sample(
        int(sample),
        device_gaps,
        master_device_index=master_index + 1,
    )
    unresolved_boundaries = list(
        unresolved_boundaries_from_offset_clusters(
            pairs,
            device_count=len(recordings),
            master_index=master_index,
            fs=master.fs,
            window_seconds=options.window_seconds,
            fallback_step_seconds=options.step_seconds,
            event_time_tolerance_seconds=options.gap_event_time_tolerance_seconds,
            gap_level_tolerance_samples=options.gap_level_tolerance_samples,
            canonicalize_master_sample=canonicalize,
        )
    )
    unresolved_boundaries.extend(
        UnresolvedBoundary(
            canonical_start_sample=canonicalize(start),
            canonical_end_sample=canonicalize(end),
            pair_slave_indices=(slave_index,),
            evidence=evidence,
        )
        for start, end, slave_index, evidence in isolated_boundary_candidates
        if canonicalize(end) > canonicalize(start)
    )
    adaptive_half_window = max(1, int(round(options.window_seconds * master.fs / 2.0)))
    unresolved_boundaries.extend(
        UnresolvedBoundary(
            canonical_start_sample=canonicalize(
                max(0, decision.canonical_sample - adaptive_half_window)
            ),
            canonical_end_sample=canonicalize(
                decision.canonical_sample + adaptive_half_window
            ),
            pair_slave_indices=decision.device_indices,
            evidence=decision.evidence,
        )
        for decision in targeted_decisions
        if decision.kind == "unresolved"
        and canonicalize(decision.canonical_sample + adaptive_half_window)
        > canonicalize(max(0, decision.canonical_sample - adaptive_half_window))
    )
    unresolved_boundaries.sort(
        key=lambda boundary: (
            boundary.canonical_start_sample,
            boundary.canonical_end_sample,
            boundary.pair_slave_indices,
        )
    )
    # Unresolved measured steps remain diagnostics only.  They are never
    # propagated into a later source coordinate; independently verified
    # segments below are the sole new mapping authority.
    device_source_steps = []
    classified_intervals = list(device_gaps_to_intervals(device_gaps))
    classified_intervals.extend(
        unresolved_boundary_to_interval(boundary, device_count=len(recordings))
        for boundary in unresolved_boundaries
    )
    device_terminal_support = [
        support
        for pair in pairs
        if (support := terminal_support_from_pair(
            pair, canonicalize_master_sample=canonicalize
        )) is not None
    ]
    canonical_end_sample = master.n_samples + sum(
        gap.missing_samples
        for gap in device_gaps
        if gap.device_index == master_index + 1
    )
    device_sync_segments: list[DeviceSyncSegment] = list(
        _master_device_segments(
            master,
            device_index=master_index + 1,
            canonical_end_sample=canonical_end_sample,
            device_gaps=device_gaps,
            unresolved_boundaries=unresolved_boundaries,
        )
    )
    for pair in pairs:
        invalid_ranges = [
            (boundary.canonical_start_sample, boundary.canonical_end_sample)
            for boundary in unresolved_boundaries
        ]
        invalid_ranges.extend(
            (gap.canonical_start_sample, gap.canonical_end_sample)
            for gap in device_gaps
            if gap.device_index == pair.slave_index
        )
        device_sync_segments.extend(
            _pair_device_segments(
                pair,
                recordings[pair.slave_index - 1],
                fs=master.fs,
                options=options,
                canonicalize=canonicalize,
                canonical_end_sample=canonical_end_sample,
                invalid_ranges=invalid_ranges,
            )
        )
    performance.end("attribution_segment_construction")
    duplication_scans: dict[int, dict[str, object]] = {}
    performance.set_workers(duplication_workers=0)
    if integrity_duplication_scan:
        audit_options = RawAuditOptions(max_parallel_workers=1)
        duplication_workers = max(1, min(int(options.max_parallel_workers), len(recordings)))
        performance.set_workers(duplication_workers=duplication_workers)
        performance.begin("duplication_confirmation")
        try:
            with ThreadPoolExecutor(max_workers=duplication_workers) as pool:
                futures = {
                    pool.submit(
                        scan_exact_duplications,
                        recording,
                        audit_options,
                        frame_hash_path=frame_hash_paths[device_index],
                    ): device_index
                    for device_index, recording in enumerate(recordings)
                }
                completed_scans = 0
                for future in as_completed(futures):
                    duplication_scans[futures[future]] = future.result()
                    completed_scans += 1
                    if progress is not None:
                        progress(
                            "integrity_scan",
                            100.0 * completed_scans / len(recordings),
                        )
        except BaseException:
            performance.end(
                "duplication_confirmation",
                bytes_read=input_bytes["amplifier.dat"],
                status="failed",
                byte_accounting="logical source bytes; overlap lookback rereads are not counted",
            )
            raise
        else:
            performance.end(
                "duplication_confirmation",
                bytes_read=input_bytes["amplifier.dat"],
                byte_accounting="logical source bytes; overlap lookback rereads are not counted",
            )
        pair_model_by_device = {pair.slave_index - 1: pair.model for pair in pairs}
        for device_index, recording in enumerate(recordings):
            model = pair_model_by_device.get(
                device_index,
                SyncModel(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0),
            )
            to_canonical = SourceToCanonicalMapper(
                device_index=device_index + 1,
                source_scale=model.source_scale(master.fs),
                intercept_samples=model.intercept_samples,
                device_gaps=device_gaps,
                source_steps=(),
                device_sync_segments=[
                    segment
                    for segment in device_sync_segments
                    if segment.device_index == device_index + 1
                ],
            )
            classified_intervals.extend(
                duplicate_destination_intervals(
                    duplication_scans[device_index],
                    device_index=device_index + 1,
                    canonicalize_current_sample=to_canonical,
                )
            )
    temporary_owner.cleanup()
    classified_intervals = list(merge_compatible_intervals(classified_intervals))
    result = PipelineResult(
        recordings=recordings,
        master_index=master_index + 1,
        pairs=pairs,
        run_id=run_id,
        status=(
            "WARN"
            if any(pair.status != "OK" for pair in pairs)
            or unresolved_boundaries
            or any(interval.kind == "duplicate_destination" for interval in classified_intervals)
            or any(
                not any(segment.device_index == pair.slave_index for segment in device_sync_segments)
                for pair in pairs
            )
            else "OK"
        ),
        output_folder=output_folder,
        device_gaps=device_gaps,
        unresolved_gap_messages=unresolved_gap_messages,
        classified_intervals=classified_intervals,
        unresolved_boundaries=unresolved_boundaries,
        device_terminal_support=device_terminal_support,
        device_source_steps=device_source_steps,
        device_sync_segments=device_sync_segments,
        targeted_attributions=[
            {
                "kind": decision.kind,
                "canonical_sample": canonicalize(decision.canonical_sample),
                "device_indices": list(decision.device_indices),
                "delta_samples": decision.delta_samples,
                "evidence": decision.evidence,
            }
            for decision in targeted_decisions
        ],
    )
    result.outputs.update(
        {
            "sync_status": result.status,
            "merge_status": "NOT_RUN",
            "pc_time_status": "NOT_RUN",
            "overall_status": "FAIL" if result.status == "FAIL" else "MERGE_ONLY",
        }
    )

    def write_attempt_manifest(
        *,
        merge_metadata: dict[str, object] | None = None,
        pc_time_metadata: dict[str, object] | None = None,
        merge_status: str = "NOT_RUN",
        pc_time_status: str = "NOT_RUN",
        overall_status: str = "FAIL",
        warnings: list[dict[str, object]] | None = None,
    ) -> Path:
        payload = {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "run_id": run_id,
            "algorithm_versions": {
                "pipeline": PIPELINE_ALGORITHM_VERSION,
                "sync": SYNC_ALGORITHM_VERSION,
                "pc_time": "wild_preprocess.pc_time.v4",
            },
            "sync_status": result.outputs["sync_status"],
            "merge_status": merge_status,
            "pc_time_status": pc_time_status,
            "overall_status": overall_status,
            "master_index": master_index + 1,
            "device_order": input_provenance,
            "options": {"sync": asdict(options), "pc_time": asdict(pc_time_options)},
            "write_event_files": write_event_files,
            "sync": _sync_metadata(result),
            "merge": merge_metadata,
            "pc_time": pc_time_metadata or {"status": "not_run"},
            "expected_output_bytes": {},
            "performance": performance.payload(),
            "warnings": warnings if warnings is not None else _manifest_warnings(pairs),
            "managed_files": [
                "wild_preprocess_run.json",
                *sorted(Path(pair.figure_file).name for pair in pairs if pair.figure_file),
            ],
        }
        return _write_manifest(attempt_folder / "wild_preprocess_run.json", payload)

    if result.status == "FAIL":
        result.outputs["run_manifest"] = str(write_attempt_manifest())
        result.outputs["attempt_folder"] = str(attempt_folder)
        return result
    if not merge:
        result.outputs["run_manifest"] = str(
            write_attempt_manifest(overall_status="MERGE_ONLY")
        )
        result.outputs["attempt_folder"] = str(attempt_folder)
        return result

    pair_models = {pair.slave_index - 1: pair.model for pair in pairs}
    figure_names = {Path(pair.figure_file).name for pair in pairs}
    extra_managed_names = {
        "wild_multilogger_session_inspection.png",
        "wild_preprocess_run.json",
    } | figure_names
    if native_pc_time:
        extra_managed_names.update({"pc_time.dat", "pc_time_fit_summary.png"})
    component = {"merge_status": "NOT_RUN", "pc_time_status": "NOT_RUN", "overall_status": "FAIL"}

    def stage_callback(staging: Path, staged_outputs: dict[str, str]) -> None:
        for pair in pairs:
            source = Path(pair.figure_file)
            shutil.copy2(source, staging / source.name)

        merge_info = json.loads(
            Path(staged_outputs["_merge_internal"]).read_text(encoding="utf-8")
        )
        terminal_intervals = [
            interval
            for support in result.device_terminal_support
            if support.supported_canonical_end_sample
            < int(merge_info["common_end_master_sample"]) + 1
            if (
                interval := terminal_support_to_interval(
                    support,
                    canonical_end_sample=int(merge_info["common_end_master_sample"]) + 1,
                )
            ) is not None
        ]
        result.classified_intervals = list(
            merge_compatible_intervals([*result.classified_intervals, *terminal_intervals])
        )
        merge_info["classified_intervals"] = [
            interval.to_dict() for interval in result.classified_intervals
        ]
        merge_info["classified_interval_summary"] = _classified_interval_summary(
            result.classified_intervals,
            canonical_start_sample=int(merge_info["common_start_master_sample"]),
            n_samples=int(merge_info["n_samples"]),
            device_count=len(recordings),
        )
        merge_info["integrity_duplication_scan"] = {
            "enabled": integrity_duplication_scan,
            "options": asdict(RawAuditOptions(max_parallel_workers=1)),
            "later_occurrence_policy": (
                "zero-fill the exact later destination with medium confidence; "
                "do not include unmatched episode-envelope bridges"
            ),
            "devices": [
                {
                    "device_index": device_index + 1,
                    "device_name": recordings[device_index].device_name,
                    **{
                        key: value
                        for key, value in duplication_scans.get(device_index, {}).items()
                        if key != "episodes"
                    },
                }
                for device_index in range(len(recordings))
            ],
        }
        merge_info["probe_indices"] = probes
        merge_info["recording_start_anchors"] = [
            item["recording_start_anchor"] for item in input_provenance
        ]
        postmerge = None
        correlation_postmerge = None
        postmerge_exclusions = []
        postmerge_warning_messages: list[str] = []
        postmerge_exclusion_rounds = 0
        performance.begin("postmerge_validation")
        if validate_postmerge:
            def validate_current_stage(
                *, structural_only: bool = False
            ) -> PostMergeValidationResult:
                return validate_segment_staged_merge(
                    Path(staged_outputs["amplifier"]),
                    recordings,
                    master_index,
                    device_segments=result.device_sync_segments,
                    validity_path=Path(staged_outputs["validity"]),
                    canonical_start_sample=int(merge_info["common_start_master_sample"]),
                    n_output_samples=int(merge_info["n_samples"]),
                    window_seconds=10.0,
                    max_lag_samples=options.tracking_max_lag_samples,
                    max_allowed_abs_lag_samples=4,
                    min_peak_correlation=options.min_peak_correlation,
                    peak_exclusion_samples=options.peak_exclusion_samples,
                    structural_only=structural_only,
                )

            postmerge = validate_current_stage()
            correlation_postmerge = postmerge
            if postmerge.status == "WARN":
                postmerge_warning_messages.append(postmerge.message)
                proposed = list(
                    postmerge_exclusion_intervals(
                        postmerge,
                        canonical_start_sample=int(merge_info["common_start_master_sample"]),
                        device_count=len(recordings),
                    )
                )
                if proposed:
                    merge_info["validity_summary"] = apply_staged_zero_fill(
                        Path(staged_outputs["amplifier"]),
                        Path(staged_outputs["validity"]),
                        recordings,
                        master_index,
                        canonical_start_sample=int(merge_info["common_start_master_sample"]),
                        n_output_samples=int(merge_info["n_samples"]),
                        intervals=proposed,
                    )
                    postmerge_exclusions.extend(proposed)
                    postmerge_exclusion_rounds = 1
                    result.classified_intervals = list(
                        merge_compatible_intervals(
                            [*result.classified_intervals, *proposed]
                        )
                    )
                    merge_info["classified_intervals"] = [
                        interval.to_dict() for interval in result.classified_intervals
                    ]
                    merge_info["classified_interval_summary"] = _classified_interval_summary(
                        result.classified_intervals,
                        canonical_start_sample=int(merge_info["common_start_master_sample"]),
                        n_samples=int(merge_info["n_samples"]),
                        device_count=len(recordings),
                    )
                    # The correction is bounded to one patch, but the patched
                    # artifact must still pass the structural validity/zero
                    # contract before publication.  A remaining local WARN is
                    # publishable only when it has no further actionable
                    # exclusion.  Otherwise some failed samples would remain
                    # claimed valid after the single permitted correction.
                    postmerge = validate_current_stage(structural_only=True)
                    remaining_exclusions = postmerge_exclusion_intervals(
                        postmerge,
                        canonical_start_sample=int(merge_info["common_start_master_sample"]),
                        device_count=len(recordings),
                    )

                    def covered_by_applied_exclusions(candidate) -> bool:
                        for device_index in candidate.affected_device_indices:
                            cursor = candidate.canonical_start_sample
                            for applied in sorted(
                                postmerge_exclusions,
                                key=lambda interval: interval.canonical_start_sample,
                            ):
                                if device_index not in applied.affected_device_indices:
                                    continue
                                if applied.canonical_end_sample <= cursor:
                                    continue
                                if applied.canonical_start_sample > cursor:
                                    break
                                cursor = max(cursor, applied.canonical_end_sample)
                                if cursor >= candidate.canonical_end_sample:
                                    break
                            if cursor < candidate.canonical_end_sample:
                                return False
                        return True

                    uncovered_exclusions = tuple(
                        interval
                        for interval in remaining_exclusions
                        if not covered_by_applied_exclusions(interval)
                    )
                    if uncovered_exclusions:
                        postmerge = replace(
                            postmerge,
                            status="FAIL",
                            message=(
                                "the single bounded post-merge correction did not establish "
                                "a valid published mapping; final validation still recommends "
                                f"{len(uncovered_exclusions)} previously uncovered exclusion "
                                f"interval(s): "
                                f"{postmerge.message}"
                            ),
                        )
        performance.end("postmerge_validation")
        if postmerge is None:
            serialized_postmerge = {
                "status": "NOT_RUN",
                "message": "disabled for compatibility caller",
            }
        elif postmerge.publishable:
            evidence_postmerge = correlation_postmerge or postmerge
            serialized_postmerge = evidence_postmerge.to_dict()
            # Validation read the private staged file. On commit that same
            # artifact is moved byte-for-byte to this canonical destination.
            serialized_postmerge["amplifier_path"] = str(output_folder / "amplifier.dat")
            serialized_postmerge["applied_exclusion_intervals"] = [
                interval.to_dict() for interval in postmerge_exclusions
            ]
            serialized_postmerge["localized_exclusion_applied"] = bool(postmerge_exclusions)
        else:
            serialized_postmerge = postmerge.to_dict()
            if correlation_postmerge is not None and correlation_postmerge is not postmerge:
                serialized_postmerge["initial_correlation_validation"] = (
                    correlation_postmerge.to_dict()
                )
            serialized_postmerge["validated_staging_amplifier_path"] = serialized_postmerge.pop(
                "amplifier_path"
            )
            serialized_postmerge["staged_artifact_retained"] = False
        if postmerge is not None:
            serialized_postmerge["localized_exclusion_rounds"] = postmerge_exclusion_rounds
            serialized_postmerge["max_localized_exclusion_rounds"] = 1
            serialized_postmerge["initial_warning_messages"] = postmerge_warning_messages
            serialized_postmerge["final_revalidation_status"] = postmerge.status
            serialized_postmerge["final_revalidation_message"] = postmerge.message
            if postmerge_exclusions and postmerge.publishable:
                serialized_postmerge["status"] = "WARN"
                if postmerge.status == "OK":
                    serialized_postmerge["message"] = (
                        f"localized post-merge exclusions were applied once and the patched "
                        f"artifact revalidated successfully; {postmerge.message}"
                    )
                else:
                    serialized_postmerge["message"] = (
                        f"localized post-merge exclusions were applied once; final "
                        f"revalidation remains {postmerge.status}; {postmerge.message}"
                    )
        merge_info["postmerge_validation"] = serialized_postmerge
        if postmerge is not None and not postmerge.publishable:
            result.status = "FAIL"
            result.outputs["run_manifest"] = str(
                write_attempt_manifest(
                    merge_metadata=merge_info,
                    merge_status="FAIL",
                    overall_status="FAIL",
                )
            )
            raise _StagedMergeRejected(postmerge)

        for pair in pairs:
            pair.figure_file = str(output_folder / Path(pair.figure_file).name)

        component["merge_status"] = (
            "WARN" if postmerge_exclusions or result.status == "WARN" else "OK"
        )
        component["pc_time_status"] = "NOT_RUN"
        component["overall_status"] = "MERGE_ONLY"
        inspection_pc_time: object | None = None
        pc_time_metadata: dict[str, object] = {"status": "not_run"}
        pc_time_warning_message = ""
        if native_pc_time:
            performance.begin("pc_time_generation")
            master = recordings[master_index]
            staged_pc_time = staging / "pc_time.dat"
            try:
                indices, packed, packed_diagnostics = collect_packed_updates(
                    master.analog_file,
                    CE64_RAW_MISC_LAYOUT,
                    return_diagnostics=True,
                )
                anchor_ms, anchor_source = resolve_recording_start_ms(
                    master.folder, explicit_recording_start_ms=recording_start_ms
                )
                pc_fit = fit_gap_aware_pc_time_model(
                    indices,
                    packed,
                    master.fs,
                    anchor_ms,
                    device_gaps=(),
                    master_device_index=master_index + 1,
                )
                pc_model = pc_fit.model
                inspection_pc_time = pc_model
                pc_validation = validate_canonical_pc_time_interval(
                    pc_fit,
                    sample_rate_hz=master.fs,
                    canonical_start_sample=int(merge_info["common_start_master_sample"]),
                    n_samples=int(merge_info["n_samples"]),
                    options=pc_time_options,
                )
                payload = pc_time_qc_payload(
                    pc_model,
                    pc_validation,
                    run_id=run_id,
                    common_start_master_sample=int(merge_info["common_start_master_sample"]),
                    n_samples=int(merge_info["n_samples"]),
                    sample_rate_hz=master.fs,
                    anchor_source=anchor_source,
                    layout_name=CE64_RAW_MISC_LAYOUT.name,
                )
                payload.update(
                    {
                        "merge_run_id": run_id,
                        "status": pc_validation.status.lower(),
                        "aligned_to_merge": pc_validation.status == "OK",
                        "raw_update_count": packed_diagnostics.raw_candidate_run_count,
                        "accepted_update_count": packed_diagnostics.accepted_update_count,
                        "rejected_unstable_update_count": (
                            packed_diagnostics.rejected_unstable_run_count
                        ),
                        "packed_update_decode": packed_diagnostics.to_dict(),
                        "canonicalized_update_count": int(pc_fit.canonical_update_indices.size),
                        "neural_master_gap_count": sum(
                            gap.device_index == master_index + 1
                            for gap in result.device_gaps
                        ),
                        "analog_master_gap_count": 0,
                        "analog_gap_policy": (
                            "neural gaps are not applied to analog/PC time without "
                            "independent analog-clock evidence"
                        ),
                    }
                )
                pc_time_metadata = payload
                write_pc_time_summary_png(
                    staging / "pc_time_fit_summary.png",
                    pc_model,
                    pc_validation,
                    common_start_master_sample=int(merge_info["common_start_master_sample"]),
                    n_samples=int(merge_info["n_samples"]),
                    sample_rate_hz=master.fs,
                )
                if pc_validation.status == "OK":
                    write_canonical_interval_pc_time(
                        staging / "pc_time.dat",
                        pc_fit,
                        sample_rate_hz=master.fs,
                        canonical_start_sample=int(merge_info["common_start_master_sample"]),
                        n_samples=int(merge_info["n_samples"]),
                    )
                    component["pc_time_status"] = "OK"
                    component["overall_status"] = "COMPLETE"
                else:
                    staged_pc_time.unlink(missing_ok=True)
                    component["pc_time_status"] = "WARN"
                    pc_time_warning_message = pc_validation.message
            except Exception as error:
                staged_pc_time.unlink(missing_ok=True)
                component["pc_time_status"] = "WARN"
                pc_time_warning_message = str(error)
                payload = {
                    "algorithm": "wild_preprocess.pc_time.v4",
                    "run_id": run_id,
                    "merge_run_id": run_id,
                    "status": "warning",
                    "aligned_to_merge": False,
                    "common_start_master_sample": int(merge_info["common_start_master_sample"]),
                    "n_samples": int(merge_info["n_samples"]),
                    "error": str(error),
                }
                inspection_pc_time = payload
                pc_time_metadata = payload
                write_pc_time_warning_png(
                    staging / "pc_time_fit_summary.png",
                    message=str(error),
                    common_start_master_sample=int(merge_info["common_start_master_sample"]),
                    n_samples=int(merge_info["n_samples"]),
                    sample_rate_hz=master.fs,
                )
            performance.end(
                "pc_time_generation",
                bytes_written=(
                    staged_pc_time.stat().st_size if staged_pc_time.is_file() else 0
                ),
            )

        result.outputs.update(component)
        validity_order = [master_index, *(index for index in range(len(recordings)) if index != master_index)]
        device_labels = [
            (
                f"master {recordings[device_index].device_name}"
                if validity_channel == 0
                else f"slave {validity_channel} {recordings[device_index].device_name}"
            )
            for validity_channel, device_index in enumerate(validity_order)
        ]
        validity_channel_by_device = {
            device_index + 1: validity_channel
            for validity_channel, device_index in enumerate(validity_order)
        }
        inspection_path = staging / "wild_multilogger_session_inspection.png"
        performance.begin("inspection_figure_generation")
        try:
            write_session_inspection_png(
                inspection_path,
                sample_rate_hz=recordings[master_index].fs,
                pairs=result.pairs,
                valid_samples_path=Path(staged_outputs["validity"]),
                n_canonical_samples=int(merge_info["n_samples"]),
                canonical_start_master_sample=int(merge_info["common_start_master_sample"]),
                device_count=len(recordings),
                device_labels=device_labels,
                reason_intervals=[
                    {
                        "canonical_start_sample": interval.canonical_start_sample
                        - int(merge_info["common_start_master_sample"]),
                        "canonical_end_sample": interval.canonical_end_sample
                        - int(merge_info["common_start_master_sample"]),
                        "reason": interval.kind,
                        "device_indices": tuple(
                            validity_channel_by_device[index]
                            for index in interval.affected_device_indices
                        ),
                    }
                    for interval in result.classified_intervals
                ],
                join_events=(
                    [
                        {
                            "time_sec": (
                                measurement.window_start_sample
                            )
                            / recordings[master_index].fs,
                            "residual_samples": measurement.lag_samples,
                            "status": "pass" if measurement.passed else "fail",
                            "label": (
                                f"{measurement.position} slave "
                                f"{measurement.slave_device_index}"
                            ),
                        }
                        for measurement in (
                            correlation_postmerge or postmerge
                        ).measurements
                        if measurement.lag_samples is not None
                        and measurement.position.startswith(("gap", "boundary"))
                    ]
                    if postmerge is not None
                    else []
                ),
                pc_time=inspection_pc_time,
                pc_time_summary=pc_time_metadata,
                status=(
                    "WARN" if component["merge_status"] == "WARN" else component["overall_status"]
                ),
                residual_tolerance_samples=options.max_model_residual_samples,
                master_gaps=[
                    gap for gap in result.device_gaps if gap.device_index == master_index + 1
                ],
                segment_summary=[
                    {
                        **segment.to_dict(),
                        "validity_channel": validity_channel_by_device[segment.device_index],
                    }
                    for segment in result.device_sync_segments
                ],
                performance_summary=performance.payload(),
            )
        except BaseException:
            performance.end("inspection_figure_generation", status="failed")
            raise
        else:
            performance.end(
                "inspection_figure_generation",
                bytes_written=inspection_path.stat().st_size,
            )
        result.outputs["inspection_figure"] = str(
            output_folder / "wild_multilogger_session_inspection.png"
        )
        managed_files = sorted(
            path.name
            for path in staging.iterdir()
            if path.is_file() and path.name != ".wild_internal_merge.json"
        )
        # Include the manifest itself because it controls removal of all
        # artifacts owned by this generation on the next overwrite.
        managed_files.append("wild_preprocess_run.json")
        expected_output_bytes = {
            "amplifier.dat": int(merge_info["n_samples"]) * int(merge_info["n_channels"]) * 2,
            "analogin.dat": int(merge_info["analog_samples"]) * int(merge_info["analog_channels"]) * 2,
            "time.dat": int(merge_info["n_samples"]) * 4,
            "valid_samples.dat": int(merge_info["n_samples"]) * len(recordings),
        }
        if component["pc_time_status"] == "OK":
            expected_output_bytes["pc_time.dat"] = int(merge_info["n_samples"]) * 4
        actual_output_bytes = {
            path.name: path.stat().st_size
            for path in staging.iterdir()
            if path.is_file() and path.name != ".wild_internal_merge.json"
        }
        performance.set_output_bytes(
            expected=expected_output_bytes,
            actual=actual_output_bytes,
        )
        warnings = _manifest_warnings(pairs)
        if postmerge_exclusions:
            warnings.append(
                {
                    "component": "postmerge_validation",
                    "status": "WARN",
                    "message": serialized_postmerge["message"],
                    "localized_exclusion_count": len(postmerge_exclusions),
                    "localized_exclusion_rounds": postmerge_exclusion_rounds,
                }
            )
        if component["pc_time_status"] == "WARN":
            warnings.append(
                {
                    "component": "pc_time",
                    "status": "WARN",
                    "message": pc_time_warning_message,
                }
            )
        if result.device_gaps:
            summaries = gap_summary(
                result.device_gaps,
                device_count=len(recordings),
                canonical_samples=int(merge_info["n_samples"]),
            )
            for item in summaries:
                if not item["gap_count"]:
                    continue
                entries = [
                    entry
                    for entry in merge_info["device_gaps"]
                    if entry["device_index"] == item["device_index"]
                ]
                filled = sum(entry["action"] == "zero_filled_with_guard" for entry in entries)
                warnings.append(
                    {
                        "device_index": item["device_index"],
                        "status": "GAPS_HANDLED",
                        "message": (
                            f"{item['gap_count']} detected gap(s), {item['missing_samples']} samples "
                            f"({100.0 * float(item['missing_fraction']):.4f}%); "
                            f"{filled} intersect output, {len(entries) - filled} cropped/outside"
                        ),
                    }
                )
        _write_manifest(
            staging / "wild_preprocess_run.json",
            {
                "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
                "run_id": run_id,
                "algorithm_versions": {
                    "pipeline": PIPELINE_ALGORITHM_VERSION,
                    "sync": SYNC_ALGORITHM_VERSION,
                    "pc_time": "wild_preprocess.pc_time.v4",
                },
                "sync_status": result.outputs["sync_status"],
                "merge_status": component["merge_status"],
                "pc_time_status": component["pc_time_status"],
                "overall_status": component["overall_status"],
                "master_index": master_index + 1,
                "device_order": input_provenance,
                "options": {"sync": asdict(options), "pc_time": asdict(pc_time_options)},
                "write_event_files": write_event_files,
                "sync": _sync_metadata(result),
                "merge": merge_info,
                "pc_time": pc_time_metadata,
                "expected_output_bytes": expected_output_bytes,
                "performance": performance.payload(),
                "warnings": warnings,
                "managed_files": managed_files,
            },
        )

    published = False

    def merge_timing(name: str, event: str, bytes_read: int, bytes_written: int) -> None:
        if event == "start":
            performance.begin(name)
        else:
            performance.end(
                name,
                bytes_read=bytes_read,
                bytes_written=bytes_written,
                status="failed" if event == "failed" else "complete",
            )

    try:
        result.outputs.update(
            merge_recordings(
                recordings,
                master_index,
                pair_models,
                output_folder,
                device_gaps=result.device_gaps,
                minimum_common_start=canonicalize_master_sample(
                    max(
                        (pair.validated_start_master_sample for pair in pairs),
                        default=0,
                    ),
                    result.device_gaps,
                    master_device_index=master_index + 1,
                ),
                classified_intervals=result.classified_intervals,
                device_source_steps=result.device_source_steps,
                device_terminal_support=result.device_terminal_support,
                device_sync_segments=result.device_sync_segments,
                preserve_device_tails=True,
                write_event_files=write_event_files,
                chunk_seconds=options.chunk_seconds,
                overwrite=overwrite,
                progress=progress,
                run_id=run_id,
                stage_callback=stage_callback,
                additional_managed_names=extra_managed_names,
                timing=merge_timing,
            )
        )
        result.outputs.update(
            {
                "run_manifest": str(output_folder / "wild_preprocess_run.json"),
            }
        )
        result.outputs.update(component)
        published = True
        return result
    except _StagedMergeRejected as error:
        result.status = "FAIL"
        result.outputs.update(
            {
                "sync_status": "OK",
                "merge_status": "FAIL",
                "pc_time_status": "NOT_RUN",
                "overall_status": "FAIL",
                "attempt_folder": str(attempt_folder),
                "run_manifest": str(attempt_folder / "wild_preprocess_run.json"),
            }
        )
        return result
    finally:
        # Successful outputs are now canonical; failed attempts remain for
        # review and never replace a prior successful generation.
        if published:
            shutil.rmtree(attempt_folder, ignore_errors=True)
