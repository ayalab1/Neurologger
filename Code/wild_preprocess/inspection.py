"""One-page post-hoc inspection figure for a corrected multi-device session."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from .binary_io import close_memmap
from .models import DeviceGap, DeviceSyncSegment, SyncPairResult
from .pc_time.infer import PcTimeModel


@dataclass(frozen=True)
class InspectionInterval:
    start_sample: int
    end_sample: int
    reason: str = "invalid"
    device_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class JoinResidual:
    time_sec: float
    residual_samples: float
    status: str = "accepted"
    label: str = ""


@dataclass(frozen=True)
class CameraCoverage:
    start_sec: float
    end_sec: float
    valid: bool = True
    label: str = "camera"


_REASON_COLORS = {
    "duplication": "#9467bd",
    "duplicate_destination": "#9467bd",
    "overwrite": "#9467bd",
    "missing": "#d62728",
    "uncertain": "#ff7f0e",
    "unresolved_boundary": "#ff7f0e",
    "unsupported": "#7f7f7f",
    "terminal_unsupported": "#7f7f7f",
    "postmerge_unverified": "#c44e52",
    "invalid": "#7f7f7f",
}
_UNVERIFIED_MAPPING_COLOR = "#9aa0a6"
_REASON_LABELS = {
    "duplication": "duplicate",
    "duplicate_destination": "duplicate",
    "overwrite": "duplicate",
    "missing": "missing",
    "uncertain": "boundary uncertain",
    "unresolved_boundary": "boundary uncertain",
    "unsupported": "unsupported tail",
    "terminal_unsupported": "unsupported tail",
    "postmerge_unverified": "sync unverified",
    "invalid": "invalid mapping",
}


def _value(item: object, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _as_float_array(values: Any) -> np.ndarray:
    if values is None:
        return np.empty(0, dtype=float)
    return np.asarray(values, dtype=float).reshape(-1)


def _as_bool_array(values: Any, size: int) -> np.ndarray:
    if values is None:
        return np.ones(size, dtype=bool)
    result = np.asarray(values, dtype=bool).reshape(-1)
    if result.size != size:
        raise ValueError("inspection boolean vector must match its observation vector")
    return result


def _pair_observations(pair: SyncPairResult | Mapping[str, Any]) -> list[object]:
    return list(_value(pair, "observations", ()) or ())


def _model_offset(pair: SyncPairResult | Mapping[str, Any], times: np.ndarray) -> np.ndarray:
    model = _value(pair, "model", None)
    if model is None:
        return np.full(times.shape, np.nan)
    offset_at = getattr(model, "offset_at_seconds", None)
    if callable(offset_at):
        return np.asarray([offset_at(float(time_sec)) for time_sec in times], dtype=float)
    intercept = float(_value(model, "intercept_samples", 0.0))
    slope = float(_value(model, "slope_samples_per_second", 0.0))
    result = intercept + slope * times
    for step in _value(model, "offset_steps", ()) or ():
        step_time = float(_value(step, "time_sec", float("inf")))
        step_size = float(_value(step, "offset_step_samples", 0.0))
        result = result + step_size * (times >= step_time)
    return result


def _read_validity(
    valid_samples: np.ndarray | None,
    valid_samples_path: str | Path | None,
    n_canonical_samples: int | None,
    device_count: int | None,
    *,
    artifact_name: str = "valid_samples",
) -> tuple[np.ndarray | None, int | None, int | None]:
    if valid_samples is not None and valid_samples_path is not None:
        raise ValueError(
            f"pass {artifact_name} or {artifact_name}_path, not both"
        )
    if valid_samples is not None:
        mask = np.asarray(valid_samples)
        if mask.ndim != 2:
            raise ValueError(
                f"{artifact_name} must have shape (samples, devices)"
            )
        if mask.dtype.kind not in "buif":
            raise ValueError(f"{artifact_name} must be a numeric 0/1 array")
        samples, devices = mask.shape
        if n_canonical_samples is not None and n_canonical_samples != samples:
            raise ValueError(
                f"n_canonical_samples does not match {artifact_name}"
            )
        if device_count is not None and device_count != devices:
            raise ValueError(f"device_count does not match {artifact_name}")
        return mask.astype(bool, copy=False), samples, devices
    if valid_samples_path is None:
        return None, n_canonical_samples, device_count
    if device_count is None or device_count <= 0:
        raise ValueError("device_count is required with valid_samples_path")
    path = Path(valid_samples_path)
    byte_count = path.stat().st_size
    if byte_count % device_count:
        raise ValueError(
            f"{artifact_name}.dat byte length is not divisible by device_count"
        )
    samples = byte_count // device_count
    if n_canonical_samples is not None and n_canonical_samples != samples:
        raise ValueError(
            f"n_canonical_samples does not match {artifact_name}.dat"
        )
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(samples, device_count)), samples, device_count


def _conservative_bins(mask: np.ndarray, max_bins: int) -> np.ndarray:
    samples, devices = mask.shape
    if samples <= max_bins:
        return np.asarray(mask, dtype=bool)
    edges = np.linspace(0, samples, max_bins + 1, dtype=np.int64)
    binned = np.empty((max_bins, devices), dtype=bool)
    for index, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        binned[index] = np.all(mask[start:stop], axis=0)
    return binned


def _validity_fractions(mask: np.ndarray) -> tuple[np.ndarray, float]:
    samples, devices = mask.shape
    counts = np.zeros(devices, dtype=np.int64)
    common = 0
    for start in range(0, samples, 1_000_000):
        block = np.asarray(mask[start : min(samples, start + 1_000_000)]) != 0
        counts += np.count_nonzero(block, axis=0)
        common += int(np.count_nonzero(np.all(block, axis=1)))
    return counts / max(1, samples), common / max(1, samples)


def _alignment_pair_fractions(mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Return master/slave and all-device aligned fractions."""

    samples, devices = mask.shape
    counts = np.zeros(max(0, devices - 1), dtype=np.int64)
    common = 0
    for start in range(0, samples, 1_000_000):
        block = np.asarray(mask[start : min(samples, start + 1_000_000)]) != 0
        if devices > 1:
            counts += np.count_nonzero(block[:, [0]] & block[:, 1:], axis=0)
        common += int(np.count_nonzero(np.all(block, axis=1)))
    return counts / max(1, samples), common / max(1, samples)


def _false_runs(values: np.ndarray) -> list[tuple[int, int]]:
    invalid = ~np.asarray(values, dtype=bool)
    transitions = np.diff(np.concatenate(([False], invalid, [False])).astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    return _false_runs(~np.asarray(values, dtype=bool))


def _binned_median(x: np.ndarray, y: np.ndarray, max_bins: int = 160) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[finite], dtype=float)
    y = np.asarray(y[finite], dtype=float)
    if x.size < 2:
        return x, y
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    bin_count = min(max_bins, max(2, x.size // 4))
    edges = np.linspace(0, x.size, bin_count + 1, dtype=np.int64)
    centers: list[float] = []
    medians: list[float] = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end > start:
            centers.append(float(np.median(x[start:end])))
            medians.append(float(np.median(y[start:end])))
    return np.asarray(centers), np.asarray(medians)


def _interval_values(item: InspectionInterval | Mapping[str, Any]) -> tuple[int, int, str, tuple[int, ...]]:
    start = int(_value(item, "start_sample", _value(item, "canonical_start_sample", 0)))
    end = _value(item, "end_sample", _value(item, "canonical_end_sample", None))
    if end is None:
        end = start + int(_value(item, "missing_samples", 0))
    devices = _value(item, "device_indices", ()) or ()
    if isinstance(devices, int):
        devices = (devices,)
    return start, int(end), str(_value(item, "reason", "invalid")), tuple(int(value) for value in devices)


def _join_values(item: JoinResidual | Mapping[str, Any], sample_rate_hz: float) -> tuple[float, float, str, str]:
    time_sec = _value(item, "time_sec", None)
    if time_sec is None:
        time_sec = float(_value(item, "canonical_sample", 0)) / sample_rate_hz
    return (
        float(time_sec),
        float(_value(item, "residual_samples", float("nan"))),
        str(_value(item, "status", "accepted")),
        str(_value(item, "label", "")),
    )


def _camera_values(item: CameraCoverage | Mapping[str, Any]) -> tuple[float, float, bool, str]:
    return (
        float(_value(item, "start_sec", 0.0)),
        float(_value(item, "end_sec", _value(item, "start_sec", 0.0))),
        bool(_value(item, "valid", True)),
        str(_value(item, "label", "camera")),
    )


def _segment_report_lines(
    segment_summary: Sequence[DeviceSyncSegment | Mapping[str, Any]] | Mapping[str, Any] | None,
    *,
    device_valid_fractions: np.ndarray,
    device_labels: Sequence[str],
) -> list[str]:
    """Return compact, defensive segment/reacquisition text for panel B.

    The inspection figure intentionally accepts either the in-memory segment
    objects or their manifest serialization.  This keeps the figure useful in
    both the pipeline and a post-hoc manifest reader without making the
    segment report another canonical output.
    """

    if segment_summary is None:
        return []
    if isinstance(segment_summary, Mapping):
        values = segment_summary.get(
            "device_sync_segments", segment_summary.get("segments", ())
        )
    else:
        values = segment_summary
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []

    per_device: dict[int, list[object]] = {}
    for segment in values:
        try:
            device_index = int(_value(segment, "device_index"))
        except (TypeError, ValueError):
            continue
        if device_index < 1:
            continue
        per_device.setdefault(device_index, []).append(segment)

    lines: list[str] = []
    for channel, label in enumerate(device_labels):
        # Segment device indices use selected input order.  The pipeline passes
        # its reordered validity-channel summary, so callers may provide the
        # optional validity_channel key when master order differs from input.
        segments = [
            segment
            for entries in per_device.values()
            for segment in entries
            if int(_value(segment, "validity_channel", -1)) == channel
        ]
        if not segments:
            segments = per_device.get(channel + 1, [])
        publishable = sum(bool(_value(segment, "publishable", False)) for segment in segments)
        reacquired = sum(
            "reacquisition" in str(_value(segment, "start_transition", "")).lower()
            for segment in segments
        )
        valid_fraction = float(device_valid_fractions[channel])
        invalid_text = (
            f"invalid {100.0 * (1.0 - valid_fraction):.3f}%"
            if np.isfinite(valid_fraction)
            else "validity unavailable"
        )
        lines.append(
            f"{label}: {publishable}/{len(segments)} verified segment(s), "
            f"reacquired {reacquired}, {invalid_text}"
        )
    return lines


def _performance_report_line(performance_summary: Mapping[str, Any] | None) -> str:
    """Format manifest-only timing metrics without assuming every stage exists."""

    if not isinstance(performance_summary, Mapping):
        return ""
    pieces: list[str] = []
    total = _value(performance_summary, "total_wall_seconds", None)
    try:
        if total is not None:
            pieces.append(f"total {float(total):.2f}s")
    except (TypeError, ValueError):
        pass
    stages = _value(performance_summary, "stages", ())
    if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
        return "timing: " + " | ".join(pieces) if pieces else ""
    preferred = (
        ("raw_evidence_feature_scan", "evidence"),
        ("coarse_correlation", "coarse"),
        ("full_rate_refinement", "refine"),
        ("attribution_segment_construction", "segments"),
        ("ephys_merge", "merge"),
        ("postmerge_validation", "postmerge"),
    )
    by_name = {
        str(_value(stage, "name", "")): stage
        for stage in stages
        if isinstance(stage, Mapping)
    }
    for stage_name, label in preferred:
        stage = by_name.get(stage_name)
        if stage is None:
            continue
        try:
            pieces.append(f"{label} {float(_value(stage, 'wall_seconds', 0.0)):.2f}s")
        except (TypeError, ValueError):
            continue
    return "timing: " + " | ".join(pieces) if pieces else ""


def _compact_segment_performance_lines(
    segment_summary: Sequence[DeviceSyncSegment | Mapping[str, Any]] | Mapping[str, Any] | None,
    *,
    device_valid_fractions: np.ndarray,
    performance_summary: Mapping[str, Any] | None,
) -> list[str]:
    """Keep panel-B reporting informative without covering the validity plot."""

    values: object = segment_summary
    if isinstance(values, Mapping):
        values = values.get("device_sync_segments", values.get("segments", ()))
    segments = (
        list(values)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes))
        else []
    )
    lines: list[str] = []
    if segments:
        publishable = sum(bool(_value(segment, "publishable", False)) for segment in segments)
        reacquired = sum(
            "reacquisition" in str(_value(segment, "start_transition", "")).lower()
            for segment in segments
        )
        fully_invalid = np.flatnonzero(np.isfinite(device_valid_fractions) & (device_valid_fractions == 0.0))
        segment_line = f"segments {publishable}/{len(segments)} verified; reacquired {reacquired}"
        if fully_invalid.size:
            segment_line += f"; fully invalid device(s) {', '.join(str(index + 1) for index in fully_invalid)}"
        lines.append(segment_line)
    timing = _performance_report_line(performance_summary)
    if timing:
        lines.append(timing)
    return lines[:2]


def _pc_time_values(pc_time: PcTimeModel | Mapping[str, Any] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if pc_time is None:
        return np.empty(0), np.empty(0), np.empty(0, dtype=bool), "PC-time unavailable"
    if isinstance(pc_time, PcTimeModel):
        return pc_time.device_ms / 1000.0, pc_time.residual_ms, pc_time.keep_mask, ""
    device_time = _as_float_array(_value(pc_time, "device_time_sec", _value(pc_time, "time_sec", None)))
    if not device_time.size:
        device_time = _as_float_array(_value(pc_time, "device_ms", None)) / 1000.0
    residual = _as_float_array(_value(pc_time, "residual_ms", None))
    if device_time.size and residual.size != device_time.size:
        raise ValueError("PC-time residual_ms must match device_time_sec")
    if device_time.size:
        return device_time, residual, _as_bool_array(_value(pc_time, "keep_mask", None), device_time.size), ""
    error = str(_value(pc_time, "error", "")).lower()
    if "packed pc-time update indices and values must be non-empty" in error:
        return (
            np.empty(0),
            np.empty(0),
            np.empty(0, dtype=bool),
            "PC-time unavailable: no packed clock updates",
        )
    return np.empty(0), np.empty(0), np.empty(0, dtype=bool), "PC-time unavailable"


def _inferred_sample_count(
    pairs: Sequence[SyncPairResult | Mapping[str, Any]],
    join_events: Sequence[JoinResidual | Mapping[str, Any]],
    camera_coverage: Sequence[CameraCoverage | Mapping[str, Any]],
    sample_rate_hz: float,
    canonical_start_master_sample: int,
) -> int:
    maximum = 0.0
    canonical_start_sec = canonical_start_master_sample / sample_rate_hz
    for pair in pairs:
        for observation in _pair_observations(pair):
            maximum = max(maximum, float(_value(observation, "center_time_sec", 0.0)) - canonical_start_sec)
    for event in join_events:
        maximum = max(maximum, _join_values(event, sample_rate_hz)[0])
    for coverage in camera_coverage:
        maximum = max(maximum, _camera_values(coverage)[1])
    return max(1, int(np.ceil(maximum * sample_rate_hz)))


def write_session_inspection_png(
    path: str | Path,
    *,
    sample_rate_hz: float,
    pairs: Sequence[SyncPairResult | Mapping[str, Any]] = (),
    valid_samples: np.ndarray | None = None,
    valid_samples_path: str | Path | None = None,
    alignment_quality: np.ndarray | None = None,
    alignment_quality_path: str | Path | None = None,
    n_canonical_samples: int | None = None,
    canonical_start_master_sample: int = 0,
    device_count: int | None = None,
    device_labels: Sequence[str] | None = None,
    reason_intervals: Sequence[InspectionInterval | Mapping[str, Any]] = (),
    join_events: Sequence[JoinResidual | Mapping[str, Any]] = (),
    pc_time: PcTimeModel | Mapping[str, Any] | None = None,
    pc_time_summary: Mapping[str, Any] | None = None,
    camera_coverage: Sequence[CameraCoverage | Mapping[str, Any]] = (),
    status: str | None = None,
    title: str = "Post-hoc multi-device sync inspection",
    residual_tolerance_samples: float | None = None,
    alignment_tolerance_samples: float | None = None,
    max_mask_bins: int = 4000,
    master_gaps: Sequence[DeviceGap | Mapping[str, Any]] = (),
    segment_summary: Sequence[DeviceSyncSegment | Mapping[str, Any]] | Mapping[str, Any] | None = None,
    performance_summary: Mapping[str, Any] | None = None,
) -> Path:
    """Write one decision-oriented QC PNG on the canonical sample axis."""

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if max_mask_bins <= 0:
        raise ValueError("max_mask_bins must be positive")
    if canonical_start_master_sample < 0:
        raise ValueError("canonical_start_master_sample must be non-negative")
    mask, samples, devices = _read_validity(
        valid_samples, valid_samples_path, n_canonical_samples, device_count
    )
    if samples is None:
        samples = _inferred_sample_count(
            pairs, join_events, camera_coverage, sample_rate_hz, canonical_start_master_sample
        )
    if devices is None:
        devices = max((int(_value(pair, "slave_index", 0)) for pair in pairs), default=0) + 1
    alignment_mask, alignment_samples, alignment_devices = _read_validity(
        alignment_quality,
        alignment_quality_path,
        samples,
        devices,
        artifact_name="alignment_quality",
    )
    if alignment_samples is not None and alignment_samples != samples:
        raise ValueError("alignment_quality must match valid_samples sample count")
    if alignment_devices is not None and alignment_devices != devices:
        raise ValueError("alignment_quality must match valid_samples device count")
    if mask is not None and alignment_mask is not None:
        for start in range(0, samples, 1_000_000):
            end = min(samples, start + 1_000_000)
            if np.any(
                (np.asarray(alignment_mask[start:end]) != 0)
                & (np.asarray(mask[start:end]) == 0)
            ):
                raise ValueError(
                    "alignment_quality cannot be true outside valid_samples"
                )
    if device_labels is None:
        labels = ["master"] + [f"slave {index}" for index in range(1, devices)]
    else:
        labels = list(device_labels)
        if len(labels) != devices:
            raise ValueError("device_labels must have one entry per validity channel")

    duration_sec = samples / sample_rate_hz
    canonical_start_sec = canonical_start_master_sample / sample_rate_hz
    sorted_pairs = sorted(pairs, key=lambda pair: int(_value(pair, "slave_index", 0)))
    sorted_intervals = sorted(
        (_interval_values(item) for item in reason_intervals),
        key=lambda item: (item[0], item[1], item[2]),
    )
    sorted_joins = sorted(
        (_join_values(item, sample_rate_hz) for item in join_events), key=lambda item: item[0]
    )
    sorted_camera = sorted(
        (_camera_values(item) for item in camera_coverage), key=lambda item: (item[0], item[1], item[3])
    )
    pc_time_sec, pc_residual_ms, pc_kept, pc_summary_text = _pc_time_values(pc_time)
    ordered_master_gaps = sorted(
        master_gaps, key=lambda gap: int(_value(gap, "canonical_start_sample", 0))
    )

    def display_times(raw_times_sec: np.ndarray) -> np.ndarray:
        finite = np.isfinite(raw_times_sec)
        raw_samples = np.zeros(raw_times_sec.shape, dtype=np.int64)
        raw_samples[finite] = np.rint(raw_times_sec[finite] * sample_rate_hz).astype(np.int64)
        canonical_samples = raw_samples.copy()
        inserted = 0
        for gap in ordered_master_gaps:
            canonical_start = int(_value(gap, "canonical_start_sample", 0))
            missing = int(_value(gap, "missing_samples", 0))
            raw_boundary = canonical_start - inserted
            canonical_samples += missing * (raw_samples >= raw_boundary)
            inserted += missing
        plotted = (canonical_samples - canonical_start_master_sample) / sample_rate_hz
        plotted[~finite] = np.nan
        return plotted

    figure, (residual_axis, validity_axis, alignment_axis, pc_axis) = plt.subplots(
        4,
        1,
        figsize=(17, 11.5),
        sharex=True,
        constrained_layout=False,
        gridspec_kw={"height_ratios": [2.2, 1.55, 1.25, 1.8]},
    )

    residual_values: list[float] = []
    pair_summaries: list[str] = []
    for color_index, pair in enumerate(sorted_pairs):
        observations = _pair_observations(pair)
        times = np.asarray([_value(item, "center_time_sec", np.nan) for item in observations], dtype=float)
        observed = np.asarray([_value(item, "observed_offset_samples", np.nan) for item in observations], dtype=float)
        residual = np.asarray([_value(item, "model_residual_samples", np.nan) for item in observations], dtype=float)
        missing = ~np.isfinite(residual) & np.isfinite(times) & np.isfinite(observed)
        if np.any(missing):
            residual[missing] = observed[missing] - _model_offset(pair, times[missing])
        accepted = np.asarray([bool(_value(item, "accepted", False)) for item in observations], dtype=bool)
        usable = np.isfinite(times) & np.isfinite(residual)
        pair_label = f"M-S{color_index + 1}"
        if np.any(usable & accepted):
            residual_axis.scatter(
                display_times(times[usable & accepted]),
                residual[usable & accepted],
                s=7,
                alpha=0.48,
                color=f"C{color_index}",
                label=f"{pair_label} fit",
            )
        if np.any(usable & ~accepted):
            residual_axis.scatter(
                display_times(times[usable & ~accepted]),
                residual[usable & ~accepted],
                s=22,
                marker="x",
                linewidths=0.9,
                color=f"C{color_index}",
                label=f"{pair_label} rejected",
            )
        residual_values.extend(float(value) for value in residual[usable])
        model = _value(pair, "model", None)
        pair_summaries.append(
            f"{pair_label}: offset {float(_value(model, 'intercept_samples', 0.0)):.2f}, "
            f"drift {float(_value(model, 'drift_ppm', 0.0)):.3g} ppm, "
            f"RMS {float(_value(model, 'residual_rms_samples', float('nan'))):.3g} samples"
        )
    boundary_labels_seen: set[str] = set()
    for time_sec, residual, event_status, _label in sorted_joins:
        passed = event_status.lower() in {"ok", "accepted", "pass"}
        boundary_label = "boundary aligned" if passed else "boundary > limit"
        residual_axis.scatter(
            [time_sec],
            [residual],
            color="#1677b3" if passed else "#c33",
            marker="D" if passed else "x",
            s=38,
            zorder=3,
            label=(
                boundary_label
                if boundary_label not in boundary_labels_seen
                else "_nolegend_"
            ),
        )
        boundary_labels_seen.add(boundary_label)
        residual_values.append(residual)
    residual_axis.axhline(0.0, color="black", linewidth=0.8, label="zero residual")
    finite_residual = np.asarray(residual_values, dtype=float)
    finite_residual = finite_residual[np.isfinite(finite_residual)]
    if finite_residual.size:
        half_range = max(1.0, float(np.percentile(np.abs(finite_residual), 99)) * 1.3)
        extreme = float(np.max(np.abs(finite_residual)))
        if extreme <= half_range * 2.5:
            half_range = max(half_range, extreme * 1.08)
        else:
            clipped_count = int(np.count_nonzero(np.abs(finite_residual) > half_range))
            pair_summaries.append(f"{clipped_count} residual point(s) outside y-range")
        residual_axis.set_ylim(-half_range, half_range)
    residual_axis.set_ylabel("sync residual (samples)")
    residual_title = "A. Master-slave residuals after correction"
    if residual_tolerance_samples is not None:
        residual_title += (
            f" | model limit ±{abs(float(residual_tolerance_samples)):g} samples"
        )
    residual_axis.set_title(residual_title)
    residual_handles, residual_labels = residual_axis.get_legend_handles_labels()
    if residual_handles:
        residual_axis.legend(
            residual_handles,
            residual_labels,
            loc="upper left",
            bbox_to_anchor=(1.005, 1.0),
            fontsize=8,
            title="\n".join(pair_summaries) if pair_summaries else None,
            title_fontsize=7.5,
        )
    elif not sorted_joins:
        residual_axis.text(0.5, 0.5, "Sync observations unavailable", ha="center", va="center", transform=residual_axis.transAxes)

    device_valid_fractions = np.full(devices, np.nan)
    common_valid_fraction = float("nan")
    segment_lines: list[str] = []
    if mask is None:
        validity_axis.text(0.5, 0.5, "valid_samples.dat unavailable", ha="center", va="center", transform=validity_axis.transAxes)
        validity_axis.set_yticks([])
    else:
        binned = _conservative_bins(mask, max_mask_bins)
        common_binned = np.all(binned, axis=1)
        device_valid_fractions, common_valid_fraction = _validity_fractions(mask)
        all_binned = np.column_stack((binned, common_binned))
        bin_duration = duration_sec / max(1, all_binned.shape[0])
        row_labels = [
            f"{label}   {100.0 * fraction:.3f}%"
            for label, fraction in zip(labels, device_valid_fractions)
        ] + [f"common data-valid   {100.0 * common_valid_fraction:.3f}%"]
        baseline_invalid_handle: object | None = None
        for row in range(devices + 1):
            validity_axis.hlines(row, 0.0, duration_sec, color="#2ca02c", linewidth=5.0, alpha=0.38)
            for start_bin, end_bin in _false_runs(all_binned[:, row]):
                bars = validity_axis.broken_barh(
                    [(start_bin * bin_duration, (end_bin - start_bin) * bin_duration)],
                    (row - 0.28, 0.56),
                    facecolors=_UNVERIFIED_MAPPING_COLOR,
                    edgecolors="none",
                )
                if baseline_invalid_handle is None:
                    baseline_invalid_handle = bars
        minimum_visible_width = duration_sec / 750.0
        reason_handles: dict[str, object] = {
            "data valid": Patch(facecolor="#2ca02c", alpha=0.38)
        }
        if baseline_invalid_handle is not None:
            reason_handles["unverified mapping"] = baseline_invalid_handle
        display_spans: dict[tuple[str, int], list[tuple[float, float]]] = {}
        for start, end, reason, affected in sorted_intervals:
            actual_start = max(0.0, start / sample_rate_hz)
            actual_end = min(duration_sec, end / sample_rate_hz)
            if actual_end <= actual_start:
                continue
            center = 0.5 * (actual_start + actual_end)
            visible_width = min(duration_sec, max(actual_end - actual_start, minimum_visible_width))
            visible_start = min(max(0.0, center - visible_width / 2.0), max(0.0, duration_sec - visible_width))
            rows = set(affected) if affected else set(range(devices))
            rows.add(devices)
            reason_key = reason.lower()
            for row in rows:
                if 0 <= row <= devices:
                    display_spans.setdefault((reason_key, row), []).append(
                        (visible_start, visible_start + visible_width)
                    )
        # Thousands of sample-scale integrity intervals must not become tens of
        # thousands of individual matplotlib artists.  Coalesce the already
        # display-expanded spans per reason/row; the exact intervals remain in
        # the manifest and validity DAT.
        for (reason_key, row), spans in display_spans.items():
            merged_spans: list[tuple[float, float]] = []
            for span_start, span_end in sorted(spans):
                if merged_spans and span_start <= merged_spans[-1][1]:
                    merged_spans[-1] = (
                        merged_spans[-1][0],
                        max(merged_spans[-1][1], span_end),
                    )
                else:
                    merged_spans.append((span_start, span_end))
            bars = validity_axis.broken_barh(
                [(start, end - start) for start, end in merged_spans],
                (row - 0.34, 0.68),
                facecolors=_REASON_COLORS.get(reason_key, "#7f7f7f"),
                edgecolors="none",
            )
            reason_handles.setdefault(_REASON_LABELS.get(reason_key, reason_key), bars)
        validity_axis.set_yticks(np.arange(devices + 1), row_labels)
        validity_axis.set_ylim(devices + 0.5, -0.5)
        segment_lines = _compact_segment_performance_lines(
            segment_summary,
            device_valid_fractions=device_valid_fractions,
            performance_summary=performance_summary,
        )
        validity_axis.legend(
            reason_handles.values(),
            reason_handles.keys(),
            loc="upper left",
            bbox_to_anchor=(1.005, 1.0),
            fontsize=8,
            title=(
                "\n".join(textwrap.fill(line, width=72) for line in segment_lines)
                if segment_lines
                else None
            ),
            title_fontsize=7.5,
        )
    validity_axis.set_ylabel("data validity")
    validity_axis.set_title(
        "B. Data validity only (short invalid intervals widened)"
    )

    common_alignment_fraction = float("nan")
    if mask is None or alignment_mask is None:
        alignment_axis.text(
            0.5,
            0.5,
            "alignment_quality.dat unavailable",
            ha="center",
            va="center",
            transform=alignment_axis.transAxes,
        )
        alignment_axis.set_yticks([])
    elif devices < 2:
        alignment_axis.text(
            0.5,
            0.5,
            "No master-slave relationship",
            ha="center",
            va="center",
            transform=alignment_axis.transAxes,
        )
        alignment_axis.set_yticks([])
    else:
        validity_binned = _conservative_bins(mask, max_mask_bins)
        alignment_binned = _conservative_bins(alignment_mask, max_mask_bins)
        pair_valid_binned = np.column_stack(
            [
                validity_binned[:, 0] & validity_binned[:, slave]
                for slave in range(1, devices)
            ]
            + [np.all(validity_binned, axis=1)]
        )
        pair_alignment_binned = np.column_stack(
            [
                alignment_binned[:, 0] & alignment_binned[:, slave]
                for slave in range(1, devices)
            ]
            + [np.all(alignment_binned, axis=1)]
        )
        pair_fractions, common_alignment_fraction = _alignment_pair_fractions(
            alignment_mask
        )
        alignment_labels = [
            f"M-S{slave} aligned   {100.0 * fraction:.3f}%"
            for slave, fraction in enumerate(pair_fractions, start=1)
        ] + [
            f"all aligned to M   {100.0 * common_alignment_fraction:.3f}%"
        ]
        alignment_bin_duration = duration_sec / max(
            1, pair_alignment_binned.shape[0]
        )
        for row in range(pair_alignment_binned.shape[1]):
            alignment_axis.hlines(
                row,
                0.0,
                duration_sec,
                color="#2ca02c",
                linewidth=5.0,
                alpha=0.55,
            )
            for start_bin, end_bin in _true_runs(~pair_valid_binned[:, row]):
                alignment_axis.broken_barh(
                    [
                        (
                            start_bin * alignment_bin_duration,
                            (end_bin - start_bin) * alignment_bin_duration,
                        )
                    ],
                    (row - 0.3, 0.6),
                    facecolors=_UNVERIFIED_MAPPING_COLOR,
                    edgecolors="none",
                )
            sync_warning = (
                pair_valid_binned[:, row]
                & ~pair_alignment_binned[:, row]
            )
            for start_bin, end_bin in _true_runs(sync_warning):
                alignment_axis.broken_barh(
                    [
                        (
                            start_bin * alignment_bin_duration,
                            (end_bin - start_bin) * alignment_bin_duration,
                        )
                    ],
                    (row - 0.34, 0.68),
                    facecolors="#f28e2b",
                    edgecolors="none",
                )
        alignment_axis.set_yticks(
            np.arange(pair_alignment_binned.shape[1]), alignment_labels
        )
        alignment_axis.set_ylim(pair_alignment_binned.shape[1] - 0.5, -0.5)
        alignment_axis.legend(
            handles=[
                Patch(facecolor="#2ca02c", alpha=0.55, label="aligned"),
                Patch(facecolor="#f28e2b", label="sync warning"),
                Patch(
                    facecolor=_UNVERIFIED_MAPPING_COLOR,
                    label="data invalid",
                ),
            ],
            loc="upper left",
            bbox_to_anchor=(1.005, 1.0),
            fontsize=8,
        )
    alignment_axis.set_ylabel("sync quality")
    alignment_title = "C. Master-referenced synchronization"
    if alignment_tolerance_samples is not None:
        tolerance_ms = (
            1000.0 * abs(float(alignment_tolerance_samples)) / sample_rate_hz
        )
        alignment_title += f" (limit ±{tolerance_ms:g} ms)"
    alignment_axis.set_title(alignment_title)

    camera_labels_seen: set[str] = set()
    for start, end, valid, label in sorted_camera:
        camera_label = str(label).strip() or "camera coverage"
        pc_axis.axvspan(
            start,
            end,
            color="#55a868" if valid else "#c44e52",
            alpha=0.18,
            label=(
                camera_label
                if camera_label not in camera_labels_seen
                else "_nolegend_"
            ),
        )
        camera_labels_seen.add(camera_label)
    if pc_time_sec.size:
        finite = np.isfinite(pc_time_sec) & np.isfinite(pc_residual_ms)
        if np.any(finite & pc_kept):
            kept_x = pc_time_sec[finite & pc_kept] - canonical_start_sec
            kept_y = pc_residual_ms[finite & pc_kept]
            pc_axis.scatter(
                kept_x,
                kept_y,
                s=7,
                alpha=0.35,
                color="#1677b3",
                label="clock anchor",
            )
            trend_x, trend_y = _binned_median(kept_x, kept_y)
            if trend_x.size:
                pc_axis.plot(
                    trend_x,
                    trend_y,
                    color="#0b4f79",
                    linewidth=1.6,
                    label="median trend",
                )
        if np.any(finite & ~pc_kept):
            pc_axis.scatter(
                pc_time_sec[finite & ~pc_kept] - canonical_start_sec,
                pc_residual_ms[finite & ~pc_kept],
                s=18,
                marker="x",
                linewidths=0.8,
                color="#c33",
                label="rejected anchor",
            )
        pc_axis.axhline(0.0, color="black", linewidth=0.7, label="zero residual")
    else:
        pc_axis.text(0.5, 0.5, pc_summary_text, ha="center", va="center", transform=pc_axis.transAxes)
    pc_axis.set_ylabel("PC residual (ms)")
    pc_axis.set_xlabel("canonical time from recording start (s)")
    pc_axis.set_title(
        "D. PC-time residuals and camera coverage"
        if sorted_camera
        else "D. PC-time residuals"
    )
    pc_legend_title: str | None = None
    if isinstance(pc_time, PcTimeModel):
        pc_legend_title = (
            f"anchors {pc_time.kept_count}/{pc_time.device_ms.size}   "
            f"drift {pc_time.drift_ppm:.3f} ppm\n"
            f"RMS {pc_time.residual_rms_ms:.3f} ms"
        )
    elif pc_time_summary:
        model_summary = _value(pc_time_summary, "model", {}) or {}
        if model_summary:
            pc_legend_title = (
                f"anchors {int(_value(model_summary, 'kept_update_count', 0))}/"
                f"{int(_value(model_summary, 'update_count', 0))}   "
                f"drift {float(_value(model_summary, 'drift_ppm', 0.0)):.3f} ppm\n"
                f"RMS {float(_value(model_summary, 'residual_rms_ms', float('nan'))):.3f} ms"
            )
    handles, legend_labels = pc_axis.get_legend_handles_labels()
    if handles:
        pc_axis.legend(
            handles,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(1.005, 1.0),
            fontsize=8,
            title=pc_legend_title,
            title_fontsize=7.5,
        )

    for axis in (residual_axis, validity_axis, alignment_axis, pc_axis):
        axis.set_xlim(0.0, duration_sec)
        axis.grid(axis="x", linewidth=0.35, alpha=0.35)
    maximum_residual = float(np.max(np.abs(finite_residual))) if finite_residual.size else float("nan")
    pc_rms = pc_time.residual_rms_ms if isinstance(pc_time, PcTimeModel) else float("nan")
    summary = f"{title} | {duration_sec / 60.0:.2f} min"
    if status:
        summary += f" | {status}"
    if np.isfinite(common_valid_fraction):
        summary += f" | common valid {100.0 * common_valid_fraction:.3f}%"
    if np.isfinite(common_alignment_fraction):
        summary += (
            f" | common sync-safe {100.0 * common_alignment_fraction:.3f}%"
        )
    if np.isfinite(maximum_residual):
        summary += f" | max residual {maximum_residual:.2f} samples"
    if np.isfinite(pc_rms):
        summary += f" | PC RMS {pc_rms:.1f} ms"
    if sorted_intervals:
        summary += f" | exclusions {len(sorted_intervals)}"
    figure.subplots_adjust(
        left=0.20,
        right=0.76,
        top=0.94,
        bottom=0.06,
        hspace=0.38,
    )
    figure.suptitle(summary, fontsize=12, y=0.985)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, metadata={"Software": "wild_preprocess"})
    plt.close(figure)
    if isinstance(mask, np.memmap):
        close_memmap(mask)
    if isinstance(alignment_mask, np.memmap):
        close_memmap(alignment_mask)
    return output
