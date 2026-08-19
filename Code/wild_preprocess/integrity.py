"""Typed, deterministic integrity decisions for post-hoc multi-device output."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from math import ceil
from typing import Any

import numpy as np

from .models import (
    ClassifiedInterval,
    DeviceGap,
    DeviceSourceStep,
    DeviceSyncSegment,
    DeviceTerminalSupport,
    RelativeOffsetStep,
    SyncObservation,
    SyncPairResult,
    UnresolvedBoundary,
)
from .sync.segments import map_source_positions_to_canonical, validate_segment_collection


Canonicalize = Callable[[int], int | None]


class SourceToCanonicalMapper:
    def __init__(
        self,
        *,
        device_index: int,
        source_scale: float,
        intercept_samples: float,
        device_gaps: Sequence[DeviceGap] = (),
        source_steps: Sequence[DeviceSourceStep] = (),
        device_sync_segments: Sequence[DeviceSyncSegment] | None = None,
    ) -> None:
        if device_index < 1 or source_scale <= 0:
            raise ValueError("invalid device index or scale")
        self.source_scale = float(source_scale)
        self.intercept_samples = float(intercept_samples)
        self.device_sync_segments = (
            validate_segment_collection(device_sync_segments, device_index=device_index)
            if device_sync_segments is not None
            else None
        )
        if self.device_sync_segments is not None:
            self.segments = ()
            return
        gaps = tuple(gap for gap in device_gaps if gap.device_index == device_index)
        steps = tuple(step for step in source_steps if step.device_index == device_index)
        boundaries = {0}
        for gap in gaps:
            boundaries.update((gap.canonical_start_sample, gap.canonical_end_sample))
        boundaries.update(step.canonical_sample for step in steps)
        ordered = sorted(boundaries)
        segments: list[tuple[int, int | None, float]] = []
        for index, lower in enumerate(ordered):
            upper = ordered[index + 1] if index + 1 < len(ordered) else None
            if any(gap.canonical_start_sample <= lower < gap.canonical_end_sample for gap in gaps):
                continue
            delta = -sum(
                gap.missing_samples for gap in gaps if lower >= gap.canonical_end_sample
            ) + sum(
                step.source_step_samples for step in steps if lower >= step.canonical_sample
            )
            segments.append((lower, upper, float(delta)))
        self.segments = tuple(segments)

    def map_array(self, source_samples: np.ndarray) -> np.ndarray:
        source = np.asarray(source_samples, dtype=np.float64)
        if self.device_sync_segments is not None:
            return map_source_positions_to_canonical(
                self.device_sync_segments,
                source,
            )
        result = np.full(source.shape, -1, dtype=np.int64)
        matches = np.zeros(source.shape, dtype=np.uint16)
        for lower, upper, delta in self.segments:
            candidate = (source - self.intercept_samples - delta) / self.source_scale
            rounded = np.rint(candidate).astype(np.int64)
            valid = (candidate >= lower) & (rounded >= lower)
            if upper is not None:
                valid &= (candidate < upper) & (rounded < upper)
            mapped = self.source_scale * rounded + self.intercept_samples + delta
            valid &= np.abs(mapped - source) <= 0.5
            result[valid] = rounded[valid]
            matches[valid] += 1
        result[matches != 1] = -1
        return result

    def __call__(self, source_sample: int) -> int | None:
        if source_sample < 0:
            raise ValueError("source sample must be non-negative")
        mapped = int(self.map_array(np.asarray([source_sample], dtype=np.int64))[0])
        return None if mapped < 0 else mapped


def classified_interval_sort_key(interval: ClassifiedInterval) -> tuple[object, ...]:
    """Stable ordering suitable for reports and deterministic rendering."""

    return (
        interval.canonical_start_sample,
        interval.canonical_end_sample,
        interval.affected_device_indices,
        interval.kind,
        interval.action,
        -1 if interval.source_start_sample is None else interval.source_start_sample,
        -1 if interval.source_end_sample is None else interval.source_end_sample,
        interval.confidence,
        interval.evidence,
    )


def sort_classified_intervals(
    intervals: Iterable[ClassifiedInterval],
) -> tuple[ClassifiedInterval, ...]:
    return tuple(sorted(intervals, key=classified_interval_sort_key))


def merge_compatible_intervals(
    intervals: Iterable[ClassifiedInterval],
) -> tuple[ClassifiedInterval, ...]:
    """Merge only contiguous intervals with identical semantics and source run."""

    merged: list[ClassifiedInterval] = []
    for interval in sort_classified_intervals(intervals):
        if not merged:
            merged.append(interval)
            continue
        previous = merged[-1]
        same_metadata = (
            previous.affected_device_indices == interval.affected_device_indices
            and previous.kind == interval.kind
            and previous.action == interval.action
            and previous.confidence == interval.confidence
            and previous.evidence == interval.evidence
        )
        contiguous_canonical = previous.canonical_end_sample == interval.canonical_start_sample
        source_contiguous = (
            previous.source_start_sample is None
            and interval.source_start_sample is None
        ) or (
            previous.source_end_sample == interval.source_start_sample
            and previous.source_start_sample is not None
            and interval.source_start_sample is not None
        )
        if same_metadata and contiguous_canonical and source_contiguous:
            merged[-1] = replace(
                previous,
                canonical_end_sample=interval.canonical_end_sample,
                source_end_sample=interval.source_end_sample,
            )
        else:
            merged.append(interval)
    return tuple(merged)


def device_gap_to_interval(gap: DeviceGap) -> ClassifiedInterval:
    """Represent an inferred loss without embedding an interpolation guard.

    Guards belong to the renderer because their width is determined by the
    interpolation method.  Keeping this source-of-truth interval exact lets a
    downstream renderer materialize the same decisions with another method.
    """

    return ClassifiedInterval(
        affected_device_indices=(gap.device_index,),
        canonical_start_sample=gap.canonical_start_sample,
        canonical_end_sample=gap.canonical_end_sample,
        kind="missing",
        action="zero_fill",
        confidence=gap.confidence,
        evidence=gap.evidence,
    )


def device_gaps_to_intervals(gaps: Iterable[DeviceGap]) -> tuple[ClassifiedInterval, ...]:
    return merge_compatible_intervals(device_gap_to_interval(gap) for gap in gaps)


def duplicate_destination_intervals(
    duplication_scan: dict[str, Any],
    *,
    device_index: int,
    canonicalize_current_sample: Canonicalize | None = None,
    confidence: str = "medium",
) -> tuple[ClassifiedInterval, ...]:
    """Convert only exact duplicate fragments to later-occurrence candidates.

    An episode envelope can contain unmatched samples introduced only to group
    nearby exact fragments.  It is deliberately not used for correction.
    The later occurrence is a policy choice supported by the stale-ring model,
    not a proof that the earlier source occurrence is genuine.
    """

    canonicalize = canonicalize_current_sample or int
    intervals: list[ClassifiedInterval] = []
    for episode in duplication_scan.get("episodes", []):
        try:
            lag = int(episode["lag_samples"])
            fragments = episode["exact_duplicate_fragments"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid exact duplication episode") from error
        if lag <= 0:
            raise ValueError("exact duplication lag must be positive")
        for fragment in fragments:
            if not isinstance(fragment, Sequence) or len(fragment) != 2:
                raise ValueError("exact duplicate fragment must be a two-element interval")
            current_start, current_end = (int(fragment[0]), int(fragment[1]))
            raw_samples = np.arange(current_start, current_end, dtype=np.int64)
            map_array = getattr(canonicalize, "map_array", None)
            if callable(map_array):
                mapped_values = np.asarray(map_array(raw_samples), dtype=np.int64)
                mapped = [None if value < 0 else int(value) for value in mapped_values]
            else:
                mapped = [canonicalize(int(sample)) for sample in raw_samples]
            run_start: int | None = None
            for offset in range(len(mapped) + 1):
                value = mapped[offset] if offset < len(mapped) else None
                if value is not None and run_start is None:
                    run_start = offset
                if value is not None or run_start is None:
                    continue
                run_values = [int(item) for item in mapped[run_start:offset] if item is not None]
                canonical_start = min(run_values)
                canonical_end = max(run_values) + 1
                if canonical_end > canonical_start:
                    raw_start = current_start + run_start
                    raw_end = current_start + offset
                    intervals.append(
                        ClassifiedInterval(
                            affected_device_indices=(device_index,),
                            canonical_start_sample=canonical_start,
                            canonical_end_sample=canonical_end,
                            source_start_sample=raw_start - lag,
                            source_end_sample=raw_end - lag,
                            kind="duplicate_destination",
                            action="zero_fill",
                            confidence=confidence,
                            evidence=(
                                f"exact all-channel duplication at lag {lag} samples; later occurrence "
                                "selected as the corrupt candidate by explicit stale-ring policy"
                            ),
                        )
                    )
                run_start = None
    return merge_compatible_intervals(intervals)


def zero_fill_spans(
    intervals: Iterable[ClassifiedInterval],
    *,
    device_index: int,
    canonical_samples: int,
    guard_samples: int = 0,
    guarded_kinds: frozenset[str] = frozenset(),
) -> tuple[tuple[int, int], ...]:
    """Return safe unioned render spans for one device.

    Guard expansion is intentionally done here, never in the classified
    interval.  Callers can restrict it to interpolation-sensitive kinds such
    as ``missing``.  All zero-fill decisions remain recoverable from the
    unexpanded interval list.
    """

    if device_index < 1 or canonical_samples < 0 or guard_samples < 0:
        raise ValueError("invalid device index, canonical sample count, or guard")
    spans: list[tuple[int, int]] = []
    for interval in intervals:
        if device_index not in interval.affected_device_indices or interval.action != "zero_fill":
            continue
        expansion = guard_samples if interval.kind in guarded_kinds else 0
        start = max(0, interval.canonical_start_sample - expansion)
        end = min(canonical_samples, interval.canonical_end_sample + expansion)
        if end > start:
            spans.append((start, end))
    spans.sort()
    unioned: list[tuple[int, int]] = []
    for start, end in spans:
        if unioned and start <= unioned[-1][1]:
            unioned[-1] = (unioned[-1][0], max(unioned[-1][1], end))
        else:
            unioned.append((start, end))
    return tuple(unioned)


def _cluster_steps(
    pairs: Sequence[SyncPairResult],
    *,
    fs: float,
    event_time_tolerance_seconds: float,
) -> tuple[tuple[tuple[SyncPairResult, RelativeOffsetStep], ...], ...]:
    events = [
        (pair, step)
        for pair in pairs
        for step in pair.model.offset_steps
    ]
    if fs <= 0:
        raise ValueError("sample rate must be positive")
    events.sort(key=lambda item: item[1].master_sample)
    clusters: list[list[tuple[SyncPairResult, RelativeOffsetStep]]] = []
    for event in events:
        if (
            not clusters
            or (event[1].master_sample - clusters[-1][-1][1].master_sample) / fs
            > event_time_tolerance_seconds
        ):
            clusters.append([event])
        else:
            clusters[-1].append(event)
    return tuple(tuple(cluster) for cluster in clusters)


def _cluster_is_attributable_missing(
    cluster: Sequence[tuple[SyncPairResult, RelativeOffsetStep]],
    *,
    expected_slave_indices: set[int],
    gap_level_tolerance_samples: float,
) -> bool:
    by_slave: dict[int, RelativeOffsetStep] = {}
    for pair, step in cluster:
        if pair.slave_index in by_slave:
            return False
        by_slave[pair.slave_index] = step
    steps = list(by_slave.values())
    negative = [step for step in steps if step.offset_step_samples < 0]
    positive = [step for step in steps if step.offset_step_samples > 0]
    if len(by_slave) == 1 and len(negative) == 1 and not positive:
        return True
    if not negative and set(by_slave) == expected_slave_indices and len(positive) == len(expected_slave_indices):
        sizes = [step.missing_samples for step in positive]
        return max(sizes) - min(sizes) <= gap_level_tolerance_samples
    return False


def _boundary_window_samples(
    pair: SyncPairResult,
    step: RelativeOffsetStep,
    *,
    fs: float,
    window_seconds: float,
    fallback_step_seconds: float,
) -> tuple[int, int]:
    accepted = [
        observation
        for observation in pair.observations
        if observation.accepted and not observation.search_mode.startswith("endpoint_probe")
    ]
    before = [observation for observation in accepted if observation.center_time_sec <= step.time_sec]
    after = [observation for observation in accepted if observation.center_time_sec >= step.time_sec]
    radius_seconds = window_seconds / 2.0 + max(0.0, fallback_step_seconds)
    start_seconds = (
        before[-1].center_time_sec - window_seconds / 2.0
        if before
        else step.time_sec - radius_seconds
    )
    end_seconds = (
        after[0].center_time_sec + window_seconds / 2.0
        if after
        else step.time_sec + radius_seconds
    )
    return max(0, int(start_seconds * fs)), max(1, int(ceil(end_seconds * fs)))


def unresolved_boundaries_from_offset_clusters(
    pairs: Sequence[SyncPairResult],
    *,
    device_count: int,
    master_index: int,
    fs: float,
    window_seconds: float,
    fallback_step_seconds: float,
    event_time_tolerance_seconds: float,
    gap_level_tolerance_samples: float,
    boundary_guard_samples: int | None = None,
    canonicalize_master_sample: Canonicalize | None = None,
) -> tuple[UnresolvedBoundary, ...]:
    """Make local, structured exclusions for non-attributable offset clusters.

    In particular, a positive step in only one master/slave pair remains
    unresolved.  It is not treated as an extra source interval because that
    requires a separate slave-to-slave confirmation.
    """

    if device_count < 2 or not 0 <= master_index < device_count or fs <= 0 or window_seconds <= 0:
        raise ValueError("invalid device count, master index, sample rate, or observation window")
    canonicalize = canonicalize_master_sample or int
    expected = {index + 1 for index in range(device_count) if index != master_index}
    boundaries: list[UnresolvedBoundary] = []
    for cluster in _cluster_steps(
        pairs,
        fs=fs,
        event_time_tolerance_seconds=event_time_tolerance_seconds,
    ):
        if device_count >= 3 and _cluster_is_attributable_missing(
            cluster,
            expected_slave_indices=expected,
            gap_level_tolerance_samples=gap_level_tolerance_samples,
        ):
            continue
        raw_starts: list[int] = []
        raw_ends: list[int] = []
        for pair, step in cluster:
            if boundary_guard_samples is None:
                start, end = _boundary_window_samples(
                    pair,
                    step,
                    fs=fs,
                    window_seconds=window_seconds,
                    fallback_step_seconds=fallback_step_seconds,
                )
            else:
                guard = max(1, int(boundary_guard_samples))
                start = max(0, int(step.master_sample) - guard)
                end = int(step.master_sample) + guard + 1
            raw_starts.append(start)
            raw_ends.append(end)
        canonical_start = int(canonicalize(min(raw_starts)))
        canonical_end = int(canonicalize(max(raw_ends)))
        if canonical_end <= canonical_start:
            raise ValueError("canonical boundary mapping must preserve interval order")
        description = ", ".join(
            f"slave {pair.slave_index}: {step.offset_step_samples:+.1f}"
            for pair, step in cluster
        )
        boundaries.append(
            UnresolvedBoundary(
                canonical_start_sample=canonical_start,
                canonical_end_sample=canonical_end,
                pair_slave_indices=tuple(sorted({pair.slave_index for pair, _ in cluster})),
                evidence=(
                    "non-attributable persistent offset cluster; local interval uses "
                    + (
                        "the supporting correlation windows"
                        if boundary_guard_samples is None
                        else f"a ±{max(1, int(boundary_guard_samples))}-sample refined-boundary guard"
                    )
                    + f" ({description})"
                ),
            )
        )
    return tuple(
        sorted(
            boundaries,
            key=lambda boundary: (
                boundary.canonical_start_sample,
                boundary.canonical_end_sample,
                boundary.pair_slave_indices,
            ),
        )
    )


def source_steps_from_unresolved_offset_clusters(
    pairs: Sequence[SyncPairResult],
    *,
    device_count: int,
    master_index: int,
    fs: float,
    event_time_tolerance_seconds: float,
    gap_level_tolerance_samples: float,
    canonicalize_master_sample: Canonicalize | None = None,
) -> tuple[DeviceSourceStep, ...]:
    """Keep the measured piecewise mapping across locally masked boundaries.

    Attributable missing-data clusters are already represented by
    :class:`DeviceGap` and must not be applied a second time.  Every remaining
    pair step is retained only as a source-coordinate jump; the surrounding
    interval is separately invalidated for every device.
    """

    if device_count < 2 or not 0 <= master_index < device_count:
        raise ValueError("invalid device count or master index")
    canonicalize = canonicalize_master_sample or int
    expected = {index + 1 for index in range(device_count) if index != master_index}
    source_steps: list[DeviceSourceStep] = []
    for cluster in _cluster_steps(
        pairs,
        fs=fs,
        event_time_tolerance_seconds=event_time_tolerance_seconds,
    ):
        if device_count >= 3 and _cluster_is_attributable_missing(
            cluster,
            expected_slave_indices=expected,
            gap_level_tolerance_samples=gap_level_tolerance_samples,
        ):
            continue
        for pair, step in cluster:
            source_steps.append(
                DeviceSourceStep(
                    device_index=pair.slave_index,
                    canonical_sample=int(canonicalize(step.master_sample)),
                    source_step_samples=float(step.offset_step_samples),
                    confidence="unresolved",
                    evidence=(
                        "measured source-coordinate continuation across a locally "
                        f"excluded boundary; {step.evidence}"
                    ),
                )
            )
    return tuple(
        sorted(
            source_steps,
            key=lambda item: (item.canonical_sample, item.device_index),
        )
    )


def source_sample_to_canonical(
    source_sample: int,
    *,
    device_index: int,
    source_scale: float,
    intercept_samples: float,
    device_gaps: Sequence[DeviceGap] = (),
    source_steps: Sequence[DeviceSourceStep] = (),
    device_sync_segments: Sequence[DeviceSyncSegment] | None = None,
) -> int | None:
    """Invert the monotone piecewise source mapping for an observed raw index."""

    if source_sample < 0 or device_index < 1 or source_scale <= 0:
        raise ValueError("invalid source sample, device index, or scale")
    return SourceToCanonicalMapper(
        device_index=device_index,
        source_scale=source_scale,
        intercept_samples=intercept_samples,
        device_gaps=device_gaps,
        source_steps=source_steps,
        device_sync_segments=device_sync_segments,
    )(source_sample)


def unresolved_boundary_to_interval(
    boundary: UnresolvedBoundary,
    *,
    device_count: int,
) -> ClassifiedInterval:
    if device_count < 1:
        raise ValueError("device_count must be positive")
    return ClassifiedInterval(
        affected_device_indices=tuple(range(1, device_count + 1)),
        canonical_start_sample=boundary.canonical_start_sample,
        canonical_end_sample=boundary.canonical_end_sample,
        kind="unresolved_boundary",
        action="zero_fill",
        confidence="unresolved",
        evidence=boundary.evidence,
    )


def terminal_support_from_pair(
    pair: SyncPairResult,
    *,
    canonicalize_master_sample: Canonicalize | None = None,
) -> DeviceTerminalSupport | None:
    """Convert a pair's first unsupported sample to device-local support."""

    if pair.terminal_crop_master_sample is None:
        return None
    canonicalize = canonicalize_master_sample or int
    end = int(canonicalize(int(pair.terminal_crop_master_sample)))
    return DeviceTerminalSupport(
        device_index=pair.slave_index,
        supported_canonical_end_sample=end,
        evidence=pair.terminal_crop_reason or "terminal sync support ended",
    )


def terminal_support_to_interval(
    support: DeviceTerminalSupport,
    *,
    canonical_end_sample: int,
) -> ClassifiedInterval | None:
    """Mark a device-only unsupported tail, without choosing a global crop."""

    if canonical_end_sample < support.supported_canonical_end_sample:
        raise ValueError("canonical end precedes the supported terminal endpoint")
    if canonical_end_sample == support.supported_canonical_end_sample:
        return None
    return ClassifiedInterval(
        affected_device_indices=(support.device_index,),
        canonical_start_sample=support.supported_canonical_end_sample,
        canonical_end_sample=canonical_end_sample,
        kind="terminal_unsupported",
        action="zero_fill",
        confidence=support.confidence,
        evidence=support.evidence,
    )


def intervals_to_dicts(intervals: Iterable[ClassifiedInterval]) -> list[dict[str, Any]]:
    return [interval.to_dict() for interval in sort_classified_intervals(intervals)]
