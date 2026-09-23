from __future__ import annotations

import numpy as np

from ..models import (
    DeviceSyncAnchor,
    DeviceSyncSegment,
    RelativeOffsetStep,
    SyncModel,
    SyncObservation,
    SyncOptions,
    validate_device_sync_segments,
)
from .gaps import AdaptiveChangePoint, detect_relative_offset_steps


def _is_reliable_model_observation(item: SyncObservation, options: SyncOptions) -> bool:
    return bool(
        item.accepted and item.model_inlier
        and np.isfinite(item.observed_offset_samples)
        and item.peak_correlation >= options.min_peak_correlation
        and item.peak_to_background >= options.min_peak_to_background
        and item.peak_margin_fraction >= options.min_peak_margin_fraction
        and not item.search_mode.startswith("coarse")
    )


def _continuous_bracket_evidence(
    observations: list[SyncObservation],
    model: SyncModel,
    options: SyncOptions,
    left: int,
    right: int,
    max_error_samples: float,
    *,
    allow_terminal_short_span: bool = False,
) -> dict[str, object] | None:
    """Check two-sided affine support, conditional on continuous stable clocks."""
    if (
        left < 0 or right >= len(observations) or left >= right
        or not np.isfinite(model.residual_rms_samples)
        or model.residual_rms_samples > options.max_model_rms_samples
        or model.residual_max_abs_samples > options.max_model_residual_samples
    ):
        return None

    if not _is_reliable_model_observation(observations[left], options) or not _is_reliable_model_observation(observations[right], options):
        return None
    horizon = max(3, 4 * options.gap_persistence_observations)
    sides = (
        [item for item in observations[max(0, left - horizon + 1):left + 1] if _is_reliable_model_observation(item, options)],
        [item for item in observations[right:right + horizon] if _is_reliable_model_observation(item, options)],
    )
    if any(len(side) < 3 for side in sides):
        return None
    first_time, last_time = sides[0][0].center_time_sec, sides[1][-1].center_time_sec
    if any(first_time <= step.time_sec <= last_time for step in model.offset_steps):
        return None
    bracket = np.asarray([observations[left].center_time_sec, observations[right].center_time_sec])
    predicted = np.asarray([model.offset_at_seconds(time) for time in bracket])
    maximum_error = 0.0
    terminal_short_span = False
    for side_index, side in enumerate(sides):
        times = np.asarray([item.center_time_sec for item in side])
        values = np.asarray([item.observed_offset_samples for item in side])
        if np.ptp(times) < options.window_seconds:
            if not (
                allow_terminal_short_span and side_index == 1
                and side[-1] is observations[-1]
                and np.ptp(times) >= options.step_seconds
            ):
                return None
            terminal_short_span = True
        # Center the fit to avoid conditioning errors late in a recording.
        fit = np.polyfit(times - bracket[0], values, 1)
        residual = float(np.max(np.abs(values - np.polyval(fit, times - bracket[0]))))
        disagreement = float(np.max(np.abs(np.polyval(fit, bracket - bracket[0]) - predicted)))
        common_error = max(abs(item.observed_offset_samples - model.offset_at_seconds(item.center_time_sec)) for item in side)
        maximum_error = max(maximum_error, residual + disagreement, common_error)
    if not np.isfinite(maximum_error) or maximum_error > max_error_samples:
        return None
    return {
        "start_time_sec": float(bracket[0]),
        "end_time_sec": float(bracket[1]),
        "before_anchor_count": len(sides[0]),
        "after_anchor_count": len(sides[1]),
        "maximum_model_disagreement_samples": maximum_error,
        "terminal_short_span": terminal_short_span,
        "evidence": "two-sided affine continuity; inferred timing, not a measured correlation",
    }


def supported_observation_gaps(
    observations: list[SyncObservation],
    model: SyncModel,
    options: SyncOptions,
    *,
    max_error_samples: float,
    barriers: tuple[AdaptiveChangePoint, ...] = (),
) -> tuple[tuple[int, ...], tuple[dict[str, object], ...]]:
    """Bracket weak/rejected model evidence without creating measured anchors."""
    supported: list[int] = []
    intervals: list[dict[str, object]] = []
    index = 0
    while index < len(observations):
        if _is_reliable_model_observation(observations[index], options):
            index += 1
            continue
        start = index
        while index < len(observations) and not _is_reliable_model_observation(observations[index], options):
            index += 1
        evidence = _continuous_bracket_evidence(
            observations, model, options, start - 1, index, max_error_samples
        )
        if evidence is None or any(
            evidence["start_time_sec"] <= point.time_sec <= evidence["end_time_sec"]
            for point in barriers
        ):
            continue
        supported.extend(range(start, index))
        intervals.append({**evidence, "observation_indices": list(range(start, index))})
    return tuple(supported), tuple(intervals)


def select_continuity_boundaries(
    points: tuple[AdaptiveChangePoint, ...],
    observations: list[SyncObservation],
    model: SyncModel,
    options: SyncOptions,
    *,
    max_error_samples: float,
) -> tuple[tuple[AdaptiveChangePoint, ...], tuple[dict[str, object], ...]]:
    """Do not promote explained lag rounding into structural data loss.

    Independently localized changes remain barriers regardless of their size.
    The detector's candidates remain available as explicit diagnostics.
    """
    retained: list[AdaptiveChangePoint] = []
    explained: list[dict[str, object]] = []
    times = np.asarray([item.center_time_sec for item in observations])
    for point in points:
        evidence = None
        if not point.sample_localized and abs(point.delta_samples) <= max_error_samples:
            right = int(np.searchsorted(times, point.time_sec, side="right"))
            evidence = _continuous_bracket_evidence(
                observations, model, options, right - 1, right, max_error_samples,
                allow_terminal_short_span=True,
            )
        if point.localization_status == "verified_alias" or evidence is not None:
            explained.append({
                "time_sec": point.time_sec,
                "delta_samples": point.delta_samples,
                "localization_status": point.localization_status,
                "evidence": "raw-verified alias" if evidence is None else evidence,
            })
        else:
            retained.append(point)
    return tuple(retained), tuple(explained)


def anchors_from_accepted_observations(
    observations: list[SyncObservation],
    fs: float,
    options: SyncOptions,
) -> tuple[DeviceSyncAnchor, ...]:
    """Convert accepted, correlation-qualified observations into segment evidence.

    A coarse or rejected correlation result cannot become a verified anchor just
    because it is convenient for a later segment fit.  The source convention is
    the established one: ``slave source = canonical master + offset``.
    """

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be finite and positive")
    anchors: list[DeviceSyncAnchor] = []
    for observation in sorted(observations, key=lambda item: item.center_time_sec):
        if not _is_reliable_model_observation(observation, options):
            continue
        canonical_sample = int(round(observation.center_time_sec * fs))
        confidence = (
            "high"
            if observation.peak_correlation >= max(0.1, options.min_peak_correlation)
            and observation.peak_margin_fraction >= max(0.05, options.min_peak_margin_fraction)
            else "medium"
        )
        anchors.append(
            DeviceSyncAnchor(
                canonical_sample=canonical_sample,
                source_sample=float(canonical_sample + observation.observed_offset_samples),
                verified=True,
                confidence=confidence,
                evidence=(
                    f"accepted {observation.search_mode} anchor; correlation "
                    f"{observation.peak_correlation:.3f}; margin "
                    f"{observation.peak_margin_fraction:.3f}"
                ),
            )
        )
    # A pair of observations at exactly the same center cannot provide a
    # two-sided affine fit; reject rather than implicitly retaining one.
    samples = [anchor.canonical_sample for anchor in anchors]
    if len(set(samples)) != len(samples):
        raise ValueError("accepted anchors must have distinct canonical samples")
    return tuple(anchors)


def _fit_segment_affine(anchors: tuple[DeviceSyncAnchor, ...]) -> tuple[float, float, np.ndarray]:
    """Robustly fit source = scale * canonical + intercept for one segment."""

    x = np.asarray([item.canonical_sample for item in anchors], dtype=np.float64)
    y = np.asarray([item.source_sample for item in anchors], dtype=np.float64)
    keep = np.ones(x.size, dtype=bool)
    coefficients = np.polyfit(x, y, 1)
    for _ in range(6):
        coefficients = np.polyfit(x[keep], y[keep], 1)
        residuals = y - np.polyval(coefficients, x)
        center = float(np.median(residuals[keep]))
        mad = float(np.median(np.abs(residuals[keep] - center)))
        gate = max(1e-6, 4.0 * 1.4826 * mad)
        updated = np.abs(residuals - center) <= gate
        if updated.sum() < 2 or np.array_equal(updated, keep):
            break
        keep = updated
    coefficients = np.polyfit(x[keep], y[keep], 1)
    return float(coefficients[0]), float(coefficients[1]), y - np.polyval(coefficients, x)


def _change_boundaries(change_points: tuple[AdaptiveChangePoint | int, ...]) -> tuple[int, ...]:
    boundaries: list[int] = []
    for point in change_points:
        boundary = point.canonical_boundary_sample if isinstance(point, AdaptiveChangePoint) else point
        if not isinstance(boundary, int) or boundary < 0:
            raise ValueError("change boundaries must be non-negative integer samples")
        boundaries.append(boundary)
    if len(set(boundaries)) != len(boundaries):
        raise ValueError("change boundaries must be unique")
    return tuple(sorted(boundaries))


def fit_independent_device_segments(
    anchors: tuple[DeviceSyncAnchor, ...] | list[DeviceSyncAnchor],
    change_points: tuple[AdaptiveChangePoint | int, ...] | list[AdaptiveChangePoint | int],
    *,
    device_index: int,
    canonical_start_sample: int,
    canonical_end_sample: int,
    source_sample_count: int,
    unresolved_ranges: tuple[tuple[int, int], ...] | list[tuple[int, int]] = (),
) -> tuple[DeviceSyncSegment, ...]:
    """Fit independently supported device mappings between change boundaries.

    Each post-boundary affine intercept is estimated solely from that range's
    anchors.  Empty, uncertain, or under-anchored ranges are absent from the
    result and consequently map to validity 0 at merge time.
    """

    if device_index < 1 or canonical_start_sample < 0 or canonical_end_sample <= canonical_start_sample:
        raise ValueError("invalid device or canonical segment bounds")
    if source_sample_count <= 0:
        raise ValueError("source_sample_count must be positive")
    ordered_anchors = tuple(anchors)
    if any(not isinstance(anchor, DeviceSyncAnchor) for anchor in ordered_anchors):
        raise ValueError("anchors must be DeviceSyncAnchor instances")
    if tuple(anchor.canonical_sample for anchor in ordered_anchors) != tuple(
        sorted(anchor.canonical_sample for anchor in ordered_anchors)
    ):
        raise ValueError("anchors must be ordered by canonical sample")
    if len({anchor.canonical_sample for anchor in ordered_anchors}) != len(ordered_anchors):
        raise ValueError("anchors must have unique canonical samples")
    ranges = tuple((int(start), int(end)) for start, end in unresolved_ranges)
    for start, end in ranges:
        if start < canonical_start_sample or end > canonical_end_sample or end <= start:
            raise ValueError("unresolved ranges must be non-empty and within canonical support")
    if any(next_start < end for (_, end), (next_start, _) in zip(ranges, ranges[1:])):
        raise ValueError("unresolved ranges must be ordered and non-overlapping")

    boundaries = _change_boundaries(tuple(change_points))
    edges = sorted(
        {
            canonical_start_sample,
            canonical_end_sample,
            *[value for value in boundaries if canonical_start_sample < value < canonical_end_sample],
            *[value for item in ranges for value in item],
        }
    )
    segments: list[DeviceSyncSegment] = []
    for start, end in zip(edges, edges[1:]):
        if any(start >= unresolved_start and end <= unresolved_end for unresolved_start, unresolved_end in ranges):
            continue
        local = tuple(
            anchor
            for anchor in ordered_anchors
            if start <= anchor.canonical_sample < end and anchor.is_publishable_evidence
        )
        if len(local) < 2:
            continue
        scale, intercept, residuals = _fit_segment_affine(local)
        if abs(scale - 1.0) <= 1e-12:
            scale = 1.0
        if abs(intercept) <= 1e-9:
            intercept = 0.0
        if not np.isfinite(scale) or scale <= 0:
            continue
        # Do not claim support where the independently fitted affine map has
        # no raw source frame.  This can trim only unsupported segment ends;
        # it never bridges a declared unresolved range.
        supported_start = max(start, int(np.ceil((-intercept - 1e-9) / scale)))
        supported_end = min(
            end,
            int(np.floor((source_sample_count - 1 - intercept + 1e-9) / scale)) + 1,
        )
        local = tuple(
            anchor
            for anchor in local
            if supported_start <= anchor.canonical_sample < supported_end
        )
        if supported_end <= supported_start or len(local) < 2:
            continue
        scale, intercept, residuals = _fit_segment_affine(local)
        if abs(scale - 1.0) <= 1e-12:
            scale = 1.0
        if abs(intercept) <= 1e-9:
            intercept = 0.0
        # Residual metadata must describe the exact coefficients that will be
        # serialized and validated.  Snapping a near-unity scale can otherwise
        # move a long-recording anchor by more than floating-point epsilon.
        residuals = np.asarray(
            [
                anchor.source_sample
                - (scale * anchor.canonical_sample + intercept)
                for anchor in local
            ],
            dtype=np.float64,
        )
        rms = float(np.sqrt(np.mean(np.square(residuals))))
        max_residual = float(np.max(np.abs(residuals)))
        confidence = "high" if all(anchor.confidence == "high" for anchor in local) else "medium"
        candidate = DeviceSyncSegment(
                device_index=device_index,
                canonical_start_sample=supported_start,
                canonical_end_sample=supported_end,
                source_start_sample=0,
                source_end_sample=source_sample_count,
                source_scale=scale,
                source_intercept_samples=intercept,
                anchors=local,
                residual_rms_samples=rms,
                residual_max_abs_samples=max_residual,
                confidence=confidence,
                start_transition=(
                    "recording_start" if supported_start == canonical_start_sample else "independent_reacquisition"
                ),
                end_transition=(
                    "recording_end" if supported_end == canonical_end_sample else "segment_boundary"
                ),
                publishable=True,
                evidence="independent affine fit from verified anchors",
            )
        if segments:
            previous_last = segments[-1].map_canonical_sample(
                segments[-1].canonical_end_sample - 1
            )
            candidate_first = candidate.map_canonical_sample(
                candidate.canonical_start_sample
            )
            if candidate_first <= previous_last:
                # This independently fitted range would reuse or reverse raw
                # source already claimed by a prior verified segment.  It is
                # therefore unsupported (validity 0), not a reason to weaken
                # the collection invariant or fail other usable devices.
                continue
        segments.append(candidate)
    return validate_device_sync_segments(segments, device_index=device_index)


def _robust_inlier_mask(x: np.ndarray, y: np.ndarray, *, constant_offset: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return a robust fit mask and [slope, intercept] coefficients."""

    keep = np.ones(x.size, dtype=bool)
    coefficients = np.asarray([0.0, float(np.median(y))], dtype=np.float64)
    for _ in range(8):
        if constant_offset:
            coefficients = np.asarray([0.0, float(np.median(y[keep]))], dtype=np.float64)
        else:
            coefficients = np.polyfit(x[keep], y[keep], 1)
        residuals = y - np.polyval(coefficients, x)
        center = np.median(residuals[keep])
        mad = np.median(np.abs(residuals[keep] - center))
        gate = max(6.0, 6.0 * 1.4826 * mad)
        updated = np.abs(residuals - center) <= gate
        minimum = 1 if constant_offset else 2
        if updated.sum() < minimum or np.array_equal(updated, keep):
            break
        keep = updated
    if constant_offset:
        coefficients = np.asarray([0.0, float(np.median(y[keep]))], dtype=np.float64)
    else:
        coefficients = np.polyfit(x[keep], y[keep], 1)
    return keep, coefficients


def fit_affine_sync_model(
    observations: list[SyncObservation],
    fs: float,
    *,
    options: SyncOptions | None = None,
    offset_steps: tuple[RelativeOffsetStep, ...] | None = None,
) -> SyncModel:
    """Fit the master-to-slave offset model from accepted observations.

    Supplying ``options`` enables the production short-recording rule.  The
    optional argument keeps the original affine-only call API available to
    legacy callers and focused numerical tests.
    """

    accepted = [observation for observation in observations if observation.accepted]
    if len(accepted) < 2:
        return SyncModel(
            intercept_samples=0.0,
            slope_samples_per_second=0.0,
            drift_ppm=0.0,
            residual_rms_samples=float("inf"),
            residual_max_abs_samples=float("inf"),
            accepted_count=len(accepted),
            observation_count=len(observations),
        )
    x = np.asarray([observation.center_time_sec for observation in accepted], dtype=np.float64)
    observed_y = np.asarray([observation.observed_offset_samples for observation in accepted], dtype=np.float64)
    if offset_steps is None:
        offset_steps = (
            detect_relative_offset_steps(observations, fs, options)
            if options is not None
            else ()
        )
    step_values = np.asarray(
        [
            sum(step.offset_step_samples for step in offset_steps if step.time_sec <= time_sec)
            for time_sec in x
        ],
        dtype=np.float64,
    )
    y = observed_y - step_values
    observed_times = [observation.center_time_sec for observation in observations]
    usable_duration = (
        max(observed_times) - min(observed_times) + options.window_seconds
        if options is not None and observed_times
        else float("inf")
    )
    constant_offset = options is not None and usable_duration < options.short_recording_seconds
    keep, coefficients = _robust_inlier_mask(x, y, constant_offset=constant_offset)
    all_residuals = y - np.polyval(coefficients, x)
    residuals = all_residuals[keep]
    for observation, is_inlier, residual in zip(accepted, keep, all_residuals):
        observation.model_inlier = bool(is_inlier)
        observation.model_residual_samples = float(residual)
    slope = float(coefficients[0])
    intercept = float(coefficients[1])
    rms = float(np.sqrt(np.mean(np.square(residuals))))
    max_abs = float(np.max(np.abs(residuals)))
    return SyncModel(
        intercept_samples=intercept,
        slope_samples_per_second=slope,
        drift_ppm=slope / fs * 1e6,
        residual_rms_samples=rms,
        residual_max_abs_samples=max_abs,
        accepted_count=int(keep.sum()),
        observation_count=len(observations),
        is_constant_offset=constant_offset,
        offset_steps=tuple(offset_steps),
    )
