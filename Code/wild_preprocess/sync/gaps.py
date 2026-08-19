from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from ..models import DeviceGap, RelativeOffsetStep, SyncObservation, SyncOptions, SyncPairResult


@dataclass(frozen=True)
class AdaptiveChangePoint:
    """A locally verified relative-offset level transition."""

    canonical_boundary_sample: int
    time_sec: float
    delta_samples: float
    before_level_samples: float
    after_level_samples: float
    uncertainty_samples: float
    before_slope_samples_per_second: float
    after_slope_samples_per_second: float
    confidence: str
    before_count: int
    after_count: int
    evidence: str


def _reliable_observations(
    observations: Sequence[SyncObservation], options: SyncOptions
) -> list[SyncObservation]:
    """Keep anchors which still meet the original correlation gates."""

    return sorted(
        (
            item
            for item in observations
            if item.accepted
            and np.isfinite(item.observed_offset_samples)
            and np.isfinite(item.peak_correlation)
            and np.isfinite(item.peak_to_background)
            and np.isfinite(item.peak_margin_fraction)
            and item.peak_correlation >= options.min_peak_correlation
            and item.peak_to_background >= options.min_peak_to_background
            and item.peak_margin_fraction >= options.min_peak_margin_fraction
        ),
        key=lambda item: item.center_time_sec,
    )


def _local_detrended_level(
    times: np.ndarray,
    offsets: np.ndarray,
    boundary_time: float,
) -> tuple[float, float, np.ndarray]:
    """Fit a local slope and return its robust level at ``boundary_time``."""

    slope = float(np.polyfit(times, offsets, 1)[0]) if times.size >= 2 else 0.0
    shifted = offsets - slope * (times - boundary_time)
    level = float(np.median(shifted))
    return slope, level, shifted - level


def _estimator_uncertainty(observations: Sequence[SyncObservation]) -> float:
    """Return recorded estimator uncertainty without inventing a loose floor."""

    residuals = np.asarray(
        [abs(item.model_residual_samples) for item in observations], dtype=np.float64
    )
    residuals = residuals[np.isfinite(residuals)]
    return float(np.median(residuals)) if residuals.size else 0.0


def detect_adaptive_change_points(
    observations: Sequence[SyncObservation],
    fs: float,
    options: SyncOptions,
    *,
    sigma_multiplier: float = 4.0,
) -> tuple[AdaptiveChangePoint, ...]:
    """Detect verified local offset levels without a fixed gap-size threshold.

    ``gap_min_step_samples`` remains a compatibility option, but is deliberately
    not used here: a verified one-sample change is not correlation jitter.
    """

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be finite and positive")
    if not np.isfinite(sigma_multiplier) or sigma_multiplier <= 0:
        raise ValueError("sigma_multiplier must be finite and positive")
    accepted = _reliable_observations(observations, options)
    persistence = max(2, int(options.gap_persistence_observations))
    if len(accepted) < 2 * persistence:
        return ()

    times = np.asarray([item.center_time_sec for item in accepted], dtype=np.float64)
    offsets = np.asarray([item.observed_offset_samples for item in accepted], dtype=np.float64)
    points: list[AdaptiveChangePoint] = []
    index = persistence
    while index <= len(accepted) - persistence:
        before_indices = np.arange(max(0, index - 4 * persistence), index, dtype=np.int64)
        # Use all nearby post-candidate evidence (up to the same four-times
        # persistence horizon as the pre-side).  Fitting a line to only two
        # samples makes a single integer-lag jitter point look like a perfectly
        # stable one-sample step followed by slope; the longer local side keeps
        # verified permanent one-sample changes while rejecting those short
        # quantization excursions.
        after_indices = np.arange(
            index,
            min(len(accepted), index + 4 * persistence),
            dtype=np.int64,
        )
        boundary_time = 0.5 * (times[index - 1] + times[index])
        before_slope, before_level, before_residuals = _local_detrended_level(
            times[before_indices], offsets[before_indices], boundary_time
        )
        after_slope, after_level, after_residuals = _local_detrended_level(
            times[after_indices], offsets[after_indices], boundary_time
        )
        residuals = np.concatenate((before_residuals, after_residuals))
        residual_center = float(np.median(residuals))
        sigma = float(1.4826 * np.median(np.abs(residuals - residual_center)))
        evidence_items = [accepted[int(item)] for item in np.concatenate((before_indices, after_indices))]
        estimator_uncertainty = _estimator_uncertainty(evidence_items)
        gate = max(1.0, sigma_multiplier * sigma, estimator_uncertainty)
        before_stable = bool(np.all(np.abs(before_residuals) <= gate + 1e-12))
        after_stable = bool(np.all(np.abs(after_residuals) <= gate + 1e-12))
        # A change exactly at the one-sample observability floor is publishable
        # only when both local levels are tighter than that floor.  Otherwise
        # integer peak quantization itself can explain the apparent change.
        before_level_spread = float(
            np.ptp(offsets[before_indices] - before_slope * times[before_indices])
        )
        after_level_spread = float(
            np.ptp(offsets[after_indices] - before_slope * times[after_indices])
        )
        quantization_resolved = (
            before_level_spread < gate - 1e-12
            and after_level_spread < gate - 1e-12
        )
        before_span = max(float(times[before_indices[-1]] - times[before_indices[0]]), 1e-9)
        after_span = max(float(times[after_indices[-1]] - times[after_indices[0]]), 1e-9)
        compatible_slopes = abs(after_slope - before_slope) <= gate / min(before_span, after_span) + 1e-12
        regular_cadence = times[index] - times[index - 1] <= 2.0 * options.step_seconds + 1e-9
        delta = after_level - before_level
        if (
            abs(delta) >= gate - 1e-12
            and before_stable
            and after_stable
            and quantization_resolved
            and compatible_slopes
            and regular_cadence
        ):
            minimum_correlation = min(item.peak_correlation for item in evidence_items)
            minimum_margin = min(item.peak_margin_fraction for item in evidence_items)
            confidence = (
                "high"
                if minimum_correlation >= max(0.1, options.min_peak_correlation)
                and minimum_margin >= max(0.05, options.min_peak_margin_fraction)
                else "medium"
            )
            points.append(
                AdaptiveChangePoint(
                    canonical_boundary_sample=int(round(boundary_time * fs)),
                    time_sec=float(boundary_time),
                    delta_samples=float(delta),
                    before_level_samples=float(before_level),
                    after_level_samples=float(after_level),
                    uncertainty_samples=float(gate),
                    before_slope_samples_per_second=float(before_slope),
                    after_slope_samples_per_second=float(after_slope),
                    confidence=confidence,
                    before_count=int(before_indices.size),
                    after_count=int(after_indices.size),
                    evidence=(
                        f"adaptive locally detrended levels; delta {delta:+.3f}; "
                        f"gate {gate:.3f} (1, {sigma_multiplier:.3g}*MAD {sigma:.3f}, "
                        f"estimator {estimator_uncertainty:.3f}); "
                        f"slopes {before_slope:.6g}/{after_slope:.6g}; "
                        f"min correlation {minimum_correlation:.3f}; "
                        f"min peak margin {minimum_margin:.3f}"
                    ),
                )
            )
            index += persistence
        else:
            index += 1
    return tuple(points)


def detect_relative_offset_steps(
    observations: Sequence[SyncObservation],
    fs: float,
    options: SyncOptions,
) -> tuple[RelativeOffsetStep, ...]:
    """Keep the legacy global-step projection stable until pipeline migration.

    The segment API above is the generalized detector.  This compatibility
    function intentionally retains the old operational threshold because its
    output is still interpreted by the old cumulative global mapping.
    """

    accepted = sorted(
        (observation for observation in observations if observation.accepted),
        key=lambda observation: observation.center_time_sec,
    )
    persistence = max(1, int(options.gap_persistence_observations))
    if len(accepted) < 2 * persistence:
        return ()
    offsets = np.asarray([item.observed_offset_samples for item in accepted], dtype=np.float64)
    times = np.asarray([item.center_time_sec for item in accepted], dtype=np.float64)
    steps: list[RelativeOffsetStep] = []
    recent_indices = list(range(persistence))
    index = persistence
    while index <= len(accepted) - persistence:
        recent = np.asarray(recent_indices[-max(3, 4 * persistence) :], dtype=np.int64)
        if recent.size >= 2:
            old_coefficients = np.polyfit(times[recent], offsets[recent], 1)
        else:
            old_coefficients = np.asarray([0.0, float(np.median(offsets[recent]))])
        predicted = float(np.polyval(old_coefficients, times[index]))
        if abs(offsets[index] - predicted) <= options.gap_level_tolerance_samples:
            recent_indices.append(index)
            index += 1
            continue
        candidate_indices = np.arange(index, index + persistence, dtype=np.int64)
        if persistence >= 2:
            new_coefficients = np.polyfit(times[candidate_indices], offsets[candidate_indices], 1)
        else:
            new_coefficients = np.asarray(
                [old_coefficients[0], offsets[index] - old_coefficients[0] * times[index]]
            )
        new_fitted = np.polyval(new_coefficients, times[candidate_indices])
        stable_after = float(np.max(np.abs(offsets[candidate_indices] - new_fitted))) <= options.gap_level_tolerance_samples
        slope_tolerance = options.gap_level_tolerance_samples / max(options.step_seconds, 1e-9)
        compatible_slope = abs(float(new_coefficients[0] - old_coefficients[0])) <= slope_tolerance
        boundary_time = 0.5 * (times[index - 1] + times[index])
        endpoint_indices = np.asarray(
            [item for item in recent if accepted[int(item)].search_mode.startswith("endpoint_probe")],
            dtype=np.int64,
        )
        if endpoint_indices.size:
            before_level = float(np.median(offsets[endpoint_indices]))
            after_level = float(np.median(offsets[candidate_indices]))
        else:
            before_level = float(np.median(offsets[recent] + old_coefficients[0] * (boundary_time - times[recent])))
            after_level = float(np.median(offsets[candidate_indices] + old_coefficients[0] * (boundary_time - times[candidate_indices])))
        step = after_level - before_level
        adjacent_time_gap = times[index] - times[index - 1]
        if (
            abs(step) >= options.gap_min_step_samples
            and stable_after
            and compatible_slope
            and adjacent_time_gap <= 2.0 * options.step_seconds + 1e-9
        ):
            evidence_items = [accepted[int(item)] for item in np.concatenate((recent[-persistence:], candidate_indices))]
            correlation = min(item.peak_correlation for item in evidence_items)
            margin = min(item.peak_margin_fraction for item in evidence_items)
            steps.append(
                RelativeOffsetStep(
                    master_sample=int(round(boundary_time * fs)),
                    time_sec=float(boundary_time),
                    offset_step_samples=float(step),
                    missing_samples=int(round(abs(step))),
                    offset_before_samples=before_level,
                    offset_after_samples=after_level,
                    confidence="high" if correlation >= 0.1 and margin >= 0.05 else "medium",
                    evidence=(
                        f"legacy persistent {persistence}-window levels; "
                        f"min correlation {correlation:.3f}; min peak margin {margin:.3f}"
                    ),
                )
            )
            recent_indices = list(candidate_indices)
            index += persistence
        else:
            index += 1
    return tuple(steps)


def detect_unconfirmed_terminal_crop(
    observations: Sequence[SyncObservation],
    model: SyncModel,
    fs: float,
    options: SyncOptions,
) -> tuple[int, float, tuple[int, ...]] | None:
    """Return a conservative crop for a large, unconfirmed terminal level.

    Normal gap detection requires ``gap_persistence_observations`` observations
    on the new level.  At the recording end, fewer observations can remain.
    Rather than treating that incomplete evidence as a gap or failing the
    otherwise validated recording, exclude from the start of the preceding
    trusted window and everything after it. Correlation windows overlap, so a
    physical gap may begin inside that preceding window even while its peak
    still selects the old level. The returned crop sample is the first excluded
    master sample.
    """

    persistence = max(2, int(options.gap_persistence_observations))
    indexed = [
        (index, observation)
        for index, observation in enumerate(observations)
        if observation.accepted and not observation.search_mode.startswith("endpoint_probe")
    ]
    maximum_suffix = min(persistence - 1, len(indexed) - persistence)
    if maximum_suffix < 1:
        return None

    residuals = np.asarray(
        [
            observation.observed_offset_samples
            - model.offset_at_seconds(observation.center_time_sec)
            for _, observation in indexed
        ],
        dtype=np.float64,
    )
    tolerance = float(options.gap_level_tolerance_samples)
    for suffix_count in range(maximum_suffix, 0, -1):
        split = len(indexed) - suffix_count
        before_count = max(persistence, min(4 * persistence, split))
        before = residuals[split - before_count : split]
        after = residuals[split:]
        before_level = float(np.median(before))
        after_level = float(np.median(after))
        shift = after_level - before_level
        adjacent_gap = (
            indexed[split][1].center_time_sec
            - indexed[split - 1][1].center_time_sec
        )
        if (
            abs(shift) < options.gap_min_step_samples
            or float(np.ptp(before)) > 2.0 * tolerance
            or float(np.ptp(after)) > tolerance
            or adjacent_gap > 2.0 * options.step_seconds + 1e-9
        ):
            continue
        previous = indexed[split - 1][1]
        first_excluded = max(
            0,
            int(np.floor((previous.center_time_sec - options.window_seconds / 2.0) * fs)),
        )
        return first_excluded, shift, tuple(index for index, _ in indexed[split:])
    return None


def detect_isolated_offset_crop(
    observations: Sequence[SyncObservation],
    model: SyncModel,
    fs: float,
    options: SyncOptions,
) -> tuple[int, float, tuple[int, ...]] | None:
    """Return a safe tail crop for one unconfirmed interior excursion.

    Missing samples permanently change relative offset.  A single robust-fit
    outlier that immediately returns to the previous level is not a confirmed
    missing-data step. It can still represent multiple nearby physical events,
    so do not discard only that observation. Crop from the preceding trusted
    window and retain the event as a warning instead.
    """

    indexed = [
        (index, observation)
        for index, observation in enumerate(observations)
        if observation.accepted and not observation.search_mode.startswith("endpoint_probe")
    ]
    if len(indexed) < 5:
        return None
    residuals = np.asarray(
        [
            observation.observed_offset_samples
            - model.offset_at_seconds(observation.center_time_sec)
            for _, observation in indexed
        ],
        dtype=np.float64,
    )
    times = np.asarray(
        [observation.center_time_sec for _, observation in indexed],
        dtype=np.float64,
    )
    tolerance = float(options.gap_level_tolerance_samples)
    for position in range(2, len(indexed) - 2):
        observation = indexed[position][1]
        if observation.model_inlier:
            continue
        before = residuals[position - 2 : position]
        after = residuals[position + 1 : position + 3]
        surrounding = np.concatenate((before, after))
        surrounding_level = float(np.median(surrounding))
        shift = float(residuals[position] - surrounding_level)
        adjacent_gaps = np.diff(times[position - 2 : position + 3])
        if (
            abs(shift) >= options.gap_min_step_samples
            and float(np.ptp(surrounding)) <= 2.0 * tolerance
            and float(np.max(adjacent_gaps)) <= 2.0 * options.step_seconds + 1e-9
        ):
            previous = indexed[position - 1][1]
            first_excluded = max(
                0,
                int(np.floor((previous.center_time_sec - options.window_seconds / 2.0) * fs)),
            )
            return first_excluded, shift, (indexed[position][0],)
    return None


def verify_isolated_offset_alias(
    master_feature: np.ndarray,
    slave_feature: np.ndarray,
    observations: Sequence[SyncObservation],
    model: SyncModel,
    candidate: tuple[int, float, tuple[int, ...]],
    fs: float,
    options: SyncOptions,
) -> tuple[bool, str]:
    """Use short fixed-mapping comparisons to rescue a false lag excursion.

    This bounded recheck runs only for a one-window excursion. It compares the
    established model offset with the excursion offset in non-overlapping short
    windows around the event. Any reliable support for the excursion keeps the
    conservative tail crop; only repeated support for the established mapping
    on both sides permits the isolated observation to be discarded.
    """

    _, shift, observation_indices = candidate
    target = observations[observation_indices[0]]
    segment_samples = max(
        4,
        int(round(min(options.endpoint_probe_seconds, options.step_seconds) * fs)),
    )
    span_samples = max(segment_samples, int(round(options.window_seconds * fs)))
    target_sample = int(round(target.center_time_sec * fs))
    first_start = max(0, target_sample - span_samples)
    final_end = min(master_feature.size, target_sample + span_samples)
    old_before = 0
    old_after = 0
    candidate_support = 0
    candidate_run = 0
    maximum_candidate_run = 0
    ambiguous = 0

    def fixed_correlation(master_start: int, master_end: int, offset: int) -> float:
        slave_start = master_start + offset
        slave_end = master_end + offset
        if slave_start < 0 or slave_end > slave_feature.size:
            return float("nan")
        master_values = np.asarray(master_feature[master_start:master_end], dtype=np.float64)
        slave_values = np.asarray(slave_feature[slave_start:slave_end], dtype=np.float64)
        master_values -= float(np.mean(master_values))
        slave_values -= float(np.mean(slave_values))
        denominator = float(np.linalg.norm(master_values) * np.linalg.norm(slave_values))
        if not np.isfinite(denominator) or denominator <= np.finfo(float).eps:
            return float("nan")
        return float(np.dot(master_values, slave_values) / denominator)

    for master_start in range(first_start, final_end - segment_samples + 1, segment_samples):
        master_end = master_start + segment_samples
        center_time_sec = (master_start + segment_samples / 2.0) / fs
        established_offset = int(round(model.offset_at_seconds(center_time_sec)))
        excursion_offset = int(round(established_offset + shift))
        old_correlation = fixed_correlation(master_start, master_end, established_offset)
        excursion_correlation = fixed_correlation(master_start, master_end, excursion_offset)
        if not np.isfinite(old_correlation) or not np.isfinite(excursion_correlation):
            ambiguous += 1
            continue
        winner = max(old_correlation, excursion_correlation)
        margin = abs(old_correlation - excursion_correlation) / max(
            abs(winner), np.finfo(float).eps
        )
        if winner < options.min_peak_correlation or margin < options.min_peak_margin_fraction:
            ambiguous += 1
            candidate_run = 0
        elif excursion_correlation > old_correlation:
            candidate_support += 1
            candidate_run += 1
            maximum_candidate_run = max(maximum_candidate_run, candidate_run)
        elif center_time_sec < target.center_time_sec:
            old_before += 1
            candidate_run = 0
        else:
            old_after += 1
            candidate_run = 0

    verified = (
        old_before >= 2
        and old_after >= 2
        and candidate_support == 0
    )
    required_run = max(2, int(options.gap_persistence_observations))
    evidence = (
        f"short-window raw recheck: established offset {old_before + old_after} "
        f"({old_before} before/{old_after} after), excursion offset "
        f"{candidate_support} (max run {maximum_candidate_run}/{required_run}), "
        f"ambiguous {ambiguous}"
    )
    return verified, evidence


def with_localized_step(step: RelativeOffsetStep, master_sample: int, fs: float, evidence: str) -> RelativeOffsetStep:
    """Refine the physical boundary without moving the observation-state transition."""

    return replace(
        step,
        master_sample=int(master_sample),
        evidence=f"{step.evidence}; {evidence}",
    )


def localize_relative_offset_step(
    master_feature: np.ndarray,
    slave_feature: np.ndarray,
    step: RelativeOffsetStep,
    *,
    fs: float,
    options: SyncOptions,
) -> RelativeOffsetStep:
    """Refine a persistent step by maximizing old/new sample-wise agreement.

    A negative pair step is evaluated as a slave missing interval, so the new
    mapping begins ``missing_samples`` canonical samples after the boundary.
    A positive pair step is evaluated as a master missing interval and switches
    between adjacent compressed-master samples.  Attribution is still decided
    later from all pairs; localization does not make an ambiguous event safe.
    """

    approximate = int(step.master_sample)
    # A correlation window can keep selecting the old level until most of the
    # window lies beyond the loss.  Search by the full window plus one step so
    # refinement can reach the physical boundary rather than only the first
    # changed observation center.
    radius = max(4, int(round((options.window_seconds + options.step_seconds) * fs)))
    missing = int(step.missing_samples)
    left = max(0, approximate - radius)
    right = min(master_feature.size, approximate + radius + (missing if step.offset_step_samples < 0 else 0))
    candidates = np.arange(max(left + 2, approximate - radius), min(approximate + radius, right - 2), dtype=np.int64)
    if candidates.size < 4:
        return step

    indices = np.arange(left, right, dtype=np.int64)
    old_indices = indices + int(round(step.offset_before_samples))
    new_indices = indices + int(round(step.offset_after_samples))
    valid = (
        (old_indices >= 0)
        & (old_indices < slave_feature.size)
        & (new_indices >= 0)
        & (new_indices < slave_feature.size)
    )
    if np.count_nonzero(valid) < max(16, int(0.5 * indices.size)):
        return step

    master_values = np.asarray(master_feature[indices], dtype=np.float64)
    old_values = np.zeros(indices.size, dtype=np.float64)
    new_values = np.zeros(indices.size, dtype=np.float64)
    old_values[valid] = np.asarray(slave_feature[old_indices[valid]], dtype=np.float64)
    new_values[valid] = np.asarray(slave_feature[new_indices[valid]], dtype=np.float64)

    def standardized(values: np.ndarray) -> np.ndarray:
        selected = values[valid]
        scale = float(np.std(selected))
        if not np.isfinite(scale) or scale <= np.finfo(float).eps:
            return np.zeros_like(values)
        return (values - float(np.mean(selected))) / scale

    master_z = standardized(master_values)
    old_score = master_z * standardized(old_values)
    new_score = master_z * standardized(new_values)
    # Coordinates unavailable under either mapping carry no evidence. A large
    # negative sentinel biases the cumulative objective toward the search edge
    # because early candidates avoid summing that sentinel, which can move an
    # early real boundary all the way to sample zero.
    old_score[~valid] = 0.0
    new_score[~valid] = 0.0
    prefix_old = np.concatenate(([0.0], np.cumsum(old_score)))
    prefix_new = np.concatenate(([0.0], np.cumsum(new_score)))
    candidate_scores = np.empty(candidates.size, dtype=np.float64)
    for candidate_index, boundary in enumerate(candidates):
        split = int(boundary - left)
        new_start = split + (missing if step.offset_step_samples < 0 else 0)
        if new_start >= indices.size:
            candidate_scores[candidate_index] = -np.inf
            continue
        candidate_scores[candidate_index] = (
            prefix_old[split]
            + prefix_new[-1]
            - prefix_new[new_start]
        )
    best_index = int(np.argmax(candidate_scores))
    if not np.isfinite(candidate_scores[best_index]):
        return step
    edge_margin = max(2, min(100, int(round(0.01 * candidates.size))))
    if best_index < edge_margin or best_index >= candidates.size - edge_margin:
        return replace(
            step,
            confidence="medium",
            evidence=(
                f"{step.evidence}; boundary refinement rejected because the optimum "
                "was at the search edge"
            ),
        )
    best = int(candidates[best_index])
    return with_localized_step(
        step,
        best,
        fs,
        f"sample-wise boundary refinement from {approximate} to {best}",
    )


def infer_device_gaps(
    pairs: Sequence[SyncPairResult],
    *,
    device_count: int,
    master_index: int,
    fs: float,
    options: SyncOptions,
) -> tuple[list[DeviceGap], list[str]]:
    """Attribute only sign-identifiable missing-only events.

    The pair convention is ``slave source = master source + offset``.  A
    slave loss therefore produces a negative step in that pair; a master loss
    produces a contemporaneous positive step in every pair.  Other sign
    patterns are retained as unresolved rather than silently zero-filled.
    Device indices are one-based in reports and gap records.
    """

    if device_count < 3:
        if any(pair.model.offset_steps for pair in pairs):
            return [], ["gap attribution requires at least three devices"]
        return [], []

    events: list[tuple[float, SyncPairResult, RelativeOffsetStep]] = []
    for pair in pairs:
        for step in pair.model.offset_steps:
            events.append((step.master_sample / fs, pair, step))
    events.sort(key=lambda item: item[0])
    if not events:
        return [], []

    tolerance = max(0.0, options.gap_event_time_tolerance_seconds)
    clusters: list[list[tuple[float, SyncPairResult, RelativeOffsetStep]]] = []
    for event in events:
        if not clusters or event[0] - clusters[-1][0][0] > tolerance:
            clusters.append([event])
        else:
            clusters[-1].append(event)

    gaps: list[DeviceGap] = []
    unresolved: list[str] = []
    cumulative_master_missing = 0
    expected_pairs = {pair.slave_index for pair in pairs}
    for cluster in clusters:
        by_slave: dict[int, RelativeOffsetStep] = {}
        duplicate_pair = False
        for _time, pair, step in cluster:
            if pair.slave_index in by_slave:
                duplicate_pair = True
            by_slave[pair.slave_index] = step
        representative_sample = int(round(np.median([item[2].master_sample for item in cluster])))
        canonical_start = representative_sample + cumulative_master_missing
        steps = list(by_slave.values())
        negative = [step for step in steps if step.offset_step_samples < 0]
        positive = [step for step in steps if step.offset_step_samples > 0]

        if not duplicate_pair and len(negative) == 1 and not positive and len(by_slave) == 1:
            pair = next(item[1] for item in cluster if item[2] is negative[0])
            size = negative[0].missing_samples
            gaps.append(
                DeviceGap(
                    device_index=pair.slave_index,
                    canonical_start_sample=canonical_start,
                    missing_samples=size,
                    duration_ms=1000.0 * size / fs,
                    confidence=negative[0].confidence,
                    evidence=f"negative offset step in slave pair; {negative[0].evidence}",
                )
            )
            continue

        if (
            not duplicate_pair
            and not negative
            and set(by_slave) == expected_pairs
            and len(positive) == len(expected_pairs)
        ):
            sizes = np.asarray([step.missing_samples for step in positive], dtype=np.float64)
            if float(np.ptp(sizes)) <= options.gap_level_tolerance_samples:
                size = int(round(float(np.median(sizes))))
                gaps.append(
                    DeviceGap(
                        device_index=master_index + 1,
                        canonical_start_sample=canonical_start,
                        missing_samples=size,
                        duration_ms=1000.0 * size / fs,
                        confidence=(
                            "high" if all(step.confidence == "high" for step in positive) else "medium"
                        ),
                        evidence="contemporaneous positive offset step in every slave pair",
                    )
                )
                cumulative_master_missing += size
                continue

        description = ", ".join(
            f"slave {pair.slave_index}: {step.offset_step_samples:+.1f}"
            for _time, pair, step in cluster
        )
        unresolved.append(
            f"unresolved offset event near master sample {representative_sample} ({description})"
        )
    return gaps, unresolved


def gap_summary(
    gaps: Sequence[DeviceGap],
    *,
    device_count: int,
    canonical_samples: int,
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for device_index in range(1, device_count + 1):
        selected = [gap for gap in gaps if gap.device_index == device_index]
        missing = sum(gap.missing_samples for gap in selected)
        summaries.append(
            {
                "device_index": device_index,
                "gap_count": len(selected),
                "missing_samples": missing,
                "missing_duration_ms": sum(gap.duration_ms for gap in selected),
                "missing_fraction": missing / max(1, canonical_samples),
                "longest_gap_samples": max((gap.missing_samples for gap in selected), default=0),
            }
        )
    return summaries


def canonicalize_master_sample(
    raw_master_sample: int,
    gaps: Sequence[DeviceGap],
    *,
    master_device_index: int,
) -> int:
    """Map one compressed raw-master boundary onto the canonical gap axis."""

    canonical_sample = int(raw_master_sample)
    for gap in sorted(gaps, key=lambda item: item.canonical_start_sample):
        if gap.device_index != master_device_index:
            continue
        if gap.canonical_start_sample > canonical_sample:
            break
        canonical_sample += gap.missing_samples
    return canonical_sample
