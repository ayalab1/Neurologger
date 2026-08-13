"""Pure vector mappings and clean fast-path construction for analog timelines."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Sequence
import hashlib
import math
from numbers import Integral

import numpy as np

from .models import (
    AnalogSyncAnchor,
    AnalogSyncSegment,
    DeviceClockPrior,
    validate_analog_sync_segments,
)


INTEGER_MAPPING_TOLERANCE_ROWS = 1e-6


def _prior_support_reference(prior: DeviceClockPrior) -> str:
    """Return a compact stable reference to support IDs stored on the prior."""

    digest = hashlib.sha256("\n".join(prior.support_ids).encode("utf-8")).hexdigest()[:16]
    return f"{prior.method}:support_count={len(prior.support_ids)}:sha256={digest}"


def validate_analog_segment_collection(
    segments: Iterable[AnalogSyncSegment], *, device_index: int | None = None
) -> tuple[AnalogSyncSegment, ...]:
    return validate_analog_sync_segments(segments, device_index=device_index)


def _canonical_rows(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows)
    if values.ndim != 1:
        raise ValueError("canonical rows must be one-dimensional")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("canonical rows must be integers")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0):
        raise ValueError("canonical rows must be non-negative")
    if values.size > 1 and np.any(np.diff(values) <= 0):
        raise ValueError("canonical rows must be strictly increasing")
    return values


def map_canonical_rows(
    segments: Iterable[AnalogSyncSegment],
    canonical_rows: np.ndarray,
    *,
    raw_row_count: int,
    interpolation_half_width: int = 0,
    device_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map canonical rows through publishable analog segments only.

    Returns ``(raw_rows, valid, segment_ids)``.  Unsupported rows are ``NaN``,
    ``False``, and ``-1``.  ``segment_ids`` are deterministic zero-based
    indices in the supplied, validated segment collection.  Interpolation
    support is checked against both the file and a single segment, so no
    kernel can cross an invalid gap.
    """

    ordered = validate_analog_segment_collection(segments, device_index=device_index)
    return _map_validated_canonical_rows(
        ordered,
        canonical_rows,
        raw_row_count=raw_row_count,
        interpolation_half_width=interpolation_half_width,
    )


def _map_validated_canonical_rows(
    ordered: tuple[AnalogSyncSegment, ...],
    canonical_rows: np.ndarray,
    *,
    raw_row_count: int,
    interpolation_half_width: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map with a collection already validated by the owning renderer."""

    rows = _canonical_rows(canonical_rows)
    if not isinstance(raw_row_count, Integral) or raw_row_count <= 0:
        raise ValueError("raw_row_count must be a positive integer")
    if not isinstance(interpolation_half_width, Integral) or interpolation_half_width < 0:
        raise ValueError("interpolation_half_width must be a non-negative integer")

    mapped = np.full(rows.shape, np.nan, dtype=np.float64)
    valid = np.zeros(rows.shape, dtype=bool)
    segment_ids = np.full(rows.shape, -1, dtype=np.int64)
    support = int(interpolation_half_width)
    if rows.size:
        first_segment = bisect_right(
            ordered,
            int(rows[0]),
            key=lambda segment: segment.canonical_end_row,
        )
        stop_segment = bisect_right(
            ordered,
            int(rows[-1]),
            key=lambda segment: segment.canonical_start_row,
        )
    else:
        first_segment = stop_segment = 0
    for segment_id in range(first_segment, stop_segment):
        segment = ordered[segment_id]
        if not segment.is_publishable:
            continue
        mask = (rows >= segment.canonical_start_row) & (rows < segment.canonical_end_row)
        if not np.any(mask):
            continue
        values = segment.raw_scale * rows[mask] + segment.raw_intercept_rows
        mapped[mask] = values
        segment_ids[mask] = segment_id
        nearest = np.rint(values)
        integral = np.abs(values - nearest) <= INTEGER_MAPPING_TOLERANCE_ROWS
        # Exact samples require one source row.  Fractional interpolation is
        # intentionally forbidden close to either raw support boundary.
        if support == 0:
            # A fractional source coordinate without an interpolation kernel
            # has no meaningful sample value.  Discrete lanes use this mode.
            file_supported = integral & (nearest >= 0) & (nearest < raw_row_count)
            segment_supported = (
                integral
                & (nearest >= segment.raw_start_row)
                & (nearest < segment.raw_end_row)
            )
        else:
            file_supported = np.where(
                integral,
                (nearest >= 0) & (nearest < raw_row_count),
                (values >= support - 1) & (values < raw_row_count - support),
            )
            segment_supported = np.where(
                integral,
                (nearest >= segment.raw_start_row) & (nearest < segment.raw_end_row),
                (values >= segment.raw_start_row + support - 1)
                & (values < segment.raw_end_row - support),
            )
        valid[mask] = np.isfinite(values) & file_supported & segment_supported

    valid_values = mapped[valid]
    if not np.all(np.isfinite(valid_values)):
        raise ValueError("valid analog mapping contains non-finite raw coordinates")
    if valid_values.size > 1 and np.any(np.diff(valid_values) <= 0):
        raise ValueError("valid analog mapping is not globally strictly monotone")
    segment_ids[~valid] = -1
    return mapped, valid, segment_ids


def map_raw_rows_to_canonical(
    segments: Iterable[AnalogSyncSegment],
    raw_rows: np.ndarray,
    *,
    device_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invert unique publishable mappings for raw-row anchor attribution.

    The return values are ``(canonical_rows, valid, segment_ids)``.  Canonical
    coordinates remain floating point so PC-time can compose this mapping and
    quantize only once on its final neural grid.  A raw row in a skipped
    insertion, an unsupported segment, or a non-unique overlap remains
    ``NaN, False, -1``.  Unlike the
    forward API, raw rows need not be ordered because clock-anchor extraction
    may retain a sparse event sequence.
    """

    ordered = validate_analog_segment_collection(segments, device_index=device_index)
    source = np.asarray(raw_rows, dtype=np.float64)
    if source.ndim != 1 or not np.all(np.isfinite(source)) or np.any(source < 0):
        raise ValueError("raw rows must be finite non-negative one-dimensional values")
    canonical = np.full(source.shape, np.nan, dtype=np.float64)
    valid = np.zeros(source.shape, dtype=bool)
    segment_ids = np.full(source.shape, -1, dtype=np.int64)
    matches = np.zeros(source.shape, dtype=np.uint8)
    candidate_rows = np.full(source.shape, np.nan, dtype=np.float64)
    candidate_ids = np.full(source.shape, -1, dtype=np.int64)
    for segment_id, segment in enumerate(ordered):
        if not segment.is_publishable:
            continue
        candidate = (source - segment.raw_intercept_rows) / segment.raw_scale
        in_segment = (
            (source >= segment.raw_start_row)
            & (source < segment.raw_end_row)
            & (candidate >= segment.canonical_start_row)
            & (candidate < segment.canonical_end_row)
        )
        candidate_rows[in_segment] = candidate[in_segment]
        candidate_ids[in_segment] = segment_id
        matches[in_segment] += 1
    unique = matches == 1
    canonical[unique] = candidate_rows[unique]
    segment_ids[unique] = candidate_ids[unique]
    valid[unique] = True
    return canonical, valid, segment_ids


def _validate_intervals(
    intervals: Sequence[tuple[int, int]], *, name: str, upper_bound: int
) -> tuple[tuple[int, int], ...]:
    normalized = tuple((int(start), int(end)) for start, end in intervals)
    previous_end = -1
    for start, end in normalized:
        if start < 0 or end <= start or end > upper_bound:
            raise ValueError(f"{name} must contain bounded non-empty half-open intervals")
        if start < previous_end:
            raise ValueError(f"{name} must be sorted and non-overlapping")
        previous_end = end
    return normalized


def _subtract_intervals(
    start: int, end: int, excluded: Sequence[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    runs: list[tuple[int, int]] = []
    cursor = start
    for lower, upper in excluded:
        if upper <= cursor or lower >= end:
            continue
        if lower > cursor:
            runs.append((cursor, min(lower, end)))
        cursor = max(cursor, upper)
        if cursor >= end:
            break
    if cursor < end:
        runs.append((cursor, end))
    return tuple(run for run in runs if run[1] > run[0])


def _clip_canonical_run_to_raw_support(
    start: int,
    end: int,
    *,
    raw_scale: float,
    raw_intercept_rows: float,
    raw_row_count: int,
) -> tuple[int, int] | None:
    """Intersect an integer canonical run with finite raw-row support."""

    supported_start = math.ceil((-raw_intercept_rows) / raw_scale)
    supported_end = math.ceil((raw_row_count - raw_intercept_rows) / raw_scale)
    clipped_start = max(int(start), int(supported_start))
    clipped_end = min(int(end), int(supported_end))
    return (clipped_start, clipped_end) if clipped_end > clipped_start else None


def build_clean_analog_segments(
    prior: DeviceClockPrior,
    *,
    canonical_start_row: int,
    canonical_end_row: int,
    raw_row_count: int,
    excluded_canonical_intervals: Sequence[tuple[int, int]] = (),
    excluded_raw_intervals: Sequence[tuple[int, int]] = (),
    confidence: str | None = None,
    evidence: str = "clean analog integrity plus smooth neural clock prior",
) -> tuple[AnalogSyncSegment, ...]:
    """Build the clean fast path, splitting explicit analog exclusions.

    Exclusions are supplied as independent analog-integrity decisions.  Raw
    exclusions are conservatively projected through the continuous prior only
    to prevent a mapping from using them; no neural discontinuity is imported.
    Each retained run receives endpoint anchors tied to the clean integrity
    evidence plus the smooth clock prior.  A one-row run is deliberately not
    publishable because it cannot have two independent endpoint anchors.
    """

    if not isinstance(canonical_start_row, Integral) or not isinstance(canonical_end_row, Integral):
        raise ValueError("canonical bounds must be integers")
    if canonical_start_row < 0 or canonical_end_row <= canonical_start_row:
        raise ValueError("canonical bounds must be a non-empty half-open interval")
    if not isinstance(raw_row_count, Integral) or raw_row_count <= 0:
        raise ValueError("raw_row_count must be a positive integer")
    if confidence is not None and confidence not in {"high", "medium"}:
        raise ValueError("clean fast-path confidence must be high or medium")
    effective_confidence = confidence or prior.confidence
    support_reference = _prior_support_reference(prior)
    scale, intercept = prior.analog_affine()
    canonical_excluded = list(
        _validate_intervals(
            excluded_canonical_intervals,
            name="excluded_canonical_intervals",
            upper_bound=canonical_end_row,
        )
    )
    raw_excluded = _validate_intervals(
        excluded_raw_intervals, name="excluded_raw_intervals", upper_bound=int(raw_row_count)
    )
    for raw_start, raw_end in raw_excluded:
        # k belongs in the exclusion iff raw_start <= scale*k+intercept < raw_end.
        lower = math.ceil((raw_start - intercept) / scale)
        upper = math.ceil((raw_end - intercept) / scale)
        lower = max(int(canonical_start_row), lower)
        upper = min(int(canonical_end_row), upper)
        if upper > lower:
            canonical_excluded.append((lower, upper))
    canonical_excluded.sort()
    merged_excluded: list[tuple[int, int]] = []
    for start, end in canonical_excluded:
        if end <= canonical_start_row or start >= canonical_end_row:
            continue
        start = max(start, int(canonical_start_row))
        end = min(end, int(canonical_end_row))
        if merged_excluded and start <= merged_excluded[-1][1]:
            merged_excluded[-1] = (merged_excluded[-1][0], max(merged_excluded[-1][1], end))
        else:
            merged_excluded.append((start, end))

    segments: list[AnalogSyncSegment] = []
    for run_start, run_end in _subtract_intervals(
        int(canonical_start_row), int(canonical_end_row), merged_excluded
    ):
        clipped = _clip_canonical_run_to_raw_support(
            run_start,
            run_end,
            raw_scale=scale,
            raw_intercept_rows=intercept,
            raw_row_count=int(raw_row_count),
        )
        if clipped is None:
            continue
        run_start, run_end = clipped
        first_raw = scale * run_start + intercept
        last_raw = scale * (run_end - 1) + intercept
        raw_start = max(0, int(math.floor(first_raw)))
        # Fractional interpolation at the final canonical row can require the
        # upper neighbouring raw row.  Keep it inside this segment's support.
        raw_end = min(int(raw_row_count), int(math.ceil(last_raw)) + 1)
        anchor_rows = (run_start,) if run_end - run_start == 1 else (run_start, run_end - 1)
        anchors = tuple(
            AnalogSyncAnchor(
                canonical_row=row,
                raw_row=scale * row + intercept,
                verified=prior.is_publishable_evidence,
                confidence=effective_confidence,
                verification_source=(
                    support_reference
                ),
                evidence=f"{evidence}; prior={support_reference}",
            )
            for row in anchor_rows
        )
        segments.append(
            AnalogSyncSegment(
                device_index=prior.device_index,
                canonical_start_row=run_start,
                canonical_end_row=run_end,
                raw_start_row=raw_start,
                raw_end_row=raw_end,
                raw_scale=scale,
                raw_intercept_rows=intercept,
                anchors=anchors,
                residual_rms_rows=prior.residual_rms_rows,
                residual_max_abs_rows=prior.residual_max_abs_rows,
                confidence=effective_confidence,
                start_transition=("recording_start" if run_start == canonical_start_row else "integrity_exclusion"),
                end_transition=("recording_end" if run_end == canonical_end_row else "integrity_exclusion"),
                publishable=prior.is_publishable_evidence and len(anchors) >= 2,
                evidence=f"{evidence}; prior={support_reference}",
            )
        )
    return validate_analog_sync_segments(segments, device_index=prior.device_index)


def _decision_field(decision: object, *names: str, default: object = None) -> object:
    if isinstance(decision, dict):
        for name in names:
            if name in decision:
                return decision[name]
    for name in names:
        if hasattr(decision, name):
            return getattr(decision, name)
    return default


def _normalized_decision(decision: object) -> tuple[str, int, int, float, str, str]:
    """Read the intentionally small integrity-event protocol without importing it."""

    kind = _decision_field(decision, "kind")
    start = _decision_field(decision, "raw_start_row", "raw_start")
    end = _decision_field(decision, "raw_end_row", "raw_end")
    displacement = _decision_field(
        decision, "displacement_rows", "displacement", "tick_displacement", default=0.0
    )
    if displacement is None:
        displacement = 0.0
    confidence = _decision_field(decision, "confidence", default="unresolved")
    evidence = _decision_field(decision, "evidence", default="")
    if not isinstance(kind, str):
        raise ValueError("analog integrity decision requires string kind")
    if not isinstance(start, Integral) or not isinstance(end, Integral) or start < 0 or end <= start:
        raise ValueError("analog integrity decision requires non-empty raw_start/raw_end")
    if not isinstance(displacement, (int, float)) or not math.isfinite(displacement):
        raise ValueError("analog integrity decision displacement must be finite")
    if confidence not in {"high", "medium", "low", "unresolved"}:
        raise ValueError("analog integrity decision confidence is unsupported")
    return kind, int(start), int(end), float(displacement), confidence, str(evidence)


def _segment_from_affine_run(
    prior: DeviceClockPrior,
    *,
    canonical_start_row: int,
    canonical_end_row: int,
    raw_scale: float,
    raw_intercept_rows: float,
    raw_row_count: int,
    start_transition: str,
    end_transition: str,
    evidence: str,
) -> AnalogSyncSegment | None:
    """Materialize one verified run, preserving the prior's provenance IDs."""

    clipped = _clip_canonical_run_to_raw_support(
        canonical_start_row,
        canonical_end_row,
        raw_scale=raw_scale,
        raw_intercept_rows=raw_intercept_rows,
        raw_row_count=raw_row_count,
    )
    if clipped is None:
        return None
    canonical_start_row, canonical_end_row = clipped
    first_raw = raw_scale * canonical_start_row + raw_intercept_rows
    last_raw = raw_scale * (canonical_end_row - 1) + raw_intercept_rows
    raw_start = max(0, int(math.floor(first_raw)))
    raw_end = min(raw_row_count, int(math.ceil(last_raw)) + 1)
    anchor_rows = (
        (canonical_start_row,)
        if canonical_end_row - canonical_start_row == 1
        else (canonical_start_row, canonical_end_row - 1)
    )
    support_reference = _prior_support_reference(prior)
    anchor_evidence = f"{evidence}; prior={support_reference}"
    anchors = tuple(
        AnalogSyncAnchor(
            canonical_row=row,
            raw_row=raw_scale * row + raw_intercept_rows,
            verified=prior.is_publishable_evidence,
            confidence=prior.confidence,
            verification_source=support_reference,
            evidence=anchor_evidence,
        )
        for row in anchor_rows
    )
    return AnalogSyncSegment(
        device_index=prior.device_index,
        canonical_start_row=canonical_start_row,
        canonical_end_row=canonical_end_row,
        raw_start_row=raw_start,
        raw_end_row=raw_end,
        raw_scale=raw_scale,
        raw_intercept_rows=raw_intercept_rows,
        anchors=anchors,
        residual_rms_rows=prior.residual_rms_rows,
        residual_max_abs_rows=prior.residual_max_abs_rows,
        confidence=prior.confidence,
        start_transition=start_transition,
        end_transition=end_transition,
        publishable=prior.is_publishable_evidence and len(anchors) >= 2,
        evidence=anchor_evidence,
    )


def build_event_driven_analog_segments(
    prior: DeviceClockPrior,
    *,
    canonical_start_row: int,
    canonical_end_row: int,
    raw_row_count: int,
    decisions: Sequence[object],
    reacquisition_priors: Sequence[tuple[int, DeviceClockPrior]] = (),
) -> tuple[AnalogSyncSegment, ...]:
    """Convert confirmed integrity decisions into independent analog runs.

    The duck-typed decision protocol is ``kind``, ``raw_start``/``raw_end``,
    ``displacement`` and ``confidence``.  Persistent counter phase
    ``d = tick - row`` changes the post-boundary affine intercept by ``-d``:
    a positive missing displacement creates a canonical invalid gap, whereas
    a negative insertion skips raw source rows.  Repeat overwrite/reorder
    decisions exclude only their destinations and retain the tail affine.

    An unresolved or low-confidence persistent boundary ends automatic
    publication.  A later result is allowed only through an explicit
    independently supported ``reacquisition_priors`` entry; no guessed step
    is propagated.
    """

    if not isinstance(canonical_start_row, Integral) or not isinstance(canonical_end_row, Integral):
        raise ValueError("canonical bounds must be integers")
    if canonical_start_row < 0 or canonical_end_row <= canonical_start_row:
        raise ValueError("canonical bounds must be a non-empty half-open interval")
    if not isinstance(raw_row_count, Integral) or raw_row_count <= 0:
        raise ValueError("raw_row_count must be a positive integer")
    normalized = [_normalized_decision(decision) for decision in decisions]
    if any(end > raw_row_count for _, _, end, _, _, _ in normalized):
        raise ValueError("integrity decision exceeds raw row count")
    if any(later[1] < earlier[1] for earlier, later in zip(normalized, normalized[1:])):
        raise ValueError("integrity decisions must be ordered by raw_start")
    if any(later[1] < earlier[2] for earlier, later in zip(normalized, normalized[1:])):
        raise ValueError("integrity decisions must not overlap in raw rows")

    scale, intercept = prior.analog_affine()
    current_start = int(canonical_start_row)
    published: list[AnalogSyncSegment] = []
    stopped_at: int | None = None

    def append_current(end: int, *, transition: str, evidence: str) -> None:
        nonlocal current_start
        segment = _segment_from_affine_run(
            prior,
            canonical_start_row=current_start,
            canonical_end_row=end,
            raw_scale=scale,
            raw_intercept_rows=intercept,
            raw_row_count=int(raw_row_count),
            start_transition=("recording_start" if current_start == canonical_start_row else transition),
            end_transition=transition,
            evidence=evidence,
        )
        if segment is not None:
            published.append(segment)

    local_kinds = {"repeat_overwrite", "reorder", "temporary_excursion"}
    insertion_kinds = {"insertion", "repeat_insertion", "extra"}
    missing_kinds = {"missing"}
    unresolved_kinds = {"unresolved", "unresolved_boundary", "counter_corruption"}
    for kind, raw_start, raw_end, displacement, confidence, evidence in normalized:
        if stopped_at is not None:
            break
        # Insertion decisions name the excess raw destination interval.  The
        # canonical timeline therefore stops at its first raw row and resumes
        # after the interval by increasing the source intercept.  Using
        # ``raw_end`` here would publish the inserted rows and skip the same
        # number of valid rows after them.
        boundary_raw = raw_start
        boundary = int(math.ceil((boundary_raw - intercept) / scale))
        boundary = min(max(boundary, current_start), int(canonical_end_row))
        destination_end = int(math.ceil((raw_end - intercept) / scale))
        destination_end = min(max(destination_end, boundary), int(canonical_end_row))
        confirmed = confidence in {"high", "medium"}
        detail = f"{kind} raw[{raw_start},{raw_end}): {evidence}".rstrip()
        if kind in missing_kinds and confirmed and displacement > 0:
            append_current(boundary, transition="integrity_missing", evidence=detail)
            gap_rows = max(1, int(math.ceil(displacement / scale)))
            current_start = min(int(canonical_end_row), boundary + gap_rows)
            intercept -= displacement
            continue
        if kind in insertion_kinds and confirmed:
            length = (
                raw_end - raw_start
                if kind == "repeat_insertion"
                else abs(displacement)
                if displacement < 0
                else raw_end - raw_start
            )
            if length <= 0:
                raise ValueError("confirmed insertion requires negative displacement or non-empty raw range")
            append_current(boundary, transition="integrity_insertion", evidence=detail)
            current_start = boundary
            intercept += length
            continue
        if kind in local_kinds and confirmed:
            append_current(boundary, transition=kind, evidence=detail)
            current_start = destination_end
            continue
        if kind in unresolved_kinds or not confirmed or kind in missing_kinds | insertion_kinds:
            append_current(boundary, transition="unresolved_boundary", evidence=detail)
            stopped_at = max(boundary, destination_end)
            break
        if kind == "sensor_stall":
            # Sensor payload validity is independent from temporal authority.
            continue
        raise ValueError(f"unsupported analog integrity decision kind: {kind}")

    if stopped_at is None:
        append_current(int(canonical_end_row), transition="recording_end", evidence="verified clean tail")
    else:
        prior_entries = tuple(reacquisition_priors)
        previous_start = stopped_at
        for index, (reacquisition_start, reacquisition_prior) in enumerate(prior_entries):
            if not isinstance(reacquisition_start, Integral) or reacquisition_start < stopped_at:
                raise ValueError("reacquisition start must be an integer at/after unresolved boundary")
            if reacquisition_prior.device_index != prior.device_index:
                raise ValueError("reacquisition prior must describe the same device")
            if index and reacquisition_start <= previous_start:
                raise ValueError("reacquisition priors must have strictly increasing starts")
            next_start = (
                int(prior_entries[index + 1][0])
                if index + 1 < len(prior_entries)
                else int(canonical_end_row)
            )
            if reacquisition_start >= next_start or reacquisition_start >= canonical_end_row:
                continue
            published.extend(
                build_clean_analog_segments(
                    reacquisition_prior,
                    canonical_start_row=int(reacquisition_start),
                    canonical_end_row=min(next_start, int(canonical_end_row)),
                    raw_row_count=int(raw_row_count),
                    evidence="explicit independent reacquisition after unresolved boundary",
                )
            )
            previous_start = int(reacquisition_start)

    return validate_analog_sync_segments(published, device_index=prior.device_index)
