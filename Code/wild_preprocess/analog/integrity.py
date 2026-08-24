"""Read-only, bounded-memory integrity checks for raw WILD ``analogin.dat``.

The counter is evidence about device-local timeline continuity, not a source of
cross-device synchronization.  In particular, a counter phase excursion is
only called a repeat when the affected complete frames are subsequently
confirmed equal.  This keeps normal IMU sample holds and a stationary animal
from becoming false duplication events.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np


IntegrityKind = Literal[
    "missing",
    "insertion",
    "repeat_insertion",
    "repeat_overwrite",
    "reorder",
    "temporary_excursion",
    "counter_corruption",
    "sensor_stall",
    "imu_all_zero",
    "imu_saturation",
    "unresolved",
]

TIMELINE_KINDS = frozenset(
    {
        "missing",
        "insertion",
        "repeat_insertion",
        "repeat_overwrite",
        "reorder",
        "temporary_excursion",
        "counter_corruption",
        "unresolved",
    }
)
RAW_SUPPORT_BLOCKING_KINDS = frozenset(
    {
        "missing",
        "insertion",
        "repeat_insertion",
        "repeat_overwrite",
        "reorder",
        "temporary_excursion",
        "unresolved",
    }
)

# These findings invalidate IMU-derived products only.  They deliberately do
# not remove source support for other analog lanes or packed PC-clock fitting.
IMU_MODALITY_INVALID_KINDS = frozenset({"sensor_stall", "imu_all_zero", "imu_saturation"})


@dataclass(frozen=True)
class AnalogIntegrityEvent:
    """One deterministic raw-row finding.

    Bounds are half-open raw row/tick intervals.  Payload-only events such as
    ``sensor_stall`` deliberately remain separate from timeline events: they
    should invalidate an IMU product, not automatically invalidate every
    analog lane at the same time.
    """

    kind: IntegrityKind
    raw_start_row: int
    raw_end_row: int
    tick_start: int | None
    tick_end: int | None
    affected_lanes: tuple[int, ...]
    displacement_rows: int | None
    confidence: Literal["high", "medium", "low", "unresolved"]
    evidence: str
    device_index: int = 1

    def __post_init__(self) -> None:
        if self.raw_start_row < 0 or self.raw_end_row <= self.raw_start_row:
            raise ValueError("raw event bounds must be non-empty and non-negative")
        if self.device_index < 1:
            raise ValueError("device index must be one-based and positive")
        if self.tick_start is not None and self.tick_end is not None and self.tick_end < self.tick_start:
            raise ValueError("tick event bounds must be ordered")
        if tuple(sorted(set(self.affected_lanes))) != self.affected_lanes:
            raise ValueError("affected lanes must be sorted and unique")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AnalogIntegrityMetrics:
    row_count: int
    channel_count: int
    counter_wrap_count: int
    counter_nonunit_delta_count: int
    counter_phase_run_count: int
    confirmed_repeat_rows: int
    imu_update_count: int
    imu_median_update_rows: float | None
    imu_max_update_rows: int | None
    imu_all_zero_rows: int
    imu_rail_rows: int
    imu_longest_static_rows: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AnalogIntegrityResult:
    """The scanner output; source data are never changed."""

    events: tuple[AnalogIntegrityEvent, ...]
    metrics: AnalogIntegrityMetrics
    device_index: int = 1

    def __post_init__(self) -> None:
        if self.device_index < 1:
            raise ValueError("device index must be one-based and positive")

    @property
    def timeline_events(self) -> tuple[AnalogIntegrityEvent, ...]:
        return tuple(event for event in self.events if event.kind in TIMELINE_KINDS)

    @property
    def clean(self) -> bool:
        return not self.events

    def valid_raw_support_runs(self) -> tuple[tuple[int, int], ...]:
        """Return raw-row support that packed-clock fitting may safely use.

        This is deliberately timeline-only.  A stalled accelerometer does not
        make an otherwise sound packed PC-clock observation unusable, whereas
        an overwrite/reorder or uncertain counter phase does.
        """

        blocked = sorted(
            (event.raw_start_row, event.raw_end_row)
            for event in self.timeline_events
            if event.kind in RAW_SUPPORT_BLOCKING_KINDS
        )
        runs: list[tuple[int, int]] = []
        cursor = 0
        for start, end in blocked:
            if start > cursor:
                runs.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < self.metrics.row_count:
            runs.append((cursor, self.metrics.row_count))
        return tuple(runs)

    def to_dict(self) -> dict[str, object]:
        return {
            "device_index": self.device_index,
            "events": [event.to_dict() for event in self.events],
            "metrics": self.metrics.to_dict(),
        }


def _counter_phase_runs(
    frames: np.ndarray,
    *,
    counter_lane: int,
    modulus: int,
    chunk_rows: int,
) -> tuple[tuple[tuple[int, int, int], ...], int, int, int]:
    """Return counter phase runs without materialising a recording-length vector."""

    if modulus <= 2 or modulus % 2:
        raise ValueError("counter modulus must be a positive even integer above two")
    if frames.shape[0] == 0:
        return (), 0, 0, 0
    half = modulus // 2
    initial_tick: int | None = None
    previous_raw: int | None = None
    wraps = 0
    nonunit = 0
    active_start = 0
    active_phase = 0
    runs: list[tuple[int, int, int]] = []
    for chunk_start in range(0, frames.shape[0], chunk_rows):
        raw = np.asarray(
            frames[chunk_start : min(frames.shape[0], chunk_start + chunk_rows), counter_lane],
            dtype=np.uint16,
        ).astype(np.int64, copy=False)
        if initial_tick is None:
            initial_tick = int(raw[0])
            previous_raw = int(raw[0])
            # The first row defines phase zero.  Subsequent rows advance from
            # it; treating that initial value as a delta would make it appear
            # to be a non-unit counter transition.
            raw_delta = np.diff(raw)
            deltas = ((raw_delta + half) % modulus) - half
            phase = np.empty(raw.size, dtype=np.int64)
            phase[0] = 0
            if deltas.size:
                phase[1:] = np.cumsum(deltas - 1, dtype=np.int64)
        else:
            assert previous_raw is not None
            raw_delta = np.empty(raw.size, dtype=np.int64)
            raw_delta[0] = raw[0] - previous_raw
            if raw.size > 1:
                raw_delta[1:] = np.diff(raw)
            deltas = ((raw_delta + half) % modulus) - half
            phase = active_phase + np.cumsum(deltas - 1, dtype=np.int64)
        wraps += int(np.count_nonzero(np.abs(raw_delta) > half))
        nonunit += int(np.count_nonzero(deltas != 1))

        # A clean recording produces one vector comparison per chunk instead
        # of one Python iteration per row.  The small loop below runs only at
        # actual phase changes, which are the events this scanner must retain.
        changed = np.flatnonzero(
            phase != np.concatenate((np.asarray([active_phase], dtype=np.int64), phase[:-1]))
        )
        for offset in changed:
            row = chunk_start + int(offset)
            runs.append((active_start, row, active_phase))
            active_start = row
            active_phase = int(phase[offset])
        active_phase = int(phase[-1])
        previous_raw = int(raw[-1])
    runs.append((active_start, frames.shape[0], active_phase))
    return tuple(runs), int(initial_tick), wraps, nonunit


def _confirmed_repeat_fragments(
    frames: np.ndarray,
    *,
    start: int,
    end: int,
    lag: int,
    chunk_rows: int,
) -> tuple[tuple[int, int], ...]:
    """Confirm exact complete-frame repeats without allocating a whole file."""

    if lag <= 0 or start < lag or end <= start:
        return ()
    fragments: list[tuple[int, int]] = []
    active_start: int | None = None
    for chunk_start in range(start, end, chunk_rows):
        chunk_end = min(end, chunk_start + chunk_rows)
        equal = np.all(
            frames[chunk_start:chunk_end] == frames[chunk_start - lag : chunk_end - lag],
            axis=1,
        )
        # Process state changes, rather than every compared frame.  The
        # explicit ``array_equal`` proof below is retained for each compact
        # equal run, so this remains byte-exact rather than hash-based.
        states = np.concatenate(
            (np.asarray([active_start is not None], dtype=bool), equal)
        )
        transitions = np.flatnonzero(states[1:] != states[:-1])
        for offset in transitions:
            row = chunk_start + int(offset)
            if bool(equal[offset]):
                active_start = row
            else:
                assert active_start is not None
                if np.array_equal(frames[active_start:row], frames[active_start - lag : row - lag]):
                    fragments.append((active_start, row))
                active_start = None
    if active_start is not None and np.array_equal(
        frames[active_start:end], frames[active_start - lag : end - lag]
    ):
        fragments.append((active_start, end))
    return tuple(fragments)


def _counter_event(
    *,
    kind: IntegrityKind,
    start: int,
    end: int,
    initial_tick: int,
    displacement: int | None,
    tick_phase: int | None = None,
    confidence: Literal["high", "medium", "low", "unresolved"],
    evidence: str,
    channel_count: int,
    affected_lanes: tuple[int, ...] | None = None,
) -> AnalogIntegrityEvent:
    phase = displacement if tick_phase is None else tick_phase
    return AnalogIntegrityEvent(
        kind=kind,
        raw_start_row=start,
        raw_end_row=end,
        tick_start=initial_tick + start + (phase or 0),
        tick_end=initial_tick + end + (phase or 0),
        affected_lanes=tuple(range(channel_count)) if affected_lanes is None else affected_lanes,
        displacement_rows=displacement,
        confidence=confidence,
        evidence=evidence,
    )


def _normalise_timeline_events(
    events: Iterable[AnalogIntegrityEvent]) -> tuple[AnalogIntegrityEvent, ...]:
    """Keep correction-oriented timeline intervals sorted and non-overlapping."""

    ordered = sorted(
        events,
        key=lambda event: (event.raw_start_row, event.raw_end_row, event.kind, event.evidence),
    )
    normalized: list[AnalogIntegrityEvent] = []
    for event in ordered:
        if not normalized or event.raw_start_row >= normalized[-1].raw_end_row:
            normalized.append(event)
            continue
        # A stronger full-frame repeat proof supersedes a counter-only local
        # anomaly at the same rows.  Other overlapping evidence is represented
        # as an unresolved interval instead of silently choosing a correction.
        previous = normalized[-1]
        if previous.kind == "counter_corruption" and event.kind.startswith("repeat_"):
            normalized[-1] = event
        elif event.kind == "counter_corruption" and previous.kind.startswith("repeat_"):
            continue
        else:
            start = previous.raw_start_row
            end = max(previous.raw_end_row, event.raw_end_row)
            normalized[-1] = AnalogIntegrityEvent(
                kind="unresolved",
                raw_start_row=start,
                raw_end_row=end,
                tick_start=min(value for value in (previous.tick_start, event.tick_start) if value is not None),
                tick_end=max(value for value in (previous.tick_end, event.tick_end) if value is not None),
                affected_lanes=previous.affected_lanes,
                displacement_rows=None,
                confidence="unresolved",
                evidence="overlapping incompatible counter-phase evidence",
            )
    return tuple(normalized)


def _payload_candidates_without_overlap(
    candidates: Sequence[tuple[int, int, int]],
    existing_events: Sequence[AnalogIntegrityEvent],
) -> tuple[tuple[int, int, int], ...]:
    """Apply the existing greedy overlap policy without quadratic scans.

    ``_payload_repeat_candidates`` returns candidates ordered by raw start.
    The historical caller accepted each candidate only when it overlapped
    neither pre-existing counter evidence nor an earlier accepted payload
    candidate.  Pre-existing intervals are indexed as a sorted union; accepted
    payload candidates need only the most recent end because the accepted set
    is ordered and non-overlapping.  This changes lookup cost, not decisions.
    """

    occupied: list[tuple[int, int]] = []
    for event in sorted(existing_events, key=lambda item: (item.raw_start_row, item.raw_end_row)):
        start = int(event.raw_start_row)
        end = int(event.raw_end_row)
        if occupied and start <= occupied[-1][1]:
            occupied[-1] = (occupied[-1][0], max(occupied[-1][1], end))
        else:
            occupied.append((start, end))
    occupied_starts = [start for start, _end in occupied]
    accepted: list[tuple[int, int, int]] = []
    last_payload_end = -1
    for start, end, lag in candidates:
        insertion = bisect_left(occupied_starts, end)
        overlaps_existing = insertion > 0 and occupied[insertion - 1][1] > start
        if overlaps_existing or last_payload_end > start:
            continue
        accepted.append((start, end, lag))
        last_payload_end = end
    return tuple(accepted)


def _exact_sequence_match_mask(
    frames: np.ndarray,
    lanes: np.ndarray,
    rows: np.ndarray,
    source_rows: np.ndarray,
    *,
    sequence_rows: int,
) -> np.ndarray:
    """Byte-confirm many newest-source proposals with bounded vector work."""

    matches = np.ones(rows.size, dtype=bool)
    for offset in range(sequence_rows):
        active = np.flatnonzero(matches)
        if not active.size:
            break
        left = frames[(rows[active] - offset)[:, None], lanes]
        right = frames[(source_rows[active] - offset)[:, None], lanes]
        matches[active] = np.all(left == right, axis=1)
    return matches


def _payload_repeat_candidates(
    frames: np.ndarray,
    *,
    counter_lane: int,
    activity_lanes: Sequence[int],
    max_lag_rows: int,
    minimum_rows: int,
    chunk_rows: int,
) -> tuple[tuple[int, int, int], ...]:
    """Find dynamic scientific-payload replay evidence with a fresh counter.

    A bounded rolling hash proposes multi-row candidates; exact payload bytes
    over at least ``minimum_rows`` rows confirm them.  Using a *sequence*
    signature is important: individual IMU frames are normally held for a few
    rows, so a latest-row-only lookup would hide a genuine replay behind the
    immediately preceding sample hold.  The table retains only the configured
    ``max_lag_rows`` horizon, so it is O(N) in source rows and bounded in
    memory.  The horizon is an operator parameter, not a firmware constant:
    counter-phase replay detection has no such lag ceiling, while a recording
    with a freshly rewritten counter can request a larger payload-only search.
    Candidate hashes and exact confirmation cover every non-counter lane, but
    only ``activity_lanes`` can satisfy the dynamic-sequence guard.  Expected
    housekeeping/timing cadence therefore cannot turn otherwise held sensor
    payload into replay evidence.  A fixed scientific payload (for example an
    animal at rest) does not reach the dynamic-sequence test.
    """

    if max_lag_rows < minimum_rows or minimum_rows < 2:
        raise ValueError("payload repeat search bounds are invalid")
    lanes = np.asarray(
        [lane for lane in range(frames.shape[1]) if lane != counter_lane],
        dtype=np.intp,
    )
    activity = tuple(int(lane) for lane in activity_lanes)
    if (
        not activity
        or tuple(sorted(set(activity))) != activity
        or any(lane < 0 or lane >= frames.shape[1] or lane == counter_lane for lane in activity)
    ):
        raise ValueError("payload activity lanes must be sorted, unique, in range, and non-counter")
    activity_positions = np.searchsorted(lanes, np.asarray(activity, dtype=np.intp))
    if np.any(activity_positions >= lanes.size) or np.any(lanes[activity_positions] != activity):
        raise ValueError("payload activity lanes must be included in the matched payload")
    # A deterministic uint64 row hash is only a candidate filter.  Exact word
    # comparison below is the proof and eliminates collision risk.
    weights = (np.arange(lanes.size, dtype=np.uint64) * np.uint64(0x9E3779B185EBCA87)) | 1
    candidates: list[tuple[int, int, int]] = []
    active_start: int | None = None
    active_source: int | None = None
    previous_row = -2
    tail_signatures = np.empty(0, dtype=np.uint64)
    tail_dynamic = np.empty(0, dtype=bool)
    tail_start_row = 0
    prefix_hashes = np.empty(0, dtype=np.uint64)
    prefix_activity = np.empty((0, len(activity)), dtype=np.uint16)
    for chunk_start in range(0, frames.shape[0], chunk_rows):
        chunk_end = min(frames.shape[0], chunk_start + chunk_rows)
        payload = np.asarray(frames[chunk_start:chunk_end, lanes], dtype=np.uint16)
        activity_payload = payload[:, activity_positions]
        row_hashes = np.bitwise_xor.reduce(payload.astype(np.uint64) * weights, axis=1)
        prefix_size = prefix_hashes.size
        all_hashes = np.concatenate((prefix_hashes, row_hashes))
        all_activity = np.concatenate((prefix_activity, activity_payload), axis=0)
        signatures = all_hashes[minimum_rows - 1 :].copy()
        # This is a candidate signature only; every reported event remains
        # byte-confirmed below.  The short fixed loop is over the configured
        # sequence length, never recording rows.
        for offset in range(1, minimum_rows):
            signatures ^= np.left_shift(
                all_hashes[minimum_rows - 1 - offset : all_hashes.size - offset],
                11 * offset,
            )
        activity_changes = np.any(all_activity[1:] != all_activity[:-1], axis=1)
        change_prefix = np.r_[0, np.cumsum(activity_changes, dtype=np.int64)]
        endpoints = np.arange(minimum_rows - 1, all_hashes.size)
        signature_dynamic = (
            change_prefix[endpoints] - change_prefix[endpoints - (minimum_rows - 1)]
        ) > 0
        prefix_hashes = all_hashes[-(minimum_rows - 1) :].copy()
        prefix_activity = all_activity[-(minimum_rows - 1) :].copy()
        # The tail is exactly the candidate horizon.  Sorting a bounded
        # chunk+tail identifies row-hash collisions/replays in C; clean,
        # dynamic recordings therefore avoid a Python dict lookup per row.
        combined = np.concatenate((tail_signatures, signatures))
        combined_dynamic = np.concatenate((tail_dynamic, signature_dynamic))
        signature_start_row = chunk_start - prefix_size + (minimum_rows - 1)
        combined_start_row = tail_start_row if tail_signatures.size else signature_start_row
        order = np.argsort(combined, kind="stable")
        ordered = combined[order]
        pair_rows: list[np.ndarray] = []
        pair_sources: list[np.ndarray] = []
        proposal_rows: list[np.ndarray] = []
        proposal_sources: list[np.ndarray] = []
        proposal_groups: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        proposal_count = 0
        group_edges = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1, ordered.size]
        duplicate_groups = np.flatnonzero(np.diff(group_edges) >= 2)
        if duplicate_groups.size:
            group_starts = group_edges[duplicate_groups]
            group_ends = group_edges[duplicate_groups + 1]
            tail_rows = tail_signatures.size
            for group_start, group_end in zip(group_starts, group_ends):
                # Stable sorting preserves chronological positions inside an
                # equal-signature group, which is the old latest-row lookup
                # order.
                positions = order[int(group_start) : int(group_end)]
                current_positions = positions[positions >= tail_rows]
                current_positions = current_positions[combined_dynamic[current_positions]]
                if not current_positions.size:
                    continue
                # A held payload produces repeated sequence signatures.  Only
                # consider sources within the configured horizon and search
                # newest-first.  Stationary/held signatures were filtered
                # above.  Propose the newest eligible source for every current
                # position, confirm those proposals in one bounded vector
                # batch below, and retain the exact older-source fallback for
                # the rare hash collision.
                lower = np.searchsorted(
                    positions,
                    current_positions - max_lag_rows,
                    side="left",
                )
                upper = np.searchsorted(
                    positions,
                    current_positions - minimum_rows,
                    side="right",
                )
                eligible = upper > lower
                if not np.any(eligible):
                    continue
                currents = current_positions[eligible]
                lower = lower[eligible]
                upper = upper[eligible]
                newest_sources = positions[upper - 1]
                rows = np.asarray(combined_start_row + currents, dtype=np.int64)
                sources = np.asarray(combined_start_row + newest_sources, dtype=np.int64)
                proposal_rows.append(rows)
                proposal_sources.append(sources)
                proposal_groups.append(
                    (proposal_count, positions, currents, lower, upper)
                )
                proposal_count += rows.size
        if proposal_rows:
            proposed_rows = np.concatenate(proposal_rows)
            proposed_sources = np.concatenate(proposal_sources)
            newest_matches = _exact_sequence_match_mask(
                frames,
                lanes,
                proposed_rows,
                proposed_sources,
                sequence_rows=minimum_rows,
            )
            for offset, positions, currents, lower, upper in proposal_groups:
                group_size = currents.size
                group_matches = newest_matches[offset : offset + group_size]
                if np.any(group_matches):
                    pair_rows.append(
                        np.asarray(
                            combined_start_row + currents[group_matches],
                            dtype=np.int64,
                        )
                    )
                    pair_sources.append(
                        np.asarray(
                            combined_start_row + positions[upper[group_matches] - 1],
                            dtype=np.int64,
                        )
                    )
                fallback_rows: list[int] = []
                fallback_sources: list[int] = []
                for current, first, stop in zip(
                    currents[~group_matches],
                    lower[~group_matches],
                    upper[~group_matches] - 1,
                ):
                    row = int(combined_start_row + current)
                    for source in positions[int(first) : int(stop)][::-1]:
                        source_row = int(combined_start_row + source)
                        if np.array_equal(
                            frames[row - minimum_rows + 1 : row + 1, lanes],
                            frames[source_row - minimum_rows + 1 : source_row + 1, lanes],
                        ):
                            fallback_rows.append(row)
                            fallback_sources.append(source_row)
                            break
                if fallback_rows:
                    pair_rows.append(np.asarray(fallback_rows, dtype=np.int64))
                    pair_sources.append(np.asarray(fallback_sources, dtype=np.int64))
        if pair_rows:
            rows = np.concatenate(pair_rows)
            sources = np.concatenate(pair_sources)
            order = np.argsort(rows, kind="stable")
            candidate_rows = rows[order]
            candidate_sources = sources[order]
        else:
            candidate_rows = np.empty(0, dtype=np.int64)
            candidate_sources = np.empty(0, dtype=np.int64)
        for row_value, source_value in zip(candidate_rows, candidate_sources):
            row = int(row_value)
            source = int(source_value)
            # Any unpaired row between candidates was a non-repeat in the
            # original streaming implementation and closes its active run.
            if active_start is not None and previous_row != row - 1:
                candidates.append((active_start, previous_row + 1, active_start - active_source))
                active_start = None
                active_source = None
            # Candidate rows are already dynamic and byte-confirmed above.
            # Repeating the identical proof here previously doubled millions
            # of small Python-to-NumPy calls without changing a decision.
            expected_source = (
                None if active_start is None or active_source is None else active_source + (row - active_start)
            )
            if active_start is None or source != expected_source:
                if active_start is not None and active_source is not None:
                    candidates.append((active_start, previous_row + 1, active_start - active_source))
                active_start = row
                active_source = source
            previous_row = row
        tail_size = min(max_lag_rows, combined.size)
        tail_signatures = combined[-tail_size:].copy()
        tail_dynamic = combined_dynamic[-tail_size:].copy()
        tail_start_row = combined_start_row + combined.size - tail_size
    if active_start is not None and active_source is not None:
        candidates.append((active_start, previous_row + 1, active_start - active_source))
    # The last ``minimum_rows - 1`` candidates establish sequence evidence,
    # but are included in the exact event once it has that support.  Sequence
    # signatures naturally skip a few held-frame boundaries; expand each
    # confirmed candidate to the full byte-equal payload span and merge its
    # same-lag neighbours.  This restores the actual 24-row replay extent
    # rather than exposing three short chunks between normal six-row holds.
    expanded: list[tuple[int, int, int]] = []
    for start, end, lag in candidates:
        start = max(0, start - minimum_rows + 1)
        while start > lag and np.array_equal(frames[start - 1, lanes], frames[start - lag - 1, lanes]):
            start -= 1
        while end < frames.shape[0] and np.array_equal(frames[end, lanes], frames[end - lag, lanes]):
            end += 1
        if end > start:
            expanded.append((start, end, lag))
    merged: list[tuple[int, int, int]] = []
    for start, end, lag in sorted(expanded):
        if merged and lag == merged[-1][2] and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end), lag)
        else:
            merged.append((start, end, lag))
    return tuple(merged)


def _imu_metrics_and_events(
    frames: np.ndarray,
    *,
    phase_runs: Sequence[tuple[int, int, int]],
    initial_tick: int,
    imu_lanes: tuple[int, ...],
    counter_lane: int,
    stall_min_rows: int,
    quality_min_rows: int,
    chunk_rows: int,
) -> tuple[list[AnalogIntegrityEvent], int, float | None, int | None, int, int, int]:
    """Measure IMU payload health without treating sample hold as corruption."""

    if frames.shape[0] == 0:
        return [], 0, None, None, 0, 0, 0
    context_lanes = tuple(index for index in range(frames.shape[1]) if index not in imu_lanes and index != counter_lane)
    events: list[AnalogIntegrityEvent] = []
    update_count = 0
    interval_histogram = np.zeros(4097, dtype=np.int64)
    last_update: int | None = None
    max_interval: int | None = None
    zero_rows = 0
    rail_rows = 0
    longest = 1
    static_start = 0
    previous_imu: np.ndarray | None = None
    phase_ends = np.asarray([end for _start, end, _phase in phase_runs], dtype=np.int64)
    phase_values = np.asarray([phase for _start, _end, phase in phase_runs], dtype=np.int64)
    zero_run_start: int | None = None
    saturation_run_starts: dict[int, int | None] = {lane: None for lane in imu_lanes}

    def append_modality_run(kind: IntegrityKind, start: int, end: int, lanes: tuple[int, ...]) -> None:
        if start >= end or end - start < quality_min_rows:
            return
        evidence = (
            f"all IMU lanes were exactly zero for {end - start} rows"
            if kind == "imu_all_zero"
            else f"IMU ADC rail saturation for {end - start} rows"
        )
        events.append(
            AnalogIntegrityEvent(
                kind=kind,
                raw_start_row=start,
                raw_end_row=end,
                tick_start=initial_tick + start,
                tick_end=initial_tick + end,
                affected_lanes=lanes,
                displacement_rows=None,
                confidence="high",
                evidence=evidence,
            )
        )

    def consume_boolean_runs(
        mask: np.ndarray,
        *,
        chunk_start: int,
        active_start: int | None,
        kind: IntegrityKind,
        lanes: tuple[int, ...],
    ) -> int | None:
        """Append exact contiguous true runs while preserving chunk state."""

        states = np.concatenate((np.asarray([active_start is not None], dtype=bool), mask))
        for offset in np.flatnonzero(states[1:] != states[:-1]):
            row = chunk_start + int(offset)
            if bool(mask[offset]):
                active_start = row
            else:
                assert active_start is not None
                append_modality_run(kind, active_start, row, lanes)
                active_start = None
        return active_start

    def phase_is_constant(start: int, end: int) -> bool:
        """Whether a static interval remains in one counter-phase regime."""

        phase_index = int(np.searchsorted(phase_ends, start, side="right"))
        if phase_index >= len(phase_runs):
            return False
        phase = phase_values[phase_index]
        while phase_index < len(phase_runs) and phase_runs[phase_index][0] < end:
            if phase_values[phase_index] != phase:
                return False
            phase_index += 1
        return True

    def finish_static(start: int, end: int) -> None:
        nonlocal longest
        length = end - start
        longest = max(longest, length)
        if length < stall_min_rows or not phase_is_constant(start, end):
            return
        # Context is intentionally compared only inside the unchanged IMU
        # interval.  An update at ``end`` begins the next static run and must
        # not be taken as evidence for the preceding one.
        context_changed = bool(
            context_lanes
            and np.any(
                frames[start + 1 : end, context_lanes]
                != frames[start : end - 1, context_lanes]
            )
        )
        if context_changed:
            events.append(
                AnalogIntegrityEvent(
                    kind="sensor_stall",
                    raw_start_row=start,
                    raw_end_row=end,
                    tick_start=initial_tick + start,
                    tick_end=initial_tick + end,
                    affected_lanes=imu_lanes,
                    displacement_rows=None,
                    confidence="medium",
                    evidence=(
                        "all IMU lanes were unchanged while non-IMU payload changed; "
                        f"duration={length} rows"
                    ),
                )
            )

    for chunk_start in range(0, frames.shape[0], chunk_rows):
        chunk_end = min(frames.shape[0], chunk_start + chunk_rows)
        chunk = np.asarray(frames[chunk_start:chunk_end])
        imu_chunk = chunk[:, imu_lanes]
        zero_mask = np.all(imu_chunk == 0, axis=1)
        rail_mask = (imu_chunk == np.iinfo(np.int16).min) | (imu_chunk == np.iinfo(np.int16).max)
        zero_rows += int(np.count_nonzero(zero_mask))
        rail_rows += int(np.count_nonzero(np.any(rail_mask, axis=1)))
        zero_run_start = consume_boolean_runs(
            zero_mask,
            chunk_start=chunk_start,
            active_start=zero_run_start,
            kind="imu_all_zero",
            lanes=imu_lanes,
        )
        for lane_offset, lane in enumerate(imu_lanes):
            saturation_run_starts[lane] = consume_boolean_runs(
                rail_mask[:, lane_offset],
                chunk_start=chunk_start,
                active_start=saturation_run_starts[lane],
                kind="imu_saturation",
                lanes=(lane,),
            )
        changed = np.empty(imu_chunk.shape[0], dtype=bool)
        if previous_imu is None:
            changed[0] = False
            update_count = 1
            last_update = chunk_start
        else:
            changed[0] = bool(np.any(imu_chunk[0] != previous_imu))
        if imu_chunk.shape[0] > 1:
            changed[1:] = np.any(imu_chunk[1:] != imu_chunk[:-1], axis=1)
        update_starts = chunk_start + np.flatnonzero(changed)
        if update_starts.size:
            assert last_update is not None
            interval_starts = np.concatenate((np.asarray([last_update], dtype=np.int64), update_starts))
            intervals = np.diff(interval_starts)
            bounded = np.minimum(intervals, interval_histogram.size - 1)
            interval_histogram += np.bincount(bounded, minlength=interval_histogram.size)[: interval_histogram.size]
            candidate_starts = np.concatenate(
                (np.asarray([static_start], dtype=np.int64), update_starts[:-1])
            )
            candidate_lengths = update_starts - candidate_starts
            longest = max(longest, int(np.max(candidate_lengths)))
            for start, end in zip(
                candidate_starts[candidate_lengths >= stall_min_rows],
                update_starts[candidate_lengths >= stall_min_rows],
            ):
                finish_static(int(start), int(end))
            update_count += int(update_starts.size)
            max_interval_value = int(np.max(intervals))
            max_interval = max_interval_value if max_interval is None else max(max_interval, max_interval_value)
            static_start = int(update_starts[-1])
            last_update = static_start
        previous_imu = imu_chunk[-1].copy()
    finish_static(static_start, frames.shape[0])
    if zero_run_start is not None:
        append_modality_run("imu_all_zero", zero_run_start, frames.shape[0], imu_lanes)
    for lane, start in saturation_run_starts.items():
        if start is not None:
            append_modality_run("imu_saturation", start, frames.shape[0], (lane,))
    intervals_total = int(interval_histogram.sum())
    if intervals_total:
        median_bin = int(np.searchsorted(np.cumsum(interval_histogram), (intervals_total + 1) // 2))
        median: float | None = float(median_bin)
    else:
        median = None
    return events, update_count, median, max_interval, zero_rows, rail_rows, longest


def scan_analog_frames(
    frames: np.ndarray,
    *,
    counter_lane: int = 11,
    counter_modulus: int = 1 << 16,
    imu_lanes: Sequence[int] = tuple(range(1, 10)),
    min_phase_run_rows: int = 3,
    imu_stall_min_rows: int = 1250,
    imu_quality_min_rows: int = 2,
    chunk_rows: int = 65_536,
    payload_repeat_max_lag_rows: int = 8192,
    payload_repeat_min_rows: int = 3,
    payload_activity_lanes: Sequence[int] | None = None,
    device_index: int = 1,
) -> AnalogIntegrityResult:
    """Scan a sample-major raw analog array without modifying it.

    ``frames`` may be a read-only ``numpy.memmap``.  The implementation only
    allocates bounded candidate chunks; source comparisons are slice views.
    For the default CE64 layout, counter-independent replay activity must occur
    in zero-based lane 0 through 9.  Exact replay confirmation still includes
    every non-counter lane.  ``payload_activity_lanes`` can state another
    explicitly supported layout without inferring lane roles from values.
    """

    matrix = np.asarray(frames)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("analog frames must be a non-empty two-dimensional array")
    if matrix.dtype.kind not in "iu" or matrix.dtype.itemsize != 2:
        raise ValueError("analog frames must use 16-bit words")
    if not 0 <= counter_lane < matrix.shape[1]:
        raise ValueError("counter lane is outside the analog frame")
    lanes = tuple(int(lane) for lane in imu_lanes)
    if not lanes or tuple(sorted(set(lanes))) != lanes or any(lane < 0 or lane >= matrix.shape[1] for lane in lanes):
        raise ValueError("IMU lanes must be sorted, unique, and in range")
    if min_phase_run_rows < 2 or imu_stall_min_rows < 2 or imu_quality_min_rows < 2 or chunk_rows < 1:
        raise ValueError("minimum run lengths and chunk size are invalid")
    if payload_repeat_max_lag_rows < payload_repeat_min_rows or payload_repeat_min_rows < 2:
        raise ValueError("payload repeat search bounds are invalid")
    repeat_activity_lanes = (
        tuple(range(10))
        if payload_activity_lanes is None
        else tuple(int(lane) for lane in payload_activity_lanes)
    )
    if (
        not repeat_activity_lanes
        or tuple(sorted(set(repeat_activity_lanes))) != repeat_activity_lanes
        or any(
            lane < 0 or lane >= matrix.shape[1] or lane == counter_lane
            for lane in repeat_activity_lanes
        )
    ):
        raise ValueError(
            "payload activity lanes must be sorted, unique, in range, and non-counter"
        )
    if device_index < 1:
        raise ValueError("device index must be one-based and positive")

    phase_runs, initial_tick, wraps, nonunit = _counter_phase_runs(
        matrix,
        counter_lane=counter_lane,
        modulus=counter_modulus,
        chunk_rows=chunk_rows,
    )
    timeline_events: list[AnalogIntegrityEvent] = []

    # ``phase_runs`` contain an absolute counter phase (tick - raw row).  A
    # persistent storage event, however, is an *incremental* correction from
    # the previously established baseline; this is the contract consumed by
    # the segment builder.  Temporary excursions are also measured relative
    # to that surrounding baseline.
    persistent_baseline = 0
    for index, (start, end, absolute_phase) in enumerate(phase_runs):
        if absolute_phase == persistent_baseline:
            continue
        length = end - start
        returns_to_baseline = (
            index + 1 < len(phase_runs)
            and phase_runs[index + 1][2] == persistent_baseline
        )
        displacement = absolute_phase - persistent_baseline
        lag = abs(displacement)
        # Exact complete-frame replay proves a local storage fault even for a
        # one- or two-row counter phase run.  Do this before classifying a
        # short run as counter-only corruption.
        fragments = _confirmed_repeat_fragments(
            matrix,
            start=start,
            end=end,
            lag=lag,
            chunk_rows=chunk_rows,
        )
        if displacement < 0 and fragments:
            repeat_kind: IntegrityKind = (
                "repeat_overwrite" if returns_to_baseline else "repeat_insertion"
            )
            gap_kind: IntegrityKind = "temporary_excursion" if returns_to_baseline else "unresolved"
            gap_confidence: Literal["high", "medium", "low", "unresolved"] = (
                "medium" if returns_to_baseline else "unresolved"
            )
            gap_evidence = (
                "temporary counter phase excursion returned to the prior baseline; "
                "non-repeated rows are locally invalid"
                if returns_to_baseline
                else "counter phase excursion contained a non-repeated payload interval; "
                "source order cannot be established"
            )
            covered_end = start
            for fragment_start, fragment_end in fragments:
                if covered_end < fragment_start:
                    timeline_events.append(
                        _counter_event(
                            kind=gap_kind,
                            start=covered_end,
                            end=fragment_start,
                            initial_tick=initial_tick,
                            displacement=displacement,
                            tick_phase=absolute_phase,
                            confidence=gap_confidence,
                            evidence=gap_evidence,
                            channel_count=matrix.shape[1],
                        )
                    )
                timeline_events.append(
                    _counter_event(
                        kind=repeat_kind,
                        start=fragment_start,
                        end=fragment_end,
                        initial_tick=initial_tick,
                        displacement=displacement,
                        tick_phase=absolute_phase,
                        confidence="high",
                        evidence=(
                            f"counter phase {displacement:+d} rows relative to persistent baseline and "
                            f"byte-confirmed complete-frame repeat at inferred lag {lag} rows"
                        ),
                        channel_count=matrix.shape[1],
                    )
                )
                covered_end = fragment_end
            persistent_insertion_fully_explained = (
                not returns_to_baseline
                and len(fragments) == 1
                and fragments[0] == (start, start + lag)
            )
            if covered_end < end and not persistent_insertion_fully_explained:
                timeline_events.append(
                    _counter_event(
                        kind=gap_kind,
                        start=covered_end,
                        end=end,
                        initial_tick=initial_tick,
                        displacement=displacement,
                        tick_phase=absolute_phase,
                        confidence=gap_confidence,
                        evidence=gap_evidence,
                        channel_count=matrix.shape[1],
                    )
                )
            if not returns_to_baseline:
                persistent_baseline = absolute_phase
            continue
        if length < min_phase_run_rows:
            timeline_events.append(
                _counter_event(
                    kind="counter_corruption",
                    start=start,
                    end=end,
                    initial_tick=initial_tick,
                    displacement=displacement,
                    tick_phase=absolute_phase,
                    confidence="medium",
                    evidence=(
                        "isolated counter phase excursion without enough support for a "
                        "storage-timeline correction"
                    ),
                    channel_count=matrix.shape[1],
                    affected_lanes=(counter_lane,),
                )
            )
            continue
        if not returns_to_baseline:
            kind: IntegrityKind = "missing" if displacement > 0 else "insertion"
            # The phase relation remains usable after a new segment/reacquisition.
            # Keep only a conservative transition guard here; a ``missing``
            # canonical interval is materialised by the later mapping, while a
            # negative phase has at most ``abs(displacement)`` excess raw rows
            # local to the boundary.
            local_start = start if displacement > 0 else max(0, start - abs(displacement))
            local_end = min(end, start + 1) if displacement > 0 else start
            timeline_events.append(
                _counter_event(
                    kind=kind,
                    start=local_start,
                    end=local_end,
                    initial_tick=initial_tick,
                    displacement=displacement,
                    tick_phase=absolute_phase,
                    confidence="medium",
                    evidence=(
                        f"persistent counter phase change of {displacement:+d} rows; "
                        "local transition guard only, later source support requires reacquisition"
                    ),
                    channel_count=matrix.shape[1],
                )
            )
            persistent_baseline = absolute_phase
        else:
            # A returned counter phase excursion is independently reacquired
            # by the surrounding baseline.  Its rows are local-invalid, but
            # do not poison the later source mapping.
            timeline_events.append(
                _counter_event(
                    kind="temporary_excursion",
                    start=start,
                    end=end,
                    initial_tick=initial_tick,
                    displacement=displacement,
                    tick_phase=absolute_phase,
                    confidence="medium",
                    evidence=(
                        f"temporary counter phase {displacement:+d} rows returned to the prior "
                        "baseline without exact full-frame repeat confirmation"
                    ),
                    channel_count=matrix.shape[1],
                )
            )

    # A fresh counter can conceal an overwrite of the other lanes.  Do a
    # counter-independent dynamic-payload search, but avoid duplicating a
    # stronger full-frame event already explained by counter phase evidence.
    payload_candidates = _payload_repeat_candidates(
        matrix,
        counter_lane=counter_lane,
        activity_lanes=repeat_activity_lanes,
        max_lag_rows=payload_repeat_max_lag_rows,
        minimum_rows=payload_repeat_min_rows,
        chunk_rows=chunk_rows,
    )
    for start, end, lag in _payload_candidates_without_overlap(
        payload_candidates,
        timeline_events,
    ):
        timeline_events.append(
            AnalogIntegrityEvent(
                kind="repeat_overwrite",
                raw_start_row=start,
                raw_end_row=end,
                tick_start=initial_tick + start,
                tick_end=initial_tick + end,
                affected_lanes=tuple(lane for lane in range(matrix.shape[1]) if lane != counter_lane),
                displacement_rows=None,
                confidence="medium",
                evidence=(
                    "counter-continuous dynamic payload replay at inferred lag "
                    f"{lag} rows; local payload overwrite with counter-verified tail"
                ),
            )
        )
    timeline_events = list(_normalise_timeline_events(timeline_events))
    imu_events, updates, median, maximum, zero_rows, rail_rows, longest_static = _imu_metrics_and_events(
        matrix,
        initial_tick=initial_tick,
        phase_runs=phase_runs,
        imu_lanes=lanes,
        counter_lane=counter_lane,
        stall_min_rows=imu_stall_min_rows,
        quality_min_rows=imu_quality_min_rows,
        chunk_rows=chunk_rows,
    )
    events = tuple(
        sorted(
            [replace(event, device_index=device_index) for event in [*timeline_events, *imu_events]],
            key=lambda event: (event.raw_start_row, event.raw_end_row, event.kind, event.affected_lanes),
        )
    )
    confirmed_rows = sum(
        event.raw_end_row - event.raw_start_row
        for event in timeline_events
        if event.kind in {"repeat_insertion", "repeat_overwrite"}
    )
    return AnalogIntegrityResult(
        events=events,
        metrics=AnalogIntegrityMetrics(
            row_count=int(matrix.shape[0]),
            channel_count=int(matrix.shape[1]),
            counter_wrap_count=wraps,
            counter_nonunit_delta_count=nonunit,
            counter_phase_run_count=sum(1 for _start, _end, value in phase_runs if value != 0),
            confirmed_repeat_rows=int(confirmed_rows),
            imu_update_count=updates,
            imu_median_update_rows=median,
            imu_max_update_rows=maximum,
            imu_all_zero_rows=zero_rows,
            imu_rail_rows=rail_rows,
            imu_longest_static_rows=longest_static,
        ),
        device_index=device_index,
    )


def scan_analog_integrity(
    path: Path,
    *,
    channel_count: int,
    device_index: int = 1,
    **kwargs: object,
) -> AnalogIntegrityResult:
    """Read and scan a raw ``analogin.dat`` file through a read-only memmap."""

    source = Path(path)
    if channel_count < 1:
        raise ValueError("channel count must be positive")
    frame_bytes = channel_count * np.dtype("<i2").itemsize
    size = source.stat().st_size
    if size == 0 or size % frame_bytes:
        raise ValueError(
            f"analogin.dat framing is not a non-zero whole number of {frame_bytes}-byte frames: {source}"
        )
    rows = size // frame_bytes
    mapped = np.memmap(source, dtype="<i2", mode="r", shape=(rows, channel_count), order="C")
    try:
        return scan_analog_frames(mapped, device_index=device_index, **kwargs)
    finally:
        memory_map = getattr(mapped, "_mmap", None)
        if memory_map is not None:
            memory_map.close()
