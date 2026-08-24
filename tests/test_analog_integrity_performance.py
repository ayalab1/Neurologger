"""Structural regression checks for bounded, vectorized analog integrity scans."""

from __future__ import annotations

import inspect
import sys
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from wild_preprocess.analog import integrity


def _dynamic_frames(rows: int) -> np.ndarray:
    frame_index = np.arange(rows, dtype=np.uint32)
    frames = np.empty((rows, 16), dtype="<i2")
    for lane in range(16):
        # Keep this synthetic clean fixture away from exact ADC rails; rail
        # hits are now intentionally surfaced as IMU modality findings.
        frames[:, lane] = (
            (frame_index * (lane * 17 + 3) + lane) % 60_000 - 30_000
        ).astype(np.int16)
    frames[:, 11] = frame_index.astype(np.uint16).view(np.int16)
    return frames


def _brute_payload_repeat_candidates(
    frames: np.ndarray,
    *,
    counter_lane: int,
    activity_lanes: tuple[int, ...],
    max_lag_rows: int,
    minimum_rows: int,
) -> tuple[tuple[int, int, int], ...]:
    lanes = np.asarray([lane for lane in range(frames.shape[1]) if lane != counter_lane])
    activity = np.asarray(activity_lanes)
    candidates: list[tuple[int, int, int]] = []
    active_start: int | None = None
    active_source: int | None = None
    previous_row = -2
    for row in range(minimum_rows - 1, frames.shape[0]):
        left = frames[row - minimum_rows + 1 : row + 1, lanes]
        source_row = None
        activity_window = frames[row - minimum_rows + 1 : row + 1, activity]
        if np.any(activity_window[1:] != activity_window[:-1]):
            lower = max(minimum_rows - 1, row - max_lag_rows)
            upper = row - minimum_rows
            for source in range(upper, lower - 1, -1):
                if np.array_equal(
                    left,
                    frames[source - minimum_rows + 1 : source + 1, lanes],
                ):
                    source_row = source
                    break
        if active_start is not None and (source_row is None or previous_row != row - 1):
            assert active_source is not None
            candidates.append((active_start, previous_row + 1, active_start - active_source))
            active_start = None
            active_source = None
        if source_row is None:
            continue
        expected_source = (
            None
            if active_start is None or active_source is None
            else active_source + (row - active_start)
        )
        if active_start is None or source_row != expected_source:
            if active_start is not None and active_source is not None:
                candidates.append((active_start, previous_row + 1, active_start - active_source))
            active_start = row
            active_source = source_row
        previous_row = row
    if active_start is not None and active_source is not None:
        candidates.append((active_start, previous_row + 1, active_start - active_source))
    expanded: list[tuple[int, int, int]] = []
    for start, end, lag in candidates:
        start = max(0, start - minimum_rows + 1)
        while start > lag and np.array_equal(
            frames[start - 1, lanes], frames[start - lag - 1, lanes]
        ):
            start -= 1
        while end < frames.shape[0] and np.array_equal(
            frames[end, lanes], frames[end - lag, lanes]
        ):
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


class AnalogIntegrityPerformanceStructureTest(unittest.TestCase):
    def test_hot_paths_do_not_restore_per_row_python_loops(self) -> None:
        counter_source = inspect.getsource(integrity._counter_phase_runs)
        repeat_source = inspect.getsource(integrity._confirmed_repeat_fragments)
        imu_source = inspect.getsource(integrity._imu_metrics_and_events)
        payload_source = inspect.getsource(integrity._payload_repeat_candidates)
        overlap_source = inspect.getsource(integrity._payload_candidates_without_overlap)
        self.assertNotIn("for offset, word in enumerate(chunk)", counter_source)
        self.assertNotIn("for local_row, row in enumerate(chunk)", imu_source)
        self.assertNotIn("for offset, is_equal in enumerate(equal)", repeat_source)
        self.assertNotIn("latest: dict", payload_source)
        self.assertIn("current_positions - max_lag_rows", payload_source)
        self.assertIn("combined_dynamic[current_positions]", payload_source)
        self.assertIn("_exact_sequence_match_mask", payload_source)
        self.assertNotIn("for event in timeline_events", overlap_source)

    def test_many_ordered_payload_candidates_use_the_same_greedy_policy(self) -> None:
        candidates = tuple((2 * row, 2 * row + 1, 3) for row in range(100_000))
        self.assertEqual(
            integrity._payload_candidates_without_overlap(candidates, ()),
            candidates,
        )

    def test_batched_payload_search_matches_exact_brute_reference(self) -> None:
        rng = np.random.default_rng(90210)
        frames = rng.integers(-20_000, 20_000, size=(240, 16), dtype=np.int16)
        frames[:, 11] = np.arange(240, dtype=np.uint16).view(np.int16)
        lanes = [*range(0, 11), *range(12, 16)]
        frames[120:145, lanes] = frames[73:98, lanes]
        expected = _brute_payload_repeat_candidates(
            frames,
            counter_lane=11,
            activity_lanes=tuple(range(10)),
            max_lag_rows=80,
            minimum_rows=3,
        )
        for chunk_rows in (7, 31, 240):
            with self.subTest(chunk_rows=chunk_rows):
                self.assertEqual(
                    integrity._payload_repeat_candidates(
                        frames,
                        counter_lane=11,
                        activity_lanes=tuple(range(10)),
                        max_lag_rows=80,
                        minimum_rows=3,
                        chunk_rows=chunk_rows,
                    ),
                    expected,
                )

    def test_large_clean_dynamic_scan_keeps_exact_metrics_across_chunks(self) -> None:
        # This is intentionally a functional, not wall-clock, guard.  It
        # exercises enough rows to make accidental per-row scanner loops
        # conspicuous in ordinary CI while remaining deterministic.
        rows = 262_144
        result = integrity.scan_analog_frames(
            _dynamic_frames(rows),
            chunk_rows=16_384,
            imu_stall_min_rows=1_250,
        )
        self.assertTrue(result.clean)
        self.assertEqual(result.metrics.row_count, rows)
        self.assertEqual(result.metrics.counter_nonunit_delta_count, 0)
        self.assertEqual(result.metrics.counter_phase_run_count, 0)
        self.assertEqual(result.metrics.imu_update_count, rows)
        self.assertEqual(result.metrics.imu_median_update_rows, 1.0)
        self.assertEqual(result.metrics.imu_max_update_rows, 1)

    def test_stationary_payload_skips_exact_candidate_comparisons(self) -> None:
        # This exercises the pathological former case: millions of equal
        # held-frame hashes.  The dynamic sequence filter must reject them
        # before source enumeration, rather than visiting every older row in
        # the lag horizon.  Counting exact comparisons is deterministic and
        # avoids a fragile wall-clock assertion.
        rows = 131_072
        frames = np.full((rows, 16), 42, dtype="<i2")
        frames[:, 11] = np.arange(rows, dtype=np.uint32).astype(np.uint16).view(np.int16)
        with mock.patch.object(integrity.np, "array_equal", wraps=np.array_equal) as equality:
            candidates = integrity._payload_repeat_candidates(
                frames,
                counter_lane=11,
                activity_lanes=tuple(range(10)),
                max_lag_rows=8_192,
                minimum_rows=3,
                chunk_rows=16_384,
            )
        self.assertEqual(candidates, ())
        self.assertEqual(equality.call_count, 0)


if __name__ == "__main__":
    unittest.main()
