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


class AnalogIntegrityPerformanceStructureTest(unittest.TestCase):
    def test_hot_paths_do_not_restore_per_row_python_loops(self) -> None:
        counter_source = inspect.getsource(integrity._counter_phase_runs)
        repeat_source = inspect.getsource(integrity._confirmed_repeat_fragments)
        imu_source = inspect.getsource(integrity._imu_metrics_and_events)
        payload_source = inspect.getsource(integrity._payload_repeat_candidates)
        self.assertNotIn("for offset, word in enumerate(chunk)", counter_source)
        self.assertNotIn("for local_row, row in enumerate(chunk)", imu_source)
        self.assertNotIn("for offset, is_equal in enumerate(equal)", repeat_source)
        self.assertNotIn("latest: dict", payload_source)
        self.assertIn("current - max_lag_rows", payload_source)
        self.assertIn("combined_dynamic[current_positions]", payload_source)

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
                max_lag_rows=8_192,
                minimum_rows=3,
                chunk_rows=16_384,
            )
        self.assertEqual(candidates, ())
        self.assertEqual(equality.call_count, 0)


if __name__ == "__main__":
    unittest.main()
