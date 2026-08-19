from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from wild_preprocess.analog.integrity import scan_analog_frames, scan_analog_integrity
from wild_preprocess.analog.models import DeviceClockPrior
from wild_preprocess.analog.segments import build_event_driven_analog_segments, map_canonical_rows


def _frames(rows: int, *, start_tick: int = 0) -> np.ndarray:
    """Dynamic complete frames with normal held IMU updates every six rows."""

    result = np.zeros((rows, 16), dtype="<i2")
    result[:, 0] = np.arange(rows, dtype=np.int16) * 3
    result[:, 10] = np.arange(rows, dtype=np.int16) * 7
    result[:, 11] = (np.arange(rows, dtype=np.uint32) + start_tick).astype(np.uint16).view(np.int16)
    update = (np.arange(rows) // 6 + 1).astype(np.int16)
    for lane in range(1, 10):
        result[:, lane] = update * (lane + 1)
    result[:, 12] = np.arange(rows, dtype=np.int16) * 11
    result[:, 13] = np.arange(rows, dtype=np.int16) * 13
    result[:, 14] = np.arange(rows, dtype=np.int16) * 17
    result[:, 15] = np.arange(rows, dtype=np.int16) * 19
    return result


class AnalogIntegrityTest(unittest.TestCase):
    def _kinds(self, result: object) -> list[str]:
        return [event.kind for event in result.events]  # type: ignore[attr-defined]

    def test_clean_dynamic_frames_and_normal_imu_holds_are_clean(self) -> None:
        result = scan_analog_frames(_frames(240), imu_stall_min_rows=30)
        self.assertTrue(result.clean)
        self.assertEqual(result.metrics.counter_nonunit_delta_count, 0)
        self.assertAlmostEqual(result.metrics.imu_median_update_rows or 0.0, 6.0)

    def test_counter_wrap_is_not_a_phase_event(self) -> None:
        result = scan_analog_frames(_frames(12, start_tick=65_530))
        self.assertTrue(result.clean)
        self.assertEqual(result.metrics.counter_wrap_count, 1)

    def test_one_row_counter_glitch_is_counter_corruption_not_missing(self) -> None:
        frames = _frames(64)
        frames[20, 11] = np.uint16(400).view(np.int16)
        result = scan_analog_frames(frames)
        event = next(event for event in result.events if event.kind == "counter_corruption")
        self.assertEqual(event.affected_lanes, (11,))
        self.assertEqual(result.valid_raw_support_runs(), ((0, 64),))
        self.assertNotIn("missing", self._kinds(result))

    def test_persistent_positive_phase_is_missing_candidate(self) -> None:
        frames = _frames(80)
        frames[31:, 11] = (np.arange(49, dtype=np.uint32) + 36).astype(np.uint16).view(np.int16)
        result = scan_analog_frames(frames)
        event = next(event for event in result.events if event.kind == "missing")
        self.assertEqual((event.raw_start_row, event.raw_end_row), (31, 32))
        self.assertEqual(event.displacement_rows, 5)
        self.assertEqual(result.valid_raw_support_runs(), ((0, 31), (32, 80)))
        self.assertNotIn("temporary_excursion", self._kinds(result))

    def test_persistent_phase_events_use_incremental_displacements(self) -> None:
        frames = _frames(140)
        frames[30:70, 11] = (np.arange(30, 70, dtype=np.uint32) + 5).astype(np.uint16).view(np.int16)
        frames[70:, 11] = (np.arange(70, 140, dtype=np.uint32) + 8).astype(np.uint16).view(np.int16)
        result = scan_analog_frames(frames)
        events = [event for event in result.events if event.kind == "missing"]
        self.assertEqual(
            [(event.raw_start_row, event.raw_end_row, event.displacement_rows) for event in events],
            [(30, 31, 5), (70, 71, 3)],
        )
        self.assertEqual([(event.tick_start, event.tick_end) for event in events], [(35, 36), (78, 79)])

    def test_persistent_negative_phase_localizes_inserted_rows_then_reacquires_tail(self) -> None:
        base = _frames(80)
        inserted = _frames(7, start_tick=40)
        inserted[:, 0] = -123
        inserted[:, 10] = -456
        inserted[:, 12:] = -789
        frames = np.vstack((base[:40], inserted, base[40:]))
        result = scan_analog_frames(frames, device_index=3)
        event = next(event for event in result.events if event.kind == "insertion")
        self.assertEqual((event.raw_start_row, event.raw_end_row), (40, 47))
        self.assertEqual(event.device_index, 3)
        self.assertEqual(result.device_index, 3)
        self.assertEqual(result.valid_raw_support_runs(), ((0, 40), (47, 87)))

    def test_arbitrary_lag_temporary_full_frame_replay_is_overwrite(self) -> None:
        frames = _frames(160)
        start, length, lag = 80, 17, 23
        frames[start : start + length] = frames[start - lag : start + length - lag]
        result = scan_analog_frames(frames, chunk_rows=7, imu_stall_min_rows=3)
        event = next(event for event in result.events if event.kind == "repeat_overwrite")
        self.assertEqual((event.raw_start_row, event.raw_end_row), (start, start + length))
        self.assertEqual(event.displacement_rows, -lag)
        self.assertEqual(result.metrics.confirmed_repeat_rows, length)

    def test_one_and_two_row_complete_frame_replays_are_not_counter_only(self) -> None:
        for length in (1, 2):
            with self.subTest(length=length):
                frames = _frames(100)
                start, lag = 50, 7
                frames[start : start + length] = frames[start - lag : start + length - lag]
                result = scan_analog_frames(frames)
                event = next(event for event in result.events if event.kind == "repeat_overwrite")
                self.assertEqual((event.raw_start_row, event.raw_end_row), (start, start + length))
                self.assertEqual(event.affected_lanes, tuple(range(16)))
                self.assertEqual(result.valid_raw_support_runs(), ((0, start), (start + length, 100)))

    def test_persistent_exact_replay_is_repeat_insertion(self) -> None:
        base = _frames(120)
        inserted = base[40:54].copy()
        frames = np.vstack((base[:54], inserted, base[54:]))
        result = scan_analog_frames(frames, chunk_rows=5)
        event = next(event for event in result.events if event.kind == "repeat_insertion")
        self.assertEqual(event.raw_start_row, 54)
        self.assertEqual(event.raw_end_row, 68)
        self.assertEqual(event.displacement_rows, -14)
        self.assertEqual(result.valid_raw_support_runs(), ((0, 54), (68, 134)))

    def test_returned_negative_counter_excursion_without_payload_repeat_is_local_invalid(self) -> None:
        frames = _frames(100)
        frames[40:48, 11] = (np.arange(8, dtype=np.uint32) + 33).astype(np.uint16).view(np.int16)
        result = scan_analog_frames(frames)
        event = next(event for event in result.events if event.kind == "temporary_excursion")
        self.assertEqual((event.raw_start_row, event.raw_end_row), (40, 48))
        self.assertEqual(result.valid_raw_support_runs(), ((0, 40), (48, 100)))
        self.assertNotIn("repeat_overwrite", self._kinds(result))

    def test_returned_forward_counter_excursion_is_local_invalid(self) -> None:
        frames = _frames(100)
        frames[40:48, 11] = (np.arange(8, dtype=np.uint32) + 47).astype(np.uint16).view(np.int16)
        result = scan_analog_frames(frames)
        event = next(event for event in result.events if event.kind == "temporary_excursion")
        self.assertEqual((event.raw_start_row, event.raw_end_row), (40, 48))
        self.assertEqual(event.displacement_rows, 7)
        self.assertEqual(event.confidence, "medium")
        self.assertEqual(result.valid_raw_support_runs(), ((0, 40), (48, 100)))

        prior = DeviceClockPrior(
            device_index=1,
            source_ephys_scale=1.0,
            source_ephys_intercept_samples=0.0,
            canonical_ephys_start_sample=0.0,
            ephys_sample_rate_hz=1_250.0,
            support_ids=("counter-anchor-0", "counter-anchor-1"),
            confidence="medium",
        )
        segments = build_event_driven_analog_segments(
            prior,
            canonical_start_row=0,
            canonical_end_row=100,
            raw_row_count=100,
            decisions=(event,),
        )
        _, valid, _ = map_canonical_rows(segments, np.arange(100), raw_row_count=100)
        self.assertTrue(np.all(valid[:40]))
        self.assertFalse(np.any(valid[40:48]))
        self.assertTrue(np.all(valid[48:]))

    def test_nonrepeat_gap_inside_returned_negative_excursion_is_local_invalid(self) -> None:
        frames = _frames(100)
        # The first three rows are a confirmed lag-seven replay; the rest
        # retain their own payload while the counter remains seven rows back.
        frames[40:43] = frames[33:36]
        frames[43:48, 11] = (np.arange(43, 48, dtype=np.uint32) - 7).astype(np.uint16).view(np.int16)
        result = scan_analog_frames(frames)
        repeat = next(event for event in result.events if event.kind == "repeat_overwrite")
        temporary = next(event for event in result.events if event.kind == "temporary_excursion")
        self.assertEqual((repeat.raw_start_row, repeat.raw_end_row), (40, 43))
        self.assertEqual((temporary.raw_start_row, temporary.raw_end_row), (43, 48))
        self.assertEqual(result.valid_raw_support_runs(), ((0, 40), (48, 100)))

    def test_stationary_imu_without_other_payload_activity_is_not_stall(self) -> None:
        frames = _frames(100)
        frames[:, 1:10] = 123
        frames[:, 0] = 0
        frames[:, 10] = 0
        frames[:, 12:] = 0
        result = scan_analog_frames(frames, imu_stall_min_rows=30)
        self.assertNotIn("sensor_stall", self._kinds(result))
        self.assertEqual(result.metrics.imu_longest_static_rows, 100)

    def test_imu_stall_requires_independent_payload_activity(self) -> None:
        frames = _frames(100)
        frames[20:70, 1:10] = frames[20, 1:10]
        result = scan_analog_frames(frames, imu_stall_min_rows=30)
        event = next(event for event in result.events if event.kind == "sensor_stall")
        self.assertEqual((event.raw_start_row, event.raw_end_row), (18, 70))
        self.assertEqual(event.affected_lanes, tuple(range(1, 10)))

    def test_valid_raw_support_never_bridges_a_confirmed_overwrite(self) -> None:
        frames = _frames(100)
        frames[50:60] = frames[31:41]
        result = scan_analog_frames(frames)
        self.assertEqual(result.valid_raw_support_runs(), ((0, 50), (60, 100)))

    def test_counter_continuous_dynamic_payload_overwrite_is_local(self) -> None:
        frames = _frames(150)
        frames[80:91, [*range(0, 11), *range(12, 16)]] = frames[51:62, [*range(0, 11), *range(12, 16)]]
        result = scan_analog_frames(
            frames,
            chunk_rows=7,
            payload_repeat_max_lag_rows=64,
        )
        event = next(event for event in result.events if event.kind == "repeat_overwrite")
        self.assertEqual((event.raw_start_row, event.raw_end_row), (80, 91))
        self.assertEqual(event.affected_lanes, tuple([*range(0, 11), *range(12, 16)]))
        self.assertEqual(result.valid_raw_support_runs(), ((0, 80), (91, 150)))

    def test_counter_continuous_payload_replay_detects_large_nonhistorical_lag(self) -> None:
        frames = _frames(1_100)
        start, length, lag = 700, 19, 333
        lanes = [*range(0, 11), *range(12, 16)]
        frames[start : start + length, lanes] = frames[start - lag : start + length - lag, lanes]
        result = scan_analog_frames(frames, chunk_rows=64)
        event = next(event for event in result.events if event.kind == "repeat_overwrite")
        self.assertEqual((event.raw_start_row, event.raw_end_row), (start, start + length))
        self.assertIn("lag 333 rows", event.evidence)

    def test_counter_fresh_payload_replay_survives_normal_sample_holds(self) -> None:
        frames = _frames(160)
        lanes = [*range(0, 11), *range(12, 16)]
        # Make every non-counter lane update only every six analog rows,
        # matching normal sample-hold behaviour.  The counter remains fresh.
        for lane in lanes:
            frames[:, lane] = (np.arange(160) // 6 * (lane + 3)).astype(np.int16)
        start, length, lag = 96, 24, 48
        frames[start : start + length, lanes] = frames[start - lag : start + length - lag, lanes]
        result = scan_analog_frames(frames, payload_repeat_max_lag_rows=64)
        event = next(event for event in result.events if event.kind == "repeat_overwrite")
        self.assertEqual((event.raw_start_row, event.raw_end_row), (start, start + length))
        self.assertEqual(event.affected_lanes, tuple(lanes))

    def test_stationary_payload_is_not_counter_independent_repeat(self) -> None:
        frames = _frames(100)
        frames[:, [*range(0, 11), *range(12, 16)]] = 42
        result = scan_analog_frames(frames, payload_repeat_max_lag_rows=32)
        self.assertNotIn("unresolved", self._kinds(result))

    def test_imu_zero_and_saturation_are_modality_only_events(self) -> None:
        frames = _frames(90)
        frames[11:19, 1:10] = 0
        frames[33:37, 3] = np.iinfo(np.int16).max
        frames[42:45, 3] = np.iinfo(np.int16).min
        result = scan_analog_frames(frames, chunk_rows=7, imu_stall_min_rows=3)
        zero = next(event for event in result.events if event.kind == "imu_all_zero")
        saturation = [event for event in result.events if event.kind == "imu_saturation"]
        self.assertEqual((zero.raw_start_row, zero.raw_end_row, zero.affected_lanes), (11, 19, tuple(range(1, 10))))
        self.assertEqual(
            [(event.raw_start_row, event.raw_end_row, event.affected_lanes) for event in saturation],
            [(33, 37, (3,)), (42, 45, (3,))],
        )
        self.assertEqual(result.valid_raw_support_runs(), ((0, 90),))

    def test_path_scanner_is_read_only_and_rejects_truncated_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "analogin.dat"
            frames = _frames(40)
            frames.tofile(path)
            before = path.read_bytes()
            result = scan_analog_integrity(path, channel_count=16)
            self.assertTrue(result.clean)
            self.assertEqual(path.read_bytes(), before)
            with (Path(temporary) / "truncated.dat").open("wb") as stream:
                stream.write(before[:-1])
            with self.assertRaisesRegex(ValueError, "framing"):
                scan_analog_integrity(Path(temporary) / "truncated.dat", channel_count=16)


if __name__ == "__main__":
    unittest.main()
