from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from wild_preprocess.analog.models import (
    AnalogSyncAnchor,
    AnalogSyncSegment,
    AnalogTimelineResult,
    DeviceClockPrior,
    validate_analog_sync_segments,
)
from wild_preprocess.analog.segments import (
    build_clean_analog_segments,
    build_event_driven_analog_segments,
    map_canonical_rows,
    map_raw_rows_to_canonical,
)


def _segment(
    canonical_start: int,
    canonical_end: int,
    raw_start: int,
    raw_end: int,
    *,
    scale: float = 1.0,
    intercept: float = 0.0,
    device_index: int = 1,
) -> AnalogSyncSegment:
    anchors = (
        AnalogSyncAnchor(canonical_start, scale * canonical_start + intercept, True, "high"),
        AnalogSyncAnchor(canonical_end - 1, scale * (canonical_end - 1) + intercept, True, "high"),
    )
    return AnalogSyncSegment(
        device_index=device_index,
        canonical_start_row=canonical_start,
        canonical_end_row=canonical_end,
        raw_start_row=raw_start,
        raw_end_row=raw_end,
        raw_scale=scale,
        raw_intercept_rows=intercept,
        anchors=anchors,
        confidence="high",
        publishable=True,
    )


class AnalogSegmentTests(unittest.TestCase):
    @staticmethod
    def _prior(*, device_index: int = 1, intercept_ephys: float = 0.0, support_prefix: str = "n") -> DeviceClockPrior:
        return DeviceClockPrior(
            device_index=device_index,
            source_ephys_scale=1.0,
            source_ephys_intercept_samples=intercept_ephys,
            canonical_ephys_start_sample=0.0,
            ephys_sample_rate_hz=20_000.0,
            support_ids=(f"{support_prefix}-anchor-0", f"{support_prefix}-anchor-1"),
            confidence="medium",
        )

    def test_forward_mapping_is_monotone_in_range_and_gap_is_invalid(self) -> None:
        segments = (
            _segment(0, 10, 0, 10),
            _segment(20, 30, 15, 25, intercept=-5.0),
        )
        rows = np.arange(30, dtype=np.int64)
        raw, valid, segment_ids = map_canonical_rows(segments, rows, raw_row_count=25)
        np.testing.assert_array_equal(np.flatnonzero(valid), np.r_[0:10, 20:30])
        np.testing.assert_allclose(raw[valid], np.r_[0:10, 15:25])
        self.assertTrue(np.all(np.diff(raw[valid]) > 0))
        self.assertTrue(np.all(np.isnan(raw[10:20])))
        self.assertTrue(np.all(segment_ids[10:20] == -1))
        np.testing.assert_array_equal(segment_ids[valid], np.r_[np.zeros(10), np.ones(10)])

    def test_overlap_and_source_reversal_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-overlapping"):
            validate_analog_sync_segments((_segment(0, 10, 0, 10), _segment(9, 20, 9, 20)))
        with self.assertRaisesRegex(ValueError, "strictly forward"):
            validate_analog_sync_segments(
                (_segment(0, 10, 10, 20, intercept=10.0), _segment(20, 30, 0, 10, intercept=-20.0))
            )

    def test_inverse_rejects_skipped_raw_rows_and_reports_segment_ids(self) -> None:
        segments = (
            _segment(0, 10, 0, 10),
            _segment(20, 30, 15, 25, intercept=-5.0),
        )
        canonical, valid, segment_ids = map_raw_rows_to_canonical(
            segments, np.array([0, 9, 10, 14, 15, 24], dtype=np.int64)
        )
        np.testing.assert_allclose(canonical, np.array([0.0, 9.0, np.nan, np.nan, 20.0, 29.0]), equal_nan=True)
        np.testing.assert_array_equal(valid, np.array([True, True, False, False, True, True]))
        np.testing.assert_array_equal(segment_ids, np.array([0, 0, -1, -1, 1, 1]))

    def test_fractional_mapping_requires_filter_support_within_one_segment(self) -> None:
        segment = _segment(0, 10, 0, 11, scale=1.0, intercept=0.5)
        raw, valid, segment_ids = map_canonical_rows(
            (segment,), np.arange(10, dtype=np.int64), raw_row_count=10, interpolation_half_width=1
        )
        self.assertTrue(valid[0])
        self.assertFalse(valid[-1])
        self.assertEqual(segment_ids[-1], -1)
        self.assertTrue(np.isfinite(raw[-1]))

    def test_fractional_inverse_retains_sub_row_coordinate(self) -> None:
        segment = _segment(0, 1_000, 0, 1_001, scale=1.0002)
        raw = np.array([500.0], dtype=np.float64)
        canonical, valid, segment_ids = map_raw_rows_to_canonical((segment,), raw)
        self.assertTrue(valid[0])
        self.assertEqual(segment_ids[0], 0)
        self.assertAlmostEqual(canonical[0], 500.0 / 1.0002, places=12)

    def test_clean_builder_splits_raw_and_canonical_exclusions_without_neural_step_copy(self) -> None:
        prior = DeviceClockPrior(
            device_index=2,
            source_ephys_scale=1.0001,
            source_ephys_intercept_samples=0.0,
            canonical_ephys_start_sample=0.0,
            ephys_sample_rate_hz=20_000.0,
            support_ids=("neural-anchor-0", "neural-anchor-1"),
            confidence="medium",
        )
        segments = build_clean_analog_segments(
            prior,
            canonical_start_row=0,
            canonical_end_row=100,
            raw_row_count=120,
            excluded_raw_intervals=((20, 25),),
            excluded_canonical_intervals=((60, 65),),
        )
        self.assertEqual(
            [(item.canonical_start_row, item.canonical_end_row) for item in segments],
            [(0, 20), (25, 60), (65, 100)],
        )
        self.assertTrue(all(item.publishable for item in segments))
        raw, valid, _ = map_canonical_rows(segments, np.arange(100), raw_row_count=120)
        self.assertFalse(np.any(valid[20:25]))
        self.assertFalse(np.any(valid[60:65]))
        self.assertTrue(np.all(np.diff(raw[valid]) > 0))

    def test_clean_builder_keeps_upper_neighbour_for_fractional_endpoint_interpolation(self) -> None:
        prior = DeviceClockPrior(
            device_index=1,
            source_ephys_scale=1.0,
            source_ephys_intercept_samples=10.0,
            canonical_ephys_start_sample=0.0,
            ephys_sample_rate_hz=20_000.0,
            support_ids=("neural-anchor-0", "neural-anchor-1"),
            confidence="medium",
        )
        # The ephys intercept maps to 0.625 analog rows.  Row 9 therefore
        # needs raw rows 9 and 10 for linear interpolation.
        segments = build_clean_analog_segments(
            prior,
            canonical_start_row=0,
            canonical_end_row=10,
            raw_row_count=11,
        )
        _, valid, _ = map_canonical_rows(
            segments, np.arange(10), raw_row_count=11, interpolation_half_width=1
        )
        self.assertTrue(valid[-1])
        _, no_kernel_valid, _ = map_canonical_rows(
            segments, np.arange(10), raw_row_count=11, interpolation_half_width=0
        )
        self.assertFalse(np.any(no_kernel_valid))

    def test_underanchored_segment_is_never_publishable(self) -> None:
        anchor = AnalogSyncAnchor(0, 0.0, True, "high")
        with self.assertRaisesRegex(ValueError, "at least two"):
            AnalogSyncSegment(
                device_index=1,
                canonical_start_row=0,
                canonical_end_row=1,
                raw_start_row=0,
                raw_end_row=1,
                raw_scale=1.0,
                raw_intercept_rows=0.0,
                anchors=(anchor,),
                confidence="high",
                publishable=True,
            )

    def test_under_supported_clock_prior_does_not_fabricate_clean_publishability(self) -> None:
        prior = DeviceClockPrior(
            device_index=1,
            source_ephys_scale=1.0,
            source_ephys_intercept_samples=0.0,
            canonical_ephys_start_sample=0.0,
            ephys_sample_rate_hz=20_000.0,
            support_ids=("one-neural-anchor",),
            confidence="medium",
        )
        segment = build_clean_analog_segments(
            prior, canonical_start_row=0, canonical_end_row=10, raw_row_count=10
        )[0]
        self.assertFalse(segment.publishable)
        self.assertFalse(any(anchor.verified for anchor in segment.anchors))

    def test_event_builder_applies_missing_and_insertion_as_distinct_mapping_corrections(self) -> None:
        prior = self._prior()
        missing = build_event_driven_analog_segments(
            prior,
            canonical_start_row=0,
            canonical_end_row=30,
            raw_row_count=30,
            decisions=({"kind": "missing", "raw_start": 10, "raw_end": 11, "displacement": 2, "confidence": "high"},),
        )
        raw, valid, _ = map_canonical_rows(missing, np.arange(30), raw_row_count=30)
        self.assertFalse(np.any(valid[10:12]))
        self.assertEqual(raw[12], 10.0)
        insertion = build_event_driven_analog_segments(
            prior,
            canonical_start_row=0,
            canonical_end_row=30,
            raw_row_count=32,
            # The raw interval [8,10) is excess data.  Canonical row 8 resumes
            # at raw row 10, so the inserted rows themselves are skipped.
            decisions=({"kind": "insertion", "raw_start": 8, "raw_end": 10, "displacement": -2, "confidence": "high"},),
        )
        raw, valid, _ = map_canonical_rows(insertion, np.arange(30), raw_row_count=32)
        self.assertTrue(np.all(valid))
        self.assertEqual(raw[8], 10.0)
        _, inverse_valid, _ = map_raw_rows_to_canonical(insertion, np.array([8.0, 9.0, 10.0]))
        np.testing.assert_array_equal(inverse_valid, np.array([False, False, True]))

    def test_repeat_insertion_skips_actual_destination_length_not_repeat_lag(self) -> None:
        segments = build_event_driven_analog_segments(
            self._prior(),
            canonical_start_row=0,
            canonical_end_row=60,
            raw_row_count=67,
            # A seven-row replay was detected at lag 19.  Only its seven
            # destination rows are inserted and therefore skipped.
            decisions=(
                {
                    "kind": "repeat_insertion",
                    "raw_start": 40,
                    "raw_end": 47,
                    "displacement": -19,
                    "confidence": "high",
                },
            ),
        )
        raw, valid, _ = map_canonical_rows(segments, np.arange(60), raw_row_count=67)
        self.assertTrue(np.all(valid))
        self.assertEqual(raw[40], 47.0)
        _, inverse_valid, _ = map_raw_rows_to_canonical(segments, np.arange(40, 48, dtype=float))
        np.testing.assert_array_equal(inverse_valid, np.array([False] * 7 + [True]))

    def test_event_builder_excludes_repeat_overwrite_without_shifting_tail(self) -> None:
        segments = build_event_driven_analog_segments(
            self._prior(),
            canonical_start_row=0,
            canonical_end_row=30,
            raw_row_count=30,
            decisions=({"kind": "repeat_overwrite", "raw_start": 10, "raw_end": 13, "confidence": "high"},),
        )
        raw, valid, _ = map_canonical_rows(segments, np.arange(30), raw_row_count=30)
        self.assertFalse(np.any(valid[10:13]))
        self.assertEqual(raw[13], 13.0)

    def test_prior_residual_metadata_is_preserved(self) -> None:
        prior = DeviceClockPrior(
            device_index=1,
            source_ephys_scale=1.0,
            source_ephys_intercept_samples=0.0,
            canonical_ephys_start_sample=0.0,
            ephys_sample_rate_hz=20_000.0,
            support_ids=("n-anchor-0", "n-anchor-1"),
            residual_rms_rows=0.2,
            residual_max_abs_rows=0.5,
            confidence="medium",
        )
        self.assertEqual(
            build_clean_analog_segments(
                prior, canonical_start_row=0, canonical_end_row=10, raw_row_count=10
            )[0].residual_max_abs_rows,
            0.5,
        )

    def test_clean_mapping_clips_only_the_unsupported_tail(self) -> None:
        segments = build_clean_analog_segments(
            self._prior(),
            canonical_start_row=0,
            canonical_end_row=100,
            raw_row_count=90,
        )
        self.assertEqual(len(segments), 1)
        self.assertEqual(
            (segments[0].canonical_start_row, segments[0].canonical_end_row),
            (0, 90),
        )
        _, valid, _ = map_canonical_rows(
            segments, np.arange(100), raw_row_count=90
        )
        self.assertTrue(np.all(valid[:90]))
        self.assertFalse(np.any(valid[90:]))

    def test_clean_mapping_clips_only_the_unsupported_head(self) -> None:
        prior = DeviceClockPrior(
            device_index=1,
            source_ephys_scale=1.0,
            source_ephys_intercept_samples=-80.0,
            canonical_ephys_start_sample=0.0,
            ephys_sample_rate_hz=20_000.0,
            support_ids=("anchor-a", "anchor-b"),
            confidence="high",
        )
        segments = build_clean_analog_segments(
            prior,
            canonical_start_row=0,
            canonical_end_row=20,
            raw_row_count=20,
        )
        self.assertEqual(
            (segments[0].canonical_start_row, segments[0].canonical_end_row),
            (5, 20),
        )

    def test_unresolved_event_never_propagates_step_without_explicit_reacquisition(self) -> None:
        prior = self._prior()
        decision = {"kind": "unresolved", "raw_start": 10, "raw_end": 11, "confidence": "unresolved"}
        segments = build_event_driven_analog_segments(
            prior,
            canonical_start_row=0,
            canonical_end_row=30,
            raw_row_count=30,
            decisions=(decision,),
        )
        _, valid, _ = map_canonical_rows(segments, np.arange(30), raw_row_count=30)
        self.assertTrue(np.all(valid[:10]))
        self.assertFalse(np.any(valid[10:]))
        reacquired = build_event_driven_analog_segments(
            prior,
            canonical_start_row=0,
            canonical_end_row=30,
            raw_row_count=30,
            decisions=(decision,),
            reacquisition_priors=((15, self._prior(support_prefix="reacquired")),),
        )
        _, valid, _ = map_canonical_rows(reacquired, np.arange(30), raw_row_count=30)
        self.assertFalse(np.any(valid[10:15]))
        self.assertTrue(np.all(valid[15:]))

    def test_timeline_serialization_is_deterministic(self) -> None:
        segment = _segment(0, 10, 0, 10)
        prior = self._prior()
        result = AnalogTimelineResult(
            device_index=1,
            segments=(segment,),
            integrity_events=({"kind": "repeat_overwrite", "raw_start_row": 10},),
            clock_prior=prior,
            status="WARN",
            warnings=("one local interval excluded",),
            source_raw_row_count=10,
        )
        first = json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":"))
        second = json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":"))
        self.assertEqual(first, second)
        self.assertEqual(first, result.to_json())
        self.assertEqual(result.mapping_hash, AnalogTimelineResult(
            device_index=1, segments=(segment,), clock_prior=prior, source_raw_row_count=10
        ).mapping_hash)

    def test_timeline_rejects_cross_device_event_or_prior(self) -> None:
        with self.assertRaisesRegex(ValueError, "integrity event device_index"):
            AnalogTimelineResult(
                device_index=1,
                integrity_events=({"device_index": 2, "kind": "missing"},),
            )
        with self.assertRaisesRegex(ValueError, "clock_prior device_index"):
            AnalogTimelineResult(device_index=1, clock_prior=self._prior(device_index=2))

    def test_many_prior_support_ids_are_not_repeated_in_every_segment(self) -> None:
        support_ids = tuple(f"anchor-{index:08d}" for index in range(500))
        prior = DeviceClockPrior(
            device_index=1,
            source_ephys_scale=1.0,
            source_ephys_intercept_samples=0.0,
            canonical_ephys_start_sample=0.0,
            ephys_sample_rate_hz=20_000.0,
            support_ids=support_ids,
            confidence="high",
        )
        segments = build_clean_analog_segments(
            prior,
            canonical_start_row=0,
            canonical_end_row=100,
            raw_row_count=100,
            excluded_canonical_intervals=((20, 30), (50, 60)),
        )
        serialized_segments = json.dumps(
            [segment.to_dict() for segment in segments], separators=(",", ":")
        )
        self.assertLess(len(serialized_segments), 10_000)
        self.assertIn("support_count=500", serialized_segments)
        self.assertNotIn("anchor-00000499", serialized_segments)
        self.assertEqual(prior.support_ids, support_ids)


if __name__ == "__main__":
    unittest.main()
