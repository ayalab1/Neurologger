from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1] / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from wild_preprocess.integrity import (
    device_gap_to_interval,
    merge_compatible_intervals,
    duplicate_destination_intervals,
    source_sample_to_canonical,
    source_steps_from_unresolved_offset_clusters,
    terminal_support_from_pair,
    terminal_support_to_interval,
    unresolved_boundaries_from_offset_clusters,
    unresolved_boundary_to_interval,
    zero_fill_spans,
)
from wild_preprocess.models import (
    ClassifiedInterval,
    DeviceGap,
    DeviceSourceStep,
    RelativeOffsetStep,
    SyncModel,
    SyncObservation,
    SyncPairResult,
)


def _model(steps: tuple[RelativeOffsetStep, ...] = ()) -> SyncModel:
    return SyncModel(
        intercept_samples=0.0,
        slope_samples_per_second=0.0,
        drift_ppm=0.0,
        residual_rms_samples=0.0,
        residual_max_abs_samples=0.0,
        accepted_count=4,
        observation_count=4,
        offset_steps=steps,
    )


def _step(value: float, *, sample: int = 100, time_sec: float = 1.0) -> RelativeOffsetStep:
    return RelativeOffsetStep(
        master_sample=sample,
        time_sec=time_sec,
        offset_step_samples=value,
        missing_samples=round(abs(value)),
        offset_before_samples=0.0,
        offset_after_samples=value,
        confidence="high",
        evidence="synthetic persistent step",
    )


def _observation(time_sec: float) -> SyncObservation:
    return SyncObservation(
        center_time_sec=time_sec,
        predicted_offset_samples=0.0,
        observed_offset_samples=0.0,
        residual_lag_samples=0.0,
        peak_correlation=0.5,
        peak_to_background=2.0,
        peak_margin_fraction=0.2,
        secondary_lag_samples=None,
        accepted=True,
    )


def _pair(slave_index: int, steps: tuple[RelativeOffsetStep, ...]) -> SyncPairResult:
    return SyncPairResult(
        master_index=1,
        slave_index=slave_index,
        master_folder="master",
        slave_folder=f"slave{slave_index}",
        initial_offset_samples=0.0,
        initial_peak_to_background=2.0,
        initial_peak_margin_fraction=0.2,
        model=_model(steps),
        observations=[_observation(0.75), _observation(1.25)],
        status="WARN",
    )


class ClassifiedIntervalsTest(unittest.TestCase):
    def test_localized_master_samples_control_cluster_attribution(self) -> None:
        first = _step(100.0, sample=1_000, time_sec=1.0)
        second = _step(102.0, sample=1_005, time_sec=9.0)
        pairs = [_pair(2, (first,)), _pair(3, (second,))]
        boundaries = unresolved_boundaries_from_offset_clusters(
            pairs,
            device_count=3,
            master_index=0,
            fs=1_000.0,
            window_seconds=1.0,
            fallback_step_seconds=0.5,
            event_time_tolerance_seconds=0.1,
            gap_level_tolerance_samples=5.0,
        )
        source_steps = source_steps_from_unresolved_offset_clusters(
            pairs,
            device_count=3,
            master_index=0,
            fs=1_000.0,
            event_time_tolerance_seconds=0.1,
            gap_level_tolerance_samples=5.0,
        )
        self.assertFalse(boundaries)
        self.assertFalse(source_steps)

    def test_source_inverse_selects_supported_side_of_device_gap(self) -> None:
        gap = DeviceGap(2, 100, 5, 5.0)
        self.assertEqual(
            source_sample_to_canonical(
                50,
                device_index=2,
                source_scale=1.0,
                intercept_samples=0.0,
                device_gaps=[gap],
            ),
            50,
        )
        self.assertEqual(
            source_sample_to_canonical(
                100,
                device_index=2,
                source_scale=1.0,
                intercept_samples=0.0,
                device_gaps=[gap],
            ),
            105,
        )
        second_gap = DeviceGap(2, 205, 5, 5.0)
        self.assertEqual(
            source_sample_to_canonical(
                200,
                device_index=2,
                source_scale=1.0,
                intercept_samples=0.0,
                device_gaps=[gap, second_gap],
            ),
            210,
        )
        self.assertEqual(
            source_sample_to_canonical(
                305,
                device_index=2,
                source_scale=1.0,
                intercept_samples=0.0,
                source_steps=[DeviceSourceStep(2, 300, 5.0, "unresolved")],
            ),
            300,
        )
        for skipped in range(300, 305):
            self.assertIsNone(
                source_sample_to_canonical(
                    skipped,
                    device_index=2,
                    source_scale=1.0,
                    intercept_samples=0.0,
                    source_steps=[DeviceSourceStep(2, 300, 5.0, "unresolved")],
                )
            )
        self.assertIsNone(
            source_sample_to_canonical(
                295,
                device_index=2,
                source_scale=1.0,
                intercept_samples=0.0,
                source_steps=[DeviceSourceStep(2, 300, -5.0, "unresolved")],
            )
        )

    def test_validates_one_based_half_open_coordinates_and_action_pairs(self) -> None:
        with self.assertRaises(ValueError):
            ClassifiedInterval((0,), 1, 2, "missing", "zero_fill", "high")
        with self.assertRaises(ValueError):
            ClassifiedInterval((1,), 2, 2, "missing", "zero_fill", "high")
        with self.assertRaises(ValueError):
            ClassifiedInterval((1,), 1, 2, "missing", "skip_source", "high")
        with self.assertRaises(ValueError):
            ClassifiedInterval((1,), 1, 2, "duplicate_destination", "zero_fill", "high")

    def test_device_gap_remains_exact_until_render_guard_is_requested(self) -> None:
        interval = device_gap_to_interval(DeviceGap(2, 100, 5, 5.0, evidence="gap"))
        self.assertEqual((interval.canonical_start_sample, interval.canonical_end_sample), (100, 105))
        self.assertEqual(interval.kind, "missing")
        self.assertEqual(interval.action, "zero_fill")
        self.assertEqual(
            zero_fill_spans(
                (interval,),
                device_index=2,
                canonical_samples=200,
                guard_samples=16,
                guarded_kinds=frozenset({"missing"}),
            ),
            ((84, 121),),
        )

    def test_duplication_uses_exact_fragments_and_later_occurrence_policy(self) -> None:
        scan = {
            "episodes": [
                {
                    "lag_samples": 20,
                    "duplicate_start_sample": 100,
                    "duplicate_end_sample": 140,
                    "exact_duplicate_fragments": [[100, 110], [120, 125]],
                }
            ]
        }
        intervals = duplicate_destination_intervals(scan, device_index=3)
        self.assertEqual(
            [(item.canonical_start_sample, item.canonical_end_sample) for item in intervals],
            [(100, 110), (120, 125)],
        )
        self.assertEqual(
            [(item.source_start_sample, item.source_end_sample) for item in intervals],
            [(80, 90), (100, 105)],
        )
        self.assertTrue(all(item.kind == "duplicate_destination" for item in intervals))

    def test_duplicate_fragment_drops_raw_samples_without_canonical_preimage(self) -> None:
        scan = {
            "episodes": [
                {"lag_samples": 10, "exact_duplicate_fragments": [[300, 310]]}
            ]
        }
        step = DeviceSourceStep(2, 300, 5.0, "unresolved")
        intervals = duplicate_destination_intervals(
            scan,
            device_index=2,
            canonicalize_current_sample=lambda sample: source_sample_to_canonical(
                sample,
                device_index=2,
                source_scale=1.0,
                intercept_samples=0.0,
                source_steps=[step],
            ),
        )
        self.assertEqual(len(intervals), 1)
        self.assertEqual(
            (intervals[0].canonical_start_sample, intervals[0].canonical_end_sample),
            (300, 305),
        )
        self.assertTrue(all("later occurrence" in item.evidence for item in intervals))

    def test_only_contiguous_equivalent_intervals_merge(self) -> None:
        first = ClassifiedInterval((2,), 100, 110, "duplicate_destination", "zero_fill", "medium", 80, 90, "x")
        second = ClassifiedInterval((2,), 110, 115, "duplicate_destination", "zero_fill", "medium", 90, 95, "x")
        different_source = ClassifiedInterval((2,), 115, 120, "duplicate_destination", "zero_fill", "medium", 1, 6, "x")
        merged = merge_compatible_intervals((different_source, second, first))
        self.assertEqual([(item.canonical_start_sample, item.canonical_end_sample) for item in merged], [(100, 115), (115, 120)])

    def test_positive_single_pair_is_unresolved_not_extra_source(self) -> None:
        boundaries = unresolved_boundaries_from_offset_clusters(
            [_pair(2, (_step(317.0),)), _pair(3, ())],
            device_count=3,
            master_index=0,
            fs=100.0,
            window_seconds=0.5,
            fallback_step_seconds=0.25,
            event_time_tolerance_seconds=0.25,
            gap_level_tolerance_samples=12.0,
        )
        self.assertEqual(len(boundaries), 1)
        interval = unresolved_boundary_to_interval(boundaries[0], device_count=3)
        self.assertEqual(interval.affected_device_indices, (1, 2, 3))
        self.assertEqual(interval.kind, "unresolved_boundary")
        self.assertEqual(interval.action, "zero_fill")
        self.assertNotIn("extra_source", interval.evidence)

    def test_unresolved_boundary_can_use_a_refined_sample_guard(self) -> None:
        boundaries = unresolved_boundaries_from_offset_clusters(
            [_pair(2, (_step(317.0),)), _pair(3, ())],
            device_count=3,
            master_index=0,
            fs=100.0,
            window_seconds=0.5,
            fallback_step_seconds=0.25,
            event_time_tolerance_seconds=0.25,
            gap_level_tolerance_samples=12.0,
            boundary_guard_samples=10,
        )

        self.assertEqual(len(boundaries), 1)
        self.assertEqual(
            (
                boundaries[0].canonical_start_sample,
                boundaries[0].canonical_end_sample,
            ),
            (90, 111),
        )
        self.assertIn("10-sample", boundaries[0].evidence)

    def test_confirmed_negative_slave_loss_is_not_an_unresolved_boundary(self) -> None:
        boundaries = unresolved_boundaries_from_offset_clusters(
            [_pair(2, (_step(-317.0),)), _pair(3, ())],
            device_count=3,
            master_index=0,
            fs=100.0,
            window_seconds=0.5,
            fallback_step_seconds=0.25,
            event_time_tolerance_seconds=0.25,
            gap_level_tolerance_samples=12.0,
        )
        self.assertEqual(boundaries, ())

    def test_terminal_support_becomes_device_local_tail(self) -> None:
        pair = _pair(3, ())
        pair.terminal_crop_master_sample = 600
        pair.terminal_crop_reason = "unconfirmed terminal step"
        support = terminal_support_from_pair(pair)
        self.assertIsNotNone(support)
        assert support is not None
        interval = terminal_support_to_interval(support, canonical_end_sample=700)
        self.assertIsNotNone(interval)
        assert interval is not None
        self.assertEqual(interval.affected_device_indices, (3,))
        self.assertEqual((interval.canonical_start_sample, interval.canonical_end_sample), (600, 700))
        self.assertEqual(interval.kind, "terminal_unsupported")


if __name__ == "__main__":
    unittest.main()
