from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from wild_preprocess.models import DeviceSyncAnchor, SyncObservation, SyncOptions, map_verified_device_sample
from wild_preprocess.sync.gaps import detect_adaptive_change_points, detect_relative_offset_steps
from wild_preprocess.sync.infer import (
    anchors_from_accepted_observations,
    fit_independent_device_segments,
)


def _observation(time_sec: float, offset: float, *, correlation: float = 0.8) -> SyncObservation:
    observation = SyncObservation(
        center_time_sec=time_sec,
        predicted_offset_samples=offset,
        observed_offset_samples=offset,
        residual_lag_samples=0.0,
        peak_correlation=correlation,
        peak_to_background=3.0,
        peak_margin_fraction=0.2,
        secondary_lag_samples=None,
        accepted=True,
    )
    observation.model_inlier = True
    return observation


def _options() -> SyncOptions:
    # Keep the legacy 50-sample option deliberately present: adaptive
    # detection must not use it as a scientific detection threshold.
    return SyncOptions(
        step_seconds=1.0,
        gap_persistence_observations=2,
        gap_min_step_samples=50.0,
        min_peak_correlation=0.05,
        min_peak_to_background=1.2,
        min_peak_margin_fraction=0.01,
    )


class AdaptiveSegmentTests(unittest.TestCase):
    def test_detects_arbitrary_verified_steps_including_below_legacy_fifty(self) -> None:
        for step in (1, 5, 49, 50, 317):
            with self.subTest(step=step):
                observations = [
                    *[_observation(float(index), 0.0) for index in range(5)],
                    *[_observation(float(index), float(step)) for index in range(5, 11)],
                ]
                points = detect_adaptive_change_points(observations, 100.0, _options())
                self.assertEqual(len(points), 1)
                self.assertAlmostEqual(points[0].delta_samples, step, delta=1e-9)
                self.assertEqual(points[0].canonical_boundary_sample, 450)
                self.assertGreaterEqual(points[0].uncertainty_samples, 1.0)
                self.assertIn("adaptive locally detrended", points[0].evidence)
                # The old global-model projection intentionally remains capped
                # until pipeline migration; it must not be used as the new
                # segment detector.
                legacy_steps = detect_relative_offset_steps(observations, 100.0, _options())
                self.assertEqual(len(legacy_steps), int(step >= 50))

    def test_noisy_sub_gate_change_is_left_unresolved(self) -> None:
        # The plateau change is smaller than the minimum one-sample gate and
        # the locally detrended noise further raises the data-derived gate.
        before = (0.2, -0.2, 0.1, -0.1, 0.0)
        after = (0.5, 0.3, 0.6, 0.4, 0.5, 0.3)
        observations = [
            *[_observation(float(index), value) for index, value in enumerate(before)],
            *[_observation(float(index + len(before)), value) for index, value in enumerate(after)],
        ]
        self.assertEqual(detect_adaptive_change_points(observations, 100.0, _options()), ())

    def test_short_integer_lag_excursion_is_not_a_persistent_change(self) -> None:
        observations = [
            *[_observation(float(index), 30.0) for index in range(8)],
            *[_observation(float(index), 31.0) for index in range(8, 12)],
            *[_observation(float(index), 30.0) for index in range(12, 22)],
        ]
        self.assertEqual(detect_adaptive_change_points(observations, 100.0, _options()), ())

    def test_unqualified_observation_cannot_be_promoted_to_verified_anchor(self) -> None:
        observations = [
            _observation(1.0, 2.0),
            _observation(2.0, 2.0, correlation=0.01),
            _observation(3.0, 2.0),
        ]
        anchors = anchors_from_accepted_observations(observations, 100.0, _options())
        self.assertEqual([item.canonical_sample for item in anchors], [100, 300])
        self.assertTrue(all(item.is_publishable_evidence for item in anchors))

    def test_coarse_observation_requires_full_rate_confirmation_before_segment_use(self) -> None:
        coarse = _observation(2.0, 2.0)
        coarse.search_mode = "coarse_tracking"
        anchors = anchors_from_accepted_observations(
            [_observation(1.0, 2.0), coarse, _observation(3.0, 2.0)],
            100.0,
            _options(),
        )
        self.assertEqual([item.canonical_sample for item in anchors], [100, 300])

    def test_robust_model_outlier_cannot_become_a_segment_anchor(self) -> None:
        outlier = _observation(2.0, 200.0)
        outlier.model_inlier = False
        anchors = anchors_from_accepted_observations(
            [_observation(1.0, 2.0), outlier, _observation(3.0, 2.0)],
            100.0,
            _options(),
        )
        self.assertEqual([item.canonical_sample for item in anchors], [100, 300])

    def test_post_boundary_segment_is_fit_from_its_own_anchors_and_gap_is_absent(self) -> None:
        anchors = (
            DeviceSyncAnchor(100, 100.0, True, "high", "pre one"),
            DeviceSyncAnchor(400, 400.0, True, "high", "pre two"),
            DeviceSyncAnchor(600, 617.0, True, "high", "post one"),
            DeviceSyncAnchor(900, 917.0, True, "high", "post two"),
        )
        segments = fit_independent_device_segments(
            anchors,
            [500],
            device_index=2,
            canonical_start_sample=0,
            canonical_end_sample=1_000,
            source_sample_count=1_200,
            unresolved_ranges=[(480, 520)],
        )
        self.assertEqual([(item.canonical_start_sample, item.canonical_end_sample) for item in segments], [(0, 480), (520, 1000)])
        self.assertAlmostEqual(segments[0].source_intercept_samples, 0.0, delta=1e-9)
        self.assertAlmostEqual(segments[1].source_intercept_samples, 17.0, delta=1e-9)
        self.assertEqual(map_verified_device_sample(segments, 479, device_index=2), 479.0)
        self.assertIsNone(map_verified_device_sample(segments, 500, device_index=2))
        self.assertAlmostEqual(map_verified_device_sample(segments, 600, device_index=2), 617.0, delta=1e-9)
        self.assertTrue(all(item.publishable for item in segments))

    def test_underanchored_post_event_range_is_not_published(self) -> None:
        anchors = (
            DeviceSyncAnchor(100, 100.0, True, "high"),
            DeviceSyncAnchor(400, 400.0, True, "high"),
            DeviceSyncAnchor(600, 617.0, True, "high"),
        )
        segments = fit_independent_device_segments(
            anchors,
            [500],
            device_index=2,
            canonical_start_sample=0,
            canonical_end_sample=1_000,
            source_sample_count=1_200,
        )
        self.assertEqual(len(segments), 1)
        self.assertIsNone(map_verified_device_sample(segments, 600, device_index=2))

    def test_residual_metadata_uses_the_serialized_near_unity_mapping(self) -> None:
        scale = 1.0 + 5e-13
        anchors = tuple(
            DeviceSyncAnchor(
                canonical,
                scale * canonical + 12.0,
                True,
                "high",
            )
            for canonical in (50_000_000, 60_000_000, 70_000_000)
        )
        segments = fit_independent_device_segments(
            anchors,
            [],
            device_index=2,
            canonical_start_sample=49_000_000,
            canonical_end_sample=71_000_000,
            source_sample_count=80_000_000,
        )
        self.assertEqual(len(segments), 1)
        segment = segments[0]
        residuals = [
            abs(
                anchor.source_sample
                - (
                    segment.source_scale * anchor.canonical_sample
                    + segment.source_intercept_samples
                )
            )
            for anchor in anchors
        ]
        self.assertAlmostEqual(segment.residual_max_abs_samples, max(residuals), delta=1e-12)

    def test_source_reversing_reacquisition_is_omitted_not_published(self) -> None:
        anchors = (
            DeviceSyncAnchor(100, 1_100.0, True, "high"),
            DeviceSyncAnchor(400, 1_400.0, True, "high"),
            DeviceSyncAnchor(600, 900.0, True, "high"),
            DeviceSyncAnchor(900, 1_200.0, True, "high"),
        )
        segments = fit_independent_device_segments(
            anchors,
            [500],
            device_index=2,
            canonical_start_sample=0,
            canonical_end_sample=1_000,
            source_sample_count=2_000,
        )
        self.assertEqual(len(segments), 1)
        self.assertIsNone(map_verified_device_sample(segments, 600, device_index=2))


if __name__ == "__main__":
    unittest.main()
