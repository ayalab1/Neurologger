from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from wild_preprocess.models import (
    DeviceSyncAnchor,
    RelativeOffsetStep,
    SyncObservation,
    SyncOptions,
    map_verified_device_sample,
)
from wild_preprocess.sync.gaps import (
    AdaptiveChangePoint,
    detect_adaptive_change_points,
    detect_relative_offset_steps,
    localize_adaptive_change_point,
    localize_relative_offset_step,
)
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
    def _point(self, *, boundary: int, delta: float) -> AdaptiveChangePoint:
        return AdaptiveChangePoint(
            canonical_boundary_sample=boundary,
            time_sec=boundary / 100.0,
            delta_samples=delta,
            before_level_samples=0.0,
            after_level_samples=delta,
            uncertainty_samples=1.0,
            before_slope_samples_per_second=0.0,
            after_slope_samples_per_second=0.0,
            confidence="high",
            before_count=4,
            after_count=4,
            evidence="synthetic adaptive transition",
        )

    def test_full_rate_localization_recovers_missing_and_extra_source_boundaries(self) -> None:
        fs = 100.0
        physical_boundary = 1_500
        rng = np.random.default_rng(341)
        master = rng.normal(size=4_000).astype(np.float32)
        options = SyncOptions(
            window_seconds=2.0,
            step_seconds=1.0,
            endpoint_probe_seconds=1.0,
            min_peak_correlation=0.05,
            min_peak_margin_fraction=0.01,
        )
        for magnitude in (1, 4, 317):
            cases = (
                (
                    -float(magnitude),
                    np.concatenate(
                        (master[:physical_boundary], master[physical_boundary + magnitude :])
                    ),
                ),
                (
                    float(magnitude),
                    np.concatenate(
                        (
                            master[:physical_boundary],
                            rng.normal(size=magnitude),
                            master[physical_boundary:],
                        )
                    ),
                ),
            )
            for delta, slave in cases:
                with self.subTest(delta=delta):
                    localized = localize_adaptive_change_point(
                        master,
                        np.asarray(slave, dtype=np.float32),
                        self._point(boundary=1_450, delta=delta),
                        fs=fs,
                        options=options,
                    )
                    self.assertEqual(localized.localization_status, "localized")
                    self.assertTrue(localized.sample_localized)
                    self.assertLessEqual(abs(localized.canonical_boundary_sample - physical_boundary), 1)
                    self.assertIn("two-window old/new preferences", localized.localization_evidence)

    def test_first_candidate_optimum_is_rejected_as_search_edge(self) -> None:
        fs = 10.0
        approximate = 20
        physical_boundary = 12
        rng = np.random.default_rng(7_712)
        master = rng.normal(size=50).astype(np.float32)
        slave = np.concatenate(
            (master[:physical_boundary], np.asarray([3.0], dtype=np.float32), master[physical_boundary:])
        )
        step = RelativeOffsetStep(
            master_sample=approximate,
            time_sec=approximate / fs,
            offset_step_samples=1.0,
            missing_samples=1,
            offset_before_samples=0.0,
            offset_after_samples=1.0,
            confidence="high",
            evidence="synthetic first-candidate transition",
        )
        refined = localize_relative_offset_step(
            master,
            slave,
            step,
            fs=fs,
            options=SyncOptions(
                window_seconds=0.5,
                step_seconds=0.5,
                unresolved_boundary_guard_samples=1,
            ),
        )
        self.assertEqual(refined.master_sample, approximate)
        self.assertIn("optimum was at the search edge", refined.evidence)

    def test_unchanged_full_rate_mapping_rejects_adaptive_plateau_as_alias(self) -> None:
        fs = 100.0
        rng = np.random.default_rng(812)
        signal = rng.normal(size=2_400).astype(np.float32)
        point = self._point(boundary=1_000, delta=4.0)
        result = localize_adaptive_change_point(
            signal,
            signal.copy(),
            point,
            fs=fs,
            options=SyncOptions(
                window_seconds=2.0,
                step_seconds=1.0,
                endpoint_probe_seconds=1.0,
            ),
        )
        self.assertEqual(result.localization_status, "verified_alias")
        self.assertFalse(result.sample_localized)

    def test_periodic_full_rate_evidence_remains_unresolved(self) -> None:
        fs = 100.0
        samples = np.arange(2_400, dtype=np.float64)
        signal = np.sin(2.0 * np.pi * samples / 4.0).astype(np.float32)
        result = localize_adaptive_change_point(
            signal,
            signal.copy(),
            self._point(boundary=1_000, delta=4.0),
            fs=fs,
            options=SyncOptions(
                window_seconds=2.0,
                step_seconds=1.0,
                endpoint_probe_seconds=1.0,
            ),
        )
        self.assertEqual(result.localization_status, "unresolved")
        self.assertFalse(result.sample_localized)

    def test_flat_boundary_plateau_wider_than_guard_remains_unresolved(self) -> None:
        fs = 1_000.0
        physical_boundary = 4_000
        rng = np.random.default_rng(981)
        master = rng.normal(size=8_000).astype(np.float32)
        master[3_700:4_300] = 0.0
        slave = np.concatenate(
            (master[:physical_boundary], np.zeros(4, dtype=np.float32), master[physical_boundary:])
        )
        result = localize_adaptive_change_point(
            master,
            slave,
            self._point(boundary=physical_boundary, delta=4.0),
            fs=fs,
            options=SyncOptions(
                window_seconds=2.0,
                step_seconds=1.0,
                endpoint_probe_seconds=1.0,
                unresolved_boundary_guard_samples=100,
            ),
        )
        self.assertEqual(result.localization_status, "unresolved")
        self.assertFalse(result.sample_localized)
        self.assertIn("boundary refinement rejected", result.localization_evidence)

    def test_large_positive_step_does_not_expand_localization_uncertainty(self) -> None:
        fs = 1_000.0
        physical_boundary = 4_000
        rng = np.random.default_rng(1_227)
        master = rng.normal(size=9_000).astype(np.float32)
        master[3_850:4_150] = 0.0
        slave = np.concatenate(
            (
                master[:physical_boundary],
                np.zeros(1_000, dtype=np.float32),
                master[physical_boundary:],
            )
        )
        result = localize_adaptive_change_point(
            master,
            slave,
            self._point(boundary=physical_boundary, delta=1_000.0),
            fs=fs,
            options=SyncOptions(
                window_seconds=2.0,
                step_seconds=1.0,
                endpoint_probe_seconds=1.0,
                unresolved_boundary_guard_samples=100,
            ),
        )
        self.assertEqual(result.localization_status, "unresolved")
        self.assertFalse(result.sample_localized)

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
