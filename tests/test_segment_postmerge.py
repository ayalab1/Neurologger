from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from wild_preprocess.models import DeviceSyncAnchor, DeviceSyncSegment, Recording
from wild_preprocess.sync.observe import LagEstimate
from wild_preprocess.sync.postmerge import (
    PostMergeMeasurement,
    PostMergeValidationResult,
    apply_postmerge_segment_corrections,
    infer_postmerge_segment_corrections,
    postmerge_alignment_warning_intervals,
    postmerge_exclusion_intervals,
    validate_segment_staged_merge,
)


def _recording(index: int, *, n_samples: int = 10_000) -> Recording:
    folder = Path(f"device-{index}")
    return Recording(
        folder=folder,
        amplifier_file=folder / "amplifier.dat",
        analog_file=folder / "analogin.dat",
        ce_params_file=folder / "CE_params.bin",
        device_name=folder.name,
        recording_name="recording",
        fs=1_000,
        n_channels=2,
        n_samples=n_samples,
        analog_channels=1,
        analog_samples=n_samples,
    )


def _segment(device_index: int, start: int, end: int) -> DeviceSyncSegment:
    return DeviceSyncSegment(
        device_index=device_index,
        canonical_start_sample=start,
        canonical_end_sample=end,
        source_start_sample=start,
        source_end_sample=end,
        source_scale=1.0,
        source_intercept_samples=0.0,
        anchors=(
            DeviceSyncAnchor(start, float(start), True, "high", "synthetic start"),
            DeviceSyncAnchor(end - 1, float(end - 1), True, "high", "synthetic end"),
        ),
        confidence="high",
        publishable=True,
        evidence="synthetic",
    )


def _staged(path: Path, *, n_samples: int = 10_000) -> list[Recording]:
    rng = np.random.default_rng(22)
    common = rng.normal(scale=600, size=n_samples)
    master = np.column_stack((common, common + 13))
    slave = np.column_stack((common, common + 17))
    np.rint(np.column_stack((master, slave))).astype("<i2").tofile(path)
    return [_recording(1, n_samples=n_samples), _recording(2, n_samples=n_samples)]


class SegmentPostMergeTest(unittest.TestCase):
    def test_output_clipped_segment_keeps_authoritative_identity_for_correction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            amplifier = root / "amplifier.dat"
            validity_path = root / "valid_samples.dat"
            recordings = _staged(amplifier)
            np.ones((10_000, 2), dtype=np.uint8).tofile(validity_path)
            master_segment = _segment(1, 0, 10_100)
            slave_segment = _segment(2, 0, 10_100)
            reliable_lag = LagEstimate(
                5,
                0.8,
                10.0,
                0.5,
                None,
                np.array([5]),
                np.array([0.8]),
            )
            with patch(
                "wild_preprocess.sync.postmerge.estimate_lag",
                return_value=reliable_lag,
            ):
                result = validate_segment_staged_merge(
                    amplifier,
                    recordings,
                    0,
                    device_segments=[master_segment, slave_segment],
                    validity_path=validity_path,
                    canonical_start_sample=100,
                    n_output_samples=10_000,
                    dense_step_seconds=2.0,
                )

            corrections = infer_postmerge_segment_corrections(
                result,
                canonical_start_sample=100,
            )
            corrected, applied, rejected = apply_postmerge_segment_corrections(
                [master_segment, slave_segment], corrections
            )

        self.assertEqual(len(corrections), 1)
        self.assertEqual(
            (
                corrections[0].canonical_start_sample,
                corrections[0].canonical_end_sample,
            ),
            (0, 10_100),
        )
        self.assertEqual(len(applied), 1)
        self.assertFalse(rejected)
        corrected_slave = next(item for item in corrected if item.device_index == 2)
        self.assertEqual(corrected_slave.source_intercept_samples, 5.0)

    def test_correction_evidence_floor_cannot_be_lowered_below_three(self) -> None:
        result = PostMergeValidationResult(
            status="WARN",
            message="synthetic",
            amplifier_path="amplifier.dat",
            master_device_index=1,
            n_output_samples=1_000,
            n_output_channels=4,
            window_samples=100,
            max_allowed_abs_lag_samples=4,
            min_peak_correlation=0.05,
            max_abs_lag_samples=5.0,
            measurements=(),
        )

        with self.assertRaisesRegex(ValueError, "at least three"):
            infer_postmerge_segment_corrections(
                result,
                canonical_start_sample=0,
                minimum_supporting_measurements=2,
            )

    def test_unavailable_local_window_warns_without_any_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            amplifier = root / "amplifier.dat"
            validity_path = root / "valid_samples.dat"
            recordings = _staged(amplifier)
            merged = np.memmap(amplifier, dtype="<i2", mode="r+", shape=(10_000, 4))
            merged[3_333:6_666, 2:] = 0
            merged.flush()
            del merged
            np.ones((10_000, 2), dtype=np.uint8).tofile(validity_path)
            segments = [
                _segment(1, 0, 10_000),
                _segment(2, 0, 3_333),
                _segment(2, 3_333, 6_666),
                _segment(2, 6_666, 10_000),
            ]

            result = validate_segment_staged_merge(
                amplifier, recordings, 0, device_segments=segments,
                validity_path=validity_path, window_seconds=0.25,
            )
            self.assertEqual(result.status, "WARN")
            exclusions = postmerge_exclusion_intervals(
                result, canonical_start_sample=0, device_count=2,
            )
            self.assertFalse(exclusions)
            self.assertTrue(any(
                item.message.startswith("lag measurement unavailable")
                for item in result.measurements
            ))

    def test_unavailable_terminal_window_warns_without_invalid_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            amplifier = root / "amplifier.dat"
            validity_path = root / "valid_samples.dat"
            recordings = _staged(amplifier)
            merged = np.memmap(amplifier, dtype="<i2", mode="r+", shape=(10_000, 4))
            merged[5_000:, 2:] = 0
            merged.flush()
            del merged
            np.ones((10_000, 2), dtype=np.uint8).tofile(validity_path)
            segments = [_segment(1, 0, 10_000), _segment(2, 0, 5_000), _segment(2, 5_000, 10_000)]

            result = validate_segment_staged_merge(
                amplifier, recordings, 0, device_segments=segments,
                validity_path=validity_path, window_seconds=0.25,
            )
            self.assertEqual(result.status, "WARN")
            exclusions = postmerge_exclusion_intervals(
                result, canonical_start_sample=0, device_count=2,
            )
            self.assertFalse(exclusions)

    def test_dense_short_invalid_fragments_select_short_islands_without_whole_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            amplifier = root / "amplifier.dat"
            validity_path = root / "valid_samples.dat"
            recordings = _staged(amplifier)
            validity = np.ones((10_000, 2), dtype=np.uint8)
            # Each 3-sample explicit invalid fragment is less than one second
            # from the next.  The ~297-sample valid islands still safely cover
            # the 100-sample narrow lag support.
            invalid_rows = np.concatenate(
                [np.arange(start, start + 3) for start in range(300, 10_000, 300)]
            )
            validity[invalid_rows, 1] = 0
            merged = np.memmap(amplifier, dtype="<i2", mode="r+", shape=(10_000, 4))
            merged[invalid_rows, 2:] = 0
            merged.flush()
            del merged
            validity.tofile(validity_path)

            result = validate_segment_staged_merge(
                amplifier,
                recordings,
                0,
                device_segments=[_segment(1, 0, 10_000), _segment(2, 0, 10_000)],
                validity_path=validity_path,
            )
            self.assertEqual(result.status, "OK")
            self.assertGreater(sum(item.window_end_sample - item.window_start_sample for item in result.measurements), 0)
            self.assertFalse(postmerge_exclusion_intervals(result, canonical_start_sample=0, device_count=2))

    def test_insufficient_window_for_requested_lag_is_unavailable_not_clipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            amplifier = root / "amplifier.dat"
            validity_path = root / "valid_samples.dat"
            recordings = _staged(amplifier)
            np.ones((10_000, 2), dtype=np.uint8).tofile(validity_path)
            with patch("wild_preprocess.sync.postmerge.estimate_lag") as estimator:
                result = validate_segment_staged_merge(
                    amplifier,
                    recordings,
                    0,
                    device_segments=[_segment(1, 0, 10_000), _segment(2, 0, 10_000)],
                    validity_path=validity_path,
                    window_seconds=0.01,
                    max_lag_samples=600,
                )
            estimator.assert_not_called()
            self.assertEqual(result.status, "WARN")
            self.assertTrue(all(item.lag_samples is None for item in result.measurements))
            self.assertFalse(
                postmerge_exclusion_intervals(
                    result, canonical_start_sample=0, device_count=2
                )
            )

    def test_structural_only_revalidation_does_not_move_to_another_qc_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            amplifier = root / "amplifier.dat"
            validity_path = root / "valid_samples.dat"
            recordings = _staged(amplifier)
            validity = np.ones((10_000, 2), dtype=np.uint8)
            validity[2_000:3_000, 1] = 0
            merged = np.memmap(amplifier, dtype="<i2", mode="r+", shape=(10_000, 4))
            merged[2_000:3_000, 2:] = 0
            merged.flush()
            del merged
            validity.tofile(validity_path)
            with patch("wild_preprocess.sync.postmerge.estimate_lag") as estimator:
                result = validate_segment_staged_merge(
                    amplifier,
                    recordings,
                    0,
                    device_segments=[_segment(1, 0, 10_000), _segment(2, 0, 10_000)],
                    validity_path=validity_path,
                    structural_only=True,
                )
            estimator.assert_not_called()
            self.assertEqual(result.status, "OK")
            self.assertEqual(result.measurements, ())

    def test_measured_local_failure_warns_without_erasing_its_segment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            amplifier = root / "amplifier.dat"
            validity_path = root / "valid_samples.dat"
            recordings = _staged(amplifier)
            np.ones((10_000, 2), dtype=np.uint8).tofile(validity_path)
            good = LagEstimate(0, 0.8, 10.0, 0.5, None, np.array([0]), np.array([0.8]))
            bad = LagEstimate(5, 0.8, 10.0, 0.5, None, np.array([5]), np.array([0.8]))
            # One segment means five global checks plus one interior check.
            with patch(
                "wild_preprocess.sync.postmerge.estimate_lag",
                side_effect=[good, bad, good, good, good, good],
            ):
                result = validate_segment_staged_merge(
                    amplifier,
                    recordings,
                    0,
                    device_segments=[_segment(1, 0, 10_000), _segment(2, 0, 10_000)],
                    validity_path=validity_path,
                )
            self.assertEqual(result.status, "WARN")
            failed = next(item for item in result.measurements if item.lag_samples == 5)
            exclusions = postmerge_exclusion_intervals(
                result, canonical_start_sample=0, device_count=2,
            )
            self.assertFalse(exclusions)
            warnings = postmerge_alignment_warning_intervals(
                result, canonical_start_sample=0
            )
            self.assertEqual(len(warnings), 1)
            self.assertEqual(warnings[0]["affected_device_indices"], [2])
            self.assertEqual(
                (
                    warnings[0]["canonical_start_sample"],
                    warnings[0]["canonical_end_sample"],
                ),
                (failed.window_start_sample, failed.window_end_sample),
            )

    def test_repeated_consistent_lag_corrects_intercept_without_extrapolation(self) -> None:
        measurements = tuple(
            PostMergeMeasurement(
                position=f"dense{index}",
                fraction=index / 4,
                nominal_output_sample=sample,
                window_start_sample=sample - 125,
                window_end_sample=sample + 125,
                slave_device_index=2,
                lag_samples=5,
                peak_correlation=0.8,
                peak_to_background=10.0,
                peak_margin_fraction=0.5,
                passed=False,
                message="residual lag 5 exceeds 4 samples",
                exclusion_device_indices=(2,),
                segment_canonical_start_sample=0,
                segment_canonical_end_sample=10_000,
            )
            for index, sample in enumerate((1_000, 5_000, 9_000), start=1)
        )
        result = PostMergeValidationResult(
            "WARN", "persistent residual lag", "stage/amplifier.dat", 1,
            10_000, 4, 250, 4, 0.05, 5.0, measurements,
        )
        corrections = infer_postmerge_segment_corrections(
            result,
            canonical_start_sample=0,
            minimum_supporting_measurements=3,
        )
        self.assertEqual(len(corrections), 1)
        corrected, applied, rejected = apply_postmerge_segment_corrections(
            [_segment(1, 0, 10_000), _segment(2, 0, 10_000)], corrections
        )
        self.assertEqual(len(applied), 1)
        self.assertFalse(rejected)
        slave = next(item for item in corrected if item.device_index == 2)
        self.assertEqual(slave.source_intercept_samples, 5.0)
        self.assertEqual(slave.canonical_end_sample, 9_995)
        self.assertEqual(slave.map_canonical_sample(1_000), 1_005.0)

    def test_isolated_reliable_lag_is_warning_not_correction(self) -> None:
        measurement = PostMergeMeasurement(
            "segment2_1_interior", 0.5, 5_000, 4_500, 5_500, 2,
            666, 0.2, 2.0, 0.1, False, "isolated lag", (2,),
            segment_canonical_start_sample=0,
            segment_canonical_end_sample=10_000,
        )
        result = PostMergeValidationResult(
            "WARN", "isolated", "stage/amplifier.dat", 1, 10_000, 4,
            1_000, 4, 0.05, 666.0, (measurement,),
        )
        self.assertFalse(
            infer_postmerge_segment_corrections(
                result, canonical_start_sample=0
            )
        )
        self.assertFalse(
            postmerge_exclusion_intervals(
                result, canonical_start_sample=0, device_count=2
            )
        )

    def test_ambiguous_peak_warns_without_invalidating_an_aligned_segment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            amplifier = root / "amplifier.dat"
            validity_path = root / "valid_samples.dat"
            recordings = _staged(amplifier)
            np.ones((10_000, 2), dtype=np.uint8).tofile(validity_path)
            good = LagEstimate(0, 0.8, 10.0, 0.5, None, np.array([0]), np.array([0.8]))
            ambiguous = LagEstimate(
                2, 0.75, 1.08, 0.04, None, np.array([2]), np.array([0.75])
            )
            with patch(
                "wild_preprocess.sync.postmerge.estimate_lag",
                side_effect=[ambiguous, good, good, good, good, good],
            ):
                result = validate_segment_staged_merge(
                    amplifier,
                    recordings,
                    0,
                    device_segments=[_segment(1, 0, 10_000), _segment(2, 0, 10_000)],
                    validity_path=validity_path,
                )
            self.assertEqual(result.status, "WARN")
            self.assertIn("peak/background", result.message)
            self.assertFalse(
                postmerge_exclusion_intervals(
                    result, canonical_start_sample=0, device_count=2
                )
            )

    def test_ambiguous_large_lag_warns_without_asserting_misalignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            amplifier = root / "amplifier.dat"
            validity_path = root / "valid_samples.dat"
            recordings = _staged(amplifier)
            np.ones((10_000, 2), dtype=np.uint8).tofile(validity_path)
            good = LagEstimate(0, 0.8, 10.0, 0.5, None, np.array([0]), np.array([0.8]))
            ambiguous = LagEstimate(
                50, 0.01, 1.01, 0.001, None, np.array([50]), np.array([0.01])
            )
            with patch(
                "wild_preprocess.sync.postmerge.estimate_lag",
                side_effect=[ambiguous, good, good, good, good, good],
            ):
                result = validate_segment_staged_merge(
                    amplifier,
                    recordings,
                    0,
                    device_segments=[_segment(1, 0, 10_000), _segment(2, 0, 10_000)],
                    validity_path=validity_path,
                )
            self.assertEqual(result.status, "WARN")
            self.assertFalse(
                postmerge_exclusion_intervals(
                    result, canonical_start_sample=0, device_count=2
                )
            )

    def test_fully_invalid_slave_is_recoverable_and_needs_no_extra_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            amplifier = root / "amplifier.dat"
            validity_path = root / "valid_samples.dat"
            recordings = _staged(amplifier)
            merged = np.memmap(amplifier, dtype="<i2", mode="r+", shape=(10_000, 4))
            merged[:, 2:] = 0
            merged.flush()
            del merged
            validity = np.ones((10_000, 2), dtype=np.uint8)
            validity[:, 1] = 0
            validity.tofile(validity_path)

            result = validate_segment_staged_merge(
                amplifier, recordings, 0, device_segments=[_segment(1, 0, 10_000)],
                validity_path=validity_path, window_seconds=0.25,
            )
            self.assertEqual(result.status, "WARN")
            self.assertIn("all-invalid", result.message)
            self.assertFalse(postmerge_exclusion_intervals(result, canonical_start_sample=0, device_count=2))

    def test_claimed_valid_sample_outside_segment_mapping_is_structural_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            amplifier = root / "amplifier.dat"
            validity_path = root / "valid_samples.dat"
            recordings = _staged(amplifier)
            np.ones((10_000, 2), dtype=np.uint8).tofile(validity_path)

            result = validate_segment_staged_merge(
                amplifier, recordings, 0,
                device_segments=[_segment(1, 0, 10_000), _segment(2, 0, 9_000)],
                validity_path=validity_path, window_seconds=0.25,
            )
            self.assertEqual(result.status, "FAIL")
            self.assertIn("without a publishable segment mapping", result.message)


if __name__ == "__main__":
    unittest.main()
