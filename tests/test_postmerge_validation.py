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

from wild_preprocess.models import ClassifiedInterval, Recording
from wild_preprocess.sync.merge import apply_staged_zero_fill
from wild_preprocess.sync.observe import LagEstimate
from wild_preprocess.sync.postmerge import postmerge_exclusion_intervals, validate_staged_merge


def _recording(index: int, *, fs: int, n_channels: int) -> Recording:
    folder = Path(f"device-{index}")
    return Recording(
        folder=folder,
        amplifier_file=folder / "amplifier.dat",
        analog_file=folder / "analogin.dat",
        ce_params_file=folder / "CE_params.bin",
        device_name=folder.name,
        recording_name="recording",
        fs=fs,
        n_channels=n_channels,
        n_samples=1,
        analog_channels=1,
        analog_samples=1,
    )


def _write_merged(path: Path, *, residual_lag: int = 0) -> list[Recording]:
    fs = 1_000
    n_samples = 10_000
    n_channels = 4
    rng = np.random.default_rng(72)
    common = rng.normal(scale=500.0, size=n_samples)
    common += 300.0 * np.sin(2 * np.pi * 183 * np.arange(n_samples) / fs)
    slave = np.zeros_like(common)
    if residual_lag >= 0:
        slave[residual_lag:] = common[: n_samples - residual_lag]
    else:
        slave[:residual_lag] = common[-residual_lag:]
    master_block = np.column_stack([common + channel * 17 for channel in range(n_channels)])
    slave_block = np.column_stack([slave + channel * 17 for channel in range(n_channels)])
    np.rint(np.column_stack([master_block, slave_block])).astype("<i2").tofile(path)
    return [_recording(1, fs=fs, n_channels=n_channels), _recording(2, fs=fs, n_channels=n_channels)]


def _estimate(*, lag: int = 0, correlation: float = 0.8) -> LagEstimate:
    return LagEstimate(
        lag_samples=lag,
        peak_correlation=correlation,
        peak_to_background=10.0,
        peak_margin_fraction=0.5,
        secondary_lag_samples=None,
        lags=np.array([lag]),
        correlations=np.array([correlation]),
    )


class PostMergeValidationTest(unittest.TestCase):
    def test_only_mapping_changing_intervals_create_required_join_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "amplifier.dat"
            recordings = _write_merged(path)
            duplication = ClassifiedInterval(
                (2,), 4_000, 4_100, "duplicate_destination", "zero_fill", "medium", 3_900, 4_000
            )
            duplication_result = validate_staged_merge(
                path,
                recordings,
                master_index=0,
                window_seconds=0.25,
                classified_intervals=[duplication],
            )
            self.assertEqual(len(duplication_result.measurements), 5)

            boundary = ClassifiedInterval(
                (1, 2), 4_000, 4_100, "unresolved_boundary", "zero_fill", "unresolved"
            )
            boundary_result = validate_staged_merge(
                path,
                recordings,
                master_index=0,
                window_seconds=0.25,
                classified_intervals=[boundary],
            )
            join_positions = [
                item.position
                for item in boundary_result.measurements
                if item.position.startswith("boundary")
            ]
            self.assertEqual(len(join_positions), 2)
            self.assertTrue(any(item.endswith("_before") for item in join_positions))
            self.assertTrue(any(item.endswith("_after") for item in join_positions))

    def test_aligned_staged_output_passes_all_five_positions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "amplifier.dat"
            recordings = _write_merged(path)
            result = validate_staged_merge(path, recordings, master_index=0, window_seconds=0.25)
            self.assertEqual(result.status, "OK")
            self.assertEqual(len(result.measurements), 5)
            self.assertTrue(all(measurement.passed for measurement in result.measurements))
            self.assertEqual(result.max_abs_lag_samples, 0.0)
            self.assertEqual(result.to_dict()["measurements"][0]["position"], "start")

    def test_residual_lag_above_hard_limit_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "amplifier.dat"
            recordings = _write_merged(path, residual_lag=5)
            result = validate_staged_merge(path, recordings, master_index=0, window_seconds=0.25)
            self.assertEqual(result.status, "FAIL")
            self.assertEqual(result.max_abs_lag_samples, 5.0)
            self.assertTrue(all(not measurement.passed for measurement in result.measurements))
            self.assertIn("reliable residual lag above", result.message)
            self.assertTrue(all("residual lag 5" in measurement.message for measurement in result.measurements))

    def test_one_low_confidence_checkpoint_is_reported_but_four_reliable_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "amplifier.dat"
            recordings = _write_merged(path)
            estimates = [_estimate(correlation=0.01)] + [_estimate() for _ in range(4)]
            with patch("wild_preprocess.sync.postmerge.estimate_lag", side_effect=estimates):
                result = validate_staged_merge(path, recordings, master_index=0, window_seconds=0.25)
            self.assertEqual(result.status, "OK")
            self.assertEqual(sum(measurement.passed for measurement in result.measurements), 4)
            self.assertIn("ignored start checkpoint", result.message)
            self.assertIn("peak correlation", result.message)

    def test_fewer_than_four_reliable_checkpoints_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "amplifier.dat"
            recordings = _write_merged(path)
            estimates = [_estimate(correlation=0.01) for _ in range(2)] + [_estimate() for _ in range(3)]
            with patch("wild_preprocess.sync.postmerge.estimate_lag", side_effect=estimates):
                result = validate_staged_merge(path, recordings, master_index=0, window_seconds=0.25)
            self.assertEqual(result.status, "FAIL")
            self.assertIn("only 3 of 5 reliable checkpoints", result.message)

    def test_reliable_lag_above_limit_fails_even_with_four_good_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "amplifier.dat"
            recordings = _write_merged(path)
            estimates = [_estimate(), _estimate(lag=5)] + [_estimate() for _ in range(3)]
            with patch("wild_preprocess.sync.postmerge.estimate_lag", side_effect=estimates):
                result = validate_staged_merge(path, recordings, master_index=0, window_seconds=0.25)
            self.assertEqual(result.status, "FAIL")
            self.assertIn("reliable residual lag above 4 samples", result.message)

    def test_boundary_failure_becomes_non_destructive_warn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "amplifier.dat"
            recordings = _write_merged(path)
            boundary = ClassifiedInterval(
                (1,), 4_000, 4_100, "missing", "zero_fill", "high"
            )
            estimates = [_estimate() for _ in range(5)] + [_estimate(lag=5), _estimate()]
            with patch("wild_preprocess.sync.postmerge.estimate_lag", side_effect=estimates):
                result = validate_staged_merge(
                    path,
                    recordings,
                    master_index=0,
                    window_seconds=0.25,
                    classified_intervals=[boundary],
                )
            self.assertEqual(result.status, "WARN")
            exclusions = postmerge_exclusion_intervals(
                result,
                canonical_start_sample=100,
                device_count=2,
            )
            self.assertFalse(exclusions)

    def test_missing_same_side_boundary_window_is_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "amplifier.dat"
            recordings = _write_merged(path)
            validity_path = Path(temporary) / "valid_samples.dat"
            validity = np.ones((10_000, 2), dtype=np.uint8)
            validity[:4_000] = 0
            validity.tofile(validity_path)
            boundary = ClassifiedInterval(
                (1, 2), 4_000, 4_100, "unresolved_boundary", "zero_fill", "unresolved"
            )
            result = validate_staged_merge(
                path,
                recordings,
                master_index=0,
                window_seconds=0.25,
                validity_path=validity_path,
                classified_intervals=[boundary],
            )
            self.assertEqual(result.status, "FAIL")
            self.assertIn("no valid same-side boundary window", result.message)

    def test_single_endpoint_lag_becomes_non_destructive_warn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "amplifier.dat"
            recordings = _write_merged(path)
            estimates = [_estimate() for _ in range(4)] + [_estimate(lag=7)]
            with patch("wild_preprocess.sync.postmerge.estimate_lag", side_effect=estimates):
                result = validate_staged_merge(path, recordings, master_index=0, window_seconds=0.25)
            self.assertEqual(result.status, "WARN")
            exclusions = postmerge_exclusion_intervals(
                result,
                canonical_start_sample=40,
                device_count=2,
            )
            self.assertFalse(exclusions)

    def test_staged_zero_fill_respects_master_first_validity_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recordings = [_recording(index, fs=1_000, n_channels=2) for index in range(1, 4)]
            amplifier_path = root / "amplifier.dat"
            validity_path = root / "valid_samples.dat"
            np.ones((20, 6), dtype="<i2").tofile(amplifier_path)
            np.ones((20, 3), dtype=np.uint8).tofile(validity_path)
            interval = ClassifiedInterval(
                (1,), 105, 110, "postmerge_unverified", "zero_fill", "unresolved"
            )
            summary = apply_staged_zero_fill(
                amplifier_path,
                validity_path,
                recordings,
                master_index=1,
                canonical_start_sample=100,
                n_output_samples=20,
                intervals=[interval],
            )
            amplifier = np.fromfile(amplifier_path, dtype="<i2").reshape(20, 6)
            validity = np.fromfile(validity_path, dtype=np.uint8).reshape(20, 3)
            self.assertTrue(np.all(amplifier[5:10, :2] == 0))
            self.assertTrue(np.all(amplifier[5:10, 2:] == 1))
            self.assertTrue(np.all(validity[5:10, 1] == 0))
            self.assertTrue(np.all(validity[5:10, (0, 2)] == 1))
            self.assertEqual(summary["valid_samples_by_channel"], [20, 15, 20])

    def test_missing_staged_output_returns_safe_failure_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "not-written.dat"
            recordings = [_recording(1, fs=1_000, n_channels=4), _recording(2, fs=1_000, n_channels=4)]
            result = validate_staged_merge(path, recordings, master_index=0)
            self.assertEqual(result.status, "FAIL")
            self.assertFalse(result.measurements)
            self.assertIn("does not exist", result.message)


if __name__ == "__main__":
    unittest.main()
