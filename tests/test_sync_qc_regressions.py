from __future__ import annotations

import sys
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from wild_preprocess.binary_io import recordings_from_folders
from wild_preprocess.models import Recording, RelativeOffsetStep, SyncObservation, SyncOptions
from wild_preprocess.sync.infer import fit_affine_sync_model
from wild_preprocess.sync.observe import LagEstimate, _tracking_rejection_reasons, estimate_lag, observe_pair
from wild_preprocess.sync.validate import validate_pair


def _initial(*, peak: float = 1.0, peak_margin: float = 0.5) -> LagEstimate:
    return LagEstimate(
        lag_samples=0,
        peak_correlation=peak,
        peak_to_background=20.0,
        peak_margin_fraction=peak_margin,
        secondary_lag_samples=None,
        lags=np.asarray([0]),
        correlations=np.asarray([peak]),
    )


def _observations(offsets: list[float], *, step_seconds: float = 5.0) -> list[SyncObservation]:
    return [
        SyncObservation(
            center_time_sec=index * step_seconds,
            predicted_offset_samples=offset,
            observed_offset_samples=offset,
            residual_lag_samples=0.0,
            peak_correlation=1.0,
            peak_to_background=20.0,
            peak_margin_fraction=0.5,
            secondary_lag_samples=None,
            accepted=True,
        )
        for index, offset in enumerate(offsets)
    ]


def _write_duration_recording(folder: Path, *, ephys_seconds: int, analog_samples: int) -> Path:
    folder.mkdir(parents=True)
    header = bytearray(512)
    struct.pack_into("<I", header, 0, 1_000)
    struct.pack_into("<I", header, 8, 4)
    (folder / "CE_params.bin").write_bytes(header)
    with (folder / "amplifier.dat").open("wb") as stream:
        stream.truncate(ephys_seconds * 1_000 * 4 * np.dtype("<i2").itemsize)
    with (folder / "analogin.dat").open("wb") as stream:
        stream.truncate(analog_samples * np.dtype("<i2").itemsize)
    return folder


class SyncQcRegressionTest(unittest.TestCase):
    def test_analog_duration_difference_has_fixed_two_second_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid_a = _write_duration_recording(
                root / "valid_a" / "recording",
                ephys_seconds=300,
                analog_samples=300 * 1_250,
            )
            valid_b = _write_duration_recording(
                root / "valid_b" / "recording",
                ephys_seconds=300,
                analog_samples=298 * 1_250,
            )
            recordings_from_folders([valid_a, valid_b])
            invalid = _write_duration_recording(
                root / "invalid" / "recording",
                ephys_seconds=300,
                analog_samples=372_499,
            )
            with self.assertRaisesRegex(ValueError, "analogin.dat duration"):
                recordings_from_folders([valid_a, invalid])

    def test_independent_noise_is_rejected_by_absolute_correlation_gate(self) -> None:
        rng = np.random.default_rng(71)
        estimate = estimate_lag(
            rng.normal(size=100_000),
            rng.normal(size=100_000),
            max_lag_samples=0,
            peak_exclusion_samples=24,
        )
        self.assertLess(estimate.peak_correlation, SyncOptions().min_peak_correlation)
        self.assertIn("low normalized correlation", _tracking_rejection_reasons(estimate, SyncOptions()))

    def test_default_absolute_normalized_correlation_gate(self) -> None:
        options = SyncOptions()
        observations = _observations([12.0, 12.1, 11.9])
        model = fit_affine_sync_model(observations, fs=20_000, options=options)
        status, message = validate_pair(_initial(peak=0.04), observations, model, options)
        self.assertEqual(status, "FAIL")
        self.assertIn("normalized correlation", message)

    def test_ambiguous_initial_competing_peak_is_a_hard_failure(self) -> None:
        options = SyncOptions()
        observations = _observations([12.0, 12.1, 11.9])
        model = fit_affine_sync_model(observations, fs=20_000, options=options)
        status, message = validate_pair(
            _initial(peak_margin=options.min_peak_margin_fraction * 0.5),
            observations,
            model,
            options,
        )
        self.assertEqual(status, "FAIL")
        self.assertIn("initial peak margin", message)

    def test_short_recording_uses_constant_offset_model(self) -> None:
        options = SyncOptions()
        observations = _observations([12.0, 12.2, 11.9])
        model = fit_affine_sync_model(observations, fs=20_000, options=options)
        status, _ = validate_pair(_initial(), observations, model, options)
        self.assertTrue(model.is_constant_offset)
        self.assertEqual(model.slope_samples_per_second, 0.0)
        self.assertEqual(status, "OK")

    def test_long_recording_requires_minimum_observation_count(self) -> None:
        options = SyncOptions()
        observations = _observations([12.0] * 8, step_seconds=10.0)
        model = fit_affine_sync_model(observations, fs=20_000, options=options)
        status, message = validate_pair(_initial(), observations, model, options)
        self.assertEqual(status, "FAIL")
        self.assertIn("accepted observations", message)

    def test_persistent_eight_sample_step_fails(self) -> None:
        options = SyncOptions()
        observations = _observations([0.0] * 10 + [8.0] * 10)
        model = fit_affine_sync_model(observations, fs=20_000, options=options)
        status, message = validate_pair(_initial(), observations, model, options)
        self.assertEqual(status, "FAIL")
        self.assertIn("persistent offset level shift 8.0", message)

    def test_small_step_after_modeled_large_gap_still_fails(self) -> None:
        options = SyncOptions()
        observations = _observations([0.0] * 10 + [100.0] * 10 + [108.0] * 10)
        modeled_gap = RelativeOffsetStep(
            master_sample=950_000,
            time_sec=47.5,
            offset_step_samples=100.0,
            missing_samples=100,
            offset_before_samples=0.0,
            offset_after_samples=100.0,
            confidence="high",
            evidence="synthetic",
        )
        model = fit_affine_sync_model(
            observations,
            fs=20_000,
            options=options,
            offset_steps=(modeled_gap,),
        )
        status, message = validate_pair(_initial(), observations, model, options)
        self.assertEqual(status, "FAIL")
        self.assertIn("persistent offset level shift 8.0", message)

    def test_off_center_persistent_eight_sample_step_fails(self) -> None:
        options = SyncOptions()
        observations = _observations([0.0] * 6 + [8.0] * 14)
        model = fit_affine_sync_model(observations, fs=20_000, options=options)
        status, message = validate_pair(_initial(), observations, model, options)
        self.assertEqual(status, "FAIL")
        self.assertIn("persistent offset level shift 8.0", message)

    def test_late_exact_eight_sample_step_meets_inclusive_threshold(self) -> None:
        options = SyncOptions()
        observations = _observations([0.0] * 14 + [8.0] * 6)
        model = fit_affine_sync_model(observations, fs=20_000, options=options)
        status, message = validate_pair(_initial(), observations, model, options)
        self.assertEqual(status, "FAIL")
        self.assertIn("persistent offset level shift 8.0", message)

    def test_persistent_four_sample_step_is_reported_but_nonblocking(self) -> None:
        options = SyncOptions()
        observations = _observations([0.0] * 10 + [4.0] * 10)
        model = fit_affine_sync_model(observations, fs=20_000, options=options)
        status, message = validate_pair(_initial(), observations, model, options)
        self.assertEqual(status, "OK")
        self.assertIn("nonblocking persistent offset level shift 4.0", message)

    def test_late_exact_four_sample_step_remains_nonblocking(self) -> None:
        options = SyncOptions()
        observations = _observations([0.0] * 14 + [4.0] * 6)
        model = fit_affine_sync_model(observations, fs=20_000, options=options)
        status, message = validate_pair(_initial(), observations, model, options)
        self.assertEqual(status, "OK")
        self.assertIn("nonblocking persistent offset level shift 4.0", message)

    def test_clean_affine_200_ppm_does_not_look_like_a_discontinuity(self) -> None:
        options = SyncOptions()
        fs = 20_000
        slope = 200e-6 * fs
        observations = _observations([12.0 + slope * index * 5.0 for index in range(20)])
        model = fit_affine_sync_model(observations, fs=fs, options=options)
        status, message = validate_pair(_initial(), observations, model, options)
        self.assertAlmostEqual(model.drift_ppm, 200.0, places=6)
        self.assertEqual(status, "OK")
        self.assertNotIn("offset level shift", message)
        self.assertNotIn("offset step", message)

    def test_clean_affine_high_drift_is_warning_not_discontinuity(self) -> None:
        options = SyncOptions()
        fs = 20_000
        slope = 600e-6 * fs
        observations = _observations([12.0 + slope * index * 5.0 for index in range(20)])
        model = fit_affine_sync_model(observations, fs=fs, options=options)
        status, message = validate_pair(_initial(), observations, model, options)
        self.assertAlmostEqual(model.drift_ppm, 600.0, places=6)
        self.assertEqual(status, "WARN")
        self.assertIn("drift 600.0 ppm", message)
        self.assertNotIn("clock discontinuity", message)
        self.assertNotIn("offset step", message)

    def test_exact_eight_sample_steps_on_high_affine_drift_fail(self) -> None:
        options = SyncOptions()
        fs = 20_000
        slope = 600e-6 * fs
        for boundary in (9, 11, 15):
            with self.subTest(boundary=boundary):
                offsets = [
                    12.0 + slope * index * 5.0 + (8.0 if index >= boundary else 0.0)
                    for index in range(20)
                ]
                observations = _observations(offsets)
                model = fit_affine_sync_model(observations, fs=fs, options=options)
                status, message = validate_pair(_initial(), observations, model, options)
                self.assertEqual(status, "FAIL")
                self.assertIn("persistent offset level shift 8.0", message)

    def test_meaningfully_subthreshold_step_remains_nonblocking(self) -> None:
        options = SyncOptions()
        observations = _observations([0.0] * 10 + [7.999] * 10)
        model = fit_affine_sync_model(observations, fs=20_000, options=options)
        status, message = validate_pair(_initial(), observations, model, options)
        self.assertEqual(status, "OK")
        self.assertNotIn("clock discontinuity", message)
        self.assertIn("nonblocking persistent offset level shift 8.0", message)

    def test_large_single_jump_still_fails_after_detrending(self) -> None:
        options = SyncOptions()
        observations = _observations([0.0] * 10 + [80.0] + [0.0] * 10)
        model = fit_affine_sync_model(observations, fs=20_000, options=options)
        status, message = validate_pair(_initial(), observations, model, options)
        self.assertEqual(status, "FAIL")
        self.assertIn("detrended offset step", message)

    def test_shorter_slave_endpoint_is_excluded_from_observation_denominator(self) -> None:
        fs = 100
        rng = np.random.default_rng(101)
        master_feature = rng.normal(size=1_000).astype("<f4")
        slave_feature = master_feature[:900].copy()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            master_path = root / "master.f32"
            slave_path = root / "slave.f32"
            master_feature.tofile(master_path)
            slave_feature.tofile(slave_path)
            master = Recording(
                folder=root / "master",
                amplifier_file=root / "unused_master.dat",
                analog_file=root / "unused_master_analog.dat",
                ce_params_file=root / "unused_master_ce.bin",
                device_name="master",
                recording_name="recording",
                fs=fs,
                n_channels=4,
                n_samples=master_feature.size,
                analog_channels=1,
                analog_samples=100,
            )
            slave = Recording(
                folder=root / "slave",
                amplifier_file=root / "unused_slave.dat",
                analog_file=root / "unused_slave_analog.dat",
                ce_params_file=root / "unused_slave_ce.bin",
                device_name="slave",
                recording_name="recording",
                fs=fs,
                n_channels=4,
                n_samples=slave_feature.size,
                analog_channels=1,
                analog_samples=90,
            )
            options = SyncOptions(
                initial_start_seconds=0.0,
                initial_duration_seconds=2.0,
                initial_max_lag_seconds=0.2,
                window_seconds=1.0,
                step_seconds=0.5,
                tracking_max_lag_samples=10,
            )
            result = observe_pair(master, slave, master_path, slave_path, options)
        self.assertEqual(len(result.observations), 17)
        self.assertTrue(all(observation.accepted for observation in result.observations))
        self.assertTrue(all("outside slave recording" not in observation.rejection_reason for observation in result.observations))


if __name__ == "__main__":
    unittest.main()
