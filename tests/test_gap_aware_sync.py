from __future__ import annotations

import json
import sys
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from wild_preprocess.models import (
    DeviceGap,
    Recording,
    RelativeOffsetStep,
    SyncModel,
    SyncObservation,
    SyncOptions,
    SyncPairResult,
)
from wild_preprocess.sync.gaps import (
    canonicalize_master_sample,
    detect_isolated_offset_crop,
    detect_unconfirmed_terminal_crop,
    infer_device_gaps,
    verify_isolated_offset_alias,
)
from wild_preprocess.sync.infer import fit_affine_sync_model
from wild_preprocess.sync.merge import _common_master_interval, _source_coordinates, _write_interleaved_stream
from wild_preprocess.sync.observe import estimate_lag_narrow_wide, observe_pair
from wild_preprocess.pipeline import (
    _unlocalized_adaptive_half_window_samples,
    run_multidevice_sync,
)
import wild_preprocess.sync.observe as observe_module


def _observation(time_sec: float, offset: float) -> SyncObservation:
    return SyncObservation(
        center_time_sec=time_sec,
        predicted_offset_samples=offset,
        observed_offset_samples=offset,
        residual_lag_samples=0.0,
        peak_correlation=0.8,
        peak_to_background=20.0,
        peak_margin_fraction=0.5,
        secondary_lag_samples=None,
        accepted=True,
    )


class AdaptiveBoundaryUncertaintyTest(unittest.TestCase):
    def test_unlocalized_boundary_covers_one_missing_observation(self) -> None:
        options = SyncOptions(
            step_seconds=5.0,
            unresolved_boundary_guard_samples=100,
        )

        self.assertEqual(
            _unlocalized_adaptive_half_window_samples(options, 20_000),
            100_000,
        )

    def test_sample_guard_can_dominate_a_fast_observation_cadence(self) -> None:
        options = SyncOptions(
            step_seconds=0.001,
            unresolved_boundary_guard_samples=100,
        )

        self.assertEqual(
            _unlocalized_adaptive_half_window_samples(options, 20_000),
            100,
        )


def _step(value: float, time_sec: float = 10.0) -> RelativeOffsetStep:
    return RelativeOffsetStep(
        master_sample=int(round(time_sec * 1_000)),
        time_sec=time_sec,
        offset_step_samples=value,
        missing_samples=int(round(abs(value))),
        offset_before_samples=0.0,
        offset_after_samples=value,
        confidence="high",
        evidence="synthetic",
    )


def _model(steps: tuple[RelativeOffsetStep, ...]) -> SyncModel:
    return SyncModel(
        intercept_samples=0.0,
        slope_samples_per_second=0.0,
        drift_ppm=0.0,
        residual_rms_samples=0.0,
        residual_max_abs_samples=0.0,
        accepted_count=20,
        observation_count=20,
        offset_steps=steps,
    )


def _pair(slave_index: int, steps: tuple[RelativeOffsetStep, ...]) -> SyncPairResult:
    return SyncPairResult(
        master_index=1,
        slave_index=slave_index,
        master_folder="master",
        slave_folder=f"slave{slave_index}",
        initial_offset_samples=0.0,
        initial_peak_to_background=20.0,
        initial_peak_margin_fraction=0.5,
        model=_model(steps),
        status="WARN" if steps else "OK",
    )


class GapAwareSyncTest(unittest.TestCase):
    def test_raw_recheck_rescues_an_isolated_false_lag(self) -> None:
        fs = 1_000
        rng = np.random.default_rng(901)
        feature = rng.normal(size=120_000).astype(np.float32)
        observations = [
            _observation(float(time_sec), 317.0 if time_sec == 55 else 0.0)
            for time_sec in range(5, 110, 5)
        ]
        options = SyncOptions(
            window_seconds=10.0,
            step_seconds=5.0,
            endpoint_probe_seconds=2.0,
            gap_persistence_observations=2,
            gap_min_step_samples=50.0,
        )
        model = fit_affine_sync_model(observations, fs, options=options)
        candidate = detect_isolated_offset_crop(observations, model, fs, options)

        self.assertIsNotNone(candidate)
        verified, evidence = verify_isolated_offset_alias(
            feature,
            feature.copy(),
            observations,
            model,
            candidate,
            fs,
            options,
        )

        self.assertTrue(verified, evidence)
        self.assertIn("excursion offset 0 (max run 0/2)", evidence)

    def test_raw_recheck_keeps_crop_when_excursion_mapping_has_support(self) -> None:
        fs = 1_000
        shift = 317
        rng = np.random.default_rng(902)
        master = rng.normal(size=120_000).astype(np.float32)
        slave = master.copy()
        supported_start = 53_000
        supported_end = 57_000
        slave[supported_start + shift : supported_end + shift] = master[
            supported_start:supported_end
        ]
        observations = [
            _observation(float(time_sec), float(shift) if time_sec == 55 else 0.0)
            for time_sec in range(5, 110, 5)
        ]
        options = SyncOptions(
            window_seconds=10.0,
            step_seconds=5.0,
            endpoint_probe_seconds=2.0,
            gap_persistence_observations=2,
            gap_min_step_samples=50.0,
        )
        model = fit_affine_sync_model(observations, fs, options=options)
        candidate = detect_isolated_offset_crop(observations, model, fs, options)

        self.assertIsNotNone(candidate)
        verified, evidence = verify_isolated_offset_alias(
            master,
            slave,
            observations,
            model,
            candidate,
            fs,
            options,
        )

        self.assertFalse(verified, evidence)
        self.assertNotIn("excursion offset 0", evidence)

    def test_one_supported_excursion_window_is_not_discarded_as_alias(self) -> None:
        fs = 1_000
        shift = 317
        rng = np.random.default_rng(903)
        master = rng.normal(size=120_000).astype(np.float32)
        slave = master.copy()
        supported_start = 53_000
        supported_end = 55_000
        slave[supported_start + shift : supported_end + shift] = master[
            supported_start:supported_end
        ]
        observations = [
            _observation(float(time_sec), float(shift) if time_sec == 55 else 0.0)
            for time_sec in range(5, 110, 5)
        ]
        options = SyncOptions(
            window_seconds=10.0,
            step_seconds=5.0,
            endpoint_probe_seconds=2.0,
            gap_persistence_observations=2,
            gap_min_step_samples=50.0,
        )
        model = fit_affine_sync_model(observations, fs, options=options)
        candidate = detect_isolated_offset_crop(observations, model, fs, options)

        self.assertIsNotNone(candidate)
        verified, evidence = verify_isolated_offset_alias(
            master,
            slave,
            observations,
            model,
            candidate,
            fs,
            options,
        )

        self.assertFalse(verified, evidence)
        self.assertIn("excursion offset 1", evidence)

    def test_raw_terminal_crop_is_mapped_after_prior_master_gaps(self) -> None:
        gaps = [
            DeviceGap(1, 1_000, 100, 100.0),
            DeviceGap(2, 1_500, 50, 50.0),
            DeviceGap(1, 2_100, 200, 200.0),
        ]

        self.assertEqual(
            canonicalize_master_sample(3_000, gaps, master_device_index=1),
            3_300,
        )
        self.assertEqual(
            canonicalize_master_sample(900, gaps, master_device_index=1),
            900,
        )

    def test_raw_master_sample_at_gap_boundary_maps_after_insertion(self) -> None:
        gaps = [DeviceGap(1, 100, 5, 5.0)]
        self.assertEqual(
            canonicalize_master_sample(100, gaps, master_device_index=1),
            105,
        )

    def test_isolated_nonpersistent_offset_excursion_crops_the_tail(self) -> None:
        fs = 1_000
        observations = [
            _observation(float(time_sec), -1_856.0 if time_sec == 55 else 0.0)
            for time_sec in range(5, 110, 5)
        ]
        options = SyncOptions(
            window_seconds=10.0,
            step_seconds=5.0,
            gap_persistence_observations=2,
            gap_min_step_samples=50.0,
        )
        model = fit_affine_sync_model(observations, fs, options=options)

        detected = detect_isolated_offset_crop(observations, model, fs, options)

        self.assertIsNotNone(detected)
        crop_sample, shift, observation_indices = detected
        self.assertEqual(crop_sample, 45_000)
        self.assertAlmostEqual(shift, -1_856.0, places=6)
        self.assertEqual(observation_indices, (10,))

    def test_unconfirmed_terminal_shift_crops_complete_anomalous_window(self) -> None:
        fs = 1_000
        observations = [
            _observation(float(time_sec), 0.0)
            for time_sec in range(5, 105, 5)
        ]
        observations.append(_observation(105.0, -1_856.0))
        options = SyncOptions(
            window_seconds=10.0,
            step_seconds=5.0,
            gap_persistence_observations=2,
            gap_min_step_samples=50.0,
            gap_level_tolerance_samples=12.0,
        )
        model = fit_affine_sync_model(observations, fs, options=options)

        detected = detect_unconfirmed_terminal_crop(observations, model, fs, options)

        self.assertIsNotNone(detected)
        crop_sample, shift, observation_indices = detected
        self.assertEqual(crop_sample, 95_000)
        self.assertAlmostEqual(shift, -1_856.0, places=6)
        self.assertEqual(observation_indices, (len(observations) - 1,))

    def test_interior_isolated_shift_is_not_treated_as_terminal_crop(self) -> None:
        fs = 1_000
        observations = [
            _observation(float(time_sec), -1_856.0 if time_sec == 55 else 0.0)
            for time_sec in range(5, 110, 5)
        ]
        options = SyncOptions(
            window_seconds=10.0,
            step_seconds=5.0,
            gap_persistence_observations=2,
            gap_min_step_samples=50.0,
        )
        model = fit_affine_sync_model(observations, fs, options=options)

        self.assertIsNone(
            detect_unconfirmed_terminal_crop(observations, model, fs, options)
        )

    def test_terminal_crop_limits_common_merge_end(self) -> None:
        recording = Recording(
            folder=Path("recording"),
            amplifier_file=Path("amplifier.dat"),
            analog_file=Path("analogin.dat"),
            ce_params_file=Path("CE_params.bin"),
            device_name="device",
            recording_name="recording",
            fs=1_000,
            n_channels=1,
            n_samples=20_000,
            analog_channels=1,
            analog_samples=25_000,
        )

        start, end, limits = _common_master_interval(
            [recording, recording, recording],
            [_model(()), _model(()), _model(())],
            0,
            maximum_common_end=12_345,
        )

        self.assertLess(start, end)
        self.assertEqual(end, 12_345)
        self.assertEqual(limits["end_limiter"]["stream"], "validated_terminal_sync")

    def test_early_slave_and_master_gaps_cannot_publish_a_corrupt_prefix(self) -> None:
        fs = 1_000
        sample_count = 40_000
        # The boundary begins just after the 2-second endpoint probe. This was
        # the adversarial case that previously localized to sample 2 and left
        # the saved prefix inside the true gap.
        gap_start = 2_100
        gap_size = 317
        rng = np.random.default_rng(123)
        canonical = rng.normal(scale=400.0, size=sample_count)
        canonical += 700.0 * np.sin(2 * np.pi * 350 * np.arange(sample_count) / fs)
        options = SyncOptions(
            initial_start_seconds=5.0,
            initial_duration_seconds=10.0,
            initial_max_lag_seconds=1.0,
            window_seconds=2.0,
            step_seconds=1.0,
            tracking_max_lag_samples=20,
            reacquisition_max_lag_seconds=1.0,
            endpoint_probe_seconds=2.0,
            highpass_hz=200.0,
            peak_exclusion_samples=12,
            gap_min_step_samples=50.0,
            gap_persistence_observations=2,
            gap_event_time_tolerance_seconds=1.0,
            max_parallel_workers=2,
            chunk_seconds=2.0,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for missing_device in (1, 0):
                with self.subTest(missing_device=missing_device):
                    case = root / f"missing_{missing_device}"
                    folders: list[Path] = []
                    for device_index in range(3):
                        signal = (
                            np.concatenate(
                                (canonical[:gap_start], canonical[gap_start + gap_size :])
                            )
                            if device_index == missing_device
                            else canonical
                        )
                        folder = case / f"device{device_index}" / "recording"
                        folder.mkdir(parents=True)
                        channels = np.column_stack([signal + channel for channel in range(4)])
                        np.clip(np.rint(channels), -32768, 32767).astype("<i2").tofile(
                            folder / "amplifier.dat"
                        )
                        np.zeros((int(np.floor(signal.size * 1_250 / fs)), 1), dtype="<i2").tofile(
                            folder / "analogin.dat"
                        )
                        header = bytearray(512)
                        struct.pack_into("<I", header, 0, fs)
                        struct.pack_into("<I", header, 8, 4)
                        (folder / "CE_params.bin").write_bytes(header)
                        folders.append(folder)

                    output = case / "output"
                    result = run_multidevice_sync(
                        folders,
                        master_index=0,
                        output_folder=output,
                        merge=True,
                        validate_postmerge=True,
                        options=options,
                    )
                    self.assertIn(
                        result.status,
                        {"OK", "WARN"},
                        [(pair.status, pair.message) for pair in result.pairs]
                        + result.unresolved_gap_messages,
                    )
                    manifest = json.loads(
                        (output / "wild_preprocess_run.json").read_text(encoding="utf-8")
                    )
                    merge_info = manifest["merge"]
                    self.assertGreaterEqual(
                        int(merge_info["common_start_master_sample"]),
                        gap_start + gap_size,
                        (
                            [(pair.model.offset_steps, pair.validated_start_master_sample) for pair in result.pairs],
                            result.device_gaps,
                        ),
                    )
                    self.assertEqual(
                        merge_info["common_interval_limits"]["start_limiter"]["stream"],
                        "validated_endpoint_probe",
                    )
                    if result.device_gaps:
                        self.assertEqual(
                            result.device_gaps[0].missing_samples,
                            gap_size,
                            [pair.model.offset_steps for pair in result.pairs],
                        )
                        self.assertGreaterEqual(
                            int(merge_info["common_start_master_sample"]),
                            result.device_gaps[0].canonical_end_sample + 16,
                        )
                        self.assertTrue(
                            all(
                                gap["action"] == "cropped_before_output"
                                for gap in merge_info["device_gaps"]
                            )
                        )

    def test_unconfirmed_terminal_gap_invalidates_only_affected_slave_tail(self) -> None:
        fs = 1_000
        sample_count = 110_000
        gap_start = 99_100
        gap_size = 317
        rng = np.random.default_rng(321)
        canonical = rng.normal(scale=400.0, size=sample_count)
        canonical += 700.0 * np.sin(2 * np.pi * 350 * np.arange(sample_count) / fs)
        options = SyncOptions(
            initial_start_seconds=30.0,
            initial_duration_seconds=60.0,
            initial_max_lag_seconds=1.0,
            window_seconds=10.0,
            step_seconds=5.0,
            tracking_max_lag_samples=20,
            reacquisition_max_lag_seconds=1.0,
            endpoint_probe_seconds=2.0,
            highpass_hz=200.0,
            peak_exclusion_samples=12,
            gap_min_step_samples=50.0,
            gap_persistence_observations=2,
            gap_event_time_tolerance_seconds=1.0,
            max_parallel_workers=2,
            chunk_seconds=2.0,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folders: list[Path] = []
            for device_index in range(3):
                signal = (
                    np.concatenate((canonical[:gap_start], canonical[gap_start + gap_size :]))
                    if device_index == 2
                    else canonical
                )
                folder = root / f"device{device_index}" / "recording"
                folder.mkdir(parents=True)
                channels = np.column_stack([signal + channel for channel in range(4)])
                np.clip(np.rint(channels), -32768, 32767).astype("<i2").tofile(
                    folder / "amplifier.dat"
                )
                np.zeros((int(np.floor(signal.size * 1_250 / fs)), 1), dtype="<i2").tofile(
                    folder / "analogin.dat"
                )
                header = bytearray(512)
                struct.pack_into("<I", header, 0, fs)
                struct.pack_into("<I", header, 8, 4)
                (folder / "CE_params.bin").write_bytes(header)
                folders.append(folder)

            output = root / "output"
            result = run_multidevice_sync(
                folders,
                master_index=0,
                output_folder=output,
                merge=True,
                validate_postmerge=True,
                options=options,
            )

            self.assertEqual(result.status, "WARN", result.unresolved_gap_messages)
            affected = next(pair for pair in result.pairs if pair.slave_index == 3)
            self.assertEqual(affected.status, "WARN")
            self.assertIsNotNone(affected.terminal_crop_master_sample)
            self.assertLessEqual(affected.terminal_crop_master_sample, gap_start)
            self.assertFalse(result.device_gaps)
            manifest = json.loads(
                (output / "wild_preprocess_run.json").read_text(encoding="utf-8")
            )
            merge_info = manifest["merge"]
            self.assertGreater(
                int(merge_info["common_end_master_sample"]),
                affected.terminal_crop_master_sample,
            )
            self.assertEqual(
                int(merge_info["common_interval_limits"]["end_limiter"]["device_index"]),
                1,
            )
            validity = np.fromfile(output / "valid_samples.dat", dtype=np.uint8).reshape(-1, 3)
            crop_row = affected.terminal_crop_master_sample - int(
                merge_info["common_start_master_sample"]
            )
            self.assertTrue(np.all(validity[crop_row:, :2] == 1))
            self.assertTrue(np.all(validity[crop_row:, 2] == 0))
            self.assertTrue(
                any(
                    interval["kind"] == "terminal_unsupported"
                    and interval["affected_device_indices"] == [3]
                    for interval in merge_info["classified_intervals"]
                )
            )
            self.assertEqual(
                manifest["sync"]["pairs"][1]["terminal_crop_master_sample"],
                affected.terminal_crop_master_sample,
            )

    def test_narrow_and_wide_candidates_share_one_fft_profile(self) -> None:
        rng = np.random.default_rng(2)
        master = rng.normal(size=4096)
        slave = np.zeros_like(master)
        slave[317:] = master[:-317]
        with patch.object(
            observe_module,
            "_correlation_profile",
            wraps=observe_module._correlation_profile,
        ) as profile:
            narrow, wide = estimate_lag_narrow_wide(
                master,
                slave,
                20,
                500,
                peak_exclusion_samples=12,
            )
        self.assertEqual(profile.call_count, 1)
        self.assertNotEqual(narrow.lag_samples, 317)
        self.assertEqual(wide.lag_samples, 317)

    def test_arbitrary_gap_is_removed_before_affine_drift_fit(self) -> None:
        fs = 1_000.0
        slope = 0.01
        observations = [
            _observation(float(index), 12.0 + slope * index - (317.0 if index >= 20 else 0.0))
            for index in range(40)
        ]
        model = fit_affine_sync_model(
            observations,
            fs,
            options=SyncOptions(
                step_seconds=1.0,
                gap_min_step_samples=50.0,
                gap_persistence_observations=2,
                short_recording_seconds=10.0,
            ),
        )
        self.assertEqual(len(model.offset_steps), 1)
        self.assertAlmostEqual(model.offset_steps[0].offset_step_samples, -317.0, places=6)
        self.assertAlmostEqual(model.intercept_samples, 12.0, places=5)
        self.assertAlmostEqual(model.drift_ppm, 10.0, places=5)
        self.assertLess(model.residual_rms_samples, 1e-8)

    def test_sign_specific_gap_attribution(self) -> None:
        options = SyncOptions(gap_event_time_tolerance_seconds=1.0)
        slave_gap, unresolved = infer_device_gaps(
            [_pair(2, (_step(-317.0),)), _pair(3, ())],
            device_count=3,
            master_index=0,
            fs=1_000.0,
            options=options,
        )
        self.assertFalse(unresolved)
        self.assertEqual([(gap.device_index, gap.missing_samples) for gap in slave_gap], [(2, 317)])

        master_gaps, unresolved = infer_device_gaps(
            [_pair(2, (_step(4096.0),)), _pair(3, (_step(4094.0),))],
            device_count=3,
            master_index=0,
            fs=1_000.0,
            options=options,
        )
        self.assertFalse(unresolved)
        self.assertEqual([(gap.device_index, gap.missing_samples) for gap in master_gaps], [(1, 4095)])

        gaps, unresolved = infer_device_gaps(
            [_pair(2, (_step(1856.0),)), _pair(3, ())],
            device_count=3,
            master_index=0,
            fs=1_000.0,
            options=options,
        )
        self.assertFalse(gaps)
        self.assertTrue(unresolved)

    def test_two_device_gap_is_not_auto_attributed(self) -> None:
        gaps, unresolved = infer_device_gaps(
            [_pair(2, (_step(-1856.0),))],
            device_count=2,
            master_index=0,
            fs=20_000.0,
            options=SyncOptions(),
        )
        self.assertFalse(gaps)
        self.assertIn("at least three devices", unresolved[0])

    def test_source_coordinates_insert_exact_missing_interval(self) -> None:
        positions = np.arange(95, 111, dtype=np.float64)
        gap = DeviceGap(
            device_index=2,
            canonical_start_sample=100,
            missing_samples=5,
            duration_ms=5.0,
        )
        source, valid = _source_coordinates(_model(()), positions, (gap,), fs=1_000.0)
        np.testing.assert_array_equal(valid, ~((positions >= 100) & (positions < 105)))
        np.testing.assert_array_equal(source[positions < 100], positions[positions < 100])
        np.testing.assert_array_equal(source[positions >= 105], positions[positions >= 105] - 5)

    def test_master_gap_expands_canonical_interval(self) -> None:
        recordings = [
            Recording(
                folder=Path(f"device{index}"),
                amplifier_file=Path("unused.dat"),
                analog_file=Path("unused-analog.dat"),
                ce_params_file=Path("unused.bin"),
                device_name=f"device{index}",
                recording_name="recording",
                fs=20_000,
                n_channels=1,
                n_samples=995 if index == 0 else 1_000,
                analog_channels=1,
                analog_samples=63,
            )
            for index in range(3)
        ]
        start, end, _limits = _common_master_interval(
            recordings,
            [_model(()), _model(()), _model(())],
            0,
            [
                DeviceGap(
                    device_index=1,
                    canonical_start_sample=500,
                    missing_samples=5,
                    duration_ms=0.25,
                )
            ],
        )
        self.assertEqual(start, 16)
        self.assertEqual(end, 983)

    def test_adaptive_observation_recaptures_a_nonstandard_gap_size(self) -> None:
        fs = 100
        rng = np.random.default_rng(9)
        master_feature = rng.normal(size=10_000).astype("<f4")
        slave_feature = np.concatenate((master_feature[:4_000], master_feature[4_030:]))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            master_path = root / "master.f32"
            slave_path = root / "slave.f32"
            master_feature.tofile(master_path)
            slave_feature.tofile(slave_path)
            master = Recording(
                folder=root / "master",
                amplifier_file=root / "unused-master.dat",
                analog_file=root / "unused-master-analog.dat",
                ce_params_file=root / "unused-master.bin",
                device_name="master",
                recording_name="recording",
                fs=fs,
                n_channels=4,
                n_samples=master_feature.size,
                analog_channels=1,
                analog_samples=625,
            )
            slave = Recording(
                folder=root / "slave",
                amplifier_file=root / "unused-slave.dat",
                analog_file=root / "unused-slave-analog.dat",
                ce_params_file=root / "unused-slave.bin",
                device_name="slave",
                recording_name="recording",
                fs=fs,
                n_channels=4,
                n_samples=slave_feature.size,
                analog_channels=1,
                analog_samples=623,
            )
            result = observe_pair(
                master,
                slave,
                master_path,
                slave_path,
                SyncOptions(
                    initial_start_seconds=1.0,
                    initial_duration_seconds=10.0,
                    initial_max_lag_seconds=1.0,
                    window_seconds=2.0,
                    step_seconds=1.0,
                    tracking_max_lag_samples=10,
                    reacquisition_max_lag_seconds=1.0,
                    peak_exclusion_samples=4,
                ),
            )
        reacquired = [item for item in result.observations if item.search_mode == "wide_reacquisition"]
        self.assertTrue(reacquired)
        self.assertTrue(any(abs(item.observed_offset_samples + 30) <= 1 for item in reacquired))

    def test_gap_aware_stream_zero_fills_only_affected_device(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fs = 1_000
            canonical = np.arange(240, dtype=np.int16) + 100
            sources = [canonical, np.concatenate((canonical[:100], canonical[105:])), canonical]
            recordings: list[Recording] = []
            for index, values in enumerate(sources):
                folder = root / f"device{index}"
                folder.mkdir()
                amplifier = folder / "amplifier.dat"
                values.astype("<i2").reshape(-1, 1).tofile(amplifier)
                analog = folder / "analogin.dat"
                np.zeros((max(1, values.size // 16), 1), dtype="<i2").tofile(analog)
                recordings.append(
                    Recording(
                        folder=folder,
                        amplifier_file=amplifier,
                        analog_file=analog,
                        ce_params_file=folder / "CE_params.bin",
                        device_name=f"device{index}",
                        recording_name="recording",
                        fs=fs,
                        n_channels=1,
                        n_samples=values.size,
                        analog_channels=1,
                        analog_samples=max(1, values.size // 16),
                    )
                )
            output = root / "merged.dat"
            _write_interleaved_stream(
                output,
                recordings,
                [_model(()), _model(()), _model(())],
                0,
                20,
                215,
                stream="ephys",
                chunk_seconds=1.0,
                overwrite=False,
                progress=None,
                device_gaps=[
                    DeviceGap(
                        device_index=2,
                        canonical_start_sample=100,
                        missing_samples=5,
                        duration_ms=5.0,
                    )
                ],
            )
            merged = np.fromfile(output, dtype="<i2").reshape(-1, 3)
        gap_rows = np.arange(100, 105) - 20
        self.assertTrue(np.all(merged[gap_rows, 1] == 0))
        self.assertTrue(np.all(merged[gap_rows, 0] != 0))
        self.assertTrue(np.all(merged[gap_rows, 2] != 0))
        self.assertEqual(int(merged[130 - 20, 1]), int(canonical[130]))

    def test_validity_stream_is_sample_major_and_master_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fs = 1_000
            source_values = [
                np.arange(240, dtype=np.int16) + 100,
                np.arange(240, dtype=np.int16) + 1_000,
                np.arange(240, dtype=np.int16) + 2_000,
            ]
            recordings: list[Recording] = []
            for index, values in enumerate(source_values):
                folder = root / f"device{index}"
                folder.mkdir()
                amplifier = folder / "amplifier.dat"
                values.astype("<i2").reshape(-1, 1).tofile(amplifier)
                analog = folder / "analogin.dat"
                np.zeros((20, 1), dtype="<i2").tofile(analog)
                recordings.append(
                    Recording(
                        folder=folder,
                        amplifier_file=amplifier,
                        analog_file=analog,
                        ce_params_file=folder / "CE_params.bin",
                        device_name=f"device{index}",
                        recording_name="recording",
                        fs=fs,
                        n_channels=1,
                        n_samples=values.size,
                        analog_channels=1,
                        analog_samples=20,
                    )
                )
            output = root / "merged.dat"
            validity_path = root / "valid_samples.dat"
            _write_interleaved_stream(
                output,
                recordings,
                [_model(()), _model(()), _model(())],
                1,
                20,
                180,
                stream="ephys",
                chunk_seconds=0.03,
                overwrite=False,
                progress=None,
                device_gaps=[
                    DeviceGap(
                        device_index=1,
                        canonical_start_sample=100,
                        missing_samples=5,
                        duration_ms=5.0,
                    )
                ],
                validity_path=validity_path,
            )
            validity = np.fromfile(validity_path, dtype=np.uint8).reshape(-1, 3)
            amplifier = np.fromfile(output, dtype="<i2").reshape(-1, 3)
            validity_size = validity_path.stat().st_size

        self.assertEqual(validity.shape, (161, 3))
        self.assertEqual(validity_size, 161 * 3)
        self.assertTrue(np.all(np.isin(validity, [0, 1])))
        self.assertTrue(np.all(validity[:, 0] == 1))
        self.assertTrue(np.all(validity[:, 2] == 1))
        invalid_rows = np.arange(84, 121) - 20
        self.assertTrue(np.all(validity[invalid_rows, 1] == 0))
        self.assertTrue(np.all(validity[np.setdiff1d(np.arange(161), invalid_rows), 1] == 1))
        self.assertTrue(np.all(amplifier[invalid_rows, 0] == 0))
        self.assertTrue(np.all(amplifier[invalid_rows, 1:] != 0))

    def test_three_device_pipeline_repairs_one_attributable_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fs = 2_000
            sample_count = 140_000
            gap_start = 70_000
            gap_size = 317
            rng = np.random.default_rng(44)
            canonical = rng.normal(scale=500.0, size=sample_count)
            canonical += 800.0 * np.sin(2 * np.pi * 350 * np.arange(sample_count) / fs)
            device_signals = [
                canonical,
                np.concatenate((canonical[:gap_start], canonical[gap_start + gap_size :])),
                canonical,
            ]
            folders: list[Path] = []
            for device_index, signal in enumerate(device_signals):
                folder = root / f"device{device_index}" / "recording"
                folder.mkdir(parents=True)
                channels = np.column_stack([signal + channel for channel in range(4)])
                np.clip(np.rint(channels), -32768, 32767).astype("<i2").tofile(
                    folder / "amplifier.dat"
                )
                analog_samples = int(np.floor(signal.size * 1250 / fs))
                np.zeros((analog_samples, 1), dtype="<i2").tofile(folder / "analogin.dat")
                header = bytearray(512)
                struct.pack_into("<I", header, 0, fs)
                struct.pack_into("<I", header, 8, 4)
                (folder / "CE_params.bin").write_bytes(header)
                folders.append(folder)
            options = SyncOptions(
                initial_start_seconds=2.0,
                initial_duration_seconds=10.0,
                initial_max_lag_seconds=1.0,
                window_seconds=2.0,
                step_seconds=1.0,
                tracking_max_lag_samples=20,
                reacquisition_max_lag_seconds=1.0,
                highpass_hz=200.0,
                peak_exclusion_samples=12,
                gap_min_step_samples=50.0,
                gap_persistence_observations=2,
                gap_event_time_tolerance_seconds=1.5,
                max_parallel_workers=2,
                chunk_seconds=2.0,
            )
            output = root / "output"
            result = run_multidevice_sync(
                folders,
                master_index=0,
                output_folder=output,
                merge=True,
                options=options,
            )
            self.assertIn(
                result.status,
                {"OK", "WARN"},
                [(pair.status, pair.message, len(pair.model.offset_steps)) for pair in result.pairs]
                + result.unresolved_gap_messages,
            )
            self.assertEqual(len(result.device_gaps), 1)
            gap = result.device_gaps[0]
            self.assertEqual(gap.device_index, 2)
            self.assertEqual(gap.missing_samples, gap_size)
            self.assertAlmostEqual(gap.canonical_start_sample, gap_start, delta=2)
            validity = np.fromfile(output / "valid_samples.dat", dtype=np.uint8).reshape(-1, 3)
            self.assertEqual(validity.shape[0], np.fromfile(output / "time.dat", dtype="<i4").size)
            self.assertTrue(np.all(validity[:, 0] == 1))
            self.assertTrue(np.all(validity[:, 2] == 1))
            self.assertTrue(np.any(validity[:, 1] == 0))
            manifest = json.loads(
                (output / "wild_preprocess_run.json").read_text(encoding="utf-8")
            )
            layout_rows = manifest["merge"]["channel_layout"]
            self.assertEqual([row["validity_channel"] for row in layout_rows[:4]], [0] * 4)
            self.assertEqual(manifest["sync"]["device_gaps"][0]["device_index"], 2)
            self.assertEqual(
                manifest["sync"]["device_gaps"][0]["missing_samples"], gap_size
            )

            serial_result = run_multidevice_sync(
                folders,
                master_index=0,
                output_folder=root / "serial_qc",
                merge=True,
                options=SyncOptions(**{**options.__dict__, "max_parallel_workers": 1}),
            )
            self.assertEqual(serial_result.status, result.status)
            self.assertEqual(serial_result.device_gaps, result.device_gaps)
            self.assertEqual(
                [pair.model for pair in serial_result.pairs],
                [pair.model for pair in result.pairs],
            )
            self.assertEqual(
                serial_result.device_sync_segments,
                result.device_sync_segments,
            )
            self.assertEqual(
                serial_result.classified_intervals,
                result.classified_intervals,
            )
            for filename in ("amplifier.dat", "valid_samples.dat", "time.dat"):
                self.assertEqual(
                    (root / "serial_qc" / filename).read_bytes(),
                    (output / filename).read_bytes(),
                )

            master_gap_output = root / "master_gap_output"
            master_gap_result = run_multidevice_sync(
                folders,
                master_index=1,
                output_folder=master_gap_output,
                merge=True,
                native_pc_time=True,
                validate_postmerge=True,
                options=options,
            )
            self.assertEqual(master_gap_result.status, "WARN")
            self.assertEqual(master_gap_result.outputs["merge_status"], "WARN")
            self.assertEqual(master_gap_result.outputs["pc_time_status"], "WARN")
            self.assertEqual(master_gap_result.outputs["overall_status"], "MERGE_ONLY")
            self.assertFalse((master_gap_output / "pc_time.dat").exists())
            self.assertTrue((master_gap_output / "amplifier.dat").exists())
            self.assertEqual(master_gap_result.device_gaps[0].device_index, 2)
            postmerge = json.loads(
                (master_gap_output / "wild_preprocess_run.json").read_text(encoding="utf-8")
            )["merge"]["postmerge_validation"]
            gap_positions = {
                measurement["position"]
                for measurement in postmerge["measurements"]
                if measurement["position"].startswith(("boundary", "segment"))
            }
            self.assertTrue(any(position.endswith("_before") for position in gap_positions))
            self.assertTrue(any(position.endswith("_after") for position in gap_positions))


if __name__ == "__main__":
    unittest.main()
