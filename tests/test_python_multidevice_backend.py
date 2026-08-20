from __future__ import annotations

import json
import struct
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

from wild_preprocess.binary_io import close_memmap, interleaved_memmap, read_ce_params_header, recording_from_folder
from wild_preprocess.models import SyncObservation, SyncOptions
from wild_preprocess.pipeline import run_multidevice_sync
from wild_preprocess.pc_time import align_pc_time_file
from wild_preprocess.sync.infer import fit_affine_sync_model
from wild_preprocess.sync.merge import (
    _linear_resampled_chunk,
    _recover_interrupted_transactions,
    _windowed_sinc_resampled_chunk,
)
from wild_preprocess.sync.observe import CorrelationProfile, LagEstimate, _select_lag, estimate_lag
from wild_preprocess.sync.validate import validate_pair
from WILD_preprocess_gui.wild_preprocess_gui import (
    RecordingInfo,
    pc_time_valid_for_recording,
    preflight_checks,
    ready_check_log_lines,
)


def _write_ce_params(path: Path, fs: int, n_channels: int) -> None:
    data = bytearray(512)
    struct.pack_into("<I", data, 0, fs)
    struct.pack_into("<I", data, 8, n_channels)
    data[440] = 0
    path.write_bytes(data)


def _write_ce_params_rtc(
    path: Path,
    fs: int,
    n_channels: int,
    *,
    day: int = 11,
    month: int = 8,
    year: int = 26,
    hours: int = 18,
    minutes: int = 13,
    seconds: int = 23,
) -> None:
    _write_ce_params(path, fs, n_channels)
    data = bytearray(path.read_bytes())
    struct.pack_into("<BBBB", data, 332, 2, month, day, year)
    struct.pack_into("<BBBB", data, 336, hours, minutes, seconds, 0)
    struct.pack_into("<IIII", data, 340, 1_839, 9_999, 0, 0)
    path.write_bytes(data)


def _write_recording(
    folder: Path,
    base_signal: np.ndarray,
    *,
    fs: int,
    n_channels: int,
    intercept: float,
    drift_ppm: float,
) -> None:
    folder.mkdir(parents=True)
    scale = 1.0 + drift_ppm * 1e-6
    device_indices = np.arange(base_signal.size, dtype=float)
    master_positions = (device_indices - intercept) / scale
    signal = np.interp(master_positions, np.arange(base_signal.size), base_signal, left=0.0, right=0.0)
    channels = np.column_stack([signal + channel * 3 for channel in range(n_channels)])
    np.clip(np.rint(channels), -32768, 32767).astype("<i2").tofile(folder / "amplifier.dat")
    analog_channels = n_channels // 4
    analog = np.zeros((round(base_signal.size * 1250 / fs), analog_channels), dtype="<i2")
    if analog.size:
        analog[:, 0] = ((np.arange(analog.shape[0]) // 100) % 2).astype(np.int16)
    analog.tofile(folder / "analogin.dat")
    _write_ce_params(folder / "CE_params.bin", fs, n_channels)


class PythonMultideviceBackendTest(unittest.TestCase):
    @staticmethod
    def _preflight_recording(folder: Path) -> RecordingInfo:
        return RecordingInfo(
            use=True,
            role="master",
            probe_index=1,
            device_id="device",
            recording_name=folder.name,
            folder=folder,
            fs=20_000,
            n_channels=64,
            n_samples=1,
            duration_sec=20.0,
            time_dat_valid=False,
            info_rhd_exists=False,
            imu_mat_exists=False,
            pc_time_exists=False,
            pc_time_valid=False,
        )

    def test_ready_check_uses_ce_rtc_independently_of_folder_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "renamed_without_timestamp"
            folder.mkdir()
            (folder / "amplifier.dat").write_bytes(bytes(128))
            (folder / "analogin.dat").write_bytes(bytes(32))
            _write_ce_params_rtc(folder / "CE_params.bin", 20_000, 64)
            checks = preflight_checks([self._preflight_recording(folder)], Path(temporary))
            recording_check = next(item for item in checks if item[1] == "device/renamed_without_timestamp")
            self.assertEqual(recording_check, ("OK", "device/renamed_without_timestamp", "ready"))

    def test_ready_check_fails_before_worker_when_ce_rtc_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "1_20260811_181323.429"
            folder.mkdir()
            (folder / "amplifier.dat").write_bytes(bytes(128))
            (folder / "analogin.dat").write_bytes(bytes(32))
            _write_ce_params(folder / "CE_params.bin", 20_000, 64)
            checks = preflight_checks([self._preflight_recording(folder)], Path(temporary))
            recording_check = next(item for item in checks if item[1] == "device/1_20260811_181323.429")
            self.assertEqual(recording_check[0], "FAIL")
            self.assertIn("invalid/missing CE RTC", recording_check[2])

    def test_ready_check_assumes_master_start_for_incompatible_slave(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "device1" / "recording"
            second = root / "device2" / "recording"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            for folder in (first, second):
                (folder / "amplifier.dat").write_bytes(bytes(128))
                (folder / "analogin.dat").write_bytes(bytes(32))
            _write_ce_params_rtc(first / "CE_params.bin", 20_000, 64, day=11)
            _write_ce_params_rtc(second / "CE_params.bin", 20_000, 64, day=12)
            first_info = self._preflight_recording(first)
            first_info.device_id = "device1"
            second_info = self._preflight_recording(second)
            second_info.device_id = "device2"
            second_info.role = "slave"
            checks = preflight_checks([first_info, second_info], root)
            compatibility = next(
                item for item in checks if item[1] == "recording start compatibility"
            )
            slave_check = next(item for item in checks if item[1] == "device2/recording")
            self.assertEqual(slave_check[0], "WARN")
            self.assertIn("assumed simultaneous with master", slave_check[2])
            self.assertEqual(compatibility[0], "OK")
            self.assertIn("1 slave start(s) assumed", compatibility[2])

    def test_ready_check_keeps_master_rtc_strict_and_tolerates_uninitialized_slave(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "device1" / "recording"
            uninitialized = root / "device2" / "recording"
            valid.mkdir(parents=True)
            uninitialized.mkdir(parents=True)
            for folder in (valid, uninitialized):
                (folder / "amplifier.dat").write_bytes(bytes(128))
                (folder / "analogin.dat").write_bytes(bytes(32))
            _write_ce_params_rtc(valid / "CE_params.bin", 20_000, 64)
            _write_ce_params_rtc(
                uninitialized / "CE_params.bin",
                20_000,
                64,
                day=1,
                month=1,
                year=0,
                hours=0,
                minutes=0,
                seconds=8,
            )
            master = self._preflight_recording(valid)
            master.device_id = "device1"
            slave = self._preflight_recording(uninitialized)
            slave.device_id = "device2"
            slave.role = "slave"

            checks = preflight_checks([master, slave], root)
            slave_check = next(item for item in checks if item[1] == "device2/recording")
            self.assertEqual(slave_check[0], "WARN")
            self.assertIn("uninitialized date 2000-01-01", slave_check[2])
            self.assertFalse(any(status == "FAIL" for status, _check, _detail in checks))

            (uninitialized / "amplifier.dat").unlink()
            checks = preflight_checks([master, slave], root)
            slave_check = next(item for item in checks if item[1] == "device2/recording")
            self.assertEqual(slave_check[0], "FAIL")
            self.assertIn("missing amplifier.dat", slave_check[2])
            (uninitialized / "amplifier.dat").write_bytes(bytes(128))

            master.role = "slave"
            slave.role = "master"
            checks = preflight_checks([master, slave], root)
            master_check = next(item for item in checks if item[1] == "device2/recording")
            self.assertEqual(master_check[0], "FAIL")
            self.assertIn("uninitialized date 2000-01-01", master_check[2])

    def test_ready_check_log_lines_include_actionable_details(self) -> None:
        checks = [
            ("OK", "selected recordings", "2 selected"),
            ("FAIL", "master", "0 selected"),
            ("WARN", "duration spread", "1977.203 sec"),
        ]
        self.assertEqual(
            ready_check_log_lines(checks),
            [
                "Ready Check: 1 fail, 1 warn",
                "  FAIL: master: 0 selected",
                "  WARN: duration spread: 1977.203 sec",
            ],
        )

    def test_ready_check_rejects_invalid_header_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "device1" / "recording"
            first.mkdir(parents=True)
            (first / "amplifier.dat").write_bytes(bytes(128))
            (first / "analogin.dat").write_bytes(bytes(32))
            _write_ce_params_rtc(first / "CE_params.bin", 20_000, 64)
            first_info = self._preflight_recording(first)
            first_info.device_id = "device1"

            first_info.fs = None
            first_info.n_channels = None
            checks = preflight_checks([first_info], root)
            recording_check = next(item for item in checks if item[1] == "device1/recording")
            self.assertEqual(recording_check[0], "FAIL")
            self.assertIn("sample rate", recording_check[2])
            self.assertIn("channel count", recording_check[2])

    def test_header_and_interleaved_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "device" / "recording"
            folder.mkdir(parents=True)
            _write_ce_params(folder / "CE_params.bin", 20_000, 64)
            data = np.arange(80, dtype=np.int16).reshape(10, 8)
            data.astype("<i2").tofile(folder / "analogin.dat")
            np.arange(640, dtype=np.int16).reshape(10, 64).astype("<i2").tofile(folder / "amplifier.dat")
            self.assertEqual(read_ce_params_header(folder / "CE_params.bin"), (20_000, 64))
            recording = recording_from_folder(folder)
            mapped = interleaved_memmap(recording.amplifier_file, 64, 10)
            np.testing.assert_array_equal(mapped[3], np.arange(192, 256, dtype=np.int16))
            close_memmap(mapped)

    def test_lag_sign_matches_source_sample_mapping(self) -> None:
        rng = np.random.default_rng(5)
        master = rng.normal(size=4096)
        slave = np.zeros_like(master)
        slave[37:] = master[:-37]
        estimate = estimate_lag(master, slave, 100, peak_exclusion_samples=12)
        self.assertEqual(estimate.lag_samples, 37)
        self.assertGreater(estimate.peak_margin_fraction, 0.5)

    def test_affine_fit_rejects_single_outlier(self) -> None:
        observations = []
        for index, time_sec in enumerate(np.arange(0.0, 100.0, 5.0)):
            offset = 12.0 + 0.02 * time_sec
            if index == 8:
                offset += 80.0
            observations.append(
                SyncObservation(
                    center_time_sec=float(time_sec),
                    predicted_offset_samples=offset,
                    observed_offset_samples=offset,
                    residual_lag_samples=0.0,
                    peak_correlation=1.0,
                    peak_to_background=20.0,
                    peak_margin_fraction=0.5,
                    secondary_lag_samples=None,
                    accepted=True,
                )
            )
        model = fit_affine_sync_model(observations, fs=20_000)
        self.assertAlmostEqual(model.intercept_samples, 12.0, places=5)
        self.assertAlmostEqual(model.drift_ppm, 1.0, places=5)
        self.assertLess(model.residual_rms_samples, 1e-6)

    def test_periodic_peak_reports_ambiguity(self) -> None:
        samples = np.arange(5000)
        periodic = np.sin(2 * np.pi * samples / 64)
        estimate = estimate_lag(periodic, periodic, 256, peak_exclusion_samples=20)
        self.assertLess(estimate.peak_margin_fraction, 0.02)

    def test_endpoint_competitor_is_included_in_peak_ambiguity(self) -> None:
        estimate = _select_lag(
            CorrelationProfile(
                lags=np.arange(-4, 5),
                correlations=np.array([0.99, 0.1, 0.2, 0.3, 1.0, 0.3, 0.2, 0.1, 0.98]),
            ),
            4,
            peak_exclusion_samples=2,
        )
        self.assertEqual(abs(estimate.secondary_lag_samples), 4)
        self.assertLess(estimate.peak_margin_fraction, 0.02)

    def test_initial_search_boundary_fails_pair_qc(self) -> None:
        initial = LagEstimate(
            lag_samples=10,
            peak_correlation=0.9,
            peak_to_background=10.0,
            peak_margin_fraction=0.5,
            secondary_lag_samples=None,
            lags=np.arange(-10, 11),
            correlations=np.ones(21),
        )
        observations = []
        for index in range(20):
            observation = SyncObservation(
                center_time_sec=float(index * 5),
                predicted_offset_samples=10.0,
                observed_offset_samples=10.0,
                residual_lag_samples=0.0,
                peak_correlation=0.9,
                peak_to_background=10.0,
                peak_margin_fraction=0.5,
                secondary_lag_samples=None,
                accepted=True,
            )
            observations.append(observation)
        model = fit_affine_sync_model(observations, fs=20_000)
        status, message = validate_pair(initial, observations, model, SyncOptions())
        self.assertEqual(status, "FAIL")
        self.assertIn("search boundary", message)

    def test_consecutive_rejections_fail_qc(self) -> None:
        rng = np.random.default_rng(20)
        signal = rng.normal(size=2000)
        initial = estimate_lag(signal, signal, 20, peak_exclusion_samples=5)
        observations = [
            SyncObservation(
                center_time_sec=float(index * 5),
                predicted_offset_samples=0.0,
                observed_offset_samples=0.0,
                residual_lag_samples=0.0,
                peak_correlation=1.0,
                peak_to_background=10.0,
                peak_margin_fraction=0.5,
                secondary_lag_samples=None,
                accepted=index < 3,
                rejection_reason="tracking search boundary" if index >= 3 else "",
            )
            for index in range(10)
        ]
        model = fit_affine_sync_model(observations, fs=20_000)
        status, message = validate_pair(initial, observations, model, SyncOptions())
        self.assertEqual(status, "FAIL")
        self.assertIn("consecutive rejected", message)

    def test_persistent_offset_discontinuities_fail_qc(self) -> None:
        rng = np.random.default_rng(21)
        signal = rng.normal(size=2000)
        initial = estimate_lag(signal, signal, 100, peak_exclusion_samples=5)
        for step_samples in (20.0, 80.0):
            with self.subTest(step_samples=step_samples):
                observations = []
                for index in range(100):
                    offset = 0.0 if index < 80 else step_samples
                    observations.append(
                        SyncObservation(
                            center_time_sec=float(index * 5),
                            predicted_offset_samples=offset,
                            observed_offset_samples=offset,
                            residual_lag_samples=0.0,
                            peak_correlation=1.0,
                            peak_to_background=20.0,
                            peak_margin_fraction=0.5,
                            secondary_lag_samples=None,
                            accepted=True,
                        )
                    )
                model = fit_affine_sync_model(observations, fs=20_000)
                status, message = validate_pair(initial, observations, model, SyncOptions())
                self.assertEqual(status, "FAIL")
                self.assertIn("clock discontinuity", message)

    def test_windowed_sinc_preserves_high_frequency_fractional_delay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "signal.dat"
            fs = 20_000
            frequency = 7_000
            samples = np.arange(4096, dtype=float)
            source = 20_000 * np.sin(2 * np.pi * frequency * samples / fs)
            np.rint(source).astype("<i2").reshape(-1, 1).tofile(path)
            mapped = np.memmap(path, dtype="<i2", mode="r", shape=(samples.size, 1))
            try:
                positions = np.arange(64, 4032, dtype=float) + 0.37
                expected = 20_000 * np.sin(2 * np.pi * frequency * positions / fs)
                sinc = _windowed_sinc_resampled_chunk(mapped, positions)[:, 0].astype(float)
                linear = _linear_resampled_chunk(mapped, positions)[:, 0].astype(float)
                sinc_rms = float(np.sqrt(np.mean(np.square(sinc - expected))))
                linear_rms = float(np.sqrt(np.mean(np.square(linear - expected))))
                self.assertLess(sinc_rms, 0.2 * linear_rms)
                self.assertLess(sinc_rms / 20_000, 0.02)
            finally:
                close_memmap(mapped)

    def test_pc_time_is_sliced_to_exact_merged_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw_pc_time.dat"
            output = root / "pc_time.dat"
            np.arange(1000, dtype="<u4").tofile(raw)
            align_pc_time_file(raw, output, common_start_master_sample=37, n_samples=411)
            aligned = np.fromfile(output, dtype="<u4")
            np.testing.assert_array_equal(aligned, np.arange(37, 448, dtype="<u4"))

    def test_pc_time_requires_matching_merge_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.arange(10, dtype="<u4").tofile(root / "time.dat")
            np.arange(10, dtype="<u4").tofile(root / "pc_time.dat")
            (root / "wild_multilogger_mergeInfo.json").write_text(
                json.dumps(
                    {
                        "run_id": "merge-a",
                        "common_start_master_sample": 5,
                        "n_samples": 10,
                    }
                ),
                encoding="utf-8",
            )
            recording = RecordingInfo(
                use=True,
                role="master",
                probe_index=1,
                device_id="master",
                recording_name="recording",
                folder=root / "raw",
                fs=20_000,
                n_channels=64,
                n_samples=10,
                duration_sec=0.0005,
                time_dat_valid=True,
                info_rhd_exists=False,
                imu_mat_exists=False,
                pc_time_exists=True,
                pc_time_valid=False,
            )
            qc_path = root / "pc_time_qc.json"
            qc_path.write_text(
                json.dumps(
                    {
                        "merge_run_id": "merge-b",
                        "status": "ok",
                        "aligned_to_merge": True,
                        "common_start_master_sample": 5,
                        "n_samples": 10,
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(pc_time_valid_for_recording(root / "pc_time.dat", recording))
            qc_path.write_text(
                json.dumps(
                    {
                        "merge_run_id": "merge-a",
                        "status": "failed",
                        "aligned_to_merge": True,
                        "common_start_master_sample": 5,
                        "n_samples": 10,
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(pc_time_valid_for_recording(root / "pc_time.dat", recording))
            qc_path.write_text(
                json.dumps(
                    {
                        "merge_run_id": "merge-a",
                        "status": "ok",
                        "aligned_to_merge": True,
                        "common_start_master_sample": 5,
                        "n_samples": 10,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(pc_time_valid_for_recording(root / "pc_time.dat", recording))
            (root / "wild_preprocess_run.json").write_text(
                json.dumps(
                    {
                        "run_id": "stale-python-run",
                        "merge": {"common_start_master_sample": 0, "n_samples": 1},
                        "pc_time": {"status": "warning", "aligned_to_merge": False},
                    }
                ),
                encoding="utf-8",
            )
            with patch("WILD_preprocess_gui.wild_preprocess_gui.SYNC_BACKEND", "matlab"):
                self.assertTrue(pc_time_valid_for_recording(root / "pc_time.dat", recording))

    def test_pc_time_accepts_matching_single_manifest_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.arange(10, dtype="<u4").tofile(root / "time.dat")
            np.arange(10, dtype="<u4").tofile(root / "pc_time.dat")
            recording = RecordingInfo(
                use=True,
                role="master",
                probe_index=1,
                device_id="master",
                recording_name="recording",
                folder=root / "raw",
                fs=20_000,
                n_channels=64,
                n_samples=10,
                duration_sec=0.0005,
                time_dat_valid=True,
                info_rhd_exists=False,
                imu_mat_exists=False,
                pc_time_exists=True,
                pc_time_valid=False,
            )
            manifest = {
                "run_id": "run-a",
                "merge": {"common_start_master_sample": 5, "n_samples": 10},
                "pc_time": {
                    "merge_run_id": "run-a",
                    "status": "ok",
                    "aligned_to_merge": True,
                    "common_start_master_sample": 5,
                    "n_samples": 10,
                },
            }
            (root / "wild_preprocess_run.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.assertTrue(pc_time_valid_for_recording(root / "pc_time.dat", recording))
            manifest["pc_time"].update(
                {"status": "warn", "published": True, "aligned_to_merge": True}
            )
            (root / "wild_preprocess_run.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.assertTrue(pc_time_valid_for_recording(root / "pc_time.dat", recording))
            manifest["pc_time"].pop("published")
            (root / "wild_preprocess_run.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.assertFalse(pc_time_valid_for_recording(root / "pc_time.dat", recording))
            manifest["pc_time"]["published"] = False
            (root / "wild_preprocess_run.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.assertFalse(pc_time_valid_for_recording(root / "pc_time.dat", recording))
            manifest["pc_time"].update({"status": "ok", "published": True})
            manifest["pc_time"]["merge_run_id"] = "run-b"
            (root / "wild_preprocess_run.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.assertFalse(pc_time_valid_for_recording(root / "pc_time.dat", recording))

    def test_interrupted_merge_transaction_is_rolled_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            backup = output / ".wild_merge_backup_interrupted"
            staging = output / ".wild_merge_stage_interrupted"
            backup.mkdir()
            staging.mkdir()
            (backup / "amplifier.dat").write_bytes(b"old")
            (output / "amplifier.dat").write_bytes(b"new")
            (output / "time.dat").write_bytes(b"new-time")
            (backup / "transaction.json").write_text(
                json.dumps(
                    {
                        "old_names": ["amplifier.dat"],
                        "new_names": ["amplifier.dat", "time.dat"],
                    }
                ),
                encoding="utf-8",
            )
            _recover_interrupted_transactions(output)
            self.assertEqual((output / "amplifier.dat").read_bytes(), b"old")
            self.assertFalse((output / "time.dat").exists())
            self.assertFalse(backup.exists())
            self.assertFalse(staging.exists())

    def test_recovery_preserves_old_file_not_yet_backed_up(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            backup = output / ".wild_merge_backup_mid_backup"
            backup.mkdir()
            (backup / "amplifier.dat").write_bytes(b"old-amplifier")
            (output / "time.dat").write_bytes(b"old-time-still-in-place")
            (backup / "transaction.json").write_text(
                json.dumps(
                    {
                        "old_names": ["amplifier.dat", "time.dat"],
                        "new_names": ["amplifier.dat", "time.dat", "analogin.dat"],
                    }
                ),
                encoding="utf-8",
            )
            _recover_interrupted_transactions(output)
            self.assertEqual((output / "amplifier.dat").read_bytes(), b"old-amplifier")
            self.assertEqual((output / "time.dat").read_bytes(), b"old-time-still-in-place")
            self.assertFalse((output / "analogin.dat").exists())

    def test_small_pipeline_estimates_affine_clock_and_writes_merge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fs = 20_000
            sample_count = 160_000
            rng = np.random.default_rng(10)
            base = rng.normal(scale=400.0, size=sample_count)
            base += 700.0 * np.sin(2 * np.pi * 3000 * np.arange(sample_count) / fs)
            folders = []
            # This fixture is intentionally shorter than the 60 s threshold;
            # short recordings use a constant offset rather than a drift fit.
            specifications = [(0.0, 0.0), (21.0, 0.0), (-13.0, 0.0)]
            for index, (intercept, drift_ppm) in enumerate(specifications):
                folder = root / f"device{index}" / "recording"
                _write_recording(
                    folder,
                    base,
                    fs=fs,
                    n_channels=4,
                    intercept=intercept,
                    drift_ppm=drift_ppm,
                )
                folders.append(folder)
            output = root / "output"
            options = SyncOptions(
                initial_start_seconds=0.5,
                initial_duration_seconds=1,
                initial_max_lag_seconds=0.01,
                window_seconds=1,
                step_seconds=0.5,
                tracking_max_lag_samples=40,
                highpass_hz=200,
                peak_exclusion_samples=8,
                min_peak_margin_fraction=0.005,
                max_model_rms_samples=5,
                max_model_residual_samples=15,
                chunk_seconds=1,
            )
            result = run_multidevice_sync(
                folders,
                master_index=0,
                output_folder=output,
                overwrite=False,
                merge=True,
                options=options,
                write_event_files=True,
            )
            self.assertEqual(result.status, "OK")
            self.assertEqual(len(result.pairs), 2)
            self.assertAlmostEqual(result.pairs[0].model.intercept_samples, 21.0, delta=2.0)
            self.assertTrue(result.pairs[0].model.is_constant_offset)
            self.assertAlmostEqual(result.pairs[0].model.drift_ppm, 0.0, delta=1e-9)
            self.assertAlmostEqual(result.pairs[1].model.intercept_samples, -13.0, delta=2.0)
            self.assertTrue(result.pairs[1].model.is_constant_offset)
            self.assertAlmostEqual(result.pairs[1].model.drift_ppm, 0.0, delta=1e-9)
            self.assertTrue((output / "amplifier.dat").is_file())
            self.assertTrue((output / "analogin.dat").is_file())
            manifest = json.loads(
                (output / "wild_preprocess_run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest["sync"]["pairs"]), 2)
            self.assertEqual(manifest["run_id"], manifest["merge"]["run_id"])
            self.assertIn("postmerge_validation", manifest["merge"])
            self.assertEqual(len(manifest["merge"]["channel_layout"]), 12)
            sync_figures = list(output.glob("wild_multilogger_sync_master_vs_*_qc.png"))
            self.assertEqual(len(sync_figures), 2)
            self.assertTrue(all(path.stat().st_size > 10_000 for path in sync_figures))
            self.assertTrue(
                all(
                    Path(pair["figure_file"]).parent == output
                    for pair in manifest["sync"]["pairs"]
                )
            )
            self.assertFalse((output / "wild_multilogger_sync_qc.mat").exists())
            self.assertFalse((output / "wild_multilogger_mergeInfo.json").exists())
            event_path = output / "device_event.dev01.d01.evt"
            first_event_ms = float(event_path.read_text(encoding="utf-8").splitlines()[0].split("\t")[0])
            self.assertEqual(
                manifest["merge"]["common_interval_limits"]["start_limiter"]["stream"],
                "validated_endpoint_probe",
            )
            # The protected one-second prefix changes the phase of this
            # synthetic 80-ms square wave relative to the saved interval.
            self.assertAlmostEqual(first_event_ms, 39.2, delta=0.8)
            merged = np.memmap(output / "amplifier.dat", dtype="<i2", mode="r")
            self.assertEqual(merged.size % 12, 0)
            close_memmap(merged)

    def test_ambiguous_initial_periodic_peak_prevents_merge_publication(self) -> None:
        """A periodic initial xcorr must not publish a potentially misaligned DAT."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fs = 20_000
            samples = 60_000
            positions = np.arange(samples, dtype=float)
            periodic = 1_000.0 * np.sin(2 * np.pi * positions / 64.0)
            folders = []
            for index in range(2):
                folder = root / f"device{index}" / "recording"
                _write_recording(
                    folder,
                    periodic,
                    fs=fs,
                    n_channels=4,
                    intercept=0.0,
                    drift_ppm=0.0,
                )
                folders.append(folder)
            output = root / "output"
            options = SyncOptions(
                initial_start_seconds=0.0,
                initial_duration_seconds=1.0,
                initial_max_lag_seconds=0.02,
                window_seconds=1.0,
                step_seconds=0.5,
                tracking_max_lag_samples=40,
                highpass_hz=200.0,
                peak_exclusion_samples=20,
                chunk_seconds=1.0,
            )
            result = run_multidevice_sync(
                folders,
                master_index=0,
                output_folder=output,
                merge=True,
                options=options,
            )

            self.assertEqual(result.status, "WARN")
            self.assertIn("initial peak margin", result.pairs[0].message)
            self.assertTrue((output / "amplifier.dat").exists())
            validity = np.fromfile(output / "valid_samples.dat", dtype=np.uint8).reshape(-1, 2)
            self.assertTrue(np.all(validity[:, 0] == 1))
            self.assertTrue(np.all(validity[:, 1] == 0))

    def test_failed_overwrite_keeps_previous_canonical_outputs(self) -> None:
        """A failed sync attempt must be quarantined instead of replacing run A."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fs = 20_000
            samples = 100_000
            rng = np.random.default_rng(91)
            base = rng.normal(scale=300.0, size=samples)
            folders = []
            for index, offset in enumerate((0.0, 17.0)):
                folder = root / f"device{index}" / "recording"
                _write_recording(
                    folder,
                    base,
                    fs=fs,
                    n_channels=4,
                    intercept=offset,
                    drift_ppm=0.0,
                )
                folders.append(folder)
            output = root / "output"
            options = SyncOptions(
                initial_start_seconds=0.5,
                initial_duration_seconds=1.0,
                initial_max_lag_seconds=0.01,
                window_seconds=1.0,
                step_seconds=0.5,
                tracking_max_lag_samples=40,
                highpass_hz=200.0,
                peak_exclusion_samples=8,
                min_peak_margin_fraction=0.005,
                chunk_seconds=1.0,
            )
            successful = run_multidevice_sync(
                folders,
                master_index=0,
                output_folder=output,
                merge=True,
                options=options,
            )
            self.assertEqual(successful.status, "OK")
            amplifier_before = (output / "amplifier.dat").read_bytes()
            manifest_before = (output / "wild_preprocess_run.json").read_text(encoding="utf-8")

            failed_options = SyncOptions(**{**options.__dict__, "min_peak_correlation": 1.1})
            failed = run_multidevice_sync(
                folders,
                master_index=0,
                output_folder=output,
                overwrite=True,
                merge=True,
                options=failed_options,
            )
            self.assertEqual(failed.status, "WARN")
            self.assertNotEqual((output / "amplifier.dat").read_bytes(), amplifier_before)
            manifest_after = json.loads(
                (output / "wild_preprocess_run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest_after["overall_status"], "MERGE_ONLY")
            validity = np.fromfile(output / "valid_samples.dat", dtype=np.uint8).reshape(-1, 2)
            self.assertTrue(np.all(validity[:, 0] == 1))
            self.assertTrue(np.all(validity[:, 1] == 0))


if __name__ == "__main__":
    unittest.main()
