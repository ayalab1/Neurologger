from __future__ import annotations

import sys
import struct
import tempfile
import unittest
import io
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from wild_preprocess.pc_time.decode import (
    CE64_RAW_MISC_LAYOUT,
    PACKED_PC_MOD_MS,
    collect_packed_updates,
    decode_ce_params_recording_start_ms,
    read_ce_params_hint,
    resolve_recording_start_ms,
)
from WILD_generate_pc_time import (
    decode_ce_params_recording_start_ms as decode_standalone_ce_params_recording_start_ms,
    main as standalone_main,
)
from wild_preprocess.pc_time.infer import PcTimeModel, fit_robust_pc_time_model
from wild_preprocess.pc_time.validate import validate_pc_time_interval
from wild_preprocess.pc_time.write import write_interval_pc_time
from wild_preprocess.pc_time.report import pc_time_qc_payload, write_pc_time_qc_json, write_pc_time_summary_png


def _packed(target_ms: np.ndarray, delay_ms: int = 11) -> np.ndarray:
    raw = (np.asarray(target_ms, dtype=np.int64) - delay_ms) % PACKED_PC_MOD_MS
    return (raw | (np.int64(delay_ms) << 20)).astype(np.uint32)


class NativePcTimeTest(unittest.TestCase):
    @staticmethod
    def _ce_params_with_rtc(
        *,
        year: int = 26,
        month: int = 8,
        day: int = 11,
        hours: int = 18,
        minutes: int = 13,
        seconds: int = 23,
        sub_seconds: int = 1_839,
        second_fraction: int = 9_999,
    ) -> bytes:
        data = bytearray(512)
        struct.pack_into("<I", data, 0, 20_000)
        struct.pack_into("<I", data, 40, 20_000)
        struct.pack_into("<I", data, 52, 1_250)
        struct.pack_into("<BBBB", data, 332, 2, month, day, year)
        struct.pack_into("<BBBB", data, 336, hours, minutes, seconds, 0)
        struct.pack_into("<IIII", data, 340, sub_seconds, second_fraction, 0, 0)
        data[356:364] = bytes.fromhex("C9E7EDC0B2E60000")
        return bytes(data)

    def _model(self, *, duration_sec: int = 120) -> tuple[PcTimeModel, np.ndarray, np.ndarray]:
        fs = 20_000.0
        indices = np.arange(0, duration_sec * fs, int(5 * fs), dtype=np.int64)
        device_ms = indices * 1000.0 / fs
        target = np.rint(4_000_000.0 + 1.0 * device_ms).astype(np.int64)
        model = fit_robust_pc_time_model(indices, _packed(target), fs, 4_000_000)
        return model, indices, target

    def _fit_and_validate(self, target_ms: np.ndarray) -> tuple[PcTimeModel, object]:
        fs = 20_000.0
        indices = np.arange(target_ms.size, dtype=np.int64) * int(5 * fs)
        model = fit_robust_pc_time_model(indices, _packed(target_ms), fs, 4_000_000)
        validation = validate_pc_time_interval(
            model,
            sample_rate_hz=fs,
            common_start_master_sample=0,
            n_samples=int((indices[-1] + 1)),
        )
        return model, validation

    def test_ce64_raw_misc_decodes_updates_and_ephys_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "analogin.dat"
            cycles = np.zeros((16, 16), dtype="<u2")
            values = _packed(np.array([10_000, 10_500, 10_000, 11_000, 11_000]))
            for index, value in zip((2, 5, 6, 9, 10), values):
                cycles[index, 14] = value & 0xFFFF
                cycles[index, 15] = value >> 16
            cycles[3] = cycles[2]
            cycles[7] = cycles[6]
            raw = np.zeros(256, dtype="<u2")
            raw[: cycles.size] = cycles.ravel()
            raw.tofile(path)
            indices, packed, diagnostics = collect_packed_updates(
                path,
                CE64_RAW_MISC_LAYOUT,
                return_diagnostics=True,
            )
            np.testing.assert_array_equal(indices, [32, 144])
            np.testing.assert_array_equal(packed, [values[0], values[3]])
            self.assertEqual(diagnostics.raw_candidate_run_count, 4)
            self.assertEqual(diagnostics.accepted_update_count, 2)
            self.assertEqual(diagnostics.rejected_unstable_run_count, 1)

    def test_robust_fit_lifts_modulo_cycle_and_rejects_corrupt_tail(self) -> None:
        fs = 20_000.0
        good_indices = np.arange(0, 130 * fs, 5 * fs, dtype=np.int64)
        bad_indices = np.arange(200 * fs, 300 * fs, 5 * fs, dtype=np.int64)
        good = _packed(np.rint(1_040_000.0 + good_indices * 1000.0 / fs).astype(np.int64))
        bad = _packed(np.full(bad_indices.size, 321_000, dtype=np.int64))
        model = fit_robust_pc_time_model(np.r_[good_indices, bad_indices], np.r_[good, bad], fs, 1_040_000)
        self.assertGreaterEqual(model.kept_count, good_indices.size)
        self.assertAlmostEqual(model.drift_ppm, 0.0, delta=0.1)
        validation = validate_pc_time_interval(
            model,
            sample_rate_hz=fs,
            common_start_master_sample=0,
            n_samples=120 * int(fs),
        )
        self.assertEqual(validation.status, "OK")

    def test_missing_anchor_is_not_replaced_with_midnight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "No absolute"):
                resolve_recording_start_ms(Path(temporary))

    def test_standalone_missing_anchor_returns_clean_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "analogin.dat").write_bytes(b"")
            stderr = io.StringIO()
            with patch.object(
                sys,
                "argv",
                ["WILD_generate_pc_time.py", str(folder), "--sample-rate", "20000"],
            ), redirect_stderr(stderr):
                code = standalone_main()
            self.assertEqual(code, 1)
            self.assertIn("recording start time not found", stderr.getvalue())

    def test_ce_rtc_uses_system_header_offsets_not_mac_bytes(self) -> None:
        data = self._ce_params_with_rtc()
        expected = ((18 * 3600 + 13 * 60 + 23) * 1000) + 816
        self.assertEqual(decode_ce_params_recording_start_ms(data), expected)
        self.assertEqual(decode_standalone_ce_params_recording_start_ms(data), expected)

    def test_ce_rtc_rejects_impossible_calendar_date(self) -> None:
        data = self._ce_params_with_rtc(month=2, day=30)
        self.assertIsNone(decode_ce_params_recording_start_ms(data))
        self.assertIsNone(decode_standalone_ce_params_recording_start_ms(data))

    def test_ce_rtc_rejects_year_outside_stm32_domain(self) -> None:
        data = self._ce_params_with_rtc(year=100)
        self.assertIsNone(decode_ce_params_recording_start_ms(data))
        self.assertIsNone(decode_standalone_ce_params_recording_start_ms(data))

    def test_ce_hint_carries_date_and_source_wins_over_folder_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "renamed_without_timestamp"
            folder.mkdir()
            (folder / "CE_params.bin").write_bytes(self._ce_params_with_rtc())
            hint = read_ce_params_hint(folder)
            self.assertEqual(hint.recording_date, "2026-08-11")
            self.assertEqual(hint.recording_start_ms, 65_603_816)
            self.assertEqual(resolve_recording_start_ms(folder), (65_603_816, "CE_params.bin"))
            self.assertEqual(
                resolve_recording_start_ms(folder, explicit_recording_start_ms=123),
                (123, "explicit"),
            )

    def test_folder_name_anchor_requires_explicit_compatibility_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "1_20260811_181323.429"
            folder.mkdir()
            with self.assertRaisesRegex(ValueError, "folder fallback is disabled"):
                resolve_recording_start_ms(folder)
            self.assertEqual(
                resolve_recording_start_ms(folder, allow_folder_name_fallback=True),
                (65_603_429, "recording folder name (explicit fallback)"),
            )

    def test_explicit_folder_fallback_reports_unsupported_nine_digit_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "1_20260811_181323.429496281"
            folder.mkdir()
            with self.assertRaisesRegex(ValueError, "not a supported HHMMSS"):
                resolve_recording_start_ms(folder, allow_folder_name_fallback=True)

    def test_too_few_merged_interval_anchors_warn(self) -> None:
        model, indices, _target = self._model(duration_sec=11)
        validation = validate_pc_time_interval(
            model,
            sample_rate_hz=20_000,
            common_start_master_sample=0,
            n_samples=11 * 20_000,
        )
        self.assertEqual(validation.status, "WARN")
        self.assertIn("retained anchors", validation.message)
        self.assertTrue(validation.publishable)

    def test_one_anchor_is_not_publishable(self) -> None:
        fs = 20_000.0
        model = fit_robust_pc_time_model(
            np.array([0], dtype=np.int64),
            _packed(np.array([4_000_000], dtype=np.int64)),
            fs,
            4_000_000,
        )
        validation = validate_pc_time_interval(
            model,
            sample_rate_hz=fs,
            common_start_master_sample=0,
            n_samples=int(fs * 600),
        )
        self.assertEqual(validation.status, "WARN")
        self.assertFalse(validation.publishable)
        self.assertIn("time-separated", "; ".join(validation.publication_blockers))

    def test_large_internal_anchor_gap_warns_but_remains_publishable(self) -> None:
        fs = 20_000.0
        seconds = np.r_[np.arange(0, 50, 5), np.arange(350, 405, 5)]
        indices = (seconds * fs).astype(np.int64)
        target = 4_000_000 + seconds.astype(np.int64) * 1_000
        model = fit_robust_pc_time_model(indices, _packed(target), fs, 4_000_000)
        validation = validate_pc_time_interval(
            model,
            sample_rate_hz=fs,
            common_start_master_sample=0,
            n_samples=int(fs * 405),
        )
        self.assertEqual(validation.status, "WARN")
        self.assertGreater(validation.max_internal_gap_sec, 120.0)
        self.assertTrue(validation.publishable)

    def test_insufficient_interval_coverage_blocks_best_effort_publication(self) -> None:
        model, _indices, _target = self._model(duration_sec=15)
        validation = validate_pc_time_interval(
            model,
            sample_rate_hz=20_000,
            common_start_master_sample=0,
            n_samples=60 * 20_000,
        )
        self.assertEqual(validation.status, "WARN")
        self.assertFalse(validation.publishable)
        self.assertTrue(
            any("coverage" in blocker for blocker in validation.publication_blockers)
        )

    def test_single_outdoorsmall_like_rate_trigger_warns_and_is_publishable(self) -> None:
        device_ms = np.arange(645, dtype=float) * 5_000.0
        residuals = np.zeros(645, dtype=float)
        residuals[237:287] = np.array(
            [
                8.394, -4.553, -10.899, 7.554, -3.392, 99.660, -14.285, -4.232,
                -4.778, -9.324, 8.329, -6.217, -2.564, 11.490, -1.057, -5.003,
                -3.549, -2.896, -3.843, 4.611, -3.735, -1.282, -5.228, -3.775,
                -3.721, 7.932, 95.385, 93.038, 74.492, -26.454, -17.400, 81.653,
                83.706, 67.160, 57.613, 64.667, -34.279, -7.625, -9.172, -5.518,
                8.335, -3.011, -4.957, 5.496, -0.850, -1.797, 7.657, 8.110,
                -15.236, -5.183,
            ],
            dtype=float,
        )
        model = PcTimeModel(
            device_ms=device_ms,
            pc_unwrapped_ms=4_000_000.0 + device_ms + residuals,
            delay_ms=np.zeros(645, dtype=float),
            residual_ms=residuals,
            keep_mask=np.ones(645, dtype=bool),
            slope=1.0,
            intercept_ms=4_000_000.0,
            slope_sem=0.0,
            intercept_sem_ms=0.0,
            recording_start_ms=4_000_000,
        )
        validation = validate_pc_time_interval(
            model,
            sample_rate_hz=20_000.0,
            common_start_master_sample=0,
            n_samples=int(device_ms[-1] * 20.0) + 1,
        )
        self.assertEqual(validation.status, "WARN")
        self.assertEqual(validation.rate_change_trigger_count, 1)
        self.assertAlmostEqual(validation.maximum_local_rate_difference_ppm, 1_017.77, places=1)
        self.assertTrue(validation.publishable)

    def test_persistent_clock_step_inside_interval_warns(self) -> None:
        model, _indices, _target = self._model()
        residuals = model.residual_ms.copy()
        residuals[12:] += 400.0
        stepped = PcTimeModel(
            model.device_ms,
            model.pc_unwrapped_ms,
            model.delay_ms,
            residuals,
            model.keep_mask,
            model.slope,
            model.intercept_ms,
            model.slope_sem,
            model.intercept_sem_ms,
            model.recording_start_ms,
        )
        validation = validate_pc_time_interval(
            stepped,
            sample_rate_hz=20_000,
            common_start_master_sample=0,
            n_samples=120 * 20_000,
        )
        self.assertEqual(validation.status, "WARN")
        self.assertTrue(validation.persistent_step_detected)
        self.assertTrue(validation.publishable)
        self.assertFalse(validation.publication_blockers)

    def test_early_persistent_clock_step_warns_end_to_end(self) -> None:
        device_ms = np.arange(24, dtype=np.int64) * 5_000
        target = 4_000_000 + device_ms
        target[4:] += 400
        _model, validation = self._fit_and_validate(target)
        self.assertEqual(validation.status, "WARN")
        self.assertTrue(validation.persistent_step_detected)
        self.assertTrue(validation.publishable)

    def test_late_persistent_clock_step_warns_end_to_end(self) -> None:
        device_ms = np.arange(24, dtype=np.int64) * 5_000
        target = 4_000_000 + device_ms
        target[20:] += 400
        _model, validation = self._fit_and_validate(target)
        self.assertEqual(validation.status, "WARN")
        self.assertTrue(validation.persistent_step_detected)

    def test_persistent_rate_change_inside_interval_warns(self) -> None:
        device_ms = np.arange(60, dtype=np.int64) * 5_000
        target = 4_000_000 + device_ms
        boundary = 30
        target[boundary:] += np.rint((device_ms[boundary:] - device_ms[boundary]) * 0.005).astype(np.int64)
        _model, validation = self._fit_and_validate(target)
        self.assertEqual(validation.status, "WARN")
        self.assertTrue(validation.persistent_rate_change_detected)
        self.assertGreaterEqual(validation.maximum_local_rate_difference_ppm, 1_000.0)
        self.assertGreater(validation.rate_change_trigger_count, 0)
        self.assertIsNotNone(validation.first_rate_change_trigger_time_sec)
        self.assertTrue(validation.publishable)

    def test_isolated_corrupt_update_is_tolerated_by_ordered_diagnostics(self) -> None:
        device_ms = np.arange(24, dtype=np.int64) * 5_000
        target = 4_000_000 + device_ms
        target[12] += 2_000
        model, validation = self._fit_and_validate(target)
        self.assertLess(model.kept_count, target.size)
        self.assertEqual(validation.status, "OK")
        self.assertFalse(validation.persistent_step_detected)
        self.assertFalse(validation.persistent_rate_change_detected)

    def test_stable_clock_with_packed_delay_jitter_is_not_a_rate_change(self) -> None:
        device_ms = np.arange(120, dtype=np.int64) * 5_000
        jitter_pattern = np.array([0, 60, -15, 55, 45, -20, 50, -10, 40, -5], dtype=np.int64)
        target = 4_000_000 + device_ms + np.resize(jitter_pattern, device_ms.size)
        _model, validation = self._fit_and_validate(target)
        self.assertEqual(validation.status, "OK")
        self.assertFalse(validation.persistent_rate_change_detected)

    def test_direct_writer_has_exact_interval_length_and_midnight_wrap(self) -> None:
        fs = 20_000.0
        indices = np.arange(0, 100 * fs, 5 * fs, dtype=np.int64)
        start_ms = 86_399_950
        model = fit_robust_pc_time_model(
            indices,
            _packed(np.rint(start_ms + indices * 1000.0 / fs).astype(np.int64)),
            fs,
            start_ms,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pc_time.dat"
            progress: list[float] = []
            write_interval_pc_time(
                path,
                model,
                sample_rate_hz=fs,
                common_start_master_sample=0,
                n_samples=2_000,
                chunk_samples=500,
                progress=progress.append,
            )
            values = np.fromfile(path, dtype="<u4")
            self.assertEqual(path.stat().st_size, 2_000 * 4)
            self.assertEqual(values[0], start_ms)
            self.assertLess(values[1_500], values[0])
            self.assertEqual(progress[0], 0.0)
            self.assertEqual(progress[-1], 100.0)
            self.assertTrue(all(left <= right for left, right in zip(progress, progress[1:])))

    def test_qc_report_carries_merge_interval_and_writes_summary(self) -> None:
        model, _indices, _target = self._model()
        validation = validate_pc_time_interval(
            model,
            sample_rate_hz=20_000,
            common_start_master_sample=0,
            n_samples=120 * 20_000,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = pc_time_qc_payload(
                model,
                validation,
                run_id="test-run",
                common_start_master_sample=0,
                n_samples=120 * 20_000,
                sample_rate_hz=20_000,
                anchor_source="explicit",
                layout_name="ce64-raw-misc",
            )
            qc_path = write_pc_time_qc_json(root / "pc_time_qc.json", payload)
            summary_path = write_pc_time_summary_png(
                root / "pc_time_fit_summary.png",
                model,
                validation,
                common_start_master_sample=0,
                n_samples=120 * 20_000,
                sample_rate_hz=20_000,
            )
            self.assertIn('"run_id": "test-run"', qc_path.read_text(encoding="utf-8"))
            self.assertGreater(summary_path.stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()
