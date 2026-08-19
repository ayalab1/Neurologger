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

from wild_preprocess.binary_io import recording_from_folder
from wild_preprocess.models import SyncModel, SyncOptions, SyncPairResult
from wild_preprocess.pc_time.canonical import CanonicalPcTimeFit
from wild_preprocess.pc_time.decode import PackedUpdateDiagnostics
from wild_preprocess.pc_time.infer import PcTimeModel
from wild_preprocess.pc_time.validate import PcTimeValidation
from wild_preprocess.pipeline import _manifest_warnings, run_multidevice_sync
from wild_preprocess.sync.merge import (
    MAX_TIME_DAT_SAMPLES,
    merge_recordings,
    validate_time_dat_length,
)
import wild_preprocess.sync.merge as merge_module
from wild_preprocess.sync.postmerge import PostMergeMeasurement, PostMergeValidationResult
from wild_preprocess.worker import _validated_job
from wild_preprocess.version import SYNC_ALGORITHM_VERSION


def _ce_header(path: Path, fs: int = 1250, channels: int = 4) -> None:
    data = bytearray(512)
    struct.pack_into("<I", data, 0, fs)
    struct.pack_into("<I", data, 8, channels)
    path.write_bytes(data)


def _set_ce_rtc(
    path: Path,
    *,
    hours: int,
    minutes: int,
    seconds: int,
    month: int = 8,
    day: int = 9,
    year: int = 26,
) -> None:
    data = bytearray(path.read_bytes())
    struct.pack_into("<BBBB", data, 332, 7, month, day, year)
    struct.pack_into("<BBBB", data, 336, hours, minutes, seconds, 0)
    struct.pack_into("<IIII", data, 340, 9_999, 9_999, 0, 0)
    path.write_bytes(data)


def _recording(folder: Path, *, samples: int = 96) -> Path:
    folder.mkdir(parents=True)
    signal = np.arange(samples, dtype=np.int16)
    np.column_stack([signal + channel for channel in range(4)]).astype("<i2").tofile(folder / "amplifier.dat")
    analog = np.zeros((samples, 1), dtype="<i2")
    analog[::10, 0] = 1
    analog.tofile(folder / "analogin.dat")
    _ce_header(folder / "CE_params.bin")
    return folder


def _model() -> SyncModel:
    return SyncModel(0.0, 0.0, 0.0, 0.0, 0.0, 3, 3, True)


def _sync_recording(folder: Path, base: np.ndarray, *, offset: int) -> Path:
    folder.mkdir(parents=True)
    source = np.zeros_like(base)
    if offset >= 0:
        source[offset:] = base[: base.size - offset]
    else:
        source[:offset] = base[-offset:]
    np.column_stack([source + channel * 5 for channel in range(4)]).astype("<i2").tofile(folder / "amplifier.dat")
    analog = np.zeros((base.size // 16, 1), dtype="<i2")
    analog[::50] = 1
    analog.tofile(folder / "analogin.dat")
    _ce_header(folder / "CE_params.bin", fs=20_000, channels=4)
    return folder


class IntegrationHardeningTest(unittest.TestCase):
    def test_manifest_warnings_exclude_ordinary_ok_pair_message(self) -> None:
        ok_pair = SyncPairResult(
            1,
            2,
            "master",
            "slave",
            0.0,
            10.0,
            0.5,
            _model(),
            status="OK",
            message="affine clock model accepted",
        )
        warning_pair = SyncPairResult(
            1,
            3,
            "master",
            "slave-warning",
            0.0,
            10.0,
            0.5,
            _model(),
            status="WARN",
            message="small drift reported",
        )
        self.assertEqual(
            _manifest_warnings([ok_pair, warning_pair]),
            [{"slave_index": 3, "status": "WARN", "message": "small drift reported"}],
        )

    def test_quality_warning_publishes_fitted_pc_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(30)
            base = np.rint(rng.normal(scale=300, size=40_000)).astype(np.int16)
            folders = [
                _sync_recording(root / "one" / "recording", base, offset=0),
                _sync_recording(root / "two" / "recording", base, offset=12),
            ]
            options = SyncOptions(
                initial_start_seconds=0.1,
                initial_duration_seconds=0.5,
                initial_max_lag_seconds=0.01,
                window_seconds=0.5,
                step_seconds=0.25,
                tracking_max_lag_samples=30,
                chunk_seconds=1.0,
            )
            device_ms = np.arange(9, dtype=float) * 500.0
            model = PcTimeModel(
                device_ms=device_ms,
                pc_unwrapped_ms=43_200_000.0 + device_ms,
                delay_ms=np.zeros(device_ms.size),
                residual_ms=np.zeros(device_ms.size),
                keep_mask=np.ones(device_ms.size, dtype=bool),
                slope=1.0,
                intercept_ms=43_200_000.0,
                slope_sem=0.0,
                intercept_sem_ms=0.0,
                recording_start_ms=43_200_000,
            )
            raw_indices = np.arange(device_ms.size, dtype=np.int64) * 10_000
            fit = CanonicalPcTimeFit(raw_indices, raw_indices.copy(), model)
            validation = PcTimeValidation(
                status="WARN",
                message="temporary PC-clock rate-regime candidate",
                retained_anchor_count=device_ms.size,
                retained_span_sec=4.0,
                coverage_fraction=1.0,
                leading_extrapolation_sec=0.0,
                trailing_extrapolation_sec=0.0,
                max_internal_gap_sec=0.5,
                residual_rms_ms=0.0,
                drift_ppm=0.0,
                persistent_step_detected=False,
                persistent_rate_change_detected=True,
                maximum_local_rate_difference_ppm=1_017.0,
                rate_change_trigger_count=1,
                first_rate_change_trigger_time_sec=1.5,
                publishable=True,
            )
            forced_success = PostMergeValidationResult(
                "OK", "forced staged success", "", 1, 1, 8, 8, 4, 0.05, 0.0, ()
            )
            progress_events: list[tuple[str, float]] = []
            anchors = [
                {
                    "milliseconds_since_midnight": 43_200_000 + index * 10,
                    "source": "validated test anchor",
                    "recording_date": "2026-08-09",
                    "recording_date_source": "test",
                }
                for index in range(2)
            ]

            with (
                patch(
                    "wild_preprocess.pipeline.validate_segment_staged_merge",
                    return_value=forced_success,
                ),
                patch(
                    "wild_preprocess.pipeline.collect_packed_updates",
                    return_value=(
                        raw_indices,
                        np.ones(raw_indices.size, dtype=np.uint32),
                        PackedUpdateDiagnostics(raw_indices.size, raw_indices.size, 0, 2),
                    ),
                ),
                patch(
                    "wild_preprocess.pipeline.resolve_recording_start_ms",
                    side_effect=AssertionError("validated anchor must not be re-resolved"),
                ),
                patch("wild_preprocess.pipeline.fit_gap_aware_pc_time_model", return_value=fit),
                patch("wild_preprocess.pipeline.validate_canonical_pc_time_interval", return_value=validation),
            ):
                result = run_multidevice_sync(
                    folders,
                    master_index=0,
                    output_folder=root / "output",
                    merge=True,
                    native_pc_time=True,
                    validate_postmerge=True,
                    options=options,
                    progress=lambda stage, percent: progress_events.append((stage, percent)),
                    recording_start_anchors=anchors,
                )

            output = root / "output"
            self.assertEqual(result.outputs["pc_time_status"], "WARN")
            self.assertEqual(result.outputs["overall_status"], "COMPLETE")
            self.assertTrue((output / "pc_time.dat").is_file())
            manifest = json.loads((output / "wild_preprocess_run.json").read_text(encoding="utf-8"))
            values = np.fromfile(output / "pc_time.dat", dtype="<u4")
            first_position = int(manifest["merge"]["common_start_master_sample"])
            expected_first = int(round(43_200_000.0 + first_position * 1000.0 / 20_000.0))
            expected_last = int(
                round(43_200_000.0 + (first_position + values.size - 1) * 1000.0 / 20_000.0)
            )
            self.assertEqual(int(values[0]), expected_first)
            self.assertEqual(int(values[-1]), expected_last)
            self.assertTrue(np.all(np.diff(values.astype(np.int64)) >= 0))
            self.assertEqual(manifest["pc_time"]["status"], "warn")
            self.assertEqual(manifest["pc_time"]["anchor_source"], "validated test anchor")
            self.assertTrue(manifest["pc_time"]["published"])
            self.assertEqual(
                manifest["pc_time"]["publication_mode"],
                "robust_affine_best_effort_warn",
            )
            self.assertTrue(manifest["pc_time"]["aligned_to_merge"])
            self.assertIn("pc_time.dat", manifest["managed_files"])
            self.assertEqual(
                manifest["expected_output_bytes"]["pc_time.dat"],
                (output / "pc_time.dat").stat().st_size,
            )
            pc_warning = next(
                item for item in manifest["warnings"] if item.get("component") == "pc_time"
            )
            self.assertTrue(pc_warning["published"])
            self.assertIn("rate-regime", pc_warning["message"])
            first_seen_stages = list(dict.fromkeys(stage for stage, _percent in progress_events))
            self.assertEqual(
                first_seen_stages,
                [
                    "inspect_inputs",
                    "build_features",
                    "sync_pairs",
                    "refine_sync",
                    "integrity_scan",
                    "write_ephys",
                    "write_analog",
                    "postmerge_qc",
                    "pc_time",
                    "inspection",
                    "publish",
                ],
            )
            by_stage: dict[str, list[float]] = {}
            for stage, percent in progress_events:
                by_stage.setdefault(stage, []).append(percent)
            for values_for_stage in by_stage.values():
                self.assertGreaterEqual(values_for_stage[0], 0.0)
                self.assertEqual(values_for_stage[-1], 100.0)

    def test_partial_pc_time_is_not_published_on_writer_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(31)
            base = np.rint(rng.normal(scale=300, size=40_000)).astype(np.int16)
            folders = [
                _sync_recording(root / "one" / "recording", base, offset=0),
                _sync_recording(root / "two" / "recording", base, offset=12),
            ]
            options = SyncOptions(
                initial_start_seconds=0.1,
                initial_duration_seconds=0.5,
                initial_max_lag_seconds=0.01,
                window_seconds=0.5,
                step_seconds=0.25,
                tracking_max_lag_samples=30,
                chunk_seconds=1.0,
            )
            device_ms = np.arange(9, dtype=float) * 500.0
            model = PcTimeModel(
                device_ms=device_ms,
                pc_unwrapped_ms=43_200_000.0 + device_ms,
                delay_ms=np.zeros(device_ms.size),
                residual_ms=np.zeros(device_ms.size),
                keep_mask=np.ones(device_ms.size, dtype=bool),
                slope=1.0,
                intercept_ms=43_200_000.0,
                slope_sem=0.0,
                intercept_sem_ms=0.0,
                recording_start_ms=43_200_000,
            )
            raw_indices = np.arange(device_ms.size, dtype=np.int64) * 10_000
            fit = CanonicalPcTimeFit(raw_indices, raw_indices.copy(), model)
            validation = PcTimeValidation(
                status="OK",
                message="",
                retained_anchor_count=device_ms.size,
                retained_span_sec=4.0,
                coverage_fraction=1.0,
                leading_extrapolation_sec=0.0,
                trailing_extrapolation_sec=0.0,
                max_internal_gap_sec=0.5,
                residual_rms_ms=0.0,
                drift_ppm=0.0,
                persistent_step_detected=False,
                persistent_rate_change_detected=False,
            )
            forced_success = PostMergeValidationResult(
                "OK", "forced staged success", "", 1, 1, 8, 8, 4, 0.05, 0.0, ()
            )

            def partial_writer(path: Path, *_args: object, **_kwargs: object) -> Path:
                Path(path).write_bytes(b"partial")
                raise OSError("injected PC-time writer failure")

            with (
                patch(
                    "wild_preprocess.pipeline.validate_segment_staged_merge",
                    return_value=forced_success,
                ),
                patch(
                    "wild_preprocess.pipeline.collect_packed_updates",
                    return_value=(
                        raw_indices,
                        np.ones(raw_indices.size, dtype=np.uint32),
                        PackedUpdateDiagnostics(raw_indices.size, raw_indices.size, 0, 2),
                    ),
                ),
                patch("wild_preprocess.pipeline.resolve_recording_start_ms", return_value=(43_200_000, "test")),
                patch("wild_preprocess.pipeline.fit_gap_aware_pc_time_model", return_value=fit),
                patch("wild_preprocess.pipeline.validate_canonical_pc_time_interval", return_value=validation),
                patch("wild_preprocess.pipeline.write_canonical_interval_pc_time", side_effect=partial_writer),
            ):
                result = run_multidevice_sync(
                    folders,
                    master_index=0,
                    output_folder=root / "output",
                    merge=True,
                    native_pc_time=True,
                    validate_postmerge=True,
                    options=options,
                )

            output = root / "output"
            self.assertEqual(result.outputs["pc_time_status"], "WARN")
            self.assertEqual(result.outputs["overall_status"], "MERGE_ONLY")
            self.assertTrue((output / "amplifier.dat").is_file())
            self.assertFalse((output / "pc_time.dat").exists())
            self.assertTrue((output / "pc_time_fit_summary.png").is_file())
            manifest = json.loads((output / "wild_preprocess_run.json").read_text(encoding="utf-8"))
            self.assertNotIn("pc_time.dat", manifest["managed_files"])
            self.assertIn("pc_time_fit_summary.png", manifest["managed_files"])

    def test_pc_time_decode_warning_retains_diagnostic_figure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(32)
            base = np.rint(rng.normal(scale=300, size=40_000)).astype(np.int16)
            folders = [
                _sync_recording(root / "one" / "recording", base, offset=0),
                _sync_recording(root / "two" / "recording", base, offset=12),
            ]
            options = SyncOptions(
                initial_start_seconds=0.1,
                initial_duration_seconds=0.5,
                initial_max_lag_seconds=0.01,
                window_seconds=0.5,
                step_seconds=0.25,
                tracking_max_lag_samples=30,
                chunk_seconds=1.0,
            )
            forced_success = PostMergeValidationResult(
                "OK", "forced staged success", "", 1, 1, 8, 8, 4, 0.05, 0.0, ()
            )
            with (
                patch(
                    "wild_preprocess.pipeline.validate_segment_staged_merge",
                    return_value=forced_success,
                ),
                patch(
                    "wild_preprocess.pipeline.collect_packed_updates",
                    side_effect=ValueError("packed PC-time evidence is empty"),
                ),
            ):
                result = run_multidevice_sync(
                    folders,
                    master_index=0,
                    output_folder=root / "output",
                    merge=True,
                    native_pc_time=True,
                    validate_postmerge=True,
                    options=options,
                )

            output = root / "output"
            summary = output / "pc_time_fit_summary.png"
            self.assertEqual(result.outputs["pc_time_status"], "WARN")
            self.assertEqual(result.outputs["overall_status"], "MERGE_ONLY")
            self.assertFalse((output / "pc_time.dat").exists())
            self.assertTrue(summary.is_file())
            self.assertGreater(summary.stat().st_size, 5_000)
            manifest = json.loads((output / "wild_preprocess_run.json").read_text(encoding="utf-8"))
            self.assertIn("packed PC-time evidence is empty", manifest["pc_time"]["error"])
            self.assertIn(summary.name, manifest["managed_files"])

    def test_constant_slave_is_published_as_all_invalid_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(33)
            master = np.rint(rng.normal(scale=300, size=40_000)).astype(np.int16)
            folders = [
                _sync_recording(root / "one" / "recording", master, offset=0),
                _sync_recording(root / "two" / "recording", np.zeros_like(master), offset=0),
            ]
            options = SyncOptions(
                initial_start_seconds=0.1,
                initial_duration_seconds=0.5,
                initial_max_lag_seconds=0.01,
                window_seconds=0.5,
                step_seconds=0.25,
                tracking_max_lag_samples=30,
                chunk_seconds=1.0,
            )
            result = run_multidevice_sync(
                folders,
                master_index=0,
                output_folder=root / "output",
                merge=True,
                native_pc_time=False,
                integrity_duplication_scan=False,
                validate_postmerge=True,
                options=options,
            )

            output = root / "output"
            manifest = json.loads((output / "wild_preprocess_run.json").read_text(encoding="utf-8"))
            sample_count = int(manifest["merge"]["n_samples"])
            merged = np.fromfile(output / "amplifier.dat", dtype="<i2").reshape(sample_count, 8)
            validity = np.fromfile(output / "valid_samples.dat", dtype=np.uint8).reshape(sample_count, 2)
            self.assertNotEqual(result.status, "FAIL")
            self.assertEqual(result.outputs["sync_status"], "WARN")
            self.assertEqual(result.outputs["merge_status"], "WARN")
            self.assertEqual(result.outputs["overall_status"], "MERGE_ONLY")
            self.assertTrue(np.all(validity[:, 0] == 1))
            self.assertTrue(np.all(validity[:, 1] == 0))
            self.assertTrue(np.all(merged[:, 4:] == 0))
            self.assertFalse(any(segment.device_index == 2 for segment in result.device_sync_segments))

    def test_worker_job_rejects_duplicate_probes_and_misaligned_starts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "device1" / "0_20260809_120000"
            second = root / "device2" / "0_20260809_120020"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            _ce_header(first / "CE_params.bin")
            _ce_header(second / "CE_params.bin")
            _set_ce_rtc(first / "CE_params.bin", hours=12, minutes=0, seconds=0)
            _set_ce_rtc(second / "CE_params.bin", hours=12, minutes=0, seconds=20)
            job = {
                "schema_version": 3,
                "device_folders": [str(first), str(second)],
                "probe_indices": [1, 2],
                "master_index": 1,
            }
            folders, probes, anchors = _validated_job(job)
            self.assertEqual(folders, [first.resolve(), second.resolve()])
            self.assertEqual(probes, [1, 2])
            self.assertEqual([item["milliseconds_since_midnight"] for item in anchors], [43_200_000, 43_220_000])
            self.assertEqual([item["recording_date"] for item in anchors], ["2026-08-09", "2026-08-09"])
            self.assertTrue(all(item["source"] == "CE_params.bin" for item in anchors))
            self.assertTrue(all(item["recording_date_source"] == "CE_params.bin" for item in anchors))
            with self.assertRaisesRegex(ValueError, "Unsupported worker job schema"):
                _validated_job({**job, "schema_version": 2})
            bad_probe = {**job, "probe_indices": [1, 1]}
            with self.assertRaisesRegex(ValueError, "probe_indices"):
                _validated_job(bad_probe)
            late = root / "device3" / "0_20260809_120100"
            late.mkdir(parents=True)
            _ce_header(late / "CE_params.bin")
            _set_ce_rtc(late / "CE_params.bin", hours=12, minutes=1, seconds=0)
            with self.assertRaisesRegex(ValueError, "differ by"):
                _validated_job({**job, "device_folders": [str(first), str(late)]})
            before = root / "device4" / "0_20260809_115932"
            before.mkdir(parents=True)
            near_after = root / "device5" / "0_20260809_120029"
            near_after.mkdir(parents=True)
            _ce_header(before / "CE_params.bin")
            _ce_header(near_after / "CE_params.bin")
            _set_ce_rtc(before / "CE_params.bin", hours=11, minutes=59, seconds=32)
            _set_ce_rtc(near_after / "CE_params.bin", hours=12, minutes=0, seconds=29)
            with self.assertRaisesRegex(ValueError, "differ by"):
                _validated_job(
                    {
                        **job,
                        "device_folders": [str(first), str(near_after), str(before)],
                        "probe_indices": [1, 2, 3],
                    }
                )

            next_day = root / "device6" / "0_20260810_120000"
            next_day.mkdir(parents=True)
            _ce_header(next_day / "CE_params.bin")
            _set_ce_rtc(
                next_day / "CE_params.bin",
                hours=12,
                minutes=0,
                seconds=0,
                day=10,
            )
            with self.assertRaisesRegex(ValueError, "differ by"):
                _validated_job({**job, "device_folders": [str(first), str(next_day)]})

            before_midnight = root / "device7" / "recording"
            after_midnight = root / "device8" / "recording"
            before_midnight.mkdir(parents=True)
            after_midnight.mkdir(parents=True)
            _ce_header(before_midnight / "CE_params.bin")
            _ce_header(after_midnight / "CE_params.bin")
            _set_ce_rtc(
                before_midnight / "CE_params.bin",
                hours=23,
                minutes=59,
                seconds=59,
                day=9,
            )
            _set_ce_rtc(
                after_midnight / "CE_params.bin",
                hours=0,
                minutes=0,
                seconds=1,
                day=10,
            )
            _validated_job(
                {
                    **job,
                    "device_folders": [str(before_midnight), str(after_midnight)],
                }
            )

            for field in (
                "allow_folder_name_start_fallback",
                "overwrite",
                "merge",
                "integrity_duplication_scan",
                "write_event_files",
                "process_imu",
            ):
                with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, f"{field} must be a boolean"
                ):
                    _validated_job({**job, field: "false"})

    def test_pipeline_rejects_conflicting_explicit_and_validated_master_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folders = [_recording(root / f"device{index}" / "recording") for index in (1, 2)]
            anchors = [
                {
                    "milliseconds_since_midnight": 43_200_000,
                    "source": "validated",
                    "recording_date": "2026-08-09",
                },
                {
                    "milliseconds_since_midnight": 43_200_010,
                    "source": "validated",
                    "recording_date": "2026-08-09",
                },
            ]
            with self.assertRaisesRegex(ValueError, "disagrees"):
                run_multidevice_sync(
                    folders,
                    master_index=0,
                    output_folder=root / "output",
                    merge=False,
                    recording_start_ms=1,
                    recording_start_anchors=anchors,
                )

    def test_requested_imu_capacity_or_layout_warning_does_not_discard_core_merge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(260818)
            base = np.rint(rng.normal(scale=300, size=40_000)).astype(np.int16)
            folders = [
                _sync_recording(root / "one" / "recording", base, offset=0),
                _sync_recording(root / "two" / "recording", base, offset=12),
            ]
            options = SyncOptions(
                initial_start_seconds=0.1,
                initial_duration_seconds=0.5,
                initial_max_lag_seconds=0.01,
                window_seconds=0.5,
                step_seconds=0.25,
                tracking_max_lag_samples=30,
                chunk_seconds=1.0,
            )
            forced_success = PostMergeValidationResult(
                "OK", "forced staged success", "", 1, 1, 8, 8, 4, 0.05, 0.0, ()
            )
            with patch(
                "wild_preprocess.pipeline.validate_segment_staged_merge",
                return_value=forced_success,
            ):
                result = run_multidevice_sync(
                    folders,
                    master_index=0,
                    output_folder=root / "output",
                    merge=True,
                    process_imu=True,
                    validate_postmerge=True,
                    options=options,
                )
            self.assertNotEqual(result.status, "FAIL")
            self.assertEqual(result.outputs["imu_status"], "WARN")
            self.assertTrue((root / "output" / "amplifier.dat").is_file())
            self.assertFalse((root / "output" / "IMU.mat").exists())

    def test_time_dat_guard_allows_boundary_and_rejects_overflow(self) -> None:
        validate_time_dat_length(MAX_TIME_DAT_SAMPLES)
        with self.assertRaisesRegex(ValueError, "cannot represent"):
            validate_time_dat_length(MAX_TIME_DAT_SAMPLES + 1)

    def test_next_transaction_removes_previous_managed_device_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folders = [_recording(root / f"device{index}" / "recording") for index in range(1, 4)]
            recordings = [recording_from_folder(folder) for folder in folders]
            output = root / "output"
            fixed = {
                "wild_preprocess_run.json",
                "wild_multilogger_sync_master_vs_device3_qc.png",
                "device_event.dev03.d01.evt",
            }

            def first_stage(staging: Path, _outputs: dict[str, str]) -> None:
                (staging / "wild_multilogger_sync_master_vs_device3_qc.png").write_bytes(b"figure-3")
                (staging / "wild_preprocess_run.json").write_text(
                    json.dumps({"managed_files": sorted(fixed)}), encoding="utf-8"
                )

            merge_recordings(
                recordings,
                0,
                {1: _model(), 2: _model()},
                output,
                chunk_seconds=1.0,
                overwrite=False,
                run_id="first",
                stage_callback=first_stage,
                additional_managed_names=fixed,
                write_event_files=True,
            )
            self.assertTrue((output / "device_event.dev03.d01.evt").is_file())
            self.assertTrue((output / "wild_multilogger_sync_master_vs_device3_qc.png").is_file())

            def second_stage(staging: Path, _outputs: dict[str, str]) -> None:
                (staging / "wild_preprocess_run.json").write_text(
                    json.dumps({"managed_files": ["wild_preprocess_run.json"]}), encoding="utf-8"
                )

            merge_recordings(
                recordings[:2],
                0,
                {1: _model()},
                output,
                chunk_seconds=1.0,
                overwrite=True,
                run_id="second",
                stage_callback=second_stage,
                additional_managed_names={"wild_preprocess_run.json"},
            )
            self.assertFalse((output / "device_event.dev03.d01.evt").exists())
            self.assertEqual(list(output.glob("device_event.dev*.evt")), [])
            self.assertFalse((output / "wild_multilogger_sync_master_vs_device3_qc.png").exists())

    def test_pipeline_recovers_interrupted_publication_before_existing_output_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folders = [_recording(root / f"device{index}" / "recording") for index in (1, 2)]
            output = root / "output"
            output.mkdir()
            backup = output / ".wild_merge_backup_interrupted"
            backup.mkdir()
            (backup / "amplifier.dat").write_bytes(b"old")
            (output / "amplifier.dat").write_bytes(b"partial-new")
            (backup / "transaction.json").write_text(
                json.dumps(
                    {"old_names": ["amplifier.dat"], "new_names": ["amplifier.dat"]}
                ),
                encoding="utf-8",
            )
            with self.assertRaises(FileExistsError):
                run_multidevice_sync(
                    folders,
                    master_index=0,
                    output_folder=output,
                    merge=True,
                    overwrite=False,
                )
            self.assertEqual((output / "amplifier.dat").read_bytes(), b"old")
            self.assertFalse(backup.exists())

    def test_staged_stream_size_mismatch_blocks_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folders = [_recording(root / f"device{index}" / "recording") for index in (1, 2)]
            recordings = [recording_from_folder(folder) for folder in folders]
            output = root / "output"

            def truncate_time(staging: Path, _outputs: dict[str, str]) -> None:
                (staging / "time.dat").write_bytes(b"")

            with self.assertRaisesRegex(RuntimeError, "byte-length validation"):
                merge_recordings(
                    recordings,
                    0,
                    {1: _model()},
                    output,
                    chunk_seconds=1.0,
                    overwrite=False,
                    stage_callback=truncate_time,
                )
            self.assertFalse((output / "amplifier.dat").exists())

    def test_manifest_is_promoted_after_all_data_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folders = [_recording(root / f"device{index}" / "recording") for index in (1, 2)]
            recordings = [recording_from_folder(folder) for folder in folders]
            output = root / "output"
            promoted: list[str] = []
            real_replace = merge_module.os.replace

            def record_replace(source, destination):
                source_path = Path(source)
                destination_path = Path(destination)
                if source_path.parent.name.startswith(".wild_merge_stage_"):
                    promoted.append(destination_path.name)
                return real_replace(source, destination)

            def add_manifest(staging: Path, _outputs: dict[str, str]) -> None:
                (staging / "wild_preprocess_run.json").write_text(
                    json.dumps({"run_id": "ordered"}), encoding="utf-8"
                )

            with patch("wild_preprocess.sync.merge.os.replace", side_effect=record_replace):
                merge_recordings(
                    recordings,
                    0,
                    {1: _model()},
                    output,
                    chunk_seconds=1.0,
                    overwrite=False,
                    stage_callback=add_manifest,
                    additional_managed_names={"wild_preprocess_run.json"},
                )
            self.assertEqual(promoted[-1], "wild_preprocess_run.json")

    def test_incomplete_rollback_preserves_recovery_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folders = [_recording(root / f"device{index}" / "recording") for index in (1, 2)]
            recordings = [recording_from_folder(folder) for folder in folders]
            output = root / "output"
            output.mkdir()
            for name in ("amplifier.dat", "analogin.dat", "time.dat", "valid_samples.dat"):
                (output / name).write_bytes(f"old-{name}".encode())
            real_replace = merge_module.os.replace

            def fail_promotion_and_restore(source, destination):
                source_path = Path(source)
                destination_path = Path(destination)
                if (
                    source_path.parent.name.startswith(".wild_merge_stage_")
                    and destination_path.parent == output
                    and destination_path.name == "amplifier.dat"
                ):
                    raise OSError("forced promotion failure")
                if (
                    source_path.parent.name.startswith(".wild_merge_backup_")
                    and source_path.name == "amplifier.dat"
                ):
                    raise OSError("forced restore failure")
                return real_replace(source, destination)

            with patch(
                "wild_preprocess.sync.merge.os.replace",
                side_effect=fail_promotion_and_restore,
            ), self.assertRaisesRegex(RuntimeError, "rollback was incomplete"):
                merge_recordings(
                    recordings,
                    0,
                    {1: _model()},
                    output,
                    chunk_seconds=1.0,
                    overwrite=True,
                )
            backups = list(output.glob(".wild_merge_backup_*"))
            self.assertEqual(len(backups), 1)
            self.assertTrue((backups[0] / "amplifier.dat").is_file())

    def test_postmerge_rejection_keeps_one_manifest_inside_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(17)
            base = np.rint(rng.normal(scale=300, size=40_000)).astype(np.int16)
            folders = [
                _sync_recording(root / "one" / "recording", base, offset=0),
                _sync_recording(root / "two" / "recording", base, offset=12),
            ]
            options = SyncOptions(
                initial_start_seconds=0.1,
                initial_duration_seconds=0.5,
                initial_max_lag_seconds=0.01,
                window_seconds=0.5,
                step_seconds=0.25,
                tracking_max_lag_samples=30,
                chunk_seconds=1.0,
            )
            output = root / "output"
            forced_success = PostMergeValidationResult(
                "OK", "forced staged success", "transient-stage/amplifier.dat", 1, 1, 8, 8, 4, 0.05, 0.0, ()
            )
            anchors = [
                {
                    "milliseconds_since_midnight": 43_200_000 + index * 100,
                    "source": "recording folder name",
                    "recording_date": "2026-08-09",
                    "recording_date_source": "recording folder name",
                }
                for index in range(2)
            ]
            with patch("wild_preprocess.pipeline.validate_segment_staged_merge", return_value=forced_success):
                first = run_multidevice_sync(
                    folders,
                    master_index=0,
                    output_folder=output,
                    merge=True,
                    options=options,
                    validate_postmerge=True,
                    probe_indices=[1, 2],
                    recording_start_anchors=anchors,
            )
            self.assertEqual(first.status, "OK")
            canonical_amplifier = str(output / "amplifier.dat")
            manifest = json.loads(
                (output / "wild_preprocess_run.json").read_text(encoding="utf-8")
            )
            merge_json = manifest["merge"]
            sync_json = manifest["sync"]
            postmerge_qc = merge_json["postmerge_validation"]
            self.assertEqual(sync_json["status"], "OK")
            self.assertEqual(manifest["algorithm_versions"]["sync"], SYNC_ALGORITHM_VERSION)
            self.assertEqual(merge_json["backend"], SYNC_ALGORITHM_VERSION)
            self.assertEqual(postmerge_qc["amplifier_path"], canonical_amplifier)
            self.assertTrue(Path(postmerge_qc["amplifier_path"]).is_file())
            self.assertEqual(
                merge_json["postmerge_validation"]["amplifier_path"], canonical_amplifier
            )
            self.assertEqual(merge_json["probe_indices"], [1, 2])
            self.assertEqual(merge_json["recording_start_anchors"], anchors)
            self.assertIn(merge_json["common_interval_limits"]["end_limiter"]["stream"], {"ephys", "analog"})
            for device in merge_json["devices"]:
                expected_start = device["scale"] * merge_json["common_start_master_sample"] + device["intercept_samples"]
                expected_end = device["scale"] * merge_json["common_end_master_sample"] + device["intercept_samples"]
                self.assertAlmostEqual(device["mapped_ephys_start_sample"], expected_start)
                self.assertAlmostEqual(device["mapped_ephys_end_sample"], expected_end)
            self.assertEqual(manifest["warnings"], [])
            canonical_manifest = (output / "wild_preprocess_run.json").read_text(
                encoding="utf-8"
            )
            legacy_metadata = {
                "wild_multilogger_sync_qc.json",
                "wild_multilogger_sync_qc.mat",
                "wild_multilogger_mergeInfo.json",
                "wild_multilogger_mergeInfo.mat",
                "wild_multilogger_postmerge_qc.json",
                "wild_multilogger_events.tsv",
                "wild_preprocess_channel_layout.tsv",
                "pc_time_qc.json",
            }
            self.assertFalse(any((output / name).exists() for name in legacy_metadata))
            forced_warning = PostMergeValidationResult(
                "WARN",
                "localized boundary warning",
                "transient-stage/amplifier.dat",
                1,
                40_000,
                8,
                10_000,
                4,
                0.05,
                5.0,
                (
                    PostMergeMeasurement(
                        "boundary1_before",
                        0.125,
                        5_000,
                        5_000,
                        6_000,
                        2,
                        5,
                        0.8,
                        10.0,
                        0.5,
                        False,
                        "residual lag 5 exceeds 4 samples",
                        (1, 2),
                    ),
                ),
            )
            forced_warning_next = PostMergeValidationResult(
                "WARN",
                "localized boundary warning persists",
                "transient-stage/amplifier.dat",
                1,
                40_000,
                8,
                10_000,
                4,
                0.05,
                5.0,
                (
                    PostMergeMeasurement(
                        "boundary1_before",
                        0.15,
                        6_000,
                        6_000,
                        7_000,
                        2,
                        5,
                        0.8,
                        10.0,
                        0.5,
                        False,
                        "residual lag 5 exceeds 4 samples",
                        (1, 2),
                    ),
                ),
            )
            forced_revalidated = PostMergeValidationResult(
                "OK", "structural revalidation passed", "transient-stage/amplifier.dat",
                1, 40_000, 8, 10_000, 4, 0.05, None, (),
            )
            with patch(
                "wild_preprocess.pipeline.validate_segment_staged_merge",
                return_value=forced_warning,
            ) as staged_validator:
                warned = run_multidevice_sync(
                    folders,
                    master_index=0,
                    output_folder=output,
                    overwrite=True,
                    merge=True,
                    options=options,
                    validate_postmerge=True,
                    probe_indices=[1, 2],
                    recording_start_anchors=anchors,
                )
            self.assertEqual(staged_validator.call_count, 1)
            self.assertNotEqual(warned.status, "FAIL")
            self.assertEqual(warned.outputs["merge_status"], "WARN")
            warned_manifest = json.loads(
                (output / "wild_preprocess_run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(warned_manifest["merge_status"], "WARN")
            self.assertEqual(
                warned_manifest["merge"]["postmerge_validation"]["status"], "WARN"
            )
            self.assertFalse(
                warned_manifest["merge"]["postmerge_validation"][
                    "localized_exclusion_applied"
                ]
            )
            self.assertEqual(
                warned_manifest["merge"]["postmerge_validation"][
                    "localized_exclusion_rounds"
                ],
                0,
            )
            self.assertEqual(
                warned_manifest["merge"]["postmerge_validation"][
                    "final_revalidation_status"
                ],
                "WARN",
            )
            self.assertIn("localized boundary warning", warned_manifest["merge"]["postmerge_validation"]["message"])
            self.assertTrue(
                warned_manifest["merge"]["postmerge_validation"]["measurements"]
            )
            self.assertEqual(
                warned_manifest["merge"]["postmerge_validation"][
                    "final_revalidation_message"
                ],
                "localized boundary warning",
            )
            self.assertFalse(
                any(interval["kind"] == "postmerge_unverified" for interval in warned_manifest["merge"]["classified_intervals"])
            )
            merged = np.fromfile(output / "amplifier.dat", dtype="<i2").reshape(-1, 8)
            validity = np.fromfile(output / "valid_samples.dat", dtype=np.uint8).reshape(-1, 2)
            alignment = np.fromfile(output / "alignment_quality.dat", dtype=np.uint8).reshape(-1, 2)
            self.assertFalse(np.all(merged[5_000:6_000] == 0))
            self.assertTrue(np.all(validity[5_000:6_000] == 1))
            self.assertTrue(np.all(alignment[5_000:6_000, 1] == 0))
            self.assertTrue(np.all(alignment[5_000:6_000, 0] == 1))
            self.assertTrue(
                any(
                    warning.get("component") == "postmerge_validation"
                    for warning in warned_manifest["warnings"]
                )
            )
            canonical_manifest = (output / "wild_preprocess_run.json").read_text(
                encoding="utf-8"
            )

            with patch(
                "wild_preprocess.pipeline.validate_segment_staged_merge",
                return_value=forced_warning_next,
            ) as persistent_validator:
                persistent = run_multidevice_sync(
                    folders,
                    master_index=0,
                    output_folder=output,
                    overwrite=True,
                    merge=True,
                    options=options,
                    validate_postmerge=True,
                    probe_indices=[1, 2],
                    recording_start_anchors=anchors,
                )
            self.assertEqual(persistent_validator.call_count, 1)
            self.assertNotEqual(persistent.status, "FAIL")
            persistent_payload = json.loads(
                (output / "wild_preprocess_run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persistent_payload["merge_status"], "WARN")
            self.assertIn(
                "localized boundary warning persists",
                persistent_payload["merge"]["postmerge_validation"]["message"],
            )
            canonical_manifest = (output / "wild_preprocess_run.json").read_text(
                encoding="utf-8"
            )

            forced_failure = PostMergeValidationResult(
                "FAIL", "forced staged failure", "", 1, 1, 8, 8, 4, 0.05, None, ()
            )
            with patch("wild_preprocess.pipeline.validate_segment_staged_merge", return_value=forced_failure):
                failed = run_multidevice_sync(
                    folders,
                    master_index=0,
                    output_folder=output,
                    overwrite=True,
                    merge=True,
                    options=options,
                    validate_postmerge=True,
                    probe_indices=[1, 2],
                    recording_start_anchors=anchors,
                )
            self.assertEqual(failed.status, "FAIL")
            attempt = Path(failed.outputs["attempt_folder"])
            self.assertEqual(
                {path.name for path in attempt.iterdir()},
                {
                    "wild_preprocess_run.json",
                    *{
                        Path(pair.figure_file).name
                        for pair in failed.pairs
                    },
                },
            )
            payload = json.loads(
                (attempt / "wild_preprocess_run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["overall_status"], "FAIL")
            self.assertEqual(payload["sync"]["status"], "OK")
            self.assertTrue(
                all(Path(pair["figure_file"]).is_file() for pair in payload["sync"]["pairs"])
            )
            self.assertTrue(
                all(
                    Path(pair["figure_file"]).is_relative_to(attempt)
                    for pair in payload["sync"]["pairs"]
                )
            )
            failed_postmerge = payload["merge"]["postmerge_validation"]
            self.assertNotIn("amplifier_path", failed_postmerge)
            self.assertIn("validated_staging_amplifier_path", failed_postmerge)
            self.assertFalse(failed_postmerge["staged_artifact_retained"])
            self.assertEqual(
                (output / "wild_preprocess_run.json").read_text(encoding="utf-8"),
                canonical_manifest,
            )


if __name__ == "__main__":
    unittest.main()
