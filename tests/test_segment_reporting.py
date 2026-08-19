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

from WILD_preprocess_gui.wild_preprocess_gui import (
    PIPELINE_PROGRESS_STAGES,
    RecordingInfo,
    _imu_job_decision,
    _published_python_run_summary,
    _python_worker_job,
    stage_progress_text,
)
from wild_preprocess.inspection import (
    _REASON_COLORS,
    _UNVERIFIED_MAPPING_COLOR,
    _performance_report_line,
    _segment_report_lines,
    write_session_inspection_png,
)
from wild_preprocess.models import DeviceSyncAnchor, DeviceSyncSegment


def _segment(device_index: int, start: int, end: int) -> DeviceSyncSegment:
    anchors = (
        DeviceSyncAnchor(start, float(start), True, "high", "first full-rate anchor"),
        DeviceSyncAnchor(end - 1, float(end - 1), True, "high", "second full-rate anchor"),
    )
    return DeviceSyncSegment(
        device_index=device_index,
        canonical_start_sample=start,
        canonical_end_sample=end,
        source_start_sample=start,
        source_end_sample=end,
        source_scale=1.0,
        source_intercept_samples=0.0,
        anchors=anchors,
        confidence="high",
        start_transition="verified_reacquisition" if start else "recording_start",
        end_transition="recording_end",
        publishable=True,
    )


class SegmentReportingTest(unittest.TestCase):
    def test_gui_progress_is_stage_local_and_numbered(self) -> None:
        self.assertEqual(len(PIPELINE_PROGRESS_STAGES), 12)
        self.assertEqual(
            stage_progress_text("sync_pairs", 41.6),
            "Step 3/12 - Synchronize loggers - 42%",
        )
        self.assertEqual(stage_progress_text("complete", 100), "Complete - 100%")

    def test_gui_worker_job_requests_supported_imu_and_has_run_identity(self) -> None:
        recording = RecordingInfo(
            use=True,
            role="master",
            probe_index=1,
            device_id="device",
            recording_name="recording",
            folder=Path("recording"),
            fs=20_000,
            n_channels=64,
            n_samples=100,
            duration_sec=0.005,
            time_dat_valid=True,
            info_rhd_exists=False,
            imu_mat_exists=False,
            pc_time_exists=False,
            pc_time_valid=False,
        )
        job = _python_worker_job(
            [recording],
            master_index=1,
            output_folder=Path("output"),
            overwrite=False,
        )
        self.assertIs(job["process_imu"], True)
        self.assertIs(job["allow_folder_name_start_fallback"], False)
        self.assertRegex(job["run_id"], r"^[0-9a-f]{32}$")

        recording.n_channels = 4
        unsupported = _python_worker_job(
            [recording],
            master_index=1,
            output_folder=Path("output"),
            overwrite=False,
        )
        self.assertIs(unsupported["process_imu"], False)

        recording.n_channels = 64
        recording.duration_sec = 6 * 60 * 60
        enabled, reason = _imu_job_decision([recording, recording, recording])
        self.assertFalse(enabled)
        self.assertIn("in-memory limit", reason)

    def test_segment_and_timing_report_is_compact_and_accepts_manifest_records(self) -> None:
        first = _segment(1, 0, 50)
        second = _segment(1, 60, 100)
        slave = _segment(2, 0, 100)
        records = [
            {**segment.to_dict(), "validity_channel": 0 if segment.device_index == 1 else 1}
            for segment in (first, second, slave)
        ]
        lines = _segment_report_lines(
            {"device_sync_segments": records},
            device_valid_fractions=np.array([0.9, 1.0]),
            device_labels=["master", "slave 1"],
        )
        self.assertEqual(len(lines), 2)
        self.assertIn("2/2 verified segment(s), reacquired 1, invalid 10.000%", lines[0])
        self.assertIn("1/1 verified segment(s), reacquired 0, invalid 0.000%", lines[1])
        fully_invalid = _segment_report_lines(
            {"device_sync_segments": [records[0]]},
            device_valid_fractions=np.array([1.0, 0.0]),
            device_labels=["master", "slave 1"],
        )
        self.assertIn("slave 1: 0/0 verified segment(s), reacquired 0, invalid 100.000%", fully_invalid)
        timing = _performance_report_line(
            {
                "total_wall_seconds": 12.345,
                "stages": [
                    {"name": "raw_evidence_feature_scan", "wall_seconds": 2.0},
                    {"name": "ephys_merge", "wall_seconds": 3.5},
                ],
            }
        )
        self.assertEqual(timing, "timing: total 12.35s | evidence 2.00s | merge 3.50s")

    def test_inspection_accepts_optional_segment_and_performance_summaries(self) -> None:
        segment = _segment(1, 0, 20)
        with tempfile.TemporaryDirectory() as temporary:
            output = write_session_inspection_png(
                Path(temporary) / "inspection.png",
                sample_rate_hz=20_000.0,
                valid_samples=np.ones((20, 1), dtype=np.uint8),
                device_labels=["master"],
                segment_summary=[{**segment.to_dict(), "validity_channel": 0}],
                performance_summary={
                    "total_wall_seconds": 1.0,
                    "stages": [{"name": "coarse_correlation", "wall_seconds": 0.2}],
                },
                status="COMPLETE",
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 5_000)

    def test_gui_treats_published_warn_and_merge_only_as_published(self) -> None:
        self.assertEqual(
            _published_python_run_summary(
                {
                    "overall_status": "COMPLETE",
                    "sync_status": "WARN",
                    "merge_status": "WARN",
                    "pc_time_status": "OK",
                }
            ),
            (True, "published with warnings: sync, merge"),
        )
        self.assertEqual(
            _published_python_run_summary(
                {
                    "overall_status": "COMPLETE",
                    "sync_status": "OK",
                    "merge_status": "OK",
                    "pc_time_status": "WARN",
                }
            ),
            (True, "published with warnings: PC-time"),
        )
        self.assertEqual(
            _published_python_run_summary(
                {
                    "overall_status": "MERGE_ONLY",
                    "sync_status": "WARN",
                    "merge_status": "OK",
                    "pc_time_status": "WARN",
                }
            ),
            (True, "published with warnings: sync, PC-time"),
        )
        self.assertEqual(
            _published_python_run_summary({"overall_status": "FAIL"}),
            (False, "not published (FAIL)"),
        )
        self.assertEqual(
            _published_python_run_summary(
                {
                    "overall_status": "COMPLETE",
                    "sync_status": "OK",
                    "merge_status": "OK",
                    "analog_status": "OK",
                    "pc_time_status": "OK",
                    "imu_status": "WARN",
                }
            ),
            (True, "published with warnings: IMU"),
        )
        self.assertEqual(
            _published_python_run_summary(
                {"run_id": "old", "overall_status": "COMPLETE"},
                expected_run_id="current",
            ),
            (False, "run manifest belongs to a different worker run"),
        )

    def test_long_join_labels_are_not_annotated_and_panel_b_is_compact(self) -> None:
        figures = []
        long_label = "boundary_master_join_very_long_internal_diagnostic_name_slave_2"
        segments = [
            {**_segment(1, 0, 20).to_dict(), "validity_channel": 0},
            {**_segment(2, 0, 20).to_dict(), "validity_channel": 1},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            with patch("wild_preprocess.inspection.plt.close", side_effect=figures.append):
                output = write_session_inspection_png(
                    Path(temporary) / "inspection.png",
                    sample_rate_hz=20_000.0,
                    valid_samples=np.column_stack((np.ones(20, dtype=np.uint8), np.zeros(20, dtype=np.uint8))),
                    device_labels=["master", "slave 1"],
                    join_events=[
                        {
                            "time_sec": 0.0005,
                            "residual_samples": 17.0,
                            "status": "fail",
                            "label": long_label,
                        }
                    ],
                    segment_summary=segments,
                    performance_summary={
                        "total_wall_seconds": 2.0,
                        "stages": [{"name": "ephys_merge", "wall_seconds": 1.0}],
                    },
                )
            self.assertTrue(output.is_file())
        self.assertEqual(len(figures), 1)
        figure = figures[0]
        try:
            all_text = [text.get_text() for axis in figure.axes for text in axis.texts]
            all_text.extend(text.get_text() for text in figure.texts)
            self.assertNotIn(long_label, all_text)
            self.assertTrue(any("segments 2/2 verified; reacquired 0; fully invalid device(s) 2" in text for text in all_text))
            self.assertTrue(any("\ntiming: total 2.00s" in text for text in all_text))
            self.assertLessEqual(
                sum("segments " in text or text.startswith("timing:") for text in all_text),
                2,
            )
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)

    def test_unverified_mapping_and_missing_are_distinct_and_pc_error_is_concise(self) -> None:
        figures = []
        with tempfile.TemporaryDirectory() as temporary:
            with patch("wild_preprocess.inspection.plt.close", side_effect=figures.append):
                write_session_inspection_png(
                    Path(temporary) / "inspection.png",
                    sample_rate_hz=20_000.0,
                    valid_samples=np.column_stack((np.ones(40, dtype=np.uint8), np.zeros(40, dtype=np.uint8))),
                    device_labels=["master", "slave 1"],
                    reason_intervals=[
                        {
                            "canonical_start_sample": 10,
                            "canonical_end_sample": 20,
                            "reason": "missing",
                            "device_indices": (1,),
                        }
                    ],
                    pc_time={
                        "error": "packed PC-time update indices and values must be non-empty equal-length vectors"
                    },
                )
        self.assertEqual(len(figures), 1)
        figure = figures[0]
        try:
            validity_legend = figure.axes[1].get_legend()
            self.assertIsNotNone(validity_legend)
            legend_labels = [text.get_text() for text in validity_legend.get_texts()]
            self.assertIn("unverified mapping", legend_labels)
            self.assertIn("missing data", legend_labels)
            self.assertNotEqual(_UNVERIFIED_MAPPING_COLOR, _REASON_COLORS["missing"])
            pc_text = [text.get_text() for text in figure.axes[2].texts]
            self.assertIn("PC-time unavailable: no packed clock updates", pc_text)
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
