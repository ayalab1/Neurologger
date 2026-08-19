from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from wild_preprocess.inspection import (
    CameraCoverage,
    InspectionInterval,
    JoinResidual,
    _conservative_bins,
    write_session_inspection_png,
)
from wild_preprocess.models import SyncModel, SyncObservation, SyncPairResult
from wild_preprocess.pc_time.infer import PcTimeModel


def _pair() -> SyncPairResult:
    model = SyncModel(
        intercept_samples=3.0,
        slope_samples_per_second=0.001,
        drift_ppm=0.1,
        residual_rms_samples=0.2,
        residual_max_abs_samples=0.4,
        accepted_count=2,
        observation_count=3,
    )
    observations = [
        SyncObservation(1.0, 3.0, 3.0, 0.0, 0.8, 2.0, 0.2, None, True, model_residual_samples=0.0),
        SyncObservation(2.0, 3.0, 3.2, 0.2, 0.8, 2.0, 0.2, None, True, model_residual_samples=0.2),
        SyncObservation(3.0, 3.0, 8.0, 5.0, 0.1, 1.0, 0.0, None, False, "weak", model_residual_samples=5.0),
    ]
    return SyncPairResult(0, 1, "master", "slave", 3.0, 2.0, 0.2, model, observations, "WARN")


class SessionInspectionTest(unittest.TestCase):
    def test_conservative_validity_downsampling_preserves_single_invalid_sample(self) -> None:
        mask = np.ones((101, 3), dtype=np.uint8)
        mask[50, 1] = 0
        binned = _conservative_bins(mask, 10)
        self.assertEqual(binned.shape, (10, 3))
        self.assertFalse(binned[:, 1].all())
        self.assertTrue(binned[:, 0].all())
        self.assertTrue(binned[:, 2].all())

    def test_writes_complete_figure_from_models_mask_pc_time_and_camera(self) -> None:
        pc_model = PcTimeModel(
            device_ms=np.array([0.0, 1000.0, 2000.0]),
            pc_unwrapped_ms=np.array([10.0, 1010.0, 2010.0]),
            delay_ms=np.array([5.0, 5.0, 5.0]),
            residual_ms=np.array([0.0, 1.0, -2.0]),
            keep_mask=np.array([True, True, False]),
            slope=1.0,
            intercept_ms=10.0,
            slope_sem=0.0,
            intercept_sem_ms=0.0,
            recording_start_ms=10,
        )
        mask = np.ones((300, 2), dtype=np.uint8)
        mask[100:110, 1] = 0
        with tempfile.TemporaryDirectory() as temporary:
            output = write_session_inspection_png(
                Path(temporary) / "inspection.png",
                sample_rate_hz=100.0,
                pairs=[_pair()],
                valid_samples=mask,
                device_labels=["master", "slave 1"],
                reason_intervals=[InspectionInterval(100, 110, "duplication", (1,))],
                join_events=[JoinResidual(1.1, 0.3, "accepted", "join-1")],
                pc_time=pc_model,
                camera_coverage=[CameraCoverage(0.2, 2.8, True)],
                status="WARN",
                residual_tolerance_samples=2.0,
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 10_000)

    def test_writes_from_interleaved_validity_file_and_degrades_without_pc_or_camera(self) -> None:
        mask = np.ones((20, 3), dtype=np.uint8)
        mask[10, 2] = 0
        serialized_pair = {
            "slave_index": 2,
            "model": {"intercept_samples": 1.0, "slope_samples_per_second": 0.0, "offset_steps": []},
            "observations": [{"center_time_sec": 0.1, "observed_offset_samples": 1.0, "accepted": True}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validity_path = root / "valid_samples.dat"
            mask.tofile(validity_path)
            output = write_session_inspection_png(
                root / "inspection.png",
                sample_rate_hz=10.0,
                pairs=[serialized_pair],
                valid_samples_path=validity_path,
                device_count=3,
                device_labels=["master", "slave 1", "slave 2"],
                reason_intervals=[{"canonical_start_sample": 10, "missing_samples": 1, "reason": "uncertain"}],
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 5_000)

    def test_rejects_ambiguous_or_misaligned_validity_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "valid_samples.dat"
            np.ones(5, dtype=np.uint8).tofile(path)
            with self.assertRaisesRegex(ValueError, "divisible"):
                write_session_inspection_png(path.with_suffix(".png"), sample_rate_hz=1.0, valid_samples_path=path, device_count=2)
            with self.assertRaisesRegex(ValueError, "not both"):
                write_session_inspection_png(
                    path.with_suffix(".png"),
                    sample_rate_hz=1.0,
                    valid_samples=np.ones((2, 1), dtype=np.uint8),
                    valid_samples_path=path,
                )


if __name__ == "__main__":
    unittest.main()
