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

from wild_preprocess.models import DeviceGap
from wild_preprocess.pc_time.canonical import (
    fit_gap_aware_pc_time_model,
    map_camera_timestamps_to_canonical,
    map_raw_master_indices_to_canonical,
    unwrap_daily_ms,
    validate_canonical_pc_time_interval,
    write_canonical_interval_pc_time,
)
from wild_preprocess.pc_time.decode import PACKED_PC_MOD_MS


def _packed(target_ms: np.ndarray, delay_ms: int = 7) -> np.ndarray:
    raw = (np.asarray(target_ms, dtype=np.int64) - delay_ms) % PACKED_PC_MOD_MS
    return (raw | (np.int64(delay_ms) << 20)).astype(np.uint32)


class CanonicalPcTimeCameraTest(unittest.TestCase):
    def test_maps_raw_master_updates_after_confirmed_master_gaps_without_mutation(self) -> None:
        raw = np.array([0, 99, 100, 194, 195, 200], dtype=np.int64)
        original = raw.copy()
        gaps = [
            DeviceGap(1, 100, 5, 0.25, confidence="high"),
            DeviceGap(1, 200, 3, 0.15, confidence="medium"),
            DeviceGap(2, 40, 4, 0.2, confidence="high"),
        ]
        canonical = map_raw_master_indices_to_canonical(raw, gaps, master_device_index=1)
        np.testing.assert_array_equal(canonical, [0, 99, 105, 199, 203, 208])
        np.testing.assert_array_equal(raw, original)

    def test_gap_aware_fit_validation_and_writer_use_canonical_coordinates(self) -> None:
        fs = 1_000.0
        raw = np.arange(0, 551, 50, dtype=np.int64)
        gaps = [DeviceGap(1, 100, 5, 5.0, confidence="high")]
        canonical = map_raw_master_indices_to_canonical(raw, gaps, master_device_index=1)
        target = 4_000_000 + canonical
        fit = fit_gap_aware_pc_time_model(
            raw,
            _packed(target),
            fs,
            4_000_000,
            device_gaps=gaps,
            master_device_index=1,
        )
        np.testing.assert_array_equal(fit.canonical_update_indices, canonical)
        self.assertAlmostEqual(fit.model.drift_ppm, 0.0, delta=0.01)
        validation = validate_canonical_pc_time_interval(
            fit,
            sample_rate_hz=fs,
            canonical_start_sample=0,
            n_samples=555,
        )
        self.assertEqual(validation.status, "OK")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "pc_time.dat"
            write_canonical_interval_pc_time(
                output,
                fit,
                sample_rate_hz=fs,
                canonical_start_sample=100,
                n_samples=8,
            )
            values = np.fromfile(output, dtype="<u4")
            np.testing.assert_array_equal(values, np.arange(4_000_100, 4_000_108, dtype=np.uint32))

    def test_unwraps_daily_pc_and_camera_timestamps_and_applies_requested_channels(self) -> None:
        pc = np.array([86_399_998, 86_399_999, 0, 1, 2], dtype=np.uint32)
        valid = np.ones((5, 3), dtype=np.uint8)
        valid[2, 1] = 0
        result = map_camera_timestamps_to_canonical(
            pc,
            [86_399_999, 0, 1, 5],
            requested_validity_channels=[0, 1],
            valid_samples=valid,
        )
        np.testing.assert_array_equal(result.canonical_sample_indices, [1, 2, 3, -1])
        np.testing.assert_array_equal(result.selected_device_valid, [True, False, True, False])
        np.testing.assert_array_equal(result.in_range, [True, True, True, False])
        np.testing.assert_allclose(result.residual_ms[:3], 0.0)
        self.assertTrue(np.isnan(result.residual_ms[3]))
        self.assertEqual(result.canonical_pc_unwrapped_ms.size, 0)
        retained = map_camera_timestamps_to_canonical(
            pc,
            [0],
            retain_canonical_pc_unwrapped=True,
        )
        np.testing.assert_array_equal(
            retained.canonical_pc_unwrapped_ms,
            [86_399_998.0, 86_399_999.0, 86_400_000.0, 86_400_001.0, 86_400_002.0],
        )
        np.testing.assert_array_equal(unwrap_daily_ms(pc), [86_399_998.0, 86_399_999.0, 86_400_000.0, 86_400_001.0, 86_400_002.0])

    def test_camera_path_mask_and_distance_gate_reject_unusable_or_distant_frames(self) -> None:
        pc = np.array([100, 110, 120, 130], dtype=np.uint32)
        valid = np.ones((4, 2), dtype=np.uint8)
        valid[1, 1] = 0
        with tempfile.TemporaryDirectory() as temporary:
            validity_path = Path(temporary) / "valid_samples.dat"
            valid.tofile(validity_path)
            result = map_camera_timestamps_to_canonical(
                pc,
                [109, 116, 500],
                requested_validity_channels=[1],
                valid_samples_path=validity_path,
                device_count=2,
                max_distance_ms=3.0,
            )
        np.testing.assert_array_equal(result.canonical_sample_indices, [1, -1, -1])
        np.testing.assert_array_equal(result.selected_device_valid, [False, False, False])
        np.testing.assert_array_equal(result.in_range, [True, False, False])

    def test_rejects_unconfirmed_master_gap_and_nonmonotonic_pc_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "confirmed"):
            map_raw_master_indices_to_canonical(
                [0, 1],
                [DeviceGap(1, 1, 2, 1.0, confidence="unresolved")],
                master_device_index=1,
            )
        with self.assertRaisesRegex(ValueError, "monotonic"):
            map_camera_timestamps_to_canonical(
                np.array([100, 90, 110], dtype=np.uint32),
                [100],
            )


if __name__ == "__main__":
    unittest.main()
