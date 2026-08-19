from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from wild_preprocess.analog.models import AnalogSyncAnchor, AnalogSyncSegment
from wild_preprocess.pc_time.analog_mapping import fit_pc_time_through_analog_mapping
from wild_preprocess.pc_time.decode import PackedUpdateDiagnostics, PackedUpdates
from wild_preprocess.pc_time.infer import fit_robust_pc_time_model


def _segment(
    canonical_start: int,
    canonical_end: int,
    raw_start: int,
    raw_end: int,
    *,
    scale: float = 1.0,
    intercept: float = 0.0,
) -> AnalogSyncSegment:
    return AnalogSyncSegment(
        device_index=1,
        canonical_start_row=canonical_start,
        canonical_end_row=canonical_end,
        raw_start_row=raw_start,
        raw_end_row=raw_end,
        raw_scale=scale,
        raw_intercept_rows=intercept,
        anchors=(
            AnalogSyncAnchor(canonical_start, scale * canonical_start + intercept, True, "high"),
            AnalogSyncAnchor(canonical_end - 1, scale * (canonical_end - 1) + intercept, True, "high"),
        ),
        confidence="high",
        publishable=True,
    )


def _updates(rows: list[int], values: list[int]) -> PackedUpdates:
    return PackedUpdates(
        np.asarray(rows, dtype=np.int64),
        np.asarray(values, dtype=np.uint32),
        PackedUpdateDiagnostics(len(rows), len(rows), 0, 2),
    )


class PcTimeAnalogMappingTests(unittest.TestCase):
    def test_fractional_analog_mapping_is_not_truncated_before_neural_rounding(self) -> None:
        # Raw row 11 maps to canonical analog row 10.5, then to neural 21.0.
        # Premature integer analog-row truncation would instead produce 20.
        fit = fit_pc_time_through_analog_mapping(
            _updates([1, 11], [1, 2]),
            (_segment(0, 20, 0, 21, scale=1.0, intercept=0.5),),
            canonical_analog_row_zero_neural_sample=0.0,
            ephys_sample_rate_hz=2_500.0,
            analog_sample_rate_hz=1_250.0,
            recording_start_ms=0,
        )
        np.testing.assert_allclose(fit.canonical_analog_rows, [0.5, 10.5])
        np.testing.assert_array_equal(fit.canonical_neural_indices, [1, 21])

    def test_invalid_raw_rows_are_skipped(self) -> None:
        fit = fit_pc_time_through_analog_mapping(
            _updates([0, 9, 10, 14, 15, 24], [1, 2, 3, 4, 5, 6]),
            (_segment(0, 10, 0, 10), _segment(20, 30, 15, 25, intercept=-5.0)),
            canonical_analog_row_zero_neural_sample=0.0,
            ephys_sample_rate_hz=1_250.0,
            analog_sample_rate_hz=1_250.0,
            recording_start_ms=0,
        )
        np.testing.assert_array_equal(fit.raw_update_rows, [0, 9, 15, 24])
        np.testing.assert_array_equal(fit.canonical_neural_indices, [0, 9, 20, 29])
        self.assertEqual(fit.diagnostics.invalid_mapping_anchor_count, 2)

    def test_agreeing_duplicate_neural_indices_collapse_deterministically(self) -> None:
        fit = fit_pc_time_through_analog_mapping(
            _updates([0, 1, 10], [9, 9, 10]),
            (_segment(0, 20, 0, 20),),
            canonical_analog_row_zero_neural_sample=0.0,
            ephys_sample_rate_hz=100.0,
            analog_sample_rate_hz=1_000.0,
            recording_start_ms=0,
        )
        np.testing.assert_array_equal(fit.raw_update_rows, [0, 10])
        np.testing.assert_array_equal(fit.canonical_neural_indices, [0, 1])
        self.assertEqual(fit.diagnostics.collapsed_agreeing_duplicate_count, 1)

    def test_conflicting_duplicate_neural_indices_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "same canonical neural sample"):
            fit_pc_time_through_analog_mapping(
                _updates([0, 1], [9, 10]),
                (_segment(0, 20, 0, 20),),
                canonical_analog_row_zero_neural_sample=0.0,
                ephys_sample_rate_hz=100.0,
                analog_sample_rate_hz=1_000.0,
                recording_start_ms=0,
            )

    def test_mapping_is_monotonic_and_provenance_is_deterministic(self) -> None:
        kwargs = dict(
            packed_updates=_updates([0, 10, 20], [1, 2, 3]),
            segments=(_segment(0, 30, 0, 30),),
            canonical_analog_row_zero_neural_sample=100.25,
            ephys_sample_rate_hz=1_250.0,
            analog_sample_rate_hz=1_250.0,
            recording_start_ms=0,
        )
        first = fit_pc_time_through_analog_mapping(**kwargs)
        second = fit_pc_time_through_analog_mapping(**kwargs)
        self.assertTrue(np.all(np.diff(first.canonical_neural_indices) > 0))
        self.assertEqual(first.provenance_hash, second.provenance_hash)
        self.assertLessEqual(first.diagnostics.max_quantization_error_seconds, 0.5 / 1_250.0)

    def test_clean_mapping_fit_matches_direct_neural_fit(self) -> None:
        values = np.array([0, 1_000, 2_000, 3_000], dtype=np.uint32)
        rows = np.array([0, 1_250, 2_500, 3_750], dtype=np.int64)
        fit = fit_pc_time_through_analog_mapping(
            _updates(rows.tolist(), values.tolist()),
            (_segment(0, 4_000, 0, 4_000),),
            canonical_analog_row_zero_neural_sample=0.0,
            ephys_sample_rate_hz=1_250.0,
            analog_sample_rate_hz=1_250.0,
            recording_start_ms=0,
        )
        direct = fit_robust_pc_time_model(rows, values, 1_250.0, 0)
        np.testing.assert_array_equal(fit.canonical_neural_indices, rows)
        self.assertAlmostEqual(fit.model.slope, direct.slope, places=12)
        self.assertAlmostEqual(fit.model.intercept_ms, direct.intercept_ms, places=12)


if __name__ == "__main__":
    unittest.main()
