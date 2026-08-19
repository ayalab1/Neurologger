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

from wild_preprocess.pc_time.decode import (
    CE64_RAW_MISC_LAYOUT,
    EXPANDED_ANALOG_LAYOUT,
    PACKED_PC_MOD_MS,
    collect_packed_update_rows,
    collect_packed_updates,
)


def _packed(target_ms: int, delay_ms: int = 11) -> np.uint32:
    raw = (int(target_ms) - delay_ms) % PACKED_PC_MOD_MS
    return np.uint32(raw | (delay_ms << 20))


def _write_ce64(path: Path, packed_by_row: dict[int, np.uint32], *, n_cycles: int = 16) -> None:
    cycles = np.zeros((n_cycles, 16), dtype="<u2")
    for row, packed in packed_by_row.items():
        cycles[row, 14] = int(packed) & 0xFFFF
        cycles[row, 15] = int(packed) >> 16
    cycles.tofile(path)


class PackedRawRowsTest(unittest.TestCase):
    def test_ce64_rows_are_raw_cycles_and_legacy_coordinates_are_x16(self) -> None:
        value_a, value_b = _packed(10_000), _packed(10_500)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "analogin.dat"
            _write_ce64(path, {2: value_a, 3: value_a, 9: value_b, 10: value_b})
            raw = collect_packed_update_rows(path)
            legacy_rows, legacy_values, legacy_diagnostics = collect_packed_updates(
                path,
                CE64_RAW_MISC_LAYOUT,
                return_diagnostics=True,
            )
        np.testing.assert_array_equal(raw.raw_row_indices, [2, 9])
        np.testing.assert_array_equal(raw.values, [value_a, value_b])
        np.testing.assert_array_equal(legacy_rows, raw.raw_row_indices * 16)
        np.testing.assert_array_equal(legacy_values, raw.values)
        self.assertEqual(raw.diagnostics, legacy_diagnostics)
        self.assertFalse(raw.raw_row_indices.flags.writeable)
        self.assertFalse(raw.values.flags.writeable)

    def test_invalid_region_does_not_duplicate_a_held_packed_update(self) -> None:
        value = _packed(10_000)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "analogin.dat"
            _write_ce64(path, {0: value, 1: value, 4: value, 5: value})
            decoded = collect_packed_update_rows(path, valid_raw_runs=((0, 2), (4, 6)))
        np.testing.assert_array_equal(decoded.raw_row_indices, [0])
        np.testing.assert_array_equal(decoded.values, [value])
        self.assertEqual(decoded.diagnostics.raw_candidate_run_count, 2)
        self.assertEqual(decoded.diagnostics.accepted_update_count, 1)

    def test_stable_run_cannot_bridge_invalid_rows(self) -> None:
        value = _packed(10_000)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "analogin.dat"
            _write_ce64(path, {1: value, 3: value})
            valid = np.zeros(16, dtype=bool)
            valid[[1, 3]] = True
            decoded = collect_packed_update_rows(path, raw_valid_mask=valid)
        self.assertEqual(decoded.raw_row_indices.size, 0)
        self.assertEqual(decoded.diagnostics.raw_candidate_run_count, 2)
        self.assertEqual(decoded.diagnostics.accepted_update_count, 0)
        self.assertEqual(decoded.diagnostics.rejected_unstable_run_count, 2)

    def test_chunk_boundary_preserves_one_stable_run_and_diagnostics(self) -> None:
        value = _packed(10_000)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "analogin.dat"
            _write_ce64(path, {1: value, 2: value, 3: value, 4: value})
            whole = collect_packed_update_rows(path)
            chunked = collect_packed_update_rows(path, chunk_rows=2)
        np.testing.assert_array_equal(chunked.raw_row_indices, [1])
        np.testing.assert_array_equal(chunked.raw_row_indices, whole.raw_row_indices)
        np.testing.assert_array_equal(chunked.values, whole.values)
        self.assertEqual(chunked.diagnostics, whole.diagnostics)

    def test_empty_validity_returns_empty_immutable_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "analogin.dat"
            _write_ce64(path, {0: _packed(10_000), 1: _packed(10_000)})
            decoded = collect_packed_update_rows(path, raw_valid_mask=np.zeros(16, dtype=bool))
        self.assertEqual(decoded.raw_row_indices.size, 0)
        self.assertEqual(decoded.values.size, 0)
        self.assertEqual(decoded.diagnostics.accepted_update_count, 0)

    def test_expanded_rows_are_frames_not_ephys_coordinates(self) -> None:
        value = _packed(10_000)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "analogin.dat"
            frames = np.zeros((5, 6), dtype="<u2")
            frames[2:4, 3] = int(value) & 0xFFFF
            frames[2:4, 4] = int(value) >> 16
            frames.tofile(path)
            decoded = collect_packed_update_rows(path, EXPANDED_ANALOG_LAYOUT)
        np.testing.assert_array_equal(decoded.raw_row_indices, [2])
        np.testing.assert_array_equal(decoded.values, [value])

    def test_framing_errors_match_supported_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "analogin.dat"
            path.write_bytes(b"\x00" * 10)
            with self.assertRaisesRegex(ValueError, "raw-misc"):
                collect_packed_update_rows(path)
            with self.assertRaisesRegex(ValueError, "expanded"):
                collect_packed_update_rows(path, EXPANDED_ANALOG_LAYOUT)


if __name__ == "__main__":
    unittest.main()
