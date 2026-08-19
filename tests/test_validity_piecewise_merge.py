from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

from wild_preprocess.models import (
    ClassifiedInterval,
    DeviceSourceStep,
    DeviceSyncAnchor,
    DeviceSyncSegment,
    DeviceTerminalSupport,
    Recording,
    SyncModel,
)
from wild_preprocess.sync.merge import (
    _common_master_interval,
    _write_interleaved_stream,
    rewrite_staged_ephys_from_segments,
    write_alignment_quality,
)


def _model() -> SyncModel:
    return SyncModel(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)


class ValidityPiecewiseMergeTests(unittest.TestCase):
    def _recording(self, root: Path, name: str, samples: int) -> Recording:
        folder = root / name
        folder.mkdir()
        amplifier = folder / "amplifier.dat"
        (np.arange(samples, dtype=np.int16) + (0 if name == "master" else 1000)).astype(
            "<i2"
        ).tofile(amplifier)
        analog = folder / "analogin.dat"
        np.arange(600, dtype="<i2").tofile(analog)
        return Recording(
            folder=folder,
            amplifier_file=amplifier,
            analog_file=analog,
            ce_params_file=folder / "CE_params.bin",
            device_name=name,
            recording_name="recording",
            fs=1000,
            n_channels=1,
            n_samples=samples,
            analog_channels=1,
            analog_samples=600,
        )

    def test_piecewise_mapping_and_device_local_validity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recordings = [
                self._recording(root, "master", 400),
                self._recording(root, "slave", 400),
            ]
            amplifier = root / "merged.dat"
            validity_path = root / "valid_samples.dat"
            _write_interleaved_stream(
                amplifier,
                recordings,
                [_model(), _model()],
                0,
                20,
                300,
                stream="ephys",
                chunk_seconds=0.05,
                overwrite=False,
                progress=None,
                validity_path=validity_path,
                classified_intervals=[
                    ClassifiedInterval(
                        (1, 2), 90, 110, "unresolved_boundary", "zero_fill", "unresolved"
                    ),
                    ClassifiedInterval(
                        (2,), 150, 160, "duplicate_destination", "zero_fill", "medium", 145, 155
                    ),
                ],
                device_source_steps=[DeviceSourceStep(2, 100, 5.0, "unresolved")],
                device_terminal_support=[DeviceTerminalSupport(2, 200, "terminal")],
            )
            merged = np.fromfile(amplifier, dtype="<i2").reshape(-1, 2)
            validity = np.fromfile(validity_path, dtype=np.uint8).reshape(-1, 2)

        self.assertEqual(int(merged[120 - 20, 1]), 1000 + 125)
        self.assertTrue(np.all(validity[90 - 20 : 110 - 20] == 0))
        self.assertTrue(np.all(validity[150 - 20 - 16 : 160 - 20 + 16, 1] == 0))
        self.assertTrue(np.all(validity[200 - 20 :, 0] == 1))
        self.assertTrue(np.all(validity[200 - 20 :, 1] == 0))
        self.assertTrue(np.all(merged[200 - 20 :, 1] == 0))

    def test_master_extent_can_outlive_short_slave(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recordings = [
                self._recording(root, "master", 400),
                self._recording(root, "slave", 120),
            ]
            _, common_end, _ = _common_master_interval(
                recordings, [_model(), _model()], 0, preserve_device_tails=False
            )
            _, preserved_end, limits = _common_master_interval(
                recordings, [_model(), _model()], 0, preserve_device_tails=True
            )

        self.assertLess(common_end, 120)
        self.assertGreater(preserved_end, 300)
        self.assertEqual(limits["end_limiter"]["device_index"], 1)

    def test_corrected_segment_rerenders_once_from_raw_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recordings = [
                self._recording(root, "master", 400),
                self._recording(root, "slave", 400),
            ]

            def segment(device: int, intercept: float, end: int) -> DeviceSyncSegment:
                return DeviceSyncSegment(
                    device_index=device,
                    canonical_start_sample=20,
                    canonical_end_sample=end,
                    source_start_sample=0,
                    source_end_sample=400,
                    source_scale=1.0,
                    source_intercept_samples=intercept,
                    anchors=(
                        DeviceSyncAnchor(20, 20.0 + intercept, True, "high", "start"),
                        DeviceSyncAnchor(end - 1, end - 1.0 + intercept, True, "high", "end"),
                    ),
                    confidence="high",
                    publishable=True,
                )

            amplifier = root / "merged.dat"
            validity_path = root / "valid_samples.dat"
            summary = rewrite_staged_ephys_from_segments(
                amplifier,
                validity_path,
                recordings,
                0,
                {1: _model()},
                common_start=20,
                common_end=120,
                chunk_seconds=0.05,
                device_sync_segments=[segment(1, 0.0, 121), segment(2, 5.0, 121)],
            )
            merged = np.fromfile(amplifier, dtype="<i2").reshape(-1, 2)

        self.assertEqual(int(merged[25 - 20, 0]), 25)
        self.assertEqual(int(merged[25 - 20, 1]), 1030)
        self.assertEqual(summary["valid_fraction_by_channel"], [1.0, 1.0])

    def test_amplifier_keeps_input_order_while_validity_is_master_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recordings = [
                self._recording(root, "slave", 400),
                self._recording(root, "master", 400),
            ]
            amplifier = root / "merged.dat"
            validity_path = root / "valid_samples.dat"
            _write_interleaved_stream(
                amplifier,
                recordings,
                [_model(), _model()],
                1,
                20,
                120,
                stream="ephys",
                chunk_seconds=0.05,
                overwrite=False,
                progress=None,
                validity_path=validity_path,
                classified_intervals=[
                    ClassifiedInterval(
                        (1,), 60, 70, "unresolved_boundary", "zero_fill", "unresolved"
                    )
                ],
            )
            merged = np.fromfile(amplifier, dtype="<i2").reshape(-1, 2)
            validity = np.fromfile(validity_path, dtype=np.uint8).reshape(-1, 2)

        self.assertEqual(int(merged[25 - 20, 0]), 1025)
        self.assertEqual(int(merged[25 - 20, 1]), 25)
        self.assertTrue(np.all(validity[:, 0] == 1))
        self.assertTrue(np.all(validity[60 - 20 : 70 - 20, 1] == 0))

    def test_alignment_quality_maps_recording_device_to_master_first_column(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validity_path = root / "valid_samples.dat"
            quality_path = root / "alignment_quality.dat"
            np.ones((20, 2), dtype=np.uint8).tofile(validity_path)
            summary = write_alignment_quality(
                validity_path,
                quality_path,
                n_samples=20,
                device_count=2,
                master_index=1,
                canonical_start_sample=100,
                warning_intervals=(
                    {
                        "canonical_start_sample": 105,
                        "canonical_end_sample": 110,
                        "affected_device_indices": [1],
                    },
                ),
            )
            quality = np.fromfile(quality_path, dtype=np.uint8).reshape(-1, 2)

        self.assertTrue(np.all(quality[:, 0] == 1))
        self.assertTrue(np.all(quality[5:10, 1] == 0))
        self.assertEqual(summary["valid_fraction_by_channel"], [1.0, 0.75])

    def test_neural_discontinuities_do_not_shift_analog_without_analog_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recordings = [
                self._recording(root, "master", 400),
                self._recording(root, "slave", 400),
            ]
            analog_path = root / "merged_analog.dat"
            _write_interleaved_stream(
                analog_path,
                recordings,
                [_model(), _model()],
                0,
                20,
                180,
                stream="analog",
                chunk_seconds=0.05,
                overwrite=False,
                progress=None,
                classified_intervals=[
                    ClassifiedInterval(
                        (1, 2), 90, 110, "unresolved_boundary", "zero_fill", "unresolved"
                    )
                ],
                device_source_steps=[DeviceSourceStep(2, 100, 5.0, "unresolved")],
                device_terminal_support=[DeviceTerminalSupport(2, 140, "terminal")],
            )
            analog = np.fromfile(analog_path, dtype="<i2").reshape(-1, 2)

        np.testing.assert_array_equal(analog[:, 0], analog[:, 1])
        self.assertTrue(np.all(analog != 0))


if __name__ == "__main__":
    unittest.main()
