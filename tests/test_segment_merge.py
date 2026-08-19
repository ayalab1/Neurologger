from __future__ import annotations

import json
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

from wild_preprocess.integrity import SourceToCanonicalMapper
from wild_preprocess.models import DeviceSyncAnchor, DeviceSourceStep, DeviceSyncSegment, Recording, SyncModel
from wild_preprocess.sync.merge import (
    _merge_recordings_into_folder,
    _windowed_sinc_resampled_chunk,
)
from wild_preprocess.sync.segments import map_canonical_positions, validate_segment_collection


def _recording(root: Path, name: str, *, offset: int = 0) -> Recording:
    folder = root / name
    folder.mkdir()
    samples = 220
    channels = np.column_stack(
        [np.arange(samples, dtype=np.int16) + offset + channel * 1000 for channel in range(4)]
    )
    channels.astype("<i2").tofile(folder / "amplifier.dat")
    analog = np.column_stack(
        [np.arange(275, dtype=np.int16) + offset]
    ).astype("<i2")
    analog.tofile(folder / "analogin.dat")
    (folder / "CE_params.bin").write_bytes(bytes(512))
    return Recording(
        folder=folder,
        amplifier_file=folder / "amplifier.dat",
        analog_file=folder / "analogin.dat",
        ce_params_file=folder / "CE_params.bin",
        device_name=name,
        recording_name="recording",
        fs=1000,
        n_channels=4,
        n_samples=samples,
        analog_channels=1,
        analog_samples=275,
    )


def _segment(device: int, canonical_start: int, canonical_end: int, intercept: float) -> DeviceSyncSegment:
    anchors = tuple(
        DeviceSyncAnchor(
            canonical_sample=canonical,
            source_sample=canonical + intercept,
            verified=True,
            confidence="high",
        )
        for canonical in (canonical_start, canonical_end - 1)
    )
    return DeviceSyncSegment(
        device_index=device,
        canonical_start_sample=canonical_start,
        canonical_end_sample=canonical_end,
        source_start_sample=int(canonical_start + intercept),
        source_end_sample=int(canonical_end + intercept),
        source_scale=1.0,
        source_intercept_samples=intercept,
        anchors=anchors,
        confidence="high",
        publishable=True,
    )


def _model() -> SyncModel:
    return SyncModel(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)


def _scaled_segment(
    device: int,
    canonical_start: int,
    canonical_end: int,
    scale: float,
) -> DeviceSyncSegment:
    mapped_start = scale * canonical_start
    mapped_last = scale * (canonical_end - 1)
    anchors = (
        DeviceSyncAnchor(canonical_start, mapped_start, True, "high"),
        DeviceSyncAnchor(canonical_end - 1, mapped_last, True, "high"),
    )
    return DeviceSyncSegment(
        device_index=device,
        canonical_start_sample=canonical_start,
        canonical_end_sample=canonical_end,
        source_start_sample=int(np.floor(mapped_start)),
        source_end_sample=int(np.floor(mapped_last)) + 2,
        source_scale=scale,
        source_intercept_samples=0.0,
        anchors=anchors,
        confidence="high",
        publishable=True,
    )


class SegmentMergeTest(unittest.TestCase):
    def test_pre_post_segments_leave_unresolved_gap_invalid_without_source_step(self) -> None:
        master = _segment(1, 16, 204, 0.0)
        slave_pre = _segment(2, 16, 70, 0.0)
        slave_post = _segment(2, 90, 204, 10.0)
        positions = np.arange(68, 92, dtype=np.float64)
        mapped, valid = map_canonical_positions(
            (slave_pre, slave_post),
            positions,
            source_sample_count=220,
            interpolation_half_width=16,
            device_index=2,
        )
        np.testing.assert_array_equal(valid, np.r_[np.ones(2, bool), np.zeros(20, bool), np.ones(2, bool)])
        self.assertEqual(mapped[0], 68.0)
        self.assertTrue(np.isnan(mapped[2]))
        self.assertEqual(mapped[-1], 101.0)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recordings = [_recording(root, "master"), _recording(root, "slave", offset=10_000)]
            outputs = _merge_recordings_into_folder(
                recordings,
                0,
                {1: _model()},
                root / "output",
                chunk_seconds=1.0,
                overwrite=False,
                device_source_steps=[DeviceSourceStep(2, 90, 100.0, "unresolved")],
                device_sync_segments=[master, slave_pre, slave_post],
            )
            merged = np.fromfile(outputs["amplifier"], dtype="<i2").reshape(-1, 8)
            validity = np.fromfile(outputs["validity"], dtype=np.uint8).reshape(-1, 2)
            canonical = np.arange(16, 204)
            gap = (canonical >= 70) & (canonical < 90)
            self.assertTrue(np.all(merged[gap, 4:] == 0))
            self.assertTrue(np.all(validity[gap, 1] == 0))
            self.assertTrue(np.all(merged[~gap, 4:] != 0))
            self.assertTrue(np.all(validity[~gap, 1] == 1))
            post_index = int(np.flatnonzero(canonical == 90)[0])
            self.assertEqual(merged[post_index, 4], 10_100)
            merge_info = json.loads(Path(outputs["_merge_internal"]).read_text(encoding="utf-8"))
            self.assertTrue(merge_info["segment_mapping_authoritative"])
            self.assertEqual(len(merge_info["device_sync_segments"]), 3)

    def test_absent_slave_segments_zero_fill_only_that_device_and_inverse_uses_segments(self) -> None:
        master = _segment(1, 16, 204, 0.0)
        mapper = SourceToCanonicalMapper(
            device_index=2,
            source_scale=1.0,
            intercept_samples=0.0,
            device_sync_segments=[_segment(2, 16, 70, 0.0), _segment(2, 90, 204, 10.0)],
        )
        np.testing.assert_array_equal(
            mapper.map_array(np.asarray([65, 70, 100, 120])),
            np.asarray([65, -1, 90, 110]),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recordings = [_recording(root, "master"), _recording(root, "slave", offset=10_000)]
            outputs = _merge_recordings_into_folder(
                recordings,
                0,
                {1: _model()},
                root / "output",
                chunk_seconds=1.0,
                overwrite=False,
                device_sync_segments=[master],
            )
            merged = np.fromfile(outputs["amplifier"], dtype="<i2").reshape(-1, 8)
            validity = np.fromfile(outputs["validity"], dtype=np.uint8).reshape(-1, 2)
            self.assertTrue(np.all(merged[:, :4] != 0))
            self.assertTrue(np.all(validity[:, 0] == 1))
            self.assertTrue(np.all(merged[:, 4:] == 0))
            self.assertTrue(np.all(validity[:, 1] == 0))

    def test_source_reversal_across_publishable_segments_is_structural_error(self) -> None:
        first = _segment(2, 16, 70, 0.0)
        reversed_second = _segment(2, 90, 204, -40.0)
        with self.assertRaisesRegex(ValueError, "strictly forward"):
            validate_segment_collection((first, reversed_second), device_index=2)

    def test_segment_scale_controls_sinc_antialias_cutoff(self) -> None:
        master = _segment(1, 16, 204, 0.0)
        slave = _scaled_segment(2, 16, 164, 1.25)
        cutoffs: list[float] = []

        def capture_cutoff(source, positions, *, half_width=16, cutoff=1.0):
            cutoffs.append(float(cutoff))
            return _windowed_sinc_resampled_chunk(
                source,
                positions,
                half_width=half_width,
                cutoff=cutoff,
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recordings = [_recording(root, "master"), _recording(root, "slave", offset=10_000)]
            with patch(
                "wild_preprocess.sync.merge._windowed_sinc_resampled_chunk",
                side_effect=capture_cutoff,
            ):
                _merge_recordings_into_folder(
                    recordings,
                    0,
                    {1: _model()},
                    root / "output",
                    chunk_seconds=1.0,
                    overwrite=False,
                    device_sync_segments=[master, slave],
                )
        self.assertTrue(any(abs(cutoff - 0.8) < 1e-12 for cutoff in cutoffs))

    def test_integer_decimation_still_applies_segment_antialias_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nyquist.dat"
            source = (1_000 * np.where(np.arange(4096) % 2 == 0, 1, -1)).astype("<i2")
            source.reshape(-1, 1).tofile(path)
            mapped = np.memmap(path, dtype="<i2", mode="r", shape=(source.size, 1))
            try:
                positions = np.arange(64, 4032, 2, dtype=np.float64)
                unfiltered = _windowed_sinc_resampled_chunk(mapped, positions, cutoff=1.0)
                antialiased = _windowed_sinc_resampled_chunk(mapped, positions, cutoff=0.5)
                self.assertGreater(float(np.mean(np.abs(unfiltered))), 900.0)
                self.assertLess(float(np.mean(np.abs(antialiased))), 25.0)
            finally:
                del mapped


if __name__ == "__main__":
    unittest.main()
