from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from wild_preprocess.models import Recording
from wild_preprocess.audit import RawAuditOptions, scan_exact_duplications
from wild_preprocess.sync.features import (
    build_coarse_feature,
    build_common_mode_feature,
    build_raw_evidence_scan,
    feature_memmap,
    frame_hash_memmap,
)


def _recording(root: Path, samples: int, channels: int, fs: int) -> Recording:
    amplifier = root / "amplifier.dat"
    return Recording(
        folder=root,
        amplifier_file=amplifier,
        analog_file=root / "analogin.dat",
        ce_params_file=root / "CE_params.bin",
        device_name="device",
        recording_name="recording",
        fs=fs,
        n_channels=channels,
        n_samples=samples,
        analog_channels=channels // 4,
        analog_samples=1,
    )


def _reference_hashes(values: np.ndarray) -> np.ndarray:
    """The established audit full-frame hash, evaluated as a dense reference."""

    rows = values.view("<u2").astype(np.uint64)
    channel_index = np.arange(1, values.shape[1] + 1, dtype=np.uint64)
    mixed = (rows + channel_index * np.uint64(0xC2B2AE3D27D4EB4F)) * (
        channel_index * np.uint64(0x9E3779B185EBCA87)
    )
    return np.bitwise_xor.reduce(mixed, axis=1)


class RawEvidenceScanTests(unittest.TestCase):
    def _values(self, samples: int, channels: int) -> np.ndarray:
        rng = np.random.default_rng(2008)
        values = rng.integers(-30_000, 30_000, size=(samples, channels), dtype=np.int16)
        # Exercise the candidate-screening invariant without deciding whether a
        # repeated hash is a duplication.  Audit still performs raw equality.
        values[1_200:1_280] = values[600:680]
        return values

    def test_fused_scan_matches_existing_feature_and_coarse_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fs, samples, channels = 4_000, 3_211, 12
            values = self._values(samples, channels)
            values.tofile(root / "amplifier.dat")
            recording = _recording(root, samples, channels, fs)

            baseline_feature = root / "baseline_feature.f32"
            baseline_coarse = root / "baseline_coarse.f32"
            build_common_mode_feature(
                recording,
                baseline_feature,
                highpass_hz=200.0,
                chunk_seconds=0.17,
            )
            _, baseline_factor = build_coarse_feature(
                baseline_feature,
                samples,
                baseline_coarse,
                fs=fs,
                target_rate_hz=160.0,
                chunk_seconds=0.11,
            )

            result = build_raw_evidence_scan(
                recording,
                root / "fused_feature.f32",
                root / "fused_coarse.f32",
                root / "fused_hashes.u64",
                highpass_hz=200.0,
                coarse_target_rate_hz=160.0,
                chunk_seconds=0.29,
            )

            audit_options = RawAuditOptions(
                max_duplication_lag_seconds=0.25,
                chunk_samples=701,
                validation_batch_samples=97,
            )
            baseline_audit = scan_exact_duplications(recording, audit_options)
            cached_audit = scan_exact_duplications(
                recording,
                audit_options,
                frame_hash_path=result.frame_hash_path,
            )
            self.assertEqual(cached_audit, baseline_audit)

            self.assertEqual(result.raw_amplifier_passes, 1)
            self.assertEqual(result.input_bytes_read, recording.amplifier_file.stat().st_size)
            self.assertEqual(result.coarse_downsample_factor, baseline_factor)
            self.assertEqual(
                result.output_bytes_written,
                sum(path.stat().st_size for path in (result.feature_path, result.coarse_feature_path, result.frame_hash_path)),
            )
            fused_feature = feature_memmap(result.feature_path, samples)
            fused_coarse = feature_memmap(
                result.coarse_feature_path,
                (samples + baseline_factor - 1) // baseline_factor,
            )
            baseline_full = feature_memmap(baseline_feature, samples)
            baseline_downsampled = feature_memmap(
                baseline_coarse,
                (samples + baseline_factor - 1) // baseline_factor,
            )
            hashes = frame_hash_memmap(result.frame_hash_path, samples)
            try:
                # Fused coarse filtering consumes the persisted float32 full-rate
                # values, so only normal float32/filter arithmetic tolerance is
                # allowed relative to the existing two-stage implementation.
                np.testing.assert_allclose(fused_feature, baseline_full, rtol=0.0, atol=1e-5)
                np.testing.assert_allclose(fused_coarse, baseline_downsampled, rtol=0.0, atol=1e-5)
                np.testing.assert_array_equal(hashes, _reference_hashes(values))
                np.testing.assert_array_equal(hashes[1_200:1_280], hashes[600:680])
            finally:
                for mapped in (fused_feature, fused_coarse, baseline_full, baseline_downsampled, hashes):
                    mapped._mmap.close()

    def test_fused_products_are_deterministic_across_chunk_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fs, samples, channels = 4_000, 4_123, 8
            self._values(samples, channels).tofile(root / "amplifier.dat")
            recording = _recording(root, samples, channels, fs)
            small = build_raw_evidence_scan(
                recording,
                root / "small_feature.f32",
                root / "small_coarse.f32",
                root / "small_hashes.u64",
                highpass_hz=200.0,
                coarse_target_rate_hz=125.0,
                chunk_seconds=0.031,
            )
            large = build_raw_evidence_scan(
                recording,
                root / "large_feature.f32",
                root / "large_coarse.f32",
                root / "large_hashes.u64",
                highpass_hz=200.0,
                coarse_target_rate_hz=125.0,
                chunk_seconds=0.407,
            )

            self.assertEqual(small.coarse_downsample_factor, large.coarse_downsample_factor)
            self.assertEqual(small.feature_path.read_bytes(), large.feature_path.read_bytes())
            self.assertEqual(small.coarse_feature_path.read_bytes(), large.coarse_feature_path.read_bytes())
            self.assertEqual(small.frame_hash_path.read_bytes(), large.frame_hash_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
