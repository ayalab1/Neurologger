from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from wild_preprocess.models import SyncOptions
from wild_preprocess.pipeline import run_multidevice_sync
from wild_preprocess.version import RUN_MANIFEST_SCHEMA_VERSION


def _write_recording(folder: Path, values: np.ndarray, *, fs: int) -> None:
    folder.mkdir(parents=True)
    n_channels = 4
    header = bytearray(512)
    struct.pack_into("<I", header, 0, fs)
    struct.pack_into("<I", header, 8, n_channels)
    header[440] = 0
    (folder / "CE_params.bin").write_bytes(header)
    np.column_stack([values + channel for channel in range(n_channels)]).astype("<i2").tofile(
        folder / "amplifier.dat"
    )
    analog = np.zeros((round(values.size * 1250 / fs), 1), dtype="<i2")
    analog[:, 0] = (np.arange(analog.shape[0]) // 20 % 2).astype(np.int16)
    analog.tofile(folder / "analogin.dat")


class PipelineTimingTest(unittest.TestCase):
    def _case(self, root: Path) -> tuple[list[Path], SyncOptions]:
        fs = 20_000
        rng = np.random.default_rng(103)
        values = np.rint(rng.normal(scale=300.0, size=100_000)).astype(np.int16)
        folders = [root / f"device{index}" / "recording" for index in range(2)]
        for folder in folders:
            _write_recording(folder, values, fs=fs)
        return folders, SyncOptions(
            initial_start_seconds=0.1,
            initial_duration_seconds=0.5,
            initial_max_lag_seconds=0.01,
            window_seconds=0.5,
            step_seconds=0.25,
            tracking_max_lag_samples=30,
            highpass_hz=200.0,
            peak_exclusion_samples=8,
            min_peak_margin_fraction=0.005,
            max_model_rms_samples=5.0,
            max_model_residual_samples=15.0,
            chunk_seconds=1.0,
            max_parallel_workers=1,
        )

    def _performance(self, manifest: dict[str, object]) -> dict[str, dict[str, object]]:
        payload = manifest["performance"]
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["clock"], "time.perf_counter")
        self.assertGreaterEqual(payload["total_wall_seconds"], 0.0)
        self.assertIn("input_bytes", payload)
        self.assertIn("output_bytes", payload)
        self.assertIn("storage", payload)
        self.assertEqual(payload["workers"]["requested_max_parallel_workers"], 1)
        stages = {stage["name"]: stage for stage in payload["stages"]}
        for stage in stages.values():
            self.assertGreaterEqual(stage["wall_seconds"], 0.0)
            self.assertGreaterEqual(stage["invocations"], 1)
        return stages

    def test_success_sync_only_and_failed_attempt_publish_timing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folders, options = self._case(root)

            success_output = root / "success"
            success = run_multidevice_sync(
                folders,
                master_index=0,
                output_folder=success_output,
                merge=True,
                options=options,
            )
            self.assertNotEqual(success.status, "FAIL")
            manifest = json.loads((success_output / "wild_preprocess_run.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], RUN_MANIFEST_SCHEMA_VERSION)
            stages = self._performance(manifest)
            self.assertTrue(
                {
                    "input_inspection",
                    "raw_evidence_feature_scan",
                    "coarse_correlation",
                    "full_rate_refinement",
                    "attribution_segment_construction",
                    "ephys_merge",
                    "analog_merge",
                    "postmerge_validation",
                    "inspection_figure_generation",
                }.issubset(stages)
            )
            self.assertEqual(
                stages["raw_evidence_feature_scan"]["bytes_read"],
                manifest["performance"]["input_bytes"]["amplifier.dat"],
            )
            self.assertEqual(
                stages["ephys_merge"]["bytes_written"],
                (success_output / "amplifier.dat").stat().st_size
                + (success_output / "valid_samples.dat").stat().st_size,
            )
            self.assertEqual(
                stages["analog_merge"]["bytes_written"],
                (success_output / "analogin.dat").stat().st_size,
            )
            self.assertEqual(
                manifest["performance"]["output_bytes"]["actual"]["amplifier.dat"],
                (success_output / "amplifier.dat").stat().st_size,
            )
            self.assertNotIn("performance.json", manifest["managed_files"])

            sync_only = run_multidevice_sync(
                folders,
                master_index=0,
                output_folder=root / "sync_only",
                merge=False,
                options=options,
            )
            sync_only_manifest = json.loads(
                Path(sync_only.outputs["run_manifest"]).read_text(encoding="utf-8")
            )
            sync_only_stages = self._performance(sync_only_manifest)
            self.assertNotIn("ephys_merge", sync_only_stages)
            self.assertNotIn("analog_merge", sync_only_stages)

            failed = run_multidevice_sync(
                folders,
                master_index=0,
                output_folder=root / "failed",
                merge=False,
                options=SyncOptions(**{**options.__dict__, "min_peak_correlation": 1.1}),
            )
            self.assertEqual(failed.status, "WARN")
            failed_manifest = json.loads(
                Path(failed.outputs["run_manifest"]).read_text(encoding="utf-8")
            )
            failed_stages = self._performance(failed_manifest)
            self.assertIn("coarse_correlation", failed_stages)
            self.assertIn("full_rate_refinement", failed_stages)


if __name__ == "__main__":
    unittest.main()
