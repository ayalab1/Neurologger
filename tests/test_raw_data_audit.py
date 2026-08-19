from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1] / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from wild_preprocess.audit import RawAuditOptions, audit_session, scan_exact_duplications
from wild_preprocess.models import Recording


class RawDataAuditTest(unittest.TestCase):
    def _recording(self, root: Path, data: np.ndarray, *, fs: int = 1000) -> Recording:
        folder = root / "DEVICE" / "recording"
        folder.mkdir(parents=True)
        amplifier = folder / "amplifier.dat"
        data.astype("<i2").tofile(amplifier)
        requested_analog_samples = int(round(data.shape[0] * 1250 / fs))
        analog_samples = ((requested_analog_samples + 15) // 16) * 16
        analog = np.zeros((analog_samples, 16), dtype="<u2")
        analog[:, 11] = np.arange(analog_samples, dtype=np.uint16)
        analog.tofile(folder / "analogin.dat")
        (folder / "CE_params.bin").write_bytes(bytes(512))
        return Recording(
            folder=folder,
            amplifier_file=amplifier,
            analog_file=folder / "analogin.dat",
            ce_params_file=folder / "CE_params.bin",
            device_name="DEVICE",
            recording_name="recording",
            fs=fs,
            n_channels=data.shape[1],
            n_samples=data.shape[0],
            analog_channels=16,
            analog_samples=analog_samples,
        )

    def test_discovers_arbitrary_lags_and_short_fragmented_duplication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rng = np.random.default_rng(4)
            data = rng.integers(-30000, 30000, size=(6000, 16), dtype=np.int16)
            data[1000:1063] = data[713:776]
            data[3000:3030] = data[2700:2730]
            data[3015] = rng.integers(-30000, 30000, size=16, dtype=np.int16)
            recording = self._recording(Path(temporary), data)

            result = scan_exact_duplications(
                recording,
                RawAuditOptions(
                    max_duplication_lag_seconds=0.5,
                    merge_gap_samples=1,
                    chunk_samples=700,
                    max_parallel_workers=1,
                ),
            )

            by_lag = {item["lag_samples"]: item for item in result["lag_summary"]}
            self.assertEqual(set(by_lag), {287, 300})
            self.assertEqual(by_lag[287]["exact_match_samples"], 63)
            self.assertEqual(by_lag[300]["exact_match_samples"], 29)
            fragmented = next(item for item in result["episodes"] if item["lag_samples"] == 300)
            self.assertEqual(fragmented["span_samples"], 30)
            self.assertEqual(fragmented["exact_match_samples"], 29)
            self.assertEqual(fragmented["exact_run_count"], 2)
            self.assertEqual(
                fragmented["exact_duplicate_fragments"], [[3000, 3015], [3016, 3030]]
            )
            self.assertEqual(result["exact_duplication_union_samples"], 92)
            self.assertEqual(result["episode_envelope_union_samples"], 93)
            self.assertEqual(result["short_duplication_episode_count_lt_100_samples"], 2)

    def test_full_channel_validation_rejects_screening_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rng = np.random.default_rng(8)
            data = rng.integers(-30000, 30000, size=(2000, 16), dtype=np.int16)
            screening = np.linspace(0, 15, 8, dtype=int)
            data[1200, screening] = data[900, screening]
            recording = self._recording(Path(temporary), data)

            result = scan_exact_duplications(
                recording,
                RawAuditOptions(
                    max_duplication_lag_seconds=0.5,
                    chunk_samples=500,
                    max_parallel_workers=1,
                ),
            )

            self.assertEqual(result["episode_count"], 0)
            self.assertEqual(result["exact_duplication_union_samples"], 0)

    def test_interposed_screening_collision_does_not_hide_real_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rng = np.random.default_rng(9)
            data = rng.integers(-30000, 30000, size=(2000, 16), dtype=np.int16)
            screening = np.linspace(0, 15, 8, dtype=int)
            data[1200] = data[900]
            data[1000, screening] = data[900, screening]
            recording = self._recording(Path(temporary), data)

            result = scan_exact_duplications(
                recording,
                RawAuditOptions(
                    max_duplication_lag_seconds=0.5,
                    chunk_samples=500,
                    max_parallel_workers=1,
                ),
            )

            self.assertEqual(result["exact_duplication_union_samples"], 1)
            self.assertEqual(result["lag_summary"][0]["lag_samples"], 300)

    def test_session_audit_is_read_only_and_writes_one_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(12)
            data = rng.integers(-30000, 30000, size=(2000, 16), dtype=np.int16)
            recording = self._recording(root, data)
            before = {
                path: (path.stat().st_size, path.stat().st_mtime_ns)
                for path in (recording.amplifier_file, recording.analog_file)
            }
            output = root / "wild_raw_data_audit.json"

            result = audit_session(
                [recording],
                output_path=output,
                options=RawAuditOptions(
                    max_duplication_lag_seconds=0.25,
                    chunk_samples=500,
                    max_parallel_workers=1,
                ),
            )

            self.assertEqual(result, output.resolve())
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "wild_preprocess.raw-audit.v2")
            self.assertTrue(payload["read_only"])
            self.assertEqual(len(payload["devices"]), 1)
            self.assertEqual(payload["devices"][0]["exact_duplication"]["episode_count"], 0)
            self.assertEqual(payload["summary"]["devices"][0]["exact_duplication_samples"], 0)
            self.assertEqual(
                before,
                {
                    path: (path.stat().st_size, path.stat().st_mtime_ns)
                    for path in (recording.amplifier_file, recording.analog_file)
                },
            )


if __name__ == "__main__":
    unittest.main()
