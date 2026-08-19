from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from wild_preprocess import worker


class WorkerProgressTest(unittest.TestCase):
    def setUp(self) -> None:
        worker._LAST_PROGRESS_STAGE = None
        worker._LAST_PROGRESS_PERCENT = -1.0

    def test_progress_is_local_to_each_stage(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            worker._progress("build_features", 80.0)
            worker._progress("sync_pairs", 5.0)
        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "WILD_PROGRESS:build_features:80.000",
                "WILD_PROGRESS:sync_pairs:5.000",
            ],
        )

    def test_progress_is_monotone_clamped_and_deduplicated_within_stage(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            worker._progress("write_ephys", -5.0)
            worker._progress("write_ephys", 0.04)
            worker._progress("write_ephys", 20.0)
            worker._progress("write_ephys", 10.0)
            worker._progress("write_ephys", 200.0)
            worker._progress("write_ephys", 100.0)
        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "WILD_PROGRESS:write_ephys:0.000",
                "WILD_PROGRESS:write_ephys:20.000",
                "WILD_PROGRESS:write_ephys:100.000",
            ],
        )

    def test_run_job_threads_explicit_folder_fallback_policy(self) -> None:
        anchors = [
            {
                "milliseconds_since_midnight": 1,
                "source": "CE_params.bin",
                "recording_date": "2026-08-11",
                "recording_date_source": "CE_params.bin",
            },
            {
                "milliseconds_since_midnight": 2,
                "source": "CE_params.bin",
                "recording_date": "2026-08-11",
                "recording_date_source": "CE_params.bin",
            },
        ]
        result = SimpleNamespace(
            status="WARN",
            outputs={"overall_status": "MERGE_ONLY"},
            pairs=[],
            unresolved_gap_messages=[],
        )
        with tempfile.TemporaryDirectory() as temporary:
            job_path = Path(temporary) / "job.json"
            job_path.write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "device_folders": ["first", "second"],
                        "probe_indices": [1, 2],
                        "master_index": 1,
                        "output_folder": temporary,
                        "allow_folder_name_start_fallback": True,
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(
                    worker,
                    "_validated_job",
                    return_value=([Path("first"), Path("second")], [1, 2], anchors),
                ),
                patch.object(worker, "run_multidevice_sync", return_value=result) as run,
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(worker.run_job(job_path), 0)
        self.assertIs(run.call_args.kwargs["allow_folder_name_start_fallback"], True)


if __name__ == "__main__":
    unittest.main()
