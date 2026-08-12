from __future__ import annotations

import json
import re
import sys
import traceback
from datetime import date
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wild_preprocess.models import SyncOptions
from wild_preprocess.pipeline import run_multidevice_sync
from wild_preprocess.pc_time import PcTimeOptions, resolve_recording_start_ms


_LAST_PROGRESS = -1.0
WORKER_JOB_SCHEMA_VERSION = 3


def _recording_date_from_folder(folder: Path) -> tuple[str | None, str | None]:
    match = re.search(r"(?:^|_)(\d{8})(?:_|$)", folder.name)
    if match is None:
        return None, None
    text = match.group(1)
    try:
        parsed = date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    except ValueError:
        return None, None
    return parsed.isoformat(), "recording folder name"


def _progress(stage: str, percent: float) -> None:
    global _LAST_PROGRESS
    ranges = {
        "build_features": (0.0, 25.0),
        "sync_pairs": (25.0, 55.0),
        "integrity_scan": (55.0, 60.0),
        "write_ephys": (60.0, 90.0),
        "write_analog": (90.0, 99.0),
    }
    start, end = ranges.get(stage, (0.0, 100.0))
    scaled = start + (end - start) * max(0.0, min(100.0, percent)) / 100.0
    if scaled < 100.0 and scaled - _LAST_PROGRESS < 0.1:
        return
    _LAST_PROGRESS = max(_LAST_PROGRESS, scaled)
    print(f"WILD_PROGRESS:{stage}:{_LAST_PROGRESS:.3f}", flush=True)


def _validated_job(job: dict[str, object]) -> tuple[list[Path], list[int], list[dict[str, object]]]:
    """Validate the GUI job before feature extraction or output mutation."""

    if job.get("schema_version") != WORKER_JOB_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported worker job schema {job.get('schema_version')!r}; "
            f"expected {WORKER_JOB_SCHEMA_VERSION}."
        )
    folders_value = job.get("device_folders")
    if not isinstance(folders_value, list) or len(folders_value) < 2:
        raise ValueError("worker job requires at least two device_folders")
    folders = [Path(str(folder)).resolve() for folder in folders_value]
    if len(set(folders)) != len(folders):
        raise ValueError("worker job selected duplicate recording folders")
    probes_value = job.get("probe_indices")
    if not isinstance(probes_value, list):
        raise ValueError("worker job requires probe_indices")
    probes = [int(value) for value in probes_value]
    if sorted(probes) != list(range(1, len(folders) + 1)):
        raise ValueError("worker job probe_indices must be unique 1..N")
    master_index = int(job["master_index"])
    if master_index < 1 or master_index > len(folders):
        raise ValueError("worker job master_index is outside selected device_folders")
    explicit = job.get("recording_start_ms")
    anchors: list[dict[str, object]] = []
    for index, folder in enumerate(folders, start=1):
        start_ms, source = resolve_recording_start_ms(
            folder,
            explicit_recording_start_ms=(int(explicit) if explicit is not None and index == master_index else None),
        )
        recording_date, date_source = _recording_date_from_folder(folder)
        anchors.append(
            {
                "milliseconds_since_midnight": start_ms,
                "source": source,
                "recording_date": recording_date,
                "recording_date_source": date_source,
            }
        )
    for left in range(len(anchors)):
        for right in range(left + 1, len(anchors)):
            first = int(anchors[left]["milliseconds_since_midnight"])
            second = int(anchors[right]["milliseconds_since_midnight"])
            difference = abs((second - first + 43_200_000) % 86_400_000 - 43_200_000)
            if difference > 30_000:
                raise ValueError(
                    f"selected recording starts differ by {difference / 1000:.3f}s "
                    f"(devices {left + 1} and {right + 1}; limit 30s)"
                )
    return folders, probes, anchors


def run_job(job_path: Path) -> int:
    global _LAST_PROGRESS
    _LAST_PROGRESS = -1.0
    job = json.loads(job_path.read_text(encoding="utf-8"))
    folders, probe_indices, recording_start_anchors = _validated_job(job)
    options = SyncOptions(**job.get("sync_options", {}))
    pc_time_options = PcTimeOptions(**job.get("pc_time_options", {}))
    result = run_multidevice_sync(
        folders,
        master_index=int(job["master_index"]) - 1,
        output_folder=Path(job["output_folder"]),
        overwrite=bool(job.get("overwrite", False)),
        merge=bool(job.get("merge", True)),
        options=options,
        progress=_progress,
        native_pc_time=True,
        recording_start_ms=job.get("recording_start_ms"),
        pc_time_options=pc_time_options,
        probe_indices=probe_indices,
        recording_start_anchors=recording_start_anchors,
        integrity_duplication_scan=bool(job.get("integrity_duplication_scan", True)),
        write_event_files=bool(job.get("write_event_files", False)),
    )
    sync_status = result.outputs.get("sync_status", result.status)
    merge_status = result.outputs.get("merge_status", "NOT_RUN")
    pc_time_status = result.outputs.get("pc_time_status", "NOT_RUN")
    overall_status = result.outputs.get("overall_status", "FAIL" if result.status == "FAIL" else "COMPLETE")
    print(f"sync_status={sync_status}")
    print(f"merge_status={merge_status}")
    print(f"pc_time_status={pc_time_status}")
    print(f"overall_status={overall_status}")
    for pair in result.pairs:
        print(
            f"pair={pair.master_index}->{pair.slave_index} status={pair.status} "
            f"offset={pair.model.intercept_samples:.3f} drift_ppm={pair.model.drift_ppm:.3f} "
            f"rms_samples={pair.model.residual_rms_samples:.3f} message={pair.message}"
        )
    for message in result.unresolved_gap_messages:
        print(f"unresolved_gap={message}")
    if overall_status == "FAIL":
        print("Python sync QC failed; merged DAT files were not written.", file=sys.stderr)
        return 2
    _progress("complete", 100.0)
    if sync_status == "WARN" or merge_status == "WARN":
        print(
            "Python outputs were published with localized synchronization or merge warnings; "
            "affected samples are zero-filled and marked invalid in valid_samples.dat."
        )
    if overall_status == "MERGE_ONLY":
        print("Python sync and merge complete with a PC-time warning; pc_time.dat was not published.")
        return 0
    print("Python multi-device sync, merge, and native PC-time generation complete.")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: worker.py <job.json>", file=sys.stderr)
        return 64
    try:
        return run_job(Path(argv[0]))
    except Exception as exc:
        print(f"Python preprocessing backend failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
