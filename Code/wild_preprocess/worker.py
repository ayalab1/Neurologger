from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wild_preprocess.models import SyncOptions
from wild_preprocess.pipeline import run_multidevice_sync
from wild_preprocess.pc_time import (
    PcTimeOptions,
    ce_recording_start_issue,
    read_ce_params_hint,
    resolve_recording_start_ms,
    validate_recording_start_compatibility,
)


_LAST_PROGRESS_STAGE: str | None = None
_LAST_PROGRESS_PERCENT = -1.0
WORKER_JOB_SCHEMA_VERSION = 3
ASSUMED_SLAVE_START_SOURCE = "assumed simultaneous with master"
_BOOLEAN_JOB_FIELDS = {
    "allow_folder_name_start_fallback": False,
    "overwrite": False,
    "merge": True,
    "integrity_duplication_scan": True,
    "write_event_files": False,
    "process_imu": False,
}


def _progress(stage: str, percent: float) -> None:
    """Emit percentage completed within the named stage, not a global estimate."""

    global _LAST_PROGRESS_STAGE, _LAST_PROGRESS_PERCENT
    normalized = max(0.0, min(100.0, float(percent)))
    if stage != _LAST_PROGRESS_STAGE:
        _LAST_PROGRESS_STAGE = stage
        _LAST_PROGRESS_PERCENT = -1.0
    normalized = max(_LAST_PROGRESS_PERCENT, normalized)
    if normalized == _LAST_PROGRESS_PERCENT:
        return
    if normalized < 100.0 and normalized - _LAST_PROGRESS_PERCENT < 0.1:
        return
    _LAST_PROGRESS_PERCENT = normalized
    print(f"WILD_PROGRESS:{stage}:{normalized:.3f}", flush=True)


def _job_boolean(job: dict[str, object], field: str) -> bool:
    value = job.get(field, _BOOLEAN_JOB_FIELDS[field])
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _resolved_recording_start_anchor(
    folder: Path,
    *,
    explicit_recording_start_ms: int | None,
    allow_folder_name_fallback: bool,
) -> dict[str, object]:
    hint = read_ce_params_hint(folder)
    start_ms, source = resolve_recording_start_ms(
        folder,
        explicit_recording_start_ms=explicit_recording_start_ms,
        allow_folder_name_fallback=allow_folder_name_fallback,
    )
    issue = ce_recording_start_issue(hint)
    if source == "CE_params.bin":
        if issue is not None:
            raise ValueError(issue)
    recording_date = hint.recording_date
    if issue is not None:
        recording_date = None
    return {
        "milliseconds_since_midnight": start_ms,
        "source": source,
        "recording_date": recording_date,
        "recording_date_source": "CE_params.bin" if recording_date is not None else None,
    }


def _assumed_slave_start_anchor(
    master_anchor: dict[str, object],
    folder: Path,
    reason: str,
) -> dict[str, object]:
    try:
        reported = read_ce_params_hint(folder)
    except OSError:
        reported_ms = None
        reported_date = None
    else:
        reported_ms = reported.recording_start_ms
        reported_date = reported.recording_date
    return {
        "milliseconds_since_midnight": int(master_anchor["milliseconds_since_midnight"]),
        "source": ASSUMED_SLAVE_START_SOURCE,
        "recording_date": master_anchor.get("recording_date"),
        "recording_date_source": ASSUMED_SLAVE_START_SOURCE,
        "reported_ce_milliseconds_since_midnight": reported_ms,
        "reported_ce_recording_date": reported_date,
        "assumption_reason": reason,
    }


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
    boolean_values = {
        field: _job_boolean(job, field) for field in _BOOLEAN_JOB_FIELDS
    }
    allow_folder_name_fallback = boolean_values["allow_folder_name_start_fallback"]
    run_id = job.get("run_id")
    if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
        raise ValueError("run_id must be a non-empty string")
    master_folder = folders[master_index - 1]
    try:
        master_anchor = _resolved_recording_start_anchor(
            master_folder,
            explicit_recording_start_ms=(int(explicit) if explicit is not None else None),
            allow_folder_name_fallback=allow_folder_name_fallback,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"{master_folder}: master recording start is unavailable: {exc}") from exc

    anchors: list[dict[str, object]] = []
    for index, folder in enumerate(folders, start=1):
        if index == master_index:
            anchors.append(master_anchor)
            continue
        try:
            candidate = _resolved_recording_start_anchor(
                folder,
                explicit_recording_start_ms=None,
                allow_folder_name_fallback=allow_folder_name_fallback,
            )
            validate_recording_start_compatibility(
                [
                    (
                        int(master_anchor["milliseconds_since_midnight"]),
                        str(master_anchor["recording_date"])
                        if master_anchor["recording_date"] is not None
                        else None,
                    ),
                    (
                        int(candidate["milliseconds_since_midnight"]),
                        str(candidate["recording_date"])
                        if candidate["recording_date"] is not None
                        else None,
                    ),
                ]
            )
        except OSError as exc:
            raise ValueError(f"{folder}: cannot read slave recording metadata: {exc}") from exc
        except ValueError as exc:
            anchors.append(_assumed_slave_start_anchor(master_anchor, folder, str(exc)))
        else:
            anchors.append(candidate)

    master_pair = (
        int(master_anchor["milliseconds_since_midnight"]),
        str(master_anchor["recording_date"])
        if master_anchor["recording_date"] is not None
        else None,
    )
    for index, anchor in enumerate(anchors, start=1):
        if index == master_index:
            continue
        validate_recording_start_compatibility(
            [
                master_pair,
                (
                    int(anchor["milliseconds_since_midnight"]),
                    str(anchor["recording_date"])
                    if anchor["recording_date"] is not None
                    else None,
                ),
            ]
        )
    return folders, probes, anchors


def run_job(job_path: Path) -> int:
    global _LAST_PROGRESS_STAGE, _LAST_PROGRESS_PERCENT
    _LAST_PROGRESS_STAGE = None
    _LAST_PROGRESS_PERCENT = -1.0
    job = json.loads(job_path.read_text(encoding="utf-8"))
    folders, probe_indices, recording_start_anchors = _validated_job(job)
    boolean_values = {
        field: _job_boolean(job, field) for field in _BOOLEAN_JOB_FIELDS
    }
    allow_folder_name_fallback = boolean_values["allow_folder_name_start_fallback"]
    options = SyncOptions(**job.get("sync_options", {}))
    pc_time_options = PcTimeOptions(**job.get("pc_time_options", {}))
    result = run_multidevice_sync(
        folders,
        master_index=int(job["master_index"]) - 1,
        output_folder=Path(job["output_folder"]),
        overwrite=boolean_values["overwrite"],
        merge=boolean_values["merge"],
        options=options,
        progress=_progress,
        native_pc_time=True,
        recording_start_ms=job.get("recording_start_ms"),
        allow_folder_name_start_fallback=allow_folder_name_fallback,
        pc_time_options=pc_time_options,
        probe_indices=probe_indices,
        recording_start_anchors=recording_start_anchors,
        integrity_duplication_scan=boolean_values["integrity_duplication_scan"],
        write_event_files=boolean_values["write_event_files"],
        process_imu=boolean_values["process_imu"],
        run_id=(str(job["run_id"]) if job.get("run_id") is not None else None),
    )
    sync_status = result.outputs.get("sync_status", result.status)
    merge_status = result.outputs.get("merge_status", "NOT_RUN")
    analog_status = result.outputs.get("analog_status", "NOT_RUN")
    pc_time_status = result.outputs.get("pc_time_status", "NOT_RUN")
    imu_status = result.outputs.get("imu_status", "NOT_RUN")
    overall_status = result.outputs.get("overall_status", "FAIL" if result.status == "FAIL" else "COMPLETE")
    print(f"sync_status={sync_status}")
    print(f"merge_status={merge_status}")
    print(f"analog_status={analog_status}")
    print(f"pc_time_status={pc_time_status}")
    print(f"imu_status={imu_status}")
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
    if sync_status == "WARN" or merge_status == "WARN" or analog_status == "WARN":
        print(
            "Python outputs were published with localized synchronization or merge warnings; "
            "confirmed missing/overwritten samples are zero-filled in validity files, while "
            "non-destructive alignment warnings are recorded in alignment_quality.dat."
        )
    if overall_status == "MERGE_ONLY":
        print("Python sync and merge complete with a PC-time warning; pc_time.dat was not published.")
        return 0
    if pc_time_status == "WARN":
        print(
            "Python native pc_time.dat was published from the fitted clock with a QC warning; "
            "review pc_time_fit_summary.png and the manifest diagnostics."
        )
    imu_text = (
        ", and synchronized IMU"
        if imu_status == "OK"
        else ", and synchronized IMU with localized warnings"
        if imu_status == "WARN"
        else ""
    )
    print(f"Python multi-device sync, merge, native PC-time{imu_text} generation complete.")
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
