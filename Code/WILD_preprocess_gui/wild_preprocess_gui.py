#!/usr/bin/env python3
"""WILD preprocessing GUI.

The UI remains a thin process orchestrator. Multi-device synchronization uses
the Python backend, with the legacy MATLAB backend available through
``WILD_SYNC_BACKEND=matlab`` during migration.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import struct
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from wild_preprocess.analog import imu_capacity_issue
from wild_preprocess.pc_time import (
    align_pc_time_file,
    read_ce_params_hint,
    validate_recording_start_compatibility,
)

PC_TIME_SCRIPT = CODE_ROOT / "WILD_generate_pc_time.py"
PYTHON_BACKEND_WORKER = CODE_ROOT / "wild_preprocess" / "worker.py"
SYNC_BACKEND = os.environ.get("WILD_SYNC_BACKEND", "python").strip().lower()
DAY_MS = 24 * 60 * 60 * 1000
PIPELINE_PROGRESS_STAGES: tuple[tuple[str, str], ...] = (
    ("inspect_inputs", "Inspect inputs"),
    ("build_features", "Build synchronization features"),
    ("sync_pairs", "Synchronize loggers"),
    ("refine_sync", "Refine timing models"),
    ("integrity_scan", "Scan data integrity"),
    ("write_ephys", "Write neural data"),
    ("write_analog", "Write analog data"),
    ("postmerge_qc", "Validate merged data"),
    ("pc_time", "Fit PC time"),
    ("imu", "Process synchronized IMU"),
    ("inspection", "Build QC report"),
    ("publish", "Publish outputs"),
)
_PIPELINE_PROGRESS_LABELS = {
    stage: (index, label)
    for index, (stage, label) in enumerate(PIPELINE_PROGRESS_STAGES, start=1)
}


def stage_progress_text(stage: str, percent: float) -> str:
    """Return a stable user-facing label for stage-local progress."""

    value = max(0, min(100, int(round(percent))))
    if stage == "complete":
        return "Complete - 100%"
    if stage in _PIPELINE_PROGRESS_LABELS:
        index, label = _PIPELINE_PROGRESS_LABELS[stage]
        return f"Step {index}/{len(PIPELINE_PROGRESS_STAGES)} - {label} - {value}%"
    return f"{stage.replace('_', ' ').strip().title()} - {value}%"


def _python_worker_job(
    selected: list[RecordingInfo],
    *,
    master_index: int,
    output_folder: Path,
    overwrite: bool,
) -> dict[str, Any]:
    """Build a traceable GUI job and request IMU only when it is supportable."""

    process_imu, _reason = _imu_job_decision(selected)

    return {
        "schema_version": 3,
        "run_id": uuid.uuid4().hex,
        "device_folders": [str(rec.folder) for rec in selected],
        "probe_indices": [int(rec.probe_index) for rec in selected],
        "master_index": master_index,
        "output_folder": str(output_folder),
        "overwrite": overwrite,
        "merge": True,
        "sync_options": {"chunk_seconds": 5.0},
        "pc_time_options": {},
        "allow_folder_name_start_fallback": False,
        "process_imu": process_imu,
    }


def _imu_job_decision(selected: list[RecordingInfo]) -> tuple[bool, str]:
    if not selected:
        return False, "no selected recordings"
    if any(rec.n_channels != 64 for rec in selected):
        return False, "requires the WILD 64-channel/16-lane raw analog layout"
    durations = [rec.duration_sec for rec in selected]
    if any(duration is None or duration <= 0 for duration in durations):
        return False, "recording duration is unavailable"
    canonical_rows = int(math.ceil(min(float(duration) for duration in durations) * 1_250.0)) + 1
    issue = imu_capacity_issue(canonical_rows, len(selected))
    if issue is not None:
        return False, issue
    return True, "supported"


@dataclass
class RecordingInfo:
    use: bool
    role: str
    probe_index: int
    device_id: str
    recording_name: str
    folder: Path
    fs: int | None
    n_channels: int | None
    n_samples: int | None
    duration_sec: float | None
    time_dat_valid: bool
    info_rhd_exists: bool
    imu_mat_exists: bool
    pc_time_exists: bool
    pc_time_valid: bool
    pc_time_status: str = "missing"
    sync_qc: str = "not run"
    notes: str = ""
    pc_time_metrics: dict[str, str] = field(default_factory=dict)

    @property
    def amplifier_file(self) -> Path:
        return self.folder / "amplifier.dat"

    @property
    def analog_file(self) -> Path:
        return self.folder / "analogin.dat"

    @property
    def ce_params_file(self) -> Path:
        return self.folder / "CE_params.bin"

    @property
    def pc_time_file(self) -> Path:
        return self.folder / "pc_time.dat"

    def expected_time_bytes(self) -> int | None:
        if self.n_samples is None:
            return None
        return int(self.n_samples) * 4


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _infer_device_and_recording(folder: Path) -> tuple[str, str]:
    recording_name = folder.name
    device_id = folder.parent.name if folder.parent != folder else ""
    return device_id, recording_name


def _read_ce_params_header(path: Path) -> tuple[int | None, int | None]:
    try:
        data = path.read_bytes()
    except OSError:
        return None, None
    if len(data) < 441:
        return None, None
    try:
        fs = struct.unpack_from("<I", data, 0)[0]
        data_version = data[440]
        if data_version == 0:
            n_channels = struct.unpack_from("<I", data, 8)[0]
        else:
            n_channels = struct.unpack_from("<H", data, 8)[0]
    except struct.error:
        return None, None
    if fs <= 0:
        fs = None
    if n_channels <= 0:
        n_channels = None
    return fs, n_channels


def _file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _write_json_object_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _published_python_run_summary(
    manifest: dict[str, Any] | None,
    *,
    expected_run_id: str | None = None,
) -> tuple[bool, str]:
    """Classify the Python manifest without mistaking published WARN for failure."""

    if not manifest:
        return False, "run manifest unavailable"
    if expected_run_id is not None and manifest.get("run_id") != expected_run_id:
        return False, "run manifest belongs to a different worker run"
    overall = str(manifest.get("overall_status", "")).upper()
    sync = str(manifest.get("sync_status", "")).upper()
    merge = str(manifest.get("merge_status", "")).upper()
    analog = str(manifest.get("analog_status", "")).upper()
    pc_time = str(manifest.get("pc_time_status", "")).upper()
    imu = str(manifest.get("imu_status", "")).upper()
    if overall not in {"COMPLETE", "MERGE_ONLY"}:
        return False, f"not published ({overall or 'unknown'})"
    warning_components = [
        name
        for name, value in (
            ("sync", sync),
            ("merge", merge),
            ("analog", analog),
            ("PC-time", pc_time),
            ("IMU", imu),
        )
        if value == "WARN"
    ]
    if overall == "MERGE_ONLY" and "PC-time" not in warning_components:
        warning_components.append("PC-time")
    if warning_components:
        return True, "published with warnings: " + ", ".join(warning_components)
    return True, "published"


def _pc_time_status(recording: RecordingInfo) -> str:
    if not recording.pc_time_exists:
        return "missing"
    if recording.pc_time_valid:
        return "ok"
    return "invalid"


def basepath_pc_time_path(basepath: Path) -> Path:
    return basepath / "pc_time.dat"


def basepath_pc_time_summary_path(basepath: Path) -> Path:
    return basepath / "pc_time_fit_summary.jpg"


def basepath_pc_time_qc_path(basepath: Path) -> Path:
    return basepath / "pc_time_qc.json"


def basepath_sync_qc_path(basepath: Path) -> Path:
    if SYNC_BACKEND == "matlab":
        return basepath / "wild_multilogger_sync_qc.mat"
    return basepath / "wild_preprocess_run.json"


def pc_time_valid_for_recording(path: Path, recording: RecordingInfo) -> bool:
    run_manifest = path.parent / "wild_preprocess_run.json"
    merge_info = path.parent / "wild_multilogger_mergeInfo.json"
    merged_time = path.parent / "time.dat"
    if SYNC_BACKEND != "matlab" and run_manifest.exists() and merged_time.exists():
        expected = _file_size(merged_time)
        manifest = _read_json_object(run_manifest)
        if not manifest:
            return False
        merge_payload = manifest.get("merge")
        pc_payload = manifest.get("pc_time")
        if not isinstance(merge_payload, dict) or not isinstance(pc_payload, dict):
            return False
        if pc_payload.get("merge_run_id") != manifest.get("run_id"):
            return False
        pc_status = str(pc_payload.get("status", "")).lower()
        published_status = pc_status == "ok" or (
            pc_status in {"warn", "warning"} and pc_payload.get("published") is True
        )
        if not published_status or not pc_payload.get("aligned_to_merge", False):
            return False
        try:
            provenance_matches = (
                int(pc_payload["common_start_master_sample"])
                == int(merge_payload["common_start_master_sample"])
                and int(pc_payload["n_samples"]) == int(merge_payload["n_samples"])
            )
        except (KeyError, TypeError, ValueError):
            return False
        if not provenance_matches:
            return False
    elif merge_info.exists() and merged_time.exists():
        expected = _file_size(merged_time)
        merge_payload = _read_json_object(merge_info)
        if merge_payload and merge_payload.get("run_id"):
            pc_payload = _read_json_object(basepath_pc_time_qc_path(path.parent))
            if not pc_payload or pc_payload.get("merge_run_id") != merge_payload["run_id"]:
                return False
            pc_status = str(pc_payload.get("status", "")).lower()
            published_status = pc_status == "ok" or (
                pc_status in {"warn", "warning"} and pc_payload.get("published") is True
            )
            if not published_status or not pc_payload.get("aligned_to_merge", False):
                return False
            try:
                provenance_matches = (
                    int(pc_payload["common_start_master_sample"])
                    == int(merge_payload["common_start_master_sample"])
                    and int(pc_payload["n_samples"]) == int(merge_payload["n_samples"])
                )
            except (KeyError, TypeError, ValueError):
                return False
            if not provenance_matches:
                return False
    else:
        expected = recording.expected_time_bytes()
    return expected is not None and path.exists() and _file_size(path) == expected


def discover_recordings(basepath: Path) -> list[RecordingInfo]:
    basepath = basepath.expanduser()
    if not basepath.exists():
        raise FileNotFoundError(f"WILD recording folder does not exist: {basepath}")
    amp_files = sorted(basepath.rglob("amplifier.dat"))
    recordings: list[RecordingInfo] = []
    seen: set[Path] = set()
    for amp_file in amp_files:
        folder = amp_file.parent
        if folder.resolve() == basepath.resolve():
            continue
        if folder in seen:
            continue
        seen.add(folder)
        ce_params = folder / "CE_params.bin"
        if not ce_params.exists():
            continue
        fs, n_channels = _read_ce_params_header(ce_params)
        amp_bytes = _file_size(amp_file)
        n_samples: int | None = None
        duration_sec: float | None = None
        if amp_bytes is not None and n_channels:
            n_samples = amp_bytes // (2 * n_channels)
            if fs:
                duration_sec = (n_samples - 1) / fs if n_samples > 0 else 0.0
        time_dat = folder / "time.dat"
        info_rhd = folder / "info.rhd"
        imu_mat = folder / "IMU.mat"
        pc_time = folder / "pc_time.dat"
        expected_time = n_samples * 4 if n_samples is not None else None
        time_valid = expected_time is not None and time_dat.exists() and _file_size(time_dat) == expected_time
        pc_time_valid = expected_time is not None and pc_time.exists() and _file_size(pc_time) == expected_time
        device_id, recording_name = _infer_device_and_recording(folder)
        recordings.append(
            RecordingInfo(
                use=False,
                role="ignore",
                probe_index=0,
                device_id=device_id,
                recording_name=recording_name,
                folder=folder,
                fs=fs,
                n_channels=n_channels,
                n_samples=n_samples,
                duration_sec=duration_sec,
                time_dat_valid=time_valid,
                info_rhd_exists=info_rhd.exists(),
                imu_mat_exists=imu_mat.exists(),
                pc_time_exists=pc_time.exists(),
                pc_time_valid=pc_time_valid,
            )
        )
    auto_select_longest(recordings)
    for rec in recordings:
        rec.pc_time_status = _pc_time_status(rec)
    return recordings


def auto_select_longest(recordings: list[RecordingInfo]) -> None:
    by_device: dict[str, list[RecordingInfo]] = {}
    for rec in recordings:
        by_device.setdefault(rec.device_id, []).append(rec)
        rec.use = False
        rec.role = "ignore"
        rec.probe_index = 0
    selected = []
    for device_records in by_device.values():
        best = max(device_records, key=lambda item: item.duration_sec or -1)
        best.use = True
        best.role = "slave"
        selected.append(best)
    selected.sort(key=lambda item: item.device_id)
    for idx, rec in enumerate(selected, start=1):
        rec.probe_index = idx


def parse_pc_time_stdout(text: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        metrics[key.strip()] = value.strip()
    return metrics


def classify_pc_time(metrics: dict[str, str], return_code: int) -> tuple[str, str]:
    if return_code != 0:
        return "failed", "pc_time command failed"
    updates = metrics.get("updates_kept", "")
    residual = _float_metric(metrics, "residual_rms_ms")
    drift = _float_metric(metrics, "drift_ppm")
    warning_reasons: list[str] = []
    if "/" in updates:
        kept_text, total_text = updates.split("/", 1)
        try:
            kept = float(kept_text)
            total = float(total_text)
            if total > 0 and kept / total < 0.5:
                warning_reasons.append(f"kept updates low ({updates})")
        except ValueError:
            warning_reasons.append(f"could not parse updates ({updates})")
    if residual is None:
        warning_reasons.append("missing residual")
    elif residual > 250.0:
        warning_reasons.append(f"residual RMS {residual:.1f} ms")
    if drift is not None and abs(drift) > 1000.0:
        warning_reasons.append(f"drift {drift:.1f} ppm")
    if warning_reasons:
        return "warning", "; ".join(warning_reasons)
    return "ok", "pc_time generated"


def _float_metric(metrics: dict[str, str], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def manifest_rows(recordings: list[RecordingInfo], basepath: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in recordings:
        rows.append(
            {
                "use": int(rec.use),
                "role": rec.role,
                "probe_index": rec.probe_index,
                "device_id": rec.device_id,
                "recording_name": rec.recording_name,
                "recording_folder": _safe_rel(rec.folder, basepath),
                "fs": rec.fs or "",
                "n_channels": rec.n_channels or "",
                "n_samples": rec.n_samples or "",
                "duration_sec": f"{rec.duration_sec:.3f}" if rec.duration_sec is not None else "",
                "time_dat_valid": int(rec.time_dat_valid),
                "info_rhd_exists": int(rec.info_rhd_exists),
                "imu_mat_exists": int(rec.imu_mat_exists),
                "pc_time_exists": int(rec.pc_time_exists),
                "pc_time_valid": int(rec.pc_time_valid),
                "pc_time_status": rec.pc_time_status,
                "sync_qc": rec.sync_qc,
                "notes": rec.notes,
                "folder_abs": str(rec.folder),
            }
        )
    return rows


def save_manifest(path: Path, recordings: list[RecordingInfo], basepath: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = manifest_rows(recordings, basepath)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def preflight_checks(recordings: list[RecordingInfo], basepath: Path | None = None) -> list[tuple[str, str, str]]:
    checks: list[tuple[str, str, str]] = []
    selected = [rec for rec in recordings if rec.use]
    masters = [rec for rec in selected if rec.role == "master"]
    checks.append(("OK" if selected else "FAIL", "selected recordings", f"{len(selected)} selected"))
    checks.append(("OK" if len(masters) == 1 else "FAIL", "master", f"{len(masters)} selected"))
    by_device: dict[str, int] = {}
    for rec in selected:
        by_device[rec.device_id] = by_device.get(rec.device_id, 0) + 1
    repeated = [device for device, count in by_device.items() if count > 1]
    checks.append(("OK" if not repeated else "FAIL", "one recording per device", ", ".join(repeated) or "unique"))
    start_anchors: list[tuple[int, str | None]] = []
    for rec in selected:
        status = "OK"
        details: list[str] = []
        if not rec.amplifier_file.exists():
            status = "FAIL"
            details.append("missing amplifier.dat")
        if not rec.analog_file.exists():
            status = "FAIL"
            details.append("missing analogin.dat")
        if not rec.ce_params_file.exists():
            status = "FAIL"
            details.append("missing CE_params.bin")
        else:
            try:
                start_hint = read_ce_params_hint(rec.folder)
            except OSError as exc:
                status = "FAIL"
                details.append(f"cannot read CE_params.bin: {exc}")
            else:
                if start_hint.recording_start_ms is None or start_hint.recording_date is None:
                    status = "FAIL"
                    details.append("invalid/missing CE RTC date/time at bytes 332-355")
                else:
                    start_anchors.append(
                        (start_hint.recording_start_ms, start_hint.recording_date)
                    )
        if rec.fs is None:
            status = "FAIL"
            details.append("invalid/missing CE ephys sample rate")
        if rec.n_channels is None:
            status = "FAIL"
            details.append("invalid/missing CE channel count")
        elif rec.n_channels % 4 != 0:
            status = "FAIL"
            details.append("CE channel count is not divisible by four")
        amp_bytes = _file_size(rec.amplifier_file)
        if amp_bytes is not None and rec.n_channels and amp_bytes % (2 * rec.n_channels) != 0:
            status = "FAIL"
            details.append("amplifier size not divisible by channels")
        analog_bytes = _file_size(rec.analog_file)
        if (
            analog_bytes is not None
            and rec.n_channels
            and rec.n_channels % 4 == 0
            and analog_bytes % (2 * (rec.n_channels // 4)) != 0
        ):
            status = "FAIL"
            details.append("analog size not divisible by inferred channels")
        if rec.duration_sec is not None and rec.duration_sec < 10:
            status = "WARN" if status == "OK" else status
            details.append(f"short duration {rec.duration_sec:.1f}s")
        checks.append((status, f"{rec.device_id}/{rec.recording_name}", "; ".join(details) or "ready"))
    if len(start_anchors) == len(selected) and len(start_anchors) >= 2:
        try:
            validate_recording_start_compatibility(start_anchors)
        except ValueError as exc:
            checks.append(("FAIL", "recording start compatibility", str(exc)))
        else:
            checks.append(("OK", "recording start compatibility", "within 30s using CE date/time"))
    if selected:
        process_imu, imu_reason = _imu_job_decision(selected)
        checks.append(
            (
                "OK" if process_imu else "WARN",
                "synchronized IMU",
                "will be generated" if process_imu else f"not requested: {imu_reason}",
            )
        )
    invalid_time = [rec for rec in selected if not rec.time_dat_valid]
    if invalid_time:
        checks.append((
            "INFO",
            "time.dat optional",
            f"{len(invalid_time)} missing/invalid; optional for pc-time alignment and regenerated by WILD_PreProcess",
        ))
    if masters:
        master = masters[0]
        pc_time_path = basepath_pc_time_path(basepath) if basepath is not None else master.pc_time_file
        if pc_time_valid_for_recording(pc_time_path, master):
            checks.append(("OK", "selected-folder pc_time.dat", f"valid: {pc_time_path}"))
        elif pc_time_path.exists():
            checks.append(("WARN", "selected-folder pc_time.dat", f"exists but size does not match master; regenerate: {pc_time_path}"))
        else:
            checks.append(("WARN", "selected-folder pc_time.dat", f"missing; generate after logger sync QC, before behavior/camera/gpio alignment: {pc_time_path}"))
    durations = [rec.duration_sec for rec in selected if rec.duration_sec is not None]
    if len(durations) > 1:
        spread = max(durations) - min(durations)
        checks.append(("WARN" if spread > 10 else "OK", "duration spread", f"{spread:.3f} sec"))
    if len(selected) >= 2:
        sync_qc_path = basepath_sync_qc_path(basepath) if basepath is not None else Path("wild_multilogger_sync_qc.mat")
        sync_status = "OK" if sync_qc_path.exists() else "WARN"
        sync_detail = f"{'found' if sync_qc_path.exists() else 'not run yet'}: {sync_qc_path}"
        if basepath is not None and SYNC_BACKEND != "matlab":
            manifest = _read_json_object(sync_qc_path)
            if manifest:
                recorded_status = str(manifest.get("sync_status", "WARN")).upper()
                sync_status = recorded_status if recorded_status in {"OK", "WARN", "FAIL"} else "WARN"
                sync_detail = (
                    f"{manifest.get('overall_status', 'unknown')} run "
                    f"{manifest.get('run_id', 'unknown')}: {sync_qc_path}"
                )
            elif sync_qc_path.exists():
                sync_status = "WARN"
                sync_detail = f"invalid run manifest; rerun synchronization: {sync_qc_path}"
        checks.append((
            sync_status,
            "logger sync QC",
            sync_detail,
        ))
    else:
        checks.append((
            "INFO",
            "logger sync QC",
            "select at least 2 loggers to run master-vs-slave sync QC",
        ))
    return checks


def has_ready_check_failure(recordings: list[RecordingInfo], basepath: Path | None = None) -> bool:
    return any(status == "FAIL" for status, _check, _detail in preflight_checks(recordings, basepath))


def main() -> int:
    try:
        from PySide6.QtCore import QProcess, Qt, QTimer
        from PySide6.QtGui import QFont, QTextCursor
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QFileDialog,
            QFrame,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QHeaderView,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPlainTextEdit,
            QProgressBar,
            QPushButton,
            QSpinBox,
            QSplitter,
            QTableWidget,
            QTableWidgetItem,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        print("PySide6 is required to run the GUI. Install it with: python -m pip install PySide6", file=sys.stderr)
        print(f"Import error: {exc}", file=sys.stderr)
        return 1

    class WildPreprocessWindow(QMainWindow):
        columns = [
            "device_id",
            "use",
            "role",
            "probe_index",
            "recording",
            "fs",
            "nch",
            "duration",
            "time.dat",
            "pc_time",
            "sync_qc",
            "notes",
        ]

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("WILD Preprocess GUI")
            self.resize(1320, 820)
            self.recordings: list[RecordingInfo] = []
            self.basepath = Path.cwd()
            self._pc_time_process: QProcess | None = None
            self._backend_process: QProcess | None = None
            self._backend_job_path: Path | None = None
            self._backend_run_id: str | None = None
            self._pc_time_raw_path: Path | None = None
            self._pc_time_stdout = ""
            self._continue_after_pc_time = False
            self._continue_after_sync_qc = False
            self._progress_re = re.compile(r"WILD_PROGRESS:([^:]+):([0-9.]+)")
            self._log_buffer = ""
            self._log_timer = QTimer(self)
            self._log_timer.setInterval(80)
            self._log_timer.setSingleShot(True)
            self._log_timer.timeout.connect(self._flush_log)
            self._apply_dark_theme()
            self._build_ui()

        def _apply_dark_theme(self) -> None:
            self.setStyleSheet(
                """
                QMainWindow, QWidget#rootWidget { background: #252525; }
                QWidget { color: #e5e5e5; font-size: 9pt; }
                QGroupBox {
                    background: #2f2f2f; border: 0; border-radius: 5px;
                    margin-top: 18px; padding: 12px; font-weight: 650;
                }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #f5f5f5; }
                QLabel { color: #d4d4d4; background: transparent; }
                QLabel#miniHead { background: #1f1f1f; color: #f5f5f5; font-weight: 700; padding: 7px 10px; border-bottom: 1px solid #444444; }
                QLineEdit, QPlainTextEdit, QComboBox {
                    background: #383838; color: #f0f0f0; border: 1px solid #555555;
                    border-radius: 5px; padding: 5px 7px; selection-background-color: #606060;
                }
                QPlainTextEdit { font-family: Consolas, ui-monospace, monospace; font-size: 9pt; }
                QPushButton {
                    background: #3a3a3a; color: #f0f0f0; border: 1px solid #555555;
                    border-radius: 5px; padding: 7px 11px;
                }
                QPushButton:hover { background: #464646; border-color: #707070; }
                QPushButton:disabled { color: #737373; border-color: #2a2a2a; background: #101010; }
                QPushButton#primaryButton { background: #3f3f3f; border-color: #707070; color: #ffffff; }
                QPushButton#dangerButton { color: #ffb088; border-color: #8a3a16; }
                QFrame#miniPanel { background: #2f2f2f; border: 0; border-radius: 6px; }
                QTableWidget { background: #303030; color: #e5e5e5; gridline-color: #505050; border: 1px solid #444444; }
                QHeaderView::section { background: #3a3a3a; color: #f5f5f5; border: 1px solid #505050; padding: 5px; }
                QTabWidget::pane { border: 0; background: #252525; }
                QTabBar::tab { background: #1f1f1f; color: #a3a3a3; border: 1px solid #444444; padding: 8px 12px; }
                QTabBar::tab:selected { background: #111111; color: #f5f5f5; border-top: 2px solid #ef4b2d; }
                QCheckBox { color: #d4d4d4; spacing: 8px; background: transparent; min-height: 22px; }
                QProgressBar {
                    background: #303030; color: #f0f0f0; border: 1px solid #555555;
                    border-radius: 5px; text-align: center; min-height: 18px;
                }
                QProgressBar::chunk { background: #ef4b2d; border-radius: 4px; }
                """
            )

        def _build_ui(self) -> None:
            root = QWidget()
            root.setObjectName("rootWidget")
            layout = QVBoxLayout(root)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(8)
            layout.addWidget(self._build_top_bar())
            splitter = QSplitter(Qt.Orientation.Horizontal)
            splitter.addWidget(self._build_controls_panel())
            splitter.addWidget(self._build_center_panel())
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes([420, 900])
            layout.addWidget(splitter, 1)
            self.setCentralWidget(root)

        def _build_top_bar(self) -> QWidget:
            panel = QWidget()
            layout = QGridLayout(panel)
            layout.setContentsMargins(0, 0, 0, 0)
            self.basepath_field = QLineEdit()
            self.basepath_field.setPlaceholderText("Select WILD subepoch / recording folder")
            browse = QPushButton("Browse WILD recording")
            browse.clicked.connect(self._browse_basepath)
            rescan = QPushButton("Rescan")
            rescan.clicked.connect(self._scan_basepath)
            layout.addWidget(browse, 0, 0)
            layout.addWidget(self.basepath_field, 0, 1)
            layout.addWidget(rescan, 0, 2)
            layout.setColumnStretch(1, 1)
            return panel

        def _build_controls_panel(self) -> QWidget:
            panel = QWidget()
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(8)

            session = QGroupBox("Output")
            session_form = QGridLayout(session)
            output_note = QLabel("outputs are written to the selected WILD recording folder")
            output_note.setWordWrap(True)
            self.overwrite_outputs = QCheckBox("overwrite generated outputs")
            session_form.addWidget(output_note, 0, 0, 1, 2)
            session_form.addWidget(self.overwrite_outputs, 1, 0, 1, 2)

            run = QGroupBox("Run")
            run_grid = QGridLayout(run)
            self.progress_label = QLabel("Idle")
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            run_selected = QPushButton("Run")
            run_selected.setObjectName("primaryButton")
            run_selected.clicked.connect(self._run_selected_steps)
            force_stop = QPushButton("Force stop")
            force_stop.setObjectName("dangerButton")
            force_stop.clicked.connect(self._force_stop)
            run_grid.addWidget(self.progress_label, 0, 0, 1, 2)
            run_grid.addWidget(self.progress_bar, 1, 0, 1, 2)
            run_grid.addWidget(run_selected, 2, 0)
            run_grid.addWidget(force_stop, 2, 1)

            layout.addWidget(session)
            layout.addWidget(run)
            layout.addStretch(1)
            panel.setMinimumWidth(410)
            return panel

        def _build_center_panel(self) -> QWidget:
            panel = QWidget()
            layout = QHBoxLayout(panel)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(8)
            left = QWidget()
            left_layout = QVBoxLayout(left)
            left_layout.setContentsMargins(0, 0, 0, 0)
            left_layout.addWidget(self._mini_panel("Discovered WILD recordings - edit use / role / probe index", self._build_recording_table()), 5)
            left_layout.addWidget(self._mini_panel("Ready Check", self._build_preflight_table()), 2)
            right = QWidget()
            right_layout = QVBoxLayout(right)
            right_layout.setContentsMargins(0, 0, 0, 0)
            self.log = QPlainTextEdit()
            self.log.setReadOnly(True)
            right_layout.addWidget(self._log_panel(), 1)
            layout.addWidget(left, 5)
            layout.addWidget(right, 2)
            return panel

        def _log_panel(self) -> QWidget:
            frame = QFrame()
            frame.setObjectName("miniPanel")
            layout = QVBoxLayout(frame)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            head = QLabel("Log")
            head.setObjectName("miniHead")
            controls = QWidget()
            controls_layout = QHBoxLayout(controls)
            controls_layout.setContentsMargins(6, 6, 6, 6)
            clear = QPushButton("Clear log")
            clear.clicked.connect(self.log.clear)
            controls_layout.addStretch(1)
            controls_layout.addWidget(clear)
            layout.addWidget(head)
            layout.addWidget(self.log, 1)
            layout.addWidget(controls)
            return frame

        def _mini_panel(self, title: str, widget: QWidget) -> QWidget:
            frame = QFrame()
            frame.setObjectName("miniPanel")
            layout = QVBoxLayout(frame)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            head = QLabel(title)
            head.setObjectName("miniHead")
            layout.addWidget(head)
            layout.addWidget(widget, 1)
            return frame

        def _build_recording_table(self) -> QWidget:
            self.recording_table = QTableWidget(0, len(self.columns))
            self.recording_table.setHorizontalHeaderLabels(self.columns)
            self.recording_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            self.recording_table.horizontalHeader().setStretchLastSection(True)
            self.recording_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            self.recording_table.setAlternatingRowColors(True)
            self.recording_table.itemChanged.connect(self._table_item_changed)
            return self.recording_table

        def _build_preflight_table(self) -> QWidget:
            self.preflight_table = QTableWidget(0, 3)
            self.preflight_table.setHorizontalHeaderLabels(["status", "check", "detail"])
            self.preflight_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            self.preflight_table.horizontalHeader().setStretchLastSection(True)
            return self.preflight_table

        def _browse_basepath(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "Select WILD subepoch / recording folder", str(self.basepath))
            if not path:
                return
            self.basepath_field.setText(path)
            self._scan_basepath()

        def _scan_basepath(self) -> None:
            text = self.basepath_field.text().strip()
            if not text:
                QMessageBox.warning(self, "Missing WILD recording", "Select a WILD subepoch / recording folder first.")
                return
            self.basepath = Path(text)
            try:
                self.recordings = discover_recordings(self.basepath)
            except Exception as exc:
                QMessageBox.critical(self, "Scan failed", str(exc))
                return
            self._append_log(f"Discovered {len(self.recordings)} logger recording folder(s) inside {self.basepath}")
            self._refresh_recording_table()
            self._refresh_preview()
            self._run_preflight()

        def _refresh_recording_table(self) -> None:
            self.recording_table.blockSignals(True)
            self.recording_table.setRowCount(len(self.recordings))
            for row, rec in enumerate(self.recordings):
                self.recording_table.setCellWidget(row, 1, self._use_checkbox(row, rec.use))
                self.recording_table.setCellWidget(row, 2, self._role_combo(row, rec.role))
                self.recording_table.setCellWidget(row, 3, self._probe_spin(row, rec.probe_index))
                self.recording_table.setCellWidget(row, 11, self._notes_field(row, rec.notes))
                values = [
                    rec.device_id,
                    "",
                    "",
                    "",
                    rec.recording_name,
                    str(rec.fs or ""),
                    str(rec.n_channels or ""),
                    f"{rec.duration_sec:.3f}" if rec.duration_sec is not None else "",
                    self._time_dat_display(rec),
                    self._pc_time_display(rec),
                    rec.sync_qc,
                    "",
                ]
                for col, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.recording_table.setItem(row, col, item)
            self.recording_table.blockSignals(False)

        def _pc_time_display(self, rec: RecordingInfo) -> str:
            if rec.role != "master":
                return "via master" if rec.use else "n/a"
            path = basepath_pc_time_path(self.basepath)
            if pc_time_valid_for_recording(path, rec):
                return "ok"
            if path.exists():
                return "invalid"
            return "missing"

        def _time_dat_display(self, rec: RecordingInfo) -> str:
            if rec.time_dat_valid:
                return "ok"
            time_path = rec.folder / "time.dat"
            if time_path.exists():
                return "exists, size mismatch"
            return "missing"

        def _use_checkbox(self, row: int, checked: bool) -> QCheckBox:
            widget = QCheckBox()
            widget.setChecked(checked)
            widget.setToolTip("Use this recording for preprocessing.")
            widget.stateChanged.connect(lambda state, r=row: self._set_recording_use(r, state == Qt.CheckState.Checked.value))
            return widget

        def _role_combo(self, row: int, role: str) -> QComboBox:
            widget = QComboBox()
            widget.addItems(["ignore", "slave", "master"])
            widget.setCurrentText(role if role in {"ignore", "slave", "master"} else "ignore")
            widget.setToolTip("Choose master/slave role. Only one master is allowed.")
            widget.currentTextChanged.connect(lambda text, r=row: self._set_recording_role(r, text))
            return widget

        def _probe_spin(self, row: int, value: int) -> QSpinBox:
            widget = QSpinBox()
            widget.setRange(0, 999)
            widget.setValue(max(0, int(value)))
            widget.setToolTip("Probe order index. Brain region mapping happens later.")
            widget.valueChanged.connect(lambda value, r=row: self._set_probe_index(r, value))
            return widget

        def _notes_field(self, row: int, text: str) -> QLineEdit:
            widget = QLineEdit(text)
            widget.setPlaceholderText("optional")
            widget.textChanged.connect(lambda value, r=row: self._set_notes(r, value))
            return widget

        def _set_recording_use(self, row: int, checked: bool) -> None:
            if row >= len(self.recordings):
                return
            rec = self.recordings[row]
            rec.use = checked
            if checked and rec.role == "ignore":
                rec.role = "slave"
            elif not checked:
                rec.role = "ignore"
            self._refresh_recording_table()
            self._refresh_preview()
            self._run_preflight()

        def _set_recording_role(self, row: int, role: str) -> None:
            if row >= len(self.recordings):
                return
            rec = self.recordings[row]
            rec.role = role
            rec.use = role != "ignore"
            if role == "master":
                for idx, other in enumerate(self.recordings):
                    if idx != row and other.role == "master":
                        other.role = "slave" if other.use else "ignore"
            self._refresh_recording_table()
            self._refresh_preview()
            self._run_preflight()

        def _set_probe_index(self, row: int, value: int) -> None:
            if row >= len(self.recordings):
                return
            self.recordings[row].probe_index = int(value)
            self._refresh_preview()

        def _set_notes(self, row: int, value: str) -> None:
            if row >= len(self.recordings):
                return
            self.recordings[row].notes = value
            self._refresh_preview()

        def _table_item_changed(self, item: QTableWidgetItem) -> None:
            row = item.row()
            col = item.column()
            if row >= len(self.recordings):
                return
            rec = self.recordings[row]
            text = item.text().strip()
            if col == 1:
                rec.use = text.lower() in {"1", "true", "yes", "y", "use", "x"}
                if rec.use and rec.role == "ignore":
                    rec.role = "slave"
            elif col == 2:
                rec.role = text.lower() if text.lower() in {"master", "slave", "ignore"} else rec.role
                rec.use = rec.role != "ignore"
                if rec.role == "master":
                    for idx, other in enumerate(self.recordings):
                        if idx != row and other.role == "master":
                            other.role = "slave" if other.use else "ignore"
            elif col == 3:
                try:
                    rec.probe_index = int(text)
                except ValueError:
                    pass
            elif col == 11:
                rec.notes = text
            self._refresh_recording_table()
            self._refresh_preview()

        def _auto_select_longest(self) -> None:
            auto_select_longest(self.recordings)
            self._refresh_recording_table()
            self._refresh_preview()
            self._run_preflight()

        def _selected_rows(self) -> list[int]:
            return sorted({idx.row() for idx in self.recording_table.selectedIndexes()})

        def _set_selected_master(self) -> None:
            rows = self._selected_rows()
            if not rows:
                return
            selected = rows[0]
            for idx, rec in enumerate(self.recordings):
                if idx == selected:
                    rec.use = True
                    rec.role = "master"
                elif rec.role == "master":
                    rec.role = "slave" if rec.use else "ignore"
            self._refresh_recording_table()
            self._refresh_preview()
            self._run_preflight()

        def _ignore_selected(self) -> None:
            for row in self._selected_rows():
                self.recordings[row].use = False
                self.recordings[row].role = "ignore"
            self._refresh_recording_table()
            self._refresh_preview()
            self._run_preflight()

        def _renumber_probe_index(self) -> None:
            selected = [rec for rec in self.recordings if rec.use]
            selected.sort(key=lambda rec: (rec.probe_index == 0, rec.probe_index, rec.device_id, rec.recording_name))
            for idx, rec in enumerate(selected, start=1):
                rec.probe_index = idx
            self._refresh_recording_table()
            self._refresh_preview()

        def _run_selected_steps(self) -> None:
            if self._backend_process is not None or self._pc_time_process is not None:
                self._append_log("Blocked: a preprocessing process is already running.")
                return
            self._run_preflight()
            selected = self._selected_recordings_for_run()
            if has_ready_check_failure(self.recordings, self.basepath):
                self._append_log("Blocked: Ready Check has FAIL items. Resolve them before running.")
                return
            if not selected:
                self._append_log("Blocked: no selected recordings.")
                return
            self._continue_after_pc_time = len(selected) == 2 and SYNC_BACKEND == "matlab"
            # The Python worker owns native PC-time generation.  Only the
            # explicit MATLAB fallback continues into the legacy script.
            self._continue_after_sync_qc = SYNC_BACKEND == "matlab"
            if SYNC_BACKEND == "matlab":
                pipeline_text = "logger synchronization check, master pc_time.dat generation, then legacy 2-logger merge"
            else:
                pipeline_text = (
                    "Python logger synchronization, merge, post-merge QC, native master "
                    "pc_time.dat generation, and synchronized IMU"
                )
            self._append_log("Run pipeline: " + pipeline_text + ".")
            self._set_progress("Starting", 0)
            if len(selected) >= 2:
                self._start_logger_sync_qc()
                return
            self._generate_master_pc_time()

        def _run_after_pc_time(self) -> None:
            self._continue_after_pc_time = False
            selected = self._selected_recordings_for_run()
            if len(selected) == 2 and SYNC_BACKEND == "matlab":
                self._start_documented_matlab_workflow()
            elif len(selected) > 2:
                self._append_log("Run complete: multi-logger amplifier.dat/analogin.dat and master pc_time.dat are ready for review.")
            else:
                self._append_log("Run complete.")

        def _selected_recordings_for_run(self) -> list[RecordingInfo]:
            selected = [rec for rec in self.recordings if rec.use]
            return sorted(selected, key=lambda rec: (rec.probe_index == 0, rec.probe_index, rec.device_id, rec.recording_name))

        def _start_documented_matlab_workflow(self) -> None:
            selected = self._selected_recordings_for_run()
            if len(selected) != 2:
                QMessageBox.critical(
                    self,
                    "2-logger merge unavailable",
                    "WILD_channelMerger/WILD_PreProcess_Multi currently supports exactly 2 selected loggers. "
                    "For 3+ loggers, stop here and define a documented alignment method before concatenation.",
                )
                self._append_log("Blocked: WILD_channelMerger merge path supports exactly 2 selected loggers.")
                return
            master_count = sum(1 for rec in selected if rec.role == "master")
            if master_count != 1:
                self._append_log("Blocked: exactly one selected logger must be marked master.")
                return
            master_index = next(idx for idx, rec in enumerate(selected, start=1) if rec.role == "master")
            commands = []
            code_root = str(CODE_ROOT).replace("'", "''")
            out_folder = str(self.basepath).replace("'", "''")
            folders = "; ".join([f"'{str(rec.folder).replace(chr(39), chr(39)+chr(39))}'" for rec in selected])
            overwrite = "true" if self.overwrite_outputs.isChecked() else "false"
            commands.append(
                "addpath('{code_root}'); "
                "deviceFolders = {{{folders}}}; "
                "result = WILD_PreProcess_Multi(deviceFolders, "
                "'MasterIndex',{master_index}, "
                "'OutputFolder','{out_folder}', "
                "'MergeMethod','channelMerger', "
                "'Overwrite',{overwrite}, "
                "'RunPreprocess',false);".format(
                    code_root=code_root,
                    folders=folders,
                    master_index=master_index,
                    out_folder=out_folder,
                    overwrite=overwrite,
                )
            )
            commands.append(
                "disp('2-logger WILD_channelMerger merge complete. Review WILD_channelMerger figure and outputs before promoting merged files to session-level amplifier.dat.');"
            )
            if not commands:
                return
            command_text = " ".join(commands)
            command = ["matlab", "-batch", command_text]
            self._append_log("Starting MATLAB 2-logger merge.")
            self._append_log(f"  master: {selected[master_index - 1].device_id}/{selected[master_index - 1].recording_name}")
            self._append_log(f"  output: {self.basepath}")
            process = QProcess(self)
            self._backend_process = process
            process.setProgram(command[0])
            process.setArguments(command[1:])
            process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            process.readyReadStandardOutput.connect(lambda: self._read_process_output(process))
            process.errorOccurred.connect(self._backend_process_error)
            process.finished.connect(lambda code, _status: self._backend_finished(code))
            process.start()

        def _start_logger_sync_qc(self) -> None:
            selected = self._selected_recordings_for_run()
            if len(selected) < 2:
                QMessageBox.warning(self, "Logger sync QC unavailable", "Select at least 2 logger recordings.")
                self._continue_after_sync_qc = False
                return
            master_count = sum(1 for rec in selected if rec.role == "master")
            if master_count != 1:
                QMessageBox.warning(self, "No master", "Select exactly one master recording first.")
                self._continue_after_sync_qc = False
                return
            master_index = next(idx for idx, rec in enumerate(selected, start=1) if rec.role == "master")
            if SYNC_BACKEND != "matlab":
                self._start_python_sync_backend(selected, master_index)
                return
            code_root = str(CODE_ROOT).replace("'", "''")
            out_folder = str(self.basepath).replace("'", "''")
            folders = "; ".join([f"'{str(rec.folder).replace(chr(39), chr(39)+chr(39))}'" for rec in selected])
            command_text = (
                "addpath('{code_root}'); "
                "deviceFolders = {{{folders}}}; "
                "result = WILD_channelMerger(deviceFolders, {overwrite_numeric}, 0, "
                "'Mode','multiMerge', "
                "'MasterIndex',{master_index}, "
                "'OutputFolder','{out_folder}', "
                "'OutputPrefix','wild_multilogger_sync'); "
                "disp('Multi-logger merge complete. Review wild_multilogger_sync_qc.tsv and wild_multilogger_mergeInfo.mat.');"
            ).format(
                code_root=code_root,
                folders=folders,
                master_index=master_index,
                out_folder=out_folder,
                overwrite_numeric="1" if self.overwrite_outputs.isChecked() else "0",
            )
            command = ["matlab", "-batch", command_text]
            self._append_log("Starting MATLAB multi-logger merge.")
            self._append_log(f"  master: {selected[master_index - 1].device_id}/{selected[master_index - 1].recording_name}")
            for idx, rec in enumerate(selected, start=1):
                if idx != master_index:
                    self._append_log(f"  slave:  {rec.device_id}/{rec.recording_name}")
            self._append_log(f"  output: {self.basepath}")
            process = QProcess(self)
            self._backend_process = process
            process.setProgram(command[0])
            process.setArguments(command[1:])
            process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            process.readyReadStandardOutput.connect(lambda: self._read_process_output(process))
            process.errorOccurred.connect(self._backend_process_error)
            process.finished.connect(lambda code, _status: self._backend_finished(code))
            process.start()

        def _start_python_sync_backend(self, selected: list[RecordingInfo], master_index: int) -> None:
            if not PYTHON_BACKEND_WORKER.exists():
                self._continue_after_sync_qc = False
                QMessageBox.critical(self, "Python backend unavailable", f"Missing backend worker:\n{PYTHON_BACKEND_WORKER}")
                return
            job = _python_worker_job(
                selected,
                master_index=master_index,
                output_folder=self.basepath,
                overwrite=self.overwrite_outputs.isChecked(),
            )
            self._backend_run_id = str(job["run_id"])
            handle = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                prefix="wild_preprocess_job_",
                encoding="utf-8",
                delete=False,
            )
            try:
                json.dump(job, handle, indent=2)
            finally:
                handle.close()
            self._backend_job_path = Path(handle.name)
            process = QProcess(self)
            self._backend_process = process
            process.setProgram(sys.executable)
            process.setArguments([str(PYTHON_BACKEND_WORKER), str(self._backend_job_path)])
            process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            process.readyReadStandardOutput.connect(lambda: self._read_process_output(process))
            process.errorOccurred.connect(self._backend_process_error)
            process.finished.connect(lambda code, _status: self._backend_finished(code))
            self._append_log("Starting Python multi-logger synchronization backend.")
            self._append_log(f"  master: {selected[master_index - 1].device_id}/{selected[master_index - 1].recording_name}")
            for idx, rec in enumerate(selected, start=1):
                if idx != master_index:
                    self._append_log(f"  slave:  {rec.device_id}/{rec.recording_name}")
            self._append_log(f"  output: {self.basepath}")
            process.start()

        def _backend_process_error(self, error: QProcess.ProcessError) -> None:
            self._append_log(f"Preprocessing backend process error: {error}")
            if error == QProcess.ProcessError.FailedToStart and self._backend_process is not None:
                self._backend_finished(-1)

        def _backend_finished(self, return_code: int) -> None:
            is_matlab = SYNC_BACKEND == "matlab"
            expected_run_id = None if is_matlab else self._backend_run_id
            self._backend_run_id = None
            backend_label = "MATLAB" if is_matlab else "Python backend"
            self._append_log(f"{backend_label} finished with exit code {return_code}.")
            self._set_progress(
                (
                    f"{backend_label} finished"
                    if return_code == 0
                    else f"{backend_label} merge-only" if not is_matlab and return_code == 3 else f"{backend_label} failed"
                ),
                100 if return_code in {0, 3} else self.progress_bar.value(),
            )
            if return_code == 0:
                for rec in self._selected_recordings_for_run():
                    rec.sync_qc = "QC done; review outputs"
            self._backend_process = None
            if self._backend_job_path is not None:
                try:
                    self._backend_job_path.unlink(missing_ok=True)
                except OSError as exc:
                    self._append_log(f"Could not remove temporary backend job: {exc}")
                self._backend_job_path = None
            self._refresh_recording_table()
            self._run_preflight()
            # A Python worker is the complete multi-device pipeline.  Keep
            # this branch ahead of the legacy continuation so an accidental
            # continuation flag can never launch WILD_generate_pc_time.py.
            if not is_matlab:
                self._continue_after_sync_qc = False
                self._continue_after_pc_time = False
                manifest_path = self.basepath / "wild_preprocess_run.json"
                manifest = None
                if return_code == 0 and manifest_path.is_file():
                    try:
                        loaded_manifest = json.loads(
                            manifest_path.read_text(encoding="utf-8")
                        )
                        if not isinstance(loaded_manifest, dict):
                            raise ValueError("run manifest root is not an object")
                        manifest = loaded_manifest
                    except (OSError, ValueError) as exc:
                        self._append_log(f"Could not read Python run manifest: {exc}")
                if return_code == 0:
                    published, publication_detail = _published_python_run_summary(
                        manifest,
                        expected_run_id=expected_run_id,
                    )
                    if published:
                        assert manifest is not None
                        self._append_log(
                            "Python manifest: "
                            f"{manifest.get('overall_status', 'unknown')} "
                            f"(sync={manifest.get('sync_status', 'unknown')}, "
                            f"merge={manifest.get('merge_status', 'unknown')}, "
                            f"analog={manifest.get('analog_status', 'unknown')}, "
                            f"pc_time={manifest.get('pc_time_status', 'unknown')}, "
                            f"imu={manifest.get('imu_status', 'unknown')})."
                        )
                        qc_text = (
                            "Published with warnings; review valid_samples.dat and inspection figure"
                            if "warnings" in publication_detail
                            else "Published; review outputs"
                        )
                        for rec in self._selected_recordings_for_run():
                            rec.sync_qc = qc_text
                    else:
                        for rec in self._selected_recordings_for_run():
                            rec.sync_qc = "Current run was not published"
                        self._append_log(
                            "Python backend exited successfully but the current run cannot be "
                            f"verified as published: {publication_detail}."
                        )
                        return
                    if published and "warnings" in publication_detail:
                        self._append_log(
                            "Python run " + publication_detail + "; outputs remain available for review."
                        )
                    if manifest is not None and manifest.get("overall_status") == "MERGE_ONLY":
                        self._append_log(
                            "Python merge completed with a PC-time warning; "
                            "merged neural streams remain available, but pc_time.dat was not published."
                        )
                    elif manifest is not None and manifest.get("pc_time_status") == "WARN":
                        self._append_log(
                            "Python pc_time.dat was published from the fitted clock with a QC warning; "
                            "review pc_time_fit_summary.png and manifest diagnostics."
                        )
                    else:
                        self._append_log("Python run complete: merged streams and native pc_time.dat are ready for review.")
                elif return_code == 3:
                    self._append_log("Python merge completed with a PC-time warning; pc_time.dat was not published.")
                else:
                    self._append_log("Python run failed before publication; existing successful outputs were preserved.")
                return
            if self._continue_after_sync_qc:
                self._continue_after_sync_qc = False
                if return_code == 0:
                    self._generate_master_pc_time()
                else:
                    self._append_log("Skipping downstream steps because logger sync QC failed.")

        def _total_analog_channels_for_selected(self) -> int:
            total = 0
            for rec in self._selected_recordings_for_run():
                if rec.n_channels:
                    total += max(1, rec.n_channels // 4)
            return max(16, total)

        def _run_preflight(self) -> None:
            checks = preflight_checks(self.recordings, self.basepath)
            self.preflight_table.setRowCount(len(checks))
            for row, (status, check, detail) in enumerate(checks):
                for col, value in enumerate((status, check, detail)):
                    item = QTableWidgetItem(value)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.preflight_table.setItem(row, col, item)
            self._append_log(f"Ready Check: {sum(1 for s, _, _ in checks if s == 'FAIL')} fail, {sum(1 for s, _, _ in checks if s == 'WARN')} warn")

        def _refresh_preview(self) -> None:
            return

        def _master_recording(self) -> RecordingInfo | None:
            masters = [rec for rec in self.recordings if rec.use and rec.role == "master"]
            return masters[0] if len(masters) == 1 else None

        def _generate_master_pc_time(self) -> None:
            if self._pc_time_process is not None:
                self._append_log("Blocked: pc_time generation is already running.")
                return
            if SYNC_BACKEND != "matlab" and (self.basepath / "wild_preprocess_run.json").is_file():
                self._continue_after_pc_time = False
                message = (
                    "Standalone pc_time regeneration is disabled for Python pipeline outputs. "
                    "Run the full preprocessing pipeline so packed-clock publication uses the "
                    "canonical analog mapping and native scientific QC gates."
                )
                self._append_log("Blocked: " + message)
                QMessageBox.warning(self, "Use full preprocessing", message)
                return
            master = self._master_recording()
            if master is None:
                self._continue_after_pc_time = False
                QMessageBox.warning(self, "No master", "Select exactly one master recording first.")
                return
            if not master.analog_file.exists():
                self._continue_after_pc_time = False
                QMessageBox.warning(self, "Missing analogin.dat", f"Missing analogin.dat:\n{master.analog_file}")
                return
            output_path = basepath_pc_time_path(self.basepath)
            summary_path = basepath_pc_time_summary_path(self.basepath)
            generator_output = output_path
            if SYNC_BACKEND != "matlab" and (self.basepath / "wild_preprocess_run.json").exists():
                handle = tempfile.NamedTemporaryFile(prefix="wild_raw_pc_time_", suffix=".dat", delete=False)
                handle.close()
                self._pc_time_raw_path = Path(handle.name)
                self._pc_time_raw_path.unlink(missing_ok=True)
                generator_output = self._pc_time_raw_path
            command = [sys.executable, str(PC_TIME_SCRIPT), str(master.folder), "-o", str(generator_output)]
            if SYNC_BACKEND == "matlab":
                command.extend(["--summary-plot", str(summary_path)])
            if output_path.exists() and not self.overwrite_outputs.isChecked():
                answer = QMessageBox.question(self, "pc_time.dat exists", f"{output_path} already exists. Regenerate it?")
                if answer != QMessageBox.StandardButton.Yes:
                    should_continue = self._continue_after_pc_time
                    self._continue_after_pc_time = False
                    if should_continue:
                        self._run_after_pc_time()
                    return
            self._append_log("Starting master pc_time.dat generation.")
            self._append_log(f"  master: {master.device_id}/{master.recording_name}")
            self._append_log(f"  pc_time: {output_path}")
            if SYNC_BACKEND == "matlab":
                self._append_log(f"  summary: {summary_path}")
            process = QProcess(self)
            self._pc_time_process = process
            self._pc_time_stdout = ""
            process.setProgram(command[0])
            process.setArguments(command[1:])
            process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            process.readyReadStandardOutput.connect(lambda: self._read_pc_time_output(process))
            process.errorOccurred.connect(lambda error: self._pc_time_process_error(master, error))
            process.finished.connect(lambda code, _status: self._pc_time_finished(master, code))
            process.start()

        def _pc_time_process_error(self, master: RecordingInfo, error: QProcess.ProcessError) -> None:
            self._append_log(f"pc_time process error: {error}")
            if error == QProcess.ProcessError.FailedToStart and self._pc_time_process is not None:
                self._pc_time_finished(master, -1)

        def _read_pc_time_output(self, process: QProcess) -> None:
            chunk = bytes(process.readAllStandardOutput()).decode(errors="replace")
            self._pc_time_stdout += chunk
            self._append_log(chunk)

        def _read_process_output(self, process: QProcess) -> None:
            chunk = bytes(process.readAllStandardOutput()).decode(errors="replace")
            self._update_progress_from_text(chunk)
            cleaned = self._remove_progress_lines(chunk)
            if cleaned.strip():
                self._append_log(cleaned)

        def _pc_time_finished(self, master: RecordingInfo, return_code: int) -> None:
            metrics = parse_pc_time_stdout(self._pc_time_stdout)
            status, detail = classify_pc_time(metrics, return_code)
            output_path = basepath_pc_time_path(self.basepath)
            aligned_to_merge = False
            run_manifest_path = self.basepath / "wild_preprocess_run.json"
            run_manifest = _read_json_object(run_manifest_path)
            if return_code == 0 and self._pc_time_raw_path is not None:
                try:
                    if not run_manifest or not isinstance(run_manifest.get("merge"), dict):
                        raise ValueError("run manifest does not contain merge metadata")
                    merge_info = run_manifest["merge"]
                    align_pc_time_file(
                        self._pc_time_raw_path,
                        output_path,
                        common_start_master_sample=int(merge_info["common_start_master_sample"]),
                        n_samples=int(merge_info["n_samples"]),
                    )
                    if _file_size(output_path) != _file_size(self.basepath / "time.dat"):
                        raise ValueError("aligned pc_time.dat does not match merged time.dat length")
                    detail += "; sliced to merged common interval"
                    aligned_to_merge = True
                except Exception as exc:
                    return_code = 1
                    status = "failed"
                    detail = f"failed to align pc_time to merged interval: {exc}"
                finally:
                    self._pc_time_raw_path.unlink(missing_ok=True)
                    self._pc_time_raw_path = None
            elif self._pc_time_raw_path is not None:
                self._pc_time_raw_path.unlink(missing_ok=True)
                self._pc_time_raw_path = None
            master.pc_time_metrics = metrics
            master.pc_time_status = status
            master.sync_qc = detail
            qc_path = basepath_pc_time_qc_path(self.basepath)
            pc_payload: dict[str, Any] = {
                master.device_id: metrics,
                "status": status,
                "detail": detail,
            }
            merge_payload = (
                run_manifest.get("merge")
                if run_manifest and isinstance(run_manifest.get("merge"), dict)
                else _read_json_object(self.basepath / "wild_multilogger_mergeInfo.json")
            )
            if merge_payload and merge_payload.get("run_id"):
                pc_payload.update(
                    {
                        "merge_run_id": merge_payload["run_id"],
                        "aligned_to_merge": aligned_to_merge,
                        "common_start_master_sample": merge_payload.get("common_start_master_sample"),
                        "n_samples": merge_payload.get("n_samples"),
                    }
                )
            if SYNC_BACKEND != "matlab" and run_manifest:
                pc_payload["merge_run_id"] = run_manifest.get("run_id")
                run_manifest["pc_time"] = pc_payload
                run_manifest["pc_time_status"] = "OK" if status == "ok" and aligned_to_merge else "WARN"
                run_manifest["overall_status"] = (
                    "COMPLETE" if run_manifest["pc_time_status"] == "OK" else "MERGE_ONLY"
                )
                if run_manifest["pc_time_status"] == "OK":
                    run_manifest.setdefault("expected_output_bytes", {})["pc_time.dat"] = _file_size(output_path)
                    managed = run_manifest.setdefault("managed_files", [])
                    if "pc_time.dat" not in managed:
                        managed.append("pc_time.dat")
                else:
                    output_path.unlink(missing_ok=True)
                    run_manifest.setdefault("expected_output_bytes", {}).pop("pc_time.dat", None)
                    managed = run_manifest.setdefault("managed_files", [])
                    if "pc_time.dat" in managed:
                        managed.remove("pc_time.dat")
                _write_json_object_atomic(run_manifest_path, run_manifest)
                qc_path = run_manifest_path
            else:
                _write_json_object_atomic(qc_path, pc_payload)
            master.pc_time_exists = output_path.exists()
            master.pc_time_valid = pc_time_valid_for_recording(output_path, master)
            self._append_log(f"pc_time finished for {master.device_id}: {status} ({detail})")
            self._append_log(f"Saved pc_time QC: {qc_path}")
            self._refresh_recording_table()
            self._refresh_preview()
            self._run_preflight()
            self._pc_time_process = None
            if self._continue_after_pc_time:
                if return_code == 0 and master.pc_time_valid:
                    self._run_after_pc_time()
                else:
                    self._continue_after_pc_time = False
                    self._append_log("Skipping downstream steps because pc_time generation failed.")
            elif return_code == 0 and master.pc_time_valid:
                selected_count = len(self._selected_recordings_for_run())
                if selected_count > 2:
                    self._append_log("Run complete: multi-logger amplifier.dat/analogin.dat and master pc_time.dat are ready for review.")
                else:
                    self._append_log("Run complete.")
                self._set_progress("Complete", 100)
            elif return_code == 0:
                self._append_log("pc_time requires review; the merged session was not marked complete.")

        def _force_stop(self) -> None:
            if self._pc_time_process is not None:
                self._pc_time_process.kill()
                self._append_log("Stopped running pc_time process.")
            if self._backend_process is not None:
                self._backend_process.kill()
                self._append_log("Stopped running preprocessing backend.")
            self._set_progress("Stopped", self.progress_bar.value())

        def _set_progress(self, label: str, value: float) -> None:
            value_int = max(0, min(100, int(round(value))))
            self.progress_label.setText(stage_progress_text(label, value_int))
            self.progress_bar.setValue(value_int)

        def _update_progress_from_text(self, text: str) -> None:
            for match in self._progress_re.finditer(text):
                stage = match.group(1)
                try:
                    value = float(match.group(2))
                except ValueError:
                    continue
                self._set_progress(stage, value)

        def _remove_progress_lines(self, text: str) -> str:
            lines = [line for line in text.splitlines() if not line.startswith("WILD_PROGRESS:")]
            if not lines:
                return ""
            suffix = "\n" if text.endswith("\n") else ""
            return "\n".join(lines) + suffix

        def _append_log(self, text: str) -> None:
            self._log_buffer += text
            if not text.endswith("\n"):
                self._log_buffer += "\n"
            self._log_timer.start()

        def _flush_log(self) -> None:
            if not self._log_buffer:
                return
            self.log.moveCursor(QTextCursor.MoveOperation.End)
            self.log.insertPlainText(self._log_buffer)
            self.log.moveCursor(QTextCursor.MoveOperation.End)
            self._log_buffer = ""

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 9))
    window = WildPreprocessWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
