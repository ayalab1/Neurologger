from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from .models import PipelineResult, SyncPairResult
from .sync.gaps import gap_summary
from .sync.observe import PairObservations
from .version import SYNC_ALGORITHM_VERSION


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__} as JSON")


def save_pair_figure(
    pair: SyncPairResult,
    observed: PairObservations,
    filename: Path,
) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)
    decimation = max(1, observed.initial_master.size // 20000)
    x = np.arange(0, observed.initial_master.size, decimation)
    axes[0].plot(x, observed.initial_master[::decimation], linewidth=0.6, label="master")
    axes[0].plot(x, observed.initial_slave[::decimation], linewidth=0.6, label="slave", alpha=0.8)
    axes[0].set_title("Initial common-mode feature")
    axes[0].legend()
    axes[0].set_xlabel("samples")

    axes[1].plot(observed.initial.lags, observed.initial.correlations, linewidth=0.7)
    axes[1].axvline(-observed.initial.lag_samples, color="tab:red", linewidth=1)
    axes[1].set_title(
        "Initial cross-correlation "
        f"(offset {observed.initial.lag_samples}, margin {observed.initial.peak_margin_fraction:.3g})"
    )
    axes[1].set_xlabel("correlation lag samples")

    times = np.asarray([item.center_time_sec for item in pair.observations], dtype=float)
    offsets = np.asarray([item.observed_offset_samples for item in pair.observations], dtype=float)
    accepted = np.asarray([item.accepted for item in pair.observations], dtype=bool)
    margins = np.asarray([item.peak_margin_fraction for item in pair.observations], dtype=float)
    if accepted.any():
        axes[2].scatter(times[accepted], offsets[accepted], s=9, label="accepted observations")
    if (~accepted).any():
        axes[2].scatter(times[~accepted], offsets[~accepted], s=18, marker="x", color="tab:red", label="rejected")
    if times.size:
        fitted = np.asarray([pair.model.offset_at_seconds(time_sec) for time_sec in times], dtype=float)
        axes[2].plot(times, fitted, color="black", linewidth=1.5, label=f"fit {pair.model.drift_ppm:.3f} ppm")
        for event_index, event in enumerate(pair.model.offset_steps):
            axes[2].axvline(
                event.time_sec,
                color="tab:red",
                linewidth=0.8,
                alpha=0.7,
                label="offset step" if event_index == 0 else None,
            )
        finite_offsets = offsets[np.isfinite(offsets)]
        if finite_offsets.size and float(np.ptp(finite_offsets)) < 1.0:
            center = float(np.median(finite_offsets))
            half_range = max(0.5, float(np.max(np.abs(finite_offsets - center))) + 0.1)
            axes[2].set_ylim(center - half_range, center + half_range)
            axes[2].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[2].set_title(f"Gap-aware offset model ({pair.status})")
    axes[2].set_xlabel("master time sec")
    axes[2].set_ylabel("slave offset samples")
    margin_axis = axes[2].twinx()
    margin_axis.plot(times, margins, color="tab:orange", linewidth=0.5, alpha=0.5)
    margin_axis.set_ylabel("primary/secondary margin")
    axes[2].legend(loc="best")
    figure.suptitle(
        f"master {Path(pair.master_folder).parent.name} vs slave {Path(pair.slave_folder).parent.name}",
        fontsize=12,
    )
    figure.savefig(filename, dpi=140)
    plt.close(figure)


def write_qc_outputs(
    result: PipelineResult,
    prefix: str = "wild_multilogger_sync",
    *,
    mode: str | None = None,
) -> dict[str, str]:
    output_folder = result.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)
    tsv_path = output_folder / f"{prefix}_qc.tsv"
    json_path = output_folder / f"{prefix}_qc.json"
    mat_path = output_folder / f"{prefix}_qc.mat"
    fieldnames = [
        "status",
        "master_index",
        "slave_index",
        "master_folder",
        "slave_folder",
        "initial_offset_samples",
        "final_offset_samples",
        "initial_peak_ratio",
        "initial_peak_margin_fraction",
        "chunk_count",
        "accepted_chunk_count",
        "wide_reacquired_chunk_count",
        "relative_offset_step_count",
        "validated_start_master_sample",
        "terminal_crop_master_sample",
        "terminal_crop_reason",
        "slave_gap_count",
        "slave_missing_samples",
        "slave_missing_fraction",
        "min_chunk_peak_ratio",
        "min_chunk_peak_margin_fraction",
        "max_abs_lag_step_samples",
        "offset_drift_samples",
        "drift_ppm",
        "model_rms_samples",
        "model_max_abs_residual_samples",
        "figure_file",
        "message",
    ]
    rows: list[dict[str, Any]] = []
    canonical_samples = (
        result.recordings[result.master_index - 1].n_samples
        + sum(gap.missing_samples for gap in result.device_gaps if gap.device_index == result.master_index)
    )
    summaries = {
        int(item["device_index"]): item
        for item in gap_summary(
            result.device_gaps,
            device_count=len(result.recordings),
            canonical_samples=canonical_samples,
        )
    }
    for pair in result.pairs:
        finite_ratios = [item.peak_to_background for item in pair.observations if np.isfinite(item.peak_to_background)]
        finite_margins = [item.peak_margin_fraction for item in pair.observations if np.isfinite(item.peak_margin_fraction)]
        accepted_offsets = [item.observed_offset_samples for item in pair.observations if item.accepted]
        max_step = max((abs(b - a) for a, b in zip(accepted_offsets, accepted_offsets[1:])), default=0.0)
        slave_summary = summaries[pair.slave_index]
        rows.append(
            {
                "status": pair.status,
                "master_index": pair.master_index,
                "slave_index": pair.slave_index,
                "master_folder": pair.master_folder,
                "slave_folder": pair.slave_folder,
                "initial_offset_samples": f"{pair.initial_offset_samples:.6g}",
                "final_offset_samples": f"{pair.final_offset_samples:.6g}",
                "initial_peak_ratio": f"{pair.initial_peak_to_background:.6g}",
                "initial_peak_margin_fraction": f"{pair.initial_peak_margin_fraction:.6g}",
                "chunk_count": len(pair.observations),
                "accepted_chunk_count": pair.model.accepted_count,
                "wide_reacquired_chunk_count": sum(
                    item.accepted and item.search_mode == "wide_reacquisition"
                    for item in pair.observations
                ),
                "relative_offset_step_count": len(pair.model.offset_steps),
                "validated_start_master_sample": pair.validated_start_master_sample,
                "terminal_crop_master_sample": (
                    "" if pair.terminal_crop_master_sample is None else pair.terminal_crop_master_sample
                ),
                "terminal_crop_reason": pair.terminal_crop_reason,
                "slave_gap_count": slave_summary["gap_count"],
                "slave_missing_samples": slave_summary["missing_samples"],
                "slave_missing_fraction": f"{float(slave_summary['missing_fraction']):.9g}",
                "min_chunk_peak_ratio": f"{min(finite_ratios, default=float('nan')):.6g}",
                "min_chunk_peak_margin_fraction": f"{min(finite_margins, default=float('nan')):.6g}",
                "max_abs_lag_step_samples": f"{max_step:.6g}",
                "offset_drift_samples": f"{pair.offset_drift_samples:.6g}",
                "drift_ppm": f"{pair.model.drift_ppm:.6g}",
                "model_rms_samples": f"{pair.model.residual_rms_samples:.6g}",
                "model_max_abs_residual_samples": f"{pair.model.residual_max_abs_samples:.6g}",
                "figure_file": pair.figure_file,
                "message": pair.message,
            }
        )
    with tsv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    output_mode = mode or ("multiMerge" if "amplifier" in result.outputs else "syncQC")
    payload = result.to_dict()
    payload["mode"] = output_mode
    payload["device_gap_summary"] = list(summaries.values())
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    observation_cells = np.empty((len(result.pairs),), dtype=object)
    step_cells = np.empty((len(result.pairs),), dtype=object)
    for index, pair in enumerate(result.pairs):
        observation_cells[index] = np.asarray(
            [
                [
                    item.center_time_sec,
                    item.predicted_offset_samples,
                    item.observed_offset_samples,
                    item.residual_lag_samples,
                    item.peak_correlation,
                    item.peak_to_background,
                    item.peak_margin_fraction,
                    float(item.accepted),
                ]
                for item in pair.observations
            ],
            dtype=float,
        )
        step_cells[index] = np.asarray(
            [
                [
                    step.master_sample,
                    step.time_sec,
                    step.offset_step_samples,
                    step.missing_samples,
                    step.offset_before_samples,
                    step.offset_after_samples,
                ]
                for step in pair.model.offset_steps
            ],
            dtype=float,
        ).reshape((-1, 6))
    legacy_pairs = np.empty(
        (1, len(result.pairs)),
        dtype=[
            ("masterIndex", "O"),
            ("slaveIndex", "O"),
            ("masterFolder", "O"),
            ("slaveFolder", "O"),
            ("masterFile", "O"),
            ("slaveFile", "O"),
            ("initialOffsetSamples", "O"),
            ("finalOffsetSamples", "O"),
            ("initialPeakRatio", "O"),
            ("status", "O"),
            ("message", "O"),
            ("chunks", "O"),
            ("figureFile", "O"),
        ],
    )
    for index, pair in enumerate(result.pairs):
        chunks = np.asarray(
            [
                [
                    item.center_time_sec,
                    item.observed_offset_samples,
                    item.residual_lag_samples,
                    item.peak_to_background,
                    item.peak_correlation,
                    item.peak_margin_fraction,
                    float(item.accepted),
                    float(item.model_inlier),
                ]
                for item in pair.observations
            ],
            dtype=float,
        )
        master_recording = result.recordings[pair.master_index - 1]
        slave_recording = result.recordings[pair.slave_index - 1]
        legacy_pairs[0, index] = (
            pair.master_index,
            pair.slave_index,
            pair.master_folder,
            pair.slave_folder,
            str(master_recording.amplifier_file),
            str(slave_recording.amplifier_file),
            pair.initial_offset_samples,
            pair.final_offset_samples,
            pair.initial_peak_to_background,
            pair.status,
            pair.message,
            chunks,
            pair.figure_file,
        )
    legacy_folders = np.asarray([str(recording.folder) for recording in result.recordings], dtype=object)
    legacy_files = np.asarray([str(recording.amplifier_file) for recording in result.recordings], dtype=object)
    legacy_sys_params = np.empty(
        (1, len(result.recordings)), dtype=[("fs", "O"), ("Nch", "O")]
    )
    for index, recording in enumerate(result.recordings):
        legacy_sys_params[0, index] = (recording.fs, recording.n_channels)
    legacy_result = {
        "runId": result.run_id,
        "mode": output_mode,
        "files": legacy_files,
        "folders": legacy_folders,
        "masterIndex": result.master_index,
        "fs": result.recordings[result.master_index - 1].fs,
        "nChannels": np.asarray([recording.n_channels for recording in result.recordings]),
        "nSamples": np.asarray([recording.n_samples for recording in result.recordings]),
        "pairs": legacy_pairs,
        "tsvFile": str(tsv_path),
        "matFile": str(mat_path),
        "deviceGapDeviceIndex": np.asarray(
            [gap.device_index for gap in result.device_gaps], dtype=np.int32
        ),
        "deviceGapCanonicalStartSample": np.asarray(
            [gap.canonical_start_sample for gap in result.device_gaps], dtype=np.int64
        ),
        "deviceGapMissingSamples": np.asarray(
            [gap.missing_samples for gap in result.device_gaps], dtype=np.int64
        ),
    }
    savemat(
        mat_path,
        {
            "result": legacy_result,
            "pairResults": legacy_pairs,
            "opts": {"Backend": SYNC_ALGORITHM_VERSION},
            "folders": legacy_folders,
            "files": legacy_files,
            "sysParams": legacy_sys_params,
            "status": result.status,
            "run_id": result.run_id,
            "master_index": result.master_index,
            "pair_summary": np.asarray(
                [
                    [
                        pair.master_index,
                        pair.slave_index,
                        pair.initial_offset_samples,
                        pair.final_offset_samples,
                        pair.model.drift_ppm,
                        pair.model.residual_rms_samples,
                        pair.model.residual_max_abs_samples,
                    ]
                    for pair in result.pairs
                ],
                dtype=float,
            ),
            "pair_observations": observation_cells,
            "pair_relative_offset_steps": step_cells,
            "pair_terminal_crop_master_sample": np.asarray(
                [
                    -1
                    if pair.terminal_crop_master_sample is None
                    else pair.terminal_crop_master_sample
                    for pair in result.pairs
                ],
                dtype=np.int64,
            ),
            "pair_terminal_crop_reason": np.asarray(
                [pair.terminal_crop_reason for pair in result.pairs], dtype=object
            ),
            "relative_offset_step_columns": np.asarray(
                [
                    "master_sample",
                    "time_sec",
                    "offset_step_samples",
                    "missing_samples",
                    "offset_before_samples",
                    "offset_after_samples",
                ],
                dtype=object,
            ),
            "device_gap_device_index": np.asarray(
                [gap.device_index for gap in result.device_gaps], dtype=np.int32
            ),
            "device_gap_canonical_start_sample": np.asarray(
                [gap.canonical_start_sample for gap in result.device_gaps], dtype=np.int64
            ),
            "device_gap_missing_samples": np.asarray(
                [gap.missing_samples for gap in result.device_gaps], dtype=np.int64
            ),
            "device_gap_duration_ms": np.asarray(
                [gap.duration_ms for gap in result.device_gaps], dtype=float
            ),
            "device_gap_confidence": np.asarray(
                [gap.confidence for gap in result.device_gaps], dtype=object
            ),
            "device_gap_action": np.asarray(
                [gap.action for gap in result.device_gaps], dtype=object
            ),
            "observation_columns": np.asarray(
                [
                    "time_sec",
                    "predicted_offset_samples",
                    "observed_offset_samples",
                    "residual_lag_samples",
                    "peak_correlation",
                    "peak_to_background",
                    "peak_margin_fraction",
                    "accepted",
                ],
                dtype=object,
            ),
        },
        do_compression=True,
    )
    return {"qc_tsv": str(tsv_path), "qc_json": str(json_path), "qc_mat": str(mat_path)}
