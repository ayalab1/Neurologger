"""Small, dependency-light PC-time QC report helpers."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .infer import PcTimeModel
from .validate import PcTimeValidation


def pc_time_qc_payload(
    model: PcTimeModel,
    validation: PcTimeValidation,
    *,
    run_id: str,
    common_start_master_sample: int,
    n_samples: int,
    sample_rate_hz: float,
    anchor_source: str,
    layout_name: str,
) -> dict[str, Any]:
    """Build JSON-safe metadata for the native PC-time component."""

    return {
        "algorithm": "wild_preprocess.pc_time.v4",
        "run_id": run_id,
        "layout": layout_name,
        "anchor_source": anchor_source,
        "milliseconds_semantics": "uint32 milliseconds since midnight; values wrap at midnight",
        "common_start_master_sample": int(common_start_master_sample),
        "n_samples": int(n_samples),
        "sample_rate_hz": float(sample_rate_hz),
        "model": {
            "slope": model.slope,
            "intercept_ms": model.intercept_ms,
            "drift_ppm": model.drift_ppm,
            "residual_rms_ms": model.residual_rms_ms,
            "update_count": int(model.device_ms.size),
            "kept_update_count": model.kept_count,
            "recording_start_ms": model.recording_start_ms,
        },
        "validation": asdict(validation),
    }


def write_pc_time_qc_json(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def write_pc_time_summary_png(
    path: Path,
    model: PcTimeModel,
    validation: PcTimeValidation,
    *,
    common_start_master_sample: int,
    n_samples: int,
    sample_rate_hz: float,
) -> Path:
    """Write a concise raw-observation diagnostic, marking the saved interval."""

    import matplotlib.pyplot as plt

    path = Path(path)
    start_sec = common_start_master_sample / sample_rate_hz
    end_sec = (common_start_master_sample + n_samples - 1) / sample_rate_hz
    time_sec = model.device_ms / 1000.0
    fig, (ax_fit, ax_residual) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    ax_fit.axvspan(start_sec, end_sec, color="#e8f4e8", label="published interval")
    ax_fit.scatter(time_sec[~model.keep_mask], model.pc_unwrapped_ms[~model.keep_mask] / 60000.0, s=10, c="#c33", label="discarded")
    ax_fit.scatter(time_sec[model.keep_mask], model.pc_unwrapped_ms[model.keep_mask] / 60000.0, s=10, c="#1677b3", label="kept")
    fitted = model.predict_unwrapped_ms(model.device_ms) / 60000.0
    ax_fit.plot(time_sec, fitted, c="black", linewidth=1.2, label="fit")
    ax_fit.set_ylabel("PC time (unwrapped min)")
    ax_fit.legend(loc="best")
    ax_fit.set_title(f"PC-time fit: {validation.status} — {validation.message or 'all merged-interval gates passed'}")
    ax_residual.axvspan(start_sec, end_sec, color="#e8f4e8")
    ax_residual.axhline(0.0, color="black", linewidth=0.8)
    ax_residual.scatter(time_sec[~model.keep_mask], model.residual_ms[~model.keep_mask], s=10, c="#c33")
    ax_residual.scatter(time_sec[model.keep_mask], model.residual_ms[model.keep_mask], s=10, c="#1677b3")
    ax_residual.set_xlabel("Master device time (s)")
    ax_residual.set_ylabel("Fit residual (ms)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_pc_time_warning_png(
    path: Path,
    *,
    message: str,
    common_start_master_sample: int,
    n_samples: int,
    sample_rate_hz: float,
) -> Path:
    """Write the retained PC-time figure when no defensible fit exists."""

    import matplotlib.pyplot as plt

    path = Path(path)
    start_sec = common_start_master_sample / sample_rate_hz
    end_sec = (common_start_master_sample + max(0, n_samples - 1)) / sample_rate_hz
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    ax.axis("off")
    ax.set_title("PC-time unavailable", color="#a33", fontsize=15, pad=18)
    ax.text(
        0.5,
        0.62,
        message or "No verified PC-time model could be constructed.",
        ha="center",
        va="center",
        wrap=True,
        transform=ax.transAxes,
        fontsize=11,
    )
    ax.text(
        0.5,
        0.28,
        (
            f"Published canonical interval: {start_sec:.3f} to {end_sec:.3f} s\n"
            "Neural output remains available as MERGE_ONLY; pc_time.dat was not published."
        ),
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="#555",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
