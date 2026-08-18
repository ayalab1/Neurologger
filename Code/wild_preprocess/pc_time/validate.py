"""Validation of PC-clock evidence for the published merged interval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .infer import PcTimeModel


@dataclass(frozen=True)
class PcTimeOptions:
    min_retained_anchors: int = 10
    min_coverage_fraction: float = 0.80
    max_leading_extrapolation_sec: float = 60.0
    max_trailing_extrapolation_sec: float = 60.0
    max_anchor_gap_sec: float = 120.0
    max_residual_rms_ms: float = 100.0
    max_abs_drift_ppm: float = 1000.0
    persistent_step_ms: float = 250.0
    persistent_step_observations: int = 3
    persistent_rate_change_ppm: float = 1000.0
    persistent_rate_observations: int = 20


@dataclass(frozen=True)
class PcTimeValidation:
    status: str
    message: str
    retained_anchor_count: int
    retained_span_sec: float
    coverage_fraction: float
    leading_extrapolation_sec: float
    trailing_extrapolation_sec: float
    max_internal_gap_sec: float
    residual_rms_ms: float
    drift_ppm: float
    persistent_step_detected: bool
    persistent_rate_change_detected: bool
    maximum_local_rate_difference_ppm: float = 0.0
    rate_change_trigger_count: int = 0
    first_rate_change_trigger_time_sec: float | None = None
    publishable: bool = True
    publication_blockers: tuple[str, ...] = ()


def _persistent_step(residual_ms: np.ndarray, threshold_ms: float, run: int) -> bool:
    if residual_ms.size < 2 * run:
        return False
    for boundary in range(run, residual_ms.size - run + 1):
        before = np.median(residual_ms[boundary - run : boundary])
        after = np.median(residual_ms[boundary : boundary + run])
        if abs(after - before) >= threshold_ms:
            return True
    return False


def _robust_residual_rate_ppm(device_ms: np.ndarray, residual_ms: np.ndarray) -> float:
    """Return a Theil-Sen residual rate, tolerating isolated updates.

    A residual slope is a difference in clock rate because device time is
    also expressed in milliseconds.  The median of all pairwise slopes uses
    the full local time span and prevents ordinary packed-delay jitter, or one
    corrupt update, from looking like a new rate regime.
    """

    if device_ms.size < 2:
        return 0.0
    delta_device = device_ms[np.newaxis, :] - device_ms[:, np.newaxis]
    delta_residual = residual_ms[np.newaxis, :] - residual_ms[:, np.newaxis]
    usable = delta_device > 0.0
    if not np.any(usable):
        return 0.0
    return float(np.median(delta_residual[usable] / delta_device[usable]) * 1_000_000.0)


def _rate_change_diagnostics(
    device_ms: np.ndarray,
    residual_ms: np.ndarray,
    threshold_ppm: float,
    run: int,
) -> tuple[bool, float, int, float | None]:
    """Summarize adjacent, sufficiently supported residual-rate regimes."""

    if device_ms.size < 2 * run:
        return False, 0.0, 0, None
    maximum_difference = 0.0
    trigger_count = 0
    first_trigger_time_sec: float | None = None
    for boundary in range(run, device_ms.size - run + 1):
        before_rate = _robust_residual_rate_ppm(
            device_ms[boundary - run : boundary], residual_ms[boundary - run : boundary]
        )
        after_rate = _robust_residual_rate_ppm(
            device_ms[boundary : boundary + run], residual_ms[boundary : boundary + run]
        )
        difference = abs(after_rate - before_rate)
        maximum_difference = max(maximum_difference, difference)
        if difference >= threshold_ppm:
            trigger_count += 1
            if first_trigger_time_sec is None:
                first_trigger_time_sec = float(device_ms[boundary] / 1000.0)
    return trigger_count > 0, maximum_difference, trigger_count, first_trigger_time_sec


def _persistent_rate_change(
    device_ms: np.ndarray,
    residual_ms: np.ndarray,
    threshold_ppm: float,
    run: int,
) -> bool:
    """Return whether the ordered observations contain a rate-change trigger."""

    return _rate_change_diagnostics(device_ms, residual_ms, threshold_ppm, run)[0]


def _ordered_interval_observations(
    model: PcTimeModel,
    start_ms: float,
    end_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return every lifted observation in the saved interval, in time order."""

    inside = (model.device_ms >= start_ms) & (model.device_ms <= end_ms)
    device_ms = model.device_ms[inside]
    residual_ms = model.residual_ms[inside]
    if device_ms.size < 2:
        return device_ms, residual_ms
    order = np.argsort(device_ms, kind="stable")
    return device_ms[order], residual_ms[order]


def validate_pc_time_interval(
    model: PcTimeModel,
    *,
    sample_rate_hz: float,
    common_start_master_sample: int,
    n_samples: int,
    options: PcTimeOptions = PcTimeOptions(),
) -> PcTimeValidation:
    """Check only anchors that support the saved merged sample interval."""

    if sample_rate_hz <= 0 or common_start_master_sample < 0 or n_samples <= 0:
        raise ValueError("invalid merged interval or sample rate")
    start_ms = common_start_master_sample * 1000.0 / sample_rate_hz
    end_ms = (common_start_master_sample + n_samples - 1) * 1000.0 / sample_rate_hz
    duration_sec = max((end_ms - start_ms) / 1000.0, 0.0)
    # Robustly kept anchors support coverage and residual-RMS estimates.
    kept_times = model.device_ms[model.keep_mask]
    kept_residuals = model.residual_ms[model.keep_mask]
    inside = (kept_times >= start_ms) & (kept_times <= end_ms)
    support_times = kept_times[inside]
    support_residuals = kept_residuals[inside]
    count = int(support_times.size)
    if count:
        leading = max(0.0, (support_times[0] - start_ms) / 1000.0)
        trailing = max(0.0, (end_ms - support_times[-1]) / 1000.0)
        span = max(0.0, (support_times[-1] - support_times[0]) / 1000.0)
        gaps = np.diff(support_times) / 1000.0
        max_gap = float(gaps.max()) if gaps.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(support_residuals))))
    else:
        leading = trailing = span = max_gap = float("inf")
        rms = float("inf")
    coverage = 1.0 if duration_sec == 0 and count else (span / duration_sec if duration_sec else 0.0)
    ordered_times, ordered_residuals = _ordered_interval_observations(model, start_ms, end_ms)
    step = _persistent_step(ordered_residuals, options.persistent_step_ms, options.persistent_step_observations)
    rate_change, maximum_rate_difference, rate_trigger_count, first_rate_trigger = (
        _rate_change_diagnostics(
            ordered_times,
            ordered_residuals,
            options.persistent_rate_change_ppm,
            options.persistent_rate_observations,
        )
    )
    checks = [
        (count >= options.min_retained_anchors, f"retained anchors {count} < {options.min_retained_anchors}"),
        (coverage >= options.min_coverage_fraction, f"anchor coverage {coverage:.3f} < {options.min_coverage_fraction:.3f}"),
        (leading <= options.max_leading_extrapolation_sec, f"leading extrapolation {leading:.3f}s exceeds limit"),
        (trailing <= options.max_trailing_extrapolation_sec, f"trailing extrapolation {trailing:.3f}s exceeds limit"),
        (max_gap <= options.max_anchor_gap_sec, f"maximum anchor gap {max_gap:.3f}s exceeds limit"),
        (rms <= options.max_residual_rms_ms, f"residual RMS {rms:.3f}ms exceeds limit"),
        (abs(model.drift_ppm) <= options.max_abs_drift_ppm, f"clock drift {model.drift_ppm:.3f}ppm exceeds limit"),
        (not step, "persistent PC-clock residual step detected"),
        (not rate_change, "persistent PC-clock rate-regime change detected"),
    ]
    failures = [message for passed, message in checks if not passed]
    publication_blockers: list[str] = []
    if count < 2 or not np.isfinite(span) or span <= 0.0:
        publication_blockers.append("fewer than two time-separated retained anchors")
    if (
        not np.isfinite(model.slope)
        or not np.isfinite(model.intercept_ms)
        or model.slope <= 0.0
    ):
        publication_blockers.append("PC-clock affine model is non-finite or non-increasing")
    if step:
        publication_blockers.append("persistent PC-clock residual step detected")
    if rate_trigger_count > 1:
        publication_blockers.append(
            f"PC-clock rate-regime change reproduced at {rate_trigger_count} tested boundaries"
        )
    return PcTimeValidation(
        status="OK" if not failures else "WARN",
        message="; ".join(failures),
        retained_anchor_count=count,
        retained_span_sec=span,
        coverage_fraction=coverage,
        leading_extrapolation_sec=leading,
        trailing_extrapolation_sec=trailing,
        max_internal_gap_sec=max_gap,
        residual_rms_ms=rms,
        drift_ppm=model.drift_ppm,
        persistent_step_detected=step,
        persistent_rate_change_detected=rate_change,
        maximum_local_rate_difference_ppm=maximum_rate_difference,
        rate_change_trigger_count=rate_trigger_count,
        first_rate_change_trigger_time_sec=first_rate_trigger,
        publishable=not publication_blockers,
        publication_blockers=tuple(publication_blockers),
    )
