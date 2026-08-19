from __future__ import annotations

import numpy as np

from ..models import SyncModel, SyncObservation, SyncOptions
from .observe import LagEstimate


def _max_rejected_run(observations: list[SyncObservation]) -> int:
    longest = 0
    current = 0
    for observation in observations:
        if observation.accepted:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _max_model_outlier_run(observations: list[SyncObservation]) -> int:
    longest = 0
    current = 0
    for observation in observations:
        if not observation.accepted or observation.model_inlier:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _usable_duration_seconds(observations: list[SyncObservation], options: SyncOptions) -> float:
    if not observations:
        return 0.0
    centers = [observation.center_time_sec for observation in observations]
    return max(centers) - min(centers) + options.window_seconds


def _persistent_offset_level_shift(
    observations: list[SyncObservation],
    model: SyncModel,
    persistence: int,
) -> float:
    """Measure a sustained step after removing the accepted affine clock model.

    Each candidate uses ``persistence`` observations on both sides.  A local
    common slope plus a side indicator recovers the step magnitude even when
    the global affine fit distributes part of a real step into its slope.  A
    clean affine drift has zero residual step regardless of its clock rate.
    """

    accepted = [observation for observation in observations if observation.accepted]
    times = np.asarray([observation.center_time_sec for observation in accepted], dtype=np.float64)
    residuals = np.asarray(
        [
            observation.observed_offset_samples
            - model.offset_at_seconds(observation.center_time_sec)
            for observation in accepted
        ],
        dtype=np.float64,
    )
    if persistence < 1 or residuals.size < 2 * persistence:
        return 0.0
    if persistence == 1:
        return float(np.max(np.abs(np.diff(residuals))))

    max_shift = 0.0
    for split in range(persistence, residuals.size - persistence + 1):
        window = slice(split - persistence, split + persistence)
        local_times = times[window]
        if any(
            local_times[0] < step.time_sec <= local_times[-1]
            for step in model.offset_steps
        ):
            continue
        side = np.concatenate((np.zeros(persistence), np.ones(persistence)))
        design = np.column_stack(
            (
                np.ones(2 * persistence),
                local_times - float(np.mean(local_times)),
                side,
            )
        )
        coefficients, *_ = np.linalg.lstsq(design, residuals[window], rcond=None)
        max_shift = max(max_shift, abs(float(coefficients[2])))
    return max_shift


def _meets_numerical_threshold(value: float, threshold: float) -> bool:
    """Compare an estimated quantity at an inclusive scientific threshold.

    Local least-squares conditioning can represent an exact configured step
    slightly below its threshold.  One nanosecond of a sample is many orders
    below both the integer observation resolution and any scientifically
    meaningful fractional-sample precision, so it absorbs numerical error
    without materially widening the configured threshold.
    """

    return value >= threshold or bool(np.isclose(value, threshold, rtol=0.0, atol=1e-9))


def validate_pair(
    initial: LagEstimate,
    observations: list[SyncObservation],
    model: SyncModel,
    options: SyncOptions,
) -> tuple[str, str]:
    failures: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    if initial.peak_correlation < options.min_peak_correlation:
        failures.append(f"initial normalized correlation {initial.peak_correlation:.3g}")
    if initial.peak_to_background < options.min_peak_to_background:
        failures.append(f"initial peak/background {initial.peak_to_background:.3g}")
    if initial.peak_margin_fraction < options.min_peak_margin_fraction:
        # The initial alignment selects the coordinate system for every
        # tracking window.  A competing initial peak means that coordinate
        # system is not identifiable, even if later local windows appear
        # internally consistent, so it must prevent publication.
        failures.append(f"initial peak margin {initial.peak_margin_fraction:.3g}")
    initial_search_half_width = int(np.max(np.abs(initial.lags))) if initial.lags.size else 0
    if initial_search_half_width and abs(initial.lag_samples) >= max(1, initial_search_half_width - 1):
        failures.append("initial correlation peak is at the search boundary")
    accepted_fraction = model.accepted_count / max(model.observation_count, 1)
    if accepted_fraction < options.min_accepted_fraction:
        failures.append(f"accepted windows {accepted_fraction:.1%}")
    accepted_observations = [observation for observation in observations if observation.accepted]
    low_normalized = [
        observation
        for observation in accepted_observations
        if not np.isfinite(observation.peak_correlation)
        or observation.peak_correlation < options.min_peak_correlation
    ]
    if low_normalized:
        failures.append(f"{len(low_normalized)} accepted windows below normalized-correlation gate")
    usable_duration = _usable_duration_seconds(observations, options)
    if usable_duration < options.short_recording_seconds:
        if model.accepted_count < options.short_min_accepted_observations:
            failures.append(
                f"short recording has {model.accepted_count} accepted observations "
                f"(requires {options.short_min_accepted_observations})"
            )
        if not model.is_constant_offset:
            failures.append("short recording did not use a constant-offset model")
    else:
        if model.accepted_count < options.min_accepted_observations:
            failures.append(
                f"accepted observations {model.accepted_count} "
                f"(requires {options.min_accepted_observations})"
            )
        accepted_span = (
            max(observation.center_time_sec for observation in accepted_observations)
            - min(observation.center_time_sec for observation in accepted_observations)
            if len(accepted_observations) >= 2
            else 0.0
        )
        if accepted_span < options.min_accepted_span_seconds:
            failures.append(
                f"accepted observation span {accepted_span:.1f} s "
                f"(requires {options.min_accepted_span_seconds:.1f} s)"
            )
    if model.residual_rms_samples > options.max_model_rms_samples:
        failures.append(f"model RMS {model.residual_rms_samples:.2f} samples")
    if model.residual_max_abs_samples > options.max_model_residual_samples:
        failures.append(f"max model residual {model.residual_max_abs_samples:.1f} samples")
    rejected_run = _max_rejected_run(observations)
    if rejected_run > options.max_consecutive_rejections:
        failures.append(f"{rejected_run} consecutive rejected windows")
    model_outlier_run = _max_model_outlier_run(observations)
    if model_outlier_run > options.max_consecutive_model_outliers:
        failures.append(f"{model_outlier_run} consecutive model-outlier windows (possible clock discontinuity)")
    # Confirmed large steps are validated globally across all pairs. The local
    # estimator skips only windows crossing those known transitions so smaller
    # unsupported discontinuities elsewhere in the recording remain visible.
    max_level_shift = _persistent_offset_level_shift(
        observations,
        model,
        options.persistent_level_shift_observations,
    )
    if _meets_numerical_threshold(max_level_shift, options.max_offset_level_shift_samples):
        failures.append(
            f"persistent offset level shift {max_level_shift:.1f} samples "
            "(possible clock discontinuity)"
        )
    elif _meets_numerical_threshold(max_level_shift, options.report_offset_level_shift_samples):
        notes.append(f"nonblocking persistent offset level shift {max_level_shift:.1f} samples")
    accepted_residual_offsets = [
        observation.observed_offset_samples - model.offset_at_seconds(observation.center_time_sec)
        for observation in observations
        if observation.accepted
    ]
    max_offset_step = max(
        (
            abs(current - previous)
            for previous, current in zip(accepted_residual_offsets, accepted_residual_offsets[1:])
        ),
        default=0.0,
    )
    if max_offset_step > options.max_observed_offset_step_samples:
        failures.append(f"detrended offset step {max_offset_step:.1f} samples")
    if abs(model.drift_ppm) > options.warn_drift_ppm:
        warnings.append(f"drift {model.drift_ppm:.1f} ppm")
    if model.offset_steps:
        warnings.append(f"{len(model.offset_steps)} persistent offset step candidate(s)")
    if failures:
        return "FAIL", "; ".join(failures + warnings + notes)
    if warnings:
        return "WARN", "; ".join(warnings + notes)
    return "OK", "; ".join(["affine clock model accepted"] + notes)
