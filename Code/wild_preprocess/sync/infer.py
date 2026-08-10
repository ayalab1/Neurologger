from __future__ import annotations

import numpy as np

from ..models import RelativeOffsetStep, SyncModel, SyncObservation, SyncOptions
from .gaps import detect_relative_offset_steps


def _robust_inlier_mask(x: np.ndarray, y: np.ndarray, *, constant_offset: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return a robust fit mask and [slope, intercept] coefficients."""

    keep = np.ones(x.size, dtype=bool)
    coefficients = np.asarray([0.0, float(np.median(y))], dtype=np.float64)
    for _ in range(8):
        if constant_offset:
            coefficients = np.asarray([0.0, float(np.median(y[keep]))], dtype=np.float64)
        else:
            coefficients = np.polyfit(x[keep], y[keep], 1)
        residuals = y - np.polyval(coefficients, x)
        center = np.median(residuals[keep])
        mad = np.median(np.abs(residuals[keep] - center))
        gate = max(6.0, 6.0 * 1.4826 * mad)
        updated = np.abs(residuals - center) <= gate
        minimum = 1 if constant_offset else 2
        if updated.sum() < minimum or np.array_equal(updated, keep):
            break
        keep = updated
    if constant_offset:
        coefficients = np.asarray([0.0, float(np.median(y[keep]))], dtype=np.float64)
    else:
        coefficients = np.polyfit(x[keep], y[keep], 1)
    return keep, coefficients


def fit_affine_sync_model(
    observations: list[SyncObservation],
    fs: float,
    *,
    options: SyncOptions | None = None,
    offset_steps: tuple[RelativeOffsetStep, ...] | None = None,
) -> SyncModel:
    """Fit the master-to-slave offset model from accepted observations.

    Supplying ``options`` enables the production short-recording rule.  The
    optional argument keeps the original affine-only call API available to
    legacy callers and focused numerical tests.
    """

    accepted = [observation for observation in observations if observation.accepted]
    if len(accepted) < 2:
        return SyncModel(
            intercept_samples=0.0,
            slope_samples_per_second=0.0,
            drift_ppm=0.0,
            residual_rms_samples=float("inf"),
            residual_max_abs_samples=float("inf"),
            accepted_count=len(accepted),
            observation_count=len(observations),
        )
    x = np.asarray([observation.center_time_sec for observation in accepted], dtype=np.float64)
    observed_y = np.asarray([observation.observed_offset_samples for observation in accepted], dtype=np.float64)
    if offset_steps is None:
        offset_steps = (
            detect_relative_offset_steps(observations, fs, options)
            if options is not None
            else ()
        )
    step_values = np.asarray(
        [
            sum(step.offset_step_samples for step in offset_steps if step.time_sec <= time_sec)
            for time_sec in x
        ],
        dtype=np.float64,
    )
    y = observed_y - step_values
    observed_times = [observation.center_time_sec for observation in observations]
    usable_duration = (
        max(observed_times) - min(observed_times) + options.window_seconds
        if options is not None and observed_times
        else float("inf")
    )
    constant_offset = options is not None and usable_duration < options.short_recording_seconds
    keep, coefficients = _robust_inlier_mask(x, y, constant_offset=constant_offset)
    all_residuals = y - np.polyval(coefficients, x)
    residuals = all_residuals[keep]
    for observation, is_inlier, residual in zip(accepted, keep, all_residuals):
        observation.model_inlier = bool(is_inlier)
        observation.model_residual_samples = float(residual)
    slope = float(coefficients[0])
    intercept = float(coefficients[1])
    rms = float(np.sqrt(np.mean(np.square(residuals))))
    max_abs = float(np.max(np.abs(residuals)))
    return SyncModel(
        intercept_samples=intercept,
        slope_samples_per_second=slope,
        drift_ppm=slope / fs * 1e6,
        residual_rms_samples=rms,
        residual_max_abs_samples=max_abs,
        accepted_count=int(keep.sum()),
        observation_count=len(observations),
        is_constant_offset=constant_offset,
        offset_steps=tuple(offset_steps),
    )
