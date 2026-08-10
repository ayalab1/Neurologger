from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.signal import correlate, correlation_lags, find_peaks

from ..binary_io import close_memmap
from ..models import Recording, SyncObservation, SyncOptions
from .features import feature_memmap


ProgressCallback = Callable[[str, float], None]


@dataclass(frozen=True)
class LagEstimate:
    lag_samples: int
    peak_correlation: float
    peak_to_background: float
    peak_margin_fraction: float
    secondary_lag_samples: int | None
    lags: np.ndarray
    correlations: np.ndarray


@dataclass(frozen=True)
class CorrelationProfile:
    lags: np.ndarray
    correlations: np.ndarray


@dataclass
class PairObservations:
    initial: LagEstimate
    observations: list[SyncObservation]
    initial_master: np.ndarray
    initial_slave: np.ndarray
    validated_start_master_sample: int = 0


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values - np.mean(values)
    scale = np.linalg.norm(values)
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        raise ValueError("Cannot correlate a constant or invalid common-mode window.")
    return values / scale


def _correlation_profile(master: np.ndarray, slave: np.ndarray) -> CorrelationProfile:
    """Compute one full FFT correlation profile for narrow and wide selection."""

    n = min(master.size, slave.size)
    if n < 4:
        raise ValueError("At least four common-mode samples are required for lag estimation.")
    a = _standardize(master[:n])
    b = _standardize(slave[:n])
    return CorrelationProfile(
        lags=correlation_lags(a.size, b.size, mode="full"),
        correlations=correlate(a, b, mode="full", method="fft"),
    )


def _select_lag(
    profile: CorrelationProfile,
    max_lag_samples: int,
    *,
    peak_exclusion_samples: int,
) -> LagEstimate:
    keep = np.abs(profile.lags) <= min(max_lag_samples, (profile.lags.size - 1) // 2)
    correlations = profile.correlations[keep]
    lags = profile.lags[keep]
    if correlations.size == 0:
        raise ValueError("Lag search range contains no correlation samples.")
    primary_index = int(np.argmax(correlations))
    primary = float(correlations[primary_index])
    lag_samples = int(-lags[primary_index])

    distance = max(1, int(peak_exclusion_samples))
    peak_indices, _ = find_peaks(correlations, distance=distance)
    candidates = [int(index) for index in peak_indices if abs(index - primary_index) >= distance]
    if candidates:
        secondary_index = max(candidates, key=lambda index: correlations[index])
        secondary = float(correlations[secondary_index])
        secondary_lag = int(-lags[secondary_index])
    else:
        secondary = 0.0
        secondary_lag = None
    background = float(np.median(np.abs(correlations)))
    peak_to_background = primary / max(background, np.finfo(float).eps)
    peak_margin_fraction = (primary - secondary) / max(abs(primary), np.finfo(float).eps)
    return LagEstimate(
        lag_samples=lag_samples,
        peak_correlation=primary,
        peak_to_background=peak_to_background,
        peak_margin_fraction=peak_margin_fraction,
        secondary_lag_samples=secondary_lag,
        lags=lags,
        correlations=correlations,
    )


def estimate_lag(
    master: np.ndarray,
    slave: np.ndarray,
    max_lag_samples: int,
    *,
    peak_exclusion_samples: int,
) -> LagEstimate:
    return _select_lag(
        _correlation_profile(master, slave),
        max_lag_samples,
        peak_exclusion_samples=peak_exclusion_samples,
    )


def estimate_lag_narrow_wide(
    master: np.ndarray,
    slave: np.ndarray,
    narrow_max_lag_samples: int,
    wide_max_lag_samples: int,
    *,
    peak_exclusion_samples: int,
) -> tuple[LagEstimate, LagEstimate]:
    """Select narrow and wide candidates from one FFT correlation profile."""

    profile = _correlation_profile(master, slave)
    narrow = _select_lag(
        profile,
        narrow_max_lag_samples,
        peak_exclusion_samples=peak_exclusion_samples,
    )
    if wide_max_lag_samples <= narrow_max_lag_samples:
        return narrow, narrow
    wide = _select_lag(
        profile,
        wide_max_lag_samples,
        peak_exclusion_samples=peak_exclusion_samples,
    )
    return narrow, wide


def _predict_offset(observations: list[SyncObservation], initial_offset: float, time_sec: float) -> float:
    accepted = [observation for observation in observations if observation.accepted]
    if len(accepted) < 4:
        return accepted[-1].observed_offset_samples if accepted else initial_offset
    x = np.asarray([observation.center_time_sec for observation in accepted], dtype=np.float64)
    y = np.asarray([observation.observed_offset_samples for observation in accepted], dtype=np.float64)
    keep = np.ones(x.size, dtype=bool)
    for _ in range(4):
        coefficients = np.polyfit(x[keep], y[keep], 1)
        residuals = y - np.polyval(coefficients, x)
        center = np.median(residuals[keep])
        mad = np.median(np.abs(residuals[keep] - center))
        gate = max(8.0, 6.0 * 1.4826 * mad)
        updated = np.abs(residuals - center) <= gate
        if updated.sum() < 3 or np.array_equal(updated, keep):
            break
        keep = updated
    return float(np.polyval(coefficients, time_sec))


def _usable_master_bounds(
    master: Recording,
    slave: Recording,
    initial_offset_samples: float,
) -> tuple[int, int]:
    """Return the master interval supported by both recordings at initial alignment.

    Endpoint shortening is a normal recording condition, rather than evidence of
    a bad correlation.  Restricting candidate windows to this initial common
    interval keeps those endpoint-only windows out of sync-QC denominators.
    Tracking can still reject a window if a later estimated clock mapping leaves
    this interval, which is a synchronization observation rather than a normal
    endpoint exclusion.
    """

    start = max(0, int(np.ceil(-initial_offset_samples)))
    end = min(master.n_samples, int(np.floor(slave.n_samples - initial_offset_samples)))
    return start, max(start, end)


def _tracking_rejection_reasons(
    estimate: LagEstimate,
    options: SyncOptions,
    search_half_width_samples: int | None = None,
) -> list[str]:
    if search_half_width_samples is None:
        search_half_width_samples = options.tracking_max_lag_samples
    reasons: list[str] = []
    if not np.isfinite(estimate.peak_correlation) or estimate.peak_correlation < options.min_peak_correlation:
        reasons.append("low normalized correlation")
    if estimate.peak_to_background < options.min_peak_to_background:
        reasons.append("low peak/background")
    if estimate.peak_margin_fraction < options.min_peak_margin_fraction:
        reasons.append("ambiguous competing peak")
    if abs(estimate.lag_samples) >= max(1, search_half_width_samples - 1):
        reasons.append("tracking search boundary")
    return reasons


def observe_pair(
    master: Recording,
    slave: Recording,
    master_feature_path: Path,
    slave_feature_path: Path,
    options: SyncOptions,
    *,
    progress: ProgressCallback | None = None,
) -> PairObservations:
    master_feature = feature_memmap(master_feature_path, master.n_samples)
    slave_feature = feature_memmap(slave_feature_path, slave.n_samples)
    try:
        initial_start = round(options.initial_start_seconds * master.fs)
        initial_n = round(options.initial_duration_seconds * master.fs)
        initial_n = min(initial_n, master.n_samples - initial_start, slave.n_samples - initial_start)
        if initial_n < master.fs:
            initial_start = 0
            initial_n = min(master.n_samples, slave.n_samples, max(master.fs, initial_n))
        initial_master = np.asarray(master_feature[initial_start : initial_start + initial_n]).copy()
        initial_slave = np.asarray(slave_feature[initial_start : initial_start + initial_n]).copy()
        initial = estimate_lag(
            initial_master,
            initial_slave,
            round(options.initial_max_lag_seconds * master.fs),
            peak_exclusion_samples=options.peak_exclusion_samples,
        )

        window_samples = max(4, round(options.window_seconds * master.fs))
        step_samples = max(1, round(options.step_seconds * master.fs))
        usable_start, usable_end = _usable_master_bounds(master, slave, float(initial.lag_samples))
        max_master_start = usable_end - window_samples
        observations: list[SyncObservation] = []

        # A long initial correlation can adopt the post-gap level when a loss
        # occurs near recording start. Probe the endpoint independently and
        # exclude the probe interval from publication. If the loss is inside
        # the probe, the uncertain prefix is cropped; if it is later, the
        # probe supplies the pre-gap level for residual/step validation.
        probe_samples = max(4, min(window_samples, round(options.endpoint_probe_seconds * master.fs)))
        probe_start = usable_start
        probe_end = min(usable_end, probe_start + probe_samples)
        if probe_end - probe_start >= 4:
            predicted = float(initial.lag_samples)
            slave_start = probe_start + round(predicted)
            if slave_start >= 0 and slave_start + (probe_end - probe_start) <= slave.n_samples:
                master_probe = np.asarray(master_feature[probe_start:probe_end]).copy()
                slave_probe = np.asarray(
                    slave_feature[slave_start : slave_start + (probe_end - probe_start)]
                ).copy()
                wide_half_width = max(
                    options.tracking_max_lag_samples,
                    round(options.reacquisition_max_lag_seconds * master.fs),
                )
                narrow, wide = estimate_lag_narrow_wide(
                    master_probe,
                    slave_probe,
                    options.tracking_max_lag_samples,
                    wide_half_width,
                    peak_exclusion_samples=options.peak_exclusion_samples,
                )
                reasons = _tracking_rejection_reasons(
                    narrow, options, options.tracking_max_lag_samples
                )
                estimate = narrow
                mode = "endpoint_probe"
                search_width = options.tracking_max_lag_samples
                if reasons:
                    wide_reasons = _tracking_rejection_reasons(wide, options, wide_half_width)
                    if not wide_reasons:
                        estimate = wide
                        reasons = []
                        mode = "endpoint_probe_wide"
                        search_width = wide_half_width
                observations.append(
                    SyncObservation(
                        center_time_sec=(probe_start + (probe_end - probe_start) / 2) / master.fs,
                        predicted_offset_samples=predicted,
                        observed_offset_samples=predicted + estimate.lag_samples,
                        residual_lag_samples=float(estimate.lag_samples),
                        peak_correlation=estimate.peak_correlation,
                        peak_to_background=estimate.peak_to_background,
                        peak_margin_fraction=estimate.peak_margin_fraction,
                        secondary_lag_samples=(
                            predicted + estimate.secondary_lag_samples
                            if estimate.secondary_lag_samples is not None
                            else None
                        ),
                        accepted=not reasons,
                        rejection_reason="; ".join(reasons),
                        search_mode=mode,
                        search_half_width_samples=search_width,
                    )
                )
        validated_start_master_sample = probe_end
        if max_master_start < usable_start:
            starts: list[int] = []
        else:
            starts = list(range(usable_start, max_master_start + 1, step_samples))
            if starts[-1] != max_master_start:
                starts.append(max_master_start)
            if probe_end - probe_start == window_samples and starts[0] == probe_start:
                starts = starts[1:]
        estimated_count = max(1, len(starts))
        for observation_index, master_start in enumerate(starts):
            center_time = (master_start + window_samples / 2) / master.fs
            accepted = [observation for observation in observations if observation.accepted]
            # A local tracker follows a newly reacquired level immediately.
            # The final continuous drift is estimated only after discrete
            # levels have been identified across all pairs.
            predicted = accepted[-1].observed_offset_samples if accepted else float(initial.lag_samples)
            slave_start = master_start + round(predicted)
            if slave_start < 0 or slave_start + window_samples > slave.n_samples:
                observations.append(
                    SyncObservation(
                        center_time_sec=center_time,
                        predicted_offset_samples=predicted,
                        observed_offset_samples=predicted,
                        residual_lag_samples=0.0,
                        peak_correlation=float("nan"),
                        peak_to_background=0.0,
                        peak_margin_fraction=0.0,
                        secondary_lag_samples=None,
                        accepted=False,
                        rejection_reason="tracked window outside slave recording",
                        search_mode="narrow",
                        search_half_width_samples=options.tracking_max_lag_samples,
                    )
                )
                continue
            master_window = np.asarray(master_feature[master_start : master_start + window_samples]).copy()
            slave_window = np.asarray(slave_feature[slave_start : slave_start + window_samples]).copy()
            wide_half_width = max(
                options.tracking_max_lag_samples,
                round(options.reacquisition_max_lag_seconds * master.fs),
            )
            narrow, wide = estimate_lag_narrow_wide(
                master_window,
                slave_window,
                options.tracking_max_lag_samples,
                wide_half_width,
                peak_exclusion_samples=options.peak_exclusion_samples,
            )
            narrow_reasons = _tracking_rejection_reasons(
                narrow,
                options,
                options.tracking_max_lag_samples,
            )
            estimate = narrow
            reasons = narrow_reasons
            search_mode = "narrow"
            search_half_width = options.tracking_max_lag_samples
            if narrow_reasons:
                wide_reasons = _tracking_rejection_reasons(wide, options, wide_half_width)
                if not wide_reasons:
                    estimate = wide
                    reasons = []
                    search_mode = "wide_reacquisition"
                    search_half_width = wide_half_width
            observed = predicted + estimate.lag_samples
            observations.append(
                SyncObservation(
                    center_time_sec=center_time,
                    predicted_offset_samples=predicted,
                    observed_offset_samples=observed,
                    residual_lag_samples=float(estimate.lag_samples),
                    peak_correlation=estimate.peak_correlation,
                    peak_to_background=estimate.peak_to_background,
                    peak_margin_fraction=estimate.peak_margin_fraction,
                    secondary_lag_samples=(
                        predicted + estimate.secondary_lag_samples
                        if estimate.secondary_lag_samples is not None
                        else None
                    ),
                    accepted=not reasons,
                    rejection_reason="; ".join(reasons),
                    search_mode=search_mode,
                    search_half_width_samples=search_half_width,
                )
            )
            if progress is not None:
                progress("sync_observation", 100.0 * (observation_index + 1) / estimated_count)
    finally:
        close_memmap(master_feature)
        close_memmap(slave_feature)
    return PairObservations(
        initial=initial,
        observations=observations,
        initial_master=initial_master,
        initial_slave=initial_slave,
        validated_start_master_sample=validated_start_master_sample,
    )
