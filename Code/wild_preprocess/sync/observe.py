from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Callable

import numpy as np
from scipy.signal import correlate, correlation_lags, find_peaks

from ..binary_io import close_memmap
from ..models import Recording, SyncObservation, SyncOptions
from .features import build_coarse_feature, feature_memmap


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


def _bounded_correlation_profile(
    master: np.ndarray,
    slave: np.ndarray,
    max_lag_samples: int,
) -> CorrelationProfile:
    """Compute only the requested lag band, not a discarded full FFT profile."""

    n = min(master.size, slave.size)
    if n < 4:
        raise ValueError("At least four common-mode samples are required for lag estimation.")
    width = min(max(0, int(max_lag_samples)), n - 1)
    a = _standardize(master[:n])
    b = _standardize(slave[:n])
    lags = np.arange(-width, width + 1, dtype=np.int64)
    correlations = np.empty(lags.size, dtype=np.float64)
    for index, lag in enumerate(lags):
        offset = -int(lag)
        if offset >= 0:
            correlations[index] = np.dot(a[: n - offset], b[offset:])
        else:
            correlations[index] = np.dot(a[-offset:], b[: n + offset])
    return CorrelationProfile(lags=lags, correlations=correlations)


def _estimate_lag_bounded(
    master: np.ndarray,
    slave: np.ndarray,
    max_lag_samples: int,
    *,
    peak_exclusion_samples: int,
) -> LagEstimate:
    return _select_lag(
        _bounded_correlation_profile(master, slave, max_lag_samples),
        max_lag_samples,
        peak_exclusion_samples=peak_exclusion_samples,
    )


def _coarse_window(
    feature: np.ndarray,
    start_sample: int,
    window_samples: int,
    factor: int,
) -> np.ndarray:
    """Return phase-zero-decimated support for one full-rate window."""

    start = max(0, int(np.ceil(start_sample / factor)))
    end = min(feature.size, int(np.ceil((start_sample + window_samples) / factor)))
    return np.asarray(feature[start:end], dtype=np.float64)


def _coarse_reacquire(
    master_feature: np.ndarray,
    slave_feature: np.ndarray,
    master_coarse: np.ndarray,
    slave_coarse: np.ndarray,
    *,
    master_start: int,
    window_samples: int,
    predicted_offset_samples: float,
    fs: float,
    factor: int,
    options: SyncOptions,
) -> tuple[LagEstimate | None, int | None, str, int, str]:
    """Geometrically reacquire at low rate, then refine at full rate.

    The returned integer is the verified full-rate offset.  A coarse search
    edge, operational ceiling, ambiguous coarse peak, or full-rate refinement
    edge is represented as unsupported (``None``), never as a measured lag.
    """

    if factor < 1:
        raise ValueError("coarse downsample factor must be positive")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("sample rate must be finite and positive")
    if options.coarse_reacquisition_max_lag_seconds <= 1.0:
        raise ValueError("coarse reacquisition ceiling must exceed one second")
    if options.coarse_reacquisition_growth_factor <= 1.0:
        raise ValueError("coarse reacquisition growth factor must exceed one")
    ceiling_full = max(1, int(round(options.coarse_reacquisition_max_lag_seconds * fs)))
    minimum_offset = -master_start
    maximum_offset = slave_feature.size - (master_start + window_samples)
    if minimum_offset > maximum_offset:
        return None, None, "no source-supported window for coarse reacquisition", 0, "unsupported"
    reference_offset = int(
        np.clip(round(predicted_offset_samples), minimum_offset, maximum_offset)
    )
    # A recording already at or below the requested coarse rate cannot gain a
    # lower-rate representation.  Preserve the established bounded one-second
    # wide path for this degenerate factor-one case; production recordings use
    # the anti-aliased multi-resolution path below.
    if factor == 1:
        legacy_width = max(
            options.tracking_max_lag_samples,
            int(round(options.reacquisition_max_lag_seconds * fs)),
        )
        legacy_slave_start = master_start + round(predicted_offset_samples)
        if (
            legacy_slave_start < 0
            or legacy_slave_start + window_samples > slave_feature.size
        ):
            return None, None, "legacy wide reacquisition lacks source support", legacy_width, "unsupported"
        try:
            legacy = estimate_lag(
                np.asarray(master_feature[master_start : master_start + window_samples]),
                np.asarray(
                    slave_feature[legacy_slave_start : legacy_slave_start + window_samples]
                ),
                legacy_width,
                peak_exclusion_samples=options.peak_exclusion_samples,
            )
        except (ValueError, FloatingPointError) as error:
            return None, None, f"legacy wide reacquisition unavailable: {error}", legacy_width, "unsupported"
        reasons = _tracking_rejection_reasons(legacy, options, legacy_width)
        if reasons:
            return (
                None,
                None,
                "legacy wide reacquisition unsupported: " + "; ".join(reasons),
                legacy_width,
                "unsupported",
            )
        return (
            legacy,
            int(round(predicted_offset_samples)) + legacy.lag_samples,
            "",
            legacy_width,
            "wide_reacquisition",
        )
    master_window = _coarse_window(master_coarse, master_start, window_samples, factor)
    slave_window = _coarse_window(
        slave_coarse,
        master_start + reference_offset,
        window_samples,
        factor,
    )
    n = min(master_window.size, slave_window.size)
    if n < 4:
        return None, None, "coarse reacquisition has insufficient source support", 0, "unsupported"
    master_window = master_window[:n]
    slave_window = slave_window[:n]
    ceiling_coarse = min(max(1, int(np.ceil(ceiling_full / factor))), n - 1)
    width = min(
        ceiling_coarse,
        max(1, int(np.ceil(options.tracking_max_lag_samples / factor))),
    )
    coarse_peak_exclusion = max(1, int(np.ceil(options.peak_exclusion_samples / factor)))
    while True:
        try:
            estimate = _estimate_lag_bounded(
                master_window,
                slave_window,
                width,
                peak_exclusion_samples=coarse_peak_exclusion,
            )
        except (ValueError, FloatingPointError) as error:
            return None, None, f"coarse reacquisition unavailable: {error}", width * factor, "unsupported"
        reasons = _tracking_rejection_reasons(estimate, options, width)
        failure_reason = "; ".join(reasons)
        refinement: LagEstimate | None = None
        full_offset: int | None = None
        if not reasons:
            candidate_offset = reference_offset + estimate.lag_samples * factor
            if abs(candidate_offset - predicted_offset_samples) > ceiling_full:
                failure_reason = "coarse candidate exceeds configured source-supported ceiling"
            else:
                refinement_samples = max(
                    4,
                    min(window_samples, int(round(options.endpoint_probe_seconds * fs))),
                )
                # Refine on the trailing portion of the tracking window.  When
                # a newly detected loss lies inside the broad window, its
                # trailing side is the candidate post-transition segment.
                refinement_master_start = master_start + window_samples - refinement_samples
                candidate_slave_start = refinement_master_start + candidate_offset
                if candidate_slave_start < 0 or candidate_slave_start + refinement_samples > slave_feature.size:
                    failure_reason = "coarse candidate lacks full-rate source support"
                else:
                    try:
                        refinement = _estimate_lag_bounded(
                            np.asarray(
                                master_feature[
                                    refinement_master_start : refinement_master_start + refinement_samples
                                ]
                            ),
                            np.asarray(
                                slave_feature[
                                    candidate_slave_start : candidate_slave_start + refinement_samples
                                ]
                            ),
                            max(options.tracking_max_lag_samples, 2 * factor),
                            peak_exclusion_samples=options.peak_exclusion_samples,
                        )
                    except (ValueError, FloatingPointError) as error:
                        failure_reason = f"full-rate refinement unavailable: {error}"
                    else:
                        refinement_width = max(options.tracking_max_lag_samples, 2 * factor)
                        refinement_reasons = _tracking_rejection_reasons(
                            refinement, options, refinement_width
                        )
                        full_offset = candidate_offset + refinement.lag_samples
                        if abs(full_offset - predicted_offset_samples) > ceiling_full:
                            refinement_reasons.append("full-rate refinement exceeds configured ceiling")
                        # Full-rate confirmation must be at least as coherent
                        # as the selected coarse peak.  This is a consistency
                        # check between two measurements, not a relaxed
                        # numerical acceptance threshold.
                        if refinement.peak_correlation + 1e-12 < estimate.peak_correlation:
                            refinement_reasons.append("full-rate refinement does not confirm coarse peak")
                        if not refinement_reasons:
                            mode = "wide_reacquisition" if factor == 1 else "coarse_reacquisition"
                            return refinement, full_offset, "", width * factor, mode
                        failure_reason = "full-rate refinement unsupported: " + "; ".join(refinement_reasons)
        if width >= ceiling_coarse:
            return (
                None,
                None,
                "coarse reacquisition unsupported at search ceiling: " + failure_reason,
                width * factor,
                "unsupported",
            )
        width = min(
            ceiling_coarse,
            max(width + 1, int(np.ceil(width * options.coarse_reacquisition_growth_factor))),
        )


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
    master_coarse_feature_path: Path | None = None,
    slave_coarse_feature_path: Path | None = None,
    coarse_downsample_factor: int | None = None,
) -> PairObservations:
    master_feature = feature_memmap(master_feature_path, master.n_samples)
    slave_feature = feature_memmap(slave_feature_path, slave.n_samples)
    coarse_temporary = (
        tempfile.TemporaryDirectory(prefix="wild_coarse_sync_")
        if master_coarse_feature_path is None or slave_coarse_feature_path is None
        else None
    )
    master_coarse: np.memmap | None = None
    slave_coarse: np.memmap | None = None
    try:
        if coarse_temporary is not None:
            temporary_path = Path(coarse_temporary.name)
            master_coarse_path, master_factor = build_coarse_feature(
                master_feature_path,
                master.n_samples,
                temporary_path / "master_coarse.f32",
                fs=master.fs,
                target_rate_hz=options.coarse_feature_rate_hz,
                chunk_seconds=options.chunk_seconds,
            )
            slave_coarse_path, slave_factor = build_coarse_feature(
                slave_feature_path,
                slave.n_samples,
                temporary_path / "slave_coarse.f32",
                fs=slave.fs,
                target_rate_hz=options.coarse_feature_rate_hz,
                chunk_seconds=options.chunk_seconds,
            )
        else:
            if coarse_downsample_factor is None or coarse_downsample_factor < 1:
                raise ValueError("prebuilt coarse features require a positive downsample factor")
            master_coarse_path = Path(master_coarse_feature_path)
            slave_coarse_path = Path(slave_coarse_feature_path)
            master_factor = slave_factor = int(coarse_downsample_factor)
        if master_factor != slave_factor:
            raise ValueError("master and slave coarse feature factors differ")
        master_coarse = feature_memmap(
            master_coarse_path,
            (master.n_samples + master_factor - 1) // master_factor,
        )
        slave_coarse = feature_memmap(
            slave_coarse_path,
            (slave.n_samples + slave_factor - 1) // slave_factor,
        )
        initial_start = round(options.initial_start_seconds * master.fs)
        initial_n = round(options.initial_duration_seconds * master.fs)
        initial_n = min(initial_n, master.n_samples - initial_start, slave.n_samples - initial_start)
        if initial_n < master.fs:
            initial_start = 0
            initial_n = min(master.n_samples, slave.n_samples, max(master.fs, initial_n))
        initial_master = np.asarray(master_feature[initial_start : initial_start + initial_n]).copy()
        initial_slave = np.asarray(slave_feature[initial_start : initial_start + initial_n]).copy()
        try:
            initial = estimate_lag(
                initial_master,
                initial_slave,
                round(options.initial_max_lag_seconds * master.fs),
                peak_exclusion_samples=options.peak_exclusion_samples,
            )
        except (ValueError, FloatingPointError):
            # An uncorrelatable slave is a recoverable all-invalid device, not
            # a pipeline exception.  The explicit non-finite diagnostic makes
            # pair QC fail while allowing the coordinator to publish the
            # canonical master and other verified devices.
            initial = LagEstimate(
                lag_samples=0,
                peak_correlation=float("nan"),
                peak_to_background=0.0,
                peak_margin_fraction=0.0,
                secondary_lag_samples=None,
                lags=np.empty(0, dtype=np.int64),
                correlations=np.empty(0, dtype=np.float64),
            )

        window_samples = max(4, round(options.window_seconds * master.fs))
        step_samples = max(1, round(options.step_seconds * master.fs))
        usable_start, usable_end = _usable_master_bounds(master, slave, float(initial.lag_samples))
        max_master_start = usable_end - window_samples
        observations: list[SyncObservation] = []

        def observe_tracking_window(
            master_start: int,
            count: int,
            predicted: float,
            *,
            default_mode: str,
        ) -> SyncObservation:
            slave_start = master_start + round(predicted)
            narrow: LagEstimate | None = None
            legacy_wide: LagEstimate | None = None
            if 0 <= slave_start and slave_start + count <= slave.n_samples:
                try:
                    legacy_half_width = max(
                        options.tracking_max_lag_samples,
                        round(options.reacquisition_max_lag_seconds * master.fs),
                    )
                    narrow, legacy_wide = estimate_lag_narrow_wide(
                        np.asarray(master_feature[master_start : master_start + count]),
                        np.asarray(slave_feature[slave_start : slave_start + count]),
                        options.tracking_max_lag_samples,
                        legacy_half_width,
                        peak_exclusion_samples=options.peak_exclusion_samples,
                    )
                    reasons = _tracking_rejection_reasons(
                        narrow,
                        options,
                        options.tracking_max_lag_samples,
                    )
                except (ValueError, FloatingPointError) as error:
                    reasons = [f"narrow tracking unavailable: {error}"]
            else:
                reasons = ["tracked window outside slave recording"]
            estimate = narrow
            observed_offset = predicted
            mode = default_mode
            search_width = options.tracking_max_lag_samples
            if not reasons and narrow is not None:
                observed_offset = predicted + narrow.lag_samples
            else:
                # Preserve the established full-rate wide measurement when it
                # is supported.  The coarse geometric path extends this
                # behaviour beyond the legacy one-second range; it does not
                # replace a more precise already-verified full-rate result.
                if legacy_wide is not None:
                    legacy_reasons = _tracking_rejection_reasons(
                        legacy_wide, options, legacy_half_width
                    )
                    if not legacy_reasons:
                        estimate = legacy_wide
                        observed_offset = predicted + legacy_wide.lag_samples
                        reasons = []
                        mode = (
                            "endpoint_probe_wide"
                            if default_mode.startswith("endpoint_probe")
                            else "wide_reacquisition"
                        )
                        search_width = legacy_half_width
                if not reasons:
                    pass
                else:
                    assert master_coarse is not None and slave_coarse is not None
                    refinement, reacquired_offset, reacquire_reason, search_width, reacquire_mode = _coarse_reacquire(
                        master_feature,
                        slave_feature,
                        master_coarse,
                        slave_coarse,
                        master_start=master_start,
                        window_samples=count,
                        predicted_offset_samples=predicted,
                        fs=master.fs,
                        factor=master_factor,
                        options=options,
                    )
                    mode = reacquire_mode
                    if refinement is not None and reacquired_offset is not None:
                        estimate = refinement
                        observed_offset = float(reacquired_offset)
                        reasons = []
                    else:
                        reasons.append(reacquire_reason)
            if estimate is None:
                peak_correlation = float("nan")
                peak_to_background = 0.0
                peak_margin = 0.0
                secondary = None
            else:
                peak_correlation = estimate.peak_correlation
                peak_to_background = estimate.peak_to_background
                peak_margin = estimate.peak_margin_fraction
                secondary = (
                    observed_offset + estimate.secondary_lag_samples
                    if estimate.secondary_lag_samples is not None
                    else None
                )
            return SyncObservation(
                center_time_sec=(master_start + count / 2) / master.fs,
                predicted_offset_samples=predicted,
                observed_offset_samples=observed_offset,
                residual_lag_samples=float(observed_offset - predicted),
                peak_correlation=peak_correlation,
                peak_to_background=peak_to_background,
                peak_margin_fraction=peak_margin,
                secondary_lag_samples=secondary,
                accepted=not reasons,
                rejection_reason="; ".join(reason for reason in reasons if reason),
                search_mode=mode,
                search_half_width_samples=search_width,
            )

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
            observations.append(
                observe_tracking_window(
                    probe_start,
                    probe_end - probe_start,
                    predicted,
                    default_mode="endpoint_probe",
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
            accepted = [observation for observation in observations if observation.accepted]
            # A local tracker follows a newly reacquired level immediately.
            # The final continuous drift is estimated only after discrete
            # levels have been identified across all pairs.
            predicted = accepted[-1].observed_offset_samples if accepted else float(initial.lag_samples)
            observations.append(
                observe_tracking_window(
                    master_start,
                    window_samples,
                    predicted,
                    default_mode="narrow",
                )
            )
            if progress is not None:
                progress("sync_observation", 100.0 * (observation_index + 1) / estimated_count)
    finally:
        if master_coarse is not None:
            close_memmap(master_coarse)
        if slave_coarse is not None:
            close_memmap(slave_coarse)
        if coarse_temporary is not None:
            coarse_temporary.cleanup()
        close_memmap(master_feature)
        close_memmap(slave_feature)
    return PairObservations(
        initial=initial,
        observations=observations,
        initial_master=initial_master,
        initial_slave=initial_slave,
        validated_start_master_sample=validated_start_master_sample,
    )
