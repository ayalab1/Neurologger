"""Independent validation of a staged multi-device amplifier merge.

The synchronisation model is fitted from raw-recording feature windows.  This
module deliberately measures the written, staged ``amplifier.dat`` instead:
it creates a median common-mode trace for each device channel block and
estimates its residual lag relative to the master block.  It is therefore an
output check, rather than another view of the fit observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from ..binary_io import close_memmap
from ..models import (
    ClassifiedInterval,
    DeviceGap,
    DeviceSyncAnchor,
    DeviceSyncSegment,
    Recording,
    validate_device_sync_segments,
)
from .observe import LagEstimate, estimate_lag


_INT16_BYTES = np.dtype("<i2").itemsize
_POSITION_NAMES = ("start", "25%", "50%", "75%", "end")
_POSITION_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class PostMergeMeasurement:
    """One slave/master residual-lag measurement from staged output."""

    position: str
    fraction: float
    nominal_output_sample: int
    window_start_sample: int
    window_end_sample: int
    slave_device_index: int
    lag_samples: int | None
    peak_correlation: float | None
    peak_to_background: float | None
    peak_margin_fraction: float | None
    passed: bool
    message: str = ""
    exclusion_device_indices: tuple[int, ...] = ()
    # The legacy validator used the measured window itself as the exclusion.
    # Segmented validation can make a stronger, data-model-aware statement:
    # an entire independently fitted segment (or its unsupported terminal
    # tail) is invalid.  Keep this optional so existing serialized results and
    # callers remain compatible.
    recommended_canonical_start_sample: int | None = None
    recommended_canonical_end_sample: int | None = None
    # Segment support is diagnostic/corrective authority, not an exclusion
    # request.  It lets repeated measurements refine the segment mapping while
    # isolated or inconsistent failures remain non-destructive warnings.
    segment_canonical_start_sample: int | None = None
    segment_canonical_end_sample: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PostMergeSegmentCorrection:
    """One evidence-gated constant residual-lag correction for a segment."""

    device_index: int
    canonical_start_sample: int
    canonical_end_sample: int
    lag_correction_samples: float
    supporting_measurement_count: int
    reliable_measurement_count: int
    maximum_support_deviation_samples: float
    support_canonical_samples: tuple[int, ...]
    support_lag_samples: tuple[float, ...]
    evidence_positions: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PostMergeValidationResult:
    """Serializable result returned before a staged merge is published."""

    status: str
    message: str
    amplifier_path: str
    master_device_index: int
    n_output_samples: int
    n_output_channels: int
    window_samples: int
    max_allowed_abs_lag_samples: int
    min_peak_correlation: float
    max_abs_lag_samples: float | None
    measurements: tuple[PostMergeMeasurement, ...]

    @property
    def passed(self) -> bool:
        return self.status == "OK"

    @property
    def publishable(self) -> bool:
        return self.status in {"OK", "WARN"}

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["measurements"] = [measurement.to_dict() for measurement in self.measurements]
        return data


def _failure(
    message: str,
    *,
    amplifier_path: Path,
    master_index: int,
    n_output_samples: int = 0,
    n_output_channels: int = 0,
    window_samples: int = 0,
    max_allowed_abs_lag_samples: int = 4,
    min_peak_correlation: float = 0.05,
    measurements: Sequence[PostMergeMeasurement] = (),
) -> PostMergeValidationResult:
    return PostMergeValidationResult(
        status="FAIL",
        message=message,
        amplifier_path=str(amplifier_path),
        master_device_index=master_index + 1,
        n_output_samples=n_output_samples,
        n_output_channels=n_output_channels,
        window_samples=window_samples,
        max_allowed_abs_lag_samples=max_allowed_abs_lag_samples,
        min_peak_correlation=min_peak_correlation,
        max_abs_lag_samples=None,
        measurements=tuple(measurements),
    )


def _device_channel_bounds(recordings: Sequence[Recording]) -> list[tuple[int, int]]:
    start = 0
    bounds: list[tuple[int, int]] = []
    for recording in recordings:
        end = start + recording.n_channels
        bounds.append((start, end))
        start = end
    return bounds


def _common_mode(block: np.ndarray) -> np.ndarray:
    """Return a robust median common-mode trace for one device block."""

    if block.ndim != 2 or block.shape[1] == 0:
        raise ValueError("Merged device channel block is empty.")
    return np.median(np.asarray(block), axis=1).astype(np.float64, copy=False)


def _window_starts(n_samples: int, window_samples: int) -> tuple[tuple[str, float, int, int], ...]:
    """Place endpoint windows at the endpoints and interior windows by center."""

    last_start = n_samples - window_samples
    positions: list[tuple[str, float, int, int]] = []
    for name, fraction in zip(_POSITION_NAMES, _POSITION_FRACTIONS):
        nominal = int(round(fraction * (n_samples - 1)))
        start = int(np.clip(nominal - window_samples // 2, 0, last_start))
        positions.append((name, fraction, nominal, start))
    return tuple(positions)


def _move_window_outside_gaps(
    start: int,
    window_samples: int,
    n_samples: int,
    gaps: Sequence[DeviceGap],
    canonical_start_sample: int,
) -> int:
    """Move a checkpoint minimally so it does not include zero-filled data."""

    candidate = start
    for _ in range(len(gaps) + 1):
        end = candidate + window_samples
        intersections = [
            gap
            for gap in gaps
            if gap.canonical_start_sample - canonical_start_sample < end
            and gap.canonical_end_sample - canonical_start_sample > candidate
        ]
        if not intersections:
            return candidate
        after = max(
            gap.canonical_end_sample - canonical_start_sample for gap in intersections
        )
        if after + window_samples <= n_samples:
            candidate = after
        else:
            before = min(
                gap.canonical_start_sample - canonical_start_sample for gap in intersections
            ) - window_samples
            candidate = max(0, before)
    return candidate


def validate_staged_merge(
    amplifier_path: Path,
    recordings: Sequence[Recording],
    master_index: int,
    *,
    n_output_samples: int | None = None,
    window_seconds: float = 10.0,
    max_lag_samples: int = 100,
    max_allowed_abs_lag_samples: int = 4,
    min_peak_correlation: float = 0.05,
    peak_exclusion_samples: int = 24,
    device_gaps: Sequence[DeviceGap] = (),
    canonical_start_sample: int = 0,
    gap_guard_samples: int = 16,
    validity_path: Path | None = None,
    classified_intervals: Sequence[ClassifiedInterval] = (),
) -> PostMergeValidationResult:
    """Validate residual master/slave lag at five staged-output positions.

    The function returns a ``FAIL`` result with a diagnostic message for
    malformed output or an unmeasurable window.  It does not publish, modify,
    or remove any data, so callers can safely use it as a staging gate.

    ``master_index`` is zero-based, matching :func:`run_multidevice_sync`.
    Device indices in returned measurements are one-based for reports.
    """

    amplifier_path = Path(amplifier_path)
    recordings = tuple(recordings)
    if len(recordings) < 2:
        return _failure(
            "post-merge validation requires at least two recordings",
            amplifier_path=amplifier_path,
            master_index=master_index,
            max_allowed_abs_lag_samples=max_allowed_abs_lag_samples,
            min_peak_correlation=min_peak_correlation,
        )
    if master_index < 0 or master_index >= len(recordings):
        return _failure(
            f"master_index {master_index} is outside {len(recordings)} recordings",
            amplifier_path=amplifier_path,
            master_index=master_index,
            max_allowed_abs_lag_samples=max_allowed_abs_lag_samples,
            min_peak_correlation=min_peak_correlation,
        )
    if not amplifier_path.is_file():
        return _failure(
            f"staged amplifier output does not exist: {amplifier_path}",
            amplifier_path=amplifier_path,
            master_index=master_index,
            max_allowed_abs_lag_samples=max_allowed_abs_lag_samples,
            min_peak_correlation=min_peak_correlation,
        )
    if window_seconds <= 0 or max_lag_samples < 1 or max_allowed_abs_lag_samples < 0:
        return _failure(
            "post-merge validation options must use positive window and lag values",
            amplifier_path=amplifier_path,
            master_index=master_index,
            max_allowed_abs_lag_samples=max_allowed_abs_lag_samples,
            min_peak_correlation=min_peak_correlation,
        )

    fs = recordings[master_index].fs
    if fs <= 0 or any(recording.fs != fs for recording in recordings):
        return _failure(
            "post-merge validation requires equal positive device sample rates",
            amplifier_path=amplifier_path,
            master_index=master_index,
            max_allowed_abs_lag_samples=max_allowed_abs_lag_samples,
            min_peak_correlation=min_peak_correlation,
        )
    total_channels = sum(recording.n_channels for recording in recordings)
    frame_bytes = total_channels * _INT16_BYTES
    byte_count = amplifier_path.stat().st_size
    if total_channels <= 0 or byte_count == 0 or byte_count % frame_bytes:
        return _failure(
            f"staged amplifier size is not divisible by {frame_bytes} bytes per sample frame",
            amplifier_path=amplifier_path,
            master_index=master_index,
            n_output_channels=total_channels,
            max_allowed_abs_lag_samples=max_allowed_abs_lag_samples,
            min_peak_correlation=min_peak_correlation,
        )
    actual_samples = byte_count // frame_bytes
    if n_output_samples is None:
        n_output_samples = actual_samples
    elif n_output_samples != actual_samples:
        return _failure(
            f"staged amplifier has {actual_samples} samples, expected {n_output_samples}",
            amplifier_path=amplifier_path,
            master_index=master_index,
            n_output_samples=actual_samples,
            n_output_channels=total_channels,
            max_allowed_abs_lag_samples=max_allowed_abs_lag_samples,
            min_peak_correlation=min_peak_correlation,
        )

    # A shorter window is used only for short recordings, preserving five
    # separated observations rather than repeating one long endpoint window.
    requested_window = max(4, int(round(window_seconds * fs)))
    window_samples = min(requested_window, max(4, n_output_samples // len(_POSITION_NAMES)))
    if n_output_samples < 5 * 4:
        return _failure(
            "staged amplifier is too short for five independent lag checks",
            amplifier_path=amplifier_path,
            master_index=master_index,
            n_output_samples=n_output_samples,
            n_output_channels=total_channels,
            window_samples=window_samples,
            max_allowed_abs_lag_samples=max_allowed_abs_lag_samples,
            min_peak_correlation=min_peak_correlation,
        )

    bounds = _device_channel_bounds(recordings)
    master_start, master_end = bounds[master_index]
    measurements: list[PostMergeMeasurement] = []
    mapped = np.memmap(
        amplifier_path,
        dtype="<i2",
        mode="r",
        shape=(n_output_samples, total_channels),
        order="C",
    )
    validity: np.memmap | None = None
    if validity_path is not None:
        validity_path = Path(validity_path)
        expected_validity_bytes = n_output_samples * len(recordings)
        if not validity_path.is_file() or validity_path.stat().st_size != expected_validity_bytes:
            close_memmap(mapped)
            return _failure(
                "valid_samples.dat is missing or has an unexpected size",
                amplifier_path=amplifier_path,
                master_index=master_index,
                n_output_samples=n_output_samples,
                n_output_channels=total_channels,
                window_samples=window_samples,
                max_allowed_abs_lag_samples=max_allowed_abs_lag_samples,
                min_peak_correlation=min_peak_correlation,
            )
        validity = np.memmap(
            validity_path,
            dtype=np.uint8,
            mode="r",
            shape=(n_output_samples, len(recordings)),
            order="C",
        )
    validity_order = [master_index, *(index for index in range(len(recordings)) if index != master_index)]
    validity_channel = {
        device_index: channel for channel, device_index in enumerate(validity_order)
    }

    def find_valid_window(start: int, slave_index: int) -> int | None:
        if validity is None:
            return start
        last_start = n_output_samples - window_samples
        channels = [0, validity_channel[slave_index]]
        for distance in range(0, last_start + window_samples, window_samples):
            candidates = (start - distance, start + distance) if distance else (start,)
            for candidate in candidates:
                candidate = int(np.clip(candidate, 0, last_start))
                if np.all(validity[candidate : candidate + window_samples, channels]):
                    return candidate
        return None

    def window_is_valid(start: int, count: int, slave_index: int) -> bool:
        if validity is None:
            return True
        channels = [0, validity_channel[slave_index]]
        return bool(np.all(validity[start : start + count, channels]))
    def measure_window(
        position: str,
        fraction: float,
        nominal: int,
        start: int,
        count: int,
        target_slaves: Sequence[int],
        exclusion_device_indices: tuple[int, ...] = (),
    ) -> None:
        end = start + count
        try:
            master_common = _common_mode(mapped[start:end, master_start:master_end])
        except (ValueError, FloatingPointError) as error:
            for slave_index in target_slaves:
                measurements.append(
                    PostMergeMeasurement(
                        position=position,
                        fraction=fraction,
                        nominal_output_sample=nominal,
                        window_start_sample=start,
                        window_end_sample=end,
                        slave_device_index=slave_index + 1,
                        lag_samples=None,
                        peak_correlation=None,
                        peak_to_background=None,
                        peak_margin_fraction=None,
                        passed=False,
                        message=f"master common-mode unavailable: {error}",
                        exclusion_device_indices=exclusion_device_indices,
                    )
                )
            return
        for slave_index in target_slaves:
            slave_start, slave_end = bounds[slave_index]
            try:
                slave_common = _common_mode(mapped[start:end, slave_start:slave_end])
                estimate = estimate_lag(
                    master_common,
                    slave_common,
                    max_lag_samples,
                    peak_exclusion_samples=peak_exclusion_samples,
                )
                lag_ok = abs(estimate.lag_samples) <= max_allowed_abs_lag_samples
                correlation_ok = estimate.peak_correlation >= min_peak_correlation
                reasons: list[str] = []
                if not lag_ok:
                    reasons.append(
                        f"residual lag {estimate.lag_samples} exceeds "
                        f"{max_allowed_abs_lag_samples} samples"
                    )
                if not correlation_ok:
                    reasons.append(
                        f"peak correlation {estimate.peak_correlation:.4g} below "
                        f"{min_peak_correlation:.4g}"
                    )
                measurements.append(
                    PostMergeMeasurement(
                        position=position,
                        fraction=fraction,
                        nominal_output_sample=nominal,
                        window_start_sample=start,
                        window_end_sample=end,
                        slave_device_index=slave_index + 1,
                        lag_samples=estimate.lag_samples,
                        peak_correlation=estimate.peak_correlation,
                        peak_to_background=estimate.peak_to_background,
                        peak_margin_fraction=estimate.peak_margin_fraction,
                        passed=lag_ok and correlation_ok,
                        message="; ".join(reasons),
                        exclusion_device_indices=exclusion_device_indices,
                    )
                )
            except (ValueError, FloatingPointError) as error:
                measurements.append(
                    PostMergeMeasurement(
                        position=position,
                        fraction=fraction,
                        nominal_output_sample=nominal,
                        window_start_sample=start,
                        window_end_sample=end,
                        slave_device_index=slave_index + 1,
                        lag_samples=None,
                        peak_correlation=None,
                        peak_to_background=None,
                        peak_margin_fraction=None,
                        passed=False,
                        message=f"lag measurement unavailable: {error}",
                        exclusion_device_indices=exclusion_device_indices,
                    )
                )

    all_slaves = [index for index in range(len(recordings)) if index != master_index]
    try:
        for position, fraction, nominal, start in _window_starts(n_output_samples, window_samples):
            start = _move_window_outside_gaps(
                start,
                window_samples,
                n_output_samples,
                device_gaps,
                canonical_start_sample,
            )
            for slave_index in all_slaves:
                valid_start = find_valid_window(start, slave_index)
                if valid_start is not None:
                    measure_window(
                        position,
                        fraction,
                        nominal,
                        valid_start,
                        window_samples,
                        [slave_index],
                    )

        boundary_window_samples = max(4, min(window_samples, int(round(fs))))
        mapping_intervals = [
            interval
            for interval in classified_intervals
            if interval.kind in {"missing", "unresolved_boundary"}
        ]
        if not any(interval.kind == "missing" for interval in mapping_intervals):
            mapping_intervals.extend(
                ClassifiedInterval(
                    affected_device_indices=(gap.device_index,),
                    canonical_start_sample=gap.canonical_start_sample,
                    canonical_end_sample=gap.canonical_end_sample,
                    kind="missing",
                    action="zero_fill",
                    confidence=gap.confidence,
                    evidence=gap.evidence,
                )
                for gap in device_gaps
            )

        def same_side_valid_start(
            *,
            side: str,
            boundary_start: int,
            boundary_end: int,
            slave_index: int,
        ) -> int | None:
            if side == "before":
                candidate = boundary_start - gap_guard_samples - boundary_window_samples
                while candidate >= 0:
                    if window_is_valid(candidate, boundary_window_samples, slave_index):
                        return candidate
                    candidate -= boundary_window_samples
                return None
            candidate = boundary_end + gap_guard_samples
            while candidate + boundary_window_samples <= n_output_samples:
                if window_is_valid(candidate, boundary_window_samples, slave_index):
                    return candidate
                candidate += boundary_window_samples
            return None

        for interval_index, interval in enumerate(mapping_intervals, start=1):
            boundary_start = interval.canonical_start_sample - canonical_start_sample
            boundary_end = interval.canonical_end_sample - canonical_start_sample
            if boundary_end <= 0 or boundary_start >= n_output_samples:
                continue
            affected = {index - 1 for index in interval.affected_device_indices}
            target_slaves = (
                all_slaves
                if master_index in affected
                else [index for index in all_slaves if index in affected]
            )
            if not target_slaves:
                target_slaves = all_slaves
            exclusion_devices = interval.affected_device_indices
            for side in ("before", "after"):
                for slave_index in target_slaves:
                    start = same_side_valid_start(
                        side=side,
                        boundary_start=boundary_start,
                        boundary_end=boundary_end,
                        slave_index=slave_index,
                    )
                    position = f"boundary{interval_index}_{side}"
                    if start is None:
                        measurements.append(
                            PostMergeMeasurement(
                                position=position,
                                fraction=float(np.clip(boundary_start, 0, n_output_samples - 1))
                                / max(1, n_output_samples - 1),
                                nominal_output_sample=boundary_start,
                                window_start_sample=max(0, min(boundary_start, n_output_samples)),
                                window_end_sample=max(0, min(boundary_start, n_output_samples)),
                                slave_device_index=slave_index + 1,
                                lag_samples=None,
                                peak_correlation=None,
                                peak_to_background=None,
                                peak_margin_fraction=None,
                                passed=False,
                                message=f"no valid {side} window on the same side of the mapping boundary",
                                exclusion_device_indices=exclusion_devices,
                            )
                        )
                        continue
                    measure_window(
                        position,
                        start / max(1, n_output_samples - 1),
                        start,
                        start,
                        boundary_window_samples,
                        [slave_index],
                        exclusion_device_indices=exclusion_devices,
                    )
    finally:
        close_memmap(mapped)
        if validity is not None:
            close_memmap(validity)

    measured_lags = [abs(measurement.lag_samples) for measurement in measurements if measurement.lag_samples is not None]
    max_abs_lag = float(max(measured_lags)) if measured_lags else None
    # Endpoint common-mode windows can occasionally have too little shared
    # signal to meet the correlation threshold even when the written data are
    # aligned.  Treat one such *unmeasurable* checkpoint as diagnostic rather
    # than rejecting an otherwise independently verified device.  A lag above
    # the hard limit is never ignored when the measurement is reliable.
    device_failures: list[str] = []
    recoverable_warnings: list[str] = []
    ignored_details: list[str] = []
    slave_indices = [index for index in range(len(recordings)) if index != master_index]
    for slave_index in slave_indices:
        device_measurements = [
            measurement
            for measurement in measurements
            if measurement.slave_device_index == slave_index + 1
        ]
        global_measurements = [
            measurement
            for measurement in device_measurements
            if not measurement.position.startswith(("gap", "boundary"))
        ]
        gap_measurements = [
            measurement
            for measurement in device_measurements
            if measurement.position.startswith(("gap", "boundary"))
        ]
        failed_gap_measurements = [measurement for measurement in gap_measurements if not measurement.passed]
        unavailable_gap_measurements = [
            measurement
            for measurement in failed_gap_measurements
            if measurement.window_end_sample <= measurement.window_start_sample
        ]
        recoverable_gap_measurements = [
            measurement
            for measurement in failed_gap_measurements
            if measurement not in unavailable_gap_measurements
        ]
        if unavailable_gap_measurements:
            details = ", ".join(
                f"{measurement.position}={measurement.message or 'failed'}"
                for measurement in unavailable_gap_measurements
            )
            device_failures.append(
                f"slave {slave_index + 1} has no valid same-side boundary window ({details})"
            )
        if recoverable_gap_measurements:
            details = ", ".join(
                f"{measurement.position}={measurement.message or 'failed'}"
                for measurement in recoverable_gap_measurements
            )
            recoverable_warnings.append(
                f"slave {slave_index + 1} has unverified gap-boundary windows ({details})"
            )
        reliable = [
            measurement
            for measurement in global_measurements
            if measurement.lag_samples is not None
            and measurement.peak_correlation is not None
            and measurement.peak_correlation >= min_peak_correlation
        ]
        reliable_lag_failures = [
            measurement
            for measurement in reliable
            if abs(measurement.lag_samples) > max_allowed_abs_lag_samples
        ]
        unreliable = [measurement for measurement in global_measurements if measurement not in reliable]

        passed_global = [measurement for measurement in global_measurements if measurement.passed]
        recoverable_endpoint_failures = [
            measurement
            for measurement in reliable_lag_failures
            if measurement.position in {"start", "end"}
        ]
        if (
            len(global_measurements) == len(_POSITION_NAMES)
            and len(passed_global) == len(_POSITION_NAMES) - 1
            and len(recoverable_endpoint_failures) == 1
            and len(reliable_lag_failures) == 1
        ):
            endpoint = recoverable_endpoint_failures[0]
            recoverable_warnings.append(
                f"slave {slave_index + 1} has one unverified {endpoint.position} window "
                f"({endpoint.message})"
            )
            reliable_lag_failures = []

        if reliable_lag_failures:
            details = ", ".join(
                f"{measurement.position}={measurement.lag_samples}" for measurement in reliable_lag_failures
            )
            device_failures.append(
                f"slave {slave_index + 1} has reliable residual lag above "
                f"{max_allowed_abs_lag_samples} samples ({details})"
            )
            continue
        if len(reliable) < 4:
            device_failures.append(
                f"slave {slave_index + 1} has only {len(reliable)} of {len(_POSITION_NAMES)} reliable checkpoints"
            )
            continue
        if len(unreliable) > 1:
            device_failures.append(
                f"slave {slave_index + 1} has {len(unreliable)} low-confidence or unavailable checkpoints"
            )
            continue
        if unreliable:
            measurement = unreliable[0]
            reason = measurement.message or "unavailable measurement"
            ignored_details.append(
                f"slave {slave_index + 1} ignored {measurement.position} checkpoint ({reason})"
            )

    if device_failures:
        return PostMergeValidationResult(
            status="FAIL",
            message="post-merge validation failed: " + "; ".join(device_failures),
            amplifier_path=str(amplifier_path),
            master_device_index=master_index + 1,
            n_output_samples=n_output_samples,
            n_output_channels=total_channels,
            window_samples=window_samples,
            max_allowed_abs_lag_samples=max_allowed_abs_lag_samples,
            min_peak_correlation=min_peak_correlation,
            max_abs_lag_samples=max_abs_lag,
            measurements=tuple(measurements),
        )
    if recoverable_warnings:
        return PostMergeValidationResult(
            status="WARN",
            message=(
                "post-merge validation found localized unverified windows that require "
                "zero-fill exclusion: " + "; ".join(recoverable_warnings)
            ),
            amplifier_path=str(amplifier_path),
            master_device_index=master_index + 1,
            n_output_samples=n_output_samples,
            n_output_channels=total_channels,
            window_samples=window_samples,
            max_allowed_abs_lag_samples=max_allowed_abs_lag_samples,
            min_peak_correlation=min_peak_correlation,
            max_abs_lag_samples=max_abs_lag,
            measurements=tuple(measurements),
        )
    return PostMergeValidationResult(
        status="OK",
        message=(
            f"all reliable post-merge lag checks passed; {'; '.join(ignored_details)}"
            if ignored_details
            else f"all {len(measurements)} post-merge lag checks passed"
        ),
        amplifier_path=str(amplifier_path),
        master_device_index=master_index + 1,
        n_output_samples=n_output_samples,
        n_output_channels=total_channels,
        window_samples=window_samples,
        max_allowed_abs_lag_samples=max_allowed_abs_lag_samples,
        min_peak_correlation=min_peak_correlation,
        max_abs_lag_samples=max_abs_lag,
        measurements=tuple(measurements),
    )


def _hierarchical_lag(
    master: np.ndarray,
    slave: np.ndarray,
    *,
    narrow_max_lag_samples: int,
    wide_max_lag_samples: int,
    peak_exclusion_samples: int,
) -> tuple[LagEstimate, bool]:
    """Measure a residual, expanding only when narrow search is exhausted.

    A peak at the narrow edge is evidence that the requested search interval
    was too small, never evidence that the physical residual equals that edge.
    The returned boolean records whether a wider search was required.
    """

    narrow = estimate_lag(
        master,
        slave,
        narrow_max_lag_samples,
        peak_exclusion_samples=peak_exclusion_samples,
    )
    if abs(narrow.lag_samples) < narrow_max_lag_samples:
        return narrow, False
    # A correlation at lag L needs a substantial overlapping support on both
    # sides of zero.  Do not turn a short final-QC island into a meaningless
    # wide search that compares only a handful of samples.
    ceiling = min(
        max(1, int(wide_max_lag_samples)),
        (min(master.size, slave.size) - 1) // 2,
    )
    width = min(ceiling, max(narrow_max_lag_samples + 1, narrow_max_lag_samples * 2))
    result = narrow
    while width > narrow_max_lag_samples:
        result = estimate_lag(
            master,
            slave,
            width,
            peak_exclusion_samples=peak_exclusion_samples,
        )
        if abs(result.lag_samples) < width or width >= ceiling:
            return result, True
        next_width = min(ceiling, width * 2)
        if next_width == width:
            break
        width = next_width
    return result, True


def validate_segment_staged_merge(
    amplifier_path: Path,
    recordings: Sequence[Recording],
    master_index: int,
    *,
    device_segments: Sequence[DeviceSyncSegment],
    validity_path: Path,
    canonical_start_sample: int = 0,
    n_output_samples: int | None = None,
    window_seconds: float = 10.0,
    max_lag_samples: int = 100,
    wide_max_lag_samples: int | None = None,
    max_allowed_abs_lag_samples: int = 4,
    min_peak_correlation: float = 0.05,
    min_peak_to_background: float = 1.2,
    min_peak_margin_fraction: float = 0.01,
    peak_exclusion_samples: int = 24,
    dense_step_seconds: float | None = None,
    structural_only: bool = False,
) -> PostMergeValidationResult:
    """Verify staged output against independently fitted device segments.

    This is deliberately a pure staging check: it neither changes the DATs
    nor inherits a source step.  It validates global checkpoints, every
    publishable slave-segment interior, and both sides of each publishable
    join using short contiguous jointly-valid islands (up to one second).
    An unavailable island is a recoverable ``WARN``. Reliable residual lag is
    recorded against its independently fitted segment so repeated consistent
    measurements can refine the mapping. No correlation measurement directly
    recommends zero-filling measured neural data.

    ``device_segments`` is flat and uses one-based device indices, matching
    :class:`DeviceSyncSegment`.  Empty slave collections are permitted only
    when their entire validity channel and output channel block are zero.
    """

    amplifier_path = Path(amplifier_path)
    validity_path = Path(validity_path)
    recordings = tuple(recordings)
    if len(recordings) < 2:
        return _failure(
            "segment post-merge validation requires at least two recordings",
            amplifier_path=amplifier_path,
            master_index=master_index,
            max_allowed_abs_lag_samples=max_allowed_abs_lag_samples,
            min_peak_correlation=min_peak_correlation,
        )
    if master_index < 0 or master_index >= len(recordings):
        return _failure(
            f"master_index {master_index} is outside {len(recordings)} recordings",
            amplifier_path=amplifier_path,
            master_index=master_index,
            max_allowed_abs_lag_samples=max_allowed_abs_lag_samples,
            min_peak_correlation=min_peak_correlation,
        )
    if (
        window_seconds <= 0
        or max_lag_samples < 1
        or max_allowed_abs_lag_samples < 0
        or (dense_step_seconds is not None and dense_step_seconds <= 0)
    ):
        return _failure(
            "segment post-merge validation options must use positive window and lag values",
            amplifier_path=amplifier_path,
            master_index=master_index,
            max_allowed_abs_lag_samples=max_allowed_abs_lag_samples,
            min_peak_correlation=min_peak_correlation,
        )
    fs = recordings[master_index].fs
    if fs <= 0 or any(recording.fs != fs for recording in recordings):
        return _failure(
            "segment post-merge validation requires equal positive device sample rates",
            amplifier_path=amplifier_path,
            master_index=master_index,
            max_allowed_abs_lag_samples=max_allowed_abs_lag_samples,
            min_peak_correlation=min_peak_correlation,
        )
    if not amplifier_path.is_file():
        return _failure(
            f"staged amplifier output does not exist: {amplifier_path}",
            amplifier_path=amplifier_path,
            master_index=master_index,
            max_allowed_abs_lag_samples=max_allowed_abs_lag_samples,
            min_peak_correlation=min_peak_correlation,
        )

    total_channels = sum(recording.n_channels for recording in recordings)
    frame_bytes = total_channels * _INT16_BYTES
    byte_count = amplifier_path.stat().st_size
    if total_channels <= 0 or byte_count == 0 or byte_count % frame_bytes:
        return _failure(
            f"staged amplifier size is not divisible by {frame_bytes} bytes per sample frame",
            amplifier_path=amplifier_path,
            master_index=master_index,
            n_output_channels=total_channels,
            max_allowed_abs_lag_samples=max_allowed_abs_lag_samples,
            min_peak_correlation=min_peak_correlation,
        )
    actual_samples = byte_count // frame_bytes
    if n_output_samples is None:
        n_output_samples = actual_samples
    elif n_output_samples != actual_samples:
        return _failure(
            f"staged amplifier has {actual_samples} samples, expected {n_output_samples}",
            amplifier_path=amplifier_path,
            master_index=master_index,
            n_output_samples=actual_samples,
            n_output_channels=total_channels,
            max_allowed_abs_lag_samples=max_allowed_abs_lag_samples,
            min_peak_correlation=min_peak_correlation,
        )
    expected_validity_bytes = n_output_samples * len(recordings)
    if not validity_path.is_file() or validity_path.stat().st_size != expected_validity_bytes:
        return _failure(
            "valid_samples.dat is missing or has an unexpected size",
            amplifier_path=amplifier_path,
            master_index=master_index,
            n_output_samples=n_output_samples,
            n_output_channels=total_channels,
            max_allowed_abs_lag_samples=max_allowed_abs_lag_samples,
            min_peak_correlation=min_peak_correlation,
        )
    if n_output_samples < 8:
        return _failure(
            "staged amplifier is too short for segment post-merge validation",
            amplifier_path=amplifier_path,
            master_index=master_index,
            n_output_samples=n_output_samples,
            n_output_channels=total_channels,
            max_allowed_abs_lag_samples=max_allowed_abs_lag_samples,
            min_peak_correlation=min_peak_correlation,
        )

    segment_groups: dict[int, tuple[DeviceSyncSegment, ...]] = {}
    try:
        raw_groups: dict[int, list[DeviceSyncSegment]] = {index: [] for index in range(1, len(recordings) + 1)}
        for segment in device_segments:
            if not isinstance(segment, DeviceSyncSegment):
                raise ValueError("device_segments must contain DeviceSyncSegment instances")
            if segment.device_index not in raw_groups:
                raise ValueError(f"segment uses unknown device index {segment.device_index}")
            raw_groups[segment.device_index].append(segment)
        segment_groups = {
            index: validate_device_sync_segments(group, device_index=index)
            for index, group in raw_groups.items()
        }
    except ValueError as error:
        return _failure(
            f"invalid device segment mapping: {error}",
            amplifier_path=amplifier_path,
            master_index=master_index,
            n_output_samples=n_output_samples,
            n_output_channels=total_channels,
            max_allowed_abs_lag_samples=max_allowed_abs_lag_samples,
            min_peak_correlation=min_peak_correlation,
        )

    requested_window = max(4, int(round(window_seconds * fs)))
    min_island_samples = max(4, 2 * max_lag_samples + 1)
    # Explicit validity gaps can be dense; require a local valid island, not
    # an uninterrupted configured (historically 10-second) window.
    maximum_window_samples = min(
        int(fs), max(4, n_output_samples // len(_POSITION_NAMES))
    )
    window_samples = min(
        max(requested_window, min_island_samples), maximum_window_samples
    )
    lag_support_available = maximum_window_samples >= min_island_samples
    bounds = _device_channel_bounds(recordings)
    validity_order = [master_index, *(index for index in range(len(recordings)) if index != master_index)]
    validity_channel = {device_index: channel for channel, device_index in enumerate(validity_order)}
    mapped = np.memmap(amplifier_path, dtype="<i2", mode="r", shape=(n_output_samples, total_channels), order="C")
    validity = np.memmap(validity_path, dtype=np.uint8, mode="r", shape=(n_output_samples, len(recordings)), order="C")
    measurements: list[PostMergeMeasurement] = []
    structural_errors: list[str] = []
    warnings: list[str] = []
    validation_chunk_samples = 1_000_000

    def local_bounds(segment: DeviceSyncSegment) -> tuple[int, int]:
        return (
            max(0, segment.canonical_start_sample - canonical_start_sample),
            min(n_output_samples, segment.canonical_end_sample - canonical_start_sample),
        )

    try:
        # Validate the central invariant before any correlation can be trusted:
        # each claimed-valid output coordinate has exactly one publishable
        # segment authority for that device.  The validity memmap can be tens
        # of gigabytes, so all validation uses bounded chunks.
        master_has_valid = False
        for device_index, (channel_start, channel_end) in enumerate(bounds, start=1):
            mapping_error = False
            zero_error = False
            for chunk_start in range(0, n_output_samples, validation_chunk_samples):
                chunk_end = min(n_output_samples, chunk_start + validation_chunk_samples)
                valid_chunk = validity[chunk_start:chunk_end]
                if np.any((valid_chunk != 0) & (valid_chunk != 1)):
                    structural_errors.append("valid_samples.dat contains values other than 0 or 1")
                    # Continue this chunk: a malformed value must not hide a
                    # second structural mapping/zero-fill violation.
                if device_index == master_index + 1 and np.any(
                    valid_chunk[:, validity_channel[master_index]] == 1
                ):
                    master_has_valid = True
                support = np.zeros(chunk_end - chunk_start, dtype=bool)
                for segment in segment_groups[device_index]:
                    if not segment.is_publishable:
                        continue
                    segment_start, segment_end = local_bounds(segment)
                    start = max(chunk_start, segment_start)
                    end = min(chunk_end, segment_end)
                    if end > start:
                        support[start - chunk_start : end - chunk_start] = True
                valid_column = valid_chunk[:, validity_channel[device_index - 1]] == 1
                if np.any(valid_column & ~support):
                    mapping_error = True
                invalid = ~valid_column
                if np.any(invalid) and np.any(
                    mapped[chunk_start:chunk_end, channel_start:channel_end][invalid] != 0
                ):
                    zero_error = True
            if mapping_error:
                structural_errors.append(
                    f"device {device_index} claims valid samples without a publishable segment mapping"
                )
            if zero_error:
                structural_errors.append(
                    f"device {device_index} has non-zero output samples marked invalid"
                )
        if not master_has_valid:
            structural_errors.append("canonical master has no valid staged samples")
        if structural_errors:
            return _failure(
                "segment post-merge validation failed: " + "; ".join(structural_errors),
                amplifier_path=amplifier_path,
                master_index=master_index,
                n_output_samples=n_output_samples,
                n_output_channels=total_channels,
                window_samples=window_samples,
                max_allowed_abs_lag_samples=max_allowed_abs_lag_samples,
                min_peak_correlation=min_peak_correlation,
            )
        if structural_only:
            return PostMergeValidationResult(
                status="OK",
                message=(
                    "post-correction structural mapping, validity, and zero-fill "
                    "contracts passed"
                ),
                amplifier_path=str(amplifier_path),
                master_device_index=master_index + 1,
                n_output_samples=n_output_samples,
                n_output_channels=total_channels,
                window_samples=window_samples,
                max_allowed_abs_lag_samples=max_allowed_abs_lag_samples,
                min_peak_correlation=min_peak_correlation,
                max_abs_lag_samples=None,
                measurements=(),
            )

        master_start, master_end = bounds[master_index]

        def find_window(
            desired_sample: int,
            lower: int,
            upper: int,
            slave_index: int,
        ) -> tuple[int, int] | None:
            """Stream a segment and retain only its nearest valid island."""

            lower = max(0, lower)
            upper = min(n_output_samples, upper)
            if not lag_support_available or upper - lower < min_island_samples:
                return None
            master_channel = validity_channel[master_index]
            slave_channel = validity_channel[slave_index]

            def best_run(search_lower: int, search_upper: int) -> tuple[int, int] | None:
                best: tuple[int, int] | None = None
                best_rank: tuple[int, int, int] | None = None
                open_start: int | None = None

                def consider(candidate: tuple[int, int]) -> None:
                    nonlocal best, best_rank
                    start, end = candidate
                    if end - start < min_island_samples:
                        return
                    distance = 0 if start <= desired_sample < end else min(
                        abs(desired_sample - start),
                        abs(desired_sample - (end - 1)),
                    )
                    rank = (distance, -(end - start), start)
                    if best_rank is None or rank < best_rank:
                        best = candidate
                        best_rank = rank

                for chunk_start in range(
                    search_lower, search_upper, validation_chunk_samples
                ):
                    chunk_end = min(
                        search_upper, chunk_start + validation_chunk_samples
                    )
                    joint = (
                        validity[chunk_start:chunk_end, master_channel] == 1
                    ) & (validity[chunk_start:chunk_end, slave_channel] == 1)
                    if open_start is not None and (not joint.size or not joint[0]):
                        consider((open_start, chunk_start))
                        open_start = None
                    padded = np.empty(joint.size + 2, dtype=bool)
                    padded[0] = False
                    padded[-1] = False
                    padded[1:-1] = joint
                    edges = np.flatnonzero(padded[1:] != padded[:-1])
                    for local_start, local_end in zip(edges[::2], edges[1::2]):
                        start = chunk_start + int(local_start)
                        end = chunk_start + int(local_end)
                        run_start = (
                            open_start
                            if local_start == 0 and open_start is not None
                            else start
                        )
                        if local_end == joint.size and joint[-1]:
                            open_start = run_start
                        else:
                            consider((run_start, end))
                            open_start = None
                if open_start is not None:
                    consider((open_start, search_upper))
                return best

            total = upper - lower
            search_span = min(
                total, max(validation_chunk_samples, min_island_samples)
            )
            island: tuple[int, int] | None = None
            while True:
                search_lower = max(lower, desired_sample - search_span // 2)
                search_upper = min(upper, search_lower + search_span)
                search_lower = max(lower, search_upper - search_span)
                island = best_run(search_lower, search_upper)
                if island is not None:
                    break
                if search_lower == lower and search_upper == upper:
                    return None
                search_span = min(total, search_span * 2)

            island_start, island_end = island
            count = min(window_samples, island_end - island_start)
            start = int(
                np.clip(
                    desired_sample - count // 2,
                    island_start,
                    island_end - count,
                )
            )
            return start, count

        def measure(
            *,
            position: str,
            fraction: float,
            nominal: int,
            window: tuple[int, int] | None,
            slave_index: int,
            segment_bounds: tuple[int, int],
            segment_identity_bounds: tuple[int, int],
        ) -> None:
            segment_start, segment_end = segment_identity_bounds
            if window is None:
                measurements.append(PostMergeMeasurement(
                    position=position,
                    fraction=fraction,
                    nominal_output_sample=nominal,
                    window_start_sample=max(0, min(n_output_samples, nominal)),
                    window_end_sample=max(0, min(n_output_samples, nominal)),
                    slave_device_index=slave_index + 1,
                    lag_samples=None,
                    peak_correlation=None,
                    peak_to_background=None,
                    peak_margin_fraction=None,
                    passed=False,
                    message="no jointly valid verification window",
                    exclusion_device_indices=(slave_index + 1,),
                    segment_canonical_start_sample=segment_start,
                    segment_canonical_end_sample=segment_end,
                ))
                return
            start, count = window
            end = start + count
            try:
                master_common = _common_mode(mapped[start:end, master_start:master_end])
                slave_start, slave_end = bounds[slave_index]
                slave_common = _common_mode(mapped[start:end, slave_start:slave_end])
                estimate, expanded = _hierarchical_lag(
                    master_common,
                    slave_common,
                    narrow_max_lag_samples=max_lag_samples,
                    wide_max_lag_samples=(wide_max_lag_samples if wide_max_lag_samples is not None else max(max_lag_samples * 4, int(fs * 30))),
                    peak_exclusion_samples=peak_exclusion_samples,
                )
                lag_ok = abs(estimate.lag_samples) <= max_allowed_abs_lag_samples
                correlation_ok = estimate.peak_correlation >= min_peak_correlation
                background_ok = estimate.peak_to_background >= min_peak_to_background
                margin_ok = estimate.peak_margin_fraction >= min_peak_margin_fraction
                peak_reliable = correlation_ok and background_ok and margin_ok
                reasons: list[str] = []
                if not lag_ok:
                    reasons.append(f"residual lag {estimate.lag_samples} exceeds {max_allowed_abs_lag_samples} samples")
                if not correlation_ok:
                    reasons.append(f"peak correlation {estimate.peak_correlation:.4g} below {min_peak_correlation:.4g}")
                if not background_ok:
                    reasons.append(
                        f"peak/background {estimate.peak_to_background:.4g} below "
                        f"{min_peak_to_background:.4g}"
                    )
                if not margin_ok:
                    reasons.append(
                        f"peak margin {estimate.peak_margin_fraction:.4g} below "
                        f"{min_peak_margin_fraction:.4g}"
                    )
                if expanded:
                    reasons.append("hierarchical wide lag search used")
                measurements.append(PostMergeMeasurement(
                    position=position,
                    fraction=fraction,
                    nominal_output_sample=nominal,
                    window_start_sample=start,
                    window_end_sample=end,
                    slave_device_index=slave_index + 1,
                    lag_samples=estimate.lag_samples,
                    peak_correlation=estimate.peak_correlation,
                    peak_to_background=estimate.peak_to_background,
                    peak_margin_fraction=estimate.peak_margin_fraction,
                    passed=lag_ok and peak_reliable,
                    message="; ".join(reasons),
                    exclusion_device_indices=(slave_index + 1,),
                    segment_canonical_start_sample=segment_start,
                    segment_canonical_end_sample=segment_end,
                ))
            except (ValueError, FloatingPointError) as error:
                measurements.append(PostMergeMeasurement(
                    position=position,
                    fraction=fraction,
                    nominal_output_sample=nominal,
                    window_start_sample=start,
                    window_end_sample=end,
                    slave_device_index=slave_index + 1,
                    lag_samples=None,
                    peak_correlation=None,
                    peak_to_background=None,
                    peak_margin_fraction=None,
                    passed=False,
                    message=f"lag measurement unavailable: {error}",
                    exclusion_device_indices=(slave_index + 1,),
                    segment_canonical_start_sample=segment_start,
                    segment_canonical_end_sample=segment_end,
                ))

        for slave_index in range(len(recordings)):
            if slave_index == master_index:
                continue
            slave_has_valid = any(
                np.any(
                    validity[start:min(n_output_samples, start + validation_chunk_samples),
                    validity_channel[slave_index]] == 1
                )
                for start in range(0, n_output_samples, validation_chunk_samples)
            )
            if not slave_has_valid:
                measurements.append(PostMergeMeasurement(
                    position=f"device{slave_index + 1}_all_invalid",
                    fraction=0.0,
                    nominal_output_sample=0,
                    window_start_sample=0,
                    window_end_sample=0,
                    slave_device_index=slave_index + 1,
                    lag_samples=None,
                    peak_correlation=None,
                    peak_to_background=None,
                    peak_margin_fraction=None,
                    passed=False,
                    message="slave has no publishable staged samples; retained as all-invalid",
                    exclusion_device_indices=(slave_index + 1,),
                ))
                warnings.append(f"slave {slave_index + 1} is all-invalid")
                continue
            publishable = [segment for segment in segment_groups[slave_index + 1] if segment.is_publishable]
            # Five global checkpoints are still checked, but each must be
            # supported by an actual independently publishable segment.
            for name, fraction, nominal, _desired_start in _window_starts(
                n_output_samples, window_samples
            ):
                containing = next(
                    (segment for segment in publishable if local_bounds(segment)[0] <= nominal < local_bounds(segment)[1]),
                    None,
                )
                if containing is None:
                    containing = min(publishable, key=lambda segment: abs((local_bounds(segment)[0] + local_bounds(segment)[1]) // 2 - nominal))
                lower, upper = local_bounds(containing)
                measure(
                    position=f"global_{name}", fraction=fraction, nominal=nominal,
                    window=find_window(nominal, lower, upper, slave_index),
                    slave_index=slave_index,
                    segment_bounds=(lower, upper),
                    segment_identity_bounds=(
                        containing.canonical_start_sample,
                        containing.canonical_end_sample,
                    ),
                )
            for segment_number, segment in enumerate(publishable, start=1):
                lower, upper = local_bounds(segment)
                midpoint = lower + (upper - lower) // 2
                measure(
                    position=f"segment{slave_index + 1}_{segment_number}_interior",
                    fraction=midpoint / max(1, n_output_samples - 1), nominal=midpoint,
                    window=find_window(midpoint, lower, upper, slave_index),
                    slave_index=slave_index,
                    segment_bounds=(lower, upper),
                    segment_identity_bounds=(
                        segment.canonical_start_sample,
                        segment.canonical_end_sample,
                    ),
                )
                if dense_step_seconds is not None:
                    dense_step = max(1, int(round(dense_step_seconds * fs)))
                    dense_number = 0
                    for dense_nominal in range(lower + dense_step // 2, upper, dense_step):
                        if abs(dense_nominal - midpoint) < max(1, window_samples // 2):
                            continue
                        dense_number += 1
                        measure(
                            position=(
                                f"segment{slave_index + 1}_{segment_number}_dense"
                                f"{dense_number}"
                            ),
                            fraction=dense_nominal / max(1, n_output_samples - 1),
                            nominal=dense_nominal,
                            window=find_window(
                                dense_nominal, lower, upper, slave_index
                            ),
                            slave_index=slave_index,
                            segment_bounds=(lower, upper),
                            segment_identity_bounds=(
                                segment.canonical_start_sample,
                                segment.canonical_end_sample,
                            ),
                        )
            for join_number, (before, after) in enumerate(zip(publishable, publishable[1:]), start=1):
                before_lower, before_upper = local_bounds(before)
                after_lower, after_upper = local_bounds(after)
                measure(
                    position=f"segment{slave_index + 1}_join{join_number}_before",
                    fraction=before_upper / max(1, n_output_samples - 1), nominal=before_upper,
                    window=find_window(
                        before_upper - 1, before_lower, before_upper, slave_index
                    ),
                    slave_index=slave_index,
                    segment_bounds=(before_lower, before_upper),
                    segment_identity_bounds=(
                        before.canonical_start_sample,
                        before.canonical_end_sample,
                    ),
                )
                measure(
                    position=f"segment{slave_index + 1}_join{join_number}_after",
                    fraction=after_lower / max(1, n_output_samples - 1), nominal=after_lower,
                    window=find_window(
                        after_lower, after_lower, after_upper, slave_index
                    ),
                    slave_index=slave_index,
                    segment_bounds=(after_lower, after_upper),
                    segment_identity_bounds=(
                        after.canonical_start_sample,
                        after.canonical_end_sample,
                    ),
                )
            # A master-only missing interval splits the canonical master's
            # mapping while a slave can remain one continuous segment.  Those
            # canonical joins still require same-side pair validation.
            master_publishable = [
                segment
                for segment in segment_groups[master_index + 1]
                if segment.is_publishable
            ]
            for join_number, (master_before, master_after) in enumerate(
                zip(master_publishable, master_publishable[1:]), start=1
            ):
                _, before_upper = local_bounds(master_before)
                after_lower, _ = local_bounds(master_after)
                if after_lower <= before_upper:
                    continue
                before_segment = next(
                    (
                        segment
                        for segment in publishable
                        if local_bounds(segment)[0] < before_upper <= local_bounds(segment)[1]
                    ),
                    None,
                )
                after_segment = next(
                    (
                        segment
                        for segment in publishable
                        if local_bounds(segment)[0] <= after_lower < local_bounds(segment)[1]
                    ),
                    None,
                )
                if before_segment is not None:
                    lower, upper = local_bounds(before_segment)
                    measure(
                        position=f"boundary_master_join{join_number}_before",
                        fraction=before_upper / max(1, n_output_samples - 1),
                        nominal=before_upper,
                        window=find_window(
                            before_upper - 1, lower, upper, slave_index
                        ),
                        slave_index=slave_index,
                        segment_bounds=(lower, upper),
                        segment_identity_bounds=(
                            before_segment.canonical_start_sample,
                            before_segment.canonical_end_sample,
                        ),
                    )
                if after_segment is not None:
                    lower, upper = local_bounds(after_segment)
                    measure(
                        position=f"boundary_master_join{join_number}_after",
                        fraction=after_lower / max(1, n_output_samples - 1),
                        nominal=after_lower,
                        window=find_window(after_lower, lower, upper, slave_index),
                        slave_index=slave_index,
                        segment_bounds=(lower, upper),
                        segment_identity_bounds=(
                            after_segment.canonical_start_sample,
                            after_segment.canonical_end_sample,
                        ),
                    )
    finally:
        close_memmap(mapped)
        close_memmap(validity)

    max_abs_lag = max((abs(item.lag_samples) for item in measurements if item.lag_samples is not None), default=None)
    measured_failures = [
        item for item in measurements if not item.passed and item.lag_samples is not None
    ]
    warnings.extend(
        f"slave {item.slave_device_index} {item.position} has a measured local verification failure"
        for item in measured_failures
    )
    local_failures = [item for item in measurements if not item.passed and not item.position.endswith("_all_invalid")]
    if local_failures or warnings:
        detail = "; ".join(warnings + [f"slave {item.slave_device_index} {item.position}: {item.message or 'unverified'}" for item in local_failures])
        return PostMergeValidationResult(
            status="WARN",
            message="segment post-merge validation found alignment warnings: " + detail,
            amplifier_path=str(amplifier_path), master_device_index=master_index + 1,
            n_output_samples=n_output_samples, n_output_channels=total_channels,
            window_samples=window_samples, max_allowed_abs_lag_samples=max_allowed_abs_lag_samples,
            min_peak_correlation=min_peak_correlation, max_abs_lag_samples=float(max_abs_lag) if max_abs_lag is not None else None,
            measurements=tuple(measurements),
        )
    return PostMergeValidationResult(
        status="OK", message=f"all {len(measurements)} segment-aware post-merge lag checks passed",
        amplifier_path=str(amplifier_path), master_device_index=master_index + 1,
        n_output_samples=n_output_samples, n_output_channels=total_channels,
        window_samples=window_samples, max_allowed_abs_lag_samples=max_allowed_abs_lag_samples,
        min_peak_correlation=min_peak_correlation, max_abs_lag_samples=float(max_abs_lag) if max_abs_lag is not None else None,
        measurements=tuple(measurements),
    )


def _measurement_peak_is_reliable(
    measurement: PostMergeMeasurement,
    *,
    min_peak_correlation: float,
    min_peak_to_background: float,
    min_peak_margin_fraction: float,
) -> bool:
    return bool(
        measurement.lag_samples is not None
        and measurement.peak_correlation is not None
        and measurement.peak_to_background is not None
        and measurement.peak_margin_fraction is not None
        and np.isfinite(measurement.peak_correlation)
        and np.isfinite(measurement.peak_to_background)
        and np.isfinite(measurement.peak_margin_fraction)
        and measurement.peak_correlation >= min_peak_correlation
        and measurement.peak_to_background >= min_peak_to_background
        and measurement.peak_margin_fraction >= min_peak_margin_fraction
    )


def infer_postmerge_segment_corrections(
    result: PostMergeValidationResult,
    *,
    canonical_start_sample: int,
    min_peak_to_background: float = 1.2,
    min_peak_margin_fraction: float = 0.01,
    minimum_supporting_measurements: int = 3,
    maximum_lag_deviation_samples: float = 2.0,
    minimum_consistent_fraction: float = 0.6,
) -> tuple[PostMergeSegmentCorrection, ...]:
    """Infer only repeated, internally consistent constant-lag corrections.

    A single reliable failure remains a warning.  A correction requires a
    majority plateau across at least three independently located windows in
    the same fitted segment.  The renderer later reuses raw input, so this
    function never shifts an already interpolated staged waveform.
    """

    if minimum_supporting_measurements < 3:
        raise ValueError("minimum_supporting_measurements must be at least three")
    if maximum_lag_deviation_samples < 0:
        raise ValueError("maximum_lag_deviation_samples must be non-negative")
    if not 0 < minimum_consistent_fraction <= 1:
        raise ValueError("minimum_consistent_fraction must be in (0, 1]")
    grouped: dict[tuple[int, int, int], list[PostMergeMeasurement]] = {}
    for measurement in result.measurements:
        start = measurement.segment_canonical_start_sample
        end = measurement.segment_canonical_end_sample
        if start is None or end is None or end <= start:
            continue
        if not _measurement_peak_is_reliable(
            measurement,
            min_peak_correlation=result.min_peak_correlation,
            min_peak_to_background=min_peak_to_background,
            min_peak_margin_fraction=min_peak_margin_fraction,
        ):
            continue
        # Tiny edge islands can identify a local warning but are not strong
        # enough to move a complete raw-source mapping.
        if measurement.window_end_sample - measurement.window_start_sample < max(
            4, result.window_samples
        ):
            continue
        grouped.setdefault((measurement.slave_device_index, start, end), []).append(
            measurement
        )

    corrections: list[PostMergeSegmentCorrection] = []
    for (device_index, start, end), measurements in sorted(grouped.items()):
        ordered = sorted(
            measurements,
            key=lambda item: (
                (item.window_start_sample + item.window_end_sample) // 2,
                item.position,
            ),
        )
        lags = np.asarray([float(item.lag_samples) for item in ordered], dtype=np.float64)
        median = float(np.median(lags))
        consistent = np.abs(lags - median) <= maximum_lag_deviation_samples
        support_count = int(np.count_nonzero(consistent))
        if support_count < minimum_supporting_measurements:
            continue
        if support_count / len(ordered) < minimum_consistent_fraction:
            continue
        support = [item for item, keep in zip(ordered, consistent) if keep]
        correction = float(np.median([float(item.lag_samples) for item in support]))
        if abs(correction) <= result.max_allowed_abs_lag_samples:
            continue
        support_samples = tuple(
            canonical_start_sample
            + (item.window_start_sample + item.window_end_sample) // 2
            for item in support
        )
        # A valid island selected near a nominal checkpoint can coincide with
        # another checkpoint. Collapse duplicate support coordinates.
        unique: dict[int, PostMergeMeasurement] = {}
        for sample, item in zip(support_samples, support):
            unique.setdefault(sample, item)
        if len(unique) < minimum_supporting_measurements:
            continue
        samples = tuple(sorted(unique))
        support = [unique[sample] for sample in samples]
        support_lags = tuple(float(item.lag_samples) for item in support)
        corrections.append(
            PostMergeSegmentCorrection(
                device_index=device_index,
                canonical_start_sample=start,
                canonical_end_sample=end,
                lag_correction_samples=correction,
                supporting_measurement_count=len(samples),
                reliable_measurement_count=len(ordered),
                maximum_support_deviation_samples=max(
                    abs(value - correction) for value in support_lags
                ),
                support_canonical_samples=samples,
                support_lag_samples=support_lags,
                evidence_positions=tuple(item.position for item in support),
            )
        )
    return tuple(corrections)


def apply_postmerge_segment_corrections(
    device_segments: Sequence[DeviceSyncSegment],
    corrections: Sequence[PostMergeSegmentCorrection],
) -> tuple[tuple[DeviceSyncSegment, ...], tuple[PostMergeSegmentCorrection, ...], tuple[str, ...]]:
    """Apply safe correction candidates without source reuse or extrapolation."""

    current = list(device_segments)
    applied: list[PostMergeSegmentCorrection] = []
    rejected: list[str] = []
    for correction in corrections:
        index = next(
            (
                position
                for position, segment in enumerate(current)
                if segment.device_index == correction.device_index
                and segment.canonical_start_sample == correction.canonical_start_sample
                and segment.canonical_end_sample == correction.canonical_end_sample
            ),
            None,
        )
        if index is None:
            rejected.append(
                f"device {correction.device_index} segment "
                f"[{correction.canonical_start_sample},{correction.canonical_end_sample}) "
                "was not found"
            )
            continue
        segment = current[index]
        intercept = segment.source_intercept_samples + correction.lag_correction_samples
        scale = segment.source_scale
        supported_start = max(
            segment.canonical_start_sample,
            int(np.ceil((segment.source_start_sample - intercept - 1e-9) / scale)),
        )
        supported_end = min(
            segment.canonical_end_sample,
            int(np.floor((segment.source_end_sample - 1 - intercept + 1e-9) / scale)) + 1,
        )
        anchors: list[DeviceSyncAnchor] = []
        residuals: list[float] = []
        for canonical, lag in zip(
            correction.support_canonical_samples, correction.support_lag_samples
        ):
            if not supported_start <= canonical < supported_end:
                continue
            source = (
                segment.source_scale * canonical
                + segment.source_intercept_samples
                + lag
            )
            residual = source - (scale * canonical + intercept)
            anchors.append(
                DeviceSyncAnchor(
                    canonical_sample=canonical,
                    source_sample=source,
                    verified=True,
                    confidence="medium",
                    evidence="repeated reliable staged residual-lag correction",
                )
            )
            residuals.append(float(residual))
        if supported_end <= supported_start or len(anchors) < 2:
            rejected.append(
                f"device {correction.device_index} correction lacks bounded raw support"
            )
            continue
        rms = float(np.sqrt(np.mean(np.square(residuals))))
        maximum = float(np.max(np.abs(residuals)))
        candidate = replace(
            segment,
            canonical_start_sample=supported_start,
            canonical_end_sample=supported_end,
            source_intercept_samples=intercept,
            anchors=tuple(anchors),
            residual_rms_samples=rms,
            residual_max_abs_samples=maximum,
            confidence="medium",
            start_transition="postmerge_corrected",
            end_transition="postmerge_corrected",
            publishable=True,
            evidence=(
                f"post-merge correction {correction.lag_correction_samples:+.3f} samples "
                f"from {correction.supporting_measurement_count} consistent windows"
            ),
        )
        proposed = list(current)
        proposed[index] = candidate
        try:
            for device_index in sorted({item.device_index for item in proposed}):
                validate_device_sync_segments(
                    [item for item in proposed if item.device_index == device_index],
                    device_index=device_index,
                )
        except ValueError as error:
            rejected.append(
                f"device {correction.device_index} correction rejected: {error}"
            )
            continue
        current = proposed
        applied.append(correction)
    return tuple(current), tuple(applied), tuple(rejected)


def postmerge_alignment_warning_intervals(
    result: PostMergeValidationResult,
    *,
    canonical_start_sample: int,
) -> tuple[dict[str, object], ...]:
    """Describe warned support without converting uncertainty into zero-fill."""

    intervals: dict[tuple[int, int, int], set[str]] = {}
    for measurement in result.measurements:
        if measurement.passed or measurement.position.endswith("_all_invalid"):
            continue
        # A failed checkpoint establishes uncertainty only in the measured
        # window. Repeated, reliable constant lag is handled separately by the
        # segment-correction path; an isolated or ambiguous peak does not
        # justify masking a complete multi-hour segment.
        start = canonical_start_sample + measurement.window_start_sample
        end = canonical_start_sample + measurement.window_end_sample
        if end <= start:
            continue
        key = (measurement.slave_device_index, int(start), int(end))
        intervals.setdefault(key, set()).add(measurement.position)
    return tuple(
        {
            "affected_device_indices": [device],
            "canonical_start_sample": start,
            "canonical_end_sample": end,
            "kind": "alignment_warn",
            "action": "keep_measured_data",
            "evidence": "post-merge QC warning: " + ", ".join(sorted(labels)),
        }
        for (device, start, end), labels in sorted(intervals.items())
    )


def postmerge_exclusion_intervals(
    result: PostMergeValidationResult,
    *,
    canonical_start_sample: int,
    device_count: int,
) -> tuple[ClassifiedInterval, ...]:
    """Return only explicit zero-fill recommendations from a WARN result.

    Segment validation no longer emits these.  The function remains for
    compatibility with callers that construct an explicit recommendation;
    unavailable or low-confidence boundary measurements never fall through to
    an implicit one-second exclusion.
    """

    if result.status != "WARN" or device_count < 1:
        return ()
    output_end = canonical_start_sample + result.n_output_samples
    all_devices = tuple(range(1, device_count + 1))
    candidates: list[tuple[tuple[int, ...], int, int, str]] = []
    for measurement in result.measurements:
        if measurement.passed:
            continue
        # Segment-aware validation records its evidence-derived interval in
        # canonical coordinates.  It intentionally takes precedence over the
        # compact correlation window so a caller can patch once, not iterate in
        # fixed one-second increments.
        if (
            measurement.recommended_canonical_start_sample is not None
            and measurement.recommended_canonical_end_sample is not None
        ):
            devices = measurement.exclusion_device_indices or all_devices
            start = max(canonical_start_sample, measurement.recommended_canonical_start_sample)
            end = min(output_end, measurement.recommended_canonical_end_sample)
            if devices and end > start:
                candidates.append((devices, start, end, measurement.position))
            continue
        continue

    merged: list[ClassifiedInterval] = []
    for devices in sorted({item[0] for item in candidates}):
        spans = sorted(
            (start, end, label)
            for candidate_devices, start, end, label in candidates
            if candidate_devices == devices
        )
        current_start: int | None = None
        current_end = 0
        labels: list[str] = []
        for start, end, label in spans:
            if current_start is None or start > current_end:
                if current_start is not None:
                    merged.append(
                        ClassifiedInterval(
                            affected_device_indices=devices,
                            canonical_start_sample=current_start,
                            canonical_end_sample=current_end,
                            kind="postmerge_unverified",
                            action="zero_fill",
                            confidence="unresolved",
                            evidence="post-merge QC exclusion: " + ", ".join(labels),
                        )
                    )
                current_start = start
                current_end = end
                labels = [label]
            else:
                current_end = max(current_end, end)
                if label not in labels:
                    labels.append(label)
        if current_start is not None:
            merged.append(
                ClassifiedInterval(
                    affected_device_indices=devices,
                    canonical_start_sample=current_start,
                    canonical_end_sample=current_end,
                    kind="postmerge_unverified",
                    action="zero_fill",
                    confidence="unresolved",
                    evidence="post-merge QC exclusion: " + ", ".join(labels),
                )
            )
    return tuple(sorted(merged, key=lambda interval: interval.canonical_start_sample))
