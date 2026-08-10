"""Independent validation of a staged multi-device amplifier merge.

The synchronisation model is fitted from raw-recording feature windows.  This
module deliberately measures the written, staged ``amplifier.dat`` instead:
it creates a median common-mode trace for each device channel block and
estimates its residual lag relative to the master block.  It is therefore an
output check, rather than another view of the fit observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..binary_io import close_memmap
from ..models import DeviceGap, Recording
from .observe import estimate_lag


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
    def measure_window(
        position: str,
        fraction: float,
        nominal: int,
        start: int,
        count: int,
        target_slaves: Sequence[int],
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
            measure_window(position, fraction, nominal, start, window_samples, all_slaves)

        # Five global checkpoints can miss a boundary error between them.
        gap_window_samples = max(4, min(window_samples, int(round(fs))))
        for gap_index, gap in enumerate(device_gaps, start=1):
            gap_start = gap.canonical_start_sample - canonical_start_sample
            gap_end = gap.canonical_end_sample - canonical_start_sample
            if gap_end <= 0 or gap_start >= n_output_samples:
                continue
            target_slaves = (
                all_slaves if gap.device_index == master_index + 1 else [gap.device_index - 1]
            )
            checks = (
                ("before", gap_start - gap_guard_samples - gap_window_samples),
                ("after", gap_end + gap_guard_samples),
            )
            for side, start in checks:
                if start < 0 or start + gap_window_samples > n_output_samples:
                    continue
                measure_window(
                    f"gap{gap_index}_{side}",
                    start / max(1, n_output_samples - 1),
                    start,
                    start,
                    gap_window_samples,
                    target_slaves,
                )
    finally:
        close_memmap(mapped)

    measured_lags = [abs(measurement.lag_samples) for measurement in measurements if measurement.lag_samples is not None]
    max_abs_lag = float(max(measured_lags)) if measured_lags else None
    # Endpoint common-mode windows can occasionally have too little shared
    # signal to meet the correlation threshold even when the written data are
    # aligned.  Treat one such *unmeasurable* checkpoint as diagnostic rather
    # than rejecting an otherwise independently verified device.  A lag above
    # the hard limit is never ignored when the measurement is reliable.
    device_failures: list[str] = []
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
            if not measurement.position.startswith("gap")
        ]
        gap_measurements = [
            measurement
            for measurement in device_measurements
            if measurement.position.startswith("gap")
        ]
        failed_gap_measurements = [measurement for measurement in gap_measurements if not measurement.passed]
        if failed_gap_measurements:
            details = ", ".join(
                f"{measurement.position}={measurement.message or 'failed'}"
                for measurement in failed_gap_measurements
            )
            device_failures.append(
                f"slave {slave_index + 1} failed gap-boundary validation ({details})"
            )
            continue
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
