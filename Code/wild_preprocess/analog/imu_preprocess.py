"""Production IMU resampling, scaling, and calibration.

This module deliberately covers only the deterministic pre-fusion portion of
``WILD_processIMU`` / ``WILD_scaleIMU``.  It neither calls nor approximates
MATLAB's proprietary ``ahrsfilter``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.signal import firls, upfirdn
from scipy.signal.windows import kaiser


IMU_RAW_SAMPLE_RATE_HZ = 1_250.0
IMU_SAMPLE_RATE_HZ = 100.0
IMU_RESAMPLE_UP = 2
IMU_RESAMPLE_DOWN = 25
IMU_RESAMPLE_NEIGHBOUR_TERMS = 10
IMU_RESAMPLE_KAISER_BETA = 5.0


@dataclass(frozen=True)
class ImuAxes:
    """One nine-axis IMU representation using the N-by-3 convention."""

    acc: np.ndarray
    gyr: np.ndarray
    mag: np.ndarray

    def __post_init__(self) -> None:
        shapes = (self.acc.shape, self.gyr.shape, self.mag.shape)
        if any(len(shape) != 2 or shape[1] != 3 for shape in shapes):
            raise ValueError("IMU axes must each be N-by-3")
        if len({shape[0] for shape in shapes}) != 1:
            raise ValueError("IMU axes must have equal row counts")

    def as_matrix(self) -> np.ndarray:
        """Return the axes as an N-by-9 matrix in analog-channel order."""

        return np.column_stack((self.acc, self.gyr, self.mag))


@dataclass(frozen=True)
class ImuPrefusionData:
    """Resampled and calibrated IMU data before sensor fusion."""

    resampled_adc: np.ndarray
    nominal: ImuAxes
    calibrated: ImuAxes
    timestamp_seconds: np.ndarray
    sample_rate_hz: float = IMU_SAMPLE_RATE_HZ

    def __post_init__(self) -> None:
        rows = self.resampled_adc.shape[0]
        if self.resampled_adc.ndim != 2 or self.resampled_adc.shape[1] != 9:
            raise ValueError("resampled ADC data must be N-by-9")
        if self.nominal.acc.shape[0] != rows or self.calibrated.acc.shape[0] != rows:
            raise ValueError("IMU axis rows must match resampled ADC rows")
        if self.timestamp_seconds.shape != (rows,):
            raise ValueError("timestamps must have one entry per resampled row")


def design_imu_resample_filter() -> np.ndarray:
    """Return MATLAB R2024b ``resample(x,100,1250)`` default FIR coefficients.

    MATLAB reduces the ratio to ``p=2, q=25`` and, with its default N=10 and
    beta=5, uses a 501-tap ``firls``/Kaiser design at the upsampled rate.  The
    returned coefficients are the unpadded filter that MATLAB exposes as the
    second output of ``resample``.
    """

    p = IMU_RESAMPLE_UP
    q = IMU_RESAMPLE_DOWN
    length = 2 * IMU_RESAMPLE_NEIGHBOUR_TERMS * max(p, q) + 1
    cutoff = 1.0 / max(p, q)
    # MATLAB's ``firls(L - 1, ...)`` specifies filter *order*.  SciPy's
    # ``firls`` instead takes number of taps, hence ``length`` here.
    coefficients = firls(length, (0.0, cutoff, cutoff, 1.0), (1.0, 1.0, 0.0, 0.0))
    coefficients *= kaiser(length, IMU_RESAMPLE_KAISER_BETA)
    coefficients *= p / np.sum(coefficients, dtype=np.float64)
    return np.asarray(coefficients, dtype=np.float64)


def _delay_padded_resample_filter(
    sample_count: int, coefficients: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """Reproduce R2024b ``findDelayAndZeroPadFilter`` exactly for one length."""

    if sample_count < 1:
        raise ValueError("MATLAB resample requires at least one input row")
    p = IMU_RESAMPLE_UP
    q = IMU_RESAMPLE_DOWN
    length = int(coefficients.size)
    half_length = (length - 1) // 2
    if length % 2 != 1:
        raise ValueError("MATLAB legacy resample filter must have odd length")
    zero_begin = int(math.floor(q - (half_length % q)))
    padded = np.concatenate((np.zeros(zero_begin, dtype=np.float64), coefficients))
    delay = int(math.floor(math.ceil(half_length + zero_begin) / q))
    output_count = int(math.ceil(sample_count * p / q))
    zero_end = 0
    while int(math.ceil(((sample_count - 1) * p + padded.size + zero_end) / q)) - delay < output_count:
        zero_end += 1
    if zero_end:
        padded = np.concatenate((padded, np.zeros(zero_end, dtype=np.float64)))
    return padded, delay, output_count


def resample_imu_1250_to_100(values: np.ndarray) -> np.ndarray:
    """Match MATLAB R2024b ``resample(values,100,1250)`` along rows.

    Values are treated as MATLAB's N-by-channel double matrix.  MATLAB uses
    zero padding outside the supplied support, compensates its linear-phase
    delay, and returns exactly ``ceil(2*N/25)`` rows.
    """

    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise ValueError("MATLAB resample input must be a non-empty N-by-channel matrix")
    if not np.issubdtype(matrix.dtype, np.number) or np.iscomplexobj(matrix):
        raise ValueError("MATLAB IMU resample input must be real numeric")
    source = np.asarray(matrix, dtype=np.float64)
    if not np.all(np.isfinite(source)):
        raise ValueError("MATLAB IMU resample input must be finite")
    padded_filter, delay, output_count = _delay_padded_resample_filter(
        source.shape[0], design_imu_resample_filter()
    )
    filtered = upfirdn(
        padded_filter,
        source,
        up=IMU_RESAMPLE_UP,
        down=IMU_RESAMPLE_DOWN,
        axis=0,
    )
    output = np.asarray(filtered[delay : delay + output_count], dtype=np.float64)
    if output.shape != (output_count, source.shape[1]):
        raise RuntimeError("MATLAB-compatible resample produced an unexpected output shape")
    return output


def scale_imu_nominal(resampled_adc: np.ndarray) -> ImuAxes:
    """Apply the nine-axis unit conversion in ``WILD_scaleIMU``."""

    values = np.asarray(resampled_adc, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 9:
        raise ValueError("resampled ADC data must be N-by-9")
    if not np.all(np.isfinite(values)):
        raise ValueError("resampled ADC data must be finite")
    acc = values[:, 0:3] * (8.0 * 9.8 / 32768.0)
    gyr = values[:, 3:6] * (2000.0 * np.pi / (180.0 * 32768.0))
    mag = values[:, 6:9] * np.asarray((1150.0, 1150.0, 2500.0), dtype=np.float64) / 32768.0
    return ImuAxes(acc=acc, gyr=gyr, mag=mag)


def calibrate_imu(nominal: ImuAxes) -> ImuAxes:
    """Apply the deterministic calibration portion of ``WILD_scaleIMU``.

    Acceleration is normalized to 9.81 m/s² by the session median vector
    norm, gyroscope has a per-axis median bias removed, and magnetometer is
    intentionally left unchanged.  Sensor fusion is explicitly out of scope.
    """

    acc_norm = np.sqrt(np.sum(nominal.acc * nominal.acc, axis=1))
    median_norm = float(np.median(acc_norm))
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = nominal.acc / median_norm * 9.81
    gyr = nominal.gyr - np.median(nominal.gyr, axis=0)
    return ImuAxes(acc=acc, gyr=gyr, mag=nominal.mag.copy())


def prepare_imu_prefusion(raw_adc: np.ndarray) -> ImuPrefusionData:
    """Run production resampling, nominal scaling, and calibration."""

    source = np.asarray(raw_adc)
    if source.ndim != 2 or source.shape[1] != 9:
        raise ValueError("raw IMU ADC input must be N-by-9")
    resampled = resample_imu_1250_to_100(source)
    nominal = scale_imu_nominal(resampled)
    calibrated = calibrate_imu(nominal)
    timestamp = np.arange(resampled.shape[0], dtype=np.float64) / IMU_SAMPLE_RATE_HZ
    return ImuPrefusionData(
        resampled_adc=resampled,
        nominal=nominal,
        calibrated=calibrated,
        timestamp_seconds=timestamp,
    )
