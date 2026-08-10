"""Robust affine inference for packed PC-clock observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decode import PACKED_PC_MOD_MS, unpack_packed_updates


ROBUST_SEED_INLIER_MS = 250.0
ROBUST_RESIDUAL_FLOOR_MS = 150.0
ROBUST_RESIDUAL_MAD_SCALE = 6.0 * 1.4826


@dataclass(frozen=True)
class PcTimeModel:
    """One affine fit plus residuals for every decoded packed update.

    ``keep_mask`` identifies observations used for the robust affine fit.  It
    must not be used to discard the ordered evidence when validating a clock
    step or a rate-regime change: ``residual_ms`` and ``pc_unwrapped_ms``
    always retain one value for every decoded update.
    """

    device_ms: np.ndarray
    pc_unwrapped_ms: np.ndarray
    delay_ms: np.ndarray
    residual_ms: np.ndarray
    keep_mask: np.ndarray
    slope: float
    intercept_ms: float
    slope_sem: float
    intercept_sem_ms: float
    recording_start_ms: int

    @property
    def drift_ppm(self) -> float:
        return (self.slope - 1.0) * 1_000_000.0

    @property
    def kept_count(self) -> int:
        return int(np.count_nonzero(self.keep_mask))

    @property
    def residual_rms_ms(self) -> float:
        values = self.residual_ms[self.keep_mask]
        return float(np.sqrt(np.mean(np.square(values)))) if values.size else float("nan")

    def predict_unwrapped_ms(self, device_ms: np.ndarray | float) -> np.ndarray:
        return self.slope * np.asarray(device_ms, dtype=float) + self.intercept_ms


def _align_intercept(intercept: float, recording_start_ms: int) -> float:
    return intercept + round((recording_start_ms - intercept) / PACKED_PC_MOD_MS) * PACKED_PC_MOD_MS


def _lift(modulo_ms: np.ndarray, device_ms: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    predicted = slope * device_ms + intercept
    cycles = np.rint((predicted - modulo_ms) / PACKED_PC_MOD_MS)
    return modulo_ms + cycles * PACKED_PC_MOD_MS


def _line_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size == 0:
        return 1.0, 0.0
    if x.size == 1 or np.ptp(x) == 0:
        return 1.0, float(y.mean() - x.mean())
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _standard_errors(x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> tuple[float, float]:
    if x.size < 3:
        return float("nan"), float("nan")
    centered = x - x.mean()
    sxx = float(np.dot(centered, centered))
    if sxx <= 0:
        return float("nan"), float("nan")
    mse = float(np.sum(np.square(y - (slope * x + intercept))) / (x.size - 2))
    return float(np.sqrt(mse / sxx)), float(np.sqrt(mse * (1.0 / x.size + x.mean() ** 2 / sxx)))


def _seed_indices(count: int, max_points: int = 48) -> np.ndarray:
    if count <= max_points:
        return np.arange(count, dtype=int)
    return np.unique(np.rint(np.linspace(0, count - 1, max_points)).astype(int))


def _candidate_seeds(device_ms: np.ndarray, modulo_ms: np.ndarray, recording_start_ms: int) -> list[tuple[float, float]]:
    seeds = [(1.0, float(recording_start_ms))]
    selected = _seed_indices(device_ms.size)
    for left_position, left in enumerate(selected[:-1]):
        for right in selected[left_position + 1 :]:
            dx = device_ms[right] - device_ms[left]
            if dx <= 0:
                continue
            cycles = round(((modulo_ms[left] + dx) - modulo_ms[right]) / PACKED_PC_MOD_MS)
            slope = (modulo_ms[right] + cycles * PACKED_PC_MOD_MS - modulo_ms[left]) / dx
            if not 0.95 <= slope <= 1.05:
                continue
            intercept = _align_intercept(float(modulo_ms[left] - slope * device_ms[left]), recording_start_ms)
            seeds.append((float(slope), intercept))
    return seeds


def _refit(
    device_ms: np.ndarray,
    modulo_ms: np.ndarray,
    keep: np.ndarray,
    slope_seed: float,
    intercept_seed: float,
    recording_start_ms: int,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    lifted = _lift(modulo_ms, device_ms, slope_seed, intercept_seed)
    slope, intercept = _line_fit(device_ms[keep], lifted[keep])
    intercept = _align_intercept(intercept, recording_start_ms)
    lifted = _lift(modulo_ms, device_ms, slope, intercept)
    return slope, intercept, lifted, lifted - (slope * device_ms + intercept)


def fit_robust_pc_time_model(
    update_indices: np.ndarray,
    packed_values: np.ndarray,
    sample_rate_hz: float,
    recording_start_ms: int,
) -> PcTimeModel:
    """Fit ``pc_ms_unwrapped = slope * device_ms + intercept`` robustly."""

    indices = np.asarray(update_indices, dtype=np.int64)
    packed = np.asarray(packed_values, dtype=np.uint32)
    if indices.ndim != 1 or packed.ndim != 1 or indices.size != packed.size or not packed.size:
        raise ValueError("packed PC-time update indices and values must be non-empty equal-length vectors")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if indices.size > 1 and np.any(np.diff(indices) <= 0):
        raise ValueError("packed PC-time update indices must be strictly increasing")
    device_ms = indices.astype(float) * (1000.0 / float(sample_rate_hz))
    _raw, delay_ms, modulo_ms = unpack_packed_updates(packed)
    if packed.size == 1:
        slope = 1.0
        intercept = _align_intercept(float(modulo_ms[0] - device_ms[0]), recording_start_ms)
        lifted = _lift(modulo_ms, device_ms, slope, intercept)
        residuals = lifted - (slope * device_ms + intercept)
        return PcTimeModel(device_ms, lifted, delay_ms, residuals, np.array([True]), slope, intercept, float("nan"), float("nan"), recording_start_ms)

    best: tuple[int, float, np.ndarray, float, float] | None = None
    for slope_seed, intercept_seed in _candidate_seeds(device_ms, modulo_ms, recording_start_ms):
        residuals = _lift(modulo_ms, device_ms, slope_seed, intercept_seed) - (slope_seed * device_ms + intercept_seed)
        keep = np.abs(residuals) <= ROBUST_SEED_INLIER_MS
        count = int(np.count_nonzero(keep))
        if count < 2:
            continue
        score = float(np.mean(np.abs(residuals[keep])))
        candidate = (count, score, keep, slope_seed, intercept_seed)
        if best is None or count > best[0] or (count == best[0] and score < best[1]):
            best = candidate
    if best is None:
        raise ValueError("no robust linear-fit PC-time inliers found")
    _count, _score, keep, slope, intercept = best
    for _ in range(10):
        slope, intercept, lifted, residuals = _refit(device_ms, modulo_ms, keep, slope, intercept, recording_start_ms)
        kept = residuals[keep]
        center = float(np.median(kept))
        mad = float(np.median(np.abs(kept - center)))
        gate = max(ROBUST_RESIDUAL_FLOOR_MS, ROBUST_RESIDUAL_MAD_SCALE * mad)
        new_keep = np.abs(residuals - center) <= gate
        if np.count_nonzero(new_keep) < 2 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    slope, intercept, lifted, residuals = _refit(device_ms, modulo_ms, keep, slope, intercept, recording_start_ms)
    kept = residuals[keep]
    center = float(np.median(kept))
    mad = float(np.median(np.abs(kept - center)))
    gate = max(ROBUST_RESIDUAL_FLOOR_MS, ROBUST_RESIDUAL_MAD_SCALE * mad)
    expanded = np.abs(residuals - center) <= gate
    if np.count_nonzero(expanded) >= np.count_nonzero(keep):
        keep = expanded
        slope, intercept, lifted, residuals = _refit(device_ms, modulo_ms, keep, slope, intercept, recording_start_ms)
    if np.count_nonzero(keep) < 2:
        raise ValueError("no robust linear-fit PC-time inliers found")
    slope_sem, intercept_sem = _standard_errors(device_ms[keep], lifted[keep], slope, intercept)
    return PcTimeModel(device_ms, lifted, delay_ms, residuals, keep, slope, intercept, slope_sem, intercept_sem, recording_start_ms)
