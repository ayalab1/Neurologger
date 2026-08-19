from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.signal import butter, sosfilt

from ..binary_io import close_memmap, interleaved_memmap
from ..models import Recording


ProgressCallback = Callable[[str, float], None]


@dataclass(frozen=True)
class RawEvidenceScan:
    """Compact products of one sequential raw-amplifier evidence pass.

    ``frame_hash_path`` stores one deterministic little-endian ``uint64`` per
    raw frame.  It is only a candidate-screening key: callers must still
    compare every channel of candidate frame pairs before classifying a
    duplication.  This preserves the raw-equality authority used by the audit.
    """

    feature_path: Path
    coarse_feature_path: Path
    frame_hash_path: Path
    coarse_downsample_factor: int
    input_bytes_read: int
    output_bytes_written: int
    raw_amplifier_passes: int = 1


def _validate_feature_options(
    *,
    fs: float,
    highpass_hz: float,
    chunk_seconds: float,
) -> None:
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("sample rate must be finite and positive")
    if not math.isfinite(highpass_hz) or not 0 < highpass_hz < fs / 2:
        raise ValueError(f"Invalid high-pass frequency {highpass_hz} for fs={fs}")
    if not math.isfinite(chunk_seconds) or chunk_seconds <= 0:
        raise ValueError("chunk duration must be finite and positive")


def _coarse_factor(*, fs: float, target_rate_hz: float) -> int:
    if not math.isfinite(target_rate_hz) or target_rate_hz <= 0:
        raise ValueError("coarse feature rate must be finite and positive")
    return max(1, int(round(fs / target_rate_hz)))


def _frame_hashes(values: np.ndarray) -> np.ndarray:
    """Hash full raw frames using bounded vector storage.

    The arithmetic intentionally matches the audit's full-frame hash.  It is
    evaluated one channel at a time so this pass never creates a float or
    uint64 array shaped like a full chunk of raw channels.
    """

    if values.ndim != 2:
        raise ValueError("raw amplifier values must be a two-dimensional frame matrix")
    hashes = np.zeros(values.shape[0], dtype=np.uint64)
    channel_index = np.arange(1, values.shape[1] + 1, dtype=np.uint64)
    weights = channel_index * np.uint64(0x9E3779B185EBCA87)
    salts = channel_index * np.uint64(0xC2B2AE3D27D4EB4F)
    for channel, (weight, salt) in enumerate(zip(weights, salts, strict=True)):
        # A column view of an interleaved memmap is strided.  Make only this
        # one column contiguous before reinterpreting signed int16 bits.
        unsigned = np.ascontiguousarray(values[:, channel]).view("<u2").astype(
            np.uint64, copy=False
        )
        hashes ^= (unsigned + salt) * weight
    return hashes


def build_common_mode_feature(
    recording: Recording,
    output_path: Path,
    *,
    highpass_hz: float,
    chunk_seconds: float = 5.0,
    progress: ProgressCallback | None = None,
) -> Path:
    """Create a filtered float32 median common-mode stream.

    A stateful causal SOS filter is used so the large neural file is read once.
    All devices receive the same filter, so its deterministic phase response does
    not change their relative lag. The first 30 seconds are excluded by the
    default initial-alignment window, avoiding the filter startup transient.
    """

    _validate_feature_options(
        fs=recording.fs,
        highpass_hz=highpass_hz,
        chunk_seconds=chunk_seconds,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapped = interleaved_memmap(recording.amplifier_file, recording.n_channels, recording.n_samples)
    feature = np.memmap(output_path, dtype="<f4", mode="w+", shape=(recording.n_samples,))
    try:
        sos = butter(2, highpass_hz, btype="highpass", fs=recording.fs, output="sos")
        state = np.zeros((sos.shape[0], 2), dtype=np.float64)
        chunk_samples = max(1, round(chunk_seconds * recording.fs))
        for start in range(0, recording.n_samples, chunk_samples):
            end = min(recording.n_samples, start + chunk_samples)
            raw = np.asarray(mapped[start:end])
            common = np.median(raw, axis=1).astype(np.float64, copy=False)
            filtered, state = sosfilt(sos, common, zi=state)
            feature[start:end] = filtered.astype(np.float32)
            if progress is not None:
                progress(f"feature_{recording.device_name}", 100.0 * end / recording.n_samples)
        feature.flush()
    finally:
        close_memmap(feature)
        close_memmap(mapped)
    return output_path


def feature_memmap(path: Path, n_samples: int) -> np.memmap:
    expected = n_samples * np.dtype("<f4").itemsize
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"Common-mode cache size mismatch: {path} ({actual} != {expected})")
    return np.memmap(path, dtype="<f4", mode="r", shape=(n_samples,))


def frame_hash_memmap(path: Path, n_samples: int) -> np.memmap:
    """Open a compact per-frame hash stream after validating its exact size."""

    expected = n_samples * np.dtype("<u8").itemsize
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"Frame-hash cache size mismatch: {path} ({actual} != {expected})")
    return np.memmap(path, dtype="<u8", mode="r", shape=(n_samples,))


def coarse_feature_length(n_samples: int, downsample_factor: int) -> int:
    """Return the exact cache length for deterministic phase-zero decimation."""

    if n_samples < 0 or downsample_factor < 1:
        raise ValueError("sample count and downsample factor must be positive")
    return (n_samples + downsample_factor - 1) // downsample_factor


def build_coarse_feature(
    feature_path: Path,
    n_samples: int,
    output_path: Path,
    *,
    fs: float,
    target_rate_hz: float,
    chunk_seconds: float = 5.0,
) -> tuple[Path, int]:
    """Create an anti-aliased, phase-stable coarse feature cache.

    The source feature and output cache are both memory-mapped.  Filtering is
    causal and stateful across chunks, so peak locations have the same fixed
    phase response for master and slave while memory stays bounded by the
    chunk size.  The returned factor maps coarse lags back to source samples.
    """

    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("sample rate must be finite and positive")
    if not math.isfinite(chunk_seconds) or chunk_seconds <= 0:
        raise ValueError("chunk duration must be finite and positive")
    factor = _coarse_factor(fs=fs, target_rate_hz=target_rate_hz)
    source = feature_memmap(feature_path, n_samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.memmap(
        output_path,
        dtype="<f4",
        mode="w+",
        shape=(coarse_feature_length(n_samples, factor),),
    )
    try:
        if factor == 1:
            for start in range(0, n_samples, max(1, int(round(chunk_seconds * fs)))):
                end = min(n_samples, start + max(1, int(round(chunk_seconds * fs))))
                output[start:end] = source[start:end]
        else:
            # Keep a generous transition band below the decimated Nyquist
            # frequency.  This is an anti-alias filter, not a synchronization
            # confidence gate; acceptance still uses the established metrics.
            cutoff_hz = 0.4 * fs / factor
            sos = butter(4, cutoff_hz, btype="lowpass", fs=fs, output="sos")
            state = np.zeros((sos.shape[0], 2), dtype=np.float64)
            chunk_samples = max(1, int(round(chunk_seconds * fs)))
            for start in range(0, n_samples, chunk_samples):
                end = min(n_samples, start + chunk_samples)
                filtered, state = sosfilt(
                    sos,
                    np.asarray(source[start:end], dtype=np.float64),
                    zi=state,
                )
                first = (-start) % factor
                selected = filtered[first::factor]
                output[(start + first) // factor : (start + first) // factor + selected.size] = selected
        output.flush()
    finally:
        close_memmap(output)
        close_memmap(source)
    return output_path, factor


def build_raw_evidence_scan(
    recording: Recording,
    feature_path: Path,
    coarse_feature_path: Path,
    frame_hash_path: Path,
    *,
    highpass_hz: float,
    coarse_target_rate_hz: float,
    chunk_seconds: float = 5.0,
    progress: ProgressCallback | None = None,
) -> RawEvidenceScan:
    """Create full-rate, coarse, and duplication-screening evidence in one pass.

    This is the fused counterpart to :func:`build_common_mode_feature` followed
    by :func:`build_coarse_feature`.  The full-rate stream keeps the existing
    float32 cache representation.  The coarse filter consumes those exact
    float32 values (rather than the high-pass float64 intermediate), preserving
    the existing coarse-cache numerical semantics while avoiding a second read
    of ``amplifier.dat`` and a full feature-cache scan.

    The function owns only bounded chunk buffers plus the three output memmaps.
    It does not decide whether a repeated hash is a duplication; audit callers
    must use it only to narrow candidates then perform their established exact
    all-channel validation.
    """

    _validate_feature_options(
        fs=recording.fs,
        highpass_hz=highpass_hz,
        chunk_seconds=chunk_seconds,
    )
    factor = _coarse_factor(fs=recording.fs, target_rate_hz=coarse_target_rate_hz)
    output_paths = (Path(feature_path), Path(coarse_feature_path), Path(frame_hash_path))
    if len({path.resolve() for path in output_paths}) != len(output_paths):
        raise ValueError("evidence output paths must be distinct")
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    mapped = interleaved_memmap(recording.amplifier_file, recording.n_channels, recording.n_samples)
    feature = np.memmap(feature_path, dtype="<f4", mode="w+", shape=(recording.n_samples,))
    coarse = np.memmap(
        coarse_feature_path,
        dtype="<f4",
        mode="w+",
        shape=(coarse_feature_length(recording.n_samples, factor),),
    )
    hashes = np.memmap(frame_hash_path, dtype="<u8", mode="w+", shape=(recording.n_samples,))
    try:
        highpass = butter(2, highpass_hz, btype="highpass", fs=recording.fs, output="sos")
        highpass_state = np.zeros((highpass.shape[0], 2), dtype=np.float64)
        lowpass: np.ndarray | None = None
        lowpass_state: np.ndarray | None = None
        if factor != 1:
            lowpass = butter(
                4,
                0.4 * recording.fs / factor,
                btype="lowpass",
                fs=recording.fs,
                output="sos",
            )
            lowpass_state = np.zeros((lowpass.shape[0], 2), dtype=np.float64)
        chunk_samples = max(1, int(round(chunk_seconds * recording.fs)))
        for start in range(0, recording.n_samples, chunk_samples):
            end = min(recording.n_samples, start + chunk_samples)
            raw = np.asarray(mapped[start:end])
            common = np.median(raw, axis=1).astype(np.float64, copy=False)
            filtered, highpass_state = sosfilt(highpass, common, zi=highpass_state)
            full_rate = filtered.astype("<f4")
            feature[start:end] = full_rate
            hashes[start:end] = _frame_hashes(raw)
            if factor == 1:
                coarse[start:end] = full_rate
            else:
                assert lowpass is not None
                assert lowpass_state is not None
                coarse_filtered, lowpass_state = sosfilt(
                    lowpass,
                    np.asarray(full_rate, dtype=np.float64),
                    zi=lowpass_state,
                )
                first = (-start) % factor
                selected = coarse_filtered[first::factor]
                coarse_start = (start + first) // factor
                coarse[coarse_start : coarse_start + selected.size] = selected
            if progress is not None:
                progress(f"evidence_{recording.device_name}", 100.0 * end / recording.n_samples)
        feature.flush()
        coarse.flush()
        hashes.flush()
    finally:
        close_memmap(hashes)
        close_memmap(coarse)
        close_memmap(feature)
        close_memmap(mapped)
    output_bytes = sum(path.stat().st_size for path in output_paths)
    return RawEvidenceScan(
        feature_path=Path(feature_path),
        coarse_feature_path=Path(coarse_feature_path),
        frame_hash_path=Path(frame_hash_path),
        coarse_downsample_factor=factor,
        input_bytes_read=recording.amplifier_file.stat().st_size,
        output_bytes_written=output_bytes,
    )
