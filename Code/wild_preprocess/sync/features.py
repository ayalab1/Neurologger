from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from scipy.signal import butter, sosfilt

from ..binary_io import close_memmap, interleaved_memmap
from ..models import Recording


ProgressCallback = Callable[[str, float], None]


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

    if not 0 < highpass_hz < recording.fs / 2:
        raise ValueError(f"Invalid high-pass frequency {highpass_hz} for fs={recording.fs}")
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
