"""Exact merged-interval PC-time binary writers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..binary_io import atomic_output_path, close_memmap, replace_atomic
from .decode import DAY_MS
from .infer import PcTimeModel


def align_pc_time_file(
    raw_pc_time_path: Path,
    output_path: Path,
    *,
    common_start_master_sample: int,
    n_samples: int,
) -> Path:
    """Compatibility helper: slice a raw-master uint32 PC-time file exactly."""

    raw_pc_time_path, output_path = Path(raw_pc_time_path), Path(output_path)
    item_size = np.dtype("<u4").itemsize
    if raw_pc_time_path.stat().st_size % item_size:
        raise ValueError(f"pc_time.dat byte length is not uint32-aligned: {raw_pc_time_path}")
    raw_count = raw_pc_time_path.stat().st_size // item_size
    start, count = int(common_start_master_sample), int(n_samples)
    if start < 0 or count <= 0 or start + count > raw_count:
        raise ValueError(f"Merged PC-time slice [{start}, {start + count}) exceeds raw master length {raw_count}.")
    mapped = np.memmap(raw_pc_time_path, dtype="<u4", mode="r", shape=(raw_count,))
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial = atomic_output_path(output_path)
        if partial.exists():
            partial.unlink()
        with partial.open("wb") as stream:
            for begin in range(start, start + count, 1_000_000):
                np.asarray(mapped[begin : min(start + count, begin + 1_000_000)], dtype="<u4").tofile(stream)
        replace_atomic(partial, output_path)
    finally:
        close_memmap(mapped)
        partial = atomic_output_path(output_path)
        if partial.exists():
            partial.unlink()
    return output_path


def write_interval_pc_time(
    output_path: Path,
    model: PcTimeModel,
    *,
    sample_rate_hz: float,
    common_start_master_sample: int,
    n_samples: int,
    chunk_samples: int = 1_000_000,
) -> Path:
    """Write one little-endian uint32 daily PC timestamp per merged ephys sample."""

    if sample_rate_hz <= 0 or common_start_master_sample < 0 or n_samples <= 0:
        raise ValueError("invalid sample rate or merged interval")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = atomic_output_path(output_path)
    if partial.exists():
        partial.unlink()
    try:
        with partial.open("wb") as stream:
            for begin in range(0, n_samples, chunk_samples):
                count = min(chunk_samples, n_samples - begin)
                positions = common_start_master_sample + np.arange(begin, begin + count, dtype=float)
                device_ms = positions * (1000.0 / sample_rate_hz)
                values = np.rint(model.predict_unwrapped_ms(device_ms)).astype(np.int64) % DAY_MS
                values.astype("<u4", copy=False).tofile(stream)
        replace_atomic(partial, output_path)
    finally:
        if partial.exists():
            partial.unlink()
    return output_path
