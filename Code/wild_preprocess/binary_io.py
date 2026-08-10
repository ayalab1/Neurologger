from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Iterable

import numpy as np

from .models import Recording


INT16_LE = np.dtype("<i2")
ANALOG_SAMPLE_RATE_HZ = 1250.0


def read_ce_params_header(path: Path) -> tuple[int, int]:
    data = path.read_bytes()[:512]
    if len(data) < 441:
        raise ValueError(f"CE_params.bin is too short: {path}")
    fs = struct.unpack_from("<I", data, 0)[0]
    data_version = data[440]
    if data_version == 0:
        n_channels = struct.unpack_from("<I", data, 8)[0]
    else:
        n_channels = struct.unpack_from("<H", data, 8)[0]
    if fs <= 0 or n_channels <= 0:
        raise ValueError(f"Invalid CE_params.bin header: fs={fs}, channels={n_channels}, file={path}")
    return int(fs), int(n_channels)


def read_ce_params_metadata(path: Path) -> dict[str, int]:
    """Return the small CE header subset retained in run provenance."""

    data = path.read_bytes()[:512]
    fs, n_channels = read_ce_params_header(path)
    return {
        "ephys_sample_rate_hz": fs,
        "neural_channel_count": n_channels,
        "data_version": int(data[440]),
        "header_bytes_read": len(data),
    }


def _validated_sample_count(path: Path, n_channels: int) -> int:
    size = path.stat().st_size
    frame_bytes = INT16_LE.itemsize * n_channels
    if size == 0 or size % frame_bytes:
        raise ValueError(
            f"DAT size is not divisible by {frame_bytes} bytes per sample frame: {path} ({size} bytes)"
        )
    return size // frame_bytes


def recording_from_folder(folder: Path) -> Recording:
    folder = Path(folder).resolve()
    amplifier_file = folder / "amplifier.dat"
    analog_file = folder / "analogin.dat"
    ce_params_file = folder / "CE_params.bin"
    for path in (amplifier_file, analog_file, ce_params_file):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required WILD file: {path}")
    fs, n_channels = read_ce_params_header(ce_params_file)
    if n_channels % 4:
        raise ValueError(f"Neural channel count must be divisible by four: {n_channels} ({folder})")
    analog_channels = n_channels // 4
    n_samples = _validated_sample_count(amplifier_file, n_channels)
    analog_samples = _validated_sample_count(analog_file, analog_channels)
    return Recording(
        folder=folder,
        amplifier_file=amplifier_file,
        analog_file=analog_file,
        ce_params_file=ce_params_file,
        device_name=folder.parent.name,
        recording_name=folder.name,
        fs=fs,
        n_channels=n_channels,
        n_samples=n_samples,
        analog_channels=analog_channels,
        analog_samples=analog_samples,
    )


def recordings_from_folders(folders: Iterable[Path]) -> list[Recording]:
    recordings = [recording_from_folder(Path(folder)) for folder in folders]
    if len(recordings) < 2:
        raise ValueError("Multi-device preprocessing requires at least two recordings.")
    fs_values = {recording.fs for recording in recordings}
    if len(fs_values) != 1:
        raise ValueError(f"All recordings must use the same sample rate: {sorted(fs_values)}")
    for recording in recordings:
        ephys_duration = recording.n_samples / recording.fs
        analog_duration = recording.analog_samples / ANALOG_SAMPLE_RATE_HZ
        if abs(ephys_duration - analog_duration) > 2.0:
            raise ValueError(
                "analogin.dat duration is inconsistent with the 1250 Hz WILD analog contract: "
                f"{recording.folder} (ephys={ephys_duration:.3f}s, analog={analog_duration:.3f}s)"
            )
    return recordings


def interleaved_memmap(path: Path, n_channels: int, n_samples: int | None = None) -> np.memmap:
    if n_samples is None:
        n_samples = _validated_sample_count(path, n_channels)
    return np.memmap(path, dtype=INT16_LE, mode="r", shape=(n_samples, n_channels), order="C")


def close_memmap(mapped: np.memmap) -> None:
    memory_map = getattr(mapped, "_mmap", None)
    if memory_map is not None:
        memory_map.close()


def read_interleaved(
    path: Path,
    n_channels: int,
    start_sample: int,
    n_samples: int,
    *,
    total_samples: int | None = None,
) -> np.ndarray:
    if start_sample < 0 or n_samples < 0:
        raise ValueError("start_sample and n_samples must be non-negative")
    if total_samples is None:
        total_samples = _validated_sample_count(path, n_channels)
    end_sample = start_sample + n_samples
    if end_sample > total_samples:
        raise ValueError(
            f"Requested samples [{start_sample}, {end_sample}) exceed {total_samples}: {path}"
        )
    mapped = interleaved_memmap(path, n_channels, total_samples)
    try:
        return np.asarray(mapped[start_sample:end_sample]).copy()
    finally:
        close_memmap(mapped)


def atomic_output_path(path: Path) -> Path:
    return path.with_name(path.name + ".partial")


def replace_atomic(partial: Path, final: Path) -> None:
    os.replace(partial, final)
