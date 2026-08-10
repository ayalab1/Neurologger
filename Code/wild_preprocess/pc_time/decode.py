"""Decode CE packed PC-clock updates without using legacy preprocessing code."""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DAY_MS = 24 * 60 * 60 * 1000
PACKED_PC_MOD_MS = 1 << 20
PACKED_PC_MS_MASK = PACKED_PC_MOD_MS - 1
PACKED_PC_DELAY_SHIFT = 20
PACKED_PC_DELAY_MASK = (1 << 12) - 1
RAW_MISC_BLOCK_BYTES = 512
EXPANDED_SAMPLES_PER_RAW_CYCLE = 16


@dataclass(frozen=True)
class PcTimeLayout:
    """Description of the two packed-PC layouts used by supported recordings."""

    name: str
    raw_misc: bool
    raw_words_per_cycle: int = 16
    raw_low_word_index: int = 14
    raw_high_word_index: int = 15
    expanded_channel_count: int = 6
    expanded_low_channel_index: int = 3
    expanded_high_channel_index: int = 4
    expand_factor: int = EXPANDED_SAMPLES_PER_RAW_CYCLE


CE64_RAW_MISC_LAYOUT = PcTimeLayout(name="ce64-raw-misc", raw_misc=True)
EXPANDED_ANALOG_LAYOUT = PcTimeLayout(name="expanded-analog", raw_misc=False)


@dataclass(frozen=True)
class CeParamsHint:
    ephys_sample_rate_hz: int | None = None
    misc_sample_rate_hz: int | None = None
    recording_start_ms: int | None = None


def infer_recording_start_from_name(path: Path) -> int | None:
    """Return milliseconds since midnight from a WILD recording folder name."""

    name = Path(path).name if Path(path).is_dir() else Path(path).parent.name
    match = re.search(r"(?:^|_)(?P<hms>\d{6})(?:\.(?P<ms>\d{1,3}))?$", name)
    if match is None:
        return None
    hms = match.group("hms")
    hours, minutes, seconds = int(hms[:2]), int(hms[2:4]), int(hms[4:])
    millis = int((match.group("ms") or "0").ljust(3, "0"))
    if hours > 23 or minutes > 59 or seconds > 59:
        return None
    return ((hours * 3600 + minutes * 60 + seconds) * 1000) + millis


def decode_ce_params_recording_start_ms(data: bytes) -> int | None:
    """Decode CE header time when the optional date/time header is populated."""

    if len(data) < 376:
        return None
    _weekday, month, day, year = struct.unpack_from("<BBBB", data, 336)
    hours, minutes, seconds, _time_format = struct.unpack_from("<BBBB", data, 356)
    sub_seconds, second_fraction, _daylight_saving, _store_operation = struct.unpack_from(
        "<IIII", data, 360
    )
    if not (0 <= year <= 99 and 1 <= month <= 12 and 1 <= day <= 31):
        return None
    if hours > 23 or minutes > 59 or seconds > 59:
        return None
    millis = 0
    if second_fraction > 0 and sub_seconds <= second_fraction:
        millis = ((second_fraction - sub_seconds) * 1000) // (second_fraction + 1)
    return ((hours * 3600 + minutes * 60 + seconds) * 1000) + millis


def read_ce_params_hint(recording_folder: Path) -> CeParamsHint:
    path = Path(recording_folder) / "CE_params.bin"
    if not path.is_file():
        return CeParamsHint()
    data = path.read_bytes()[:512]
    if len(data) < 56:
        return CeParamsHint()
    ephys_rate = struct.unpack_from("<I", data, 0)[0]
    sampling_rate_0 = struct.unpack_from("<I", data, 40)[0]
    misc_rate = struct.unpack_from("<I", data, 52)[0]
    return CeParamsHint(
        ephys_sample_rate_hz=(ephys_rate or sampling_rate_0) or None,
        misc_sample_rate_hz=misc_rate or None,
        recording_start_ms=decode_ce_params_recording_start_ms(data),
    )


def resolve_recording_start_ms(
    recording_folder: Path,
    *,
    explicit_recording_start_ms: int | None = None,
) -> tuple[int, str]:
    """Resolve an absolute daily anchor; never silently use midnight."""

    if explicit_recording_start_ms is not None:
        if not 0 <= int(explicit_recording_start_ms) < DAY_MS:
            raise ValueError("explicit recording start must be milliseconds in [0, 86400000)")
        return int(explicit_recording_start_ms), "explicit"
    hint = read_ce_params_hint(recording_folder)
    if hint.recording_start_ms is not None:
        return hint.recording_start_ms, "CE_params.bin"
    from_name = infer_recording_start_from_name(recording_folder)
    if from_name is not None:
        return from_name, "recording folder name"
    raise ValueError("No absolute recording-start anchor in CE_params.bin, folder name, or worker job.")


def _packed_updates_from_words(words: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    packed = words[:, 0].astype(np.uint32) | (words[:, 1].astype(np.uint32) << 16)
    valid = packed != 0
    packed, indices = packed[valid], indices[valid]
    if packed.size == 0:
        return indices.astype(np.int64), packed
    changed = np.r_[True, packed[1:] != packed[:-1]]
    return indices[changed].astype(np.int64), packed[changed].astype(np.uint32)


def collect_packed_updates(analog_path: Path, layout: PcTimeLayout = CE64_RAW_MISC_LAYOUT) -> tuple[np.ndarray, np.ndarray]:
    """Return unique packed updates and their ephys-sample coordinates.

    CE64 raw-misc cycles are expanded to ephys coordinates using ``expand_factor``.
    The expanded-analog layout is kept for legacy recordings with six int16 lanes.
    """

    analog_path = Path(analog_path)
    size = analog_path.stat().st_size
    if layout.raw_misc:
        cycle_bytes = layout.raw_words_per_cycle * 2
        if cycle_bytes <= 0 or RAW_MISC_BLOCK_BYTES % cycle_bytes:
            raise ValueError("raw words per cycle must divide 512-byte raw-misc blocks")
        if size == 0 or size % RAW_MISC_BLOCK_BYTES:
            raise ValueError("raw-misc analogin.dat length must be a non-zero multiple of 512 bytes")
        words = np.memmap(analog_path, dtype="<u2", mode="r")
        try:
            cycles = words.reshape(-1, layout.raw_words_per_cycle)
            lanes = cycles[:, [layout.raw_low_word_index, layout.raw_high_word_index]]
            indices = np.arange(cycles.shape[0], dtype=np.int64) * layout.expand_factor
            return _packed_updates_from_words(np.asarray(lanes), indices)
        finally:
            mmap = getattr(words, "_mmap", None)
            if mmap is not None:
                mmap.close()
    frame_bytes = layout.expanded_channel_count * 2
    if size == 0 or size % frame_bytes:
        raise ValueError("expanded analogin.dat length must be a non-zero whole number of frames")
    words = np.memmap(analog_path, dtype="<u2", mode="r").reshape(-1, layout.expanded_channel_count)
    try:
        lanes = words[:, [layout.expanded_low_channel_index, layout.expanded_high_channel_index]]
        indices = np.arange(words.shape[0], dtype=np.int64)
        return _packed_updates_from_words(np.asarray(lanes), indices)
    finally:
        mmap = getattr(words, "_mmap", None)
        if mmap is not None:
            mmap.close()


def unpack_packed_updates(packed_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    packed = np.asarray(packed_values, dtype=np.uint32)
    pc_ms20 = (packed & PACKED_PC_MS_MASK).astype(float)
    delay_ms = ((packed >> PACKED_PC_DELAY_SHIFT) & PACKED_PC_DELAY_MASK).astype(float)
    corrected_ms20 = np.mod(pc_ms20 + delay_ms, PACKED_PC_MOD_MS)
    return pc_ms20, delay_ms, corrected_ms20
