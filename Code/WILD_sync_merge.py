#!/usr/bin/env python3
"""
Chunked sync checking and merge utility for multi-device WILD recordings.

The script scans a session root for recording folders containing CE_params.bin,
groups folders by the CE header start time decoded with the same layout used by
WILD_ReadHeader.m, estimates inter-device lag from accelerometer channels 1:3
of analogin.dat, pads shorter recordings with zeros, and writes a graphical
sync report plus merged channelized outputs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import math
import re
import shutil
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ACCEL_CHANNELS = (1, 2, 3)
DEFAULT_MERGE_STREAMS = ("amplifier", "analogin", "digitalin", "supply", "pc_time")
DAY_MS = 24 * 60 * 60 * 1000
REPORT_CHUNK_SCORE_THRESHOLD = 0.45
DEFAULT_PC_TIME_LAYOUT = {
    "name": "ce64-raw-misc",
    "raw_misc": True,
    "raw_words_per_cycle": 16,
    "raw_low_word_index": 14,
    "raw_high_word_index": 15,
    "expand_factor": 16,
}


@dataclass(frozen=True)
class HeaderInfo:
    data_version: int
    ephys_fs_hz: int
    misc_fs_hz: int
    misc_ratio: int
    n_channels: int
    recording_start: dt.datetime | None
    recording_start_ms: int | None
    mac: str


@dataclass
class RecordingInfo:
    recording_dir: Path
    device_id: str
    device_label: str
    header: HeaderInfo
    amplifier_path: Path
    analog_path: Path
    amplifier_samples: int
    analog_samples: int
    analog_channels: int
    digitalin_path: Path | None
    digitalin_samples: int
    supply_path: Path | None
    supply_samples: int
    adc_path: Path | None
    adc_samples: int
    pc_time_path: Path | None
    pc_time_samples: int
    start_key: str
    start_label: str
    offset_misc: int = 0
    offset_ephys: int = 0


@dataclass(frozen=True)
class AlignmentWindow:
    window_start_sec: float
    lag_misc_samples: int
    score: float
    activity: float


@dataclass(frozen=True)
class ChunkLag:
    center_sec: float
    lag_misc_samples: int
    score: float


@dataclass(frozen=True)
class StreamInput:
    recording: RecordingInfo
    path: Path
    samples: int
    channels: int
    offset_samples: int
    normalized_offset_samples: int


@dataclass(frozen=True)
class StreamPlan:
    key: str
    output_name: str
    dtype: str
    sample_rate_hz: int
    total_samples: int
    total_channels: int
    global_start_samples: int
    inputs: list[StreamInput]


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def parse_stream_list(text: str) -> tuple[str, ...]:
    names = [part.strip().lower() for part in text.split(",") if part.strip()]
    if not names:
        raise argparse.ArgumentTypeError("stream list cannot be empty")
    return tuple(names)


def round_down(value: int, factor: int) -> int:
    if factor <= 1:
        return value
    return value - (value % factor)


def format_ms_of_day(ms: int | None) -> str:
    if ms is None:
        return "n/a"
    ms %= DAY_MS
    hours, remainder = divmod(ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def format_duration_seconds(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "n/a"
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_offset_ms(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:+.1f}"


def wrap_day_delta_ms(a_ms: int, b_ms: int) -> int:
    delta = int(a_ms) - int(b_ms)
    half_day = DAY_MS // 2
    if delta > half_day:
        delta -= DAY_MS
    elif delta < -half_day:
        delta += DAY_MS
    return delta


def safe_sample_count(path: Path, frame_bytes: int) -> int:
    size = path.stat().st_size
    if size <= 0:
        return 0
    if frame_bytes <= 0 or size % frame_bytes != 0:
        raise ValueError(f"{path} length {size} is not a multiple of frame size {frame_bytes}")
    return size // frame_bytes


def infer_start_from_folder_name(path: Path) -> dt.datetime | None:
    match = re.search(
        r"(?P<date>\d{8})_(?P<hms>\d{6})(?:\.(?P<ms>\d{1,3}))?$",
        path.name,
    )
    if match is None:
        return None
    date_text = match.group("date")
    hms_text = match.group("hms")
    millis = int((match.group("ms") or "0").ljust(3, "0"))
    try:
        return dt.datetime(
            year=int(date_text[0:4]),
            month=int(date_text[4:6]),
            day=int(date_text[6:8]),
            hour=int(hms_text[0:2]),
            minute=int(hms_text[2:4]),
            second=int(hms_text[4:6]),
            microsecond=millis * 1000,
        )
    except ValueError:
        return None


def infer_misc_channel_count(path: Path, preferred: int) -> int:
    size = path.stat().st_size
    candidates = []
    for candidate in (preferred, 16, 8, 6):
        if candidate > 0 and size % (candidate * 2) == 0:
            candidates.append(candidate)
    if not candidates:
        raise ValueError(f"cannot infer analog channel count for {path}")
    return candidates[0]


def read_wild_header(path: Path) -> HeaderInfo:
    data = path.read_bytes()
    if len(data) < 364:
        raise ValueError(f"CE header is too short: {path}")

    data_version = int(data[440]) if len(data) > 440 else 0
    offset = 0

    def read(fmt: str, count: int = 1) -> tuple[int | float, ...]:
        nonlocal offset
        block = struct.unpack_from("<" + (fmt * count), data, offset)
        offset += struct.calcsize("<" + fmt) * count
        return block

    ephys_fs_hz = int(read("I")[0])
    _aux_mode = read("I")[0]
    sampling_rates = [0] * 8

    if data_version == 0:
        n_channels = int(read("I")[0])
        _conv_cmd = read("H", 64)
    else:
        n_channels_each = [int(value) for value in read("H", 8)]
        n_channels = n_channels_each[0] if n_channels_each else 0
        _speed_rates = read("H", 8)
        sampling_rates = [int(value) for value in read("I", 8)]
        _channel_list = read("B", 64)
        _unassigned = read("I")[0]

    _disp_ch = read("I")[0]
    _func1 = read("I")[0]
    _func2 = read("I")[0]
    _cmd_ch = read("I")[0]
    _rec_ch = read("I")[0]
    _stim_mode = read("I")[0]
    _cl_mode = read("I")[0]
    _stim_interval = read("I", 4)
    _pulse_width = read("I", 4)
    _pulse_count = read("I", 4)
    _stim_delay = read("I", 4)
    _stim_random_delay = read("I", 4)
    _trigger_train_start = read("I")[0]
    _trigger_train_duration = read("I")[0]
    _trigger_gain = read("f", 4)
    _sd_capacity = read("I")[0]
    _led_pulse_count = read("I")[0]
    _preview_channel_bank = read("I")[0]
    _system_status = read("I")[0]
    _stim_intensity = read("I", 4)
    _stim_ch = read("I", 4)

    misc_ratio = int(read("B")[0])
    _preview_ratio = read("B")[0]
    _misc_interval = read("H")[0]
    _error_code = read("I")[0]
    _firmware_version = read("H")[0]
    _hardware_version = read("H")[0]
    date_word = int(read("I")[0])
    time_words = [int(value) for value in read("I", 5)]
    mac_bytes = bytes(int(value) for value in read("B", 8))

    misc_fs_hz = 0
    positive_aux_rates = [rate for rate in sampling_rates[1:] if 0 < rate < ephys_fs_hz]
    if positive_aux_rates:
        misc_fs_hz = min(positive_aux_rates)
    elif misc_ratio > 0 and ephys_fs_hz > 0:
        misc_fs_hz = int(round(ephys_fs_hz / misc_ratio))

    recording_start: dt.datetime | None = None
    recording_start_ms: int | None = None
    date_bytes = struct.pack("<I", date_word)
    time_bytes = struct.pack("<I", time_words[0]) if time_words else b"\x00\x00\x00\x00"

    weekday = date_bytes[0]
    month = date_bytes[1]
    day = date_bytes[2]
    year = 2000 + date_bytes[3]
    hours = time_bytes[0]
    minutes = time_bytes[1]
    seconds = time_bytes[2]
    sub_seconds = time_words[1] if len(time_words) > 1 else 0
    second_fraction = time_words[2] if len(time_words) > 2 and time_words[2] > 0 else 9999

    if 0 <= sub_seconds <= second_fraction:
        millis = int(round(((second_fraction - sub_seconds) * 1000.0) / (second_fraction + 1)))
    else:
        millis = 0
    millis = max(0, min(999, millis))

    try:
        recording_start = dt.datetime(
            year=year,
            month=month,
            day=day,
            hour=hours,
            minute=minutes,
            second=seconds,
            microsecond=millis * 1000,
        )
        recording_start_ms = ((hours * 3600 + minutes * 60 + seconds) * 1000) + millis
    except ValueError:
        warn(
            f"failed to decode CE header date/time for {path} "
            f"(weekday={weekday}, date={month:02d}/{day:02d}/{year}, time={hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d})"
        )

    mac = "".join(f"{value:02X}" for value in mac_bytes[:6]).rstrip("0")
    if not mac:
        mac = "".join(f"{value:02X}" for value in mac_bytes).rstrip("0")

    return HeaderInfo(
        data_version=data_version,
        ephys_fs_hz=ephys_fs_hz,
        misc_fs_hz=misc_fs_hz,
        misc_ratio=misc_ratio,
        n_channels=n_channels,
        recording_start=recording_start,
        recording_start_ms=recording_start_ms,
        mac=mac,
    )


def parse_recording_labels(root: Path) -> dict[str, str]:
    info_path = root / "recording-info.txt"
    labels: dict[str, str] = {}
    if not info_path.exists():
        return labels
    for raw_line in info_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        token = line.split()[0].upper()
        labels[token] = line
    return labels


def discover_recordings(root: Path) -> list[RecordingInfo]:
    labels = parse_recording_labels(root)
    recordings: list[RecordingInfo] = []

    for ce_path in sorted(root.rglob("CE_params.bin")):
        recording_dir = ce_path.parent
        amplifier_path = recording_dir / "amplifier.dat"
        analog_path = recording_dir / "analogin.dat"
        if not amplifier_path.exists() or not analog_path.exists():
            continue

        header = read_wild_header(ce_path)
        analog_channels = infer_misc_channel_count(
            analog_path,
            preferred=16 if header.n_channels >= 64 else 8,
        )
        amplifier_samples = safe_sample_count(amplifier_path, header.n_channels * 2)
        analog_samples = safe_sample_count(analog_path, analog_channels * 2)

        device_id = recording_dir.parent.name
        suffix = device_id[-3:].upper() if len(device_id) >= 3 else device_id.upper()
        device_label = labels.get(device_id.upper(), labels.get(suffix, device_id))

        start_dt = header.recording_start or infer_start_from_folder_name(recording_dir)
        if start_dt is None:
            start_key = recording_dir.name
            start_label = recording_dir.name
        else:
            start_key = start_dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            start_label = start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        digitalin_path = recording_dir / "digitalin.dat"
        supply_path = recording_dir / "supply.dat"
        adc_path = recording_dir / "adc.dat"
        pc_time_path = recording_dir / "pc_time.dat"

        recordings.append(
            RecordingInfo(
                recording_dir=recording_dir,
                device_id=device_id,
                device_label=device_label,
                header=header,
                amplifier_path=amplifier_path,
                analog_path=analog_path,
                amplifier_samples=amplifier_samples,
                analog_samples=analog_samples,
                analog_channels=analog_channels,
                digitalin_path=digitalin_path if digitalin_path.exists() else None,
                digitalin_samples=safe_sample_count(digitalin_path, 4) if digitalin_path.exists() else 0,
                supply_path=supply_path if supply_path.exists() else None,
                supply_samples=safe_sample_count(supply_path, 4) if supply_path.exists() else 0,
                adc_path=adc_path if adc_path.exists() else None,
                adc_samples=safe_sample_count(adc_path, 2) if adc_path.exists() else 0,
                pc_time_path=pc_time_path if pc_time_path.exists() else None,
                pc_time_samples=safe_sample_count(pc_time_path, 4) if pc_time_path.exists() else 0,
                start_key=start_key,
                start_label=start_label,
            )
        )

    return recordings


def load_analog_memmap(recording: RecordingInfo) -> np.memmap:
    return np.memmap(
        recording.analog_path,
        dtype="<i2",
        mode="r",
        shape=(recording.analog_samples, recording.analog_channels),
    )


def accel_feature(
    analog_map: np.memmap,
    start_sample: int,
    stop_sample: int,
    downsample: int,
) -> np.ndarray:
    stop_sample = min(stop_sample, analog_map.shape[0])
    if stop_sample - start_sample < 4:
        return np.empty(0, dtype=np.float32)

    block = np.asarray(analog_map[start_sample:stop_sample, ACCEL_CHANNELS], dtype=np.float32)
    if block.shape[0] < 4:
        return np.empty(0, dtype=np.float32)

    diff = np.diff(block, axis=0)
    feature = np.sqrt(np.sum(diff * diff, axis=1, dtype=np.float32), dtype=np.float32)

    trim = (feature.size // downsample) * downsample
    if trim <= 0:
        return np.empty(0, dtype=np.float32)
    feature = feature[:trim].reshape(-1, downsample).mean(axis=1)

    feature -= np.median(feature)
    mad = np.median(np.abs(feature))
    scale = mad * 1.4826
    if scale > 1e-6:
        feature /= scale
    else:
        std = float(feature.std())
        if std > 1e-6:
            feature /= std
    feature -= feature.mean()
    return feature.astype(np.float32, copy=False)


def fft_xcorr(reference: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = int(reference.size + target.size - 1)
    nfft = 1 << (n - 1).bit_length()
    corr = np.fft.irfft(
        np.fft.rfft(reference, nfft) * np.conj(np.fft.rfft(target, nfft)),
        nfft,
    )
    corr = np.concatenate((corr[-(target.size - 1) :], corr[: reference.size]))
    lags = np.arange(-(target.size - 1), reference.size, dtype=np.int64)
    return corr.astype(np.float32, copy=False), lags


def weighted_median(values: list[int], weights: list[float]) -> int:
    if not values:
        return 0
    order = np.argsort(np.asarray(values, dtype=np.int64))
    values_sorted = np.asarray(values, dtype=np.int64)[order]
    weights_sorted = np.asarray(weights, dtype=np.float64)[order]
    cumulative = np.cumsum(weights_sorted)
    midpoint = cumulative[-1] * 0.5
    index = int(np.searchsorted(cumulative, midpoint, side="left"))
    return int(values_sorted[min(index, values_sorted.size - 1)])


def select_reference_windows(
    recording: RecordingInfo,
    first_span_sec: float,
    window_sec: float,
    step_sec: float,
    downsample: int,
) -> list[tuple[int, float, np.ndarray]]:
    analog_map = load_analog_memmap(recording)
    fs = recording.header.misc_fs_hz
    span_samples = min(recording.analog_samples, round_down(int(first_span_sec * fs), downsample))
    window_samples = max(downsample * 8, round_down(int(window_sec * fs), downsample))
    step_samples = max(downsample, round_down(int(step_sec * fs), downsample))

    if span_samples <= window_samples:
        starts = [0]
    else:
        starts = list(range(0, span_samples - window_samples + 1, step_samples))

    windows: list[tuple[int, float, np.ndarray]] = []
    for start_sample in starts:
        feature = accel_feature(analog_map, start_sample, start_sample + window_samples, downsample)
        if feature.size == 0:
            continue
        activity = float(np.mean(np.abs(feature)))
        windows.append((start_sample, activity, feature))

    return windows


def detect_lag_from_feature(
    reference_feature: np.ndarray,
    target_feature: np.ndarray,
    reference_start_sample: int,
    target_start_sample: int,
    downsample: int,
    min_lag_samples: int,
    max_lag_samples: int,
) -> tuple[int, float]:
    if reference_feature.size == 0 or target_feature.size == 0:
        return 0, 0.0

    corr, lags = fft_xcorr(reference_feature, target_feature)
    base_shift_ds = (target_start_sample - reference_start_sample) // downsample
    lag_samples = (lags + base_shift_ds) * downsample
    mask = (lag_samples >= min_lag_samples) & (lag_samples <= max_lag_samples)
    if not np.any(mask):
        return 0, 0.0

    sub_corr = corr[mask]
    sub_lags = lag_samples[mask]
    index = int(np.argmax(sub_corr))
    peak = float(sub_corr[index])
    denom = float(np.linalg.norm(reference_feature) * np.linalg.norm(target_feature))
    score = peak / denom if denom > 0 else 0.0
    return int(sub_lags[index]), score


def detect_start_offset_misc(
    reference: RecordingInfo,
    target: RecordingInfo,
    args: argparse.Namespace,
) -> tuple[int, list[AlignmentWindow]]:
    reference_windows = select_reference_windows(
        recording=reference,
        first_span_sec=args.start_search_span_seconds,
        window_sec=args.start_window_seconds,
        step_sec=args.start_window_step_seconds,
        downsample=args.downsample,
    )
    if not reference_windows:
        return 0, []

    target_map = load_analog_memmap(target)
    max_lag_samples = round_down(int(args.max_start_lag_seconds * reference.header.misc_fs_hz), args.downsample)

    windows: list[AlignmentWindow] = []
    lags: list[int] = []
    weights: list[float] = []

    for reference_start_sample, activity, reference_feature in reference_windows:
        target_start_sample = max(0, reference_start_sample - max_lag_samples)
        target_stop_sample = min(
            target.analog_samples,
            reference_start_sample
            + round_down(int(args.start_window_seconds * reference.header.misc_fs_hz), args.downsample)
            + max_lag_samples,
        )
        target_feature = accel_feature(
            target_map,
            target_start_sample,
            target_stop_sample,
            args.downsample,
        )
        lag_misc, score = detect_lag_from_feature(
            reference_feature=reference_feature,
            target_feature=target_feature,
            reference_start_sample=reference_start_sample,
            target_start_sample=target_start_sample,
            downsample=args.downsample,
            min_lag_samples=-max_lag_samples,
            max_lag_samples=max_lag_samples,
        )
        window = AlignmentWindow(
            window_start_sec=reference_start_sample / reference.header.misc_fs_hz,
            lag_misc_samples=lag_misc,
            score=score,
            activity=activity,
        )
        windows.append(window)
        if score >= args.min_alignment_score:
            lags.append(lag_misc)
            weights.append(max(score, 1e-6) * max(activity, 1e-6))

    strong_score = max(args.min_alignment_score * 2.0, 0.25)
    early_focus_seconds = min(
        args.start_search_span_seconds,
        max(args.start_window_seconds * 2.0, 180.0),
    )
    strong_early = [
        window
        for window in windows
        if window.window_start_sec <= early_focus_seconds and window.score >= strong_score
    ]
    if strong_early:
        best = max(strong_early, key=lambda window: (window.score, -window.window_start_sec))
        return best.lag_misc_samples, windows

    early_windows = [
        window
        for window in windows
        if window.window_start_sec <= early_focus_seconds and window.score >= args.min_alignment_score
    ]
    if early_windows:
        early_lags = [window.lag_misc_samples for window in early_windows]
        early_weights = [
            max(window.score, 1e-6) * max(window.activity, 1e-6) / (1.0 + (window.window_start_sec / max(1.0, args.start_window_seconds)))
            for window in early_windows
        ]
        return weighted_median(early_lags, early_weights), windows

    if not lags:
        best = max(windows, key=lambda window: window.score)
        return best.lag_misc_samples, windows

    return weighted_median(lags, weights), windows


def compute_chunk_lags(
    reference: RecordingInfo,
    target: RecordingInfo,
    base_lag_misc: int,
    args: argparse.Namespace,
) -> list[ChunkLag]:
    reference_map = load_analog_memmap(reference)
    target_map = load_analog_memmap(target)
    fs = reference.header.misc_fs_hz
    downsample = args.downsample
    chunk_samples = max(downsample * 8, round_down(int(args.chunk_window_seconds * fs), downsample))
    step_samples = max(downsample, round_down(int(args.chunk_step_seconds * fs), downsample))
    search_samples = max(downsample, round_down(int(args.chunk_search_seconds * fs), downsample))

    reference_start = max(0, -base_lag_misc)
    reference_stop = min(reference.analog_samples, target.analog_samples - base_lag_misc)
    if reference_stop - reference_start < chunk_samples:
        return []

    chunk_starts = list(range(reference_start, reference_stop - chunk_samples + 1, step_samples))
    if not chunk_starts:
        chunk_starts = [reference_start]

    results: list[ChunkLag] = []
    for reference_start_sample in chunk_starts:
        reference_feature = accel_feature(
            reference_map,
            reference_start_sample,
            reference_start_sample + chunk_samples,
            downsample,
        )
        target_start_sample = max(0, reference_start_sample + base_lag_misc - search_samples)
        target_stop_sample = min(
            target.analog_samples,
            reference_start_sample + base_lag_misc + chunk_samples + search_samples,
        )
        target_feature = accel_feature(
            target_map,
            target_start_sample,
            target_stop_sample,
            downsample,
        )
        lag_misc, score = detect_lag_from_feature(
            reference_feature=reference_feature,
            target_feature=target_feature,
            reference_start_sample=reference_start_sample,
            target_start_sample=target_start_sample,
            downsample=downsample,
            min_lag_samples=base_lag_misc - search_samples,
            max_lag_samples=base_lag_misc + search_samples,
        )
        results.append(
            ChunkLag(
                center_sec=(reference_start_sample + (chunk_samples * 0.5)) / fs,
                lag_misc_samples=lag_misc,
                score=score,
            )
        )

    return results


def choose_reference_recording(
    recordings: list[RecordingInfo],
    reference_hint: str | None,
) -> RecordingInfo:
    if reference_hint:
        reference_hint_upper = reference_hint.upper()
        for recording in recordings:
            if recording.device_id.upper() == reference_hint_upper:
                return recording
            if recording.device_id[-3:].upper() == reference_hint_upper:
                return recording
            if reference_hint_upper in recording.device_label.upper():
                return recording
    return max(recordings, key=lambda recording: recording.amplifier_samples)


def load_pc_time_helper(helper_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("wild_generate_pc_time_helper", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper module: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ensure_pc_time_file(
    recording: RecordingInfo,
    group_dir: Path,
    helper_module: Any | None,
) -> tuple[Path | None, bool]:
    if recording.pc_time_path is not None and recording.pc_time_samples == recording.amplifier_samples:
        return recording.pc_time_path, False
    if helper_module is None:
        return None, False

    derived_dir = group_dir / "derived_pc_time"
    derived_dir.mkdir(parents=True, exist_ok=True)
    output_path = derived_dir / f"{recording.device_id}_pc_time.dat"

    if output_path.exists():
        generated_samples = safe_sample_count(output_path, 4)
        if generated_samples == recording.amplifier_samples:
            return output_path, True

    layout = helper_module.Layout(**DEFAULT_PC_TIME_LAYOUT)
    recording_start_ms = recording.header.recording_start_ms
    recording_start_source = "CE_params.bin"
    if recording_start_ms is None:
        inferred = infer_start_from_folder_name(recording.recording_dir)
        if inferred is not None:
            recording_start_ms = (
                (inferred.hour * 3600 + inferred.minute * 60 + inferred.second) * 1000
                + (inferred.microsecond // 1000)
            )
            recording_start_source = "folder name"
        else:
            recording_start_ms = 0
            recording_start_source = "00:00 fallback"

    helper_module.generate_pc_time(
        analog_path=recording.analog_path,
        output_path=output_path,
        summary_plot_path=None,
        layout=layout,
        sample_rate_hz=float(recording.header.ephys_fs_hz),
        recording_start_ms=int(recording_start_ms),
        recording_start_source=recording_start_source,
    )
    return output_path, True


def overlap_indices(reference: RecordingInfo, target: RecordingInfo) -> tuple[int, int, int] | None:
    if target.offset_ephys >= 0:
        reference_start = target.offset_ephys
        target_start = 0
    else:
        reference_start = 0
        target_start = -target.offset_ephys

    overlap_length = min(
        reference.amplifier_samples - reference_start,
        target.amplifier_samples - target_start,
    )
    if overlap_length <= 0:
        return None
    return reference_start, target_start, overlap_length


def load_scalar_memmap(path: Path, dtype: str, count: int) -> np.memmap:
    return np.memmap(path, dtype=dtype, mode="r", shape=(count,))


def compute_ble_deltas_ms(
    reference: RecordingInfo,
    target: RecordingInfo,
    reference_pc_path: Path,
    target_pc_path: Path,
) -> tuple[int | None, int | None]:
    overlap = overlap_indices(reference, target)
    if overlap is None:
        return None, None
    reference_start, target_start, overlap_length = overlap
    reference_pc = load_scalar_memmap(reference_pc_path, "<u4", reference.amplifier_samples)
    target_pc = load_scalar_memmap(target_pc_path, "<u4", target.amplifier_samples)

    start_delta = wrap_day_delta_ms(
        int(target_pc[target_start]),
        int(reference_pc[reference_start]),
    )
    end_delta = wrap_day_delta_ms(
        int(target_pc[target_start + overlap_length - 1]),
        int(reference_pc[reference_start + overlap_length - 1]),
    )
    return start_delta, end_delta


def build_stream_plan(
    key: str,
    output_name: str,
    dtype: str,
    sample_rate_hz: int,
    inputs: list[tuple[RecordingInfo, Path, int, int, int]],
) -> StreamPlan | None:
    valid_inputs = [item for item in inputs if item[2] > 0]
    if not valid_inputs:
        return None
    global_start = min(item[4] for item in valid_inputs)
    stream_inputs: list[StreamInput] = []
    total_channels = 0
    total_samples = 0

    for recording, path, samples, channels, offset_samples in valid_inputs:
        normalized_offset = offset_samples - global_start
        stream_inputs.append(
            StreamInput(
                recording=recording,
                path=path,
                samples=samples,
                channels=channels,
                offset_samples=offset_samples,
                normalized_offset_samples=normalized_offset,
            )
        )
        total_channels += channels
        total_samples = max(total_samples, normalized_offset + samples)

    return StreamPlan(
        key=key,
        output_name=output_name,
        dtype=dtype,
        sample_rate_hz=sample_rate_hz,
        total_samples=total_samples,
        total_channels=total_channels,
        global_start_samples=global_start,
        inputs=stream_inputs,
    )


def build_stream_plans(
    recordings: list[RecordingInfo],
    reference: RecordingInfo,
    merge_streams: tuple[str, ...],
    pc_time_paths: dict[str, Path],
) -> list[StreamPlan]:
    plans: list[StreamPlan] = []
    merge_set = set(merge_streams)

    if "amplifier" in merge_set:
        plan = build_stream_plan(
            key="amplifier",
            output_name="amplifier_merged.dat",
            dtype="<i2",
            sample_rate_hz=reference.header.ephys_fs_hz,
            inputs=[
                (
                    recording,
                    recording.amplifier_path,
                    recording.amplifier_samples,
                    recording.header.n_channels,
                    recording.offset_ephys,
                )
                for recording in recordings
            ],
        )
        if plan is not None:
            plans.append(plan)

    if "analogin" in merge_set:
        plan = build_stream_plan(
            key="analogin",
            output_name="analogin_merged.dat",
            dtype="<i2",
            sample_rate_hz=reference.header.misc_fs_hz,
            inputs=[
                (
                    recording,
                    recording.analog_path,
                    recording.analog_samples,
                    recording.analog_channels,
                    recording.offset_misc,
                )
                for recording in recordings
            ],
        )
        if plan is not None:
            plans.append(plan)

    if "digitalin" in merge_set:
        plan = build_stream_plan(
            key="digitalin",
            output_name="digitalin_merged.dat",
            dtype="<u4",
            sample_rate_hz=reference.header.ephys_fs_hz,
            inputs=[
                (
                    recording,
                    recording.digitalin_path,
                    recording.digitalin_samples,
                    1,
                    recording.offset_ephys,
                )
                for recording in recordings
                if recording.digitalin_path is not None
            ],
        )
        if plan is not None:
            plans.append(plan)

    if "supply" in merge_set:
        plan = build_stream_plan(
            key="supply",
            output_name="supply_merged.dat",
            dtype="<u4",
            sample_rate_hz=reference.header.ephys_fs_hz,
            inputs=[
                (
                    recording,
                    recording.supply_path,
                    recording.supply_samples,
                    1,
                    recording.offset_ephys,
                )
                for recording in recordings
                if recording.supply_path is not None
            ],
        )
        if plan is not None:
            plans.append(plan)

    if "pc_time" in merge_set:
        plan = build_stream_plan(
            key="pc_time",
            output_name="pc_time_merged.dat",
            dtype="<u4",
            sample_rate_hz=reference.header.ephys_fs_hz,
            inputs=[
                (
                    recording,
                    pc_time_paths[recording.device_id],
                    recording.amplifier_samples,
                    1,
                    recording.offset_ephys,
                )
                for recording in recordings
                if recording.device_id in pc_time_paths
            ],
        )
        if plan is not None:
            plans.append(plan)

    return plans


def write_time_dat(path: Path, sample_count: int, chunk_samples: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        start = 0
        while start < sample_count:
            count = min(chunk_samples, sample_count - start)
            block = np.arange(start, start + count, dtype=np.int32)
            block.tofile(stream)
            start += count


def merge_stream(plan: StreamPlan, output_path: Path, chunk_samples: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    memmaps: list[np.memmap] = []
    dtype = np.dtype(plan.dtype)

    for stream_input in plan.inputs:
        shape = (stream_input.samples, stream_input.channels) if stream_input.channels > 1 else (stream_input.samples,)
        memmaps.append(np.memmap(stream_input.path, dtype=dtype, mode="r", shape=shape))

    with output_path.open("wb") as stream:
        start_sample = 0
        while start_sample < plan.total_samples:
            count = min(chunk_samples, plan.total_samples - start_sample)
            block = np.zeros((count, plan.total_channels), dtype=dtype)
            channel_offset = 0

            for stream_input, source_map in zip(plan.inputs, memmaps):
                source_start = max(0, start_sample - stream_input.normalized_offset_samples)
                source_stop = min(
                    stream_input.samples,
                    start_sample + count - stream_input.normalized_offset_samples,
                )
                if source_stop > source_start:
                    destination_start = stream_input.normalized_offset_samples + source_start - start_sample
                    destination_stop = destination_start + (source_stop - source_start)
                    data = np.asarray(source_map[source_start:source_stop], dtype=dtype)
                    if stream_input.channels == 1:
                        block[destination_start:destination_stop, channel_offset] = data.reshape(-1)
                    else:
                        block[destination_start:destination_stop, channel_offset : channel_offset + stream_input.channels] = data.reshape(-1, stream_input.channels)
                channel_offset += stream_input.channels

            block.tofile(stream)
            start_sample += count


def write_summary_csv(
    path: Path,
    recordings: list[RecordingInfo],
    reference: RecordingInfo,
    chunk_lags_by_device: dict[str, list[ChunkLag]],
    ble_deltas_by_device: dict[str, tuple[int | None, int | None]],
) -> None:
    rows: list[dict[str, Any]] = []
    for recording in recordings:
        chunk_lags = chunk_lags_by_device.get(recording.device_id, [])
        chunk_lags_for_summary = [lag for lag in chunk_lags if lag.score >= REPORT_CHUNK_SCORE_THRESHOLD]
        lag_ms = [
            lag.lag_misc_samples * 1000.0 / max(1, recording.header.misc_fs_hz)
            for lag in chunk_lags_for_summary
        ]
        rows.append(
            {
                "device_id": recording.device_id,
                "device_label": recording.device_label,
                "reference_device": reference.device_id,
                "recording_dir": str(recording.recording_dir),
                "header_start": recording.start_label,
                "ephys_fs_hz": recording.header.ephys_fs_hz,
                "misc_fs_hz": recording.header.misc_fs_hz,
                "amplifier_samples": recording.amplifier_samples,
                "analog_samples": recording.analog_samples,
                "offset_misc_samples": recording.offset_misc,
                "offset_ephys_samples": recording.offset_ephys,
                "offset_ms": (recording.offset_misc * 1000.0 / max(1, recording.header.misc_fs_hz)),
                "chunk_lag_median_ms": float(np.median(lag_ms)) if lag_ms else math.nan,
                "chunk_lag_min_ms": float(np.min(lag_ms)) if lag_ms else math.nan,
                "chunk_lag_max_ms": float(np.max(lag_ms)) if lag_ms else math.nan,
                "ble_start_delta_ms": ble_deltas_by_device.get(recording.device_id, (None, None))[0],
                "ble_end_delta_ms": ble_deltas_by_device.get(recording.device_id, (None, None))[1],
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_chunk_lag_csv(path: Path, chunk_lags_by_device: dict[str, list[ChunkLag]], misc_fs_hz: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("device_id", "center_sec", "lag_misc_samples", "lag_ms", "score"),
        )
        writer.writeheader()
        for device_id, chunk_lags in chunk_lags_by_device.items():
            for lag in chunk_lags:
                writer.writerow(
                    {
                        "device_id": device_id,
                        "center_sec": f"{lag.center_sec:.3f}",
                        "lag_misc_samples": lag.lag_misc_samples,
                        "lag_ms": f"{lag.lag_misc_samples * 1000.0 / max(1, misc_fs_hz):.3f}",
                        "score": f"{lag.score:.6f}",
                    }
                )


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                Path("C:/Windows/Fonts/consolab.ttf"),
                Path("C:/Windows/Fonts/arialbd.ttf"),
            ]
        )
    else:
        candidates.extend(
            [
                Path("C:/Windows/Fonts/consola.ttf"),
                Path("C:/Windows/Fonts/arial.ttf"),
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def draw_axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: str) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline="#D6DEE7", width=2)
    draw.line((left + 52, top + 18, left + 52, bottom - 34), fill=color, width=2)
    draw.line((left + 52, bottom - 34, right - 18, bottom - 34), fill=color, width=2)


def create_sync_report_image(
    root: Path,
    group_key: str,
    recordings: list[RecordingInfo],
    reference: RecordingInfo,
    chunk_lags_by_device: dict[str, list[ChunkLag]],
    ble_deltas_by_device: dict[str, tuple[int | None, int | None]],
) -> Image.Image:
    width = 1800
    height = 1280
    image = Image.new("RGB", (width, height), "#F4F7FB")
    draw = ImageDraw.Draw(image)

    title_font = load_font(34, bold=True)
    heading_font = load_font(22, bold=True)
    body_font = load_font(18)
    small_font = load_font(16)

    title = f"WILD Sync Report: {root.name}  [{group_key}]"
    subtitle = f"Reference device: {reference.device_label} ({reference.device_id})"
    draw.text((44, 28), title, fill="#102A43", font=title_font)
    draw.text((46, 74), subtitle, fill="#486581", font=body_font)

    table_box = (36, 118, width - 36, 412)
    lag_box = (36, 438, width - 36, 846)
    coverage_box = (36, 874, 1100, height - 34)
    ble_box = (1124, 874, width - 36, height - 34)

    for box in (table_box, lag_box, coverage_box, ble_box):
        draw.rounded_rectangle(box, radius=20, fill="#FFFFFF", outline="#D9E2EC", width=2)

    draw.text((table_box[0] + 24, table_box[1] + 18), "Device Summary", fill="#102A43", font=heading_font)
    header_line = (
        f"{'Device':<22} {'Amp Samples':>12} {'Duration':>10} {'Lag(ms)':>10} "
        f"{'Lag(ephys)':>11} {'BLE Start':>11} {'BLE End':>11}"
    )
    summary_lines = [header_line, "-" * len(header_line)]
    for recording in recordings:
        ble_start, ble_end = ble_deltas_by_device.get(recording.device_id, (None, None))
        duration_sec = recording.amplifier_samples / max(1, recording.header.ephys_fs_hz)
        lag_ms = recording.offset_misc * 1000.0 / max(1, recording.header.misc_fs_hz)
        line = (
            f"{recording.device_label[:22]:<22} "
            f"{recording.amplifier_samples:>12d} "
            f"{format_duration_seconds(duration_sec):>10} "
            f"{lag_ms:>+10.1f} "
            f"{recording.offset_ephys:>11d} "
            f"{format_offset_ms(ble_start):>11} "
            f"{format_offset_ms(ble_end):>11}"
        )
        summary_lines.append(line)
    draw.multiline_text(
        (table_box[0] + 26, table_box[1] + 58),
        "\n".join(summary_lines),
        fill="#243B53",
        font=small_font,
        spacing=10,
    )

    draw.text((lag_box[0] + 24, lag_box[1] + 18), "Chunk-Wise Accelerometer Lag", fill="#102A43", font=heading_font)
    draw.text(
        (lag_box[0] + 360, lag_box[1] + 22),
        f"High-confidence windows only (score >= {REPORT_CHUNK_SCORE_THRESHOLD:.2f})",
        fill="#52606D",
        font=small_font,
    )
    axis_box = (lag_box[0] + 16, lag_box[1] + 48, lag_box[2] - 16, lag_box[3] - 18)
    draw_axes(draw, axis_box, "#7B8794")

    chunk_lags_for_plot = {
        device_id: [lag for lag in lags if lag.score >= REPORT_CHUNK_SCORE_THRESHOLD]
        for device_id, lags in chunk_lags_by_device.items()
    }
    colors = ["#1F78B4", "#D94841", "#2F9E44", "#9467BD", "#F59E0B", "#0F766E"]
    all_points = [
        lag.lag_misc_samples * 1000.0 / max(1, reference.header.misc_fs_hz)
        for lags in chunk_lags_for_plot.values()
        for lag in lags
    ]
    y_min = min(all_points + [0.0]) if all_points else -5.0
    y_max = max(all_points + [0.0]) if all_points else 5.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    y_pad = max(1.0, (y_max - y_min) * 0.12)
    y_min -= y_pad
    y_max += y_pad

    max_time_sec = max(
        [recording.amplifier_samples / max(1, recording.header.ephys_fs_hz) for recording in recordings] + [1.0]
    )

    plot_left = axis_box[0] + 52
    plot_top = axis_box[1] + 18
    plot_right = axis_box[2] - 18
    plot_bottom = axis_box[3] - 34

    for tick in range(5):
        frac = tick / 4
        y_value = y_max - frac * (y_max - y_min)
        y = plot_top + frac * (plot_bottom - plot_top)
        draw.line((plot_left, y, plot_right, y), fill="#EEF2F7", width=1)
        draw.text((axis_box[0] + 4, y - 10), f"{y_value:+.1f}", fill="#52606D", font=small_font)

    for tick in range(6):
        frac = tick / 5
        x_value = frac * max_time_sec / 60.0
        x = plot_left + frac * (plot_right - plot_left)
        draw.line((x, plot_bottom, x, plot_top), fill="#F0F4F8", width=1)
        draw.text((x - 16, plot_bottom + 6), f"{x_value:.0f}", fill="#52606D", font=small_font)

    zero_y = plot_bottom - ((0.0 - y_min) / (y_max - y_min)) * (plot_bottom - plot_top)
    draw.line((plot_left, zero_y, plot_right, zero_y), fill="#9FB3C8", width=2)
    draw.text((plot_left + 8, plot_top + 4), "Lag relative to reference (ms)", fill="#52606D", font=small_font)
    draw.text((plot_right - 130, plot_bottom + 6), "Time (min)", fill="#52606D", font=small_font)

    legend_y = lag_box[1] + 18
    color_index = 0
    for recording in recordings:
        device_lags = chunk_lags_for_plot.get(recording.device_id, [])
        if not device_lags:
            continue
        color = colors[color_index % len(colors)]
        color_index += 1
        draw.rounded_rectangle((plot_right - 280, legend_y, plot_right - 258, legend_y + 14), radius=3, fill=color)
        draw.text((plot_right - 248, legend_y - 4), recording.device_label[:24], fill="#243B53", font=small_font)
        legend_y += 22

        points: list[tuple[float, float]] = []
        for lag in device_lags:
            x = plot_left + (lag.center_sec / max_time_sec) * (plot_right - plot_left)
            y_value = lag.lag_misc_samples * 1000.0 / max(1, reference.header.misc_fs_hz)
            y = plot_bottom - ((y_value - y_min) / (y_max - y_min)) * (plot_bottom - plot_top)
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline="#FFFFFF")

    draw.text((coverage_box[0] + 24, coverage_box[1] + 18), "Amplifier Coverage After Padding", fill="#102A43", font=heading_font)
    coverage_left = coverage_box[0] + 34
    coverage_top = coverage_box[1] + 64
    coverage_right = coverage_box[2] - 24
    coverage_bar_width = coverage_right - coverage_left - 250
    global_start = min(recording.offset_ephys for recording in recordings)
    global_end = max(recording.offset_ephys + recording.amplifier_samples for recording in recordings)
    merged_samples = global_end - global_start

    color_index = 0
    for row_index, recording in enumerate(recordings):
        row_y = coverage_top + row_index * 62
        color = colors[color_index % len(colors)]
        color_index += 1
        normalized_offset = recording.offset_ephys - global_start
        actual_width = coverage_bar_width * (recording.amplifier_samples / max(1, merged_samples))
        start_x = coverage_left + 210 + coverage_bar_width * (normalized_offset / max(1, merged_samples))
        draw.text((coverage_left, row_y - 2), recording.device_label[:18], fill="#243B53", font=body_font)
        draw.rectangle(
            (coverage_left + 210, row_y + 6, coverage_left + 210 + coverage_bar_width, row_y + 28),
            fill="#E9EEF5",
            outline="#CBD2D9",
        )
        draw.rectangle(
            (start_x, row_y + 6, start_x + actual_width, row_y + 28),
            fill=color,
            outline=color,
        )
        text = (
            f"{recording.amplifier_samples / 1_000_000:.2f} M samples, "
            f"pad={normalized_offset}/{merged_samples - normalized_offset - recording.amplifier_samples}"
        )
        draw.text((coverage_left + 210, row_y + 34), text, fill="#52606D", font=small_font)

    draw.text((ble_box[0] + 24, ble_box[1] + 18), "BLE Start / End Delta", fill="#102A43", font=heading_font)
    ble_values = [
        value
        for device_id, deltas in ble_deltas_by_device.items()
        if device_id != reference.device_id
        for value in deltas
        if value is not None
    ]
    ble_scale = max(10.0, max((abs(value) for value in ble_values), default=10.0) * 1.25)
    ble_center_x = (ble_box[0] + ble_box[2]) // 2
    ble_plot_top = ble_box[1] + 66
    ble_row_gap = 84
    draw.line((ble_center_x, ble_plot_top - 18, ble_center_x, ble_box[3] - 28), fill="#9FB3C8", width=2)
    draw.text((ble_center_x + 6, ble_plot_top - 28), "0 ms", fill="#52606D", font=small_font)

    color_index = 0
    non_reference = [recording for recording in recordings if recording.device_id != reference.device_id]
    for row_index, recording in enumerate(non_reference):
        color = colors[color_index % len(colors)]
        color_index += 1
        ble_start, ble_end = ble_deltas_by_device.get(recording.device_id, (None, None))
        row_y = ble_plot_top + row_index * ble_row_gap
        draw.text((ble_box[0] + 22, row_y - 18), recording.device_label[:24], fill="#243B53", font=body_font)
        for index, value in enumerate((ble_start, ble_end)):
            if value is None:
                continue
            bar_y = row_y + index * 26
            bar_width = (abs(value) / ble_scale) * ((ble_box[2] - ble_box[0]) * 0.38)
            if value >= 0:
                x0 = ble_center_x
                x1 = ble_center_x + bar_width
            else:
                x0 = ble_center_x - bar_width
                x1 = ble_center_x
            fill = color if index == 0 else "#F59E0B"
            draw.rectangle((x0, bar_y, x1, bar_y + 16), fill=fill, outline=fill)
            label = "start" if index == 0 else "end"
            text_x = ble_box[0] + 22
            draw.text((text_x, bar_y + 2), f"{label}: {value:+d} ms", fill="#52606D", font=small_font)

    return image


def write_report_assets(
    group_dir: Path,
    root: Path,
    group_key: str,
    recordings: list[RecordingInfo],
    reference: RecordingInfo,
    chunk_lags_by_device: dict[str, list[ChunkLag]],
    ble_deltas_by_device: dict[str, tuple[int | None, int | None]],
    stream_plans: list[StreamPlan],
) -> None:
    summary_csv_path = group_dir / "sync_summary.csv"
    chunk_csv_path = group_dir / "chunk_lags.csv"
    summary_json_path = group_dir / "sync_summary.json"
    report_png_path = group_dir / "sync_report.png"

    write_summary_csv(summary_csv_path, recordings, reference, chunk_lags_by_device, ble_deltas_by_device)
    write_chunk_lag_csv(chunk_csv_path, chunk_lags_by_device, reference.header.misc_fs_hz)

    summary_payload = {
        "root": str(root),
        "group_key": group_key,
        "reference_device": reference.device_id,
        "reference_label": reference.device_label,
        "recordings": [
            {
                "device_id": recording.device_id,
                "device_label": recording.device_label,
                "recording_dir": str(recording.recording_dir),
                "header_start": recording.start_label,
                "ephys_fs_hz": recording.header.ephys_fs_hz,
                "misc_fs_hz": recording.header.misc_fs_hz,
                "amplifier_samples": recording.amplifier_samples,
                "analog_samples": recording.analog_samples,
                "offset_misc_samples": recording.offset_misc,
                "offset_ephys_samples": recording.offset_ephys,
                "ble_start_delta_ms": ble_deltas_by_device.get(recording.device_id, (None, None))[0],
                "ble_end_delta_ms": ble_deltas_by_device.get(recording.device_id, (None, None))[1],
                "chunk_lags": [asdict(lag) for lag in chunk_lags_by_device.get(recording.device_id, [])],
            }
            for recording in recordings
        ],
        "merged_streams": [
            {
                "key": plan.key,
                "output_name": plan.output_name,
                "dtype": plan.dtype,
                "sample_rate_hz": plan.sample_rate_hz,
                "total_samples": plan.total_samples,
                "total_channels": plan.total_channels,
                "global_start_samples": plan.global_start_samples,
                "inputs": [
                    {
                        "device_id": stream_input.recording.device_id,
                        "device_label": stream_input.recording.device_label,
                        "path": str(stream_input.path),
                        "samples": stream_input.samples,
                        "channels": stream_input.channels,
                        "offset_samples": stream_input.offset_samples,
                        "normalized_offset_samples": stream_input.normalized_offset_samples,
                    }
                    for stream_input in plan.inputs
                ],
            }
            for plan in stream_plans
        ],
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    report = create_sync_report_image(
        root=root,
        group_key=group_key,
        recordings=recordings,
        reference=reference,
        chunk_lags_by_device=chunk_lags_by_device,
        ble_deltas_by_device=ble_deltas_by_device,
    )
    report.save(report_png_path)


def prepare_group_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge same-start WILD recordings using accelerometer-based alignment and BLE timing QC."
    )
    parser.add_argument("root", type=Path, help="session root containing device folders and recording folders")
    parser.add_argument(
        "-o",
        "--output-root",
        type=Path,
        help="output root for merged groups; default is <root>/merged_sync",
    )
    parser.add_argument(
        "--reference-device",
        help="reference device id, suffix, or label fragment; default is the longest amplifier.dat",
    )
    parser.add_argument(
        "--streams",
        type=parse_stream_list,
        default=DEFAULT_MERGE_STREAMS,
        help="comma-separated merged streams: amplifier,analogin,digitalin,supply,pc_time",
    )
    parser.add_argument("--report-only", action="store_true", help="generate QC outputs without writing merged .dat files")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing group output folders")
    parser.add_argument("--downsample", type=int, default=5, help="downsample factor for accelerometer lag detection")
    parser.add_argument("--start-search-span-seconds", type=float, default=900.0, help="search span for start alignment windows")
    parser.add_argument("--start-window-seconds", type=float, default=90.0, help="window length for start alignment")
    parser.add_argument("--start-window-step-seconds", type=float, default=30.0, help="step between start alignment windows")
    parser.add_argument("--start-window-count", type=int, default=6, help="number of high-activity start windows to evaluate")
    parser.add_argument("--max-start-lag-seconds", type=float, default=30.0, help="maximum absolute start lag to search")
    parser.add_argument("--min-alignment-score", type=float, default=0.10, help="minimum normalized score to contribute to start lag voting")
    parser.add_argument("--chunk-window-seconds", type=float, default=120.0, help="chunk size for lag drift checks")
    parser.add_argument("--chunk-step-seconds", type=float, default=600.0, help="step between lag drift chunks")
    parser.add_argument("--chunk-search-seconds", type=float, default=2.0, help="local search width around the detected lag for chunk QC")
    parser.add_argument("--snap-offset-ms", type=float, default=10.0, help="set detected start lags with absolute value below this threshold to zero")
    parser.add_argument("--merge-chunk-samples", type=int, default=125_000, help="write chunk size in samples for merged outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_root = (args.output_root.expanduser().resolve() if args.output_root else (root / "merged_sync"))

    recordings = discover_recordings(root)
    if not recordings:
        print(f"error: no recordings with CE_params.bin + amplifier.dat + analogin.dat found under {root}", file=sys.stderr)
        return 1

    groups: dict[str, list[RecordingInfo]] = {}
    for recording in recordings:
        groups.setdefault(recording.start_key, []).append(recording)

    helper_path = Path(__file__).with_name("WILD_generate_pc_time.py")
    helper_module: Any | None = None
    if helper_path.exists():
        try:
            helper_module = load_pc_time_helper(helper_path)
        except Exception as exc:
            warn(f"failed to load WILD_generate_pc_time.py: {exc}")

    for group_key, group_recordings in sorted(groups.items()):
        group_recordings = sorted(group_recordings, key=lambda recording: recording.device_id)
        reference = choose_reference_recording(group_recordings, args.reference_device)
        group_dir = output_root / group_key
        prepare_group_directory(group_dir, overwrite=args.overwrite)

        unique_ephys = {recording.header.ephys_fs_hz for recording in group_recordings}
        unique_misc = {recording.header.misc_fs_hz for recording in group_recordings}
        if len(unique_ephys) != 1 or len(unique_misc) != 1:
            print(
                f"error: group {group_key} has inconsistent sample rates: ephys={sorted(unique_ephys)}, misc={sorted(unique_misc)}",
                file=sys.stderr,
            )
            return 1

        chunk_lags_by_device: dict[str, list[ChunkLag]] = {}
        alignment_windows_by_device: dict[str, list[AlignmentWindow]] = {}

        reference.offset_misc = 0
        reference.offset_ephys = 0
        chunk_lags_by_device[reference.device_id] = [
            ChunkLag(center_sec=0.0, lag_misc_samples=0, score=1.0)
        ]
        alignment_windows_by_device[reference.device_id] = []

        for recording in group_recordings:
            if recording.device_id == reference.device_id:
                continue
            offset_misc, windows = detect_start_offset_misc(reference, recording, args)
            offset_ms = offset_misc * 1000.0 / max(1, reference.header.misc_fs_hz)
            if abs(offset_ms) <= args.snap_offset_ms:
                offset_misc = 0
            recording.offset_misc = offset_misc
            recording.offset_ephys = int(round(offset_misc * reference.header.ephys_fs_hz / max(1, reference.header.misc_fs_hz)))
            alignment_windows_by_device[recording.device_id] = windows
            chunk_lags_by_device[recording.device_id] = compute_chunk_lags(reference, recording, offset_misc, args)

        pc_time_paths: dict[str, Path] = {}
        for recording in group_recordings:
            path, _generated = ensure_pc_time_file(recording, group_dir, helper_module)
            if path is not None:
                pc_time_paths[recording.device_id] = path

        ble_deltas_by_device: dict[str, tuple[int | None, int | None]] = {
            reference.device_id: (0, 0)
        }
        reference_pc_path = pc_time_paths.get(reference.device_id)
        if reference_pc_path is not None:
            for recording in group_recordings:
                if recording.device_id == reference.device_id:
                    continue
                target_pc_path = pc_time_paths.get(recording.device_id)
                if target_pc_path is None:
                    ble_deltas_by_device[recording.device_id] = (None, None)
                    continue
                ble_deltas_by_device[recording.device_id] = compute_ble_deltas_ms(
                    reference=reference,
                    target=recording,
                    reference_pc_path=reference_pc_path,
                    target_pc_path=target_pc_path,
                )
        else:
            for recording in group_recordings:
                ble_deltas_by_device[recording.device_id] = (None, None)

        stream_plans = build_stream_plans(group_recordings, reference, args.streams, pc_time_paths)
        write_report_assets(
            group_dir=group_dir,
            root=root,
            group_key=group_key,
            recordings=group_recordings,
            reference=reference,
            chunk_lags_by_device=chunk_lags_by_device,
            ble_deltas_by_device=ble_deltas_by_device,
            stream_plans=stream_plans,
        )

        if not args.report_only:
            for plan in stream_plans:
                merge_stream(plan, group_dir / plan.output_name, chunk_samples=args.merge_chunk_samples)
            amplifier_plan = next((plan for plan in stream_plans if plan.key == "amplifier"), None)
            if amplifier_plan is not None:
                write_time_dat(group_dir / "time.dat", amplifier_plan.total_samples, args.merge_chunk_samples)

        amplifier_plan = next((plan for plan in stream_plans if plan.key == "amplifier"), None)
        merged_amp_samples = amplifier_plan.total_samples if amplifier_plan is not None else max(
            recording.amplifier_samples for recording in group_recordings
        )
        print(f"group={group_key}")
        print(f"output_dir={group_dir}")
        print(f"reference={reference.device_id}")
        print(f"recordings={len(group_recordings)}")
        print(f"merged_amplifier_samples={merged_amp_samples}")
        print(f"report={group_dir / 'sync_report.png'}")
        for recording in group_recordings:
            ble_start, ble_end = ble_deltas_by_device.get(recording.device_id, (None, None))
            offset_ms = recording.offset_misc * 1000.0 / max(1, recording.header.misc_fs_hz)
            print(
                f"device={recording.device_id} "
                f"offset_misc={recording.offset_misc} "
                f"offset_ephys={recording.offset_ephys} "
                f"offset_ms={offset_ms:+.3f} "
                f"ble_start_ms={ble_start} "
                f"ble_end_ms={ble_end}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
