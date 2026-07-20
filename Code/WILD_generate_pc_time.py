#!/usr/bin/env python3
"""
Generate pc_time.dat from a WILD/CE analogin.dat export.

The output matches CE32_console's PcTimeEstimator: little-endian uint32
milliseconds since midnight, one value per amplifier-sample time point.

Typical current CE64 use:
    python generate_pc_time.py C:\\Data\\WILD\\0_20260516_011122.208

If the recording folder name does not end in HHMMSS.mmm, pass the start anchor:
    python generate_pc_time.py analogin.dat --sample-rate 20000 --recording-start 01:11:22.208

To inspect the fit, add --summary-plot to also write pc_time_fit_summary.jpg:
    python generate_pc_time.py C:\\Data\\WILD\\0_20260516_011122.208 --summary-plot
"""

from __future__ import annotations

import argparse
import math
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ANALOG_CHANNEL_COUNT = 6
PACKED_LOW_CHANNEL_INDEX = 3
PACKED_HIGH_CHANNEL_INDEX = 4
BYTES_PER_ANALOG_SAMPLE = ANALOG_CHANNEL_COUNT * 2
RAW_MISC_BLOCK_BYTES = 512
EXPANDED_SAMPLES_PER_RAW_CYCLE = 16
# Current CE64 Core packs compressed host time as [delay_ms:12 | pc_ms modulo 2^20:20]
# in sys.PC_time.Milliseconds and mirrors that word into raw-misc lanes 14/15.
PACKED_PC_MS_MASK = (1 << 20) - 1
PACKED_PC_DELAY_SHIFT = 20
PACKED_PC_DELAY_MASK = (1 << 12) - 1
PACKED_PC_MOD_MS = 1 << 20
CE_PARAMS_DATE_OFFSET = 332
CE_PARAMS_TIME_OFFSET = 336
CE_PARAMS_DATE_SIZE = 4
CE_PARAMS_TIME_SIZE = 20
ROBUST_MODEL_MAX_SEED_POINTS = 48
ROBUST_MODEL_SEED_INLIER_MS = 250.0
ROBUST_MODEL_RESIDUAL_FLOOR_MS = 150.0
ROBUST_MODEL_RESIDUAL_MAD_SCALE = 6.0 * 1.4826
DAY_MS = 24 * 60 * 60 * 1000
READ_CHUNK_SAMPLES = 262_144
WRITE_CHUNK_SAMPLES = 262_144
RAW_READ_CHUNK_BLOCKS = 4096


@dataclass(frozen=True)
class Layout:
    name: str
    raw_misc: bool
    raw_words_per_cycle: int = 16
    raw_low_word_index: int = 14
    raw_high_word_index: int = 15
    expand_factor: int = EXPANDED_SAMPLES_PER_RAW_CYCLE


@dataclass(frozen=True)
class CeParamsHint:
    ephys_sample_rate_hz: int | None = None
    misc_sample_rate_hz: int | None = None
    recording_start_ms: int | None = None


@dataclass(frozen=True)
class GenerationSummary:
    output_path: Path
    summary_plot_path: Path | None
    sample_count: int
    update_count_all: int
    update_count_kept: int
    sample_rate_hz: float
    offset_ms: float
    offset_sem_ms: float
    drift_ppm: float
    drift_sem_ppm: float
    residual_rms_ms: float
    recording_start_ms: int
    recording_start_source: str


@dataclass(frozen=True)
class FitDiagnostics:
    dev_ms_abs: list[float]
    pc_corr_ms_abs: list[float]
    delay_ms: list[float]
    residual_ms: list[float]
    keep_mask: list[bool]
    slope: float
    intercept: float
    slope_sem: float
    intercept_sem: float


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def parse_positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def parse_nonnegative_int(text: str) -> int:
    value = int(text, 0)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def parse_ms_of_day(text: str) -> int:
    value = text.strip()
    if re.fullmatch(r"\d+", value):
        ms = int(value, 10)
        if not 0 <= ms < DAY_MS:
            raise argparse.ArgumentTypeError("milliseconds must be in [0, 86400000)")
        return ms

    match = re.fullmatch(
        r"(?P<h>\d{1,2})(?::?(?P<m>\d{2}))(?::?(?P<s>\d{2}))?(?:\.(?P<ms>\d{1,3}))?",
        value,
    )
    if match is None:
        raise argparse.ArgumentTypeError(
            "expected milliseconds since midnight, HH:MM:SS.mmm, or HHMMSS.mmm"
        )

    hours = int(match.group("h"))
    minutes = int(match.group("m"))
    seconds = int(match.group("s") or "0")
    millis_text = (match.group("ms") or "0").ljust(3, "0")
    millis = int(millis_text)

    if hours > 23 or minutes > 59 or seconds > 59:
        raise argparse.ArgumentTypeError("time is outside a single day")

    return ((hours * 3600 + minutes * 60 + seconds) * 1000) + millis


def infer_recording_start_from_name(path: Path) -> int | None:
    name = path.name if path.is_dir() else path.parent.name
    match = re.search(r"(?:^|_)(?P<hms>\d{6})(?:\.(?P<ms>\d{1,3}))?$", name)
    if match is None:
        return None

    hms = match.group("hms")
    millis_text = (match.group("ms") or "0").ljust(3, "0")
    hours = int(hms[0:2])
    minutes = int(hms[2:4])
    seconds = int(hms[4:6])
    millis = int(millis_text)
    if hours > 23 or minutes > 59 or seconds > 59:
        return None
    return ((hours * 3600 + minutes * 60 + seconds) * 1000) + millis


def format_ms_of_day(ms: int) -> str:
    ms %= DAY_MS
    hours, remainder = divmod(ms, 3600 * 1000)
    minutes, remainder = divmod(remainder, 60 * 1000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def resolve_input_paths(input_path: Path, output_path: Path | None) -> tuple[Path, Path, Path]:
    input_path = input_path.expanduser()
    if input_path.is_dir():
        record_dir = input_path
        analog_path = record_dir / "analogin.dat"
    else:
        analog_path = input_path
        record_dir = input_path.parent

    if output_path is None:
        output_path = record_dir / "pc_time.dat"
    else:
        output_path = output_path.expanduser()

    return record_dir, analog_path, output_path


def resolve_summary_plot_path(record_dir: Path, summary_plot_arg: str | None) -> Path | None:
    if summary_plot_arg is None:
        return None

    if summary_plot_arg == "__default__":
        return record_dir / "pc_time_fit_summary.jpg"

    path = Path(summary_plot_arg).expanduser()
    if path.suffix == "":
        path = path.with_suffix(".jpg")
    elif path.suffix.lower() not in (".jpg", ".jpeg"):
        raise ValueError("summary plot output must use a .jpg or .jpeg extension")
    return path


def decode_ce_params_recording_start_ms(data: bytes) -> int | None:
    min_length = max(
        CE_PARAMS_DATE_OFFSET + CE_PARAMS_DATE_SIZE,
        CE_PARAMS_TIME_OFFSET + CE_PARAMS_TIME_SIZE,
    )
    if len(data) < min_length:
        return None

    _, month, day, year = struct.unpack_from("<BBBB", data, CE_PARAMS_DATE_OFFSET)
    hours, minutes, seconds, _time_format = struct.unpack_from("<BBBB", data, CE_PARAMS_TIME_OFFSET)
    sub_seconds, second_fraction, _daylight_saving, _store_operation = struct.unpack_from(
        "<IIII",
        data,
        CE_PARAMS_TIME_OFFSET + 4,
    )

    if not (0 <= year <= 99 and 1 <= month <= 12 and 1 <= day <= 31):
        return None
    if hours > 23 or minutes > 59 or seconds > 59:
        return None

    if second_fraction > 0 and sub_seconds <= second_fraction:
        sub_elapsed = second_fraction - sub_seconds
        millis = (sub_elapsed * 1000) // (second_fraction + 1)
    else:
        millis = 0

    return ((hours * 3600 + minutes * 60 + seconds) * 1000) + millis


def read_ce_params_hint(record_dir: Path) -> CeParamsHint:
    path = record_dir / "CE_params.bin"
    if not path.exists():
        return CeParamsHint()

    data = path.read_bytes()[:512]
    if len(data) < 56:
        return CeParamsHint()

    def u32(offset: int) -> int:
        return struct.unpack_from("<I", data, offset)[0]

    # CE32_systemParam starts with ephys_fs, then sampling_rate[0..7] at offset 40.
    ephys_rate = u32(0)
    sampling_rate_0 = u32(40)
    sampling_rate_misc = u32(52)
    misc_ratio = data[320] if len(data) > 320 else 0

    if ephys_rate <= 0:
        ephys_rate = sampling_rate_0
    if ephys_rate <= 0:
        ephys_rate = None
    if sampling_rate_misc <= 0 and ephys_rate is not None and misc_ratio > 0:
        sampling_rate_misc = int(round(ephys_rate / misc_ratio))
    if sampling_rate_misc <= 0:
        sampling_rate_misc = None
    recording_start_ms = decode_ce_params_recording_start_ms(data)

    return CeParamsHint(
        ephys_sample_rate_hz=ephys_rate,
        misc_sample_rate_hz=sampling_rate_misc,
        recording_start_ms=recording_start_ms,
    )


def resolve_sample_rate(
    explicit_sample_rate: float | None,
    explicit_base_fs: float | None,
    layout: Layout,
    hints: CeParamsHint,
) -> float:
    if explicit_sample_rate is not None:
        return explicit_sample_rate

    if hints.ephys_sample_rate_hz is not None and hints.ephys_sample_rate_hz > 0:
        return float(hints.ephys_sample_rate_hz)

    if layout.raw_misc:
        if explicit_base_fs is not None:
            return explicit_base_fs * layout.expand_factor
        if hints.misc_sample_rate_hz is not None and hints.misc_sample_rate_hz > 0:
            return float(hints.misc_sample_rate_hz * layout.expand_factor)
        warn("no sample rate found; using current CE64 default 1250 Hz raw misc * 16 = 20000 Hz")
        return 1250.0 * layout.expand_factor

    warn("no sample rate found; using 20000 Hz")
    return 20000.0


def get_sample_count(file_length: int, layout: Layout) -> int:
    if file_length <= 0:
        raise ValueError("analogin.dat is empty")

    if layout.raw_misc:
        raw_cycle_bytes = layout.raw_words_per_cycle * 2
        if raw_cycle_bytes <= 0 or RAW_MISC_BLOCK_BYTES % raw_cycle_bytes != 0:
            raise ValueError("raw words per cycle must divide 512-byte raw misc blocks")
        if file_length % RAW_MISC_BLOCK_BYTES != 0:
            raise ValueError("raw-misc analogin.dat length is not a multiple of 512 bytes")
        raw_cycle_count = file_length // raw_cycle_bytes
        return raw_cycle_count * layout.expand_factor

    if file_length % BYTES_PER_ANALOG_SAMPLE != 0:
        raise ValueError("expanded analogin.dat length is not a multiple of 12 bytes")
    return file_length // BYTES_PER_ANALOG_SAMPLE


def collect_packed_updates_from_raw_misc(analog_path: Path, layout: Layout) -> tuple[list[int], list[int]]:
    raw_cycle_bytes = layout.raw_words_per_cycle * 2
    if RAW_MISC_BLOCK_BYTES % raw_cycle_bytes != 0:
        raise ValueError("raw words per cycle must divide 512-byte raw misc blocks")
    if layout.raw_low_word_index >= layout.raw_words_per_cycle:
        raise ValueError("low word index is outside the raw misc cycle")
    if layout.raw_high_word_index >= layout.raw_words_per_cycle:
        raise ValueError("high word index is outside the raw misc cycle")

    update_indices: list[int] = []
    update_packed_values: list[int] = []
    raw_cycle_base = 0
    last_packed: int | None = None
    chunk_bytes = RAW_READ_CHUNK_BLOCKS * RAW_MISC_BLOCK_BYTES

    with analog_path.open("rb") as stream:
        while True:
            data = stream.read(chunk_bytes)
            if not data:
                break
            if len(data) % RAW_MISC_BLOCK_BYTES != 0:
                raise ValueError("raw-misc read ended on a partial 512-byte block")

            words = memoryview(data).cast("H")
            cycles = len(words) // layout.raw_words_per_cycle
            for cycle in range(cycles):
                base = cycle * layout.raw_words_per_cycle
                low = int(words[base + layout.raw_low_word_index])
                high = int(words[base + layout.raw_high_word_index])
                packed = low | (high << 16)
                if packed == 0:
                    continue
                if last_packed is None or packed != last_packed:
                    update_indices.append((raw_cycle_base + cycle) * layout.expand_factor)
                    update_packed_values.append(packed)
                    last_packed = packed

            raw_cycle_base += cycles

    return update_indices, update_packed_values


def collect_packed_updates_from_expanded_analog(analog_path: Path) -> tuple[list[int], list[int]]:
    update_indices: list[int] = []
    update_packed_values: list[int] = []
    sample_base = 0
    last_packed: int | None = None
    chunk_bytes = READ_CHUNK_SAMPLES * BYTES_PER_ANALOG_SAMPLE

    with analog_path.open("rb") as stream:
        while True:
            data = stream.read(chunk_bytes)
            if not data:
                break
            valid_bytes = len(data) - (len(data) % BYTES_PER_ANALOG_SAMPLE)
            if valid_bytes != len(data):
                raise ValueError("expanded analog read ended on a partial sample")

            words = memoryview(data).cast("H")
            samples = len(words) // ANALOG_CHANNEL_COUNT
            for sample in range(samples):
                base = sample * ANALOG_CHANNEL_COUNT
                low = int(words[base + PACKED_LOW_CHANNEL_INDEX])
                high = int(words[base + PACKED_HIGH_CHANNEL_INDEX])
                packed = low | (high << 16)
                if packed == 0:
                    continue
                if last_packed is None or packed != last_packed:
                    update_indices.append(sample_base + sample)
                    update_packed_values.append(packed)
                    last_packed = packed

            sample_base += samples

    return update_indices, update_packed_values


def fit_line(x: list[float], y: list[float]) -> tuple[float, float]:
    if len(x) != len(y) or not x:
        return 1.0, 0.0
    if len(x) == 1:
        return 1.0, y[0] - x[0]

    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    sxx = 0.0
    sxy = 0.0
    for xi, yi in zip(x, y):
        dx = xi - mean_x
        dy = yi - mean_y
        sxx += dx * dx
        sxy += dx * dy

    if sxx <= 0.0:
        return 1.0, mean_y - mean_x
    slope = sxy / sxx
    intercept = mean_y - (slope * mean_x)
    return slope, intercept


def fit_line_standard_errors(x: list[float], y: list[float], slope: float, intercept: float) -> tuple[float, float]:
    count = len(x)
    if count < 3:
        return math.nan, math.nan

    mean_x = sum(x) / count
    sxx = sum((xi - mean_x) ** 2 for xi in x)
    if sxx <= 0.0:
        return math.nan, math.nan

    residual_sum_squares = sum((yi - ((slope * xi) + intercept)) ** 2 for xi, yi in zip(x, y))
    dof = count - 2
    if dof <= 0:
        return math.nan, math.nan

    mse = residual_sum_squares / dof
    slope_sem = math.sqrt(mse / sxx)
    intercept_sem = math.sqrt(mse * ((1.0 / count) + ((mean_x * mean_x) / sxx)))
    return slope_sem, intercept_sem


def unpack_packed_updates(update_packed_values: list[int]) -> tuple[list[float], list[float], list[float]]:
    pc_ms20 = [float(packed & PACKED_PC_MS_MASK) for packed in update_packed_values]
    delay_ms = [float((packed >> PACKED_PC_DELAY_SHIFT) & PACKED_PC_DELAY_MASK) for packed in update_packed_values]
    pc_corr_ms20 = [(pc_ms20[i] + delay_ms[i]) % PACKED_PC_MOD_MS for i in range(len(update_packed_values))]
    return pc_ms20, delay_ms, pc_corr_ms20


def build_candidate_observations(
    update_indices: list[int],
    update_packed_values: list[int],
    sample_rate_hz: float,
) -> tuple[list[float], list[float], list[float]]:
    _, delay_ms, pc_corr_ms20 = unpack_packed_updates(update_packed_values)
    dev_ms_abs = [(sample_index * 1000.0) / sample_rate_hz for sample_index in update_indices]
    return dev_ms_abs, pc_corr_ms20, delay_ms


def align_intercept_to_recording_start(intercept: float, recording_start_ms: int) -> float:
    return intercept + round((recording_start_ms - intercept) / PACKED_PC_MOD_MS) * PACKED_PC_MOD_MS


def lift_modulo_times_to_line(
    dev_ms_abs: list[float],
    pc_corr_ms20: list[float],
    slope: float,
    intercept: float,
) -> list[float]:
    aligned: list[float] = []
    for dev_ms, pc_corr20 in zip(dev_ms_abs, pc_corr_ms20):
        predicted_ms = (slope * dev_ms) + intercept
        cycle_offset = round((predicted_ms - pc_corr20) / PACKED_PC_MOD_MS) * PACKED_PC_MOD_MS
        aligned.append(pc_corr20 + cycle_offset)
    return aligned


def compute_line_residuals(
    dev_ms_abs: list[float],
    pc_corr_ms_abs: list[float],
    slope: float,
    intercept: float,
) -> list[float]:
    return [pc_ms - ((slope * dev_ms) + intercept) for dev_ms, pc_ms in zip(dev_ms_abs, pc_corr_ms_abs)]


def choose_seed_indices(count: int, max_seeds: int) -> list[int]:
    if count <= max_seeds:
        return list(range(count))

    return sorted({
        round((idx * (count - 1)) / (max_seeds - 1))
        for idx in range(max_seeds)
    })


def build_seed_models(
    dev_ms_abs: list[float],
    pc_corr_ms20: list[float],
    recording_start_ms: int,
) -> list[tuple[float, float]]:
    models: list[tuple[float, float]] = [(1.0, float(recording_start_ms))]
    seed_indices = choose_seed_indices(len(dev_ms_abs), ROBUST_MODEL_MAX_SEED_POINTS)

    for idx0, sample0 in enumerate(seed_indices[:-1]):
        x0 = dev_ms_abs[sample0]
        y0 = pc_corr_ms20[sample0]
        for sample1 in seed_indices[idx0 + 1 :]:
            x1 = dev_ms_abs[sample1]
            if x1 <= x0:
                continue

            y1 = pc_corr_ms20[sample1]
            cycle_delta = round(((y0 + (x1 - x0)) - y1) / PACKED_PC_MOD_MS)
            y1_abs = y1 + (cycle_delta * PACKED_PC_MOD_MS)
            slope = (y1_abs - y0) / (x1 - x0)
            if not math.isfinite(slope) or not (0.95 <= slope <= 1.05):
                continue

            intercept = align_intercept_to_recording_start(y0 - (slope * x0), recording_start_ms)
            models.append((slope, intercept))

    return models


def fit_line_from_inliers(
    dev_ms_abs: list[float],
    pc_corr_ms20: list[float],
    keep_mask: list[bool],
    slope_seed: float,
    intercept_seed: float,
    recording_start_ms: int,
) -> tuple[float, float, list[float], list[float]]:
    pc_corr_ms_abs = lift_modulo_times_to_line(dev_ms_abs, pc_corr_ms20, slope_seed, intercept_seed)
    kept_x = [x for x, keep in zip(dev_ms_abs, keep_mask) if keep]
    kept_y = [y for y, keep in zip(pc_corr_ms_abs, keep_mask) if keep]
    if len(kept_x) >= 2:
        slope, intercept = fit_line(kept_x, kept_y)
    elif kept_x:
        slope = slope_seed
        intercept = kept_y[0] - (slope * kept_x[0])
    else:
        slope = slope_seed
        intercept = intercept_seed

    intercept = align_intercept_to_recording_start(intercept, recording_start_ms)
    pc_corr_ms_abs = lift_modulo_times_to_line(dev_ms_abs, pc_corr_ms20, slope, intercept)
    residuals = compute_line_residuals(dev_ms_abs, pc_corr_ms_abs, slope, intercept)
    return slope, intercept, pc_corr_ms_abs, residuals


def fit_robust_linear_model(
    update_indices: list[int],
    update_packed_values: list[int],
    sample_rate_hz: float,
    recording_start_ms: int,
) -> FitDiagnostics:
    count = len(update_packed_values)
    if count == 0:
        raise ValueError("no nonzero packed PC-time updates found")

    dev_ms_abs, pc_corr_ms20, delay_ms = build_candidate_observations(
        update_indices,
        update_packed_values,
        sample_rate_hz,
    )
    if count == 1:
        intercept = align_intercept_to_recording_start(pc_corr_ms20[0] - dev_ms_abs[0], recording_start_ms)
        pc_corr_ms_abs = lift_modulo_times_to_line(dev_ms_abs, pc_corr_ms20, 1.0, intercept)
        residuals = compute_line_residuals(dev_ms_abs, pc_corr_ms_abs, 1.0, intercept)
        return FitDiagnostics(
            dev_ms_abs=dev_ms_abs,
            pc_corr_ms_abs=pc_corr_ms_abs,
            delay_ms=delay_ms,
            residual_ms=residuals,
            keep_mask=[True],
            slope=1.0,
            intercept=intercept,
            slope_sem=math.nan,
            intercept_sem=math.nan,
        )

    seed_threshold_ms = ROBUST_MODEL_SEED_INLIER_MS
    best_keep = [True] * count
    best_score = -1
    best_mean_abs_residual = float("inf")
    best_slope = 1.0
    best_intercept = float(recording_start_ms)

    for slope_seed, intercept_seed in build_seed_models(dev_ms_abs, pc_corr_ms20, recording_start_ms):
        pc_corr_ms_abs = lift_modulo_times_to_line(dev_ms_abs, pc_corr_ms20, slope_seed, intercept_seed)
        residuals = compute_line_residuals(dev_ms_abs, pc_corr_ms_abs, slope_seed, intercept_seed)
        keep = [abs(residual) <= seed_threshold_ms for residual in residuals]
        inlier_count = sum(1 for flag in keep if flag)
        if inlier_count < 2:
            continue

        mean_abs_residual = sum(abs(residual) for residual, flag in zip(residuals, keep) if flag) / inlier_count
        if (inlier_count > best_score) or (
            inlier_count == best_score and mean_abs_residual < best_mean_abs_residual
        ):
            best_keep = keep
            best_score = inlier_count
            best_mean_abs_residual = mean_abs_residual
            best_slope = slope_seed
            best_intercept = intercept_seed

    keep = best_keep
    slope = best_slope
    intercept = best_intercept
    residual_gate_ms = seed_threshold_ms

    for _ in range(10):
        slope, intercept, pc_corr_ms_abs, residuals = fit_line_from_inliers(
            dev_ms_abs,
            pc_corr_ms20,
            keep,
            slope,
            intercept,
            recording_start_ms,
        )
        kept_residuals = [residual for residual, flag in zip(residuals, keep) if flag]
        if len(kept_residuals) < 2:
            break

        center = percentile(kept_residuals, 0.5)
        mad = percentile([abs(residual - center) for residual in kept_residuals], 0.5)
        # Robust residual gate around the fitted drift*t + offset model after
        # aligning each point to the nearest 2^20-ms cycle of the candidate line.
        residual_gate_ms = max(ROBUST_MODEL_RESIDUAL_FLOOR_MS, ROBUST_MODEL_RESIDUAL_MAD_SCALE * mad)
        new_keep = [abs(residual - center) <= residual_gate_ms for residual in residuals]
        if sum(1 for flag in new_keep if flag) < 2:
            break
        if new_keep == keep:
            keep = new_keep
            break
        keep = new_keep

    slope, intercept, pc_corr_ms_abs, residuals = fit_line_from_inliers(
        dev_ms_abs,
        pc_corr_ms20,
        keep,
        slope,
        intercept,
        recording_start_ms,
    )
    kept_residuals = [residual for residual, flag in zip(residuals, keep) if flag]
    if kept_residuals:
        center = percentile(kept_residuals, 0.5)
        mad = percentile([abs(residual - center) for residual in kept_residuals], 0.5)
        residual_gate_ms = max(ROBUST_MODEL_RESIDUAL_FLOOR_MS, ROBUST_MODEL_RESIDUAL_MAD_SCALE * mad)
        expanded_keep = [abs(residual - center) <= residual_gate_ms for residual in residuals]
        if sum(1 for flag in expanded_keep if flag) >= sum(1 for flag in keep if flag):
            keep = expanded_keep
            slope, intercept, pc_corr_ms_abs, residuals = fit_line_from_inliers(
                dev_ms_abs,
                pc_corr_ms20,
                keep,
                slope,
                intercept,
                recording_start_ms,
            )

    if sum(1 for flag in keep if flag) < 2:
        raise ValueError("no robust linear-fit PC-time inliers found")

    kept_x = [x for x, flag in zip(dev_ms_abs, keep) if flag]
    kept_y = [y for y, flag in zip(pc_corr_ms_abs, keep) if flag]
    slope_sem, intercept_sem = fit_line_standard_errors(kept_x, kept_y, slope, intercept)

    return FitDiagnostics(
        dev_ms_abs=dev_ms_abs,
        pc_corr_ms_abs=pc_corr_ms_abs,
        delay_ms=delay_ms,
        residual_ms=residuals,
        keep_mask=keep,
        slope=slope,
        intercept=intercept,
        slope_sem=slope_sem,
        intercept_sem=intercept_sem,
    )


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return (ordered[lower] * (1.0 - fraction)) + (ordered[upper] * fraction)


def padded_range(min_value: float, max_value: float, pad_fraction: float, min_span: float) -> tuple[float, float]:
    if not math.isfinite(min_value) or not math.isfinite(max_value):
        return -min_span / 2.0, min_span / 2.0

    if min_value == max_value:
        half_span = max(min_span / 2.0, abs(min_value) * pad_fraction, 1.0)
        return min_value - half_span, max_value + half_span

    span = max(max_value - min_value, min_span)
    pad = span * pad_fraction
    return min_value - pad, max_value + pad


def symmetric_robust_range(values: list[float], quantile: float, min_half_span: float) -> tuple[float, float, int]:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return -min_half_span, min_half_span, 0

    half_span = max(min_half_span, percentile([abs(value) for value in finite_values], quantile) * 1.2)
    clipped = sum(1 for value in finite_values if abs(value) > half_span)
    return -half_span, half_span, clipped


def upper_robust_range(values: list[float], quantile: float, min_upper: float) -> tuple[float, float, int]:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return 0.0, min_upper, 0

    upper = max(min_upper, percentile(finite_values, quantile) * 1.1)
    clipped = sum(1 for value in finite_values if value > upper)
    return 0.0, upper, clipped


def choose_time_scale(max_ms: float) -> tuple[float, str]:
    if max_ms >= 2.0 * 60.0 * 60.0 * 1000.0:
        return 60.0 * 60.0 * 1000.0, "h"
    if max_ms >= 2.0 * 60.0 * 1000.0:
        return 60.0 * 1000.0, "min"
    return 1000.0, "s"


def format_tick(value: float, span: float) -> str:
    magnitude = max(abs(value), abs(span))
    if magnitude >= 1000.0:
        return f"{value:.0f}"
    if magnitude >= 100.0:
        return f"{value:.1f}"
    if magnitude >= 10.0:
        return f"{value:.2f}"
    if magnitude >= 1.0:
        return f"{value:.3f}"
    return f"{value:.4f}"


def _legacy_format_value_with_sem(value: float, sem: float, value_format: str) -> str:
    value_text = format(value, value_format)
    if math.isfinite(sem):
        return f"{value_text} ± {format(sem, value_format)}"
    return value_text


def format_value_with_sem(value: float, sem: float, value_format: str) -> str:
    value_text = format(value, value_format)
    if math.isfinite(sem):
        return f"{value_text} +/- {format(sem, value_format)}"
    return value_text


def write_fit_summary_jpg(
    output_path: Path,
    diagnostics: FitDiagnostics,
    summary: GenerationSummary,
) -> None:
    def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        candidates = [
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
            "arialbd.ttf" if bold else "arial.ttf",
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        ]
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> tuple[int, int]:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def draw_text(
        draw: ImageDraw.ImageDraw,
        x: float,
        y: float,
        text: str,
        *,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        fill: str,
        align: str = "left",
    ) -> None:
        width, _ = text_size(draw, text, font)
        if align == "center":
            x -= width / 2.0
        elif align == "right":
            x -= width
        draw.text((int(round(x)), int(round(y))), text, font=font, fill=fill)

    def map_value(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
        if src_max <= src_min:
            return (dst_min + dst_max) / 2.0
        return dst_min + ((value - src_min) / (src_max - src_min)) * (dst_max - dst_min)

    def draw_dashed_horizontal(
        draw: ImageDraw.ImageDraw,
        x0: float,
        x1: float,
        y: float,
        *,
        fill: str,
        width: int,
        dash_px: int = 6,
        gap_px: int = 5,
    ) -> None:
        start = int(round(x0))
        end = int(round(x1))
        ypos = int(round(y))
        while start < end:
            dash_end = min(start + dash_px, end)
            draw.line((start, ypos, dash_end, ypos), fill=fill, width=width)
            start = dash_end + gap_px

    def append_scatter_panel(
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        title: str,
        x_label: str,
        x_values: list[float],
        y_values: list[float],
        keep_mask: list[bool],
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        fit_values: tuple[float, float] | None = None,
        hlines: list[tuple[float, str, str]] | None = None,
        show_legend: bool = False,
    ) -> None:
        panel_border = "#c8c8c8"
        grid_color = "#e8e8e8"
        axis_color = "#666666"
        kept_color = "#1f77b4"
        discarded_color = "#d62728"
        fit_color = "#202020"

        draw.rectangle((x, y, x + width, y + height), fill="#ffffff", outline=panel_border, width=1)
        draw_text(draw, x + 18, y + 14, title, font=font_panel_title, fill="#111111")

        left = x + 72
        right = x + width - 22
        top = y + 48
        bottom = y + height - 56

        draw.rectangle((left, top, right, bottom), fill="#ffffff", outline=axis_color, width=1)

        x_ticks = 6
        y_ticks = 5
        x_span = x_range[1] - x_range[0]
        y_span = y_range[1] - y_range[0]
        for tick_index in range(x_ticks + 1):
            tick_value = x_range[0] + (x_span * tick_index / x_ticks)
            tick_x = map_value(tick_value, x_range[0], x_range[1], left, right)
            draw.line((tick_x, top, tick_x, bottom), fill=grid_color, width=1)
            draw_text(
                draw,
                tick_x,
                bottom + 10,
                format_tick(tick_value, x_span),
                font=font_tick,
                fill=axis_color,
                align="center",
            )

        for tick_index in range(y_ticks + 1):
            tick_value = y_range[0] + (y_span * tick_index / y_ticks)
            tick_y = map_value(tick_value, y_range[0], y_range[1], bottom, top)
            draw.line((left, tick_y, right, tick_y), fill=grid_color, width=1)
            draw_text(
                draw,
                left - 10,
                tick_y - 8,
                format_tick(tick_value, y_span),
                font=font_tick,
                fill=axis_color,
                align="right",
            )

        if hlines is not None:
            for hline_value, label, color in hlines:
                if y_range[0] <= hline_value <= y_range[1]:
                    hline_y = map_value(hline_value, y_range[0], y_range[1], bottom, top)
                    draw_dashed_horizontal(draw, left, right, hline_y, fill=color, width=1)
                    draw_text(draw, right - 4, hline_y - 18, label, font=font_small, fill=color, align="right")

        if fit_values is not None:
            draw.line(
                (
                    left,
                    map_value(fit_values[0], y_range[0], y_range[1], bottom, top),
                    right,
                    map_value(fit_values[1], y_range[0], y_range[1], bottom, top),
                ),
                fill=fit_color,
                width=3,
            )

        for x_value, y_value, keep in zip(x_values, y_values, keep_mask):
            point_x = map_value(x_value, x_range[0], x_range[1], left, right)
            point_y = map_value(y_value, y_range[0], y_range[1], bottom, top)
            point_x = min(max(point_x, left), right)
            point_y = min(max(point_y, top), bottom)
            color = kept_color if keep else discarded_color
            radius = 3
            draw.ellipse(
                (
                    point_x - radius,
                    point_y - radius,
                    point_x + radius,
                    point_y + radius,
                ),
                fill=color,
                outline=color,
            )

        draw_text(draw, (left + right) / 2.0, y + height - 34, x_label, font=font_axis, fill=axis_color, align="center")

        if show_legend:
            legend_x = right - 205
            legend_y = y + 16
            draw.ellipse((legend_x - 5, legend_y - 5, legend_x + 5, legend_y + 5), fill=kept_color, outline=kept_color)
            draw_text(draw, legend_x + 12, legend_y - 9, "kept", font=font_axis, fill="#111111")
            draw.ellipse((legend_x + 67, legend_y - 5, legend_x + 77, legend_y + 5), fill=discarded_color, outline=discarded_color)
            draw_text(draw, legend_x + 84, legend_y - 9, "discarded", font=font_axis, fill="#111111")
            draw.line((legend_x + 165, legend_y, legend_x + 188, legend_y), fill=fit_color, width=3)
            draw_text(draw, legend_x + 194, legend_y - 9, "fit", font=font_axis, fill="#111111")

    width = 1600
    height = 1040
    image = Image.new("RGB", (width, height), "#fbfbfb")
    draw = ImageDraw.Draw(image)

    font_title = load_font(26, bold=True)
    font_subtitle = load_font(16)
    font_panel_title = load_font(18, bold=True)
    font_axis = load_font(14)
    font_tick = load_font(13)
    font_small = load_font(12)

    x_scale_ms, x_unit = choose_time_scale(max(diagnostics.dev_ms_abs, default=0.0))
    x_values = [value / x_scale_ms for value in diagnostics.dev_ms_abs]
    fit_y_values = [
        (value - summary.recording_start_ms) / x_scale_ms
        for value in diagnostics.pc_corr_ms_abs
    ]
    residual_values = diagnostics.residual_ms
    delay_values = diagnostics.delay_ms

    x_range = padded_range(min(x_values, default=0.0), max(x_values, default=1.0), 0.04, 1.0)
    fit_line_x_ms = (x_range[0] * x_scale_ms, x_range[1] * x_scale_ms)
    fit_line_y = (
        ((diagnostics.slope * fit_line_x_ms[0]) + diagnostics.intercept - summary.recording_start_ms) / x_scale_ms,
        ((diagnostics.slope * fit_line_x_ms[1]) + diagnostics.intercept - summary.recording_start_ms) / x_scale_ms,
    )
    fit_y_range = padded_range(
        min(fit_y_values + [fit_line_y[0], fit_line_y[1]], default=0.0),
        max(fit_y_values + [fit_line_y[0], fit_line_y[1]], default=1.0),
        0.05,
        1.0,
    )
    residual_range = symmetric_robust_range(residual_values, 0.95, 5.0)
    delay_range = upper_robust_range(delay_values, 0.98, 5.0)

    residual_title = "Fit residuals to drift*t + offset (ms)"
    if residual_range[2] > 0:
        residual_title += f" - clipped {residual_range[2]}"

    delay_title = "Packed PC delay (ms)"
    if delay_range[2] > 0:
        delay_title += f" - clipped {delay_range[2]}"

    draw_text(draw, 28, 18, "PC-time fit summary", font=font_title, fill="#111111")
    draw_text(
        draw,
        28,
        54,
        (
            f"kept {summary.update_count_kept}/{summary.update_count_all} updates   "
            f"drift {format_value_with_sem(summary.drift_ppm, summary.drift_sem_ppm, '.3f')} ppm   "
            f"offset {format_value_with_sem(summary.offset_ms, summary.offset_sem_ms, '.1f')} ms rel. to start   "
            f"residual RMS {summary.residual_rms_ms:.3f} ms   "
            f"sample rate {summary.sample_rate_hz:g} Hz"
        ),
        font=font_subtitle,
        fill="#444444",
    )
    draw_text(
        draw,
        28,
        78,
        (
            f"Recording start {format_ms_of_day(summary.recording_start_ms)} from {summary.recording_start_source}. "
            "Discarded updates are shown in red after nearest-cycle alignment to the fitted drift*t + offset model."
        ),
        font=font_subtitle,
        fill="#444444",
    )

    append_scatter_panel(
        draw,
        24,
        120,
        1552,
        460,
        title=f"Corrected PC time relative to recording start ({x_unit})",
        x_label=f"Device time ({x_unit})",
        x_values=x_values,
        y_values=fit_y_values,
        keep_mask=diagnostics.keep_mask,
        x_range=x_range,
        y_range=fit_y_range,
        fit_values=fit_line_y,
        show_legend=True,
    )
    append_scatter_panel(
        draw,
        24,
        608,
        764,
        392,
        title=residual_title,
        x_label=f"Device time ({x_unit})",
        x_values=x_values,
        y_values=residual_values,
        keep_mask=diagnostics.keep_mask,
        x_range=x_range,
        y_range=(residual_range[0], residual_range[1]),
        hlines=[(0.0, "0", "#666666")],
    )
    append_scatter_panel(
        draw,
        812,
        608,
        764,
        392,
        title=delay_title,
        x_label=f"Device time ({x_unit})",
        x_values=x_values,
        y_values=delay_values,
        keep_mask=diagnostics.keep_mask,
        x_range=x_range,
        y_range=(delay_range[0], delay_range[1]),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=95, subsampling=0)


def write_estimated_pc_time(
    output_path: Path,
    sample_count: int,
    sample_rate_hz: float,
    slope: float,
    intercept: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ms_per_sample = 1000.0 / sample_rate_hz
    buffer = bytearray(WRITE_CHUNK_SAMPLES * 4)

    with output_path.open("wb") as stream:
        sample_base = 0
        while sample_base < sample_count:
            chunk_samples = min(WRITE_CHUNK_SAMPLES, sample_count - sample_base)
            byte_offset = 0
            for i in range(chunk_samples):
                dev_ms_abs = (sample_base + i) * ms_per_sample
                pc_ms = round((slope * dev_ms_abs) + intercept)
                wrapped_ms = pc_ms % DAY_MS
                struct.pack_into("<I", buffer, byte_offset, wrapped_ms)
                byte_offset += 4

            stream.write(buffer[: chunk_samples * 4])
            sample_base += chunk_samples


def generate_pc_time(
    analog_path: Path,
    output_path: Path,
    summary_plot_path: Path | None,
    layout: Layout,
    sample_rate_hz: float,
    recording_start_ms: int,
    recording_start_source: str,
) -> GenerationSummary:
    if not analog_path.exists():
        raise FileNotFoundError(f"analogin.dat not found: {analog_path}")

    file_length = analog_path.stat().st_size
    sample_count = get_sample_count(file_length, layout)
    if layout.raw_misc:
        update_indices, update_packed_values = collect_packed_updates_from_raw_misc(analog_path, layout)
    else:
        update_indices, update_packed_values = collect_packed_updates_from_expanded_analog(analog_path)

    if not update_packed_values:
        raise ValueError("no nonzero packed PC-time updates found")

    diagnostics = fit_robust_linear_model(
        update_indices=update_indices,
        update_packed_values=update_packed_values,
        sample_rate_hz=sample_rate_hz,
        recording_start_ms=recording_start_ms,
    )
    kept_count = sum(1 for keep in diagnostics.keep_mask if keep)
    residual_sum_squares = sum(
        residual * residual
        for residual, keep in zip(diagnostics.residual_ms, diagnostics.keep_mask)
        if keep
    )
    residual_rms_ms = math.sqrt(residual_sum_squares / kept_count)

    summary = GenerationSummary(
        output_path=output_path,
        summary_plot_path=summary_plot_path,
        sample_count=sample_count,
        update_count_all=len(update_packed_values),
        update_count_kept=kept_count,
        sample_rate_hz=sample_rate_hz,
        offset_ms=diagnostics.intercept - recording_start_ms,
        offset_sem_ms=diagnostics.intercept_sem,
        drift_ppm=(diagnostics.slope - 1.0) * 1e6,
        drift_sem_ppm=diagnostics.slope_sem * 1e6 if math.isfinite(diagnostics.slope_sem) else math.nan,
        residual_rms_ms=residual_rms_ms,
        recording_start_ms=recording_start_ms,
        recording_start_source=recording_start_source,
    )

    if summary_plot_path is not None:
        write_fit_summary_jpg(summary_plot_path, diagnostics, summary)

    write_estimated_pc_time(output_path, sample_count, sample_rate_hz, diagnostics.slope, diagnostics.intercept)

    return summary


def build_layout(args: argparse.Namespace) -> Layout:
    if args.layout == "ce64-raw-misc":
        defaults = Layout("ce64-raw-misc", raw_misc=True, raw_words_per_cycle=16, raw_low_word_index=14, raw_high_word_index=15)
    elif args.layout == "legacy-raw-misc":
        defaults = Layout("legacy-raw-misc", raw_misc=True, raw_words_per_cycle=8, raw_low_word_index=5, raw_high_word_index=6)
    elif args.layout == "expanded-analog":
        defaults = Layout("expanded-analog", raw_misc=False)
    else:
        raise ValueError(f"unsupported layout: {args.layout}")

    if not defaults.raw_misc:
        return defaults

    return Layout(
        defaults.name,
        raw_misc=True,
        raw_words_per_cycle=args.raw_words_per_cycle or defaults.raw_words_per_cycle,
        raw_low_word_index=args.raw_low_word_index if args.raw_low_word_index is not None else defaults.raw_low_word_index,
        raw_high_word_index=args.raw_high_word_index if args.raw_high_word_index is not None else defaults.raw_high_word_index,
        expand_factor=args.expand_factor or defaults.expand_factor,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pc_time.dat from WILD/CE analogin.dat packed PC-time lanes."
    )
    parser.add_argument("input", type=Path, help="recording folder or analogin.dat path")
    parser.add_argument("-o", "--output", type=Path, help="output path; default is <recording folder>/pc_time.dat")
    parser.add_argument(
        "--layout",
        choices=("ce64-raw-misc", "legacy-raw-misc", "expanded-analog"),
        default="ce64-raw-misc",
        help="packed-lane layout; current CE64 downloads use ce64-raw-misc",
    )
    parser.add_argument(
        "--sample-rate",
        type=parse_positive_float,
        help="output PC-time sample rate in Hz; defaults to CE_params.bin ephys_fs when available",
    )
    parser.add_argument(
        "--base-fs",
        type=parse_positive_float,
        help="raw misc source cadence in Hz; used only when --sample-rate and CE_params.bin are absent",
    )
    parser.add_argument(
        "--recording-start",
        type=parse_ms_of_day,
        help="recording start time as ms since midnight, HH:MM:SS.mmm, or HHMMSS.mmm; defaults to CE_params.bin, then folder suffix",
    )
    parser.add_argument(
        "--summary-plot",
        nargs="?",
        const="__default__",
        help="write a fit summary JPG; default path is <recording folder>/pc_time_fit_summary.jpg",
    )
    parser.add_argument("--raw-words-per-cycle", type=parse_nonnegative_int, help="override raw misc uint16 words per cycle")
    parser.add_argument("--raw-low-word-index", type=parse_nonnegative_int, help="override 0-based packed low word index")
    parser.add_argument("--raw-high-word-index", type=parse_nonnegative_int, help="override 0-based packed high word index")
    parser.add_argument("--expand-factor", type=parse_nonnegative_int, help="raw cycles to output samples expansion factor")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record_dir, analog_path, output_path = resolve_input_paths(args.input, args.output)
    layout = build_layout(args)
    hints = read_ce_params_hint(record_dir)
    sample_rate_hz = resolve_sample_rate(args.sample_rate, args.base_fs, layout, hints)

    if args.recording_start is not None:
        recording_start_ms = args.recording_start
        recording_start_source = "--recording-start"
    elif hints.recording_start_ms is not None:
        recording_start_ms = hints.recording_start_ms
        recording_start_source = "CE_params.bin"
    else:
        inferred_start = infer_recording_start_from_name(record_dir)
        if inferred_start is None:
            inferred_start = 0
            recording_start_source = "00:00 fallback"
            warn(
                "recording start time not found in CE_params.bin or folder name; "
                "anchoring 2^20-ms PC-time cycle to 00:00:00.000. "
                "Pass --recording-start for an explicit reference."
            )
        else:
            recording_start_source = "folder name"
        recording_start_ms = inferred_start

    try:
        summary_plot_path = resolve_summary_plot_path(record_dir, args.summary_plot)
        summary = generate_pc_time(
            analog_path=analog_path,
            output_path=output_path,
            summary_plot_path=summary_plot_path,
            layout=layout,
            sample_rate_hz=sample_rate_hz,
            recording_start_ms=recording_start_ms,
            recording_start_source=recording_start_source,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"generated={summary.output_path}")
    if summary.summary_plot_path is not None:
        print(f"summary_plot={summary.summary_plot_path}")
    print(f"layout={layout.name}")
    print(f"sample_rate_hz={summary.sample_rate_hz:g}")
    print(f"recording_start_ms={summary.recording_start_ms}")
    print(f"recording_start_hms={format_ms_of_day(summary.recording_start_ms)}")
    print(f"recording_start_source={summary.recording_start_source}")
    print(f"samples={summary.sample_count}")
    print(f"updates_kept={summary.update_count_kept}/{summary.update_count_all}")
    print(f"offset_ms={summary.offset_ms:.3f}")
    print(f"offset_sem_ms={summary.offset_sem_ms:.3f}")
    print(f"drift_ppm={summary.drift_ppm:.3f}")
    print(f"drift_sem_ppm={summary.drift_sem_ppm:.3f}")
    print(f"residual_rms_ms={summary.residual_rms_ms:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
