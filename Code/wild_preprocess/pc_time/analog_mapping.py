"""Fit packed PC-clock anchors through authoritative analog segments.

Packed clock words live in a device's raw ``analogin.dat`` row coordinate.
They must therefore be attributed by the analog mapping, not by neural-storage
gap arithmetic.  This module composes the two continuous coordinate changes
and rounds exactly once, onto the canonical neural sample grid used by
``pc_time.dat``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

import numpy as np

from ..analog.models import AnalogSyncSegment
from ..analog.segments import map_raw_rows_to_canonical, validate_analog_segment_collection
from .decode import PackedUpdates
from .infer import PcTimeModel, fit_robust_pc_time_model


@dataclass(frozen=True)
class AnalogPcTimeMappingDiagnostics:
    """Compact accounting for raw-to-canonical PC-clock attribution."""

    input_anchor_count: int
    mapping_valid_anchor_count: int
    invalid_mapping_anchor_count: int
    collapsed_agreeing_duplicate_count: int
    retained_anchor_count: int
    max_quantization_error_seconds: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalAnalogPcTimeFit:
    """A robust PC-time model with analog-mapping provenance.

    All retained coordinate arrays describe the same fitted anchor rows.  The
    canonical analog positions remain fractional to preserve the continuous
    analog evidence; only ``canonical_neural_indices`` is rounded, once, for
    the final ephys-grid fit.
    """

    raw_update_rows: np.ndarray
    packed_values: np.ndarray
    canonical_analog_rows: np.ndarray
    canonical_neural_positions: np.ndarray
    canonical_neural_indices: np.ndarray
    segment_ids: np.ndarray
    quantization_error_seconds: np.ndarray
    provenance_hash: str
    diagnostics: AnalogPcTimeMappingDiagnostics
    model: PcTimeModel

    def __post_init__(self) -> None:
        length = int(np.asarray(self.raw_update_rows).size)
        fields = (
            ("raw_update_rows", self.raw_update_rows, np.int64),
            ("packed_values", self.packed_values, np.uint32),
            ("canonical_analog_rows", self.canonical_analog_rows, np.float64),
            ("canonical_neural_positions", self.canonical_neural_positions, np.float64),
            ("canonical_neural_indices", self.canonical_neural_indices, np.int64),
            ("segment_ids", self.segment_ids, np.int64),
            ("quantization_error_seconds", self.quantization_error_seconds, np.float64),
        )
        for name, values, dtype in fields:
            array = np.array(values, dtype=dtype, copy=True)
            if array.ndim != 1 or array.size != length:
                raise ValueError(f"{name} must be a one-dimensional fitted-anchor vector")
            array.flags.writeable = False
            object.__setattr__(self, name, array)
        if length == 0:
            raise ValueError("canonical analog PC-time fit requires at least one retained anchor")
        if np.any(self.raw_update_rows < 0) or (length > 1 and np.any(np.diff(self.raw_update_rows) <= 0)):
            raise ValueError("raw PC-time rows must be strictly increasing")
        if not np.all(np.isfinite(self.canonical_analog_rows)) or not np.all(
            np.isfinite(self.canonical_neural_positions)
        ):
            raise ValueError("retained PC-time anchors must have finite canonical coordinates")
        if np.any(self.segment_ids < 0):
            raise ValueError("retained PC-time anchors must have an analog segment id")
        if np.any(self.quantization_error_seconds < 0) or not np.all(
            np.isfinite(self.quantization_error_seconds)
        ):
            raise ValueError("PC-time quantization errors must be finite and non-negative")
        if length > 1 and np.any(np.diff(self.canonical_neural_indices) <= 0):
            raise ValueError("canonical neural PC-time indices must be strictly increasing")
        if len(self.provenance_hash) != 64 or any(char not in "0123456789abcdef" for char in self.provenance_hash):
            raise ValueError("provenance_hash must be a SHA-256 hexadecimal digest")


def _mapping_provenance_hash(
    segments: tuple[AnalogSyncSegment, ...],
    *,
    canonical_analog_row_zero_neural_sample: float,
    ephys_sample_rate_hz: float,
    analog_sample_rate_hz: float,
) -> str:
    payload = {
        "segments": [segment.to_dict() for segment in segments],
        "canonical_analog_row_zero_neural_sample": float(canonical_analog_row_zero_neural_sample),
        "ephys_sample_rate_hz": float(ephys_sample_rate_hz),
        "analog_sample_rate_hz": float(analog_sample_rate_hz),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _collapse_quantized_duplicates(
    raw_rows: np.ndarray,
    packed_values: np.ndarray,
    canonical_analog_rows: np.ndarray,
    canonical_neural_positions: np.ndarray,
    canonical_neural_indices: np.ndarray,
    segment_ids: np.ndarray,
    quantization_error_seconds: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], int]:
    """Retain one deterministic representative for agreeing quantized anchors."""

    if canonical_neural_indices.size < 2:
        return (
            raw_rows,
            packed_values,
            canonical_analog_rows,
            canonical_neural_positions,
            canonical_neural_indices,
            segment_ids,
            quantization_error_seconds,
        ), 0
    differences = np.diff(canonical_neural_indices)
    if np.any(differences < 0):
        raise ValueError("analog-mapped PC-time anchors are not monotonic on the canonical neural grid")
    keep = np.r_[True, differences != 0]
    duplicate_count = int(np.count_nonzero(~keep))
    for index in np.flatnonzero(~keep):
        if packed_values[index] != packed_values[index - 1]:
            raise ValueError(
                "distinct packed PC-time values quantize to the same canonical neural sample"
            )
    return (
        raw_rows[keep],
        packed_values[keep],
        canonical_analog_rows[keep],
        canonical_neural_positions[keep],
        canonical_neural_indices[keep],
        segment_ids[keep],
        quantization_error_seconds[keep],
    ), duplicate_count


def fit_pc_time_through_analog_mapping(
    packed_updates: PackedUpdates,
    segments: Iterable[AnalogSyncSegment],
    *,
    canonical_analog_row_zero_neural_sample: float,
    ephys_sample_rate_hz: float,
    analog_sample_rate_hz: float,
    recording_start_ms: int,
    device_index: int | None = None,
) -> CanonicalAnalogPcTimeFit:
    """Map raw PC-clock anchors through analog segments and fit at ephys rate.

    Invalid raw rows are discarded because they have no verified source
    mapping.  A row's canonical analog coordinate is converted continuously
    to a canonical neural coordinate and rounded with :func:`numpy.rint` only
    at that final grid.  Multiple raw anchors that round to one neural sample
    are allowed only if their packed values agree exactly; otherwise their
    contradictory clock evidence is rejected.
    """

    if not isinstance(packed_updates, PackedUpdates):
        raise ValueError("packed_updates must be a PackedUpdates instance")
    for name, value in (
        ("canonical_analog_row_zero_neural_sample", canonical_analog_row_zero_neural_sample),
        ("ephys_sample_rate_hz", ephys_sample_rate_hz),
        ("analog_sample_rate_hz", analog_sample_rate_hz),
    ):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if float(ephys_sample_rate_hz) <= 0 or float(analog_sample_rate_hz) <= 0:
        raise ValueError("sample rates must be positive")
    if not isinstance(recording_start_ms, (int, np.integer)):
        raise ValueError("recording_start_ms must be an integer")

    ordered_segments = validate_analog_segment_collection(segments, device_index=device_index)
    canonical_rows, valid, segment_ids = map_raw_rows_to_canonical(
        ordered_segments,
        packed_updates.raw_row_indices,
        device_index=device_index,
    )
    raw_rows = packed_updates.raw_row_indices[valid]
    packed_values = packed_updates.values[valid]
    canonical_rows = canonical_rows[valid]
    segment_ids = segment_ids[valid]
    if raw_rows.size == 0:
        raise ValueError("no packed PC-time anchors have a valid analog mapping")

    neural_positions = float(canonical_analog_row_zero_neural_sample) + canonical_rows * (
        float(ephys_sample_rate_hz) / float(analog_sample_rate_hz)
    )
    neural_indices = np.rint(neural_positions).astype(np.int64)
    quantization_error_seconds = np.abs(neural_positions - neural_indices) / float(ephys_sample_rate_hz)
    tolerance = (0.5 / float(ephys_sample_rate_hz)) + np.finfo(float).eps
    if np.any(quantization_error_seconds > tolerance):
        raise ValueError("canonical PC-time quantization exceeds half a neural sample")

    retained, collapsed_duplicates = _collapse_quantized_duplicates(
        raw_rows,
        packed_values,
        canonical_rows,
        neural_positions,
        neural_indices,
        segment_ids,
        quantization_error_seconds,
    )
    (
        raw_rows,
        packed_values,
        canonical_rows,
        neural_positions,
        neural_indices,
        segment_ids,
        quantization_error_seconds,
    ) = retained
    if neural_indices.size > 1 and np.any(np.diff(neural_indices) <= 0):
        raise ValueError("canonical neural PC-time indices must be strictly increasing")

    model = fit_robust_pc_time_model(
        neural_indices,
        packed_values,
        float(ephys_sample_rate_hz),
        int(recording_start_ms),
    )
    diagnostics = AnalogPcTimeMappingDiagnostics(
        input_anchor_count=int(packed_updates.raw_row_indices.size),
        mapping_valid_anchor_count=int(np.count_nonzero(valid)),
        invalid_mapping_anchor_count=int(np.count_nonzero(~valid)),
        collapsed_agreeing_duplicate_count=collapsed_duplicates,
        retained_anchor_count=int(neural_indices.size),
        max_quantization_error_seconds=float(np.max(quantization_error_seconds)),
    )
    return CanonicalAnalogPcTimeFit(
        raw_rows,
        packed_values,
        canonical_rows,
        neural_positions,
        neural_indices,
        segment_ids,
        quantization_error_seconds,
        _mapping_provenance_hash(
            ordered_segments,
            canonical_analog_row_zero_neural_sample=float(canonical_analog_row_zero_neural_sample),
            ephys_sample_rate_hz=float(ephys_sample_rate_hz),
            analog_sample_rate_hz=float(analog_sample_rate_hz),
        ),
        diagnostics,
        model,
    )
