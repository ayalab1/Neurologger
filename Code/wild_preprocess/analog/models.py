"""Immutable source-mapping models for the analog/IMU timeline.

The analog timeline deliberately has its own authority.  Neural clock fits may
seed a clean mapping, but neural storage steps and exclusions must never be
silently projected onto ``analogin.dat``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Integral
from typing import Any, Iterable, Mapping, Protocol


_CONFIDENCES = frozenset({"high", "medium", "low", "unresolved"})
_PUBLISHABLE_CONFIDENCES = frozenset({"high", "medium"})
_STATUSES = frozenset({"NOT_RUN", "OK", "WARN", "FAIL"})


class _IntegrityEventSerializable(Protocol):
    """Small structural protocol that keeps the mapping package decoupled."""

    def to_dict(self) -> Mapping[str, object]: ...


def _integrity_event_dict(
    event: Mapping[str, object] | _IntegrityEventSerializable,
) -> dict[str, object]:
    """Normalize one event while refusing arbitrary manifest payloads."""

    payload: Mapping[str, object]
    if isinstance(event, Mapping):
        payload = event
    elif hasattr(event, "to_dict"):
        payload = event.to_dict()
    else:
        raise ValueError("integrity events must be mappings or provide to_dict()")
    if not isinstance(payload, Mapping):
        raise ValueError("integrity event to_dict() must return a mapping")
    normalized = dict(payload)
    try:
        json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("integrity event must be JSON serializable") from error
    return normalized


def _half_open_interval(name: str, start: int, end: int) -> None:
    if not isinstance(start, Integral) or not isinstance(end, Integral):
        raise ValueError(f"{name} bounds must be integers")
    if start < 0 or end <= start:
        raise ValueError(f"{name} bounds must be a non-empty half-open interval")


@dataclass(frozen=True)
class DeviceClockPrior:
    """Smooth neural clock relation expressed in analog-row coordinates.

    ``source_ephys_scale`` and ``source_ephys_intercept_samples`` describe a
    *continuous* raw-device ephys coordinate as a function of canonical ephys
    coordinate.  It intentionally has no neural storage step/gap fields.
    ``canonical_ephys_start_sample`` is the canonical ephys coordinate of
    analog output row zero and ``phase_ephys_samples`` is the fixed offset of
    raw analog row zero in the device ephys coordinate.  Consequently,

    ``raw_analog_row = scale * canonical_analog_row + intercept``

    is obtained without assuming that a neural storage discontinuity is an
    analog discontinuity.
    """

    device_index: int
    source_ephys_scale: float
    source_ephys_intercept_samples: float
    canonical_ephys_start_sample: float
    ephys_sample_rate_hz: float
    analog_sample_rate_hz: float = 1_250.0
    phase_ephys_samples: float = 0.0
    method: str = "neural_clean_prior"
    support_ids: tuple[str, ...] = ()
    residual_rms_rows: float = 0.0
    residual_max_abs_rows: float = 0.0
    confidence: str = "unresolved"
    evidence: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.device_index, Integral) or self.device_index < 1:
            raise ValueError("device_index must be one-based and positive")
        for name, value in (
            ("source_ephys_scale", self.source_ephys_scale),
            ("source_ephys_intercept_samples", self.source_ephys_intercept_samples),
            ("canonical_ephys_start_sample", self.canonical_ephys_start_sample),
            ("ephys_sample_rate_hz", self.ephys_sample_rate_hz),
            ("analog_sample_rate_hz", self.analog_sample_rate_hz),
            ("phase_ephys_samples", self.phase_ephys_samples),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.source_ephys_scale <= 0:
            raise ValueError("source_ephys_scale must be strictly positive")
        if self.ephys_sample_rate_hz <= 0 or self.analog_sample_rate_hz <= 0:
            raise ValueError("sample rates must be strictly positive")
        if self.method not in {"neural_clean_prior", "shared_ttl", "master_hardware_identity"}:
            raise ValueError("unsupported clock-prior method")
        support_ids = tuple(str(identifier) for identifier in self.support_ids)
        if not support_ids or tuple(sorted(set(support_ids))) != support_ids:
            raise ValueError("support_ids must be non-empty, sorted, and unique")
        object.__setattr__(self, "support_ids", support_ids)
        if (
            not math.isfinite(self.residual_rms_rows)
            or self.residual_rms_rows < 0
            or not math.isfinite(self.residual_max_abs_rows)
            or self.residual_max_abs_rows < 0
            or self.residual_rms_rows > self.residual_max_abs_rows
        ):
            raise ValueError("clock-prior residual bounds must be finite and 0 <= RMS <= maximum")
        if self.confidence not in _CONFIDENCES:
            raise ValueError(f"unsupported clock-prior confidence: {self.confidence}")

    def analog_affine(self) -> tuple[float, float]:
        """Return canonical analog row -> raw analog row affine coefficients."""

        rate_ratio = self.analog_sample_rate_hz / self.ephys_sample_rate_hz
        return (
            float(self.source_ephys_scale),
            float(
                (
                    self.source_ephys_scale * self.canonical_ephys_start_sample
                    + self.source_ephys_intercept_samples
                    - self.phase_ephys_samples
                )
                * rate_ratio
            ),
        )

    @property
    def is_publishable_evidence(self) -> bool:
        """Whether this prior can substantiate a clean published segment."""

        return self.confidence in _PUBLISHABLE_CONFIDENCES and len(self.support_ids) >= 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalogSyncAnchor:
    """One verified canonical analog row to raw analog row correspondence."""

    canonical_row: int
    raw_row: float
    verified: bool
    confidence: str
    verification_source: str = ""
    evidence: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_row, Integral) or self.canonical_row < 0:
            raise ValueError("anchor canonical_row must be a non-negative integer")
        if not math.isfinite(self.raw_row) or self.raw_row < 0:
            raise ValueError("anchor raw_row must be finite and non-negative")
        if self.confidence not in _CONFIDENCES:
            raise ValueError(f"unsupported anchor confidence: {self.confidence}")

    @property
    def is_publishable_evidence(self) -> bool:
        return self.verified and self.confidence in _PUBLISHABLE_CONFIDENCES

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_row": int(self.canonical_row),
            "raw_row": float(self.raw_row),
            "verified": bool(self.verified),
            "confidence": self.confidence,
            "verification_source": self.verification_source,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class AnalogSyncSegment:
    """A verified half-open canonical-analog to raw-analog mapping.

    A publishable segment contains at least two verified medium/high anchors.
    Its source mapping is affine, finite, positive, in support, and is only
    valid within these half-open canonical and raw-row bounds.
    """

    device_index: int
    canonical_start_row: int
    canonical_end_row: int
    raw_start_row: int
    raw_end_row: int
    raw_scale: float
    raw_intercept_rows: float
    anchors: tuple[AnalogSyncAnchor, ...] = ()
    residual_rms_rows: float = 0.0
    residual_max_abs_rows: float = 0.0
    confidence: str = "unresolved"
    start_transition: str = "recording_start"
    end_transition: str = "recording_end"
    publishable: bool = False
    evidence: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.device_index, Integral) or self.device_index < 1:
            raise ValueError("device_index must be one-based and positive")
        _half_open_interval("canonical", self.canonical_start_row, self.canonical_end_row)
        _half_open_interval("raw", self.raw_start_row, self.raw_end_row)
        if not math.isfinite(self.raw_scale) or self.raw_scale <= 0:
            raise ValueError("raw_scale must be finite and strictly positive")
        if not math.isfinite(self.raw_intercept_rows):
            raise ValueError("raw_intercept_rows must be finite")
        if (
            not math.isfinite(self.residual_rms_rows)
            or self.residual_rms_rows < 0
            or not math.isfinite(self.residual_max_abs_rows)
            or self.residual_max_abs_rows < 0
            or self.residual_rms_rows > self.residual_max_abs_rows
        ):
            raise ValueError("segment residual bounds must be finite and 0 <= RMS <= maximum")
        if self.confidence not in _CONFIDENCES:
            raise ValueError(f"unsupported segment confidence: {self.confidence}")

        anchors = tuple(self.anchors)
        if any(not isinstance(anchor, AnalogSyncAnchor) for anchor in anchors):
            raise ValueError("segment anchors must be AnalogSyncAnchor instances")
        anchor_rows = tuple(anchor.canonical_row for anchor in anchors)
        if anchor_rows != tuple(sorted(anchor_rows)) or len(set(anchor_rows)) != len(anchor_rows):
            raise ValueError("segment anchors must be strictly ordered by canonical row")
        object.__setattr__(self, "anchors", anchors)

        if not self._raw_in_support(self._raw_at(self.canonical_start_row)) or not self._raw_in_support(
            self._raw_at(self.canonical_end_row - 1)
        ):
            raise ValueError("segment affine mapping leaves declared raw support")
        for anchor in anchors:
            if not self.contains_canonical_row(anchor.canonical_row):
                raise ValueError("segment anchor lies outside canonical support")
            if not self._raw_in_support(anchor.raw_row):
                raise ValueError("segment anchor lies outside raw support")
            if abs(self._raw_at(anchor.canonical_row) - anchor.raw_row) > self.residual_max_abs_rows + 1e-9:
                raise ValueError("segment anchor residual exceeds recorded maximum residual")
        if self.publishable:
            if self.confidence not in _PUBLISHABLE_CONFIDENCES:
                raise ValueError("publishable segment requires high or medium confidence")
            if len(anchors) < 2 or not all(anchor.is_publishable_evidence for anchor in anchors):
                raise ValueError(
                    "publishable segment requires at least two verified high/medium-confidence anchors"
                )

    def _raw_at(self, canonical_row: int | float) -> float:
        return self.raw_scale * canonical_row + self.raw_intercept_rows

    def _raw_in_support(self, raw_row: float) -> bool:
        return math.isfinite(raw_row) and self.raw_start_row <= raw_row < self.raw_end_row

    def contains_canonical_row(self, canonical_row: int) -> bool:
        return (
            isinstance(canonical_row, Integral)
            and self.canonical_start_row <= canonical_row < self.canonical_end_row
        )

    def map_canonical_row(self, canonical_row: int) -> float:
        if not self.contains_canonical_row(canonical_row):
            raise ValueError("canonical row lies outside segment support")
        raw_row = self._raw_at(int(canonical_row))
        if not self._raw_in_support(raw_row):
            raise ValueError("canonical row maps outside segment raw support")
        return raw_row

    def map_canonical_rows(self, canonical_rows: Iterable[int]) -> tuple[float, ...]:
        rows = tuple(canonical_rows)
        if any(not isinstance(row, Integral) for row in rows):
            raise ValueError("canonical rows must be integers")
        if any(later <= earlier for earlier, later in zip(rows, rows[1:])):
            raise ValueError("canonical rows must be strictly increasing")
        mapped = tuple(self.map_canonical_row(int(row)) for row in rows)
        if any(later <= earlier for earlier, later in zip(mapped, mapped[1:])):
            raise ValueError("segment produced a non-monotone raw mapping")
        return mapped

    @property
    def is_publishable(self) -> bool:
        return self.publishable

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_index": int(self.device_index),
            "canonical_start_row": int(self.canonical_start_row),
            "canonical_end_row": int(self.canonical_end_row),
            "raw_start_row": int(self.raw_start_row),
            "raw_end_row": int(self.raw_end_row),
            "raw_scale": float(self.raw_scale),
            "raw_intercept_rows": float(self.raw_intercept_rows),
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "residual_rms_rows": float(self.residual_rms_rows),
            "residual_max_abs_rows": float(self.residual_max_abs_rows),
            "confidence": self.confidence,
            "start_transition": self.start_transition,
            "end_transition": self.end_transition,
            "publishable": bool(self.publishable),
            "evidence": self.evidence,
        }


def validate_analog_sync_segments(
    segments: Iterable[AnalogSyncSegment], *, device_index: int | None = None
) -> tuple[AnalogSyncSegment, ...]:
    """Validate ordered per-device segments without silently reordering them.

    Canonical gaps and raw-row gaps are allowed.  A raw gap is how a verified
    insertion can be skipped.  Source overlap or reversal would make the
    inverse ambiguous, so it is rejected even across a canonical gap.
    """

    ordered = tuple(segments)
    if any(not isinstance(segment, AnalogSyncSegment) for segment in ordered):
        raise ValueError("segments must be AnalogSyncSegment instances")
    if device_index is not None and (not isinstance(device_index, Integral) or device_index < 1):
        raise ValueError("device_index must be one-based and positive")
    if not ordered:
        return ()
    expected_device = ordered[0].device_index if device_index is None else int(device_index)
    previous_canonical_end = -1
    previous_last_raw: float | None = None
    for segment in ordered:
        if segment.device_index != expected_device:
            raise ValueError("all segments must describe the same device")
        if segment.canonical_start_row < previous_canonical_end:
            raise ValueError("analog segments must be ordered and non-overlapping")
        first_raw = segment.map_canonical_row(segment.canonical_start_row)
        last_raw = segment.map_canonical_row(segment.canonical_end_row - 1)
        if previous_last_raw is not None and first_raw <= previous_last_raw:
            raise ValueError("analog segments must have a globally strictly forward raw mapping")
        previous_canonical_end = segment.canonical_end_row
        previous_last_raw = last_raw
    return ordered


@dataclass(frozen=True)
class AnalogTimelineResult:
    """One device's authoritative analog mapping and compact provenance."""

    device_index: int
    segments: tuple[AnalogSyncSegment, ...] = ()
    integrity_events: tuple[Mapping[str, object] | _IntegrityEventSerializable, ...] = ()
    clock_prior: DeviceClockPrior | None = None
    status: str = "NOT_RUN"
    warnings: tuple[str, ...] = ()
    canonical_sample_rate_hz: float = 1_250.0
    raw_sample_rate_hz: float = 1_250.0
    phase_ephys_samples: float = 0.0
    source_raw_row_count: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.device_index, Integral) or self.device_index < 1:
            raise ValueError("device_index must be one-based and positive")
        if self.status not in _STATUSES:
            raise ValueError(f"unsupported analog status: {self.status}")
        for name, value in (
            ("canonical_sample_rate_hz", self.canonical_sample_rate_hz),
            ("raw_sample_rate_hz", self.raw_sample_rate_hz),
            ("phase_ephys_samples", self.phase_ephys_samples),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.canonical_sample_rate_hz <= 0 or self.raw_sample_rate_hz <= 0:
            raise ValueError("analog sample rates must be positive")
        if self.source_raw_row_count is not None and (
            not isinstance(self.source_raw_row_count, Integral) or self.source_raw_row_count <= 0
        ):
            raise ValueError("source_raw_row_count must be a positive integer when supplied")
        segments = validate_analog_sync_segments(self.segments, device_index=int(self.device_index))
        if self.source_raw_row_count is not None and any(
            segment.raw_end_row > self.source_raw_row_count for segment in segments
        ):
            raise ValueError("segment raw support exceeds source_raw_row_count")
        object.__setattr__(self, "segments", segments)
        events = tuple(self.integrity_events)
        # Validate now so a later canonical transaction cannot fail after data
        # materialization because of arbitrary, unserializable metadata.
        for event in events:
            payload = _integrity_event_dict(event)
            event_device = payload.get("device_index")
            if event_device is not None and event_device != self.device_index:
                raise ValueError("integrity event device_index does not match timeline device_index")
        if self.clock_prior is not None:
            if not isinstance(self.clock_prior, DeviceClockPrior):
                raise ValueError("clock_prior must be a DeviceClockPrior when supplied")
            if self.clock_prior.device_index != self.device_index:
                raise ValueError("clock_prior device_index does not match timeline device_index")
        object.__setattr__(self, "integrity_events", events)
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))

    @property
    def mapping_hash(self) -> str:
        """Stable provenance key shared by analog, PC-time, and IMU products."""

        mapping = {
            "device_index": int(self.device_index),
            "canonical_sample_rate_hz": float(self.canonical_sample_rate_hz),
            "raw_sample_rate_hz": float(self.raw_sample_rate_hz),
            "phase_ephys_samples": float(self.phase_ephys_samples),
            "source_raw_row_count": self.source_raw_row_count,
            "clock_prior": None if self.clock_prior is None else self.clock_prior.to_dict(),
            "segments": [segment.to_dict() for segment in self.segments],
        }
        encoded = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_index": int(self.device_index),
            "segments": [segment.to_dict() for segment in self.segments],
            "integrity_events": [_integrity_event_dict(event) for event in self.integrity_events],
            "clock_prior": None if self.clock_prior is None else self.clock_prior.to_dict(),
            "mapping_hash": self.mapping_hash,
            "status": self.status,
            "warnings": list(self.warnings),
            "canonical_sample_rate_hz": float(self.canonical_sample_rate_hz),
            "raw_sample_rate_hz": float(self.raw_sample_rate_hz),
            "phase_ephys_samples": float(self.phase_ephys_samples),
            "source_raw_row_count": (
                None if self.source_raw_row_count is None else int(self.source_raw_row_count)
            ),
        }

    def to_json(self) -> str:
        """Return compact deterministic metadata for the canonical manifest."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
