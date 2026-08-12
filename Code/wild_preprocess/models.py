from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Recording:
    folder: Path
    amplifier_file: Path
    analog_file: Path
    ce_params_file: Path
    device_name: str
    recording_name: str
    fs: int
    n_channels: int
    n_samples: int
    analog_channels: int
    analog_samples: int

    @property
    def duration_sec(self) -> float:
        return self.n_samples / self.fs


@dataclass(frozen=True)
class SyncOptions:
    initial_start_seconds: float = 30.0
    initial_duration_seconds: float = 120.0
    initial_max_lag_seconds: float = 30.0
    window_seconds: float = 10.0
    step_seconds: float = 5.0
    tracking_max_lag_samples: int = 100
    highpass_hz: float = 200.0
    peak_exclusion_samples: int = 24
    min_peak_correlation: float = 0.05
    min_peak_to_background: float = 1.2
    min_peak_margin_fraction: float = 0.01
    min_accepted_fraction: float = 0.75
    min_accepted_observations: int = 10
    min_accepted_span_seconds: float = 60.0
    short_recording_seconds: float = 60.0
    short_min_accepted_observations: int = 3
    max_model_rms_samples: float = 4.0
    max_model_residual_samples: float = 12.0
    max_consecutive_rejections: int = 4
    max_consecutive_model_outliers: int = 2
    max_observed_offset_step_samples: float = 50.0
    max_offset_level_shift_samples: float = 8.0
    persistent_level_shift_observations: int = 3
    report_offset_level_shift_samples: float = 4.0
    warn_drift_ppm: float = 500.0
    chunk_seconds: float = 5.0
    # New options remain after the original positional fields so legacy
    # positional SyncOptions construction keeps its previous meaning.
    reacquisition_max_lag_seconds: float = 1.0
    gap_min_step_samples: float = 50.0
    gap_persistence_observations: int = 2
    gap_level_tolerance_samples: float = 12.0
    gap_event_time_tolerance_seconds: float = 0.25
    max_parallel_workers: int = 2
    endpoint_probe_seconds: float = 2.0
    # Coarse-to-fine reacquisition settings are intentionally appended so
    # legacy positional construction retains its original meaning.
    coarse_feature_rate_hz: float = 1_000.0
    coarse_reacquisition_max_lag_seconds: float = 30.0
    coarse_reacquisition_growth_factor: float = 2.0


@dataclass
class SyncObservation:
    center_time_sec: float
    predicted_offset_samples: float
    observed_offset_samples: float
    residual_lag_samples: float
    peak_correlation: float
    peak_to_background: float
    peak_margin_fraction: float
    secondary_lag_samples: float | None
    accepted: bool
    rejection_reason: str = ""
    search_mode: str = "narrow"
    search_half_width_samples: int = 0
    model_inlier: bool = False
    model_residual_samples: float = float("nan")


@dataclass(frozen=True)
class RelativeOffsetStep:
    """One persistent source-offset change measured for a master/slave pair."""

    master_sample: int
    time_sec: float
    offset_step_samples: float
    missing_samples: int
    offset_before_samples: float
    offset_after_samples: float
    confidence: str
    evidence: str


@dataclass(frozen=True)
class DeviceGap:
    """One confidently attributed missing interval on the canonical time axis."""

    device_index: int
    canonical_start_sample: int
    missing_samples: int
    duration_ms: float
    confidence: str = "high"
    action: str = "fill_or_crop"
    evidence: str = ""

    @property
    def canonical_end_sample(self) -> int:
        return self.canonical_start_sample + self.missing_samples


_CLASSIFIED_INTERVAL_ACTIONS: dict[str, frozenset[str]] = {
    "missing": frozenset({"zero_fill"}),
    "duplicate_destination": frozenset({"zero_fill"}),
    "extra_source": frozenset({"skip_source"}),
    "unresolved_boundary": frozenset({"zero_fill"}),
    "terminal_unsupported": frozenset({"zero_fill"}),
    "interpolation_guard": frozenset({"zero_fill"}),
    "postmerge_unverified": frozenset({"zero_fill"}),
}
_CLASSIFIED_INTERVAL_CONFIDENCES = frozenset({"high", "medium", "low", "unresolved"})


@dataclass(frozen=True)
class ClassifiedInterval:
    """One explicit post-hoc correction decision on the canonical sample axis.

    Device indices are one-based to match reports.  Canonical and optional
    source coordinates are half-open intervals.  The source coordinates name
    the affected raw source interval when that is meaningful; a duplication keeps
    both its later canonical destination and its earlier source occurrence.
    """

    affected_device_indices: tuple[int, ...]
    canonical_start_sample: int
    canonical_end_sample: int
    kind: str
    action: str
    confidence: str
    source_start_sample: int | None = None
    source_end_sample: int | None = None
    evidence: str = ""

    def __post_init__(self) -> None:
        devices = tuple(int(index) for index in self.affected_device_indices)
        if not devices or any(index < 1 for index in devices):
            raise ValueError("affected_device_indices must contain one-based positive indices")
        if tuple(sorted(set(devices))) != devices:
            raise ValueError("affected_device_indices must be sorted and unique")
        if self.canonical_start_sample < 0 or self.canonical_end_sample <= self.canonical_start_sample:
            raise ValueError("canonical coordinates must be a non-empty half-open interval")
        allowed_actions = _CLASSIFIED_INTERVAL_ACTIONS.get(self.kind)
        if allowed_actions is None:
            raise ValueError(f"unsupported classified interval kind: {self.kind}")
        if self.action not in allowed_actions:
            raise ValueError(
                f"unsupported action {self.action!r} for classified interval kind {self.kind!r}"
            )
        if self.confidence not in _CLASSIFIED_INTERVAL_CONFIDENCES:
            raise ValueError(f"unsupported classified interval confidence: {self.confidence}")
        source_start = self.source_start_sample
        source_end = self.source_end_sample
        if (source_start is None) != (source_end is None):
            raise ValueError("source interval requires both start and end coordinates")
        if source_start is not None:
            if source_start < 0 or source_end is None or source_end <= source_start:
                raise ValueError("source coordinates must be a non-empty half-open interval")
        if self.kind in {"duplicate_destination", "extra_source"} and source_start is None:
            raise ValueError(f"{self.kind} requires source coordinates")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UnresolvedBoundary:
    """A local all-device exclusion where a sync mapping is not attributable."""

    canonical_start_sample: int
    canonical_end_sample: int
    pair_slave_indices: tuple[int, ...]
    evidence: str
    confidence: str = "unresolved"

    def __post_init__(self) -> None:
        pairs = tuple(int(index) for index in self.pair_slave_indices)
        if not pairs or any(index < 1 for index in pairs):
            raise ValueError("pair_slave_indices must contain one-based positive indices")
        if tuple(sorted(set(pairs))) != pairs:
            raise ValueError("pair_slave_indices must be sorted and unique")
        if self.canonical_start_sample < 0 or self.canonical_end_sample <= self.canonical_start_sample:
            raise ValueError("boundary coordinates must be a non-empty half-open interval")
        if self.confidence != "unresolved":
            raise ValueError("unresolved boundaries must use unresolved confidence")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeviceTerminalSupport:
    """Exclusive canonical endpoint supported by one device's sync evidence."""

    device_index: int
    supported_canonical_end_sample: int
    evidence: str
    confidence: str = "unresolved"

    def __post_init__(self) -> None:
        if self.device_index < 1:
            raise ValueError("device_index must be one-based and positive")
        if self.supported_canonical_end_sample < 0:
            raise ValueError("supported_canonical_end_sample must be non-negative")
        if self.confidence not in _CLASSIFIED_INTERVAL_CONFIDENCES:
            raise ValueError(f"unsupported terminal-support confidence: {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeviceSourceStep:
    """One piecewise source-coordinate jump retained after a masked boundary."""

    device_index: int
    canonical_sample: int
    source_step_samples: float
    confidence: str
    evidence: str = ""

    def __post_init__(self) -> None:
        if self.device_index < 1:
            raise ValueError("device_index must be one-based and positive")
        if self.canonical_sample < 0:
            raise ValueError("canonical_sample must be non-negative")
        if not math.isfinite(self.source_step_samples) or self.source_step_samples == 0:
            raise ValueError("source_step_samples must be finite and non-zero")
        if self.confidence not in _CLASSIFIED_INTERVAL_CONFIDENCES:
            raise ValueError(f"unsupported source-step confidence: {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_PUBLISHABLE_SEGMENT_CONFIDENCES = frozenset({"high", "medium"})


@dataclass(frozen=True)
class DeviceSyncAnchor:
    """One independently verified canonical-to-source correspondence.

    ``canonical_sample`` names an integer sample on the canonical neural time
    axis.  ``source_sample`` remains floating point because an affine clock
    can legitimately map an integer canonical sample between source samples.
    The anchor itself is immutable evidence; the segment validates that its
    affine mapping agrees with this measurement within the segment's recorded
    residual bound.
    """

    canonical_sample: int
    source_sample: float
    verified: bool
    confidence: str
    evidence: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_sample, Integral) or self.canonical_sample < 0:
            raise ValueError("anchor canonical_sample must be a non-negative integer")
        if not math.isfinite(self.source_sample) or self.source_sample < 0:
            raise ValueError("anchor source_sample must be finite and non-negative")
        if self.confidence not in _CLASSIFIED_INTERVAL_CONFIDENCES:
            raise ValueError(f"unsupported anchor confidence: {self.confidence}")

    @property
    def is_publishable_evidence(self) -> bool:
        """Whether this anchor can support a published segment."""

        return self.verified and self.confidence in _PUBLISHABLE_SEGMENT_CONFIDENCES

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_sample": int(self.canonical_sample),
            "source_sample": float(self.source_sample),
            "verified": bool(self.verified),
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class DeviceSyncSegment:
    """A verified, independently reacquired source mapping for one device.

    Coordinates are half-open: all canonical samples in
    ``[canonical_start_sample, canonical_end_sample)`` map through
    ``source_scale * canonical_sample + source_intercept_samples`` and must
    remain in the raw source support ``[source_start_sample,
    source_end_sample)``.  A segment marked publishable has at least two
    independently verified, high/medium-confidence anchors.  Gaps between
    segments are intentional unsupported canonical intervals, rather than an
    invitation to inherit an unresolved source-coordinate step.
    """

    device_index: int
    canonical_start_sample: int
    canonical_end_sample: int
    source_start_sample: int
    source_end_sample: int
    source_scale: float
    source_intercept_samples: float
    anchors: tuple[DeviceSyncAnchor, ...] = ()
    residual_rms_samples: float = 0.0
    residual_max_abs_samples: float = 0.0
    confidence: str = "unresolved"
    start_transition: str = "recording_start"
    end_transition: str = "recording_end"
    publishable: bool = False
    evidence: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.device_index, Integral) or self.device_index < 1:
            raise ValueError("device_index must be one-based and positive")
        coordinates = (
            ("canonical_start_sample", self.canonical_start_sample),
            ("canonical_end_sample", self.canonical_end_sample),
            ("source_start_sample", self.source_start_sample),
            ("source_end_sample", self.source_end_sample),
        )
        if any(not isinstance(value, Integral) for _, value in coordinates):
            raise ValueError("segment canonical and source bounds must be integers")
        if self.canonical_start_sample < 0 or self.canonical_end_sample <= self.canonical_start_sample:
            raise ValueError("segment canonical bounds must be a non-empty half-open interval")
        if self.source_start_sample < 0 or self.source_end_sample <= self.source_start_sample:
            raise ValueError("segment source bounds must be a non-empty half-open interval")
        if not math.isfinite(self.source_scale) or self.source_scale <= 0:
            raise ValueError("segment source_scale must be finite and strictly positive")
        if not math.isfinite(self.source_intercept_samples):
            raise ValueError("segment source_intercept_samples must be finite")
        if (
            not math.isfinite(self.residual_rms_samples)
            or self.residual_rms_samples < 0
            or not math.isfinite(self.residual_max_abs_samples)
            or self.residual_max_abs_samples < 0
            or self.residual_rms_samples > self.residual_max_abs_samples
        ):
            raise ValueError("segment residual bounds must be finite and 0 <= RMS <= maximum")
        if self.confidence not in _CLASSIFIED_INTERVAL_CONFIDENCES:
            raise ValueError(f"unsupported segment confidence: {self.confidence}")

        anchors = tuple(self.anchors)
        if any(not isinstance(anchor, DeviceSyncAnchor) for anchor in anchors):
            raise ValueError("segment anchors must be DeviceSyncAnchor instances")
        if tuple(anchor.canonical_sample for anchor in anchors) != tuple(
            sorted(anchor.canonical_sample for anchor in anchors)
        ) or len({anchor.canonical_sample for anchor in anchors}) != len(anchors):
            raise ValueError("segment anchors must be strictly ordered by canonical sample")
        object.__setattr__(self, "anchors", anchors)

        first_source = self._raw_source_at(self.canonical_start_sample)
        last_source = self._raw_source_at(self.canonical_end_sample - 1)
        if not self._source_in_support(first_source) or not self._source_in_support(last_source):
            raise ValueError("segment affine mapping leaves declared source support")
        for anchor in anchors:
            if not self.contains_canonical_sample(anchor.canonical_sample):
                raise ValueError("segment anchor lies outside canonical support")
            if not self._source_in_support(anchor.source_sample):
                raise ValueError("segment anchor lies outside source support")
            residual = abs(self._raw_source_at(anchor.canonical_sample) - anchor.source_sample)
            if residual > self.residual_max_abs_samples + 1e-9:
                raise ValueError("segment anchor residual exceeds recorded maximum residual")
        if self.publishable:
            if self.confidence not in _PUBLISHABLE_SEGMENT_CONFIDENCES:
                raise ValueError("publishable segment requires high or medium confidence")
            if len(anchors) < 2 or not all(anchor.is_publishable_evidence for anchor in anchors):
                raise ValueError(
                    "publishable segment requires at least two verified high/medium-confidence anchors"
                )

    def _raw_source_at(self, canonical_sample: int) -> float:
        return self.source_scale * canonical_sample + self.source_intercept_samples

    def _source_in_support(self, source_sample: float) -> bool:
        return (
            math.isfinite(source_sample)
            and self.source_start_sample <= source_sample < self.source_end_sample
        )

    def contains_canonical_sample(self, canonical_sample: int) -> bool:
        return (
            isinstance(canonical_sample, Integral)
            and self.canonical_start_sample <= canonical_sample < self.canonical_end_sample
        )

    def map_canonical_sample(self, canonical_sample: int) -> float:
        """Map one supported canonical sample or raise instead of extrapolating."""

        if not self.contains_canonical_sample(canonical_sample):
            raise ValueError("canonical sample lies outside segment support")
        source_sample = self._raw_source_at(int(canonical_sample))
        if not self._source_in_support(source_sample):
            raise ValueError("canonical sample maps outside segment source support")
        return source_sample

    def map_canonical_samples(self, canonical_samples: Iterable[int]) -> tuple[float, ...]:
        """Pure ordered mapping for a valid canonical run.

        The positive scale validated at construction makes the returned source
        coordinates strictly increasing whenever the input canonical samples
        are strictly increasing.  Reject unsorted input rather than silently
        turning a non-monotone caller request into a publishable mapping.
        """

        samples = tuple(canonical_samples)
        if any(not isinstance(sample, Integral) for sample in samples):
            raise ValueError("canonical samples must be integers")
        if any(later <= earlier for earlier, later in zip(samples, samples[1:])):
            raise ValueError("canonical samples must be strictly increasing")
        mapped = tuple(self.map_canonical_sample(int(sample)) for sample in samples)
        if any(later <= earlier for earlier, later in zip(mapped, mapped[1:])):
            raise ValueError("segment produced a non-monotone source mapping")
        return mapped

    @property
    def is_publishable(self) -> bool:
        """Construction has already verified all publishability requirements."""

        return self.publishable

    def to_dict(self) -> dict[str, Any]:
        """Return an explicitly ordered, JSON-ready stable representation."""

        return {
            "device_index": int(self.device_index),
            "canonical_start_sample": int(self.canonical_start_sample),
            "canonical_end_sample": int(self.canonical_end_sample),
            "source_start_sample": int(self.source_start_sample),
            "source_end_sample": int(self.source_end_sample),
            "source_scale": float(self.source_scale),
            "source_intercept_samples": float(self.source_intercept_samples),
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "residual_rms_samples": float(self.residual_rms_samples),
            "residual_max_abs_samples": float(self.residual_max_abs_samples),
            "confidence": self.confidence,
            "start_transition": self.start_transition,
            "end_transition": self.end_transition,
            "publishable": bool(self.publishable),
            "evidence": self.evidence,
        }


def validate_device_sync_segments(
    segments: Iterable[DeviceSyncSegment],
    *,
    device_index: int | None = None,
) -> tuple[DeviceSyncSegment, ...]:
    """Validate one device's ordered, non-overlapping segment collection.

    The caller must retain canonical ordering explicitly; this function does
    not sort segments because changing an input order can conceal a malformed
    or ambiguous mapping.  Gaps are allowed and mean unsupported output.
    """

    ordered = tuple(segments)
    if any(not isinstance(segment, DeviceSyncSegment) for segment in ordered):
        raise ValueError("segments must be DeviceSyncSegment instances")
    if device_index is not None and (not isinstance(device_index, Integral) or device_index < 1):
        raise ValueError("device_index must be one-based and positive")
    if not ordered:
        return ()
    expected_device = ordered[0].device_index if device_index is None else int(device_index)
    previous_end = -1
    previous_publishable_last_source: float | None = None
    for segment in ordered:
        if segment.device_index != expected_device:
            raise ValueError("all segments must describe the same device")
        if segment.canonical_start_sample < previous_end:
            raise ValueError("device segments must be ordered and non-overlapping")
        if segment.is_publishable:
            first_source = segment.map_canonical_sample(segment.canonical_start_sample)
            if (
                previous_publishable_last_source is not None
                and first_source <= previous_publishable_last_source
            ):
                raise ValueError(
                    "publishable device segments must have a strictly forward source mapping"
                )
            previous_publishable_last_source = segment.map_canonical_sample(
                segment.canonical_end_sample - 1
            )
        previous_end = segment.canonical_end_sample
    return ordered


def map_verified_device_sample(
    segments: Iterable[DeviceSyncSegment],
    canonical_sample: int,
    *,
    device_index: int | None = None,
) -> float | None:
    """Map a sample only through a publishable verified segment.

    ``None`` is the deliberate result for a segment gap, an unsupported
    segment, or an out-of-range canonical sample.  Merge code can therefore
    convert this directly to zero-filled data and validity 0 without trusting
    an unresolved continuation.
    """

    ordered = validate_device_sync_segments(segments, device_index=device_index)
    if not isinstance(canonical_sample, Integral) or canonical_sample < 0:
        raise ValueError("canonical_sample must be a non-negative integer")
    for segment in ordered:
        if segment.contains_canonical_sample(canonical_sample):
            return segment.map_canonical_sample(int(canonical_sample)) if segment.is_publishable else None
    return None


@dataclass(frozen=True)
class SyncModel:
    intercept_samples: float
    slope_samples_per_second: float
    drift_ppm: float
    residual_rms_samples: float
    residual_max_abs_samples: float
    accepted_count: int
    observation_count: int
    is_constant_offset: bool = False
    offset_steps: tuple[RelativeOffsetStep, ...] = ()

    def affine_offset_at_seconds(self, time_sec: float) -> float:
        return self.intercept_samples + self.slope_samples_per_second * time_sec

    def offset_at_seconds(self, time_sec: float) -> float:
        step = sum(
            event.offset_step_samples
            for event in self.offset_steps
            if event.time_sec <= time_sec
        )
        return self.affine_offset_at_seconds(time_sec) + step

    def source_scale(self, fs: float) -> float:
        return 1.0 + self.slope_samples_per_second / fs


@dataclass
class SyncPairResult:
    master_index: int
    slave_index: int
    master_folder: str
    slave_folder: str
    initial_offset_samples: float
    initial_peak_to_background: float
    initial_peak_margin_fraction: float
    model: SyncModel
    observations: list[SyncObservation] = field(default_factory=list)
    status: str = "FAIL"
    message: str = ""
    figure_file: str = ""
    validated_start_master_sample: int = 0
    terminal_crop_master_sample: int | None = None
    terminal_crop_reason: str = ""

    @property
    def final_offset_samples(self) -> float:
        if not self.observations:
            return self.model.intercept_samples
        return self.model.offset_at_seconds(self.observations[-1].center_time_sec)

    @property
    def offset_drift_samples(self) -> float:
        return self.final_offset_samples - self.initial_offset_samples

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    recordings: list[Recording]
    master_index: int
    pairs: list[SyncPairResult]
    run_id: str
    status: str
    output_folder: Path
    outputs: dict[str, str] = field(default_factory=dict)
    device_gaps: list[DeviceGap] = field(default_factory=list)
    unresolved_gap_messages: list[str] = field(default_factory=list)
    classified_intervals: list[ClassifiedInterval] = field(default_factory=list)
    unresolved_boundaries: list[UnresolvedBoundary] = field(default_factory=list)
    device_terminal_support: list[DeviceTerminalSupport] = field(default_factory=list)
    device_source_steps: list[DeviceSourceStep] = field(default_factory=list)
    device_sync_segments: list[DeviceSyncSegment] = field(default_factory=list)
    targeted_attributions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordings": [asdict(recording) for recording in self.recordings],
            "master_index": self.master_index,
            "pairs": [pair.to_dict() for pair in self.pairs],
            "run_id": self.run_id,
            "status": self.status,
            "output_folder": str(self.output_folder),
            "outputs": self.outputs,
            "device_gaps": [asdict(gap) for gap in self.device_gaps],
            "unresolved_gap_messages": list(self.unresolved_gap_messages),
            "classified_intervals": [interval.to_dict() for interval in self.classified_intervals],
            "unresolved_boundaries": [boundary.to_dict() for boundary in self.unresolved_boundaries],
            "device_terminal_support": [support.to_dict() for support in self.device_terminal_support],
            "device_source_steps": [step.to_dict() for step in self.device_source_steps],
            "device_sync_segments": [segment.to_dict() for segment in self.device_sync_segments],
            "targeted_attributions": list(self.targeted_attributions),
        }
