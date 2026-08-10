from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


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
        }
