"""Analog-domain integrity and timeline utilities."""

from .integrity import (
    IMU_MODALITY_INVALID_KINDS,
    AnalogIntegrityEvent,
    AnalogIntegrityMetrics,
    AnalogIntegrityResult,
    scan_analog_frames,
    scan_analog_integrity,
)
from .models import (
    AnalogSyncAnchor,
    AnalogSyncSegment,
    AnalogTimelineResult,
    DeviceClockPrior,
    validate_analog_sync_segments,
)
from .segments import (
    build_clean_analog_segments,
    build_event_driven_analog_segments,
    map_canonical_rows,
    map_raw_rows_to_canonical,
)
from .write import CanonicalAnalogWriteResult, write_canonical_analog
from .imu import (
    ImuFusionData,
    SynchronizedImuDevice,
    SynchronizedImuResult,
    build_imu_from_merged,
    build_synchronized_imu,
    build_filtered_imu_from_merged,
    project_raw_imu_intervals_to_canonical,
    write_synchronized_imu_mat,
)
from .imu_fusion import AhrsFilter, AhrsResult, fuse_imu_ahrs
from .imu_preprocess import (
    ImuAxes,
    ImuPrefusionData,
    scale_imu_nominal,
    resample_imu_1250_to_100,
    calibrate_imu,
    prepare_imu_prefusion,
)

__all__ = [
    "AnalogIntegrityEvent",
    "IMU_MODALITY_INVALID_KINDS",
    "AnalogIntegrityMetrics",
    "AnalogIntegrityResult",
    "scan_analog_frames",
    "scan_analog_integrity",
    "AnalogSyncAnchor",
    "AnalogSyncSegment",
    "AnalogTimelineResult",
    "DeviceClockPrior",
    "validate_analog_sync_segments",
    "build_clean_analog_segments",
    "build_event_driven_analog_segments",
    "map_canonical_rows",
    "map_raw_rows_to_canonical",
    "CanonicalAnalogWriteResult",
    "write_canonical_analog",
    "ImuFusionData",
    "SynchronizedImuDevice",
    "SynchronizedImuResult",
    "build_synchronized_imu",
    "build_filtered_imu_from_merged",
    "build_imu_from_merged",
    "project_raw_imu_intervals_to_canonical",
    "write_synchronized_imu_mat",
    "AhrsFilter",
    "AhrsResult",
    "fuse_imu_ahrs",
    "ImuAxes",
    "ImuPrefusionData",
    "scale_imu_nominal",
    "resample_imu_1250_to_100",
    "calibrate_imu",
    "prepare_imu_prefusion",
]
