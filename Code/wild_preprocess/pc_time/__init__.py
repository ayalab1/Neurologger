"""Native packed PC-time decoding, fitting, validation, and writing.

This package deliberately has no runtime dependency on the legacy
``WILD_generate_pc_time.py`` script.  ``align_pc_time_file`` remains exported
temporarily for callers which still have an already-generated raw-master file.
"""

from .decode import (
    CE64_RAW_MISC_LAYOUT,
    EXPANDED_ANALOG_LAYOUT,
    PackedUpdates,
    PackedUpdateDiagnostics,
    PcTimeLayout,
    collect_packed_update_rows,
    collect_packed_updates,
    decode_ce_params_recording_start_ms,
    infer_recording_start_from_name,
    read_ce_params_hint,
    resolve_recording_start_ms,
    validate_recording_start_compatibility,
)
from .infer import PcTimeModel, fit_robust_pc_time_model
from .validate import PcTimeOptions, PcTimeValidation, validate_pc_time_interval
from .write import align_pc_time_file, write_interval_pc_time
from .report import (
    pc_time_qc_payload,
    write_pc_time_qc_json,
    write_pc_time_summary_png,
    write_pc_time_warning_png,
)
from .canonical import (
    CameraTimestampMapping,
    CanonicalPcTimeFit,
    fit_gap_aware_pc_time_model,
    map_camera_timestamps_to_canonical,
    map_raw_master_indices_to_canonical,
    unwrap_daily_ms,
    validate_canonical_pc_time_interval,
    write_canonical_interval_pc_time,
)
from .analog_mapping import (
    AnalogPcTimeMappingDiagnostics,
    CanonicalAnalogPcTimeFit,
    fit_pc_time_through_analog_mapping,
)

__all__ = [
    "CE64_RAW_MISC_LAYOUT",
    "CameraTimestampMapping",
    "CanonicalPcTimeFit",
    "CanonicalAnalogPcTimeFit",
    "EXPANDED_ANALOG_LAYOUT",
    "PackedUpdates",
    "AnalogPcTimeMappingDiagnostics",
    "PackedUpdateDiagnostics",
    "PcTimeLayout",
    "PcTimeModel",
    "PcTimeOptions",
    "PcTimeValidation",
    "align_pc_time_file",
    "collect_packed_update_rows",
    "collect_packed_updates",
    "decode_ce_params_recording_start_ms",
    "fit_robust_pc_time_model",
    "fit_gap_aware_pc_time_model",
    "fit_pc_time_through_analog_mapping",
    "infer_recording_start_from_name",
    "map_camera_timestamps_to_canonical",
    "map_raw_master_indices_to_canonical",
    "read_ce_params_hint",
    "resolve_recording_start_ms",
    "validate_recording_start_compatibility",
    "validate_pc_time_interval",
    "validate_canonical_pc_time_interval",
    "unwrap_daily_ms",
    "pc_time_qc_payload",
    "write_pc_time_qc_json",
    "write_pc_time_summary_png",
    "write_pc_time_warning_png",
    "write_interval_pc_time",
    "write_canonical_interval_pc_time",
]
