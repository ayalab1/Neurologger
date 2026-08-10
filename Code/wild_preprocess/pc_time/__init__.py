"""Native packed PC-time decoding, fitting, validation, and writing.

This package deliberately has no runtime dependency on the legacy
``WILD_generate_pc_time.py`` script.  ``align_pc_time_file`` remains exported
temporarily for callers which still have an already-generated raw-master file.
"""

from .decode import (
    CE64_RAW_MISC_LAYOUT,
    EXPANDED_ANALOG_LAYOUT,
    PcTimeLayout,
    collect_packed_updates,
    infer_recording_start_from_name,
    resolve_recording_start_ms,
)
from .infer import PcTimeModel, fit_robust_pc_time_model
from .validate import PcTimeOptions, PcTimeValidation, validate_pc_time_interval
from .write import align_pc_time_file, write_interval_pc_time
from .report import pc_time_qc_payload, write_pc_time_qc_json, write_pc_time_summary_png

__all__ = [
    "CE64_RAW_MISC_LAYOUT",
    "EXPANDED_ANALOG_LAYOUT",
    "PcTimeLayout",
    "PcTimeModel",
    "PcTimeOptions",
    "PcTimeValidation",
    "align_pc_time_file",
    "collect_packed_updates",
    "fit_robust_pc_time_model",
    "infer_recording_start_from_name",
    "resolve_recording_start_ms",
    "validate_pc_time_interval",
    "pc_time_qc_payload",
    "write_pc_time_qc_json",
    "write_pc_time_summary_png",
    "write_interval_pc_time",
]
