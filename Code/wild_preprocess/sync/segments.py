"""Authoritative vector mappings for independently verified device segments."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..models import DeviceSyncSegment, validate_device_sync_segments


INTEGER_MAPPING_TOLERANCE_SAMPLES = 1e-6


def validate_segment_collection(
    segments: Iterable[DeviceSyncSegment], *, device_index: int | None = None
) -> tuple[DeviceSyncSegment, ...]:
    """Validate a device collection without reordering its evidence."""

    return validate_device_sync_segments(segments, device_index=device_index)


def map_canonical_positions(
    segments: Iterable[DeviceSyncSegment],
    canonical_positions: np.ndarray,
    *,
    source_sample_count: int,
    interpolation_half_width: int,
    device_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source coordinates and verified validity for canonical samples.

    Uncovered or non-publishable segment intervals deliberately remain invalid.
    Valid coordinates have verified affine support, are strictly increasing, and
    have enough raw source support for the renderer's interpolation kernel.
    """

    ordered = validate_segment_collection(segments, device_index=device_index)
    positions = np.asarray(canonical_positions, dtype=np.float64)
    if positions.ndim != 1 or not np.all(np.isfinite(positions)):
        raise ValueError("canonical_positions must be a finite one-dimensional array")
    if positions.size > 1 and np.any(np.diff(positions) <= 0):
        raise ValueError("canonical_positions must be strictly increasing")
    if source_sample_count <= 0 or interpolation_half_width < 0:
        raise ValueError("source sample count and interpolation support must be valid")

    mapped = np.full(positions.shape, np.nan, dtype=np.float64)
    valid = np.zeros(positions.shape, dtype=bool)
    for segment in ordered:
        if not segment.is_publishable:
            continue
        in_segment = (
            (positions >= segment.canonical_start_sample)
            & (positions < segment.canonical_end_sample)
        )
        if not np.any(in_segment):
            continue
        values = segment.source_scale * positions[in_segment] + segment.source_intercept_samples
        mapped[in_segment] = values
        nearest = np.rint(values)
        integer = np.abs(values - nearest) <= INTEGER_MAPPING_TOLERANCE_SAMPLES
        supported = np.where(
            integer,
            (nearest >= 0) & (nearest < source_sample_count),
            (values >= interpolation_half_width - 1)
            & (values <= source_sample_count - 1 - interpolation_half_width),
        )
        declared_support = (
            (values >= segment.source_start_sample)
            & (values < segment.source_end_sample)
        )
        valid[in_segment] = np.isfinite(values) & declared_support & supported

    valid_values = mapped[valid]
    if not np.all(np.isfinite(valid_values)):
        raise ValueError("valid segment mapping contains non-finite source coordinates")
    if valid_values.size > 1 and np.any(np.diff(valid_values) <= 0):
        raise ValueError("valid segment mapping is not strictly monotone")
    return mapped, valid


def map_source_positions_to_canonical(
    segments: Iterable[DeviceSyncSegment],
    source_positions: np.ndarray,
    *,
    device_index: int | None = None,
) -> np.ndarray:
    """Invert verified segments for raw-source duplication attribution.

    ``-1`` denotes unsupported, non-integrally mappable, or ambiguous source
    samples.  This inverse intentionally sees only publishable segments.
    """

    ordered = validate_segment_collection(segments, device_index=device_index)
    source = np.asarray(source_positions, dtype=np.float64)
    if not np.all(np.isfinite(source)):
        raise ValueError("source_positions must be finite")
    canonical = np.full(source.shape, -1, dtype=np.int64)
    matches = np.zeros(source.shape, dtype=np.uint8)
    for segment in ordered:
        if not segment.is_publishable:
            continue
        candidate = (source - segment.source_intercept_samples) / segment.source_scale
        rounded = np.rint(candidate).astype(np.int64)
        in_segment = (
            (source >= segment.source_start_sample)
            & (source < segment.source_end_sample)
            & (candidate >= segment.canonical_start_sample)
            & (candidate < segment.canonical_end_sample)
            & (rounded >= segment.canonical_start_sample)
            & (rounded < segment.canonical_end_sample)
        )
        remapped = segment.source_scale * rounded + segment.source_intercept_samples
        in_segment &= np.abs(remapped - source) <= 0.5
        canonical[in_segment] = rounded[in_segment]
        matches[in_segment] += 1
    canonical[matches != 1] = -1
    return canonical
