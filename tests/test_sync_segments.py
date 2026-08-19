from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from wild_preprocess.models import (
    DeviceSyncAnchor,
    DeviceSyncSegment,
    map_verified_device_sample,
    validate_device_sync_segments,
)


def _segment(**overrides: object) -> DeviceSyncSegment:
    values: dict[str, object] = {
        "device_index": 2,
        "canonical_start_sample": 100,
        "canonical_end_sample": 200,
        "source_start_sample": 1_000,
        "source_end_sample": 1_101,
        "source_scale": 1.0,
        "source_intercept_samples": 900.0,
        "anchors": (
            DeviceSyncAnchor(100, 1_000.0, True, "high", "full-rate start anchor"),
            DeviceSyncAnchor(199, 1_099.0, True, "medium", "full-rate end anchor"),
        ),
        "residual_rms_samples": 0.0,
        "residual_max_abs_samples": 0.0,
        "confidence": "high",
        "publishable": True,
    }
    values.update(overrides)
    return DeviceSyncSegment(**values)  # type: ignore[arg-type]


class DeviceSyncSegmentTests(unittest.TestCase):
    def test_verified_segment_maps_only_its_half_open_supported_range(self) -> None:
        segment = _segment()

        self.assertTrue(segment.is_publishable)
        self.assertEqual(segment.map_canonical_sample(100), 1_000.0)
        self.assertEqual(segment.map_canonical_sample(199), 1_099.0)
        self.assertEqual(segment.map_canonical_samples((100, 150, 199)), (1_000.0, 1_050.0, 1_099.0))
        with self.assertRaisesRegex(ValueError, "outside segment support"):
            segment.map_canonical_sample(200)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            segment.map_canonical_samples((100, 100))

    def test_segment_constructor_rejects_nonfinite_nonmonotone_and_unsupported_mapping(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            _segment(source_scale=0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            _segment(source_intercept_samples=math.nan)
        with self.assertRaisesRegex(ValueError, "half-open"):
            _segment(canonical_end_sample=100)
        with self.assertRaisesRegex(ValueError, "leaves declared source support"):
            _segment(source_end_sample=1_099)
        with self.assertRaisesRegex(ValueError, "anchor residual"):
            _segment(
                anchors=(
                    DeviceSyncAnchor(100, 1_000.0, True, "high"),
                    DeviceSyncAnchor(199, 1_098.0, True, "high"),
                )
            )

    def test_publishable_segment_requires_verified_confident_anchor_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two verified"):
            _segment(anchors=(DeviceSyncAnchor(100, 1_000.0, True, "high"),))
        with self.assertRaisesRegex(ValueError, "at least two verified"):
            _segment(
                anchors=(
                    DeviceSyncAnchor(100, 1_000.0, False, "high"),
                    DeviceSyncAnchor(199, 1_099.0, True, "high"),
                )
            )
        with self.assertRaisesRegex(ValueError, "high or medium confidence"):
            _segment(confidence="low")

        unsupported = _segment(anchors=(), confidence="unresolved", publishable=False)
        self.assertFalse(unsupported.is_publishable)
        self.assertIsNone(map_verified_device_sample((unsupported,), 150))

    def test_collection_mapping_rejects_overlap_and_serializes_deterministically(self) -> None:
        first = _segment()
        second = _segment(
            canonical_start_sample=220,
            canonical_end_sample=320,
            source_start_sample=1_200,
            source_end_sample=1_301,
            source_intercept_samples=980.0,
            anchors=(
                DeviceSyncAnchor(220, 1_200.0, True, "high"),
                DeviceSyncAnchor(319, 1_299.0, True, "high"),
            ),
        )
        segments = validate_device_sync_segments((first, second), device_index=2)
        self.assertEqual(map_verified_device_sample(segments, 150, device_index=2), 1_050.0)
        self.assertIsNone(map_verified_device_sample(segments, 210, device_index=2))
        self.assertEqual(
            first.to_dict(),
            {
                "device_index": 2,
                "canonical_start_sample": 100,
                "canonical_end_sample": 200,
                "source_start_sample": 1_000,
                "source_end_sample": 1_101,
                "source_scale": 1.0,
                "source_intercept_samples": 900.0,
                "anchors": [
                    {
                        "canonical_sample": 100,
                        "source_sample": 1_000.0,
                        "verified": True,
                        "confidence": "high",
                        "evidence": "full-rate start anchor",
                    },
                    {
                        "canonical_sample": 199,
                        "source_sample": 1_099.0,
                        "verified": True,
                        "confidence": "medium",
                        "evidence": "full-rate end anchor",
                    },
                ],
                "residual_rms_samples": 0.0,
                "residual_max_abs_samples": 0.0,
                "confidence": "high",
                "start_transition": "recording_start",
                "end_transition": "recording_end",
                "publishable": True,
                "evidence": "",
            },
        )

        overlapping = _segment(
            canonical_start_sample=199,
            canonical_end_sample=250,
            source_start_sample=1_099,
            source_end_sample=1_150,
            anchors=(
                DeviceSyncAnchor(199, 1_099.0, True, "high"),
                DeviceSyncAnchor(249, 1_149.0, True, "high"),
            ),
        )
        with self.assertRaisesRegex(ValueError, "non-overlapping"):
            validate_device_sync_segments((first, overlapping), device_index=2)

        reversed_source = _segment(
            canonical_start_sample=220,
            canonical_end_sample=320,
            source_start_sample=900,
            source_end_sample=1_001,
            source_intercept_samples=680.0,
            anchors=(
                DeviceSyncAnchor(220, 900.0, True, "high"),
                DeviceSyncAnchor(319, 999.0, True, "high"),
            ),
        )
        with self.assertRaisesRegex(ValueError, "strictly forward"):
            validate_device_sync_segments((first, reversed_source), device_index=2)


if __name__ == "__main__":
    unittest.main()
