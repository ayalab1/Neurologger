from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1] / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from wild_preprocess.sync.attribution import (
    SlaveSlaveEvidence,
    VerifiedPairChange,
    attribute_targeted_events,
)


class TargetedAttributionTest(unittest.TestCase):
    def _attribute(self, changes, slave_evidence=()):
        return attribute_targeted_events(
            changes,
            device_count=3,
            master_device_index=1,
            observed_slave_indices=(2, 3),
            slave_slave_evidence=slave_evidence,
            event_tolerance_samples=3,
        )

    def test_slave_loss_uses_signed_arbitrary_step_and_local_consistency(self) -> None:
        decisions = self._attribute(
            [VerifiedPairChange(2, 1_000, -317, boundary_localized=True)],
            [SlaveSlaveEvidence(2, 3, 1_001, 317)],
        )
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].kind, "slave")
        self.assertEqual(decisions[0].device_indices, (2,))
        self.assertEqual(decisions[0].delta_samples, -317)
        self.assertTrue(decisions[0].boundary_localized)

    def test_master_loss_requires_common_steps_and_stable_slave_pair(self) -> None:
        decisions = self._attribute(
            [VerifiedPairChange(3, 2_003, 20_000), VerifiedPairChange(2, 2_000, 20_000)],
            [SlaveSlaveEvidence(2, 3, 2_001, 0)],
        )
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].kind, "master")
        self.assertEqual(decisions[0].device_indices, (1,))
        self.assertEqual(decisions[0].delta_samples, 20_000)

    def test_discordant_localized_boundaries_cannot_become_narrow_master_loss(self) -> None:
        decisions = attribute_targeted_events(
            [
                VerifiedPairChange(2, 100_000, 4, boundary_localized=True),
                VerifiedPairChange(3, 104_000, 4, boundary_localized=True),
            ],
            device_count=3,
            master_device_index=1,
            observed_slave_indices=(2, 3),
            slave_slave_evidence=(SlaveSlaveEvidence(2, 3, 102_000, 0),),
            event_tolerance_samples=5_000,
            localized_boundary_tolerance_samples=100,
        )
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].kind, "unresolved")
        self.assertFalse(decisions[0].boundary_localized)

    def test_concordant_localized_boundaries_can_become_narrow_master_loss(self) -> None:
        decisions = attribute_targeted_events(
            [
                VerifiedPairChange(2, 100_000, 4, boundary_localized=True),
                VerifiedPairChange(3, 100_080, 4, boundary_localized=True),
            ],
            device_count=3,
            master_device_index=1,
            observed_slave_indices=(2, 3),
            slave_slave_evidence=(SlaveSlaveEvidence(2, 3, 100_040, 0),),
            event_tolerance_samples=5_000,
            localized_boundary_tolerance_samples=100,
        )
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].kind, "master")
        self.assertTrue(decisions[0].boundary_localized)

    def test_positive_single_device_requires_slave_slave_confirmation_for_extra_source(self) -> None:
        decisions = self._attribute(
            [VerifiedPairChange(2, 300, 49)],
            [SlaveSlaveEvidence(2, 3, 300, -49)],
        )
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].kind, "extra_source")
        self.assertEqual(decisions[0].device_indices, (2,))
        self.assertEqual(decisions[0].delta_samples, 49)

    def test_conflicting_signs_remain_unresolved(self) -> None:
        decisions = self._attribute(
            [VerifiedPairChange(2, 400, -11), VerifiedPairChange(3, 400, 11)],
            [SlaveSlaveEvidence(2, 3, 400, 22)],
        )
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].kind, "unresolved")

    def test_two_device_evidence_is_unresolved(self) -> None:
        decisions = attribute_targeted_events(
            [VerifiedPairChange(2, 100, -5)],
            device_count=2,
            master_device_index=1,
            observed_slave_indices=(2,),
        )
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].kind, "unresolved")

    def test_insufficient_extra_source_evidence_remains_unresolved(self) -> None:
        decisions = self._attribute([VerifiedPairChange(2, 300, 49)])
        self.assertEqual(decisions[0].kind, "unresolved")

    def test_negative_single_device_without_slave_evidence_remains_unresolved(self) -> None:
        decisions = self._attribute([VerifiedPairChange(2, 300, -49)])
        self.assertEqual(decisions[0].kind, "unresolved")

    def test_output_order_is_independent_of_input_order(self) -> None:
        decisions = attribute_targeted_events(
            [VerifiedPairChange(3, 900, 7), VerifiedPairChange(2, 100, -5)],
            device_count=3,
            master_device_index=1,
            observed_slave_indices=(2, 3),
            event_tolerance_samples=0,
        )
        self.assertEqual([item.canonical_sample for item in decisions], [100, 900])

    def test_event_cluster_span_cannot_exceed_tolerance_by_transitive_chaining(self) -> None:
        decisions = attribute_targeted_events(
            [
                VerifiedPairChange(2, 0, 7),
                VerifiedPairChange(3, 9, 7),
                VerifiedPairChange(4, 18, 7),
            ],
            device_count=4,
            master_device_index=1,
            observed_slave_indices=(2, 3, 4),
            event_tolerance_samples=10,
        )
        self.assertEqual([item.canonical_sample for item in decisions], [0, 18])


if __name__ == "__main__":
    unittest.main()
