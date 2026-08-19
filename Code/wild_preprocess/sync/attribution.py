"""Deterministic attribution of locally verified synchronization transitions.

The normal full-session tracker measures master/slave relationships.  This
module consumes only the small set of *verified* measurements made around a
candidate transition, including optional targeted slave/slave measurements.
It deliberately does not infer a source mapping or modify a global clock.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral


_ATTRIBUTION_KINDS = frozenset({"slave", "master", "extra_source", "unresolved"})


@dataclass(frozen=True)
class VerifiedPairChange:
    """A verified change in ``slave source - master source`` at one boundary.

    Device indices are one-based.  ``delta_samples`` is an arbitrary signed
    integer: negative values put the slave behind the master, while positive
    values put it ahead.  A zero value is not a transition and is represented
    instead by including that slave in ``observed_slave_indices``.
    """

    slave_device_index: int
    canonical_sample: int
    delta_samples: int
    evidence: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.slave_device_index, Integral) or self.slave_device_index < 1:
            raise ValueError("slave_device_index must be a one-based positive integer")
        if not isinstance(self.canonical_sample, Integral) or self.canonical_sample < 0:
            raise ValueError("canonical_sample must be a non-negative integer")
        if not isinstance(self.delta_samples, Integral) or self.delta_samples == 0:
            raise ValueError("delta_samples must be a non-zero integer")


@dataclass(frozen=True)
class SlaveSlaveEvidence:
    """A local slave/slave result using ``second source - first source``.

    A zero delta is affirmative evidence that the two slaves remained aligned.
    ``verified=False`` records an attempted but unusable correlation and never
    contributes to an attribution.
    """

    first_device_index: int
    second_device_index: int
    canonical_sample: int
    second_minus_first_delta_samples: int
    verified: bool = True
    evidence: str = ""

    def __post_init__(self) -> None:
        indices = (self.first_device_index, self.second_device_index)
        if any(not isinstance(index, Integral) or index < 1 for index in indices):
            raise ValueError("slave/slave device indices must be one-based positive integers")
        if self.first_device_index == self.second_device_index:
            raise ValueError("slave/slave evidence requires two distinct devices")
        if not isinstance(self.canonical_sample, Integral) or self.canonical_sample < 0:
            raise ValueError("canonical_sample must be a non-negative integer")
        if not isinstance(self.second_minus_first_delta_samples, Integral):
            raise ValueError("second_minus_first_delta_samples must be an integer")


@dataclass(frozen=True)
class AttributionDecision:
    """One device-level event classification, or a conservative unresolved one."""

    kind: str
    canonical_sample: int
    device_indices: tuple[int, ...]
    delta_samples: int
    evidence: str

    def __post_init__(self) -> None:
        if self.kind not in _ATTRIBUTION_KINDS:
            raise ValueError(f"unsupported attribution kind: {self.kind}")
        if not isinstance(self.canonical_sample, Integral) or self.canonical_sample < 0:
            raise ValueError("canonical_sample must be a non-negative integer")
        if not isinstance(self.delta_samples, Integral) or self.delta_samples == 0:
            raise ValueError("delta_samples must be a non-zero integer")
        devices = tuple(int(index) for index in self.device_indices)
        if not devices or any(index < 1 for index in devices):
            raise ValueError("device_indices must contain one-based positive indices")
        if tuple(sorted(set(devices))) != devices:
            raise ValueError("device_indices must be sorted and unique")


def _validate_configuration(
    *, device_count: int, master_device_index: int, event_tolerance_samples: int,
    magnitude_tolerance_samples: int,
) -> set[int]:
    if not isinstance(device_count, Integral) or device_count < 2:
        raise ValueError("device_count must be an integer of at least two")
    if not isinstance(master_device_index, Integral) or not 1 <= master_device_index <= device_count:
        raise ValueError("master_device_index must be a valid one-based device index")
    if not isinstance(event_tolerance_samples, Integral) or event_tolerance_samples < 0:
        raise ValueError("event_tolerance_samples must be a non-negative integer")
    if not isinstance(magnitude_tolerance_samples, Integral) or magnitude_tolerance_samples < 0:
        raise ValueError("magnitude_tolerance_samples must be a non-negative integer")
    return set(range(1, device_count + 1)) - {master_device_index}


def _cluster_changes(
    changes: tuple[VerifiedPairChange, ...], *, event_tolerance_samples: int
) -> tuple[tuple[VerifiedPairChange, ...], ...]:
    ordered = tuple(
        sorted(
            changes,
            key=lambda item: (item.canonical_sample, item.slave_device_index, item.delta_samples, item.evidence),
        )
    )
    clusters: list[list[VerifiedPairChange]] = []
    for change in ordered:
        if (
            not clusters
            or change.canonical_sample - clusters[-1][0].canonical_sample > event_tolerance_samples
        ):
            clusters.append([change])
        else:
            clusters[-1].append(change)
    return tuple(tuple(cluster) for cluster in clusters)


def _cluster_sample(cluster: tuple[VerifiedPairChange, ...]) -> int:
    """Use the lower middle sample so even-sized clusters remain integral."""

    samples = sorted(change.canonical_sample for change in cluster)
    return samples[(len(samples) - 1) // 2]


def _nearby_slave_evidence(
    evidence: tuple[SlaveSlaveEvidence, ...], *, canonical_sample: int,
    event_tolerance_samples: int,
) -> tuple[SlaveSlaveEvidence, ...]:
    return tuple(
        sorted(
            (
                item
                for item in evidence
                if item.verified
                and abs(item.canonical_sample - canonical_sample) <= event_tolerance_samples
            ),
            key=lambda item: (
                item.canonical_sample,
                item.first_device_index,
                item.second_device_index,
                item.second_minus_first_delta_samples,
                item.evidence,
            ),
        )
    )


def _relation_delta_for_device(item: SlaveSlaveEvidence, device_index: int) -> int | None:
    """Return ``device source - other source`` for a slave/slave result."""

    if item.first_device_index == device_index:
        return -item.second_minus_first_delta_samples
    if item.second_device_index == device_index:
        return item.second_minus_first_delta_samples
    return None


def _stable_slave_graph(
    evidence: tuple[SlaveSlaveEvidence, ...], slaves: set[int]) -> bool:
    """Require a connected graph of verified zero-change slave/slave edges."""

    if len(slaves) < 2:
        return False
    connected = {min(slaves)}
    while True:
        expanded = set(connected)
        for item in evidence:
            if item.second_minus_first_delta_samples != 0:
                continue
            if item.first_device_index in connected and item.second_device_index in slaves:
                expanded.add(item.second_device_index)
            if item.second_device_index in connected and item.first_device_index in slaves:
                expanded.add(item.first_device_index)
        if expanded == connected:
            return connected == slaves
        connected = expanded


def _unresolved(
    *, canonical_sample: int, cluster: tuple[VerifiedPairChange, ...], reason: str
) -> AttributionDecision:
    return AttributionDecision(
        kind="unresolved",
        canonical_sample=canonical_sample,
        device_indices=tuple(sorted({change.slave_device_index for change in cluster})),
        delta_samples=cluster[0].delta_samples,
        evidence=reason,
    )


def attribute_targeted_events(
    pair_changes: Iterable[VerifiedPairChange],
    *,
    device_count: int,
    master_device_index: int,
    observed_slave_indices: Iterable[int],
    slave_slave_evidence: Iterable[SlaveSlaveEvidence] = (),
    event_tolerance_samples: int = 0,
    magnitude_tolerance_samples: int = 0,
) -> tuple[AttributionDecision, ...]:
    """Classify local transitions without fabricating attribution from one pair.

    ``observed_slave_indices`` must name every slave whose master/slave
    relationship was successfully checked in the local window, including those
    with no change.  A two-device setup, partial coverage, conflicting signs,
    unequal common steps, and missing required slave/slave confirmation remain
    unresolved.  Returned decisions are sorted by canonical sample and device
    indices, independent of input iteration order.

    Pipeline hook: convert a ``slave`` or ``master`` decision into a local
    missing interval after boundary localization; convert ``extra_source``
    only after a post-event segment establishes its raw source coordinates.
    ``unresolved`` must instead end/reacquire segments and invalidate the
    unsupported interval, never propagate its measured step.
    """

    expected_slaves = _validate_configuration(
        device_count=device_count,
        master_device_index=master_device_index,
        event_tolerance_samples=event_tolerance_samples,
        magnitude_tolerance_samples=magnitude_tolerance_samples,
    )
    observed = {int(index) for index in observed_slave_indices}
    if any(index not in expected_slaves for index in observed):
        raise ValueError("observed_slave_indices must contain only non-master devices")
    changes = tuple(pair_changes)
    evidence = tuple(slave_slave_evidence)
    for change in changes:
        if change.slave_device_index not in expected_slaves:
            raise ValueError("pair change must name a non-master device")
    for item in evidence:
        if (
            item.first_device_index not in expected_slaves
            or item.second_device_index not in expected_slaves
        ):
            raise ValueError("slave/slave evidence must name only non-master devices")

    decisions: list[AttributionDecision] = []
    for cluster in _cluster_changes(changes, event_tolerance_samples=event_tolerance_samples):
        sample = _cluster_sample(cluster)
        nearby = _nearby_slave_evidence(
            evidence,
            canonical_sample=sample,
            event_tolerance_samples=event_tolerance_samples,
        )
        changed_by_slave: dict[int, VerifiedPairChange] = {}
        duplicate_slave = False
        for change in cluster:
            if change.slave_device_index in changed_by_slave:
                duplicate_slave = True
            changed_by_slave[change.slave_device_index] = change

        if device_count < 3:
            decisions.append(_unresolved(
                canonical_sample=sample,
                cluster=cluster,
                reason="two-device relative evidence cannot attribute a device-local transition",
            ))
            continue
        if observed != expected_slaves:
            decisions.append(_unresolved(
                canonical_sample=sample,
                cluster=cluster,
                reason="master/slave local coverage is incomplete",
            ))
            continue
        if duplicate_slave:
            decisions.append(_unresolved(
                canonical_sample=sample,
                cluster=cluster,
                reason="multiple pair changes for one slave at one candidate transition",
            ))
            continue

        values = tuple(change.delta_samples for change in changed_by_slave.values())
        if len(changed_by_slave) == 1:
            change = next(iter(changed_by_slave.values()))
            related = tuple(
                item for item in nearby
                if change.slave_device_index in (item.first_device_index, item.second_device_index)
            )
            unrelated_nonzero = any(
                item.second_minus_first_delta_samples != 0
                and change.slave_device_index not in (item.first_device_index, item.second_device_index)
                for item in nearby
            )
            consistent = all(
                _relation_delta_for_device(item, change.slave_device_index) == change.delta_samples
                for item in related
            )
            if (
                change.delta_samples < 0
                and related
                and not unrelated_nonzero
                and consistent
            ):
                decisions.append(AttributionDecision(
                    kind="slave",
                    canonical_sample=sample,
                    device_indices=(change.slave_device_index,),
                    delta_samples=change.delta_samples,
                    evidence="single verified negative master/slave change; local relationships support slave loss",
                ))
                continue
            if (
                change.delta_samples > 0
                and related
                and not unrelated_nonzero
                and consistent
            ):
                decisions.append(AttributionDecision(
                    kind="extra_source",
                    canonical_sample=sample,
                    device_indices=(change.slave_device_index,),
                    delta_samples=change.delta_samples,
                    evidence="positive single-device change confirmed by targeted slave/slave evidence",
                ))
                continue
            decisions.append(_unresolved(
                canonical_sample=sample,
                cluster=cluster,
                reason="single-device transition lacks consistent targeted attribution evidence",
            ))
            continue

        all_positive = all(value > 0 for value in values)
        equal_magnitude = (
            max(values) - min(values) <= magnitude_tolerance_samples
            if values else False
        )
        if (
            set(changed_by_slave) == expected_slaves
            and all_positive
            and equal_magnitude
            and _stable_slave_graph(nearby, expected_slaves)
        ):
            representative = sorted(values)[(len(values) - 1) // 2]
            decisions.append(AttributionDecision(
                kind="master",
                canonical_sample=sample,
                device_indices=(master_device_index,),
                delta_samples=representative,
                evidence="common positive master/slave changes with stable targeted slave/slave graph",
            ))
            continue
        decisions.append(_unresolved(
            canonical_sample=sample,
            cluster=cluster,
            reason="conflicting, unequal, or multi-device-ambiguous local transition",
        ))

    return tuple(
        sorted(
            decisions,
            key=lambda item: (item.canonical_sample, item.device_indices, item.kind, item.delta_samples),
        )
    )
