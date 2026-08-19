from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from wild_preprocess.models import Recording, SyncOptions
from wild_preprocess.sync.observe import observe_pair


def _recording(root: Path, name: str, samples: int, fs: int) -> Recording:
    return Recording(
        folder=root / name,
        amplifier_file=root / f"{name}.dat",
        analog_file=root / f"{name}_analog.dat",
        ce_params_file=root / f"{name}.bin",
        device_name=name,
        recording_name="recording",
        fs=fs,
        n_channels=1,
        n_samples=samples,
        analog_channels=1,
        analog_samples=1,
    )


def _observe(master_values: np.ndarray, slave_values: np.ndarray, options: SyncOptions):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        master_path = root / "master.f32"
        slave_path = root / "slave.f32"
        master_values.astype("<f4").tofile(master_path)
        slave_values.astype("<f4").tofile(slave_path)
        return observe_pair(
            _recording(root, "master", master_values.size, 1_000),
            _recording(root, "slave", slave_values.size, 1_000),
            master_path,
            slave_path,
            options,
        )


def _options(*, ceiling_seconds: float) -> SyncOptions:
    return SyncOptions(
        initial_start_seconds=1.0,
        initial_duration_seconds=2.0,
        initial_max_lag_seconds=0.5,
        window_seconds=4.0,
        step_seconds=2.0,
        tracking_max_lag_samples=40,
        peak_exclusion_samples=8,
        coarse_feature_rate_hz=100.0,
        coarse_reacquisition_max_lag_seconds=ceiling_seconds,
        coarse_reacquisition_growth_factor=2.0,
        chunk_seconds=0.25,
    )


class CoarseReacquisitionTests(unittest.TestCase):
    def test_reacquires_a_verified_offset_larger_than_one_second(self) -> None:
        rng = np.random.default_rng(1101)
        master = rng.normal(size=18_000).astype(np.float32)
        loss = 2_200
        slave = np.concatenate((master[:6_000], master[6_000 + loss :]))

        result = _observe(master, slave, _options(ceiling_seconds=3.0))

        reacquired = [
            observation
            for observation in result.observations
            if observation.accepted and observation.search_mode == "coarse_reacquisition"
        ]
        self.assertTrue(reacquired, [(item.search_mode, item.rejection_reason) for item in result.observations])
        self.assertTrue(
            any(abs(item.observed_offset_samples + loss) <= 1 for item in reacquired),
            [(item.center_time_sec, item.observed_offset_samples) for item in reacquired],
        )

    def test_ceiling_exhaustion_is_unsupported_not_a_false_lag(self) -> None:
        rng = np.random.default_rng(1102)
        master = rng.normal(size=20_000).astype(np.float32)
        loss = 4_000
        slave = np.concatenate((master[:6_000], master[6_000 + loss :]))

        result = _observe(master, slave, _options(ceiling_seconds=2.0))

        exhausted = [
            observation
            for observation in result.observations
            if "search ceiling" in observation.rejection_reason
        ]
        self.assertTrue(exhausted)
        self.assertTrue(all(not observation.accepted for observation in exhausted))
        self.assertTrue(all(abs(observation.observed_offset_samples) < loss for observation in exhausted))

    def test_clean_alignment_stays_on_the_narrow_full_rate_path(self) -> None:
        rng = np.random.default_rng(1103)
        master = rng.normal(size=16_000).astype(np.float32)

        result = _observe(master, master.copy(), _options(ceiling_seconds=3.0))

        self.assertTrue(result.observations)
        self.assertTrue(all(observation.accepted for observation in result.observations))
        self.assertTrue(all(observation.search_mode in {"narrow", "endpoint_probe"} for observation in result.observations))
        self.assertTrue(all(abs(observation.observed_offset_samples) <= 1 for observation in result.observations))


if __name__ == "__main__":
    unittest.main()
