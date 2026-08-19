from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
import struct
import sys

import numpy as np
import pytest
from scipy.io import loadmat


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from wild_preprocess.models import SyncOptions
from wild_preprocess.pipeline import run_multidevice_sync
from wild_preprocess.analog.integrity import AnalogIntegrityEvent, scan_analog_integrity
from wild_preprocess.analog.segments import build_event_driven_analog_segments


def _recording(folder: Path, signal: np.ndarray, *, source_offset: int) -> Path:
    folder.mkdir(parents=True)
    source = np.zeros_like(signal)
    source[source_offset:] = signal[: signal.size - source_offset]
    np.column_stack([source + lane for lane in range(64)]).astype("<i2").tofile(
        folder / "amplifier.dat"
    )
    analog_rows = signal.size // 16
    analog = np.zeros((analog_rows, 16), dtype="<i2")
    analog[:, 0] = (np.arange(analog_rows) // 100 % 2).astype(np.int16)
    analog[:, 11] = np.arange(analog_rows, dtype=np.uint16).view(np.int16)
    held = (np.arange(analog_rows) // 6 + 1).astype(np.int16)
    for lane in range(1, 10):
        analog[:, lane] = held * (lane + 1)
    analog.tofile(folder / "analogin.dat")
    header = bytearray(512)
    struct.pack_into("<I", header, 0, 20_000)
    struct.pack_into("<I", header, 8, 64)
    (folder / "CE_params.bin").write_bytes(header)
    return folder


def test_pipeline_publishes_analog_authority_and_opt_in_imu(tmp_path: Path) -> None:
    rng = np.random.default_rng(20260812)
    signal = np.rint(rng.normal(scale=300.0, size=160_000)).astype(np.int16)
    folders = [
        _recording(tmp_path / "master" / "recording", signal, source_offset=0),
        _recording(tmp_path / "slave" / "recording", signal, source_offset=17),
    ]
    output = tmp_path / "output"
    result = run_multidevice_sync(
        folders,
        master_index=0,
        output_folder=output,
        merge=True,
        process_imu=True,
        options=SyncOptions(
            initial_start_seconds=0.5,
            initial_duration_seconds=1.0,
            initial_max_lag_seconds=0.01,
            window_seconds=1.0,
            step_seconds=0.5,
            tracking_max_lag_samples=40,
            min_peak_margin_fraction=0.005,
            chunk_seconds=0.5,
        ),
    )

    assert result.outputs["merge_status"] in {"OK", "WARN"}
    assert result.outputs["analog_status"] == "OK"
    assert result.outputs["imu_status"] == "OK"
    manifest = json.loads((output / "wild_preprocess_run.json").read_text(encoding="utf-8"))
    assert manifest["merge"]["analog_status"] == "OK"
    assert manifest["analog_status"] == "OK"
    assert manifest["merge"]["analog_validity_samples_file"] == "valid_analog_samples.dat"
    analog_rows = int(manifest["merge"]["analog_samples"])
    validity = np.fromfile(output / "valid_analog_samples.dat", dtype=np.uint8).reshape(
        analog_rows, 2
    )
    assert np.all(validity[:, 0] == 1)
    assert np.count_nonzero(validity[:, 1] == 0) <= 1
    assert (output / "IMU.mat").is_file()
    loaded = loadmat(output / "IMU.mat", struct_as_record=False, squeeze_me=True)["IMU"]
    assert loaded.masterIndex == 1
    assert loaded.fusionStatus == "OK"
    assert loaded.fusionMethod == "matlab_r2024b_ahrsfilter_defaults_per_valid_run"
    assert manifest["imu_status"] == "OK"
    assert (
        manifest["imu"]["fusion_method"]
        == "matlab_r2024b_ahrsfilter_defaults_per_valid_run"
    )
    assert manifest["imu"]["fusion_status"] == "OK"
    assert loaded.device[0].fusionData.quaternion.shape[1] == 4
    assert loaded.device[0].fusionData.orientation.shape[:2] == (3, 3)
    assert manifest["imu"]["source_domain"] == "canonical_merged_analog"
    assert manifest["imu"]["source_analog_file"] == "analogin.dat"
    assert manifest["imu"]["source_validity_file"] == "valid_analog_samples.dat"
    assert all(
        device["fusion_method"]
        == "matlab_r2024b_ahrsfilter_defaults_per_valid_run"
        for device in manifest["imu"]["devices"]
    )
    assert all(device["fusion_valid_count"] > 0 for device in manifest["imu"]["devices"])
    assert manifest["merge"]["analog_channel_device_order"] == [
        {
            "block": 0,
            "device_index": 1,
            "device_name": "master",
            "channel_start_zero_based": 0,
            "channel_count": 16,
        },
        {
            "block": 1,
            "device_index": 2,
            "device_name": "slave",
            "channel_start_zero_based": 16,
            "channel_count": 16,
        },
    ]
    assert [
        item["device_index"]
        for item in manifest["merge"]["analog_validity_device_order"]
    ] == [1, 2]
    timeline_hashes = {
        str(item["device_index"]): item["mapping_hash"]
        for item in manifest["merge"]["analog_timelines"]
    }
    assert manifest["imu"]["analog_timeline_mapping_hashes"] == timeline_hashes
    assert loaded.sourceAnalogFile == "analogin.dat"
    assert np.asarray(loaded.sourceAnalogFiles).size == 0


def test_pipeline_rejects_missing_master_analog_mapping_transactionally(
    tmp_path: Path, monkeypatch
) -> None:
    rng = np.random.default_rng(20260813)
    signal = np.rint(rng.normal(scale=300.0, size=160_000)).astype(np.int16)
    folders = [
        _recording(tmp_path / "master" / "recording", signal, source_offset=0),
        _recording(tmp_path / "slave" / "recording", signal, source_offset=17),
    ]
    original_scan = scan_analog_integrity

    def scan_with_unavailable_master(path, *args, **kwargs):
        result = original_scan(path, *args, **kwargs)
        if kwargs.get("device_index") != 1:
            return result
        return replace(
            result,
            events=(
                AnalogIntegrityEvent(
                    kind="unresolved",
                    raw_start_row=0,
                    raw_end_row=result.metrics.row_count,
                    tick_start=0,
                    tick_end=result.metrics.row_count,
                    affected_lanes=tuple(range(16)),
                    displacement_rows=None,
                    confidence="unresolved",
                    evidence="synthetic unavailable master mapping",
                    device_index=1,
                ),
            ),
        )

    monkeypatch.setattr(
        "wild_preprocess.pipeline.scan_analog_integrity", scan_with_unavailable_master
    )
    output = tmp_path / "output"
    output.mkdir()
    previous = b"previous canonical amplifier"
    (output / "amplifier.dat").write_bytes(previous)
    with pytest.raises(ValueError, match="master analog mapping"):
        run_multidevice_sync(
            folders,
            master_index=0,
            output_folder=output,
            merge=True,
            overwrite=True,
            options=SyncOptions(
                initial_start_seconds=0.5,
                initial_duration_seconds=1.0,
                initial_max_lag_seconds=0.01,
                window_seconds=1.0,
                step_seconds=0.5,
                tracking_max_lag_samples=40,
                min_peak_margin_fraction=0.005,
                chunk_seconds=0.5,
            ),
        )
    assert (output / "amplifier.dat").read_bytes() == previous
    assert not (output / "valid_analog_samples.dat").exists()


def test_requested_imu_structural_failure_rolls_back_transaction(
    tmp_path: Path, monkeypatch
) -> None:
    rng = np.random.default_rng(20260814)
    signal = np.rint(rng.normal(scale=300.0, size=160_000)).astype(np.int16)
    folders = [
        _recording(tmp_path / "master" / "recording", signal, source_offset=0),
        _recording(tmp_path / "slave" / "recording", signal, source_offset=17),
    ]

    def reject_malformed_staged_pair(*args, **kwargs):
        raise ValueError("synthetic malformed staged analog/validity structure")

    monkeypatch.setattr(
            "wild_preprocess.pipeline.build_imu_from_merged",
        reject_malformed_staged_pair,
    )
    output = tmp_path / "output"
    output.mkdir()
    previous = b"previous canonical amplifier"
    (output / "amplifier.dat").write_bytes(previous)
    with pytest.raises(ValueError, match="malformed staged analog/validity"):
        run_multidevice_sync(
            folders,
            master_index=0,
            output_folder=output,
            merge=True,
            overwrite=True,
            process_imu=True,
            options=SyncOptions(
                initial_start_seconds=0.5,
                initial_duration_seconds=1.0,
                initial_max_lag_seconds=0.01,
                window_seconds=1.0,
                step_seconds=0.5,
                tracking_max_lag_samples=40,
                min_peak_margin_fraction=0.005,
                chunk_seconds=0.5,
            ),
        )
    assert (output / "amplifier.dat").read_bytes() == previous
    assert not (output / "IMU.mat").exists()
    assert not (output / "valid_analog_samples.dat").exists()


def test_unavailable_slave_analog_is_warn_and_zero_filled(
    tmp_path: Path, monkeypatch
) -> None:
    rng = np.random.default_rng(20260815)
    signal = np.rint(rng.normal(scale=300.0, size=160_000)).astype(np.int16)
    folders = [
        _recording(tmp_path / "master" / "recording", signal, source_offset=0),
        _recording(tmp_path / "slave" / "recording", signal, source_offset=17),
    ]

    def omit_slave_support(prior, **kwargs):
        segments = build_event_driven_analog_segments(prior, **kwargs)
        if prior.device_index == 1:
            return segments
        return tuple(
            replace(segment, publishable=False, confidence="unresolved")
            for segment in segments
        )

    monkeypatch.setattr(
        "wild_preprocess.sync.merge.build_event_driven_analog_segments",
        omit_slave_support,
    )
    output = tmp_path / "output"
    result = run_multidevice_sync(
        folders,
        master_index=0,
        output_folder=output,
        merge=True,
        options=SyncOptions(
            initial_start_seconds=0.5,
            initial_duration_seconds=1.0,
            initial_max_lag_seconds=0.01,
            window_seconds=1.0,
            step_seconds=0.5,
            tracking_max_lag_samples=40,
            min_peak_margin_fraction=0.005,
            chunk_seconds=0.5,
        ),
    )
    assert result.outputs["analog_status"] == "WARN"
    manifest = json.loads((output / "wild_preprocess_run.json").read_text(encoding="utf-8"))
    slave_timeline = next(
        item for item in manifest["merge"]["analog_timelines"] if item["device_index"] == 2
    )
    assert slave_timeline["status"] == "WARN"
    assert "no publishable analog source support" in slave_timeline["warnings"]
    rows = int(manifest["merge"]["analog_samples"])
    validity = np.fromfile(output / "valid_analog_samples.dat", dtype=np.uint8).reshape(rows, 2)
    analog = np.memmap(output / "analogin.dat", dtype="<i2", mode="r", shape=(rows, 32))
    assert np.all(validity[:, 1] == 0)
    assert np.all(analog[:, 16:32] == 0)


def test_clipped_slave_analog_tail_is_warn(
    tmp_path: Path, monkeypatch
) -> None:
    rng = np.random.default_rng(20260816)
    signal = np.rint(rng.normal(scale=300.0, size=160_000)).astype(np.int16)
    folders = [
        _recording(tmp_path / "master" / "recording", signal, source_offset=0),
        _recording(tmp_path / "slave" / "recording", signal, source_offset=17),
    ]

    def clip_slave_tail(prior, **kwargs):
        segments = build_event_driven_analog_segments(prior, **kwargs)
        if prior.device_index == 1:
            return segments
        last = segments[-1]
        clipped_end = last.canonical_end_row - 100
        last_raw = last.raw_scale * (clipped_end - 1) + last.raw_intercept_rows
        clipped_anchor = replace(
            last.anchors[-1], canonical_row=clipped_end - 1, raw_row=last_raw
        )
        return (
            *segments[:-1],
            replace(
                last,
                canonical_end_row=clipped_end,
                raw_end_row=math.ceil(last_raw) + 1,
                anchors=(last.anchors[0], clipped_anchor),
            ),
        )

    monkeypatch.setattr(
        "wild_preprocess.sync.merge.build_event_driven_analog_segments",
        clip_slave_tail,
    )
    output = tmp_path / "output"
    result = run_multidevice_sync(
        folders,
        master_index=0,
        output_folder=output,
        merge=True,
        options=SyncOptions(
            initial_start_seconds=0.5,
            initial_duration_seconds=1.0,
            initial_max_lag_seconds=0.01,
            window_seconds=1.0,
            step_seconds=0.5,
            tracking_max_lag_samples=40,
            min_peak_margin_fraction=0.005,
            chunk_seconds=0.5,
        ),
    )
    assert result.outputs["analog_status"] == "WARN"
    manifest = json.loads((output / "wild_preprocess_run.json").read_text(encoding="utf-8"))
    slave_timeline = next(
        item for item in manifest["merge"]["analog_timelines"] if item["device_index"] == 2
    )
    assert slave_timeline["status"] == "WARN"
    assert any("covers" in warning for warning in slave_timeline["warnings"])
