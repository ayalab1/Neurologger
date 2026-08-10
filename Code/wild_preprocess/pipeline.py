from __future__ import annotations

import os
import json
import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

from .binary_io import close_memmap, read_ce_params_metadata, recordings_from_folders
from .models import PipelineResult, SyncOptions, SyncPairResult
from .report import save_pair_figure, write_qc_outputs
from .sync.features import build_common_mode_feature, feature_memmap
from .sync.gaps import (
    canonicalize_master_sample,
    detect_isolated_offset_crop,
    detect_unconfirmed_terminal_crop,
    gap_summary,
    infer_device_gaps,
    localize_relative_offset_step,
    verify_isolated_offset_alias,
)
from .sync.infer import fit_affine_sync_model
from .sync.merge import (
    add_postmerge_validation_to_merge_mat,
    add_traceability_to_merge_mat,
    merge_recordings,
)
from .sync.observe import observe_pair
from .sync.postmerge import PostMergeValidationResult, validate_staged_merge
from .sync.validate import validate_pair
from .version import PIPELINE_ALGORITHM_VERSION, RUN_MANIFEST_SCHEMA_VERSION, SYNC_ALGORITHM_VERSION
from .pc_time import (
    CE64_RAW_MISC_LAYOUT,
    PcTimeOptions,
    collect_packed_updates,
    fit_robust_pc_time_model,
    pc_time_qc_payload,
    resolve_recording_start_ms,
    validate_pc_time_interval,
    write_interval_pc_time,
    write_pc_time_qc_json,
    write_pc_time_summary_png,
)


ProgressCallback = Callable[[str, float], None]


class _StagedMergeRejected(RuntimeError):
    """A validation gate rejected a private staging directory."""

    def __init__(self, validation: PostMergeValidationResult):
        super().__init__(validation.message)
        self.validation = validation


def _overall_status(pairs: list[SyncPairResult]) -> str:
    statuses = {pair.status for pair in pairs}
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "OK"


def _manifest_warnings(pairs: list[SyncPairResult]) -> list[dict[str, object]]:
    """Return only component warnings; normal OK diagnostics stay in sync QC."""

    return [
        {"slave_index": pair.slave_index, "status": pair.status, "message": pair.message}
        for pair in pairs
        if pair.status != "OK"
    ]


def _write_attempt_qc(result: PipelineResult, attempt_folder: Path) -> dict[str, str]:
    """Write failure QC beside its figures, never in the canonical folder."""

    canonical_folder = result.output_folder
    result.output_folder = attempt_folder
    try:
        return write_qc_outputs(result)
    finally:
        result.output_folder = canonical_folder


def _validate_selected_inputs(
    recordings: list,
    *,
    probe_indices: list[int] | None,
    recording_start_anchors: list[dict[str, object]] | None,
) -> list[int]:
    folders = [recording.folder.resolve() for recording in recordings]
    if len(set(folders)) != len(folders):
        raise ValueError("Selected recording folders must be unique.")
    probes = list(range(1, len(recordings) + 1)) if probe_indices is None else [int(value) for value in probe_indices]
    if sorted(probes) != list(range(1, len(recordings) + 1)):
        raise ValueError("Selected probe indices must be unique and exactly 1..N.")
    if recording_start_anchors is not None:
        if len(recording_start_anchors) != len(recordings):
            raise ValueError("recording_start_anchors must provide one anchor for each selected recording.")
        values: list[int] = []
        for index, anchor in enumerate(recording_start_anchors, start=1):
            if not isinstance(anchor, dict) or "milliseconds_since_midnight" not in anchor:
                raise ValueError(f"recording start anchor {index} is missing milliseconds_since_midnight")
            value = int(anchor["milliseconds_since_midnight"])
            if not 0 <= value < 86_400_000:
                raise ValueError(f"recording start anchor {index} is outside one day")
            values.append(value)
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                circular_difference = abs(
                    (values[right] - values[left] + 43_200_000) % 86_400_000 - 43_200_000
                )
                if circular_difference > 30_000:
                    raise ValueError(
                        f"Selected recording start anchors differ by {circular_difference / 1000:.3f}s "
                        f"(devices {left + 1} and {right + 1}; limit 30s)."
                    )
    return probes


def _input_provenance(
    recordings: list,
    probe_indices: list[int],
    recording_start_anchors: list[dict[str, object]] | None,
) -> list[dict[str, object]]:
    anchors = recording_start_anchors or [
        {
            "milliseconds_since_midnight": None,
            "source": None,
            "recording_date": None,
            "recording_date_source": None,
        }
        for _ in recordings
    ]
    rows: list[dict[str, object]] = []
    for index, (recording, probe, anchor) in enumerate(zip(recordings, probe_indices, anchors), start=1):
        rows.append(
            {
                "device_order": index,
                "probe_index": probe,
                "folder": str(recording.folder),
                "device_name": recording.device_name,
                "recording_name": recording.recording_name,
                "files": {
                    "amplifier.dat": {"path": str(recording.amplifier_file), "bytes": recording.amplifier_file.stat().st_size},
                    "analogin.dat": {"path": str(recording.analog_file), "bytes": recording.analog_file.stat().st_size},
                    "CE_params.bin": {"path": str(recording.ce_params_file), "bytes": recording.ce_params_file.stat().st_size},
                },
                "ce_header": read_ce_params_metadata(recording.ce_params_file),
                "recording_start_anchor": anchor,
            }
        )
    return rows


def run_multidevice_sync(
    device_folders: Iterable[Path],
    *,
    master_index: int,
    output_folder: Path,
    overwrite: bool = False,
    merge: bool = True,
    options: SyncOptions | None = None,
    progress: ProgressCallback | None = None,
    native_pc_time: bool = False,
    recording_start_ms: int | None = None,
    validate_postmerge: bool | None = None,
    pc_time_options: PcTimeOptions | None = None,
    probe_indices: list[int] | None = None,
    recording_start_anchors: list[dict[str, object]] | None = None,
) -> PipelineResult:
    """Estimate multi-device timing and optionally write merged streams.

    `master_index` is zero-based in the Python API.  When ``native_pc_time`` is
    enabled this is the complete worker path: sync, private merge staging,
    post-merge validation, native PC-time generation, and atomic publication.
    The default remains false for callers that previously used this as a
    sync/merge-only library function.
    """

    options = options or SyncOptions()
    pc_time_options = pc_time_options or PcTimeOptions()
    if validate_postmerge is None:
        # Preserve the previous library-level sync/merge behaviour.  The
        # production worker always enables native PC time and therefore the
        # publication gate; direct legacy callers can opt in explicitly.
        validate_postmerge = native_pc_time
    recordings = recordings_from_folders(device_folders)
    probes = _validate_selected_inputs(
        recordings,
        probe_indices=probe_indices,
        recording_start_anchors=recording_start_anchors,
    )
    input_provenance = _input_provenance(recordings, probes, recording_start_anchors)
    if master_index < 0 or master_index >= len(recordings):
        raise ValueError(f"master_index {master_index} is outside {len(recordings)} recordings")
    output_folder = Path(output_folder).resolve()
    if any(output_folder == recording.folder for recording in recordings):
        raise ValueError("The merged output folder must not be one of the raw recording folders.")
    output_folder.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    attempt_folder = output_folder / f".wild_sync_attempt_{run_id}"
    protected_outputs = [
        output_folder / "wild_multilogger_sync_qc.tsv",
        output_folder / "wild_multilogger_sync_qc.json",
        output_folder / "wild_multilogger_sync_qc.mat",
    ]
    if merge:
        protected_outputs.extend(
            [
                output_folder / "amplifier.dat",
                output_folder / "analogin.dat",
                output_folder / "time.dat",
                output_folder / "wild_multilogger_mergeInfo.mat",
            ]
        )
    existing = [path for path in protected_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output exists; enable overwrite to regenerate: " + ", ".join(str(path) for path in existing)
        )
    attempt_folder.mkdir(parents=True, exist_ok=False)

    pairs: list[SyncPairResult] = []
    cache_root_text = os.environ.get("WILD_SYNC_CACHE_DIR", "").strip()
    cache_root = Path(cache_root_text).expanduser().resolve() if cache_root_text else None
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wild_sync_", dir=cache_root) as temporary:
        temporary_path = Path(temporary)
        feature_paths = [
            temporary_path / f"common_mode_{index:02d}.f32"
            for index in range(len(recordings))
        ]
        feature_percent = [0.0] * len(recordings)
        feature_lock = Lock()

        def build_feature(index: int) -> Path:
            def feature_progress(stage: str, percent: float) -> None:
                if progress is None:
                    return
                with feature_lock:
                    feature_percent[index] = percent
                    progress("build_features", sum(feature_percent) / len(feature_percent))

            return build_common_mode_feature(
                recordings[index],
                feature_paths[index],
                highpass_hz=options.highpass_hz,
                chunk_seconds=options.chunk_seconds,
                progress=feature_progress,
            )

        feature_worker_count = max(
            1, min(int(options.max_parallel_workers), len(recordings))
        )
        if feature_worker_count == 1:
            for index in range(len(recordings)):
                build_feature(index)
        else:
            with ThreadPoolExecutor(
                max_workers=feature_worker_count,
                thread_name_prefix="wild-sync-feature",
            ) as executor:
                futures = [executor.submit(build_feature, index) for index in range(len(recordings))]
                for future in as_completed(futures):
                    future.result()

        slave_indices = [index for index in range(len(recordings)) if index != master_index]
        master = recordings[master_index]

        def observe_slave(slave_index: int):
            return observe_pair(
                master,
                recordings[slave_index],
                feature_paths[master_index],
                feature_paths[slave_index],
                options,
                progress=None,
            )

        observed_by_slave: dict[int, object] = {}
        worker_count = max(1, min(int(options.max_parallel_workers), len(slave_indices)))
        if worker_count == 1:
            for completed, slave_index in enumerate(slave_indices, start=1):
                observed_by_slave[slave_index] = observe_slave(slave_index)
                if progress is not None:
                    progress("sync_pairs", 100.0 * completed / len(slave_indices))
        else:
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="wild-sync-pair") as executor:
                futures = {executor.submit(observe_slave, slave_index): slave_index for slave_index in slave_indices}
                for completed, future in enumerate(as_completed(futures), start=1):
                    slave_index = futures[future]
                    observed_by_slave[slave_index] = future.result()
                    if progress is not None:
                        progress("sync_pairs", 100.0 * completed / len(slave_indices))

        for slave_index in slave_indices:
            slave = recordings[slave_index]
            observed = observed_by_slave[slave_index]
            model = fit_affine_sync_model(observed.observations, master.fs, options=options)
            if model.offset_steps:
                master_feature = feature_memmap(feature_paths[master_index], master.n_samples)
                slave_feature = feature_memmap(feature_paths[slave_index], slave.n_samples)
                try:
                    localized_steps = tuple(
                        localize_relative_offset_step(
                            master_feature,
                            slave_feature,
                            step,
                            fs=master.fs,
                            options=options,
                        )
                        for step in model.offset_steps
                    )
                finally:
                    close_memmap(master_feature)
                    close_memmap(slave_feature)
                model = fit_affine_sync_model(
                    observed.observations,
                    master.fs,
                    options=options,
                    offset_steps=localized_steps,
                )
            isolated_crop = detect_isolated_offset_crop(
                observed.observations,
                model,
                master.fs,
                options,
            )
            isolated_alias_message = ""
            isolated_recheck_evidence = ""
            if isolated_crop is not None:
                master_feature = feature_memmap(feature_paths[master_index], master.n_samples)
                slave_feature = feature_memmap(feature_paths[slave_index], slave.n_samples)
                try:
                    isolated_alias, isolated_recheck_evidence = verify_isolated_offset_alias(
                        master_feature,
                        slave_feature,
                        observed.observations,
                        model,
                        isolated_crop,
                        master.fs,
                        options,
                    )
                finally:
                    close_memmap(master_feature)
                    close_memmap(slave_feature)
                if isolated_alias:
                    isolated_alias_message = (
                        "rejected 1 isolated offset alias after raw recheck; "
                        + isolated_recheck_evidence
                    )
                    for observation_index in isolated_crop[2]:
                        observation = observed.observations[observation_index]
                        observation.accepted = False
                        observation.model_inlier = False
                        observation.rejection_reason = isolated_alias_message
                    model = fit_affine_sync_model(
                        observed.observations,
                        master.fs,
                        options=options,
                        offset_steps=model.offset_steps,
                    )
                    isolated_crop = None
            terminal_crop = detect_unconfirmed_terminal_crop(
                observed.observations,
                model,
                master.fs,
                options,
            )
            terminal_crop_master_sample: int | None = None
            terminal_crop_reason = ""
            validation_observations = observed.observations
            crop_candidates: list[tuple[int, float, tuple[int, ...], str]] = []
            if isolated_crop is not None:
                crop_candidates.append(
                    (
                        *isolated_crop,
                        "isolated nonpersistent offset excursion; "
                        + isolated_recheck_evidence,
                    )
                )
            if terminal_crop is not None:
                crop_candidates.append((*terminal_crop, "terminal offset shift"))
            if crop_candidates:
                (
                    terminal_crop_master_sample,
                    terminal_shift,
                    terminal_indices,
                    crop_kind,
                ) = min(crop_candidates, key=lambda item: item[0])
                terminal_crop_reason = (
                    f"unconfirmed {crop_kind} {terminal_shift:+.1f} samples; "
                    f"output cropped before master sample {terminal_crop_master_sample}"
                )
                crop_time_sec = terminal_crop_master_sample / master.fs
                terminal_index_set = set(terminal_indices)
                for observation_index, observation in enumerate(observed.observations):
                    if (
                        observation_index in terminal_index_set
                        or observation.center_time_sec - options.window_seconds / 2.0
                        >= crop_time_sec
                    ):
                        observation.accepted = False
                        observation.model_inlier = False
                        observation.rejection_reason = terminal_crop_reason
                validation_observations = [
                    observation
                    for observation in observed.observations
                    if observation.center_time_sec + options.window_seconds / 2.0
                    <= crop_time_sec
                ]
                validated_steps = tuple(
                    step
                    for step in model.offset_steps
                    if step.master_sample < terminal_crop_master_sample
                )
                model = fit_affine_sync_model(
                    validation_observations,
                    master.fs,
                    options=options,
                    offset_steps=validated_steps,
                )
            status, message = validate_pair(
                observed.initial,
                validation_observations,
                model,
                options,
            )
            operational_warnings = [
                item for item in (isolated_alias_message, terminal_crop_reason) if item
            ]
            if operational_warnings and status != "FAIL":
                status = "WARN"
                message = "; ".join([message] + operational_warnings)
            validated_start = observed.validated_start_master_sample
            early_horizon = validated_start + round(
                (2.0 * options.window_seconds + options.step_seconds) * master.fs
            )
            for step in model.offset_steps:
                if step.master_sample > early_horizon:
                    continue
                stable_after = [
                    item
                    for item in observed.observations
                    if item.accepted and item.center_time_sec >= step.time_sec
                ]
                persistence = max(1, int(options.gap_persistence_observations))
                if len(stable_after) >= persistence:
                    validated_start = max(
                        validated_start,
                        round(
                            (
                                stable_after[persistence - 1].center_time_sec
                                + options.window_seconds / 2
                            )
                            * master.fs
                        ),
                    )
            safe_slave = "".join(character if character.isalnum() or character in "_.-" else "_" for character in slave.device_name)
            figure_path = attempt_folder / f"wild_multilogger_sync_master_vs_{safe_slave}_{slave.recording_name}_qc.png"
            pair = SyncPairResult(
                master_index=master_index + 1,
                slave_index=slave_index + 1,
                master_folder=str(master.folder),
                slave_folder=str(slave.folder),
                initial_offset_samples=float(observed.initial.lag_samples),
                initial_peak_to_background=observed.initial.peak_to_background,
                initial_peak_margin_fraction=observed.initial.peak_margin_fraction,
                model=model,
                observations=observed.observations,
                status=status,
                message=message,
                figure_file=str(figure_path),
                validated_start_master_sample=validated_start,
                terminal_crop_master_sample=terminal_crop_master_sample,
                terminal_crop_reason=terminal_crop_reason,
            )
            save_pair_figure(pair, observed, figure_path)
            pairs.append(pair)

        device_gaps, unresolved_gap_messages = infer_device_gaps(
            pairs,
            device_count=len(recordings),
            master_index=master_index,
            fs=master.fs,
            options=options,
        )

    result = PipelineResult(
        recordings=recordings,
        master_index=master_index + 1,
        pairs=pairs,
        run_id=run_id,
        status=(
            "FAIL"
            if any(pair.status == "FAIL" for pair in pairs) or unresolved_gap_messages
            else "OK"
        ),
        output_folder=output_folder,
        device_gaps=device_gaps,
        unresolved_gap_messages=unresolved_gap_messages,
    )
    result.outputs.update(
        {
            "sync_status": result.status,
            "merge_status": "NOT_RUN",
            "pc_time_status": "NOT_RUN",
            "overall_status": "FAIL" if result.status == "FAIL" else "MERGE_ONLY",
        }
    )
    if result.status == "FAIL":
        result.outputs.update(_write_attempt_qc(result, attempt_folder))
        result.outputs["attempt_folder"] = str(attempt_folder)
        return result
    if not merge:
        result.outputs.update(_write_attempt_qc(result, attempt_folder))
        result.outputs["attempt_folder"] = str(attempt_folder)
        return result

    pair_models = {pair.slave_index - 1: pair.model for pair in pairs}
    figure_names = {Path(pair.figure_file).name for pair in pairs}
    extra_managed_names = {
        "wild_multilogger_sync_qc.tsv",
        "wild_multilogger_sync_qc.json",
        "wild_multilogger_sync_qc.mat",
        "wild_multilogger_postmerge_qc.json",
        "wild_preprocess_run.json",
    } | figure_names
    if native_pc_time:
        extra_managed_names.update({"pc_time.dat", "pc_time_qc.json", "pc_time_fit_summary.png"})
    component = {"merge_status": "NOT_RUN", "pc_time_status": "NOT_RUN", "overall_status": "FAIL"}

    def write_staged_sync_qc(staging: Path) -> dict[str, str]:
        """Reuse the legacy-compatible reporter without publishing early."""

        canonical_folder = result.output_folder
        result.output_folder = staging
        try:
            staged = write_qc_outputs(result, mode="multiMerge")
        finally:
            result.output_folder = canonical_folder
        # The JSON is an operator-facing file, so retain the canonical session
        # location even though it was physically generated in staging.
        json_path = Path(staged["qc_json"])
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        payload["output_folder"] = str(output_folder)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return staged

    def stage_callback(staging: Path, staged_outputs: dict[str, str]) -> None:
        # The pair figures were intentionally withheld from the session folder
        # until every merge gate has passed.
        for pair in pairs:
            source = Path(pair.figure_file)
            destination = staging / source.name
            shutil.copy2(source, destination)

        merge_info = json.loads(Path(staged_outputs["merge_json"]).read_text(encoding="utf-8"))
        merge_info["probe_indices"] = probes
        merge_info["recording_start_anchors"] = [
            item["recording_start_anchor"] for item in input_provenance
        ]
        postmerge = validate_staged_merge(
            Path(staged_outputs["amplifier"]),
            recordings,
            master_index,
            n_output_samples=int(merge_info["n_samples"]),
            window_seconds=10.0,
            max_lag_samples=options.tracking_max_lag_samples,
            max_allowed_abs_lag_samples=4,
            min_peak_correlation=options.min_peak_correlation,
            peak_exclusion_samples=options.peak_exclusion_samples,
            device_gaps=result.device_gaps,
            canonical_start_sample=int(merge_info["common_start_master_sample"]),
        ) if validate_postmerge else None
        if postmerge is None:
            serialized_postmerge = {
                "status": "NOT_RUN",
                "message": "disabled for compatibility caller",
            }
        elif postmerge.passed:
            serialized_postmerge = postmerge.to_dict()
            # Validation read the private staged file. On commit that same
            # artifact is moved byte-for-byte to this canonical destination.
            serialized_postmerge["amplifier_path"] = str(output_folder / "amplifier.dat")
        else:
            serialized_postmerge = postmerge.to_dict()
            serialized_postmerge["validated_staging_amplifier_path"] = serialized_postmerge.pop(
                "amplifier_path"
            )
            serialized_postmerge["staged_artifact_retained"] = False
        merge_info["postmerge_validation"] = serialized_postmerge
        Path(staged_outputs["merge_json"]).write_text(json.dumps(merge_info, indent=2), encoding="utf-8")
        add_traceability_to_merge_mat(
            Path(staged_outputs["merge_mat"]),
            probe_indices=probes,
            recording_start_anchors=merge_info["recording_start_anchors"],
        )
        # A rejected stage is discarded, so its merge MAT is never published;
        # keep the useful failure diagnostic in the attempt JSON instead.
        if postmerge is None or postmerge.passed:
            add_postmerge_validation_to_merge_mat(
                Path(staged_outputs["merge_mat"]), merge_info["postmerge_validation"]
            )
        (staging / "wild_multilogger_postmerge_qc.json").write_text(
            json.dumps(serialized_postmerge, indent=2),
            encoding="utf-8",
        )
        result.outputs["postmerge_qc"] = str(output_folder / "wild_multilogger_postmerge_qc.json")
        if postmerge is not None and not postmerge.passed:
            # Keep an inspectable failed attempt without touching canonical QC.
            (attempt_folder / "wild_multilogger_postmerge_qc.json").write_text(
                json.dumps(serialized_postmerge, indent=2), encoding="utf-8"
            )
            result.status = "FAIL"
            result.outputs.update(_write_attempt_qc(result, attempt_folder))
            raise _StagedMergeRejected(postmerge)

        # Only now do report paths point to canonical files.  On rejection,
        # they retain the existing attempt-folder locations.
        for pair in pairs:
            pair.figure_file = str(output_folder / Path(pair.figure_file).name)

        component["merge_status"] = "OK"
        component["pc_time_status"] = "NOT_RUN"
        component["overall_status"] = "MERGE_ONLY"
        master_gap_count = sum(
            gap.device_index == master_index + 1 for gap in result.device_gaps
        )
        if native_pc_time and master_gap_count:
            component["pc_time_status"] = "FAIL"
            payload = {
                "algorithm": "wild_preprocess.pc_time.v1",
                "run_id": run_id,
                "merge_run_id": run_id,
                "status": "failed",
                "aligned_to_merge": False,
                "common_start_master_sample": int(merge_info["common_start_master_sample"]),
                "n_samples": int(merge_info["n_samples"]),
                "error": (
                    f"native PC time is not gap-aware; master has {master_gap_count} "
                    "zero-filled interior gap(s)"
                ),
            }
            (staging / "pc_time_qc.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            result.outputs["pc_time_qc"] = str(output_folder / "pc_time_qc.json")
        elif native_pc_time:
            master = recordings[master_index]
            try:
                indices, packed = collect_packed_updates(master.analog_file, CE64_RAW_MISC_LAYOUT)
                anchor_ms, anchor_source = resolve_recording_start_ms(
                    master.folder, explicit_recording_start_ms=recording_start_ms
                )
                pc_model = fit_robust_pc_time_model(indices, packed, master.fs, anchor_ms)
                pc_validation = validate_pc_time_interval(
                    pc_model,
                    sample_rate_hz=master.fs,
                    common_start_master_sample=int(merge_info["common_start_master_sample"]),
                    n_samples=int(merge_info["n_samples"]),
                    options=pc_time_options,
                )
                payload = pc_time_qc_payload(
                    pc_model,
                    pc_validation,
                    run_id=run_id,
                    common_start_master_sample=int(merge_info["common_start_master_sample"]),
                    n_samples=int(merge_info["n_samples"]),
                    sample_rate_hz=master.fs,
                    anchor_source=anchor_source,
                    layout_name=CE64_RAW_MISC_LAYOUT.name,
                )
                payload.update(
                    {
                        "merge_run_id": run_id,
                        "status": pc_validation.status.lower(),
                        "aligned_to_merge": pc_validation.status == "OK",
                    }
                )
                write_pc_time_qc_json(staging / "pc_time_qc.json", payload)
                write_pc_time_summary_png(
                    staging / "pc_time_fit_summary.png",
                    pc_model,
                    pc_validation,
                    common_start_master_sample=int(merge_info["common_start_master_sample"]),
                    n_samples=int(merge_info["n_samples"]),
                    sample_rate_hz=master.fs,
                )
                result.outputs["pc_time_qc"] = str(output_folder / "pc_time_qc.json")
                result.outputs["pc_time_figure"] = str(output_folder / "pc_time_fit_summary.png")
                if pc_validation.status == "OK":
                    write_interval_pc_time(
                        staging / "pc_time.dat",
                        pc_model,
                        sample_rate_hz=master.fs,
                        common_start_master_sample=int(merge_info["common_start_master_sample"]),
                        n_samples=int(merge_info["n_samples"]),
                    )
                    component["pc_time_status"] = "OK"
                    component["overall_status"] = "COMPLETE"
                else:
                    component["pc_time_status"] = "FAIL"
            except Exception as error:
                component["pc_time_status"] = "FAIL"
                payload = {
                    "algorithm": "wild_preprocess.pc_time.v1",
                    "run_id": run_id,
                    "merge_run_id": run_id,
                    "status": "failed",
                    "aligned_to_merge": False,
                    "common_start_master_sample": int(merge_info["common_start_master_sample"]),
                    "n_samples": int(merge_info["n_samples"]),
                    "error": str(error),
                }
                (staging / "pc_time_qc.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
                result.outputs["pc_time_qc"] = str(output_folder / "pc_time_qc.json")

        result.outputs.update(component)
        write_staged_sync_qc(staging)
        managed_files = sorted(
            path.name for path in staging.iterdir() if path.is_file()
        )
        # Include the manifest itself because it controls removal of all
        # artifacts owned by this generation on the next overwrite.
        managed_files.append("wild_preprocess_run.json")
        expected_output_bytes = {
            "amplifier.dat": int(merge_info["n_samples"]) * int(merge_info["n_channels"]) * 2,
            "analogin.dat": int(merge_info["analog_samples"]) * int(merge_info["analog_channels"]) * 2,
            "time.dat": int(merge_info["n_samples"]) * 4,
        }
        if component["pc_time_status"] == "OK":
            expected_output_bytes["pc_time.dat"] = int(merge_info["n_samples"]) * 4
        warnings = _manifest_warnings(pairs)
        if result.device_gaps:
            summaries = gap_summary(
                result.device_gaps,
                device_count=len(recordings),
                canonical_samples=int(merge_info["n_samples"]),
            )
            for item in summaries:
                if not item["gap_count"]:
                    continue
                entries = [
                    entry
                    for entry in merge_info["device_gaps"]
                    if entry["device_index"] == item["device_index"]
                ]
                filled = sum(entry["action"] == "zero_filled_with_guard" for entry in entries)
                warnings.append(
                    {
                        "device_index": item["device_index"],
                        "status": "GAPS_HANDLED",
                        "message": (
                            f"{item['gap_count']} detected gap(s), {item['missing_samples']} samples "
                            f"({100.0 * float(item['missing_fraction']):.4f}%); "
                            f"{filled} intersect output, {len(entries) - filled} cropped/outside"
                        ),
                    }
                )
        (staging / "wild_preprocess_run.json").write_text(
            json.dumps(
                {
                    "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
                    "run_id": run_id,
                    "algorithm_versions": {
                        "pipeline": PIPELINE_ALGORITHM_VERSION,
                        "sync": SYNC_ALGORITHM_VERSION,
                        "pc_time": "wild_preprocess.pc_time.v1",
                    },
                    "sync_status": result.outputs["sync_status"],
                    "merge_status": component["merge_status"],
                    "pc_time_status": component["pc_time_status"],
                    "overall_status": component["overall_status"],
                    "master_index": master_index + 1,
                    "device_order": input_provenance,
                    "options": {"sync": asdict(options), "pc_time": asdict(pc_time_options)},
                    "merge": merge_info,
                    "expected_output_bytes": expected_output_bytes,
                    "warnings": warnings,
                    "managed_files": managed_files,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    try:
        terminal_crop_pairs = [
            pair for pair in pairs if pair.terminal_crop_master_sample is not None
        ]
        terminal_crop_pair = min(
            terminal_crop_pairs,
            key=lambda pair: int(pair.terminal_crop_master_sample),
            default=None,
        )
        maximum_common_end = (
            canonicalize_master_sample(
                int(terminal_crop_pair.terminal_crop_master_sample),
                result.device_gaps,
                master_device_index=master_index + 1,
            )
            - 1
            if terminal_crop_pair is not None
            else None
        )
        result.outputs.update(
            merge_recordings(
                recordings,
                master_index,
                pair_models,
                output_folder,
                device_gaps=result.device_gaps,
                minimum_common_start=max(
                    (pair.validated_start_master_sample for pair in pairs), default=0
                ),
                maximum_common_end=maximum_common_end,
                maximum_common_end_device_index=(
                    terminal_crop_pair.slave_index if terminal_crop_pair is not None else None
                ),
                maximum_common_end_reason=(
                    terminal_crop_pair.terminal_crop_reason
                    if terminal_crop_pair is not None
                    else ""
                ),
                chunk_seconds=options.chunk_seconds,
                overwrite=overwrite,
                progress=progress,
                run_id=run_id,
                stage_callback=stage_callback,
                additional_managed_names=extra_managed_names,
            )
        )
        result.outputs.update(
            {
                "qc_tsv": str(output_folder / "wild_multilogger_sync_qc.tsv"),
                "qc_json": str(output_folder / "wild_multilogger_sync_qc.json"),
                "qc_mat": str(output_folder / "wild_multilogger_sync_qc.mat"),
                "run_manifest": str(output_folder / "wild_preprocess_run.json"),
            }
        )
        result.outputs.update(component)
        return result
    except _StagedMergeRejected as error:
        result.status = "FAIL"
        result.outputs.update(
            {
                "sync_status": "OK",
                "merge_status": "FAIL",
                "pc_time_status": "NOT_RUN",
                "overall_status": "FAIL",
                "attempt_folder": str(attempt_folder),
                "postmerge_qc": str(attempt_folder / "wild_multilogger_postmerge_qc.json"),
            }
        )
        return result
    finally:
        # Successful outputs are now canonical; failed attempts remain for
        # review and never replace a prior successful generation.
        if result.status == "OK":
            shutil.rmtree(attempt_folder, ignore_errors=True)
