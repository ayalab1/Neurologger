# Python

Python utilities cover camera decoding, audio handling, GPIO logging, and validation after SD-card export.

The WILD preprocessing GUI also uses a Python backend for multi-device synchronization and merged DAT generation. The GUI remains the normal operator entry point; the backend package under `Code/wild_preprocess/` is an internal, testable implementation rather than a separate public CLI.

## Scripts

| Script | Purpose |
| --- | --- |
| [`WILD_VideoDecodewAudio.py`](https://github.com/ayalab1/Neurologger/blob/main/Code/WILD_VideoDecodewAudio.py) | Decode camera data with audio handling. |
| [`WILD_VideoDecodewAudio_v2.py`](https://github.com/ayalab1/Neurologger/blob/main/Code/WILD_VideoDecodewAudio_v2.py) | Alternative video decode pipeline. |
| [`WILD_VideoDecodewAudio_folder.py`](https://github.com/ayalab1/Neurologger/blob/main/Code/WILD_VideoDecodewAudio_folder.py) | Batch folder processing. |
| [`WILD_GPIO_Logger.py`](https://github.com/ayalab1/Neurologger/blob/main/Code/WILD_GPIO_Logger.py) | GPIO logging utility. |

## Typical Use

- Decode camera or audio files after exporting a recording folder.
- Batch-process multiple recording folders with the folder variants.
- Log GPIO or serial events during synchronization validation.
- Keep generated media and logs separate from the raw SD export.

## Interactive Commands

Run the Python tools from the repository `Code` folder:

```powershell
cd C:\code\github\Neurologger\Code
python .\WILD_VideoDecodewAudio.py
python .\WILD_VideoDecodewAudio_folder.py
python .\WILD_GPIO_Logger.py
```

- `WILD_VideoDecodewAudio.py` opens a file picker for one `misc.dat` recording and expects the matching `adc.dat` file when audio is present.
- `WILD_VideoDecodewAudio_folder.py` opens a folder picker and recursively processes `misc.dat` files below that root.
- `WILD_GPIO_Logger.py` prompts for a COM port and output folder, then writes a timestamped text log.

## Expected Outputs

```text
example_recording/
  misc.dat
  adc.dat
  adc.wav
  adc.mp3
  misc.mp4
  misc_wAudio.mp4
```

Output names depend on the script path and whether audio is present, but the public decoder workflow writes reviewable media next to the source export by default.

## Multi-Device Python Backend

The preprocessing GUI starts `Code/wild_preprocess/worker.py` in one separate Python process for the default Python path. The GUI submits a versioned schema-v2 job, so synchronization, merge, post-merge validation, and PC-time generation belong to the same run while the existing UI stays responsive.

Install the numerical backend dependencies into the same Python environment used to launch the GUI:

```powershell
python -m pip install -r Code\wild_preprocess\requirements.txt
```

The backend:

1. creates one filtered common-mode feature stream per selected logger, using at most two workers;
2. measures each master-versus-slave window first in the normal narrow range and, when needed, reacquires from one bounded wide range using the same FFT correlation profile;
3. separates persistent integer offset steps from continuous sample-clock drift and fits a robust piecewise-affine coordinate model;
4. attributes sign-identifiable steps to a logger only when the three-device evidence is sufficient, and otherwise stops with an unresolved-gap failure;
5. writes a staged channel-interleaved amplifier/analog/time/events/layout merge on one canonical sample axis, inserting zero only in the attributed logger's missing interval;
6. independently remeasures residual device lag in the staged amplifier output at start, 25%, 50%, 75%, and end;
7. decodes packed master PC-time updates, fits and validates one PC-clock model, and directly writes only the merged master interval.

The step size is inferred and is not hard-coded to the 1,856-sample firmware pattern. For the pair convention `slave source = master source + offset`, one negative step confined to a slave pair is treated as a missing interval in that slave. A contemporaneous positive step in every slave pair is treated as a master missing interval. Other patterns, including a positive step seen in only one pair and every two-device discontinuity, remain unresolved and block publication because the missing logger is not identifiable from those measurements alone. A separate two-second endpoint probe protects against a loss that occurs before the ordinary tracking windows establish both offset levels: that probe supplies the early level when possible, and its entire interval is excluded from the published common range so a boundary hidden inside it cannot corrupt the saved prefix. If an early step follows the probe, the common start advances through two complete stable post-step windows. A boundary optimum at the localization search edge is rejected rather than trusted and its confidence is reduced to `medium`. Such an early boundary can be a conservative location rather than the exact physical sample; merge metadata distinguishes `cropped_before_output` from `zero_filled_with_guard` so it is not presented as an interior fill.

A missing-data step must persist. One large offset observation bracketed by stable observations at the original level receives a bounded raw-feature recheck in non-overlapping two-second windows. If any reliable short window supports the excursion mapping, the backend keeps only the validated prefix and crops every device from the beginning of the preceding trusted correlation window. Only when the excursion mapping has zero reliable support and the established mapping has at least two reliable windows on both sides is the isolated lag rejected and the tail retained. A new level appearing only at the recording end cannot obtain recovery evidence and therefore remains a conservative crop. The common-end limiter, pair QC, merge metadata, `time.dat`, and `pc_time.dat` all use the same validated end when cropping occurs; no extra terminal report is generated.

All reliable post-merge measurements must be within 4 samples. A low-signal global checkpoint is retained as a diagnostic rather than rejecting an otherwise verified device only when at least four of its five positions are reliable; a reliable lag above the limit always fails publication. Every inferred gap also receives mandatory one-second stable-region checks immediately before and after its interpolation guard. Confirmed gaps are never reconstructed: the affected logger is zero-filled for the exact canonical interval, interpolation is restarted on each side, and a 16-ephys-sample zero guard is added on both sides so sinc taps cannot cross the discontinuity. Both the missing interval and actual guarded zero-fill interval are recorded in merge JSON/MAT. Digital events whose detected edge falls at a gap boundary are excluded and counted in the existing events summary TSV. Gap locations, counts, missing samples, durations, fractions, confidence, and attribution evidence are stored in the existing QC/merge metadata rather than a new mask or gap-report file.

Raw neural samples use a 32-tap Kaiser-windowed sinc fractional-delay interpolator. Continuous analog inputs use linear interpolation, while the packed digital input uses nearest-neighbor sampling so bit fields are never interpolated. A complete run is staged before session-level files are replaced. The run manifest owns the generated files, so an overwrite also removes managed artifacts that no longer belong to the new device selection (for example, a stale third-device QC figure). Failed work is retained in a run-specific hidden `.wild_sync_attempt_<run-id>/` folder for inspection and never replaces a prior canonical run.

The final manifest reports independent component states:

| Overall status | Meaning |
| --- | --- |
| `COMPLETE` | Synchronization, merge, post-merge validation, and native PC-time validation passed. The merged streams and `pc_time.dat` were published together. |
| `MERGE_ONLY` | Synchronization, merge, and post-merge validation passed, but native PC-time validation failed. Merged streams and PC-time QC diagnostics are published, while `pc_time.dat` is deliberately absent so stale timing data cannot be mistaken for the current merge. |
| `FAIL` | Synchronization or staged post-merge validation failed. New canonical merged outputs are not published; inspect the attempt diagnostics. |

The native PC-time package has no runtime dependency on `WILD_generate_pc_time.py`. That legacy script remains available for manual/legacy use and is still part of the explicit MATLAB fallback workflow; it is not launched after a successful default Python worker run.

QC, merge, and PC-time metadata carry a shared run ID. New sync QC and merge metadata use the `python-gap-aware-sync-v4` algorithm label; the internal GUI worker job remains schema v2, while the published run manifest is schema v3. Successful merge QC uses `mode=multiMerge`; rejected and QC-only attempts use `mode=syncQC`. The GUI accepts an existing merged `pc_time.dat` only when its QC provenance matches that run ID, common start sample, and output sample count. A master gap can still yield a verified merge, but native PC time is deliberately reported as failed and `pc_time.dat` is not published because its current clock map is not gap-aware. Large common-mode feature caches use the system temporary directory by default; set `WILD_SYNC_CACHE_DIR` to choose another local cache location.

Generated outputs include:

```text
amplifier.dat
analogin.dat
time.dat
pc_time.dat                       # COMPLETE only
pc_time_qc.json
pc_time_fit_summary.png
wild_multilogger_sync_qc.tsv
wild_multilogger_sync_qc.json
wild_multilogger_sync_qc.mat
wild_multilogger_sync_master_vs_*_qc.png
wild_multilogger_mergeInfo.json
wild_multilogger_mergeInfo.mat
wild_multilogger_postmerge_qc.json
wild_preprocess_channel_layout.tsv
wild_multilogger_events.tsv
wild_preprocess_run.json
```

`wild_preprocess_run.json` records the schema version, run ID, component states, input/device order, selected options, merge interval and its limiting device/stream, per-device mapped source endpoints, output byte counts, warnings, and managed files. Recording-start anchors and available date provenance are retained. `pc_time_qc.json` records the PC-time model and exactly which merged interval it supports. The summary image shows retained and discarded PC-time updates. On `MERGE_ONLY`, the QC JSON/image are useful review artifacts but `pc_time.dat` is not listed as an expected output. A legacy `pc_time_fit_summary.jpg` may coexist in the session folder; the Python backend owns and manages its PNG report, not that legacy JPG.

The bundled WT4 positive-control session (`test_data/WT4_day19_indoorLearn_071826`) completed the final unified worker path in 637.0 seconds: sync, merge, post-merge validation, and native PC time all reported `OK`/`COMPLETE`. Its merged interval contains 63,366,081 samples; `time.dat` and `pc_time.dat` were both 253,464,324 bytes. The manifest, sync QC, merge metadata, and PC-time QC all carry run ID `f40f3d93bbb74b77b460339a66caf154`.

Gap-aware sync-only checks were also run on the firmware-affected recordings without writing merged DAT files. WT4 day 30 completed in 127.3 seconds, attributed 40 events and left 7 single-positive-pair events unresolved. RM028 verification completed in 220.2 seconds versus 293.8 seconds for the earlier sequential QC path, attributed 90 events and left 10 single-positive-pair events unresolved. Around C9E7EDC0B2E6's isolated 4,400-second excursion, the local raw recheck supported the established mapping in seven windows, the excursion mapping in one reliable window, and left two windows ambiguous. Since the excursion has positive raw support, the safe simple policy keeps the common-tail crop at master sample 87,800,000 (4,390 seconds). RM028 remains `FAIL` because the ten separate positive single-pair events are unresolved, so no canonical merge was published.

During migration, set `WILD_SYNC_BACKEND=matlab` before launching the GUI to use the legacy MATLAB synchronization backend. The Python backend intentionally does not generate multi-device IMU sensor-fusion output yet; that remains a separately documented migration item. Automatic gap attribution currently requires at least three devices and assumes the firmware failure mode is missing samples rather than duplicated samples; ambiguous evidence remains a blocking QC result.
