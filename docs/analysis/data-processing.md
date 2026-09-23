# Data processing

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

## Multi-device preprocessing

Use the WILD preprocessing GUI for multi-device synchronization. The GUI runs the Python backend in a separate process; `Code/wild_preprocess/worker.py` is an internal entry point rather than a separate operator workflow.

Install the numerical backend dependencies into the same Python environment used to launch the GUI:

```powershell
python -m pip install -r Code\wild_preprocess\requirements.txt
```

The backend synchronizes the neural streams, independently reacquires each device after discontinuities, and verifies the merged result. Missing, repeated, or unsupported source intervals are zero-filled and marked `0` in `valid_samples.dat`. A temporarily weak correlation can retain a valid mapping when reliable observations on both sides support the same continuous affine clock model within the analysis tolerance. Such timing interpolation is recorded in each pair's `interpolated_intervals`; the original correlation-acceptance and model-inlier flags are preserved, and these observations do not become new anchors. Unlocalized small lag changes explained by that model are retained as `continuity_supported_boundaries` diagnostics, rather than becoming structural gaps. Raw-localized discontinuities and explicit missing/repeated intervals remain barriers. This inference assumes stable clocks between supporting observations and does not establish directly measured alignment throughout the interpolated interval.

Level-shift and adjacent-offset statistics use robust model inliers, so isolated endpoint or reacquisition aliases do not reject an otherwise supported recording. Rejected observations and robust-model outliers form the same low-confidence run for two-sided interpolation; a coarse alias inside a weak interval cannot become its supporting anchor. Unsupported, sustained model-outlier runs remain a separate discontinuity check and can only use the small-offset continuity allowance when their measured offsets stay within the analysis tolerance.

An unlocalized sub-tolerance boundary near the recording end can use three qualified terminal observations spanning at least one tracking step even if the recording ends before a full observation-window span is available. The last observation must itself be qualified, and the existing two-sided affine-agreement requirements remain in force. This terminal exception does not extend unsupported recording tails or alter missing/repeated-data handling.

Final alignment QC uses the synchronization high-pass cutoff (200 Hz by default), a causal second-order filter with valid warm-up, and up to the configured observation-window duration (10 seconds by default). The production tolerance remains 1 ms. Repeated-failure, recovery, and residual-correction decisions require non-overlapping measurement windows for independent support. `alignment_quality.dat` records timing warnings separately from structural validity and does not zero measured waveforms. Direct callers of `validate_segment_staged_merge` should pass `highpass_hz` to use this feature definition; omission retains the unfiltered direct-call behavior.

Retention rates use the common recording overlap within the output timeline. Raw recording endpoints are projected through the first and last authoritative affine maps; the valid-segment endpoints do not determine this denominator. Internal invalid intervals and unsupported tails inside that overlap still reduce retention. `merge.recording_overlap` records the half-open interval, per-device recording extents, and sample count. `validity_summary` and `alignment_quality_summary` counts and fractions refer to that interval; their `output_` fields retain the original full-output statistics. The inspection figure uses the same denominator while displaying the complete output timeline. If verified edge mappings or common overlap are unavailable, output-based statistics are retained with an explicit unavailable status. Projecting endpoints for these statistics does not authorize rendering outside supported segments.

PC time is fitted from stable packed-clock updates on the master logger's verified analog timeline. A supported, monotone canonical fit is published even when non-blocking quality checks such as an internal anchor gap or a single rate-change candidate report `WARN`; the warning and its diagnostics remain in the manifest and fit figure. Publication is still withheld for fewer than two time-separated anchors, a non-finite/non-increasing model, a persistent clock step, or a rate change reproduced at multiple tested boundaries.

Outputs are staged before publication, so a failed run does not replace an existing canonical dataset.

The final manifest reports independent component states:

| Overall status | Meaning |
| --- | --- |
| `COMPLETE` | Neural data and canonical `pc_time.dat` were published. Component quality may still be `WARN`; consult the manifest. |
| `MERGE_ONLY` | Neural data were published, but no defensible PC-time model was constructed or its canonical file could not be written. |
| `FAIL` | A structural mapping, DAT, write, or transaction error prevented publication. |

Generated outputs include:

```text
amplifier.dat
analogin.dat
time.dat
valid_samples.dat
pc_time.dat                       # COMPLETE; may carry PC-time QC WARN
pc_time_fit_summary.png
wild_multilogger_sync_master_vs_*_qc.png
wild_multilogger_session_inspection.png
wild_preprocess_run.json
device_event.devXX.dYY.evt        # optional explicit export
```

`wild_preprocess_run.json` is the single metadata file for the Python run. `pc_time.published` records file availability independently of `pc_time_status`; a published `WARN` fit is usable as a fitted coordinate but may have locally elevated uncertainty. Use the pair QC figures, session inspection figure, and `pc_time_fit_summary.png` for review. Always apply the relevant columns of `valid_samples.dat` during analysis: a timestamp over a zero-filled interval locates canonical time but does not make the neural signal valid.

The Python workflow obtains each recording's absolute start anchor from the RTC
metadata in `CE_params.bin`. Ready Check reports a failure before processing if
that structured date/time is missing or invalid. A timestamp-like recording
folder name is not authoritative and is not used by default because exported
folders may be renamed or contain malformed suffixes. Programmatic worker and
standalone-generator callers may explicitly enable the folder-name fallback
only for legacy recovery; the recorded provenance identifies that fallback.
No path silently substitutes midnight.

Set `WILD_SYNC_BACKEND=matlab` before launching the GUI only when the legacy MATLAB backend is required. The standard Python GUI backend generates synchronized multi-device IMU output; the legacy MATLAB fallback does not gain that processing path.

For a published Python manifest, regenerate PC time by rerunning the full preprocessing pipeline. The GUI does not allow the legacy standalone PC-time generator to replace a native result because that path does not share the canonical analog mapping or publication blockers.

The GUI standard run always generates synchronized `IMU.mat`; this is no longer an operator option. LFP generation is not part of this preprocessing workflow and is not shown in the GUI. During a run, the progress bar reports the current numbered pipeline step and the percentage completed within that step. The percentage resets when processing advances to the next step instead of presenting a synthetic whole-run estimate.
