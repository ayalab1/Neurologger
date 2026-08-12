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

The backend synchronizes the neural streams, independently reacquires each device after discontinuities, and verifies the merged result. Unknown or failed local intervals are zero-filled and marked `0` in `valid_samples.dat`; samples marked `1` always have a verified source mapping. PC time is generated from the master logger's packed clock only when its full-interval QC succeeds.

Outputs are staged before publication, so a failed run does not replace an existing canonical dataset.

The final manifest reports independent component states:

| Overall status | Meaning |
| --- | --- |
| `COMPLETE` | Neural data and validated `pc_time.dat` were published. |
| `MERGE_ONLY` | Neural data were published, but PC time was unavailable or failed QC. |
| `FAIL` | A structural mapping, DAT, write, or transaction error prevented publication. |

Generated outputs include:

```text
amplifier.dat
analogin.dat
time.dat
valid_samples.dat
pc_time.dat                       # COMPLETE only
pc_time_fit_summary.png
wild_multilogger_sync_master_vs_*_qc.png
wild_multilogger_session_inspection.png
wild_preprocess_run.json
device_event.devXX.dYY.evt        # optional explicit export
```

`wild_preprocess_run.json` is the single metadata file for the Python run. Use the pair QC figures, session inspection figure, and `pc_time_fit_summary.png` for review. Always apply the relevant columns of `valid_samples.dat` during analysis.

Set `WILD_SYNC_BACKEND=matlab` before launching the GUI only when the legacy MATLAB backend is required. Multi-device IMU processing is not part of the Python backend.
