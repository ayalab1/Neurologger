# Multi-device sync CI

## Date and git state

- Date: 2026-07-19
- Base commit at verification: `d8c71ea2887f97fcdfafaec4d67bba765cc7eaaa`
- Status: uncommitted at the time this log was written
- Implementation plan: [../implementation_plan/2026-07-19-multi-device-sync-ci.md](../implementation_plan/2026-07-19-multi-device-sync-ci.md)

## What changed

- Reviewed the `multi-device-sync` branch changes before push.
- Added multi-device synchronization support in `Code/WILD_channelMerger.m`, including:
  - `Mode`, `MasterIndex`, `OutputFolder`, and QC/merge threshold options;
  - multi-logger master-vs-slave common-mode sync QC;
  - QC MAT/TSV/PNG outputs;
  - multi-logger merged `amplifier.dat`, `analogin.dat`, `time.dat`, channel layout, event summary, merge info, and IMU output when QC passes.
- Kept the previous two-logger `WILD_channelMerger` behavior as the default for exactly two selected recordings.
- Made the legacy two-logger alignment window use the existing `InitialStartSeconds`, `InitialDurationSeconds`, `InitialMaxLagSeconds`, `ChunkSeconds`, and `ChunkMaxLagSeconds` options while preserving the default values.
- Added `Code/WILD_generate_pc_time.py` for generating `pc_time.dat` from packed WILD/CE analog PC-time lanes, including robust fit diagnostics and optional summary plot output.
- Added `Code/WILD_preprocess_gui/wild_preprocess_gui.py`, a PySide6 GUI that discovers WILD recording folders, lets the user select master/slave recordings, runs ready checks, launches MATLAB multi-logger merge, generates master `pc_time.dat`, and records QC metrics.
- Updated `Code/WILD_scaleIMU.m` so missing `ahrsfilter` does not fail IMU scaling; it now returns scaled IMU metadata and warns when sensor fusion is unavailable.
- Normalized the final newline in `Code/WILD_processIMU.m`.
- Updated `.gitignore` to exclude local test data, caches, agent notes, Obsidian files, and local GUI QC artifacts.
- Added CI workflow `.github/workflows/ci.yml` with MATLAB unit tests and Python syntax smoke checks.
- Added `tests/WILDChannelMergerLegacyTest.m`, which creates temporary synthetic WILD-like recordings and verifies the legacy two-file merge output sizes and near-zero offset.
- Added repository documentation indexes under `implementation_plan/` and `change_log/`.

## Why

The branch expands preprocessing from a two-logger workflow toward multi-device synchronization and session-level outputs. Before pushing, it needed explicit regression coverage for the older two-file merge path so the new routing and multi-logger code do not silently break existing users.

## Verification

Ran:

```powershell
git diff --check
python -m py_compile Code\WILD_generate_pc_time.py Code\WILD_preprocess_gui\wild_preprocess_gui.py
matlab -batch "addpath('Code'); results = runtests('tests'); assertSuccess(results);"
```

Results:

- `git diff --check`: passed
- Python syntax smoke check: passed
- MATLAB unit test: passed; `WILDChannelMergerLegacyTest` generated synthetic two-logger data, merged it through `WILD_channelMerger`, and verified expected merged amplifier/analog file sizes and offset

## Known limitations and next steps

- CI uses synthetic data and does not commit or depend on large local `test_data/` recordings.
- The MATLAB CI job requests Signal Processing Toolbox because `WILD_channelMerger` uses `butter`, `filtfilt`, and `xcorr`.
- Real multi-logger datasets should still be reviewed through the generated QC TSV, MAT, and PNG outputs before relying on merged session-level files.
