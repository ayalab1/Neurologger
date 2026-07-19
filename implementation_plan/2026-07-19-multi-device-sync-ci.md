# Multi-device sync CI

## Goal and motivation

Prepare the `multi-device-sync` branch for push by reviewing the current implementation changes and adding CI coverage that protects the previous two-logger `WILD_channelMerger` behavior.

## Current problem

The branch currently extends `Code/WILD_channelMerger.m` with multi-logger sync QC and merge behavior, adds `Code/WILD_generate_pc_time.py`, and adds a PySide6 preprocessing GUI under `Code/WILD_preprocess_gui/`. The repository only has a documentation deployment workflow, so there is no CI check that exercises the MATLAB preprocessing code before push. The legacy two-logger merge path is especially important because the new `WILD_channelMerger` routing now dispatches to a multi-logger path when more than two recordings are selected or when `Mode` requests sync QC.

## Why this is needed now

Before pushing `multi-device-sync`, the branch needs a clear summary of what changed and an automated regression check showing that the older two-file merge entry point still works with synthetic WILD recordings.

## Git state

- Branch: `multi-device-sync`
- Base at planning time: `d8c71ea2887f97fcdfafaec4d67bba765cc7eaaa`
- Existing modified files at planning time: `.gitignore`, `Code/WILD_PreProcess.m`, `Code/WILD_channelMerger.m`, `Code/WILD_processIMU.m`, `Code/WILD_scaleIMU.m`
- Existing untracked files at planning time: `Code/WILD_generate_pc_time.py`, `Code/WILD_preprocess_gui/wild_preprocess_gui.py`

## Affected modules and files

- `Code/WILD_channelMerger.m`: keep legacy two-logger behavior as the default path and make the existing initial alignment window options usable by that path so a compact synthetic regression test can run quickly.
- `tests/`: add MATLAB unit tests that create temporary WILD-like recordings and verify the legacy two-file merge output.
- `.github/workflows/`: add a CI workflow for MATLAB tests.
- `implementation_plan/` and `change_log/`: add repository-required documentation and indexes.

## Public parameters and API changes

No new public parameters are planned. Existing `WILD_channelMerger` name-value options `InitialStartSeconds`, `InitialDurationSeconds`, and `InitialMaxLagSeconds` will also affect the legacy two-logger alignment window. Their default values remain unchanged, so existing callers retain the old 30 second start, 120 second duration, and 30 second maximum lag behavior.

## Algorithm details

For the legacy two-logger path, replace the hard-coded initial alignment read window:

```text
start = 30 * 20000
end = (120 + 30) * 20000
max lag = 30 * fs
```

with the existing option values:

```text
start = round(InitialStartSeconds * fs)
duration = round(InitialDurationSeconds * fs)
end = start + duration - 1
max lag = round(InitialMaxLagSeconds * fs)
```

The default option values preserve the old behavior for the usual 20 kHz recordings.

## Expected behavior

- Calling `WILD_channelMerger({fileA,fileB}, overwrite, use_cache)` with exactly two files still writes the legacy merged amplifier and analog files.
- Calling `WILD_channelMerger` with more than two files, or with `Mode` set to `syncQC`, still routes to the multi-logger QC/merge path.
- CI runs the MATLAB regression test on push and pull request.

## Verification

- Run MATLAB unit tests locally with:

```powershell
matlab -batch "addpath('Code'); results = runtests('tests'); assertSuccess(results);"
```

- Run `git diff --check`.
- Review `git status --short` before staging/pushing.

## Non-goals

- Do not redesign the multi-logger sync/merge algorithm.
- Do not commit local large test data under `test_data/`.
- Do not change notebook or public analysis APIs unless the verification exposes a direct need.
