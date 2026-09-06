# WILD Preprocess GUI

## Open the GUI

From PowerShell in the repository root, open the GUI with:

```powershell
python Code\WILD_preprocess_gui\wild_preprocess_gui.py
```

This command is sufficient when the active Python environment already contains
the required packages. If packages are missing, install them once with:

```powershell
python -m pip install -r Code\wild_preprocess\requirements.txt PySide6
```

## Use the GUI

1. Click **Browse WILD recording**.
2. Select the session or subepoch folder that contains the individual logger
   recording folders.
3. Check the discovered recordings in the table.
4. In the **use** column, select one recording from each device.
5. In the **role** column, assign exactly one `master`. Assign the other
   selected recordings as `slave`.
6. Set unique **probe index** values in the intended output order.
7. Review the **Ready Check** table.
   - `OK`: the check passed.
   - `WARN`: processing is allowed, but review the detail.
   - `FAIL`: processing is blocked; correct the listed problem.
8. Leave **overwrite generated outputs** cleared for the first run.
9. Click **Run**.
10. Monitor the current step, progress bar, and **Log** pane.
11. Wait for **Publish outputs** and the final completion message before using
    the generated files.

Use **Force stop** only when processing must be interrupted. A stopped run is
not a completed output generation.

## Explanation

### Input folder

The selected folder must contain one or more logger recording folders. Each
selected recording requires:

- `amplifier.dat`;
- `analogin.dat`; and
- `CE_params.bin`.

Discovery is recursive. Generated outputs are written to the selected session
or subepoch folder. Raw files inside the logger recording folders are not
modified.

### Master and slaves

The master defines the canonical output timeline and provides the recording
start used for `pc_time.dat`. Prefer a reliable recording that covers the full
experiment and is at least as long as the slaves.

- If a slave stops early, the master and other slaves continue. The stopped
  slave is zero-filled and marked invalid after its verified endpoint.
- If the master stops early, all canonical outputs end at the master support
  limit. Later slave-only data remain in the raw files but are not included in
  the merged output.

### Ready Check

`0 fail, N warn` is runnable. Read every warning before starting. Pressing
**Run** prints the current warning and failure details in the Log pane.

A missing or invalid master recording start is a failure. An invalid slave RTC
may be accepted with a warning because the slave's neural offset and drift are
measured from signal correlation.

### Main outputs

| File | Meaning |
| --- | --- |
| `amplifier.dat` | Merged neural data |
| `analogin.dat` | Merged analog data |
| `time.dat` | Canonical sample-index timeline |
| `pc_time.dat` | Master-derived PC-clock timeline, when publishable |
| `valid_samples.dat` | Physical data and mapping-validity mask |
| `alignment_quality.dat` | Stricter master-referenced timing-quality mask |
| `valid_analog_samples.dat` | Analog mapping-validity mask |
| `IMU.mat` | Synchronized IMU output |
| `wild_preprocess_run.json` | Run status, warnings, mappings, and output metadata |

Do not treat zero-filled invalid data as measured zero. Apply
`valid_samples.dat` to neural analysis and also apply `alignment_quality.dat`
when the analysis requires reliable master-referenced timing.

### Run status

- `COMPLETE`: the output set and `pc_time.dat` were published. Review any
  component warnings.
- `MERGE_ONLY`: merged data were published, but `pc_time.dat` was not. Do not
  claim camera or behavior-time alignment from this output.
- `FAIL`: a required check or publication gate failed. Do not use the attempt
  as a completed generation.

The authoritative result is `wild_preprocess_run.json`. Review its
`sync_status`, `merge_status`, `analog_status`, `pc_time_status`, and
`imu_status` together with the QC figures.
