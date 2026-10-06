# FM69 camera hotfix with Bootloader V6

Select one fused HEX matching the CE64 hardware pinout and intended role.
Use WILD Console 3.4.2.182 or later for the included USB camera controls.

| Hardware | Automatic | Forced Master | Forced Slave |
|---|---|---|---|
| HW1 | [Auto](CE64_V6_FM69_HW1_Auto_20261006.hex) | [Master](CE64_V6_FM69_HW1_Master_20261006.hex) | [Slave](CE64_V6_FM69_HW1_Slave_20261006.hex) |
| HW2 | [Auto](CE64_V6_FM69_HW2_Auto_20261006.hex) | [Master](CE64_V6_FM69_HW2_Master_20261006.hex) | [Slave](CE64_V6_FM69_HW2_Slave_20261006.hex) |

[Release notes](FM69_RELEASE_NOTES.md) · [SHA256 checksums](SHA256SUMS.txt) ·
[Package checks](FM69_PACKAGE_VALIDATION.json) ·
[Update compatibility checks](FM69_UPDATE_COMPATIBILITY.json) ·
[Console installer](../../Software/README.md)

All six builds, the 13 established static/model gates, and offline SD/BLE/USB
extraction and role checks passed. The recorder path is unchanged. Camera
image quality, full chronology and endurance remain unqualified.

## Updating

Stop recording and preserve the SD data. Full-image SWD or full-flash USB DFU
is required to install the resident USB camera fix. SD staging and BLE OTA
extract the app only and leave the installed loader and resident service
unchanged. Loss of power during full flash can require SWD recovery.

The matching six HEX files are also installed under WILD Console's `firmware`
folder. Installing WILD Console does not flash a logger. Older releases are
preserved in their existing directories.
