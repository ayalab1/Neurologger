# FM68 / Bootloader V6 — September 30, 2026

Use one fused HEX matching the hardware pinout and intended device role:

| Hardware | Automatic | Forced Master | Forced Slave |
|---|---|---|---|
| HW1 | [Auto](CE64_V6_FM68_HW1_Auto_20260930.hex) | [Master](CE64_V6_FM68_HW1_Master_20260930.hex) | [Slave](CE64_V6_FM68_HW1_Slave_20260930.hex) |
| HW2 | [Auto](CE64_V6_FM68_HW2_Auto_20260930.hex) | [Master](CE64_V6_FM68_HW2_Master_20260930.hex) | [Slave](CE64_V6_FM68_HW2_Slave_20260930.hex) |

[Release notes and validation limits](FM68_RELEASE_NOTES.md) ·
[SHA256 checksums](SHA256SUMS.txt) ·
[Exact package validation identity](FM68_PACKAGE_VALIDATION.json)

FM68 is a versioned release of the tested FM67/V6 recorder. All six builds,
static/model checks, memory limits and fused SD/BLE/USB extraction/role checks
pass. Strict binary comparisons confirm only version constants changed in
the app and resident USB service; loader core bytes are identical.

Fresh hardware evidence is a roughly 31-second 20 kHz ephys + 160 kHz ADC +
camera-configured run on the matching FM67 HW2-Master baseline. Directory
finalization and sampled layout/address/CRC/ADC checks passed. Camera image
quality, complete losslessness, paired endurance and physical update paths
are not certified by that smoke test. Read the limits before deployment.

## Updating

- Stop recording and preserve existing recordings before updating.
- For SD staging or BLE OTA, select the matching fused HEX in a compatible
  WILD Console. It extracts the application; the installed bootloader stays
  unchanged, including any older loader limitations.
- To install V6 itself, use full-image SWD or full-flash USB DFU. Do not expect
  an app-only SD/BLE update to replace the bootloader.
- This release does not rebuild the WILD Console installer. Its bundled
  images may be older; select the downloaded FM68 file explicitly.

Retain original SD data: Console .169/.170 have the documented bank-B decoder
defect and this firmware-only release does not repair that downloader.
Previous packages remain available for rollback.
