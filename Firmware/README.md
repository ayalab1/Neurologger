# CE64 firmware releases

Latest firmware: **FM67 / Bootloader V5, 2026-09-14**.

Download one fused HEX matching the logger's hardware pinout and intended role:

| Hardware | Automatic role | Forced Master | Forced Slave |
|---|---|---|---|
| HW1 | [Auto](CE64_V5_FM67_HW1_Auto_20260914.hex) | [Master](CE64_V5_FM67_HW1_Master_20260914.hex) | [Slave](CE64_V5_FM67_HW1_Slave_20260914.hex) |
| HW2 | [Auto](CE64_V5_FM67_HW2_Auto_20260914.hex) | [Master](CE64_V5_FM67_HW2_Master_20260914.hex) | [Slave](CE64_V5_FM67_HW2_Slave_20260914.hex) |

See [release notes and validation limits](FM67_RELEASE_NOTES.md) and
[SHA256 checksums](FM67_SHA256SUMS.txt). All six builds passed the fused-image
extraction/role and memory gates. The HW2-Master application differs from the
tested September 13 candidate only in its FM version constants.

The candidate passed separate 31-minute 20 kHz ephys-only and fs=0 recordings,
periodic directory checkpoints, reconnect/preview/Stop, disconnected status
advertisements and sampled readback. This does not certify every sample of a
complete recording, paired endurance, camera quality or all physical updates.

FM67 preserves the 64 MHz 20 kHz ephys-only policy. Its peak class is 108.8 MHz
(overclock), with the tested SD descriptor path and ordinary idle WFI restored.

## Updating

The same fused HEX is used for SD staging, BLE OTA, USB DFU or full SWD flash.
A compatible WILD Console extracts the application for SD/BLE; those paths
**do not replace the installed bootloader**. Installing/replacing Bootloader V5
requires full-image programming by SWD or the supported full-flash USB option.
An older installed loader retains its own update limitations.

Bootloader V5 automatically installs a valid, different staged SD application.
No intermediate `.bin` is required for current fused-HEX-aware Console workflows.
Host extraction fixtures are not a guarantee of an interrupted-device recovery.

## Downloader warning

Console 3.4.2.169 and 3.4.2.170 have a known bank-B signed-carry decoder
defect. Validation used the corrected development decoder. The .170 PC-control
hotfix does not change that decoder or certify its exports. Keep original SD
data until downloading with the corrected decoder.

Older releases remain recoverable in [legacy](legacy/), including the archived
[FM66 package and notes](legacy/FM66_20260910/). No recordings were erased or
devices flashed as part of this publication.
