# Bootloader V6 + unchanged FM67 - USB-startup test release

Published 2026-09-16 for manual device testing. **Physical USB startup validation
is pending**; build, boot-dispatch, package and extraction checks passed.
The earlier V6 core/service review passed. The additional final fused-package
review was unavailable (reviewer execution quota); it is not claimed as passed.

Source: [CE64 V1.6 commit 747be92](https://github.com/zifangzhao/CE64/commit/747be92dbdc2c2e00b46ab7914244fb3d9014ae2).
[Detailed implementation and validation notes](https://github.com/zifangzhao/CE64/blob/747be92dbdc2c2e00b46ab7914244fb3d9014ae2/docs/V6_FM67_FUSED_20260916.md).

Download one complete fused HEX for your hardware and role:

| Hardware | Auto | Master | Slave |
|---|---|---|---|
| HW1 | [Auto](CE64_V6_FM67_HW1_Auto_20260916.hex) | [Master](CE64_V6_FM67_HW1_Master_20260916.hex) | [Slave](CE64_V6_FM67_HW1_Slave_20260916.hex) |
| HW2 | [Auto](CE64_V6_FM67_HW2_Auto_20260916.hex) | [Master](CE64_V6_FM67_HW2_Master_20260916.hex) | [Slave](CE64_V6_FM67_HW2_Slave_20260916.hex) |

[SHA256 checksums](V6_FM67_SHA256SUMS.txt) / [package validation](V6_FM67_PACKAGE_VALIDATION.json).

## What changed

V6 restores automatic USB detection at startup with a valid app. A real USB
host reaching CONFIGURED retains USB mode; VBUS or the master/slave cable alone
does not. If no host enumerates, the existing two-second window after service
initialization expires; the bootloader checks SD and starts the logger without
probing USB again. Total startup also includes initialization and the SD check.
Explicit USB, DFU, SD-update and recovery requests keep priority.

The matching resident service records the return-to-logger path before setup,
so a reset during initialization does not repeatedly enter USB. A normal reset
from USB also uses that one-shot return; a no-mailbox cold boot probes USB again.
Different valid SD images are still installed when returning to the logger.

All six FM67 applications **and their manifests are byte-for-byte unchanged**
from the September 14 V5/FM67 release, including HW/role flags. No recording,
ADC/LMT70, advertising or Console decoder fix is included. Existing
[FM67 limitations](../FM67_RELEASE_NOTES.md) still apply.

## How to install

1. Stop recording. Download the matching fused HEX above.
2. Clear a stale staged SD firmware image, or remove the card for the first boot.
   Otherwise the loader can reinstall that older/different-role app after USB
   exits, making the intended application appear to revert.
3. Program the **entire fused HEX using SWD or USB DFU with full-flash upgrade
   enabled**, with stable power. Do not interrupt a bootloader write; recovery
   may need SWD or ROM DFU. Full-chip erase also erases device configuration.
4. Test cold boot with a USB data cable: the USB service should remain available.
   Then test without USB: the device should return to normal logger/BLE startup.

**BLE OTA, SD update and application-only USB update do not install V6.**
The app still reports FM67; that label alone cannot confirm the loader changed.
WILD Console remains .170 and its installer still contains V5 images, so select
this separately downloaded V6 file rather than the bundled firmware file.

No hardware was flashed in preparing this release. Electrical enumeration,
no-host startup power/timing, shared-wire immunity and actual SD/USB update
execution remain to be tested. The V5 files remain available in the parent
folder for rollback using full-image programming.
