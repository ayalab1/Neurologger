# Current installers

[`wild_console_Setup_3.4.2.182.exe`](wild_console_Setup_3.4.2.182.exe) is the current WILD Console installer.
`wild_console_latest.json` is the updater manifest for that installer.

Version 3.4.2.182 contains the shared USB camera controls and media tooling
from .180 plus the published .181/.182 profile-control hotfixes. This package
uses the standard `wild_console` install identity rather than the separate
camera-prototype identity.

The installer bundles exactly the six CE64 FM69 / V6 HW/role fused HEX files
in [`Firmware/V6_FM69_CAMERA_HOTFIX_20261006`](../Firmware/V6_FM69_CAMERA_HOTFIX_20261006/).
No NP firmware is included. Installing the console does not flash a device.
To install the resident USB camera fix, use the matching fused HEX through
full-image SWD or full-flash USB DFU; SD/BLE app-only updates leave that service
unchanged.

The existing released .182 executable is unchanged, and all 28 installer
payloads were compared byte-for-byte. Six firmware builds and offline gates
passed. Camera image quality, full chronology, endurance and physical update
delivery remain unqualified. Preserve original SD data. Read the
[console release notes](WILD_CONSOLE_3.4.2.182.md) and
[firmware limits](../Firmware/V6_FM69_CAMERA_HOTFIX_20261006/FM69_RELEASE_NOTES.md).

Older WILD Console and superseded USB interface installers are retained in
their existing locations and [`Legacy/`](Legacy/). The other top-level installers target separate supported
utilities and are not replaced by WILD Console.
