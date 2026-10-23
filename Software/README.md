# Current installers

[`WILD_console_Setup_3.4.2.206.exe`](WILD_console_Setup_3.4.2.206.exe) is the current WILD Console installer.
`wild_console_latest.json` is the updater manifest for that installer.

Version 3.4.2.206 adds per-record compatibility for older zero-extended15-bit
MISO2-A captures. A conservative prefix-signature check enables one-bit left
alignment when supported by the data, while native16-bit, positive and flat
captures remain unchanged. The decision, masks and valid bits are recorded in
`ce128_layout.json`. No channel remapping or frame removal is performed.

The exact previously checked installer is republished without rebuilding or
repackaging. Its SHA256 is
`59F79818F70222DC5FA50047657489D39FCB462E8BAA41ADEE987E513984F6AB`.
The actual .206 binary passed96 export cases, compatibility/JSON guards and
beginning/middle/end windows from the provided recording. All22 installer
payloads match checked inputs. Companion assets are unchanged; no private
Neuropixels updates were integrated. Missing physical LSBs and residual
artifacts are not recovered. No new hardware or installed-pair qualification.

This installer contains no firmware and does not flash a device. Existing
[FM69 firmware](../Firmware/V6_FM69_CAMERA_HOTFIX_20261006/) is not changed by
this publication. Read the [release notes](WILD_CONSOLE_3.4.2.206.md) and
[recorded payload validation](WILD_CONSOLE_3.4.2.206_PAYLOAD_VALIDATION.json).

Older WILD Console and superseded USB interface installers are retained in
[`Legacy/`](Legacy/). The other top-level installers target separate supported
utilities and are not replaced by WILD Console.
