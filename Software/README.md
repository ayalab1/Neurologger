# Current installers

[`WILD_console_Setup_3.4.2.205.exe`](WILD_console_Setup_3.4.2.205.exe) is the current WILD Console installer.
`wild_console_latest.json` is the updater manifest for that installer.

Version 3.4.2.205 fixes malformed CE128 download metadata JSON. It uses the
committed WILD Console source, excluding private Neuropixels/BLE/media edits.
The CE128 waveform and MISO repair logic is unchanged from that committed
baseline; this is not a reconstructed build of the non-frozen .204 media app.

The exact previously checked installer is republished without rebuilding or
repackaging. Its SHA256 is
`D3D5199B1F8F3174F7E311C07C786B37D533DAA40197760A6D5A23D4368F1245`.
The actual .205 binary passed 96 CE128 export cases and strict JSON checks;
all 22 extracted runtime files matched their checked inputs. Companion assets
are unchanged. No new hardware or installed-pair qualification was performed.

This installer contains no firmware and does not flash a device. Existing
[FM69 firmware](../Firmware/V6_FM69_CAMERA_HOTFIX_20261006/) is not changed by
this publication. Read the [release notes](WILD_CONSOLE_3.4.2.205.md) and
[recorded payload validation](WILD_CONSOLE_3.4.2.205_PAYLOAD_VALIDATION.json).

Older WILD Console and superseded USB interface installers are retained in
[`Legacy/`](Legacy/). The other top-level installers target separate supported
utilities and are not replaced by WILD Console.
