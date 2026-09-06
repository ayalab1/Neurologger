# Current installers

`wild_console_Setup_3.4.2.169.exe` is the current WILD Console installer.
`wild_console_latest.json` is the updater manifest for that installer.

Version 3.4.2.169 adds the four-panel CL events viewer: two triggered
waveforms and two PC-computed Gabor spectra, independent microvolt scales,
and received-event counters. Spectrum work is bounded and runs outside the UI
thread. It also includes the scheduler reply/reconnect and MISC export fixes
committed since .166. Firmware remains the same six FM65 V5 HW/role images.

Build, deterministic spectrum tests, BLE sync/preview regression checks,
and independent source review passed. Current-device end-to-end CL testing
is still pending. This viewer does not enable the separate continuous MCU
spectrum mode. See [Live Visualization](../docs/software/live-visualization.md).

Older WILD Console and superseded USB interface installers are retained in
[`Legacy/`](Legacy/). The other top-level installers target separate supported
utilities and are not replaced by WILD Console.
