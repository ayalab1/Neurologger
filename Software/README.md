# Current installers

`wild_console_Setup_3.4.2.170.exe` is the current WILD Console installer.
`wild_console_latest.json` is the updater manifest for that installer.

Version 3.4.2.170 is a PC-only device-information hotfix. After initial sync,
it retries a missing parameter reply without restarting synchronization or
disconnecting. It cancels stale requests on link replacement and recording
start/attach, preserving the new connection's request ownership.

The installer bundles the six already-published FM67 / V5 HW/role images;
firmware and the recording path are unchanged. It retains the .169 four-panel
CL events viewer. See [release notes and validation limits](WILD_CONSOLE_3.4.2.170.md)
and [Live Visualization](../docs/software/live-visualization.md).

Build and deterministic BLE/ownership, scheduler, spectrum, MISC and fused-image
extraction tests passed. The live precheck connected but timed out on device
information; a new-core hardware pass is not claimed. The malformed advertisement
issue and the known bank-B signed-carry export limitation remain unresolved.
Preserve original SD data until using a corrected decoder.

Older WILD Console and superseded USB interface installers are retained in
[`Legacy/`](Legacy/). The other top-level installers target separate supported
utilities and are not replaced by WILD Console.
