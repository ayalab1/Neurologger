# WILD Console 3.4.2.170 — device-information read recovery

Released 2026-09-16. Source: CE32_console `WILD_console` commit
`1c7e3cbb` (PC-only hotfix on .169).

Installer: `wild_console_Setup_3.4.2.170.exe`  
SHA256: `F267FF831EEDA01C02EBA7F429D08F070D69B355B8E3E52BD70F741619D2E64F`

## Changes

- Wait 500 ms after initial synchronization before requesting device information.
- Retry only the idempotent `0x90` read, at most three attempts with a
  three-second response window each. Preserve time-sync state; do not reconnect
  or issue a full resync to recover a missing reply.
- Serialize the system/DSP parameter-read chain. Cancel stale requests on link
  replacement, recording start/attach and slave attach. Recheck the connection
  epoch and recording/OTA eligibility after acquiring the transmit lock.
- Retire canceled request ownership immediately so delayed old-link cleanup
  cannot clear the replacement connection's pending response.
- Report exhausted retries explicitly; Read Params remains available for a
  manual retry. Existing idle reconnect, recording live-PC-time attachment,
  slave behavior, urgent Stop/Reset and wire command formats are unchanged.

This installer retains the four-panel CL events viewer and bundles the six
already-published full V5/FM67 HW1/HW2 Auto/Master/Slave HEX images. Each packaged
image hash matches the published firmware. No firmware, recording, DMA, clocks,
stream layout or advertisement encoder was changed.

## Validation

- Release build and installer build passed with existing compiler warnings.
- Fifteen deterministic retry/ownership checks passed, including a test using
  the actual production BLE class to overlap old/new connection ownership without
  sending any BLE command.
- BLE sync/preview/priority-Stop contracts, scheduler framing/protocol, spectrum,
  MISC export and all six FM67 fused-image extraction fixtures passed.
- Dedicated BLE source reviewer: PASS; the reconnect owner race found during
  review was fixed and retested. No remaining P0/P1 blocker in this hotfix delta.
- Fresh hardware precheck with the previous pinned CLI established GATT on
  Nucleo `10A0022FF7F4` but timed out on device information. Therefore a new-core
  live pass is **not claimed**. No recording, reset, flash, configuration or SD
  command was sent in this hotfix's hardware check.

The executable's informational version includes `.dirty` because this legacy
project tracks generated version/build files, which the build regenerates.
Production source was built from the commit above; unrelated development edits
were not included. Installer and bundled firmware hashes were checked after copy.

## Remaining limitations

The malformed MAC-prefixed FM67 advertisement remains a separate firmware/radio
issue. This release does not claim to eliminate every sync/GATT failure or newly
qualify recording-time live behavior and old-firmware hardware compatibility.

The known bank-B signed-carry decoder defect in .169 is also unchanged in this
control-plane hotfix. **Keep original SD data until using a corrected decoder.**
The passing MISC and extraction tests do not certify that amplifier export path.
