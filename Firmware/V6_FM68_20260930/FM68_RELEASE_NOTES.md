# FM68 / Bootloader V6 — 2026-09-30

Firmware-only release of the tested FM67 recorder baseline. Six fused HEX
images cover HW1/HW2 and Auto/Master/Slave; application base remains
`0x08020000`. No console installer or new bootloader behavior is included.

## Why this release

An older V6/FM66 package aborted combined camera recording at retained start
code `E002`, before acquisition. Its start policy demanded the full preparation
margin again after camera/AUX/header setup had consumed it. The established
FM67 source separates full ARM admission from the 500 ms final-ready check,
rechecks the unchanged deadline after DMA setup, and only then admits follower
ETR clocks. FM68 preserves that policy and the validated BLE physical-drain
fence rather than introducing another recorder implementation.

The source base is CE64 commit `747be92` (FM67/V6). The only production-source
change for this release is `FM_VERSION: 67 -> 68`. Sampling timers, SPI/DMA
cadence, recorder rings, SD scheduling/checkpoints and file layout are unchanged.
Compiler policy remains all-C `-O3`; the release matrix retains the tested
application `CE32_FILE_SD_SOURCE_PLAN=2`. Peak clock remains 108.8 MHz (an
overclock), and 20 kHz ephys-only with ADC/camera off remains 64 MHz.

## Fresh hardware evidence (FM67 baseline, not all six FM68 variants)

Nucleo A010 / MAC `10A0022FF7F4` / probe `0672FF515049657187221812`:

- Full fused flash and verification passed at 480 kHz. The application was
  launched directly, so automatic USB startup was not exercised.
- Natural sync; 20 kHz ephys + 160 kHz ADC + camera configured 16 fps;
  approximately 31.125 seconds of acquisition; normal Stop and finalization.
- Valid new entry: 342120 sectors / 175165440 bytes including the reserved
  header gap; 170971136 payload bytes. Earlier directory entries unchanged.
- Zero reported storage faults, setup disconnects or recording disconnects.
- Three header sectors, 278 consecutive beginning/end layout-window sectors
  and sixteen distributed payload sectors passed returned-address/CRC checks.
- Sampled MISC counters were 0..15 and the expected 38976..38991 at payload
  offset 333732. Left-aligned ADC codes were live near half scale in both
  windows. Test rates were cleared and a fresh GATT query confirmed idle.

Tested fused SHA256:
`78e4060520fbcd7d58095e23f7537c07fb5d4d4e9d0710dcfc63707db2436b96`.
Tested application SHA256:
`d2f5002f7bc990b65449b1453d116a95e2d1af8d0f52e53fb30e7b1a12f8a2aa`.
Local evidence: `_hil/a010_camera_20260930/RESULT.md` in the primary workspace.

## Validation limits

All six FM68 builds passed the thirteen established static/model gates,
SD/BLE/USB extraction and role checks, and per-variant memory checks. Stack
margin is 512 bytes and linked RAM headroom is 748 bytes; recorder storage was
not reduced. Strict comparison against each FM67/V6 variant found unchanged
address coverage, sizes, vectors and role flags: five application bytes and
three USB-service bytes change only from version 67 to 68. Loader core bytes
are identical. Manifest generations, CRCs and build hashes are regenerated.

Those offline checks are not physical tests of every new variant. Camera payload
warnings for repeated/all-zero pixels remain; image quality and actual frame
rate are not certified. Entire-file chronology/losslessness, physical ephys
channel identity, paired endurance and long-duration ADC behavior remain open.
The final baseline parameter block retained legacy error `0x81`; a successful
smoke run does not prove those diagnostics cleared. No calibrated temperature
accuracy or power-loss recovery guarantee follows from this test.

## Installation

Select one fused HEX matching hardware and intended role. A fused-HEX-aware
WILD Console extracts only the application for SD/BLE updates; those paths do
not replace the installed bootloader. Full-image SWD or full-flash USB DFU is
needed to replace the bootloader. Existing loader limitations still apply.
Bootloader V6 behavior is unchanged from the tested baseline; physical USB
startup/update and the separate consumable-stage experiment are not certified
by this camera recording test.

Keep original SD recordings until exported with a corrected downloader:
Console .169/.170 have the previously documented bank-B signed-carry decoder
defect. This firmware-only release does not update the installer or certify
those old decoder exports. Prior firmware packages remain available for rollback.

Package generation: `2026093001`; build stamp: `20260930`. Release artifacts
and SHA256 checksums are published under Neurologger `Firmware/V6_FM68_20260930/`.
See `FM68_PACKAGE_VALIDATION.json` for the exact six-file publication identity.
