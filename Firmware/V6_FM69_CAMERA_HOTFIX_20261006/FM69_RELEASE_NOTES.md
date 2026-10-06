# FM69 camera hotfix release

FM69 publishes the existing CE64 camera fixes on the FM68 recorder baseline.
Bootloader V6 is retained. The companion WILD Console 3.4.2.182 installer uses
the standard installation identity and bundles all six matching CE64 images.
This is a camera hotfix, not a lossless-camera or endurance qualification.

## Camera fixes

- The discard-only SPI3 TX DMA channel no longer treats a FIFO flag as fatal
  in direct mode. RX errors, TX transfer/direct-mode errors, completion,
  transfer counts and SPI busy/timeout fences remain enforced.
- USB camera state, clock/rates and memory are prepared before enumeration.
  The camera path no longer invokes unrelated Intan initialization, and ARM
  uses the prepared allocation.
- Unused in-capacity virtual-disk sectors read as zero; command reads remain
  inert. USB suspend/resume respects the existing low-power PHY gating.
- The USB mode-selector trailer offset works with both supported base LBAs.

Sampling timers, Intan DMA, recorder allocator/rings, SD scheduling and
housekeeping, logging clocks, sensor formats and DMA priorities are unchanged.
The loader source is unchanged. Source base is CE64 `0f11b709`; generation is
`2026100601`. The companion source release is
[CE64 commit a68d62e6](https://github.com/zifangzhao/CE64/commit/a68d62e6d4056808b175ed3ebe5a5b35c419fb2a)
on the `V1.6` branch.

## Checks

PASS: six fresh ARMCC5 all-C `-O3` builds; 13 established static/model gates;
seven camera discard-status model checks; USB camera preparation/source
guards; fused-image manifests, CRCs, address coverage and six-way SD/BLE/USB
extraction preserving hardware and role. All six app maps retain 326,932 bytes
linked RW/ZI in 327,680 bytes SRAM, leaving 748 bytes static headroom.

The scoped source review passed. WILD Console's exact published 3.4.2.182 EXE
is reused unchanged. Its standard installer was checked by read-only NSIS
decompression: all 28 staged payloads match byte-for-byte, including the six
CE64 HEX files. No NP firmware is included and installation does not flash
the logger. See the [console notes](../../Software/WILD_CONSOLE_3.4.2.182.md).

## Required update method

Stop recording and preserve the original SD data. Choose the correct HW/role
fused HEX. Full-image SWD or full-flash USB DFU is needed to replace the
resident USB camera service. SD staging and BLE OTA extract and install only
the app; they leave the loader and resident service unchanged. Loss of power
during full flash can require SWD recovery.

## Remaining limits

Previous short USB tests produced complete 160,000-byte frames, and the scoped
SD discard-DMA fix removed its reproduced hard stop. The sampled SD content
gate still had 11 warnings. Later replacement-camera attempts failed ARM with
service error 9 before a frame; that error alone does not identify the cause.

No hardware was flashed or operated for this release packaging. FM69 runtime,
decoded image quality, full camera chronology, combined peak throughput,
paired recording, physical update delivery and endurance are UNVALIDATED.
Retain original SD data; complete channel ordering and download losslessness
are not certified by this release. Older packages remain available.
