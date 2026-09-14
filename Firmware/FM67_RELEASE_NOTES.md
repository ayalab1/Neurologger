# FM67 / Bootloader V5 - 2026-09-14

Firmware source: [CE64/V1.6 commit cf1b229](https://github.com/zifangzhao/CE64/commit/cf1b22922e3d2cffd9bdc627eb638934334d598a).
Independent final packaging review: PASS for all six artifacts, exact roles,
generations, manifests, vectors, checksums and version-only binary differences.

Firmware-only publication of the September 13 recorder optimization candidate.
Six fused HEX images cover HW1/HW2 and Auto/Master/Slave. Application base remains
0x08020000; the fused package also contains the loader, both manifests and the
resident V4-compatible USB service. Select the correct hardware pinout and role.

## Configuration and changes since the published FM66

- Larger 278528-byte recording arena; compact camera thumbnail and shared
  control scratch storage. Configuration that cannot safely run is rejected.
- SD hot-path scheduling, descriptor preparation and direct ADC-ring handling
  reduce foreground work. The release matrix explicitly selects the tested
  `CE32_FILE_SD_SOURCE_PLAN=2` for the application only. Speculative descriptors
  are revalidated before use; no DMA source is owned by a descriptor alone.
- Best-effort camera handling, qualified startup and asynchronous discard.
  Secondary spike/FFT services remain disabled for full-speed recording.
- Peak class is 108.8 MHz (an overclock); ordinary idle WFI is restored. The
  established 20 kHz ephys-only, ADC/camera-off policy remains 64 MHz. The 1.28 MHz
  carrier and sample cadence are unchanged by the versioned publication.
- Application C compilation remains plain O3, not Otime. Recorder allocations,
  five-minute AU-boundary directory checkpoints and slow-ADC policy are retained
  from the tested candidate.
- Includes the staged boot/update and BLE control fixes in this development
  checkout. Fresh hardware SD/BLE/USB upgrade qualification is not claimed here.

The release-specific source changes are the FM66-to-FM67 version increment and
the build script's explicit selection of the already-tested descriptor mode.
No new recorder optimization was introduced during packaging.

Final packaging gates passed: all six fresh builds (70 application C units
per variant, plain O3), role-preserving fused extraction, and stack/RAM checks
(2096-byte stack, 1584-byte reported use, 512-byte margin; 326932/327680 bytes
static RAM). Existing source/model gates passed; the old static wrapper's
missing default map and single-variant input were superseded by explicit checks
of all six release maps and the complete release matrix, not ignored failures.

Strict sparse-HEX comparison to the tested HW2-Master proves identical address
coverage and a byte-identical 7728-byte loader. The 273344-byte application
changes only five FM version bytes (66 to 67); the 232496-byte resident service
changes only three FM version bytes. Manifest generations, hashes and CRCs were
regenerated. No timer, writer or sampling instructions changed in this rebuild.

HW2-Master fused SHA256:
`acad128d1bc8145d4754d1caf812950b68be235b3fdab16791db321676a2c931`.

## Existing hardware evidence (FM66-labelled precursor)

Tested HW2-Master fused SHA256:
`27BAF10BFF08B5E76A6B626A9406775DF395B3A282FD149C86DC3D968AE5C3B8`.
FM67 is rebuilt and separately checked; these are not physical tests of each
of its six new HEX files.

| Gate | 20 kHz ephys only | fs=0 |
|---|---:|---:|
| Duration | 31m24s | 31m29s |
| Durable payload bytes | 4,899,343,872 | 75,574,272 |
| Accepted periodic checkpoints | 6 | 6 |
| SD failures / peak overcapacity / invalid / payload fault | all zero | all zero |
| Fresh reconnect and preview checks | 7 | 7 |
| Disconnected advertisement windows | 4 | 6 |
| Normal Stop, final directory, sampled CRC/readback | PASS | PASS |

Fresh live-time attach, preview On/Off and control succeeded without stopping
recording. Disconnected V7 advertisement pages showed recording and valid
storage, with no critical errors and advancing elapsed time. Payload refresh
was approximately minute-scale, not on every advertising packet.

Short all-rate framing checks (0/1250/2500/5000/10000/20000 Hz and the combined
20 kHz ephys + 160 ksps ADC + camera configuration) passed the corrected decoder.
The combined full-load run sustained about 5.480 MB/s for 62 seconds without a
retained overcapacity/payload fault. This is not camera-quality certification.

## Known limitations and download warning

- The installed/public WILD Console 3.4.2.169 decoder has a bank-B signed-carry
  defect. The saved-excerpt and format tests used the corrected development
  decoder, not that installed executable. This firmware-only release does not
  replace the Console installer; do not treat old-decoder exports as validated.
- Entire-file/every-sample losslessness, physical channel identity, power-loss
  recovery and long-duration paired recording are not certified by these tests.
- A separate camera/ephys run reported 4608-byte peak overcapacity and an invalid
  flag. Camera payloads were largely/all zero on the test input. Camera quality,
  lossless camera capture and its long-run reliability remain unvalidated.
- High-speed ADC startup had roughly 39 ms of zero values. Input settling is a
  hypothesis, not an established explanation. Long-run high-speed ADC integrity
  and injected battery accuracy remain open qualification items.
- Slow raw AUX was present in recorded MISC data. External temperature was
  unavailable because conversion rejected the raw value; the camera/microphone
  ribbon is not a calibrated LMT70 fixture. MCU temperature is intentionally
  unavailable while high-speed ADC is enabled. No calibrated AUX-temperature
  accuracy or low-battery-stop guarantee follows from this release.
- 108.8 MHz is above the nominal MCU specification and has only the stated
  device/card evidence, not a general hardware qualification.

Previous FM66 images and their original notes/checksums are retained under
Neurologger `Firmware/legacy/FM66_20260910/`. Public deliverables remain six
fused `.hex` files; intermediate binaries and symbols are not public updates.

## Reproduction

Run `MDK-ARM/build_ce64_release_matrix.ps1 -BootloaderVersion 5
-FirmwareVersion 67 -Generation 2026091416 -BuildStamp 20260914
-OutputRoot <new-output-directory> -PreserveSymbols` in this source checkout.
The six workers use isolated output prefixes. The matrix checks the fused
SD/USB/BLE extraction and role-preservation contracts; it does not exercise a
physical firmware update.

Local HIL evidence: `_hil/sd_1088_sleep_20260913/endurance/active/RESULT.md`,
`qc/RESULT.md` and `all_rates/RESULT.md` under the primary CE64 workspace.
