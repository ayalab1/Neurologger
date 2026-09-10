# CE64 firmware releases

The six `CE64_V5_FM66_*_20260910.hex` files are the latest user-requested fused
release. **Read [FM66 release notes](FM66_RELEASE_NOTES.md) before deployment:
ADC startup zeros remain unresolved and the new paired endurance test is pending.**
FM66 must not be treated as an error-free, fully validated production image.
Each file contains Bootloader V5, the application manifest, the CE64
application, and the resident V4-compatible USB service image.

Select the image by hardware pinout (`HW1` or `HW2`) and compiled role
(`Auto`, `Master`, or `Slave`). WILD Console accepts the fused HEX directly for
BLE OTA, SD update staging, USB DFU, or full-flash programming.

Bootloader V5 restores automatic installation of a valid, different SD image,
accepts both legacy `BOOTLOAD` and V3 staging manifests, and cold power-cycles
the SD rail before probing. The application publishes a recoverable directory
checkpoint every five minutes at a completed allocation-unit boundary.

FM66 adds DMA-backed slow AUX telemetry during high-speed ADC and UART receive
robustness. Its 120-second development test demonstrated changing AUX and
clean recording stop, but saved ADC data had at least 38.4 ms of startup zeros.
Battery freshness during high-speed ADC, ADC-off AUX, analog accuracy, and
full chronology remain unvalidated. No new Console installer accompanies FM66;
the existing WILD Console 3.4.2.169 package is unchanged.

The prior FM65 baseline restores deterministic completion margin for the 1.28 MHz Intan one-shot
without changing the sampling cadence, SD writer, or recorder buffers. It also
hardens the synchronized follower wake/role handoff, keeps advertisement work
outside unsafe recording windows, and limits LMT70/VDD1 duty cycling to fs=0
recording. The 64 MHz, 20 kHz ephys-only path and equal Master/Slave sector
counts passed its release HIL gates. These results do not constitute an exact
FM66 hardware pass. FM65's six images remain available in `legacy/`.

Earlier CE64 images are retained in [`legacy/`](legacy/), including FM65 for
users needing the prior validated baseline or reproducing an older installation.
