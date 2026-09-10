# FM66 / Bootloader V5 — 2026-09-10

User-requested versioned release before the new paired endurance test.
**Known limitations remain; this is not an error-free or fully endurance-
validated release. FM65 remains the prior validated baseline.**

## Scope

- AUX/PA0 slow cache and MISC telemetry can reuse completed regular ADC DMA
  samples during high-speed ADC recording. No extra AUX conversion, DMA
  stream, or per-sample interrupt is added; regular samples are not doubled.
- MCU temperature is deliberately unavailable during high-speed ADC, rather
  than presenting a stale reading as current.
- A guarded HW2 battery-injection candidate uses the heavily buffered external
  divider, never reconfigures PA0's active sampling time, and marks a potentially
  affected ADC word with bit0 while retaining its position and upper12 bits.
  This battery behavior is not hardware-certified (see limitations).
- Ephys-only AUX sensing is permitted when VDD1 is already powered. The rail
  is enabled before acquisition and held steady; fs=0 retains duty cycling.
- BLE/inter-device UART initialization uses16x oversampling at115200 baud;
  BLE configuration/rebind reapplies the same framing.
- Intan timer/SPI cadence, SD housekeeping, recorder allocations, file widths,
  and the five-minute allocation-unit-boundary checkpoint policy are unchanged.
- Bootloader V5 and its compatible resident service are unchanged in source.

This release is isolated from unrelated USB/AI/impedance work in the primary
development checkout. Six separate HW1/HW2 x Auto/Master/Slave fused HEX files
are built from explicit defines. The application remains at0x08020000.

## Evidence and known limits

The FM65-labeled development candidate underlying this release passed static
checks and a120-second20kHz ephys +160ksps ADC test for changing raw AUX
cache/MISC values (6176..6688 after startup), clean stop, durable directory
growth, and dynamic middle/end ADC samples. FM66 versioned artifacts require
their own subsequent hardware gate; prior evidence is not an exact-FM66 pass.

Saved-data acceptance FAILED at startup: at least24 zero ADC sectors
(6144 samples,38.4ms) precede a tiny nonzero sample. Stream attribution is
supported by the raw prefix and writer-credit reconstruction. Input warm-up
is plausible, but not proven; stale/premature publication is not excluded.
Do not use the startup interval as validated signal or claim this is fixed.

Other open items:

- Ephys-only / high-speed-ADC-OFF AUX returned rawzero in the available test;
  its restoration remains unvalidated.
- HW2 battery stayed at4882mV in the high-speed run: freshness and shortened
  acquisition accuracy are unvalidated. No claim that low-battery stopping
  has been hardware-proven in that mode.
- HW1 internal VBAT keeps its minimum acquisition time; common160ksps
  configurations cannot satisfy the one-slot guard and will skip that read.
- Bit0 quality marking, single-slot disturbance, full ADC chronology, analog
  noise, disconnected telemetry freshness, and long-term stall immunity still
  require hardware validation. Older readers do not recognize the quality bit.
- A cold host sync returned mode00/ABORT once; the next session reached valid
  mode01 with64 samples. Root cause of the abort remains unproven.

No clocks, DMA phase, or settling delays were changed speculatively to mask
these observations.

## Paired endurance plan (after publication)

Use matched FM66 HW2 Master/Slave images on the Nucleo and mapped replacement
CE64 only. Start a short paired gate through the Master, verify both durable
records and equal ephys sample counts, then run one hour at20kHz ephys-only,
64MHz, ADC/camera OFF. Stop on recording corruption/fault. No SWD attachment
while recording. Do not infer equal samples solely from BLE notifications.
If the second CE64 is unavailable, report the missing paired gate rather than
using CE128 or claiming a single-device test validates synchronization.

No new WILD Console installer is part of this firmware-only release.
