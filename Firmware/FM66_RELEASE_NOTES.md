# FM66 / Bootloader V5 — 2026-09-10

User-requested versioned release before the new paired endurance test.
**Known limitations remain; this is not an error-free or fully endurance-
validated release. FM65 remains the prior validated baseline.**

## Same-version replacement: peer UART clock fix

At the user's request, the six current FM66 HEX files were replaced without
incrementing FM. Build stamp `20260910_peerclock1`, generation `1789074246`;
use the updated SHA256 sums to distinguish this replacement from original FM66.
Public filenames are unchanged. Both versions report FM66.

Firmware source: CE64 commit `b52e3ca2a5bde65797322639cfd00e9e901cb011`
on branch `V1.6`.

The BLE-preserving clock-transition path previously skipped retiming the peer
UART. On a Slave at 64 MHz APB2, USART6 retained BRR=0xD0 from 24 MHz, producing
about 307692 baud instead of 115200. The replacement updates the divider from
the actual APB clock, fences new peer transmissions during clock changes, and
waits for active TX to finish before applying the divider. Queues and RX state
are preserved. This omission predates FM66.

No sampling clocks, Intan timing/DMA, ADC policy, SD housekeeping, recording
format, or recorder allocations were changed. Bootloader core bytes are
unchanged; the resident USB service was rebuilt with the shared control source.

Validation of the exact replacement:

- Six-variant build, source/model gates, role-preserving update fixture and
  independent recording/control and packaging reviews passed.
- All variants retain 245248-byte recorder arena, 327168/327680-byte static
  RAM use, and 608-byte reported stack margin.
- Both mapped HW2 Master/Slave devices passed full flash verification.
- Slave hardware readback at 64 MHz confirms BRR=0x22C (~115108 baud).
- Master-only 30-second, 20 kHz ephys, ADC/camera OFF test passed on both:
  166808 total sectors / 158616 payload sectors each, representing 624704
  stored frames per channel. Both distributed 16-sector payload checks passed
  CRC, raw-content and bank-alignment checks. Slave was idle after Master Stop.
- Slave received 19 additional valid peer messages with zero new RX failures.
- One-hour endurance started September 10 at 17:13 EDT and remains pending
  at publication. Equal stored slots and sparse CRC are not full chronology
  proof. All ADC and other limitations below remain in force.

HW2 Master fused SHA256: `f959fc189dacada20ebf2f53c62c35e93a06460fc162f9c27b4458c5fafb169e`.
HW2 Slave fused SHA256: `b80f557a09c737a0605e3b8e779848d4bf2edd35ff3dc9691877ff0e8855dde6`.

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
- Bootloader V5 core is unchanged; the compatible resident service includes
  the shared peer-UART clock correction described above.

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
