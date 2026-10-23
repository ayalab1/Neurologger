# WILD3.4.2.206 — MISO2-A compatibility and recording repair

[Download installer](WILD_console_Setup_3.4.2.206.exe).
[Source release](https://github.com/zifangzhao/CE32_console/commit/7ca0ea72be63339c8f63eaa72b57b967a27d5231).
This publication reuses the exact checked installer; it adds no rebuilding,
repackaging, waveform-data publication or further hardware/testing cycle.

Base: committed205 (`3f5217b68959d06c120f22c0d06e68137c9c6e8c`). Product
changes are confined to `Ce128SdDecoder.cs` and the console version. Private
Neuropixels, BLE/media and firmware changes were not integrated. Existing
runtime companions were reused unchanged; no hardware was accessed.

Some older captures contain zero-extended signed15-bit words in MISO2-A,
although the previous decoder assumed a native16-bit A word. This makes a
near-zero negative value such as0x7FFF appear as+32767. Left alignment yields
0xFFFE/-2, restoring the captured sign and scale, but not the missing LSB.

The constructor probes at most512 complete lane groups, preserves stream
position/counts, and changes only physical bank2 when the prefix shows all of:
zero bit15, at least8192 observed words,64 near32768-count sign wraps across
at least8 channels, at least90% of large steps matching those wraps, and at
least99% reduction of8192-count steps after alignment. Zero MSB alone never
enables repair. Ambiguous/short prefixes remain native. A nullable explicit
override is available through the new overload; the original five-argument
constructor remains binary-compatible. This is conservative evidence, not a
guarantee that every possible capture width can be inferred from sample values.

Metadata records the choice/probe statistics and valid bits. A repaired two-chip
record has general mask0x0E, B-only mask0x0A and valid bits16/15/15/15. There is
no remapping, filtering, interpolation, timestamp editing or frame removal.

One production console build passed. The actual206 assembly passed the existing
96-case/34,961,393-check export regression and JSON checks. New targeted tests
preserve native signed, positive and zero data; cover explicit overrides; and
decode reconstructed raw-lane windows from the provided recording's beginning,
middle and end at read sizes1/17/65536. Every expected sample byte and channel/
frame position matches. Reconstruction reverses the already-applied B shifts
and inserts MISC sectors for this check; it is not independent original-SD
readback. Test runner: `Tools/Test-Ce128Miso2ACompatibility.ps1`.

The full provided12.1-minute export was repaired into a separate derived copy:
14,517,496 frames,128 channels,20kHz. Only channels65–96 (1-based) were left
aligned. Other channels and every copied sidecar are byte-identical; the output
was independently read back and hashed. Original file hashes/size/mtime checks
passed. Original amplifier SHA256:
`e440b53e25c680e289bd2e5a59fb6efa883fae1375721830a8f7056ac79ea171`.
Repaired amplifier SHA256:
`9aed13772f7c7650e621c6ebc142f6309cd9dc071483532b99a6a5b278940d28`.
The derived metadata and repair manifest describe the operation and provenance.

Residual outliers remain: bank2's >4096-count steps reduce from14,933,483 to
69,910;254 steps exceed8192 after alignment. These counts do not establish
which remaining events are signal, saturation or acquisition artifacts. No
physical LSB, missing hardware samples or lost analog precision is claimed
recovered. Generated export timestamps cannot certify lossless acquisition.

The installer was packaged once; independent extraction verified22 runtime
files against checked inputs. Application SHA256:
`6E383AF17395E27A079DD9167FE4CDA6D1B228D9E2D5A3E4E746E66D182E13FE`.
Installer SHA256:
`59F79818F70222DC5FA50047657489D39FCB462E8BAA41ADEE987E513984F6AB`.
See the adjacent `WILD_CONSOLE_3.4.2.206_PAYLOAD_VALIDATION.json`. No installer execution, new device
recording, installed-pair or Neuropixels/media qualification was performed.
The minimal-change workflow reused the shared decoder, tests and runtime assets.
