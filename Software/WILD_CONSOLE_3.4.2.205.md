# WILD console 3.4.2.205 — CE128 metadata release

[Download installer](WILD_console_Setup_3.4.2.205.exe).
Source release: [CE32_console commit 3f5217b](https://github.com/zifangzhao/CE32_console/commit/3f5217b68959d06c120f22c0d06e68137c9c6e8c).
Publication in Neurologger reuses that exact checked installer and its existing
evidence, without a rebuild, repackaging or additional hardware/testing cycle.

User requested the CE128 fix and release publication without private Neuropixels
updates. This release uses committed `WILD_console` source at
`74893e47f753babf3b869e84b392b03a3518114a`, not the dirty source used for the
previously packaged 204 media build. No unpublished Neuropixels/BLE/media edits
were imported. It is not claimed to be a byte-equivalent 204 application with
only metadata changed; Neuropixels/media behavior is outside this qualification.

The production change removes one extra closing bracket from
`Ce128SdDecoder.WriteMetadata`. Both incomplete and completed exports now emit
valid JSON. Waveform, MISO correction, file-format routing and firmware are
unchanged from the committed baseline. Version is 3.4.2.205.

One application build reused installed NuGet/SDK assets and the exact checked
204 runtime companions, without rebuilding any companion. A temporary build
wrapper disabled the companion-build target and referenced the checked
`NpV2SdDatExport.dll`. The wrapper is not a product/source change.

Targeted checks against the actual 205 executable passed: 96 CE128 export cases,
34,961,393 checks, all four amplifier layouts, exact samples/MISC/timestamps,
record boundaries/incomplete gates, and a 25.6 MB synthetic run. All six emitted
metadata JSON files parse and match their topology/completion fields. The added
`Tools/Test-Ce128Metadata.ps1` separately covers all four layouts before and
after export completion using the production decoder and a JSON deserializer.
The negative control from installed 204 rejected all six original malformed
metadata files; that retained evidence was not rerun as a hardware test.

The checked binary was packaged once. All 22 independently extracted runtime
files match their checked inputs; all companion/runtime hashes remain equal to
204. The installer contains no firmware. The adjacent
`WILD_CONSOLE_3.4.2.205_PAYLOAD_VALIDATION.json` records exact identities;
superseded installers are retained in `Software/Legacy`.

Evidence: `C:/Frank/Code/WILD_SPI/_artifacts/wild-release-205-20261021`.
No firmware change, device access, recording/card readback, installer execution,
installed-window inspection or end-to-end firmware/installer-pair qualification
was performed. The minimal-change workflow kept private project updates out.
