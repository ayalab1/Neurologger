# WILD Console 3.4.2.182 with CE64 FM69

[Download the standard installer](wild_console_Setup_3.4.2.182.exe).
The install folder and application identity are `wild_console`, matching the
existing WILD installation. This is not the separate camera-prototype package.

The published .182 executable from CE32_console commit `fe4fe88b` is reused
unchanged. It inherits the shared CE64 USB camera controls and media tooling
released in .180; .181/.182 repair NP V1 profile confirmation and preserve
camera/audio selections. Those NP fixes do not change CE64 firmware.

## Included CE64 firmware

The `firmware` folder contains exactly six FM69/V6 fused HEX files: HW1 and
HW2, each with Auto, Master and Slave roles. NP firmware is not bundled.
Installing WILD does not program any device.

Stop recording and preserve the SD data before an update. Choose the correct
HW/role HEX. Full-image SWD or full-flash USB DFU installs the resident USB
camera fix; SD staging and BLE OTA extract the application only and leave the
installed loader and service unchanged. Loss of power during full flash can
require SWD recovery.

## Verification and limits

Read-only NSIS decompression compared all 28 staged payloads byte-for-byte,
including the exact released EXE and six HEX files. The installer is
93,590,275 bytes. No installer or logger was executed for this packaging.
The source .182 release records a successful build, 141 protocol/runtime
checks and 228 compiled-GUI/preview checks; these are inherited results, not
a new CE64 hardware qualification.

The six firmware builds, 13 static/model gates and fused SD/BLE/USB extraction
and role checks passed. See the
[FM69 release notes and remaining limits](../Firmware/V6_FM69_CAMERA_HOTFIX_20261006/FM69_RELEASE_NOTES.md).
Camera image quality, complete stream chronology, download channel ordering,
combined throughput, paired recording and endurance remain unqualified.
Preserve original SD data. Prior packages are retained for rollback.

Installer SHA-256:
`4A9203765D1FA88ED7CB817B6F701D5B99A36C02A61BA280279226C6519571D8`

WILD executable SHA-256:
`60F76B88C779A338BE71D0121D8A7C88971FC88CA4B5570FC8C6EF64DF091BD2`

[Exact installer payload validation](WILD_CONSOLE_3.4.2.182_PAYLOAD_VALIDATION.json)
