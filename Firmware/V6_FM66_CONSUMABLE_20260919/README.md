# Bootloader V6 + FM66 — consumable SD-stage test package

Published 2026-09-19. This is a **separate bootloader test package**. It
contains the FM66 application, so it does not replace the newer V6/FM67 or
V5/FM67 packages in the parent directory.

## What this V6 build changes

On a normal logger boot, the loader checks a valid staged SD application before
entering the ordinary logger path. If the staged application is different and
passes its integrity checks, V6 installs it transactionally. Only after the
installed application and manifest verify does V6 erase and verify the staged
image header. The stage is therefore consumed: it cannot be installed again on
the following boot.

Invalid or interrupted stages remain non-bootable until their manifest is
validly committed; they are not treated as an installed application.

## Fused images

Choose the image that exactly matches both board pinout and intended role:

| Hardware | Auto | Master | Slave |
|---|---|---|---|
| HW1 | [Auto](WILD64X_HW1_Auto/WILD64X_HW1_Auto_V6_FM66_consumable_stage_20260919.hex) | [Master](WILD64X_HW1_Master/WILD64X_HW1_Master_V6_FM66_consumable_stage_20260919.hex) | [Slave](WILD64X_HW1_Slave/WILD64X_HW1_Slave_V6_FM66_consumable_stage_20260919.hex) |
| HW2 | [Auto](WILD64X_HW2_Auto/WILD64X_HW2_Auto_V6_FM66_consumable_stage_20260919.hex) | [Master](WILD64X_HW2_Master/WILD64X_HW2_Master_V6_FM66_consumable_stage_20260919.hex) | [Slave](WILD64X_HW2_Slave/WILD64X_HW2_Slave_V6_FM66_consumable_stage_20260919.hex) |

The per-image JSON files, [file manifest](consumable_stage_20260919_file_manifest.json),
and [update compatibility fixture](_validation/consumable_stage_20260919_v6_update_compatibility.json)
record the exact hashes, role flags, and app-region extraction checks.

## Installation boundary

These are complete fused images. Installing V6 requires a full-image method
(SWD or a full-flash USB upgrade) with stable power. An SD, BLE, or app-only USB
update extracts and replaces only the application region; it **cannot replace
an already installed V5 bootloader with V6**.

Do not interrupt a full bootloader flash. Clear an old staged SD image before a
full-image migration, otherwise its valid application may be selected at the
next ordinary boot. Preserve recordings before any destructive operation.

## Evidence and limits

All six fused files built successfully; static policy and fused
SD/USB/BLE extraction/role-preservation fixtures passed. The installed-stage
invalidation logic is covered by the static policy test. This package has not
yet completed a physical end-to-end SD-stage-and-consume or BLE/USB update run,
so those device-level behaviors remain unvalidated.
