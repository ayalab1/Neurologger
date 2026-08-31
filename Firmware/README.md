# CE64 production firmware

The six `CE64_V5_FM64_*_20260831.hex` files are the current fused production
images. Each file contains Bootloader V5, the application manifest, the CE64
application, and the resident V4-compatible USB service image.

Select the image by hardware pinout (`HW1` or `HW2`) and compiled role
(`Auto`, `Master`, or `Slave`). WILD Console accepts the fused HEX directly for
BLE OTA, SD update staging, USB DFU, or full-flash programming.

Bootloader V5 restores automatic installation of a valid, different SD image,
accepts both legacy `BOOTLOAD` and V3 staging manifests, and cold power-cycles
the SD rail before probing. The application publishes a recoverable directory
checkpoint every five minutes at a completed allocation-unit boundary.

Superseded CE64 images are retained in [`legacy/`](legacy/). Do not use a file
from `legacy/` for a new deployment unless reproducing or recovering a known
older installation.
