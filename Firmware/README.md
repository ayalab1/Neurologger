# CE64 production firmware

The six `CE64_V4_FM63_*_20260831.hex` files are the current fused production
images. Each file contains Bootloader V4, the application manifest, the CE64
application, and the V4 USB service image.

Select the image by hardware pinout (`HW1` or `HW2`) and compiled role
(`Auto`, `Master`, or `Slave`). WILD Console accepts the fused HEX directly for
BLE OTA, SD update staging, USB DFU, or full-flash programming.

Superseded CE64 images are retained in [`legacy/`](legacy/). Do not use a file
from `legacy/` for a new deployment unless reproducing or recovering a known
older installation.
