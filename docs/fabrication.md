# Fabrication

Board packages support ordering, assembly, and inspection for WILD hardware revisions.

For day-to-day bench work after PCBA delivery, use the detailed [WILD soldering and rework SOP](fabrication/soldering-rework.md). It covers microscope assembly, double-sided fixturing, hot-air rework, STM32 QFN replacement, Intan BGA reballing, and Omnetics connector hand soldering.

## Fabrication Files

| Board | Files |
| --- | --- |
| Datalogger | [Gerbers, drill files, assembly files](https://github.com/ayalab1/Neurologger/tree/main/PCB/Datalogger) |
| Opto module | [Gerbers, drill files, assembly files](https://github.com/ayalab1/Neurologger/tree/main/PCB/Optomodule) |
| Camera unit | [Gerbers, drill files, assembly files](https://github.com/ayalab1/Neurologger/tree/main/PCB/CameraUnit) |
| USB-GPIO board | [Gerber package, BOM, and Eagle source files](https://github.com/ayalab1/Neurologger/tree/main/PCB/USBboard) |

## USB-GPIO Board Package

The USB-GPIO board fabrication package is stored in [`PCB/USBboard`](https://github.com/ayalab1/Neurologger/tree/main/PCB/USBboard). It includes the dated Gerber package, BOM files, and Eagle board/schematic source files.

| File | Purpose |
| --- | --- |
| [`WILD_USB_2026-06-18.zip`](https://github.com/ayalab1/Neurologger/blob/main/PCB/USBboard/WILD_USB_2026-06-18.zip) | Gerber package for PCB fabrication. |
| [`WILD_USB.xlsx`](https://github.com/ayalab1/Neurologger/blob/main/PCB/USBboard/WILD_USB.xlsx) | BOM workbook for assembly review. |
| [`WILD_USB.csv`](https://github.com/ayalab1/Neurologger/blob/main/PCB/USBboard/WILD_USB.csv) | BOM export for manufacturer upload or quick inspection. |
| [`WILD_USB.brd`](https://github.com/ayalab1/Neurologger/blob/main/PCB/USBboard/WILD_USB.brd) and [`WILD_USB.sch`](https://github.com/ayalab1/Neurologger/blob/main/PCB/USBboard/WILD_USB.sch) | Eagle board and schematic source files. |

## PCB Fabrication

Use the zipped Gerber and drill files for the target WILD board revision. Upload the package to a PCB manufacturer that supports fine-pitch multilayer boards; [NextPCB](https://www.nextpcb.com/) is one option used for ordering.

Recommended fabrication parameters for the current WILD datalogger board:

| Parameter | Value |
| --- | --- |
| Board thickness | 1.0 mm |
| Layer count | 4 layers |
| Minimum drill hole | 0.15 mm |
| Trace width | 3.5 mil |
| Blind vias | Rank 1 blind vias |
| Surface finish | Engineering gold, 1 uin |

Before submitting the order, confirm that the fabrication house can accept the drill files, blind-via definition, board outline, and stackup used by the selected WILD revision.

## Assembly Order

1. Upload the board package again for PCBA quotation, including the `Assembly` folder supplied with the selected WILD board revision.
2. Upload the BOM and placement files supplied for the same WILD board revision.
3. Review unavailable or special-order parts before approving assembly.
4. Confirm with the manufacturer which parts will be machine assembled and which parts will be left for hand soldering.
5. Finish hand soldering, connector inspection, and continuity checks before first power-up.

## Special Components

Some WILD components may need advance ordering or manual handling:

| Component | Manufacturing note |
| --- | --- |
| BMX160 IMU | This part is obsolete. BMI270 is a possible 6-axis replacement candidate, but the public WILD driver path is not developed yet. Do not substitute it unless the release image explicitly supports the replacement. |
| HJ580 BLE module | Confirm availability directly with HJ-SIP. Validated builds should use the HJ580 module firmware version documented for WILD, currently firmware version 20220507. |
| Omnetics A79025 | This high-density Omnetics probe connector has a high failure rate during standard reflow. Prefer manual soldering with an alignment fixture, then inspect alignment, solder bridges, and continuity under magnification. The detailed connector workflow is documented in the [WILD soldering and rework SOP](fabrication/soldering-rework.md). |

## Post-Assembly Bring-Up

Prepare a bench cable before applying power to a newly assembled WILD device.

1. Solder the WILD 4-pin JST-SH 1.0 mm cable according to the connector illustration in [Hardware Setup](getting-started/hardware-setup.md).
2. Pre-crimped JST-SH 1.0 mm wire leads or equivalent qualified leads can be used for cable preparation after verifying polarity and pin order.
3. Build a 4-pin USB power/programming cable by soldering the JST-SH lead to a USB-A connector breakout or equivalent bench-supply adapter. Keep the cable labeled and reuse it only after continuity checks.
4. Place a USB power meter between the PC or USB supply and the 4-pin cable.
5. Connect the WILD device for the first time and check the current before programming. Current below 10 mA is the expected first-pass check for no obvious short circuit; disconnect immediately if the current is higher than expected or unstable.
6. Momentarily short the DFU mode-select pin to VCC with metal tweezers while connecting USB, using the connector schematic to avoid shorting the wrong pins.
7. Confirm that the PC detects an `STM32-DFU` device. If the driver is missing, install the ST DFU driver package from [STSW-STM32080](https://www.st.com/en/development-tools/stsw-stm32080.html).
8. Open WILD Console, go to the **Advanced** tab, and use the packed DFU update pipeline to select the latest validated `.hex` release image.
9. After the first DFU installation succeeds, use the documented microSD release-image update path for later upgrades when the release notes support it.

## Inspection Checks

- Confirm top and bottom orientation against the board files.
- Inspect the microSD socket, battery connector, BLE UART pins, sensor connectors, stimulation output path, and probe connector before power-up.
- Check for solder bridges around fine-pitch parts and high-density connectors.
- Keep the PCB revision, manufacturer, assembly date, special part substitutions, and release-image filename with the experiment metadata.
- Match each board revision to a compatible release image before recording.
