# WILD Soldering and Rework SOP

This SOP is for operators assembling or repairing the double-sided WILD datalogger PCB. It covers microscope work, hot-air rework, reflow-oven use, STM32 UFQFPN48 replacement, Intan RHD2164 BGA bake/reball/reinstall, and Omnetics A79025 connector microsoldering.

Use this document together with the WILD datalogger assembly files:

- [WILD datalogger assembly folder](https://github.com/ayalab1/Neurologger/tree/main/PCB/Datalogger/Assembly)
- [XE64_HDI part list](https://github.com/ayalab1/Neurologger/blob/main/PCB/Datalogger/Assembly/XE64_HDI.txt)
- [PnP front](https://github.com/ayalab1/Neurologger/blob/main/PCB/Datalogger/Assembly/PnP_XE64_HDI_front.txt)
- [PnP back](https://github.com/ayalab1/Neurologger/blob/main/PCB/Datalogger/Assembly/PnP_XE64_HDI_back.txt)

## Scope

This SOP is written for the current WILD datalogger assembly, a dense double-sided HDI board with parts on both sides.

It is intended as a public process guide rather than a fixed recipe. Validate temperatures, dwell times, alloys, and moisture-control steps against the active component datasheets, solder materials, and rework equipment before using the procedure on a device.

| Target | WILD ref | Package | Side | Notes |
| --- | --- | --- | --- | --- |
| Main MCU | `U$6` | STM32F411 `UFQFPN48` | Back | 7 x 7 mm, 0.5 mm pitch, exposed pad |
| Neural frontend | `U$1` | Intan `RHD2164` `BGA104_9X7` | Front | 9 x 7 mm, 104-pin, 0.5 mm pitch |
| Probe connector | `S1`, `S3` | Omnetics `A79025` | Front and back | Manual soldering only in this SOP |
| Board style | whole board | double-sided dense PCB | Both | 0201/0402 passives, nearby heat-sensitive plastics |

## Non-Negotiable Rules

1. Work under ESD protection, with microscope, fume extraction, and a rigid PCB support fixture.
2. Do not use hot air on a populated WILD board without controlling airflow, dwell time, and shielding of nearby parts.
3. Do not rely on station dial numbers. Use measured board/package temperature during initial setup.
4. Do not pry QFN, BGA, or Omnetics parts off the board. Lift only after full reflow.
5. Do not scrub BGA pads with dry braid or force. Wicking is capillary action, not abrasion.
6. For Omnetics work, minimize housing temperature and dwell. The connector material can be damaged near its approximate `220 C` thermal limit unless the supplier documentation for the exact part revision says otherwise.
7. For Intan reballing, use a matching laser-cut stencil, a stable alignment fixture, and preformed solder balls that match the intended alloy.
8. If alloy history is unknown, stop and document it before rework. Do not mix alloys casually on the target board.
9. If a chip pad, board pad, or connector contact is visibly damaged, stop and escalate before continuing.

## Required Bench Setup

- Stereo microscope with enough clearance for hot air and tweezers
- ESD mat, wrist strap, ESD-safe chip tray, and grounded tools
- Hot-air station with small nozzles and controllable airflow
- Fine soldering iron with small chisel or hoof tips
- Fine tweezers, vacuum pickup, Kapton tape, and metal heat shields
- Flux: one tacky gel flux for rework and one liquid flux for touch-up/drag soldering
- Fresh solder braid in small widths; replace braid often during BGA cleanup
- Thin solder wire for hand work; choose an alloy that matches the intended process and rework plan
- Solvent and lint-free swabs for cleaning
- Continuity meter and bench supply with current limit

## WILD Board Fixturing

The WILD datalogger is not a flat board once one side is populated. Support strategy matters.

- Use a fixture with relief pockets for already-installed opposite-side parts.
- Never press the board flat onto a bench once either Omnetics connector is installed.
- Support the board close to the work area so local pressure does not flex the HDI stackup.
- During front-side Intan work, protect the nearby microSD socket and camera-side hardware from direct airflow.
- During back-side STM32 work, protect the HJ580 module and nearby 0201 parts from airflow and accidental bumping.

## Standard Build Order for New Boards

Recommended sequence for the current WILD datalogger:

1. Inspect bare boards for warpage, pad damage, solder-mask defects, and contamination.
2. Verify the exact board revision and pull the matching assembly files before placing parts.
3. Confirm dry-pack/MSL status for moisture-sensitive ICs. If the original packaging state is unknown, follow the component supplier guidance and current JEDEC moisture-control rules before heating the part.
4. First oven pass: assemble the back side first, centered around the STM32 side (`U$6`, HJ580, passives, regulators).
5. Inspect, clean bridges, and confirm the first side is mechanically stable.
6. Second oven pass: assemble the front side with the Intan BGA (`U$1`), microSD, camera-side components, and the remaining low-profile parts.
7. Leave both Omnetics connectors (`S1`, `S3`) out of the oven process unless a separate validated profile exists. Install them by hand after both oven passes.
8. Perform hand rework and final inspection only after both reflow passes are complete.

### Reflow Oven Guidance

- Use the solder paste vendor's profile as the starting point, not a generic toaster-oven recipe.
- Record the first successful profile with thermocouples on a setup board or process coupon.
- For WILD, confirm that the second pass does not shift first-pass small parts or remelt unsupported heavy parts.
- Keep time above liquidus and peak temperature as low as the paste vendor allows while still producing full wetting.
- If a board contains Omnetics connectors, do not send it back through an unvalidated full-board reflow cycle.

## General Local Rework Method

Use this sequence for all local hot-air work on populated boards:

1. Photograph the area before touching it.
2. Fixture the board so it cannot sag or rock.
3. Warm the area gradually before pushing to reflow temperature. The goal is to reduce thermal shock and avoid excessive top-side dwell.
4. Shield nearby plastics and small passives with foil or a dedicated heat shield.
5. Apply flux before heating.
6. Use the smallest nozzle that still heats the full package evenly.
7. Keep airflow low enough that adjacent 0201 parts do not move.
8. Remove the part only when it lifts freely.
9. Clean and inspect pads before placing the replacement.
10. After rework, clean residue, inspect under microscope, then do electrical checks before power.

## STM32 UFQFPN48 SOP

The WILD datalogger uses `U$6`, a 48-lead `UFQFPN48` STM32 package on the back side of the board.

### Removal

1. Support the front side so no installed connector or socket is carrying load.
2. Start with a gentle warm-up pass over the package area.
3. Add flux around all four sides of the QFN and around the exposed-pad area through thermal relief where accessible.
4. Heat evenly with hot air until the package releases without force.
5. Lift vertically. Do not lever under an edge with tweezers.

### Pad Cleanup

1. Flood pads with flux.
2. Wick the perimeter pads flat with a small braid and minimal pressure.
3. Wick the exposed pad until it is flat and bright, but do not leave a large solder mound.
4. Clean, then inspect for lifted pads, solder-mask damage, or bridged vias in the thermal region.

### Replacement

1. Apply a controlled amount of paste or solder to the perimeter pads.
2. Keep solder volume on the exposed pad low. Too much solder on the center pad will float the package and open the perimeter leads.
3. Place the new STM32 with correct `Pin 1` orientation under microscope.
4. Reflow with controlled local top heat until the package settles and self-centers.
5. Touch up only if needed. If drag soldering is required, use liquid flux and minimal solder.

### Acceptance

- Package sits flat with no visible rock
- All four sides are centered and evenly wetted
- No bridges between side pads
- Exposed pad is soldered to ground as required by the package guidance
- No short from rail to ground before power-up

### References

- [QFN hot-air rework tutorial](https://www.youtube.com/watch?v=kmX_oLPy08o)

## Omnetics A79025 Microsoldering SOP

The WILD board uses Omnetics `A79025` connectors on both sides (`S1` on the front, `S3` on the back). This is the highest-risk manual connector step on the board.

### Process Rules

- Treat the connector as a hand-soldered part, not a standard full-oven part.
- Keep connector-body temperature well below its approximate `220 C` housing limit.
- Use a fixture or alignment support so the connector mouth stays parallel to the board edge.
- Use an alloy and flux system compatible with the connector plating, board finish, and the rest of the assembly.

### Placement and Tacking

1. Inspect the footprint and confirm front/back orientation from the placement file before opening the connector tray.
2. Lightly flux the footprint.
3. Pre-tin one small anchor point or one corner pad only.
4. Place the connector under microscope and tack one end.
5. Re-check alignment against the footprint and board edge.
6. Tack the opposite end only after the first tack is confirmed correct.

### Soldering

1. Use a temperature-controlled iron and work in short dwell intervals so the connector body does not overheat.
2. Work with a fine tip and very small solder additions.
3. Solder from one side toward the other with fresh flux.
4. If drag soldering is used, keep the solder bead very small and clean the tip often.
5. Let the connector cool between rows if the body is heating up.
6. Clean bridges immediately while the connector is still firmly supported.

### Removal/Rework

- Do not pull on the connector shell.
- Reflow the full row or side evenly before lifting.
- If the body discolors, softens, or shifts, scrap the connector and inspect the PCB before trying again.

### Acceptance

- Connector mouth is parallel to the board edge and sits flat
- No visible bridges at the contact row
- No pad lifting at the ends
- Continuity passes on all channels that should connect
- No shorts between adjacent high-density contacts

### References

- [Omnetics FAQ](https://www.omnetics.com/about/faqs/)
- [Omnetics engineering support](https://www.omnetics.com/contact/engineering-support/)
- [Omnetics catalogs and part search](https://www.omnetics.com/resources/catalogs/)

## Intan RHD2164 BGA SOP

This is the most critical rework on the WILD board. The WILD datalogger uses `U$1`, an Intan `RHD2164` in a `104-pin`, `9 x 7 mm`, `0.5 mm pitch` BGA package.

### When to Reball Instead of Replace

Reball only if all of the following are true:

- The package body is flat and uncracked
- No corner is chipped
- No pad is visibly missing after cleanup
- The board failure points to a solder-joint problem, not confirmed silicon damage

If any of those checks fail, replace the chip instead of reballing it.

### Dry Handling and Bake

- Treat removed or previously opened Intan BGAs as moisture-sensitive.
- If the original bag label and floor-life state are available, follow that JEDEC/package handling information.
- If the original dry-pack state is unknown, follow the supplier handling guidance and current JEDEC moisture-control rules before reballing or reinstalling. Do not guess a high-temperature bake at the bench.
- After dry-out, keep the part in a dry box or sealed dry pack until the reball job starts.

### Board-Side Removal

1. Protect the nearby microSD socket and adjacent components.
2. Warm the area gradually before pushing to reflow temperature.
3. Apply flux around the BGA perimeter.
4. Heat evenly with low-to-moderate airflow until the package releases.
5. Lift vertically with vacuum pickup. Do not slide the chip sideways across softened pads.

### Board Pad Cleanup

1. Flood with flux.
2. Add a small amount of fresh solder only if needed to improve wetting.
3. Wick pads flat with light pressure.
4. Clean until the full site is flat, bright, and level.
5. Inspect every pad field edge for lifted mask or torn pads.

### Chip Underside Cleanup

This is the step most likely to damage a reusable BGA package.

1. Place the removed chip underside-up in a stable ESD fixture.
2. Preheat gently so the braid is not doing all the thermal work.
3. Apply flux across one small area only.
4. Use fresh braid and a flat tip. Touch, let the solder wick, then lift.
5. Move in short passes. Replace braid often.
6. If residue remains, re-flux and repeat. Do not push harder.
7. If the old solder is stubborn, add a very small amount of fresh compatible solder first, then wick again.
8. Never scrape pads with a blade, fiberglass pen, or dry braid.

The goal is to leave a clean, flat pad array without gouging the package metallization.

### Reballing With a Laser Stencil and Cold Alignment Tool

1. Clean the underside and verify the pad field is complete before loading the fixture.
2. Load the part into the cold alignment tool.
3. Apply only a very thin film of tacky flux. Excess flux causes ball float and bridges.
4. Align the laser stencil by the outer pad rows and corners under microscope.
5. Roll or brush the preformed solder balls into the stencil until every aperture is filled.
6. Remove stray balls before heating.
7. Reflow just long enough for each ball to wet and form. Do not over-dwell.
8. Cool without movement.
9. Inspect for missing balls, bridges, collapsed balls, and row skew before taking the chip out of the fixture.

### Reinstalling the Reballed Intan

1. Confirm the board site is flat and clean.
2. Apply a thin flux film to the board site.
3. Align the chip carefully to the pad grid under microscope.
4. Use controlled local top heat until the BGA self-centers and settles.
5. Do not press on the package while the solder is liquid.
6. Let the board cool undisturbed before cleaning.

### Acceptance

- Outer rows appear evenly collapsed and centered
- Package edges are parallel to the footprint
- No immediate VDD-to-GND short
- No obvious SPI-line shorts to neighbors
- If X-ray is available, use it on initial Intan rework runs and on any board with uncertain alignment

### Stop Conditions

Stop and escalate if any of these occur:

- Board pad lift
- BGA pad loss on the chip
- Package corner crack or body warpage
- Multiple missing balls after two reball attempts
- Package will not self-center during reinstall

### References

- [Intan RHD2164 datasheet](https://intantech.com/files/Intan_RHD2164_datasheet.pdf)
- [Intan RHD2000 series datasheet](https://intantech.com/files/Intan_RHD2000_series_datasheet.pdf)
- [IPC/JEDEC J-STD-020E table of contents](https://www.ipc.org/TOC/IPC-JEDEC-J-STD-020E.pdf)
- [BGA reflow tutorial](https://www.youtube.com/watch?v=cTfuaNqfPKI)

## Final Inspection and Power-Up

Before first power after any major rework:

1. Inspect the reworked area at full microscope zoom.
2. Measure resistance from each main rail to ground and compare with a reference board when one is available.
3. Confirm there is no obvious bridge at the STM32, Intan, or Omnetics site.
4. Clean flux residue well enough that later inspection is still possible.
5. Bring the board up on a current-limited supply or USB power meter before normal use.
6. Record the operator, date, board revision, alloy used, rework type, and any replaced part lot.

## Training Materials

Use these before doing independent rework on a live WILD board:

- Read the WILD assembly files for orientation and exact reference designators.
- Read the STM32 package section before replacing `U$6`.
- Read the Intan package and footprint pages before reballing `U$1`.
- Review the linked QFN and BGA tutorial videos before using hot air on dense boards.
- Pull the current Omnetics drawing or catalog entry before touching `S1` or `S3`, especially if the connector lot or finish has changed.

For training, the first Intan reball and the first Omnetics hand-solder job should be done on practice boards or spare parts before touching a live WILD board.
