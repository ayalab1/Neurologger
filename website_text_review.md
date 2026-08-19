# WILD Website Text Review Bundle

Generated from the current MkDocs Markdown source files, including the navigation pages and the built PCB compatibility page.

Use the Source file marker before each section to map edits back to the website source.

---

# Source file: docs/index.md

---
hide:
  - navigation
  - toc
---

<section class="wild-hero">
  <div>
    <h1>WILD: wireless neural recording and closed-loop control in freely behaving animals</h1>
    <p>The Wireless, Interactive, Lightweight Datalogger (WILD) is a head-mounted local-storage platform for neural recording and closed-loop experimental control in freely behaving small animals. The system combines multichannel electrophysiology, auxiliary behavioral sensing, onboard processing, microSD storage, wireless configuration, and validated stimulation workflows.</p>
    <div class="wild-actions">
      <a class="md-button md-button--primary" href="getting-started/">Get Started</a>
      <a class="md-button" href="hardware/">Hardware</a>
      <a class="md-button" href="software/">Software</a>
      <a class="md-button" href="https://github.com/ayalab1/Neurologger">GitHub</a>
    </div>
    <div class="wild-signal-row">
      <span>64-channel electrophysiology</span>
      <span>Closed-loop DSP</span>
      <span>9-axis IMU</span>
      <span>USV audio</span>
      <span>Head-mounted camera</span>
    </div>
  </div>
  <div class="wild-hero-media wild-hero-media-tiled">
    <figure class="wild-hero-tile wild-hero-device-tile">
      <img src="images/WIrelessEphys_Github_1_devicePicture.jpg" alt="WILD wireless modular datalogger device">
      <figcaption class="wild-caption">WILD integrates local neural recording, behavioral monitoring, embedded processing, microSD storage, wireless control, and stimulation support in a compact head-mounted platform.</figcaption>
    </figure>
    <figure class="wild-hero-tile wild-hero-ephys-tile">
      <img src="images/WIrelessEphys_Github_13_EphysSample.jpg" alt="Example WILD electrophysiology recording workflow">
    </figure>
  </div>
</section>

<section class="wild-section wild-paper-banner">
  <h2>Publication status</h2>
  <p>A manuscript describing the WILD platform is currently under review. This website provides public-facing documentation for the hardware files, release images, WILD Console usage, post-processing workflows, metadata requirements, and validation checks associated with the platform.</p>
  <p>Final peer-reviewed citation information will be added after publication. Until then, report the repository, release image, hardware revision, WILD Console version, SD card, battery, enabled modalities, and analysis scripts used for each dataset.</p>
</section>

<section class="wild-section">
  <h2>Why WILD?</h2>
  <div class="wild-grid wild-feature-grid">
    <div class="wild-card wild-feature-card">
      <h3>Naturalistic neuroscience</h3>
      <p>Record neural activity while animals explore, interact socially, vocalize, or move through large environments where tethered systems are limiting.</p>
    </div>
    <div class="wild-card wild-feature-card">
      <h3>Local full-resolution recording</h3>
      <p>High-bandwidth neural and multimodal data are written to onboard microSD storage. BLE is used for discovery, configuration, synchronization support, status, preview, and commands.</p>
    </div>
    <div class="wild-card wild-feature-card">
      <h3>Closed-loop control</h3>
      <p>Embedded DSP and curated TinyML workflows support device-local detection and responsive stimulation without requiring continuous full-bandwidth wireless streaming.</p>
    </div>
    <div class="wild-card wild-feature-card">
      <h3>Open and reproducible</h3>
      <p>Hardware files, release images, WILD Console installers, MATLAB scripts, Python tools, and documentation are organized for release-specific methods reporting.</p>
    </div>
  </div>
</section>

<section class="wild-section">
  <div class="wild-grid two">
    <div class="wild-card">
      <h2>Open-source release</h2>
      <p><strong>Included in the public workflow:</strong> 64-channel local-storage WILD recording, WILD Console on Windows, validated release images, optional stimulation workflows, and post-processing through documented MATLAB and Python scripts.</p>
      <p><strong>Not part of the stable public workflow:</strong> continuous full-bandwidth BLE telemetry, arbitrary runtime AI-model upload, and separate higher-performance research variants unless a release note explicitly supports them.</p>
    </div>
    <div class="wild-card">
      <h2>Reproducibility record</h2>
      <p>For each experiment, record the exact release tag, device image filename, hardware revision, WILD Console version, SD-card model, battery, enabled modalities, and analysis-script commit.</p>
      <p><a href="https://github.com/ayalab1/Neurologger/releases/latest">Open the latest GitHub release</a>.</p>
    </div>
  </div>
</section>

<section class="wild-zoom-tour" aria-label="WILD device zoom tour">
  <div class="wild-zoom-stage" aria-live="polite">
    <h2 id="wild-zoom-title">Embedded control at the headstage</h2>
    <p id="wild-zoom-description">The MCU, IMU, and board-level control electronics sit close to the animal, reducing external cabling while keeping timing-sensitive acquisition local.</p>
    <figure class="wild-zoom-frame">
      <img id="wild-zoom-device" src="images/WIrelessEphys_Github_1_devicePicture.jpg" alt="Real WILD device with labeled functional regions">
      <figcaption id="wild-zoom-label">MCU and IMU</figcaption>
    </figure>
  </div>

  <div class="wild-zoom-scroll" aria-hidden="true">
    <div class="wild-zoom-step" data-zoom-label="MCU and IMU" data-zoom-x="10%" data-zoom-y="10%" data-zoom-scale="2.5" data-zoom-title="Embedded control at the headstage" data-zoom-description="The MCU, IMU, and board-level control electronics sit close to the animal, reducing external cabling while keeping timing-sensitive acquisition local."></div>
    <div class="wild-zoom-step" data-zoom-label="microSD storage" data-zoom-x="10%" data-zoom-y="62%" data-zoom-scale="2.35" data-zoom-title="Local storage for long sessions" data-zoom-description="Recording data are stored on-device, preserving full-resolution datasets for long sessions while BLE remains available for control and preview."></div>
    <div class="wild-zoom-step" data-zoom-label="Amplifier and electrode pads" data-zoom-x="43%" data-zoom-y="82%" data-zoom-scale="2.3" data-zoom-title="Neural interface and probe connection" data-zoom-description="Amplifier and electrode-pad regions route neural signals into the acquisition stack for downstream preprocessing and analysis."></div>
    <div class="wild-zoom-step" data-zoom-label="External I/O and battery connectors" data-zoom-x="47%" data-zoom-y="0%" data-zoom-scale="2.5" data-zoom-title="Modular power and I/O" data-zoom-description="Connector regions support external modules, battery connection, synchronization, and configuration workflows for different experiment designs."></div>
    <div class="wild-zoom-step" data-zoom-label="Camera and microphone" data-zoom-x="115%" data-zoom-y="70%" data-zoom-scale="1.8" data-zoom-title="Behavioral sensing add-ons" data-zoom-description="Camera and microphone modules extend WILD from neural recording into synchronized neuro-behavioral datasets."></div>
  </div>
</section>

<section class="wild-section">
  <h2>Key capabilities</h2>
  <div class="wild-grid">
    <div class="wild-card">
      <h3>Electrophysiology</h3>
      <p>Up to 64 neural channels are recorded locally at release-configured sampling rates for downstream Intan-style analysis and spike sorting.</p>
    </div>
    <div class="wild-card">
      <h3>Behavioral sensing</h3>
      <p>IMU, digital inputs, ultrasonic audio, and head-mounted camera streams can be captured with the neural recording to support synchronized neuro-behavioral analysis.</p>
    </div>
    <div class="wild-card">
      <h3>Responsive stimulation</h3>
      <p>Onboard biomarker detection, DSP states, and curated model outputs can trigger stimulation in supported release images and validated hardware configurations.</p>
    </div>
    <div class="wild-card">
      <h3>Multi-device experiments</h3>
      <p>External I/O, digital events, and post-export alignment workflows support experiments involving multiple WILD devices and behavioral systems.</p>
    </div>
    <div class="wild-card">
      <h3>Field-ready workflow</h3>
      <p>Local storage, low-power modes, and wireless control make the platform suitable for long recordings and outdoor or large-arena experiments after bench validation.</p>
    </div>
  </div>
</section>

<section class="wild-section">
  <h2>Specification summary</h2>
  <p>The public WILD workflow combines local high-bandwidth recording with low-bandwidth wireless control and documented post-processing.</p>
  <div class="wild-spec-strip" aria-label="WILD device specification strip">
    <span>64 channels</span>
    <span>~1.48 g logger</span>
    <span>microSD logging</span>
    <span>BLE control</span>
    <span>9-axis IMU</span>
    <span>USV audio</span>
    <span>Head-mounted camera</span>
    <span>Closed-loop DSP</span>
  </div>
  <p class="wild-table-note">Mode-dependent power values are listed in <a href="hardware/power/#measured-device-modes">Hardware / Power</a>, together with release image, SD-card, and battery metadata.</p>
</section>

<section class="wild-section">
  <h2>System overview</h2>
  <p>A typical session moves from animal-mounted acquisition to local storage, synchronization, export, and offline analysis.</p>
  <div class="wild-system-panels">
    <div class="wild-flow wild-flow-vertical" aria-label="WILD system overview">
      <div>Animal</div>
      <div>Device</div>
      <div>Sensors</div>
      <div>Storage</div>
      <div>Synchronization</div>
      <div>Analysis</div>
    </div>
    <figure class="wild-image-frame wild-system-figure">
      <img src="images/WIrelessEphys_Github_2_deviceDiagram.jpg" alt="WILD system diagram">
    </figure>
  </div>
</section>

<section class="wild-section">
  <h2>Research highlights</h2>
  <div class="wild-grid wild-highlight-grid">
    <div class="wild-card wild-highlight-card">
      <h3>Outdoor recordings</h3>
      <p>Local storage and wireless control support electrophysiological recordings in large naturalistic environments where tethering is impractical.</p>
    </div>
    <div class="wild-card wild-highlight-card">
      <h3>Multi-animal experiments</h3>
      <p>Wired sync, BLE calibration, and clock-correction workflows coordinate multiple WILD devices and external behavioral systems.</p>
    </div>
    <div class="wild-card wild-highlight-card">
      <h3>Closed-loop ripple detection</h3>
      <p>DSP filters support ripple-band detection for responsive hippocampal experiments in validated release images.</p>
    </div>
    <div class="wild-card wild-highlight-card">
      <h3>Theta phase stimulation</h3>
      <p>Hilbert and filter modes support phase-aware online control designs when enabled and validated for the release image.</p>
    </div>
    <div class="wild-card wild-highlight-card">
      <h3>Multimodal behavior</h3>
      <p>IMU, video, USV, and neural data support studies of social interaction, navigation, and freely expressed behavior.</p>
    </div>
    <div class="wild-card wild-highlight-card">
      <h3>Open hardware iteration</h3>
      <p>The repository contains hardware files, validated release assets, WILD Console installers, and analysis scripts.</p>
    </div>
  </div>
</section>

<section class="wild-section wild-quickstart">
  <h2>Quick start</h2>
  <div class="wild-grid two">
    <div class="wild-card wild-step">
      <h3>1. Hardware setup</h3>
      <p>Check connectors, battery polarity, microSD card, probe cabling, sensor cabling, and the release image before the first recording.</p>
    </div>
    <div class="wild-card wild-step">
      <h3>2. Data acquisition</h3>
      <p>Use WILD Console for BLE connection, synchronization support, recording configuration, closed-loop settings, and local logging startup.</p>
    </div>
    <div class="wild-card wild-step">
      <h3>3. Data analysis</h3>
      <p>Convert exported recordings into Intan-style files, IMU outputs, camera or audio media, and spike-sorting inputs.</p>
    </div>
  </div>
  <p class="wild-table-note">Recommended first target: complete a short bench recording, export the folder, confirm `amplifier.dat`, `analogin.dat`, `time.dat`, `info.rhd`, and `CE_params.bin`, then open the result with the documented MATLAB or Python workflow.</p>
</section>

<section class="wild-section">
  <h2>Publication and citation</h2>
  <div class="wild-grid two">
    <div class="wild-card">
      <h3>Publication status</h3>
      <p>A manuscript describing the WILD platform is currently under review. Final citation information, including journal, DOI, volume, issue, and page details, will be added only after publication.</p>
    </div>
    <div class="wild-card">
      <h3>Repository and release versions</h3>
      <p>Before publication, cite the repository or release record and report the hardware revision, release image, WILD Console version, SD-card model, battery configuration, enabled modalities, and analysis scripts used for each dataset.</p>
    </div>
  </div>

```bibtex
@misc{wild_repository,
  title        = {WILD: Wireless, Interactive, Lightweight Datalogger},
  author       = {Zhao, Zifang and Chang, Hongyu and Paudel, Praveen and Park, Jaehyo and Liu, Can and Aurelio, Maria Q. and Oliva, Azahara and Fernandez-Ruiz, Antonio},
  howpublished = {\url{https://github.com/ayalab1/Neurologger}},
  note         = {Public documentation and release files. Report the release tag, device image, hardware revision, WILD Console version, and analysis-script commit used in the experiment.},
  year         = {2026}
}
```
</section>

<section class="wild-section">
  <h2>Community</h2>
  <div class="wild-grid">
    <div class="wild-card">
      <h3>GitHub</h3>
      <p><a href="https://github.com/ayalab1/Neurologger">Browse hardware files, release images, WILD Console installers, and analysis scripts.</a></p>
    </div>
    <div class="wild-card">
      <h3>Issues and pull requests</h3>
      <p>Use GitHub issues for reproducible bugs or documentation gaps, and pull requests for tested fixes, examples, and compatibility updates.</p>
    </div>
    <div class="wild-card">
      <h3>Contributing</h3>
      <p><a href="contributing/">Issues and pull requests are appropriate for hardware notes, analysis examples, compatibility reports, and documentation improvements.</a></p>
    </div>
  </div>
</section>

---

# Source file: docs/getting-started/index.md

# Getting Started

Start here if you are setting up a WILD device for the first time or reproducing the public WILD workflow associated with the manuscript under review.

<div class="wild-grid wild-nav-grid">
  <a class="wild-card wild-card-link wild-card-compact" href="hardware-setup/">
    <div>
      <h3>Hardware setup</h3>
      <p>Prepare device</p>
    </div>
  </a>
  <a class="wild-card wild-card-link wild-card-compact" href="data-acquisition/">
    <div>
      <h3>Data acquisition</h3>
      <p>Record and export</p>
    </div>
  </a>
  <a class="wild-card wild-card-link wild-card-compact" href="data-analysis/">
    <div>
      <h3>Data analysis</h3>
      <p>Convert outputs</p>
    </div>
  </a>
</div>

## Before You Begin

- Windows 10 or later for WILD Console.
- Bluetooth 4.0 or later for BLE control.
- A tested microSD card installed in the WILD device. Use Samsung EVO series or tested Lexar cards for first recordings; some SanDisk cards may be unreliable in specific WILD operating modes.
- A battery that can boot the WILD device and sustain the planned recording mode.
- The correct release image for the hardware revision and experiment mode.
- Windows clock preparation before timing-sensitive recordings: disable automatic time setting and automatic time-zone changes, run `TimeCalibrator`, and keep the calibration record for post-export offset and drift correction.

!!! warning "Recording prerequisite: Windows clock preparation"
    WILD logger time and PC system time are independent clocks. Windows automatic time correction can create an abrupt PC-clock step during a session, so timing-sensitive recordings should not begin until Windows automatic time adjustment is disabled and `TimeCalibrator` has been run.

## Repository Assets

| Asset | Location |
| --- | --- |
| Prebuilt device images | [Latest GitHub release](https://github.com/ayalab1/Neurologger/releases/latest) |
| WILD Console installers | [Latest GitHub release](https://github.com/ayalab1/Neurologger/releases/latest) |
| Fabrication guide | [Fabrication](../fabrication.md) |
| Board source files | [PCB](https://github.com/ayalab1/Neurologger/tree/main/PCB) |
| 3D-print files | [3Dprint](https://github.com/ayalab1/Neurologger/tree/main/3Dprint) |
| MATLAB and Python analysis scripts | [Code](https://github.com/ayalab1/Neurologger/tree/main/Code) |

!!! tip "Recommended first path"
    Start with Hardware Setup, then Data Acquisition, then Data Analysis. The SD card, battery, release image, and boot state provide the baseline for troubleshooting.

## First Successful Dry Run

**Goal:** complete a short bench recording and confirm that the public export and analysis path works before changing advanced settings.

**Minimum output folder:**

- `amplifier.dat`
- `analogin.dat`
- `time.dat`
- `info.rhd`
- `CE_params.bin`

**Optional outputs, depending on mode:**

- `adc.dat` for audio workflows.
- `misc.dat` for camera workflows.
- `device_event.*.evt` files after MATLAB preprocessing.
- `IMU.mat` after IMU processing.

**Success criteria:**

1. The WILD device is discovered and connected in WILD Console.
2. A short recording starts and stops cleanly.
3. SD-card export completes without missing-core-file errors.
4. The exported folder contains the expected files for the recording mode.
5. MATLAB or Python post-processing runs without manual file repair.

---

# Source file: docs/getting-started/hardware-setup.md

# Hardware Setup

This checklist covers the physical preparation required before connecting WILD to software.

## Device and Connectors

Connector images identify the stimulation, auxiliary, sensor, sync, and battery interfaces before board power-up.

![WILD connector overview](../images/WIrelessEphys_Github_8_connectors.jpg){ .wild-readable-figure }

The connector-orientation schematic below highlights the power, ground, and USB data-line positions used during the first hardware check. Confirm the actual connector orientation on the assembled WILD device before applying power.

![WILD connector orientation schematic](../images/WIrelessEphys_Github_6_Connectors.jpg){ .wild-readable-figure }

The WILD device uses these connectors to support local acquisition, storage, wireless control, synchronization, and optional stimulation in a compact head-mounted unit. Before powering the device, confirm which interfaces are required for the planned session.

## Preparation Checklist

1. Inspect the PCB and connectors for visible damage.
2. Confirm battery polarity on the JST-SH connector.
3. Insert a tested microSD card.
4. Confirm the recording probe or flexible-probe connection.
5. Confirm the stimulation output, IMU, microphone, camera, and external-sync connections needed for the experiment.
6. Confirm that the device is running the expected released device image.

## New Device Bring-Up

For a newly assembled WILD device, complete the bench power and programming checks before connecting a battery or probe.

1. Prepare a 4-pin JST-SH 1.0 mm cable using the connector orientation shown above. Pre-crimped JST-SH wire leads or equivalent lab-qualified leads can be used if the polarity and pin order are verified before connection.
2. Build a USB-to-4-pin test cable by soldering the JST-SH lead to a USB-A connector breakout or equivalent bench-supply adapter. Record the adapter type in local bring-up notes if the cable will be reused.
3. Insert a USB power meter between the PC or USB supply and the WILD device.
4. Connect the WILD device without a battery and check the first current reading. A current below 10 mA is the expected first-pass check for no obvious short circuit.
5. If current is higher than expected, disconnect the device and inspect the power connector, regulator area, microSD socket, and fine-pitch solder joints before continuing.
6. To enter DFU mode, momentarily short the DFU mode-select pin to VCC with metal tweezers while connecting USB. Use the connector schematic and avoid shorting adjacent pins.
7. Confirm that Windows detects an `STM32-DFU` device. If needed, install the ST DFU driver from [STSW-STM32080](https://www.st.com/en/development-tools/stsw-stm32080.html).
8. In WILD Console, open the **Advanced** tab and use the DFU update pipeline to install the latest validated `.hex` release image for the hardware revision.
9. After the first release image is installed, later supported updates can be performed through microSD when the release notes document that path.

Detailed board-ordering, assembly, and special-component notes are listed in [Fabrication](../fabrication.md).

## microSD Card

Format the card from WILD Console before recording. The microSD card affects both reliability and power draw.

![microSD power comparison](../images/WIrelessEphys_Github_10_SDcard_power.jpg){ .wild-readable-figure }

Battery and SD-card guidance are part of the experimental protocol, not optional accessory setup.

![Battery examples](../images/WIrelessEphys_Github_9_batteries.jpg){ .wild-readable-figure }

## Released Device Image

Public setup documentation is based on prebuilt WILD release images. Each release image is matched to a hardware revision and experimental configuration. Keep the release tag and image filename in dataset metadata.

Release images are distributed through the [latest GitHub release](https://github.com/ayalab1/Neurologger/releases/latest). Follow the release notes for the specific image used in the experiment.

Treat the released device image as part of the hardware setup record. The WILD device, WILD Console version, SD card, battery, and analysis scripts should all be traceable to the dataset.

---

# Source file: docs/getting-started/data-acquisition.md

# Data Acquisition

WILD Console is the main operator interface for BLE connection, synchronization support, recording configuration, selected live preview, closed-loop control, and SD-card export.

WILD is a local-storage neurologger. High-bandwidth neural and multimodal data are written to the device microSD card. BLE supports setup, timing calibration, status messages, command delivery, and selected preview signals.

The current public workflow uses WILD Console on Windows. Full-resolution recordings are recovered from the microSD card.

The WILD device does not need continuous full-bandwidth wireless streaming to preserve the recording. Keep BLE connected when the session needs online preview, parameter changes, or live commands. Otherwise, use BLE for setup and timing coordination, then recover the full dataset from the microSD card.

!!! tip "First-session scope"
    For the first dry run, use only the minimum path needed to connect, record, stop, and export. Leave closed-loop, camera, stimulation, GPIO, and advanced parameter panels unchanged until the basic local-storage workflow is confirmed.

## Recording Prerequisites

Before starting a recording that will use PC-side timestamps, USB-GPIO triggers, camera triggers, behavior events, or multi-device alignment:

1. Disable Windows automatic time setting and automatic time-zone changes.
2. Run `TimeCalibrator` on the acquisition computer.
3. Keep the TimeCalibrator record with the WILD export.
4. Use the calibration record after export to correct PC-to-WILD time offset and drift.

WILD and the PC use independent clocks. Windows automatic time management does not calibrate PC drift against the WILD device clock, and an automatic correction can introduce an abrupt shift between PC time and WILD logger time during a recording.

## First-Pass Button Path

For a basic dry run, only a small part of WILD Console is needed. The closed-loop, camera, stimulation, GPIO, and advanced parameter panels can stay unchanged until the device connects, synchronizes, records, and exports reliably.

The online and live figures below are runtime screenshots from WILD Console during hardware sessions. Some window-title or device-list strings may still show older internal labels; the public documentation uses the WILD name consistently.

<div class="wild-operator-path">
  <div class="wild-operator-step">
    <span class="wild-button-label">Online tab</span>
    <p><strong>Connect to the device.</strong> Work in the Online tab for BLE discovery, synchronization, preview, and recording control.</p>
  </div>
  <div class="wild-operator-step">
    <span class="wild-button-label">Rescan</span>
    <p><strong>Find nearby WILD devices.</strong> If the device is not listed, rescan after confirming power, battery, and BLE status.</p>
  </div>
  <div class="wild-operator-step">
    <span class="wild-button-label">Device List</span>
    <p><strong>Select the target logger.</strong> The selected row updates device information, available space, and connection state.</p>
  </div>
  <div class="wild-operator-step">
    <span class="wild-button-label primary">Connect</span>
    <p><strong>Open the BLE session.</strong> Wait for device-state and synchronization labels before starting a recording.</p>
  </div>
  <div class="wild-operator-step">
    <span class="wild-button-label primary">Recording Start</span>
    <p><strong>Start local logging.</strong> Full-resolution data are written to the device microSD card, not streamed continuously over BLE.</p>
  </div>
  <div class="wild-operator-step">
    <span class="wild-button-label">Recording Stop</span>
    <p><strong>Close the recording cleanly.</strong> Stop before removing power or taking out the microSD card.</p>
  </div>
  <div class="wild-operator-step">
    <span class="wild-button-label">Offline tab</span>
    <p><strong>Move to export after recording.</strong> Insert the microSD card into the PC and use the Offline tab for SD-card download.</p>
  </div>
  <div class="wild-operator-step">
    <span class="wild-button-label primary">Save to Disk</span>
    <p><strong>Export the recording folder.</strong> WILD Console decodes the SD-card recording into public analysis files.</p>
  </div>
</div>

<p class="wild-operator-note">
  First-session rule: ignore <span class="wild-button-label muted">Closed-loop</span>, <span class="wild-button-label muted">Camera</span>, <span class="wild-button-label muted">Advanced</span>, and parameter panels until a simple recording can be started, stopped, and exported.
</p>

![WILD Console runtime screenshot during connected acquisition](../images/WIrelessEphys_Github_4_onlineAPI.jpg){ .wild-readable-figure }

## Connect

1. Launch `WILD Console_EXE_PLACEHOLDER`.
2. Click <span class="wild-button-label">Rescan</span> if the device is not listed.
3. Select the WILD device from <span class="wild-button-label">Device List</span>.
4. Click <span class="wild-button-label primary">Connect</span> and wait for device-state and synchronization updates.
5. Confirm that the TX/RX indicators update. If BLE connects but synchronization does not start, check the microSD card and timing setup first.

**Expected signs of success**

- The device appears in the list after rescan.
- The selected row updates device information.
- Connect succeeds without repeated disconnect loops.
- Status labels move past discovery into an active connected state.

## Record

1. Confirm the expected sampling speed and channel count.
2. Leave closed-loop and camera settings unchanged unless the experiment needs them.
3. Use `Wait for start` only when coordinating multiple WILD Console instances.
4. Click <span class="wild-button-label primary">Recording Start</span> to begin local logging on the device.
5. Click <span class="wild-button-label">Recording Stop</span> when the experiment is complete.

**Expected signs of success**

- Recording starts without a parameter or storage error.
- Session duration advances normally.
- The device remains stable through stop.
- Exported files are longer than a trivial test header and have plausible file sizes.

## Field Validation

Before a field or multi-animal session, run a short dry run with every WILD device that will be used.

1. Connect each device and confirm it reaches the synchronized state.
2. Start and stop a short local recording on each device.
3. Export the SD card immediately and confirm expected duration, file size, and representative channels.
4. Use the release image specified for the experiment and hardware revision.
5. For multi-device sessions, repeat the dry run with the planned start order, external I/O wiring, and any camera or TTL sync source attached.

A connected or synchronized GUI state is not proof that data were written for the full intended duration. Treat unexpectedly short files, stalled file-size indicators, unstable offsets, frame drops, or missing channels as blockers before animal recordings.

## Live Display

The real-time view supports selected neural previews, auxiliary signals, IMU and DSP states, stimulation triggers, and threshold overlays. See [Live Visualization](../software/live-visualization.md) for the runtime preview screenshot and display controls.

Disable preview to save power and BLE bandwidth when live monitoring is not needed. Recover full-resolution recording data from the microSD card after the session.

## Download

Insert the SD card and click <span class="wild-button-label primary">Save to Disk</span>. WILD Console exports recordings into timestamped folders with neural, auxiliary, ADC, camera, metadata, and parameter files.

**Expected signs of success**

- Export finishes without stopping at the first folder.
- Core files appear in the export folder: `amplifier.dat`, `analogin.dat`, `CE_params.bin`.
- `time.dat` and `info.rhd` are generated by the documented preprocessing path.
- Optional `adc.dat` and `misc.dat` appear only when audio or camera were enabled.

<p class="wild-operator-note">
  Normal export does not require the red maintenance buttons. <span class="wild-button-label warning">Quick Format</span> is only for preparing a known empty card, and release-image maintenance is outside the routine recording path.
</p>

![WILD Console runtime screenshot of the offline export workflow](../images/WIrelessEphys_Github_5_offlineAPI.jpg){ .wild-readable-figure }

## Common Blockers

| Symptom | First check |
| --- | --- |
| Device not listed after rescan | Battery, power, BLE discoverability, and whether the expected release image is installed. |
| BLE connects but synchronization never settles | microSD presence, timing wiring, and whether the session is waiting on external sync conditions. |
| Recording starts but export is unexpectedly short | Stop sequence, SD-card write stability, battery stability, and whether the device remained powered through the run. |
| Export folder is missing expected files | Recording mode, whether audio or camera were enabled, and whether preprocessing was run after export. |
| Preview is unstable | Treat preview as optional; confirm local recording first, then revisit BLE preview bandwidth or adapter quality. |

---

# Source file: docs/getting-started/data-analysis.md

# Data Analysis

The repository includes MATLAB and Python scripts for converting WILD recordings into analysis-ready files.

## Standard Post-processing

1. Export data from WILD Console.
2. Run `WILD_PreProcess.m` to generate corrected headers and timing files.
3. Run `WILD_processIMU.m` for IMU processing.
4. Run `WILD_VideoDecodewAudio.py` or the folder variants for camera recordings.
5. Prepare spike-sorting inputs from `amplifier.dat`.

## Script Locations

| Task | Script |
| --- | --- |
| Header and time correction | [`WILD_PreProcess.m`](https://github.com/ayalab1/Neurologger/blob/main/Code/WILD_PreProcess.m) |
| IMU processing | [`WILD_processIMU.m`](https://github.com/ayalab1/Neurologger/blob/main/Code/WILD_processIMU.m) |
| Camera decode with audio | [`WILD_VideoDecodewAudio.py`](https://github.com/ayalab1/Neurologger/blob/main/Code/WILD_VideoDecodewAudio.py) |
| Intan header generation | [`WILD_genIntanHeader.m`](https://github.com/ayalab1/Neurologger/blob/main/Code/WILD_genIntanHeader.m) |

!!! note
    Keep raw SD exports unchanged. Run conversion scripts into a separate analysis folder so the original recording can be reprocessed if metadata or analysis settings change.

---

# Source file: docs/hardware/index.md

# Hardware

Hardware documentation covers the WILD device architecture, onboard sensors, optional modules, mechanical assembly, power choices, and release-image compatibility for supported device revisions.

<div class="wild-grid wild-nav-grid">
  <a class="wild-card wild-card-link wild-card-compact" href="device-overview/">
    <div>
      <h3>Device</h3>
      <p>Architecture</p>
    </div>
  </a>
  <a class="wild-card wild-card-link wild-card-compact" href="onboard-sensors/">
    <div>
      <h3>Sensors</h3>
      <p>IMU, audio, camera</p>
    </div>
  </a>
  <a class="wild-card wild-card-link wild-card-compact" href="opto-module/">
    <div>
      <h3>Opto module</h3>
      <p>Stimulation</p>
    </div>
  </a>
  <a class="wild-card wild-card-link wild-card-compact" href="usb-board/">
    <div>
      <h3>USB-GPIO</h3>
      <p>Trigger alignment</p>
    </div>
  </a>
  <a class="wild-card wild-card-link wild-card-compact" href="mechanical/">
    <div>
    <h3>Mechanical</h3>
      <p>Mounts and enclosures</p>
    </div>
  </a>
  <a class="wild-card wild-card-link wild-card-compact" href="power/">
    <div>
    <h3>Power</h3>
      <p>Battery, SD, mode power</p>
    </div>
  </a>
  <a class="wild-card wild-card-link wild-card-compact" href="https://github.com/ayalab1/Neurologger/releases/latest">
    <div>
      <h3>Releases</h3>
      <p>Firmware images</p>
    </div>
  </a>
</div>

## Hardware Asset Map

| Hardware area | Repository path |
| --- | --- |
| Datalogger PCB | [PCB/Datalogger](https://github.com/ayalab1/Neurologger/tree/main/PCB/Datalogger) |
| Opto module | [PCB/Optomodule](https://github.com/ayalab1/Neurologger/tree/main/PCB/Optomodule) |
| Camera unit | [PCB/CameraUnit](https://github.com/ayalab1/Neurologger/tree/main/PCB/CameraUnit) |
| USB-GPIO board | [PCB/USBboard](https://github.com/ayalab1/Neurologger/tree/main/PCB/USBboard) |
| USB-GPIO firmware | [Firmware/WILD_USB_GPIO.hex](https://github.com/ayalab1/Neurologger/blob/main/Firmware/WILD_USB_GPIO.hex) |
| Opto-module firmware | [Firmware/Opto_module.hex](https://github.com/ayalab1/Neurologger/blob/main/Firmware/Opto_module.hex) |
| Mechanical parts | [3Dprint](https://github.com/ayalab1/Neurologger/tree/main/3Dprint) |
| Battery table | [LipoBattery_selection.csv](https://github.com/ayalab1/Neurologger/blob/main/docs/LipoBattery_selection.csv) |
| Prebuilt device images | [Latest GitHub release](https://github.com/ayalab1/Neurologger/releases/latest) |

## Manufacturing Workflow

For a new WILD device build, start with the [Fabrication](../fabrication.md) page. The workflow covers fabrication parameters, PCBA submission, special components, manual soldering notes, and first DFU image installation.

## Compatibility Record

Document the following for each supported WILD hardware build or dataset:

| Record | Why it matters |
| --- | --- |
| PCB or device revision | Distinguishes connector layout, sensor options, and supported release images. |
| Release tag and image filename | Ties behavior to a reproducible device image. |
| WILD Console version | Defines the export and control tool behavior used in the session. |
| SD-card model | Affects boot reliability and long-session write stability. |
| Battery model | Affects boot margin, runtime, and peak-load behavior. |
| Enabled modalities | States whether the session used ephys, IMU, audio, video, stimulation, or external sync. |

## Bring-Up Checklist

Before the first recording on a newly assembled WILD device:

1. Confirm that the PCB revision, assembly files, and intended release image match.
2. Use the USB-to-4-pin bench cable and USB power meter for the first power check.
3. Install the first validated `.hex` release image through WILD Console **Advanced** DFU update.
4. Confirm battery polarity and boot stability.
5. Confirm microSD insertion and basic discoverability.
6. Confirm BLE connection and synchronization state.
7. Run a short recording, export it, and verify the expected files before any animal session.

## Visual Asset Policy

Public hardware visuals should be traceable to real WILD material: device photographs, CAD or EDA exports, microscope images, measured drawings, software screenshots, or hand-authored diagrams derived from measured hardware. AI-generated device photographs or device-like renders are excluded from WILD hardware documentation.

---

# Source file: docs/hardware/pcb.md

# Fabrication

The PCB manufacturing and assembly workflow is now documented as a top-level [Fabrication](../fabrication.md) page.

Use that page for board ordering parameters, PCBA files, special components, first DFU image installation, and post-assembly inspection.

---

# Source file: docs/hardware/device-overview.md

# Device Overview

The WILD device is a Wireless, Interactive, Lightweight Datalogger: an ultra-lightweight multimodal neurologger for freely behaving animals. It combines local neural data logging, auxiliary sensing, onboard processing, BLE configuration and status, synchronization support, and responsive stimulation.

WILD is not a BLE telemetry system. High-bandwidth neural data are recorded locally to microSD; wireless links are used for discovery, configuration, synchronization support, status, low-bandwidth preview, and online control commands.

The device is best understood as a head-mounted acquisition and control unit. Neural acquisition, auxiliary sensing, embedded processing, microSD storage, external I/O, and optional stimulation are handled on the device. The host computer or controller supports setup, timing coordination, status review, selected preview, command delivery, and export.

![WILD device](../images/WIrelessEphys_Github_1_devicePicture.jpg){ .wild-readable-figure }

## Functional Blocks

<div class="wild-functional-blocks" aria-label="WILD functional block diagram">
  <div class="wild-fb-group">
    <h3>Inputs</h3>
    <div class="wild-fb-card">Neural probe</div>
    <div class="wild-fb-card">IMU, audio, camera, and digital inputs</div>
  </div>
  <div class="wild-fb-group">
    <h3>Device core</h3>
    <div class="wild-fb-card wild-fb-primary">Analog front end</div>
    <div class="wild-fb-card wild-fb-primary">MCU and embedded DSP</div>
  </div>
  <div class="wild-fb-group">
    <h3>Local outputs</h3>
    <div class="wild-fb-card">microSD storage</div>
    <div class="wild-fb-card">Stimulation outputs</div>
  </div>
  <div class="wild-fb-group">
    <h3>Control links</h3>
    <div class="wild-fb-card">BLE control, sync, status, and preview</div>
    <div class="wild-fb-card">External I/O and USB setup</div>
  </div>
  <div class="wild-fb-group">
    <h3>Analysis</h3>
    <div class="wild-fb-card">MATLAB, Python, and spike sorting</div>
  </div>
</div>

## Device Interfaces

| Interface | Public documentation focus |
| --- | --- |
| Neural input | Standard 64-channel connector workflows and WILD flexible-probe workflows. |
| Local storage | Full-resolution neural, IMU, audio, camera, and event data are recorded to microSD. |
| Wireless control | BLE supports discovery, configuration, synchronization support, status checks, selected preview, and commands. |
| External I/O | Sync and control lines connect WILD devices with cameras, behavior systems, stimulation modules, and multi-device sessions. |
| Bench setup | USB and wired interfaces can support setup and validation workflows when high-bandwidth local monitoring is needed. |

Detailed sensor, opto-module, and USB-GPIO board notes are documented in the dedicated Hardware chapters.

## Current Public Scope

- The current open-source WILD release is the 64-channel local-storage neurologger workflow.
- Higher-performance Neuropixels-compatible and active-SPI-probe workflows are separate research and variant targets unless explicitly supported by a release note.
- Multi-device experiments use explicit synchronization workflows rather than BLE timestamps alone.
- Mass is configuration-dependent; report device-only mass, probe or module mass, battery mass, and complete implant mass separately.

## Core Specifications

| Category | Current WILD specification |
| --- | --- |
| Neural channels | 64 |
| Device mass | Approximately 1.5 g for the logger board, configuration-dependent |
| Board dimensions | 23.3 x 15.7 mm |
| MCU | Cortex-M4 class MCU for acquisition, embedded DSP, TinyML scheduling, and device control |
| Power input | Use release-specific approved battery options; internal rails are generated by a converter/regulator chain |
| Sampling rates | 1,250-20,000 Hz |
| ADC resolution | 16-bit electrophysiology and microphone data; 10-bit camera data |
| Storage | microSD, Class 10 or better; low-power endurance cards are preferred |
| Recommended cards | Samsung EVO older orange U1 generation or tested Lexar cards |
| BLE | Control link for discovery, synchronization support, status, selected preview, and low-bandwidth commands |
| IMU | BMX160, up to 500 Hz |
| USV microphone | SPH6611LR5H, approximately 1-80 kHz bandwidth |
| Camera | NanEye-C, 320 x 320 px, 16 FPS, 10-bit |
| Stimulation | Optional two-channel laser-diode stimulation module based on TPS6115x |

## Supported Modalities

- Local neural electrophysiology recording.
- Responsive closed-loop stimulation from onboard detection.
- IMU sensing and sensor fusion.
- Ultrasonic vocalization audio through ADC data.
- Head-mounted camera data through `misc.dat`.
- Digital input and synchronization signals.
- External I/O for synchronization, stimulation control, and integration with behavioral equipment.

---

# Source file: docs/hardware/onboard-sensors.md

# Onboard Sensors

WILD sensor documentation describes the auxiliary channels recorded alongside neural data or used during bench validation. Treat each enabled sensor as part of the experiment configuration: record the release image, sampling mode, connected module, and whether the output was used for monitoring, downstream analysis, or both.

## IMU

The WILD device includes a 9-axis motion-sensor path for acceleration, gyroscope, and magnetic-field measurements. IMU data support motion-state review, behavioral segmentation, artifact inspection, and alignment with external tracking systems.

For reproducible datasets, document:

- IMU enable state.
- Configured sample rate.
- Axis orientation relative to the mounted device.
- Whether IMU channels were exported in `analogin.dat` or processed into `IMU.mat`.
- Any filtering, resampling, or coordinate transform applied after export.

## Camera

The head-mounted camera path supports synchronized behavioral context when video is enabled in the release image and hardware configuration. Camera payloads are recovered from local device storage during export; live preview is used for setup and focus checks rather than as the full-quality data source.

Record the camera hardware, release image, frame rate, working distance, illumination condition, and decode script version with each dataset.

### Camera Imaging Profile

![Camera imaging profile at 4 mm, 8 mm, and 12 mm working distances](../images/WIrelessEphys_Github_11_cameraProfile.jpg){ .wild-readable-figure }

Use the camera imaging profile during bench setup to choose the working distance and focus range before mounting the camera module. Confirm the expected target distance with live preview before treating camera data as part of the experiment record.

## Microphone

The WILD microphone path uses the SPH6611LR5H ultrasonic microphone for vocalization and high-frequency acoustic recordings. In the documented WILD workflow, microphone data are sampled at 160 kHz, supporting acoustic content up to approximately 80 kHz when the analog path, sampling mode, and export settings are configured for USV recording.

The microphone ribbon assembly includes an embedded 10x amplifier stage before the logger input. Record whether the microphone ribbon and amplifier stage are present, the microphone orientation, and the release image used for acquisition.

For microphone sessions, document:

- Microphone model: SPH6611LR5H.
- Sampling rate: 160 kHz in the USV recording workflow.
- Approximate frequency response target: up to 80 kHz.
- Ribbon/amplifier configuration, including the embedded 10x amplifier stage.
- Audio export file path, decode script version, and any filtering or gain applied after export.

## Digital Inputs

Digital input channels are used for synchronization pulses, behavioral event markers, camera triggers, stimulation markers, or bench validation signals. These inputs are most useful when the same physical pulse is also recorded by the external system being aligned.

For timing-sensitive work, keep the raw digital input record and any generated event files. Post-export correction should preserve both raw event times and the corrected WILD-time event table.

---

# Source file: docs/hardware/opto-module.md

# Opto Module

The WILD opto-module is the optional stimulation hardware path for optical manipulation experiments. It should be documented separately from the logger board because supported stimulation behavior depends on the opto-module hardware, release image, wiring, and validation procedure.

## Hardware Assets

| Asset | Repository path |
| --- | --- |
| PCB fabrication files | [PCB/Optomodule/GerberFiles](https://github.com/ayalab1/Neurologger/tree/main/PCB/Optomodule/GerberFiles) |
| Drill files | [PCB/Optomodule/DrillFiles](https://github.com/ayalab1/Neurologger/tree/main/PCB/Optomodule/DrillFiles) |
| Assembly files | [PCB/Optomodule/Assembly](https://github.com/ayalab1/Neurologger/tree/main/PCB/Optomodule/Assembly) |
| Prebuilt module image | [Firmware/Opto_module.hex](https://github.com/ayalab1/Neurologger/blob/main/Firmware/Opto_module.hex) |

## Supported Operations

| Operation | Purpose | Validation |
| --- | --- | --- |
| Manual stimulation test | Confirm optical output and wiring before closed-loop use. | Use a bench photodiode, optical power meter, or other approved output check. |
| Closed-loop stimulation output | Deliver stimulation in response to WILD DSP, threshold, or TinyML state when supported by the release image. | Compare event files, stimulation markers, and observed output timing. |
| External trigger stimulation | Drive stimulation from external timing equipment when the experiment requires external coordination. | Record the external trigger and WILD event path for later alignment. |
| Parameter review | Confirm pulse width, intensity, channel selection, and enable state before the animal session. | Save release image, WILD Console version, and stimulation settings with the dataset. |

## Bring-Up Checks

Before connecting the opto-module to an animal-mounted device:

1. Confirm the opto-module PCB revision and compatible release image.
2. Inspect connector orientation and solder joints.
3. Confirm current limit, load, and optical output path on the bench.
4. Run a manual test pulse and record the measured output.
5. Confirm that stimulation markers are exported with the recording when stimulation timing is part of the analysis.

## Metadata

Record the opto-module PCB revision, release image, stimulation channel, optical output calibration, cable or connector path, and validation file. For closed-loop studies, also record the DSP or model state that generated each stimulation event.

---

# Source file: docs/hardware/usb-board.md

# USB Board

The WILD USB-GPIO board is used for bench validation, external trigger logging, and timing integration with behavioral equipment. It provides a PC-connected GPIO path that can be used alongside WILD BLE control and logger-side digital inputs.

![USB-GPIO board with IO0 to IO3 and USB interface labels](../images/WIrelessEphys_Github-12_GPIOBoard.jpg){ .wild-readable-figure }

## Interface

The board exposes four external I/O channels, labeled `IO0` to `IO3`, and a USB connection to the host PC. The USB path is useful for recording external events at the PC, generating TTL outputs for external equipment, and validating synchronization behavior before animal recordings.

## Fabrication Files

USB-GPIO board manufacturing files are available in [`PCB/USBboard`](https://github.com/ayalab1/Neurologger/tree/main/PCB/USBboard).

| File | Purpose |
| --- | --- |
| [`WILD_USB_2026-06-18.zip`](https://github.com/ayalab1/Neurologger/blob/main/PCB/USBboard/WILD_USB_2026-06-18.zip) | Gerber package for PCB fabrication. |
| [`WILD_USB.xlsx`](https://github.com/ayalab1/Neurologger/blob/main/PCB/USBboard/WILD_USB.xlsx) | BOM workbook for assembly review. |
| [`WILD_USB.csv`](https://github.com/ayalab1/Neurologger/blob/main/PCB/USBboard/WILD_USB.csv) | BOM export for manufacturer upload or quick inspection. |
| [`WILD_USB.brd`](https://github.com/ayalab1/Neurologger/blob/main/PCB/USBboard/WILD_USB.brd) and [`WILD_USB.sch`](https://github.com/ayalab1/Neurologger/blob/main/PCB/USBboard/WILD_USB.sch) | Eagle board and schematic source files. |

## Windows Host Timing

The WILD device and the Windows acquisition PC use independent clocks. Keeping those clocks aligned matters when USB-GPIO trigger timestamps are merged with WILD logger events. Windows automatic time management does not calibrate drift against WILD time, and an automatic time correction can create an abrupt shift between the PC clock and the device clock during a recording.

For best synchronization accuracy with the USB-GPIO board on Windows, disable automatic time setting and automatic time-zone changes before the recording session. Download and run `TimeCalibrator` on the acquisition computer, then use the calibration record to correct PC-to-WILD time offset and drift after export.

Keep the raw USB-GPIO log, TimeCalibrator record, and corrected timestamp table with the WILD export so the synchronization correction can be audited later.

## Supported Operations

| Operation | Direction | Use case | Timing note |
| --- | --- | --- | --- |
| TTL input logging | External equipment -> USB-GPIO -> PC | Log camera, behavior, or stimulation trigger edges at the host PC. | Host timestamps must be aligned to WILD logger time before merging with neural data. |
| TTL output control | WILD or PC command -> PC -> USB-GPIO -> external equipment | Drive external cameras, stimulation controllers, or behavior systems. | Output timing includes BLE, PC scheduling, USB transfer, and GPIO latency when WILD initiates the command. |
| Periodic sync pulses | Shared pulse to USB-GPIO and WILD digital input | Estimate offset and drift between PC timestamps and WILD logger time. | Use repeated pulses across the recording, not only a start pulse. |
| Bench validation | PC plus WILD test setup | Check wiring, edge polarity, event capture, and software logging before animal sessions. | Keep a short validation recording with raw USB log and WILD export. |

See [GPIO-through-USB Trigger Alignment](../software/data-format.md#gpio-through-usb-trigger-alignment) for timestamp correction, drift monitoring, and the difference between incoming and outgoing trigger paths.

## Recommended Use

1. Assign channel names before the session, for example `camera_trigger`, `behavior_event`, `sync_pulse`, or `stim_marker`.
2. Confirm input or output direction for each channel.
3. Run a short bench pulse train through the same path used by the experiment.
4. Save the USB-GPIO log together with the WILD export.
5. Apply post-export correction before combining PC-side trigger times with WILD neural, IMU, audio, camera, or stimulation events.

## Metadata

Record the USB-GPIO board revision, connected channels, edge polarity, host software, WILD Console version, BLE controller path, and whether the board was used for input logging, output control, or both.

---

# Source file: docs/hardware/mechanical.md

# Mechanical

Mechanical files are stored in the repository for baseplates, covers, and camera-related parts.

## Available Files

| Part | Repository path |
| --- | --- |
| Top cover | [WILD_TopCover.STL](https://github.com/ayalab1/Neurologger/blob/main/3Dprint/MouseCase/WILD_TopCover.STL) |
| Baseplate | [WILD_Baseplate.STL](https://github.com/ayalab1/Neurologger/blob/main/3Dprint/MouseCase/WILD_Baseplate.STL) |
| Camera baseplate | [WILD_Baseplate_Camera.STL](https://github.com/ayalab1/Neurologger/blob/main/3Dprint/MouseCase/WILD_Baseplate_Camera.STL) |
| Mouse-case CAD source | [Wild_Case.SLDPRT](https://github.com/ayalab1/Neurologger/blob/main/3Dprint/MouseCase/Wild_Case.SLDPRT) and [WILD_Case.f3d](https://github.com/ayalab1/Neurologger/blob/main/3Dprint/MouseCase/WILD_Case.f3d) |
| Cambridge NeuroTech probe assembly | [WILDAssemblyVertical.stp](https://github.com/ayalab1/Neurologger/blob/main/3Dprint/CambridgeNT_Probe/WILDAssemblyVertical.stp), [WILDAssemblyVertical.f3d](https://github.com/ayalab1/Neurologger/blob/main/3Dprint/CambridgeNT_Probe/WILDAssemblyVertical.f3d), and printable support parts in `3Dprint/CambridgeNT_Probe/` |

## Rendered Model Views

<div class="wild-grid two">
  <figure class="wild-image-frame">
    <img src="../images/WILD_Case_render.png" alt="Rendered WILD mouse case model with baseplate and cover">
    <figcaption class="wild-caption">Mouse-case render showing the baseplate and cover geometry for device protection and mounting.</figcaption>
  </figure>
  <figure class="wild-image-frame">
    <img src="../images/WILDAssemblyVertical_render.png" alt="Rendered vertical WILD assembly for Cambridge NeuroTech probe workflow">
    <figcaption class="wild-caption">Vertical assembly render for the Cambridge NeuroTech probe workflow, useful for checking model orientation and enclosure clearance. Courtesy of Adam Lowet, UC Berkeley.</figcaption>
  </figure>
</div>

## Design Considerations

- Keep cable strain relief and connector access visible in photos.
- Document orientation of the baseplate relative to probes, stimulation leads, and camera modules.
- Include printing material, layer height, support recommendations, and post-processing steps.
- Add animal-specific mounting notes only after local approval and protocol review.

---

# Source file: docs/hardware/power.md

# Power

Power planning is central to WILD experiments because battery selection, microSD choice, recording mode, selected preview signals, stimulation, and camera use all affect runtime.

## Power Architecture

The WILD device conditions the selected battery through a converter and regulator chain before powering analog acquisition, digital control, and peripheral subsystems. Battery choice should therefore follow validated hardware-revision guidance rather than a single nominal-voltage number.

Boot, microSD writes, camera use, stimulation, and live preview can create different peak-load conditions. Validate the complete experiment mode on the bench before using a battery in an animal session.

## Measured Device Modes

Representative WILD power modes are summarized here because they directly affect battery selection and runtime planning. Report these values together with the release image, SD-card model, battery, and enabled peripherals used for the session.

<div class="wild-spec-scroll" markdown="1">

| Power (mW) | Channels | Samples/s | Mode | Weight (g) | Video | Audio | Motion sensor | Processing |
| --- | ---: | ---: | --- | ---: | --- | --- | --- | --- |
| 0.11 | 0 | 0 | Sleep | 1.48 | - | - | - | - |
| 24.42 | 8 | 4.0e4 | Ephys, IMU | 1.48 | - | - | Accelerometer, gyroscope, magnetosensor | Spectral power, TinyML |
| 194.04 | 64 | 2.6e6 | Ephys, IMU | 1.48 | - | - | Accelerometer, gyroscope, magnetosensor | Spectral power, TinyML |
| 208.67 | 64 | 2.92e6 | Ephys, IMU, audio | 1.48 | - | 160 kHz ultrasonic audio path | Accelerometer, gyroscope, magnetosensor | Spectral power, TinyML |
| 245.28 | 64 | 5.48e6 | Ephys, IMU, audio, video | 1.48 | 320 x 320 px, 16 Hz | 160 kHz ultrasonic audio path | Accelerometer, gyroscope, magnetosensor | Spectral power, TinyML |

</div>

The table is a planning reference, not a substitute for bench validation. Current draw can change with release image, recording duration, SD-card behavior, battery voltage, preview state, stimulation, and camera configuration.

## Battery

The WILD device should use a battery approved for the specific hardware revision and experiment mode. Before mounting the device, confirm that the battery can boot the device without voltage sag, start recording, and sustain the planned sensors, preview state, stimulation, or camera load.

![Battery examples](../images/WIrelessEphys_Github_9_batteries.jpg){ .wild-readable-figure }

The [battery selection table](https://github.com/ayalab1/Neurologger/blob/main/docs/LipoBattery_selection.csv) provides baseline capacity and runtime-planning references.

## Battery Connector

The WILD device battery input uses the JST-SH connector orientation documented in the hardware setup workflow. The connector is part of the experiment record because polarity, strain relief, and repeated mating can affect boot stability and long-session reliability.

![WILD connector orientation schematic](../images/WIrelessEphys_Github_6_Connectors.jpg){ .wild-readable-figure }

For each hardware revision, record:

- Battery connector type and orientation.
- Battery lead polarity as assembled.
- Whether the battery lead was modified, extended, or re-terminated.
- Any strain-relief method used between the battery, enclosure, and animal-mounted device.

Before a recording session, inspect the connector shell, solder joints, and battery lead for looseness or damage. Do not infer polarity from wire color alone; confirm the assembled connector orientation against the WILD hardware revision being used.

See [Hardware Setup](../getting-started/hardware-setup.md) for the first-device connector checklist.

## First Power Check

For a newly assembled WILD device, use the USB-to-4-pin bench cable and a USB power meter before connecting a battery. The first connection should draw less than 10 mA before a release image is installed; a higher or unstable current is a stop condition for solder-bridge and connector inspection.

The DFU mode-select pin can be temporarily shorted to VCC during USB connection to expose the `STM32-DFU` device for first image installation. Use the connector schematic, keep the short temporary, and verify the pin orientation before applying power.

## Battery Charger

The charger is external to the WILD recording path. Charge batteries with a single-cell lithium-polymer charger that matches the battery chemistry, capacity, connector, and manufacturer charge-current rating. Record the charger model when reporting runtime or power stability results.

The WILD device should not be treated as a battery charger unless a specific hardware revision and release note explicitly documents charging support. For the public workflow, charge and inspect the battery before connecting it to the WILD device.

Recommended charger record:

| Item | Record |
| --- | --- |
| Charger model | Identifies the charge circuit used before the session. |
| Battery model and capacity | Links runtime to the actual energy source. |
| Charge current | Helps interpret battery heating, aging, and boot margin. |
| Connector adapter | Documents any JST-SH adapter or lead conversion used between the charger and battery. |
| Pre-session check | Confirms battery voltage, connector condition, and absence of swelling or cable damage. |

## microSD Power

The microSD card is part of the power and reliability budget. Format cards with WILD Console before recording, and validate the selected card in the same sampling and modality mode planned for the experiment.

![microSD power comparison](../images/WIrelessEphys_Github_10_SDcard_power.jpg){ .wild-readable-figure }

## Runtime Reporting

Runtime depends on sampling rate, active modalities, preview state, stimulation use, microSD-card model, battery capacity, and ambient conditions. When reporting a WILD recording mode, include:

- Release image.
- SD-card model and format state.
- Battery model and capacity.
- Sampling rate and enabled modalities.
- Preview state.
- Stimulation or TinyML state, if used.
- Session environment, especially for outdoor or long-duration recordings.

---

# Source file: docs/fabrication.md

# Fabrication

Board packages support ordering, assembly, and inspection for WILD hardware revisions.

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
| HJ580 BLE module | Confirm availability directly with HJ-SIP. The HJ-580XP-UART model can also be ordered directly from DigiKey: [HJ-580XP-UART](https://www.digikey.com/en/products/detail/hjsip/HJ-580XP-UART/29274934). Validated builds should use the HJ580 module firmware version documented for WILD, currently firmware version 20220507. |
| Omnetics A79025 | This 64-channel probe connector has a high failure rate during standard reflow. Prefer manual soldering with an alignment fixture, then inspect alignment, solder bridges, and continuity under magnification. |

## Post-Assembly Bring-Up

Prepare a bench cable before applying power to a newly assembled WILD device.

1. Solder the WILD 4-pin JST-SH 1.0 mm cable according to the connector illustration in [Hardware Setup](getting-started/hardware-setup.md).
2. Pre-crimped JST-SH 1.0 mm wire leads or equivalent lab-qualified leads can be used for cable preparation after verifying polarity and pin order.
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

---

# Source file: docs/software/index.md

# Software

Software documentation covers how to connect to the WILD device, configure recordings in WILD Console, export data from the device microSD card, and process recordings for analysis.

## Software Components

<div class="wild-grid wild-nav-grid">
  <a class="wild-card wild-card-link wild-card-compact" href="acquisition/">
    <div>
    <h3>WILD Console</h3>
      <p>Connect and export</p>
    </div>
  </a>
  <a class="wild-card wild-card-link wild-card-compact" href="artificial-intelligence/">
    <div>
      <h3>Embedded AI</h3>
      <p>Validated models</p>
    </div>
  </a>
  <a class="wild-card wild-card-link wild-card-compact" href="api-cli/">
    <div>
      <h3>Scripts</h3>
      <p>Batch tools</p>
    </div>
  </a>
  <a class="wild-card wild-card-link wild-card-compact" href="../analysis/">
    <div>
      <h3>Analysis</h3>
      <p>MATLAB and Python</p>
    </div>
  </a>
</div>

## Install

Download WILD Console from the [latest GitHub release](https://github.com/ayalab1/Neurologger/releases/latest). The link always opens the newest public WILD release.

## Public Workflow Boundary

The current public software path is:

1. Connect and record with WILD Console.
2. Export from the device microSD card.
3. Run documented MATLAB or Python post-processing.

The public documentation does not treat BLE as a continuous high-bandwidth acquisition path, and it does not promise a stable general-purpose SDK yet.

## Wireless Control

WILD Console remains the stable public control and export workflow for BLE discovery, connection, synchronization support, status checks, selected preview, low-bandwidth commands, and SD-card export.

Full-resolution WILD recordings remain local to the device microSD card and are exported after the session.

## Requirements

- Windows 10 or later.
- .NET Framework 4.8 for the current GUI.
- Bluetooth 4.0 or later.
- Administrator privileges for some SD-formatting workflows.

## Optional Runtime Files

- `dll_upfirdn.dll` for resampling.
- WILD BLE backend DLL for BLE support in older installer layouts.
- `ffmpeg.exe` for some camera processing workflows.

---

# Source file: docs/software/acquisition.md

# Acquisition

The acquisition workflow starts by connecting to a WILD device in WILD Console and ends with exported recording folders from the device microSD card.

The WILD device records high-bandwidth neural and multimodal data locally. BLE is used for discovery, synchronization support, configuration, status, selected preview, and control commands rather than continuous full-bandwidth data streaming.

WILD Console is the stable public control path today. Full-resolution recordings remain stored on the device microSD card.

## Recording Prerequisites

Before acquisition, prepare the Windows host clock for any timing-sensitive session. Disable automatic time setting and automatic time-zone changes, run `TimeCalibrator`, and keep the calibration record with the exported dataset. This is required when USB-GPIO logs, behavior timestamps, camera triggers, stimulation markers, or multi-device sessions will be aligned to WILD logger time.

The WILD device and PC use independent clocks. Windows automatic time management does not calibrate drift against the WILD clock and can create an abrupt PC-clock shift during the session if an automatic correction occurs.

## Wireless Connection Model

The WILD device can be connected for discovery, synchronization support, parameter review, selected preview, and command delivery. After setup and timing coordination, local recording does not depend on continuous full-bandwidth wireless streaming because the device writes the full dataset to microSD.

For multi-device sessions, keep continuous BLE connected only for devices that need live preview or online commands. Devices that only need local logging can be configured and synchronized first, then managed with the planned start, stop, external I/O, and export workflow.

## Routine Operation Map

Most first-session work uses four button groups:

| Stage | Main UI area | Primary click |
| --- | --- | --- |
| Discover and connect | Online tab, Device List | <span class="wild-button-label">Rescan</span>, select the WILD device, then <span class="wild-button-label primary">Connect</span> |
| Start recording | Online tab, recording controls | <span class="wild-button-label primary">Recording Start</span> |
| Stop recording | Online tab, recording controls | <span class="wild-button-label">Recording Stop</span> |
| Export data | Offline tab, File panel | <span class="wild-button-label primary">Save to Disk</span> |

Closed-loop settings, camera controls, stimulation parameters, GPIO options, and advanced panels are optional experiment-specific controls. Configure them after the basic connect-record-export path is working.

![WILD Console runtime screenshot of the offline export workflow](../images/WIrelessEphys_Github_5_offlineAPI.jpg){ .wild-readable-figure }

## Typical Session

1. Start WILD Console.
2. Complete Windows clock preparation if the session uses PC-side timestamps or external timing alignment.
3. Scan for the WILD device.
4. Connect over BLE.
5. Read current parameters and change experiment-specific settings only when needed.
6. Configure sampling, closed-loop, camera, stimulation, and GPIO options as needed.
7. Start local recording.
8. Monitor selected preview and state signals.
9. Stop recording.
10. Export data from the SD card.
11. Check duration, file size, representative channels, and sync markers before treating the session as complete.

## Multi-Device Sessions

Use one validated release set across all WILD devices. Assign the intended master/follower roles before the session, and keep the same wiring and start order used in the dry run.

For camera workflows, test the external TTL or camera sync path during a short continuous run and confirm that frames, event pulses, and recording files stay aligned before using the setup with animals.

After export, compare recording durations, file sizes, and sample counts across devices. A successful merge should show stable offsets across the recording; jumps, drifting offsets, or one logger ending early should be investigated before the data are treated as synchronized.

## Exported Files

The exported folder can include:

- `amplifier.dat`.
- `analogin.dat`.
- `digitalin.dat`.
- `adc.dat`.
- `misc.dat`.
- `supply.dat`.
- `time.dat`.
- `info.rhd`.
- WILD parameter binary.

See [Data Format](data-format.md) for field descriptions.

---

# Source file: docs/software/live-visualization.md

# Live Visualization

WILD Console includes real-time panels for checking selected neural channels, auxiliary signals, camera preview, closed-loop state, and device status during setup and recording.

Live visualization is a monitoring surface. Full-resolution neural, IMU, audio, camera, and event data are written locally to the device microSD card and recovered during export. Use the live view to confirm that the device is configured correctly, sensors are connected, and closed-loop state variables behave as expected.

![WILD Console runtime screenshot with live preview and camera windows](../images/WIrelessEphys_Github_4_onlineAPI.jpg){ .wild-readable-figure }

## What Live View Is For

Use live visualization to check:

- BLE connection state and TX/RX counters.
- Selected neural preview channels.
- Signal scale, saturation, obvious disconnects, and large motion artifacts.
- IMU and auxiliary-channel activity.
- Camera framing and gross focus during setup.
- DSP state, threshold crossings, stimulation state, and TinyML state messages when enabled.
- Recording state before handling the animal.

## Display Controls

- Select neural preview channels.
- Select auxiliary signals from IMU and DSP outputs.
- Adjust display length.
- Adjust display gain.
- Show threshold overlays when closed-loop channels are selected.
- Monitor internal state, stimulation triggers, DSP state, and TinyML state messages.

## Neural Preview

The neural preview is intended for fast quality checks, not for replacing exported electrophysiology data. Use it to confirm that the expected channels are active, that the signal is not saturated, and that preview gain and time scale are appropriate for the bench condition.

For recording sessions, treat exported `amplifier.dat`, event files, and metadata as the analysis source of truth. The live preview may be bandwidth-limited or channel-limited depending on configured mode and wireless conditions.

## Auxiliary and Closed-Loop Monitoring

Auxiliary previews help confirm IMU activity, digital input edges, stimulation markers, and onboard DSP or TinyML states. During closed-loop tests, use the live display to verify that thresholds and state variables change in the expected direction before relying on the configuration in an animal session.

For timing-sensitive triggers, combine live monitoring with exported files and the synchronization procedures documented in [Data Format](data-format.md). Live display timing is useful for operator feedback, but exported event timestamps and correction records are used for analysis.

## Camera Preview

Camera preview is useful for checking field of view, lighting, gross focus, and cable stability before a session. Camera characterization, working distance, and sensor hardware are documented in [Onboard Sensors](../hardware/onboard-sensors.md#camera).

## Power and Bandwidth

Live preview consumes device power and BLE bandwidth. Disable preview when the experiment prioritizes maximum runtime over live monitoring, and recover the full dataset from microSD after the session.

## Status Checks

- Confirm connection and TX/RX counters after BLE discovery.
- Confirm synchronization before starting a recording.
- Confirm recording state before handling the animal.
- Confirm export completion after removing the microSD card.
- Treat SD, BLE, low-battery, and release-image mismatch messages as session blockers until resolved.

---

# Source file: docs/software/data-format.md

# Data Format

Each WILD device recording session exports a folder with neural data, auxiliary signals, metadata, and optional camera or audio files.

## Files

| File | Verified public handling | Notes |
| --- | --- | --- |
| `amplifier.dat` | Read by `WILD_PreProcess.m` as channel-interleaved 16-bit neural data. | Sample count is computed as `file_bytes / 2 / Nch`, with `Nch` read from `CE_params.bin`. |
| `analogin.dat` | Used as the auxiliary stream for digital inputs, IMU, and timing-related channels. | Public MATLAB scripts read this stream as 16 channels at 1250 Hz for preprocessing and IMU extraction. |
| `digitalin.dat` | Optional digital-input export artifact. | Not every workflow generates a separate file; public trigger generation currently derives digital events from `analogin.dat`. |
| `adc.dat` | ADC or audio stream used by microphone workflows. | Public Python decode scripts read this file as `uint16` mono audio and generate `.wav` or `.mp3` outputs. |
| `misc.dat` | Raw camera payload stream. | Public Python decode scripts convert this file into reviewable video outputs. |
| `time.dat` | Generated sample-index timeline. | Public MATLAB preprocessing writes `0:(Nsamples-1)` as `int32`. |
| `info.rhd` | Intan-compatible metadata header. | Generated from `CE_params.bin` by `WILD_genIntanHeader.m`. |
| `CE_params.bin` | WILD parameter binary for system and DSP settings. | The public docs refer to this as the WILD parameter binary; MATLAB tools use the filename `CE_params.bin`. |

## Export Decoding

The WILD device records compact local streams on its microSD card. During the current SD-card download workflow, WILD Console decodes the on-device recording into analysis-facing files: neural samples are written to `amplifier.dat`, auxiliary and timing words to `analogin.dat`, ADC or audio streams to `adc.dat`, camera payloads to `misc.dat`, and session parameters to the WILD parameter binary.

The exported folder is therefore the decoded public data interface. Raw device storage blocks are not the expected analysis input; downstream MATLAB and Python tools operate on the downloaded files and generate derived outputs such as `info.rhd`, `time.dat`, event files, media files, IMU outputs, and spike-sorting inputs.

![WILD Console runtime screenshot of the offline export workflow](../images/WIrelessEphys_Github_5_offlineAPI.jpg){ .wild-readable-figure }

## Time Synchronization

The WILD device keeps high-bandwidth recordings local while WILD Console provides PC-device coordination over BLE. At connection and recording setup, the console synchronizes device state with the PC session and records timing context with the exported dataset.

Timing metadata should be interpreted as a layered system: device sample counts provide the primary sample timeline, PC-device timing coordination supports session-level alignment, and external I/O or digital events provide the most direct alignment path for external cameras, behavior systems, stimulation hardware, and multi-device sessions.

`time.dat` stores the sample-index timeline used by Intan-style workflows, while the WILD parameter binary preserves device-side recording time, hardware version, release image identity, sampling configuration, and DSP settings. External sync lines and digital inputs in `analogin.dat` or generated event files should be retained with the export whenever the experiment depends on cross-device or behavior alignment.

PC-device time synchronization is useful for session organization, export metadata, and cross-device coordination. It should not be treated as a substitute for hardware sync or digital event channels when the analysis requires high-precision alignment.

## Windows Timing Preparation

WILD logger time and PC system time are independent clocks. They can have a fixed offset and a small relative drift during a session. Windows automatic time management is designed to keep the PC close to network time; it does not calibrate drift against the WILD device clock. If Windows applies an automatic correction during an experiment, the PC clock can step abruptly relative to WILD logger time, which creates a discontinuity in PC-side trigger timestamps.

Treat this as a recording prerequisite for any session that uses PC-side timestamps or external timing alignment. For the best PC-side timestamp accuracy on Windows, prepare the acquisition computer before synchronization trials:

1. Disable Windows automatic time adjustment during the recording session, including automatic time setting and automatic time-zone changes.
2. Download and run `TimeCalibrator` on the acquisition computer before the experiment.
3. Use `TimeCalibrator` records together with WILD sync pulses to estimate PC-to-WILD time offset and drift.
4. Apply the offset and drift correction after export before merging USB-GPIO, behavior, camera, stimulation, or multi-device event timestamps with WILD logger time.

This step improves the stability of PC-side timestamps. It does not replace device-side sample timing or direct hardware TTL synchronization when the experiment requires the highest alignment accuracy.

## GPIO-through-USB Trigger Alignment

USB-GPIO and BLE can bridge external equipment and the WILD device, but the bridge has direction-dependent latency. Keep the two cases separate:

- **Incoming external trigger:** `TTL -> USB-GPIO -> PC -> BLE -> WILD`. The PC can timestamp the TTL edge, but the WILD device receives the related BLE command later. The logged WILD event reflects logger-side receive, handling, or sampled digital-input time, not the original PC timestamp unless the delay has been calibrated.
- **Outgoing WILD trigger:** `WILD -> BLE -> PC -> USB-GPIO -> TTL`. The WILD event begins on the logger time base, but the external TTL output appears later after BLE notification, PC scheduling, USB transfer, and GPIO output latency.

These two paths are not interchangeable. The incoming path maps PC-observed trigger time into WILD logger time; the outgoing path maps WILD events into external-equipment time, or is inverted to express external TTL outputs back on the WILD time base. Each direction should be calibrated with pulses that follow the same direction as the real experiment signal.

For high-precision alignment, prefer a hardware TTL split that is recorded directly by WILD and the external system. When a PC/BLE bridge is part of the experiment, treat the USB-GPIO-to-WILD or WILD-to-USB-GPIO delay as a measured timing relationship rather than a constant assumed to be zero. The delay can include GPIO edge detection, USB packet scheduling, operating-system timestamping, BLE connection interval, firmware handling, and the logger sampling edge.

![USB-GPIO trigger timestamp alignment](../images/WILD_usb_gpio_timestamp_alignment.svg){ .wild-readable-figure }

Recommended synchronization workflow:

1. Choose the bridge direction used by the experiment: incoming `TTL -> PC -> BLE -> WILD`, outgoing `WILD -> BLE -> PC -> TTL`, or a direct hardware TTL split.
2. On the Windows host, disable automatic time adjustment and run `TimeCalibrator` before the session.
3. Add periodic calibration pulses that traverse the same path as the experiment trigger.
4. Record PC/USB-GPIO timestamps, `t_usb`, and export the matching WILD event or digital-input timestamps as logger time, `t_wild = sample_index / fs`.
5. Pair the same pulses across both records and reject missed, duplicated, or ambiguous edges.
6. Fit the direction-specific map. For incoming triggers, use `t_wild = a * t_usb + b`. For outgoing triggers, fit the observed external TTL time against the originating WILD event and invert the map when external timestamps must be expressed in WILD time.
7. Store the direction, TimeCalibrator record, fit coefficients, sync-pulse residuals, release image, WILD Console version, USB-GPIO interface, BLE controller path, and trigger wiring with the exported dataset.

Periodic drift monitoring should continue across the recording, not only at the start. A short pre-session pulse train estimates initial delay, but repeated pulses during the session reveal PC clock drift, USB scheduling changes, logger clock drift, missed edges, or an interrupted connection. After fitting the USB-to-WILD time map, inspect residuals over time. A stable residual trace supports a single affine correction; jumps or curvature indicate that the session needs segmented correction, dropped-pulse review, or exclusion from high-precision timing analysis.

Post-export timestamp correction should be applied before merging external triggers with neural, IMU, audio, camera, or stimulation events. The corrected external trigger time is the timestamp expressed on the WILD logger time base, not the original USB host time. Keep both the raw USB-GPIO log and the corrected WILD-time trigger table so the alignment can be audited later.

## Verified Script Assumptions

The following public assumptions are visible in the repository scripts and should be treated as the current documented behavior unless a release note states otherwise:

- `WILD_PreProcess.m` reads `amplifier.dat` as 16-bit neural samples and uses `CE_params.bin` to determine channel count and sampling rate.
- `time.dat` is written as a monotonically increasing `int32` sample index.
- `WILD_processIMU.m` reads channels `2:10` from `analogin.dat`, assumes a 1250 Hz source rate, and writes `IMU.mat` after resampling to 100 Hz by default.
- `WILD_PreProcess.m` reads channel `1` of `analogin.dat` as a `uint16` digital bitfield and generates `device_event.*.evt` files from its bit transitions.
- `WILD_VideoDecodewAudio.py` reads `adc.dat` as `uint16` audio, converts it to signed audio outputs, and assumes a 160 kHz audio path.
- `WILD_VideoDecodewAudio.py` reads `misc.dat` in `320 * 500` byte frame blocks and writes 320 x 320, 16 Hz decoded video outputs.

Where a file layout depends on a specific release image or modality, report the exact release tag and mode together with the dataset. Do not assume that every export contains every optional modality file.

## Multi-Device and Behavior Alignment

For multi-logger sessions, keep the raw export folder for each device and store merge or sync-estimation outputs alongside the derived files. Useful validation checks include matched `amplifier.dat` duration and file size, stable estimated sample offsets, continuous external TTL or digital events, and no unexpected gaps in camera frames.

Behavior datasets should state whether video, UWB, IMU, and ephys streams are expressed on corrected timestamps. Camera calibration, coordinate transforms, identity curation, and delay correction are part of the dataset metadata, especially for outdoor multi-animal work.

When a behavior pipeline produces files such as `behavior_all.mat`, document whether fields are aligned to corrected timestamps such as `timestamps_corrected` and which correction was used. Post-hoc delay estimation or time warping can help evaluate a session, but it should not replace acquisition-side sync validation.

## Intan-Compatible Layout

WILD exports are arranged to be familiar to users of Intan-style recording folders while preserving WILD-specific metadata for sensors, DSP, stimulation, and camera workflows.

## Best Practice

Keep three folders per experiment:

1. Raw SD export.
2. Converted analysis copy.
3. Downstream results from spike sorting, behavior alignment, or machine-learning analysis.

---

# Source file: docs/software/artificial-intelligence.md

# Embedded Inference

The WILD device supports onboard biomarker detection and behavioral classification for responsive stimulation.

Current WILD releases are intentionally conservative: embedded models are curated into validated release images and reviewed for RAM use, compute timing, sampling schedules, storage interactions, and closed-loop behavior.

## Current Model Integration

Current support uses validated, experiment-specific models rather than arbitrary runtime uploads.

- Models are distributed as part of validated release images.
- RAM use is curated before deployment.
- Inference timing is checked against acquisition, storage, BLE, and DSP tasks.
- Model output is integrated with device state reporting and closed-loop logic.
- Release image and model identity are recorded with each experiment.

This approach keeps timing behavior predictable on resource-constrained embedded hardware.

## Generic Model Support

Generic model loading is not part of the stable public workflow yet. Current releases prioritize predictable timing, memory use, and closed-loop behavior.

Closed-loop model releases define:

- Accepted model format and quantization requirements.
- Maximum model size and tensor arena limits.
- Input and output tensor conventions.
- Inference scheduling relative to acquisition and DSP.
- Validation procedure before animal experiments.
- How model version, release image, and experiment metadata are recorded.

!!! warning "Deployment discipline"
    Embedded models are part of the validated device release. A model that passes desktop inference tests may still be unsafe for closed-loop use if RAM pressure or inference timing affects acquisition, storage, or stimulation behavior.

---

# Source file: docs/software/api-cli.md

# Programmatic Workflows

WILD Console and the listed scripts form the supported public interface for acquisition, export, post-processing, and validation.

WILD Console handles BLE discovery, synchronization support, configuration, selected preview, and SD-card export. Full-resolution recordings are recovered from local storage rather than streamed continuously over BLE.

!!! warning "No stable SDK yet"
    The supported public programmable surface is WILD Console plus the documented repository scripts. The project does not currently document a stable general-purpose SDK with a versioned command contract.

## Current Support

| Area | Current operation style | Purpose |
| --- | --- | --- |
| Acquisition | WILD Console | BLE discovery, connection, synchronization support, recording setup, selected live preview, closed-loop configuration, and SD-card export. |
| Post-processing | MATLAB scripts | Header generation, timing correction, event export, IMU processing, and Intan-compatible output preparation. |
| Camera/audio decoding | Python scripts | Decode `misc.dat` and related audio data into reviewable media outputs. |
| Batch processing | Python and MATLAB scripts | Process folders of recordings after SD export. |
| GPIO logging | Python utility | Log serial/GPIO event data during validation or synchronization tests. |
| Release and model tracking | Release metadata workflow | Record release image, model identity, and tool version for reproducibility. |

## Supported Scripted Operations

The current public API surface is organized around these operations:

- Export recording folders from WILD Console.
- Generate corrected `info.rhd` and `time.dat` outputs.
- Convert raw WILD camera/audio data into media files.
- Process IMU channels into calibrated signals.
- Batch-process recording folders.
- Log GPIO or serial synchronization events.
- Preserve software, release-image, and model versions in experiment notes.

## Device Operation Categories

WILD Console and controller workflows expose device operations at a task level. Public documentation should describe these operations by purpose rather than by low-level command ID.

| Category | Typical operations |
| --- | --- |
| Recording control | Set recording parameters, start local recording, stop local recording, and confirm recording state. |
| Preview control | Start or stop selected-channel preview and choose preview channels without treating BLE as the full-resolution data path. |
| DSP and TinyML setup | Load curated DSP or model parameters from a validated release image and record the selected configuration in metadata. |
| Stimulation control | Configure stimulation intensity, thresholds, timing, and manual test triggers when the hardware revision supports stimulation. |
| Time and synchronization | Check the connection, coordinate PC-device time, set device time, and retain external I/O events for experiment-level alignment. |
| Device status | Read parameters, battery state, available storage, device identity, and release-image metadata. |
| Power state | Request sleep or reset only as part of the documented maintenance or field-session workflow. |
| External I/O | Route GPIO and synchronization signals to cameras, behavioral systems, stimulation modules, or additional WILD devices. |

## Tool Metadata

For reproducible use, record the following metadata with tool outputs and example datasets:

- Tool name and version.
- Required operating system and dependencies.
- Input folder or file layout.
- Command or launch procedure used.
- Output files and naming convention.
- Expected runtime and memory requirements.
- Failure modes and troubleshooting notes.
- Validation file or representative dataset, when available.

## SDK Status

WILD Console operations and documented processing scripts are the supported public surface. Script internals remain implementation details unless a workflow is explicitly documented.

---

# Source file: docs/analysis/index.md

# Analysis

Analysis documentation covers conversion from WILD exports to analysis-ready files, with timing metadata, MATLAB and Python workflows, and spike-sorting preparation kept explicit.

## Entry Points

<div class="wild-grid wild-nav-grid">
  <a class="wild-card wild-card-link wild-card-compact" href="python/">
    <div>
    <h3>Python</h3>
      <p>Video and GPIO</p>
    </div>
  </a>
  <a class="wild-card wild-card-link wild-card-compact" href="matlab/">
    <div>
    <h3>MATLAB</h3>
      <p>Preprocessing</p>
    </div>
  </a>
  <a class="wild-card wild-card-link wild-card-compact" href="spike-sorting/">
    <div>
      <h3>Spike sorting</h3>
      <p>Ephys pipeline</p>
    </div>
  </a>
</div>

## Reproducibility Checklist

- Record release image filename.
- Record WILD Console version.
- Preserve the raw export folder.
- Preserve the WILD parameter binary.
- Record probe, channel map, sampling rate, and stimulation configuration.
- Track post-processing script version or commit hash.

---

# Source file: docs/analysis/python.md

# Python

Python utilities cover camera decoding, audio handling, GPIO logging, and validation after SD-card export.

## Scripts

| Script | Purpose |
| --- | --- |
| [`WILD_VideoDecodewAudio.py`](https://github.com/ayalab1/Neurologger/blob/main/Code/WILD_VideoDecodewAudio.py) | Decode camera data with audio handling. |
| [`WILD_VideoDecodewAudio_v2.py`](https://github.com/ayalab1/Neurologger/blob/main/Code/WILD_VideoDecodewAudio_v2.py) | Alternative video decode pipeline. |
| [`WILD_VideoDecodewAudio_folder.py`](https://github.com/ayalab1/Neurologger/blob/main/Code/WILD_VideoDecodewAudio_folder.py) | Batch folder processing. |
| [`WILD_GPIO_Logger.py`](https://github.com/ayalab1/Neurologger/blob/main/Code/WILD_GPIO_Logger.py) | GPIO logging utility. |

## Typical Use

- Decode camera or audio files after exporting a recording folder.
- Batch-process multiple recording folders with the folder variants.
- Log GPIO or serial events during synchronization validation.
- Keep generated media and logs separate from the raw SD export.

## Interactive Commands

Run the Python tools from the repository `Code` folder:

```powershell
cd C:\code\github\Neurologger\Code
python .\WILD_VideoDecodewAudio.py
python .\WILD_VideoDecodewAudio_folder.py
python .\WILD_GPIO_Logger.py
```

- `WILD_VideoDecodewAudio.py` opens a file picker for one `misc.dat` recording and expects the matching `adc.dat` file when audio is present.
- `WILD_VideoDecodewAudio_folder.py` opens a folder picker and recursively processes `misc.dat` files below that root.
- `WILD_GPIO_Logger.py` prompts for a COM port and output folder, then writes a timestamped text log.

## Expected Outputs

```text
example_recording/
  misc.dat
  adc.dat
  adc.wav
  adc.mp3
  misc.mp4
  misc_wAudio.mp4
```

Output names depend on the script path and whether audio is present, but the public decoder workflow writes reviewable media next to the source export by default.

---

# Source file: docs/analysis/matlab.md

# MATLAB

MATLAB scripts currently provide the main preprocessing path for WILD recordings.

## Scripts

| Script | Purpose |
| --- | --- |
| [`WILD_PreProcess.m`](https://github.com/ayalab1/Neurologger/blob/main/Code/WILD_PreProcess.m) | Generate corrected headers, timing files, and event outputs. |
| [`WILD_PreProcessFolder.m`](https://github.com/ayalab1/Neurologger/blob/main/Code/WILD_PreProcessFolder.m) | Process multiple recordings. |
| [`WILD_ReadHeader.m`](https://github.com/ayalab1/Neurologger/blob/main/Code/WILD_ReadHeader.m) | Read WILD parameter headers. |
| [`WILD_processIMU.m`](https://github.com/ayalab1/Neurologger/blob/main/Code/WILD_processIMU.m) | Process IMU data. |
| [`WILD_scaleIMU.m`](https://github.com/ayalab1/Neurologger/blob/main/Code/WILD_scaleIMU.m) | Scale IMU signals. |
| [`WILD_genIntanHeader.m`](https://github.com/ayalab1/Neurologger/blob/main/Code/WILD_genIntanHeader.m) | Generate Intan-compatible header output. |

## Minimal Workflow

```matlab
cd("C:\code\github\Neurologger\Code");
recording_path = "C:\WILD\example_recording";
WILD_PreProcess(fullfile(recording_path, "amplifier.dat"));
WILD_processIMU(fullfile(recording_path, "analogin.dat"), 100);
```

Adjust paths and sampling assumptions for your recording mode.

## Expected Folder Layout

**Input**

```text
example_recording/
  amplifier.dat
  analogin.dat
  CE_params.bin
  adc.dat           # optional
  misc.dat          # optional
```

**Outputs generated by the public MATLAB path**

```text
example_recording/
  time.dat
  info.rhd
  IMU.mat
  device_event.d01.evt   # if digital events are present
  device_event.s01.evt   # if stimulation-related event files are generated
```

## Known-Good First Pass

1. Run `WILD_PreProcess` on `amplifier.dat`.
2. Confirm that `time.dat` and `info.rhd` appear.
3. Run `WILD_processIMU` on `analogin.dat`.
4. Confirm that `IMU.mat` is generated.
5. Keep the raw export unchanged and write downstream results into a separate analysis copy when reprocessing.

---

# Source file: docs/analysis/spike-sorting.md

# Spike Sorting

WILD neural data can be organized for downstream spike sorting after export and preprocessing.

## Recommended Workflow

1. Preserve the raw WILD export folder.
2. Run WILD preprocessing to generate `info.rhd`, `time.dat`, and corrected metadata.
3. Confirm sampling rate, channel count, channel ordering, and probe geometry.
4. Prepare a sorter-specific working directory.
5. Run quality-control checks before interpreting units.

## Recommended Sorter

Kilosort is the recommended starting point for spike sorting exported WILD electrophysiology recordings. Keep the sorter working directory separate from the raw SD export and keep channel-map files with the processed dataset.

!!! note
    Do not assume a generic channel map. Document the exact probe and channel ordering used for each dataset.

---

# Source file: docs/examples/index.md

# Examples

These example workflows summarize common WILD device use cases and the metadata needed to make each experiment reproducible.

## Workflow Index

| Example | Scope | Expected validation |
| --- | --- | --- |
| Basic neural recording | Battery, SD format, BLE connect, recording, export, preprocessing. | `amplifier.dat`, `analogin.dat`, `time.dat`, `info.rhd`, `CE_params.bin`. |
| Closed-loop ripple detection | Filter setup, thresholding, stimulation configuration, validation waveform. | Event files, threshold traces, and stimulation timing review. |
| Theta phase stimulation | Hilbert mode setup, phase logic, stimulation timing, event export. | Phase-locked trigger timing and stimulation event checks. |
| Multi-animal synchronization | Multiple WILD Console instances, start timing, sync line, analysis alignment. | Matched durations, stable offsets, and retained sync events. |
| Outdoor recording | Power planning, SD choice, wireless checks, data integrity checks. | Complete export, no unexpected truncation, and logged power configuration. |
| Camera plus IMU | Camera decode, IMU processing, and behavioral alignment. | Reviewable video output and generated `IMU.mat`. |

## Required Metadata

For each public dataset or protocol, record:

- Goal.
- Hardware revision.
- Release image.
- Software version.
- Required accessories.
- Step-by-step protocol.
- Expected outputs.
- Troubleshooting notes.
- Example data or a small validation file.

## Priority for Expanded Protocol Pages

The highest-value example pages to expand next are:

1. Basic neural recording.
2. Multi-animal synchronization.
3. Closed-loop ripple detection.

These three paths cover the first successful dry run, cross-device timing, and one validated responsive-control workflow.

---

# Source file: docs/community/index.md

# Community

Community pages collect citation guidance, contribution rules, and support expectations for the public WILD workflow.

## Start Here

<div class="wild-grid two">
  <a class="wild-card wild-card-link" href="../publications/">
    <h3>Publication</h3>
    <p>Use the current publication-status guidance and include release metadata for reproducible methods reporting.</p>
  </a>
  <a class="wild-card wild-card-link" href="../contributing/">
    <h3>Contributing</h3>
    <p>Report tested hardware revision, release image, software version, and validation notes with every documentation or compatibility update.</p>
  </a>
  <a class="wild-card wild-card-link" href="https://github.com/ayalab1/Neurologger/issues">
    <h3>Support Path</h3>
    <p>Use GitHub issues and pull requests for reproducible bugs, documentation problems, compatibility reports, and analysis examples.</p>
  </a>
  <a class="wild-card wild-card-link" href="../contributing/#documentation-style">
    <h3>Visual Policy</h3>
    <p>Public device figures must come from real hardware, CAD, PCB, microscope, or hand-authored source-derived diagrams. AI-generated device imagery is excluded.</p>
  </a>
</div>

## Recommended Metadata for Public Reports

When sharing a method, bug report, example dataset, or compatibility note, include:

- Hardware revision.
- Release image or release tag.
- WILD Console version.
- SD-card model.
- Battery model.
- Recording mode and sampling configuration.
- Whether audio, video, IMU, stimulation, or external sync were enabled.

---

# Source file: docs/publications/index.md

# Publications

Publication notes collect the current WILD manuscript status and the release metadata needed for reproducible methods reporting.

## Publication Status

A manuscript describing the WILD platform is currently under review. This documentation provides public-facing information for the hardware files, release images, WILD Console usage, post-processing workflows, metadata requirements, and validation checks associated with the platform.

Final peer-reviewed citation information, including journal, DOI, volume, issue, and page details, will be added after publication.

## Citation Before Publication

Before publication, cite the repository or release record and report the exact release context used in the work. Do not include journal, DOI, volume, issue, or page information until those details are final.

```bibtex
@misc{wild_repository,
  title        = {WILD: Wireless, Interactive, Lightweight Datalogger},
  author       = {Zhao, Zifang and Chang, Hongyu and Paudel, Praveen and Park, Jaehyo and Liu, Can and Aurelio, Maria Q. and Oliva, Azahara and Fernandez-Ruiz, Antonio},
  howpublished = {\url{https://github.com/ayalab1/Neurologger}},
  note         = {Public documentation and release files. Report the release tag, device image, hardware revision, WILD Console version, and analysis-script commit used in the experiment.},
  year         = {2026}
}
```

## Repository Citation

When citing a specific software or hardware release, include the GitHub repository URL and the exact release image, software version, and hardware revision used.

Repository: [https://github.com/ayalab1/Neurologger](https://github.com/ayalab1/Neurologger)

## Reproducible Methods Reporting

For reproducible methods reporting, include:

- Repository URL.
- Release tag.
- Release image filename.
- Hardware revision.
- WILD Console version.
- Analysis script version or commit hash, when applicable.

The repository includes a root `CITATION.cff` file for software-style citation metadata.

---

# Source file: docs/contributing.md

# Contributing

WILD is an open-source hardware and software project. Contributions are welcome when they make the system easier to reproduce, validate, or extend.

## Ways to Contribute

- Improve setup and troubleshooting documentation.
- Share hardware assembly notes or photos.
- Report WILD Console, release-image, or hardware compatibility issues with version details.
- Share analysis examples and validation datasets.
- Improve closed-loop or TinyML examples.
- Provide compatibility notes for probes, SD cards, batteries, and BLE adapters.

## Pull Request Checklist

1. Describe the hardware revision, release image, and software version used for testing.
2. Keep documentation in Markdown.
3. Place images under `docs/images/` only when they are useful for assembly, operation, or validation.
4. Do not use AI-generated imagery for any WILD device depiction. Device photos, device illustrations, PCB renders, CAD renders, connector diagrams, pinouts, and schematics must come from real photographs, project CAD/EDA exports, measurements, or hand-authored diagrams derived from source files.
5. Use clear filenames and alt text for figures.
6. Avoid claims that are not tied to a tested configuration.
7. Run `mkdocs build --strict` before submitting documentation changes.

## Documentation Style

- Prefer short procedural steps for setup pages.
- Prefer diagrams and tables for architecture pages.
- Keep WILD hardware visuals traceable to real hardware, CAD, PCB, or measured source material.
- Separate validated behavior from unvalidated claims.
- Keep safety, protocol, and animal-use details local to approved lab procedures unless they are public and reviewed.

---

# Source file: docs/faq.md

# FAQ

## Is WILD a replacement for wired acquisition?

WILD is designed for wireless and naturalistic experiments where tethered acquisition is limiting. The correct choice depends on channel count, runtime, signal quality, synchronization, and behavioral constraints.

## Is WILD a BLE telemetry system?

No. WILD is primarily a local-storage neurologger. High-bandwidth neural and multimodal data are recorded to the device microSD card. BLE is used for discovery, configuration, synchronization support, status, selected low-bandwidth preview, and control commands.

## What is the current recording scale?

The current open-source workflow is the 64-channel WILD system. Neuropixels-compatible and active-SPI-probe variants are higher-performance research targets and are separate from the current public release unless a release note explicitly states otherwise.

## How do I install the PC software?

The latest installer is available from the [latest GitHub release](https://github.com/ayalab1/Neurologger/releases/latest), with setup details in the acquisition guide.

## BLE connects, but recording does not start

The microSD card is the first item to verify. SD format and card compatibility are common causes when BLE connects but synchronization does not begin.

## Does WILD support TinyML?

WILD supports online inference of curated, experiment-specific models through validated release images. Runtime generic model loading is not part of the stable public workflow yet.

## How do I cite WILD?

Use the current citation guidance on the [Publications](publications/index.md) page. Until the manuscript is published, cite the repository or release record and include the WILD hardware revision, release image, WILD Console version, and analysis-script version.

## Where do new diagrams go?

Source diagrams are best kept editable where possible, with web-ready images exported into `docs/images/`. For readable architecture diagrams, Mermaid in Markdown is preferred.
