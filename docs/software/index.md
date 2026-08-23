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
  <a class="wild-card wild-card-link wild-card-compact" href="usb-mode/">
    <div>
      <h3>USB Mode</h3>
      <p>Stream, test, and update</p>
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

The current public software paths are:

1. Connect and record with WILD Console.
2. Export from the device microSD card.
3. Run documented MATLAB or Python post-processing.

For supported CE64 V4 images, the Console also provides a separate wired [USB mode](usb-mode.md) for 64-channel/5 kHz ephys and IMU recording, impedance testing, safe logger handoff, and fused-HEX firmware installation.

The public documentation does not treat BLE as a continuous high-bandwidth acquisition path, and it does not promise a stable general-purpose SDK yet.

## Wireless Control

WILD Console remains the stable public control and export workflow for BLE discovery, connection, synchronization support, status checks, selected preview, low-bandwidth commands, and SD-card export.

Normal untethered WILD recordings remain local to the device microSD card and are exported after the session. The explicit CE64 V4 USB mode can instead record supported wired streams directly to the PC.

## Requirements

- Windows 10 or later.
- .NET Framework 4.8 for the current GUI.
- Bluetooth 4.0 or later.
- Administrator privileges for some SD-formatting workflows.
- A data-capable USB cable for CE64 V4 USB mode.
- STM32CubeProgrammer only for full-flash USB upgrade or recovery.

## Optional Runtime Files

- `dll_upfirdn.dll` for resampling.
- WILD BLE backend DLL for BLE support in older installer layouts.
- `ffmpeg.exe` for some camera processing workflows.
