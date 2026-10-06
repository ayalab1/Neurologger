# WILD

**WILD** (Wireless, Interactive, Lightweight Datalogger) is an ultra-lightweight multimodal neurologger for long-term neural and behavioral recording in freely behaving animals. It combines local neural data logging, behavioral sensing, onboard processing, and responsive stimulation in a head-mounted platform designed for naturalistic and multi-animal experiments.

## Documentation

The documentation portal is built with MkDocs Material and is maintained through Markdown files in [`docs/`](docs/).

- [Getting Started](docs/getting-started/index.md)
- [Hardware](docs/hardware/index.md)
- [Software](docs/software/index.md)
- [Analysis](docs/analysis/index.md)
- [Community](docs/contributing.md)

## Platform Scope

- Local microSD electrophysiology recording with BLE-based control, synchronization, status, and low-bandwidth preview
- Closed-loop stimulation through embedded DSP and curated TinyML pipelines
- Current open-source 64-channel workflows, with Neuropixels-compatible and active-probe variants documented separately as research targets
- IMU, ultrasonic audio, camera, and digital event recording
- TinyML inference support through validated release-integrated models
- Long-term recording and multi-animal experiment workflows

## Repository Layout

| Path | Purpose |
| --- | --- |
| [`docs/`](docs/) | MkDocs documentation source |
| [`Firmware/`](Firmware/) | Current fused CE64 images and validation limits |
| [Tagged GitHub releases](https://github.com/ayalab1/Neurologger/releases) | Archived release assets |
| [`Software/`](Software/) | WILD Console installers |
| [`Code/`](Code/) | MATLAB and Python analysis scripts |
| [`PCB/`](PCB/) | PCB fabrication and assembly files |
| [`3Dprint/`](3Dprint/) | Mechanical parts and CAD files |

## Local Documentation Build

```bash
pip install -r requirements.txt
mkdocs serve
```

The GitHub Pages workflow builds the site with:

```bash
mkdocs build --strict
```

## License

See [LICENSE](LICENSE).
