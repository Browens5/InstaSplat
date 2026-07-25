# macOS setup

## Requirements

- macOS 13+ recommended (Apple Silicon preferred)
- Homebrew
- Python 3.11+
- Node.js 18+ (for splat-transform)
- PyTorch with MPS (included via `pip install -e .`)
- Insta360 Studio (for equirect exports)

## Quick setup

```bash
git clone <this-repo> && cd InstaSplat
./scripts/setup_macos.sh          # brew tools + splat-transform
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[gui,dev]"       # includes torch for metal_equirect
instasplat doctor
```

If `doctor` reports PyTorch without MPS on Apple Silicon, install the official
MPS wheel from https://pytorch.org .

## Manual installs

```bash
brew install ffmpeg exiftool colmap git
npm install -g @playcanvas/splat-transform
```

## Insta360 Studio export checklist

1. Open the capture in Insta360 Studio
2. Enable optical-flow stitch / FlowState as desired
3. Export **equirectangular** MP4 (not reframed flat)
4. Save next to the `.insv` or pass the MP4 directly to InstaSplat

## YOLO models

First mask run downloads weights (e.g. `yolov8m-seg.pt`). For air-gapped machines, place weights in the working directory or Ultralytics cache.

## COLMAP notes

- Homebrew COLMAP on Apple Silicon is typically **CPU** for mapping — fine for small jobs.
- Cubemap SfM (`perspective_cubemap`) is the default. Poses are lifted to equirect for
  **metal_equirect** training (see [METAL_EQUIRECT_TRAINER.md](METAL_EQUIRECT_TRAINER.md)).

## GUI

```bash
pip install 'instasplat[gui]'
instasplat gui
```
