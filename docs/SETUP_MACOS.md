# macOS setup

## Requirements

- macOS 13+ recommended (Apple Silicon preferred)
- Homebrew
- Python 3.11+
- Node.js 18+ (for splat-transform)
- Rust 1.88+ (to build Brush)
- Insta360 Studio (for equirect exports)

## Quick setup

```bash
git clone <this-repo> && cd InstaSplat
./scripts/setup_macos.sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[gui,dev]"
instasplat doctor
```

## Manual installs

```bash
brew install ffmpeg exiftool colmap
npm install -g @playcanvas/splat-transform

# Brush (from source)
git clone https://github.com/ArthurBrussee/brush.git
cd brush && cargo build --release
# put target/release/brush on your PATH
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
- For native equirectangular cameras, use a recent COLMAP build with `EQUIRECTANGULAR`, or stick to `perspective_cubemap` (default).

## GUI

```bash
pip install 'instasplat[gui]'
instasplat gui
```
