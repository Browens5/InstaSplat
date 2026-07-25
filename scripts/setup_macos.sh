#!/usr/bin/env bash
# Best-effort macOS dependency bootstrap for InstaSplat.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "==> InstaSplat macOS setup"

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Install from https://brew.sh and re-run."
  exit 1
fi

brew install ffmpeg exiftool git || true
brew install colmap || echo "WARN: colmap brew install failed — build from source if needed"

if command -v npm >/dev/null 2>&1; then
  npm install -g @playcanvas/splat-transform || true
else
  echo "WARN: npm not found — install Node.js then: npm i -g @playcanvas/splat-transform"
fi

echo
echo "Python deps (includes PyTorch for metal_equirect training):"
echo "  python3 -m venv .venv && source .venv/bin/activate"
echo "  pip install -e '.[gui,dev]'"
echo "  # Apple Silicon: if pip torch lacks MPS, install from https://pytorch.org"
echo
echo "Then run: instasplat doctor"
echo "Long 360: instasplat mac-360 -i ./capture_equirect_8k.mp4 -o ./runs -n walk"
echo "Stitch tip: export equirectangular MP4 from Insta360 Studio on macOS (MediaSDK is Win/Linux)."
