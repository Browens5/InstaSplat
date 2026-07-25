#!/usr/bin/env bash
# Best-effort macOS dependency bootstrap for InstaSplat.
set -euo pipefail

echo "==> InstaSplat macOS setup"

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Install from https://brew.sh and re-run."
  exit 1
fi

brew install ffmpeg exiftool || true
brew install colmap || echo "WARN: colmap brew install failed — build from source if needed"

if command -v npm >/dev/null 2>&1; then
  npm install -g @playcanvas/splat-transform || true
else
  echo "WARN: npm not found — install Node.js then: npm i -g @playcanvas/splat-transform"
fi

if command -v rustc >/dev/null 2>&1; then
  echo "Rust present: $(rustc --version)"
  echo "Build Brush from https://github.com/ArthurBrussee/brush (cargo build --release)"
else
  echo "WARN: Rust not found — install via https://rustup.rs to build Brush"
fi

echo
echo "Python deps: python3 -m venv .venv && source .venv/bin/activate && pip install -e '.[gui,dev]'"
echo "Then run: instasplat doctor"
echo "Stitch tip: export equirectangular MP4 from Insta360 Studio on macOS (MediaSDK is Win/Linux)."
