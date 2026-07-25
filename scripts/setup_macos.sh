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

# Rust + Brush (Metal/WebGPU trainer)
if ! command -v rustc >/dev/null 2>&1; then
  echo "==> Installing Rust via rustup"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env" || true
fi

echo "==> Building / installing Brush"
if [[ -x "$ROOT/scripts/install_brush.sh" ]]; then
  "$ROOT/scripts/install_brush.sh" || echo "WARN: Brush install failed — run: instasplat install-brush (Apple Silicon uses GitHub release; needs Rust 1.88+ for --from-source)"
elif command -v instasplat >/dev/null 2>&1; then
  instasplat install-brush || true
else
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 -c \
    "from instasplat.utils.brush_install import install_brush; r=install_brush(); print(r.message); raise SystemExit(0 if r.ok else 1)" \
    || echo "WARN: Brush install failed — after pip install -e ., run: instasplat install-brush"
fi

echo
echo "Python deps: python3 -m venv .venv && source .venv/bin/activate && pip install -e '.[gui,dev]'"
echo "Then run: instasplat doctor"
echo "Long 360: instasplat mac-360 -i ./capture_equirect_8k.mp4 -o ./runs -n walk"
echo "Stitch tip: export equirectangular MP4 from Insta360 Studio on macOS (MediaSDK is Win/Linux)."
