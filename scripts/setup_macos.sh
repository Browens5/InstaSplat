#!/usr/bin/env bash
# One-shot macOS bootstrap for InstaSplat (tools + Python env).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WITH_GUI="${WITH_GUI:-1}"
SKIP_BREW="${SKIP_BREW:-0}"
SKIP_PIP="${SKIP_PIP:-0}"

echo "==> InstaSplat macOS setup"
echo "    repo: $ROOT"

# —— System tools ——
if [[ "$SKIP_BREW" != "1" ]]; then
  if ! command -v brew >/dev/null 2>&1; then
    echo "ERROR: Homebrew not found. Install from https://brew.sh then re-run."
    exit 1
  fi
  echo "==> Homebrew packages (ffmpeg, exiftool, colmap, git)"
  brew install ffmpeg exiftool git || true
  brew install colmap || echo "WARN: colmap brew install failed — build from source if needed"
else
  echo "==> Skipping brew (SKIP_BREW=1)"
fi

if command -v npm >/dev/null 2>&1; then
  echo "==> splat-transform (npm global)"
  npm install -g @playcanvas/splat-transform || true
else
  echo "WARN: npm not found — install Node.js 18+, then:"
  echo "      npm i -g @playcanvas/splat-transform"
fi

# —— Python env ——
if [[ "$SKIP_PIP" != "1" ]]; then
  if [[ ! -d "$ROOT/.venv" ]]; then
    echo "==> Creating .venv"
    python3 -m venv "$ROOT/.venv"
  fi
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  echo "==> pip install -e . (metal_equirect trainer + deps)"
  python -m pip install -U pip wheel
  EXTRAS="dev"
  if [[ "$WITH_GUI" == "1" ]]; then
    EXTRAS="gui,dev"
  fi
  python -m pip install -e ".[${EXTRAS}]"
else
  echo "==> Skipping pip (SKIP_PIP=1) — activate your venv and pip install -e '.[gui,dev]'"
fi

# —— Verify ——
echo "==> Verifying"
if command -v instasplat >/dev/null 2>&1; then
  instasplat setup --no-system || true
  instasplat doctor || true
else
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python -m instasplat.cli setup --no-system || true
fi

cat <<EOF

============================================================
Setup finished.

Activate the venv (new shells):
  source $ROOT/.venv/bin/activate

Read the metal splat workflow:
  docs/METAL_SPLAT_WORKFLOW.md

Run a long 360 job:
  instasplat mac-360 -i ./capture_equirect.mp4 -o ./runs -n walk

Or open the GUI:
  instasplat gui

Stitch tip: export equirectangular MP4 from Insta360 Studio
(MediaSDK is Windows/Linux only).
============================================================
EOF
