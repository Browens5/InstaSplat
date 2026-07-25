#!/usr/bin/env bash
# Auto-clone and release-build ArthurBrussee/brush for InstaSplat (Mac Metal/WebGPU).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if command -v instasplat >/dev/null 2>&1; then
  exec instasplat install-brush "$@"
fi

# Fallback when package not installed yet
python3 - <<'PY'
from instasplat.utils.brush_install import install_brush
import sys
force = "--force" in sys.argv
r = install_brush(force_rebuild=force)
print(r.message)
sys.exit(0 if r.ok else 1)
PY
