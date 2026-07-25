#!/usr/bin/env bash
# Best local Mac entrypoint: long 360 equirect → tiled Metal Gaussian splat.
set -euo pipefail

INPUT="${1:-}"
OUT="${2:-./runs}"
NAME="${3:-walk_360}"
TRAINER="${TRAINER:-brush}"

if [[ -z "${INPUT}" ]]; then
  cat <<EOF
Usage: $0 <equirect.mp4|capture.insv> [output_dir] [project_name]

Examples:
  $0 ./capture_equirect_8k.mp4 ./runs beach_walk
  TRAINER=opensplat $0 ./capture_equirect_8k.mp4

Prefer a Studio-stitched equirect MP4. Keep the sibling .insv (or gyro.csv/gps.csv)
beside it for telemetry. See docs/MAC_LONG_360.md.
EOF
  exit 1
fi

if ! command -v instasplat >/dev/null 2>&1; then
  echo "instasplat not on PATH — run: pip install -e '.[gui]'" >&2
  exit 1
fi

instasplat doctor || true
exec instasplat mac-360 \
  --input "${INPUT}" \
  --output "${OUT}" \
  --name "${NAME}" \
  --trainer "${TRAINER}" \
  --formats ply,sog,spz
