"""Vendored PlayCanvas SuperSplat Viewer (MIT) for the InstaSplat live GUI.

Assets from ``@playcanvas/supersplat-viewer`` — see VERSION and LICENSE.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PACKAGE_DIR / "assets"
SETTINGS_PATH = PACKAGE_DIR / "settings.json"
VERSION_PATH = PACKAGE_DIR / "VERSION"


def viewer_version() -> str:
    try:
        return VERSION_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"


def assets_ready() -> bool:
    return (
        (ASSETS_DIR / "index.html").is_file()
        and (ASSETS_DIR / "index.js").is_file()
        and (ASSETS_DIR / "index.css").is_file()
        and SETTINGS_PATH.is_file()
    )
