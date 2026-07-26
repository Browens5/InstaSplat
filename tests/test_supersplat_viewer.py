"""Tests for the embedded SuperSplat viewer server + asset packaging."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from instasplat.gui.supersplat import ASSETS_DIR, assets_ready, viewer_version
from instasplat.gui.supersplat.server import SuperSplatLocalServer
from instasplat.metal_equirect.gaussians import export_ply, gaussians_from_points


def test_supersplat_assets_packaged() -> None:
    assert assets_ready()
    assert (ASSETS_DIR / "index.html").stat().st_size > 1000
    assert (ASSETS_DIR / "index.js").stat().st_size > 100_000
    assert viewer_version()
    assert viewer_version()[0].isdigit()


def test_supersplat_local_server_serves_ply(tmp_path: Path) -> None:
    xyz = np.random.randn(32, 3).astype(np.float32) * 0.1
    rgb = np.random.rand(32, 3).astype(np.float32)
    model = gaussians_from_points(xyz, rgb, sh_degree=0, max_points=32)
    ply = tmp_path / "live.ply"
    export_ply(model, ply)

    srv = SuperSplatLocalServer()
    try:
        base = srv.start()
        assert srv.port > 0
        mtime = srv.set_ply(ply)
        assert mtime > 0
        url = srv.viewer_url(cache_bust=mtime)
        assert "content=/content.ply" in url
        assert "webgl" in url
        assert "noui" in url

        import urllib.request

        with urllib.request.urlopen(f"{base}/index.html", timeout=5) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        assert "SuperSplat" in html or "application-canvas" in html

        with urllib.request.urlopen(f"{base}/settings.json", timeout=5) as resp:
            settings = resp.read().decode("utf-8")
        assert "version" in settings

        with urllib.request.urlopen(f"{base}/content.ply", timeout=5) as resp:
            data = resp.read()
        assert data[:3] == b"ply" or data[:4] == b"ply\n" or b"format" in data[:200]
        assert len(data) == ply.stat().st_size
    finally:
        srv.stop()


def test_supersplat_widget_importable() -> None:
    # Import path must work even without a display / WebEngine session
    from instasplat.gui.supersplat.widget import SuperSplatViewerWidget, webengine_available

    assert callable(SuperSplatViewerWidget)
    assert isinstance(webengine_available(), bool)


@pytest.mark.skipif(
    __import__("os").environ.get("QT_QPA_PLATFORM") == "offscreen"
    and not __import__("os").environ.get("INSTASPLAT_TEST_WEBENGINE"),
    reason="WebEngine widget needs a real display unless explicitly enabled",
)
def test_viewer_panel_has_supersplat_stack() -> None:
    from PySide6.QtWidgets import QApplication

    from instasplat.gui.gl_view import configure_default_surface_format
    from instasplat.gui.viewer_panel import ViewerPanel

    configure_default_surface_format()
    app = QApplication.instance() or QApplication([])
    panel = ViewerPanel()
    assert hasattr(panel, "splat_view")
    assert panel._view_stack.count() == 2
    panel.shutdown()
    del panel
    del app
