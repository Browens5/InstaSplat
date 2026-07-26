"""Qt WebEngine host for the vendored SuperSplat viewer."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtWidgets import QLabel, QStackedWidget, QVBoxLayout, QWidget

from instasplat.gui.supersplat import assets_ready, viewer_version
from instasplat.gui.supersplat.server import SuperSplatLocalServer

log = logging.getLogger("instasplat.gui.supersplat")


def webengine_available() -> bool:
    try:
        from PySide6.QtWebEngineWidgets import QWebEngineView  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


class SuperSplatViewerWidget(QWidget):
    """
    Embedded SuperSplat (PlayCanvas) Gaussian splat viewer.

    Serves vendored viewer assets + the current ``live.ply`` over localhost and
    displays them in ``QWebEngineView``. Falls back to a status label when
    WebEngine or assets are unavailable.
    """

    status = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._server = SuperSplatLocalServer()
        self._ply_path: Path | None = None
        self._last_url: str = ""
        self._view = None

        self._stack = QStackedWidget(self)
        self._fallback = QLabel("SuperSplat viewer unavailable")
        self._fallback.setAlignment(Qt.AlignCenter)
        self._fallback.setObjectName("tagline")
        self._fallback.setWordWrap(True)
        self._stack.addWidget(self._fallback)  # index 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._stack, 1)

        if not assets_ready():
            self._fallback.setText(
                "SuperSplat assets missing — reinstall InstaSplat GUI package."
            )
            return
        if not webengine_available():
            self._fallback.setText(
                "Qt WebEngine not available. Install full PySide6 (with WebEngine) "
                "for the SuperSplat live viewer."
            )
            return
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView

            self._server.start()
            self._view = QWebEngineView(self)
            self._view.setContextMenuPolicy(Qt.NoContextMenu)
            self._stack.addWidget(self._view)  # index 1
            self._stack.setCurrentIndex(0)
            self.status.emit(f"SuperSplat viewer {viewer_version()} ready")
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to start SuperSplat WebEngine view: %s", exc)
            self._view = None
            self._fallback.setText(f"SuperSplat viewer failed to start:\n{exc}")

    @property
    def available(self) -> bool:
        return self._view is not None

    def load_ply(self, path: Path | None, *, title: str = "") -> None:
        """Point the viewer at a Gaussian PLY (typically ``live.ply`` / ``scene.ply``)."""
        if path is not None:
            path = Path(path)
        self._ply_path = path if path is not None and path.is_file() else None

        if self._view is None:
            if self._ply_path is not None:
                self._fallback.setText(
                    f"{title or self._ply_path.name}\n"
                    "(SuperSplat WebEngine unavailable — open PLY externally)"
                )
            else:
                self._fallback.setText(title or "Waiting for training live.ply…")
            self._stack.setCurrentIndex(0)
            self.status.emit(self._fallback.text())
            return

        if self._ply_path is None:
            self._fallback.setText(title or "Waiting for training live.ply…")
            self._stack.setCurrentIndex(0)
            self.status.emit(self._fallback.text())
            return

        mtime = self._server.set_ply(self._ply_path)
        url = self._server.viewer_url(cache_bust=mtime)
        self._stack.setCurrentIndex(1)
        if url != self._last_url:
            self._last_url = url
            self._view.setUrl(QUrl(url))
        self.status.emit(title or self._ply_path.name)

    def clear(self) -> None:
        self.load_ply(None, title="No splat loaded")

    def shutdown(self) -> None:
        try:
            self._server.stop()
        except Exception:  # noqa: BLE001
            pass
