"""Local HTTP server that hosts SuperSplat assets + the current live PLY."""

from __future__ import annotations

import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from instasplat.gui.supersplat import ASSETS_DIR, SETTINGS_PATH


class _ViewerState:
    """Thread-safe pointer to the PLY currently served as ``/content.ply``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.ply_path: Path | None = None
        self.ply_mtime: float = -1.0

    def set_ply(self, path: Path | None) -> float:
        with self._lock:
            if path is None or not path.is_file():
                self.ply_path = None
                self.ply_mtime = -1.0
                return -1.0
            self.ply_path = Path(path)
            try:
                self.ply_mtime = self.ply_path.stat().st_mtime
            except OSError:
                self.ply_mtime = -1.0
            return self.ply_mtime

    def snapshot(self) -> tuple[Path | None, float]:
        with self._lock:
            return self.ply_path, self.ply_mtime


def _make_handler(
    state: _ViewerState,
    assets_dir: Path,
    settings_path: Path,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = unquote(parsed.path or "/")
            if path in {"/", "/index.html"}:
                self._send_file(assets_dir / "index.html", "text/html; charset=utf-8")
                return
            if path == "/index.js":
                self._send_file(assets_dir / "index.js", "text/javascript; charset=utf-8")
                return
            if path == "/index.css":
                self._send_file(assets_dir / "index.css", "text/css; charset=utf-8")
                return
            if path == "/settings.json":
                self._send_file(settings_path, "application/json; charset=utf-8")
                return
            if path in {"/content.ply", "/scene.ply", "/live.ply"}:
                ply, _ = state.snapshot()
                if ply is None or not ply.is_file():
                    self.send_error(404, "No splat PLY loaded yet")
                    return
                self._send_file(ply, "application/octet-stream")
                return
            self.send_error(404, f"Not found: {path}")

        def _send_file(self, file_path: Path, content_type: str | None = None) -> None:
            try:
                data = file_path.read_bytes()
            except OSError:
                self.send_error(404, f"Missing {file_path.name}")
                return
            ctype = (
                content_type
                or mimetypes.guess_type(str(file_path))[0]
                or "application/octet-stream"
            )
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

    return Handler


class SuperSplatLocalServer:
    """Background ``127.0.0.1`` server for the embedded SuperSplat viewer."""

    def __init__(
        self,
        *,
        assets_dir: Path | None = None,
        settings_path: Path | None = None,
        host: str = "127.0.0.1",
    ) -> None:
        self.assets_dir = Path(assets_dir or ASSETS_DIR)
        self.settings_path = Path(settings_path or SETTINGS_PATH)
        self.host = host
        self.state = _ViewerState()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.port: int = 0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> str:
        if self._httpd is not None:
            return self.base_url
        handler = _make_handler(self.state, self.assets_dir, self.settings_path)
        httpd = ThreadingHTTPServer((self.host, 0), handler)
        self._httpd = httpd
        self.port = int(httpd.server_address[1])
        self._thread = threading.Thread(
            target=httpd.serve_forever,
            name="instasplat-supersplat-http",
            daemon=True,
        )
        self._thread.start()
        return self.base_url

    def stop(self) -> None:
        httpd = self._httpd
        self._httpd = None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        t = self._thread
        self._thread = None
        if t is not None and t.is_alive():
            t.join(timeout=2.0)

    def set_ply(self, path: Path | None) -> float:
        return self.state.set_ply(path)

    def viewer_url(
        self,
        *,
        noui: bool = True,
        webgl: bool = True,
        cache_bust: float | None = None,
    ) -> str:
        if self.port <= 0:
            self.start()
        qs = [
            "content=/content.ply",
            "settings=/settings.json",
        ]
        if noui:
            qs.append("noui")
        if webgl:
            # WebGL is more reliable inside Qt WebEngine than WebGPU
            qs.append("webgl")
        if cache_bust is not None and cache_bust >= 0:
            qs.append(f"v={int(cache_bust * 1000)}")
        return f"{self.base_url}/index.html?{'&'.join(qs)}"
