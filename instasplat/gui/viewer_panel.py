"""Live reconstruction viewer + artifact browser for the InstaSplat GUI."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from instasplat.gui.geometry import (
    PointCloud,
    discover_colmap_model,
    discover_splat_ply,
    find_latest_ply,
    load_colmap_sparse,
    load_splat_ply,
)
from instasplat.gui.gl_view import create_point_cloud_widget
from instasplat.utils.paths import JobPaths


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
OPEN_EXTS = {
    ".ply",
    ".sog",
    ".spz",
    ".glb",
    ".html",
    ".json",
    ".csv",
    ".txt",
    ".log",
    ".yaml",
    ".mp4",
    ".bin",
}


class ViewerPanel(QWidget):
    """Right-hand panel: live 3D preview + job artifact browser."""

    status = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._job_root: Path | None = None
        self._mode = "auto"  # auto | sparse | splat
        self._last_loaded: str | None = None
        self._last_mtime: float = -1.0

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        hdr = QHBoxLayout()
        title = QLabel("Live viewer")
        title.setObjectName("status")
        hdr.addWidget(title)
        hdr.addStretch(1)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Auto (stage-aware)", "auto")
        self.mode_combo.addItem("COLMAP sparse", "sparse")
        self.mode_combo.addItem("Gaussian splat", "splat")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        hdr.addWidget(self.mode_combo)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setObjectName("secondary")
        self.refresh_btn.clicked.connect(lambda: self.refresh(force=True))
        hdr.addWidget(self.refresh_btn)
        root.addLayout(hdr)

        self.caption = QLabel("Open a job or start a run to preview reconstructions.")
        self.caption.setObjectName("tagline")
        self.caption.setWordWrap(True)
        root.addWidget(self.caption)

        tabs = QTabWidget()
        # —— 3D tab ——
        view_page = QWidget()
        view_layout = QVBoxLayout(view_page)
        view_layout.setContentsMargins(0, 0, 0, 0)
        self.gl = create_point_cloud_widget()
        view_layout.addWidget(self.gl, 1)
        hint = QLabel("Drag to orbit · scroll/right-drag to zoom")
        hint.setObjectName("tagline")
        view_layout.addWidget(hint)
        tabs.addTab(view_page, "3D")

        # —— Artifacts tab ——
        art_page = QWidget()
        art_layout = QVBoxLayout(art_page)
        art_layout.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Vertical)
        self.artifact_list = QListWidget()
        self.artifact_list.currentItemChanged.connect(self._on_artifact_selected)
        self.artifact_list.itemDoubleClicked.connect(self._open_artifact_item)
        split.addWidget(self.artifact_list)
        preview_wrap = QWidget()
        preview_layout = QVBoxLayout(preview_wrap)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_label = QLabel("Select an artifact")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(160)
        self.preview_label.setObjectName("tagline")
        self.preview_label.setWordWrap(True)
        preview_layout.addWidget(self.preview_label, 1)
        btns = QHBoxLayout()
        self.open_btn = QPushButton("Open externally")
        self.open_btn.setObjectName("secondary")
        self.open_btn.clicked.connect(self._open_selected_artifact)
        self.show_3d_btn = QPushButton("Show in 3D")
        self.show_3d_btn.setObjectName("secondary")
        self.show_3d_btn.clicked.connect(self._show_selected_in_3d)
        btns.addWidget(self.open_btn)
        btns.addWidget(self.show_3d_btn)
        preview_layout.addLayout(btns)
        split.addWidget(preview_wrap)
        split.setSizes([220, 180])
        art_layout.addWidget(split)
        tabs.addTab(art_page, "Artifacts")
        root.addWidget(tabs, 1)

        self._tabs = tabs
        self._poll = QTimer(self)
        self._poll.setInterval(2000)
        self._poll.timeout.connect(lambda: self.refresh(force=False))

    # —— Public API ——
    def set_job_root(self, job_root: Path | None) -> None:
        self._job_root = Path(job_root) if job_root else None
        self._last_loaded = None
        self._last_mtime = -1.0
        self.refresh_artifacts()
        self.refresh(force=True)

    def start_live(self) -> None:
        if not self._poll.isActive():
            self._poll.start()

    def stop_live(self) -> None:
        self._poll.stop()

    def on_stage_progress(self, stage: str, status: str) -> None:
        """Called from MainWindow on ProgressEvent — bias Auto mode."""
        if self.mode_combo.currentData() != "auto":
            return
        if stage in {"sfm", "process_chunks", "align_chunks"} and status in {
            "running",
            "finished",
        }:
            self._mode = "sparse"
            self.refresh(force=status == "finished")
        elif stage in {"train", "export", "merge_chunks", "package"} and status in {
            "running",
            "finished",
        }:
            self._mode = "splat"
            self.refresh(force=status == "finished")

    def _resolve_mode(self) -> str:
        """Return sparse | splat based on combo + stage bias + available artifacts."""
        selected = self.mode_combo.currentData()
        if selected in {"sparse", "splat"}:
            return selected
        if self._mode in {"sparse", "splat"}:
            # Prefer stage bias, but fall forward to splat when a newer PLY exists
            if self._mode == "sparse":
                ply = discover_splat_ply(self._job_root) if self._job_root else None
                model = discover_colmap_model(self._job_root) if self._job_root else None
                if ply is not None and model is None:
                    return "splat"
                return "sparse"
            return "splat"
        # Pure auto: splat if present, else sparse
        if self._job_root and discover_splat_ply(self._job_root) is not None:
            return "splat"
        return "sparse"

    def refresh(self, *, force: bool = False) -> None:
        if self._job_root is None or not self._job_root.exists():
            self.caption.setText("No job folder selected.")
            return
        mode = self._resolve_mode()
        cloud: PointCloud | None = None
        title = ""
        mtime = -1.0
        path_key = ""

        if mode == "sparse":
            model = discover_colmap_model(self._job_root)
            if model is not None:
                pts = model / "points3D.bin"
                if not pts.exists():
                    pts = model / "points3D.txt"
                mtime = pts.stat().st_mtime if pts.exists() else model.stat().st_mtime
                path_key = f"sparse:{model}"
                if force or path_key != self._last_loaded or mtime > self._last_mtime:
                    cloud = load_colmap_sparse(model)
                    try:
                        rel = model.relative_to(self._job_root)
                    except ValueError:
                        rel = model
                    title = f"COLMAP sparse — {rel}"
        else:
            ply = discover_splat_ply(self._job_root)
            if ply is not None:
                mtime = ply.stat().st_mtime
                path_key = f"splat:{ply}"
                if force or path_key != self._last_loaded or mtime > self._last_mtime:
                    cloud = load_splat_ply(ply)
                    try:
                        rel = ply.relative_to(self._job_root)
                    except ValueError:
                        rel = ply
                    title = f"Splat — {rel}"

        if cloud is not None:
            self.gl.set_cloud(cloud, title=f"{title} ({cloud.n:,} pts)")
            self.caption.setText(self.gl.status_text())
            self._last_loaded = path_key
            self._last_mtime = mtime
            self.status.emit(self.caption.text())
        elif force:
            waiting = (
                "Waiting for COLMAP sparse model (after mapper)…"
                if mode == "sparse"
                else "Waiting for training PLY export…"
            )
            self.gl.set_cloud(None, title=waiting)
            self.caption.setText(waiting)

        if force:
            self.refresh_artifacts()

    def refresh_artifacts(self) -> None:
        self.artifact_list.clear()
        if self._job_root is None or not self._job_root.exists():
            return
        paths = JobPaths(self._job_root)
        entries: list[tuple[str, Path]] = []

        def add_glob(label: str, root: Path, pattern: str, limit: int = 40) -> None:
            if not root.exists():
                return
            found = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            for p in found[:limit]:
                try:
                    rel = p.relative_to(self._job_root)
                except ValueError:
                    rel = p
                entries.append((f"{label}: {rel}", p))

        add_glob("frame", paths.equirect_frames, "*.jpg", 12)
        add_glob("frame", paths.equirect_frames, "*.png", 12)
        add_glob("mask", paths.equirect_masks, "*.png", 12)
        add_glob("sfm image", paths.cubemap_images, "*.jpg", 8)
        for model in (
            paths.colmap_model,
            paths.scaled_model,
            paths.root / "03b_refine" / "sparse" / "0",
        ):
            for name in ("points3D.txt", "points3D.bin", "images.txt", "cameras.txt"):
                p = model / name
                if p.exists():
                    entries.append((f"sfm: {p.relative_to(self._job_root)}", p))
        add_glob("train", paths.brush_export, "**/*.ply", 20)
        add_glob("export", paths.export, "*", 30)
        add_glob("merged", paths.merged, "*", 20)
        if paths.chunks.is_dir():
            for c in sorted(paths.chunks.glob("chunk_*"))[:12]:
                ply = find_latest_ply(c / "06_export") or find_latest_ply(c / "05_train" / "exports")
                if ply:
                    entries.append((f"tile: {ply.relative_to(self._job_root)}", ply))

        # Deduplicate by path
        seen: set[str] = set()
        for label, path in entries:
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, str(path))
            self.artifact_list.addItem(item)

    # —— Internals ——
    def _on_mode_changed(self) -> None:
        data = self.mode_combo.currentData()
        if data != "auto":
            self._mode = data
        self.refresh(force=True)

    def _selected_path(self) -> Path | None:
        item = self.artifact_list.currentItem()
        if item is None:
            return None
        raw = item.data(Qt.UserRole)
        return Path(raw) if raw else None

    def _on_artifact_selected(self, current: QListWidgetItem | None, _prev) -> None:
        if current is None:
            return
        path = Path(current.data(Qt.UserRole))
        if not path.exists():
            self.preview_label.setText("Missing file")
            return
        if path.suffix.lower() in IMAGE_EXTS:
            pix = QPixmap(str(path))
            if pix.isNull():
                self.preview_label.setText(path.name)
            else:
                self.preview_label.setPixmap(
                    pix.scaled(
                        self.preview_label.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )
        else:
            size_kb = path.stat().st_size / 1024.0
            self.preview_label.setText(f"{path.name}\n{size_kb:.1f} KB\n{path}")

    def _open_selected_artifact(self) -> None:
        path = self._selected_path()
        if path is None or not path.exists():
            return
        QDesktopServices.openUrl(path.as_uri())

    def _open_artifact_item(self, item: QListWidgetItem) -> None:
        path = Path(item.data(Qt.UserRole))
        if path.exists():
            QDesktopServices.openUrl(path.as_uri())

    def _show_selected_in_3d(self) -> None:
        path = self._selected_path()
        if path is None or not path.exists():
            return
        cloud = None
        title = path.name
        if path.name.startswith("points3D") or path.parent.name in {"0", "0_txt"}:
            model = path.parent
            cloud = load_colmap_sparse(model)
            title = f"COLMAP — {model.name}"
        elif path.suffix.lower() == ".ply":
            cloud = load_splat_ply(path)
            title = f"Splat — {path.name}"
        if cloud is None:
            self.caption.setText(f"Cannot preview {path.name} in 3D")
            return
        self.gl.set_cloud(cloud, title=f"{title} ({cloud.n:,} pts)")
        self.caption.setText(self.gl.status_text())
        self._tabs.setCurrentIndex(0)
        self.status.emit(self.caption.text())
