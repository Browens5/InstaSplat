"""Mac desktop GUI for InstaSplat."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt, QObject, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from instasplat.config import PipelineConfig
from instasplat.gui.gl_view import configure_default_surface_format
from instasplat.gui.viewer_panel import ViewerPanel
from instasplat.pipeline import Pipeline
from instasplat.utils.control import RunController
from instasplat.utils.deps import report_dict
from instasplat.utils.jobs import inspect_job, is_job_dir, load_job_config
from instasplat.utils.progress import (
    ProgressEvent,
    format_duration,
    live_progress_from_event,
    stage_fraction,
)
from instasplat.utils.stages import ALL_STAGES, SINGLE_STAGES, STAGE_HELP, TILED_STAGES


# Forest floor / moss / bark — warm earthy greens and browns (not neon, not purple)
STYLE = """
QWidget {
  background-color: #1a1712;
  color: #e8e0d4;
  font-size: 14px;
}
QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget {
  background-color: #1a1712;
}
QLabel#brand {
  color: #d4c4a0;
  font-size: 36px;
  font-weight: 700;
  letter-spacing: 0.6px;
}
QLabel#tagline {
  color: #9a8f7a;
  font-size: 13px;
}
QLabel#status {
  color: #c4b896;
  font-size: 14px;
  font-weight: 600;
}
QLabel#eta {
  color: #b8a990;
  font-family: "SF Mono", Menlo, monospace;
  font-size: 12px;
}
QLabel#stageName {
  color: #d8cbb0;
  font-size: 13px;
  min-width: 130px;
}
QLabel#stageHelp {
  color: #8a8070;
  font-size: 11px;
}
QLabel#jobBanner {
  color: #c4b896;
  font-size: 13px;
  font-weight: 600;
  padding: 8px 10px;
  background: #252018;
  border: 1px solid #4a3f32;
  border-radius: 6px;
}
QGroupBox {
  border: 1px solid #3d3428;
  border-radius: 8px;
  margin-top: 14px;
  padding: 14px 12px 12px 12px;
  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
    stop:0 #221e18, stop:0.55 #1e1b15, stop:1 #252018);
}
QGroupBox::title {
  subcontrol-origin: margin;
  left: 12px;
  padding: 0 8px;
  color: #a8b87a;
  font-weight: 600;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {
  background: #14110e;
  border: 1px solid #4a3f32;
  border-radius: 6px;
  padding: 8px;
  color: #e8e0d4;
  selection-background-color: #4a5c34;
}
QComboBox::drop-down {
  border: none;
  width: 22px;
}
QComboBox QAbstractItemView {
  background: #1e1b15;
  border: 1px solid #4a3f32;
  selection-background-color: #3d4f2a;
  color: #e8e0d4;
}
QPushButton {
  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
    stop:0 #5a6e3a, stop:1 #3d4f2a);
  color: #f2ead8;
  border: 1px solid #6a7e4a;
  border-radius: 6px;
  padding: 10px 16px;
  font-weight: 600;
}
QPushButton:hover {
  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
    stop:0 #6a8248, stop:1 #4a5f34);
}
QPushButton:disabled {
  background: #2a2620;
  border-color: #3a342c;
  color: #7a7060;
}
QPushButton#secondary {
  background: transparent;
  border: 1px solid #5a4e3e;
  color: #d4c4a0;
}
QPushButton#secondary:hover {
  background: #2a241c;
  border-color: #7a6a50;
}
QPushButton#warning {
  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
    stop:0 #8a6a30, stop:1 #6a4e20);
  border: 1px solid #a08040;
  color: #fff6df;
}
QPushButton#danger {
  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
    stop:0 #7a3a2a, stop:1 #5a2a1e);
  border: 1px solid #9a5040;
  color: #ffe8e0;
}
QProgressBar {
  border: 1px solid #4a3f32;
  border-radius: 5px;
  background: #14110e;
  text-align: center;
  color: #e8e0d4;
  height: 16px;
  font-size: 11px;
}
QProgressBar::chunk {
  background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 #4a5c34, stop:0.5 #6a8248, stop:1 #8a9a58);
  border-radius: 4px;
}
QProgressBar#overall {
  height: 22px;
  font-size: 12px;
  font-weight: 600;
}
QProgressBar#overall::chunk {
  background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 #3d4f2a, stop:0.45 #5a6e3a, stop:1 #c4a060);
}
QProgressBar#task {
  height: 14px;
  max-width: 220px;
  min-width: 120px;
}
QProgressBar#taskDone::chunk {
  background: #6a8248;
}
QProgressBar#taskActive::chunk {
  background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 #8a6a30, stop:1 #c4a060);
}
QCheckBox {
  spacing: 10px;
  color: #e0d6c4;
}
QCheckBox::indicator {
  width: 18px;
  height: 18px;
  border: 1px solid #5a4e3e;
  border-radius: 4px;
  background: #14110e;
}
QCheckBox::indicator:checked {
  background: #5a6e3a;
  border-color: #8a9a58;
}
QCheckBox::indicator:hover {
  border-color: #8a7a58;
}
QFrame#stageRow {
  background: transparent;
  border: none;
  padding: 2px 0;
}
QFrame#stickyFooter {
  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
    stop:0 #221e18, stop:1 #1a1712);
  border-top: 1px solid #3d3428;
}
QTextEdit#log {
  font-family: "SF Mono", Menlo, monospace;
  font-size: 12px;
  background: #100e0b;
  border: 1px solid #3d3428;
  border-radius: 6px;
  color: #c8bca8;
}
QTabWidget::pane {
  border: 1px solid #3d3428;
  border-radius: 6px;
  background: #1e1b15;
}
QTabBar::tab {
  background: #252018;
  color: #b8a990;
  padding: 8px 14px;
  border: 1px solid #3d3428;
  border-bottom: none;
  border-top-left-radius: 6px;
  border-top-right-radius: 6px;
  margin-right: 2px;
}
QTabBar::tab:selected {
  background: #3d4f2a;
  color: #f2ead8;
}
QListWidget {
  background: #14110e;
  border: 1px solid #4a3f32;
  border-radius: 6px;
  padding: 4px;
  color: #e8e0d4;
}
QListWidget::item:selected {
  background: #3d4f2a;
}
QScrollBar:vertical {
  background: #1a1712;
  width: 12px;
  margin: 0;
}
QScrollBar::handle:vertical {
  background: #4a3f32;
  border-radius: 5px;
  min-height: 28px;
}
QScrollBar::handle:vertical:hover {
  background: #6a5a42;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
  height: 0;
}
QScrollBar:horizontal {
  height: 0;
}
"""


class Worker(QObject):
    log = Signal(str)
    progress = Signal(object)  # ProgressEvent
    finished = Signal(bool, str)

    def __init__(self, cfg: PipelineConfig, controller: RunController):
        super().__init__()
        self.cfg = cfg
        self.controller = controller

    def run(self) -> None:
        def on_progress(ev: ProgressEvent) -> None:
            self.progress.emit(ev)
            # Heartbeats update bars only — avoid flooding the log widget
            if not ev.quiet:
                self.log.emit(ev.terminal_line())

        try:
            result = Pipeline(
                self.cfg,
                on_progress=on_progress,
                controller=self.controller,
            ).run()
            if result.success:
                self.finished.emit(True, str(result.paths.root))
            else:
                self.finished.emit(False, result.error or "Unknown error")
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))


class StageRow(QWidget):
    """Checkbox + label + per-task progress bar for one pipeline stage."""

    def __init__(self, name: str, help_text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.name = name
        self._help_text = help_text
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(10)

        self.checkbox = QCheckBox(name)
        self.checkbox.setToolTip(help_text)
        self.checkbox.setMinimumWidth(140)

        self.help_label = QLabel(help_text)
        self.help_label.setObjectName("stageHelp")
        self.help_label.setWordWrap(False)
        self.help_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.bar = QProgressBar()
        self.bar.setObjectName("task")
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        self.bar.setFormat("%p%")
        self.bar.setTextVisible(True)

        layout.addWidget(self.checkbox, 0)
        layout.addWidget(self.help_label, 1)
        layout.addWidget(self.bar, 0)

    def set_checked(self, checked: bool) -> None:
        self.checkbox.setChecked(checked)

    def is_checked(self) -> bool:
        return self.checkbox.isChecked()

    def set_detail(self, detail: str | None) -> None:
        """Show live phase text (e.g. COLMAP feature extraction) while active."""
        text = (detail or "").strip()
        if text:
            self.help_label.setText(text)
            self.help_label.setToolTip(text)
        else:
            self.help_label.setText(self._help_text)
            self.help_label.setToolTip(self._help_text)

    def set_progress(self, frac: float, *, state: str = "idle", detail: str | None = None) -> None:
        value = int(max(0.0, min(1.0, frac)) * 1000)
        # Avoid redundant polish churn on every 1s tick when state/value unchanged
        prev_state = getattr(self, "_ui_state", None)
        prev_value = getattr(self, "_ui_value", None)
        self._ui_state = state
        self._ui_value = value
        self.bar.setValue(value)
        if state == "done":
            self.bar.setObjectName("taskDone")
            self.bar.setFormat("done")
            self.set_detail(None)
        elif state == "active":
            self.bar.setObjectName("taskActive")
            pct = int(round(frac * 100))
            self.bar.setFormat(f"{pct}%")
            if detail is not None:
                self.set_detail(detail)
        elif state == "pending":
            self.bar.setObjectName("task")
            self.bar.setFormat("…")
            self.bar.setValue(0)
            self._ui_value = 0
            self.set_detail(None)
        else:
            self.bar.setObjectName("task")
            self.bar.setFormat(f"{int(round(frac * 100))}%" if value else "")
            if state == "idle":
                self.set_detail(None)
        if prev_state != state:
            self.bar.style().unpolish(self.bar)
            self.bar.style().polish(self.bar)
        elif prev_value != value:
            # Force a repaint when only the value changes (some styles skip it)
            self.bar.update()

    def reset_progress(self) -> None:
        self.set_progress(0.0, state="idle")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("InstaSplat")
        self.resize(1280, 820)
        self.setMinimumSize(1100, 640)
        self.thread: QThread | None = None
        self.worker: Worker | None = None
        self.controller = RunController()
        self._last_event: ProgressEvent | None = None
        self._paused = False
        self._run_active = False
        self._run_stages: list[str] = []
        self._run_wall_t0: float | None = None
        self._stage_wall_t0: float | None = None
        self._job_dir: Path | None = None
        self._suppress_stage_reset = False
        self.stage_rows: dict[str, StageRow] = {}

        shell = QWidget()
        self.setCentralWidget(shell)
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        # —— Scrollable body ——
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(28, 24, 28, 20)
        layout.setSpacing(14)

        brand = QLabel("InstaSplat")
        brand.setObjectName("brand")
        tag = QLabel(
            "Long 360 → tiled Metal splat  ·  forest-floor pipeline for Mac"
        )
        tag.setObjectName("tagline")
        layout.addWidget(brand)
        layout.addWidget(tag)

        form_box = QGroupBox("Capture")
        form = QFormLayout(form_box)
        form.setSpacing(10)

        open_row = QHBoxLayout()
        self.open_job_btn = QPushButton("Open previous run…")
        self.open_job_btn.setObjectName("secondary")
        self.open_job_btn.setToolTip(
            "Load an existing job folder (config.yaml or 00_ingest) to continue"
        )
        self.open_job_btn.clicked.connect(self._open_previous_run)
        self.clear_job_btn = QPushButton("New project")
        self.clear_job_btn.setObjectName("secondary")
        self.clear_job_btn.setEnabled(False)
        self.clear_job_btn.clicked.connect(self._clear_opened_job)
        open_row.addWidget(self.open_job_btn)
        open_row.addWidget(self.clear_job_btn)
        open_row.addStretch(1)
        form.addRow("Continue", open_row)

        self.job_banner = QLabel("New project — choose an input below, or open a previous run.")
        self.job_banner.setObjectName("jobBanner")
        self.job_banner.setWordWrap(True)
        form.addRow(self.job_banner)

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Select .insv or stitched equirect .mp4")
        browse = QPushButton("Browse")
        browse.setObjectName("secondary")
        browse.clicked.connect(self._browse_input)
        row = QHBoxLayout()
        row.addWidget(self.input_edit, 1)
        row.addWidget(browse)
        form.addRow("Input", row)

        self.output_edit = QLineEdit(str(Path.cwd() / "runs"))
        out_browse = QPushButton("Browse")
        out_browse.setObjectName("secondary")
        out_browse.clicked.connect(self._browse_output)
        row2 = QHBoxLayout()
        row2.addWidget(self.output_edit, 1)
        row2.addWidget(out_browse)
        form.addRow("Output", row2)

        self.name_edit = QLineEdit("instasplat_job")
        form.addRow("Project", self.name_edit)
        layout.addWidget(form_box)

        # —— Mode ——
        mode_box = QGroupBox("Mode")
        mode_form = QFormLayout(mode_box)
        mode_form.setSpacing(8)
        self.large_8k_cb = QCheckBox("Tiled long-360 (auto-chunk + merge)")
        self.large_8k_cb.setChecked(True)
        self.large_8k_cb.setToolTip(
            "Split long captures into overlapping tiles, train each, then merge."
        )
        mode_form.addRow(self.large_8k_cb)
        self.fps = QDoubleSpinBox()
        self.fps.setRange(0.1, 30.0)
        self.fps.setValue(6.0)
        self.fps.setSingleStep(0.5)
        self.fps.setToolTip("Frame sample rate for extract / tiling")
        mode_form.addRow("Sample FPS", self.fps)
        self.mask_cb = QCheckBox("Mask people (YOLO)")
        self.mask_cb.setChecked(True)
        mode_form.addRow(self.mask_cb)
        layout.addWidget(mode_box)

        # —— Reconstruction ——
        recon_box = QGroupBox("Reconstruction")
        recon_form = QFormLayout(recon_box)
        recon_form.setSpacing(8)
        self.sfm_mode = QComboBox()
        self.sfm_mode.addItems(["equirectangular", "auto", "perspective_cubemap"])
        self.sfm_mode.setToolTip(
            "equirectangular (default): full 360 in COLMAP ≥ 4.1.\n"
            "perspective_cubemap is a legacy opt-in."
        )
        recon_form.addRow("SfM mode", self.sfm_mode)
        self.sfm_mapper = QComboBox()
        self.sfm_mapper.addItems(["incremental", "global"])
        self.sfm_mapper.setToolTip(
            "incremental: classic COLMAP mapper.\n"
            "global: GLOMAP / global_mapper (needs COLMAP with GLOMAP)."
        )
        recon_form.addRow("Mapper", self.sfm_mapper)
        self.scale_mode = QComboBox()
        self.scale_mode.addItems(["gps", "none", "known_distance", "stereo_baseline"])
        recon_form.addRow("Metric scale", self.scale_mode)
        self.refine_cb = QCheckBox("Refine poses (BA + GPS/gyro)")
        self.refine_cb.setChecked(True)
        recon_form.addRow(self.refine_cb)
        layout.addWidget(recon_box)

        # —— Training ——
        train_box = QGroupBox("Training")
        train_form = QFormLayout(train_box)
        train_form.setSpacing(8)
        train_note = QLabel("metal_equirect · live viewer refreshes every ~100 steps (subsampled)")
        train_note.setObjectName("tagline")
        train_form.addRow(train_note)
        self.steps = QSpinBox()
        self.steps.setRange(100, 100_000)
        self.steps.setSingleStep(500)
        self.steps.setValue(15_000)
        self.steps.setToolTip("Optimization steps per job / tile")
        train_form.addRow("Steps", self.steps)
        self.max_gaussians = QSpinBox()
        self.max_gaussians.setRange(5_000, 500_000)
        self.max_gaussians.setSingleStep(5_000)
        self.max_gaussians.setValue(40_000)
        self.max_gaussians.setToolTip("Cap on Gaussian count (init + densify)")
        train_form.addRow("Max Gaussians", self.max_gaussians)
        self.sh_degree = QSpinBox()
        self.sh_degree.setRange(0, 3)
        self.sh_degree.setValue(1)
        self.sh_degree.setToolTip(
            "Spherical harmonics degree for view-dependent color (0=diffuse … 3=richest)"
        )
        train_form.addRow("SH degree (0–3)", self.sh_degree)
        self.export_every = QSpinBox()
        self.export_every.setRange(50, 1000)
        self.export_every.setSingleStep(50)
        self.export_every.setValue(500)
        self.export_every.setToolTip("Write equirect_XXXXXX.ply checkpoints every N steps")
        train_form.addRow("PLY export every", self.export_every)
        layout.addWidget(train_box)

        # —— Export ——
        export_box = QGroupBox("Export")
        export_form = QFormLayout(export_box)
        export_form.setSpacing(8)
        export_wrap = QWidget()
        export_layout = QHBoxLayout(export_wrap)
        export_layout.setContentsMargins(0, 0, 0, 0)
        export_layout.setSpacing(10)
        self.format_cbs: dict[str, QCheckBox] = {}
        for fmt in ["ply", "sog", "spz", "glb", "html"]:
            cb = QCheckBox(fmt)
            cb.setChecked(fmt in ("ply", "sog", "spz"))
            self.format_cbs[fmt] = cb
            export_layout.addWidget(cb)
        export_layout.addStretch(1)
        export_form.addRow("Formats", export_wrap)
        self.lod_cb = QCheckBox("Streamed LOD (lod-meta.json)")
        self.lod_cb.setChecked(True)
        export_form.addRow(self.lod_cb)
        self.cloud_cb = QCheckBox("Package manifests (cloud + quality)")
        self.cloud_cb.setChecked(True)
        export_form.addRow(self.cloud_cb)
        layout.addWidget(export_box)

        # Stages: checkboxes + per-task progress
        stages_box = QGroupBox("Pipeline stages")
        stages_layout = QVBoxLayout(stages_box)
        stages_layout.setSpacing(4)
        hint = QLabel(
            "Check the stages to run. Opening a previous run pre-checks remaining work "
            "(skip_existing keeps finished tiles/artifacts)."
        )
        hint.setObjectName("tagline")
        hint.setWordWrap(True)
        stages_layout.addWidget(hint)

        for name in ALL_STAGES:
            row_w = StageRow(name, STAGE_HELP.get(name, ""))
            self.stage_rows[name] = row_w
            stages_layout.addWidget(row_w)

        stage_btns = QHBoxLayout()
        sel_all = QPushButton("All for mode")
        sel_all.setObjectName("secondary")
        sel_all.clicked.connect(self._select_mode_stages)
        sel_clear = QPushButton("Clear")
        sel_clear.setObjectName("secondary")
        sel_clear.clicked.connect(self._clear_stages)
        stage_btns.addWidget(sel_all)
        stage_btns.addWidget(sel_clear)
        stage_btns.addStretch(1)
        stages_layout.addLayout(stage_btns)
        self.large_8k_cb.toggled.connect(lambda _=False: self._select_mode_stages())
        self._select_mode_stages()
        layout.addWidget(stages_box)

        note = QLabel(
            "Open a previous run to continue. Viewer: COLMAP sparse during SfM, "
            "then subsampled live.ply every ~100 train steps. Pause freezes; Stop terminates."
        )
        note.setWordWrap(True)
        note.setObjectName("tagline")
        layout.addWidget(note)
        layout.addStretch(1)

        scroll.setWidget(body)

        # Left: form · Right: live 3D + artifacts
        mid = QSplitter(Qt.Horizontal)
        mid.addWidget(scroll)
        self.viewer = ViewerPanel()
        self.viewer.status.connect(lambda msg: None)
        mid.addWidget(self.viewer)
        mid.setStretchFactor(0, 3)
        mid.setStretchFactor(1, 2)
        mid.setSizes([720, 480])
        shell_layout.addWidget(mid, 1)

        # —— Sticky footer: status, overall bar, buttons, log ——
        footer = QFrame()
        footer.setObjectName("stickyFooter")
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(28, 14, 28, 18)
        footer_layout.setSpacing(10)

        status_box = QGroupBox("Run status")
        status_layout = QVBoxLayout(status_box)
        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("status")
        self.eta_label = QLabel("Task ETA — · Overall ETA — · Elapsed —")
        self.eta_label.setObjectName("eta")
        self.eta_label.setWordWrap(True)

        task_row = QHBoxLayout()
        task_lbl = QLabel("Current task")
        task_lbl.setObjectName("stageName")
        self.task_progress = QProgressBar()
        self.task_progress.setObjectName("taskActive")
        self.task_progress.setRange(0, 1000)
        self.task_progress.setValue(0)
        self.task_progress.setFormat("—")
        task_row.addWidget(task_lbl)
        task_row.addWidget(self.task_progress, 1)

        overall_lbl = QLabel("Overall")
        overall_lbl.setObjectName("stageName")
        overall_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("overall")
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        overall_row.addWidget(overall_lbl)
        overall_row.addWidget(self.progress_bar, 1)

        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.eta_label)
        status_layout.addLayout(task_row)
        status_layout.addLayout(overall_row)
        footer_layout.addWidget(status_box)

        btns = QHBoxLayout()
        self.doctor_btn = QPushButton("Check dependencies")
        self.doctor_btn.setObjectName("secondary")
        self.doctor_btn.clicked.connect(self._doctor)
        self.run_btn = QPushButton("Run pipeline")
        self.run_btn.clicked.connect(self._run)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setObjectName("warning")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        btns.addWidget(self.doctor_btn)
        btns.addStretch(1)
        btns.addWidget(self.pause_btn)
        btns.addWidget(self.stop_btn)
        btns.addWidget(self.run_btn)
        footer_layout.addLayout(btns)

        self.log = QTextEdit()
        self.log.setObjectName("log")
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(100)
        self.log.setMaximumHeight(160)
        footer_layout.addWidget(self.log)

        shell_layout.addWidget(footer, 0)

        self._tick = QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._refresh_eta)
        self._tick.start()

    def _browse_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select INSV or MP4",
            "",
            "Insta360 / Video (*.insv *.mp4 *.mov);;All files (*)",
        )
        if path:
            self.input_edit.setText(path)

    def _browse_output(self) -> None:
        start = self.output_edit.text().strip() or str(Path.cwd() / "runs")
        path = QFileDialog.getExistingDirectory(self, "Select output directory", start)
        if path:
            self.output_edit.setText(path)

    def _open_previous_run(self) -> None:
        start = self.output_edit.text().strip() or str(Path.cwd() / "runs")
        path = QFileDialog.getExistingDirectory(
            self,
            "Open previous InstaSplat run (job folder)",
            start,
        )
        if not path:
            return
        job = Path(path)
        # If user picked the runs/ parent, ask them to pick a project subfolder
        if not is_job_dir(job):
            QMessageBox.warning(
                self,
                "InstaSplat",
                "Select a job folder (contains config.yaml or 00_ingest/), "
                "not the parent runs/ directory.",
            )
            return
        try:
            info = inspect_job(job)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "InstaSplat", str(exc))
            return
        self._apply_job(info)

    def _apply_job(self, info) -> None:
        self._job_dir = info.job_dir
        self._suppress_stage_reset = True
        try:
            self._apply_config_to_form(info.config)
            self._set_checked_stages(info.suggested_stages)
        finally:
            self._suppress_stage_reset = False
        self.clear_job_btn.setEnabled(True)
        self.run_btn.setText("Continue run")
        detail = " · ".join(info.details[:4])
        self.job_banner.setText(
            f"Continuing {info.job_dir.name} ({info.mode_label})"
            + (
                f" — tiles {info.tiles_done}/{info.tiles_total}"
                if info.tiled and info.tiles_total
                else ""
            )
            + f"\nSuggested: {', '.join(info.suggested_stages)}"
            + (f"\n{detail}" if detail else "")
        )
        self.status_label.setText(f"Opened job — {info.summary}")
        self.log.append(f"Opened previous run: {info.job_dir}")
        for line in info.details:
            self.log.append(f"  · {line}")
        self.log.append(f"Suggested stages: {', '.join(info.suggested_stages)}")
        self.viewer.set_job_root(info.job_dir)
        self.viewer.start_live()

    def _clear_opened_job(self) -> None:
        self._job_dir = None
        self.clear_job_btn.setEnabled(False)
        self.run_btn.setText("Run pipeline")
        self.job_banner.setText(
            "New project — choose an input below, or open a previous run."
        )
        self.status_label.setText("Idle")
        self._select_mode_stages()
        self.viewer.stop_live()
        self.viewer.set_job_root(None)
        self.log.append("Cleared opened job — starting a new project")

    def _apply_config_to_form(self, cfg: PipelineConfig) -> None:
        inp = cfg.input_path
        video = cfg.work_dir() / "00_ingest" / "equirect.mp4"
        if video.exists():
            self.input_edit.setText(str(video))
        elif inp.exists():
            self.input_edit.setText(str(inp))
        else:
            self.input_edit.setText(str(inp))
        self.output_edit.setText(str(cfg.output_dir))
        self.name_edit.setText(cfg.project_name)
        tiled = cfg.mode == "tiled" or cfg.chunk.enabled
        self.large_8k_cb.setChecked(tiled)
        if tiled:
            self.fps.setValue(float(cfg.chunk.base_fps))
        else:
            self.fps.setValue(float(cfg.extract.fps))
        self.mask_cb.setChecked(bool(cfg.mask.enabled))
        # Trainer is always metal_equirect
        self.refine_cb.setChecked(bool(cfg.refine.enabled))
        self.lod_cb.setChecked(bool(cfg.export.streamed_lod))
        self.cloud_cb.setChecked(
            bool(cfg.package.cloud_manifest or cfg.package.quality_report)
        )
        idx = self.sfm_mode.findText(cfg.sfm.mode)
        if idx >= 0:
            self.sfm_mode.setCurrentIndex(idx)
        idx = self.sfm_mapper.findText(cfg.sfm.mapper)
        if idx >= 0:
            self.sfm_mapper.setCurrentIndex(idx)
        idx = self.scale_mode.findText(cfg.scale.mode)
        if idx >= 0:
            self.scale_mode.setCurrentIndex(idx)
        self.steps.setValue(int(cfg.train.total_steps))
        self.max_gaussians.setValue(int(getattr(cfg.train, "max_gaussians", 40_000)))
        self.sh_degree.setValue(max(0, min(int(cfg.train.sh_degree), 3)))
        exp_every = int(cfg.train.export_every)
        self.export_every.setValue(max(50, min(exp_every, 1000)))
        wanted = set(cfg.export.formats or [])
        for fmt, cb in self.format_cbs.items():
            cb.setChecked(fmt in wanted)

    def _set_checked_stages(self, stages: list[str]) -> None:
        wanted = set(stages)
        for name, row in self.stage_rows.items():
            row.set_checked(name in wanted)

    def _doctor(self) -> None:
        data = report_dict()
        lines = ["Dependency check:"]
        for d in data["deps"]:
            mark = "OK" if d["available"] else "MISSING"
            lines.append(f"  [{mark}] {d['name']}: {d.get('notes') or d.get('path') or ''}")
        lines.append("Ready stages: " + ", ".join(f"{k}={v}" for k, v in data["ready_stages"].items()))
        if not data["ready_stages"].get("train"):
            lines.append("Tip: pip install torch (MPS wheel on Apple Silicon) for training")
        self.log.append("\n".join(lines))


    def _select_mode_stages(self) -> None:
        if self._suppress_stage_reset:
            return
        wanted = set(TILED_STAGES if self.large_8k_cb.isChecked() else SINGLE_STAGES)
        for name, row in self.stage_rows.items():
            row.set_checked(name in wanted)

    def _clear_stages(self) -> None:
        for row in self.stage_rows.values():
            row.set_checked(False)

    def _selected_stages(self) -> list[str]:
        names = [name for name, row in self.stage_rows.items() if row.is_checked()]
        # Prefer mode order so tiled jobs keep package after merge, etc.
        mode_order = TILED_STAGES if self.large_8k_cb.isChecked() else SINGLE_STAGES
        order = {s: i for i, s in enumerate(mode_order)}
        next_i = len(mode_order)
        for s in ALL_STAGES:
            if s not in order:
                order[s] = next_i
                next_i += 1
        names.sort(key=lambda n: order.get(n, 999))
        return names

    def _reset_stage_bars(self, stages: list[str]) -> None:
        selected = set(stages)
        for name, row in self.stage_rows.items():
            if name in selected:
                row.set_progress(0.0, state="pending")
            else:
                row.reset_progress()

    def _update_stage_bars(self, ev: ProgressEvent) -> None:
        run = self._run_stages or list(self.stage_rows.keys())
        # Map stage name → index in this run
        index_of = {n: i for i, n in enumerate(run)}
        cur_idx = index_of.get(ev.stage, ev.stage_index)
        detail = (ev.message or "").strip()

        if ev.status == "finished":
            cur_frac = 1.0
        else:
            cur_frac = stage_fraction(ev.stage_elapsed_sec, ev.stage_eta_sec)

        for name, row in self.stage_rows.items():
            if name not in index_of:
                continue
            idx = index_of[name]
            if ev.status == "finished" and name == ev.stage:
                row.set_progress(1.0, state="done")
            elif idx < cur_idx:
                row.set_progress(1.0, state="done")
            elif idx == cur_idx:
                if ev.status == "finished":
                    row.set_progress(1.0, state="done")
                else:
                    row.set_progress(cur_frac, state="active", detail=detail or None)
            else:
                row.set_progress(0.0, state="pending")

        # Sticky current-task bar — include COLMAP / phase detail
        if ev.status == "finished" and cur_idx >= len(run) - 1:
            self.task_progress.setValue(1000)
            self.task_progress.setFormat("complete")
        elif ev.status == "finished":
            self.task_progress.setValue(1000)
            self.task_progress.setFormat(f"{ev.stage} done")
        else:
            pct = int(round(cur_frac * 100))
            self.task_progress.setValue(int(cur_frac * 1000))
            phase = detail if detail and detail.lower() != f"starting {ev.stage}" else ""
            if phase:
                # Keep format readable in the narrow bar
                short = phase if len(phase) <= 56 else phase[:53] + "…"
                self.task_progress.setFormat(f"{ev.stage} · {short}  {pct}%")
            else:
                self.task_progress.setFormat(f"{ev.stage}  {pct}%")
            self.task_progress.update()

    def _build_config(self) -> PipelineConfig:
        formats = [fmt for fmt, cb in self.format_cbs.items() if cb.isChecked()]
        if not formats:
            formats = ["ply", "sog"]
        stages = self._selected_stages()
        if not stages:
            raise ValueError("Select at least one pipeline stage")

        if self._job_dir is not None:
            cfg = load_job_config(self._job_dir)
            # Keep job identity unless user edited project/output
            out = Path(self.output_edit.text().strip() or str(cfg.output_dir))
            name = self.name_edit.text().strip() or cfg.project_name
            cfg.output_dir = out
            cfg.project_name = name
            # Re-bind to opened folder if name/output still point at it
            if (out / name).resolve() != self._job_dir.resolve():
                # User moved project identity — still OK, but warn in log later
                pass
        else:
            inp_text = self.input_edit.text().strip()
            if not inp_text:
                raise ValueError("Choose an input file, or open a previous run")
            cfg = PipelineConfig(
                input_path=Path(inp_text),
                output_dir=Path(self.output_edit.text().strip() or "./runs"),
                project_name=self.name_edit.text().strip() or "instasplat_job",
            )
            if self.large_8k_cb.isChecked():
                cfg.enable_mac_long_360_defaults()

        inp_text = self.input_edit.text().strip()
        if inp_text:
            cfg.input_path = Path(inp_text)
        elif not cfg.input_path.exists():
            video = cfg.work_dir() / "00_ingest" / "equirect.mp4"
            if video.exists():
                cfg.input_path = video
            else:
                raise ValueError("Choose an input file (or open a job with ingested video)")

        if self.large_8k_cb.isChecked():
            if cfg.mode != "tiled" and self._job_dir is None:
                cfg.enable_mac_long_360_defaults()
            cfg.mode = "tiled"
            cfg.chunk.enabled = True
            cfg.chunk.base_fps = float(self.fps.value())
        else:
            cfg.mode = "single"
            cfg.chunk.enabled = False
            cfg.extract.fps = float(self.fps.value())

        cfg.stages = stages
        cfg.skip_existing = True
        cfg.mask.enabled = self.mask_cb.isChecked()
        cfg.train.backend = "metal_equirect"
        cfg.refine.enabled = self.refine_cb.isChecked()
        cfg.export.streamed_lod = self.lod_cb.isChecked()
        cfg.package.cloud_manifest = self.cloud_cb.isChecked()
        cfg.package.quality_report = self.cloud_cb.isChecked()
        cfg.package.cpu_lod = self.cloud_cb.isChecked()
        cfg.sfm.mode = self.sfm_mode.currentText()  # type: ignore[assignment]
        cfg.sfm.mapper = self.sfm_mapper.currentText()  # type: ignore[assignment]
        cfg.scale.mode = self.scale_mode.currentText()  # type: ignore[assignment]
        cfg.train.total_steps = int(self.steps.value())
        cfg.train.max_gaussians = int(self.max_gaussians.value())
        cfg.train.sh_degree = int(self.sh_degree.value())
        cfg.train.export_every = int(self.export_every.value())
        cfg.train.viewer_every = 100
        cfg.export.formats = formats  # type: ignore[assignment]
        return cfg

    def _run(self) -> None:
        try:
            cfg = self._build_config()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "InstaSplat", str(exc))
            return
        self.controller = RunController()
        self._paused = False
        self._run_active = True
        self._run_wall_t0 = time.time()
        self._stage_wall_t0 = self._run_wall_t0
        self._last_event = None
        self._run_stages = list(cfg.stages)
        self._reset_stage_bars(self._run_stages)
        self.task_progress.setValue(0)
        self.task_progress.setFormat("starting…")
        self.progress_bar.setValue(0)
        self.run_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("Pause")
        self.stop_btn.setEnabled(True)
        action = "Continuing" if self._job_dir is not None else "Starting"
        self.status_label.setText(f"{action}…")
        self.log.append(f"{action} job → {cfg.work_dir()}")
        self.log.append(f"Stages: {', '.join(cfg.stages)}")
        if cfg.skip_existing:
            self.log.append("skip_existing=on — finished tiles/artifacts will be reused")
        self.viewer.set_job_root(cfg.work_dir())
        self.viewer.start_live()
        self.thread = QThread()
        self.worker = Worker(cfg, self.controller)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.log.append)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.start()

    def _toggle_pause(self) -> None:
        if not self._paused:
            self.controller.pause()
            self._paused = True
            self.pause_btn.setText("Unpause")
            self.status_label.setText("Paused — pipeline + training frozen")
            self.log.append("⏸ Paused (stages wait at checkpoints)")
        else:
            self.controller.resume()
            self._paused = False
            self.pause_btn.setText("Pause")
            self.status_label.setText("Unpaused")
            self.log.append("▶ Unpaused")

    def _stop(self) -> None:
        self.controller.stop()
        self.status_label.setText("Stopping…")
        self.log.append("■ Stop requested")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

    def _on_progress(self, ev: object) -> None:
        if not isinstance(ev, ProgressEvent):
            return
        # New stage → reset wall clock used for live bar ticks
        if (
            self._last_event is None
            or self._last_event.stage != ev.stage
            or (self._last_event.status == "finished" and ev.status == "running")
        ):
            self._stage_wall_t0 = time.time() - max(0.0, ev.stage_elapsed_sec)
        prev_msg = self._last_event.message if self._last_event else None
        self._last_event = ev
        # Always refresh status for phase changes (incl. quiet COLMAP heartbeats)
        if (
            not ev.quiet
            or ev.status != "running"
            or (ev.message and ev.message != prev_msg)
        ):
            status = "PAUSED" if self._paused or ev.status == "paused" else ev.stage
            self.status_label.setText(
                f"{status}  ({ev.stage_index + 1}/{ev.stage_count}) — {ev.message}"
            )
        self.progress_bar.setValue(int(ev.overall_frac * 1000))
        self._update_stage_bars(ev)
        self._refresh_eta()
        if not ev.quiet or ev.status == "finished":
            self.viewer.on_stage_progress(ev.stage, ev.status)

    def _refresh_eta(self) -> None:
        ev = self._last_event
        if ev is None:
            return
        # While a stage is running, recompute bars/ETA from wall clock every tick
        if (
            self._run_active
            and not self._paused
            and ev.status == "running"
            and self._stage_wall_t0 is not None
            and self._run_wall_t0 is not None
        ):
            live = live_progress_from_event(
                ev,
                stage_wall_t0=self._stage_wall_t0,
                run_wall_t0=self._run_wall_t0,
            )
            self._last_event = live
            self.progress_bar.setValue(int(live.overall_frac * 1000))
            self._update_stage_bars(live)
            ev = live
        if self._paused:
            self.eta_label.setText(
                f"PAUSED · Task elapsed {format_duration(ev.stage_elapsed_sec)} · "
                f"Overall elapsed {format_duration(ev.overall_elapsed_sec)}"
            )
            return
        self.eta_label.setText(
            f"Task ETA {format_duration(ev.stage_eta_sec)} · "
            f"Overall ETA {format_duration(ev.overall_eta_sec)} · "
            f"Task elapsed {format_duration(ev.stage_elapsed_sec)} · "
            f"Overall elapsed {format_duration(ev.overall_elapsed_sec)}"
        )

    def _on_finished(self, ok: bool, message: str) -> None:
        self._run_active = False
        self.run_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("Pause")
        self.stop_btn.setEnabled(False)
        self._paused = False
        self.viewer.refresh(force=True)
        if ok:
            self.status_label.setText("Finished")
            self.progress_bar.setValue(1000)
            self.task_progress.setValue(1000)
            self.task_progress.setFormat("complete")
            for name in self._run_stages:
                row = self.stage_rows.get(name)
                if row:
                    row.set_progress(1.0, state="done")
            self.log.append(f"SUCCESS: {message}")
            QMessageBox.information(self, "InstaSplat", f"Finished.\n{message}")
        else:
            self.status_label.setText("Failed / stopped")
            self.log.append(f"FAILED: {message}")
            QMessageBox.critical(self, "InstaSplat", message)


def launch() -> None:
    configure_default_surface_format()
    app = QApplication(sys.argv)
    app.setApplicationName("InstaSplat")
    app.setStyle("Fusion")
    for family in ("Avenir Next", "Futura", "Helvetica Neue"):
        if family in QFontDatabase.families():
            app.setFont(QFont(family, 13))
            break
    app.setStyleSheet(STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch()
