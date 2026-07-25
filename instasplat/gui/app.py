"""Mac desktop GUI for InstaSplat."""

from __future__ import annotations

import sys
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from instasplat.config import PipelineConfig
from instasplat.pipeline import Pipeline
from instasplat.utils.control import RunController
from instasplat.utils.deps import report_dict
from instasplat.utils.progress import ProgressEvent, format_duration
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

    def set_progress(self, frac: float, *, state: str = "idle") -> None:
        value = int(max(0.0, min(1.0, frac)) * 1000)
        self.bar.setValue(value)
        if state == "done":
            self.bar.setObjectName("taskDone")
            self.bar.setFormat("done")
        elif state == "active":
            self.bar.setObjectName("taskActive")
            self.bar.setFormat("%p%")
        elif state == "pending":
            self.bar.setObjectName("task")
            self.bar.setFormat("…")
            self.bar.setValue(0)
        else:
            self.bar.setObjectName("task")
            self.bar.setFormat("%p%" if value else "")
        # Re-apply stylesheet for objectName changes
        self.bar.style().unpolish(self.bar)
        self.bar.style().polish(self.bar)

    def reset_progress(self) -> None:
        self.set_progress(0.0, state="idle")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("InstaSplat")
        self.resize(980, 720)
        self.setMinimumSize(720, 480)
        self.thread: QThread | None = None
        self.worker: Worker | None = None
        self.controller = RunController()
        self._last_event: ProgressEvent | None = None
        self._paused = False
        self._run_stages: list[str] = []
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

        opts = QGroupBox("Pipeline options")
        opts_form = QFormLayout(opts)
        opts_form.setSpacing(10)
        self.fps = QDoubleSpinBox()
        self.fps.setRange(0.1, 30.0)
        self.fps.setValue(6.0)
        self.fps.setSingleStep(0.5)
        opts_form.addRow("Base sample FPS", self.fps)

        self.mask_cb = QCheckBox("Mask people with YOLO")
        self.mask_cb.setChecked(True)
        opts_form.addRow(self.mask_cb)

        self.large_8k_cb = QCheckBox(
            "Mac long-360 tiled mode (auto-chunk + Metal + GPS/gyro merge)"
        )
        self.large_8k_cb.setChecked(True)
        opts_form.addRow(self.large_8k_cb)

        self.trainer = QComboBox()
        self.trainer.addItems(["brush", "opensplat"])
        opts_form.addRow("Trainer", self.trainer)

        self.refine_cb = QCheckBox("Refine poses (COLMAP BA + GPS/gyro blend)")
        self.refine_cb.setChecked(True)
        opts_form.addRow(self.refine_cb)

        self.lod_cb = QCheckBox("Streamed LOD export (lod-meta.json)")
        self.lod_cb.setChecked(True)
        opts_form.addRow(self.lod_cb)

        self.cloud_cb = QCheckBox("Write cloud_job.json + quality.json packaging")
        self.cloud_cb.setChecked(True)
        opts_form.addRow(self.cloud_cb)

        self.sfm_mode = QComboBox()
        self.sfm_mode.addItems(["perspective_cubemap", "equirectangular", "auto"])
        opts_form.addRow("SfM mode", self.sfm_mode)

        self.scale_mode = QComboBox()
        self.scale_mode.addItems(["gps", "none", "known_distance", "stereo_baseline"])
        opts_form.addRow("Metric scale", self.scale_mode)

        self.steps = QSpinBox()
        self.steps.setRange(1000, 100000)
        self.steps.setSingleStep(1000)
        self.steps.setValue(20000)
        opts_form.addRow("Train steps / chunk", self.steps)

        # Export formats as checkboxes
        export_wrap = QWidget()
        export_layout = QHBoxLayout(export_wrap)
        export_layout.setContentsMargins(0, 0, 0, 0)
        export_layout.setSpacing(12)
        self.format_cbs: dict[str, QCheckBox] = {}
        for fmt in ["ply", "sog", "spz", "glb", "html", "csv", "compressed.ply"]:
            cb = QCheckBox(fmt)
            cb.setChecked(fmt in ("ply", "sog", "spz"))
            self.format_cbs[fmt] = cb
            export_layout.addWidget(cb)
        export_layout.addStretch(1)
        opts_form.addRow("Exports", export_wrap)
        layout.addWidget(opts)

        # Stages: checkboxes + per-task progress
        stages_box = QGroupBox("Pipeline stages")
        stages_layout = QVBoxLayout(stages_box)
        stages_layout.setSpacing(4)
        hint = QLabel("Check the stages to run. Progress fills per task while the pipeline runs.")
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
            "Pause freezes the pipeline between stages and SIGSTOP's Brush/OpenSplat training. "
            "Resume continues; Stop terminates the run. "
            "See docs/MAC_LONG_360.md."
        )
        note.setWordWrap(True)
        note.setObjectName("tagline")
        layout.addWidget(note)
        layout.addStretch(1)

        scroll.setWidget(body)
        shell_layout.addWidget(scroll, 1)

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
        self.install_brush_btn = QPushButton("Install Brush")
        self.install_brush_btn.setObjectName("secondary")
        self.install_brush_btn.clicked.connect(self._install_brush)
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
        btns.addWidget(self.install_brush_btn)
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
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self.output_edit.setText(path)

    def _doctor(self) -> None:
        data = report_dict()
        lines = ["Dependency check:"]
        for d in data["deps"]:
            mark = "OK" if d["available"] else "MISSING"
            lines.append(f"  [{mark}] {d['name']}: {d.get('notes') or d.get('path') or ''}")
        lines.append("Ready stages: " + ", ".join(f"{k}={v}" for k, v in data["ready_stages"].items()))
        if not data["ready_stages"].get("train"):
            lines.append("Tip: click Install Brush (or run instasplat install-brush)")
        self.log.append("\n".join(lines))

    def _install_brush(self) -> None:
        self.install_brush_btn.setEnabled(False)
        self.log.append("Installing Brush (cargo release build)…")
        QApplication.processEvents()
        try:
            from instasplat.utils.brush_install import install_brush

            result = install_brush()
            self.log.append(result.message)
            if result.ok:
                QMessageBox.information(self, "InstaSplat", result.message)
            else:
                QMessageBox.warning(self, "InstaSplat", result.message)
        except Exception as exc:  # noqa: BLE001
            self.log.append(f"Brush install failed: {exc}")
            QMessageBox.critical(self, "InstaSplat", str(exc))
        finally:
            self.install_brush_btn.setEnabled(True)

    def _select_mode_stages(self) -> None:
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

        # In-stage fraction from elapsed vs ETA
        if ev.status == "finished" and ev.stage_eta_sec == 0.0:
            cur_frac = 1.0
        else:
            rem = ev.stage_eta_sec if ev.stage_eta_sec is not None else 0.0
            denom = ev.stage_elapsed_sec + max(rem, 0.0)
            cur_frac = (ev.stage_elapsed_sec / denom) if denom > 0 else 0.0
            cur_frac = min(0.99, cur_frac)

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
                    row.set_progress(cur_frac, state="active")
            else:
                row.set_progress(0.0, state="pending")

        # Sticky current-task bar
        self.task_progress.setFormat(f"{ev.stage}  %p%")
        if ev.status == "finished" and cur_idx >= len(run) - 1:
            self.task_progress.setValue(1000)
            self.task_progress.setFormat("complete")
        else:
            self.task_progress.setValue(int(cur_frac * 1000) if ev.status != "finished" else 1000)

    def _build_config(self) -> PipelineConfig:
        inp = Path(self.input_edit.text().strip())
        if not self.input_edit.text().strip():
            raise ValueError("Choose an input file")
        formats = [fmt for fmt, cb in self.format_cbs.items() if cb.isChecked()]
        if not formats:
            formats = ["ply", "sog"]
        stages = self._selected_stages()
        if not stages:
            raise ValueError("Select at least one pipeline stage")
        cfg = PipelineConfig(
            input_path=inp,
            output_dir=Path(self.output_edit.text().strip() or "./runs"),
            project_name=self.name_edit.text().strip() or "instasplat_job",
        )
        if self.large_8k_cb.isChecked():
            cfg.enable_mac_long_360_defaults()
            cfg.chunk.base_fps = float(self.fps.value())
        else:
            cfg.extract.fps = float(self.fps.value())
        cfg.stages = stages
        cfg.mask.enabled = self.mask_cb.isChecked()
        cfg.train.backend = self.trainer.currentText()  # type: ignore[assignment]
        cfg.refine.enabled = self.refine_cb.isChecked()
        cfg.export.streamed_lod = self.lod_cb.isChecked()
        cfg.package.cloud_manifest = self.cloud_cb.isChecked()
        cfg.package.quality_report = self.cloud_cb.isChecked()
        cfg.package.cpu_lod = self.cloud_cb.isChecked()
        cfg.sfm.mode = self.sfm_mode.currentText()  # type: ignore[assignment]
        cfg.scale.mode = self.scale_mode.currentText()  # type: ignore[assignment]
        cfg.train.total_steps = int(self.steps.value())
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
        self._run_stages = list(cfg.stages)
        self._reset_stage_bars(self._run_stages)
        self.task_progress.setValue(0)
        self.task_progress.setFormat("starting…")
        self.progress_bar.setValue(0)
        self.run_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("Pause")
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Starting…")
        self.log.append(f"Starting job → {cfg.work_dir()}")
        self.log.append(f"Stages: {', '.join(cfg.stages)}")
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
            self.pause_btn.setText("Resume")
            self.status_label.setText("Paused — pipeline + training frozen")
            self.log.append("⏸ Paused (stages wait; Brush/OpenSplat SIGSTOP)")
        else:
            self.controller.resume()
            self._paused = False
            self.pause_btn.setText("Pause")
            self.status_label.setText("Resumed")
            self.log.append("▶ Resumed")

    def _stop(self) -> None:
        self.controller.stop()
        self.status_label.setText("Stopping…")
        self.log.append("■ Stop requested")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

    def _on_progress(self, ev: object) -> None:
        if not isinstance(ev, ProgressEvent):
            return
        self._last_event = ev
        status = "PAUSED" if self._paused or ev.status == "paused" else ev.stage
        self.status_label.setText(
            f"{status}  ({ev.stage_index + 1}/{ev.stage_count}) — {ev.message}"
        )
        self.progress_bar.setValue(int(ev.overall_frac * 1000))
        self._update_stage_bars(ev)
        self._refresh_eta()

    def _refresh_eta(self) -> None:
        ev = self._last_event
        if ev is None:
            return
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
        self.run_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("Pause")
        self.stop_btn.setEnabled(False)
        self._paused = False
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
