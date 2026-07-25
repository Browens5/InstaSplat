"""Mac desktop GUI for InstaSplat."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
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


STYLE = """
QWidget {
  background-color: #0f1412;
  color: #e7efe9;
  font-size: 14px;
}
QMainWindow {
  background-color: #0f1412;
}
QLabel#brand {
  color: #9ef0c8;
  font-size: 34px;
  font-weight: 700;
  letter-spacing: 0.5px;
}
QLabel#tagline {
  color: #9bb0a4;
  font-size: 13px;
}
QLabel#status {
  color: #9ef0c8;
  font-size: 14px;
  font-weight: 600;
}
QLabel#eta {
  color: #cfe7da;
  font-family: "SF Mono", Menlo, monospace;
  font-size: 13px;
}
QGroupBox {
  border: 1px solid #24312b;
  border-radius: 10px;
  margin-top: 14px;
  padding: 12px;
  background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
    stop:0 #121a17, stop:1 #18221d);
}
QGroupBox::title {
  subcontrol-origin: margin;
  left: 10px;
  padding: 0 6px;
  color: #9ef0c8;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QListWidget {
  background: #0c110f;
  border: 1px solid #2b3b34;
  border-radius: 8px;
  padding: 8px;
  selection-background-color: #1f6f4a;
}
QPushButton {
  background: #1f6f4a;
  color: #eafff3;
  border: none;
  border-radius: 8px;
  padding: 10px 16px;
  font-weight: 600;
}
QPushButton:hover { background: #27865a; }
QPushButton:disabled { background: #2a3530; color: #7d8c84; }
QPushButton#secondary {
  background: transparent;
  border: 1px solid #355246;
  color: #cfe7da;
}
QPushButton#warning {
  background: #6b4e16;
  color: #fff6df;
}
QPushButton#danger {
  background: #6b2a2a;
  color: #ffe8e8;
}
QProgressBar {
  border: 1px solid #2b3b34;
  border-radius: 6px;
  background: #0c110f;
  text-align: center;
  color: #e7efe9;
  height: 18px;
}
QProgressBar::chunk {
  background: #1f6f4a;
  border-radius: 5px;
}
QCheckBox { spacing: 8px; }
QTextEdit#log {
  font-family: "SF Mono", Menlo, monospace;
  font-size: 12px;
  background: #080b0a;
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


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("InstaSplat")
        self.resize(1020, 780)
        self.thread: QThread | None = None
        self.worker: Worker | None = None
        self.controller = RunController()
        self._last_event: ProgressEvent | None = None
        self._paused = False

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)

        brand = QLabel("InstaSplat")
        brand.setObjectName("brand")
        tag = QLabel("Long 360 → tiled Metal splat (YOLO MPS → COLMAP → Brush → ply/sog/spz)")
        tag.setObjectName("tagline")
        layout.addWidget(brand)
        layout.addWidget(tag)

        form_box = QGroupBox("Capture")
        form = QFormLayout(form_box)
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
        self.fps = QDoubleSpinBox()
        self.fps.setRange(0.1, 30.0)
        self.fps.setValue(6.0)
        self.fps.setSingleStep(0.5)
        opts_form.addRow("Base sample FPS", self.fps)

        self.mask_cb = QCheckBox("Mask people with YOLO")
        self.mask_cb.setChecked(True)
        opts_form.addRow(self.mask_cb)

        self.large_8k_cb = QCheckBox(
            "Mac long-360 tiled mode (best local: auto-chunk + Metal + GPS/gyro merge)"
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

        self.formats = QListWidget()
        self.formats.setSelectionMode(QListWidget.MultiSelection)
        for fmt in ["ply", "sog", "spz", "glb", "html", "csv", "compressed.ply"]:
            item = QListWidgetItem(fmt)
            self.formats.addItem(item)
            if fmt in ("ply", "sog", "spz"):
                item.setSelected(True)
        self.formats.setMaximumHeight(110)
        opts_form.addRow("Exports", self.formats)
        layout.addWidget(opts)

        status_box = QGroupBox("Run status")
        status_layout = QVBoxLayout(status_box)
        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("status")
        self.eta_label = QLabel("Task ETA — · Overall ETA — · Elapsed —")
        self.eta_label.setObjectName("eta")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.eta_label)
        status_layout.addWidget(self.progress_bar)
        layout.addWidget(status_box)

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
        layout.addLayout(btns)

        self.log = QTextEdit()
        self.log.setObjectName("log")
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

        note = QLabel(
            "Pause freezes the pipeline between stages and SIGSTOP's Brush/OpenSplat training. "
            "Resume continues; Stop terminates the run. "
            "See docs/MAC_LONG_360.md."
        )
        note.setWordWrap(True)
        note.setObjectName("tagline")
        layout.addWidget(note)

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

    def _build_config(self) -> PipelineConfig:
        inp = Path(self.input_edit.text().strip())
        if not inp:
            raise ValueError("Choose an input file")
        formats = [i.text() for i in self.formats.selectedItems()]
        if not formats:
            formats = ["ply", "sog"]
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
        self.run_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("Pause")
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Starting…")
        self.log.append(f"Starting job → {cfg.work_dir()}")
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
        self._refresh_eta()

    def _refresh_eta(self) -> None:
        ev = self._last_event
        if ev is None:
            return
        # Recompute live elapsed while a stage is running
        from instasplat.utils.progress import StageProgressTracker  # noqa: F401

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
