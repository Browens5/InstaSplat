"""OpenGL (or software) point-cloud viewer for sparse COLMAP / splat previews."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QSurfaceFormat,
    QWheelEvent,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

if TYPE_CHECKING:
    from instasplat.gui.geometry import PointCloud

try:
    from PySide6.QtGui import QMatrix4x4
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    from PySide6.QtOpenGL import (
        QOpenGLBuffer,
        QOpenGLShader,
        QOpenGLShaderProgram,
        QOpenGLVertexArrayObject,
    )

    _HAS_GL_WIDGET = True
except Exception:  # pragma: no cover
    QOpenGLWidget = object  # type: ignore[misc, assignment]
    _HAS_GL_WIDGET = False


VERTEX_SRC = """#version 410 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
uniform mat4 uMVP;
out vec3 vColor;
void main() {
    vColor = aColor;
    gl_Position = uMVP * vec4(aPos, 1.0);
    gl_PointSize = 2.5;
}
"""

FRAGMENT_SRC = """#version 410 core
in vec3 vColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(vColor, 1.0);
}
"""


def configure_default_surface_format() -> None:
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)


def _look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = center - eye
    f = f / (np.linalg.norm(f) + 1e-9)
    s = np.cross(f, up)
    s = s / (np.linalg.norm(s) + 1e-9)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    t = np.eye(4, dtype=np.float32)
    t[0, 3] = -eye[0]
    t[1, 3] = -eye[1]
    t[2, 3] = -eye[2]
    return (m @ t).astype(np.float32)


def _perspective(fovy_deg: float, aspect: float, z_near: float, z_far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / max(aspect, 1e-6)
    m[1, 1] = f
    m[2, 2] = (z_far + z_near) / (z_near - z_far)
    m[2, 3] = (2 * z_far * z_near) / (z_near - z_far)
    m[3, 2] = -1.0
    return m


class _OrbitState:
    def __init__(self) -> None:
        self.yaw = 0.6
        self.pitch = 0.35
        self.distance = 3.0
        self.target = np.zeros(3, dtype=np.float32)

    def eye(self) -> np.ndarray:
        cp = math.cos(self.pitch)
        return self.target + np.array(
            [
                self.distance * cp * math.sin(self.yaw),
                self.distance * math.sin(self.pitch),
                self.distance * cp * math.cos(self.yaw),
            ],
            dtype=np.float32,
        )


class _PointCloudViewMixin:
    """Shared API expected by ViewerPanel."""

    _title: str
    _status: str

    def set_cloud(self, cloud: Optional["PointCloud"], *, title: str = "") -> None:
        if cloud is None or cloud.n == 0:
            self.set_point_cloud(None, title=title)
            self._status = title or "No reconstruction yet"
            return
        label = title or f"{cloud.n:,} points"
        self.set_point_cloud(cloud.xyz, cloud.rgb, title=label, fit=True)
        self._status = label

    def status_text(self) -> str:
        return getattr(self, "_status", None) or getattr(self, "_title", "")


class SoftwarePointCloudWidget(_PointCloudViewMixin, QWidget):
    """CPU-projected point cloud — always available for sparse / splat centers."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(280, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._xyz: Optional[np.ndarray] = None
        self._rgb: Optional[np.ndarray] = None
        self._title = "No reconstruction yet"
        self._status = self._title
        self._orbit = _OrbitState()
        self._last_pos: Optional[QPoint] = None
        self._image: Optional[QImage] = None
        self._dirty = True
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._maybe_redraw)
        self._timer.start()

    def set_point_cloud(
        self,
        xyz: Optional[np.ndarray],
        rgb: Optional[np.ndarray] = None,
        *,
        title: str = "",
        fit: bool = True,
    ) -> None:
        if xyz is None or len(xyz) == 0:
            self._xyz = None
            self._rgb = None
            self._title = title or "No reconstruction yet"
            self._status = self._title
            self._image = None
            self.update()
            return
        self._xyz = np.asarray(xyz, dtype=np.float32)
        self._rgb = None if rgb is None else np.asarray(rgb, dtype=np.float32)
        self._title = title or f"{len(self._xyz):,} points"
        self._status = self._title
        if fit:
            center = self._xyz.mean(axis=0)
            extent = float(np.linalg.norm(self._xyz - center, axis=1).max() + 1e-6)
            self._orbit.target = center.astype(np.float32)
            self._orbit.distance = max(1.5 * extent, 0.5)
        self._dirty = True
        self.update()

    def clear(self) -> None:
        self.set_point_cloud(None)

    def _maybe_redraw(self) -> None:
        if self._dirty:
            self._render()
            self._dirty = False
            self.update()

    def _mvp(self) -> np.ndarray:
        aspect = max(self.width() / max(self.height(), 1), 0.1)
        view = _look_at(
            self._orbit.eye(),
            self._orbit.target,
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        proj = _perspective(50.0, aspect, 0.05, 500.0)
        return proj @ view

    def _render(self) -> None:
        w, h = max(self.width(), 1), max(self.height(), 1)
        img = QImage(w, h, QImage.Format.Format_RGB32)
        img.fill(QColor(12, 18, 14))
        if self._xyz is None:
            self._image = img
            return
        mvp = self._mvp()
        n = len(self._xyz)
        step = max(1, n // 80_000)
        pts = self._xyz[::step]
        ones = np.ones((len(pts), 1), dtype=np.float32)
        homo = np.concatenate([pts, ones], axis=1)
        clip = (mvp @ homo.T).T
        w_c = clip[:, 3:4]
        w_c = np.where(np.abs(w_c) < 1e-8, 1e-8, w_c)
        ndc = clip[:, :3] / w_c
        mask = (
            (ndc[:, 2] > -1.0)
            & (ndc[:, 2] < 1.0)
            & (np.abs(ndc[:, 0]) < 1.15)
            & (np.abs(ndc[:, 1]) < 1.15)
        )
        ndc = ndc[mask]
        if len(ndc) == 0:
            self._image = img
            return
        xs = ((ndc[:, 0] * 0.5 + 0.5) * (w - 1)).astype(np.int32)
        ys = ((1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (h - 1)).astype(np.int32)
        order = np.argsort(ndc[:, 2])[::-1]
        xs, ys = xs[order], ys[order]
        if self._rgb is not None:
            cols = (np.clip(self._rgb[::step][mask][order], 0, 1) * 255).astype(np.uint8)
        else:
            cols = np.full((len(xs), 3), 180, dtype=np.uint8)

        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs_v, ys_v, cols_v = xs[valid], ys[valid], cols[valid]
        # Compose BGRA buffer then write back (PySide6 bits() may be memoryview)
        buf = np.zeros((h, w, 4), dtype=np.uint8)
        buf[:, :, 3] = 255
        # dark green-black background already from fill; keep alpha
        bg = np.array([14, 18, 12, 255], dtype=np.uint8)  # B,G,R,A
        buf[:, :] = bg
        buf[ys_v, xs_v, 0] = cols_v[:, 2]
        buf[ys_v, xs_v, 1] = cols_v[:, 1]
        buf[ys_v, xs_v, 2] = cols_v[:, 0]
        buf[ys_v, xs_v, 3] = 255
        out = QImage(buf.data, w, h, w * 4, QImage.Format.Format_RGB32)
        self._image = out.copy()  # detach from numpy buffer

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(12, 18, 14))
        if self._image is not None:
            p.drawImage(0, 0, self._image)
        p.setPen(QColor(196, 214, 190))
        p.drawText(12, 22, self._title)
        if self._xyz is None:
            p.setPen(QColor(140, 160, 140))
            p.drawText(
                12,
                48,
                "Sparse cloud appears after COLMAP mapper; splat PLY during training.",
            )
        p.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._last_pos = event.position().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._last_pos is None or not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        cur = event.position().toPoint()
        dx = cur.x() - self._last_pos.x()
        dy = cur.y() - self._last_pos.y()
        self._last_pos = cur
        self._orbit.yaw += dx * 0.01
        self._orbit.pitch = float(np.clip(self._orbit.pitch + dy * 0.01, -1.4, 1.4))
        self._dirty = True

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        factor = 0.9 if delta > 0 else 1.1
        self._orbit.distance = float(np.clip(self._orbit.distance * factor, 0.05, 500.0))
        self._dirty = True

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._dirty = True


if _HAS_GL_WIDGET:

    class PointCloudGLWidget(_PointCloudViewMixin, QOpenGLWidget):
        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self.setMinimumSize(280, 220)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self._program: Optional[QOpenGLShaderProgram] = None
            self._vao: Optional[QOpenGLVertexArrayObject] = None
            self._vbo: Optional[QOpenGLBuffer] = None
            self._n_points = 0
            self._title = "No reconstruction yet"
            self._status = self._title
            self._orbit = _OrbitState()
            self._last_pos: Optional[QPoint] = None
            self._pending: Optional[tuple[np.ndarray, Optional[np.ndarray], str, bool]] = None
            self._gl_ok = False

        def set_point_cloud(
            self,
            xyz: Optional[np.ndarray],
            rgb: Optional[np.ndarray] = None,
            *,
            title: str = "",
            fit: bool = True,
        ) -> None:
            if xyz is None or len(xyz) == 0:
                self._pending = None
                self._n_points = 0
                self._title = title or "No reconstruction yet"
                self._status = self._title
                self.update()
                return
            xyz_a = np.asarray(xyz, dtype=np.float32)
            rgb_a = None if rgb is None else np.asarray(rgb, dtype=np.float32)
            self._title = title or f"{len(xyz_a):,} points"
            self._status = self._title
            self._pending = (xyz_a, rgb_a, self._title, fit)
            self.update()

        def clear(self) -> None:
            self.set_point_cloud(None)

        def initializeGL(self) -> None:  # noqa: N802
            try:
                self._program = QOpenGLShaderProgram(self)
                if not self._program.addShaderFromSourceCode(
                    QOpenGLShader.ShaderTypeBit.Vertex, VERTEX_SRC
                ):
                    raise RuntimeError(self._program.log())
                if not self._program.addShaderFromSourceCode(
                    QOpenGLShader.ShaderTypeBit.Fragment, FRAGMENT_SRC
                ):
                    raise RuntimeError(self._program.log())
                if not self._program.link():
                    raise RuntimeError(self._program.log())
                self._vao = QOpenGLVertexArrayObject(self)
                self._vao.create()
                self._vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
                self._vbo.create()
                self._gl_ok = True
                funcs = self.context().functions()
                funcs.glClearColor(0.05, 0.07, 0.055, 1.0)
                funcs.glEnable(0x0B71)  # GL_DEPTH_TEST
                funcs.glEnable(0x8642)  # GL_PROGRAM_POINT_SIZE
            except Exception:
                self._gl_ok = False

        def _upload(self, xyz: np.ndarray, rgb: Optional[np.ndarray], fit: bool) -> None:
            if not self._gl_ok or self._program is None or self._vao is None or self._vbo is None:
                return
            n = len(xyz)
            if rgb is None or rgb.shape[0] != n:
                rgb = np.full((n, 3), 0.75, dtype=np.float32)
            interleaved = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
            self._vao.bind()
            self._vbo.bind()
            self._vbo.allocate(interleaved.tobytes(), interleaved.nbytes)
            self._program.bind()
            self._program.enableAttributeArray(0)
            self._program.setAttributeBuffer(0, 0x1406, 0, 3, 24)  # GL_FLOAT
            self._program.enableAttributeArray(1)
            self._program.setAttributeBuffer(1, 0x1406, 12, 3, 24)
            self._program.release()
            self._vbo.release()
            self._vao.release()
            self._n_points = n
            if fit:
                center = xyz.mean(axis=0)
                extent = float(np.linalg.norm(xyz - center, axis=1).max() + 1e-6)
                self._orbit.target = center.astype(np.float32)
                self._orbit.distance = max(1.5 * extent, 0.5)

        def paintGL(self) -> None:  # noqa: N802
            funcs = self.context().functions()
            funcs.glClear(0x00004000 | 0x00000100)
            if self._pending is not None:
                xyz, rgb, title, fit = self._pending
                self._title = title
                self._status = title
                self._upload(xyz, rgb, fit)
                self._pending = None
            if not self._gl_ok or self._n_points == 0 or self._program is None or self._vao is None:
                painter = QPainter(self)
                painter.setPen(QColor(196, 214, 190))
                painter.drawText(12, 22, self._title)
                painter.end()
                return
            aspect = max(self.width() / max(self.height(), 1), 0.1)
            view = _look_at(
                self._orbit.eye(),
                self._orbit.target,
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
            )
            proj = _perspective(50.0, aspect, 0.05, 500.0)
            mvp = (proj @ view).astype(np.float32)
            qm = QMatrix4x4(
                float(mvp[0, 0]),
                float(mvp[0, 1]),
                float(mvp[0, 2]),
                float(mvp[0, 3]),
                float(mvp[1, 0]),
                float(mvp[1, 1]),
                float(mvp[1, 2]),
                float(mvp[1, 3]),
                float(mvp[2, 0]),
                float(mvp[2, 1]),
                float(mvp[2, 2]),
                float(mvp[2, 3]),
                float(mvp[3, 0]),
                float(mvp[3, 1]),
                float(mvp[3, 2]),
                float(mvp[3, 3]),
            )
            self._program.bind()
            self._program.setUniformValue("uMVP", qm)
            self._vao.bind()
            funcs.glDrawArrays(0x0000, 0, self._n_points)  # GL_POINTS
            self._vao.release()
            self._program.release()
            painter = QPainter(self)
            painter.setPen(QColor(220, 230, 210))
            painter.drawText(12, 22, self._title)
            painter.end()

        def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
            self._last_pos = event.position().toPoint()

        def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
            if self._last_pos is None or not (event.buttons() & Qt.MouseButton.LeftButton):
                return
            cur = event.position().toPoint()
            dx = cur.x() - self._last_pos.x()
            dy = cur.y() - self._last_pos.y()
            self._last_pos = cur
            self._orbit.yaw += dx * 0.01
            self._orbit.pitch = float(np.clip(self._orbit.pitch + dy * 0.01, -1.4, 1.4))
            self.update()

        def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
            delta = event.angleDelta().y()
            factor = 0.9 if delta > 0 else 1.1
            self._orbit.distance = float(np.clip(self._orbit.distance * factor, 0.05, 500.0))
            self.update()

else:

    class PointCloudGLWidget(SoftwarePointCloudWidget):  # type: ignore[no-redef]
        pass


def create_point_cloud_widget(parent: Optional[QWidget] = None) -> QWidget:
    """Prefer OpenGL on real displays; use software projection offscreen / when forced."""
    import os
    import sys

    force_sw = os.environ.get("INSTASPLAT_SOFTWARE_VIEWER", "").lower() in {
        "1",
        "true",
        "yes",
    }
    platform = os.environ.get("QT_QPA_PLATFORM", "").lower()
    offscreen = platform in {"offscreen", "minimal", "null"}
    if force_sw or offscreen or not _HAS_GL_WIDGET:
        return SoftwarePointCloudWidget(parent)
    # Linux headless cloud agents often lack a usable GLX/EGL context
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get(
        "WAYLAND_DISPLAY"
    ):
        return SoftwarePointCloudWidget(parent)
    try:
        return PointCloudGLWidget(parent)
    except Exception:
        return SoftwarePointCloudWidget(parent)


class ViewerFallback(SoftwarePointCloudWidget):
    """Alias kept for imports."""

    pass
