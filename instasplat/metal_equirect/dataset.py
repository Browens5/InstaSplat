"""Assemble equirect training views from COLMAP + equirect frames."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from instasplat.metal_equirect.cameras import yaw_pitch_to_rotmat
from instasplat.stages.sfm import FACE_NAMES, FACE_YAW_PITCH
from instasplat.utils.paths import JobPaths
from instasplat.utils.scale import qvec_to_rotmat, read_images_txt, read_points3d_txt


_FACE_RE = re.compile(
    r"^(?P<stem>.+)_(?P<face>front|right|back|left|up|down)\.(jpg|jpeg|png)$",
    re.IGNORECASE,
)


@dataclass
class TrainView:
    name: str
    image_path: Path
    mask_path: Path | None
    R_w2c: np.ndarray  # 3x3
    t_w2c: np.ndarray  # 3
    width: int
    height: int


@dataclass
class EquirectDataset:
    views: list[TrainView]
    points_xyz: np.ndarray  # (P, 3)
    points_rgb: np.ndarray  # (P, 3) float 0-1
    equirect_dir: Path

    def __len__(self) -> int:
        return len(self.views)


def _face_index(name: str) -> int | None:
    try:
        return FACE_NAMES.index(name.lower())
    except ValueError:
        return None


def _lift_cubemap_pose_to_equirect(
    R_face: np.ndarray, t_face: np.ndarray, face: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a cubemap-face world-to-camera pose into equirect world-to-camera.

    COLMAP pose for face: x_face = R_face x_w + t_face
    Face frame relates to equirect by R_eq_to_face (yaw/pitch).
    x_face = R_eq_to_face x_eq  ⇒  R_eq = R_eq_to_face.T @ R_face
    """
    idx = _face_index(face)
    if idx is None:
        return R_face, t_face
    yaw, pitch = FACE_YAW_PITCH[idx]
    R_eq_to_face = yaw_pitch_to_rotmat(yaw, pitch)
    R_eq = R_eq_to_face.T @ R_face
    t_eq = R_eq_to_face.T @ t_face
    return R_eq, t_eq


def _find_equirect_image(equirect_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        p = equirect_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _find_mask(mask_dir: Path | None, stem: str) -> Path | None:
    if mask_dir is None or not mask_dir.exists():
        return None
    p = mask_dir / f"{stem}.png"
    return p if p.exists() else None


def _read_points(model_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    txt = model_dir / "points3D.txt"
    if txt.exists():
        pts = read_points3d_txt(txt)
        if pts:
            xyz = np.stack(list(pts.values()), axis=0).astype(np.float32)
            # colors not in read_points3d_txt — gray init
            rgb = np.full((len(xyz), 3), 0.7, dtype=np.float32)
            return xyz, rgb
    # Try GUI loader path for RGB
    from instasplat.gui.geometry import load_colmap_sparse

    cloud = load_colmap_sparse(model_dir, max_points=500_000)
    if cloud is None or cloud.n == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    return cloud.xyz.astype(np.float32), cloud.rgb.astype(np.float32)


def load_equirect_dataset(
    paths: JobPaths,
    model_dir: Path,
    *,
    max_width: int = 1024,
) -> EquirectDataset:
    """
    Build equirect training views.

    Prefers native equirect image names in COLMAP; otherwise lifts cubemap
    ``*_front`` (or any face) poses to panorama poses and pairs with
    ``01_frames/equirect`` images.
    """
    equirect_dir = paths.equirect_frames
    if not equirect_dir.is_dir():
        alt = paths.sfm / "images_equirect"
        if alt.is_dir():
            equirect_dir = alt
    mask_dir = paths.equirect_masks if paths.equirect_masks.is_dir() else None
    if mask_dir is None:
        altm = paths.sfm / "masks_equirect"
        if altm.is_dir():
            mask_dir = altm

    images_txt = model_dir / "images.txt"
    if not images_txt.exists():
        # binary-only: convert hint — try sibling 0_txt
        for cand in (model_dir.parent / "0_txt", model_dir / "0_txt"):
            if (cand / "images.txt").exists():
                images_txt = cand / "images.txt"
                model_dir = cand
                break
    if not images_txt.exists():
        raise FileNotFoundError(
            f"Need images.txt in {model_dir} (run model_converter or use text model)"
        )

    colmap_images = read_images_txt(images_txt)
    views_by_stem: dict[str, TrainView] = {}

    for im in colmap_images:
        name = im["name"]
        R = qvec_to_rotmat(
            np.array([im["qw"], im["qx"], im["qy"], im["qz"]], dtype=np.float64)
        )
        t = np.array([im["tx"], im["ty"], im["tz"]], dtype=np.float64)
        m = _FACE_RE.match(name)
        if m:
            stem = m.group("stem")
            face = m.group("face").lower()
            # Prefer front; otherwise first face wins until front appears
            if stem in views_by_stem and face != "front":
                continue
            if stem in views_by_stem and face == "front":
                pass  # replace
            elif stem in views_by_stem:
                continue
            R_eq, t_eq = _lift_cubemap_pose_to_equirect(R, t, face)
            img_path = _find_equirect_image(equirect_dir, stem)
            if img_path is None:
                continue
            # Probe size (may downscale later)
            probe = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if probe is None:
                continue
            h, w = probe.shape[:2]
            views_by_stem[stem] = TrainView(
                name=stem,
                image_path=img_path,
                mask_path=_find_mask(mask_dir, stem),
                R_w2c=R_eq.astype(np.float32),
                t_w2c=t_eq.astype(np.float32),
                width=w,
                height=h,
            )
        else:
            # Native equirect naming
            stem = Path(name).stem
            img_path = _find_equirect_image(equirect_dir, stem)
            if img_path is None:
                # COLMAP may store image under images_equirect
                for root in (equirect_dir, paths.sfm / "images_equirect", model_dir.parent.parent / "images"):
                    cand = root / name
                    if cand.exists():
                        img_path = cand
                        break
            if img_path is None:
                continue
            probe = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if probe is None:
                continue
            h, w = probe.shape[:2]
            views_by_stem[stem] = TrainView(
                name=stem,
                image_path=img_path,
                mask_path=_find_mask(mask_dir, stem),
                R_w2c=R.astype(np.float32),
                t_w2c=t.astype(np.float32),
                width=w,
                height=h,
            )

    views = sorted(views_by_stem.values(), key=lambda v: v.name)
    if not views:
        raise RuntimeError(
            "No equirect training views could be paired with COLMAP images. "
            "Ensure 01_frames/equirect contains source panoramas and SfM used "
            "cubemap faces named {stem}_front.jpg (or EQUIRECTANGULAR images)."
        )

    # Apply max_width scaling metadata (actual resize at load time)
    for v in views:
        if v.width > max_width:
            scale = max_width / v.width
            v.width = max_width
            v.height = max(1, int(round(v.height * scale)))

    xyz, rgb = _read_points(model_dir)
    return EquirectDataset(views=views, points_xyz=xyz, points_rgb=rgb, equirect_dir=equirect_dir)


def load_view_tensors(
    view: TrainView,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Load RGB [0,1] HWC, optional mask, R, t on device."""
    img = cv2.imread(str(view.image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(view.image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[1] != view.width or img.shape[0] != view.height:
        img = cv2.resize(img, (view.width, view.height), interpolation=cv2.INTER_AREA)
    rgb = torch.from_numpy(img.astype(np.float32) / 255.0).to(device)
    mask_t = None
    if view.mask_path is not None and view.mask_path.exists():
        m = cv2.imread(str(view.mask_path), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            if m.shape[1] != view.width or m.shape[0] != view.height:
                m = cv2.resize(m, (view.width, view.height), interpolation=cv2.INTER_NEAREST)
            mask_t = torch.from_numpy((m.astype(np.float32) / 255.0)).to(device)
    R = torch.from_numpy(view.R_w2c).to(device)
    t = torch.from_numpy(view.t_w2c).to(device)
    return rgb, mask_t, R, t
