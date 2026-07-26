"""Assemble equirect training views from COLMAP + equirect frames."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from instasplat.metal_equirect.cameras import yaw_pitch_to_rotmat
from instasplat.metal_equirect.colmap_bin import ensure_images_txt, read_images_bin
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


def _list_image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        # Skip broken symlinks
        if not p.exists():
            continue
        out.append(p)
    return out


def _image_search_dirs(paths: JobPaths, model_dir: Path) -> list[Path]:
    """Ordered directories that may hold panoramas (or legacy cubemap faces)."""
    candidates = [
        paths.equirect_frames,
        paths.equirect_sfm_images,
        paths.sfm / "images_equirect",
        paths.cubemap_images,
        model_dir.parent.parent / "images_equirect",
        model_dir.parent.parent / "images",
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for d in candidates:
        try:
            key = d.resolve()
        except OSError:
            key = d
        if key in seen:
            continue
        if d.is_dir():
            seen.add(key)
            out.append(d)
    return out


def _find_equirect_image(equirect_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = equirect_dir / f"{stem}{ext}"
        if p.exists():
            return p
    # Case-insensitive stem match (APFS usually OK; Linux CI is strict)
    stem_l = stem.lower()
    for p in _list_image_files(equirect_dir):
        if p.stem.lower() == stem_l:
            return p
    return None


def _resolve_training_image(
    paths: JobPaths,
    model_dir: Path,
    colmap_name: str,
    *,
    prefer_equirect_stem: str | None = None,
) -> Path | None:
    """
    Locate the file for a COLMAP IMAGE name.

    Handles basename-only names, directory prefixes, symlinks into
    ``images_equirect``, and case-insensitive stems.
    """
    basename = Path(colmap_name).name
    stem = prefer_equirect_stem or Path(basename).stem
    dirs = _image_search_dirs(paths, model_dir)

    # Exact basename / full relative name in each search dir
    for d in dirs:
        for cand in (d / basename, d / colmap_name):
            try:
                if cand.is_file() and cand.exists():
                    return cand
            except OSError:
                continue

    # Stem match (equirect frame_000001.jpg ↔ COLMAP frame_000001.jpg)
    for d in dirs:
        found = _find_equirect_image(d, stem)
        if found is not None:
            return found

    return None


def _find_mask(mask_dir: Path | None, stem: str) -> Path | None:
    if mask_dir is None or not mask_dir.exists():
        return None
    p = mask_dir / f"{stem}.png"
    if p.exists():
        return p
    stem_l = stem.lower()
    for cand in mask_dir.glob("*.png"):
        if cand.stem.lower() == stem_l:
            return cand
    return None


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


def _load_colmap_images(model_dir: Path) -> tuple[list[dict], Path]:
    """
    Load COLMAP image poses, preferring binary (avoids empty-POINTS2D TXT bugs).
    """
    model_dir = Path(model_dir)
    bin_path = model_dir / "images.bin"
    if bin_path.exists():
        images = read_images_bin(bin_path)
        if images:
            return images, model_dir

    images_txt = model_dir / "images.txt"
    if not images_txt.exists():
        converted = ensure_images_txt(model_dir)
        if converted is not None:
            images_txt = converted
        else:
            for cand in (model_dir.parent / "0_txt", model_dir / "0_txt"):
                if (cand / "images.txt").exists():
                    images_txt = cand / "images.txt"
                    model_dir = cand
                    break
    if not images_txt.exists():
        raise FileNotFoundError(
            f"Need images.txt or images.bin in {model_dir} "
            "(run model_converter or use a text/binary COLMAP model)"
        )
    images = read_images_txt(images_txt)
    if not images and bin_path.exists():
        # TXT present but unreadable — last chance binary
        images = read_images_bin(bin_path)
    return images, model_dir


def _pairing_error(
    paths: JobPaths,
    model_dir: Path,
    colmap_images: list[dict],
) -> str:
    dirs = _image_search_dirs(paths, model_dir)
    dir_notes = []
    for d in dirs:
        n = len(_list_image_files(d))
        dir_notes.append(f"  - {d}: {n} image(s)")
    sample_names = [im["name"] for im in colmap_images[:5]]
    return (
        "No equirect training views could be paired with COLMAP images.\n"
        f"COLMAP registered {len(colmap_images)} image(s); "
        f"sample names: {sample_names}\n"
        "Search dirs:\n" + "\n".join(dir_notes) + "\n"
        "Expected matching files under 01_frames/equirect or "
        "03_sfm/images_equirect (same basename as COLMAP NAME).\n"
        "If SfM used cubemap faces, names look like frame_000001_front.jpg "
        "and the panorama must be frame_000001.jpg."
    )


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
    ``01_frames/equirect`` / ``03_sfm/images_equirect`` images.
    """
    search_dirs = _image_search_dirs(paths, model_dir)
    equirect_dir = paths.equirect_frames
    for d in search_dirs:
        if _list_image_files(d):
            equirect_dir = d
            break

    mask_dir = paths.equirect_masks if paths.equirect_masks.is_dir() else None
    if mask_dir is None or not any(mask_dir.glob("*.png")):
        altm = paths.equirect_sfm_masks
        if altm.is_dir() and any(altm.glob("*.png")):
            mask_dir = altm

    colmap_images, model_dir = _load_colmap_images(model_dir)
    if not colmap_images:
        raise RuntimeError(
            f"COLMAP model at {model_dir} has 0 registered images "
            "(empty images.txt / images.bin)."
        )

    views_by_stem: dict[str, TrainView] = {}
    skipped_missing = 0
    skipped_unreadable = 0

    for im in colmap_images:
        name = im["name"]
        R = qvec_to_rotmat(
            np.array([im["qw"], im["qx"], im["qy"], im["qz"]], dtype=np.float64)
        )
        t = np.array([im["tx"], im["ty"], im["tz"]], dtype=np.float64)
        basename = Path(name).name
        m = _FACE_RE.match(basename)
        if m:
            stem = m.group("stem")
            face = m.group("face").lower()
            # Prefer front; otherwise first face wins until front appears
            if stem in views_by_stem and face != "front":
                continue
            R_eq, t_eq = _lift_cubemap_pose_to_equirect(R, t, face)
            img_path = _resolve_training_image(
                paths, model_dir, name, prefer_equirect_stem=stem
            )
            if img_path is None:
                skipped_missing += 1
                continue
            probe = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if probe is None:
                skipped_unreadable += 1
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
            stem = Path(basename).stem
            img_path = _resolve_training_image(paths, model_dir, name)
            if img_path is None:
                skipped_missing += 1
                continue
            probe = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if probe is None:
                skipped_unreadable += 1
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
            _pairing_error(paths, model_dir, colmap_images)
            + f"\nSkipped missing={skipped_missing}, unreadable={skipped_unreadable}."
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
