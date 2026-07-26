"""Assemble equirect training views from COLMAP + equirect frames."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
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
# Trailing frame index: e_000001 / frame_000001 / IMG_12 → 1 / 1 / 12
_TRAILING_INDEX_RE = re.compile(r"(\d+)$")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


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


@dataclass
class _ImageCatalog:
    """Index of on-disk panoramas for fast / fuzzy COLMAP name resolution."""

    files: list[Path] = field(default_factory=list)
    by_basename: dict[str, Path] = field(default_factory=dict)
    by_stem: dict[str, Path] = field(default_factory=dict)
    by_index: dict[int, Path] = field(default_factory=dict)
    ambiguous_indices: set[int] = field(default_factory=set)
    sample_names: list[str] = field(default_factory=list)


def _face_index(name: str) -> int | None:
    try:
        return FACE_NAMES.index(name.lower())
    except ValueError:
        return None


def _strip_face_suffix(stem: str) -> str:
    base = stem
    lower = base.lower()
    for face in FACE_NAMES:
        suf = f"_{face}"
        if lower.endswith(suf):
            return base[: -len(suf)]
    return base


def _trailing_index(stem: str) -> int | None:
    """Extract trailing integer from a frame stem (after dropping cubemap face)."""
    base = _strip_face_suffix(stem)
    m = _TRAILING_INDEX_RE.search(base)
    if not m:
        return None
    try:
        return int(m.group(1))
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
        if p.suffix.lower() not in _IMAGE_EXTS:
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


def _build_image_catalog(dirs: list[Path]) -> _ImageCatalog:
    """
    Index panoramas by basename, stem, and unique trailing frame index.

    Trailing-index match pairs COLMAP ``e_000001.jpg`` with on-disk
    ``frame_000001.jpg`` (and the reverse) when the numeric suffix is unique.
    """
    cat = _ImageCatalog()
    index_hits: dict[int, list[Path]] = {}
    seen_resolved: set[Path] = set()

    for d in dirs:
        for p in _list_image_files(d):
            try:
                key = p.resolve()
            except OSError:
                key = p
            if key in seen_resolved:
                continue
            seen_resolved.add(key)
            cat.files.append(p)
            cat.by_basename.setdefault(p.name.lower(), p)
            cat.by_stem.setdefault(p.stem.lower(), p)
            idx = _trailing_index(p.stem)
            if idx is not None:
                index_hits.setdefault(idx, []).append(p)

    for idx, hits in index_hits.items():
        # Same stem in multiple dirs (frames + images_equirect) is fine — pick one.
        stems = {h.stem.lower() for h in hits}
        if len(stems) == 1:
            cat.by_index[idx] = hits[0]
            continue
        preferred = [h for h in hits if h.stem.lower().startswith("frame")]
        pref_stems = {h.stem.lower() for h in preferred}
        if len(pref_stems) == 1:
            cat.by_index[idx] = preferred[0]
        else:
            # Distinct stems sharing an index (frame_1 vs shot_1) — do not guess
            cat.ambiguous_indices.add(idx)

    cat.sample_names = [p.name for p in cat.files[:8]]
    return cat


def _resolve_training_image(
    catalog: _ImageCatalog,
    colmap_name: str,
    *,
    prefer_equirect_stem: str | None = None,
) -> Path | None:
    """
    Locate the file for a COLMAP IMAGE name.

    Resolution order:
      1. Exact basename / relative path basename
      2. Exact stem (case-insensitive)
      3. Trailing numeric index (``e_000001`` ↔ ``frame_000001``)
    """
    basename = Path(colmap_name).name
    stem = prefer_equirect_stem or Path(basename).stem

    hit = catalog.by_basename.get(basename.lower())
    if hit is not None:
        return hit

    hit = catalog.by_stem.get(stem.lower())
    if hit is not None:
        return hit

    # Cubemap COLMAP name → panorama stem already preferred above; also try
    # index of the equirect stem and of the raw basename stem.
    for candidate_stem in (stem, Path(basename).stem):
        idx = _trailing_index(candidate_stem)
        if idx is None or idx in catalog.ambiguous_indices:
            continue
        hit = catalog.by_index.get(idx)
        if hit is not None:
            return hit
    return None


def _find_mask(mask_dir: Path | None, stem: str, catalog_stem: str | None = None) -> Path | None:
    if mask_dir is None or not mask_dir.exists():
        return None
    for key in (stem, catalog_stem or ""):
        if not key:
            continue
        p = mask_dir / f"{key}.png"
        if p.exists():
            return p
        stem_l = key.lower()
        for cand in mask_dir.glob("*.png"):
            if cand.stem.lower() == stem_l:
                return cand
    # Numeric alias for masks (frame_000001.png ↔ e_000001)
    idx = _trailing_index(stem)
    if idx is not None:
        for cand in mask_dir.glob("*.png"):
            if _trailing_index(cand.stem) == idx:
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


def _pose_is_finite(R: np.ndarray, t: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(R)) and np.all(np.isfinite(t)))


def _load_colmap_images(model_dir: Path) -> tuple[list[dict], Path]:
    """
    Load COLMAP image poses, preferring binary (avoids empty-POINTS2D TXT bugs).

    After loading, drop entries with non-finite / zero-norm quaternions so a
    partial corrupt model does not poison training.
    """
    model_dir = Path(model_dir)
    bin_path = model_dir / "images.bin"
    images: list[dict] = []
    if bin_path.exists():
        images = read_images_bin(bin_path)

    if not images:
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
        if images_txt.exists():
            images = read_images_txt(images_txt)

    # Validate poses; if binary yielded zero usable poses but TXT exists, retry TXT
    usable = _filter_usable_images(images)
    if not usable and bin_path.exists():
        txt = model_dir / "images.txt"
        if txt.exists():
            usable = _filter_usable_images(read_images_txt(txt))
    return usable, model_dir


def _filter_usable_images(images: list[dict]) -> list[dict]:
    out: list[dict] = []
    for im in images:
        q = np.array([im["qw"], im["qx"], im["qy"], im["qz"]], dtype=np.float64)
        t = np.array([im["tx"], im["ty"], im["tz"]], dtype=np.float64)
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(t)):
            continue
        if float(np.linalg.norm(q)) < 1e-12:
            continue
        out.append(im)
    return out


def _pairing_error(
    paths: JobPaths,
    model_dir: Path,
    colmap_images: list[dict],
    catalog: _ImageCatalog,
) -> str:
    dirs = _image_search_dirs(paths, model_dir)
    dir_notes = []
    for d in dirs:
        n = len(_list_image_files(d))
        dir_notes.append(f"  - {d}: {n} image(s)")
    sample_names = [im["name"] for im in colmap_images[:5]]
    disk_sample = catalog.sample_names or ["(none)"]
    return (
        "No equirect training views could be paired with COLMAP images.\n"
        f"COLMAP registered {len(colmap_images)} image(s); "
        f"sample names: {sample_names}\n"
        f"On-disk sample names: {disk_sample}\n"
        "Search dirs:\n" + "\n".join(dir_notes) + "\n"
        "Matching tries: exact basename, case-insensitive stem, then unique "
        "trailing frame index (e.g. e_000001.jpg ↔ frame_000001.jpg).\n"
        "Expected files under 01_frames/equirect or 03_sfm/images_equirect.\n"
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

    Name resolution also matches on unique trailing numeric indices so
    ``e_000001.jpg`` pairs with ``frame_000001.jpg``.
    """
    search_dirs = _image_search_dirs(paths, model_dir)
    catalog = _build_image_catalog(search_dirs)
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
            f"COLMAP model at {model_dir} has 0 usable registered images "
            "(empty or corrupt images.txt / images.bin)."
        )

    views_by_stem: dict[str, TrainView] = {}
    skipped_missing = 0
    skipped_unreadable = 0
    skipped_bad_pose = 0
    paired_by_index = 0

    for im in colmap_images:
        name = im["name"]
        try:
            R = qvec_to_rotmat(
                np.array([im["qw"], im["qx"], im["qy"], im["qz"]], dtype=np.float64)
            )
        except ValueError:
            skipped_bad_pose += 1
            continue
        t = np.array([im["tx"], im["ty"], im["tz"]], dtype=np.float64)
        if not _pose_is_finite(R, t):
            skipped_bad_pose += 1
            continue

        basename = Path(name).name
        m = _FACE_RE.match(basename)
        if m:
            stem = m.group("stem")
            face = m.group("face").lower()
            # Prefer front; otherwise first face wins until front appears
            if stem in views_by_stem and face != "front":
                continue
            R_eq, t_eq = _lift_cubemap_pose_to_equirect(R, t, face)
            if not _pose_is_finite(R_eq, t_eq):
                skipped_bad_pose += 1
                continue
            img_path = _resolve_training_image(
                catalog, name, prefer_equirect_stem=stem
            )
            if img_path is None:
                skipped_missing += 1
                continue
            if _trailing_index(stem) is not None and img_path.stem.lower() != stem.lower():
                paired_by_index += 1
            probe = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if probe is None:
                skipped_unreadable += 1
                continue
            h, w = probe.shape[:2]
            views_by_stem[stem] = TrainView(
                name=stem,
                image_path=img_path,
                mask_path=_find_mask(mask_dir, stem, img_path.stem),
                R_w2c=R_eq.astype(np.float32),
                t_w2c=t_eq.astype(np.float32),
                width=w,
                height=h,
            )
        else:
            # Native equirect naming
            stem = Path(basename).stem
            img_path = _resolve_training_image(catalog, name)
            if img_path is None:
                skipped_missing += 1
                continue
            if img_path.stem.lower() != stem.lower():
                paired_by_index += 1
            probe = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if probe is None:
                skipped_unreadable += 1
                continue
            h, w = probe.shape[:2]
            # Stable view key: prefer on-disk stem so masks/exports align with frames
            view_key = img_path.stem
            views_by_stem[view_key] = TrainView(
                name=view_key,
                image_path=img_path,
                mask_path=_find_mask(mask_dir, stem, img_path.stem),
                R_w2c=R.astype(np.float32),
                t_w2c=t.astype(np.float32),
                width=w,
                height=h,
            )

    views = sorted(views_by_stem.values(), key=lambda v: v.name)
    if not views:
        raise RuntimeError(
            _pairing_error(paths, model_dir, colmap_images, catalog)
            + f"\nSkipped missing={skipped_missing}, unreadable={skipped_unreadable}, "
            f"bad_pose={skipped_bad_pose}."
        )

    # Apply max_width scaling metadata (actual resize at load time)
    for v in views:
        if v.width > max_width:
            scale = max_width / v.width
            v.width = max_width
            v.height = max(1, int(round(v.height * scale)))

    xyz, rgb = _read_points(model_dir)
    # Drop non-finite points (corrupt sparse models)
    if len(xyz):
        ok = np.isfinite(xyz).all(axis=1)
        xyz, rgb = xyz[ok], rgb[ok]

    ds = EquirectDataset(
        views=views, points_xyz=xyz, points_rgb=rgb, equirect_dir=equirect_dir
    )
    # Stash diagnostics lightly via attribute for logs (optional)
    ds.paired_by_index = paired_by_index  # type: ignore[attr-defined]
    ds.skipped_bad_pose = skipped_bad_pose  # type: ignore[attr-defined]
    return ds


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
