"""Filesystem layout helpers for a job workspace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JobPaths:
    """Canonical directories under ``output_dir/project_name``."""

    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def ingest(self) -> Path:
        return self.root / "00_ingest"

    @property
    def video(self) -> Path:
        return self.ingest / "equirect.mp4"

    @property
    def gyro_csv(self) -> Path:
        return self.ingest / "gyro.csv"

    @property
    def accel_csv(self) -> Path:
        return self.ingest / "accel.csv"

    @property
    def gps_csv(self) -> Path:
        return self.ingest / "gps.csv"

    @property
    def metadata_json(self) -> Path:
        return self.ingest / "metadata.json"

    @property
    def frames(self) -> Path:
        return self.root / "01_frames"

    @property
    def equirect_frames(self) -> Path:
        return self.frames / "equirect"

    @property
    def masks(self) -> Path:
        return self.root / "02_masks"

    @property
    def equirect_masks(self) -> Path:
        return self.masks / "equirect"

    @property
    def sfm(self) -> Path:
        return self.root / "03_sfm"

    @property
    def cubemap_images(self) -> Path:
        """Legacy perspective cubemap faces (only when sfm.mode=perspective_cubemap)."""
        return self.sfm / "images"

    @property
    def cubemap_masks(self) -> Path:
        return self.sfm / "masks"

    @property
    def equirect_sfm_images(self) -> Path:
        """Staged full equirect panoramas for COLMAP EQUIRECTANGULAR SfM."""
        return self.sfm / "images_equirect"

    @property
    def equirect_sfm_masks(self) -> Path:
        return self.sfm / "masks_equirect"

    def resolve_sfm_image_dir(self, mode: str | None = None) -> Path:
        """Pick the COLMAP image directory for the active / detected SfM mode."""
        eq = self.equirect_sfm_images
        has_eq = eq.is_dir() and any(eq.iterdir())
        if mode in {"equirectangular", "telemetry_fallback"} and has_eq:
            return eq
        if mode == "perspective_cubemap":
            return self.cubemap_images
        if has_eq:
            return eq
        return self.cubemap_images

    def resolve_sfm_mask_dir(self, mode: str | None = None) -> Path | None:
        img = self.resolve_sfm_image_dir(mode)
        if img == self.equirect_sfm_images:
            m = self.equirect_sfm_masks
            return m if m.is_dir() and any(m.glob("*.png")) else None
        m = self.cubemap_masks
        return m if m.is_dir() and any(m.glob("*.png")) else None

    @property
    def colmap_db(self) -> Path:
        return self.sfm / "database.db"

    @property
    def colmap_sparse(self) -> Path:
        return self.sfm / "sparse"

    @property
    def colmap_model(self) -> Path:
        """Primary sparse model used for training (model 0)."""
        return self.colmap_sparse / "0"

    @property
    def scale(self) -> Path:
        return self.root / "04_scale"

    @property
    def scaled_model(self) -> Path:
        return self.scale / "sparse" / "0"

    @property
    def train(self) -> Path:
        return self.root / "05_train"

    @property
    def train_export(self) -> Path:
        """Gaussian export directory (``05_train/exports``)."""
        return self.train / "exports"

    @property
    def brush_export(self) -> Path:
        """Alias for ``train_export`` (legacy name kept for job compatibility)."""
        return self.train_export

    @property
    def export(self) -> Path:
        return self.root / "06_export"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def chunks(self) -> Path:
        return self.root / "10_chunks"

    @property
    def merged(self) -> Path:
        return self.root / "11_merged"

    def ensure(self) -> None:
        for p in (
            self.root,
            self.ingest,
            self.equirect_frames,
            self.equirect_masks,
            self.equirect_sfm_images,
            self.equirect_sfm_masks,
            self.cubemap_images,
            self.cubemap_masks,
            self.colmap_sparse,
            self.scale,
            self.train,
            self.brush_export,
            self.export,
            self.logs,
            self.chunks,
            self.merged,
        ):
            p.mkdir(parents=True, exist_ok=True)
