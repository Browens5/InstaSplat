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
        return self.sfm / "images"

    @property
    def cubemap_masks(self) -> Path:
        return self.sfm / "masks"

    @property
    def colmap_db(self) -> Path:
        return self.sfm / "database.db"

    @property
    def colmap_sparse(self) -> Path:
        return self.sfm / "sparse"

    @property
    def colmap_model(self) -> Path:
        """Primary sparse model used by Brush (model 0)."""
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
    def brush_export(self) -> Path:
        return self.train / "exports"

    @property
    def export(self) -> Path:
        return self.root / "06_export"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def ensure(self) -> None:
        for p in (
            self.root,
            self.ingest,
            self.equirect_frames,
            self.equirect_masks,
            self.cubemap_images,
            self.cubemap_masks,
            self.colmap_sparse,
            self.scale,
            self.train,
            self.brush_export,
            self.export,
            self.logs,
        ):
            p.mkdir(parents=True, exist_ok=True)
