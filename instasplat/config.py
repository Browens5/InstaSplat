"""Pipeline configuration models and defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import yaml

ExportFormat = Literal[
    "ply",
    "sog",
    "compressed.ply",
    "spz",
    "glb",
    "html",
    "csv",
]

SfMMode = Literal["perspective_cubemap", "equirectangular", "auto"]
ScaleMode = Literal["none", "known_distance", "gps", "stereo_baseline"]
StitchMode = Literal["studio_mp4", "mediasdk", "ffmpeg_fallback", "prestitched"]
PipelineMode = Literal["single", "tiled"]


@dataclass
class ExtractConfig:
    """Frame + gyro extraction settings."""

    fps: float = 2.0
    start_sec: float | None = None
    end_sec: float | None = None
    max_frames: int | None = None
    image_format: Literal["jpg", "png"] = "jpg"
    jpeg_quality: int = 95
    stitch_mode: StitchMode = "studio_mp4"
    # When stitching is unavailable, accept a pre-exported equirectangular MP4.
    allow_prestitched_mp4: bool = True


@dataclass
class ChunkConfig:
    """
    Auto-chunk / tile settings for large 8K@30fps captures.

    Instead of dumping every frame, densify sampling on turns (gyro) and split
    the timeline into overlapping tiles that are reconstructed independently,
    then aligned with GPS/gyro and merged.
    """

    enabled: bool = False
    duration_sec: float = 25.0
    overlap_sec: float = 5.0
    base_fps: float = 6.0
    max_fps: float = 15.0
    source_fps_hint: float = 30.0
    max_frames_per_chunk: int = 180
    # Soft target path length per chunk when GPS exists (meters)
    target_path_length_m: float | None = 40.0
    max_parallel_chunks: int = 1


@dataclass
class MetalConfig:
    """Apple Metal / MPS preferences for large local jobs."""

    prefer_metal: bool = True
    # Keep Brush train serial to avoid Metal memory pressure across tiles
    serialize_brush: bool = True
    # YOLO device override; empty = auto-detect MPS
    torch_device: str = ""


@dataclass
class MaskConfig:
    """YOLO people-masking settings."""

    enabled: bool = True
    model: str = "yolov8m-seg.pt"
    conf: float = 0.35
    iou: float = 0.5
    classes: list[int] = field(default_factory=lambda: [0])  # COCO person
    device: str = "mps"  # Apple Silicon; falls back to cpu
    dilate_px: int = 4
    # Brush expects masks where white = keep, black = ignore (or alpha).
    invert: bool = False


@dataclass
class SfMConfig:
    """COLMAP / spherical SfM settings."""

    mode: SfMMode = "perspective_cubemap"
    camera_model: str = "SIMPLE_PINHOLE"
    cubemap_faces: int = 6
    face_fov_deg: float = 90.0
    face_resolution: int = 1024
    matcher: Literal["exhaustive", "sequential", "vocab_tree"] = "sequential"
    sequential_overlap: int = 15
    quality: Literal["low", "medium", "high", "extreme"] = "high"
    use_gpu: bool = False  # COLMAP GPU often unavailable on Mac; CPU is fine


@dataclass
class ScaleConfig:
    """Metric scale recovery for true-to-scale splats."""

    mode: ScaleMode = "none"
    # known_distance: user measures a distance between two reconstructed points
    known_distance_m: float | None = None
    point_a_name: str | None = None
    point_b_name: str | None = None
    # stereo_baseline: Insta360 dual-lens separation (approximate, model-dependent)
    stereo_baseline_m: float = 0.065
    # gps: use GPS track if present in INSV trailer
    gps_csv: Path | None = None


@dataclass
class TrainConfig:
    """Brush Gaussian splat training settings."""

    total_steps: int = 30_000
    max_resolution: int = 1600
    with_viewer: bool = False
    export_every: int = 5_000
    # Brush binary name or absolute path
    brush_bin: str = "brush"
    extra_args: list[str] = field(default_factory=list)


@dataclass
class ExportConfig:
    """splat-transform export options."""

    formats: list[ExportFormat] = field(default_factory=lambda: ["ply", "sog"])
    filter_nan: bool = True
    translate: tuple[float, float, float] | None = None
    rotate_deg: tuple[float, float, float] | None = None
    scale: float | None = None
    filter_bands: int | None = None
    # Optional opacity filter: keep opacity > threshold
    min_opacity: float | None = None
    splat_transform_bin: str = "splat-transform"


@dataclass
class PipelineConfig:
    """Top-level InstaSplat job configuration."""

    input_path: Path
    output_dir: Path
    project_name: str = "instasplat_job"
    mode: PipelineMode = "single"
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    metal: MetalConfig = field(default_factory=MetalConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    sfm: SfMConfig = field(default_factory=SfMConfig)
    scale: ScaleConfig = field(default_factory=ScaleConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    # Stages to run; empty = all
    stages: list[str] = field(
        default_factory=lambda: [
            "ingest",
            "extract",
            "mask",
            "sfm",
            "scale",
            "train",
            "export",
        ]
    )
    dry_run: bool = False
    skip_existing: bool = True

    def work_dir(self) -> Path:
        return self.output_dir / self.project_name

    def enable_large_8k_defaults(self) -> None:
        """Apply recommended settings for 8K@30 large tiled splats on Metal."""
        self.mode = "tiled"
        self.chunk.enabled = True
        self.chunk.base_fps = 6.0
        self.chunk.max_fps = 15.0
        self.chunk.source_fps_hint = 30.0
        self.chunk.duration_sec = 25.0
        self.chunk.overlap_sec = 5.0
        self.chunk.max_frames_per_chunk = 180
        self.metal.prefer_metal = True
        self.metal.serialize_brush = True
        # Device resolved at runtime via detect_metal()
        self.mask.device = "mps"
        self.sfm.face_resolution = 1280
        self.sfm.quality = "high"
        self.train.max_resolution = 1600
        self.train.total_steps = 20_000
        self.export.formats = ["ply", "sog"]
        if self.scale.mode == "none":
            self.scale.mode = "gps"
        self.stages = [
            "ingest",
            "plan_chunks",
            "process_chunks",
            "align_chunks",
            "merge_chunks",
        ]

    def to_dict(self) -> dict[str, Any]:
        def _convert(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, tuple):
                return list(obj)
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _convert(v) for k, v in asdict(obj).items()}
            if isinstance(obj, list):
                return [_convert(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            return obj

        return _convert(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        def _section(name: str, typ: type) -> Any:
            raw = data.get(name, {}) or {}
            allowed = {f.name for f in fields(typ)}
            cleaned = {k: v for k, v in raw.items() if k in allowed}
            # Path fields
            for key, val in list(cleaned.items()):
                f = next(f for f in fields(typ) if f.name == key)
                if val is not None and "Path" in str(f.type):
                    cleaned[key] = Path(val)
                if key in ("translate", "rotate_deg") and isinstance(val, list):
                    cleaned[key] = tuple(val)
            return typ(**cleaned)

        return cls(
            input_path=Path(data["input_path"]),
            output_dir=Path(data["output_dir"]),
            project_name=data.get("project_name", "instasplat_job"),
            mode=data.get("mode", "single"),
            extract=_section("extract", ExtractConfig),
            chunk=_section("chunk", ChunkConfig),
            metal=_section("metal", MetalConfig),
            mask=_section("mask", MaskConfig),
            sfm=_section("sfm", SfMConfig),
            scale=_section("scale", ScaleConfig),
            train=_section("train", TrainConfig),
            export=_section("export", ExportConfig),
            stages=list(data.get("stages") or []),
            dry_run=bool(data.get("dry_run", False)),
            skip_existing=bool(data.get("skip_existing", True)),
        )

    @classmethod
    def load(cls, path: Path) -> PipelineConfig:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


DEFAULT_CONFIG_TEMPLATE = """\
# InstaSplat pipeline config
input_path: /path/to/VID_....insv
output_dir: ./runs
project_name: my_capture
mode: tiled                 # single | tiled  (tiled = auto-chunk large 8K jobs)

extract:
  fps: 2.0                  # used in single mode
  stitch_mode: studio_mp4   # studio_mp4 | mediasdk | ffmpeg_fallback | prestitched
  image_format: jpg

chunk:
  enabled: true
  duration_sec: 25.0
  overlap_sec: 5.0
  base_fps: 6.0             # densify above this on turns
  max_fps: 15.0             # still far below 30 source fps, but uses more data
  source_fps_hint: 30.0
  max_frames_per_chunk: 180
  target_path_length_m: 40.0
  max_parallel_chunks: 1

metal:
  prefer_metal: true
  serialize_brush: true
  torch_device: ""          # empty = auto MPS

mask:
  enabled: true
  model: yolov8m-seg.pt
  conf: 0.35
  device: mps               # mps on Apple Silicon, cpu otherwise

sfm:
  mode: perspective_cubemap # perspective_cubemap | equirectangular | auto
  face_resolution: 1280
  matcher: sequential
  quality: high

scale:
  mode: gps                 # none | known_distance | gps | stereo_baseline
  # known_distance_m: 2.0
  stereo_baseline_m: 0.065

train:
  total_steps: 20000
  max_resolution: 1600
  with_viewer: false
  brush_bin: brush

export:
  formats: [ply, sog]
  filter_nan: true
  splat_transform_bin: splat-transform

stages: [ingest, plan_chunks, process_chunks, align_chunks, merge_chunks]
skip_existing: true
"""


LARGE_8K_CONFIG_TEMPLATE = """\
# Large 8K@30fps tiled Gaussian splat job (Metal-first)
input_path: /path/to/capture_equirect_8k.mp4
output_dir: ./runs
project_name: walk_8k
mode: tiled

chunk:
  enabled: true
  duration_sec: 25.0
  overlap_sec: 5.0
  base_fps: 6.0
  max_fps: 15.0
  source_fps_hint: 30.0
  max_frames_per_chunk: 180
  target_path_length_m: 40.0
  max_parallel_chunks: 1

metal:
  prefer_metal: true
  serialize_brush: true

mask:
  enabled: true
  model: yolov8m-seg.pt
  device: mps

sfm:
  mode: perspective_cubemap
  face_resolution: 1280
  matcher: sequential
  quality: high

scale:
  mode: gps

train:
  total_steps: 20000
  max_resolution: 1600
  brush_bin: brush

export:
  formats: [ply, sog]
  filter_nan: true

stages: [ingest, plan_chunks, process_chunks, align_chunks, merge_chunks]
"""
