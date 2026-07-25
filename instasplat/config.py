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
    # LongSplat-style prune of low-opacity Gaussians before/after merge (0–1)
    merge_prune_opacity: float = 0.05
    # Require this fraction of temporal overlap vs duration (warn if lower)
    min_overlap_ratio: float = 0.15


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
    device: str = "mps"  # Apple Silicon; per-frame CPU fallback on MPS crashes
    dilate_px: int = 4
    # Brush expects masks where white = keep, black = ignore (or alpha).
    invert: bool = False
    # retina_masks=True triggers intermittent PyTorch MPS indexing crashes in YOLO-seg
    retina_masks: bool = False
    # After this many MPS AcceleratorErrors, switch remaining frames to CPU
    mps_fail_limit: int = 3


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
    # If COLMAP fails, synthesize poses from gyro/GPS (LongSplat / on-the-fly style)
    telemetry_fallback: bool = True


@dataclass
class RefineConfig:
    """Self-Cali-inspired pose / intrinsic refine before training."""

    enabled: bool = True
    run_colmap_ba: bool = True
    refine_intrinsics: bool = True
    refine_distortion: bool = False  # cubemap pinhole usually has no distortion
    # Blend factor toward GPS/gyro priors (0=COLMAP only, 1=telemetry only)
    pose_blend: float = 0.25
    # Reject tile alignments with RMSE above this (meters) when GPS exists
    max_align_rmse_m: float = 8.0


@dataclass
class PackageConfig:
    """Downstream packaging for cloud trainers / LOD."""

    nerfstudio: bool = True
    copy_images: bool = False
    hierarchy_manifest: bool = True
    # CPU XYZ+opacity LOD previews from tile anchors (no CUDA Kerbl merger)
    cpu_lod: bool = True
    # Portable cloud_job.json for 3DGUT / LichtFeld / splatfacto workers
    cloud_manifest: bool = True
    # Always write quality.json after package
    quality_report: bool = True


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


TrainerBackend = Literal["brush", "opensplat"]


@dataclass
class TrainConfig:
    """Gaussian splat training settings (Metal-capable backends)."""

    # brush = ArthurBrussee/brush (WebGPU/Metal); opensplat = pierotofy/OpenSplat (MPS)
    backend: TrainerBackend = "brush"
    total_steps: int = 30_000
    max_resolution: int = 1600
    with_viewer: bool = False
    export_every: int = 5_000
    brush_bin: str = "brush"
    opensplat_bin: str = "opensplat"
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
    # hierarchical / streaming delivery (splat-transform lod-meta.json)
    streamed_lod: bool = False
    # Niantic SPZ uses RUB coords by default — document in export notes
    spz_coordinate_note: bool = True


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
    refine: RefineConfig = field(default_factory=RefineConfig)
    scale: ScaleConfig = field(default_factory=ScaleConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    package: PackageConfig = field(default_factory=PackageConfig)
    # Stages to run; empty = all
    stages: list[str] = field(
        default_factory=lambda: [
            "ingest",
            "extract",
            "mask",
            "sfm",
            "scale",
            "refine",
            "train",
            "export",
            "package",
        ]
    )
    dry_run: bool = False
    skip_existing: bool = True
    # Mac long-360 safety rails
    preflight: bool = True
    allow_unstitched: bool = False  # dual-fisheye remux is opt-in only
    allow_partial_merge: bool = False  # require all successful tiles by default

    def work_dir(self) -> Path:
        return self.output_dir / self.project_name

    def enable_large_8k_defaults(self) -> None:
        """Alias for Mac long-360 tiled defaults."""
        self.enable_mac_long_360_defaults()

    def enable_mac_long_360_defaults(self) -> None:
        """
        Best local-Mac settings for long 360 video → tiled Gaussian splat.

        Metal-first: YOLO MPS → COLMAP CPU cubemap → Brush/OpenSplat Metal →
        GPS/gyro tile align → splat-transform merge. No CUDA / LingBot-Map.
        """
        self.mode = "tiled"
        self.chunk.enabled = True
        self.chunk.base_fps = 6.0
        self.chunk.max_fps = 15.0
        self.chunk.source_fps_hint = 30.0
        self.chunk.duration_sec = 25.0
        self.chunk.overlap_sec = 5.0
        self.chunk.max_frames_per_chunk = 180
        self.chunk.min_overlap_ratio = 0.15
        self.chunk.merge_prune_opacity = 0.05
        self.metal.prefer_metal = True
        self.metal.serialize_brush = True
        self.mask.device = "mps"
        self.sfm.mode = "perspective_cubemap"
        self.sfm.face_resolution = 1280
        self.sfm.quality = "high"
        self.sfm.matcher = "sequential"
        self.sfm.sequential_overlap = 18
        self.sfm.telemetry_fallback = True
        self.train.max_resolution = 1600
        self.train.total_steps = 20_000
        self.train.backend = "brush"
        self.train.with_viewer = False
        self.export.formats = ["ply", "sog", "spz"]
        self.export.min_opacity = 0.05
        self.export.streamed_lod = True
        self.refine.enabled = True
        self.refine.pose_blend = 0.25
        self.refine.max_align_rmse_m = 8.0
        self.package.nerfstudio = True
        self.package.hierarchy_manifest = True
        self.package.cpu_lod = True
        self.package.cloud_manifest = True
        self.package.quality_report = True
        self.preflight = True
        self.allow_unstitched = False
        self.allow_partial_merge = False
        # Prefer GPS when available; preflight soft-falls back to none
        if self.scale.mode == "none":
            self.scale.mode = "gps"
        self.stages = [
            "ingest",
            "plan_chunks",
            "preflight",
            "process_chunks",
            "align_chunks",
            "merge_chunks",
            "package",
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
            refine=_section("refine", RefineConfig),
            scale=_section("scale", ScaleConfig),
            train=_section("train", TrainConfig),
            export=_section("export", ExportConfig),
            package=_section("package", PackageConfig),
            stages=list(data.get("stages") or []),
            dry_run=bool(data.get("dry_run", False)),
            skip_existing=bool(data.get("skip_existing", True)),
            preflight=bool(data.get("preflight", True)),
            allow_unstitched=bool(data.get("allow_unstitched", False)),
            allow_partial_merge=bool(data.get("allow_partial_merge", False)),
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
  backend: brush            # brush | opensplat (Metal MPS)
  total_steps: 20000
  max_resolution: 1600
  with_viewer: false
  brush_bin: brush
  opensplat_bin: opensplat

export:
  formats: [ply, sog, spz]
  filter_nan: true
  min_opacity: 0.05
  streamed_lod: false
  splat_transform_bin: splat-transform

refine:
  enabled: true
  pose_blend: 0.25

package:
  nerfstudio: true
  hierarchy_manifest: true
  cpu_lod: true
  cloud_manifest: true
  quality_report: true

preflight: true
allow_unstitched: false
allow_partial_merge: false

stages: [ingest, plan_chunks, preflight, process_chunks, align_chunks, merge_chunks, package]
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
  merge_prune_opacity: 0.05

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
  backend: brush            # or opensplat for C++ Metal MPS
  total_steps: 20000
  max_resolution: 1600
  brush_bin: brush
  opensplat_bin: opensplat

export:
  formats: [ply, sog, spz]
  filter_nan: true
  min_opacity: 0.05
  streamed_lod: true

package:
  nerfstudio: true
  hierarchy_manifest: true
  cpu_lod: true
  cloud_manifest: true
  quality_report: true

preflight: true
allow_unstitched: false
allow_partial_merge: false

stages: [ingest, plan_chunks, preflight, process_chunks, align_chunks, merge_chunks, package]
"""
