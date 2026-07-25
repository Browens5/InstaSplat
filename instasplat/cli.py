"""Typer CLI for InstaSplat."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from instasplat import __version__
from instasplat.config import DEFAULT_CONFIG_TEMPLATE, LARGE_8K_CONFIG_TEMPLATE, PipelineConfig
from instasplat.pipeline import Pipeline
from instasplat.utils.deps import print_report
from instasplat.utils.metal import metal_report
from instasplat.utils.stages import (
    ALL_STAGES,
    SINGLE_STAGES,
    STAGE_HELP,
    TILED_STAGES,
    select_stages,
)

app = typer.Typer(
    name="instasplat",
    help="Insta360 INSV → metric 3D Gaussian splat pipeline for macOS",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()

DEFAULT_OUT = Path("./runs")
DEFAULT_CONFIG_PATH = Path("instasplat.yaml")


@app.callback()
def main() -> None:
    """InstaSplat command group."""


@app.command("version")
def version() -> None:
    """Print version."""
    console.print(__version__)


@app.command("doctor")
def doctor(
    brush_bin: str = typer.Option("brush", help="Brush binary name/path"),
    splat_transform_bin: str = typer.Option(
        "splat-transform", help="splat-transform binary name/path"
    ),
) -> None:
    """Check local dependencies and which stages can run on this Mac."""
    from instasplat.utils import deps as deps_mod

    data = deps_mod.report_dict(brush_bin, splat_transform_bin)
    print_report(console)
    metal = metal_report()
    console.print("\n[bold]Metal / Apple GPU[/bold]")
    console.print(metal)
    ready = data["ready_stages"]
    if not ready.get("official_stitch"):
        console.print(
            "\n[yellow]Note:[/yellow] Official Insta360 MediaSDK is not available on macOS. "
            "Export a stitched equirectangular MP4 from Insta360 Studio, or use a Linux "
            "cloud/Docker worker for MediaSDK stitching."
        )
    ready = data["ready_stages"].get("mac_long_360")
    console.print(
        f"\n[cyan]Mac long-360 ready:[/cyan] {'yes' if ready else 'no — install missing tools above'}"
    )
    if not data["ready_stages"].get("train"):
        console.print(
            "\n[yellow]Brush missing:[/yellow] run `instasplat install-brush` "
            "(auto-clones + cargo build --release)."
        )
    console.print(
        "[cyan]Tip:[/cyan] `instasplat mac-360 -i ./capture_equirect_8k.mp4 -o ./runs -n walk` "
        "for the best local tiled Metal pipeline."
    )


@app.command("init-config")
def init_config(
    out: Path = typer.Option(DEFAULT_CONFIG_PATH, "--out", "-o"),
    large_8k: bool = typer.Option(False, "--large-8k", help="Write tiled 8K@30 defaults"),
) -> None:
    """Write a starter pipeline YAML config."""
    if out.exists():
        raise typer.BadParameter(f"{out} already exists")
    text = LARGE_8K_CONFIG_TEMPLATE if large_8k else DEFAULT_CONFIG_TEMPLATE
    out.write_text(text, encoding="utf-8")
    console.print(f"Wrote {out}")


@app.command("stages")
def stages_cmd(
    mode: str = typer.Option(
        "all",
        "--mode",
        help="Which catalog: all | single | tiled",
    ),
) -> None:
    """List pipeline sections you can run individually."""
    if mode not in {"all", "single", "tiled"}:
        raise typer.BadParameter("mode must be all, single, or tiled")
    names = {"all": ALL_STAGES, "single": SINGLE_STAGES, "tiled": TILED_STAGES}[mode]
    table = Table(title=f"InstaSplat stages ({mode})")
    table.add_column("Stage")
    table.add_column("Description")
    for name in names:
        table.add_row(name, STAGE_HELP.get(name, ""))
    console.print(table)
    console.print(
        "\nExamples:\n"
        "  instasplat stage mask -j ./runs/walk_360\n"
        "  instasplat run --job ./runs/walk_360 --only process_chunks\n"
        "  instasplat run -i ./cap.mp4 --from sfm --to export\n"
        "  instasplat run --large-8k -i ./cap.mp4 --only plan_chunks,preflight"
    )


@app.command("stage")
def stage_cmd(
    stage_name: str = typer.Argument(..., help="Stage to run (see `instasplat stages`)"),
    job: Path | None = typer.Option(
        None, "--job", "-j", help="Existing job directory (loads config.yaml)"
    ),
    input_path: Path | None = typer.Option(None, "--input", "-i"),
    output_dir: Path = typer.Option(DEFAULT_OUT, "--output", "-o"),
    project_name: str = typer.Option("instasplat_job", "--name", "-n"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    large_8k: bool = typer.Option(False, "--large-8k", "--tiled"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run a single pipeline section (on a new or existing job)."""
    if stage_name not in ALL_STAGES:
        raise typer.BadParameter(
            f"Unknown stage '{stage_name}'. Run `instasplat stages` for the list."
        )
    console.print(f"[bold cyan]Running stage[/bold cyan] {stage_name}")
    cfg = _build_run_config(
        input_path=input_path,
        output_dir=output_dir,
        project_name=project_name,
        config=config,
        job=job,
        stages=None,
        only=stage_name,
        from_stage=None,
        to_stage=None,
        fps=None,
        no_mask=False,
        export_formats=None,
        dry_run=dry_run,
        with_viewer=False,
        large_8k=large_8k,
        trainer=None,
        no_refine=False,
        streamed_lod=False,
        no_cloud_manifest=False,
        no_quality=False,
        no_preflight=False,
        allow_unstitched=False,
        allow_partial_merge=False,
    )
    _execute_pipeline(cfg)


@app.command("run")
def run(
    input_path: Path | None = typer.Option(
        None, "--input", "-i", help="INSV or prestitched equirect MP4"
    ),
    output_dir: Path = typer.Option(DEFAULT_OUT, "--output", "-o"),
    project_name: str = typer.Option("instasplat_job", "--name", "-n"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    job: Path | None = typer.Option(
        None,
        "--job",
        "-j",
        help="Existing job dir — resume / run selected stages using its config.yaml",
    ),
    stages: str | None = typer.Option(
        None,
        "--stages",
        help="Comma-separated stages (alias of --only)",
    ),
    only: str | None = typer.Option(
        None,
        "--only",
        help="Run only these stages, e.g. mask or plan_chunks,process_chunks",
    ),
    from_stage: str | None = typer.Option(
        None, "--from", help="Start at this stage (inclusive)"
    ),
    to_stage: str | None = typer.Option(
        None, "--to", help="Stop after this stage (inclusive)"
    ),
    fps: float | None = typer.Option(None, help="Override extract FPS (single mode)"),
    no_mask: bool = typer.Option(False, help="Disable YOLO people masking"),
    export_formats: str | None = typer.Option(
        None, "--formats", help="Comma-separated: ply,sog,spz,glb,html,csv,compressed.ply"
    ),
    dry_run: bool = typer.Option(False, help="Print/plan without executing heavy tools"),
    with_viewer: bool = typer.Option(False, help="Open Brush viewer while training"),
    large_8k: bool = typer.Option(
        False,
        "--large-8k",
        help="Enable Mac long-360 tiled mode (alias of mac-360 defaults)",
    ),
    tiled: bool = typer.Option(False, "--tiled", help="Alias for enabling chunk.mode=tiled"),
    trainer: str | None = typer.Option(
        None,
        "--trainer",
        help="Training backend: brush (default) or opensplat (Metal MPS)",
    ),
    no_refine: bool = typer.Option(False, "--no-refine", help="Disable pose refine stage"),
    streamed_lod: bool = typer.Option(
        False, "--streamed-lod", help="Also export lod-meta.json streamed SOG"
    ),
    no_cloud_manifest: bool = typer.Option(
        False, "--no-cloud-manifest", help="Skip writing cloud_job.json"
    ),
    no_quality: bool = typer.Option(False, "--no-quality", help="Skip quality.json report"),
    no_preflight: bool = typer.Option(False, "--no-preflight", help="Skip Mac long-360 preflight"),
    allow_unstitched: bool = typer.Option(
        False, "--allow-unstitched", help="Allow dual-fisheye remux (testing only)"
    ),
    allow_partial_merge: bool = typer.Option(
        False, "--allow-partial-merge", help="Merge even if some tiles failed"
    ),
) -> None:
    """Run the full pipeline, or selected sections with --only / --from / --to."""
    cfg = _build_run_config(
        input_path=input_path,
        output_dir=output_dir,
        project_name=project_name,
        config=config,
        job=job,
        stages=stages,
        only=only,
        from_stage=from_stage,
        to_stage=to_stage,
        fps=fps,
        no_mask=no_mask,
        export_formats=export_formats,
        dry_run=dry_run,
        with_viewer=with_viewer,
        large_8k=large_8k or tiled,
        trainer=trainer,
        no_refine=no_refine,
        streamed_lod=streamed_lod,
        no_cloud_manifest=no_cloud_manifest,
        no_quality=no_quality,
        no_preflight=no_preflight,
        allow_unstitched=allow_unstitched,
        allow_partial_merge=allow_partial_merge,
    )
    _execute_pipeline(cfg)


@app.command("mac-360")
def mac_360(
    input_path: Path = typer.Option(
        ..., "--input", "-i", help="Studio equirect MP4 (preferred) or INSV+sibling MP4"
    ),
    output_dir: Path = typer.Option(DEFAULT_OUT, "--output", "-o"),
    project_name: str = typer.Option("walk_360", "--name", "-n"),
    trainer: str = typer.Option("brush", "--trainer", help="brush | opensplat"),
    no_mask: bool = typer.Option(False, help="Disable YOLO people masking"),
    dry_run: bool = typer.Option(False, help="Plan chunks + preflight only"),
    allow_unstitched: bool = typer.Option(False, "--allow-unstitched"),
    allow_partial_merge: bool = typer.Option(False, "--allow-partial-merge"),
    formats: str = typer.Option("ply,sog,spz", "--formats"),
) -> None:
    """
    Best local Mac pipeline: long 360 video → tiled Metal Gaussian splat.

    Expects a stitched equirectangular MP4 from Insta360 Studio. Keep the
    original .insv beside it (or gyro.csv/gps.csv sidecars) for turn densify
    and metric GPS scale.
    """
    console.print(
        "[bold cyan]InstaSplat mac-360[/bold cyan] — tiled Metal pipeline "
        "(YOLO MPS → COLMAP → Brush/OpenSplat → GPS/gyro merge)"
    )
    cfg = _build_run_config(
        input_path=input_path,
        output_dir=output_dir,
        project_name=project_name,
        config=None,
        job=None,
        stages=None,
        only=None,
        from_stage=None,
        to_stage=None,
        fps=None,
        no_mask=no_mask,
        export_formats=formats,
        dry_run=dry_run,
        with_viewer=False,
        large_8k=True,
        trainer=trainer,
        no_refine=False,
        streamed_lod=True,
        no_cloud_manifest=False,
        no_quality=False,
        no_preflight=False,
        allow_unstitched=allow_unstitched,
        allow_partial_merge=allow_partial_merge,
    )
    _execute_pipeline(cfg)


def _load_job_config(job: Path) -> PipelineConfig:
    """Load an existing job's config.yaml (or synthesize a minimal one)."""
    job = job.resolve()
    cfg_path = job / "config.yaml"
    if cfg_path.exists():
        cfg = PipelineConfig.load(cfg_path)
        cfg.output_dir = job.parent
        cfg.project_name = job.name
        return cfg
    # Fallback: point at job video if present
    video = job / "00_ingest" / "equirect.mp4"
    inp = video if video.exists() else job
    return PipelineConfig(input_path=inp, output_dir=job.parent, project_name=job.name)


def _build_run_config(
    *,
    input_path: Path | None,
    output_dir: Path,
    project_name: str,
    config: Path | None,
    job: Path | None,
    stages: str | None,
    only: str | None,
    from_stage: str | None,
    to_stage: str | None,
    fps: float | None,
    no_mask: bool,
    export_formats: str | None,
    dry_run: bool,
    with_viewer: bool,
    large_8k: bool,
    trainer: str | None,
    no_refine: bool,
    streamed_lod: bool,
    no_cloud_manifest: bool,
    no_quality: bool,
    no_preflight: bool,
    allow_unstitched: bool,
    allow_partial_merge: bool,
) -> PipelineConfig:
    if job is not None:
        cfg = _load_job_config(job)
        if input_path is not None:
            cfg.input_path = input_path
    elif config is not None:
        cfg = PipelineConfig.load(config)
        if input_path is not None:
            cfg.input_path = input_path
        cfg.output_dir = output_dir
        cfg.project_name = project_name
    else:
        if input_path is None:
            raise typer.BadParameter("Provide --input, --config, or --job")
        cfg = PipelineConfig(
            input_path=input_path,
            output_dir=output_dir,
            project_name=project_name,
        )

    # Applying large-8k defaults resets stages — do it before stage selection
    if large_8k and job is None:
        cfg.enable_mac_long_360_defaults()
    elif large_8k and job is not None and cfg.mode != "tiled":
        cfg.enable_mac_long_360_defaults()

    if trainer:
        if trainer not in {"brush", "opensplat"}:
            raise typer.BadParameter("trainer must be 'brush' or 'opensplat'")
        cfg.train.backend = trainer  # type: ignore[assignment]
    if no_refine:
        cfg.refine.enabled = False
    if streamed_lod:
        cfg.export.streamed_lod = True
    if no_cloud_manifest:
        cfg.package.cloud_manifest = False
    if no_quality:
        cfg.package.quality_report = False
    if no_preflight:
        cfg.preflight = False
    if allow_unstitched:
        cfg.allow_unstitched = True
    if allow_partial_merge:
        cfg.allow_partial_merge = True

    mode = "tiled" if (cfg.mode == "tiled" or cfg.chunk.enabled or large_8k) else "single"
    try:
        selected = select_stages(
            mode=mode,  # type: ignore[arg-type]
            only=only,
            from_stage=from_stage,
            to_stage=to_stage,
            stages=stages,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if selected is not None:
        cfg.stages = selected
        console.print(f"[cyan]Stages:[/cyan] {', '.join(selected)}")

    if fps is not None:
        cfg.extract.fps = fps
        cfg.chunk.base_fps = fps
    if no_mask:
        cfg.mask.enabled = False
    if export_formats:
        cfg.export.formats = [s.strip() for s in export_formats.split(",") if s.strip()]  # type: ignore[assignment]
    if dry_run:
        cfg.dry_run = True
    if with_viewer:
        cfg.train.with_viewer = True
        cfg.metal.serialize_brush = False
    return cfg


def _execute_pipeline(cfg: PipelineConfig) -> None:
    from instasplat.utils.progress import ProgressEvent, format_duration

    with Progress() as progress:
        task = progress.add_task("pipeline", total=1.0)

        def on_progress(ev: ProgressEvent) -> None:
            progress.update(
                task,
                completed=ev.overall_frac,
                description=(
                    f"{ev.stage}: {ev.message} "
                    f"(elapsed {format_duration(ev.stage_elapsed_sec)}, "
                    f"ETA {format_duration(ev.stage_eta_sec)})"
                ),
            )
            console.log(ev.terminal_line())

        result = Pipeline(cfg, on_progress=on_progress).run()

    if result.stage_timings:
        console.print("\n[bold]Stage elapsed[/bold]")
        for name, secs in result.stage_timings.items():
            console.print(f"  {name}: {format_duration(secs)}")

    if result.success:
        console.print(f"[green]Done[/green] → {result.paths.root}")
        if result.export:
            for k, v in result.export.outputs.items():
                console.print(f"  {k}: {v}")
        if result.tiled and result.tiled.chunk_results:
            ok = sum(1 for v in result.tiled.chunk_results.values() if v)
            console.print(f"  chunks ok: {ok}/{len(result.tiled.chunk_results)}")
        qpath = result.paths.root / "quality.json"
        if qpath.exists():
            console.print(f"  quality: {qpath}")
        if (result.paths.root / "cloud_job.json").exists():
            console.print(f"  cloud job: {result.paths.root / 'cloud_job.json'}")
        console.print(f"  preflight: {result.paths.root / 'preflight.json'}")
    else:
        console.print(f"[red]Failed[/red]: {result.error}")
        raise typer.Exit(code=1)


@app.command("install-brush")
def install_brush_cmd(
    force: bool = typer.Option(False, "--force", help="Rebuild even if brush is on PATH"),
) -> None:
    """Clone and build ArthurBrussee/brush (Metal/WebGPU), install to ~/.local/bin."""
    from instasplat.utils.brush_install import install_brush

    console.print("[cyan]Installing Brush…[/cyan] (Rust release build; may take several minutes)")
    result = install_brush(force_rebuild=force)
    if result.ok:
        console.print(f"[green]OK[/green] {result.message}")
        if result.brush_path:
            console.print(f"  binary: {result.brush_path}")
    else:
        console.print(f"[red]Failed[/red] {result.message}")
        if result.log_path:
            console.print(f"  log: {result.log_path}")
        raise typer.Exit(code=1)


@app.command("validate")
def validate(
    job: Path | None = typer.Option(
        None, "--job", "-j", help="Existing job directory with ingest/chunks"
    ),
    overlap_sec: float = typer.Option(5.0, help="Planned chunk overlap (seconds)"),
    chunk_duration_sec: float = typer.Option(25.0, help="Planned chunk duration"),
    base_fps: float = typer.Option(6.0, help="Planned base sample FPS"),
) -> None:
    """Check capture / tile health and print a quality report (no training)."""
    from instasplat.utils.paths import JobPaths
    from instasplat.utils.quality import build_quality_report, validate_capture

    if job is None:
        raise typer.BadParameter("Provide --job pointing at a run directory")
    paths = JobPaths(job)
    cfg = PipelineConfig(
        input_path=paths.video if paths.video.exists() else job,
        output_dir=job.parent,
        project_name=job.name,
    )
    cfg.chunk.overlap_sec = overlap_sec
    cfg.chunk.duration_sec = chunk_duration_sec
    cfg.chunk.base_fps = base_fps
    if (job / "config.yaml").exists():
        try:
            cfg = PipelineConfig.load(job / "config.yaml")
        except Exception:  # noqa: BLE001
            pass

    report = build_quality_report(cfg, paths)
    # Also surface standalone capture issues even without manifest
    if not (paths.chunks / "manifest.json").exists():
        extra = validate_capture(
            duration_sec=0.0,
            overlap_sec=cfg.chunk.overlap_sec,
            chunk_duration_sec=cfg.chunk.duration_sec,
            min_overlap_ratio=cfg.chunk.min_overlap_ratio,
            base_fps=cfg.chunk.base_fps,
            gyro_csv=paths.gyro_csv if paths.gyro_csv.exists() else None,
            gps_csv=paths.gps_csv if paths.gps_csv.exists() else None,
        )
        report.issues.extend(extra)

    console.print(f"[bold]Grade[/bold]: {report.grade} (score {report.score:.0f})")
    for issue in report.issues:
        color = {"error": "red", "warn": "yellow", "info": "cyan"}.get(issue.level, "white")
        console.print(f"  [{color}]{issue.level}[/{color}] {issue.code}: {issue.message}")
    if report.metrics:
        console.print("[bold]Metrics[/bold]")
        for k, v in report.metrics.items():
            console.print(f"  {k}: {v}")
    out = job / "quality.json"
    report.save(out)
    console.print(f"Wrote {out}")
    if report.grade == "poor":
        raise typer.Exit(code=2)


@app.command("gui")
def gui() -> None:
    """Launch the desktop GUI (requires PySide6)."""
    try:
        from instasplat.gui.app import launch
    except ImportError as exc:
        console.print(
            "[red]GUI dependencies missing.[/red] Install with: pip install 'instasplat[gui]'"
        )
        raise typer.Exit(code=1) from exc
    launch()


if __name__ == "__main__":
    app()
