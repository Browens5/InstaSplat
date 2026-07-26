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


@app.command("setup")
def setup_cmd(
    no_system: bool = typer.Option(
        False,
        "--no-system",
        "--verify",
        help="Only check readiness (skip brew/npm installs)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show actions without installing"),
) -> None:
    """Install/verify tools for the metal_equirect pipeline (streamlined)."""
    from instasplat.utils.setup_env import run_setup

    console.print("[bold cyan]InstaSplat setup[/bold cyan]")
    report = run_setup(install_system=not no_system, dry_run=dry_run)
    table = Table(title="Setup checklist")
    table.add_column("Step")
    table.add_column("OK")
    table.add_column("Detail")
    for step in report.steps:
        table.add_row(step.name, "✓" if step.ok else "✗", step.detail or "—")
    console.print(table)
    if report.ready:
        console.print("\n[green]Ready[/green] for metal splat training.")
    else:
        console.print(
            "\n[yellow]Not fully ready.[/yellow] "
            "On a fresh Mac, prefer: ./scripts/setup_macos.sh"
        )
        raise typer.Exit(code=1)
    console.print("\n[bold]Next[/bold]")
    for line in report.next_commands:
        console.print(f"  {line}")


@app.command("doctor")
def doctor(
    splat_transform_bin: str = typer.Option(
        "splat-transform", help="splat-transform binary name/path"
    ),
) -> None:
    """Check local dependencies and which stages can run on this Mac."""
    from instasplat.utils import deps as deps_mod

    data = deps_mod.report_dict(splat_transform_bin)
    print_report(console)
    metal = metal_report()
    console.print("\n[bold]Metal / Apple GPU[/bold]")
    console.print(metal)
    if not data["ready_stages"].get("official_stitch"):
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
            "\n[yellow]PyTorch missing:[/yellow] run `./scripts/setup_macos.sh` or "
            "`pip install -e .` (Apple Silicon: MPS wheel from pytorch.org)."
        )
    console.print(
        "[cyan]Tip:[/cyan] see docs/METAL_SPLAT_WORKFLOW.md — then "
        "`instasplat mac-360 -i ./capture_equirect.mp4 -o ./runs -n walk`"
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
        large_8k=large_8k,
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
    large_8k: bool = typer.Option(
        False,
        "--large-8k",
        help="Enable Mac long-360 tiled mode (alias of mac-360 defaults)",
    ),
    tiled: bool = typer.Option(False, "--tiled", help="Alias for enabling chunk.mode=tiled"),
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
        large_8k=large_8k or tiled,
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
    no_mask: bool = typer.Option(False, help="Disable YOLO people masking"),
    dry_run: bool = typer.Option(False, help="Plan chunks + preflight only"),
    allow_unstitched: bool = typer.Option(False, "--allow-unstitched"),
    allow_partial_merge: bool = typer.Option(False, "--allow-partial-merge"),
    formats: str = typer.Option("ply,sog,spz", "--formats"),
) -> None:
    """
    Best local Mac pipeline: long 360 video → tiled Metal equirect splat.

    Expects a stitched equirectangular MP4 from Insta360 Studio. Keep the
    original .insv beside it (or gyro.csv/gps.csv sidecars) for turn densify
    and metric GPS scale.
    """
    console.print(
        "[bold cyan]InstaSplat mac-360[/bold cyan] — tiled Metal pipeline "
        "(YOLO MPS → COLMAP → metal_equirect → GPS/gyro merge)"
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
        large_8k=True,
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
    from instasplat.utils.jobs import load_job_config

    return load_job_config(job)


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
    large_8k: bool,
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

    cfg.train.backend = "metal_equirect"
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
            if not ev.quiet:
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


@app.command("train-equirect")
def train_equirect_cmd(
    job: Path = typer.Option(..., "--job", "-j", help="Job folder with equirect frames + SfM"),
    steps: int | None = typer.Option(None, help="Override train.total_steps"),
    max_resolution: int | None = typer.Option(None, help="Equirect train width"),
    composite: str = typer.Option("tile", help="tile | oit"),
    no_eval3d: bool = typer.Option(False, "--no-eval3d", help="Disable 3D response term"),
    dry_run: bool = typer.Option(False, help="Load dataset only"),
) -> None:
    """Run the Mac-native metal_equirect trainer on an existing job."""
    from instasplat.metal_equirect.backend import run_metal_equirect_train
    from instasplat.utils.jobs import inspect_job
    from instasplat.utils.paths import JobPaths

    info = inspect_job(job)
    cfg = info.config
    cfg.train.backend = "metal_equirect"
    cfg.dry_run = dry_run
    if steps is not None:
        cfg.train.total_steps = steps
    if max_resolution is not None:
        cfg.train.max_resolution = max_resolution
    cfg.train.composite = composite  # type: ignore[assignment]
    cfg.train.with_eval3d = not no_eval3d
    paths = JobPaths(info.job_dir)
    # Prefer refined → scaled → colmap model
    model = paths.root / "03b_refine" / "sparse" / "0"
    if not (model / "images.txt").exists() and not (model / "images.bin").exists():
        model = paths.scaled_model
    if not (model / "images.txt").exists() and not (model / "images.bin").exists():
        model = paths.colmap_model
    console.print(
        f"[cyan]metal_equirect[/cyan] job={paths.root} model={model} "
        f"steps={cfg.train.total_steps} composite={cfg.train.composite}"
    )
    result = run_metal_equirect_train(cfg, paths, model)
    if cfg.dry_run:
        console.print("[green]dry_run OK[/green]")
        return
    console.print(
        f"[green]OK[/green] device={result.device} gaussians={result.n_gaussians} "
        f"loss={result.final_loss:.5f}"
    )
    if result.ply_path:
        console.print(f"  ply: {result.ply_path}")
    if result.preview_path:
        console.print(f"  preview: {result.preview_path}")



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
