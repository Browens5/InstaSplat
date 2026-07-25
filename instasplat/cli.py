"""Typer CLI for InstaSplat."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress

from instasplat import __version__
from instasplat.config import DEFAULT_CONFIG_TEMPLATE, LARGE_8K_CONFIG_TEMPLATE, PipelineConfig
from instasplat.pipeline import Pipeline
from instasplat.utils.deps import print_report
from instasplat.utils.metal import metal_report

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
    console.print(
        "\n[cyan]Large 8K tip:[/cyan] use `instasplat run --large-8k ...` to auto-chunk, "
        "gyro/GPS-align tiles, and prefer Metal for YOLO + Brush."
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


@app.command("run")
def run(
    input_path: Path | None = typer.Option(
        None, "--input", "-i", help="INSV or prestitched equirect MP4"
    ),
    output_dir: Path = typer.Option(DEFAULT_OUT, "--output", "-o"),
    project_name: str = typer.Option("instasplat_job", "--name", "-n"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    stages: str | None = typer.Option(
        None,
        "--stages",
        help="Comma-separated stages",
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
        help="Enable tiled 8K@30 mode: auto-chunk, gyro/GPS align, Metal-first",
    ),
    tiled: bool = typer.Option(False, "--tiled", help="Alias for enabling chunk.mode=tiled"),
    trainer: str | None = typer.Option(
        None,
        "--trainer",
        help="Training backend: brush (default) or opensplat (Metal MPS)",
    ),
) -> None:
    """Run the reconstruction pipeline."""
    if config is not None:
        cfg = PipelineConfig.load(config)
        if input_path is not None:
            cfg.input_path = input_path
        cfg.output_dir = output_dir
        cfg.project_name = project_name
    else:
        if input_path is None:
            raise typer.BadParameter("Provide --input or --config")
        cfg = PipelineConfig(
            input_path=input_path,
            output_dir=output_dir,
            project_name=project_name,
        )

    if large_8k or tiled:
        cfg.enable_large_8k_defaults()
    if trainer:
        if trainer not in {"brush", "opensplat"}:
            raise typer.BadParameter("trainer must be 'brush' or 'opensplat'")
        cfg.train.backend = trainer  # type: ignore[assignment]
    if stages:
        cfg.stages = [s.strip() for s in stages.split(",") if s.strip()]
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

    with Progress() as progress:
        task = progress.add_task("pipeline", total=1.0)

        def on_progress(stage: str, frac: float, msg: str) -> None:
            progress.update(task, completed=frac, description=f"{stage}: {msg}")
            console.log(msg)

        result = Pipeline(cfg, on_progress=on_progress).run()

    if result.success:
        console.print(f"[green]Done[/green] → {result.paths.root}")
        if result.export:
            for k, v in result.export.outputs.items():
                console.print(f"  {k}: {v}")
        if result.tiled and result.tiled.chunk_results:
            ok = sum(1 for v in result.tiled.chunk_results.values() if v)
            console.print(f"  chunks ok: {ok}/{len(result.tiled.chunk_results)}")
    else:
        console.print(f"[red]Failed[/red]: {result.error}")
        raise typer.Exit(code=1)


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
