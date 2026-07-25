"""Typer CLI for InstaSplat."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress

from instasplat import __version__
from instasplat.config import DEFAULT_CONFIG_TEMPLATE, PipelineConfig
from instasplat.pipeline import Pipeline
from instasplat.utils.deps import print_report

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
    ready = data["ready_stages"]
    if not ready.get("official_stitch"):
        console.print(
            "\n[yellow]Note:[/yellow] Official Insta360 MediaSDK is not available on macOS. "
            "Export a stitched equirectangular MP4 from Insta360 Studio, or use a Linux "
            "cloud/Docker worker for MediaSDK stitching."
        )


@app.command("init-config")
def init_config(
    out: Path = typer.Option(DEFAULT_CONFIG_PATH, "--out", "-o"),
) -> None:
    """Write a starter pipeline YAML config."""
    if out.exists():
        raise typer.BadParameter(f"{out} already exists")
    out.write_text(DEFAULT_CONFIG_TEMPLATE, encoding="utf-8")
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
        help="Comma-separated stages: ingest,extract,mask,sfm,scale,train,export",
    ),
    fps: float | None = typer.Option(None, help="Override extract FPS"),
    no_mask: bool = typer.Option(False, help="Disable YOLO people masking"),
    export_formats: str | None = typer.Option(
        None, "--formats", help="Comma-separated: ply,sog,spz,glb,html,csv,compressed.ply"
    ),
    dry_run: bool = typer.Option(False, help="Print/plan without executing heavy tools"),
    with_viewer: bool = typer.Option(False, help="Open Brush viewer while training"),
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

    if stages:
        cfg.stages = [s.strip() for s in stages.split(",") if s.strip()]
    if fps is not None:
        cfg.extract.fps = fps
    if no_mask:
        cfg.mask.enabled = False
    if export_formats:
        cfg.export.formats = [s.strip() for s in export_formats.split(",") if s.strip()]  # type: ignore[assignment]
    if dry_run:
        cfg.dry_run = True
    if with_viewer:
        cfg.train.with_viewer = True

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
