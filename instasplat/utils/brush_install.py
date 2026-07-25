"""Clone + build ArthurBrussee/brush and install the binary on PATH."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from instasplat.utils.deps import check_brush
from instasplat.utils.process import get_logger


BRUSH_REPO = "https://github.com/ArthurBrussee/brush.git"
DEFAULT_SRC = Path.home() / ".cache" / "instasplat" / "brush"
DEFAULT_BIN_DIR = Path.home() / ".local" / "bin"


@dataclass
class BrushInstallResult:
    ok: bool
    brush_path: Path | None
    message: str
    log_path: Path | None = None


def _run(cmd: list[str], *, cwd: Path | None = None, log_file: Path | None = None) -> int:
    log = get_logger("instasplat.brush_install")
    log.info("$ %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"$ {' '.join(cmd)}\n")
            f.write(proc.stdout or "")
            f.write(proc.stderr or "")
            f.write(f"\nEXIT {proc.returncode}\n\n")
    return proc.returncode


def ensure_rust() -> tuple[bool, str]:
    if shutil.which("cargo") and shutil.which("rustc"):
        return True, "Rust already installed"
    rustup = shutil.which("rustup")
    if rustup:
        code = _run([rustup, "show"])
        return code == 0, "rustup present"
    # Non-interactive rustup install
    installer = subprocess.run(
        ["curl", "--proto", "=https", "--tlsv1.2", "-sSf", "https://sh.rustup.rs"],
        capture_output=True,
        text=True,
        check=False,
    )
    if installer.returncode != 0 or not installer.stdout:
        return False, "Failed to download rustup — install from https://rustup.rs"
    proc = subprocess.run(
        ["sh", "-s", "--", "-y"],
        input=installer.stdout,
        text=True,
        check=False,
        capture_output=True,
    )
    cargo_env = Path.home() / ".cargo" / "env"
    if cargo_env.exists():
        # Ensure current process can see cargo on next which() via PATH patch
        cargo_bin = Path.home() / ".cargo" / "bin"
        os.environ["PATH"] = f"{cargo_bin}:{os.environ.get('PATH', '')}"
    return proc.returncode == 0, "Installed Rust via rustup" if proc.returncode == 0 else proc.stderr


def find_built_brush(src: Path) -> Path | None:
    candidates = [
        src / "target" / "release" / "brush",
        src / "target" / "release" / "brush-app",
        src / "target" / "release" / "brush_cli",
    ]
    # Also scan release dir for a brush* binary
    release = src / "target" / "release"
    if release.is_dir():
        for p in sorted(release.iterdir()):
            if p.is_file() and os.access(p, os.X_OK) and "brush" in p.name.lower():
                if p.name.endswith(".d") or p.suffix:
                    continue
                candidates.insert(0, p)
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return c
    return None


def install_brush(
    *,
    src_dir: Path | None = None,
    bin_dir: Path | None = None,
    force_rebuild: bool = False,
) -> BrushInstallResult:
    """
    Clone ArthurBrussee/brush (if needed), `cargo build --release`, and install
    the binary to ``~/.local/bin/brush`` (or ``~/.cargo/bin/brush``).
    """
    log = get_logger("instasplat.brush_install")
    src = src_dir or DEFAULT_SRC
    bin_dir = bin_dir or DEFAULT_BIN_DIR
    log_path = src / "instasplat_build.log"

    existing = check_brush("brush")
    if existing.available and existing.path and not force_rebuild:
        return BrushInstallResult(True, Path(existing.path), f"Brush already available: {existing.path}")

    ok_rust, rust_msg = ensure_rust()
    if not ok_rust:
        return BrushInstallResult(False, None, rust_msg, log_path)
    cargo = shutil.which("cargo")
    if not cargo:
        cargo_bin = Path.home() / ".cargo" / "bin" / "cargo"
        if cargo_bin.exists():
            cargo = str(cargo_bin)
            os.environ["PATH"] = f"{cargo_bin.parent}:{os.environ.get('PATH', '')}"
        else:
            return BrushInstallResult(False, None, "cargo not found after rustup", log_path)

    src.parent.mkdir(parents=True, exist_ok=True)
    if (src / ".git").exists():
        log.info("Updating Brush checkout at %s", src)
        _run(["git", "-C", str(src), "fetch", "--depth", "1", "origin", "main"], log_file=log_path)
        _run(["git", "-C", str(src), "reset", "--hard", "origin/main"], log_file=log_path)
    else:
        if src.exists():
            shutil.rmtree(src)
        log.info("Cloning Brush into %s", src)
        code = _run(
            ["git", "clone", "--depth", "1", BRUSH_REPO, str(src)],
            log_file=log_path,
        )
        if code != 0:
            return BrushInstallResult(False, None, "git clone failed — see log", log_path)

    log.info("Building Brush (release) — this can take a while…")
    code = _run(
        [cargo, "build", "--release"],
        cwd=src,
        log_file=log_path,
    )
    if code != 0:
        return BrushInstallResult(
            False,
            None,
            f"cargo build --release failed — see {log_path}",
            log_path,
        )

    built = find_built_brush(src)
    if built is None:
        return BrushInstallResult(False, None, f"Build succeeded but binary not found under {src}/target/release", log_path)

    bin_dir.mkdir(parents=True, exist_ok=True)
    dest = bin_dir / "brush"
    try:
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(built.resolve())
    except OSError:
        shutil.copy2(built, dest)
        dest.chmod(0o755)

    # Also link into ~/.cargo/bin when present (common on PATH after rustup)
    cargo_bin_dir = Path.home() / ".cargo" / "bin"
    if cargo_bin_dir.is_dir():
        cargo_dest = cargo_bin_dir / "brush"
        try:
            if cargo_dest.exists() or cargo_dest.is_symlink():
                cargo_dest.unlink()
            cargo_dest.symlink_to(built.resolve())
        except OSError:
            pass

    os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"
    verify = check_brush(str(dest))
    if not verify.available:
        return BrushInstallResult(
            True,
            dest,
            f"Installed {dest} (symlink → {built}); open a new shell if `brush` is not on PATH",
            log_path,
        )
    return BrushInstallResult(True, dest, f"Brush installed: {dest} → {built}", log_path)


def ensure_brush(auto_install: bool = True) -> BrushInstallResult:
    status = check_brush("brush")
    if status.available and status.path:
        return BrushInstallResult(True, Path(status.path), f"Brush OK: {status.path}")
    if not auto_install:
        return BrushInstallResult(False, None, status.notes or "Brush not found")
    return install_brush()
