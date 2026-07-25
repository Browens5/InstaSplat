"""Install ArthurBrussee/brush — prefer GitHub release binary, else cargo build."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from instasplat.utils.deps import check_brush
from instasplat.utils.process import get_logger


BRUSH_REPO = "https://github.com/ArthurBrussee/brush.git"
BRUSH_RELEASES_API = "https://api.github.com/repos/ArthurBrussee/brush/releases/latest"
DEFAULT_SRC = Path.home() / ".cache" / "instasplat" / "brush"
DEFAULT_BIN_DIR = Path.home() / ".local" / "bin"
# Brush workspace uses edition 2024; README asks for Rust 1.88+
MIN_RUST = (1, 88, 0)


@dataclass
class BrushInstallResult:
    ok: bool
    brush_path: Path | None
    message: str
    log_path: Path | None = None


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    log_file: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    log = get_logger("instasplat.brush_install")
    log.info("$ %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"$ {' '.join(cmd)}\n")
            f.write(out)
            f.write(f"\nEXIT {proc.returncode}\n\n")
    return proc.returncode, out


def _tail(text: str, lines: int = 40) -> str:
    parts = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not parts:
        return ""
    return "\n".join(parts[-lines:])


def _parse_rust_version(text: str) -> tuple[int, int, int] | None:
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _rustc_version(rustc: str) -> tuple[int, int, int] | None:
    code, out = _run([rustc, "--version"])
    if code != 0:
        return None
    return _parse_rust_version(out)


def _prefer_rustup_path() -> None:
    """Put ~/.cargo/bin ahead of Homebrew so rustup toolchains win."""
    cargo_bin = Path.home() / ".cargo" / "bin"
    if cargo_bin.is_dir():
        os.environ["PATH"] = f"{cargo_bin}:{os.environ.get('PATH', '')}"


def ensure_rust() -> tuple[bool, str, str | None]:
    """
    Ensure a new-enough rustc/cargo is available.

    Returns (ok, message, cargo_path).
    """
    _prefer_rustup_path()
    cargo = shutil.which("cargo")
    rustc = shutil.which("rustc")
    if cargo and rustc:
        ver = _rustc_version(rustc)
        if ver and ver >= MIN_RUST:
            return True, f"Rust {ver[0]}.{ver[1]}.{ver[2]} OK", cargo
        if ver:
            msg = (
                f"Rust {ver[0]}.{ver[1]}.{ver[2]} is too old for Brush "
                f"(need {MIN_RUST[0]}.{MIN_RUST[1]}+ / edition 2024). Upgrading via rustup…"
            )
        else:
            msg = "Could not parse rustc version; ensuring rustup toolchain…"
        get_logger("instasplat.brush_install").warning(msg)

    rustup = shutil.which("rustup") or str(Path.home() / ".cargo" / "bin" / "rustup")
    if not Path(rustup).exists() and not shutil.which("rustup"):
        installer = subprocess.run(
            ["curl", "--proto", "=https", "--tlsv1.2", "-sSf", "https://sh.rustup.rs"],
            capture_output=True,
            text=True,
            check=False,
        )
        if installer.returncode != 0 or not installer.stdout:
            return (
                False,
                "Failed to download rustup — install from https://rustup.rs "
                "(Brush needs Rust 1.88+). Or use a GitHub release binary via "
                "`instasplat install-brush` on Apple Silicon.",
                None,
            )
        proc = subprocess.run(
            ["sh", "-s", "--", "-y"],
            input=installer.stdout,
            text=True,
            check=False,
            capture_output=True,
        )
        if proc.returncode != 0:
            return False, proc.stderr or "rustup install failed", None
        _prefer_rustup_path()
        rustup = shutil.which("rustup") or str(Path.home() / ".cargo" / "bin" / "rustup")

    # Install/update a recent stable toolchain
    for cmd in (
        [rustup, "toolchain", "install", "stable"],
        [rustup, "default", "stable"],
        [rustup, "update", "stable"],
    ):
        _run(cmd)

    _prefer_rustup_path()
    cargo = shutil.which("cargo")
    rustc = shutil.which("rustc")
    # Explicitly prefer rustup shims
    cargo_home = Path.home() / ".cargo" / "bin" / "cargo"
    rustc_home = Path.home() / ".cargo" / "bin" / "rustc"
    if cargo_home.exists():
        cargo = str(cargo_home)
    if rustc_home.exists():
        rustc = str(rustc_home)
    if not cargo or not rustc:
        return False, "cargo/rustc not found after rustup", None
    ver = _rustc_version(rustc)
    if not ver or ver < MIN_RUST:
        return (
            False,
            f"Rust still too old after rustup ({ver}); need {MIN_RUST[0]}.{MIN_RUST[1]}+. "
            f"Run: rustup update stable && rustup default stable",
            cargo,
        )
    return True, f"Rust {ver[0]}.{ver[1]}.{ver[2]} ready", cargo


def find_built_brush(src: Path) -> Path | None:
    candidates = [
        src / "target" / "release" / "brush",
        src / "target" / "release" / "brush_app",
        src / "target" / "release" / "brush-app",
        src / "target" / "release" / "brush-cli",
        src / "target" / "release" / "brush_cli",
    ]
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


def _host_release_asset_names() -> list[str]:
    """Ordered candidate asset basenames for this machine."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        arch = "aarch64"
    elif machine in {"x86_64", "amd64"}:
        arch = "x86_64"
    else:
        return []

    if system == "darwin":
        return [f"brush-app-{arch}-apple-darwin.tar.xz"]
    if system == "linux":
        return [f"brush-app-{arch}-unknown-linux-gnu.tar.xz"]
    if system == "windows":
        return [f"brush-app-{arch}-pc-windows-msvc.zip"]
    return []


def _http_json(url: str) -> dict:
    req = Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "instasplat"})
    with urlopen(req, timeout=60) as resp:  # noqa: S310 — fixed GitHub API URL
        import json

        return json.loads(resp.read().decode("utf-8"))


def _download(url: str, dest: Path) -> None:
    req = Request(url, headers={"User-Agent": "instasplat"})
    with urlopen(req, timeout=300) as resp, dest.open("wb") as f:  # noqa: S310
        shutil.copyfileobj(resp, f)


def _extract_brush_binary(archive: Path, dest_dir: Path) -> Path | None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    if archive.suffix == ".zip" or archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)
            names = zf.namelist()
    else:
        with tarfile.open(archive, "r:*") as tf:
            # filter= avoids Python 3.14 deprecation; ignored on older Pythons
            try:
                tf.extractall(dest_dir, filter="data")  # type: ignore[call-arg]
            except TypeError:
                tf.extractall(dest_dir)
            names = [m.name for m in tf.getmembers() if m.isfile()]

    # Prefer brush / brush_app executables
    preferred = ("brush", "brush_app", "brush-app", "brush-cli")
    found: list[Path] = []
    for root, _, files in os.walk(dest_dir):
        for name in files:
            p = Path(root) / name
            if not p.is_file():
                continue
            if name in preferred or name.replace("-", "_") in {"brush", "brush_app", "brush_cli"}:
                found.append(p)
    for want in preferred:
        for p in found:
            if p.name == want or p.name.replace("-", "_") == want.replace("-", "_"):
                if not os.access(p, os.X_OK):
                    p.chmod(0o755)
                return p
    # fallback: any executable with brush in the name
    for p in found:
        if os.access(p, os.X_OK) or True:
            p.chmod(0o755)
            return p
    # last resort: scan extracted paths from archive listing
    for name in names:
        p = dest_dir / name
        if p.is_file() and "brush" in p.name.lower() and p.suffix not in {".md", ".txt", ".json"}:
            p.chmod(0o755)
            return p
    return None


def install_from_release(*, bin_dir: Path, log_path: Path) -> BrushInstallResult | None:
    """
    Download the latest GitHub release binary for this platform.

    Returns None when no matching asset exists (caller should fall back to source).
    """
    log = get_logger("instasplat.brush_install")
    candidates = _host_release_asset_names()
    if not candidates:
        return None
    try:
        data = _http_json(BRUSH_RELEASES_API)
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        log.warning("Could not query Brush releases: %s", exc)
        return None

    assets = {a["name"]: a for a in data.get("assets") or [] if isinstance(a, dict)}
    asset = None
    for name in candidates:
        if name in assets:
            asset = assets[name]
            break
    if asset is None:
        log.info("No prebuilt Brush asset for this platform among %s", candidates)
        return None

    tag = data.get("tag_name") or "latest"
    url = asset.get("browser_download_url")
    if not url:
        return None

    log.info("Downloading Brush %s (%s)…", tag, asset["name"])
    with tempfile.TemporaryDirectory(prefix="instasplat-brush-") as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / asset["name"]
        try:
            _download(url, archive)
        except (URLError, TimeoutError, OSError) as exc:
            return BrushInstallResult(
                False,
                None,
                f"Failed to download Brush release: {exc}",
                log_path,
            )
        extract_dir = tmp_path / "out"
        built = _extract_brush_binary(archive, extract_dir)
        if built is None:
            return BrushInstallResult(
                False,
                None,
                f"Downloaded {asset['name']} but could not find brush binary inside",
                log_path,
            )
        # Persist binary under cache so symlink target stays valid
        cache_bin = DEFAULT_SRC.parent / "brush-release" / tag
        cache_bin.mkdir(parents=True, exist_ok=True)
        persisted = cache_bin / "brush"
        shutil.copy2(built, persisted)
        persisted.chmod(0o755)
        return _link_brush(persisted, bin_dir, log_path, note=f"from GitHub release {tag}")


def _link_brush(
    built: Path,
    bin_dir: Path,
    log_path: Path | None,
    *,
    note: str = "",
) -> BrushInstallResult:
    bin_dir.mkdir(parents=True, exist_ok=True)
    dest = bin_dir / "brush"
    try:
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(built.resolve())
    except OSError:
        shutil.copy2(built, dest)
        dest.chmod(0o755)

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
    suffix = f" ({note})" if note else ""
    return BrushInstallResult(
        True,
        dest,
        f"Brush installed: {dest} → {built}{suffix}",
        log_path,
    )


def install_from_source(
    *,
    src_dir: Path,
    bin_dir: Path,
    log_path: Path,
) -> BrushInstallResult:
    log = get_logger("instasplat.brush_install")
    ok_rust, rust_msg, cargo = ensure_rust()
    if not ok_rust or not cargo:
        return BrushInstallResult(False, None, rust_msg, log_path)
    log.info("%s", rust_msg)

    src_dir.parent.mkdir(parents=True, exist_ok=True)
    if (src_dir / ".git").exists():
        log.info("Updating Brush checkout at %s", src_dir)
        _run(["git", "-C", str(src_dir), "fetch", "--depth", "1", "origin", "main"], log_file=log_path)
        _run(["git", "-C", str(src_dir), "reset", "--hard", "origin/main"], log_file=log_path)
    else:
        if src_dir.exists():
            shutil.rmtree(src_dir)
        log.info("Cloning Brush into %s", src_dir)
        code, out = _run(
            ["git", "clone", "--depth", "1", BRUSH_REPO, str(src_dir)],
            log_file=log_path,
        )
        if code != 0:
            return BrushInstallResult(
                False,
                None,
                f"git clone failed:\n{_tail(out)}",
                log_path,
            )

    # Build only the desktop app binary — full workspace pulls WASM/Android targets
    log.info("Building Brush (release, -p brush-app --bin brush) — this can take a while…")
    env = os.environ.copy()
    # Avoid accidentally using an older Homebrew cargo via PATH mid-build
    _prefer_rustup_path()
    cargo = shutil.which("cargo") or cargo
    code, out = _run(
        [cargo, "build", "--release", "-p", "brush-app", "--bin", "brush"],
        cwd=src_dir,
        log_file=log_path,
        env=env,
    )
    if code != 0:
        # Fallback: older layouts may only expose brush-cli
        log.warning("brush-app build failed; trying brush-cli…")
        code2, out2 = _run(
            [cargo, "build", "--release", "-p", "brush-cli", "--bin", "brush-cli"],
            cwd=src_dir,
            log_file=log_path,
            env=env,
        )
        if code2 != 0:
            return BrushInstallResult(
                False,
                None,
                "cargo build failed (needs Rust 1.88+).\n"
                f"Last log lines:\n{_tail(out + out2)}\n"
                f"Full log: {log_path}\n"
                "Tip: on Apple Silicon, `instasplat install-brush` downloads a prebuilt binary.",
                log_path,
            )

    built = find_built_brush(src_dir)
    if built is None:
        return BrushInstallResult(
            False,
            None,
            f"Build succeeded but binary not found under {src_dir}/target/release",
            log_path,
        )
    return _link_brush(built, bin_dir, log_path, note="from source build")


def install_brush(
    *,
    src_dir: Path | None = None,
    bin_dir: Path | None = None,
    force_rebuild: bool = False,
    from_source: bool = False,
) -> BrushInstallResult:
    """
    Install Brush.

    Default: download the latest GitHub release for this platform (fast).
    Fallback / ``from_source=True``: clone + ``cargo build -p brush-app --bin brush``
    with Rust 1.88+ via rustup (Homebrew cargo is often too old).
    """
    log = get_logger("instasplat.brush_install")
    src = src_dir or DEFAULT_SRC
    bin_dir = bin_dir or DEFAULT_BIN_DIR
    log_path = src.parent / "brush_install.log"
    if src_dir:
        log_path = src / "instasplat_build.log"

    existing = check_brush("brush")
    if existing.available and existing.path and not force_rebuild:
        return BrushInstallResult(True, Path(existing.path), f"Brush already available: {existing.path}")

    if not from_source:
        release = install_from_release(bin_dir=bin_dir, log_path=log_path)
        if release is not None and release.ok:
            return release
        if release is not None and not release.ok:
            log.warning("Release install failed: %s — trying source build", release.message)
        else:
            log.info("No prebuilt Brush for this platform — building from source")

    return install_from_source(src_dir=src, bin_dir=bin_dir, log_path=log_path)


def ensure_brush(auto_install: bool = True) -> BrushInstallResult:
    status = check_brush("brush")
    if status.available and status.path:
        return BrushInstallResult(True, Path(status.path), f"Brush OK: {status.path}")
    if not auto_install:
        return BrushInstallResult(False, None, status.notes or "Brush not found")
    return install_brush()
