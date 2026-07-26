# macOS setup — install the 360 → splat pipeline

Install the tools InstaSplat needs to run the **full Mac pipeline**
(ffmpeg, COLMAP, PyTorch MPS, splat-transform, optional GUI).

- Why / architecture: **[MAC_360_PIPELINE.md](MAC_360_PIPELINE.md)**  
- After install, runbook: **[METAL_SPLAT_WORKFLOW.md](METAL_SPLAT_WORKFLOW.md)**

## Requirements

- macOS 13+ (Apple Silicon strongly preferred)
- [Homebrew](https://brew.sh)
- Python 3.11+
- Node.js 18+ (for `splat-transform`)
- Insta360 Studio (equirect export — MediaSDK is not on macOS)

## One-shot install

```bash
git clone https://github.com/Browens5/InstaSplat.git
cd InstaSplat
./scripts/setup_macos.sh
source .venv/bin/activate
instasplat doctor
```

`doctor` should report **mac_long_360 = yes** when the pipeline can run locally.

What the script does:

1. `brew install ffmpeg exiftool colmap git`
2. `npm i -g @playcanvas/splat-transform`
3. Create `.venv` and `pip install -e ".[gui,dev]"` (includes **torch**)
4. Run `instasplat setup` / `doctor`

| Environment variable | Meaning |
|----------------------|---------|
| `WITH_GUI=0` | Install without PySide6 |
| `SKIP_BREW=1` | Skip Homebrew |
| `SKIP_PIP=1` | Skip venv / pip |

## Verify / repair

```bash
instasplat setup              # may brew/npm install missing tools
instasplat setup --verify     # check only (alias: --no-system)
instasplat doctor             # stage readiness + MPS report
```

If Apple Silicon shows PyTorch **without MPS**, install the official wheel from
[pytorch.org](https://pytorch.org), then re-check with `doctor`.

## Manual install (equivalent)

```bash
brew install ffmpeg exiftool colmap git
npm i -g @playcanvas/splat-transform
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[gui,dev]"
instasplat doctor
```

## Capture input (pipeline expects this)

1. Export **equirectangular** MP4 from Insta360 Studio  
2. Keep sibling `.insv` (or `gyro.csv` / `gps.csv`) beside it  

## Next

- Create / understand the pipeline: [MAC_360_PIPELINE.md](MAC_360_PIPELINE.md)  
- Run it: [METAL_SPLAT_WORKFLOW.md](METAL_SPLAT_WORKFLOW.md)  
- Long 8K tiled jobs: [MAC_LONG_360.md](MAC_LONG_360.md)  
- GUI: `instasplat gui`  
