# SuperSplat live viewer (vendored)

InstaSplat embeds the official [PlayCanvas SuperSplat Viewer](https://github.com/playcanvas/supersplat-viewer)
(`@playcanvas/supersplat-viewer`) so the GUI shows **true Gaussian splats** during training,
not just point centers.

| File | Role |
|------|------|
| `assets/` | `index.html` / `index.css` / `index.js` from the npm package |
| `settings.json` | Default Experience Settings (v2) for the live preview |
| `server.py` | Localhost HTTP server serving assets + current `live.ply` |
| `widget.py` | `QWebEngineView` host |

Version pinned in `VERSION`. License: MIT (see `LICENSE`).

## Reload loop

1. Training writes subsampled `05_train/exports/live.ply` every `viewer_every` steps.
2. GUI polls `train_heartbeat.json` (~1s) and reloads the viewer URL with a cache-bust
   query when the PLY mtime / train step changes.
3. SuperSplat fetches `/content.ply` from the local server.

## Updating the viewer

```bash
npm pack @playcanvas/supersplat-viewer@<version>
tar -xzf playcanvas-supersplat-viewer-*.tgz
cp package/public/index.{html,css,js} instasplat/gui/supersplat/assets/
echo '<version>' > instasplat/gui/supersplat/VERSION
cp package/LICENSE instasplat/gui/supersplat/LICENSE
```
