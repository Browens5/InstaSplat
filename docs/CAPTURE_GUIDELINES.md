# Capture guidelines (Insta360 → Mac splat pipeline)

How to film so InstaSplat’s **full Mac 360 → splat pipeline** has a chance to
succeed. Pipeline overview: [MAC_360_PIPELINE.md](MAC_360_PIPELINE.md).

Adapted from on-the-fly-nvs, LongSplat, and Self-Cali-GS practices for the
sequential tiled path.

## Do

1. **Walk a continuous path** — ordered frames along a trajectory (not random orbits).
2. **Keep overlap** — ≥30–40% visual overlap between consecutive keyframes; our tiles use ~5s temporal overlap by default.
3. **Move smoothly** — constant walking speed; pause briefly at corners (gyro densifies there).
4. **Export stitched equirect** from Insta360 Studio (FlowState on) before InstaSplat on Mac.
5. **Enable GPS** when outdoors for metric tile alignment.
6. **Mask people/pets** (pipeline default) — dynamics destroy splat consistency (LongSplat warning).

## Don’t

1. Don’t teleport / cut discontinuous clips into one job without re-chunking.
2. Don’t spin in place for long (weak baseline for SfM).
3. Don’t rely on raw dual-fisheye without stitch for production quality.
4. Local path uses COLMAP EQUIRECTANGULAR + metal_equirect; cloud 3DGUT remains an optional scale-up.

## Suggested settings

| Capture | `chunk.base_fps` | `chunk.max_fps` | Notes |
|---------|------------------|-----------------|-------|
| Slow indoor walk | 8 | 15 | More texture, shorter tiles |
| Outdoor walk | 6 | 12–15 | GPS align on |
| Fast transit | 4 | 10 | Larger `target_path_length_m` |

LongSplat-style speed tip: if debugging, temporarily resize/equirect-downscale; for final 8K quality keep high `sfm.face_resolution` and full equirect width.
