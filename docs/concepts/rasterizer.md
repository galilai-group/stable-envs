# The fused rasterizer

One Triton kernel turns any `Scene2D` into `(B,H,W,3)` uint8 frames.

All numbers on this page were measured on one NVIDIA H200 with torch 2.11.0+cu130 and Triton 3.6.0.
They use the internal Scene2D PushT scene (`stable_envs/envs/pusht/legacy.py`) at 96 px unless stated.
The public PushT classes render with pygame. The tables come from separate microbenchmarks, so compare
ratios, not absolute values, across tables.

## Design

Each pixel's colour in a flat 2-D scene is a closed-form function of a few numbers per primitive. The
kernel evaluates the whole supersample block for an output pixel in registers and writes only the
result. It allocates no canvas.

The torch reference backend composites with tensor ops on a `(B,3,R,R)` float canvas, one pass per
primitive. It is kept as a readable specification and as a CPU fallback. The conformance suite
compares the two backends.

| batch | torch compositing | fused Triton | ratio |
|---|---|---|---|
| 64 | 3,870 frames/s | 327,850 frames/s | 84.7× |
| 256 | 15,346 frames/s | 1,121,346 frames/s | 73.1× |

On a 16-frame batch the two backends agree on 100% of pixels, with maximum difference 0.

## Output bandwidth

The kernel reads a small parameter block per primitive and writes the output. Whole `render()` calls,
including scene construction:

| batch | 96 px, GB/s | 224 px, GB/s |
|---|---|---|
| 256 | 3.0 | 15.2 |
| 1,024 | 11.4 | 47.4 |
| 4,096 | 37.6 | 101.4 |
| 16,384 | 89.1 | 141.8 |
| 65,536 | 131.9 | 155.2 |

The rasterizer alone reaches 155.0 GB/s at 96 px and 162.1 GB/s at 224 px.

## Supersampling

`supersample=ss` averages `ss × ss` subpixels per output pixel. The default is 1.

PushT against the official `gym_pusht` renderer, 64 states from the reset distribution:

| | interior pixels exact | all pixels exact | mean abs. diff |
|---|---|---|---|
| 96 px, ss=1 | 100.00% | 90.70% | 2.47 / 255 |
| 96 px, ss=2 | 100.00% | 93.79% | 1.35 / 255 |
| 224 px, ss=1 | 99.9985% | 95.60% | 1.19 / 255 |
| 224 px, ss=2 | 99.9985% | 97.04% | 0.57 / 255 |

Supersampling changes edge pixels only. gym_pusht draws at 512 px and resizes with `cv2.resize`, so
its edge pixels come from its own resampling filter.

Cost through the full `render()` call at batch 4,096:

| supersample | frames/s | µs/frame | cost vs ss=1 |
|---|---|---|---|
| 1 | 1,178,952 | 0.848 | 1.00× |
| 2 | 744,591 | 1.343 | 1.58× |
| 3 | 441,990 | 2.262 | 2.67× |

These include about 2 ms of scene construction per call, which does not depend on supersampling.

## Implementation notes

- **64-bit output offsets.** The frame base `b * H * W * 3` overflows int32 at large batches (at
  224 px, `b = 16383` gives 2.47e9). The kernel computes offsets in int64.
- **Outlines are capsules.** pygame's `draw_fat_segment` paints a disc at each endpoint, so corners are
  round. The kernel takes the union of per-edge capsules. An SDF band would give mitred corners.

## Entry points

```python
from stable_envs.render.raster2d import render, render_triton, render_torch

img = render(scene, out_size=96, supersample=1, backend="auto")   # Triton, torch fallback
```

`backend="auto"` uses Triton when available and torch otherwise. `Env.render` accepts an `out=` tensor:
a preallocated `(B,H,W,3)` uint8 buffer. This removes the allocation from the timed region and from a
training loop.

## Cost outside the kernel

`Env.render` builds a `Scene2D` from the states, then rasterizes it. `scene()` issues a few dozen small
torch ops from Python, so its cost is nearly independent of the batch. The rasterizer is one launch
that scales with output pixels. Below about a thousand environments, `scene()` dominates.

API details: [2D raster rendering](../api/raster2d.md).
