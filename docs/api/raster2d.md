# 2D raster rendering

`raster2d` turns a `Scene2D` primitive batch into `(B, H, W, 3)` uint8
frames. The navigation family and the internal Scene2D PushT scene use it. Torch provides the
CPU implementation and a readable CUDA reference. Triton fuses compositing
and supersample reduction into one CUDA kernel.

## Rendering contract

Primitive order determines compositing order. A renderer may bake a static background
only when its geometry and appearance are the same for the whole batch. Output
offsets use int64 so large batches cannot wrap the address calculation.

Tests compare Torch and Triton pixels exactly. Agreement with an upstream
environment is a separate measurement, because its antialiasing and resize rules
may differ. Supersampling changes edge coverage and cost.

Classic control uses the separate `gfxdraw` kernels to reproduce SDL blending
and edge rules.

## Design

The Triton kernel computes each output pixel's colour in registers from a few numbers per primitive. It
evaluates the whole supersample block and writes only the result, so it allocates no canvas. The Torch
backend composites on a `(B,3,R,R)` float canvas, one pass per primitive. It is the readable
specification and the CPU path.

Outlines are unions of per-edge capsules, so corners are round, as in pygame's `draw_fat_segment`.

## Supersampling

`supersample=ss` averages `ss × ss` subpixels per output pixel. The default is 1. Supersampling changes
edge pixels only. A larger `ss` costs more time per frame.

## Entry points

```python
from stable_envs.render.raster2d import render, render_triton, render_torch

img = render(scene, out_size=96, supersample=1, backend="auto")
```

`backend="auto"` uses Triton when it is available on a CUDA device, and Torch otherwise.
`Env.render` accepts `out=`, a preallocated `(B,H,W,3)` uint8 buffer. It removes the allocation from a
training loop.

## Cost outside the kernel

`Env.render` builds a `Scene2D` from the State, then rasterizes it. Scene construction issues a few
dozen small Torch ops from Python, so its cost changes little with the batch. The kernel is one launch
that scales with output pixels. Below about a thousand environments, scene construction takes most of
the time.

::: stable_envs.render.raster2d
