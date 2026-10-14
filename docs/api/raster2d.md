# 2D raster rendering

`raster2d` turns a `Scene2D` primitive batch into `(B, H, W, 3)` uint8
frames. The navigation family and the internal Scene2D PushT scene use it. Torch provides the
CPU implementation and a readable CUDA reference; Triton fuses compositing
and supersample reduction into one CUDA kernel.

## Rendering contract

Primitive order determines compositing order. Static backgrounds may be baked
only when their geometry and appearance are shared across the batch. Output
offsets use int64 so large batches cannot wrap the address calculation.

Tests compare Torch and Triton pixels exactly. Agreement with an upstream
environment is a separate measurement, because its antialiasing and resize rules
may differ. Supersampling changes edge coverage and cost.

Classic control uses the separate `gfxdraw` kernels to reproduce SDL blending
and edge rules.

## Performance

On one H200, at 96 px and batch 256, the fused Triton path rendered 1,121,346 FPS and Torch
compositing 15,346 FPS. This compares two implementations in this library, not the upstream
renderer. See [The fused rasterizer](../concepts/rasterizer.md).

::: stable_envs.render.raster2d
