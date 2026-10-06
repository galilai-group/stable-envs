# 2D raster rendering

`raster2d` turns a `Scene2D` primitive batch into `(B, H, W, 3)` uint8
frames. PushT and several navigation render paths use it. Torch provides the
CPU implementation and a readable CUDA reference; Triton fuses compositing
and supersample reduction into one CUDA kernel.

## Rendering contract

Primitive order determines compositing order. Static backgrounds may be baked
only when their geometry and appearance are shared across the batch. Output
offsets use int64 so large batches cannot wrap the address calculation.

The backend checks compare Torch and Triton pixels exactly for their supported
scene configurations. Agreement with an upstream environment is a separate
measurement: its antialiasing and resize rules may differ. Supersampling changes
edge coverage and cost, and does not by itself establish source parity.

Classic control and CarRacing use the separate `gfxdraw` kernels to reproduce
SDL primitive blending and edge rules. Their regression tests compare the fused
kernels against a Torch reference and the per-primitive implementation.

## Recorded performance

The archived PushT microbenchmark at 96 px measured 1,121,346 FPS for the fused
Triton path and 15,346 FPS for Torch compositing at batch 256 on one H200. This
73.1× ratio compares two implementations in this library, **not the upstream
simulator's renderer**. The archive does not report a timing standard deviation.
It is not a fresh result for the current cleanup.

See the [original measurement scope](../concepts/rasterizer.md) and the
[testing guide](../testing.md) for source-comparison requirements.

::: stable_envs.render.raster2d
