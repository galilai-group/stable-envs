# Writing a bit-exact Triton kernel

Most renderers here must match a reference with zero differing pixels. A rasterizer turns a float
comparison into a branch, and a resampler turns a float into an integer. One ulp on the wrong side is
a whole pixel or a whole colour level. Triton's defaults are not built for this. This page lists the
places where you must change them.

## 1. Division and square root are not correctly rounded

`a / b` on float32 lowers to PTX `div.full.f32`, and `tl.sqrt` lowers to `sqrt.approx.f32`. Both are
accurate to about 1 ulp. They do not give IEEE round-to-nearest. The error shows when the kernel
rounds or compares the result:

- `render/gfxdraw/fused.py` takes `floor` of a float32 quotient. An exact `-42.0 / 14.0 = -3.0` returned
  a value slightly below `-3`, and a scanline moved by one pixel.
- `render/raycast3d.py` uses `disc > 0` as the hit test. A 1-ulp error in the inputs changed hits.

**Fix:** use `tl.math.div_rn(a, b)` and `tl.sqrt_rn(x)`.

## 2. Multiply-add contraction is not stable across compilations

ptxas may contract `a*b + c*d` into an FMA. Whether it does depends on the surrounding code, including
`constexpr` values. Two specializations of one expression can then round differently. A contracted
`fma.rn.f32` also rounds once, where torch rounds `a*b + c` twice.

**Fix:**

- If the kernel must match a torch reference or another compilation of itself bit for bit, pass
  `enable_fp_fusion=False` to the launch. `render/raycast3d.py` does this.
- If it only has to match the reference renderer, fusion can stay on. Then do not write a test that
  compares two specializations at large batch. State the choice next to the kernel.

## 3. Integer division truncates

`//` on Triton integers truncates toward zero, like C. Python's `//` and
`torch.div(..., rounding_mode="floor")` round down. They differ for negative numerators, which a
scanline fill produces for polygons above the top edge.

**Fix:** prove the numerator is non-negative where the kernel uses the result, or write the floor explicitly.

## 4. Integer width is part of the specification

`gfxdraw`'s blend works on SDL's packed 32-bit channel word and relies on unsigned wrap-around. Doing
the arithmetic in int64 and masking with `0xFFFFFFFF` reproduces it. int32 does not, because
`diff * alpha` reaches 1.1e12.

Do not widen without need either. An int64 accumulator uses two registers instead of one and can cause
spills. Keep accumulators in int32 and widen only the expression that needs it.

## 5. Benchmark on the real workload

A result on the smallest scene can point the wrong way on a large one. For example,
`tl.static_range` helps with a few primitives. It is slower with about a hundred, and it does not
compile above about 340 (see `render/raster2d.py`). Always state which scene a number came from.

## 6. Equal metrics do not mean equal images

A mean over 64 frames cannot see a few badly wrong pixels. When the claim is "this edit does not
change the output", compare bytes, at every resolution you rely on. Use metrics only for fidelity
claims.

## 7. `cpu%` does not show where time goes

PyTorch's default synchronization spins. A GPU-bound render and a Python-bound render both show 100%
of a core. Measure device kernel time against wall time instead:

```python
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
    ...
evs = [e for e in pr.key_averages() if e.device_type == DeviceType.CUDA]
kernel_seconds = sum(e.self_device_time_total for e in evs) / 1e6
```

Keep the `device_type` filter. `key_averages()` lists both the `aten::` op and its kernels, so without
the filter the sum counts every kernel twice.

Still record `cpu%` and host load average with a benchmark. Some families depend on the host CPU.

## Keep an independent implementation

Every fused kernel has a slower implementation to compare against:

- `raster2d.render_torch` specifies the 2-D rasterizer.
- The per-primitive path in `gfxdraw/__init__.py` specifies `gfxdraw/fused.py`.
- `tests/gpu/test_conformance.py` compares both with the official renderer.
