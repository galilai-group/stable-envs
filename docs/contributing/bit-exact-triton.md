# Bit-exact Triton kernels

Most renderers here must match a reference with zero differing pixels. A rasterizer turns a float comparison into a branch, and a resampler turns a float into an integer. One ulp on the wrong side is then a whole pixel or a whole color level. Triton's defaults for division, square roots, FMA contraction and integer math aren't built for this, and each section below names what to use instead.

## Division and square root aren't correctly rounded

`a / b` on float32 lowers to PTX `div.full.f32`, and `tl.sqrt` lowers to `sqrt.approx.f32`. Both are accurate to about 1 ulp, short of IEEE round-to-nearest, and the error shows when the kernel rounds or compares the result:

- `render/gfxdraw/fused.py` takes the `floor` of a float32 quotient. With `div.full.f32`, the exact quotient `-42.0 / 14.0 = -3.0` can come out slightly below `-3`, which moves a scanline by one pixel.
- `render/raycast3d.py` uses `disc > 0` as the hit test, where a 1-ulp error in the inputs changes hits.

Use `tl.math.div_rn(a, b)` and `tl.sqrt_rn(x)` instead.

## Multiply-add contraction isn't stable across compilations

ptxas may contract `a*b + c*d` into an FMA, and whether it does depends on the surrounding code, including `constexpr` values. Two specializations of one expression can then round differently. A contracted `fma.rn.f32` also rounds once, where torch rounds `a*b + c` twice.

If the kernel must match a torch reference or another compilation of itself bit for bit, pass `enable_fp_fusion=False` to the launch, as `render/raycast3d.py` does. If it only has to match the upstream renderer, fusion can stay on, and no test should then compare two specializations at a large batch. Either way, say which choice you made next to the kernel.

## Integer division truncates

`//` on Triton integers truncates toward zero, like C, but Python's `//` and `torch.div(..., rounding_mode="floor")` round down. They differ for negative numerators, which a scanline fill produces for polygons above the top edge. Prove the numerator is non-negative where the kernel uses the result, or write the floor out explicitly.

On CPU, torch shows both roundings:

```python
import torch

numerator = torch.tensor([-7, 7], dtype=torch.int32)
print(torch.div(numerator, 2, rounding_mode="trunc"))  # tensor([-3,  3], dtype=torch.int32), like Triton's //
print(torch.div(numerator, 2, rounding_mode="floor"))  # tensor([-4,  3], dtype=torch.int32), like Python's //
```

## Integer width is part of the specification

`gfxdraw`'s blend works on SDL's packed 32-bit channel word and relies on unsigned wrap-around. Doing the arithmetic in int64 and masking with `0xFFFFFFFF` reproduces it. In int32, `diff * alpha` overflows: it reaches 1.1e12.

Widen only for a reason. An int64 accumulator uses two registers instead of one and can cause spills. Keep accumulators in int32 and widen just the expression that needs it.

## Benchmark on the real workload

A result on the smallest scene can point the wrong way on a large one. `tl.static_range` helps with a few primitives, for example, but it's slower with about a hundred. Always say which scene a number came from, with `cpu%` and the host load average next to it: some families depend on the host CPU.

## Equal metrics don't mean equal images

A mean over 64 [frames](../terms.md#frame) hides a few badly wrong pixels. Before you claim an edit leaves the output unchanged, compare bytes at every resolution you rely on. Save metrics for fidelity comparisons.

## Keep an independent implementation

Every fused kernel has a slower implementation to compare against:

- `raster2d.render_torch` specifies the 2-D rasterizer.
- The per-primitive path in `gfxdraw/__init__.py` specifies `gfxdraw/fused.py`.
- `tests/gpu/test_conformance.py` compares both with the upstream renderer (see [GPU conformance](testing.md#gpu-conformance)).
