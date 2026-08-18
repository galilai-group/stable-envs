# Writing a bit-exact Triton kernel

Every renderer in this repo is measured against a reference implementation, and most are held to
**zero differing pixels**, not to a tolerance. That bar changes what "correct floating point" means.
A kernel that is accurate to a few ulps is fine for a loss function and useless here: a rasterizer
turns a float comparison into a branch, so one ulp on the wrong side of a boundary is a whole pixel,
and a resampler turns a float into an integer, so one ulp is a whole level.

Triton's defaults are tuned for the first kind of code. This page is the list of places where they
are wrong for the second kind, each of which has now cost this project a real bug.

## 1. Division is not correctly rounded

`a / b` on `float32` lowers to PTX `div.full.f32`, which is fast and **not** IEEE round-to-nearest.
`tl.sqrt` lowers to `sqrt.approx.f32`. Both are accurate to ~1 ulp, which is invisible until
something *rounds* the result — and a renderer rounds everything.

!!! danger "Measured twice, from opposite ends of the codebase"

    **`render/gfxdraw_fused.py`** reimplements `pygame.draw.polygon`'s scanline rule, which divides
    in float32 and then takes `floor` or `ceil` of the quotient. On one CarRacing wheel the exact
    quotient is `-42.0 / 14.0 = -3.0`. The approximate reciprocal returned a hair below `-3`, `floor`
    gave `-4`, the scanline crossing moved a whole pixel and the polygon lost a boundary pixel.
    Frequency: **1 pixel in 19.2 million** — it fires only where the geometry makes the quotient
    exactly an integer, which is why a spot check will not find it and a 48-state parity run will.

    **`render/raycast3d.py`**, fusing the 3-D ray-cast, measured **1,249,701 differing results over
    4.19 M inputs** for `a / b` and **709,903** for `sqrt(x)`. There one ulp in `dl` amplifies to
    2.4e-4 in `disc = b*b - a*c`, and `disc > 0` is the hit/miss decision.

Use `tl.math.div_rn(a, b)` and `tl.sqrt_rn(x)`. They are the IEEE round-to-nearest forms and cost a
few extra instructions on a path that is almost never the bottleneck. In `raycast3d` those two
substitutions plus rule 2 took all four mismatch counts to exactly **zero**.

## 2. Multiply-add contraction is not stable across compilations

ptxas is free to contract `a*b + c*d` into an FMA, and whether it does depends on the surrounding
code — which means it can depend on a `constexpr` you changed for reasons that have nothing to do
with arithmetic. Two specializations of *one source expression* can then round differently.

!!! example "Measured on `raster2d`"

    The polygon inside test is `max` over the outward edge planes `<= 0`. Over intersection-v0 and
    racetrack-v0 at 600 px, batch 1024 (368 M pixels), the tiled and untiled specializations of the
    same kernel disagreed on 3 and 1 pixels — every one of them a pixel whose centre lies within
    ~3e-5 world units of the edge that decides the sign. The tell that it was contraction and not a
    logic error: the untiled path disagreed **with itself** at two different block sizes on exactly
    those pixels, while two different tile shapes of the tiled path agreed exactly.

It also changes the result relative to a **torch** reference, for a different reason: torch evaluates
`a*b + c` as two kernels and rounds twice, where a contracted `fma.rn.f32` rounds once. So a kernel
whose specification is a torch expression must not contract either.

Two consequences:

* If a kernel must agree bit for bit with a torch reference, or with another *compilation* of itself,
  pass `enable_fp_fusion=False` to the launch — Triton's `-ffp-contract=off`. That is what
  `render/raycast3d.py` does, and it is why the multiply-adds there can be written normally.
* If it only has to agree with the reference *renderer*, leave fusion on and do not write a test that
  compares two specializations at a batch large enough to hit the coincidence. Say which one you
  chose and why, next to the kernel.

## 3. Integer division truncates, and the reference usually floors

`//` on Triton integers follows C: it truncates toward zero. `torch.div(..., rounding_mode="floor")`
and Python's `//` floor. They differ for negative numerators, which is exactly what a scanline fill
produces for a polygon hanging off the top of the surface. Either prove the numerator is
non-negative wherever the result is used — `raster2d`'s 16.16 fill does, because the crossing is only
read on rows the edge is active for — or write the floor out.

## 4. Width is part of the specification

`gfxdraw`'s blend runs on SDL's packed 32-bit channel word and *relies* on the unsigned wrap: it is
what turns a truncating `>>8` into a ceiling when the source is darker than the destination. Doing
that arithmetic in int64 and masking with `0xFFFFFFFF` reproduces it; doing it in int32 and hoping
does not, because the intermediate `diff * alpha` reaches 1.1e12.

The converse is also a rule: **do not widen for comfort.** An int64 accumulator is two live tile
registers instead of one, and the fused CarRacing kernel spilled its three colour accumulators until
they were narrowed to int32 with the widening pushed into the one expression that needs it — 27.7 ms
to 12.4 ms on an 8x128 tile.

## 5. Check on the workload, not on the smallest instance

Not a floating-point rule, but the failure that has cost this project the most, twice, and it lands
on the same kernels:

* **The unrolled primitive loop.** `tl.static_range` measured 7.6% *faster* on PushT's 9 primitives
  and 4x *slower* on highway-env's 134 — and above ~340 primitives ptxas will not compile it at all.
  See `render/raster2d.py`.
* **`scene()`'s cost.** Recorded as "~2.0 ms of Python-issued small torch ops, flat regardless of
  batch", measured on the same 9 primitives. On highway-env, which builds 157 to 436, it was
  **100–315 ms** and the dominant cost of four of the six envs — still flat in the batch, so still
  pure dispatch, but two orders of magnitude larger. The fix was to stop issuing one call per
  primitive, and it was invisible for as long as the benchmark was PushT.

A micro-benchmark on the smallest instance of a workload can recommend the exact opposite of what
the workload wants. Quote the scene the number came from, every time.

## 6. Metric equality is not identity

An aggregate over a batch of frames cannot see a small number of badly wrong pixels, and the
aggregates this repo declares are means over 64 frames. So "the metrics did not move" is not evidence
that the picture did not move, and the two have been observed to disagree by 112/255.

Measured on the five DM Control TrackCom cells, comparing `enable_backface_culling=False` against the
shipped context. The frontier sweep reported **all four fidelity metrics unchanged to four decimal
places on every cell** — mean |diff| moved by less than 0.005, within-8 and SSIM by less than 0.0001.
A byte comparison over 4 draws x 64 states says something else:

| cell | 84 px | 256 px |
|---|---:|---:|
| cheetah-run | **0/255** | **0/255** |
| walker-walk | 0/255 | 92/255 |
| hopper-hop | 0/255 | 76/255 |
| humanoid-stand | 68/255 | 112/255 |
| swimmer6 | 95/255 | 109/255 |

The flag is free on one cell of five, and the aggregate said it was free on all five. It also says
*where* to look: the disagreement appears at 256 px and not at 84 px on two of the cells, so a check
at one resolution would have passed as well.

The rule: **when the claim is "this changes nothing", assert bytes.** Keep the metric comparison for
claims that are about fidelity — "this is no worse than JPEG q95" is a metric claim and is measured in
metrics — but a claim of identity is measured in identity, at every size it will be relied on.

## 7. `cpu%` cannot tell you where the time goes

PyTorch's default synchronisation **spins**. A render that is entirely GPU-bound and a render that is
entirely interpreter-bound both report 100% of a core, so process CPU time is useless as an indicator
of *which* — and it has been read as evidence of host-boundness on this repo's ray-cast families,
where the correct answer turned out to be 40–83% GPU.

The instrument that does work is the profiler's **device-side kernel time against the wall clock**:

```python
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
    ...
evs = [e for e in pr.key_averages() if e.device_type == DeviceType.CUDA]
kernel_seconds = sum(e.self_device_time_total for e in evs) / 1e6
```

The `device_type` filter is load-bearing. `key_averages()` lists both the `aten::` op and the kernels
it launched, and the op's `self_device_time_total` is the sum of its children's — so summing
everything counts every kernel twice. The first version of `scratch/perf3d/residual.py` did exactly
that and reported 150% of wall time, i.e. a *negative* interpreter share, which is at least an error
that announces itself.

`cpu%` is still worth recording, for a different reason: these families are host-sensitive, and the
same env has measured 3,664 f/s on an idle node and 271 f/s on a contended one. Report it, and the
node's load average, so a reader can see whether a number is a property of the renderer or of the
cluster — but read the split from the kernel share, not from `cpu%`.

## What to do about all of this

The rule that actually catches these is not a coding convention, it is that **every fused kernel
keeps an independent implementation to be diffed against**:

* `raster2d.render_torch` is the executable specification for the 2-D rasterizer;
* `gfxdraw.py`'s per-primitive path is the specification for `gfxdraw_fused.py`;
* and both are diffed against the *official* renderer in `tests/gpu/test_conformance.py`.

Every bug on this page was found by a diff, none by reading the code — including the ones whose
mechanism is obvious in hindsight. A fast kernel with no independent implementation to check it
against is a kernel nobody can trust.
