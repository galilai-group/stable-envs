# The fused rasterizer

One Triton kernel turns any `Scene2D` into `(B,H,W,3)` uint8. It is the engine the repo stands on, and
the only piece of code that has to be fast.

All numbers on this page were measured on one NVIDIA H200, torch 2.11.0+cu130, triton 3.6.0, PushT at
96px unless stated otherwise. The GB/s table comes from `examples/bench_gpu_vs_cpu.py`; the
backend, supersample and bake microbenchmarks come from a separate allocation, so compare ratios across
tables rather than absolute values.

## Why one kernel and not a library of them

Every colour in a flat 2-D scene is a closed-form function of a handful of numbers per primitive. That
means an entire supersample block for one output pixel can be evaluated in registers and reduced before
anything is written, and the kernel never needs a canvas.

The alternative — compositing with tensor ops — allocates a `(B,3,R,R)` float canvas and touches it once
per primitive. That is `N` passes over a buffer that is `4/3 × N` times larger than the answer, and it is
memory traffic all the way down. The torch reference implementation in this repo does exactly that, which
makes the cost of the approach measurable rather than asserted:

| batch | torch compositing | fused Triton | ratio |
|---|---|---|---|
| 64 | 3,870 frames/s | 327,850 frames/s | **84.7×** |
| 256 | 15,346 frames/s | 1,121,346 frames/s | **73.1×** |

Both draw the same picture: on a 16-frame batch the two backends agree on **100.00%** of pixels, max
absolute difference **0**. That agreement is the reason the slow path is kept rather than deleted — it is
short enough to read as a *definition* of what the picture should be, it runs where Triton does not, and
the conformance suite diffs the two on every env. A fast kernel with no independent implementation to
check it against is a kernel nobody can trust.

## Where the memory traffic goes

The kernel writes the output and reads a small per-primitive parameter block. No canvas, no intermediate
tensor. That puts the ceiling at output bandwidth, which is the correct place to be bound, and the
measured curve says it gets there:

| batch | 96px, GB/s | 224px, GB/s |
|---|---|---|
| 256 | 3.0 | 15.2 |
| 1,024 | 11.4 | 47.4 |
| 4,096 | 37.6 | 101.4 |
| 16,384 | 89.1 | 141.8 |
| 65,536 | 131.9 | 155.2 |

Those are whole-`render()` figures, so they include building the scene. The rasterizer on its own reaches
**155.0 GB/s** at 96px and **162.1 GB/s** at 224px.

At 96px a frame is 27 KB, so even 16,384 of them is only 0.45 GB of output — small enough that per-launch
overhead is still visible. The 224px column saturates earlier for the same reason it costs more: more
bytes per launch.

## Anti-aliasing is supersampling, and the default is 1

`ss × ss` subpixels per output pixel, box-averaged. `ss=1` is the default because that is what the
evidence supports, not because it is cheapest.

PushT against the official `gym_pusht` renderer, 64 states from the env's own reset distribution:

| | interior pixels exact | all pixels exact | mean abs. diff |
|---|---|---|---|
| 96px, ss=1 | **100.00%** | 90.70% | 2.47 / 255 |
| 96px, ss=2 | **100.00%** | 93.79% | 1.35 / 255 |
| 224px, ss=1 | 99.9985% | 95.60% | 1.19 / 255 |
| 224px, ss=2 | 99.9985% | 97.04% | 0.57 / 255 |

Supersampling buys edge treatment and nothing else — the interior column does not move, because interior
pixels have no coverage ambiguity to resolve. And at an edge neither renderer is ground truth: gym_pusht
draws at 512 and `cv2.resize`s down, so its edge pixels are its own resampling filter, not the true
coverage. Paying for `ss=2` buys agreement with an artifact.

What it costs, measured through the full `render()` call at batch 4096:

| supersample | frames/s | µs/frame | cost vs ss=1 |
|---|---|---|---|
| 1 | 1,178,952 | 0.848 | — |
| 2 | 744,591 | 1.343 | 1.58× |
| 3 | 441,990 | 2.262 | 2.67× |

Those are end-to-end `render()` calls at batch 4,096, which include a supersample-independent ~2 ms of
scene construction; the kernel-only ratio is steeper.

## Two bugs the kernel's shape encodes

**64-bit output offsets.** The frame base is `b * H * W * 3`, which overflows int32 well inside the batch
sizes this repo exists to serve: at 224px, `b = 16383` gives `16383 × 50176 × 3 = 2.47e9` against an
int32 ceiling of 2.15e9. It surfaced as `CUDA error: an illegal memory access` — a crash rather than
silent corruption — but only because the wrapped address happened to land outside the allocation. A
smaller overflow would have quietly written into another tensor. The offset is computed in int64.

**Outlines are capsules, not an SDF band.** Taking the outline as the band where `|sdf| <= w` mitres the
corners of a polygon. pygame's `draw_fat_segment` explicitly paints a disc of the stroke radius at each
endpoint, so the reference has *round* corners. Measured on PushT, the band costs 2.2% of interior pixels
— 97.8% exact instead of 100% — all of it at the eight corners of the T's two quads. The kernel takes the
union of per-edge capsules instead.

## The two entry points

```python
from stable_envs.render.raster2d import render, render_triton, render_torch

img = render(scene, out_size=96, supersample=1, backend="auto")   # Triton, torch fallback
```

`backend="auto"` prefers Triton and falls back to torch when it is unavailable. `Env.render` also
accepts an `out=` tensor, which is what the benchmarks use: rendering into a pre-allocated
`(B,H,W,3)` uint8 buffer removes the allocation from the timed region, and in a training loop it removes
it from the steady state.

## What `render()` costs beyond the kernel

`Env.render` does two things: build a `Scene2D` from the states, then rasterize it. Those halves
have different cost models and the split matters, because at small batch the total is dominated by the
half that is not the kernel. `scene()` is a few dozen small torch ops issued from Python, so its cost is
launch-overhead-bound and nearly independent of the batch, while the rasterizer is one launch that scales
with output pixels. Below roughly a thousand environments you are timing Python, not the GPU.

API details: [`stable_envs.render.raster2d`](../api/raster2d.md).
