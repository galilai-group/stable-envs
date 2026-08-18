# The Scene2D IR

`Scene2D` is the whole interface between an environment and the renderer. An env turns
`(state, variation)` into one, and exactly one kernel turns it into pixels. Nothing else crosses the
line: no env handle, no simulator, no callback into Python during rasterization.

## Why an IR at all

The prototype this repo grew out of was a Triton kernel written *for PushT*. It knew there was a T-block
made of two quads and an agent that was a disc, and it was fast because it knew. That does not survive
the second environment, and it certainly does not survive a zoo.

What generalises is a much duller observation: the 2-D environments the field trains on draw from a tiny
vocabulary. CartPole is two polygons and a disc. Pendulum is a polygon and two discs. A MiniGrid tile is
a rectangle. PushT is two quads for the goal, four thick segments for the walls, a disc for the agent and
two more quads for the block — nine primitives, three kinds. So the kernel should not know about T-blocks;
it should know about convex polygons, discs and thick segments, composited back to front with flat
colours. That is the entire IR.

## The layout

Primitives are packed into fixed-width tensors so the kernel does flat, predictable loads:

| field | shape | meaning |
|---|---|---|
| `kind` | `(B,N)` int32 | `0` disabled, `1` disc, `2` convex polygon, `3` segment |
| `geom` | `(B,N,2V)` f32 | disc `[cx, cy, r, inner_r, ...]`; poly `[x0,y0, ... xV-1,yV-1]` CCW; segment `[x0,y0, x1,y1, half_w, ...]` |
| `fill` | `(B,N,3)` f32 | interior colour, 0–255 |
| `line` | `(B,N,3)` f32 | outline colour (also the disc's inner colour) |
| `line_w` | `(B,N)` f32 | outline half-width in world units; `0` disables it |
| `background` | `(B,3)` f32 | per-environment background colour |
| `world` | 4 floats | `(x0, y0, x1, y1)`, the extent the primitives live in |
| `flip_y` | bool | set when the env's `+y` points up and the image's points down |
| `static_prefix` | int | how many leading primitives the env believes are batch- and call-invariant |

The leading dimension is the batch, and that is the point rather than a convenience. Geometry *and*
colour are per-environment tensors, so 16,384 environments can each carry a different block colour or a
different agent radius and still rasterize in one launch. Domain randomisation stops being a for-loop
over envs and becomes a tensor you already have. (What is actually per-environment today is narrower than
what the layout permits — see [Variation spaces](variation-spaces.md) for the measured list.)

## Three details that are load-bearing

**`kind == 0` is a feature, not padding hygiene.** A batch is one rectangular tensor, so an env whose
primitive count depends on its state — a maze with a variable number of walls, a goal marker switched off
by `variation['rendering']['render_goal']` — expresses that by *disabling slots*, never by changing shape.
Static shapes are what make the whole step capturable in a CUDA graph, and they are what let the kernel
unroll its primitive loop at compile time.

**Draw order is index order, back to front.** Painter's algorithm over flat colours has no depth to
resolve, so ordering the array *is* the z-buffer. That makes the order explicit and diffable, which is
worth more than it sounds: in PushT, whether the agent or the block paints last is documented nowhere —
`pymunk.debug_draw` walks `space.shapes`, a dict-iteration detail — and had to be recovered by
pixel-diffing both orders against the reference. An IR where that decision is a line number is an IR
where the answer can be recorded.

**Polygons are padded by repeating the last vertex.** A four-vertex builder holding a triangle stores the
third vertex twice, which creates a zero-length edge. That edge is harmless for the inside test (its
cross product is zero, which passes) but its normal is undefined, and a naive outline would take the
distance to it as zero everywhere and paint the entire plane. Both backends detect `L2 < 1e-12` and drop
the edge from the SDF.

## Baking the static prefix

Most scenes are mostly static. PushT's goal T and four walls are six of its nine primitives and depend
only on the variation, never on the state. `SceneBuilder.mark_static()` *requests* that prefix; the
rasterizer renders it once into a cached base image at full supersample resolution and starts every
subsequent frame from that image instead of from the background colour.

A request rather than a promise, because the precondition is not something an env can assert. The base
image is one image for the whole batch, so baking a prefix that differs per environment would paint
every environment with environment 0's goal, walls and background — the second of the
[three collapses](../api/style.md) this repo has shipped. So the rasterizer *measures*: it compares the
prefix against its own first row, bakes when they agree and draws every primitive per frame when they do
not. Asking for a prefix that is not shared costs the speedup and never the picture, which is what lets
PushT declare one and still render 4,096 different goal colours correctly.

Measured on an H200, supersample 1, on the code as it ships — which matters, because before the fix in
the warning at the end of this section the bake never ran above batch 1 at all, so none of these cells
existed. From a pre-built scene, so the comparison is rasterization only:

| | | PushT 96px | PushT 224px | SimplePointMaze 500px |
|---|---|---|---|---|
| | *static / total primitives* | 6 / 9 | 6 / 9 | 6 / 13 |
| batch 64 | | — | — | — |
| batch 256 | | — | **1.33×** | **1.65×** |
| batch 1,024 | | — | **1.92×** | **1.72×** |
| batch 4,096 | | **1.81×** | **2.18×** | **1.74×** |
| batch 16,384 | | **2.15×** | **2.26×** | (over the output budget) |

End to end through `env.render()`, where `scene()`'s flat ~2.2 ms of Python sits in both columns, the same
cells are 1.24× / 1.68× (96px at 4,096 / 16,384), 1.04× / 1.32× / 1.79× / 2.11× (224px at 256 / 1,024 /
4,096 / 16,384) and 1.32× / 1.56× / 1.69× (SimplePointMaze at 256 / 1,024 / 4,096). Baked and unbaked
output is **byte-identical** at every cell — the bake substitutes for re-evaluating those primitives
exactly, because it is rasterized at the same subpixel resolution the kernel would have used.

The dashes are the point of the next section: they are cells where the bake is *not run*, and the
measured speedup there is 0.99–1.00×, i.e. nothing gained and nothing paid.

### The measurement is the cache key, and there is a floor below which neither is worth doing

The cache is keyed on the static primitives' exact bytes, not on the env object. That is deliberate: a
variation that recolours the goal must invalidate the bake, and an identity keyed on the env would go on
serving the old picture forever.

Those bytes are also what decides whether the prefix is shared, so both questions are answered by one
device-to-host transfer: the six arrays are reinterpreted as raw bytes, concatenated into one `(B, M)`
block, compared against row 0 in a single kernel, and row 0 plus the one-byte verdict are copied to the
host together. That costs **0.083 ms** at batch 1, where there are no rows to compare and only the key is
fetched, and **0.21–0.29 ms** at any larger batch — flat from 64 to 16,384, because it is a
synchronisation rather than a quantity of data.

A flat cost against a saving that grows with the work skipped has a crossover, and the crossover is
measurable rather than a matter of taste. The bake saves about `K/N` of the rasterizer's evaluations,
i.e. `K/N × B × Rh × Rw` of them, and all three cells above break even between **6e6 and 9e6** of them —
a single number that predicts the switch for a 9,216-pixel frame and a 250,000-pixel one, 27× apart. So
`raster2d._BAKE_MIN_WORK` is 8e6, and below it the request is dropped *before anything synchronises*: a
small batch pays nothing at all for a prefix it cannot use, and a loop of single renders — the worst case
for any per-call overhead — is the furthest below the line.

An env that hands its `StyleTensor` to the builder (`SceneBuilder(B, ..., style=sty)`) also skips the
transfer on the declining path, above the floor: a static prefix is a function of the variation alone, so
a per-environment appearance means a per-environment prefix, and `StyleTensor.shared` is a cached bool.
The two envs that ship a prefix today do not pass it, and 0.21–0.29 ms per render is what that costs them
when a caller randomises the appearance per environment.

!!! warning "Detection by stride was wrong, and the bake was dead code because of it"

    Until this release the builder decided per-environment-ness from `stride(0) != 0` on the incoming
    values, and refused the whole prefix when it fired. A `(B, w)` slice of a style tensor has a non-zero
    row stride whether or not its rows differ, and `add_rect` hands `add_poly` freshly computed vertices
    whatever it was given — so **every** prefix was called per-environment as soon as `B > 1`, and the
    bake never ran outside batch 1 in a repo whose subject is batches of thousands. The number this page
    used to quote for it (1.83–2.13×, and ~5× in an earlier draft) was measured, but not on a path any
    batched caller reached. Sharedness is a property of the values, so it is now decided from the values.

## Building one

Envs build declaratively, in draw order. Every argument takes either a scalar (shared by the batch) or a
`(B, ...)` tensor (one value per environment), and the builder broadcasts, which is what makes
per-environment appearance free at the call site:

```python
from stable_envs import SceneBuilder

sb = SceneBuilder(batch=B, device="cuda", vertices=4, style=sty)   # style= is optional
sb.add_poly(goal_verts, goal_color)                     # (B,4,2) world vertices
sb.add_segment((5, 506), (5, 5), 2.5, wall_color)
sb.mark_static()                                        # bake everything above, if it is shared
sb.add_disc(state[:, 0:2], 15.0, agent_color, inner_radius=11.0, inner_color=agent_inner)
sb.add_poly(block_verts, block_fill, outline_color=block_line, outline_w=2.5)
scene = sb.build(background=bg, world=(0, 0, 512, 512), flip_y=False)
```

`add_rect` is a convenience over `add_poly` for the most common case, with an optional rotation.
`build()` runs `Scene2D.validate()`, which checks the shape and dtype contract once rather than letting a
transposed tensor become a silently wrong picture.

## World coordinates

An env keeps its own units — PushT's 512-unit board, CartPole's metres, MiniGrid's tile grid — and the
rasterizer maps `world` onto the output image. No env converts to pixels by hand. That conversion is
trivial and everyone gets it wrong eventually; centralising it is what stops a y-flip from being
re-derived, incorrectly, in every new env. `flip_y` exists for the same reason: it is one flag on the
scene rather than a sign convention smeared through an env's geometry code.

API details: [`stable_envs.render.scene`](../api/scene.md).
