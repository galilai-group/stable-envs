# The Scene2D IR

`Scene2D` is the interface between a 2-D environment and the rasterizer. The env turns
`(state, variation)` into a `Scene2D`. One kernel turns it into pixels. No env handle, simulator or
Python callback crosses this boundary.

The IR has three primitive kinds: discs, convex polygons and thick segments. They are composited back
to front with flat colours.

## Layout

Primitives are packed into fixed-width tensors:

| field | shape | meaning |
|---|---|---|
| `kind` | `(B,N)` int32 | `0` disabled, `1` disc, `2` convex polygon, `3` segment |
| `geom` | `(B,N,2V)` f32 | disc `[cx, cy, r, inner_r, ...]`; polygon `[x0,y0, ... xV-1,yV-1]` CCW; segment `[x0,y0, x1,y1, half_w, ...]` |
| `fill` | `(B,N,3)` f32 | interior colour, 0–255 |
| `line` | `(B,N,3)` f32 | outline colour (also the disc's inner colour) |
| `line_w` | `(B,N)` f32 | outline half-width in world units; `0` disables it |
| `background` | `(B,3)` f32 | per-environment background colour |
| `world` | 4 floats | `(x0, y0, x1, y1)`, the extent of the primitives |
| `flip_y` | bool | set when the env's `+y` points up |
| `static_prefix` | int | number of leading primitives the env requests to bake |

The leading dimension is the batch. Geometry and colour are per-environment tensors, so each lane can
have its own colours and sizes in one launch.

## Invariants

- **`kind == 0` disables a slot.** A batch is one rectangular tensor. A scene whose primitive count
  depends on the state disables slots instead of changing shape. Static shapes allow CUDA graph capture
  and compile-time unrolling.
- **Draw order is index order, back to front.** There is no depth buffer. The array order decides what
  is on top.
- **Polygons are padded by repeating the last vertex.** This creates a zero-length edge. Both backends
  drop edges with `L2 < 1e-12` from the signed distance, so the outline stays correct.

## Static prefix

`SceneBuilder.mark_static()` requests that the primitives added so far be baked once into a cached
base image. Later frames start from that image.

The bake is a request. The base image is shared by the whole batch, so the rasterizer bakes only if the
prefix is identical across lanes. It compares each row's static primitives with row 0 in one kernel.
If they differ, it draws every primitive per frame. The output is the same either way.

The cache key is the exact bytes of the static primitives. A Variation that recolours a static
primitive invalidates the bake.

Baking is skipped when the saved work is small. The threshold is `raster2d._BAKE_MIN_WORK = 8e6`
evaluations, compared against `K/N × B × H × W` (K static primitives out of N, at the supersampled
resolution). Below it, no comparison or transfer happens.

An env that passes its `StyleTensor` to the builder (`SceneBuilder(B, ..., style=sty)`) also skips the
comparison when the style differs per lane.

Measured on one H200, supersample 1, from a pre-built scene (rasterization only). Speedup of baked over
unbaked; a dash means the bake is below the threshold and not run:

| batch | PushT 96 px | PushT 224 px | SimplePointMaze 500 px |
|---|---|---|---|
| static / total primitives | 6 / 9 | 6 / 9 | 6 / 13 |
| 256 | — | 1.33× | 1.65× |
| 1,024 | — | 1.92× | 1.72× |
| 4,096 | 1.81× | 2.18× | 1.74× |
| 16,384 | 2.15× | 2.26× | over the output budget |

The PushT columns use the internal Scene2D PushT scene (`stable_envs/envs/pusht/legacy.py`). The public
PushT classes render with pygame.

## Building a scene

Add primitives in draw order. Each argument takes a scalar (shared by the batch) or a `(B, ...)`
tensor (one value per lane):

```python
from stable_envs.render.scene import SceneBuilder

sb = SceneBuilder(batch=B, device="cuda", vertices=4, style=sty)   # style= is optional
sb.add_poly(goal_verts, goal_color)                     # (B,4,2) world vertices
sb.add_segment((5, 506), (5, 5), 2.5, wall_color)
sb.mark_static()                                        # request a bake of everything above
sb.add_disc(state[:, 0:2], 15.0, agent_color, inner_radius=11.0, inner_color=agent_inner)
sb.add_poly(block_verts, block_fill, outline_color=block_line, outline_w=2.5)
scene = sb.build(background=bg, world=(0, 0, 512, 512), flip_y=False)
```

`add_rect` wraps `add_poly` for rectangles, with optional rotation. `build()` runs
`Scene2D.validate()`, which checks shapes and dtypes once.

## World coordinates

Each env keeps its own units: PushT's 512-unit board, CartPole's metres, a grid of tiles. The
rasterizer maps `world` onto the output image. Envs do not convert to pixels themselves. Set `flip_y`
instead of flipping coordinates in env code.

API details: [`stable_envs.render.scene`](../api/scene.md).
