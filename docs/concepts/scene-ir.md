# The Scene2D IR

`Scene2D` is the interface between a 2-D environment and the rasterizer. The env turns
`(state, variation)` into a `Scene2D`. One kernel turns it into pixels. No env handle, simulator or
Python callback crosses this boundary.

The IR has three primitive kinds: discs, convex polygons and thick segments. The rasterizer composites
them back to front with flat colours.

## Layout

The IR packs primitives into fixed-width tensors:

| field | shape | meaning |
|---|---|---|
| `kind` | `(B,N)` int32 | `0` disabled, `1` disc, `2` convex polygon, `3` segment |
| `geom` | `(B,N,2V)` f32 | disc: `[cx, cy, r, inner_r, ...]`. Polygon: `[x0,y0, ... xV-1,yV-1]` CCW. Segment: `[x0,y0, x1,y1, half_w, ...]` |
| `fill` | `(B,N,3)` f32 | interior colour, 0–255 |
| `line` | `(B,N,3)` f32 | outline colour (also the disc's inner colour) |
| `line_w` | `(B,N)` f32 | outline half-width in world units. `0` disables it. |
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
- **Padding repeats the last vertex of a polygon.** This creates a zero-length edge. Both backends
  drop edges with `L2 < 1e-12` from the signed distance, so the outline stays correct.

## Static prefix

`SceneBuilder.mark_static()` requests a bake. The rasterizer draws the primitives added so far once
into a cached base image. Later frames start from that image.

The bake is a request. The whole batch shares the base image, so the rasterizer bakes only if the
prefix is identical across lanes. It compares each row's static primitives with row 0 in one kernel.
If they differ, it draws every primitive per frame. The output is the same either way.

The cache key holds the exact bytes of the static primitives and the background colour. It also
holds the supersampled resolution, `world`, `flip_y`, the vertex count, the static prefix length and
the device. A Variation that recolours a static primitive or the background invalidates the bake.

The rasterizer skips the bake when the saved work is small. The threshold is `raster2d._BAKE_MIN_WORK = 8e6`
evaluations, compared against `K/N × B × H × W` (K static primitives out of N, at the supersampled
resolution). Below it, no comparison or transfer happens.

An env that passes its `StyleTensor` to the builder (`SceneBuilder(B, ..., style=sty)`) also skips the
comparison when the style differs per lane.

These measurements use one H200, supersample 1 and a pre-built scene (rasterization only). Each cell
is the speedup of baked over unbaked. A dash means the bake is below the threshold and does not run.

| batch | PushT 96 px | PushT 224 px | SimplePointMaze 500 px |
|---|---|---|---|
| static / total primitives | 6 / 9 | 6 / 9 | 6 / 13 |
| 256 | — | 1.33× | 1.65× |
| 1,024 | — | 1.92× | 1.72× |
| 4,096 | 1.81× | 2.18× | 1.74× |
| 16,384 | 2.15× | 2.26× | over the output budget |

The PushT columns use the internal Scene2D PushT scene (`stable_envs/envs/pusht/legacy.py`). The public
PushT classes draw the pygame frames with their own Numba (CPU) and Triton (CUDA) rasterizers.

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
`Scene2D.validate()` once. It checks the shape of each tensor, the `int32` dtype of `kind`, and that
the `world` extent is not degenerate.

## World coordinates

Each env keeps its own units: PushT's 512-unit board, CartPole's metres, a grid of tiles. The
rasterizer maps `world` onto the output image. Envs do not convert to pixels themselves. Set `flip_y`
instead of flipping coordinates in env code.

See [`stable_envs.render.scene`](../api/scene.md) for API details.
