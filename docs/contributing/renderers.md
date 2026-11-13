# Renderers

A family draws [frames](../terms.md#frame) either with its [upstream library](../terms.md#upstream-library)'s code or with one of the shared renderers in `stable_envs.render`. Each class table on a family page names the renderer it uses on CPU and on CUDA. Each section below takes one shared renderer and gives the families that use it and what it promises about backends, exactness and cost. [The Scene2D IR](#the-scene2d-ir) gives the tensor layout of a 2D scene, and [The compiled variation](#the-compiled-variation) covers `StyleTensor`, the form of a variation that renderers read.

| Module | Draws | Used by | Devices |
|---|---|---|---|
| `scene`, `raster2d` | discs, convex polygons and capsules | `SimplePointMaze`; the navigation dot renderers reuse its tiling | Torch on CPU, Triton on CUDA |
| `gfxdraw` | SDL_gfx primitives | classic control | Torch or Numba on CPU, Triton on CUDA |
| `raycast3d` | MuJoCo primitive scenes | OGBench mazes, fixed-camera DM Control tasks | Torch and Numba on CPU, Torch and Triton on CUDA |
| `cpu3d` | MuJoCo primitive scenes with a floor and a tracking camera | DM Control body-tracking tasks on CPU | Torch and Numba, CPU only |
| `nvdiff` | textured, lit triangle meshes | `PyFlytRocketLanding` | CUDA only |
| `mujoco_raster` | MuJoCo scenes, as MuJoCo's OpenGL renderer draws them | Fetch, OGBench Cube and Scene (`backend="raster"`, the default) | Numba on CPU, nvdiffrast and Triton on CUDA |
| `mjwarp` | MuJoCo models | DM Control body-tracking tasks on CUDA, Fetch on CUDA without nvdiffrast, OGBench Cube and Scene at `supersample` 2 or more | CUDA only |

The other MuJoCo paths use [MuJoCo's OpenGL renderer](../install.md#what-each-family-needs). The [benchmarks](../benchmarks.md) measure the throughput of the ray caster and MJWarp.

## 2D scenes and rasterizer

`stable_envs.render.scene` defines `Scene2D`, a batch of discs, convex polygons and capsules in draw order, and `SceneBuilder`, which appends them. A family that draws in 2D returns a `Scene2D` from `Env.scene`, and `stable_envs.render.raster2d` draws it into `(B, H, W, 3)` uint8 frames.

```python
import torch
from stable_envs.render.raster2d import render
from stable_envs.render.scene import SceneBuilder

b = SceneBuilder(batch=2, device="cpu")
b.add_rect(center=(0.5, 0.5), size=(0.6, 0.2), color=(40, 90, 200))      # shared by both rows
b.add_disc(center=torch.tensor([[0.3, 0.5], [0.7, 0.5]]), radius=0.1,    # one center per row
           color=(220, 30, 30))
scene = b.build(background=(255, 255, 255), world=(0.0, 0.0, 1.0, 1.0))
print(scene.batch, scene.n_prims)  # 2 2
print(scene.kind.tolist())         # [[2, 1], [2, 1]]: a polygon, then a disc

frames = render(scene, out_size=64, supersample=2)
print(frames.shape, frames.dtype)                            # torch.Size([2, 64, 64, 3]) torch.uint8
print(frames[0, 32, 19].tolist(), frames[0, 2, 2].tolist())  # [220, 30, 30] [255, 255, 255]
```

Every `SceneBuilder` argument takes one value shared by the batch or one value per batch row. Coordinates are in the family's own world units, and `world=(x0, y0, x1, y1)` maps them onto the frame. Colors are RGB in 0 to 255. `enabled` turns a primitive off per row, and the scene keeps its shape when the number of visible objects changes. `build` raises `ValueError` when nothing was added. See [The Scene2D IR](#the-scene2d-ir) for the tensor layout and the static-prefix cache that `mark_static` requests.

Primitives are composited in the order they were added. A static background is baked only when its geometry and appearance are the same for the whole batch. Output offsets are int64, which keeps the address calculation of large batches from overflowing. Outlines are unions of per-edge capsules with round corners, as in pygame's `draw_fat_segment`.

`backend="auto"` uses Triton when it's installed and the scene is on CUDA, and Torch otherwise. `backend="triton"` with a CPU scene raises `ValueError`. The Triton kernel computes each pixel's color in registers and writes only the result, while the Torch backend composites on a `(B, 3, R, R)` float canvas, one pass per primitive. Tests compare their pixels exactly. Both take `out`, a contiguous `(B, H, W, 3)` uint8 buffer that `Env.render(out=...)` passes through, and a training loop can reuse one buffer for every call. `supersample=ss` averages `ss × ss` subpixels per output pixel, which changes only edge pixels.

Building a `Scene2D` from state issues a few dozen small Torch operations from Python, and its cost changes little with batch size. The kernel is one launch that scales with output pixels. Below about a thousand environments, building the scene takes most of the time.

::: stable_envs.render.scene
    options:
      show_root_heading: false
      members: [Scene2D, SceneBuilder]

::: stable_envs.render.raster2d
    options:
      show_root_heading: false
      members: [render, render_torch, render_triton, row_groups, tile_shape]

## The Scene2D IR

`Scene2D` is the interface between a 2D environment and the [rasterizer](#2d-scenes-and-rasterizer). The environment turns `(state, variation)` into a `Scene2D`, and one kernel turns that into pixels. The kernel sees only the scene, never an environment handle, simulator or Python callback.

```python
from stable_envs.render import raster2d
from stable_envs.render.scene import SceneBuilder

agent = torch.tensor([[256.0, 256.0], [100.0, 400.0]])  # one position per batch row

sb = SceneBuilder(batch=2, device="cpu", vertices=4)
sb.add_segment((5, 506), (5, 5), 2.5, (0, 0, 0))  # a wall, shared by the batch
sb.mark_static()                                  # request a bake of everything above
sb.add_disc(agent, 15.0, (65, 105, 225))          # (B, 2) centers, one per row
scene = sb.build(background=(255, 255, 255), world=(0, 0, 512, 512))

frames = raster2d.render_torch(scene, 96)
print(frames.shape, frames.dtype)  # torch.Size([2, 96, 96, 3]) torch.uint8
```

The IR has three primitive kinds, discs, convex polygons and capsules, composited back to front with flat colors and packed into fixed-width tensors:

| Field | Shape | Meaning |
|---|---|---|
| `kind` | `(B, N)` int32 | `0` disabled, `1` disc, `2` convex polygon, `3` capsule |
| `geom` | `(B, N, 2V)` f32 | disc: `[cx, cy, r, inner_r, ...]`. Polygon: `[x0, y0, ... xV-1, yV-1]` CCW. Capsule: `[x0, y0, x1, y1, half_w, ...]` |
| `fill` | `(B, N, 3)` f32 | interior color, 0–255 |
| `line` | `(B, N, 3)` f32 | outline color, also the disc's inner color |
| `line_w` | `(B, N)` f32 | outline half-width in world units, where `0` disables it |
| `background` | `(B, 3)` f32 | per-environment background color |
| `world` | 4 floats | `(x0, y0, x1, y1)`, the extent of the primitives |
| `flip_y` | bool | set when the environment's `+y` points up |
| `static_prefix` | int | number of leading primitives the environment asks to bake |

- `kind == 0` disables a slot. A batch is one rectangular tensor: a scene whose primitive count depends on the state disables slots instead of changing shape, which allows CUDA graph capture and compile-time unrolling.
- Draw order is index order, back to front, with no depth buffer.
- Padding repeats the last vertex of a polygon. That makes a zero-length edge, which both backends drop from the signed distance (edges with `L2 < 1e-12`), and the outline stays correct.

Each environment keeps its own units, such as PushT's 512-unit board or CartPole's meters, and the rasterizer maps `world` onto the output image. Set `flip_y` instead of flipping coordinates in environment code. `add_disc` takes an optional `inner_radius` and `inner_color`, `add_poly` takes `(B, V, 2)` world vertices with an optional `outline_color` and `outline_w`, and `add_rect` wraps `add_poly` with an optional rotation. `build()` runs `Scene2D.validate()` once, which checks each tensor's shape, the `int32` dtype of `kind`, and that `world` isn't degenerate.

### Static prefix

`SceneBuilder.mark_static()` asks the rasterizer to draw the primitives added so far once into a cached base image that later frames start from. The whole batch shares the base image. It bakes only if the prefix is identical across batch rows, which one kernel checks against row 0, and otherwise draws every primitive per frame. The output is the same either way.

The cache key holds the exact bytes of the static primitives and the background color, the supersampled resolution, `world`, `flip_y`, the vertex count, the prefix length and the device. A variation that recolors a static primitive or the background therefore invalidates the bake. The rasterizer skips the bake when `K/N × B × H × W` (K static primitives out of N, at the supersampled resolution) is below `raster2d._BAKE_MIN_WORK = 8e6` evaluations, and then no comparison or transfer happens. An environment that passes its `StyleTensor` to the builder (`SceneBuilder(B, ..., style=sty)`) also skips the comparison when the style differs per batch row.

Speedup of baked over unbaked on one H200, supersample 1, timing rasterization of a pre-built scene:

| Batch | PushT 96 px | PushT 224 px | SimplePointMaze 500 px |
|---|---|---|---|
| static / total primitives | 6 / 9 | 6 / 9 | 6 / 13 |
| 256 | not baked | 1.33× | 1.65× |
| 1,024 | not baked | 1.92× | 1.72× |
| 4,096 | 1.81× | 2.18× | 1.74× |
| 16,384 | 2.15× | 2.26× | over the output budget |

The PushT columns use an internal Scene2D PushT scene. The public PushT classes draw with their own Numba (CPU) and Triton (CUDA) rasterizers.

## gfxdraw

`stable_envs.render.gfxdraw` is a batched Torch version of the SDL_gfx primitives behind `pygame.gfxdraw`, and classic control frames match Gymnasium's pygame drawing. On CUDA the Triton kernels in `stable_envs.render.gfxdraw.fused` draw, and on CPU a Numba twin in `stable_envs.render.cpu_raster.classic` draws when Numba is installed (`classic` extra), with the Torch functions as the fallback. With `backend="auto"`, some CPU batches with the default appearance use Gymnasium's own renderer instead, and every choice gives the same frames.

It's separate from `raster2d`: SDL_gfx works in integers and takes `Sint16` vertices. The module computes geometry in float64 and truncates every coordinate before drawing. Scanline fills, circle spans, clipping and packed-channel blending keep SDL's integer operation order, and the CUDA path repeats it. The conformance comparison found 0 differing pixels out of 250.9 M.

::: stable_envs.render.gfxdraw
    options:
      show_root_heading: false
      members: [Gfx, StyleCanvas, polygon, circle, aaline, fill_polygon_mask, fill_circle_mask, blend, blit_over, rotate, to_u8]

## Ray casting and CPU3D

`stable_envs.render.raycast3d` intersects rays with spheres, capsules, cylinders, ellipsoids, boxes and finite planes. `stable_envs.render.cpu3d` uses the same intersections to draw a whole primitive scene on CPU when a moving camera rules out a static background, adding sky and floor textures, directional shadows and specular light. Its constructor rejects unsupported geometry and shadow-floor setups.

Fixed-camera rendering bakes a static RGB and depth plate, casts the moving primitives inside conservative screen bounds, and shades only the covered samples. Three MuJoCo conventions decide silhouette and highlight pixels: the mono camera averages its two stereo eyes, the depth buffer measures perpendicular distance while the compositor compares radial ray parameters, and specular light is added before texture modulation.

`backend="auto"` picks Triton for packed float32 inputs on CUDA and Torch otherwise. An explicit `"triton"` needs CUDA float32 inputs on one device with packed trailing axes and raises `ValueError` otherwise. The Torch cast takes `bounds=False` to cast every pixel, for diagnostics. In tests, override the backend in a scoped fixture rather than with a process-wide switch. On CPU, DM Control uses Numba kernels when Numba imports: `cpu_raster.rays3d` repeats the float32 operation order of the Torch intersectors and tests each tile of rays only against primitives its rays can reach, `cpu_raster.suite3d` composites the fixed-camera plate, and `cpu_raster.scene3d` draws a CPU3D scene. They use the IEEE square root, while Torch's CPU square root can round through MKL. Rare pixels differ from the Torch backend as a result.

Tests require exact agreement between the optimized compositor and its full-window reference, between the Torch and Triton casts on float32 inputs, and between bounded and unbounded Torch casts. Agreement with the upstream library's OpenGL frames is measured per class.

::: stable_envs.render.raycast3d
    options:
      show_root_heading: false
      members: [Camera, Lights, intersect, cast_nearest, cast_nearest_torch, any_blocking, any_blocking_torch, cached_rays, shade, static_depth, composite]

## nvdiffrast

`stable_envs.render.nvdiff` rasterizes textured, lit triangle meshes on CUDA with nvdiffrast and returns uint8 frames. The PyFlyt rocket reuses its scene and shading code with its own rasterizer, which follows Bullet's triangle coverage rules. DM Control, Fetch and CPU3D borrow its material and texture helpers, `effective_rgba` and `texture_array`, which run without nvdiffrast. nvdiffrast is [installed from source](../install.md#nvdiffrast).

Projection uses a fixed arithmetic order, and every batch size takes the same matrix-multiply rounding path. Texture origin, row direction and byte rounding are explicit. The backend copies MuJoCo view arrays before later simulator calls can change them. Mipmapping and optional anisotropic filtering control texture footprints, with one tap by default, and antialiasing and supersampling control edge coverage. Each setting has its own cost and needs its own source-image check. The output isn't bit-exact with upstream OpenGL or TinyRenderer images. Exact tests check only that frames stay the same across batch sizes. On x86 CPU, nvdiffrast has a one-ULP batch dependence in a sparse stage, and no registered class uses that path on CPU.

::: stable_envs.render.nvdiff
    options:
      show_root_heading: false
      members: [Scene, BatchRenderer, Camera, Lights, TinyShading, effective_rgba, texture_array, close_cached_renderer]

## MJWarp

`stable_envs.render.mjwarp` renders a batch of MuJoCo configurations on CUDA with MuJoCo Warp, each family with its own model preparation, static background and material handling. Model preparation selects the visible geom groups, moves invisible geometry out of them, sets shadow casting and records texture mappings MuJoCo Warp can't sample. The backend converts MuJoCo Warp's packed BGRA output to RGB, and tracking cameras need center-of-mass data.

- GL-compatible rounding must be active when the backend captures the render graph. A captured graph keeps the pack function it was captured with.
- Building a render context patches MuJoCo Warp's texture-upload and byte-packing helpers for the duration of the call. These patches aren't thread-safe: build contexts only while no other thread uses those helpers.
- Importing the module sets `WARP_CACHE_PATH` to `~/.cache/warp` when it's unset.
- The light specialization applies only to the MuJoCo Warp version and model it supports, and other configurations use stock MuJoCo Warp.
- Exact tests compare the backend with itself across changes, and texture filtering, coverage and shading leave a per-class MSE against the upstream library's OpenGL frames.

See [OGBench](../environments/ogbench.md#cube-and-scene-render-backends) for how Cube and Scene choose between the MuJoCo rasterizer, OpenGL and MJWarp.

::: stable_envs.render.mjwarp
    options:
      show_root_heading: false
      members: [BatchRenderer, prepare_model, Prepared, create_render_context, release_render_context, release_cached_ctx, gl_rounding, needs_com_pos]

## MuJoCo rasterizer

`stable_envs.render.mujoco_raster` draws MuJoCo scenes with a rasterizer that follows MuJoCo's fixed-function OpenGL renderer (`render_gl3.c`). It has a CPU backend (Numba) and a CUDA backend (nvdiffrast and Triton). Both backends read one scene model and one set of per-frame inputs. It is the default renderer of Fetch and of the OGBench Cube and Scene tasks (`backend="raster"`). `backend="opengl"` stays the exact reference.

### Scene model

`stable_envs.render.mujoco_scene.extract` reads one `mjv_updateScene` of a model. It gives the drawn geoms with the triangles of their display lists, the materials and texture mappings, the lights, the camera, the textures with their mipmaps, the skybox and the shadow-map settings. `mujoco_scene.pack` turns the scene into flat arrays: corners, triangles, materials with a table of specular powers, textures, light rows, the camera row and the shadow matrices.

The extraction records the MuJoCo minor version. `mujoco_scene.VERIFIED` lists the scene features that were compared with the OpenGL renderer of each version. A scene that needs another feature raises `NotImplementedError`. Fetch uses MuJoCo 3.11. Cube and Scene use MuJoCo 3.17.

### Per-frame inputs

A family gives these arrays for each frame:

| Input | Shape | Content |
|---|---|---|
| `gpos` | (B, G, 3) | Geom positions |
| `gmat` | (B, G, 9) | Geom rotations, row major |
| `rgba` | (B, G, 4) | Geom colors; an alpha below 0.995 makes the geom translucent in that frame |
| `cams` | (B, 12) | Camera position, right, up and forward |
| `lights` | (B, L, 23) | Light rows; `mujoco_scene.headlight_rows` points the headlight along each camera |
| `sky` | (B, 3) | A flat sky color on the 0-255 scale, or -1 for the skybox |

### Rendering contract

- Lighting is per vertex, with a local viewer, the headlight and the model lights. Both backends use the same formula, the same light rows and the same table of specular powers. The CPU backend computes in float64, the CUDA backend in float32.
- Each frame has 4 samples per pixel. The CPU backend uses the sample positions of Mesa and shades at the pixel center. The CUDA backend uses the sample positions of NVIDIA and shades at the pixel center when the center is in the triangle, else at the sample.
- A shadow light adds its light where the shadow map has no nearer caster. The static geoms make one map per scene; the moving geoms make a map per frame.
- Translucent geoms blend after the opaque geoms and the skybox, sorted by distance to the camera less their bounding radius, once front to back and once back to front.
- The CPU backend clips triangles at the near plane. On CUDA, nvdiffrast clips the opaque triangles; the translucent triangles that cross the camera plane are not drawn.
- On CUDA, a static plane with a 2-D texture at the world origin is drawn as background, and a geom with a cube texture is drawn in the mean color of the texture.
- A draw of the same inputs gives the same frame each time, alone or in any batch.

### Plates

Frames with the same camera, light rows, sky color and colors of the static geoms share a plate on CPU. The plate holds the static opaque geoms, their shadows and the skybox. Each frame starts from its plate and draws the other geoms. The CUDA backend keeps the skybox and the background plane for each camera, light rows, plane color and sky color.

### Segmentation and depth

`mujoco_raster.semantic` gives the geom of each pixel center and its depth along the optical axis, as the segmentation and depth rendering of MuJoCo give them. A pixel shows the nearest opaque geom, or a translucent geom in front of it. Fetch and Cube/Scene give `render_semantic(rows)`: the segmentation as `(objid, objtype)` pairs in the model of the reference environment, and the depth in model units.

## The compiled variation

`stable_envs._core.style.StyleTensor` is the compiled form of variation: one flat `(B, variation_dim)` float32 tensor plus a layout that maps columns to leaves such as `cart.color`. Application code uses `variation_spec`, `default_variation`, `sample_variation` and `render(..., variation=)` instead.

```python
import stable_envs as snv
from stable_envs._core.style import StyleTensor

with snv.CartPole(num_envs=2, device="cpu") as env:
    style = StyleTensor.from_space(env.variation_space, env.num_envs, device=env.device)
    print(style.values.shape, style.values.dtype)  # torch.Size([2, 22]) torch.float32
    print(style.get("cart.color").shape)           # torch.Size([2, 3])
    print(style.for_batch(4).values.shape)         # torch.Size([4, 22])
```

`StyleTensor.for_batch` is the only operation that may change the batch size of a compiled row. A shared style broadcasts its one row to the new batch size. A style whose rows differ raises `ValueError`, since nothing says which rows the smaller batch should keep. Slice the style together with the states instead.

::: stable_envs._core.style
    options:
      show_root_heading: false
