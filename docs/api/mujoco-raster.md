# MuJoCo rasterizer

`stable_envs.render.mujoco_raster` draws MuJoCo scenes with a rasterizer that follows MuJoCo's fixed-function
OpenGL renderer (`render_gl3.c`). It has a CPU backend (Numba) and a CUDA backend (nvdiffrast and Triton). Both
backends read one scene model and one set of per-frame inputs. Fetch and the OGBench Cube and Scene tasks use it
with `backend="raster"`.

## Scene model

`stable_envs.render.mujoco_scene.extract` reads one `mjv_updateScene` of a model. It gives the drawn geoms with
the triangles of their display lists, the materials and texture mappings, the lights, the camera, the textures
with their mipmaps, the skybox and the shadow-map settings. `mujoco_scene.pack` turns the scene into flat
arrays: corners, triangles, materials with a table of specular powers, textures, light rows, the camera row and
the shadow matrices.

The extraction records the MuJoCo minor version. `mujoco_scene.VERIFIED` lists the scene features that were
compared with the OpenGL renderer of each version. A scene that needs another feature raises
`NotImplementedError`. Fetch uses MuJoCo 3.11. Cube and Scene use MuJoCo 3.17.

## Per-frame inputs

A family gives these arrays for each frame:

| Input | Shape | Content |
| --- | --- | --- |
| `gpos` | (B, G, 3) | Geom positions |
| `gmat` | (B, G, 9) | Geom rotations, row major |
| `rgba` | (B, G, 4) | Geom colours; an alpha below 0.995 makes the geom translucent in that frame |
| `cams` | (B, 12) | Camera position, right, up and forward |
| `lights` | (B, L, 23) | Light rows; `mujoco_scene.headlight_rows` points the headlight along each camera |
| `sky` | (B, 3) | A flat sky colour on the 0-255 scale, or -1 for the skybox |

## Rendering contract

- Lighting is per vertex, with a local viewer, the headlight and the model lights. Both backends use the same
  formula, the same light rows and the same table of specular powers. The CPU backend computes in float64, the
  CUDA backend in float32.
- Each frame has 4 samples per pixel. The CPU backend uses the sample positions of Mesa and shades at the pixel
  centre. The CUDA backend uses the sample positions of NVIDIA and shades at the pixel centre when the centre is
  in the triangle, else at the sample.
- A shadow light adds its light where the shadow map has no nearer caster. The static geoms make one map per
  scene; the moving geoms make a map per frame.
- Translucent geoms blend after the opaque geoms and the skybox, sorted by distance to the camera less their
  bounding radius, once front to back and once back to front.
- The CPU backend clips triangles at the near plane. On CUDA, nvdiffrast clips the opaque triangles; the
  translucent triangles that cross the camera plane are not drawn.
- On CUDA, a static plane with a 2-D texture at the world origin is drawn as background, and a geom with a cube
  texture is drawn in the mean colour of the texture.
- A draw of the same inputs gives the same frame each time, alone or in any batch.

## Plates

Frames with the same camera, light rows, sky colour and colours of the static geoms share a plate on CPU. The
plate holds the static opaque geoms, their shadows and the skybox. Each frame starts from its plate and draws
the other geoms. The CUDA backend keeps the skybox and the background plane for each camera, light rows, plane
colour and sky colour.

## Segmentation and depth

`mujoco_raster.semantic` gives the geom of each pixel centre and its depth along the optical axis, as MuJoCo's
segmentation and depth rendering give them. A pixel shows the nearest opaque geom, or a translucent geom in
front of it. Fetch and Cube/Scene give `render_semantic(rows)`: the segmentation as `(objid, objtype)` pairs in
the model of the reference env, and the depth in model units.
