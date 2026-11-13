# MuJoCo rasterizer

`stable_envs.render.mujoco_raster` draws MuJoCo scenes on CPU with a Numba rasterizer that follows MuJoCo's
fixed-function OpenGL renderer (`render_gl3.c`). It is the CPU default renderer of the OGBench Cube and Scene
tasks (`backend="numba"`). `backend="opengl"` stays the exact reference. On CUDA, `backend="numba"` raises
`ValueError`. Fetch uses it for `render_semantic` only.

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

- Lighting is per vertex, in float64, with a local viewer, the headlight and the model lights.
- Each frame has 4 samples per pixel at the sample positions of Mesa, shaded at the pixel centre.
- A shadow light adds its light where the shadow map has no nearer caster. The static geoms make one map per
  scene; the moving geoms make a map per frame.
- Translucent geoms blend after the opaque geoms and the skybox, sorted by distance to the camera less their
  bounding radius, once front to back and once back to front.
- Triangles are clipped at the near plane.
- A draw of the same inputs gives the same frame each time, alone or in any batch.

## Plates

Frames with the same camera, light rows, sky colour and colours of the static geoms share a plate. The plate
holds the static opaque geoms, their shadows and the skybox. Each frame starts from its plate and draws the
other geoms.

## Segmentation and depth

`mujoco_raster.semantic` gives the geom of each pixel centre and its depth along the optical axis, as MuJoCo's
segmentation and depth rendering give them. A pixel shows the nearest opaque geom, or a translucent geom in
front of it. Fetch and Cube/Scene give `render_semantic(rows)`: the segmentation as `(objid, objtype)` pairs in
the model of the reference env, and the depth in model units.
