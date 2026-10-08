# Triangle rendering with nvdiffrast

`nvdiff` provides CUDA triangle rasterization, texture sampling, lighting, and
uint8 output. MuJoCo locomotion can select this backend. Rocket reuses its scene
and shading support with a family-specific rasterizer that follows Bullet's
triangle coverage rules. Neither path is a CPU renderer.

## Rendering contract

Scenes contain tessellated geometry, normals, materials, and texture coordinates.
Projection uses a fixed arithmetic order so changing the batch does not select
a different matrix-multiply rounding path. Texture origin, output row direction,
and byte rounding are explicit. MuJoCo view arrays are copied before later
simulator calls can mutate them.

Mipmapping and optional anisotropic filtering control texture footprints;
antialiasing and supersampling control edge coverage. These settings have
different costs and need separate source-image checks. Anisotropic sampling uses
the screen-x derivatives `(du/dx, dv/dx)` and screen-y derivatives
`(du/dy, dv/dy)`. The default remains one tap.

This backend does not guarantee bit-exact upstream OpenGL or TinyRenderer
images. Lighting, texture filtering, transparency, shadows, and triangle edge
rules must be evaluated for each environment and configuration. Internal exact
tests protect batch invariance and unchanged output during refactoring.

## Recorded performance

The archived textured antmaze-medium scene benchmark measured **12,992 FPS at
batch 256** and **15,994 FPS at batch 4,096**, at 224 px on one H200. Geometry
poses were already on device; this excludes simulation and state preparation.
The report does not supply timing standard deviations. These are historical
backend measurements, not current end-to-end or source-renderer speedups.

An earlier anisotropic-filter conclusion predates the corrected derivative layout
and should not guide tuning.
Use the [testing guide](../testing.md) for a new source comparison.

::: stable_envs.render.nvdiff
