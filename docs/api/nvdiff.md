# Triangle rendering with nvdiffrast

`nvdiff` provides CUDA triangle rasterization, texture sampling, lighting, and
uint8 output. Rocket reuses its scene and shading support with a family-specific
rasterizer that follows Bullet's triangle coverage rules. It is not a CPU renderer.

## Rendering contract

Scenes contain tessellated geometry, normals, materials, and texture coordinates.
Projection uses a fixed arithmetic order so changing the batch does not select
a different matrix-multiply rounding path. Texture origin, output row direction,
and byte rounding are explicit. The backend copies MuJoCo view arrays before
later simulator calls can mutate them.

Mipmapping and optional anisotropic filtering control texture footprints.
Antialiasing and supersampling control edge coverage. Each setting has its own
cost and needs its own source-image check. Anisotropic sampling uses the
screen-x derivatives `(du/dx, dv/dx)` and screen-y derivatives `(du/dy, dv/dy)`.
The default is one tap.

This backend does not guarantee bit-exact upstream OpenGL or TinyRenderer
images. Check lighting, texture filtering, transparency, shadows and triangle
edge rules for each environment. Internal exact tests check batch invariance.

## Performance

A textured antmaze-medium scene rendered at 12,992 FPS at batch 256 and 15,994 FPS at batch 4,096, at
224 px on one H200. Poses were already on the GPU. The timing does not include simulation and state preparation.

See the [testing guide](../testing.md) for source comparisons.

::: stable_envs.render.nvdiff
