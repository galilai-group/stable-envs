# Triangle rendering with nvdiffrast

`nvdiff` provides CUDA triangle rasterization, texture sampling, lighting, and
uint8 output. Rocket reuses its scene and shading support with a family-specific
rasterizer that follows Bullet's triangle coverage rules. It is not a CPU renderer.

`Scene.from_triangles()` accepts meshes without importing MuJoCo. Supply per-geom
positions `(B,G,3)`, rotations `(B,G,3,3)` and a `Camera` to `BatchRenderer.render()`.
MuJoCo model import and camera/light resolution are optional conveniences.
`res=(height,width)` supports rectangular images; integer resolution retains the
existing square path. Diagnostic `mode="depth"` returns positive camera-axis
float32 distance; `mode="ids"` returns one-based int16 geom IDs. Both use zero for
background and require `supersample=1`. IDs are geom order, not source semantic
labels unless the family explicitly makes that order agree.

`render_many(xpos, xmat, camera, modes=("rgb", "depth", "ids"), out=buffers)`
returns a dictionary and shares one transform/coverage pass across outputs.
RGB and depth also share attribute interpolation. Optional output buffers must
match every requested key, shape, dtype and device; validation precedes writes.
`render()` retains its single-Tensor/out contract and the same shading path.
Diagnostics require `supersample=1`, including in a multi-output request.

The G34 path uses the same pipeline with higher-precision coverage and cached
static plates; see [engine-free replay](../adding/object-permanence-recordings.md#read-and-render).

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
