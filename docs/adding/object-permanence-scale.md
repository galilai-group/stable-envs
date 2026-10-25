# Sampled positive scale animation

G146's `string_rod` has constant local triangles and origin, but its Z scale
shrinks from frame 31 to 38. The source keeps a positive scale of about `1e-4`
afterwards; it never deletes the string. The original origin at the top of the
string is retained, so shortening occurs towards the support beam. The ball and
release hook have independently sampled rigid motion.

This extension supports positive diagonal scale animation of otherwise constant
meshes. It does not enable shear, reflection, zero scale, vertex deformation,
changing topology, hiding objects or changing camera/materials. The initial
scale is still baked into the asset once. No motion formula is transcribed.

## Recording contract

`stable-envs/object-permanence-rigid-v3` adds optional `scales.npy`, CPU float32
`[T,D,3]`, with convention `stable-envs/asset-relative-positive-diagonal-v1`.
Its node axis and frame order match `poses.npy`. Each component is the source
world-matrix scale divided by its first-frame component. The first row is exactly
one; all values must be finite and strictly positive. Missing arrays mean unit
scale. V1/V2 recordings cannot silently acquire a scale array.

Assets retain their original object frame, including any offset from the origin.
For asset vertex `v`, normal `n`, relative scale `s`, pose rotation `R` and
translation `p`, world geometry is:

```text
vertex = p + R @ (s * v)
normal = normalize(R @ (n / s))
```

`poses.npy` remains float32 `[T,D,7]` xyz+wxyz; `frames.npy` remains int64 `[T]`.
`RigidTrajectory.poses`, `object_poses`, G34 State and render interfaces retain
their existing layouts. Additive `has_scales`, `scales(indices)` and
`object_scales(indices)` expose owned CPU arrays, defaulting to ones for old data.
The optional file is read-only memory mapped and included in manifest hashes and
clip identity. Invalid shapes, dtypes, values, conventions or hashes are rejected.

The CLI opts in only for audited `ANIMATED_SCALE_TASKS`. Low-level
`sample_scene(..., allow_animated_scale=True)` retains the existing local-mesh,
material, visibility and camera checks. Scale changes are detected within the
existing `2e-6` source decomposition noise budget. Scale-only objects become
dynamic even if their source tag, translation and rotation are constant. No
array/version change is emitted for a scene whose scale remains constant.

## Rendering and static cache

Scaled clips explicitly select their recorded opaque-pixel-ray visibility
contract. This is separate from the v3 file schema: perspective uses v1 visibility,
orthographic uses v2. Coverage, camera rays, coplanar IDs and thresholds stay the
same. Online rendering needs no Blender or MuJoCo.

The shared renderer uploads sampled dynamic poses and scales, then forms
`R diag(s)` on CUDA. Vertices and triangle planes use that affine matrix; shading
normals use its inverse transpose and are normalized before interpolation.
Only scaled clips enable this normal path; the existing unscaled arithmetic and
shared `nvdiff.py` are unchanged.

Static objects retain unit relative scale. The renderer reuses its existing exact
static visibility seed and draws all dynamic objects against it. A shortening
string cannot enter the static seed. Exact cache keys and invalidation are
unchanged; neither a looser comparison nor scene-specific cache logic is needed.

## G146 validation and reproduction

G146 has 8 mesh objects and 3 dynamic nodes (`hanging_ball`,
`release_hook_support`, `string_rod`). Its 120-frame scale file is 4,448 bytes,
including the NPY header. The independent source oracle records absolute
float64 `[T,G,3]` source scales as well as depth, IDs and poses. It evaluates the
actual Blender world triangles, without reading candidate assets or scale arrays.

All-frame results: depth MSE at most **1.057e-16**, every-object mask IoU **1.0**;
world vertices differ by at most **8.345e-8**, corner normals **2.326e-7**.
Cached/full RGB, depth and IDs are bit-identical at 160×90 and 320×180, across
repeated/shuffled frames and chunk sizes 1/3/4/7, with one seed and no fallback.
Synthetic tests also cover combined rotation/anisotropic scale, an offset origin,
nearly flat positive geometry, both camera types and rejected unsupported motion.
All existing fixtures and acceptance thresholds remain unchanged.

Use the pinned Blender interpreter from the [recording guide](object-permanence-recordings.md)
for the first two commands, and the CUDA interpreter for the third:

```bash
python -m scripts.export.object_permanence export --source-root <source> \
  --output <recordings> --task G146
python -m scripts.testing.freeze_object_permanence_rays --source-root <source> \
  --output <new-oracle> --task G146 --visibility-contract stable-envs/opaque-pixel-rays-v1
python -m scripts.testing.check_object_permanence_rendering --recordings <recordings> \
  --oracle <new-oracle> --task G146 --visibility-contract stable-envs/opaque-pixel-rays-v1 \
  --report <report.json>
```

No other deferred first nominal member is blocked solely by positive scale.
The audit now certifies **143/150**: 7 legacy, 92 v1 and 44 v2 visibility cases.
Five transparency and two independent source-normal failures remain deferred;
47 other member scripts remain unregistered.
