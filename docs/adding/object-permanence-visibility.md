# Versioned opaque pixel-ray visibility

`stable-envs/opaque-pixel-rays-v1` makes source geometry and replay agree on pixel
rays and equal-depth semantic IDs. It is explicitly selected; existing renderers,
State interfaces, raw Blender BVH references and acceptance thresholds are unchanged.
The seven legacy-certified scenes remain certified under their original contract.
G63/G66/G80/G81 retain their legacy BVH deferrals and additionally pass this new
contract. G65/G83 are certified only under the new contract.

## Geometric contract

For a top-left image pixel `(x,y)` of width `W`, height `H`, use the recorded
OpenGL perspective projection `P` and camera-to-world matrix `C`, in binary64:

```text
nx = 2*(x+0.5)/W - 1
ny = 1 - 2*(y+0.5)/H
local_direction = ((nx+P[0,2])/P[0,0], (ny+P[1,2])/P[1,1], -1)
origin = C[:3,3]
direction = C[:3,:3] @ local_direction
point(t) = origin + t*direction
```

Do not normalize the direction: positive `t` is the camera-axis depth parameter.
The recorded matrix, rather than independently rounded `view_frame` corners,
defines both paths. Lens shifts represented by the projection are included.
All opaque triangles are two-sided, with closed edges. Accept hits inside the
recorded `near <= t <= far` interval; background depth/ID is zero.

Select the smallest binary64 plane-ray depth. Exactly equal depths select the
lowest positive semantic object ID from the recording manifest. Within an object,
choose the lowest original triangle index for interpolation. Object enumeration,
thread scheduling and source BVH traversal do not define semantic priority.
There is **no depth bin, silhouette epsilon or near-surface tie tolerance**.
A one-ULP-separated nearer surface wins even if it has a larger semantic ID.

Depth is evaluated from normalized plane coefficients rather than a separate
barycentric depth interpolation for each triangle. Rational local plane construction
canonicalizes represented coplanar vertices before conversion to binary64. The
largest absolute normal component becomes +1 (first axis wins a magnitude tie).
Runtime transforms these planes in binary64. The independent source reference
constructs normalized planes directly from evaluated world triangles. Arithmetic
and source/export geometry remain finite precision; this is a specified numerical
contract with measured source-fidelity gates, not exact real-number geometry.

This preserves occlusion semantics: all objects compete at the same pixel, the
nearest surface hides farther targets, and hidden poses remain in the trajectory.
The ID rule resolves overlapping surfaces that have no unique nearest object;
it never reveals a target through a closer occluder. Shader RGB is not an oracle.

## Implementations and scope

- `rendering/visibility.py` declares the contract and canonical local planes.
- `rendering/visibility_cuda.py` builds world rays, vertices and planes on CUDA.
  A first pass finds depth, a second selects semantic ID/triangle, and a resolve
  pass supplies barycentrics and depth. Existing nvdiff interpolation/shading
  produces RGB; depth comes directly from the same winning ray-plane intersection.
- `scripts/testing/object_permanence_ray_reference.py` independently uses NumPy
  projected edge coverage and rational world-plane construction. It imports no
  candidate geometry, runtime hit routine, Torch or Blender BVH query.
- `scripts/testing/freeze_object_permanence_rays.py` prepares independently audited
  source scenes and reads Blender evaluated world triangles/poses each frame.
  It records source/tool hashes, versions, camera, object IDs and frame numbers.

Enable it with `RigidSceneRenderer(clip, visibility_contract=CONTRACT)`, importing
`CONTRACT` from `stable_envs.envs.object_permanence.rendering.visibility`.
Float32 mesh/pose storage and CPU pose upload stay unchanged; quaternion-to-matrix
reconstruction and visibility use float64. Single-mode and `render_many` interfaces,
output dtypes, arbitrary frame sampling and caller-owned buffers remain available.

This first implementation draws complete coverage. `cache_static=True` still
retains static transforms and uploads only dynamic poses, but does not cache this
new visibility pass. The default legacy renderer retains its existing static
visibility cache. No speed claim is made for the new correctness path.

Supported scope is fixed perspective cameras and opaque rigid meshes. Deformation,
animated materials/visibility, orthographic cameras and triangles crossing depth
planes remain unsupported. Crossing geometry raises an error; it does not silently
switch to a different rasterization contract. New scene groups still require an
entry-point audit, all-frame export checks and independent pixel gates.

## References, validation and reproduction

New references coexist at
`tests/fixtures/semantics/object_permanence/<task>/opaque_pixel_rays_v1/nominal/`.
Their schema is `stable-envs/object-permanence-rigid-oracle-v2`; `visibility_contract`
records the exact ID above. Existing legacy references retain their original paths.
New clips share `tests/fixtures/saved/object_permanence_rigid/`, mesh assets and
licenses. No existing fixture is rewritten. Depth chunks remain bitwise lossless.

```bash
# Blender 4.4 interpreter, clean pinned source; never overwrite an existing output.
python -m scripts.export.batch_object_permanence --source-root <source> \
  --output <recordings> --tasks G63,G65,G66,G80,G81,G83 \
  --visibility-contract stable-envs/opaque-pixel-rays-v1 --report <batch.json>
python -m scripts.testing.freeze_object_permanence_rays --source-root <source> \
  --task G63 --output <new-oracle-directory>
# CUDA interpreter; repeat for every scene's independent reference.
python -m scripts.testing.check_object_permanence_rendering \
  --recordings <recordings> --oracle <new-oracle-directory> --task G63 \
  --visibility-contract stable-envs/opaque-pixel-rays-v1 --report <check.json>
```

Validation requires matching source identity, camera, IDs, frames and contract.
Mixing old/new oracle versions or diagnostic camera overrides is rejected.
Every frame and every visible object uses the unchanged **depth MSE <= 1e-8** and
**mask IoU >= .99** gates. Cached-transform and complete-upload paths both pass.

| Nominal scene | Frames | Maximum depth MSE | Minimum object mask IoU |
|---|---:|---:|---:|
| G63 comb | 120 | 0 | 1.0 |
| G66 carousel | 120 | 3.53e-15 | 1.0 |
| G80 sliding doors | 120 | 0 | 1.0 |
| G81 shutter | 120 | 0 | 1.0 |
| G65 relocating box, lid and two balls | 120 | 0 | 1.0 |
| G83 four cups, three swaps and carried ball | 120 | 0 | 1.0 |

G65/G83 source `main()` calls the audited `build_scene_by_kind`/`animate_scene` pair.
Both animate rigid locations with a fixed perspective camera and opaque materials;
all-frame export verifies constant geometry/scale, source world triangles and
hidden poses. G65 translates the closed box with both balls, then reopens its lid.
G83 moves separate cup-wall/cap meshes and the ball through three swaps and a final
reveal. Their nominal runs validate multi-object composition without a motion formula
in the runtime. These results do not certify every diversity member or all 150 tasks.

For visual review, render source previews with `render_object_permanence_preview`,
then call `benchmarks.object_permanence_recordings.comparison_gif` with
`visibility_contract=CONTRACT`. Both new GIFs contain 120 aligned source/native
Cycles CPU preview and replay frames. Preview quality is 16 samples with denoising,
not original dataset MP4 settings. GIFs/reports remain local development artifacts;
RGB appearance and downstream model accuracy are not asserted equal.
