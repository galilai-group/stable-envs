# Versioned opaque pixel-ray visibility

`stable-envs/opaque-pixel-rays-v1` makes source geometry and replay agree on pixel
rays and equal-depth semantic IDs. It is explicitly selected; existing renderers,
State interfaces, raw Blender BVH references and acceptance thresholds are unchanged.
The seven legacy-certified scenes remain certified under their original contract.
G63/G66/G80/G81 retain their legacy BVH deferrals and additionally pass this new
contract. G65/G82/G83/G84/G86 are certified only under the new contract.

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

`cache_static=True` now also caches static ray visibility. The seed stores exact
int64 depth bits and the original semantic-ID/triangle selection key. Each batch
clones these buffers, evaluates only dynamic triangles, clears an old winner only
when a strictly nearer surface changes depth, then applies the same semantic tie
rule. Static and dynamic geometry share final resolve/interpolation/shading.
`cache_static=False` retains complete drawing and full pose upload for comparison.

`rendering/visibility_cache.py` owns the partition and seeds without knowing scene
names. Static clip/world vertices, planes, rays, origin, labels, clipping bounds
and resolution invalidate the seed when changed. Within-batch static variation
uses a full pass rather than caching an incorrect first frame. Topology mutation
or replacement is rejected; output buffers cannot alias the seed. Closing releases
all retained validation snapshots and pixel buffers. RGB is still shaded each call,
so material color changes do not require a cached background color plate.

At 320x180 the two pixel buffers use 900 KiB; validation snapshots add about
1.34-1.38 MiB for these three new scenes, plus partition indices. All vertices,
planes and shading attributes are still transformed; dynamic coverage, full-frame
resolve, interpolation/shading, validation synchronization and small-chunk launches
remain. This cache removes repeated static ray/triangle tests, not every static
computation. The legacy renderer keeps its separate existing visibility cache.

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
**mask IoU >= .99** gates. Cached and complete drawing paths both pass; all RGB,
depth and ID outputs also match bit-for-bit over all 120 frames at the comparison
resolution, including shuffled/repeated sampling.

| Nominal scene | Frames | Maximum depth MSE | Minimum object mask IoU |
|---|---:|---:|---:|
| G63 comb | 120 | 0 | 1.0 |
| G66 carousel | 120 | 3.53e-15 | 1.0 |
| G80 sliding doors | 120 | 0 | 1.0 |
| G81 shutter | 120 | 0 | 1.0 |
| G65 relocating box, lid and two balls | 120 | 0 | 1.0 |
| G83 four cups, three swaps and carried ball | 120 | 0 | 1.0 |
| G82 folding six-panel screen | 120 | 1.67e-15 | 1.0 |
| G84 two balls and three cups | 120 | 0 | 1.0 |
| G86 covered conveyor boxes | 120 | 4.22e-16 | 1.0 |

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


## Next rigid batch and cache measurements

G82, G84 and G86 all use the audited `build_scene_by_kind`/`animate_scene` source
entry pairs. G82 returns `(scene, meta)` and animates six panels' locations/yaws;
G84/G86 return dictionaries and animate only locations. No renderer task branch
or runtime animation formula is added. Live export checks every frame's source
world triangles/normals, constant mesh/scale, camera and opaque material properties.
Source keyframing is not evidence of a geometry or visibility change by itself.

| Source scene | Mesh objects | Dynamic nodes | Measured target occlusion |
|---|---:|---:|---|
| G82 | 15 | 6 | Red cube hidden 66 frames, blue cylinder 54; both revealed unchanged |
| G84 | 11 | 8 | Each ball hidden 88 frames while its cup moves in x/y, then revealed |
| G86 | 33 | 26 | Ball hidden 92 frames while covers and belt stripes advance, then revealed |

Each independent oracle and source/replay GIF covers all 120 frames. Preview
quality remains native Cycles CPU, 16 samples, denoising. New clip/reference
paths follow the same versioned layout above; earlier fixtures are not rewritten.

The reusable cache benchmark holds pose upload constant on both sides: static
transforms remain resident and only dynamic poses are uploaded. Its full-coverage
case disables only visibility seeds. It checks bit-identical outputs before timing,
uses one CPU thread, reuses caller outputs, alternates case order and measures
synchronized wall time after warmup. Constructor/JIT/static bake and image download
or GIF encoding are excluded; validation and internal allocations are included.

```bash
python -m benchmarks.object_permanence_visibility \
  --recordings tests/fixtures/saved/object_permanence_rigid \
  --tasks g82,g84,g86 --batches 1,256 --warmup 3 --repeats 7 --profile \
  --output <evaluation>/ray-cache-benchmark.json
```

Timings are rendering throughput, not training speedups. Single-frame cache checks
can exceed the saved work; batch throughput is the intended comparison. The report
retains every sample, seed/snapshot memory, partition sizes and optional profiles.


Measured on RTX 3070 Ti Laptop, Torch 2.14.1+cu130 / Triton 3.8 / nvdiffrast 0.4,
180x320 RGB, chunk size 4, 3 warmups and 7 alternating synchronized repeats:

| Scene | Batch | Complete ray coverage fps | Cached visibility fps | Speedup |
|---|---:|---:|---:|---:|
| G82 | 1 | 172 | 220 | 1.28x |
| G82 | 256 | 334 | 712 | 2.13x |
| G84 | 1 | 91 | 103 | 1.13x |
| G84 | 256 | 247 | 469 | 1.90x |
| G86 | 1 | 113 | 125 | 1.10x |
| G86 | 256 | 233 | 550 | 2.36x |

The batch-256 comparisons upload the same 168/224/728 bytes per frame for
G82/G84/G86, respectively. Each static seed is built once. Static/dynamic face
counts are 284/72, 36/5544 and 372/2508. Face count alone does not predict work:
floor/backdrop triangles cover many pixels, unlike small cup facets. Profiles
confirm reduced ray-hit work; transforms, interpolation/shading, full-frame resolve,
seed clones, validation synchronization and small-chunk dispatch still remain.
Single-frame gains are smaller and sensitive to fixed costs/GPU clock behavior;
the earlier exploratory run included a small single-frame regression. These are
local steady-state measurements, not a promise of speedup on every device/batch.
