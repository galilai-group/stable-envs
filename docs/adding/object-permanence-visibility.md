# Versioned opaque pixel-ray visibility

Current coverage: all 150 first nominal members plus two additional optical members.
G98/G132 now use [native source realizations](object-permanence-native-scenes.md).
The deferral/batch sections below preserve historical diagnostics; 45 additional
member scripts remain unaudited.

`stable-envs/opaque-pixel-rays-v1` makes source geometry and replay agree on pixel
rays and equal-depth semantic IDs. It is explicitly selected; existing renderers,
State interfaces, raw Blender BVH references and acceptance thresholds are unchanged.
The seven legacy-certified scenes remain certified under their original contract.
G63/G66/G80/G81 retain their legacy BVH deferrals and additionally pass this new
contract. In total, 91 scenes are certified under this explicitly selected
contract; the source audit catalog lists their IDs.

Fixed orthographic cameras use the separate [V2 contract](object-permanence-orthographic.md).
Rigid Curve/FONT objects use [offline evaluated mesh conversion](object-permanence-converted-geometry.md).
The import history below records earlier V1 deferrals; the
[source audit](object-permanence-audit.json) is the current acceptance record.

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


## Further reusable batches and unsupported capabilities

Two further batches use the existing pipeline without changing runtime code,
recording schema, ray contract, independent hit algorithm or acceptance budgets:

| Batch | Task | Audited source behavior | Meshes / dynamic nodes | Maximum depth MSE | Minimum mask IoU |
|---|---|---|---:|---:|---:|
| Nested/orbiting/shuffling cups | G85 | Nested cups translate together, then reveal the ball in two stages | 8 / 5 | 0 | 1.0 |
| Nested/orbiting/shuffling cups | G87 | Four cups and a disc rotate; the ball follows its carrying cup | 13 / 10 | 3.94e-15 | 1.0 |
| Nested/orbiting/shuffling cups | G88 | Flat front/back-lane swaps, empty-cup fakeout, then actual ball reveal | 10 / 7 | 0 | 1.0 |
| Height constraints | G89 | A short rolling puck passes under a bar; a tall cylinder stops before it | 7 / 2 | 2.74e-15 | 1.0 |

All four source `main()` functions call `build_scene_by_kind` followed by
`animate_scene`; their selected builders return dictionaries. The audited source
sets only location/rotation keyframes. All-frame export and independent source
world-triangle/normal checks verify the actual opaque rigid meshes, fixed
perspective camera and sampled motion, including hidden objects. G88's old
comments mention arcing cups, but its active `_swap_positions` uses opposite y
lanes on the table: the exported data preserves the code, not the prose.

Each new oracle/GIF covers all 120 source frames at 320x180. Both cached and
complete drawing pass all-object depth/mask gates. A separate native-resolution
comparison checks bit-identical replay RGB/depth/IDs, reverse/shuffled/repeated
sampling and one static-seed build. Automated cache equivalence now also covers
both 160x90 and 320x180 for **every** versioned scene, not just these four.
RGB equality here means replay paths; source Cycles RGB is still a visual preview,
not a pixel-value oracle. GIFs and diagnostic reports stay local artifacts.

Source masks show the G85/G87/G88 ball fully hidden for 91/95/99 frames. G89's two
objects remain visible: a complete hiding interval is not invented for a height
constraint task. G89 defines a glass material in its helper library, but no active
mesh uses it; the existing exporter correctly accepts its actual scene.

### Skip G90: active transparent funnel endcaps

G90's selected `build_funnel_size_sorter_scene` calls `create_v_funnel` twice.
Unlike G89's unused material, this constructs four **active** glass meshes:
`small_funnel_endcap_F/B` and `large_funnel_endcap_F/B`. All use `mat_glass_clear`
with alpha `0.20000000298023224`; the source intends the balls to remain visible
through the caps. The source is still the pinned commit above, script
`object_permanence/generator/tasks/G90_funnel_size_sorter/pb_task_00000136_pb_funnel_size_sorter_0137_blender.py`
(SHA256 `a16375a2d967c0999962fbfe8a0576937b6e6b52917e220e1da09bd31c1a917a`).

Evaluating that source and calling the **unmodified** `sample_scene` at frame 1
raises `unsupported transparency/material links: large_funnel_endcap_B`.
`test_next_source_task_with_active_glass_is_rejected_by_unchanged_opaque_contract`
reproduces this with the optional pinned Blender/source environment. The separate
[alpha extension](object-permanence-transparency.md) now provides G90's explicit
v4 recording, independent layered oracle and preparation entry. It remains
outside this opaque contract. The original four legacy BVH
deferrals remain unchanged.

This is a material/visibility-contract boundary, not a silhouette error that can
be accommodated by a larger tolerance. Dropping the caps or making them opaque
would change the source's visible-object semantics. Continuing needs an explicit
transparent-surface policy: what RGB transmission/compositing to retain, whether
depth denotes the first surface or an opaque surface, and how object masks account
for a target visible through glass. Such a version needs matching material export,
rendering and independently generated oracles alongside v1. The alpha extension
implements this separately, without silently substituting it here. Later opaque perspective scenes can be audited
independently without extending this contract.

### Three more batches: G93–G97/G99/G100

G91/G92 are also skipped: executing their selected build/animate functions
produces orthographic cameras. Optional live source tests require the unchanged
exporter to reject them. G96–G98 have an unused orthographic helper option, but
their actual selected cameras are perspective; a script-wide flag alone cannot
decide support.

| Batch | Task and evaluated source motion | Meshes / dynamic nodes | Maximum depth MSE | Minimum mask IoU |
|---|---|---:|---:|---:|
| Support and falling | G93 shelf tips after prop removal | 6 / 3 | 1.14e-14 | 1.0 |
| Support and falling | G94 hinged trapdoor and falling ball | 9 / 2 | 8.65e-16 | 1.0 |
| Support and falling | G95 belt stripes and ball falling from edge | 20 / 5 | 2.98e-16 | 1.0 |
| Opening containers | G96 hinged clamshell panels | 9 / 3 | 1.60e-14 | 1.0 |
| Opening containers | G97 nine rigid tambour slats | 17 / 10 | 1.30e-16 | 1.0 |
| Scripted collisions | G99 equal-mass ball velocity exchange | 4 / 2 | 6.04e-16 | 1.0 |
| Scripted collisions | G100 heavy/light ball collision and drop | 5 / 2 | 8.61e-15 | 1.0 |

Each is the first nominal member, with an independent 120-frame v2 oracle at
320x180. Both cached and complete drawing pass the unchanged MSE/IoU gates.
Generic tests compare RGB/depth/IDs bitwise at 160x90 and 320x180, including reverse,
shuffled and repeated frames, output mutation isolation and one seed build.
Source world-triangle/normal checks also cover all 120 frames.

These are preparation bindings, not scene-specific runtime implementations.
G94's parented hinge motion is sampled in world space. G96's hinge origins are
constant mesh construction, not deformation.
G99/G100 source animation accumulates rolling orientation across frames: run its
animation once offline, then sample evaluated poses, including hidden motion.
Replay neither reimplements collision formulae nor claims action-driven physics.

This batch brought the scope to seven legacy scenes plus 20 versioned scenes, 27
audited first members out of 197 scripts. G90/G91/G92/G98 remain outside the registry;
later tasks and diversity members still need separate audits. No runtime code,
schema, contract, old fixture or acceptance budget changes in this batch.

Native Blender preview PNGs are temporary GIF inputs. Generate them in an owned
temporary directory, build the aligned source/replay GIF with `comparison_gif`,
verify all 120 frames decode, then remove that directory. Keep only comparison
GIFs under `.cache/object_permanence/evaluation/`; frozen numeric oracles and
recordings stay under `tests/fixtures/` and must not be treated as scratch files.

### Historical G98 deferral: independent source-normal reproducibility

This deferral is resolved for a hash-pinned native realization; see the
[native scene guide](object-permanence-native-scenes.md). The original failure
measurements below remain historical evidence.

G98's selected scene is perspective and opaque. Its dome has a solidify modifier
applied during construction and only translates later, so a first export passed
all-frame triangle/normal checks. Its initial candidate also passed all 120 depth
and mask frames (MSE 0, IoU 1.0) and cached/full bit parity.

A fresh live-profile export failed the **existing** independent normal gate:
maximum difference `7.70698694e-6`, above `atol=3e-6`, `rtol=0`. Additional isolated
source builds compared against the original and fresh exports failed at frame 1,
`liftoff_dome_lid`, with differences `5.03981316e-6` and `3.05238322e-6`.
Position checks passed before these failures. Thus pixel fidelity alone is not
sufficient to certify repeatable asset extraction. The cause of the corner-normal
variation needs investigation; no rendering or sampling defect is established.

Source: `G98_liftoff_dome_lid/pb_task_00000144_pb_liftoff_dome_lid_0145_blender.py`,
SHA256 `e43ced40aafb0bb7fc52d0f31c4adc6fb9adaee0b61fcdce5f25ec56a4fa4fc4`.
The diagnostic calls are `build_scene_by_kind()` then `animate_scene_by_kind()`;
the independent world-triangle/normal checker provides the unchanged gate.
G98 has no accepted recording, oracle or preparation registry entry. Its local
GIF is stored in `evaluation/deferred-previews/g98-deferred-source-vs-pixel-rays-v1.gif`. Normal budgets,
runtime implementation and reference generation algorithms are unchanged.

## Early nominal entries: G01–G10

The next audit starts at G01 and covers the **first nominal member** of each
task through G10. Two batches reuse the same recording format and simplified
renderer. Other members, including G02's two-ball variant and G05's six other
scripts, remain unregistered; batch reports correctly mark them unsupported.

| Batch | Task and source behavior | Meshes / dynamic nodes | Maximum depth MSE | Minimum mask IoU |
|---|---|---:|---:|---:|
| Ramp balls | G02 rolling ball behind an opaque tunnel | 5 / 1 | 4.18e-14 | 1.0 |
| Ramp balls | G03 rolling ball blocked by a solid wall | 8 / 1 | 4.36e-14 | 1.0 |
| Rigid vehicles | G05 car reaches wall and stops | 11 / 5 | 3.53e-17 | 1.0 |
| Rigid vehicles | G06 car falls through a bridge opening | 34 / 5 | 4.56e-17 | 1.0 |
| Rigid vehicles | G07 car stops at a closed guillotine gate | 17 / 10 | 2.95e-17 | 1.0 |

All five have independent 120-frame oracles. Cached and full drawing pass every
depth/object-mask gate. Generic tests check RGB/depth/IDs bitwise at 160x90 and
320x180 with reverse/shuffled/repeated requests, output isolation and a single
static seed build. Independent source triangles/normals check all frames, not
just visible objects. The car body, wheels and spokes retain separate sampled
poses; no vehicle-specific runtime code is needed.

### Offline entry preparation

G02/G03 are older scripts with scene creation and animation at module top level,
followed by native rendering, directory deletion and file saving. Importing the
complete file would execute those operations. An explicit audited
`$prefix:render_png` recipe executes the unchanged construction/animation prefix,
ending before the `render_png` definition. It omits only the top-level
`os.makedirs(OUT_DIR, exist_ok=True)` output setup. The pinned source remains
clean. Unit tests require all 120 animation statements to survive and require
missing boundaries/unreviewed output setup to fail.

G05's source animation takes several named parameters, rather than one object
dictionary. Offline preparation binds those parameter names to the dictionary
returned by its builder, exactly as its original `main()` does. Existing
single-dictionary, tuple and G27 entry recipes keep their behavior. The oracle
has independent source-entry preparation and never imports the candidate
exporter or renderer. Geometry extraction, ray rules, thresholds and runtime
rendering are unchanged; no shader, shadow or antialiasing work is added here.

### Skipped source entries and previews

| Task | Actual source limitation | Retained preview |
|---|---|---|
| G01 | Active acrylic materials have alpha .28/.42; the opaque material gate rejects them | 120-frame source-only GIF |
| G04 | Default mesh-only extraction rejects `CURVE`; explicit evaluated conversion now certifies the scene | 120-frame source/replay GIF |
| G08 | The later source-context audit resolves missing `DIVERSITY`; the camera remains orthographic | 120-frame source-only GIF with audited context |
| G09/G10 | Actual selected cameras are orthographic | 120-frame source-only GIFs |

Live tests retain transparency and default-path Curve/orthographic rejections. These five
tasks have no accepted recordings/oracles or preparation entries. G08's original
direct-call failure was a missing driver context at `drawer_travel_scale`, not
proof of a broken source generator; the context audit below runs its animation.

Accepted source/replay GIFs stay in `evaluation/`. Skipped diagnostics go in
`evaluation/deferred-previews/`, with visible **source only / NOT CERTIFIED**
labels. Unsupported scenes cannot honestly provide a certified replay
comparison. G08's old initial-scene diagnostic is superseded by its full source GIF.
Each successful GIF is decoded to verify its expected frame count before its
temporary PNG directory is deleted. Only GIFs remain in the artifact cache.

This early batch brought the scope to **32 audited first members: seven legacy
and 25 versioned scenes**, out of 197 scripts. Old fixtures, interfaces and
acceptance budgets remained unchanged.

## Complete nominal audit: G11–G150

All 150 task groups now have a measured disposition in the
[source audit catalog](object-permanence-audit.json). The first nominal member is
the supported import unit: **95 are certified, 55 are deferred**. The other 47
member scripts are explicitly listed as unregistered entries, not certified by
similarity to the first member. There are no unclassified nominal tasks.

Four further reusable batches add 61 recordings without changing the renderer,
recording schema, State interfaces, ray contract or thresholds:

| Source range | Newly certified | Representative behavior | Maximum depth MSE | Minimum mask IoU |
|---|---:|---|---:|---:|
| G28–G42 | 9 | Tunnels, carrying carts, vehicle identities, hidden lifts, trapdoor and ring track | 9.39e-16 | 1.0 |
| G56–G79 | 12 | Cup swaps, hinged lids, pendulums, slot/height gates, scattered balls and blinds | 3.31e-13 | 1.0 |
| G104–G127 | 20 | Folding doors, lids, transport trays, flaps, support withdrawal and gates | 1.06e-13 | 1.0 |
| G129–G150 | 20 | Rotating/lifting mechanisms, traps, supports and separating rollers | 2.01e-14 | 1.0 |

Every newly certified recording passed all-frame source extraction, an additional fresh
Blender-process world-triangle/normal comparison, and both cached and full
120-frame depth/all-object-mask gates. RGB/depth/IDs also agree bitwise at 160x90
and 320x180 for reverse, shuffled and repeated requests. Output mutation isolation
and the existing single static-seed construction assertion remain enforced.
The catalog records per-scene metrics, source hashes, entry recipes and independent
world-check errors; the inventory separates these measured results from AST hints.

Visibility probes are the task objects designated by source roles (targets,
balls, shown objects or stationary task objects), not every moving mesh. For
example, rear wheels, hidden carriage panels and strings remain in the complete
poses and all-object ID/mask comparisons even when never visible. They are not
used as visibility probes requiring at least one visible frame. The catalog
records the probe names and their raw source roles independently of motion tags.

### Historical deferred capabilities and stability

| Primary reason | Nominal tasks | Policy |
|---|---:|---|
| Optical transmission/refraction | 1 | G01 requires a different optical policy; no opaque or alpha-only replacement |
| Independent source-normal variation | 2 | G98/G132 exceed the existing 3e-6 normal budget |

Orthographic projection and constant Curve/FONT geometry are now supported by
the linked extension guides, retaining cameras, rails and identity labels.
[Positive diagonal scale animation](object-permanence-scale.md) also certifies
G146 without changing pixel rays or thresholds; general deformation remains unsupported.
G132's fresh-process corner-normal difference is `3.16480663e-6`; its provisional
data is excluded, rather than raising the normal tolerance.

G103's former cache-retention deferral is resolved by the batch-independent
visibility transforms described below. Its original source geometry and numeric
reference were retained; the single-build assertion and exact cache keys remain.

### Preview artifacts and cleanup

Certified comparisons contain every source frame. Current-contract comparisons
for G63/G66/G80/G81 coexist with their historical legacy diagnostics. Deferred
previews use 24 evenly sampled source frames and visible **NOT CERTIFIED** labels;
the five source-context scenes now have complete 120-frame source previews.
The other earlier previews retain their original frame counts.
G98's existing full comparison remains a deferred diagnostic. G103 now has a
certified comparison in `evaluation/`, replacing its obsolete deferred preview.

Native previews initially used Cycles CPU, 16 samples and denoising. Most of this
bulk pass uses the same original source scene with Cycles CUDA, 64 samples and
denoising disabled; this reduces preview-generation cost and can leave visible
sampling noise. These settings affect only illustrative RGB, never the independent
CPU geometry oracle or shared simplified renderer. RGB equality with finished
dataset MP4s is not asserted.

Accepted GIFs live in `evaluation/`; deferred/historical diagnostic GIFs live in
`evaluation/deferred-previews/`. Each output's frame count is decoded before
temporary PNGs are removed. Frozen numeric fixtures and source provenance stay
in the repository; the artifact cache retains GIFs only.

## Audited source execution context: G08/G44/G45/G46/G48

The source driver's `core/render.py` injects `DIVERSITY` before script execution.
Its `load_script_diversity(..., profile="surface")` returns empty values/bindings,
no factorial sample and no viewpoint change. Both the exporter and independently
implemented reference preparation now obtain that official configuration and
verify that `apply_source_bindings` leaves the source text unchanged. Only the
five explicitly audited entries receive these globals; older namespaces and
fixtures are unchanged. Source `main()`, file output and driver appearance/render
hooks do not execute. This is source construction with the surface-profile
globals, not a claim to reproduce the complete surface-profile RGB pipeline.

The CLI defaults to a different `task` diversity profile. At sample index 0,
first member, that profile gives G08 a drawer-travel factor of .82 and the other
four scenes their respective scale/spacing factors of .84; it also requests
viewpoint changes. Those are additional dataset variants, not these nominal
recordings. The profile selection is explicit rather than silently replacing
the CLI's default randomized configuration with an empty dictionary.

| Task | Source-defined default | Full-animation audit result |
|---|---|---|
| G08 | `drawer_travel_scale=1.0` | Orthographic camera; certified under v2 |
| G44 | `ball_radius_scale=1.0` | Opaque perspective scene; certified |
| G45 | `ball_radius_scale=1.0` | Orthographic camera; certified under v2 |
| G46 | `cover_footprint_scale=1.0` | Converted FONT objects `box_A_label`/`box_B_label` and orthographic camera; certified under v2 |
| G48 | `object_spacing_scale=1.0` | Orthographic camera; certified under v2 |

G44 contains five mesh objects and two recorded moving balls. All 120 frames pass
fresh-process source-world checks (position error `6.68e-8`, normal error `4.41e-7`),
cached/full depth MSE `1.16e-16`, and minimum object mask IoU 1.0. Cache/full
RGB/depth/IDs are bit-identical at 160x90 and 320x180, including shuffled/repeated
requests and output mutation isolation; each cache builds one static seed. Its
independent oracle was regenerated in another process with byte-identical numeric
archives. Motion is sampled from the original animation; no collision formula is
implemented in the runtime.

New recording/oracle manifests retain the execution profile, actual globals,
absence of bindings, and hashes of the upstream driver/diversity module. The ray
oracle additionally records its preparation-module hash. Comparisons reject an
execution-context mismatch, including preview comparisons. Both absent contexts
continue to match for older recordings/oracles. The source pin, recording schema,
opaque visibility contract, State interfaces and acceptance budgets are unchanged.

The measured catalog records the resolved source context and camera/geometry
capabilities. All five now have 120-frame source/replay GIFs. Their earlier
NOT CERTIFIED previews and temporary PNGs are removed after full decoding.

## Batch-independent visibility transforms and G103

G103's unchanged static poses produced one seed for a 120-frame request, then
two/four cumulative builds after a five-frame request and its repetition. The
final chunk has one row instead of four. Input poses, matrix inverses, world and
clip vertices were bit-identical; the plane-normal `einsum` contraction changed
392 normal components by up to `1.11e-16`, affecting 524 normalized plane
coefficients by up to `4.44e-16`. Exact invalidation correctly detected these
different inputs. This was shape-dependent GEMM rounding, not source motion or
an incorrect depth/mask result.

`transforms.row_matrix3` evaluates vertex and plane-normal transformations with
three explicit products/additions in a fixed order. Camera projection reuses
`nvdiff._mm4`, the existing fixed four-term product. Independent random seven-object
scenes also exposed the corresponding vertex-transform issue, so the fix applies
to the complete visibility transform path rather than one scene or coefficient.
Matrix inversion, precision, plane normalization, hit tests, semantic ties and
`StaticRayCoverage` are unchanged. No epsilon, rounded key or ignored input is added.

Generic CUDA regressions use 7/41 arbitrarily rotated meshes, a nontrivial camera,
all-static and mixed partitions, and changing batch shapes. They failed before
the fix and now retain one seed while preserving all RGB/depth/ID outputs.
Real static changes still rebuild, within-batch static variation still uses full
drawing, and even a one-ULP plane change invalidates the exact seed.

G103 is now imported with 99 meshes and one sampled dynamic node. Fresh source
world checks cover all 120 frames (position error `3.08e-7`, normal error `3.78e-7`);
the target is fully occluded in 15 source frames, with its hidden poses retained.
Both cached/full source gates pass maximum depth MSE `8.19e-16` and minimum mask
IoU 1.0. All outputs are bit-identical to the pre-fix G103 rendering at both
160x90 and 320x180. Requests of 1/2/3/4/5/7/9/16/31/64/120 frames across chunk
sizes 1/3/4/7/16 retain one seed, no fallback, and identical selected outputs.
Frozen regressions additionally exercise reverse/shuffled/repeated requests,
chunk changes and output mutation isolation without changing their budgets.

The recording schema, source entry functions, opaque visibility contract and old
fixtures are unchanged. G103's independent source oracle was regenerated in a
second process with byte-identical numeric archives. Its 120-frame source/replay
GIF uses the existing simplified renderer; temporary PNGs are removed after decoding.
