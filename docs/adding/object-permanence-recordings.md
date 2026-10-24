# Rigid object-permanence recordings

The next porting unit is an exported scene and its sampled motion, not a new
physics adapter per task. The source is pinned to
[`824baddcdad087810d35638bf8f273b884f74975`](https://github.com/hokindeng/object-permanence/tree/824baddcdad087810d35638bf8f273b884f74975).
The [source inventory](object-permanence-inventory.md) covers all 150 task groups
and 197 member scripts. Its flags are **static, script-wide signals**, including
unused builders; they are not a claim that every nominal/diversity variant works.

## Audited scenes and supported capabilities

G34 is the existing single ball behind a screen. The second pilot is
[G62 two balls cross tunnel](https://github.com/hokindeng/object-permanence/blob/824baddcdad087810d35638bf8f273b884f74975/object_permanence/generator/tasks/G62_two_balls_cross_tunnel/pb_task_00000108_pb_two_balls_cross_tunnel_0109_blender.py):
two balls travel in opposite directions on parallel lanes through an opaque
tunnel. Both have stored poses while hidden; their IDs and colors remain distinct.
Its 120-frame, 24 FPS nominal scene has 15 mesh objects and two dynamic nodes.
The first expansion batch adds G37 (curved U-track ball), G38 (pendulum bob and
rod), and G64 (hinged windowed board in front of two stationary objects).
All audited scenes use the unmodified first member's build/animate functions, bypassing
the source main function's video, screenshot and scene-saving operations.
The source animation produces poses; the exporter does not transcribe its motion.

| Capability | First recording version |
|---|---|
| Arbitrary sampled rigid translations/rotations, including inherited parent motion | Supported, world-space xyz + wxyz |
| Several independently moving objects | Supported |
| Constant positive scale | Baked into local mesh; later scale checked within 2e-6 source decomposition noise |
| Constant evaluated modifiers | Supported if exported geometry stays identical across all frames |
| Static opaque base colors and one material per mesh | Supported; simplified rendering does not preserve source shaders |
| Fixed perspective camera | World and projection matrices exported |
| Orthographic camera | Not supported in v1; e.g. G43/G77 require an extension |
| Animated scale/shear, deformation, topology changes | Rejected |
| Moving camera, changing base colors, visibility changes | Rejected |
| Generated instances, textures, transparency, depth of field | Rejected |
| Dataset-driver diversity variants | Not implemented |

Every exported frame is checked. Source metadata `pb_is_dynamic` is retained
through dynamic-node selection, but actual world motion is also detected, so an
incorrect static tag cannot discard motion inherited from a parent. Occlusion is
not a visibility flag: hidden-behind-geometry objects retain their pose rows.
Static inventory signals identify further audits: G05/G146 include scale paths,
G42/G44 include dimensions paths, and G73 includes rigid-body configuration.
Unused branches can explain a signal; inspect the selected builder before grouping.

## Files and identities

```text
recordings/                         # user dataset/cache, outside source control
  licenses/<license-text-sha256>.txt # one copy per distinct license text
  assets/<shape-or-role>/<mesh-content-sha256>.npz
  clips/g34-nominal/
    manifest.json
    poses.npy
    frames.npy
  clips/g62-nominal/
    ...
```

An **asset** is reusable local geometry, independent of placement, motion and
color. One asset can be used by two differently colored balls in one scene, or
by objects in different recordings. A **scene/clip** binds assets to named
objects, constant transforms, colors, camera and one sampled trajectory.
Different geometry or camera diversity can therefore produce a different scene
even when the source task ID is unchanged.

Each mesh NPZ contains only numeric arrays: `vertices float32[V,3]`,
`normals float32[V,3]`, `triangles int32[F,3]`. Corner normals are preserved by
splitting vertices at normal discontinuities. Constant scale is applied once.
Vertex/normal pairs and oriented triangles are ordered canonically: Blender can
enumerate the same mesh faces differently across builds. Canonical ordering
changes no coordinates or triangle winding and permits actual content reuse.

The asset key hashes array names, NPY headers (dtype/shape) and numeric bytes in
sorted-name order. Its compressed file has a separate SHA256 integrity checksum.
Before writing an existing key, the exporter verifies and reuses its content.
Hashing does not itself prevent downloads: a downloader must use these references
and its local cache. Bundling all mesh files into each independent clip archive
would still duplicate assets, even if their hashes were known.

`poses.npy` is little-endian float32 `[T,D,7]`, each row `(x,y,z,qw,qx,qy,qz)`.
`frames.npy` is int64 `[T]` with consecutive original Blender frame numbers;
reader indices are zero-based. Poses are evaluated source samples, not a physics
continuation state. Quaternion signs are made continuous without changing rotation.
A **dynamic node** is one independently recorded rigid world transform, indexed
in the D axis. Objects reference a node index, or `-1` and their constant pose.
Several parts can share a node if their mesh coordinates use that node's frame;
the current exporter emits one node per moving source mesh object rather than
guessing which parts are mechanically connected.

The JSON manifest records:

- `schema` and `conventions`: how to interpret arrays, axes, units and quaternion order.
- `source.commit`, `script`, script/task-manifest hashes: identify the exact source definition.
- `source.profile`, `parameters`, `seed`: distinguish nominal export from randomized samples;
  these audited nominal recordings use no diversity parameters or random seed.
- `source.versions` and exporter hash: track the evaluation and extraction implementation.
- `fps`, source frames and source `video_split`: preserve timing and the task's clip boundary.
- `objects`: names, stable positive mask IDs, mesh references, node mapping, static poses and colors.
- `camera`: fixed world/projection matrices, original image dimensions and clipping planes.
- `assets` and `arrays`: checksums plus trajectory shape/dtype; detect corruption or mismatched data.
- Source URL and license: preserve attribution; `license_ref` binds the shared license text checksum.

Assets live in source-shape folders such as `sphere`, `cube` and `cylinder`.
If an old recording has no shape label, its source role is used instead; no
shape is guessed from the object name. Object names, source IDs and semantic
roles remain in the manifest. Its `assets` entries directly bind `file` paths
and byte checksums; no global index is needed to load or render a clip.

After a batch of exports, run `python -m scripts.export.object_permanence organize
--output /tmp/recordings` (one command). It moves legacy flat assets into semantic
folders, consolidates per-clip licenses and removes the former derived index.
Run without concurrent organizers. Mesh/pose bytes are unchanged, but manifest
identities change; recreate cursors referencing old identities. Flat-layout and
per-clip-license recordings remain readable. New exports write semantic paths
and shared licenses directly.

`RigidTrajectory.identity` hashes the canonical manifest, thereby binding poses,
assets, camera and metadata. This is stronger than hashing pose bytes alone. A
future replay cursor can reference this identity plus a frame index. The existing
G34Replay State/hash contract is not changed.

## Export and read

Use the existing isolated Python 3.11 environment pinned by
`scripts/testing/requirements-occlusion-oracle.txt` (`bpy==4.4.0`, NumPy 1.26.4).
The production exporter imports no Torch, MuJoCo or candidate renderer. Its source
checkout must match the pin and be clean. It refuses to overwrite a clip.

```bash
.venvs/occlusion-oracle-py3.11/bin/python -m scripts.export.object_permanence export \
  --source-root /tmp/object-permanence-source --output /tmp/recordings --task G34
.venvs/occlusion-oracle-py3.11/bin/python -m scripts.export.object_permanence export \
  --source-root /tmp/object-permanence-source --output /tmp/recordings --task G62
```

If this machine uses the optional shared-library fallback, set `LD_LIBRARY_PATH`
as described in the [G34 environment setup](occlusion-ball.md#local-environment-layout).
The exported dataset is source-derived CC-BY-NC-4.0 data, not MIT library code.

```python
from stable_envs.envs.object_permanence import G34Replay, RigidTrajectory

with RigidTrajectory("/tmp/recordings", "g34-nominal") as clip:
    rows = clip.poses([44])              # owned CPU NumPy float32 [1,1,7]
    geometry = clip.triangles()          # NumPy inputs to nvdiff.Scene.from_triangles
    objects = clip.object_poses([44])    # CPU float32 [1,G,7], including static objects

    # Existing G34Replay remains usable with this nominal scene's full trajectory.
    with G34Replay(clip.poses(list(range(len(clip))))[:, 0]) as replay:
        images = replay.render(replay.advance(44))
```

Opening validates file hashes and scans pose chunks without cloning the whole
trajectory. Selected poses are copied from a read-only memory map. Meshes are
loaded once per reader; `triangles()` assembles geometry instances for the existing
nvdiff API, which still duplicates per-instance geometry in its GPU scene.
The reader and recording do not require CUDA. `RigidSceneRenderer` borrows an
open recording and supplies one shared full-frame CUDA path for all audited scenes:

```python
from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer

with RigidTrajectory("/tmp/recordings", "g62-nominal") as clip:
    with RigidSceneRenderer(clip, size=180) as renderer:
        fields = renderer.render_many([0, 44, 59, 119])
        # CUDA rgb uint8 [4,180,320,3], depth float32 [4,180,320],
        # ids int16 [4,180,320]; all share coverage and object occlusion.
```

Frame indices select recording rows, not simulator State. Static transforms stay
on GPU; only CPU float32 `[B,D,7]` dynamic-node poses are uploaded, then mapped to
object transforms and rotation matrices on GPU. Multiple mesh objects can share
one node. All objects share depth visibility; geometry IDs map to manifest IDs.

With default refined coverage, static triangles' pixel visibility is cached once
per renderer: int64 ordered float64 depth keys and int32 original face IDs.
Dynamic triangles update copies of that seed using the full pass's depth and
face-ID tie rules. This avoids float32 layer-composition errors at contact and
coplanar surfaces. RGB, depth and IDs are generated from the final shared raster.
Static vertex transforms, interpolation and shading still run; this is a coverage
cache, not a cached RGB plate. At 320x180 the depth/ID seed takes 675 KiB, plus
cached static clip vertices and partition indices.

The cache is geometry-based, with no task-name special cases or mover-count
limit. Static clip-vertex or resolution changes rebuild it; static vertices that
vary within a batch use the complete pass. Depth clipping or unavailable Triton
retains the existing scaled fallback. Scaled/native modes do not use this cache.
Immutable topology is required for the renderer's lifetime. `cache_static=False`
retains full drawing and CPU object-pose expansion/upload as a comparison baseline.
Chunking bounds memory without changing sampling. G34's separate public replay,
optimized renderer and frozen fixture remain unchanged.

## Validation and next ports

`scripts/testing/check_object_permanence_export.py` rebuilds the pinned source and
compares every object's reconstructed world triangles and normals, and the camera,
at all 120 frames. It matches geometry independently of face enumeration and does
not call candidate pose/mesh extraction. Its position budget is 2e-6 source units;
normal budget is 3e-6. These serialization checks complement, not replace, depth
MSE/mask IoU. They do not certify source RGB or downstream learning accuracy.

```bash
.venvs/occlusion-oracle-py3.11/bin/python -m scripts.testing.check_object_permanence_export \
  --source-root /tmp/object-permanence-source --dataset /tmp/recordings --task G62
```

Unit tests cover sharing, mapping, ownership, corruption and invalid contracts.
Live integration tests run all five exports and the independent checker when
`SE_OBJECT_PERMANENCE_SOURCE` and `SE_OBJECT_PERMANENCE_BLENDER_PYTHON` are set.
Normal family CI requires no Blender. The G34 source oracle has been extended to 120 frames without changing any
original sample values or renderer interfaces; no production code reads the oracle.

The independent `freeze_object_permanence_g62.py` generates all 120 source depth
and mask frames at 320x180 without importing candidate code. It never overwrites
an existing directory. Float32 depth bits are XORed against frame one before ZIP
compression and reconstructed exactly; this only removes repeated background bytes.
G34 now has a 120-frame v2 oracle; original 21 samples are bit-identical. The shared renderer uses unchanged
MSE <= 1e-8 and per-object mask IoU >= .99, including static objects. All-frame
pose comparisons additionally validate motion hidden from the camera.

| Shared renderer | Source frames | Maximum depth MSE | Minimum visible object IoU |
|---|---:|---:|---:|
| G34 recorded triangles | All 120 | <1e-12 | 1.0 |
| G62 recorded triangles | All 120 | 9.26e-13 | 1.0 |
| G37 curved track | All 120 | 5.93e-13 | 1.0 |
| G38 pendulum | All 120 | 5.30e-13 | 1.0 |
| G64 windowed board | All 120 | 9.42e-13 | 1.0 |

Small candidate recordings under `tests/fixtures/saved/object_permanence_rigid`
let offline CPU/GPU CI exercise the full path without Blender or the user's cache.
They are separate from the independent geometry oracles. Their README describes
regeneration. Cached and uncached paths pass the same all-frame source gates;
additional tests compare all outputs and exercise equal-depth ties, multiple
movers, cache invalidation, and all-static/all-dynamic partitions. Further source
tasks still need audited entry points and their own independent comparisons.

## Preview and upload-inclusive baseline

`benchmarks.object_permanence_recordings` compares cached `shared`, uncached
`shared-full`, and G34's existing specialized paths. `scripts.export.render_object_permanence_preview`
renders the source nominal scene to PNGs in Blender Cycles CPU at 320x180, 16
samples and denoising. This is a native preview with original scene/camera,
not the dataset MP4 render-quality preset. Side-by-side GIFs preserve all 120
frames and their alignment; shader/shadow/color differences are expected.

```bash
# Repeat with --task G62 and a source-g62 output directory.
python -m scripts.export.render_object_permanence_preview --source-root <source> \
  --output <evaluation>/source-g34 --task G34
python -m benchmarks.object_permanence_recordings --recordings <recordings> \
  --output <evaluation>/throughput-rgb.json --source-previews <evaluation> --profile
```

Run previews with Blender's interpreter and GPU measurements with the CUDA profile.
On the RTX 3070 Ti Laptop GPU, Torch 2.14.1+cu130, one CPU thread, 3 warmups and
11 synchronized repeats, historical **uncached** RGB median frames/s at 180x320 were:

| Batch | G34 uncached | G62 uncached | G34 cropped/refined | G34 fused raycast |
|---:|---:|---:|---:|---:|
| 1 | 191 | 127 | 281 | 3,964 |
| 64 | 532 | 226 | 3,582 | 22,850 |
| 256 | 580 | 224 | 4,150 | 23,800 |
| 1024 | 550 | 226 | 4,152 | 25,514 |

Every call includes CPU frame lookup/expansion, pose upload and rendering into
reused CUDA outputs. First draw/JIT/static bake, output allocation, download,
GIF encoding and model training are excluded. Internal temporary allocations
are included. The raycast's one-time plate bake uses native MuJoCo; its steady
state does not. Shaders differ. Default scaled G34 crop reached 13,495 fps at
batch 256, but its two known full-sequence edge gaps prevent treating that setting
as strict 120-frame parity. Refined crop and raycast match that stronger gate.

A same-run comparison with 3 warmups, 7 repeats and batch 256, including pose
upload, measured:

| Scene | Uncached RGB fps | Cached RGB fps | Speedup | Upload bytes/frame, before → after |
|---|---:|---:|---:|---:|
| G34 | 480 | 821 | 1.71× | 140 → 28 |
| G62 | 223 | 591 | 2.65× | 420 → 56 |

```bash
python -m benchmarks.object_permanence_recordings --recordings <recordings> \
  --output <evaluation>/throughput-static-cache.json --backends shared,shared-full \
  --batches 1,64,256 --repeats 7 --profile
# Repeat with --modes rgb,depth,ids for combined outputs.
```

At batch 256, profiling reduced triangle-kernel totals from 221 to 49 ms for G34
and 853 to 98 ms for G62. Overall wall-clock gains are smaller: chunk size 4 still
makes 64 coverage calls and 128 dynamic triangle passes; transforms, full-frame
resolve/interpolation/shading, launches and cache-validation synchronizations
remain. First draw/JIT/cache construction are excluded from steady-state timing
and recorded separately. Results are machine-specific rendering measurements,
not end-to-end training speedups.

Larger memory-bounded chunks, static transformed-vertex reuse, and selectively
shading dynamic pixels are possible later improvements. RGB plates need explicit
appearance/lighting invalidation; moving shadows or transparency need a different
contract. Preserve shared multi-object depth testing and the full-frame comparison
path when adding any further precision or cache optimization.

## Scaling imports before optimization

The sequence is source audit, faithful recording, independent pose/mesh/pixel
checks, then performance work. A rigid object does not imply a fixed camera,
constant opacity/material or fixed object set. Those are correctness capabilities,
not optimizations to defer. The v1 exporter explicitly rejects unsupported cases.

```bash
python -m scripts.export.batch_object_permanence --source-root <source> \
  --output <recordings> --report <plan.json> --tasks all --plan-only
python -m scripts.export.batch_object_permanence --source-root <source> \
  --output <new-recordings> --report <report.json> --tasks G37,G38,G64 \
  --blender-python <isolated-blender-python>
```

The current plan lists all 197 members across 150 groups: five audited nominal
preparations and 192 unaudited members. Unaudited does not mean non-rigid. Execution
runs one isolated process per supported member, performs independent source
triangle checks, and reports failures without reusing contaminated Blender state.
Each job includes a conservative source-only capability review group. The
regenerable inventory separates appearance/visibility, orthographic, geometry/scale,
physics-bake and rigid-entry review queues; unused helpers can overflag a member.
Only explicit preparation audits and all-frame runtime checks establish support.
It never executes unaudited main functions, overwrites clips or silently counts
unsupported jobs as success. Pixel certification remains the per-task oracle suite.

Next, audit source preparation recipes in small groups, key future variants by
(task, member, profile), and add their source comparisons before enabling them.
Extend shared capture/render capabilities for moving cameras, visibility or
materials when required; preserve explicit rejection until they are validated.
Mesh-local geometry is reusable across scenes; sampled poses handle irregular
motion without new motion formulas. Keep units, frame conventions, semantic IDs,
source pins and immutable manifests stable when publishing downloadable datasets.


## First expansion batch: G37, G38 and G64

All three add recording data and source references, without changing the runtime
reader, renderer, precision thresholds or the existing G34/G62 frozen files.

| Scene | Mesh objects | Recorded nodes | Source-visible targets |
|---|---:|---:|---|
| G37 | 5 | 1 | `orange_ball`: completely hidden for 18 frames |
| G38 | 6 | 3 | `red_pendulum_bob`: completely hidden for 20 frames |
| G64 | 5 | 1 | Both stationary targets remain partially visible through a board window |

G38 records rod and bob motion, and conservatively retains the stationary pivot
because the source marks it dynamic. No movement formula is reconstructed.
G64's actual Boolean-cut window disagrees with source prose/custom tags claiming
complete hiding. Preserve the evaluated mesh and its actual partial occlusion;
do not close the window or require a hidden interval that the source lacks.
G43/G77 are deferred because their selected cameras are orthographic. An applied
Boolean modifier such as G64's is already supported when its evaluated mesh stays
constant; the presence of a modifier alone is not evidence of deformation.

`scripts/testing/freeze_object_permanence.py` has an independent audited entry
list and queries Blender rays/world poses directly, importing no candidate
exporter or renderer. References live under
`tests/fixtures/semantics/object_permanence/<task>/nominal/`; candidate input clips
and reusable meshes live separately under `tests/fixtures/saved/object_permanence_rigid/`.
The main reference NPZ stores poses, source frames, camera, masks and depth base.
Four 30-frame `depth_*.npz` files hold lossless uint32 XOR depth bits, each hashed
in the manifest. This keeps files below the repository size limit without changing
any depth bit, frame or acceptance threshold. The old G34/G62 encodings stay readable.

```bash
# Use the Blender reference interpreter; repeat for G38 and G64.
python -m scripts.testing.freeze_object_permanence --source-root <source> \
  --output <new-oracle>/g37/nominal --task G37
python -m scripts.export.render_object_permanence_preview --source-root <source> \
  --output <evaluation>/source-g37 --task G37
```

Use the CUDA interpreter for the existing comparison helper (no timing run needed):

```python
from pathlib import Path
from benchmarks.object_permanence_recordings import comparison_gif
from stable_envs.envs.object_permanence import RigidTrajectory

for task in ("g37", "g38", "g64"):
    with RigidTrajectory("<recordings>", f"{task}-nominal") as clip:
        comparison_gif(clip, Path(f"<evaluation>/source-{task}"),
                       Path(f"<evaluation>/{task}-source-vs-shared.gif"), size=180)
```

Tests check every frame's world pose, including hidden motion, and depth/masks for
all mesh objects on both cached and uncached paths. Exact hidden counts come from
the independent source masks; G64 explicitly tests partial visibility instead.
Native preview GIFs show original source lighting beside simplified shared RGB;
they are visual aids, while depth MSE and per-object mask IoU are the geometry gates.
