# Rigid object-permanence recordings

The next porting unit is an exported scene and its sampled motion, not a new
physics adapter per task. The source is pinned to
[`824baddcdad087810d35638bf8f273b884f74975`](https://github.com/hokindeng/object-permanence/tree/824baddcdad087810d35638bf8f273b884f74975).
The [source inventory](object-permanence-inventory.md) covers all 150 task groups
and 197 member scripts. Its flags are **static, script-wide signals**, including
unused builders; they are not a claim that every nominal/diversity variant works.

## Pilots and supported capabilities

G34 is the existing single ball behind a screen. The second pilot is
[G62 two balls cross tunnel](https://github.com/hokindeng/object-permanence/blob/824baddcdad087810d35638bf8f273b884f74975/object_permanence/generator/tasks/G62_two_balls_cross_tunnel/pb_task_00000108_pb_two_balls_cross_tunnel_0109_blender.py):
two balls travel in opposite directions on parallel lanes through an opaque
tunnel. Both have stored poses while hidden; their IDs and colors remain distinct.
Its 120-frame, 24 FPS nominal scene has 15 mesh objects and two dynamic nodes.
Both pilots use the unmodified first member's build/animate functions, bypassing
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
  these pilots use no diversity parameters or random seed.
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
open recording and supplies a shared full-frame CUDA path for both pilots:

```python
from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer

with RigidTrajectory("/tmp/recordings", "g62-nominal") as clip:
    with RigidSceneRenderer(clip, size=180) as renderer:
        fields = renderer.render_many([0, 44, 59, 119])
        # CUDA rgb uint8 [4,180,320,3], depth float32 [4,180,320],
        # ids int16 [4,180,320]; all share coverage and object occlusion.
```

Frame indices select recording rows, not simulator State. CPU poses are expanded
through node/object mappings, uploaded, and converted to rotation matrices on GPU.
All objects share the depth buffer; geometry IDs are remapped to manifest mask IDs.
Recorded camera matrices are reused. Chunking bounds memory without changing
sampling. The default refined pixel-center coverage reuses existing nvdiff helpers.
This is a fidelity baseline with full-frame drawing, not the cropped/cached G34
optimization. It introduces no G62 physics Env or dummy actions. G34's public
renderer/replay interfaces, optimized path and frozen fixture remain intact;
its unchanged quaternion helper is now shared by both renderers.

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
Live integration tests run both exports and the independent checker when
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

Small candidate recordings under `tests/fixtures/saved/object_permanence_rigid`
let offline CPU/GPU CI exercise the full path without Blender or the user's cache.
They are separate from the independent geometry oracles. Their README describes
regeneration. Next, measure the baseline and enable background/window caching only
for scenes meeting its requirements; further source tasks still need audited entry
points and their own independent comparisons.

## Preview and upload-inclusive baseline

`benchmarks.object_permanence_recordings` compares the shared full-frame baseline
with G34's existing specialized paths. `scripts.export.render_object_permanence_preview`
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
11 synchronized repeats, RGB median frames/s at 180x320 were:

| Batch | G34 shared | G62 shared | G34 cropped/refined | G34 fused raycast |
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

The shared renderer uploads 140 bytes/frame for G34 and 420 for G62 (including
expanded static object poses). The specialized paths upload 28. At batch 256,
profiling attributed about 79%/93% of G34/G62 self CUDA time to the `_triangles`
precision kernel. Shared chunk size 4 produces 64 coverage calls and 128 triangle
passes; its clipping guard also synchronizes. These measurements support fixing
repeated full-frame/static coverage and launch overhead before optimizing the
small pose transfer. This baseline is deliberately slower than scene-specific
paths and is retained to certify future optimizations.

Candidate future optimizations are shared static depth/RGB/ID plates where camera
and appearance are fixed; GPU-resident static transforms with only D dynamic
poses uploaded; larger memory-bounded chunks; and conservative native coverage
with sparse precision correction. Multiple movers must share depth visibility,
including overlapping windows. Keep the full-frame fallback and test every
precision/cache change against the same source oracle. No new performance path
is enabled by these measurements.

## Scaling imports before optimization

The sequence is source audit, faithful recording, independent pose/mesh/pixel
checks, then performance work. A rigid object does not imply a fixed camera,
constant opacity/material or fixed object set. Those are correctness capabilities,
not optimizations to defer. The v1 exporter explicitly rejects unsupported cases.

```bash
python -m scripts.export.batch_object_permanence --source-root <source> \
  --output <recordings> --report <plan.json> --tasks all --plan-only
python -m scripts.export.batch_object_permanence --source-root <source> \
  --output <new-recordings> --report <report.json> --tasks G34,G62 \
  --blender-python <isolated-blender-python>
```

The current plan lists all 197 members across 150 groups: two audited pilot
preparations and 195 unaudited members, not 195 proven non-rigid scenes. Execution
runs one isolated process per supported member, performs independent source
triangle checks, and reports failures without reusing contaminated Blender state.
It never executes unaudited main functions, overwrites clips or silently counts
unsupported jobs as success. Pixel certification remains the per-task oracle suite.

Next, audit source preparation recipes in small groups, key future variants by
(task, member, profile), and add their source comparisons before enabling them.
Extend shared capture/render capabilities for moving cameras, visibility or
materials when required; preserve explicit rejection until they are validated.
Mesh-local geometry is reusable across scenes; sampled poses handle irregular
motion without new motion formulas. Keep units, frame conventions, semantic IDs,
source pins and immutable manifests stable when publishing downloadable datasets.
