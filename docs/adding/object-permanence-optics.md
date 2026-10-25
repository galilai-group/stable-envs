# Sampled dielectric replay

G01's covered-hole and open-hole members, and G03's additional transparent-wall
member, use the explicit `stable-envs/sampled-optical-rays-v1` policy. This is
ideal surface optics, with simplified shading. It is **not** a reproduction of
Blender's complete Principled BSDF, rough scattering, shadows or caustics.
The pinned source actually selects EEVEE Next with scene ray tracing disabled,
although its materials contain nonzero transmission and IOR. Matching this
independently specified geometric oracle therefore does not establish native
EEVEE/Cycles pixel equivalence or downstream learning accuracy.

G01 has transmission 0.25, IOR about 1.45 and alpha 0.28/0.42; G03's extra member
uses transmission about 0.18 and the same IOR. These are sampled from source
materials, not substituted with alpha-only glass. Existing opaque/alpha policies,
G34 State, interfaces, fixtures and acceptance thresholds remain unchanged.

## Data and precision

Recording v5 selects `constant-dielectric-surfaces-v1` and the optical policy.
Each object stores constant `optics` (`transmission`, `ior`, `roughness`) and its
exact initial `transform`. Vertices/normals remain float32 local source data and
triangles int32. For this version, initial scale stays in the transform rather
than being baked and rounded into vertices. Existing content hashes, semantic
asset directories and shared license files still apply.

The familiar float32 `[T,D,7]` `poses.npy` remains available with the same xyz+wxyz
layout. A new float64 `[T,D,4,4]` `transforms.npy` additionally preserves the exact
evaluated source matrix coefficients, including scale. `RigidTrajectory.transforms`
returns owned CPU rows selected from a hash-checked, read-only memory map.
Blender coefficients can already be float32; float64 storage/compute does not
invent extra source precision. Static matrices live once in the manifest. Dynamic source matrices are the
optical renderer's authoritative input; it does not reconstruct them from the
compatibility pose array. This prevents touching glass panels from separating
through pose decomposition or scale-baking roundoff.

For one moving object, those arrays contain 28 + 128 bytes per frame before
headers, versus 172,800 bytes for one raw 320x180 RGB frame. Assets/camera/materials
are one-time data. This is a raw-size comparison, not a measured MP4 compression
or training benchmark. Users need the recordings' manifests, referenced assets,
license and numeric arrays; developer oracles, GIFs and Blender are unnecessary
for online replay.

## Optical events and outputs

Eight deterministic samples share each pixel centre. They sample **surface
events**, not subpixel coverage. The counter stream depends on pixel, sample and
event index, never frame order, batch or cache history. At a hit with alpha `a`
and transmission `t`, the probabilities are:

- `1-a`: null/straight passage;
- `a*(1-t)`: terminate with simple diffuse surface color;
- `a*t`: dielectric interaction, choosing exact unpolarized Fresnel reflection
  or Snell transmission; total internal reflection forces reflection.

The model tracks membership in closed dielectric volumes, using their common
IOR for the occupied union. Null crossings update geometric membership too;
reflection does not cross a boundary. Exact equal-depth junctions select the
lowest semantic ID and original face, consistently in full/cache/reference
queries. More than one IOR, more than 62 objects, nonclosed dielectric meshes,
metallic/coated materials, linked/changing parameters, deformation and moving
cameras are rejected. Positive source transforms and fixed perspective cameras
are supported. Roughness is retained as provenance but **not executed** by this
ideal model. There is no spectral or volumetric absorption.

Intersections use float64 arithmetic and canonical planes. A `1e-9` source-unit
secondary ray-start interval avoids self-hits; it is part of this new policy,
not a silhouette mask adjustment. Geometry at smaller separations requires a
new policy. Each sampled path must terminate within 64 events; overflow raises
instead of silently dropping a live ray. The fixed sample count/seed/event limit
are bound in recording and oracle metadata; changing sampling requires explicit
versioning and fresh independent references.

| Mode | CUDA dtype and shape | Meaning |
|---|---|---|
| `rgb` | uint8 `[B,H,W,3]` | Mean sampled terminal colors |
| `depth` | float32 `[B,H,W]` | First primary surface's camera-axis depth |
| `ids` | int16 `[B,H,W]` | That surface's semantic ID |
| `masks` | bool `[B,G,H,W]` | Object visited by at least one of the fixed samples |
| `path_depth` | float32 `[B,G,H,W]` | Minimum cumulative geometric path distance to that object |

Zero means no hit. `path_depth` after a bend is not camera-axis depth or optical
length multiplied by IOR. Sampled masks are not the union of every physically
possible light path, nor a perceptual visibility threshold. Flat alpha-layer
modes remain v4-specific: a single layer list cannot describe branching optics.
The default `render`/`render_many` RGB/depth/ID entry points remain the same.

```python
from stable_envs.envs.object_permanence import RigidTrajectory
from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer

with RigidTrajectory(recordings, "g01-member1") as clip:
    with RigidSceneRenderer(
        clip, visibility_contract="stable-envs/sampled-optical-rays-v1"
    ) as renderer:
        fields = renderer.render_many([0, 44, 119], modes=("rgb", "masks"))
```

## Reusable execution and cache

`optics.py` defines volume validation and local BVHs (bounding-volume
hierarchies). `optics_cuda.py` performs stackless arbitrary-ray queries against
resident local meshes with inverse source transforms. `optical_scene.py` drives
material events and outputs. No Blender, MuJoCo, nvdiffrast or per-frame graphics
context is needed by this path; rasterization alone cannot follow bent rays.
The existing `nvdiff.py` implementations and their callers are unchanged.

The correctness baseline traces the complete scene. For immutable static assets,
camera and materials, a cache records static paths' origins, directions,
distances, hit keys and prefixes, plus terminal sample colors. If every mover is
opaque, it can only terminate a cached path earlier. Each frame then queries
all movers against those segments, bounded by the next static hit distance, and `optical_cache_cuda.py` fuses hit ordering,
RGB replacement, primary outputs and optional path-mask accumulation. Static
paths are not flattened into a background RGB/depth plate. Multiple opaque
movers share the same query/ordering rules. Moving glass safely uses full tracing.
Recreate the renderer for a different immutable scene/material/camera.

Primary samples share a ray, so the cached dynamic query traces it once per
pixel. The retained camera origin is float64 `[1,3]` and directions `[H*W,3]`;
zero-stride origin views are read directly by the query kernel. The first static
segment shares these buffers, rather than retaining eight copies per pixel.
Subsequent bent segments retain independent per-sample origins/directions.
`StaticSegment` stores sample IDs, origins/directions, next-hit distances/faces
and cumulative distances; it does not retain surface normals. The fused resolve
uses only newly queried dynamic normals. Diagnostic `fuse_cache=False` reconstructs
static normals transiently from local triangles and inverse transforms, using
the query kernel's exact arithmetic order; it does not rebuild or enlarge the cache.
The eight samples, event counter, float64 hits and semantic ties are unchanged.

Variable active-ray counts are runtime kernel arguments, avoiding a new
Triton compilation for every different frame. Diagnostic `fuse_cache=False` on
the owned optical renderer retains the Torch cache comparison. `cache_static=False`
retains full tracing. Diagnostic `use_ray_limits=False` preserves unbounded
cached queries, including exact equal-depth ties. Output allocation is bounded to requested fields, caller
buffers are supported, and unrequested dense object masks are not retained.
The current optical implementation processes frames sequentially within a public
batch; `chunk_size` compatibility is retained, but does not imply simultaneous
frame tracing. Batch-size scaling remains a future optimization.

Static transport is a compute/memory tradeoff, not an image-size reduction:
records cover multiple surface events per sample, and the terminal seed retains
sample RGB/depth/IDs before averaging. Downloads still contain compact CPU states
and shared assets. A caller may instead render a fixed clip in small frame batches,
retain its GPU uint8 RGB, then close the renderer; this does not require uploading
the whole trajectory at once. Cached pixels must be regenerated if camera,
resolution, material or rendering policy changes. Cross-scene GPU sharing and a
byte-budgeted renderer/pixel cache are not currently automatic.

## Reproduction and acceptance

Use the existing isolated Blender 4.4.0 / Python 3.11 / NumPy 1.26.4 environment
for offline work. The ordinary CUDA replay environment remains unchanged.
Use the [recording guide's native-library fallback](object-permanence-recordings.md)
when this machine needs `LD_LIBRARY_PATH` before importing `bpy`.

Known packaging detail: the official Linux `bpy 4.4.0` wheel is named `cp311`
([PyPI release files](https://pypi.org/project/bpy/4.4.0/#files)), but its internal
`WHEEL` tag says `cp39`. The installed metadata matches the official artifact
byte-for-byte; `uv pip check` consequently reports a platform mismatch. Do not
retag the package or switch to Python 3.9 to silence that report. Functional
source checks use the required Python 3.11 and pass; the project lock check is
separate. No source dependency pin or installed wheel metadata was changed.

```bash
# --task defaults to G01; member 0 is its nominal member, member 1 its open hole.
python -m scripts.export.object_permanence_optical --source-root <source> \
  --output <new-recordings> --task G01 --member 1
python -m scripts.export.object_permanence_optical --source-root <source> \
  --output <new-recordings> --task G03 --member 1
python -m scripts.testing.freeze_object_permanence_optical --source-root <source> \
  --output <new-oracle> --task G01 --member 1
# CUDA environment; repeat with --uncached for the full baseline.
python -m scripts.testing.check_object_permanence_optical --recordings <recordings> \
  --oracle <oracle> --report <report.json>
python -m benchmarks.object_permanence_optical --recordings <recordings> \
  --report <benchmark.json> --profile <profile.txt>
# Measure only the production cache, with alias-aware retained storage counts.
python -m benchmarks.object_permanence_optical --recordings <recordings> \
  --variants static-fused --batches 1 16 --repeats 7 --report <cache-benchmark.json>
```

The independent source oracle never imports runtime extraction, recording arrays
or CUDA hit code. It evaluates original world triangles and materials, uses a
separate recursive world-space BVH and plane/barycentric implementation, and
checks source static geometry/camera/material invariance on every frame. Its
counter stream and optical equations are independent implementations of the
specified policy. Hashes bind source, execution recipe, materials, camera,
objects, frames, policy and generator versions. Lossless XOR parts preserve all
120 frames alongside every previous frozen sample. Depth, IDs and bit-packed
masks use bounded lossless XOR parts; no new fixture exceeds 1,000 KiB.

Acceptance uses the existing depth MSE `<=1e-8`, mask IoU `>=.99`, world-vertex
`2e-6` and corner-normal `3e-6` limits. Primary depth and each object's path depth
are checked; union masks penalize missing paths in depth too. Tests also require
bitwise RGB/depth/ID/mask/path-depth equality at 90p and 180p, all frames,
shuffled/repeated batches, single modes, caller buffers and one retained cache.
Synthetic cases cover multiple movers, moving-glass fallback, analytic Snell/TIR,
Fresnel choices, arbitrary rays and semantic-ID ties.

For optional native source GIF previews, install Pillow into the source
interpreter (`uv pip install --python <source-python> Pillow==13.0.0`). Source
preview PNGs reuse one temporary file and are removed automatically. On machines
with software OpenGL, `--preview-samples 8` reduces only the qualitative EEVEE
preview's AA sampling; it does not change source files, oracle geometry or replay.

```bash
python -m scripts.testing.object_permanence_optical_preview source \
  --source-root <source> --task G01 --member 1 --preview-samples 8 --output <source.gif>
python -m scripts.testing.object_permanence_optical_preview compare \
  --recordings <recordings> --clip g01-member1 --source-gif <source.gif> --output <comparison.gif>
```

Only final comparison GIFs belong in the evaluation cache. Benchmark reports
belong separately under `.cache/benchmarks/object_permanence/optics/`. Throughput
includes CPU frame selection, exact-matrix upload and CUDA synchronization;
initialization/first-call cost and resident/peak allocation are reported
separately. These measurements do not compare MP4 decoding or training speed.

The earlier, pre-compaction implementation measured on RTX 3070 Ti Laptop,
Torch 2.14.1+cu130, 320x180 RGB, batch 16, median of three synchronized samples
after warmup (same frame indices per variant):

| Clip | Full FPS | Torch cache FPS | Fused unbounded FPS | Fused bounded FPS | Full→bounded |
|---|---:|---:|---:|---:|---:|
| G01 covered hole | 1.93 | 10.17 | 45.85 | 56.09 | 29.04x |
| G01 open hole | 1.92 | 8.01 | 26.33 | 40.87 | 21.31x |
| G03 transparent wall | 8.12 | 16.85 | 144.83 | 154.15 | 18.99x |

Those full/bounded peak PyTorch allocations were 125.94/155.46, 127.09/159.10 and
88.32/95.96 MiB respectively. These include persistent tensors, not total driver
memory. Static transport raises resident allocation from about 30–31 MiB to
70–125 MiB. Fusion lowers peak allocation relative to the Torch cache, but the
cache still costs more resident memory than full tracing. Compact downloaded
states save dataset storage; a faster online renderer does not automatically
mean a smaller GPU cache. Different frame sets explain some batch FPS variation;
this backend currently traces frames sequentially.

Compacting primary rays and dropping retained static normals changes the production
cache as follows. The same machine, frame indices and upload-inclusive calls use
seven synchronized samples after warmup, with the original and new implementations
loaded in separate processes. MiB is 1,048,576 bytes; storage counts shared tensor
allocations once, whereas resident/peak include PyTorch workspace and block rounding.

| Clip | Retained storage MiB before→after | Resident MiB before→after | Batch-16 peak MiB before→after | Batch-16 FPS before→after |
|---|---:|---:|---:|---:|
| G01 covered hole | 108.78→67.78 | 123.92→83.39 | 156.17→116.95 | 57.43→57.67 |
| G01 open hole | 111.84→70.08 | 124.70→83.83 | 159.10→118.76 | 41.64→41.30 |
| G03 transparent wall | 62.10→31.64 | 70.24→39.78 | 95.96→61.84 | 162.13→155.03 |

Primary ray storage drops from 21.53 to 1.76 MiB in all three clips. G01's static
normal buffers previously used another 21.23/21.99 MiB. Terminal RGB/depth/ID samples
still use 14.94 MiB; bent path records remain necessary. This is a memory reduction,
not a throughput speedup claim: traversal work is unchanged, G01 FPS is nearly
flat, and the G03 sample is 4.4% slower (the original pre-change run was 155.75 FPS).
All 120 frames preserve exact full/cache outputs and source depth MSE 0 / mask IoU 1.

For open-hole G01, earlier profiling of 16 frames attributes 97.63% of CUDA time to BVH
queries and 0.81% to fused resolve. Bounds reduce query time from 556.6 to
346.9 ms in the sampled profile without changing its 240 query calls. Remaining
work should target conservative BVH/ray-packet traversal and frame batching;
state upload and composition are not the main device costs. Preserve float64
final hits and semantic ties when investigating faster broad-phase arithmetic.
