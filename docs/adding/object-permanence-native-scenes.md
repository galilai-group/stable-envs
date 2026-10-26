# Reproducible native source scenes

G98 and G132 use one hash-pinned native Blender realization as their source input.
All 150 first nominal members are now certified within their declared scope;
23 additional member scripts remain outside the certified set (including two
explicit G05 capability deferrals).
No existing recording, oracle, visibility contract or threshold is replaced.

## Why the source input is pinned

Both source scripts apply Solidify to a cut UV-sphere shell. Fresh source builds
have the same canonical triangle connectivity, but shell coordinates can differ
by about `6e-8` and corresponding corner normals by up to `1.54e-5`. A single-CPU
probe does not remove the variation. The latter exceeds the existing `3e-6`
normal budget; selecting a lucky comparison or rounding normals is not a fix.

`capture_object_permanence_scene.py` runs the unmodified pinned source's
clear/build/animate calls once, then saves its native mesh and animation in a
compressed `.blend`. No candidate arrays, replacement normals, motion formulas
or source rewrites enter the capture. Two files of about 247/266 KiB live under
`tests/fixtures/source/object_permanence/{g98,g132}/nominal/`, with manifests and
CC-BY-NC-4.0 attribution. Manifests bind source/script hashes, capture generator,
Blender version, frames and the entire scene-file checksum.

This certifies **one actual native realization**, not equivalence of every fresh
procedural rebuild. Capturing again creates a new identity and needs fresh
recordings/oracles; capture refuses existing output. Old deferral evidence remains
in the audit history and diagnostic GIFs.

## Independent reads and online replay

The exporter validates the source pin, script/task hashes and scene checksum,
loads the native scene with embedded scripts disabled, then samples all frames
through the ordinary rigid exporter. The independent oracle separately validates
and opens the same source artifact; it imports no candidate extraction or runtime
geometry. World-position/normal checks also reload it in fresh processes.
The scene realization's file and manifest hashes are bound into both recording
and oracle metadata. A mismatch is rejected before pixel comparison.

Online users download only the usual manifests, shared numeric assets and pose
arrays. They do not need the developer source `.blend`, Blender or MuJoCo.
G98/G132 retain float32 `[T,D,7]` poses and the existing opaque pixel-ray V1 policy.
Both have two declared dynamic nodes, preserving source object tags even where
a tagged ball's sampled pose happens to remain constant.

Three independent 120-frame world checks give identical results:

| Scene | Maximum world-vertex error | Maximum corner-normal error | Depth MSE | Minimum mask IoU |
|---|---:|---:|---:|---:|
| G98 | 0 | 3.774e-8 | 0 | 1.0 |
| G132 | 2.087e-7 | 3.272e-7 | 7.688e-16 | 1.0 |

Existing budgets stay `2e-6` vertices, `3e-6` normals, depth MSE `1e-8`, mask IoU
`.99`. Full/cached and original/fused outputs match bitwise at 90p and 180p,
including shuffled/repeated frames and chunk sizes 1/3/4/5, with one static seed.

## Shared geometry preparation

`visibility_transform_cuda.py` fuses fixed-order float64 world-vertex, canonical
plane and camera transforms without allocating a full object matrix per vertex
or face. The shading path retains the original sparse float32 arithmetic, but
omits an unused projection pass and its retained float64 sparse matrix. This
applies to opaque V1/V2, positive scales and constant alpha, without scene-name
branches. Diagnostic `renderer.renderer.fuse_geometry=False` retains the original
Torch implementation for exact comparisons. Intersection, ID ties and static-cache
invalidation rules are unchanged; `nvdiff.py` and optical tracing are untouched.

Measured on RTX 3070 Ti Laptop, Torch 2.14.1+cu130, 320x180 RGB, batch 16,
median of seven synchronized calls after warmup. Calls include CPU selection,
pose upload and output allocation; only one renderer is retained at a time.

| Scene | Resident MiB original→fused | Peak MiB original→fused | FPS original→fused |
|---|---:|---:|---:|
| G98 | 21.37→20.42 | 59.76→56.89 | 421.40→476.81 |
| G132 | 26.21→23.61 | 65.87→61.51 | 436.08→560.19 |

These are PyTorch allocations, including common workspace, not total driver
memory or training benchmarks. Single-frame G98 changes 109.44→105.88 FPS;
G132 changes 124.63→139.87 FPS. The optimization does not accelerate every case.

## Reproduction

Use the existing Blender 4.4.0 reference interpreter and native-library fallback.
Run capture into a **new** directory; copy a deliberately selected realization
into the source-fixture location before freezing a new dataset identity.

```bash
python -m scripts.testing.capture_object_permanence_scene --source-root <source> \
  --output <new-native-scene> --task G98
python -m scripts.export.object_permanence export --source-root <source> \
  --output <new-recordings> --task G98
python -m scripts.testing.freeze_object_permanence_rays --source-root <source> \
  --output <new-oracle> --task G98
python -m scripts.testing.check_object_permanence_rendering --recordings <recordings> \
  --oracle <oracle> --task G98 --visibility-contract stable-envs/opaque-pixel-rays-v1 \
  --report <report.json>
python -m benchmarks.object_permanence_transforms --recordings <recordings> \
  --tasks g98,g132 --batches 1,16 --report <benchmark.json>
```

Native comparison previews use the same snapshot, source materials and 16-sample
Cycles GPU rendering; they are qualitative views, not the original MP4 pixels.
`render_object_permanence_preview --device GPU` explicitly requests CUDA; its
existing CPU default remains unchanged. Comparison GIFs retain all 120 frames;
temporary PNGs are deleted after assembly. Reports belong under
`.cache/benchmarks/object_permanence/source-realizations/`, final GIFs under
`.cache/object_permanence/evaluation/`.
