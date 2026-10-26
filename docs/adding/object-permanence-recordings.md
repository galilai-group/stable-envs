# Object-permanence recordings

This family records evaluated source animation once, then renders selected frames
on the GPU during training. Users store numeric poses and shared meshes rather
than rendered videos. Online replay does not need Blender or MuJoCo.

The source is pinned to
[`hokindeng/object-permanence@824badd`](https://github.com/hokindeng/object-permanence/tree/824baddcdad087810d35638bf8f273b884f74975).
The importer has been exercised on all 197 source scripts: 150 first nominal
members and 47 additional members. This describes the audited source defaults,
not arbitrary driver diversity or downstream model accuracy. G05’s remaining
members reuse the existing opaque and positive-scale contracts; no nominal
source members remain deferred.

## Read and render

```python
from stable_envs.envs.object_permanence import RigidTrajectory
from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer

with RigidTrajectory("recordings", "g34-nominal") as clip:
    with RigidSceneRenderer(clip, size=180) as renderer:
        fields = renderer.render_many([0, 44, 119])
        # CUDA: rgb uint8 [3,180,320,3], depth float32 [3,180,320],
        # ids int16 [3,180,320]. All outputs share object visibility.
```

G34 uses this same path as every other recorded scene; change the clip identifier
(e.g. `g62-nominal`) to select another task.

Frame indices select recorded rows. These are source-animation poses, not
checkpoints for action-driven physical simulation. A ball retains its pose while
occluded; the renderer determines whether it is visible. `render(..., mode="rgb")`
and the existing single-output buffer interface remain available.

The reader uses read-only memory maps and returns owned NumPy selections.
Meshes and static transforms are reused on GPU; each call uploads the selected
dynamic poses and optional scales. Fixed-camera static visibility is cached,
then combined with moving objects using shared depth and semantic-ID rules.
Changing the camera or relevant geometry invalidates the cache.

## Dataset layout

```text
recordings/
  assets/<shape>/<content-hash>.npz
  licenses/<license-hash>.txt
  clips/g62-nominal/
    manifest.json
    frames.npy
    poses.npy
```

The manifest binds source/script hashes, camera, object IDs, mesh references,
materials, static poses, numeric-array hashes and coordinate conventions.
Assets contain local `float32[V,3]` vertices/normals and `int32[F,3]` triangles.
Object names and roles describe semantics; hashes identify exact reusable geometry.
Colors belong to instances, so differently colored balls can share one mesh.

`poses.npy` is `float32[T,D,7]`, storing world-space xyz + quaternion wxyz for D
dynamic nodes. `frames.npy` is `int64[T]` with original source frame numbers.
Objects reference a dynamic node or use their constant manifest transform.
Parent motion, constraints and animation are evaluated in Blender before export;
no source motion formula is reimplemented at runtime.

| Recording | Additional capability | Rendering policy |
|---|---|---|
| v1 | Fixed perspective camera, rigid opaque objects | Opaque pixel rays v1 |
| v2 | Fixed orthographic camera | Opaque pixel rays v2 |
| v3 | Optional positive `float32[T,D,3]` scales | Explicit opaque policy |
| v4 | Constant alpha materials | Ordered transparent surface layers |
| v5 | Exact `float64[T,D,4,4]` transforms | Deterministic sampled dielectric paths |

Curve/FONT objects are converted to evaluated triangle meshes offline, retaining
source tessellation, open ends and text identity. Deformation, actual instancing,
changing object membership/materials and moving cameras are rejected. Permanently
render-hidden nonsolid annotations are omitted; a marker becoming visible still
changes the rendered object set and is rejected. An object hidden behind another
surface remains recorded. Scale clips require the renderer’s explicit
`visibility_contract=clip.manifest["visibility_contract"]`.
Optical v5 uses ideal Snell/Fresnel events, a fixed sample stream and one common
IOR; it does not reproduce Blender's complete BSDF, roughness, shadows or caustics.
The contracts are defined in `rendering/visibility.py`, `transparency.py` and
`optics.py`; the reader rejects incompatible policy/data combinations.

## Import and independently validate

Use a clean checkout at the source pin and an isolated Blender interpreter with
`scripts/testing/requirements-object-permanence-oracle.txt` (Blender 4.4). Use a separate
CUDA interpreter with Torch, Triton and nvdiffrast for rendering.
`member` is a zero-based position in the source task's `member_scripts` list;
`nominal` selects that script's audited defaults.

```bash
python -m scripts.export.batch_object_permanence \
  --source-root <pinned-source> --tasks G13 --members 0 \
  --output .cache/object_permanence/corpus/recordings \
  --oracle-root .cache/object_permanence/corpus/oracles \
  --report .cache/object_permanence/g13-report.json --stage all \
  --visibility-contract stable-envs/opaque-pixel-rays-v2 \
  --blender-python <blender-python> --render-python <cuda-python>
```

The stages are `export`, `oracle`, `validate` and `all`. Export and oracle refuse
existing destinations; use `validate` to recheck immutable inputs. A plan with
`--plan-only` executes nothing. Additional members require an explicitly audited
entry and matching contract, never an assumption based on another member.

Audited multi-stage sources execute their original `main()` construction prefix
before the first render call, skipping directory setup and output operations.
Source scripts are unchanged; additional members explicitly opt into scale
sampling where needed.

The exporter checks every source frame for constant geometry, valid transforms,
materials and camera. The independent oracle separately evaluates original source
triangles; it does not consume exported poses or candidate intersection code.
Versioned oracles coexist with legacy captures. Their explicit semantic IDs
preserve recorded labels and determine coplanar ties.
Validation compares every frame and every object, then checks cached/full outputs
across repeated and shuffled frames, chunk sizes and caller mutation.

| Check | Acceptance budget |
|---|---:|
| Depth MSE | <= 1e-8 |
| Visible-object mask IoU | >= 0.99 |
| Source world-vertex / normal error | <= 2e-6 / 3e-6 |
| Cached/full RGB, depth and IDs | Bit-identical |

RGB shader differences are expected. Geometric agreement tests task-relevant
occlusion, not equivalence of original video pixels or downstream learning.
G98/G132 additionally use hash-pinned native `.blend` inputs under
`tests/fixtures/source/object_permanence`: repeated procedural shell builds can
change normals. These small source inputs are needed to reproduce that declared
realization; online users do not need them.

The complete corpus, audit reports and GIF-generation intermediates are developer
data, kept outside Git. CI keeps recordings for G34, G62, G13 and G146, their
referenced meshes, and representative independent geometry oracles.
Default CI tests core behavior; complete scene/member certification uses the CLI
above. Runtime downloads need manifests, referenced assets/licenses and motion
arrays, not developer oracles, source snapshots or GIFs. No automatic downloader
or hosted full-corpus URL is currently provided; export from the pinned source
or use a separately distributed recording bundle.

## Visual checks and timing

Generate native source previews into a temporary directory with
`scripts.export.render_object_permanence_preview`. Combine them with shared
rendering through `benchmarks.object_permanence_recordings.comparison_gif`,
then delete temporary PNGs after verifying the GIF. Previews use Cycles at reduced
quality; they are not decoded original MP4 frames.

The PR examples show the source on the left and replay on the right:
[G34](../assets/media/object-permanence/g34.gif),
[G62](../assets/media/object-permanence/g62.gif),
[G14](../assets/media/object-permanence/g14.gif).

```bash
python -m benchmarks.object_permanence_recordings --recordings <dataset> \
  --output .cache/object_permanence/throughput.json \
  --backends shared,shared-full --batches 1,64,256 --modes rgb,depth,ids --profile
```

Measurements include CPU frame lookup and pose upload. Startup/static bake,
output download and training are separate costs; rendering throughput is not an
end-to-end training speedup.
