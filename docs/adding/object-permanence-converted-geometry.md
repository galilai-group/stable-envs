# Evaluated Curve/FONT geometry

The six remaining Curve/FONT first nominal members use the same rigid recording
and shared renderer. Conversion is offline in Blender 4.4; online replay reads
ordinary triangle assets and sampled poses, without Blender, MuJoCo or font files.
The existing opaque pixel-ray contracts, defaults and thresholds are unchanged.

| Task | Source geometry converted | Identity/occlusion role | Visibility |
|---|---|---|---|
| G04 | 4 Curve rails | High/low crossing tracks for two identified balls | v1 perspective |
| G13 | 3 FONT labels (`1/2/3`) | Labels follow their drawers; contents keep separate IDs | v2 orthographic |
| G14 | 3 Curve tubes | Source open ends and bevel cross sections remain intact | v2 orthographic |
| G29 | 8 Curve rails | Container entry/exit and blocked rail continuity | v1 perspective |
| G31 | 4 Curve rails | Two-lane identity through the opaque tunnel | v1 perspective |
| G46 | 2 FONT labels (`A/B`) | Labels follow the swapping boxes; hidden contents retain poses | v2 orthographic |

All 120 frames pass: worst depth MSE **8.372e-15**, minimum every-object mask IoU
**1.0**. G13's labels occupy at most 6/10/10 source pixels at 320×180, so even one
missing label pixel would violate the existing IoU budget. Labels participate in
source-world triangle/normal checks as well as image checks; they are never
replaced by rectangles, annotations or a screen-space text overlay.

The current audit certifies **150/150** first nominal members: 7 legacy,
94 opaque-v1, 44 opaque-v2 and 4 [alpha-surface scenes](object-permanence-transparency.md),
and 1 [sampled optical case](object-permanence-optics.md), including the separate
[scale extension](object-permanence-scale.md).
The two source-normal deferrals are resolved through [native realizations](object-permanence-native-scenes.md).
Twenty-nine additional members (two optical and twenty-seven opaque variants) are supported; 18 remain unimported, including two explicit G05 capability deferrals.

## Conversion and lifetime

`scripts/export/object_permanence_geometry.py` provides three reusable operations:

- `evaluated_mesh(obj, graph)` borrows the evaluated object's temporary mesh and
  always calls `to_mesh_clear()`, including extraction failures. MESH objects keep
  their original API path. Curve/FONT use Blender's evaluated bevel, caps,
  extrusion and glyph outlines; source objects and settings are not rewritten.
- `source_geometry(obj, graph)` records the original converted type and, for
  FONT, the text body and font name. Changing text identity is rejected even if
  its visible triangles stay unchanged, such as adding a trailing space.
- `validate_instances(...)` distinguishes the object's own evaluated mesh proxy
  from genuine collection, face/vertex or multiple instancing.

Blender 4.4 exposes a Curve/FONT's draw mesh as `is_instance=True` in the dependency
graph. Allowing all such entries would accidentally enable real instancing.
Conversion therefore requires one visible MESH proxy whose original identity and
instance original are the same source Curve/FONT, with exactly the parent's world
matrix and no actual instancer mode. Missing, transformed or multiple proxies are
rejected. Native-MESH generated instances remain rejected too.

Converted geometry must use the source viewport tessellation for rendering:
`render_resolution_u` must be zero (use `resolution_u`) or equal to it. Render-only
modifier visibility and different subdivision levels are rejected. The six audited
entries use render resolution zero and no Curve/FONT modifiers. No render setting
is adjusted to make a comparison pass. The rigid/material gates still require
constant geometry, constant positive scale, one opaque color/material, fixed
camera, stable object membership and no hide flags or actual instancing.

Open-ended curves stay open. The visibility contract's **closed triangle** rule
means inclusive triangle boundaries; it does not add caps or require watertight
meshes. Constant scale is baked into vertices/normals once. Geometry and source
identity are checked at every frame; world transforms include parent motion.

## Storage and interface compatibility

`sample_scene(..., allow_evaluated_geometry=True)` opts into conversion. Its default
remains mesh-only, so existing callers retain their capability rejection behavior.
The CLI enables conversion only for audited `CONVERTED_GEOMETRY_TASKS` entries.

File schemas and array layouts stay unchanged: local float32 vertices/normals,
int32 triangles, CPU float32 `[T,D,7]` xyz+wxyz poses and int64 source frames.
Converted meshes use the same content-hashed semantic asset directories. Five
labels and nineteen curves retain separate named objects, IDs, materials and
static/dynamic-node assignments; parent movement cannot be discarded by a static tag.

Only the new converted recordings add `geometry_conversion` metadata with policy
`stable-envs/evaluated-rigid-mesh-v1`, plus per-object `source_geometry` descriptions.
This extraction policy is separate from the v1/v2 camera visibility contracts.
`RigidTrajectory`, `RigidSceneRenderer`, State and render/render_many interfaces
need no changes. The runtime renderer has no Curve/FONT branch.

## Independent oracle and validation

Candidate conversion uses `source_object.evaluated_get(graph).to_mesh()`.
The source oracle independently enumerates actual MESH draw entries in the source
dependency graph, maps converted proxies to their original semantic objects, and
copies them with `bpy.data.meshes.new_from_object`. Copies are removed after use.
It never reads candidate arrays or imports candidate conversion helpers.
NumPy projected-edge coverage remains independent of CUDA ray/triangle coverage.
Native-MESH-only entries retain their previous oracle path.

Source-world checks compare all vertices, triangle connectivity and corner normals
at every frame, including hidden objects. CPU regressions bind source types, text,
IDs and conversion/preparation metadata. GPU gates compare every frame's depth and
every object's mask, with unchanged MSE ≤ 1e-8 and IoU ≥ .99. Cached/full RGB,
depth and IDs must remain bit-identical at 160×90 and 320×180, across shuffled,
repeated and differently chunked requests, with one static seed and no fallback.

The optional live export suite parametrizes source checks per scene and shares
only the output dataset/cache. This avoids accumulating every scene's work inside
one fixture's 900-second test deadline. Each source process keeps its existing
180-second limit; no timeout or geometric budget is increased.

Export and independently freeze in separate processes. Use the Blender and CUDA
profiles described in the [recording guide](object-permanence-recordings.md):

```bash
python -m scripts.export.batch_object_permanence --source-root <source> \
  --output <recordings> --tasks G04,G29,G31 --report <v1-export.json> \
  --visibility-contract stable-envs/opaque-pixel-rays-v1
python -m scripts.export.batch_object_permanence --source-root <source> \
  --output <recordings> --tasks G13,G14,G46 --report <v2-export.json> \
  --visibility-contract stable-envs/opaque-pixel-rays-v2
python -m scripts.testing.freeze_object_permanence_rays --source-root <source> \
  --output <new-oracle> --task G46 \
  --visibility-contract stable-envs/opaque-pixel-rays-v2
python -m scripts.testing.check_object_permanence_rendering --recordings <recordings> \
  --oracle <new-oracle> --task G46 --report <check.json> \
  --visibility-contract stable-envs/opaque-pixel-rays-v2
```

Freezers refuse to overwrite existing references. New oracles live in each task's
existing v1/v2 directory convention; all old fixtures remain byte-identical.
Source/replay GIFs contain all 120 aligned frames and retain simplified shading.
Temporary native PNGs are deleted after decoding the completed GIF. The source
pin, versions, conversion policy and tool hashes are retained in metadata/audit.
