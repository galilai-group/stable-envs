# Step 6G — `cpu-3d` findings

Two reports in one file: **Phase 1** (design and characterisation, §A, classified `FINDING` +
Fetch `BLOCKER`) and **Phase 2 / M1** (the CPU backend for the four DM Control trackcom cells, §B,
classified `PATCH`; certified `CPU_FASTER` + `BELOW_G4` on all four). The architecture note is `examples/results/step6/cpu-3d/design.md` (§11 there
is "M1 as built"). The Fetch feasibility ticket (CPU3D-FETCH-1) is reported in §C.

---

# B. Phase 2 / M1 — `PATCH`: a CPU-native backend for `DMCCheetahRun`, `DMCHopperHop`, `DMCWalkerWalk`, `DMCHumanoidStand`

- **Baseline SHA** `89fa5ca81341dc4df7186ec325cb691431a2ef17`; the M1 parent is the
  `step6/integration` merge `5b3bcc6` (merged as `9e2e0a1`); **branch HEAD** = the commit carrying
  this file (M1 code: `28419aa`, cuda replay: `749a19c`).
- **Hardware/venvs**: cuda replay on `gpu2106` (RTX 3090, `se-step5`, one job); CPU certification
  on `node2333` (Xeon Platinum 8268, `batch`/`normal`, 1 torch thread; `se-vis311` = mujoco 3.11.0 +
  dm_control 1.0.44 for the declared reference, `se-step6` = 3.12 for the record); unit tests
  locally in `/Users/samib/se6-venv`.
- **Affected configs**: the four above (visual tier, 84 px). Nothing else renders through the
  changed code: `DMControlSuite` (10 parity cells) and every other family are untouched by
  construction; `raycast3d.py` and `mjwarp.py` are not edited.

## What was built (`stable_envs/render/cpu3d.py`, dispatch in `stable_envs/envs/dmcontrol.py`)

Architecture (d) of the design note, MJWarp's semantics transcribed from `mujoco_warp/_src/render.py`:
per-pixel rays as `compute_ray`; the skybox baked once per (orientation, resolution) through
`sample_skybox`'s cube-map law (the trackcom camera never rotates: 99.4-99.7 % of MJWarp's own
skybox pixels are constant across the corpus); the plane by `ray_plane` (front-face, bounded) with
the PLANE texgen law and one bilinear tap; the 6-19 capsules/spheres cast through the shared
`raycast3d.cast_nearest` inside their projected per-env window; shadow rays through
`raycast3d.any_blocking` only on mover hits and on floor hits inside the movers' light-space
footprint; the megakernel's shading order with its constants (headlight as a directional light
toward the camera, spot cutoff and `pow(cos, exponent)`, shadow factor **0.3**, `128*shininess`).
`DMControlTrackCom` dispatches on device: `cuda` -> the MJWarp path, unchanged byte for byte;
`cpu` -> `cpu3d`; anything else raises `ValueError` -- `device="cpu"` can no longer reach the GPU.
Per-env appearance (`material.*`, `light.diffuse_scale`) is applied as tables, as on the GPU path;
`VISIBILITY_KNOB`/`CAMERA_KNOB` stay `False` on both devices (same variation space per device).
**Coverage**: the four cells above on CPU; **`DMCSwimmerSwimmer6`** (the fifth `DMControlTrackCom`
cell, outside the frozen 49, no replay obligation) is **refused at construction** on `device="cpu"`
with `ValueError: DMCSwimmerSwimmer6 (swimmer-swimmer6) has no CPU 3-D backend: ... movers include
mjGEOM_ELLIPSOID (mjtGeom 4) ...` -- its bodies are ellipsoids, which no CPU intersector covers; it
used to render on the GPU behind a CPU request. Everything else the megakernel does is implemented
or refused at `Scene.__init__` (design.md §13: POINT/IMAGE lights, intrinsics, directional shadow
casters, `light_active`, the `nlight == 0` fallback, texel-(0,0) modulation, the plane as an
occluder, the backface cull, two per-chunk shadow preconditions).

## Bottleneck / mechanism (measured, `design.md` §4, §11)

There was no CPU path (`CPU_BACKEND_MISSING`; Step 3's "cpu" MJWarp rows were the GPU behind
`device="cpu"`, `DEVICE_NOT_HONOURED`). The binding bar on the certification host is the decode
conjunct: `cores_cpu >= 1` = 6,629 fps at 84 px = 21 ns/px, while torch costs 2 us per operator and
0.9-2.2 ns/element/op and `raycast3d`'s unfused CPU cast 210-225 ns per ray-primitive test. M1's
cost is 60-75 % that cast on the compacted windows, which RAYCAST-5 and the shared fused cast
address; `cpu3d.py`'s own passes (plane, sky, shade) are the remainder.

## A finding on the GPU path, made while matching it

**The four trackcom cells' certified frames are truncated, not rounded.** `_tc_context` wraps the
context build and the warm-up in `MW.gl_rounding()` but captures the CUDA graph outside it (the
defect `fetch.py` fixed for Fetch on 2026-08-11 and `mjwarp.gl_rounding`'s docstring warns about).
Measured on the frozen states: a rounding CPU backend sits exactly one level above MJWarp on
83-91 % of floor and body pixels; a truncating one does not. `cpu3d.QUANT = "trunc"` by default so
the CPU draws the picture as certified. Moving the capture inside the block is a one-line GPU-path
change that would move the four cuda digests (and the declared numbers slightly toward the
official frames); **not applied -- the coordinator's call, since the cuda frames must stay
byte-identical in this ticket.**

## Tests run

- `tests/unit/test_cpu3d.py` (new): honest device (CPU scene built, no MJWarp context; an unknown
  device raises), no GPU runtime import in `cpu3d.py`, determinism, batch-size / chunk-boundary /
  order invariance, within-a-level agreement against a committed slice of MJWarp frames
  (`replay/mjwarp_frames_first4.npz`), render at another resolution. **15 passed** (`MUJOCO_GL=glfw`).
- `tests/unit` whole suite locally: **674 passed, 6 skipped, 2 failed** -- both failures
  (`test_every_cited_scratch_path_resolves`, `test_examples[...robocasa_visual/render_mjwarp.py]`)
  reproduce identically on the integration commit `5b3bcc6` in a scratch worktree: pre-existing.
- cuda side: `examples/fidelity_step4.py --shard dmc --device cuda` on the branch tree (job
  6354766); the frozen-digest comparison shows the known baseline drift (`mean_pixel` 79.24->79.08
  etc., reacher-correctness), identical on both trees.

## cuda replay: integration-vs-branch identical (job **6354766**, `gpu2106`, RTX 3090, driver 575.57.08, one job, two trees)

`verify_pixels.py` (reacher-correctness's method) on the four cells, `$SCR/se6/integration` vs this
branch: **`cuda_pixels_identical=True` on all four; `CUDA PIXEL VERDICT: IDENTICAL`**
(`replay/gpu/pixels_{before_integration,after_branch}.json`). The GPU path did not move by a bit. **This does NOT reproduce the frozen Step-4 digests** -- the
pre-existing baseline drift (reacher-correctness) is present on both trees identically, so the cuda
replay is "integration-vs-branch identical", never "passes the frozen oracle".
The same-tree cuda frame stack (`replay/gpu/frames_cuda_gpu2106.npz`) is bit-identical to the
texture-fidelity stack from gpu2108/gpu2113, so MJWarp is stable across nodes in `se-step5`; the
frozen Step-4 digests (a different venv, job 6176998) remain the soft oracle they were.

## Cross-backend: CPU vs our MJWarp frames, 64 frozen states per cell (same tree, same states)

| cell | mean abs /255 | exact | within 1 | max | sky exact | floor exact | body exact | > 2 levels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DMCCheetahRun | 0.018 | 0.98209 | 0.99998007 | 65 | 0.980 | 0.9985 | 0.969 | 5 px / stack (1.107e-5) |
| DMCHopperHop | 0.018 | 0.98256 | 0.99998893 | 66 | 0.981 | 0.998 | 0.974 | 3 px (6.64e-6) |
| DMCWalkerWalk | 0.016 | 0.98511 | 0.99968555 | 65 | 0.977 | 0.998 | 0.982 | 6 px (1.329e-5) |
| DMCHumanoidStand | 0.004 (no sky pixels in this frame) | 0.99653 | 0.99996457 | 115 | -- | 0.9986 | 0.945 | 7 px (1.550e-5) |

(`replay/cross_backend_vs_fresh_cuda_local.json`, `m1_local/compare_frozen.json` with SIGNED
statistics: the +1/-1 channel fractions are balanced, 0.4-0.5 % each, mean signed difference
|0.0008| or less -- no offset remains.) The pixels beyond two levels are 3-7 per 64-frame stack
(0.047-0.109 per frame): silhouette hit/miss flips between two analytic intersectors. Within-1 is
>= 0.99968 on every cell (Walker 0.99968555). Ablations: `m1_local_quant_round/` (the ROUNDING
quantiser, relabelled from the misnamed `m1_local_frac0`: exact 0.09-0.72, +1 channel fractions
0.16-0.61 -- the one-level offset), `m1_local_frac0/` (float texture weights, `TEX_FRAC_BITS=0`,
knob now live: exact moves by +-0.001-0.002 with no consistent sign). **The
declared bound `backend_agreement_exact_frac_min = 0.99`** (`tests/gpu/test_conformance.py::
test_backend_agreement`) is declared for the Scene2D Triton-vs-torch check and is `not_applicable`
on these configs; *applied as written* to CPU-vs-MJWarp it would read 0.982-0.997 exact -- **below
0.99 on three of four cells for a renderer that is semantically equal and within one level on
99.97 % of pixels**. Per the instruction: the numbers are reported and **this stops for a
coordinator/Sami decision**; the bound is not touched and no cross-backend bound is declared here.
The proposal stands as in `design.md` §8: a cross-backend `within1_min`/`mean_abs_max` bound set
from this spread, alongside the visual-tier declaration.

## Certification of the CPU frames (job **6354767**, `node2333`, 1 thread)

Job 6354767 ran the existing machinery unchanged (`examples/fidelity_step4.py --device cpu`,
`scratch/step4/certify.py`, `scratch/step6/gfxdraw/certify_shard.py`, the frozen Step-3 evaluator).
Node load was 23-26 throughout (a shared node; the paired reference spread of 1.08-1.13 is inside
the 1.25 bound, `certifiable: True`). `--max-load` was left disabled by policy, recorded in the shard
(`provenance.max_load_policy`): batch nodes are shared, the reference is PAIRED so load enters the
ratio symmetrically, and the spread bound is what rejects a node whose load moved the reference.
These artifacts carry no tree SHA (`git_commit: null`, the rsync has no `.git`); the re-run queued in
§"Pending Oscar runs" stamps it through `STABLE_ENVS_GIT_COMMIT`/`GIT_COMMIT` (`sync.sh`).

**(i) Gate A under the declared reference (`se-vis311`: mujoco 3.11.0 + dm_control 1.0.44) --
`replay/certify311_cpu/certification.json`: 3 PASS, 1 FAIL.** Every mandatory check passes on all
four cells on CPU: determinism, batch invariance, batch-size invariance, chunked render, order
independence, render purity, style-is-state, set_state round trip, state dims matter, variations
do something, per-env variation.

| cell | `check_visual_fidelity` (mean / within8 / ssim) vs declared ceilings | verdict | `check_resolution_consistency` (corpus) | bound - margin |
|---|---|---|---:|---:|
| DMCCheetahRun | 5.12 / 0.872 / 0.662 vs <= 6.5 / >= 0.84 / >= 0.62 | **PASS** | 0.5311 pass | 0.52885 |
| DMCHopperHop | 4.52 / 0.898 / 0.687 vs 5.7 / 0.86 / 0.64 | **PASS** | 0.5535 pass | 0.54522 |
| DMCWalkerWalk | 8.03 / 0.711 / 0.825 vs 10.5 / 0.65 / 0.77 | visual PASS; **gate FAIL** on the downsample | 0.6185 **fail** | 0.62395 |
| DMCHumanoidStand | 5.95 / 0.742 / 0.881 vs 8.2 / 0.69 / 0.82 | **PASS** | 0.6095 pass | 0.59865 |

The visual numbers are the GPU path's own (texture-fidelity's 3.11 arm: 4.99 / 4.37 / 7.86 / 6.27)
to within 0.1-0.3/255 -- the reference here is llvmpipe rather than hardware EGL, which is where
that spread comes from. Walker's FAIL is the known bound-provenance defect, not the backend: the
merged `fidelity_step4._resolution` measures on the 64-state frozen corpus, where the GPU path
scores 0.6188 (texture-fidelity, both arms) and the CPU path 0.6185; on the bound's OWN protocol
(8 states from `render_worker.STATE_SEED`) the CPU path reproduces every recorded floor to
<= 0.001: **0.53456 / 0.55098 / 0.62879 / 0.60367 vs recorded 0.53385 / 0.55022 / 0.62895 / 0.60365**
(`replay/downsample_protocol_cpu_local.json`). Same finding, same fix owner as texture-fidelity §6.

**(ii) The 3.12 arm (`se-step6`, `replay/certify312_cpu/`): 3 BLOCKED, 1 FAIL**, exactly as the
provenance certifier is designed to answer a drifted reference (`reference version mismatch: the
bound was measured against mujoco 3.11.0`); the blocked observations (6.80 / 6.83 / 16.17 / 5.99)
are MJWarp's own 3.12 numbers (6.51 / 6.55 / 15.95 / 6.33) plus the llvmpipe-vs-EGL reference spread.
Walker's FAIL is the same downsample check.

**(iii) Cross-backend on the certification node** (`replay/cross_backend_node2333/`): identical to
the local table above to every printed digit (mean 0.018 / 0.018 / 0.016 / 0.004, exact 0.982 /
0.983 / 0.985 / 0.997, within-1 >= 0.999).

**Performance -- the paired Step-3-shaped sweep** (`sweep/after/cpu_1thread_paired_after_node2333.json`,
labelled by the frozen Step-3 evaluator (`scratch/step3/certify.py`, which the report generator applies
to the raw shard itself; the ad-hoc `tiers_after_*.json` was dropped); 1 torch thread; official =
dm_control's own renderer on the same node = Mesa llvmpipe, interleaved between cells):

| cell | ours fps B=1 / best (batch) | official llvmpipe fps, median [min-max] | decode fps/core | **S_cpu** | **cores_cpu** | **tier** |
|---|---|---|---:|---:|---:|---|
| DMCCheetahRun | 37.3 / **115.6** (64) | 23.4 [22.6-25.5] | 6,831 | **4.93** | 0.017 | `CPU_FASTER` + `BELOW_G4` |
| DMCHopperHop | 52.7 / **187.8** (256) | 27.4 [25.1-28.2] | 6,758 | **6.86** | 0.028 | `CPU_FASTER` + `BELOW_G4` |
| DMCWalkerWalk | 31.6 / **63.1** (16) | 25.0 [24.2-26.1] | 7,572 | **2.52** | 0.008 | `CPU_FASTER` + `BELOW_G4` |
| DMCHumanoidStand | 19.8 / **45.8** (64) | 10.9 [10.6-11.4] | 7,551 | **4.22** | 0.006 | `CPU_FASTER` + `BELOW_G4` |

Honestly: `S_cpu >= 2` on all four (`CPU_FASTER`), 2.5-6.9x the official renderer on the same
core; `cores_cpu` 0.006-0.028, i.e. 35-160x short of the decode bar, so `BELOW_G4` caps every row
and none is `CPU_FAST` -- the outcome the design predicted for M1 (§4.2: 260-800 fps estimated on
an idle core; 46-188 fps measured on a node at load 24 with `raycast3d`'s unfused cast still on the
path). The Walker and Humanoid 1024 cells were skipped by the 12 s per-cell budget; the queued re-run
keeps them (`--max-cell-s 60`, reps recorded). Hopper clears `S_cpu >= 5`; the other three would need 1.02-2x, which RAYCAST-5's bounded
CPU cast is expected to give without touching `cpu3d.py`. M2 (the fused SIMD kernel) remains
unauthorized and is the only route to `cores_cpu >= 1`.

## Tiers before/after, pixels changed, semantic follow-up

- Tiers: `CPU_UNMEASURED` (no path; `DEVICE_NOT_HONOURED`) -> **`CPU_FASTER` + `BELOW_G4` on all four** (S_cpu 2.52-6.86, cores 0.006-0.028).
- Pixels changed: **no** on cuda (byte-identical); the CPU frames are new pixels, within a level of
  the cuda ones as tabulated.
- Semantic follow-up: none required for the GPU frames; for the CPU frames the DINO tripwire on
  the visual tier (bound 0.25) is the proportionate check if the coordinator wants one -- the CPU
  frames are within 0.02/255 of frames that sit far inside it.

## Round 2 (review): provenance-stamped re-run, job **6356047**, `node2407` (AMD EPYC 9684X, load 124-136), commit `5f020fe`

`replay/certify311_cpu/`, `replay/certify312_cpu/` (both `provenance.git_commit = 5f020fe…`,
`git_commit_source = env:STABLE_ENVS_GIT_COMMIT`), `replay/gl_renderer_node2407.json`
(`llvmpipe (LLVM 19.1.7, 256 bits)`, vendor Mesa, GL 4.5 Mesa 24.2.8), `replay/cross_backend_node2407/`:
**every verdict and every printed number is identical to round 1** (3.11 arm 3 PASS + Walker's
downsample FAIL at 0.6185; 3.12 arm 3 BLOCKED + the same FAIL; cross-backend mean 0.018 / 0.018 /
0.016 / 0.004, exact 0.982 / 0.983 / 0.985 / 0.997) — on a different CPU family, which is the
determinism claim made twice over. The sweep step (`sweep/after/cpu_1thread_paired_after_node2407.json`,
label `cpu-3d-paired-after`, `tree_commit` stamped, reps 10, OMP/MKL 1 recorded, `max_load_policy`
recorded) measured **all mandatory cells including 1024** — Cheetah 209.9, Hopper 333.6, Walker
114.9, Humanoid 81.7 fps at B=1024, flat against B=256 (209.2 / 332.4 / 118.3 / 81.4) — but the
shard is **`certifiable: False`**: the interleaved official renderer's readings spread **1.67×**
(bound 1.25) on that node, so the pattern refuses to derive a tier from it; the numerator stands.
The `CPU_FASTER` labels above therefore rest on the node2333 shard (spread 1.08-1.13, certifiable,
no 1024 cells for Walker/Humanoid, no SHA); a sweep-only job with the load guard re-enabled
(`sweep_cpu.sbatch`, `--max-load 12`) was run three times (jobs 6356330, 6356449, 6356450) and
aborted at start every time (loads 39 / 126 / 29: every batch node in rotation is shared); the
rejected shards are filed under `sweep/rejected/` with a README. A shard with all three -- cells,
SHA, spread inside the bound -- needs a quiet node or an exclusive allocation (which pends under the
per-user QOS CPU limit); the job is ready to resubmit.

## Oscar runs of this review round

1. `certify_cpu.sbatch` — job 6356047, done (above).
2. `fetch_feasibility.sbatch` — job 6356048, done (§C).
3. `sweep_cpu.sbatch` — jobs 6356330 / 6356449 / 6356450, all aborted on load at start; pending a quiet node.

## SLURM

6345025 (Phase 1 characterisation), 6354766 (cuda replay), 6354767 (CPU certification, round 1),
6354890 (CPU3D-FETCH-1, round 1), 6356047 (certification, round 2, stamped), 6356048 (CPU3D-FETCH-1,
round 2, culled + phases), and the queued load-guarded sweep.

## Open questions for the coordinator

1. The cross-backend bound (above): `backend_agreement_exact_frac_min` as declared would fail this
   renderer on three cells at 0.982-0.985 exact; decide whether the CPU backend is certified under
   the visual-tier declaration alone, under a new within-1 bound, or both.
2. The GPU capture outside `gl_rounding()`: fix (moves the cuda digests) or leave (the CPU backend
   then keeps `QUANT="trunc"`).
3. `raycast3d` interface: none needed for M1. RAYCAST-5's per-primitive bounding will speed the
   CPU path directly through the unchanged `cast_nearest`/`any_blocking` calls.

---

# A. Phase 1 -- `FINDING` (design ready) + Fetch `BLOCKER` (measured), as reported at commit `6505e71`

- Affected: the 4 DMC trackcom cells (visual, 84 px, MJWarp-only) and the 12 Fetch cells (parity,
  480 px). Step 3's "cpu" rows: `CPU_BACKEND_MISSING` / `DEVICE_NOT_HONOURED`.
- Measured (job 6345025, `node2339`): same-host official llvmpipe 24.4/27.1/25.5/10.7 fps (DMC) and
  39.6-53.3 fps (Fetch); decode 6,629 fps at 84 px, 776-881 at 480 px; torch 2 us dispatch floor,
  6.3 us/full-frame op at 84 px; `raycast3d` unfused cast 210-225 ns/ray-primitive; compiled scalar
  floor 95-150 ns/px (DMC, tiled) and 540-1,640 ns/triangle (480 px).
- DMC: 6-19 capsules/spheres + one textured plane, camera and spotlight `trackcom` (translate only),
  skybox constant in camera space (99.4-99.7 %), movers 2.5-7.2 % of pixels. Fetch: 15 meshes /
  31,720 triangles at 8-10.5 % of pixels, fixed camera, no shadow, exact plate.
- Recommendation (approved): architecture (d) in torch -> `CPU_FASTER` (M1, done above); `CPU_FAST`
  only by a fused SIMD kernel, gated on <= 21 ns/px; Fetch `CPU_FAST` a measured blocker (39 ns/tri
  required incl. fill vs 540-1,640 scalar), `CPU_PARITY` reachable.
- Full detail: `design.md` §0-§10.

---

# C. CPU3D-FETCH-1 -- feasibility of a shared CPU backend for Fetch at the frozen `CPU_FAST` bar (measurement only)

**Classification: `BLOCKER` for `CPU_FAST` (measured on the real mesh set, backface-culled, phase-resolved);
a shared CPU backend at `CPU_PARITY`/`CPU_FASTER` is feasible.** No implementation, no decimation.

Two measurements, both on the ACTUAL visible mover meshes of `FetchReach` and `FetchPickAndPlace`
(15 meshes, **31,720 triangles**, all in view, **median triangle edge 0.50 px** at 480 px), put
through MuJoCo FK at 16 frozen states and the `se:cam` frustum by `fetch_mesh_dump.py`
(`se-s3b-fetch311`), then rasterised by `span_raster.c` (scalar C, `-O3 -march=native`, bbox-bounded
edge-function scan, z-test, Gouraud normals, Blinn-Phong):

| job / node / core | mode | ms / frame | fps (1 core) | ns / triangle | pixel writes | bbox tests |
|---|---|---:|---:|---:|---:|---:|
| 6354890 / node2334 / Xeon 8268 (round 1) | un-culled (both windings) | 8.8-10.4 | 97-114 | 277-327 | 49,151 | 942,741 |
| 6356048 / node2407 / **AMD EPYC 9684X**, load 134-140 (round 2) | un-culled | 6.79 | 147 | 214 | 49,151 | 942,741 |
| " | **culled** (front faces only) | **4.28** | **234** | **135** | 36,609 | 501,433 |
| " | culled, setup only (transform, bbox, area, cull) | 0.77 | 1,305 | **24** | -- | -- |
| " | culled, setup + bbox scan + z-test, no shading | 3.20 | 313 | 101 | 36,609 | 501,433 |

(`fetch_feasibility/span_raster_*_node2334.jsonl`, `*_node2407.jsonl`, `provenance_node2407.json`;
PickAndPlace within 1 % of Reach in every row.) Round 2 ran on a different, faster core than the
certification Xeon and at load ~135, so its absolute numbers are not the Xeon's; the RATIOS between
its rows are the finding: **culling is worth 1.59x** (the un-culled overdraw was 2x the visible
pixels), and the culled cost splits **24 ns setup + 77 ns bbox scan/z-test + 34 ns shade** per
triangle.

Against the frozen bars on the Xeon (design.md §3.2): `CPU_FAST` needs <= 1.13-1.29 ms per frame =
**<= 36-41 ns per triangle including fill**. Culled scalar on the EPYC is **135 ns, 3.3-3.8x short**
(the un-culled round-1 figure of 7-9x is superseded); a rasteriser that wasted no bbox tests at all
would still pay the 24 ns/triangle setup plus ~36 k pixel writes -- ~1.0-1.3 ms, i.e. AT the bar on a
faster core than the certification host, with nothing left for the composite and the FK. That is the
architecture-independent margin: **~3.5-4x short in scalar, and no better than break-even for a
perfect SIMD binning rasteriser on the EPYC, worse on the Xeon.** The `S_cpu >= 5` conjunct also
fails: 234 fps (EPYC, culled) against a same-node official that was not measured in this job; against
the Xeon's llvmpipe 39.6-53.3 fps it would be 4.4-5.9x, and round 1's same-node pairing gave
1.8-2.8x -- either way `cores_cpu` (0.27-0.30 even at 234 fps against the Xeon's 776-881 fps decode
bar) fails `CPU_FAST` on its own. What IS feasible: a shared span-rasteriser backend (the plate is
exact and free; movers 4-10 ms/frame) at 100-230 fps = `CPU_PARITY`/`CPU_FASTER`, or option (g)
llvmpipe at ~46 fps = `CPU_PARITY`. Verdict unchanged: **do not fund a Fetch CPU backend for
`CPU_FAST`; fund one only if `device="cpu"` honoured at `CPU_PARITY`/`CPU_FASTER` is wanted for its
own sake.** The Phase-1 synthetic probe (`cost_floor tri`) was re-timed with triangle generation
outside the timed region: 180-183 ns/triangle at 3 px edges on the EPYC
(`fetch_feasibility/cost_floor_tri_node2407.jsonl`); design.md §4.1's Xeon figures included the
generation and are superseded by the real-mesh rows above.
