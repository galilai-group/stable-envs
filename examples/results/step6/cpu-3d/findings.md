# Step 6G — `cpu-3d` findings

Two reports in one file: **Phase 1** (design and characterisation, §A, classified `FINDING` +
Fetch `BLOCKER`) and **Phase 2 / M1** (the CPU backend for the four DM Control trackcom cells, §B,
classified `PATCH`). The architecture note is `examples/results/step6/cpu-3d/design.md` (§11 there
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

## Step-4 replay -- cuda (job **6354766**, `gpu2106`, RTX 3090, driver 575.57.08, one job, two trees)

`verify_pixels.py` (reacher-correctness's method) on the four cells, `$SCR/se6/integration` vs this
branch: **`cuda_pixels_identical=True` on all four; `CUDA PIXEL VERDICT: IDENTICAL`**
(`replay/gpu/pixels_{before_integration,after_branch}.json`). The GPU path did not move by a bit.
The same-tree cuda frame stack (`replay/gpu/frames_cuda_gpu2106.npz`) is bit-identical to the
texture-fidelity stack from gpu2108/gpu2113, so MJWarp is stable across nodes in `se-step5`; the
frozen Step-4 digests (a different venv, job 6176998) remain the soft oracle they were.

## Cross-backend: CPU vs our MJWarp frames, 64 frozen states per cell (same tree, same states)

| cell | mean abs /255 | exact | within 1 | max | sky exact | floor exact | body exact | > 2 levels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DMCCheetahRun | 0.018 | 0.982 | 0.99998 | 65 | 0.980 | 0.9985 | 0.969 | 1.7e-5 |
| DMCHopperHop | 0.018 | 0.983 | 0.99999 | 66 | 0.981 | 0.998 | 0.974 | 1.2e-5 |
| DMCWalkerWalk | 0.016 | 0.985 | 0.9997 | 65 | 0.977 | 0.998 | 0.982 | 1.5e-5 |
| DMCHumanoidStand | 0.004 | 0.997 | 0.99996 | 115 | -- | 0.9986 | 0.945 | 1.6e-5 |

(`replay/cross_backend_vs_fresh_cuda_local.json`, `m1_local/compare_frozen.json`.) The pixels beyond
two levels are ~1 per frame: silhouette hit/miss flips between two analytic intersectors. **The
declared bound `backend_agreement_exact_frac_min = 0.99`** (`tests/gpu/test_conformance.py::
test_backend_agreement`) is declared for the Scene2D Triton-vs-torch check and is `not_applicable`
on these configs; *applied as written* to CPU-vs-MJWarp it would read 0.982-0.997 exact -- **below
0.99 on three of four cells for a renderer that is semantically equal and within one level on
99.97 % of pixels**. Per the instruction: the numbers are reported and **this stops for a
coordinator/Sami decision**; the bound is not touched and no cross-backend bound is declared here.
The proposal stands as in `design.md` §8: a cross-backend `within1_min`/`mean_abs_max` bound set
from this spread, alongside the visual-tier declaration.

## Certification of the CPU frames (job **6354767**, `node2333`, 1 thread)

PENDING -- filled from `replay/certify311_cpu/`, `replay/certify312_cpu/`, `sweep/` when the job
completes.

## Tiers before/after, pixels changed, semantic follow-up

- Tiers: `CPU_UNMEASURED` -> see the certification block (expected `CPU_FASTER`, `BELOW_G4`).
- Pixels changed: **no** on cuda (byte-identical); the CPU frames are new pixels, within a level of
  the cuda ones as tabulated.
- Semantic follow-up: none required for the GPU frames; for the CPU frames the DINO tripwire on
  the visual tier (bound 0.25) is the proportionate check if the coordinator wants one -- the CPU
  frames are within 0.02/255 of frames that sit far inside it.

## SLURM

6345025 (Phase 1 characterisation), 6354766 (cuda replay), 6354767 (CPU certification), 6354890
(CPU3D-FETCH-1 measurement).

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

PENDING -- job 6354890: a span rasteriser over the ACTUAL Fetch mover triangles (31,720 per frame,
480 px, per frozen state) on a certification core; verdict against 776-881 fps (decode) and
39.6-53.3 fps (llvmpipe official).
