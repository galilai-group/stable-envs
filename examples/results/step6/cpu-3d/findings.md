# Step 6G — `cpu-3d` findings (Phase 1: design and characterisation, no implementation)

**Classification: `FINDING`** — design ready for coordinator approval for the four DM Control
trackcom cells (reachable tier stated per architecture), carrying a **measured `BLOCKER` for
`CPU_FAST` on the twelve Fetch cells**. The architecture note is
`examples/results/step6/cpu-3d/design.md`; this file is the report in AGENT_COMMON's format.

- Baseline SHA: `89fa5ca81341dc4df7186ec325cb691431a2ef17` (`sami/pre-step6-baseline`).
- Branch: `step6/cpu-3d`, HEAD = the commit carrying this file. `stable_envs/` untouched;
  `examples/results/step4/` and `step5/` untouched. Only `scratch/step6/cpu-3d/` and
  `examples/results/step6/cpu-3d/` were added.
- Hardware/venv: certification numbers from **SLURM job 6345025** (`se6-cpu-3d-char`, `node2339`,
  Intel Xeon Platinum 8268 @ 2.9 GHz, `--partition=batch --qos=normal`, 4 CPUs, 1 torch thread,
  `OMP/MKL_NUM_THREADS=1`), venvs `se-step6` (torch 2.14 cpu path, dm_control 1.0.45, mujoco 3.12),
  `se-step5` (mujoco_warp 3.12.0 source dump only), `se-s3b-fetch311` (mujoco 3.11, gymnasium-robotics).
  Model census on the Mac (mujoco 3.12; geometry/materials/textures byte-identical to 3.11 per
  texture-fidelity §2). No GPU job: reference frames came from the texture-fidelity workstream's
  frozen-state frame stacks.

## Affected configs (16)

- `DMCCheetahRun`, `DMCHopperHop`, `DMCWalkerWalk`, `DMCHumanoidStand` — `DMControlTrackCom`, 84 px,
  visual tier, MJWarp-only on the render path (the `raycast3d` entry in `backends` is a module-level
  artefact; confirmed by the raycast3d workstream's census).
- `Fetch{Reach,Push,Slide,PickAndPlace}` × `{'',Dense,Dict}` — `FetchBase`, 480 px, parity tier
  (interior ≥ 0.99), baked MuJoCo plate + MJWarp mover pass.
- NOT in scope (already a genuine CPU path, owned by `step6/raycast3d`): the 10 `DMControlSuite`
  parity cells.

## Profile-before artifacts

There is no CPU path to profile (`CPU_BACKEND_MISSING`; Step 3's "cpu" MJWarp shard ran on the
GPU behind `device="cpu"`, labelled `DEVICE_NOT_HONOURED`). What was measured instead:
`examples/results/step6/cpu-3d/census/` (DMC model census ×4, `frames_census.json`,
`shadow_candidates.json`) and `examples/results/step6/cpu-3d/characterize/`
(`cpu_micro_node2339.json`, `cost_floor_node2339.jsonl`, `census_fetch_node2339.json`,
`mujoco_warp_sampling_source.txt`, official frames). Scripts: `scratch/step6/cpu-3d/`.

## Bottleneck mechanism (measured)

1. **The binding bar is `cores_cpu ≥ 1`, not `S_cpu ≥ 5`.** On the certification host the official
   renderer is Mesa llvmpipe: 24.4 / 27.1 / 25.5 / 10.7 fps at 84 px (DMC), 39.6–53.3 fps at 480 px
   (Fetch); so `S_cpu ≥ 5` is 54–136 fps (DMC) / 198–267 fps (Fetch). One core of JPEG-q90 decode
   is **6,629 fps at 84 px** and **776–881 fps at 480 px**, i.e. `cores ≥ 1` needs **21.4 ns/px**
   (DMC) and **36–41 ns per triangle including fill** for Fetch's 31,720 mover triangles.
2. **Torch on one core is dispatch/bandwidth-bound**: 2.0 µs dispatch floor, 6.3 µs per full-frame
   op at 84 px (0.9 ns/elem in L2), 2.2 ns/elem out of L2 (batching *hurts*). `raycast3d`'s unfused
   CPU cast: 210–225 ns per ray-primitive test (reproduces the raycast3d workstream's 145–240).
   A full-frame 2-light `shade` alone is 0.99 ms/frame at 84 px.
3. **Compiled scalar floor** (`cost_floor.c`, `-O3 -march=native`, no SIMD intrinsics): whole DMC
   frame (plane + skybox + checker + headlight + spot + shadow + K capsules, tiled) **95 ns/px at
   K=8 (1,484 fps), 150 ns/px at K=19 (943 fps)** — 4.5–7× short of the decode bar; triangle raster
   540–1,640 ns/tri at 480 px — 14–40× short of Fetch's bar.
4. **Scene structure (DMC)**: 6–19 capsules/spheres over one textured plane; camera and spotlight
   are `trackcom` on the same body and never rotate; in our MJWarp frames **99.4–99.7 % of skybox
   pixels are bit-identical across the 64 frozen states** (bakeable plate, 55–85 % of three
   frames); floor 0 % constant (checker phase); movers cover 2.5–7.2 % of pixels, bbox 7–28 %;
   FK 3–10 µs/state. **Fetch**: fixed camera, one directional light without shadow, flat
   materials, plate exact (10.8–13.5 ms warm bake on the same llvmpipe), robot = 15 meshes /
   31,720 triangles covering 8–10.5 % of pixels with an 50–89 % screen bbox.

## Proposed intervention (design; nothing implemented)

- **DMC four — architecture (d)**: per-(camera orientation, res) baked skybox plate (exact vs
  MJWarp by construction) + analytic floor with MJWarp's PLANE UV law and one bilinear tap +
  compacted `raycast3d.cast_nearest_torch` on the per-env mover window + `any_blocking_torch`
  shadow rays on the floor candidates + an MJWarp-law shader (headlight at the camera, spot
  cutoff/exponent, shadow factor 0.3, `128·shininess`, round-to-nearest). New module
  `stable_envs/render/cpu3d.py`; `DMControlTrackCom` dispatches on device and never builds an MJWarp
  context on CPU. Estimated **260–800 fps at 84 px → `CPU_FASTER` (S_cpu 24–30 same-host, cores
  0.04–0.12)**. `CPU_FAST` only via a fused SIMD whole-frame kernel (M2), gated on a stand-alone
  micro-benchmark reaching ≤ 21 ns/px at K=19 on the certification host; plausible, unproven.
- **Shared fused CPU cast**: one kernel behind `raycast3d.cast_nearest`/`any_blocking` (Triton
  `_tri_cast`'s shape, SIMD across rays, `-ffp-contract=off`/IEEE div+sqrt discipline, asserted
  bit-identical to `cast_nearest_torch`), **owned by `step6/raycast3d`**; this workstream consumes it
  and layers the whole-frame kernel on it. Proposed split, coordinator's call.
- **Fetch — `CPU_FAST` is a measured blocker** (39 ns/tri needed vs 540–1,640 measured; even a
  SIMD binning rasteriser is fill-bound at 1–3 px triangles). A genuine `device="cpu"` path at
  `CPU_PARITY` exists (option (g): llvmpipe mover pass over the plate) if wanted; decimating the
  meshes is not admissible under the parity contract.

## Tests run

None applicable (no code under `stable_envs/` changed). Characterisation scripts ran to completion
in job 6345025 (all four steps; `COMPLETED`, output files verified).

## Step-4 replay

Not applicable in Phase 1 (no pixels produced by any changed path). Phase-2 obligation stated in
`design.md` §6: the four trackcom configs on cuda must stay byte-identical (CUDA path untouched);
a new CPU shard under `examples/results/step6/cpu-3d/replay/`.

## Tiers before/after, pixels changed, semantic follow-up

Before: all 16 `CPU_UNMEASURED`. After: unchanged (design only). Pixels changed: **no**. Semantic
follow-up: none now; for a Phase-2 CPU path within a level of the MJWarp frames, the DINO tripwire on
the CPU frames (visual-tier bound 0.25), no Gate-H rerun.

## Open questions for the coordinator

1. **Contract for a CPU backend** (`design.md` §8): the only cross-backend bound
   (`backend_agreement_exact_frac_min` 0.99, Scene2D-only) is the wrong one; the frozen MJWarp
   digests are a **soft oracle** (MJWarp 0.02 % run-to-run noise; reacher-correctness: the baseline
   misses the four trackcom digests by 0.16–0.61/255 `mean_pixel` on the same card). Proposed: a
   new cross-backend within-1/mean bound against our frozen frames (B) plus a per-device
   visual-tier declaration under the 3.11 reference (A); Fetch stays on parity (C). Both A and B
   change a schema/declaration → approval needed.
2. **Is `CPU_FASTER` for the DMC four worth M1 (3–5 days + certification)** given `CPU_FAST` needs
   the gated M2 kernel and a compiled-extension packaging decision?
3. **Fetch**: accept the `CPU_FAST` blocker as the record, or fund option (g) for `device="cpu"`
   honouring at `CPU_PARITY`?
4. **Ownership of the fused CPU cast** (raycast3d, as proposed) and scheduling of M2 after it.
5. Same-host `S_cpu` denominators are llvmpipe (11–53 fps), not the GPU-node hardware EGL numbers
   Step 3 carries (355–970): the certification record should say which.

SLURM: **6345025** only. Sibling findings read (not depended on): raycast3d `c7a9330`/`473a37b`,
reacher-correctness `af81aa2`, texture-fidelity `f484ab9`, fetch-gpu `c568fde`.
