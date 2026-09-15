# Step 7 W4 — `dmc-quadruped` (`swm/QuadrupedDMControl-v0`), Phase 0

**Classification: `FINDING`** (diagnosis and shared-capability design; nothing under `stable_envs/`
modified). The one shared capability the cell needs — an ELLIPSOID intersector in `raycast3d` (torch
twin + Triton twin) admitted by `cpu3d` — is design-validated below and is the `PATCH_CANDIDATE` for
the next phase. No heightfield capability is needed for the SWM id. No blocker on either backend.

- Parent baseline `76dc5116f432e61dd47fba8f9e879cf9212c8e46`; branch `step7/dmc-quadruped`, worktree
  `/Users/samib/stable-envs-step7-dmc-quadruped`; HEAD = the commit carrying this file.
- Hardware/venvs: Apple M5, `/Users/samib/se6-venv` (mujoco 3.12.0, dm_control 1.0.45, torch 2.14 CPU,
  `MUJOCO_GL=glfw`), plus a scratch venv with the DECLARED reference pair (mujoco 3.11.0 + dm_control
  1.0.44, built with `uv` under the session scratchpad, numpy/pillow only) for the reference-version
  check. **Oscar was DENIED at auth for the whole phase**: every GPU/MJWarp item is listed as pending
  in §8. MJWarp is not installed locally; its capability answers come from reading the source (§4).
- Affected configs: one new cell (proposed `DMCQuadrupedWalk`, `dmc-quadruped-walk`,
  `reference_id dm_control/quadruped-walk`, camera 2). Nothing existing renders through anything this
  phase touched, because nothing was touched. The eventual patch touches shared files
  (`raycast3d.py`, `cpu3d.py`) and therefore the four certified trackcom cells and the 21 raycast
  cells: their pixels cannot move (a new `kind == 4` branch is dead code for every existing scene)
  and the replay obligation is stated in §7.
- Scripts: `scratch/step7/dmc-quadruped/gen_states_ref.py`, `scratch/step7/dmc-quadruped/swm_model_roundtrip.py`,
  `scratch/step7/dmc-quadruped/compare_refs.py`, `scratch/step7/dmc-quadruped/proto_cpu3d.py`; artifacts: `examples/results/step7/dmc-quadruped/phase0/` (§9).

## 1. What the SWM id is, exactly

Read off `/Users/samib/stable-worldmodel/stable_worldmodel/envs/dmcontrol/{dmcontrol.py,quadruped.py}`
(the installed SWM source; `stable_worldmodel` itself is not importable in the local venv):

| item | value |
|---|---|
| task | `walk` (default; `run` selectable): `quadruped.make_model(floor_size=_DEFAULT_TIME_LIMIT * 0.5 = 10)` + `quadruped.Move(desired_speed=0.5)` — i.e. the suite's `quadruped-walk` model and task, built by hand rather than through `suite.load` |
| camera | **`camera_id = 2`** (`self.camera_id = 2 if domain == 'quadruped' else 0`): the `y` camera, `<camera name="y" pos="0 4 2" xyaxes="-1 0 0 0 -.5 1" mode="trackcom"/>` on the `torso` body, fovy 45. NOT camera 0 (`global`, a world-body trackcom camera 17 m from the COM, on which the robot is ~20 px of an 84 px frame). Camera 2's orientation is a constant of the model (`cam_xmat = [[-1,0,0],[0,-.4472,.8944],[0,.8944,.4472]]`) and its position is `subtree_com[torso] + poscom0`, which is exactly the trackcom contract `cpu3d.Scene` asserts and `_tc_context` handles through `com_pos` |
| size | `render(width=224, height=224, camera_id=None)`: 224 px by default; this family declares at 84 px (`_DMCModel.default_size`), so both are measured below |
| model round trip | the wrapper routes the XML through `dm_control.mjcf` (`from_xml_string` -> `modify_mjcf_model` -> `export_with_assets` -> `Physics.from_xml_path`). Replicated locally (`swm_model_roundtrip.py`, `phase0/swm_model_roundtrip.json`): geometry, bodies, sites, cameras, lights, `qpos0`, texture bytes are identical to `suite.load("quadruped","walk")`; the material and texture TABLES are permuted (`ball` first), ids differ, bytes do not |
| defaults that differ from the suite | `modify_mjcf_model` runs at every `reset()` with the variation space's `init_value`s: material `self` -> rgba **(0.6, 0.3, 0.3, 1)** where the suite has (0.7, 0.5, 0.3, 1) — the same qpos renders mean 1.34/255 apart, 90.9 % of pixels exact, max 86 (camera 2, 84 px); and `foot_back_left` density **1000** where the suite's `body` class gives 500 (a mass change: `body_mass` differs; physics, not pixels). `grid` rgb1/rgb2, floor friction, torso density and light diffuse (0.7) are set to the suite's own values |
| per-episode model edits | none: `Move.initialize_episode` only writes `qpos` (`_find_non_contacting_height`), confirmed by the coordinator's census (no edits across 4 resets) and by `_episode_slots`' logic (no drawn model arrays) |

So the cell is `("quadruped", "walk", 2)` in `TRACKCOM_DOMAINS` terms, and the SWM default appearance
is the suite's default with one material recoloured — a `material.self` knob value, which the family's
appearance space already carries (`_used_materials`). §10 asks which default the declaration is
measured at.

## 2. State sufficiency (Step-1 method, official renderer, local)

`state_sufficiency` in `scratch/dmcontrol/probe_load.py` on `quadruped-walk`, 84 px, 6 visited states, **all
four cameras** (`phase0/state_sufficiency_walk_84px.json`):

| camera | render twice | qvel x100 | different history | live qpos dims |
|---|---|---|---|---|
| 0 global, 1 x, **2 y**, 3 egocentric | 0 | 0 | 0 | **23 / 23** |

The frame is exactly `f(qpos)`, `nq = 23`, no dead dimension, no widening (no episode slots): the
state is `qpos` alone, as for cheetah/hopper/walker/humanoid.

## 3. What the default task renders (geom table, per-surface shares)

`quadruped-walk` compiles to `ngeom = 20`, `nhfield = 1`, `nmesh = 0`, `nsite = 29`, `ntex = 3`,
`nmat = 14`, `nlight = 1`, `ncam = 4` (`phase0/geom_table_walk.json`):

| id | name | type | body | size | material (rgba) | textured |
|---|---|---|---|---|---|---|
| 0 | floor | **PLANE** (finite 10 x 10, half-thickness 0.5) | world | 10, 10, 0.5 | `grid` (1,1,1,1), texture `grid` 300x300 checker `mark="edge"`, texrepeat 1 1, texuniform, **reflectance 0.2** | yes |
| 1, 2 | eye_r, eye_l | CYLINDER | torso | r .05, hh .107 | `self` (.7,.5,.3) | no |
| 3 | torso | **ELLIPSOID** | torso | **.3, .27, .2** | `self` | no |
| 4-6, 8-10, 12-14, 16-18 | thigh/shin/foot x4 | CAPSULE (12) | legs | r .08/.065/.055 | `self` | no |
| 7, 11, 15, 19 | toe x4 | SPHERE (4) | toes | r .08 | `effector` (.7,.4,.2) | no |

All twenty are group 0 with effective alpha 1. Sites: `pupil_r`/`pupil_l` (group 0, sphere r .033,
material `eye` (0,.2,1)) and `torso` (group 0, r .05, red — inside the torso ellipsoid, occluded);
`torso_touch` has alpha 0; the other 25 sites are groups 3/4 (hidden at the default `MjvOption`).
One SPOT light on the torso, `mode="trackcom"`, `pos 0 0 4`, `castshadow 1`, cutoff 45, exponent 10,
default attenuation, diffuse 0.7; headlight active (ambient .4, diffuse .8, specular .1); skybox
800x4800 gradient with `mark="random"`. Textured non-plane geom: **none** (the `ball` cube-texture
material is worn by no geom in `walk`; it belongs to `fetch`).

**The heightfield is declared and not drawn.** The `terrain` asset (201x201, size 30 30 5 .1) is
compiled into every task's model because it is in the shared XML, but `make_model` deletes the
`terrain` GEOM unless `terrain=True`, which only `escape` passes; `walk`/`run`/`fetch` compile with
**zero `mjGEOM_HFIELD` geoms** and `hfield_data` all zero (`escape.initialize_episode` regenerates it
per episode). The floor of the SWM task is the PLANE above. No heightfield capability is needed for
`swm/QuadrupedDMControl-v0`; it would be needed only for a future `quadruped-escape` cell.

Per-surface shares off MuJoCo's segmentation buffer, camera 2, 84 px, 64 states (§5 corpus):

| surface | share of frame | note |
|---|---:|---|
| sky | 15.1 % | the finite plane ends inside the frustum |
| floor (plane) | 80.3 % | textured, reflective, spotlit, shadowed |
| body (19 geoms) | 4.6 % | |
| of which the ellipsoid | **1.17 %** (~83 px at 84 px; ~580 at 224) | **visible in every state** (7.8-16.6 % of body pixels at the 16-state probe) |
| of which the eye cylinders | 0.09 % | |
| sites (pupils) | 0.013 % (0-2 px per frame) | drawn by the official renderer, by neither MJWarp nor cpu3d |

(At camera 0, for the record: sky 24.8 %, floor 74.9 %, body **0.3 %**, ellipsoid 0.08 %.)

## 4. Capability gap per backend

| backend | draws today | gap for this cell | evidence |
|---|---|---|---|
| **MJWarp** (`render/mjwarp.py` -> `mujoco_warp`) | PLANE, HFIELD (as a triangle mesh through the mesh BVH, `hfield_bvh_id`), SPHERE, ELLIPSOID (`ray_ellipsoid`), CAPSULE, CYLINDER, BOX, MESH, FLEX; **no sites**; UVs for PLANE and MESH only | **none** for `walk` (ellipsoid is dispatched; hfield is not in the model; no textured primitive). `needs_com_pos` is true (camera and light mode 2) so `_tc_context`'s assertion passes; `prepare_model` hides `torso_touch` (alpha 0) and finds no zero-alpha geom | `mujoco_warp/_src/render.py` on `main` (read 2026-09-14): the megakernel's `wp.static(int(GeomType.X) in geom_ray_types)` chain lists PLANE, HFIELD, SPHERE, ELLIPSOID, CAPSULE, CYLINDER, BOX, MESH, FLEX; `mujoco_warp/_src/ray.py::ray_ellipsoid` maps the ray into the geom frame, forms `s = 1/size^2`, solves `a x^2 + 2 b x + c = 0` with `a = (s*lvec).lvec`, `b = (s*lvec).lpnt`, `c = (s*lpnt).lpnt - 1`, normal `normalize(s * l)` rotated by `mat`; `ray_hfield` exists too. No `site_*` reference in `render.py`. **Cluster verification (pending, Oscar denied):** the RELEASE the certified cells run on, `$SCR/venvs/se-step5/lib/python3.12/site-packages/mujoco_warp/_src/render.py` (grep `ELLIPSOID`, `HFIELD`, `site`) and `.../ray.py::ray_ellipsoid`; `se-vis311` for 3.11. Independent evidence that the release draws ellipsoids: `DMCSwimmerSwimmer6` (ellipsoid bodies) ships on MJWarp 3.11 in the visual tier at declared numbers |
| **cpu3d** (`render/cpu3d.py`) | movers of kind SPHERE/CAPSULE/CYLINDER/BOX over one static finite textured plane, baked skybox, SPOT/DIRECTIONAL lights, headlight, shadows by any-hit | **`Scene.__init__` refuses `mjGEOM_ELLIPSOID` via `MOVER_KINDS`** (the swimmer refusal, design.md S13). Everything else this scene needs is implemented and its preconditions hold: camera mode 2 with constant orientation; exactly one static plane; SPOT caster with default attenuation, strictly above the plane (4 m above the torso COM); camera outside every mover's bounding sphere (4.5 m from the torso); tracking spot light already supported. Confirmed by running the prototype over 64 states at 84 and 224 px with the per-chunk checks live: no raise | prototype §5 |
| **raycast3d** (`render/raycast3d.py`) | `_sphere/_capsule/_cylinder/_box/_plane`, torch and Triton twins | **no `_ellipsoid`**: `intersect` raises, `_kind_tensor` raises for the fused kernel, `Primitives.bound_radius` returns INF, `_bound_radius` returns None | source |

**hfield: not needed, on any backend, for the SWM id** (§3). Recorded for the future `escape` case
only: MJWarp already has it; cpu3d/raycast3d would need a heightfield surface (a bilinear or
triangle-pair intersection over a 201x201 grid, i.e. a BVH-class primitive, not a closed form) — out of
scope here and not designed.

### 4.1 The ELLIPSOID intersector — design (not implemented)

**Geometry.** With `ol = R^T (o - pos)`, `dl = R^T d` (the `_mtv` frame change every other primitive
uses) and semi-axes `s = (sx, sy, sz)`, scale the ray into the frame where the ellipsoid is the unit
sphere: `ol' = ol / s`, `dl' = dl / s` (componentwise). Then

    a = dl'.dl'      b = ol'.dl'      c = ol'.ol' - 1
    a t^2 + 2 b t + c = 0   ->   t = (-b -+ sqrt(b^2 - a c)) / a

nearer positive root first, `disc > 0` and `t > 0` as `_sphere` (the repo's conventions; MuJoCo and
mujoco_warp use `>= 0` and the `1/s^2` weighting — same surface, different rounding, and no
bit-exactness against MJWarp is claimed, none being achievable, cpu3d docstring). The normal is the
gradient of `sum (x_i/s_i)^2` at the hit, `n_l = p_l / s^2`, normalised, rotated back with `_mv`. When
`sx = sy = sz = r` it is algebraically `_sphere` (the same quadratic with `a = 1/r^2` factored out).
The prototype's `_ellipsoid` in `scratch/step7/dmc-quadruped/proto_cpu3d.py` is this, verbatim.

**Bit-exactness discipline (what the repo requires of a new kind).** Elementwise and unrolled
(nine multiplies, six adds per frame change; the quadratic in the written association order) so the
frame is batch-size, order and chunk invariant by construction; the Triton twin is a `kind == 4`
branch of `_tri_isect` that is line-for-line the same expressions with `tl.math.div_rn` for every
division (`1/s`, `/a`, the normal), `tl.sqrt_rn` for the root, under the launch's existing
`enable_fp_fusion=False`; `tests/gpu/test_fast_paths.py` diffs fused-vs-torch on the cell itself;
`tests/unit/test_raycast3d_primitive_bounds.py`'s differential fuzz is extended to kind 4 (bounded vs
unbounded must be bit-identical, including the non-unit-ray guard). Note the same `disc = b*b - a*c`
cancellation `_capsule` documents applies; at this cell's 4-5 m camera distance it is far inside the
capsule regime.

**Bounds.** `Primitives.bound_radius` and `_bound_radius`: `max(sx, sy, sz)` (a sphere of the largest
semi-axis contains the ellipsoid about its own origin) — `torch.maximum(torch.maximum(s0, s1), s2)`
from the sizes the caller passed, so a `size_scale` is already in it. `_BOUNDED_KINDS` is **left
unchanged** in the first patch: membership is a MEASUREMENT (`r6_kind_cost.json`, ratio > 1 rule), so
ellipsoid stays unbounded (always tested — correct, only slower) until `r5_kind_cost.py` prices it on
the certification host; by construction it costs a capsule-class intersector (one frame change, one
quadratic) so it is expected to qualify. `_kind_tensor`'s allowed set gains 4.

**Interface, smallest.** `raycast3d.ELLIPSOID = 4`; `_ellipsoid(o, d, pos, R, size)`; one branch in
`intersect`, one in `_tri_isect`, one in `_kind_tensor`, two in the bound helpers. `cpu3d.ELLIPSOID
= 4` and `MOVER_KINDS += (ELLIPSOID,)` — nothing else in cpu3d changes: the window and the light
footprint read radii from `bound_radius`, the shade path is kind-agnostic. Total on the order of 60
lines plus tests; no architecture change in either module (Step-7 rule).

## 5. Design-validating measurement (local, CPU, no `stable_envs/` edit)

`scratch/step7/dmc-quadruped/proto_cpu3d.py` builds `cpu3d.Scene` for camera 2 and renders through
`cpu3d.render` and the shared `raycast3d.cast_nearest`/`any_blocking` **verbatim**, with the §4.1
intersector supplied by monkeypatching `raycast3d.intersect`, `Primitives.bound_radius` and
`cpu3d.MOVER_KINDS` at runtime. Poses come from MuJoCo's own `mj_kinematics`/`mj_comPos`/
`mj_camlight` as `DMControlTrackCom._cpu_poses` does. Reference: 64 states from the domain's own
visited distribution (`initialize_episode` + 0-40 random steps, seed 100 — `visual_calib.py`'s first
draw; `phase0/corpus/states.npz`), rendered by `physics.render` under BOTH arms (mujoco 3.12.0 /
dm_control 1.0.45 and the declared 3.11.0 / 1.0.44) with the segmentation buffer saved. Metrics are
`stable_envs.testing.conformance.pixel_agreement` and `.ssim` — the conformance suite's own.

**Reference-version drift first** (`phase0/official_311_vs_312.json`): official 3.11 vs official
3.12 on the same states, camera 2: 84 px mean **0.583**/255, 70.8 % exact, 99.99 % within 8, max 11;
floor 0.72, sky 0.001, body 0.07; segmentation identical. (224 px: 0.445.) This is the humanoid case,
not the cheetah/hopper/walker case (their official frames move 1.99/2.41/10.02 with floors at
14.8-25.2): a 10 x 10 plane under `texuniform` shifts by a whole number of checker periods when 3.12
re-anchors the texture, so the cell is essentially immune to the drift that produced Step 4's three
FAILs. The declaration still goes under the family's 3.11 pair (§7); the numbers barely depend on it.

**Prototype vs official** (`phase0/proto/proto_results_cam2.json`; contact sheets
`sheet_{84,224}px_official311_proto_diffx8.png`: official / prototype / |diff| x8):

| variant (camera 2) | vs arm | mean /255 | all-pixel exact | within 8 | SSIM | max |
|---|---|---:|---:|---:|---:|---:|
| **84 px, ellipsoid, trunc** (the proposal, frames as MJWarp captures them) | 3.11 | **5.273** | 0.103 | **0.826** | **0.881** | 241 |
| | 3.12 | 5.327 | 0.102 | 0.821 | 0.879 | 241 |
| 84 px, ellipsoid, round (GL quantiser) | 3.11 | 5.262 | 0.134 | 0.829 | 0.881 | 241 |
| 84 px, ellipsoid as its **bounding sphere** (error-sizing only) | 3.11 | 6.021 | 0.101 | 0.816 | 0.874 | 241 |
| 84 px, ellipsoid **omitted** (what cpu3d would draw if it did not refuse) | 3.11 | 6.844 | 0.101 | 0.817 | 0.866 | 241 |
| **224 px, ellipsoid, trunc** | 3.11 | **3.552** | 0.150 | **0.901** | **0.952** | 255 |
| 224 px, ellipsoid, round | 3.11 | 3.478 | 0.253 | 0.902 | 0.952 | 255 |
| 224 px, bounding sphere / omitted | 3.11 | 4.389 / 5.193 | 0.148 / 0.148 | 0.889 / 0.891 | 0.943 / 0.938 | 255 |

For calibration, the family's declared 3.11 numbers (mean / within-8 / SSIM): cheetah 5.01 / 0.882 /
0.661, hopper 4.35 / 0.905 / 0.684, walker 8.07 / 0.700 / 0.826, humanoid 6.26 / 0.735 / 0.868.
Quadruped at 84 px sits inside that band on every metric (its SSIM is the best of the five: 80 % of
its frame is floor, whose structure the checker carries, and only 15 % is the speckled sky).
`interior_exact_frac` is undefined at 84 px (`interior_frac` 0.0003) as on three of the five.

**Where the residual sits** (84 px, ellipsoid/trunc vs 3.11; per-surface off the 3.11 segmentation,
`resid` = share of the summed |diff|; `floor_residual_split.json` for the near/far split):

| surface | share of frame | mean /255 | within 8 | share of residual | mechanism |
|---|---:|---:|---:|---:|---|
| floor, **near the body** (dilated body box + its mirror) | 34.4 % | 7.12 | 0.750 | **46.5 %** | the body's **reflection** (`grid` has `reflectance=.2`; MJWarp has no reflection pass, so neither does cpu3d) and the **shadow** (2048 shadow map vs the 0.3 any-hit factor) — visible as the mirrored quadruped and the shadow patch in the diff sheets |
| floor, **far** from the body | 45.8 % | 3.48 | 0.877 | 30.2 % | the checker's `mark="edge"` lines and the one-tap sampling (mild minification at this camera distance: far-floor luminance spread 14.6 official vs 16.1 ours, against 0.1 vs 9.7 on humanoid) |
| sky | 15.1 % | 3.67 | 0.908 | 10.5 % | the speckled gradient skybox sampled with one tap — the family's named mechanism (4.0-4.7 on the other cells) |
| body (18 primitives) | 3.4 % | 15.0 | 0.59 | 10.6 % | tessellated-vs-analytic silhouettes and GL vs fixed-function specular, as on the other cells (bodies 11-18 there) |
| **ellipsoid** | 1.17 % | **7.83** | 0.797 | **1.7 %** | with the real intersector; **25.9** (within-8 0.24, 5.0 % of residual) as a bounding sphere, **127** (21.7 %) omitted |
| sites (pupils) | 0.013 % | 131 | 0 | 0.4 % | 0-2 px per frame drawn by MuJoCo and by neither backend |

At 224 px the same split reads floor near 5.83 / far 2.18, sky 1.46, body 6.14, ellipsoid 4.21
(within-8 0.90). So: the ellipsoid intersector is worth 0.75/255 whole-frame over its bounding sphere
and 1.6/255 over omission, and after it the body's own residual is that of the other cells; the
remaining residual is the floor and it is the visual tier's known mechanisms plus one this cell has
more of than the others — the body's reflection under a camera 4.5 m away — which no backend here
draws. It is not closable in Phase 0 and is recorded as the cell's third named mechanism for
`VISUAL_NOTE`.

**Throughput, local, not certification** (Apple M5, 1 torch thread, render only, poses excluded):
84 px 76 / 116 / 165 fps at B = 1 / 16 / 64; 224 px 31 / 26 / 23 fps. Slower than the four certified
cells locally (228-888 fps at 84 px post-bounds) for structural reasons: 19 movers over a 4.6 % body
window, and the ellipsoid unbounded (§4.1) until measured. The certification numbers come from the
paired sweep on the batch partition (§7).

## 6. Tier

**Visual**, like the other four `DMControlTrackCom` cells, under the declared reference pair
(`_AT_VERSIONS = {"mujoco": "3.11.0"}`, `measured_with` mujoco_warp 3.11.0): the frame is MJWarp's
raytrace (cuda) / cpu3d's same picture (cpu) of dm_control's scene at dm_control's camera 2, not the
official pixels. The declared numbers must be measured the family's way — through `se.make(...)` on
the MJWarp path, 64 states x 3 independent draws (`scratch/dmcontrol/visual_calib.py`), ceiling clear
of the worst draw — on Oscar; the CPU prototype above predicts them (the cpu-3d workstream measured
CPU-vs-MJWarp at mean 0.004-0.018/255 on the sibling cells, so the 3.11 numbers here — 5.27 / 0.826 /
0.881 at 84 px — are the expected declaration to within ~0.1-0.3/255 of llvmpipe-vs-EGL spread).
Expected thresholds by the family's margin convention: `mean_abs_max` ~6.9, `within8_min` ~0.78,
`ssim_min` ~0.83, to be set from the three-draw spread, not from this one.

## 7. Certification obligations (for the cell to ship)

1. **Gate A, mandatory, both devices** (`tests/gpu/test_conformance.py` via `examples/fidelity_step4.py`
   / `scratch/step4/certify.py`): determinism, batch invariance, batch-size invariance, chunked render,
   order independence, render purity, style-is-state, set_state round trip, state dims matter (23),
   variations do something, per-env variation — cuda under `se-step5`/`se-vis311`, cpu under
   `se-vis311` (declared reference) with the 3.12 arm recorded as `blocked_environment` by design.
2. **Visual bounds vs the official 3.11 reference** (`check_visual_fidelity`): declared per §6 with
   `reference_versions`/`measured_with` fields; `check_resolution_consistency` needs a
   `DOWNSAMPLE_EXACT_FRAC` recorded for the new cell ON THE PROTOCOL IT IS APPLIED WITH (the walker
   defect, texture-fidelity S6), with a margin sized from the state-set spread.
3. **cuda digests** for the new frozen corpus, with the family's caveat that MJWarp's batched render is
   not run-to-run bit-reproducible (texture-fidelity S5: 1-2 px in 10,000 by one level), so the digest
   is a soft oracle and `check_visual_fidelity` the gating detector.
4. **Cross-backend** CPU vs MJWarp on the same tree (cpu-3d's `verify_pixels.py` method): expected
   >= 0.999 within-1 as on the siblings; the cross-backend bound question (cpu-3d open question 1)
   applies to this cell too.
5. **A new frozen corpus** under `examples/results/step7/dmc-quadruped/` (states, per-config contract,
   digests/metrics) proposed for the Step-8 freeze; the Step-4/5/6 corpora untouched.
6. **Replay of the frozen Step-4 configs** that route through the two shared files after the patch:
   the four trackcom cells on cpu and cuda and the 21 raycast cells (`examples/fidelity_step4.py`
   shards `dmc` and the maze/mujoco families) — identity expected, since kind 4 never occurs in them.
7. **Performance measured, honestly** (Gate F / PERFORMANCE_MEASURED): paired CPU sweep on
   `batch`/`normal` at 1 thread against the same-node official renderer, GPU sweep on a 3090; the cell
   need not be CPU_FAST/GPU_FAST to ship.
8. Semantic follow-up: the visual-tier DINO tripwire (bound 0.25) — the proportionate check for a new
   visual cell.

## 8. Pending cluster items (Oscar denied at auth throughout)

- Verify the RELEASE `mujoco_warp` sources (§4 paths) for the ELLIPSOID dispatch, the absence of site
  drawing, and the hfield path; record the version.
- MJWarp frames of `quadruped-walk` at camera 2 on the 64 corpus states (both venvs), per-surface vs
  official 3.11 on the same node — the cuda side of §5, and the three-draw declaration of §6.
- Items 1-8 of §7 after the patch; `r5_kind_cost.py` for ellipsoid's `_BOUNDED_KINDS` entry.

## 9. Artifacts

`examples/results/step7/dmc-quadruped/phase0/`: `geom_table_walk.json`, `state_sufficiency_walk_84px.json`,
`swm_model_roundtrip.json`, `official_311_vs_312.json`, `official_probe16{,_cam2}_84px.npz` + sheets
(`official_sheet{,_cam2}_84px.png`, `official_reset{,_cam2}_336px.png`), `corpus/` (`states.npz`, the
84 px camera-2 official stacks + segmentation for both arms, `provenance_mj31{1,2}0.json`; the 224 px
and camera-0 stacks are regenerable with `gen_states_ref.py --states corpus/states.npz` under each
venv and are left untracked, as Step 6 left its 4.6 MB stacks), `proto/` (`proto_results_cam2.json`,
`floor_residual_split.json`, `proto_84px_ellipsoid_trunc.npz`, the two contact sheets).

## 10. Open questions for the coordinator

1. **Camera and size**: register at camera 2 (SWM's) and the family's 84 px default, with 224 px
   measured and reported (the SWM `render()` default)? The other DMC cells are all camera 0; this is
   the first at another camera, and every code path reads `cls.CAMERA` already.
2. **Default appearance**: declare against the suite's default (`self` = .7 .5 .3, the family's
   convention and `reference_id dm_control/quadruped-walk`), exposing SWM's (.6 .3 .3) through the
   existing `material.self` knob — or against SWM's default?
3. **Swimmer side effect**: admitting ELLIPSOID in `cpu3d.MOVER_KINDS` makes `DMCSwimmerSwimmer6`
   drawable on `device="cpu"` (its refusal test flips). Certify swimmer-CPU in this step, or keep it
   refused by an explicit allow-list until certified?
4. **The reflection mechanism**: accept as the cell's third named residual (as the floor
   minification is for humanoid), or hold the cell until a reflection pass exists on both backends
   (out of Step-7 scope by the rules)?
5. Ownership: the ellipsoid patch touches `raycast3d.py` (raycast owner) and one tuple in `cpu3d.py`;
   this workstream can carry both if assigned.

---

# Phase 1 — `PATCH` (authorized): the ellipsoid capability, the cpu3d admission, and the cell

**Classification: `PATCH`** — three commits on `step7/dmc-quadruped` over the Phase-0 findings commit
`494b747`: (a) `e120029` raycast3d ELLIPSOID + tests, (b) `68b411c` cpu3d admission + swimmer
allow-list, (c) the commit carrying this section: `DMCQuadrupedWalk` registered at the SWM default
with a PROVISIONAL visual declaration, plus scripts, artifacts and the prepared sbatch files.
Nothing under `examples/results/step4|5|6/` touched. Oscar still denied: every GPU/certification-host
item is prepared (§P4) and pending.

## P1. What changed (coordinator answers applied)

| file | change |
|---|---|
| `stable_envs/render/raycast3d.py` | `ELLIPSOID = 4`; `_ellipsoid` (torch, §4.1 closed form: frame change by `_mtv`, scale to the unit sphere, sphere quadratic with a leading coefficient, nearer positive root, `disc > 0`/`t > 0`, normal = gradient `p/size^2` by `_mv`; a zero semi-axis is a MISS, never a division); the `kind == 4` branch of `_tri_isect`, line for line, with `div_rn`/`sqrt_rn`; `intersect`, `_kind_tensor`; bound helpers `max(size)` in `Primitives.bound_radius` and `_bound_radius`; `_BOUNDED_KINDS` UNCHANGED (measurement pending, `certify_cpu.sbatch` step 5) |
| `stable_envs/render/cpu3d.py` | `MOVER_KINDS += (ELLIPSOID,)` — the one-tuple change; nothing else is kind-specific |
| `stable_envs/envs/dmcontrol.py` (TrackCom region + registration block only; W3's `DMControlSuite` region untouched — hunks at `TRACKCOM_DOMAINS`, `DMControlTrackCom`, and the module tail after `TRACKCOM_RENDER_FLAGS`) | `CPU_UNCERTIFIED` allow-list (swimmer refused by NAME on `device="cpu"`, answer 3); `_swm_quadruped_env` (SWM's model: `make_model(floor_size=10)`, the mjcf round trip with SWM's default `init_value`s — `self` rgba .6 .3 .3, `foot_back_left` density 1000 — `Move(desired_speed=0.5)`, `action_scale`); `_SWMQuadrupedModel` mixin overriding `_bundle`/`sample_states`/`reference_env` and `default_size = 224` (answers 1, 2; a loader hook on `_DMCModel` is the cleanup once W3 lands); `DMControlTrackCom._used_materials` restricted to GEOM wearers (see P3); `TRACKCOM_DOMAINS["DMCQuadrupedWalk"] = ("quadruped", "walk", 2)`; `_QUADRUPED_AT`/`_QUADRUPED_AT_MEASURED_WITH`; the `VISUAL_FIDELITY` entry; `QUADRUPED_NOTE` (the reflection residual, answer 4); `TRACKCOM_BASES` in `_make_classes` |
| tests | `tests/unit/test_raycast3d_ellipsoid.py` (new: axis-aligned/rotated/inside/miss closed forms, float64 fuzz with a DERIVED tolerance — 16 ulp × the `eps·(|o−pos|/size_min)²` cancellation term, measured 0.2–2.1× eps/2 — normals checked at the returned point, sphere equivalence, degenerate sizes, dispatch/bounds); `test_raycast3d_primitive_bounds.py` (kind 4 in the sampler + 5 pinned cases incl. degenerate, tangent at 100 m, admission edge); `tests/gpu/test_fast_paths.py` (unknown-code test moved to code 8 = SDF; new `test_the_ellipsoid_twins_are_bit_identical_on_both_casts`, cuda); `tests/unit/test_cpu3d.py` (swimmer allow-list; trackcom material census); `tests/unit/test_declarations.py` (the two new `_QUADRUPED_AT*` names in the AST want-set) |
| scripts | `scratch/step7/dmc-quadruped/measure_cell.py` (ours / official / compare), `gateA_local.py`, `cross_backend.py`, `gpu_declare.sbatch`, `certify_cpu.sbatch` |

## P2. Verification, local (Apple M5, `se6-venv`, `MUJOCO_GL=glfw`, 1 torch thread)

**Identity of everything that renders through the two shared files** (`phase1/replay/`,
`scratch/step6/raycast3d/replay_step4.py --dump-dir` on each tree, `--compare-dumps`; baseline =
`git archive 76dc511` of `stable_envs/`): **14 of 14 comparable configs byte-identical**
(`all_identical: true`, 0 of 451,584 px differing per config) — the 10 raycast3d parity cells
(`DMCAcrobotSwingup, DMCBallInCupCatch, DMCCartpoleBalance, DMCCartpoleSwingup, DMCFingerSpin,
DMCFingerTurnEasy, DMCFingerTurnHard, DMCPendulumSwingup, DMCReacherEasy, DMCReacherHard`) and the
4 trackcom cpu3d cells (`DMCCheetahRun, DMCHopperHop, DMCWalkerWalk, DMCHumanoidStand`). The 11
OGBench maze cells (Antmaze/Humanoidmaze/Pointmaze ×) **could not run locally** — `ogbench` is not in
the local venv (`ModuleNotFoundError`, both legs) — so their CPU identity is `certify_cpu.sbatch`
step 1 and the cuda identity of all 25 is `gpu_declare.sbatch` step 2. (The replay's "fail=14" is
the frozen Step-4 CUDA digests not matching CPU frames, expected and irrelevant to the identity.)

**Unit suite**: `tests/unit`: **734 passed, 6 skipped** with `test_examples.py` deselected (runs every
example script; not run here) and `test_repo_contracts.py::test_every_cited_scratch_path_resolves`
deselected — it fails on the baseline too (`scratch/carracing/logs/`, `scratch/robocasa/out/corpus`,
`scratch/sboughan/...` from Step-6 texts); two malformed citations it flagged in the Phase-0
findings.md were mine and are fixed in this commit. The new raycast3d tests: 146 passed.

**The registered cell vs official** (`phase1/cell/`, `measure_cell.py`: 64 states from the cell's own
`sample_states` seed 100; official = `physics.render` of the SAME SWM-built model — model digest
asserted equal across the two interpreters — under the declared 3.11 pair and 3.12):

| registered default: **224 px, camera 2, SWM appearance**, CPU | mean /255 | all-exact | within 8 | SSIM | max |
|---|---:|---:|---:|---:|---:|
| **vs official mujoco 3.11.0 + dm_control 1.0.44** | **3.417** | 0.153 | **0.902** | **0.957** | 237 |
| vs official 3.12.0 + 1.0.45 | 3.451 | 0.153 | 0.900 | 0.955 | 237 |
| supplementary, 84 px, vs 3.11 (one draw, not the certified row) | 6.877 | 0.102 | 0.813 | 0.837 | 255 |

Per residual at 224 px vs 3.11 (segmentation of the official frame; `resid` = share of Σ|diff|):

| surface | share of frame | mean /255 | within 8 | share of residual |
|---|---:|---:|---:|---:|
| floor near the body (reflection + shadow) | 34.9 % | 5.54 | 0.815 | **56.6 %** |
| floor far (checker edges, one-tap sampling) | 45.3 % | 2.20 | 0.957 | 29.1 % |
| sky (speckled gradient, one tap) | 15.1 % | 1.47 | 0.955 | 6.5 % |
| body, 18 primitives | 4.6 % | 5.37 | 0.844 | 7.3 % |
| **ellipsoid torso** (the new intersector) | 1.2 % | **3.51** | 0.914 | 1.2 % |
| sites (pupils; drawn by neither backend) | 0.01 % | 148 | 0 | 0.5 % |

Local throughput (NOT certification; render only): 224 px 27 / 23 / 25 fps at B = 1/16/64;
84 px 73 / 161 / 164 fps.

**Gate A, local** (`phase1/gateA/cpu_local_mj3120.json`, `gateA_local.py` = `fidelity_step4.measure`
with the config dict the frozen universe does not yet carry; the local arm is mujoco 3.12, so the
reference-judged checks are recorded, not judged): **every mandatory check at its passing
observation** — determinism identical; batch, batch-size (1/7/64), chunked (1/7/64/4096), order,
duplicate, style-is-state, default-style, input-form: max |diff| 0; render pure; set_state copies
and round-trips (0.0); `state_dims_matter`: no dead dim of 23; `per_env_variation`: no collapse;
`variations_do_something`: **no inert variation** (P3); `fidelity_declaration` agrees.
Family-specific: `colors_apply_exactly` pass; `check_visual_fidelity` (3.12 arm) 3.447 / 0.900 /
0.955 — inside the declared ceilings even on the undeclared arm; `check_resolution_consistency`
**0.8119** exact on its own 8-state protocol (seed 20260809) — this is the number the cell's
`DOWNSAMPLE_EXACT_FRAC` entry must record (no bound exists yet; Step-8 freeze), with a margin sized
from the state-set spread per texture-fidelity §6. The candidate frozen corpus is
`phase1/corpus/states/DMCQuadrupedWalk.npz` (64 of a 256 pool, `corpus_for`'s selection, sha256
`2bf9cd38…`). 222 s.

## P3. Two things found while landing it

1. **A material knob no backend could honour.** `_DMCModel._used_materials` counts site wearers, and
   quadruped's `eye` material is worn only by the two pupil SITES, which MJWarp and cpu3d never
   draw: the base census declared `material.eye` and `check_variations_do_something` would have
   failed on it. `DMControlTrackCom._used_materials` now counts geom wearers only; on the four
   certified cells and swimmer the list is unchanged (asserted in `test_cpu3d.py`), on quadruped
   it is `grid, self, effector`.
2. **The 84 px picture is worse than Phase 0's** (6.88 vs 5.27 mean): different states (SWM's foot
   density changes the visited distribution) and SWM's darker `self`; reported as supplementary only.

## P4. Pending on Oscar (scripts prepared, never run)

`scratch/step7/dmc-quadruped/gpu_declare.sbatch` (3090, `norm-gpu`): the release `mujoco_warp` source
check; cuda identity branch-vs-baseline on all 25 configs; the ellipsoid twin test + fast paths;
**the declaration draws** (`visual_calib.py`, 3 × 64, 224 px, under `se-vis311` — replaces the
PROVISIONAL CPU-measured entry and its `_QUADRUPED_AT` string); Gate A on cuda on the new corpus,
both arms; `localize.py` stacks (the cross-backend oracle). `certify_cpu.sbatch` (`batch`/`normal`):
CPU identity on the certification family incl. the 11 maze cells; Gate A on CPU under the declared
pair; cross-backend CPU vs MJWarp per surface (`cross_backend.py`); the paired sweep (the tier);
`r5_kind_cost` for the ELLIPSOID `_BOUNDED_KINDS` entry; the unit suite. Prerequisites for the
coordinator: `$SCR/se7/dmc-quadruped` (rsync of this worktree with `GIT_COMMIT`) and
`$SCR/se7/baseline` (a pristine 76dc511 checkout). The universe entry for the Step-8 freeze
(`scratch/swm_matrix.json` / census) is not made here.

## P5. Review follow-up (commit (d), APPROVE-WITH-CHANGES on 9188c2c)

Required, all done in this one commit: (1) `_swm_quadruped_env` scopes its export directory to the
call (`TemporaryDirectory`, removed once `Physics.from_xml_path` has compiled it) — measured 0 new
directories over 3 `sample_states` + 1 `reference_env` (57 stale ones from the old code deleted);
(2) `QUADRUPED_NOTE` now declares residual (4): the two pupil sites and the red torso site that
`physics.render` draws and neither backend does (~50 of 50,176 px, max 186/255, 0.5 % of the
residual); (3) the stale swimmer comment in `_cpu_scene`'s handler now states the allow-list.
Recommended: `test_the_swm_quadruped_loader_overrides_are_the_base_methods_with_the_load_line_swapped`
pins the three overrides to `_DMCModel`'s source (docstring dropped, one load line swapped, 84->224
for `reference_env`) so the copy cannot drift; the Gate-A artifact regenerated on the (d) tree —
identical verdicts, 18 checks measured, visual 3.447 / 0.900 / 0.955, downsample 0.8119 — with
`git_commit 9188c2c, dirty true` recorded, which is what an artifact committed alongside the code
can say (the cluster re-run stamps the merged SHA through `STABLE_ENVS_GIT_COMMIT`); the 11
untracked regenerable `.npz` stacks deleted (kept: `states.npz`, the 84 px camera-2 official stacks
of phase 0, and `ours`/`official-3.11` at 224 px of phase 1 — what the committed numbers rest on).
Tests after (d): `tests/unit` 735 passed, 6 skipped (same two deselections); raycast3d suites inside
it. No render-path file changed in (d) (dmcontrol loader, note, comment, tests only), so the 14/14
identity of P2 stands as measured.

## P6. Cluster results (Oscar, tree `fa4ac53` at `$SCR/se7/dmc-quadruped`, merged as `0a029bd` on `step7/integration`; baseline `$SCR/se7/baseline` = `git archive 76dc511`)

Jobs: **6394721** `se7-dmc-quadruped-gpu` (gpu2104, RTX 3090, driver 575.57.08, 5:56), **6394987**
`se7-dmc-quadruped-gpu2` (gpu2107, same card class, 3:47 — steps 4-6 re-run; see the EGL note),
**6394722** `se7-dmc-quadruped-cpu` (node1913, `batch`, 58:53), **6395512** `se7-dmc-quadruped-sweep`
(node2330, 0:09, load-aborted). Artifacts under `phase1/gpu/` and `phase1/cpu_oscar/`. Untracked on purpose (regenerable by
the sbatch set): `gpu/localize_311/frames/DMCQuadrupedWalk.npz` and `gpu/localize_312/frames/
DMCQuadrupedWalk.npz` (8 MB each, ours + official + segmentation stacks), `gpu/localize_312/sheets/`
(the 3.12 duplicate of the committed 3.11 sheets), and every `*.log` (gitignored). Everything else
from the four jobs is committed.

**Identity of every configuration that renders through the two shared files, branch vs 76dc511:**
- cuda, one card each: **21/21 raycast3d configs byte-identical** (`cuda_identity_raycast21.json`,
  `se-step6`, the venv with ogbench+triton) and **4/4 trackcom cells byte-identical**
  (`cuda_identity_trackcom4.json`, `se-step5`, the venv with mujoco_warp);
- cpu, certification family (Xeon, `se-step6`): **25/25 byte-identical** incl. the 11 maze cells the
  local run could not reach (`cpu_identity25_node1913.json`).
Ellipsoid twin test on cuda: `test_the_ellipsoid_twins_are_bit_identical_on_both_casts` **passed**;
the raycast3d fast-path suite on cuda: 120 passed, 22 skipped (`fast_paths_*.txt`). The release
`mujoco-warp 3.11.0` (`se-vis311`) dispatches `GeomType.ELLIPSOID` (render.py:283) and `HFIELD`
(:263) and defines `ray_ellipsoid` (ray.py:303) / `ray_hfield` (:452); no `site_` reference in
render.py (`mujoco_warp_source_se-vis311.txt`) — §4 confirmed on the release.

**The declaration (replaces the provisional entry in this commit)** — `visual_calib.py`, 3 independent
draws × 64 states, 224 px, camera 2, SWM's default appearance, through `se.make(..., device="cuda")`
(MJWarp), against `physics.render` under the declared **mujoco 3.11.0 + dm_control 1.0.44**
(`visual_calib_311/visual_DMCQuadrupedWalk.json`):

| arm | mean /255 (worst) | within 8 (worst) | SSIM (worst) | all-exact | interior exact |
|---|---:|---:|---:|---:|---:|
| **3.11 (declared)** | **3.523 (3.527)** | **0.8973 (0.8964)** | **0.955 (0.9536)** | 0.171 | 0.204-0.208 |
| 3.12 (record) | 3.562 (3.566) | 0.895 (0.8939) | 0.9533 (0.9518) | 0.171 | 0.205-0.206 |
| 3.11, 84 px (supplementary) | 5.387 (5.391) | 0.8139 (0.8132) | 0.8753 (0.8734) | 0.095 | 0.05-0.09 |

Declared: `mean_abs 3.52, within8_frac 0.897, ssim 0.955`, thresholds by the family's margin
convention `mean_abs_max 4.6, within8_min 0.85, ssim_min 0.90`, `draw_spread` recorded,
`reference_versions {mujoco: 3.11.0}`, `measured_with {mujoco_warp: 3.11.0}`, `_QUADRUPED_AT`
naming the card, driver and versions. The local CPU one-draw number (3.42 / 0.902 / 0.957) sat
0.1/255 below it, as the sibling cells' CPU-vs-MJWarp spread predicted. Per surface at the
declaration (`localize_311/localize.json`, MJWarp vs official 3.11): floor share 0.806, mean 3.75,
86 % of the residual; bodies 0.046, mean 6.30, 8 %; skybox 0.148, mean 1.35, 6 %.

**Gate A on the new corpus** (`gateA_local.py` = `fidelity_step4.measure`; 64 of 256 states,
`corpus/states/DMCQuadrupedWalk.npz`): **every mandatory check at its passing observation on BOTH
devices under BOTH arms** — determinism identical; batch / batch-size / chunked / order / duplicate /
style-is-state / default-style / input-form max |diff| 0; render pure; set_state copies, round trip
0.0; 23/23 live dims; no per-env collapse; no inert variation; declaration agrees — plus
`colors_apply_exactly` pass. `check_visual_fidelity` under the declared 3.11 pair: **cuda 3.509 /
0.900 / 0.956, cpu 3.375 / 0.9035 / 0.9586** (both inside 4.6 / 0.85 / 0.90); the 3.12 arm 3.549 /
0.897 / 0.954 (cuda), 3.417 / 0.901 / 0.956 (cpu) — recorded, to be `blocked_environment` by the
certifier's version rule. `check_resolution_consistency` **0.812** (cuda) / **0.8119** (cpu) on the
8-state protocol, the number the cell's `DOWNSAMPLE_EXACT_FRAC` entry records at the Step-8 freeze.
Oracle digests: cuda `7c298c925f6439e1` (3.11 arm) / `3f668f6dd8a46bca` (3.12 arm) — MJWarp's
batched render is not run-to-run bit-reproducible on this family (texture-fidelity §5), so these
are soft oracles; cpu `da826751c4933013` under both arms (the CPU backend is bit-reproducible within
a CPU family).

**Cross-backend, CPU vs MJWarp, same states, same tree** (`cross_backend_31{1,2}_node1913.json`):
mean **0.0069/255**, **99.40 % exact**, **99.994 % within one level**, 99.998 % within two, 48-50 px of
3.2 M beyond two levels (max 125: silhouette hit/miss flips between two analytic intersectors, as
on the siblings), per surface: sky 0.023, floor 0.003, body 0.029, **ellipsoid 0.004-0.005 (99.74 %
exact)**. CPU-vs-official and cuda-vs-official agree to three decimals (3.531 / 0.8969 / 0.9567 on
the 3.11 stack). Same picture, different backend — the Step-6 policy, measured.

**Tier — `PERFORMANCE_MEASURED` NOT reached on the certification host:** the paired sweep
(`certify_shard_cpu3d.py --max-load 12`) **aborted at start twice** — node1913 at load 34.5 (in
6394722) and node2330 at load 32.4 (6395512) — the shared-batch-partition situation Step 6 recorded
(three attempts, best node 1.65x reference spread); no tier is derived from an aborted shard
(`certifiable: false`, `not_certifiable_because` in both files). Reported, not certified: local
Apple M5, 1 thread, render only — 224 px 27 / 23 / 25 fps at B = 1/16/64, 84 px 73 / 161 / 164 fps.
Coordinator's call whether to re-attempt on a quiet window or accept `CPU_UNMEASURED` for the
Step-8 freeze; the GPU sweep was not part of this job set.

**`_BOUNDED_KINDS` — ELLIPSOID stays UNBOUNDED, and the measurement is recorded as undecided.**
`r5_kind_cost.py` (now pricing the kind) ran three times on node1913 at load ~34
(`kind_cost_ellipsoid_node1913_run{1,2,3}.json`, aggregate `kind_cost_ellipsoid_node1913_agg.json`):
ELLIPSOID's worst ratio per run 0.89 / 0.69 / 0.90 and best 1.43 / 1.21 / 1.37 — it straddles 1, so
by the rule (> 1 at every width in every run) it is not bounded; and the same runs put **BOX at
0.95** in run 2 against the frozen table's 1.11-1.13, i.e. the node was too loaded to reproduce the
committed predicate. `_BOUNDED_KINDS` is untouched, and the aggregate IS committed under the
canonical name the bounds test reads (`cpu_oscar/kind_cost_ellipsoid.json`, load evidence in-band):
the test's complement clause was wrong in (e) (`max(best) < 1`, unsatisfiable for a kind that
straddles 1 whichever way the predicate went — re-check of ad842a8) and is now the true complement of
"bounded iff worst > 1 in every run", i.e. `min(worst) <= 1` — under which this straddling
measurement is an honest, green, UNBOUNDED verdict. The quiet-window job (6396525) re-takes the
three runs with loadavg recorded in-band; if a clean measurement says otherwise the file is replaced
in (f). Correctness is unaffected either way — unbounded means always tested.

**Unit suite on the certification host** (`unit_tests_node1913.txt`, `se-step6`): 719 passed, 22
skipped, 1 failed = `test_every_cited_scratch_path_resolves`. At ad842a8 that test's broken set had
grown from the baseline's 9 (Step-6 citations) to 21: twelve absolute `/oscar/scratch/...` paths
inside four committed stdout captures of this workstream (`egl_probe_*`, `mujoco_warp_source_*`) —
scrubbed to `$SCR/...` in (e2), which returns the set to the baseline's 9.

**Provenance of the committed artifacts (e2).** The Gate-A shards under `gpu/` and `cpu_oscar/`
were measured at tree `fa4ac53`, i.e. BEFORE the shipped declaration: their
`check_fidelity_declaration` evidence shows the provisional entry (3.42 / 0.902 / 0.957), and their
`check_visual_fidelity` observations are judged here against the shipped thresholds (4.6 / 0.85 /
0.90), which they clear; the sbatch set re-run at the merged SHA re-stamps them. `visual_calib_*`
and the three identity JSONs now carry host / job / commit / card stamps; `interior_exact_frac` in
the declaration is the MEAN over draws (0.206), as the siblings declare it.

**EGL note (not a node fault).** 6394721's log showed `OpenGL.raw.EGL._errors.EGLError` after every
official-renderer call and I resubmitted steps 4-6 as 6394987 with gpu2104 excluded. The lines were
`Exception ignored in GLContext.__del__` — the EGL `free` at interpreter exit — which 6394987's
probe printed right after `EGL_OK` on gpu2107 too, and 6394721 had already written its
`visual_calib_311`, `gateA_cuda_*` and `localize_*` outputs before 6394987 started. Both nodes
rendered the official reference; the outputs kept are 6394987's (they overwrote 6394721's, same
states, same card class). No NODE-2 entry; W3's identical observation on gpu2104 agrees.

## P6.1 INTEG-1 (after W3's merge, integration `1d9dccb`): the loader hook replaces the copies

W3 changed `_DMCModel._bundle` (`geom_type`/`geom_size` now in `_pristine` for its shape leaf) and
`_SWMQuadrupedModel._bundle`, a copy, drifted — exactly what the drift test in (d) existed to
catch, and the answer is to stop copying. `step7/integration` merged into this branch (clean), then:
`_DMCModel._load_env(cls, seed)` is now THE ONE PLACE a cell's model comes from — `suite.load` by
default, called by `_bundle`, `sample_states` and `reference_env` (whose `size` now defaults to
`cls.default_size`); `_SWMQuadrupedModel` overrides `_load_env` (`_swm_quadruped_env`) and
`default_size = 224` and nothing else, and `tests/unit/test_cpu3d.py::
test_the_swm_quadruped_mixin_overrides_only_the_loader_hook_and_the_size` asserts exactly that
(the three methods are the base class's objects; every base loader goes through the hook). The base
edit is three lines plus the hook; W3's `DMControlSuite` region is untouched.

Verified on the merged branch: `tests/unit` **812 passed, 7 skipped, 1 failed** (the same
pre-existing citation test; its set is the baseline's 9); **CPU byte-identity 8/8 vs the integration
tree `1d9dccb`** — `DMCCheetahRun, DMCHopperHop, DMCWalkerWalk, DMCHumanoidStand, DMCQuadrupedWalk,
DMCManipulatorBringBall, DMCReacherHard, DMCCartpoleSwingup`, sampled states seed 0, 0 px differing
(`phase1/replay_hook/cpu_hook_vs_integration_identity.json`): the hook moved no pixel and no state.
