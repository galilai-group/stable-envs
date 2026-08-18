# Fetch (gymnasium-robotics): landing the family the criterion already cleared

**Verdict: GO, landed.** All 12 `swm/Fetch*-v3` ids are covered by 12 registered `stable_envs`
environments over 4 distinct scenes. Interior-pixel agreement with the official renderer is
**0.9988–0.9990**, whole-frame exactness **0.922–0.937**, mean |diff| **0.79–0.85/255**, at
**5,885–6,027 f/s** against the official EGL renderer's 923–994.

The interior figure was **0.9919–0.9948** until 2026-08-11, and the numbers below §4 are the older
pass. What moved it is not a change to this family: `mujoco_warp.render` defines its megakernel inside
the call, so a CUDA graph captured outside a `with gl_rounding()` block replays the *truncating*
float→uint8 kernel for the rest of the run. Against correct rounding the captured graph differed on
15.8 % of Fetch interior channels — signed `{0, -1}` and nothing else, the signature of a quantisation
swap — and was bit-identical to the truncating kernel. Capturing inside the block moves `FetchReach`
from 0.99230 to 0.99880, `FetchPickAndPlace` to 0.99880 and `FetchSlide` to 0.99901. The declared
`parity_interior_min` is still 0.99, and the `parity_note` on `FetchBase` still quotes the pre-fix
per-scene values.

[`swm_coverage.md`](swm_coverage.md) recorded Fetch as "GO on the criterion, blocked on the renderer",
with the block attributed to `nmesh = 18` and `ntex = 2`. One of those two was the real obstacle and
the other was never one:

* **the textures are not a problem, and were not one even before the alpha correction.** Fetch has two
  textures — a gradient skybox and `texture_block` — and `texture_block` is declared in `shared.xml`
  and **used by no material**. Every material in every Fetch scene is flat, with `reflectance=0`.
  `texture_gaps()` returns the empty list on all four models, so the one real `mujoco_warp` texture
  limitation (UVs for `PLANE` and `MESH` only) never fires here.
* **the alpha bug was not Fetch's blocker either.** Fetch has *zero* geoms at alpha 0. It has four at
  alpha 0.7 and 0.1, which is the case the correction box in [`backend3d.md`](backend3d.md) says has
  "no correct handling at all". They are excluded, and §5 measures exactly what that costs.
* **the 18 meshes were the real obstacle**, and the answer is not a new ray-caster. Every one of them
  is a link of the robot arm and every joint is in `qpos`, so nothing about them bakes;
  `stable_envs.render.raycast3d` ships analytic primitives only. The mover pass is therefore
  `mujoco_warp`'s batch raytracer, which has mesh BVHs, and the static pass is the OGBench plate.

Everything below was measured on this cluster (one H200, driver 590.48.01) through
`scratch/fetch/probe{1,2,3,4,5,6}*.py`, `scratch/fetch/calibrate.py` and `scratch/fetch/attribute.py`.
Raw JSON in `/mnt/vast/home/randall.balestriero/stable_envs_logs/fetch/`.

---

## 1. The numbers that ship

64 states per scene drawn from the env's own pool, 480 px (gymnasium-robotics' native `DEFAULT_SIZE`),
scored through `reference_render`.

| env | interior exact | all-pixel exact | mean /255 | within 8 | f/s @256 |
|---|---:|---:|---:|---:|---:|
| `FetchReach` (+`Dense`, +`Dict`) | **0.992240** | 0.90921 | 0.8376 | 0.97456 | 5,923 |
| `FetchPush` (+`Dense`, +`Dict`) | **0.991894** | 0.90832 | 0.8707 | 0.97420 | 5,920 |
| `FetchSlide` (+`Dense`, +`Dict`) | **0.994751** | 0.92804 | 0.7919 | 0.97784 | 6,027 |
| `FetchPickAndPlace` (+`Dense`, +`Dict`) | **0.992075** | 0.90847 | 0.8561 | 0.97445 | 5,885 |

`parity_interior_min = 0.99`, declared with the `parity_note` §5 reproduces. The three ids per scene
render the same picture: `Dense` selects a reward and `Dict` an observation container, and neither
reaches a pixel — the same relationship OGBench's `navigate`/`stitch`/`explore` suffixes have.

Throughput, same node, 480 px:

| | 480 px | 224 px | note |
|---|---:|---:|---|
| ours, end to end at batch 256 | **5,906–6,027** | **16,202** | plate composite, channel swap and stream sync included |
| the ray-cast pass alone | 11,051–11,835 | — | i.e. the composite costs about half |
| official EGL, render only, warm | 923–994 | 2,258–2,563 | `scratch/fetch/fps_official.py` |
| **speedup, renderer against renderer** | **6.19×** | **6.47×** | `scratch/fetch/probe7_interference.py` |
| official EGL, from a stored state | 118–148 | — | `reference_render`, i.e. including the `mj_forward` a caller needs to restore the state at all |

Measured at BOTH resolutions on purpose, because on a small scene MuJoCo's rasterizer is genuinely
competitive — it costs triangles where a ray-caster costs rays, so the advantage narrows as the scene
shrinks and the frame grows. Fetch is a small scene, and it is a win at both: **6.19× at 480 and 6.47×
at 224**. The 224 px figure of 16,202 f/s also clears `when_does_this_pay_off.md`'s G4 bar for this
family, the 6,762 f/s/core JPEG-decode baseline, by 2.4×.

The end-to-end figure against a *stored state* is larger still (~45×) because our path never runs a
physics step where MuJoCo's does — `reference_render` pays three `mj_forward` calls per frame, and on a
settled scene the constraint solver costs more than the raster. The renderer-against-renderer number
is the conservative one and it is the one quoted.

---

## 2. The design: a baked plate plus a mesh ray-caster

The same shape as the OGBench mazes, with `mujoco_warp` where `raycast3d` sits there.

**The static half is provably static, which is the part DM Control cannot claim.** Every Fetch
material has `reflectance=0` and the model's one light is `castshadow="false"`, so a mover casts no
shadow and appears in no mirror. Measured rather than asserted: with every mover hidden, 8 states 25
random actions apart give **0/255 max difference**, and `FetchBase._check_plate_is_constant` re-runs
that at construction so the design cannot rot. Floor, table and skybox are **91% of the frame**, and
baking them through MuJoCo makes them exact by construction — the attribution in §4 shows those three
classes at **interior exact 1.0**.

**MJWarp's segmentation buffer decides which pixels are movers**, so occlusion of the robot by the
table is resolved by the ray-caster rather than by comparing two depth buffers with an epsilon.

**Supersampling was measured and rejected.** 2× costs 2.9× the time and moves whole-frame exactness by
+0.0016 (0.92313 → 0.92472 on Reach). The residual is shading, not coverage, so antialiasing cannot
reach it.

---

## 3. Five constants that read plausibly and are wrong

Each cost a measurement; each is invisible by eye.

**The official renderer's camera is a FREE camera and MJWarp only has model cameras.** The pose is
recovered from MuJoCo's own `mjvScene` as the **average of the two `mjvGLCamera` eyes** — they are
`vis.global_.ipd` = 0.068 apart — and then written into an added `se:cam`. Proved rather than trusted:
our model rendered through MuJoCo at `se:cam` agrees with the official free-camera frame on **99.99%
of interior pixels** and 98.6% of all pixels (the rest being the markers of §5).

**MJWarp resolves a geom's colour from its material whenever it has one; MuJoCo prefers the geom's own
`rgba` unless that is the compiler default `(0.5,0.5,0.5,1)`.** Fetch trips this on 11 of 24 geoms:
`robot0:arm_mat` is white and is worn by both the white and the blue arm links, so MJWarp paints the
whole arm white. Fixed by cloning **one material per geom**, carrying the resolved colour and the
original specular/shininess — the shading path is untouched. The mirror image of this bit later: the
plate bake writes `mat_rgba` and MuJoCo reads `geom_rgba`, so `floor.color` and `table.color` were
INERT in the plate while working perfectly in the ray-cast. Both arrays are now written.

**MuJoCo draws sites; MJWarp draws none** — `grep site mujoco_warp/_src/render.py` is empty. `target0`,
the red goal sphere and the only thing the goal reaches, is a site, so it is re-expressed as a geom.
Leaving the sites in place is worse than inconsequential, in two separate ways: the env moves the goal
by writing `model.site_pos`, so a site left at its model position draws a **second** red sphere at
(0.8, 0.75, 0.5); and `object0`'s site — a red sphere that lives *inside* the block and is invisible
in the official frame — becomes visible the moment the block is hidden for the plate bake, which made
the "static" plate depend on `qpos[15:17]` over 1123 pixels.

**`mujoco_warp`'s `kinematics` does not recompute `geom_xpos` for world-body geoms.** The goal sphere
was first added to the worldbody with its position driven through the (per-world) `geom_pos` field. It
never moved: it stayed at the origin, out of frame. The symptom was three dead state dimensions
(`goal_x/y/z`) **and** an inert `goal.color` at the same time, which is what "the primitive is not in
the picture at all" looks like from two different checks. The sphere now lives on a **mocap body**,
whose pose is `data.mocap_pos` — per-world by construction, and applied by kinematics.

**Reset states do not exercise this family.** `_reset_sim` restores `initial_state` and randomises only
the object, so every reset frame carries the identical arm pose, and a parity number computed from
them says nothing about the 11 joints that are most of the picture. `sample_states` draws from a pool
built by resetting and applying 0–25 random actions. Two details of that pool are load-bearing and
both are recorded in the code: `action_space.seed(0)`, because `action_space.sample()` draws from the
*space's* generator which `reset(seed=)` does not touch (without it the pool differs between processes
and `test_seeded_sampling_reproduces_in_a_fresh_process` fails); and the block's yaw is randomised,
because `_reset_sim` leaves its quaternion at the identity and `check_state_dims_matter` perturbs
`object_quat_w` by +25, −25, 1.0 and 0.5, **every one of which normalises straight back to the
identity** — the same trap the OGBench port records for a free-joint torso.

---

## 4. MJWarp truncates where OpenGL rounds — worth more than everything else combined

The largest single fidelity term in this family is not shading, geometry or textures. It is one line.

`pack_rgba_to_uint32` is `int(a)<<24 | int(r)<<16 | int(g)<<8 | int(b)` on values already scaled to
[0,255], and C's `int` truncates; the GL pipeline MuJoCo renders through converts float to uint8 by
rounding to nearest. Fetch is the ideal scene to see this on, because it has no textures, no shadows
and no reflections, so nothing else can be responsible.

The signed difference on flat interior pixels, MJWarp against MuJoCo on the *identical* model:

| difference | −3 | −2 | **−1** | **0** | +1 | +2 | +3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| share of interior channels | 0.006% | 0.012% | **13.11%** | **86.83%** | 0.003% | 0.001% | 0.008% |

Never above, one level below on an eighth of the frame, and nothing else above 0.01%. That is one
float quantised two ways and it is not what any shading hypothesis predicts. Correcting the rule:

| Fetch, 480 px, 16 states, 4 scenes | truncating | rounding |
|---|---:|---:|
| interior pixels bit-exact | 0.791 | **0.9988** |
| all pixels bit-exact | 0.718 | **0.910** |
| mean \|diff\| | 1.24 | 1.05 |

`stable_envs.render.mjwarp.gl_rounding()` is the patch, written as a context manager because the
patched function has to be visible while warp *generates* the megakernel (context creation) **and**
while it *compiles* it (first launch), and for no longer than that. Any env in this repo that uses
MJWarp wants it.

### Where the residual is, after that

Per geom class, over 32 states, using MuJoCo's own segmentation buffer on the reference model
(`scratch/fetch/attribute.py`, PickAndPlace shown; the other three agree to a few thousandths):

| class | share of frame | all-pixel exact | share of INTERIOR pixels | interior exact | share of interior error |
|---|---:|---:|---:|---:|---:|
| background (skybox) | 71.5% | 0.9951 | 79.3% | **1.000000** | 0.000 |
| floor | 8.8% | 0.9955 | 9.5% | **0.999998** | 0.000 |
| table | 9.6% | 0.9951 | 10.2% | **1.000000** | 0.000 |
| robot meshes | 8.8% | 0.153 | 1.0% | 0.247 | **0.983** |
| gripper boxes | 0.08% | 0.138 | 0.004% | 0.183 | 0.001 |
| object | 0.08% | 0.472 | 0.03% | 0.472 | 0.016 |
| mocap markers | 1.2% | 0.000 | **0.000%** | — | 0.000 |

The plate is exact, and **98.3% of the interior residual is the robot's 18 meshes**: MuJoCo draws them
as triangles with Gouraud interpolation of per-vertex colours and this renderer evaluates
Blinn-Phong per pixel. The two land within a level of each other — interior mean |diff| is
**0.009/255** — and not on each other. That is the same limit the OGBench port documents and refuted
three ways there (Gouraud over MuJoCo's own tessellation scored *worse* than plain per-pixel), and it
bites here only because a Fetch mover is 9% of the frame rather than 20 pixels.

---

## 5. What is not drawn, and why the state is `qpos + goal`

`robot0:mocap` carries four **translucent** marker geoms: a 1 cm cube at `rgba="0 0.5 0 0.7"` and
three 2 m crosshair bars at alpha 0.1. `mujoco_warp`'s shader reads `color[0..2]` and packs a constant
255 — it has no alpha path at all — so it would paint them opaque saturated green. They are excluded.

Measured against a reference with the same geoms hidden:

| | value |
|---|---|
| share of all pixels touched | 1.2–1.5% |
| mean over the whole frame | 0.32/255 |
| share of INTERIOR pixels touched | **0.000002** |

A translucent bar over a flat surface leaves no flat 3×3 neighbourhood, so essentially none of it
reaches the interior metric — re-scoring against the marker-free reference moves interior agreement by
0.0007, and *the wrong way*.

**This is why `mocap_pos`/`mocap_quat` are not in the state.** They reach no other pixel. Carrying them
would declare seven dimensions the renderer provably ignores, and `check_state_dims_matter` would
correctly fail. `reference_render` closes the loop by putting the reference's mocap where the env's own
`reset_mocap2body_xpos` puts it — at the welded gripper body — so the official frame is a function of
`(qpos, goal)` and `test_state_is_sufficient` is a real check rather than a vacuous one.

This **refines rather than contradicts** `swm_coverage.md`'s G1 result. That probe restored
`qpos + qvel + mocap + goal` and got 0/255, which is true and is a statement about the *reference*. The
render-relevant state is smaller: **18 floats (Reach) or 25 (the other three)**, against 480·480·3 =
691,200 bytes of frame — 1/9,600 and 1/6,900. `qvel` is measured irrelevant here too (scaling it by
100 and adding 1 changes 0 pixels), and the seven mocap floats buy 1.4% of pixels that no alpha-blind
renderer can draw.

---

## 6. Variation space

Defaults are the official appearance, and colours are float `Box`es in [0,1] rather than `RGBBox` for
the reason the OGBench port gives: MuJoCo's material colours are floats (0.93, 0.93, 0.93) and a uint8
channel cannot hold one.

| leaf | reaches | plate? |
|---|---|---|
| `floor.color`, `table.color` | the baked static scene | yes |
| `light.diffuse` | everything | yes |
| `camera.distance_scale`, `camera.azimuth_delta`, `camera.elevation_delta` | everything | yes |
| `goal.color` | the goal sphere | no |
| `arm.color` | the five links wearing the MJCF's `robot0:blue` | no |
| `object.color` | the block/puck (absent from `Reach`, which has no object) | no |

Only the plate leaves group the batch, and only the plate is re-baked per distinct value; the MJWarp
pass is **one launch for the whole batch** regardless, because colours, light and camera are per-world
model fields. So the common case — randomise the goal or the arm, keep the room — never splits
anything.

**Two knobs are deliberately absent.** The **skybox** is 72% of the frame and would be the most useful
of all, but it is a texture and `mujoco_warp` uploads textures when the render context is built, so a
per-value skybox means a render context per value — exactly the "do not scale context creation with
anything" rule this repo enforces. The **headlight** is compiled into the megakernel as a static, for
the same reason.

---

## 7. Deployment: this family does not register on the shared interpreter, on purpose

`mujoco_warp` requires `mujoco >= 3.11`; the environment this repo's suite runs in carries 3.9 and has
neither `mujoco_warp` nor its own dependency `warp`, and it cannot get them without moving all 25
OGBench parity numbers onto a different MuJoCo. So the only question is whether this family fails
loudly at import or quietly at the first frame, and the second is not hypothetical: a lazy import
inside `render` let all 12 classes register on that interpreter and then produced **300 failures**
reading `ModuleNotFoundError: No module named 'warp'`, which looks like a broken renderer rather than
a missing wheel.

`stable_envs/envs/fetch.py` therefore raises `ImportError` at module scope, which is the mechanism the
package already has for an optional dependency and the one `dmcontrol.py` uses for its `dm_control`
pins. **All three conditions are checked**, because any one alone gives the wrong answer:
`mujoco_warp` present without `warp` is the failure actually observed, and both present against
mujoco 3.9 fails later still, inside `put_model`. They are `find_spec` lookups and a version read
rather than imports, so nothing pulls `warp` (~2 s) into every `import stable_envs`.

Verified both ways: on the shared conda env the registry holds 65 envs, no `Fetch*`, and
`stable_envs.envs.import_errors()['fetch']` names the missing modules, the MuJoCo version and the
venv; in `~/venvs/se-fetch` the registry holds all 12 and `import_errors()['fetch']` is `None`.

```
venv:   /mnt/vast/home/randall.balestriero/venvs/se-fetch
        (mujoco 3.11.0 + mujoco-warp 3.11.0 + warp-lang 1.16.0 layered over the shared conda env,
         which keeps its own mujoco 3.9.0 — the venv is --system-site-packages, so gymnasium,
         gymnasium-robotics, torch, triton and scipy come from there and only mujoco is shadowed)
logs:   /mnt/vast/home/randall.balestriero/stable_envs_logs/fetch/
```

| script | what it settles |
|---|---|
| `probe1_model.py` | model census, G3 across resets, alpha and texture census, marker and site impact |
| `probe2_mjwarp.py` | the diagnostic ladder: camera+materials, then markers, then MJWarp |
| `probe3_split.py` | error split by geom class; the signed-difference histogram that found the rounding rule |
| `probe4_plate.py` | the rounding fix and the plate composite, measured separately and together |
| `probe5_ss.py` | supersampling variants, throughput, plate invariance |
| `probe6_plate_invariance.py` | why the plate was not constant (it was `object0`'s site) |
| `calibrate.py` | the numbers in §1, through the env as registered |
| `attribute.py` | the interior residual by geom class (§4) |
| `fps_official.py` | the official renderer's warm render-only throughput |

```bash
srun --partition=open_research --account=open_research --qos=normal --gres=gpu:1 \
     --cpus-per-task=8 --mem=64G --time=01:00:00 \
     env PYTHONPATH=. MUJOCO_GL=egl \
     /mnt/vast/home/randall.balestriero/venvs/se-fetch/bin/python -u scratch/fetch/calibrate.py
```

## 8. Three ways these numbers could have been wrong, checked

Raised by the MuJoCo-locomotion work and answered by measurement rather than by argument
(`scratch/fetch/probe7_interference.py`).

**A second GL context in the same process did not corrupt the reference: 0/255 over the whole batch,
0.0% of pixels.** `OffScreenViewer.render` does not bind its context, so an interloper with a
*different* multisample config silently rewrites a live env's pixels. This env builds
`mujoco.Renderer` objects for the plate bake in the same process as the reference, so it is exactly
the exposed case — and it is safe here for a reason rather than by luck: `_build_bundle` copies
`vis.quality.offsamples` from the official model, so both contexts are 4-sample and the interloper
changes nothing. The measurement is the frames produced in a process holding our renderers, diffed
against frames produced in a subprocess where no `stable_envs` renderer exists at all.

**The free camera is a constant of the family, not of construction history.** Gymnasium freezes the
free camera at the first render and its own default lookat is `median(geom_xpos)`, which would make it
a function of whatever state the viewer was built in. `MujocoFetchEnv` overrides all four parameters
from `DEFAULT_CAMERA_CONFIG`, and the live viewer camera reads `(FREE, distance 2.5, azimuth 132,
elevation −14, lookat (1.3, 0.75, 0.55))` identically after three reset-plus-five-steps-plus-render
cycles and identically again in a clean process. `FetchBase._check_camera` asserts it on the first
`reference_render` call, so an upstream change would present as a moved camera rather than as a
diffuse parity regression.

**Plate versus MJWarp is not either/or here, and this env is not the MJWarp-only design the question
assumes.** Three other families in this repo measured `mujoco_warp` and rejected it; this one uses the
same baked MuJoCo plate they do for **91% of the frame**, and reaches for MJWarp only for the movers.
Three measured facts separate this case from theirs:

* **What the movers are.** DM Control's floor is a checkerboard under a spotlight and its sky is a
  speckled gradient, so 92–99% of its frame is textured and MJWarp's texture path is the whole
  problem. Fetch has **no textured material at all** (§1), and its movers are 18 STL meshes forming
  the arm, every joint of which is in `qpos`. `stable_envs.render.raycast3d` has no mesh path, so
  "put MJWarp behind the analytic renderer" has no analytic renderer to go behind — it would mean
  writing a mesh ray-caster, which is a new backend rather than a fallback.
* **What such a renderer could reach.** Its ceiling on robot pixels is the same one measured in §4:
  MuJoCo Gouraud-interpolates per-vertex colours over the triangles and any per-pixel shader lands
  near it, not on it (the OGBench port refuted three variants of the reverse). MJWarp already scores
  0.247 exact there; an analytic mesh caster would be solving the easy half of the problem and
  inheriting the hard half.
* **The 20–53%-exact figure predates the fix.** It was measured on textured scenes and before the
  quantisation rule in §4, which is worth 0.791 → 0.9988 interior on its own. This family measures
  **0.908–0.928 all-pixel exact**, i.e. above the suite's 90% gate rather than below it, and that is
  the number the gate actually reads.

The convergent advice held in the part that matters: bake the plate, and use a ray-caster only for
what moves. Where this family differs is that what moves is a mesh.

## 9. Reproducing

One process-level warning worth carrying: **an EGL context released is every other context corrupted**,
and this family reproduced it. A probe that rebound one `mujoco.Renderer` per scene in a loop dropped
the previous one, and every *later* official render came back as a single flat colour — mean 0.0, one
distinct colour, which reads exactly like "the reference is broken" and is not. Every renderer here is
cached and never closed.
