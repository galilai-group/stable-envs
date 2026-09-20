# Step 7 / W3 `dmc-manipulator` — phase 0 findings

**Target:** `swm/ManipulatorDMControl-v0` -> `dm_control/manipulator-bring_ball`, camera `fixed` (id 0).

**Classification: `PATCH_CANDIDATE`** — the plate + mover design works on this domain and validates at
interior parity 1.000000 with 0 bad interior pixels out of 4,154,763, so this is a small patch to
`stable_envs/envs/dmcontrol.py` and not a new renderer. It carries **one honest blocker to a clean
Gate A pass** (three state dimensions that rotate a SPHERE reach the official frame and cannot reach
ours), which is a declaration decision for the coordinator, not a bug to fix.

- Baseline SHA: `76dc5116f432e61dd47fba8f9e879cf9212c8e46`
- Branch: `step7/dmc-manipulator`, worktree `/Users/samib/stable-envs-step7-dmc-manipulator`
- Measured on: Apple M5 Mac, `/Users/samib/se6-venv`, **mujoco 3.12.0 + dm_control 1.0.45**,
  `MUJOCO_GL=glfw`, `device="cpu"`. The declared DMC reference pair is **mujoco 3.11.0 +
  dm_control 1.0.44**; every number below must be re-measured on Oscar's `se-vis311` before it is
  certified. Oscar is **DENIED at SSH auth** right now (`Permission denied (keyboard-interactive)`),
  so all cluster items are PENDING.
- Nothing under `stable_envs/` was edited. Nothing under `examples/results/step4|step5|step6` was
  touched.
- Probes: `scratch/step7/dmc-manipulator/probe_{census,prototype,residual,gate,deaddims,reflect}.py`
- Artifacts: `examples/results/step7/dmc-manipulator/phase0/{census_224,prototype_224_ss1,
  prototype_84_ss1,residual_224,gate_224,dead_dims_224,reflect_224}.json`

---

## 1. Which SWM default the registered id uses

Read from the vendored source at `/Users/samib/stable-worldmodel/stable_worldmodel/envs/dmcontrol/`
(`stable_worldmodel` is **not installed** in the local venv; this is source reading plus the
`scratch/swm_matrix.json` probe row, and it must be confirmed live on Oscar).

| question | answer | evidence |
|---|---|---|
| task | `Bring(use_peg=False, insert=False, fully_observable=True)` = `manipulator-bring_ball` | `manipulator.py`; matrix `resolved_reference.how = "exact"` |
| camera id | **0** (`fixed`) | `DMControlWrapper.__init__`: `self.camera_id = 2 if domain == 'quadruped' else 0`; `render()` is `camera_id or self.camera_id`, and `0 or 0 == 0` |
| size | **224 x 224** | `DMControlWrapper.render(width=224, height=224)`; matrix `frame_shape = [224,224,3]` |
| action_repeat | 2 (physics only, no pixel effect) | matrix `observed_config` |

**Must be verified on Oscar, because it can move a pixel.** SWM does **not** call
`suite.load("manipulator", "bring_ball")`. It calls `manipulator.make_model(use_peg=False,
insert=False)`, rewrites the asset paths, round-trips the XML through `dm_control.mjcf`
(`mjcf.from_xml_string` -> `mjcf.export_with_assets` -> `Physics.from_xml_path`), and then builds
`Bring` itself. The mjcf round-trip can rename or reorder elements and can re-emit defaults, and
`_episode_slots` / `_movers` / `_prims` all key on model ORDER and on `mj_id2name`.
**Open question O1:** compile both models on Oscar and diff the render-model signature (geoms, sites,
materials, camera, lights, after `reset()`), as Gate A §4.2 requires.

**And SWM's default appearance is not dm_control's.** `modify_mjcf_model` runs on *every* reset, and
with the shipped `init_value`s it writes:

| SWM leaf | init | effect on the frame |
|---|---|---|
| `rendering.render_target` | **0** | sets `material target` alpha to **0** — the registered default frame **does not draw the target ball at all** |
| `target.shape` | 1 (sphere) | `geom target_ball` stays `type=sphere` but at `size=.015` (dm_control's own is `sphere size=.02`) |
| `target.color` | `.6 .3 .3` | same as dm_control |
| `agent.color` | `.7 .5 .3` | same as dm_control (`material self`) |
| `floor.color` | `[[.1 .2 .3],[.2 .3 .4]]` | same values as dm_control, but written into the **texture** `grid`'s `rgb1`/`rgb2` |
| `light.intensity` | 0.7 | dm_control's `light` has `diffuse=.6` — **SWM's default is a different light** |
| `agent.upper_arm_density`, `agent.hand_density` | 1000 | mass only, no pixel |
| `agent.upper_arm_length` | 1000 | **dead leaf** — `modify_mjcf_model` never reads it (`# TODO should modify length of arm`) |

So the SWM-registered default differs from `suite.load`'s in at least three ways that move pixels
(target hidden, target size `.015` vs `.02`, light diffuse `.7` vs `.6`). Gate A §4.2 says the
configuration certified must be the registered SWM default. **Open question O2:** does this cell
certify against `suite.load` (the whole family's current practice, and what every DMC parity number
in the repo is measured against) or against SWM's modified model? The two are not the same scene.

---

## 2. State sufficiency — PASSES, the Step-1 way, at the registered size

224 px, camera 0, states from a real rollout (`probe_census.py`):

| adversarial condition | disagreeing pixels |
|---|---|
| render the same `qpos` twice | **0 / 255, 0 px** |
| `qvel x 100` | **0 / 255, 0 px** |
| the same `qpos` reached through a different history | **0 / 255, 0 px** |
| every `qpos` entry moves pixels | **11 / 11**, 15-1424 px each |

`mujoco.Renderer` at the default `MjvOption` equals `physics.render` at **0/255**, so the reference
bridge is the one line the family already uses.

**`visualize_reward` is OFF by default** and no reward-coloured material is live: `material self ==
self_default` and `target == target_default` after six real episodes; the three `*_highlight`
materials exist in the asset table and are worn by nothing.

### The per-episode draw, precisely

Over `N_DRAWS = 6` real episodes, diffing all ten arrays in `EPISODE_ARRAYS`, exactly four floats move
and they are all one body's pose:

```
body_pos[33]  target_ball_pos_x     span 0.661
body_pos[35]  target_ball_pos_z     span 0.248
body_quat[44] target_ball_quat_qw   span 0.619
body_quat[46] target_ball_quat_qy   span 1.385
```

which is exactly `Bring.initialize_episode`:
`model.body_pos['target_ball', ['x','z']] = target_x, target_z` and
`model.body_quat['target_ball', ['qw','qy']] = cos(a/2), sin(a/2)`.

So the state is **`nq` 11 + 4 = 15**, named
`arm_root ... ball_y, target_ball_pos_x, target_ball_pos_z, target_ball_quat_qw, target_ball_quat_qy`.

**The plate is exact under that widening, measured both ways** (224 px):

| plate (MuJoCo, every mover hidden) | disagreement |
|---|---|
| within one episode, 4 poses | **0 / 255** |
| across 4 episodes | **0 / 255** |
| plate vs. the full frame | 1.70-1.94 % of pixels, max 198-204, mean 1.12-1.32 / 255 |

(1.7-1.9 % is the cartpole band — the family ships at 0.15-3.0 %.) Note the target does **not** need
the `reacher`/`finger` special case: `target_ball` is its own BODY, so the existing
"anything off body 0 is a mover" rule already catches it; only the STATE widening is new.

---

## 3. Smallest missing capability against `DMControlSuite` as it exists

Census (mine, 224 px, confirms the coordinator's): ngeom 21 (all drawn — 4 textured planes `floor`,
`wall1`, `wall2`, `background`; 1 cylinder `arm_root`; 14 arm capsules/spheres; `ball`;
`target_ball`), nsite 9 (**0 drawn** — groups 3/4), nmesh 0, nhfield 0, ntex 3, ntendon 2 (both
`<fixed>` -> `_find_tendon` returns `None`, no catenary), 1 model light, 2 cameras, `shadowsize 2048`,
`shadowclip .5`, `offsamples 4`.

| candidate gap | verdict | evidence |
|---|---|---|
| **DIRECTIONAL `castshadow=1` light** | **already covered.** Not a gap. | `_Lights` reads `mjvScene` and gets `kind = [1, 1]`, `castshadow = [0, 1]` — the headlight and the model light are both directional and only the model light casts. `_bake`'s second plate (diffuse+specular zeroed on light index `sl - headlight_active = 0`) fires and **differs from plate 1**. `_occluded` and `_static_shadow_rays` both branch on `LIGHT_DIRECTIONAL` explicitly and issue a `lim = INF` shadow ray. The Step-6 note that a shadow-casting directional light is refused is about **`stable_envs/render/cpu3d.py:347`**, the MJWarp-parity CPU backend for the TrackCom/locomotion cells; `DMControlSuite` never calls it. |
| **fovy 4.0 camera** | **already covered.** | `R3.Camera.from_mjv` takes the frustum off `mjvScene.camera[0..1]`; nothing in the family assumes a fovy. Parity at this camera is the best in the family (§4). |
| **textured walls / background planes** | **already covered** — they are STATIC and MuJoCo bakes them. | `statics = [floor, wall1, wall2, background, arm_root]`, and the plate is 99.9988-99.9992 % bit-exact on the pixels it owns. |
| **per-episode `target_ball` `body_pos`** | **already covered** by `_episode_slots` + `_fk`. | `body_pos` is not in the guard list; `_fk` writes the slot and runs `mj_kinematics`. |
| **per-episode `target_ball` `body_quat`** | **THE ONE HARD GAP.** `_episode_slots` **raises** on any slot ending `_quat`. | `dmcontrol.py` ~line 445: `bad = [s for s in slots if s[0].endswith(("_size", "_quat", "_dir"))]` -> `RuntimeError`. So `DMControlSuite` refuses this domain at construction today. |
| **three reflective planes, `_bake` keeps one** | a real shared-code limitation, **immaterial here**. | `floor`, `wall1`, `wall2` all wear `material grid` (`reflectance=.2`) and `_bake` `break`s on the first. Measured: MuJoCo's whole reflection signal at this camera is **11 pixels over 16 frames** (1.4e-5 of the frame, max 20/255) — at fovy 4 from 16 m the floor and both walls are seen almost edge-on. Our reflection pass contributes 5 px and parity is identical with it on or off (mean 0.25491 vs 0.25499). Record it; do not fix it for this cell. |

### The `_quat` guard is over-broad, and the argument that justifies it does not extend to orientation

The guard's own docstring gives the reason: *"The mover primitives carry one size and one orientation
for the whole batch"*. Half of that is true and half is not, in the shipped code:

* **size** really is shared — `_prims` reads `geom_size` once at bake time, `screen_span` derives its
  bounding radius from `sc.prims.size`, and a per-row size would have to thread through every
  intersector. The `_size` half of the guard is correct.
* **orientation is already per row.** `_fk` writes the drawn slot into the model, runs
  `mj_kinematics`, and reads `d.geom_xmat` into `xm`, a `(B, P, 3, 3)` tensor. `_cast` ->
  `R3.cast_nearest(kind, sizes, xpos, xmat, ...)`, `_occluded` -> `R3.any_blocking(..., xmat, ...)`
  and `_reflect` all consume `xmat` per row; `screen_span` never reads it. So a per-episode `*_quat`
  needs **no renderer change at all**.

Tested, not argued: removing only the `_quat` term from the guard in a scratch process (the patch in
`probe_prototype.patched_episode_slots`, which changes nothing else) produces a cell that renders,
and it renders at parity — §4.

---

## 4. The design-validating measurement

`DMControlSuite` subclassed, `DOMAIN/TASK/CAMERA = manipulator/bring_ball/0`, the shipped three-plate
bake, the shipped mover pass through `raycast3d`'s public API, the shipped composite, `ss = 1`,
`device="cpu"`, against `physics.render` on states from the domain's own visited distribution.

**224 px (the registered size), 3 draws x 64 states = 192 frames:**

| metric | min | median | max |
|---|---|---|---|
| **`interior_exact_frac`** | **1.000000** | **1.000000** | **1.000000** |
| `interior_frac` | 0.4347 | 0.4353 | 0.4354 |
| interior bad px | **0 of 4,154,763** | | |
| `exact_frac` (all pixels) | 0.98289 | 0.98304 | 0.98306 |
| `mean_abs` | 0.2603 | 0.2616 | 0.2671 / 255 |
| `max_abs` | 205 | 208 | 222 |
| SSIM | 0.99366 | 0.99408 | 0.99419 |
| plate-region `exact_frac` | 0.9999879 | 0.9999914 | 0.9999924 |
| share of disagreeing px inside the mover region | 0.99930 | 0.99950 | 0.99956 |

**84 px (the family's `default_size`), 2 draws x 64:** `exact_frac` 0.9779-0.9780,
`interior_exact_frac` **1.0** (`interior_frac` 0.0750), mean 0.528-0.547, SSIM 0.9870-0.9871.
Parity holds at both sizes.

Three things are worth reading carefully.

1. **The interior mask is the best-defined in the family.** `interior_frac = 0.435` at 224 px against
   0.000000 on acrobot/point_mass/both reachers and 0.111-0.120 on the four cells that have any. The
   `background` plane is a large flat region, so the strong assertion is not falling back to all-pixel
   agreement here — it is measured on 1.38 M pixels per draw and finds **zero** errors.
2. **The residual is 100 % silhouette, and that is measured, not inferred.** Of the mover-region
   pixels (reference != MuJoCo's own movers-hidden plate), **`mover_region_edge_frac = 1.000` — ZERO
   of them survive the interior mask.** At fovy 4 the arm capsules are 4-8 px ribbons and every mover
   pixel is a silhouette pixel, where MuJoCo's `offsamples=4` MSAA and a one-ray-per-pixel analytic
   cast are entitled to disagree. That is why `mover_region_exact_frac` reads a startling 0.085-0.089
   while the frame is 98.3 % exact and the interior claim is unconditional.
3. **`ss = 1` is the right default here too**, for the reason the family documents: at ss = 2
   `exact_frac` falls 0.9828 -> 0.9808 and the plate region falls 0.999995 -> 0.998493, because the
   box filter mixes mover colour into pixels the segmentation calls plate.

`check_resolution_consistency` at 224/factor 2 gives `exact_frac 0.848`, mean 1.18 — **better than
every shipped DMC cell** (DMCCartpoleSwingup 0.317, DMCBallInCupCatch 0.424, DMCReacherHard 0.791).

### Indicative CPU throughput (local Mac — NOT a certification number)

224 px, `device="cpu"`, `ss=1`: B=1 47 f/s, B=64 **90 f/s**, B=256 26 f/s. Certification numbers must
come from Oscar `--partition=batch --qos=normal --cpus-per-task=4` with threads pinned, and the GPU
arm from a 3090 under `--partition=gpu --qos=norm-gpu`. PENDING.

---

## 5. Gate A mandatory checks, run locally on the prototype

`probe_gate.py`, 224 px, `device="cpu"`:

| check | result |
|---|---|
| `check_state_roundtrip` | pass (0.0) |
| `check_batch_invariance` | pass (0) |
| `check_batch_size_invariance` (1,3,8) | pass (0) |
| `check_order_independence` | pass (0, 0) |
| `check_determinism` | pass |
| `check_render_is_pure` | pass (all 0) |
| `check_chunked_render_matches` (1,3,8) | pass (0) |
| `check_input_form_invariance` | pass (no disagreements) |
| `check_set_state_copies` | pass |
| `check_style_is_state` | pass (0) |
| `check_default_style_is_official` | pass (0) |
| `check_per_env_variation` | pass (no collapsed variations) |
| `check_variations_do_something` | pass (no inert variations) |
| `check_colors_apply_exactly` | pass (none unapplied) |
| `check_resolution_consistency` | 0.848 exact — best in family |
| **`check_state_dims_matter`** | **FAIL — `dead_dims = ["ball_y", "target_ball_quat_qw", "target_ball_quat_qy"]`** |

### The blocker, and its mechanism, measured on both sides

All three "dead" dimensions do exactly one thing: **they orient a SPHERE.** `ball` is
`type=sphere size=.022`, `target_ball` is `type=sphere size=.02`, and neither wears a texture
(`material self` / `material target`, no `texid`). Measured over 16 states at 224 px
(`probe_deaddims.py`), perturbing each dimension and diffing each renderer against its own baseline:

| dimension | pixels moved in **our** frame | pixels moved in the **official** frame | official max |
|---|---|---|---|
| every other dim (12 of 15) | 119 - 17,632 | 242 - 20,208 | 60 - 219 |
| `ball_y` (+0.15 rad) | **0** | 896 | 55 / 255 |
| `target_ball_quat_qw` (+0.4 rad about y) | **0** | 734 | 33 / 255 |
| `target_ball_quat_qy` (same rotation) | **0** | 734 | 33 / 255 |

**The mechanism.** MuJoCo draws a sphere as a TESSELLATED mesh shaded per vertex, so rotating the
sphere rotates the mesh and moves the Gouraud interpolation and the specular vertex; an analytic
per-pixel sphere is exactly rotationally invariant and cannot move. This is the same disagreement the
family already declares as *"a one-level rim on curved silhouettes — MuJoCo shades a tessellated
sphere or capsule per vertex; an analytic per-pixel evaluation is within a few levels of that, not
equal to it"* — except that here it is the ONLY thing three state dimensions do, so it surfaces as a
dead dimension rather than as a rim.

Two facts bound it. (a) Parity is **unaffected**: `exact_frac` after each of the three perturbations
is 0.98275-0.98277 against a 0.98280 baseline, i.e. the moved pixels are already inside the
silhouette band that disagrees anyway. (b) It is not fixable within this design — reproducing it
needs MuJoCo's icosphere tessellation and per-vertex shading in the mover pass, which would replace
the analytic caster for **every** DM Control cell to recover ~800 px at <= 55/255 on one of them.
Recommended: **do not implement it.**

---

## 6. Fidelity tier and the disposition of the blocker

**Recommended: `fidelity_tier = "parity"`, `parity_interior_min = 1.0` not relaxed, Gate A `partial`**
with a declared `state_dims_note` naming the three dimensions and the tessellated-sphere mechanism.

Justification against `docs/research/certification.md`:

* §4.3 `parity` asks for `interior_exact_frac >= parity_interior_min`. Measured **1.000000, 0 bad of
  4,154,763**, on the largest interior mask in the family. Nothing is relaxed. `parity_all_exact_min`
  is **not** needed either: 0.9829 worst at 224 px and 0.9779 worst at 84 px both clear the shared
  0.90 bar by a wide margin (cf. `DMCFingerTurnEasy`, which declares 0.89 at 0.9103 worst).
* §4.1 makes `check_state_dims_matter` mandatory with **no declared-exclusion escape**, and it fails.
* §4.2/§4.3: `partial` is *"a passing state for this gate and it caps the ladder"* — a `partial` env
  certifies per backend at any performance tier and can never be `ENVIRONMENT_DONE`. That is exactly
  the shape of this cell: the renderer covers a **documented proper subset** of the reference's
  response to the state (everything except sphere self-rotation), and the exclusion is measured.
  `swm/CraftaxClassicPixels-v1` is the existing precedent for `partial`.

**Open question O3 for the coordinator:** `partial` versus a third option — widen
`check_state_dims_matter` so a dimension is "alive" if it moves the REFERENCE frame and the env
declares it as reference-only. That would be a change to a shared mandatory check on behalf of one
cell and I have not made it.

**Shadows of the moving arm, exactly how the existing DMC cells handle them** — asked in the brief,
so stated precisely. There is no shadow *map*. Three mechanisms compose:

1. `_bake` takes a SECOND plate with the shadow-casting light's `diffuse` and `specular` set to 0.
   That is, by construction, the colour a static surface has where that light does not reach it.
2. `_shadow_candidates` restricts the per-frame shadow test to the samples where plate 1 and plate 2
   actually differ AND the static ray hit something — outside that set the test cannot change a pixel.
3. Per frame, `_static_shadow_rays` + `_occluded` fire ONE analytic occlusion ray per candidate sample
   from `o + t_static*d` toward the light (for a DIRECTIONAL light: direction `-normalize(light.dir)`,
   `lim = INF`), and a sample that is blocked selects plate 2 instead of plate 1.

So a shadowed pixel is as exact as the plate; only the shadow's *shape* is the ray-caster's, and its
residual is a one-pixel band where a 2048-tap shadow map meets one analytic ray. On manipulator that
band is inside the silhouette residual already counted (the interior mask finds 0 errors), and
`plate_shadow_differs = True` confirms the second bake is live. Mover-on-mover shadows use the same
`_occluded` call with `skip = i_w` so a convex mover cannot shadow its own front face.

---

## 7. Smallest implementation plan

### Files

1. **`stable_envs/envs/dmcontrol.py`** — three edits, all small.
   * `_episode_slots`: narrow the guard from `("_size", "_quat", "_dir")` to `("_size", "_dir")`, and
     replace the docstring paragraph with the measured reason above (orientation is already per row
     through `mj_kinematics` -> `d.geom_xmat` -> `(B,P,3,3)`; only SIZE is shared at bake time). This
     is a **shared-file edit inside this workstream's own family**, but W4 `dmc-quadruped` also lands
     in `dmcontrol.py`. **Coordinator should assign ownership of `dmcontrol.py` between W3 and W4.**
   * `_hideable()`: today it iterates `_slots()` and `continue`s on any slot whose element kind is not
     `geom`/`site`. Manipulator's draw is a **BODY**, so the cell gets **no `visibility.*` leaf at
     all** (confirmed: its variation space is exactly `camera.fovy_scale`, `light.diffuse_scale`,
     `material.{background,decoration,effector,grid,self,target}`). Extend it to map a drawn body to
     the drawn geoms hanging off it, named by the body — `visibility.target_ball`. Needed for Gate F
     (SWM's `rendering.render_target`), and the existing `_apply_variation` visibility path then works
     unchanged, because it already zeroes the ELEMENT's own rgba alpha.
   * `DOMAINS`: add `"DMCManipulatorBringBall": ("manipulator", "bring_ball", 0)`, and extend the
     class docstring's exclusion list (manipulator/stacker are no longer "untested").
2. **`docs/research/dmcontrol.md` §8.2** — replace the "not re-probed since the first pass ... untested,
   not refuted" row with the measured result, and add the widened-state row for manipulator
   (`nq + 4`) to §2.1.
3. **`docs/research/swm_coverage.md`** — move `ManipulatorDMControl` out of the gap list.
4. **No change to `stable_envs/render/raycast3d.py`.** The design needs no new renderer capability.

### Tests

* `tests/unit/test_declarations.py`, `tests/unit/test_repo_contracts.py` — pick the new cell up from
  `REGISTRY` automatically; run both.
* `tests/gpu/test_conformance.py`, `test_reproducibility.py`, `test_fast_paths.py` — parameterised over
  `REGISTRY`; the new cell joins them. Includes `test_render_parity_with_official_renderer` (needs
  `parity_interior_min`), `test_frames_are_bit_identical_in_a_fresh_process`, and the
  `SHADOW_COMPACTION` / `COVER_COMPACTION` off-vs-on equality in `test_fast_paths.py`.
* **New unit test** (the thing that was never covered): a per-episode `*_quat` slot must reach the
  frame. Assert that substituting `target_ball_quat_{qw,qy}` from another sampled state changes the
  REFERENCE frame, and that `_fk` writes a per-row `xmat` that differs across rows. This is the test
  that would have caught the guard being wrong in either direction.
* `tests/unit/test_dmcontrol_tendon.py` — unaffected (`_find_tendon` returns `None` here).
* Regression: the other 15 DM Control cells must be **bit-identical** before and after, since
  `_episode_slots` and `_hideable` are shared. `finger-turn`'s `site_pos` slots and `swimmer`'s
  `light_pos` slot are the ones that exercise the changed lines.

### Replay / certification obligations

* **Step-4 replay:** `examples/fidelity_step4.py` over every affected frozen config — i.e. all 16 DM
  Control cells, because the two edited functions are family-shared. Expect byte-identical; write only
  under `examples/results/step7/dmc-manipulator/replay/`. Step-4/5/6 corpora stay untouched.
* **Gate A:** every mandatory check above, on Oscar, at the registered 224 px AND at the family's
  `default_size = 84`. Declared result: `partial`, with `state_dims_note`.
* **Declared tier:** `parity`; `parity_interior_min = 1.0` (not relaxed); no `parity_all_exact_min`.
  Re-measure the interior claim as a DRAW COUNT (the family's rule — >= 40 independent draws of 64
  states, not one large evaluation) on `se-vis311` at mujoco 3.11.0 / dm_control 1.0.44 before
  declaring.
* **Gate F — SWM render compatibility**, leaf by leaf:

  | SWM leaf | status | what is needed |
  |---|---|---|
  | `agent.color` | **covered** | `material.self` |
  | `target.color` | **covered** | `material.target` |
  | `light.intensity` | **covered in range** | SWM writes `light.diffuse` absolutely in [0,1]; ours is `light.diffuse_scale` in [0.2,1.8] on a base of .6 -> [.12, 1.0] after the clip. Map, and record that SWM's 0 is reachable only in the limit. |
  | `rendering.render_target` | **needs the `_hideable` fix** | -> `visibility.target_ball` |
  | `target.shape` (box / sphere) | **not covered; small** | a plate-path Discrete leaf writing `geom_type` + `geom_size` of `target_ball`. `_prims` already reads `geom_type` at bake time, so a re-bake carries it; nothing per-row is needed. |
  | `floor.color` | **not covered; the only real work** | SWM edits the `grid` **texture**'s `rgb1`/`rgb2`, not a material rgba. `_bundle` builds **one `mujoco.Renderer` per (domain, size) and never closes it** — the module's own comment says only a TEXTURE edit would need a second GL context. Options: (a) regenerate the builtin checker into `tex_data` in place and re-upload, (b) key the Renderer on the texture as well and accept N live contexts, (c) declare the leaf out of scope with the mechanism named. **Measure (a) first.** |
  | `agent.upper_arm_density`, `agent.hand_density` | n/a (physics) | no pixel effect; Gate F is a render gate |
  | `agent.upper_arm_length` | **dead in SWM itself** | `modify_mjcf_model` never reads it (`# TODO`). Record; do not implement a knob the reference does not honour. |

* **Frozen corpus** under `examples/results/step7/dmc-manipulator/`:
  `states/DMCManipulatorBringBall.npz` (the domain's own visited distribution, seeded),
  `certification.json` (the per-config contract), and `conformance/*.json` (digests + metrics),
  mirroring the layout of `examples/results/step4/`. Propose for the Step-8 freeze; do not freeze in
  Step 7.
* **Performance:** measure, do not gate (Step-7 rule). CPU on `batch/normal` with 4 cpus and threads
  pinned; GPU before/after on the SAME 3090 in one job. Job names `se6-dmc-manipulator-*` (or the
  Step-7 prefix the coordinator sets), logs to `$SCR/logs/%x_%j.out`.

### Not in scope for this workstream

* The three-reflective-planes limitation in `_bake` (measured at 11 px here; belongs to whoever needs it).
* MuJoCo's sphere tessellation in the mover pass (§6 — would replace the analytic caster family-wide).
* `manipulator-bring_peg` / `insert_*` and `stacker` — same design, not on the Step-7 target list.

---

## 8. Report summary

* **Classification:** `PATCH_CANDIDATE`.
* **Affected configs:** new `DMCManipulatorBringBall`; the 15 existing DM Control cells are touched
  only by two shared functions and must replay byte-identical.
* **Pixels changed on existing cells:** expected **no**; to be proven by the Step-4 replay.
* **Tiers:** new cell `parity` (interior 1.000000, 0/4,154,763), Gate A **`partial`** on
  `check_state_dims_matter`.
* **Semantic follow-up recommendation:** **none.** No existing env's pixels change, and the new cell
  is a parity cell. If the coordinator instead chooses to ship it with the three dimensions pruned
  from the state, that becomes a DINO tripwire question.
* **SLURM job IDs:** none — Oscar is DENIED at SSH auth (`Permission denied (keyboard-interactive)`).
  Every cluster obligation in §7 is PENDING, including the mujoco 3.11.0 / dm_control 1.0.44
  certification arm on `se-vis311` and the live `stable_worldmodel` check of O1/O2.
* **Open questions:** O1 (mjcf round-trip vs `suite.load` model signature), O2 (certify against SWM's
  modified default appearance or against `suite.load`), O3 (`partial` vs widening
  `check_state_dims_matter`), plus ownership of `stable_envs/envs/dmcontrol.py` between W3 and W4 and
  the `default_size` question (family 84 vs registered 224 — parity holds at both).

---

> **Phase 1 landed.** The plan above was implemented on this branch; see
> [`findings_phase1.md`](findings_phase1.md) for what changed and what it measured. Two items in §7
> above were settled by coordinator ruling rather than by me: O1/O2 (certify against `suite.load`,
> reach the SWM default through leaves — the Reacher pattern) and O3 (no Gate A `partial`, no
> widening of `check_state_dims_matter`; the three sphere-orientation dims ship as a recorded FAIL
> with a validated-exclusion contract proposed for approval). `floor.color` stays a Gate-F gap with a
> proposed measurement. `stable_envs/render/raycast3d.py` was never touched — W4 owns it.
