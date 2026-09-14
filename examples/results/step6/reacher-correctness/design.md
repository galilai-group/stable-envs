# Step 6B — Reacher visibility + fidelity guard: design note

Written before any code was changed, from measurements taken on the baseline
(`89fa5ca8`) in this worktree. Every claim below has a probe behind it; where a number
appears, the command that produced it is named.

Local probing runs on the Mac with `MUJOCO_GL=glfw` (macOS has no EGL, and
`stable_envs/envs/dmcontrol.py` defaults `MUJOCO_GL=egl`), `device="cpu"`,
`/Users/samib/se6-venv/bin/python`. Certification numbers come from Oscar.

---

## 1. Where the Step-5 offscreen workaround lives, and what it does

Four places, none of them in `stable_envs/`:

| file | what it does |
|---|---|
| `examples/downstream_step5/reacher/swap_renderer.py` | `OFFSCREEN = (10.0, 10.0, 0.01)`; `TARGET_MODES = ("offscreen", "as_placed")`; `Renderer.from_live_env` passes `target_override={"target": OFFSCREEN}` and `Renderer.from_state` writes `OFFSCREEN[comp]` into the widened state columns |
| `examples/downstream_step5/reacher/state_bridge.py` | `read_state(..., target_override=...)` substitutes those numbers for the values read off the official model |
| `examples/downstream_step5/reacher/parity_probe.py` | `OFFSCREEN = (10.0, 10.0)`, the same trick in the pre-flight probe |
| `tests/unit/test_step5.py::test_the_offscreen_placement_is_outside_any_reacher_scene` | asserts the constant is outside any reacher scene |

Mechanism: `DMCReacher*`'s state is `qpos` plus the two floats the episode DRAWS into
`geom_pos['target']` (`_episode_slots`). The harness overwrites those two state columns
with 10 m, so the target geom is still built, still transformed, still cast against —
it just lands outside the camera frustum. Visibility is expressed as a *position*.

Why it was done that way is recorded in `docs/research/step5_downstream_semantics.md`
issue 1: stable-envs' DM Control appearance space carries material colour, light diffuse
and camera fovy and **no alpha**, so "do not paint this geom" had no representation.

What it costs: the state no longer means what it says (a stored corpus row carries a
target position that is not the episode's), the geometry is still paid for in `_fk` and
in every ray pass, and nothing stops a state whose *real* target happens to be far away
from being silently reinterpreted.

## 2. How `stable_worldmodel`'s `render_target = 0` maps onto the official live renderer

`stable_worldmodel`'s reacher is not `suite.load('reacher', 'hard')`: it recompiles
through `mjcf` after editing materials and geom types, and its variation default
`rendering.render_target = 0` sets the **`target` material's alpha to zero**.
`mujoco.Renderer` at the default `MjvOption` then does not draw the geom at all. Step 5
measured that from the outside rather than reading it: moving the reference's target
changes **0/255** (`examples/results/step5/reacher/parity_probe.json`,
`reference_target_drawn: false`).

Two things follow, and they are different facts that have been conflated:

* **SWM's reacher default hides the target.**
* **stable-envs' `DMCReacher*` reference is `dm_control`, and dm_control's reacher DRAWS
  it.** Measured here: `mat_rgba['target'] = [0.6, 0.3, 0.3, 1.0]`, `geom_rgba['target']`
  is the compiler default, so `_effective_alpha` resolves to 1.0; hiding the geom in the
  reference changes **19 pixels, max 215/255** at 84 px
  (scratchpad `probe_hide.py`, `reacher-hard geom 'target'`).

So the stable-envs DEFAULT must keep drawing the target — that is what
`parity_interior_min = 1.0` and the frozen Step-4 digests
(`digest_ours == digest_ours_default_style`) are measured against. "Faithful to SWM"
here means **the knob must exist and must reproduce `render_target = 0` exactly when a
caller asks for it**, not that the stable-envs default flips. This note records that
reading explicitly because the brief's parenthetical can be read the other way; flipping
the default would break the Step-4 parity contract for `dm_control/reacher-{easy,hard}`.

## 3. How stable-envs represents per-geom appearance / style-as-state today

* Style is already **per-row**: `stable_envs/core/style.py::StyleTensor` is `(B, style_dim)`
  float32 with a path→(offset, width) layout, and `for_batch` is the single operation that
  may change its batch.
* `_DMCModel.make_variation_space` declares `material.<name>` (RGB `Box`, defaults read
  from the **pristine** compiled model), `light.diffuse_scale`, `camera.fovy_scale`.
  There is no alpha and no visibility leaf.
* `DMControlSuite.PLATE_PATHS` is *every* leaf, so any new leaf automatically keys the
  plate cache and `render()` already groups the batch by distinct style and bakes once
  per group. A per-row visibility knob therefore needs **no change to the render loop.**
* **Visibility already exists inside the renderer, just not as state.**
  `dmcontrol._effective_alpha(m, kind)` implements MuJoCo's own resolution rule
  (the element's own rgba unless it is the compiler default and a material supplies one);
  `DMControlSuite._visible` drops anything in a hidden group or with resolved alpha ≤ 0;
  `_movers` and `_bake` are built on top of it. Nothing downstream of `_apply_variation`
  needs to learn a new concept — it only has to be *reachable from style*.

## 4. Proposed representation

**`visibility.<element>` — one `Discrete(2)` leaf per element the EPISODE DRAWS, on the
parity tier only, applied by zeroing that element's own rgba alpha.**

* **Which elements.** Derived, not a name list: exactly the `geom`/`site` elements that
  `_episode_slots` found the task drawing per episode. Those are the scene's task-relevant
  landmarks by construction — they are already the reason the state is widened and already
  forced into the mover set. Measured over all 16 DM Control cells (scratchpad
  `probe_slots.py`): `DMCReacherEasy`, `DMCReacherHard` (geom `target`),
  `DMCFingerTurnEasy`, `DMCFingerTurnHard` (site `target`), and `DMCSwimmerSwimmer6`
  (geom `target` + light `target_light`, trackcom — see below). The other 11 cells draw
  nothing per episode and gain no leaf, so **their `describe()` and `style_dim` do not move.**
* **Default.** Read from the pristine model's resolved alpha, so the default appearance is
  the compiled model's by construction — 1 (drawn) on all four cells.
* **Binary, not continuous.** MuJoCo blends `0 < alpha < 1`; the analytic ray-caster is
  opaque-only. Declaring a continuous alpha would advertise something the renderer cannot
  honour and no check could catch. `Discrete(2)` says exactly what is true: drawn, or not.
* **Application.** In `DMControlSuite._apply_variation`, `getattr(m, f"{kind}_rgba")[i, 3] = 0`
  when the knob is 0. `_effective_alpha` then returns 0, `_visible` drops the element,
  `_movers` never lists it, and the three MuJoCo plate shots do not paint it either.
  Measured on the reference renderer (scratchpad `probe_hide.py`): zeroing the element's
  own alpha changes 19 px / max 215 on reacher-hard and 43 px / max 209 on finger-turn_hard,
  and restoring is 0/255. Per-*element* rather than per-*material* on purpose: on
  finger-turn the `target` material is worn by both the `target` and `tip` sites, so a
  material-alpha knob hides 58 px where the element knob hides 43.
* **Batching / purity / order.** Untouched. The knob is a style column, so the existing
  group-by-style path bakes one scene per distinct visibility and scatters rows back by
  index; `_apply_variation` still assigns absolutely from the pristine snapshot.

### Backends

`DMCReacher*` render through **raycast3d only**. The `["mjwarp", "raycast3d"]` pair in
`examples/results/step4/certification.json` is an artefact of how the census computes it
(`scratch/census/census.py::_backends` unions the `stable_envs.render.*` modules imported
by the *module* that defines `render` — `dmcontrol.py` imports both because the trackcom
tier uses MJWarp). No Reacher pixel goes through `stable_envs/render/mjwarp.py`.

MJWarp **cannot** carry this knob per row, and that is a shared-path finding rather than
something to work around: `_tc_context` calls `MW.prepare_model(m, VISIBLE_GROUPS)` and
its own comment records that "MJWarp's shader never reads alpha" — zero-alpha elements are
hidden by moving them out of the enabled geom groups at **context-creation** time, and
`mjw.create_render_context` reads the geometry once. A per-world visibility would need
either one captured context per distinct visibility combination or an alpha-aware shader
in `mujoco_warp`. So `DMControlTrackCom.VISIBILITY_KNOB = False`, following the existing
`CAMERA_KNOB = False` precedent (which is switched off there for the same class of reason:
`put_model` will not carry `cam_fovy` per world). **No edit to
`stable_envs/render/mjwarp.py` or `stable_envs/render/raycast3d.py` is required.**

### One latent bug this activates, and fixes

`_DMCModel._used_materials` reads `_effective_alpha` off the **live** shared `MjModel`,
which every bake mutates. Today that is harmless because `_apply_variation` restores rgba
from the pristine snapshot at the start of each bake. Once a bake can leave an element at
alpha 0, a later `_used_materials()` would see a different material set and a *new env's
declared variation space would depend on what the previous env last drew* — the exact
failure `_pristine`'s docstring records. Fixed by resolving alpha from the pristine
snapshot. Local `dmcontrol._effective_alpha` gains optional `rgba`/`mat_rgba` overrides;
`stable_envs/render/mjwarp.effective_alpha` is not touched.

### Files Problem 1 touches

* `stable_envs/envs/dmcontrol.py` — the whole of it.
* `examples/downstream_step5/reacher/{swap_renderer,state_bridge,parity_probe}.py` —
  switch the harness from `OFFSCREEN` to the knob (kept as a mode so the Step-5 arm stays
  reproducible).
* `tests/unit/test_step5.py` — the offscreen assertion follows the harness.
* A new targeted test file for the knob.

## 5. Problem 2 — the blind spot, and the guard

`interior_mask` keeps only pixels whose 3×3 neighbourhood in the reference is one flat
colour. `DMCReacher*`'s frame has **`interior_frac = 0.0` at 84 px** (frozen Step-4
`render_parity_with_official_renderer.observed`), and at 224 px the two Step-5 target
placements report `interior_exact_frac = 0.9998668941566535` and `interior_px = 82641`
to the last digit *in both cases*, while `mean_abs` moves 0.459 → 0.171
(`examples/results/step5/reacher/parity_probe.json`). A whole object appears and the
metric does not move, because the disc lands on a textured floor whose pixels the mask
already excluded.

**Guard: `conformance.check_task_object_visibility`.** One check, keyed to the elements
that now carry a `visibility.*` leaf — i.e. exactly the declared task-relevant objects.
For each such leaf it measures, on states the env actually reaches:

1. **footprint** — `M = (render(on) != render(off)).any(-1)`, the pixels our renderer
   attributes to the object, and `footprint_px` pooled over the batch. Zero footprint
   means the knob is inert.
2. **per-row** — a style whose rows alternate on/off must reproduce the all-on render on
   the on-rows and the all-off render on the off-rows, bit for bit. This is the collapse
   check `check_per_env_variation` structurally cannot make for a `Discrete` leaf
   (it skips anything without `.low`).
3. **the guard proper** — on `M`, the fraction of pixels where our frame agrees with the
   **official** frame (`|d| <= 8`) with the object ON, and with it OFF. The env's declared
   default must be the arm that agrees:
   `presence_margin = agree(declared) - agree(other) >= TASK_OBJECT_PRESENCE_MARGIN`.

That is decisive in the way `interior_exact_frac` is not: if our renderer draws an object
the reference does not draw (the Step-5 situation), the declared arm agrees with the
reference on ~0% of the object's own pixels and the margin goes negative. It measures a
population the interior mask throws away by construction, and it **adds** a bound rather
than redefining one — `interior_mask`, `pixel_agreement`, `INTERIOR_EXACT_MIN`,
`ALL_EXACT_MIN`, `MEAN_ABS_MAX` and every frozen Step-2 threshold are untouched.

The bound is measured before it is set, over independent draws, and set clear of the
worst draw — the rule the rest of the suite already follows.

### Files Problem 2 touches

* `stable_envs/testing/conformance.py` — the new check (additive).
* `tests/gpu/test_conformance.py` — the assertion + the module-level bound.
* `examples/fidelity_step4.py` — records the check in `family_specific` so a replay
  carries it (additive key; `not_applicable` for every env with no `visibility.*` leaf).
* A synthetic appear/disappear test that shows it FIRES.
