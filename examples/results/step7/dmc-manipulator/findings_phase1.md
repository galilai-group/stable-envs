# Step 7 / W3 `dmc-manipulator` — phase 1: PATCH

`swm/ManipulatorDMControl-v0` -> `DMCManipulatorBringBall`, registered, at the certified size 224 px.

- Baseline `76dc5116f432e61dd47fba8f9e879cf9212c8e46`; phase-0 commit `995e3b2`.
- Local venv `/Users/samib/se6-venv`, **mujoco 3.12.0 + dm_control 1.0.45**, `MUJOCO_GL=glfw`,
  `device="cpu"`, Apple M5. The declared DMC pair is **3.11.0 / 1.0.44**; that arm is PENDING (Oscar
  is DENIED at SSH auth: `Permission denied (keyboard-interactive)`).
- `stable_envs/render/raycast3d.py` and the `DMControlTrackCom` region are **untouched** (W4 owns both).
- Nothing under `examples/results/step4|step5|step6` was modified.

## 1. What changed

All in the `DMControlSuite` region of `stable_envs/envs/dmcontrol.py`, plus two research docs and one
new unit-test file.

| # | change | why |
|---|---|---|
| a | `_episode_slots` guard `("_size","_quat","_dir")` -> `("_size","_dir")`, docstring rewritten | Orientation is already per row: `_fk` writes the slot, runs `mj_kinematics`, reads `d.geom_xmat` into `(B,P,3,3)`, and `cast_nearest`/`any_blocking`/`_reflect` all consume it per row while `screen_span` (bounding SPHERE) never reads it. Size genuinely is shared at bake time (`_prims`, `screen_span`) and a light `_dir` is captured once per bake by `_Lights`, so both stay refused. |
| b | `_hideable` returns `(kind, name, ids)` and resolves a drawn **body** to the drawn geoms on it | manipulator's episode draws `body_pos`/`body_quat` of `target_ball`, a FRAME with no rgba. Without this the cell declared **no `visibility.*` leaf at all** and SWM's `render_target = 0` was unreachable. Raises if a drawn body carries both drawn geoms and drawn sites (nothing shipped does). |
| c | `SHAPE_KNOBS` + `<geom>.shape` Discrete leaf; `geom_type`/`geom_size` added to `_pristine` and restored in `_apply_variation` | Gate F: SWM's `target.shape`. It fits the plate path cleanly because `_prims` reads `geom_type`/`geom_size` AT BAKE TIME, so a style-group re-bake carries it with no per-row machinery — which is exactly why a per-STYLE size is fine while a per-EPISODE size is not. |
| d | `DOMAINS` entry, `DEFAULT_SIZES` (224), `SHAPE_KNOB_SPECS`, `STATE_DIMS_NOTE` | registration at the certified size, with the declaration attached. |
| e | `tests/unit/test_dmcontrol_episode_quat.py`, 6 tests | see §4. |
| f | `docs/research/dmcontrol.md` §2.1 + §8.2, `docs/research/swm_coverage.md` | the "untested, not refuted" row is now measured. |

`target_ball.shape` is `Discrete(3)`, not SWM's `Discrete(2)`: **neither** SWM alternative is the
compiled model's own target (`sphere size=.02` from `manipulator.xml`'s `ghost` class), so the leaf
carries SWM's `.015` box (0), SWM's `.015` sphere (1) and dm_control's `.02` sphere (2), and
`make_variation_space` asserts that **exactly one** alternative matches the pristine model and makes
that index the `init_value`. What that buys, stated precisely: **the default style renders the
compiled model** — the appearance every parity number here is measured under — and a model whose
target changes shape upstream fails loudly instead of silently redefining "default". It is *not*
`check_default_style_is_official` that establishes this: that check compares `render(states)` against
`render(states, style=default_style())` and never touches the official renderer, so a default that had
drifted off the compiled model would pass it. The `len(cur) != 1` assertion is the cover.

## 2. Certified configuration (coordinator ruling applied)

Parity is measured against `suite.load("manipulator","bring_ball")` (`reference_id
dm_control/manipulator-bring_ball`), as for the other 16 DMC cells. **The registered SWM default is
hidden-target and is reached through leaves** — the same accepted pattern as Reacher at Step-6 closure.
Measured at 224 px over 16 states, each against the suite-default frame:

| leaf | SWM leaf it reproduces | pixels moved | max |
|---|---|---|---|
| `visibility.target_ball = 0` | `rendering.render_target = 0` | 775 | 204 |
| `target_ball.shape = 0` | `target.shape = 0` (.015 box) | 783 | 204 |
| `target_ball.shape = 1` | `target.shape = 1` (.015 sphere) | 765 | 204 |
| `light.diffuse_scale = 0.7/0.6` | `light.intensity = 0.7` | 17,237 | 18 |
| **all three together** | **the SWM registered default** | **17,780 (2.21 %)** | 204 |

So the SWM default frame is reachable and differs from the suite default by 2.21 % of pixels — it is a
different appearance of the same scene, not a different scene. **O1 remains open and must be measured
before certification:** SWM round-trips the XML through `dm_control.mjcf` rather than calling
`suite.load`, and `_episode_slots`/`_prims` key on model order and `mj_id2name`.
`scratch/step7/dmc-manipulator/probe_swm_signature.py` + `oscar_o1_signature.sbatch` are written and
ready; they diff the full render-model signature (counts, geoms, sites, materials, textures incl. byte
hashes, cameras, lights, visual/quality block, and the discovered episode draw) between the two.

## 3. Results

### Parity (registered cell, no patching, `device="cpu"`, ss = 1)

**224 px, 12 independent draws of 64 states:**

| metric | worst | median | best |
|---|---|---|---|
| **`interior_exact_frac`** | **1.000000** | **1.000000** | **1.000000** |
| interior bad px | **0 of 16,612,773** | | |
| `interior_frac` | 0.43459 | 0.43495 | 0.43542 |
| `exact_frac` (all pixels) | 0.982859 | 0.982943 | 0.983072 |
| `mean_abs` | 0.2671 | 0.2615 | 0.2595 / 255 |
| `max_abs` | 222 | 205 | 204 |
| SSIM | 0.99366 | 0.99395 | 0.99419 |

**84 px, 3 draws of 64:** interior **1.0** (0 bad of ~33,080 per draw), `exact_frac`
0.97769–0.97799, mean 0.528–0.547, SSIM 0.98632–0.98709.

Declaration that follows: **`fidelity_tier = "parity"`, `parity_interior_min = 1.0` not relaxed, and
NO `parity_all_exact_min`** — the worst draw is 0.9829 against the suite's shared 0.90 bar. To be
re-declared from the ≥ 40-draw arm at the 3.11 pin before certification.

### Gate A, locally, at both sizes

`check_state_roundtrip`, `check_batch_invariance`, `check_batch_size_invariance`,
`check_order_independence`, `check_determinism`, `check_render_is_pure`,
`check_chunked_render_matches`, `check_input_form_invariance`, `check_set_state_copies`,
`check_style_is_state`, `check_default_style_is_official`, `check_per_env_variation`,
`check_variations_do_something`, `check_colors_apply_exactly` — **all pass at 224 and at 84**.

`check_task_object_visibility` (the bound the interior mask is blind to): `per_row_max_abs_diff = 0`,
`footprint_px` 734 at 224 / 104 at 84, **`presence_margin` 0.8696 at 224 and 0.7978 at 84** against
the `TASK_OBJECT_PRESENCE_MARGIN = 0.50` bar.

`check_resolution_consistency`: 0.8481 at 224, 0.4433 at 84 — the 84 px number sits between
DMCBallInCupCatch (0.4237) and DMCReacherHard (0.7908) and above DMCCartpoleSwingup (0.3169).

**`check_state_dims_matter` FAILS at both sizes: `["ball_y","target_ball_quat_qw",
"target_ball_quat_qy"]`.** Declared in `STATE_DIMS_NOTE`, not hidden — see §5.

### The other 15 DM Control cells are byte-identical

`probe_identity.py` run against a checkout of `76dc511` and against this branch, same venv, same
seeded states, at each cell's own default size, **at the default style and at a style that moves every
leaf** (colours to a fixed triple, scales to an end of range, every Discrete to 0 — which hides a
visibility leaf and switches a shape leaf): **16 cells compared, 0 differing** in `states_sha`,
`default_sha`, `moved_style_sha`, `state_dim`, `state_names` or `leaves`. Only
`DMCManipulatorBringBall` is new. (`DMCSwimmerSwimmer6` raises the same pre-existing CPU-backend
`ValueError` on both trees.) That covers the two functions the patch touches on the cells that
exercise them: `finger-turn`'s `site_pos` slots, `reacher`'s `geom_pos` slots and `swimmer`'s
`light_pos` slot.

### Frozen Step-4 DMC replay

`examples/fidelity_step4.py --shard dmc --device cpu` run on both trees:
**14 configs, 0 differing** once wall-clock `seconds` and absolute-path `traceback` strings are
dropped; `digest_ours` and `digest_official` match on every config. Output at
`examples/results/step7/dmc-manipulator/replay/dmc_cpu.json`.

### Test suite

`tests/unit`: **829 passed, 6 skipped, 2 failed** — `test_examples.py::…[robocasa_visual/
render_mjwarp.py]` and `test_repo_contracts.py::test_every_cited_scratch_path_resolves`. Both
reproduce on a clean checkout of `76dc511`, and the second one's broken-path set is **identical**
between the two trees (diffed), so neither is mine. The new
`tests/unit/test_dmcontrol_episode_quat.py` is 6 passed.

### Candidate frozen corpus

`examples/results/step7/dmc-manipulator/corpus/states/DMCManipulatorBringBall.npz` — the **selected
64** x 15, produced by importing `fidelity_step4.select` at `fidelity_step4.STATE_SEED = 20260909`
rather than reimplementing it, in the exact Step-4 `states`/`meta` schema (`env`, `seed`, `selection`,
`state_dim`, `state_names`, `sha256`, `pool`, `kept`, `n_typical`, `n_extreme`, `n_spread`, `indices`
— **no Step-4 key missing**, verified against `DMCReacherHard.npz`), plus additive Step-7 provenance
keys `corpus_for` does not read. `sha256 = 48db2b1adf54a75d816eb068f815257ffa7e1ab238218c4f9e64568873b749d8`;
`n_typical 1 / n_extreme 25 / n_spread 38`. The 256-row **pool** is kept beside it at
`corpus/pool/DMCManipulatorBringBall_pool.npz`, out of the path `corpus_for` reads.
Digests re-emitted on the selected 64: at 224 px `digest_ours 0327b3b4…`, `digest_official 73c43f90…`,
exact 0.983230, **interior 1.0, 0 bad of 1,386,228**; at 84 px `bb2b48ad…` / `1367ea83…`, exact
0.978310, **interior 1.0, 0 bad of 33,081**. **Candidate only; not frozen in Step 7.**

### Throughput (local Mac CPU — NOT a certification number)

224 px, ss = 1: B=1 43 f/s, B=64 54 f/s, B=256 38 f/s. Certification numbers come from Oscar.

## 4. The new unit test

`tests/unit/test_dmcontrol_episode_quat.py` asserts the claims the patch rests on, not that the cell
loads:

1. the guard's shape is literally `("_size", "_dir")` and `_quat` is not in it — so widening it
   further to allow a per-episode SIZE cannot happen silently;
2. the census finds exactly `body_pos` + `body_quat` of `target_ball` and `state_dim == nq + 4 == 15`;
3. **`_fk`'s `xmat` differs ACROSS ROWS** for two rows that share every `qpos` entry and every
   positional slot and differ only by 0.7 rad of target rotation — the property that would silently
   fail if orientation ever became batch-shared, and which no existing test could see;
4. the OFFICIAL renderer moves pixels for those slots, so the state carries nothing the reference
   ignores (asserted about `physics.render`, deliberately not about us — see §5);
5. a body slot becomes one `visibility.target_ball` leaf over the body's geoms, and hiding it changes
   the frame;
6. `target_ball.shape`'s default is the compiled model's, and **each** of the other two alternatives
   changes the frame.

The reference env is taken from `conformance.reference_env` rather than bound to a local: the first
version of test 4 failed the repo's `EGL_CONTEXT_RELEASED` resource guard, which is the rule this
family has paid for four times.

## 5. The declared failure, and the contract change it asks for

`check_state_dims_matter` reports `ball_y`, `target_ball_quat_qw`, `target_ball_quat_qy` dead. The
failure is real, is recorded as FAIL, and is **not** worked around — no widening of the shared check,
and no Gate A `partial` (per the ruling: `partial` means a documented subset of the OBSERVATION, and
three rotation-invariant sphere dims are not that).

All three orient an **untextured sphere**: `ball` is `sphere size=.022` on material `self`,
`target_ball` is `sphere size=.02` on material `target`, and neither material carries a texture.
MuJoCo draws a sphere as a **tessellated mesh shaded per vertex**, so rotating it rotates the mesh and
moves the Gouraud interpolation and the specular vertex; an analytic per-pixel sphere is **exactly**
rotation-invariant and cannot move. Measured over 16 states at 224 px, each dimension perturbed and
each renderer diffed against its own baseline:

| dimension | ours | official | official max |
|---|---|---|---|
| the other 12 of 15 | 119 – 17,632 px | 242 – 20,208 px | 60 – 219 |
| `ball_y` (+0.15 rad) | **0** | 896 px | 55 / 255 |
| `target_ball_quat_qw` (+0.4 rad about y) | **0** | 734 px | 33 / 255 |
| `target_ball_quat_qy` (same rotation) | **0** | 734 px | 33 / 255 |

Bounded three ways. (i) Parity is unaffected: all-pixel exactness after each perturbation is
0.98275–0.98277 against a 0.98280 baseline and interior agreement stays 1.000000 — the moved pixels
are already inside the silhouette band that disagrees anyway. (ii) The two quaternion slots become
**LIVE under `target_ball.shape = 0`**, where the target is a BOX: measured 203 px, max 145/255,
against 0 px under the default sphere. So this is a property of the sphere, demonstrated, not a
dropped dimension. (iii) Closing it inside this design means reproducing MuJoCo's icosphere
tessellation and per-vertex shading in the mover pass — replacing the analytic caster for **every** DM
Control cell to recover ~800 px at ≤ 55/255 on one of them.

**Proposed contract change, for Sami to approve or refuse — written down, not applied.** A per-class
declared exclusion list that `check_state_dims_matter` itself **validates**, so the escape cannot be
used to hide a dropped dimension:

> A dimension may be declared excluded only if, for the same write the check makes, the renderer's
> scene changes **exclusively** in the orientation of SPHERE geoms. The check verifies this from the
> model rather than trusting the declaration: after writing the dimension it compares `geom_xpos`,
> `geom_size`, `geom_type`, every material/light/camera array, and the `geom_xmat` of every
> **non-sphere** geom, before and after. If any of those move, the exclusion is refused and the
> dimension is reported dead as it is today. A dimension that is genuinely rotation-invariant geometry
> passes; a dimension the renderer simply failed to draw cannot.

Until that exists, this cell ships with the check recorded FAIL and `STATE_DIMS_NOTE` attached to the
class, carrying the mechanism, the measurement and this proposal.

## 6. Still pending — every one needs Oscar (DENIED at SSH auth, no job IDs)

Scripts are written, committed and rsync-ready under `scratch/step7/dmc-manipulator/`:

| script | what it does |
|---|---|
| `oscar_o1_signature.sbatch` + `probe_swm_signature.py` | **O1**: diff SWM's mjcf round-trip against `suite.load` (needs `se-step5`, the only venv with `stable_worldmodel`) |
| `oscar_certify311.sbatch` + `probe_draws.py`, `probe_freeze.py` | the **3.11.0 / 1.0.44** arm on `se-vis311`: ≥ 40 draws of 64 at 224 and 84, the DMC shard conformance run, and the corpus at the declared pin. A 3.11-certified cell travels in its own shard (Step-6 rule). |
| `oscar_gpu.sbatch` | CUDA: the 15 pre-existing cells BEFORE (`$SCR/se7/baseline`) and AFTER on the **same 3090 in the same job**, the new cell's GPU conformance, and its throughput sweep. One GPU job, `--time 03:00:00`, `--cpus-per-task 4`, `--mem 48G`. |

Also pending: `tests/gpu/*` (CUDA-only locally unreachable), the paired performance sweeps, and
`floor.color`.

## 7. Gate F — SWM render compatibility

| SWM leaf | status |
|---|---|
| `agent.color` | covered — `material.self` |
| `target.color` | covered — `material.target` |
| `light.intensity` | covered in range — `light.diffuse_scale` in [0.2, 1.8] on base .6 gives [.12, 1.0] after the clip; SWM's 0.7 is reached at scale 7/6 **exact to the style pipeline's quantisation, not bit-exact** — see below |
| `rendering.render_target` | **covered now** — `visibility.target_ball` (change b) |
| `target.shape` | **covered now** — `target_ball.shape`, both SWM alternatives reachable (change c) |
| `agent.upper_arm_density`, `agent.hand_density` | n/a — physics (mass); no pixel effect |
| `agent.upper_arm_length` | **dead in SWM itself** — `modify_mjcf_model` never reads it (`# TODO should modify length of arm`). Not implemented: a knob the reference does not honour. |
| `floor.color` | **GAP, out of phase-1 scope by ruling — and a RANGE gap, not a default gap.** SWM edits the builtin `grid` **texture**'s `rgb1`/`rgb2`, not a material rgba, and `_bundle` holds **one `mujoco.Renderer` per (domain, size) that is never closed** — releasing an EGL context corrupts every other live one. **Proposed measurement:** regenerate the builtin checker into `m.tex_data`/`m.tex_rgb` in place from `(rgb1, rgb2, width, height, mark, markrgb)` and re-upload, then assert byte-equality against a model compiled from an XML carrying those colours directly, over the 64-state corpus at 224 px. If that is exact, `floor.color` becomes an ordinary plate leaf at zero GL cost; if not, the fallbacks are keying the Renderer on the texture (N live contexts) or declaring the leaf out of scope with this measurement attached. |

**On `light.intensity`'s exactness (C4/R8), with the mechanism rather than the label.** Applied
directly, the scale is bit-exact: `_apply_variation` with `7/6` writes `light_diffuse = 0.699999988079071`,
which *is* `float32(0.7)`, delta 0.0. Through `render(variation=...)` it is not, and float32 is not why —
`render` builds its style-group key with `np.round(key.astype(np.float64), 6)`, so the scale that
reaches the bake is `1.166667` and the light lands at `0.7000002264976501`, **delta +2.4e-07**. Measured
over 16 states at 224 px, that difference moves **0 pixels** (max 0). So: *exact to the style
pipeline's 6-decimal quantisation, and pixel-identical at this scene's precision* — but not the word
"exact" unqualified.

**On `floor.color` being a range gap (R8).** SWM's `floor.color` `init_value` is
`[[.1,.2,.3],[.2,.3,.4]]`, which is exactly `common/materials.xml`'s `grid` texture `rgb1`/`rgb2`. So
the registered SWM **default** floor is already the compiled model's floor and needs no leaf at all —
the gap is only that non-default values of that leaf are unreachable. That is a materially smaller
claim than "the SWM default is not reproducible", and the §2 table above is unaffected: the default
frame is reached through the other three leaves.

## 8. Report summary

* **Classification: `PATCH`.**
* **Affected configs:** new `DMCManipulatorBringBall` (parity, 224 px); 15 existing DM Control cells
  byte-identical, proven against a `76dc511` checkout at two styles plus the frozen Step-4 replay.
* **Pixels changed on existing cells: NO** — 0 of 16 cells differ, 0 of 14 replay configs differ.
* **Tiers:** new cell `parity`, `parity_interior_min = 1.0` unrelaxed, no all-exact floor needed.
  Gate A has one declared FAIL (`check_state_dims_matter`, §5); no other check fails.
* **Semantic follow-up: none.** No existing env's pixels move and the new cell is a parity cell.
* **SLURM job IDs:** none. Oscar DENIED at SSH auth.
* **Open for the coordinator:** O1 (mjcf round-trip signature diff — script ready); approval of the
  §5 validated-exclusion contract mechanism; `floor.color` measurement scheduling; and confirmation
  that `Discrete(3)` for `target_ball.shape` (SWM's two alternatives plus the compiled model's own) is
  the right shape for the Gate-F mapping.

---

## 9. Review follow-up (independent review of `a635d3a`: APPROVE-WITH-CHANGES)

All four required changes applied, plus the three cheap recommendations. Nothing in the renderer's
behaviour moves: the 16-cell byte-identity and the Step-4 replay were re-run after these edits and are
unchanged (§3), and the new cell's own digests are unchanged except where C3 deliberately changed which
states the corpus holds.

**C1 — `_movers` filed a BODY id into the SITE id set.** `(ep_g if kind == "geom" else ep_s).add(eid)`
put manipulator's `target_ball` **body id 11** into `ep_s` against `nsite = 9`. Inert by arithmetic
accident — no site carries id 11 — and manipulator is the first cell with a body slot, so nothing had
exercised it. Now an explicit three-way branch that ignores body and light, with the reason recorded in
the docstring: the second mover rule exists only to rescue an element the FIRST rule misclassifies, and
a body slot cannot be in that position (`target_ball` is its own body, so every geom on it already has
`geom_bodyid != 0` and rule one takes it). An id-space confusion that is inert by accident is a defect:
a domain whose drawn body id collided with a real site id would silently promote an unrelated site to a
mover and pull it out of the plate.

**C2 — four `check_default_style_is_official` overclaims** (dmcontrol.py:293 and :953, the phase-1
findings, and the `a635d3a` commit body). That check compares `render(states)` against
`render(states, style=default_style())` and never touches the official renderer, so it cannot see a
default that drifted off the compiled model. All four now say what the `len(cur) != 1` assertion
actually protects — *the default style renders the compiled model, which is the appearance parity
against `suite.load` is measured under* — and name the check's blind spot explicitly so the
substitution is not made again. The commit body cannot be edited (it is `a635d3a`'s), so the correction
is recorded here and in the two source comments.

**C3 (R5) — the corpus held the POOL, not the selection.** `corpus_for` never resamples an existing
file, so at the Step-8 freeze `states/<cell>.npz` would have handed the harness **256** states rather
than 64, against a corpus that had never been through the selection rule, with no `sha256` or
`indices` to verify it and a `config_id` key where the schema wants `env`. Rewritten to import
`fidelity_step4.select` and `STATE_SEED` themselves — reimplementing the selection would be a second
definition of the corpus, and the two would drift exactly once — and to write the selected 64 in the
exact Step-4 schema. See §3 for the digests on the selected rows.

**C4 (R8) — two wording corrections**, both now carried with their mechanism rather than a label: the
`light.intensity` mapping is exact to the style pipeline's 6-decimal quantisation (delta +2.4e-07 on
the light's diffuse, **0 pixels moved**), not bit-exact through `render`; and `floor.color` is a
**range** gap, not a default gap, because SWM's `init_value` already equals the compiled `grid`
texture's `rgb1`/`rgb2`. Both are in §7.

**R2 (recommended, done)** — `tests/unit/test_dmcontrol_episode_quat.py` gains two synthetic-model
tests, because no registered env can reach either path: `target_ball` carries exactly one drawn geom.
A hand-built body with four geoms (two drawn, one at alpha 0, one in group 3) must resolve to exactly
the two drawn ones, and a body carrying both drawn geoms and drawn sites must raise. 8 tests pass.

**R3 (recommended, done)** — a geom named `material`/`light`/`camera`/`visibility` would put its shape
leaf at `<group>.shape` and **replace** that whole variation group, silently deleting every colour
knob in it. `make_variation_space` now raises on the collision rather than losing leaves.

**R4 (recommended, done)** — a docstring note where `_apply_variation` writes `geom_type`/`geom_size`:
`geom_rbound` and `geom_aabb` are compiled from those arrays and are **not** recomputed, so they go
stale. Harmless here, and the note says why rather than asserting it — rendering never reads them
(`mjv_updateScene` and `_prims` both read type/size directly, and `screen_span`'s bound is rebuilt each
bake), the only geom any `SHAPE_KNOBS` entry touches is a `ghost`-class target with
`contype = conaffinity = 0`, and nothing in this file steps the physics. A shape leaf on a *colliding*
geom would need `mj_setConst` and would be changing dynamics, which is not what an appearance knob is.

**Recorded, no action taken.** (i) `light_pos` is an allowed episode slot while `light_dir` is refused,
and `DMCSwimmerSwimmer6` already carries two `light_pos` slots — a probable latent family defect, since
`_Lights` is captured once per bake and a per-episode light position is no more per-row than a
direction. Not touched here: it is a live shipped cell's behaviour and belongs to Step 8 with its own
measurement. (ii) **0.9829 is not promoted to a floor** — the reviewer's independent draw hit 0.982834,
below my worst 0.982859, which is exactly the margin-of-luck problem `PARITY_NOTE` records; the cell
declares `parity_interior_min = 1.0` and no all-exact floor. (iii) The shape leaf's fidelity against
SWM's *recompiled* MJCF is O1's question, not one this branch can answer locally.
