# Step 7 / W3 `dmc-manipulator` — INDEPENDENT REVIEW of phase 1

**Verdict: APPROVE-WITH-CHANGES.**

Reviewer did not write this code. Every headline claim was re-measured with the reviewer's own
scripts (not the author's probes) and every one reproduced. Two changes are required before merge;
both are small and neither invalidates a measurement. Six further recommendations are advisory.

- Branch `step7/dmc-manipulator`, worktree `/Users/samib/stable-envs-step7-dmc-manipulator`
- HEAD `a635d3a1b45f14245a2d45b980caa2844ba56a11`, phase-0 `995e3b2`, baseline
  `76dc5116f432e61dd47fba8f9e879cf9212c8e46` (`git merge-base --is-ancestor` confirms ancestry)
- Reviewer baseline tree: fresh detached worktree at `…/scratchpad/w3base` @ `76dc511`, clean
- Environment: `/Users/samib/se6-venv/bin/python`, mujoco **3.12.0**, dm_control **1.0.45**,
  `MUJOCO_GL=glfw`, `device="cpu"`, Apple M5. **The declared DMC reference pair is 3.11.0 / 1.0.44
  and that arm does not exist locally.** No CUDA locally. Oscar SSH is denied by the review brief and
  was NOT attempted. Nothing in this review is a certification number.
- No file in the worktree was edited; nothing was committed or pushed.

---

## 1. `_episode_slots`: is "orientation is per row" true for EVERY consumer?

**Yes, by code trace and by measurement.** Traced every consumer of a mover's orientation in the
`DMControlSuite` path:

| consumer | how it gets orientation | per row? |
|---|---|---|
| `_fk` (dmcontrol.py:1830) | `np.take(d.geom_xmat, gid, out=xm[i, …])` inside the `for i in range(B)` loop after a per-row `mj_kinematics`/`mj_forward` | **yes**, `xm` is `(B,P,3,3)` |
| `_cast` -> `R3.cast_nearest(kind, sizes, xpos, xmat, o, d)` | `R` documented `(B,K,3,3)` (raycast3d.py:665-688) | **yes** |
| `_occluded` -> `R3.any_blocking(…, xmat, …)` | same tensor | **yes** |
| `_reflect` (dmcontrol.py:2169-2186) | forwards the same `xmat` into `_cast` | **yes** |
| `screen_span` (raycast3d.py:1606) | `prims.radii()` + `xpos` only; **never reads `xmat`** | rotation-irrelevant |
| `bound_radius` (raycast3d.py:223) | `SPHERE -> s0`, `CAPSULE/CYLINDER -> s0+s1`, `BOX -> ‖s‖` | rotation-invariant, and correct for the BOX the shape leaf can produce |
| plate bake (`_bake`, 1534-1645) | movers are hidden (`m.geom_group[gmov] = HIDDEN_GROUP`) for all three plate shots; `sc.prims, _, _ = self._prims(...)` **discards** the bake-time mover `xpos`/`xmat` | no bake-time capture |
| `sc.floor` | `d.geom_xmat` of the STATIC reflective plane only | static, correct |

**No bake-time capture of a mover's orientation exists.** The only things captured once per bake are
`prims.size` / `prims.mat_rgb` / `_Lights` / the three plates / the static prims — exactly the set the
rewritten docstring names.

Additional note: this adds nothing new to the fused CUDA kernel either. Mover orientations already
vary per row on every existing cell (cartpole's pole is a `qpos`-driven `xmat`); a per-episode `quat`
enters the same `(B,P,3,3)` tensor by the same route. So the CUDA arm being unmeasured locally is not
a gap specific to this change.

**SIZE and light `_dir` remain refused** — `bad = [s for s in slots if s[0].endswith(("_size", "_dir"))]`
catches `geom_size`, `site_size`, `light_dir`.

**No existing cell's declared episode slots changed** — see §4; `slots`, `state_dim`, `state_names`
and `hideable` are identical on all 16 pre-existing cells.

## 2. `_hideable` body -> geom resolution

Measured on the compiled `manipulator-bring_ball` model (reviewer script `w3_probe_model.py`,
`w3_body.py`):

```
body target_ball id 11, geom id 20, site id 8
slots: (('body_pos',33), ('body_pos',35), ('body_quat',44), ('body_quat',46))   # body 11, x/z/qw/qy
geoms on body 11: [(20,'target_ball',group 0)]      sites on body 11: [(8,'target_ball',group 3)]
VISIBLE_GROUPS = (0,1,2)  ->  site 8 is not drawn  ->  no raise
hideable: (('geom','target_ball',(20,)),)   visibility.target_ball init_value = 1
```

- The resolution expression is `geom_bodyid[i] == eid and pristine group in VISIBLE_GROUPS and
  alpha > 0`, i.e. **all drawn geoms on the body and only those**, by construction. Applied to every
  body in the model it returns exactly the body's own drawn geoms (verified for all 12 bodies,
  including the 3-geom `hand` and the 2-geom `thumb`/`finger`/`*tip` bodies).
- The raise fires. Forcing site 8 into group 0 with alpha 1 in a throwaway process gives:
  `DMCManipulatorBringBall: the episode draws body 'target_ball', which carries both drawn geoms [20]
  and drawn sites [8]; one visibility leaf cannot name both kinds`.
- **Reacher / finger-turn / swimmer behaviour is unchanged**: `kind in ("geom","site")` takes the old
  branch with `ids = (eid,)`, `init_value = int(any(a[i] > 0 for i in (eid,)))` is the old expression,
  and §4 shows their `hideable`, `leaves`, `init` values and pixels are byte-identical.
- The census now reads the PRISTINE snapshot rather than the live (last-baked) model — a strict
  improvement, consistent with the `_pristine` docstring's own argument.

## 3. `target_ball.shape`

Verified against both sources.

- **Pristine model**: `geom_type[target_ball] = 2 (SPHERE)`, `geom_size = [0.02, 0, 0]`. Confirmed in
  `dm_control/suite/manipulator.xml`: `<body name="target_ball" childclass="ghost"><geom
  name="target_ball" type="sphere" size=".02"/>`.
- **SWM registration** (`/Users/samib/stable-worldmodel/stable_worldmodel/envs/dmcontrol/manipulator.py`):
  `_SMALL_TARGET = 0.015`; `target.shape = Discrete(2, init_value=1)`; `shape_id == 0 -> type 'box',
  size [.015,.015,.015]`, else `type 'sphere', size [.015]`. **Both branches write `size`**, so SWM's
  own default is a `.015` sphere, not dm_control's `.02`. `swm/ManipulatorDMControl-v0` is registered
  in `stable_worldmodel/envs/__init__.py:111`, and `DMControlWrapper.reset` calls
  `modify_mjcf_model` **unconditionally** (dmcontrol.py:74), so those init values *are* the registered
  default appearance. `render(width=224, height=224)` confirms `DEFAULT_SIZES = 224`.
- `SHAPE_KNOB_SPECS` maps exactly onto that: `(BOX,(.015,)*3)`, `(SPHERE,(.015,0,0))`,
  `(SPHERE,(.02,0,0))`; live value `{'target_ball': ((6,(0.015,0.015,0.015)), (2,(0.015,0,0)),
  (2,(0.02,0,0)))}`. `6 == mjGEOM_BOX`, `2 == mjGEOM_SPHERE`, matching `raycast3d`'s
  `PLANE, SPHERE, CAPSULE, CYLINDER, BOX = 0, 2, 3, 5, 6`.
- `init_value` is **derived**: `len(cur) != 1` raises, and the measured `init = 2`.
- **Re-bake is correct and not stale.** `PLATE_PATHS` is `tuple(p for p, _, _ in
  variation_space.flat_layout())` — every leaf — so `target_ball.shape` is in the `_SCENE_CACHE` key
  and in `render`'s style-grouping key. Two styles differing only in shape cannot share a bake.
  `_apply_variation` restores `geom_type`/`geom_size` from pristine before applying, so bake order
  does not matter.
- **No leak into other cells.** `SHAPE_KNOBS == {}` on every other cell; `m.geom_type[:] = pr[...]`
  and `m.geom_size[:] = pr[...]` are numeric no-ops there (nothing else in the file writes either
  array — `grep` shows the only writers are the restore and the knob). Byte-identity confirmed in §4.

### Opinion on `Discrete(3)` vs `Discrete(2)` (argued, not decided)

**For `Discrete(3)` (the author's choice).** The family's parity contract is *measured at default
style against `suite.load`*, and this cell reports `interior_exact_frac = 1.000000` under exactly that
definition. A `Discrete(2)` leaf over SWM's two alternatives would force the default to be SWM's
`.015` sphere, which is **not** the compiled model — so either the default no longer equals the
official appearance (breaking the family invariant every other DMC cell holds) or parity would have to
be re-defined for this one cell against a non-`suite.load` reference. The style contract is *total*
(`default_style()` packs every leaf; `check_style_is_state`), so "leave the knob unset" is not
representable. Given those two constraints, a third alternative is the only way to keep both
"default == compiled model" and "both SWM arms reachable", and deriving the default index from the
model rather than transcribing it is the right hardening.

**Against `Discrete(3)`.** Gate F is about *reproducing* the reference's variation space, and this
leaf is a strict superset: index 2 has no SWM meaning. `uniform_style` and any downstream
style-sampling loop will draw it ~1/3 of the time, producing appearances the reference env cannot
produce — which weakens what "covers SWM's variation space" means for a training/eval consumer that
samples styles. The cheap mitigations are (a) reorder so the compiled-model alternative is index 0
and the SWM indices are 1/2 (makes the extra value obviously ours, but breaks the SWM index mapping
outright), or (b) keep the order and export the SWM->index mapping explicitly (`SHAPE_KNOBS` already
is that table) plus a note in the Gate-F doc that index 2 is outside the reference space. I lean
toward keeping `Discrete(3)` with (b), but this is the coordinator's call.

## 4. Regression: are the 15 pre-existing DM Control cells untouched?

**Reproduced, and more strictly than the author did.** Reviewer script `w3_ident.py` run under
identical seeds against `…/scratchpad/w3base` (pristine `76dc511`) and against the branch, comparing
per cell: `size`, `state_dim`, `state_names`, `leaves`, `_slots()`, `_hideable()`, every leaf's
`init_value`, `states_sha`, and FOUR frame digests — default style, a style with every Discrete at 0
(hides the visibility leaf, switches the shape leaf) and colours/scales moved, a style with every
Discrete at 1 and `light.diffuse_scale = 1.8`, and a render at a NON-default size (84) which crosses
the `_bundle(default_size)` / `_bundle(size)` pristine boundary.

```
base cells 16, branch cells 17;  new on branch: ['DMCManipulatorBringBall'];  removed: []
cells compared: 16   differing: 0
DMCSwimmerSwimmer6: SKIP ValueError on BOTH trees (pre-existing: no CPU 3-D backend)
```

`tests/unit` (deselecting `tests/unit/test_examples.py` per the brief), same venv, both trees:

| tree | result |
|---|---|
| `76dc511` | **1 failed, 713 passed, 6 skipped** |
| branch | **1 failed, 719 passed, 6 skipped** (+6 = the new `test_dmcontrol_episode_quat.py`) |

The single failure is `tests/unit/test_repo_contracts.py::test_every_cited_scratch_path_resolves` on
both, and its **broken-path set is byte-identical** (32 paths, `diff` empty). Pre-existing, not the
author's.

## 5. Parity and Gate A, re-measured independently

Reviewer driver `w3_gate.py` (calls `stable_envs.testing.conformance` directly; `measure()` in
`examples/fidelity_step4.py` needs a `certification.json` row this cell does not have yet). It
reproduces `fidelity_step4.py`'s `reseed(env)` before every check and before every parity draw —
**omitting that was my own first-run error and produced garbage; the numbers below are the
reseeded run.**

`se.make("DMCManipulatorBringBall", batch=64, size=S, device="cpu")`; 3 parity draws of 64 states at
seeds 7000/7001/7002 (different from the author's).

### 224 px

| parity draw | `interior_exact_frac` | interior bad / px | `exact_frac` | mean\|d\| | SSIM | `interior_frac` |
|---|---|---|---|---|---|---|
| 0 | **1.000000** | 0 / 1,384,693 | 0.982834 | 0.2691 | 0.99356 | 0.43508 |
| 1 | **1.000000** | 0 / 1,384,599 | 0.982967 | 0.2631 | 0.99378 | 0.43505 |
| 2 | **1.000000** | 0 / 1,384,831 | 0.982924 | 0.2648 | 0.99397 | 0.43512 |

**0 bad interior pixels of 4,154,123.** Matches the author's 12-draw `0 of 16,612,773`. (Their
committed `phase1/draws_224_cpu.json` is internally consistent: 12 draws, interior 1.0 on every one,
worst all-pixel `exact_frac` 0.982859. My worst is 0.982834 — slightly under their "≥ 0.9829" prose,
on a different draw. Harmless *because no `parity_all_exact_min` is declared*; see recommendation R6.)

Gate A at 224: `check_state_roundtrip` 0.0 · `check_batch_invariance` 0 · `check_batch_size_invariance`
0 (1/7/64) · `check_order_independence` 0/0 · `check_determinism` true · `check_render_is_pure` 0 ·
`check_chunked_render_matches` 0 (1/7/64/4096) · `check_input_form_invariance` {} ·
`check_set_state_copies` 0 · `check_style_is_state` 0 · **`check_default_style_is_official` 0** ·
`check_per_env_variation` [] · `check_variations_do_something` **[]** · `check_colors_apply_exactly`
[] — **14/14 pass.**

`check_task_object_visibility` (64 states): `footprint_px` 2944, `rows_with_footprint` 63,
`per_row_max_abs_diff` **0**, `presence_mean_abs_declared` 13.30 vs `flipped` 102.92,
**`presence_margin` 0.8708** against the `TASK_OBJECT_PRESENCE_MARGIN = 0.50` bar (author: 0.8696 on
16 states).

`check_resolution_consistency` 0.7875 (author 0.8481 — different random style and state draw; not a
thresholded quantity).

### 84 px

3 draws: `interior_exact_frac` **1.0** on all three (0 bad of 33,077 / 33,077 / 33,068),
`exact_frac` 0.97754 / 0.97772 / 0.97767, SSIM 0.9855–0.9875. Same 14/14 Gate A.
`check_task_object_visibility`: `footprint_px` 418, `per_row_max_abs_diff` 0,
**`presence_margin` 0.7442** (author 0.7978). Still far above 0.50.

### `check_state_dims_matter`

**FAILS on exactly `["ball_y", "target_ball_quat_qw", "target_ball_quat_qy"]` and nothing else**, at
both 224 and 84, reproduced independently. I then re-measured the declared mechanism myself
(`w3_dims.py`, 16 states at seed 31337, donor substitution, our renderer vs `physics.render`, and our
renderer with `target_ball.shape = 0`):

```
dim                       ours px ours max  official px  off max  ours@box px  box max
arm_root                    21200      204        24938      212        21049      204
...  (every other dim moves in BOTH renderers) ...
ball_y                          0        0          591       58            0        0
target_ball_pos_x            1473      204         1814      192         1082      192
target_ball_quat_qw             0        0          685       37          135      192
target_ball_quat_qy             0        0          713       40          144      192
```

This confirms all three legs of `STATE_DIMS_NOTE`: (i) the three dims move 0 of our pixels and
591–713 of the official ones at max 37–58/255; (ii) every other dimension is live in both; (iii) the
two quaternion slots become **LIVE under `target_ball.shape = 0`** (135 and 144 px at max 192/255),
which demonstrates the residual is a property of the untextured SPHERE and not a dropped dimension.

The check is **recorded as FAIL and not widened** — `stable_envs/testing/` is untouched by the diff.
There is no Gate-A `partial`, no `certification.json` edit, and no coverage claim in this branch.

### Is the measured cell the registered SWM default?

The cell is measured against `suite.load("manipulator","bring_ball")` (`reference_id
dm_control/manipulator-bring_ball`) at `default_size = 224`, camera 0 (`fixed`, `pos 0 -16 .4`,
`fovy 4` — confirmed on the model), under the model's one DIRECTIONAL light (`diffuse .6 .6 .6`).
Reviewer re-measured the Reacher-pattern leaf reachability (`w3_swmleaf.py`, 16 states at seed 555,
each against the suite-default frame):

| leaf | SWM leaf | px moved (mine) | px moved (author) |
|---|---|---|---|
| `visibility.target_ball = 0` | `rendering.render_target = 0` | 779 | 775 |
| `target_ball.shape = 0` | `target.shape = 0` (.015 box) | 777 | 783 |
| `target_ball.shape = 1` | `target.shape = 1` (.015 sphere) | 773 | 765 |
| `light.diffuse_scale = 7/6` | `light.intensity = 0.7` | 17,175 | 17,237 |
| all three | the SWM registered default | **17,719 (2.207 %)** | 17,780 (2.21 %) |

**Plausible and reproduced.** Two supporting facts I checked that the findings do not spell out, and
which matter for "reached through leaves":

- SWM's OTHER default values are already the compiled model's, so no further leaf is needed for the
  default frame: `agent.color [0.7,0.5,0.3]` == `material self rgba ".7 .5 .3 1"`; `target.color
  [0.6,0.3,0.3]` == `material target rgba ".6 .3 .3 1"`; `floor.color [[.1,.2,.3],[.2,.3,.4]]` ==
  the builtin `grid` texture's `rgb1`/`rgb2`. So the `floor.color` GAP in the author's Gate-F table
  is a gap in the knob's RANGE, not in the default — worth stating explicitly, because it is the
  difference between "the SWM default is reachable" (true) and "the SWM variation space is covered"
  (not yet).
- `light.diffuse_scale = 7/6` gives float32 `0.70000005` where SWM writes float32 `0.69999999`.
  Exact to float32 rounding, ~7e-8, no pixel consequence — but "exact" in the Gate-F table is a
  slight overclaim.

### Frozen Step-4 DMC replay

`examples/fidelity_step4.py --shard dmc --device cpu` run by the reviewer on **both** trees:
**14 configs on each tree, 0 differing.** After dropping only wall-clock `seconds` and the
absolute-path `traceback` strings (which name the tree root and therefore always differ), the two
JSON documents compare **equal on every config**, including `digest_ours`, `digest_official`,
`digest_ours_default_style`, `digest_ours_corpus_style`, `mean_pixel` and `distinct_colors`:

```
DMCAcrobotSwingup  DMCBallInCupCatch  DMCCartpoleBalance  DMCCartpoleSwingup  DMCCheetahRun
DMCFingerSpin      DMCFingerTurnEasy  DMCFingerTurnHard   DMCHopperHop        DMCHumanoidStand
DMCPendulumSwingup DMCReacherEasy     DMCReacherHard      DMCWalkerWalk
  -> configs with any differing digest: 0 of 14
```

This independently reproduces the author's replay claim.

## 6. Hygiene

```
git log --format=%B 76dc5116..HEAD | grep -ci -E "co-authored-by|generated with"   ->  0
git log --format='%an <%ae>' 76dc5116..HEAD | sort -u   ->  sami-bg <samibghanem@gmail.com>
git diff --name-only 76dc5116..HEAD | grep -E '^examples/results/step(4|5|6)/'      ->  (empty)
git diff --name-only 76dc5116..HEAD | grep -E 'raycast3d|render/'                   ->  (empty)
git diff 76dc5116..HEAD -- stable_envs/envs/dmcontrol.py | grep -i trackcom          ->  1 context line only
git diff --name-only 76dc5116..HEAD | grep -i 'certification|declarations'          ->  (empty)
git merge-base --is-ancestor 76dc511 HEAD                                           ->  true
```

Two commits, both plain-messaged, both `sami-bg`. `stable_envs/render/raycast3d.py` and the
`DMControlTrackCom` region are untouched (W4's ownership respected). Files changed: `dmcontrol.py`,
two research docs, one new test file, `examples/results/step7/dmc-manipulator/**`,
`scratch/step7/dmc-manipulator/**`.

**Candidate corpus — Step-4 format only at the container level.** `corpus/states/
DMCManipulatorBringBall.npz` has the right keys (`states` float32 + `meta` as a JSON string), but:
it holds **256 rows**, which is `POOL`, where Step-4's frozen files hold the SELECTED `N_STATES = 64`
(`POOL, N_STATES = 256, 64`); and its `meta` uses `config_id` where Step-4 uses `env`, and omits
`selection` / `indices` / `sha256`. `corpus_for()` reads the npz opaquely, so dropping this file into
`examples/results/step4/states/` at the Step-8 freeze would silently hand the harness 256 states
instead of 64. See R5.

## 7. Policy

| rule | verdict |
|---|---|
| no per-environment hack in shared code | **OK.** `SHAPE_KNOBS` is a generic class attribute defaulting to `{}` on `_DMCModel`; the `_apply_variation` and `make_variation_space` code paths are name-agnostic loops. `SHAPE_KNOB_SPECS` is a module-level per-cell table, the same idiom the file already uses for `ALL_EXACT_FLOOR`, `VISUAL_FIDELITY`, `TRACKCOM_RENDER_FLAGS`, `DEFAULT_SIZES`. |
| no silent CUDA fallback | **OK.** The diff touches no device selection. `DMCSwimmerSwimmer6` still raises `ValueError` on a CPU request on both trees rather than falling back. |
| no widening of `check_state_dims_matter` | **OK.** `stable_envs/testing/` is not in the diff. The failure is recorded, and the proposed contract change is written down for approval rather than applied. |
| no Gate-A `partial` misuse | **OK.** No `partial` anywhere in the source diff; no `certification.json` row added. |
| frozen step4/5/6 corpora untouched | **OK** (see §6). |

---

# Required changes (both small; neither invalidates a measurement)

## C1 — `_movers` does not handle a `body` episode slot; this patch is what makes that reachable

`stable_envs/envs/dmcontrol.py:1524-1527` (unchanged by the patch, but newly live because of it):

```python
for arr, i in self._slots():
    kind, eid = _slot_element(m, arr, i)
    (ep_g if kind == "geom" else ep_s).add(eid)     # kind == "body" falls into the SITE set
```

`_slot_element` returns `("body", 11)` for manipulator's four slots, so a **BODY id is inserted into
the SITE id set**. Measured:

```
ep_g = set()          ep_s = {11}          # 11 is a BODY id, and nsite is 9
```

It is inert here only by arithmetic accident (`nsite = 9 < 11`, and manipulator has no drawn sites at
all). No pre-existing cell could hit it because no pre-existing cell had a `body_*` slot; this cell is
the first. The next domain down the list is `stacker`, which the patch's own docs now advertise as
having `manipulator` for a "worked precedent" — and a `stacker` body id that collides with a drawn
world-body site id would silently promote that site to a mover, pulling it out of the plate.

The body slot needs nothing from this rule (every geom and site on a drawn body already satisfies
`bodyid != 0`), so the fix is to stop mis-filing it:

```python
for arr, i in self._slots():
    kind, eid = _slot_element(m, arr, i)
    if kind == "geom":
        ep_g.add(eid)
    elif kind == "site":
        ep_s.add(eid)
    # "body" needs no entry: every geom/site on a drawn body is already off the world body.
    # "light" is not a drawn primitive.
```

with a one-line comment saying why, so the next reader does not "fix" it back.

## C2 — four places name `check_default_style_is_official` for a guarantee that check does not make

`check_default_style_is_official` (`stable_envs/testing/conformance.py:495-507`) compares
`render(states)` against `render(states, style=env.default_style())`. It never touches the official
renderer. It would return `max_abs_diff = 0` for **any** `init_value`, including a wrong one.

So these four claims are wrong as written:

- `dmcontrol.py:293` (`SHAPE_KNOB_SPECS`): *"That is what keeps `check_default_style_is_official`
  a statement about the compiled model"*
- `dmcontrol.py:953` (`SHAPE_KNOBS`): *"that index becomes the leaf's `init_value` — so
  `check_default_style_is_official` is true by construction rather than by a transcribed constant"*
- commit `a635d3a` body: *"exactly one alternative must match the pristine model and that index is
  the `init_value`, so `check_default_style_is_official` holds by construction"*
- `findings_phase1.md:30` (§1): *"`check_default_style_is_official` is therefore true by construction."*

The invariant the `len(cur) != 1` assertion actually protects is **"the default style renders the
compiled model, which is what parity against `suite.load` is measured under"** — enforced by the
parity measurement (`interior_exact_frac = 1.000000` at default style), not by that check. The
assertion is still worth having and the substance is correct; only the named check is wrong. In a
file where comments are the primary contract, name the right thing.

# Recommendations (advisory)

**R1 — the guard's remaining refusal set is arguably still incomplete, by its own reasoning.**
The rewritten message refuses `_dir` because *"the lights are captured once per bake"* (`_Lights` is
built from `r.scene` inside `_bake`). By that identical argument `light_pos` should be refused too —
and it is not. `DMCSwimmerSwimmer6` **already carries two of them**:

```
slots: [('geom_pos',3,'target_pos_x'), ('geom_pos',4,'target_pos_y'),
        ('light_pos',0,'target_light_pos_x'), ('light_pos',1,'target_light_pos_y')]
```

A light is not a mover, so `_fk` writes those two floats into the model per row and nothing reads
them per row — every environment in a batch would be lit from the bake's light position. This is
**pre-existing** (swimmer raises `ValueError` before it can render, on both trees) and out of W3's
scope to fix, but this patch is the one that rewrote exactly this justification, so it should either
refuse `light_*` wholesale or state in the docstring why `light_pos` is exempt. Flagging for the
coordinator as a probable latent defect in the family, independent of this merge.

**R2 — the new capability has no test at N > 1.** What `_hideable` gained is *multi-geom* body
resolution; manipulator's `target_ball` has exactly one geom, and
`test_a_body_slot_becomes_one_visibility_leaf_over_its_geoms` asserts `len(ids) >= 1` and that each id
is named `"target_ball"`. The N>1 branch, and the "hides all of them and only them" property, are
untested. A synthetic test (monkeypatch `_slots` to name a multi-geom body such as `hand` or
`thumbtip`, assert the id set equals that body's drawn geoms exactly) would cover it cheaply. The raise
branch is likewise only reachable synthetically and is untested.

**R3 — one-line guard on the shape leaf's namespace.** `space[gname] = Dict({"shape": …})` places the
geom name at the TOP level of the variation space, after `space["visibility"]`, `space["material"]`,
`space["camera"]`, `space["light"]` are assigned. A model with a geom named `visibility` / `material`
/ `camera` / `light` would silently clobber an entire group and delete leaves. Unlikely, but this is
generic shared machinery and the assert is free.

**R4 — document that `SHAPE_KNOBS` leaves derived model arrays stale.** Writing `m.geom_type[gid]` and
`m.geom_size[gid]` on a compiled `MjModel` does not recompute `geom_rbound` / `geom_aabb`. Harmless
for this cell (the `ghost` class is `contype=0 conaffinity=0`, and `DMControlSuite` implements no
`step`), but the docstring of a generic mechanism should say so, or a future user of `SHAPE_KNOBS` on
a colliding geom will get silently wrong broadphase.

**R5 — fix the candidate corpus before the Step-8 freeze.** Store the SELECTED 64 (Step-4's
`N_STATES`), and use Step-4's `meta` schema (`env`, `selection`, `indices`, `sha256`) so
`corpus_for()` and the certifier read it unchanged. See §6.

**R6 — do not promote 0.9829 to a floor.** My draws hit 0.982834, below the author's 12-draw worst of
0.982859. The declaration is correct as it stands (`parity_interior_min = 1.0`, no
`parity_all_exact_min`); this is a caution for the ≥40-draw re-declaration at the 3.11 pin.

**R7 — the shape leaf's FIDELITY is unmeasured, only its reachability.** We mutate a compiled model;
SWM recompiles the MJCF. That our `.015` box/sphere renders what SWM's recompiled model renders is not
shown. This is the same family as the author's own open O1 (mjcf round-trip signature) and should be
measured in the same pass, not merged as established.

**R8 — say plainly that `floor.color` is a RANGE gap, not a default gap.** SWM's `floor.color` init
already equals the compiled `grid` texture's `rgb1`/`rgb2`, so the SWM *default frame* is reachable
without it. The Gate-F table reads as though a hole exists in the default; it does not.

# What is NOT blocking

- mujoco 3.12.0 / dm_control 1.0.45 rather than the declared 3.11.0 / 1.0.44: **the 3.11 arm is
  unavailable on this machine**, and Oscar SSH is denied by the brief and was not attempted. Every
  number in this review, and in the author's findings, is at the 3.12 pin. The 3.11 re-measurement,
  the ≥40-draw declaration, the CUDA arm, O1 and `tests/gpu/*` all remain genuinely pending, are
  named as pending by the author, and their scripts are committed.
- `check_state_dims_matter` FAIL: correctly recorded, mechanism independently confirmed, check not
  widened, no `partial` used. The proposed validated-exclusion contract is Sami's to accept or refuse.

# Verdict

**APPROVE-WITH-CHANGES.** Apply **C1** and **C2**. Everything measurable locally reproduces: the
per-row orientation claim is true of every consumer, the body->geom resolution is correct and raises
where it says it does, the shape leaf's default is the compiled model and does not leak, the 16
pre-existing cells are byte-identical across five digests each, and parity is
`interior_exact_frac = 1.000000` with `presence_margin` 0.87 / 0.74. The declared failure is real,
bounded and honestly recorded.

# Commands run (reviewer)

```bash
git worktree add --detach .../scratchpad/w3base 76dc5116f432e61dd47fba8f9e879cf9212c8e46
cd <tree> && MUJOCO_GL=glfw $PY -m pytest tests/unit -q -p no:randomly --deselect tests/unit/test_examples.py
cd <tree> && MUJOCO_GL=glfw PYTHONPATH=<tree> $PY .../w3_ident.py --out ident_{base,branch}.json
cd branch && ... $PY .../w3_probe_model.py           # model facts, slots, hideable, leaves
cd branch && ... $PY .../w3_movers.py                # _movers body-slot mis-bucketing (C1)
cd branch && ... $PY .../w3_body.py                  # per-body geom resolution + raise condition
cd branch && ... $PY .../w3_gate.py --size 224 --draws 3 --n 64 --out w3_gate_224.json
cd branch && ... $PY .../w3_gate.py --size  84 --draws 3 --n 64 --out w3_gate_84.json
cd branch && ... $PY .../w3_swmleaf.py               # SWM-default-through-leaves
cd branch && ... $PY .../w3_dims.py                  # dead dims: ours vs official vs shape=0
cd <tree> && ... $PY examples/fidelity_step4.py --shard dmc --device cpu --out replay_{base,branch}.json
```

Artifacts under `/private/tmp/claude-501/-Users-samib-stable-envs/9c7dc93a-.../scratchpad/`:
`w3_gate_224.json`, `w3_gate_84.json`, `ident_base.json`, `ident_branch.json`, `w3_dims.txt`,
`w3_swmleaf.txt`, `w3_unit_{base,branch}.txt`, `replay_{base,branch}.json`.
