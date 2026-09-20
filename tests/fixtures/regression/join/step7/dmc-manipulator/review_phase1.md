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

---
---

# RE-REVIEW — tip `0013050e80a5c6ec9af9a86efbb43c8ff3b7350e` (2026-09-15)

**Verdict: APPROVE-WITH-CHANGES.** Every item the coordinator listed was re-run and reproduced. The
cell-level work is complete and correct. Two **required** hardenings remain, both inside the new
SHARED rule in `stable_envs/testing/conformance.py`, and both demonstrated with a working false
positive rather than argued. One small new regression (D3). Nothing found invalidates a measurement:
the shipped env's signature is honest and complete for everything its own state can reach, and I
verified grant/refuse on all 15 dimensions in both styles.

Commits reviewed since `a635d3a`: `8b39e94`, `4396163`, `0013050`. Worktree read-only throughout;
`git status` clean, HEAD unmoved. Environment unchanged (`/Users/samib/se6-venv`, mujoco 3.12.0 /
dm_control 1.0.45, `MUJOCO_GL=glfw`, CPU). **The 3.11.0/1.0.44 arm and the CUDA arm are cluster-only
and I could not re-run them**; for those I verified only internal consistency of the committed
artifacts and that the declared pins are recorded.

## (1) C1-C4 and R2/R3/R4 — satisfied as specified

| item | verdict |
|---|---|
| **C1** `_movers` body-slot id space | **Done.** Explicit `if kind == "geom" / elif kind == "site"`, body and light skipped with the reason in the docstring. |
| **C2** four `check_default_style_is_official` overclaims | **Done.** Every surviving mention (`dmcontrol.py:295`, `:961`, the `:1253` error message, `findings_phase1.md:33`) now states the check's blind spot rather than claiming it. The `a635d3a` commit body cannot be edited and the correction is recorded in §9. |
| **C3/R5** corpus | **Done, verified below (6).** |
| **C4/R8** float32 + `floor.color` | **Done, and better than my finding.** The author traced the real cause: applied directly, `7/6` writes `light_diffuse = 0.699999988079071`, which *is* `float32(0.7)`, delta 0.0; the `+2.4e-07` I saw comes from `render`'s style-group key `np.round(…, 6)` → scale `1.166667`. Measured 0 px moved. `floor.color` is now stated as a RANGE gap. |
| **R2** multi-geom body tests | **Done.** Two synthetic-model tests: a 4-geom body (2 drawn, 1 alpha-0, 1 group-3) must resolve to exactly the 2 drawn; a body with drawn geoms *and* drawn sites must raise. Both reachable from no registered env, which is why they are synthetic. |
| **R3** group-name collision | **Done.** `make_variation_space` raises if a `SHAPE_KNOBS` geom name collides with an existing variation group. |
| **R4** stale `geom_rbound`/`geom_aabb` | **Done**, with the mechanism (rendering never reads them; the only knobbed geom is `ghost` with `contype = conaffinity = 0`; nothing steps physics). |
| **R1** `light_pos` | Recorded, no action — correct: it is a live shipped cell's behaviour and belongs to Step 8. |
| **R6** 0.9829 | Recorded, not promoted to a floor. The 41-draw cluster arm's worst is **0.98273**, below my 0.982834, which vindicates the caution. `parity_all_exact_min` is `None` on the class. |

## (2) The exclusion rule — minimal, but TWO demonstrated false positives

What is right, verified by running `_validate_render_invariant` on **all 15** dimensions:

```
arm_root/arm_shoulder/arm_elbow/arm_wrist/thumb/thumbtip/finger/fingertip,
ball_x, ball_z, target_ball_pos_x, target_ball_pos_z   -> REFUSED ("changed geom_xpos")
ball_y, target_ball_quat_qw, target_ball_quat_qy       -> GRANTED ("rotates only the sphere geom(s)")
```

- **Name alone excludes nothing**: emptying `RENDER_INVARIANT_DIMS` returns the three to
  `dead_dims` with no `excluded_dims` key (run below).
- **Reported visibly**: `check_state_dims_matter` returns `excluded_dims` as `{name: full reason}`,
  and `fidelity_step4.run()` puts the whole dict under `evidence`, so it reaches the Gate-A record
  verbatim. `gate.py` has no summary that projects only `dead_dims` (its `is_excluded` is about check
  *names*). One reporting caveat: because `dead_dims` is now `[]`, a downstream consumer that judges
  purely on `dead_dims` sees a plain pass — the evidence is present but a terse summary could elide
  it. Worth a line in the certifier's renderer, not a defect in the rule.
- **Undeclared envs untouched**: confirmed end-to-end by the frozen replay — see (5).

### D1 (REQUIRED) — a frozen field the env does not emit is SILENTLY SKIPPED

`_validate_render_invariant` iterates `for k in _RENDER_INVARIANT_FROZEN: if k not in a: continue`.
A signature that omits a field is not an error; that field is simply never compared. So the
enumeration the docstring calls a bound ("bounded by an enumeration rather than by a promise") is
advisory. **This code path is already live**: the shipped env omits `mat_texid`, which is therefore
unchecked today.

Demonstrated false positive (`w3_attack.py`). `thumbtip` rotates *only* the two `fingertip`-class
SPHERES (`thumbtip1`, `thumbtip2`) and also translates them, so `geom_xpos` is the single field
standing between it and a grant:

```
honest signature   -> granted = False | "...changed geom_xpos, which a claim of sphere-orientation-only..."
geom_xpos omitted  -> granted = True  | "...rotates only the sphere geom(s) ['thumbtip1','thumbtip2']..."
thumbtip moves 60 px (max 206/255)    -> that grant is WRONG
```

**Fix:** treat the frozen set as required. Refuse with a reason naming the missing key, e.g.

```python
missing = [k for k in _RENDER_INVARIANT_FROZEN if k not in a]
if missing:
    return False, (f"render_scene_signature omits {missing}, which the render-invariance rule must "
                   f"compare; an omitted field is not an unchanged one")
```

and either emit `mat_texid` from `DMControlSuite.render_scene_signature` or move it to an explicit,
documented `_RENDER_INVARIANT_OPTIONAL` tuple. Silence is the wrong default for a rule whose whole
purpose is to be un-gameable.

### D2 (REQUIRED) — `geom_is_sphere` is supplied by the env being judged and never cross-checked

The single most load-bearing predicate in the rule is read straight from the signature:
`sphere = np.asarray(a.get("geom_is_sphere"), bool)`. Conformance already receives and compares
`geom_type`, but never checks that the two agree. So the check does not, in fact, own the definition
of "sphere" — the env does.

Demonstrated false positive, with `geom_type` left entirely honest:

```
honest signature, BOX target        -> granted = False | "...rotated NON-sphere geoms ['target_ball']..."
geom_is_sphere = all True, BOX      -> granted = True  | "...rotates only the sphere geom(s) ['target_ball']..."
```

This matters beyond a lying env: it is a plausible *bug* surface for the next author of a signature,
and W4's quadruped brings `ELLIPSOID`, which is emphatically not rotation-invariant while being the
most sphere-like thing in the suite.

**Fix (one line, and it makes the design claim literally true):** derive the predicate in
conformance from the field it already freezes —

```python
sphere = np.asarray(a["geom_type"]) == SPHERE_TYPE      # and drop geom_is_sphere from the contract
```

or, if the env is to keep supplying it, refuse on disagreement:

```python
if not np.array_equal(sphere, np.asarray(a["geom_type"]) == SPHERE_TYPE):
    return False, "geom_is_sphere disagrees with geom_type; the rule defines sphere, not the env"
```

### Signature completeness against `_prims` / `_static_t` / the plate — as asked

I walked every render-side reader against the signature's fields.

| reader | fields | covered? |
|---|---|---|
| `_prims` (movers and statics) | `geom_type`, `geom_size`, `d.geom_xpos`, `d.geom_xmat`, `geom_matid`→`mat_rgba`/`geom_rgba`, `mat_specular`, `mat_shininess`; same for sites | **yes** — geoms and sites are concatenated into one primitive list, which is the right call and fixes the group-3-site false refusal the docstring records |
| `_static_t` | the same static prims | **yes** |
| `sc.floor` | reflective plane `geom_size`/`xpos`/`xmat`, `mat_reflectance` | **yes** (it is a drawn geom) |
| `_Lights` | light pos/dir/diffuse/specular/ambient/castshadow | **yes** |
| plate bake (MuJoCo) | the above plus textures, `m.vis.headlight`, `geom_group`/`site_group`, skybox | textures/`m.vis` are **not** in the signature but are **not writable from a state dimension**, so unreachable; a group change alters the drawn set and is caught because `geom_names` differs and `geom_names` *is* frozen |
| camera | `cam_pos`, `cam_quat`, `cam_fovy` (MODEL arrays) | **yes for a fixed camera**, which is every `DMControlSuite` cell |

**Two gaps that are unreachable today and become live for a future cell** (note, not a defect now):

- **N1 — tendon.** `_fk` builds the tendon's capsule chain and colour from `mj_forward`; nothing about
  it is in the signature. `ball_in_cup` declares no candidates, so it cannot be hit — but a tendon
  domain that ever declares one would have its tendon invisible to the rule.
- **N2 — resolved camera pose.** Only the model `cam_pos`/`cam_quat` are compared, not `d.cam_xpos`/
  `d.cam_xmat`. Harmless while every cell using this is `mode="fixed"` on the world body; a
  body-mounted or `trackcom` camera moves with `qpos` and the rule would not see it.

Cheapest durable fix for both: have `_validate_render_invariant` refuse outright when the env's model
carries a tendon or a non-fixed camera, unless the signature reports them.

**N3 (minor) — the validation perturbation is not the one that proved deadness.** Deadness is decided
by donor substitution across the whole batch plus `±25 / 1.0 / 0.5` nudges; the validation then applies
`+0.2` to **row 0 only** (`states[:1]`). The two never have to agree. It is benign here (validation is
only reached once a dim is dead under every one of those), but validating with the same perturbations
that were tried would remove the gap for free.

## (3) Regression, both directions — reproduced independently

`w3_both.py`, 16 states at seed 8181 (mine, not the author's):

```
224 px SPHERE   ball_y 0 px  quat_qw 0 px  quat_qy 0 px      all GRANTED
224 px BOX      ball_y 0 px GRANTED | quat_qw 221 px max 151 REFUSED | quat_qy 221 px max 151 REFUSED
 84 px SPHERE   ball_y 0 px  quat_qw 0 px  quat_qy 0 px      all GRANTED
 84 px BOX      ball_y 0 px GRANTED | quat_qw  27 px max 145 REFUSED | quat_qy  27 px max 145 REFUSED
negatives       arm_root / ball_x / target_ball_pos_x -> all REFUSED ("changed geom_xpos")
declared        {'dead_dims': [], 'excluded_dims': {ball_y, target_ball_quat_qw, target_ball_quat_qy}}
emptied         {'dead_dims': ['ball_y','target_ball_quat_qw','target_ball_quat_qy']}   # no excluded_dims key
```

Author reported 201 px / max 145 at 224 and 28 px at 84; I get 221 / 151 and 27 on a different draw —
same magnitude, same verdict. The refusal reason is the `NON-sphere` branch in both cases, and
`ball_y` correctly stays granted under the box because it orients the `ball` prop, which
`target_ball.shape` does not switch.

## (4) Order-dependence fix does not alter `render()`

`w3_order.py`, two processes. The `dirty` one runs, before every measurement: a BOX render (leaving a
box scene cached and on the shared model), a full `check_state_dims_matter` (15 signature probes, each
mutating the shared `MjModel` and clearing `_SCENE_CACHE`), a bare `render_scene_signature` under the
box variation, a hidden-target render, and a direct `_validate_render_invariant` under the box.

```
                       clean              dirty
default_ss1            1dc63a62c3b3e10f   1dc63a62c3b3e10f   OK
box_ss1                41a62d1b75c1dfeb   41a62d1b75c1dfeb   OK
default_ss2            f9d288ae183345f7   f9d288ae183345f7   OK   <- forces a fresh _static_t
default_ss1_again      1dc63a62c3b3e10f   1dc63a62c3b3e10f   OK
hidden_ss2             d51f0fbdf7c32eb1   d51f0fbdf7c32eb1   OK
size84                 593c670741760e5c   593c670741760e5c   OK
                       + post-probe re-renders at ss1 and ss2: identical again
DIFFERING: none
```

The `_apply_variation(m, pr, None)` + `_SCENE_CACHE.clear()` pair in the `finally` restores the
pre-existing invariant exactly. Note for the record that clearing a module-global cache is a real
side-effect of a conformance probe — it drops every *other* env's baked scenes too — but it is a pure
memo, so the cost is time, not pixels.

## (5) Regression suite

- **Frozen Step-4 `dmc` shard, `76dc511` vs `0013050`, both run by me on this machine:
  14 configs, 0 differing** (after dropping only `seconds` and absolute-path `traceback` strings) —
  `digest_ours`, `digest_official`, `digest_ours_default_style`, `digest_ours_corpus_style`,
  `mean_pixel`, `distinct_colors` all equal. And the shared change is provably inert for undeclared
  envs: their payload is still exactly `{"dead_dims": []}` with no new keys.
- **All 17 cells byte-identical across four digests.** 16 pre-existing cells vs `76dc511`: 0
  differing on `size`/`state_dim`/`state_names`/`leaves`/`_slots`/`_hideable`/every `init_value`/
  `states_sha`/default digest/two moved-style digests/an 84-px digest. And
  `DMCManipulatorBringBall` is **identical between `a635d3a` and `0013050`** on all of those — the
  three follow-up commits moved no pixel anywhere.
- **`tests/unit`** (deselecting `test_examples.py`): **1 failed, 726 passed, 6 skipped** vs baseline
  **1 failed, 713 passed** — +13 new tests, same single pre-existing failure. **But see D3.**
- **No other check or threshold touched**: the only hunk in `stable_envs/testing/` is inside
  `check_state_dims_matter` plus the two new module-level helpers; `dead_dims` keeps its meaning and
  the return dict gains keys only when non-empty.

### D3 (new, small) — `test_every_cited_scratch_path_resolves`'s broken-path set GREW by 2

The test was already red on the baseline, so the headline "same failure" is true — but its content is
not the same. At `a635d3a` the set was **identical** to `76dc511` (32 paths, `diff` empty). At
`0013050` it is **34**, and both additions come from `4396163`:

```
+ examples/results/step7/dmc-manipulator/profile/DMCManipulatorBringBall_cuda_gpu2104.md:6
+ scratch/sboughan/se7/dmc-manipulator/stable_envs/__init__.py
```

from that note's provenance lines:

```
line  6: - tree measured: `/oscar/scratch/sboughan/se7/dmc-manipulator/stable_envs/__init__.py`
line 10: - measured tree: `$SCR/se7/dmc-manipulator/stable_envs/__init__.py`
```

It mirrors an existing wart in the Step-6 profile notes, so it is not a new *class* of problem, and it
changes no measurement. But the previous tip added none, and a repo-contract test whose failure list
grows silently is how that test stops being read. **Fix:** in the committed note, render the measured
tree as plain text rather than a backticked path (or drop the duplicate line 10), so the contract test
does not treat it as a citation. Coordinator's call whether to hold the merge for it.

## (6) Corpus

```
states: (64, 15) float32                       # the SELECTION, not the pool
meta keys ⊇ Step-4's exactly: env, indices, kept, n_extreme, n_spread, n_typical, pool,
           seed, selection, sha256, state_dim, state_names  (+ provenance extras)
seed 20260909 == fidelity_step4.STATE_SEED     pool 256 == POOL     kept 64 == N_STATES
declared sha256 48db2b1adf54a75d816eb068f815257ffa7e1ab238218c4f9e64568873b749d8
recomputed      48db2b1adf54a75d816eb068f815257ffa7e1ab238218c4f9e64568873b749d8   MATCH
indices: 64 unique, all < 256
corpus_for(env, "DMCManipulatorBringBall", …) -> (64, 15)
pool kept aside at corpus/pool/DMCManipulatorBringBall_pool.npz  (256, 15)
```

Generated at the declared pin (`meta` records mujoco 3.11.0 / dm_control 1.0.44, node2306). Exactly as
specified.

## (7) Hygiene

```
trailers 76dc5116..HEAD                     -> 0
authors                                     -> sami-bg <samibghanem@gmail.com>
examples/results/step4|5|6 changed           -> (none)
raycast3d / render/ changed                  -> (none)
DMControlTrackCom region                     -> untouched (context lines only)
git status in the worktree                   -> clean, HEAD unmoved
```

## Cluster artifacts — consistency only (cannot re-run locally)

- **41-draw declaration arm** at mujoco 3.11.0 / dm_control 1.0.44: 41 draws at 224, `interior_exact_frac`
  1.0 on every one, **0 bad of 56,575,531**; worst all-pixel 0.98273. 84 px likewise 41 draws, interior
  1.0, 0 bad, worst 0.97745. `declaration` records `parity_interior_min = 1.0`, no all-exact floor.
- **CUDA identity**: 16 common cells, 0 differing; 1 new. Consistent with the claim.
- **O1** — I checked this rather than taking "benign". The diff's `counts` and `visual` are **empty**,
  and the `episode_draw` census is **identical on both sides**
  (`body_pos[33]`, `body_pos[35]`, `body_quat[44]`, `body_quat[46]`), which is the load-bearing
  question. The `geoms`/`sites`/`materials`/`textures` entries are an asset-**table rotation**, not a
  scene difference: resolving by name, **every geom's effective material name is identical**, every
  material's rgba is identical except `target` (alpha 0 = SWM's `render_target = 0` default), every
  texture sha matches by name (`skybox` f6a21dd7, `grid` d6efb069, `background` 1217cedb), and the one
  light differs by diffuse 0.6 vs 0.7 = SWM's `light.intensity` default. All three are exactly the
  leaves already declared. O1 is benign, and now for a stated reason.

## Re-review verdict

**APPROVE-WITH-CHANGES.** Required: **D1** and **D2** (both in `_validate_render_invariant`, both
demonstrated, both a few lines). **D3** is small and the coordinator's call. **N1/N2/N3** are notes for
whoever declares the second cell. Everything else on the list is verified done and reproduced:
C1-C4, R2-R4, the exclusion rule's minimality and reporting, both regression directions at both
sizes, order-independence byte-identity, 14/14 replay configs, 17/17 cells across four digests, the
corpus, and hygiene.

### Re-review commands

```bash
git worktree add --detach .../scratchpad/w3base2 76dc5116
cd <tree> && MUJOCO_GL=glfw PYTHONPATH=<tree> $PY examples/fidelity_step4.py --shard dmc --device cpu --out replay_{base,branch2}.json
cd tip   && MUJOCO_GL=glfw $PY -m pytest tests/unit -q -p no:randomly --deselect tests/unit/test_examples.py
cd tip   && ... $PY .../w3_ident.py   --out ident_tip.json      # 17 cells x 4 digests
cd tip   && ... $PY .../w3_attack.py                            # D1, D2 false positives
cd tip   && ... $PY .../w3_both.py                              # granted/refused both directions
cd tip   && ... $PY .../w3_order.py {clean,dirty}               # order-independence
cd tip   && ... $PY .../w3_corpus.py                            # corpus_for + sha256
```

Artifacts: `w3_attack.txt`, `w3_both.txt`, `w3_order_{clean,dirty}.json`, `ident_tip.json`,
`replay_branch2.json`, `w3_unit_branch2.txt`, `paths_branch2.txt`.

---
---

# FINAL VERIFICATION — tip `d1f65d4c6107f9013f2b467fb8e3ec5c9c139b6d` (2026-09-15)

**Verdict: APPROVE.**

Both false positives I built against `0013050` are now refused, with the stated reasons, and both are
regression tests in the repo. The broken-citation set is identical to the baseline's by the contract
test's own machinery. The conformance diff is limited to the rule. No defects remain.

Worktree read-only; `git status` clean, HEAD unmoved. `/Users/samib/se6-venv`, mujoco 3.12.0 /
dm_control 1.0.45, `MUJOCO_GL=glfw`, CPU. **The 3.11.0/1.0.44 and CUDA arms remain cluster-only and
were not re-run here.** Hygiene re-checked over `76dc5116..d1f65d4`: 0 trailers, author `sami-bg`,
nothing under `examples/results/step4|5|6`, nothing in `render/` or `raycast3d.py`.

## D1 — closed

`_RENDER_INVARIANT_OPTIONAL` exists and is `{}`; `FROZEN fields NOT emitted by the env` is now `[]`
(`mat_texid` is emitted). Re-running my construction (`w3_attack2.py`, the real env's signature
wrapped):

```
honest signature, thumbtip      -> REFUSED: "...changed geom_xpos, which a claim of sphere-orientation-only invariance forbids"
geom_xpos POPPED, thumbtip      -> REFUSED: "render_scene_signature does not emit ['geom_xpos'], and the frozen
                                             enumeration is the whole bound on this exclusion..."   [names the key: True]
```

And every other frozen key behaves the same way — I popped five at random and each produced a refusal
naming itself: `geom_type`, `mat_texid`, `light_castshadow`, `cam_fovy`, `geom_names`.

## D2 — closed

`_MJGEOM_SPHERE = 2`, which I confirmed equals `mujoco.mjtGeom.mjGEOM_SPHERE`. Sphere-ness is derived
from `geom_type`; an emitted `geom_is_sphere` must agree.

```
honest signature, quat_qw under BOX     -> REFUSED: "...rotated NON-sphere geoms ['target_ball']..."
geom_is_sphere = all True, BOX          -> REFUSED: "render_scene_signature's geom_is_sphere disagrees with the
                                                     geom_type it reports..."   [disagreement branch: True]
geom_is_sphere POPPED, quat_qw / SPHERE -> GRANTED   (derivation works with no flag at all)
geom_is_sphere POPPED, quat_qw / BOX    -> REFUSED   (and still refuses correctly)
```

The `mjGEOM_ELLIPSOID` note is on the constant, which is the right place for it — that is the kind the
next author is most likely to get wrong, and W4 has now put it in the registry.

**One thing I could not break, and it is not a defect.** I also forged `geom_type` itself to all-SPHERE
in *both* states, which is internally consistent and therefore granted a BOX target. That is the
irreducible trust floor: every field is env-supplied, so a fabricated primary model fact defeats any
rule that reads the signature rather than the model. The guarantee the rule now makes — *given a
truthful signature, nothing that moves anything but sphere orientation can be excluded* — holds, and
D2 was precisely about the rule no longer accepting a **second, redundant** assertion that contradicts
the primary one. Recording it so nobody re-derives it as a finding later.

## D3 — closed, and my earlier count corrected

I measured this the authoritative way this time, importing the contract test's own `CITATIONS` and
`_expand` in each tree rather than scraping the assertion text:

```
BASELINE 76dc511 : citations 304   broken 9
TIP      d1f65d4 : citations 306   broken 9
SETS IDENTICAL   : True      only-on-tip: []      only-on-baseline: []
  scratch/carracing/logs/            scratch/robocasa/out/corpus
  scratch/sboughan/                  scratch/sboughan/se6/baseline/stable_envs/__init__.py
  scratch/sboughan/se6/gfxdraw       scratch/sboughan/se6/tilemap/stable_envs/__init__.py
  scratch/sboughan/venvs/se-step6/bin/python
  scratch/sboughan/venvs/se6-gfxdraw-box2d/bin/python
  scratch/step6/gfxdraw/cost_floor.{py,cpp
```

The tip adds two citations and neither is broken; no `se7` entry remains. **Correction to my
re-review:** my 32/34 figures came from a regex over the assertion text and conflated broken citations
with the *sites* that cite them. The real delta at `0013050` was one new broken citation
(`scratch/sboughan/se7/dmc-manipulator/stable_envs/__init__.py`, matched out of the absolute Oscar path
on line 6 of the CUDA profile note), not two. The finding was real; the count was mine to correct.

## N1 / N2 / N3

N1 (spatial tendon geometry) and N2 (non-fixed camera pose) are now written into
`_validate_render_invariant`'s docstring, stated as out of reach of every dimension any current cell
declares and as things that must be added before one that is not. That is the right disposition. N3
(validation perturbs `+0.2` on row 0 while deadness is proved by donor substitution over the batch) is
left alone as benign — I agree.

## Scope of the shared change

I checked this mechanically rather than by eye: mapping every changed line range in
`stable_envs/testing/conformance.py` against the file's top-level AST spans over `76dc5116..HEAD`
gives exactly

```
check_state_dims_matter, _render_invariant_candidates, _validate_render_invariant,
and the module-level constant block between them (_RENDER_INVARIANT_FROZEN /
_RENDER_INVARIANT_OPTIONAL / _MJGEOM_SPHERE, all new)
```

No other check, no threshold, no existing constant. **The conformance diff is limited to the rule.**

## Re-verification

- **Frozen Step-4 `dmc` shard**, `76dc511` vs `d1f65d4`, both run by me: **14 configs, 0 differing.** After dropping only
  wall-clock `seconds` and absolute-path `traceback` strings, the two JSON documents compare equal on
  every config; no config has a differing `digest_ours` or `digest_official`. And the shared rule is
  provably inert for undeclared envs: their `check_state_dims_matter` payload is still exactly
  `{"dead_dims": []}` with no new keys (checked on `DMCReacherHard`, `DMCBallInCupCatch`,
  `DMCCheetahRun`).

- **The frozen enumeration was not weakened to make "nothing missing" true.** I checked the obvious
  cheat: `_RENDER_INVARIANT_FROZEN` is byte-identical between `0013050` and `d1f65d4` and still holds
  all 20 fields including `mat_texid`. The gap was closed by the ENV emitting the field, which is the
  correct direction.
- **17 cells byte-identical across four digests.** 16 pre-existing vs `76dc511`: **0 differing** on
  `size`/`state_dim`/`state_names`/`leaves`/`_slots`/`_hideable`/every `init_value`/`states_sha`/
  default digest/two moved-style digests/an 84-px digest. And **all 17 cells are identical between
  `0013050` and `d1f65d4`** — the hardening moved no pixel.
- **`tests/unit`** (deselecting `test_examples.py`): **1 failed, 729 passed, 6 skipped** vs baseline
  **1 failed, 713 passed** — +16 tests, same single pre-existing failure, whose broken set is the
  identical 9 above. (The author's "839 passed, 2 failures" is the same run with `test_examples.py`
  included.)
- **Both directions, my own seed 8181, on the final tip** — unchanged from `0013050`:

```
224 SPHERE  ball_y/qw/qy  0 px            all GRANTED
224 BOX     ball_y 0 px GRANTED | qw 221 px max 151 REFUSED | qy 221 px max 151 REFUSED
 84 SPHERE  ball_y/qw/qy  0 px            all GRANTED
 84 BOX     ball_y 0 px GRANTED | qw  27 px max 145 REFUSED | qy  27 px max 145 REFUSED
negatives   arm_root / ball_x / target_ball_pos_x  -> all REFUSED ("changed geom_xpos")
declared    {'dead_dims': [], 'excluded_dims': {ball_y, target_ball_quat_qw, target_ball_quat_qy}}
emptied     {'dead_dims': ['ball_y','target_ball_quat_qw','target_ball_quat_qy']}   # no excluded_dims
```

(Author reports 201 px / 28 px under the box on their draw; mine is 221 / 27 on a different seed —
same magnitude, same verdict, same refusal branch.)

- **The two false positives are now regression tests**, built by wrapping the real env's signature —
  `test_a_missing_frozen_field_is_refused_and_named_not_skipped`,
  `test_a_lying_sphere_flag_cannot_grant_an_exclusion`,
  `test_the_rule_derives_sphereness_even_when_the_env_omits_the_flag` — and the first carries a guard
  (`assert not np.array_equal(base, moved), "thumbtip draws nothing; this test has lost its subject"`)
  so it cannot quietly stop testing anything.

## FINAL VERDICT: APPROVE

No remaining defects. Every claim in the request was re-run and reproduced. The two demonstrated false
positives are closed at the rule and pinned by tests; the citation regression is closed and verified
against a reconstructed baseline worktree; the shared change is confined to `check_state_dims_matter`
and its two helpers; and nothing outside the new cell has moved a pixel across the whole of
`76dc5116..d1f65d4`.

Standing caveats, unchanged and already declared by the author: the 3.11.0/1.0.44 declaration arm, the
CUDA identity/tier arm and O1 are cluster-measured and I verified only their internal consistency; the
signature does not yet describe tendon geometry or a non-fixed camera pose (N1/N2), which the next cell
to declare a candidate must add first.

### Final-verification commands

```bash
git worktree add --detach .../scratchpad/w3base3 76dc5116
cd tip && MUJOCO_GL=glfw PYTHONPATH=tip $PY .../w3_attack2.py     # D1/D2 re-run against the hardened rule
for T in w3base3 tip; do ... $PY .../w3_cites.py $T; done          # CITATIONS/_expand, both trees
cd <tree> && ... $PY examples/fidelity_step4.py --shard dmc --device cpu --out replay_{base,tip3}.json
cd tip && ... $PY .../w3_ident.py --out ident_tip3.json            # 17 cells x 4 digests
cd tip && ... $PY .../w3_both.py                                   # both directions, both sizes
cd tip && MUJOCO_GL=glfw $PY -m pytest tests/unit -q -p no:randomly --deselect tests/unit/test_examples.py
```

Artifacts: `w3_attack2.txt`, `cites_w3base3.json`, `cites_stable-envs-step7-dmc-manipulator.json`,
`replay_tip3.json`, `ident_tip3.json`, `w3_both3.txt`, `w3_unit_tip3.txt`.

## One housekeeping observation (not a defect, not mine)

The worktree carries **two untracked files** I did not create and did not touch:

```
?? examples/results/step7/dmc-manipulator/replay/dmc_cpu_oscar311_BASE.json
?? examples/results/step7/dmc-manipulator/replay/dmc_cpu_oscar311_HEAD.json
```

They are the output of the committed `scratch/step7/dmc-manipulator/oscar_paired_replay311.sbatch`,
rsynced back from the cluster and left uncommitted while `replay/dmc_cpu_oscar311.json` *is* committed.
They are also useful evidence — reading them confirms a cluster claim I could not otherwise check:
**14 configs each, 0 differing `digest_ours` between BASE and HEAD at the 3.11.0/1.0.44 pin.**
The coordinator should decide whether to commit them as the paired-replay evidence or drop them; left
as-is they will dirty the integration merge. Working tree otherwise clean, HEAD unmoved at `d1f65d4`.
