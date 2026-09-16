# Step 8 Workstream A — VariationMaps for the classic-control and DM Control families

**Baseline** `43ac95238f7472137619f3e69e7636e11dd088b1` (the accepted Step-7 milestone).
**Branch** `step8/variations-dmc-control`.
**Measured against** `galilai-group/stable-worldmodel` at `176d545`, installed editable into
`/Users/samib/se8-vm-venv` (python 3.12, torch 2.14.0 CPU, mujoco 3.12.0, dm_control 1.0.45,
gymnasium 1.3.0, numpy 2.5.3, `MUJOCO_GL=glfw`) — the same pins as `/Users/samib/se6-venv`, which was
not modified.

Seventeen `stable_worldmodel` ids, eighteen stable-envs cells (the two cartpole tasks are one id),
**161 variation leaves**. Every leaf is now exactly one of four things, and every classification has a
measurement behind it:

| class | leaves | what it means |
|---|---:|---|
| `mapped` | **46** | reachable through a stable-envs leaf |
| `unsupported` | **94** | a real upstream effect this cell cannot express |
| `dead_upstream` | **21** | reaches no pixel and no state in `stable_worldmodel` itself |
| `not_applicable` | **0** | reserved by the taxonomy and unused here — see §1 |

Machine-readable, one row per (id, leaf):
[`examples/results/step8/variations-dmc-control/leaf_classification.json`](../../examples/results/step8/variations-dmc-control/leaf_classification.json).
The raw probe outputs it is built from are under
[`examples/results/step8/variations-dmc-control/measure/`](../../examples/results/step8/variations-dmc-control/measure/)
and the probes themselves under `scratch/step8/variations-dmc-control/`.

---

## 1. Why no leaf here is `not_applicable`

Plan §9's taxonomy offers four labels. This family uses three, and the reason is a ruling the
coordinator made while this workstream was running: **a physics-only leaf of a render-from-state cell
is `unsupported`-by-that-cell, not `not_applicable`.** It asks something real of stable-envs — a
dynamics surface — and `not_applicable` would say it asks nothing.

So the 94 `unsupported` leaves are of three kinds, and the reason text on every one of them says
which:

| kind | leaves | example |
|---|---:|---|
| appearance the cell has no knob for | 11 | `floor.color`, `WalkerDMControl.floor.rotation_y`, `BallInCup.ball.size`, `Finger.spinner.color` |
| appearance that arrives through the physics | 13 | the ten `trackcom` mass leaves (§4) and ball-in-cup's three gravity leaves |
| physics-only, measured at 0 px | 70 | every density/mass/friction/gravity leaf of a non-`trackcom` DM Control cell, and 21 of the 24 classic-control `physics.*` leaves |

The two *sources* of "physics-only" are different and both are named in the reason. The five
classic-control cells **ship a batched simulator** certified against the official env to 1e-13 over
100 steps, so `physics.gravity` is a feature they could carry and do not. The thirteen DM Control
cells **ship a renderer over states and no simulator at all** (`hasattr(env, 'step')` is `False` on
every one), so `agent.torso_density` asks them for something that does not exist. Neither is dead: the
swm wrapper really does write the value, the compiled model really does change, and the trajectory
really would differ. Only the *frame* is unaffected, and that is measured rather than assumed.

## 2. The classic-control families: an exact-match recolor is not the same picture

`stable_worldmodel.envs.gymnasium_control.base.GymControlWrapper.render` renders the **official**
gymnasium frame and then swaps every pixel **exactly equal** to a source RGB for the target colour.
stable-envs draws the primitive **in** the colour. The two agree everywhere the reference wrote the
source colour verbatim, and disagree on the anti-aliased boundary, where swm keeps a blend of the
ORIGINAL colour and we produce a blend of the new one.

Measured over 8 states per env, one leaf at a time at 90 % of its range
(`measure/classic_measure.json`):

| env | frame | registered default | one leaf at a probe |
|---|---|---|---|
| Acrobot | 8 x 500 x 500 | **0 px** | 2 - 2,725 px (<= 0.14 %) |
| CartPole | 8 x 400 x 600 | **0 px** | 64 - 957 px (<= 0.05 %) |
| Pendulum | 8 x 500 x 500 | **0 px** | 64 - 2,726 px (<= 0.14 %) |
| MountainCar | 8 x 400 x 600 | **0 px** | 11,033 - 11,326 px (0.59 %) |
| MountainCarContinuous | 8 x 400 x 600 | **0 px** | 11,031 - 11,325 px (0.59 %) |

**The registered default is byte-identical on all five**, which is what Step-8 closure asks for, and
it is asserted by `test_classic_default_matches_the_official_swm_render` against swm's own frame
rather than against ours. The divergence is confined to non-default styles and to shape boundaries.
It is recorded rather than removed: removing it would mean rendering the default frame and then
recolouring it, which is a renderer change and a worse picture.

### Two arithmetic details that a name-matched map would get wrong

**The quantisation.** swm's colours are floats in [0, 1]; ours are 0-255 channels, and
`gfxdraw.to_u8` **truncates**, while swm writes `uint8(v * 255 + 0.5)`. A bare factor of 255 turns
swm's 0.3 (= 77) into 76. The new `VariationMap.rgb255` field carries swm's own rounding, and
`test_rgb255_is_swm_quantisation` checks it over 513 values rather than at the endpoints, because the
disagreement is at the half-integers.

**One swm leaf drives several of ours.** swm swaps a source colour *wherever it occurs*, and black
occurs on more than one element. `MountainCar.visuals.fg` is the car body, the 99 terrain segments
**and** the flag pole; `CartPole.visuals.cart` is the cart **and** the track. stable-envs holds those
as separate leaves, so one swm leaf drives three of ours. A map that moved only `car.color` would
leave the terrain black.

### What is unsupported, and it is not all dynamics

Three classic-control `physics.*` leaves are **drawn**, not merely simulated:

| leaf | official pixels moved (8 states) | mechanism |
|---|---:|---|
| `Acrobot.physics.link_length_1` | 37,966 / 2,000,000 | `AcrobotEnv.render` draws each link at `LINK_LENGTH_1 * scale` |
| `Acrobot.physics.link_length_2` | 38,258 / 2,000,000 | the same, second link |
| `CartPole.physics.length` | 6,621 / 1,920,000 | `polelen = scale * 2 * length` |

`SCALE`, `POLELEN` and the unit-length link quad are class constants on the stable-envs cells (they
are also what the `WIN_*`/`KMAX_*` raster bounds are derived from), and no cell declares a geometry
leaf. The other 21 `physics.*` leaves move **0 pixels** and are unsupported as dynamics.

---

## 3. Twelve DM Control leaves are dead in `stable_worldmodel` itself

Probed by hashing **every** `numpy` array on the compiled `MjModel` (110+ arrays, plus `opt.gravity`
and the size scalars) at the leaf's default and at an interior probe, through
`reset(options={'variation_values': ...})` on the live wrapper (`measure/model_reach.json`). Sixteen
leaves change **not one array**. Two mechanisms, both dead:

**(a) the wrapper never reads it** — 0 textual references outside the `variation_space` literal:

| leaf | |
|---|---|
| `AcrobotDMControl.agent.upper_arm_locked` | all seven `*_locked` leaves are declared and never read |
| `CheetahDMControl.agent.back_foot_locked` | |
| `HopperDMControl.agent.foot_locked` | |
| `HumanoidDMControl.agent.left_knee_locked` | |
| `QuadrupedDMControl.agent.knee_back_left_locked` | |
| `ReacherDMControl.agent.finger_locked` | |
| `WalkerDMControl.agent.right_knee_locked` | |
| `BallInCupDMControl.ball.color` | |
| `CartpoleDMControl.agent.cart_shape` | |
| `PendulumDMControl.agent.mass_shape` | |
| `ManipulatorDMControl.agent.upper_arm_length` | the swm source carries `# TODO should modify length of arm` at the point it would be read; found in Step 7, re-measured here |

**(b) MuJoCo ignores the write.** The wrapper sets the geom's `density` and recompiles, but the geom
carries an **explicit mass**, and an explicit mass wins over density:

| leaf | the mass that wins |
|---|---|
| `AcrobotDMControl.agent.upper_arm_density` | acrobot's `<default class="/"><geom type="capsule" mass="1"/>` — every geom |
| `AcrobotDMControl.agent.lower_arm_density` | the same |
| `CartpoleDMControl.agent.pole_density` | cartpole's `pole` default class |
| `PendulumDMControl.agent.mass_density` | `<geom name="mass" ... mass="1"/>` |
| `PendulumDMControl.agent.pole_density` | `<geom name="pole" ... mass="0"/>` |

Verified end to end on acrobot: the exported XML really does carry `density="1450"`, the wrapper
really does recompile (`_dirty` goes back to `False`), and `body_mass` stays `[0.0, 1.0, 1.0]`.

**Five more leaves reach the model and no pixel**, because the element they reach is never drawn.
`mjvOption`'s default `geomgroup` rasterises groups 0-2 only:

| leaf | group | measured |
|---|---|---|
| `AcrobotDMControl.target.color` / `.shape` | site `target`, **group 3** | 0 px over 8 states |
| `BallInCupDMControl.rendering.render_target` / `target.color` / `.shape` | site `target`, **group 4** | 0 px over 8 states, and 0 px again with `render_target` forced to 1 |

Contrast finger, reacher and manipulator, whose targets are in **group 0** and are hidden by a
material *alpha* instead. There the same leaf moves 150-459 px, and there it is mapped.

> **Upstream findings bundle.** All twelve belong in the Step-8 upstream bundle (Workstream D). None
> of them is a stable-envs defect and none should be implemented: implementing a knob the reference
> does not honour would make stable-envs and `stable_worldmodel` disagree, which is the opposite of
> the mapping's job.

---

## 4. On the five `trackcom` cells, a mass leaf moves pixels at a fixed state

`cheetah`, `walker`, `hopper`, `humanoid` and `quadruped` all render through a `mode="trackcom"`
camera, whose position comes from `subtree_com`. Changing a density changes `body_mass`, which moves
the **camera**, which moves every pixel — with `qpos` and `geom_xpos` verified bit-identical:

| cell | leaf | px moved / 8 states |
|---|---|---:|
| `DMCCheetahRun` | `agent.torso_density` | 2,115 / 56,448 |
| | `agent.back_foot_density` | 4,521 / 56,448 |
| `DMCWalkerWalk` | `agent.torso_density` | 26,328 / 56,448 |
| | `agent.left_foot_density` | 21,374 / 56,448 |
| `DMCHopperHop` | `agent.torso_density` | 10,918 / 56,448 |
| | `agent.foot_density` | 11,167 / 56,448 |
| `DMCHumanoidStand` | `agent.torso_density` | 36,302 / 56,448 |
| | `agent.right_lower_arm_density` | 25,419 / 56,448 |
| `DMCQuadrupedWalk` | `agent.torso_density` | 135,622 / 401,408 |
| | `agent.foot_back_left_density` | 137,533 / 401,408 |

Every non-trackcom DM Control cell's density leaves measure **exactly 0 px**, which is what makes the
mechanism attribution safe rather than a guess. So these ten leaves are **not** "physics with no
pixel": they are appearance-affecting leaves the cell cannot express, and they are classified
`unsupported` with the count. The remaining density/mass/friction leaves on the other cells stay
`not_applicable`.

`BallInCupDMControl`'s gravity is the same shape of surprise for a different reason: the cup and ball
are joined by a **spatial tendon**, which MuJoCo draws as a hanging catenary through `mjv_catenary`,
and `mjv_catenary` reads the model's gravity. At a fixed `qpos`, `gravity.x` moves 178 px and
`gravity.y` 110 px (`gravity.z` alone moves 0, because scaling z does not rotate the gravity
direction). All three are `unsupported`; on every other DM Control cell gravity is `not_applicable` at
a measured 0 px.

---

## 5. `floor.color` — a range gap, measured, and an escalation candidate

Carried forward from the Step-7 manipulator finding and now answered for all thirteen cells with the
same mechanism.

swm writes the builtin `grid` **texture**'s `rgb1`/`rgb2` — a `Box` of shape `(2, 3)`, **six** free
scalars — and recompiles. stable-envs has exactly one floor knob, `material.grid`, a **3**-vector
multiplying the floor material's rgba; and `dmcontrol._tex_rgb` reduces the whole checker to its
**mean** colour, so the checker *pattern* is not something this renderer varies at all.

* **Three degrees of freedom cannot cover six.** The reachable set is a measure-zero subset of swm's
  Box: **0 % of the range**, not a narrow band.
* **The registered default is reachable, and needs no leaf.** swm's `init_value`
  `[[.1,.2,.3],[.2,.3,.4]]` is character-for-character `common/materials.xml`'s own `grid` texture, so
  the default floor **is** the compiled model's floor and `material.grid` stays at its default
  `[1,1,1]`. This is a **range** gap, not a default gap — the distinction W3's Step-7 reviewer drew,
  now confirmed for the whole family.
* **Closing it is a renderer change.** It needs regenerating the builtin checker into `m.tex_data` and
  re-uploading, and `_bundle` holds one `mujoco.Renderer` per `(domain, size)` that is never closed.
  Per plan §4 this is **not** done here.

> **ESCALATION CANDIDATE for the coordinator.** `floor.color` is the one leaf in this family that
> looks feasible and is arguably semantically required, and it needs a renderer change to deliver. The
> decision is whether Step 8 opens that or ships the range gap as a `DOCUMENTED_LIMITATION`. This
> workstream recommends the latter — the default is exact, the mechanism is written down, and a
> texture-edit path touches `stable_envs/render/` and `_bundle`'s context policy, which this
> workstream does not own.

---

## 6. `light.intensity` — mapped, with a documented dark-end gap

swm writes the light's diffuse **absolutely** in [0, 1]; `light.diffuse_scale` **multiplies** the
compiled model's own diffuse and clips to [0, 1]. The factor is `1 / base`, and `base` is read off
`_pristine()["light_diffuse"]`: **0.7** on ten cells, **0.6** on ball-in-cup, finger and manipulator.
Every one of these models has exactly one light (`nlight == 1`) with an equal diffuse on all three
channels, so one scalar scale really is the whole leaf.

| base | reachable intensity | swm range | gap |
|---|---|---|---|
| 0.7 | [0.140, 1.000] | [0, 1] | [0, 0.140) |
| 0.6 | [0.120, 1.000] | [0, 1] | [0, 0.120) |

The bright end costs nothing (`1.8 * base` clips to 1.0, and swm's range stops at 1.0). swm's default
0.7 is reached at scale `1.0` and `7/6` respectively.
`test_light_intensity_lands_on_the_model` asserts the agreement on `light_diffuse` itself over 25
points of the reachable range, to 3e-7, and asserts the gap is real.

---

## 7. The registered default, cell by cell

Ten of eighteen cells: the swm registered default **is** the cell's default style, and the render is
byte-identical over 8 states. Five differ, by design and by the Step-7 coordinator ruling ("certify
against `suite.load`, reach the SWM default through leaves — the Reacher pattern"), and the mapping
reaches them through exactly these leaves and nothing else:

| cell | leaves the mapping moves | px vs the cell's default style |
|---|---|---:|
| `DMCBallInCupCatch` | `light.diffuse_scale` (0.7 vs base 0.6) | 14,835 / 56,448 |
| `DMCCheetahRun` | `material.self` (swm's `.6 .3 .3` vs dm_control's `.7 .5 .3`) | 2,365 / 56,448 |
| `DMCFingerTurnHard` | `light.diffuse_scale`, `visibility.target` | 15,644 / 56,448 |
| `DMCManipulatorBringBall` | `light.diffuse_scale`, `target_ball.shape`, `visibility.target_ball` | 9,109 / 401,408 |
| `DMCReacherHard` | `visibility.target` | 115 / 56,448 |

`test_registered_default_is_reproducible_through_the_mapping` asserts the exact leaf SET, so a cell
that quietly starts needing a sixth leaf fails rather than drifts.

`DMCManipulatorBringBall.target.shape` is the family's one `enum` mapping: swm's `Discrete(2)` is
(0 = box .015, 1 = sphere .015); ours is `Discrete(3)` whose index 2 is the compiled model's own
sphere .02 **and is the default**, which is what keeps the default style a render of the compiled
model. Both swm alternatives are reachable at the same indices, and an index the table does not cover
raises rather than silently picking something.

---

## 8. What changed in the shared file, and why it is backward compatible

`stable_envs/compat/worldmodel.py` — the only shared file this workstream touched:

1. **`VariationMap` gains seven optional fields**, all defaulting to empty: `affine`, `rgb255`,
   `enum`, `unsupported`, `dead_upstream`, `not_applicable`, `notes`; and four methods,
   `affine_pairs()`, `unmapped()`, `mapped_swm_paths()` and `classification()`. `swm_only` is
   untouched and remains the bucket the four pre-Step-8 maps use.
   `test_the_four_pre_step8_maps_are_untouched` asserts that for each of them
   `unmapped() == swm_only`, `mapped_swm_paths() == {t for t, _ in pairs().values()}` and
   `affine_pairs() == {o: (t, f, 0.0) for o, (t, f) in pairs().items()}` — the expressions the old
   code used — so the extension is additive by test, not by inspection.
   **`affine` is `ours = theirs * factor + offset`**, added at the coordinator's request for
   Workstream C's `OGBMaze.floor.pixel_encoding`: an inverted switch is `(-1.0, 1.0)`. It is a NEW
   field rather than a widening of `scaled`, because `scaled`'s two-tuple is what the pre-Step-8 maps
   are written in and a silent arity change is the one edit a reader cannot see; `pairs()` still
   returns two-tuples and `affine_pairs()` is the uniform three-tuple view. A `factor` of 0 is not
   invertible and is refused in `__post_init__`.
2. **`style_from_variation` / `variation_from_style` / `check_mapping`** honour `affine`, `rgb255` and
   `enum` and use `unmapped()`/`mapped_swm_paths()` where they used `swm_only`/`pairs()`.
   `check_mapping` gains three keys (`classification`, `reasons`, `notes`) and loses none.
3. **New `swm_variation_space(swm_env)`**, a bug fix rather than a feature: `check_mapping` and
   `style_from_swm_env` read `ref.unwrapped.variation_space`, which is right for the envs whose
   variation space is on the env itself and **wrong for the five classic-control ids**, where
   `GymControlWrapper` wraps a stock gymnasium env and `unwrapped` reaches past the wrapper to a
   `CartPoleEnv` with no variation space at all. Both call sites now walk the stack.
4. **One registration line**, as the brief allows:
   `from .variation_maps import install as _install_variation_maps; _install_variation_maps(VARIATION_MAPS)`.

`stable_envs/compat/variation_maps/` is new. `__init__.py` collects one module per family and
**refuses** to overwrite an existing key or to let two modules claim one cell, so Workstreams B and C
add a module and one line in `_MODULES` and never edit `worldmodel.py`.

---

## 9. One finding about the swm wrapper that affects how these tests run

Re-measuring `dead_upstream` means driving the live swm wrapper through a variation reset, and every
reset that changes a model-level leaf makes swm **recompile and drop the previous `dm_control`
`Physics`** — which frees an EGL context on garbage collection.
`stable_envs/testing/resource_guard.py` fails a test that releases a context, correctly: releasing one
corrupts every other live context in the process. So `test_dead_upstream_leaves_really_are_dead` runs
its probe in a **subprocess**. That is not a workaround for the guard; it is a property of the swm
wrapper, and the isolation is what makes the measurement safe to run beside the rest of the suite.

---

## 10. Tests

`tests/unit/test_variation_maps_dmc_control.py` — **138 pass** in
`/Users/samib/se8-vm-venv` (with `stable_worldmodel`); **10 pass, 128 skip** in
`/Users/samib/se6-venv` (without it), which is the repository's default test venv.
`tests/unit` as a whole: **1043 pass, 2 fail**, and both failures
(`test_examples.py::...robocasa_visual/render_mjwarp.py` and
`test_repo_contracts.py::test_every_cited_scratch_path_resolves`) reproduce unchanged at the baseline
SHA with this branch's changes stashed.

---

## 11. Per-id leaf tables

### `swm/AcrobotControl-v1`

cell(s): `Acrobot` — registered size 500 px — **11 leaves: 4 mapped, 7 unsupported, 0 dead upstream, 0 not applicable**

The swm registered default IS the cell's default style: **0 of 2,000,000 px** differ over 8 states.

| leaf | swm type / range / default | class | stable-envs control | evidence |
|---|---|---|---|---|
| `physics.link_com_pos_1` | Box[0.25, 0.75], default 0.5 | **unsupported** | — | 0 px / 2,000,000 in the official frame |
| `physics.link_com_pos_2` | Box[0.25, 0.75], default 0.5 | **unsupported** | — | 0 px / 2,000,000 in the official frame |
| `physics.link_length_1` | Box[0.5, 1.5], default 1 | **unsupported** | — | 37,966 px / 2,000,000 in the official frame |
| `physics.link_length_2` | Box[0.5, 1.5], default 1 | **unsupported** | — | 38,258 px / 2,000,000 in the official frame |
| `physics.link_mass_1` | Box[0.5, 1.5], default 1 | **unsupported** | — | 0 px / 2,000,000 in the official frame |
| `physics.link_mass_2` | Box[0.5, 1.5], default 1 | **unsupported** | — | 0 px / 2,000,000 in the official frame |
| `physics.link_moi` | Box[0.5, 1.5], default 1 | **unsupported** | — | 0 px / 2,000,000 in the official frame |
| `visuals.bg` | Box[0.6, 1] shape (3,), default [1.0, 1.0, 1.0] | **mapped** | `background.color` | swm-vs-ours 0 px at the default, 2,569 / 2,000,000 at a probe |
| `visuals.joint` | Box[0, 1] shape (3,), default [0.800000011920929, 0.800000011920929, 0.0] | **mapped** | `joint.color` | swm-vs-ours 0 px at the default, 704 / 2,000,000 at a probe |
| `visuals.line` | Box[0, 0.4] shape (3,), default [0.0, 0.0, 0.0] | **mapped** | `ground.color` | swm-vs-ours 0 px at the default, 2 / 2,000,000 at a probe |
| `visuals.link` | Box[0, 1] shape (3,), default [0.0, 0.800000011920929, 0.800000011920929] | **mapped** | `link.color` | swm-vs-ours 0 px at the default, 2,725 / 2,000,000 at a probe |

* `visuals.bg` — MAPPED, and NOT bit-identical away from the default. swm recolours the rendered frame by EXACT uint8 match; stable-envs draws the primitive in the colour. The two agree wherever the reference wrote the source colour verbatim and differ on anti-aliased boundaries, where swm keeps a blend of the ORIGINAL colour. Measured over 8 states, one leaf at a time at 90 % of its range: 0 disagreeing pixels at the registered default on all five envs, and at most 0.6 % of the frame (11,326 px of 8x400x600, MountainCar) at a probe value. The conversion itself is exact: this leaf uses `rgb255`, which applies swm's own uint8(v * 255 + 0.5).
* `visuals.joint` — MAPPED, and NOT bit-identical away from the default. swm recolours the rendered frame by EXACT uint8 match; stable-envs draws the primitive in the colour. The two agree wherever the reference wrote the source colour verbatim and differ on anti-aliased boundaries, where swm keeps a blend of the ORIGINAL colour. Measured over 8 states, one leaf at a time at 90 % of its range: 0 disagreeing pixels at the registered default on all five envs, and at most 0.6 % of the frame (11,326 px of 8x400x600, MountainCar) at a probe value. The conversion itself is exact: this leaf uses `rgb255`, which applies swm's own uint8(v * 255 + 0.5).
* `visuals.line` — MAPPED, and NOT bit-identical away from the default. swm recolours the rendered frame by EXACT uint8 match; stable-envs draws the primitive in the colour. The two agree wherever the reference wrote the source colour verbatim and differ on anti-aliased boundaries, where swm keeps a blend of the ORIGINAL colour. Measured over 8 states, one leaf at a time at 90 % of its range: 0 disagreeing pixels at the registered default on all five envs, and at most 0.6 % of the frame (11,326 px of 8x400x600, MountainCar) at a probe value. The conversion itself is exact: this leaf uses `rgb255`, which applies swm's own uint8(v * 255 + 0.5).
* `visuals.link` — MAPPED, and NOT bit-identical away from the default. swm recolours the rendered frame by EXACT uint8 match; stable-envs draws the primitive in the colour. The two agree wherever the reference wrote the source colour verbatim and differ on anti-aliased boundaries, where swm keeps a blend of the ORIGINAL colour. Measured over 8 states, one leaf at a time at 90 % of its range: 0 disagreeing pixels at the registered default on all five envs, and at most 0.6 % of the frame (11,326 px of 8x400x600, MountainCar) at a probe value. The conversion itself is exact: this leaf uses `rgb255`, which applies swm's own uint8(v * 255 + 0.5).
* `physics.link_com_pos_1` — dynamics only, and this cell simulates: the value is a class constant on the stable-envs cell, whose integrator is certified against the reference at the reference's own constants. A per-environment value is a dynamics parameter, not a style row. Measured 0 disagreeing pixels over 8 states of the official render.
* `physics.link_com_pos_2` — dynamics only, and this cell simulates: the value is a class constant on the stable-envs cell, whose integrator is certified against the reference at the reference's own constants. A per-environment value is a dynamics parameter, not a style row. Measured 0 disagreeing pixels over 8 states of the official render.
* `physics.link_length_1` — GEOMETRY as well as dynamics: AcrobotEnv.render draws each link at LINK_LENGTH_1 * scale, and moving it to 1.45 moves 37,966 px of 8x500x500 in the official frame. The stable-envs cell holds SCALE and the unit-length link quad as class constants and declares no geometry leaf, so the appearance is not reachable.
* `physics.link_length_2` — the same, on the second link: 38,258 px of 8x500x500 at 1.45.
* `physics.link_mass_1` — dynamics only, and this cell simulates: the value is a class constant on the stable-envs cell, whose integrator is certified against the reference at the reference's own constants. A per-environment value is a dynamics parameter, not a style row. Measured 0 disagreeing pixels over 8 states of the official render.
* `physics.link_mass_2` — dynamics only, and this cell simulates: the value is a class constant on the stable-envs cell, whose integrator is certified against the reference at the reference's own constants. A per-environment value is a dynamics parameter, not a style row. Measured 0 disagreeing pixels over 8 states of the official render.
* `physics.link_moi` — dynamics only, and this cell simulates: the value is a class constant on the stable-envs cell, whose integrator is certified against the reference at the reference's own constants. A per-environment value is a dynamics parameter, not a style row. Measured 0 disagreeing pixels over 8 states of the official render.

### `swm/CartPoleControl-v1`

cell(s): `CartPole` — registered size 400 px — **10 leaves: 4 mapped, 6 unsupported, 0 dead upstream, 0 not applicable**

The swm registered default IS the cell's default style: **0 of 1,920,000 px** differ over 8 states.

| leaf | swm type / range / default | class | stable-envs control | evidence |
|---|---|---|---|---|
| `physics.force_mag` | Box[5, 15], default 10 | **unsupported** | — | 0 px / 1,920,000 in the official frame |
| `physics.gravity` | Box[4.9, 14.7], default 9.8 | **unsupported** | — | 0 px / 1,920,000 in the official frame |
| `physics.length` | Box[0.25, 0.75], default 0.5 | **unsupported** | — | 6,621 px / 1,920,000 in the official frame |
| `physics.masscart` | Box[0.5, 1.5], default 1 | **unsupported** | — | 0 px / 1,920,000 in the official frame |
| `physics.masspole` | Box[0.05, 0.5], default 0.1 | **unsupported** | — | 0 px / 1,920,000 in the official frame |
| `physics.tau` | Box[0.01, 0.04], default 0.02 | **unsupported** | — | 0 px / 1,920,000 in the official frame |
| `visuals.axle` | Box[0, 1] shape (3,), default [0.5058823823928833, 0.5176470875740051, 0.7960784435272217] | **mapped** | `axle.color` | swm-vs-ours 0 px at the default, 64 / 1,920,000 at a probe |
| `visuals.bg` | Box[0.6, 1] shape (3,), default [1.0, 1.0, 1.0] | **mapped** | `background.color` | swm-vs-ours 0 px at the default, 805 / 1,920,000 at a probe |
| `visuals.cart` | Box[0, 0.4] shape (3,), default [0.0, 0.0, 0.0] | **mapped** | `cart.color`, `track.color` | swm-vs-ours 0 px at the default, 99 / 1,920,000 at a probe |
| `visuals.pole` | Box[0, 1] shape (3,), default [0.7921568751335144, 0.5960784554481506, 0.3960784375667572] | **mapped** | `pole.color` | swm-vs-ours 0 px at the default, 957 / 1,920,000 at a probe |

* `visuals.axle` — MAPPED, and NOT bit-identical away from the default. swm recolours the rendered frame by EXACT uint8 match; stable-envs draws the primitive in the colour. The two agree wherever the reference wrote the source colour verbatim and differ on anti-aliased boundaries, where swm keeps a blend of the ORIGINAL colour. Measured over 8 states, one leaf at a time at 90 % of its range: 0 disagreeing pixels at the registered default on all five envs, and at most 0.6 % of the frame (11,326 px of 8x400x600, MountainCar) at a probe value. The conversion itself is exact: this leaf uses `rgb255`, which applies swm's own uint8(v * 255 + 0.5).
* `visuals.bg` — MAPPED, and NOT bit-identical away from the default. swm recolours the rendered frame by EXACT uint8 match; stable-envs draws the primitive in the colour. The two agree wherever the reference wrote the source colour verbatim and differ on anti-aliased boundaries, where swm keeps a blend of the ORIGINAL colour. Measured over 8 states, one leaf at a time at 90 % of its range: 0 disagreeing pixels at the registered default on all five envs, and at most 0.6 % of the frame (11,326 px of 8x400x600, MountainCar) at a probe value. The conversion itself is exact: this leaf uses `rgb255`, which applies swm's own uint8(v * 255 + 0.5).
* `visuals.cart` — MAPPED, and NOT bit-identical away from the default. swm recolours the rendered frame by EXACT uint8 match; stable-envs draws the primitive in the colour. The two agree wherever the reference wrote the source colour verbatim and differ on anti-aliased boundaries, where swm keeps a blend of the ORIGINAL colour. Measured over 8 states, one leaf at a time at 90 % of its range: 0 disagreeing pixels at the registered default on all five envs, and at most 0.6 % of the frame (11,326 px of 8x400x600, MountainCar) at a probe value. The conversion itself is exact: this leaf uses `rgb255`, which applies swm's own uint8(v * 255 + 0.5). ONE-TO-MANY: this leaf also drives `track.color`, because the reference paints the track black too.
* `visuals.pole` — MAPPED, and NOT bit-identical away from the default. swm recolours the rendered frame by EXACT uint8 match; stable-envs draws the primitive in the colour. The two agree wherever the reference wrote the source colour verbatim and differ on anti-aliased boundaries, where swm keeps a blend of the ORIGINAL colour. Measured over 8 states, one leaf at a time at 90 % of its range: 0 disagreeing pixels at the registered default on all five envs, and at most 0.6 % of the frame (11,326 px of 8x400x600, MountainCar) at a probe value. The conversion itself is exact: this leaf uses `rgb255`, which applies swm's own uint8(v * 255 + 0.5).
* `physics.force_mag` — dynamics only, and this cell simulates: the value is a class constant on the stable-envs cell, whose integrator is certified against the reference at the reference's own constants. A per-environment value is a dynamics parameter, not a style row. Measured 0 disagreeing pixels over 8 states of the official render.
* `physics.gravity` — dynamics only, and this cell simulates: the value is a class constant on the stable-envs cell, whose integrator is certified against the reference at the reference's own constants. A per-environment value is a dynamics parameter, not a style row. Measured 0 disagreeing pixels over 8 states of the official render.
* `physics.length` — GEOMETRY as well as dynamics: CartPoleEnv.render sets polelen = scale * 2 * length, and 0.725 moves 6,621 px of 8x400x600 in the official frame. POLELEN is a class constant here (it is also what WIN_POLE and KMAX_POLE are derived from) and there is no geometry leaf, so the appearance is not reachable.
* `physics.masscart` — dynamics only, and this cell simulates: the value is a class constant on the stable-envs cell, whose integrator is certified against the reference at the reference's own constants. A per-environment value is a dynamics parameter, not a style row. Measured 0 disagreeing pixels over 8 states of the official render.
* `physics.masspole` — dynamics only, and this cell simulates: the value is a class constant on the stable-envs cell, whose integrator is certified against the reference at the reference's own constants. A per-environment value is a dynamics parameter, not a style row. Measured 0 disagreeing pixels over 8 states of the official render.
* `physics.tau` — dynamics only, and this cell simulates: the value is a class constant on the stable-envs cell, whose integrator is certified against the reference at the reference's own constants. A per-environment value is a dynamics parameter, not a style row. Measured 0 disagreeing pixels over 8 states of the official render.

### `swm/PendulumControl-v1`

cell(s): `Pendulum` — registered size 500 px — **9 leaves: 3 mapped, 6 unsupported, 0 dead upstream, 0 not applicable**

The swm registered default IS the cell's default style: **0 of 2,000,000 px** differ over 8 states.

| leaf | swm type / range / default | class | stable-envs control | evidence |
|---|---|---|---|---|
| `physics.dt` | Box[0.025, 0.1], default 0.05 | **unsupported** | — | 0 px / 2,000,000 in the official frame |
| `physics.g` | Box[5, 15], default 10 | **unsupported** | — | 0 px / 2,000,000 in the official frame |
| `physics.l` | Box[0.5, 1.5], default 1 | **unsupported** | — | 0 px / 2,000,000 in the official frame |
| `physics.m` | Box[0.5, 1.5], default 1 | **unsupported** | — | 0 px / 2,000,000 in the official frame |
| `physics.max_speed` | Box[4, 12], default 8 | **unsupported** | — | 0 px / 2,000,000 in the official frame |
| `physics.max_torque` | Box[1, 4], default 2 | **unsupported** | — | 0 px / 2,000,000 in the official frame |
| `visuals.bg` | Box[0.6, 1] shape (3,), default [1.0, 1.0, 1.0] | **mapped** | `background.color` | swm-vs-ours 0 px at the default, 2,726 / 2,000,000 at a probe |
| `visuals.hub` | Box[0, 0.4] shape (3,), default [0.0, 0.0, 0.0] | **mapped** | `axle.color` | swm-vs-ours 0 px at the default, 64 / 2,000,000 at a probe |
| `visuals.rod` | Box[0, 1] shape (3,), default [0.800000011920929, 0.3019607961177826, 0.3019607961177826] | **mapped** | `rod.color` | swm-vs-ours 0 px at the default, 1,819 / 2,000,000 at a probe |

* `visuals.bg` — MAPPED, and NOT bit-identical away from the default. swm recolours the rendered frame by EXACT uint8 match; stable-envs draws the primitive in the colour. The two agree wherever the reference wrote the source colour verbatim and differ on anti-aliased boundaries, where swm keeps a blend of the ORIGINAL colour. Measured over 8 states, one leaf at a time at 90 % of its range: 0 disagreeing pixels at the registered default on all five envs, and at most 0.6 % of the frame (11,326 px of 8x400x600, MountainCar) at a probe value. The conversion itself is exact: this leaf uses `rgb255`, which applies swm's own uint8(v * 255 + 0.5).
* `visuals.hub` — MAPPED, and NOT bit-identical away from the default. swm recolours the rendered frame by EXACT uint8 match; stable-envs draws the primitive in the colour. The two agree wherever the reference wrote the source colour verbatim and differ on anti-aliased boundaries, where swm keeps a blend of the ORIGINAL colour. Measured over 8 states, one leaf at a time at 90 % of its range: 0 disagreeing pixels at the registered default on all five envs, and at most 0.6 % of the frame (11,326 px of 8x400x600, MountainCar) at a probe value. The conversion itself is exact: this leaf uses `rgb255`, which applies swm's own uint8(v * 255 + 0.5).
* `visuals.rod` — MAPPED, and NOT bit-identical away from the default. swm recolours the rendered frame by EXACT uint8 match; stable-envs draws the primitive in the colour. The two agree wherever the reference wrote the source colour verbatim and differ on anti-aliased boundaries, where swm keeps a blend of the ORIGINAL colour. Measured over 8 states, one leaf at a time at 90 % of its range: 0 disagreeing pixels at the registered default on all five envs, and at most 0.6 % of the frame (11,326 px of 8x400x600, MountainCar) at a probe value. The conversion itself is exact: this leaf uses `rgb255`, which applies swm's own uint8(v * 255 + 0.5).
* `physics.dt` — dynamics only, and this cell simulates: the value is a class constant on the stable-envs cell, whose integrator is certified against the reference at the reference's own constants. A per-environment value is a dynamics parameter, not a style row. Measured 0 disagreeing pixels over 8 states of the official render.
* `physics.g` — dynamics only, and this cell simulates: the value is a class constant on the stable-envs cell, whose integrator is certified against the reference at the reference's own constants. A per-environment value is a dynamics parameter, not a style row. Measured 0 disagreeing pixels over 8 states of the official render.
* `physics.l` — dynamics only, and this cell simulates: the value is a class constant on the stable-envs cell, whose integrator is certified against the reference at the reference's own constants. A per-environment value is a dynamics parameter, not a style row. Measured 0 disagreeing pixels over 8 states of the official render.
* `physics.m` — dynamics only, and this cell simulates: the value is a class constant on the stable-envs cell, whose integrator is certified against the reference at the reference's own constants. A per-environment value is a dynamics parameter, not a style row. Measured 0 disagreeing pixels over 8 states of the official render.
* `physics.max_speed` — dynamics only, and this cell simulates: the value is a class constant on the stable-envs cell, whose integrator is certified against the reference at the reference's own constants. A per-environment value is a dynamics parameter, not a style row. Measured 0 disagreeing pixels over 8 states of the official render.
* `physics.max_torque` — dynamics only, and this cell simulates: the value is a class constant on the stable-envs cell, whose integrator is certified against the reference at the reference's own constants. A per-environment value is a dynamics parameter, not a style row. Measured 0 disagreeing pixels over 8 states of the official render.

### `swm/MountainCarControl-v0`

cell(s): `MountainCar` — registered size 400 px — **5 leaves: 2 mapped, 3 unsupported, 0 dead upstream, 0 not applicable**

The swm registered default IS the cell's default style: **0 of 1,920,000 px** differ over 8 states.

| leaf | swm type / range / default | class | stable-envs control | evidence |
|---|---|---|---|---|
| `physics.force` | Box[0.0005, 0.0015], default 0.001 | **unsupported** | — | 0 px / 1,920,000 in the official frame |
| `physics.gravity` | Box[0.00125, 0.00375], default 0.0025 | **unsupported** | — | 0 px / 1,920,000 in the official frame |
| `physics.max_speed` | Box[0.035, 0.105], default 0.07 | **unsupported** | — | 0 px / 1,920,000 in the official frame |
| `visuals.bg` | Box[0.6, 1] shape (3,), default [1.0, 1.0, 1.0] | **mapped** | `background.color` | swm-vs-ours 0 px at the default, 11,326 / 1,920,000 at a probe |
| `visuals.fg` | Box[0, 0.4] shape (3,), default [0.0, 0.0, 0.0] | **mapped** | `car.color`, `flag.pole_color`, `terrain.color` | swm-vs-ours 0 px at the default, 11,033 / 1,920,000 at a probe |

* `visuals.bg` — MAPPED, and NOT bit-identical away from the default. swm recolours the rendered frame by EXACT uint8 match; stable-envs draws the primitive in the colour. The two agree wherever the reference wrote the source colour verbatim and differ on anti-aliased boundaries, where swm keeps a blend of the ORIGINAL colour. Measured over 8 states, one leaf at a time at 90 % of its range: 0 disagreeing pixels at the registered default on all five envs, and at most 0.6 % of the frame (11,326 px of 8x400x600, MountainCar) at a probe value. The conversion itself is exact: this leaf uses `rgb255`, which applies swm's own uint8(v * 255 + 0.5).
* `visuals.fg` — MAPPED, and NOT bit-identical away from the default. swm recolours the rendered frame by EXACT uint8 match; stable-envs draws the primitive in the colour. The two agree wherever the reference wrote the source colour verbatim and differ on anti-aliased boundaries, where swm keeps a blend of the ORIGINAL colour. Measured over 8 states, one leaf at a time at 90 % of its range: 0 disagreeing pixels at the registered default on all five envs, and at most 0.6 % of the frame (11,326 px of 8x400x600, MountainCar) at a probe value. The conversion itself is exact: this leaf uses `rgb255`, which applies swm's own uint8(v * 255 + 0.5). ONE-TO-MANY: this leaf drives `car.color`, `terrain.color` AND `flag.pole_color`, because the reference paints all three black.
* `physics.force` — dynamics only, and this cell simulates: the value is a class constant on the stable-envs cell, whose integrator is certified against the reference at the reference's own constants. A per-environment value is a dynamics parameter, not a style row. Measured 0 disagreeing pixels over 8 states of the official render.
* `physics.gravity` — dynamics only, and this cell simulates: the value is a class constant on the stable-envs cell, whose integrator is certified against the reference at the reference's own constants. A per-environment value is a dynamics parameter, not a style row. Measured 0 disagreeing pixels over 8 states of the official render.
* `physics.max_speed` — dynamics only, and this cell simulates: the value is a class constant on the stable-envs cell, whose integrator is certified against the reference at the reference's own constants. A per-environment value is a dynamics parameter, not a style row. Measured 0 disagreeing pixels over 8 states of the official render.

### `swm/MountainCarContinuousControl-v0`

cell(s): `MountainCarContinuous` — registered size 400 px — **4 leaves: 2 mapped, 2 unsupported, 0 dead upstream, 0 not applicable**

The swm registered default IS the cell's default style: **0 of 1,920,000 px** differ over 8 states.

| leaf | swm type / range / default | class | stable-envs control | evidence |
|---|---|---|---|---|
| `physics.max_speed` | Box[0.035, 0.105], default 0.07 | **unsupported** | — | 0 px / 1,920,000 in the official frame |
| `physics.power` | Box[0.00075, 0.00225], default 0.0015 | **unsupported** | — | 0 px / 1,920,000 in the official frame |
| `visuals.bg` | Box[0.6, 1] shape (3,), default [1.0, 1.0, 1.0] | **mapped** | `background.color` | swm-vs-ours 0 px at the default, 11,325 / 1,920,000 at a probe |
| `visuals.fg` | Box[0, 0.4] shape (3,), default [0.0, 0.0, 0.0] | **mapped** | `car.color`, `flag.pole_color`, `terrain.color` | swm-vs-ours 0 px at the default, 11,031 / 1,920,000 at a probe |

* `visuals.bg` — MAPPED, and NOT bit-identical away from the default. swm recolours the rendered frame by EXACT uint8 match; stable-envs draws the primitive in the colour. The two agree wherever the reference wrote the source colour verbatim and differ on anti-aliased boundaries, where swm keeps a blend of the ORIGINAL colour. Measured over 8 states, one leaf at a time at 90 % of its range: 0 disagreeing pixels at the registered default on all five envs, and at most 0.6 % of the frame (11,326 px of 8x400x600, MountainCar) at a probe value. The conversion itself is exact: this leaf uses `rgb255`, which applies swm's own uint8(v * 255 + 0.5).
* `visuals.fg` — MAPPED, and NOT bit-identical away from the default. swm recolours the rendered frame by EXACT uint8 match; stable-envs draws the primitive in the colour. The two agree wherever the reference wrote the source colour verbatim and differ on anti-aliased boundaries, where swm keeps a blend of the ORIGINAL colour. Measured over 8 states, one leaf at a time at 90 % of its range: 0 disagreeing pixels at the registered default on all five envs, and at most 0.6 % of the frame (11,326 px of 8x400x600, MountainCar) at a probe value. The conversion itself is exact: this leaf uses `rgb255`, which applies swm's own uint8(v * 255 + 0.5). ONE-TO-MANY: this leaf drives `car.color`, `terrain.color` AND `flag.pole_color`, because the reference paints all three black.
* `physics.max_speed` — dynamics only, and this cell simulates: the value is a class constant on the stable-envs cell, whose integrator is certified against the reference at the reference's own constants. A per-environment value is a dynamics parameter, not a style row. Measured 0 disagreeing pixels over 8 states of the official render.
* `physics.power` — dynamics only, and this cell simulates: the value is a class constant on the stable-envs cell, whose integrator is certified against the reference at the reference's own constants. A per-environment value is a dynamics parameter, not a style row. Measured 0 disagreeing pixels over 8 states of the official render.

### `swm/AcrobotDMControl-v0`

cell(s): `DMCAcrobotSwingup` — registered size 84 px — **11 leaves: 2 mapped, 4 unsupported, 5 dead upstream, 0 not applicable**

The swm registered default IS the cell's default style: **0 of 56,448 px** differ over 8 states.

| leaf | swm type / range / default | class | stable-envs control | evidence |
|---|---|---|---|---|
| `agent.color` | Box[0, 1] shape (3,), default [0.7, 0.5, 0.3] | **mapped** | `material.self` | moves `mat_rgba`; 684 px / 8 states |
| `agent.lower_arm_density` | Box[500, 1500], default 1000 | **dead_upstream** | — | no MjModel array moves; 0 px / 8 states |
| `agent.upper_arm_density` | Box[500, 1500], default 1000 | **dead_upstream** | — | no MjModel array moves; 0 px / 8 states |
| `agent.upper_arm_locked` | Discrete(2), default 1 | **dead_upstream** | — | no MjModel array moves; 0 px / 8 states |
| `floor.color` | Box[0, 1] shape (2, 3), default [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]] | **unsupported** | — | moves `tex_data`; 13,036 px / 8 states |
| `gravity.x` | Box[-5, 5], default 0 | **unsupported** | — | moves `__opt_gravity`; 0 px / 8 states |
| `gravity.y` | Box[-5, 5], default 0 | **unsupported** | — | moves `__opt_gravity`; 0 px / 8 states |
| `gravity.z` | Box[-20, 0], default -9.81 | **unsupported** | — | moves `__opt_gravity`; 0 px / 8 states |
| `light.intensity` | Box[0, 1], default 0.7 | **mapped** | `light.diffuse_scale` | moves `light_diffuse`; 13,103 px / 8 states |
| `target.color` | Box[0, 1] shape (3,), default [0.6, 0.3, 0.3] | **dead_upstream** | — | moves `mat_rgba`; 0 px / 8 states |
| `target.shape` | Discrete(2), default 1 | **dead_upstream** | — | moves `site_size`, `site_type`; 0 px / 8 states |

* `light.intensity` — mapped to light.diffuse_scale with factor 1/0.7 — the compiled model's own light diffuse. RANGE GAP at the dark end: our scale is Box(0.2, 1.8), so the reachable intensity is [0.140, 1.0] (1.8 * 0.7 clips to 1.0 at the bright end, which costs nothing because swm's range stops at 1.0), and swm's [0, 0.140) is not reachable. swm's default 0.7 is reached at scale 1.000000.
* `agent.lower_arm_density` — DEAD UPSTREAM: the wrapper writes the value onto the geom's ``density`` and recompiles, but acrobot's <default class="/"><geom type="capsule" mass="1"/> means every geom carries an EXPLICIT mass, and in MuJoCo an explicit mass wins over density. Measured: not one MjModel array changes, body_mass included.
* `agent.upper_arm_density` — DEAD UPSTREAM: the wrapper writes the value onto the geom's ``density`` and recompiles, but acrobot's <default class="/"><geom type="capsule" mass="1"/> means every geom carries an EXPLICIT mass, and in MuJoCo an explicit mass wins over density. Measured: not one MjModel array changes, body_mass included.
* `agent.upper_arm_locked` — DEAD UPSTREAM: the leaf is declared by the swm wrapper and its own ``modify_mjcf_model`` never reads it (0 textual references outside the space literal). Measured: not one of the 110+ MjModel arrays changes between the leaf's two values, so it reaches neither a pixel nor a state.
* `floor.color` — UNSUPPORTED, and it is a RANGE gap rather than a default gap. swm writes the builtin ``grid`` TEXTURE's rgb1/rgb2 (2x3 = six free scalars) and recompiles. stable-envs has exactly one floor knob, ``material.grid``, a 3-vector that multiplies the material rgba — and ``_tex_rgb`` reduces the whole checker to its MEAN colour, so even the checker pattern is not a thing this renderer varies. Three degrees of freedom cannot cover six: the reachable set is a measure-zero subset of swm's Box, i.e. 0 % of the range. What IS reachable is the one value that matters for certification: swm's init_value [[.1,.2,.3],[.2,.3,.4]] is character-for-character common/materials.xml's own grid texture, so the REGISTERED DEFAULT floor is the compiled model's floor and needs no leaf at all (material.grid stays at its default [1,1,1]). Closing the range needs a texture edit — regenerating the builtin checker into m.tex_data and re-uploading — and ``_bundle`` holds one mujoco.Renderer per (domain, size) that is never closed, so that is a renderer change. ESCALATION CANDIDATE, carried forward from the Step-7 manipulator finding.
* `gravity.x` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: gravity is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `gravity.y` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: gravity is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `gravity.z` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: gravity is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `target.color` — DEAD UPSTREAM: the leaf does reach the compiled model, and the element it reaches is never drawn. The target is in geom/site GROUP 3, and mjvOption's default ``geomgroup`` rasterises groups 0-2 only. Measured 0 disagreeing pixels over 8 states (site `target`, material `target`).
* `target.shape` — DEAD UPSTREAM: the leaf does reach the compiled model, and the element it reaches is never drawn. The target is in geom/site GROUP 3, and mjvOption's default ``geomgroup`` rasterises groups 0-2 only. Measured 0 disagreeing pixels over 8 states (site `target`; the leaf does move site_type and site_size).

### `swm/BallInCupDMControl-v0`

cell(s): `DMCBallInCupCatch` — registered size 84 px — **13 leaves: 2 mapped, 7 unsupported, 4 dead upstream, 0 not applicable**

The swm REGISTERED DEFAULT is a different appearance from the compiled model's and is reached through `light.diffuse_scale` (14,835 of 56,448 px differ from the cell's default style over 8 states).

| leaf | swm type / range / default | class | stable-envs control | evidence |
|---|---|---|---|---|
| `agent.color` | Box[0, 1] shape (3,), default [0.7, 0.5, 0.3] | **mapped** | `material.self` | moves `mat_rgba`; 482 px / 8 states |
| `agent.density` | Box[500, 1500], default 1000 | **unsupported** | — | moves `actuator_acc0`, `body_inertia`, `body_invweight0`...; 0 px / 8 states |
| `ball.color` | Box[0, 1] shape (3,), default [0.7, 0.5, 0.3] | **dead_upstream** | — | no MjModel array moves; 0 px / 8 states |
| `ball.density` | Box[500, 1500], default 1000 | **unsupported** | — | moves `body_inertia`, `body_invweight0`, `body_mass`...; 0 px / 8 states |
| `ball.size` | Box[0.01, 0.05], default 0.025 | **unsupported** | — | moves `body_inertia`, `body_invweight0`, `body_mass`...; 902 px / 8 states |
| `floor.color` | Box[0, 1] shape (2, 3), default [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]] | **unsupported** | — | moves `tex_data`; 14,765 px / 8 states |
| `gravity.x` | Box[-5, 5], default 0 | **unsupported** | — | moves `__opt_gravity`; 178 px / 8 states |
| `gravity.y` | Box[-5, 5], default 0 | **unsupported** | — | moves `__opt_gravity`; 110 px / 8 states |
| `gravity.z` | Box[-20, 0], default -9.81 | **unsupported** | — | moves `__opt_gravity`; 0 px / 8 states |
| `light.intensity` | Box[0, 1], default 0.7 | **mapped** | `light.diffuse_scale` | moves `light_diffuse`; 14,915 px / 8 states |
| `rendering.render_target` | Discrete(2), default 0 | **dead_upstream** | — | moves `site_rgba`; 0 px / 8 states |
| `target.color` | Box[0, 1] shape (3,), default [0.6, 0.3, 0.3] | **dead_upstream** | — | moves `site_rgba`; 0 px / 8 states |
| `target.shape` | Discrete(2), default 1 | **dead_upstream** | — | moves `site_type`; 0 px / 8 states |

* `light.intensity` — mapped to light.diffuse_scale with factor 1/0.6 — the compiled model's own light diffuse. RANGE GAP at the dark end: our scale is Box(0.2, 1.8), so the reachable intensity is [0.120, 1.0] (1.8 * 0.6 clips to 1.0 at the bright end, which costs nothing because swm's range stops at 1.0), and swm's [0, 0.120) is not reachable. swm's default 0.7 is reached at scale 1.166667.
* `agent.density` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: the cup's density is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `ball.color` — DEAD UPSTREAM: the leaf is declared by the swm wrapper and its own ``modify_mjcf_model`` never reads it (0 textual references outside the space literal). Measured: not one of the 110+ MjModel arrays changes between the leaf's two values, so it reaches neither a pixel nor a state. The ball is drawn with material `self`, which ``agent.color`` already reaches.
* `ball.density` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: the ball's density is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `ball.size` — the ball's radius, which is GEOMETRY: 902 disagreeing pixels of 8x84x84 at 0.0475. The cell declares colour, light and (elsewhere) visibility leaves and no size leaf, so the appearance is not reachable.
* `floor.color` — UNSUPPORTED, and it is a RANGE gap rather than a default gap. swm writes the builtin ``grid`` TEXTURE's rgb1/rgb2 (2x3 = six free scalars) and recompiles. stable-envs has exactly one floor knob, ``material.grid``, a 3-vector that multiplies the material rgba — and ``_tex_rgb`` reduces the whole checker to its MEAN colour, so even the checker pattern is not a thing this renderer varies. Three degrees of freedom cannot cover six: the reachable set is a measure-zero subset of swm's Box, i.e. 0 % of the range. What IS reachable is the one value that matters for certification: swm's init_value [[.1,.2,.3],[.2,.3,.4]] is character-for-character common/materials.xml's own grid texture, so the REGISTERED DEFAULT floor is the compiled model's floor and needs no leaf at all (material.grid stays at its default [1,1,1]). Closing the range needs a texture edit — regenerating the builtin checker into m.tex_data and re-uploading — and ``_bundle`` holds one mujoco.Renderer per (domain, size) that is never closed, so that is a renderer change. ESCALATION CANDIDATE, carried forward from the Step-7 manipulator finding.
* `gravity.x` — UNSUPPORTED rather than physics-only: ball_in_cup's cup and ball are joined by a SPATIAL TENDON, which MuJoCo draws as a hanging catenary through mjv_catenary, and mjv_catenary reads the model's gravity. Measured at a fixed qpos: gravity.x moves 178 px and gravity.y 110 px of 8x84x84 (gravity.z alone moves 0, because scaling z does not rotate the gravity direction). The cell reproduces mjv_catenary from the compiled model's gravity and declares no gravity leaf.
* `gravity.y` — UNSUPPORTED rather than physics-only: ball_in_cup's cup and ball are joined by a SPATIAL TENDON, which MuJoCo draws as a hanging catenary through mjv_catenary, and mjv_catenary reads the model's gravity. Measured at a fixed qpos: gravity.x moves 178 px and gravity.y 110 px of 8x84x84 (gravity.z alone moves 0, because scaling z does not rotate the gravity direction). The cell reproduces mjv_catenary from the compiled model's gravity and declares no gravity leaf.
* `gravity.z` — UNSUPPORTED rather than physics-only: ball_in_cup's cup and ball are joined by a SPATIAL TENDON, which MuJoCo draws as a hanging catenary through mjv_catenary, and mjv_catenary reads the model's gravity. Measured at a fixed qpos: gravity.x moves 178 px and gravity.y 110 px of 8x84x84 (gravity.z alone moves 0, because scaling z does not rotate the gravity direction). The cell reproduces mjv_catenary from the compiled model's gravity and declares no gravity leaf.
* `rendering.render_target` — DEAD UPSTREAM: the leaf does reach the compiled model, and the element it reaches is never drawn. The target is in geom/site GROUP 4, and mjvOption's default ``geomgroup`` rasterises groups 0-2 only. Measured 0 disagreeing pixels over 8 states (site `target`), and 0 px again with render_target forced to 1.
* `target.color` — DEAD UPSTREAM: the leaf does reach the compiled model, and the element it reaches is never drawn. The target is in geom/site GROUP 4, and mjvOption's default ``geomgroup`` rasterises groups 0-2 only. Measured 0 disagreeing pixels over 8 states (site `target`, rgba on the site itself).
* `target.shape` — DEAD UPSTREAM: the leaf does reach the compiled model, and the element it reaches is never drawn. The target is in geom/site GROUP 4, and mjvOption's default ``geomgroup`` rasterises groups 0-2 only. Measured 0 disagreeing pixels over 8 states (site `target`; the leaf does move site_type).

### `swm/CartpoleDMControl-v0`

cell(s): `DMCCartpoleBalance`, `DMCCartpoleSwingup` — registered size 84 px — **9 leaves: 2 mapped, 5 unsupported, 2 dead upstream, 0 not applicable**

The swm registered default IS the cell's default style: **0 of 56,448 px** differ over 8 states.

| leaf | swm type / range / default | class | stable-envs control | evidence |
|---|---|---|---|---|
| `agent.cart_mass` | Box[0.5, 1.5], default 1 | **unsupported** | — | moves `actuator_acc0`, `body_inertia`, `body_invweight0`...; 0 px / 8 states |
| `agent.cart_shape` | Discrete(2), default 1 | **dead_upstream** | — | no MjModel array moves; 0 px / 8 states |
| `agent.color` | Box[0, 1] shape (3,), default [0.7, 0.5, 0.3] | **mapped** | `material.self` | moves `mat_rgba`; 1,394 px / 8 states |
| `agent.pole_density` | Box[500, 1500], default 1000 | **dead_upstream** | — | no MjModel array moves; 0 px / 8 states |
| `floor.color` | Box[0, 1] shape (2, 3), default [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]] | **unsupported** | — | moves `tex_data`; 19,333 px / 8 states |
| `gravity.x` | Box[-5, 5], default 0 | **unsupported** | — | moves `__opt_gravity`; 0 px / 8 states |
| `gravity.y` | Box[-5, 5], default 0 | **unsupported** | — | moves `__opt_gravity`; 0 px / 8 states |
| `gravity.z` | Box[-20, 0], default -9.81 | **unsupported** | — | moves `__opt_gravity`; 0 px / 8 states |
| `light.intensity` | Box[0, 1], default 0.7 | **mapped** | `light.diffuse_scale` | moves `light_diffuse`; 19,633 px / 8 states |

* `light.intensity` — mapped to light.diffuse_scale with factor 1/0.7 — the compiled model's own light diffuse. RANGE GAP at the dark end: our scale is Box(0.2, 1.8), so the reachable intensity is [0.140, 1.0] (1.8 * 0.7 clips to 1.0 at the bright end, which costs nothing because swm's range stops at 1.0), and swm's [0, 0.140) is not reachable. swm's default 0.7 is reached at scale 1.000000.
* `agent.cart_mass` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: the cart's mass is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `agent.cart_shape` — DEAD UPSTREAM: the leaf is declared by the swm wrapper and its own ``modify_mjcf_model`` never reads it (0 textual references outside the space literal). Measured: not one of the 110+ MjModel arrays changes between the leaf's two values, so it reaches neither a pixel nor a state.
* `agent.pole_density` — DEAD UPSTREAM: the wrapper writes the value onto the geom's ``density`` and recompiles, but cartpole.xml's ``pole`` default class, which sets an explicit mass on every pole geom carries an EXPLICIT mass, and in MuJoCo an explicit mass wins over density. Measured: not one MjModel array changes, body_mass included.
* `floor.color` — UNSUPPORTED, and it is a RANGE gap rather than a default gap. swm writes the builtin ``grid`` TEXTURE's rgb1/rgb2 (2x3 = six free scalars) and recompiles. stable-envs has exactly one floor knob, ``material.grid``, a 3-vector that multiplies the material rgba — and ``_tex_rgb`` reduces the whole checker to its MEAN colour, so even the checker pattern is not a thing this renderer varies. Three degrees of freedom cannot cover six: the reachable set is a measure-zero subset of swm's Box, i.e. 0 % of the range. What IS reachable is the one value that matters for certification: swm's init_value [[.1,.2,.3],[.2,.3,.4]] is character-for-character common/materials.xml's own grid texture, so the REGISTERED DEFAULT floor is the compiled model's floor and needs no leaf at all (material.grid stays at its default [1,1,1]). Closing the range needs a texture edit — regenerating the builtin checker into m.tex_data and re-uploading — and ``_bundle`` holds one mujoco.Renderer per (domain, size) that is never closed, so that is a renderer change. ESCALATION CANDIDATE, carried forward from the Step-7 manipulator finding.
* `gravity.x` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: gravity is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `gravity.y` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: gravity is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `gravity.z` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: gravity is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.

### `swm/CheetahDMControl-v0`

cell(s): `DMCCheetahRun` — registered size 84 px — **10 leaves: 2 mapped, 7 unsupported, 1 dead upstream, 0 not applicable**

The swm REGISTERED DEFAULT is a different appearance from the compiled model's and is reached through `material.self` (2,365 of 56,448 px differ from the cell's default style over 8 states).

| leaf | swm type / range / default | class | stable-envs control | evidence |
|---|---|---|---|---|
| `agent.back_foot_density` | Box[500, 1500], default 1000 | **unsupported** | — | moves `actuator_acc0`, `body_inertia`, `body_invweight0`...; 4,521 px / 8 states |
| `agent.back_foot_locked` | Discrete(2), default 1 | **dead_upstream** | — | no MjModel array moves; 0 px / 8 states |
| `agent.color` | Box[0, 1] shape (3,), default [0.6, 0.3, 0.3] | **mapped** | `material.self` | moves `mat_rgba`; 3,337 px / 8 states |
| `agent.torso_density` | Box[500, 1500], default 1000 | **unsupported** | — | moves `actuator_acc0`, `body_inertia`, `body_invweight0`...; 2,115 px / 8 states |
| `floor.color` | Box[0, 1] shape (2, 3), default [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]] | **unsupported** | — | moves `tex_data`; 7,969 px / 8 states |
| `floor.friction` | Box[0, 1], default 1 | **unsupported** | — | moves `geom_friction`; 0 px / 8 states |
| `gravity.x` | Box[-5, 5], default 0 | **unsupported** | — | moves `__opt_gravity`; 0 px / 8 states |
| `gravity.y` | Box[-5, 5], default 0 | **unsupported** | — | moves `__opt_gravity`; 0 px / 8 states |
| `gravity.z` | Box[-20, 0], default -9.81 | **unsupported** | — | moves `__opt_gravity`; 0 px / 8 states |
| `light.intensity` | Box[0, 1], default 0.7 | **mapped** | `light.diffuse_scale` | moves `light_diffuse`; 8,323 px / 8 states |

* `light.intensity` — mapped to light.diffuse_scale with factor 1/0.7 — the compiled model's own light diffuse. RANGE GAP at the dark end: our scale is Box(0.2, 1.8), so the reachable intensity is [0.140, 1.0] (1.8 * 0.7 clips to 1.0 at the bright end, which costs nothing because swm's range stops at 1.0), and swm's [0, 0.140) is not reachable. swm's default 0.7 is reached at scale 1.000000.
* `agent.back_foot_density` — UNSUPPORTED, not merely physics: this cell renders through a mode="trackcom" camera, whose position comes from subtree_com, so a mass change MOVES THE CAMERA. Measured 4,521 disagreeing pixels of 56,448 over 8 states with qpos and geom_xpos bit-identical. The cell declares no mass leaf and its camera follows the compiled model's own COM, so the appearance is not reachable.
* `agent.back_foot_locked` — DEAD UPSTREAM: the leaf is declared by the swm wrapper and its own ``modify_mjcf_model`` never reads it (0 textual references outside the space literal). Measured: not one of the 110+ MjModel arrays changes between the leaf's two values, so it reaches neither a pixel nor a state.
* `agent.torso_density` — UNSUPPORTED, not merely physics: this cell renders through a mode="trackcom" camera, whose position comes from subtree_com, so a mass change MOVES THE CAMERA. Measured 2,115 disagreeing pixels of 56,448 over 8 states with qpos and geom_xpos bit-identical. The cell declares no mass leaf and its camera follows the compiled model's own COM, so the appearance is not reachable.
* `floor.color` — UNSUPPORTED, and it is a RANGE gap rather than a default gap. swm writes the builtin ``grid`` TEXTURE's rgb1/rgb2 (2x3 = six free scalars) and recompiles. stable-envs has exactly one floor knob, ``material.grid``, a 3-vector that multiplies the material rgba — and ``_tex_rgb`` reduces the whole checker to its MEAN colour, so even the checker pattern is not a thing this renderer varies. Three degrees of freedom cannot cover six: the reachable set is a measure-zero subset of swm's Box, i.e. 0 % of the range. What IS reachable is the one value that matters for certification: swm's init_value [[.1,.2,.3],[.2,.3,.4]] is character-for-character common/materials.xml's own grid texture, so the REGISTERED DEFAULT floor is the compiled model's floor and needs no leaf at all (material.grid stays at its default [1,1,1]). Closing the range needs a texture edit — regenerating the builtin checker into m.tex_data and re-uploading — and ``_bundle`` holds one mujoco.Renderer per (domain, size) that is never closed, so that is a renderer change. ESCALATION CANDIDATE, carried forward from the Step-7 manipulator finding.
* `floor.friction` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: the floor's sliding friction is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `gravity.x` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: gravity is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `gravity.y` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: gravity is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `gravity.z` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: gravity is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.

### `swm/FingerDMControl-v0`

cell(s): `DMCFingerTurnHard` — registered size 84 px — **11 leaves: 4 mapped, 7 unsupported, 0 dead upstream, 0 not applicable**

The swm REGISTERED DEFAULT is a different appearance from the compiled model's and is reached through `light.diffuse_scale`, `visibility.target` (15,644 of 56,448 px differ from the cell's default style over 8 states).

| leaf | swm type / range / default | class | stable-envs control | evidence |
|---|---|---|---|---|
| `agent.color` | Box[0, 1] shape (3,), default [0.7, 0.5, 0.3] | **mapped** | `material.self` | moves `mat_rgba`; 1,317 px / 8 states |
| `agent.fingertip_density` | Box[500, 1500], default 1000 | **unsupported** | — | moves `actuator_acc0`, `body_inertia`, `body_invweight0`...; 0 px / 8 states |
| `agent.proximal_density` | Box[500, 1500], default 1000 | **unsupported** | — | moves `actuator_acc0`, `body_inertia`, `body_invweight0`...; 0 px / 8 states |
| `floor.color` | Box[0, 1] shape (2, 3), default [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]] | **unsupported** | — | moves `tex_data`; 14,755 px / 8 states |
| `light.intensity` | Box[0, 1], default 0.7 | **mapped** | `light.diffuse_scale` | moves `light_diffuse`; 15,922 px / 8 states; 15,985 px with the target visible |
| `rendering.render_target` | Discrete(2), default 0 | **mapped** | `visibility.target` | moves `mat_rgba`; 382 px / 8 states |
| `spinner.color` | Box[0, 1] shape (3,), default [0.7, 0.5, 0.3] | **unsupported** | — | moves `geom_rgba`; 2,191 px / 8 states; 2,184 px with the target visible |
| `spinner.density` | Box[500, 1500], default 1000 | **unsupported** | — | moves `body_inertia`, `body_invweight0`, `body_ipos`...; 0 px / 8 states |
| `spinner.friction` | Box[0, 1], default 0.1 | **unsupported** | — | moves `dof_frictionloss`; 0 px / 8 states |
| `target.color` | Box[0, 1] shape (3,), default [0.6, 0.3, 0.3] | **mapped** | `material.target` | moves `mat_rgba`; 0 px / 8 states; 323 px with the target visible |
| `target.shape` | Discrete(2), default 1 | **unsupported** | — | moves `site_size`, `site_type`; 0 px / 8 states; 459 px with the target visible |

* `light.intensity` — mapped to light.diffuse_scale with factor 1/0.6 — the compiled model's own light diffuse. RANGE GAP at the dark end: our scale is Box(0.2, 1.8), so the reachable intensity is [0.120, 1.0] (1.8 * 0.6 clips to 1.0 at the bright end, which costs nothing because swm's range stops at 1.0), and swm's [0, 0.120) is not reachable. swm's default 0.7 is reached at scale 1.166667.
* `agent.fingertip_density` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: the fingertip's density is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `agent.proximal_density` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: the proximal link's density is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `floor.color` — UNSUPPORTED, and it is a RANGE gap rather than a default gap. swm writes the builtin ``grid`` TEXTURE's rgb1/rgb2 (2x3 = six free scalars) and recompiles. stable-envs has exactly one floor knob, ``material.grid``, a 3-vector that multiplies the material rgba — and ``_tex_rgb`` reduces the whole checker to its MEAN colour, so even the checker pattern is not a thing this renderer varies. Three degrees of freedom cannot cover six: the reachable set is a measure-zero subset of swm's Box, i.e. 0 % of the range. What IS reachable is the one value that matters for certification: swm's init_value [[.1,.2,.3],[.2,.3,.4]] is character-for-character common/materials.xml's own grid texture, so the REGISTERED DEFAULT floor is the compiled model's floor and needs no leaf at all (material.grid stays at its default [1,1,1]). Closing the range needs a texture edit — regenerating the builtin checker into m.tex_data and re-uploading — and ``_bundle`` holds one mujoco.Renderer per (domain, size) that is never closed, so that is a renderer change. ESCALATION CANDIDATE, carried forward from the Step-7 manipulator finding.
* `spinner.color` — the spinner's colour is a per-geom ``rgba`` OVERRIDE (the leaf moves geom_rgba, not mat_rgba) and this cell's colour knobs are material-level, one per material carried by a visible element. Measured 2,191 disagreeing pixels of 8x84x84, so the leaf is real; there is simply no leaf here that reaches a single geom's rgba.
* `spinner.density` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: the spinner's density is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `spinner.friction` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: the spinner joint's friction loss is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `target.shape` — the target site's box/sphere alternative. Measured 459 disagreeing pixels of 8x84x84 WITH the target visible, so the leaf is real. This cell declares SHAPE_KNOBS = {} — only DMCManipulatorBringBall carries a shape leaf — so the default (sphere, index 1) is the compiled model's own and reproduces exactly, and the box alternative is unreachable.

### `swm/HopperDMControl-v0`

cell(s): `DMCHopperHop` — registered size 84 px — **10 leaves: 2 mapped, 7 unsupported, 1 dead upstream, 0 not applicable**

The swm registered default IS the cell's default style: **0 of 56,448 px** differ over 8 states.

| leaf | swm type / range / default | class | stable-envs control | evidence |
|---|---|---|---|---|
| `agent.color` | Box[0, 1] shape (3,), default [0.7, 0.5, 0.3] | **mapped** | `material.self` | moves `mat_rgba`; 2,095 px / 8 states |
| `agent.foot_density` | Box[500, 1500], default 1000 | **unsupported** | — | moves `actuator_acc0`, `body_inertia`, `body_invweight0`...; 11,167 px / 8 states |
| `agent.foot_locked` | Discrete(2), default 1 | **dead_upstream** | — | no MjModel array moves; 0 px / 8 states |
| `agent.torso_density` | Box[500, 1500], default 1000 | **unsupported** | — | moves `actuator_acc0`, `body_inertia`, `body_invweight0`...; 10,918 px / 8 states |
| `floor.color` | Box[0, 1] shape (2, 3), default [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]] | **unsupported** | — | moves `tex_data`; 8,354 px / 8 states |
| `floor.friction` | Box[0, 1], default 1 | **unsupported** | — | moves `geom_friction`; 0 px / 8 states |
| `gravity.x` | Box[-5, 5], default 0 | **unsupported** | — | moves `__opt_gravity`; 0 px / 8 states |
| `gravity.y` | Box[-5, 5], default 0 | **unsupported** | — | moves `__opt_gravity`; 0 px / 8 states |
| `gravity.z` | Box[-20, 0], default -9.81 | **unsupported** | — | moves `__opt_gravity`; 0 px / 8 states |
| `light.intensity` | Box[0, 1], default 0.7 | **mapped** | `light.diffuse_scale` | moves `light_diffuse`; 8,758 px / 8 states |

* `light.intensity` — mapped to light.diffuse_scale with factor 1/0.7 — the compiled model's own light diffuse. RANGE GAP at the dark end: our scale is Box(0.2, 1.8), so the reachable intensity is [0.140, 1.0] (1.8 * 0.7 clips to 1.0 at the bright end, which costs nothing because swm's range stops at 1.0), and swm's [0, 0.140) is not reachable. swm's default 0.7 is reached at scale 1.000000.
* `agent.foot_density` — UNSUPPORTED, not merely physics: this cell renders through a mode="trackcom" camera, whose position comes from subtree_com, so a mass change MOVES THE CAMERA. Measured 11,167 disagreeing pixels of 56,448 over 8 states with qpos and geom_xpos bit-identical. The cell declares no mass leaf and its camera follows the compiled model's own COM, so the appearance is not reachable.
* `agent.foot_locked` — DEAD UPSTREAM: the leaf is declared by the swm wrapper and its own ``modify_mjcf_model`` never reads it (0 textual references outside the space literal). Measured: not one of the 110+ MjModel arrays changes between the leaf's two values, so it reaches neither a pixel nor a state.
* `agent.torso_density` — UNSUPPORTED, not merely physics: this cell renders through a mode="trackcom" camera, whose position comes from subtree_com, so a mass change MOVES THE CAMERA. Measured 10,918 disagreeing pixels of 56,448 over 8 states with qpos and geom_xpos bit-identical. The cell declares no mass leaf and its camera follows the compiled model's own COM, so the appearance is not reachable.
* `floor.color` — UNSUPPORTED, and it is a RANGE gap rather than a default gap. swm writes the builtin ``grid`` TEXTURE's rgb1/rgb2 (2x3 = six free scalars) and recompiles. stable-envs has exactly one floor knob, ``material.grid``, a 3-vector that multiplies the material rgba — and ``_tex_rgb`` reduces the whole checker to its MEAN colour, so even the checker pattern is not a thing this renderer varies. Three degrees of freedom cannot cover six: the reachable set is a measure-zero subset of swm's Box, i.e. 0 % of the range. What IS reachable is the one value that matters for certification: swm's init_value [[.1,.2,.3],[.2,.3,.4]] is character-for-character common/materials.xml's own grid texture, so the REGISTERED DEFAULT floor is the compiled model's floor and needs no leaf at all (material.grid stays at its default [1,1,1]). Closing the range needs a texture edit — regenerating the builtin checker into m.tex_data and re-uploading — and ``_bundle`` holds one mujoco.Renderer per (domain, size) that is never closed, so that is a renderer change. ESCALATION CANDIDATE, carried forward from the Step-7 manipulator finding.
* `floor.friction` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: the floor's sliding friction is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `gravity.x` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: gravity is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `gravity.y` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: gravity is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `gravity.z` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: gravity is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.

### `swm/HumanoidDMControl-v0`

cell(s): `DMCHumanoidStand` — registered size 84 px — **10 leaves: 2 mapped, 7 unsupported, 1 dead upstream, 0 not applicable**

The swm registered default IS the cell's default style: **0 of 56,448 px** differ over 8 states.

| leaf | swm type / range / default | class | stable-envs control | evidence |
|---|---|---|---|---|
| `agent.color` | Box[0, 1] shape (3,), default [0.7, 0.5, 0.3] | **mapped** | `material.self` | moves `mat_rgba`; 4,257 px / 8 states |
| `agent.left_knee_locked` | Discrete(2), default 1 | **dead_upstream** | — | no MjModel array moves; 0 px / 8 states |
| `agent.right_lower_arm_density` | Box[500, 1500], default 1000 | **unsupported** | — | moves `actuator_acc0`, `body_inertia`, `body_invweight0`...; 25,419 px / 8 states |
| `agent.torso_density` | Box[500, 1500], default 1000 | **unsupported** | — | moves `actuator_acc0`, `body_inertia`, `body_invweight0`...; 36,302 px / 8 states |
| `floor.color` | Box[0, 1] shape (2, 3), default [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]] | **unsupported** | — | moves `tex_data`; 54,599 px / 8 states |
| `floor.friction` | Box[0, 1], default 1 | **unsupported** | — | moves `geom_friction`; 0 px / 8 states |
| `gravity.x` | Box[-5, 5], default 0 | **unsupported** | — | moves `__opt_gravity`; 0 px / 8 states |
| `gravity.y` | Box[-5, 5], default 0 | **unsupported** | — | moves `__opt_gravity`; 0 px / 8 states |
| `gravity.z` | Box[-20, 0], default -9.81 | **unsupported** | — | moves `__opt_gravity`; 0 px / 8 states |
| `light.intensity` | Box[0, 1], default 0.7 | **mapped** | `light.diffuse_scale` | moves `light_diffuse`; 36,515 px / 8 states |

* `light.intensity` — mapped to light.diffuse_scale with factor 1/0.7 — the compiled model's own light diffuse. RANGE GAP at the dark end: our scale is Box(0.2, 1.8), so the reachable intensity is [0.140, 1.0] (1.8 * 0.7 clips to 1.0 at the bright end, which costs nothing because swm's range stops at 1.0), and swm's [0, 0.140) is not reachable. swm's default 0.7 is reached at scale 1.000000.
* `agent.left_knee_locked` — DEAD UPSTREAM: the leaf is declared by the swm wrapper and its own ``modify_mjcf_model`` never reads it (0 textual references outside the space literal). Measured: not one of the 110+ MjModel arrays changes between the leaf's two values, so it reaches neither a pixel nor a state.
* `agent.right_lower_arm_density` — UNSUPPORTED, not merely physics: this cell renders through a mode="trackcom" camera, whose position comes from subtree_com, so a mass change MOVES THE CAMERA. Measured 25,419 disagreeing pixels of 56,448 over 8 states with qpos and geom_xpos bit-identical. The cell declares no mass leaf and its camera follows the compiled model's own COM, so the appearance is not reachable.
* `agent.torso_density` — UNSUPPORTED, not merely physics: this cell renders through a mode="trackcom" camera, whose position comes from subtree_com, so a mass change MOVES THE CAMERA. Measured 36,302 disagreeing pixels of 56,448 over 8 states with qpos and geom_xpos bit-identical. The cell declares no mass leaf and its camera follows the compiled model's own COM, so the appearance is not reachable.
* `floor.color` — UNSUPPORTED, and it is a RANGE gap rather than a default gap. swm writes the builtin ``grid`` TEXTURE's rgb1/rgb2 (2x3 = six free scalars) and recompiles. stable-envs has exactly one floor knob, ``material.grid``, a 3-vector that multiplies the material rgba — and ``_tex_rgb`` reduces the whole checker to its MEAN colour, so even the checker pattern is not a thing this renderer varies. Three degrees of freedom cannot cover six: the reachable set is a measure-zero subset of swm's Box, i.e. 0 % of the range. What IS reachable is the one value that matters for certification: swm's init_value [[.1,.2,.3],[.2,.3,.4]] is character-for-character common/materials.xml's own grid texture, so the REGISTERED DEFAULT floor is the compiled model's floor and needs no leaf at all (material.grid stays at its default [1,1,1]). Closing the range needs a texture edit — regenerating the builtin checker into m.tex_data and re-uploading — and ``_bundle`` holds one mujoco.Renderer per (domain, size) that is never closed, so that is a renderer change. ESCALATION CANDIDATE, carried forward from the Step-7 manipulator finding.
* `floor.friction` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: the floor's sliding friction is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `gravity.x` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: gravity is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `gravity.y` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: gravity is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `gravity.z` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: gravity is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.

### `swm/ManipulatorDMControl-v0`

cell(s): `DMCManipulatorBringBall` — registered size 224 px — **9 leaves: 5 mapped, 3 unsupported, 1 dead upstream, 0 not applicable**

The swm REGISTERED DEFAULT is a different appearance from the compiled model's and is reached through `light.diffuse_scale`, `target_ball.shape`, `visibility.target_ball` (9,109 of 401,408 px differ from the cell's default style over 8 states).

| leaf | swm type / range / default | class | stable-envs control | evidence |
|---|---|---|---|---|
| `agent.color` | Box[0, 1] shape (3,), default [0.7, 0.5, 0.3] | **mapped** | `material.self` | moves `mat_rgba`; 6,906 px / 8 states; 6,902 px with the target visible |
| `agent.hand_density` | Box[500, 1500], default 1000 | **unsupported** | — | moves `actuator_acc0`, `body_inertia`, `body_invweight0`...; 0 px / 8 states |
| `agent.upper_arm_density` | Box[500, 1500], default 1000 | **unsupported** | — | moves `actuator_acc0`, `body_inertia`, `body_invweight0`...; 0 px / 8 states |
| `agent.upper_arm_length` | Box[500, 1500], default 1000 | **dead_upstream** | — | no MjModel array moves; 0 px / 8 states |
| `floor.color` | Box[0, 1] shape (2, 3), default [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]] | **unsupported** | — | moves `tex_data`; 6,408 px / 8 states |
| `light.intensity` | Box[0, 1], default 0.7 | **mapped** | `light.diffuse_scale` | moves `light_diffuse`; 9,411 px / 8 states; 9,494 px with the target visible |
| `rendering.render_target` | Discrete(2), default 0 | **mapped** | `visibility.target_ball` | moves `mat_rgba`; 274 px / 8 states |
| `target.color` | Box[0, 1] shape (3,), default [0.6, 0.3, 0.3] | **mapped** | `material.target` | moves `mat_rgba`; 0 px / 8 states; 270 px with the target visible |
| `target.shape` | Discrete(2), default 1 | **mapped** | `target_ball.shape` | moves `body_inertia`, `body_mass`, `body_subtreemass`...; 0 px / 8 states; 371 px with the target visible |

* `light.intensity` — mapped to light.diffuse_scale with factor 1/0.6 — the compiled model's own light diffuse. RANGE GAP at the dark end: our scale is Box(0.2, 1.8), so the reachable intensity is [0.120, 1.0] (1.8 * 0.6 clips to 1.0 at the bright end, which costs nothing because swm's range stops at 1.0), and swm's [0, 0.120) is not reachable. swm's default 0.7 is reached at scale 1.166667.
* `rendering.render_target` — MAPPED to visibility.target_ball. swm's registered default is 0 (the target is NOT drawn), and our leaf's default is 1, so the registered default is reached through the leaf. Measured 274 disagreeing pixels of 8x224x224 between the two values upstream.
* `target.shape` — MAPPED at both alternatives. swm's Discrete(2) is (0 = box .015, 1 = sphere .015); ours is Discrete(3) whose index 2 is the compiled model's own sphere .02 and is the DEFAULT, which is what keeps the default style a render of the compiled model. swm's registered default is 1, so reaching it is a leaf write, not a no-op.
* `agent.hand_density` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: the hand's density is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `agent.upper_arm_density` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: the upper arm's density is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `agent.upper_arm_length` — DEAD UPSTREAM: the leaf is declared by the swm wrapper and its own ``modify_mjcf_model`` never reads it (0 textual references outside the space literal). Measured: not one of the 110+ MjModel arrays changes between the leaf's two values, so it reaches neither a pixel nor a state. The swm source carries ``# TODO should modify length of arm`` at the point it would be read; recorded in Step 7 and re-measured here.
* `floor.color` — UNSUPPORTED, and it is a RANGE gap rather than a default gap. swm writes the builtin ``grid`` TEXTURE's rgb1/rgb2 (2x3 = six free scalars) and recompiles. stable-envs has exactly one floor knob, ``material.grid``, a 3-vector that multiplies the material rgba — and ``_tex_rgb`` reduces the whole checker to its MEAN colour, so even the checker pattern is not a thing this renderer varies. Three degrees of freedom cannot cover six: the reachable set is a measure-zero subset of swm's Box, i.e. 0 % of the range. What IS reachable is the one value that matters for certification: swm's init_value [[.1,.2,.3],[.2,.3,.4]] is character-for-character common/materials.xml's own grid texture, so the REGISTERED DEFAULT floor is the compiled model's floor and needs no leaf at all (material.grid stays at its default [1,1,1]). Closing the range needs a texture edit — regenerating the builtin checker into m.tex_data and re-uploading — and ``_bundle`` holds one mujoco.Renderer per (domain, size) that is never closed, so that is a renderer change. ESCALATION CANDIDATE, carried forward from the Step-7 manipulator finding.

### `swm/PendulumDMControl-v0`

cell(s): `DMCPendulumSwingup` — registered size 84 px — **9 leaves: 2 mapped, 4 unsupported, 3 dead upstream, 0 not applicable**

The swm registered default IS the cell's default style: **0 of 56,448 px** differ over 8 states.

| leaf | swm type / range / default | class | stable-envs control | evidence |
|---|---|---|---|---|
| `agent.color` | Box[0, 1] shape (3,), default [0.7, 0.5, 0.3] | **mapped** | `material.self` | moves `mat_rgba`; 572 px / 8 states |
| `agent.mass_density` | Box[500, 1500], default 1000 | **dead_upstream** | — | no MjModel array moves; 0 px / 8 states |
| `agent.mass_shape` | Discrete(2), default 1 | **dead_upstream** | — | no MjModel array moves; 0 px / 8 states |
| `agent.pole_density` | Box[500, 1500], default 1000 | **dead_upstream** | — | no MjModel array moves; 0 px / 8 states |
| `floor.color` | Box[0, 1] shape (2, 3), default [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]] | **unsupported** | — | moves `tex_data`; 46,691 px / 8 states |
| `gravity.x` | Box[-5, 5], default 0 | **unsupported** | — | moves `__opt_gravity`; 0 px / 8 states |
| `gravity.y` | Box[-5, 5], default 0 | **unsupported** | — | moves `__opt_gravity`; 0 px / 8 states |
| `gravity.z` | Box[-20, 0], default -9.81 | **unsupported** | — | moves `__opt_gravity`; 0 px / 8 states |
| `light.intensity` | Box[0, 1], default 0.7 | **mapped** | `light.diffuse_scale` | moves `light_diffuse`; 45,267 px / 8 states |

* `light.intensity` — mapped to light.diffuse_scale with factor 1/0.7 — the compiled model's own light diffuse. RANGE GAP at the dark end: our scale is Box(0.2, 1.8), so the reachable intensity is [0.140, 1.0] (1.8 * 0.7 clips to 1.0 at the bright end, which costs nothing because swm's range stops at 1.0), and swm's [0, 0.140) is not reachable. swm's default 0.7 is reached at scale 1.000000.
* `agent.mass_density` — DEAD UPSTREAM: the wrapper writes the value onto the geom's ``density`` and recompiles, but pendulum.xml's <geom name="mass" ... mass="1"/> carries an EXPLICIT mass, and in MuJoCo an explicit mass wins over density. Measured: not one MjModel array changes, body_mass included.
* `agent.mass_shape` — DEAD UPSTREAM: the leaf is declared by the swm wrapper and its own ``modify_mjcf_model`` never reads it (0 textual references outside the space literal). Measured: not one of the 110+ MjModel arrays changes between the leaf's two values, so it reaches neither a pixel nor a state.
* `agent.pole_density` — DEAD UPSTREAM: the wrapper writes the value onto the geom's ``density`` and recompiles, but pendulum.xml's <geom name="pole" ... mass="0"/> carries an EXPLICIT mass, and in MuJoCo an explicit mass wins over density. Measured: not one MjModel array changes, body_mass included.
* `floor.color` — UNSUPPORTED, and it is a RANGE gap rather than a default gap. swm writes the builtin ``grid`` TEXTURE's rgb1/rgb2 (2x3 = six free scalars) and recompiles. stable-envs has exactly one floor knob, ``material.grid``, a 3-vector that multiplies the material rgba — and ``_tex_rgb`` reduces the whole checker to its MEAN colour, so even the checker pattern is not a thing this renderer varies. Three degrees of freedom cannot cover six: the reachable set is a measure-zero subset of swm's Box, i.e. 0 % of the range. What IS reachable is the one value that matters for certification: swm's init_value [[.1,.2,.3],[.2,.3,.4]] is character-for-character common/materials.xml's own grid texture, so the REGISTERED DEFAULT floor is the compiled model's floor and needs no leaf at all (material.grid stays at its default [1,1,1]). Closing the range needs a texture edit — regenerating the builtin checker into m.tex_data and re-uploading — and ``_bundle`` holds one mujoco.Renderer per (domain, size) that is never closed, so that is a renderer change. ESCALATION CANDIDATE, carried forward from the Step-7 manipulator finding.
* `gravity.x` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: gravity is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `gravity.y` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: gravity is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `gravity.z` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: gravity is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.

### `swm/QuadrupedDMControl-v0`

cell(s): `DMCQuadrupedWalk` — registered size 224 px — **10 leaves: 2 mapped, 7 unsupported, 1 dead upstream, 0 not applicable**

The swm registered default IS the cell's default style: **0 of 401,408 px** differ over 8 states.

| leaf | swm type / range / default | class | stable-envs control | evidence |
|---|---|---|---|---|
| `agent.color` | Box[0, 1] shape (3,), default [0.6, 0.3, 0.3] | **mapped** | `material.self` | moves `mat_rgba`; 29,758 px / 8 states |
| `agent.foot_back_left_density` | Box[500, 1500], default 1000 | **unsupported** | — | moves `actuator_acc0`, `body_inertia`, `body_invweight0`...; 137,533 px / 8 states |
| `agent.knee_back_left_locked` | Discrete(2), default 1 | **dead_upstream** | — | no MjModel array moves; 0 px / 8 states |
| `agent.torso_density` | Box[500, 1500], default 1000 | **unsupported** | — | moves `actuator_acc0`, `body_inertia`, `body_invweight0`...; 135,622 px / 8 states |
| `floor.color` | Box[0, 1] shape (2, 3), default [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]] | **unsupported** | — | moves `tex_data`; 321,610 px / 8 states |
| `floor.friction` | Box[0, 1], default 1 | **unsupported** | — | moves `geom_friction`; 0 px / 8 states |
| `gravity.x` | Box[-5, 5], default 0 | **unsupported** | — | moves `__opt_gravity`; 0 px / 8 states |
| `gravity.y` | Box[-5, 5], default 0 | **unsupported** | — | moves `__opt_gravity`; 0 px / 8 states |
| `gravity.z` | Box[-20, 0], default -9.81 | **unsupported** | — | moves `__opt_gravity`; 0 px / 8 states |
| `light.intensity` | Box[0, 1], default 0.7 | **mapped** | `light.diffuse_scale` | moves `light_diffuse`; 263,946 px / 8 states |

* `light.intensity` — mapped to light.diffuse_scale with factor 1/0.7 — the compiled model's own light diffuse. RANGE GAP at the dark end: our scale is Box(0.2, 1.8), so the reachable intensity is [0.140, 1.0] (1.8 * 0.7 clips to 1.0 at the bright end, which costs nothing because swm's range stops at 1.0), and swm's [0, 0.140) is not reachable. swm's default 0.7 is reached at scale 1.000000.
* `agent.foot_back_left_density` — UNSUPPORTED, not merely physics: this cell renders through a mode="trackcom" camera, whose position comes from subtree_com, so a mass change MOVES THE CAMERA. Measured 137,533 disagreeing pixels of 401,408 over 8 states with qpos and geom_xpos bit-identical. The cell declares no mass leaf and its camera follows the compiled model's own COM, so the appearance is not reachable.
* `agent.knee_back_left_locked` — DEAD UPSTREAM: the leaf is declared by the swm wrapper and its own ``modify_mjcf_model`` never reads it (0 textual references outside the space literal). Measured: not one of the 110+ MjModel arrays changes between the leaf's two values, so it reaches neither a pixel nor a state.
* `agent.torso_density` — UNSUPPORTED, not merely physics: this cell renders through a mode="trackcom" camera, whose position comes from subtree_com, so a mass change MOVES THE CAMERA. Measured 135,622 disagreeing pixels of 401,408 over 8 states with qpos and geom_xpos bit-identical. The cell declares no mass leaf and its camera follows the compiled model's own COM, so the appearance is not reachable.
* `floor.color` — UNSUPPORTED, and it is a RANGE gap rather than a default gap. swm writes the builtin ``grid`` TEXTURE's rgb1/rgb2 (2x3 = six free scalars) and recompiles. stable-envs has exactly one floor knob, ``material.grid``, a 3-vector that multiplies the material rgba — and ``_tex_rgb`` reduces the whole checker to its MEAN colour, so even the checker pattern is not a thing this renderer varies. Three degrees of freedom cannot cover six: the reachable set is a measure-zero subset of swm's Box, i.e. 0 % of the range. What IS reachable is the one value that matters for certification: swm's init_value [[.1,.2,.3],[.2,.3,.4]] is character-for-character common/materials.xml's own grid texture, so the REGISTERED DEFAULT floor is the compiled model's floor and needs no leaf at all (material.grid stays at its default [1,1,1]). Closing the range needs a texture edit — regenerating the builtin checker into m.tex_data and re-uploading — and ``_bundle`` holds one mujoco.Renderer per (domain, size) that is never closed, so that is a renderer change. ESCALATION CANDIDATE, carried forward from the Step-7 manipulator finding.
* `floor.friction` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: the floor's sliding friction is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `gravity.x` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: gravity is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `gravity.y` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: gravity is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `gravity.z` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: gravity is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.

### `swm/ReacherDMControl-v0`

cell(s): `DMCReacherHard` — registered size 84 px — **9 leaves: 4 mapped, 4 unsupported, 1 dead upstream, 0 not applicable**

The swm REGISTERED DEFAULT is a different appearance from the compiled model's and is reached through `visibility.target` (115 of 56,448 px differ from the cell's default style over 8 states).

| leaf | swm type / range / default | class | stable-envs control | evidence |
|---|---|---|---|---|
| `agent.arm_density` | Box[500, 1500], default 1000 | **unsupported** | — | moves `actuator_acc0`, `body_inertia`, `body_invweight0`...; 0 px / 8 states |
| `agent.color` | Box[0, 1] shape (3,), default [0.7, 0.5, 0.3] | **mapped** | `material.self` | moves `mat_rgba`; 850 px / 8 states; 846 px with the target visible |
| `agent.finger_density` | Box[500, 1500], default 1000 | **unsupported** | — | moves `actuator_acc0`, `body_inertia`, `body_invweight0`...; 0 px / 8 states |
| `agent.finger_locked` | Discrete(2), default 1 | **dead_upstream** | — | no MjModel array moves; 0 px / 8 states |
| `floor.color` | Box[0, 1] shape (2, 3), default [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]] | **unsupported** | — | moves `tex_data`; 53,246 px / 8 states; 53,177 px with the target visible |
| `light.intensity` | Box[0, 1], default 0.7 | **mapped** | `light.diffuse_scale` | moves `light_diffuse`; 56,448 px / 8 states |
| `rendering.render_target` | Discrete(2), default 0 | **mapped** | `visibility.target` | moves `mat_rgba`; 150 px / 8 states |
| `target.color` | Box[0, 1] shape (3,), default [0.6, 0.3, 0.3] | **mapped** | `material.target` | moves `mat_rgba`; 0 px / 8 states; 148 px with the target visible |
| `target.shape` | Discrete(2), default 1 | **unsupported** | — | moves `geom_rbound`, `geom_size`, `geom_type`; 0 px / 8 states; 236 px with the target visible |

* `light.intensity` — mapped to light.diffuse_scale with factor 1/0.7 — the compiled model's own light diffuse. RANGE GAP at the dark end: our scale is Box(0.2, 1.8), so the reachable intensity is [0.140, 1.0] (1.8 * 0.7 clips to 1.0 at the bright end, which costs nothing because swm's range stops at 1.0), and swm's [0, 0.140) is not reachable. swm's default 0.7 is reached at scale 1.000000.
* `agent.arm_density` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: the arm's density is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `agent.finger_density` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: the finger's density is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `agent.finger_locked` — DEAD UPSTREAM: the leaf is declared by the swm wrapper and its own ``modify_mjcf_model`` never reads it (0 textual references outside the space literal). Measured: not one of the 110+ MjModel arrays changes between the leaf's two values, so it reaches neither a pixel nor a state.
* `floor.color` — UNSUPPORTED, and it is a RANGE gap rather than a default gap. swm writes the builtin ``grid`` TEXTURE's rgb1/rgb2 (2x3 = six free scalars) and recompiles. stable-envs has exactly one floor knob, ``material.grid``, a 3-vector that multiplies the material rgba — and ``_tex_rgb`` reduces the whole checker to its MEAN colour, so even the checker pattern is not a thing this renderer varies. Three degrees of freedom cannot cover six: the reachable set is a measure-zero subset of swm's Box, i.e. 0 % of the range. What IS reachable is the one value that matters for certification: swm's init_value [[.1,.2,.3],[.2,.3,.4]] is character-for-character common/materials.xml's own grid texture, so the REGISTERED DEFAULT floor is the compiled model's floor and needs no leaf at all (material.grid stays at its default [1,1,1]). Closing the range needs a texture edit — regenerating the builtin checker into m.tex_data and re-uploading — and ``_bundle`` holds one mujoco.Renderer per (domain, size) that is never closed, so that is a renderer change. ESCALATION CANDIDATE, carried forward from the Step-7 manipulator finding.
* `target.shape` — the target geom's box/sphere alternative. Measured 236 disagreeing pixels of 8x84x84 WITH the target visible, so the leaf is real. This cell declares SHAPE_KNOBS = {}, so the default (sphere, index 1) is the compiled model's own and reproduces exactly, and the box alternative is unreachable.

### `swm/WalkerDMControl-v0`

cell(s): `DMCWalkerWalk` — registered size 84 px — **11 leaves: 2 mapped, 8 unsupported, 1 dead upstream, 0 not applicable**

The swm registered default IS the cell's default style: **0 of 56,448 px** differ over 8 states.

| leaf | swm type / range / default | class | stable-envs control | evidence |
|---|---|---|---|---|
| `agent.color` | Box[0, 1] shape (3,), default [0.7, 0.5, 0.3] | **mapped** | `material.self` | moves `mat_rgba`; 7,074 px / 8 states |
| `agent.left_foot_density` | Box[500, 1500], default 1000 | **unsupported** | — | moves `actuator_acc0`, `body_inertia`, `body_invweight0`...; 21,374 px / 8 states |
| `agent.right_knee_locked` | Discrete(2), default 1 | **dead_upstream** | — | no MjModel array moves; 0 px / 8 states |
| `agent.torso_density` | Box[500, 1500], default 1000 | **unsupported** | — | moves `actuator_acc0`, `body_inertia`, `body_invweight0`...; 26,328 px / 8 states |
| `floor.color` | Box[0, 1] shape (2, 3), default [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]] | **unsupported** | — | moves `tex_data`; 24,002 px / 8 states |
| `floor.friction` | Box[0, 1], default 1 | **unsupported** | — | moves `geom_friction`; 0 px / 8 states |
| `floor.rotation_y` | Box[-10, 10], default 0 | **unsupported** | — | moves `bvh_aabb`, `geom_quat`, `geom_sameframe`; 24,004 px / 8 states |
| `gravity.x` | Box[-5, 5], default 0 | **unsupported** | — | moves `__opt_gravity`; 0 px / 8 states |
| `gravity.y` | Box[-5, 5], default 0 | **unsupported** | — | moves `__opt_gravity`; 0 px / 8 states |
| `gravity.z` | Box[-20, 0], default -9.81 | **unsupported** | — | moves `__opt_gravity`; 0 px / 8 states |
| `light.intensity` | Box[0, 1], default 0.7 | **mapped** | `light.diffuse_scale` | moves `light_diffuse`; 24,753 px / 8 states |

* `light.intensity` — mapped to light.diffuse_scale with factor 1/0.7 — the compiled model's own light diffuse. RANGE GAP at the dark end: our scale is Box(0.2, 1.8), so the reachable intensity is [0.140, 1.0] (1.8 * 0.7 clips to 1.0 at the bright end, which costs nothing because swm's range stops at 1.0), and swm's [0, 0.140) is not reachable. swm's default 0.7 is reached at scale 1.000000.
* `agent.left_foot_density` — UNSUPPORTED, not merely physics: this cell renders through a mode="trackcom" camera, whose position comes from subtree_com, so a mass change MOVES THE CAMERA. Measured 21,374 disagreeing pixels of 56,448 over 8 states with qpos and geom_xpos bit-identical. The cell declares no mass leaf and its camera follows the compiled model's own COM, so the appearance is not reachable.
* `agent.right_knee_locked` — DEAD UPSTREAM: the leaf is declared by the swm wrapper and its own ``modify_mjcf_model`` never reads it (0 textual references outside the space literal). Measured: not one of the 110+ MjModel arrays changes between the leaf's two values, so it reaches neither a pixel nor a state.
* `agent.torso_density` — UNSUPPORTED, not merely physics: this cell renders through a mode="trackcom" camera, whose position comes from subtree_com, so a mass change MOVES THE CAMERA. Measured 26,328 disagreeing pixels of 56,448 over 8 states with qpos and geom_xpos bit-identical. The cell declares no mass leaf and its camera follows the compiled model's own COM, so the appearance is not reachable.
* `floor.color` — UNSUPPORTED, and it is a RANGE gap rather than a default gap. swm writes the builtin ``grid`` TEXTURE's rgb1/rgb2 (2x3 = six free scalars) and recompiles. stable-envs has exactly one floor knob, ``material.grid``, a 3-vector that multiplies the material rgba — and ``_tex_rgb`` reduces the whole checker to its MEAN colour, so even the checker pattern is not a thing this renderer varies. Three degrees of freedom cannot cover six: the reachable set is a measure-zero subset of swm's Box, i.e. 0 % of the range. What IS reachable is the one value that matters for certification: swm's init_value [[.1,.2,.3],[.2,.3,.4]] is character-for-character common/materials.xml's own grid texture, so the REGISTERED DEFAULT floor is the compiled model's floor and needs no leaf at all (material.grid stays at its default [1,1,1]). Closing the range needs a texture edit — regenerating the builtin checker into m.tex_data and re-uploading — and ``_bundle`` holds one mujoco.Renderer per (domain, size) that is never closed, so that is a renderer change. ESCALATION CANDIDATE, carried forward from the Step-7 manipulator finding.
* `floor.friction` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: the floor's sliding friction is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `floor.rotation_y` — walker is the one domain whose swm wrapper TILTS the floor: the leaf writes the floor geom's quat and moves 24,004 disagreeing pixels of 8x84x84 at 9 degrees. The cell bakes the floor plane from the compiled model and declares no pose leaf, so the tilt is not reachable. Its default, 0, is the compiled model's own.
* `gravity.x` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: gravity is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `gravity.y` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: gravity is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
* `gravity.z` — PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: gravity is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable is the trajectory.
