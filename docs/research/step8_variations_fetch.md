# Step 8 — VariationMaps for the twelve Fetch ids

`stable_worldmodel` reaches four gymnasium-robotics scenes through twelve `swm/Fetch*-v3` ids. This
page is the leaf-by-leaf account of how their variation spaces relate to ours: what we render, what
we deliberately do not, what the official renderer itself ignores, and what is a state rather than an
appearance. The machine-readable form, with every measurement, is
`examples/results/step8/variations-fetch/leaf_classification.json`; the map itself is
`stable_envs/compat/variation_maps/fetch.py`, registered into `compat.worldmodel.VARIATION_MAPS` by
one line.

## One family, not twelve inventions

The `Dense` suffix selects a reward and the `Dict` suffix an observation container, and neither
reaches a pixel. Enumerated from the live `FetchWrapper` rather than assumed: the twelve variation
spaces are **identical per scene** — 11 leaves on Push, Slide and PickAndPlace, 8 on Reach, which has
no block. So the map is built from one vocabulary plus a per-scene difference table, and
`tests/unit/test_variation_maps_fetch.py::test_the_three_suffixes_of_a_scene_share_one_map_body`
asserts that the only field which differs between `FetchPush`, `FetchPushDense` and `FetchPushDict`
is `swm_id`. That is what "the aliases do not duplicate renderer logic" means here, checked rather
than claimed — and it is already true one level down, where all three ids are `FetchBase` subclasses
sharing a scene's compiled model, baked plate and render context.

## The family leaf table

Applies to all twelve ids. The two per-task differences are marked and nothing else varies.

| swm leaf | class | our leaf | units | evidence |
|---|---|---|---|---|
| `table.color` | **mapped** | `table.color` | identical, `[0,1]³` | same geom (22, `table_mat`) |
| `object.color` | **mapped** ¹ | `object.color` | identical, `[0,1]³` | same geom (23, `block_mat` / `puck_mat`); 0.07–0.12 % of the frame |
| `background.color` | **partial** | `floor.color` | identical, `[0,1]³` | floor 9.2–9.4 % of the frame, reproduced exactly; skybox a further 61.7–73.0 %, not rendered |
| `light.intensity` | **mapped, scaled** | `light.diffuse` | `ours = theirs × 1.25` | the model's `light_diffuse` is 0.8 on all four scenes |
| `camera.angle_delta` | **dead upstream** | — | — | 0/255 over 0 % of the frame at ±10° on all four scenes |
| `rendering.transparent_arm` | **unsupported (backend)** | — | — | live upstream at 9.3–9.4 % of pixels; `mujoco_warp` has no alpha path |
| `agent.start_position` | **not applicable** (state) | *state:* arm `qpos` | — | a reset distribution in swm |
| `goal.start_position` | **not applicable** (state) | *state:* `goal_x/y/z` | — | a reset distribution in swm |
| `block.start_position` ² | **not applicable** (state) | *state:* `object_x/y` | — | a reset distribution in swm |
| `block.angle` ² | **not applicable** (state) | *state:* `object_quat_*` | — | a reset distribution in swm |
| `block.mass` ² | **not applicable** (dynamics) | — | — | 0/255 at 0.01 kg vs 50 kg on all three object scenes |

¹ on Reach this leaf is **dead upstream** instead — see below. ² absent from Reach's space.

Totals over the twelve ids and their 123 leaves: **33 mapped, 12 partial, 12 unsupported,
15 dead upstream, 51 not applicable, 0 undocumented.** The twelve `partial` leaves are
`background.color` on each of the twelve ids; `VariationMap.classification()` is where that label
comes from, so `check_mapping`, the Gate-F matrix and every generated support table inherit it and
none of them can count the leaf as full support. Regenerating
`examples/results/step8/variations-fetch/leaf_classification.json` is asserted against the maps by
`tests/unit/test_variation_maps_fetch.py::test_the_committed_classification_artifact_agrees_with_the_maps`.

### Per-task differences, and there are exactly two

Reach has no object, and that is the whole of it:

1. `block.start_position`, `block.angle` and `block.mass` are **absent** from Reach's variation
   space. Nothing to classify.
2. `object.color` is still **offered** on Reach and reaches nothing. The wrapper resolves the object
   through the `block_mat`/`puck_mat` material and `reach.xml` compiles neither onto a geom, so the
   write lands on an empty geom list. Measured: two maximally different object colours on Reach give
   **0/255 over 0 % of the frame**; on the other three scenes the same pair moves 240/255 over
   0.07–0.12 % of the frame. Reach's map therefore classifies it `dead_upstream` and our Reach class
   correctly has no `object.color` leaf at all.

Push, Slide and PickAndPlace are identical to each other leaf for leaf; the only difference between
them is which material carries the object (`block_mat` on Push and PickAndPlace, `puck_mat` on
Slide), and both resolve to geom 23 in their respective models.

## What we do not carry, and why

### `camera.angle_delta` — dead upstream

The wrapper rotates `model.cam_pos` for every **model** camera. `MujocoFetchEnv` renders through the
**free** camera it configures in `DEFAULT_CAMERA_CONFIG` — which `FetchBase._check_camera` already
asserts independently, because our `se:cam` is built from exactly those four numbers — and never
selects a model camera. The Fetch models carry four model cameras (`ncam == 4`), which is what makes
this look alive from the model side, and `model.cam_pos` really does move by 0.2257 when the leaf is
set. The frame does not: **0/255 over 0 % of the frame at ±10° on all four scenes.**

We *could* honour it: `camera.azimuth_delta`, `camera.elevation_delta` and `camera.distance_scale`
are live knobs of ours that move the frame correctly. Mapping the swm leaf onto them would make our
frame move where the reference's does not — turning a truthful "the reference ignores this" into a
silent fidelity divergence — so it is classified, not fabricated. Same treatment as Rocket's
`environment.sky_color`.

### `rendering.transparent_arm` — unsupported by the backend

Real in MuJoCo: 9.3–9.4 % of pixels change between 0 and 1. It sets `geom_rgba[i][3] = 0.3` on every
`robot0:` geom, and `mujoco_warp`'s render kernel reads only `color[0..2]` and packs a constant 255 —
it has no alpha path at all. That is the same measured fact recorded in `FetchBase.parity_note`,
which is why this family *hides* the four translucent `robot0:mocap` markers instead of painting them
opaque saturated green. Approximating the leaf by darkening the arm would move pixels the alpha-blind
renderer cannot get right anyway, so it is declared unsupported with the mechanism named.

Classification: `DOCUMENTED_LIMITATION`. The fix is an alpha path in a first-party DeepMind
dependency, not a stable-envs change.

### `background.color` — **partial**, with a measured residual and a recorded ruling

swm's "background" is two things at once: the `floor_mat` geoms **and** the skybox texture,
recoloured by overwriting `model.tex_data` and re-uploading it. Measured split at 480 px, floor
versus floor-plus-skybox:

| scene | floor only | floor + skybox | skybox alone |
|---|---|---|---|
| Reach | 9.18 % | 81.74 % | 73.03 % |
| Push | 9.18 % | 81.90 % | 73.19 % |
| Slide | 9.45 % | 70.69 % | 61.68 % |
| PickAndPlace | 9.18 % | 81.73 % | 73.03 % |

We render the floor half exactly and the skybox not at all, because `mujoco_warp` uploads textures
when a render context is built, so a per-value skybox costs **one render context per value** — the
one thing this repo's renderer rules forbid scaling. `FetchBase.make_variation_space` already records
that reason for not offering a skybox knob at all.

The leaf is neither dropped nor counted as full support. It is classified **`partial`**: the rename
is real — the floor half is genuinely what swm writes, and a caller asking for a dark background gets
a dark floor — and the skybox half is genuinely missing, and the classification says both.

That is the coordinator's ruling (Sami), recorded here because it decides a contract question this
workstream raised under plan §4: **a feasible, semantically required leaf is not declared
unsupported merely because a renderer change is needed.** The skybox half is a renderer *capability*
— a texture upload that does not cost a render context per value — and it is aggregated with the
other families' capability needs after Wave 1 rather than implemented here. Until that decision
lands, `partial` plus the measured residual is the truthful label, and it is carried by
`VariationMap.partial` (added centrally by the coordinator after Workstream A merged) rather than by
prose beside the map.

### Four leaves are our state, not our style

`agent.start_position`, `goal.start_position`, `block.start_position` and `block.angle` are reset
*distributions* in swm — the wrapper writes them into `initial_gripper_xpos`, `data.qpos` and
`env.goal` at reset time. All four are live pixels, and all four are dimensions of our state vector
(`qpos + goal`). This is the classification PushT's `block.angle` and `agent.start_position` already
carry, and it is strictly more expressive: a caller sets the state directly instead of sampling from
a distribution.

`block.mass` is pure dynamics and reaches no pixel: **0/255 at 0.01 kg versus 50 kg** on all three
object scenes.

## What we have that swm does not

`goal.color` (the red target sphere), `arm.color` (the five `robot0:blue` links), and the three
working camera knobs `camera.distance_scale` / `camera.azimuth_delta` / `camera.elevation_delta`.
swm offers no counterpart for any of them: `rendering.transparent_arm` is the nearest thing to an
arm-appearance leaf and it is an alpha, not a colour.

## The one unit shift, and why the default still reproduces exactly

swm assigns the light's diffuse **absolutely** (`light_diffuse[0][:3] = [i,i,i]`, `Box(0, 1)`,
declared default 0.7). Ours is a **multiplier** on the model's own `light_diffuse`, which is 0.8 on
all four scenes, with default 1.0 — because that is what makes "our default is the official
appearance" an identity a test can assert. So `ours = theirs × 1.25`, and:

- the factor is exactly representable in binary, so swm's 0.8 round-trips to our 1.0 with no error
  and `test_registered_default_is_byte_identical_through_the_mapping` can use `array_equal` rather
  than `allclose`;
- swm's full `[0, 1]` range maps onto `[0, 1.25]`, comfortably inside our `Box(0, 2)`. **The whole
  swm range is representable** — nothing clips, and no supported range was widened to make that true.

## Upstream findings recorded while measuring

These belong in the Step-8 upstream findings bundle. None of them is a stable-envs defect and none of
them changes the mapping.

1. **The visual leaves permanently unbind three materials.**
   `FetchWrapper._apply_visual_variations` sets `model.geom_matid[i] = -1` on the table, floor and
   object geoms the first time **any** of its six visual leaves is active, for the life of that env.
   Those geoms then render at the compiler default 0.5 grey instead of their material colours (table
   0.93, floor 0.2, object 0.2). Measured with `camera.angle_delta` — a dead leaf — held at its own
   default, so the whole difference is the unbind: **19.0–32.3 % of the frame, up to 102/255.** Since
   `DEFAULT_VARIATIONS` activates five visual leaves, a plain `reset()` on a swm Fetch id always
   triggers it.

2. **swm's declared leaf defaults are not the official appearance.** `table.color` 0.3 against the
   model's 0.93; `object.color` (0.8, 0.1, 0.1) against 0.2 grey; `background.color` 0.1 against a
   floor at 0.2 plus a textured skybox; `light.intensity` 0.7 against the model's 0.8;
   `agent.start_position` (1.3418, 0.7491) against the env's own initial gripper xy (up to 55/255 on
   Reach). Consequence for this workstream: `style_from_variation` carries a swm config's values
   through unchanged, which is right, and *our* defaults — read off the compiled model — are what
   reproduce the official picture. A parity number computed through the swm wrapper is measuring the
   wrapper, not stable-envs.

3. **`variation_values` entries are used unconverted.** A plain Python list for `background.color`
   raises `AttributeError: 'list' object has no attribute 'astype'` inside
   `_apply_visual_variations`. Callers must pass `np.ndarray`.

4. **`compat.worldmodel.check_mapping` cannot read a Fetch variation space.** It reaches for
   `gymnasium.make(swm_id).unwrapped.variation_space`; `FetchWrapper` is a `gym.Wrapper` around
   `MujocoFetchEnv`, so `.unwrapped` is the official env and carries no `variation_space` —
   `AttributeError` for all twelve ids. `variation_maps.fetch.swm_variation_space` walks the wrapper
   chain instead, so this module's own tests do not depend on the fix. A one-line change in a shared
   file this workstream does not own; handed to the coordinator.

5. **gymnasium-robotics + numpy 2.5.3 + mujoco 3.12 cannot construct a Fetch env.**
   `mujoco_utils.set_joint_qpos` asserts `model.jnt_type[i] in (mjJNT_HINGE, mjJNT_SLIDE)`; the
   needle is an `np.int32` and mujoco's pybind11 `mjtJoint.__eq__` returns `NotImplemented` for it,
   so `in` is `False` where `==` is `True`. Worked around in a scratchpad probe shim only — nothing in
   this repository changes for it — and Oscar's `se-s3b-fetch311` (mujoco 3.11.0), which is the
   interpreter the frozen Fetch corpora were measured in, is unaffected.

## How the claims were measured

Two machines, because the two halves of the question live in different places.

**Model side and official renderer** — `/Users/samib/se8-fetch-venv` (python 3.12, mujoco 3.12.0,
gymnasium 1.3.0, gymnasium-robotics 1.4.2, numpy 2.5.3, `stable_worldmodel` from the source checkout
176d545 installed editable, `MUJOCO_GL=glfw`), on an Apple M5. Every per-leaf pixel number on this
page is the **official** gymnasium-robotics renderer driven through the swm `FetchWrapper`, never a
stable-envs frame: which leaf reaches which array, and whether the reference itself moves.

**Our renderer** — Oscar, one GPU job, `se-s3b-fetch311` (mujoco / mujoco_warp 3.11.0,
gymnasium-robotics 1.4.2), which is the interpreter the frozen Fetch corpora were measured in. Fetch
renders only through MJWarp on cuda, so the registered-default reproduction and the isolation checks
that need pixels run there:
`examples/results/step8/variations-fetch/gpu_evidence.json` and `pixel_parity.json`.

Measured on an **NVIDIA GeForce RTX 3090** (`gpu2264.oscar.ccv.brown.edu`, driver 575.57.08,
CUDA 12.6, torch 2.14.0+cu126, mujoco 3.11.0, mujoco_warp 3.11.0, warp 1.17.0), SLURM job **6452376**,
tested tree SHA `4b3d09fae90270c59d1273cfb7d5fc37e041ed75`, 8 states per scene at 128 px,
**PASS on all four scenes**:

- the registered default is **byte-identical** through the mapping — style and frame, 0/255;
- `variation_from_style(default)` gives `light.intensity = 0.800000011920929`, the model's own
  diffuse rather than swm's declared 0.7, and re-applying it returns the default byte for byte;
- `light.intensity` set to 0.8 reproduces the default frame at **0/255**, which is what the exact
  1.25 factor buys;
- all seven declared-unmapped leaves (five on Reach) move **0/255**;
- every mapped leaf moves pixels and only its own style slot: `table.color` 9.8-23.4 % of the frame,
  `background.color -> floor.color` **9.30-9.64 %**, `object.color` 0.04-0.09 %, `light.intensity`
  14.9-20.9 %.

That `floor.color` figure is worth reading twice: the official renderer's floor half of
`background.color` was measured independently, on a different machine and a different renderer, at
**9.18-9.45 %**. The two agree, which is the corroboration that the rename is the right one and that
the residual really is the skybox.

`pixel_parity.json` renders this branch's default frames and the Step-8 baseline tree's on the same
card in the same job: **0/255 on all four scenes**, verdict `IDENTICAL`. Those are randomly sampled
states, so it is a smoke check; the merge-gate replay of the **Step-4 frozen Fetch corpora** through
both trees is `frozen_corpus_replay.json` in the same directory.

No performance work was done on this family, and no renderer file was touched.
