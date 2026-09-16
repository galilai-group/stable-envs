# Step 8 — VariationMaps for the Step-7 new cells and the OGB / Rocket / maze families

**Workstream C.** Baseline `43ac95238f7472137619f3e69e7636e11dd088b1`, branch `step8/variations-newcells`.

Five `stable_worldmodel` ids, 49 offered variation leaves, and one question per leaf:

> when a caller asks this env for this leaf, do they get what they asked for, a documented refusal, or silence?

Before Step 8 the answer for all five was silence. `stable_envs.compat.worldmodel.VARIATION_MAPS`
carried five maps, none of them for these families, and a cell with leaves and no map is a Gate-F
gap: the renderer is right and the caller cannot ask it for the appearance the wrapper offers.

| swm id | stable-envs | leaves | mapped | unsupported | dead upstream | n/a |
|---|---|---|---|---|---|---|
| `swm/OGBCube-v0` | `OGBCube` | 11 | 2 | 3 | 1 | 5 |
| `swm/OGBScene-v0` | `OGBScene` | 12 | 2 | 4 | 1 | 5 |
| `swm/PFRocketLanding-v0` | `PFRocketLanding` | 9 | 5 | 0 | 1 | 3 |
| `swm/OGBMaze-v0` | `PointmazeLarge{Navigate,Stitch}` | 5 | 2 | 2 | 1 | 0 |
| `swm/SimplePointMaze-v0` | `SimplePointMaze` | 12 | 7 | 0 | 0 | 5 |
| **total** | | **49** | **18** | **9** | **4** | **18** |

The four OGB `mapped` rows are the **authorised follow-up**: `cube.color` and `floor.color` on both
cells, implemented in `stable_envs/envs/ogb_manip.py` after the original delivery classified them
`unsupported`. That change moves a certified cell's declaration from `style_dim: 0` to
`style_dim: 9`, and the section below says what was measured before it was believed.

Plus two ids that offer nothing at all: `swm/CraftaxClassicPixels-v1` and `swm/CraftaxPixels-v1`
both set `variation_space = None` in `CraftaxWrapper.__init__`, confirmed live. Zero leaves is not a
gap, and recording it is what lets the Step-8 count say "every id is accounted for" without an
implicit exception.

The per-leaf tables are generated, not written:
[`examples/results/step8/variations-newcells/leaf_tables.md`](../../examples/results/step8/variations-newcells/leaf_tables.md)
and `leaf_classification.json`, both from
`scratch/step8/variations-newcells/generate_report.py --check`.

## The method, and the trap in it

Every verdict is a measurement. The rule that produced the numbers above is that a leaf is
`dead_upstream` only when the OFFICIAL renderer was asked to move it and did not — never when we
merely failed to find a reader in the source.

That rule has a failure mode which this workstream walked into once and which is worth writing down,
because it produces confident nonsense: **a probe that silently fails to apply the variation returns
all zeros, which looks exactly like a dead leaf.** The first Rocket sweep reported all nine leaves
dead. They are not; the probe was passing `options={'variation_values': ...}`, and
`PFRocketLanding` reads only `options={'variation': [names to resample]}`. Each of the five ids
turned out to accept a *different* mechanism:

| id | how a variation is actually applied | how the probe was validated |
|---|---|---|
| `OGBCube`, `OGBScene` | `options['variation_values']`, exact values | a known-live leaf (`cube.color`) moves `geom_rgba` |
| `PFRocketLanding` | `options['variation']` only; `variation_values` is ignored | `rocket.body_color` moves 35,551 px |
| `OGBMaze` | `options['variation_values']` | `wall.color` moves 26,310 px *and* `mat_rgba` reads back |
| `SimplePointMaze` | `options['variation']` only; `variation_values` is ignored | `background.color` moves 211,173 px |

Two further noise floors had to be pinned before any count meant anything. OGBench draws
`cur_task_id` from the **global** `np.random`, so two identical `reset(seed=0)` calls differ by
3,000–5,500 px — larger than several real leaf effects; the probe pins the global seed, the task id
and the qpos/qvel, after which base-vs-base is 0. And `SceneEnv`'s reset is not seed-reproducible at
all: `geom_rgba`, `mat_rgba`, `site_pos` and `dof_damping` move between two identical seed-0 resets,
while seed 0 and seed 7 give byte-identical models — the seed has no effect whatever. That is why
the Scene table attributes `lock_color` by mover-geom membership rather than by array diff.

## Rocket — five colours, one measured-dead leaf

The cell already had the style space; what it lacked was the map. The five RGB material leaves map
by identical path and identical units (0–255 on both sides), and `flame.ignited` is `ours_only`
because upstream PyFlyt recolours the flame from the thrust command at step time — it is a
consequence of the action there, and a style bit here only because our renderer is a pure function
of `(state, style)` and the flame colour is not derivable from the 13-dim state.

**`environment.sky_color` is dead upstream, measured.** Resampled alone over five seeds spanning the
RGB cube — drawn values as far apart as `(43,202,87)` and `(149,165,44)` — it moved **0 of 691,200
pixels**, max channel difference 0, with all three frame stacks hashing to the same digest. The
static picture agrees: the string occurs exactly once in `stable_worldmodel` (its own declaration at
`pyflyt_rocketlanding.py:127`), not at all in PyFlyt, and pybullet's TinyRenderer clear colour is
fixed. Offering it would advertise a knob that cannot work.

**`pad.color` is mapped on model evidence rather than pixel evidence,** and the distinction matters.
A pixel probe sees nothing — the landing pad is never in the reference render camera's view over 120
steps from a ~430 m drop — but `getVisualShapeData` shows the colour reaching the model with the
3-decimal URDF quantisation our `material_table` already reproduces: `(104,60,3)` becomes
`(0.408, 0.235, 0.012)`. A leaf that reaches the model and not this camera is mapped, not dead.

One default disagreement is recorded rather than smoothed over: swm declares `pad.color`'s
`init_value` as `(200,200,200)`, but a plain reset loads the **unmodified** pad URDF —
`_modify_pad_urdf` runs only when a variation is requested — so the registered default pad is white.
Our default of `(255,255,255)` is the one a plain reset produces, which is what "the registered
default" has to mean.

## OGB Cube / Scene — two colours honoured, four escalated, and why the split is where it is

Their frame is a MuJoCo-baked static plate composited with one mujoco_warp launch for the movers,
and **both halves are built from a single compiled `MjModel` whose appearance arrays
`_appearance_digest` asserts never change**. swm applies its appearance leaves by rewriting the MJCF
XML and recompiling, so honouring them *in general* means one compiled model, one plate rig and one
captured mujoco_warp graph per appearance vector — which is why these cells shipped with an empty
style space.

Measuring which model arrays each leaf actually writes is what turns that from a worry into a
decision. Five of the six split cleanly by cost:

| leaf | writes | geoms | what honouring it costs |
|---|---|---|---|
| `cube.color` | `geom_rgba`, no recompile | 2–3 **movers** | nothing new — `_upload_button_colours` already writes per-world `geom_rgba`/`mat_rgba` by exactly this route |
| `floor.color` | `tex_data` only | static floor | a plate cache key — the plate is the only half that sees it |
| `light.intensity` | `light_diffuse` row 0 | lights both halves | a **per-world model array** inside one mujoco_warp context |
| `camera.angle_delta` | `cam_mat0`, `cam_quat` | the camera both halves use | a per-world camera |
| `cube.size` | `geom_size` + 12 derived | 2 **movers** | a per-world `geom_size` |

The first two are cache-key work and are now **implemented and mapped**. The last three are a
renderer change, and per plan §4 they remain **escalation candidates** rather than being declared
unsupported on this workstream's authority: each looks feasible and is arguably semantically
required, and that combination is exactly the case the plan reserves for Sami. Scene's `lock_color`
is the same shape of question and is held with them.

### What the two implemented leaves actually do

`cube.color` writes `geom_rgba` on `object_0` and `target_object_0` — both **movers** — through the
per-world route `_upload_button_colours` already uses, so it never touches the baked plate. Each
geom keeps its own alpha, because swm writes `rgba[:3]` and leaves the alpha to the MJCF: the cube
is opaque and the goal marker is translucent, and flattening them to one alpha would be a visible
change dressed up as a colour.

`floor.color` writes the compiled checker texture and nothing else, and the floor is static, so it
is a property of the plate alone. swm reaches its picture by rewriting the MJCF `rgb1`/`rgb2` and
recompiling; we reach the same bytes by substituting the texels in the plate rig's **private** copy
and calling `mjr_uploadTexture`, which is the half that is easy to forget — `mujoco.Renderer`
uploads textures when it is built, so writing `tex_data` afterwards changes the model and not the
GL context, and the leaf would read as inert. The `markrgb` texels are deliberately left alone: swm
does not touch them either.

Two contracts are worth stating because they are the ones a later change could quietly break.

**Shape.** swm declares `cube.color` as `(num_cubes, 3)` and both registered cells are one-cube
configurations, so we carry the leaf **flat, as `(3,)`**, with the one-cube assumption asserted where
the default is read rather than assumed at the `Box`. `floor.color` stays `(2, 3)`, because the
checker's two colours are not interchangeable. `style_from_variation` flattens before slicing, so
swm's `(1, 3)` lands in our width-3 slot unchanged; `variation_from_style` emits a flat vector back,
which needs a reshape before it is fed to a live swm env. That asymmetry is a property of
`compat.worldmodel`'s flat layout, shared with every multi-row leaf in the repository, and is
recorded rather than papered over.

**Default restore.** The module caches are keyed by cell, not by renderer instance. The mujoco_warp
context's per-world `geom_rgba` is mutable and shared across renders, across chunks and across
renderer instances, and OGBCube has no button upload that would rewrite the block — so a render that
skipped the upload because "this is the default anyway" would draw whichever colour an earlier render
left in that world, with the result depending on batch size, chunk boundaries and process history.
The cube colour is therefore uploaded for every world of every chunk of every render, including the
default, and the floor texture is written on every plate **bake**. Neither is a tolerance question;
the first version of this patch used an `np.allclose` shortcut and the reviewer reproduced the leak
on CPU without a GPU. Six regression tests in
`tests/unit/test_variation_maps_newcells.py` now fail against that shortcut.

**`agent.color` is dead upstream for a reason worth naming.** `CubeEnv.modify_mjcf_model` computes

```python
agent_color_changed = np.allclose(current_material_rgba, desired)
```

which is `True` exactly when the colour did **not** change — the negation is missing. So changing
`agent.color` alone never sets the dirty flag, the model is never recompiled, and the staged MJCF
edit never reaches the compiled model. Measured: `agent.color=(1,0,0)` alone leaves `mat_rgba` at
the default `(0.61,0.28,0.82)`; the same request co-varied with `light.intensity` — which sets the
flag correctly — does turn the arm red. It is dead in isolation and live when co-varied, which is
why the reason names the mechanism instead of stopping at "dead".

## OGB maze — the configuration decides the answer

All five leaves are classified in `ob_type='pixels'`, because that is the configuration our renderer
reproduces (`reference_id` is `ogbench/pointmaze-large-v0`, not the swm id). Stating the
configuration is not pedantry here:

> **Upstream finding.** The registered id `swm/OGBMaze-v0` defaults to `ob_type='states'`, and in
> that configuration **all five leaves move zero pixels** — `compile_model()` rebuilds ogbench's
> `custom_renderer` only for `'pixels'`, so a states-mode render keeps drawing the pre-recompile
> model even though `mat_rgba` provably changed.

Two leaves map. `agent.color` is identical in path, units and default. `wall.color` is **renamed**,
not identical — swm spells it singular and we spell it `walls.color` — and mapping it by name would
have mapped nothing at all, silently, because neither side would find the other's path.

That leaf also resolves an apparent default conflict. swm's declared `init_value` reads back as
`[1,1,1]`, while our pixel-certified default is `[0.6,0.6,0.6]`. Both are "correct": swm reads the
value from the loaded XML, and ogbench rewrites the wall rgba to `.6 .6 .6` **only** for
`ob_type='pixels'`. The `1.0` is a states-mode artifact. Setting the leaf to our 0.6 reproduces the
default frame bit-exactly (0 of 50,176 px), which settles it by measurement.

**`floor.color` is dead at the appearance we render.** It has an effect only when
`floor.pixel_encoding == 0`; the official OGBench appearance is encoding 1 (the position-encoding
gradient), where the recompile overwrites `tex_data` and the leaf moves 0 of 50,176 px. At encoding
0 it moves 21,029 px. A second upstream oddity sits underneath: swm's declared default for
`pixel_encoding` is 0, but OGBench constructs the env with the gradient **on**, so a plain swm reset
silently turns the official floor encoding off.

**`floor.pixel_encoding` is the one leaf whose renderer support already exists and whose *map*
cannot express it.** Our `rendering.decolor_floor` is the same switch with the opposite sense (swm's
encoding 1 is our `decolor_floor=0`, and both defaults are the official gradient). `VariationMap`
can express `ours = theirs * factor` and nothing else, so an inverting relation has no
representation. That is a capability gap in the shared class, not a renderer gap — see the ownership
request below.

**`light.intensity` is an escalation candidate.** It writes `light_diffuse` row 0 and nothing else
(0 moves 43,931 px; 1.0, the default, moves 0). Our renderer has no light leaf because its lighting
is a *fitted* law with a fixed per-light split calibrated at the default intensity, so scaling the
diffuse term is not obviously the operation the official renderer performs. Offering it needs the
law re-derived as a function of intensity and re-certified.

## SimplePointMaze — seven mapped, five that describe something we do not have

Seven appearance leaves map by identical path, identical units and identical defaults — verified
leaf by leaf against the live space rather than assumed from the names, including
`walls.number` as `Discrete(3, start=4)` with init 5 and the radii in world units on the 5×5 domain.

The other five are `not_applicable`, and the distinction from `unsupported` is load-bearing.
`agent.position` is our **state**: swm's `render()` draws `self.state` and never the leaf, so setting
it after a reset moves 0 of 250,000 px — it is a reset distribution that seeds the state.
`agent.speed` is dynamics and reaches no draw call (0 px at both 0.05 and 2.0). And the three
`goal.*` leaves are not drawn because this class is the `show_goal=False` configuration — measured
0 px with the goal off and 1,264–6,812 px with it on. That divergence predates Step 8 and is
documented in `docs/research/swm_coverage.md`; `reference_env` passes `show_goal=False` too, so the
fidelity numbers compare like with like. The renderer *could* draw a goal; this registration does
not have one, which is a different statement.

## What this workstream did not do

No renderer was changed. No cell's `make_variation_space` was touched, so no `style_dim` or
`style_digest` declaration moved, and **the registered default of every one of the five cells is
byte-identical to the baseline by construction** rather than by re-measurement. The patch is the map
module, one import line in `compat/worldmodel.py`, the tests and these artifacts.

That statement no longer covers OGBCube and OGBScene. The authorised follow-up changes
`stable_envs/envs/ogb_manip.py`, so their declaration moved — `style_dim` 0 → 9 on both cells — and
"the default did not move" became something to measure rather than to argue. It was measured on the
frozen Step-7 corpus, on the card, against `$SCR/se8/baseline`; the numbers and the merge-gate table
are in
[`examples/results/step8/variations-newcells/findings.md`](../../examples/results/step8/variations-newcells/findings.md).

## Open items for the coordinator

1. **Escalation (plan §4)** — `light.intensity`, `camera.angle_delta` and `cube.size` on OGB
   Cube/Scene, and `light.intensity` on OGB maze. Each appears feasible and semantically required,
   and each needs a per-world model array or a re-derived lighting law, i.e. a renderer change. Not
   declared unsupported on this workstream's authority.
2. ~~**Cheap follow-ups, if authorised**~~ — **DONE.** `cube.color` and `floor.color` on OGB
   Cube/Scene were authorised and are implemented, with the declaration delta recorded and the
   default re-measured on the frozen corpus rather than assumed. Scene's `lock_color` was NOT
   included: it is the same mechanism as `cube.color`, but it is held with the light/camera/size
   group pending the shared-capability decision.
3. **Ownership request** — `VariationMap` cannot express `ours = 1 - theirs`. One `inverted` flag,
   or an affine `(factor, offset)`, would let `floor.pixel_encoding` become `mapped`. The class is
   owned by the DMC/classic-control workstream.
4. **Upstream findings bundle** — the `agent.color` inverted dirty flag; `OGBMaze` states-mode
   renderer staleness; `SceneEnv` reset ignoring its seed entirely; `pixel_encoding`'s declared
   default disagreeing with OGBench's constructed appearance; `pad.color`'s declared default not
   being what a plain reset produces; `PFRocketLanding` and `SimplePointMaze` ignoring
   `variation_values`; `sky_color` confirmed dead.
