# DM Control: a frame is `f(state)`, and the two renderers that reproduce it

**Verdict: sixteen cells, eleven at parity and five at the visual tier.** The five that landed first —
`cartpole-swingup`, `cartpole-balance`, `pendulum-swingup`, `acrobot-swingup`, `point_mass-easy` — are
joined by **six more at full parity**: `reacher-easy`, `reacher-hard`, `finger-spin`,
`finger-turn_easy`, `finger-turn_hard`, `ball_in_cup-catch`. `parity_interior_min` stays at **1.0 on
every one of the eleven**, measured over 64 states on three independent draws. Five body-tracking
domains — `cheetah-run`, `walker-walk`, `hopper-hop`, `humanoid-stand`, `swimmer-swimmer6` — ship in the
**visual tier** through MJWarp, at mean 4.4–8.1/255 with 70–91% of pixels within 8 and SSIM 0.66–0.96.

> **Update (2026-08-10).** The first version of this page recorded three blockers. Two are now closed
> and the third is closed here.
>
> * **"a target moves per EPISODE, so the frame is `f(qpos, target)`"** — closed by widening the state,
>   which is Meta-World's result ([metaworld.md](metaworld.md) §1) applied to a family that needs 2–4
>   floats of it rather than 14 arrays. The draw is *discovered* by diffing episodes rather than
>   transcribed, and that immediately corrected a fact this page had wrong: `finger-spin` and
>   `fish-upright` draw **nothing** per episode — their `initialize_episode` writes the same constants
>   every time — so `finger-spin` needed no widening at all, only the zero-alpha rule (§2.1, §9).
> * **"camera 0 is `trackcom`, so there is no plate"** — a limitation of the plate, not of the domain.
>   MJWarp implements `mjCAMLIGHT_TRACKCOM` and lands on MuJoCo's `cam_xpos` to **1e-8 m** once
>   `com_pos` runs between `kinematics` and `camlight`, exactly as
>   [mujoco_locomotion.md](mujoco_locomotion.md) §6.2 measured. Five of the six ship (§8).
> * **"`ball_in_cup` draws a spatial tendon, a mover kind the ray-caster lacks"** — closed. MuJoCo does
>   not draw a slack tendon as a connector: it **hangs**, as 28 capsules along a catenary, and a taut
>   one is a single capsule recoloured by its limit constraint's impedance. `mjv_catenary` is exported
>   from `libmujoco` and absent from the Python bindings; it is reproduced here in numpy to **0 ULP**
>   (§5.1).

`docs/research/classic_control.md` recorded this family as "needs a mesh/3-D path"; that was right, and
this is that path.

Everything below was measured on this cluster (H200, driver 590.48.01, `MUJOCO_GL=egl`) under SLURM.
Scripts in `scratch/dmcontrol/`, raw JSON in `/mnt/vast/home/randall.balestriero/logs/dmc/` and
`.../logs/dmc2/`.

---

## 1. The loader, and the pin that fixes it

`dm_control` could not be loaded at all in the shared environment:

```
AttributeError: 'MjModel' object has no attribute 'flex_bandwidth'
  dm_control/mujoco/index.py:635 struct_indexer -> getattr(struct, field_name)
```

Not a subtle skew. `dm_control` builds its named-index layer from a generated table of every `MjModel`
array field (`mjbindings/sizes.py`), `mujoco` **3.9.0 removed `flex_bandwidth`**, and `dm_control`
1.0.41's table still lists it. The failure happens inside `Physics.__init__`, so *every* domain in the
suite raises before returning.

The dependency metadata does not prevent it — 1.0.41 declares `mujoco>=3.8.1` while only 3.8.1 actually
works — so the pin has to come from the table, not from the resolver:

| dm_control | declares | `flex_bandwidth` in `sizes.py` | usable with |
|---|---|---|---|
| 1.0.41 | `mujoco>=3.8.1` | **yes** | mujoco **3.8.1 only** |
| 1.0.43 | `mujoco>=3.10.0` | no | mujoco ≥ 3.10 |
| 1.0.44 | `mujoco>=3.11.0` | no | mujoco ≥ 3.11 |

There is no `dm_control` that works with mujoco 3.9, which is what the shared conda env has and what
the 25 OGBench cells' parity numbers were calibrated against. So the fix is a venv, not an upgrade:

```
/mnt/vast/home/randall.balestriero/venvs/dmc
    python 3.11 · mujoco 3.11.0 · dm_control 1.0.44 · mujoco-warp 3.11.0
    warp-lang 1.16.0 · torch 2.11.0+cu130 · numpy 2.4.6 · scipy 1.17.1 · pytest 9.1.1
```

**Working pin: `mujoco==3.11.0` + `dm_control==1.0.44`.** All 17 domains probed load and render.
`mujoco==3.10.0` + `dm_control==1.0.43` is the other consistent pair; `mujoco==3.8.1` +
`dm_control==1.0.41` is the backward one. Do **not** upgrade the shared env: mujoco 3.11 would move
every OGBench parity number with it.

---

## 2. The state: a frame is exactly `f(qpos)`

Measured, not assumed, by the repo's standard method (`probe_load.py`, 84 px, camera 0, states from a
real rollout of the actual task):

| domain | nq | render twice | `qvel` × 100 | same qpos, different history | qpos entries that move pixels |
|---|---:|---:|---:|---:|---:|
| cartpole-swingup / balance | 2 | 0 px | 0 px | 0 px | 2 / 2 |
| pendulum-swingup | 1 | 0 | 0 | 0 | 1 / 1 |
| acrobot-swingup | 2 | 0 | 0 | 0 | 2 / 2 |
| point_mass-easy | 2 | 0 | 0 | 0 | 2 / 2 |
| ball_in_cup-catch | 4 | 0 | 0 | 0 | 4 / 4 |
| reacher-easy | 2 | 0 | 0 | 0 | 2 / 2 |
| finger-spin | 3 | 0 | 0 | 0 | 3 / 3 |
| cheetah-run | 9 | 0 | 0 | 0 | 9 / 9 |
| walker-walk | 9 | 0 | 0 | 0 | 9 / 9 |
| hopper-hop | 7 | 0 | 0 | 0 | 7 / 7 |
| swimmer-swimmer6 | 8 | 0 | 0 | 0 | 8 / 8 |
| fish-swim | 14 | 0 | 0 | 0 | 14 / 14 |
| humanoid-stand | 28 | 0 | 0 | 0 | 28 / 28 |
| manipulator-bring_ball | 11 | 0 | 0 | 0 | 11 / 11 |
| stacker-stack_2 | 14 | 0 | 0 | 0 | 14 / 14 |
| quadruped-walk | 23 | 0 | 0 | 0 | 23 / 23 |

**Zero pixels differ under every adversarial condition, on all 17 domains, and there is no dead
dimension anywhere.** So the compact state is 1–28 floats against 84·84·3 = 21,168 bytes of frame.

Two caveats that are properties of the *task*, not the renderer, and that decide what can ship:

* **Per-episode model edits.** Within an episode the frame is still `f(qpos)`; across episodes it is
  `f(qpos, draw)`. §2.1 measures the draw.
* **`visualize_reward` is off by default.** `dm_control.suite.base.Task._set_reward_colors` blends the
  `self`/`effector`/`target` materials toward their `_highlight` variants as a function of reward, so
  with it on the frame is `f(qpos, reward)` and the state is no longer minimal. Nothing here enables
  it, and the shipped envs keep the `_default` materials in step with the knob so it stays true.

## 2.1 The widened state, and the fact this page had wrong

The first version of this page recorded "`reacher` / `finger` / `fish` / `swimmer` move a target geom
or site per EPISODE" as one fact. It is four different facts, and one of them was not true. Measured by
running `N_DRAWS = 6` real episodes and diffing every model array a renderer reads
(`probe_widen.py`, and the same census the shipped `_episode_slots` performs at construction):

| domain-task | what `initialize_episode` writes | of which VARIES per episode | state |
|---|---|---|---:|
| `reacher-easy` / `-hard` | `geom_pos['target']` x,y (+ `geom_size` on hard) | `geom_pos['target']` x,y | nq + **2** |
| `finger-turn_easy` / `-hard` | `site_pos['target']` x,z, `site_size['target']` | `site_pos['target']` x,z | nq + **2** |
| `swimmer-swimmer6` | `geom_pos['target']` x,y, `light_pos['target_light']` x,y | all four | nq + **4** |
| `fish-swim` | `geom_pos['target']` x,y,z | all three | nq + **3** |
| **`finger-spin`** | `site_rgba['target'/'tip']` alpha → 0, `dof_damping` | **nothing** | nq |
| **`fish-upright`** | `geom_rgba['target']` alpha → 0 | **nothing** | nq |
| `ball_in_cup-catch`, `cheetah`, `walker`, `hopper`, `humanoid` | nothing | nothing | nq |

`finger-spin` and `fish-upright` write their model at *every* reset and write the same value every
time, so one reset cannot separate a task constant from an episode draw and six can. The consequence is
concrete: `finger-spin` needed no state widening, and what it *did* need was for the model to be put
through an episode before anything was read off it, so that its two zero-alpha sites are hidden (§9).

The ladder, restoring episode A's state into a *different* episode B and diffing the frames:

| rung | reacher-easy | reacher-hard | finger-turn_easy | swimmer-swimmer6 | fish-swim | ball_in_cup |
|---|---|---|---|---|---|---|
| R0 `qpos` only (control) | 176/255 over 3.7% | 171 over 0.6% | 219 over 3.8% | 167 over **90.2%** | 56 over 1.3% | **0** |
| **R1 `qpos` + the draw** | **0/255** | **0/255** | **0/255** | **0/255** | **0/255** | **0/255** |
| R2 `qpos` + every array | 0/255 | 0/255 | 0/255 | 0/255 | 0/255 | 0/255 |

R1 is exact on every domain, and R2 buys nothing over it — the 2–4 floats are the whole of it. Swimmer's
control row is the one to read: a target the camera rarely sees still moves **90% of the frame**,
because the episode draws a *light* as well as a geom.

**The `reset` staleness rule from Meta-World was checked and does not apply here.**
[metaworld.md](metaworld.md) §1.1 records 28 of 50 tasks writing `model.body_pos`/`site_pos` after
their last forward, leaving `data.site_xpos` stale straight out of `reset()`. On all thirteen DM
Control domains probed, the frame immediately after `reset()` is bit-identical to the frame one
`mj_forward` later — **0/255, four resets each** — because `dm_control`'s `Task.after_reset` forwards
after `initialize_episode` returns. The rule was worth checking rather than assuming; it is a property
of the write order and nothing about DM Control guarantees it a priori.

**Treated as movers, not as a re-bake.** The drawn elements sit on the *world* body, so the mover rule
"anything off body 0" calls them static and they go into the plate — which is exactly why the plate was
constant within an episode (0/255) and not across them. Putting their position in the state and moving
them into the mover set makes the plate 0/255 across episodes too, and costs no bake:

| domain | plate, movers hidden, WITHIN an episode | ACROSS episodes | + the drawn element hidden |
|---|---|---|---|
| `reacher-easy` | **0/255** | 215/255 over 3.8% | **0/255** |
| `reacher-hard` | **0/255** | 214/255 over 0.5% | **0/255** |
| `finger-turn_easy` | **0/255** | 224/255 over 4.2% | **0/255** |
| `finger-spin` | **0/255** | **0/255** | — |

The alternative — key the plate on the episode, as [metaworld.md](metaworld.md) §6 must — would cost
three MuJoCo renders per distinct draw, i.e. 3,072 renders for a batch of 1,024 states from 1,024
episodes. As movers they cost nothing.

---

## 3. What a DM Control frame is actually made of

Read off the compiled model and the live `mjvScene` (`probe_scene.py`, `probe_render.py`), because a
static reading of the XML misses everything MuJoCo adds.

**Geometry is primitive-only.** `nmesh = 0`, `nhfield = 0` on all 17 domains, including `quadruped`.
Every geom is a plane, sphere, capsule, cylinder or box, and everything is in group 0. `mujoco.Renderer`
at its default `MjvOption` shows geom groups **[1,1,1,0,0,0]** and site groups **[1,1,1,0,0,0]**, and it
matches `physics.render()` at **0/255 on every domain** — so the reference bridge is one line and the
group-3 sites (`acrobot`'s `target`, `finger`'s ellipsoids) are genuinely not drawn.

**Two lights, and they are nothing like OGBench's.**

| light | type | ambient | diffuse | specular | cutoff | exponent | castshadow |
|---|---|---|---|---|---|---|---|
| headlight (`mjvScene` only) | **directional**, along the camera axis | .4 | .8 | .1 | — | — | **0** |
| model `light` | **spot** at a fixed position | 0 | .7 | .3 | **45°** | **10** | **1** |

The model light is a *spotlight with exponent 10*, which is why `point_mass` looks like a glow rather
than a flat plate, and it is the only light that casts a shadow, under `<quality shadowsize="2048"/>`.
OGBench's two lights both point straight down and can be folded into one direction; these cannot.

**Almost the whole frame is textured.** From MuJoCo's own segmentation buffer at 84 px:

| domain | skybox | textured geoms (checker floor) | **textured total** |
|---|---:|---:|---:|
| cartpole-swingup | 62.0% | 34.1% | **96.1%** |
| pendulum-swingup | 16.7% | 82.2% | **98.9%** |
| acrobot-swingup | 77.1% | 22.0% | **99.1%** |
| point_mass-easy | 0.0% | 95.1% | **95.1%** |
| ball_in_cup-catch | 73.9% | 0.0% | **73.9%** |
| cheetah-run | 81.9% | 13.9% | **95.8%** |
| walker-walk | 58.1% | 34.6% | **92.7%** |
| reacher-easy | 0.0% | 91.8% | **91.8%** |
| finger-spin | 68.2% | 25.0% | **93.2%** |

**92–99% of a DM Control frame is a texture** — a 300×300 `builtin="checker"` ground with
`reflectance=.2`, and an 800×4800 `builtin="gradient"` skybox with `mark="random"` speckles. For
calibration, RoboCasa's textured share is 82%. Any backend that cannot sample those two textures is
not rendering this suite.

**What a static plate cannot absorb.** Rendering the reference twice per state, once with every moving
geom and site in a disabled group, splits the frame into "what a bake gives" and "what the mover pass
must produce" (`probe_plate.py`, 8 states, 84 px):

| domain | movers | plate constant? | pixels the mover pass owns | of which the movers' OWN pixels |
|---|---:|---|---:|---:|
| point_mass-easy | 1 | **0/255** | 0.15% | 0.14% |
| ball_in_cup-catch | 6 | 150/255 ✗ | 1.56% | 1.30% |
| acrobot-swingup | 3+1 site | **0/255** | 1.69% | 1.21% |
| reacher-easy | 3 | **0/255** | 1.70% | 1.61% |
| pendulum-swingup | 3 | **0/255** | 2.87% | 1.28% |
| cartpole-swingup | 2 | **0/255** | 3.02% | 1.96% |
| finger-spin | 7 | **0/255** | 9.49% | 7.77% |
| cheetah / walker / hopper / swimmer / fish / humanoid | 6–19 | 51–220/255 ✗ | — | — |

The gap between the last two columns is the part that has no analogue in OGBench: a moving geom is
**mirrored** in a `reflectance=.2` floor and **darkens** floor pixels it does not cover.

---

## 4. MJWarp: measured, and not used for the fixed-camera cells

`docs/research/backend3d.md` recommends MJWarp's batch raytracer. It was tried first, with all three
of that document's traps closed — BGRA channel swap applied, geom groups matched explicitly on both
sides, and `mjRND_SHADOW`/`REFLECTION`/`SKYBOX` set on `MjvScene.flags` after `update_scene`
(`probe_render.py`, 84 px, 4 states, camera 0, against `mujoco.Renderer` on the identical model):

| domain | exact px | MAE /255 | PSNR dB | exact px, channels **not** swapped | MAE, not swapped |
|---|---:|---:|---:|---:|---:|
| acrobot-swingup | 30.9% | 3.43 | 27.4 | 0.0% | 36.11 |
| cartpole-swingup | 29.5% | 3.90 | 27.4 | 0.0% | 37.56 |
| cheetah-run | 29.0% | 4.57 | 25.6 | 0.0% | 36.71 |
| hopper-hop | 26.0% | 8.17 | 21.7 | 0.0% | 37.95 |
| ball_in_cup-catch | 21.7% | 3.65 | 26.5 | 0.0% | 30.73 |
| finger-spin | 20.4% | 3.75 | 26.5 | 0.0% | 33.48 |
| walker-walk | 19.5% | 15.17 | 17.3 | 0.0% | 39.01 |
| pendulum-swingup | 18.7% | 3.08 | 31.3 | 0.0% | 40.37 |
| reacher-easy | 0.5% | 23.25 | 19.5 | 0.0% | 61.80 |
| point_mass-easy | 0.5% | 23.29 | 19.5 | 0.0% | 61.95 |

The BGRA trap is real and enormous here: **without the swap, zero pixels are exact on every domain**,
and the mean error is 8–14× worse. Two further findings:

* **`use_shadows=True` changed the score by less than 1e-4 on every domain** — identical to four
  decimal places. MJWarp is not casting the spotlight's shadow, which on this suite is not a small
  omission.
* The two top-down domains (`point_mass`, `reacher`) are the worst by 5× in MAE, and the difference is
  visible in `logs/dmc/png/point_mass-easy_strip.png`: MuJoCo's spotlight paints a strong radial
  falloff on the floor that MJWarp renders nearly flat.

**This is the lightly-textured band, not a texture-path failure.** The corrected reading of the
RoboCasa result — MJWarp's shader never reads alpha, so 154 invisible marker geoms were painted opaque
over ~40% of that frame — does not apply here, and that was checked rather than assumed: across all 17
domains, **`cartpole`, `pendulum`, `acrobot`, `point_mass`, `ball_in_cup`, `reacher`, `finger`,
`cheetah`, `walker`, `hopper`, `fish` and `humanoid` have zero invisible geoms**, and **no domain in the
suite has a textured non-plane primitive**, so MJWarp's other known gap (`sample_texture` derives UVs
for `PLANE` and `MESH` only) does not bite either. The numbers above stand as measured.

Two domains *would* need the alpha pre-pass, and they are recorded because the failure is silent:
`swimmer` has **6 invisible `inertial` geoms and 6 invisible sites**, and `fish` and `finger` each have
invisible sites. The pre-pass now lives in `stable_envs/render/mjwarp.py` (`prepare_model`), covers
**sites as well as geoms**, and resolves the effective alpha through the material the way MuJoCo does;
`texture_gaps` reports the textured non-plane primitives in the same call.

So MJWarp is a fine backend for this suite *if the target is a perceptual match*. It is not one if the
target is the repo's bar. At MAE 3.1–4.6/255 and 19–31% exact it is 10–20× further from the reference
than the renderer below, and interior-pixel exactness is not reachable from there at all.

That is a statement about the fixed-camera cells, and it does not carry to the six whose camera rides
the body: there is no plate to be 10–20× better than, so the choice is MJWarp or nothing. Five of them
ship on it in the **visual tier**, at mean 4.35–8.07/255 with the same corrections plus `gl_rounding`
and the zero-alpha pre-pass — see §8, where the numbers are re-measured through the shipped env rather
than through this probe.

The two top-down domains are the exception in both directions and §8.1 is where that is followed up:
`point_mass` and `reacher` are the worst here by 5×, and `fish`, which has the same camera geometry and
no plate available, is the one `trackcom` domain that does not ship at all.

---

## 5. The parity renderer: three plates, a ray-cast, and a catenary

`stable_envs/envs/dmcontrol.py`. The camera of these domains is a `fixed` model camera and the static
half of the scene never moves, so floor, rails, walls, decorations and skybox are one constant image —
the OGBench design. What is new is how the two mover-dependent effects are handled, and in both cases
the trick is to bake the answer rather than to model it.

**Shadows: a second plate, not a shadow model.** Bake the static scene again with the shadow-casting
light's `diffuse` and `specular` set to zero. That is *exactly* the colour a static surface has when
that light does not reach it — ambient survives, which is what a shadow does — so a shadowed pixel
selects the second plate and stays as exact as the plate itself. Nothing about MuJoCo's 2048-tap
shadow map is reproduced; only the *shape* of the shadow is computed, by one occlusion ray per sample.

**Reflections: a third plate, and MuJoCo's own compositing rule.** MuJoCo draws a reflective plane as
`reflectance · mirror + (1 − reflectance) · surface`. Baking with `mat_reflectance = 0` gives `surface`
directly, so a mover appearing in the mirror is `r · mover + (1 − r) · plate_norefl` — an inversion of
the reference's own rule rather than a blend factor chosen to look right. The alternative, scaling the
mover's colour into the existing plate, cannot be correct: the plate already holds whatever the mirror
showed at that pixel and there is no way to subtract it back out.

**Three bakes per (domain, camera, size, appearance)**, one `mujoco.Renderer` per (domain, size), never
closed. Per frame: forward kinematics through `mj_kinematics`, one analytic intersection per mover
primitive, an occlusion ray to the spotlight for the static hit points and for the movers, and a
reflected ray for the pixels that are the mirror.

### 5.1 The tendon: a mover with no geom behind it

`ball_in_cup`'s blocker was recorded as "a spatial tendon, a mover kind the ray-caster does not have".
That was right and it understated the problem, because a spatial tendon is not one capsule. Read off
the live `mjvScene` rather than from the XML, `ball_in_cup`'s string is **28 `mjGEOM_CAPSULE` geoms
tracing a curve** — since MuJoCo 3.2 a slack tendon *hangs*, and `addSpatialTendonGeoms` draws
`min(vis.quality.numslices + 1, 100) - 1 = 28` segments along a catenary.

Three branches, and a random policy reaches all three:

| branch | condition | drawn | colour |
|---|---|---|---|
| catenary | `ten_length < tendon_range[1]` | 29 points → **28 capsules** | `tendon_rgba` = (0.5, 0.5, 0.5) |
| near-vertical bead | `length > 100 · h`, inside `mjv_catenary` | 3 points → **2 capsules** | same |
| **taut** | `mjv_isCatenary` returns 0 | wrap points → **1 capsule** | `(1-imp)·rgba + imp·vis.rgba.constraint` |

Over 256 states from a random policy: **168 slack, 88 taut (34%)**, and the bead branch fires in 3 of
64 sampled states. So the taut branch is not an edge case a renderer can decline — skipping it would
paint a third of a dataset's strings grey instead of red.

**`mjv_catenary` is exported from `libmujoco.so` and is not in the Python bindings.** It is reproduced
in numpy (`catenary` in `stable_envs/envs/dmcontrol.py`) and validated against the C symbol through
`ctypes` and, independently, against the capsule endpoints MuJoCo puts in the `mjvScene`:

| test | max abs error | max ULP |
|---|---:|---:|
| 200 random `(x0, x1, gravity, length)`, all three branches (37 taut / 16 bead / 147 full) | **0.0** | **0** |
| 240 more at `ncatenary` ∈ {3, 5, 29, 100} | **0.0** | **0** |
| 64 real `ball_in_cup-catch` rollout states, against the C | **0.0** | **0** |
| the same 64, against the live `mjvScene` capsules | 2.4e-08 | — |

Bit-for-bit. The 2.4e-08 residual against the scene is the float32 round trip through `mjvGeom`, and it
is the check that says the ctypes path is not self-consistent-but-wrong. Two things buy the exactness
and neither is stylistic: `math.*` is a direct call into the same libm MuJoCo's `mju_*` macros expand
to where a numpy ufunc is not, and the solve is MuJoCo's own — Newton on a change-of-variables form,
50 iterations, tolerance 1e-9, seeded at `intercept/√24`, with a backtracking halving loop of at most
10 that accepts the first step reducing `|res|`. Two quirks are reproduced because they are what MuJoCo
*draws*: points are spaced linearly in the **horizontal** coordinate, and divided by `ncatenary` rather
than `ncatenary − 1`, so the final capsule comes out about twice as long as the others.

The taut colour is `efc_KBIP` **column 2**, the limit constraint's impedance, and that is measured
rather than guessed. Over 88 taut states the four columns reproduce MuJoCo's own scene colour to
**1384, 52, 2.9e-08 and 23** respectively; the 2.9e-08 is the float32 round trip again. Computing it
needs the constraint set, so a scene with a tendon runs `mj_forward` per state rather than
`mj_kinematics` — affordable on a 4-dof model and the reason the mover is priced in §7.

Two mechanical details make it fit a fixed mover set. The chain has to be a constant number of
primitives because the batch holds all three branches at once, so each real segment is split into
collinear pieces (28 → 1 each, 2 → 14 each, 1 → 28). That is exact rather than an approximation: the
union of collinear capsules covering a segment *is* that segment's capsule, and a point on an interior
piece's cap sits exactly on the shared axis's perpendicular, so its normal is the radial vector the full
capsule gives there. And the plate has to lose the tendon with the rest of the movers, which no geom
group can do — `mjVIS_TENDON` off during the bake is what takes `ball_in_cup`'s plate from **97/255 over
0.78% of the frame to 0/255**.

**One measured constant, found the way the OGBench study finds them.** Cartpole's cart is a BOX, so its
front face has a single normal and per-vertex and per-pixel shading coincide on it by construction —
the one surface in this family where a shading constant can be read off directly. MuJoCo paints that
face `(227, 166, 104)` in every one of 64 states. With material `(0.7, 0.5, 0.3)` and ambient+diffuse
1.2, each channel constrains the additive white specular term `s` to the interval that rounds to the
observed level:

```
R  214.2 + 255s -> 227   =>  s in [0.04824, 0.05216)
G  153.0 + 255s -> 166   =>  s in [0.04902, 0.05294)
B   91.8 + 255s -> 104   =>  s in [0.04588, 0.04980)
                              intersection: s in [0.04902, 0.04980)
```

and the nominal value — material specular 0.5 × the headlight's 0.1 at `n·h = 1` — is **0.05000, just
outside it**. `SPECULAR_SCALE = 0.988` puts it at the midpoint. This is the same shape of finding the
OGBench study reports (specular weight measuring 0.25–0.26 against a nominal 0.30), and it is worth
0.42–0.60 → **1.00** of the interior-pixel metric on the two cartpole cells and nothing anywhere else.

### Mechanisms tried and refuted, so they are not retried

* **Per-vertex (Gouraud) shading of the box face.** The reference's face is one flat colour in every
  state while a per-pixel Blinn-Phong evaluation varies across it, which reads exactly like Gouraud.
  Implementing it — shade the four face corners with the local-viewer half vector, interpolate
  bilinearly — made cartpole's interior agreement **0.60 → 0.00**: the corner half-vectors swing with
  the cart's screen position, so the interpolated colour varies with `x` while the reference does not.
  The reference being *constant* is the evidence against a local-viewer per-vertex model, and for a
  specular weight slightly under nominal.
* **MJWarp with shadows on** (§4): no measurable change, because it is not casting them.

---

## 6. Parity

`scratch/dmcontrol/parity.py`, 84 px, supersample 2, **64 states per env** drawn from the domain's own
`initialize_episode` plus random-action rollouts, default appearance, against
`physics.render(84, 84, camera_id=0)`:

| env | **interior exact** | all-pixel exact | mean /255 | max /255 | mover pass | disagreement ON the mover pass | plate pixels exact |
|---|---|---:|---:|---:|---:|---:|---:|
| `DMCPointMassEasy` | *undefined* | **99.824%** | 0.023 | 52 | 0.17% | 100.0% | **1.000000** |
| `DMCAcrobotSwingup` | *undefined* | 98.325% | 0.255 | 51 | 1.54% | 99.0% | 0.999840 |
| `DMCCartpoleBalance` | **1.000000** | 98.273% | 0.288 | 52 | 2.50% | 100.0% | **1.000000** |
| `DMCReacherHard` | *undefined* | 98.083% | 0.227 | 74 | 1.93% | 100.0% | **1.000000** |
| `DMCBallInCupCatch` | **1.000000** | 98.058% | 0.267 | 87 | 1.94% | 98.4% | 0.999670 |
| `DMCPendulumSwingup` | **1.000000** | 97.864% | 0.332 | 70 | 2.39% | 100.0% | **1.000000** |
| `DMCCartpoleSwingup` | **1.000000** | 97.656% | 0.369 | 52 | 3.04% | 100.0% | **1.000000** |
| `DMCReacherEasy` | *undefined* | 96.222% | 0.325 | 74 | 3.94% | 100.0% | **1.000000** |
| `DMCFingerSpin` | **1.000000** | 92.571% | 0.541 | 66 | 9.35% | 99.9% | 0.999870 |
| `DMCFingerTurnHard` | **1.000000** | 92.099% | 0.587 | 95 | 9.90% | 99.8% | 0.999860 |
| `DMCFingerTurnEasy` | **1.000000** | 90.684% | 0.673 | 95 | 11.69% | 99.9% | 0.999860 |

**`parity_interior_min` is not relaxed on any of the eleven.** The measured interior agreement is
exactly 1.0 wherever the metric is defined, on three independent draws of 64 states each (spread:
all-pixel 0.90649–0.90734 on the widest, `finger-turn_easy`).

Three things that number needs said about it, and they are why the last three columns are here:

* **On four of the eleven the interior metric is undefined, and reporting 0.0 would be wrong.** The mask
  keeps only pixels whose 3×3 neighbourhood in the reference is one flat colour, and a DM Control frame
  has almost none: the checker floor sits under a spotlight gradient and the skybox is a speckled
  gradient. `interior_frac` is 0.000000 on `acrobot`, `point_mass` and both reachers, 0.0001–0.0052 on
  the rest. `conformance.pixel_agreement` returns `None` in that case and the test falls back to
  all-pixel agreement, which is the honest behaviour and the reason it exists.
* **Every disagreeing pixel is in the mover pass.** 98.4–100.0% of them, with the baked plate exact on
  0.999670–1.000000 of the pixels it owns. The number to defend is therefore not "90.7% of the frame"
  — the plate is most of the frame and it is exact by construction — but "the 0.2–11.7% the ray-caster
  is responsible for", and there the residual is a ±1-level rim on curved silhouettes and a one-pixel
  band at the shadow edge, mean 0.02–0.67/255 over the whole frame and max 51–95 on a handful of
  edge pixels.
* **One cell declares `parity_all_exact_min`, and nothing else.** All-pixel exactness is interior
  exactness *weighted by how much of the frame moves*, and finger's seven movers cover 9.4–11.7% of an
  84 px frame against cartpole's 3.0% — on the identical backend with the identical per-pixel residual.
  The interior floor, which is the number that describes the renderer, is untouched at 1.0.
  `finger-turn_easy` measures 0.9065–0.9073 over three draws, i.e. it *clears* the shared 0.90 bar by
  7.6× that spread; the floor is declared at 0.89 so the margin is the scene's rather than luck's.
  `finger-spin` and `finger-turn_hard` clear the same bar by 0.021–0.026 and declare nothing.

Supersampling 4 instead of 2 lowers the mean (0.288 → 0.105 on `DMCCartpoleBalance`, 0.370 → 0.184 on
`DMCCartpoleSwingup`) without moving the interior metric at all — confirming that what is left at the
interior is not coverage. 2 is the shipped default.

---

## 7. Throughput

`bench.py`, H200, 84 px, state in and frames out including forward kinematics. Reported as a
**(ratio, batch) pair with its crossover** — the batch at which this renderer overtakes `dm_control` —
because a bare peak hides that both families are *slower* than the reference at B=1.

**Parity tier**, supersample 2:

| env | movers | B=1 | B=64 | B=256 | **B=1024** | dm_control | **(ratio, batch)** | crossover |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| `DMCPointMassEasy` | 1 | 121 | 7,079 | — | **13,826** f/s | 386 f/s | **(36×, 1024)** | B≈4 |
| `DMCCartpoleSwingup` | 2 | 51 | 3,163 | 4,051 | **4,381** | 361 | **(12×, 1024)** | B≈8 |
| `DMCReacherEasy` | 3 | 38 | 2,312 | 3,236 | **3,324** | 377 | **(8.8×, 1024)** | B≈11 |
| `DMCAcrobotSwingup` | 3+1 site | 33 | 1,882 | — | **2,685** | 363 | **(7.4×, 1024)** | B≈13 |
| `DMCFingerSpin` | 7 | 17 | 1,024 | 1,351 | **1,372** | 372 | **(3.7×, 1024)** | B≈23 |
| `DMCFingerTurnEasy` | 7 | 16 | 963 | 1,288 | **1,306** | 370 | **(3.5×, 1024)** | B≈25 |
| `DMCBallInCupCatch` | 6 + **28 tendon** | 3.7 | 217 | 272 | **274** | 351 | **(0.78×, 1024)** | **never** |

**Visual tier**, one ray per pixel, CUDA-graph-captured, 64 worlds per context:

| env | B=1 | B=16 | B=64 | B=256 | **B=1024** | dm_control | **(ratio, batch)** | crossover |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| `DMCHopperHop` | 1,613 | 25,330 | 102,053 | 116,151 | **122,367** f/s | 337 f/s | **(364×, 1024)** | **B=1** |
| `DMCCheetahRun` | 1,448 | 23,318 | 99,540 | 112,852 | **117,931** | 355 | **(332×, 1024)** | **B=1** |
| `DMCWalkerWalk` | 1,508 | 24,018 | 95,823 | 109,188 | **113,474** | 363 | **(313×, 1024)** | **B=1** |
| `DMCHumanoidStand` | 1,375 | 21,874 | 85,743 | 94,942 | **99,244** | 302 | **(329×, 1024)** | **B=1** |
| `DMCSwimmerSwimmer6` | 977 | 15,121 | 58,470 | 62,635 | **64,110** | 284 | **(225×, 1024)** | **B=1** |

Three readings the tables force, and each of them is a property of a design rather than of the GPU.

* **The two tiers are 30–90× apart, and the plate is what costs it.** The plate makes a pixel exact by
  construction; it also makes three of the passes full-frame. The mover cast can be tiled — a DM
  Control mover is 0.2–11.7% of the frame — but the shadow test starts from *every static pixel* and
  the reflection test from every floor pixel, so both run over the whole image no matter how small the
  movers are. MJWarp has neither a plate nor a reflection pass and pays neither. Restricting the shadow
  pass to the movers' light-space footprint and the reflection pass to the plane's screen extent is
  still the obvious next step, and is worth roughly the ratio between the mover share and 1.
* **The parity tier scales with the mover count and nothing else.** 1 mover 36×, 7 movers 3.7×, 34
  movers 0.78×. It is **not** forward kinematics: `mj_kinematics` is a Python loop but costs ~17 ms of
  cartpole's 234 ms at B=1024.
* **`ball_in_cup` is the one cell this renderer is slower on than `dm_control` at every batch**, and
  the tendon is the whole of it: 28 extra capsules through three full-frame passes, i.e. 5.7× the
  primitive count of the next-heaviest cell. What it buys is the thing throughput cannot — the frame is
  the official frame, so a stored dataset and a regenerated one are the same bytes. A caller who wants
  `ball_in_cup` frames fast and does not need that should render them through `dm_control`.

The comparison is like-for-like in what it computes and not in what it *is*: dm_control renders one
frame at a time through EGL, so its 284–386 f/s does not improve with batch, while every column above
does until the GPU saturates.

---

## 8. The visual tier: the body-tracking domains

Camera 0 on these six is `mode="trackcom"`, so the "static" scene moves with the state and there is no
plate to bake. A backend that draws the whole scene every frame does not have that problem, and the
camera turns out not to be the obstacle at all: MJWarp's `mjCAMLIGHT_TRACKCOM` lands on MuJoCo's
`cam_xpos` to **1e-8–2e-7 m** on every domain, provided `com_pos` runs between `kinematics` and
`camlight`. `mujoco_warp.kinematics` writes `xpos`/`xquat`/`xmat`/`xipos` and **not** `subtree_com`,
which is what a `trackcom` camera reads; omitted, every frame is drawn from the pose `put_data`
uploaded, with no error and a plausible-looking picture. `mjwarp.needs_com_pos` exists for exactly this
and is asserted at context construction here rather than assumed.

Measured through the shipped env, 84 px, **three independent draws of 64 states each**
(`visual_calib.py`), against `physics.render` at camera 0:

| env | mean /255 | within 8 | **SSIM** | all-pixel exact | interior exact | skybox share | floor share |
|---|---:|---:|---:|---:|---:|---:|---:|
| `DMCHopperHop` | **4.35** | 90.5% | 0.684 | 30.7% | *undefined* | 89.3% | 8.2% |
| `DMCCheetahRun` | **5.01** | 88.2% | 0.661 | 29.0% | *undefined* | 83.1% | 12.7% |
| `DMCHumanoidStand` | **6.26** | 73.5% | 0.868 | 2.3% | 0.0002 | 0.0% | 95.9% |
| `DMCSwimmerSwimmer6` | **6.58** | 81.6% | 0.964 | 13.3% | 0.649 | 0.0% | 98.1% |
| `DMCWalkerWalk` | **8.07** | 70.0% | 0.826 | 20.5% | 0.005 | 47.8% | 44.8% |

For calibration, the repo's other visual-tier family (RoboCasa) declares mean 5.79–32.10 with 36.7–84.3%
within 8 and SSIM 0.821–0.921.

**Two named mechanisms carry all of it**, and they are separated by MuJoCo's own segmentation buffer:

* **the speckled gradient skybox** — 48–89% of the frame on cheetah, hopper and walker. DM Control's
  skybox is an 800×4800 `builtin="gradient"` texture with `mark="random"`, i.e. white specks scattered
  by MuJoCo's RNG, and MJWarp samples it with one unfiltered tap: mean **4.13–4.68/255** with
  89–92% of those pixels within 8 and 29–38% bit-exact. It is what holds SSIM down — a speck one pixel
  out of place costs structure twice — which is why the two skybox-dominated cells have the *best*
  mean and the *worst* SSIM and the two floor-dominated ones have the reverse.
* **the minified checker floor** — the same mechanism that decides
  [mujoco_locomotion.md](mujoco_locomotion.md). In the far tenth of the floor MuJoCo's mipmap has
  averaged the checker into one grey (luminance spread **0.11** on humanoid) and MJWarp point-samples
  it at full contrast (**9.70**). Floor mean 4.32–8.18/255.

Neither is closable here. The floor needs plane-texture level-of-detail selection in `mujoco_warp`'s
render kernel, which needs ray differentials the kernel does not compute.

**nvdiffrast was the obvious alternative and it is not a drop-in.** It is what
[mujoco_locomotion.md](mujoco_locomotion.md) §7 lands Gymnasium's six locomotion domains on, at mean
1.97–4.39/255 and SSIM 0.947–0.981 — better than the numbers above — and its mipmapped sampler is
exactly what closes the minified floor (0.0001 → 0.66–0.81 of far-band pixels bit-exact,
[nvdiffrast.md](nvdiffrast.md) §4.1). Three things stop the same substitution here, and only the first
is about fidelity:

* **the residual is a different one.** Gymnasium's skybox is a smooth gradient and its floor is 37–95%
  of the frame; DM Control's skybox is *speckled* and is 48–89% of the frame on three of the five, at
  mean 4.13–4.68/255. Closing the floor would leave the larger term untouched.
* **these scenes' model lights are `trackcom` too** — they ride the body, so their position is a
  function of the state. `nvdiff.Lights` carries one `pos` per light for the whole batch, and MJWarp's
  `camlight` computes them per world. A per-world light is a change to `stable_envs/render/nvdiff.py`,
  not a flag.
* **throughput**: 225–364× here against 23–27× there, on scenes an order of magnitude smaller.

Supersampling was measured on the identical mechanism and refuted: 16 samples per pixel take the
far field's mean error from 100 to 8–14/255 and *lower* whole-frame exactness on every domain, because
a box average over 16 samples is not GL's mipmap pyramid.

### 8.1 `fish` is measured and does not ship

`fish` is the sixth `trackcom` domain and it is the one this backend cannot draw. Camera 0
(`tracking_top`) looks straight down and the floor is **97–98% of the frame**, so the domain is
entirely the surface MuJoCo lights with a `cutoff=45`, `exponent=10` spotlight — and MJWarp renders
that falloff nearly flat:

| env | mean /255 | within 8 | floor mean | floor within 8 | far-band luminance spread (ref / MJWarp) |
|---|---:|---:|---:|---:|---|
| `fish-upright` | **24.92** | **1.2%** | 25.08 | **0.0%** | 14.0 / 16.3 |
| `fish-swim` | **24.46** | **2.1%** | 24.75 | 0.04% | 13.6 / 16.0 |
| `reacher-easy` (same geometry, ships at PARITY) | 28.43 | 11.7% | 29.06 | 9.6% | 19.4 / 23.5 |

It is **not** minification — the far-band luminance spreads agree to 15% where humanoid's disagree by
90× — it is the radial falloff, the same finding §4 records for `point_mass` and `reacher`. Mean 25/255
with 1% of pixels within 8 is not a picture worth declaring, so `fish` is recorded rather than shipped.
`reacher` is the control that makes the point: identical camera geometry, and it ships at **parity**,
because the plate holds that falloff exactly — MuJoCo painted it.

## 8.2 What is still out

| family | blocker | is it structural? |
|---|---|---|
| `fish-swim`, `fish-upright` | top-down camera over a spotlit floor that fills 97% of the frame; MJWarp renders the radial falloff flat (mean 24.9/255, 1.2% within 8) | **no.** A spotlight-aware shader, or a backend with per-world moving lights and a mipmapped floor. The state is measured and would be `qpos + 3`. |
| `dog`, `quadruped` | the only two suite XMLs with `<mesh>`; `quadruped` compiles to `nmesh = 0` but `dog` does not | mesh assets are out of scope for an analytic caster, as in RoboCasa. MJWarp ships a BVH mesh intersector, so a visual-tier port is not blocked on geometry. |
| `manipulator`, `stacker` | not re-probed since the first pass; both have fixed cameras and `nmesh = 0`, so the plate design should apply | untested, not refuted. |

None of these is a NO-GO on the thesis. Every one of the 17 domains passes state sufficiency with zero
dead dimensions; what varies is how much renderer each needs.

---

## 9. Bugs found, and one worth carrying to other envs

**Appearance defaults must not be read off a mutable shared model.** `make_variation_space` read each
material's `init_value` from the live `MjModel` in the env cache — the same model every bake mutates
and restores. Between bakes it carries the *last appearance rendered*, so a new env constructed after
another env had drawn an extreme variation inherited that colour as its "official" default. Every
same-process check passed: the default plate and the default style agreed with each other, because both
were wrong in the same way. The only test that could see it was
`test_frames_are_bit_identical_in_a_fresh_process`, where a subprocess with no previous bake disagreed
with the parent — five envs, all of them, and the parent's default plate's sha256 changed the moment
`check_variations_do_something` ran. Defaults now come from the pristine snapshot taken at compile
time, and `_apply_variation` restores the geom and site groups too, so a bake that raises part-way
cannot leave the movers hidden for the next one.

**A probe that releases an EGL context corrupts the probe.** The first parity script built a fresh
`mujoco.Renderer` per env for the segmentation buffer and dropped it at the end of each loop
iteration. The first env's numbers were right and every later one was nonsense (a "mover mask"
covering 87% of the frame). This is the fourth recorded instance of the same rule and the first in a
scratch script rather than in shipped code: **in this process an EGL context is created and never
released.**

**`mujoco_warp.kinematics` never recomputes a world-body geom's transform, so a per-world model edit
reaches nothing.** `_geom_local_to_global` early-outs on geometry whose world pose cannot change under
kinematics — geoms and sites hanging off the world body — which is exactly where `swimmer`'s
per-episode `target` sits. Writing the drawn `geom_pos` into the per-world model array was therefore
correct and inert: measured, moving `target_pos_x` by 1.5 m changed the official frame by **216/255**
and ours by **0**, while the per-episode *light* in the same state vector moved ours by 102. The symptom
was a state dimension the conformance suite called dead, which reads like a state-minimality problem
and is a backend one. The fix is to write the derived `data.geom_xpos`/`site_xpos` row as well, which
is free because the world body's frame is the identity; [metaworld.md](metaworld.md) §7 records the
same early-out from the other direction.

**`efc_KBIP`'s impedance is column 2, and three of the four columns produce a plausible colour.** The
taut tendon's colour is `(1 − imp) · tendon_rgba + imp · vis.rgba.constraint`, and the bindings do not
document the column order. Over 88 taut states the four columns reproduce MuJoCo's own scene colour to
1384, 52, **2.9e-08** and 23 — so the wrong choice is not subtly wrong, but two of the three wrong ones
still give a *colour*, and only comparing against the reference tells them apart.

**`dm_control` wraps `MjData`, and the pybind signatures reject the wrapper.** `physics.data` is a
`dm_control.mujoco.wrapper.core.MjData`, and `mjv_updateScene` raises `TypeError` on it rather than
duck-typing; `physics.model` has the same wrapper and the same `.ptr`. Every probe here takes `.ptr`.
Recorded because the error names the function rather than the argument, and the same call works from a
`mujoco.MjData` the caller built themselves.

**`import dm_control` succeeding is not the same as dm_control working.** Under the skew of §1 the
package imports fine and every *model* raises, so the five classes registered cleanly in the shared
conda env and then failed in the test fixture — 5 envs × 30 tests of errors, in a run that has nothing
to do with DM Control. The module now probes the failing path at import time
(`Physics.from_xml_string("<mujoco/>")`, milliseconds, no GL context) and raises `ImportError` naming
the three consistent pins. Verified both ways: the shared env registers **49** envs and reports
`dmcontrol` in `import_errors()`; the `dmc` venv registers **54**. That is the behaviour
`stable_envs/envs/__init__.py` is built for, and a machine-level version skew is exactly a
missing-optional-dependency, not a runtime error.

---

## 10. Reproducing

```
venv:   /mnt/vast/home/randall.balestriero/venvs/dmc     (mujoco 3.11.0, dm_control 1.0.44,
                                                          mujoco-warp 3.11.0, torch 2.11+cu130)
logs:   /mnt/vast/home/randall.balestriero/logs/dmc/
run:    srun --partition=open_research --account=open_research --qos=normal --gres=gpu:1 \
             --cpus-per-task=8 --mem=48G --time=01:00:00 ... MUJOCO_GL=egl PYTHONPATH=. <script>
```

| script | what it establishes |
|---|---|
| `probe_load.py` | dm_control loads; per-domain inventory; frame = `f(qpos)` under three adversarial conditions |
| `probe_scene.py` | `mujoco.Renderer` == `physics.render()`; camera modes; default geom/site groups; the `mjvScene` lights and both eyes; plate constancy with moving geoms AND sites hidden |
| `probe_plate.py` | the plate/mover split, and what the reflection and the shadow are worth |
| `probe_render.py` | frame composition from the segmentation buffer; per-flag ablation; MJWarp agreement under both channel readings and three configs |
| `probe_shade.py` | the shading residual, per primitive, on the pixels the interior mask actually keeps |
| `parity.py` | the headline table in §6, plus where the disagreement is |
| `bench.py` | end-to-end state → frames, against dm_control's own renderer |
| `probe_widen.py` | §2.1: the per-episode census over six real episodes, the `reset` staleness check, the zero-alpha census, the plate ladder, and the state ladder — one domain per process |
| `probe_catenary.py` | §5.1: `mjv_catenary` through `ctypes`, the numpy port, and 0-ULP agreement on 440 synthetic inputs and 64 real states |
| `probe_tendon_colour.py` | §5.1: which of the three tendon branches fires how often, and which `efc_KBIP` column is the impedance |
| `probe_trackcom.py` | §8: MJWarp on the six `trackcom` domains — the camera control first, then per-surface agreement and the far-band luminance test |
| `visual_calib.py` | §8: the visual tier's three declared numbers, through the shipped env, over three independent draws |
| `parity_spread.py` | §6: the same parity numbers over three draws, which is what `parity_all_exact_min` is set from |
| `install_nvdiffrast.sh` | adds nvdiffrast to the `dmc` venv — `senvdiff` is pinned to mujoco 3.9.0 and cannot load a single DM Control domain |

Tests:

```bash
MUJOCO_GL=egl /mnt/vast/home/randall.balestriero/venvs/dmc/bin/python -m pytest tests/gpu -k DMC -q
# 510 passed, 114 skipped, 3208 deselected in 489.88s
# resources: 18 EGL contexts created, 0 released, 18 live (peak 18)
```

Also landed here, and deliberately not env-specific: **`stable_envs/render/mjwarp.py`** — MJWarp behind
a `render(qpos) -> (B,H,W,3)` surface with the BGRA swap and shadow default built in, plus
`prepare_model` (the zero-alpha pre-pass, for geoms **and** sites, resolving the effective alpha
through the material) and `texture_gaps` (the textured non-plane primitives whose UVs MJWarp cannot
derive). Neither is specific to this family; any MuJoCo model with invisible marker geometry hits the
first silently, and the second is why a textured box renders as one texel.
