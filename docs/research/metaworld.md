# Meta-World

**Built.** All 50 v3 tasks are registered as `stable_envs.envs.metaworld.MetaWorld*`, in the **parity
tier**, at `parity_interior_min = 0.978` and `parity_all_exact_min = 0.80` with a `parity_note`. The
2026-08-09 NO-GO that this note used to carry is withdrawn twice over: both gates it rested on are
closed, and the port that closes them exists and is measured.

Measured on one NVIDIA H200 via SLURM, `MUJOCO_GL=egl`, `metaworld` 3.1.1, `mujoco` 3.11.0 /
`mujoco-warp` 3.11.0, `warp` 1.16.0, `gymnasium` 1.2.3 (`~/venvs/se-fetch`), through the model camera
`corner`. Probes and the per-task calibration: `scratch/metaworld/`. Raw JSON under
`/mnt/vast/.../se_logs/metaworld/`.

| | |
|---|---|
| registered | **50 / 50** tasks, one class each, `metaworld-<task>` |
| state | `qpos` + the episode's live model slice — **0 to 6 floats**, named per task |
| interior-exact vs the official renderer (480 px, 64 states/task) | **median 0.999700**, min **0.980766**, 48/50 above 0.99, 50/50 above 0.98 |
| all-pixel exact / mean \|diff\| | median 0.879925 / **0.797/255** (max 1.109) |
| throughput | **9.2x at batch 256** (2,313 f/s against 251), 1.6x at batch 1 — **no crossover**; 21.1x at 224 px |
| conformance | **1,600 passed, 350 skipped, 0 failed, 0 crashed** over all 50 (two halves of 25, 20 min each) |

---

## 1. The state gate is closed, on 50/50 tasks

Meta-World writes the episode's placement into the **model** at `reset`, not into `qpos`. That much
was right in the original note. What was wrong is the conclusion drawn from it.

`probe_state_gate.py` walks a ladder of progressively larger restores. Episode A is rolled out, its
state captured, and a DIFFERENT episode B is then restored to it, rung by rung. The control is the
top row: with nothing restored the two frames differ by a median of 239/255, so the test is not
vacuous.

| rung | what is restored | bit-exact |
|---|---|---:|
| R0 | nothing (control) | 0 / 50 |
| R1 | `qpos`/`qvel`/`act`/`ctrl`/`mocap` + `mj_forward` | 5 / 50 |
| **R2** | **R1 + the 14 model arrays** | **50 / 50** |
| R3 | R1 + *every* model array | 50 / 50 |
| R4 | R3 + the derived `data` arrays the renderer reads | 50 / 50 |
| R5 | R1 + `site_xpos`/`site_xmat` written after `mj_forward` | 18 / 50 |

**R2 is exact on every task**: max 0/255, mean 0, zero differing pixels.

### 1.1 What the port actually carries, per task

The env does not store 14 arrays. It resets the task six times with a pinned RNG, takes the
**elements** of those arrays that MOVE between draws, and keeps the ones that reach a pixel — one
MuJoCo render per candidate, measured rather than assumed. The result is 0-6 floats, and they are
named:

| tasks | live model slice |
|---|---|
| 4 (`bin-picking`, `hammer`, `sweep`, `sweep-into`) | **none** — the draw is entirely in `qpos` |
| 18 | `site_pos.goal.{x,y,z}` — the goal marker |
| 20 | `body_pos.{box,door,drawer,window,shelf,dial,lever,faucetBase,coffee_machine,…}.{x,y,z}` |
| 8 | both, e.g. `door-open-v3`: `body_pos.door.{x,y}` + `site_pos.goal.{x,y}` |

The elements that vary and reach no pixel are **pinned on both sides** — `reference_render` restores
the port's own snapshot before writing the state — so an excluded element cannot make the reference
frame history-dependent, and cannot show up as a dead state dimension either. On 50/50 tasks
`check_state_dims_matter` reports no dead dimension and `test_state_is_sufficient` reports 0/255.

One case is worth naming because no downstream test can see it: `sweep-v3` has exactly one candidate
(`site_pos[4]`) that the liveness probe drops. Both sides pin it identically, so it is invisible to
parity AND to state sufficiency; it is visible in `bundle.info["dropped_as_unseen"]`, which exists for
that reason.

### 1.2 Why 3 of the 5 originally probed tasks appeared to resist

They did not. **The reference frame was the stale one, not the restored one.**

`_set_obj_xyz` is what forwards the simulation inside `reset_model`. Several tasks write
`model.body_pos` / `model.site_pos` *after* their last call to it, so `data.site_xpos` and `data.xpos`
still hold the PREVIOUS draw's placement when `reset()` returns.

| | tasks | worst |
|---|---|---|
| stale straight after `reset()` | **28 / 50** (median max 58/255) | 169/255 (`window-close-v3`, `coffee-pull-v3`), `site_xpos` off by up to 0.124 m |
| stale after one `step()` | **0 / 50** | — |

`step()` forwards, so any state a rollout actually visits is clean. This is a **probe protocol rule,
not a Meta-World defect a port has to model**: `reference_render` calls `mj_forward` after writing the
state, and the state pool is read after a rollout rather than out of a bare `reset`.

## 2. The episode draw IS reproducible — by two undocumented routes

`SawyerXYZEnv.reset`'s docstring says *"seed: The seed to use. Ignored, use `seed()` instead."* Both
documented routes fail; two others succeed on every task. Measured as the spread of `_last_rand_vec`
over three draws:

| route | reproducible |
|---|---:|
| `reset(seed=1)` repeatedly | 0 / 50 |
| `env.seed(1); reset()` | 0 / 50 |
| **`env.seeded_rand_vec = True` + reseeding `np_random` before each `reset`** | **50 / 50** |
| **`_freeze_rand_vec = True` with `_last_rand_vec` pinned** | **50 / 50** |

The cause is visible in the source and is not "seeding is broken": `_get_state_rand_vec` falls back to
the **process-global `np.random`** unless `seeded_rand_vec` is set (it defaults to `False`), and
`SawyerXYZEnv.reset` calls `reset_model()` *before* `super().reset()`, so the draw happens before the
seed is applied. The port uses the third route, which is what makes `sample_states(rng=)` regenerate
the same corpus in a fresh process — `test_seeded_sampling_reproduces_in_a_fresh_process` asserts it.

A second fact any port must know: `reset()` runs `reset_model()` **twice** — once itself, once through
`MujocoEnv.reset`'s `_reset_simulation()` — so each `reset` consumes **two** draws. Deterministic, and
only surprising.

## 3. The renderer gate: MJWarp draws these scenes, through the plate

The mesh objection was aimed at `raycast3d`, which is closed-form-per-primitive by design and has no
mesh path. It does not transfer to MJWarp. What had to be measured instead is the split that decides
every port of this shape — **how much of the frame MOVES and is undrawable** — since a static pixel is
baked through MuJoCo itself and is exact by construction.

Frame budget, mean over 12 reachable states, all 50 tasks (`probe_backend.py`):

| | median | min | max |
|---|---:|---:|---:|
| **moving & undrawable** | **0.0000** | 0.0000 | **0.0150** |
| moving & drawable (the Sawyer meshes) | 0.1275 | 0.1232 | 0.2112 |
| static & undrawable (**baked — harmless**) | 0.3538 | 0.3167 | 0.3743 |
| static & drawable | 0.1248 | 0.1041 | 0.1994 |
| sites | 0.0004 | 0.0003 | 0.0014 |
| background | 0.3941 | 0.3225 | 0.3993 |

The 5-10 textured non-mesh geoms the original note flagged are the table and wall — **static**, so
they are baked and MJWarp's UV gap never touches them.

The other half of the plate design also holds, and it is the half Franka Kitchen fails:

| | Meta-World | Franka Kitchen |
|---|---|---|
| shadow-casting lights | **1** (of 3) | 4 |
| shadow pass changes | 42.7 % of the frame | 85.6 % |
| **plate's non-mover INTERIOR exactness, shadows on** | **0.999895** | 0.8338 |

One casting light rather than four is the whole difference — and §5 turns that one light from a
declared residual into a reproduced one.

## 4. What MJWarp does on these models, and the fault that had to be worked around

**MJWarp hard-faults on Meta-World exactly as it does on Franka Kitchen**: CUDA error 700 on
`mjTEXTURE_CUBE`. Every task carries 2-4 cube textures reached by a material.

That fault is **avoidable, not fatal**, and the reason is what the plate does: in the composite a
static pixel comes from MuJoCo's bake, so MJWarp has to get static geometry's **depth** right — a
mover behind the table must be occluded — but never its **colour**. So every material is pointed away
from its cube texture, keeping `mat_rgba` and the skybox, at no cost to any composited pixel.

Three more model-surgery facts, all silent when wrong:

- **`put_model` validates the model's COLLISION configuration even when the caller only wants a render
  context**, and refuses *every* Meta-World model: the Sawyer gripper pads carry `geom_margin` 0.001
  on BOX/MESH pairs, which its MULTICCD path rejects (`mujoco_warp/_src/io.py:_check_margin`). A margin
  reaches no pixel, so zeroing `geom_margin`/`pair_margin` for the render context is sound — and it
  must be restored, because it is the env's own `MjModel`. Note `mjENBL_MULTICCD` **no longer exists**
  on `mujoco.mjtEnableBit` in 3.11, so disabling the flag is not the route; zeroing the margin is.
- **A CUDA 700 poisons the process.** After the first fault every later task reported
  `Failed to allocate 4 bytes on device 'cuda:0'`, which reads like a memory problem and is not one.
  One task per process — which is why the calibration is a 50-way array job and not a loop.
- **`mujoco_warp`'s `kinematics` never recomputes `geom_xpos` for a body welded to the world**
  (`_geom_local_to_global` returns early), so no per-world `body_pos` reaches the table-mounted
  scenery Meta-World places per episode. It DOES walk static bodies for `xpos` (a kinematics branch is
  an ancestor chain), so only the geom transform is missing — and the plate bake has already computed
  exactly that on the CPU. The port uploads it per world. Left out, every episode's movers are
  ray-cast against episode zero's furniture and occluded in the wrong places.

## 5. The shadow the plate cannot hold, and the third bake that puts it back

This is the one thing the original design study did not anticipate, and it is 8.6% of the frame.

Fetch's scene has `castshadow="false"` on its only light, so its plate design never had to answer the
question. Meta-World has one casting light of three, and **hiding a geom to bake the plate hides its
shadow with it**. Measured on `reach-v3`, 8 states at 480 px: the naive composite disagrees with the
official frame on 19.8% of pixels, and **100% of the non-mover disagreement is a pixel the movers
reach** — 8.6% of the frame at a mean of 18.9/255, signed −13..−19 (we were too bright). It lands
almost entirely where the interior mask does not look, because the shadow falls on a wood-textured
table that has no flat 3×3 neighbourhood. That is how a missing shadow hides behind an interior
number of 0.9992.

The port reproduces it with MuJoCo's arithmetic on both branches:

* a MuJoCo pixel is `ambient + shadow * (diffuse + specular)` per light, so a **second bake with the
  casting light's diffuse and specular zeroed** is exactly what MuJoCo draws where that light is fully
  occluded. Both bakes are per-episode, so both are cached with the plate.
* **which** static pixels are shadowed comes from MJWarp, by rendering the same scene twice: once as
  usual, and once with the movers removed from `rc.enabled_geom_ids` — each mover's BVH slot
  re-pointed at a static geom, so the tree keeps its shape and one static surface is simply listed
  twice. There is no per-geom enable flag in `mujoco_warp` to use instead, and a second context would
  double the megakernel compile and the texture upload. The movers-off frame is a function of the
  model, the lights and the camera and NOT of `qpos`, so it is cached on the plate — a 500-step
  rollout through one episode pays one extra graph replay in total.

| `reach-v3`, 8 states, 480 px | naive | + shadow pass |
|---|---:|---:|
| all-pixel exact | 0.8017 | **0.8826** |
| mean \|diff\| | 2.396 | **0.920** |
| within 8/255 | 0.898 | **0.978** |
| interior-exact | 0.99923 | **0.99978** |
| f/s at batch 256 | 2,391 | 2,454 |

The two shadow masks agree on **94.5%** (IoU); what is left is a one-pixel rim where MuJoCo filters a
4096 shadow map and MJWarp casts one ray.

## 6. Two rules for the movers' colours, both measured

**A mover whose texture MJWarp cannot show is drawn in `mat_rgba` × the texture's MEAN.** MuJoCo
paints a textured surface as `texture × rgba`, so the material's colour is a modulation factor and not
the surface's colour; `mat_rgba` alone is that factor with the thing it modulates thrown away. On
`door-close-v3`'s door panel (`dl_metal2`, `mat_rgba` (0.30, 0.32, 0.35), texture mean
(0.577, 0.589, 0.624)) that is 1.7× too bright over 4.7% of the frame.

| 32 states, 480 px | `mat_rgba` | `mat_rgba` × mean |
|---|---:|---:|
| `door-close-v3` mean \|diff\| | 2.959 | **1.105** |
| `door-close-v3` interior | 0.98768 | **0.99275** |
| `box-close-v3` mean \|diff\| | 3.338 | **0.890** |
| `reach-v3` mean \|diff\| | 0.920 | **0.770** |
| `window-open-v3` (no textured mover) | 2.334 | 2.334 |

It never costs anything, because the plate is drawn by MuJoCo with its textures intact and only mover
pixels can move. This is deliberately **not** `mjwarp.bake_primitive_textures`, whose docstring forbids
it on a parity path: that tool answers "should a surface the backend COULD texture be flattened?" and
its answer here is still no. The question here is which constant stands in for a flat draw that is
already forced.

**A translucent mover is blended against the layer behind it, which has to be rendered.** MJWarp's
shader reads only `color[0..2]` and packs a constant 255, so `window_glass` (`rgba="0 .3 .4 .1"`, a
MESH) is painted opaque. Attributing the residual per geom, that ONE geom is **56.9% of
`window-open-v3`'s total error**: 1.56% of the frame at a mean of 88.1/255.

The cheap version — blend the pane against the plate, no extra launch — was tried first and **moves
the wrong way** (window-open mean 2.334 → 2.418), because what is behind a window pane in these scenes
is the rest of the window, which is a mover and is not in the plate. Rendering that layer (the same
`enabled_geom_ids` trick, restricted to the translucent slots) gives both the colour behind and the
segmentation that says whether it is a mover or the plate:

| 32 states, 480 px | opaque | blended against the layer behind |
|---|---:|---:|
| `window-open-v3` mean \|diff\| | 2.334 | **1.113** |
| `window-close-v3` mean \|diff\| | 1.889 | **1.081** |

Unlike the movers-off pass this one cannot be cached (what is behind the glass moves with `qpos`), so
it is gated on the task having a translucent mover at all: **48 of the 50 pay nothing**.

## 7. The measured port

MuJoCo bakes the plate, MJWarp ray-casts the movers, MJWarp's own segmentation buffer selects between
them — the Fetch design, plus §5 and §6. All 50 tasks, 64 states each from the env's own pool, 480 px
(`calibrate.py`, one task per process):

| | median | min | max |
|---|---:|---:|---:|
| **interior-exact** | **0.999700** | **0.980766** | 0.999830 |
| all-pixel exact | 0.879925 | 0.815260 | 0.883819 |
| mean \|diff\| /255 | 0.7971 | 0.4816 | 1.1091 |
| within 8/255 | 0.9783 | 0.9616 | 0.9807 |

| interior-exact ≥ | tasks |
|---|---:|
| 0.999 | 32 / 50 |
| 0.995 | 45 / 50 |
| 0.99 | **48 / 50** |
| 0.98 | **50 / 50** |

The five worst and the five best, with the 224 px column beside them:

| worst | 480 px | 224 px | best | 480 px |
|---|---:|---:|---|---:|
| `window-open-v3` | 0.980766 | 0.989388 | `dial-turn-v3` | 0.999830 |
| `window-close-v3` | 0.984043 | 0.992574 | `plate-slide-back-side-v3` | 0.999822 |
| `drawer-close-v3` | 0.990377 | 0.996128 | `door-lock-v3` | 0.999810 |
| `drawer-open-v3` | 0.990921 | 0.995238 | `hammer-v3` | 0.999785 |
| `door-close-v3` | 0.991085 | 0.994040 | `hand-insert-v3` | 0.999760 |

**The interior number is resolution-dependent, and that is a property of the interior MASK rather than
of the renderer.** At 224 px the same code measures min 0.989388 and median 0.999936; at 480 px, min
0.980766 and median 0.999700. A flat 3×3 neighbourhood is rarer at 224, so less of the mesh-shading
residual is counted as interior. 480 px is the family's default because it is `SawyerXYZEnv`'s own
(`width=height=480`), and the floor is set from what is measured there.

### 7.1 What the residual IS

Per-geom attribution over 8 states (`diag_attrib.py`), after §5 and §6:

| task | top contributor | its share of the error |
|---|---|---:|
| `window-open-v3` | `window_glass` (translucent MESH), then the window's frame meshes | 0.57, then 0.11 / 0.07 / 0.05 |
| `reach-v3` | the Sawyer's arm-link meshes (`geom18`, `geom19`, `geom21`) | 0.22 / 0.20 / 0.15 |
| `door-close-v3` | the Sawyer's arm-link meshes, then the door panel | 0.23 / 0.09 / 0.07 |

so what is left is **mesh shading**: MuJoCo interpolates vertex colours (Gouraud) over triangles and an
analytic per-pixel Blinn-Phong evaluation lands near it, not on it. Looked at SIGNED on the movers'
interior it is `{0: 27.8%, +1: 22.7%, −1: 8.4%, +2: 7.3%, −2: 4.0%, …}` — a broad spread, which is what
interpolation-versus-evaluation looks like and is **not** what a quantisation rule looks like. That
signature is Fetch's `{0: 86.8%, −1: 13.1%}`, and it was found and fixed here too: `mujoco_warp`
truncates float→uint8 where GL rounds (`gl_rounding()`), worth interior 0.791 → 0.9988 on its own.

It therefore bites in proportion to the **mover share of the frame**, which is also why the family sits
below the shared 0.90 all-pixel bar and declares `parity_all_exact_min = 0.80`: a Sawyer plus its
manipulandum covers 12.5-16.7% of this frame against Fetch's 8.9%, while *per mover pixel* this port is
the better of the two (27.8% of the arm's interior pixels are bit-exact here, against ~0 for Fetch's).
The interior floor — the number that describes the renderer rather than the scene — is not relaxed by
that.

### 7.2 Throughput

Report is `(ratio, batch)` with the crossover, per the house rule. All 50 tasks, 480 px:

| batch | f/s (median) | vs MuJoCo render-only (251 f/s) |
|---:|---:|---:|
| 1 | 408 | **1.63x** |
| 8 | 1,496 | 5.91x |
| 64 | 2,211 | 8.79x |
| 256 | **2,313** | **9.23x** (min 4.26x, `window-open-v3`) |

**There is no crossover**: the composite is ahead at every batch size including one. Against the
operation a dataset consumer actually performs — restore the state, then render, 155 f/s — it is
~15x end to end. At **224 px the same code is 21.1x at batch 256** (6,369 f/s against 303), because
MuJoCo's raster is nearly resolution-independent at these sizes while a ray-caster is linear in pixels;
a bare "30x" without its resolution and its batch is not a number anyone can reproduce.

Graph capture is the whole game: un-captured, the same four kernels run at ~430 f/s, which is 1.4x
and not 9x. A throughput number near 1.4x means the capture is not happening.

## 8. The one design cost: the plate is per-episode

The plate is bit-identical across 12 states **within** an episode on **50/50 tasks** (drift 0/255,
`probe_backend2.py`). Across episodes it is not, because the per-episode placement is written into the
model and moves scenery. Closing the mover set under the kinematic subtree narrows this a long way and
does not remove it (`probe_backend4.py`, 50 tasks, 6 episodes):

| mover set | plate constant across episodes | worst drift |
|---|---:|---:|
| bodies a reset writes, directly | 0 / 50 | 255/255 over 10.71 % of pixels |
| **closed under the subtree** | 0 / 50 | **54/255 over 1.46 % of pixels** |

So `_plate`'s cache key is `(task, resolution, device, every style leaf, THE STATE'S MODEL SLICE)` —
the episode, not the appearance, which is the one place this port departs from `fetch.py`. It is keyed
on the exact float32 values rather than a rounded copy, because two states whose placements differ in
the last bit must not share a bake or which of them baked it would depend on the order the batch
arrived in.

The cost is one MuJoCo render (two, with §5's dark bake) per distinct episode rather than per distinct
appearance: ~6% of a 500-step episode's batched rendering, and it is declared rather than discovered.
The env asserts the half that IS true — two states of the same episode must bake the same plate — at
construction, and that assertion earns its keep: it caught 6 of 50 calibration runs on a node whose
EGL raster was returning different pixels for the same scene (`gb-west-a1-gpu-058`; the same six tasks
passed elsewhere, and their near-identical siblings passed on other nodes at the same time).

## 9. Per-task table

`interior` and `mean` are at 480 px over 64 states from the env's own pool; `dim` is `state_dim`,
`mv` the number of mover geoms.

| task | interior | all-exact | mean | f/s@256 | dim | mv |
|---|---:|---:|---:|---:|---:|---:|
| window-open-v3 | 0.980766 | 0.8407 | 1.109 | 1,027 | 14 | 39 |
| window-close-v3 | 0.984043 | 0.8442 | 1.077 | 1,028 | 12 | 39 |
| drawer-close-v3 | 0.990377 | 0.8713 | 0.862 | 2,340 | 12 | 31 |
| drawer-open-v3 | 0.990921 | 0.8746 | 0.826 | 2,362 | 12 | 31 |
| door-close-v3 | 0.991085 | 0.8153 | 1.107 | 2,184 | 14 | 31 |
| door-open-v3 | 0.995606 | 0.8521 | 0.860 | 2,243 | 14 | 31 |
| button-press-v3 | 0.996946 | 0.8644 | 0.920 | 2,322 | 12 | 26 |
| button-press-wall-v3 | 0.997191 | 0.8644 | 0.920 | 2,303 | 13 | 26 |
| plate-slide-v3 | 0.997757 | 0.8811 | 0.785 | 2,208 | 15 | 23 |
| plate-slide-side-v3 | 0.997801 | 0.8806 | 0.787 | 2,176 | 13 | 23 |
| coffee-button-v3 | 0.998197 | 0.8639 | 0.939 | 2,292 | 20 | 27 |
| coffee-push-v3 | 0.998265 | 0.8640 | 0.934 | 2,291 | 23 | 27 |
| coffee-pull-v3 | 0.998292 | 0.8643 | 0.932 | 2,295 | 23 | 27 |
| plate-slide-back-v3 | 0.998298 | 0.8814 | 0.783 | 2,170 | 12 | 23 |
| peg-insert-side-v3 | 0.998624 | 0.8823 | 0.793 | 2,383 | 20 | 22 |
| handle-pull-v3 | 0.998732 | 0.8791 | 0.806 | 2,296 | 13 | 25 |
| stick-pull-v3 | 0.998740 | 0.8692 | 0.838 | 2,181 | 20 | 35 |
| stick-push-v3 | 0.998742 | 0.8691 | 0.836 | 2,176 | 20 | 35 |
| handle-press-side-v3 | 0.999008 | 0.8797 | 0.800 | 2,298 | 13 | 25 |
| handle-press-v3 | 0.999146 | 0.8795 | 0.794 | 2,293 | 13 | 25 |
| shelf-place-v3 | 0.999247 | 0.8832 | 0.777 | 2,323 | 19 | 22 |
| bin-picking-v3 | 0.999597 | 0.8835 | 0.787 | 2,476 | 16 | 23 |
| peg-unplug-side-v3 | 0.999672 | 0.8831 | 0.769 | 2,327 | 22 | 29 |
| assembly-v3 | 0.999695 | 0.8790 | 0.801 | 2,194 | 20 | 34 |
| handle-pull-side-v3 | 0.999700 | 0.8810 | 0.794 | 2,272 | 12 | 25 |
| lever-pull-v3 | 0.999701 | 0.8650 | 0.913 | 2,276 | 14 | 29 |
| button-press-topdown-wall-v3 | 0.999709 | 0.8658 | 0.918 | 2,331 | 12 | 26 |
| faucet-close-v3 | 0.999709 | 0.8643 | 0.920 | 2,347 | 14 | 27 |
| faucet-open-v3 | 0.999709 | 0.8643 | 0.920 | 2,357 | 14 | 27 |
| button-press-topdown-v3 | 0.999712 | 0.8658 | 0.918 | 2,341 | 12 | 26 |
| disassemble-v3 | 0.999733 | 0.8643 | 0.925 | 2,169 | 22 | 34 |
| door-unlock-v3 | 0.999738 | 0.8811 | 0.793 | 2,242 | 12 | 23 |
| basketball-v3 | 0.999740 | 0.8826 | 0.781 | 2,348 | 18 | 24 |
| pick-out-of-hole-v3 | 0.999751 | 0.8838 | 0.785 | 2,394 | 19 | 22 |
| soccer-v3 | 0.999753 | 0.8832 | 0.785 | 2,063 | 20 | 24 |
| pick-place-v3 | 0.999758 | 0.8832 | 0.769 | 2,423 | 19 | 22 |
| pick-place-wall-v3 | 0.999758 | 0.8832 | 0.769 | 2,446 | 19 | 22 |
| reach-v3 | 0.999758 | 0.8832 | 0.769 | 2,454 | 19 | 22 |
| reach-wall-v3 | 0.999758 | 0.8828 | 0.772 | 2,453 | 19 | 22 |
| box-close-v3 | 0.999759 | 0.8669 | 0.891 | 2,393 | 20 | 31 |
| push-back-v3 | 0.999759 | 0.8830 | 0.768 | 2,499 | 18 | 22 |
| push-v3 | 0.999759 | 0.8832 | 0.769 | 2,502 | 18 | 22 |
| push-wall-v3 | 0.999759 | 0.8832 | 0.770 | 2,417 | 18 | 22 |
| sweep-v3 | 0.999759 | 0.8828 | 0.769 | 2,503 | 16 | 23 |
| hand-insert-v3 | 0.999760 | 0.8838 | 0.771 | 2,511 | 19 | 22 |
| sweep-into-v3 | 0.999760 | 0.8837 | 0.772 | 2,542 | 16 | 22 |
| hammer-v3 | 0.999785 | 0.8593 | 0.946 | 2,137 | 17 | 36 |
| door-lock-v3 | 0.999810 | 0.8820 | 0.788 | 2,219 | 12 | 23 |
| plate-slide-back-side-v3 | 0.999822 | 0.8827 | 0.788 | 2,191 | 12 | 23 |
| dial-turn-v3 | 0.999830 | 0.8802 | 0.804 | 2,424 | 14 | 26 |

## 10. Upstream-worthy findings

1. **`mujoco_warp` hard-faults with CUDA 700 on `mjTEXTURE_CUBE`.** Reproduced independently on two
   unrelated model families (Franka Kitchen, all 50 Meta-World tasks) on 3.11.0 and on `main`. **No
   issue for it exists upstream.** The workaround — point materials away from cube slots — is only
   available to designs that discard those pixels.
2. **`put_model` refuses models on COLLISION grounds when only a render context is wanted.** A renderer
   should not need `geom_margin` to be zero.
3. **`kinematics` does not recompute `geom_xpos` for world-welded bodies, but `put_model` accepts a
   per-world `body_pos` for them.** The two together are a silent wrong answer: the batched field is
   accepted, reaches `xpos`, and never reaches the geoms. A shape check cannot see it and a picture can.
4. **No alpha path in the render kernel.** `pack_rgba_to_uint32` writes a constant 255, so any
   translucent geom is opaque. Fetch could hide its four; a window pane cannot be hidden.
5. **`_set_pos_site` writes `data.site_xpos` directly**, which `mj_forward` overwrites, and several
   `reset_model` implementations mutate the model after their last forward. Both make `reset()`-adjacent
   frames history-dependent. One `mj_forward` at the end of `reset` would remove it for all 28 affected
   tasks.
6. **`_get_state_rand_vec` falls back to the process-global `np.random`** unless `seeded_rand_vec` is
   set, and the draw happens before `super().reset()` applies the seed — so `reset(seed=k)` does not
   reproduce an episode and the docstring's recommended `seed()` does not either.

## Reproducing

```
python: ~/venvs/se-fetch/bin/python   (mujoco 3.11.0, mujoco-warp 3.11.0, warp 1.16.0,
                                       metaworld 3.1.1 installed --no-deps, gymnasium 1.2.3)
calibrate: sbatch --array=0-49%12 scratch/metaworld/calibrate.sbatch          # 480 px
           sbatch --array=0-49%12 scratch/metaworld/calibrate.sbatch --size 224
tests:     sbatch scratch/metaworld/run_tests.sbatch -k "<half the class names>"
```

The suite is run in **two halves of 25 envs** and that is a resource statement, not a convenience:
each env holds one `mujoco.Renderer` and its reference env holds another, and the run reports
`context_created 50 / context_freed 0, peak live 50` per half. All 50 in one interpreter is ~100 live
EGL contexts on top of whatever else the suite built, and the measured ceiling this repo has evidence
for is 96 (`stable_envs/testing/conformance.py`). Each half: **800 passed, 175 skipped, 0 failed, 0
crashed** in ~20 minutes on one H200. The 7 skips per env are structural and none of them is a parity
check: no `Scene2D` (the ray-cast path), no batched GPU `step`, not the visual tier, and the two
resolution tests that the plate families delegate to `PLATE_REPRESENTATIVES`. Every env ran
`test_render_parity_with_official_renderer` and `test_state_is_sufficient` — 25 + 25 of each, all
passed.

| script | what it answers |
|---|---|
| `scratch/metaworld/probe_metaworld.py` | per-task model facts, mesh pixel share |
| `scratch/metaworld/probe_state_gate.py` | the state ladder, `reset` staleness, the four seed routes |
| `scratch/metaworld/probe_backend.py` | frame budget, shadow share, plate quality, cube textures |
| `scratch/metaworld/probe_backend2.py` | plate constancy within an episode; first MJWarp attempt |
| `scratch/metaworld/probe_backend3.py` | `full` / `nocube` / `composite`, parity and f/s |
| `scratch/metaworld/probe_backend4.py` | whether subtree closure makes the plate episode-independent |
| `scratch/metaworld/calibrate.py` | **the shipped env**: per-task parity, throughput, every conformance check |
| `scratch/metaworld/diag_residual.py` | is the residual on the movers or on the plate? |
| `scratch/metaworld/diag_shadow.py` | the §5 shadow pass, before and after, with the mask IoU |
| `scratch/metaworld/diag_movers.py` | the signed histogram that separates shading from quantisation |
| `scratch/metaworld/diag_attrib.py` | which GEOM the residual belongs to, per task |
| `scratch/metaworld/diag_mover_colour.py` | §6's A/B, one variant per process |

Constructing a single task (the gymnasium ids are benchmark VECTOR envs, not tasks; the two
single-task ids take `(env_name, seed)` and cannot be given a camera or a resolution):

```python
from metaworld.env_dict import ALL_V3_ENVIRONMENTS
env = ALL_V3_ENVIRONMENTS["reach-v3"](render_mode="rgb_array", width=480, height=480,
                                      camera_name="corner")
env._partially_observable = False
env._freeze_rand_vec = False
env._set_task_called = True
```
