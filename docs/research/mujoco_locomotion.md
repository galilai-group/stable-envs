# Gymnasium's MuJoCo family

**Verdict: all ten ship — four at full parity, six in the visual tier — and the one thing that
decides which is which is the minification filter on the checker floor.**

`Reacher-v5`, `Pusher-v5`, `InvertedPendulum-v5` and `InvertedDoublePendulum-v5` ship in
[`stable_envs/envs/mujoco_locomotion.py`](../api/envs.md) with 100% of interior pixels exact against
gymnasium's own renderer. `HalfCheetah`, `Hopper`, `Walker2d`, `Ant`, `Humanoid` and `Swimmer` are a
**parity NO-GO** and ship in the [visual tier](../concepts/fidelity-tiers.md) instead, at SSIM
0.947–0.981 and 23–27× the official rasterizer. Their frame is a pure function of `qpos` and their
models are mesh-free primitives — the two things a port usually founders on — and they still cannot
reach parity. The numbers are in §5 (the plate + ray-cast design), §6 (MJWarp, re-measured 2026-08-10
with every correction this repo knows about) and §7 (nvdiffrast, which closes the blocker as a
mechanism without closing it to exactness).

> **Correction (2026-08-10).** The first version of this page rested the NO-GO on three blockers and
> gave them equal weight. Re-measured on MJWarp — which renders the whole scene per frame and needs
> no plate — two of the three collapse and one is decisive.
>
> * **"The camera rides the body, so there is no static plate" is a limitation of the plate design,
>   not of the env.** MJWarp implements `mjCAMLIGHT_TRACKCOM` and its camera lands on MuJoCo's to
>   **0.0** (§6.2). It has to be told to: `mujoco_warp.kinematics` does not compute `subtree_com`, so
>   the four-call frame loop this repo uses everywhere else leaves the camera frozen. That is a
>   backend trap, now fixed in `stable_envs.render.mjwarp`, not a property of these six.
> * **The mirror is drawn, and it reaches no pixel the gate scores.** `MatPlane` declares
>   `reflectance="0.5"` on all six and MuJoCo's reflection pass does fire — unlike RoboCasa, where
>   six materials declared 0.5-1.0 and the pass reached 0.0000 of pixels. Here it changes 1.0-3.3% of
>   the frame by up to 81-128/255 — and **0.0000 of interior pixels on all six**. Switching it off
>   moves whole-frame exactness by +0.003 to +0.007 and interior exactness by ±0.005.
> * **The floor's minification filter decides it, and the number is not close.** In the far tenth of
>   the floor MuJoCo's mipmap has averaged the checker to one grey (luminance spread 1.5-3.4) and
>   MJWarp point-samples it at full contrast (spread 92.1-92.7). Bit-exact agreement there is
>   **0.0017-0.0069% of pixels**. Remove the textures from both sides and the same pipeline reaches
>   0.86-1.00 interior exact, which is the flat-material parity `backend3d.md` reports.
>
> The earlier "MJWarp scores 78.2% on antmaze / 38.3% on humanoidmaze" calibration is superseded:
> those figures predate `gl_rounding()` and are about different scenes. The measurement that belongs
> on this page is §6.

All measurements: one NVIDIA H200, driver 590.48.01, `gymnasium` 1.2.3, via `srun` on
`open_research`. §1-§5 on `mujoco` 3.9.0; §6 on `mujoco` 3.11.0 + `mujoco-warp` 3.11.0 + `warp` 1.16.0
in `~/venvs/se-fetch`. Scripts in `scratch/mujoco_locomotion/`, raw JSON under
`/mnt/vast/home/randall.balestriero/stable_envs_logs/mjloco/`.

---

## 1. Step 0: is the frame a function of a compact state?

Yes, on all ten, and it is measured rather than assumed (`probe_state.py`, 128 px, 8 states each):

| probe | result on all ten |
|---|---|
| render the same `qpos` twice | **0/255** |
| draw `qvel` from `U[-100, 100]` instead of zero | **0/255** |
| reach the same `qpos` from the reverse rollout | **0/255** |
| every `qpos` entry moves pixels | yes — **0 dead dims** |

So the render state is `qpos`: 2 floats (InvertedPendulum) to 24 (Humanoid), against 172,800 bytes of
frame at 240 px.

**The render state is not the observation, and the gap runs in both directions.** The locomotion envs
default to `exclude_current_positions_from_observation=True`, so their observation drops root
coordinates the renderer needs; the four that ship here instead *transform* `qpos` — Reacher's
observation carries `cos` and `sin` of its two joint angles rather than the angles, and Pusher's
carries the object and the goal as body positions rather than as the slide coordinates `qpos` holds. `state_names` in the module names the
`qpos` entries, read off the compiled model's joint layout rather than transcribed — a free joint
contributes seven entries and a hand-written list is right until someone adds a joint.

## 2. What actually splits the family in two

Not the state, and not the geometry: `nmesh = 0` on all ten, and every geom is a plane, sphere,
capsule or cylinder, which the analytic ray-caster intersects in closed form.

| env | `nq` | geoms (moving/static) | textures | render camera | static plate over 8 states |
|---|---:|---|---:|---|---|
| Reacher-v5 | 4 | 4 / 6 | 0 | free, frozen | **0/255** |
| Pusher-v5 | 11 | 19 / 1 | 0 | free, frozen | **0/255** |
| InvertedPendulum-v5 | 2 | 2 / 1 | 0 | free, frozen | **0/255** |
| InvertedDoublePendulum-v5 | 3 | 3 / 2 | 0 | free, frozen | **0/255** |
| HalfCheetah-v5 | 9 | 8 / 1 | 3 | model `track` | 204/255 over 32.4% of pixels |
| Hopper-v5 | 6 | 4 / 1 | 3 | model `track` | 202/255 over 32.0% |
| Walker2d-v5 | 9 | 7 / 1 | 3 | model `track` | 204/255 over 33.6% |
| Ant-v5 | 15 | 13 / 1 | 3 | model `track` | 202/255 over 33.8% |
| Humanoid-v5 | 24 | 17 / 1 | 3 | model `track` | 202/255 over 29.1% |
| Swimmer-v5 | 5 | 3 / 1 | 3 | model `track` | 255/255 over 57.0% |

The last column is the decisive one and it is a direct measurement: hide every geom whose body is not
the world body, render eight different `qpos`, and diff. Zero means the static half of the picture is
a constant and can be baked once. It is zero for exactly the four that ship.

## 3. The four that ship

### Design

The same plate + analytic-mover design as [OGBench](ogbench.md) and [DM Control](dmcontrol.md). The
plate is baked **through MuJoCo itself**, with the moving geoms and sites moved to a hidden geom
group, so every static pixel is the reference's own rasterization rather than a reimplementation of
it — 95.6-99.7% of the frame, exact by construction. Only the 2-19 moving primitives are ray-cast per
frame, shaded with the fixed-function composition measured in the [OGBench note](ogbench.md):

```
primary = mat * (A + sum_l D_l * max(0, n.l))  +  sum_l S_l * spec * max(0, n.h)^(128*shininess)
```

Every constant is read off `MjvScene` rather than the XML, because the XML does not contain them:
with no material, each geom resolves to `specular = 0.5`, `shininess = 0.5`, `emission = 0`,
`reflectance = 0`, and the headlight to ambient 0.1 / diffuse 0.4 / specular 0.5. Three of the four
have **no model light at all** (`nlight = 0`); their only light is that headlight, directional along
the camera axis, casting no shadow.

Pusher has one more thing and gets it the way DM Control does. Its model light is a **spotlight**
(`cutoff = 45`, `exponent = 10`, attenuation `(1, 0, 0)`) with `castshadow = 1`, so the arm darkens
the table under it. A second plate is baked with that light's diffuse and specular zeroed — which
*is* the colour of a static surface the light does not reach — and a shadowed pixel selects it.
MuJoCo's shadow *map* is never modelled; only the shape of the shadow, by one occlusion ray.

### Parity

Against `gymnasium.make(id, render_mode="rgb_array").render()` at 240 px, on states drawn from the
env's own reset distribution plus a random-policy rollout. 32 states for the headline row, 256 for
the interior audit.

| env | all pixels exact | interior exact | mean abs | max abs | movers as % of frame |
|---|---:|---:|---:|---:|---:|
| MujocoReacher | 99.640% | **100.000000%** | 0.019 | 44 | 0.35% |
| MujocoPusher | 98.478% | **100.000000%** | 0.068 | 57 | 2.11% |
| MujocoInvertedPendulum | 96.862% | 99.999963% | 0.103 | 32 | 4.42% |
| MujocoInvertedDoublePendulum | 99.659% | **100.000000%** | 0.015 | 28 | 0.32% |

The interior column is audited over **256 states per env**, which is 12.1-14.4 million interior
pixels each (`interior_probe.py`):

| env | interior pixels | disagreeing | where | max |
|---|---:|---:|---|---:|
| MujocoReacher | 12,383,020 | **0** | — | 0 |
| MujocoPusher | 12,112,519 | **0** | — | 0 |
| MujocoInvertedDoublePendulum | 14,445,239 | **0** | — | 0 |
| MujocoInvertedPendulum | 13,401,944 | 5 | all inside a mover | 1/255 |

InvertedPendulum is therefore the only class that declares `parity_interior_min` below 1.0, at
`0.99999` — 27× the measured rate — with the mechanism in its `parity_note`. Those five pixels are
**unchanged at supersample 4**, which is what rules out a coverage explanation and leaves the one in
`docs/adding-an-env.md`: MuJoCo shades curved primitives per vertex over a tessellation, so a large
smooth capsule has flat 3×3 neighbourhoods that a per-pixel evaluation can land one level from. It is
the only env here whose movers are big enough to reach the interior mask at all (4.4% of the frame
against 0.3-2.1%).

### The mover-pixel residual, and a fit that was refuted

Interior parity is the gate, but it is not the number the shader is responsible for: the interior
mask keeps only pixels with a flat neighbourhood, and a shaded capsule almost never has one. Measured
on mover pixels alone, at 240 px:

| | exact | within 1/255 | mean |
|---|---:|---:|---:|
| all mover pixels | 16-35% | 64-78% | 2.0-3.1 |
| eroded by one (coverage removed) | 20-37% | 67-88% | 1.3-2.0 |
| eroded, vs a **non-multisampled** reference | 24-40% | 71-90% | 1.27-1.86 |

The third row locates a large part of the gap in MuJoCo's 4× multisampling rather than in shading.
The rest is per-vertex versus per-pixel, and that diagnosis is measured, not asserted: a least-squares
fit of `(ambient, diffuse, specular, exponent)` on 18,231 mover-interior pixels returns **ambient
0.103 and diffuse 0.388** against the nominal 0.1 and 0.4 — the diffuse law is right — and moves only
the highlight, to a broader, weaker lobe (0.177 at exponent 32 against 0.25 at 64), which is what
interpolating a per-pixel highlight across a triangle looks like.

Adopting that lobe was tried at four settings and **refuted as a family constant**: it improves
InvertedDoublePendulum (within-1 0.872 → 0.885) and InvertedPendulum (0.728 → 0.736), and makes
Reacher (0.685 → 0.668) and Pusher (0.729 → 0.677) worse. The documented law is kept, with the one
scale (`SPECULAR_SCALE = 0.988`) carried from the DM Control study, where an untextured box face at
normal incidence pinned it to [0.04902, 0.04980) against a nominal 0.05 — and re-measured here rather
than inherited: against a nominal 1.0 it improves or ties within-1 agreement on all four domains, by
0.739 → 0.872 on InvertedDoublePendulum and by a fraction of a point elsewhere.

### Throughput

Frames per second at supersample 2, against the official renderer driven the same way (200 frames,
warmed up, single-threaded — which is the only way it runs, since it needs a GL context).

| env | ours, 84 px, B=1024 | official, 84 px | ours, 240 px, B=256 | official, 240 px | ours, 480 px, B=64 |
|---|---:|---:|---:|---:|---:|
| MujocoReacher | 12,867 | 5,116 | 1,734 | 2,939 | 436 |
| MujocoPusher | 1,721 | 536 | 215 | 497 | 54 |
| MujocoInvertedPendulum | 14,377 | 6,489 | 1,954 | 3,502 | 492 |
| MujocoInvertedDoublePendulum | 9,500 | 6,167 | 1,313 | 3,016 | 331 |

**This is the one family in the repo where the official renderer is competitive, and saying so is the
point.** These scenes are 3-20 primitives; MuJoCo's rasterizer costs triangles while an analytic
caster costs rays, so ours wins 1.5-3.2× at 84 px — the resolution pixel RL actually trains at — and
loses 0.4-0.6× at 240 px. What the GPU path buys here is not raw throughput: it needs no GL context
per worker (EGL *degrades* with threads, 185 → 82 fps from 1 to 16 in
[backend3d.md](backend3d.md)), it leaves frames on the GPU with no host round-trip, and it renders a
different appearance per environment in one call. A fused kernel would change the picture — the
caster is a chain of eager elementwise ops and runs an order of magnitude below memory bandwidth —
and that is the obvious next thing to do here.

Two exactness-preserving optimisations are already in, both worth stating because they change nothing
about the output: the ray grid is cached per (scene, supersample) instead of being rebuilt in numpy
per chunk, and the shadow passes run only on rays that can possibly change. The static-shadow
candidate set is exactly `plate != plate_shadow`, since a pixel where the two plates agree cannot
care what the occlusion test returns; on Pusher, whose frame is 84% background, that plus restricting
the mover self-shadow to covered rays took it from 113 to 215 fps.

## 4. Two findings that cost time

### The free camera is frozen at the first render, and the state it saw is in the picture

None of the four has a camera in its XML, so `MujocoRenderer` falls back to `camera_id = -1` and
builds a free camera. `OffScreenViewer._init_camera` then sets its `lookat` to
`median(data.geom_xpos)` **at the moment the viewer is created**, which is the first `render()` call,
at whatever state the env happens to be in. `DEFAULT_CAMERA_CONFIG` overrides only what it names —
Reacher names `trackbodyid`, which a free camera ignores entirely.

So gymnasium's own frame is a function of `qpos` *and* of the state its first render happened to see,
and two processes that reset differently get different pictures from the same `qpos`. Measured
`lookat` after `reset(seed=0)`: Reacher `(0.0213, 0.00034, 0.01)`, Pusher `(0.375, -0.6, 0)`,
InvertedPendulum `(0.00186, 0, 0)`, InvertedDoublePendulum `(0, 0, 0)`.

No renderer can fix that, so it is made explicit instead: **the camera is style.** Its default is the
pose gymnasium freezes after `reset(seed=0)` plus one `render()`, `reference_env` performs exactly
that procedure so the two sides are comparable by construction, and `_check_camera_matches` asserts
it rather than trusting it. `camera.distance_scale`, `camera.azimuth` and `camera.elevation` are then
ordinary variation knobs.

One detail that is a trap rather than a design choice: `camera.azimuth` is declared as a ±60° window
around the frozen default, **not** as `[0, 360]`. A full turn makes the two endpoints the same
camera, and `check_per_env_variation` — which splits the batch between the two ends and demands two
different pictures — would report a knob that works perfectly as collapsed.

### gymnasium's offscreen viewer never binds its GL context

`OffScreenViewer.render` calls `mjv_updateScene` and `mjr_render` without binding its context, and
`MujocoRenderer._get_viewer` re-binds only when one renderer owns several viewers (the rgb_array +
human case). So whatever context was last made current is the one an env's `MjrContext` gets resolved
against — and this class interleaves two envs by design, since the bundle env bakes plates while the
conformance suite's reference env renders the frames compared against them.

The shape of the result is why this is worth a paragraph (`probe_context.py`, four envs at 240 px,
same env, same state, before and after an interloper):

| interloper | effect on the first env's next frame |
|---|---|
| a plain `mujoco.Renderer` | 0/255 |
| a segmentation `mujoco.Renderer` driven with the viewer's own `MjvCamera` | 0/255 |
| a second gymnasium env, same id and size | 0/255 |
| a gymnasium env whose model sets `offsamples=1` | **0.4-8.1% of pixels change, up to 44-84/255** |
| an explicit `make_context_current()` after any of them | back to 0/255 |

The hazard is not "another context exists", it is "another context with a different multisample
configuration became current" — so every cheap test of the obvious kind passes while it is present.
It first appeared as a calibration script reporting 74/255 on frames that were fine an hour earlier.
One `eglMakeCurrent` per frame closes it, and `GymMujoco._shot` is that line.

## 5. The six that do not ship: what rules out the plate + ray-cast design

Their render state is `qpos` (§1) and their geometry is primitives. Three facts about the *picture*
put them outside **this** design. Only the second of the three survives as a reason the domains
themselves cannot be ported, and §6 is where that is settled.

**The camera rides the body.** All six render through the model's `track` camera (`mode="trackcom"`),
a fixed camera id that MuJoCo repositions from the tracked subtree's centre of mass. That is still a
pure function of `qpos` — but it means the static scene moves through the frame, and the plate that
makes this design work does not exist: hiding every moving geom and rendering eight states leaves the
"static" frame differing by **202-255/255 over 29-57% of its pixels**. This rules out the plate. It
does not rule out a backend that draws the whole scene every frame, which is the whole point of §6.

**The floor is a minification-filtered checker.** `texplane` is a `checker` texture with
`texrepeat="60 60"` on a 40 m plane seen from ~3 m, so it is heavily minified toward the horizon and
what the reference draws out there is GL's filtered average of many texels.

Counting distinct colours on the floor is suggestive but not decisive, because the material also
carries `specular="1" shininess="1"` and a mirror, and both vary continuously across a plane. Isolated
by overwriting the checker's texels with a flat grey before the GL context is built — same material,
lights, camera and mirror (`probe_floor_texture.py`, 128 px):

| env | floor as % of frame | checker | flat texture | flat texture, no mirror |
|---|---:|---:|---:|---:|
| HalfCheetah-v5 | 43.5% | 591 | 187 | 70 |
| Hopper-v5 | 42.4% | 526 | 97 | 9 |
| Walker2d-v5 | 45.1% | 499 | 113 | 24 |
| Ant-v5 | 36.7% | 510 | 93 | 71 |
| Humanoid-v5 | 41.7% | 441 | 57 | 22 |
| Swimmer-v5 | **94.9%** | 613 | 249 | 42 |

So the texture is the dominant contributor but not the only one, and the count alone would be a weak
argument. The decisive measurement is the *filter*, and it needs no UV model at all
(`probe_minification.py`): split the floor by image row and compare the spread of luminance in the
nearest tenth against the farthest tenth. Nearest-texel sampling keeps the checker's full black-to-
white contrast at every distance — the far field aliases into speckle — while a minification filter
averages it away.

| env | near band: std (p5 → p95) | far band: std (p5 → p95) | far/near std |
|---|---|---|---:|
| HalfCheetah-v5 | 85.6 (0 → 189) | **5.9** (93.7 → 109) | 0.068 |
| Hopper-v5 | 87.6 (0 → 187) | **0.0** (93.7 → 93.7) | 0.000 |
| Walker2d-v5 | 90.1 (0 → 189) | **0.0** (93.7 → 93.7) | 0.000 |
| Ant-v5 | 83.2 (0 → 189) | **6.8** (93.7 → 109) | 0.081 |
| Humanoid-v5 | 74.9 (0 → 186) | **1.7** (93.7 → 93.7) | 0.022 |
| Swimmer-v5 | 116.9 (0 → 236) | 106.6 (0 → 235) | 0.911 |

Five of the six collapse to a single far-field grey — **93.7 on all five**, which is what the
checker's two colours average to under that lighting — against a near field that spans the full
range. That is the filter, and an analytic caster that resolves one texel per ray paints speckle
there instead.

**Swimmer is the exception, and it fails differently.** Its `track` camera sits at `pos="0 -3 3"`
looking down from close range, so its floor is magnified rather than minified (far/near 0.911) and
the checker is resolved everywhere. Its blocker here is the moving camera, plus GL's *magnification*
filter interpolating between texels — a smaller problem than mipmapping but not nothing on a floor
that is 94.9% of the frame. §6 measures what that costs a renderer that does not need a plate: 0.285
of interior pixels exact, the lowest of the six.

**The floor is a mirror.** `MatPlane` carries `reflectance="0.5"`; zeroing it changes 0.5-3.7% of
pixels by up to 68-101/255. §6.3 re-measures it on the metric the gate actually reads and finds it
reaches **0.0000 of interior pixels**, so this one is a footnote rather than a blocker.

## 6. MJWarp on these six, with every correction applied

The backend [backend3d.md](backend3d.md) recommends draws the whole scene every frame, so the plate
that §5 says does not exist is not something it needs. It was previously ruled out here on two
grounds and both have since weakened: the version constraint (`mujoco_warp` needs `mujoco >= 3.11`,
the conformance interpreter has 3.9) is an environment problem that
[`fetch.md`](fetch.md) §7 and [`dmcontrol.md`](dmcontrol.md) both solved with an isolated venv plus an
import-time guard; and the fidelity figure predates `stable_envs.render.mjwarp.gl_rounding()`, which
on Fetch was worth interior parity 0.791 → 0.9988.

So it was re-measured. Six corrections are applied, each of which is silent when omitted:

| correction | why it is not optional |
|---|---|
| `gl_rounding()` | MJWarp truncates float→uint8, the GL pipeline rounds ([fetch.md](fetch.md) §4) |
| `prepare_model()`, geoms and sites | MJWarp never reads alpha. Fires on nothing here — **0 zero-alpha geoms and 0 sites on all six** — but that is a measurement, not an assumption |
| BGRA → RGB | a byte view of `rgb_data` is B,G,R,A; invisible on grey scenes |
| shadows on, skybox on, groups 0-2 both sides | MuJoCo does all three by default |
| **one material per geom carrying the resolved colour** | MJWarp reads a geom's colour from its *material*, MuJoCo prefers the geom's own `rgba`. All six trip it: `MatPlane` has no `rgba` and the floor geom is `0.8 0.9 0.8`. Uncorrected, MJWarp's floor was 13-19/255 too bright, and it was worth interior **0.016 → 0.939** on the flat-material control |
| **`com_pos` between `kinematics` and `camlight`** | §6.2 |

One further edit reaches no pixel and is needed to load the model at all: `Humanoid-v5` sets
`solver="PGS"`, which `mujoco_warp.put_model` refuses, so the render-only model uses Newton — a
constraint solver the rasterizer never reads, the same class of edit as Fetch zeroing `geom_margin`.
The whole surgery is then **proved inert**: the surgical model rendered through MuJoCo is
bit-identical to the official model rendered through MuJoCo, `exact_frac 1.0 / max_abs 0` on all six.

The reference is `mujoco.Renderer` at the model's `track` camera, which is itself measured
**bit-identical to gymnasium's own frame** (`exact_frac 1.0`, `max_abs 0`, all six). That is what
makes the ablation ladder below comparable to the headline rather than to a different reference.

### 6.1 The headline

240 px, 32 states per env from the env's own reset distribution plus a 0-60 step random-policy
rollout (`mjw_probe4_resolved.py`).

| env | all-pixel exact | interior exact | interior frac | mean /255 | within 8 |
|---|---:|---:|---:|---:|---:|
| HalfCheetah-v5 | 0.3393 | 0.2242 | 0.159 | 44.13 | 0.666 |
| Hopper-v5 | 0.2726 | 0.5323 | 0.255 | 43.49 | 0.640 |
| Walker2d-v5 | 0.2779 | 0.4800 | 0.312 | 45.85 | 0.655 |
| Ant-v5 | 0.3442 | 0.2527 | 0.133 | 36.71 | 0.713 |
| Humanoid-v5 | 0.2708 | 0.5551 | 0.236 | 42.62 | 0.636 |
| Swimmer-v5 | 0.2455 | 0.2851 | 0.742 | 120.70 | 0.468 |

The gate is 90% of all pixels exact. This is 25-34%, and the mean absolute error is 37-121/255, which
is not a parity discussion. It is not a resolution artefact either — at gymnasium's native 480 px the
same pipeline gives 0.3448/0.4974 (HalfCheetah), 0.2823/0.4939 (Humanoid) and 0.2527/0.2831
(Swimmer).

**Essentially all of the interior error is on the floor**, by MuJoCo's own segmentation buffer:

| env | background | floor | bodies | floor's share of interior error | floor interior exact |
|---|---:|---:|---:|---:|---:|
| HalfCheetah-v5 | 52.5% | 43.3% | 4.2% | **0.99997** | 0.193 |
| Hopper-v5 | 54.1% | 42.9% | 3.0% | **0.99572** | 0.130 |
| Walker2d-v5 | 52.0% | 45.0% | 3.0% | **0.99747** | 0.201 |
| Ant-v5 | 52.2% | 37.5% | 10.3% | **0.99996** | 0.217 |
| Humanoid-v5 | 55.4% | 42.1% | 2.5% | **0.99450** | 0.095 |
| Swimmer-v5 | 0.0% | 95.2% | 4.8% | **1.00000** | 0.285 |

The bodies never reach the interior mask at all (a shaded capsule has no flat 3×3 neighbourhood), and
the skybox contributes 0.002-0.008 of the interior error at a mean of 0.07-0.24/255, i.e. a level.

### 6.2 The camera is not the problem, once `com_pos` is called

`mujoco_warp` implements `mjCAMLIGHT_TRACKCOM`, and with the right frame loop its camera lands on
MuJoCo's **exactly**. The right frame loop is not the one this repo uses elsewhere:

| frame loop | camera spread over 8 states | max distance from MuJoCo's `cam_xpos` |
|---|---:|---:|
| `kinematics → camlight → refit_bvh → render` | **0.0** (frozen) | 0.725 m (HalfCheetah), 1.754 m (Ant) |
| `kinematics → com_pos → camlight → refit_bvh → render` | 0.930 / 1.420 m, i.e. the reference's | **0.0** |

`kinematics` writes `xpos/xquat/xmat/xipos` and not `subtree_com`, which is what a `trackcom` camera
reads. Omitted, every frame is drawn from the pose `put_data` uploaded, with no error and a
plausible-looking picture. `stable_envs.render.mjwarp.needs_com_pos` detects the case and
`BatchRenderer` now inserts the launch inside the captured CUDA graph;
`mjw_probe6_batchrenderer.py` asserts the camera moves through that path.

**So blocker 1 does not apply to this backend.** It was a limitation of the plate, and the plate is
gone.

### 6.3 The reflection pass reaches the frame, and reaches no pixel the gate scores

`MatPlane` declares `reflectance="0.5"` on all six — one material each — and unlike RoboCasa's six
declared-but-never-drawn materials, MuJoCo's reflection pass does fire. Measured by zeroing
`mat_reflectance` and diffing the reference against itself:

| env | pixels changed | max /255 | mean /255 | **interior pixels changed** |
|---|---:|---:|---:|---:|
| HalfCheetah-v5 | 1.25% | 128 | 0.65 | **0.0000** |
| Hopper-v5 | 1.19% | 94 | 0.55 | **0.0000** |
| Walker2d-v5 | 1.02% | 117 | 0.55 | **0.0000** |
| Ant-v5 | 2.32% | 106 | 0.82 | **0.0000** |
| Humanoid-v5 | 3.25% | 81 | 0.60 | **0.0000** |

A mirror image of a thin body over a textured floor leaves no flat 3×3 neighbourhood anywhere, so
none of it survives the interior mask. Turning it off on both sides moves whole-frame exactness by
+0.0026 to +0.0068 and interior exactness by −0.005 to +0.002 — in both directions, i.e. noise.
MJWarp implements no reflection pass at all (`grep -i reflect mujoco_warp/_src/render*.py` is empty)
and it does not matter.

**So blocker 3 is a footnote.** It is real, it is drawn, and it is worth half a percent.

### 6.4 The ladder: it is the checker, and only the checker

Every rung modifies **both** renderers identically. Interior exact / all-pixel exact:

| rung | HalfCheetah | Hopper | Walker2d | Ant | Humanoid | Swimmer |
|---|---|---|---|---|---|---|
| L0 the official scene | .224 / .339 | .532 / .273 | .480 / .278 | .253 / .344 | .555 / .271 | .285 / .246 |
| L1 `reflectance = 0` | .224 / .342 | .527 / .275 | .479 / .281 | .254 / .350 | .553 / .272 | .285 / .252 |
| L4 shadows off | .221 / .340 | .531 / .272 | .472 / .278 | .238 / .345 | .553 / .271 | .237 / .205 |
| L2 plane texels ← their mean | .016 / .291 | .216 / .244 | .205 / .228 | .019 / .300 | .220 / .248 | .150 / .141 |
| L3 every `mat_texid = -1` | **.939** / .682 | **.931** / .667 | **.874** / .612 | **.864** / .602 | **.927** / .662 | **1.000** / .944 |
| L5 L3 + `specular = 0` | **.998** / .715 | **.931** / .670 | **.933** / .679 | **.947** / .644 | **.927** / .662 | **1.000** / .945 |

Read down the column. The mirror is worth nothing, the shadows are worth nothing, and **removing the
textures takes interior agreement from 0.22-0.56 to 0.86-1.00** — onto the flat-material parity
`backend3d.md` reports, with `SPECULAR` accounting for most of what is left (L5). On the untextured
scene the MJWarp pipeline in this repo is, as advertised, a match.

The decisive rung is L0 restricted to the far tenth of the floor, where the minification hypothesis
makes a prediction no other hypothesis makes:

| env | far band, % of frame | **all-pixel exact** | reference luminance std | MJWarp luminance std |
|---|---:|---:|---:|---:|
| HalfCheetah-v5 | 3.95% | **0.000069** | 2.65 | 92.69 |
| Hopper-v5 | 3.86% | **0.000028** | 2.78 | 92.07 |
| Walker2d-v5 | 4.00% | **0.000041** | 1.51 | 92.30 |
| Ant-v5 | 2.95% | **0.000018** | 3.07 | 92.63 |
| Humanoid-v5 | 3.12% | **0.000017** | 3.44 | 92.53 |
| Swimmer-v5 (not minified) | 9.58% | 0.2028 | 113.01 | 116.28 |

Seventeen millionths. MuJoCo's mipmap has averaged the checker into one grey out there and MJWarp
paints full black-to-white speckle over it. The near band is not fine either — 0.14-0.21 all-pixel
exact — because at magnification GL's bilinear tap and `wp.texture_sample`'s do not land on the same
texel phase.

**So blocker 2 decides it**, and it decides it on the backend that was supposed to be the way around
the other two.

### 6.5 The two escape hatches, closed by measurement

**"Replace the checker with an equivalent pre-filtered texture."** L2 is the strongest possible form
of that idea: the plane's texels are overwritten with their own mean, so the texture is *constant* and
no sampler, filter or level of detail can disagree about it. The floor still scores **0.0000
all-pixel exact and 0.0000 interior exact**, at an interior mean of 0.85-2.39/255. Compare L3, which
paints the same flat colour through the *material* path instead: floor interior exact 0.86-1.00. The
texture path itself carries a systematic level that `gl_rounding()` does not cover, so a pre-filtered
floor cannot be exact on this backend even before asking whether one texture can be right at two
distances. And it cannot: the required filter width is a function of distance, the camera moves with
the body, so "bake the floor per camera pose" is "call the official renderer once per frame".

**"Supersample."** MJWarp casts one ray per pixel; four other MuJoCo domains in this repo supersample.
Measured at 1×, 2× and 4× with a box average down (`mjw_probe5_supersample.py`, 16 states):

| env | | ss 1 | ss 2 | ss 4 |
|---|---|---:|---:|---:|
| HalfCheetah-v5 | all-pixel exact | 0.3402 | 0.3003 | 0.2932 |
| | interior exact | 0.2278 | 0.0671 | 0.0423 |
| | far-band exact | 0.000082 | 0.0031 | 0.0091 |
| | far-band mean /255 | 99.99 | 39.38 | 14.10 |
| Humanoid-v5 | far-band exact | 0.000000 | 0.000514 | 0.0142 |
| | far-band mean /255 | 99.99 | 46.80 | 7.93 |
| Swimmer-v5 | all-pixel exact | 0.2444 | 0.1137 | 0.0762 |

Sixteen samples per pixel converges the far field toward the mip average — the mean error falls from
100 to 8-14/255 — and reaches 0.6-1.4% exact, because a box average over 16 samples is not GL's
mipmap pyramid and the far field holds hundreds of texels per pixel. Meanwhile everything the
renderer had right gets blurred: whole-frame exactness *falls* on all six and Swimmer, whose floor is
magnified rather than minified, loses two thirds of it. `use_fast_math=False` was swept in the same
run and changes the numbers in the sixth decimal.

### 6.6 Two further limits, recorded so they are not rediscovered

**MJWarp cannot derive UVs for a textured capsule.** `texture_gaps()` returns 4 geoms on Hopper, 17
on Humanoid and 3 on Swimmer — their bodies wear `material="geom"`, whose `texgeom` is a `CUBE`
texture, and `sample_texture` handles `PLANE` and `MESH` only, so those surfaces come out one flat
texel. HalfCheetah, Walker2d and Ant are unaffected (their geoms carry no material). It is not what
decides this page — the bodies are 2.5-10.3% of the frame and reach 0.0% of interior pixels — but it
would be a second declared residual on three of the six.

**The gradient skybox is 52-55% of the frame and 44-55% of it is bit-exact**, at an interior mean of
0.07-0.24/255. So even on the counterfactual untextured scene (L3/L5) whole-frame exactness caps at
0.60-0.72 for five of the six. That is a one-level disagreement on a smooth gradient rather than a
wrong picture, but the gate this repo applies is exactness, and it would fail on its own.

**What would have to change**, in order of difficulty: a plane-texture sampler with level-of-detail
selection that matches GL's minification filter. Warp has the machinery (`num_mip_levels`,
`mip_filter_mode`) and `mujoco_warp` builds every texture at one level and samples with no LOD, so
this is a change to the backend's render kernel — it needs ray differentials the kernel does not
compute — rather than a flag. After that, the skybox sampler and a texture-modulate quantisation rule
in the same class as `gl_rounding()`. Until the first exists, a port would be a low-fidelity picture
with a parity number nobody should trust, and the honest form of that is this page.

## 7. The six DO ship — in the visual tier, on a third backend

**Verdict: §6's parity NO-GO stands, and it is no longer what decides whether these six are usable.**
The operation `mujoco_warp` does not have — a prefiltered, mipmapped texture sample — is what
[`nvdiffrast`](nvdiffrast.md) has by construction, and with it the six ship in the
[visual tier](../concepts/fidelity-tiers.md): mean **1.97–4.39/255**, **83.9–94.5%** of pixels within
8/255, **SSIM 0.947–0.981**, at **23–27× the official rasterizer** at batch 256. They set
`parity_interior_min = None`, which makes the parity gate *inapplicable* rather than relaxed, and they
neither displace nor touch the four parity envs above.

Measured through the env's own surface by
`stable_envs.testing.conformance.check_visual_fidelity` — the same function the conformance suite
asserts against — so the declared numbers and the tested numbers are not two experiments.

### 7.1 What the six declare

240 px, supersample 1, 64 states × **4 independent draws** from each env's own `sample_states`, one
H200. Reference: gymnasium's own `render()`, which is what §6 already measures bit-identical to
`mujoco.Renderer` at the `track` camera — re-checked here at MuJoCo's **default `offsamples=4`**
rather than at the 0 the nvdiffrast study used, `max_abs 0` on all six.

| env | mean /255 | within 8 | **SSIM** | all exact | interior exact | interior mean |
|---|---:|---:|---:|---:|---:|---:|
| HalfCheetah-v5 | 4.09 | 0.873 | **0.9666** | 0.634 | 0.594 | 0.64 |
| Hopper-v5 | 4.35 | 0.846 | **0.9483** | 0.606 | 0.853 | 0.49 |
| Walker2d-v5 | 3.86 | 0.870 | **0.9467** | 0.623 | 0.799 | 0.37 |
| Ant-v5 | 3.43 | 0.894 | **0.9771** | 0.629 | 0.642 | 0.60 |
| Humanoid-v5 | 4.39 | 0.839 | **0.9516** | 0.607 | 0.880 | 0.62 |
| Swimmer-v5 | **1.97** | **0.945** | **0.9810** | 0.462 | 0.542 | 0.46 |

Every column is the WORST of the four draws; the spread across draws is 0.02–0.13/255 of mean and
0.001–0.002 of SSIM, which is what the declared thresholds are set clear of. For scale, the same
metric on the same scenes through MJWarp (§6.1) is mean 36.7–120.7/255.

Interior exactness is reported and **not** asserted — the tier does not claim it, and a number
reported for honesty stops being reported once it becomes a target. It is also the column that reads
backwards: Swimmer is the best env on every other measure and the worst here, because its
floor is 94.9% of the frame and magnified rather than minified, so nearly every pixel of it has the
flat 3×3 neighbourhood the interior mask keeps *and* a magnification filter to disagree about.

### 7.2 The shadow pass, and the one number that decides it

§4.2 of [`nvdiffrast.md`](nvdiffrast.md) names shadows as the largest term left after the floor. They
are drawn here, by one orthographic depth pass over the moving geoms (`_ShadowPass`) — all six light
the scene with a single directional light straight down plus MuJoCo's headlight, which casts nothing,
so the general shadow problem does not arise and the special case is the cheapest one there is.

Same states, same process, shadows the only difference (`nvd_probe6_shadowbias.py`, 32 states):

| env | mean, no shadow | mean, shadow | SSIM, no shadow | SSIM, shadow |
|---|---:|---:|---:|---:|
| HalfCheetah-v5 | 4.85 | **4.09** | 0.9592 | **0.9669** |
| Hopper-v5 | 4.51 | **4.29** | 0.9447 | **0.9488** |
| Walker2d-v5 | 3.98 | **3.79** | 0.9450 | **0.9484** |
| Ant-v5 | 6.21 | **3.50** | 0.9470 | **0.9772** |
| Humanoid-v5 | 4.78 | **4.35** | 0.9468 | **0.9529** |
| Swimmer-v5 | 2.68 | **1.91** | 0.9683 | **0.9831** |

It pays on all six and it is not close to uniform: Ant is worth 2.71/255 and Walker2d 0.19, which is
the difference between a body that lies flat and presents its whole silhouette to a light straight
down and one that stands up and presents almost none of it.

**The bias is the whole pass.** A shadow map holds one depth per texel, so a curved surface is
compared against a depth taken up to a texel from where it actually is, and the first version — a
slope-scaled *depth* bias — put **80% of the pixels it changed on the BODY**, where MuJoCo puts 40%
(`nvd_probe5_shadowmask.py`). That is self-shadow acne, and it cost mean 4.51 → **7.08** on Hopper,
3.98 → 6.55 on Walker2d and 2.68 → 6.00 on Swimmer while *improving* HalfCheetah, which is the shape
of a result that would have been reported as "shadows help one env" if it had not been chased.
Pushing the sample point along the surface NORMAL instead fixes it, and how far is a measurement:

| normal offset (texels) | HalfCheetah | Hopper | Ant | Swimmer |
|---|---:|---:|---:|---:|
| no shadow | 4.85 | 4.51 | 6.21 | 2.68 |
| 0 | 4.13 | 4.36 | 4.99 | 2.60 |
| **1** | **4.09** | **4.29** | **3.50** | **1.91** |
| 2 | 4.17 | 4.30 | 3.50 | 1.91 |
| 5 | 4.24 | 4.30 | 3.50 | 1.91 |

One texel is the best or tied-best on all four and zero is clearly worse, so the constant is one
texel and it is not fitted per env. A 512-texel map was measured against the 256 that ships and is
**not** better (HalfCheetah 3.799 against 3.788 at the old bias) at 12% less throughput, so the map is
sized against the SCREEN — an output pixel is ~1.0 cm on the floor at this camera and 256 texels over
the box the movers occupy is ~1.6 cm.

> **`mjRND_SHADOW = 0` is not "the same frame without its shadows", and treating it as one would have
> settled this wrongly.** Disabling MuJoCo's shadow flag changes **54% of Swimmer's frame by a mean of
> 45/255** and moves the reference AWAY from our (shadowless) render, from 2.63 to 24.96 — so the flag
> carries a global lighting difference on top of the shadow. The budget it appears to give
> (HalfCheetah 1.04/255) is therefore an estimate of the wrong quantity, and every number in this
> section is against the official frame instead.

### 7.3 Refuted here, and recorded so they are not re-run

* **Supersampling.** The obvious answer to the reference's `offsamples=4`, and the parity four use it
  (`default_supersample = 2`). On these six it makes every number worse — HalfCheetah mean 4.88 →
  **6.28**, SSIM 0.960 → 0.944, and the same direction on all six — because supersampling also
  averages the floor TEXTURE and picks the mip level of detail on the finer grid, sharpening exactly
  the surface MuJoCo's mipmap has smoothed. `default_supersample = 1`.
* **Anisotropic filtering**, implemented and refuted in [`nvdiffrast.md`](nvdiffrast.md) §4.1 (far
  band 0.757 → 0.277 at 4 taps).
* **The mirror**, refuted twice: §6.3 for MJWarp and `nvdiffrast.md` §4.2 for this backend
  (−0.012 to +0.005 interior exactness, in both directions).

### 7.4 Throughput, as a (ratio, batch) pair with its crossover

240 px, states in and uint8 frames out — the env's own surface, including MuJoCo's CPU kinematics —
against gymnasium's own renderer on the same machine (328–390 f/s).

| env | B=1 | B=8 | B=64 | **B=256** | B=1024 | official | **ratio at 256** |
|---|---:|---:|---:|---:|---:|---:|---:|
| HalfCheetah-v5 | 254 | 1,565 | 7,010 | **9,802** | 9,621 | 380 | **25.8×** |
| Hopper-v5 | 244 | 1,471 | 6,959 | **9,846** | 9,642 | 358 | **27.5×** |
| Walker2d-v5 | 248 | 1,619 | 6,979 | **9,804** | 9,040 | 378 | **25.9×** |
| Ant-v5 | 233 | 1,493 | 6,597 | **9,104** | 8,351 | 362 | **25.2×** |
| Humanoid-v5 | 236 | 1,528 | 6,402 | **7,466** | 8,835 | 328 | **22.8×** |
| Swimmer-v5 | 244 | 1,519 | 6,950 | **9,806** | 9,586 | 360 | **27.2×** |

**At batch 1 this backend LOSES to the official renderer**, 233–254 f/s against 328–390, and that is
the honest headline number for a caller rendering one frame at a time. **The crossover is batch 2**
(HalfCheetah 488 f/s against 350; Ant 473 against 333), it is 4.1–4.7× by batch 8, and it saturates by
batch 256 — batch 1024 buys nothing and is slower on four of six, because the chunk budget splits it.
Peak device memory is 3.3–3.5 GB at batch 256. The shadow pass costs **~30%** of that (HalfCheetah
13,961 → 9,802 f/s at batch 256) and the throughput above is *with* it, because the declared fidelity
is with it.

Two things had to be fixed to get there, and both are in `nvdiffrast.md` §5's blind spot — that
benchmark drove a SHARED camera, and a tracking camera is per world:

1. **`_ray_dirs` built the skybox's per-pixel ray grid in numpy, on the host, every frame.** It is a
   function of resolution, camera orientation and fovy alone, and the module carried a vestigial
   `key = (R, B, id(cam))` for a cache that was never written. At 240 px it cost **3.6 ms a call** —
   more than the entire rasterization at small batches. Caching it on the orientation's bytes took
   batch 1 from 158 to 376 f/s and batch 64 from 6.8 k to 10.8 k, with no pixel changed.
2. **A per-world camera MATRIX makes that grid `(B, H, W, 3)`,** i.e. 353 MB of host float64 at batch
   256. A `trackcom` camera translates and never rotates, so the env reduces the batch of matrices to
   one when they are bitwise equal (`_one_orientation`) and nvdiffrast expands it: **217 f/s → 6.3 k**
   at batch 256 before the cache above existed. It is exact, not an approximation, and the per-env
   `camera.pan`/`camera.tilt` knobs fall back to the slow path by construction.

### 7.5 Two traps this landing paid for

**A shadow box fitted to the BATCH breaks batch invariance, silently.** The first version sized the
orthographic volume from the widest body in the chunk, which makes a state's own frame depend on what
it was rendered alongside: `render(x)[i]` differed from `render(x[i])` by up to **227/255**, on a
suite whose first assertion is that it does not. Every quantity in `_ShadowPass.build` is per world
now. Nothing about the picture flagged it — the frames looked right in both.

**`MatPlane` is NOT `texuniform`, and assuming it was would have hidden a real difference.**
`nvdiffrast.md` §2.1 records that `mujoco_warp`'s plane texgen omits the size term, which is correct
exactly when `texuniform="true"`. Gymnasium's `MatPlane` declares no `texuniform` at all and MuJoCo's
compiler resolves it to **`texuniform = 0`** on all six (`nvd_probe1_facts.py`), so the size term is
load-bearing here: `texrepeat 60` on a 40 m plane is a slope of 0.75 per metre under the measured law
and 30 under MJWarp's. `nvdiff.texgen` implements the measured law; the point is that the
"does not bite you here" reading was wrong and only a measurement said so.

## 8. What is deliberately not claimed

No `step`. The reference CPU simulator stays the definition of the dynamics, and rendering parity and
simulator parity are different claims that must never be stated in the same words. The conformance
suite reports these four as "no batched GPU simulator declared as checkable", which is accurate.

## Reproducing

```
§1-§5: /mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/python   (gymnasium 1.2.3, mujoco 3.9.0)
§6:    /mnt/vast/home/randall.balestriero/venvs/se-fetch/bin/python         (mujoco 3.11.0, mujoco-warp 3.11.0, warp 1.16.0)
§7:    /mnt/vast/home/randall.balestriero/venvs/senvdiff/bin/python         (nvdiffrast 0.4.0 over the §1-§5 stack)
run:   srun --partition=open_research --account=open_research --qos=normal --gres=gpu:1 ... MUJOCO_GL=egl
tests: pytest tests/ -q -rs -k Mujoco      ->  332 passed, 58 skipped   (all ten envs, both tiers)
       pytest tests/ -q                    ->  2355 passed, 410 skipped (the whole registry)
logs:  /mnt/vast/home/randall.balestriero/stable_envs_logs/mjloco/ and .../nvdloco/
```

One process per environment for everything in §6. Two MuJoCo envs alive in one process can return
byte-identical frames from cameras that provably differ, and §4 records the multisample-interloper
hazard that makes every cheap version of that check pass.

| script | what it answers |
|---|---|
| `scratch/mujoco_locomotion/probe_state.py` | is the frame `f(qpos)`; model facts; is there a static plate |
| `scratch/mujoco_locomotion/probe_scene.py` | resolved lights/materials; the floor-texture and mirror numbers |
| `scratch/mujoco_locomotion/probe_context.py` | the GL-context measurement in §4 |
| `scratch/mujoco_locomotion/parity.py` | parity and mover-pixel agreement per env |
| `scratch/mujoco_locomotion/interior_probe.py` | the 256-state interior audit and the supersample-4 control |
| `scratch/mujoco_locomotion/calibrate.py` | the offsamples=1 comparison, the shading fit, the lobe sweep |
| `scratch/mujoco_locomotion/probe_floor_texture.py` | isolates the floor's colour count from shading and the mirror |
| `scratch/mujoco_locomotion/probe_minification.py` | the near/far luminance-spread test that decides the filter |
| `scratch/mujoco_locomotion/refbench.py` | the official renderer's throughput |
| `scratch/mujoco_locomotion/mjw_common.py` | the MJWarp harness: corrections, scoring, state pool |
| `scratch/mujoco_locomotion/mjw_probe1_facts.py` | is `mujoco.Renderer` at `track` the same reference gymnasium draws; the reflection measurement of §6.3; model census |
| `scratch/mujoco_locomotion/mjw_probe3_compos.py` | §6.2, one arm per process because the second arm inherits the first arm's `subtree_com` |
| `scratch/mujoco_locomotion/mjw_probe4_resolved.py` | §6.1 and §6.4: the resolved-colour surgery, its inertness proof, the ladder, the per-class and near/far attribution |
| `scratch/mujoco_locomotion/mjw_probe5_supersample.py` | §6.5, the supersample sweep and `use_fast_math` |
| `scratch/mujoco_locomotion/mjw_probe6_batchrenderer.py` | the `com_pos` fix through the CUDA-graph-captured `BatchRenderer` |
| `scratch/mujoco_locomotion/nvd_probe1_facts.py` | §7: is gymnasium's frame `mujoco.Renderer`'s at the DEFAULT offsamples; `cam_xpos` against the mono GL camera; is `kinematics + comPos + camlight` enough; `texuniform`; the first (contaminated) shadow budget |
| `scratch/mujoco_locomotion/nvd_probe2_tier.py` | §7.1 and §7.4: the declared numbers over independent draws, batch/chunk/permutation invariance, the downsample bound, the throughput curve |
| `scratch/mujoco_locomotion/nvd_probe3_shadow.py` | §7.2: shadows off / 256 / 512, against the official frame |
| `scratch/mujoco_locomotion/nvd_probe4_context.py` | the one that came back NEGATIVE: an nvdiffrast context does not disturb a live EGL viewer (0/255 through three contexts, a rasterization, a deletion, a second gym env and a `mujoco.Renderer`) |
| `scratch/mujoco_locomotion/nvd_probe5_shadowmask.py` | §7.2: where our shadow lands against where MuJoCo's does, per surface |
| `scratch/mujoco_locomotion/nvd_probe6_shadowbias.py` | §7.2: the normal-offset sweep |

§7 runs in `~/venvs/senvdiff` (nvdiffrast 0.4.0 over the same mujoco 3.9.0 / gymnasium 1.2.3 stack as
§1-§5), raw JSON under `/mnt/vast/home/randall.balestriero/stable_envs_logs/nvdloco/`.
