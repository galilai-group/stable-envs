# Franka Kitchen (`FrankaKitchen-v1`) — feasibility for `stable-envs`

**Verdict: NO-GO — but on ONE gate, not two. Gate 1 has since been closed.**

> **Amended 2026-08-10.** Gate 1 below records `mujoco_warp` hard-faulting with CUDA 700 on this
> model and concludes the scene cannot be loaded at all. **The fault is avoidable.** Pointing every
> MATERIAL away from its `mjTEXTURE_CUBE` slot — keeping `mat_rgba`, keeping the skybox — makes
> MJWarp render this kitchen: 10 slots neutralised, 346 unique colours over the range 4–255, no
> fault (`scratch/kitchen/probe12_nocube.py`). The kitchen's cube textures are the counters,
> cabinetry, walls and oven, which are STATIC and therefore baked through MuJoCo anyway, so in a
> plate composite their MJWarp colour never reaches a pixel. The same workaround is what makes
> [Meta-World](metaworld.md) portable.
>
> **Gate 2 is unchanged and still decides the family**, and Gate 1 is no longer load-bearing for it:
> the second-plate shadow rescue needs the mesh BVH, and the mesh BVH is now reachable. So the
> verdict now rests entirely on four casting lights over 85.6 % of the frame, and on the plate losing
> 0.1662 of the non-mover interior pixels it is responsible for — against 0.0001 on Meta-World, which
> has one casting light. That is a scale-and-light-count argument, and it stands on its own.

The state is fine, the camera is recoverable exactly, and 67.7 % of the frame is static geometry that
would bake perfectly. Neither of those helps, because the shadow design fails on measured numbers:

> **1. `mujoco_warp` cannot render this model at all.** Twelve of the kitchen's fourteen textures are
> `mjTEXTURE_CUBE`, and the render megakernel faults with `CUDA error 700: an illegal memory access`
> on every configuration that enables textures — 9 of 9 tested. Strip the 67 cube-texture slots and it
> runs; keep *only* the cube textures and it faults. With textures disabled — the one configuration
> that executes — the frame differs from the one the family actually produces by **mean 110.86/255,
> with 0.0 of interior pixels exact**.
>
> **2. The plate design is unavailable at this scale and this light count.** The shadow pass changes
> **85.6 %** of the frame. A plate baked through MuJoCo with every mover hidden reproduces
> **1.000000** of non-mover interior pixels with shadows off and **0.8338** with them on — it loses
> **0.1662** to mover shadows alone. DM Control rescues a comparable situation with a second plate,
> but its residual is 0.15–3.0 % and it has one casting light; this scene has four, so the rescue
> becomes up to 16 plates, and its shadow-shape ray-cast needs the mesh BVH that Gate 1 has already
> removed.

The Fetch port works because its one light is `castshadow="false"` and every material has
`reflectance=0`, which is what makes 91 % of a Fetch frame a function of the appearance alone. A
shadow-casting light does not by itself remove the plate — DM Control ships one and stays
interior-exact by baking a second plate with that light switched off. What removes it here is the
scale (85.6 % of the frame against DM Control's 0.15–3.0 %), the light count (four casting lights
against one, so up to 16 plates rather than two), and the fact that the rescue's own shadow-shape
ray-cast needs a mesh BVH this backend cannot load. See
[Why DM Control's second-plate shadow trick does not rescue this](#why-dm-controls-second-plate-shadow-trick-does-not-rescue-this).

All numbers measured on an H200 via SLURM under `~/venvs/se-fetch` (mujoco 3.11.0 / mujoco-warp 3.11.0
/ warp 1.16.0), gymnasium-robotics 1.4.2, at the family's native 480 px.
Probes: `scratch/kitchen/probe*.py`.

---

## What passes

**The camera, exactly.** `FrankaKitchen-v1` is the case Adroit is not: its `DEFAULT_CAMERA_CONFIG`
sets all four fields (`distance` 2.2, `azimuth` 70.0, `elevation` −35.0, `lookat` (−0.2, 0.5, 2.0)),
so nothing falls through to gymnasium's `median(geom_xpos)` default and the viewpoint is a constant of
the family rather than of the history. Reading the pose off MuJoCo's own `mjvScene` as the average of
the two `mjvGLCamera` eyes and re-expressing it as a model camera reproduces the official free-camera
frame **bit-identically: 1.000000 exact, max 0/255**.

That comparison was re-taken in **separate processes** and only then believed. As first measured it
had a `mujoco.Renderer` and gymnasium's `OffScreenViewer` live in one process, and
`OffScreenViewer.render` never binds its GL context — so both renders can resolve against whichever
EGL context was created last, and "the two frames are identical" is exactly what that failure looks
like. The frames were also checked to be non-degenerate first (6,829 unique colours, full 0–255
range), because a dead context returning a constant makes parity pass against nothing.

**Compression.** The render-relevant state is `qpos`: 30 float64 = 240 B against a 691,200 B frame,
0.00035×. The task/goal dictionary reaches no pixel. Reset is a single fixed state
(`reset_model` restores `init_qpos` with no randomisation at all), so a state pool has to come from
random-action rollouts plus the object joints driven over the range `OBS_ELEMENT_GOALS` defines — the
same shape as Fetch's pool, and the distribution used for every number below.

**The static half is genuinely static.** With every mover hidden *and every site hidden*, the plate is
state-independent to **0/255**. 67.7 % of the frame is drawn by static geometry.

## Gate 1 — `mujoco_warp` faults on cube textures

| configuration | textures | result |
|---|---|---|
| baseline (shadows, specular, skybox, 16 worlds) | on | **CUDA 700** |
| no shadows | on | **CUDA 700** |
| no specular | on | **CUDA 700** |
| no skybox | on | **CUDA 700** |
| 1 world | on | **CUDA 700** |
| 3 lights only / 1 light only | on | **CUDA 700** |
| baseline | **off** | renders |
| no shadows | **off** | renders |

Pinned to the texture *type* rather than left at "textures break it":

| model | result |
|---|---|
| all textures | CUDA 700 |
| 67 cube-texture slots stripped, 2-D + skybox kept | **renders** |
| 2-D + skybox stripped, cube textures kept | CUDA 700 |

`marble1.png`, `metal1.png` and `wood1.png` are declared `type="cube"` and carry the counters, the
cabinetry, the walls, the oven, the microwave and the kettle. This is a different limitation from the
UV gap `stable_envs.render.mjwarp.texture_gaps` already documents (which is a *wrong colour*, one
texel, on `BOX`/`CYLINDER`/`SPHERE`/`CAPSULE`); this one is a crash, and it takes the textured meshes
with it.

Running with textures off is not a fallback. Against the frame the family actually produces:

| | mean \|diff\| | exact | within 8 | interior exact |
|---|---|---|---|---|
| `mujoco_warp`, textures off, vs official | **110.86/255** | 0.0000003 | 0.0079 | **0.0** |

## Gate 2 — the movers shadow 85.6 % of the frame, across four casting lights

Which pass is responsible was measured rather than assumed, and the answer is one-sided:

| pass | share of pixels it changes | max |
|---|---|---|
| shadow | **0.8559** | 201/255 |
| reflection | **0.0000** | 0 |

The reflection result is worth stating because the MJCF reads the other way: `counter_black`,
`counter_blue`, `kettle_*`, `micro_*` and `oven_metal` all declare `reflectance` 0.5–1.0. MuJoCo's
reflection pass does not fire on any of them, so reflections are a non-issue here and shadows are the
whole story.

Plate quality, restricted to the pixels where no mover is drawn — the pixels a composite would take
from the plate — over 12 states from the env's own distribution:

| | non-mover exact | non-mover mean | **non-mover INTERIOR exact** |
|---|---|---|---|
| shadows on | 0.7924 | 5.38/255 | **0.8338** (worst state 0.7605) |
| shadows off | 0.9914 | 0.21/255 | **1.000000** |

The mover silhouette is 0.2993 of the frame, so this is not a small correction applied to a small
region: roughly a fifth of the pixels the plate is responsible for are wrong, because the arm, the
doors and the kettle throw shadows across the counters and the floor.

## Why DM Control's second-plate shadow trick does not rescue this

The obvious objection to Gate 2 is that this repo *already* bakes plates under a shadow-casting light.
`stable_envs/envs/dmcontrol.py` does exactly that — a 45° spotlight with `castshadow=1` under
`<quality shadowsize="2048"/>`, over a floor with `reflectance=.2` — and it reaches interior-exact with
`parity_interior_min` unrelaxed and no `parity_note`. It does it with a technique this document should
not pretend does not exist:

> a **second plate** is baked with the shadow-casting light's diffuse and specular set to zero, which
> is exactly the colour a static surface has when that light does not reach it, and a shadowed pixel
> selects the second plate instead of the first. Only the *shape* of the shadow is left to the
> ray-caster. The mirror is handled the same way, with a **third plate** at `mat_reflectance = 0` and
> MuJoCo's own compositing rule `frame = reflectance * mirror + (1 - reflectance) * surface`.

**The technique is sound and it is what the repo does. It fails here on scale and light count, not in
principle.** Three measured differences, in the order they bite:

| | DM Control | Franka Kitchen |
|---|---|---|
| shadow + reflection residual | **0.15 % – 3.0 %** of the frame (point_mass 0.15, acrobot 1.7, pendulum 2.9, cartpole 3.0) | shadow pass changes **85.6 %** |
| what the plate loses to shadows | a one-pixel band at the shadow edge | **0.1662** of non-mover interior pixels (1.000000 → 0.8338) |
| shadow-casting lights | **1** (the headlight is directional, sits at the camera, and never casts) | **4** — three directional plus the `ovenlight` spot, all `castshadow=1` |

The light count is the part that does not scale. With one casting light there are two states a static
pixel can be in — lit or occluded — so the rescue is one extra bake. With four, a static pixel can be
occluded by any *subset* of them, so the generalisation is a plate per occluded-light subset, up to
**16** per appearance, plus a per-light occlusion mask to choose between them. That is not the same
correction applied harder; it is different bookkeeping, and it is applied to most of the picture
rather than to a one-pixel rim.

**Gate 1 removes this rescue as a side effect, which is what makes the verdict robust.** Producing the
shadow *shape* is still a ray-cast, and DM Control's is `stable_envs/render/raycast3d.py`, which ships
analytic primitives only — spheres, capsules, boxes, planes. That is sufficient there because those
domains are built from primitives. It is not sufficient here: **32 of the kitchen's 118 visible geoms
are meshes**, and they are the movers that matter — the Franka arm, both cabinet doors, the microwave
door, the kettle. Casting occlusion rays against those needs a mesh BVH, which on this stack means
`mujoco_warp`, which hard-faults on this model. So even if the 16-plate bookkeeping were affordable,
the backend that would have to draw the shadow edges cannot load the scene.

## The remaining design, and its ceiling

With the plate gone the only design left is full-scene `mujoco_warp`, which ray-casts its own shadows.
Measured with textures off on both sides (the only way both renderers can be pointed at the same
scene), against MuJoCo rendering the same surgical model:

| scene | shadows | interior exact | mean | within 8 |
|---|---|---|---|---|
| kitchen | on, both sides | **0.1581** | 24.22/255 | 0.491 |
| kitchen | off, both sides | **0.4411** | 10.43/255 | 0.837 |
| **Fetch `reach.xml`, same harness** | on, both sides | **0.7982** | 0.78/255 | 0.985 |

The Fetch row is the control, and it is why the kitchen rows are believable. Full-scene
`mujoco_warp` on a flat-material scene with no shadows reaches 0.798 interior — and the shipped Fetch
env reports 0.992 precisely *because* it does not render the scene that way: it bakes 91 % of the
frame through MuJoCo and ray-casts only the 9 % that moves. The kitchen cannot make that trade, and
without it the same code is five times worse on the same hardware.

MuJoCo filters a 4096 shadow map; `mujoco_warp` casts hard any-hit shadow rays. Over 85.6 % of the
frame that is a different picture, not a different rounding.

## Frame budget, by sub-limit

Mean over 12 states, from MuJoCo's segmentation buffer (which is `(H, W, 2)` = **(id, type)**, not
(type, id) — verified by resolving the ids back to named geoms):

| | share | fixability |
|---|---|---|
| static geometry | 0.6768 | exact, if it could be baked |
| movers | 0.2992 | ray-cast |
| — of which textured non-mesh **movers** | 0.0380 | partially: re-express as a mesh with cube texcoords |
| textured non-mesh **statics** | 0.1448 | harmless, baked through MuJoCo |
| alpha < 1 (microwave window, on the door) | 0.0189 | nothing — `mujoco_warp` has no alpha path |
| sites (`end_effector`, on `panda0_link7`) | 0.0001 | re-express as a geom |
| skybox | 0.0239 | — |

The two limits the brief flagged as the first gate — 3.8 % of the frame to un-UV-able moving
primitives and 1.9 % to alpha — are real and are **not** what kills this family. They would have been
declarable in a `parity_note`. The crash and the shadows are what kill it.

## Measurement traps hit, and what each one said before it was caught

- **`model.vis.quality.shadowsize = 0` after a `Renderer` exists is a no-op.** The shadow FBO was
  allocated when the `MjrContext` was built. The first contamination probe therefore reported
  *byte-identical* numbers for shadows on and shadows off, which reads exactly like "shadows are not
  the cause" — the opposite of the truth. The working switch is `scn.flags[mjRND_SHADOW]`, on
  `MjvScene`, not on `MjvOption`. (Independently recorded in `robocasa.md` §6.)
- **A visible site on a moving body is not a mover by `body_weldid`.** `end_effector` (group 0, on
  `panda0_link7`) stayed visible in the first plate bake and made the "static" plate depend on
  `qpos[0:7]` — up to 149/255 over ~180 pixels. It looked like a shadow effect and was a sphere.
  `mujoco_warp` draws no sites at all, so every site has to be hidden or re-expressed as a geom.
- **`MjModel.light_directional` no longer exists in mujoco 3.11**; it is `light_type`.
- **Two viewers in one process cannot be compared** — see the camera section above.
- **Six materials declare `reflectance` 0.5–1.0 and MuJoCo's reflection pass does not fire on any of
  them** (0.0000 of pixels changed, max 0/255). `counter_black`, `counter_blue`, `kettle_*`, `micro_*`
  and `oven_metal` all read like mirrors in the MJCF, and the floor's `MatPlane` carries
  `reflectance='0.05'`. Budgeting a third plate for the mirror — which is what DM Control needs — would
  have been work spent on a pass that never runs here.

## What would reopen this

**One thing now, not two.** Gate 1 was closed on 2026-08-10 by the cube-texture workaround recorded
at the top of this document, so what remains is:

- a shadow treatment that survives four casting lights over 85.6 % of the frame — either a shadow
  model agreeing with MuJoCo's filtered map, or the plate-per-occluded-light-subset generalisation of
  DM Control's technique, up to 16 bakes per appearance plus a per-light occlusion mask.

The measured target for that work is now concrete rather than hypothetical: the plate has to recover
the 0.1662 of non-mover interior pixels it currently loses to mover shadows (0.8338 → 1.000000).
Meta-World reaches 0.999895 on the same measurement with one casting light, so the technique is sound
and the light count is the whole gap.

Even with both, the full-scene ceiling measured on a *simpler* scene through the same harness is 0.798
interior, against a repo floor of 1.0 that only the plate has ever reached. The honest reading is that
Franka Kitchen needs a rasterizer, not a ray-caster — and the rasterizer it needs is the official one,
which is not a port.
