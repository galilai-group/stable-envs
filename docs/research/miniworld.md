# MiniWorld

Verdict: **NO-GO.** Two independent arguments, either of which is sufficient, and they fail in
different directions so closing one does not open the other.

1. **The frame cannot be reproduced against even a single pinned driver.** Three separately measured
   terms — trilinear mipmapping (**29–66 % of pixels**), per-vertex Gouraud lighting over whole-wall
   quads (**up to 165/255**), and `GL_POLYGON`/`GL_QUADS` tessellation — each individually exceed the
   entire 10 % budget that `exact_frac >= 0.90` allows. Every one is an *implementation-defined*
   part of the GL pipeline with no written specification to target. All three are measured **on the
   reference alone**, with no second implementation involved.
2. **And separately, there is no single target to aim at.** Two conforming OpenGL implementations,
   given the identical scene, state and multisample count, agree on **3.8–68.4 % of pixels (median
   6.9 %)**.

The structural reason MiniWorld cannot be rescued the way every other 3-D family here was: **the
plate-bake requires a fixed camera, and MiniWorld's camera *is* the state.** Fetch, OGBench and DM
Control all survive an implementation-defined reference by baking the static scene through the
reference renderer on the user's own machine — so the driver-dependent pixels stay locally consistent
— and ray-casting only the movers analytically. MiniWorld is first-person (`agent.cam_pos`,
`agent.cam_dir`), so there is no static image to bake and no pixel that is not a function of the
state. Every pixel would have to be reproduced analytically, including all of the texture
minification in argument 1.

Nothing is registered. `stable_envs/envs/miniworld.py` does not exist. Probes: `scratch/miniworld/`.

Package: `miniworld` 2.1.0, `pyglet` 1.5.31, on an H200 node (NVIDIA 590.48.01) against Mesa
25.2.8 llvmpipe (LLVM 20.1.2). 15 of 23 ids were carried to the deciding gate.

## The four gates

| gate | deciding number | verdict |
|---|---|---|
| **G1 purity, w.r.t. state, one driver** | poses written into an env re-reset under a *different* seed: **max abs diff 0/255** on 15 ids over 6 state pairs each. Re-rendering the same state 5× on one env: **0/255** | **PASS** |
| **G1 purity, w.r.t. the renderer** | *(argument 1)* mipmap term **29–66 %** of pixels, Gouraud term **up to 165/255**, both measured on the reference alone, each larger than the 10 % budget. *(argument 2)* NVIDIA vs Mesa llvmpipe with MSAA pinned equal: **all-exact 0.0382–0.6839, median 0.0450** | **FAIL** |
| **G2 compression** | render slice / frame = **0.0022–0.0211×** against the native 80×60 observation (32–304 B of state against 14,400 B), **0.000022–0.000211×** against the 800×600 `render()` | **PASS** |
| **G3 shared decoder** | **1 distinct static scene over 64 seeds** on 15 ids; 47 texture PNGs and 33 meshes shared by the whole family. 6 ids regenerate the scene per episode and fail: Maze **64/64 distinct**, PutNext **64/64**, PickupObjects **61/64**, MazeS3 **18/64**, RoomObjects **6/64**, MazeS2 **2/64** | **PASS** on 15, **FAIL** on 6 |
| **G4 economics** | official renderer **3,783–3,955 f/s** at the native 80×60 obs, **142–158 f/s** at 800×600 (single env, warm, batch = 1 — the reference has no batch parameter to sweep). Render is **76.2–94.2 %** of an env step | not reached |
| **methodology: interior metric is near-vacuous** | `interior_frac` = **0.16 / 0.68 / 1.45 / 2.80 %** on TMazeLeft / CollectHealth / Hallway / OneRoom (67.60 % FourRooms, 21.12 % Sidewalk are the exceptions). Every surface is textured, so almost no 3×3 neighbourhood is flat | see below |

G4 was never the binding constraint and is recorded only for completeness.

## Argument 1 — unreproducible against one pinned driver

This argument uses **one** GL implementation. It does not depend on llvmpipe, or on any second
renderer existing, and it is the argument that kills the port.

The conformance bar is `exact_frac >= 0.90`: at most 10 % of pixels may differ. Each term below is
larger than that budget on its own, so it is not enough to model each one *well* — each must be
bit-exact. And each is a part of the GL pipeline the specification explicitly leaves
implementation-defined, i.e. there is no document to implement against, only a particular vendor's
binary to reverse-engineer.

**Trilinear mipmapping — 29–66 % of pixels.** Textures are uploaded `GL_LINEAR_MIPMAP_LINEAR` at 512
texels/m (`miniworld/opengl.py:Texture.load`, `miniworld.py:TEX_DENSITY`), so a wall a few metres
away is minified by 1–2 orders of magnitude and every one of its pixels is a trilinear blend of two
driver-built mip levels. Re-rendering the official scene with only `GL_TEXTURE_MIN_FILTER` switched
to `GL_LINEAR` — one GL state change, official renderer on both sides — moves:

| id | all-exact, trilinear vs level-0 | pixels moved |
|---|---|---|
| CollectHealth | 0.3397 | 66.0 % |
| TMazeLeft | 0.3916 | 60.8 % |
| OneRoom | 0.4510 | 54.9 % |
| Sidewalk | 0.5832 | 41.7 % |
| FourRooms | 0.6468 | 35.3 % |
| Hallway | 0.7106 | 28.9 % |

Two things would have to be bit-exact to land inside the 10 % budget: `glGenerateMipmap`'s downsample
filter, which the spec does not fix, and the per-pixel LOD chosen from a screen-space derivative
approximation, which the spec also does not fix.

**Per-vertex Gouraud lighting — up to 165/255.** A whole wall is **one quad**:
`Room._gen_static_data` → `gen_seg_poly` emits 4 vertices per wall segment, floor to ceiling. And
`GL_LIGHT0` is a *positional* light at `light_pos = [0, 2.5, 0]`. Fixed-function GL evaluates the
lighting equation at the 4 corners and interpolates; an analytic per-pixel shader evaluates it per
pixel. Over the real room geometry those differ by:

| id | max | mean |
|---|---|---|
| Sidewalk | **165.1/255** | 5.3 |
| Hallway | **101.5/255** | 14.6 |
| TMazeLeft | **101.5/255** | 8.8 |
| OneRoom | 9.4/255 | 2.6 |
| CollectHealth | 8.7/255 | 2.5 |
| FourRooms | 6.5/255 | 1.1 |

Matching this means reproducing the driver's tessellation and interpolation, not the lighting model —
the model is simple and known; where its output is *sampled* is not.

**Tessellation.** Floors and ceilings are `GL_POLYGON` n-gons and walls are `GL_QUADS`; the split
into triangles, and hence what the interpolation above is interpolating across, is driver-chosen.

## Argument 2 — the target is not well posed

Independently of whether a renderer could be written, there is no single frame to write it against.
`probe4_driver.py` writes four fixed agent poses per id directly into the env — no RNG, no episode
history — and renders them under two GL implementations in two processes.

**Multisample count pinned to 1 (no multisampling at all), both sides:**

| id | all-exact | within 8 | mean | max | interior % | interior-exact |
|---|---|---|---|---|---|---|
| OneRoom | **0.0449** | 0.9988 | 1.844 | 114 | 2.80 % | 0.9964 |
| Hallway | **0.0920** | 0.9991 | 1.355 | 112 | 1.45 % | 0.9956 |
| TMazeLeft | **0.0449** | 0.9992 | 1.595 | 110 | 0.16 % | 0.9486 |
| FourRooms | **0.6839** | 0.9990 | 0.583 | 80 | 67.60 % | 0.9999 |
| Sidewalk | **0.2512** | 0.9987 | 1.246 | 120 | 21.12 % | 0.9907 |
| CollectHealth | **0.0382** | 0.9980 | 1.781 | 84 | 0.68 % | 0.8919 |

Pinning to 4 samples instead of 1 changes these by less than 0.0003. **MSAA is not the cause** —
which had to be checked, because as shipped the two are not comparable: MiniWorld asks for 16
samples, NVIDIA honours it, and llvmpipe caps at `GL_MAX_SAMPLES = 4`. The same-driver control
isolates that term: NVIDIA at 16 samples against NVIDIA at 4 is 0.27–0.76 all-exact, so the sample
count costs real edge pixels and still does not explain a disagreement that survives matching it.

The objection this forecloses: *"llvmpipe is a software fallback nobody trains on — scope the parity
claim to NVIDIA GL."* Scoping does not help, because argument 1 already fails against NVIDIA alone.
What argument 2 adds is that the only way to satisfy argument 1 — reverse-engineering one vendor's
undocumented mip chain, LOD rule and tessellation — produces a renderer whose output is correct on
that vendor and wrong elsewhere by 32–96 % of pixels. That is precisely the claim `conformance.py`
says the repo makes and this family cannot support: *delete your frames, keep the states, get the
same pixels back **on another machine***.

## The interior metric is near-vacuous here, which is its own finding

`interior_frac` — pixels whose 3×3 neighbourhood in the reference is flat — is **0.16–2.80 %** on
four of the six ids measured (0.00–0.93 % on four of six under rollout states rather than the fixed
poses above). Every surface is textured, so almost no neighbourhood is flat. This is the Craftax
situation `conformance.py` already documents, one step further along: interior agreement is not
undefined, it is defined on a few hundred pixels of a 480,000-pixel frame.

This is stated up front because it is exactly how this family could be "rescued" on paper. Quoting
**interior-exact 0.9964** for OneRoom would be true, meaningless, and would conceal a frame that
agrees with itself-on-another-driver on **4.5 %** of pixels. The binding constraint for MiniWorld is
`exact_frac`, and it is missed by ~13× on the median id.

## What passes, recorded so the NO-GO is not read as "MiniWorld is a mess"

15 ids are genuinely clean on everything except the renderer: **OneRoom, OneRoomS6, OneRoomS6Fast,
Hallway, TMaze, TMazeLeft, TMazeRight, YMaze, YMazeLeft, YMazeRight, FourRooms, ThreeRooms, WallGap,
Sidewalk, CollectHealth, Sign**. The state is small and obvious — agent `(x, y, z, dir)` plus
`(x, y, z, dir)` per entity, 8 dims for the single-box envs up to 76 for CollectHealth — the layout
is a constant of the class over 64 seeds, and poses written into a freshly re-reset env reproduce the
frame at 0/255. If the rendering question were ever answered, the rest of the port is straightforward.

MazeS2 is the cautionary entry: its static fingerprint was constant over 8 seeds and **not** over 64,
and the G1 purity check caught at 169/255 what the G3 sample missed. 8 seeds was never an adequate
sample for that gate.

## Three traps, all silent, recorded so they are not rediscovered

**The headless import is an ordering problem, not an env-var problem.** `miniworld/opengl.py` sets
`pyglet.options["headless"]` from `PYOPENGL_PLATFORM`, but it is imported *after*
`miniworld/entity.py` has already done `from pyglet.gl import ...`, and that import creates pyglet's
shadow window at import time (`pyglet/window/__init__.py:1919 → gl._create_shadow_window()`). So
setting `PYOPENGL_PLATFORM=egl` is too late and `import miniworld` still dies with
`NoSuchDisplayException: Cannot connect to "None"`. `pyglet.options["headless"] = True` must be set
before `pyglet.gl` is ever touched. This is why "`import X` succeeding is not the same as X working"
cuts both ways here: on the login node the import fails for want of libGL, and on a GPU node it fails
for want of an X display unless the option is set first.

**Display list 1 is a process-global, and two envs clobber each other.** `_render_static` does
`glDeleteLists(1, 1); glNewList(1, GL_COMPILE)` — a hard-coded id — and pyglet windows share an
object space, so the *last env to reset owns the static geometry for every env in the process*.
Measured: build a TMazeLeft after a OneRoom, re-render the OneRoom at an unchanged state, and
**88.56 % of pixels move by up to 215/255**. Calling `_render_static()` immediately before each
render repairs it exactly (0/255), and every probe here does. This matters beyond MiniWorld: the
conformance suite deliberately keeps one reference env per `(class, size)` alive for the whole run,
so a MiniWorld port would have had to carry that repair in `reference_render` or every parity number
after the first env would have been measured against another env's room.

**llvmpipe silently caps multisampling.** `FrameBuffer.__init__` catches the failure and prints
"Falling back to num_samples=4" — on stdout, not as a warning — so a cross-implementation comparison
that does not pin the sample count is comparing 16× MSAA against 4× and attributing the difference to
the wrong thing.

## What would change the verdict

Only one thing, and it would close both arguments at once: a target that is a specification rather
than a driver. If MiniWorld upstream replaced fixed-function GL + driver mipmaps with an explicit
shader path whose filtering, tessellation and lighting are pinned in the source, the family becomes a
strong candidate — G2 and G3 already pass by wide margins and the state is 8–76 floats. Failing that,
the only honest port would declare `parity_interior_min` against *one* driver on *one* machine, on a
metric that covers 0.16–2.80 % of the frame, which is the claim this repo exists not to make.
