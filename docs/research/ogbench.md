# OGBench maze envs — can a frame be a pure function of a compact state?

**Scope**: the 25 OGBench maze cells (`pointmaze` / `antmaze` / `humanoidmaze` × `medium` / `large` /
`giant` / `teleport` × `navigate` / `stitch` / `explore`). The dataset variant (navigate/stitch/explore)
never touches the renderer, so there are only **11 distinct scenes**: 3 locomotion bodies × 4 maze layouts,
minus `humanoid-teleport`, which is not one of the 25.

**Verdict up front**

| family | verdict | evidence |
| --- | --- | --- |
| `pointmaze` (8 cells) | **EXACT-reproducible** | 99.64–99.95% of pixels bit-identical to MuJoCo at its default settings; residual confined to the agent's 24–150-pixel disc |
| `antmaze` (11 cells) | **EXACT-reproducible** | 99.92% bit-identical; agent is 11–21 px of 50176 |
| `humanoidmaze` (6 cells) | **EXACT-reproducible** | 99.93% bit-identical; agent is ~24 px |

No family is a NO-GO. Every scene is built from **analytic primitives only — `nmesh = 0` in all 11
models** — and MuJoCo's per-pixel geometry is reproduced by an independent ray-caster at **100.000%
agreement on interior pixels, 0 mismatched interior pixels, in every scene tested**.

The one caveat that keeps this from being 100.000% rather than 99.9%: MuJoCo lights **per vertex**, and
the agent's curved bodies are drawn as tessellated meshes, so their *interior shading* is a Gouraud
interpolation that a per-pixel Blinn-Phong evaluation approximates to ~3/255 rather than matching exactly.
Silhouettes are exact; interiors are off by a few levels on a few dozen pixels. See
[the gap](#closing-the-shading-gap-what-was-wrong-and-what-is-left).

Everything below was measured on SLURM (`MUJOCO_GL=egl`, H200); scripts in
`scratch/ogbench/`, raw JSON in `scratch/ogbench/out/`.

---

## 1. The minimal state vector

**A frame is exactly `f(qpos)`. Nothing else.** Measured by rendering the same `qpos` under adversarial
conditions and diffing (`scratch/ogbench/purity.py`, 224px, 7 scenes):

| scene | nq | nv | render twice | qvel × 100 randomized | same qpos, different rollout | qpos entries that move pixels |
| --- | --- | --- | --- | --- | --- | --- |
| pointmaze-{medium,large,giant,teleport} | **2** | 2 | 0 px differ | 0 px differ | 0 px differ | 2 / 2 |
| antmaze-{large,giant} | **15** | 14 | 0 px differ | 0 px differ | 0 px differ | 15 / 15 |
| humanoidmaze-large | **28** | 27 | 0 px differ | 0 px differ | 0 px differ | 27 / 28 |

- `qvel` is **render-irrelevant**: scaling it by 100 changes exactly zero pixels. The frame depends on
  `qpos` only through forward kinematics (`geom_xpos` / `geom_xmat`).
- No hidden renderer state: reaching the same `qpos` via a different rollout (different task, different
  number of steps) gives a bit-identical frame.
- So the "dataset of states" is: `pointmaze` **2 floats**, `antmaze` **15 floats** (7 free-joint + 8
  hinges), `humanoidmaze` **28 floats** (7 free-joint + 21 hinges). At 224×224×3 = 150,528 bytes per
  frame, that is a **compression of 9,400× / 1,250× / 670×** against float32 states (and the states are
  already stored in the OGBench datasets).

**Maze layout is static per env.** The walls are `geom`s added to the XML at construction from a fixed
`maze_map` (`ogbench/locomaze/maze.py`), never moved. Wall grid: unit 4.0, height `0.5*4 = 2.0`
(box from z=0 to z=2, half-extent 2×2×1), block `(i,j)` centred at `(4j-4, 4i-4, 1)`. The floor plane is
sized to exactly cover the maze: `pos = (2(C-3), 2(R-3), 0)`, `size = (2C, 2R, 0.2)` for an `R×C` map.
`teleport` mazes add 5 cylinders (radius 1, half-height 0.05, at z=0.05) that are also static.

**`decolor_floor` / `mark_agent_red`** (rjepa's wrapper, `rjepa/envs/ogbench_maze.py`) are *static model
edits*, not per-frame effects:

- `decolor_floor` overwrites the 300×300 `grid` texture. OGBench's pixels mode fills it with a per-texel
  `[r, g, 128]` position ramp (a position leak); rjepa replaces it with a single flat colour
  **(60, 64, 72)**. Measured effect: **40–66% of pixels change**, and afterwards the texture has exactly
  **1 unique colour**. Consequence for a rasterizer: the floor's base colour becomes a constant, so no
  texture sampling is needed at all.
- `mark_agent_red` recolours the free-joint subtree to `(0.90, 0.12, 0.12, 1)` and scales the torso geom
  by 4×. Applied to **humanoid only**. Measured effect: **0.1% of pixels**. It changes `geom_rgba` /
  `geom_size`, i.e. static model params.

---

## 2. How the official render actually works

Dumped from the compiled `MjModel` after OGBench's `ob_type='pixels'` XML surgery
(`scratch/ogbench/probe_scene.py` → `out/scene_*.json`).

### Camera — top-down, **perspective**, fixed per env

`maze.py` builds a free `MjvCamera` with `elevation = -90`, `lookat = (2(C-3), 2(R-3), 0)`,
`distance = 5(C-2)`, azimuth left at its default 90. `vis.global_.fovy = 45`, `orthographic = 0`.
The pose MuJoCo actually computes:

| scene | camera pos | forward | up | fovy |
| --- | --- | --- | --- | --- |
| medium (8×8) | (10, 10, 30) | (0, 0, −1) | (0, 1, 0) | 45.0000° |
| large (9×12) | (18, 12, 50) | (0, 0, −1) | (0, 1, 0) | 45.0000° |
| giant (12×16) | (26, 18, 70) | (0, 0, −1) | (0, 1, 0) | 45.0000° |
| teleport (9×12) | (18, 12, 50) | (0, 0, −1) | (0, 1, 0) | 45.0000° |

Two things that a reading of the source gets wrong and a measurement gets right:

1. **It is perspective, not orthographic.** Walls are 2 units tall under a camera 30–70 units up, so wall
   *side faces* are visible and grow toward the frame edge. A 2-D top-down rasterizer cannot draw them.
2. **`mjv_updateScene` fills two `mjvGLCamera`** (left/right eye, separated by `vis.global_.ipd = 0.068`),
   and mono rendering uses **their average**, not `camera[0]`. Scored on MuJoCo's own segmentation buffer:

   | camera candidate | pointmaze-large | antmaze-large | humanoidmaze-large |
   | --- | --- | --- | --- |
   | `camera[0]` (left eye) | 0.99281 | 0.99275 | 0.99281 |
   | `camera[1]` (right eye) | 0.99516 | 0.99504 | 0.99512 |
   | **average** | **1.00000** | **1.00000** | **1.00000** |

   Using `camera[0]` costs ~0.7% of pixels — a 34 mm lateral offset, invisible by eye, obvious in a diff.

### Lights — exactly two, both **directional, straight down**

| light | type | dir | ambient | diffuse | specular | castshadow |
| --- | --- | --- | --- | --- | --- | --- |
| MuJoCo headlight (`vis.headlight`, active) | directional, along camera forward | (0, 0, −1) | .1 .1 .1 | .4 .4 .4 | .5 .5 .5 | 0 |
| model light `global` | directional | (0, 0, −1) | **0 0 0** | 1 1 1 | .1 .1 .1 | 1 |

The `global` light's ambient is `.2 .2 .2` in the XML; **OGBench deletes it in pixels mode**
(`light.attrib.pop('ambient')`), and also deletes the `torso_light` that would otherwise track the agent.
Because the camera looks straight down, the headlight direction *coincides* with the model light: one
effective light direction `l = (0, 0, 1)`.

### Materials

Every material in every scene has `specular = 0.5`, `shininess = 0.5`, `reflectance = 0`, `emission = 0`.
Base colours after OGBench's pixels surgery + rjepa's decolor:

| surface | base rgb |
| --- | --- |
| floor (`grid` material, flat texture) | (0.2353, 0.2510, 0.2824) = (60, 64, 72)/255 |
| walls (`wall`, set to `.6 .6 .6` in pixels mode) | (0.6, 0.6, 0.6) |
| point agent / ant (`self`) | (0.7, 0.5, 0.3) |
| ant's white leg (`self_white`) | (0.8, 0.8, 0.8) |
| humanoid (marked) | (0.90, 0.12, 0.12) |

### Which render effects matter

Per-flag ablation with the baseline restored each time (`scene.flags` are **not** reset by
`mjv_updateScene` — without restoring, the ablation silently becomes cumulative):

| flag | default | pixels changed |
| --- | --- | --- |
| `mjRND_SHADOW` | on | 43–66% (see below) |
| `mjRND_SKYBOX` | on | 0% (medium) / 9.8% (large, giant — the frame is larger than the floor) |
| `mjRND_REFLECTION` | on | **0%** |
| `mjRND_FOG` / `mjRND_HAZE` / `mjRND_CULL_FACE` | — | **0%** |

**There are essentially no cast shadows.** The `mjRND_SHADOW` number is misleading: switching it off puts
MuJoCo on an unlit path where the floor renders at exactly its base colour (measured shading scalar
`1.000` on 31867/31870 floor pixels). Genuine occlusion — a pixel that is darker *because* something is
above it — covers **0.016–0.054% of the frame**, and an analytic "is any geom above this point" test
reproduces that mask at **100.0% agreement** on `pointmaze-large` and `antmaze-large`. With a light
pointing straight down, wall shadows fall exactly under the walls.

### Shading law (fitted, not assumed)

The measured ratio `observed / base_colour` is identical across R, G and B to 4 decimals on the floor —
an additive white specular term cannot do that — so the shading is a **pure multiplier**. That is
fixed-function OpenGL folding specular into the primary colour before texture modulation. Fitting
`S = A + D·max(0, n·l) + Sp·max(0, n·h)^p` jointly on floor pixels (`n·l = 1`) and wall-side pixels
(`n·l = 0`, which is what breaks the A/D degeneracy) over a grid of `p`:

| scene | A | D | Sp | p | RMS(S) |
| --- | --- | --- | --- | --- | --- |
| pointmaze-medium | 0.098039 | 1.3892 | 0.2612 | 59 | 0.0221 |
| pointmaze-large | 0.098039 | 1.3984 | 0.2532 | 63 | 0.0124 |
| humanoidmaze-large | 0.098039 | 1.4012 | 0.2505 | 64 | 0.0023 |

Consistent with the light table (`A = 0.1`, `D = 0.4 + 1.0 = 1.4`, `Sp = (0.5 + 0.1)·0.5 = 0.3`,
`p = 128·shininess = 64`) except for the specular weight, which measures 0.25–0.26 rather than 0.30. The
half-vector is the **local-viewer** one (`h = normalize(l + v)`), not the infinite-viewer one — scored:

| specular form | RMS(S) on the floor |
| --- | --- |
| Blinn, local viewer | **0.0187** |
| Blinn, infinite viewer | 0.0569 |
| Phong (`r·v`) | 0.0187 (degenerate with Blinn here; `p ≈ 15` instead of 60) |

Applying the fit per pixel:

| surface | exact | ≤ 1/255 | mean err |
| --- | --- | --- | --- |
| wall side faces | **100.00%** | 100% | 0.000 |
| floor | 80.7 – 96.4% | 99.95 – 100% | 0.04 – 0.23 |
| wall tops | 22.7 – 25.8% | 28% | 6.3 |
| agent (curved) | 8.9 – 50% | 60% | 3.3 |

Wall side faces come out as a single colour, **(15, 15, 15)**, everywhere: `n·l = 0`, so they are pure
ambient, `round(255 · 0.6 · 0.098) = 15`, zero variance across thousands of pixels.

---

## 3. Is the scene reducible? (pointmaze in particular)

**Not to 2-D — but to a trivial 3-D ray-cast.** The camera is perspective, so a top-down 2-D rasterizer
is structurally wrong. The right reduction is different and better: **1 plane + N boxes + 1 sphere, all
axis-aligned, under a fixed pinhole**.

An independent ray-caster (`scratch/ogbench/raycast.py`, ~120 lines of numpy: slab test, plane test,
sphere/capsule/cylinder quadratics) was scored against MuJoCo's own segmentation buffer, per pixel, over
5 states per scene:

| scene | geom-id agreement (all px) | boundary px | **interior agreement** | interior mismatches | depth max err (interior) |
| --- | --- | --- | --- | --- | --- |
| pointmaze-medium | 0.99968 | 7.8% | **1.000000** | **0** | 0.0118 m |
| pointmaze-large | **1.000000** | 13.0% | **1.000000** | **0** | 0.0077 m |
| pointmaze-giant | 0.99940 | 17.2% | **1.000000** | **0** | 0.0058 m |
| pointmaze-teleport | 0.99990 | 13.6% | **1.000000** | **0** | 0.0096 m |
| antmaze-large | **1.000000** | 13.0% | **1.000000** | **0** | 0.0013 m |
| humanoidmaze-large | **1.000000** | 13.0% | **1.000000** | **0** | 0.0016 m |

Depth agreement (0.001–0.012 m against `zfar = 1700–3300` m) also confirms the frustum convention:
MuJoCo's depth buffer is **perpendicular** distance (`t · (d·forward)`), not radial.

The visible-3-D content of a pointmaze frame is small but real. Composition, measured:

| scene | floor | wall tops | **wall sides** | agent | skybox |
| --- | --- | --- | --- | --- | --- |
| pointmaze-medium | 63.5% | 28.3% | **7.9%** | 0.29% | 0% |
| pointmaze-large | 40.6% | 43.5% | **6.0%** | 0.10% | 9.8% |

The wall side faces are 6–8% of every frame and exist only because the projection is perspective; an
orthographic 2-D rasterizer would be wrong on ~15,000 pixels per 224² frame. A frame has only **199–529
distinct colours**, and the side faces are a single constant (15, 15, 15). It is a "2.5-D" picture — but
the 2.5-D-ness must be drawn, not assumed away.

---

## 4. Articulated bodies (antmaze / humanoidmaze)

`nmesh = 0` in **every** scene. No mesh assets, no hfields, no skins. Full inventory:

| scene | geoms | plane | box (walls) | cylinder | sphere | capsule |
| --- | --- | --- | --- | --- | --- | --- |
| point-medium / large / giant | 40 / 64 / 108 | 1 | 38 / 62 / 106 | — | 1 | — |
| point-teleport | 70 | 1 | 63 | 5 | 1 | — |
| ant-medium / large / giant | 52 / 76 / 120 | 1 | 38 / 62 / 106 | — | 1 | 12 |
| ant-teleport | 82 | 1 | 63 | 5 | 1 | 12 |
| humanoid-medium / large / giant | 58 / 82 / 126 | 1 | 38 / 62 / 106 | — | 3 | 16 |

So the whole "articulated body" problem is **13 primitives for the ant** (1 sphere torso + 12 capsules)
and **19 for the humanoid** (3 spheres + 16 capsules). What they additionally require over pointmaze is
**forward kinematics**: `qpos → (geom_xpos, geom_xmat)`. That is cheap and pure:

| scene | `mj_kinematics` | MuJoCo full render | ratio |
| --- | --- | --- | --- |
| pointmaze-large | 0.93 µs | 2.17 ms | 2330× |
| antmaze-large | 1.73 µs | 2.19 ms | 1264× |
| humanoidmaze-large | 2.62 µs | 2.16 ms | 825× |

FK is a chain of quaternion composes over 9 (ant) / 13 (humanoid) bodies and batches trivially on GPU.

The sting is elsewhere: at the whole-maze top-down camera the **agent occupies 11–24 pixels of 50176**
(0.02–0.05%). This is the same fact recorded in `project_noproprio_camera_blocker`. It makes exactness
easy (there is almost nothing to get wrong) and makes the render almost pointless for the legged bodies
at this camera — which is a *dataset* problem, not a renderer problem.

### The official `visual-*` envs use a different camera

`ogbench/locomaze/__init__.py` registers `visual-antmaze-*` / `visual-humanoidmaze-*` with
`camera_name='back'`, 64×64 — a `trackcom` camera riding the torso, not the top-down view. (There is no
`visual-pointmaze-*` at all.) Re-running the geometry check there:

| env | camera | interior geom-id agreement | interior mismatches |
| --- | --- | --- | --- |
| antmaze-large | `back`, 64px | **1.000000** (6 states) | **0** |
| humanoidmaze-large | `back`, 64px | **1.000000** (6 states) | **0** |

The approach is camera-agnostic; the tracking camera pose is `f(qpos)` too. The extra work for that
camera is the **skybox** (a cube-map lookup — the fixed top-down camera lets you bake it instead).

---

## 5. The recommended renderer, and what it actually scores

Two facts make the design obvious:

1. the camera is **fixed per env** and the maze **never moves**, so floor + walls + teleport pads +
   skybox are **one constant image and one constant depth buffer**;
2. only the agent's 1–19 primitives move.

So: **bake the static plate once (from MuJoCo itself, with the agent's alpha set to 0), then per frame
ray-cast only the agent primitives and composite by depth.** Baking inherits MuJoCo's per-vertex shading
of the walls exactly, which is precisely the part a per-pixel model gets wrong.

Scored against **MuJoCo at its default `offsamples=4`** (what OGBench datasets actually contain), 12
random states each (`scratch/ogbench/baked_msaa.py`):

| scene | bit-exact | ≤ 1/255 | ≤ 8/255 | mean err |
| --- | --- | --- | --- | --- |
| pointmaze-medium | **99.64%** | 99.69% | 99.86% | 0.068 |
| pointmaze-large | **99.87%** | 99.88% | 99.94% | 0.026 |
| antmaze-large | **99.92%** | 99.92% | 99.92% | 0.038 |
| humanoidmaze-large | **99.93%** | 99.94% | 99.94% | 0.031 |

and against `offsamples=0` (no MSAA), 16 states each (`scratch/ogbench/general_check.py`):

| scene | bit-exact | mean err | agent px | non-boundary exact | non-boundary bad px |
| --- | --- | --- | --- | --- | --- |
| pointmaze-medium | 99.67% | 0.055 | 147 | 99.76% | 120 |
| pointmaze-large | 99.89% | 0.018 | 51 | 99.94% | 31 |
| pointmaze-giant | 99.95% | 0.006 | 24 | 99.98% | 10 |
| pointmaze-teleport | 99.88% | 0.020 | 51 | 99.94% | 32 |
| antmaze-large | 99.94% | 0.016 | 21 | 99.99% | 2.8 |
| antmaze-giant | 99.97% | 0.009 | 11 | **100.00%** | 0.8 |
| humanoidmaze-large | 99.96% | 0.008 | 24 | 99.99% | 5.3 |

**Every mismatched pixel is inside the agent's silhouette.** For `antmaze-giant` the interior is exact.

### Throughput (H200, 224×224)

| scene | prims | full-frame | **tiled** | MuJoCo (same GPU) | speed-up |
| --- | --- | --- | --- | --- | --- |
| pointmaze-large | 1 | 94,184 f/s | **466,687 f/s** | 460 f/s | **1014×** |
| pointmaze-giant | 1 | 94,199 f/s | **467,322 f/s** | 428 f/s | **1092×** |
| antmaze-large | 13 | 4,103 f/s | **70,842 f/s** | 455 f/s | **156×** |
| humanoidmaze-large | 19 | 2,943 f/s | **52,275 f/s** | 462 f/s | **113×** |

"tiled" restricts the primitive intersections to a 48×48 screen-space tile around the agent (the agent
covers ≤150 of 50176 pixels, so the full-frame cast wastes ~99.9% of its rays). The tiled path was
verified to produce a **bit-identical** frame to the full-frame path before being timed.

For calibration: PushT's GPU renderer hits 14.1 M f/s, but a PushT frame is 5 SDF tests over the whole
canvas; here most of the canvas is a memcpy of the baked plate. At 466k f/s the pointmaze renderer is
**bandwidth-bound on writing the output** (466k × 150 kB = 70 GB/s), not compute-bound — if the consumer
can take "plate + patch" instead of a materialized frame, the ceiling is much higher.

### Cross-node determinism

MuJoCo's EGL raster is reproducible across machines: the SHA-256 of 12 frames from a fixed state sequence
is **identical on `gb-west-a1-gpu-116` and `gb-west-a1-gpu-058`**, and MuJoCo-vs-MuJoCo on a re-render is
100% exact. So "bit-exact" is a well-posed target here. (The known-bad render nodes 047/070/120/184 were
excluded; per `reference_faulty_render_nodes` they are not trustworthy for this.)

---

## Closing the shading gap: what was wrong, and what is left

The first pass explained the agent residual as Gouraud shading over MuJoCo's tessellation, on the
strength of a bilinear-per-face fit of wall tops (0.93–1.21/255 residual) against a per-pixel Blinn model
(12–13/255). **That diagnosis was wrong, and the evidence for it was an artifact of a wrong composition.**

### The composition, not the constants

The shading is not a multiplier. It is fixed-function OpenGL:

```
primary = mat_rgb * (A + Σ_l D_l · max(0, n·l_l))  +  Σ_l S_l · spec · max(0, n·h_l)^p     # white, ADDED
final   = primary * tex_rgb                                                                # tex = 1 if untextured
```

Both behaviours are visible in the data, and telling them apart is what fixed it: the channel spread of
`obs / base` is **3e-4 on the floor** and **6.8e-2 on the agent**. The floor's material colour is
(1,1,1) and all its colour comes from a texture, so there the added specular reads as a pure multiplier;
the agent and walls are untextured, so it is genuinely additive. Fitting a multiplicative law on the
floor and extrapolating it is what overshot wall tops by 6.5/255, and it is what a bilinear per-face fit
was quietly absorbing.

With the composition corrected:

| surface | before (multiplicative) | after (additive, one per-pixel law) |
| --- | --- | --- |
| vertical wall faces | mean 99/255 | **100.00% bit-exact** |
| wall tops | worst 13/255, 23% exact | worst **2/255**, 98–100% within 2 |
| floor | ≤1/255 on 99.9% | ≤1/255 on 99.9% |

### Gouraud: tested and refuted

With the composition fixed, the tessellation hypothesis was tested directly — build the mesh, shade the
vertices, interpolate barycentrically over the hit triangle — across ten variants (`numslices` 28,
`numstacks` 16 and 8, phase offset 0 and π/nslice, both latitude directions, both quad diagonals):

| model | agent-pixel exact |
| --- | --- |
| plain per-pixel | **1.6%** |
| best Gouraud variant | 0.2% |
| worst Gouraud variant | 0.0% |

Every variant is worse than not interpolating at all. This is also what the geometry predicts: a 0.7-radius
sphere is ~13 px across here and carries ~224 front-facing quads, so the triangles are sub-pixel and
Gouraud and per-pixel coincide by construction. **MuJoCo is not interpolating vertex shades here.**

### The lobe: also refuted

A 24-knot piecewise-linear lobe was fitted nonparametrically under four light/specular geometries —
headlight directional (what the light table says) or a point light at the camera, crossed with Blinn
`n·h` and Phong `r·v`. The best of the four reaches 81% of agent pixels within 1/255 and none explains
floor and agent with one curve. A basis that expressive failing is the evidence that the residual is not
a function of the specular argument at all.

### Where it actually is

Calibrating the (already-correct-in-form) law **on agent pixels**, which is the only place it is used
once the plate is baked, over 13,069 pooled agent-interior pixels across the four pointmaze scenes:

    A = 0.078438   D = 1.416189   Sp = 0.230554   p = 45

| reference | agent-interior within 1/255 | mean |
| --- | --- | --- |
| MuJoCo, **no** multisampling | **92.4%** | 0.88/255 |
| MuJoCo, default 4× MSAA (what datasets hold), our ss=2 | 47% (medium) – 81% (large) | 1.3 – 2.5/255 |

The 92.4% → 47-81% drop when multisampling is switched on is the remaining gap, and it is a **coverage**
effect, not a shading one: MSAA sample positions are GPU-specific and are not recoverable by fitting.
Supersampling on our side recovers most but not all of it (ss=1 → ss=2 takes pointmaze-large from 24% to
81%; ss=4 adds 3 points for 4× the cost, so ss=2 is the default).

For calibration, fitting the same law on **static** pixels instead gives (0.098, 1.384, 0.260, 52) and
only 35% of agent pixels within 1/255 — the floor is the one surface the shader never has to serve
(`n·l ≡ 1`, `n·h` within a few degrees of 1), so an exponent fitted there is pure extrapolation onto a
sphere.

### The legged families have almost no agent to check

At the official whole-maze camera the ant is ~7 px and the humanoid ~8 px of 50176; eroding by one pixel
leaves **23 and 1** interior pixels over 32 frames. Their agent agreement is a statement about a
silhouette under someone else's coverage rule, not about a renderer. That is a property of OGBench's
camera, and it is the same fact that makes proprioception unreadable from these observations.

---

## Packaged: `stable_envs.envs.ogbench`, 25 registered cells

`stable_envs/render/raycast3d.py` (camera, primitive intersectors, shading, tiled compositing) and
`stable_envs/envs/ogbench.py` (25 env classes). Parity for **all 25 cells over 64 states each**, at the
shipped defaults (224 px, supersample 2, tile 32):

| family | cells | interior-exact | whole-frame exact | mean | non-agent exact | agent px/frame | agent-interior ≤1/255 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pointmaze | 8 | **1.000000** | 99.69 – 99.94% | 0.014 – 0.072 | 99.94 – 99.98% | 24 – 138 | 0.48 – 0.84 |
| antmaze | 11 | **1.000000** | 99.74 – 99.95% | 0.019 – 0.138 | 99.87 – 99.97% | 12 – 70 | 0.00 – 0.51 |
| humanoidmaze | 6 | **1.000000** | 99.92 – 99.98% | 0.007 – 0.032 | 99.97 – 99.99% | 3.5 – 21 | 0.00 – 0.03 |

`parity_interior_min` is therefore **not relaxed** — the measured interior agreement is exactly 1.0 on
every cell, so a tolerance would declare a limitation the evidence does not show. The agent-pixel
residual is real and is carried in `parity_note` instead, because it does not reach that metric: the
interior mask keeps only pixels whose 3×3 neighbourhood in the reference is one flat colour, and a small
shaded sphere has almost none.

Throughput, end to end (`qpos` → frame, including forward kinematics), H200, batch 1024, 224 px:

| family | frames/s | vs MuJoCo (460 f/s) |
| --- | --- | --- |
| pointmaze | 60,600 | 132× |
| antmaze | 4,980 | 11× |
| humanoidmaze | 3,780 | 8× |

The legged families pay for 13 and 19 primitives against pointmaze's 1, and for forward kinematics,
which is still a CPU loop over the batch (`mj_kinematics`, 1.7–2.6 µs per state). pointmaze's FK is
closed form and stays on the GPU — and is checked against `mj_kinematics` at construction rather than
assumed.

### Style is per environment, and the batch is grouped, not flattened

The style vector splits in two, and the split is what keeps batched appearance randomisation real:

| leaf | applied | cost |
| --- | --- | --- |
| `agent.color`, `agent.scale` | per environment inside the ray-cast (`mat_rgb (B,P,3)`, `size_scale (B,)`) | free — never splits the batch |
| `floor.tint`, `walls.color`, `camera.distance_scale`, `camera.elevation`, `rendering.decolor_floor` | changes the static scene, so one baked plate per distinct value | one bake per distinct plate, cached by content |

`render` groups the batch by the plate half of the style and renders one call per group. The common case
— randomise the agent, keep the maze — is a single group. Taking row 0 and applying it to everything
would have passed every conformance check (they compare an env against itself) and silently handed a
caller doing 4096-way domain randomisation one appearance repeated 4096 times.

The colour knobs are float `Box` in [0,1], not `RGBBox`. Two reasons, and the second is the honest one:
MuJoCo's material colours are floats (0.7, 0.5, 0.3) that uint8 cannot hold — 179/255 shifts the agent a
level — and in a lit 3-D scene **no material colour appears verbatim in any pixel**, because the pixel is
the material colour times the lighting. `test_color_knobs_apply_the_requested_color` only checks leaves
declared 0–255, so it skips these; a knob here can only be checked by "does the requested colour reach
the shader", which is what the variation and style tests do.

### One bug worth recording

An early version pooled reference envs and closed the oldest to bound EGL contexts. Releasing an EGL
context **corrupted rendering in other live envs**: two fresh reference envs driven to the same `qpos`
came back 44–107/255 apart, which reads exactly like *the state does not determine the frame* — the one
conclusion this repo must never reach by accident. The tell was that the first two env names passed and
every one after them failed, i.e. failures began the moment the pool started evicting.

The competing hypothesis — that the state-restore path wrote `qpos` without a forward pass, so the
renderer drew a stale `data.xpos` (the PushT `reindex_shapes_for_body` failure mode) — was tested
directly and **refuted on every check** (`scratch/ogbench/two_instances.py`):

| check | result |
| --- | --- |
| `MujocoEnv.set_state` source contains `mj_forward` | yes |
| `geom_xpos` after `set_state` vs a fresh `mj_kinematics` | **0.0** |
| render A, B, A — frame 3 vs frame 1 | **0/255** |
| same `qpos` after `set_state(B)` vs after 7 random sim steps | **0/255** |
| two fresh env instances, same `qpos` | **0/255** |

Reference envs are now never pooled or closed — **and neither is any other renderer**. The plate bake
used to build a `mujoco.Renderer` per bake and close it; with 7 style leaves across 25 names that was
~175 context releases, and it reproduced the same corruption (one cell's full test set failed even
though the sufficiency test alone passed for all 25). Renderers are now cached one per *texture state*
per scene — material colours, geom colours and the camera are read per frame and need no re-upload, so
only `decolor_floor` needs a second one — which bounds them at 22 for the whole package.

The rule this leaves behind: **in this process an EGL context is created and never released.** Anything
that wants to bound them has to bound how many are *created*, not close the ones that exist.

---

## Biggest risk

**Not the geometry, not the shading — the fact that the top-down camera is a fixed plate with a 20-pixel
agent on it.** The renderer's job reduces to "draw a small disc/blob in the right place at the right
colour", which is why it hits 99.9% bit-exact so easily; but it also means the exactness statistic is
dominated by pixels no model will ever learn from. The number to defend is not "99.9% of the frame" but
"**the agent's ~20–150 pixels are right**", and there the honest figure is: silhouette exact (0 interior
geom-id mismatches, every scene), interior shading within ~3/255 mean and 48/255 worst until the Gouraud
tessellation of spheres/capsules is replicated.

The runner-up risk: this study calibrated against **rjepa's** render config (224px, `decolor_floor`,
humanoid marker 4.0, top-down camera). The official `visual-*` OGBench datasets use `camera_name='back'`
at 64px **with the position-gradient floor texture intact**. The geometry check passes there too (100%
interior, 0 mismatches), but that configuration additionally needs a live floor-texture lookup and a
skybox cube-map — neither of which the fixed-camera bake can absorb.
