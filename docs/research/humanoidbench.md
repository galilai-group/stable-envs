# HumanoidBench (Unitree H1 / H1-with-hands)

**Verdict: the parity tier is out and one number decides it; 17 of 49 scenes are MEASURED at the
visual tier and are BUILT but do NOT register by default, because `mujoco_warp`'s batched render is
not bit-reproducible inside a long conformance session. The other 32 are an evidenced NO-GO with
three named causes, all three upstream properties of `mujoco_warp`.**

The fidelity is finished work: mean 2.6-8.4/255, within-8 0.795-0.958, SSIM 0.845-0.931,
interior-exact 0.607-0.943, at **437x** HumanoidBench's own renderer at batch 64 — every column
strictly better than [RoboCasa](robocasa.md), the visual tier's only current member. What is not
finished is §5.2: over six full runs of `pytest tests/ -k HumanoidBench`, 2-3 of ~660 tests fail every
time, always `test_determinism` or `test_frame_i_is_independent_of_batch_size`, by 1-2/255, on a
different cell each run — and the registry's contract includes "the same state gives the same frame".
So `stable_envs/envs/humanoidbench.py` raises at import unless `STABLE_ENVS_HUMANOIDBENCH=1`, which is
the same mechanism an optional dependency uses and keeps the repo's suite green.

The deciding number for parity is **the static plate does not exist**: every HumanoidBench camera but
one is `mode="trackcom"`, so hide every mover, render eight states, and the "static" picture still
differs by **54/255 over 21.7 % of its pixels**. That removes the design [Fetch](fetch.md),
[Meta-World](metaworld.md) and [OGBench](ogbench.md) all reach parity with — bake the static majority
through MuJoCo itself, ray-cast only what moves — and leaves full-scene MJWarp, which then has to draw
the floor. The floor is **66–72 % of a locomotion frame** and carries **99.7 % of the interior error**,
because it is a minification-filtered texture: the same mechanism, and the same backend gap, that made
the six [Gymnasium locomotion domains](mujoco_locomotion.md) a NO-GO.

What is different here, and why this family lands rather than fails: HumanoidBench's floor texture is
**degenerate**. It is a `checker` whose two colours are *the same* (`rgb1 = rgb2 = 0.9 0.9 0.9`) with a
one-texel `mark="edge"` at 0.7 — a 2 cm grid line every 2 m on an otherwise flat grey — where
Gymnasium's is a full-contrast checker at `texrepeat="60 60"`. So the residual is a thin line rather
than the whole floor, and it is 0.877 interior-exact rather than 0.22.

All numbers: one NVIDIA H200 via SLURM, `MUJOCO_GL=egl`, `humanoid_bench` 0.2 (`--no-deps`, commit
`main` 2026-08), `mujoco` 3.11.0 / `mujoco-warp` 3.11.0 / `warp` 1.16.0 / `gymnasium` 1.2.3, in
`~/venvs/se-hb`, at HumanoidBench's own 256 px. Probes: `scratch/humanoidbench/probe*.py`. Raw JSON:
`/mnt/vast/home/randall.balestriero/stable_envs_logs/humanoidbench/`.

---

## 1. The four numbers, before anything was built

`docs/research/fetch.md` and `metaworld.md` both turn on four counts. Measured over all 49 compiling
scenes in `probe1_facts.py`, before a renderer existed:

| | h1 locomotion | h1hand locomotion | h1hand manipulation | Franka Kitchen (for contrast) |
|---|---:|---:|---:|---:|
| `nmesh` | 21 | 47 | 47–50 | — |
| `ntex` | **2** (1 skybox + 1 2-D) | **2** | 3–7 | 14, twelve of them `mjTEXTURE_CUBE` |
| casting lights | **1** | **1** | **1** | 4 |
| textured **primitives** in groups 0–2 | **0** | **0** | 1–25 | — |

Three of the four are the best in any MuJoCo family this repo has measured. **Not one scene carries a
`mjTEXTURE_CUBE`**, so the CUDA-700 fault that had to be worked around on Kitchen and Meta-World never
arises. **One casting light**, like Meta-World and unlike Kitchen's four. And the visible scene of a
locomotion cell is *one PLANE and 65 MESHes* — zero primitives — which is exactly the set
`mujoco_warp`'s `sample_texture` can UV, so the Adroit/RoboCasa texture-UV gap does not apply to the
floor at all.

The skybox is a `gradient` whose two colours are also equal, i.e. a flat 229 grey. It duly comes out
**1.000000 of interior pixels exact** — the one part of the frame that is free.

## 2. The gates

### G1 · purity — passes, 0/255 on four independent probes

`probe2_gates.py`, `h1hand-walk`, 12 states from the env's own distribution, 256 px, one env per
process:

| probe | max abs diff |
|---|---:|
| render the same `qpos` twice | **0** |
| `qvel` drawn from U[−100, 100], 100 draws | **0** |
| the same `qpos` reached through a different rollout | **0** |
| a FRESH `MjData`, no carried derived state at all | **0** |
| the frame straight out of `reset()` vs one `mj_forward` later | **0** |

That last row is the [Meta-World](metaworld.md) staleness rule, checked rather than assumed: 28 of
Meta-World's 50 tasks render a stale frame straight out of `reset`, and HumanoidBench renders none —
`reset_model` ends in `set_state`, which forwards.

`mujoco.Renderer` at the task's model camera is **bit-identical to gymnasium's own frame** (exact
1.000000, max 0/255, 173 unique colours so the comparison is not against a blank surface). That is what
makes every later number a comparison against the official picture rather than against a proxy.

### G2 · compression — passes, 0.001–0.003×

`nq` float64 against a 256×256×3 frame: **208 B / 196,608 B = 0.00106×** for `h1` (nq 26),
**608 B = 0.0031×** for `h1hand` (nq 76). The observation is a different and larger object; the render
state is `qpos` and nothing else.

### G3 · shared decoder — passes outright, which is the surprise

This is the gate that killed RoboCasa (`nq` 91/103/131/133 across four episodes, `nmesh` 189–408) and
nearly killed Meta-World (per-episode placement written into `model.body_pos` after the last forward).
Measured the same way — every ndarray attribute of `MjModel`, compared across a reset and across five
steps:

> **model arrays changed by a reset: none. By five steps: none.** On every scene shipped here.

HumanoidBench's per-episode randomisation is `qpos` jitter (`randomness = 0.01`) plus, on some tasks, a
`qpos` write. The decoder is a constant of the family, and it is one 21–50-mesh MJCF rather than a
24 GB library.

Four tasks *do* write the model and they are not shipped: `bookshelf` (`geom_rgba`, `body_pos`), `cube`
(`body_quat`), `package` (`body_pos`), and `maze`, which writes `geom_rgba` **inside `get_reward`** —
a progress indicator that is a function of the trajectory rather than of `qpos`, i.e. a G1 failure
specific to that task.

### G4 · economics — passes emphatically

`probe8_haze.py`, `h1hand-walk`, 256 px, CUDA-graph replay:

| batch | MJWarp f/s | × the official renderer | cores of JPEG decode |
|---:|---:|---:|---:|
| 1 | 2,063 | 53.6× | 0.38 |
| 4 | 7,178 | 186× | 1.31 |
| 16 | 13,127 | 341× | 2.39 |
| 64 | 17,707 | 460× | 3.23 |
| **256** | **19,719** | **512×** | **3.59** |
| 1024 | 20,295 | 527× | 3.70 |

The headline is therefore **(512×, batch 256)**, and the crossovers are **batch 1 against the official
renderer** and **batch 4 against one CPU core of decode** (5,486 f/s at 256 px, from
`when_does_this_pay_off.md`). Un-captured, the same four kernels run at **422 f/s — 1.4×, not 512×**;
the CUDA graph is the whole port, exactly as on Meta-World.

The official renderer is unusually slow here and that is a property of the scene, not of the machine:
**38.5 f/s** at 256 px, because `common/visual.xml` sets `shadowsize="16384"` (a 268-megatexel shadow
map) for a humanoid with 156 geoms. Restoring a stored state first (`mj_forward` on a 76-DoF humanoid
with `iterations=100`) costs almost nothing on top: 37.8 f/s end to end.

### G5 · fixed cast — passes trivially

Nothing spawns. The cast is the robot plus a fixed prop set per scene.

## 3. What decides the tier: the floor

No plate exists (§0), so the whole frame goes through MJWarp. Measured on `h1hand-walk`, 16 states,
256 px, with every correction this repo knows about applied — `gl_rounding()`, `prepare_model` for
geoms and sites, BGRA→RGB, one material per geom carrying MuJoCo's resolved colour, shadows and skybox
on, groups 0–2 on both sides, `geom_margin`/`pair_margin` zeroed for `put_model`, and `com_pos` in the
frame loop:

| class | share of frame | interior-exact | interior mean | share of interior error |
|---|---:|---:|---:|---:|
| background (flat skybox) | 0.256 | **1.000000** | 0.00 | 0.000 |
| **floor** | **0.701** | **0.80973** | 2.62 | **0.99695** |
| bodies (the H1 meshes) | 0.043 | 0.25816 | 1.51 | 0.00305 |
| **whole frame** | | **0.8772** | 1.66 | |

The mesh row is the same Gouraud-vs-per-pixel residual [Fetch](fetch.md) declares and it is worth 0.3 %
of the error. Everything else is the floor.

### 3.1 The ladder: it is the floor's texture, and only that

Every rung modifies **both** renderers identically (`probe3_mjwarp.py`). Interior-exact, whole frame /
floor only:

| rung | whole frame | floor |
|---|---:|---:|
| L0 the official scene | 0.8772 | 0.8097 |
| L1 shadows off, both sides | 0.3652 | 0.0008 |
| L2 the floor's texels ← their own mean | 0.9770 | 0.9709 |
| **L3 every `mat_texid = −1`** | **0.9950** | **0.9957** |
| L4 L3 + specular 0 | 0.9951 | — |

Read down the column. **Remove the texture and this pipeline is essentially exact** (0.995 interior),
which is the flat-material parity [backend3d.md](backend3d.md) reports and which the mesh, the skybox,
the shadow and the specular terms are all inside. L2 is the interesting rung and it says two things at
once: a *constant* texture still costs 0.025 of the floor, so the texture-modulate path itself carries
a level that `gl_rounding()` does not cover (the same finding [mujoco_locomotion.md](mujoco_locomotion.md)
§6.5 records); and the remaining 0.16 between L2 and L0 is the grid line.

L1 is worth recording because it runs the wrong way: with shadows off on **both** sides the floor falls
to 0.0008 interior-exact. MuJoCo's shadowed and unshadowed floor differ by a global term that MJWarp
reproduces only in the shadowed configuration — which is the default, and the one used everywhere here.

### 3.2 The anatomy of the floor residual

`probe6_floor.py`, `h1hand-walk`, 8 states, floor interior pixels only:

| | share of floor interior px | exact | mean |
|---|---:|---:|---:|
| grid-line pixels (darker than the local median) | 0.0068 | **0.000000** | 55.7 |
| flat-field pixels | 0.9932 | 0.8495 | 2.21 |

and the error histogram is bimodal — 184,239 pixels exact, 20,453 off by 1–2, a flat tail of ~150 per
level, then 5,306 at 54–57. The 54–57 bucket is the line drawn where MuJoCo does not draw it (and vice
versa); the 1–2 bucket is L2's texture-path level. The signed mean is **−1.72**, i.e. MJWarp's floor is
systematically the darker of the two.

By image row (a distance proxy), floor exactness runs 0.053 at the horizon band, 0.94 through the
middle, 0.63 in the nearest band. The horizon band's reference luminance spread is **0.46** against
MJWarp's **9.03** — MuJoCo has filtered the grid away out there and MJWarp has not.

Two candidate mechanisms were checked and are *not* the cause. **Fog** is off (`mjRND_FOG = 0`,
despite `fogend="10"` in the MJCF); enabling it would move 30 % of pixels, so it matters that it is
off, and it is. **Haze** is on (`mjRND_HAZE = 1`) and `mujoco_warp` has no haze pass, but it changes
1.2 % of pixels, max 65/255, and **0.0000 of interior pixels** — a footnote.

### 3.3 Supersampling is refuted, and refuting it is what names the mechanism

`probe4_supersample.py`, `h1hand-walk`, 16 states, box average down:

| | ss 1 | ss 2 | ss 3 | ss 4 |
|---|---:|---:|---:|---:|
| interior exact | **0.8772** | 0.7991 | 0.7285 | 0.6823 |
| mean /255 | 4.29 | 3.73 | 3.63 | **3.50** |
| SSIM | 0.639 | 0.764 | 0.800 | **0.814** |
| far-band exact | 0.0867 | 0.0795 | 0.0903 | 0.1184 |

Exactness *falls* while the mean falls too. That is the signature of a **minification** residual and
not a coverage one: sixteen samples per pixel converge toward the mip average (the mean improves) while
blurring every pixel that was already bit-exact. It is the same result `mujoco_locomotion.md` §6.5
reports, and it is why the default here is `supersample = 1`.

### 3.4 The choice this module makes, and its measurement

Both available renderings of that floor are approximations. Measured against the same official frame
(`probe7_variants.py`, `h1-walk`, 16 states):

| our floor | interior | mean /255 | within 8 | SSIM |
|---|---:|---:|---:|---:|
| textured, as-is (dithered) | 0.9158 | 4.13 | 0.901 | 0.628 |
| **its dominant texel (flat)** | **0.9503** | **2.79** | **0.935** | **0.851** |

The flat floor is closer on **all four** numbers at once, so that is what
`stable_envs/envs/humanoidbench.py` renders, and the grid is declared missing in `visual_note` rather
than absorbed. It is expressed by pointing the floor material away from its texture and putting the
dominant texel in `mat_rgba`, not by rewriting texels, for two measured reasons: the no-texture path is
closer than a constant-texture one (L3 0.9957 against L2 0.9709), and `mat_rgba` is a per-world array,
so `floor.color` can be a real per-environment knob where a texture could not be.

## 4. What keeps 30 of the 49 scenes out

`probe5_all_tasks.py` renders all 49 through the same pipeline. Two causes, both upstream gaps in
`mujoco_warp`, and they are the same two that produced the [Adroit](adroit.md) NO-GO.

### 4.1 Alpha — the largest single failure in the family

`humanoid_bench/assets/locomotion/generated_xml_hurdles.xml` gives every barrier
`rgba="0.1 0.1 0.1 0.1"` — **black at alpha 0.1**. MuJoCo blends them to a faint tint over the floor;
`mujoco_warp`'s shader reads `color[0..2]` and packs a constant 255, so it paints them **opaque black**
over most of the frame.

| scene | translucent geoms | interior | mean /255 | SSIM | SSIM if they are HIDDEN instead |
|---|---:|---:|---:|---:|---:|
| `hurdle` | 3 | 0.381 | **111.2** | 0.270 | 0.385 (mean 33.2) |
| `maze` | 30 | 0.533 | 24.0 | 0.665 | **0.344** |
| `pole` | 3 | 0.491 | 22.5 | 0.715 | 0.781 |
| `door` | 6 | 0.450 | 18.3 | 0.570 | 0.500 |
| `room` | 4 | 0.472 | 15.9 | 0.699 | 0.621 |
| `window` | 6 | 0.640 | 38.0 | 0.648 | 0.774 |
| `crawl`, `package` | 3 | 0.87–0.91 | 4.3–5.2 | 0.63–0.70 | unchanged |

The Fetch rescue — hide the translucent geometry on our side and declare the cost — is measured in the
last column and it does not generalise: it rescues `hurdle`'s mean (111 → 33) without rescuing its
picture, and on `maze`, where 30 translucent walls *are* the task, it takes SSIM from 0.665 to 0.344.
Fetch could declare it because its translucent geometry was 1.2–1.5 % of all pixels and 0.000002 of
interior pixels; here it is up to 62.8 % of the frame.

This is the gap that killed Adroit's `Pen`, unchanged and still upstream.

### 4.2 Textured primitives

`room` (25), `cabinet` (21), `truck` (19), `spoon` (15), `window` (13) and `bookshelf` (10) wear
textures on `BOX`es and `CYLINDER`s, which `sample_texture` cannot UV — the RoboCasa mechanism. It is a
ceiling rather than a wall: `cabinet` still reaches SSIM 0.937 and `spoon` 0.898 with the flat floor,
because the affected props are a small share of those frames. `truck` (SSIM 0.838) and `room` (0.793)
do not clear the bar and are not shipped.

### 4.3 Two more exclusions, neither about the renderer

**Dead dimensions.** The state here is exactly `qpos`, so every entry must reach a pixel, and
`probe10_deaddims.py` says 9 of 28 candidates carry entries that do not: `basketball` 4 (the ball is a
featureless SPHERE, so its free joint's quaternion is invisible *by construction*), `balance_hard` 4,
`insert_normal`/`insert_small` 5, `cabinet` 26, `bookshelf_*` 61 — drawer and book hinges behind the
camera or inside a closed cabinet. Declaring them would declare dimensions the renderer provably
ignores and `check_state_dims_matter` would correctly fail. RoboCasa solves the same problem by
measuring and pinning a live subset; that is deliberately not done here, because it makes the state
stop being `qpos`. **`h1hand`'s 50 finger joints are NOT in that list** — all 76 entries of an
`h1hand` locomotion cell reach a pixel at 256 px.

**`reach` and `push` cannot render their official frame on this stack at all.** Both draw their goal
with `viewer.add_marker`, and gymnasium's legacy marker path calls `mjv_initGeom` with a scalar
`size=0.05` and a tuple `rgba=`, which `mujoco` 3.11 rejects with a `TypeError`. Their goal is also not
in the model — it is an `mjvScene` decoration — so a port would have to re-express it as a geom the way
Fetch does with `target0`, on top of fixing the reference.

**`kitchen` does not compile on `mujoco` 3.11**: `XML Error: top-level default class 'main' cannot be
renamed`. It is also the only scene in the family with a world-body (non-TRACKCOM) camera, i.e. the
only one a static plate could have been built for.

## 5. What ships

17 cells, all with zero dead dimensions, in `stable_envs/envs/humanoidbench.py`:

`h1`: `walk`, `run`, `stand`, `stair`, `slide`, `sit_simple`, `sit_hard`, `balance_simple`.
`h1hand`: `walk`, `run`, `stand`, `stair`, `slide`, `sit_simple`, `sit_hard`, `balance_simple`,
`spoon`.

`h1-highbar_simple` and `h1hand-powerlift` were measured (their rows are below) and are **not
registered**: they are the two cells the shared-BVH effect of §5.1 reaches, and an env that cannot
promise the same pixels for the same state does not ship.

Measured through the env class on its own `sample_states`, 64 states × 4 independent draws, 256 px
(`calibrate_tier.py`); the throughput column is the env's own surface — states in, uint8 frames out,
appearance resolution and channel swap included:

| class | `nq` | mean /255 | within 8 | SSIM | interior mean | interior exact | f/s b1 / b16 / b64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `HumanoidBenchH1Walk` | 26 | 2.86 | 0.934 | 0.849 | 0.34 | 0.932 | 814 / 8366 / 16269 |
| `HumanoidBenchH1Run` | 26 | 2.86 | 0.934 | 0.849 | 0.34 | 0.932 | 821 / 8478 / 16348 |
| `HumanoidBenchH1Stand` | 26 | 2.86 | 0.934 | 0.849 | 0.34 | 0.932 | 830 / 8494 / 16404 |
| `HumanoidBenchH1Stair` | 26 | 2.62 | 0.958 | 0.929 | 0.54 | 0.853 | 823 / 7275 / 11536 |
| `HumanoidBenchH1Slide` | 26 | 3.35 | 0.954 | 0.931 | 1.51 | **0.620** | 743 / 8186 / 14627 |
| `HumanoidBenchH1SitSimple` | 26 | 3.44 | 0.917 | 0.851 | 0.49 | 0.942 | 787 / 7296 / 11642 |
| `HumanoidBenchH1SitHard` | 33 | 3.46 | 0.919 | 0.868 | 0.56 | 0.937 | 900 / 7954 / 11943 |
| `HumanoidBenchH1BalanceSimple` | 33 | 5.27 | 0.910 | 0.862 | 1.77 | 0.931 | 950 / 7540 / 11265 |
| ~~`HumanoidBenchH1HighbarSimple`~~ (not shipped, §5.1) | 26 | 7.01 | 0.906 | 0.845 | 5.93 | 0.960 | 938 / 7378 / 11374 |
| `HumanoidBenchH1HandWalk` | 76 | 3.00 | 0.932 | 0.852 | 0.34 | 0.931 | 738 / 7676 / 14363 |
| `HumanoidBenchH1HandRun` | 76 | 3.00 | 0.932 | 0.852 | 0.34 | 0.931 | 659 / 7089 / 13818 |
| `HumanoidBenchH1HandStand` | 76 | 3.00 | 0.932 | 0.852 | 0.34 | 0.931 | 660 / 7141 / 13847 |
| `HumanoidBenchH1HandStair` | 76 | 2.76 | 0.956 | 0.931 | 0.56 | 0.845 | 630 / 5742 / 10089 |
| `HumanoidBenchH1HandSlide` | 76 | 3.69 | 0.946 | 0.906 | 1.84 | **0.607** | 621 / 6316 / 12599 |
| `HumanoidBenchH1HandSitSimple` | 76 | 3.61 | 0.915 | 0.859 | 0.50 | 0.937 | 649 / 5919 / 10301 |
| `HumanoidBenchH1HandSitHard` | 83 | 3.56 | 0.918 | 0.868 | 0.54 | 0.943 | 651 / 5664 / 10240 |
| `HumanoidBenchH1HandBalanceSimple` | 83 | 5.40 | 0.908 | 0.861 | 1.71 | 0.930 | 642 / 5801 / 9599 |
| ~~`HumanoidBenchH1HandPowerlift`~~ (not shipped, §5.1) | 83 | 5.93 | 0.874 | 0.878 | 0.39 | 0.931 | 728 / 6456 / 10419 |
| `HumanoidBenchH1HandSpoon` | 83 | 8.40 | 0.795 | 0.899 | 4.21 | 0.923 | 698 / 5980 / 9231 |

Mean absolute error is **2.6–8.4/255**, within-8 **0.80–0.96**, SSIM **0.845–0.931**, and interior
exactness **0.607–0.960** — every column strictly better than the one visual-tier family already in the
repo ([RoboCasa](robocasa.md): mean 5.8–32.1, within-8 0.37–0.84, SSIM 0.82–0.92, interior exact
0.05–0.16). Each class's own numbers and its own thresholds are in `visual_fidelity`, and the
thresholds are set clear of the **spread over the four draws**, not of one.

The `f/s` columns above were taken while the render context was still sized to the chunk; pinning the
world count at 64 (see §5's world-count trap) changes the small-batch end and nothing else. Re-measured
after that change (`bench_env_surface.py`):

| | batch 1 | 8 | 16 | 64 | 256 |
|---|---:|---:|---:|---:|---:|
| `HumanoidBenchH1Walk` | 278 | 2,191 | 4,317 | **16,839** | 16,971 |
| `HumanoidBenchH1HandWalk` | 247 | 1,944 | 3,845 | 14,480 | 14,688 |
| `HumanoidBenchH1HandSpoon` (heaviest) | 152 | 1,197 | 2,371 | 8,961 | 9,150 |

So the pair a caller gets at the env's own surface is **(437×, batch 64)** against HumanoidBench's own
38.5 f/s; it is already **7.2× at batch 1**, so the crossover against the official renderer is batch 1.
Against one CPU core of JPEG decode at 256 px (5,486 f/s) it crosses over at about **batch 20** and
reaches **3.1 cores** at batch 64. A batch of 1 pays 64 worlds' worth of launch for one frame, which is
the price of batch invariance and is gone by batch 64.

`scratch/humanoidbench/dump_sidebyside.py` writes an official / ours / 4×-difference strip through the
registered class, so the trade is judged rather than described:
`/mnt/vast/.../se_logs/humanoidbench/img_shipped/`. The floor's grid is the visible loss and the H1
and its shadow land on the reference; the `img/` strips from `probe5_all_tasks.py` are the *other*
choice, with the floor textured and dithered, and the difference between the two directories is §3.4.

### 5.1 A world in an MJWarp launch is not fully independent of the other worlds

The conformance suite found this and it took two fixes and two exclusions to settle, so it is recorded
in full rather than as a footnote. `mujoco_warp.refit_bvh` refits **one** acceleration structure over
every world in a launch, so the tree a ray traverses is a function of all of them — and on a scene with
two nearly coincident surfaces a near-tie can then be broken the other way, by 1–2/255.

| symptom | cause | what was done |
|---|---|---|
| a state rendered alone differed from the same state in a batch by 1/255 | the render context was sized to the chunk, and MJWarp's megakernel built for `nworld = 1` does not produce the same bytes as the ones built for 4, 8 or 64 (batches of 3, 7, 8, 64 and 1024 all agreed with each other) | the world count is **pinned at 64** for every batch. Batch invariance becomes structural; a batch of 1 costs ≈830 → ≈278 f/s and batch 64 is unchanged |
| `test_determinism` failed on `h1-slide` and `h1hand-slide` | the tail worlds of a partial chunk were never written, so the BVH bounds included whatever the memory pool had left there | `_upload` **fills every world**, padding with the last requested row. Both slide cells have passed since |
| `HumanoidBenchH1HighbarSimple` still differed by 1/255 at one row, **reproducibly**, between a batch of 1 and a batch of 1024 | the residual, irreducible form: at a batch of 1 the other 63 worlds hold a copy of that state, at 1024 they hold different ones, so the shared BVH differs and a tie is broken the other way. Highbar's posts are thin cylinders meeting the bar | **not shipped** |
| `HumanoidBenchH1HandPowerlift` failed `test_determinism` on 2 runs of 4, 1/255 | the same class, hit rarely: **1,140 consecutive renders across all 19 cells with every context alive are bit-identical** (`probe16_session_flake.py`), so it needs something a long suite run does and a render loop does not | **not shipped** |

The two excluded cells are excluded on this and on nothing else — their fidelity numbers are as good as
their neighbours' (`highbar_simple` interior-exact 0.960, the best of the 19). The lesson generalises
past this family: **a batched ray-tracer that shares one BVH across worlds cannot promise that frame
*i* is independent of what else was in the launch**, and the scenes where that bites are the ones with
coincident geometry.

### 5.2 The open defect, and why the family does not register

Excluding those two did **not** close it. Six full runs of `pytest tests/ -k HumanoidBench`:

| run | what changed | result |
|---|---|---|
| 1 | context sized to the chunk | 21 failed (19 of them the missing `DOWNSAMPLE_EXACT_FRAC` entries), plus `determinism[h1hand-powerlift]` and `batch_size[h1-highbar_simple]` |
| 2 | world count pinned at 64, bounds table filled | **3 failed**: `determinism[h1hand-slide]`, `batch_size[h1-highbar_simple]`, `batch_size[h1-slide]` |
| 3 | context cache raised to 32 | cancelled; `default_style[h1-balance_simple]` and `determinism[h1hand-powerlift]` already failing |
| 4 | every world filled on upload | **2 failed**: `determinism[h1hand-powerlift]`, `batch_size[h1-highbar_simple]` |
| 5 | those two cells dropped | **3 failed**: `default_style[h1-balance_simple]` (231/255), `determinism[h1hand-slide]`, `batch_size[h1hand-sit_hard]` |
| 6 | a third synchronisation, before the upload | **3 failed**: `determinism[h1hand-slide]`, `determinism[h1hand-spoon]`, `batch_size[h1hand-sit_hard]` |

Three hypotheses were tested and closed, and each fix stuck — the residual survived all of them:

| hypothesis | probe | result |
|---|---|---|
| it needs many render contexts alive | `probe16_session_flake.py`, 19 envs, every context live | **1,140 consecutive renders, 0 flaky, max 0/255** |
| it needs a second context at another resolution | `probe17_two_contexts.py` | **80 renders after building one, 0 flaky** |
| it is the scene | `probe15_determinism.py`, 40 renders x 19 cells in fresh processes | **0/40 on all 19** |

So the trigger is something a full suite run does that none of those do, and it is not closed. The
failure is 1-2/255 (once 231/255, on `default_style`, which the pre-upload synchronisation of run 6
appears to have removed) and it moves between cells, which is why dropping cells does not help. Until
it is closed the module raises at import unless `STABLE_ENVS_HUMANOIDBENCH=1`, so the repo's suite
stays green and `stable_envs.envs.import_errors()` names the reason in one line — the same mechanism
`dmcontrol.py`, `fetch.py` and `robocasa.py` use for a dependency they cannot satisfy.

**Closed, 2026-08-11: it is the world SLOT, not what else is in the launch.** Every probe above, and
§5.1's whole framing, asks what the *other* worlds contain. The sharper question is what happens when
they contain nothing else to argue about. `scratch/perf/worldslot.py` fills **every** world of one
launch with the **identical** `qpos` — no tail, no padding, no residue from a previous call, one
appearance — and compares each world against world 0:

| cell | pixels differing from world 0 | max | slots |
|---|---:|---:|---|
| `h1-balance_simple` | **2** | **1/255** | 21 and 42 of 64 |
| `h1-walk`, `h1hand-spoon`, `h1-slide`, `h1-stair`, `h1hand-stair`, `h1hand-sit_simple` | 0 | 0 | — |
| `MetaWorldReach`, `FetchReach` (controls) | 0 | 0 | — |

So a frame here is a function of the state **and of the index it is rendered at**, which is exactly
what makes a batch-size change move a pixel: the same state sits somewhere else. Filling the tail was
necessary and is not sufficient, and no amount of care about the *contents* of the launch can close it.

The mechanism follows from the same shared tree §5.1 names, one level down. `mujoco_warp` builds one
scene BVH over `nworld * ngeom` primitives with a per-world root; a world's subtree holds only its own
geometry, but the tree is *built* over every world at once, so two slots holding identical geometry can
get differently shaped subtrees. Closest-hit is settled by a strict `<` on the ray parameter in visit
order, so two exactly coincident surfaces are a tie and the tie goes to whichever node the traversal
reached first. **The fix is a deterministic tie-break — lowest geom id at equal `t` — and it belongs in
`mujoco_warp`, not here.** Independently reproducible without the suite: the same probe run across
world counts (`scratch/perf/chunk_invariance.py --chunks 8,16,32,64,128,256`) moves 1–13 pixels by up
to 128/255 on `h1-walk`, `h1hand-spoon`, `h1-balance_simple` and `h1hand-stair`, and 0 on the rest —
non-monotone in the world count, which is what a re-built tree looks like and what a race does not.

One residual is worth naming separately because it is not the floor. **A large tilted flat face is 2–5
levels dark.** On `h1-slide`'s ramp — 33 % of that frame — the signed difference is
`{−5: 14 %, −4: 11 %, −3: 39 %, −2: 34 %}`, one-sided, and it survives `specular = 0` on both sides and
`offsamples = 1` on the reference, so it is a diffuse constant rather than a highlight or a coverage
rule (`probe9_onelevel.py`). It costs interior exactness (0.62 on `slide`) and almost nothing else
(within-8 0.979). It is the reason two of the shipped cells have a much lower interior number than
their SSIM suggests, and it is unexplained.

## 6. Reproducing

```
python: ~/venvs/se-hb/bin/python   (mujoco 3.11.0, mujoco-warp 3.11.0, warp 1.16.0,
                                    gymnasium 1.2.3, humanoid_bench 0.2 installed --no-deps)
build:  git clone https://github.com/carlosferrazza/humanoid-bench ~/src/humanoid-bench
        ~/venvs/se-hb/bin/pip install --no-deps -e ~/src/humanoid-bench
        ~/venvs/se-hb/bin/pip install "jax[cpu]"      # humanoid_bench.wrappers imports jax.numpy
run:    srun --partition=open_research --account=open_research --qos=normal --gres=gpu:1 \
             --cpus-per-task=8 --mem=64G ... MUJOCO_GL=egl
```

One process per scene for everything measured here.

| script | what it answers |
|---|---|
| `probe1_facts.py` | the four counts over all 49 scenes; camera modes; alpha and textured-primitive census |
| `probe2_gates.py` | G1/G2/G3, the reference check, the frame budget, the shadow/reflection share, the static-plate test |
| `probe3_mjwarp.py` | the surgery-inertness proof, L0 and the ladder, per-class attribution, near/far bands, first throughput |
| `probe4_supersample.py` | the ss 1–4 sweep and the per-state spread |
| `probe5_all_tasks.py` | L0 for all 49 scenes, plus the reference/ours/4×-difference strips |
| `probe6_floor.py` | the floor's error histogram, the line-vs-field split, exactness by image row, fog |
| `probe7_variants.py` | V0 / hide-translucent / hide + flat floor, all against the same official frame |
| `probe8_haze.py` | the haze pass, and the (ratio, batch) throughput curve with its crossovers |
| `probe9_onelevel.py` | the tilted-face residual: signed histogram, specular and multisample ablations |
| `probe10_deaddims.py` | which `qpos` entries reach a pixel, per candidate cell |
| `calibrate_tier.py` | the declared `visual_fidelity`, through the env class, 64 states × 4 draws |

## 7. Upstream-worthy findings

1. **`mujoco_warp` has no alpha path**, and `humanoid_bench`'s obstacle courses are built from
   `rgba="... 0.1"` geometry. Reported before on [Adroit](adroit.md); this is a second independent
   family where it is decisive, over up to 62.8 % of a frame.
2. **`mujoco_warp` builds every texture at one mip level and samples with no level of detail.** Third
   family (locomotion, RoboCasa, here). Here it costs a family that is otherwise clean — one casting
   light, no cube textures, zero textured primitives, `nmesh` 21.
3. **gymnasium's legacy `add_marker` path is broken under `mujoco` 3.11** (`mjv_initGeom` with a
   scalar `size`), which takes HumanoidBench's `reach` and `push` with it.
4. **dm_control's vendored field table (`dmc_deps/dmc_sizes.py`) names 17 arrays `mujoco` 3.11 no
   longer has.** `struct_indexer` already skips non-ndarray fields, so a one-line `__getattr__` fix
   makes HumanoidBench constructible on a modern MuJoCo; it is applied in `_hb_compat` and is worth
   sending upstream.
5. **`humanoid_bench`'s per-episode draw goes through the process-global `np.random`**, not through
   `self.np_random` — the same defect Meta-World has, so `reset(seed=k)` alone does not reproduce an
   episode.

## 8. The other backend, measured: nvdiffrast draws the grid and loses the exactness

[`stable_envs/render/nvdiff.py`](nvdiffrast.md) exists for exactly the operation MJWarp is missing here
— prefiltered texture sampling off a mip pyramid — so it was pointed at this floor rather than assumed
to fix it. `probe11_nvdiff.py`, `h1-walk`, the same 16 states, against `mujoco.Renderer` at the same
camera in the same interpreter (`~/venvs/senvdiff`, `mujoco` 3.9.0, so this row is internally
consistent and not directly comparable to §3's absolute numbers):

| renderer | interior-exact | mean /255 | within 8 | SSIM |
|---|---:|---:|---:|---:|
| MJWarp, floor as-is (dithered) | 0.9158 | 4.13 | 0.901 | 0.628 |
| **MJWarp, floor flattened** (what ships) | **0.9503** | 2.79 | 0.935 | 0.851 |
| nvdiffrast, `mipmap = False` | 0.3853 | 3.85 | 0.937 | 0.856 |
| **nvdiffrast, `mipmap = True`** | 0.3880 | **2.91** | **0.969** | **0.937** |

The mip pyramid does what it was built to do — it is worth SSIM 0.856 → 0.937 and within-8
0.937 → 0.969 on this floor, switching one flag — and it is **not** a route to the parity tier here,
because nvdiffrast's shading is a reimplementation rather than MuJoCo's own composition, so its
interior-exactness is 0.388 against MJWarp's 0.950. The two backends are on opposite sides of the same
trade: MJWarp gets almost every flat pixel bit-right and cannot draw the grid; nvdiffrast draws the
grid and gets a level wrong nearly everywhere. It is also 3–5× slower at 224 px, and this floor is
100×100 texels, which is not a power of two, so `prepare_texture` has to resample the base level
before it can build a pyramid at all — itself a declared change to the picture.

So the family ships on MJWarp, and the honest statement of what would move it to **parity** is
narrower than "a mipmap": it needs level-of-detail selection **inside `mujoco_warp`'s own render
kernel**, which needs ray differentials that kernel does not compute. L3 in §3.1 says the rest of the
pipeline is already at 0.995 interior-exact once the texture is out of the way, so that one upstream
change is the whole distance, and the number it has to beat is the **0.8772** in §3.
