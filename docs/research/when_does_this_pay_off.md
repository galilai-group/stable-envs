# When does state → GPU frame pay off?

Twenty-four families and 196 environments have now been measured against the thesis, across 15
interpreters (`scratch/census/union.json`). The interesting part is still that **the wins and the
losses come out of the same rule** — not from a list of successes with the boundary left implicit.

**The thesis has two halves and they no longer have the same answer, so this document keeps them
apart.** *Storage* holds for every family here: summed over the registry the state that determines a
frame is 37,228 B against 71,805,384 B of raw pixels, a factor of 1,929, and the per-environment
median is 4,332× (`examples/results/decode/ALL.json`, `state_b` and `raw_b`). No throughput result
moves that. *Regeneration cost* is per family, and the gate that decides it is G4. Against one core of
JPEG decode almost everything wins; against **eight worker processes batch-decoding JPEG-in-Lance**,
which is the strongest disk-based path this repo can measure, several families do not. G4 below is
stated against one core, because that is what the historical rows use; the eight-worker column and the
families that lose to it are in `examples/paper/tables/family.tex`, generated from the same JSONs.

Two of the entries sharpen the rule rather than lengthen the list. **Crafter** is Craftax's game
in another language and it fails G2 as shipped for exactly the same reason and by the same fix, which
turns "store the render window, not `EnvState`" from an observation about one library into a property
of the genre. **Griddly** is the family where the official renderer is *slower than reading the
dataset it would replace* (0.24–0.65 cores of decode, `griddly.md` §1) while the render is 98–99.5 %
of the env step — the mirror image of RoboCasa, where the renderer was fine and the decoder rebuild
was fatal.

This document states that rule as four gates, each measurable **before** writing a renderer, most of
them in an afternoon. Every number below is cited from a measurement in `docs/research/` or from the
probes in `scratch/`; derived figures are marked as such and gaps are marked as gaps.

---

## The criterion

> **The approach pays when the frame is a *lookup into a decoder that is a constant of the family*,
> indexed by a small per-step state. It fails when the decoder is itself content — regenerated per
> episode (RoboCasa) or written per title (Atari).**

Operationally, four gates. A family must clear all four; each one has killed at least one family, which
is what makes it a criterion rather than a victory lap.

### G1 · Purity — is the frame a function of a state the env exposes?

**Test.** Drive two *different* internal states to the same candidate state, render both, diff. Or:
render the same state twice, after a different rollout, with irrelevant state fields randomised.
**Threshold: 0 differing pixels.** This is a binary property, not a tolerance to negotiate.

This gate is cheap and it is the one that produced the cleanest NO-GO in the set. Forcing two ALE
emulator states to byte-identical 128-byte RAM gives **different next frames on 13.75 % of pairs**
(29.2 % writing RAM into a freshly reset console; **100 % of pairs** for `pong`, `asterix`, `qbert`,
`amidar`, `chopper_command`). The 2600 has no framebuffer — the picture exists only after a
cycle-accurate 6507 has raced the TIA for a frame.

It also disqualifies LunarLander for a different reason: `render()` **mutates what it draws**
(`obj.ttl -= 0.15`, then `_clean_particles`), so rendering twice gives two different images.

### G2 · Compression — is the *render-relevant* state smaller than the frame?

**Test.** Identify the slice of state the renderer actually reads, and divide by the frame size.
**Threshold: ratio < 1**, computed on the render slice and amortized per episode — **not** on the
state object the env hands you.

That qualifier is the whole content of this gate. Craftax's shipped `EnvState` is **6.37× bigger than
its frame** (273,276 B vs 42,900 B) because it carries all 9 levels of a 48×48 world, almost none of
it on screen; the render window is **5,268 B = 0.12×**. Procgen's blob is 1.8–3.0× the frame, of
which ~90 % is three `std::mt19937` states serialized as **decimal text**, none of which reaches
`game_draw`. Both families look like refutations as shipped and are wins once sliced.

Applied honestly it also fails ALE a second time: the minimal sufficient state (`cloneSystemState()`
= 14,439 B) is **0.43× the 33,600-byte indexed screen** — a factor of 2.3, not 100.

### G3 · Shared decoder — is the state→pixel map a constant of the family, or per-episode content?

**Test.** Reset twice. Compare the compiled scene / atlas / primitive set. **Threshold: identical
across episodes, and small enough to stay resident on the GPU.**

This is the load-bearing gate, and the cheapest decisive one in the set — two resets and a diff of
`ngeom` would have settled RoboCasa in about 20 seconds. RoboCasa's model changes **shape** every
episode: `nq` 91 / 103 / 131 / 133, `ngeom` 1,201–2,341, `nmesh` 189–408. Pinning layout **and** style
still yields four different XMLs. The decoder is a **24.07 GB asset library** plus a 6–18 s MuJoCo
compile. (The line that used to stand here — "RoboCasa's own 6.4 kB scene descriptor reproduces only
94.95 % of pixels" — is **retracted**: it reproduces **99.6 %**, and the residual is one robot-base
geom. The old figure came from restoring state by index across two compilations that order joints
differently. G3 still fails, on the compile and the 24 GB library, not on the descriptor.)

Contrast the wins, where the decoder is a single small constant serving the entire family: MiniGrid's
**6.08 MB atlas covers all 172 env ids**; MiniHack's is 1,082 tiles / 831 kB; Craftax's is 143 PNGs /
212 kB; OGBench has **`nmesh = 0` in all 11 models** and a fixed camera, so floor + walls bake to one
constant plate; classic control is **four `gfxdraw` primitives** for the whole family.

G3 also predicts failures *before* rendering anything. DM Control's 2-D domains all include
`common/skybox.xml` and `common/materials.xml` — gradient skybox, checkerboard floor with
`reflectance=.2`, `shadowsize=2048`, perspective cameras — so the decoder is not a finite tile set and
the family needs a 3-D path, no pixel diff required to know it.

### G4 · Economics — does regenerating beat reading, and is there anything to speed up?

**Test.** Time (state → frame) against decoding a stored frame on the same machine. Separately,
measure the render's share of an env step: that is an Amdahl bound on any data-generation speedup.

**Threshold: regeneration faster than decode, using a renderer that meets the family's parity bar.**
Otherwise the dataset you replaced was cheaper than the renderer you wrote — or the renderer you
wrote is drawing something else. The qualifier is not pedantry: RoboCasa has a renderer 5.1× a core
of decode and a renderer that is bit-exact, and they are not the same renderer.

**Measure the loop and the setup separately, and amortize the setup over the episodes that actually
share it.** This qualifier is to G4 what the render-slice qualifier is to G2, and it was added after
RoboCasa's original G4 number turned out to be measured three ways wrong (details below). A single
end-to-end number hides which term you are losing to, and they have completely different fixes:
a slow loop wants a better renderer, a heavy setup wants a *shared decoder* — which is G3.

The two measured anchors bracket the entire range:

| | render share of one env step | regenerate 500 frames | decode 500 stored JPEGs |
|---|---|---|---|
| **PushT** (96²) | **77.0 %** (physics 16.1 %) | — | — |
| **RoboCasa** (128²) | **3.3 %** (physics 96.7 %) | **2.245 s** EGL · **10.05 s** MJWarp · **0.028 s** loop only | **0.097 s** |

RoboCasa's decomposition is the instructive case, and the original entry here — "4.03 s, 34.7×
slower" — was wrong in three ways, each inflating the regeneration side:

1. it averaged a **cold** MJCF compile into the compile term (3.85 / 1.23 / 1.02 s → 2.03 s reported;
   warm is **0.97 s**);
2. it ran `mj_forward` per frame — full dynamics for a state already in hand. `mj_kinematics` +
   `mj_camlight` is **0.029 ms** against `mj_forward`'s **0.954 ms**, so half the "render loop" was
   physics that regeneration does not need;
3. it rendered geom groups 0–2, i.e. with the collision primitives painted over the scene — a frame
   RoboCasa never produces, which also mis-sizes the JPEG it is compared against.

Corrected, and with the render loop batched onto the GPU (MJWarp, 128², group 1, one H200):

| | seconds per 500-frame episode | vs. one core of decode |
|---|---|---|
| MuJoCo EGL, end to end | 2.245 (setup 1.79 + loop 0.452) | 23.1× slower |
| MuJoCo EGL, render loop only | 0.452 | 4.6× slower |
| MJWarp, end to end, K = 1 | 10.05 (setup 10.02 + loop 0.028) | **103× slower** |
| **MJWarp, render loop only** (batch ≥ 256) | **0.028** | **3.5× faster** |
| decode 500 stored JPEGs, one core | 0.097 | — |

So the gate flips depending on which term you look at, and the split is the finding: **the loop wins
by 3.5×, the setup loses by 103×.** (Both figures are superseded below.) Break-even needs
**K ≈ 145 episodes sharing one compiled
kitchen**, and RoboCasa gives you K = 1 — every reset resamples the object set, and its own replay
code stores a fresh 385 KB MJCF per demo. Moving to the GPU makes it *worse* end to end, because
`put_model` + render-context construction is 7.6 s where an EGL context is 0.83 s.

**Third pass: RoboCasa fails G4 independently after all, and this is the durable answer.** The two
readings above are both about *setup*, and setup turned out to be reducible — `create_render_context`
re-uploads MuJoCo's whole concatenated texture blob once per texture, so context creation falls
8.98 s → 1.34 s, `put_model` falls 3.70 s → 0.048 s on a model stripped to what it draws (which
renders bit-identically), per-scene setup falls **10.04 s → 1.41 s**, and the break-even drops from
K ≈ 145 to **K ≈ 21**. Real corpora clear that: bucketed by kitchen,
RoboCasa's shipped corpora hold **12 / 240 / 2,500 episodes per `(layout, style)`**.

None of it helps, because the gate fails on the LOOP once the parity requirement is applied. Every
env in this repo declares `parity_interior_min = 1.0`, and for RoboCasa exactly one renderer can meet
that — MuJoCo's own, because it *is* the official renderer. With the model already compiled and
resident, stripped to only the drawn geometry (verified bit-exact), it saturates at **1,925 f/s
across 16 processes at 128 px against 5,137 f/s for one core of decode = 0.37 cores.** The fast
alternative, MJWarp at 26,400 f/s (5.1 cores), reaches **5.9–23.1 % of interior pixels exact**.

**Exact costs 2.7× more than reading; fast is 4–17× short of the bar.** Both hold at K = ∞, so this
is a G4 failure that does not depend on G3 at all. At a 3.3 % render share a *free* renderer would
still speed up data *generation* by at most 1.03×, which is the one part of the original entry that
has stood through all three passes.

ALE fails this gate by irrelevance rather than by cost: rollouts are fully determined by
(ROM, seed, actions) = **1 byte per step**, and the emulator regenerates at **6.6k–54k fps on a single
core**. There was never a storage problem to solve.

That last sentence was an argument until it was measured, and it now is. On one core of the same
node, `ale-py` replays at **5,011–6,973 f/s** against **2,912–3,291 f/s** of PNG-RGB decode (**1.52–2.40×**). The honest comparison is against **indexed** PNG, because this document's own table records the ALE frame as 33.6 kB *indexed*: decoded that way the baseline runs at **5,024–5,295 f/s** and replay is **0.96–1.39×** it — slower than reading, for `breakout`. So regeneration and reading are the same order here and the G4 verdict is *moot*, not a win, which is the only
family in this document where the gate is decided by the *reference* rather than by a port. A GPU
emulator does not change it: CuLE (built from `MehrdadMoghimi/cule` plus `chengscott`'s CUDA-13
fixes) needs **64 parallel environments to break even with that single core**, and its frames are not
ALE's — 4 of 22 ROMs bit-exact at 200 steps, **0 of 7 at 2,000**, max abs diff 252, two ROMs
rendering 210×160 where ALE renders 250×160. `pixel_native.md` §1.1 has the full table.

### G5 · Fixed cast — can a fixed-width state hold every entity *and* keep every dimension live?

**Test.** Harvest real rollouts. Take `max(entity_count)` — the cap cannot be below it without the
renderer silently dropping an entity. Then take the entity count a random draw of the *pool* reaches
— no slot above that can change a frame, so no slot above that is a legal state dimension.
**Threshold: those two numbers must not cross.**

This gate was added last and it is the only one derived from a family that had already passed the
other four. Procgen's `bigfish` renders **bit-exactly — 240 frames, 100.0000 % of pixels, max abs
diff 0** — and still cannot ship: fish spawn at Bernoulli(1/10) per step, so 12,800 harvested states
reach **24 at once** (p99 13, tail `{17: 2, 20: 1, 21: 1, 24: 1}`) while the pool holds **mean 4.3,
max 10**, leaving slots 7+ dead at B=8. The cap must be ≥ 24 and ≤ about 15. Nothing in between
exists, and biasing the pool toward busy frames would close the gap only by making `sample_states`
stop being the env's own distribution — which is the assumption every parity number in this repo
rests on.

The gate is cheap (one histogram of entity counts) and it is *predictive*: applied to procgen's other
fixed-camera games without rendering anything, it passes `chaser` (3 orbs, 3 eggs, 3 enemies, fixed
in `game_reset` — and `chaser` duly ships bit-exact) and fails `bossfight` (4–142), `dodgeball`
(8–17), `plunder` (3–16) and `starpilot` (1–22). G4 agrees independently, which is the useful check
on a new gate: `bigfish` is **0.47× its own reference at batch 4096 and never crosses over**, because
33 entity slots is 33 serial full-frame composites.

The general statement: **G2 asks whether the render state is small; G5 asks whether it is a fixed
shape.** A decoder that is a family-wide constant is not enough if what indexes into it is a variable-
length list. Every win in the table below has a cast fixed by the game — a grid, an agent, a bounded
set of pieces — and that had been true by accident until a family made it a requirement.

---

## The table

Every family for which evidence exists. Sources: `pixel_native.md`, `minigrid.md`, `ogbench.md`,
`classic_control.md`, `robocasa.md`, and the repo README for PushT.

| family | G1 purity | G2 render-state / frame | G3 decoder | G4 regen vs decode | verdict | **gate that decided** |
|---|---|---|---|---|---|---|
| **PushT** | ✅ 100 % interior px exact | ✅ **1/1382** (5 f32 = 20 B vs 27,648 B) | ✅ handful of polys + discs, one Triton kernel | ✅ 2.96 M f/s; render = **77.0 %** of step | **GO** (shipped) | — |
| **MiniGrid + BabyAI** | ✅ **21,216 frames, max abs diff 0**; 6,880 round-trips, 0 mismatches | ✅ **1/1013** (545 B raw encode vs 551,835 B; 1/2967 at 186 B packed) | ✅ **6.08 MB atlas, all 172 ids**, 492 distinct tiles | ✅ 6.08 M f/s vs 2,378 official = **2,557×**; 1.1 TB/s | **GO** (shipped) | — |
| **Classic control** (CartPole, Pendulum, Acrobot, MountainCar) | ✅ **100.00000 %**, 256/256 frames each, 0 px of 250.9 M | ✅ **~1/45000** (2 floats vs 600×400×3) | ✅ **4 `gfxdraw` primitives** + 228-bitmap LUT (3.9 MB) | ✅ 40–67 k f/s = **59–133×** pygame | **GO** | — |
| **MiniHack / NLE** | ✅ bit-exact 40/40, max abs diff 0 | ✅ **1/384** (3,318 B vs 1,274,112 B) | ✅ 1,082-tile atlas, 831 kB; frame = one gather | ✅ 247 k f/s vs 178 official = **1,391×**; 315 GB/s | **GO** | — |
| **OGBench mazes** (25 cells / 11 scenes) | ✅ 0 px differ under re-render, `qvel`×100, different rollout | ✅ **1/9408** (point, 2 f64) · **1/1254** (ant, 15) · **1/672** (humanoid, 28) vs 224²×3 | ✅ **`nmesh = 0` in all 11**; fixed camera → one baked plate | ✅ 52 k–467 k f/s = **113–1,092×** MuJoCo | **GO** (99.64–99.95 % bit-exact) | — |
| **Craftax** | ✅ `render(s)·f32(1/255)` == `get_obs`, 100 % of steps | ⚠️ **as shipped 6.37× BIGGER** (273,276 B vs 42,900 B); slice = **0.12×** | ✅ 143 PNGs / 212 kB, 16×16 atlas | ✅ 172 k (full) / 1.3 M (Classic) f/s under `vmap` | **GO, conditional** — store the slice, not `EnvState` | **G2** |
| **Crafter** | ✅ 1,111 frames bit-identical, max abs diff 0 — *given* `daylight ≥ 0.5`; below it the reference draws noise from the world RNG | ⚠️ world arrays alone **1.67× BIGGER** (20,480 B vs 12,288 B); render window = **0.026×** (38.4×, or 153.6× packed) | ✅ **56 PNGs / 9,452 B of base64**, 370-tile atlas composed per tile size; byte-identical across seeds and resets | ✅ 702 k f/s vs 1,375 official = **510×**; ≈ 59 cores of decode | **GO** (shipped) | **G2** (same shape as Craftax) |
| **D4RL maze2d / PointMaze** | ✅ 0 px under a different history, `qvel`×100, and a fresh instance — *after* pinning the reference's history-dependent free camera (37.9 % of pixels apart otherwise) | ✅ **1/9,408** (4 f32 vs 224²×3) | ✅ `nmesh = 0`; a plane, 19–63 boxes, 2 spheres; one baked plate per scene | ✅ 48.6 k f/s vs 2,293 MuJoCo = **21×**; ≈ 7 cores | **GO** (shipped, 99.49–99.77 % exact, interior 1.0) | — |
| **Overcooked-AI** | ✅ max abs diff 0 over 800 round-trips and 633 cross-history pairs; 1,980 frames rebuilt from a position-independent tile atlas, 0 mismatches | ✅ **1/22,500** (15 B vs 337,500 B) | ✅ 93 sprite frames / 44,093 B, class-level singletons shared by all 45 layouts | ✅ render = **89.9 %** of a step (highest in the repo); measured **1,241–1,861 cores replaced** | **GO — shipped, 5 JaxMARL layouts, 100 % exact** | — |
| **Griddly** | ✅ 0 collisions over 5,352 cross-history repeats, 13 games; `render()` does not mutate | ✅ **1/216 – 1/763**, and passes **as shipped** (`get_state()` = 0.029–0.042×) | ✅ **207 PNGs / 270 kB for all 32 games**; 11–23 tiles per game, saturating in 2–6 episodes; index is a **3×3 stencil** (`WALL_16`) | ✅ official renderer is **0.24–0.65 cores**, i.e. SLOWER than decode; render = **98.1–99.5 %** of a step; measured **1,789–3,648 cores replaced** | **GO — shipped, 7 games, 100 % exact** | — |
| **Procgen** | ✅ `get_state()` round-trips exactly, 16/16 games | ⚠️ blob 1.8–3.0× bigger (90 % = RNG as decimal text); render state **0.06–1.40×**, per-step delta **13–95 B** | ✅ 779 sprites / 25 MB constant; the raster rules turned out to belong to `BasicAbstractGame` and be shared by all 16 (measured, not read) | ✅ 2.6–28× the official renderer at B=4096, **crossover B=128–2048**; `bigfish` **0.47× and never crosses** | **GO** for `maze`/`heist`/`miner`/`chaser` (shipped, 100 % exact) | **a fifth gate — see below** |
| **Griddly** *(rotation obstacle re-measured)* | ✅ (unchanged) | ✅ (unchanged) | ✅ **atlas ENUMERATED: 385 composed tiles / 0.648 MB for all 32 games**, rotation costs ×1.19 median (not the ×4 estimated) | ✅ `tilemap.blit` reproduces **1,200/1,200 POV frames and 480–666 global frames bit-exactly, max abs 0**, at 3.1–13.6 M f/s = **1,941–3,615 cores**, crossover **B=1** | **GO** — shipped | — |
| **Meta-World** (50 v3 tasks) | ✅ *after a withdrawn NO-GO*: a reset writes placement into the **model**, not `qpos` — but a restore ladder (`scratch/metaworld/probe_state_gate.py`) makes a *different* episode bit-exact on **50/50** once 14 model arrays travel with the state, against **0/50** for the do-nothing control at a median 239/255 | ✅ `qpos` + the episode's live model slice, **0–6 named floats**, vs 480²×3 | ✅ *withdrawn*: ported and measured, interior-exact **median 0.999700**, min 0.980766, 48/50 above 0.99 | ⚠️ **1.2–3.8 cores** against one core; **0.19–0.48×** against eight workers | **GO at the parity tier, LOSES G4 on the strong bar** — shipped with the ratio printed | **G1+G3, both withdrawn** |
| **Fetch** (12 ids) | ✅ | ✅ **1/6,912 – 1/9,600** | ✅ a baked plate plus the arm's 18 STL meshes | ✅ **4.3–6.9 cores** against one core, and **1.23–1.28×** against eight workers — the only MJWarp family that clears the strong bar | **GO** (shipped, interior 0.9988 with a `parity_note`) | — |
| **LunarLander-v3** | ❌ `render()` mutates particle `ttl`; obs lossy (no leg joint angles) | — | ⚠️ three different AA paths in one frame | — | **GO — shipped**, 99.969 % all-pixel, interior 1.0. The G1 objection was about `render()` mutating particle `ttl` and about the *observation*; the state the env exposes is sufficient, and the frame is a function of it | **G1, withdrawn** |
| **DM Control 2-D domains** | — | — | ❌ skybox + checkerboard floor + `shadowsize=2048` + perspective cameras | — | **GO via the 3-D backend — 16 cells shipped**, 11 in the parity tier (interior 1.000000 where the metric is defined) and 5 in the visual tier. The 2-D-path NO-GO stands; it was never a verdict on the family | **G3, routed not rejected** |
| **HumanoidBench** (H1 / H1-hands, 49 scenes) | ✅ 0/255 on four probes — re-render, `qvel`×100, a different rollout, a FRESH `MjData` | ✅ **0.001–0.003×** (`nq` 26–83 f64 vs 256²×3) | ✅ **not one model array changes** across a reset or 5 steps; 21–50 STL meshes, `ntex = 2`, **1** casting light, **0** textured primitives on the locomotion cells | ⚠️ 9,330–16,835 f/s = **2.3–6.8 cores** of decode, and **0.46–0.78×** against eight workers batch-decoding JPEG-in-Lance — a pass on one core and a fail on the strong bar | **VISUAL tier, 17 of 49 — shipped behind `STABLE_ENVS_HUMANOIDBENCH=1`** (`mujoco_warp`'s batched render is not bit-reproducible across a long suite session: 2-3 of ~660 tests fail every run by 1-2/255, on a different cell each time) | **the FLOOR** decides the tier — no plate exists (every camera is TRACKCOM), and MJWarp point-samples a minified 2 cm grid → interior 0.877 |
| **RoboCasa / robosuite** | ✅ 0/255 max diff — **but only given a fixed compiled model** | ✅ 135× per step (364 B vs 49,152 B); the scene descriptor is `ep_meta` at **1.7 kB gzipped**, which replays **99.6 %** of pixels (the recorded 94.95 % was state restored by index across two compilations) | ❌ compiled model is per-episode (**24/24 distinct** at a pinned kitchen) — but the **drawn** kitchen is constant (357 visible geoms identical over 64 episodes); a per-kitchen union model exists and its compile cost converges to ~0.4 s/episode against 0.097 s to decode one | ❌ **exact path 0.37 cores** (MuJoCo EGL, 1,925 f/s over 16 procs, setup = 0); fast path 3.2–5.1 cores at **5.9–23.1 % interior-exact** vs a required 100 % | **NO-GO at the parity tier; SHIPPED at the visual tier**, 4 cells at SSIM 0.821–0.921. The parity claim is what G4 refuses to fund, not the family | **G4 on the loop** (G3 also fails, but is no longer what decides) |
| **Atari / ALE** | ❌ identical 128 B RAM → **different frames on 13.75 %** of pairs (100 % for 5 ROMs) | ❌ **0.43×** (14,439 B state vs 33,600 B screen) | ❌ cycle-accurate 6507+TIA, **a different program per ROM**; no atlas | ❌ moot — emulator already 6.6 k–54 k fps/core from 1 B/step | **NO-GO** | **all four** |
| **MinAtar** (`gpu_native.md`) | ✅ 29,415 cross-history pairs, **0 differing pixels** | ✅ **0.127–0.167×** sliced (❌ 1.33–3.33× as shipped) | ✅ **84 bytes** of palette for all 5 games | ❌ nothing to regenerate: the observation IS the state, 1 B/step, 28 k–177 k steps/s/core | **NO-GO** | **G4** |
| **Octax** (CHIP-8 in JAX) | ✅ full-state round-trip into a **fresh instance, 0/12** — the test ALE fails | ❌ **1.015×** the frame; the render slice **is** the frame | ✅ vacuously — 2 RGB triples, 6 bytes, all 14 games | ❌ moot — 1.34 M steps/s from (ROM, seed, actions) | **NO-GO** | **G2 (degenerate)** |
| **PixelBrax** | ✅ re-render 0, different history 0 | ✅ **0.087–0.113×** | ✅ `n_mesh = 0`, geoms identical across resets — but the map is a shaded triangle rasteriser | ❌ **1.43 cores**; render = **98.8 %** of the step | **NO-GO** | **G4** |
| **KAGE-Bench** | ✅ 0 differing pixels after a different 29-step rollout | ⚠️ **5.68–6.69× BIGGER** as shipped (`bg_image` is one whole frame inside the state); **0.051–0.060×** with the window bit-packed AND the background stored as an index | ✅ **6.46 MB**, constant across resets | ⚠️ **2.3 cores** at the benchmark's own visuals (3,388 at plain ones) | **CONDITIONAL** — store the slice and an asset index | **G2** |

### Derived: how many CPU cores of JPEG decoding does one GPU replace?

The per-family reports each benchmark against *their own* official renderer, which answers "is the GPU
path faster than what it replaces" but not "is regenerating cheaper than reading". A single decode
baseline makes them comparable (`scratch/robocasa/bench_decode.py`, one core, decoding from memory —
the *favourable* case for stored datasets): **11,842 f/s @64² · 8,341 @96² · 7,483 @128² · 6,762
@224² · 5,486 @256²**.

| family | envs | frame | official f/s | ours f/s | decode f/s/core | **cores replaced** |
|---|---|---|---|---|---|---|
| Griddly | 7 | 144x120… | 99–145 | 2,307,114–9,438,816 | 851–5,276 | **1,789–3,648** |
| Sokoban | 1 | 160×160 | 253 | 9,571,978 | 3,983 | **2,403** |
| Overcooked (JaxMARL) | 5 | 192×224… | 41–1,595 | 3,615,605–8,145,368 | 2,914–4,399 | **1,241–1,861** |
| MiniGrid / BabyAI | 7 | 224×224… | 117–942 | 711,991–4,479,313 | 608–4,011 | **330–1,226** |
| MiniHack | 1 | 336×1264 | — | 244,456 | 698 | **350** |
| Jumanji | 8 | 200×200 | 3–88 | 409,277–1,286,366 | 3,086–4,387 | **105–340** |
| Procgen | 4 | 64×64 | 899–975 | 313,372–1,588,402 | 8,344–11,055 | **28–190** |
| PushT | 1 | 96×96 | 472 | 1,256,557 | 10,259 | **122.5** |
| OGBench | 25 | 224×224 | 82–491 | 48,978–206,740 | 1,893–4,249 | **11.9–93.6** |
| swm nav | 5 | 224×224… | 52–772 | 71,140–187,629 | 1,083–4,985 | **15.3–86.7** |
| Crafter | 1 | 64×64 | — | 665,836 | 8,186 | **81.3** |
| Craftax | 1 | 112×144 | — | 569,934 | 7,039 | **81.0** |
| PGX | 4 | 240×240… | 21–108 | 78,322–109,484 | 2,055–3,160 | **25.0–45.7** |
| classic control | 5 | 400×600… | 249–541 | 33,216–39,843 | 1,120–1,626 | **22.3–35.6** |
| D4RL PointMaze | 4 | 224×224 | 1,846–2,238 | 88,303–88,935 | 2,919–4,403 | **20.1–30.3** |
| DM Control | 16 | 84×84 | 272–370 | 3,813–140,391 | 6,110–10,430 | **0.4–23.0** |
| LunarLander | 1 | 400×600 | 336 | 18,786 | 1,165 | **16.1** |
| highway-env | 6 | 150×600… | — | 1,520–20,389 | 660–3,205 | **2.3–7.2** |
| Fetch | 12 | 480×480 | 323–404 | 6,008–6,164 | 896–1,399 | **4.3–6.9** |
| HumanoidBench | 17 | 256×256 | 31–59 | 9,330–16,835 | 2,347–4,424 | **2.3–6.8** |
| MuJoCo locomotion | 10 | 240×240 | 329–2,591 | 4,404–10,931 | 2,390–4,834 | **1.6–4.2** |
| Meta-World | 50 | 480×480 | 8–219 | 1,039–2,554 | 655–1,066 | **1.2–3.8** |
| RoboCasa | 4 | 256×256 | 87–560 | 5,122–11,687 | 2,511–3,718 | **1.7–3.1** |
| CarRacing | 1 | 96×96 | 133 | 179 | 9,884 | **0.0** |

Copied from the `## By family` table of `examples/results/decode/ALL.md`, which is regenerated from the
per-shard JSONs by `scratch/census/merge_bench.py`. Every row is at the environment's own
`default_supersample` and its own native frame size, against one core decoding the *same* frames.

**Two corrections are folded into these numbers and both moved them down.** The published rows used to
be taken with `--ss 1` while 44 registered classes declare `default_supersample = 2` — every OGBench
and D4RL maze, 11 DM Control domains, 4 MuJoCo locomotion envs — so they described a picture four
times cheaper in rays than `se.make(name)` returns, and it is the picture every parity number is
measured on. And the ray-cast families are **host-bound**: `cpu%` runs 100–206 % of the timed loop, so
those rows are a property of the node, not of the device. `DMCBallInCupCatch` measured 3,664 f/s idle
and 271–978 contended, a 13× spread. The old blanket claim that "the GPU column reproduces across
nodes to 0.4 %" was verified on the fused 2-D rasterizer and is false here.

**One core is the generous bar.** Against eight worker processes batch-decoding JPEG-in-Lance at
256 px — the strongest disk-based path measured — a family multiplies down by roughly the worker
count, and several of the rows above cross below 1. That column is in
`examples/paper/tables/family.tex`.

**Griddly is still the only family whose OFFICIAL renderer is slower than decode** (525–1,455 f/s
against 875–7,011 f/s per core), which is why its crossover is batch 1 and its headroom is the largest
in the table. Its rows, and the JaxMARL Overcooked row, are no longer upper bounds: they are
`tilemap.blit` driven by an index derived from state against the real atlas, reproducing the official
frames bit-exactly (1,200/1,200 POV, 480–666 global, 512/512 Overcooked; max abs 0).

**RoboCasa is measured separately because its setup and its loop have different answers**
(`probe_g4_egl.py`, `probe_g4_mjwarp.py`, on real RoboCasa frames, which decode slower than a
synthetic baseline). End to end, 500 frames including the per-episode MJCF compile and render-context
build, is **0.010 cores** (MJWarp) and 0.043 (EGL); with the setup amortized away it is **5.1 cores**
for MJWarp's loop at 5.9–23.1 % interior-exact, and **0.37 cores** for MuJoCo EGL at 100 % exact. The
two ends differ by a factor of 355 and the whole of that difference is the per-episode decoder
rebuild.

One GPU replacing over a thousand cores of decoding is the thesis working. RoboCasa's rows are the
thesis failing in a specific and instructive way: **the renderer that is fast is not exact, and the
renderer that is exact is slower than reading.** The 0.010 figure is the K = 1 access pattern nobody
uses; the 5.1 and the 0.37 are what survive when the setup is amortized away, and neither of them is a
win at the parity bar. That is why RoboCasa ships in the *visual* tier, which asserts an SSIM rather
than a pixel count, and why the parity tier stays closed to it.

---

## The boundary, in two sentences

**Build it when the decoder is a fixed, family-wide constant — an atlas, a tile set, a handful of
analytic primitives, or a baked static plate — that a small per-step state indexes into; then the
frame is a gather, exactness is structural rather than fitted, and one GPU replaces tens to hundreds
of CPU cores of decoding.** **Do not build it when the decoder is itself content: regenerated per
episode from a large asset library (RoboCasa), or written per title as an emulator (Atari) — there the
state stops being compact, the batch stops being shareable, and the cost of *rebuilding the decoder*
swamps whatever the renderer saves.**

The RoboCasa re-measurement sharpened the second sentence. Its renderer is fine — batched on a GPU it
beats reading the stored dataset by 3.5× at K = 1 — and the third pass showed even that is not the
sharp version: setup comes down 7×, corpora bucket 12–2,500 episodes per kitchen, and the family
still fails, because the renderer that is fast is not the renderer that matches. The thing that makes a family a NO-GO is not that
regeneration is slow; it is that **regenerating the decoder** is, and that is a property you can test
in twenty seconds with two resets.

## Why this is a criterion and not a highlight reel

The same rule is doing the work in both directions, and each gate has a family that only it catches:

- **G1 alone** kills Atari: the frame simply is not a function of any state the env exposes, and no
  amount of engineering changes that. It flagged LunarLander and Meta-World too, and both of those
  were **overturned by measurement** — LunarLander's objection was about `render()` mutating what it
  draws and about a lossy observation, and Meta-World's was about placement living in the model rather
  than in `qpos`. Neither survived a restore ladder. G1 says "the state you were about to store is not
  sufficient"; only sometimes does it say "no state is".
- **G2 alone** flags Craftax and Procgen — both of which would look like refutations if you took the
  state object at face value, and both of which are wins on the render slice. The gate is what turns
  "the state is 6.37× bigger than the frame" from a NO-GO into a storage-format instruction.
- **G3 alone** predicts DM Control's 2-D domains fail the 2-D path without rendering a single frame.
  It is the cheapest test in the set: two resets and a comparison of `ngeom`. It still fails for
  RoboCasa, and it is no longer what decides RoboCasa — the third pass showed the per-episode setup
  it measures is reducible by 7× and that real corpora bucket 12–2,500 episodes per kitchen, and the
  verdict did not move.
- **G4 alone** is the gate that says Craftax is a *low-marginal-value* win (the JAX renderer already
  does 1.3 M f/s), and the gate that catches a family with a stable decoder and a slow renderer.
  **RoboCasa is now that family**, and it is the one case that forced the gate to be stated properly:
  regeneration must beat decode *at the fidelity the family requires*. Measured on the loop with
  setup deleted, RoboCasa's exact renderer replaces 0.37 of a core and its 5.1-core renderer draws a
  different picture. A G4 that ignores which renderer you are allowed to use would have passed it.

Note the direction of the surprises, which is the real test of whether a rule is predictive: the
families that looked hardest are not the ones that failed. OGBench's articulated humanoid reduces to
**19 analytic primitives and `nmesh = 0`**; NetHack's 1.27 MB frame is a **single gather**. RoboCasa
failed not because it is photo-realistic but because its scene graph is regenerated per episode — and
Atari, the visually simplest family in the entire set, fails hardest of all, on every gate. Visual
complexity is not the axis. **Decoder stability is.**

## Running the gates on a new family

In cost order — the cheapest test is also the most decisive:

1. **G3, ~20 s.** Reset twice. Compare the compiled scene / atlas / primitive count. If it changes
   shape, stop.
2. **G1, an afternoon.** Force two different internal states to the same candidate state; render; diff.
   Demand 0 differing pixels.
3. **G2, minutes.** Slice the state to what the renderer reads. Divide by frame bytes. Amortize any
   per-episode payload over a real episode length.
4. **G4, an hour.** Time regeneration against decode on one machine, and measure the render's share of
   an env step. Split regeneration into **setup** (compile, context, asset upload) and **loop**
   (per-frame), and amortize the setup only over episodes that genuinely share it — one end-to-end
   number cannot tell a slow renderer from a heavy per-episode rebuild, and those have different
   verdicts. Do not run dynamics in the loop: a state→frame decoder needs kinematics, not `mj_forward`
   (33× cheaper for RoboCasa), and it must render the frame the env actually shows, not the
   renderer's defaults.

Only then write a renderer. Atari was reachable from gates 1–3 alone, for a few hours of work against
the weeks a renderer costs. RoboCasa needed G4, and needed it measured twice. The rejections that did
*not* survive — LunarLander, DM Control's family, Meta-World, CarRacing — were every one of them a
gate scored from an effort estimate or a missing package rather than from a measurement, which is the
single most useful thing this table has taught: **an unmeasured gate is a gap, not a verdict.**

## Gaps

Recorded rather than guessed:

- **`backend3d.md` now exists** and adopts MJWarp's batch raytracer; DM Control and RoboCasa both hand
  off to it. Its RoboCasa fidelity entry is superseded by `robocasa.md` §7 — the "7.6 % exact, flat
  green" result was MJWarp drawing zero-alpha marker geoms opaque, and the corrected figure is
  interior mean 4.96 / 89.4 % within 8.
- **Closed: Procgen, LunarLander and DM Control were all listed here as unmeasured and all three are
  now measured.** Procgen has a throughput benchmark and 4 games ship; LunarLander is pixel-verified
  and ships; DM Control's version skew was resolved and 16 cells ship. The pattern is worth naming —
  every gap in this list that was an *absence of measurement* rather than a measurement resolved in
  favour of building, so an unmeasured gate should be read as unknown and not as a soft NO.
- **Render share of an env step** is measured for PushT (77.0 %), RoboCasa (3.3 %), Griddly
  (98.1–99.5 %, `griddly.md`), Overcooked (89.9 %, `overcooked.md`) and PixelBrax (98.8 %). The
  intermediate families are still unmeasured on this axis.
- **The ray-cast families' throughput does not travel.** `cpu%` is 100–206 % of the timed loop, so
  those rows measure the node's processor as much as the device, and a fix means moving the per-chunk
  loop off Python. Until then their ratios are an order of magnitude, not three digits.
- **G4 is stated here against one core and in the paper against eight workers**, and the two bars
  disagree about several families. Both are reported; neither is the single right answer, because
  which one applies depends on how many cores the reader's loader is allowed to burn.
- The **cores-replaced** column mixes one GPU against one CPU core by construction. It is the right
  comparison for "should I store frames or regenerate them on the node I have", and the wrong one for
  a per-watt or per-dollar argument.
- **G4 is the only gate whose verdict can change without any code changing**, since it is a ratio
  against hardware and against a decode library. A family recorded here as losing it should be
  re-measured before being dropped, not inherited.
