# Griddly

Verdict: **landed — seven games, exact, and the family with the largest headroom in the repo, because
Griddly's own renderer is the slow half of its own env.** Every other family here is compared against a
reference renderer that is already faster than reading the dataset it would replace. Griddly's is
**slower than the decode it would replace** (0.24–0.65 cores of JPEG decode) and the render is
**98.1–99.5 %** of an env step, so the interesting number is not a ratio but a core count.

Package: `griddly` **1.6.7**, and that is a pin rather than a floor — it is the only PyPI release
(2023-03-15), upstream's last commit is **2024-04-07**, there are 36 open issues, an unreleased
`develop` branch carrying gymnasium support, and no successor and no deprecation notice.
Module: `stable_envs/envs/griddly.py`. Probes: `scratch/griddly/`, `scratch/griddly_atlas/`,
`scratch/griddly_pack/`.

## 1. Results

| quantity | value |
|---|---|
| games registered | **7** (`sokoban`, `sokoban2`, `labyrinth`, `butterflies`, `bait`, `clusters`, `cookmepasta`) |
| render parity vs the official Vulkan renderer | **interior 100 %, all pixels 100 %, max abs diff 0**, on all 7 |
| frames the BUILD checks before it will hand over a decoder | **200 per game, 1,400 total, max abs diff 0** |
| atlas | **9–25 composed tiles per game**, 15.5–43.2 kB; 141 tiles / **0.244 MB** for all seven |
| decoder table | **5–10 codes x 16 masks** (x1 for `cookmepasta`) |
| build cost | **3–5 renders and about a second** per (game, tile size), then cached on disk |
| GPU throughput, B = 8,192, one H200 | **2.29–9.18 M frames/s**, 952–3,053 GB/s |
| official Vulkan renderer, one pinned core | **499–1,352 f/s** |
| speedup | **(x3,687–x8,015 at batch 8,192), and ahead from batch 1** |
| cores of JPEG decode replaced | **1,628–3,450** |
| state | 4 bytes a cell against 1,728 bytes a cell of frame — **432x**, or **1,728x** packed to the byte the alphabet needs |
| render at 2S, box-downsampled to S | `exact_frac` **0.22–0.70**, mean 1.1–2.9/255 — Griddly resamples its 24 px sprite art at each `TileSize`, so 2S is a finer resampling and not a supersample |
| conformance suite | **238 passed, 35 skipped, 0 failed, 1 segment, supervisor rc 0** |

The crossover is the entry worth reading twice. Batched rendering usually has to earn its keep — the
GPU path is behind at small batch and pulls ahead somewhere in the hundreds. Here it is ahead at
**batch 1**, 3,186–5,736 f/s against 499–1,352, and that follows directly from Griddly's renderer being
slower than decode rather than from anything clever on this side.

## 2. Throughput, per game

Frame is `grid_h*T x grid_w*T` at the native `TileSize = 24`. "official" is `game.observe()` with the
scene already on the board — the call a data-generation loop makes after every step. "state to frame"
additionally puts a stored state back through a level string, which is what regenerating a dataset
through the reference would cost; the two are reported apart so this repo is not credited with
Griddly's state-restore cost.

| env | frame | state | official | state->frame | decode f/s/core | ours B=1 | ours best (B=8192) | ratio | crossover | cores replaced |
|---|---|---|---|---|---|---|---|---|---|---|
| `GriddlySokoban` | 216x312 | 117 | 799 | 149 | 1,857 | 3,193 | **6.41 M** (2,590 GB/s) | x8,015 | **B=1** | **3,450** |
| `GriddlySokoban2` | 168x192 | 56 | 1,182 | 168 | 4,117 | 3,186 | **8.07 M** (1,562 GB/s) | x6,832 | **B=1** | **1,961** |
| `GriddlyLabyrinth` | 336x384 | 224 | 620 | 118 | 950 | 3,235 | **2.29 M** (1,770 GB/s) | x3,687 | **B=1** | **2,406** |
| `GriddlyButterflies` | 264x672 | 308 | 499 | 95 | 873 | 3,238 | **2.79 M** (2,970 GB/s) | x5,586 | **B=1** | **3,196** |
| `GriddlyBait` | 144x120 | 30 | 1,352 | 180 | 5,636 | 3,218 | **9.18 M** (952 GB/s) | x6,786 | **B=1** | **1,628** |
| `GriddlyClusters` | 240x312 | 130 | 864 | 150 | 2,361 | 3,195 | **5.82 M** (2,614 GB/s) | x6,734 | **B=1** | **2,465** |
| `GriddlyCookMePasta` | 264x336 | 154 | 730 | 149 | 1,699 | 5,736 | **5.73 M** (3,048 GB/s) | x7,845 | **B=1** | **3,371** |

Griddly's own renderer is **0.24–0.65 cores** of the decode it would replace. That is the sentence the
rest of this note exists to make true.

## 3. The decoder is a table, and it had to be, not merely could be

**A Griddly tile is NOT a function of what stands in its cell.** 23 of the 32 shipped games declare
`TilingMode: WALL_16` on their walls, which picks the wall image from neighbour occupancy and leaves
`RenderTileId` at 0. Over **609,103 cells across 8 games** a per-cell key collides 676–59,773 times; a
cell + 8-neighbour key collides **0 times in all 8**, and 0 over 13,520 + 25,687 stencil keys drawn from
400 generated maps. That is a genuinely different shape from MiniGrid, MiniHack, Craftax and Crafter,
all of which are per-cell gathers — and it is still a gather, one extra table lookup before the blit.

A 3x3 stencil cannot be *stored* as a table (`n_codes ** 9` entries), and one HARVESTED from rollouts
would not be complete. That second point is the one that decided the implementation: the conformance
suite proves a state dimension is live by substituting a donor code into a real state, which
manufactures neighbourhoods no rollout ever produces, and a harvested table falls off its own end
there. So the stencil is FACTORED:

```
tile = table[cell code, sum over live offsets of class(neighbour code) * m**bit]
```

and `class`, `m` and the live offsets are asked of Griddly rather than read off the yaml. Every
(centre, neighbour) pair is rendered on a lattice of 3x3 probe blocks; codes no centre can tell apart
share a class, and an offset that changes no tile is dropped. Measured:

| game | codes | classes `m` | live offsets | table | atlas |
|---|---|---|---|---|---|
| `sokoban` | 6 | 2 (wall / not) | 4 orthogonal | 6x16 | 21 tiles, 36.3 kB |
| `sokoban2` | 8 | 2 | 4 orthogonal | 8x16 | 22 tiles, 38.0 kB |
| `labyrinth` | 5 | 2 | 4 orthogonal | 5x16 | 20 tiles, 34.6 kB |
| `butterflies` | 6 | 2 | 4 orthogonal | 6x16 | 21 tiles, 36.3 kB |
| `bait` | 8 | 2 | 4 orthogonal | 8x16 | 23 tiles, 39.7 kB |
| `clusters` | 10 | 2 | 4 orthogonal | 10x16 | 25 tiles, 43.2 kB |
| `cookmepasta` | 9 | **1** | **none** | 9x1 | 9 tiles, 15.6 kB |

`cookmepasta` is the control, and it is why the neighbour term is a measurement rather than an
assumption: it declares no `TilingMode`, the probe finds one neighbour class and no live offset, and
its table collapses to a single column — a plain per-cell gather, arrived at by the same procedure that
finds four live offsets everywhere else. Out-of-grid neighbours count as class 0 (not-wall) in all
seven, also measured: the build tries every convention and keeps the one that reconstructs exactly.

**The build does not trust itself.** After enumerating, it rebuilds 200 frames of real rollouts from
the table and raises unless max abs pixel difference is **0** on every one. That check is also what
decides which games ship.

## 4. Nothing is placed through `load_state`, and that is two measured facts

`GymWrapper.load_state` looked like the obvious way to put a chosen scene on the board. It is not:

* **it returns a NEW wrapper** and leaves the receiver untouched, and each wrapper owns a Vulkan
  device. Holding them dies at ~260 with `vkGetDeviceQueue: Invalid device`. Dropping them is safe
  (600 created and dropped, zero pixel difference), but then every reference frame costs a device
  create and destroy;
* **it loses information.** Restoring an avatar with `Orientation: RIGHT` reads back `LEFT`, and `LEFT`
  reads back `RIGHT`; `UP` and `DOWN` are unchanged. That is Griddly issue **#301** — approved fix PR
  **#300, unmerged since 2024-03-27**. The frame agrees with what the env reports, so this is a state
  bug rather than a renderer bug, and it makes `load_state` unusable as a way to *ask* for a scene.

`reset(level_string=...)` has neither problem. It mutates the wrapper in place, accepts any grid size,
restores exactly (`reset(level_id=0)` after an excursion is max abs 0 against the original), and its
grammar has a `/` separator for a cell holding more than one object — measured, `A/h` reads back as
`('avatar', 'hole')`, which is what Sokoban's avatar-standing-in-a-hole cell needs. So one wrapper
builds the atlas AND serves as the parity reference, and this module creates exactly one Griddly
wrapper per (game, tile size).

## 5. What is refused, and why the refusal is the measurement

A level string cannot set an object's `Orientation`, its `RenderTileId` or its `PlayerId`, so the cell
code does not carry them either. Games whose picture depends on them are rejected by the reconstruction
check rather than by a list of opinions. Over 15 candidate games:

| game | outcome | why |
|---|---|---|
| sokoban, sokoban2, labyrinth, butterflies, bait, clusters, cookmepasta | **registered** | reconstruction max abs 0 over 200 frames |
| zelda, spider-nest | refused | an oriented avatar (and `RenderTileId` 0/1 for `has_key`), so the frame depends on a field the cell code does not carry |
| minigrid-drunkdwarf, -doggo, -eyeball, -spiders | refused | the same, for an oriented avatar and oriented monsters |
| foragers | refused | four `harvester`s that differ only by `PlayerId`, which collapse to one code |
| zenpuzzle | refused | the `walked` object occurs on the board and declares **no MapCharacter at all**, so no level string can place it |

`clusters` is worth a line because it is the trap the note used to warn about: its `MapCharacter`s are
the digits `1`/`2`/`3`, and Griddly's level parser reads a digit *following a character* as a player id.
Written as space-separated tokens they parse correctly, and the round-trip assertion is what proves it
rather than a reading of the parser.

## 6. The interpreter-exit segfault, and how it is contained

Griddly segfaults at interpreter exit on **6/6** runs, after all output is flushed. That is the one
crash shape crash isolation cannot resume past: pytest has already written its report, so no test is in
flight, and `stable_envs.testing.supervisor` would have to report a green run as a session-level
failure.

The obvious fix — stop `GymWrapper.__del__` calling `game.release()` — **does not work**, and that is
worth recording because it was the first thing tried. Four modes, six runs each, exit codes only
(`scratch/griddly_pack/p6_exit.py`):

| mode | what holds the wrapper | `__del__` | exit codes |
|---|---|---|---|
| `stock` | a script-level name | griddly's (`game.release()`) | `-11` x6 |
| `noreload` | a script-level name | overridden to do nothing | `-11` x6 |
| `stockcache` | an imported module's dict | griddly's | `-11` x6 |
| `cached` | `_open()`'s process-wide cache only | overridden to do nothing | **`0` x6** |

Two independent things are needed. Removing the finaliser is not enough — the pybind11 object's own
destructor is fatal too — and keeping the wrapper alive in a module is not enough either, because the
stock finaliser still runs; what the cache buys is that a script-level name is finalised early in
teardown and an imported module's dict is not. Both together are clean, 6/6, and that is what this
module does: every wrapper is a `_NoReleaseGymWrapper` and is only ever handed out from `_WRAPPERS`,
so nothing user-facing holds the only reference.

Same create-and-never-free rule `testing/conformance.py` states for EGL contexts, same reason — the
release is what corrupts — and the cost is a few megabytes held for the life of the process. It is
**not** a replacement for the supervisor, which is still how the suite runs; it is what stops the
supervisor being asked to tell a real crash apart from an exit-time one. End to end, under the
supervisor: **238 passed, 35 skipped, 0 failed, 1 segment, rc 0**.

## 7. The four gates, as they now stand

| gate | deciding number | verdict |
|---|---|---|
| **G1 purity** | **0 collisions, max abs diff 0** — 8,196 global frames over 13 games produced 5,459 repeats of the render key, of which **5,352 were reached from a different full state**. Re-rendering with no step between: max abs diff 0 | **PASS** |
| **G2 compression** | 4 bytes a cell against **1,728 bytes a cell** of frame (**432x**), or **1,728x** packed. Griddly's own `get_state()` is 0.029–0.042x | **PASS** |
| **G3 shared decoder** | **207 distinct sprite PNGs, 270,199 B on disk**, covering all 32 shipped games. The per-game composed alphabet is **9–25 tiles** and saturates within 2–6 episodes; `sokoban` and `labyrinth` held at 20 tiles across 200 arbitrary generated maps | **PASS** |
| **G4 economics** | the official renderer is **0.24–0.65 cores** of the decode of its own frames, and render is **98.1–99.5 %** of a step (0.99–1.60 ms render vs 0.005–0.030 ms game step). **1,628–3,450 cores replaced** | **PASS** |

## 8. What was estimated wrong, in our favour

An earlier pass costed the atlas by counting source PNGs and multiplying by four orientations. Every
number came back smaller when it was enumerated by calling Griddly instead:

| the estimate | what is true |
|---|---|
| x4 for baking view rotation | **x1.19 median, x2.17 worst** over the 15 rotating games |
| 1.79 MB with orientations baked | **0.648 MB — 385 composed tiles for all 32 games**; the 1.79 MB counted source PNGs, and what a blit path stores is composed `T x T` output tiles. The seven registered here are **141 tiles / 0.244 MB** |
| the POV index needs position | it does not: the rotated POV tile is a function of `(3x3 world stencil, avatar orientation)` with **0 collisions** over 923–2,987 keys per game |

## 9. What is deliberately not here

**The partially-observed observer.** `tilemap.blit` reproduces **1,200/1,200 rotated POV frames**
bit-exactly for `zelda_po`, `labyrinth_po` and `sokoban2_po` at 13.6 M f/s, so the renderer is not the
obstacle. The STATE is: a POV frame's edge tiles read world cells one step OUTSIDE the window (that is
what the neighbour term does), so the window is not sufficient to draw itself. Registering it would
mean either a state that is not sufficient or a state that carries the whole world and reports most of
its dimensions dead. The global observer holds the whole grid and every dimension of it is live.

**A `step`.** Griddly's transition is a per-game YAML rule engine — `Actions.Behaviours` with `mov`,
`cascade`, `remove`, `reward` and per-object variables — not a fixed decision table over a fixed
alphabet the way MiniGrid's and Sokoban's are. Porting it would be porting an interpreter, and it would
be an interpreter whose reference is 0.005–0.030 ms a step: the game step is already **0.5–1.9 %** of
what an env costs. Every millisecond in this family is in the renderer, which is what is ported.

**The 6 isometric games.** `Isometric` is a different projection; their `Sprite2D` observer is
unaffected and nothing here depends on the distinction.

## 10. Reproducing

```bash
python scratch/griddly_pack/p5_builder.py          # the decoder over all 15 candidate games
python scratch/griddly_pack/smoke.py griddly       # parity + the conformance checks that bite first
python scratch/griddly_pack/bench.py griddly       # (ratio, batch) with crossover, and cores replaced
python -m stable_envs.testing.supervisor -- tests/ -k Griddly -v -rs
```

The venv is `~/venvs/se-tiles` (`griddly==1.6.7` over the shared torch/triton). Griddly renders
headlessly through Vulkan on the GPU nodes, so `SDL_VIDEODRIVER` is not involved; every frame is checked
non-uniform (29–234 distinct byte values) before any parity number is taken from it.
