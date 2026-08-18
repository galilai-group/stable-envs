# Overcooked

Verdict: **landed — five layouts, exact, against JaxMARL's renderer. This is NOT the `overcooked_ai_py`
picture, and the difference is declared rather than glossed.**

Package: `jaxmarl` **0.1.0** (`jax` 0.4.38). Module: `stable_envs/envs/overcooked.py`.
`reference_id = "jaxmarl/Overcooked-V1"`. Probes: `scratch/overcooked/` (the `overcooked_ai_py`
gates), `scratch/overcooked_jax/` (the JAX ones), `scratch/griddly_pack/` (the shipped-class smoke and
bench, shared with the Griddly family).

> **Read this before quoting anything below.** Two Overcookeds exist and they draw different pictures.
> `overcooked_ai_py`'s `StateVisualizer` blits **93 named 15x15 sprite frames**; JaxMARL's
> `OvercookedVisualizer` has **zero sprite assets** and draws every tile from MiniGrid's
> `point_in_rect` / `point_in_circle` / `point_in_triangle` primitives at 3x supersampling. On the same
> `cramped_room` board they render **450x525 against 300x375** (JaxMARL draws an extra one-cell wall
> ring) with **48 distinct colours against 33**. No pixel comparison between them is meaningful, and
> **this module reproduces JaxMARL's renderer and does not cover `overcooked_ai_py` at all.**

## 1. Results

| quantity | value |
|---|---|
| layouts registered | **5** — `cramped_room`, `asymm_advantages`, `coord_ring`, `forced_coord`, `counter_circuit` |
| render parity vs `OvercookedVisualizer._render_grid` | **interior 100 %, all pixels 100 %, max abs diff 0** on all 5 |
| cell alphabet | **64** codes, ENUMERATED from jaxmarl's own constants (1 empty + 7 objects + 24 pot statuses + 32 agents) |
| atlas | **64 tiles, 196,608 B** at the default 32 px, plus a `(64, 96, 96)` label map for recolouring |
| GPU throughput, B = 8,192, one H200 | **3.56–7.90 M frames/s**, 1,685–2,935 GB/s |
| official renderer (`_render_grid`, warm tile cache) | **1,078–1,888 f/s** |
| speedup | **(x3,303–x5,713 at batch 8,192), and ahead from batch 1** |
| cores of JPEG decode replaced | **1,351–1,997** |
| state | 4 bytes a cell against 3,072 bytes a cell of frame — **768x**, or **3,072x** packed to the byte the alphabet needs |
| render at 2S, box-downsampled to S | `exact_frac` **0.981–0.987**, mean 0.13–0.18/255 |
| conformance suite | **165 passed, 30 skipped, 0 failed, 1 segment, supervisor rc 0** |

## 2. Throughput, per layout

Frame is `grid_h*32 x grid_w*32` on the PADDED grid (the layout plus jaxmarl's one-cell wall ring).
"official" is `_render_grid` with the class-level tile cache already warm — the fastest the reference
goes. "state to frame" is the same call plus writing a stored state back into a grid array, which is
what regenerating a dataset through the reference costs.

| env | frame | state | official | state->frame | decode f/s/core | ours B=1 | ours best (B=8192) | ratio | crossover | cores replaced |
|---|---|---|---|---|---|---|---|---|---|---|
| `OvercookedCrampedRoom` | 192x224 | 42 | 1,888 | 176 | 3,957 | 7,192 | **7.90 M** (2,039 GB/s) | x4,184 | **B=1** | **1,997** |
| `OvercookedAsymmAdvantages` | 224x352 | 77 | 1,078 | 60 | 2,637 | 7,496 | **3.56 M** (1,685 GB/s) | x3,303 | **B=1** | **1,351** |
| `OvercookedCoordRing` | 224x224 | 49 | 1,647 | 302 | 3,581 | 7,327 | **6.85 M** (2,061 GB/s) | x4,156 | **B=1** | **1,912** |
| `OvercookedForcedCoord` | 224x224 | 49 | 1,613 | 245 | 3,520 | 7,461 | **6.85 M** (2,061 GB/s) | x4,244 | **B=1** | **1,945** |
| `OvercookedCounterCircuit` | 224x320 | 70 | 1,195 | 73 | 3,566 | 7,419 | **6.82 M** (2,935 GB/s) | x5,713 | **B=1** | **1,914** |

The crossover is batch 1 — 7,192–7,496 f/s against 1,078–1,888 — because the reference is a Python
double loop over cells that assembles a frame one 32x32 tile at a time.

## 3. The obstacle that stopped this, and the measurement that removed it

The earlier note recorded one thing standing in the way, and it was real. `overcooked_ai_py` draws the
pot's cooking timer as **font-rendered Roboto text at a fixed 20 px, applied AFTER the x5 tile scale**,
at a horizontal offset that depends on the glyph string's rendered width — so its pot tiles are
`tile_size`-specific, cannot be composed from the 15x15 source art, and small tile sizes do not
reproduce the reference layout at all.

JaxMARL draws the same information as a **bar in normalised tile coordinates**:

```python
progress_fn = rendering.point_in_rect(0.1, 0.9 - (0.9 - 0.1) / 20 * pot_status, 0.83, 0.88)
rendering.fill_coords(img, progress_fn, COLORS["green"])
```

Confirmed by rendering rather than by reading: **24/24 pot statuses stay distinct with zero duplicates
at T in {8, 12, 20, 32, 64, 75}**, and a token scan of all four jaxmarl viz modules for
`freetype|font|SysFont|Roboto|pygame|ImageFont|putText|cv2` comes back empty. The tile composes at any
size, so the obstacle is gone — for JaxMARL's picture, which is a different picture.

**MEAL (arXiv 2506.14990) does not restore the option, and the premise that it might was wrong.** It did
not make the visualisation JIT/vmap-able; it replaced JaxMARL's vector renderer with a *pygame*
reimplementation of the `overcooked_ai_py` sprite look, **including the Roboto digit timer**
(`cooking_timer_font_size: 40`). Madrona's Overcooked ships **no renderer at all** — its 3.5 M steps/s
is the ~26-channel symbolic featurisation. CoGrid's bar sits in the identical `0.83, 0.88` band as
JaxMARL's.

## 4. The state is one index per cell, over an alphabet that is enumerated and not harvested

```
1  empty                                                            (the `obj is None` tile)
7  wall, onion, onion_pile, plate, plate_pile, goal, dish
24 pot statuses          (23 empty, 22..20 filling, 19..1 cooking, 0 done)
32 agents                (2 hat colours x 4 facings x 4 held items)
-- 64 codes
```

Enumerated from `OBJECT_INDEX_TO_VEC` and `COLOR_TO_INDEX` rather than from a rollout, because a random
policy reaches **4 of the 24 pot codes and no dish at all**: harvesting would have produced an atlas
missing every cooking-timer tile, which is the one thing this family exists to show.

Two properties of the reference make the per-cell key exactly right, and both were checked rather than
assumed:

* **the held item is a property of the CELL.** `_render_tile` resolves it as
  `agent_inv[COLOR_TO_AGENT_INDEX[obj[1]]]` — it looks the item up *by the hat colour standing in that
  cell* — so the tile is a function of (colour, facing, held item) alone;
* **`agent_dir_idx` is not read for a two-agent layout.** `_render_tile` overwrites the facing with
  `agent_dir_idx[0]` only `if len(agent_dir_idx) == 1`, which is jaxmarl's single-agent minimax path.
  Every shipped Overcooked layout has two agents, so the facing that reaches the frame is the one in
  `maze_map`, per cell.

The grid rendered is `maze_map[3:-3, 3:-3]` — `agent_view_size - 2`, the crop `animate` uses — which is
the layout plus jaxmarl's one-cell wall ring, and is why a 5x4 `cramped_room` is a 7x6 grid of 42 cells.

The structural claim that decides whether a tile atlas is the right decoder is stronger than G1 and it
holds on the `overcooked_ai_py` side too: **1,980 frames across 9 layouts were reconstructed
bit-exactly from an atlas keyed only by `(terrain_char, occupant)`** — 0 mismatches, 0
position-independence violations, 69 atlas entries learned.

## 5. Sampling has to force the cooking, and does it with the reference's own map writes

Uniform random joint actions never cook. Measured on `cramped_room`, 2,000 random states reach 4 of the
24 pot codes and no dish, so 20 of the 64 tiles would never appear in a parity comparison and the
`orange` colour knob would be reported inert because nothing in the batch paints it — which is exactly
what the first conformance run reported, on all five layouts.

The fix is not a synthetic prior. Each dialled row is a real rollout row with the substitutions
`make_overcooked_map` **itself** performs: a pot cell moved to another `pot_status`, a counter cell
holding a placed `onion`/`plate`/`dish`, an agent cell holding another item. Those are the map writes
the env makes when a chef cooks, puts something down, or picks something up. The batch is then
stratified over the atlas's *labels* — the semantic colour slots — so every declared colour knob
provably has something to paint.

## 6. Variations recolour through a label map, and `purple` is not a knob

Alongside the RGB atlas the build records, at jaxmarl's own 3x supersampled resolution, which of ten
semantic slots painted each subpixel: background, grid line, and the eight `COLORS` entries Overcooked
actually paints. Two reasons it has to be a label map rather than a recolour of the finished tile, both
invisible by eye:

* the grid lines are literally `(100, 100, 100)` and so is `COLORS['grey']`, so no colour match can
  separate a wall from the line it is drawn over;
* `downsample` averages 3x3 subpixels, so a recolour applied after it is not the recolour of the tile.

The build asserts that `palette[label]`, downsampled in jaxmarl's own axis order and dtype, reproduces
jaxmarl's atlas **byte for byte** at the default palette, so a wrong constant cannot survive it.

`purple` is deliberately absent from the knobs: it is the one `COLORS` entry no Overcooked tile paints,
and a knob that cannot move the picture is what `check_variations_do_something` exists to reject. The
build asserts that too — it fails if a declared colour never reaches a pixel, or if an undeclared one
does.

## 7. The four gates

| gate | deciding number | verdict |
|---|---|---|
| **G1 purity** | **max abs diff 0** on every probe: render-twice over 128 states; `to_dict`/`from_dict` round-trip over 800 states; 96 alphabet states x 3 construction paths; `timestep` randomised to 1e6 over 200 states; **633 cross-history pairs over 360 distinct states** (`overcooked_ai_py`), and **1,685 cross-history pairs over 2,489 distinct render states** (JaxMARL) | **PASS** |
| **G2 compression** | 4 bytes a cell against **3,072 bytes a cell** of frame (**768x**), or **3,072x** packed. `overcooked_ai_py`'s own encoding is 1/22,500 | **PASS** |
| **G3 shared decoder** | **64 tiles / 196,608 B**, enumerated over the whole code space, with **0** of the five layouts' observed tiles falling outside it. The tile cache is a class-level singleton shared by every layout and every instance | **PASS** |
| **G4 economics** | the official renderer is **0.33–0.48 cores** of the decode of its own frames, and render is **89.9 %** of an env step on the `overcooked_ai_py` side (0.382 ms render vs 0.043 ms `get_state_transition`). **1,351–1,997 cores replaced** | **PASS** |

## 8. What is deliberately not here

**A `step`.** Overcooked's transition is discrete, but jaxmarl's own `Overcooked.step` is already a
jitted, `vmap`-able JAX function on the same GPU — there is no CPU simulator to replace, and a torch
port would be a second implementation of something that is not the bottleneck. What jaxmarl does *not*
have on the GPU is its renderer, and that is the whole gap this module closes.

**`overcooked_ai_py`.** Its four gates pass (§7 records them) and it is still blocked on the same
thing: the Roboto timer means the pot tiles must be baked through pygame at each target tile size, or
the env requires `overcooked_ai_py` at render time and loses its conformance coverage on a machine
without it. A second footgun is waiting behind it — **`Recipe.cook_time` is class-level global state**
mutated by `Recipe.configure` per layout (the union across the 45 shipped layouts is
`{2,4,5,6,7,10,14,15,20,21,22,29,30,37,45,100}`), so constructing layout B silently reconfigures layout
A. Nothing in this module is affected by either, because none of it is `overcooked_ai_py`.

**`OvercookedV2`.** Its visualiser puts the same bar in the same `0.83, 0.88` band and is jitted and
`vmap`-able, so it is a different (and already GPU-native) proposition; V1 is what
`reference_id` names.

## 9. Reproducing

```bash
python scratch/griddly_pack/smoke.py overcooked     # parity + the conformance checks that bite first
python scratch/griddly_pack/bench.py overcooked     # (ratio, batch) with crossover, and cores replaced
python -m stable_envs.testing.supervisor -- tests/ -k Overcooked -v -rs
```

The venv is `~/venvs/se-oc`, built with a **clean** `python -m venv` — `--system-site-packages` breaks
jax's CUDA plugin discovery. jax runs on the CPU there deliberately: only the reference renderer and the
state pool need it, and keeping it off the device leaves the GPU to torch. `matplotlib` is a hard
requirement of the reference and not a nicety: `jaxmarl.viz.window`, which the visualiser imports at
module scope, responds to a missing matplotlib by calling `sys.exit(-1)` rather than by raising, so the
import guard probes for it explicitly instead of letting an import kill the interpreter.
