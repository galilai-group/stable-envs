# MiniGrid + BabyAI

Verdict: **go, and exact by construction.** A MiniGrid frame is a grid of tiles drawn
from a finite set of 492 distinct images. Precompute that set once and rendering becomes
a pure gather — no rasterisation on the GPU at all, so there is no approximation to
validate away. Measured: 21,216 frames across all 172 instantiable env ids and two tile
sizes, every one bit-identical to the official renderer, max absolute pixel difference 0.
Throughput on one H200: 6.1M agent-POV frames/s (224x224) against 2,378 frames/s for the
official CPU renderer, a 2,557x speedup, running at 1.1 TB/s — memory-bandwidth-bound,
not compute-bound.

Shipped in `stable_envs/render/tilemap.py` (atlas, recolouring, blit and visibility kernels) and
`stable_envs/envs/minigrid.py` (seven registered classes plus a `register_all()` factory for the
remaining 165 ids). Prototype and offline validation: `scratch/minigrid/` (`calibrate.py`,
`mg_gpu.py`, `mg_triton.py`, `validate.py`, `validate_all.py`, `stress.py`, `bench.py`,
`list_envs.py`). Package version: `minigrid` 3.1.0.

## 1. The minimal state

Per environment, rendering needs exactly:

| field | shape | dtype | notes |
|---|---|---|---|
| `grid.encode()` | (W, H, 3) | uint8 | (type, colour, state) per cell |
| `agent_pos` | (2,) | int | grid cell |
| `agent_dir` | () | int | 0..3 |
| `carrying` | (3,) | uint8 | encode of the held object; **POV frame only** |

plus three per-env-class constants: `agent_view_size` (7 for every registered env),
`see_through_walls`, and the render options `tile_size` / `highlight`.

This was tested, not assumed. `validate.py` takes the state out of a live env, decodes it
into a **fresh** env instance built from a different seed, and renders with the official
renderer: 6,880 round-trips over 172 env ids, 0 pixel mismatches for both the full and
POV views.

Two facts make the state this small, both confirmed against the source:

- **The full top-down frame does not depend on `carrying`.** A picked-up object is
  removed from the grid, and `get_full_render` only consults `carrying` indirectly via
  `gen_obs_grid`, which writes it into the sliced view *after* `process_vis` — so it
  affects neither the visibility mask nor the world grid.
- **`Box.contains` is invisible.** `Box.render` ignores it and `Box.encode` drops it.
  Verified at pixel level: a `Box("red")` and a `Box("red", contains=Key("blue"))` render
  identically across all 10 (agent_dir, highlight) combinations.

Caveat, and it matters for how a state dataset gets used: this is the state sufficient to
**render**. It is not sufficient to **simulate**. Stepping the env also needs
`Box.contains`, the mission string, `step_count`, and the RNG. Rendering is what this
repo replaces; dynamics are not ported here.

Storage, averaged over all 172 envs: state is 545 bytes as a raw encode, or 186 bytes at
one tile-id byte per cell (only 58 encodings are reachable, so a cell fits in a byte).
The corresponding `tile_size=32` full frame is 551,835 bytes — **1,013x larger**. For the
standard partial-obs setup, a 224x224 POV frame is 1,024x the 7x7x3 observation it is
rendered from.

## 2. How the official renderer works, and why the atlas is exact

`Grid.render` loops cells and blits `Grid.render_tile(cell, agent_dir, highlight, tile_size)`.
That function's **own cache key is `obj.encode() + (agent_dir, highlight, tile_size)`** —
upstream already asserts the tile image is a function of nothing else. Each tile is drawn
at 3x supersampling by scanline predicates (`point_in_rect` / `_circle` / `_line` /
`_triangle`), highlighted by an alpha blend if in view, then box-downsampled by 3.

`calibrate.py` checks the purity claim rather than trusting it: six pairs of objects that
share an encoding but differ in construction path or hidden fields (`Box.contains`,
`is_locked` under `is_open`, `decode()` vs constructor, `init_pos`/`cur_pos`), each
rendered with the cache cleared between calls, across 5 agent directions x 2 highlight
states. **0 mismatches.**

So the appearance set is a finite product:

| axis | size |
|---|---|
| object type | 11 |
| colour | 6 |
| state (door open/closed/locked) | 3 |
| agent_dir (4 headings + "no agent") | 5 |
| highlight | 2 |
| **dense product** | **1,980** |

Of those 1,980 slots, **492 are distinct images**; 580 are reachable from a real
`encode()` (58 reachable encodings x 5 x 2). At `tile_size=32` the dense atlas is
**6.08 MB** — small enough to sit in an H200's 50 MB L2, which is why the blit runs at
memory-bus speed.

The atlas is built by *calling the official `render_tile`* and casting exactly the way
`Grid.render` casts when it assigns the float tile into a uint8 buffer. Nothing is
reimplemented, so exactness is structural, not empirical.

One subtlety worth recording: the atlas must **not** be built via `WorldObj.decode`.
`decode` throws colour away for `Goal` and `Lava` (it always returns a green goal), while
`Goal.render` reads `self.color`. An atlas built through `decode` would be silently wrong
for a non-green goal. `calibrate.make_obj` constructs faithfully instead.

Three type indices — `unseen`(0), `empty`(1), `agent`(10) — all mean "no object" and must
share a tile image; asserted at pixel level, 0 mismatches.

## 3. Partial observation vs full view

Both are implemented and both validate exactly. Neither needs anything beyond the state
above.

**Agent-centric POV** (`get_pov_render`, V x V tiles). Three stages:

1. `Grid.slice` + `rotate_left` x (dir+1). This composite is a pure gather through an
   offset table `offs[d, vj, vi] -> (dx, dy)`. The table was extracted by tracing the real
   `slice`/`rotate_left` with tagged objects, confirmed translation-invariant, and
   cross-checked against the *independently written* coordinate formula inside
   `get_full_render` — they agree for **V in {3, 5, 7, 9, 11}**. Out-of-bounds cells
   become grey `Wall`, matching `Grid.slice`'s padding.
2. `process_vis`, the occlusion sweep. This one is genuinely sequential: the
   left-to-right pass reads cells it just wrote, and the right-to-left pass reads what the
   left-to-right pass wrote. Transcribing it to torch costs ~670 tiny kernels. The fast
   path packs the whole V x V mask into one int64 (bit k = i*V + j) and unrolls all
   2*V*(V-1) steps inside a single Triton launch; it is checked against the literal torch
   transcription on every env.
3. Blank invisible cells, drop `carrying` at the agent cell, draw the agent facing up.

**Full top-down** (`get_full_render`). Same visibility mask, scattered back to world
coordinates through the same offset table to build the highlight mask; agent triangle at
`agent_pos` with `agent_dir`. With `highlight=False` the visibility computation is
skipped entirely and the frame is a one-shot gather from `encode()`.

## 4. Coverage

178 registered ids: **76 MiniGrid + 96 BabyAI + 6 WFC**. 172 instantiate and all 172 were
validated pixel-exact.

Every one of the 172 is a stock `MiniGridEnv` with a stock `Grid`, and
`get_full_render` / `get_pov_render` / `get_frame` are all the unoverridden base-class
methods — there is no per-env render code anywhere in the package, and there are no
`WorldObj` subclasses outside `core/world_object.py`. **Nothing in the family breaks the
tile model.** No continuous positions, no per-env art. Grid sizes span 4x3 to 25x25 across
28 distinct shapes; `agent_view_size` is 7 everywhere; `see_through_walls` takes both
values.

The 6 `MiniGrid-WFC-*` ids fail to instantiate, but not for a rendering reason: the wheel
ships `wfc/patterns.py` without the `patterns/*.png` assets it loads. Their `_gen_grid`
emits an ordinary `Wall`/`Goal` grid, so they are covered by the tile model the moment the
assets are present.

Beyond what registered envs emit: across all 172 ids, rollouts only ever produced object
types {1,2,4,5,6,7,8,9} — never `Floor`, never a coloured `Wall`, never a non-green
`Goal`. `stress.py` closes that gap by sampling grids uniformly from all 58 reachable
encodings and dropping the agent anywhere, including onto walls and lava. 576 frames, all
exact.

## 5. Measured results

### Exactness

| test | frames | exact | max abs diff |
|---|---|---|---|
| 172 env ids x 3 modes, tile_size=32 | 10,320 | 10,320 | 0 |
| 172 env ids x 3 modes, tile_size=14 | 10,320 | 10,320 | 0 |
| full-alphabet stress, 6 configs x 2 modes | 576 | 576 | 0 |
| **total** | **21,216** | **21,216** | **0** |

Modes are full-view-with-highlight, full-view-no-highlight, and agent POV. `tile_size=14`
is included because rjepa renders MiniGrid at 14, where `T*3 = 42` is not 4-byte aligned
and the blit kernel drops to 2-byte words. Alongside: state-sufficiency round-trips 0/6,880
mismatches; Triton blit == torch blit on every env; Triton bitmask visibility == torch
`process_vis` on every env.

### Throughput (1x H200, batch 1024–4096)

CPU baseline is the official renderer with a warm tile cache, i.e. its best case.

| view | pixels | CPU official | GPU end-to-end | speedup |
|---|---|---|---|---|
| POV, T=32 (FourRooms) | 224x224 | 2,378 f/s | **6.08M f/s** | 2,557x |
| POV, T=32 (Empty-8x8) | 224x224 | 2,608 f/s | **6.36M f/s** | 2,439x |
| full, T=32 (FourRooms 19x19) | 608x608 | 532 f/s | **1.12M f/s** | 2,105x |
| full, T=32 (MultiRoom 25x25) | 800x800 | 317 f/s | **0.62M f/s** | 1,956x |
| POV, T=8 | 56x56 | — | **17.9M f/s** | — |
| full, T=8 (8x8 grid) | 64x64 | — | **13.6M f/s** | — |

End-to-end means state tensors in, pixels out — visibility sweep, index construction and
blit — replayed from a captured CUDA graph. At 608x608 this sustains **1,128 GB/s**, and
the blit in isolation reaches ~1.1 TB/s, so the renderer is at the memory roofline; the
only way past it is to emit fewer bytes (smaller `tile_size`).

Component breakdown, and the one optimisation that mattered:

| stage | before | after |
|---|---|---|
| index build (POV, FourRooms, B=4096) | 0.71M f/s | **9.24M f/s** |

The torch transcription of `process_vis` was ~670 kernel launches per batch and dominated
everything. The int64-bitmask Triton kernel folds it into one launch — a 13x index-path
speedup — and removing the last host-to-device constant copies made the whole pipeline
CUDA-graph capturable, worth a further 1.1–4x at small batch.

The single-pass blit kernel is 2.5–3.5x faster than `atlas[idx].permute().reshape()`,
which materialises a full-size tiled buffer and then transposes it.

## 6. GPU design

**Atlas.** `uint8[1980, T, T, 3]`, flat index
`((((highlight*5 + dir)*11 + type)*6 + colour)*3 + state)`. 6.08 MB at T=32, 1.16 MB at
T=14, 0.38 MB at T=8. Built once by `calibrate.py` from the official `render_tile`.

**Index tensor.** `int32[B, H, W]` for the full view, `int32[B, V, V]` for POV. Built by:

1. gather the agent view — `offs[dir]` lookup + world-bounds clamp, OOB to grey wall;
2. visibility — pack `see_behind()` to an int64 bitmask, one Triton scan, unpack;
3. compose — `type/colour/state` from `encode`, `dir` = 4 everywhere except the agent
   cell, `highlight` scattered from the visibility mask (invalid entries routed to a
   scratch slot so the scatter needs no host sync).

**Blit kernel.** Output `(B, H*T, W*T, 3)` uint8. One program per `(batch, output row,
column block)`; each lane computes `tile = idx[b, y/T, x/T]` and copies from
`atlas[tile, y%T]`. Addressed in the widest word that divides a tile row (`T*3` bytes):
4 for T=32, 2 for T=14. Every offset formed is a multiple of a tile row on both sides, so
alignment holds. A tile row is 96 contiguous bytes at T=32, giving coalesced reads, and
the atlas is L2-resident.

**Not built here.** GPU dynamics. The 7 discrete actions are trivial grid updates and
would port cleanly, but this work covers rendering only — states currently come from
CPU-stepped envs.

## 7. As shipped

Seven classes are registered: `MiniGridEmpty`, `MiniGridDoorKey`, `MiniGridFourRooms`,
`MiniGridLavaCrossing`, `MiniGridKeyCorridor`, `BabyAIGoToLocal` (top-down) and
`MiniGridPOVFetch` (agent-centric). `register_all()` adds the other 165 on demand; it is
not called at import because the conformance suite's reference leg renders on the CPU and
a ten-minute suite stops being run.

The conformance suite reproduces the offline result: all seven report
`exact_frac 1.0, max_abs 0`, holding the default `parity_interior_min = 1.0` with no
`parity_note` — nothing is relaxed. Full suite including PushT: **66 passed, 7 skipped**,
the skips being `test_backend_agreement`, which needs a `Scene2D` these envs deliberately
do not build.

Two shipped-path notes that differ from the prototype numbers above:

- Cells are stored as **one code per cell**, an index into the 58 reachable encodings,
  not the raw 3-channel encoding. That is not just 3x smaller — in the raw encoding the
  state channel of a non-door cell provably cannot change the picture, so those dimensions
  are dead, and the contract requires every state dimension to reach the render.
- Throughput through the class, not the prototype: **6.55M POV frames/s** at 224x224
  (987 GB/s) and 1.30M 256x256 top-down frames/s. The top-down path is slower because it
  also computes the field-of-view highlight, whose ~25 small kernels dominate at that frame
  size; the prototype's higher top-down figures were CUDA-graph replays.

Variations recolour through a **label map** rather than by editing pixels: alongside the
RGB atlas, a map records which of 22 semantic slots painted each supersampled subpixel.
Two things make the naive approach wrong and are invisible by eye — grey walls and grid
lines are both literally `(100,100,100)`, so colour matching cannot separate them; and
`highlight_img` truncates to uint8 *before* the downsample, so a highlighted tile is not
the alpha blend of the downsampled tile. The build asserts the label reconstruction
reproduces the official atlas byte for byte at the default palette, so a wrong constant
cannot survive it. Building costs ~79 s of pure-Python scanline fills and is cached to
`~/.cache/stable-envs/`.
