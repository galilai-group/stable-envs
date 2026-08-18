# Pixel-native families: Atari / Procgen / Craftax / NetHack

Verdict, one line each:

- **MiniHack / NLE — go, exact by construction.** The frame is `atlas[glyph2tile[glyphs]]`. Rebuilt from
  the 3318-byte glyph grid, bit-identical to the official `pixel` observation on 40/40 steps, max
  absolute difference 0. The frame is 1.27 MB, so the state is 384x smaller. The kernel is one gather:
  a five-line torch prototype hits 247k frames/s on one H200 against 178 frames/s for the official
  renderer, a 1391x speedup, bit-exact on every validation frame.
- **Craftax — go.** Already a pure JAX function of `EnvState` over a 16x16 texture atlas, and exactly
  reproducible: `render(state) * float32(1/255)` equals `get_obs(state)` on 100% of steps for both
  variants. Caveat with teeth: the full `EnvState` is 6.4x *bigger* than the frame, so the win requires
  storing the render window (0.12x) or the seed and actions, not the state as shipped.
- **Procgen — go for `maze` / `heist` / `miner` / `chaser`, bit-exact.** This entry used to read "conditional",
  on the grounds that the renderer was Qt5 `QPainter` behind 10.5 kLOC of per-game C++ and that the
  Python API "hands out an opaque blob". **Both grounds were wrong and section 2 now says how.** The
  blob is `Game::serialize`, a fixed-order stream of 4-byte fields that re-serialises byte for byte
  once transcribed; and Qt's raster rules did not have to be re-derived, because `resource_root` is a
  procgen option, so the shipped binary can be pointed at an asset tree where every sprite pixel
  encodes its own coordinates and reports exactly which source pixel each output pixel sampled.
  Three rules came back from that, all measured: the 16.16 nearest-neighbour walk (4096/4096 px), the
  rotated blit's pixel-centre coverage (539/539, 0 false positives) and its half-point-free sampling
  (539/539 against 28.9% for the plausible variant). **240 frames per game, 100.0000% of pixels
  exact, max |diff| 0**, and 1.69M frames/s on maze at batch 4096 — with a crossover at batch 128,
  below which procgen's own renderer is faster and this one should not be used. **Two more games were
  then built on the same machinery, and they part company** (§2.9): `chaser` ships bit-exact, and
  `bigfish` renders bit-exactly and does NOT ship, because a spawner cast cannot be given a
  fixed-width state that is both lossless and free of dead dimensions. Neither needed a new raster
  rule. That distinction, not draw complexity, is what decides the remaining 11.
- **Atari / ALE — no-go.** The 128-byte RAM does not determine the frame: forcing two emulator states to
  identical RAM gives different next frames on 13.75% of state pairs across 30 ROMs, 29.2% when the RAM
  is written into a freshly reset console, and on every single pair for five of those ROMs
  (`pong`, `asterix`, `qbert`, `amidar`, `chopper_command`). The minimal sufficient state is ALE's 14.4 kB system
  state, only 2.3x smaller than the indexed screen, and converting it to pixels means running a
  cycle-accurate 6507 and TIA — per ROM. No atlas, no index grid, no shared renderer.

Method follows `experiments/bench/pusht_gpu/{calibrate,validate}.py`: extract from the official renderer
empirically, then diff pixels. Every number below comes from running the shipped binary under SLURM;
nothing is argued from what the source "should" do. Probes and raw outputs in
`scratch/pixel_native/` and, for the Procgen renderer, `scratch/procgen/`; the CuLE / envpool replay
measurements are in `scratch/cule/`. Versions: `ale-py` 0.12.0,
`procgen` 0.10.7 (built from `openai/procgen` @ 37b521d, which is that tag, and linking Qt **5.9.7** —
the raster rules below are that version's), `craftax` 1.6.1, `nle` 1.3.0 + `minihack` 1.0.2.

| family | frame | state that determines it | state / frame | frame = f(state)? |
| --- | --- | --- | --- | --- |
| MiniHack / NLE | 336x1264x3 = 1.27 MB | glyph grid 21x79 int16 = 3318 B | 0.0026x | yes, bit-exact 40/40 |
| Craftax Classic | 63x63x3 = 11.9 kB | render window ~976 B (full `EnvState` 21 kB) | 0.08x | yes, exact |
| Craftax full | 130x110x3 = 42.9 kB | render window ~5.3 kB (full `EnvState` 273 kB) | 0.12x | yes, exact |
| Procgen | 64x64x3 = 12.3 kB | grid + entities + camera, 1.2-17 kB | 0.10-1.40x | yes, round-trip exact 16/16 |
| Procgen `maze`/`heist`/`miner` | 64x64x3 = 12.3 kB | 629 / 196 / 407 float32 | 0.20 / 0.06 / 0.13x | yes, **100% of pixels, 240 frames/game** |
| Procgen `chaser` | 64x64x3 = 12.3 kB | 199 float32 | 0.06x | yes, **100% of pixels, 240 frames** |
| Procgen `bigfish` (not shipped, §2.9) | 64x64x3 = 12.3 kB | 230 float32 | 0.07x | yes, **100% of pixels, 240 frames** — but the slot cap is unsatisfiable |
| Atari / ALE | 210x160 = 33.6 kB indexed | not the 128 B RAM; full system state 14.4 kB | 0.43x | no |

## 1. Atari / ALE: no-go

`scratch/pixel_native/probe_ale.py`, `probe_ale_ext.py`.

The tempting state is the 128-byte console RAM, which ale-py exposes and lets you overwrite byte by
byte. Settling it needs two emulator states that agree on all 128 RAM bytes and disagree on the next
frame: restore full system state `S_t`, write `RAM = ram_t`, emulate one frame, read the screen; then
the same starting from a different full state `S_u`. Both branches enter the frame with byte-identical
RAM (verified by reading RAM back after the write), so any pixel disagreement is one RAM cannot explain.

Over 30 ROMs x 8 state pairs:

| measurement | value |
| --- | --- |
| same frame after 1 step from identical RAM | 86.25% of pairs |
| same frame when the RAM is written into a freshly reset console | 70.83% |
| same RAM after 1 step | 89.58% |
| frames identical before divergence (horizon 8) | 6.55 |

The failures are not noise, not confined to obscure ROMs, and not a handful of outliers: **11 of the 30
ROMs fail on at least one pair.** The worst:

| ROM | ROM bytes | same frame after 1 step | same frame from a fresh console | frames before divergence |
| --- | --- | --- | --- | --- |
| `berzerk` | 4096 | 25.0% | 12.5% | 2.00 / 8 |
| `gravitar` | 8192 | 37.5% | 37.5% | 3.00 / 8 |
| `alien` | 4096 | 50.0% | 100.0% | 4.00 / 8 |
| `bank_heist` | 4096 | 50.0% | 37.5% | 4.00 / 8 |
| `fishing_derby` | 2048 | 50.0% | 100.0% | 4.00 / 8 |
| `chopper_command` | 4096 | 62.5% | 0.0% | 5.00 / 8 |
| `hero` | 8192 | 62.5% | 100.0% | 5.00 / 8 |
| `breakout` | 2048 | 87.5% | 75.0% | 2.62 / 8 |

From a fresh console, `chopper_command`, `pong`, `asterix`, `qbert` and `amidar` disagree on **every
single pair**. And `pong` diverges after 1.88 frames on average even when RAM is forced equal, so RAM
does not pin the dynamics either.

The 128 bytes are the RIOT's RAM only. The rest of the machine — 6507 registers, the RIOT timer, the
TIA's write-only registers and object counters, the cartridge bank register and any on-cartridge RAM —
sits outside it. These measurements do not attribute the failures to one of those in particular; what
they establish is that something outside the 128 bytes changes the picture, on 30 ROMs, repeatedly.

The next candidate state is what ALE itself serializes: `cloneSystemState()` = 14439 B,
`cloneState()` = 7717 B, against a 33600-byte indexed screen. A factor of 2.3, not a factor of 100. And
the screen is not in that state — restoring a system state and calling `getScreen()` reproduces the
captured frame on 0 of 10 ROMs while reproducing the captured RAM on 10 of 10, so the restore works and
the framebuffer simply is not part of the state. The 2600 has no framebuffer. The picture exists only after the
6507 has been run for a frame, with the TIA racing the beam and the program rewriting registers
mid-scanline.

That is why this family fails the thesis rather than merely being expensive. The "renderer" for Atari is
a cycle-accurate CPU+TIA emulator, it is a different program for each of the 100+ ROMs, and there is no
sprite atlas or index grid to key it on.

There is also no problem to solve. Atari rollouts are fully determined by (ROM, seed, action sequence),
which is 1 byte per step, and the emulator regenerates frames at 6.6k-54k fps on a single core
(measured, `emu_fps_single_core` in `ale_state.json`). Anyone storing Atari frames can already stop
without a renderer.

### 1.1 The replay route: CuLE built, and it fails both halves independently

`scratch/cule/` (`parity_cule.json`, `parity_cule_long.json`, `cule_bench.json`,
`envpool_bench.json`, `atari_storage.json`, `screen_dims.json`).

The paragraph above used to end by naming CuLE and moving on, which left the interesting question
unasked. **The state -> frame thesis failing says nothing about whether Atari can be *served*.** A
rollout is (ROM, seed, actions) = 1 byte per step and replaying the emulator regenerates the frames
exactly; if the emulator itself runs on the GPU, the storage benefit arrives by the replay route
instead. So CuLE was built and measured rather than cited.

**It builds, on a recipe worth recording.** Upstream `NVlabs/cule` is dormant (last commit
**2022-11-17**, README pinned to CUDA 10.0 / torch 1.2 / py3.6) and does not compile against a modern
toolchain; the frequently-cited `andyljones/cule` fork **does not exist**. Two live forks do:
`MehrdadMoghimi/cule` (pushed **2026-07-21**; `atari_py`->`ale-py`, `gym`->`gymnasium`, 14
construction crashes fixed, a warp-cooperative render kernel, and a ROM-identity bug where **76 of
108 ale-py ROM dumps had md5s missing from `rom_game_map` and silently ran with bowling's decoder**)
and `chengscott/cule` (pushed **2026-07-31**, CUDA 13 / gcc 15 / py3.14 toolchain fixes). Neither
alone works here — chengscott's still imports `atari_py`, which has no py3.12 wheel. Mehrdad's tree
plus chengscott's three CUDA-13 fixes builds in **11 min 30 s** on an H200 (sm_90, gcc 13.3, C++17, a
CUDA-10-era Thrust 1.9 tree appended for the headers CCCL 3.x removed).

**Its frames are not ALE's.** 24 ROMs x 200 steps, frameskip 1, sticky actions off, identical action
sequences after mapping CuLE's *bitmask* action encoding to ALE's ordinals (the action sets then agree
24/24):

| measurement | value |
| --- | --- |
| ROMs 100% frame-exact at 200 steps | **4 of 22** (`breakout`, `freeway`, `kangaroo`, `hero`) |
| ROMs 100% frame-exact at 2,000 steps | **0 of 7** |
| mean fraction of exactly-matching frames | **0.320** at 200 steps, **0.140** at 2,000 |
| mean fraction of exactly-matching pixels | 0.9771 |
| max abs diff | **252** |
| ROMs whose frame is not even the same SHAPE | 2 (`amidar`, `bank_heist`: ALE renders **250x160**, CuLE **210x160**) |

`breakout` falls from 1.000 to **0.124** between 200 and 2,000 steps, so the exact ROMs are exact only
briefly. The gap is the emulator and not the hardware, and that is measured too: **CuLE's own GPU and
CPU backends agree with each other on 0.971 of frames at 200 steps and 0.997 at 2,000**, while
agreeing with Stella on 0.320. Under an all-NOOP rollout with a +/-30-frame shift search, 12 of 22
ROMs reach 1.0 at *some* shift (7 at shift 0), so roughly half the mismatch is a boot-phase offset —
and the other 10 never align at any shift (`qbert` 0.005, `chopper_command` 0.0, `gravitar` 0.0,
`fishing_derby` 0.0). CuLE's 128-byte RAM matches ALE's on **0.091** of steps.

**And there is nothing to buy even if the frames matched.** On one core of the same node, `ale-py`
replays at **5,011–6,973 f/s** against **2,912–3,291 f/s** of PNG-RGB decode (**1.52–2.40×**). The honest comparison is against **indexed** PNG, because this document's own table records the ALE frame as 33.6 kB *indexed*: decoded that way the baseline runs at **5,024–5,295 f/s** and replay is **0.96–1.39×** it — slower than reading, for `breakout`. So regeneration and reading are the same order here and the G4 verdict is *moot*, not a win. CuLE needs 64 parallel environments
just to reach that single core:

| envs | `pong` | `breakout` | `space_invaders` |
| --- | --- | --- | --- |
| 1 | 265 (0.03x) | 204 (0.03x) | 268 (0.03x) |
| 8 | 2,053 (0.22x) | 1,629 (0.24x) | 2,028 (0.25x) |
| **64** | 16,124 (**1.73x**) | 12,900 (**1.87x**) | 15,876 (**1.97x**) |
| 1024 | 213,491 (22.9x) | 176,716 (25.7x) | 208,565 (25.9x) |
| 4096 | **295,227 (31.6x)** | **213,931 (31.1x)** | **276,183 (34.2x)** |

**The crossover is 64 environments in every configuration measured**, and below it a whole H200 is
30-50x slower than one CPU core. CuLE's own CPU backend never crosses at all (0.10-0.82x up to 256).

**envpool does not change the storage argument, and its whole API says why.** It is alive again
(0.8.4 in 2023, then **0.9.1 on 2026-03-24** and **1.2.5 on 2026-05-20**, with cp311-cp314 wheels —
it installs cleanly on 3.12, so the wheel worry was unfounded). On 8 cores it peaks at **3.53-3.85x**
one core at n=256, crossover at **n=2**, and it overtakes `gymnasium.vector` async at n=16 and is
2.45x it at n=256 — real, and entirely a scheduling win. Its public surface is
`reset/step/send/recv/seed/render/config`: **no state clone, restore or serialise anywhere in the
package.** It makes replay cheaper; it does not make the frame a function of a small exposed state.

**Verdict: the replay route does not admit Atari, and it fails on both halves separately.** On
fidelity, CuLE clears this repo's `parity_interior_min = 1.0` on 4 of 22 ROMs at 200 frames and 0 of
7 at 2,000, with two ROMs whose frames are different shapes. On economics, there is no gap to close,
because single-core `ale-py` already beats decoding. The only contract Atari could enter under is an
explicit lossy-replay one — "regenerated with a different emulator, not the frames ALE produces" —
which is the contract every other family here is forbidden from taking.

## 2. Procgen: `maze`, `heist`, `miner` and `chaser` are bit-exact

`scratch/pixel_native/probe_procgen.py`, `decompose_procgen_state.py`, `probe_procgen_levelseed.py`,
`dbg_procgen_layout.py` (the state-side evidence, sections 2.1-2.3);
`scratch/procgen/state_io.py`, `probe0{1..9}*.py` (the renderer, sections 2.4-2.7);
`stable_envs/envs/procgen.py` (what landed).

**This section used to open with a blocker "stated first so nobody re-derives it".** The blocker was
that the renderer is Qt5 `QPainter`, that matching it means reproducing the raster engine's fixed-point
stepping "on top of 10.5 kLOC of per-game C++ draw rules that the Python API does not expose in
structured form", and that this is "a calibration project of the PushT kind, times sixteen". Three of
the three claims in that sentence turned out to be false, and the failure mode is worth naming because
it is not specialised to procgen:

| the claim | what it cost to check | what is true |
|---|---|---|
| the Python API hands out an opaque blob | one afternoon transcribing `serialize` | it is a fixed-order stream of 4-byte ints and floats; the parser re-serialises **byte for byte, 24/24 states in each of 16 games** |
| the draw rules are 10.5 kLOC of C++ | reading three files (`maze.cpp` 138 lines, `heist.cpp` 226, `miner.cpp` 326) plus the shared base | 10.5 kLOC is the whole `src/` tree, all 16 games and their physics; the three scoped games contribute 690 lines and override almost nothing |
| Qt's raster engine has to be re-derived | `resource_root` is a procgen **option** | point the shipped binary at coordinate-coded assets and it tells you which source pixel it sampled, for every output pixel |

The last row is the general lesson. The renderer was treated as a thing to reimplement from source
when it was available as a thing to *interrogate*: replace every sprite with an image whose pixel
`(x, y)` holds `(x, y)` and an id, render a real state, and the frame is a complete readout of the
sampling rule. Every constant below came back from that, and none of it was read off Qt.

`docs/research/carracing.md` records the same pattern from the other direction: four blockers filed
against CarRacing, all four falsified on re-measurement.

Sections 2.1-2.3 are the state-side evidence (unchanged, and it was always right). 2.4-2.7 are the
renderer.

### 2.1 The frame is a function of the state

`get_state()` round-trips exactly: restoring a state blob into a *fresh* env constructed with a
different `rand_seed` reproduces the observation bit for bit, on 8 sampled steps, in all 16 games. So
there is no hidden renderer state.

### 2.2 The blob is bigger than the frame; the render state is not

The raw blob is 22-37 kB, 1.8-3.0x the 12288-byte observation. That reads like a refutation and is not
one. `RandGen::serialize` dumps each `std::mt19937` with `ostream << stdgen`, writing the 624-word
Mersenne state as decimal *text*, and the game does that three times (`level_seed_rand_gen`,
`rand_gen`, `asset_rand_gen`). None of it reaches `game_draw`.

Locating the three dumps and the tile grid inside the blob — the entity stride is calibrated from the
observed gaps rather than assumed, and comes out at 124 B = 31 words, matching `Entity::serialize` for
this version — gives:

```
game         total    rng-text        grid          entities      render-state   vs frame
bigfish      22388   20129 (90%)   20x20  1612 B     2-4  (496 B)      2152 B      0.18x
bossfight    26682   20097 (75%)   20x20  1612 B     6-92(4712 B)      6368 B      0.52x
caveflyer    27414   20113 (73%)   40x40  6412 B     6-9  (744 B)      7200 B      0.59x
chaser       23103   20129 (87%)   13x13   688 B     8    (992 B)      1724 B      0.14x
climber      26299   20128 (77%)   20x64  5132 B     7    (868 B)      6044 B      0.49x
coinrun      37439   20128 (54%)   64x64 16396 B     6    (744 B)     17184 B      1.40x
dodgeball    23516   20119 (86%)   20x20  1612 B    11-14(1612 B)      3268 B      0.27x
fruitbot     33644   20120 (60%)   20x60  4812 B    67-70(8556 B)     13412 B      1.09x
heist        22470   20129 (90%)   13x13   688 B    12   (1488 B)      2220 B      0.18x
jumper       28431   20113 (71%)   40x40  6412 B     5-14(1736 B)      8192 B      0.67x
leaper       23187   20113 (87%)   15x15   912 B    15-20(1984 B)      2940 B      0.24x
maze         23037   20129 (87%)   25x25  2512 B     2    (248 B)      2804 B      0.23x
miner        22249   20120 (90%)   20x20  1612 B     3    (372 B)      2028 B      0.17x
ninja        37441   20128 (54%)   64x64 16396 B     3-7  (744 B)     17184 B      1.40x
plunder      23020   20129 (87%)   20x20  1612 B     6-9  (992 B)      2648 B      0.22x
starpilot    29126   20129 (69%)   none              2-9 (1116 B)      1172 B      0.10x
```

Entity counts are identified only up to an additive constant — the gap equation fixes `C mod E`, so a
game's true count may be 1-2 higher than shown, up to 248 B. It moves nothing: entities are the small
term everywhere except `fruitbot`.

Median render-relevant state is 0.25x the frame. Only `coinrun`, `ninja` and `fruitbot` reach parity,
and those are the games with a 64x64 or 20x60 tile grid. Two further measurements collapse even those:

- the grid is constant within a level, so the *per-step* change in the entire blob is 13-95 bytes
  (median per game) — entities and camera only;
- the grid is a function of the level seed. Two envs built with different `rand_seed` but the same
  `start_level` produce byte-identical grids in 15/16 games (`starpilot` has no grid at all) and
  byte-identical frames in 16/16, over 5 level seeds.

So a trajectory costs a 4-byte level seed plus roughly 50 bytes per step. That is the thesis working.

### 2.3 The atlas is finite and the observation path has no antialiasing

The wheel ships 1288 PNGs (25 MB); `resources.cpp` references 779 sprite paths and 83 background
entries (78 distinct files) as string literals. An entity's sprite is
`basic_assets[image_type + image_theme * MAX_ASSETS]` with `MAX_ASSETS = 100`,
`MAX_IMAGE_THEMES = 10`, so at most 1000 atlas slots per game; with the default
`use_generated_assets = false` the slot resolves to `asset_for_type(type)[theme]`, a per-game static
list indexed by two serialized integers, with no run-time RNG in the choice. The procedural-texture path
is not even serializable — `BasicAbstractGame::serialize` asserts `!options.use_generated_assets`.

`Entity` carries exactly what a sprite draw needs: `x, y, rx, ry` (centre and half-extents), `rotation`,
`is_reflected`, `alpha`, `image_type`, `image_theme`, `render_z`. Grid cells with
`image_type >= USE_ASSET_THRESHOLD (100)` are drawn as a solid `fillRect` in `color_for_type`.

One piece of luck: the 64x64 observation is produced by `render_to_buf(..., antialias=false)`, so Qt
uses nearest-neighbour pixmap transform and hard-edged fills. Only the 512x512 human render turns
antialiasing on. A GPU rasterizer has to match nearest-neighbour sampling, not Qt's antialiased coverage
arithmetic.

### 2.4 The blob is a documented stream, and parsing it makes procgen a programmable probe

`scratch/procgen/state_io.py`, `probe01_parse.py`.

`get_state()` is `Game::serialize` followed by `BasicAbstractGame::serialize`, `Entity::serialize`,
`Grid::serialize` and a per-game tail — every field a 4-byte little-endian int or float, strings as
`int length` plus bytes, and `vecgame.cpp` appending one `END_OF_BUFFER` sentinel. Transcribing that
order gives a parser whose output re-serialises **byte for byte on 24/24 sampled states in each of the
three games** (and in all 16, with the per-game tail kept verbatim), so a schema error cannot pass
silently.

Two things follow, and the second is what actually unlocked the port:

- the render-relevant fields are now named rather than inferred. `view_dim` for heist reads
  **12.999999**, not 13, and `x_off` reads **2.3475061e-06**, not 0 — because `prepare_for_drawing`
  computes `64/visibility` in C `float`. Anyone doing that arithmetic in float64 puts a rect edge a
  whole pixel out and will look for the bug in the rasterizer.
- `set_state` takes a blob back, and `vecgame.cpp` calls `observe()` immediately after `deserialize`.
  So any state can be *written*, and the shipped renderer will draw it. Every measurement in 2.5-2.7
  is a crafted state driven through the official binary, and `reference_render` in the shipped module
  places the official env the same way.

### 2.5 Coordinate-coded assets: reading the raster rules off instead of deriving them

`scratch/procgen/probe02_coded_assets.py`, `probe03_fit_blit.py`, `probe06_rotation.py`,
`probe07_fit_rotation.py`.

`resource_root` is a procgen option. Build an asset tree where every sprite pixel holds
`(R, G, B) = (x, y, sprite_id)` and every background holds its own coordinates, point procgen at it,
and each of the 4096 output pixels reports which sprite painted it and which source pixel it read.
That converts "reproduce Qt's raster engine" into "check a closed form against a complete readout".

**Axis-aligned blits** — `qt_scale_image`'s 16.16 nearest-neighbour walk:

```
tr    = QRectF::toRect() intersected with the device rect
ix    = int(65536 * srcW / dstW)
basex = uint32(ceil((x0 + 0.5 - dstLeft) * srcW/dstW * 65536) - 1)      # x0 = clipped left
sample(i) = (basex + i*ix) >> 16                                       # in uint32
```

with `QRectF::toRect()` = `QRect(qRound(x), qRound(y), qRound(x+w) - qRound(x), ...)`. Checked by
rebuilding each coded frame from the parsed state alone: **4096/4096 pixels on 7 scenes across the
three games**, covering grid cells, entities at sub-pixel float positions, mirrored sprites
(`QImage::mirrored`), the 500x500 and 512x512 topdown backgrounds and miner's non-square platform
backgrounds with their `bg_pct_x` offset.

**Rotated blits** — heist's agent calls `Entity::face_direction`, so `draw_image` takes the
`translate/rotate/drawImage` branch instead. Two independent questions, and the coded frames answer
them separately over 40 rotations:

| question | rule | measured |
|---|---|---|
| which pixels are written | device pixel **centre** inside the rotated quad | 539/539 written, **0** false positives |
| which source pixel is read | `fetchTransformed`'s fixed point on the inverted matrix, `px = int((m21(y+.5) + m11(x+.5) + dx)*65536) >> 16` | 539/539 |

The last one is the entry worth keeping. The plausible variant — the `- (1<<15)` half-point bias that
appears in `blend_transformed_argb` — scores **28.9%**. Reading either function and picking the
familiar one would have produced a renderer that is right on 3 pixels in 10 and looks approximately
correct.

**Colour** — `convertToFormat(ARGB32_Premultiplied)` is `qPremultiply`'s integer form, and the
compositor is `d = s + BYTE_MUL(d, 255 - sa)` with
`BYTE_MUL(x,a) = ((x*a + ((x*a)>>8) + 0x80) >> 8) & 0xff`. `BYTE_MUL(d, 255) == d` exactly for every
`d`, which is why a coverage mask can ride in the alpha and the GPU renderer needs no `where` over the
frame.

### 2.6 Parity

`scratch/procgen/probe08_parity.py` (numpy reference), `probe10_env_parity.py` (the shipped module,
driven through `set_state`). 20 levels x 12 steps per game, `distribution_mode="hard"`, backgrounds on:

| game | frames | pixels exact | max abs diff |
|---|---|---|---|
| `maze` | 240 | **100.0000%** | 0 |
| `heist` | 240 | **100.0000%** | 0 |
| `miner` | 240 | **100.0000%** | 0 |
| `chaser` | 240 | **100.0000%** | 0 |
| `bigfish` (measured, not shipped — §2.9) | 240 | **100.0000%** | 0 |

Three facts the renderer's *shape* depends on, each measured rather than assumed
(`probe09_facts.py`): `RENDER_EPS = 0.02` widens a cell by 0.04 of a cell, so a device column or row
is covered by **at most 2** grid cells at every size from 64 to 512 — the grid is four composited
layers, not a loop over 625 cells; heist's z=0 entities **never** share a device pixel (reversing
their order changed 0 of 300 frames), so a fixed slot order is safe; and all 9 topdown backgrounds are
square, so `offset_x = bg_pct_x * (bg_ar - world_ar)` is identically 0 and `bg_pct_x` is genuinely not
part of maze's or heist's render state, while 52 of miner's 62 backgrounds are not square and it is
part of miner's.

### 2.7 Throughput, and the compositor lesson

The first working GPU renderer ran at **59k frames/s** on maze at batch 4096 — 0.73 GB/s of frame
output, about **100x off the roofline** for a 12.3 kB frame, and flat in batch size, which is the
signature of overhead rather than work. Two causes, both worth carrying forward:

- one SourceOver blend expressed in torch is nine elementwise kernels over a `(B,S,S,3)` int32 canvas
  plus a four-channel gather. Fusing the gather, the mask and the blend into one Triton kernel that
  reads the canvas once and writes it once took maze to **376k f/s**;
- `draw_background` looped in Python over the distinct background indices in the batch — 9 groups for
  maze, up to 62 for miner — which cost about **10 ms per call regardless of batch size**. The row map
  depends on nothing but the background's height, so baking every background pre-resampled to S rows
  turns the whole layer into one gather.

And the number that a peak ratio would have hidden: **the crossover.** procgen's own renderer is C++
drawing 12.3 kB frames and it is genuinely fast, so this one wins only once the batch amortises a
chain of serial composites. Both sides swept, 8 threads for the reference
(`scratch/procgen/probe12_crossover.py`):

| batch | `maze` | `heist` | `miner` | `chaser` | `bigfish` |
|---|---|---|---|---|---|
| 1 | 482 (0.01x) | 91 (0.00x) | 312 (0.00x) | 92 (0.00x) | 26 (0.00x) |
| 128 | 60,619 (**1.00x**) | 11,491 (0.11x) | 40,757 (0.60x) | 11,515 (0.08x) | 3,237 (0.01x) |
| 256 | 120,608 (2.00x) | 23,026 (0.22x) | 81,119 (**1.19x**) | 23,022 (0.17x) | 6,469 (0.03x) |
| 1024 | 476,390 (7.89x) | 91,305 (0.88x) | 330,954 (4.86x) | 91,487 (0.67x) | 25,665 (0.12x) |
| 2048 | 928,461 (15.38x) | 175,736 (**1.70x**) | 647,174 (9.51x) | 182,264 (**1.33x**) | 51,504 (0.23x) |
| 4096 | **1,688,605 (27.97x)** | **317,259 (3.06x)** | **1,213,165 (17.83x)** | **359,105 (2.63x)** | **102,555 (0.47x)** |

Ratios are against each game's official **peak over `num`** — `maze` 60,374 at `num=64`, `heist`
103,661 at `num=256`, `miner` 68,056 at `num=256`, `chaser` 136,610 at `num=256`, `bigfish` 219,795
at `num=256`. An earlier draft of this page quoted the
reference at `num=64` for all three, which understates heist's by 10% and flatters the ratio; the
sweep is the honest baseline. The official figure is step+render either way, because procgen exposes
no render-only entry point (`Game::observe` is called from `Game::step`, and `gym3`'s `observe()`
only copies buffers).

**The crossover is batch 128 for `maze`, 256 for `miner`, 2048 for `heist` and 2048 for `chaser`**,
and below it this renderer is slower than the thing it reproduces — by two orders of magnitude at
batch 1. Anybody rendering a handful of frames at a time should call procgen. heist is the outlier
for a structural reason: 12 entity slots and a rotated-blit path give it roughly three times maze's
serial chain over the same 4096 pixels, and its ~11 ms per-call floor is what the batch has to pay
off. chaser's 10 slots and its extra fill layer put it in the same place.

**`bigfish` never crosses over at all** — 0.47x at batch 4096, and the trend is flat in the ratio, so
no reachable batch fixes it. That is the same structure taken one step further: 33 slots is 33 serial
composites over the same 4,096 pixels, against a reference that is *faster* here than on any other
game in the family (219,795 f/s, because bigfish has no grid loop to run). The number is worth
keeping next to the crossover column, because it makes the rule explicit: **what this renderer sells
is amortising a fixed number of full-frame passes over a batch, and a game whose entity count is a
spawner rather than a constant buys more passes than the batch can pay for.**

This is the same shape `docs/research/carracing.md` reports, where the chain is ~150 serial
alpha-blends over 800,000 pixels and the crossover never arrives (1.01x at batch 1024). Serial
blending against a live canvas is the structure to check for before quoting a speedup: the peak
ratio is real, and on its own it is not the whole claim.

### 2.8 Scope

`distribution_mode="hard"`, default options. All four shipped games hold
`options.center_agent` false in hard mode — maze, heist and miner through
`(distribution_mode == MemoryMode)`, chaser by setting it outright — so the camera is a fixed
full-world view and the scrolling-camera path is not implemented; memory mode is out of scope.

For the remaining 12, both raster rules belong to `BasicAbstractGame` and are shared by all 16, so
what each game adds is its own `asset_for_type`, `image_for_type`, `theme_for_grid_obj`,
`get_adjusted_image_rect` and `get_tile_aspect_ratio` — plus, for the six scrolling games
(`caveflyer`, `climber`, `coinrun`, `fruitbot`, `jumper`, `ninja`), `choose_center` and the
`center_agent` branch of `draw_foreground`. **None of that needs another measurement campaign of the
kind §2.5 describes.** This entry used to stop there and call the rest effort rather than
feasibility; §2.9 is the correction. What decides a game is whether its entity cast is a constant of
the game or a spawner, because only the first admits a fixed-width state that is both lossless and
free of dead dimensions — which rules out `bigfish`, `bossfight`, `dodgeball`, `plunder` and
`starpilot` on the contract rather than on cost, and leaves `leaper` as the one fixed-camera game
still worth measuring.

### 2.9 Two more games, and the line that actually decides the remaining eleven

`scratch/procgen/probe{13..20}*.py`, `scratch/procgen/bigfish_env.py`.

§2.8 used to say the other 13 games were "effort and not feasibility". Two of them were then done,
and that sentence turns out to be half right in a way worth stating precisely: the *renderer* is
effort, and the *state contract* is not always reachable at all.

**The seven games with a fixed camera.** `game_reset` sets `center_agent = false` outright in
`bigfish`, `bossfight`, `chaser`, `leaper`, `plunder` and `starpilot`, and to
`distribution_mode == MemoryMode` (so false in hard mode) in `dodgeball` as well as in maze / heist /
miner. Those seven are the ones the shipped camera already covers; `caveflyer`, `climber`,
`coinrun`, `fruitbot`, `jumper` and `ninja` scroll and still need `choose_center`.

**`chaser` ships, bit-exact, and cost three new measurements rather than a new raster rule:**

| what | reading that looks right | what is true | how it was settled |
|---|---|---|---|
| the ORB grid value is a `fillRect`, not a blit | Qt's `fillRect(QRectF)` goes through `toNormalizedFillRect`: `qRound(edge + 0.5 - 1/64)` | plain `QRectF::toRect()` — the same rule the axis-aligned blit already uses | one ORB cell at a time against a black plate: **169/169 cells exact** for `toRect`, **1/169** for the `toNormalizedFillRect` form |
| the enemy sprite | a constant of the entity, as in every other game here | `image_for_type(ENEMY)` reads `cur_time` and `eat_time`: ENEMY_WEAK while `cur_time - eat_time < eat_timeout`, else `ENEMY + rem` with `rem = (cur_time/2) % 4` folded `3 -> 1` | enumerated over 32 `(cur_time, eat_time)` pairs against the four sprites rendered alone: reachable ids exactly `{3, 6, 7, 8}`, **0 unmatched** |
| the z=0 slot order | needs to be recovered, since enemies can stand on orbs | unobservable, as it was for heist | reversing the z=0 entity list changed **0 of 360** real frames, and **0 of 360** had two z=0 entities overlapping even in world space |

Three consequences for the state. The enemy sprite is *shared*, so three enemies cost **one**
dimension over four images, not three. The ORB fill is 1, 2 or 4 device pixels per cell at 64 px,
exactly `(0, 255, 0)`, and **no two cells' fills share a pixel** — which is what lets it be one
coverage table composited after the four grid gathers instead of a fifth gather. And
`load_background_images` points at `topdown_simple_backgrounds`, which holds **one** image, so chaser
has no background dimension at all: declaring one would be a dead dimension rather than a knob.
199 float32 = 796 B against a 12,288-byte frame.

**`bigfish` renders bit-exactly and does not ship, and the reason is not the renderer.** 240 frames,
100.0000% of pixels, max |diff| 0, with all three fish sprites, both reflections, radii 0.2505 to
1.9936, five background indices and `bg_pct_x` from 0.010 to 0.967 exercised. It needed three
generalisations — per-entity radius, per-entity sprite (and therefore a per-environment *source
extent*, since the three fish are 78x46, 112x88 and 108x58), and no grid at all — and it produced the
first game in the family where **the z=0 draw order is observable**: fish overlap, and reversing the
z=0 list changed **4 of 166** real frames against heist's and chaser's 0.

What stops it is that a fixed-width slot state has to satisfy two requirements that bigfish makes
contradictory, and both sides are measured:

- **the cap must be at least the reachable maximum**, or the renderer silently drops an entity.
  12,800 states harvested from 32 parallel envs over 2,000 steps hold up to **24 fish at once**
  (p99 13, tail `{17: 2, 20: 1, 21: 1, 24: 1}`);
- **every declared dimension must change the frame**, and slot `i`'s coordinates only do when slot
  `i` is occupied. The shipped pool holds **mean 4.3 / max 10** fish, so slots 7+ are dead at B=8 and
  9+ at B=32; a steady-state pool tops out near 15, because a 24-fish state occurs **once in
  12,800**.

So the cap must be ≥ 24 and ≤ about 15. No value satisfies both, and the gap is not an artefact of
the pool: it is the ratio between a spawner's tail and its mode. Biasing the pool to busy frames
would close it and would also stop `sample_states` being the env's own distribution, which is the one
thing that makes every parity number here mean something.

**That is the line, and it predicts the rest of the family without rendering anything.** chaser ships
because its cast is a constant of the game — 3 orbs, 3 eggs, 3 enemies, fixed in `game_reset`. Of the
other six fixed-camera games, `bigfish` (1–24), `bossfight` (4–142), `dodgeball` (8–17),
`plunder` (3–16) and `starpilot` (1–22) are all spawner casts and fail the same way; `leaper` (2–68)
fixes its logs and cars at level generation, so it is the one worth checking. The economics agree
independently: bigfish is **0.47x its own reference at batch 4096 and never crosses over**, because
33 slots is 33 serial full-frame composites. `scratch/procgen/bigfish_env.py` keeps the working
renderer so the parity number stays reproducible and so the boundary can be re-tested if the contract
ever grows a variable-length state.

## 3. Craftax: the cleanest fit in the group

`scratch/pixel_native/probe_craftax.py`, `verify_craftax_render.py`.

### 3.1 It is already the right shape

`make_craftax_pixel_renderer(BLOCK_PIXEL_SIZE_AGENT)` is a pure JAX function of `EnvState` over a fixed
16x16 texture atlas (143 PNGs for full Craftax, 59 for Classic; 212 kB and 150 kB on disk, nearest-
neighbour-resized to 10 px at load). The frame is an index grid times an atlas in the strict sense: a
9x11 (full) or 7x9 (classic) window of `state.map[player_level]` scaled to 10 px per tile, then item,
mob, projectile and player layers alpha-composited, then a 4-row inventory strip.

Called on the stored state it reproduces the env's own observation to 5.96e-08 in [0,1] on every one of
40 steps, and is deterministic on repeated calls.

That residual is not a rendering difference and it is fully accounted for. `get_obs` is
`self._render_fn(state) / 255.0`; XLA contracts the division into a reciprocal multiply. Recomputing
both forms in numpy float32 settles it (`craftax_ulp_attribution.py`):

| form | matches `get_obs` |
| --- | --- |
| `raw / float32(255)` | 0.0% of steps |
| `raw * float32(1/255)` | 100.0% of steps, both variants |

So the renderer is exactly reproducible; a port only has to match the reciprocal multiply. One residual
caveat: on CPU, `get_obs` compiled standalone disagrees with the same `get_obs` fused inside the jitted
`step_env` on 2.5% of steps for full Craftax (on GPU, 0%), so bit-agreement with `step()`'s observation
is not something a port fully controls. Quantised to uint8 the renderer matches the env observation on
97.5% of frames, and where it differs it is one value straddling a .5 rounding boundary.

### 3.2 The state as shipped is bigger than the frame

The full `EnvState` is 273276 B, 6.37x the 42900-byte frame, because it carries all 9 levels of a 48x48
world (`map`, `item_map`, `light_map` at 82944 B each) almost none of which is on screen. Classic is
20952 B, 1.76x its frame. Storing whole `EnvState`s instead of images is a loss. The win needs one of:

- store the render-relevant slice — the `OBS_DIM` window of each map layer plus the player, inventory
  and achievement scalars: about 5268 B for full Craftax (0.12x the frame), 976 B for Classic (0.08x);
- or store `(seed, action sequence)`, since the whole env is a deterministic JAX function.

### 3.3 Port notes

- The night lighting term draws `jax.random.uniform(state.state_rng, ...)`, so the renderer stays a pure
  function of the state, but bit-exact night frames need JAX's threefry2x32 counter PRNG reproduced
  exactly.
- Match the final scale as a reciprocal multiply if the goal is agreement with `get_obs` rather than
  with the raw renderer.

Throughput of the reference renderer under `jax.vmap`, on one H200 (`craftax_render_gpu.json`) and on
CPU for contrast:

| batch | full, GPU | Classic, GPU | full, CPU | Classic, CPU |
| --- | --- | --- | --- | --- |
| 1 | 911 f/s | 1994 f/s | 709 f/s | 3298 f/s |
| 64 | 65099 | 107687 | 2728 | 16175 |
| 256 | 123854 | 397401 | 2474 | 18354 |
| 1024 | 168567 | 960612 | — | — |
| 4096 | 172453 | 1298286 | — | — |

The thing to weigh is marginal value: those GPU numbers are the *existing* JAX implementation, not a
stable-envs one. A port buys a torch-native path and drops the JAX dependency; it does not unlock
something impossible today.

## 4. NetHack / MiniHack: the strongest result

`scratch/pixel_native/probe_minihack.py`.

`minihack.tiles.glyph_mapper` is literally `tiles[glyph2tile[glyph]]`. Rebuilding the `pixel`
observation from the `glyphs` observation alone reproduces it bit for bit on 40/40 steps of
`MiniHack-Room-5x5-v0`, worst absolute difference 0.

| quantity | value |
| --- | --- |
| glyph grid | 21x79 int16 = 3318 B |
| frame | 336x1264x3 = 1274112 B |
| ratio | 384x |
| atlas | 1082 tiles of 16x16 RGB = 831 kB |
| glyph ids | 5976, mapping onto 1023 distinct tiles |

The GPU kernel is a single gather plus a reshape: `frame = atlas[glyph2tile[glyphs]]` viewed as
`(H, W, 16, 16, 3)` and transposed. No geometry, no blending, no float arithmetic, nothing to calibrate.
Batching over thousands of envs is one indexing op.

`bench_minihack_gpu.py` is that kernel (five lines of torch: two index ops, a permute, a reshape),
validated against the 16 exported official frames (bit-exact, worst absolute difference 0) and then
timed on one H200:

| batch | frames/s | GB/s written |
| --- | --- | --- |
| 1 | 26380 | 33.6 |
| 64 | 234751 | 299.1 |
| 256 | 244729 | 311.8 |
| 1024 | 247234 | 315.0 |
| 4096 | 247476 | 315.3 |

Against the official `GlyphMapper.to_rgb` at 178 frames/s (5.62 ms/frame, a Python double loop over
`np.concatenate`), that is a 1391x speedup, and it saturates at 315 GB/s of frame output — bandwidth
bound, which is the correct place for a pure gather to stop.

The limitation: this is a renderer only. NetHack's dynamics stay in the C binary, so stable-envs would
not be stepping the game on GPU. That is exactly the case the thesis targets, though — the NetHack
datasets that exist are large precisely because they store frames, and glyph grids are 384x smaller.

## 5. What landed in the package

All three go-verdicts are implemented and under the conformance suite.

| env | module | state | frame | ratio | parity vs official | throughput (H200) |
| --- | --- | --- | --- | --- | --- | --- |
| `MiniHack` | `stable_envs/envs/minihack.py` | 1659 glyph ids | 336x1264x3 | 192x (384x as int16) | 100% of all pixels exact, max \|diff\| 0 | 246,056 f/s @ 4096 |
| `CraftaxClassic` | `stable_envs/envs/craftax.py` | 127 (63 blocks, 63 mobs, facing) | 112x144x3 | 95x | 100% of all pixels exact, max \|diff\| 0 | 707,019 f/s @ 16384 |
| `ProcgenMaze` | `stable_envs/envs/procgen.py` | 629 (625 cells, agent, background) | 64x64x3 | 4.9x (19x as int8 cells) | 100% of all pixels exact, max \|diff\| 0 | 1,688,605 f/s @ 4096 (crossover B=128) |
| `ProcgenHeist` | `stable_envs/envs/procgen.py` | 196 (169 cells, agent, 3 keys, 3 locks, exit, key ring) | 64x64x3 | 15.7x | 100% of all pixels exact, max \|diff\| 0 | 317,259 f/s @ 4096 (crossover B=2048) |
| `ProcgenMiner` | `stable_envs/envs/procgen.py` | 407 (400 cells, agent, exit, background) | 64x64x3 | 7.5x | 100% of all pixels exact, max \|diff\| 0 | 1,213,165 f/s @ 4096 (crossover B=256) |
| `ProcgenChaser` | `stable_envs/envs/procgen.py` | 199 (169 cells, agent, 3+3+3 slots, enemy sprite) | 64x64x3 | 15.4x | 100% of all pixels exact, max \|diff\| 0 | 359,105 f/s @ 4096 (crossover B=2048) |

MiniHack and Craftax carry their assets embedded (zlib+base64: 68 kB for the NetHack tileset, 8.7 kB
for Craftax's two texture sets), because needing a NetHack build or a JAX install in order to
re-render a stored state dataset would defeat the purpose. **Procgen deliberately does not**: its
assets are 25 MB of PNGs and miner alone reaches 62 distinct backgrounds totalling 176 MB decoded, so
it packs a sprite atlas and a row-resampled background atlas from the installed wheel and caches them
under `~/.cache/stable-envs/procgen`. That follows MiniGrid, which likewise builds its atlas from the
installed upstream package, and procgen is a pip wheel with a prebuilt `.so` rather than a build.

`CraftaxClassic` renders the **world layer only** — the inventory strip, night lighting and the
sleeping player are out of scope and the module docstring says exactly what that costs;
`reference_render` crops the official frame to the same region so the 100% is not flattered by
comparing against a crop nobody declared.

Two things the Procgen exercise turned up that generalise:

  * **`interior_exact_frac` is `None` for `ProcgenMaze`** — a 25x25 maze at 2.56 px per cell has no
    flat 3x3 neighbourhood anywhere, so the interior statistic is undefined and the suite falls back
    to all-pixel agreement, which is 1.0. Craftax hit the same thing for the same reason; it is a
    property of dense pixel art, not of the renderer.
  * **A state pool must come from real rollouts far enough in.** Built from pristine resets, miner's
    grid is almost entirely dirt and heist's keys are all uncollected, so `bg_index`, `bg_pct_x` and
    `has_key*` present as dead state dimensions — the background is only visible through cells the
    agent has since dug. That is a property of the pool and it looks exactly like a wrong state
    contract.

Two things the Craftax exercise turned up that are worth carrying forward:

  * Craftax's alphas are **not** binary — {0, 128/255, 1} — so the observation is fractional-valued
    float32 and this repo's uint8 frame is a rounding of it. Both sides round half-to-even and agree
    exactly, but the frame is not a lossless view of the reference and should not be described as one.
  * The conformance suite's interior statistic is **NaN, not 0, when a reference frame has no flat 3x3
    neighbourhood**, and `NaN >= floor` is False, so a bit-exact renderer fails the parity assertion and
    lowering `parity_interior_min` does not rescue it. Craftax's pixel art triggers this at both 7 px
    and 16 px. The env-side fix is that `sample_states` puts an `OUT_OF_BOUNDS` band on the window edge
    half the time — a real state, since the renderer pads a map view that way at the world edge, and the
    only block whose texture is flat.

## 6. Ranking, and what is left

1. **MiniHack / NLE.** Bit-exact from a 3.3 kB state, 384x smaller than the frame, one gather, 1391x
   faster than the official renderer. Shipped.
2. **Craftax (Classic).** Exactly reproducible against an already-batched reference. Shipped, world
   layer only. Full Craftax still wants the threefry PRNG for night frames and a render-state slice so
   the stored state is 0.1x the frame rather than 6x.
3. **Procgen `maze` / `heist` / `miner` / `chaser`.** Shipped, bit-exact, 2.6-28x the official
   renderer at batch 4096 and *slower* than it below batch 128-2048 depending on the game. This
   entry used to read "real chance of stalling on pixel fidelity"; it did not stall, and section 2
   records why the estimate was wrong. It then read "the remaining 13 games are an afternoon of
   per-game draw overrides each"; §2.9 corrects that too. Two more games were done — `chaser` ships,
   `bigfish` renders bit-exactly and does not — and the reason `bigfish` does not is the state
   contract rather than the renderer: **a spawner cast cannot be given a fixed-width slot state that
   is both lossless and free of dead dimensions.** That is what separates the remaining eleven, not
   draw complexity.
4. **Atari / ALE. No-go**, on the evidence in section 1, and with no storage problem to solve even if it
   were tractable. §1.1 closes the one door that was left open rhetorically rather than by
   measurement: a GPU emulator would deliver the same storage benefit by replaying instead of by
   decoding a state, so CuLE was built (it does, on a 2026 fork plus three CUDA-13 fixes) and
   measured. It is **not** bit-identical to ALE — 4 of 22 ROMs exact at 200 frames, **0 of 7 at
   2,000**, max abs diff 252, and two ROMs whose frames are different *shapes* — and it needs **64
   parallel environments to match one CPU core**, against which single-core `ale-py` already replays
   the same order as PNG decode (1.52-2.40x RGB, 0.96-1.39x indexed). Both halves fail
   independently.

The one methodological point to carry out of this document: **two of the four verdicts here were
decided by an effort estimate rather than by a measurement, and both estimates were wrong in the same
direction.** Procgen's blocker dissolved when the renderer was interrogated instead of read, and
CarRacing's four blockers (`docs/research/carracing.md`) all fell on re-measurement. Atari's no-go is
different in kind and survives: it rests on 30 ROMs x 8 state pairs disagreeing on the next frame from
identical RAM, which is a number, not a judgement about cost.
