# CarRacing-v3

Verdict: **go — landed, and bit-exact.** `stable_envs/envs/carracing.py` reproduces the official
96x96 `state_pixels` observation on **100.0000% of pixels, max deviation 0**, interior included:

```
CarRacing vs CarRacing-v3: {'exact_frac': 1.0, 'interior_exact_frac': 1.0, 'interior_frac': 0.6273,
                            'mean_abs': 0.0, 'max_abs': 0}
```

The previous version of this page said **no-go on four independent grounds**. All four are falsified
below, each by the measurement that falsifies it, and the pattern is worth stating before the detail:
**every one of the four was an effort or feasibility estimate, and not one of them survived being
measured.** The resample was called an obstacle without anyone recovering its filter (it is a
separable integer box filter, exact, 79 µs/frame); the throughput argument assumed a CPU
rasterisation; the anti-aliasing objection was answered by code already in this repo; and the
primitive-count ceiling had been removed before the note was written.

Two things the earlier pass never looked at turned out to be the real work, and both were tractable:

- the *car* is drawn with `pygame.draw.polygon`, a **different rasterizer** from the `gfxdraw` calls
  the road uses — they agree on only **14 of 404** polygons — and its crossing rule flips rounding
  direction *inside a single edge*, which defeats every per-edge formula. It is a per-row parity
  counter over edges in vertex order; see blocker 5, now closed, **804/804** and **225/225** polygons
  bit-identical;
- a genuine **state-sufficiency violation**: driving over a tile permanently repaints it, so the
  frame is not a function of the car's pose and the score alone. That is carried as a per-tile
  visited bit, which is more state, not an impossibility.

`docs/research/pixel_native.md` records the same pattern for Procgen, whose "conditional" verdict
also rested on an effort estimate and also did not survive contact with a measurement.

Package: `gymnasium` 1.2.3 (`gymnasium/envs/box2d/car_racing.py`), `Box2D` 2.3.10,
`pygame-ce` 2.5.7 (SDL 2.32.10). Probes: `scratch/carracing/probe_*.py`, logs in
`scratch/carracing/logs/`. Every number below is from one of them, on an H200 via
`srun --partition=open_research --gres=gpu:1`.

## The reference, re-measured

| quantity | 96x96 `state_pixels` | 600x400 `rgb_array` |
|---|---|---|
| official renderer, 100 renders | **174.5 f/s** (5.730 ms) | 140.5 f/s (7.117 ms) |
| `road_poly` quads in the episode | 394 | 394 |
| grass quads in the episode | 400 | 400 |
| quads surviving the reference's *own* cull, per frame | median **81**, worst **118** | same |
| interior (flat 3x3) fraction | 0.6896 | — |

The earlier pass recorded 211 and 169 f/s for the same two modes. Same code, different node; treat
the ratio, not the absolute, as the thing to beat.

The cull number is the one that matters and it was never taken before. `_draw_colored_polygon` skips
a polygon unless one of its vertices lands inside a box `MAX_SHAPE_DIM = 381.8 px` larger than the
window, and over 64 poses drawn from the on-track distribution that leaves **12–16 grass quads and
29–81 road quads**, not 794 (`probe_census_font.py`).

## Blocker 1 — "the observation is a resample, not a render"

**Premise true, conclusion falsified.** `_create_image_array` does call
`pygame.transform.smoothscale`, and a renderer evaluating coverage at 96x96 pixel centres cannot
produce its output. But smoothscale is not an obstacle: it is a separable integer box filter whose
rule was recovered by measurement and runs on the GPU as a gather.

The rule, per axis, for a shrink from `n` to `m` (`probe_smoothscale4.py`):

```
space = ((1 << 14) * n) // m
recip = (1 << 30) // space
acc, counter = 0, space
for each source sample v:
    if counter > 0x4000:  acc += v;  counter -= 0x4000
    else:
        frac = 0x4000 - counter
        emit ((acc + ((v * counter) >> 14)) * recip) >> 16
        acc = (v * frac) >> 14
        counter = space - frac
```

The unit is **2^14, not 2^16**, and that is the whole difference between right and wrong; it is what
an `_mm_mulhi_epu16` SIMD path needs, and the plain 16.16 reading of the same algorithm gets 90.3%
of samples. Swept over 6 size pairs x 4 image families x {X only, Y only, X-then-Y}:

| fixed-point unit | 2^16 | 2^15 | **2^14** | 2^13 | 2^12 | 2^8 |
|---|---|---|---|---|---|---|
| samples bit-exact | 90.30% | 90.41% | **100.0000%** | 99.64% | 88.50% | 69.38% |
| max deviation | 4 | 4 | **0** | 3 | 12 | 234 |

Two supporting measurements. The pass order is X then Y through an **8-bit intermediate**: calling
smoothscale twice, 1000x800 -> 96x800 -> 96x96, is bit-identical to the single call on every trial,
so the quantisation between the passes is real and has to be reproduced. And on 16 **real** CarRacing
1000x800 surfaces (`probe_smoothscale_gpu.py`) the reimplementation is exact:

| target | pixels compared | exact | max deviation |
|---|---|---|---|
| 96x96 | 442,368 | **100.0000%** | 0 |
| 600x400 | 11,520,000 | **100.0000%** | 0 |

The rule also *predicts* the observation the earlier pass used as evidence that the resample was
intractable. Flat `bg_color` 102 becomes 101 after the X pass (1000 -> 96) and 100 after the Y pass
(800 -> 96), giving `(100, 202, 100)`; at 600x400 the Y pass is an exact halving (`space` is
131072 exactly, `recip` 32768) so 102 -> 101 -> 101, giving `(101, 203, 101)`. Both match the frames.

Cost, as a batched gather of at most 12 taps per output sample: **79 µs per frame**
(20.2 ms for a batch of 256), i.e. **12,658 resamples/s**, against 5.730 ms for the official
renderer's entire frame. The resample is 1.4% of the budget it was said to consume.

## Blocker 2 — "that kills the speedup"

**Every premise falsified; the conclusion survives in weakened form, for a different reason.**

The old derivation was ~150 primitives x 800,000 pixels at a measured 5.9e10 primitive-pixels/s,
giving 490 f/s. Both inputs are wrong and so is the model:

- the pixel work is not the cost. `fill_polygon_mask` + `paint` at 1000x800 costs 2.09 ms for a
  batch of 128 frames, i.e. **49.0 Gpx/s** — a polygon-covering-the-frame fill for 128 frames in the
  time the official renderer takes to draw one third of a frame;
- there are not 150 primitives; there are 81 (median) after the reference's own cull;
- but the frame **cannot be drawn in one launch**, and that is the real constraint. `_render_road`
  calls `gfxdraw.aapolygon` before `gfxdraw.filled_polygon` for every quad, and `aapolygon` blends
  against whatever is already on the canvas, so polygon *i*'s fringe depends on polygon *i-1*'s
  fill. The draw is a chain of 81–118 alpha-blend steps and no two steps can be merged.

That cost is per **step**, not per edge or per pixel. One `aaline` call takes essentially the same
wall time whatever it covers (`probe_census_font.py`):

| edges in one `aaline` call | 4 | 64 | 800 | 3200 |
|---|---|---|---|---|
| ms | 8.068 | 8.083 | 8.181 | 8.258 |
| µs/edge | 2017 | 126 | 10.2 | **2.58** |

So the honest throughput model is *steps x per-step overhead*, and 118 steps of
`aapolygon(4 edges) + filled_polygon` measure as (`probe_pipeline.py`):

| batch | eager | as a CUDA graph |
|---|---|---|
| 8 | 1.018 s → 7.9 f/s | 0.189 s → 42.4 f/s |
| 32 | 1.021 s → 31.3 f/s | 0.237 s → 134.7 f/s |
| 128 | 1.021 s → 125.4 f/s | 0.431 s → **297.2 f/s** |

Eager is **below** the official renderer at every batch size — 8.65 ms per polygon is python
dispatch, and it does not amortise because the steps are serial. Captured as a CUDA graph the
dispatch disappears and the remaining 1.60 ms/polygon (B=8) to 3.65 ms/polygon (B=128) is real GPU
work. Fitting those three graph points in B separates the two terms:

```
per render call  =  118 x 1.465 ms          (serial, batch-INDEPENDENT: the Wu pass + launch chain)
                 +  118 x 0.0171 ms x B     (the fill, which is the only part that scales)
                 =  0.173 s + 0.00202 s x B
```

Which of the two dominates decides how much headroom is left, and they say different things:

- **with the mask fill as written**, the batch term wins and throughput saturates at
  `1/0.00202` = **~500 f/s** — this is the number to quote today;
- **with a fused fill** — one kernel per primitive computing span bounds per row in registers and
  writing only covered pixels, instead of a `(B,H,W)` boolean mask plus three `masked_fill_` passes
  (see `stable_envs/envs/procgen.py::_blend_kernel` for the shape of it) — the batch term goes
  towards zero and throughput becomes linear in batch against the 0.173 s serial floor:
  **740 f/s at B=128, ~3,000 at B=512, ~5,900 at B=1024**, the last needing a 2.4 GB canvas.

Both of those are extrapolations from the three measured graph points and neither has been built;
the measured number is **297.2 f/s at B=128**. The serial floor is the part that no kernel work
removes, because it is what the reference's draw order forces.

**So the measured speedup is 1.7x and the argued ceiling is 3-30x depending on batch, not the 1000x
this repo gets elsewhere** — PushT is 2.96 M f/s and MiniGrid 6.1 M f/s because their frames are
small *and* their draws are one launch; highway's 604-primitive roundabout gets 1,716 f/s against
90 f/s on CPU because its primitives are order-independent and go through the fused kernel in one
pass. CarRacing has neither property, and 118 serial alpha-blend steps is the price. The old
"about 2x" figure was right by accident: its derivation was wrong at every step.

## Blocker 3 — "the frame is not a function of the geometric state" (the score text)

**Falsified as a blocker, and the state issue is real but elsewhere.**

The glyph path is finite and bakeable. `font.render("%04i" % self.reward, True, (255,255,255),
(0,0,0))` blits an opaque surface, so the score region is exactly one bitmap per string:

- `freesansbold.ttf` at 42: digit advance 23 px, `-` 13 px, all glyphs 42 px tall;
- **kerning is real**, so per-position composition is out: `render("1234")` is 94 px wide where the
  advances sum to 92, and the columns a single digit moves are not disjoint (position 0 moves
  columns 1..72, position 1 moves 26..72);
- but the *string* set is finite. `reward` reachable in [-100, 1000] gives **1,101 strings**, all
  46 px tall, 78–102 px wide, grey (R=G=B, 118 levels) — a **5.17 MB** atlas indexed by one state
  dimension. The blit rect follows from `text_rect.center = (60, 750)`; for `"0000"` it is
  `Rect(11, 727, 99, 46)`.

Measured worth: moving `reward` by 137 changes **0.52%** of the 96x96 observation (48 of 9216 px),
mean |d| 0.193/255, max 166 — reproducing the earlier pass's "46 of 9216 pixels" observation.

The indicator panel named in the same blocker is not a glyph problem at all: those quads are
axis-aligned, so their scanline crossings are constant per row and every candidate rasterizer rule
agrees on them. They are worth **12.09%** of the observation (mean |d| 23.250/255), i.e. more than
20x the score text, and they are the easy part.

## Blocker 4 — "aapolygon blends, and 810 primitives exceeds the kernel's ~340 ceiling"

**Falsified on both halves.**

The blended edges are already reimplemented. `stable_envs/render/gfxdraw.py` reproduces SDL_gfx's
Wu line, its packed-channel-word blend and its Cohen–Sutherland endpoint clipping bit-exactly — 0
differing pixels over 250.9 M pixels on the classic-control family. Nothing about CarRacing's
`aapolygon` calls is new. Measured worth of the fringe: skipping it moves **3.05%** of the
observation, mean |d| 0.052/255, max 21.

The primitive ceiling no longer exists and never applied. `raster2d.py`'s primitive loop is a
runtime loop, not `tl.static_range` (`docs/research/highway.md` §4 records the 604-primitive
roundabout at 1,716 f/s), and the gfxdraw path is a different code path from the fused SDF kernel
anyway. The count is also wrong by 7x: 81 median / 118 worst after the cull, not 810.

## Blocker 5 (new, and now closed) — `pygame.draw.polygon` is a second rasterizer

CarRacing draws through two different pygame entry points, and only one of them is the one this repo
already reimplements:

| source line | call | what it draws |
|---|---|---|
| `car_racing.py:784-785` | `gfxdraw.aapolygon` + `gfxdraw.filled_polygon` | background, 400 grass, 394 road |
| `car_racing.py:704, 731` | `pygame.draw.polygon` | indicator panel and bars |
| `car_dynamics.py:300, 335` | `pygame.draw.polygon` | 4 hull fixtures, 4 wheels, wheel stripes |

`stable_envs/render/gfxdraw.py::fill_polygon_mask` reimplements the first row and **not** the others.
Measured three ways on identical integer vertices — 404 polygons (the three worked examples below
plus 401 random convex quads), `probe_fill_threeway.py`:

| comparison | polygons identical | differing pixels |
|---|---|---|
| `gfxdraw.filled_polygon` vs `fill_polygon_mask` | **404/404** | **0** |
| `pygame.draw.polygon` vs `gfxdraw.filled_polygon` | **14/404** | 17,167 |
| `pygame.draw.polygon` vs `fill_polygon_mask` | **14/404** | 17,167 |

So the repo's fill is exact for the road and wrong for the car, and confirming the first row does not
transfer to the others. This is `render/gfxdraw.py`'s own rule 5 — "`pygame.draw.*` is a different
renderer" — reaching an env that needs both. Side by side on the rotated rectangle below:

| row | `pygame.draw.polygon` | `gfxdraw.filled_polygon` = `fill_polygon_mask` |
|---|---|---|
| 85 | 67..112 | 68..112 |
| 86 | 68..113 | 68..112 |
| 89 | 70..110 | 69..110 |
| 91 | 70..104 | 70..105 |
| 92 | 71..101 | 70..102 |

Two of the three parts of the `pygame.draw.polygon` rule were recovered:

- **vertices truncate toward zero to integers.** A triangle apex at x = 100.99 renders identically
  to one at 100.00 (last painted column at y = 35, 65, 95 is 3, 6, 9 for D = 10.00, 10.40 and 10.99,
  and 3, 7, 10 for D = 11.00); a quad starting at y = 39.3, 39.5 or 39.8 starts at row 39 in all
  three cases.
- **the edge activation is half-open** — `(y1 <= y < y2)`, plus the usual special case on the last
  row — which the `y = maxy` rows confirm.

The third part is the crossing-to-column rule, and it is not a function of the edge. On a controlled
quad whose left boundary is one edge and whose right boundary is a vertical line, the painted column
is exactly `floor` of the exact crossing on **18,392 of 18,392 rows** (ceil 5.013%, round 51.696%,
C-truncation-toward-zero anchored at the top vertex 51.599%). But on a rotated rectangle,
`px = [74, 56, 95, 113]`, `py = [102, 53, 39, 88]`, the left boundary comes from the *single* edge
(56,53)–(74,102) for every row 53..102, and the reference switches rule in the middle of it:

| row | exact left crossing | reference column | rule |
|---|---|---|---|
| 86 | 68.122 | 68 | floor |
| 87 | 68.490 | 68 | floor |
| 88 | 68.857 | **69** | ceil |
| 89 | 69.224 | **70** | ceil |
| 92 | 70.327 | **71** | ceil |

Row 88 is the y of the **right-hand** vertex (113,88), where the *other* boundary changes edge — so
the left column depends on the right edge, and no rule written per edge or per side can produce it.
That killed a sweep over floor, ceil, round-half-up, C-truncation at either anchor, gfxdraw's own
16.16 rule and "round outward while the span widens". The best of them, span
`[floor(min crossing), ceil(max crossing)]`, reached only **476 of 800** polygons.

### Resolved: it is a per-row parity counter

The rule is `draw_fillpoly` in pygame-ce 2.5.7, and the part that defeats fitting is one line:

```c
intersect = (y - y1) * (x2 - x1) / (float)(y2 - y1);      /* float32, not double, not integer */
if (n_intersections % 2 == 0) intersect = floor(intersect);
else                          intersect = ceil(intersect);
x_intersect[n_intersections++] = (int)intersect + x1;
```

`n_intersections` counts edges **in vertex order** and resets every row, so whether a crossing is
floored or ceiled depends on how many *earlier-indexed* edges happen to be active on that row. When
`y` passes a vertex the active set changes, a given geometric edge changes parity, and its rounding
flips mid-edge. That is exactly the row-88 flip. Two further details matter: the activation test is
`(y == maxy && y2 == maxy)`, not gfxdraw's form, and polygons with exactly **3** vertices take a
different function entirely (`draw_filltri`, three float32 slopes with a `1e-17` guard and truncating
casts) — `polygon()` dispatches on `length != 3`.

Verified against the shipped binary on three independent case sets:

| case set | polygons bit-identical | differing px |
|---|---|---|
| 3–8 vertices, convex and not, both windings, off-surface, degenerate | 804/804 | 0 |
| the 800 that scored 476/800 under floor/ceil (thin, off-screen, 8-gons) | **800/800** | **0** |
| **real** car fixtures, wheel stripes and indicator quads at real poses | **225/225** | **0** |

Including the polygon that produced the table above: 0 differing pixels. Credit for the rule goes to
a parallel session; the reproduction on the other two case sets is `probe_rule_verify.py`, and
`probe_drawpoly_rule.py` is the transcription plus its own test.

**Both rasterizers CarRacing uses are therefore now exactly reproducible**, and blocker 5 is closed.
It was worth 0.52% of the observation (48 of 9216 px, mean |d| 0.545/255) — the car's 12 rotated
polygons; the indicator quads are axis-aligned and were never at risk.

It batches: `active[e]` and `rank[e] = cumsum(active) - active` are closed form per row, so the
parity is a `cumsum` along the edge axis and the crossing is
`where(rank % 2 == 0, floor(v), ceil(v))`, with `v` kept in float32 — `render/gfxdraw.py` rule 1 is
about exactly this, and float64 lands on a different integer.

## The state-sufficiency violation

Separate from every rendering question, and it decides what the contract would have to say.
`FrictionDetector._contact` runs `tile.color[:] = self.env.road_color` — an **in-place** write to the
same numpy array `road_poly` holds — so driving over a tile permanently repaints it from
`road_color + 0.01*(i%3)*255` to `road_color`. Measured (`probe_state_sufficiency.py`), after 120
steps from a seed-0 reset:

- **12 of 394** road quads had their colour mutated (e.g. tile 2: 107.1 -> 102.0);
- restoring the complete geometric state (hull position, angle, linear and angular velocity; four
  wheel positions, angles, phases and omegas) and the reward, then re-rendering, leaves **1,104 of
  9,216 pixels** different — **1,056** of them outside the indicator strip with max |d| exactly
  **5/255**, which is the 107 -> 102 repaint and nothing else;
- control: two consecutive renders of the same state differ on **0** pixels, so the reference is
  deterministic and this is history, not noise.

Skid particles are alive (2 after that rollout) but `_render` passes `draw_particles=False` for
`state_pixels`, so they do not reach the observation.

The consequence: a compact state of car pose and score is **not** sufficient. The port has to carry a
per-tile visited bit alongside it — 319 for this track, padded to a fixed width because `state_dim`
is a class constant while the track length is per-episode. That is expressible and it is not a
blocker, but two details decide whether it survives the suite:

- **`sample_states` has to spread the car around the track.** `check_state_dims_matter` renders a
  batch and calls a dimension alive if any frame changes, so a pool drawn from the first seconds of
  an episode would report most of the visited bits dead — a property of the pool, not of the
  renderer. Cars at evenly spaced track positions each see 29-81 road quads (measured above), so a
  batch of 16 covers the mask several times over.
- The mask is monotone within an episode (a tile is repainted once and never restored), which is
  what makes it storable rather than a simulator handle in disguise.

Stating this as "23 floats plus a per-episode track", as the previous version of this page did, was
wrong.

## What landed

`stable_envs/envs/carracing.py`. Nothing below is speculative; each piece is measured exact.

- **Per-step state, 28 floats**: hull pose (3), hull linear velocity (2, drawn only through
  `true_speed`), hull angular velocity (1), four wheel poses (12), four wheel phases (4, the white
  stripe), four wheel omegas (4, the ABS bars), `reward` (1), `t` (1, because
  `zoom = 0.1*SCALE*max(1-t,0) + ZOOM*SCALE*min(t,1)` ramps for the first 50 steps).
  The steering bar is **not** among them: a Box2D revolute joint's angle is
  `bodyB.angle - bodyA.angle`, so `wheels[0].joint.angle` is already determined by wheel 0 and the
  hull. Carrying it would be a dimension `reference_set_state` could not honour — the joint angle is
  not settable — and the round-trip assert in `reference_render` checks the identity instead.
- **Per-episode**: one baked track (seed 0) — 394 road quads with their colours, the grass grid and
  the playfield quad, plus `road_color` / `bg_color` / `grass_color`, read off a live reset rather
  than transcribed. The visited mask is 319 wide (the tile count; the other 75 `road_poly` entries
  are the red/white border quads, which the contact listener never repaints).
- **Resample**: solved, exact, 79 µs/frame.
- **Road, grass, background**: `stable_envs/render/gfxdraw.py` unchanged — measured exact against
  `gfxdraw.filled_polygon`, 404/404 polygons, 0 differing pixels — one sequential step per quad,
  ~81 steps median.
- **Score**: a 5.17 MB string atlas.
- **Indicator panel and bars**: `pygame.draw.polygon`, but axis-aligned, so the crossing is constant
  per row and blocker 5's rounding question does not arise. 12.09% of the observation, and easy.
- **Car**: `pygame.draw.polygon`'s parity-counter rule, closed in blocker 5. It was the whole of the
  remaining fidelity work and it is 0.52% of the observation.
- **Throughput of the env as landed**, measured, against the official renderer's 174.5 f/s:

  | batch | 8 | 32 | 128 | 256 | 512 | 1024 |
  |---|---|---|---|---|---|---|
  | s per call | 2.661 | 3.784 | 3.717 | 4.233 | 4.553 | 5.835 |
  | frames/s | 3.0 | 8.5 | 34.4 | 60.5 | 112.4 | **175.5** |

  **This env does not beat its reference; it draws with it.** 175.5 f/s at batch 1024 against
  174.5 f/s on one CPU core is a 1.01x "speedup", and it is the only env in this repo whose GPU
  renderer is not faster than the thing it reproduces. The reason is structural and was measured
  before it was built: `aapolygon` blends against the live canvas, so ~150 primitives after the cull
  are ~150 *serial* steps, and each step's fill is `O(B x 800,000)`. The cost is therefore nearly
  flat in batch (2.7 s at B=8, 5.8 s at B=1024), which is what makes throughput linear in batch and
  the crossover so late.

  Two measured routes exist and neither is implemented here. Capturing the step chain as a CUDA
  graph removed the python dispatch and gave 297.2 f/s at batch 128 on the synthetic pipeline
  (`probe_pipeline.py`), 8.6x the eager 34.4 f/s at the same batch. Replacing the `(B,H,W)` boolean
  mask plus three `masked_fill_` passes with a fused kernel that computes span bounds per row in
  registers would remove the batch-dependent term entirely, leaving a 0.173 s serial floor — 740 f/s
  at B=128, ~5,900 at B=1024. Both are throughput work on an env whose fidelity is already settled,
  which is the right order to do them in.

## Throughput, and the ceiling it runs into

The per-primitive path this env shipped with was 203 f/s at batch 1024, against 75,146 f/s to read
the same frames back as JPEGs on eight cores — the worst ratio in the registry. The fix is
`render/gfxdraw_fused.py`: the same rules, evaluated per PIXEL in one kernel instead of per primitive
in ~110,000 torch launches. Three things make the closed form possible, and they are the interesting
part of the port:

* **the Wu walk inverts.** `aalineColor` accumulates `E` over `k` steps along the major axis, which
  reads as a loop, but `k` is recoverable from the pixel — on a y-major edge `k = y - y0` — so a
  pixel asks each edge one question instead of each edge visiting every pixel;
* **an edge writes a given pixel at most once.** The solid initial pixel sits at `k = 0`, or at
  `k = major` when `clipLine` swapped the endpoints, and the Wu pairs run `1 <= k < major`; within a
  pair the two writes differ in the major coordinate. So a pixel takes ONE weight per edge rather
  than replaying an ordered list of writes;
* **order is just the order of the register updates** — painter's order across primitives, and the
  fringe blending before the fill overwrites it.

**Measured, batch 1024, H200: 203 -> 21,820 f/s, 107x, and still 100.000000% exact against pygame
(0 of 442,368 pixels differing, max deviation 0, over 48 states).**

### Why it stops at ~22,000 f/s and not higher

The env clears the 16,889 f/s median for reading a stored 256px dataset, at 1.29x. It does **not**
beat reading back its own frames at 75,146 f/s, and that is structural rather than a missing
optimisation. The decomposition at batch 1024 (45.1 ms):

| stage | ms | what it is |
|---|---|---|
| surface kernel — drawing | 20.0 | the rasterization proper |
| surface kernel — writing the surface | 5.9 | 819 M pixels of output, nothing drawn |
| resample | 6.4 | the two `smoothscale` passes |
| `edge_table` | 6.8 | `clipLine` + the Wu constants, in torch |
| geometry assembly | 5.7 | culling, gathers, the visited-tile recolour |
| binning | 1.3 | per-tile primitive lists |
| score text | 0.4 | the glyph blit |

Two of those are a floor no implementation can go below, because the observation is *defined* as a
box filter over a fixed 1000x800 surface and that filter reads every one of its 800,000 pixels
exactly once. Measured directly: writing the surface and resampling it **without drawing anything**
is 11.1 ms, i.e. **91,893 f/s**. So the contract by itself does not forbid beating the dataset read.

What forbids it is the drawing, and the drawing is already close to the hardware. After binning, the
mean primitive list a pixel walks is **4.10** (median 3, p95 9) — a pixel is genuinely covered by the
playfield quad, a grass quad, and usually a road or border quad, so there is very little left to cull.
That is 3.4 G primitive-pixel evaluations per batch, done in 20.0 ms, i.e. **168.5 G evaluations/s**,
each one a four-edge 16.16 scanline test plus the edge-box rejects for the fringe. Making the bins
finer makes it *worse*, monotonically (46.3 ms at 64x128 bins, 70.1 ms at 4x128), which is the
measurement that says the list length is not what this kernel is bound by.

The removable part is the host side. `edge_table` is 6.79 ms at batch 1024 and 6.46 ms at batch 128 —
**flat in the batch, so it is dispatch, not work**, and fusing it into the kernel would return it
almost entirely; binning and geometry assembly are partly the same. Removing all three would leave
~32 ms, or **~31,000 f/s**.

So: **CarRacing tops out near 31,000 f/s — 0.41x the cost of reading its own frames — because the
observation is a box filter over a fixed 1000x800 surface, so the 96x96 output is 1.2% of the pixels
that must be rasterized, and a 3.4 kB JPEG of that output is the cheapest thing in the registry to
decode.** It is the one env where the storage baseline is genuinely favourable, and the honest
comparison is against the 16,889 f/s registry median, which it beats.

## Test status

```
$ pytest tests/gpu -v -rs -k CarRacing        # se-envs venv, SDL_VIDEODRIVER=dummy, one H200
31 passed, 6 skipped, 2415 deselected, 3 warnings in 427.22s (0:07:07)
```

Render parity, state sufficiency, batch invariance, per-environment style, the `out=` contract, the
input-form and ordering checks and `test_seeded_sampling_reproduces_in_a_fresh_process` all pass.
The last of those is worth naming because it is the repo's central claim in one assertion — the same
`sample_states(rng=)` bytes in a new interpreter — and it holds here because the track is *baked from
one seeded reset and cached*, not produced by driving Box2D: `_bake_track` calls `reset(seed=0)` and
reduces the episode to tables, so a fresh process re-derives the identical table from the identical
seed.

All six skips are self-explaining and none is a relaxation:

| skipped | why |
|---|---|
| `check_backend_agreement` | the env renders without a `Scene2D`; both of its rasterizers are integer, so there is no fused-kernel path to agree with |
| `test_gpu_step_tracks_the_reference_simulator` | no batched GPU simulator is declared — this is a renderer, and Box2D's contact solver is not ported |
| `test_rollout_seed_reproduces_in_a_fresh_process` | same reason: no `step`, so there is no rollout to seed |
| `test_resolution_downsample_agrees_within_the_measured_bound` | `render(size=192)` raises: the observation is a `smoothscale` of a **fixed** 1000x800 surface, so another output size is a different filter, not a different resolution. A `DOWNSAMPLE_EXACT_FRAC` entry would not un-skip it |
| `test_render_size_does_not_leak_into_later_calls` | same — one size means no size can leak |
| `test_cuda_graph_step_matches_eager` | no CUDA-graph-captured `step` |

`parity_interior_min` is left at its default 1.0 and no `parity_note` is declared, because none is
needed: the frame is exact, not exact-modulo-something.
