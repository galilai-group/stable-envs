# Classic 2-D control: can we render it exactly, on the GPU, from the state vector?

Current semantic recovery: `CartPole` and `Pendulum` execute the historical SWM
defaults through Gymnasium CPU simulators, with full physical, lifecycle and RNG
State. Their former physical-only `transition`/`unroll` path is explicitly refused.
See [native verification](native_verification.md) for exact task evidence and
remaining native-family blockers. The renderer measurements below are historical
evidence; they do not certify environment semantics or GPU physics.

Yes, and not approximately. **CartPole-v1, Pendulum-v1, Acrobot-v1 and MountainCar-v0 render
bit-identically** — 0 differing pixels out of 250.9 M, 1024/1024 frames exact — from a batched torch
rasterizer that never touches pygame. This is a stronger result than the PushT reference
(`experiments/bench/pusht_gpu`, "100 % of *interior* pixels bit-identical", anti-aliased edges
excluded), and the reason is structural rather than lucky: gymnasium's classic-control renderers
draw through `pygame.gfxdraw`, whose primitives take **Sint16** vertices. Every float coordinate is
truncated to an integer before SDL_gfx sees it, so the entire renderer — fills, discs, and the
anti-aliased edges — is integer arithmetic that a GPU can redo exactly. There is no sub-pixel
geometry to approximate and therefore no anti-aliasing tolerance to hide behind.

Everything below was established by pixel-diffing pygame, not by reading the C source and hoping.
Scripts are in `scratch/classic_control/`.

---

## 1. Verdict table

| env | render state | dim (obs dim) | canvas | primitives | AA path | 2-D vector art? | **measured exactness** |
|---|---|---|---|---|---|---|---|
| **CartPole-v1** | `(x, theta)` | 2 (4) | 600×400 | 2 quads, 1 disc, 1 hline | `aapolygon`/`aacircle` | **yes** | **100.00000 %**, 256/256 frames exact |
| **Pendulum-v1** | `(theta, last_u)` | 2 (3) | 500×500 | 1 quad, 3 discs, +1 sprite | `aapolygon`/`aacircle` | **yes** ¹ | **100.00000 %**, 256/256 frames exact ¹ |
| **Acrobot-v1** | `(theta1, theta2)` | 2 (6) | 500×500 | 1 line, 2 quads, 2 discs | `aapolygon`/`aacircle` | **yes** | **100.00000 %**, 256/256 frames exact |
| **MountainCar-v0** | `(position,)` | 1 (2) | 600×400 | 1 quad, 2 discs, +static layer | `aapolygon`/`aacircle` + `draw.aalines` ² | **yes** ² | **100.00000 %**, 256/256 frames exact |
| **MountainCarContinuous-v0** | `(position,)` | 1 (2) | 600×400 | same primitives, **different flag position** ³ | same | **yes** | **100.00000 %**, 256/256 frames exact |
| **CartPoleVectorEnv** | `(x, theta)` × N | 2N | 600×400 ×N | identical | same | **yes** | inherits CartPole ⁴ |
| **LunarLander-v3** | terrain + lander + legs + exhaust — see §5 | **38** | 600×400 | polys + discs + 3 different AA paths | `draw.polygon` + `aapolygon` + `draw.aalines` | yes, on a wider state | **100 % interior**, 99.969 % all-pixel ⁵ |
| **DM Control** cartpole / pendulum / reacher / finger / ball_in_cup / point_mass / acrobot | — | — | — | MuJoCo meshes | OpenGL MSAA | **no — needs a 3-D path** | see §6 |

¹ Once `last_u` is set (i.e. after any `step`), `render()` also blits a torque-arrow PNG scaled by
`|u|`. That is a bitmap, not vector art — but `smoothscale` truncates its target size to an int, so
the sprite is one of exactly **114 sizes × 2 flips = 228 bitmaps, 3.9 MB baked**, and the composite
is SDL's own `d + ((s-d)*a) >> 8`, recovered by diffing over four backgrounds (100 % of pixels,
against 96.2 % for the plausible `//255` forms). Both are implemented, so the exactness above
**includes the torque arrow**. Two further measurements make `(theta, torque)` a clean 2-D state:
`u = 0` renders identically to `last_u = None` (verified over 9 angles — the zero-size sprite blits
nothing), so no "is there a torque" flag is needed; and `last_u` must be **float32**, because `step`
stores the clipped action from a float32 array and the sprite size is `trunc(scale·|last_u|/2)`.

² MountainCar's terrain (99 `pygame.draw.aalines` segments) is a constant of the environment, not a
function of the state, so it is rasterised once by pygame and cached. The state-dependent part
(car + 2 wheels) and the flag are pure gfxdraw and exact. The flag is drawn **after** the car, so it
cannot join the baked layer.

³ **The two mountain cars do not draw the same scene, despite drawing it with the same code.**
`continuous_mountain_car.render` is byte-identical to `mountain_car.render` apart from one comment,
which makes reusing the renderer unchanged very tempting — and wrong by **23 984 pixels across 64
frames**, because the flag is placed from `self.goal_position` and the two envs disagree: **0.45 for
MountainCarContinuous-v0, 0.5 for MountainCar-v0**, with a source comment attributing the difference
to a third-party port. Identical draw code over a different constant is exactly what "the frames look
the same" misses and a pixel diff catches on the first run.

⁴ `CartPoleVectorEnv.render()` is a Python `for` loop over `num_envs` pygame surfaces drawing the
same primitives — it is literally the per-env baseline, so a batched rasterizer is a drop-in.

### Ranking (research popularity × ease of exact reproduction)

1. **CartPole-v1** — the most-used RL environment there is, and the simplest scene in the suite
   (2 quads + 1 disc + 1 span). Already exact and already batched. Do this first.
2. **Pendulum-v1** — the canonical continuous-control toy; one quad and three discs. Only wrinkle is
   the torque sprite, and that reduces to a 228-entry lookup table.
3. **MountainCar-v0 / MountainCarContinuous-v0** — two popular envs for nearly the price of one
   (identical draw code, one differing constant — footnote 3), and a **1-dimensional** render state,
   which makes it the cheapest possible dataset generator. Costs one baked static layer.
4. **Acrobot-v1** — same primitives as Pendulum, no baked layer, no sprite; the easiest of the four
   in absolute terms, ranked below MountainCar only because it is less used.
5. **LunarLander-v3** — high popularity, and the render is not a pure function of the observation:
   `render()` mutates the particle list it draws. That turned out to be an argument for widening the
   state rather than for excluding the env, and it is now landed (§5).
6. **DM Control 2-D domains** — high popularity, but structurally out of scope for a 2-D rasterizer.

---

## 2. The four rules (this is the reusable part)

A census of every drawing call in `gymnasium/envs/classic_control/`:

```
 10 gfxdraw.aapolygon      10 gfxdraw.filled_polygon     2 gfxdraw.hline
  8 gfxdraw.aacircle        8 gfxdraw.filled_circle      2 gfxdraw.vline
  2 pygame.draw.aalines     1 pygame.draw.line           7 pygame.transform.flip
  1 pygame.transform.smoothscale
```

So **four gfxdraw primitives** cover the whole family. `car_racing` and `bipedal_walker` reuse
`aapolygon`/`filled_polygon` too. Implement these once and the family falls out.

### 2.1 Float vertex → Sint16: truncation toward zero

`gfxdraw` converts each coordinate with C truncation, not rounding: `x = 10.9` and `x = 10.0` fill
the same first column, `x = 11.0` does not. **This is the single most important fact in this
document** — it means the rasterized geometry is integer, so exactness is achievable, and it also
means the geometry pipeline must be **float64**. A float32 pipeline lands on the wrong integer
whenever a vertex falls within ~1e-5 of one, and the frame is off by a whole row.

### 2.2 `filled_polygon` — fixed-point scanline fill

Verified against pygame on 400 random rotated quads and 200 random triangles: **0 mismatching
pixels**.

```
for y in [miny, maxy]:                        # NOT clamped to the surface (measured)
    for each edge, ordered so y1 < y2:
        active iff (y1 <= y < y2) or (y == maxy and y1 < y <= y2)
        X = (65536*(y-y1)) // (y2-y1) * (x2-x1) + 65536*x1
    sort X; for each consecutive pair (lo, hi):
        hline(round16(lo + 1), round16(hi - 1), y)
    round16(v) = (v >> 16) + ((v & 0x8000) >> 15)
```

For convex polygons — and classic control draws nothing but convex quads and one triangle — there
are exactly two crossings per scanline, so the sort collapses to `min`/`max`. Cost is
`O(H × edges)` of span arithmetic plus one compare per pixel, which is **cheaper than the per-edge
signed-distance test an SDF rasterizer uses**, and it is the exact rule rather than a supersampled
approximation of it.

### 2.3 `filled_circle` — a tabulated midpoint circle

`filledCircleColor` paints spans while walking the midpoint circle, but each span only the **first**
time its octant coordinate changes (the `ocx`/`ocy` guards). Two natural guesses are both wrong:

| radius | `gfxdraw` row half-widths | `floor(sqrt(r²−dy²))` | midpoint walk *without* the guards |
|---|---|---|---|
| 5 | `[5,5,5,4,3,0]` | `[5,4,4,4,3,0]` ✗ | `[5,5,5,4,3,1]` ✗ |
| 10 | `[10,10,10,10,9,9,8,7,6,4,0]` | `[10,9,9,9,9,8,8,7,6,4,0]` ✗ | — |

The closed form disagrees for **every** radius ≥ 2. The guarded walk is exact for r = 0…40
(`check_hw.py`). Getting this wrong costs 64 px per CartPole frame — a 5-pixel spur across the top
and bottom of every disc, invisible by eye and instant in a diff.

### 2.4 `aapolygon` — a Wu line, in closed form

`aapolygonColor` calls `_aalineColor(..., draw_endpoint=0)` per edge, in vertex order, closing edge
last. That routine looks GPU-hostile — an incremental 32-bit fixed-point error accumulator — but the
state after k steps has a closed form, so **every AA pixel of an edge can be emitted in parallel**:

```
E      = (minor << 16) // major            # minor/major = the shorter/longer axis extent
wraps  = (k * E) >> 16                     # how far the minor axis has advanced by step k
weight = ((k * E) & 0xFFFF) >> 8
major-axis pixel: (…, weight = 255 - wgt)   minor+1 pixel: (…, weight = wgt)
```

Within one edge the major axis advances every step, so no two writes collide and the scatter needs
no ordering. Three details decide correctness:

* **Axis-aligned edges get no anti-aliasing at all.** SDL_gfx short-circuits `dx == 0` / `dy == 0`
  to a solid `vline`/`hline`. Measured: an axis-aligned `aapolygon` produces 0 blended pixels. So
  CartPole's cart is a hard-edged rectangle; only the pole is anti-aliased.
* **The blend runs on SDL's packed channel word**, `R = (dR + (((sR - dR) * alpha) >> 8)) & Rmask`,
  in *unsigned* 32-bit. The `>>8` therefore truncates at a different bit position per channel, and
  the unsigned wrap turns it into a *ceiling* when the source is darker than the destination. A
  plain `(s*a + d*(255-a))//255` gets 63 % of pixels right; the masked-word form gets 100 %.
  The alpha is `(255*w) >> 8`, which **never reaches 255** — a weight-255 AA pixel still blends, at
  alpha 254. Only the line's initial pixel (`pixelColor`, unweighted) writes the colour outright.
* **`aalineColorInt` clips its endpoints before the walk** (SDL_gfx `clipLine`, Cohen-Sutherland on
  Sint16 with a float32 slope, including an endpoint *swap* when the first point is inside). An edge
  that leaves the surface restarts its accumulator from the clipped endpoint, laying down a
  completely different fringe. Skipping this costs ~150 px/frame on CartPole past |θ| ≈ 2.4 and on
  MountainCar near the goal, where the car hangs off the right edge. Cohen-Sutherland converges in
  at most four clips, so on GPU it runs as a fixed six-iteration loop with no host sync.

Note the ordering the gymnasium idiom produces: `aapolygon` **then** `filled_polygon`, same
vertices. The AA fringe is blended against the canvas *at draw time* and the hard fill then
overwrites the interior — so what survives is a one-pixel blended rim strictly outside the fill.
Measured: ~43 fringe px per quad, ~0.05–0.13 % of a classic-control frame. That is the entire
difference between the `aa=False` and `aa=True` rows in §4.

### 2.5 `aacircle` — baked, not transcribed

`aaellipseColor`'s write set is a constant of the radius (verified: the solid-colour pixel set is
identical with and without it), so the fringe is read off pygame once per (radius, colour) and
stored as a stencil, which is cheaper and safer than re-deriving an ellipse walker. One catch: it
writes a handful of pixels **twice**. A single equivalent weight fitted on one background reproduces
that pixel on that background and is wrong by 1/255 everywhere else — which is exactly the residual
that remained wherever a MountainCar wheel or a Pendulum rod-end disc sat on something other than
the page. Fitting across five backgrounds and emitting an ordered *pair* of writes for those pixels
closes it.

### 2.6 `pygame.draw.*` is a different renderer

`pygame.draw.line` with float endpoints **truncates** the coordinate (`363.6364` → row 363, not
364). Using `round` puts Acrobot's ground line one row off — 1000 px a frame, and easy to stare
past. `pygame.draw.aalines` is a genuinely sub-pixel AA path with float endpoints, unrelated to
gfxdraw's; MountainCar's terrain uses it, which is one more reason to bake that layer.

---

## 3. Per-env primitive lists

All coordinates are **pre-flip**: every renderer builds the surface with y pointing down and calls
`pygame.transform.flip(surf, False, True)` at the end, so output row = `H - 1 - y`.
`rotate_rad(a)` is `[[cos,-sin],[sin,cos]]` applied in that y-down frame (measured:
`(1,0).rotate_rad(π/2) = (0,1)`).

### CartPole-v1 — 600×400, state `(x, θ)`

`scale = 600/4.8 = 125`, `cartx = 125·x + 300`, `carty = 100`, `axleoffset = 7.5`,
`polelen = 125`, `polewidth = 10`.

| # | primitive | geometry | colour |
|---|---|---|---|
| 1 | `aapolygon` + `filled_polygon` | cart quad `(cartx ± 25, carty ± 15)` — axis-aligned, so no AA | `(0,0,0)` |
| 2 | `aapolygon` + `filled_polygon` | pole quad, local `[(-5,-5),(-5,120),(5,120),(5,-5)]` rotated by `-θ`, translated to `(cartx, carty+7.5)` | `(202,152,101)` |
| 3 | `aacircle` + `filled_circle` | axle, centre `(int(cartx), int(107.5)=107)`, r = `int(5)` | `(129,132,203)` |
| 4 | `hline(0, 600, 100)` | the track — **drawn last, over the cart** | `(0,0,0)` |

Background `(255,255,255)`. Velocities `x_dot`, `θ_dot` never reach the screen.

### Pendulum-v1 — 500×500, state `(θ, last_u)`

`scale = 500/4.4 ≈ 113.636`, `offset = 250`, `rod_length = scale`, `rod_width = 0.2·scale ≈ 22.73`.

| # | primitive | geometry | colour |
|---|---|---|---|
| 1 | `aapolygon` + `filled_polygon` | rod quad, local `[(0,∓11.36),(rod_length,±11.36)]` rotated by `θ + π/2`, `+250` | `(204,77,77)` |
| 2 | `aacircle` + `filled_circle` | pivot cap at `(250,250)`, r = `int(11.36) = 11` | `(204,77,77)` |
| 3 | `aacircle` + `filled_circle` | rod-end cap at `int(rot((rod_length,0), θ+π/2) + 250)`, r = 11 | `(204,77,77)` |
| 4 | `blit(smoothscale(clockwise.png, s), …)` | **iff `last_u is not None`**; `s = int(scale·|u|/2) ∈ [0,113]`, flipped vertically and horizontally iff `u > 0`, centred on `(250,250)` | 329×312 single-colour alpha mask |
| 5 | `aacircle` + `filled_circle` | axle at `(250,250)`, r = `int(0.05·scale) = 5` | `(0,0,0)` |

`last_u` is **not in the observation** — it is set by `step()` and cleared by `reset()`. A dataset
generator must carry it as a render state dimension.

### Acrobot-v1 — 500×500, state `(θ1, θ2)`

`scale = 500/4.4`, `offset = 250`.

| # | primitive | geometry | colour |
|---|---|---|---|
| 1 | `pygame.draw.line` | ground, row `int(scale + 250) = 363`, full width | `(0,0,0)` |
| 2 | `aapolygon` + `filled_polygon` | link 1 quad, local `[(0,∓0.1·scale),(scale,±0.1·scale)]` rotated by `θ1 − π/2`, at `(250,250)` | `(0,204,204)` |
| 3 | `aacircle` + `filled_circle` | joint 1 at `(250,250)`, r = `int(0.1·scale) = 11` | `(204,204,0)` |
| 4 | `aapolygon` + `filled_polygon` | link 2, same local quad rotated by `θ1+θ2 − π/2`, at `(250 + scale·sin θ1, 250 − scale·cos θ1)` | `(0,204,204)` |
| 5 | `aacircle` + `filled_circle` | joint 2 at `int()` of that centre, r = 11 | `(204,204,0)` |

Trap in the source: `xys = np.array([[0,0], p1, p2])[:, ::-1]` **swaps x and y**, so link 2 hangs off
`(p1_y, p1_x)`. `p2` is computed and never drawn (only two links are zipped against two `thetas`).

### MountainCar-v0 / MountainCarContinuous-v0 — 600×400, state `(position,)`

`scale = 600/1.8 ≈ 333.33`, `height(x) = 0.45·sin(3x) + 0.55`, `clearance = 10`.

| # | primitive | geometry | colour | state-dependent? |
|---|---|---|---|---|
| 1 | `pygame.draw.aalines` | terrain, 100 samples of `height` over `x ∈ [-1.2, 0.6]` | `(0,0,0)` | **no — bake** |
| 2 | `aapolygon` + `filled_polygon` | car quad, local `[(-20,0),(-20,20),(20,20),(20,0)]` rotated by `cos(3·pos)`, at `((pos+1.2)·scale, 10 + height(pos)·scale)` | `(0,0,0)` | yes |
| 3–4 | `aacircle` + `filled_circle` | 2 wheels at `int(rot((±10,0), cos(3·pos)) + centre)`, r = `int(20/2.5) = 8` | `(128,128,128)` | yes |
| 5 | `vline(flagx, flagy1, flagy2)` | `flagx = int(1.7·scale) = 566`, `flagy1 = int(height(0.5)·scale)`, `flagy2 = flagy1+50` | `(0,0,0)` | **no** |
| 6 | `aapolygon` + `filled_polygon` | flag triangle `[(flagx,flagy2),(flagx,flagy2−10),(flagx+25,flagy2−5)]` | `(204,204,0)` | **no** |

Note the flag is drawn **after** the car, and the rotation angle is `cos(3·pos)` *in radians* — a
quirk of the source, not a bug to fix.

---

## 4. Measured results

Hardware: 1× NVIDIA H200, torch 2.11.0+cu130, gymnasium 1.2.3, pygame-ce 2.5.7 (SDL 2.32.10).
States are half real rollouts and half sampled across the full render range, because a rollout
almost never reaches the poses where a geometry bug hides.

### Pixel agreement — 256 states per env, native resolution

| env | mode | exact | differing px | worst | frames bit-identical |
|---|---|---|---|---|---|
| CartPole-v1 | **aa (exact rules)** | **100.00000 %** | **0** / 61 440 000 | 0/255 | **256/256** |
| | no-aa (fills only) | 99.94504 % | 33 768 | 202/255 | 0/256 |
| Pendulum-v1 | **aa** | **100.00000 %** | **0** / 64 000 000 | 0/255 | **256/256** |
| | no-aa | 99.92909 % | 45 380 | 183/255 | 0/256 |
| Acrobot-v1 | **aa** | **100.00000 %** | **0** / 64 000 000 | 0/255 | **256/256** |
| | no-aa | 99.87279 % | 81 416 | 252/255 | 0/256 |
| MountainCar-v0 | **aa** | **100.00000 %** | **0** / 61 440 000 | 0/255 | **256/256** |
| | no-aa | 99.94351 % | 34 710 | 255/255 | 0/256 |

The `no-aa` rows are the honest cost of skipping the Wu-line pass: **0.055–0.13 % of pixels**, all on
shape boundaries. They are included because that is the mode you would actually run if you wanted
maximum throughput and did not need bit-exactness, and because they show the AA pass is not
cosmetic bookkeeping — it is the whole residual.

### Throughput — frames/s and Gpixel/s

Two tables, because they answer different questions. The first is the prototype in
`scratch/classic_control`, which is where the CUDA-graph measurement was taken; the second is the
landed `stable_envs` classes, measured through the shipped API on the same node.

Prototype, `pygame` being one `env.render()` call, `+graph` the same code captured in a CUDA graph:

| env | mode | B | frames/s | Gpix/s | vs pygame |
|---|---|---|---|---|---|
| **CartPole-v1** (400×600) | pygame | 1 | 694 | 0.167 | 1× |
| | aa | 8192 | 47 367 | 11.37 | 68× |
| | **aa + graph** | 8192 | **57 868** | **13.89** | **83×** |
| | **no-aa + graph** | 8192 | **66 453** | **15.95** | **96×** |
| **Pendulum-v1** (500×500) | pygame | 1 | 424 | 0.106 | 1× |
| | **aa + graph** | 8192 | **52 378** | **13.09** | **123×** |
| | **no-aa + graph** | 8192 | **55 991** | **14.00** | **132×** |
| **Acrobot-v1** (500×500) | pygame | 1 | 627 | 0.157 | 1× |
| | **aa + graph** | 8192 | **46 251** | **11.56** | **74×** |
| **MountainCar-v0** (400×600) | pygame | 1 | 676 | 0.162 | 1× |
| | **aa + graph** | 8192 | **47 785** | **11.47** | **71×** |

Landed `stable_envs` classes, eager (no graph capture), B = 4096. Pendulum's exact mode also
composites the torque sprite, which the prototype did not:

| env | pygame f/s | aa f/s | aa Gpix/s | no-aa f/s | no-aa Gpix/s | vs pygame (aa) |
|---|---|---|---|---|---|---|
| CartPole | 558 | 36 549 | 8.77 | 66 105 | 15.87 | **66×** |
| Pendulum | 336 | 38 781 | 9.70 | 51 289 | 12.82 | **115×** |
| Acrobot | 493 | 31 855 | 7.96 | 52 047 | 13.01 | **65×** |
| MountainCar | 515 | 33 752 | 8.10 | 51 440 | 12.35 | **66×** |
| MountainCarContinuous | 534 | 34 259 | 8.22 | 51 447 | 12.35 | **64×** |

**Report Gpixel/s, not frames/s, when comparing across environments.** The PushT reference's
14.1 M frames/s is at 96×96; a classic-control frame is 25–27× larger, so 14.1 M frames/s there is
~130 Gpix/s and the 58 k frames/s here is ~14 Gpix/s. Quoting frames/s alone would understate this
by 25×, and quoting it the other way would flatter it — neither is the comparison anyone wants.

The exact-AA path is **launch-bound, not bandwidth-bound**, and the CUDA-graph row is the proof
rather than the assertion: capture buys **5.4× at B = 64** and only **1.2× at B = 8192**, exactly the
signature of per-kernel launch latency being amortised. The no-aa path, which issues far fewer
kernels, gains 1.3× at B = 64 and 1.00× at B = 4096 — it is already saturated. That saturation point,
~16 Gpix/s of uint8 RGB over ~20 passes, is ~1 TB/s against the H200's 4.8 TB/s, so there is real
headroom left in the fills too:

1. ~~**CUDA-graph capture**~~ — done, above. Getting there required hoisting every constant out of
   `render()`: a single `torch.tensor([...], device="cuda")` inside the hot path is not merely slow,
   it makes the whole render **un-capturable** (`Cannot copy between CPU and CUDA tensors during
   CUDA graph capture`). Worth knowing before designing the API.
2. **int32 instead of int64** in the scanline fill — all intermediates fit (`65536·(y−ylo) ≤ 2.6e7`),
   so this halves the traffic of the dominant pass.
3. **Fold the final vertical flip into the row indexing** instead of `torch.flip`, saving one full
   `B×H×W×3` copy per frame.
4. **Rasterize a `uint8` material id and expand through a palette at the end** — a third of the fill
   traffic, since these scenes have ≤ 5 colours.

---

## 5. LunarLander-v3: landed, on a wider contract

**This section is superseded by [`research/lunarlander.md`](lunarlander.md), which has the
measurements.** What follows is the reasoning that pointed at them, kept because it is the shape of
the question rather than the answer.

The blocking issue was never the drawing, it was the **state**:

* `render()` **mutates** what it draws: `for obj in self.particles: obj.ttl -= 0.15` and then
  `self._clean_particles(False)`. Rendering twice gives two different images. That breaks
  "render is a pure function of state" unless the particle buffer is part of the state.
* Engine exhaust particles are created in `step()` at randomised positions and are **simulated
  Box2D bodies** with their own dynamics; their colour is a function of `ttl`.
* The terrain is regenerated per `reset()` from 12 random chunk heights (`sky_polys`), and the
  helipad position derives from them. Not in the observation.
* The 8-D observation is a *lossy* function of the render state: it carries the lander's position,
  velocity, angle, angular velocity and two boolean leg contacts, but not the leg **joint angles**,
  which are drawn.

Three measurements decide it, and all three came back yes: the mutation inside `render()` is
deterministic (0/255 on a restored state, against 63/255 without restoring); the particle list is
bounded at **6 drawn** whenever `render` is called as often as `step`, which is the only regime that
produces frames; and a **different env instance** driven to the same 38 numbers renders **0/255**
against the original. So the state is `(11 terrain heights) + (3 poses x 3) + (6 particles x 3)` and
the contract is "replay the simulator's render state" — the same one MiniGrid's top-down classes and
PushT use, not "render the observation".

Drawing uses **three different rasterizers in one frame** — `pygame.draw.polygon` (aliased) *plus*
`gfxdraw.aapolygon` (Wu) for the body, `pygame.draw.aalines` (sub-pixel) for the outline, and
`pygame.draw.circle` for the exhaust — and only the gfxdraw one is the integer path §2 reproduces
exactly. `pygame.draw.polygon` and `gfxdraw.filled_polygon` differ on **3.63 % of lit pixels**, all at
the ends of scanlines. What that costs is 74 px a frame, none of it interior.

⁵ Interior pixels are exact over 32 states; the all-pixel residual decomposes into 34.5 px a frame of
un-ported `pygame.draw.aalines` outline and 30.6 px of the `pygame.draw.polygon` fill rule. See
[`research/lunarlander.md`](lunarlander.md).

---

## 6. DM Control: not 2-D vector art, under any camera

> **Followed up (2026-08-10).** The loader skew below is closed (`mujoco==3.11.0` +
> `dm_control==1.0.44`, in `venvs/dmc`), the verdict in this section is confirmed by measurement rather
> than by XML reading, and the 3-D path it calls for now exists: five fixed-camera cells ship at
> interior parity 1.0. See [`dmcontrol.md`](dmcontrol.md).

`dm_control` is installed but **cannot be loaded** in this environment (`'MjModel' object has no
attribute 'flex_bandwidth'` — a version skew with the installed `mujoco`), so this is a static
reading of the suite's XML, not a render.

Every 2-D-physics domain (`cartpole`, `pendulum`, `acrobot`, `reacher`, `finger`, `ball_in_cup`,
`point_mass`) includes `common/skybox.xml` and `common/materials.xml`, which means every frame has:

* a **gradient skybox** with `mark="random"` white speckles (`rgb1=".4 .6 .8" rgb2="0 0 0"`),
* a **checkerboard-textured ground plane** with `reflectance=".2"`,
* a **headlight** (`ambient .4 / diffuse .8 / specular .1`) plus a per-domain point or directional
  light, and `<quality shadowsize="2048"/>` — i.e. **shadow mapping**,
* geoms that are `capsule`/`sphere`/`cylinder`/`box` primitives rendered as shaded 3-D meshes,
* **perspective** cameras (MuJoCo default `fovy = 45`) at oblique positions —
  e.g. cartpole `pos="0 -4 1"`, acrobot `pos="0 -6 2"`, finger `pos="0 -1 .8" xyaxes="1 0 0 0 1 2"`.

So even where the *physics* is planar, the *image* is not an affine function of the plane: there is
no world→pixel map a polygon rasterizer can use, and the shading, shadows, floor texture and skybox
are depth cues a flat rasterizer cannot emit at all. `reacher` and `point_mass` are the closest
(`camera "fixed" pos="0 0 .75" quat="1 0 0 0"`, looking straight down), and even there the camera is
perspective and the floor is textured.

For calibration, a flat vector renderer produces almost no smooth gradient: measured on gymnasium's
own frames, the fraction of pixels with a small non-zero Laplacian is 0.039 % (CartPole), 0.031 %
(Pendulum), 0.110 % (Acrobot), 0.102 % (MountainCar) — i.e. the anti-aliased rims and nothing else.
A shaded MuJoCo frame is orders of magnitude above that. **Verdict: DM Control needs a mesh/3-D
path.** It does not belong in the same rasterizer.

---

## 7. What landed

The prototype is now the shipped code, in two files:

* **`stable_envs/render/gfxdraw.py`** — the shared exact primitive layer: `filled_polygon`,
  `filled_circle`, `aapolygon` (Wu closed form + packed-word blend + `clipLine`), `aacircle` (embedded
  fringe table), the SDL blit composite, and a `Gfx` constant cache whose whole reason for existing is
  to make the no-host-copy rule the path of least resistance. Its docstring carries the six rules.
  This is deliberately **not** env-specific: any env whose reference draws through `pygame.gfxdraw`
  should use it in preference to the SDF rasterizer in `render/raster2d.py`, because an SDF rasterizer
  is the wrong tool when the reference is an integer scanline fill — it can only ever approximate what
  this reproduces exactly. `car_racing` and `bipedal_walker` are the next candidates.
* **`stable_envs/envs/classic_control.py`** — `CartPole`, `Pendulum`, `Acrobot`, `MountainCar`,
  `MountainCarContinuous`. Registered names are the class names. Each overrides `render()` rather than
  `scene()`, for the same reason.

Two things surfaced only once the envs were driven through the conformance suite, and both are the
same shape of bug — identical draw code over a different constant:

* the **flag position** difference between the two mountain cars (footnote 3), worth 23 984 px;
* **`world_width` must be `MAXP - MINP`, not the literal `1.8`.** `0.6 - (-1.2)` is
  `1.7999999999999998` in float64 and the literal is `1.8000000000000000` — 2 ULP. That survives into
  `int((goal_position - min_position) * scale)`: at goal `0.45` the product is `550.0000000000001`
  one way and `549.9999999999999` the other, so the flag lands on a different column. Cost: ~35 800 px
  over 256 states, **all of it on edges**, so interior parity stayed at 100.00000 % and only the
  all-pixel diff caught it. A 2-ULP difference in a float literal is a one-pixel difference in the
  frame, and only an exact renderer can see it at all.

Neither would have been visible to a tolerance-based check, which is the argument for keeping the
floor at exactly 1.0 where the reference permits it.

Anti-aliasing is exposed as `rendering.antialias` rather than hidden: `1` (the default) is
bit-identical, `0` drops the Wu pass for 0.055-0.13 % of pixels and ~1.5x throughput. The variation
defaults reproduce the official appearance exactly, so "matches gymnasium" is an assertion about the
shipped defaults and not about a configuration nobody runs.

## 8. Reproducing this

```
scratch/classic_control/
  probe_gfxdraw.py        rotate_rad sign, float->int rule, AA-vs-fill anatomy, hline/vline bounds
  probe_exact_rules.py    filled_polygon scanline rule (0/600 mismatches), filled_circle tables
  probe_aaline.py         closed-form Wu line, first blend fit
  probe_blend.py          inverts the blend: alpha_eff == (255*w)>>8 on 99.6 % of weights
  probe_blend2.py         SDL's masked-channel-word blend, 5 backgrounds x 6 colours
  probe_clip.py           filled_polygon needs no clamping; aaline DOES clip first (0 px after)
  probe_assets.py         Pendulum's clockwise.png: 329x312, single colour + alpha
  probe_pendulum_sprite.py  114 distinct smoothscale sizes -> 228 bakeable bitmaps, 3.9 MB
  probe_dmc.py            DM Control probe (blocked by the mujoco version skew) + gradient baseline
  check_hw.py             filled_circle half-widths exact for r = 0..40
  ref_gfx.py              exact numpy reference for all four primitives + clipLine + stencil bake
  ref_envs.py             CPU reference renderers for the four envs
  verify_ref.py           CPU reference vs gymnasium
  gpu_render.py           the batched torch rasterizer
  validate_gpu.py         GPU vs gymnasium, per-env, aa on/off
  debug_gpu.py            three-way diff (gymnasium / CPU ref / GPU), per primitive
  diag_resid.py           residual maps; how the Acrobot ground-line row was pinned down
  bench_gpu.py            frames/s and Gpix/s vs pygame
  bench_graph.py          CUDA-graph capture, to separate launch cost from bandwidth
  out/*_sheet.png         gymnasium | GPU | 8x |diff| contact sheets (the diff column is black)
```

Everything runs under SLURM; nothing runs on the login node.

## 9. What this means for `stable-envs`

* **The thesis holds, exactly, for this family.** Four of the most-used environments in RL render
  bit-identically from a 1- or 2-dimensional state vector at 46–66 k frames/s on one GPU. A million
  CartPole frames is 2 float64 × 10⁶ = 16 MB of state instead of 720 GB of raw RGB — and the state
  reconstructs the pixels *exactly*, so the dataset and the renderer are the same artefact.
* **Build the primitive layer, not four renderers.** `filled_polygon`, `filled_circle`, `aapolygon`,
  `aacircle` + `hline`/`vline` cover all of classic control and reach into `car_racing` and
  `bipedal_walker`. The per-env code is then 30 lines of geometry each.
* **Keep `aa` as a switch, and capture the graph.** Exactness costs ~1.15× throughput at large batch
  once the render is CUDA-graph captured, versus ~1.4× eager and much worse at small batch. Both
  modes are legitimate; only one of them is bit-identical, and the docs should say which.
* **No host→device copies inside `render()`.** Constants belong in `__init__`. This is worth ~5×
  at small batch and is the difference between capturable and not.
* **float64 geometry is not optional**, because the whole pipeline hinges on `trunc()`.
* **Bake what is constant.** MountainCar's terrain and Pendulum's 228 sprites are functions of the
  environment, not the state. Rasterizing them once is exact by construction and free per frame.
* **Draw the line at shading.** The moment a renderer emits a gradient — DM Control, MuJoCo, any lit
  3-D scene — this approach stops being exact and starts being an approximation. Better to say so
  than to ship a "close enough" MuJoCo.
