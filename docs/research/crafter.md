# Crafter

Verdict: **go, exact, and a second instance of the Craftax finding.** A Crafter frame is a
9x7 window of tiles drawn from a finite sprite set, plus a 2x9 inventory strip, plus two
per-pixel post-processes. Precompute the tiles and the frame is a gather. Measured against
`crafter` 1.8.3: **1,111 frames bit-identical, max absolute pixel difference 0** — 599 along a
live 690-step rollout and 512 more over the whole state pool the env samples from, driven
through a world rebuilt from the state. Throughput on one H200: **701,594 frames/s at batch
16,384** against **1,375 f/s** for `crafter.Env.render`, a **510x** speedup.

Shipped in `stable_envs/envs/crafter.py`. Probes and generators: `scratch/crafter/`
(`probe.py`, `textures.py`, `resize_rule.py`, `gen_assets.py`, `harvest.py`, `parity.py`,
`debug.py`).

## The four gates

| gate | test | result |
|---|---|---|
| **G1 purity** | render the same window from a world rebuilt from scratch, in a different episode, with a different RNG cursor | **0 differing pixels** on 512/512 states — *provided* `daylight >= 0.5`; below that the reference draws noise from `world.random` and the frame is **not** a function of any state |
| **G2 compression** | render slice vs frame | crafter's own `_mat_map` + `_obj_map` alone are **20,480 B against a 12,288 B frame = 1.67x BIGGER**; the render window is **320 B = 0.026x** (38.4x smaller), or 80 B packed (153.6x) |
| **G3 shared decoder** | build two envs from different seeds, reset both, compare the atlas | **byte-identical**; the decoder is 56 16x16 PNGs, **9,452 bytes of base64**, from which a 370-tile atlas is composed per tile size |
| **G4 economics** | GPU regeneration vs the official renderer and vs JPEG decode | **701,594 f/s** vs 1,375 f/s official (**510x**) and vs ~11,800 f/s/core of 64² JPEG decode — **≈ 59 cores replaced** |

G2 is the interesting one, and it is the same shape as Craftax's: the state the simulator
hands you is *larger than the frame*, and the render slice is 60x smaller than it. Crafter and
Craftax are the same game in two languages and they fail and pass G2 for exactly the same
reason — the state carries a 64x64 (Crafter) or 9x48x48 (Craftax) world, and the frame shows a
9x7 window of it. **The instruction the gate produces is a storage format, not a verdict.**

## The state

80 numbers:

| field | count | notes |
|---|---|---|
| `cell[x,y]` | 63 | one index into `material x object-texture` (13 x 16 = 208), plus one code for "outside the world" |
| `inv_*` | 16 | the inventory counts, each 0..9 |
| `daylight` | 1 | restricted to [0.5, 1]; see G1 |

Two things are deliberately **not** dimensions:

- **the player's facing**, and **whether the player is asleep**. `Player.texture` is exactly the
  function `facing -> {player-left, player-right, player-up, player-down}` with `player-sleep`
  overriding it, so both already live in the centre cell's code. Adding a flag could only create
  states that contradict the picture.
- **the material under an opaque object.** One code per cell rather than a (material, object)
  pair, for the reason MiniGrid's note gives: a split would leave the material dimension unable
  to change the frame wherever the object is opaque, and a dimension that cannot reach the
  render means the state is not minimal or the renderer is wrong.

## Constants that were measured, not read

The whole reason this note exists in a repo that already has a Craftax note is that four of
Crafter's five rendering constants read plausibly and are wrong.

| what | the plausible reading | what is actually true | how it was found |
|---|---|---|---|
| texture resize | `floor((i+0.5)*src/dst)` — the standard nearest rule, and the one Craftax's atlas uses | Pillow **divides before it multiplies and disagrees on 34 of the first 64 output sizes**: at `src=16, dst=6` the rule says source row 12, Pillow takes 11, because `2.6666666666666665 * 4.5 == 11.999999999999998` | rendered a row-index ramp through `Image.resize` and read the map back (`resize_rule.py`); the map is now embedded rather than derived |
| sprite compositing | blend in float64 and round | `engine._draw_alpha` blends in **float32** and multiplies back with `.astype(np.uint8)`, which **truncates** | atlas rebuilt both ways and compared against crafter's own `_draw_alpha` at units 3-40 |
| material tiles | alpha-blended like sprites | `engine._draw` **discards** the alpha channel (`texture[..., :3]`), so a transparent material texel paints its RGB over the canvas fill | 3 of the 12 materials ship an alpha channel and would otherwise show the 127 grey through |
| the daylight blend | a rounding | `_light` returns float64 and `Env.render` assigns it into a `uint8` canvas — a **C cast**, i.e. truncation | rounding costs ~1/3 of all pixels one level |
| the inventory strip | lit like the world | `render` concatenates the item view **after** `_light` has run, so the strip is the same picture at midnight and at noon | the strip matched at daylight 1.0 and not at 0.8 until the concatenation order was fixed |

Two more that are about *our* side rather than crafter's, and both were caught by measurement
rather than by reading:

- **Every tile must be stored transposed.** Crafter loads each PNG with
  `image.transpose((1,0)+...)`, draws into an `(x, y, 3)` canvas, and returns
  `canvas.transpose((1,0,2))` — so the reference's frame is the transpose of the composition,
  *tile content included*. Laying tiles out in row-major order without transposing each one gave
  a frame that was right wherever the art is symmetric: **1,980 of 4,096 pixels differed while
  interior agreement still reported 1.000**, because the interior mask only keeps flat
  neighbourhoods and a transposed tile is flat in the same places.
- **The blend factors cannot live in the style tensor as raw values.** Style is float32,
  `float32(0.4)` widens to 0.4000000059604645, and the reference computes `a + alpha*(b - a)` in
  C double — so wherever `b - a` is negative and divisible by 5 the product crosses an integer
  from the other side and the pixel lands one level low, on ~10% of channels of every lit frame.
  The constants stay in float64 in the module and the variation space exposes **scales that
  default to exactly 1.0**, which is the same device PushT's `scale` uses and for the same
  reason.

## What is excluded

- **Night below half daylight.** `LocalView._light` mixes in a vignette of noise drawn from
  `self._world.random` when `daylight < 0.5`. That is a **G1 failure in the reference**: the same
  state renders differently depending on where the world's RNG cursor happens to be. It is
  excluded rather than approximated, `reference_render` refuses such a state by name, and the
  half of the cycle that is pure — **49% of steps** — is carried exactly, noise-free tinting and
  desaturation included.
- **`step`.** This state is an *observation*. The next window depends on cells outside it, on
  objects outside it, on the player's hidden hunger/thirst/fatigue counters and on the world
  RNG. That is the same split MiniGrid draws between its top-down classes, which do ship a
  batched simulator, and its POV class, which cannot.

## Resolution

`Env.render` **centres** the view inside the frame (`border = (size - (size // 9) * 9) // 2`), so
64 px puts it at offset 0 and 128 px at offset 1. A 2S render box-downsampled to S is therefore
the S picture shifted by half an output pixel, on top of an atlas resampled from 16x16 art at a
different tile size. Measured agreement is **38.226% of pixels exact, mean 12.519/255** — by far
the loosest of the tile families, and entirely a property of the reference's centring rule. It is
recorded in `DOWNSAMPLE_EXACT_FRAC` as a floor because it still moves the instant a tile lands
somewhere else.
