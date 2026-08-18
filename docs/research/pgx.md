# PGX — board games whose state IS the board

[PGX](https://github.com/sotetsuk/pgx) (InstaDeep) is 20 JAX environments: chess, go, shogi,
backgammon, 2048, othello, connect-four, hex, poker variants and the five MinAtar games. It looks
like the purest possible case for this repo — the state is a board, the frame is a gather over a
finite piece set — and for **4 of the 20 ids it is exactly that**, at bit-exact interior parity and
up to 3,710× the reference. The other 16 fail, and they fail for three *different* reasons, none of
which is the one that was expected going in.

**Shipped:** `PgxTicTacToe`, `PgxConnectFour`, `PgxOthello`, `PgxGo9x9` —
`stable_envs/envs/pgx.py`. Probes in `scratch/pgx/`.

---

## The result

| env | size | cells | interior exact | all exact | mean/255 | f/s @ 4096 | reference f/s |
|---|---|---|---|---|---|---|---|
| `PgxTicTacToe` | 240 | 9 | **1.000000** | 0.983681 | 0.245 | 108,821 | 109.6 |
| `PgxConnectFour` | 280 | 42 | **1.000000** | 0.980568 | 0.409 | 78,288 | 79.7 |
| `PgxOthello` | 270 | 64 | **1.000000** | 0.956694 | 0.799 | 81,833 | 22.1 |
| `PgxGo9x9` | 250 | 81 | **1.000000** | 0.915308 | 1.922 | 93,671 | 45.3 |

16 states from each env's own play pool, native canvas, H200. `pytest tests/gpu -k Pgx` is
**120 passed, 28 skipped, 0 failed**.

Speedup is a (ratio, batch) pair, so: **there is no crossover — this renderer is faster at every
batch, including B = 1.**

| env | B=1 | B=8 | B=64 | B=256 | B=1024 | B=4096 |
|---|---|---|---|---|---|---|
| `PgxTicTacToe` | 11.7× | 92.2× | 536× | 860× | 973× | 993× |
| `PgxConnectFour` | 16.4× | 124.5× | 630× | 892× | 975× | 982× |
| `PgxOthello` | 60.3× | 456.1× | 2,341× | 3,316× | 3,662× | 3,710× |
| `PgxGo9x9` | 29.2× | 222.5× | 1,245× | 1,822× | 2,039× | 2,070× |

The denominator is **both halves** of what produces a reference frame — PGX's `to_svg` plus resvg's
raster, warm, per frame. Neither half has a batch parameter (`to_svg` walks one state, resvg
rasterizes one document), so that rate is the reference's peak by construction. The split is not
where one would guess: on othello the **SVG string generation is 29.1 ms against the rasterizer's
16.2 ms**; on go_19x19 it is 34.2 ms against 76.0 ms.

Winning at B = 1 follows from the design rather than from luck. The plate is a cached tensor, the
coverage maps are cached per (resolution, mark scale), and a frame is one gather plus one convex
combination — there is no serial per-primitive blend against a live canvas, which is the structure
that gives other families a late crossover.

---

## G0 · There is no official renderer, so one had to be named

This is the unusual thing about the family and everything else depends on it. **PGX ships no
rasterizer.** `save_svg` writes vector text, `_repr_html_` returns the same string for a notebook,
and there is no `render()` anywhere in the package. "The official renderer" is not a thing that
exists to be matched; it has to be *named*, and the name is part of the claim.

Two candidates, and the choice was decided by measurement, not preference:

| | availability | determinism | canvas |
|---|---|---|---|
| `cairosvg` | binds `libcairo.so.2` via `ctypes.util.find_library` **at call time**: fails on this cluster's login node (`no library called "cairo" was found`), succeeds on the compute nodes | stable when it works | — |
| `resvg_py` | self-contained Rust wheel (tiny-skia), no system library lookup | identical across two calls **and two processes** (sha256 of the PNG, all 5 games) | honours the declared canvas exactly at 0.5×, 1×, 2× — no resampling |

`cairosvg` is disqualified by the first column: a reference whose availability depends on which host
the test lands on cannot anchor a parity number. Note that `import cairosvg` **succeeds either way** —
this is the "probe the actual failing path" rule, and it is why `stable_envs/envs/pgx.py` rasterizes
a 1×1 document at import instead of importing a name.

resvg is therefore the reference, and it is a hard import-time dependency rather than a
reference-only extra, because it also bakes the plate.

---

## The design: bake the furniture through the reference, gather the marks

Every one of these drawings is (board furniture) + (one mark per occupied cell), and the furniture —
background, grid lines, border, star points — is a function of the appearance **alone**. So it is
baked the way the Fetch family bakes its floor and table: by asking the reference to draw the scene
with every mover removed. Here "every mover removed" is spelled **the empty board**, which is a state
PGX draws natively. The plate is `resvg(pgx.to_svg(empty))` and the furniture is exact *by
construction*.

That is most of the frame and all of the hard part. On go_19x19 the furniture is 34 grid lines at
`stroke-width="0.5px"`. A 0.5 px line has no interior: resvg paints it as a partial-coverage grey
straddling one pixel column, and any independent rasterizer lands on a different grey. Reproducing
those 34 lines with our own rasterizer put **6.1 % of the frame (15,300 of 250,000 pixels)** into
permanent disagreement before a single stone was drawn.

The mover pass is a **lattice gather**. Measured off the emitted documents, all four geometries
reduce to the same rule, `centre = (index + 1) × GRID`, and in every game the mark's outer radius is
below half the grid pitch (go 11.86 < 12.5, othello 13.0 < 15, connect_four 12.17 < 17.5,
tic_tac_toe 25.5 < 30) — so **only the nearest lattice point can cover a pixel**. Cost is O(pixels),
independent of board size: 361 cells cost what 81 cells cost at the same resolution.

### Constants that read plausibly and are wrong

| what | the plausible reading | what is actually true | how it was found |
|---|---|---|---|
| the first `<rect>` | the background | **dead** — `_make_*_dwg` adds it to the *drawing*, and the Visualizer's own full-canvas background is emitted after it and covers it | using it as the background gives a canvas 25–35 px too small on every side |
| go's board | a board of ±1 | **chain IDs** (measured −317, −254, −160, 52, 169, 182); the drawing does `clip(board,−1,1)` | the render state is 361 ternaries, not 361 chain ids |
| othello's board | `_x.board` | `_get_abs_board` = `board if turn==0 else −board` | storing `board` alone makes the frame depend on something not in the state |
| a stone's colour | `ColorSet.p1_color` / `p2_color` | tic_tac_toe strokes **both** its X and its O with `grid_color` | 11,504 disagreeing pixels at a **mean of 146.7/255**, 34 of them flat-area at the full 255 |
| stroke geometry | fill out to `r` | stroke *straddles* the path: fill to `r−0.5`, stroke `[r−0.5, r+0.5]` | half a pixel of colour boundary on every stone |
| line caps | this repo's `add_segment` (round caps) | SVG default is **butt** | round caps extend each of the X's four arms 1.5 px past its endpoint — PGX's own r=0.84 endpoint discs are the tell |

The mark-colour row is the one that matters most, and the fix is structural rather than a corrected
constant: `_mark_colors` **places one mark of each player and diffs the drawable list against the
empty document**, so the renderer asks the reference which colour it paints with instead of reading
an intent out of `pgx/_src/dwg/`.

### The anti-aliasing residual, and three things that did not fix it

Interior parity is 1.0, so the whole residual is on the 1–2 px rim of a mark. An isolated stone
(`probe7_onestone.py`) localises it precisely: **every transition is at the identical pixel and every
interior matches**; one pixel per edge differs. Three attempts to close it:

| change | go_19x19 all-exact | mean/255 |
|---|---|---|
| 1 px linear ramp | 0.889234 | 2.579 |
| uniform sub-sampling 8×8 → 128×128 | 0.889234 → 0.887414 | 2.579 → 2.513 |
| sub-scanlines, exact horizontal extent, 1 → 64 | 0.902120 → 0.887368 | 3.427 → 2.464 |
| disjoint regions instead of SVG's overlapping fill-then-stroke | 0.887610 | **2.469** |

Two lessons. First, uniform sub-sampling has the wrong error law here: for a nearly axis-aligned
edge all *S* sub-rows cross in the same sub-column, so the error is **1/(2S), not 1/S²** — measured
5.8 % at S=8 against a predicted 6.25 %. Second, the compositing model is a real correctness issue
independent of sampling: painting fill(`d<r`) then stroke(`|d−r|≤w/2`) *src-over* double-counts their
overlap and gives a combined coverage of 0.898 where the true covered area is 0.843, which on a black
stone with a black stroke is a bright ring inside every stone. Disjoint regions give the exact union
and improve the mean; **they do not change the exact count**, because the reference sits at 0.864 —
near the union but not on it, about **0.011 px of effective radius** inside tiny-skia's stroker.

That is a rasterizer's own rule, not a constant this repo can measure its way to, and it is exactly
the situation `interior_mask` exists for: *neither rasterizer is ground truth at an edge*.

`exact_frac` is highest at 1 sub-scanline (0.9021) and the mean is best at 16 (2.464). **The
fidelity-optimal setting was chosen, not the metric-optimal one** — picking 1 to clear a threshold
while the mean got 39 % worse would be gaming the number.

---

## `go_19x19`: built, measured, NOT registered

Same renderer, same interior agreement — **1.000000, bit-exact** — but all-pixel agreement is
**0.8876 against the suite's shared `ALL_EXACT_MIN` of 0.90**, so registering it would ship a red
test. The cause is geometric: at its native 500 px a mid-game board carries ~150 stones whose 1–2 px
anti-aliased rims are **11.3 % of the frame**, which caps `exact_frac` at ≈ 0.89 however good the
coverage model is.

It is not registered, but it *is* constructed (`stable_envs.envs.pgx.PgxGo19x19`), so every number
above is reproducible. Rendering it is 27,447 f/s at 500 px — **3,023× the reference's 9.1 f/s.**

The honest framing: this is the one id where the repo's shared all-pixel floor and an
analytically-antialiased reference disagree about what "correct" means. Raising `default_size` would
pass it (rim fraction halves with resolution) and would be metric-gaming, since 500 is the canvas PGX
draws.

---

## G-gate failures: why 16 of 20 ids are out

### 14 text-bearing ids — the reference does not draw their text

A census (`probe1_svg_census.py`) splits the 20 ids cleanly:

| group | count | ids |
|---|---|---|
| no `<text>`, no `<image>` | 6 | tic_tac_toe, connect_four, othello, go_9x9, go_19x19, hex |
| `<text>` and/or `<image>` | 14 | chess, shogi, animal_shogi, gardner_chess, backgammon, kuhn_poker, leduc_holdem, sparrow_mahjong, 2048, + the 5 minatar |

The obvious objection is that a font can simply be **pinned** — name the reference as "resvg plus
this font file" and the decoder is a constant of the family again, exactly as naming resvg made the
rasterizer one. Chess is the prize: its pieces are `<image>` references to SVG **path** files (a
fixed 12-piece alphabet, an atlas), and its only text is the static rank/file labels. Tested, and it
fails for a sharper reason than expected:

**resvg draws no text at all.** A minimal document containing one `<text>` element rasterizes to
**zero** non-white pixels under the default configuration, under `skip_system_fonts=True`, and under
an explicitly pinned `font_files=[DejaVuSerif.ttf]` — while the same document with a `<rect>` paints
1,200:

| document | default | `skip_system_fonts` | pinned `DejaVuSerif.ttf` | `serif_family="DejaVu Serif"` |
|---|---|---|---|---|
| `<text font-family="Serif">Ag8</text>` | **0** | 0 | 0 | 828 |
| `<text font-family="Courier">Ag8</text>` | **0** | 0 | 0 | 828 |
| `<text>` with no family | **0** | 0 | 0 | 828 |
| `<rect>` (control) | 1,200 | 1,200 | 1,200 | 1,200 |

Consequently chess, shogi and 2048 produce a **bit-identical raster (equal sha256)** whether the
machine's 8 system fonts are visible, hidden, or replaced by one pinned file — the labels are absent
from all of them. A chess parity number would have been measured against a frame with its
coordinates silently missing, **and it would have passed**. This is the same shape of trap as a
pygame viewer under `SDL_VIDEODRIVER=dummy` rendering a blank surface.

Text *can* be forced to appear, and how it appears is itself disqualifying: `serif_family="DejaVu
Serif"` yields 828 glyph pixels — but the **identical** 828 for `Serif`, for `Courier`, and for a
document naming no family. The override collapses every family request onto one face, and PGX asks
for three (`Serif`, `Courier`, `sans serif`).

Shogi closes it independently: its labels include CJK (`一 七 三 九 二 五 先 八 六`) and the player
glyphs `☖ ☗`, which DejaVu does not contain and which no font on this machine provides.

**So the text ids are not deferred because a font is per-installation content. They are deferred
because the named reference does not render their text, and the only configuration that renders any
text renders the wrong one.** Either way the raster is not the document's appearance, and a parity
claim against it would be a claim about a frame nobody sees.

*If this is ever revisited*, the route is a rasterizer that actually draws text (a resvg build with a
fontdb populated per-family, or a browser), pinned and named — at which point chess is genuinely
attractive: 12 piece paths and 16 static labels, all bakeable into the plate, with the pieces an
atlas gather over a lattice exactly like the stones here.

### 5 MinAtar ids — the reference is a matplotlib figure

MinAtar does not use `svgwrite` at all. `to_svg` calls `pgx.minatar.utils.visualize_minatar`, which
builds a **matplotlib figure** (`plt.subplots`, `ax.imshow`, `ax.set_axis_off`,
`plt.savefig(format="svg", bbox_inches="tight")`).

The *content* is the purest case in the whole family — the frame is `cmap[argmax(obs)]` over a 10×10
grid with a fixed palette, and the render state is tiny:

| id | observation | bytes |
|---|---|---|
| `minatar-breakout` | (10, 10, 4) bool | 400 |
| `minatar-asterix` | (10, 10, 4) bool | 400 |
| `minatar-freeway` | (10, 10, 7) bool | 700 |
| `minatar-seaquest` | (10, 10, 10) bool | 1,000 |
| `minatar-space_invaders` | (10, 10, 6) bool | 600 |

It fails **G3 (shared decoder)** on the reference side rather than the content side: the frame's
geometry is whatever matplotlib's layout engine and `bbox_inches="tight"` produce at the installed
version's defaults, and the picture is an embedded resampled PNG inside the SVG. That is
per-installation content, and matplotlib is not a PGX dependency — it is not installed here, and
`visualize_minatar` **calls `sys.exit(1)`** rather than raising when it is missing, which terminates
the interpreter (observed: `srun` exit code 1).

MinAtar is the one entry on this list worth revisiting, because the disagreement is entirely about
the reference. If the reference were defined as the indexed-colour grid itself rather than
matplotlib's rendering of it, all five are a trivial exact tile gather.

### `hex` — text-free, and still out, for geometric reasons

- The document is **133 polygons**, of which the four player-edge markers are **concave 25-gons**
  (with `rotate(180, cx, cy)` transforms). This repo's `Scene2D.add_poly` is documented "a convex
  polygon, vertices CCW"; concave shapes need ear-clipping that does not exist here.
- The canvas is **525 × 315.789 — non-square and not an integer**, while `default_size` is a single
  `int` and `check_resolution_consistency` needs an exact integer multiple at 2×.

Neither is fundamental; both are real work, and neither is about the thesis.

---

## Against the four gates

| gate | verdict |
|---|---|
| **G1 · purity** | **Pass**, and asserted rather than assumed. `test_state_is_sufficient` drives one reference through two histories to the same board: 0/255. The plate constancy is separately asserted at construction — two plates from two different states must be bit-identical. |
| **G2 · compression** | **Pass by 2–5 orders of magnitude.** go_9x9 is 81 floats (324 B) against a 250×250×3 = 187,500 B frame — **0.0017×**. tic_tac_toe is 36 B against 172,800 B — 0.0002×. Nothing had to be sliced: PGX's `State` carries a PRNG key, a step count and a legal-action mask, none of which reaches a pixel, and the render slice is just the board. |
| **G3 · shared decoder** | **Pass.** The decoder is one baked plate per appearance (≈ 0.2–0.75 MB) plus two cached coverage maps. It is identical across episodes by construction — it is built from the *empty board*. |
| **G4 · economics** | **Pass, with no crossover.** 11.7–141× at B = 1 and up to 3,710× at batch. Notably the reference is slow in a way the naive guess gets backwards: on othello, generating the SVG *string* costs more than rasterizing it. |

---

## Why no `step`

This repo's `step` exists to provide a **batched GPU simulator where the reference is CPU-only** —
that is its whole value for MiniGrid, whose reference runs at 8–10k steps/s on one core. PGX's own
`step` is already `jax.jit`-compiled, already vmappable and already on the GPU; it is what the
library is for. A reimplementation would add a second transition function to keep in agreement with
the first and buy no capability `jax.vmap(env.step)` does not already provide.

Enumeration would have been *cheap* — tic_tac_toe has 5,478 reachable positions — and that is
exactly why the reason for not shipping one has to be the absence of a gap rather than the
difficulty of proving one. `step_reference_checkable` is not set.

---

## Reproducing

```bash
# clean venv — NOT --system-site-packages, which breaks jax's CUDA plugin discovery
python -m venv ~/venvs/se-pgx
~/venvs/se-pgx/bin/pip install pgx resvg-py pillow torch gymnasium scipy numpy "pytest>=8"

pytest tests/gpu -v -rs -k Pgx                       # 120 passed, 28 skipped
python scratch/pgx/probe1_svg_census.py              # the 20-id text/image census
python scratch/pgx/probe5_parity.py --n 16           # the parity table
python scratch/pgx/probe7_onestone.py                # one stone, one slice, level by level
python scratch/pgx/probe10_throughput.py             # the batch sweep and crossover
python scratch/pgx/probe11_fonts.py                  # chess/shogi/2048 under 3 font configs
python scratch/pgx/probe12_textcheck.py              # the decisive "does resvg draw text" table
```

`stable_envs.envs.import_errors()` names this module if `pgx`, `resvg_py` or Pillow is missing, and
the check *rasterizes* rather than imports — because that is the failure `cairosvg` has here.

Registry size is interpreter-dependent and should always be quoted with one: in the `se-pgx` venv it
is **65** (the four PGX ids present, the MuJoCo families absent — `dmcontrol`, `fetch`, `procgen`,
`sokoban` and `lunarlander` do not import there).
