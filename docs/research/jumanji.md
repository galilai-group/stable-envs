# Jumanji

`jumanji` 1.1.2 (InstaDeep) — 26 registered JAX environments, drawn by matplotlib viewers.
**Eight ids ported across six renderers, every one bit-identical to the official renderer on ALL
pixels.** Two measured and refused, each on a named mechanism; the remaining sixteen not attempted,
with reasons.

Eight over six because `RubiksCube-partly-scrambled-v0` and `Sudoku-very-easy-v0` differ from their
siblings only in the reset distribution — same viewer, same alphabet, same atlas — so they share the
parent's atlas and differ only in `sample_states`.

The interesting result is not the count. It is that the construction which produced them is a
*weakening* of the MiniGrid atlas that survives a renderer with none of MiniGrid's conveniences — no
tiles, no pixel alignment, no flat colours, and alpha blending against a live canvas — and that the same
construction, applied honestly, is what refuses the other two.

---

## What the official renderer is

`env.render(state)` returns **`None` for all 26 ids**. That is not a missing feature and it is the first
thing a port has to get past: every viewer is constructed with `render_mode="human"`, and
`MatplotlibViewer` binds `self._display` to the window branch at construction. The rgb_array branch is
reached by building the viewer with `render_mode="rgb_array"`.

Two ids need more than that. **`Snake.render` and `Sokoban.render` call their viewer and then discard
its return value** — so `env.render` is `None` for them however the display branch is bound. The
official renderer for this family is therefore the **viewer's** `render`, not the env's, and everything
below goes through it.

| property | value |
|---|---|
| frame | `figsize=(10,10)` inches at the figure's dpi → **1000×1000×3** by default |
| determinism, 2 calls, same process | **0/255** on every id that returns an array |
| a second env's viewer changes the first's frames | **0/255** — figures are keyed by name through `plt.fignum_exists`, and the names differ per env |
| `size` | honoured by setting the figure dpi to `size/10` per render; this is the official renderer at another resolution, not a resample. Parity measured separately at 200, 500 and 1000 |

The cross-env check is worth dwelling on, because matplotlib is among the most global-state-heavy
libraries in Python and `testing/conformance.py` deliberately keeps one live reference env per
`(class, size)` for a whole run. Jumanji's viewers never share a figure *across ids* because
`_get_fig_ax` looks one up by the viewer's own `name`, and those differ per env.

### The isolation fix that was itself a bug

Within one id the hazard is real: two viewers of the same id at two sizes *do* share a figure name, and
the suite renders at `2 * env.size` in `check_resolution_consistency` — so building the 400 px atlas
would re-dpi the figure the cached 200 px *reference* viewer is holding.

The obvious repair is to give each `(id, size)` a unique viewer **name**. That is wrong, and the way it
is wrong is the most useful thing in this document:

> **`RubiksCubeViewer` and `MinesweeperViewer` call `fig.suptitle(self._name)` — they DRAW their name.**

Renaming therefore changes the picture. Measured after renaming: RubiksCube disagreed with the official
renderer on exactly **100 pixels of every frame**, all in rows 4–7 (the suptitle band), all *unowned*
and so baked into the plate. The failure mode that matters is Minesweeper's, though, because it looked
*perfect*: its atlas and its reference were renamed identically, so it scored **1.000000 against a title
this port had invented**. A self-consistent parity number against a modified reference is exactly the
error class this repo's `reference_render` assertions exist to prevent, and it slipped past them because
the modification was upstream of the assertion.

The fix: leave the name exactly as the official env sets it, keep **one** viewer per id and never close
it, and set the figure's dpi **per render** instead — which changes size without changing a pixel of
what the reference draws. Generalised: *when isolating two configurations of a reference renderer,
isolate on something the renderer does not read.*

---

## The construction: a frame as a per-pixel lookup

MiniGrid's atlas is exact because it is built by **calling** `Grid.render_tile` rather than by
reimplementing it. The same move is wanted here, but a matplotlib cell is not pixel-aligned, not
rectangular in general, and anti-aliased — there is no tile to cut out. What survives is the weaker
statement, and it is sufficient:

> **Each output pixel is a function of exactly one cell's symbol.**

Given that, the frame is a gather:

1. `P*V` official renders, one per (cell `p`, symbol `v`), against a fixed base grid;
2. `owner[y,x]` = the single cell whose symbol moves this pixel. **A pixel moved by two cells is a model
   failure and is counted, not tolerated**;
3. `LUT[k,v]` = owned pixel `k`'s colour when its owner holds `v`, taken from those renders;
4. everything unowned is the static plate.

Every colour in a rendered frame came out of the official renderer, so exactness is *by construction* —
anti-aliased glyph fringes, alpha-blended paths and shared cell borders included, none of them modelled.

**Steps 1–4 only prove each cell is separable on its own.** The check that decides it is step 5: compose
grids with every cell different at once and diff against the official frame. That is the last two
columns below.

### Gate table (200 px; 12 uniform-random and 12 reset-distribution grids per id)

| id | P | V | owned % | conflict px | LUT MB | verify uniform | verify real |
|---|---|---|---|---|---|---|---|
| `Maze-v0` | 100 | 4 | 56.3 | **0** | 0.27 | **0/255** | **0/255** |
| `RubiksCube-v0` | 54 | 6 | 45.1 | **0** | 0.32 | **0/255** | **0/255** |
| `Sudoku-v0` | 81 | 10 | 2.4 | **0** | 0.03 | **0/255** | **0/255** |
| `Connector-v3` | 100 | 31 | 67.2 | **0** | 2.50 | **0/255** | **0/255** |
| `Sokoban-v0` | 100 | 5 | 81.0 | **0** | 0.49 | **0/255** | **0/255** |
| `Minesweeper-v0` | 100 | 10 | 4.8 | **0** | 0.06 | **0/255** | **0/255** |
| `Cleaner-v0` | 100 | 4 | 55.7 | **225** | — | 255/255 | 255/255 |
| `SlidingTilePuzzle-v0` | 25 | 25 | 6.0 | **28,235** | — | 255/255 | 255/255 |

Sudoku's 2.4 % is the thesis in one number: the only pixels that depend on the board are the digit
glyphs, so **977 pixels of a 40,000-pixel frame carry the entire state** and the other 97.6 % is a
constant of the family.

### Parity against the official renderer

64 states per env from its own reset distribution, 200 px, through the shipped `env.render`:

| class | all pixels exact | interior exact | mean/255 | max/255 |
|---|---|---|---|---|
| `JumanjiMaze` | **1.000000** | 1.000000 | 0.0000 | 0 |
| `JumanjiRubiksCube` | **1.000000** | 1.000000 | 0.0000 | 0 |
| `JumanjiRubiksCubePartlyScrambled` | **1.000000** | 1.000000 | 0.0000 | 0 |
| `JumanjiSudoku` | **1.000000** | 1.000000 | 0.0000 | 0 |
| `JumanjiSudokuVeryEasy` | **1.000000** | 1.000000 | 0.0000 | 0 |
| `JumanjiConnector` | **1.000000** | 1.000000 | 0.0000 | 0 |
| `JumanjiSokoban` | **1.000000** | 1.000000 | 0.0000 | 0 |
| `JumanjiMinesweeper` | **1.000000** | 1.000000 | 0.0000 | 0 |

**All** pixels, not merely interior ones. That is the point of building the atlas out of official frames
rather than matching a rasterizer: there is no anti-aliasing residual to negotiate, because the
anti-aliased pixels were *copied* rather than recomputed. `parity_interior_min` stays at the default
`1.0` and no `parity_note` is needed.

### The atlas is verified per size, and one id is 200 px only

`verified_sizes` is measured, not declared for caution. At 400 px five of the six renderers still
separate exactly. **Connector does not.**

Its cells carry a `linewidth=1` black edge, and a linewidth is in **points**: one point is 0.28 px at
dpi 20 but 0.55 px at dpi 40, so the shared border stops quantising to fully-black pixels and becomes a
genuine blend of the two cells it divides. **5,841 pixels are moved by more than one cell** — the
per-cell model is false there, and `_build_atlas` refuses rather than producing an atlas that is wrong
on exactly the states that matter. `JumanjiConnector` raises `ValueError` for any size but 200, which is
the path the suite already understands ("an env that renders at one size only says so") and skips on.

This is the sharpest statement of the model's boundary in this family: **separability is not a property
of a renderer, it is a property of a renderer at a resolution.** Anything drawn in points rather than
pixels can cross that boundary as the dpi changes, and the only safe response is to measure per size and
refuse the sizes that fail.

**Connector is the entry that was expected to break and did not.** Its cells are `plt.Rectangle` patches
with `edgecolor="black", linewidth=1` — adjacent cells share those edges — and its paths are painted at
**alpha 0.25 against a live Agg canvas**, which is the serial-blend shape that makes a renderer
order-dependent. It separates exactly anyway, and the reason is instructive: a shared border is black
whichever neighbour drew it, so it never *moves* with a cell's symbol and lands in the plate rather than
in a conflict. Alpha blending is not by itself a barrier to the per-pixel model — *state-dependent*
blending is.

---

## The two refusals

Both are decisive, cheap, and worth more than another port.

### `SlidingTilePuzzle-v0` — 28,235 conflict pixels, of which 25,671 were the probe's fault

```python
ax.imshow(state.puzzle, cmap=self._color_map)     # no vmin, no vmax
```

With no `vmin`/`vmax`, matplotlib normalises the colormap to the **data's own min and max**. That much
is right. What this entry used to conclude from it — "every tile's colour is a function of the whole
board", "the cleanest possible violation of the per-cell model", "no atlas repairs it" — is wrong
about the states that matter, and re-measuring it (`scratch/jumanji/probe_slidingtile_reachable.py`)
splits the 28,235 into two mechanisms of very different size.

**A sliding-tile puzzle's board is a permutation.** Over 576 states from 24 real episodes,
`state.puzzle` is a permutation of `0..24` in **576/576**, and the only `(min, max)` that occurs is
**`(0, 24)`**. So on the reachable set the normalisation is a *constant*, and the mechanism the
refusal rests on cannot fire there. What fired was the **probe**: `_build_atlas` renders `base()` with
one cell overwritten, the base is the solved board holding exactly one 0 and one 24, and overwriting
either extreme moves the normalisation of all 25 tiles — a board no episode ever reaches.

Anchoring the base so that a single-cell probe *cannot* move the extremes (two 0s and two 24s, still a
board `imshow` accepts) reproduces the split exactly:

| base | conflict pixels |
|---|---|
| solved board — what the refusal used | **28,235** (reproduced exactly) |
| anchored, min and max pinned by construction | **2,564** |

So **91 % of the recorded conflict was the colormap and it is removable without touching the
reference**, which answers the question the refusal raised: pinning `vmin`/`vmax` would indeed be
matching a modified renderer, and it is not necessary — probing on boards whose normalisation matches
the reachable set's is the same move this repo already makes when it insists a state pool come from
real rollouts.

**The verdict still stands, on the other 9 %, and it is a mechanism this page already documents
elsewhere.** All **2,564** residual conflicts lie **within 1 px of a cell edge** (2,503 owned by 2
cells, 60 by 4 — the corners), which is `Connector`'s shared-border blend: cells drawn with an edge
whose width is in *points* against a live canvas, so the border between two tiles is a genuine
function of both. Connector survives only because at 200 px its border quantises to fully black and at
400 px it does not; SlidingTilePuzzle's fires **at** 200 px.

That leaves a cheap, well-defined next step rather than a refusal: **sweep the size the way Connector
was swept.** If some dpi makes the tile border quantise to a single flat colour, the anchored
construction closes at 0 conflicts and the id is buildable with a declared scope of "boards that are
permutations" — which is every board the env produces. Recorded as measured and not attempted, not
as impossible.

### `Cleaner-v0` — 225 conflict pixels, which is exactly one cell

`CleanerViewer._create_grid_image` paints `COLORS[grid]` and then overwrites every cell in
`state.agents_locations` with the agent colour. That looks like a 4th symbol — clean, wall, dirty,
agent — and it nearly is. What breaks it is that the env has **exactly three agents and no way to
express none**: the corner of the grid alphabet where the last agent symbol is removed still has to put
three agents *somewhere*, and that somewhere turns red. So the cells are not independently settable,
and the 225 conflicting pixels are exactly the one cell the agents fall back to.

The coupling is real rather than an artifact of the encoding: any state vector for this viewer must
either carry agent positions separately (in which case a cell's pixels are moved by both a grid symbol
and a position, i.e. two owners) or accept the constraint. Recorded rather than worked around.

### Not attempted, with reasons

| id | why |
|---|---|
| `Game2048-v1`, `Tetris-v0`, `PacMan-v1` | `fig.suptitle(f"... Score: {state.score}")` — the frame carries an **unbounded integer rendered as centred text**, whose glyph positions shift with its digit count. Not a finite alphabet at a fixed position, so not a per-pixel lookup. Reachable only by cropping the title, as the Craftax port crops its inventory strip |
| `BinPack-v2`, `JobShop-v0`, `Knapsack-v1` | same shape: a title that is a function of non-grid state (`Placed: 0/19`, `Time=0`, `Total value: 0.00`) |
| `Snake-v1` | body cells carry a per-cell **age index** (`body_state`), so V scales with P: 144 cells × 144 values = 20,736 official renders per atlas. Not refused on structure — refused on build cost |
| `CVRP-v1`, `TSP-v1`, `MultiCVRP-v0`, `GraphColoring-v1`, `MMST-v0` | node-link and scatter plots over **continuous** coordinates; there is no cell alphabet at all |
| `FlatPack-v0`, `LevelBasedForaging-v0`, `RobotWarehouse-v0`, `SearchAndRescue-v0` | not measured; time |

---

## Against the four gates

**G1 · purity.** Two questions, asked separately because they are different claims. The LUT verification
shows the frame is a function of the grid *given everything else held fixed*. It does not show that no
other leaf reaches a pixel — so every non-grid leaf of a real reset `State` was substituted from a
different episode's and the official env-level frame re-rendered: **0/255 on all six**. Minesweeper's
`flat_mine_locations` and Sokoban's `agent_location` are genuinely not in the state. Connector is
stronger and structural: `ConnectorViewer.render` is handed `state.grid` itself, so nothing else is in
scope.

**G2 · compression.** Measured on the render slice, not on the shipped `State` — the pytree carries PRNG
keys, `step_count` and `action_mask`, none of which reach a pixel. Against the 200 px frame
(120,000 bytes):

| id | render state (float32) | vs frame | packed 1 B/cell |
|---|---|---|---|
| `RubiksCube-v0` | 216 B | 555× | 2,222× |
| `Sudoku-v0` | 324 B | 370× | 1,481× |
| `Maze-v0` / `Connector-v3` / `Sokoban-v0` / `Minesweeper-v0` | 400 B | 300× | 1,200× |

**G3 · shared decoder.** The decoder is the atlas, and it is a constant of the id: reset twice and the
plate, owner map and LUT are unchanged, because they are built from the viewer's own drawing code and
the viewer holds no per-episode content. Sizes are 0.03–2.50 MB — trivially GPU-resident.

**G4 · economics.** See [Throughput](#throughput) below: our f/s at five batch sizes against the
fastest honest form of the official viewer, with the crossover batch.

---

## Throughput

One H200, 200 px, `scratch/jumanji/probe4_bench.py`. Our side goes through `env.render(states, out=…)`
as a caller gets it, timed with `torch.cuda.synchronize()` inside the loop. The atlas build is excluded
and reported separately above — it is a one-off per `(id, size)` per machine.

**The denominator is the fastest honest form of the official viewer.** matplotlib has a slow form (a new
`Figure` per frame) and a fast one (one `Figure` reused, artists redrawn), and quoting a ratio against
the slow form would be a strawman. Both are measured; the larger is used:

| id | official, reused figure | official, fresh figure | denominator |
|---|---|---|---|
| `Maze-v0` | **91.16** f/s | 17.25 f/s | 91.16 |
| `Sudoku-v0` | 18.77 f/s | **19.08** f/s | 19.08 |
| `Minesweeper-v0` | **17.10** f/s | 12.66 f/s | 17.10 |
| `RubiksCube-v0` | **8.79** f/s | 7.10 f/s | 8.79 |
| `Sokoban-v0` | **7.83** f/s | 6.20 f/s | 7.83 |
| `Connector-v3` | **7.16** f/s | 6.96 f/s | 7.16 |

Maze is the fastest reference by an order of magnitude because its viewer is a single `imshow`; Connector
is the slowest because it adds 100 `Rectangle` patches per frame. Sudoku's two forms are within noise of
each other and the fresh one happened to win, which is why the rule is "take the max" rather than "assume
reuse is faster".

### Ours, and the ratio

| id | B=1 | B=8 | B=64 | B=256 | B=4096 | crossover |
|---|---|---|---|---|---|---|
| `Maze-v0` | 3,346 (36.7×) | 26,168 (287×) | 206,845 (2,269×) | 362,022 (3,971×) | **482,575** (5,294×) | **none** |
| `Sudoku-v0` | 3,390 (178×) | 26,976 (1,414×) | 212,204 (11,121×) | 671,266 (35,178×) | **1,209,900** (63,405×) | **none** |
| `Minesweeper-v0` | 3,164 (185×) | 26,360 (1,541×) | 209,628 (12,256×) | 638,327 (37,321×) | **1,107,219** (64,736×) | **none** |
| `RubiksCube-v0` | 3,362 (382×) | 26,805 (3,048×) | 212,406 (24,151×) | 420,845 (47,851×) | **580,691** (66,027×) | **none** |
| `Sokoban-v0` | 3,353 (428×) | 26,823 (3,427×) | 186,694 (23,849×) | 281,220 (35,924×) | **355,190** (45,373×) | **none** |
| `Connector-v3` | 3,252 (455×) | 27,035 (3,778×) | 196,427 (27,448×) | 310,639 (43,408×) | **402,232** (56,207×) | **none** |

**No crossover on any id: faster at every batch measured, including B=1**, where the tightest case
(Maze, against the fastest reference) is still 36.7×. That is the expected shape for a plate-plus-gather
against a live Agg canvas and it is worth stating as a property rather than leaving as an absence —
matplotlib composites artists serially against the canvas, so its cost is per frame with no batch term
at all, while ours is one copy plus one gather.

The B=4096 spread is the LUT size, not the renderer: `Sudoku` and `Minesweeper` own 977 and 1,916 pixels
so their gather is nearly free and the plate copy dominates (1.2M and 1.1M f/s, ~145 GB/s of frame
output), while `Sokoban` owns 32,400 and pays for it (355k f/s). The B=1 and B=8 columns are identical
across ids for the same reason they are in MiniGrid: at that width the render is launch-bound.


---

## Cost of building an atlas

`P*V` official matplotlib renders, memoised to `~/.cache/stable-envs/jumanji/` and keyed `(id, size)`.
This is the one number that decides whether an id can ship, because the build happens inside whichever
test touches the env first and `testing/safety_plugin.py` applies a 900 s per-test wall limit. The
conformance suite touches **two** sizes per id: `env.size`, and `2 * env.size` in
`check_resolution_consistency`.

Measured on one H200 node in a single warm process, matplotlib `Agg`, CPU jax (`JAX_PLATFORMS=cpu` —
the reference viewer is CPU-only and the GPU is the renderer's):

| id | renders (`P*V`) | 200 px | 400 px | disk (npz) |
|---|---|---|---|---|
| `Maze-v0` | 400 | 6.0 s | 7.3 s | 0.00–0.01 MB |
| `RubiksCube-v0` | 324 | 32.5 s | 33.9 s | 0.01–0.02 MB |
| `Sudoku-v0` | 810 | 29.2 s | 31.8 s | 0.02–0.05 MB |
| `Minesweeper-v0` | 1,000 | 44.2 s | 58.5 s | 0.03 MB |
| `Sokoban-v0` | 500 | 56.0 s | 78.0 s | 0.02–0.07 MB |
| `Connector-v3` | 3,100 | **340.7 s** | *n/a — model fails at 400 px* | 0.03 MB |

Two things worth reading off this. **Build time tracks the number of renders, not the resolution** —
matplotlib's cost is per artist, so 400 px costs within ~30 % of 200 px — and **the largest single build
is Connector's 340.7 s**, comfortably inside the suite's 900 s per-test wall limit, so no id can time out
on a cold machine. The atlases themselves are tiny (≤ 0.07 MB compressed), because a plate of flat
colours and a LUT over a handful of symbols compress extremely well.

Cold versus warm `pytest tests/gpu -k Jumanji` (8 envs, 254 tests):

| | time |
|---|---|
| warm (atlases cached) | **119 s**, measured |
| cold (every atlas built inside the run) | ≈ 119 s + 509 s (200 px) + 210 s (400 px) ≈ **14 min**, derived from the column above |

The build is not re-run per test: `_ATLAS` memoises in-process by `(atlas id, size, device)`, so
`check_style_is_state` — which deliberately constructs a *fresh* env of the same class — hits the cache
rather than rebuilding. Verified by the warm run's 119 s total, which is less than a single Connector
build.

Warm the cache before a first run rather than paying it inside a test:

```bash
srun ... ~/venvs/se-jumanji/bin/python scratch/jumanji/build_atlases.py --sizes 200 400
```

---

## No `step`, deliberately

MiniGrid ships a batched GPU `step` because its reference is a **CPU** simulator at 8,100–10,700
env-steps/s, so the port buys 158× at B=256 and 7,500× at B=16,384 — a capability that did not exist.
Jumanji's envs are already JAX: `env.step` is `jax.jit`-compilable and `jax.vmap`-able onto the same GPU
this renderer runs on, at whatever batch the caller wants. Reimplementing those transitions here would
add a second solver to keep in sync with upstream and buy nothing. `step_reference_checkable` is unset
on every class and the suite's simulator check skips them by design.

What this family contributes is the half Jumanji does not already do on the GPU: **the frame**.

---

## Colour variations, and what they are not

The atlas is not rebuilt per palette — that is `P*V` matplotlib renders and cannot be on a render path.
A palette entry rewrites exactly the LUT entries whose colour is that symbol's **modal** colour.

For a symbol that paints one flat colour the recolour is total, and the result is the frame the
reference would draw at that palette. For a symbol whose pixels are already a blend — a Sokoban sprite,
a Sudoku or Minesweeper digit and its anti-aliased fringe, a Connector path at alpha 0.25 — only the
dominant colour moves and the fringe keeps a blend toward the old colour, so **such a frame is not one
the official renderer would produce**. Measured, as distinct colours per symbol: Maze `[1,1,1,1]`,
Sokoban `[1,14,22,28,53]`, RubiksCube `[9]×6`, Connector 6–11, Sudoku `[1,203,224,…]`, Minesweeper
`[3,246,221,…]`.

Only the **default** appearance is claimed to reproduce the reference, and that is measured at 0/255. A
randomised palette is a domain randomisation, not a reference frame. This is the same distinction
MiniGrid's per-subpixel label decomposition exists to erase; it is not erased here, and saying so is
cheaper than a decomposition that would have to model matplotlib's blending.

---

## Reproducing

```bash
# clean venv — NOT --system-site-packages, which breaks jax's CUDA plugin discovery
python -m venv ~/venvs/se-jumanji
~/venvs/se-jumanji/bin/pip install jumanji "jax[cuda12]" torch numpy scipy pytest matplotlib

srun ... ~/venvs/se-jumanji/bin/python scratch/jumanji/probe1b_rgb.py      # what the reference is
srun ... ~/venvs/se-jumanji/bin/python scratch/jumanji/probe6_all.py       # the gate table
srun ... ~/venvs/se-jumanji/bin/python scratch/jumanji/probe7_purity.py    # G1, non-grid leaves
srun ... ~/venvs/se-jumanji/bin/python scratch/jumanji/build_atlases.py    # warm the cache
srun ... ~/venvs/se-jumanji/bin/python -m pytest tests/gpu -v -rs -k Jumanji
```
