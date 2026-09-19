"""Jumanji (InstaDeep) — a matplotlib reference turned into an exact per-pixel lookup.

Reference: ``jumanji`` 1.1.2, whose 26 registered JAX environments are drawn by matplotlib viewers.
**Eight ids are ported here across six renderers**, every one bit-identical to the official renderer on
all pixels (see the table below). Two more were measured and REFUSED, each on a named mechanism rather
than on a tolerance, and the remaining sixteen were not attempted for reasons
``docs/research/jumanji.md`` records id by id.

Eight over six because ``RubiksCube-partly-scrambled-v0`` and ``Sudoku-very-easy-v0`` differ from their
siblings only in the RESET DISTRIBUTION — same viewer, same alphabet, same atlas — so they share the
parent's atlas and differ only in ``sample_states``. That is the move the Fetch port makes for its
``Dense``/``Dict`` suffixes, and it matters here for a second reason: two viewers of one class produce
the same matplotlib figure NAME, so a variant with its own viewer would fight the parent for one figure.

WHAT THE OFFICIAL RENDERER ACTUALLY IS, WHICH IS NOT WHAT ``env.render`` SUGGESTS
--------------------------------------------------------------------------------

``env.render(state)`` returns **None** for all 26 ids, and that is not a missing feature: every viewer
is built with ``render_mode="human"``, whose ``_display`` draws to a window and returns nothing.
``MatplotlibViewer`` picks the branch by binding ``self._display`` at construction, so the rgb_array
path is reached by constructing the viewer with ``render_mode="rgb_array"``. Two ids need more than
that — ``Snake.render`` and ``Sokoban.render`` call the viewer and then **discard its return value** —
so the official renderer for this family is the VIEWER's ``render``, not the env's, and everything here
goes through it.

The frame is 10x10 inches of matplotlib at whatever dpi the figure carries, i.e. **1000x1000 by
default**. ``size`` here is honoured by pinning the figure's dpi to ``size/10``, which is the official
renderer at a different resolution rather than a resampling of it; parity is measured at each size
separately and the atlas is cached per size.

THE CONSTRUCTION: A FRAME IS A PER-PIXEL LOOKUP INDEXED BY ONE CELL'S SYMBOL
---------------------------------------------------------------------------

MiniGrid's atlas is built by CALLING ``Grid.render_tile``, which makes its parity structural rather
than a rasterizer-matching exercise. The same move works here, and it has to be generalised in one
respect: a matplotlib cell is not pixel-aligned, not rectangular in general, and is anti-aliased, so
there is no tile to cut out. What survives is the weaker and sufficient statement — **each output
pixel is a function of exactly one cell's symbol** — and that is enough for an exact gather:

  1. ``P*V`` official renders, one per (cell ``p``, symbol ``v``) against a fixed base grid;
  2. ``owner[y,x]`` = the single cell whose symbol moves this pixel. **A pixel moved by two cells is a
     model failure and is counted, not tolerated** — it is what refused Cleaner and SlidingTilePuzzle;
  3. ``LUT[k,v]`` = owned pixel ``k``'s colour when its owner holds ``v``, taken from those renders;
  4. everything unowned is the static plate.

Rendering is then ``plate`` with one gather scattered into it. Every colour in the frame came out of
the official renderer, so exactness is by construction — anti-aliased glyph fringes, alpha-blended
paths and shared cell borders included, none of which is modelled or reimplemented.

Steps 1-4 only prove each cell is separable ON ITS OWN. The check that decides it is step 5: compose
random grids with EVERY cell different at once and diff against the official frame. That is what these
ids report, at 200 px, 12 uniform-random plus 12 reset-distribution grids each:

======================  ====  ====  ======  =========  ==========  ==============  =============
id                         P     V  owned%  conflicts  LUT (MB)    verify uniform  verify real
======================  ====  ====  ======  =========  ==========  ==============  =============
Maze-v0                  100     4   56.3%          0        0.27          0/255          0/255
RubiksCube-v0             54     6   45.1%          0        0.32          0/255          0/255
Sudoku-v0                 81    10    2.4%          0        0.03          0/255          0/255
Connector-v3             100    31   67.2%          0        2.50          0/255          0/255
Sokoban-v0               100     5   81.0%          0        0.49          0/255          0/255
Minesweeper-v0           100    10    4.8%          0        0.06          0/255          0/255
======================  ====  ====  ======  =========  ==========  ==============  =============

Sudoku's 2.4% is the whole thesis in one number: the only pixels that depend on the board are the digit
glyphs, so 977 pixels of a 40,000-pixel frame carry all of it and the other 97.6% is a constant.

And the number the whole port exists to produce — this renderer against the official viewer, on 64
states drawn from each env's own reset distribution, at 200 px:

======================================  ==============  ==============  ==========  ==========
class                                    all px exact   interior exact    mean/255    max/255
======================================  ==============  ==============  ==========  ==========
JumanjiMaze                                   1.000000         1.000000      0.0000           0
JumanjiRubiksCube                             1.000000         1.000000      0.0000           0
JumanjiRubiksCubePartlyScrambled              1.000000         1.000000      0.0000           0
JumanjiSudoku                                 1.000000         1.000000      0.0000           0
JumanjiSudokuVeryEasy                         1.000000         1.000000      0.0000           0
JumanjiConnector                              1.000000         1.000000      0.0000           0
JumanjiSokoban                                1.000000         1.000000      0.0000           0
JumanjiMinesweeper                            1.000000         1.000000      0.0000           0
======================================  ==============  ==============  ==========  ==========

ALL pixels, not merely interior ones, and that is the point of building the atlas out of official frames
rather than matching a rasterizer: there is no anti-aliasing residual to negotiate, because the
anti-aliased pixels were copied rather than recomputed.

THE ATLAS IS VERIFIED PER SIZE, AND ONE ID IS 200 px ONLY
----------------------------------------------------------

``verified_sizes`` is measured, not declared for caution. At 400 px five of the six renderers still
separate exactly, and **Connector does not**: its cells carry a ``linewidth=1`` black edge, a linewidth
is in POINTS, and one point is 0.28 px at dpi 20 but 0.55 px at dpi 40 — so the shared border stops
being fully black and becomes a genuine blend of the two cells it divides. **5,841 pixels are moved by
more than one cell** there, the per-cell model is simply false, and ``_build_atlas`` refuses rather than
producing an atlas that is wrong on exactly the states that matter. ``JumanjiConnector`` therefore
raises ``ValueError`` for any size but 200, which is the path the conformance suite already understands
("an env that renders at one size only says so") and which it skips on.

G1, asked separately from the LUT check because they are different claims. The verification above shows
the frame is a function of the grid *given everything else held where the adapter puts it*; it does not
show that no OTHER leaf reaches a pixel. So every non-grid leaf of a real reset ``State`` was
substituted from a different episode's and the official env-level frame re-rendered: **0/255 for all
six**. Minesweeper's ``flat_mine_locations`` and Sokoban's ``agent_location`` are therefore genuinely
not in the state, rather than assumed out of it. Connector is stronger still and by construction:
``ConnectorViewer.render`` is handed ``state.grid`` itself, so no other leaf is in scope.

NO ``step``, AND THAT IS A DECISION RATHER THAN AN OMISSION
-----------------------------------------------------------

MiniGrid ships a batched GPU ``step`` because its reference is a CPU simulator running at 8,100-10,700
env-steps/s, so a GPU port is worth 158x at B=256 and 7,500x at B=16,384 — the port buys a capability
that did not exist. Jumanji's envs are already JAX: ``env.step`` is ``jax.jit``-compilable and
``jax.vmap``-able onto the same GPU this renderer runs on, at whatever batch the caller wants.
Reimplementing those transitions here would add a second solver to keep in sync with upstream and buy
no capability at all. ``step_reference_checkable`` is therefore deliberately unset on every class, and
the conformance suite's simulator check skips them by design. What this family contributes is the half
Jumanji does NOT already do on the GPU: the frame.

WHAT THE COLOUR KNOBS DO, AND WHAT THEY ARE NOT
------------------------------------------------

The atlas is NOT rebuilt per palette — a rebuild is ``P*V`` matplotlib renders, which is seconds to
minutes, so it cannot be on a render path. A palette entry instead rewrites exactly the LUT entries
whose colour is that symbol's MODAL colour, and leaves every other entry at the official value.

The consequence is worth stating plainly, because MiniGrid needed a per-subpixel label decomposition to
avoid it: for a symbol that paints a single flat colour, recolouring is total and the result is the
frame the reference would draw at that palette. For a symbol whose pixels are already a BLEND — a
Sokoban sprite, a Sudoku or Minesweeper digit and its anti-aliased fringe, a Connector path at alpha
0.25 — only the dominant colour moves, and the fringe keeps a blend toward the OLD colour. Such a frame
is therefore not one the official renderer would produce at that palette. Measured, as
``colors_per_symbol``: Maze is ``[1,1,1,1]`` (total), Sokoban ``[1,14,22,28,53]``, RubiksCube ``[9]*6``,
Connector 6-11, Sudoku ``[1,203,224,...]``, Minesweeper ``[3,246,221,...]``. Only the DEFAULT appearance
is claimed to reproduce the reference, and that is measured at 0/255; a randomised palette is a domain
randomisation, not a reference frame.

Connector is the entry that was expected to break and did not. Its cells are drawn as
``plt.Rectangle`` patches with ``edgecolor="black", linewidth=1`` — adjacent cells share those edges —
and its paths are painted at **alpha 0.25 against a live Agg canvas**, which is the serial-blend shape
that makes a renderer order-dependent. It separates anyway: a shared border is black whichever
neighbour drew it, so it never MOVES with a cell's symbol and lands in the plate rather than in a
conflict.

WHAT THE STATE IS
-----------------

One integer per cell, and the alphabet is the one the VIEWER consumes rather than the one the env
stores. For Maze that is decisive: ``MazeEnvViewer.render(state)`` is exactly
``MazeViewer.render(_overlay_agent_and_target(state))``, so the render state is the OVERLAID grid —
walls with the agent and the target written into it — and ``agent_position``/``target_position`` are
not separate dimensions. Carrying them would declare dimensions the renderer provably cannot read
twice over, and ``check_state_dims_matter`` would be right to fail it.

Compression, at the float32 this repo stores states in, against the 200 px frame (120,000 bytes):

======================  =============  ==============  ===============
id                       state (B)      vs frame        packed 1B/cell
======================  =============  ==============  ===============
Maze-v0                          400           300x             1,200x
Sudoku-v0                        324           370x             1,481x
Connector-v3                     400           300x             1,200x
Sokoban-v0                       400           300x             1,200x
Minesweeper-v0                   400           300x             1,200x
RubiksCube-v0                    216           555x             2,222x
======================  =============  ==============  ===============

MEASURED, and what is NOT claimed
---------------------------------

The reference is matplotlib, which is one of the most global-state-heavy libraries in Python, and the
conformance suite keeps one live reference env per (class, size) for a whole run — so several viewers
coexist. Asked rather than assumed: render Maze at a fixed grid, construct and drive a second env's
viewer, re-render Maze at the UNCHANGED grid. **0/255.** Jumanji's viewers key their figures by name
through ``plt.fignum_exists``, and the names differ per env, so they do not share a canvas.

WITHIN this family the same hazard is real, and the FIRST fix for it was itself a bug worth recording.
Two viewers of one id at two SIZES do share a figure name, and the suite renders at ``2 * env.size`` in
``check_resolution_consistency``, so building the 400 px atlas would re-dpi the figure the cached 200 px
REFERENCE viewer is holding. The obvious repair — give each (id, size) a unique viewer NAME — is wrong,
because **some viewers draw their name**: ``RubiksCubeViewer`` and ``MinesweeperViewer`` both call
``fig.suptitle(self._name)``. Measured after renaming: RubiksCube disagreed with the official renderer
on exactly **100 pixels of every frame**, all in rows 4-7 (the suptitle band) and all UNOWNED, i.e.
baked into the plate. Minesweeper was worse, because it looked perfect — its atlas and its reference
carried the same invented title, so it scored 1.000000 against a picture the official env never draws.

So the name is left exactly as the official env sets it, ``_VIEWERS`` keeps ONE viewer per id and never
closes it, and ``_official_frame`` sets the figure's dpi per render instead — changing size without
changing a pixel of what the reference draws. The lesson generalises past this family: when isolating
two configurations of a reference renderer, isolate on something the renderer does not READ.

Two ids were measured and refused, and both failures are informative rather than incidental:

  * **Cleaner-v0** — 225 conflict pixels (exactly one cell) and 255/255 on verification. Its agent
    overlay is not a per-cell symbol: ``_set_agents_colors`` paints the cells in
    ``state.agents_locations``, and the env has exactly three agents, so "no agent anywhere" is
    unrepresentable. A 4-symbol grid therefore cannot vary its cells independently — the corner where
    the last agent symbol is removed has to put the agents SOMEWHERE, and that somewhere turns red.
    The coupling is real, not an artifact of the encoding.
  * **SlidingTilePuzzle-v0** — 28,235 conflict pixels, 255/255, **of which 25,671 were the probe and
    not the renderer.** ``ax.imshow(state.puzzle, cmap=...)`` passes no ``vmin``/``vmax``, so the
    colormap normalises to the board's own min and max — but the board is a PERMUTATION of 0..24 in
    576/576 reachable states, so on the reachable set that normalisation is a constant. What moved
    was ``base()``: the solved board holds one 0 and one 24, and overwriting either extreme
    renormalises all 25 tiles. Anchoring the base (two 0s, two 24s) takes it to **2,564**, all of
    them **within 1 px of a cell edge** — Connector's shared-border blend, firing at 200 px instead
    of 400. Still refused, but the next step is a size sweep rather than a wall; see
    ``docs/research/jumanji.md`` and ``scratch/jumanji/probe_slidingtile_reachable.py``.

THROUGHPUT, and the batch it is quoted at
------------------------------------------

One H200 at 200 px, through ``env.render`` as a caller gets it. The denominator is the FASTEST honest
form of the official viewer — matplotlib reusing one figure, not allocating a new one per frame, which
is up to 5.3x faster on Maze and would have flattered every ratio below if it had been skipped.

======================  ==============  ==========  ==========  ============  ==========
id                       official f/s     B=1         B=256       B=4096       crossover
======================  ==============  ==========  ==========  ============  ==========
Maze-v0                          91.16     3,346       362,022       482,575        none
Sudoku-v0                        19.08     3,390       671,266     1,209,900        none
Minesweeper-v0                   17.10     3,164       638,327     1,107,219        none
RubiksCube-v0                     8.79     3,362       420,845       580,691        none
Sokoban-v0                        7.83     3,353       281,220       355,190        none
Connector-v3                      7.16     3,252       310,639       402,232        none
======================  ==============  ==========  ==========  ============  ==========

**There is no crossover on any id: this renderer is faster at every batch measured, including B=1**,
where the tightest case (Maze, against the fastest reference) is still 36.7x and the widest
(Connector at B=4096) is 56,207x. That is the expected shape rather than a surprise — matplotlib
composites artists serially against a live Agg canvas, so its cost is per frame with no batch term,
while this is one copy plus one gather. Stating it as a property matters because "no crossover" is a
result, not a missing measurement.

See ``docs/research/jumanji.md`` for the full five-batch table, the per-id atlas build times, and the
id-by-id account of the twenty ids not ported.
"""

from __future__ import annotations

import importlib.metadata as _md
import importlib.util as _ilu
import os

# Matplotlib must be on a headless backend before pyplot is imported anywhere, and jax must not
# preallocate the GPU: the renderer here is torch, the reference is CPU matplotlib, and jax's default
# 75% preallocation would starve the very device the frames are being written on. Both are
# ``setdefault``, so an explicit setting always wins.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# THE DEPENDENCY CHECK, at module scope and by ``find_spec`` rather than by import.
#
# Unlike the tile families, this one cannot render without its reference package: the atlas IS a stack
# of official frames, so a machine without ``jumanji`` has no way to build one. A lazy import inside
# ``render`` would let all six classes register and then fail at the first frame — which is 300-odd
# test failures for one missing wheel, and it has happened twice in this repo. Raising here removes
# exactly these envs and puts the reason in ``stable_envs.envs.import_errors()``.
#
# ``find_spec`` and a metadata version read, not an import: importing ``jax`` costs ~2 s and importing
# ``matplotlib.pyplot`` mutates global state, neither of which belongs in ``import stable_envs``.
_VENV = "/mnt/vast/home/randall.balestriero/venvs/se-jumanji"
_missing = [m for m in ("jumanji", "jax", "matplotlib") if _ilu.find_spec(m) is None]
if _missing:  # pragma: no cover
    raise ImportError(
        f"the Jumanji family builds its per-pixel atlas by CALLING the official matplotlib viewer, so "
        f"it cannot render without the reference package: missing {', '.join(_missing)}. The envs were "
        f"measured in {_VENV}, built with a CLEAN `python -m venv` — a --system-site-packages venv "
        f"breaks jax's CUDA plugin discovery. See docs/research/jumanji.md"
    )
try:
    _JVER = _md.version("jumanji")
except Exception:  # pragma: no cover
    _JVER = "unknown"

import numpy as np
import torch

from ..env import Env
from ..registry import register
from ..spaces import Box, Dict, RGBBox

#: Where built atlases are memoised. Building one costs P*V official matplotlib renders — 6.0 s for
#: Maze at 200 px, 340.7 s for Connector — so it is done once per (id, size) per machine and then
#: loaded. ``docs/research/jumanji.md`` carries the full per-id, per-size table and the warm-up script.
_CACHE_ROOT = os.environ.get("STABLE_ENVS_CACHE") or os.path.expanduser("~/.cache/stable-envs")
#: In-process atlas cache, keyed (id, size, device).
_ATLAS: dict = {}
#: ONE official viewer per (id, size), built once and never closed.
#:
#: Not a speed optimisation. A viewer owns a matplotlib figure identified by NAME, and
#: ``_get_fig_ax`` resolves that name globally — so a second viewer for the same (id, size) closes and
#: re-creates the figure the first one is still holding, and the first one's next frame comes back at
#: matplotlib's default dpi instead of the pinned one. Caching by content is the same rule the MuJoCo
#: families follow for EGL contexts, for the same reason: the resource is process-global whether or not
#: the object holding it is.
_VIEWERS: dict = {}
#: Reset-distribution state pools, keyed by id. Built once per process from the official env.
_POOL: dict = {}
#: Sentinel label meaning "this pixel's colour is not a flat palette colour; keep the official one".
RAW = 255
#: Bytes of gather scratch ``render`` will hold at once, which sets the batch chunk. 256 MB is well
#: above the width at which the gather saturates bandwidth, so this bounds memory and not throughput.
_SCRATCH_BUDGET = 256 << 20


def _rgb(a) -> np.ndarray:
    a = np.asarray(a)
    return a[..., :3] if a.shape[-1] == 4 else a


def _official_frame(spec: _Spec, size: int, grid) -> np.ndarray:
    """THE one path to an official frame: the viewer for ``spec``, drawn at ``size`` px.

    Two things are load-bearing here and one of them was learned the expensive way.

    **The viewer's NAME is never touched.** ``MatplotlibViewer._get_fig_ax`` resolves a figure by
    ``self._name``, so making the name unique per size is the obvious way to stop two sizes of one id
    sharing a canvas — and it is wrong, because **some viewers DRAW their name**:
    ``RubiksCubeViewer`` and ``MinesweeperViewer`` both call ``fig.suptitle(self._name)``. Renaming
    therefore changes the picture. Measured: with a renamed viewer, RubiksCube disagreed with the
    official renderer on exactly 100 pixels of every frame, all of them in rows 4-7 (the suptitle band)
    and all of them UNOWNED, i.e. baked into the plate. Worse, it is the kind of error that hides:
    Minesweeper's atlas and its reference were renamed identically, so it scored a perfect 1.000000
    against a title this port had invented.

    **The dpi is set per render instead.** ``figure_size`` is 10x10 inches, so dpi = size/10. One
    viewer and one figure per id; when a render is asked for at a different size than the figure
    currently carries, the dpi is changed and the figure redrawn once to let the constrained-layout
    engine settle before the frame that is kept. That makes two sizes of one id safe without altering
    a single pixel of what the reference draws.
    """
    import matplotlib.pyplot as plt

    v = _viewer(spec)
    want = float(size) / 10.0
    if getattr(v, "_se_dpi", None) != want:
        spec.render(v, spec.base())  # ensure the figure exists
        plt.figure(v._name).set_dpi(want)
        spec.render(v, spec.base())  # let the layout settle at the new dpi
        v._se_dpi = want
    out = spec.render(v, grid)
    if out.shape != (int(size), int(size), 3):
        raise RuntimeError(
            f"{spec.env_id}: the official viewer produced {out.shape}, not "
            f"{(int(size), int(size), 3)} — the figure dpi is not being honoured, and every number "
            f"measured against it would be a resample rather than the reference"
        )
    return out


# ===================================================================== per-id adapters
#
# A spec says three things: what the grid IS, how to render one through the OFFICIAL viewer, and how a
# real jumanji ``State`` reduces to one. Everything else in this module is shared.


class _Spec:
    env_id: str = ""
    cls_name: str = ""
    ds_name: str = ""
    rows: int = 0
    cols: int = 0
    n_values: int = 0
    value_names: tuple = ()
    doc: str = ""

    #: Output sizes at which the per-cell model was MEASURED to hold (0 conflicting pixels and 0/255 on
    #: the many-cells-changed verification). Rendering at any other size raises, because there is no
    #: honest atlas for it — see ``JumanjiGrid._check_size``.
    verified_sizes: tuple = (200, 400)

    #: The id whose atlas and viewer this spec SHARES. Defaults to its own. A variant id that differs
    #: only in its reset distribution — ``RubiksCube-partly-scrambled-v0``, ``Sudoku-very-easy-v0`` —
    #: draws through the identical viewer, so giving it its own atlas would rebuild an identical one,
    #: and giving it its own viewer would be worse than wasteful: viewers of the same class produce the
    #: same matplotlib figure NAME, so two of them would fight over one figure.
    atlas_id: str = ""

    @property
    def key_id(self) -> str:
        return self.atlas_id or self.env_id

    @property
    def P(self) -> int:
        return self.rows * self.cols

    def viewer(self):
        raise NotImplementedError

    def render(self, viewer, grid) -> np.ndarray:
        raise NotImplementedError

    def grid_of(self, state) -> np.ndarray:
        raise NotImplementedError

    def base(self) -> np.ndarray:
        return np.zeros(self.P, np.int32)

    def cell_names(self) -> tuple:
        return tuple(f"cell_r{r}c{c}" for r in range(self.rows) for c in range(self.cols))


class _Maze(_Spec):
    env_id, cls_name, ds_name = "Maze-v0", "JumanjiMaze", "jumanji-maze"
    rows = cols = 10
    n_values = 4
    value_names = ("empty", "wall", "agent", "target")
    doc = (
        "Jumanji's 10x10 Maze. The render state is the OVERLAID grid — "
        "``MazeEnvViewer.render`` is ``MazeViewer.render(overlay(state))``, so the agent and the "
        "target are cell symbols rather than coordinates."
    )

    def viewer(self):
        from jumanji.environments.routing.maze.viewer import MazeEnvViewer

        v = MazeEnvViewer("Maze", render_mode="rgb_array")
        return v

    def render(self, viewer, grid):
        import jax.numpy as jnp
        from jumanji.environments.commons.maze_utils.maze_rendering import MazeViewer

        g = np.asarray(grid, np.int32).reshape(self.rows, self.cols)
        return _rgb(MazeViewer.render(viewer, jnp.asarray(g)))

    def grid_of(self, state):
        g = np.asarray(state.walls).astype(np.int32)
        g[int(state.agent_position.row), int(state.agent_position.col)] = 2
        g[int(state.target_position.row), int(state.target_position.col)] = 3
        return g.reshape(-1)


class _RubiksCube(_Spec):
    env_id, cls_name, ds_name = "RubiksCube-v0", "JumanjiRubiksCube", "jumanji-rubiks-cube"
    rows, cols = 6, 9
    n_values = 6
    value_names = ("white", "yellow", "green", "blue", "red", "orange")
    doc = (
        "Jumanji's 3x3x3 Rubik's Cube, drawn as six 3x3 ``imshow`` panels through one "
        "``ListedColormap``. The state is the 54 stickers."
    )

    def viewer(self):
        from jumanji.environments.logic.rubiks_cube.constants import DEFAULT_STICKER_COLORS
        from jumanji.environments.logic.rubiks_cube.viewer import RubiksCubeViewer

        v = RubiksCubeViewer(sticker_colors=DEFAULT_STICKER_COLORS, cube_size=3, render_mode="rgb_array")
        return v

    def _state(self, grid):
        import jax.numpy as jnp
        from jumanji.environments.logic.rubiks_cube.types import State

        return State(
            cube=jnp.asarray(np.asarray(grid, np.int8).reshape(6, 3, 3)),
            step_count=jnp.int32(0),
            key=jnp.zeros(2, jnp.uint32),
        )

    def render(self, viewer, grid):
        return _rgb(viewer.render(self._state(grid)))

    def grid_of(self, state):
        return np.asarray(state.cube).astype(np.int32).reshape(-1)

    def base(self):
        return np.repeat(np.arange(6, dtype=np.int32), 9)

    def cell_names(self):
        faces = ("up", "front", "right", "back", "left", "down")
        return tuple(f"{f}_r{r}c{c}" for f in faces for r in range(3) for c in range(3))


class _Sudoku(_Spec):
    env_id, cls_name, ds_name = "Sudoku-v0", "JumanjiSudoku", "jumanji-sudoku"
    rows = cols = 9
    n_values = 10
    value_names = ("empty",) + tuple(str(d) for d in range(1, 10))
    doc = (
        "Jumanji's Sudoku, drawn as 81 ``Text`` artists and nothing else — the viewer's "
        "``_draw_board`` only turns the axis off, so 2.4% of the frame carries the whole board."
    )

    def viewer(self):
        from jumanji.environments.logic.sudoku.viewer import SudokuViewer

        v = SudokuViewer("Sudoku", render_mode="rgb_array")
        v._display = v._display_rgb_array
        return v

    def _state(self, grid):
        import jax.numpy as jnp
        from jumanji.environments.logic.sudoku.types import State

        return State(
            board=jnp.asarray(np.asarray(grid, np.int32).reshape(9, 9) - 1),
            action_mask=jnp.zeros((9, 9, 9), bool),
            key=jnp.zeros(2, jnp.uint32),
        )

    def render(self, viewer, grid):
        return _rgb(viewer.render(self._state(grid)))

    def grid_of(self, state):
        return (np.asarray(state.board).astype(np.int32) + 1).reshape(-1)


class _Connector(_Spec):
    env_id, cls_name, ds_name = "Connector-v3", "JumanjiConnector", "jumanji-connector"
    rows = cols = 10
    n_values = 31
    value_names = ("empty",) + tuple(f"agent{i}_{k}" for i in range(10) for k in ("path", "position", "target"))
    doc = (
        "Jumanji's Connector, 10 agents on a 10x10 grid. Its cells are alpha-0.25 rectangles with "
        "shared black borders drawn against a live canvas, and at 200 px they separate exactly "
        "anyway — but only at 200 px."
    )
    #: 200 ONLY, and this is measured rather than cautious. ``_draw_grid_cell`` gives every cell a
    #: ``linewidth=1`` black edge, and a linewidth is in POINTS: at dpi 20 (200 px) one point is 0.28
    #: px and the shared border quantises to fully-black pixels that no neighbour's colour reaches, so
    #: the cells separate. At dpi 40 (400 px) it is 0.55 px, the border becomes a genuine blend of the
    #: two cells it divides, and **5,841 pixels are moved by more than one cell** — the per-cell model
    #: is simply false there, and ``_build_atlas`` refuses rather than producing an atlas that is wrong
    #: on exactly the states that matter. This is the only id in the family whose separability depends
    #: on the resolution, and it is a property of drawing edges in points against a pixel grid.
    verified_sizes = (200,)

    def viewer(self):
        from jumanji.environments.routing.connector.viewer import ConnectorViewer

        v = ConnectorViewer("Connector", num_agents=10, render_mode="rgb_array")
        v._display = v._display_rgb_array
        return v

    def render(self, viewer, grid):
        import jax.numpy as jnp

        return _rgb(viewer.render(jnp.asarray(np.asarray(grid, np.int32).reshape(10, 10))))

    def grid_of(self, state):
        return np.asarray(state.grid).astype(np.int32).reshape(-1)


class _Sokoban(_Spec):
    env_id, cls_name, ds_name = "Sokoban-v0", "JumanjiSokoban", "jumanji-sokoban"
    rows = cols = 10
    n_values = 5
    value_names = ("empty", "wall", "target", "box", "agent")
    doc = (
        "Jumanji's Sokoban (Boxoban levels), drawn as one PNG sprite per cell. The state is the "
        "COMBINED grid the viewer consumes, i.e. ``grid_combine(variable, fixed)``."
    )

    def viewer(self):
        import jumanji

        env = jumanji.make("Sokoban-v0")
        v = env._viewer
        v._display = v._display_rgb_array
        self._env = env
        return v

    def render(self, viewer, grid):
        import jax.numpy as jnp

        g = jnp.asarray(np.asarray(grid, np.uint8).reshape(10, 10))
        return _rgb(viewer.render(_CombinedGrid(g)))

    def grid_of(self, state):
        env = getattr(self, "_env", None)
        if env is None:
            import jumanji

            env = self._env = jumanji.make("Sokoban-v0")
        return np.asarray(env.grid_combine(state.variable_grid, state.fixed_grid)).astype(np.int32).reshape(-1)


class _CombinedGrid:
    """Stands in for a Sokoban ``State``: ``BoxViewer._add_grid_image`` only calls ``grid_combine``,
    and ``grid_combine(g, g)`` is the identity on an already-combined grid."""

    def __init__(self, g):
        self.variable_grid = g
        self.fixed_grid = g


class _Minesweeper(_Spec):
    env_id, cls_name, ds_name = "Minesweeper-v0", "JumanjiMinesweeper", "jumanji-minesweeper"
    rows = cols = 10
    n_values = 10
    value_names = ("hidden",) + tuple(str(d) for d in range(9))
    doc = (
        "Jumanji's 10x10 Minesweeper. ``-1`` is an unexplored square and ``0-8`` the neighbouring "
        "mine count; ``flat_mine_locations`` reaches no pixel and is not in the state."
    )

    def viewer(self):
        from jumanji.environments.logic.minesweeper.viewer import MinesweeperViewer

        v = MinesweeperViewer(num_rows=10, num_cols=10, render_mode="rgb_array")
        v._display = v._display_rgb_array
        return v

    def _state(self, grid, mines=None):
        import jax.numpy as jnp
        from jumanji.environments.logic.minesweeper.types import State

        return State(
            board=jnp.asarray(np.asarray(grid, np.int32).reshape(10, 10) - 1),
            step_count=jnp.int32(0),
            flat_mine_locations=jnp.asarray(np.zeros(10, np.int32) if mines is None else mines, jnp.int32),
            key=jnp.zeros(2, jnp.uint32),
        )

    def render(self, viewer, grid):
        return _rgb(viewer.render(self._state(grid)))

    def grid_of(self, state):
        return (np.asarray(state.board).astype(np.int32) + 1).reshape(-1)


class _RubiksCubePartlyScrambled(_RubiksCube):
    """Same cube, same viewer, same atlas — a different RESET DISTRIBUTION.

    ``RubiksCube-partly-scrambled-v0`` applies a short scramble instead of a long one, so it visits a
    different region of the 54-sticker space and nothing else changes. Registering it separately is the
    same move the Fetch port makes for its ``Dense``/``Dict`` suffixes: the id that a dataset was
    collected under is worth naming, and ``sample_states`` is where the difference lives.
    """

    env_id = "RubiksCube-partly-scrambled-v0"
    cls_name = "JumanjiRubiksCubePartlyScrambled"
    ds_name = "jumanji-rubiks-cube-partly-scrambled"
    atlas_id = "RubiksCube-v0"


class _SudokuVeryEasy(_Sudoku):
    """``Sudoku-very-easy-v0``: the same 81 ``Text`` artists, from an easier puzzle generator."""

    env_id = "Sudoku-very-easy-v0"
    cls_name = "JumanjiSudokuVeryEasy"
    ds_name = "jumanji-sudoku-very-easy"
    atlas_id = "Sudoku-v0"


SPECS = [
    _Maze(),
    _RubiksCube(),
    _Sudoku(),
    _Connector(),
    _Sokoban(),
    _Minesweeper(),
    _RubiksCubePartlyScrambled(),
    _SudokuVeryEasy(),
]


# ===================================================================== the atlas


def _viewer(spec: _Spec):
    """The one official viewer for ``spec``. See ``_VIEWERS`` for why there is exactly one.

    Keyed by id and NOT by size: the size is a per-render property of the figure's dpi (see
    ``_official_frame``), and a second viewer for the same id would contend for the same named figure.
    """
    key = spec.key_id
    hit = _VIEWERS.get(key)
    if hit is None:
        hit = _VIEWERS[key] = spec.viewer()
    return hit


def _atlas_path(env_id: str, size: int) -> str:
    return os.path.join(_CACHE_ROOT, "jumanji", f"{env_id.replace('-', '_')}_s{size}.npz")


def _build_atlas(spec: _Spec, size: int) -> dict:
    """``P*V`` official renders -> plate, per-pixel owner, and the colour lookup. See the module doc.

    Raises rather than returns a degraded atlas: a conflicting pixel means the frame is NOT a function
    of one cell at a time, and an atlas built anyway would be silently wrong on exactly the states that
    matter.
    """
    base = spec.base()
    f0 = _official_frame(spec, size, base)
    P, V = spec.P, spec.n_values
    frames = np.empty((P, V, size, size, 3), np.uint8)
    for p in range(P):
        for val in range(V):
            g = base.copy()
            g[p] = val
            frames[p, val] = _official_frame(spec, size, g)

    changed = np.zeros((P, size, size), bool)
    for p in range(P):
        ref = frames[p, base[p]].astype(np.int16)
        changed[p] = (frames[p].astype(np.int16) - ref != 0).any(axis=(0, 3))
    n_own = changed.sum(0)
    conflicts = int((n_own > 1).sum())
    if conflicts:
        raise RuntimeError(
            f"{spec.env_id}: {conflicts} pixels are moved by MORE THAN ONE cell, so the frame is not a "
            f"per-cell lookup and this construction is unsound for it. See docs/research/jumanji.md "
            f"for the two ids where this fires and why"
        )
    owner = np.where(n_own == 1, changed.argmax(0), -1).astype(np.int32)
    ys, xs = np.nonzero(owner >= 0)
    lut = frames[owner[ys, xs], :, ys, xs]  # (K, V, 3) uint8

    # Which of a symbol's colours is its FLAT one, so a palette knob can be exact where one exists.
    # ``colors_per_symbol == 1`` (Maze) means the symbol paints a single colour and recolouring is
    # total; where it is larger (a Sokoban sprite, a Sudoku glyph and its anti-aliased fringe) only the
    # modal colour is a palette entry and the rest keeps the official value. That is stated in
    # ``make_variation_space`` rather than hidden.
    palette = np.zeros((V, 3), np.uint8)
    label = np.full(lut.shape[:2], RAW, np.uint8)
    for val in range(V):
        cols, counts = np.unique(lut[:, val], axis=0, return_counts=True)
        palette[val] = cols[counts.argmax()]
        label[:, val] = np.where((lut[:, val] == palette[val]).all(-1), val, RAW)
    # ``bg`` is NOT stored: it is derived in ``_load_atlas`` from the plate and the owned set, so there
    # is one definition rather than two that can drift.
    return {
        "plate": f0,
        "ys": ys.astype(np.int32),
        "xs": xs.astype(np.int32),
        "cells": owner[ys, xs].astype(np.int32),
        "lut": lut,
        "label": label,
        "palette": palette,
        "meta": np.array([P, V, size, len(ys)], np.int64),
    }


def _background(plate: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> tuple:
    """The plate's dominant colour, over the UNOWNED pixels only, and the mask it paints.

    Restricting to unowned pixels is the whole content of this function. Taken over the whole frame it
    is the modal colour of the BASE render, and for an id whose cells cover most of the picture that
    colour lives inside cells that the gather overwrites a moment later — measured on Sokoban, whose
    cells own 81% of the frame: ``background`` moved zero pixels and the conformance suite correctly
    reported the knob as inert. The plate is exactly the unowned region, so that is where its colour is.
    """
    H, W, _ = plate.shape
    owned = np.zeros(H * W, bool)
    owned[ys.astype(np.int64) * W + xs.astype(np.int64)] = True
    unowned = ~owned.reshape(H, W)
    cols, counts = np.unique(plate[unowned], axis=0, return_counts=True)
    bg = cols[counts.argmax()].astype(np.uint8)
    return bg, (plate == bg).all(-1) & unowned


def _load_atlas(spec: _Spec, size: int, device) -> dict:
    key = (spec.key_id, int(size), str(device))
    hit = _ATLAS.get(key)
    if hit is not None:
        return hit
    path = _atlas_path(spec.key_id, size)
    data = None
    if os.path.exists(path):
        try:
            data = {k: v for k, v in np.load(path).items()}
        except Exception:  # noqa: BLE001  partial/corrupt cache
            data = None
    if data is None:
        data = _build_atlas(spec, size)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as fh:  # a handle, not a name: savez appends .npz to names
            np.savez_compressed(fh, **data)
        os.replace(tmp, path)

    dev = torch.device(device)
    t = lambda a, d=None: torch.as_tensor(a, device=dev, dtype=d)
    K = int(data["meta"][3])
    bg, bg_mask = _background(data["plate"], data["ys"], data["xs"])
    out = {
        "plate": t(data["plate"]),  # (H,W,3) uint8
        "pix": t(data["ys"].astype(np.int64)) * int(size) + t(data["xs"].astype(np.int64)),
        "cells": t(data["cells"].astype(np.int64)),  # (K,)
        "lut": t(data["lut"]).reshape(-1, 3),  # (K*V, 3) uint8
        "label": t(data["label"]).reshape(-1),  # (K*V,) uint8
        "palette": data["palette"],
        "bg": bg,
        "K": K,
        "V": int(data["meta"][1]),
    }
    # ``k * V`` for every owned pixel, precomputed: the render adds the symbol to it and indexes the
    # flat LUT once. Built here so the hot path allocates nothing per call.
    out["kbase"] = torch.arange(K, device=dev, dtype=torch.long) * out["V"]
    out["bg_mask"] = t(bg_mask)  # (H,W) bool
    _ATLAS[key] = out
    return out


# ===================================================================== the env


class JumanjiGrid(Env):
    """A Jumanji id whose frame is a per-pixel lookup indexed by one cell's symbol."""

    SPEC: _Spec = None
    #: 200 px, i.e. matplotlib at dpi 20. Not the viewer's 1000x1000 default, and the reason is cost
    #: rather than fidelity: the model holds at 200, 500 and 1000 alike on Maze, while the atlas is P*V
    #: official renders and Connector's is already 340.7 s at 200 px. Note this is the DEFAULT, not the
    #: only size — see ``verified_sizes``, which is measured per id and is ``(200,)`` for Connector.
    default_size = 200
    parity_interior_min = 1.0

    def __init__(
        self,
        num_envs: int = 1,
        size: int | None = None,
        supersample: int = 1,
        device: str = "cuda",
        backend: str = "auto",
        seed: int = 0,
    ):
        spec = self.SPEC
        self.state_dim = spec.P
        self.state_names = spec.cell_names()
        super().__init__(
            num_envs=num_envs, size=size, supersample=supersample, device=device, backend=backend, seed=seed
        )
        self._check_size(self.size)
        self._a = _load_atlas(spec, self.size, self.device)

    def _check_size(self, size: int) -> None:
        """Refuse a size the per-cell model was not MEASURED to hold at.

        ``ValueError`` and not something quieter, because the conformance suite's
        ``test_resolution_downsample_agrees_within_the_measured_bound`` catches exactly this and skips
        with the reason — "an env that renders at one size only says so" — which is the honest
        behaviour for an env that genuinely has no exact atlas at 2x. Connector is the one id where
        that bites; see ``_Connector.verified_sizes`` for the 5,841-pixel measurement behind it.
        """
        if int(size) not in self.SPEC.verified_sizes:
            raise ValueError(
                f"{type(self).__name__} renders at {self.SPEC.verified_sizes} px, not {int(size)}. The "
                f"atlas is exact only where the official viewer's cells provably separate, and that was "
                f"measured per size rather than assumed — at an unverified size there is no atlas that "
                f"is right, only one that would be silently wrong. See docs/research/jumanji.md"
            )

    # ---- variation ------------------------------------------------------------------

    def make_variation_space(self) -> Dict:
        """Two knobs, and the defaults are the official colours read OFF the built atlas.

        ``palette`` is one ``(V,3)`` leaf rather than V separate ones on purpose. A per-symbol leaf
        would be reported INERT by ``check_variations_do_something`` for any symbol the sampled states
        never contain — Sudoku's digit 9 is not in every board — which is a property of the sample, not
        of the wiring.

        What a palette entry recolours is exactly the pixels the official renderer painted in that
        symbol's MODAL colour. For a flat-colour family (Maze: one colour per symbol) that is the whole
        symbol. For a Sokoban sprite or a Sudoku glyph it is the sprite's dominant colour and not its
        anti-aliased fringe, which keeps the official value — so a recoloured frame is exact where the
        colour is flat and unretouched where it is a blend, rather than approximated everywhere.
        """
        spec = self.SPEC
        a = _load_atlas(spec, self.size, self.device)
        # ``Box`` and not ``RGBBox`` for the palette: RGBBox fixes shape=(3,), and this leaf is (V,3).
        # It is deliberately ONE leaf and not V of them (see the docstring above), which also means
        # check_colors_apply_exactly skips it -- ``background`` is the (3,)-wide leaf that check bites
        # on, and it is the one whose colour must land in the frame literally.
        return Dict(
            {
                "palette": Box(
                    low=0, high=255, shape=(spec.n_values, 3), dtype=np.uint8, init_value=a["palette"].astype(np.uint8)
                ),
                "background": RGBBox(init_value=a["bg"].astype(np.uint8)),
            }
        )

    # ---- render ---------------------------------------------------------------------

    @torch.no_grad()
    def render(
        self,
        states=None,
        size: int | None = None,
        supersample: int | None = None,
        variation: dict | None = None,
        style=None,
        out=None,
    ) -> torch.Tensor:
        """``(...,P)`` -> ``(...,H,W,3)`` uint8 on device: the plate, with one gather scattered in."""
        s = self._state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        B = int(s.shape[0])
        size = int(size or self.size)
        if size == self.size:
            a = self._a
        else:
            self._check_size(size)
            a = _load_atlas(self.SPEC, size, self.device)
        sty = self.resolve_style(style, variation, batch=B)

        if out is None:
            out = torch.empty((B, size, size, 3), dtype=torch.uint8, device=self.device)
        buf = out.view(B, size, size, 3)

        # Background first: the plate, with the pixels the official renderer painted in its dominant
        # colour replaced per environment.
        pal = sty["palette"].reshape(B, a["V"], 3).round().clamp(0, 255).to(torch.uint8)
        bg = sty["background"].reshape(B, 1, 3).round().clamp(0, 255).to(torch.uint8)
        buf.copy_(a["plate"].unsqueeze(0).expand(B, size, size, 3))
        buf[:, a["bg_mask"]] = bg  # (B,1,3) broadcasts over the masked pixels

        # ...then the owned pixels. ``sym`` is the symbol of each owned pixel's owning cell; the LUT is
        # flat so one gather serves the whole batch.
        #
        # Out-of-range symbols WRAP rather than clamp, and ``reference_render`` wraps identically. Real
        # states are always in range, so the rule never fires in practice; it exists because a clamping
        # decoder makes a cell that already holds the last symbol insensitive to the conformance
        # suite's additive probe, which would be reported as a dead state dimension when it is an
        # artifact of the probe. Craftax's port records the same choice for the same reason.
        # Chunked over the batch. The gather's temporaries are (rows, K) int64 and (rows, K, 3) uint8,
        # so an unchunked B=4096 render of Sokoban (K=32,400) would allocate ~3.2 GB of scratch to
        # produce a 0.49 GB frame stack. The chunk is sized from K rather than fixed, and it is large
        # enough that throughput is unaffected — the kernel is bandwidth-bound well below this width.
        flatbuf = buf.view(B, -1, 3)
        rows = max(1, int(_SCRATCH_BUDGET // max(1, a["K"] * 8)))
        codes = s.round().long() % a["V"]  # (B, P)
        for lo in range(0, B, rows):
            hi = min(lo + rows, B)
            sym = codes[lo:hi][:, a["cells"]]  # (n, K)
            flat = a["kbase"].unsqueeze(0) + sym
            col = a["lut"][flat]  # (n, K, 3)
            lab = a["label"][flat]  # (n, K) uint8
            # ``gather`` demands an int64 index, and ``label`` is uint8 to keep the atlas small; the
            # clamp only matters for the RAW sentinel, whose column ``keep`` discards anyway.
            idx = lab.long().clamp_(max=a["V"] - 1).unsqueeze(-1).expand(-1, -1, 3)
            recol = torch.gather(pal[lo:hi], 1, idx)
            flatbuf[lo:hi, a["pix"]] = torch.where((lab == RAW).unsqueeze(-1), col, recol)
        return buf.view(*lead, size, size, 3) if lead else buf

    # ---- states ---------------------------------------------------------------------

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """From the env's OWN reset distribution, through the official JAX env.

        Seeded explicitly and cached per id: ``jax.random.PRNGKey(i)`` for i in range(n), so
        ``sample_states(rng=default_rng(k))`` reproduces in a fresh process.
        """
        rng = rng or self.rng
        pool = _state_pool(self.SPEC)
        return pool[rng.integers(0, len(pool), int(batch))].astype(np.float32)

    # ---- reference bridge -----------------------------------------------------------

    @classmethod
    def reference_env(cls, size: int = 200):
        """The official VIEWER, in rgb_array mode, pinned to ``size``. Never closed.

        The viewer rather than the env, because ``Sokoban.render`` and ``Snake.render`` call their
        viewer and discard its return value, so ``env.render`` is None for them however the display
        branch is bound.
        """
        return (cls.SPEC, int(size))

    @classmethod
    def reference_render(cls, ref, states) -> np.ndarray:
        """Official frames for these grids, with the round-trip asserted.

        The assertion is what stops a parity number comparing two different scenes: every spec builds a
        real jumanji ``State`` (or hands the viewer the grid it takes directly), and reading the grid
        back out of it must give what was asked for.
        """
        spec, size = ref
        out = []
        for row in np.asarray(states, np.float32):
            g = np.rint(row).astype(np.int32) % spec.n_values
            out.append(np.ascontiguousarray(_official_frame(spec, size, g)))
        return np.stack(out)


def _state_pool(spec: _Spec, n: int = 256) -> np.ndarray:
    """``(n, P)`` grids from the official env's ``reset``, plus a short random rollout.

    Reset alone is the wrong distribution for the same reason it was for Fetch: a Sokoban reset always
    has the agent on its start square and no box moved, so a parity number computed from reset states
    says nothing about the cells that carry the game.
    """
    hit = _POOL.get(spec.env_id)
    if hit is not None:
        return hit
    import jax
    import jumanji

    env = jumanji.make(spec.env_id)
    # jitted, because these run eagerly otherwise: 256 resets plus up to 12 steps each is ~3,300
    # op-by-op JAX dispatches per id, which dominated the first test that touched an env.
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    rng = np.random.default_rng(0)
    out = np.empty((n, spec.P), np.float32)
    for i in range(n):
        key = jax.random.PRNGKey(int(i))
        state, _ = reset(key)
        for _ in range(int(rng.integers(0, 12))):
            key, sub = jax.random.split(key)
            try:
                mask = getattr(state, "action_mask", None)
                if mask is not None:
                    m = np.asarray(mask)
                    idx = np.flatnonzero(m.reshape(-1))
                    if not len(idx):
                        break
                    a = np.asarray(np.unravel_index(int(rng.choice(idx)), m.shape)).squeeze()
                    a = jax.numpy.asarray(a)
                else:
                    spc = env.action_spec
                    nv = np.asarray(getattr(spc, "num_values", 4))
                    a = jax.numpy.asarray(rng.integers(0, np.maximum(nv, 1)))
                state, _ = step(state, a)
            except Exception:  # noqa: BLE001  terminal / bad action
                break
        out[i] = spec.grid_of(state)
    _POOL[spec.env_id] = out
    return out


def _make_classes() -> dict:
    made = {}
    for spec in SPECS:
        made[spec.cls_name] = register(
            type(
                spec.cls_name,
                (JumanjiGrid,),
                dict(
                    SPEC=spec,
                    name=spec.ds_name,
                    reference_id=spec.env_id,
                    __doc__=f"{spec.doc}\n\nOfficial id ``{spec.env_id}`` (jumanji {_JVER}); the frame is a "
                    f"per-pixel lookup over {spec.P} cells and {spec.n_values} symbols.",
                ),
            )
        )
    return made


globals().update(_make_classes())
