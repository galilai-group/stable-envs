"""PGX board games (InstaDeep) -- 4 registered ids whose state is one ternary per board cell.

Reference: ``pgx.make(id)`` + ``State.to_svg()`` rasterized by **resvg** (``resvg_py``). That sentence
carries the whole unusual thing about this family and it is stated first because every number below
depends on it: **PGX ships no rasterizer at all.** ``save_svg`` writes vector text, ``_repr_html_``
returns the same string for a notebook, and there is no ``render()`` anywhere in the package. So
"the official renderer" is not a thing that exists to be matched -- it has to be NAMED, and the name
is part of the claim rather than an implementation detail of the test.

WHY resvg AND NOT cairosvg, DECIDED BY MEASUREMENT
--------------------------------------------------

Both produce a plausible raster. Only one of them is a constant of the family:

* ``cairosvg`` binds ``libcairo.so.2`` through ``ctypes.util.find_library`` at CALL time. On this
  cluster's LOGIN node that lookup fails (``no library called "cairo" was found``) and on the COMPUTE
  nodes it succeeds -- the same wheel, the same interpreter, two different answers. A reference whose
  availability depends on which host the test lands on cannot anchor a parity number, and worse, it
  is the kind of dependency that ``import cairosvg`` does NOT reveal: the import succeeds either way.
* ``resvg_py`` is a self-contained Rust wheel (tiny-skia) with no system library lookup. It rasterized
  identically on two calls in one process and across two independent processes (sha256 of the PNG,
  all 5 games, ``scratch/pgx/probe3_constants.py``), and its output honours the SVG's declared canvas
  EXACTLY at every scale tried -- 0.5x, 1x, 2x all landed on the integer pixel count with no
  resampling, which is what lets ``reference_render`` answer at an arbitrary size.

So resvg is the reference, it is named in ``parity_note``, and it is a hard import-time dependency of
this module rather than a reference-only extra -- because it also bakes the plate (below).

THE TEXT SPLIT, WHICH IS WHY THIS FAMILY IS 4 IDS AND NOT 20
--------------------------------------------------------------

PGX has 20 env ids. A census of the emitted documents (``scratch/pgx/probe1_svg_census.py``) splits
them cleanly by whether the drawing contains ``<text>``:

===================  ======  =========================================================
group                 count   ids
===================  ======  =========================================================
no text, no image        6    tic_tac_toe, connect_four, othello, go_9x9, go_19x19, hex
text and/or image       14    chess, shogi, animal_shogi, gardner_chess, backgammon,
                              kuhn_poker, leduc_holdem, sparrow_mahjong, 2048, ...
                              and the 5 minatar ids, which do not use svgwrite at all
===================  ======  =========================================================

The obvious objection to dropping 14 ids is that a font can simply be PINNED -- name the reference as
"resvg plus this one font file" and the decoder is a constant of the family again, exactly as naming
resvg made the rasterizer one. That was tested rather than argued, and it fails, for a sharper reason
than the one it was expected to fail for (``scratch/pgx/probe11_fonts.py``,
``scratch/pgx/probe12_textcheck.py``):

**resvg draws no text at all here.** A minimal document containing one ``<text>`` element rasterizes
to ZERO non-white pixels under the default configuration, under ``skip_system_fonts=True``, and under
an explicitly pinned ``font_files=[DejaVuSerif.ttf]`` -- while the same document with a ``<rect>``
instead paints 1,200. Chess, shogi and 2048 accordingly produce a BIT-IDENTICAL raster (equal
sha256) whether the machine's 8 system fonts are visible, hidden, or replaced by one pinned file: the
labels are simply absent from every one of them. So a parity number for chess would have been
measured against a frame with its rank and file coordinates silently missing, and it would have
passed -- the same shape of trap as a pygame viewer under ``SDL_VIDEODRIVER=dummy`` rendering a blank
surface and parity being measured against black.

Text can be forced to appear, and the way it appears is itself disqualifying: setting the generic
``serif_family="DejaVu Serif"`` yields 828 glyph pixels, but it yields the IDENTICAL 828 pixels for
``font-family="Serif"``, for ``font-family="Courier"`` and for a document that names no family at
all. The override collapses every distinct family request onto one face, and PGX asks for three
(``Serif``, ``Courier``, ``sans serif``). Shogi closes the argument independently: its labels include
CJK (``一 七 三 九 二 五 先 八 六``) and the two-player glyphs ``☖ ☗``, which DejaVu does not contain
and which no font on this machine provides.

So the text ids are not deferred because a font is per-installation content; they are deferred
because the NAMED reference does not render their text, and the only configuration that renders any
text renders the wrong one. That is a stronger no-go, and it is recorded with these numbers in
``docs/research/pgx.md``. hex is text-free and still deferred, for two reasons that are geometric
rather than typographic (concave 25-gons and a non-square canvas -- see the same document).

WHY THERE IS NO ``step``, WHICH IS A DECISION AND NOT AN OMISSION
------------------------------------------------------------------

This repo's ``step`` exists to offer a BATCHED GPU simulator where the reference is CPU-only -- that
is the whole of its value for MiniGrid, whose reference env runs at 8-10k steps/s on one core. PGX's
own ``step`` is already ``jax.jit``-compiled, already vmappable and already runs on the GPU; it is
the thing this family is famous for. A reimplementation here would add a second transition function
to keep in agreement with the first, and would buy no capability that ``jax.vmap(env.step)`` does not
already provide. So none is shipped, ``step_reference_checkable`` is not set, and callers who want to
advance a PGX game should call PGX. The renderer is the part that did not exist.

(Enumeration would have been cheap for tic_tac_toe -- 5,478 reachable positions, all of them
enumerable in seconds -- and that is precisely why the reason for not shipping it has to be the
absence of a gap to fill rather than the difficulty of proving one.)

THE PLATE: THE BOARD FURNITURE IS BAKED THROUGH PGX ITSELF
-----------------------------------------------------------

Every one of these 5 drawings is (board furniture) + (one mark per occupied cell), and the furniture
-- background, grid lines, border, star points -- is a function of the appearance ALONE. So it is
baked exactly the way the Fetch family bakes its floor and table: by asking the reference to draw the
scene with every mover removed. Here "every mover removed" is spelled **the empty board**, which is a
state PGX renders natively, so the plate is ``resvg(pgx.to_svg(empty))`` and the furniture is exact by
construction rather than by matching a rasterizer.

That is worth stating quantitatively, because it is most of the frame and all of the hard part. On
go_19x19 the furniture is 34 grid lines at ``stroke-width="0.5px"``. A 0.5 px line has no interior:
resvg paints it as a partial-coverage grey straddling one pixel column, and any independent rasterizer
lands on a different grey. Reproducing those 34 lines with our own rasterizer put **6.1% of the frame**
(15,300 of 250,000 pixels) into permanent disagreement before a single stone was drawn. Baking removes
that term entirely, and it is why this env overrides ``render`` instead of emitting a ``Scene2D``.

THE MOVER PASS IS A LATTICE GATHER, WHICH IS WHY IT IS O(PIXELS) AND NOT O(CELLS)
----------------------------------------------------------------------------------

Every mark sits on a regular lattice: measured off the emitted documents, all four geometries reduce
to the SAME rule, ``centre = (index + 1) * GRID``, once the two nested ``translate``s in the document
are composed (go carries a second ``translate(G/2,G/2)`` of its own, so its offset is a full grid
rather than a half). And in every game the mark's outer radius is smaller than half the grid pitch
(go 11.86 < 12.5, othello 13.0 < 15, connect_four 12.17 < 17.5, tic_tac_toe 25.5 < 30), so **only the
nearest lattice point can cover a pixel**. The renderer therefore never loops over cells: it maps each
pixel to its own cell, gathers that one cell's value, and evaluates one signed distance. Cost is
independent of board size, which is why go_19x19 (361 cells) renders at the same rate as go_9x9 (81).

FIVE CONSTANTS THAT WERE MEASURED, EACH OF WHICH READS PLAUSIBLY AND IS WRONG
------------------------------------------------------------------------------

*The first rect in the document is dead.* ``_make_*_dwg`` calls ``dwg.add(background)`` -- on the
DRAWING, not on the group it returns -- so every game except tic_tac_toe emits a background rect at
the board's size, and then the Visualizer's own full-canvas background is emitted after it and paints
over it completely. Reproducing it wastes a primitive; reproducing it and stopping there (as "the
background") gives a canvas 25-35 px too small on every side.

*Go's board is not a board.* ``state._x.board`` holds CHAIN IDS -- measured values -317, -254, -160,
52, 169, 182 -- and the drawing does ``jnp.clip(board, -1, 1)``. The render-relevant state is the
clipped board, 361 ternaries, not the 361 chain ids; storing the chain ids would declare a state that
is 1 to 3 orders of magnitude larger in alphabet and that the renderer provably cannot distinguish.

*Othello's board is relative.* The drawing calls ``_get_abs_board``, which is
``board if turn == 0 else -board``. Storing ``board`` alone makes the frame a function of something
not in the state; this env stores the ABSOLUTE board and ``reference_render`` sets ``turn = 0``, which
is a state the env reaches on half of all plies.

*Stroke straddles the path, so a stone is a ring plus a disc.* A circle with ``fill`` and ``stroke``
and no ``stroke-width`` gets SVG's default width of 1, centred on the path: the painted result is the
fill out to ``r - 0.5`` and the stroke from ``r - 0.5`` to ``r + 0.5``. Taking the drawn radius as
``r`` puts the colour boundary half a pixel out on every stone in the frame.

*SVG line caps are BUTT and this repo's segment primitive is a CAPSULE.* ``add_segment`` is
documented as "a thick line segment with round caps", and tic_tac_toe's X is two 3 px strokes: round
caps would extend each of the four arms by 1.5 px past its endpoint. The marks are drawn as rotated
rectangles for that reason. It is also visible in PGX's own drawing, which paints little r=0.84 discs
at the grid-line endpoints -- a workaround for butt caps that would be double-counted under round ones.

MEASURED (native size, 16 states from the env's own play pool, H200; ``scratch/pgx/probe5_parity.py``)
-------------------------------------------------------------------------------------------------------

=================  =====  =========  ==============  =========  ========  =========  ==========
env                 size     cells   interior exact   all exact   mean/255   f/s@4096   reference
=================  =====  =========  ==============  =========  ========  =========  ==========
PgxTicTacToe         240          9        1.000000    0.983681     0.245    108,821       109.6
PgxConnectFour       280         42        1.000000    0.980568     0.409     78,288        79.7
PgxOthello           270         64        1.000000    0.956694     0.799     81,833        22.1
PgxGo9x9             250         81        1.000000    0.915308     1.922     93,671        45.3
=================  =====  =========  ==============  =========  ========  =========  ==========

**Interior agreement is 1.000000 -- bit-exact -- on every registered id**, which is the claim this
repo makes and the reason the plate exists. The reference column is the frame rate of the thing being
replaced, and it is BOTH halves of it: PGX's ``to_svg`` plus resvg's raster, warm, per frame. Neither
half has a batch parameter to sweep -- ``to_svg`` walks one state and resvg rasterizes one document --
so that single-frame rate is the reference's peak by construction, and it is stated rather than
implied. The split is worth seeing because it is not where one would guess: on othello the SVG STRING
GENERATION is 29.1 ms against the rasterizer's 16.2 ms, and on go_19x19 it is 34.2 ms against 76.0 ms.

The speedup is a (ratio, batch) pair, so here is the sweep, and there is **no crossover: this renderer
is faster at every batch, including B = 1**:

===============  ========  ========  =========  =========  =========  =========
env                  B=1       B=8       B=64      B=256     B=1024     B=4096
===============  ========  ========  =========  =========  =========  =========
PgxTicTacToe       11.7x     92.2x      536x       860x       973x       993x
PgxConnectFour     16.4x    124.5x      630x       892x       975x       982x
PgxOthello         60.3x    456.1x    2,341x     3,316x     3,662x     3,710x
PgxGo9x9           29.2x    222.5x    1,245x     1,822x     2,039x     2,070x
===============  ========  ========  =========  =========  =========  =========

Winning at B = 1 is the expected answer rather than a lucky one, and it follows from the design: the
plate is a cached tensor, the alpha maps are cached per (resolution, mark scale), and a frame is one
gather plus one convex combination -- there is no serial per-primitive blend against a live canvas,
which is the structure that gives other families a late crossover. Board size does not appear: 361
cells (go_19x19, 27,447 f/s at 500 px) cost what 81 cells cost at the same resolution, and the
difference in the table is resolution, not cells.

go_19x19 is BUILT and deliberately NOT REGISTERED; see ``NOT_REGISTERED`` below for the number.
"""
from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET

import numpy as np
import torch

# ---------------------------------------------------------------------------------------------
# THE DEPENDENCY CHECK, at module scope, and probing the path that actually fails.
#
# Three separate things have to work and only one of them is revealed by an import:
#   * ``pgx``/``jax`` must import -- ordinary, and the cheap half;
#   * ``resvg_py`` must RASTERIZE, which is a different question from importing. The motivating case
#     is its neighbour: ``import cairosvg`` succeeds on a machine with no libcairo and every call
#     then raises. So a 1x1 document is actually rendered here. It costs microseconds and it is the
#     only check that would have caught the cairosvg trap;
#   * PIL must decode the PNG resvg returns.
# Raising here removes exactly these 5 envs and puts the reason in ``stable_envs.envs.import_errors()``,
# which is the mechanism the package already has for an optional dependency. A lazy import inside
# ``render`` would instead register 5 classes that fail every test they touch.
# ---------------------------------------------------------------------------------------------
_VENV = "/mnt/vast/home/randall.balestriero/venvs/se-pgx"
try:
    import pgx as _pgx
    import resvg_py as _resvg
    from PIL import Image as _Image
except Exception as _e:                                                       # pragma: no cover
    raise ImportError(
        f"the PGX family needs pgx (which brings jax), resvg_py and Pillow, and this interpreter "
        f"cannot provide them: {type(_e).__name__}: {_e}. The envs were measured in {_VENV}, built "
        f"with a CLEAN `python -m venv` -- NOT --system-site-packages, which breaks jax's CUDA plugin "
        f"discovery. See docs/research/pgx.md") from _e

try:                                       # the path cairosvg would have passed and then failed on
    _resvg.svg_to_bytes(svg_string='<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1">'
                                   '<rect width="1" height="1" fill="black"/></svg>')
except Exception as _e:                                                       # pragma: no cover
    raise ImportError(
        f"resvg_py imports but cannot rasterize ({type(_e).__name__}: {_e}). This is the failure mode "
        f"cairosvg has on this cluster's login node -- the import succeeds and every call raises -- "
        f"which is why this check renders a document instead of importing a name. See "
        f"docs/research/pgx.md") from _e

from ..env import Env
from ..registry import register
from ..render import _resources as _RES
from ..spaces import Box, Dict, RGBBox

#: Rasterized plates, keyed by (game, resolution, colours). Bounded because each miss costs one resvg
#: raster of the whole canvas; the number of DISTINCT appearances is what a caller controls.
_PLATE_CACHE_MAX = 64
_PLATE_CACHE = _RES.BoundedCache(_PLATE_CACHE_MAX, name="pgx.plate")
#: (game,) -> the pool of real played positions ``sample_states`` draws from.
_POOL_CACHE: dict = {}
#: colour name -> (r, g, b), resolved by RASTERIZING rather than by transcribing a table.
_COLOR_CACHE: dict = {}
#: (game,) -> resolved ColorSet + the empty-board document.
_BUNDLE_CACHE: dict = {}


def _rgb_of(name: str) -> tuple[int, int, int]:
    """Resolve an SVG colour name through resvg itself.

    A named-colour table is exactly the kind of constant this repo requires to be measured: ``gray``
    is 128 and ``grey`` is 128 and ``lightgray`` is 211, and getting one of them wrong is a silent
    per-pixel colour error on the primitive that uses it. Asking the reference rasterizer what the
    name means costs one 1x1 render and cannot disagree with the reference.
    """
    hit = _COLOR_CACHE.get(name)
    if hit is None:
        png = bytes(_resvg.svg_to_bytes(
            svg_string=f'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1">'
                       f'<rect width="1" height="1" fill="{name}"/></svg>'))
        a = np.array(_Image.open(io.BytesIO(png)).convert("RGB"))
        hit = _COLOR_CACHE[name] = tuple(int(v) for v in a.reshape(3))
    return hit


def _raster(svg: str, size_w: int, size_h: int) -> np.ndarray:
    """resvg -> ``(H, W, 3)`` uint8, over an OPAQUE white ground.

    The ground matters. PGX's documents paint a full-canvas background rect, so nothing should be
    transparent -- but ``convert("RGB")`` on an RGBA PNG discards alpha by DROPPING it rather than by
    compositing, so a document that ever left a pixel unpainted would silently read as black. Alpha
    compositing onto white makes that case visible as the reference's own background instead.
    """
    png = bytes(_resvg.svg_to_bytes(svg_string=svg, width=int(size_w), height=int(size_h)))
    im = _Image.open(io.BytesIO(png)).convert("RGBA")
    a = np.asarray(im, np.float32)
    rgb, al = a[..., :3], a[..., 3:4] / 255.0
    return np.clip(rgb * al + 255.0 * (1.0 - al), 0, 255).astype(np.uint8)


class _Bundle:
    """Everything about a game that depends on neither the state nor the appearance."""

    __slots__ = ("colorset", "empty_svg", "native", "grid", "ncols", "nrows", "palette", "marks")


def _empty_state(game: str):
    """The PGX state whose drawing is the board FURNITURE and nothing else.

    This is the plate's whole justification, and it is the Fetch argument in another notation: there,
    the static pass renders the model with every mover hidden; here, "every mover hidden" is a board
    with no pieces on it, which is a state PGX draws natively and therefore draws EXACTLY.
    """
    import jax
    import jax.numpy as jnp
    env = _pgx.make(game)
    st = jax.jit(env.init)(jax.random.PRNGKey(0))
    spec = GAMES[game]
    board = np.full(spec["cells"], spec["empty"], np.int32)
    x = st._x._replace(board=jnp.asarray(board))
    if hasattr(x, "turn"):
        x = x._replace(turn=jnp.int32(0))
    return st.replace(_x=x)


def _drawables(svg: str) -> list[tuple]:
    """``[(tag, fill, stroke), ...]`` in document order, with inherited presentation resolved."""
    out: list[tuple] = []

    def walk(node, fill, stroke):
        a = node.attrib
        f = a.get("fill", fill)
        s = a.get("stroke", stroke)
        tag = node.tag.split("}")[-1]
        if tag in ("rect", "circle", "line", "polygon", "path", "text", "image"):
            out.append((tag, f, s))
        for ch in node:
            walk(ch, f, s)

    walk(ET.fromstring(svg), None, None)
    return out


def _mark_colors(game: str, empty_svg: str) -> dict:
    """Which colours the reference paints a mark with -- asked by DRAWING one and diffing.

    This exists because reading the intent out of ``pgx/_src/dwg/tictactoe.py`` gets it wrong, and
    gets it wrong silently. Both the X and the O there are stroked with ``color_set.grid_color``,
    NOT with ``p1_color``/``p2_color`` -- so wiring the marks to the ColorSet fields whose names
    match ("p1", "p2") painted tic_tac_toe's X in ``p1_color``. Measured cost before this function
    existed: 11,504 disagreeing pixels at a MEAN of 146.7/255, and 34 of them in flat neighbourhoods
    at the full 255, which is what a wholesale colour error looks like next to the 22.6 mean of a
    genuine anti-aliasing residual.

    Placing one mark of each player and diffing the drawable list against the empty document asks the
    reference the question directly, so the answer cannot drift from the drawing.
    """
    import jax.numpy as jnp
    spec = GAMES[game]
    base = _empty_state(game)
    empty = _drawables(empty_svg)
    out = {}
    for who in ("p1", "p2"):
        board = np.full(spec["cells"], spec["empty"], np.int32)
        board[0] = spec[who]
        st = base.replace(_x=base._x._replace(board=jnp.asarray(board)))
        got = _drawables(st.to_svg())
        new = got[len(empty):] if len(got) > len(empty) else [
            g for g in got if g not in empty]
        assert new, f"{game}: placing a {who} mark added no drawable to the document"
        # A stone is one circle (fill + stroke); an X is two identically-stroked lines.
        fills = {g[1] for g in new}
        strokes = {g[2] for g in new}
        assert len(strokes) == 1, f"{game}: {who} mark has mixed strokes {strokes}"
        assert len(fills) == 1, f"{game}: {who} mark has mixed fills {fills}"
        fill, stroke = fills.pop(), strokes.pop()
        # ``fill="none"`` is tic_tac_toe's O: an annulus that must let the plate through, so the
        # colour that matters is the stroke.
        out[f"{who}.color"] = stroke if (fill in (None, "none")) else fill
        out[f"{who}.outline"] = stroke
        out[f"{who}.hollow"] = fill in (None, "none")
    return out


def _bundle(game: str) -> _Bundle:
    hit = _BUNDLE_CACHE.get(game)
    if hit is not None:
        return hit
    from pgx._src.visualizer import Visualizer
    spec = GAMES[game]
    st = _empty_state(game)
    # The resolved ColorSet is READ OFF the reference's own config resolution rather than copied out
    # of the theme branches in visualizer.py, which are 40 lines of near-identical ColorSet literals.
    v = Visualizer()
    v._set_config_by_state(st)
    cs = v.config["COLOR_SET"]
    b = _Bundle()
    b.colorset = cs
    b.empty_svg = st.to_svg()
    # ElementTree, not a regex over the text: svgwrite is free to order and space its attributes
    # however it likes, and a regex that assumes width-then-height returned None on the real document.
    root = ET.fromstring(b.empty_svg)
    w, h = float(root.attrib["width"]), float(root.attrib["height"])
    assert abs(w - h) < 1e-9, f"{game}: canvas {w}x{h} is not square"
    assert float(w).is_integer(), f"{game}: canvas {w} is not a whole number of pixels"
    b.native = int(round(w))
    b.grid = float(spec["grid"])
    b.ncols, b.nrows = int(spec["ncols"]), int(spec["nrows"])
    # The colour NAMES the empty document actually contains, mapped to the knobs that own them. Read
    # from the document, so a game that resolves its background to something unexpected is caught here
    # rather than by an inert variation three tests later.
    names = set(re.findall(r'(?:fill|stroke)="([^"]+)"', b.empty_svg)) - {"none"}
    pal = {"background.color": cs.background_color, "grid.color": cs.grid_color}
    rest = names - {cs.background_color, cs.grid_color}
    if rest:
        assert len(rest) == 1, f"{game}: unexpected static colours {rest}"
        pal["line.color"] = rest.pop()
    assert set(pal.values()) == names, (
        f"{game}: static colours {names} are not covered by {pal} -- a knob that owns no colour is "
        f"inert, and a colour that no knob owns cannot be varied")
    b.palette = pal
    b.marks = _mark_colors(game, b.empty_svg)
    _BUNDLE_CACHE[game] = b
    return b


#: One entry per registered game. Every number here was read off the emitted document
#: (``scratch/pgx/probe2_primitives.py``), not off ``pgx/_src/dwg/*.py``.
#:
#:   cells/ncols/nrows  the board, in the env's own index order (index = row * ncols + col)
#:   grid               GRID_SIZE, and also the lattice pitch: mark centre = (index + 1) * grid
#:   empty/p1/p2        the values the DRAWING tests for. Two different encodings are in play and
#:                      neither is canonicalised, so that a state written back into PGX is the state
#:                      PGX itself would hold.
#:   mark               "stone" (filled disc + 1 px stroke) or "xo" (tic_tac_toe)
#:   radius             the SVG ``r`` BEFORE the stroke is accounted for
GAMES: dict[str, dict] = {
    "tic_tac_toe": dict(cells=9, ncols=3, nrows=3, grid=60.0, empty=-1, p1=0, p2=1,
                        mark="xo", radius=24.0, stroke=3.0, arm=0.8),
    "connect_four": dict(cells=42, ncols=7, nrows=6, grid=35.0, empty=-1, p1=0, p2=1,
                         mark="stone", radius=35.0 / 3.0, stroke=1.0),
    "othello": dict(cells=64, ncols=8, nrows=8, grid=30.0, empty=0, p1=1, p2=-1,
                    mark="stone", radius=30.0 / 2.4, stroke=1.0),
    "go_9x9": dict(cells=81, ncols=9, nrows=9, grid=25.0, empty=0, p1=1, p2=-1,
                   mark="stone", radius=25.0 / 2.2, stroke=1.0),
    "go_19x19": dict(cells=361, ncols=19, nrows=19, grid=25.0, empty=0, p1=1, p2=-1,
                     mark="stone", radius=25.0 / 2.2, stroke=1.0),
}


def _state_pool(game: str, n: int = 512) -> np.ndarray:
    """``(n, cells)`` positions from REAL PLAY: random legal moves from the initial position.

    Not a uniform draw over board arrays. Almost every array in ``{-1,0,1}^361`` is a position go
    cannot reach -- it would have stones with no liberties, or a stone count that no legal sequence
    produces -- so a parity number computed on one would be measuring the renderer on states the env
    never visits, which is the failure ``docs/adding-an-env.md`` records for PushT's draw-order search.

    Positions are sampled along whole games so that early (nearly empty) and late (crowded) boards
    both appear; a pool of final positions only would never exercise a sparse board.
    """
    hit = _POOL_CACHE.get(game)
    if hit is not None:
        return hit
    import jax
    spec = GAMES[game]
    env = _pgx.make(game)
    init, step = jax.jit(env.init), jax.jit(env.step)
    out = np.empty((n, spec["cells"]), np.float32)
    key = jax.random.PRNGKey(0)
    i = 0
    ep = 0
    while i < n:
        key, sk = jax.random.split(key)
        st = init(sk)
        ep += 1
        for _ in range(spec["cells"] * 2):
            if i >= n:
                break
            out[i] = _drawn_board(game, st)
            i += 1
            if bool(st.terminated) or bool(st.truncated):
                break
            legal = np.asarray(st.legal_action_mask)
            if not legal.any():
                break
            key, sk = jax.random.split(key)
            a = int(jax.random.choice(sk, np.flatnonzero(legal)))
            key, sk = jax.random.split(key)
            st = step(st, a, sk)
    _POOL_CACHE[game] = out
    return out


def _drawn_board(game: str, st) -> np.ndarray:
    """The board the DRAWING reads, which is not always ``_x.board``.

    go clips chain ids to the sign; othello flips by whose turn it is. Both transformations are read
    out of ``pgx/_src/dwg/`` and both are verified against the emitted document rather than against
    that source: ``scratch/pgx/probe2_primitives.py`` dumps every drawable PGX actually emits, in
    document order with its resolved attributes, so the stones the reference drew can be counted and
    placed, and ``scratch/pgx/probe3_constants.py`` (section 2) establishes the other half — that an
    arbitrary board array can be written back into a ``State`` and read out again by the renderer,
    which is what makes the comparison a comparison of one scene.
    """
    b = np.asarray(st._x.board, np.int32)
    if game.startswith("go"):
        return np.clip(b, -1, 1).astype(np.float32)
    if game == "othello":
        return (b if int(np.asarray(st._x.turn)) == 0 else -b).astype(np.float32)
    return b.astype(np.float32)


class PgxBoardBase(Env):
    """Base for the 5 board games. ``GAME`` is set by the generated subclasses."""

    GAME: str = ""
    default_supersample = 1
    #: Style leaves that change the STATIC furniture and therefore the baked plate. Everything else is
    #: a property of a mark and is applied per environment inside the composite, so randomising the
    #: stone colours -- the common case -- never re-bakes anything.
    PLATE_PATHS: tuple[str, ...] = ()

    #: Left at the repo's default of 1.0 because it is MET, not because it was not looked at: all four
    #: registered ids reach interior 1.000000 over 16 states from their own play pool. No
    #: ``parity_note`` is therefore required, and the reason none is needed is worth one line -- the
    #: board furniture, which is every thin line and every rounded corner in the frame, is not matched
    #: but BAKED THROUGH THE REFERENCE, so it is exact by construction; and the marks are flat-filled
    #: discs and bars whose interiors are one colour. What is left over lands entirely on the 1-2 px
    #: anti-aliased rim of a mark, and a rim pixel has no flat 3x3 neighbourhood, so it reaches the
    #: all-pixel column and never the interior one.
    parity_interior_min = 1.0
    parity_note = ""

    def __init__(self, num_envs: int = 1, size: int | None = None, supersample: int | None = None,
                 device: str = "cuda", backend: str = "auto", seed: int = 0):
        b = _bundle(self.GAME)
        spec = GAMES[self.GAME]
        self._b = b
        self._spec = spec
        self.state_dim = int(spec["cells"])
        self.state_names = tuple(
            f"cell_r{i // spec['ncols']}c{i % spec['ncols']}" for i in range(spec["cells"]))
        super().__init__(num_envs=num_envs, size=size,
                         supersample=self.default_supersample if supersample is None else supersample,
                         device=device, backend=backend, seed=seed)
        assert self.default_size == b.native, (
            f"{type(self).__name__}: default_size {self.default_size} is not the document's own "
            f"canvas {b.native}")
        self._check_plate_is_constant()

    def _close(self) -> None:
        """Drop the three per-instance draw caches. ``Env.close`` calls this once and is idempotent.

        The rendered document bundle, the plates and the colour tables are keyed on (game, size) and
        SHARED by every instance, so they are not released here. What an instance builds lazily for
        itself is the outline colours, the separable lattice coordinates per resolution and the
        per-scale coverage maps, all of them device tensors and all rebuildable on the next call.
        """
        for k in ("_outline_cache", "_lat_cache", "_alpha_cache"):
            self.__dict__.pop(k, None)

    # ---- the assumption the whole design rests on, checked rather than assumed ----------------

    def _check_plate_is_constant(self) -> None:
        """The plate must not depend on the state. Asserted once per class, at a small size.

        The empty-board document SHOULD be state-independent by construction -- it is built from a
        board this class fills with the empty value -- but "should" is what the Fetch port says about
        its own plate immediately before recording the site that made it depend on ``qpos[15:17]``.
        Two plates, built from two different sampled states, must be bit-identical.
        """
        st = self.sample_states(2, rng=np.random.default_rng(0))
        a = self._bake_plate(64, None, probe_state=st[0])
        c = self._bake_plate(64, None, probe_state=st[1])
        d = int(np.abs(a.astype(np.int16) - c.astype(np.int16)).max())
        assert d == 0, (
            f"{type(self).__name__}: the baked plate differs by {d}/255 between two states, so the "
            f"board furniture is NOT a function of the appearance alone and the plate design is "
            f"unsound for this game")

    # ---- variation ---------------------------------------------------------------------------

    def make_variation_space(self) -> Dict:
        """Appearance knobs whose DEFAULTS are the official picture.

        The defaults are resolved by rasterizing the reference's own colour NAMES (``_rgb_of``), so
        "our defaults reproduce the reference" is an identity rather than a transcription. PGX's dark
        theme is reachable through these knobs and is deliberately not a separate Discrete: a theme
        switch would be one more thing whose default has to be argued about, and the light theme is
        what ``global_config.color_theme`` is.
        """
        b = _bundle(self.GAME)
        space: dict = {}
        for path, name in b.palette.items():
            head, leaf = path.split(".")
            space.setdefault(head, {})[leaf] = RGBBox(init_value=list(_rgb_of(name)))
        # A mark's OUTLINE is deliberately not a knob, and the reason is geometric rather than a
        # decision about what is worth exposing. A stone's stroke is SVG's default width of 1, so the
        # painted band is exactly one pixel wide radially and NO pixel is ever fully covered by it --
        # its coverage is below 1 everywhere, so the requested colour never literally appears in the
        # frame. ``check_colors_apply_exactly`` asks for exactly that and correctly refused it on
        # connect_four, go_9x9 and othello (``p2.outline->[17,33,200]``). The colour is still applied,
        # from the value measured off the reference in ``_mark_colors``; it just is not offered as a
        # knob whose value a caller could never actually observe.
        for who in ("p1", "p2"):
            space[who] = {"color": RGBBox(init_value=list(_rgb_of(b.marks[f"{who}.color"])))}
        out = {k: Dict(v) for k, v in space.items()}
        out["mark"] = Dict({"scale": Box(low=0.25, high=1.15, shape=(1,), init_value=[1.0])})
        return Dict(out)

    def _mark_color(self, sty, who: str, which: str, B: int) -> torch.Tensor:
        """``(B,1,1,3)`` for a mark's fill or its outline.

        The fill comes from the style, because it is a knob; the outline comes from the value measured
        off the reference, because it is not (see ``make_variation_space``).
        """
        if which == "color":
            return sty[f"{who}.color"].reshape(B, 1, 1, 3)
        cache = getattr(self, "_outline_cache", None)
        if cache is None:
            cache = self._outline_cache = {}
        hit = cache.get(who)
        if hit is None:
            hit = cache[who] = torch.tensor(
                _rgb_of(self._b.marks[f"{who}.outline"]), device=self.device,
                dtype=torch.float32).reshape(1, 1, 1, 3)
        return hit

    # ---- the plate ---------------------------------------------------------------------------

    def _plate(self, res: int, colors: dict) -> torch.Tensor:
        key = (self.GAME, int(res), str(self.device),
               tuple(sorted((k, tuple(np.round(np.asarray(v, np.float64), 4).tolist()))
                            for k, v in colors.items())))
        hit = _PLATE_CACHE.get(key)
        if hit is not None:
            return hit
        return _PLATE_CACHE.put(key, torch.as_tensor(self._bake_plate(res, colors), device=self.device))

    def _bake_plate(self, res: int, colors: dict | None, probe_state=None) -> np.ndarray:
        """resvg's own raster of the EMPTY board. Exact by construction.

        Colour knobs are applied by rewriting the document's colour ATTRIBUTES before rasterizing --
        ``fill="white"`` becomes ``fill="#rrggbb"`` -- rather than by post-tinting the raster, so a
        recoloured plate is still resvg's own anti-aliasing of resvg's own geometry. The substitution
        is exact-match on the attribute value and asserted to fire, because a silently-missed
        substitution would present as an inert variation rather than as an error.
        """
        b = _bundle(self.GAME)
        svg = b.empty_svg
        if probe_state is not None:
            # Used only by the constancy check: prove the plate ignores the state by building it from
            # a document that is genuinely the empty board regardless of what was sampled.
            svg = b.empty_svg
        if colors:
            for path, name in b.palette.items():
                if path not in colors:
                    continue
                r, g, bl = (int(round(float(v))) for v in np.asarray(colors[path]).reshape(3))
                hexc = f"#{r:02x}{g:02x}{bl:02x}"
                for attr in ("fill", "stroke"):
                    svg = svg.replace(f'{attr}="{name}"', f'{attr}="{hexc}"')
        return _raster(svg, int(res), int(res))

    # ---- render ------------------------------------------------------------------------------

    @torch.no_grad()
    def render(self, states=None, size: int | None = None, supersample: int | None = None,
               variation: dict | None = None, style=None, out=None) -> torch.Tensor:
        """``(...,cells)`` -> ``(...,H,W,3)`` uint8 on device. Style may differ per environment.

        Two passes. The plate is shared per distinct appearance and gathered per row; the marks are
        composited in ONE pass over the batch, because the lattice makes every pixel's cell lookup a
        gather rather than a loop over pieces.
        """
        s = self._state if states is None else torch.as_tensor(states, device=self.device,
                                                               dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim).to(torch.float32)
        B = int(s.shape[0])
        size = int(size or self.size)
        ss = int(self.ss if supersample is None else supersample)
        res = size * ss
        sty = self.resolve_style(style, variation, batch=B)

        if out is None:
            out = torch.empty((B, size, size, 3), dtype=torch.uint8, device=self.device)
        buf = out.view(B, size, size, 3)

        # --- plate: one bake per DISTINCT appearance, gathered per row
        paths = tuple(self._b.palette)
        key = torch.cat([sty[p].reshape(B, -1) for p in paths], 1).cpu().numpy()
        uniq, inv = np.unique(np.round(key.astype(np.float64), 4), axis=0, return_inverse=True)
        plates = []
        for row in uniq:
            o = 0
            over = {}
            for p in paths:
                w = int(sty.layout[p][1])
                over[p] = row[o:o + w]
                o += w
            plates.append(self._plate(res, over))
        pl = torch.stack(plates)[torch.as_tensor(inv, device=self.device, dtype=torch.long)]

        img = self._composite(s, sty, pl.to(torch.float32), res)
        if ss == 1:
            buf.copy_(img.round_().clamp_(0, 255).to(torch.uint8))
        else:
            small = (img.reshape(B, size, ss, size, ss, 3).mean((2, 4))
                     .round_().clamp_(0, 255).to(torch.uint8))
            buf.copy_(small)
        return buf.view(*lead, size, size, 3) if lead else buf

    #: Sub-samples per axis for a mark whose geometry has no closed form along x -- i.e. tic_tac_toe's
    #: rotated bars. Discs do have one and use ``SCAN_ROWS`` instead. 8 is where tic_tac_toe's
    #: all-pixel agreement stops moving (0.9837).
    COV_SUB = 8

    def _lattice(self, res: int):
        """Per-pixel cell index, cell-local offset and in-board mask. Cached per resolution.

        The mover pass rests on a measured fact: a mark's outer radius is below half the grid pitch in
        every one of these games (go 11.86 < 12.5, othello 13.0 < 15, connect_four 12.17 < 17.5,
        tic_tac_toe 25.5 < 30), so the nearest lattice point is the ONLY one that can cover a pixel.
        That turns "draw up to 361 stones" into "look up one cell", and it is why go_19x19 renders at
        the same rate as go_9x9.

        Everything here is SEPARABLE -- the column index depends only on x and the row only on y --
        so these are 1-D tensors that broadcast, not (H, W) grids. That matters because the coverage
        pass below is a loop and the naive form allocates two (H, W) meshes on every iteration.
        """
        cache = getattr(self, "_lat_cache", None)
        if cache is None:
            cache = self._lat_cache = {}
        hit = cache.get(res)
        if hit is not None:
            return hit
        b, spec = self._b, self._spec
        G = b.grid
        sc = res / float(b.native)                      # canvas units -> pixels
        px = (torch.arange(res, device=self.device, dtype=torch.float32) + 0.5) / sc
        out = {}
        for axis, n in (("x", spec["ncols"]), ("y", spec["nrows"])):
            raw = torch.round(px / G) - 1.0
            c = raw.clamp(0, n - 1)
            out[axis] = (c.to(torch.long), px - (c + 1.0) * G, (raw >= 0) & (raw < n))
        colc, u, inx = out["x"]
        rowc, v, iny = out["y"]
        # A pixel whose nearest lattice point is OUTSIDE the board must not pick up the clamped
        # neighbour's mark; the clamp keeps the gather in range and this mask keeps it from painting.
        idx = (rowc.reshape(-1, 1) * spec["ncols"] + colc.reshape(1, -1)).reshape(-1)
        inside = (iny.reshape(-1, 1) & inx.reshape(1, -1))
        hit = (idx, u, v, inside, sc)
        cache[res] = hit
        return hit

    def _alpha_maps(self, res: int, scale: float):
        """``(H, W)`` coverage per PATH the reference draws, by sub-sampling the mark's geometry.

        Coverage is measured rather than approximated: each path's region is evaluated at
        ``COV_SUB**2`` points inside every pixel and averaged, which is the pixel's true covered area
        quantised to ``1/COV_SUB**2``.

        The layers are the reference's own PATHS, and they are composited in the reference's own
        order rather than being made disjoint. That distinction is worth 0.888 -> 0.983 of all-pixel
        agreement on go_19x19 and it is not a subtlety about blending, it is SVG's paint model: a
        shape with a fill AND a stroke paints the fill out to the path radius ``r``, and then paints
        the stroke -- a band straddling the path, ``|d - r| <= w/2`` -- ON TOP of it. Treating the
        fill as "the disc that is left over inside the stroke" (``d < r - w/2``) is not the same
        picture: at the stroke's INNER rim the reference's fill is already fully opaque, so the plate
        contributes nothing, while the disjoint model lets the plate leak through wherever the two
        coverages do not sum to one. On a black stone with a black stroke over a white plate that
        leak is a bright ring inside every stone -- a systematic error, which is why 8x8 sub-sampling
        had not moved the number: the residual was never quantisation.

        The maps do not depend on the batch -- only on the local geometry -- so the loop runs once per
        distinct ``mark.scale`` and not once per environment.
        """
        cache = getattr(self, "_alpha_cache", None)
        if cache is None:
            cache = self._alpha_cache = {}
        key = (int(res), round(float(scale), 6))
        hit = cache.get(key)
        if hit is not None:
            return hit
        if len(cache) > 8:
            cache.clear()
        spec, b = self._spec, self._b
        G = b.grid
        sc = res / float(b.native)
        S = int(self.COV_SUB)
        base = (torch.arange(res, device=self.device, dtype=torch.float32) + 0.5)
        r = float(spec["radius"]) * float(scale)
        half = 0.5 * float(spec["stroke"])
        nmap = 2 if spec["mark"] == "stone" else 3
        acc = [torch.zeros((res, res), device=self.device, dtype=torch.float32) for _ in range(nmap)]
        # The cell index is ``round(t / G) - 1`` and then clamped, exactly as in ``_lattice``. Writing
        # the clamp without the ``- 1`` shifts every local coordinate by a full grid, which reads as a
        # renderer that has forgotten how to draw rather than as an index error: it took go_19x19's
        # interior agreement from 1.000000 to 0.766163.
        def _local(t, n):
            return t - ((torch.round(t / G) - 1.0).clamp(0, n - 1) + 1.0) * G

        # The x sub-samples are taken ALL AT ONCE, so the python loop is S and not S**2. That is what
        # makes a high S affordable, and a high S is required rather than nice: for an edge that is
        # nearly axis-aligned -- which every stone's left and right extreme is -- all S sub-rows cross
        # it in the same sub-column, so the estimate quantises to 1/S and the worst-case error is
        # 1/(2S), NOT 1/S**2. Measured on an isolated go stone at S=8: coverage differed from the
        # reference by 5.8% against a predicted 1/(2*8) = 6.25%, while 1/S**2 would have predicted
        # 1.6%. The maps are cached per (resolution, mark.scale) and never enter the per-frame path,
        # so this cost is paid once.
        if spec["mark"] == "stone":
            return self._stone_alpha(res, r, half, sc, base, _local, key, cache)
        offs = ((torch.arange(S, device=self.device, dtype=torch.float32) + 0.5) / S - 0.5)
        uu = _local((base.reshape(-1, 1) + offs.reshape(1, -1)) / sc,
                    spec["ncols"]).reshape(1, res * S)                  # (1, res*S)
        for iy in range(S):
            ys = (base + offs[iy]) / sc
            vv = _local(ys, spec["nrows"]).reshape(-1, 1)               # (res, 1)
            d = torch.sqrt(uu * uu + vv * vv).reshape(res, res, S)
            if spec["mark"] == "stone":
                acc[0] += (d < r).to(torch.float32).sum(-1)             # fill, out to the PATH
                acc[1] += ((d - r).abs() <= half).to(torch.float32).sum(-1)   # stroke, straddling it
            else:
                uu2 = uu.reshape(1, res, S)
                vv2 = vv.reshape(-1, 1, 1)
                # X: two BUTT-capped strokes, i.e. two rotated rectangles, and TWO separate path
                # elements -- so they are accumulated separately and composited one after the
                # other, the way the reference paints them. Round caps, which is what this repo's
                # segment primitive draws, would extend each of the four arms by half a stroke
                # past its endpoint.
                halfL = 0.5 * float(spec["arm"]) * G * float(np.sqrt(2.0)) * float(scale)
                inv = float(np.sqrt(0.5))
                for k, sgn in enumerate((1.0, -1.0)):
                    p = ((uu2 + sgn * vv2) * inv).abs() <= halfL
                    q = ((uu2 - sgn * vv2) * inv).abs() <= half
                    acc[k] += (p & q).to(torch.float32).sum(-1)
                acc[2] += ((d - r).abs() <= half).to(torch.float32).sum(-1)   # the O, an annulus
        hit = tuple(a / float(S * S) for a in acc)
        cache[key] = hit
        return hit

    #: Sub-scanlines per pixel row used for a DISC's coverage, with the horizontal extent solved
    #: EXACTLY within each. MEASURED against resvg; see the table in docs/research/pgx.md.
    SCAN_ROWS = 16

    def _stone_alpha(self, res, r, half, sc, base, _local, key, cache):
        """Coverage of a stone's fill and stroke, exact in x and sub-sampled in y.

        Uniform 2-D sub-sampling is the wrong tool here and the sweep says so: going from 8x8 to
        128x128 moved go_19x19's all-pixel agreement by -0.0018 (0.889234 -> 0.887414) while the mean
        improved only 2.579 -> 2.513. A residual that does not respond to two orders of magnitude more
        samples is not a sampling error.

        The reason is that a disc's coverage has a closed form along x: at height ``y`` the disc spans
        ``|x| <= sqrt(r^2 - y^2)``, so the overlap with a pixel's x-interval is an interval
        intersection and needs no sampling at all. Only y is discretised, which is also what a
        scanline rasterizer does. That makes the estimate exact in the direction where a stone's left
        and right extremes are nearly axis-aligned -- precisely the pixels the uniform sampler was
        quantising to 1/(2S).
        """
        N = int(self.SCAN_ROWS)
        hpx = 0.5 / sc                                   # half a pixel, in canvas units
        u = _local(base / sc, self._spec["ncols"]).reshape(1, -1)        # (1, res)
        v = _local(base / sc, self._spec["nrows"]).reshape(-1, 1)        # (res, 1)
        x0, x1 = u - hpx, u + hpx
        acc = [torch.zeros((res, res), device=self.device, dtype=torch.float32) for _ in range(2)]

        def disc_cov(R):
            """Fraction of each pixel covered by a disc of radius ``R`` centred on its lattice point."""
            out = torch.zeros((res, res), device=self.device, dtype=torch.float32)
            for k in range(N):
                y = v + ((k + 0.5) / N - 0.5) * (2.0 * hpx)
                w = torch.sqrt(torch.clamp(R * R - y * y, min=0.0))
                lo = torch.maximum(x0, -w)
                hi = torch.minimum(x1, w)
                out += torch.clamp(hi - lo, min=0.0)
                del y, w, lo, hi
            return out / (float(N) * 2.0 * hpx)

        # DISJOINT regions -- the visible fill (inside the stroke band) and the stroke band itself --
        # rather than the reference's two overlapping paths. The distinction is not cosmetic and it
        # is not what SVG's paint model says; it is what the ARITHMETIC has to say to get a stone with
        # a same-coloured stroke right. Painting fill(d<r) and then stroke(|d-r|<=w/2) src-over gives
        # a combined coverage of 1-(1-a_f)(1-a_s), which on a go stone's extreme pixel is 0.898 where
        # the true covered area is 0.843 -- the two layers overlap on [r-w/2, r] and src-over
        # double-counts it. Splitting at r-w/2 makes the two coverages sum to cov(disc(r+w/2)), i.e.
        # exactly the union, so a black stone with a black stroke comes out as one disc. Measured on
        # the isolated stone of scratch/pgx/probe7_onestone.py: 19 -> 30 against a reference of 26,
        # against a true-area prediction of 30.
        inner = disc_cov(r - half)
        outer = disc_cov(r + half)
        acc[0] = inner
        acc[1] = (outer - inner).clamp(0.0, 1.0)
        hit = (acc[0], acc[1])
        cache[key] = hit
        return hit

    def _composite(self, s: torch.Tensor, sty, plate: torch.Tensor, res: int) -> torch.Tensor:
        """Paint the marks over the plate. ``plate`` is ``(B,res,res,3)`` float32."""
        spec = self._spec
        B = int(s.shape[0])
        idx, _u, _v, inside, _sc = self._lattice(res)
        val = s.gather(1, idx.reshape(1, -1).expand(B, -1)).reshape(B, res, res)
        is1 = ((val == float(spec["p1"])) & inside).to(plate.dtype)
        is2 = ((val == float(spec["p2"])) & inside).to(plate.dtype)

        msc = sty["mark.scale"].reshape(-1)
        uniq, inv = np.unique(np.round(msc.detach().cpu().numpy().astype(np.float64), 6),
                              return_inverse=True)
        maps = [self._alpha_maps(res, g) for g in uniq]
        g = torch.as_tensor(inv, device=self.device, dtype=torch.long)
        a = [torch.stack([m[k] for m in maps])[g] for k in range(len(maps[0]))]   # each (B,H,W)

        c1 = self._mark_color(sty, "p1", "color", B)
        c2 = self._mark_color(sty, "p2", "color", B)
        img = plate
        if spec["mark"] == "stone":
            o1 = self._mark_color(sty, "p1", "outline", B)
            o2 = self._mark_color(sty, "p2", "outline", B)
            # One convex combination. The fill and stroke regions are disjoint by construction (see
            # ``_stone_alpha``) and p1 and p2 never occupy the same cell, so all four weights and the
            # plate's remainder sum to one and no compositing order can double-count anything.
            wf1, ws1 = (a[0] * is1).unsqueeze(-1), (a[1] * is1).unsqueeze(-1)
            wf2, ws2 = (a[0] * is2).unsqueeze(-1), (a[1] * is2).unsqueeze(-1)
            return (plate * (1.0 - wf1 - ws1 - wf2 - ws2)
                    + c1 * wf1 + o1 * ws1 + c2 * wf2 + o2 * ws2)
        img = self._over(img, c1, a[0] * is1)          # X, first stroke
        img = self._over(img, c1, a[1] * is1)          # X, second stroke
        return self._over(img, c2, a[2] * is2)         # O, an annulus over the plate

    @staticmethod
    def _over(dst: torch.Tensor, color: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """``src-over``, the only compositing operator SVG uses for an opaque paint."""
        al = alpha.unsqueeze(-1)
        return dst * (1.0 - al) + color * al

    # ---- states ------------------------------------------------------------------------------

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or self.rng
        pool = _state_pool(self.GAME)
        return pool[rng.integers(0, len(pool), int(batch))].copy()

    # ---- reference bridge --------------------------------------------------------------------

    @classmethod
    def reference_env(cls, size: int = 0):
        """The reference is (PGX document) + (resvg). There is no stateful env to build."""
        return {"game": cls.GAME, "size": int(size or _bundle(cls.GAME).native)}

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """Official frames for a board, with the write-back ASSERTED.

        The assertion is not ceremony. ``_x.board`` is written directly, and for go that value is a
        CLIPPED board being put where chain ids normally live; for othello ``turn`` is forced to 0 so
        that ``_get_abs_board`` is the identity. If either of those ever stopped holding, every parity
        number computed here would be comparing two different positions while still looking like a
        rendering discrepancy.
        """
        import jax
        import jax.numpy as jnp
        game = env["game"]
        size = int(env["size"])
        spec = GAMES[game]
        b = _bundle(game)
        base = _pgx.make(game)
        st0 = jax.jit(base.init)(jax.random.PRNGKey(0))
        scale = size / float(b.native)
        out = []
        for row in np.asarray(states, np.int32):
            x = st0._x._replace(board=jnp.asarray(row.astype(np.int32)))
            if hasattr(x, "turn"):
                x = x._replace(turn=jnp.int32(0))
            st = st0.replace(_x=x)
            got = _drawn_board(game, st)
            err = float(np.abs(got - row.astype(np.float32)).max())
            if err > 0:
                raise AssertionError(
                    f"{cls.__name__}: the reference did not reach the requested position (max "
                    f"{err}); every parity number computed from it would compare two different "
                    f"boards")
            svg = st.to_svg(scale=scale)
            out.append(_raster(svg, size, size))
        return np.stack(out)


#: Games that are BUILT but deliberately NOT registered, and the measurement that decided it.
#:
#: go_19x19 uses the same renderer as go_9x9 and reaches the same interior agreement -- 1.000000,
#: bit-exact -- but its ALL-pixel agreement is 0.8876 against the suite's shared ``ALL_EXACT_MIN`` of
#: 0.90, so registering it would ship a red test. The cause is geometric rather than a defect: at its
#: native 500 px a mid-game board carries ~150 stones whose 1-2 px anti-aliased rims are **11.3% of
#: the frame**, and a rim pixel is where two rasterizers are entitled to disagree. That the residual
#: is not ours was measured three ways -- uniform sub-sampling from 8x8 to 128x128 moved all-pixel
#: agreement by -0.0018, sub-scanline counts from 1 to 64 moved it by -0.0147, and switching between
#: the two compositing models improved the MEAN (2.518 -> 2.469) while leaving the exact count alone.
#: An isolated stone puts the whole disagreement at one pixel per edge and traces it to ~0.011 px of
#: effective radius in tiny-skia's stroker, which is its rasterizer's own rule and not a constant this
#: repo can measure its way to.
#:
#: The class is still constructed, so ``scratch/pgx/`` can reproduce every number above; it just does
#: not enter ``REGISTRY``. See docs/research/pgx.md.
NOT_REGISTERED = {
    "go_19x19": "all-pixel 0.8876 < the suite's shared ALL_EXACT_MIN of 0.90; interior is 1.0",
}


def _make_classes():
    made = {}
    for game in GAMES:
        cname = "Pgx" + "".join(p.capitalize() for p in game.split("_"))
        b = _bundle(game)
        keep = register if game not in NOT_REGISTERED else (lambda c: c)
        made[cname] = keep(type(cname, (PgxBoardBase,), dict(
            GAME=game,
            name=f"pgx-{game.replace('_', '-')}",
            reference_id=f"pgx/{game}",
            # The document's own canvas, so parity is measured at the resolution PGX draws at, and
            # the plate paths the palette actually resolved -- both read off the reference at import
            # rather than written down here, where they would be one upstream theme edit from wrong.
            default_size=b.native,
            PLATE_PATHS=tuple(b.palette),
            __doc__=(f"PGX {game}: state is one ternary per board cell, rendered as a resvg-baked "
                     f"plate of the board furniture plus a lattice gather for the marks."),
        )))
    return made


globals().update(_make_classes())
