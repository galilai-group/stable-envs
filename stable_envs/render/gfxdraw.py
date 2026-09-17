"""Exact batched reimplementation of the SDL_gfx primitives that ``pygame.gfxdraw`` exposes.

Why this exists next to ``raster2d.py``. The SDF rasterizer there is the right tool when the reference
draws anti-aliased vector art with sub-pixel geometry: it approximates coverage by supersampling, and
the honest claim it can make is "interior pixels bit-identical, edges within tolerance", because two
independent rasterizers are entitled to disagree about a pixel an edge only partly covers.

``pygame.gfxdraw`` is not that reference. Its primitives take **Sint16** vertices, so every float
coordinate is truncated to an integer before SDL_gfx sees it, and everything downstream -- scanline
fill, disc spans, even the anti-aliased edges -- is integer arithmetic on integer geometry. There is no
sub-pixel coverage to approximate and therefore no tolerance to hide behind. Redoing the same integer
arithmetic on the GPU is not a better approximation, it is the same computation, and the result is
**0 differing pixels** rather than a high percentage. Measured on gymnasium's classic-control family:
250.9 M pixels, 1024/1024 frames bit-identical (``docs/research/classic_control.md``).

Exactness here is therefore structural, not tuned. Nothing below has a fudge factor, and no constant
was fitted to make a number look good -- each rule was recovered by pixel-diffing pygame and then
checked to reproduce it everywhere, not on average.

Any env whose reference renders through ``pygame.gfxdraw`` should use this rather than ``raster2d``.
That is gymnasium classic control today; ``car_racing`` and ``bipedal_walker`` reuse
``aapolygon``/``filled_polygon`` too.

THE SIX RULES, all measured (``scratch/classic_control/probe_*.py``):

1. **float64 geometry, never float32.** The output depends on ``trunc()`` of the vertex coordinates, so
   a float32 pipeline lands on the wrong integer whenever a vertex falls within ~1e-5 of one and the
   frame is off by a whole row. States may arrive as float32; the arithmetic may not be.

2. **``filled_polygon`` is a 16.16 fixed-point scanline fill**, not a coverage test::

       active edge iff (y1 <= y < y2) or (y == maxy and y1 < y <= y2)
       X = (65536*(y-y1)) // (y2-y1) * (x2-x1) + 65536*x1
       span = [round16(Xlo + 1), round16(Xhi - 1)],  round16(v) = (v>>16) + ((v & 0x8000)>>15)

   Verified 0/600 mismatching pixels on random quads and triangles. ``miny``/``maxy`` are NOT clamped
   to the surface (also measured), so a polygon hanging off an edge keeps the same spans. For convex
   polygons there are exactly two crossings, so the sort collapses to min/max -- which makes this
   cheaper than the per-edge signed-distance test an SDF rasterizer does, as well as exact.

3. **``filled_circle`` spans come from a midpoint walk with the ``ocx``/``ocy`` paint-once guards.**
   ``floor(sqrt(r^2 - dy^2))`` disagrees for **every** radius >= 2, and dropping the guards (taking the
   max over iterations) widens the last row from 0 to 1 and paints a 5-pixel spur across the top and
   bottom of every disc -- invisible by eye, 64 px/frame in a diff.

4. **``aapolygon`` is a Wu line per edge, and four things about it matter.**
   * It has a **closed form**, so it parallelises: with ``E = (minor<<16)//major``,
     ``wraps = (k*E)>>16`` and ``weight = ((k*E) & 0xFFFF)>>8``. Within an edge the major axis advances
     every step, so no two writes collide and the scatter needs no ordering.
   * The blend runs on SDL's **packed channel word**,
     ``R = (dR + (((sR - dR) * alpha) >> 8)) & Rmask`` in unsigned 32-bit, so the truncating ``>>8``
     lands at a different bit position per channel and the unsigned wrap turns it into a *ceiling* when
     the source is darker than the destination. A plain ``(s*a + d*(255-a))//255`` gets 63% of pixels.
   * ``alpha = (255*w) >> 8`` **never reaches 255**, so a weight-255 AA pixel still blends (at 254).
     Only the line's initial pixel goes through ``pixelColor`` and writes the colour outright.
   * ``aalineColorInt`` **clips its endpoints first** (``clipLine``, Cohen-Sutherland on Sint16 with a
     float32 slope, including an endpoint swap). An edge leaving the surface restarts its accumulator
     from the clipped endpoint and lays down a completely different fringe; skipping this costs
     ~150 px/frame wherever a shape touches an edge.
   Axis-aligned edges get **no anti-aliasing at all** -- SDL_gfx short-circuits them to a solid
   ``hline``/``vline``.

5. **``pygame.draw.*`` is a different renderer.** ``pygame.draw.line`` with float endpoints
   **truncates** the coordinate (363.6364 -> row 363); rounding puts a horizontal rule one row off,
   1000 px/frame. ``pygame.draw.aalines`` is a sub-pixel AA path unrelated to gfxdraw's and is not
   reimplemented here -- where it draws something state-independent, bake it once instead.

6. **No host->device copies inside the render.** Every constant belongs in ``__init__``. A single
   ``torch.tensor([...], device="cuda")`` in the hot path is not merely slow: it makes the whole render
   **un-capturable** as a CUDA graph (``Cannot copy between CPU and CUDA tensors during CUDA graph
   capture``), and capture is worth 5.4x at small batch on the anti-aliased path. ``Gfx`` exists to
   make that easy to get right: it owns the caches, and nothing in the drawing functions allocates from
   the host.
"""
from __future__ import annotations

import torch

#: mask for the unsigned-32 arithmetic SDL_gfx's blend relies on
M32 = (1 << 32) - 1
#: sentinel weight meaning ``pixelColor`` (write the colour outright) rather than ``pixelColorWeight``
SOLID = 256
#: a plain ``pygame.Surface`` is 32-bit with masks (0xFF0000, 0xFF00, 0xFF), so R/G/B sit at 16/8/0
SHIFT = (16, 8, 0)

#: ``gfxdraw.aacircle``'s fringe: radius -> ((dy, dx, weight, pass), ...).
#: ``weight`` 256 means ``pixelColor`` (solid); anything else goes through ``pixelColorWeight`` at
#: alpha ``(255*w) >> 8``. ``pass`` is 1 for the second of the two writes ``aaellipseColor`` makes to a
#: few pixels, which must land in order.
#: Generated by ``scratch/classic_control/gen_stencils.py``, fitted jointly over 7 colours x 6
#: backgrounds so the weights are the walk's own and carry no colour assumption. Embedded rather than
#: baked at construction so this module stays pure torch: no pygame import, nothing to warm up.
#: ``aaellipseColor`` is not transcribed because its write set is a constant of the radius (checked:
#: the solid pixels are identical with and without it), so a table is both cheaper and safer than
#: re-deriving an ellipse walker.
AACIRCLE_FRINGE: dict[int, tuple[tuple[int, int, int, int], ...]] = {
    1: ((-1, -1, 2, 1), (-1, -1, 255, 0), (-1, 1, 2, 1), (-1, 1, 255, 0), (1, -1, 2, 1),
        (1, -1, 255, 0), (1, 1, 2, 1), (1, 1, 255, 0)),
    2: ((-2, -1, 192, 0), (-2, 1, 192, 0), (2, -1, 192, 0), (2, 1, 192, 0)),
    3: ((-3, -2, 63, 0), (-3, -1, 213, 0), (-3, 1, 213, 0), (-3, 2, 63, 0), (3, -2, 63, 0),
        (3, -1, 213, 0), (3, 1, 213, 0), (3, 2, 63, 0)),
    4: ((-4, -2, 127, 0), (-4, -1, 224, 0), (-4, 1, 224, 0), (-4, 2, 127, 0), (-3, -3, 170, 0),
        (-3, 3, 170, 0), (-2, -4, 127, 0), (-2, 4, 127, 0), (2, -4, 127, 0), (2, 4, 127, 0),
        (3, -3, 170, 0), (3, 3, 170, 0), (4, -2, 127, 0), (4, -1, 224, 0), (4, 1, 224, 0),
        (4, 2, 127, 0)),
    5: ((-5, -2, 153, 0), (-5, -1, 230, 0), (-5, 1, 230, 0), (-5, 2, 153, 0), (5, -2, 153, 0),
        (5, -1, 230, 0), (5, 1, 230, 0), (5, 2, 153, 0)),
    6: ((-6, -3, 51, 0), (-6, -2, 170, 0), (-6, -1, 234, 0), (-6, 1, 234, 0), (-6, 2, 170, 0),
        (-6, 3, 51, 0), (-5, -4, 127, 0), (-5, 4, 127, 0), (-3, -6, 51, 0), (-3, 6, 51, 0),
        (3, -6, 51, 0), (3, 6, 51, 0), (5, -4, 127, 0), (5, 4, 127, 0), (6, -3, 51, 0),
        (6, -2, 170, 0), (6, -1, 234, 0), (6, 1, 234, 0), (6, 2, 170, 0), (6, 3, 51, 0)),
    7: ((-7, -3, 85, 0), (-7, -2, 183, 0), (-7, -1, 237, 0), (-7, 1, 237, 0), (-7, 2, 183, 0),
        (-7, 3, 85, 0), (-6, -4, 192, 0), (-6, 4, 192, 0), (-3, -7, 85, 0), (-3, 7, 85, 0),
        (3, -7, 85, 0), (3, 7, 85, 0), (6, -4, 192, 0), (6, 4, 192, 0), (7, -3, 85, 0),
        (7, -2, 183, 0), (7, -1, 237, 0), (7, 1, 237, 0), (7, 2, 183, 0), (7, 3, 85, 0)),
    8: ((-8, -3, 109, 0), (-8, -2, 192, 0), (-8, -1, 240, 0), (-8, 1, 240, 0), (-8, 2, 192, 0),
        (-8, 3, 109, 0), (-7, -5, 63, 0), (-7, -4, 237, 0), (-7, 4, 237, 0), (-7, 5, 63, 0),
        (-6, -6, 76, 0), (-6, 6, 76, 0), (-3, -8, 109, 0), (-3, 8, 109, 0), (3, -8, 109, 0),
        (3, 8, 109, 0), (6, -6, 76, 0), (6, 6, 76, 0), (7, -5, 63, 0), (7, -4, 237, 0),
        (7, 4, 237, 0), (7, 5, 63, 0), (8, -3, 109, 0), (8, -2, 192, 0), (8, -1, 240, 0),
        (8, 1, 240, 0), (8, 2, 192, 0), (8, 3, 109, 0)),
    9: ((-9, -4, 15, 0), (-9, -3, 127, 0), (-9, -2, 199, 0), (-9, -1, 241, 0), (-9, 1, 241, 0),
        (-9, 2, 199, 0), (-9, 3, 127, 0), (-9, 4, 15, 0), (-8, -5, 127, 0), (-8, 5, 127, 0),
        (-7, -6, 183, 0), (-7, 6, 183, 0), (-6, -7, 183, 0), (-6, 7, 183, 0), (-5, -8, 127, 0),
        (-5, 8, 127, 0), (-4, -9, 15, 0), (-4, 9, 15, 0), (-3, -9, 127, 0), (-3, 9, 127, 0),
        (3, -9, 127, 0), (3, 9, 127, 0), (4, -9, 15, 0), (4, 9, 15, 0), (5, -8, 127, 0),
        (5, 8, 127, 0), (6, -7, 183, 0), (6, 7, 183, 0), (7, -6, 183, 0), (7, 6, 183, 0),
        (8, -5, 127, 0), (8, 5, 127, 0), (9, -4, 15, 0), (9, -3, 127, 0), (9, -2, 199, 0),
        (9, -1, 241, 0), (9, 1, 241, 0), (9, 2, 199, 0), (9, 3, 127, 0), (9, 4, 15, 0)),
    10: ((-10, -4, 42, 0), (-10, -3, 141, 0), (-10, -2, 204, 0), (-10, -1, 243, 0), (-10, 1, 243, 0),
         (-10, 2, 204, 0), (-10, 3, 141, 0), (-10, 4, 42, 0), (-9, -5, 170, 0), (-9, 5, 170, 0),
         (-8, -7, 36, 0), (-8, 7, 36, 0), (-4, -10, 42, 0), (-4, 10, 42, 0), (4, -10, 42, 0),
         (4, 10, 42, 0), (8, -7, 36, 0), (8, 7, 36, 0), (9, -5, 170, 0), (9, 5, 170, 0),
         (10, -4, 42, 0), (10, -3, 141, 0), (10, -2, 204, 0), (10, -1, 243, 0), (10, 1, 243, 0),
         (10, 2, 204, 0), (10, 3, 141, 0), (10, 4, 42, 0)),
    11: ((-11, -4, 63, 0), (-11, -3, 151, 0), (-11, -2, 209, 0), (-11, -1, 244, 0), (-11, 1, 244, 0),
         (-11, 2, 209, 0), (-11, 3, 151, 0), (-11, 4, 63, 0), (-10, -6, 56, 0), (-10, -5, 204, 0),
         (-10, 5, 204, 0), (-10, 6, 56, 0), (-9, -7, 127, 0), (-9, 7, 127, 0), (-8, -8, 144, 0),
         (-8, 8, 144, 0), (-7, -9, 127, 0), (-7, 9, 127, 0), (-6, -10, 56, 0), (-6, 10, 56, 0),
         (-4, -11, 63, 0), (-4, 11, 63, 0), (4, -11, 63, 0), (4, 11, 63, 0), (6, -10, 56, 0),
         (6, 10, 56, 0), (7, -9, 127, 0), (7, 9, 127, 0), (8, -8, 144, 0), (8, 8, 144, 0),
         (9, -7, 127, 0), (9, 7, 127, 0), (10, -6, 56, 0), (10, -5, 204, 0), (10, 5, 204, 0),
         (10, 6, 56, 0), (11, -4, 63, 0), (11, -3, 151, 0), (11, -2, 209, 0), (11, -1, 244, 0),
         (11, 1, 244, 0), (11, 2, 209, 0), (11, 3, 151, 0), (11, 4, 63, 0)),
    12: ((-12, -4, 81, 0), (-12, -3, 160, 0), (-12, -2, 213, 0), (-12, -1, 245, 0), (-12, 1, 245, 0),
         (-12, 2, 213, 0), (-12, 3, 160, 0), (-12, 4, 81, 0), (-11, -6, 102, 0), (-11, -5, 232, 0),
         (-11, 5, 232, 0), (-11, 6, 102, 0), (-10, -7, 192, 0), (-10, 7, 192, 0), (-6, -11, 102, 0),
         (-6, 11, 102, 0), (-4, -12, 81, 0), (-4, 12, 81, 0), (4, -12, 81, 0), (4, 12, 81, 0),
         (6, -11, 102, 0), (6, 11, 102, 0), (10, -7, 192, 0), (10, 7, 192, 0), (11, -6, 102, 0),
         (11, -5, 232, 0), (11, 5, 232, 0), (11, 6, 102, 0), (12, -4, 81, 0), (12, -3, 160, 0),
         (12, -2, 213, 0), (12, -1, 245, 0), (12, 1, 245, 0), (12, 2, 213, 0), (12, 3, 160, 0),
         (12, 4, 81, 0)),
    13: ((-13, -4, 95, 0), (-13, -3, 167, 0), (-13, -2, 216, 0), (-13, -1, 246, 0), (-13, 1, 246, 0),
         (-13, 2, 216, 0), (-13, 3, 167, 0), (-13, 4, 95, 0), (-12, -6, 139, 0), (-12, -5, 255, 0),
         (-12, 5, 255, 0), (-12, 6, 139, 0), (-11, -8, 63, 0), (-11, 8, 63, 0), (-10, -9, 99, 0),
         (-10, 9, 99, 0), (-8, -11, 63, 0), (-8, 11, 63, 0), (-4, -13, 95, 0), (-4, 13, 95, 0),
         (4, -13, 95, 0), (4, 13, 95, 0), (8, -11, 63, 0), (8, 11, 63, 0), (10, -9, 99, 0),
         (10, 9, 99, 0), (11, -8, 63, 0), (11, 8, 63, 0), (12, -6, 139, 0), (12, -5, 255, 0),
         (12, 5, 255, 0), (12, 6, 139, 0), (13, -4, 95, 0), (13, -3, 167, 0), (13, -2, 216, 0),
         (13, -1, 246, 0), (13, 1, 246, 0), (13, 2, 216, 0), (13, 3, 167, 0), (13, 4, 95, 0)),
    14: ((-14, -5, 19, 0), (-14, -4, 107, 0), (-14, -3, 174, 0), (-14, -2, 219, 0), (-14, -1, 246, 0),
         (-14, 1, 246, 0), (-14, 2, 219, 0), (-14, 3, 174, 0), (-14, 4, 107, 0), (-14, 5, 19, 0),
         (-13, -7, 31, 0), (-13, -6, 167, 0), (-13, -5, 236, 0), (-13, 5, 236, 0), (-13, 6, 167, 0),
         (-13, 7, 31, 0), (-12, -8, 127, 0), (-12, 8, 127, 0), (-11, -9, 186, 0), (-11, 9, 186, 0),
         (-8, -12, 127, 0), (-8, 12, 127, 0), (-7, -13, 31, 0), (-7, 13, 31, 0), (-5, -14, 19, 0),
         (-5, 14, 19, 0), (-4, -14, 107, 0), (-4, 14, 107, 0), (4, -14, 107, 0), (4, 14, 107, 0),
         (5, -14, 19, 0), (5, 14, 19, 0), (7, -13, 31, 0), (7, 13, 31, 0), (8, -12, 127, 0),
         (8, 12, 127, 0), (11, -9, 186, 0), (11, 9, 186, 0), (12, -8, 127, 0), (12, 8, 127, 0),
         (13, -7, 31, 0), (13, -6, 167, 0), (13, -5, 236, 0), (13, 5, 236, 0), (13, 6, 167, 0),
         (13, 7, 31, 0), (14, -5, 19, 0), (14, -4, 107, 0), (14, -3, 174, 0), (14, -2, 219, 0),
         (14, -1, 246, 0), (14, 1, 246, 0), (14, 2, 219, 0), (14, 3, 174, 0), (14, 4, 107, 0),
         (14, 5, 19, 0)),
    15: ((-15, -5, 36, 0), (-15, -4, 118, 0), (-15, -3, 179, 0), (-15, -2, 221, 0), (-15, -1, 247, 0),
         (-15, 1, 247, 0), (-15, 2, 221, 0), (-15, 3, 179, 0), (-15, 4, 118, 0), (-15, 5, 36, 0),
         (-14, -7, 68, 0), (-14, -6, 192, 0), (-14, -5, 219, 0), (-14, 5, 219, 0), (-14, 6, 192, 0),
         (-14, 7, 68, 0), (-13, -8, 177, 0), (-13, 8, 177, 0), (-12, -10, 46, 0), (-12, 10, 46, 0),
         (-11, -11, 51, 0), (-11, 11, 51, 0), (-7, -14, 68, 0), (-7, 14, 68, 0), (-5, -15, 36, 0),
         (-5, 15, 36, 0), (-4, -15, 118, 0), (-4, 15, 118, 0), (4, -15, 118, 0), (4, 15, 118, 0),
         (5, -15, 36, 0), (5, 15, 36, 0), (7, -14, 68, 0), (7, 14, 68, 0), (11, -11, 51, 0),
         (11, 11, 51, 0), (12, -10, 46, 0), (12, 10, 46, 0), (13, -8, 177, 0), (13, 8, 177, 0),
         (14, -7, 68, 0), (14, -6, 192, 0), (14, -5, 219, 0), (14, 5, 219, 0), (14, 6, 192, 0),
         (14, 7, 68, 0), (15, -5, 36, 0), (15, -4, 118, 0), (15, -3, 179, 0), (15, -2, 221, 0),
         (15, -1, 247, 0), (15, 1, 247, 0), (15, 2, 221, 0), (15, 3, 179, 0), (15, 4, 118, 0),
         (15, 5, 36, 0)),
    16: ((-16, -5, 51, 0), (-16, -4, 127, 0), (-16, -3, 184, 0), (-16, -2, 224, 0), (-16, -1, 248, 0),
         (-16, 1, 248, 0), (-16, 2, 224, 0), (-16, 3, 184, 0), (-16, 4, 127, 0), (-16, 5, 51, 0),
         (-15, -7, 100, 0), (-15, -6, 213, 0), (-15, -5, 204, 0), (-15, 5, 204, 0), (-15, 6, 213, 0),
         (-15, 7, 100, 0), (-14, -9, 58, 0), (-14, -8, 219, 0), (-14, 8, 219, 0), (-14, 9, 58, 0),
         (-13, -10, 127, 0), (-13, 10, 127, 0), (-12, -11, 160, 0), (-12, 11, 160, 0),
         (-10, -13, 127, 0), (-10, 13, 127, 0), (-9, -14, 58, 0), (-9, 14, 58, 0), (-7, -15, 100, 0),
         (-7, 15, 100, 0), (-5, -16, 51, 0), (-5, 16, 51, 0), (-4, -16, 127, 0), (-4, 16, 127, 0),
         (4, -16, 127, 0), (4, 16, 127, 0), (5, -16, 51, 0), (5, 16, 51, 0), (7, -15, 100, 0),
         (7, 15, 100, 0), (9, -14, 58, 0), (9, 14, 58, 0), (10, -13, 127, 0), (10, 13, 127, 0),
         (12, -11, 160, 0), (12, 11, 160, 0), (13, -10, 127, 0), (13, 10, 127, 0), (14, -9, 58, 0),
         (14, -8, 219, 0), (14, 8, 219, 0), (14, 9, 58, 0), (15, -7, 100, 0), (15, -6, 213, 0),
         (15, -5, 204, 0), (15, 5, 204, 0), (15, 6, 213, 0), (15, 7, 100, 0), (16, -5, 51, 0),
         (16, -4, 127, 0), (16, -3, 184, 0), (16, -2, 224, 0), (16, -1, 248, 0), (16, 1, 248, 0),
         (16, 2, 224, 0), (16, 3, 184, 0), (16, 4, 127, 0), (16, 5, 51, 0)),
}


# --------------------------------------------------------------------------- span rules
def circle_hw(r: int) -> list[int]:
    """Row half-widths of ``gfxdraw.filled_circle``, indexed by ``|dy|``. Rule 3.

    ``filledCircleColor`` walks the midpoint circle but paints each span only the FIRST time its octant
    coordinate changes (the ``ocx``/``ocy`` guards). Verified exact against pygame for r = 0..40.
    """
    hw = [-1] * (r + 1)

    def span(dy, half):
        if half > hw[dy]:
            hw[dy] = half

    cx, cy = 0, r
    ocx = ocy = -1
    df, d_e, d_se = 1 - r, 3, -2 * r + 5
    while True:
        if ocy != cy:
            span(cy, cx)
            ocy = cy
        if ocx != cx:
            if cx != cy:
                span(cx, cy)
            ocx = cx
        if df < 0:
            df, d_e, d_se = df + d_e, d_e + 2, d_se + 2
        else:
            df, d_e, d_se = df + d_se, d_e + 2, d_se + 4
            cy -= 1
        cx += 1
        if cx > cy:
            return hw


def _round16(v: torch.Tensor) -> torch.Tensor:
    """SDL_gfx's 16.16 fixed-point round-to-nearest: ``(v >> 16) + ((v & 0x8000) >> 15)``."""
    return (v >> 16) + ((v & 32768) >> 15)


def fill_polygon_mask(vx: torch.Tensor, vy: torch.Tensor, ys: torch.Tensor, xs: torch.Tensor,
                      W: int) -> torch.Tensor:
    """``gfxdraw.filled_polygon`` for a batch of CONVEX polygons -> ``(B,H,W)`` bool. Rule 2.

    ``vx``/``vy`` are ``(B,n)`` int64 vertices, ALREADY truncated. ``ys``/``xs`` are the row and column
    index vectors. Convexity gives exactly two crossings per scanline, so the sort in the C routine
    collapses to a min and a max.
    """
    ay, by = vy.roll(1, 1), vy
    ax, bx = vx.roll(1, 1), vx
    swap = ay > by
    ylo = torch.where(swap, by, ay).unsqueeze(-1)                     # (B,n,1)
    yhi = torch.where(swap, ay, by).unsqueeze(-1)
    xlo = torch.where(swap, bx, ax).unsqueeze(-1)
    xhi = torch.where(swap, ax, bx).unsqueeze(-1)
    live = (ay != by).unsqueeze(-1)
    maxy = vy.max(1, keepdim=True).values.unsqueeze(-1)               # (B,1,1)

    y = ys.view(1, 1, -1)
    act = live & (((y >= ylo) & (y < yhi)) | ((y == maxy) & (y > ylo) & (y <= yhi)))
    den = (yhi - ylo).clamp_min(1)
    X = torch.div(65536 * (y - ylo), den, rounding_mode="floor") * (xhi - xlo) + 65536 * xlo
    BIG = 1 << 40
    xa = _round16(torch.where(act, X, BIG).amin(1) + 1).clamp_min(0)  # (B,H)
    xb = _round16(torch.where(act, X, -BIG).amax(1) - 1).clamp_max(W - 1)
    # A row with fewer than two crossings draws nothing. Folded into the span rather than kept as a
    # third term, because ``ok`` is (B,H) and the term it would join is (B,H,W): an empty span
    # already selects no column, so the row test costs H elements instead of H*W.
    xa = torch.where(act.sum(1) >= 2, xa, torch.full_like(xa, W))
    x = xs.view(1, 1, -1)
    return (x >= xa.unsqueeze(-1)) & (x <= xb.unsqueeze(-1))


def disc_offsets(r: int) -> list[tuple[int, int]]:
    """``gfxdraw.filled_circle``'s painted pixels as ``(dy, dx)`` offsets from the centre. Rule 3.

    The same spans as ``fill_circle_mask``, enumerated instead of tested. A disc's write set is a
    constant of its RADIUS, so a filled circle is a scatter of a few hundred pixels rather than a
    predicate over a window -- and that matters because a disc's window is not small: MountainCar's
    wheels ride a sine, so their rows span 341 of 400 even though each wheel is 17 px across, and the
    window form evaluated 205,000 pixels per wheel to paint 200.
    """
    hw = circle_hw(int(r))
    out = []
    for dy in range(-int(r), int(r) + 1):
        h = hw[abs(dy)]
        for dx in range(-h, h + 1):
            out.append((dy, dx))
    return out


def fill_circle_mask(cx: torch.Tensor, cy: torch.Tensor, r: int, ys: torch.Tensor,
                     xs: torch.Tensor, hw_t: torch.Tensor) -> torch.Tensor:
    """``gfxdraw.filled_circle`` -> ``(B,H,W)`` bool. ``cx``/``cy`` are ``(B,)`` int64. Rule 3."""
    dy = (ys.view(1, -1) - cy.view(-1, 1)).abs()                      # (B,H)
    h = torch.where(dy <= r, hw_t[dy.clamp(0, r)], torch.full_like(dy, -1))
    dx = (xs.view(1, 1, -1) - cx.view(-1, 1, 1)).abs()                # (B,1,W)
    return dx <= h.unsqueeze(-1)


# --------------------------------------------------------------------------- blends
def blend(src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor,
          sh: torch.Tensor) -> torch.Tensor:
    """``_putPixelAlpha`` on SDL's masked channel word. ``src``/``dst`` (...,3) int64, ``w`` (...,1).

    Rule 4: the blend runs on the channel in its packed bit position, so the truncating ``>>8`` lands
    below the channel for green and blue, and the unsigned wrap makes it a ceiling when the source is
    darker than the destination. ``alpha = (255*w) >> 8`` maxes out at 254, so only ``SOLID`` writes the
    colour outright.
    """
    a = (255 * w) >> 8
    S = src << sh
    D = dst << sh
    v = ((((S - D) & M32) * a) & M32) >> 8
    return torch.where(w == SOLID, src.expand_as(dst), ((D + v) & (255 << sh)) >> sh)


def blit_over(dst: torch.Tensor, src_rgb: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """SDL's blit of a per-pixel-alpha surface onto an opaque one: ``d + ((s - d) * a) >> 8``.

    A *different* rule from ``blend`` above -- this is ``SDL_BlitSurface``, not SDL_gfx's pixel writer,
    and it takes the alpha straight rather than through ``(255*w)>>8``. Recovered by diffing a scaled
    sprite over four backgrounds: exact on 100% of pixels, where the plausible ``//255`` forms reach
    96.2%. Used for Pendulum's torque arrow, which is a blitted PNG rather than vector art.
    """
    return dst + (((src_rgb - dst) * alpha) >> 8)


# --------------------------------------------------------------------------- clipping
def _clip_encode(x: torch.Tensor, y: torch.Tensor, W: int, H: int) -> torch.Tensor:
    """Cohen-Sutherland's four-bit outcode: left 1, right 2, bottom 4, top 8."""
    z = torch.zeros_like(x)
    c = torch.where(x < 0, z + 1, torch.where(x > W - 1, z + 2, z))
    return c | torch.where(y < 0, z + 8, torch.where(y > H - 1, z + 4, z))


def clip_line(x1: torch.Tensor, y1: torch.Tensor, x2: torch.Tensor, y2: torch.Tensor,
              W: int, H: int, iters: int = 6) -> tuple[torch.Tensor, ...]:
    """SDL_gfx ``clipLine``: Cohen-Sutherland on Sint16 with a float32 slope, batched. Rule 4.

    Returns ``(draw, x1, y1, x2, y2)``. The endpoint SWAP is load-bearing: when the first point is
    already inside, the two are exchanged and the exchange survives into the Wu walk, where it decides
    which end gets the solid initial pixel. Cohen-Sutherland converges in at most four clips, so the
    loop runs to a fixed bound -- a data-dependent ``while`` would need a host sync and break rule 6.
    """
    draw = torch.zeros_like(x1, dtype=torch.bool)
    done = torch.zeros_like(draw)
    for _ in range(iters):
        c1 = _clip_encode(x1, y1, W, H)
        c2 = _clip_encode(x2, y2, W, H)
        acc = ((c1 | c2) == 0) & ~done
        draw = draw | acc
        done = done | acc | (((c1 & c2) != 0) & ~done)
        act = ~done
        sw = act & (c1 == 0)
        x1, x2 = torch.where(sw, x2, x1), torch.where(sw, x1, x2)
        y1, y2 = torch.where(sw, y2, y1), torch.where(sw, y1, y2)
        c1 = torch.where(sw, c2, c1)

        dxne = x2 != x1
        den = (x2 - x1).to(torch.float32)
        m = torch.where(dxne, (y2 - y1).to(torch.float32) / torch.where(dxne, den, den + 1),
                        torch.ones_like(den))
        msafe = torch.where(m == 0, torch.ones_like(m), m)
        L = act & ((c1 & 1) != 0)
        Rt = act & ~L & ((c1 & 2) != 0)
        Bt = act & ~L & ~Rt & ((c1 & 4) != 0)
        Tp = act & ~L & ~Rt & ~Bt & ((c1 & 8) != 0)
        dyL = torch.trunc((0 - x1).to(torch.float32) * m).to(torch.int64)
        dyR = torch.trunc((W - 1 - x1).to(torch.float32) * m).to(torch.int64)
        dxB = torch.trunc((H - 1 - y1).to(torch.float32) / msafe).to(torch.int64)
        dxT = torch.trunc((0 - y1).to(torch.float32) / msafe).to(torch.int64)
        ny = torch.where(L, y1 + dyL, torch.where(Rt, y1 + dyR, torch.where(
            Bt, torch.full_like(y1, H - 1), torch.where(Tp, torch.zeros_like(y1), y1))))
        nx = torch.where(L, torch.zeros_like(x1), torch.where(
            Rt, torch.full_like(x1, W - 1), torch.where(
                Bt & dxne, x1 + dxB, torch.where(Tp & dxne, x1 + dxT, x1))))
        x1, y1 = nx, ny
    return draw, x1, y1, x2, y2


# --------------------------------------------------------------------------- canvas
def select_(dst: torch.Tensor, mask: torch.Tensor, src: torch.Tensor) -> None:
    """``dst = mask ? src : dst``, in place, over whatever shapes broadcast. Integer, exact.

    The obvious spelling is ``torch.where(mask, src, dst, out=dst)`` and it is what CUDA gets, but on
    the CPU it runs at ~340 Mpx/s on an interleaved ``(B,H,W,3)`` surface and ~2.6 Gpx/s on a plane,
    because ``where`` over a broadcast bool takes a scalar path. The bitwise identity
    ``d ^= (d ^ s) * m`` is three vectorised uint8 passes and measures 4.8 Gpx/s on the same plane --
    3.7x -- at the cost of one temporary. Both compute the same integers; this is a codegen
    difference, not a numerical one, and the choice is by device rather than by taste.
    """
    if dst.device.type == "cuda":
        torch.where(mask, src, dst, out=dst)
    else:
        dst.bitwise_xor_((dst ^ src) * mask.view(torch.uint8))


class Canvas:
    """A batch of RGB surfaces, in the reference's PRE-FLIP orientation (y down).

    One trailing trash pixel absorbs every out-of-bounds anti-aliasing write, so clipping costs an
    index clamp instead of the boolean-mask host sync that ``buf[mask]`` would force (rule 6).

    PLANAR, ``(3, B, H, W)``, not interleaved ``(B, H, W, 3)``, and that is a throughput decision
    with one number behind it. Every opaque primitive writes ``mask ? colour : surface``, where the
    mask is per PIXEL and the colour is per channel; interleaved, the mask has to broadcast across a
    trailing axis of 3 and the write runs at 340 Mpx/s, while planar it broadcasts across the outer
    axis over a contiguous ``B*H*W`` block and runs at 4.8 Gpx/s. Measured at batch 64 on a 400x600
    surface: 45.4 ms per primitive against 3.2. The cost is paid once at the end, where the three
    planes are interleaved into the ``(B,H,W,3)`` the caller asked for (18.4 ms at that size), rather
    than once per primitive.

    ``flat`` is correspondingly ``(3, B*H*W+1)``: a scatter still touches all three channels in one
    indexing operation, because the channel axis is the one NOT being indexed.
    """

    def __init__(self, B, H, W, device, sh, ys, xs, bg=(255, 255, 255), gfx=None):
        self.B, self.H, self.W = B, H, W
        self.device = device
        self.flat = torch.empty((3, B * H * W + 1), dtype=torch.uint8, device=device)
        self.buf = self.flat[:, :-1].view(3, B, H, W)
        for c in range(3):
            self.buf[c].fill_(int(bg[c]))
        self.ys, self.xs, self.sh = ys, xs, sh
        self.trash = B * H * W
        self.gfx = gfx

    def steps(self, kmax: int) -> torch.Tensor:
        """``arange(kmax+1)`` as a row. Cached: it is a constant of the primitive, and rule 6 says a
        draw call may not build a tensor from the host."""
        if self.gfx is not None:
            return self.gfx.steps(kmax)
        return torch.arange(kmax + 1, device=self.device).view(1, -1)

    def solid(self) -> torch.Tensor:
        """The ``pixelColor`` weight, cached as ONE element. Callers ``expand`` it to their write count.

        An edge used to allocate and fill a ``B*(kmax+1)`` vector of a constant before every scatter
        -- three allocations per edge that the profile counted and the picture never used. A
        stride-0 ``expand`` of this removes the allocation while keeping ``len(w) == len(valid)``,
        which ``blend_at`` overrides are entitled to rely on.
        """
        if self.gfx is not None:
            return self.gfx.solid
        return torch.tensor([SOLID], dtype=torch.int64, device=self.device)

    def color_u8(self, color) -> torch.Tensor:
        """A python colour literal as a cached ``(3,)`` uint8 tensor.

        ``torch.tensor([...], device=...)`` in a draw call is a host-to-device copy in the hot path,
        which is rule 6 -- and under CUDA-graph capture it is not merely slow, it makes the render
        uncapturable. Cached on the ``Gfx`` where every other constant lives; the fallback allocates
        only for a bare ``Canvas`` built without one.
        """
        if self.gfx is not None:
            return self.gfx.color_u8(color)
        return torch.tensor([int(v) for v in color], dtype=torch.uint8, device=self.device)

    def always(self) -> torch.Tensor:
        """An all-true ``valid``, cached as one element; callers ``expand`` it to their write count."""
        if self.gfx is not None:
            return self.gfx.always
        return torch.ones(1, dtype=torch.bool, device=self.device)

    def window(self, win):
        """``self.buf`` restricted to ``(y0, y1, x0, x1)``; the whole surface when ``win`` is None."""
        return self.buf if win is None else self.buf[:, :, win[0]:win[1], win[2]:win[3]]

    def rows(self, win):
        """The row and column index vectors a primitive with window ``win`` evaluates over."""
        if win is None:
            return self.ys, self.xs
        return self.ys[win[0]:win[1]], self.xs[win[2]:win[3]]

    def fill_from(self, base: torch.Tensor) -> None:
        """Start from a prebaked ``(3,1,H,W)`` or ``(3,B,H,W)`` layer instead of a flat colour."""
        self.buf.copy_(base)

    def paint(self, mask, color, window=None) -> None:
        """Opaque fill: three scalar ``masked_fill_``, so no allocation and no host copy."""
        buf = self.window(window)
        for c in range(3):
            buf[c].masked_fill_(mask, int(color[c]))

    def _disc_index(self, cy, cx, off, gate=None):
        """Flat indices of a disc stencil placed at per-environment centres; out of range -> trash."""
        B = cy.shape[0]
        y = cy.view(-1, 1) + off[:, 0].view(1, -1)
        x = cx.view(-1, 1) + off[:, 1].view(1, -1)
        ok = (y >= 0) & (y < self.H) & (x >= 0) & (x < self.W)
        if gate is not None:
            ok = ok & gate.view(-1, 1)
        b = torch.arange(B, device=self.device).view(-1, 1)
        return torch.where(ok, (b * self.H + y.clamp(0, self.H - 1)) * self.W
                           + x.clamp(0, self.W - 1), self.trash).reshape(-1)

    def paint_disc(self, cy, cx, off, color) -> None:
        """``filled_circle`` as a scatter of its own pixels. Same write set as the window form.

        Offsets within one disc are distinct and discs of different environments land in different
        frames, so no two writes of this scatter share a destination and there is no ordering to
        lose -- the out-of-range ones all go to the trash pixel, whose value nothing reads.
        """
        idx = self._disc_index(cy, cx, off)
        n = off.shape[0] * cy.shape[0]
        self.flat[:, idx] = self.color_u8(color).view(3, 1).expand(3, n)

    def blend_at(self, b, y, x, color_t, w, valid) -> None:
        """Blend ``color_t`` ``(3,1)`` int64 at flat-indexed pixels; ``valid`` False goes to the trash."""
        ok = valid & (y >= 0) & (y < self.H) & (x >= 0) & (x < self.W)
        flat = torch.where(ok, (b * self.H + y.clamp(0, self.H - 1)) * self.W
                           + x.clamp(0, self.W - 1), self.trash)
        dst = self.flat[:, flat].to(torch.int64)
        self.flat[:, flat] = blend(color_t, dst, w.reshape(1, -1),
                                   self.sh.view(3, 1)).to(torch.uint8)

    def rgb(self, out=None) -> torch.Tensor:
        """The finished frame, with the reference's final ``transform.flip(surf, False, True)``.

        The flip is a row reversal of the planes and the interleave is the transpose back to the
        caller's ``(B,H,W,3)``; this is the one pass over the surface the planar canvas costs per
        FRAME, against the one per PRIMITIVE it removes.

        One plane at a time, although ``stack`` over the three flipped planes measures the same
        18.5 ms at batch 64: the whole-surface form needs a second copy of the surface alive while
        it builds the third, and at batch 4096 on a 500x500 frame that is 3.1 GB of peak rather than
        1.0. Same instruction count, a third of the transient.
        """
        if out is None:
            out = torch.empty((self.B, self.H, self.W, 3), dtype=torch.uint8, device=self.device)
        for c in range(3):
            out[..., c].copy_(self.buf[c].flip(1))
        return out


def to_u8(color: torch.Tensor) -> torch.Tensor:
    """``(B,3)`` float style values -> uint8, truncating, which is what ``int(c)`` did."""
    return color.to(torch.uint8)


class StyleCanvas(Canvas):
    """A canvas whose current colour is one row PER ENVIRONMENT, with two per-environment gates.

    The primitives above take a colour as a python tuple (for the opaque ``masked_fill_``) plus a
    ``(1,3)`` tensor (for the Wu blend), both shared by the whole batch. That is the right shape for a
    single appearance and cannot express 4096 different ones, and widening every primitive signature
    would touch every call site. The colour is instead carried on the canvas: ``use`` sets a ``(B,3)``
    tensor and both write paths -- the opaque fill and the anti-aliased blend -- read it from there, so
    ``polygon`` and ``circle`` are called unchanged and every environment gets its own colour out of
    one launch.

    ``aa`` is a per-environment gate rather than a python ``bool`` for the same reason: anti-aliasing
    is a declared variation, so a batch may hold both settings. The Wu pass then always runs and the
    disabled environments' writes are routed to the canvas's trash pixel, which is what ``blend_at``
    already does for out-of-bounds writes.

    ``gate`` is the same mechanism for a primitive that EXISTS in some environments and not in others.
    A batch is one static sequence of draw calls, so an env with four exhaust particles and one with
    six cannot issue different numbers of them; the six slots are always drawn and the missing two are
    gated off. That is not only about the particles themselves -- LunarLander draws its landing flags
    once per FIXTURE, so the number of particles alive decides how many times the flags' anti-aliased
    fringe is blended, and getting the count wrong is worth up to 152/255 on those pixels.
    """

    def __init__(self, gfx: "Gfx", B: int, bg):
        super().__init__(B, gfx.H, gfx.W, gfx.device, gfx.sh, gfx.ys, gfx.xs, (0, 0, 0), gfx=gfx)
        self.buf.copy_(to_u8(bg).t().view(3, B, 1, 1))
        self.col = None
        self.aa = None
        self.gate = None

    def use(self, color) -> "StyleCanvas":
        """Set the colour subsequent primitives paint with. ``color`` is ``(B,3)``."""
        self.col = color
        return self

    def gated(self, gate) -> "StyleCanvas":
        """Restrict subsequent primitives to the environments where ``gate`` (``(B,)`` bool) is True."""
        self.gate = gate
        return self

    def paint(self, mask, color=None, window=None) -> None:
        """Opaque fill. ``window`` is ``(y0, y1, x0, x1)`` and makes ``mask`` cover that box only.

        The window is what ``polygon``/``circle`` build their mask over as well (``Canvas.rows``), so
        a primitive with a known bounding box neither evaluates its span rule nor writes its colour
        outside that box.

        The window is a throughput argument, not a correctness one, and it is the difference between
        a usable renderer and an unusable one when a scene has many small primitives: without it every
        primitive reads and rewrites the whole ``(B, H, W, 3)`` buffer, so LunarLander's 31 polygons
        moved 9 GB per batch of 256 and rendered at 266 frames/s -- SLOWER than the official pygame
        renderer it is supposed to replace. A primitive whose columns are known (its vertices are
        constants, or its x-extent is) pays for its own box instead.
        """
        if self.gate is not None:
            mask = mask & self.gate.view(-1, 1, 1)
        c = to_u8(self.col if color is None else color).t().reshape(3, self.B, 1, 1)
        select_(self.window(window), mask.contiguous().unsqueeze(0), c)

    def paint_at(self, offsets, color=None, gate=None) -> None:
        """Write a colour at a CONSTANT set of frame-relative flat pixel offsets, per environment.

        For a primitive whose pixel set does not depend on the state -- LunarLander's flagpoles and
        flag triangles are at fixed coordinates -- the whole rasterisation is a compile-time constant
        and the draw is a scatter of a few hundred pixels. Gated environments are routed to the
        canvas's trash pixel, the same way an out-of-bounds anti-aliased write is.
        """
        g = self.gate if gate is None else gate
        n = int(offsets.numel())
        idx = (torch.arange(self.B, device=self.device).view(-1, 1) * (self.H * self.W)
               + offsets.view(1, -1))
        if g is not None:
            idx = torch.where(g.view(-1, 1), idx, torch.full_like(idx, self.trash))
        c = to_u8(self.col if color is None else color)
        self.flat[:, idx.reshape(-1)] = c.t().reshape(3, self.B, 1).expand(3, self.B, n).reshape(3, -1)

    def paint_disc(self, cy, cx, off, color=None) -> None:
        B, S = cy.shape[0], off.shape[0]
        idx = self._disc_index(cy, cx, off, self.gate)
        c = to_u8(self.col if color is None else color)
        self.flat[:, idx] = c.t().reshape(3, B, 1).expand(3, B, S).reshape(3, -1)

    def blend_at(self, b, y, x, color_t, w, valid) -> None:
        if self.aa is not None:
            valid = valid & self.aa[b]
        if self.gate is not None:
            valid = valid & self.gate[b]
        super().blend_at(b, y, x, self.col.to(torch.int64).t()[:, b], w, valid)


# --------------------------------------------------------------------------- primitives
def edge_geometry(x1: torch.Tensor, y1: torch.Tensor, x2: torch.Tensor, y2: torch.Tensor,
                  W: int, H: int) -> tuple[torch.Tensor, ...]:
    """``clip_line`` plus every constant ``aalineColor``'s Wu walk needs, for a FLAT batch of edges.

    **The one place this arithmetic is written down.** All three anti-aliased paths in this package
    need exactly it -- the per-primitive path (:func:`edge_setup` -> :func:`aaline_emit`), the fused
    CarRacing kernel's pre-pass (``gfxdraw_fused.edge_table``) and, through that table, the fused
    classic-control kernel. It used to be transcribed twice, character for character, in two modules
    that are certified against each other pixel for pixel; a single edited line in one copy would
    have moved a fringe in one path and not the other, and the diff test would have found it only on
    the polygons it happened to cover.

    Rule 4, and the reason the setup is split from the writes: six Cohen-Sutherland iterations are
    ~180 tiny operator calls whatever the edge's length, so at batch 1 they are the renderer (3.1 ms
    of an 8.1 ms CartPole frame, 38%, in eight calls). Doing them for every edge at once makes that
    one call. The WRITES stay per edge and in order -- two edges of a polygon can land on the same
    pixel at a shared vertex, and a batched scatter would read every destination before writing any
    of them, so merging those would change the picture at corners.

    Returns ``(draw, x1, y1, x2, y2, xx0, yy0, dx, dy, xdir, vert, horiz, ymaj, major, E, lo)``,
    where ``x1, y1, x2, y2`` are the CLIPPED endpoints (the clip's endpoint swap included, because
    it decides which end gets the solid initial pixel) and ``xx0, yy0`` is the walk's origin after
    the walk's own top-to-bottom flip.
    """
    draw, x1, y1, x2, y2 = clip_line(x1, y1, x2, y2, W, H)
    flip = y1 > y2
    xx0 = torch.where(flip, x2, x1)
    yy0 = torch.where(flip, y2, y1)
    dxs = torch.where(flip, x1, x2) - xx0
    dy = torch.where(flip, y1, y2) - yy0
    xdir = torch.where(dxs >= 0, 1, -1)
    dx = dxs.abs()
    vert = dx == 0
    horiz = (dy == 0) & ~vert
    ymaj = dy > dx
    major = torch.where(ymaj, dy, dx)
    minor = torch.where(ymaj, dx, dy)
    E = torch.div(minor << 16, major.clamp_min(1), rounding_mode="floor")
    lo = torch.minimum(xx0, xx0 + xdir * dx)
    return (draw, x1, y1, x2, y2, xx0, yy0, dx, dy, xdir, vert, horiz, ymaj, major, E, lo)


def edge_setup(x1: torch.Tensor, y1: torch.Tensor, x2: torch.Tensor, y2: torch.Tensor,
               W: int, H: int) -> tuple[torch.Tensor, ...]:
    """:func:`edge_geometry` in the order :func:`aaline_emit` unpacks it (no clipped far endpoint)."""
    draw, sx, sy, _x2, _y2, xx0, yy0, dx, dy, xdir, vert, horiz, ymaj, major, E, lo = \
        edge_geometry(x1, y1, x2, y2, W, H)
    return (draw, sx, sy, xx0, yy0, dx, dy, xdir, vert, horiz, ymaj, major, E, lo)


def aaline_emit(cv: Canvas, p, color_t, kmax, bidx) -> None:
    """The five scatters of ``_aalineColor(..., draw_endpoint=0)``, for one already-clipped edge.

    ``kmax`` is a STATIC bound on the step count, supplied by the caller from the primitive's own
    geometry. Deriving it from the data would need ``.max().item()`` and a host sync per edge.
    """
    draw, sx, sy, xx0, yy0, dx, dy, xdir, vert, horiz, ymaj, major, E, lo = p
    B = xx0.shape[0]
    # ONE weight per write, as a stride-0 view of a cached scalar. ``blend`` would broadcast a
    # 1-element weight, but ``blend_at`` is an overridable seam and a subclass is entitled to index
    # the weight by its own mask -- LunarLander's _AARecorder does exactly that to bake the flag
    # fringe, and a (1,) weight against a (B*(kmax+1),) mask is an IndexError inside its __init__.
    # ``expand`` keeps what the cache was for (no allocation, no host copy, rule 6) and keeps the
    # contract the baseline had: len(w) == len(valid).
    solid = cv.solid()
    cv.blend_at(bidx, sy, sx, color_t, solid.expand(B), draw)

    k = cv.steps(kmax)
    n = B * (kmax + 1)
    run = solid.expand(n)
    bb = bidx.view(-1, 1).expand(B, kmax + 1).reshape(-1)
    dw = draw.view(-1, 1)
    # axis-aligned edges: SDL_gfx short-circuits to a solid vline/hline, with no AA whatsoever
    cv.blend_at(bb, (yy0.view(-1, 1) + k).reshape(-1),
                xx0.view(-1, 1).expand(B, kmax + 1).reshape(-1), color_t, run,
                (dw & vert.view(-1, 1) & (dy > 0).view(-1, 1) & (k <= dy.view(-1, 1))).reshape(-1))
    cv.blend_at(bb, yy0.view(-1, 1).expand(B, kmax + 1).reshape(-1),
                (lo.view(-1, 1) + k).reshape(-1), color_t, run,
                (dw & horiz.view(-1, 1) & (k <= dx.view(-1, 1))).reshape(-1))

    gen = ~vert & ~horiz
    kk = k[:, 1:]
    acc = kk * E.view(-1, 1)
    wraps = acc >> 16
    wgt = (acc & 0xFFFF) >> 8
    live = (dw & gen.view(-1, 1) & (kk < major.view(-1, 1))).reshape(-1)
    ym = ymaj.view(-1, 1)
    base_x, base_y, d = xx0.view(-1, 1), yy0.view(-1, 1), xdir.view(-1, 1)
    xa = torch.where(ym, base_x + d * wraps, base_x + d * kk)
    ya = torch.where(ym, base_y + kk, base_y + wraps)
    xb = torch.where(ym, base_x + d * (wraps + 1), xa)
    yb = torch.where(ym, ya, ya + 1)
    bk = bidx.view(-1, 1).expand_as(xa).reshape(-1)
    cv.blend_at(bk, ya.reshape(-1), xa.reshape(-1), color_t, (255 - wgt).reshape(-1), live)
    cv.blend_at(bk, yb.reshape(-1), xb.reshape(-1), color_t, wgt.reshape(-1), live)


def aaline(cv: Canvas, x1, y1, x2, y2, color_t, kmax, bidx) -> None:
    """``_aalineColor(..., draw_endpoint=0)``: clip, solid initial pixel, then the Wu pairs. Rule 4."""
    aaline_emit(cv, edge_setup(x1, y1, x2, y2, cv.W, cv.H), color_t, kmax, bidx)


def polygon(cv: Canvas, verts, color, color_t, bidx, kmax, aa: bool = True,
            fill_first: bool = False, window=None) -> None:
    """The gymnasium idiom: ``aapolygon`` then ``filled_polygon``, on the same float64 vertices.

    The AA fringe is blended against the canvas at draw time and the hard fill then overwrites the
    interior, so what survives is a one-pixel blended rim strictly outside the fill.

    ``fill_first`` reverses the two, which is the order LunarLander uses (``pygame.draw.polygon`` and
    THEN ``gfxdraw.aapolygon``). It is a different picture, not a reordering of the same one: with the
    fill last the rim survives only outside it, and with the fill first the rim is blended over the
    fill's own edge pixels as well.

    ``kmax`` may be one bound for every edge or one PER edge. A rotated rectangle's edges alternate
    between the long and the short side, and the Wu pass costs ``kmax`` steps whether or not the edge
    is that long, so a single bound makes every short edge pay the long edge's price.

    ``window`` is ``(y0, y1, x0, x1)``, a STATIC superset of the polygon's footprint supplied by the
    caller from its own geometry. It bounds the scanline fill only -- the fringe is a scatter and
    clips itself -- and it is why a primitive no longer reads and rewrites the whole surface.
    """
    vi = torch.trunc(verts).to(torch.int64)
    vx, vy = vi[..., 0], vi[..., 1]
    n = vx.shape[1]
    km = [kmax] * n if isinstance(kmax, int) else list(kmax)

    def _aa():
        # one clip for every edge of the polygon, then the writes edge by edge and in order
        nx, ny = vx.roll(-1, 1), vy.roll(-1, 1)
        pars = edge_setup(vx.t().reshape(-1), vy.t().reshape(-1),
                          nx.t().reshape(-1), ny.t().reshape(-1), cv.W, cv.H)
        B = vx.shape[0]
        for e in range(n):
            aaline_emit(cv, tuple(t[e * B:(e + 1) * B] for t in pars), color_t, km[e], bidx)

    if aa and not fill_first:
        _aa()
    ys, xs = cv.rows(window)
    cv.paint(fill_polygon_mask(vx, vy, ys, xs, cv.W), color, window)
    if aa and fill_first:
        _aa()


def circle(cv: Canvas, cx, cy, r, color, color_t, bidx, hw_t, stencil, aa: bool = True,
           window=None, disc=None) -> None:
    """``aacircle`` (fringe stencil) then ``filled_circle``.

    ``stencil`` is a per-pass list of ``(dy, dx, w)`` tensors: ``aaellipseColor`` writes a few pixels
    twice and the two writes must land in order, so they cannot share a scatter.

    ``disc`` is the filled disc's own ``(S,2)`` offset stencil (``Gfx.disc``). When it is given the
    fill is a scatter of exactly the pixels it paints; ``hw_t``/``window`` are then unused and the
    predicate form is kept only as the readable definition the scatter is checked against.
    """
    if aa and stencil is not None:
        B = cx.shape[0]
        for st in stencil:
            S = st.shape[0]
            if not S:
                continue
            bb = bidx.view(-1, 1).expand(B, S).reshape(-1)
            cv.blend_at(bb, (cy.view(-1, 1) + st[:, 0].view(1, -1)).reshape(-1),
                        (cx.view(-1, 1) + st[:, 1].view(1, -1)).reshape(-1), color_t,
                        st[:, 2].view(1, -1).expand(B, S).reshape(-1),
                        cv.always().expand(B * S))
    if disc is not None:
        cv.paint_disc(cy, cx, disc, color)
    else:
        ys, xs = cv.rows(window)
        cv.paint(fill_circle_mask(cx, cy, r, ys, xs, hw_t), color, window)


def rotate(v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """``pygame.math.Vector2.rotate_rad``: ``[[cos,-sin],[sin,cos]]`` in a y-DOWN raster.

    Measured, not assumed: ``(1,0).rotate_rad(pi/2)`` is ``(0,1)``, so the matrix is the ordinary CCW
    one applied before the surface is flipped.
    """
    c, s = torch.cos(a), torch.sin(a)
    return torch.stack([v[..., 0] * c - v[..., 1] * s, v[..., 0] * s + v[..., 1] * c], -1)


class Gfx:
    """Per-(canvas size, device) constant cache and canvas factory.

    Exists to make rule 6 the path of least resistance: every tensor a draw call needs -- row/column
    indices, channel shifts, batch indices, circle span tables, AA stencils, vertex literals -- is
    materialised here, once, so the render path allocates nothing from the host and the whole call is
    CUDA-graph capturable.
    """

    def __init__(self, H: int, W: int, device):
        self.H, self.W = int(H), int(W)
        self.device = torch.device(device)
        self.ys = torch.arange(self.H, device=self.device)
        self.xs = torch.arange(self.W, device=self.device)
        self.sh = torch.tensor(SHIFT, dtype=torch.int64, device=self.device)
        self._hw: dict[int, torch.Tensor] = {}
        self._st: dict[int, list[torch.Tensor]] = {}
        self._disc: dict[int, torch.Tensor] = {}
        self._bi: dict[int, torch.Tensor] = {}
        self._k: dict[tuple, torch.Tensor] = {}
        self._steps: dict[int, torch.Tensor] = {}
        self.solid = torch.tensor([SOLID], dtype=torch.int64, device=self.device)
        self.always = torch.ones(1, dtype=torch.bool, device=self.device)

    def canvas(self, B: int, bg=(255, 255, 255)) -> Canvas:
        return Canvas(B, self.H, self.W, self.device, self.sh, self.ys, self.xs, bg, gfx=self)

    def steps(self, kmax: int) -> torch.Tensor:
        k = int(kmax)
        if k not in self._steps:
            self._steps[k] = torch.arange(k + 1, device=self.device).view(1, -1)
        return self._steps[k]

    def batch_index(self, B: int) -> torch.Tensor:
        if B not in self._bi:
            self._bi[B] = torch.arange(B, device=self.device)
        return self._bi[B]

    def hw(self, r: int) -> torch.Tensor:
        if r not in self._hw:
            self._hw[r] = torch.tensor(circle_hw(int(r)), dtype=torch.int64, device=self.device)
        return self._hw[r]

    def disc(self, r: int) -> torch.Tensor:
        """``(S,2)`` int64 offsets of the filled disc of radius ``r``, cached (rule 6)."""
        if r not in self._disc:
            self._disc[r] = torch.tensor(disc_offsets(int(r)), dtype=torch.int64,
                                         device=self.device).view(-1, 2)
        return self._disc[r]

    def stencil(self, r: int) -> list[torch.Tensor]:
        if r not in self._st:
            rows = AACIRCLE_FRINGE.get(int(r))
            if rows is None:
                raise KeyError(
                    f"no aacircle fringe table for radius {r}; regenerate with "
                    f"scratch/classic_control/gen_stencils.py rather than approximating it")
            self._st[r] = [torch.tensor([[a, b, w] for (a, b, w, p) in rows if p == pas],
                                        dtype=torch.int64, device=self.device).view(-1, 3)
                           for pas in (0, 1)]
        return self._st[r]

    def verts(self, vals) -> torch.Tensor:
        """A float64 vertex literal, cached. float64 is rule 1 and is not negotiable."""
        key = ("v", tuple(map(tuple, vals)))
        if key not in self._k:
            self._k[key] = torch.tensor(vals, dtype=torch.float64, device=self.device)
        return self._k[key]

    def color_u8(self, c) -> torch.Tensor:
        """A ``(3,)`` uint8 colour literal, cached. The opaque-fill twin of :meth:`color`."""
        key = ("u8", tuple(int(v) for v in c))
        if key not in self._k:
            self._k[key] = torch.tensor(key[1], dtype=torch.uint8, device=self.device)
        return self._k[key]

    def color(self, c) -> torch.Tensor:
        key = ("c", tuple(int(v) for v in c))
        if key not in self._k:
            self._k[key] = torch.tensor(key[1], dtype=torch.int64, device=self.device).view(3, 1)
        return self._k[key]
