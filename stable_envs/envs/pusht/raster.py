"""Batched PushT frames equal to the pygame frames: the pygame 2.6.1 draw calls and cv2 resize in integer arithmetic.

The host turns render state into a list of primitives per frame in pygame's painter order: polygon
fills (``draw_fillpoly``), filled circles (``draw_circle_filled``), width-4 lines (``draw_line_width``)
and antialiased contact lines (``draw_aaline``). The device evaluates the primitives only at the 512 px
canvas pixels that ``cv2.resize(INTER_LINEAR)`` reads, then applies the fixed-point resize.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from functools import cache, lru_cache

import numpy as np
import torch

from .simulation.numeric import get_value

CANVAS = 512
#: Primitive kinds.
POLY, CIRCLE, LINE = 1, 2, 3
#: A polygon or line given as per-row x intervals (see ``raster_device.rows``).
ROWS = 4
#: ROWS: offset of the first row of intervals (four ``lo, hi`` pairs per row).
ROFF = 6
#: Record layout: kind, packed color, inclusive bounding box, then kind-specific integers.
KIND, COLOR, BX0, BY0, BX1, BY1 = range(6)
#: POLY: point count, then up to five x and five y values.
PN, PX, PY = 6, 7, 12
#: CIRCLE: center and radius.
CX, CY, CR = 6, 7, 8
#: LINE: start, direction signs, extents, initial error, step count, and 1 when spans are horizontal.
LX, LY, LSX, LSY, LDX, LDY, LERR, LN, LXINC = range(6, 15)
#: 1 for primitives that only the variation and goal place (goal, walls).
STATIC = 17
RECORD = 18
#: Most contact segments a state holds.
SEGMENTS = 90
WALL_COLOR = (211, 211, 211)
AALINE_COLOR = (231, 76, 60)


def pack(rgb):
    rgb = np.asarray(rgb, np.int64)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


@cache
def circle_spans(radius):
    """Return ``[2r+2, 2]`` inclusive x offsets per row offset ``dy + r + 1`` drawn by ``draw_circle_filled``."""
    spans = np.full((2 * radius + 2, 2), (1, 0), np.int64)

    def add(y, x1, x2):
        row = spans[y + radius + 1]
        if row[0] > row[1]:
            row[:] = (x1, x2)
        else:
            row[:] = (min(row[0], x1), max(row[1], x2))

    f, ddx, ddy, x, y = 1 - radius, 0, -2 * radius, 0, radius
    while x < y:
        if f >= 0:
            y -= 1
            ddy += 2
            f += ddy
        x += 1
        ddx += 2
        f += ddx + 1
        if f >= 0:
            add(y - 1, -x, x - 1)
            add(-y, -x, x - 1)
        add(x - 1, -y, y - 1)
        add(-x, -y, y - 1)
    return spans


@cache
def circle_table(max_radius):
    """Return ``[max_radius + 1, 2 * max_radius + 2, 2]`` span offsets; empty rows have lo > hi."""
    table = np.zeros((max_radius + 1, 2 * max_radius + 2, 2), np.int64)
    table[..., 0] = 1
    for r in range(1, max_radius + 1):
        spans = circle_spans(r)
        table[r, max_radius - r : max_radius + r + 2] = spans
    return table


class Template:
    """Local geometry and debug-draw order of one PushT scene type.

    ``items`` lists ``("poly", body, local_vertices)``, ``("circle", body, offset, radius)`` and
    ``("wall", a, b, radius)`` in the order ``space.debug_draw`` draws them.
    """

    def __init__(self, values):
        import pymunk

        from .simulation.task import PushT

        scene = PushT(resolution=CANVAS)
        try:
            scene.reset(seed=0, options=dict(variation=[], variation_values=values))
            shapes = []

            class Order(pymunk.SpaceDebugDrawOptions):
                def color_for_shape(self, shape):
                    shapes.append(shape)
                    return super().color_for_shape(shape)

                def draw_circle(self, *args):
                    pass

                def draw_segment(self, *args):
                    pass

                def draw_fat_segment(self, *args):
                    pass

                def draw_polygon(self, *args):
                    pass

                def draw_dot(self, *args):
                    pass

            options = Order()
            options.flags = options.DRAW_SHAPES
            scene.space.debug_draw(options)
            self.items = []
            for shape in shapes:
                body = "agent" if shape.body is scene.agent else "block" if shape.body is scene.block else None
                if isinstance(shape, pymunk.Poly):
                    self.items.append(("poly", body, np.array([tuple(v) for v in shape.get_vertices()], np.float64)))
                elif isinstance(shape, pymunk.Circle):
                    self.items.append(("circle", body, tuple(shape.offset), float(shape.radius)))
                else:
                    self.items.append(("wall", tuple(shape.a), tuple(shape.b), float(shape.radius)))
            self.goal = [
                np.array([tuple(v) for v in shape.get_vertices()], np.float64)
                for shape in scene.block.shapes
                if isinstance(shape, pymunk.Poly)
            ]
            if any(isinstance(shape, pymunk.Circle) for shape in scene.block.shapes):
                raise NotImplementedError("PushT goal discs are not supported by the batched renderer")
        finally:
            scene.close()


@lru_cache(maxsize=256)
def _template(key):
    values = {}
    for name, kind, value in key:
        if kind.startswith("ndarray:"):
            value = np.array(value, np.dtype(kind.removeprefix("ndarray:")))
        group, leaf = name.split(".")
        values.setdefault(group, {})[leaf] = value
    return Template(values)


def template_key(fields, lane):
    """Return the anchor shape and scale values of row ``lane`` with their Python types."""
    key = []
    for body in ("agent", "block"):
        for leaf in ("shape", "scale"):
            base = f"anchor/variation/{body}/{leaf}/current"
            value = get_value(
                {base + "/value": fields[base + "/value"][lane], base + "/type": fields[base + "/type"][lane]}, base
            )
            # Equal values of different types (Python float, NumPy float32) give different vertices.
            if isinstance(value, np.ndarray):
                key.append((f"{body}.{leaf}", "ndarray:" + value.dtype.str, value.item()))
            else:
                key.append((f"{body}.{leaf}", type(value).__name__, value))
    return tuple(key)


def _light(color):
    """``light_color(c).as_int()`` on ``[..., 3]`` integer colors: float32 ``min(1.2 * c, 255)``, rounded half to even."""
    scaled = np.minimum(np.float32(1.2) * np.asarray(color, np.float32), np.float32(255))
    return np.rint(scaled).astype(np.int64)


def _unique_rows(rows):
    """``np.unique(rows, axis=0, return_index=True, return_inverse=True)[1:]`` through one byte key per row."""
    rows = np.ascontiguousarray(rows)
    keys = rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).reshape(-1)
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    return first, inverse.reshape(-1)


class _Layout:
    """Slot layout of one template: where each primitive of a frame goes, in drawing order."""

    def __init__(self, template):
        from . import raster_cpu as cpu

        if any(len(g) != 4 for g in template.goal):
            raise NotImplementedError("The batched PushT renderer draws four-vertex goal polygons")
        slot = len(template.goal)
        self.goal = np.stack(template.goal) if template.goal else np.zeros((0, 4, 2))
        self.polys = {"agent": [], "block": []}
        self.fills = {"agent": [], "block": []}
        self.edges = {"agent": [], "block": []}
        self.circles = []
        walls = []
        for item in template.items:
            if item[0] == "poly":
                if len(item[2]) != 4:
                    raise NotImplementedError("The batched PushT renderer draws four-vertex polygons")
                self.polys[item[1]].append(item[2])
                self.fills[item[1]].append(slot)
                self.edges[item[1]].append(slot + 1 + np.arange(16).reshape(4, 4))
                slot += 17
            elif item[0] == "circle":
                self.circles.append((item[1], item[2], item[3], slot))
                slot += 2
            else:
                walls.append((item[1], item[2], slot))
                slot += 4
        self.slots = slot
        for body in ("agent", "block"):
            self.polys[body] = np.stack(self.polys[body]) if self.polys[body] else np.zeros((0, 4, 2))
            self.fills[body] = np.asarray(self.fills[body], np.int64)
            self.edges[body] = np.concatenate(self.edges[body]) if self.edges[body] else np.zeros((0, 4), np.int64)
        # The walls do not depend on the frame: draw their records once.
        self.wall_slots = np.array([base + np.arange(4) for _, _, base in walls], np.int64).reshape(-1, 4)
        self.walls = np.zeros((len(walls) * 4, RECORD), np.int32)
        for k, (a, b, _) in enumerate(walls):
            cpu.fat_segment(self.walls[4 * k : 4 * k + 4], pack(WALL_COLOR), *np.rint((*a, *b)).astype(np.int32))
        self.walls[:, STATIC] = 1
        polys = [self.goal, self.polys["agent"], self.polys["block"]]
        self.shapes = np.concatenate(polys)
        self.owners = np.repeat(np.arange(3), [len(p) for p in polys])
        self.starts = np.concatenate([np.arange(len(self.goal)), self.fills["agent"], self.fills["block"]])
        owner = {"agent": 1, "block": 2}
        discs = [(owner[o], *xy, round(r), round(r - 4), s) for o, xy, r, s in self.circles if o in owner]
        self.discs = np.array(discs, np.float64).reshape(-1, 6)
        # The CPU renderer redraws one box per body polygon and per disc: the slots of each share a group.
        self.groups = np.full(slot, -1, np.int64)
        group = 0
        for body in ("agent", "block"):
            for start in self.fills[body]:
                self.groups[start : start + 17] = group
                group += 1
        for o, _, _, s in self.circles:
            if o in owner:
                self.groups[s : s + 2] = group
                group += 1
        self.radius = max(2, int(self.discs[:, 3].max(initial=0)))


@cache
def _layout(template):
    return _Layout(template)


def primitives(fields):
    """Return ``(records [N, P, RECORD], counts [N], fallback [N])`` for the render fields of ``N`` rows.

    ``fallback`` marks rows that this renderer does not cover; draw those with pygame.
    """
    from . import raster_cpu as cpu

    count = len(fields["recipe/config"])
    config = fields["recipe/config"]
    fallback = config[:, 2] != 0  # action markers
    # Rows with equal anchor shape and scale fields (values and type tags) share a template.
    first, inverse = _unique_rows(_shape_bytes(fields, count))
    layouts = {}
    for index, lane0 in enumerate(first.tolist()):
        try:
            layouts[index] = _layout(_template(template_key(fields, lane0)))
        except NotImplementedError:
            fallback[inverse == index] = True
    records = np.empty((count, max([1] + [layout.slots for layout in layouts.values()]), RECORD), np.int32)
    records[fallback] = 0
    counts = np.zeros(count, np.int64)
    pose, color, light, goal, _ = _scene(fields, count)
    for index, layout in layouts.items():
        lanes = np.flatnonzero(inverse == index)
        cpu.build(
            lanes, pose, color, light, goal, layout.shapes, layout.owners, layout.starts,
            layout.discs, layout.walls, layout.wall_slots.reshape(-1), records,
        )  # fmt: skip
        counts[lanes] = layout.slots
    return records, counts, fallback


_SHAPE_FIELDS = tuple(
    f"anchor/variation/{body}/{leaf}/current/{part}"
    for body in ("agent", "block")
    for leaf in ("shape", "scale")
    for part in ("value", "type")
)


def _shape_bytes(fields, count):
    """``[N, bytes]``: the anchor shape and scale fields of each row, values and type tags, as bytes."""
    return np.concatenate(
        [np.ascontiguousarray(fields[key]).reshape(count, -1).view(np.uint8) for key in _SHAPE_FIELDS], 1
    )


def _scene(fields, count):
    """``(pose [N, 3, 5], color [N, 3], light [N, 3], goal [N], background [N])`` of host render ``fields``.

    A pose is (x, y, centre of gravity x, y, angle) of the goal, the agent and the block; colours are packed RGB.
    ``goal`` is True where the goal is drawn.
    """
    config = fields["recipe/config"]
    pose = np.zeros((count, 3, 5))
    pose[:, 0, :2] = fields["owned/variation/goal/position/current/value"]
    pose[:, 0, 4] = fields["owned/variation/goal/angle/current/value"].reshape(count)
    for b, body in enumerate(("agent", "block"), 1):
        pose[:, b, :2] = fields[f"current/body/{body}/position"]
        pose[:, b, 2:4] = fields[f"current/body/{body}/center_of_gravity"]
        pose[:, b, 4] = fields[f"current/body/{body}/angle"].reshape(count)
    color = np.stack([fields[f"owned/variation/{name}/color/current/value"] for name in ("goal", "agent", "block")], 1)
    color = color.astype(np.int64)
    goal = (fields["owned/variation/rendering/render_goal/current/value"].reshape(count) != 0) & (config[:, 3] != 0)
    background = pack(fields["owned/variation/background/color/current/value"].astype(np.int64))
    return pose, pack(color), pack(_light(color)), goal, background


def contact_writes(fields):
    """Return ``(x, y, brightness, color, count)`` blend writes per frame ``[N, W]`` in drawing order."""
    from . import raster_cpu as cpu

    segments = fields["current/render_segments"]
    counts = fields["current/render_segment_count"].astype(np.int64).reshape(-1)
    total = np.zeros(len(counts), np.int64)
    empty = np.zeros((len(counts), 0), np.int64)
    cpu.contacts(segments, counts, (empty, empty, empty.astype(np.float32), empty), total)
    shape = (len(counts), max(1, int(total.max(initial=0))))
    writes = tuple(np.zeros(shape, dtype) for dtype in (np.int64, np.int64, np.float32, np.int64))
    cpu.contacts(segments, counts, writes, total)
    return (*writes, total)


@cache
def resize_coefficients(source, size):
    """``cv2.resize`` INTER_LINEAR offsets and 11-bit weights for one axis, as OpenCV 5.0 computes them."""
    scale = 1.0 / (size / source)
    f = ((np.arange(size) + 0.5) * scale - 0.5).astype(np.float32)
    s = np.floor(f).astype(np.int64)
    f = (f - s.astype(np.float32)).astype(np.float32)
    a0 = np.rint((np.float32(1) - f) * np.float32(2048)).astype(np.int64)
    a1 = np.rint(f * np.float32(2048)).astype(np.int64)
    return s, a0, a1


def resize_supported(size):
    """Sizes whose rows OpenCV's SIMD loop covers on every instruction set, and that need no border taps."""
    s, _, _ = resize_coefficients(CANVAS, size)
    return size % 4 == 0 and size <= CANVAS and s.min() >= 0 and s.max() + 1 < CANVAS


try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - CPU-only installs
    HAVE_TRITON = False

if HAVE_TRITON:
    # Jitted code reads module constants only as ``constexpr``.
    _POLY, _CIRCLE, _ROWS, _ROFF = tl.constexpr(POLY), tl.constexpr(CIRCLE), tl.constexpr(ROWS), tl.constexpr(ROFF)
    _KIND, _COLOR, _BX0, _BY0, _BX1, _BY1 = (tl.constexpr(v) for v in (KIND, COLOR, BX0, BY0, BX1, BY1))
    _PN, _PX, _PY = tl.constexpr(PN), tl.constexpr(PX), tl.constexpr(PY)
    _CX, _CY, _CR = tl.constexpr(CX), tl.constexpr(CY), tl.constexpr(CR)
    _LX, _LY, _LSX, _LSY = tl.constexpr(LX), tl.constexpr(LY), tl.constexpr(LSX), tl.constexpr(LSY)
    _LDX, _LDY, _LERR, _LN, _LXINC = (tl.constexpr(v) for v in (LDX, LDY, LERR, LN, LXINC))
    _STATIC_ = tl.constexpr(STATIC)

    @triton.jit
    def _pick(i, v0, v1, v2, v3, v4):
        return tl.where(i == 0, v0, tl.where(i == 1, v1, tl.where(i == 2, v2, tl.where(i == 3, v3, v4))))

    @triton.jit
    def _sort2(a, b):
        return tl.minimum(a, b), tl.maximum(a, b)

    @triton.jit
    def _edge(i, n, x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, maxy, yv, BIG: tl.constexpr):
        """One ``draw_fillpoly`` edge: its intersection per row (BIG where it misses)."""
        prev = tl.where(i == 0, n - 1, i - 1)
        xa, ya = _pick(prev, x0, x1, x2, x3, x4), _pick(prev, y0, y1, y2, y3, y4)
        xb, yb = _pick(i, x0, x1, x2, x3, x4), _pick(i, y0, y1, y2, y3, y4)
        up = ya < yb
        ly, hy = tl.where(up, ya, yb), tl.where(up, yb, ya)
        lx, hx = tl.where(up, xa, xb), tl.where(up, xb, xa)
        edge = (i < n) & (ya != yb)
        on = edge & (((yv >= ly) & (yv < hy)) | ((yv == maxy) & (hy == maxy)))
        den = tl.where(edge, hy - ly, 1)
        # Integer division truncates toward zero, as the C cast of the double quotient does.
        q = ((yv - ly) * (hx - lx) + lx * den) // den
        return tl.where(on, q, BIG)

    @triton.jit
    def _level(i, n, x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, miny, maxy, yv, xa_v, xb_v, cover_a, cover_b):
        """Add the span of edge ``i`` when it is horizontal and strictly between the top and bottom rows."""
        prev = tl.where(i == 0, n - 1, i - 1)
        xa, ya = _pick(prev, x0, x1, x2, x3, x4), _pick(prev, y0, y1, y2, y3, y4)
        xb, yb = _pick(i, x0, x1, x2, x3, x4), _pick(i, y0, y1, y2, y3, y4)
        if (i < n) & (ya == yb) & (miny < yb) & (yb < maxy):
            lo, hi = tl.minimum(xa, xb), tl.maximum(xa, xb)
            row = (yv == yb)[:, None]
            cover_a = cover_a | (row & (xa_v[None, :] >= lo) & (xa_v[None, :] <= hi))
            cover_b = cover_b | (row & (xb_v[None, :] >= lo) & (xb_v[None, :] <= hi))
        return cover_a, cover_b

    @triton.jit
    def _poly_cover(yv, xa_v, xb_v, base, BIG: tl.constexpr):
        """``draw_fillpoly`` coverage of rows ``yv`` at columns ``xa_v`` and ``xb_v``."""
        n = tl.load(base + _PN)
        x0, x1, x2 = tl.load(base + _PX), tl.load(base + _PX + 1), tl.load(base + _PX + 2)
        x3, x4 = tl.load(base + _PX + 3), tl.load(base + _PX + 4)
        y0, y1, y2 = tl.load(base + _PY), tl.load(base + _PY + 1), tl.load(base + _PY + 2)
        y3, y4 = tl.load(base + _PY + 3), tl.load(base + _PY + 4)
        y1u, y2u = tl.where(n > 1, y1, y0), tl.where(n > 2, y2, y0)
        y3u, y4u = tl.where(n > 3, y3, y0), tl.where(n > 4, y4, y0)
        x1u, x2u = tl.where(n > 1, x1, x0), tl.where(n > 2, x2, x0)
        x3u, x4u = tl.where(n > 3, x3, x0), tl.where(n > 4, x4, x0)
        miny = tl.minimum(tl.minimum(tl.minimum(y0, y1u), tl.minimum(y2u, y3u)), y4u)
        maxy = tl.maximum(tl.maximum(tl.maximum(y0, y1u), tl.maximum(y2u, y3u)), y4u)
        if miny == maxy:
            minx = tl.minimum(tl.minimum(tl.minimum(x0, x1u), tl.minimum(x2u, x3u)), x4u)
            maxx = tl.maximum(tl.maximum(tl.maximum(x0, x1u), tl.maximum(x2u, x3u)), x4u)
            row = (yv == miny)[:, None]
            cover_a = row & (xa_v[None, :] >= minx) & (xa_v[None, :] <= maxx)
            cover_b = row & (xb_v[None, :] >= minx) & (xb_v[None, :] <= maxx)
        else:
            h0 = _edge(0, n, x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, maxy, yv, BIG)
            h1 = _edge(1, n, x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, maxy, yv, BIG)
            h2 = _edge(2, n, x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, maxy, yv, BIG)
            h3 = _edge(3, n, x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, maxy, yv, BIG)
            h4 = _edge(4, n, x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, maxy, yv, BIG)
            h0, h1 = _sort2(h0, h1)
            h3, h4 = _sort2(h3, h4)
            h2, h4 = _sort2(h2, h4)
            h2, h3 = _sort2(h2, h3)
            h0, h3 = _sort2(h0, h3)
            h0, h2 = _sort2(h0, h2)
            h1, h4 = _sort2(h1, h4)
            h1, h3 = _sort2(h1, h3)
            h1, h2 = _sort2(h1, h2)
            rows = (yv >= miny) & (yv <= maxy)
            # Rows with fewer than two (four) intersections get empty first (second) spans.
            two = rows & (h1 != BIG)
            four = rows & (h3 != BIG)
            lo1, hi1 = tl.where(two, h0, 1), tl.where(two, h1, 0)
            lo2, hi2 = tl.where(four, h2, 1), tl.where(four, h3, 0)
            xa, xb = xa_v[None, :], xb_v[None, :]
            cover_a = ((xa >= lo1[:, None]) & (xa <= hi1[:, None])) | ((xa >= lo2[:, None]) & (xa <= hi2[:, None]))
            cover_b = ((xb >= lo1[:, None]) & (xb <= hi1[:, None])) | ((xb >= lo2[:, None]) & (xb <= hi2[:, None]))
            for i in tl.static_range(5):
                cover_a, cover_b = _level(
                    i, n, x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, miny, maxy, yv, xa_v, xb_v, cover_a, cover_b
                )
        return cover_a, cover_b

    @triton.jit
    def _circle_cover(yv, xa_v, xb_v, base, table, max_radius):
        cx, cy, r = tl.load(base + _CX), tl.load(base + _CY), tl.load(base + _CR)
        dy = yv - cy
        valid = (dy >= -max_radius - 1) & (dy <= max_radius)
        row = tl.minimum(tl.maximum(dy + max_radius + 1, 0), 2 * max_radius + 1)
        index = (tl.minimum(r, max_radius) * (2 * max_radius + 2) + row) * 2
        lo = tl.where(valid, tl.load(table + index) + cx, 1)
        hi = tl.where(valid, tl.load(table + index + 1) + cx, 0)
        cover_a = (xa_v[None, :] >= lo[:, None]) & (xa_v[None, :] <= hi[:, None])
        cover_b = (xb_v[None, :] >= lo[:, None]) & (xb_v[None, :] <= hi[:, None])
        return cover_a, cover_b

    @triton.jit
    def _line_cover(yv, xa_v, xb_v, base):
        lx, ly = tl.load(base + _LX), tl.load(base + _LY)
        sx, sy = tl.load(base + _LSX), tl.load(base + _LSY)
        dx, dy = tl.load(base + _LDX), tl.load(base + _LDY)
        err, steps, xinc = tl.load(base + _LERR), tl.load(base + _LN), tl.load(base + _LXINC)
        if xinc != 0:
            # Steep: one four-pixel span per row.
            ky = (yv - ly) * sy
            nk = tl.where(dy > 0, (tl.maximum(ky, 0) * dx + err + dy - 1) // tl.maximum(dy, 1), 0)
            xk = lx + sx * nk
            valid = (ky >= 0) & (ky <= steps)
            lo, hi = tl.where(valid, xk - 1, 1), tl.where(valid, xk + 2, 0)
            cover_a = (xa_v[None, :] >= lo[:, None]) & (xa_v[None, :] <= hi[:, None])
            cover_b = (xb_v[None, :] >= lo[:, None]) & (xb_v[None, :] <= hi[:, None])
        else:
            # Shallow: one four-pixel span per column.
            ka = (xa_v - lx) * sx
            ya = ly + sy * ((tl.maximum(ka, 0) * dy - err + dx - 1) // tl.maximum(dx, 1))
            ya_lo, ya_hi = (
                tl.where((ka >= 0) & (ka <= steps), ya - 1, 1),
                tl.where((ka >= 0) & (ka <= steps), ya + 2, 0),
            )
            kb = (xb_v - lx) * sx
            yb = ly + sy * ((tl.maximum(kb, 0) * dy - err + dx - 1) // tl.maximum(dx, 1))
            yb_lo, yb_hi = (
                tl.where((kb >= 0) & (kb <= steps), yb - 1, 1),
                tl.where((kb >= 0) & (kb <= steps), yb + 2, 0),
            )
            y = yv[:, None]
            cover_a = (y >= ya_lo[None, :]) & (y <= ya_hi[None, :])
            cover_b = (y >= yb_lo[None, :]) & (y <= yb_hi[None, :])
        return cover_a, cover_b

    @triton.jit
    def _rows_cover(yv, xa_v, xb_v, base, intervals):
        offset, top, bottom = tl.load(base + _ROFF).to(tl.int64), tl.load(base + _BY0), tl.load(base + _BY1)
        inside = (yv >= top) & (yv <= bottom)
        row = (offset + tl.where(inside, yv - top, 0)) * 8
        xa, xb = xa_v[None, :], xb_v[None, :]
        cover_a = (yv[:, None] < 0) & (xa < 0)
        cover_b = cover_a
        for k in tl.static_range(4):
            lo = tl.load(intervals + row + 2 * k, mask=inside, other=1)[:, None]
            hi = tl.load(intervals + row + 2 * k + 1, mask=inside, other=0)[:, None]
            cover_a = cover_a | ((xa >= lo) & (xa <= hi))
            cover_b = cover_b | ((xb >= lo) & (xb <= hi))
        return cover_a, cover_b

    @triton.jit
    def _paint(yv, xa_v, xb_v, ca, cb, base, color, kind, table, max_radius, BIG: tl.constexpr, intervals):
        """Paint ``color`` where the primitive covers rows ``yv`` at columns ``xa_v`` (into ``ca``) and ``xb_v``."""
        if kind == _ROWS:
            cover_a, cover_b = _rows_cover(yv, xa_v, xb_v, base, intervals)
        elif kind == _POLY:
            cover_a, cover_b = _poly_cover(yv, xa_v, xb_v, base, BIG)
        elif kind == _CIRCLE:
            cover_a, cover_b = _circle_cover(yv, xa_v, xb_v, base, table, max_radius)
        else:
            cover_a, cover_b = _line_cover(yv, xa_v, xb_v, base)
        return tl.where(cover_a, color, ca), tl.where(cover_b, color, cb)

    @triton.jit
    def _blend_at(c, yv, xv, wy, wx, b, color):
        hit = (yv[:, None] == wy) & (xv[None, :] == wx)
        out = c * 0
        bb = (c * 0).to(tl.float32) + b
        for k in tl.static_range(3):
            s = 16 - 8 * k
            fg = ((color >> s) & 255).to(tl.float32) + bb * 0
            bg = ((c >> s) & 255).to(tl.float32)
            # Separate roundings: the C expression has no fused multiply-add on x86-64.
            v = libdevice.add_rn(libdevice.mul_rn(bb, fg), libdevice.mul_rn(1.0 - bb, bg))
            out = out | (tl.minimum(tl.maximum(v.to(tl.int32), 0), 255) << s)
        return tl.where(hit, out, c)

    @triton.jit
    def _blend4(c00, c01, c10, c11, y0, y1, x0, x1, wy, wx, b, color):
        return (
            _blend_at(c00, y0, x0, wy, wx, b, color),
            _blend_at(c01, y0, x1, wy, wx, b, color),
            _blend_at(c10, y1, x0, wy, wx, b, color),
            _blend_at(c11, y1, x1, wy, wx, b, color),
        )

    @triton.jit
    def _put(c00, c01, c10, c11, y0, y1, x0, x1, ylo, yhi, xlo, xhi, steep, major, minor, b, color, valid):
        """One ``draw_aaline`` blend at (major, minor), swapped back when the line is steep."""
        px = tl.where(steep, minor, major)
        py = tl.where(steep, major, minor)
        inside = valid & (px >= 0) & (px < 512) & (py >= 0) & (py < 512)
        if inside & (px >= xlo) & (px <= xhi) & (py >= ylo) & (py <= yhi):
            c00, c01, c10, c11 = _blend4(c00, c01, c10, c11, y0, y1, x0, x1, py, px, b, color)
        return c00, c01, c10, c11

    @triton.jit
    def _aaline(c00, c01, c10, c11, y0, y1, x0, x1, ylo, yhi, xlo, xhi, fx, fy, tx, ty, color):
        """Matches pygame 2.6.1 ``draw_aaline`` (blend on) in float32 with explicit roundings."""
        dx = libdevice.sub_rn(tx, fx)
        dy = libdevice.sub_rn(ty, fy)
        adx = tl.abs(dx.to(tl.float64))
        ady = tl.abs(dy.to(tl.float64))
        if (adx < 0.0001) & (ady < 0.0001):
            px = tl.floor(fx.to(tl.float64) + 0.5).to(tl.int32)
            py = tl.floor(fy.to(tl.float64) + 0.5).to(tl.int32)
            one = fx * 0.0 + 1.0
            c00, c01, c10, c11 = _put(
                c00, c01, c10, c11, y0, y1, x0, x1, ylo, yhi, xlo, xhi, False, px, py, one, color, True
            )
        else:
            steep = adx < ady
            fx, fy = tl.where(steep, fy, fx), tl.where(steep, fx, fy)
            tx, ty = tl.where(steep, ty, tx), tl.where(steep, tx, ty)
            dx, dy = tl.where(steep, dy, dx), tl.where(steep, dx, dy)
            back = dx < 0
            fx, tx = tl.where(back, tx, fx), tl.where(back, fx, tx)
            fy, ty = tl.where(back, ty, fy), tl.where(back, fy, ty)
            dx, dy = tl.where(back, -dx, dx), tl.where(back, -dy, dy)
            cl = fx * 0.0 - 1.0
            cr = fx * 0.0 + 512.0
            live = (tx > cl) & (fx < cr)
            g = libdevice.div_rn(dy, dx)
            left = fx < cl
            fy = tl.where(left, libdevice.add_rn(fy, libdevice.mul_rn(g, libdevice.sub_rn(cl, fx))), fy)
            fx = tl.where(left, cl, fx)
            right = tx > cr
            ty = tl.where(right, libdevice.add_rn(ty, libdevice.mul_rn(g, libdevice.sub_rn(cr, tx))), ty)
            tx = tl.where(right, cr, tx)
            down = g > 0
            live = live & tl.where(down, (ty > cl) & (fy < cr), (fy > cl) & (ty < cr))
            a = down & (fy < cl)
            fx = tl.where(a, libdevice.add_rn(fx, libdevice.div_rn(libdevice.sub_rn(cl, fy), g)), fx)
            fy = tl.where(a, cl, fy)
            a = down & (ty > cr)
            tx = tl.where(a, libdevice.add_rn(tx, libdevice.div_rn(libdevice.sub_rn(cr, ty), g)), tx)
            ty = tl.where(a, cr, ty)
            a = (g <= 0) & (ty < cl)
            tx = tl.where(a, libdevice.add_rn(tx, libdevice.div_rn(libdevice.sub_rn(cl, ty), g)), tx)
            ty = tl.where(a, cl, ty)
            a = (g <= 0) & (fy > cr)
            fx = tl.where(a, libdevice.add_rn(fx, libdevice.div_rn(libdevice.sub_rn(cr, fy), g)), fx)
            fy = tl.where(a, cr, fy)
            if live:
                fy = libdevice.add_rn(fy, 1.0)
                ty = libdevice.add_rn(ty, 1.0)
                xs = fx.to(tl.int32)
                ye = libdevice.add_rn(fy, libdevice.mul_rn(g, libdevice.sub_rn(xs.to(tl.float32), fx)))
                iy = ye
                first = tx > libdevice.add_rn(cl, 1.0)
                gap = libdevice.sub_rn((1 + xs).to(tl.float32), fx)
                yi = ye.to(tl.int32)
                br = libdevice.sub_rn(ye, yi.to(tl.float32))
                c00, c01, c10, c11 = _put(
                    c00,
                    c01,
                    c10,
                    c11,
                    y0,
                    y1,
                    x0,
                    x1,
                    ylo,
                    yhi,
                    xlo,
                    xhi,
                    steep,
                    xs,
                    yi,
                    libdevice.mul_rn(br, gap),
                    color,
                    first & (yi.to(tl.float32) < ye),
                )
                c00, c01, c10, c11 = _put(
                    c00,
                    c01,
                    c10,
                    c11,
                    y0,
                    y1,
                    x0,
                    x1,
                    ylo,
                    yhi,
                    xlo,
                    xhi,
                    steep,
                    xs,
                    yi - 1,
                    libdevice.mul_rn(libdevice.sub_rn(1.0, br), gap),
                    color,
                    first,
                )
                iy = tl.where(first, libdevice.add_rn(iy, g), iy)
                xs = tl.where(first, xs + 1, xs)
                xe = tl.ceil(tx.to(tl.float64)).to(tl.int32)
                second = fx < libdevice.sub_rn(cr, 1.0)
                ye2 = libdevice.add_rn(ty, libdevice.mul_rn(g, libdevice.sub_rn(xe.to(tl.float32), tx)))
                gap2 = libdevice.add_rn((1 - xe).to(tl.float32), tx)
                yi2 = ye2.to(tl.int32)
                br2 = libdevice.sub_rn(ye2, yi2.to(tl.float32))
                c00, c01, c10, c11 = _put(
                    c00,
                    c01,
                    c10,
                    c11,
                    y0,
                    y1,
                    x0,
                    x1,
                    ylo,
                    yhi,
                    xlo,
                    xhi,
                    steep,
                    xe,
                    yi2,
                    libdevice.mul_rn(br2, gap2),
                    color,
                    second & (yi2.to(tl.float32) < ye2),
                )
                c00, c01, c10, c11 = _put(
                    c00,
                    c01,
                    c10,
                    c11,
                    y0,
                    y1,
                    x0,
                    x1,
                    ylo,
                    yhi,
                    xlo,
                    xhi,
                    steep,
                    xe,
                    yi2 - 1,
                    libdevice.mul_rn(libdevice.sub_rn(1.0, br2), gap2),
                    color,
                    second,
                )
                for x in range(xs, xe):
                    y = iy.to(tl.int32)
                    br = libdevice.add_rn(libdevice.sub_rn(1.0, iy), y.to(tl.float32))
                    c00, c01, c10, c11 = _put(
                        c00, c01, c10, c11, y0, y1, x0, x1, ylo, yhi, xlo, xhi, steep, x, y - 1, br, color, True
                    )
                    c00, c01, c10, c11 = _put(
                        c00,
                        c01,
                        c10,
                        c11,
                        y0,
                        y1,
                        x0,
                        x1,
                        ylo,
                        yhi,
                        xlo,
                        xhi,
                        steep,
                        x,
                        y,
                        libdevice.sub_rn(1.0, br),
                        color,
                        y.to(tl.float32) < iy,
                    )
                    iy = libdevice.add_rn(iy, g)
        return c00, c01, c10, c11

    @triton.jit
    def _bin_kernel(
        records,
        counts,
        extents,
        lists,
        sizes,
        touch,
        segments,
        segment_counts,
        P,
        width,
        TILES: tl.constexpr,
        BLOCK: tl.constexpr,
        RECORD_: tl.constexpr,
        WRITE: tl.constexpr,
        SEGMENTS: tl.constexpr,
    ):
        """Count (and with ``WRITE`` list, in drawing order) the primitives of one frame per output tile."""
        f = tl.program_id(0).to(tl.int64)
        t = tl.arange(0, BLOCK)
        on = t < TILES
        ex0, ey0 = tl.load(extents + t * 4, mask=on, other=0), tl.load(extents + t * 4 + 1, mask=on, other=0)
        ex1, ey1 = tl.load(extents + t * 4 + 2, mask=on, other=-1), tl.load(extents + t * 4 + 3, mask=on, other=-1)
        filled = tl.zeros([BLOCK], tl.int32)
        moving = tl.zeros([BLOCK], tl.int32)
        count = tl.load(counts + f)
        for p in range(0, count):
            base = records + (f * P + p) * RECORD_
            if tl.load(base + _KIND) != 0:
                bx0, by0 = tl.load(base + _BX0), tl.load(base + _BY0)
                bx1, by1 = tl.load(base + _BX1), tl.load(base + _BY1)
                hit = on & (bx0 <= ex1) & (bx1 >= ex0) & (by0 <= ey1) & (by1 >= ey0)
                if WRITE:
                    tl.store(lists + (f * TILES + t) * width + filled, p, mask=hit)
                    if tl.load(base + _STATIC_) == 0:
                        moving += hit.to(tl.int32)
                filled += hit.to(tl.int32)
        tl.store(sizes + f * TILES + t, filled, mask=on)
        if WRITE:
            if SEGMENTS > 0:
                for k in range(0, tl.load(segment_counts + f)):
                    sb = segments + (f * SEGMENTS + k) * 8
                    ax = libdevice.rint(tl.load(sb)).to(tl.int32)
                    ay = libdevice.rint(tl.load(sb + 1)).to(tl.int32)
                    bx = libdevice.rint(tl.load(sb + 2)).to(tl.int32)
                    by = libdevice.rint(tl.load(sb + 3)).to(tl.int32)
                    near = (tl.minimum(ax, bx) - 2 <= ex1) & (tl.maximum(ax, bx) + 2 >= ex0)
                    near = near & (tl.minimum(ay, by) - 2 <= ey1) & (tl.maximum(ay, by) + 2 >= ey0)
                    moving += (on & near).to(tl.int32)
            tl.store(touch + f * TILES + t, moving, mask=on)

    @triton.jit
    def _frames_kernel(
        records,
        counts,
        background,
        wx_ptr,
        wy_ptr,
        wb_ptr,
        wc_ptr,
        wn_ptr,
        taps,
        w0_ptr,
        w1_ptr,
        table,
        out,
        static_index,
        static_out,
        dynamic,
        segments,
        segment_counts,
        intervals,
        bins,
        bin_counts,
        touch,
        out_words,
        static_words,
        P,
        W,
        size,
        max_radius,
        RECORD_: tl.constexpr,
        TILE: tl.constexpr,
        BIG: tl.constexpr,
        SEGMENTS: tl.constexpr = 0,
        BINS: tl.constexpr = 0,
        TOUCH: tl.constexpr = 0,
        WORDS: tl.constexpr = 0,
    ):
        # Frames on the first grid axis, which allows more than 65535 of them; one band of tile rows each.
        f = tl.program_id(0).to(tl.int64)
        band = tl.program_id(1)
        tiles = tl.cdiv(size, TILE)
        o = tl.arange(0, TILE)
        oy = band * TILE + o
        si = tl.load(static_index + f)
        if WORDS > 0:
            if si >= 0:
                # Tiles that only the background, goal and walls reach copy the frame they draw alone,
                # as 32-bit words of whole rows (``size`` and ``TILE`` are multiples of 4).
                j = tl.arange(0, WORDS)
                line = size * 3 // 4
                flag = tl.load(touch + (f * tiles + band) * tiles + (4 * j) // (3 * TILE), mask=j < line, other=1)
                keep = (oy < size)[:, None] & (flag == 0)[None, :]
                row = oy[:, None].to(tl.int64) * line + j[None, :]
                plane = size * line
                words = tl.load(static_words + si.to(tl.int64) * plane + row, mask=keep)
                tl.store(out_words + f * plane + row, words, mask=keep)
        my = oy < size
        y0 = tl.load(taps + oy, mask=my, other=0)
        y1 = y0 + 1
        ylo, yhi = tl.min(tl.where(my, y0, BIG)), tl.max(tl.where(my, y1, -BIG))
        for tx in range(0, tiles):
            tile = band * tiles + tx
            ox = tx * TILE + o
            mx = ox < size
            x0 = tl.load(taps + ox, mask=mx, other=0)
            x1 = x0 + 1
            xlo, xhi = tl.min(tl.where(mx, x0, BIG)), tl.max(tl.where(mx, x1, -BIG))
            pixel = ((f * size + oy[:, None]) * size + ox[None, :]) * 3
            mask = my[:, None] & mx[None, :]
            dx0, dy0 = tl.load(dynamic + f * 4), tl.load(dynamic + f * 4 + 1)
            dx1, dy1 = tl.load(dynamic + f * 4 + 2), tl.load(dynamic + f * 4 + 3)
            if TOUCH:
                touched = tl.load(touch + f * tiles * tiles + tile) > 0
            else:
                touched = (dx0 <= xhi) & (dx1 >= xlo) & (dy0 <= yhi) & (dy1 >= ylo)
            if (si >= 0) & (touched == 0):
                # Only the background, goal and walls reach this tile: copy it from the frame they draw alone.
                if WORDS == 0:
                    source = ((si.to(tl.int64) * size + oy[:, None]) * size + ox[None, :]) * 3
                    for k in tl.static_range(3):
                        tl.store(out + pixel + k, tl.load(static_out + source + k, mask=mask, other=0), mask=mask)
            else:
                bg = tl.load(background + f)
                c00 = tl.zeros([TILE, TILE], tl.int32) + bg
                c01 = c00
                c10 = c00
                c11 = c00
                if BINS > 0:
                    n = tl.load(bin_counts + f * tiles * tiles + tile)
                else:
                    n = tl.load(counts + f)
                for q in range(0, n):
                    if BINS > 0:
                        p = tl.load(bins + (f * tiles * tiles + tile) * BINS + q)
                    else:
                        p = q
                    base = records + (f * P + p) * RECORD_
                    kind = tl.load(base + _KIND)
                    bx0, by0 = tl.load(base + _BX0), tl.load(base + _BY0)
                    bx1, by1 = tl.load(base + _BX1), tl.load(base + _BY1)
                    if (kind != 0) & (bx0 <= xhi) & (bx1 >= xlo) & (by0 <= yhi) & (by1 >= ylo):
                        color = tl.load(base + _COLOR)
                        c00, c01 = _paint(y0, x0, x1, c00, c01, base, color, kind, table, max_radius, BIG, intervals)
                        c10, c11 = _paint(y1, x0, x1, c10, c11, base, color, kind, table, max_radius, BIG, intervals)
                if SEGMENTS > 0:
                    # Contact lines from the state: (x0, y0, x1, y1, r, g, b, a) per segment, drawn in order.
                    ns = tl.load(segment_counts + f)
                    for k in range(0, ns):
                        sb = segments + (f * SEGMENTS + k) * 8
                        ax = libdevice.rint(tl.load(sb)).to(tl.int32)
                        ay = libdevice.rint(tl.load(sb + 1)).to(tl.int32)
                        bx = libdevice.rint(tl.load(sb + 2)).to(tl.int32)
                        by = libdevice.rint(tl.load(sb + 3)).to(tl.int32)
                        near = (tl.minimum(ax, bx) - 2 <= xhi) & (tl.maximum(ax, bx) + 2 >= xlo)
                        near = near & (tl.minimum(ay, by) - 2 <= yhi) & (tl.maximum(ay, by) + 2 >= ylo)
                        if near:
                            r = libdevice.rint(tl.load(sb + 4)).to(tl.int32)
                            g = libdevice.rint(tl.load(sb + 5)).to(tl.int32)
                            b = libdevice.rint(tl.load(sb + 6)).to(tl.int32)
                            color = (r << 16) | (g << 8) | b
                            c00, c01, c10, c11 = _aaline(
                                c00, c01, c10, c11, y0, y1, x0, x1, ylo, yhi, xlo, xhi,
                                ax.to(tl.float32), ay.to(tl.float32), bx.to(tl.float32), by.to(tl.float32), color,
                            )  # fmt: skip
                else:
                    nw = tl.load(wn_ptr + f)
                    for w in range(0, nw):
                        wx, wy = tl.load(wx_ptr + f * W + w), tl.load(wy_ptr + f * W + w)
                        if (wx >= xlo) & (wx <= xhi) & (wy >= ylo) & (wy <= yhi):
                            b, color = tl.load(wb_ptr + f * W + w), tl.load(wc_ptr + f * W + w)
                            c00 = _blend_at(c00, y0, x0, wy, wx, b, color)
                            c01 = _blend_at(c01, y0, x1, wy, wx, b, color)
                            c10 = _blend_at(c10, y1, x0, wy, wx, b, color)
                            c11 = _blend_at(c11, y1, x1, wy, wx, b, color)
                a0 = tl.load(w0_ptr + ox, mask=mx, other=0)[None, :]
                a1 = tl.load(w1_ptr + ox, mask=mx, other=0)[None, :]
                b0 = tl.load(w0_ptr + oy, mask=my, other=0)[:, None]
                b1 = tl.load(w1_ptr + oy, mask=my, other=0)[:, None]
                for k in tl.static_range(3):
                    s = 16 - 8 * k
                    h0 = ((c00 >> s) & 255) * a0 + ((c01 >> s) & 255) * a1
                    h1 = ((c10 >> s) & 255) * a0 + ((c11 >> s) & 255) * a1
                    v = (((h0 >> 4) * b0) >> 16) + (((h1 >> 4) * b1) >> 16)
                    v = tl.minimum(tl.maximum((v + 2) >> 2, 0), 255)
                    tl.store(out + pixel + k, v.to(tl.uint8), mask=mask)


_STAGING = {}


def _upload(array, dtype, device):
    """Copy ``array`` to ``device`` through reused pinned buffers, so later host work overlaps the copy."""
    host = torch.from_numpy(np.ascontiguousarray(array)).to(dtype)
    flat = host.reshape(-1)
    slots = _STAGING.setdefault((dtype, str(device)), [])
    for slot in slots:
        buffer, event = slot
        if buffer.numel() >= flat.numel() and event.query():
            break
    else:
        buffer = torch.empty(max(flat.numel(), 1 << 16), dtype=dtype, pin_memory=True)
        slot = [buffer, torch.cuda.Event()]
        slots.append(slot)
    view = slot[0][: flat.numel()]
    view.copy_(flat)
    target = view.to(device, non_blocking=True).view(host.shape)
    slot[1] = torch.cuda.Event()
    slot[1].record()
    return target


def _launch(records, counts, background, writes, size, device, static_index, static_out, dynamic, tile):
    s, a0, a1 = resize_coefficients(CANVAS, size)
    n = len(records)
    max_radius = max(2, int(records[..., CR].max(initial=0, where=records[..., KIND] == CIRCLE)))
    wx, wy, wb, wc, wn = writes
    out = torch.empty((n, size, size, 3), dtype=torch.uint8, device=device)
    if n == 0:
        return out
    # One upload: every argument is a 32-bit slice of one buffer; the brightness keeps its float bits.
    parts = [
        records, counts, background, wx, wy, np.asarray(wb, np.float32).view(np.int32), wc, wn, s, a0, a1,
        circle_table(max_radius), static_index, dynamic,
    ]  # fmt: skip
    flat = [np.ascontiguousarray(part).astype(np.int32, copy=False).reshape(-1) for part in parts]
    packed = _upload(np.concatenate(flat), torch.int32, device)
    views, offset = [], 0
    for part in flat:
        views.append(packed[offset : offset + len(part)])
        offset += len(part)
    views[5] = views[5].view(torch.float32)
    _frames_kernel[(n, triton.cdiv(size, tile))](
        *views[:12], out, views[12], static_out, views[13], out, views[1], views[1], views[1], views[1],
        views[1], out, out, records.shape[1], wx.shape[1], size, max_radius,
        RECORD_=RECORD, TILE=tile, BIG=2**30,
    )  # fmt: skip
    return out


def launch_segments(
    records,
    counts,
    background,
    segments,
    segment_counts,
    size,
    tile=32,
    intervals=None,
    static=None,
    bins=None,
    warps=4,
):
    """Draw device ``records`` with the state contact lines ``segments [N, S, 8]``, all on the device.

    ``static`` is ``(index [N], frames [K, size, size, 3], dynamic boxes [N, 4])``: frames of the background,
    goal and walls alone, with index −1 for frames that draw them;
    ``bins`` is ``(lists [N, tiles, K], counts [N, tiles])`` from ``tile_bins``.
    """
    device = records.device
    n = len(records)
    out = torch.empty((n, size, size, 3), dtype=torch.uint8, device=device)
    if n == 0:
        return out
    tables = _device_tables(size, device)
    max_radius = tables["max_radius"]
    none = torch.full((n,), -1, dtype=torch.int32, device=device)
    index, frames, boxes = (
        static if static is not None else (none, out, torch.zeros((n, 4), dtype=torch.int32, device=device))
    )
    lists, sizes, touch = (*bins, None)[:3] if bins is not None else (none, none, None)
    words = size % 4 == 0 and tile % 4 == 0 and static is not None and touch is not None
    _frames_kernel[(n, triton.cdiv(size, tile))](
        records.contiguous(), counts.to(torch.int32), background.to(torch.int32),
        none, none, none.to(torch.float32), none, none,
        tables["taps"], tables["w0"], tables["w1"], tables["table"], out,
        index, frames, boxes,
        segments.contiguous(), segment_counts.reshape(-1).to(torch.int32),
        none if intervals is None else intervals.contiguous(), lists, sizes,
        none if touch is None else touch,
        out.view(-1).view(torch.int32) if words else out, frames.reshape(-1).view(torch.int32) if words else out,
        records.shape[1], 1, size, max_radius,
        RECORD_=RECORD, TILE=tile, BIG=2**30, SEGMENTS=segments.shape[1],
        BINS=0 if bins is None else lists.shape[-1], TOUCH=int(touch is not None and static is not None),
        WORDS=triton.next_power_of_2(size * 3 // 4) if words else 0, num_warps=warps,
    )  # fmt: skip
    return out


def tile_extents(size, tile, device):
    """Inclusive canvas extents ``[tiles, 4]`` (x0, y0, x1, y1) that each output tile reads."""
    s, _, _ = resize_coefficients(CANVAS, size)
    count = -(-size // tile)
    lo = np.array([s[k * tile] for k in range(count)])
    hi = np.array([s[min(size - 1, k * tile + tile - 1)] + 1 for k in range(count)])
    ty, tx = np.divmod(np.arange(count * count), count)
    return torch.as_tensor(np.stack([lo[tx], lo[ty], hi[tx], hi[ty]], 1), device=device)


def tile_bins(records, counts, size, tile, segments=None, segment_counts=None, width=None):
    """Per frame and output tile, the indices of the primitives that reach it, in drawing order.

    On CUDA, with ``segments``, also returns per tile the number of agent, block and contact
    primitives that reach it; tiles without any copy the goal-and-wall frame. With ``width`` (at
    least the most primitives that one tile can list), the lists come from one pass without a sync.
    """
    extents = tile_extents(size, tile, records.device)
    if HAVE_TRITON and records.is_cuda:
        n, slots = records.shape[:2]
        tiles = len(extents)
        records = records.to(torch.int32).contiguous()
        counts = counts.to(torch.int32).contiguous()
        extents = extents.to(torch.int32).contiguous()
        sizes = torch.empty((n, tiles), dtype=torch.int32, device=records.device)
        block = triton.next_power_of_2(tiles)
        touch = torch.empty((n, tiles), dtype=torch.int32, device=records.device)
        given = segments is not None
        if not given:
            segments = torch.zeros((n, 0, 8), dtype=torch.float64, device=records.device)
            segment_counts = torch.zeros(n, dtype=torch.int32, device=records.device)
        segments = segments.to(torch.float64).contiguous()
        segment_counts = segment_counts.reshape(-1).to(torch.int32)
        meta = dict(TILES=tiles, BLOCK=block, RECORD_=RECORD, SEGMENTS=segments.shape[1])
        args = (touch, segments, segment_counts, slots)
        if width is None:
            _bin_kernel[(n,)](records, counts, extents, sizes, sizes, *args, 0, WRITE=False, **meta)
            width = max(1, int(sizes.max().item()))
        dtype = torch.uint8 if slots <= 256 else torch.int32
        lists = torch.empty((n, tiles, width), dtype=dtype, device=records.device)
        _bin_kernel[(n,)](records, counts, extents, lists, sizes, *args, width, WRITE=True, **meta)
        return (lists, sizes, touch) if given else (lists, sizes)
    kind = records[..., KIND]
    live = (kind != 0) & (torch.arange(records.shape[1], device=records.device).view(1, -1) < counts.view(-1, 1))
    bx0, by0, bx1, by1 = (records[..., k].unsqueeze(-1) for k in (BX0, BY0, BX1, BY1))
    hit = (
        live.unsqueeze(-1)
        & (bx0 <= extents[:, 2])
        & (bx1 >= extents[:, 0])
        & (by0 <= extents[:, 3])
        & (by1 >= extents[:, 1])
    )
    hit = hit.transpose(1, 2)  # [N, tiles, P]
    position = torch.cumsum(hit, -1, dtype=torch.int32) - 1
    sizes = position[..., -1] + 1
    width = max(1, int(sizes.max().item()))
    frame, tile_index, prim = torch.nonzero(hit, as_tuple=True)
    lists = torch.zeros((*hit.shape[:2], width), dtype=torch.int32, device=records.device)
    lists[frame, tile_index, position[frame, tile_index, prim]] = prim.to(torch.int32)
    return lists, sizes.to(torch.int32).contiguous()


@cache
def _device_tables(size, device):
    s, a0, a1 = resize_coefficients(CANVAS, size)
    # Agent discs reach a radius of 0.375 * 60; the table covers every variation scale.
    max_radius = 32
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.int32, device=device)  # noqa: E731
    return dict(taps=t(s), w0=t(a0), w1=t(a1), table=t(circle_table(max_radius)), max_radius=max_radius)


def draw_cuda(records, counts, background, writes, size, device, tile=32):
    """Return ``[N, size, size, 3]`` uint8 frames of host ``records`` and ``writes`` (see ``contact_writes``) on CUDA.

    Tiles that no agent, block or contact primitive reaches copy the frame that the background,
    goal and walls draw alone, which frames with the same goal and colors share.
    """
    n = len(records)
    static = (records[..., STATIC] != 0) & (records[..., KIND] != 0)
    dynamic_kind = (records[..., KIND] != 0) & ~static
    big = 1 << 30
    box = np.stack(
        [
            np.where(dynamic_kind, records[..., BX0], big).min(1, initial=big),
            np.where(dynamic_kind, records[..., BY0], big).min(1, initial=big),
            np.where(dynamic_kind, records[..., BX1], -big).max(1, initial=-big),
            np.where(dynamic_kind, records[..., BY1], -big).max(1, initial=-big),
        ],
        1,
    )
    wx, wy, _, _, wn = writes
    used = np.arange(wx.shape[1])[None] < wn[:, None]
    box[:, 0] = np.minimum(box[:, 0], np.where(used, wx, big).min(1, initial=big))
    box[:, 1] = np.minimum(box[:, 1], np.where(used, wy, big).min(1, initial=big))
    box[:, 2] = np.maximum(box[:, 2], np.where(used, wx, -big).max(1, initial=-big))
    box[:, 3] = np.maximum(box[:, 3], np.where(used, wy, -big).max(1, initial=-big))
    fixed = np.where(static[..., None], records, 0)
    # Every frame draws the same walls, so the background and the leading goal slots identify the frame.
    goal = np.cumprod(records[..., STATIC] != 0, 1).astype(np.bool_)[:, :3]
    keys = np.concatenate([background[:, None], np.where(goal[..., None], records[:, :3], 0).reshape(n, -1)], 1)
    first, inverse = _unique_rows(keys)
    if n and len(first) * 4 <= n:
        none = np.zeros((len(first), 1), np.int64)
        static_out = _launch(
            fixed[first], counts[first], background[first], (none, none, none.astype(np.float32), none, none[:, 0]),
            size, device, np.full(len(first), -1), torch.empty(0, dtype=torch.uint8, device=device),
            np.zeros((len(first), 4), np.int64), tile,
        )  # fmt: skip
        index = inverse
    else:
        static_out, index = torch.empty(0, dtype=torch.uint8, device=device), np.full(n, -1)
    return _launch(records, counts, background, writes, size, device, index, static_out, box, tile)


#: CPU render state kept between calls: the template of each shape byte string, the frames of the background, goal
#: and walls alone ("plates") with their least recently used order, and the plates that one call left undrawn.
_CPU_LOCK = threading.Lock()
_CPU_LAYOUTS: dict = {}
_CPU_PLATES: OrderedDict = OrderedDict()
_CPU_SEEN: OrderedDict = OrderedDict()
_CPU_PLATE_LIMIT = 64
_CPU_SCRATCH = threading.local()
#: At least the blend writes of one contact line (``raster_cpu._aaline`` writes at most 2 per column and 4 more).
_LINE_WRITES = 1040


@cache
def _cpu_tables(size, radius):
    """Canvas taps of the ``size`` resize, their positions, the circle spans and the resize taps and weights."""
    s, a0, a1 = resize_coefficients(CANVAS, size)
    taps = np.unique(np.concatenate([s, s + 1]))
    position = np.full(CANVAS, -1, np.int64)
    position[taps] = np.arange(len(taps))
    edge = np.searchsorted(taps, np.arange(CANVAS + 1))
    return taps, edge, position, circle_table(radius), position[s], position[s + 1], a0, a1


def _cpu_scratch(threads, slots, taps, width):
    """Record, canvas and blend-write buffers for ``threads`` Numba threads, kept on this Python thread."""
    held = _CPU_SCRATCH.__dict__
    canvas = held.get("canvas")
    if canvas is None or len(canvas) < threads or canvas.shape[1] != taps:
        canvas = held["canvas"] = np.empty((threads, taps, taps), np.int32)
    writes = held.get("writes")
    if writes is None or len(writes[0]) < threads or writes[0].shape[1] < width:
        dtypes = (np.int64, np.int64, np.float32, np.int64)
        writes = held["writes"] = tuple(np.empty((threads, width), dtype) for dtype in dtypes)
    records = np.empty((threads, slots, RECORD), np.int32)
    return (records, canvas[:threads], *(w[:threads] for w in writes))


def _cpu_layouts(fields, count):
    """``([(template key, layout, rows)], unsupported [N])``: the rows of each template the CPU renderer draws."""
    raw = _shape_bytes(fields, count)
    if (raw == raw[0]).all():
        first, inverse = [0], None
    else:
        first, inverse = _unique_rows(raw)
        first = first.tolist()
    groups, unsupported = [], np.zeros(count, np.bool_)
    for index, lane in enumerate(first):
        rows = np.arange(count) if inverse is None else np.flatnonzero(inverse == index)
        name = raw[lane].tobytes()
        entry = _CPU_LAYOUTS.get(name)
        if entry is None:
            key = template_key(fields, lane)
            try:
                entry = (key, _layout(_template(key)))
            except NotImplementedError:
                entry = (key, None)
            with _CPU_LOCK:
                if len(_CPU_LAYOUTS) >= 256:
                    _CPU_LAYOUTS.clear()
                _CPU_LAYOUTS[name] = entry
        if entry[1] is None:
            unsupported[rows] = True
        else:
            groups.append((entry[0], entry[1], rows))
    return groups, unsupported


class _CpuDraw:
    """``raster_cpu.frames`` for the rows of one template: ``(rows, dest, plate, plates, static, target)``."""

    def __init__(self, layout, size, scene, segments, used, width):
        self.layout, self.tables = layout, _cpu_tables(size, layout.radius)
        self.inputs, self.width = (scene, segments, used), width
        self.arrays = (
            layout.shapes, layout.owners, layout.starts, layout.discs, layout.walls, layout.wall_slots.reshape(-1),
        )  # fmt: skip

    def __call__(self, rows, dest, plate, plates, static, target):
        import numba

        from . import raster_cpu as cpu

        threads = min(numba.get_num_threads(), len(rows))
        scratch = _cpu_scratch(threads, self.layout.slots, len(self.tables[0]), self.width)
        cpu.frames(
            rows, dest, *self.inputs, plate, plates, self.arrays, self.layout.groups, self.tables, scratch, static,
            target,
        )  # fmt: skip


def _cpu_plates(key, rows, scene, size, draw):
    """Return the plate index of each of ``rows`` (−1: draw the whole frame) and the plates ``[K, size, size, 3]``.

    A plate is the frame of the background, goal and walls alone. The plates are kept between calls. A call draws
    the plates it needs when they are at most a quarter of its rows, or when an earlier call needed them too.
    """
    pose, color, _, goal, background = scene
    n, shown = len(rows), goal[rows]
    keys = np.zeros((n, 6), np.int64)
    keys[:, 0] = background[rows]
    keys[:, 1] = np.where(shown, color[rows, 0], 0)
    keys[:, 2] = shown
    keys[:, 3:] = np.where(shown[:, None], np.ascontiguousarray(pose[rows, 0][:, [0, 1, 4]]).view(np.int64), 0)
    if (keys == keys[0]).all():
        first, inverse = np.zeros(1, np.int64), np.zeros(n, np.int64)
    else:
        first, inverse = _unique_rows(keys)
    index, plates, missing = np.full(len(first), -1, np.int64), [], []
    with _CPU_LOCK:
        for u, row in enumerate(first.tolist()):
            name = (key, size, keys[row].tobytes())
            plate = _CPU_PLATES.get(name)
            if plate is not None:
                _CPU_PLATES.move_to_end(name)
            elif len(first) * 4 <= n or name in _CPU_SEEN:
                missing.append((u, name))
            else:
                _CPU_SEEN[name] = None
                if len(_CPU_SEEN) > 4 * _CPU_PLATE_LIMIT:
                    _CPU_SEEN.popitem(last=False)
                continue
            index[u] = len(plates)
            plates.append(plate)
    if missing:
        drawn = np.empty((len(missing), size, size, 3), np.uint8)
        reps = rows[first[[u for u, _ in missing]]]
        none = np.full(len(goal), -1, np.int64)
        draw(reps, np.arange(len(reps)), none, np.empty((0, size, size, 3), np.uint8), True, drawn)
        with _CPU_LOCK:
            for (u, name), frame in zip(missing, drawn):
                plates[index[u]] = _CPU_PLATES[name] = frame.copy()
                _CPU_SEEN.pop(name, None)
            while len(_CPU_PLATES) > _CPU_PLATE_LIMIT:
                _CPU_PLATES.popitem(last=False)
    stack = np.stack(plates) if plates else np.empty((0, size, size, 3), np.uint8)
    return index[inverse], stack


def _markers(fields, frames, rows, size):
    """Draw the source's action marker, ``cv2.drawMarker`` on the resized frame, on ``frames[rows]``."""
    import cv2

    for lane in rows.tolist():
        row = {key: fields[key][lane] for key in ("current/latest/value", "current/latest/type")}
        coord = (np.array(get_value(row, "current/latest")) / 512 * 96).astype(np.int32)
        frames[lane] = cv2.drawMarker(
            frames[lane], coord, color=(255, 0, 0), markerType=cv2.MARKER_CROSS,
            markerSize=int(8 / 96 * size), thickness=int(1 / 96 * size),
        )  # fmt: skip


def render_cpu(fields, size, fallback, out=None):
    """Return ``[N, size, size, 3]`` uint8 CPU frames for host render ``fields`` with the Numba rasterizer.

    One Numba call draws the frames of each template. ``fallback(rows)`` draws the rows whose shapes this renderer
    does not cover with pygame. The frames are drawn into ``out`` when it is a matching contiguous CPU tensor.
    """
    import numba

    count = len(fields["recipe/config"])
    shape = (count, size, size, 3)
    if (
        out is None
        or (out.shape, out.dtype, out.device.type) != (shape, torch.uint8, "cpu")
        or not out.is_contiguous()
    ):
        out = torch.empty(shape, dtype=torch.uint8)
    if count == 0:
        return out
    frames = out.numpy()
    groups, unsupported = _cpu_layouts(fields, count)
    scene = _scene(fields, count)
    segments = np.ascontiguousarray(fields["current/render_segments"], np.float64)
    used = fields["current/render_segment_count"].astype(np.int64).reshape(-1)
    width = _LINE_WRITES * max(1, int(min(used.max(), segments.shape[1])))
    threads = numba.get_num_threads()
    numba.set_num_threads(max(1, min(threads, count)))
    try:
        for key, layout, rows in groups:
            draw = _CpuDraw(layout, size, scene, segments, used, width)
            index, plates = _cpu_plates(key, rows, scene, size, draw)
            plate = np.full(count, -1, np.int64)
            plate[rows] = index
            draw(rows, rows, plate, plates, False, frames)
    finally:
        numba.set_num_threads(threads)
    config = fields["recipe/config"]
    if (config[:, 2] != 0).any():
        present = np.asarray(fields["current/latest_present"]).reshape(count, -1)[:, 0] != 0
        _markers(fields, frames, np.flatnonzero((config[:, 2] != 0) & present & ~unsupported), size)
    if unsupported.any():
        lanes = np.flatnonzero(unsupported)
        frames[lanes] = np.asarray(fallback({key: value[lanes] for key, value in fields.items()}))
    return out


def render(fields, size, device, fallback, chunk=1024):
    """Return ``[N, size, size, 3]`` uint8 frames on CUDA ``device`` for host render ``fields``.

    ``fallback(rows)`` draws the rows this renderer does not cover (action markers) with pygame.
    The host prepares each chunk while the GPU draws the previous one.
    """
    count = len(fields["recipe/config"])
    parts = []
    for start in range(0, count, chunk):
        rows = {key: value[start : start + chunk] for key, value in fields.items()}
        records, counts, unsupported = primitives(rows)
        background = pack(rows["owned/variation/background/color/current/value"].astype(np.int64))
        frames = draw_cuda(records, counts, background, contact_writes(rows), size, device)
        if unsupported.any():
            lanes = np.flatnonzero(unsupported)
            drawn = fallback({key: value[lanes] for key, value in rows.items()})
            frames[torch.as_tensor(lanes, device=device)] = torch.as_tensor(drawn, device=device)
        parts.append(frames)
    if len(parts) == 1:
        return parts[0]
    return torch.cat(parts) if parts else torch.empty((0, size, size, 3), dtype=torch.uint8, device=device)
