"""Batched PushT frames equal to the pygame frames: integer ports of the pygame 2.6.1 draw calls and cv2 resize.

The host turns render State into a list of primitives per frame in pygame's painter order: polygon
fills (``draw_fillpoly``), filled circles (``draw_circle_filled``), width-4 lines (``draw_line_width``)
and antialiased contact lines (``draw_aaline``). The device evaluates the primitives only at the 512 px
canvas pixels that ``cv2.resize(INTER_LINEAR)`` reads, then applies the fixed-point resize.
"""

from __future__ import annotations

import math
from functools import cache, lru_cache

import numpy as np
import torch

from .simulation.numeric import get_value

CANVAS = 512
#: Primitive kinds.
POLY, CIRCLE, LINE = 1, 2, 3
#: Record layout: kind, packed colour, inclusive bounding box, then kind-specific integers.
KIND, COLOR, BX0, BY0, BX1, BY1 = range(6)
#: POLY: point count, then up to five x and five y values.
PN, PX, PY = 6, 7, 12
#: CIRCLE: centre and radius.
CX, CY, CR = 6, 7, 8
#: LINE: start, direction signs, extents, initial error, step count, and 1 when spans are horizontal.
LX, LY, LSX, LSY, LDX, LDY, LERR, LN, LXINC = range(6, 15)
#: 1 for primitives that only the Variation and goal place (goal, walls).
STATIC = 15
RECORD = 17
#: Most contact segments a State holds.
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


def circle_table(max_radius):
    """Return ``[max_radius + 1, 2 * max_radius + 2, 2]`` span offsets; empty rows have lo > hi."""
    table = np.zeros((max_radius + 1, 2 * max_radius + 2, 2), np.int64)
    table[..., 0] = 1
    for r in range(1, max_radius + 1):
        spans = circle_spans(r)
        table[r, max_radius - r : max_radius + r + 2] = spans
    return table


@lru_cache(maxsize=4096)
def _pow_half(n):
    # Python's ``** 0.5`` calls C ``pow``, which need not equal ``sqrt``.
    return float(n) ** 0.5


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
    """``light_color(c).as_int()`` on ``[..., 3]`` integer colours: float32 ``min(1.2 * c, 255)``, rounded half to even."""
    scaled = np.minimum(np.float32(1.2) * np.asarray(color, np.float32), np.float32(255))
    return np.rint(scaled).astype(np.int64)


def _transform(px, py, angle, cogx, cogy):
    """Chipmunk's body transform ``(a, b, c, d, tx, ty)`` after ``body.angle = angle; body.position = (px, py)``.

    Setting the position stores the centre of gravity ``R cog + position``; the transform then
    subtracts ``R cog`` again, which can differ from ``position`` in the last bit.
    """
    cos = np.array([math.cos(a) for a in angle.tolist()], np.float64)
    sin = np.array([math.sin(a) for a in angle.tolist()], np.float64)
    gx = (cos * cogx + -sin * cogy) + px
    gy = (sin * cogx + cos * cogy) + py
    tx = gx - (cogx * cos - cogy * sin)
    ty = gy - (cogx * sin + cogy * cos)
    return cos, sin, -sin, cos, tx, ty


def _apply(t, x, y):
    """``cpTransformPoint`` for local points ``x, y`` of shape ``[n]`` and per-frame transforms ``[N]``."""
    a, b, c, d, tx, ty = (v[:, None] for v in t)
    return a * x + c * y + tx, b * x + d * y + ty


def _sign(v, otherwise):
    return 1 if v > 0 else (-1 if v < 0 else otherwise)


def _clip_line(x1, y1, x2, y2):
    """Pygame's ``clip_line`` on the full canvas; None when the line misses it."""
    p1 = x1 - x2
    p2, p3 = -p1, y1 - y2
    p4 = -p3
    q1, q2, q3, q4 = x1, CANVAS - x1, y1, CANVAS - y1
    nmax, pmin = 0.0, 1.0
    if (p1 == 0 and q1 < 0) or (p2 == 0 and q2 < 0) or (p3 == 0 and q3 < 0) or (p4 == 0 and q4 < 0):
        return None
    for p, q, pp, qq in ((p1, q1, p2, q2), (p3, q3, p4, q4)):
        if p:
            r1, r2 = q / p, qq / pp
            if p < 0:
                nmax, pmin = max(nmax, r1) if r1 > nmax else nmax, r2 if r2 < pmin else pmin
            else:
                nmax, pmin = r2 if r2 > nmax else nmax, r1 if r1 < pmin else pmin
    if nmax > pmin:
        return None

    def offset(v):
        return int(v - 0.5) if v < 0 else int(v + 0.5)

    return x1 + offset(p2 * nmax), y1 + offset(p4 * nmax), x1 + offset(p2 * pmin), y1 + offset(p4 * pmin)


def clipped_line(x1, y1, x2, y2, width=4):
    """Return ``(x, y, steps)`` of a ``draw_line_width`` call whose ends leave the canvas, or None.

    pygame keeps the Bresenham error of the whole line, restarts it at the clipped start, and
    continues past the clipped end while a span still touches the canvas.
    """
    xinc = abs(x1 - x2) <= abs(y1 - y2)
    dx, sx = abs(x2 - x1), (1 if x1 < x2 else -1)
    dy, sy = abs(y2 - y1), (1 if y1 < y2 else -1)
    err = dx if dx > dy else -dy
    err = err // 2 if err >= 0 else -((-err) // 2)
    end_x, end_y = x2, y2
    clipped = _clip_line(x1, y1, x2, y2)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    x, y = x1, y1
    lt = (x1 if xinc else y1) - (width - 1) // 2
    rb = (x1 if xinc else y1) + width // 2
    steps = 0
    while _sign(x1 - x2, sx) != sx or _sign(y1 - y2, sy) != sy:
        steps += 1
        e2 = err
        if e2 > -dx:
            err -= dy
            x1 += sx
            if xinc:
                lt, rb = lt + sx, rb + sx
        if e2 < dy:
            err += dx
            y1 += sy
            if not xinc:
                lt, rb = lt + sy, rb + sy

    def inside(px, py):
        return 0 <= px < CANVAS and 0 <= py < CANVAS

    if xinc:
        while y1 != end_y and (inside(lt, y1) or inside(rb, y1)):
            steps += 1
            e2 = err
            if e2 > -dx:
                err -= dy
                x1 += sx
                lt, rb = lt + sx, rb + sx
            if e2 < dy:
                err += dx
                y1 += sy
    else:
        while x1 != end_x and (inside(x1, lt) or inside(x1, rb)):
            steps += 1
            e2 = err
            if e2 > -dx:
                err -= dy
                x1 += sx
            if e2 < dy:
                err += dx
                y1 += sy
                lt, rb = lt + sy, rb + sy
    return x, y, steps


def _unique_rows(rows):
    """``np.unique(rows, axis=0, return_index=True, return_inverse=True)[1:]`` through one byte key per row."""
    rows = np.ascontiguousarray(rows)
    keys = rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).reshape(-1)
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    return first, inverse.reshape(-1)


def _pow_half_array(values):
    """``float(n) ** 0.5`` per element, through C ``pow`` as Python computes it."""
    unique, inverse = np.unique(values, return_inverse=True)
    return np.array([_pow_half(int(v)) for v in unique.tolist()], np.float64)[inverse.reshape(-1)]


class _Group:
    """Primitive records of frames that share a template; every frame uses the same slots.

    Builders take arrays of shape ``[N]`` or ``[N, E]`` and return records ``[..., RECORD]``.
    """

    def __init__(self, count, capacity=256):
        self.count = count
        self.buffer = np.zeros((count, capacity, RECORD), np.int64)
        self.used = 0
        self.static = False

    def _take(self, slots):
        """Return the next ``slots`` record columns ``[N, slots, RECORD]``."""
        if self.used + slots > self.buffer.shape[1]:
            grown = np.zeros((self.count, 2 * (self.used + slots), RECORD), np.int64)
            grown[:, : self.used] = self.buffer[:, : self.used]
            self.buffer = grown
        view = self.buffer[:, self.used : self.used + slots]
        self.used += slots
        return view

    def _record(self, kind, color, enabled, shape, rec=None):
        rec = np.zeros((*shape, RECORD), np.int64) if rec is None else rec
        rec[..., KIND] = kind if enabled is None else np.where(enabled, kind, 0)
        rec[..., COLOR] = color if np.ndim(color) == len(shape) else np.reshape(color, (-1,) + (1,) * (len(shape) - 1))
        rec[..., STATIC] = self.static
        return rec

    def _poly(self, color, xs, ys, enabled=None, rec=None):
        """``draw_fillpoly`` with integer points ``[..., n]``."""
        rec = self._record(POLY, color, enabled, xs.shape[:-1], rec)
        n = xs.shape[-1]
        rec[..., BX0], rec[..., BX1] = xs.min(-1), xs.max(-1)
        rec[..., BY0], rec[..., BY1] = ys.min(-1), ys.max(-1)
        rec[..., PN] = n
        rec[..., PX : PX + n] = xs
        rec[..., PY : PY + n] = ys
        return rec

    def _circle(self, color, cx, cy, radius, enabled=None, rec=None):
        # ``pygame.draw.circle`` draws nothing for a radius below one.
        on = radius >= 1 if enabled is None else enabled & (radius >= 1)
        rec = self._record(CIRCLE, color, on, cx.shape, rec)
        rec[..., BX0], rec[..., BX1] = cx - radius, cx + radius
        rec[..., BY0], rec[..., BY1] = cy - radius, cy + radius
        rec[..., CX], rec[..., CY], rec[..., CR] = cx, cy, radius
        return rec

    def _line(self, color, x1, y1, x2, y2, rec=None):
        """``draw_line_width`` with width 4, including pygame's restart of clipped lines."""
        rec = self._record(LINE, color, None, x1.shape, rec)
        dx, dy = np.abs(x2 - x1), np.abs(y2 - y1)
        xinc = dx <= dy
        rec[..., LN] = np.maximum(dx, dy)
        sx, sy = x1.copy(), y1.copy()
        outside = (np.minimum(x1, x2) < 0) | (np.maximum(x1, x2) >= CANVAS)
        outside |= (np.minimum(y1, y2) < 0) | (np.maximum(y1, y2) >= CANVAS)
        for i in zip(*np.nonzero(outside)):
            run = clipped_line(int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]))
            if run is None:
                rec[i + (KIND,)] = 0
            else:
                sx[i], sy[i], rec[i + (LN,)] = run
        rec[..., BX0] = np.minimum(x1, x2) - np.where(xinc, 1, 0)
        rec[..., BX1] = np.maximum(x1, x2) + np.where(xinc, 2, 0)
        rec[..., BY0] = np.minimum(y1, y2) - np.where(xinc, 0, 1)
        rec[..., BY1] = np.maximum(y1, y2) + np.where(xinc, 0, 2)
        # A clipped run restarts at the clipped point and may pass the clipped end; each step moves
        # the minor axis at most one pixel.
        steps = rec[..., LN]
        rec[..., BX0] = np.where(outside, sx - steps - 1, rec[..., BX0])
        rec[..., BX1] = np.where(outside, sx + steps + 2, rec[..., BX1])
        rec[..., BY0] = np.where(outside, sy - steps - 1, rec[..., BY0])
        rec[..., BY1] = np.where(outside, sy + steps + 2, rec[..., BY1])
        rec[..., LX], rec[..., LY] = sx, sy
        rec[..., LSX] = np.where(x1 < x2, 1, -1)
        rec[..., LSY] = np.where(y1 < y2, 1, -1)
        rec[..., LDX], rec[..., LDY] = dx, dy
        # ``(dx > dy ? dx : -dy) / 2`` with C truncation.
        rec[..., LERR] = np.where(dx > dy, dx // 2, -(dy // 2))
        rec[..., LXINC] = xinc
        return rec

    def poly(self, color, xs, ys, enabled=None):
        self._poly(color, xs, ys, enabled, self._take(1)[:, 0])

    def circle(self, color, cx, cy, radius, enabled=None):
        self._circle(color, cx, cy, radius, enabled, self._take(1)[:, 0])

    def fat_segments(self, color, ax, ay, bx, by):
        """``DrawOptions.draw_fat_segment`` with radius 2 for float endpoints ``[N, E]``, edge after edge."""
        p1x, p1y, p2x, p2y = (np.rint(v).astype(np.int64) for v in (ax, ay, bx, by))
        # Each edge takes four consecutive slots: line, quad, then the two end circles.
        slots = self._take(4 * p1x.shape[1])
        self._line(color, p1x, p1y, p2x, p2y, slots[:, 0::4])
        o0, o1 = np.abs(p2y - p1y), np.abs(p2x - p1x)
        drawn = (o0 != 0) | (o1 != 0)
        squares = o0 * o0 + o1 * o1
        scale = 2.0 / np.where(drawn, _pow_half_array(squares).reshape(squares.shape), 1.0)
        q0, q1 = np.rint(o0 * scale).astype(np.int64), np.rint(o1 * scale).astype(np.int64)
        xs = np.stack([p1x - q0, p1x + q0, p2x + q0, p2x - q0], -1)
        ys = np.stack([p1y - q1, p1y + q1, p2y + q1, p2y - q1], -1)
        self._poly(color, xs, ys, drawn, slots[:, 1::4])
        two = np.full(p1x.shape, 2, np.int64)
        self._circle(color, p1x, p1y, two, drawn, slots[:, 2::4])
        self._circle(color, p2x, p2y, two, drawn, slots[:, 3::4])

    def records(self):
        return self.buffer[:, : self.used]


def _colors(fields, name, lanes):
    return fields[f"owned/variation/{name}/color/current/value"][lanes].astype(np.int64)


def primitives(fields):
    """Return ``(records [N, P, RECORD], counts [N], fallback [N])`` for the render fields of ``N`` rows.

    ``fallback`` marks rows that this renderer does not cover; draw those with pygame.
    """
    count = len(fields["recipe/config"])
    config = fields["recipe/config"]
    fallback = config[:, 2] != 0  # action markers
    parts = []
    # Rows with equal anchor shape and scale fields (values and type tags) share a template.
    raw = np.concatenate(
        [
            np.ascontiguousarray(fields[f"anchor/variation/{body}/{leaf}/current/{part}"])
            .reshape(count, -1)
            .view(np.uint8)
            for body in ("agent", "block")
            for leaf in ("shape", "scale")
            for part in ("value", "type")
        ],
        1,
    )
    first, inverse = _unique_rows(raw)
    for index, lane0 in enumerate(first.tolist()):
        key = template_key(fields, lane0)
        lanes = np.flatnonzero(inverse == index)
        try:
            template = _template(key)
        except NotImplementedError:
            fallback[lanes] = True
            continue
        n = len(lanes)
        group = _Group(n)
        goal_pos = fields["owned/variation/goal/position/current/value"][lanes].astype(np.float64)
        goal_angle = fields["owned/variation/goal/angle/current/value"][lanes].astype(np.float64).reshape(n)
        render_goal = fields["owned/variation/rendering/render_goal/current/value"][lanes].reshape(n) != 0
        render_goal &= config[lanes, 3] != 0
        goal = _transform(goal_pos[:, 0], goal_pos[:, 1], goal_angle, np.zeros(n), np.zeros(n))
        goal_color = pack(_colors(fields, "goal", lanes))
        group.static = True
        for local in template.goal:
            wx, wy = _apply(goal, local[:, 0], local[:, 1])
            xs, ys = np.rint(wx).astype(np.int64), np.rint(wy).astype(np.int64)
            group.poly(goal_color, np.concatenate([xs, xs[:, :1]], 1), np.concatenate([ys, ys[:, :1]], 1), render_goal)
        bodies = {}
        for body in ("agent", "block"):
            pos = fields[f"current/body/{body}/position"][lanes]
            cog = fields[f"current/body/{body}/center_of_gravity"][lanes]
            angle = fields[f"current/body/{body}/angle"][lanes].reshape(n)
            fill = _colors(fields, body, lanes)
            bodies[body] = (
                _transform(pos[:, 0], pos[:, 1], angle, cog[:, 0], cog[:, 1]),
                pack(fill),
                pack(_light(fill)),
            )
        wall = np.full(n, pack(WALL_COLOR))
        walls = []

        def flush(pending):
            # Consecutive walls in one call; ``fat_segments`` keeps their drawing order.
            if pending:
                group.static = True
                ends = np.array([[*item[1], *item[2]] for item in pending], np.float64)
                group.fat_segments(wall, *(np.broadcast_to(ends[:, k], (n, len(pending))) for k in range(4)))
                group.static = False
                pending.clear()

        for item in template.items:
            if item[0] == "poly":
                transform, fill, light = bodies[item[1]]
                local = item[2]
                wx, wy = _apply(transform, local[:, 0], local[:, 1])
                xs, ys = np.rint(wx).astype(np.int64), np.rint(wy).astype(np.int64)
                group.poly(light, np.concatenate([xs, xs[:, :1]], 1), np.concatenate([ys, ys[:, :1]], 1))
                follow = np.roll(np.arange(len(local)), -1)
                group.fat_segments(fill, wx, wy, wx[:, follow], wy[:, follow])
            elif item[0] == "circle":
                transform, fill, light = bodies[item[1]]
                (ox, oy), radius = item[2], item[3]
                wx, wy = _apply(transform, np.array([ox]), np.array([oy]))
                cx, cy = np.rint(wx[:, 0]).astype(np.int64), np.rint(wy[:, 0]).astype(np.int64)
                group.circle(fill, cx, cy, np.full(n, round(radius), np.int64))
                group.circle(light, cx, cy, np.full(n, round(radius - 4), np.int64))
            else:
                walls.append(item)
                continue
            flush(walls)
        flush(walls)
        parts.append((lanes, group.records()))
    width = max([1] + [rec.shape[1] for _, rec in parts])
    records = np.zeros((count, width, RECORD), np.int32)
    counts = np.zeros(count, np.int64)
    for lanes, rec in parts:
        records[lanes, : rec.shape[1]] = rec
        counts[lanes] = rec.shape[1]
    return records, counts, fallback


def aaline_writes(fx, fy, tx, ty):
    """Port ``draw_aaline`` for segments ``[S]``: return ordered ``(x, y, brightness, valid)`` of shape ``[S, K]``.

    All arithmetic is float32 and every ``(int)`` cast truncates, as in pygame's C code.
    """
    f32, one = np.float32, np.float32(1)
    fx, fy, tx, ty = (np.asarray(v, f32).copy() for v in (fx, fy, tx, ty))
    count = len(fx)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        dx, dy = (tx - fx).astype(f32), (ty - fy).astype(f32)
        point = (np.abs(dx.astype(np.float64)) < 0.0001) & (np.abs(dy.astype(np.float64)) < 0.0001)
        # A segment of length zero is one full-brightness pixel at the rounded start point.
        single = (
            np.floor(fx.astype(np.float64) + 0.5).astype(np.int64),
            np.floor(fy.astype(np.float64) + 0.5).astype(np.int64),
        )
        cl = ct = f32(-1.0)
        cr = cb = f32(f32(-1.0) + f32(CANVAS) + one)
        steep = np.abs(dx.astype(np.float64)) < np.abs(dy.astype(np.float64))
        fx, fy = np.where(steep, fy, fx), np.where(steep, fx, fy)
        tx, ty = np.where(steep, ty, tx), np.where(steep, tx, ty)
        dx, dy = np.where(steep, dy, dx), np.where(steep, dx, dy)
        back = dx < 0
        fx, tx = np.where(back, tx, fx), np.where(back, fx, tx)
        fy, ty = np.where(back, ty, fy), np.where(back, fy, ty)
        dx, dy = np.where(back, -dx, dx), np.where(back, -dy, dy)
        live = ~point & ~((tx <= cl) | (fx >= cr))
        g = (dy / dx).astype(f32)
        left = fx < cl
        fy = np.where(left, (fy + (g * (cl - fx).astype(f32)).astype(f32)).astype(f32), fy)
        fx = np.where(left, cl, fx)
        right = tx > cr
        ty = np.where(right, (ty + (g * (cr - tx).astype(f32)).astype(f32)).astype(f32), ty)
        tx = np.where(right, cr, tx)
        down = g > 0
        live &= np.where(down, ~((ty <= ct) | (fy >= cb)), ~((fy <= ct) | (ty >= cb)))
        a = down & (fy < ct)
        fx = np.where(a, (fx + ((ct - fy).astype(f32) / g).astype(f32)).astype(f32), fx)
        fy = np.where(a, ct, fy)
        a = down & (ty > cb)
        tx = np.where(a, (tx + ((cb - ty).astype(f32) / g).astype(f32)).astype(f32), tx)
        ty = np.where(a, cb, ty)
        a = ~down & (ty < ct)
        tx = np.where(a, (tx + ((ct - ty).astype(f32) / g).astype(f32)).astype(f32), tx)
        ty = np.where(a, ct, ty)
        a = ~down & (fy > cb)
        fx = np.where(a, (fx + ((cb - fy).astype(f32) / g).astype(f32)).astype(f32), fx)
        fy = np.where(a, cb, fy)
        fy, ty = (fy + one).astype(f32), (ty + one).astype(f32)
        xs = np.trunc(fx).astype(np.int64)
        ye = (fy + (g * (xs.astype(f32) - fx).astype(f32)).astype(f32)).astype(f32)
        iy = ye.copy()
        writes = []

        def put(x, y, brightness, valid):
            # ``steep`` swaps the axes back.
            px, py = np.where(steep, y, x), np.where(steep, x, y)
            inside = (px >= 0) & (px < CANVAS) & (py >= 0) & (py < CANVAS)
            writes.append((px, py, brightness.astype(f32), valid & inside))

        first = live & (tx > (cl + one).astype(f32))
        gap = (f32(1) * (1 + xs).astype(f32) - fx).astype(f32)
        yi = np.trunc(ye).astype(np.int64)
        br = (ye - yi.astype(f32)).astype(f32)
        put(xs, yi, (br * gap).astype(f32), first & (yi.astype(f32) < ye))
        put(xs, yi - 1, ((one - br).astype(f32) * gap).astype(f32), first)
        iy = np.where(first, (iy + g).astype(f32), iy)
        xs = np.where(first, xs + 1, xs)
        xe = np.ceil(tx.astype(np.float64)).astype(np.int64)
        second = live & (fx < (cr - one).astype(f32))
        ye2 = (ty + (g * (xe.astype(f32) - tx).astype(f32)).astype(f32)).astype(f32)
        gap2 = ((1 - xe).astype(f32) + tx).astype(f32)
        yi2 = np.trunc(ye2).astype(np.int64)
        br2 = (ye2 - yi2.astype(f32)).astype(f32)
        put(xe, yi2, (br2 * gap2).astype(f32), second & (yi2.astype(f32) < ye2))
        put(xe, yi2 - 1, ((one - br2).astype(f32) * gap2).astype(f32), second)
        steps = int(np.max(np.where(live, xe - xs, 0), initial=0))
        for k in range(steps):
            x = xs + k
            on = live & (x < xe)
            y = np.trunc(iy).astype(np.int64)
            br = ((one - iy).astype(f32) + y.astype(f32)).astype(f32)
            put(x, y - 1, br, on)
            put(x, y, (one - br).astype(f32), on & (y.astype(f32) < iy))
            iy = np.where(on, (iy + g).astype(f32), iy)
    if count == 0:
        empty = np.zeros((0, 0), np.int64)
        return empty, empty, empty.astype(f32), empty.astype(np.bool_)
    px = np.stack([w[0] for w in writes], 1)
    py = np.stack([w[1] for w in writes], 1)
    pb = np.stack([w[2] for w in writes], 1)
    pv = np.stack([w[3] for w in writes], 1)
    inside = (single[0] >= 0) & (single[0] < CANVAS) & (single[1] >= 0) & (single[1] < CANVAS)
    px = np.concatenate([single[0][:, None], px], 1)
    py = np.concatenate([single[1][:, None], py], 1)
    pb = np.concatenate([np.ones((count, 1), f32), pb], 1)
    pv = np.concatenate([(point & inside)[:, None], pv], 1)
    return px, py, pb, pv


def contact_writes(fields):
    """Return ``(x, y, brightness, colour, count)`` blend writes per frame ``[N, W]`` in drawing order."""
    segments = fields["current/render_segments"]
    counts = fields["current/render_segment_count"].astype(np.int64).reshape(-1)
    n = len(counts)
    frame, index = np.nonzero(np.arange(segments.shape[1])[None] < counts[:, None])
    seg = segments[frame, index]
    # ``to_pygame`` and ``SpaceDebugColor.as_int`` round half to even.
    ends = np.rint(seg[:, :4]).astype(np.int64)
    color = pack(np.rint(seg[:, 4:7]).astype(np.int64))
    px, py, pb, pv = aaline_writes(ends[:, 0], ends[:, 1], ends[:, 2], ends[:, 3])
    per = pv.sum(1) if len(seg) else np.zeros(0, np.int64)
    totals = np.bincount(frame, per, minlength=n).astype(np.int64)
    width = max(1, int(totals.max(initial=0)))
    out = np.zeros((4, n, width), np.float64)
    if len(seg):
        order = np.cumsum(pv, 1) - 1
        start = np.zeros(len(seg), np.int64)
        # Segments of one frame follow each other in index order.
        start[1:] = np.cumsum(per)[:-1]
        first_of_frame = np.zeros(n, np.int64)
        first_of_frame[1:] = np.cumsum(totals)[:-1]
        slot = start[:, None] + order - first_of_frame[frame][:, None]
        rows = np.broadcast_to(frame[:, None], pv.shape)[pv]
        cols = slot[pv]
        out[0, rows, cols] = px[pv]
        out[1, rows, cols] = py[pv]
        out[2, rows, cols] = pb[pv]
        out[3, rows, cols] = np.broadcast_to(color[:, None], pv.shape)[pv]
    return out[0].astype(np.int64), out[1].astype(np.int64), out[2].astype(np.float32), out[3].astype(np.int64), totals


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


def _shade(records, y, x, color, table, max_radius):
    """Paint ``records [n, P, RECORD]`` over integer grids ``y [1, R, 1]`` and ``x [1, 1, C]`` into ``color``."""
    for p in range(records.shape[1]):
        rec = records[:, p]
        kind = rec[:, KIND].view(-1, 1, 1)
        if not bool((kind != 0).any()):
            continue
        r = {
            name: rec[:, index].view(-1, 1, 1)
            for name, index in (("bx0", BX0), ("by0", BY0), ("bx1", BX1), ("by1", BY1))
        }
        box = (x >= r["bx0"]) & (x <= r["bx1"]) & (y >= r["by0"]) & (y <= r["by1"])
        cover = torch.zeros_like(box)
        if bool((kind == POLY).any()):
            cover |= (kind == POLY) & _poly(rec, y, x)
        if bool((kind == CIRCLE).any()):
            cover |= (kind == CIRCLE) & _circle(rec, y, x, table, max_radius)
        if bool((kind == LINE).any()):
            cover |= (kind == LINE) & _line(rec, y, x)
        color = torch.where(box & cover, rec[:, COLOR].view(-1, 1, 1).to(color.dtype), color)
    return color


def _poly(rec, y, x):
    n = torch.where(rec[:, KIND] == POLY, rec[:, PN], 0).clamp(0, 5).view(-1, 1, 1)
    xs, ys = rec[:, PX : PX + 5].view(-1, 1, 1, 5), rec[:, PY : PY + 5].view(-1, 1, 1, 5)
    use = torch.arange(5, device=rec.device).view(1, 1, 1, 5) < n.unsqueeze(-1)
    big = torch.iinfo(torch.int32).max
    miny = torch.where(use, ys, big).amin(-1)
    maxy = torch.where(use, ys, -big).amax(-1)
    minx = torch.where(use, xs, big).amin(-1)
    maxx = torch.where(use, xs, -big).amax(-1)
    prev = (torch.arange(5, device=rec.device).view(1, 1, 1, 5) - 1) % n.unsqueeze(-1).clamp(min=1)
    xa, ya = torch.gather(xs, 3, prev), torch.gather(ys, 3, prev)
    up = ya < ys
    y1, y2 = torch.where(up, ya, ys), torch.where(up, ys, ya)
    x1, x2 = torch.where(up, xa, xs), torch.where(up, xs, xa)
    edge = use & (ya != ys)
    yy, xx = y.unsqueeze(-1), x.unsqueeze(-1)
    on = edge & (((yy >= y1) & (yy < y2)) | ((yy == maxy.unsqueeze(-1)) & (y2 == maxy.unsqueeze(-1))))
    den = torch.where(edge, y2 - y1, torch.ones_like(y2))
    # ``(int)((y - y1) * (x2 - x1) / (y2 - y1) + x1)`` in double truncates the exact quotient.
    hits = torch.where(on, torch.div((yy - y1) * (x2 - x1) + x1 * den, den, rounding_mode="trunc"), big)
    hits, _ = torch.sort(hits, -1)
    count = (hits != big).sum(-1)
    inside = (count >= 2) & (x >= hits[..., 0]) & (x <= hits[..., 1])
    inside |= (count >= 4) & (x >= hits[..., 2]) & (x <= hits[..., 3])
    # Horizontal edges strictly between the top and bottom rows get their own span.
    level = use & (ya == ys) & (miny.unsqueeze(-1) < ys) & (ys < maxy.unsqueeze(-1))
    special = level & (yy == ys) & (xx >= torch.minimum(xa, xs)) & (xx <= torch.maximum(xa, xs))
    inside = (inside & (y >= miny) & (y <= maxy)) | special.any(-1)
    return torch.where(miny == maxy, (y == miny) & (x >= minx) & (x <= maxx), inside)


def _circle(rec, y, x, table, max_radius):
    cx, cy, radius = (rec[:, i].view(-1, 1, 1) for i in (CX, CY, CR))
    dy = y - cy
    valid = (dy >= -max_radius - 1) & (dy <= max_radius)
    row = (dy + max_radius + 1).clamp(0, 2 * max_radius + 1)
    span = table[radius.clamp(0, max_radius), row]
    return valid & (x - cx >= span[..., 0]) & (x - cx <= span[..., 1])


def _line(rec, y, x):
    lx, ly, sx, sy, dx, dy, err, steps, xinc = (rec[:, i].view(-1, 1, 1) for i in range(LX, LXINC + 1))
    k = (y - ly) * sy
    nk = torch.div(k * dx + err + dy - 1, dy.clamp(min=1), rounding_mode="floor")
    xk = lx + sx * torch.where(dy > 0, nk, 0)
    rows = (k >= 0) & (k <= steps) & (x >= xk - 1) & (x <= xk + 2)
    k = (x - lx) * sx
    mk = torch.div(k * dy - err + dx - 1, dx.clamp(min=1), rounding_mode="floor")
    yk = ly + sy * mk
    cols = (k >= 0) & (k <= steps) & (y >= yk - 1) & (y <= yk + 2)
    return torch.where(xinc != 0, rows, cols)


def draw(records, background, writes, size, device, chunk=64):
    """Return ``[N, size, size, 3]`` uint8 frames for host ``records`` and ``writes`` (see ``contact_writes``)."""
    s, a0, a1 = resize_coefficients(CANVAS, size)
    taps = np.unique(np.concatenate([s, s + 1]))
    position = np.full(CANVAS, -1, np.int64)
    position[taps] = np.arange(len(taps))
    grid = torch.as_tensor(taps, device=device, dtype=torch.int32)
    y, x = grid.view(1, -1, 1), grid.view(1, 1, -1)
    max_radius = max(2, int(records[..., CR].max(initial=0)) if len(records) else 2)
    table = torch.as_tensor(circle_table(max_radius), device=device, dtype=torch.int32)
    i0 = torch.as_tensor(position[s], device=device)
    i1 = torch.as_tensor(position[s + 1], device=device)
    w0 = torch.as_tensor(a0, device=device, dtype=torch.int32)
    w1 = torch.as_tensor(a1, device=device, dtype=torch.int32)
    wx, wy, wb, wc, wn = writes
    frames = []
    for start in range(0, len(records), chunk):
        stop = min(start + chunk, len(records))
        rec = torch.as_tensor(records[start:stop], device=device, dtype=torch.int32)
        color = torch.as_tensor(background[start:stop], device=device, dtype=torch.int32).view(-1, 1, 1)
        color = color.expand(-1, len(taps), len(taps)).contiguous()
        color = _shade(rec, y, x, color, table, max_radius)
        color = _blend(
            color, wx[start:stop], wy[start:stop], wb[start:stop], wc[start:stop], wn[start:stop], position, device
        )
        frames.append(_resize(color, i0, i1, w0, w1))
    return torch.cat(frames) if frames else torch.zeros((0, size, size, 3), dtype=torch.uint8, device=device)


def _blend(color, wx, wy, wb, wc, wn, position, device):
    """Apply ``get_antialiased_color`` blends in order; writes to pixels the resize does not read are dropped."""
    if not int(wn.max(initial=0)):
        return color
    n = color.shape[0]
    rows = torch.as_tensor(position[wy], device=device)
    cols = torch.as_tensor(position[wx], device=device)
    bright = torch.as_tensor(wb, device=device)
    paint = torch.as_tensor(wc, device=device)
    frame = torch.arange(n, device=device)
    for w in range(int(wn.max())):
        on = (w < torch.as_tensor(wn, device=device)) & (rows[:, w] >= 0) & (cols[:, w] >= 0)
        if not bool(on.any()):
            continue
        f, r, c = frame[on], rows[on, w], cols[on, w]
        old = color[f, r, c]
        b = bright[on, w]
        new = paint[on, w]
        out = torch.zeros_like(old)
        for shift in (16, 8, 0):
            fg = ((new >> shift) & 255).to(torch.float32)
            bg = ((old >> shift) & 255).to(torch.float32)
            value = (b * fg + (1 - b) * bg).to(torch.int32).clamp(0, 255)
            out |= value << shift
        color[f, r, c] = out
    return color


def _resize(color, i0, i1, w0, w1):
    """OpenCV 5.0 uint8 INTER_LINEAR from the sampled canvas: 11-bit rows, then the SIMD vertical pass."""
    channels = []
    for shift in (16, 8, 0):
        c = (color >> shift) & 255
        h0 = c[:, i0][:, :, i0] * w0 + c[:, i0][:, :, i1] * w1
        h1 = c[:, i1][:, :, i0] * w0 + c[:, i1][:, :, i1] * w1
        v = (((h0 >> 4) * w0.view(-1, 1)) >> 16) + (((h1 >> 4) * w1.view(-1, 1)) >> 16)
        channels.append(((v + 2) >> 2).clamp(0, 255).to(torch.uint8))
    return torch.stack(channels, -1)


try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - CPU-only installs
    HAVE_TRITON = False

if HAVE_TRITON:
    # Jitted code reads module constants only as ``constexpr``.
    _POLY, _CIRCLE = tl.constexpr(POLY), tl.constexpr(CIRCLE)
    _KIND, _COLOR, _BX0, _BY0, _BX1, _BY1 = (tl.constexpr(v) for v in (KIND, COLOR, BX0, BY0, BX1, BY1))
    _PN, _PX, _PY = tl.constexpr(PN), tl.constexpr(PX), tl.constexpr(PY)
    _CX, _CY, _CR = tl.constexpr(CX), tl.constexpr(CY), tl.constexpr(CR)
    _LX, _LY, _LSX, _LSY = tl.constexpr(LX), tl.constexpr(LY), tl.constexpr(LSX), tl.constexpr(LSY)
    _LDX, _LDY, _LERR, _LN, _LXINC = (tl.constexpr(v) for v in (LDX, LDY, LERR, LN, LXINC))

    @triton.jit
    def _pick(i, v0, v1, v2, v3, v4):
        return tl.where(i == 0, v0, tl.where(i == 1, v1, tl.where(i == 2, v2, tl.where(i == 3, v3, v4))))

    @triton.jit
    def _sort2(a, b):
        return tl.minimum(a, b), tl.maximum(a, b)

    @triton.jit
    def _edge(i, n, x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, miny, maxy, yv, xv, BIG: tl.constexpr):
        """One ``draw_fillpoly`` edge: its intersection per row, and its horizontal-edge span."""
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
        level = (i < n) & (ya == yb) & (miny < yb) & (yb < maxy)
        lo, hi = tl.minimum(xa, xb), tl.maximum(xa, xb)
        span = level & (yv[:, None] == yb) & (xv[None, :] >= lo) & (xv[None, :] <= hi)
        return tl.where(on, q, BIG), span

    @triton.jit
    def _poly_cover(yv, xv, base, BIG: tl.constexpr):
        n = tl.load(base + _PN)
        x0, x1, x2 = tl.load(base + _PX), tl.load(base + _PX + 1), tl.load(base + _PX + 2)
        x3, x4 = tl.load(base + _PX + 3), tl.load(base + _PX + 4)
        y0, y1, y2 = tl.load(base + _PY), tl.load(base + _PY + 1), tl.load(base + _PY + 2)
        y3, y4 = tl.load(base + _PY + 3), tl.load(base + _PY + 4)
        y1u, y2u, y3u, y4u = (
            tl.where(n > 1, y1, y0),
            tl.where(n > 2, y2, y0),
            tl.where(n > 3, y3, y0),
            tl.where(n > 4, y4, y0),
        )
        x1u, x2u, x3u, x4u = (
            tl.where(n > 1, x1, x0),
            tl.where(n > 2, x2, x0),
            tl.where(n > 3, x3, x0),
            tl.where(n > 4, x4, x0),
        )
        miny = tl.minimum(tl.minimum(tl.minimum(y0, y1u), tl.minimum(y2u, y3u)), y4u)
        maxy = tl.maximum(tl.maximum(tl.maximum(y0, y1u), tl.maximum(y2u, y3u)), y4u)
        minx = tl.minimum(tl.minimum(tl.minimum(x0, x1u), tl.minimum(x2u, x3u)), x4u)
        maxx = tl.maximum(tl.maximum(tl.maximum(x0, x1u), tl.maximum(x2u, x3u)), x4u)
        h0, s0 = _edge(0, n, x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, miny, maxy, yv, xv, BIG)
        h1, s1 = _edge(1, n, x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, miny, maxy, yv, xv, BIG)
        h2, s2 = _edge(2, n, x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, miny, maxy, yv, xv, BIG)
        h3, s3 = _edge(3, n, x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, miny, maxy, yv, xv, BIG)
        h4, s4 = _edge(4, n, x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, miny, maxy, yv, xv, BIG)
        h0, h1 = _sort2(h0, h1)
        h3, h4 = _sort2(h3, h4)
        h2, h4 = _sort2(h2, h4)
        h2, h3 = _sort2(h2, h3)
        h0, h3 = _sort2(h0, h3)
        h0, h2 = _sort2(h0, h2)
        h1, h4 = _sort2(h1, h4)
        h1, h3 = _sort2(h1, h3)
        h1, h2 = _sort2(h1, h2)
        count = (h0 != BIG).to(tl.int32) + (h1 != BIG).to(tl.int32) + (h2 != BIG).to(tl.int32)
        count += (h3 != BIG).to(tl.int32) + (h4 != BIG).to(tl.int32)
        x = xv[None, :]
        inside = ((count >= 2)[:, None] & (x >= h0[:, None]) & (x <= h1[:, None])) | (
            (count >= 4)[:, None] & (x >= h2[:, None]) & (x <= h3[:, None])
        )
        rows = (yv >= miny) & (yv <= maxy)
        inside = (inside & rows[:, None]) | s0 | s1 | s2 | s3 | s4
        flat = (yv[:, None] == miny) & (x >= minx) & (x <= maxx)
        return tl.where(miny == maxy, flat, inside)

    @triton.jit
    def _circle_cover(yv, xv, base, table, max_radius):
        cx, cy, r = tl.load(base + _CX), tl.load(base + _CY), tl.load(base + _CR)
        dy = yv - cy
        valid = (dy >= -max_radius - 1) & (dy <= max_radius)
        row = tl.minimum(tl.maximum(dy + max_radius + 1, 0), 2 * max_radius + 1)
        index = (tl.minimum(r, max_radius) * (2 * max_radius + 2) + row) * 2
        lo, hi = tl.load(table + index), tl.load(table + index + 1)
        dx = xv[None, :] - cx
        return valid[:, None] & (dx >= lo[:, None]) & (dx <= hi[:, None])

    @triton.jit
    def _line_cover(yv, xv, base):
        lx, ly = tl.load(base + _LX), tl.load(base + _LY)
        sx, sy = tl.load(base + _LSX), tl.load(base + _LSY)
        dx, dy = tl.load(base + _LDX), tl.load(base + _LDY)
        err, steps, xinc = tl.load(base + _LERR), tl.load(base + _LN), tl.load(base + _LXINC)
        # Spans of rows when the line is steep, spans of columns otherwise.
        ky = (yv - ly) * sy
        nk = tl.where(dy > 0, (tl.maximum(ky, 0) * dx + err + dy - 1) // tl.maximum(dy, 1), 0)
        xk = lx + sx * nk
        rows = (
            ((ky >= 0) & (ky <= steps))[:, None]
            & (xv[None, :] >= (xk - 1)[:, None])
            & (xv[None, :] <= (xk + 2)[:, None])
        )
        kx = (xv - lx) * sx
        mk = (tl.maximum(kx, 0) * dy - err + dx - 1) // tl.maximum(dx, 1)
        yk = ly + sy * mk
        cols = (
            ((kx >= 0) & (kx <= steps))[None, :]
            & (yv[:, None] >= (yk - 1)[None, :])
            & (yv[:, None] <= (yk + 2)[None, :])
        )
        return tl.where(xinc != 0, rows, cols)

    @triton.jit
    def _paint(yv, xv, c, base, color, kind, table, max_radius, BIG: tl.constexpr):
        if kind == _POLY:
            cover = _poly_cover(yv, xv, base, BIG)
        elif kind == _CIRCLE:
            cover = _circle_cover(yv, xv, base, table, max_radius)
        else:
            cover = _line_cover(yv, xv, base)
        return tl.where(cover, color, c)

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
        P,
        W,
        size,
        max_radius,
        RECORD_: tl.constexpr,
        TILE: tl.constexpr,
        BIG: tl.constexpr,
    ):
        tile = tl.program_id(0)
        f = tl.program_id(1).to(tl.int64)
        tiles = tl.cdiv(size, TILE)
        o = tl.arange(0, TILE)
        oy = (tile // tiles) * TILE + o
        ox = (tile % tiles) * TILE + o
        my, mx = oy < size, ox < size
        y0 = tl.load(taps + oy, mask=my, other=0)
        x0 = tl.load(taps + ox, mask=mx, other=0)
        y1, x1 = y0 + 1, x0 + 1
        ylo, yhi = tl.min(tl.where(my, y0, BIG)), tl.max(tl.where(my, y1, -BIG))
        xlo, xhi = tl.min(tl.where(mx, x0, BIG)), tl.max(tl.where(mx, x1, -BIG))
        pixel = ((f * size + oy[:, None]) * size + ox[None, :]) * 3
        mask = my[:, None] & mx[None, :]
        si = tl.load(static_index + f)
        dx0, dy0 = tl.load(dynamic + f * 4), tl.load(dynamic + f * 4 + 1)
        dx1, dy1 = tl.load(dynamic + f * 4 + 2), tl.load(dynamic + f * 4 + 3)
        touched = (dx0 <= xhi) & (dx1 >= xlo) & (dy0 <= yhi) & (dy1 >= ylo)
        if (si >= 0) & (touched == 0):
            # Only the background, goal and walls reach this tile: copy it from the frame they draw alone.
            source = ((si.to(tl.int64) * size + oy[:, None]) * size + ox[None, :]) * 3
            for k in tl.static_range(3):
                tl.store(out + pixel + k, tl.load(static_out + source + k, mask=mask, other=0), mask=mask)
        else:
            bg = tl.load(background + f)
            c00 = tl.zeros([TILE, TILE], tl.int32) + bg
            c01 = c00
            c10 = c00
            c11 = c00
            n = tl.load(counts + f)
            for p in range(0, n):
                base = records + (f * P + p) * RECORD_
                kind = tl.load(base + _KIND)
                bx0, by0 = tl.load(base + _BX0), tl.load(base + _BY0)
                bx1, by1 = tl.load(base + _BX1), tl.load(base + _BY1)
                if (kind != 0) & (bx0 <= xhi) & (bx1 >= xlo) & (by0 <= yhi) & (by1 >= ylo):
                    color = tl.load(base + _COLOR)
                    c00 = _paint(y0, x0, c00, base, color, kind, table, max_radius, BIG)
                    c01 = _paint(y0, x1, c01, base, color, kind, table, max_radius, BIG)
                    c10 = _paint(y1, x0, c10, base, color, kind, table, max_radius, BIG)
                    c11 = _paint(y1, x1, c11, base, color, kind, table, max_radius, BIG)
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
    max_radius = max(2, int(records[..., CR].max(initial=0)) if n else 2)

    def t(a, dtype=torch.int32):
        return _upload(a, dtype, device)

    wx, wy, wb, wc, wn = writes
    out = torch.empty((n, size, size, 3), dtype=torch.uint8, device=device)
    if n == 0:
        return out
    tiles = triton.cdiv(size, tile) ** 2
    _frames_kernel[(tiles, n)](
        t(records), t(counts), t(background), t(wx), t(wy), t(wb, torch.float32), t(wc), t(wn), t(s),
        t(a0), t(a1), t(circle_table(max_radius)), out,
        t(static_index), static_out, t(dynamic),
        records.shape[1], wx.shape[1], size, max_radius,
        RECORD_=RECORD, TILE=tile, BIG=2**30,
    )  # fmt: skip
    return out


def draw_cuda(records, counts, background, writes, size, device, tile=32):
    """Triton version of ``draw`` on CUDA.

    Tiles that no agent, block or contact primitive reaches copy the frame that the background,
    goal and walls draw alone, which frames with the same goal and colours share.
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
