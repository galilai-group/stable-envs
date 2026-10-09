"""PushT display lists computed from render rows with the source's draw arithmetic.

A :class:`Template` captures what a pymunk scene fixes for one body geometry: the shape parts
in debug-draw order, their local vertices and the walls. :func:`build` then computes every
frame's draw calls from the rows alone: Chipmunk's body transforms and polygon vertices, the
``round`` of ``to_pygame``, ``light_color``, the fat-segment arithmetic, the goal outline and
the contact segments.
"""

from __future__ import annotations

import math

import numpy as np
import pymunk

from stable_envs.envs.pusht.simulation.numeric import get_value


def _rint(a):
    return np.rint(a).astype(np.int64)


_NORM = {}


def _scale(o0, o1, radius):
    """``radius / (o0*o0 + o1*o1) ** 0.5`` with Python float ``pow``, memoized per integer pair."""
    out = np.zeros(o0.shape, np.float64)
    for i, key in enumerate((o0 * o0 + o1 * o1).ravel().tolist()):
        if key:
            v = _NORM.get(key)
            if v is None:
                v = _NORM[key] = key**0.5
            out.flat[i] = radius / v
    return out


def _light(color):
    return _rint(np.minimum(np.float32(1.2) * color.astype(np.float32), np.float32(255)))


def _cos_sin(angle):
    a = angle.tolist()
    return np.array([math.cos(v) for v in a]), np.array([math.sin(v) for v in a])


def _apply(c, s, tx, ty, v):
    """``cpTransformPoint`` of local points v (k, 2) for B transforms -> (B, k, 2)."""
    x = (c[:, None] * v[None, :, 0] + (-s)[:, None] * v[None, :, 1]) + tx[:, None]
    y = (s[:, None] * v[None, :, 0] + c[:, None] * v[None, :, 1]) + ty[:, None]
    return np.stack([x, y], -1)


def _body_transform(angle, pos, cog):
    """Transform after ``body.angle = angle`` then ``body.position = pos``."""
    c, s = _cos_sin(angle)
    tvx = c * cog[0] - s * cog[1]
    tvy = s * cog[0] + c * cog[1]
    return c, s, (tvx + pos[:, 0]) - tvx, (tvy + pos[:, 1]) - tvy


class Template:
    """Draw-relevant geometry of one scene."""

    def __init__(self, scene):
        shapes = [(name, shape) for name in ("block", "agent") for shape in getattr(scene, name).shapes]
        self.cog = {n: tuple(getattr(scene, n).center_of_gravity) for n in ("block", "agent")}
        self.parts = []
        for name, shape, radius in self._draw_order(scene, shapes):
            if isinstance(shape, pymunk.Poly):
                self.parts.append((name, "poly", np.array([tuple(v) for v in shape.get_vertices()], np.float64)))
            else:
                self.parts.append((name, "circle", (np.array(tuple(shape.offset), np.float64), float(radius))))
        self.goal_parts = []
        for shape in scene.block.shapes:
            if isinstance(shape, pymunk.Poly):
                self.goal_parts.append(("poly", np.array([tuple(v) for v in shape.get_vertices()], np.float64)))
            else:
                self.goal_parts.append(("circle", (np.array(tuple(shape.offset), np.float64), float(shape.radius))))
        self.walls = [
            (np.array(tuple(s.a)), np.array(tuple(s.b)), float(s.radius), tuple(int(v) for v in s.color)[:3])
            for s in scene.space.shapes
            if isinstance(s, pymunk.Segment)
        ]

    @staticmethod
    def _draw_order(scene, shapes):
        seen = []

        class Recorder(pymunk.SpaceDebugDrawOptions):
            def draw_polygon(self, verts, radius, outline, fill):
                seen.append(("poly", [tuple(v) for v in verts]))

            def draw_circle(self, pos, angle, radius, outline, fill):
                seen.append(("circle", (tuple(pos), radius)))

            def draw_fat_segment(self, a, b, radius, outline, fill):
                pass

            def draw_segment(self, a, b, color):
                pass

            def draw_dot(self, size, pos, color):
                pass

        opts = Recorder()
        opts.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
        scene.space.debug_draw(opts)
        order = []
        for kind, data in seen:
            for name, shape in shapes:
                body = shape.body
                if kind == "poly" and isinstance(shape, pymunk.Poly):
                    if [tuple(body.local_to_world(v)) for v in shape.get_vertices()] == data:
                        order.append((name, shape, None))
                        break
                elif kind == "circle" and isinstance(shape, pymunk.Circle):
                    if tuple(body.local_to_world(shape.offset)) == data[0]:
                        order.append((name, shape, data[1]))
                        break
            else:
                raise RuntimeError("drawn PushT shape does not match any body shape")
        return order


def build(template, rows):
    """Draw calls for frames sharing ``template``: a list of (kind, rgb (B, 3), geometry, enabled (B,))."""
    B = len(rows)

    def get(key):
        return np.stack([np.asarray(get_value(r, key), np.float64) for r in rows])

    on = np.ones(B, bool)
    ops = [("fill", get("owned/variation/background/color/current").astype(np.int64), None, on)]
    show_goal = get("owned/variation/rendering/render_goal/current").astype(bool) & np.array(
        [bool(r["recipe/config"][3]) for r in rows]
    )
    goal_rgb = get("owned/variation/goal/color/current").astype(np.int64)
    gpos = get("anchor/variation/goal/position/current")
    gc, gs = _cos_sin(get("anchor/variation/goal/angle/current"))
    for kind, data in template.goal_parts:
        if kind == "poly":
            pts = _rint(_apply(gc, gs, gpos[:, 0], gpos[:, 1], data))
            ops.append(("poly", goal_rgb, np.concatenate([pts, pts[:, :1]], 1), show_goal))
        else:
            centre = _rint(_apply(gc, gs, gpos[:, 0], gpos[:, 1], data[0][None])[:, 0])
            ops.append(("circle", goal_rgb, (centre, np.full(B, int(data[1]))), show_goal))
    rgb = {n: get(f"owned/variation/{n}/color/current").astype(np.int64) for n in ("block", "agent")}
    angle = {n: np.array([np.float64(r[f"current/body/{n}/angle"]) for r in rows]) for n in ("block", "agent")}
    pos = {
        n: np.stack([np.asarray(r[f"current/body/{n}/position"], np.float64) for r in rows])
        for n in ("block", "agent")
    }
    for name, kind, data in template.parts:
        c, s, tx, ty = _body_transform(angle[name], pos[name], template.cog[name])
        fill = rgb[name]
        if kind == "circle":
            centre = _rint(_apply(c, s, tx, ty, data[0][None])[:, 0])
            ops.append(("circle", fill, (centre, np.full(B, round(data[1]))), on))
            ops.append(("circle", _light(fill), (centre, np.full(B, round(data[1] - 4))), on))
            continue
        pts = _rint(_apply(c, s, tx, ty, data))
        ops.append(("poly", _light(fill), np.concatenate([pts, pts[:, :1]], 1), on))
        for i in range(len(data)):
            _fat_segment(ops, pts[:, i], pts[:, (i + 1) % len(data)], 2.0, fill, on)
    for a, b, radius, color in template.walls:
        _fat_segment(
            ops,
            np.broadcast_to(_rint(a), (B, 2)),
            np.broadcast_to(_rint(b), (B, 2)),
            radius,
            np.broadcast_to(np.array(color, np.int64), (B, 3)),
            on,
        )
    count = np.array([int(r["current/render_segment_count"]) for r in rows])
    for k in range(int(count.max())):
        segs = np.stack([r["current/render_segments"][k] for r in rows]).astype(np.float64)
        ops.append(("aaline", _rint(segs[:, 4:7]), (_rint(segs[:, 0:2]), _rint(segs[:, 2:4])), count > k))
    return ops


def _fat_segment(ops, p1, p2, radius, fill, on):
    B = len(p1)
    r = round(max(1, radius * 2))
    ops.append(("line", fill, (p1, p2, np.full(B, r)), on))
    if r > 2:
        o0, o1 = np.abs(p2[:, 1] - p1[:, 1]), np.abs(p2[:, 0] - p1[:, 0])
        scale = _scale(o0, o1, radius)
        q0, q1 = _rint(o0 * scale), _rint(o1 * scale)
        pts = np.stack(
            [
                np.stack([p1[:, 0] - q0, p1[:, 1] - q1], -1),
                np.stack([p1[:, 0] + q0, p1[:, 1] + q1], -1),
                np.stack([p2[:, 0] + q0, p2[:, 1] + q1], -1),
                np.stack([p2[:, 0] - q0, p2[:, 1] - q1], -1),
            ],
            1,
        )
        enabled = on & ~((o0 == 0) & (o1 == 0))
        rr = np.full(B, round(radius))
        ops.append(("poly", fill, pts, enabled))
        ops.append(("circle", fill, (p1, rr), enabled))
        ops.append(("circle", fill, (p2, rr), enabled))
