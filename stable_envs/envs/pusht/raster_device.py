"""PushT primitive records built on the device from State tensors, equal to ``raster.primitives``."""

from __future__ import annotations

import numpy as np
import torch

from . import raster as R

#: Most polygons per body in any PushT shape (``+``).
POLYS = 3
#: State leaves that pick the shape template of a row.
ANCHORS = tuple(
    f"anchor/variation/{body}/{leaf}/current/{part}"
    for body in ("agent", "block")
    for leaf in ("shape", "scale")
    for part in ("value", "type")
)


class Tables:
    """Device tables for every template seen so far, indexed by template id."""

    def __init__(self, device):
        self.device = device
        self.keys = {}
        self.layouts = []
        self.tensors = None
        # Anchor byte rows seen so far, their template ids, and a hash per row for the device lookup.
        self.rows = {}
        self.known = None

    def ids(self, fields_host, count):
        """Template id per row from host anchor columns; new templates extend the tables."""
        raw = np.concatenate(
            [np.ascontiguousarray(fields_host[name]).reshape(count, -1).view(np.uint8) for name in ANCHORS], 1
        )
        first, inverse = R._unique_rows(raw)
        ids = np.empty(len(first), np.int64)
        changed = False
        for index, lane0 in enumerate(first.tolist()):
            key = R.template_key(fields_host, lane0)
            if key not in self.keys:
                try:
                    layout = R._layout(R._template(key))
                except NotImplementedError:
                    layout = None
                self.keys[key] = len(self.layouts)
                self.layouts.append(layout)
                changed = True
            ids[index] = self.keys[key]
            row = raw[lane0].tobytes()
            if row not in self.rows:
                self.rows[row] = ids[index]
                self.known = None
        if changed or self.tensors is None:
            self._build()
        return ids[inverse]

    def lookup(self, fields, count):
        """``(ids, unknown)`` per row on the device, without a sync; ``learn`` adds the unknown rows."""
        raw = torch.cat([fields[name].reshape(count, -1).contiguous().view(torch.uint8) for name in ANCHORS], 1)
        if self.known is None or self.known[0].shape[1] != raw.shape[1]:
            return torch.zeros(count, dtype=torch.int64, device=self.device), torch.ones(
                count, dtype=torch.bool, device=self.device
            )
        rows, hashes, order, ids, weights = self.known
        hash_ = (raw.to(torch.int64) * weights).sum(1)
        at = torch.searchsorted(hashes, hash_).clamp(max=len(hashes) - 1)
        match = order[at]
        known = (hashes[at] == hash_) & (raw == rows[match]).all(1)
        return ids[match], ~known

    def learn(self, fields, count):
        """Add the templates of host rows ``fields`` to the tables."""
        raw_width = sum(np.ascontiguousarray(fields[name][:1]).nbytes for name in ANCHORS)
        self.ids(fields, count)
        if self.known is None:
            self._index(raw_width)

    def device_ids(self, fields, count):
        """Template id per row as a device tensor; only rows with unseen anchors go through the host."""
        ids, unknown = self.lookup(fields, count)
        if not bool(unknown.any()):
            return ids
        self.learn({name: fields[name].cpu().numpy() for name in ANCHORS}, count)
        return self.lookup(fields, count)[0]

    def _index(self, width):
        rows = torch.as_tensor(
            np.frombuffer(b"".join(self.rows), np.uint8).reshape(-1, width).copy(), device=self.device
        )
        ids = torch.as_tensor(np.fromiter(self.rows.values(), np.int64), device=self.device)
        # Byte weights below 2**40 keep every row sum inside int64.
        weights = torch.as_tensor(np.random.default_rng(0).integers(1, 1 << 40, width), device=self.device)
        hashes, order = torch.sort((rows.to(torch.int64) * weights).sum(1))
        self.known = (rows, hashes, order, ids, weights)

    def _build(self):
        count = len(self.layouts)
        slots = max([1] + [layout.slots for layout in self.layouts if layout is not None])
        goal = np.zeros((count, POLYS, 4, 2))
        goal_slots = np.full((count, POLYS), -1, np.int64)
        polys = np.zeros((count, 2, POLYS, 4, 2))
        fills = np.full((count, 2, POLYS), -1, np.int64)
        edges = np.full((count, 2, POLYS, 4, 4), -1, np.int64)
        circle = np.full((count, 2), -1, np.int64)
        radii = np.zeros((count, 2), np.int64)
        walls = np.zeros((count, slots, R.RECORD), np.int64)
        totals = np.zeros(count, np.int64)
        supported = np.zeros(count, np.bool_)
        for t, layout in enumerate(self.layouts):
            if layout is None:
                continue
            supported[t] = True
            totals[t] = layout.slots
            g = len(layout.goal)
            goal[t, :g] = layout.goal
            goal_slots[t, :g] = np.arange(g)
            for b, body in enumerate(("agent", "block")):
                k = len(layout.polys[body])
                polys[t, b, :k] = layout.polys[body]
                fills[t, b, :k] = layout.fills[body]
                edges[t, b, :k] = layout.edges[body].reshape(k, 4, 4)
            for owner, (ox, oy), radius, slot in layout.circles:
                if owner != "agent" or (ox, oy) != (0.0, 0.0):
                    raise NotImplementedError("Only the agent may be a disc at its body origin")
                circle[t] = (slot, slot + 1)
                radii[t] = (round(radius), round(radius - 4))
            walls[t, layout.wall_slots.reshape(-1)] = layout.walls
        d = self.device
        self.tensors = dict(
            goal=torch.as_tensor(goal, device=d),
            goal_slots=torch.as_tensor(goal_slots, device=d),
            polys=torch.as_tensor(polys, device=d),
            fills=torch.as_tensor(fills, device=d),
            edges=torch.as_tensor(edges, device=d),
            circle=torch.as_tensor(circle, device=d),
            radii=torch.as_tensor(radii, device=d),
            walls=torch.as_tensor(walls, device=d, dtype=torch.int32),
            totals=torch.as_tensor(totals, device=d),
            supported=torch.as_tensor(supported, device=d),
        )
        self.slots = slots


#: Distance from a rounding midpoint under which a device ``cos``/``sin`` result may round otherwise than libm's.
GUARD = 1e-9


def _near_half(value):
    """Rows of ``value [N, ...]`` that lie within ``GUARD`` of a ``k + 0.5`` rounding midpoint."""
    off = ((value - torch.floor(value)) - 0.5).abs() < GUARD
    return off.reshape(len(off), -1).any(1)


def _transform(px, py, cos, sin, cogx, cogy):
    gx = (cos * cogx + -sin * cogy) + px
    gy = (sin * cogx + cos * cogy) + py
    return cos, sin, -sin, cos, gx - (cogx * cos - cogy * sin), gy - (cogx * sin + cogy * cos)


def _apply(t, x, y):
    a, b, c, d, tx, ty = (v.view(-1, *([1] * (x.dim() - 1))) for v in t)
    return a * x + c * y + tx, b * x + d * y + ty


def _pack(color):
    color = color.to(torch.int64)
    return (color[..., 0] << 16) | (color[..., 1] << 8) | color[..., 2]


def _light(color):
    scaled = torch.minimum(
        torch.tensor(1.2, dtype=torch.float32, device=color.device) * color.to(torch.float32),
        torch.tensor(255.0, device=color.device),
    )
    return torch.round(scaled).to(torch.int64)


def _put(records, slots, values):
    """Scatter ``values [N, ..., RECORD]`` into ``records [N, P, RECORD]`` at ``slots [N, ...]`` (−1 skips)."""
    n = records.shape[0]
    flat_slots = slots.reshape(n, -1)
    flat_values = values.reshape(n, -1, R.RECORD)
    keep = flat_slots >= 0
    frame = torch.arange(n, device=records.device).view(-1, 1).expand_as(flat_slots)
    records[frame[keep], flat_slots[keep]] = flat_values[keep].to(records.dtype)


def _put_mask(mask, slots, values):
    """Set ``mask [N, P]`` at ``slots [N, ...]`` where ``values`` is true (−1 skips)."""
    n = mask.shape[0]
    flat_slots = slots.reshape(n, -1)
    keep = (flat_slots >= 0) & values.reshape(n, -1)
    frame = torch.arange(n, device=mask.device).view(-1, 1).expand_as(flat_slots)
    mask[frame[keep], flat_slots[keep]] = True


def _clip(records, clipped):
    """Give the line records marked in ``clipped`` the start and length of pygame's clipped run.

    Such lines are rare (a body partly off the canvas), so the runs come from ``raster.clipped_line``.
    """
    where = torch.nonzero(clipped)
    if not len(where):
        return
    rows = records[where[:, 0], where[:, 1]].cpu().numpy()
    for rec in rows:
        x1, y1 = int(rec[R.LX]), int(rec[R.LY])
        x2, y2 = x1 + int(rec[R.LSX] * rec[R.LDX]), y1 + int(rec[R.LSY] * rec[R.LDY])
        run = R.clipped_line(x1, y1, x2, y2)
        if run is None:
            rec[R.KIND] = 0
            run = x1, y1, int(rec[R.LN])
        sx, sy, steps = run
        rec[R.LX], rec[R.LY], rec[R.LN] = sx, sy, steps
        rec[R.BX0], rec[R.BX1] = sx - steps - 1, sx + steps + 2
        rec[R.BY0], rec[R.BY1] = sy - steps - 1, sy + steps + 2
    records[where[:, 0], where[:, 1]] = torch.as_tensor(rows, device=records.device)


def _poly(color, xs, ys, enabled, static):
    rec = torch.zeros((*xs.shape[:-1], R.RECORD), dtype=torch.int64, device=xs.device)
    rec[..., R.KIND] = torch.where(enabled, R.POLY, 0)
    rec[..., R.COLOR] = color
    rec[..., R.STATIC] = int(static)
    k = xs.shape[-1]
    rec[..., R.BX0], rec[..., R.BX1] = xs.amin(-1), xs.amax(-1)
    rec[..., R.BY0], rec[..., R.BY1] = ys.amin(-1), ys.amax(-1)
    rec[..., R.PN] = k
    rec[..., R.PX : R.PX + k] = xs
    rec[..., R.PY : R.PY + k] = ys
    return rec


def _circle(color, cx, cy, radius, enabled, static):
    rec = torch.zeros((*cx.shape, R.RECORD), dtype=torch.int64, device=cx.device)
    rec[..., R.KIND] = torch.where(enabled & (radius >= 1), R.CIRCLE, 0)
    rec[..., R.COLOR] = color
    rec[..., R.STATIC] = int(static)
    rec[..., R.BX0], rec[..., R.BX1] = cx - radius, cx + radius
    rec[..., R.BY0], rec[..., R.BY1] = cy - radius, cy + radius
    rec[..., R.CX], rec[..., R.CY], rec[..., R.CR] = cx, cy, radius
    return rec


def _line(color, x1, y1, x2, y2):
    """Width-4 line records; lines that leave the canvas are flagged for ``_clip``."""
    rec = torch.zeros((*x1.shape, R.RECORD), dtype=torch.int64, device=x1.device)
    dx, dy = (x2 - x1).abs(), (y2 - y1).abs()
    xinc = (dx <= dy).to(torch.int64)
    rec[..., R.KIND] = R.LINE
    rec[..., R.COLOR] = color
    rec[..., R.LN] = torch.maximum(dx, dy)
    rec[..., R.BX0] = torch.minimum(x1, x2) - xinc
    rec[..., R.BX1] = torch.maximum(x1, x2) + 2 * xinc
    rec[..., R.BY0] = torch.minimum(y1, y2) - (1 - xinc)
    rec[..., R.BY1] = torch.maximum(y1, y2) + 2 * (1 - xinc)
    rec[..., R.LX], rec[..., R.LY] = x1, y1
    rec[..., R.LSX] = torch.where(x1 < x2, 1, -1)
    rec[..., R.LSY] = torch.where(y1 < y2, 1, -1)
    rec[..., R.LDX], rec[..., R.LDY] = dx, dy
    rec[..., R.LERR] = torch.where(dx > dy, dx // 2, -(dy // 2))
    rec[..., R.LXINC] = xinc
    outside = (torch.minimum(x1, x2) < 0) | (torch.maximum(x1, x2) >= R.CANVAS)
    outside |= (torch.minimum(y1, y2) < 0) | (torch.maximum(y1, y2) >= R.CANVAS)
    return rec, outside


class Roots:
    """``float(n) ** 0.5`` for integer ``n``, as Python's C ``pow`` gives it, looked up on the device."""

    def __init__(self, device):
        # Edges with both ends on the canvas have squared lengths below this size.
        size = 2 * R.CANVAS * R.CANVAS
        self.table = torch.as_tensor(np.array([float(n) ** 0.5 for n in range(size)]), device=device)

    def __call__(self, squares):
        return self.table[squares.clamp(0, len(self.table) - 1)]


def _fat_segments(color, ax, ay, bx, by, enabled, roots, static=False):
    """Records ``[N, E, 4, RECORD]`` (line, quad, two circles) of ``draw_fat_segment`` with radius 2."""
    p1x, p1y, p2x, p2y = (torch.round(v).to(torch.int64) for v in (ax, ay, bx, by))
    line, outside = _line(color, p1x, p1y, p2x, p2y)
    line[..., R.KIND] = torch.where(enabled, line[..., R.KIND], 0)
    o0, o1 = (p2y - p1y).abs(), (p2x - p1x).abs()
    drawn = enabled & ((o0 != 0) | (o1 != 0))
    scale = 2.0 / torch.where(drawn, roots(o0 * o0 + o1 * o1), 1.0)
    q0, q1 = torch.round(o0 * scale).to(torch.int64), torch.round(o1 * scale).to(torch.int64)
    xs = torch.stack([p1x - q0, p1x + q0, p2x + q0, p2x - q0], -1)
    ys = torch.stack([p1y - q1, p1y + q1, p2y + q1, p2y - q1], -1)
    two = torch.full_like(p1x, 2)
    out = torch.stack(
        [
            line,
            _poly(color, xs, ys, drawn, static),
            _circle(color, p1x, p1y, two, drawn, static),
            _circle(color, p2x, p2y, two, drawn, static),
        ],
        -2,
    )
    out[..., R.STATIC] = int(static)
    return out, outside & enabled


@torch.no_grad()
def primitives(fields, tables, roots, ids):
    """Device version of ``raster.primitives`` for device render ``fields`` and template ``ids``.

    Returns ``(records [N, P, RECORD] int32, counts [N], fallback [N])``; fallback rows need the host path.
    """
    T = tables.tensors
    device = tables.device
    n = len(ids)
    ids = torch.as_tensor(ids, device=device)
    records = T["walls"][ids].clone()
    clipped = torch.zeros(records.shape[:2], dtype=torch.bool, device=device)
    counts = T["totals"][ids]
    config = fields["recipe/config"]
    fallback = (config[:, 2] != 0) | ~T["supported"][ids]
    color = {name: fields[f"owned/variation/{name}/color/current/value"] for name in ("agent", "block", "goal")}
    # Goal polygons: points rounded, first point repeated.
    goal_pos = fields["owned/variation/goal/position/current/value"].to(torch.float64)
    goal_angle = fields["owned/variation/goal/angle/current/value"].to(torch.float64).reshape(n)
    render_goal = (fields["owned/variation/rendering/render_goal/current/value"].reshape(n) != 0) & (config[:, 3] != 0)
    zero = torch.zeros(n, dtype=torch.float64, device=device)
    angles = torch.stack(
        [goal_angle, fields["current/body/agent/angle"].reshape(n), fields["current/body/block/angle"].reshape(n)], 1
    )
    # CUDA cos/sin may differ from libm in the last bits; rows with a coordinate near a rounding midpoint fall back.
    cos, sin = torch.cos(angles), torch.sin(angles)
    goal = _transform(goal_pos[:, 0], goal_pos[:, 1], cos[:, 0], sin[:, 0], zero, zero)
    local = T["goal"][ids]
    wx, wy = _apply(goal, local[..., 0], local[..., 1])
    fallback |= _near_half(torch.where(render_goal.view(-1, 1, 1), wx, 0.0)) | _near_half(
        torch.where(render_goal.view(-1, 1, 1), wy, 0.0)
    )
    xs, ys = torch.round(wx).to(torch.int64), torch.round(wy).to(torch.int64)
    xs, ys = torch.cat([xs, xs[..., :1]], -1), torch.cat([ys, ys[..., :1]], -1)
    goal_slots = T["goal_slots"][ids]
    on = render_goal.view(-1, 1) & (goal_slots >= 0)
    _put(records, goal_slots, _poly(_pack(color["goal"]).view(-1, 1), xs, ys, on, True))
    for b, body in enumerate(("agent", "block")):
        pos = fields[f"current/body/{body}/position"]
        cog = fields[f"current/body/{body}/center_of_gravity"]
        transform = _transform(pos[:, 0], pos[:, 1], cos[:, b + 1], sin[:, b + 1], cog[:, 0], cog[:, 1])
        fill = _pack(color[body])
        light = _pack(_light(color[body]))
        local = T["polys"][ids, b]
        fills = T["fills"][ids, b]
        present = fills >= 0
        wx, wy = _apply(transform, local[..., 0], local[..., 1])
        drawn = present.unsqueeze(-1)
        fallback |= _near_half(torch.where(drawn, wx, 0.0)) | _near_half(torch.where(drawn, wy, 0.0))
        xs, ys = torch.round(wx).to(torch.int64), torch.round(wy).to(torch.int64)
        xs, ys = torch.cat([xs, xs[..., :1]], -1), torch.cat([ys, ys[..., :1]], -1)
        _put(records, fills, _poly(light.view(-1, 1), xs, ys, present, False))
        follow = [1, 2, 3, 0]
        segs, outside = _fat_segments(
            fill.view(-1, 1, 1),
            wx,
            wy,
            wx[..., follow],
            wy[..., follow],
            present.unsqueeze(-1).expand(-1, -1, 4),
            roots,
        )
        _put_mask(clipped, T["edges"][ids, b][..., 0], outside)
        _put(records, T["edges"][ids, b], segs)
        if body == "agent":
            slots = T["circle"][ids]
            radii = T["radii"][ids]
            has = slots[:, 0] >= 0
            fallback |= has & (_near_half(transform[4].view(-1, 1)) | _near_half(transform[5].view(-1, 1)))
            cx = torch.round(transform[4]).to(torch.int64)
            cy = torch.round(transform[5]).to(torch.int64)
            rings = torch.stack(
                [_circle(fill, cx, cy, radii[:, 0], has, False), _circle(light, cx, cy, radii[:, 1], has, False)], 1
            )
            _put(records, slots, rings)
    _clip(records, clipped)
    return records.to(torch.int32), counts, fallback


def render_keys():
    """State leaves that ``render`` reads, as ``/``-joined names."""
    names = ["recipe/config", "current/render_segments", "current/render_segment_count"]
    names += [f"current/body/{b}/{f}" for b in ("agent", "block") for f in ("position", "angle", "center_of_gravity")]
    names += [f"owned/variation/{n}/color/current/value" for n in ("agent", "block", "goal", "background")]
    names += ["owned/variation/goal/position/current/value", "owned/variation/goal/angle/current/value"]
    names += ["owned/variation/rendering/render_goal/current/value"]
    return names + list(ANCHORS)


class StaticFrames:
    """Frames of the background, goal and walls alone, kept on the device across calls per output size."""

    #: Most frames kept per size; the cache starts again when it is full.
    LIMIT = 512

    def __init__(self, device):
        self.device = device
        self.entries = {}

    @staticmethod
    def keys(records, background, ids):
        """Per row: template id, background and goal records (int64), which fix the static frame."""
        static = (records[..., R.STATIC] != 0) & (records[..., R.KIND] != 0)
        goal = torch.cumprod(static.to(torch.int32), 1).bool()[:, :3]
        parts = [
            ids.view(-1, 1),
            background.view(-1, 1),
            torch.where(goal.unsqueeze(-1), records[:, :3], 0).flatten(1),
        ]
        return torch.cat([part.to(torch.int64) for part in parts], 1)

    def lookup(self, keys, size):
        """``(index, frames, missing)``: index −1 marks rows without a cached frame; no sync."""
        entry = self.entries.get(size)
        if entry is None:
            missing = torch.ones(len(keys), dtype=torch.bool, device=self.device)
            return torch.full((len(keys),), -1, dtype=torch.int32, device=self.device), None, missing
        rows, hashes, order, frames, weights = entry
        hash_ = (keys * weights).sum(1)
        at = torch.searchsorted(hashes, hash_).clamp(max=len(hashes) - 1)
        index = order[at]
        found = (hashes[at] == hash_) & (keys == rows[index]).all(1)
        return torch.where(found, index, -1).to(torch.int32), frames, ~found

    def add(self, keys, missing, records, counts, background, segments, size, tile):
        """Draw the frames of the missing keys; skip when they are too many to share."""
        lanes = torch.nonzero(missing).view(-1)
        new, inverse = torch.unique(keys[lanes], dim=0, return_inverse=True)
        if len(new) * 4 > max(len(keys), 16):
            return
        first = lanes[
            torch.zeros(len(new), dtype=torch.int64, device=self.device).scatter_(
                0, inverse, torch.arange(len(lanes), device=self.device)
            )
        ]
        static = (records[first, :, R.STATIC] != 0) & (records[first, :, R.KIND] != 0)
        fixed = torch.where(static.unsqueeze(-1), records[first], 0)
        empty = torch.zeros(len(first), dtype=torch.int32, device=self.device)
        frames = R.launch_segments(fixed, counts[first], background[first], segments[first][:, :1], empty, size, tile)
        entry = self.entries.get(size)
        if entry is not None and len(entry[0]) + len(new) <= self.LIMIT:
            new, frames = torch.cat([entry[0], new]), torch.cat([entry[3], frames])
        weights = torch.as_tensor(np.random.default_rng(1).integers(1, 1 << 40, new.shape[1]), device=self.device)
        hashes, order = torch.sort((new * weights).sum(1))
        self.entries[size] = (new, hashes, order, frames, weights)


class DeviceRenderer:
    """Render PushT State tensors on CUDA with Triton, with primitives built on the device.

    A call reads one small flag tensor back to the host; templates, clipped lines, static frames and
    host fallbacks that it reports take extra work only when they occur.
    """

    def __init__(self, device):
        self.device = torch.device(device)
        self.tables = Tables(self.device)
        self.roots = Roots(self.device)
        self.static = StaticFrames(self.device)

    @torch.no_grad()
    def __call__(self, fields, size, fallback, tile=16, warps=2):
        """``fields``: ``{name: tensor}`` on the device (see ``render_keys``); ``fallback(rows)`` draws host rows."""
        n = len(fields["recipe/config"])
        if self.tables.tensors is None:
            self.tables.learn({name: fields[name].cpu().numpy() for name in ANCHORS}, n)
        ids, unknown = self.tables.lookup(fields, n)
        max_radius = R._device_tables(size, self.device)["max_radius"]
        background = _pack(fields["owned/variation/background/color/current/value"])
        segments = fields["current/render_segments"].to(torch.float64)
        segment_counts = fields["current/render_segment_count"]
        records, counts, unsupported, clipped = fused_primitives(fields, self.tables, self.roots, ids, max_radius)
        keys = self.static.keys(records, background, ids)
        index, frames, missing = self.static.lookup(keys, size)
        flags = torch.stack([unknown.any(), clipped.any(), unsupported.any(), missing.any()]).cpu().tolist()
        if flags[0]:
            self.tables.learn({name: fields[name].cpu().numpy() for name in ANCHORS}, n)
            return self(fields, size, fallback, tile, warps)
        if flags[1]:
            _clip(records, clipped)
        if flags[3]:
            self.static.add(keys, missing, records, counts, background, segments, size, tile)
            index, frames, _ = self.static.lookup(keys, size)
        static = (
            None if frames is None else (index, frames, torch.zeros((n, 4), dtype=torch.int32, device=self.device))
        )
        bins = R.tile_bins(records, counts, size, tile, segments, segment_counts, width=records.shape[1])
        out = R.launch_segments(
            records, counts, background, segments, segment_counts, size, tile, None, static, bins, warps
        )
        if flags[2]:
            lanes = torch.nonzero(unsupported).view(-1)
            host = {k: v[lanes].cpu().numpy() for k, v in fields.items()}
            out[lanes] = torch.as_tensor(fallback(host), device=self.device)
        return out


BIG = 1 << 30


def _sort5(h):
    """Sort the last axis of five values with the network the kernel uses."""
    h = list(h.unbind(-1))
    for i, j in ((0, 1), (3, 4), (2, 4), (2, 3), (0, 3), (0, 2), (1, 4), (1, 3), (1, 2)):
        h[i], h[j] = torch.minimum(h[i], h[j]), torch.maximum(h[i], h[j])
    return h


@torch.no_grad()
def rows(records):
    """Turn the dynamic polygons and lines of ``records`` into ``ROWS`` primitives.

    Returns ``(records, intervals [rows, 8])``: per covered canvas row, up to four inclusive
    ``lo, hi`` x intervals that equal the pygame fill of that row.
    """
    records = records.clone()
    kind = records[..., R.KIND]
    static = records[..., R.STATIC] != 0
    pick = ((kind == R.POLY) | (kind == R.LINE)) & ~static
    frame, slot = torch.nonzero(pick, as_tuple=True)
    rec = records[frame, slot].to(torch.int64)
    top, bottom = rec[:, R.BY0], rec[:, R.BY1]
    heights = (bottom - top + 1).clamp(min=0)
    starts = torch.cumsum(heights, 0) - heights
    total = int(heights.sum().item()) if len(heights) else 0
    owner = torch.repeat_interleave(torch.arange(len(rec), device=rec.device), heights)
    y = top[owner] + torch.arange(total, device=rec.device) - starts[owner]
    out = torch.zeros((total, 8), dtype=torch.int64, device=rec.device)
    out[:, 0::2] = 1  # empty intervals
    r = rec[owner]
    poly = r[:, R.KIND] == R.POLY
    # Polygons: draw_fillpoly per row.
    n = torch.where(poly, r[:, R.PN], 0).clamp(0, 5)
    xs, ys = r[:, R.PX : R.PX + 5], r[:, R.PY : R.PY + 5]
    use = torch.arange(5, device=rec.device).view(1, 5) < n.view(-1, 1)
    miny = torch.where(use, ys, BIG).amin(1)
    maxy = torch.where(use, ys, -BIG).amax(1)
    minx = torch.where(use, xs, BIG).amin(1)
    maxx = torch.where(use, xs, -BIG).amax(1)
    prev = (torch.arange(5, device=rec.device).view(1, 5) - 1) % n.view(-1, 1).clamp(min=1)
    xa, ya = torch.gather(xs, 1, prev), torch.gather(ys, 1, prev)
    up = ya < ys
    y1, y2 = torch.where(up, ya, ys), torch.where(up, ys, ya)
    x1, x2 = torch.where(up, xa, xs), torch.where(up, xs, xa)
    edge = use & (ya != ys)
    yy = y.view(-1, 1)
    on = edge & (((yy >= y1) & (yy < y2)) | ((yy == maxy.view(-1, 1)) & (y2 == maxy.view(-1, 1))))
    den = torch.where(edge, y2 - y1, 1)
    hits = torch.where(on, torch.div((yy - y1) * (x2 - x1) + x1 * den, den, rounding_mode="trunc"), BIG)
    h = _sort5(hits)
    count = (hits != BIG).sum(1)
    flat = miny == maxy
    first = (count >= 2) & ~flat
    second = (count >= 4) & ~flat
    out[:, 0] = torch.where(poly & first, h[0], torch.where(poly & flat, minx, out[:, 0]))
    out[:, 1] = torch.where(poly & first, h[1], torch.where(poly & flat, maxx, out[:, 1]))
    out[:, 2] = torch.where(poly & second, h[2], out[:, 2])
    out[:, 3] = torch.where(poly & second, h[3], out[:, 3])
    # Horizontal edges strictly between the top and bottom rows add their own span.
    level = use & (ya == ys) & (miny.view(-1, 1) < ys) & (ys < maxy.view(-1, 1)) & (yy == ys) & ~flat.view(-1, 1)
    level &= poly.view(-1, 1)
    spans = level.sum(1)
    lo, hi = torch.minimum(xa, xs), torch.maximum(xa, xs)
    order = torch.sort((~level).to(torch.int8), dim=1, stable=True).indices
    lo, hi = torch.gather(lo, 1, order), torch.gather(hi, 1, order)
    out[:, 4] = torch.where(spans >= 1, lo[:, 0], out[:, 4])
    out[:, 5] = torch.where(spans >= 1, hi[:, 0], out[:, 5])
    out[:, 6] = torch.where(spans >= 2, lo[:, 1], out[:, 6])
    out[:, 7] = torch.where(spans >= 2, hi[:, 1], out[:, 7])
    overflow = (spans > 2) | (poly & (count > 4))
    # Steep lines: one horizontal span per row.
    line = ~poly
    lx, ly, sx, sy = r[:, R.LX], r[:, R.LY], r[:, R.LSX], r[:, R.LSY]
    dx, dy, err, steps, xinc = r[:, R.LDX], r[:, R.LDY], r[:, R.LERR], r[:, R.LN], r[:, R.LXINC] != 0
    k = (y - ly) * sy
    nk = torch.where(dy > 0, torch.div(k.clamp(min=0) * dx + err + dy - 1, dy.clamp(min=1), rounding_mode="floor"), 0)
    xk = lx + sx * nk
    steep = line & xinc & (k >= 0) & (k <= steps)
    out[:, 0] = torch.where(steep, xk - 1, out[:, 0])
    out[:, 1] = torch.where(steep, xk + 2, out[:, 1])
    # Shallow lines: four-pixel columns, gathered into one x interval per row.
    lines = torch.nonzero((rec[:, R.KIND] == R.LINE) & (rec[:, R.LXINC] == 0)).view(-1)
    if len(lines):
        q = rec[lines]
        width = q[:, R.LN] + 1
        cstart = torch.cumsum(width, 0) - width
        col_owner = torch.repeat_interleave(torch.arange(len(q), device=rec.device), width)
        kk = torch.arange(int(width.sum().item()), device=rec.device) - cstart[col_owner]
        qq = q[col_owner]
        mk = torch.div(
            kk * qq[:, R.LDY] - qq[:, R.LERR] + qq[:, R.LDX] - 1, qq[:, R.LDX].clamp(min=1), rounding_mode="floor"
        )
        xk = qq[:, R.LX] + qq[:, R.LSX] * kk
        yk = qq[:, R.LY] + qq[:, R.LSY] * mk
        base = starts[lines][col_owner]
        top_q = q[col_owner][:, R.BY0]
        lo_row = torch.full((total,), BIG, dtype=torch.int64, device=rec.device)
        hi_row = torch.full((total,), -BIG, dtype=torch.int64, device=rec.device)
        for d in (-1, 0, 1, 2):
            index = base + (yk + d - top_q)
            lo_row.scatter_reduce_(0, index, xk, "amin")
            hi_row.scatter_reduce_(0, index, xk, "amax")
        set_rows = lo_row <= hi_row
        out[:, 0] = torch.where(set_rows, lo_row, out[:, 0])
        out[:, 1] = torch.where(set_rows, hi_row, out[:, 1])
    rec[:, R.KIND] = R.ROWS
    rec[:, R.ROFF] = starts
    records[frame, slot] = rec.to(records.dtype)
    bad = torch.zeros(records.shape[0], dtype=torch.bool, device=records.device)
    if overflow.any():
        bad[frame[owner[overflow]]] = True
    return records, out.to(torch.int32), bad


if R.HAVE_TRITON:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    _REC = tl.constexpr(R.RECORD)
    _KIND, _COLOR, _BX0, _BY0, _BX1, _BY1 = (tl.constexpr(v) for v in (R.KIND, R.COLOR, R.BX0, R.BY0, R.BX1, R.BY1))
    _PN, _PX, _PY, _CX, _CY, _CR = (tl.constexpr(v) for v in (R.PN, R.PX, R.PY, R.CX, R.CY, R.CR))
    _LX, _LY, _LSX, _LSY, _LDX, _LDY = (tl.constexpr(v) for v in (R.LX, R.LY, R.LSX, R.LSY, R.LDX, R.LDY))
    _LERR, _LN, _LXINC, _STATIC = (tl.constexpr(v) for v in (R.LERR, R.LN, R.LXINC, R.STATIC))
    _POLY, _CIRCLE, _LINE, _CANVAS = (tl.constexpr(v) for v in (R.POLY, R.CIRCLE, R.LINE, R.CANVAS))
    _POLYS = tl.constexpr(POLYS)
    _GUARD = tl.constexpr(GUARD)

    @triton.jit
    def _k_near(v):
        return tl.abs((v - libdevice.floor(v)) - 0.5) < _GUARD

    @triton.jit
    def _k_color(ptr):
        r = tl.load(ptr).to(tl.int32)
        g = tl.load(ptr + 1).to(tl.int32)
        b = tl.load(ptr + 2).to(tl.int32)
        return (r << 16) | (g << 8) | b

    @triton.jit
    def _k_lit(c):
        v = libdevice.rint(tl.minimum(c.to(tl.float32) * 1.2, 255.0))
        return v.to(tl.int32)

    @triton.jit
    def _k_light_color(ptr):
        return (_k_lit(tl.load(ptr)) << 16) | (_k_lit(tl.load(ptr + 1)) << 8) | _k_lit(tl.load(ptr + 2))

    @triton.jit
    def _k_transform(px, py, c, s, gx_, gy_):
        """Chipmunk body transform ``(tx, ty)`` for position, cos/sin and centre of gravity."""
        gx = libdevice.add_rn(libdevice.add_rn(libdevice.mul_rn(c, gx_), libdevice.mul_rn(-s, gy_)), px)
        gy = libdevice.add_rn(libdevice.add_rn(libdevice.mul_rn(s, gx_), libdevice.mul_rn(c, gy_)), py)
        tx = libdevice.sub_rn(gx, libdevice.sub_rn(libdevice.mul_rn(gx_, c), libdevice.mul_rn(gy_, s)))
        ty = libdevice.sub_rn(gy, libdevice.add_rn(libdevice.mul_rn(gx_, s), libdevice.mul_rn(gy_, c)))
        return tx, ty

    @triton.jit
    def _k_apply(c, s, tx, ty, x, y):
        wx = libdevice.add_rn(libdevice.add_rn(libdevice.mul_rn(c, x), libdevice.mul_rn(-s, y)), tx)
        wy = libdevice.add_rn(libdevice.add_rn(libdevice.mul_rn(s, x), libdevice.mul_rn(c, y)), ty)
        return wx, wy

    @triton.jit
    def _k_round(v):
        return libdevice.rint(v).to(tl.int32)

    @triton.jit
    def _store_poly(rec, kind, color, static, n: tl.constexpr, x0, x1, x2, x3, y0, y1, y2, y3):
        """``_poly`` record of four points, or five with the first repeated."""
        tl.store(rec + _KIND, kind)
        tl.store(rec + _COLOR, color)
        tl.store(rec + _STATIC, static)
        tl.store(rec + _BX0, tl.minimum(tl.minimum(x0, x1), tl.minimum(x2, x3)))
        tl.store(rec + _BX1, tl.maximum(tl.maximum(x0, x1), tl.maximum(x2, x3)))
        tl.store(rec + _BY0, tl.minimum(tl.minimum(y0, y1), tl.minimum(y2, y3)))
        tl.store(rec + _BY1, tl.maximum(tl.maximum(y0, y1), tl.maximum(y2, y3)))
        tl.store(rec + _PN, n)
        tl.store(rec + _PX, x0)
        tl.store(rec + _PX + 1, x1)
        tl.store(rec + _PX + 2, x2)
        tl.store(rec + _PX + 3, x3)
        tl.store(rec + _PY, y0)
        tl.store(rec + _PY + 1, y1)
        tl.store(rec + _PY + 2, y2)
        tl.store(rec + _PY + 3, y3)
        if n == 5:
            tl.store(rec + _PX + 4, x0)
            tl.store(rec + _PY + 4, y0)

    @triton.jit
    def _store_circle(rec, kind, color, cx, cy, r):
        tl.store(rec + _KIND, kind)
        tl.store(rec + _COLOR, color)
        tl.store(rec + _STATIC, 0)
        tl.store(rec + _BX0, cx - r)
        tl.store(rec + _BX1, cx + r)
        tl.store(rec + _BY0, cy - r)
        tl.store(rec + _BY1, cy + r)
        tl.store(rec + _CX, cx)
        tl.store(rec + _CY, cy)
        tl.store(rec + _CR, r)

    @triton.jit
    def _fat_segment(records, clipped, slot_ptr, color, ax, ay, bx, by, roots, top):
        """``_fat_segments`` for one edge: width-4 line, quad and two end circles at the slots in ``slot_ptr``."""
        p1x, p1y, p2x, p2y = _k_round(ax), _k_round(ay), _k_round(bx), _k_round(by)
        s_line, s_quad = tl.load(slot_ptr), tl.load(slot_ptr + 1)
        s_c1, s_c2 = tl.load(slot_ptr + 2), tl.load(slot_ptr + 3)
        dx, dy = tl.abs(p2x - p1x), tl.abs(p2y - p1y)
        xinc = (dx <= dy).to(tl.int32)
        if s_line >= 0:
            rec = records + s_line * _REC
            tl.store(rec + _KIND, _LINE)
            tl.store(rec + _COLOR, color)
            tl.store(rec + _STATIC, 0)
            tl.store(rec + _LN, tl.maximum(dx, dy))
            tl.store(rec + _BX0, tl.minimum(p1x, p2x) - xinc)
            tl.store(rec + _BX1, tl.maximum(p1x, p2x) + 2 * xinc)
            tl.store(rec + _BY0, tl.minimum(p1y, p2y) - (1 - xinc))
            tl.store(rec + _BY1, tl.maximum(p1y, p2y) + 2 * (1 - xinc))
            tl.store(rec + _LX, p1x)
            tl.store(rec + _LY, p1y)
            tl.store(rec + _LSX, tl.where(p1x < p2x, 1, -1))
            tl.store(rec + _LSY, tl.where(p1y < p2y, 1, -1))
            tl.store(rec + _LDX, dx)
            tl.store(rec + _LDY, dy)
            tl.store(rec + _LERR, tl.where(dx > dy, dx // 2, -(dy // 2)))
            tl.store(rec + _LXINC, xinc)
            out = (tl.minimum(p1x, p2x) < 0) | (tl.maximum(p1x, p2x) >= _CANVAS)
            out |= (tl.minimum(p1y, p2y) < 0) | (tl.maximum(p1y, p2y) >= _CANVAS)
            if out:
                tl.store(clipped + s_line, 1)
        drawn = (dx != 0) | (dy != 0)
        o0, o1 = dy, dx
        square = tl.minimum(o0.to(tl.int64) * o0 + o1.to(tl.int64) * o1, top)
        root = tl.where(drawn, tl.load(roots + square), 1.0)
        scale = 2.0 / root
        q0 = _k_round(libdevice.mul_rn(o0.to(tl.float64), scale))
        q1 = _k_round(libdevice.mul_rn(o1.to(tl.float64), scale))
        kind = tl.where(drawn, _POLY, 0)
        if s_quad >= 0:
            _store_poly(
                records + s_quad * _REC,
                kind,
                color,
                0,
                4,
                p1x - q0,
                p1x + q0,
                p2x + q0,
                p2x - q0,
                p1y - q1,
                p1y + q1,
                p2y + q1,
                p2y - q1,
            )
        if s_c1 >= 0:
            _store_circle(records + s_c1 * _REC, tl.where(drawn, _CIRCLE, 0), color, p1x, p1y, 2)
        if s_c2 >= 0:
            _store_circle(records + s_c2 * _REC, tl.where(drawn, _CIRCLE, 0), color, p2x, p2y, 2)

    @triton.jit
    def _primitives_kernel(
        records,
        clipped,
        fallback,
        ids,
        config,
        colors,
        goal_pos,
        goal_angle,
        render_goal,
        positions,
        angles,
        cogs,
        goal_table,
        goal_slots,
        polys,
        fills,
        edges,
        circle,
        radii,
        supported,
        roots,
        top,
        P,
        max_radius,
    ):
        """One frame per program: write the goal, body and agent-disc records of ``primitives``."""
        f = tl.program_id(0).to(tl.int64)
        t = tl.load(ids + f)
        rec = records + f * P * _REC
        clip = clipped + f * P
        bad = (tl.load(config + f * 4 + 2) != 0) | (tl.load(supported + t) == 0)
        show = (tl.load(render_goal + f) != 0) & (tl.load(config + f * 4 + 3) != 0)
        # Goal polygons: rotation about the goal position.
        ga = tl.load(goal_angle + f)
        gc, gs = tl.cos(ga), tl.sin(ga)
        zero = tl.full((), 0.0, tl.float64)
        gtx, gty = _k_transform(tl.load(goal_pos + f * 2), tl.load(goal_pos + f * 2 + 1), gc, gs, zero, zero)
        goal_color = _k_color(colors + f * 9 + 6)
        for g in tl.static_range(_POLYS):
            slot = tl.load(goal_slots + t * _POLYS + g)
            if slot >= 0:
                base = goal_table + (t * _POLYS + g) * 8
                x0, y0 = _k_apply(gc, gs, gtx, gty, tl.load(base), tl.load(base + 1))
                x1, y1 = _k_apply(gc, gs, gtx, gty, tl.load(base + 2), tl.load(base + 3))
                x2, y2 = _k_apply(gc, gs, gtx, gty, tl.load(base + 4), tl.load(base + 5))
                x3, y3 = _k_apply(gc, gs, gtx, gty, tl.load(base + 6), tl.load(base + 7))
                near = (
                    _k_near(x0)
                    | _k_near(x1)
                    | _k_near(x2)
                    | _k_near(x3)
                    | _k_near(y0)
                    | _k_near(y1)
                    | _k_near(y2)
                    | _k_near(y3)
                )
                bad |= show & near
                _store_poly(
                    rec + slot * _REC,
                    tl.where(show, _POLY, 0),
                    goal_color,
                    1,
                    5,
                    _k_round(x0),
                    _k_round(x1),
                    _k_round(x2),
                    _k_round(x3),
                    _k_round(y0),
                    _k_round(y1),
                    _k_round(y2),
                    _k_round(y3),
                )
        for b in tl.static_range(2):
            px, py = tl.load(positions + f * 4 + b * 2), tl.load(positions + f * 4 + b * 2 + 1)
            cx_, cy_ = tl.load(cogs + f * 4 + b * 2), tl.load(cogs + f * 4 + b * 2 + 1)
            a = tl.load(angles + f * 2 + b)
            c, s = tl.cos(a), tl.sin(a)
            tx, ty = _k_transform(px, py, c, s, cx_, cy_)
            fill = _k_color(colors + f * 9 + b * 3)
            light = _k_light_color(colors + f * 9 + b * 3)
            for k in tl.static_range(_POLYS):
                slot = tl.load(fills + (t * 2 + b) * _POLYS + k)
                if slot >= 0:
                    base = polys + ((t * 2 + b) * _POLYS + k) * 8
                    x0, y0 = _k_apply(c, s, tx, ty, tl.load(base), tl.load(base + 1))
                    x1, y1 = _k_apply(c, s, tx, ty, tl.load(base + 2), tl.load(base + 3))
                    x2, y2 = _k_apply(c, s, tx, ty, tl.load(base + 4), tl.load(base + 5))
                    x3, y3 = _k_apply(c, s, tx, ty, tl.load(base + 6), tl.load(base + 7))
                    bad |= (
                        _k_near(x0)
                        | _k_near(x1)
                        | _k_near(x2)
                        | _k_near(x3)
                        | _k_near(y0)
                        | _k_near(y1)
                        | _k_near(y2)
                        | _k_near(y3)
                    )
                    _store_poly(
                        rec + slot * _REC,
                        _POLY,
                        light,
                        0,
                        5,
                        _k_round(x0),
                        _k_round(x1),
                        _k_round(x2),
                        _k_round(x3),
                        _k_round(y0),
                        _k_round(y1),
                        _k_round(y2),
                        _k_round(y3),
                    )
                    e = edges + ((t * 2 + b) * _POLYS + k) * 16
                    _fat_segment(rec, clip, e, fill, x0, y0, x1, y1, roots, top)
                    _fat_segment(rec, clip, e + 4, fill, x1, y1, x2, y2, roots, top)
                    _fat_segment(rec, clip, e + 8, fill, x2, y2, x3, y3, roots, top)
                    _fat_segment(rec, clip, e + 12, fill, x3, y3, x0, y0, roots, top)
            if b == 0:
                s0 = tl.load(circle + t * 2)
                if s0 >= 0:
                    r0, r1 = tl.load(radii + t * 2).to(tl.int32), tl.load(radii + t * 2 + 1).to(tl.int32)
                    bad |= _k_near(tx) | _k_near(ty) | (r0 > max_radius)
                    ox, oy = _k_round(tx), _k_round(ty)
                    _store_circle(rec + s0 * _REC, tl.where(r0 >= 1, _CIRCLE, 0), fill, ox, oy, r0)
                    s1 = tl.load(circle + t * 2 + 1)
                    _store_circle(rec + s1 * _REC, tl.where(r1 >= 1, _CIRCLE, 0), light, ox, oy, r1)
        tl.store(fallback + f, bad.to(tl.int8))


def fused_primitives(fields, tables, roots, ids, max_radius):
    """``primitives`` in one Triton kernel; also flags rows whose agent disc exceeds ``max_radius``.

    Returns ``(records, counts, fallback, clipped)``; the lines marked in ``clipped`` still need ``_clip``.
    """
    T = tables.tensors
    device = tables.device
    n = len(ids)
    ids = torch.as_tensor(ids, device=device)
    records = T["walls"][ids]
    clipped = torch.zeros(records.shape[:2], dtype=torch.int8, device=device)
    fallback = torch.empty(n, dtype=torch.int8, device=device)

    def rows(name, width):
        return fields[name].reshape(n, width).to(torch.float64).contiguous()

    colors = torch.cat([rows(f"owned/variation/{b}/color/current/value", 3) for b in ("agent", "block", "goal")], 1)
    positions = torch.cat([rows(f"current/body/{b}/position", 2) for b in ("agent", "block")], 1)
    cogs = torch.cat([rows(f"current/body/{b}/center_of_gravity", 2) for b in ("agent", "block")], 1)
    angles = torch.cat([rows(f"current/body/{b}/angle", 1) for b in ("agent", "block")], 1)
    _primitives_kernel[(n,)](
        records, clipped, fallback, ids,
        fields["recipe/config"].reshape(n, 4).contiguous(), colors,
        rows("owned/variation/goal/position/current/value", 2), rows("owned/variation/goal/angle/current/value", 1),
        rows("owned/variation/rendering/render_goal/current/value", 1), positions, angles, cogs,
        T["goal"], T["goal_slots"], T["polys"], T["fills"], T["edges"], T["circle"], T["radii"],
        T["supported"].to(torch.int8), roots.table, len(roots.table) - 1, records.shape[1], max_radius,
    )  # fmt: skip
    return records, T["totals"][ids], fallback.bool(), clipped.bool()
