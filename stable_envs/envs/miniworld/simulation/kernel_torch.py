"""Torch step and reset of MovingBlockWorld environments, vectorized over environments, on any device.

The same rules as ``kernel_cpu.py``; blocks that may collide with each other move one after another as in the
source, other blocks all at once. Every shape is fixed, so the CUDA step can be captured in a graph.
"""

from __future__ import annotations

import math

import torch

from . import rng as R
from .kernel_cpu import AGENT_RADIUS, CAM_HEIGHT, CANDIDATES, GRID_TRIES
from .world import (
    P_AGENT_CENTER,
    P_AGENT_OVERLAP,
    P_BALL_RADIUS,
    P_BLOCK_H,
    P_BLOCK_XY,
    P_BOX_AND_BALL,
    P_BOX_OVERLAP,
    P_BOX_TEX,
    P_BOX_TEX_RANDOM,
    P_CARDINAL,
    P_CEIL_TEX,
    P_ENSURE_BASE,
    P_FLOOR_TEX,
    P_FLOOR_TEX_RANDOM,
    P_FWD,
    P_GRID,
    P_HEADING_ZERO,
    P_MAX_FWD,
    P_MAX_STEPS,
    P_RANDOM_ORIENT,
    P_REPEAT,
    P_ROOM,
    P_SPAWN_BUFFER,
    P_SPEED_SCALE,
    P_STATIC,
    P_TORUS,
    P_TURN,
    P_VMAX,
    P_VMIN,
    P_WALL_TEX,
    P_WALL_TEX_RANDOM,
    P_WRAP_HEADING,
    World,
)

#: Corners ``(ax, az, bx, bz)`` of the wall segments of an ``S`` x ``S`` room, as factors of ``S``.
_SEGMENTS = ((1, 0, 1, 1), (0, 0, 1, 0), (0, 1, 0, 0), (1, 1, 0, 1))


class TorchKernel:
    """Step and reset functions over a dict of State tensors (see ``batched.py``)."""

    def __init__(self, world: World, device: torch.device):
        P = [float(v) for v in world.P]
        self.P, self.world, self.device = P, world, device
        self.S = P[P_ROOM]
        self.grid, self.torus, self.static = P[P_GRID] > 0, P[P_TORUS] > 0, P[P_STATIC] > 0
        self.box_overlap, self.agent_overlap = P[P_BOX_OVERLAP] > 0, P[P_AGENT_OVERLAP] > 0
        as_t = lambda a: torch.as_tensor(a, dtype=torch.int64, device=device)  # noqa: E731
        self.pool, self.base_palette = as_t(world.pool), as_t(world.base_palette)
        # The rounded point, then its neighbours in the source's order (dx outer, dz inner).
        offsets = [(0.0, 0.0)] + [(float(dx), float(dz)) for dx in range(-1, 2) for dz in range(-1, 2)]
        self._dx = torch.tensor([o[0] for o in offsets], dtype=torch.float64, device=device)
        self._dz = torch.tensor([o[1] for o in offsets], dtype=torch.float64, device=device)
        self.floor_pool, self.wall_pool, self.box_pool = (
            as_t(world.floor_pool),
            as_t(world.wall_pool),
            as_t(world.box_pool),
        )

    def wall_hit(self, x, z, r):
        """``intersect_circle_segs`` against the four walls, elementwise."""
        S, hit = self.S, None
        for ax, az, bx, bz in _SEGMENTS:
            ax, az, bx, bz = ax * S, az * S, bx * S, bz * S
            abx, abz = bx - ax, bz - az
            proj = ((x - ax) * abx + (z - az) * abz) / (abx * abx + abz * abz)
            proj = proj.clamp(0.0, 1.0)
            dx = ax + proj * abx - x
            dz = az + proj * abz - z
            h = torch.sqrt(dx * dx + dz * dz) < r
            hit = h if hit is None else hit | h
        return hit

    @staticmethod
    def near(x, z, r, ox, oz, orad):
        dx, dz = ox - x, oz - z
        return torch.sqrt(dx * dx + dz * dz) < r + orad

    def block_hits(self, x, z, r, s, count, exclude=None):
        """``(..., M)`` hits of points ``(B, K)`` against the present blocks, ``exclude`` ``(B, K)`` left out."""
        M = s["bpos"].shape[1]
        bx, bz = s["bpos"][:, None, :, 0], s["bpos"][:, None, :, 2]
        hits = self.near(x[..., None], z[..., None], r[..., None], bx, bz, s["rad"][:, None, :])
        j = torch.arange(M, device=x.device)
        hits = hits & (j < count[:, None, None])
        if exclude is not None:
            hits = hits & (j != exclude[..., None])
        return hits

    def agent_blocked(self, x, z, s, count, r=AGENT_RADIUS, walls=True):
        """Whether the agent at ``(B, K)`` points intersects a wall or (without agent overlap) a block."""
        r = torch.full_like(x, r)
        hit = self.wall_hit(x, z, r) if walls else torch.zeros_like(x, dtype=torch.bool)
        if not self.agent_overlap:
            hit = hit | self.block_hits(x, z, r, s, count).any(-1)
        return hit

    def block_blocked(self, x, z, r, who, s, count, walls=True):
        """Whether block ``who`` ``(B, K)`` at ``(B, K)`` points intersects a wall, the agent or another block."""
        hit = self.wall_hit(x, z, r) if walls else torch.zeros_like(x, dtype=torch.bool)
        if not self.agent_overlap:
            ax, az = s["apos"][:, None, 0], s["apos"][:, None, 2]
            hit = hit | self.near(x, z, r, ax, az, AGENT_RADIUS)
        if not self.box_overlap:
            hit = hit | self.block_hits(x, z, r, s, count, exclude=who).any(-1)
        return hit

    def carry_pos(self, ax, az, d, rad, height):
        dist = AGENT_RADIUS + rad + self.P[P_MAX_FWD]
        x = ax + (torch.cos(d) * 1.05) * dist
        z = az + (-torch.sin(d) * 1.05) * dist
        y = (CAM_HEIGHT - height - 0.3).clamp(min=0.0)
        return x, y, z

    def snap(self, x, z, blocked):
        """``_snap_entity_to_grid`` of ``(B,)`` points: ``blocked(x, z)`` takes ``(B, 10)`` candidates."""
        sx, sz = torch.round(x), torch.round(z)
        cx, cz = sx[:, None] + self._dx, sz[:, None] + self._dz
        free = ~blocked(cx, cz)
        first = torch.argmax(free.to(torch.int8), dim=1, keepdim=True)
        ok = free.any(1)
        return (
            torch.where(ok, cx.gather(1, first)[:, 0], x),
            torch.where(ok, cz.gather(1, first)[:, 0], z),
        )

    def step(self, s: dict, a: torch.Tensor, active: torch.Tensor | None) -> dict:
        """One step of every environment (``active`` ones only, when given); returns the new State dict."""
        P, B = self.P, a.shape[0]
        M = s["bpos"].shape[1]
        count = s["count"]
        apos, adir, carry = s["apos"].clone(), s["adir"].clone(), s["carry"].clone()
        bpos, bdir, vel = s["bpos"].clone(), s["bdir"].clone(), s["vel"].clone()
        steps = s["steps"] + 1
        j = torch.arange(M, device=a.device)
        has = carry >= 0
        ci = carry.clamp(min=0)
        rc = s["rad"].gather(1, ci[:, None])[:, 0]
        hc = s["height"].gather(1, ci[:, None])[:, 0]
        onehot = (j == ci[:, None]) & has[:, None]
        ax, ay, az, d = apos[:, 0], apos[:, 1], apos[:, 2], adir
        cur = {"apos": apos, "bpos": bpos, "rad": s["rad"]}

        def set_carried(mask, x, y, z):
            m = (onehot & mask[:, None])[..., None]
            return torch.where(m, torch.stack([x, y, z], -1)[:, None, :], bpos)

        # move forward or back
        f = torch.where(a == 2, 1.0, -1.0).to(torch.float64) * P[P_FWD]
        nx = ax + torch.cos(d) * f + torch.sin(d) * 0.0
        nz = az + (-torch.sin(d)) * f + torch.cos(d) * 0.0
        free = ~self.agent_blocked(nx[:, None], nz[:, None], cur, count)[:, 0]
        cx, cy, cz = self.carry_pos(nx, nz, d, rc, hc)
        cfree = ~self.block_blocked(cx[:, None], cz[:, None], rc[:, None], ci[:, None], cur, count)[:, 0]
        moved = ((a == 2) | (a == 3)) & free & (~has | cfree)
        bpos = set_carried(moved, cx, ay + cy, cz)
        apos = torch.where(moved[:, None], torch.stack([nx, ay, nz], -1), apos)
        # turn
        turn = torch.where(a == 0, 1.0, -1.0).to(torch.float64) * P[P_TURN]
        nd = d + turn * (math.pi / 180)
        tx, ty, tz = self.carry_pos(ax, az, nd, rc, hc)
        cur = {"apos": apos, "bpos": bpos, "rad": s["rad"]}
        tfree = ~self.block_blocked(tx[:, None], tz[:, None], rc[:, None], ci[:, None], cur, count)[:, 0]
        turned = ((a == 0) | (a == 1)) & (~has | tfree)
        bpos = set_carried(turned, tx, ay + ty, tz)
        bdir = torch.where(onehot & turned[:, None], nd[:, None], bdir)
        adir = torch.where(turned, nd, adir)
        # pick up: the first block hit, unless a wall is hit first
        px = ax + (torch.cos(d) * 1.5) * AGENT_RADIUS
        pz = az + (-torch.sin(d) * 1.5) * AGENT_RADIUS
        r = torch.full_like(px, 1.2 * AGENT_RADIUS)
        wall = self.wall_hit(px, pz, r)
        if self.agent_overlap:
            pick = torch.zeros_like(has)
            first = ci
        else:
            hits = self.block_hits(px[:, None], pz[:, None], r[:, None], cur, count)[:, 0]
            first = torch.argmax(hits.to(torch.int8), dim=1)
            pick = (a == 5) & ~has & ~wall & hits.any(1)
        # drop
        drop = (a == 6) & has
        bpos = torch.where((onehot & drop[:, None])[..., None] & (torch.arange(3, device=a.device) == 1), 0.0, bpos)
        carry = torch.where(pick, first, torch.where(drop, -1, carry))
        # the carried block follows the agent
        has = carry >= 0
        ci = carry.clamp(min=0)
        rc = s["rad"].gather(1, ci[:, None])[:, 0]
        hc = s["height"].gather(1, ci[:, None])[:, 0]
        onehot = (j == ci[:, None]) & has[:, None]
        x, y, z = self.carry_pos(apos[:, 0], apos[:, 2], adir, rc, hc)
        bpos = set_carried(has, x, apos[:, 1] + y, z)
        bdir = torch.where(onehot, adir[:, None], bdir)
        truncated = steps >= P[P_MAX_STEPS]
        go = ~truncated
        if self.grid:
            cur = {"apos": apos, "bpos": bpos, "rad": s["rad"]}
            sx, sz = self.snap(apos[:, 0], apos[:, 2], lambda x, z: self.agent_blocked(x, z, cur, count))
            apos = torch.where(go[:, None], torch.stack([sx, apos[:, 1], sz], -1), apos)
            q = math.pi / 2
            adir = torch.where(go, torch.round(adir / q) * q, adir)
            bdir = torch.where(onehot & go[:, None], adir[:, None], bdir)
            cur = {"apos": apos, "bpos": bpos, "rad": s["rad"]}
            cpos = bpos.gather(1, ci[:, None, None].expand(B, 1, 3))[:, 0]
            sx, sz = self.snap(
                cpos[:, 0],
                cpos[:, 2],
                lambda x, z: self.block_blocked(x, z, rc[:, None].expand_as(x), ci[:, None].expand_as(x), cur, count),
            )
            bpos = set_carried(go & has, sx, cpos[:, 1], sz)
        if not self.static:
            moving = go[:, None] & (j < count[:, None]) & ~onehot
            if self.box_overlap:
                bpos, vel = self._move_blocks(apos, bpos, vel, s["rad"], j[None, :].expand(B, M), moving, count, s)
            else:
                for m in range(M):
                    sl = slice(m, m + 1)
                    pm, vm = self._move_blocks(
                        apos, bpos, vel, s["rad"], torch.full((B, 1), m, device=a.device), moving[:, sl], count, s, sl
                    )
                    bpos = torch.cat([bpos[:, :m], pm, bpos[:, m + 1 :]], 1)
                    vel = torch.cat([vel[:, :m], vm, vel[:, m + 1 :]], 1)
        if P[P_WRAP_HEADING] > 0:
            adir = torch.remainder(adir, 2.0 * math.pi)
        out = {"apos": apos, "adir": adir, "carry": carry, "bpos": bpos, "bdir": bdir, "vel": vel, "steps": steps}
        if active is not None:
            out = {k: torch.where(active.view(-1, *[1] * (v.dim() - 1)), v, s[k]) for k, v in out.items()}
            truncated = truncated & active
        out["truncated"] = truncated
        return out

    def _move_blocks(self, apos, bpos, vel, rad, who, moving, count, s, sl=slice(None)):
        """Move blocks ``sl`` (``who`` their indices) by their velocity, bouncing off what they would hit."""
        S, torus = self.S, self.torus
        pos, v3, r = bpos[:, sl].clone(), vel[:, sl].clone(), rad[:, sl]
        for axis in (0, 2):
            cur = {
                "apos": apos,
                "bpos": torch.cat([bpos[:, : sl.start], pos, bpos[:, sl.stop :]], 1) if sl.start is not None else pos,
                "rad": rad,
            }
            v, c = v3[..., axis], pos[..., axis]
            cand = c + v
            other = pos[..., 2 - axis]

            def blocked(p):
                x, z = (p, other) if axis == 0 else (other, p)
                return self.block_blocked(x, z, r, who, cur, count, walls=not torus)

            go = moving & (v != 0.0)
            free = ~blocked(cand)
            back = c - v
            free_back = ~blocked(back)
            new = torch.where(free, cand, torch.where(free_back, back, c))
            nv = torch.where(free, v, -v)
            if torus:
                wrap = (cand + r > S) | (cand - r < 0.0)
                wrapped = torch.where(cand + r > S, torch.maximum(cand - S, r), torch.minimum(cand + S, S - r))
                new = torch.where(wrap, wrapped, new)
                nv = torch.where(wrap, v, nv)
            new = torch.where(go, new, c)
            if self.grid:
                rounded = torch.round(new)
                if torus:
                    rounded = rounded.clamp(1.0, S - 1.0)
                new = torch.where(moving, rounded, new)
            pos = pos.clone()
            pos[..., axis] = new
            v3 = v3.clone()
            v3[..., axis] = torch.where(go, nv, v)
        return pos, v3

    def reset(self, s: dict) -> dict:
        """Every environment's next episode from its ``seed`` and ``episode``; returns the new State dict."""
        P, S, dev = self.P, self.S, self.device
        B, M = s["kind"].shape
        base = R.stream_t(s["seed"], s["episode"])
        u = lambda k: R.uniform_t(base, k)  # noqa: E731
        uk = lambda k: R.uniform_t(base[:, None], k)  # noqa: E731
        lo, hi = self.world.count_range
        n = lo + (u(R.COUNT) * (hi - lo + 1)).long()
        j = torch.arange(M, device=dev)
        room = torch.empty((B, 3), dtype=torch.int64, device=dev)
        room[:, 0] = self.floor_pool[(u(R.FLOOR) * 3).long()] if P[P_FLOOR_TEX_RANDOM] > 0 else int(P[P_FLOOR_TEX])
        room[:, 1] = self.wall_pool[(u(R.WALL) * 3).long()] if P[P_WALL_TEX_RANDOM] > 0 else int(P[P_WALL_TEX])
        room[:, 2] = int(P[P_CEIL_TEX])
        pool, npool = self.pool, len(self.pool)
        with_replacement = pool[(uk(R.COLOR + j) * npool).long()]
        if P[P_ENSURE_BASE] > 0:
            nb = len(self.base_palette)
            order = torch.argsort(uk(R.PERM + torch.arange(nb, device=dev)), dim=1)
            perm = self.base_palette[order]
            color = with_replacement.clone()
            color[:, : min(nb, M)] = perm[:, : min(nb, M)]
        elif not (P[P_REPEAT] > 0):
            order = torch.argsort(uk(R.PERM + torch.arange(npool, device=dev)), dim=1)
            perm = pool[order]
            color = with_replacement.clone()
            k = min(npool, M)
            color[:, :k] = torch.where((n <= npool)[:, None], perm[:, :k], with_replacement[:, :k])
        else:
            color = with_replacement
        present = j < n[:, None]
        # sizes and kinds
        b = R.BLOCK + R.BLOCK_STRIDE * j
        texture = (
            self.box_pool[(uk(b + R.B_TEX) * 3).long()]
            if P[P_BOX_TEX_RANDOM] > 0
            else torch.full((B, M), int(P[P_BOX_TEX]), dtype=torch.int64, device=dev)
        )
        ball = (
            (uk(b + R.B_BALL) * 3).long() == 1
            if P[P_BOX_AND_BALL] > 0
            else torch.zeros((B, M), dtype=torch.bool, device=dev)
        )
        if P[P_BLOCK_XY] >= 0 or P[P_BLOCK_H] >= 0:
            sx_ = P[P_BLOCK_XY] if P[P_BLOCK_XY] >= 0 else 0.75
            sy_ = P[P_BLOCK_H] if P[P_BLOCK_H] >= 0 else sx_
            sx = torch.full((B, M), sx_, dtype=torch.float64, device=dev)
            sy = torch.full((B, M), sy_, dtype=torch.float64, device=dev)
        else:
            sx = 0.6 + 0.25 * uk(b + R.B_SIZE)
            sy = sx
        sx = torch.where(ball, 0.85, sx)
        sy = torch.where(ball, 0.85, sy)
        kind = ball.long()
        texture = torch.where(ball, -1, texture)
        rad = torch.where(ball, P[P_BALL_RADIUS] * 0.85, torch.sqrt(sx * sx + sx * sx) / 2)
        height = sy
        size = torch.stack([sx, sy, sx], -1)
        # padding
        kind, color, texture = kind.where(present, 0), color.where(present, 0), texture.where(present, -1)
        size = torch.where(present[..., None], size, 1.0)
        rad = torch.where(present, rad, math.sqrt(2.0) / 2)
        height = torch.where(present, height, 1.0)
        zero = torch.zeros(B, dtype=torch.float64, device=dev)
        apos = torch.zeros((B, 3), dtype=torch.float64, device=dev)
        bpos = torch.zeros((B, M, 3), dtype=torch.float64, device=dev)
        cur = {"apos": apos, "bpos": bpos, "rad": rad}
        none = torch.zeros(B, dtype=torch.int64, device=dev)
        # the agent
        if P[P_AGENT_CENTER] > 0:
            ok, mnx, mxx, mnz, mxz = self._extents(AGENT_RADIUS)
            if not ok:
                mnx, mxx, mnz, mxz = 0.0, S, 0.0, S
            cx, cz = (mnx + mxx) / 2.0, (mnz + mxz) / 2.0
            if self.grid:
                cx = round(cx) - 1.0 if abs(cx - round(cx)) < 1e-6 else math.floor(cx)
                cz = round(cz) - 1.0 if abs(cz - round(cz)) < 1e-6 else math.floor(cz)
            else:
                cx = cx - 0.5 if abs(cx - round(cx)) < 1e-6 else cx
                cz = cz - 0.5 if abs(cz - round(cz)) < 1e-6 else cz
            apos[:, 0], apos[:, 2] = min(max(cx, mnx), mxx), min(max(cz, mnz), mxz)
        else:
            x, z = self._place(
                base,
                R.AGENT,
                torch.full((B,), AGENT_RADIUS, dtype=torch.float64, device=dev),
                lambda x, z: self.agent_blocked(x, z, cur, none),
            )
            apos = torch.stack([x, zero, z], -1)
        adir = -math.pi + 2 * math.pi * u(R.AGENT_DIR)
        # the blocks, one after another
        for m in range(M):
            cur = {"apos": apos, "bpos": bpos, "rad": rad}
            placed = torch.full((B,), m, dtype=torch.int64, device=dev)
            r = rad[:, m]
            who = torch.full((B, 1), m, device=dev)
            x, z = self._place(
                base,
                R.BLOCK + R.BLOCK_STRIDE * m + R.B_PLACE,
                r,
                lambda x, z, r=r, who=who, cur=cur, placed=placed: self.block_blocked(
                    x, z, r[:, None].expand_as(x), who.expand_as(x), cur, placed
                ),
            )
            p = torch.stack([x, zero, z], -1).where(present[:, m, None], 0.0)
            bpos = torch.cat([bpos[:, :m], p[:, None], bpos[:, m + 1 :]], 1)
        if P[P_RANDOM_ORIENT] > 0:
            bdir = (-math.pi + 2 * math.pi * uk(b + R.B_DIR)).where(present, 0.0)
        else:
            bdir = torch.zeros((B, M), dtype=torch.float64, device=dev)
        if self.grid:
            cur = {"apos": apos, "bpos": bpos, "rad": rad}
            sx_, sz_ = self.snap(apos[:, 0], apos[:, 2], lambda x, z: self.agent_blocked(x, z, cur, n))
            apos = torch.stack([sx_, apos[:, 1], sz_], -1)
            for m in range(M):
                cur = {"apos": apos, "bpos": bpos, "rad": rad}
                r, who = rad[:, m : m + 1], torch.full((B, 1), m, device=dev)
                sx_, sz_ = self.snap(
                    bpos[:, m, 0],
                    bpos[:, m, 2],
                    lambda x, z, r=r, who=who, cur=cur: self.block_blocked(
                        x, z, r.expand_as(x), who.expand_as(x), cur, n
                    ),
                )
                p = torch.stack([sx_, bpos[:, m, 1], sz_], -1).where(present[:, m, None], bpos[:, m])
                bpos = torch.cat([bpos[:, :m], p[:, None], bpos[:, m + 1 :]], 1)
        vel = self._velocities(uk, b, B, M)
        vel = torch.where(present[..., None], vel, 0.0)
        if P[P_HEADING_ZERO] > 0:
            adir = torch.zeros_like(adir)
        if P[P_WRAP_HEADING] > 0:
            adir = torch.remainder(adir, 2.0 * math.pi)
        return {
            "room": room,
            "kind": kind,
            "color": color,
            "texture": texture,
            "size": size,
            "count": n,
            "rad": rad,
            "height": height,
            "apos": apos,
            "adir": adir,
            "carry": torch.full_like(n, -1),
            "bpos": bpos,
            "bdir": bdir,
            "vel": vel,
            "steps": torch.zeros_like(n),
        }

    def _extents(self, r):
        P, S = self.P, self.S
        if P[P_SPAWN_BUFFER] < 0:
            return False, 0.0, S, 0.0, S
        buf = P[P_SPAWN_BUFFER] + r
        lo, hi = buf, S + buf - 1.0
        if hi <= lo:
            return False, 0.0, S, 0.0, S
        return True, lo, hi, lo, hi

    def _place(self, base, slot, r, blocked):
        """``place_entity`` by rejection over ``CANDIDATES`` draws; ``r`` is ``(B,)``."""
        S, P, dev = self.S, self.P, self.device
        k = torch.arange(CANDIDATES, device=dev)
        if P[P_SPAWN_BUFFER] < 0:
            lx = lz = torch.zeros_like(r)
            hx = hz = torch.full_like(r, S)
        else:
            buf = P[P_SPAWN_BUFFER] + r
            ok = (S + buf - 1.0) > buf
            lx = lz = torch.where(ok, buf, 0.0)
            hx = hz = torch.where(ok, S + buf - 1.0, S)
        rr = r[:, None]
        x = (lx[:, None] - rr) + ((hx[:, None] + rr) - (lx[:, None] - rr)) * R.uniform_t(base[:, None], slot + 2 * k)
        z = (lz[:, None] - rr) + ((hz[:, None] + rr) - (lz[:, None] - rr)) * R.uniform_t(
            base[:, None], slot + 2 * k + 1
        )
        good = (x > 0.0) & (x < S) & (z > 0.0) & (z < S) & ~blocked(x, z)
        first = torch.argmax(good.to(torch.int8), dim=1, keepdim=True)
        ok = good.any(1)
        lx_, lz_ = torch.minimum(torch.maximum(x[:, -1], r), S - r), torch.minimum(torch.maximum(z[:, -1], r), S - r)
        return torch.where(ok, x.gather(1, first)[:, 0], lx_), torch.where(ok, z.gather(1, first)[:, 0], lz_)

    def _velocities(self, uk, b, B, M):
        P, dev = self.P, self.device
        zero = torch.zeros((B, M), dtype=torch.float64, device=dev)
        if self.static:
            vx, vz = zero, zero
        elif self.grid and P[P_CARDINAL] > 0:
            speed = max(1.0, max(abs(P[P_VMIN]), abs(P[P_VMAX])))
            choice = (uk(b + R.B_VEL) * 4).long()
            vx = torch.where(choice == 0, speed, torch.where(choice == 1, -speed, 0.0)).to(torch.float64)
            vz = torch.where(choice == 2, speed, torch.where(choice == 3, -speed, 0.0)).to(torch.float64)
        elif self.grid:
            span = P[P_VMAX] - P[P_VMIN] + 1.0
            t = torch.arange(GRID_TRIES, device=dev)
            slots = b[:, None] + R.B_GRID + 2 * t
            tx = P[P_VMIN] + torch.floor(uk(slots.reshape(-1)).reshape(B, M, GRID_TRIES) * span)
            tz = P[P_VMIN] + torch.floor(uk((slots + 1).reshape(-1)).reshape(B, M, GRID_TRIES) * span)
            good = (tx != 0.0) | (tz != 0.0)
            first = torch.argmax(good.to(torch.int8), dim=2, keepdim=True)
            ok = good.any(2)
            vx = torch.where(ok, tx.gather(2, first)[..., 0], P[P_VMAX])
            vz = torch.where(ok, tz.gather(2, first)[..., 0], 0.0)
        else:
            theta = -math.pi + 2 * math.pi * uk(b + R.B_THETA)
            step = P[P_FWD] * P[P_SPEED_SCALE]
            vx, vz = step * torch.cos(theta), step * torch.sin(theta)
        return torch.stack([vx, zero, vz], -1)
