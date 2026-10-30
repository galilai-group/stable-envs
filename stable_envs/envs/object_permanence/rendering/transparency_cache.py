"""Exact static event lists: retain depths/IDs, then merge moving surfaces."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _merge(
    StaticDepth,
    StaticWinner,
    DynamicDepth,
    DynamicWinner,
    Alpha,
    OutDepth,
    OutWinner,
    P: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    L: tl.constexpr,
    F: tl.constexpr,
    B: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pixel = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    world = tl.program_id(1)
    valid = pixel < P
    si = tl.full((BLOCK,), 0, tl.int32)
    di = tl.full((BLOCK,), 0, tl.int32)
    live = valid
    for layer in range(L):
        sd = tl.load(StaticDepth + si * P + pixel, valid & (si < S), other=9223372036854775807)
        sw = tl.load(StaticWinner + si * P + pixel, valid & (si < S), other=9223372036854775807)
        dd = tl.load(DynamicDepth + (di * B + world) * P + pixel, valid & (di < D), other=9223372036854775807)
        dw = tl.load(DynamicWinner + (di * B + world) * P + pixel, valid & (di < D), other=9223372036854775807)
        use_static = (sd < dd) | ((sd == dd) & (sw < dw))
        depth = tl.where(use_static, sd, dd)
        winner = tl.where(use_static, sw, dw)
        hit = live & (winner != 9223372036854775807)
        offset = (layer * B + world) * P + pixel
        tl.store(OutDepth + offset, tl.where(hit, depth, 9223372036854775807), valid)
        tl.store(OutWinner + offset, tl.where(hit, winner, 9223372036854775807), valid)
        alpha = tl.load(Alpha + winner % (F + 1), hit, other=1)
        live = hit & (alpha < 1)
        si += use_static.to(tl.int32)
        di += (~use_static).to(tl.int32)


class StaticAlphaCoverage:
    """Cache only exact static visibility; geometry and alpha invalidate it."""

    def __init__(self, faces, static_faces):
        self.faces = faces
        self._face_version = faces._version
        self.static = static_faces.nonzero().flatten().to(torch.int32).contiguous()
        self.dynamic = (~static_faces).nonzero().flatten().to(torch.int32).contiguous()
        self.vertices = faces[self.static.long()].flatten().unique().long()
        self.builds = self.fallbacks = 0
        self._snapshot = self._key = self._depth = self._winner = None

    def seed(self, positions, vertices, planes, rays, origin, labels, alpha, resolution, near, far, capacity, draw):
        if self.faces._version != self._face_version:
            raise ValueError("alpha cache requires immutable topology")
        if not len(self.static):
            return None
        values = (positions[:, self.vertices], vertices[:, self.vertices], planes[:, self.static.long()])
        if any(not torch.equal(value, value[:1].expand_as(value)) for value in values):
            self.fallbacks += 1
            return None
        current = (*(value[:1] for value in values), rays, origin, labels, alpha)
        key = (tuple(resolution), near, far, capacity)
        if (
            self._key != key
            or self._snapshot is None
            or any(
                a.dtype != b.dtype or a.device != b.device or not torch.equal(a, b)
                for a, b in zip(current, self._snapshot, strict=True)
            )
        ):
            depth, winner = draw(positions[:1], self.static)
            # Own only actual layers, not a view retaining the capacity buffer.
            self._depth, self._winner = depth.clone(), winner.clone()
            self._snapshot = tuple(value.clone() for value in current)
            self._key = key
            self.builds += 1
        return self._depth, self._winner, self.dynamic

    def clear(self):
        self.faces = self.static = self.dynamic = self.vertices = None
        self._snapshot = self._key = self._depth = self._winner = None


def merge(static, dynamic, alpha, capacity, face_count):
    sd, sw = static
    dd, dw = dynamic
    _, b, h, w = dd.shape
    # One extra event detects overflow after merging; opaque termination may
    # remove excess farther events. No capacity-dependent image approximation.
    count = min(capacity + 1, len(sd) + len(dd))
    depth = torch.empty((count, b, h, w), dtype=torch.int64, device=dd.device)
    winner = torch.empty_like(depth)
    _merge[(tr.cdiv(h * w, 128), b)](
        sd, sw, dd, dw, alpha, depth, winner, h * w, len(sd), len(dd), count, face_count, b, 128
    )
    if count > capacity and (winner[-1] != torch.iinfo(torch.int64).max).any():
        raise ValueError("transparent layer capacity exceeded while merging static and dynamic events")
    return depth[:capacity], winner[:capacity]
