"""Arbitrary-ray CUDA queries for local rigid triangle BVHs.

Mesh hierarchies stay resident; only object transforms change between frames.
Float64 intersections and original face order define deterministic hit ties.
This module does not create a Blender, MuJoCo or graphics context.
"""

import numpy as np
import torch
import triton as tr
import triton.language as tl

from .optics import hierarchy


@tr.jit
def _component(P, i, k: tl.constexpr, stride: tl.constexpr, valid):
    return tl.load(P + i * stride + k, valid, 0).to(tl.float64)


@tr.jit
def _vector(P, i, stride: tl.constexpr, valid):
    return _component(P, i, 0, stride, valid), _component(P, i, 1, stride, valid), _component(P, i, 2, stride, valid)


@tr.jit(do_not_specialize=["Count"])
def _query(
    O,
    D,
    Bounds,
    Nodes,
    Order,
    Triangles,
    Roots,
    Ends,
    Position,
    Inverse,
    Selected,
    Planes,
    Semantics,
    Limit,
    Distance,
    Face,
    Normal,
    Count,
    G: tl.constexpr,
    O_ROW: tl.constexpr,
    O_COL: tl.constexpr,
    D_ROW: tl.constexpr,
    D_COL: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = q < Count
    ox = tl.load(O + q * O_ROW, valid, 0).to(tl.float64)
    oy = tl.load(O + q * O_ROW + O_COL, valid, 0).to(tl.float64)
    oz = tl.load(O + q * O_ROW + 2 * O_COL, valid, 0).to(tl.float64)
    dx = tl.load(D + q * D_ROW, valid, 0).to(tl.float64)
    dy = tl.load(D + q * D_ROW + D_COL, valid, 0).to(tl.float64)
    dz = tl.load(D + q * D_ROW + 2 * D_COL, valid, 0).to(tl.float64)
    best = (
        tl.load(Limit + q, valid, float("inf")).to(tl.float64)
        if HAS_LIMIT
        else tl.full((BLOCK,), float("inf"), tl.float64)
    )
    winner = tl.full((BLOCK,), -1, tl.int32)
    semantic = tl.full((BLOCK,), 32768, tl.int32)
    wx, wy, wz = (
        tl.full((BLOCK,), 0.0, tl.float64),
        tl.full((BLOCK,), 0.0, tl.float64),
        tl.full((BLOCK,), 0.0, tl.float64),
    )
    for g in range(G):
        if tl.load(Selected + g):
            px, py, pz = _vector(Position, g, 3, True)
            x, y, z = ox - px, oy - py, oz - pz
            a, b, c = _vector(Inverse, g, 9, True)
            e, f, h = _vector(Inverse + 3, g, 9, True)
            j, k, l = _vector(Inverse + 6, g, 9, True)
            lx, ly, lz = a * x + b * y + c * z, e * x + f * y + h * z, j * x + k * y + l * z
            vx, vy, vz = a * dx + b * dy + c * dz, e * dx + f * dy + h * dz, j * dx + k * dy + l * dz
            root, end = tl.load(Roots + g), tl.load(Ends + g)
            node = tl.full((BLOCK,), 0, tl.int32) + root
            while tl.sum((valid & (node < end)).to(tl.int32), 0) > 0:
                live = valid & (node < end)
                lowx, lowy, lowz = _vector(Bounds, node, 6, live)
                highx, highy, highz = _vector(Bounds + 3, node, 6, live)
                tx0, tx1 = (lowx - lx) / vx, (highx - lx) / vx
                ty0, ty1 = (lowy - ly) / vy, (highy - ly) / vy
                tz0, tz1 = (lowz - lz) / vz, (highz - lz) / vz
                nx = tl.where(
                    vx == 0, tl.where((lx >= lowx) & (lx <= highx), -float("inf"), float("inf")), tl.minimum(tx0, tx1)
                )
                fx = tl.where(vx == 0, float("inf"), tl.maximum(tx0, tx1))
                ny = tl.where(
                    vy == 0, tl.where((ly >= lowy) & (ly <= highy), -float("inf"), float("inf")), tl.minimum(ty0, ty1)
                )
                fy = tl.where(vy == 0, float("inf"), tl.maximum(ty0, ty1))
                nz = tl.where(
                    vz == 0, tl.where((lz >= lowz) & (lz <= highz), -float("inf"), float("inf")), tl.minimum(tz0, tz1)
                )
                fz = tl.where(vz == 0, float("inf"), tl.maximum(tz0, tz1))
                hitbox = live & (
                    tl.maximum(tl.maximum(nx, ny), tl.maximum(nz, 1e-9))
                    <= tl.minimum(tl.minimum(fx, fy), tl.minimum(fz, best + 1e-12))
                )
                escape = tl.load(Nodes + 3 * node, live, end)
                offset = tl.load(Nodes + 3 * node + 1, live, 0)
                size = tl.load(Nodes + 3 * node + 2, live, 0)
                for leaf in range(4):
                    test = hitbox & (leaf < size)
                    face = tl.load(Order + offset + leaf, test, 0)
                    ax, ay, az = _vector(Triangles, face, 9, test)
                    bx, by, bz = _vector(Triangles + 3, face, 9, test)
                    cx, cy, cz = _vector(Triangles + 6, face, 9, test)
                    ux, uy, uz = bx - ax, by - ay, bz - az
                    ex, ey, ez = cx - ax, cy - ay, cz - az
                    hx, hy, hz = vy * ez - vz * ey, vz * ex - vx * ez, vx * ey - vy * ex
                    det = ux * hx + uy * hy + uz * hz
                    sx, sy, sz = lx - ax, ly - ay, lz - az
                    u = (sx * hx + sy * hy + sz * hz) / det
                    qx, qy, qz = sy * uz - sz * uy, sz * ux - sx * uz, sx * uy - sy * ux
                    v = (vx * qx + vy * qy + vz * qz) / det
                    pnx, pny, pnz = _vector(Planes, face, 4, test)
                    offset_plane = tl.load(Planes + 4 * face + 3, test, 0).to(tl.float64)
                    t = -(pnx * ox + pny * oy + pnz * oz + offset_plane) / (pnx * dx + pny * dy + pnz * dz)
                    label = tl.load(Semantics + face, test, 32768)
                    take = (
                        test
                        & (det != 0)
                        & (u >= 0)
                        & (v >= 0)
                        & (u + v <= 1)
                        & (t > 1e-9)
                        & ((t < best) | ((t == best) & ((label < semantic) | ((label == semantic) & (face < winner)))))
                    )
                    fnx, fny, fnz = uy * ez - uz * ey, uz * ex - ux * ez, ux * ey - uy * ex
                    wx = tl.where(take, a * fnx + e * fny + j * fnz, wx)
                    wy = tl.where(take, b * fnx + f * fny + k * fnz, wy)
                    wz = tl.where(take, c * fnx + h * fny + l * fnz, wz)
                    best, winner = tl.where(take, t, best), tl.where(take, face, winner)
                    semantic = tl.where(take, label, semantic)
                node = tl.where(hitbox & (size == 0), node + 1, escape)
    length = tl.sqrt(wx * wx + wy * wy + wz * wz)
    tl.store(Distance + q, tl.where(winner >= 0, best, float("inf")), valid)
    tl.store(Face + q, winner, valid)
    tl.store(Normal + 3 * q, wx / length, valid)
    tl.store(Normal + 3 * q + 1, wy / length, valid)
    tl.store(Normal + 3 * q + 2, wz / length, valid)


@tr.jit(do_not_specialize=["Count"])
def _normals(Triangles, Labels, Inverse, Face, Normal, Count, BLOCK: tl.constexpr):
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = q < Count
    face = tl.load(Face + q, valid, -1)
    hit = valid & (face >= 0)
    g = tl.load(Labels + face, hit, 0)
    ax, ay, az = _vector(Triangles, face, 9, hit)
    bx, by, bz = _vector(Triangles + 3, face, 9, hit)
    cx, cy, cz = _vector(Triangles + 6, face, 9, hit)
    ux, uy, uz = bx - ax, by - ay, bz - az
    ex, ey, ez = cx - ax, cy - ay, cz - az
    fnx, fny, fnz = uy * ez - uz * ey, uz * ex - ux * ez, ux * ey - uy * ex
    a, b, c = _vector(Inverse, g, 9, hit)
    e, f, h = _vector(Inverse + 3, g, 9, hit)
    j, k, l = _vector(Inverse + 6, g, 9, hit)
    # Match _query's exact inverse-transpose and normalization order, including
    # NaN normals for misses. No new intersection or persistent normal buffer.
    wx = a * fnx + e * fny + j * fnz
    wy = b * fnx + f * fny + k * fnz
    wz = c * fnx + h * fny + l * fnz
    length = tl.sqrt(wx * wx + wy * wy + wz * wz)
    tl.store(Normal + 3 * q, wx / length, valid)
    tl.store(Normal + 3 * q + 1, wy / length, valid)
    tl.store(Normal + 3 * q + 2, wz / length, valid)


class TriangleQuery:
    """One local BVH per object, supporting rigid and positive-scale transforms."""

    def __init__(self, triangles, *, device="cuda", semantic_ids=None):
        bounds, nodes, order, roots, ends, faces, labels = [], [], [], [], [], [], []
        node_offset = face_offset = order_offset = 0
        for g, mesh in enumerate(triangles):
            b, n, o = hierarchy(mesh)
            roots.append(node_offset)
            n[:, 0] += node_offset
            n[:, 1] += order_offset
            bounds.append(b)
            nodes.append(n)
            order.append(o + face_offset)
            faces.append(mesh)
            labels.extend([g] * len(mesh))
            node_offset += len(n)
            face_offset += len(mesh)
            order_offset += len(o)
            ends.append(node_offset)
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

        def upload(a, dtype):
            return torch.as_tensor(np.asarray(a), dtype=dtype, device=self.device).contiguous()

        self.bounds = upload(np.concatenate(bounds), torch.float64)
        self.nodes = upload(np.concatenate(nodes), torch.int32)
        self.order = upload(np.concatenate(order), torch.int32)
        self.triangles = upload(np.concatenate(faces), torch.float64)
        self.roots = upload(roots, torch.int32)
        self.ends = upload(ends, torch.int32)
        self.labels = upload(labels, torch.int64)
        self.count = len(triangles)
        if semantic_ids is None:
            semantic_ids = np.arange(1, self.count + 1)
        semantic_ids = upload(semantic_ids, torch.int32)
        self.semantics = semantic_ids[self.labels].contiguous()

    def __call__(self, origins, directions, positions, inverse, selected, planes, *, limit=None):
        """Return distance, face and normal; ray rows may be broadcast/strided."""
        n = len(origins)
        if limit is not None and (
            limit.shape != (n,)
            or limit.dtype != torch.float64
            or limit.device != self.device
            or not limit.is_contiguous()
        ):
            raise ValueError("ray limits must be contiguous CUDA float64 [N]")
        distance = torch.empty(n, dtype=torch.float64, device=self.device)
        face = torch.empty(n, dtype=torch.int64, device=self.device)
        normal = origins.new_empty((n, 3))
        if n:
            _query[(tr.cdiv(n, 64),)](
                origins,
                directions,
                self.bounds,
                self.nodes,
                self.order,
                self.triangles,
                self.roots,
                self.ends,
                positions,
                inverse,
                selected,
                planes,
                self.semantics,
                limit if limit is not None else distance,
                distance,
                face,
                normal,
                n,
                self.count,
                *origins.stride(),
                *directions.stride(),
                limit is not None,
                64,
                enable_fp_fusion=False,
            )
        return distance, face, normal

    def normals(self, faces, inverse):
        """Reconstruct queried geometric normals for the diagnostic cache path."""
        normal = torch.empty((len(faces), 3), dtype=torch.float64, device=self.device)
        if len(faces):
            _normals[(tr.cdiv(len(faces), 128),)](
                self.triangles,
                self.labels,
                inverse,
                faces,
                normal,
                len(faces),
                128,
                enable_fp_fusion=False,
            )
        return normal
