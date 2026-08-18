"""A batched z-buffer triangle rasterizer in Triton, and what it costs on these two families.

``probe_rays.py`` measured the shape of the problem before a line of this was written, which is what
``docs/concepts/bit-exact-triton.md`` rule 5 asks for -- check on the workload, not on its smallest
instance:

======================================  ==========  ============  ==============  ================
scene (256 px = 65,536 pixels)           triangles   in frustum    sub-pixel       projected area
======================================  ==========  ============  ==============  ================
HumanoidBench h1-walk                      295,328      295,328    **99.63 %**     9,461 px total
HumanoidBench h1hand-balance_simple        398,172      398,172    **99.74 %**    10,445 px
RoboCasa L52S03 agentview_left             442,466      166,413    **94.97 %**   496,968 px
RoboCasa L52S03 eye_in_hand                442,466       24,187      66.32 %     191,713 px
======================================  ==========  ============  ==============  ================

Two consequences decide the design. First, **there is no binning pass here.** Per-tile binning exists
to amortise a triangle over the many pixels it covers, and the mean triangle here covers 0.026-3.1
pixels; binning 300k-440k triangles per world into 256 tiles would cost more than the coverage it
saves. Each triangle is instead scattered directly by one lane, which for a sub-pixel triangle is a
single ``atomic_min``. Second, **the throughput bar is enormous**: matching MJWarp on h1-walk means
295,328 triangles in 47.5 us, i.e. **6.2 G triangle/s**, and RoboCasa needs 7.1 G/s to reach the
stored-frame bar. Those are the numbers this file exists to test against.

Depth only, plus a triangle-id buffer. Shading is deliberately not written: if the visibility pass
alone cannot reach the bar, no shader can rescue it, and if it can, the shader is the next question
rather than this one. The depth buffer is validated against ``mujoco.Renderer``'s own depth on the
pixels MuJoCo's segmentation says are mesh, so "it is fast" is never reported without "it draws the
same surfaces".

Floating point (``docs/concepts/bit-exact-triton.md``): the perspective divide feeds a ``floor`` that
picks a pixel, which is rule 1's exact hazard, so it is ``tl.math.div_rn``. FP contraction is left ON
and no test compares two specializations of this kernel, because its specification is the OFFICIAL
renderer and not a torch expression -- rule 2's second branch, stated here as that rule asks.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch
import triton
import triton.language as tl

#: Pixels of bounding box a single LANE will walk in the small-triangle pass. 95-99.7% of these
#: scenes' triangles fit in 4x4.
TILE = 4
#: Pixel block one PROGRAM covers in the large-triangle pass.
BIG_BLK = 16
#: Large triangles per world the second pass has room for. Overflow is counted and reported, never
#: silently dropped -- dropping them is exactly the bug that made the first version of this kernel
#: report 15.5 G triangle/s while painting 3.7% of a frame whose real coverage is 74.5%.
MAXBIG = 24576
#: Half-extent given to a MuJoCo INFINITE plane, and the grid it is cut into. MuJoCo draws the plane
#: to the horizon; a finite quad cannot, and the coverage check reports what that costs.
PLANE_EXTENT = 200.0
PLANE_GRID = 128


@triton.jit
def _raster(vert, face, tgeom, xpos, xmat, campos, cammat, zbuf, idbuf, big_uv, big_cnt,
            NT: tl.constexpr, NG: tl.constexpr, RES: tl.constexpr, FPIX: tl.constexpr,
            NBLK: tl.constexpr, BLOCK: tl.constexpr, TILE: tl.constexpr,
            WRITE_ID: tl.constexpr, MAXBIG: tl.constexpr):
    pid = tl.program_id(0)
    w = pid // NBLK
    b = pid % NBLK
    offs = b * BLOCK + tl.arange(0, BLOCK)
    m = offs < NT

    i0 = tl.load(face + offs * 3 + 0, mask=m, other=0)
    i1 = tl.load(face + offs * 3 + 1, mask=m, other=0)
    i2 = tl.load(face + offs * 3 + 2, mask=m, other=0)
    g = tl.load(tgeom + offs, mask=m, other=0)

    gb = w * NG * 9 + g * 9
    r00 = tl.load(xmat + gb + 0, mask=m, other=0.0)
    r01 = tl.load(xmat + gb + 1, mask=m, other=0.0)
    r02 = tl.load(xmat + gb + 2, mask=m, other=0.0)
    r10 = tl.load(xmat + gb + 3, mask=m, other=0.0)
    r11 = tl.load(xmat + gb + 4, mask=m, other=0.0)
    r12 = tl.load(xmat + gb + 5, mask=m, other=0.0)
    r20 = tl.load(xmat + gb + 6, mask=m, other=0.0)
    r21 = tl.load(xmat + gb + 7, mask=m, other=0.0)
    r22 = tl.load(xmat + gb + 8, mask=m, other=0.0)
    pb = w * NG * 3 + g * 3
    px_ = tl.load(xpos + pb + 0, mask=m, other=0.0)
    py_ = tl.load(xpos + pb + 1, mask=m, other=0.0)
    pz_ = tl.load(xpos + pb + 2, mask=m, other=0.0)

    cx = tl.load(campos + w * 3 + 0)
    cy = tl.load(campos + w * 3 + 1)
    cz = tl.load(campos + w * 3 + 2)
    c00 = tl.load(cammat + w * 9 + 0)
    c01 = tl.load(cammat + w * 9 + 1)
    c02 = tl.load(cammat + w * 9 + 2)
    c10 = tl.load(cammat + w * 9 + 3)
    c11 = tl.load(cammat + w * 9 + 4)
    c12 = tl.load(cammat + w * 9 + 5)
    c20 = tl.load(cammat + w * 9 + 6)
    c21 = tl.load(cammat + w * 9 + 7)
    c22 = tl.load(cammat + w * 9 + 8)

    half = RES * 0.5

    # --- three vertices: model -> world -> camera -> pixel -----------------------------------
    ax = tl.load(vert + i0 * 3 + 0, mask=m, other=0.0)
    ay = tl.load(vert + i0 * 3 + 1, mask=m, other=0.0)
    az = tl.load(vert + i0 * 3 + 2, mask=m, other=0.0)
    wx = r00 * ax + r01 * ay + r02 * az + px_
    wy = r10 * ax + r11 * ay + r12 * az + py_
    wz = r20 * ax + r21 * ay + r22 * az + pz_
    dx = wx - cx
    dy = wy - cy
    dz = wz - cz
    e0 = c00 * dx + c10 * dy + c20 * dz
    e1 = c01 * dx + c11 * dy + c21 * dz
    e2 = c02 * dx + c12 * dy + c22 * dz
    z0 = -e2
    u0 = FPIX * tl.math.div_rn(e0, tl.where(z0 > 1e-6, z0, 1e-6)) + half
    v0 = -FPIX * tl.math.div_rn(e1, tl.where(z0 > 1e-6, z0, 1e-6)) + half

    ax = tl.load(vert + i1 * 3 + 0, mask=m, other=0.0)
    ay = tl.load(vert + i1 * 3 + 1, mask=m, other=0.0)
    az = tl.load(vert + i1 * 3 + 2, mask=m, other=0.0)
    wx = r00 * ax + r01 * ay + r02 * az + px_
    wy = r10 * ax + r11 * ay + r12 * az + py_
    wz = r20 * ax + r21 * ay + r22 * az + pz_
    dx = wx - cx
    dy = wy - cy
    dz = wz - cz
    e0 = c00 * dx + c10 * dy + c20 * dz
    e1 = c01 * dx + c11 * dy + c21 * dz
    e2 = c02 * dx + c12 * dy + c22 * dz
    z1 = -e2
    u1 = FPIX * tl.math.div_rn(e0, tl.where(z1 > 1e-6, z1, 1e-6)) + half
    v1 = -FPIX * tl.math.div_rn(e1, tl.where(z1 > 1e-6, z1, 1e-6)) + half

    ax = tl.load(vert + i2 * 3 + 0, mask=m, other=0.0)
    ay = tl.load(vert + i2 * 3 + 1, mask=m, other=0.0)
    az = tl.load(vert + i2 * 3 + 2, mask=m, other=0.0)
    wx = r00 * ax + r01 * ay + r02 * az + px_
    wy = r10 * ax + r11 * ay + r12 * az + py_
    wz = r20 * ax + r21 * ay + r22 * az + pz_
    dx = wx - cx
    dy = wy - cy
    dz = wz - cz
    e0 = c00 * dx + c10 * dy + c20 * dz
    e1 = c01 * dx + c11 * dy + c21 * dz
    e2 = c02 * dx + c12 * dy + c22 * dz
    z2 = -e2
    u2 = FPIX * tl.math.div_rn(e0, tl.where(z2 > 1e-6, z2, 1e-6)) + half
    v2 = -FPIX * tl.math.div_rn(e1, tl.where(z2 > 1e-6, z2, 1e-6)) + half

    front = (z0 > 1e-4) & (z1 > 1e-4) & (z2 > 1e-4)
    x0 = tl.math.floor(tl.minimum(tl.minimum(u0, u1), u2)).to(tl.int32)
    y0 = tl.math.floor(tl.minimum(tl.minimum(v0, v1), v2)).to(tl.int32)
    x1 = tl.math.floor(tl.maximum(tl.maximum(u0, u1), u2)).to(tl.int32)
    y1 = tl.math.floor(tl.maximum(tl.maximum(v0, v1), v2)).to(tl.int32)
    onscreen = m & front & (x1 >= 0) & (y1 >= 0) & (x0 < RES) & (y0 < RES)
    small = (x1 - x0 < TILE) & (y1 - y0 < TILE)
    live = onscreen & small

    area = (u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0)
    inv_area = tl.math.div_rn(1.0, tl.where(tl.abs(area) > 1e-12, area, 1e-12))
    sgn = tl.where(area >= 0.0, 1.0, -1.0)
    # Perspective-correct depth: 1/z is what is linear in screen space. Affine z is invisible on a
    # sub-pixel triangle and wrong by metres on a floor quad seen at grazing incidence, which is most
    # of a HumanoidBench frame.
    q0 = tl.math.div_rn(1.0, z0)
    q1 = tl.math.div_rn(1.0, z1)
    q2 = tl.math.div_rn(1.0, z2)

    base = w * RES * RES
    for ty in tl.static_range(TILE):
        for tx in tl.static_range(TILE):
            qx = x0 + tx
            qy = y0 + ty
            ok = live & (qx >= 0) & (qy >= 0) & (qx < RES) & (qy < RES) & (qx <= x1) & (qy <= y1)
            fx = qx.to(tl.float32) + 0.5
            fy = qy.to(tl.float32) + 0.5
            w0 = ((u2 - u1) * (fy - v1) - (v2 - v1) * (fx - u1)) * sgn
            w1 = ((u0 - u2) * (fy - v2) - (v0 - v2) * (fx - u2)) * sgn
            w2 = ((u1 - u0) * (fy - v0) - (v1 - v0) * (fx - u0)) * sgn
            inside = ok & (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
            lam = tl.abs(inv_area)
            iz = (w0 * q0 + w1 * q1 + w2 * q2) * lam
            zz = tl.math.div_rn(1.0, tl.where(iz > 1e-9, iz, 1e-9))
            zi = zz.to(tl.int32, bitcast=True)
            addr = base + qy * RES + qx
            tl.atomic_min(zbuf + addr, zi, mask=inside)
            if WRITE_ID:
                tl.atomic_min(idbuf + addr, offs.to(tl.int32), mask=inside)

    # Everything wider than one lane's walk is handed to the second pass, projected coordinates and
    # all, so it is transformed exactly once.
    # ONE SLOT PER LANE, which needs a BLOCK of pointers and not a scalar one: an atomic on a scalar
    # pointer is a reduction and hands every lane the same index, so nine of ten large triangles would
    # be written on top of each other. Same address, block of pointers -- each lane's atomic returns
    # its own pre-value, which is exactly the unique slot wanted.
    big = onscreen & (small == 0)
    cnt_ptr = big_cnt + w + tl.zeros([BLOCK], tl.int32)
    slot = tl.atomic_add(cnt_ptr, tl.full([BLOCK], 1, tl.int32), mask=big)
    room = big & (slot < MAXBIG)
    bb = w * MAXBIG * 9 + slot * 9
    tl.store(big_uv + bb + 0, u0, mask=room)
    tl.store(big_uv + bb + 1, v0, mask=room)
    tl.store(big_uv + bb + 2, q0, mask=room)
    tl.store(big_uv + bb + 3, u1, mask=room)
    tl.store(big_uv + bb + 4, v1, mask=room)
    tl.store(big_uv + bb + 5, q1, mask=room)
    tl.store(big_uv + bb + 6, u2, mask=room)
    tl.store(big_uv + bb + 7, v2, mask=room)
    tl.store(big_uv + bb + 8, q2, mask=room)


@triton.jit
def _raster_big(big_uv, big_cnt, zbuf, idbuf,
                RES: tl.constexpr, MAXBIG: tl.constexpr, BLK: tl.constexpr,
                WRITE_ID: tl.constexpr):
    """One program per large triangle, walking its own bounding box in ``BLK x BLK`` blocks.

    There is no per-tile binning here and that is deliberate: the large triangles are 0.4-5 % of the
    scene, so a bin pass would sort a rounding error. What they are not is a rounding error in
    COVERAGE -- two floor quads own 70 % of a HumanoidBench frame -- so they get a pass whose cost is
    proportional to the area they cover rather than to how many of them there are.
    """
    pid = tl.program_id(0)
    w = pid // MAXBIG
    k = pid % MAXBIG
    n = tl.load(big_cnt + w)
    if k < n:
        _big_one(big_uv, zbuf, idbuf, w, k, RES, MAXBIG, BLK, WRITE_ID)


@triton.jit
def _big_one(big_uv, zbuf, idbuf, w, k, RES: tl.constexpr, MAXBIG: tl.constexpr,
             BLK: tl.constexpr, WRITE_ID: tl.constexpr):
    bb = w * MAXBIG * 9 + k * 9
    u0 = tl.load(big_uv + bb + 0)
    v0 = tl.load(big_uv + bb + 1)
    q0 = tl.load(big_uv + bb + 2)
    u1 = tl.load(big_uv + bb + 3)
    v1 = tl.load(big_uv + bb + 4)
    q1 = tl.load(big_uv + bb + 5)
    u2 = tl.load(big_uv + bb + 6)
    v2 = tl.load(big_uv + bb + 7)
    q2 = tl.load(big_uv + bb + 8)

    x0 = tl.maximum(tl.math.floor(tl.minimum(tl.minimum(u0, u1), u2)).to(tl.int32), 0)
    y0 = tl.maximum(tl.math.floor(tl.minimum(tl.minimum(v0, v1), v2)).to(tl.int32), 0)
    x1 = tl.minimum(tl.math.floor(tl.maximum(tl.maximum(u0, u1), u2)).to(tl.int32), RES - 1)
    y1 = tl.minimum(tl.math.floor(tl.maximum(tl.maximum(v0, v1), v2)).to(tl.int32), RES - 1)
    area = (u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0)
    inv_area = tl.math.div_rn(1.0, tl.where(tl.abs(area) > 1e-12, area, 1e-12))
    sgn = tl.where(area >= 0.0, 1.0, -1.0)
    lam = tl.abs(inv_area)

    ar = tl.arange(0, BLK)
    base = w * RES * RES
    for by in range(y0, y1 + 1, BLK):
        for bx in range(x0, x1 + 1, BLK):
            qx = bx + ar[None, :]
            qy = by + ar[:, None]
            ok = (qx <= x1) & (qy <= y1)
            fx = qx.to(tl.float32) + 0.5
            fy = qy.to(tl.float32) + 0.5
            w0 = ((u2 - u1) * (fy - v1) - (v2 - v1) * (fx - u1)) * sgn
            w1 = ((u0 - u2) * (fy - v2) - (v0 - v2) * (fx - u2)) * sgn
            w2 = ((u1 - u0) * (fy - v0) - (v1 - v0) * (fx - u0)) * sgn
            inside = ok & (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
            iz = (w0 * q0 + w1 * q1 + w2 * q2) * lam
            zz = tl.math.div_rn(1.0, tl.where(iz > 1e-9, iz, 1e-9))
            zi = zz.to(tl.int32, bitcast=True)
            addr = base + qy * RES + qx
            tl.atomic_min(zbuf + addr, zi, mask=inside)
            if WRITE_ID:
                tl.atomic_min(idbuf + addr, tl.zeros([BLK, BLK], tl.int32), mask=inside)


@triton.jit
def _transform_only(vert, face, tgeom, xpos, xmat, campos, cammat, acc,
                    NT: tl.constexpr, NG: tl.constexpr, RES: tl.constexpr, FPIX: tl.constexpr,
                    NBLK: tl.constexpr, BLOCK: tl.constexpr):
    """The same vertex work with the scatter removed: the floor under any rasterizer."""
    pid = tl.program_id(0)
    w = pid // NBLK
    b = pid % NBLK
    offs = b * BLOCK + tl.arange(0, BLOCK)
    m = offs < NT
    i0 = tl.load(face + offs * 3 + 0, mask=m, other=0)
    g = tl.load(tgeom + offs, mask=m, other=0)
    gb = w * NG * 9 + g * 9
    r00 = tl.load(xmat + gb + 0, mask=m, other=0.0)
    r01 = tl.load(xmat + gb + 1, mask=m, other=0.0)
    r02 = tl.load(xmat + gb + 2, mask=m, other=0.0)
    pb = w * NG * 3 + g * 3
    px_ = tl.load(xpos + pb + 0, mask=m, other=0.0)
    s = tl.zeros([BLOCK], tl.float32)
    for k in tl.static_range(3):
        j = tl.load(face + offs * 3 + k, mask=m, other=0)
        ax = tl.load(vert + j * 3 + 0, mask=m, other=0.0)
        ay = tl.load(vert + j * 3 + 1, mask=m, other=0.0)
        az = tl.load(vert + j * 3 + 2, mask=m, other=0.0)
        s += r00 * ax + r01 * ay + r02 * az + px_
    _ = i0
    tl.atomic_add(acc + w, tl.sum(s, axis=0))


# ------------------------------------------------------------------ scene -> triangle soup


def primitive_mesh(gtype, size):
    """A MuJoCo primitive as a triangle mesh, so the soup is the whole scene and not just the meshes."""
    import mujoco
    import trimesh
    t = int(gtype)
    s = np.asarray(size, np.float64)
    if t == int(mujoco.mjtGeom.mjGEOM_PLANE):
        # A GRID, not a quad, and that is not cosmetic. This rasterizer has no near-plane clipping:
        # a triangle with any vertex behind the camera is dropped whole, and a single 400 m floor quad
        # always has one. Measured: with two triangles the floor vanished and coverage came out at
        # 4.0% of a frame whose real coverage is 74.5%. Split into cells, only the cells that straddle
        # the camera are lost and those are under and behind it, outside the frustum. The cost is
        # ~32k triangles, 11% on top of a HumanoidBench scene, and it is charged in every number here.
        e = float(s[0]) if s[0] > 0 else PLANE_EXTENT
        f = float(s[1]) if s[1] > 0 else e
        n = PLANE_GRID
        gx = np.linspace(-e, e, n + 1)
        gy = np.linspace(-f, f, n + 1)
        X, Y = np.meshgrid(gx, gy, indexing="ij")
        v = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
        i = np.arange(n * n).reshape(n, n)
        a0 = (i // n) * (n + 1) + (i % n)
        faces = np.concatenate([
            np.stack([a0.ravel(), a0.ravel() + 1, a0.ravel() + n + 2], 1),
            np.stack([a0.ravel(), a0.ravel() + n + 2, a0.ravel() + n + 1], 1)])
        return v, faces
    if t == int(mujoco.mjtGeom.mjGEOM_BOX):
        b = trimesh.creation.box(extents=2 * s[:3])
        return np.asarray(b.vertices), np.asarray(b.faces)
    if t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        b = trimesh.creation.icosphere(subdivisions=2, radius=float(s[0]))
        return np.asarray(b.vertices), np.asarray(b.faces)
    if t == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        b = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        return np.asarray(b.vertices) * s[:3], np.asarray(b.faces)
    if t == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        b = trimesh.creation.cylinder(radius=float(s[0]), height=2 * float(s[1]), sections=24)
        return np.asarray(b.vertices), np.asarray(b.faces)
    if t == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        b = trimesh.creation.capsule(radius=float(s[0]), height=2 * float(s[1]), count=[12, 12])
        return np.asarray(b.vertices), np.asarray(b.faces)
    return None


def build_soup(m, ids, decimate=1.0):
    """(vert (NV,3), face (NT,3), tgeom (NT,)) in each geom's LOCAL frame, for every drawn geom."""
    import mujoco
    V, F, G, skipped = [], [], [], []
    nv = 0
    for gi in ids:
        t = int(m.geom_type[gi])
        if t == int(mujoco.mjtGeom.mjGEOM_MESH):
            did = int(m.geom_dataid[gi])
            if did < 0:
                skipped.append((int(gi), t))
                continue
            va, n = int(m.mesh_vertadr[did]), int(m.mesh_vertnum[did])
            fa, k = int(m.mesh_faceadr[did]), int(m.mesh_facenum[did])
            v = np.asarray(m.mesh_vert[va:va + n], np.float64).reshape(n, 3)
            f = np.asarray(m.mesh_face[fa:fa + k], np.int64).reshape(k, 3)
        else:
            got = primitive_mesh(t, m.geom_size[gi])
            if got is None:
                skipped.append((int(gi), t))
                continue
            v, f = got
        if decimate < 1.0 and len(f) > 64:
            import fast_simplification
            v2, f2 = fast_simplification.simplify(np.ascontiguousarray(v, np.float32),
                                                  np.ascontiguousarray(f, np.int32),
                                                  1.0 - decimate)
            if len(f2) >= 4:
                v, f = np.asarray(v2, np.float64), np.asarray(f2, np.int64)
        V.append(v)
        F.append(f + nv)
        G.append(np.full(len(f), int(gi), np.int64))
        nv += len(v)
    return (np.concatenate(V).astype(np.float32), np.concatenate(F).astype(np.int32),
            np.concatenate(G).astype(np.int32), skipped)


def timed(fn, reps=30, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--decimate", type=float, default=1.0,
                    help="keep this fraction of triangles (1.0 = the model as compiled)")
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()

    import mujoco
    import stable_envs as se
    cls = se.REGISTRY[a.env]
    if a.env.startswith("RoboCasa"):
        from stable_envs.envs import robocasa as M
        s = M._scene(cls.KITCHEN)
        m, groups = s.model, M.VISIBLE_GROUPS
        cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cls.CAMERA)
        env = se.make(a.env, batch=1, size=a.res, device="cuda")
        st = env.sample_states(a.nworld)
        qpos = env._full_qpos(torch.as_tensor(st, device="cuda")).cpu().numpy().astype(np.float64)
    else:
        from stable_envs.envs import humanoidbench as M
        b = M._build_bundle(cls.ROBOT, cls.TASK)
        m, groups, cam = b.model, M.VISIBLE_GROUPS, int(b.info["cam"])
        env = se.make(a.env, batch=1, size=a.res, device="cuda")
        qpos = np.asarray(env.sample_states(a.nworld), np.float64)
    del env
    torch.cuda.empty_cache()

    ids = np.where(np.isin(np.asarray(m.geom_group, np.int64), list(groups)))[0]
    vert, face, tgeom, skipped = build_soup(m, ids, a.decimate)
    NT, NV = len(face), len(vert)
    out = {"env": a.env, "res": a.res, "nworld": a.nworld, "decimate": a.decimate,
           "gpu": torch.cuda.get_device_name(0), "n_tri": int(NT), "n_vert": int(NV),
           "n_geom_drawn": int(len(ids)), "skipped_geoms": skipped[:20],
           "n_skipped": len(skipped)}
    print(json.dumps({k: out[k] for k in ("env", "n_tri", "n_vert", "n_geom_drawn", "n_skipped")}),
          flush=True)

    # Per-world geom transforms and camera, from MuJoCo itself: identical inputs to the ray tracer's.
    d = mujoco.MjData(m)
    G = int(m.ngeom)
    xpos = np.empty((a.nworld, G, 3), np.float32)
    xmat = np.empty((a.nworld, G, 9), np.float32)
    campos = np.empty((a.nworld, 3), np.float32)
    cammat = np.empty((a.nworld, 9), np.float32)
    for i in range(a.nworld):
        d.qpos[:] = qpos[i % len(qpos)]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        xpos[i] = d.geom_xpos
        xmat[i] = d.geom_xmat
        campos[i] = d.cam_xpos[cam]
        cammat[i] = d.cam_xmat[cam]
    fpix = a.res / (2.0 * np.tan(np.deg2rad(float(m.cam_fovy[cam])) / 2.0))

    dev = "cuda"
    tv = torch.as_tensor(vert, device=dev)
    tf = torch.as_tensor(face, device=dev)
    tg = torch.as_tensor(tgeom, device=dev)
    tp = torch.as_tensor(xpos.reshape(-1), device=dev)
    tm = torch.as_tensor(xmat.reshape(-1), device=dev)
    tcp = torch.as_tensor(campos.reshape(-1), device=dev)
    tcm = torch.as_tensor(cammat.reshape(-1), device=dev)
    zb = torch.empty((a.nworld, a.res, a.res), dtype=torch.int32, device=dev)
    ib = torch.empty_like(zb)
    big_uv = torch.zeros((a.nworld, MAXBIG, 9), dtype=torch.float32, device=dev)
    big_cnt = torch.zeros(a.nworld, dtype=torch.int32, device=dev)
    INF = np.int32(np.float32(1e30).view(np.int32))
    NBLK = triton.cdiv(NT, a.block)
    grid = (a.nworld * NBLK,)
    grid_big = (a.nworld * MAXBIG,)

    def run(write_id, big=True):
        zb.fill_(int(INF))
        big_cnt.zero_()
        if write_id:
            ib.fill_(2 ** 30)
        _raster[grid](tv, tf, tg, tp, tm, tcp, tcm, zb, ib, big_uv, big_cnt,
                      NT=NT, NG=G, RES=a.res, FPIX=float(fpix), NBLK=NBLK,
                      BLOCK=a.block, TILE=TILE, WRITE_ID=write_id, MAXBIG=MAXBIG, num_warps=4)
        if big:
            _raster_big[grid_big](big_uv, big_cnt, zb, ib, RES=a.res, MAXBIG=MAXBIG,
                                  BLK=BIG_BLK, WRITE_ID=write_id, num_warps=4)

    run(True)
    torch.cuda.synchronize()
    out["big_tri_per_world"] = {"mean": float(big_cnt.float().mean()),
                                "max": int(big_cnt.max()),
                                "overflow": bool(int(big_cnt.max()) > MAXBIG),
                                "maxbig": MAXBIG}
    print(json.dumps(out["big_tri_per_world"]), flush=True)

    out["raster_small_only_ms"] = 1e3 * timed(lambda: run(False, big=False), a.reps)
    out["raster_ms"] = 1e3 * timed(lambda: run(False), a.reps)
    out["raster_id_ms"] = 1e3 * timed(lambda: run(True), a.reps)
    acc = torch.zeros(a.nworld, dtype=torch.float32, device=dev)
    out["transform_only_ms"] = 1e3 * timed(
        lambda: _transform_only[grid](tv, tf, tg, tp, tm, tcp, tcm, acc, NT=NT, NG=G, RES=a.res,
                                      FPIX=float(fpix), NBLK=NBLK, BLOCK=a.block, num_warps=4),
        a.reps)
    for k in ("raster_small_only_ms", "raster_ms", "raster_id_ms", "transform_only_ms"):
        out[k.replace("_ms", "_fps")] = a.nworld / (out[k] / 1e3)
        out[k.replace("_ms", "_gtris")] = a.nworld * NT / (out[k] / 1e3) / 1e9
    print(json.dumps({k: out[k] for k in out if k.endswith(("_ms", "_fps", "_gtris"))}), flush=True)

    # --- does it draw the same surfaces? depth against mujoco's own, on mesh pixels -----------
    zf = zb.view(torch.float32)
    vopt = mujoco.MjvOption()
    vopt.geomgroup[:] = 0
    for gg in groups:
        vopt.geomgroup[gg] = 1
    r = mujoco.Renderer(m, width=a.res, height=a.res)
    errs, cov = [], []
    for i in range(min(4, a.nworld)):
        d.qpos[:] = qpos[i % len(qpos)]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cam, scene_option=vopt)
        r.enable_depth_rendering()
        dep = r.render().astype(np.float64)
        r.disable_depth_rendering()
        mine = zf[i].cpu().numpy().astype(np.float64)
        hit = mine < 1e29
        real = dep < dep.max() * 0.999
        both = hit & real
        cov.append({"ours_hit": float(hit.mean()), "mujoco_hit": float(real.mean()),
                    "agree_mask": float((hit == real).mean())})
        if both.any():
            rel = np.abs(mine[both] - dep[both]) / np.maximum(dep[both], 1e-6)
            errs.append({"rel_depth_mean": float(rel.mean()),
                         "rel_depth_median": float(np.median(rel)),
                         "within_1pct": float((rel < 0.01).mean())})
    out["depth_check"] = {"coverage": cov, "error": errs}
    print(json.dumps(out["depth_check"]), flush=True)

    p = (f"/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/altrender/"
         f"raster_{a.env}_d{a.decimate}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
