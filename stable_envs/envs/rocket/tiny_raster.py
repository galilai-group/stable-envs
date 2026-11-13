"""Rocket frames from a batched Numba port of PyBullet's TinyRenderer.

The port follows pybullet 3.2.7: ``examples/TinyRenderer`` (``our_gl.cpp``, ``TinyRenderer.cpp``, ``model.cpp``)
and ``TinyRendererVisualShapeConverter.cpp``. It keeps their arithmetic: float32 vertex transforms with
reverse-order dot products, near-plane clipping, double barycentrics at integer pixel positions, a float32
z-buffer that a double depth must reach, the default light without shadows, and two truncations to uint8.
The meshes are the ones TinyRenderer builds from the URDFs: Bullet's 42-direction convex hull of each cylinder
(``btShapeHull`` and ``HullLibrary``), the textured cube of each box and the ``plane100.obj`` ground.

All lanes draw in one call, one lane per Numba thread.
"""

from __future__ import annotations

import contextlib
import functools
import math

import numpy as np
from numba import njit, prange

from . import legacy as geometry

F32 = np.float32
_ZERO = F32(0.0)
_ONE = F32(1.0)
_TWO = F32(2.0)
_M_ONE = F32(-1.0)
_FLT_MAX = F32(np.finfo(np.float32).max)
#: Light coefficients of ``TinyRendererVisualShapeConverter::render``.
_AMBIENT = F32(0.6)
_DIFFUSE = F32(0.35)
_SPECULAR = F32(0.05)
#: A pixel of a triangle with one vertex normal shades within 3e-4 of the vertex shade (see ``_frames``).
_FLAT_MARGIN = F32(2e-3)

#: ``render/rgba`` rows in ``render.py`` order.
BODY_COLOR, FIN_COLOR, LEG_COLOR, BOOSTER_COLOR, PAD_COLOR, FLAME_COLOR = range(6)
#: ``(name, body, link pose, mesh, colour row)`` of each TinyRenderer object, in draw order. ``body`` indexes
#: the ``(ground, rocket, pad)`` poses; a colour row of -1 keeps the URDF white.
_BOX, _CYL, _PLANE = 0, 1, 2
OBJECTS = (
    ("ground", 0, None, (_PLANE,), -1),
    ("body", 1, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), (_CYL, 4.77, 0.185), BODY_COLOR),
    ("fueltank", 1, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), (_CYL, 1.0, 0.1), -1),
    ("booster", 1, ((0.0, 0.0, -2.0), (0.0, 0.0, 0.0)), (_CYL, 0.5, 0.25), BOOSTER_COLOR),
    ("fin_pos_x", 1, ((0.35, 0.0, 2.051), (0.0, 0.0, 0.0)), (_BOX, 0.3, 0.03, 0.3), FIN_COLOR),
    ("fin_neg_x", 1, ((-0.35, 0.0, 2.051), (0.0, 0.0, 0.0)), (_BOX, 0.3, 0.03, 0.3), FIN_COLOR),
    ("fin_pos_y", 1, ((0.0, 0.35, 2.051), (0.0, 0.0, 0.0)), (_BOX, 0.03, 0.3, 0.3), FIN_COLOR),
    ("fin_neg_y", 1, ((0.0, -0.35, 2.051), (0.0, 0.0, 0.0)), (_BOX, 0.03, 0.3, 0.3), FIN_COLOR),
    ("leg_1", 1, ((0.0, 0.35, -2.4), (0.0, 0.0, 0.0)), (_BOX, 0.05, 0.5, 0.05), LEG_COLOR),
    ("leg_2", 1, ((0.3031, -0.175, -2.4), (0.0, 0.0, 4.188)), (_BOX, 0.05, 0.5, 0.05), LEG_COLOR),
    ("leg_3", 1, ((-0.3031, -0.175, -2.4), (0.0, 0.0, -4.188)), (_BOX, 0.05, 0.5, 0.05), LEG_COLOR),
    ("flame", 1, ((0.0, 0.0, -3.4), (0.0, 0.0, 0.0)), (_CYL, 2.0, 0.15), FLAME_COLOR),
    ("pad", 2, None, (_CYL, 0.1, 2.0), PAD_COLOR),
)

# Position and normal of the 24 vertices of ``cube_vertices_textured`` and its 12 triangles (ShapeData.h).
_CUBE = np.array(
    [
        [-1, -1, 1, 0, 0, 1],
        [1, -1, 1, 0, 0, 1],
        [1, 1, 1, 0, 0, 1],
        [-1, 1, 1, 0, 0, 1],
        [-1, -1, -1, 0, 0, -1],
        [1, -1, -1, 0, 0, -1],
        [1, 1, -1, 0, 0, -1],
        [-1, 1, -1, 0, 0, -1],
        [-1, -1, -1, -1, 0, 0],
        [-1, 1, -1, -1, 0, 0],
        [-1, 1, 1, -1, 0, 0],
        [-1, -1, 1, -1, 0, 0],
        [1, -1, -1, 1, 0, 0],
        [1, 1, -1, 1, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [1, -1, 1, 1, 0, 0],
        [-1, -1, -1, 0, -1, 0],
        [-1, -1, 1, 0, -1, 0],
        [1, -1, 1, 0, -1, 0],
        [1, -1, -1, 0, -1, 0],
        [-1, 1, -1, 0, 1, 0],
        [-1, 1, 1, 0, 1, 0],
        [1, 1, 1, 0, 1, 0],
        [1, 1, -1, 0, 1, 0],
    ],
    np.float32,
)
_CUBE_FACES = np.array(
    [[0, 1, 2], [0, 2, 3], [6, 5, 4], [7, 6, 4], [10, 9, 8], [11, 10, 8]]
    + [[12, 13, 14], [12, 14, 15], [18, 17, 16], [19, 18, 16], [20, 21, 22], [20, 22, 23]],
    np.int64,
)

_SIMD_2_PI = 2.0 * math.pi
_FLT_MAX_D = float(np.finfo(np.float32).max)


# -- Bullet's convex hull of a cylinder (btShapeHull::buildHull(0) and HullLibrary), in double precision ----------


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _length(a):
    return math.sqrt(_dot(a, a))


def _normalized(a):
    return _scale(a, 1.0 / _length(a))


def _cylinder_support_points(length, radius, margin=geometry.HULL_MARGIN):
    """``btConvexHullShape::localGetSupportingVertex`` of the 32-step cylinder at the 42 unit-sphere points."""
    ring = []
    for i in range(32):
        angle = _SIMD_2_PI * (i / 32)
        x, y = radius * math.sin(angle), radius * math.cos(angle)
        ring += [(x, y, length / 2.0), (x, y, -length / 2.0)]
    out = []
    for d in geometry.UNIT_SPHERE_42:
        d = (float(d[0]), float(d[1]), float(d[2]))
        best, index = -math.inf, -1
        for i, p in enumerate(ring):
            dot = p[0] * d[0] + p[1] * d[1] + p[2] * d[2]
            if dot > best:
                best, index = dot, i
        n = _normalized(d)
        p = ring[index]
        out.append((p[0] + margin * n[0], p[1] + margin * n[1], p[2] + margin * n[2]))
    return out


class _Triangle:
    __slots__ = ("v", "n", "id", "vmax", "rise")

    def __init__(self, a, b, c, ident):
        self.v = [a, b, c]
        self.n = [-1, -1, -1]
        self.id = ident
        self.vmax = -1
        self.rise = 0.0

    def neib(self, a, b):
        for i in range(3):
            i1 = (i + 1) % 3
            if (self.v[i] == a and self.v[i1] == b) or (self.v[i] == b and self.v[i1] == a):
                return (i + 2) % 3
        raise AssertionError("HullLibrary: no shared edge")


class _HullLibrary:
    """Bullet's ``HullLibrary`` (Stan Melax's incremental hull) for ``QF_TRIANGLES`` output."""

    def __init__(self):
        self.tris = []

    def _allocate(self, a, b, c):
        t = _Triangle(a, b, c, len(self.tris))
        self.tris.append(t)
        return t

    def _release(self, t):
        self.tris[t.id] = None

    @staticmethod
    def _tri_normal(v0, v1, v2):
        cp = _cross(_sub(v1, v0), _sub(v2, v1))
        m = _length(cp)
        if m == 0:
            return (1.0, 0.0, 0.0)
        return _scale(cp, 1.0 / m)

    def _above(self, verts, t, p, epsilon):
        n = self._tri_normal(verts[t[0]], verts[t[1]], verts[t[2]])
        return _dot(n, _sub(p, verts[t[0]])) > epsilon

    @staticmethod
    def _maxdirfiltered(p, d, allow):
        m = -1
        for i in range(len(p)):
            if allow[i] and (m == -1 or _dot(p[i], d) > _dot(p[m], d)):
                m = i
        return m

    @staticmethod
    def _orth(v):
        a = _cross(v, (0.0, 0.0, 1.0))
        b = _cross(v, (0.0, 1.0, 0.0))
        return _normalized(a) if _length(a) > _length(b) else _normalized(b)

    def _maxdirsterid(self, p, d, allow):
        rads = _SIMD_2_PI / 360.0
        m = -1
        while m == -1:
            m = self._maxdirfiltered(p, d, allow)
            if allow[m] == 3:
                return m
            u = self._orth(d)
            v = _cross(u, d)

            def toward(angle):
                s, c = math.sin(rads * angle), math.cos(rads * angle)
                return tuple(d[k] + (u[k] * s + v[k] * c) * 0.025 for k in range(3))

            ma = -1
            x = 0.0
            while x <= 360.0:
                mb = self._maxdirfiltered(p, toward(x), allow)
                if ma == m and mb == m:
                    allow[m] = 3
                    return m
                if ma != -1 and ma != mb:
                    mc = ma
                    xx = x - 40.0
                    while xx <= x:
                        md = self._maxdirfiltered(p, toward(xx), allow)
                        if mc == m and md == m:
                            allow[m] = 3
                            return m
                        mc = md
                        xx += 5.0
                ma = mb
                x += 45.0
            allow[m] = 0
            m = -1
        raise AssertionError("HullLibrary: no extreme point")

    def _find_simplex(self, verts, allow):
        basis0 = (0.01, 0.02, 1.0)
        p0 = self._maxdirsterid(verts, basis0, allow)
        p1 = self._maxdirsterid(verts, _scale(basis0, -1.0), allow)
        basis0 = _sub(verts[p0], verts[p1])
        if p0 == p1 or basis0 == (0.0, 0.0, 0.0):
            return None
        basis1 = _cross((1.0, 0.02, 0.0), basis0)
        basis2 = _cross((-0.02, 1.0, 0.0), basis0)
        basis1 = _normalized(basis1) if _length(basis1) > _length(basis2) else _normalized(basis2)
        p2 = self._maxdirsterid(verts, basis1, allow)
        if p2 in (p0, p1):
            p2 = self._maxdirsterid(verts, _scale(basis1, -1.0), allow)
        if p2 in (p0, p1):
            return None
        basis1 = _sub(verts[p2], verts[p0])
        basis2 = _normalized(_cross(basis1, basis0))
        p3 = self._maxdirsterid(verts, basis2, allow)
        if p3 in (p0, p1, p2):
            p3 = self._maxdirsterid(verts, _scale(basis2, -1.0), allow)
        if p3 in (p0, p1, p2):
            return None
        if _dot(_sub(verts[p3], verts[p0]), _cross(_sub(verts[p1], verts[p0]), _sub(verts[p2], verts[p0]))) < 0:
            p2, p3 = p3, p2
        return p0, p1, p2, p3

    def _b2bfix(self, s, t):
        for i in range(3):
            a, b = s.v[(i + 1) % 3], s.v[(i + 2) % 3]
            sab, tba = s.n[s.neib(a, b)], t.n[t.neib(b, a)]
            x = self.tris[sab]
            x.n[x.neib(b, a)] = tba
            y = self.tris[tba]
            y.n[y.neib(a, b)] = sab

    def _removeb2b(self, s, t):
        self._b2bfix(s, t)
        self._release(s)
        self._release(t)

    def _extrude(self, t0, v):
        t = list(t0.v)
        n = len(self.tris)
        ta = self._allocate(v, t[1], t[2])
        ta.n = [t0.n[0], n + 1, n + 2]
        x = self.tris[t0.n[0]]
        x.n[x.neib(t[1], t[2])] = n + 0
        tb = self._allocate(v, t[2], t[0])
        tb.n = [t0.n[1], n + 2, n + 0]
        x = self.tris[t0.n[1]]
        x.n[x.neib(t[2], t[0])] = n + 1
        tc = self._allocate(v, t[0], t[1])
        tc.n = [t0.n[2], n + 0, n + 1]
        x = self.tris[t0.n[2]]
        x.n[x.neib(t[0], t[1])] = n + 2
        for tx in (ta, tb, tc):
            nb = self.tris[tx.n[0]]
            if v in nb.v:
                self._removeb2b(tx, nb)
        self._release(t0)

    def _extrudable(self, epsilon):
        t = None
        for x in self.tris:
            if t is None or (x is not None and t.rise < x.rise):
                t = x
        return t if t.rise > epsilon else None

    def _calchullgen(self, verts, vlimit):
        if len(verts) < 4:
            return False
        allow = [1] * len(verts)
        isextreme = [0] * len(verts)
        bmin, bmax = list(verts[0]), list(verts[0])
        for p in verts:
            for k in range(3):
                bmin[k] = p[k] if p[k] < bmin[k] else bmin[k]
                bmax[k] = p[k] if bmax[k] < p[k] else bmax[k]
        epsilon = _length(_sub(bmax, bmin)) * 0.001
        simplex = self._find_simplex(verts, allow)
        if simplex is None:
            return False
        p = simplex
        total = verts[p[0]]
        for k in (1, 2, 3):
            total = tuple(total[c] + verts[p[k]][c] for c in range(3))
        center = _scale(total, 1.0 / 4.0)
        t0 = self._allocate(p[2], p[3], p[1])
        t0.n = [2, 3, 1]
        t1 = self._allocate(p[3], p[2], p[0])
        t1.n = [3, 2, 0]
        t2 = self._allocate(p[0], p[1], p[3])
        t2.n = [0, 1, 3]
        t3 = self._allocate(p[1], p[0], p[2])
        t3.n = [1, 0, 2]
        for k in p:
            isextreme[k] = 1
        for t in self.tris:
            n = self._tri_normal(verts[t.v[0]], verts[t.v[1]], verts[t.v[2]])
            t.vmax = self._maxdirsterid(verts, n, allow)
            t.rise = _dot(n, _sub(verts[t.vmax], verts[t.v[0]]))
        vlimit -= 4
        while vlimit > 0:
            te = self._extrudable(epsilon)
            if te is None:
                break
            v = te.vmax
            isextreme[v] = 1
            j = len(self.tris)
            while j:
                j -= 1
                t = self.tris[j]
                if t is not None and self._above(verts, list(t.v), verts[v], 0.01 * epsilon):
                    self._extrude(t, v)
            j = len(self.tris)
            while j:
                j -= 1
                t = self.tris[j]
                if t is None:
                    continue
                if v not in t.v:
                    break
                nt = list(t.v)
                skinny = _length(_cross(_sub(verts[nt[1]], verts[nt[0]]), _sub(verts[nt[2]], verts[nt[1]])))
                if self._above(verts, nt, center, 0.01 * epsilon) or skinny < epsilon * epsilon * 0.1:
                    self._extrude(self.tris[t.n[0]], v)
                    j = len(self.tris)
            j = len(self.tris)
            while j:
                j -= 1
                t = self.tris[j]
                if t is None:
                    continue
                if t.vmax >= 0:
                    break
                n = self._tri_normal(verts[t.v[0]], verts[t.v[1]], verts[t.v[2]])
                t.vmax = self._maxdirsterid(verts, n, allow)
                if isextreme[t.vmax]:
                    t.vmax = -1
                else:
                    t.rise = _dot(n, _sub(verts[t.vmax], verts[t.v[0]]))
            vlimit -= 1
        return True

    @staticmethod
    def _cleanup(points, normal_epsilon):
        """``CleanupVertices``: points scaled to their bounding box, near-duplicates merged."""
        bmin, bmax = [_FLT_MAX_D] * 3, [-_FLT_MAX_D] * 3
        for p in points:
            for j in range(3):
                bmin[j] = p[j] if p[j] < bmin[j] else bmin[j]
                bmax[j] = p[j] if p[j] > bmax[j] else bmax[j]
        dx, dy, dz = bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]
        if min(dx, dy, dz) < 0.000001 or len(points) < 3:
            raise ValueError("HullLibrary port: flat point clouds are not supported")
        center = [dx * 0.5 + bmin[0], dy * 0.5 + bmin[1], dz * 0.5 + bmin[2]]
        scale = (dx, dy, dz)
        recip = (1 / dx, 1 / dy, 1 / dz)
        center = [center[k] * recip[k] for k in range(3)]

        def dist(x, y, z):
            ex, ey, ez = x - center[0], y - center[1], z - center[2]
            return ex * ex + ey * ey + ez * ez

        out, mapping = [], []
        for p in points:
            px, py, pz = p[0] * recip[0], p[1] * recip[1], p[2] * recip[2]
            j = 0
            while j < len(out):
                v = out[j]
                if (
                    abs(v[0] - px) < normal_epsilon
                    and abs(v[1] - py) < normal_epsilon
                    and abs(v[2] - pz) < normal_epsilon
                ):
                    if dist(px, py, pz) > dist(*v):
                        out[j] = (px, py, pz)
                    break
                j += 1
            if j == len(out):
                out.append((px, py, pz))
            mapping.append(j)
        return out, scale

    def create(self, points, normal_epsilon=float(np.float32(0.001)), max_vertices=4096):
        """``CreateConvexHull`` with ``QF_TRIANGLES`` and ``HullDesc`` defaults: ``(vertices, indices)``."""
        cleaned, scale = self._cleanup(points, normal_epsilon)
        verts = [tuple(v[k] * scale[k] for k in range(3)) for v in cleaned]
        self.tris = []
        if not self._calchullgen(verts, max_vertices):
            raise ValueError("HullLibrary port: no hull")
        indices = [i for t in self.tris if t is not None for i in t.v]
        # BringOutYourDead: number the used vertices in order of first use.
        remap, out = {}, []
        for i, v in enumerate(indices):
            if v not in remap:
                remap[v] = len(out)
                out.append(verts[v])
            indices[i] = remap[v]
        return out, indices


def _normalize_f32(v):
    """``vec<3, float>::normalize``: ``v * (1 / sqrt(x*x + y*y + z*z))`` in float32."""
    x, y, z = F32(v[0]), F32(v[1]), F32(v[2])
    norm = F32(math.sqrt(float(x * x + y * y + z * z)))
    inv = _ONE / norm
    return np.array([x * inv, y * inv, z * inv], np.float32)


def _stable_normal(v):
    """A vertex normal after TinyRenderer's in-place renormalizations reach their fixed point."""
    n = np.asarray(v, np.float32)
    for _ in range(16):
        m = _normalize_f32(n)
        if np.array_equal(m, n):
            break
        n = m
    return n


@functools.cache
def bullet_cylinder(length, radius):
    """TinyRenderer's mesh of a URDF cylinder: float64 hull vertices, float32 normals and int64 triangles."""
    vertices, indices = _HullLibrary().create(_cylinder_support_points(length, radius))
    vert = np.array(vertices, np.float64)
    norm = np.empty((len(vert), 3), np.float32)
    for i, p in enumerate(vertices):
        l2 = _dot(p, p)
        n = _scale(p, 1.0 / math.sqrt(l2)) if l2 >= np.finfo(np.float64).eps ** 2 else (1.0, 0.0, 0.0)
        norm[i] = _stable_normal(np.array(n, np.float32))
    return vert, norm, np.array(indices, np.int64).reshape(-1, 3)


def bullet_box(size):
    """TinyRenderer's mesh of a URDF box of full ``size``: the textured cube scaled by half extents."""
    half = np.array([s * 0.5 for s in size], np.float64)
    vert = (half * _CUBE[:, :3].astype(np.float64)).astype(np.float32).astype(np.float64)
    return vert, _CUBE[:, 3:].copy(), _CUBE_FACES.copy()


def ground_mesh():
    """``plane100.obj`` as TinyRenderer holds it: one vertex per face corner, flat normals and the obj UVs."""
    corners = np.array(
        [[-geometry.GROUND_HALF, -geometry.GROUND_HALF], [geometry.GROUND_HALF, geometry.GROUND_HALF]], np.float64
    )
    quad = np.array(
        [[corners[1, 0], corners[0, 1]], [corners[1, 0], corners[1, 1]], [corners[0, 0], corners[1, 1]], corners[0]]
    )
    uv = np.array([[100.0, 0.0], [100.0, 100.0], [0.0, 100.0], [0.0, 0.0]])
    corner_ids = [0, 1, 2, 0, 2, 3]
    vert = np.zeros((6, 3))
    vert[:, :2] = quad[corner_ids]
    norm = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (6, 1))
    return vert, norm, np.arange(6, dtype=np.int64).reshape(2, 3), uv[corner_ids]


def object_mesh(spec):
    kind = spec[0]
    if kind == _PLANE:
        return ground_mesh()
    vert, norm, face = bullet_cylinder(spec[1], spec[2]) if kind == _CYL else bullet_box(spec[1:])
    return vert, norm, face, np.zeros((len(vert), 2))


class Scene:
    """The TinyRenderer objects of the Rocket scene as flat arrays for the kernel."""

    def __init__(self):
        import pybullet

        verts, norms, uvs, faces = [], [], [], []
        vstart, fstart = [0], [0]
        for _, _, _, spec, _ in OBJECTS:
            v, n, f, uv = object_mesh(spec)
            verts.append(v.astype(np.float32))
            norms.append(np.asarray(n, np.float32))
            uvs.append(uv.astype(np.float32))
            faces.append(f.astype(np.int32))
            vstart.append(vstart[-1] + len(v))
            fstart.append(fstart[-1] + len(f))
        self.verts = np.ascontiguousarray(np.concatenate(verts))
        self.norms = np.ascontiguousarray(np.concatenate(norms))
        self.uvs = np.ascontiguousarray(np.concatenate(uvs))
        self.faces = np.ascontiguousarray(np.concatenate(faces))
        self.vstart = np.array(vstart, np.int32)
        self.fstart = np.array(fstart, np.int32)
        self.textured = np.array([spec[0] == _PLANE for _, _, _, spec, _ in OBJECTS])
        self.texture = np.ascontiguousarray(geometry.checker_texture())
        self.color_rows = np.array([row for *_, row in OBJECTS], np.int64)
        projection = pybullet.computeProjectionMatrixFOV(geometry.CAM_FOV, 1.0, geometry.CAM_NEAR, geometry.CAM_FAR)
        self.projection = np.asarray(projection, np.float32).reshape(4, 4).T.copy()
        p10, p14 = F32(projection[10]), F32(projection[14])
        self.near = p14 / (p10 - _ONE)
        self.far = p14 / (p10 + _ONE)
        light = np.array([-50.0, 30.0, 100.0])
        self.light = (light * (1.0 / math.sqrt(_dot(light, light)))).astype(np.float32)
        self.links = [_link_constants(*pose) if pose is not None else None for _, _, pose, _, _ in OBJECTS]


@functools.cache
def scene():
    return Scene()


# -- Poses ---------------------------------------------------------------------------------------------------------


def _bt_matrix(q):
    """``btMatrix3x3::setRotation`` (and pybullet's ``getMatrixFromQuaternion``) of ``(..., 4)`` xyzw rows."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    d = x * x + y * y + z * z + w * w
    s = 2.0 / d
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz, zz = y * ys, y * zs, z * zs
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0], m[..., 0, 1], m[..., 0, 2] = 1.0 - (yy + zz), xy - wz, xz + wy
    m[..., 1, 0], m[..., 1, 1], m[..., 1, 2] = xy + wz, 1.0 - (xx + zz), yz - wx
    m[..., 2, 0], m[..., 2, 1], m[..., 2, 2] = xz - wy, yz + wx, 1.0 - (xx + yy)
    return m


def _qmul(a, b):
    """``btQuaternion operator*``."""
    ax, ay, az, aw = (a[..., i] for i in range(4))
    bx, by, bz, bw = (b[..., i] for i in range(4))
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by + ay * bw + az * bx - ax * bz,
            aw * bz + az * bw + ax * by - ay * bx,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        -1,
    )


def _quat_rotate(q, v):
    """``quatRotate``: ``(q * v) * q.inverse()`` with Bullet's operation order."""
    x, y, z, w = (q[..., i] for i in range(4))
    vx, vy, vz = (v[..., i] for i in range(3))
    px = w * vx + y * vz - z * vy
    py = w * vy + z * vx - x * vz
    pz = w * vz + x * vy - y * vx
    pw = -x * vx - y * vy - z * vz
    ix, iy, iz, iw = -x, -y, -z, w
    return np.stack(
        [
            pw * ix + px * iw + py * iz - pz * iy,
            pw * iy + py * iw + pz * ix - px * iz,
            pw * iz + pz * iw + px * iy - py * ix,
        ],
        -1,
    )


def _bt_rotation(m):
    """``btMatrix3x3::getRotation`` of one 3x3 matrix."""
    trace = m[0][0] + m[1][1] + m[2][2]
    q = [0.0] * 4
    if trace > 0.0:
        s = math.sqrt(trace + 1.0)
        q[3] = s * 0.5
        s = 0.5 / s
        q[0] = (m[2][1] - m[1][2]) * s
        q[1] = (m[0][2] - m[2][0]) * s
        q[2] = (m[1][0] - m[0][1]) * s
    else:
        i = (2 if m[1][1] < m[2][2] else 1) if m[0][0] < m[1][1] else (2 if m[0][0] < m[2][2] else 0)
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(m[i][i] - m[j][j] - m[k][k] + 1.0)
        q[i] = s * 0.5
        s = 0.5 / s
        q[3] = (m[k][j] - m[j][k]) * s
        q[j] = (m[j][i] + m[i][j]) * s
        q[k] = (m[k][i] + m[i][k]) * s
    return np.array(q)


def _link_constants(xyz, rpy):
    """``(rotParentToThis, rVector)`` of a fixed URDF joint, as ``URDF2Bullet`` and ``btMultibodyLink`` cache them."""
    phi, the, psi = rpy[0] / 2.0, rpy[1] / 2.0, rpy[2] / 2.0
    q = np.array(
        [
            math.sin(phi) * math.cos(the) * math.cos(psi) - math.cos(phi) * math.sin(the) * math.sin(psi),
            math.cos(phi) * math.sin(the) * math.cos(psi) + math.sin(phi) * math.cos(the) * math.sin(psi),
            math.cos(phi) * math.cos(the) * math.sin(psi) - math.sin(phi) * math.sin(the) * math.cos(psi),
            math.cos(phi) * math.cos(the) * math.cos(psi) + math.sin(phi) * math.sin(the) * math.sin(psi),
        ]
    )
    q = q * (1.0 / math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]))
    rot = _bt_rotation(_bt_matrix(q).T)
    rvec = np.array([-0.0, -0.0, -0.0]) + _quat_rotate(rot, np.asarray(xyz, np.float64))
    return rot, rvec


def model_matrices(body):
    """Float32 ``(B, O, 4, 4)`` model matrices of the TinyRenderer objects for ``(B, 3, 7)`` body poses.

    Each matrix is the link collider's world transform, as ``btMultiBody::updateCollisionObjectWorldTransforms``
    places it after ``resetBasePositionAndOrientation``, cast to float32 as TinyRenderer stores it.
    """
    body = np.asarray(body, np.float64)
    out = np.zeros((len(body), len(OBJECTS), 4, 4), np.float32)
    out[..., 3, 3] = 1.0
    links = scene().links
    for o, (_, which, pose, _, _) in enumerate(OBJECTS):
        position, q = body[:, which, :3], body[:, which, 3:]
        if pose is not None:
            rot, rvec = links[o]
            world_to_local = _qmul(np.broadcast_to(rot, q.shape), np.concatenate([-q[:, :3], q[:, 3:]], -1))
            q = np.concatenate([-world_to_local[:, :3], world_to_local[:, 3:]], -1)
            position = position + _quat_rotate(q, np.broadcast_to(rvec, position.shape))
        out[:, o, :3, :3] = _bt_matrix(q)
        out[:, o, :3, 3] = position
    return out


def _libm(fn, *arrays):
    """``fn`` of the C library, as pybullet calls it, element by element."""
    flat = [np.ravel(a) for a in arrays]
    return np.fromiter(map(fn, *flat), np.float64, len(flat[0])).reshape(np.shape(arrays[0]))


def _euler(q):
    """The ``getEulerFromQuaternion`` of pybullet for ``(B, 4)`` rows."""
    x, y, z, w = (q[:, i] for i in range(4))
    sqx, sqy, sqz, squ = x * x, y * y, z * z, w * w
    sarg = -2 * (x * z - w * y)
    out = np.empty((len(q), 3))
    low, high = sarg <= -0.99999, sarg >= 0.99999
    mid = ~(low | high)
    out[low, 0] = 0.0
    out[low, 1] = -0.5 * math.pi
    out[low, 2] = 2 * _libm(math.atan2, x[low], -y[low])
    out[high, 0] = 0.0
    out[high, 1] = 0.5 * math.pi
    out[high, 2] = 2 * _libm(math.atan2, -x[high], y[high])
    out[mid, 0] = _libm(math.atan2, 2 * (y * z + w * x)[mid], (squ - sqx - sqy + sqz)[mid])
    out[mid, 1] = _libm(math.asin, sarg[mid])
    out[mid, 2] = _libm(math.atan2, 2 * (x * y + w * z)[mid], (squ + sqx - sqy - sqz)[mid])
    return out


def _quaternion(rpy):
    """The ``getQuaternionFromEuler`` of pybullet for ``(B, 3)`` rows."""
    phi, the, psi = rpy[:, 0] / 2.0, rpy[:, 1] / 2.0, rpy[:, 2] / 2.0
    sp, cp = _libm(math.sin, phi), _libm(math.cos, phi)
    st, ct = _libm(math.sin, the), _libm(math.cos, the)
    ss, cs = _libm(math.sin, psi), _libm(math.cos, psi)
    q = np.stack(
        [
            sp * ct * cs - cp * st * ss,
            cp * st * cs + sp * ct * ss,
            cp * ct * ss - sp * st * cs,
            cp * ct * cs + sp * st * ss,
        ],
        -1,
    )
    length = np.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    return q / length[:, None]


def _normalized_f32(v):
    length = np.sqrt((v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1]) + v[:, 2] * v[:, 2])
    return v * (_ONE / length)[:, None]


def _cross_f32(a, b):
    return np.stack(
        [
            a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
            a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
        ],
        -1,
    )


def _dot_f32(a, b):
    return (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]) + a[:, 2] * b[:, 2]


def view_matrices(links):
    """Float32 ``(B, 4, 4)`` view matrices of ``(B, 7)`` camera link poses: ``render.frame_views`` without pybullet.

    The double-precision part follows ``render.camera_view`` (pybullet's quaternion helpers); the float32 part is
    ``b3ComputeViewMatrixFromPositions``.
    """
    links = np.asarray(links, np.float64)
    q = links[:, 3:7]
    eye = links[:, :3] + np.matmul(_bt_matrix(q), geometry.CAM_OFFSET)
    angles = _euler(q)
    angles[:, 1] += math.radians(geometry.CAM_PITCH_DEG)
    up = _bt_matrix(_quaternion(angles))[:, :, 2]
    eye32, target32, up32 = (np.asarray(a, np.float32) for a in (eye, links[:, :3], up))
    f = _normalized_f32(target32 - eye32)
    u = _normalized_f32(up32)
    s = _normalized_f32(_cross_f32(f, u))
    u = _cross_f32(s, f)
    out = np.zeros((len(links), 4, 4), np.float32)
    out[:, 0, :3], out[:, 1, :3], out[:, 2, :3] = s, u, -f
    out[:, 0, 3], out[:, 1, 3], out[:, 2, 3] = -_dot_f32(s, eye32), -_dot_f32(u, eye32), _dot_f32(f, eye32)
    out[:, 3, 3] = 1.0
    return out


def object_colors(colors):
    """Float32 ``(B, O, 4)`` colours of the objects from ``(B, 6, 4)`` ``render/rgba`` rows."""
    colors = np.asarray(colors, np.float64)
    rows = scene().color_rows
    out = np.ones((len(colors), len(rows), 4), np.float32)
    painted = rows >= 0
    out[:, painted] = colors[:, rows[painted]].astype(np.float32)
    return out


# -- Kernel ----------------------------------------------------------------------------------------------------------


@njit(cache=True, error_model="numpy", inline="always")
def _det2(a00, a01, a10, a11):
    return (_ZERO + a01 * (a10 * _M_ONE)) + a00 * a11


@njit(cache=True, error_model="numpy")
def _det3(m):
    d02 = _det2(m[1, 0], m[1, 1], m[2, 0], m[2, 1])
    d01 = _det2(m[1, 0], m[1, 2], m[2, 0], m[2, 2])
    d00 = _det2(m[1, 1], m[1, 2], m[2, 1], m[2, 2])
    return ((_ZERO + m[0, 2] * d02) + m[0, 1] * (d01 * _M_ONE)) + m[0, 0] * d00


@njit(cache=True, error_model="numpy")
def _invert_transpose(m, out, minor):
    """TinyRenderer's ``mat::invert_transpose``: the adjugate divided by its first row's dot with ``m``'s."""
    for i in range(4):
        for j in range(4):
            r = 0
            for a in range(4):
                if a == i:
                    continue
                c = 0
                for b in range(4):
                    if b == j:
                        continue
                    minor[r, c] = m[a, b]
                    c += 1
                r += 1
            d = _det3(minor)
            out[i, j] = d * _M_ONE if (i + j) % 2 else d
    tmp = (((_ZERO + out[0, 3] * m[0, 3]) + out[0, 2] * m[0, 2]) + out[0, 1] * m[0, 1]) + out[0, 0] * m[0, 0]
    for i in range(4):
        for j in range(4):
            out[i, j] = out[i, j] / tmp


@njit(cache=True, error_model="numpy")
def _matmul(a, b, out):
    for i in range(4):
        for j in range(4):
            out[i, j] = (((_ZERO + a[i, 3] * b[3, j]) + a[i, 2] * b[2, j]) + a[i, 1] * b[1, j]) + a[i, 0] * b[0, j]


@njit(cache=True, error_model="numpy", inline="always")
def _point(m, i, x, y, z):
    return (((_ZERO + m[i, 3] * _ONE) + m[i, 2] * z) + m[i, 1] * y) + m[i, 0] * x


@njit(cache=True, error_model="numpy", inline="always")
def _direction(m, i, x, y, z):
    return (((_ZERO + m[i, 3] * _ZERO) + m[i, 2] * z) + m[i, 1] * y) + m[i, 0] * x


@njit(cache=True, error_model="numpy", inline="always")
def _sqrt32(x):
    return F32(math.sqrt(np.float64(x)))


@njit(cache=True, error_model="numpy")
def _light_term(nx, ny, nz, light):
    """``D*diffuse + S*specular`` of Shader::fragment for an interpolated normal."""
    inv = _ONE / _sqrt32((nx * nx + ny * ny) + nz * nz)
    nx, ny, nz = nx * inv, ny * inv, nz * inv
    d = ((_ZERO + nz * light[2]) + ny * light[1]) + nx * light[0]
    k = d * _TWO
    rx, ry, rz = nx * k - light[0], ny * k - light[1], nz * k - light[2]
    rz = rz * (_ONE / _sqrt32((rx * rx + ry * ry) + rz * rz))
    r = rz if rz > _ZERO else _ZERO
    specular = F32(np.float64(r) * np.float64(r))
    diffuse = _ZERO if _ZERO > d else d
    return _DIFFUSE * diffuse + _SPECULAR * specular


@njit(cache=True, error_model="numpy", inline="always")
def _channel(c8, k):
    value = _AMBIENT * c8 + k * c8
    if value != value:
        return 0
    v = int(value)
    return v if v < 255 else 255


@njit(cache=True, error_model="numpy", inline="always")
def _wrap(u):
    f = u - F32(np.trunc(u))
    if f < _ZERO:
        f = f + _ONE
    return f


@njit(cache=True, error_model="numpy")
def _texel(tex, u, v, channel):
    h, w = tex.shape[0], tex.shape[1]
    fx = _wrap(u) * F32(w)
    fy = _wrap(v) * F32(h)
    x = int(fx) if fx == fx and fx < F32(2**30) else 0
    y = int(fy) if fy == fy and fy < F32(2**30) else 0
    x = 0 if x < 0 else (w - 1 if x >= w else x)
    y = 0 if y < 0 else (h - 1 if y >= h else y)
    return tex[y, x, channel]


@njit(cache=True, error_model="numpy")
def _shade(b0, b1, b2, nrm, uv, rgba, textured, tex, light, flat_k, flat, out, idx):
    """Shader::fragment for float32 barycentrics; writes RGB at ``out[idx]``.

    With ``flat`` set the triangle's three normals are equal and ``flat_k`` is its vertex shade. A pixel shade is then
    within 3e-4 of it per channel, so the vertex shade gives the truncation unless it lies within ``_FLAT_MARGIN`` of an
    integer; such channels take the per-pixel path.
    """
    u, v = _ZERO, _ZERO
    if textured:
        u = ((_ZERO + uv[2, 0] * b2) + uv[1, 0] * b1) + uv[0, 0] * b0
        v = ((_ZERO + uv[2, 1] * b2) + uv[1, 1] * b1) + uv[0, 1] * b0
    k = F32(np.nan)
    for c in range(3):
        base = _texel(tex, u, v, c) if textured else np.uint8(255)
        c8 = F32(np.uint8(F32(base) * rgba[c]))
        if flat:
            if c8 == _ZERO:
                out[idx + c] = 0
                continue
            value = _AMBIENT * c8 + flat_k * c8
            whole = F32(np.floor(value))
            if value - whole > _FLAT_MARGIN and value - whole < _ONE - _FLAT_MARGIN:
                v8 = int(whole)
                out[idx + c] = v8 if v8 < 255 else 255
                continue
        if k != k:
            nx = ((_ZERO + nrm[2, 0] * b2) + nrm[1, 0] * b1) + nrm[0, 0] * b0
            ny = ((_ZERO + nrm[2, 1] * b2) + nrm[1, 1] * b1) + nrm[0, 1] * b0
            nz = ((_ZERO + nrm[2, 2] * b2) + nrm[1, 2] * b1) + nrm[0, 2] * b0
            k = _light_term(nx, ny, nz, light)
        out[idx + c] = _channel(c8, k)


@njit(cache=True, error_model="numpy", inline="always")
def _screen(clip, i, half):
    x = (((_ZERO + half * clip[i, 3]) + _ZERO * clip[i, 2]) + _ZERO * clip[i, 1]) + half * clip[i, 0]
    y = (((_ZERO + half * clip[i, 3]) + _ZERO * clip[i, 2]) + half * clip[i, 1]) + _ZERO * clip[i, 0]
    w = (((_ZERO + _ONE * clip[i, 3]) + _ZERO * clip[i, 2]) + _ZERO * clip[i, 1]) + _ZERO * clip[i, 0]
    return x / w, y / w, w


@njit(cache=True, error_model="numpy")
def _barycentric_setup(pts):
    """The pixel-independent parts of ``barycentric``: ``(Ax, Ay, Cx-Ax, Bx-Ax, Cy-Ay, By-Ay, u.z)``."""
    ax, ay = np.float64(pts[0, 0]), np.float64(pts[0, 1])
    bx, by = np.float64(pts[1, 0]), np.float64(pts[1, 1])
    cx, cy = np.float64(pts[2, 0]), np.float64(pts[2, 1])
    s00, s01 = cx - ax, bx - ax
    s10, s11 = cy - ay, by - ay
    return ax, ay, s00, s01, s10, s11, s00 * s11 - s01 * s10


@njit(cache=True, error_model="numpy", inline="always")
def _span(lo, hi):
    """The integer range ``for (P = lo; P <= hi; P++)`` visits; ``lo`` is at least 0 and ``hi`` at most R - 1."""
    if lo != lo or hi != hi or lo >= F32(2**30) or hi < _ZERO:
        return 0, -1
    return int(lo), int(hi)


@njit(cache=True, error_model="numpy")
def _triangle(
    clip, org, clipped, nrm, uv, rgba, textured, tex, light, flat_k, flat, zbuf, img, R, near, far, pts, opts
):
    """Draw ``triangle`` (``clipped`` False) or ``triangleClipped`` of our_gl.cpp.

    ``clip`` holds the clip coordinates of the drawn triangle and ``org`` those of the face.
    """
    half = F32(R) / _TWO
    mnx, mny, mxx, mxy = _FLT_MAX, _FLT_MAX, -_FLT_MAX, -_FLT_MAX
    clamp = F32(R - 1)
    for i in range(3):
        sx, sy, sw = _screen(clip, i, half)
        pts[i, 0], pts[i, 1], pts[i, 2] = sx, sy, sw
        m = mnx if mnx < sx else sx
        mnx = _ZERO if _ZERO > m else m
        m = mny if mny < sy else sy
        mny = _ZERO if _ZERO > m else m
        m = mxx if mxx > sx else sx
        mxx = clamp if clamp < m else m
        m = mxy if mxy > sy else sy
        mxy = clamp if clamp < m else m
    x0, x1 = _span(mnx, mxx)
    y0, y1 = _span(mny, mxy)
    if x0 > x1 or y0 > y1:
        return
    ax, ay, s00, s01, s10, s11, uz = _barycentric_setup(pts)
    if not abs(uz) > 1e-2:
        return
    w0, w1, w2 = np.float64(pts[0, 2]), np.float64(pts[1, 2]), np.float64(pts[2, 2])
    z0, z1, z2 = np.float64(clip[0, 2]), np.float64(clip[1, 2]), np.float64(clip[2, 2])
    oax, oay, o00, o01, o10, o11, ouz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ow0, ow1, ow2 = 1.0, 1.0, 1.0
    if clipped:
        for i in range(3):
            sx, sy, sw = _screen(org, i, half)
            opts[i, 0], opts[i, 1], opts[i, 2] = sx, sy, sw
        oax, oay, o00, o01, o10, o11, ouz = _barycentric_setup(opts)
        ow0, ow1, ow2 = np.float64(opts[0, 2]), np.float64(opts[1, 2]), np.float64(opts[2, 2])
    near64, far64 = np.float64(near), np.float64(-far)
    for y in range(y0, y1 + 1):
        fy = np.float64(y)
        s12 = ay - fy
        for x in range(x0, x1 + 1):
            fx = np.float64(x)
            s02 = ax - fx
            ux = s01 * s12 - s02 * s11
            uy = s02 * s10 - s00 * s12
            b1 = uy / uz
            b2 = ux / uz
            if b1 < 0 or b2 < 0:
                continue
            b0 = 1.0 - (ux + uy) / uz
            if b0 < 0:
                continue
            c0, c1, c2 = b0 / w0, b1 / w1, b2 / w2
            total = (c0 + c1) + c2
            c0, c1, c2 = c0 / total, c1 / total, c2 / total
            depth = -1.0 * (((0.0 + z2 * c2) + z1 * c1) + z0 * c0)
            index = x + y * R
            if np.float64(zbuf[index]) > depth:
                continue
            if clipped:
                q12 = oay - fy
                q02 = oax - fx
                if abs(ouz) > 1e-2:
                    vx = o01 * q12 - q02 * o11
                    vy = q02 * o10 - o00 * q12
                    d0, d1, d2 = 1.0 - (vx + vy) / ouz, vy / ouz, vx / ouz
                else:
                    d0, d1, d2 = -1.0, 1.0, 1.0
                d0, d1, d2 = d0 / ow0, d1 / ow1, d2 / ow2
                total = (d0 + d1) + d2
                _shade(
                    F32(d0 / total), F32(d1 / total), F32(d2 / total),
                    nrm, uv, rgba, textured, tex, light, flat_k, False, img, index * 3,
                )  # fmt: skip
            else:
                if depth < far64 or depth > near64:
                    continue
                _shade(F32(c0), F32(c1), F32(c2), nrm, uv, rgba, textured, tex, light, flat_k, flat, img, index * 3)
            zbuf[index] = F32(depth)


@njit(cache=True, error_model="numpy", inline="always")
def _inside(clip, i):
    return clip[i, 3] > _ZERO and clip[i, 2] > -clip[i, 3]


@njit(cache=True, error_model="numpy")
def _clip_edge(tri, a, b, poly, count):
    """``clipEdge``: append the edge ``a``-``b`` of ``tri`` clipped to ``z > -w`` to ``poly[:count]``."""
    a_in, b_in = _inside(tri, a), _inside(tri, b)
    if not (a_in or b_in):
        return count
    for c in range(4):
        poly[count, c] = tri[a, c]
        poly[count + 1, c] = tri[b, c]
    if not (a_in and b_in):
        d0 = tri[a, 2] + tri[a, 3]
        d1 = tri[b, 2] + tri[b, 3]
        factor = F32(1.0 / np.float64(d1 - d0))
        row = count + 1 if a_in else count
        for c in range(4):
            poly[row, c] = (tri[a, c] * d1 - tri[b, c] * d0) * factor
    return count + 2


@njit(parallel=True, cache=True, error_model="numpy")
def _frames(verts, norms, uvs, faces, vstart, fstart, textured, tex, models, views, proj, colors, light, near, far, ss,
            fast, out):  # fmt: skip
    """Draw lane ``b`` of ``models``, ``views`` and ``colors`` into ``out[b]``, one lane per thread."""
    B, size = out.shape[0], out.shape[1]
    R = size * ss
    nobj = len(vstart) - 1
    vmax = 0
    for o in range(nobj):
        vmax = max(vmax, vstart[o + 1] - vstart[o])
    for b in prange(B):
        zbuf = np.full(R * R, -far, np.float32)
        img = np.full(R * R * 3, 255, np.uint8)
        clip = np.empty((vmax, 4), np.float32)
        world = np.empty((vmax, 3), np.float64)
        normal = np.empty((vmax, 3), np.float32)
        org = np.empty((3, 4), np.float32)
        fan = np.empty((3, 4), np.float32)
        poly = np.empty((6, 4), np.float32)
        nrm = np.empty((3, 3), np.float32)
        uv = np.empty((3, 2), np.float32)
        pts = np.empty((3, 3), np.float32)
        opts = np.empty((3, 3), np.float32)
        minor = np.empty((3, 3), np.float32)
        inv = np.empty((4, 4), np.float32)
        mv = np.empty((4, 4), np.float32)
        pmv = np.empty((4, 4), np.float32)
        view = views[b]
        _invert_transpose(view, inv, minor)
        ex, ey, ez = np.float64(inv[3, 0]), np.float64(inv[3, 1]), np.float64(inv[3, 2])
        for o in range(nobj):
            rgba = colors[b, o]
            if rgba[3] == _ZERO:
                continue
            model = models[b, o]
            _invert_transpose(model, inv, minor)
            _matmul(view, model, mv)
            _matmul(proj, mv, pmv)
            v0 = vstart[o]
            for v in range(vstart[o + 1] - v0):
                x, y, z = verts[v0 + v, 0], verts[v0 + v, 1], verts[v0 + v, 2]
                for i in range(4):
                    clip[v, i] = _point(pmv, i, x, y, z)
                for i in range(3):
                    world[v, i] = np.float64(_point(model, i, x, y, z))
                for i in range(3):
                    normal[v, i] = _direction(inv, i, norms[v0 + v, 0], norms[v0 + v, 1], norms[v0 + v, 2])
            for f in range(fstart[o], fstart[o + 1]):
                i0, i1, i2 = faces[f, 0], faces[f, 1], faces[f, 2]
                ax, ay, az = world[i1, 0] - world[i0, 0], world[i1, 1] - world[i0, 1], world[i1, 2] - world[i0, 2]
                bx, by, bz = world[i2, 0] - world[i0, 0], world[i2, 1] - world[i0, 1], world[i2, 2] - world[i0, 2]
                cx, cy, cz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
                if (world[i0, 0] - ex) * cx + (world[i0, 1] - ey) * cy + (world[i0, 2] - ez) * cz >= 0:
                    continue
                for k in range(3):
                    i = faces[f, k]
                    for c in range(4):
                        org[k, c] = clip[i, c]
                    for c in range(3):
                        nrm[k, c] = normal[i, c]
                    uv[k, 0], uv[k, 1] = uvs[v0 + i, 0], uvs[v0 + i, 1]
                if org[0, 3] < _ZERO and org[1, 3] < _ZERO and org[2, 3] < _ZERO:
                    continue
                if org[0, 3] >= _ZERO and org[1, 3] >= _ZERO and org[2, 3] >= _ZERO:
                    flat = (
                        fast
                        and org[0, 3] > _ZERO
                        and org[1, 3] > _ZERO
                        and org[2, 3] > _ZERO
                        and nrm[0, 0] == nrm[1, 0] == nrm[2, 0]
                        and nrm[0, 1] == nrm[1, 1] == nrm[2, 1]
                        and nrm[0, 2] == nrm[1, 2] == nrm[2, 2]
                    )
                    flat_k = _light_term(nrm[0, 0], nrm[0, 1], nrm[0, 2], light) if flat else _ZERO
                    _triangle(org, org, False, nrm, uv, rgba, textured[o], tex, light, flat_k, flat, zbuf, img, R,
                              near, far, pts, opts)  # fmt: skip
                    continue
                count = _clip_edge(org, 0, 1, poly, 0)
                count = _clip_edge(org, 1, 2, poly, count)
                count = _clip_edge(org, 2, 0, poly, count)
                for t in range(1, count - 1):
                    for c in range(4):
                        fan[0, c], fan[1, c], fan[2, c] = poly[0, c], poly[t, c], poly[t + 1, c]
                    _triangle(fan, org, True, nrm, uv, rgba, textured[o], tex, light, _ZERO, False, zbuf, img, R,
                              near, far, pts, opts)  # fmt: skip
        # TinyRenderer's row 0 is the bottom row; supersampled pixels average to the nearest integer, ties to even.
        n = ss * ss
        for r in range(size):
            for col in range(size):
                for c in range(3):
                    total = 0
                    for dy in range(ss):
                        row = R - 1 - (r * ss + dy)
                        for dx in range(ss):
                            total += np.int64(img[(row * R + col * ss + dx) * 3 + c])
                    q = total // n
                    rem = total - q * n
                    if 2 * rem > n or (2 * rem == n and q % 2 == 1):
                        q += 1
                    out[b, r, col, c] = q


@contextlib.contextmanager
def _threads(frames):
    import numba

    from ...render import cpu_raster

    cpu_raster.available()
    before = numba.get_num_threads()
    numba.set_num_threads(max(1, min(before, frames)))
    try:
        yield
    finally:
        numba.set_num_threads(before)


def draw(body, links, colors, size, ss, out=None, *, views=None, fast=True):
    """Draw ``(B, 3, 7)`` body poses, ``(B, 7)`` camera link poses and ``(B, 6, 4)`` colours to uint8 frames.

    ``views`` are the float32 view matrices of ``links`` when known. ``fast=False`` shades every pixel of a flat
    triangle; the frames are the same.
    """
    s = scene()
    batch = len(body)
    frames = np.empty((batch, size, size, 3), np.uint8) if out is None else out
    if batch == 0:
        return frames
    models = model_matrices(body)
    views = view_matrices(links) if views is None else np.ascontiguousarray(views, np.float32)
    rgba = object_colors(colors)
    with _threads(batch):
        _frames(
            s.verts, s.norms, s.uvs, s.faces, s.vstart, s.fstart, s.textured, s.texture,
            models, views, s.projection, rgba, s.light, s.near, s.far, int(ss), bool(fast), frames,
        )  # fmt: skip
    return frames
