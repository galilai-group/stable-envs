"""Scene extraction for rasterizers that follow MuJoCo's classic OpenGL renderer (render_gl3.c, render_context.c).

``extract`` reads one ``mjv_updateScene`` of a model and returns what ``mjr_render`` draws: each drawn geom with the
triangles of its display list, its material and texture mapping, the lights as the fixed-function pipeline sets
them, the averaged camera, the textures with their mipmaps, the skybox and the shadow-map settings. Quads split
into triangles as Mesa splits ``GL_QUADS``. The values are host NumPy arrays; a renderer adds the per-frame geom
poses and colours.
"""

from __future__ import annotations

import math

import numpy as np

PLANE, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, MESH = 0, 2, 3, 4, 5, 6, 7
SPOT, DIRECTIONAL = 0, 1
TEX_NONE, TEX_2D, TEX_CUBE = 0, 1, 2
MAXPLANEGRID = 200
TRANSPARENT_ALPHA = 0.995


class Scene:
    """Drawn geoms, lights, camera, textures and shadow settings of one model configuration."""


def _f(x):
    return float(np.float32(x))


def _quad_tris(quad):
    """Mesa's triangles of one ``GL_QUADS`` quad."""
    return ((quad[0], quad[1], quad[3]), (quad[1], quad[2], quad[3]))


class _Builder:
    """Triangles with per-corner positions, normals and 2-D texture coordinates in the unscaled display list."""

    def __init__(self):
        self.v, self.n, self.t = [], [], []

    def tri(self, corners):
        self.v.append([c[0] for c in corners])
        self.n.append([c[1] for c in corners])
        self.t.append([c[2] for c in corners])

    def quad(self, corners):
        for tri in _quad_tris(corners):
            self.tri(tri)

    def arrays(self):
        return (
            np.asarray(self.v, np.float64).reshape(-1, 3, 3),
            np.asarray(self.n, np.float64).reshape(-1, 3, 3),
            np.asarray(self.t, np.float64).reshape(-1, 3, 2),
        )


def _sphere_vertex(az, el, sign):
    v = (_f(math.cos(az) * math.cos(el)), _f(math.sin(az) * math.cos(el)), _f(sign + math.sin(el)))
    return v, (v[0], v[1], _f(v[2] - sign))


def sphere(b, nslice, nstack):
    pi = math.pi
    for sign in (-1, 1):
        el1 = _f((0.5 * pi * sign * (nstack // 2 - 1)) / _f(nstack // 2))
        v_el1 = _f(0.5 - sign * (nstack // 2 - 1) / nstack)
        v3_uv = 0.0 if sign > 0 else 1.0
        for j in range(nslice):
            az1, az2 = _f((2.0 * pi * j) / nslice), _f((2.0 * pi * (j + 1)) / nslice)
            p1, n1 = _sphere_vertex(az1, el1, 0)
            p2, n2 = _sphere_vertex(az2, el1, 0)
            p3 = n3 = (0.0, 0.0, float(sign))
            u1, u2 = az1 / (2 * pi), az2 / (2 * pi)
            u3 = (az1 + az2) / 2 / (2 * pi)
            c1, c2, c3 = (p1, n1, (u1, v_el1)), (p2, n2, (u2, v_el1)), (p3, n3, (u3, v3_uv))
            b.tri((c1, c2, c3) if sign > 0 else (c3, c2, c1))
    for sign in (-1, 1):
        for i in range(nstack // 2 - 1):
            el1 = _f((0.5 * pi * sign * i) / _f(nstack // 2))
            el2 = _f((0.5 * pi * sign * (i + 1)) / _f(nstack // 2))
            ve1, ve2 = _f(0.5 - sign * i / nstack), _f(0.5 - sign * (i + 1) / nstack)
            for j in range(nslice):
                az1, az2 = _f((2.0 * pi * j) / nslice), _f((2.0 * pi * (j + 1)) / nslice)
                u1, u2 = az1 / (2 * pi), az2 / (2 * pi)
                q = [
                    (*_sphere_vertex(az1, el1, 0), (u1, ve1)),
                    (*_sphere_vertex(az2, el1, 0), (u2, ve1)),
                    (*_sphere_vertex(az2, el2, 0), (u2, ve2)),
                    (*_sphere_vertex(az1, el2, 0), (u1, ve2)),
                ]
                b.quad(q if sign > 0 else q[::-1])


def half_sphere(b, sign, nslice, nstack, lift=0.0):
    pi = math.pi
    el1 = _f((pi / 2.0 * sign * (nstack - 1)) / nstack)
    for j in range(nslice):
        az1, az2 = _f((2.0 * pi * j) / nslice), _f((2.0 * pi * (j + 1)) / nslice)
        p1, n1 = _sphere_vertex(az1, el1, sign)
        p2, n2 = _sphere_vertex(az2, el1, sign)
        p3, n3 = (0.0, 0.0, 2.0 * sign), (0.0, 0.0, float(sign))
        v_el1, v3_uv = (nstack - 1) / nstack, 1.0
        if sign > 0:
            v_el1, v3_uv = 1.0 - v_el1, 0.0
        u1, u2 = az1 / (2 * pi), az2 / (2 * pi)
        u3 = (az1 + az2) / 2 / (2 * pi)
        c1, c2, c3 = (p1, n1, (u1, v_el1)), (p2, n2, (u2, v_el1)), (p3, n3, (u3, v3_uv))
        b.tri((c1, c2, c3) if sign > 0 else (c3, c2, c1))
    for i in range(nstack - 1):
        el1, el2 = _f((pi / 2.0 * sign * i) / nstack), _f((pi / 2.0 * sign * (i + 1)) / nstack)
        ve1, ve2 = i / nstack, (i + 1) / nstack
        if sign > 0:
            ve1, ve2 = 1.0 - ve1, 1.0 - ve2
        for j in range(nslice):
            az1, az2 = _f((2.0 * pi * j) / nslice), _f((2.0 * pi * (j + 1)) / nslice)
            u1, u2 = az1 / (2 * pi), az2 / (2 * pi)
            q = [
                (*_sphere_vertex(az1, el1, sign), (u1, ve1)),
                (*_sphere_vertex(az2, el1, sign), (u2, ve1)),
                (*_sphere_vertex(az2, el2, sign), (u2, ve2)),
                (*_sphere_vertex(az1, el2, sign), (u1, ve2)),
            ]
            b.quad(q if sign > 0 else q[::-1])


def disk(b, sign, nslice, nstack):
    pi = math.pi
    normal = (0.0, 0.0, float(-1 if sign == 0 else sign))

    def corner(az, r):
        p = (_f(math.cos(az) * r), _f(math.sin(az) * r), float(sign))
        return p, normal, (0.5 + 0.5 * p[0], 0.5 + 0.5 * p[1])

    r1 = _f(1.0 / nstack)
    for j in range(nslice):
        az1, az2 = _f((2.0 * pi * j) / nslice), _f((2.0 * pi * (j + 1)) / nslice)
        c1, c2 = corner(az1, r1), corner(az2, r1)
        c3 = ((0.0, 0.0, float(sign)), normal, (0.5, 0.5))
        b.tri((c1, c2, c3) if sign > 0 else (c3, c2, c1))
    for i in range(nstack - 1):
        ra, rb = _f((i + 1) / nstack), _f((i + 2) / nstack)
        for j in range(nslice):
            az1, az2 = _f((2.0 * pi * j) / nslice), _f((2.0 * pi * (j + 1)) / nslice)
            q = [corner(az1, rb), corner(az2, rb), corner(az2, ra), corner(az1, ra)]
            b.quad(q if sign > 0 else q[::-1])


def cylinder_side(b, nslice, nstack):
    pi = math.pi
    for i in range(nstack):
        h1, h2 = _f(2 * i / nstack - 1), _f(2 * (i + 1) / nstack - 1)
        w1, w2 = (1.0 - h1) / 2.0, (1.0 - h2) / 2.0
        for j in range(nslice):
            az1, az2 = _f((2.0 * pi * j) / nslice), _f((2.0 * pi * (j + 1)) / nslice)
            u1, u2 = j / nslice, (j + 1) / nslice
            q = []
            for az, h, u, w in ((az1, h1, u1, w1), (az2, h1, u2, w1), (az2, h2, u2, w2), (az1, h2, u1, w2)):
                p = (_f(math.cos(az)), _f(math.sin(az)), h)
                r = math.sqrt(p[0] * p[0] + p[1] * p[1])
                q.append((p, (p[0] / r, p[1] / r, 0.0), (u, w)))
            b.quad(q)


def box(b, numquads):
    d, nq = 2.0 / numquads, float(numquads)
    for x in range(numquads):
        for y in range(numquads):
            top = [(x / nq, 1 - y / nq), ((x + 1) / nq, 1 - y / nq), ((x + 1) / nq, 1 - (y + 1) / nq), (x / nq, 1 - (y + 1) / nq)]
            bot = [(x / nq, 1 - (y + 1) / nq), ((x + 1) / nq, 1 - (y + 1) / nq), ((x + 1) / nq, 1 - y / nq), (x / nq, 1 - y / nq)]
            a0, a1, b0, b1 = d * x - 1, d * (x + 1) - 1, d * y - 1, d * (y + 1) - 1
            faces = (
                ((0, 0, 1), top, ((a0, b0, 1), (a1, b0, 1), (a1, b1, 1), (a0, b1, 1))),
                ((0, 0, -1), bot, ((a0, b1, -1), (a1, b1, -1), (a1, b0, -1), (a0, b0, -1))),
                ((1, 0, 0), top, ((1, a0, b0), (1, a1, b0), (1, a1, b1), (1, a0, b1))),
                ((-1, 0, 0), bot, ((-1, a0, b1), (-1, a1, b1), (-1, a1, b0), (-1, a0, b0))),
                ((0, -1, 0), top, ((a0, -1, b0), (a1, -1, b0), (a1, -1, b1), (a0, -1, b1))),
                ((0, 1, 0), bot, ((a0, 1, b1), (a1, 1, b1), (a1, 1, b0), (a0, 1, b0))),
            )
            for normal, uvs, points in faces:
                b.quad([(p, normal, uv) for p, uv in zip(points, uvs)])


def plane(b, m, g):
    """The plane display list of geom ``g``: a grid with zero-width padding cells, as makePlane builds it."""
    zfar = float(m.vis.map.zfar * m.stat.extent)
    sz = [float(m.geom_size[g, 0]), float(m.geom_size[g, 1])]
    grid, nn = [None, None], [0, 0]
    for k in range(2):
        if sz[k] > 0:
            sz2 = max(float(m.geom_size[g, 2]), sz[k] / (MAXPLANEGRID - 2))
            n = int(math.floor(sz[k] / max(1e-15, sz2)))
            n -= n % 2
            pad = sz[k] - n * sz2
            n += 2
            gk = np.zeros(n + 1)
            for x in range(n):
                left = -sz[k] if x == 0 else -sz[k] + pad + (x - 1) * sz2 * 2
                right = left + (pad if x in (0, n - 1) else sz2 * 2)
                gk[x], gk[x + 1] = left, max(left, right)
            grid[k], nn[k] = gk, n
        else:
            mid = int(m.geom_matid[g])
            rep = float(m.mat_texrepeat[mid, k]) if mid >= 0 else 0.0
            step = 2 / rep if rep > 0 else 2.1 * zfar / (MAXPLANEGRID - 2)
            d = (2.1 * zfar + 2 * step) / MAXPLANEGRID
            grid[k] = np.array([d * x - d * (MAXPLANEGRID // 2) for x in range(MAXPLANEGRID + 1)])
            nn[k] = MAXPLANEGRID
    up = (0.0, 0.0, 1.0)
    for x in range(nn[0]):
        if grid[0][x + 1] <= grid[0][x]:
            continue
        for y in range(nn[1]):
            if grid[1][y + 1] <= grid[1][y]:
                continue
            if sz[0] > 0:
                u0, u1 = ((grid[0][x + i] + sz[0]) / (2.0 * sz[0]) for i in (0, 1))
            else:
                u0, u1 = 0.5 * grid[0][x], 0.5 * grid[0][x + 1]
            if sz[1] > 0:
                v0, v1 = (1.0 - (grid[1][y + i] + sz[1]) / (2.0 * sz[1]) for i in (0, 1))
            else:
                v0, v1 = -0.5 * grid[1][y], -0.5 * grid[1][y + 1]
            b.quad(
                [
                    ((grid[0][x], grid[1][y], 0.0), up, (u0, v0)),
                    ((grid[0][x + 1], grid[1][y], 0.0), up, (u1, v0)),
                    ((grid[0][x + 1], grid[1][y + 1], 0.0), up, (u1, v1)),
                    ((grid[0][x], grid[1][y + 1], 0.0), up, (u0, v1)),
                ]
            )


def mesh(m, meshid):
    """Triangles of a mesh with the corner normals mjr_uploadMesh gives them, and its texture coordinates."""
    va, na, ta = int(m.mesh_vertadr[meshid]), int(m.mesh_normaladr[meshid]), int(m.mesh_texcoordadr[meshid])
    fa, nf = int(m.mesh_faceadr[meshid]), int(m.mesh_facenum[meshid])
    faces = np.asarray(m.mesh_face[fa : fa + nf], np.int64)
    vert = np.asarray(m.mesh_vert, np.float32)[va + faces]
    corner = np.asarray(m.mesh_normal, np.float32)[na + np.asarray(m.mesh_facenormal[fa : fa + nf], np.int64)]
    if ta >= 0:
        uv = np.asarray(m.mesh_texcoord, np.float32)[ta + np.asarray(m.mesh_facetexcoord[fa : fa + nf], np.int64)]
        normals = corner
    else:
        uv = np.zeros((nf, 3, 2), np.float32)
        e1 = (vert[:, 1] - vert[:, 0]).astype(np.float32)
        e2 = (vert[:, 2] - vert[:, 0]).astype(np.float32)
        face = np.cross(e1, e2).astype(np.float32)
        face /= np.maximum(np.linalg.norm(face, axis=-1, keepdims=True), np.float32(1e-15))
        keep = (corner * face[:, None, :]).sum(-1) >= 0.8
        normals = np.where(keep[..., None], corner, face[:, None, :])
    return vert.astype(np.float64), normals.astype(np.float64), uv.astype(np.float64), ta >= 0


def _scaled(v, n, size):
    """Vertices and normals after ``glScalef(size)`` with ``GL_NORMALIZE``."""
    size = np.asarray(size, np.float64)
    n = n / size
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-30)
    return v * size, n


def geometry(m, kind, size, dataid, objid):
    """Triangles of one drawn geom in its frame: scaled positions, normals, unscaled object coordinates and UVs.

    ``size`` is the ``mjvGeom`` size (for example radius, radius, half-length for cylinders and capsules).
    """
    q = m.vis.quality
    size = np.asarray(size, np.float32).astype(np.float64)
    b = _Builder()
    if kind == MESH:
        v, n, uv, _ = mesh(m, dataid // 2)
        return v, n, v.copy(), uv
    if kind == PLANE:
        if dataid < 0:
            raise NotImplementedError("planes without a grid display list are not drawn")
        plane(b, m, objid)
        v, n, uv = b.arrays()
        return v, n, v.copy(), uv
    if kind == BOX:
        box(b, int(q.numquads))
        v, n, uv = b.arrays()
        sv, sn = _scaled(v, n, size)
        return sv, sn, v, uv
    if kind in (SPHERE, ELLIPSOID):
        sphere(b, int(q.numslices), int(q.numstacks))
        v, n, uv = b.arrays()
        sv, sn = _scaled(v, n, size)
        return sv, sn, v, uv
    if kind == CYLINDER:
        disk(b, -1, int(q.numslices), int(q.numstacks) // 2)
        cylinder_side(b, int(q.numslices), int(q.numstacks))
        disk(b, 1, int(q.numslices), int(q.numstacks) // 2)
        v, n, uv = b.arrays()
        sv, sn = _scaled(v, n, size)
        return sv, sn, v, uv
    if kind == CAPSULE:
        r, h = float(size[0]), float(size[2])
        s3 = np.array([r, r, h])
        side = _Builder()
        cylinder_side(side, int(q.numslices), int(q.numstacks))
        v0, n0, t0 = side.arrays()
        sv0, sn0 = _scaled(v0, n0, s3)
        av = _f(0.5 * (float(size[0]) + float(size[1])))
        caps = np.array([r, r, h * av / h])
        parts = [(sv0, sn0, v0, t0)]
        for sign, lift in ((1, (h - av) / av), (-1, (h - av) / av + 2 * (av - h) / av)):
            hb = _Builder()
            half_sphere(hb, sign, int(q.numslices), int(q.numstacks) // 2)
            v1, n1, t1 = hb.arrays()
            v1 = v1 + np.array([0.0, 0.0, lift])
            sv1, sn1 = _scaled(v1, n1, caps)
            parts.append((sv1, sn1, v1, t1))
        return tuple(np.concatenate([p[k] for p in parts]) for k in range(4))
    raise NotImplementedError(f"no display list for geom type {kind}")


def _texture_levels(data):
    """Mipmap chain of an (H, W, C) uint8 image with even sides: rounded 2x2 box averages, down to 1x1."""
    levels = [data]
    cur = data.astype(np.uint32)
    while cur.shape[0] > 1 or cur.shape[1] > 1:
        if cur.shape[0] % 2 or cur.shape[1] % 2:
            raise NotImplementedError("mipmaps of textures with odd sides are not drawn")
        cur = (cur[0::2, 0::2] + cur[1::2, 0::2] + cur[0::2, 1::2] + cur[1::2, 1::2] + 2) // 4
        levels.append(cur.astype(np.uint8))
    return levels


def texture(m, t):
    """Mipmap chains of texture ``t``: one chain for a 2-D texture, six (+X, -X, +Y, -Y, +Z, -Z) for cube maps."""
    w, h, nc = int(m.tex_width[t]), int(m.tex_height[t]), int(m.tex_nchannel[t])
    data = np.asarray(m.tex_data[int(m.tex_adr[t]) : int(m.tex_adr[t]) + w * h * nc], np.uint8).reshape(h, w, nc)
    data = np.ascontiguousarray(data[..., :3])
    if int(m.tex_type[t]) == 0:
        if int(m.tex_colorspace[t]) == 2:
            raise NotImplementedError("sRGB 2-D textures are not drawn")
        return [_texture_levels(data)]
    faces = [data] * 6 if w == h else [data[i * w : (i + 1) * w] for i in range(6)]
    return [_texture_levels(np.ascontiguousarray(f)) for f in faces]


def average_camera(scn):
    """``mjv_averageCamera`` of the two scene cameras."""
    c0, c1 = scn.camera[0], scn.camera[1]
    avg = {k: 0.5 * (np.asarray(getattr(c0, k), np.float64) + np.asarray(getattr(c1, k), np.float64)) for k in ("pos", "forward", "up")}
    f = avg["forward"] / np.linalg.norm(avg["forward"])
    u = avg["up"] - f * float(np.dot(avg["up"], f))
    avg["forward"], avg["up"] = f, u / np.linalg.norm(u)
    for k in ("frustum_center", "frustum_width", "frustum_bottom", "frustum_top", "frustum_near", "frustum_far", "orthographic"):
        avg[k] = 0.5 * (float(getattr(c0, k)) + float(getattr(c1, k)))
    return avg


def look_at(eye, forward, up):
    """Rows of ``mjr_lookAt``: right, up and back axes, and the translation."""
    f = np.asarray(forward, np.float64)
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    u /= np.linalg.norm(u)
    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = s, u, -f
    view[:3, 3] = [-s @ eye, -u @ eye, f @ eye]
    return view


def ortho_vec(v):
    """``mjr_orthoVec``: the shadow camera's up direction."""
    res = np.cross(v, np.array([-1.0, 0.0, 0.0]))
    if float(res @ res) > 0.01:
        return res / np.linalg.norm(res)
    res = np.cross(v, np.array([0.0, 1.0, 0.0]))
    return res / np.linalg.norm(res)


def geom_rgba(m, g):
    """The colour ``mjv_updateScene`` gives model geom ``g``: its own RGBA unless that is the default and it has a material."""
    rgba = np.asarray(m.geom_rgba[g], np.float32)
    mat = int(m.geom_matid[g])
    if mat >= 0 and np.array_equal(rgba, np.array([0.5, 0.5, 0.5, 1.0], np.float32)):
        return np.asarray(m.mat_rgba[mat], np.float32)
    return rgba


def extract(m, d, camera, option=None):
    """The scene ``mjr_render`` draws from fixed camera ``camera`` (name or id) after ``mjv_updateScene``.

    Raises NotImplementedError for features the scene uses that are not described here: reflections,
    non-geom objects, unsupported light types with shadows and textures on geoms without a texture mapping.
    """
    import mujoco

    option = mujoco.MjvOption() if option is None else option
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = camera if isinstance(camera, int) else m.camera(camera).id
    scn = mujoco.MjvScene(m, maxgeom=max(1000, 2 * m.ngeom))
    mujoco.mjv_updateScene(m, d, option, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
    geoms = [scn.geoms[i] for i in range(scn.ngeom)]
    if any(int(g.objtype) != int(mujoco.mjtObj.mjOBJ_GEOM) for g in geoms):
        raise NotImplementedError("only model geoms are drawn")
    s = Scene()
    s.objid = np.array([g.objid for g in geoms], np.int64)
    s.type = np.array([g.type for g in geoms], np.int64)
    s.category = np.array([g.category for g in geoms], np.int64)
    s.dataid = np.array([g.dataid for g in geoms], np.int64)
    s.matid = np.array([g.matid for g in geoms], np.int64)
    s.size = np.array([g.size for g in geoms], np.float32)
    s.rgba = np.array([g.rgba for g in geoms], np.float32)
    s.specular = np.array([g.specular for g in geoms], np.float64)
    s.shininess = np.array([g.shininess for g in geoms], np.float64)
    s.emission = np.array([g.emission for g in geoms], np.float64)
    s.reflectance = np.array([g.reflectance for g in geoms], np.float64)
    s.rbound = np.array([g.modelrbound for g in geoms], np.float32)
    s.texid = np.array([g.texid for g in geoms], np.int64)
    s.texcoord = np.array([g.texcoord for g in geoms], np.int64)
    s.transparent = s.rgba[:, 3] < TRANSPARENT_ALPHA
    if np.any(np.isin(s.type, (PLANE, BOX)) & ~s.transparent & (s.reflectance > 0)):
        raise NotImplementedError("reflective planes and boxes are not drawn")
    body = m.geom_bodyid[s.objid]
    s.static = (m.body_weldid[body] == 0) & (m.body_mocapid[body] < 0)
    model_rgba = np.array([geom_rgba(m, int(g)) for g in s.objid], np.float32)
    if not np.allclose(model_rgba, s.rgba):
        raise RuntimeError("scene colours differ from the model colour rule")
    # Per geom: triangles in the geom frame and the texture mapping MuJoCo sets for it.
    s.verts, s.normals, s.texcoords, s.texmode, s.texslot = [], [], [], [], []
    s.textures = []
    slots = {}
    for i, g in enumerate(geoms):
        kind = int(g.type)
        v, n, obj, uv = geometry(m, kind, s.size[i], int(g.dataid), int(g.objid))
        mid = int(g.matid)
        tid = int(g.texid) if int(g.texid) >= 0 else (int(m.mat_texid[mid, 1]) if mid >= 0 else -1)
        mode, tc = TEX_NONE, np.zeros(v.shape, np.float64)
        if tid >= 0:
            uniform = bool(m.mat_texuniform[mid]) if mid >= 0 else False
            rep = np.asarray(m.mat_texrepeat[mid], np.float64) if mid >= 0 else np.zeros(2)
            size = s.size[i].astype(np.float64)
            tex2d = int(m.tex_type[tid]) == 0
            builtin_uv = kind in (PLANE, SPHERE, ELLIPSOID, BOX, CYLINDER, CAPSULE)
            if tex2d and (int(g.texcoord) or builtin_uv):
                scl = np.where(rep > 0, rep, 1.0)
                if uniform:
                    scl = np.where(size[:2] > 0, scl * size[:2], scl)
                tc[..., :2] = uv * scl
                if kind == PLANE and (size[0] <= 0 or size[1] <= 0):
                    tc[..., :2] -= 0.5 * scl
                mode = TEX_2D
            elif tex2d:
                scl = rep.copy()
                if int(g.dataid) >= 0:
                    scl = np.where(size[:2] > 0, scl / np.maximum(1e-15, size[:2]), scl)
                if uniform:
                    scl = np.where(size[:2] > 0, scl * size[:2], scl)
                tc[..., 0] = 0.5 * scl[0] * obj[..., 0] - 0.5
                tc[..., 1] = -0.5 * scl[1] * obj[..., 1] - 0.5
                mode = TEX_2D
            else:
                tc = obj * (size if uniform else 1.0)
                mode = TEX_CUBE
            if tid not in slots:
                slots[tid] = len(s.textures)
                s.textures.append(texture(m, tid))
        s.verts.append(v)
        s.normals.append(n)
        s.texcoords.append(tc)
        s.texmode.append(mode)
        s.texslot.append(slots.get(tid, -1) if mode != TEX_NONE else -1)
    s.texmode = np.asarray(s.texmode, np.int64)
    s.texslot = np.asarray(s.texslot, np.int64)
    # Lights in the order mjr_render enables them; a spot cutoff GL rejects (above 90 degrees) stays 180.
    lights = []
    for i in range(min(scn.nlight, 8)):
        light = scn.lights[i]
        kind = int(light.type)
        if kind not in (SPOT, DIRECTIONAL):
            continue
        cutoff = float(light.cutoff)
        if kind == DIRECTIONAL or cutoff > 90.0:
            cutoff = 180.0
        lights.append(
            {
                "type": kind,
                "headlight": bool(light.headlight),
                "castshadow": bool(light.castshadow),
                "pos": np.asarray(light.pos, np.float64).copy(),
                "dir": np.asarray(light.dir, np.float64).copy(),
                "ambient": np.asarray(light.ambient, np.float64).copy(),
                "diffuse": np.asarray(light.diffuse, np.float64).copy(),
                "specular": np.asarray(light.specular, np.float64).copy(),
                "attenuation": np.asarray(light.attenuation, np.float64).copy(),
                "cutoff": cutoff,
                "exponent": float(light.exponent),
            }
        )
    s.lights = lights
    s.global_ambient = 0.0 if lights else 0.3
    s.camera = average_camera(scn)
    s.headpos = s.camera["pos"].astype(np.float32).astype(np.float64)
    s.shadow_size = int(m.vis.quality.shadowsize)
    s.shadow_clip = float(m.stat.extent * m.vis.map.shadowclip)
    s.shadow_scale = float(m.vis.map.shadowscale)
    s.sky = None
    for t in range(m.ntex):
        if int(m.tex_type[t]) == 2:
            s.sky = texture(m, t)
            break
    s.geom_pos = np.array([g.pos for g in geoms], np.float64)
    s.geom_mat = np.array([g.mat for g in geoms], np.float64).reshape(len(geoms), 9)
    return s


def shadow_matrix(s, light):
    """3x4 map from world position to the shadow map's window x, y (texels, viewport offset 1) and depth.

    The depth is the reverse-Z window depth of ``ARB_clip_control``: 1 at the camera's near distance, 0 at far.
    """
    if light["type"] != DIRECTIONAL:
        raise NotImplementedError("shadows from spot lights are not drawn")
    view = look_at(light["pos"], light["dir"], ortho_vec(light["dir"]))
    near, far = s.camera["frustum_near"], s.camera["frustum_far"]
    clip, n = s.shadow_clip, s.shadow_size
    inner = n - 2
    out = np.zeros((3, 4))
    out[0] = view[0] * (0.5 * inner / clip)
    out[0, 3] += 0.5 * inner + 1.0
    out[1] = view[1] * (0.5 * inner / clip)
    out[1, 3] += 0.5 * inner + 1.0
    # z_win = 0.5 - 0.5 * (-2 / (f - n) * z_eye - (f + n) / (f - n))
    out[2] = view[2] * (1.0 / (far - near))
    out[2, 3] += 0.5 + 0.5 * (far + near) / (far - near)
    return out


def shadow_lookup(s, light):
    """3x4 map from world position to the shadow texture coordinates s, t (in texels of the full map) and r."""
    win = shadow_matrix(s, light)
    n, inner = s.shadow_size, s.shadow_size - 2
    out = win.copy()
    # Window x = (x_ndc + 1) * inner / 2 + 1; lookup s * n = (x_ndc + 1) * n / 2.
    for k in range(2):
        out[k] = (win[k] - np.array([0.0, 0.0, 0.0, 1.0])) * (n / inner)
    return out
