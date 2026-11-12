"""Scene extraction for rasterizers that follow MuJoCo's classic OpenGL renderer (render_gl3.c, render_context.c).

``extract`` reads one ``mjv_updateScene`` of a model and returns what ``mjr_render`` draws: each drawn geom with the
triangles of its display list, its material and texture mapping, the lights as the fixed-function pipeline sets
them, the averaged camera, the textures with their mipmaps, the skybox and the shadow-map settings. Quads split
into triangles as Mesa splits ``GL_QUADS``. The values are host NumPy arrays; a renderer adds the per-frame geom
poses and colours.

``pack`` turns an extracted scene into the flat arrays that the CPU and CUDA backends of ``render.mujoco_raster``
read: corners, triangles, materials, textures, light rows, camera row and shadow matrices.

The extraction follows the MuJoCo versions in ``VERIFIED``. Each version lists the scene features whose frames
were compared with that version's OpenGL renderer; a scene that needs another feature raises NotImplementedError.
"""

from __future__ import annotations

import math

import numpy as np

PLANE, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, MESH = 0, 2, 3, 4, 5, 6, 7
SPOT, DIRECTIONAL = 0, 1
TEX_NONE, TEX_2D, TEX_CUBE = 0, 1, 2
MAXPLANEGRID = 200
TRANSPARENT_ALPHA = 0.995
POW_STEPS = 4096
#: Width of a light row: type, role, direction or position, ambient, diffuse, specular, attenuation, spot
#: direction, spot cutoff cosine, exponent and the headlight flag.
LIGHT_ROW = 23
#: Scene features checked against the OpenGL renderer of each MuJoCo minor version (Fetch pins 3.11, Cube and
#: Scene pin 3.17).
VERIFIED = {
    "3.11": frozenset({"mesh", "primitive", "plane", "transparency", "skybox", "light"}),
    "3.17": frozenset({"mesh", "primitive", "plane", "transparency", "skybox", "light", "shadow", "texture"}),
}


def mujoco_version():
    """The installed MuJoCo minor version, for example ``"3.17"``."""
    from importlib.metadata import version

    return ".".join(version("mujoco").split(".")[:2])


def require(s, feature):
    """Raise NotImplementedError unless ``feature`` is verified for the MuJoCo version of scene ``s``."""
    if feature not in VERIFIED.get(s.version, frozenset()):
        raise NotImplementedError(f"MuJoCo {s.version}: the {feature} rule of the OpenGL renderer is not verified")


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
            top = [
                (x / nq, 1 - y / nq),
                ((x + 1) / nq, 1 - y / nq),
                ((x + 1) / nq, 1 - (y + 1) / nq),
                (x / nq, 1 - (y + 1) / nq),
            ]
            bot = [
                (x / nq, 1 - (y + 1) / nq),
                ((x + 1) / nq, 1 - (y + 1) / nq),
                ((x + 1) / nq, 1 - y / nq),
                (x / nq, 1 - y / nq),
            ]
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
    avg = {
        k: 0.5 * (np.asarray(getattr(c0, k), np.float64) + np.asarray(getattr(c1, k), np.float64))
        for k in ("pos", "forward", "up")
    }
    f = avg["forward"] / np.linalg.norm(avg["forward"])
    u = avg["up"] - f * float(np.dot(avg["up"], f))
    avg["forward"], avg["up"] = f, u / np.linalg.norm(u)
    for k in (
        "frustum_center",
        "frustum_width",
        "frustum_bottom",
        "frustum_top",
        "frustum_near",
        "frustum_far",
        "orthographic",
    ):
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
    s.version = mujoco_version()
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
    kinds = set(s.type.tolist())
    for feature, present in (
        ("mesh", MESH in kinds),
        ("plane", PLANE in kinds),
        ("primitive", bool(kinds - {MESH, PLANE})),
        ("transparency", bool(s.transparent.any())),
        ("texture", bool((s.texmode != TEX_NONE).any())),
    ):
        if present:
            require(s, feature)
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
                "id": int(getattr(light, "id", -1)),
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
    if lights:
        require(s, "light")
    s.sky = None
    for t in range(m.ntex):
        if int(m.tex_type[t]) == 2:
            require(s, "skybox")
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
    require(s, "shadow")
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


class Packed:
    """Flat host arrays of one extracted scene, read by both backends of ``render.mujoco_raster``.

    Geometry: ``cpos``, ``cnor``, ``ctex`` (corners), ``tris`` (corner indices), ``cstarts`` and ``starts`` (first
    corner and first triangle of each geom) and ``rbound`` (largest corner distance from the geom frame).
    ``material`` rows are specular, shininess (OpenGL exponent), emission, ``powlut`` row, texture mode and texture
    slot. ``lights`` rows follow ``light_rows``. ``cam`` is position, right, up and forward; ``tanh`` the tangent
    of half the vertical field of view. ``smat`` and ``slook`` map world points to the shadow map (see
    ``shadow_matrix`` and ``shadow_lookup``); ``shadow_light`` is -1 without one.
    """


def pack_geoms(s):
    """Corners and triangles of every drawn geom.

    Corners that share a position, normal and texture coordinate are transformed and lit once per frame.
    """
    cpos, cnor, ctex, tris, cstarts, starts = [], [], [], [], [0], [0]
    for v, n, t in zip(s.verts, s.normals, s.texcoords):
        key = np.concatenate([v.reshape(-1, 3), n.reshape(-1, 3), t.reshape(-1, 3)], 1)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        cpos.append(uniq[:, :3])
        cnor.append(uniq[:, 3:6])
        ctex.append(uniq[:, 6:9])
        tris.append(inv.reshape(-1, 3) + cstarts[-1])
        cstarts.append(cstarts[-1] + len(uniq))
        starts.append(starts[-1] + len(v))
    cpos = np.ascontiguousarray(np.concatenate(cpos), np.float64)
    cstarts = np.asarray(cstarts, np.int64)
    rbound = np.array([np.linalg.norm(cpos[cstarts[i] : cstarts[i + 1]], axis=1).max() for i in range(len(s.verts))])
    return (
        cpos,
        np.ascontiguousarray(np.concatenate(cnor), np.float64),
        np.ascontiguousarray(np.concatenate(ctex), np.float64),
        np.ascontiguousarray(np.concatenate(tris), np.int64),
        cstarts,
        np.asarray(starts, np.int64),
        rbound,
    )


def pack_textures(textures):
    """Mip levels of every texture face in one float array.

    ``levels`` rows are (offset, width, height) and ``info`` rows are (first level row, faces, levels per face).
    """
    data, levels, info = [], [], []
    offset = 0
    for chains in textures:
        info.append((len(levels), len(chains), len(chains[0])))
        for chain in chains:
            for level in chain:
                h, w = level.shape[:2]
                data.append(level.reshape(-1).astype(np.float32) / np.float32(255.0))
                levels.append((offset, w, h))
                offset += h * w * 3
    if not data:
        return np.zeros(3, np.float32), np.zeros((1, 3), np.int64), np.zeros((1, 3), np.int64)
    return np.concatenate(data), np.asarray(levels, np.int64), np.asarray(info, np.int64)


def light_rows(s, shadow):
    """Light rows of the lights of ``s``; light ``shadow`` gets the shadow role.

    Columns: type, role (0 first pass, 1 shadow light), direction or position, ambient, diffuse, specular,
    attenuation, spot direction, spot cutoff cosine (-2 for none), exponent and the headlight flag.
    """
    out = np.zeros((len(s.lights), LIGHT_ROW))
    for i, light in enumerate(s.lights):
        out[i, 0] = light["type"]
        out[i, 1] = 1.0 if i == shadow else 0.0
        if light["type"] == DIRECTIONAL:
            v = -light["dir"]
            out[i, 2:5] = v / max(np.linalg.norm(v), 1e-30)
            out[i, 14:17] = (1.0, 0.0, 0.0)
        else:
            out[i, 2:5] = light["pos"]
            out[i, 14:17] = light["attenuation"]
        out[i, 5:8] = light["ambient"]
        out[i, 8:11] = light["diffuse"]
        out[i, 11:14] = light["specular"]
        sd = light["dir"]
        out[i, 17:20] = sd / max(np.linalg.norm(sd), 1e-30)
        out[i, 20] = -2.0 if light["cutoff"] >= 180.0 else math.cos(math.radians(light["cutoff"]))
        out[i, 21] = light["exponent"]
        out[i, 22] = 1.0 if light["headlight"] else 0.0
    return out


def camera_row(pos, forward, up):
    """Camera row: position, right, up and forward, as ``mjr_lookAt`` builds the view."""
    f = np.asarray(forward, np.float64)
    right = np.cross(f, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    up /= np.linalg.norm(up)
    return np.concatenate([np.asarray(pos, np.float64), right, up, f]).astype(np.float64)


def headlight_rows(rows, cams):
    """Per-frame light rows (B, L, LIGHT_ROW) whose headlight shines along each frame's camera ``cams`` (B, 12)."""
    out = np.repeat(np.asarray(rows, np.float64)[None], len(cams), 0)
    for i in np.flatnonzero(rows[:, 22] > 0.5):
        if int(rows[i, 0]) != DIRECTIONAL:
            raise NotImplementedError("a spot headlight is not drawn")
        f = cams[:, 9:12]
        out[:, i, 2:5] = -f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-30)
        out[:, i, 17:20] = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-30)
    return out


def pack(s, cast=None):
    """Packed arrays of extracted scene ``s``.

    ``cast`` is another extraction of the same geoms whose triangles cast the shadows (for example exact meshes
    when ``s`` draws simplified ones); by default the drawn triangles cast them.
    """
    p = Packed()
    p.version = s.version
    p.objid = s.objid.copy()
    if any(light["castshadow"] and light["type"] != DIRECTIONAL for light in s.lights):
        raise NotImplementedError("shadows of spot lights are not drawn")
    shadow = [i for i, light in enumerate(s.lights) if light["castshadow"]]
    if len(shadow) > 1:
        raise NotImplementedError("one shadow light is drawn")
    p.shadow_light = shadow[0] if shadow else -1
    p.cpos, p.cnor, p.ctex, p.tris, p.cstarts, p.starts, p.rbound = pack_geoms(s)
    if cast is None:
        p.cast = (p.cpos, p.tris, p.cstarts, p.starts, p.rbound)
    else:
        if not np.array_equal(cast.objid, s.objid):
            raise ValueError("the casting scene draws other geoms")
        c = pack_geoms(cast)
        p.cast = (c[0], c[3], c[4], c[5], c[6])
    shins, row = np.unique(s.shininess * 128.0, return_inverse=True)
    grid = np.linspace(0.0, 1.0, POW_STEPS + 1)
    p.powlut = np.ascontiguousarray(grid[None, :] ** np.minimum(shins, 128.0)[:, None], np.float64)
    p.material = np.stack(
        [s.specular, s.shininess * 128.0, s.emission, row.reshape(-1).astype(np.float64), s.texmode, s.texslot], 1
    ).astype(np.float64)
    p.texdata, p.texlevels, p.texinfo = pack_textures(s.textures)
    p.textures = s.textures
    p.sky = s.sky
    p.sky_tex = pack_textures([s.sky] if s.sky is not None else [])
    p.has_sky = s.sky is not None
    p.static = s.static.copy()
    p.transparent = s.transparent.copy()
    p.casters = np.flatnonzero(~s.static).astype(np.int64)
    p.sort_rbound = s.rbound.astype(np.float32)
    p.rgba = s.rgba.astype(np.float64)
    p.type = s.type.copy()
    p.texmode = s.texmode.copy()
    p.texslot = s.texslot.copy()
    p.size = s.size.copy()
    cam = s.camera
    if cam["orthographic"] or cam["frustum_center"] != 0.0 or cam["frustum_width"] != 0.0:
        raise NotImplementedError("symmetric perspective cameras only are drawn")
    p.cam = camera_row(cam["pos"], cam["forward"], cam["up"])
    p.tanh = float(cam["frustum_top"]) / float(cam["frustum_near"])
    p.near = float(cam["frustum_near"])
    p.far = float(cam["frustum_far"])
    p.lights = light_rows(s, p.shadow_light)
    p.light_ids = np.array([light["id"] for light in s.lights], np.int64)
    p.ambient = float(s.global_ambient)
    p.nshadow = int(s.shadow_size)
    if p.shadow_light >= 0:
        light = s.lights[p.shadow_light]
        p.smat = shadow_matrix(s, light)
        p.slook = shadow_lookup(s, light)
    else:
        p.smat = np.zeros((3, 4))
        p.slook = np.zeros((3, 4))
    p.static_pos = s.geom_pos.copy()
    p.static_mat = s.geom_mat.copy()
    return p


def transparent_order(gpos, cams, sort_rbound, alpha):
    """MuJoCo's blend order of the translucent geoms of each frame.

    Returns ranks (B, G): the position of each translucent geom when sorted by its float32 distance to the camera
    less its bounding radius (ties in geom order), and -1 for geoms drawn opaque in that frame.
    """
    head = np.asarray(cams[:, :3], np.float32)
    diff = np.asarray(gpos, np.float32) - head[:, None, :]
    dist = np.sqrt((diff * diff).sum(-1, dtype=np.float32)).astype(np.float32) - sort_rbound[None]
    glass = alpha < TRANSPARENT_ALPHA
    dist = np.where(glass, dist, np.float32(np.inf))
    order = np.argsort(dist, axis=1, kind="mergesort")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(order.shape[1])[None].repeat(len(order), 0), axis=1)
    return np.where(glass, rank, -1)


OBJ_GEOM, OBJ_SITE = 5, 6


def geom_keys(model, ref_model, extra=None):
    """The ``(objid, objtype)`` of each geom of ``model`` in ``ref_model`` (``mjtObj`` numbering), -1 if none.

    Geoms match by body name and by their order within the body; names must agree. ``extra`` maps geoms with no
    counterpart, such as a geom that draws a site of the reference, to their pair.
    """
    import mujoco

    def by_body(m):
        out = {}
        for g in range(int(m.ngeom)):
            b = int(m.geom_bodyid[g])
            out.setdefault(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"#{b}", []).append(g)
        return out

    def name(m, g):
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""

    extra = extra or {}
    keys = np.full((int(model.ngeom), 2), -1, np.int64)
    ref = by_body(ref_model)
    for body, geoms in by_body(model).items():
        geoms = [g for g in geoms if g not in extra]
        if not geoms:
            continue
        theirs = ref.get(body, [])
        if len(theirs) != len(geoms):
            raise ValueError(f"body {body!r}: {len(geoms)} geoms here, {len(theirs)} in the reference")
        for a, b in zip(geoms, theirs):
            if name(model, a) != name(ref_model, b):
                raise ValueError(f"geom {a} is {name(model, a)!r}, reference geom {b} is {name(ref_model, b)!r}")
            keys[a] = (b, OBJ_GEOM)
    for g, pair in extra.items():
        keys[int(g)] = pair
    return keys


def segmentation(index, objid, keys):
    """``(B, H, W, 2)`` reference ``(objid, objtype)`` pairs of drawn-geom indices ``index`` (-1 for background)."""
    pairs = keys[objid]
    out = np.full((*index.shape, 2), -1, np.int64)
    hit = index >= 0
    out[hit] = pairs[index[hit]]
    return out
