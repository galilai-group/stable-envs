"""Turn an MjModel into one static triangle soup plus a per-vertex geom index.

Everything stable-envs wants to render is a rigid body: the triangles never change,
only the per-geom frame does. So the model is flattened once into

    vert   (V, 3)  local-frame position, already scaled by geom_size
    norm   (V, 3)  local-frame normal
    rgb    (V, 3)  vertex colour, from geom_rgba or its material
    vgeom  (V,)    which geom each vertex rides on
    face   (T, 3)  int32 indices into vert

and a frame is then `xmat[b, vgeom] @ vert + xpos[b, vgeom]` followed by a camera
matrix. This module is backend-agnostic; both the nvdiffrast and the torch paths
consume it.
"""
from __future__ import annotations

import numpy as np

try:
    import mujoco

    G = mujoco.mjtGeom
    PLANE, HFIELD, SPHERE, CAPSULE = G.mjGEOM_PLANE, G.mjGEOM_HFIELD, G.mjGEOM_SPHERE, G.mjGEOM_CAPSULE
    ELLIPSOID, CYLINDER, BOX, MESH = G.mjGEOM_ELLIPSOID, G.mjGEOM_CYLINDER, G.mjGEOM_BOX, G.mjGEOM_MESH
except ImportError:  # pragma: no cover
    PLANE, HFIELD, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, MESH = range(8)


def _uv_sphere(radii, nu=16, nv=12):
    """Lat-long sphere scaled by `radii` (a 3-vector, so this covers ellipsoids)."""
    u = np.linspace(0, 2 * np.pi, nu + 1)[:-1]
    v = np.linspace(0, np.pi, nv + 1)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    n = np.stack([np.sin(vv) * np.cos(uu), np.sin(vv) * np.sin(uu), np.cos(vv)], -1).reshape(-1, 3)
    p = n * np.asarray(radii, dtype=np.float64)
    idx = np.arange((nu) * (nv + 1)).reshape(nu, nv + 1)
    f = []
    for i in range(nu):
        i2 = (i + 1) % nu
        for j in range(nv):
            a, b, c, d = idx[i, j], idx[i2, j], idx[i2, j + 1], idx[i, j + 1]
            f.append([a, b, c])
            f.append([a, c, d])
    return p, n / np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-9), np.array(f, dtype=np.int64)


def _capsule(r, hl, nu=16, nv=6):
    """Cylinder of half-length `hl` about z capped by two hemispheres of radius `r`."""
    verts, norms, faces = [], [], []
    u = np.linspace(0, 2 * np.pi, nu + 1)[:-1]
    cu, su = np.cos(u), np.sin(u)

    # side wall
    for z in (-hl, hl):
        verts.append(np.stack([r * cu, r * su, np.full(nu, z)], -1))
        norms.append(np.stack([cu, su, np.zeros(nu)], -1))
    for i in range(nu):
        i2 = (i + 1) % nu
        faces += [[i, i2, nu + i2], [i, nu + i2, nu + i]]
    off = 2 * nu

    # hemispherical caps
    for sign, hl_z in ((1.0, hl), (-1.0, -hl)):
        vang = np.linspace(0, np.pi / 2, nv + 1)
        base = off
        for k, a in enumerate(vang):
            ring_r, ring_z = r * np.cos(a), sign * r * np.sin(a)
            verts.append(np.stack([ring_r * cu, ring_r * su, np.full(nu, hl_z + ring_z)], -1))
            norms.append(np.stack([np.cos(a) * cu, np.cos(a) * su, np.full(nu, sign * np.sin(a))], -1))
        for k in range(nv):
            for i in range(nu):
                i2 = (i + 1) % nu
                a, b = base + k * nu + i, base + k * nu + i2
                c, d = base + (k + 1) * nu + i2, base + (k + 1) * nu + i
                faces += [[a, b, c], [a, c, d]]
        off += (nv + 1) * nu
    return np.concatenate(verts), np.concatenate(norms), np.array(faces, dtype=np.int64)


def _cylinder(r, hl, nu=16):
    u = np.linspace(0, 2 * np.pi, nu + 1)[:-1]
    cu, su = np.cos(u), np.sin(u)
    side = [np.stack([r * cu, r * su, np.full(nu, z)], -1) for z in (-hl, hl)]
    sn = np.stack([cu, su, np.zeros(nu)], -1)
    verts = [side[0], side[1], side[0], side[1], np.array([[0, 0, -hl], [0, 0, hl]])]
    norms = [sn, sn,
             np.tile([0, 0, -1.0], (nu, 1)), np.tile([0, 0, 1.0], (nu, 1)),
             np.array([[0, 0, -1.0], [0, 0, 1.0]])]
    faces = []
    for i in range(nu):
        i2 = (i + 1) % nu
        faces += [[i, i2, nu + i2], [i, nu + i2, nu + i]]
        faces += [[4 * nu, 2 * nu + i2, 2 * nu + i]]          # bottom cap
        faces += [[4 * nu + 1, 3 * nu + i, 3 * nu + i2]]      # top cap
    return np.concatenate(verts), np.concatenate(norms), np.array(faces, dtype=np.int64)


def _box(half):
    s = np.asarray(half, dtype=np.float64)
    v, n, f = [], [], []
    axes = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    for a, b, c in axes:
        for sign in (1.0, -1.0):
            quad = np.zeros((4, 3))
            corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
            for k, (p, q) in enumerate(corners):
                quad[k, a] = sign * s[a]
                quad[k, b] = p * s[b] * (sign if a == 0 else 1.0)
                quad[k, c] = q * s[c]
            nrm = np.zeros(3)
            nrm[a] = sign
            base = len(v) * 4
            v.append(quad)
            n.append(np.tile(nrm, (4, 1)))
            f += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    return np.concatenate(v), np.concatenate(n), np.array(f, dtype=np.int64)


def _plane(size, extent):
    half = [size[0] if size[0] > 0 else 20.0 * extent, size[1] if size[1] > 0 else 20.0 * extent]
    v = np.array([[-half[0], -half[1], 0], [half[0], -half[1], 0], [half[0], half[1], 0], [-half[0], half[1], 0]])
    n = np.tile([0, 0, 1.0], (4, 1))
    return v, n, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)


def geom_rgb(m, g):
    mat = m.geom_matid[g]
    rgba = m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g]
    return np.asarray(rgba[:3], dtype=np.float64)


def tessellate(m, sphere_res=(16, 12), enabled_groups=(0, 1, 2)):
    """Flatten every renderable geom of `m` into one triangle soup."""
    extent = float(m.stat.extent)
    V, N, C, GI, F = [], [], [], [], []
    nv = 0
    for g in range(m.ngeom):
        if m.geom_group[g] not in enabled_groups:
            continue
        if m.geom_rgba[g][3] == 0.0:
            continue
        t, sz = int(m.geom_type[g]), m.geom_size[g]
        if t == PLANE:
            v, n, f = _plane(sz, extent)
        elif t == SPHERE:
            v, n, f = _uv_sphere([sz[0]] * 3, *sphere_res)
        elif t == ELLIPSOID:
            v, n, f = _uv_sphere(sz[:3], *sphere_res)
        elif t == CAPSULE:
            v, n, f = _capsule(sz[0], sz[1])
        elif t == CYLINDER:
            v, n, f = _cylinder(sz[0], sz[1])
        elif t == BOX:
            v, n, f = _box(sz[:3])
        elif t == MESH:
            did = int(m.geom_dataid[g])
            va, vn = int(m.mesh_vertadr[did]), int(m.mesh_vertnum[did])
            fa, fn = int(m.mesh_faceadr[did]), int(m.mesh_facenum[did])
            v = np.asarray(m.mesh_vert[va:va + vn], dtype=np.float64)
            f = np.asarray(m.mesh_face[fa:fa + fn], dtype=np.int64)
            if m.mesh_normal is not None and len(m.mesh_normal) > va:
                na = int(m.mesh_normaladr[did]) if hasattr(m, "mesh_normaladr") else va
                nn = int(m.mesh_normalnum[did]) if hasattr(m, "mesh_normalnum") else vn
                n = np.asarray(m.mesh_normal[na:na + nn], dtype=np.float64)
                if len(n) != len(v):
                    n = v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-9)
            else:
                n = v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-9)
        else:  # hfield / sdf: not tessellated here
            continue
        V.append(v)
        N.append(n)
        C.append(np.tile(geom_rgb(m, g), (len(v), 1)))
        GI.append(np.full(len(v), g, dtype=np.int64))
        F.append(f + nv)
        nv += len(v)

    return dict(
        vert=np.concatenate(V).astype(np.float32),
        norm=np.concatenate(N).astype(np.float32),
        rgb=np.concatenate(C).astype(np.float32),
        vgeom=np.concatenate(GI).astype(np.int64),
        face=np.concatenate(F).astype(np.int32),
    )


def camera_matrices(m, d, cam, res, aspect=1.0):
    """View-projection for a MuJoCo camera, OpenGL convention (camera looks down -z)."""
    if cam < 0 or m.ncam == 0:
        pos = np.array([0.0, -3.0 * m.stat.extent, 2.0 * m.stat.extent])
        fwd = -pos / np.linalg.norm(pos)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        R = np.stack([right, up, -fwd], -1)
        fovy = 45.0
    else:
        pos = np.asarray(d.cam_xpos[cam], dtype=np.float64)
        R = np.asarray(d.cam_xmat[cam], dtype=np.float64).reshape(3, 3)
        fovy = float(m.cam_fovy[cam])

    view = np.eye(4)
    view[:3, :3] = R.T
    view[:3, 3] = -R.T @ pos

    znear = max(float(m.vis.map.znear) * float(m.stat.extent), 1e-3)
    zfar = float(m.vis.map.zfar) * float(m.stat.extent)
    f = 1.0 / np.tan(np.deg2rad(fovy) / 2.0)
    proj = np.zeros((4, 4))
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (zfar + znear) / (znear - zfar)
    proj[2, 3] = 2 * zfar * znear / (znear - zfar)
    proj[3, 2] = -1.0
    return (proj @ view).astype(np.float32), pos.astype(np.float32)
