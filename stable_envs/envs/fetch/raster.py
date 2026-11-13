"""Numba CPU rasterizer for Fetch frames, following MuJoCo's fixed-function OpenGL renderer.

The scene is the render model of the OpenGL backend: the same geoms, materials, lights and camera.
Each frame is drawn with 4 coverage samples per pixel at the OpenGL 4x sample positions, as the
model's multisampled offscreen buffer does. Lighting is evaluated per vertex (Gouraud), with the
headlight, a local viewer and no global ambient, and the colours are interpolated with perspective
correction. Translucent geoms (the mocap markers and a translucent arm) are blended after the
opaque scene and the skybox, sorted by distance to the camera, once front to back and once back to
front, without depth writes. Frames differ from the OpenGL frames by small edge and rounding
differences.
"""

from __future__ import annotations

import copy
import math
import threading

import numba
import numpy as np
import torch
from numba import njit, prange

_LOCK = threading.RLock()
_SCENES = {}
_PLATES = {}
# OpenGL 4x MSAA sample positions, with y measured down from the top of the pixel.
_SX = np.array([0.375, 0.875, 0.125, 0.625])
_SY = np.array([0.875, 0.625, 0.375, 0.125])
TRANSPARENT_ALPHA = 0.995
POW_STEPS = 4096


class _Scene:
    """Static geometry of the visible geoms, the lights, the camera optics and the skybox."""


def _quads_to_tris(quads):
    out = []
    for q in quads:
        out.append((q[0], q[1], q[3]))
        out.append((q[1], q[2], q[3]))
    return out


def _box(size, numquads):
    d = 2.0 / numquads
    v, n = [], []
    for x in range(numquads):
        for y in range(numquads):
            a0, a1, b0, b1 = d * x - 1, d * (x + 1) - 1, d * y - 1, d * (y + 1) - 1
            faces = (
                ((0, 0, 1), ((a0, b0, 1), (a1, b0, 1), (a1, b1, 1), (a0, b1, 1))),
                ((0, 0, -1), ((a0, b1, -1), (a1, b1, -1), (a1, b0, -1), (a0, b0, -1))),
                ((1, 0, 0), ((1, a0, b0), (1, a1, b0), (1, a1, b1), (1, a0, b1))),
                ((-1, 0, 0), ((-1, a0, b1), (-1, a1, b1), (-1, a1, b0), (-1, a0, b0))),
                ((0, -1, 0), ((a0, -1, b0), (a1, -1, b0), (a1, -1, b1), (a0, -1, b1))),
                ((0, 1, 0), ((a0, 1, b1), (a1, 1, b1), (a1, 1, b0), (a0, 1, b0))),
            )
            for normal, quad in faces:
                for tri in _quads_to_tris([quad]):
                    v.append(tri)
                    n.append((normal, normal, normal))
    return _scaled(np.array(v, np.float64), np.array(n, np.float64), size)


def _sphere_vertex(az, el):
    c = math.cos(el)
    p = (math.cos(az) * c, math.sin(az) * c, math.sin(el))
    return p, p


def _sphere(size, nslice, nstack):
    """MuJoCo's builtin unit sphere: pole triangles and quads, scaled to ``size``."""
    v, n = [], []
    f32 = np.float32
    for sign in (-1, 1):
        el1 = float(f32(0.5 * math.pi * sign * (nstack // 2 - 1) / (nstack // 2)))
        for j in range(nslice):
            az1 = float(f32(2.0 * math.pi * j / nslice))
            az2 = float(f32(2.0 * math.pi * (j + 1) / nslice))
            p1, n1 = _sphere_vertex(az1, el1)
            p2, n2 = _sphere_vertex(az2, el1)
            p3 = n3 = (0.0, 0.0, float(sign))
            if sign > 0:
                v.append((p1, p2, p3))
                n.append((n1, n2, n3))
            else:
                v.append((p3, p2, p1))
                n.append((n3, n2, n1))
    for sign in (-1, 1):
        for i in range(nstack // 2 - 1):
            el1 = float(f32(0.5 * math.pi * sign * i / (nstack // 2)))
            el2 = float(f32(0.5 * math.pi * sign * (i + 1) / (nstack // 2)))
            for j in range(nslice):
                az1 = float(f32(2.0 * math.pi * j / nslice))
                az2 = float(f32(2.0 * math.pi * (j + 1) / nslice))
                q = [
                    _sphere_vertex(az1, el1),
                    _sphere_vertex(az2, el1),
                    _sphere_vertex(az2, el2),
                    _sphere_vertex(az1, el2),
                ]
                if sign < 0:
                    q = q[::-1]
                for a, b, c in ((0, 1, 3), (1, 2, 3)):
                    v.append((q[a][0], q[b][0], q[c][0]))
                    n.append((q[a][1], q[b][1], q[c][1]))
    return _scaled(np.array(v, np.float64), np.array(n, np.float64), size)


def _cylinder(size, nslice, nstack):
    """MuJoCo's builtin closed unit cylinder: two disks and the side quads, scaled to ``size``."""
    f32 = np.float32
    az = [float(f32(2.0 * math.pi * j / nslice)) for j in range(nslice + 1)]
    v, n = [], []

    def disk(sign, nst):
        normal = (0.0, 0.0, float(sign))
        r1 = 1.0 / nst
        for j in range(nslice):
            p1 = (math.cos(az[j]) * r1, math.sin(az[j]) * r1, float(sign))
            p2 = (math.cos(az[j + 1]) * r1, math.sin(az[j + 1]) * r1, float(sign))
            p3 = (0.0, 0.0, float(sign))
            v.append((p1, p2, p3) if sign > 0 else (p3, p2, p1))
            n.append((normal,) * 3)
        for i in range(nst - 1):
            r1, r2 = (i + 1) / nst, (i + 2) / nst
            for j in range(nslice):
                q = [
                    (math.cos(az[j]) * r2, math.sin(az[j]) * r2, float(sign)),
                    (math.cos(az[j + 1]) * r2, math.sin(az[j + 1]) * r2, float(sign)),
                    (math.cos(az[j + 1]) * r1, math.sin(az[j + 1]) * r1, float(sign)),
                    (math.cos(az[j]) * r1, math.sin(az[j]) * r1, float(sign)),
                ]
                if sign < 0:
                    q = q[::-1]
                for tri in _quads_to_tris([q]):
                    v.append(tri)
                    n.append((normal,) * 3)

    disk(-1, nstack // 2)
    for i in range(nstack):
        h1, h2 = 2.0 * i / nstack - 1.0, 2.0 * (i + 1) / nstack - 1.0
        for j in range(nslice):
            q = [(az[j], h1), (az[j + 1], h1), (az[j + 1], h2), (az[j], h2)]
            pts = [(math.cos(a), math.sin(a), h) for a, h in q]
            nrm = [(math.cos(a), math.sin(a), 0.0) for a, _ in q]
            for a, b, c in ((0, 1, 3), (1, 2, 3)):
                v.append((pts[a], pts[b], pts[c]))
                n.append((nrm[a], nrm[b], nrm[c]))
    disk(1, nstack // 2)
    return _scaled(np.array(v, np.float64), np.array(n, np.float64), size)


def _plane(size):
    sx, sy = float(size[0]), float(size[1])
    quad = ((-sx, -sy, 0.0), (sx, -sy, 0.0), (sx, sy, 0.0), (-sx, sy, 0.0))
    v = np.array(_quads_to_tris([quad]), np.float64)
    n = np.zeros_like(v)
    n[..., 2] = 1.0
    return v, n


def _scaled(v, n, size):
    size = np.asarray(size, np.float64)
    n = n / size
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-30)
    return v * size, n


def _mesh(m, mesh):
    """Triangles of a mesh with the normals MuJoCo's OpenGL renderer gives each corner."""
    va, fa, na = int(m.mesh_vertadr[mesh]), int(m.mesh_faceadr[mesh]), int(m.mesh_normaladr[mesh])
    nf = int(m.mesh_facenum[mesh])
    faces = np.asarray(m.mesh_face[fa : fa + nf], np.int64)
    fnorm = np.asarray(m.mesh_facenormal[fa : fa + nf], np.int64)
    vert = np.asarray(m.mesh_vert, np.float32)[va + faces]
    corner = np.asarray(m.mesh_normal, np.float32)[na + fnorm]
    e1 = (vert[:, 1] - vert[:, 0]).astype(np.float32)
    e2 = (vert[:, 2] - vert[:, 0]).astype(np.float32)
    face = np.cross(e1, e2).astype(np.float32)
    face /= np.maximum(np.linalg.norm(face, axis=-1, keepdims=True), np.float32(1e-15))
    keep = (corner * face[:, None, :]).sum(-1) >= 0.8
    normals = np.where(keep[..., None], corner, face[:, None, :])
    return vert.astype(np.float64), normals.astype(np.float64)


def _build_scene(scene: str) -> _Scene:
    import mujoco

    from . import envs as fetch

    b = fetch._build_bundle(scene)
    m = copy.copy(b.model)
    fetch._restore(m, b.pristine)
    markers, groups = b.info["markers"]
    m.geom_group[markers] = groups
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    q = m.vis.quality
    visible, verts, norms, starts = [], [], [], [0]
    for g in range(m.ngeom):
        if int(m.geom_group[g]) not in (0, 1, 2) or float(b.pristine["geom_rgba"][g, 3]) == 0.0:
            continue
        kind = int(m.geom_type[g])
        size = np.asarray(m.geom_size[g], np.float64)
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            v, n = _mesh(m, int(m.geom_dataid[g]))
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            v, n = _box(size, int(q.numquads))
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            v, n = _sphere(size[[0, 0, 0]], int(q.numslices), int(q.numstacks))
        elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            v, n = _sphere(size, int(q.numslices), int(q.numstacks))
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            v, n = _cylinder(size[[0, 0, 1]], int(q.numslices), int(q.numstacks))
        elif kind == mujoco.mjtGeom.mjGEOM_PLANE:
            v, n = _plane(size)
        else:
            raise NotImplementedError(f"Fetch CPU rasterizer has no geom type {kind}")
        visible.append(g)
        verts.append(v)
        norms.append(n)
        starts.append(starts[-1] + len(v))
    s = _Scene()
    s.model, s.data, s.bundle = m, d, b
    s.geoms = np.asarray(visible, np.int64)
    # Corners that share a position and a normal are transformed and lit once per frame.
    cpos, cnor, tris, cstarts = [], [], [], [0]
    for v, n in zip(verts, norms):
        key = np.concatenate([v.reshape(-1, 3), n.reshape(-1, 3)], 1)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        cpos.append(uniq[:, :3])
        cnor.append(uniq[:, 3:])
        tris.append(inv.reshape(-1, 3) + cstarts[-1])
        cstarts.append(cstarts[-1] + len(uniq))
    s.cpos = np.ascontiguousarray(np.concatenate(cpos), np.float64)
    s.cnor = np.ascontiguousarray(np.concatenate(cnor), np.float64)
    s.tris = np.ascontiguousarray(np.concatenate(tris), np.int64)
    s.cstarts = np.asarray(cstarts, np.int64)
    s.starts = np.asarray(starts, np.int64)
    # Culling radius about each geom's frame; MuJoCo's own radius orders the translucent geoms.
    s.rbound = np.array(
        [np.linalg.norm(s.cpos[s.cstarts[i] : s.cstarts[i + 1]], axis=1).max() for i in range(len(s.geoms))]
    )
    s.sort_rbound = np.asarray(m.geom_rbound[s.geoms], np.float64)
    mat = np.asarray(m.geom_matid[s.geoms], np.int64)
    s.material = np.stack(
        [
            np.where(mat >= 0, m.mat_specular[mat], fetch.DEFAULT_SPECULAR),
            np.where(mat >= 0, m.mat_shininess[mat], fetch.DEFAULT_SHININESS) * 128.0,
            np.where(mat >= 0, m.mat_emission[mat], fetch.DEFAULT_EMISSION),
        ],
        axis=1,
    ).astype(np.float64)
    s.rgba = np.asarray(b.pristine["geom_rgba"][s.geoms], np.float64)
    # Specular powers x ** shininess on [0, 1], sampled for linear interpolation.
    shins, row = np.unique(s.material[:, 1], return_inverse=True)
    grid = np.linspace(0.0, 1.0, POW_STEPS + 1)
    s.powlut = np.ascontiguousarray(grid[None, :] ** shins[:, None], np.float64)
    s.material = np.concatenate([s.material, row.reshape(-1, 1).astype(np.float64)], axis=1)
    index = {int(g): i for i, g in enumerate(s.geoms)}
    s.slot = lambda g: index.get(int(g), -1)
    # Weld equalities that reset_mocap2body_xpos follows: (mocap id, body id).
    s.welds = []
    for e in range(m.neq):
        if int(m.eq_type[e]) != int(mujoco.mjtEq.mjEQ_WELD):
            continue
        a, c = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
        if int(m.body_mocapid[a]) >= 0:
            s.welds.append((int(m.body_mocapid[a]), c))
        else:
            s.welds.append((int(m.body_mocapid[c]), a))
    lights = []
    head = m.vis.headlight
    if int(head.active):
        lights.append((-1, np.array(head.ambient), np.array(head.diffuse), np.array(head.specular)))
    directional = getattr(m, "light_type", None)
    for i in range(m.nlight):
        if not bool(m.light_active[i]):
            continue
        is_dir = (
            int(directional[i]) == int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
            if directional is not None
            else bool(m.light_directional[i])
        )
        if not is_dir:
            raise NotImplementedError("Fetch CPU rasterizer draws directional lights only")
        lights.append((i, np.array(m.light_ambient[i]), np.array(m.light_diffuse[i]), np.array(m.light_specular[i])))
    s.lights = lights
    s.light_dir = np.asarray(d.light_xdir, np.float64).copy()
    s.near = float(m.vis.map.znear * m.stat.extent)
    s.fovy = float(m.cam_fovy[b.info["cam"]])
    t = int(b.info["sky"])
    s.faces = _sky_faces(m, t)
    s.sky_default = fetch.sky_color(m)
    return s


def _sky_faces(m, t) -> np.ndarray:
    w, h, nc = int(m.tex_width[t]), int(m.tex_height[t]), int(m.tex_nchannel[t])
    data = np.asarray(m.tex_data[int(m.tex_adr[t]) : int(m.tex_adr[t]) + w * h * nc], np.uint8).reshape(-1, w, nc)
    faces = np.repeat(data[None, :w], 6, 0) if w == h else data.reshape(6, w, w, nc)
    return np.ascontiguousarray(faces[..., :3], np.float64)


def scene(name: str) -> _Scene:
    with _LOCK:
        if name not in _SCENES:
            _SCENES[name] = _build_scene(name)
        return _SCENES[name]


def render(env, states=None, size=None, supersample=None, variation=None, style=None, out=None):
    env._check_open("render")
    rows = torch.as_tensor(env._render_state if states is None else states, device=env.device)
    lead = tuple(rows.shape[:-1])
    rows = rows.reshape(-1, env.state_dim).detach().cpu().numpy()
    size, ss = int(size or env.size), int(env.ss if supersample is None else supersample)
    if size < 1 or not 1 <= ss <= 4:
        raise ValueError("Invalid Fetch render size/supersample")
    if not len(rows):
        raise ValueError("Cannot render an empty Fetch batch")
    style = env.resolve_style(style, variation, batch=len(rows))
    values = {k: style[k].detach().cpu().numpy().reshape(len(rows), -1) for k in env.variation_space.names()}
    shape = (*lead, size, size, 3)
    if out is not None and (
        out.shape != shape or out.device != env.device or out.dtype != torch.uint8 or not out.is_contiguous()
    ):
        raise ValueError("Invalid Fetch render output buffer")
    host = None if out is None or env.device.type != "cpu" else out.view(-1, size, size, 3).numpy()
    images = np.empty((len(rows), size, size, 3), np.uint8) if host is None else host
    draw(env.SCENE, rows, values, ss, images)
    if host is not None:
        return out
    result = torch.from_numpy(images).to(env.device).reshape(shape)
    return result if out is None else out.copy_(result)


def _quat_mat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def prepare(name, rows, values):
    """Per-frame geom poses, colours, cameras, lights and sky for ``draw``."""
    import mujoco

    from . import envs as fetch

    s = scene(name)
    b, info = s.bundle, s.bundle.info
    B, G = len(rows), len(s.geoms)
    with _LOCK:
        m, d = s.model, s.data
        gpos = np.empty((B, G, 3), np.float64)
        gmat = np.empty((B, G, 9), np.float64)
        nq, goal = b.nq, info["goal_mocap"]
        geoms = s.geoms
        for i in range(B):
            d.qpos[:] = rows[i, :nq]
            d.mocap_pos[goal] = rows[i, nq : nq + 3]
            mujoco.mj_kinematics(m, d)
            for mocap, body in s.welds:
                d.mocap_pos[mocap] = d.xpos[body]
                d.mocap_quat[mocap] = d.xquat[body]
            if s.welds:
                mujoco.mj_kinematics(m, d)
            gpos[i] = d.geom_xpos[geoms]
            gmat[i] = d.geom_xmat[geoms]
        cams = np.empty((B, 12), np.float64)
        cd = b.cam_defaults
        keys = np.stack(
            [
                values["camera.distance_scale"][:, 0],
                values["camera.azimuth_delta"][:, 0],
                values["camera.elevation_delta"][:, 0],
            ],
            1,
        ).astype(np.float64)
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        for u, (ds, az, el) in enumerate(uniq):
            pos, quat = fetch._free_cam_pose(
                m, d, cd["lookat"], cd["distance"] * ds, cd["azimuth"] + az, cd["elevation"] + el
            )
            r = _quat_mat(np.asarray(quat, np.float64))
            cams[inv.reshape(-1) == u] = np.concatenate([pos, r[:, 0], r[:, 1], -r[:, 2]])
    rgba = np.repeat(s.rgba[None], B, 0)
    if "rendering.transparent_arm" in values:
        robot = [s.slot(g) for g in info["robot"] if s.slot(g) >= 0]
        flag = values["rendering.transparent_arm"][:, 0] > 0.5
        rgba[np.ix_(flag, robot, [3])] = fetch.ARM_ALPHA
    for name_, geoms in (
        ("floor", [info["floor"]]),
        ("table", [info["table"]]),
        ("object", [info["object"]]),
        ("arm", info["arm"]),
        ("goal", [info["target"]]),
    ):
        if name_ + ".color" not in values:
            continue
        idx = [s.slot(g) for g in geoms if g >= 0 and s.slot(g) >= 0]
        if idx:
            rgba[:, idx, :3] = values[name_ + ".color"][:, None, :3]
    L = len(s.lights)
    lights = np.zeros((B, L, 12), np.float64)
    for k, (i, amb, dif, spec) in enumerate(s.lights):
        lights[:, k, 3:6] = amb
        lights[:, k, 9:12] = spec
        if i < 0:
            lights[:, k, 0:3] = -cams[:, 9:12]
            lights[:, k, 6:9] = dif
        else:
            lights[:, k, 0:3] = -s.light_dir[i]
            lights[:, k, 6:9] = np.asarray(b.pristine["light_diffuse"][i]) * values["light.diffuse"][:, :1]
    sky = np.full((B, 3), -1.0)
    if "sky.color" in values:
        default = s.sky_default
        for i, color in enumerate(values["sky.color"]):
            c = np.asarray(color, np.float64).reshape(3)
            if not np.array_equal(np.round(c, 6), np.round(default, 6)):
                sky[i] = (np.clip(c, 0.0, 1.0) * 255.0).astype(np.uint8)
    return gpos, gmat, rgba, cams, lights, sky, inv.reshape(-1), uniq


def _sky_plates(s, name, res, cams, sky, cam_index, cam_keys, tanh):
    """Skybox colours of every pixel for each camera the default-sky frames use, cached by camera."""
    need = np.unique(cam_index[sky[:, 0] < 0])
    plate_index = np.zeros(len(cams), np.int64)
    if not len(need):
        return np.zeros((1, 1, 1, 3), np.uint8), plate_index
    plates = np.empty((len(need), res, res, 3), np.uint8)
    with _LOCK:
        for j, u in enumerate(need):
            key = (name, res, tuple(np.asarray(cam_keys[u], np.float64).tolist()))
            hit = _PLATES.get(key)
            if hit is None:
                first = int(np.flatnonzero(cam_index == u)[0])
                hit = np.empty((res, res, 3), np.uint8)
                _sky_plate(s.faces, cams[first], tanh, hit)
                _PLATES[key] = hit
                while len(_PLATES) > 64:
                    _PLATES.pop(next(iter(_PLATES)))
            plates[j] = hit
            plate_index[cam_index == u] = j
    return plates, plate_index


@njit(parallel=True, cache=True, error_model="numpy", fastmath=True)
def _sky_plate(faces, cam, tanh, out):
    res = out.shape[0]
    half = 0.5 * res
    for py in prange(res):
        sk = np.empty(3)
        for px in range(res):
            a = (px + 0.5 - half) / half * tanh
            e = (half - py - 0.5) / half * tanh
            dx = cam[9] + a * cam[3] + e * cam[6]
            dy = cam[10] + a * cam[4] + e * cam[7]
            dz = cam[11] + a * cam[5] + e * cam[8]
            _cube(faces, dx, dz, -dy, sk)
            for ch in range(3):
                out[py, px, ch] = np.uint8(math.floor(sk[ch] + 0.5))


def draw(name, rows, values, ss, images):
    """Draw compact ``rows`` with appearance ``values`` into ``(B, size, size, 3)`` uint8 ``images`` in place."""
    from ...render import cpu_raster

    cpu_raster.available()
    s = scene(name)
    gpos, gmat, rgba, cams, lights, sky, cam_index, cam_keys = prepare(name, rows, values)
    size = images.shape[1]
    tanh = math.tan(math.radians(s.fovy) / 2.0)
    plates, plate_index = _sky_plates(s, name, size * ss, cams, sky, cam_index, cam_keys, tanh)
    threads = numba.get_num_threads()
    bands = max(1, min(size, -(-threads // len(rows)))) if len(rows) < threads else 1
    # One parallel launch at a time: Numba's default threading layer is not reentrant across threads.
    with _LOCK:
        _render(
            s.cpos,
            s.cnor,
            s.tris,
            s.cstarts,
            s.starts,
            s.rbound,
            s.sort_rbound,
            s.material,
            s.powlut,
            gpos,
            gmat,
            rgba,
            cams,
            lights,
            sky,
            plates,
            plate_index,
            tanh,
            s.near,
            ss,
            bands,
            images,
        )


@njit(cache=True, inline="always", error_model="numpy", fastmath=True)
def _light_vertex(px, py, pz, nx, ny, nz, cx, cy, cz, r, g, bb, a, spec, lut, emis, lights, out, o):
    vx, vy, vz = cx - px, cy - py, cz - pz
    vn = math.sqrt(vx * vx + vy * vy + vz * vz)
    if vn > 0:
        vx, vy, vz = vx / vn, vy / vn, vz / vn
    o0, o1, o2 = emis * r, emis * g, emis * bb
    for k in range(lights.shape[0]):
        lx, ly, lz = lights[k, 0], lights[k, 1], lights[k, 2]
        o0 += lights[k, 3] * r
        o1 += lights[k, 4] * g
        o2 += lights[k, 5] * bb
        ndl = nx * lx + ny * ly + nz * lz
        if ndl > 0:
            o0 += ndl * lights[k, 6] * r
            o1 += ndl * lights[k, 7] * g
            o2 += ndl * lights[k, 8] * bb
            if spec > 0:
                hx, hy, hz = lx + vx, ly + vy, lz + vz
                hn = math.sqrt(hx * hx + hy * hy + hz * hz)
                if hn > 0:
                    ndh = (nx * hx + ny * hy + nz * hz) / hn
                    if ndh > 0:
                        t = min(ndh, 1.0) * (lut.shape[0] - 1)
                        i = min(int(t), lut.shape[0] - 2)
                        f = (lut[i] + (t - i) * (lut[i + 1] - lut[i])) * spec
                        o0 += f * lights[k, 9]
                        o1 += f * lights[k, 10]
                        o2 += f * lights[k, 11]
    out[o, 0] = min(max(o0, 0.0), 1.0)
    out[o, 1] = min(max(o1, 0.0), 1.0)
    out[o, 2] = min(max(o2, 0.0), 1.0)
    out[o, 3] = min(max(a, 0.0), 1.0)


@njit(cache=True, inline="always", error_model="numpy", fastmath=True)
def _light_corner(k, cc, done, ctx):
    """Light corner ``k`` of the geom in ``ctx`` into ``cc[k]``."""
    ex, ey, ew, cnor, c0, gm, cam, r, g, bb, a, spec, lut, emis, lights = ctx
    nlx, nly, nlz = cnor[c0 + k, 0], cnor[c0 + k, 1], cnor[c0 + k, 2]
    nx = gm[0] * nlx + gm[1] * nly + gm[2] * nlz
    ny = gm[3] * nlx + gm[4] * nly + gm[5] * nlz
    nz = gm[6] * nlx + gm[7] * nly + gm[8] * nlz
    nn = math.sqrt(nx * nx + ny * ny + nz * nz)
    if nn > 0:
        nx, ny, nz = nx / nn, ny / nn, nz / nn
    cx, cy, cz = cam[0], cam[1], cam[2]
    wx = cx + ex[k] * cam[3] + ey[k] * cam[6] + ew[k] * cam[9]
    wy = cy + ex[k] * cam[4] + ey[k] * cam[7] + ew[k] * cam[10]
    wz = cz + ex[k] * cam[5] + ey[k] * cam[8] + ew[k] * cam[11]
    _light_vertex(wx, wy, wz, nx, ny, nz, cx, cy, cz, r, g, bb, a, spec, lut, emis, lights, cc, k)
    done[k] = True


@njit(cache=True, inline="always", error_model="numpy", fastmath=True)
def _tri(x0, y0, q0, x1, y1, q1, x2, y2, q2, k0, k1, k2, cc, done, ctx, res, y0b, y1b, depth, color, blend):
    """Fill one screen triangle with vertices ``(x, y, 1/w)`` in pixel rows ``[y0b, y1b)``.

    The colours of corners ``k0, k1, k2`` are ``cc`` rows; a corner not ``done`` is lit from ``ctx`` when the
    triangle first covers a visible sample. ``depth`` and ``color`` hold 4 samples per pixel of those rows,
    at the OpenGL 4x sample positions.
    """
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if area == 0.0:
        return
    if area < 0:
        x1, y1, q1, k1, x2, y2, q2, k2 = x2, y2, q2, k2, x1, y1, q1, k1
        area = -area
    fx0 = min(x0, min(x1, x2))
    fx1 = max(x0, max(x1, x2))
    fy0 = min(y0, min(y1, y2))
    fy1 = max(y0, max(y1, y2))
    # Samples lie on a quarter-pixel grid offset by 1/8: skip a triangle whose box holds no grid line.
    if math.floor(fx1 * 4.0 - 0.5) < math.ceil(fx0 * 4.0 - 0.5) or math.floor(fy1 * 4.0 - 0.5) < math.ceil(
        fy0 * 4.0 - 0.5
    ):
        return
    xmin = max(int(math.floor(fx0)), 0)
    xmax = min(int(math.floor(fx1)), res - 1)
    ymin = max(int(math.floor(fy0)), y0b)
    ymax = min(int(math.floor(fy1)), y1b - 1)
    if xmin > xmax or ymin > ymax:
        return
    lit = False
    flat = False
    r0 = g0 = c0b = al0 = r1 = g1 = c1b = al1 = r2 = g2 = c2b = al2 = 0.0
    inv = 1.0 / area
    # Edge function e_k is opposite vertex k, positive inside, and e_0 + e_1 + e_2 = area.
    a0, b0 = (y1 - y2), (x2 - x1)
    a1, b1 = (y2 - y0), (x0 - x2)
    a2, b2 = (y0 - y1), (x1 - x0)
    c0 = -(a0 * x1 + b0 * y1)
    c1 = -(a1 * x2 + b1 * y2)
    c2 = -(a2 * x0 + b2 * y0)
    o00 = -0.125 * a0 + 0.375 * b0
    o10 = -0.125 * a1 + 0.375 * b1
    o20 = -0.125 * a2 + 0.375 * b2
    dz0 = (o00 * q0 + o10 * q1 + o20 * q2) * inv
    o01 = 0.375 * a0 + 0.125 * b0
    o11 = 0.375 * a1 + 0.125 * b1
    o21 = 0.375 * a2 + 0.125 * b2
    dz1 = (o01 * q0 + o11 * q1 + o21 * q2) * inv
    o02 = -0.375 * a0 + -0.125 * b0
    o12 = -0.375 * a1 + -0.125 * b1
    o22 = -0.375 * a2 + -0.125 * b2
    dz2 = (o02 * q0 + o12 * q1 + o22 * q2) * inv
    o03 = 0.125 * a0 + -0.375 * b0
    o13 = 0.125 * a1 + -0.375 * b1
    o23 = 0.125 * a2 + -0.375 * b2
    dz3 = (o03 * q0 + o13 * q1 + o23 * q2) * inv
    m0 = max(max(o00, o01), max(o02, o03))
    m1 = max(max(o10, o11), max(o12, o13))
    m2 = max(max(o20, o21), max(o22, o23))
    n0 = min(min(o00, o01), min(o02, o03))
    n1 = min(min(o10, o11), min(o12, o13))
    n2 = min(min(o20, o21), min(o22, o23))
    narrow = xmax - xmin < 4
    for py in range(ymin, ymax + 1):
        pxa = xmin
        pxb = xmax
        if not narrow:
            # Pixels whose samples can lie inside: the union over the sample rows of the x-span.
            lo = 1e300
            hi = -1e300
            for k in range(4):
                ys = py + _SY[k]
                left = -1e300
                right = 1e300
                d0 = b0 * ys + c0
                d1 = b1 * ys + c1
                d2 = b2 * ys + c2
                if a0 > 0:
                    left = max(left, -d0 / a0)
                elif a0 < 0:
                    right = min(right, -d0 / a0)
                elif d0 < 0:
                    continue
                if a1 > 0:
                    left = max(left, -d1 / a1)
                elif a1 < 0:
                    right = min(right, -d1 / a1)
                elif d1 < 0:
                    continue
                if a2 > 0:
                    left = max(left, -d2 / a2)
                elif a2 < 0:
                    right = min(right, -d2 / a2)
                elif d2 < 0:
                    continue
                if left <= right:
                    lo = min(lo, left - _SX[k])
                    hi = max(hi, right - _SX[k])
            if lo > hi:
                continue
            pxa = max(xmin, int(math.floor(lo)))
            pxb = min(xmax, int(math.floor(hi)) + 1)
        row = (py - y0b) * res
        cy = py + 0.5
        e0 = a0 * (pxa + 0.5) + b0 * cy + c0
        e1 = a1 * (pxa + 0.5) + b1 * cy + c1
        e2 = a2 * (pxa + 0.5) + b2 * cy + c2
        for px in range(pxa, pxb + 1):
            if px > pxa:
                e0 += a0
                e1 += a1
                e2 += a2
            if e0 + m0 < 0 or e1 + m1 < 0 or e2 + m2 < 0:
                continue
            zc = (e0 * q0 + e1 * q1 + e2 * q2) * inv
            base = (row + px) * 4
            mask = 0
            inside = e0 + n0 >= 0 and e1 + n1 >= 0 and e2 + n2 >= 0
            if inside or (e0 + o00 >= 0 and e1 + o10 >= 0 and e2 + o20 >= 0):
                z = zc + dz0
                if z >= depth[base + 0]:
                    if not blend:
                        depth[base + 0] = z
                    mask |= 1
            if inside or (e0 + o01 >= 0 and e1 + o11 >= 0 and e2 + o21 >= 0):
                z = zc + dz1
                if z >= depth[base + 1]:
                    if not blend:
                        depth[base + 1] = z
                    mask |= 2
            if inside or (e0 + o02 >= 0 and e1 + o12 >= 0 and e2 + o22 >= 0):
                z = zc + dz2
                if z >= depth[base + 2]:
                    if not blend:
                        depth[base + 2] = z
                    mask |= 4
            if inside or (e0 + o03 >= 0 and e1 + o13 >= 0 and e2 + o23 >= 0):
                z = zc + dz3
                if z >= depth[base + 3]:
                    if not blend:
                        depth[base + 3] = z
                    mask |= 8
            if mask == 0:
                continue
            if not lit:
                for k in (k0, k1, k2):
                    if not done[k]:
                        _light_corner(k, cc, done, ctx)
                r0, g0, c0b, al0 = cc[k0, 0], cc[k0, 1], cc[k0, 2], cc[k0, 3]
                r1, g1, c1b, al1 = cc[k1, 0], cc[k1, 1], cc[k1, 2], cc[k1, 3]
                r2, g2, c2b, al2 = cc[k2, 0], cc[k2, 1], cc[k2, 2], cc[k2, 3]
                flat = r0 == r1 and r1 == r2 and g0 == g1 and g1 == g2 and c0b == c1b and c1b == c2b
                flat = flat and al0 == al1 and al1 == al2
                lit = True
            if flat:
                r = min(max(r0, 0.0), 1.0)
                g = min(max(g0, 0.0), 1.0)
                bl = min(max(c0b, 0.0), 1.0)
                al = min(max(al0, 0.0), 1.0)
            else:
                w0 = e0 * q0
                w1 = e1 * q1
                w2 = e2 * q2
                ws = 1.0 / (w0 + w1 + w2)
                w0 *= ws
                w1 *= ws
                w2 *= ws
                r = min(max(w0 * r0 + w1 * r1 + w2 * r2, 0.0), 1.0)
                g = min(max(w0 * g0 + w1 * g1 + w2 * g2, 0.0), 1.0)
                bl = min(max(w0 * c0b + w1 * c1b + w2 * c2b, 0.0), 1.0)
                al = min(max(w0 * al0 + w1 * al1 + w2 * al2, 0.0), 1.0)
            if blend:
                u = 1.0 - al
                for k in range(4):
                    if mask & (1 << k):
                        sk = (base + k) * 3
                        color[sk] = np.uint8(math.floor(r * al * 255.0 + color[sk] * u + 0.5))
                        color[sk + 1] = np.uint8(math.floor(g * al * 255.0 + color[sk + 1] * u + 0.5))
                        color[sk + 2] = np.uint8(math.floor(bl * al * 255.0 + color[sk + 2] * u + 0.5))
            else:
                r8 = np.uint8(math.floor(r * 255.0 + 0.5))
                g8 = np.uint8(math.floor(g * 255.0 + 0.5))
                b8 = np.uint8(math.floor(bl * 255.0 + 0.5))
                for k in range(4):
                    if mask & (1 << k):
                        sk = (base + k) * 3
                        color[sk] = r8
                        color[sk + 1] = g8
                        color[sk + 2] = b8


@njit(cache=True, inline="always", error_model="numpy", fastmath=True)
def _fan(x, y, q, c, n, ready, ctx, res, y0b, y1b, depth, color, blend):
    """Fill the fan of ``n`` screen vertices ``(x, y, 1/w)`` with the lit colours ``c``."""
    for t in range(1, n - 1):
        _tri(
            x[0],
            y[0],
            q[0],
            x[t],
            y[t],
            q[t],
            x[t + 1],
            y[t + 1],
            q[t + 1],
            0,
            t,
            t + 1,
            c,
            ready,
            ctx,
            res,
            y0b,
            y1b,
            depth,
            color,
            blend,
        )


@njit(cache=True, error_model="numpy", fastmath=True)
def _geom(
    gi,
    cpos,
    cnor,
    tris,
    cstarts,
    starts,
    rbound,
    material,
    powlut,
    gp,
    gm,
    col4,
    cam,
    lights,
    tanh,
    near,
    res,
    y0b,
    y1b,
    depth,
    color,
    blend,
    ex,
    ey,
    ew,
    sx,
    sy,
    cc,
    done,
    x,
    y,
    q,
    c,
):
    px, py, pz = gp[0], gp[1], gp[2]
    cx, cy, cz = cam[0], cam[1], cam[2]
    rx, ry, rz = cam[3], cam[4], cam[5]
    ux, uy, uz = cam[6], cam[7], cam[8]
    fx, fy, fz = cam[9], cam[10], cam[11]
    dx, dy, dz = px - cx, py - cy, pz - cz
    w = dx * fx + dy * fy + dz * fz
    rb = rbound[gi]
    if w + rb < near:
        return
    xe = dx * rx + dy * ry + dz * rz
    ye = dx * ux + dy * uy + dz * uz
    cs = 1.0 / math.sqrt(1.0 + tanh * tanh)
    sn = tanh * cs
    if xe * cs - w * sn > rb or -xe * cs - w * sn > rb or ye * cs - w * sn > rb or -ye * cs - w * sn > rb:
        return
    half = 0.5 * res
    scale = half / tanh
    if w - rb > near:
        # Rows the bounding sphere can cover; a band that misses them draws nothing of this geom.
        top = max((ye + rb) / (w - rb), (ye + rb) / (w + rb))
        bottom = min((ye - rb) / (w - rb), (ye - rb) / (w + rb))
        if half - top * scale >= y1b or half - bottom * scale < y0b:
            return
    spec, emis = material[gi, 0], material[gi, 2]
    lut = powlut[int(material[gi, 3])]
    r, g, bb, a = col4[0], col4[1], col4[2], col4[3]
    c0 = cstarts[gi]
    nc = cstarts[gi + 1] - c0
    ctx = (ex, ey, ew, cnor, c0, gm, cam, r, g, bb, a, spec, lut, emis, lights)
    ready = np.ones(4, np.bool_)
    # Camera-space positions of the geom's corners; camera-relative world offsets for lighting.
    for j in range(nc):
        lx, ly, lz = cpos[c0 + j, 0], cpos[c0 + j, 1], cpos[c0 + j, 2]
        vx = gm[0] * lx + gm[1] * ly + gm[2] * lz + dx
        vy = gm[3] * lx + gm[4] * ly + gm[5] * lz + dy
        vz = gm[6] * lx + gm[7] * ly + gm[8] * lz + dz
        e = vx * rx + vy * ry + vz * rz
        f = vx * ux + vy * uy + vz * uz
        h = vx * fx + vy * fy + vz * fz
        ex[j] = e
        ey[j] = f
        ew[j] = h
        if h >= near:
            sx[j] = half + e / h * scale
            sy[j] = half - f / h * scale
        done[j] = False
    for t in range(starts[gi], starts[gi + 1]):
        k0 = tris[t, 0] - c0
        k1 = tris[t, 1] - c0
        k2 = tris[t, 2] - c0
        b0 = ew[k0] < near
        b1 = ew[k1] < near
        b2 = ew[k2] < near
        if b0 and b1 and b2:
            continue
        clip = b0 or b1 or b2
        if not clip:
            x0, y0, x1, y1, x2, y2 = sx[k0], sy[k0], sx[k1], sy[k1], sx[k2], sy[k2]
            area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
            # Counter-clockwise in GL window coordinates (y up) is clockwise here.
            if area >= 0.0:
                continue
            fy0 = min(y0, min(y1, y2))
            fy1 = max(y0, max(y1, y2))
            if fy1 < y0b or fy0 >= y1b:
                continue
            fx0 = min(x0, min(x1, x2))
            fx1 = max(x0, max(x1, x2))
            if fx1 < 0 or fx0 >= res:
                continue
            # Samples lie on a quarter-pixel grid offset by 1/8: a box between grid lines holds none.
            if math.floor(fx1 * 4.0 - 0.5) < math.ceil(fx0 * 4.0 - 0.5) or math.floor(fy1 * 4.0 - 0.5) < math.ceil(
                fy0 * 4.0 - 0.5
            ):
                continue
        if not clip:
            _tri(
                x0,
                y0,
                1.0 / ew[k0],
                x1,
                y1,
                1.0 / ew[k1],
                x2,
                y2,
                1.0 / ew[k2],
                k0,
                k1,
                k2,
                cc,
                done,
                ctx,
                res,
                y0b,
                y1b,
                depth,
                color,
                blend,
            )
            continue
        for kk in range(3):
            k = tris[t, kk] - c0
            if not done[k]:
                _light_corner(k, cc, done, ctx)
        # Clip against the near plane (Sutherland-Hodgman), then cull the clipped polygon.
        n = 0
        for kk in range(3):
            ka = tris[t, kk] - c0
            kb = tris[t, (kk + 1) % 3] - c0
            ina = ew[ka] >= near
            inb = ew[kb] >= near
            if ina:
                x[n] = ex[ka]
                y[n] = ey[ka]
                q[n] = ew[ka]
                for j in range(4):
                    c[n, j] = cc[ka, j]
                n += 1
            if ina != inb:
                s = (near - ew[ka]) / (ew[kb] - ew[ka])
                x[n] = ex[ka] + s * (ex[kb] - ex[ka])
                y[n] = ey[ka] + s * (ey[kb] - ey[ka])
                q[n] = near
                for j in range(4):
                    c[n, j] = cc[ka, j] + s * (cc[kb, j] - cc[ka, j])
                n += 1
        for k in range(n):
            x[k] = half + x[k] / q[k] * scale
            y[k] = half - y[k] / q[k] * scale
            q[k] = 1.0 / q[k]
        area = 0.0
        for k in range(n):
            k2 = (k + 1) % n
            area += x[k] * y[k2] - x[k2] * y[k]
        if area >= 0.0:
            continue
        _fan(x, y, q, c, n, ready, ctx, res, y0b, y1b, depth, color, blend)


@njit(cache=True, inline="always", error_model="numpy", fastmath=True)
def _cube(faces, dx, dy, dz, out):
    """Bilinear, clamp-to-edge lookup of an OpenGL cube map along ``(dx, dy, dz)``."""
    ax, ay, az = abs(dx), abs(dy), abs(dz)
    if ax >= ay and ax >= az:
        f = 0 if dx > 0 else 1
        sc, tc, ma = (-dz if dx > 0 else dz), -dy, ax
    elif ay >= az:
        f = 2 if dy > 0 else 3
        sc, tc, ma = dx, (dz if dy > 0 else -dz), ay
    else:
        f = 4 if dz > 0 else 5
        sc, tc, ma = (dx if dz > 0 else -dx), -dy, az
    n = faces.shape[1]
    u = (sc / ma + 1.0) * 0.5 * n - 0.5
    v = (tc / ma + 1.0) * 0.5 * n - 0.5
    u0 = int(math.floor(u))
    v0 = int(math.floor(v))
    fu = u - u0
    fv = v - v0
    ua = min(max(u0, 0), n - 1)
    ub = min(max(u0 + 1, 0), n - 1)
    va = min(max(v0, 0), n - 1)
    vb = min(max(v0 + 1, 0), n - 1)
    for ch in range(3):
        top = faces[f, va, ua, ch] * (1 - fu) + faces[f, va, ub, ch] * fu
        bot = faces[f, vb, ua, ch] * (1 - fu) + faces[f, vb, ub, ch] * fu
        out[ch] = top * (1 - fv) + bot * fv


@njit(cache=True, error_model="numpy", fastmath=True)
def _fill_sky(sky, plate, res, y0b, y1b, depth, color):
    """Skybox where no opaque sample was drawn, shaded once per pixel at its centre."""
    flat = sky[0] >= 0
    for py in range(y0b, y1b):
        for px in range(res):
            p0 = ((py - y0b) * res + px) * 4
            for k in range(4):
                if depth[p0 + k] == 0:
                    for ch in range(3):
                        color[(p0 + k) * 3 + ch] = np.uint8(sky[ch]) if flat else plate[py, px, ch]


@njit(parallel=True, cache=True, error_model="numpy", fastmath=True)
def _render(
    cpos,
    cnor,
    tris,
    cstarts,
    starts,
    rbound,
    sort_rbound,
    material,
    powlut,
    gpos,
    gmat,
    rgba,
    cams,
    lights,
    sky,
    plates,
    plate_index,
    tanh,
    near,
    ss,
    bands,
    out,
):
    """Draw ``out`` (B, size, size, 3); each frame splits into ``bands`` row bands drawn as separate tasks."""
    B, G = gpos.shape[0], gpos.shape[1]
    size = out.shape[1]
    res = size * ss
    per = -(-size // bands)
    maxc = 0
    for gi in range(G):
        maxc = max(maxc, cstarts[gi + 1] - cstarts[gi])
    jobs = B * bands
    nthr = min(numba.get_num_threads(), jobs)
    # One task per thread reuses its buffers across the frames it draws.
    for task in prange(nthr):
        depth_all = np.empty(per * ss * res * 4, np.float32)
        color_all = np.empty(per * ss * res * 12, np.uint8)
        ex = np.empty(maxc)
        ey = np.empty(maxc)
        ew = np.empty(maxc)
        sx = np.empty(maxc)
        sy = np.empty(maxc)
        cc = np.empty((maxc, 4))
        done = np.empty(maxc, np.bool_)
        x = np.empty(4)
        y = np.empty(4)
        q = np.empty(4)
        c = np.empty((4, 4))
        for job in range(task, jobs, nthr):
            _frame(
                job,
                bands,
                per,
                size,
                ss,
                res,
                cpos,
                cnor,
                tris,
                cstarts,
                starts,
                rbound,
                sort_rbound,
                material,
                powlut,
                gpos,
                gmat,
                rgba,
                cams,
                lights,
                sky,
                plates,
                plate_index,
                tanh,
                near,
                out,
                depth_all,
                color_all,
                ex,
                ey,
                ew,
                sx,
                sy,
                cc,
                done,
                x,
                y,
                q,
                c,
            )


@njit(cache=True, error_model="numpy", fastmath=True)
def _frame(
    job,
    bands,
    per,
    size,
    ss,
    res,
    cpos,
    cnor,
    tris,
    cstarts,
    starts,
    rbound,
    sort_rbound,
    material,
    powlut,
    gpos,
    gmat,
    rgba,
    cams,
    lights,
    sky,
    plates,
    plate_index,
    tanh,
    near,
    out,
    depth_all,
    color_all,
    ex,
    ey,
    ew,
    sx,
    sy,
    cc,
    done,
    x,
    y,
    q,
    c,
):
    G = gpos.shape[1]
    b = job // bands
    band = job % bands
    o0 = band * per
    o1 = min(size, o0 + per)
    if o0 >= o1:
        return
    y0b = o0 * ss
    y1b = o1 * ss
    rows = y1b - y0b
    depth = depth_all[: rows * res * 4]
    color = color_all[: rows * res * 12]
    for i in range(rows * res * 4):
        depth[i] = 0
    for i in range(rows * res * 12):
        color[i] = 0
    cam = cams[b]
    # Opaque geoms near to far, so that hidden samples fail the depth test before they are shaded.
    near_first = np.empty(G)
    for gi in range(G):
        ddx = gpos[b, gi, 0] - cam[0]
        ddy = gpos[b, gi, 1] - cam[1]
        ddz = gpos[b, gi, 2] - cam[2]
        near_first[gi] = math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz) - rbound[gi]
    for gi in np.argsort(near_first):
        if rgba[b, gi, 3] >= TRANSPARENT_ALPHA:
            _geom(
                gi,
                cpos,
                cnor,
                tris,
                cstarts,
                starts,
                rbound,
                material,
                powlut,
                gpos[b, gi],
                gmat[b, gi],
                rgba[b, gi],
                cam,
                lights[b],
                tanh,
                near,
                res,
                y0b,
                y1b,
                depth,
                color,
                False,
                ex,
                ey,
                ew,
                sx,
                sy,
                cc,
                done,
                x,
                y,
                q,
                c,
            )
    _fill_sky(sky[b], plates[plate_index[b]], res, y0b, y1b, depth, color)
    # Translucent geoms sorted by distance to the camera less the bounding radius.
    nt = 0
    order = np.empty(G, np.int64)
    dist = np.empty(G)
    for gi in range(G):
        if rgba[b, gi, 3] < TRANSPARENT_ALPHA:
            ddx = gpos[b, gi, 0] - cam[0]
            ddy = gpos[b, gi, 1] - cam[1]
            ddz = gpos[b, gi, 2] - cam[2]
            dist[nt] = np.float32(math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)) - np.float32(sort_rbound[gi])
            order[nt] = gi
            nt += 1
    if nt > 0:
        idx = np.argsort(dist[:nt], kind="mergesort")
        for p in range(2 * nt):
            gi = order[idx[p]] if p < nt else order[idx[2 * nt - 1 - p]]
            _geom(
                gi,
                cpos,
                cnor,
                tris,
                cstarts,
                starts,
                rbound,
                material,
                powlut,
                gpos[b, gi],
                gmat[b, gi],
                rgba[b, gi],
                cam,
                lights[b],
                tanh,
                near,
                res,
                y0b,
                y1b,
                depth,
                color,
                True,
                ex,
                ey,
                ew,
                sx,
                sy,
                cc,
                done,
                x,
                y,
                q,
                c,
            )
    # Resolve the samples, then average supersampled pixels.
    if ss == 1:
        for oy in range(o0, o1):
            for ox in range(size):
                p0 = ((oy - o0) * res + ox) * 12
                for ch in range(3):
                    v = (
                        np.int32(color[p0 + ch])
                        + np.int32(color[p0 + 3 + ch])
                        + np.int32(color[p0 + 6 + ch])
                        + np.int32(color[p0 + 9 + ch])
                        + 2
                    )
                    out[b, oy, ox, ch] = np.uint8(v >> 2)
        return
    for oy in range(o0, o1):
        for ox in range(size):
            for ch in range(3):
                acc = 0
                for sy_ in range(ss):
                    for sx_ in range(ss):
                        p0 = (((oy - o0) * ss + sy_) * res + ox * ss + sx_) * 4
                        acc += (
                            int(color[(p0) * 3 + ch])
                            + int(color[(p0 + 1) * 3 + ch])
                            + int(color[(p0 + 2) * 3 + ch])
                            + int(color[(p0 + 3) * 3 + ch])
                            + 2
                        ) // 4
                out[b, oy, ox, ch] = np.uint8(np.rint(acc / (ss * ss)))
