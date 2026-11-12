"""Batched CUDA rasterizer for Cube/Scene frames, following MuJoCo 3.17's fixed-function OpenGL renderer.

The scene is the OpenGL backend's render model with the arm meshes simplified as in ``opengl_fast``: the same
geoms, materials, lights, camera, floor grid texture and skybox. Each frame has the 4 samples per pixel of
MuJoCo's multisampled buffer, at NVIDIA's sample positions; a pixel is shaded at its centre when the centre lies
in the triangle, else at the sample. Lighting is per vertex with the headlight and the model lights, a local
viewer and OpenGL's clamp of each vertex colour.

As in ``mjr_render``, opaque geoms are lit by the lights without shadows, and the shadow light adds its light
where the top-down shadow map has no higher caster at the sample's texel; the map holds each caster's height
less MuJoCo's polygon offset. The floor and the skybox depend only on the camera, the light intensity and the
floor colours, so they are drawn once per such look: the floor interpolates the lit colours of MuJoCo's plane
grid and samples the grid texture with OpenGL's trilinear filter. nvdiffrast draws the opaque geoms other than
the floor. Triton kernels draw the shadow map and list the fragments of the transparent geoms (the see-through
arm and the window glass) per sample, then blend them as MuJoCo does: geoms sorted by camera distance less
their bounding radius, first front to back, then back to front, each geom in its triangle order, without depth
writes. Frames differ from the OpenGL frames by small edge, shadow and rounding differences.
"""

from __future__ import annotations

import math
import threading

import numpy as np
import torch

from . import manipulation_opengl as _gl
from .simulation import render_state as pixels

_LOCK = threading.RLock()
_SCENES: dict = {}
_CONTEXTS: dict = {}
_SKY: dict = {}

#: MuJoCo draws geoms with a smaller alpha after the opaque geoms, with blending.
TRANSPARENT_ALPHA = 0.995
#: Arm meshes keep one vertex per cell of this grid in metres, as ``opengl_fast`` draws them.
CLUSTER_CELL = 0.003
#: Transparent fragments kept per sample.
LAYERS = 16
#: Shadow map texels per side over MuJoCo's shadow clip square.
SHADOW_SIZE = 2048
#: A receiver lies in shadow when the offset occluder is higher by more than this many metres.
SHADOW_BIAS = 1e-5

PLANE, CAPSULE, CYLINDER, BOX, MESH = 0, 3, 5, 6, 7
# Colour sources of a geom: its model RGBA, cube.color, agent.color, a Scene button or a Scene handle.
FIXED, CUBE, AGENT, BUTTON0, BUTTON1, HANDLE0, HANDLE1 = range(7)


class _Scene:
    """Static triangles, materials, kinematic tree and lights of one family on one device."""


def _quads(quads, normals):
    v, n = [], []
    for q, qn in zip(quads, normals):
        for a, b, c in ((0, 1, 3), (1, 2, 3)):
            v.append((q[a], q[b], q[c]))
            n.append((qn[a], qn[b], qn[c]))
    return v, n


def _box(size, numquads):
    """MuJoCo's builtin box: ``numquads`` x ``numquads`` quads per face, scaled to ``size``."""
    d = 2.0 / numquads
    quads, normals = [], []
    for x in range(numquads):
        for y in range(numquads):
            a0, a1, b0, b1 = d * x - 1, d * (x + 1) - 1, d * y - 1, d * (y + 1) - 1
            for normal, quad in (
                ((0, 0, 1), ((a0, b0, 1), (a1, b0, 1), (a1, b1, 1), (a0, b1, 1))),
                ((0, 0, -1), ((a0, b1, -1), (a1, b1, -1), (a1, b0, -1), (a0, b0, -1))),
                ((1, 0, 0), ((1, a0, b0), (1, a1, b0), (1, a1, b1), (1, a0, b1))),
                ((-1, 0, 0), ((-1, a0, b1), (-1, a1, b1), (-1, a1, b0), (-1, a0, b0))),
                ((0, -1, 0), ((a0, -1, b0), (a1, -1, b0), (a1, -1, b1), (a0, -1, b1))),
                ((0, 1, 0), ((a0, 1, b1), (a1, 1, b1), (a1, 1, b0), (a0, 1, b0))),
            ):
                quads.append(quad)
                normals.append((normal,) * 4)
    v, n = _quads(quads, normals)
    return _scaled(np.asarray(v, np.float64), np.asarray(n, np.float64), size)


def _cylinder(size, nslice, nstack):
    """MuJoCo's builtin closed cylinder: two disks and the side quads, scaled to ``size``."""
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
                qv, qn = _quads([q], [(normal,) * 4])
                v.extend(qv)
                n.extend(qn)

    disk(-1, nstack // 2)
    side_v, side_n = _open_cylinder(nslice, nstack)
    v.extend(side_v)
    n.extend(side_n)
    disk(1, nstack // 2)
    return _scaled(np.asarray(v, np.float64), np.asarray(n, np.float64), size)


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
    face = np.cross(vert[:, 1] - vert[:, 0], vert[:, 2] - vert[:, 0]).astype(np.float32)
    face /= np.maximum(np.linalg.norm(face, axis=-1, keepdims=True), np.float32(1e-15))
    keep = (corner * face[:, None, :]).sum(-1) >= 0.8
    normals = np.where(keep[..., None], corner, face[:, None, :])
    return vert.astype(np.float64), normals.astype(np.float64)


def _plane_grid(m, g):
    """Triangles of MuJoCo's display list of one plane geom: a grid of quads."""
    size = np.asarray(m.geom_size[g], np.float64)
    grids = []
    for k in range(2):
        if size[k] <= 0:
            raise NotImplementedError("Cube/Scene nvdiff rasterizer draws finite planes only")
        step = max(size[2], size[k] / (200 - 2))
        n = int(math.floor(size[k] / max(1e-15, step)))
        n -= n % 2
        pad = size[k] - n * step
        n += 2
        line = []
        for x in range(n):
            if x == 0:
                left = -size[k]
                right = left + pad
            else:
                left = -size[k] + pad + (x - 1) * step * 2
                right = left + (pad if x == n - 1 else step * 2)
            line.append(left)
            if x == n - 1:
                line.append(max(left, right))
        grids.append(np.asarray(line))
    gx, gy = grids
    quads = [
        ((gx[x], gy[y], 0.0), (gx[x + 1], gy[y], 0.0), (gx[x + 1], gy[y + 1], 0.0), (gx[x], gy[y + 1], 0.0))
        for x in range(len(gx) - 1)
        for y in range(len(gy) - 1)
    ]
    v, n = _quads(quads, [((0.0, 0.0, 1.0),) * 4] * len(quads))
    return np.asarray(v, np.float64), np.asarray(n, np.float64)


def _half_sphere(sign, nslice, nstack):
    f32 = np.float32

    def vertex(az, el):
        p = (math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), sign + math.sin(el))
        return p, (p[0], p[1], p[2] - sign)

    v, n = [], []
    el1 = float(f32(math.pi / 2 * sign * (nstack - 1) / nstack))
    for j in range(nslice):
        az1, az2 = float(f32(2 * math.pi * j / nslice)), float(f32(2 * math.pi * (j + 1) / nslice))
        (p1, n1), (p2, n2) = vertex(az1, el1), vertex(az2, el1)
        p3, n3 = (0.0, 0.0, 2.0 * sign), (0.0, 0.0, float(sign))
        if sign > 0:
            v.append((p1, p2, p3))
            n.append((n1, n2, n3))
        else:
            v.append((p3, p2, p1))
            n.append((n3, n2, n1))
    quads, normals = [], []
    for i in range(nstack - 1):
        el1 = float(f32(math.pi / 2 * sign * i / nstack))
        el2 = float(f32(math.pi / 2 * sign * (i + 1) / nstack))
        for j in range(nslice):
            az1, az2 = float(f32(2 * math.pi * j / nslice)), float(f32(2 * math.pi * (j + 1) / nslice))
            q = [vertex(az1, el1), vertex(az2, el1), vertex(az2, el2), vertex(az1, el2)]
            if sign < 0:
                q = q[::-1]
            quads.append([p for p, _ in q])
            normals.append([c for _, c in q])
    qv, qn = _quads(quads, normals)
    return v + qv, n + qn


def _open_cylinder(nslice, nstack):
    f32 = np.float32
    quads, normals = [], []
    for i in range(nstack):
        h1, h2 = 2 * i / nstack - 1, 2 * (i + 1) / nstack - 1
        for j in range(nslice):
            az1, az2 = float(f32(2 * math.pi * j / nslice)), float(f32(2 * math.pi * (j + 1) / nslice))
            q = [(az1, h1), (az2, h1), (az2, h2), (az1, h2)]
            quads.append([(math.cos(a), math.sin(a), h) for a, h in q])
            normals.append([(math.cos(a), math.sin(a), 0.0) for a, _ in q])
    return _quads(quads, normals)


def _capsule(radius, half, nslice, nstack):
    """MuJoCo's capsule: an open cylinder and two half spheres of ``nstack // 2`` stacks."""
    v, n = _open_cylinder(nslice, nstack)
    v = [[(radius * x, radius * y, half * z) for x, y, z in t] for t in v]
    for sign in (1, -1):
        hv, hn = _half_sphere(sign, nslice, nstack // 2)
        v += [[(radius * x, radius * y, radius * z + sign * (half - radius)) for x, y, z in t] for t in hv]
        n += hn
    v, n = np.asarray(v, np.float64), np.asarray(n, np.float64)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-30)
    return v, n


def _mean_cube_texture(m, t):
    w, h, nc = int(m.tex_width[t]), int(m.tex_height[t]), int(m.tex_nchannel[t])
    data = np.asarray(m.tex_data[int(m.tex_adr[t]) : int(m.tex_adr[t]) + w * h * nc], np.float64)
    return data.reshape(-1, nc)[:, :3].mean(0) / 255.0


def _grid_weights(env, m, t):
    """Per-texel weights of the grid texture's two colours and its mark colour, and the colour rounding."""
    w, h, nc = int(m.tex_width[t]), int(m.tex_height[t]), int(m.tex_nchannel[t])
    data = np.asarray(m.tex_data[int(m.tex_adr[t]) : int(m.tex_adr[t]) + w * h * nc]).reshape(h, w, nc)[..., :3]
    floor = env.variation_space["floor"]["color"].value
    colors = [np.asarray(floor[0]), np.asarray(floor[1]), None]
    weights = np.zeros((h, w, 3), np.float32)
    matched = np.zeros((h, w), bool)
    rounding = None
    for k, color in enumerate(colors[:2]):
        for mode, q in (("round", np.round(255 * color)), ("trunc", np.floor(255 * color))):
            hit = np.all(data == q.astype(np.uint8), -1) & ~matched
            if hit.sum() > 0.05 * h * w:
                weights[hit, k] = 1
                matched |= hit
                rounding = rounding or mode
                break
    rest = ~matched
    mark = np.unique(data[rest].reshape(-1, 3), axis=0)
    if len(mark) != 1:
        raise ValueError("Cube/Scene grid texture has more than three colours")
    weights[rest, 2] = 1
    if rounding is None:
        raise ValueError("Cube/Scene grid texture colours do not follow the floor colours")
    return weights, mark[0].astype(np.float32) / 255.0, rounding


def _floor(env, m, g, mid, rgba):
    """Grid lines, texture scale, material and grid texture weights of the floor plane."""
    import mujoco

    size = np.asarray(m.geom_size[g], np.float64)
    v, _ = _plane_grid(m, g)
    lines = [np.unique(v[..., k]) for k in range(2)]
    for line in lines:
        step = np.diff(line)
        step = step[step > 0]
        if not np.allclose(step, step[0], rtol=0, atol=1e-12):
            raise NotImplementedError("Cube/Scene nvdiff rasterizer expects an even floor grid")
    tex = int(m.mat_texid[mid][int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
    repeat = np.asarray(m.mat_texrepeat[mid], np.float64)
    scale = np.where(repeat > 0, repeat, 1.0)
    if bool(m.mat_texuniform[mid]):
        scale = scale * np.where(size[:2] > 0, size[:2], 1.0)
    weights, mark, rounding = _grid_weights(env, m, tex)
    return {
        "size": size[:2],
        "origin": np.array([lines[0][0], lines[1][0]]),
        "step": np.array([np.diff(lines[0]).max(), np.diff(lines[1]).max()]),
        "cells": np.array([len(lines[0]) - 1, len(lines[1]) - 1]),
        "scale": scale,
        "rgba": np.asarray(rgba, np.float32),
        "material": (float(m.mat_specular[mid]), float(m.mat_shininess[mid]) * 128.0, float(m.mat_emission[mid])),
        "weights": weights,
        "mark": mark,
        "rounding": rounding,
    }


def _build(family, device):
    import mujoco

    env = pixels.build_model(family)
    try:
        m = env.model
        _gl.cluster_meshes(m, CLUSTER_CELL)
        q = m.vis.quality
        nslice, nstack, numquads = int(q.numslices), int(q.numstacks), int(q.numquads)
        s = _Scene()
        s.family = family
        s.floor = None
        hidden = {int(i) for ids in env._cube_target_geom_ids_list for i in np.atleast_1d(ids)}
        cube = {int(i) for ids in env._cube_geom_ids_list for i in np.atleast_1d(ids)}
        agent = {int(m.material(name).id) for name in pixels.AGENT_MATERIALS}
        locks = {}
        lock_materials = {}
        if family == "Scene":
            for i, ids in enumerate(env._button_geom_ids_list):
                for gid in np.atleast_1d(ids):
                    locks[int(gid)] = BUTTON0 + i
            for i, name in enumerate(("drawer_handle", "window_handle")):
                lock_materials[int(m.material(name).id)] = HANDLE0 + i
            s.white = np.asarray(env._colors["white"][:3], np.float32)
            s.btn_red = np.asarray(m.material("btn_red").rgba, np.float32)
        default = np.float32([0.5, 0.5, 0.5, 1.0])
        geoms, sources, rgbas, mats, textures, cube_flags = [], [], [], [], [], []
        verts, norms = [], []
        for g in range(m.ngeom):
            if int(m.geom_group[g]) > 2 or g in hidden:
                continue
            mid = int(m.geom_matid[g])
            rgba = np.asarray(m.geom_rgba[g], np.float32).copy()
            if mid >= 0 and np.array_equal(rgba, default):
                rgba = np.asarray(m.mat_rgba[mid], np.float32).copy()
            if rgba[3] == 0:
                continue
            kind = int(m.geom_type[g])
            size = np.asarray(m.geom_size[g], np.float64)
            if kind == MESH:
                v, n = _mesh(m, int(m.geom_dataid[g]))
            elif kind == BOX:
                v, n = _box(np.ones(3) if g in cube else size, numquads)
            elif kind == CYLINDER:
                v, n = _cylinder(size[[0, 0, 1]], nslice, nstack)
            elif kind == CAPSULE:
                v, n = _capsule(size[0], size[1], nslice, nstack)
            elif kind == PLANE:
                if s.floor is not None:
                    raise NotImplementedError("Cube/Scene nvdiff rasterizer draws one plane")
                if np.any(m.geom_pos[g] != 0) or np.any(m.geom_quat[g] != [1, 0, 0, 0]) or int(m.geom_bodyid[g]) != 0:
                    raise NotImplementedError("Cube/Scene nvdiff rasterizer draws a plane at the world origin")
                s.floor = _floor(env, m, g, mid, rgba)
                continue
            else:
                raise NotImplementedError(f"Cube/Scene nvdiff rasterizer has no geom type {kind}")
            texture = 0
            tex = int(m.mat_texid[mid][int(mujoco.mjtTextureRole.mjTEXROLE_RGB)]) if mid >= 0 else -1
            if tex >= 0:
                if int(m.tex_type[tex]) == int(mujoco.mjtTexture.mjTEXTURE_2D):
                    raise NotImplementedError("Cube/Scene nvdiff rasterizer draws a 2D texture on the floor only")
                texture = 2
            if g in cube:
                source = CUBE
            elif mid in agent:
                source = AGENT
            elif g in locks:
                source = locks[g]
            elif mid in lock_materials and np.array_equal(np.asarray(m.geom_rgba[g], np.float32), default):
                source = lock_materials[mid]
            else:
                source = FIXED
            spec, shin, emis = (
                (0.5, 0.5, 0.0) if mid < 0 else (m.mat_specular[mid], m.mat_shininess[mid], m.mat_emission[mid])
            )
            geoms.append(g)
            sources.append(source)
            rgbas.append(rgba)
            mats.append((float(spec), float(shin) * 128.0, float(emis)))
            textures.append(_mean_cube_texture(m, tex) if texture == 2 else np.ones(3))
            cube_flags.append(g in cube)
            verts.append(v)
            norms.append(n)
        s.geoms = np.asarray(geoms, np.int64)
        G = len(geoms)
        # Corners that share a position and a normal are one vertex.
        cpos, cnor, cgeom, tris, tri_geom, starts = [], [], [], [], [], [0]
        base = 0
        for i, (v, n) in enumerate(zip(verts, norms)):
            key = np.concatenate([v.reshape(-1, 3), n.reshape(-1, 3)], 1)
            uniq, inv = np.unique(key, axis=0, return_inverse=True)
            cpos.append(uniq[:, :3])
            cnor.append(uniq[:, 3:6])
            cgeom.append(np.full(len(uniq), i))
            tris.append(inv.reshape(-1, 3) + base)
            tri_geom.append(np.full(len(v), i))
            base += len(uniq)
            starts.append(starts[-1] + len(v))
        pos = np.concatenate(cpos)
        nor = np.concatenate(cnor)
        nor /= np.maximum(np.linalg.norm(nor, axis=-1, keepdims=True), 1e-30)
        tri = np.concatenate(tris)
        tri_geom = np.concatenate(tri_geom)
        alpha = np.asarray(rgbas, np.float32)[:, 3]
        transparent = alpha < TRANSPARENT_ALPHA
        t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)  # noqa: E731
        s.vpos, s.vnor = t(pos), t(nor)
        s.vgeom = t(np.concatenate(cgeom), torch.int64)
        opaque_tri = tri[~transparent[tri_geom]]
        s.tri_opaque = t(opaque_tri, torch.int32)
        used, compact = np.unique(opaque_tri, return_inverse=True)
        s.opaque_vertices = t(used, torch.int64)
        s.tri_opaque_compact = t(compact.reshape(-1, 3), torch.int32)
        s.opaque_geom32 = t(tri_geom[~transparent[tri_geom]], torch.int32)
        trans = transparent[tri_geom]
        s.tri_trans = t(tri[trans], torch.int32)
        # Triangle order within a geom: its index from the geom's first triangle.
        local = np.arange(len(tri)) - np.asarray(starts)[tri_geom]
        s.trans_geom32 = t(tri_geom[trans], torch.int32)
        s.trans_local32 = t(local[trans], torch.int32)
        # Every drawn geom casts a shadow, the transparent ones too.
        s.tri_caster = t(tri, torch.int32)
        s.G = G
        s.trans_index = t(np.flatnonzero(transparent), torch.int64)
        s.sources = np.asarray(sources)
        s.rgba = t(np.asarray(rgbas, np.float32))
        s.material = t(np.asarray(mats, np.float32))
        s.texmod = t(np.asarray(textures, np.float32))
        s.cube_flags = t(np.asarray(cube_flags), torch.bool)
        s.rbound = np.asarray(m.geom_rbound[geoms], np.float32)
        s.rbound_t = t(s.rbound)
        if s.floor is None:
            raise NotImplementedError("Cube/Scene nvdiff rasterizer expects the floor plane")
        s.floor["weights"] = torch.as_tensor(s.floor["weights"], device=device)
        # Kinematic tree, as mj_kinematics reads it.
        s.nq = int(m.nq)
        s.body_parent = np.asarray(m.body_parentid)
        s.body_pos = np.asarray(m.body_pos, np.float64)
        s.body_quat = np.asarray(m.body_quat, np.float64)
        s.joints = [
            [
                (
                    int(m.jnt_type[j]),
                    int(m.jnt_qposadr[j]),
                    np.asarray(m.jnt_axis[j], np.float64),
                    np.asarray(m.jnt_pos[j], np.float64),
                )
                for j in range(int(m.body_jntadr[b]), int(m.body_jntadr[b]) + int(m.body_jntnum[b]))
            ]
            for b in range(m.nbody)
        ]
        s.qpos0 = np.asarray(m.qpos0, np.float64)
        s.geom_body = np.asarray(m.geom_bodyid[geoms])
        s.geom_pos = np.asarray(m.geom_pos[geoms], np.float64)
        s.geom_quat = np.asarray(m.geom_quat[geoms], np.float64)
        bodies = set()
        for b in s.geom_body:
            while b > 0:
                bodies.add(int(b))
                b = int(s.body_parent[b])
        s.bodies = sorted(bodies)
        # Camera and lights, with the fixed camera on the world body.
        cam = m.camera("front_pixels").id
        if int(m.cam_bodyid[cam]) != 0:
            raise ValueError("Cube/Scene front_pixels must be a world camera")
        s.cam_pos = np.asarray(m.cam_pos[cam], np.float64)
        s.cam_mat = _quat_mat(np.asarray(m.cam_quat[cam], np.float64))
        s.fovy = float(m.cam_fovy[cam])
        s.znear = float(m.vis.map.znear * m.stat.extent)
        s.zfar = float(m.vis.map.zfar * m.stat.extent)
        head = m.vis.headlight
        s.head = (
            np.asarray(head.ambient, np.float32),
            np.asarray(head.diffuse, np.float32),
            np.asarray(head.specular, np.float32),
        )
        if not int(head.active):
            raise ValueError("Cube/Scene nvdiff rasterizer expects MuJoCo's headlight")
        s.lights = []
        for i in range(m.nlight):
            if not bool(m.light_active[i]) or int(m.light_bodyid[i]) != 0:
                raise ValueError("Cube/Scene nvdiff rasterizer expects active world lights")
            directional = int(m.light_type[i]) == int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
            if not directional and int(m.light_type[i]) != int(mujoco.mjtLightType.mjLIGHT_SPOT):
                raise ValueError("Cube/Scene nvdiff rasterizer draws directional and spot lights")
            cutoff = float(m.light_cutoff[i])
            # OpenGL rejects a spot cutoff outside [0, 90] other than 180 and keeps 180: no cone.
            cone = not directional and cutoff <= 90.0
            s.lights.append(
                {
                    "directional": directional,
                    "pos": np.asarray(m.light_pos[i], np.float64),
                    "dir": np.asarray(m.light_dir[i], np.float64) / np.linalg.norm(m.light_dir[i]),
                    "ambient": np.asarray(m.light_ambient[i], np.float32),
                    "diffuse": np.asarray(m.light_diffuse[i], np.float32),
                    "specular": np.asarray(m.light_specular[i], np.float32),
                    "attenuation": np.asarray(m.light_attenuation[i], np.float32),
                    "cone": (cutoff, float(m.light_exponent[i])) if cone else None,
                    "shadow": bool(m.light_castshadow[i]),
                    "name": m.light(i).name,
                }
            )
        shadows = [k for k, light in enumerate(s.lights) if light["shadow"]]
        if len(shadows) != 1 or not s.lights[shadows[0]]["directional"]:
            raise ValueError("Cube/Scene nvdiff rasterizer expects one directional shadow light")
        s.shadow_light = shadows[0]
        if not np.allclose(s.lights[s.shadow_light]["dir"], [0, 0, -1]):
            raise ValueError("Cube/Scene nvdiff rasterizer expects the shadow light to point down")
        s.shadow_clip = float(m.stat.extent * m.vis.map.shadowclip)
        s.global_light = [k for k, light in enumerate(s.lights) if light["name"] == "global"][0]
        s.sky = _sky_texture(m, device)
        return s
    finally:
        env.close()


def _sky_texture(m, device):
    import mujoco

    for t in range(m.ntex):
        if int(m.tex_type[t]) == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX):
            w, h, nc = int(m.tex_width[t]), int(m.tex_height[t]), int(m.tex_nchannel[t])
            data = np.asarray(m.tex_data[int(m.tex_adr[t]) : int(m.tex_adr[t]) + w * h * nc], np.uint8).reshape(
                -1, w, nc
            )
            faces = np.repeat(data[None, :w], 6, 0) if w == h else data.reshape(6, w, w, nc)
            return torch.as_tensor(np.ascontiguousarray(faces[..., :3]), dtype=torch.float32, device=device) / 255.0
    return None


def scene(family, device):
    key = (family, str(device))
    with _LOCK:
        hit = _SCENES.get(key)
        if hit is None:
            hit = _SCENES[key] = _build(family, device)
        return hit


# ---------------------------------------------------------------- kinematics (float64, as mj_kinematics)


def _quat_mat(q):
    q = np.asarray(q, np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def _qmul(a, b):
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        -1,
    )


def _matvec(m, v):
    """``m @ v`` over leading axes, element by element so that a row does not depend on the batch."""
    return m[..., 0] * v[..., None, 0] + m[..., 1] * v[..., None, 1] + m[..., 2] * v[..., None, 2]


def _matmul(a, b):
    """``a @ b`` of trailing 3x3 or 4x4 matrices, element by element."""
    return sum(a[..., :, k : k + 1] * b[..., k : k + 1, :] for k in range(a.shape[-1]))


def _rot(q, v):
    return _matvec(_quat_mat(q), v)


def _normalize4(q):
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(n < 1e-15, np.array([1.0, 0.0, 0.0, 0.0]), q / np.maximum(n, 1e-15))


def kinematics(s, qpos):
    """Geom positions (B, G, 3) and rotation matrices (B, G, 3, 3) of the drawn geoms from ``qpos`` (B, nq)."""
    B = len(qpos)
    nb = len(s.body_parent)
    xpos = np.zeros((B, nb, 3))
    xquat = np.zeros((B, nb, 4))
    xquat[:, 0, 0] = 1
    for b in s.bodies:
        joints = s.joints[b]
        if len(joints) == 1 and joints[0][0] == 0:
            adr = joints[0][1]
            pos = qpos[:, adr : adr + 3].copy()
            quat = _normalize4(qpos[:, adr + 3 : adr + 7])
        else:
            p = int(s.body_parent[b])
            if p:
                pos = _matvec(_quat_mat(xquat[:, p]), s.body_pos[b]) + xpos[:, p]
                quat = _qmul(xquat[:, p], s.body_quat[b])
            else:
                pos = np.broadcast_to(s.body_pos[b], (B, 3)).copy()
                quat = np.broadcast_to(s.body_quat[b], (B, 4)).copy()
            for kind, adr, axis, jpos in joints:
                anchor = _rot(quat, jpos) + pos
                if kind == 2:
                    pos = pos + _rot(quat, axis) * (qpos[:, adr] - s.qpos0[adr])[:, None]
                elif kind == 3:
                    angle = (qpos[:, adr] - s.qpos0[adr])[:, None] * 0.5
                    local = np.concatenate([np.cos(angle), axis * np.sin(angle)], -1)
                    quat = _qmul(quat, local)
                    pos = anchor - _rot(quat, jpos)
                else:
                    raise NotImplementedError("Cube/Scene kinematics has hinge, slide and free joints")
        xpos[:, b] = pos
        xquat[:, b] = _normalize4(quat)
    body = s.geom_body
    gpos = _matvec(_quat_mat(xquat[:, body]), s.geom_pos) + xpos[:, body]
    gmat = _quat_mat(_qmul(xquat[:, body], s.geom_quat[None]))
    return gpos, gmat


# ---------------------------------------------------------------- per-frame inputs


def _camera(s, angle):
    """Camera rotation (B, 3, 3), columns right, up and backward, for camera.angle_delta (B, 2) in degrees."""
    yaw, pitch = np.deg2rad(angle[:, 0]), np.deg2rad(angle[:, 1])
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    one, zero = np.ones_like(cy), np.zeros_like(cy)
    rx = np.stack([np.stack([one, zero, zero], -1), np.stack([zero, cp, -sp], -1), np.stack([zero, sp, cp], -1)], -2)
    rz = np.stack([np.stack([cy, -sy, zero], -1), np.stack([sy, cy, zero], -1), np.stack([zero, zero, one], -1)], -2)
    return _matmul(_matmul(rx, rz), s.cam_mat)


def _colors(s, host, device):
    """Per-frame RGBA (B, G, 4) of the drawn geoms."""
    B = len(host["pose", "qpos"])
    rgba = np.broadcast_to(s.rgba.cpu().numpy(), (B, s.G, 4)).copy()
    cube = np.float32(host["appearance", "cube.color"]).reshape(B, 3)
    agent = np.float32(host["appearance", "agent.color"]).reshape(B, 3)
    for i, source in enumerate(s.sources):
        if source == CUBE:
            rgba[:, i, :3] = cube
            rgba[:, i, 3] = 1.0
        elif source == AGENT:
            rgba[:, i, :3] = agent
        elif source in (BUTTON0, BUTTON1, HANDLE0, HANDLE1):
            k = (source - BUTTON0) % 2
            lock = np.float32(host["appearance", "lock_color"]).reshape(B, 2, 3)[:, k]
            pressed = np.asarray(host["pose", "buttons"]).reshape(B, 2)[:, k] != 0
            color = np.where(pressed[:, None], s.white[None], lock)
            rgba[:, i, :3] = color
            rgba[:, i, 3] = 1.0
            if source in (BUTTON0, BUTTON1):
                # A geom RGBA equal to MuJoCo's default (0.5, 0.5, 0.5, 1) draws the material colour.
                default = np.all(color == np.float32(0.5), -1)
                rgba[default, i] = s.btn_red
    return torch.as_tensor(rgba, device=device)


def prepare(s, host, device):
    """Device inputs of ``draw`` from validated host RenderState columns."""
    qpos = np.asarray(host["pose", "qpos"], np.float64)
    B = len(qpos)
    gpos, gmat = kinematics(s, qpos)
    scale = np.ones((B, s.G))
    flags = s.cube_flags.cpu().numpy()
    scale[:, flags] = np.asarray(host["appearance", "cube.size"], np.float64).reshape(B, 1)
    transform = np.concatenate([gmat * scale[..., None, None], gpos[..., None]], -1)
    cam = _camera(s, np.asarray(host["appearance", "camera.angle_delta"], np.float64).reshape(B, 2))
    out = {
        "transform": torch.as_tensor(transform, dtype=torch.float32, device=device),
        "rotation": torch.as_tensor(gmat, dtype=torch.float32, device=device),
        "position": torch.as_tensor(gpos, dtype=torch.float32, device=device),
        "cam_mat": torch.as_tensor(cam, dtype=torch.float32, device=device),
        "cam_pos": torch.as_tensor(np.broadcast_to(s.cam_pos, (B, 3)), dtype=torch.float32, device=device),
        "intensity": torch.as_tensor(
            np.float32(np.asarray(host["appearance", "light.intensity"], np.float64).reshape(B)), device=device
        ),
        "floor": torch.as_tensor(
            np.asarray(host["appearance", "floor.color"], np.float64).reshape(B, 2, 3), device=device
        ),
        "rgba": _colors(s, host, device),
    }
    return out


# ---------------------------------------------------------------- drawing


def _context(device):
    import nvdiffrast.torch as dr

    key = str(device)
    with _LOCK:
        hit = _CONTEXTS.get(key)
        if hit is None:
            hit = _CONTEXTS[key] = dr.RasterizeCudaContext(device=torch.device(device))
        return hit


def _projection(s):
    f = 1.0 / math.tan(math.radians(s.fovy) / 2.0)
    n, fa = s.znear, s.zfar
    p = np.zeros((4, 4))
    p[0, 0] = p[1, 1] = f
    p[2, 2] = (fa + n) / (n - fa)
    p[2, 3] = 2 * fa * n / (n - fa)
    p[3, 2] = -1
    return p


def _light(s, inputs, wp, wn, shin):
    """Per-point light factors of OpenGL's fixed-function lighting with a local viewer.

    Returns the ambient+diffuse factor ``d1`` and specular factor ``s1`` of the headlight and the lights without
    shadows, and ``d2``, ``s2`` of the shadow light. ``wp`` and ``wn`` are (C, N, 3); ``shin`` broadcasts to (C, N).
    """
    campos = inputs["cam_pos"][:, None]
    view = campos - wp
    view = view / view.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    fwd = -inputs["cam_mat"][:, :, 2]
    amb, dif, spe = (float(x[0]) for x in s.head)
    head_dir = -fwd[:, None]
    ndl = (wn * head_dir).sum(-1)
    d1 = amb + dif * ndl.clamp_min(0)
    s1 = _specular(wn, head_dir, view, ndl, shin) * spe if spe else torch.zeros_like(d1)
    d2 = torch.zeros_like(d1)
    s2 = torch.zeros_like(d1)
    for k, light in enumerate(s.lights):
        if light["directional"]:
            ldir = torch.as_tensor(-light["dir"], dtype=torch.float32, device=wp.device).expand_as(wp)
            att = None
        else:
            rel = torch.as_tensor(light["pos"], dtype=torch.float32, device=wp.device) - wp
            dist = rel.norm(dim=-1, keepdim=True)
            ldir = rel / dist.clamp_min(1e-12)
            a = light["attenuation"]
            att = 1.0 / (float(a[0]) + float(a[1]) * dist[..., 0] + float(a[2]) * dist[..., 0] ** 2)
            if light["cone"] is not None:
                cutoff, exponent = light["cone"]
                sdir = torch.as_tensor(light["dir"], dtype=torch.float32, device=wp.device)
                cos = (-ldir * sdir).sum(-1)
                spot = torch.where(
                    cos >= math.cos(math.radians(cutoff)), cos.clamp_min(0) ** exponent, torch.zeros_like(cos)
                )
                att = att * spot
        ndl = (wn * ldir).sum(-1)
        diffuse = float(light["diffuse"][0])
        if k == s.global_light:
            diffuse = inputs["intensity"].view(-1, *([1] * (ndl.dim() - 1)))
        d = float(light["ambient"][0]) + diffuse * ndl.clamp_min(0)
        sp = _specular(wn, ldir, view, ndl, shin) * float(light["specular"][0])
        if att is not None:
            d, sp = d * att, sp * att
        if light["shadow"]:
            d2, s2 = d2 + d, s2 + sp
        else:
            d1, s1 = d1 + d, s1 + sp
    return d1, s1, d2, s2


def _specular(n, ldir, view, ndl, shin):
    h = ldir + view
    h = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    ndh = (n * h).sum(-1).clamp_min(0)
    return torch.where(ndl > 0, torch.pow(ndh, shin), torch.zeros_like(ndh))


def _view_projection(s, cam, pos):
    """(C, 4, 4) float64 projection times view matrices of camera rotations (C, 3, 3) and positions (C, 3)."""
    cam, pos = cam.double(), pos.double()
    view = torch.zeros(len(cam), 4, 4, dtype=torch.float64, device=cam.device)
    view[:, :3, :3] = cam.transpose(1, 2)
    view[:, :3, 3] = -_matvec(cam.transpose(1, 2), pos)
    view[:, 3, 3] = 1
    return _matmul(torch.as_tensor(_projection(s), device=cam.device), view)


#: NVIDIA's 4x multisample positions in a pixel, x right and y up from its lower left corner.
SAMPLES = ((0.125, 0.625), (0.625, 0.875), (0.375, 0.125), (0.875, 0.375))


def _offset(clip, res, sample):
    """Clip positions moved so that pixel centres sample position ``sample`` of each pixel."""
    sx, sy = SAMPLES[sample]
    shift = torch.tensor([(0.5 - sx) * 2 / res, (0.5 - sy) * 2 / res, 0.0, 0.0], device=clip.device)
    return (clip + shift * clip[..., 3:4]).contiguous()


def _centroid(rast, rast_db, sample):
    """Barycentrics at the point where NVIDIA OpenGL shades a multisampled pixel.

    That is the pixel centre when the centre lies in the triangle, else the sample itself.
    """
    sx, sy = SAMPLES[sample]
    dx, dy = 0.5 - sx, 0.5 - sy
    u = rast[..., 0] + rast_db[..., 0] * dx + rast_db[..., 1] * dy
    v = rast[..., 1] + rast_db[..., 2] * dx + rast_db[..., 3] * dy
    outside = (u < 0) | (v < 0) | (u + v > 1)
    u = torch.where(outside, rast[..., 0], u)
    v = torch.where(outside, rast[..., 1], v)
    return torch.stack([u, v, rast[..., 2], rast[..., 3]], -1).contiguous()


def _rays(s, cam, S, px, py):
    """World ray directions (U, S, S, 3), not normalised, through window positions ``px``, ``py`` (S, S)."""
    tanh = math.tan(math.radians(s.fovy) / 2.0)
    x = (px / S * 2 - 1) * tanh
    y = (py / S * 2 - 1) * tanh
    return (
        x[None, ..., None] * cam[:, None, None, :, 0]
        + y[None, ..., None] * cam[:, None, None, :, 1]
        - cam[:, None, None, :, 2]
    )


def _floor_hits(s, campos, rays):
    """World points where ``rays`` meet the floor plane, and whether they meet it in front within its extent."""
    t = -campos[:, None, None, 2:3] / rays[..., 2:3]
    point = campos[:, None, None, :] + t * rays
    size = torch.as_tensor(s.floor["size"], device=rays.device, dtype=rays.dtype)
    inside = (t[..., 0] > 0) & (point[..., 0].abs() <= size[0]) & (point[..., 1].abs() <= size[1])
    return point, inside


def _background(s, keys, S, device):
    """Floor and sky of each sample for camera, light intensity and floor colour keys, cached by key.

    The floor is lit per vertex on MuJoCo's plane grid, so each sample interpolates the lit colours of the three
    grid corners of its triangle; it is textured with the grid texture's trilinear filter.
    """
    out = [None] * len(keys)
    missing = []
    for i, key in enumerate(keys):
        hit = _SKY.get((s.family, S, key))
        if hit is None:
            missing.append(i)
        else:
            out[i] = hit
    if missing:
        made = _make_background(s, [keys[i] for i in missing], S, device)
        for j, i in enumerate(missing):
            entry = {name: value[j] for name, value in made.items()}
            if len(_SKY) >= 64:
                _SKY.pop(next(iter(_SKY)))
            _SKY[(s.family, S, keys[i])] = entry
            out[i] = entry
    return {name: torch.stack([entry[name] for entry in out]) for name in out[0]}


def _make_background(s, keys, S, device):
    import nvdiffrast.torch as dr

    U = len(keys)
    cam = torch.as_tensor(np.stack([np.frombuffer(k[0], np.float64).reshape(3, 3) for k in keys]), device=device)
    intensity = torch.as_tensor(np.array([np.frombuffer(k[1], np.float32)[0] for k in keys]), device=device)
    floor_colors = torch.as_tensor(
        np.stack([np.frombuffer(k[2], np.float64).reshape(2, 3) for k in keys]), device=device
    )
    campos = torch.as_tensor(np.broadcast_to(s.cam_pos, (U, 3)).copy(), device=device)
    f = s.floor
    i = torch.arange(S, device=device, dtype=torch.float64)
    py, px = torch.meshgrid(i, i, indexing="ij")
    centre, centre_in = _floor_hits(s, campos, _rays(s, cam, S, px + 0.5, py + 0.5))
    pv = _view_projection(s, cam, campos)
    cover, depth = [], []
    for sx, sy in SAMPLES:
        point, inside = _floor_hits(s, campos, _rays(s, cam, S, px + sx, py + sy))
        m = pv[:, None, None]
        clip = (
            m[..., :, 0] * point[..., None, 0]
            + m[..., :, 1] * point[..., None, 1]
            + m[..., :, 2] * point[..., None, 2]
            + m[..., :, 3]
        )
        z = clip[..., 2] / clip[..., 3]
        inside = inside & (z >= -1) & (z <= 1)
        cover.append(inside)
        depth.append(torch.where(inside, z, torch.ones_like(z)))
    # Gouraud shading on the plane grid: the lit colours of the three corners of the sample's triangle.
    origin = torch.as_tensor(f["origin"], device=device)
    step = torch.as_tensor(f["step"], device=device)
    cells = torch.as_tensor(f["cells"], device=device)
    local = (centre[..., :2] - origin) / step
    cell = torch.minimum(torch.maximum(torch.floor(local), torch.zeros_like(local)), (cells - 1).to(local.dtype))
    fx, fy = (local - cell).unbind(-1)
    lower = fx + fy <= 1
    corner = torch.stack(
        [
            torch.where(
                lower[..., None],
                torch.stack([cell[..., 0], cell[..., 1]], -1),
                torch.stack([cell[..., 0] + 1, cell[..., 1]], -1),
            ),
            torch.where(
                lower[..., None],
                torch.stack([cell[..., 0] + 1, cell[..., 1]], -1),
                torch.stack([cell[..., 0] + 1, cell[..., 1] + 1], -1),
            ),
            torch.stack([cell[..., 0], cell[..., 1] + 1], -1),
        ],
        -2,
    )
    weight = torch.stack(
        [torch.where(lower, 1 - fx - fy, 1 - fy), torch.where(lower, fx, fx + fy - 1), torch.where(lower, fy, 1 - fx)],
        -1,
    )
    wp = torch.cat([origin + corner * step, torch.zeros_like(corner[..., :1])], -1).float()
    wn = torch.zeros_like(wp)
    wn[..., 2] = 1
    spec, shin, emis = f["material"]
    inputs = {"cam_pos": campos.float(), "cam_mat": cam.float(), "intensity": intensity.float()}
    d1, s1, d2, s2 = _light(s, inputs, wp.reshape(U, -1, 3), wn.reshape(U, -1, 3), shin)
    rgb = torch.as_tensor(f["rgba"][:3], device=device)
    c1 = (rgb * (d1[..., None] + emis) + spec * s1[..., None]).clamp(0, 1).reshape(U, S, S, 3, 3)
    c2 = (rgb * d2[..., None] + spec * s2[..., None]).clamp(0, 1).reshape(U, S, S, 3, 3)
    c1 = (c1 * weight[..., None].float()).sum(-2)
    c2 = (c2 * weight[..., None].float()).sum(-2)
    # Grid texture at the pixel centre, with derivatives over each 2x2 pixel quad as the GPU takes them.
    size = torch.as_tensor(f["size"], device=device)
    scale = torch.as_tensor(f["scale"], device=device)
    u = (centre[..., 0] + size[0]) / (2 * size[0]) * scale[0]
    v = (1 - (centre[..., 1] + size[1]) / (2 * size[1])) * scale[1]
    uv = torch.stack([u, v], -1)
    quad = uv[:, 0::2, 0::2]
    du_dx = (uv[:, 0::2, 1::2] - quad).repeat_interleave(2, 1).repeat_interleave(2, 2)
    du_dy = (uv[:, 1::2, 0::2] - quad).repeat_interleave(2, 1).repeat_interleave(2, 2)
    uv_da = torch.stack([du_dx[..., 0], du_dy[..., 0], du_dx[..., 1], du_dy[..., 1]], -1)
    weights = dr.texture(
        f["weights"][None],
        uv.float().contiguous(),
        uv_da.float().contiguous(),
        filter_mode="linear-mipmap-linear",
        boundary_mode="wrap",
    )
    q = torch.round(floor_colors * 255) if f["rounding"] == "round" else torch.floor(floor_colors * 255)
    palette = torch.cat([q / 255, torch.as_tensor(f["mark"], device=device, dtype=q.dtype).expand(U, 1, 3)], 1).float()
    tex = sum(weights[..., k : k + 1] * palette[:, None, None, k] for k in range(3))
    first = torch.floor((c1 * tex).clamp(0, 1) * 255 + 0.5)
    second = torch.floor((first + (c2 * tex).clamp(0, 1) * 255).clamp(max=255) + 0.5)
    # Skybox at the pixel centre. MuJoCo turns a world direction 90 degrees about x into cube space.
    ray = _rays(s, cam, S, px + 0.5, py + 0.5)
    ray = ray / ray.norm(dim=-1, keepdim=True)
    cube = torch.stack([ray[..., 0], ray[..., 2], -ray[..., 1]], -1).float().contiguous()
    sky = torch.floor(dr.texture(s.sky[None], cube, filter_mode="linear", boundary_mode="cube") * 255 + 0.5)
    return {
        "first": first.to(torch.uint8),
        "second": second.to(torch.uint8),
        "sky": sky.to(torch.uint8),
        "cover": torch.stack(cover, 1),
        "depth": torch.stack(depth, 1).float(),
        "xy": centre[..., :2].float(),
        "centre_in": centre_in,
    }


def _shadow_window(s, xy, cover):
    """Texel range [i0, i1) x [j0, j1) of MuJoCo's shadow square that holds every floor point and the camera."""
    clip = s.shadow_clip
    light = s.lights[s.shadow_light]["pos"]
    texel = 2 * clip / SHADOW_SIZE
    seen = xy[cover.any(1)]
    lo = (
        torch.minimum(seen.min(0).values.double().cpu(), torch.as_tensor(s.cam_pos[:2]))
        if len(seen)
        else torch.zeros(2)
    )
    hi = (
        torch.maximum(seen.max(0).values.double().cpu(), torch.as_tensor(s.cam_pos[:2]))
        if len(seen)
        else torch.zeros(2)
    )
    lo = np.floor((lo.numpy() - light[:2] + clip) / texel).astype(int) - 2
    hi = np.ceil((hi.numpy() - light[:2] + clip) / texel).astype(int) + 2
    lo = np.clip(lo, 0, SHADOW_SIZE)
    hi = np.clip(hi, 0, SHADOW_SIZE)
    hi = lo + np.maximum(8, -(-(hi - lo) // 8) * 8)
    return lo, hi


def _shadow_map(s, attr, window):
    """Top-down caster heights (C, nj, ni), less MuJoCo's polygon offset, over a window of its shadow square."""
    from . import manipulation_nvdiff_kernels as kernels

    clip = s.shadow_clip
    light = s.lights[s.shadow_light]["pos"]
    origin = (float(light[0]) - clip, float(light[1]) - clip)
    return kernels.shadow_heights(attr, s.tri_caster, window, origin, 2 * clip / SHADOW_SIZE)


def _vertex_stage(s, inputs):
    from . import manipulation_nvdiff_kernels as kernels

    device = inputs["rgba"].device
    pv = _view_projection(s, inputs["cam_mat"], inputs["cam_pos"]).float().reshape(-1, 16)
    camera = torch.cat([pv, inputs["cam_pos"], -inputs["cam_mat"][:, :, 2]], 1).contiguous()
    tr = inputs["transform"]
    xf = torch.cat([tr[..., :3].reshape(*tr.shape[:2], 9), tr[..., 3]], -1).contiguous()
    if getattr(s, "light_table", None) is None or s.light_table.device != device:
        rows = []
        for k, light in enumerate(s.lights):
            point = not light["directional"]
            q = light["pos"] if point else -light["dir"]
            cone = light["cone"]
            rows.append(
                [
                    float(point),
                    float(light["shadow"]),
                    float(k == s.global_light),
                    *q,
                    float(light["ambient"][0]),
                    float(light["diffuse"][0]),
                    float(light["specular"][0]),
                    *light["attenuation"],
                    float(cone is not None),
                    math.cos(math.radians(cone[0])) if cone else 0.0,
                    cone[1] if cone else 0.0,
                    *light["dir"],
                ]
            )
        s.light_table = torch.tensor(rows, dtype=torch.float32, device=device).contiguous()
        s.vgeom32 = s.vgeom.to(torch.int32)
    return kernels.vertices(
        s.vpos,
        s.vnor,
        s.vgeom32,
        xf,
        inputs["rotation"].reshape(len(pv), -1, 9).contiguous(),
        inputs["rgba"].contiguous(),
        s.material,
        camera,
        inputs["intensity"].contiguous(),
        s.light_table,
        [float(x[0]) for x in s.head],
    )


def draw(s, inputs, size, ss, out):
    """Draw one chunk of frames into ``out`` (C, size, size, 3) uint8 on the device."""
    import nvdiffrast.torch as dr

    device = out.device
    ctx = _context(device)
    C = inputs["transform"].shape[0]
    S = size * ss
    from . import manipulation_nvdiff_kernels as kernels

    clip, attr, cT = _vertex_stage(s, inputs)
    # Floor and sky, shared by frames with the same camera, light and floor colours.
    cams = inputs["cam_mat"].double().cpu().numpy()
    intensity = inputs["intensity"].cpu().numpy()
    floors = inputs["floor"].double().cpu().numpy()
    keys = [(cams[c].tobytes(), intensity[c : c + 1].tobytes(), floors[c].tobytes()) for c in range(C)]
    unique = list(dict.fromkeys(keys))
    index = torch.as_tensor([unique.index(k) for k in keys], device=device)
    bg = _background(s, unique, S, device)
    window = _shadow_window(s, bg["xy"], bg["cover"])
    heights = _shadow_map(s, attr, window)
    clip_ = s.shadow_clip
    light = s.lights[s.shadow_light]["pos"]
    origin = (float(light[0]) - clip_, float(light[1]) - clip_)
    texel = 2 * clip_ / SHADOW_SIZE
    background = {name: value.contiguous() for name, value in bg.items()}
    background["cover"] = background["cover"].to(torch.uint8)
    index32 = index.to(torch.int32)
    base = torch.empty((C, 4, S, S, 3), dtype=torch.uint8, device=device)
    zbuf = torch.empty((C, 4, S, S), device=device)
    clip_opaque = clip[:, s.opaque_vertices]
    for k in range(4):
        if len(s.tri_opaque):
            rast, rast_db = dr.rasterize(
                ctx, _offset(clip_opaque, S, k), s.tri_opaque_compact, resolution=[S, S], grad_db=True
            )
        else:
            rast = torch.zeros((C, S, S, 4), device=device)
            rast_db = rast
        kernels.opaque(
            rast,
            rast_db,
            s.tri_opaque,
            attr,
            s.opaque_geom32,
            s.texmod,
            index32,
            background,
            heights,
            window,
            origin,
            texel,
            SHADOW_BIAS,
            base,
            zbuf,
            k,
            SAMPLES[k],
        )
    if len(s.tri_trans):
        _blend(s, inputs, clip, cT, base, zbuf, out, S, size, ss)
    else:
        _resolve(base, out, S, size, ss)
    return out


def _resolve(base, out, S, size, ss):
    pixel = torch.floor((base.float().sum(1) + 2) / 4).flip(1)
    if ss > 1:
        pixel = torch.round(pixel.reshape(len(base), size, ss, size, ss, 3).mean((2, 4)))
    out.copy_(pixel.to(torch.uint8))


def _blend(s, inputs, clip, cT, base, zbuf, out, S, size, ss):
    """Blend the transparent geoms over each sample in MuJoCo's order, then resolve the samples."""
    from . import manipulation_nvdiff_kernels as kernels

    C = len(clip)
    device = clip.device
    # MuJoCo's order: sorted by camera distance less the bounding radius, ties in geom order.
    gi = s.trans_index
    gpos = inputs["position"][:, gi]
    dist = (gpos - inputs["cam_pos"][:, None]).norm(dim=-1) - s.rbound_t[gi]
    order = torch.argsort(dist, dim=1, stable=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(len(gi), device=device).expand(C, -1).contiguous())
    rank_of_geom = torch.zeros((C, s.G), dtype=torch.int32, device=device)
    rank_of_geom[:, gi] = rank.to(torch.int32)
    pixel = kernels.blend(
        clip,
        s.tri_trans,
        cT,
        rank_of_geom,
        s.trans_geom32,
        s.trans_local32,
        inputs["rgba"][..., 3].contiguous(),
        base,
        zbuf,
        LAYERS,
        SAMPLES,
    )
    if ss > 1:
        pixel = torch.round(pixel.float().reshape(C, size, ss, size, ss, 3).mean((2, 4))).to(torch.uint8)
    out.copy_(pixel)


def render(family, state, device, chunk_size=None, validated=False):
    """Frames (B, size, size, 3) uint8 of a Cube/Scene RenderState on a CUDA device."""
    host, device, size = _gl.prepare(family, state, device, chunk_size, validated)
    if device.type != "cuda":
        raise ValueError("Cube/Scene backend 'nvdiff' draws on CUDA devices")
    columns = _gl.columns(host)
    ss = int(columns["renderer", "supersample"][0, 0])
    B = len(columns["renderer", "size"])
    result = torch.empty((B, size, size, 3), dtype=torch.uint8, device=device)
    limit = chunk_size or max(1, (128 * 224 * 224) // (size * size * ss * ss))
    with _LOCK, torch.cuda.device(device):
        s = scene(family, device)
        for start in range(0, B, limit):
            stop = min(B, start + limit)
            part = {key: value[start:stop] for key, value in columns.items()}
            draw(s, prepare(s, part, device), size, ss, result[start:stop])
    return result
