"""Numba CPU rasterizer for Cube and Scene frames, following MuJoCo 3.17's classic OpenGL renderer.

The scene is the private render model of the OpenGL backend, read by ``render.mujoco_scene``: the same geoms,
materials, textures, lights, camera and shadow map. Each frame is drawn with 4 coverage samples per pixel at the
OpenGL 4x sample positions, as the source's multisampled offscreen buffer does. Lighting is per vertex (Gouraud)
with a local viewer; colours and texture coordinates are interpolated at the pixel centre with perspective
correction, and textures are sampled trilinearly from their mipmaps. The shadow light adds its light where the
4096x4096 shadow map (24-bit depth, MuJoCo's polygon offset, nearest texel) sees the fragment. The static opaque
geoms, their static shadows and the skybox are drawn once per configuration; each frame draws the moving opaque
geoms, tests the shadow-map texels of the visible fragments against the moving geoms and blends the translucent
geoms (the arm) after the skybox, sorted by distance to the camera, once front to back and once back to front.
Frames differ from the OpenGL frames by small edge, rounding and filtering differences.
"""

from __future__ import annotations

import copy
import hashlib
import math
import threading

import numba
import numpy as np
import torch
from numba import njit, prange

from ...render import mujoco_scene as ms
from ...render._resources import BoundedCache
from .simulation import render_state as pixels
from .simulation.compatibility import MODEL_VALUES, CompatibleCube, CompatibleScene

_LOCK = threading.RLock()
_RIGS = BoundedCache(64, name="ogbench.numba_rig")
_SHADOWS = BoundedCache(4, name="ogbench.numba_shadow")
_SX = np.array([0.375, 0.875, 0.125, 0.625])
_SY = np.array([0.875, 0.625, 0.375, 0.125])
POW_STEPS = 4096
DEPTH_BITS = float(2**24 - 1)
OFFSET_FACTOR, OFFSET_UNITS = -1.5, -4.0
CELL = 16
# Arm meshes are drawn, and cast shadows, simplified on grids of these many metres (0 keeps them exact).
ARM_CELL = (0.0, 0.0)


def _key(family, columns, index, cell):
    h = hashlib.sha256(f"{family}:{cell}".encode())
    for k in MODEL_VALUES:
        if k != "agent.color":
            h.update(np.ascontiguousarray(columns["appearance", k][index]).tobytes())
    return h.hexdigest()


class _Rig:
    """The private render model of one configuration, its packed scene and its per-row update."""

    def __init__(self, family, columns, index, cell):
        import mujoco

        from .manipulation_opengl import cluster_meshes

        env = (CompatibleCube if family == "Cube" else CompatibleScene)(seed=0)
        for k in MODEL_VALUES:
            env.variation_space._walk(k)._value = np.array(columns["appearance", k][index]).copy()
        env._mjcf_model = env.modify_mjcf_model(env.build_mjcf_model())
        env.compile_model_and_data()
        self.family, self.env = family, env
        m, d = env.model, env.data
        for ids in env._cube_target_geom_ids_list:
            m.geom_rgba[ids, 3] = pixels.TARGET_ALPHA
        if family == "Scene":
            env._cur_button_states = np.zeros(2, np.int64)
            env._apply_button_states()
        mujoco.mj_forward(m, d)
        view_cell, cast_cell = cell
        exact = copy.copy(m) if view_cell or cast_cell else m
        if view_cell:
            cluster_meshes(m, view_cell)
        s = ms.extract(m, d, "front_pixels")
        self.scene = s
        G = len(s.objid)
        self.objid = s.objid
        if any(light["castshadow"] and light["type"] != ms.DIRECTIONAL for light in s.lights):
            raise NotImplementedError("Cube/Scene CPU rasterizer draws shadows of directional lights only")
        shadow = [i for i, light in enumerate(s.lights) if light["castshadow"]]
        if len(shadow) > 1:
            raise NotImplementedError("Cube/Scene CPU rasterizer draws one shadow light")
        self.shadow_light = shadow[0] if shadow else -1
        self.cpos, self.cnor, self.ctex, self.tris, self.cstarts, self.starts, self.rbound = _pack(s)
        if cast_cell == view_cell:
            self.cast = (self.cpos, self.tris, self.cstarts, self.starts, self.rbound)
        else:
            if cast_cell:
                cluster_meshes(exact, cast_cell)
            c = _pack(ms.extract(exact, d, "front_pixels"))
            self.cast = (c[0], c[3], c[4], c[5], c[6])
        shins, row = np.unique(s.shininess * 128.0, return_inverse=True)
        grid = np.linspace(0.0, 1.0, POW_STEPS + 1)
        self.powlut = np.ascontiguousarray(grid[None, :] ** np.minimum(shins, 128.0)[:, None], np.float64)
        self.material = np.stack(
            [s.specular, s.shininess * 128.0, s.emission, row.astype(np.float64), s.texmode, s.texslot], 1
        ).astype(np.float64)
        self.texdata, self.texlevels, self.texinfo = _pack_textures(s.textures)
        self.static = s.static.copy()
        self.transparent = s.transparent.copy()
        self.opaque_static = np.flatnonzero(s.static & ~s.transparent).astype(np.int64)
        self.opaque_movers = np.flatnonzero(~s.static & ~s.transparent).astype(np.int64)
        self.glass = np.flatnonzero(s.transparent).astype(np.int64)
        self.casters = np.flatnonzero(~s.static).astype(np.int64)
        self.sort_rbound = s.rbound.astype(np.float32)
        cam = s.camera
        if cam["orthographic"] or cam["frustum_center"] != 0.0 or cam["frustum_width"] != 0.0:
            raise NotImplementedError("Cube/Scene CPU rasterizer draws symmetric perspective cameras only")
        f = cam["forward"]
        right = np.cross(f, cam["up"])
        right /= np.linalg.norm(right)
        up = np.cross(right, f)
        up /= np.linalg.norm(up)
        self.cam = np.concatenate([cam["pos"], right, up, f]).astype(np.float64)
        self.tanh = float(cam["frustum_top"]) / float(cam["frustum_near"])
        self.near = float(cam["frustum_near"])
        self.headpos = s.headpos.astype(np.float32)
        self.lights = _lights(s, self.shadow_light)
        self.ambient = float(s.global_ambient)
        self.nshadow = int(s.shadow_size)
        if self.shadow_light >= 0:
            light = s.lights[self.shadow_light]
            self.smat = ms.shadow_matrix(s, light)
            self.slook = ms.shadow_lookup(s, light)
        else:
            self.smat = np.zeros((3, 4))
            self.slook = np.zeros((3, 4))
        self.sky_tex = _pack_textures([s.sky] if s.sky is not None else [])
        self.has_sky = s.sky is not None
        self.plates = {}
        self.looks = {}
        self._default_rgba = s.rgba.astype(np.float64)
        self.render_kinematics = family == "Scene"
        # Static geom poses from the configuration's own kinematics.
        self.static_pos = s.geom_pos.copy()
        self.static_mat = s.geom_mat.copy()

    def shadow_map(self):
        """The depth map of the static casters, shared by configurations with the same static geometry."""
        if self.shadow_light < 0:
            return np.zeros((1, 1), np.float32)
        geoms = np.flatnonzero(self.static)
        h = hashlib.sha256(self.smat.tobytes())
        h.update(np.int64(self.nshadow).tobytes())
        for g in geoms:
            h.update(self.cpos[self.cstarts[g] : self.cstarts[g + 1]].tobytes())
            h.update(self.tris[self.starts[g] : self.starts[g + 1]].tobytes())
            h.update(self.static_pos[g].tobytes() + self.static_mat[g].tobytes())
        key = h.hexdigest()
        hit = _SHADOWS.get(key)
        if hit is None:
            smap = np.zeros((self.nshadow, self.nshadow), np.float32)
            _shadow_dense(
                self.cpos, self.tris, self.cstarts, self.starts, geoms, self.static_pos, self.static_mat, self.smat, smap
            )
            hit = _SHADOWS.put(key, smap)
        return hit

    def rgba(self, look):
        """Drawn geom colours for one (cube, agent, lock, buttons) look, as the OpenGL path writes them."""
        hit = self.looks.get(look)
        if hit is not None:
            return hit
        env, m = self.env, self.env.model
        cube, agent, lock, buttons = look
        env.variation_space._walk("cube.color")._value = np.frombuffer(cube, np.float32).reshape(1, 3).astype(np.float64)
        for material in pixels.AGENT_MATERIALS:
            m.material(material).rgba[:3] = np.frombuffer(agent, np.float32)
        env._cube_colors_from_variation()
        for ids in env._cube_target_geom_ids_list:
            m.geom_rgba[ids, 3] = pixels.TARGET_ALPHA
        if self.family == "Scene":
            from .manipulation_opengl import _apply_render_button_colors

            locks = np.frombuffer(lock, np.float32).reshape(2, 3).astype(np.float64)
            env.variation_space._walk("lock_color")._value = locks.copy()
            env._cur_button_states = np.frombuffer(buttons, np.int64).copy()
            _apply_render_button_colors(env, locks)
        out = np.array([ms._rgba(m, int(g)) for g in self.objid], np.float64)
        fixed = self.opaque_static
        if not np.array_equal(out[fixed], self._default_rgba[fixed]):
            raise ValueError("Cube/Scene appearance colours a static geom, which the cached plate holds")
        if len(self.looks) > 256:
            self.looks.clear()
        self.looks[look] = out
        return out

    def poses(self, columns, indices):
        """Geom positions, rotations and colours of rows ``indices`` of host RenderState ``columns``."""
        import mujoco

        m, d = self.env.model, self.env.data
        qpos, mpos, mquat = (columns["pose", k] for k in ("qpos", "mocap_pos", "mocap_quat"))
        cube = np.ascontiguousarray(columns["appearance", "cube.color"], np.float32)
        agent = np.ascontiguousarray(columns["appearance", "agent.color"], np.float32)
        scene = self.family == "Scene"
        if scene:
            locks = np.ascontiguousarray(columns["appearance", "lock_color"], np.float32)
            buttons = np.ascontiguousarray(columns["pose", "buttons"]).astype(np.int64)
        n, G = len(indices), len(self.objid)
        gpos = np.empty((n, G, 3))
        gmat = np.empty((n, G, 9))
        rgba = np.empty((n, G, 4))
        objid = self.objid
        for k, i in enumerate(indices):
            d.qpos[:] = qpos[i]
            d.mocap_pos[:] = mpos[i]
            d.mocap_quat[:] = mquat[i]
            mujoco.mj_kinematics(m, d)
            gpos[k] = d.geom_xpos[objid]
            gmat[k] = d.geom_xmat[objid]
            look = (
                cube[i].tobytes(),
                agent[i].tobytes(),
                locks[i].tobytes() if scene else b"",
                buttons[i].tobytes() if scene else b"",
            )
            rgba[k] = self.rgba(look)
        return gpos, gmat, rgba

    def plate(self, res):
        """Samples of the static opaque geoms with their static shadows, and the skybox colour of each pixel."""
        hit = self.plates.get(res)
        if hit is not None:
            return hit
        smap = self.shadow_map()
        n = res * res * 4
        buf = (
            np.zeros(n, np.float32),
            np.zeros(n * 3, np.uint8),
            np.zeros(n * 3, np.uint8),
            np.zeros(n, np.uint8),
            np.zeros(n, np.int32),
            np.zeros(n, np.float32),
            np.zeros(n, np.uint8),
        )
        _plate(
            self._geo(),
            self._tex(),
            self.opaque_static,
            self.static_pos,
            self.static_mat,
            self._default_rgba,
            self.cam,
            self.lights,
            self.ambient,
            self.tanh,
            self.near,
            res,
            self.slook,
            smap,
            buf,
        )
        depth, col, add, kind, key, rr, _ = buf
        lit = col.copy()
        on = np.repeat(kind == 1, 3)
        lit[on] = np.minimum(col[on].astype(np.int32) + add[on], 255).astype(np.uint8)
        sky = np.zeros((res, res, 3), np.uint8)
        if self.has_sky:
            _sky_plate(self.sky_tex[:2], self.sky_tex[2], self.cam, self.tanh, sky)
        hit = (depth, col, lit, kind, key, rr, sky)
        self.plates[res] = hit
        return hit

    def _geo(self):
        return (
            self.cpos,
            self.cnor,
            self.ctex,
            self.tris,
            self.cstarts,
            self.starts,
            self.rbound,
            self.material,
            self.powlut,
        )

    def _tex(self):
        return (self.texdata, self.texlevels, self.texinfo)


def _pack(s):
    """Corners and triangles of every drawn geom; corners that share a position, normal and texture coordinate
    are transformed and lit once per frame."""
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


def _pack_textures(textures):
    """Mip levels of every texture face in one float array; ``levels`` rows are (offset, width, height)."""
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


def _lights(s, shadow):
    """Rows: type, role (0 first pass, 1 shadow light), direction or position, ambient, diffuse, specular,
    attenuation, spot direction, spot cutoff cosine (-2 for none) and exponent."""
    out = np.zeros((len(s.lights), 22))
    for i, light in enumerate(s.lights):
        out[i, 0] = light["type"]
        out[i, 1] = 1.0 if i == shadow else 0.0
        if light["type"] == ms.DIRECTIONAL:
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
    return out


def _rig(family, columns, index, cell):
    key = _key(family, columns, index, cell)
    rig = _RIGS.get(key)
    if rig is None:
        rig = _RIGS.put(key, _Rig(family, columns, index, cell))
    return rig


def _groups(family, columns, cell):
    groups = {}
    b = len(columns["renderer", "size"])
    for i in range(b):
        groups.setdefault(_key(family, columns, i, cell), []).append(i)
    return groups


def draw(family, columns, images, cell=None):
    """Draw validated host RenderState ``columns`` into ``(B, size, size, 3)`` uint8 ``images`` in place."""
    from ...render import cpu_raster

    cpu_raster.available()
    cell = ARM_CELL if cell is None else cell
    size, ss = (int(columns["renderer", k][0, 0]) for k in ("size", "supersample"))
    res = size * ss
    with _LOCK:
        for _, indices in _groups(family, columns, cell).items():
            rig = _rig(family, columns, indices[0], cell)
            gpos, gmat, rgba = rig.poses(columns, indices)
            depth, col, add, kind, key, rr, sky = rig.plate(res)
            smap = rig.shadow_map()
            out = images[np.asarray(indices)] if len(indices) != len(images) else images
            threads = numba.get_num_threads()
            bands = max(1, min(size, -(-threads // len(indices)))) if len(indices) < threads else 1
            _render(
                rig._geo(),
                rig._tex(),
                (rig.opaque_movers, rig.glass, rig.casters),
                gpos,
                gmat,
                rgba,
                rig.cam,
                rig.lights,
                rig.ambient,
                rig.tanh,
                rig.near,
                rig.headpos,
                rig.sort_rbound,
                rig.smat,
                rig.slook,
                smap,
                (depth, col, add, kind, key, rr, sky),
                rig.cast,
                ss,
                bands,
                out,
            )
            if out is not images:
                images[np.asarray(indices)] = out


def frames(family, columns, device):
    """Frames of host RenderState columns on ``device``."""
    from .manipulation_opengl import check_versions

    check_versions("numba")
    b, size = len(columns["renderer", "size"]), int(columns["renderer", "size"][0, 0])
    images = np.empty((b, size, size, 3), np.uint8)
    draw(family, columns, images)
    result = torch.from_numpy(images)
    return result if torch.device(device).type == "cpu" else result.to(device)


# ----------------------------------------------------------------------------------------------- kernels


@njit(cache=True, inline="always", error_model="numpy", fastmath=True)
def _shade(wx, wy, wz, nx, ny, nz, cam, r, g, bb, spec, lut, emis, ambient, lights, which, out, o):
    """Fixed-function lighting of one vertex; ``which`` 0: first-pass lights and emission, 1: the shadow light,
    2: every light and emission. Writes the clamped RGB into ``out[o, :3]``."""
    vx, vy, vz = cam[0] - wx, cam[1] - wy, cam[2] - wz
    vn = math.sqrt(vx * vx + vy * vy + vz * vz)
    if vn > 0:
        vx, vy, vz = vx / vn, vy / vn, vz / vn
    if which == 1:
        o0 = ambient * r
        o1 = ambient * g
        o2 = ambient * bb
    else:
        o0 = (emis + ambient) * r
        o1 = (emis + ambient) * g
        o2 = (emis + ambient) * bb
    for k in range(lights.shape[0]):
        role = lights[k, 1]
        if (which == 0 and role != 0.0) or (which == 1 and role != 1.0):
            continue
        if lights[k, 0] == 1.0:
            lx, ly, lz = lights[k, 2], lights[k, 3], lights[k, 4]
            att = 1.0
        else:
            lx, ly, lz = lights[k, 2] - wx, lights[k, 3] - wy, lights[k, 4] - wz
            dist = math.sqrt(lx * lx + ly * ly + lz * lz)
            if dist > 0:
                lx, ly, lz = lx / dist, ly / dist, lz / dist
            att = 1.0 / (lights[k, 14] + lights[k, 15] * dist + lights[k, 16] * dist * dist)
            if lights[k, 20] > -1.5:
                sd = -(lx * lights[k, 17] + ly * lights[k, 18] + lz * lights[k, 19])
                if sd < lights[k, 20]:
                    att = 0.0
                else:
                    att *= max(sd, 0.0) ** lights[k, 21]
        if att == 0.0:
            continue
        t0 = lights[k, 5] * r
        t1 = lights[k, 6] * g
        t2 = lights[k, 7] * bb
        ndl = nx * lx + ny * ly + nz * lz
        if ndl > 0:
            t0 += ndl * lights[k, 8] * r
            t1 += ndl * lights[k, 9] * g
            t2 += ndl * lights[k, 10] * bb
            if spec > 0:
                hx, hy, hz = lx + vx, ly + vy, lz + vz
                hn = math.sqrt(hx * hx + hy * hy + hz * hz)
                if hn > 0:
                    ndh = (nx * hx + ny * hy + nz * hz) / hn
                    if ndh > 0:
                        t = min(ndh, 1.0) * (lut.shape[0] - 1)
                        i = min(int(t), lut.shape[0] - 2)
                        f = (lut[i] + (t - i) * (lut[i + 1] - lut[i])) * spec
                        t0 += f * lights[k, 11]
                        t1 += f * lights[k, 12]
                        t2 += f * lights[k, 13]
        o0 += att * t0
        o1 += att * t1
        o2 += att * t2
    out[o, 0] = min(max(o0, 0.0), 1.0)
    out[o, 1] = min(max(o1, 0.0), 1.0)
    out[o, 2] = min(max(o2, 0.0), 1.0)


@njit(cache=True, inline="always", error_model="numpy", fastmath=True)
def _light_corner(k, ctx, cc, ca, cl, done):
    """Light corner ``k`` of the geom in ``ctx``: first-pass colour into ``cc``, shadow light into ``ca`` and the
    shadow texture coordinates into ``cl`` for an opaque geom; every light into ``cc`` for a translucent one."""
    ex, ey, ew, cnor, gm, cam, r, g, bb, a, spec, lut, emis, ambient, lights, blend, slook = ctx
    nlx, nly, nlz = cnor[k, 0], cnor[k, 1], cnor[k, 2]
    nx = gm[0] * nlx + gm[1] * nly + gm[2] * nlz
    ny = gm[3] * nlx + gm[4] * nly + gm[5] * nlz
    nz = gm[6] * nlx + gm[7] * nly + gm[8] * nlz
    nn = math.sqrt(nx * nx + ny * ny + nz * nz)
    if nn > 0:
        nx, ny, nz = nx / nn, ny / nn, nz / nn
    wx = cam[0] + ex[k] * cam[3] + ey[k] * cam[6] + ew[k] * cam[9]
    wy = cam[1] + ex[k] * cam[4] + ey[k] * cam[7] + ew[k] * cam[10]
    wz = cam[2] + ex[k] * cam[5] + ey[k] * cam[8] + ew[k] * cam[11]
    if blend:
        _shade(wx, wy, wz, nx, ny, nz, cam, r, g, bb, spec, lut, emis, ambient, lights, 2, cc, k)
    else:
        _shade(wx, wy, wz, nx, ny, nz, cam, r, g, bb, spec, lut, emis, ambient, lights, 0, cc, k)
        _shade(wx, wy, wz, nx, ny, nz, cam, r, g, bb, spec, lut, emis, ambient, lights, 1, ca, k)
        for j in range(3):
            cl[k, j] = slook[j, 0] * wx + slook[j, 1] * wy + slook[j, 2] * wz + slook[j, 3]
    cc[k, 3] = min(max(a, 0.0), 1.0)
    done[k] = True


@njit(cache=True, inline="always", error_model="numpy", fastmath=True)
def _bilinear(data, off, w, h, u, v, wrap, out):
    """Bilinear lookup at texel coordinates ``(u, v)`` (texel centres at +0.5); repeat or clamp to edge."""
    x = u - 0.5
    y = v - 0.5
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    fx = x - x0
    fy = y - y0
    if wrap:
        xa, xb = x0 % w, (x0 + 1) % w
        ya, yb = y0 % h, (y0 + 1) % h
    else:
        xa, xb = min(max(x0, 0), w - 1), min(max(x0 + 1, 0), w - 1)
        ya, yb = min(max(y0, 0), h - 1), min(max(y0 + 1, 0), h - 1)
    for ch in range(3):
        top = data[off + (ya * w + xa) * 3 + ch] * (1 - fx) + data[off + (ya * w + xb) * 3 + ch] * fx
        bot = data[off + (yb * w + xa) * 3 + ch] * (1 - fx) + data[off + (yb * w + xb) * 3 + ch] * fx
        out[ch] = top * (1 - fy) + bot * fy


@njit(cache=True, inline="always", error_model="numpy", fastmath=True)
def _trilinear(data, levels, first, nlev, u, v, rho, wrap, out, tmp):
    """Trilinear (``LINEAR_MIPMAP_LINEAR``) lookup at level-0 normalized ``(u, v)``; ``rho`` is the level-0 texel
    footprint of one pixel."""
    w0, h0 = levels[first, 1], levels[first, 2]
    lam = math.log2(rho) if rho > 0 else -1.0
    if lam <= 0.0:
        _bilinear(data, levels[first, 0], w0, h0, u * w0, v * h0, wrap, out)
        return
    lam = min(lam, nlev - 1.0)
    l0 = int(math.floor(lam))
    f = lam - l0
    l1 = min(l0 + 1, nlev - 1)
    r0 = first + l0
    r1 = first + l1
    _bilinear(data, levels[r0, 0], levels[r0, 1], levels[r0, 2], u * levels[r0, 1], v * levels[r0, 2], wrap, out)
    if f > 0:
        _bilinear(data, levels[r1, 0], levels[r1, 1], levels[r1, 2], u * levels[r1, 1], v * levels[r1, 2], wrap, tmp)
        for ch in range(3):
            out[ch] = out[ch] * (1 - f) + tmp[ch] * f


@njit(cache=True, inline="always", error_model="numpy", fastmath=True)
def _cube_face(dx, dy, dz):
    """GL cube map face and normalized face coordinates of a direction."""
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
    if ma <= 0:
        return 0, 0.5, 0.5
    return f, (sc / ma + 1.0) * 0.5, (tc / ma + 1.0) * 0.5


@njit(cache=True, inline="always", error_model="numpy", fastmath=True)
def _cube_coords(f, dx, dy, dz):
    """Normalized coordinates of a direction on cube face ``f`` (for derivatives)."""
    if f == 0:
        sc, tc, ma = -dz, -dy, dx
    elif f == 1:
        sc, tc, ma = dz, -dy, -dx
    elif f == 2:
        sc, tc, ma = dx, dz, dy
    elif f == 3:
        sc, tc, ma = dx, -dz, -dy
    elif f == 4:
        sc, tc, ma = dx, -dy, dz
    else:
        sc, tc, ma = -dx, -dy, -dz
    if ma == 0:
        ma = 1e-30
    return (sc / ma + 1.0) * 0.5, (tc / ma + 1.0) * 0.5


@njit(cache=True, inline="always", error_model="numpy", fastmath=True)
def _sample(tex, slot, mode, t0, t1, t2, dx0, dx1, dx2, dy0, dy1, dy2, out, tmp):
    """Texture colour of texture coordinate ``t`` whose neighbours one pixel right and down are ``dx`` and ``dy``."""
    data, levels, info = tex
    first, nfaces, nlev = info[slot, 0], info[slot, 1], info[slot, 2]
    if mode == 1:
        w0, h0 = levels[first, 1], levels[first, 2]
        ax, ay = (dx0 - t0) * w0, (dx1 - t1) * h0
        bx, by = (dy0 - t0) * w0, (dy1 - t1) * h0
        rho = max(math.sqrt(ax * ax + ay * ay), math.sqrt(bx * bx + by * by))
        _trilinear(data, levels, first, nlev, t0, t1, rho, True, out, tmp)
        return
    f, u, v = _cube_face(t0, t1, t2)
    ux, vx = _cube_coords(f, dx0, dx1, dx2)
    uy, vy = _cube_coords(f, dy0, dy1, dy2)
    n = levels[first + f * nlev, 1]
    rho = max(math.sqrt((ux - u) ** 2 + (vx - v) ** 2), math.sqrt((uy - u) ** 2 + (vy - v) ** 2)) * n
    _trilinear(data, levels, first + f * nlev, nlev, u, v, rho, False, out, tmp)


@njit(cache=True, inline="always", error_model="numpy", fastmath=True)
def _q8(x):
    return np.uint8(math.floor(min(max(x, 0.0), 1.0) * 255.0 + 0.5))


@njit(cache=True, inline="always", error_model="numpy", fastmath=True)
def _tri(
    x0, y0, q0, x1, y1, q1, x2, y2, q2, k0, k1, k2, ctx, cc, ca, cl, done, tc, texmode, texslot, tex, smap,
    res, y0b, y1b, buf, blend, tmp,
):
    """Fill one screen triangle with vertices ``(x, y, 1/w)`` in pixel rows ``[y0b, y1b)`` of the sample buffers.

    Opaque triangles write depth, first-pass colour, shadow-light colour and the shadow-map texel and depth of each
    covered sample; translucent ones blend into the colour of the samples in front of the stored depth.
    """
    depth, col, add, kind, key, rr, mv = buf
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
    inv = 1.0 / area
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
    nmap = smap.shape[0]
    narrow = xmax - xmin < 4
    for py in range(ymin, ymax + 1):
        pxa = xmin
        pxb = xmax
        if not narrow:
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
                if zc + dz0 >= depth[base + 0]:
                    mask |= 1
            if inside or (e0 + o01 >= 0 and e1 + o11 >= 0 and e2 + o21 >= 0):
                if zc + dz1 >= depth[base + 1]:
                    mask |= 2
            if inside or (e0 + o02 >= 0 and e1 + o12 >= 0 and e2 + o22 >= 0):
                if zc + dz2 >= depth[base + 2]:
                    mask |= 4
            if inside or (e0 + o03 >= 0 and e1 + o13 >= 0 and e2 + o23 >= 0):
                if zc + dz3 >= depth[base + 3]:
                    mask |= 8
            if mask == 0:
                continue
            if not lit:
                for k in (k0, k1, k2):
                    if not done[k]:
                        _light_corner(k, ctx, cc, ca, cl, done)
                lit = True
            w0 = e0 * q0
            w1 = e1 * q1
            w2 = e2 * q2
            ws = 1.0 / (w0 + w1 + w2)
            w0 *= ws
            w1 *= ws
            w2 *= ws
            r = min(max(w0 * cc[k0, 0] + w1 * cc[k1, 0] + w2 * cc[k2, 0], 0.0), 1.0)
            g = min(max(w0 * cc[k0, 1] + w1 * cc[k1, 1] + w2 * cc[k2, 1], 0.0), 1.0)
            bl = min(max(w0 * cc[k0, 2] + w1 * cc[k1, 2] + w2 * cc[k2, 2], 0.0), 1.0)
            al = min(max(w0 * cc[k0, 3] + w1 * cc[k1, 3] + w2 * cc[k2, 3], 0.0), 1.0)
            tr = tg = tb = 1.0
            if texmode != 0:
                # Texture coordinates at the pixel centre and one pixel right and down, for the mipmap level.
                vx0 = (e0 + a0) * q0
                vx1 = (e1 + a1) * q1
                vx2 = (e2 + a2) * q2
                sx_ = 1.0 / (vx0 + vx1 + vx2)
                vy0 = (e0 + b0) * q0
                vy1 = (e1 + b1) * q1
                vy2 = (e2 + b2) * q2
                sy_ = 1.0 / (vy0 + vy1 + vy2)
                t = np.empty(9)
                for j in range(3):
                    t[j] = w0 * tc[k0, j] + w1 * tc[k1, j] + w2 * tc[k2, j]
                    t[3 + j] = (vx0 * tc[k0, j] + vx1 * tc[k1, j] + vx2 * tc[k2, j]) * sx_
                    t[6 + j] = (vy0 * tc[k0, j] + vy1 * tc[k1, j] + vy2 * tc[k2, j]) * sy_
                _sample(tex, texslot, texmode, t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], tmp, tmp[3:])
                tr, tg, tb = tmp[0], tmp[1], tmp[2]
            if blend:
                u = 1.0 - al
                rs = r * tr * al * 255.0
                gs = g * tg * al * 255.0
                bs = bl * tb * al * 255.0
                for k in range(4):
                    if mask & (1 << k):
                        sk = (base + k) * 3
                        col[sk] = np.uint8(math.floor(rs + col[sk] * u + 0.5))
                        col[sk + 1] = np.uint8(math.floor(gs + col[sk + 1] * u + 0.5))
                        col[sk + 2] = np.uint8(math.floor(bs + col[sk + 2] * u + 0.5))
                continue
            r8 = _q8(r * tr)
            g8 = _q8(g * tg)
            b8 = _q8(bl * tb)
            ar = min(max(w0 * ca[k0, 0] + w1 * ca[k1, 0] + w2 * ca[k2, 0], 0.0), 1.0) * tr
            ag = min(max(w0 * ca[k0, 1] + w1 * ca[k1, 1] + w2 * ca[k2, 1], 0.0), 1.0) * tg
            ab = min(max(w0 * ca[k0, 2] + w1 * ca[k1, 2] + w2 * ca[k2, 2], 0.0), 1.0) * tb
            ar8 = _q8(ar)
            ag8 = _q8(ag)
            ab8 = _q8(ab)
            kd = np.uint8(0)
            kk = np.int32(0)
            rv = np.float32(0.0)
            if (ar8 | ag8 | ab8) != 0 and al >= 0.99 and nmap > 1:
                s_ = w0 * cl[k0, 0] + w1 * cl[k1, 0] + w2 * cl[k2, 0]
                t_ = w0 * cl[k0, 1] + w1 * cl[k1, 1] + w2 * cl[k2, 1]
                rz = min(max(w0 * cl[k0, 2] + w1 * cl[k1, 2] + w2 * cl[k2, 2], 0.0), 1.0)
                si = min(max(int(math.floor(s_)), 0), nmap - 1)
                ti = min(max(int(math.floor(t_)), 0), nmap - 1)
                if rz >= smap[ti, si]:
                    kd = np.uint8(1)
                    kk = np.int32(ti * nmap + si)
                    rv = np.float32(rz)
            for k in range(4):
                if mask & (1 << k):
                    sk = base + k
                    depth[sk] = zc + (dz0 if k == 0 else (dz1 if k == 1 else (dz2 if k == 2 else dz3)))
                    col[sk * 3] = r8
                    col[sk * 3 + 1] = g8
                    col[sk * 3 + 2] = b8
                    add[sk * 3] = ar8
                    add[sk * 3 + 1] = ag8
                    add[sk * 3 + 2] = ab8
                    kind[sk] = kd
                    key[sk] = kk
                    rr[sk] = rv
                    mv[sk] = 1


@njit(cache=True, error_model="numpy", fastmath=True)
def _geom(gi, geo, tex, gp, gm, col4, cam, lights, ambient, tanh, near, slook, smap, res, y0b, y1b, buf, blend, work):
    """Draw geom ``gi``; its camera-space corners and lit colours stay in ``work`` for a second draw in the frame."""
    cpos, cnor, ctex, tris, cstarts, starts, rbound, material, powlut = geo
    ex, ey, ew, sx, sy, cc, ca, cl, done, xf, tmp = work
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
        top = max((ye + rb) / (w - rb), (ye + rb) / (w + rb))
        bottom = min((ye - rb) / (w - rb), (ye - rb) / (w + rb))
        if half - top * scale >= y1b or half - bottom * scale < y0b:
            return
    spec, emis = material[gi, 0], material[gi, 2]
    lut = powlut[int(material[gi, 3])]
    texmode = int(material[gi, 4])
    texslot = int(material[gi, 5])
    r, g, bb, a = col4[0], col4[1], col4[2], col4[3]
    c0 = cstarts[gi]
    c1 = cstarts[gi + 1]
    ctx = (ex, ey, ew, cnor, gm, cam, r, g, bb, a, spec, lut, emis, ambient, lights, blend, slook)
    if not xf[gi]:
        xf[gi] = True
        for j in range(c0, c1):
            lx, ly, lz = cpos[j, 0], cpos[j, 1], cpos[j, 2]
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
                ih = scale / h
                sx[j] = half + e * ih
                sy[j] = half - f * ih
            done[j] = False
    for t in range(starts[gi], starts[gi + 1]):
        k0 = tris[t, 0]
        k1 = tris[t, 1]
        k2 = tris[t, 2]
        # Triangles that reach the near plane are not drawn; no geom of these scenes comes that close.
        if ew[k0] < near or ew[k1] < near or ew[k2] < near:
            continue
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
        if math.floor(fx1 * 4.0 - 0.5) < math.ceil(fx0 * 4.0 - 0.5) or math.floor(fy1 * 4.0 - 0.5) < math.ceil(
            fy0 * 4.0 - 0.5
        ):
            continue
        _tri(
            x0, y0, 1.0 / ew[k0], x1, y1, 1.0 / ew[k1], x2, y2, 1.0 / ew[k2], k0, k1, k2, ctx, cc, ca, cl, done, ctex,
            texmode, texslot, tex, smap, res, y0b, y1b, buf, blend, tmp,
        )


@njit(cache=True, error_model="numpy", fastmath=True)
def _work_nb(ncorner, ngeom):
    return (
        np.empty(ncorner),
        np.empty(ncorner),
        np.empty(ncorner),
        np.empty(ncorner),
        np.empty(ncorner),
        np.empty((ncorner, 4)),
        np.empty((ncorner, 3)),
        np.empty((ncorner, 3)),
        np.empty(ncorner, np.bool_),
        np.zeros(ngeom, np.bool_),
        np.empty(6),
    )


@njit(cache=True, error_model="numpy", fastmath=True)
def _plate(geo, tex, geoms, gpos, gmat, rgba, cam, lights, ambient, tanh, near, res, slook, smap, buf):
    """Draw the static opaque ``geoms`` over the whole frame into the sample buffers."""
    cstarts = geo[4]
    work = _work_nb(cstarts[-1], cstarts.shape[0] - 1)
    for gi in geoms:
        _geom(gi, geo, tex, gpos[gi], gmat[gi], rgba[gi], cam, lights, ambient, tanh, near, slook, smap, res, 0, res, buf, False, work)


@njit(cache=True, inline="always", error_model="numpy", fastmath=True)
def _quantize(z):
    return math.floor(min(max(z, 0.0), 1.0) * DEPTH_BITS + 0.5) / DEPTH_BITS


@njit(cache=True, error_model="numpy", fastmath=True)
def _shadow_dense(cpos, tris, cstarts, starts, geoms, gpos, gmat, smat, smap):
    """Depth (with MuJoCo's polygon offset, 24-bit) of every texel of ``smap`` that ``geoms`` cover from the light."""
    n = smap.shape[0]
    maxc = 1
    for gi in geoms:
        maxc = max(maxc, cstarts[gi + 1] - cstarts[gi])
    lx = np.empty(maxc)
    ly = np.empty(maxc)
    lz = np.empty(maxc)
    for gi in geoms:
        c0 = cstarts[gi]
        nc = cstarts[gi + 1] - c0
        gm = gmat[gi]
        gp = gpos[gi]
        for j in range(nc):
            px, py, pz = cpos[c0 + j, 0], cpos[c0 + j, 1], cpos[c0 + j, 2]
            wx = gm[0] * px + gm[1] * py + gm[2] * pz + gp[0]
            wy = gm[3] * px + gm[4] * py + gm[5] * pz + gp[1]
            wz = gm[6] * px + gm[7] * py + gm[8] * pz + gp[2]
            lx[j] = smat[0, 0] * wx + smat[0, 1] * wy + smat[0, 2] * wz + smat[0, 3]
            ly[j] = smat[1, 0] * wx + smat[1, 1] * wy + smat[1, 2] * wz + smat[1, 3]
            lz[j] = smat[2, 0] * wx + smat[2, 1] * wy + smat[2, 2] * wz + smat[2, 3]
        for t in range(starts[gi], starts[gi + 1]):
            i0, i1, i2 = tris[t, 0] - c0, tris[t, 1] - c0, tris[t, 2] - c0
            x0, y0, z0 = lx[i0], ly[i0], lz[i0]
            x1, y1, z1 = lx[i1], ly[i1], lz[i1]
            x2, y2, z2 = lx[i2], ly[i2], lz[i2]
            det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
            if det == 0.0:
                continue
            dzdx = ((z1 - z0) * (y2 - y0) - (z2 - z0) * (y1 - y0)) / det
            dzdy = ((x1 - x0) * (z2 - z0) - (x2 - x0) * (z1 - z0)) / det
            off = OFFSET_FACTOR * max(abs(dzdx), abs(dzdy)) + OFFSET_UNITS / DEPTH_BITS
            xa = max(int(math.ceil(min(x0, min(x1, x2)) - 0.5)), 1)
            xb = min(int(math.floor(max(x0, max(x1, x2)) - 0.5)), n - 2)
            ya = max(int(math.ceil(min(y0, min(y1, y2)) - 0.5)), 1)
            yb = min(int(math.floor(max(y0, max(y1, y2)) - 0.5)), n - 2)
            sgn = 1.0 if det > 0 else -1.0
            for ty in range(ya, yb + 1):
                cy = ty + 0.5
                for tx in range(xa, xb + 1):
                    cx = tx + 0.5
                    e0 = ((x1 - cx) * (y2 - cy) - (x2 - cx) * (y1 - cy)) * sgn
                    e1 = ((x2 - cx) * (y0 - cy) - (x0 - cx) * (y2 - cy)) * sgn
                    e2 = ((x0 - cx) * (y1 - cy) - (x1 - cx) * (y0 - cy)) * sgn
                    if e0 < 0 or e1 < 0 or e2 < 0:
                        continue
                    z = _quantize(z0 + dzdx * (cx - x0) + dzdy * (cy - y0) + off)
                    if z > smap[ty, tx]:
                        smap[ty, tx] = z


@njit(parallel=True, cache=True, error_model="numpy", fastmath=True)
def _sky_plate(levels_data, info, cam, tanh, out):
    """Skybox colour at each pixel centre: the cube map along the view ray (MuJoCo maps x, y, z to s, -r, t)."""
    data, levels = levels_data
    res = out.shape[0]
    half = 0.5 * res
    first, nlev = info[0, 0], info[0, 2]
    for py in prange(res):
        c = np.empty(3)
        tmp = np.empty(3)
        for px in range(res):
            dirs = np.empty((3, 3))
            for k in range(3):
                ox = 1.0 if k == 1 else 0.0
                oy = 1.0 if k == 2 else 0.0
                a = (px + 0.5 + ox - half) / half * tanh
                e = (half - py - 0.5 - oy) / half * tanh
                dirs[k, 0] = cam[9] + a * cam[3] + e * cam[6]
                dirs[k, 1] = cam[10] + a * cam[4] + e * cam[7]
                dirs[k, 2] = cam[11] + a * cam[5] + e * cam[8]
            f, u, v = _cube_face(dirs[0, 0], dirs[0, 2], -dirs[0, 1])
            ux, vx = _cube_coords(f, dirs[1, 0], dirs[1, 2], -dirs[1, 1])
            uy, vy = _cube_coords(f, dirs[2, 0], dirs[2, 2], -dirs[2, 1])
            n = levels[first + f * nlev, 1]
            rho = max(math.sqrt((ux - u) ** 2 + (vx - v) ** 2), math.sqrt((uy - u) ** 2 + (vy - v) ** 2)) * n
            _trilinear(data, levels, first + f * nlev, nlev, u, v, rho, False, c, tmp)
            for ch in range(3):
                out[py, px, ch] = np.uint8(math.floor(min(max(c[ch], 0.0), 1.0) * 255.0 + 0.5))


@njit(parallel=True, cache=True, error_model="numpy", fastmath=True)
def _render(geo, tex, sets, gpos, gmat, rgba, cam, lights, ambient, tanh, near, headpos, sort_rbound, smat, slook, smap, plate, cast, ss, bands, out):
    """Draw ``out`` (B, size, size, 3); each frame splits into ``bands`` row bands drawn as separate tasks."""
    B = gpos.shape[0]
    size = out.shape[1]
    res = size * ss
    per = -(-size // bands)
    cstarts = geo[4]
    G = gpos.shape[1]
    jobs = B * bands
    nthr = min(numba.get_num_threads(), jobs)
    nsamp = per * ss * res * 4
    cells = (smap.shape[0] + CELL - 1) // CELL
    for task in prange(nthr):
        buf = (
            np.empty(nsamp, np.float32),
            np.empty(nsamp * 3, np.uint8),
            np.empty(nsamp * 3, np.uint8),
            np.empty(nsamp, np.uint8),
            np.empty(nsamp, np.int32),
            np.empty(nsamp, np.float32),
            np.empty(nsamp, np.uint8),
        )
        qbuf = (
            np.empty(nsamp, np.int32),
            np.empty(nsamp, np.int32),
            np.empty(nsamp, np.int32),
            np.empty(nsamp, np.int32),
            np.empty(nsamp),
            np.empty(cells * cells + 1, np.int32),
            np.empty(nsamp, np.int32),
        )
        work = _work_nb(cstarts[-1], G)
        nc = cast[2][-1]
        lwork = (np.empty(nc), np.empty(nc), np.empty(nc))
        for job in range(task, jobs, nthr):
            _frame(job, bands, per, size, ss, res, geo, tex, sets, gpos, gmat, rgba, cam, lights, ambient, tanh, near,
                   headpos, sort_rbound, smat, slook, smap, plate, cast, out, buf, qbuf, work, lwork)


@njit(cache=True, error_model="numpy", fastmath=True)
def _frame(job, bands, per, size, ss, res, geo, tex, sets, gpos, gmat, rgba, cam, lights, ambient, tanh, near, headpos,
           sort_rbound, smat, slook, smap, plate, cast, out, buf, qbuf, work, lwork):
    movers, glass, casters = sets
    pdepth, pcol, plit, pkind, pkey, prr, sky = plate
    depth, col, add, kind, key, rr, mv = buf
    xf = work[9]
    b = job // bands
    band = job % bands
    o0 = band * per
    o1 = min(size, o0 + per)
    if o0 >= o1:
        return
    y0b = o0 * ss
    y1b = o1 * ss
    n = (y1b - y0b) * res * 4
    s0 = y0b * res * 4
    # The frame starts from the static samples, fully lit; ``mv`` marks the samples the moving geoms draw.
    for i in range(n):
        depth[i] = pdepth[s0 + i]
        mv[i] = 0
    for i in range(3 * n):
        col[i] = plit[3 * s0 + i]
    for gi in range(xf.shape[0]):
        xf[gi] = False
    bufn = (depth, col, add, kind, key, rr, mv)
    for gi in movers:
        _geom(gi, geo, tex, gpos[b, gi], gmat[b, gi], rgba[b, gi], cam, lights, ambient, tanh, near, slook, smap, res,
              y0b, y1b, bufn, False, work)
    qd = qbuf[4]
    qid = qbuf[6]
    _mover_shadows(cast, casters, gpos[b], gmat[b], smat, smap.shape[0], n, s0, mv, kind, key, pkind, pkey, qbuf, lwork)
    for i in range(n):
        if mv[i]:
            if kind[i] == 1 and (qid[i] < 0 or rr[i] >= qd[qid[i]]):
                for ch in range(3):
                    v = int(col[i * 3 + ch]) + int(add[i * 3 + ch])
                    col[i * 3 + ch] = np.uint8(min(v, 255))
        elif qid[i] >= 0 and prr[s0 + i] < qd[qid[i]]:
            for ch in range(3):
                col[i * 3 + ch] = pcol[3 * (s0 + i) + ch]
        if depth[i] == 0:
            p = i // 4
            py = y0b + p // res
            px = p % res
            for ch in range(3):
                col[i * 3 + ch] = sky[py, px, ch]
    # Translucent geoms sorted by distance to the camera less the bounding radius.
    nt = glass.shape[0]
    if nt > 0:
        dist = np.empty(nt)
        for j in range(nt):
            gi = glass[j]
            ddx = np.float32(gpos[b, gi, 0]) - headpos[0]
            ddy = np.float32(gpos[b, gi, 1]) - headpos[1]
            ddz = np.float32(gpos[b, gi, 2]) - headpos[2]
            dist[j] = np.float32(math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)) - sort_rbound[gi]
        idx = np.argsort(dist, kind="mergesort")
        for p in range(2 * nt):
            gi = glass[idx[p]] if p < nt else glass[idx[2 * nt - 1 - p]]
            _geom(gi, geo, tex, gpos[b, gi], gmat[b, gi], rgba[b, gi], cam, lights, ambient, tanh, near, slook, smap,
                  res, y0b, y1b, bufn, True, work)
    if ss == 1:
        for oy in range(o0, o1):
            for ox in range(size):
                p0 = ((oy - o0) * res + ox) * 12
                for ch in range(3):
                    v = (
                        np.int32(col[p0 + ch])
                        + np.int32(col[p0 + 3 + ch])
                        + np.int32(col[p0 + 6 + ch])
                        + np.int32(col[p0 + 9 + ch])
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
                            int(col[(p0) * 3 + ch])
                            + int(col[(p0 + 1) * 3 + ch])
                            + int(col[(p0 + 2) * 3 + ch])
                            + int(col[(p0 + 3) * 3 + ch])
                            + 2
                        ) // 4
                out[b, oy, ox, ch] = np.uint8(np.rint(acc / (ss * ss)))


@njit(cache=True, error_model="numpy", fastmath=True)
def _mover_shadows(cast, casters, gpos, gmat, smat, nmap, n, s0, mv, kind, key, pkind, pkey, qbuf, lwork):
    """Depth of the moving casters at the shadow-map texels the lit samples look up.

    Each pixel's distinct texels inside the casters' light-space bounds become queries, bucketed in cells of
    ``CELL`` texels; each caster triangle that faces the light tests the queries of the cells under it (along a
    light ray the first surface of a closed mesh faces the light). ``qbuf[6]`` gets each sample's query (-1 for
    none) and ``qbuf[4]`` the 24-bit depth of each query (0 where no caster covers it). Samples the moving geoms
    did not draw read their shadow data from the plate arrays ``pkind`` and ``pkey`` at offset ``s0``.
    """
    cpos, tris, cstarts, starts, rbound = cast
    qs, qt, qc, order, qd, counts, qid = qbuf
    lx, ly, lz = lwork
    for i in range(n):
        qid[i] = -1
    if casters.shape[0] == 0 or nmap <= 1:
        return
    sc = math.sqrt(smat[0, 0] ** 2 + smat[0, 1] ** 2 + smat[0, 2] ** 2)
    bx0 = by0 = 1e300
    bx1 = by1 = -1e300
    for gi in casters:
        cx = smat[0, 0] * gpos[gi, 0] + smat[0, 1] * gpos[gi, 1] + smat[0, 2] * gpos[gi, 2] + smat[0, 3]
        cy = smat[1, 0] * gpos[gi, 0] + smat[1, 1] * gpos[gi, 1] + smat[1, 2] * gpos[gi, 2] + smat[1, 3]
        rad = rbound[gi] * sc + 1.0
        bx0 = min(bx0, cx - rad)
        bx1 = max(bx1, cx + rad)
        by0 = min(by0, cy - rad)
        by1 = max(by1, cy + rad)
    sx0 = max(int(math.floor(bx0)), 1)
    sx1 = min(int(math.floor(bx1)), nmap - 2)
    sy0 = max(int(math.floor(by0)), 1)
    sy1 = min(int(math.floor(by1)), nmap - 2)
    if sx0 > sx1 or sy0 > sy1:
        return
    cx0 = sx0 // CELL
    cy0 = sy0 // CELL
    ncx = sx1 // CELL - cx0 + 1
    ncy = sy1 // CELL - cy0 + 1
    for c in range(ncx * ncy + 1):
        counts[c] = 0
    nq = 0
    for p in range(n // 4):
        for k in range(4):
            i = p * 4 + k
            if mv[i]:
                if kind[i] != 1:
                    continue
                kk = key[i]
            else:
                if pkind[s0 + i] != 1:
                    continue
                kk = pkey[s0 + i]
            s = kk % nmap
            t = kk // nmap
            if s < sx0 or s > sx1 or t < sy0 or t > sy1:
                continue
            found = -1
            for j in range(k):
                if qid[p * 4 + j] >= 0 and qs[qid[p * 4 + j]] == s and qt[qid[p * 4 + j]] == t:
                    found = qid[p * 4 + j]
            if found >= 0:
                qid[i] = found
                continue
            qs[nq] = s
            qt[nq] = t
            c = (t // CELL - cy0) * ncx + (s // CELL - cx0)
            qc[nq] = c
            counts[c + 1] += 1
            qd[nq] = 0.0
            qid[i] = nq
            nq += 1
    if nq == 0:
        return
    for c in range(ncx * ncy):
        counts[c + 1] += counts[c]
    for q in range(nq):
        c = qc[q]
        order[counts[c]] = q
        counts[c] += 1
    # counts[c] is now the end of cell c; cell c starts at counts[c - 1] (0 for the first cell).
    for gi in casters:
        cxg = smat[0, 0] * gpos[gi, 0] + smat[0, 1] * gpos[gi, 1] + smat[0, 2] * gpos[gi, 2] + smat[0, 3]
        cyg = smat[1, 0] * gpos[gi, 0] + smat[1, 1] * gpos[gi, 1] + smat[1, 2] * gpos[gi, 2] + smat[1, 3]
        rad = rbound[gi] * sc + 1.0
        if cxg + rad < sx0 or cxg - rad > sx1 + 1 or cyg + rad < sy0 or cyg - rad > sy1 + 1:
            continue
        c0 = cstarts[gi]
        c1 = cstarts[gi + 1]
        gm = gmat[gi]
        gp = gpos[gi]
        # World to light window coordinates of this geom's corners, composed once per geom.
        m00 = smat[0, 0] * gm[0] + smat[0, 1] * gm[3] + smat[0, 2] * gm[6]
        m01 = smat[0, 0] * gm[1] + smat[0, 1] * gm[4] + smat[0, 2] * gm[7]
        m02 = smat[0, 0] * gm[2] + smat[0, 1] * gm[5] + smat[0, 2] * gm[8]
        m10 = smat[1, 0] * gm[0] + smat[1, 1] * gm[3] + smat[1, 2] * gm[6]
        m11 = smat[1, 0] * gm[1] + smat[1, 1] * gm[4] + smat[1, 2] * gm[7]
        m12 = smat[1, 0] * gm[2] + smat[1, 1] * gm[5] + smat[1, 2] * gm[8]
        m20 = smat[2, 0] * gm[0] + smat[2, 1] * gm[3] + smat[2, 2] * gm[6]
        m21 = smat[2, 0] * gm[1] + smat[2, 1] * gm[4] + smat[2, 2] * gm[7]
        m22 = smat[2, 0] * gm[2] + smat[2, 1] * gm[5] + smat[2, 2] * gm[8]
        t0 = cxg
        t1 = cyg
        t2 = smat[2, 0] * gp[0] + smat[2, 1] * gp[1] + smat[2, 2] * gp[2] + smat[2, 3]
        for j in range(c0, c1):
            px, py, pz = cpos[j, 0], cpos[j, 1], cpos[j, 2]
            lx[j] = m00 * px + m01 * py + m02 * pz + t0
            ly[j] = m10 * px + m11 * py + m12 * pz + t1
            lz[j] = m20 * px + m21 * py + m22 * pz + t2
        for t in range(starts[gi], starts[gi + 1]):
            i0, i1, i2 = tris[t, 0], tris[t, 1], tris[t, 2]
            x0, y0 = lx[i0], ly[i0]
            x1, y1 = lx[i1], ly[i1]
            x2, y2 = lx[i2], ly[i2]
            det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
            if det <= 0.0:
                continue
            xa = max(int(math.ceil(min(x0, min(x1, x2)) - 0.5)), sx0)
            xb = min(int(math.floor(max(x0, max(x1, x2)) - 0.5)), sx1)
            if xa > xb:
                continue
            ya = max(int(math.ceil(min(y0, min(y1, y2)) - 0.5)), sy0)
            yb = min(int(math.floor(max(y0, max(y1, y2)) - 0.5)), sy1)
            if ya > yb:
                continue
            z0, z1, z2 = lz[i0], lz[i1], lz[i2]
            dzdx = ((z1 - z0) * (y2 - y0) - (z2 - z0) * (y1 - y0)) / det
            dzdy = ((x1 - x0) * (z2 - z0) - (x2 - x0) * (z1 - z0)) / det
            off = OFFSET_FACTOR * max(abs(dzdx), abs(dzdy)) + OFFSET_UNITS / DEPTH_BITS
            for cyc in range(ya // CELL - cy0, yb // CELL - cy0 + 1):
                for cxc in range(xa // CELL - cx0, xb // CELL - cx0 + 1):
                    c = cyc * ncx + cxc
                    start = counts[c - 1] if c > 0 else 0
                    for o in range(start, counts[c]):
                        q = order[o]
                        tx = qs[q]
                        ty = qt[q]
                        if tx < xa or tx > xb or ty < ya or ty > yb:
                            continue
                        cx = tx + 0.5
                        cy = ty + 0.5
                        e0 = (x1 - cx) * (y2 - cy) - (x2 - cx) * (y1 - cy)
                        e1 = (x2 - cx) * (y0 - cy) - (x0 - cx) * (y2 - cy)
                        e2 = (x0 - cx) * (y1 - cy) - (x1 - cx) * (y0 - cy)
                        if e0 < 0 or e1 < 0 or e2 < 0:
                            continue
                        z = _quantize(z0 + dzdx * (cx - x0) + dzdy * (cy - y0) + off)
                        if z > qd[q]:
                            qd[q] = z
