"""Translucent robot arm for the MJWarp Fetch path, blended the way MuJoCo's OpenGL renderer does.

MuJoCo draws translucent geoms after the opaque scene with depth writes off, sorted by distance
from the camera to each geom's bounding sphere, once front to back and once back to front, and
blends every front-facing fragment that passes the opaque depth test. MJWarp keeps only the closest
hit, so the opaque scene is rendered with the arm excluded and this kernel peels every front face
of every arm geom in front of it and blends them into the 8-bit frame in MuJoCo's order.
"""

# Warp kernels need int(...)/float(...) to declare variables that loops may reassign.
# ruff: noqa: UP018

from __future__ import annotations

import numpy as np
import warp as wp

MAX_FRAGS = 48
MAX_PEEL = 8
#: Same constant MJWarp scales material shininess by.
MAX_SHININESS = 128.0
_FAR = 1.0e10
_EPS = 1.0e-5


@wp.func
def _shade(
    base: wp.vec3,
    spec: float,
    shin: float,
    emis: float,
    normal: wp.vec3,
    view: wp.vec3,
    w: int,
    nlight: int,
    light_xdir: wp.array2d[wp.vec3],
    light_diffuse: wp.array2d[wp.vec3],
    light_specular: wp.array2d[wp.vec3],
    light_ambient: wp.array2d[wp.vec3],
    light_active: wp.array2d[bool],
    head_dir: wp.vec3,
    head_ambient: wp.vec3,
    head_diffuse: wp.vec3,
    head_specular: wp.vec3,
):
    out = base * emis + wp.cw_mul(base, head_ambient)
    for i in range(nlight + 1):
        L = head_dir
        diff = head_diffuse
        lspec = head_specular
        if i < nlight:
            if not light_active[w % light_active.shape[0], i]:
                continue
            out += wp.cw_mul(base, light_ambient[w % light_ambient.shape[0], i])
            L = -light_xdir[w, i]
            diff = light_diffuse[w % light_diffuse.shape[0], i]
            lspec = light_specular[w % light_specular.shape[0], i]
        ndotl = wp.max(0.0, wp.dot(normal, L))
        if ndotl > 0.0:
            out += wp.cw_mul(base, diff * ndotl)
            if spec > 0.0 and shin > 0.0:
                H = wp.normalize(L + view)
                out += lspec * (spec * wp.pow(wp.max(0.0, wp.dot(normal, H)), shin))
    return wp.max(wp.min(out, wp.vec3(1.0)), wp.vec3(0.0))


@wp.func
def _box_entry(lpnt: wp.vec3, lvec: wp.vec3, size: wp.vec3):
    """Distance and local normal of the face a ray enters a box through; -1 if it does not."""
    tmin = float(-_FAR)
    tmax = float(_FAR)
    axis = int(-1)
    side = float(0.0)
    for i in range(3):
        if wp.abs(lvec[i]) < 1.0e-12:
            if wp.abs(lpnt[i]) > size[i]:
                return -1.0, wp.vec3(0.0)
        else:
            a = (-size[i] - lpnt[i]) / lvec[i]
            b = (size[i] - lpnt[i]) / lvec[i]
            s = float(-1.0)
            if a > b:
                a, b = b, a
                s = 1.0
            if a > tmin:
                tmin = a
                axis = i
                side = s
            tmax = wp.min(tmax, b)
    if axis < 0 or tmin > tmax or tmin < 0.0:
        return -1.0, wp.vec3(0.0)
    n = wp.vec3(0.0)
    n[axis] = side
    return tmin, n


@wp.kernel
def _blend_arm(
    flags: wp.array[int],
    arm: wp.array[int],
    arm_mesh: wp.array[int],
    arm_size: wp.array[wp.vec3],
    arm_rbound: wp.array[float],
    arm_mat: wp.array[int],
    mesh_bvh_id: wp.array[wp.uint64],
    geom_xpos: wp.array2d[wp.vec3],
    geom_xmat: wp.array2d[wp.mat33],
    cam_xpos: wp.array2d[wp.vec3],
    cam_xmat: wp.array2d[wp.mat33],
    cam: int,
    ray: wp.array[wp.vec3],
    depth: wp.array2d[float],
    mat_rgba: wp.array2d[wp.vec4],
    mat_specular: wp.array2d[float],
    mat_shininess: wp.array2d[float],
    mat_emission: wp.array2d[float],
    nlight: int,
    light_xdir: wp.array2d[wp.vec3],
    light_diffuse: wp.array2d[wp.vec3],
    light_specular: wp.array2d[wp.vec3],
    light_ambient: wp.array2d[wp.vec3],
    light_active: wp.array2d[bool],
    head_ambient: wp.vec3,
    head_diffuse: wp.vec3,
    head_specular: wp.vec3,
    alpha: float,
    frame: wp.array3d[wp.uint8],
    touched: wp.array2d[wp.uint8],
):
    w, p = wp.tid()
    touched[w, p] = wp.uint8(0)
    if flags[w] == 0:
        return
    narm = arm.shape[0]
    local = ray[p]
    origin = cam_xpos[w, cam]
    cmat = cam_xmat[w, cam]
    vec = cmat @ local
    view = -vec
    head_dir = wp.vec3(cmat[0, 2], cmat[1, 2], cmat[2, 2])
    limit = float(_FAR)
    if depth[w, p] > 0.0:
        limit = depth[w, p] / (-local[2])

    key = wp.vector(0, length=MAX_FRAGS, dtype=int)
    cr = wp.vector(0.0, length=MAX_FRAGS, dtype=float)
    cg = wp.vector(0.0, length=MAX_FRAGS, dtype=float)
    cb = wp.vector(0.0, length=MAX_FRAGS, dtype=float)
    n = int(0)
    for k in range(narm):
        g = arm[k]
        pos = geom_xpos[w, g]
        # Skip geoms whose bounding sphere the ray misses or meets behind the opaque scene.
        oc = origin - pos
        b = wp.dot(oc, vec)
        disc = b * b - wp.dot(oc, oc) + arm_rbound[k] * arm_rbound[k]
        if disc < 0.0 or -b - wp.sqrt(disc) >= limit:
            continue
        mat = geom_xmat[w, g]
        matT = wp.transpose(mat)
        lpnt = matT @ oc
        lvec = matT @ vec
        mid = arm_mat[k]
        c = mat_rgba[w % mat_rgba.shape[0], mid]
        base = wp.vec3(c[0], c[1], c[2])
        spec = mat_specular[w % mat_specular.shape[0], mid]
        shin = mat_shininess[w % mat_shininess.shape[0], mid] * MAX_SHININESS
        emis = mat_emission[w % mat_emission.shape[0], mid]
        if arm_mesh[k] >= 0:
            start = float(0.0)
            for _layer in range(MAX_PEEL):
                t = float(0.0)
                u = float(0.0)
                v = float(0.0)
                sign = float(0.0)
                nrm = wp.vec3(0.0)
                f = int(-1)
                if not wp.mesh_query_ray(
                    mesh_bvh_id[arm_mesh[k]], lpnt + lvec * start, lvec, limit - start, t, u, v, sign, nrm, f
                ):
                    break
                start += t
                if wp.dot(lvec, nrm) < 0.0 and n < MAX_FRAGS:
                    col = _shade(
                        base,
                        spec,
                        shin,
                        emis,
                        wp.normalize(mat @ nrm),
                        view,
                        w,
                        nlight,
                        light_xdir,
                        light_diffuse,
                        light_specular,
                        light_ambient,
                        light_active,
                        head_dir,
                        head_ambient,
                        head_diffuse,
                        head_specular,
                    )
                    key[n] = k * (1 << 24) + f
                    cr[n] = col[0]
                    cg[n] = col[1]
                    cb[n] = col[2]
                    n += 1
                start += _EPS
        else:
            t, nrm = _box_entry(lpnt, lvec, arm_size[k])
            if t >= 0.0 and t < limit and n < MAX_FRAGS:
                col = _shade(
                    base,
                    spec,
                    shin,
                    emis,
                    mat @ nrm,
                    view,
                    w,
                    nlight,
                    light_xdir,
                    light_diffuse,
                    light_specular,
                    light_ambient,
                    light_active,
                    head_dir,
                    head_ambient,
                    head_diffuse,
                    head_specular,
                )
                key[n] = k * (1 << 24)
                cr[n] = col[0]
                cg[n] = col[1]
                cb[n] = col[2]
                n += 1
    if n == 0:
        return

    # Order geoms as MuJoCo does: by distance from the camera to the bounding sphere.
    for i in range(n):
        k = key[i] >> 24
        d0 = wp.length(geom_xpos[w, arm[k]] - origin) - arm_rbound[k]
        rank = int(0)
        for j in range(narm):
            dj = wp.length(geom_xpos[w, arm[j]] - origin) - arm_rbound[j]
            if dj < d0 or (dj == d0 and j < k):
                rank += 1
        key[i] = rank * (1 << 24) + (key[i] & ((1 << 24) - 1))

    for i in range(1, n):
        kk = key[i]
        r = cr[i]
        gg = cg[i]
        bb = cb[i]
        j = i - 1
        while j >= 0 and key[j] > kk:
            key[j + 1] = key[j]
            cr[j + 1] = cr[j]
            cg[j + 1] = cg[j]
            cb[j + 1] = cb[j]
            j -= 1
        key[j + 1] = kk
        cr[j + 1] = r
        cg[j + 1] = gg
        cb[j + 1] = bb

    a = alpha * 255.0
    keep = 1.0 - alpha
    R = float(frame[w, p, 0])
    G = float(frame[w, p, 1])
    B = float(frame[w, p, 2])
    # Front to back over the sorted geoms, then back to front; each geom's fragments keep face order.
    for i in range(n):
        R = wp.floor(a * cr[i] + keep * R + 0.5)
        G = wp.floor(a * cg[i] + keep * G + 0.5)
        B = wp.floor(a * cb[i] + keep * B + 0.5)
    hi = n - 1
    while hi >= 0:
        rank = key[hi] >> 24
        lo = hi
        while lo > 0 and (key[lo - 1] >> 24) == rank:
            lo -= 1
        for i in range(lo, hi + 1):
            R = wp.floor(a * cr[i] + keep * R + 0.5)
            G = wp.floor(a * cg[i] + keep * G + 0.5)
            B = wp.floor(a * cb[i] + keep * B + 0.5)
        hi = lo - 1
    frame[w, p, 0] = wp.uint8(int(R))
    frame[w, p, 1] = wp.uint8(int(G))
    frame[w, p, 2] = wp.uint8(int(B))
    touched[w, p] = wp.uint8(1)


class ArmBlender:
    """Static per-scene inputs of the blend kernel, read once from the compiled render model."""

    def __init__(self, m, arm_geoms):
        import mujoco

        mesh, box = int(mujoco.mjtGeom.mjGEOM_MESH), int(mujoco.mjtGeom.mjGEOM_BOX)
        kinds = {int(m.geom_type[g]) for g in arm_geoms}
        if kinds - {mesh, box}:
            raise NotImplementedError(f"translucent arm geoms must be meshes or boxes, got types {sorted(kinds)}")
        if any(int(t) != int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL) for t in m.light_type):
            raise NotImplementedError("the translucent arm blend models directional lights only")
        if any(int(x) >= 0 for g in arm_geoms for x in np.atleast_1d(m.mat_texid[m.geom_matid[g]])):
            raise NotImplementedError("translucent arm geoms must be untextured")
        g = np.asarray(arm_geoms, np.int64)
        self.arm = wp.array(g.astype(np.int32), dtype=int)
        self.mesh = wp.array(np.where(m.geom_type[g] == mesh, m.geom_dataid[g], -1).astype(np.int32), dtype=int)
        self.size = wp.array(np.asarray(m.geom_size[g], np.float32), dtype=wp.vec3)
        self.rbound = wp.array(np.asarray(m.geom_rbound[g], np.float32), dtype=float)
        self.mat = wp.array(np.asarray(m.geom_matid[g], np.int32), dtype=int)
        self.nlight = int(m.nlight)
        hl = m.vis.headlight
        self.head = (wp.vec3(*hl.ambient), wp.vec3(*hl.diffuse), wp.vec3(*hl.specular))

    def launch(self, wm, wd, rc, cam: int, flags, frame, touched, alpha: float):
        """Blend the arm into ``frame`` ``(nworld, npix, 3)`` uint8 for worlds with a nonzero flag.

        ``rc`` must have rendered depth with the arm excluded; ``touched`` receives the pixels changed.
        """
        nworld, npix = int(frame.shape[0]), int(frame.shape[1])
        wp.launch(
            _blend_arm,
            dim=(nworld, npix),
            inputs=[
                flags,
                self.arm,
                self.mesh,
                self.size,
                self.rbound,
                self.mat,
                rc.mesh_bvh_id,
                wd.geom_xpos,
                wd.geom_xmat,
                wd.cam_xpos,
                wd.cam_xmat,
                int(cam),
                rc.ray,
                rc.depth_data,
                wm.mat_rgba,
                wm.mat_specular,
                wm.mat_shininess,
                wm.mat_emission,
                self.nlight,
                wd.light_xdir,
                wm.light_diffuse,
                wm.light_specular,
                wm.light_ambient,
                wm.light_active,
                *self.head,
                float(alpha),
            ],
            outputs=[frame, touched],
        )
