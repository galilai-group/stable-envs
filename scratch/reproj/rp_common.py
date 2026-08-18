"""Shared machinery for the depth-reprojection experiments. Nothing here touches ``stable_envs``.

The one fact everything below rests on: ``mujoco_warp``'s megakernel reads its primary ray as
``ray[rayid]`` -- a FLAT array of directions in the CAMERA's own frame, shared by every world -- and
then transforms it by ``cam_xmat[worldid]`` about ``cam_xpos[worldid]``
(``mujoco_warp/_src/render.py``, the ``use_precomputed_rays`` branch). Two consequences are used
throughout:

* a context built at ``cam_res=(S,S)`` whose ``rc.ray`` is overwritten with an ARBITRARY gather of
  another table's directions renders exactly those pixels -- the same pixel SET in every world;
* the camera pose is per world, so one launch can bake 64 plates from 64 different viewpoints.

``rc.depth_data`` is PLANAR depth (``dist * -ray_z``, i.e. the projection onto the optical axis) and
is 0.0 exactly on a miss.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch

FAMILIES = ("HumanoidBench", "RoboCasa")

#: What a miss is reprojected as. Far enough that a 3 cm baseline moves it by <1e-5 px, near enough
#: that float32 keeps the direction: the skybox is a function of the WORLD ray direction only, so a
#: pure translation must leave it where it was and a rotation must carry it. Both fall out of putting
#: the point at this distance and running it through the same transform as everything else.
SKY_DEPTH = 1.0e4


def family(name: str) -> str:
    for f in FAMILIES:
        if name.startswith(f):
            return f
    raise SystemExit(f"unknown family for {name}")


def module_for(name: str):
    import importlib
    return importlib.import_module({
        "HumanoidBench": "stable_envs.envs.humanoidbench",
        "RoboCasa": "stable_envs.envs.robocasa"}[family(name)])


def model_of(mod, cls):
    if family(cls.__name__) == "HumanoidBench":
        return mod._build_bundle(cls.ROBOT, cls.TASK).model
    return mod._scene(cls.KITCHEN).model


def cam_id_of(mod, cls, m) -> int:
    import mujoco
    if family(cls.__name__) == "HumanoidBench":
        return int(mod._build_bundle(cls.ROBOT, cls.TASK).info["cam"])
    return int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cls.CAMERA))


def full_qpos(mod, cls, env, m, states: np.ndarray) -> np.ndarray:
    """``(N, state_dim)`` -> ``(N, nq)`` the way the env's own render does it."""
    st = np.asarray(states, np.float64)
    if family(cls.__name__) == "HumanoidBench":
        q = np.zeros((len(st), int(m.nq)), np.float64)
        q[:, :] = st[:, :int(m.nq)]
        return q
    s = mod._scene(cls.KITCHEN)
    q = np.repeat(s.pinned[None], len(st), 0)
    q[:, env._live] = st
    return q


def kinematics_cpu(m, qpos: np.ndarray, cid: int):
    """``(N,nq)`` -> ``cam_xpos (N,3)``, ``cam_xmat (N,3,3)``, ``geom_xpos (N,ngeom,3)``,
    ``geom_xmat (N,ngeom,3,3)``. MuJoCo on the CPU, which is the reference for where things are."""
    import mujoco
    d = mujoco.MjData(m)
    cp, cm, gp, gm = [], [], [], []
    for row in np.asarray(qpos, np.float64):
        d.qpos[:] = row
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        cp.append(np.array(d.cam_xpos[cid], np.float64))
        cm.append(np.array(d.cam_xmat[cid], np.float64).reshape(3, 3))
        gp.append(np.array(d.geom_xpos, np.float64))
        gm.append(np.array(d.geom_xmat, np.float64).reshape(-1, 3, 3))
    return np.stack(cp), np.stack(cm), np.stack(gp), np.stack(gm)


def moving_geoms(m, gp: np.ndarray, gm: np.ndarray, vis_groups) -> np.ndarray:
    """Visible geoms whose world transform MOVES over the states actually handed in.

    ``body_weldid != 0`` is the wrong question on a pinned scene: a RoboCasa kitchen hangs every
    cabinet door off a hinge, so 176 of 344 visible geoms are 'not welded' while the env pins their
    joints and no state it can be handed will ever move them.
    """
    moved = ((gp - gp[0]).any(0).any(-1) | (gm - gm[0]).reshape(len(gm), -1, 9).any(0).any(-1))
    return np.array([g for g in np.flatnonzero(moved)
                     if int(m.geom_group[g]) in set(int(x) for x in vis_groups)], np.int64)


# --------------------------------------------------------------------------- render contexts

class Ctx:
    __slots__ = ("wm", "wd", "rc", "bgra", "depth", "graph", "graph_kin", "graph_draw", "nworld",
                 "res", "wp", "mjw", "cam", "needs_com", "ray0", "mod", "m")


def make_ctx(mod, cls, m, res: int, nworld: int, hide=(), depth: bool = True,
             skybox: bool = True, share: Ctx | None = None) -> Ctx:
    """A render context built the way the env's own ``_context`` builds it, plus depth and a hide set.

    ``hide`` is applied to ``m.geom_group`` for the duration of ``create_render_context`` ONLY --
    MJWarp reads the geometry once, there, so restoring afterwards un-draws nothing and keeps the
    model usable by the CPU reference.
    """
    import mujoco
    import warp as wp

    import mujoco_warp as mjw

    import stable_envs.render.mjwarp as MW
    from stable_envs.render import mjwarp_lightspec as LS

    fam = family(cls.__name__)
    cid = cam_id_of(mod, cls, m)
    d0 = mujoco.MjData(m)
    if fam == "RoboCasa":
        d0.qpos[:] = mod._scene(cls.KITCHEN).pinned
    mujoco.mj_forward(m, d0)

    c = Ctx()
    c.wp, c.mjw, c.nworld, c.res, c.cam, c.mod, c.m = wp, mjw, int(nworld), int(res), cid, mod, m
    c.needs_com = MW.needs_com_pos(m)
    if fam == "RoboCasa":
        batched = {k: int(nworld) for k in ("cam_pos", "cam_quat", "light_diffuse")}
    else:
        batched = {k: int(nworld) for k in
                   ("mat_rgba", "light_diffuse", "cam_poscom0", "cam_mat0")}
    g0 = np.array(m.geom_group, copy=True)
    if len(hide):
        m.geom_group[np.asarray(hide, np.int64)] = MW.HIDDEN_GROUP
    if share is not None:
        # Several render contexts over ONE set of poses. ``kinematics`` is a property of the state,
        # not of the window it is about to be drawn into, so a ladder of window sizes that shares its
        # ``Data`` runs it once and lets each context draw from the same answer.
        c.wm, c.wd = share.wm, share.wd
    else:
        c.wm = mjw.put_model(m, batch_sizes=batched)
        if fam == "RoboCasa":
            c.wm.block_dim.render = 32
        c.wd = mjw.put_data(m, d0, nworld=int(nworld))
    # RoboCasa's shipped context quantises float->uint8 by ROUNDING; HumanoidBench's does not, and
    # both are deliberate (see each module). The capture is inside the patch because
    # ``mujoco_warp.render`` regenerates its megakernel source on every call.
    rounding = MW.gl_rounding() if fam == "RoboCasa" else _null()
    try:
        with rounding, LS.specialised_lights():
            kw = dict(nworld=int(nworld), cam_res=(int(res), int(res)), render_rgb=True,
                      cam_active=[i == cid for i in range(m.ncam)],
                      enabled_geom_groups=list(mod.VISIBLE_GROUPS), use_textures=True,
                      use_shadows=True, render_skybox=bool(skybox), render_depth=bool(depth))
            if fam == "HumanoidBench":
                kw["enable_specular"] = True
            c.rc = MW.create_render_context(m, **kw)
            frame(c)
            wp.synchronize()
            c.bgra = wp.to_torch(c.rc.rgb_data).view(torch.uint8).reshape(
                int(nworld), int(res), int(res), 4)
            c.depth = (wp.to_torch(c.rc.depth_data).reshape(int(nworld), int(res), int(res))
                       if depth else None)
            c.ray0 = wp.to_torch(c.rc.ray).clone()
            with wp.ScopedCapture() as cap:
                frame(c)
            c.graph = cap.graph
            # The same frame, cut in two. A windowed pass has to know WHERE the movers are before it
            # can choose its window, and that is ``kinematics``' answer -- so the halves are captured
            # separately and the caller runs the pose half, reads it, sets the ray table, and runs
            # the draw half. Splitting it costs nothing: no work is repeated, only the launch.
            with wp.ScopedCapture() as cap:
                c.mjw.kinematics(c.wm, c.wd)
                if c.needs_com:
                    c.mjw.com_pos(c.wm, c.wd)
                c.mjw.camlight(c.wm, c.wd)
            c.graph_kin = cap.graph
            with wp.ScopedCapture() as cap:
                c.mjw.refit_bvh(c.wm, c.wd, c.rc)
                c.mjw.render(c.wm, c.wd, c.rc)
            c.graph_draw = cap.graph
    finally:
        m.geom_group[:] = g0
    return c


import contextlib  # noqa: E402


@contextlib.contextmanager
def _null():
    yield


def frame(c: Ctx) -> None:
    c.mjw.kinematics(c.wm, c.wd)
    if c.needs_com:
        c.mjw.com_pos(c.wm, c.wd)
    c.mjw.camlight(c.wm, c.wd)
    c.mjw.refit_bvh(c.wm, c.wd, c.rc)
    c.mjw.render(c.wm, c.wd, c.rc)


def launch(c: Ctx) -> None:
    torch.cuda.synchronize()
    c.wp.capture_launch(c.graph)
    c.wp.synchronize()


def rgb_of(c: Ctx, n: int | None = None) -> torch.Tensor:
    return c.bgra[:n if n is not None else c.nworld][..., [2, 1, 0]].clone()


def parent_frame(m, cid: int, qpos: np.ndarray):
    """The pose of the body the camera hangs off, at ``qpos``. A camera at an ARBITRARY world pose is
    expressed as ``cam_pos``/``cam_quat`` in this frame, which are per-world model fields, so a batch
    of 64 arbitrary viewpoints is one launch."""
    import mujoco
    d = mujoco.MjData(m)
    d.qpos[:] = np.asarray(qpos, np.float64)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    b = int(m.cam_bodyid[cid])
    return np.array(d.xpos[b], np.float64), np.array(d.xmat[b], np.float64).reshape(3, 3)


def local_from_world(bpos, bmat, xpos: np.ndarray, xmat: np.ndarray):
    """World camera poses -> ``(cam_pos, cam_quat)`` in the parent body's frame."""
    import mujoco
    lp = (xpos - bpos) @ bmat                                  # bmat^T (xpos - bpos)
    lm = np.einsum("ji,njk->nik", bmat, xmat)                  # bmat^T @ xmat
    q = np.empty((len(xpos), 4))
    for i in range(len(xpos)):
        mujoco.mju_mat2Quat(q[i], np.ascontiguousarray(lm[i]).reshape(9))
    return lp, q


def set_camera(c: Ctx, cam_pos: np.ndarray, cam_quat: np.ndarray, n: int) -> None:
    t = lambda x: torch.as_tensor(np.asarray(x, np.float32), device="cuda")
    c.wp.to_torch(c.wm.cam_pos)[:n, c.cam] = t(cam_pos)
    c.wp.to_torch(c.wm.cam_quat)[:n, c.cam] = t(cam_quat)


def camera_error(c: Ctx, xpos: np.ndarray, xmat: np.ndarray, n: int):
    """After a launch: how far is the camera MJWarp actually used from the one that was asked for?

    The whole plate cache rests on being able to place the camera anywhere, so it is asserted rather
    than assumed -- ``cam_xmat`` is written by ``camlight`` inside the captured graph and there are
    two conventions it could have followed."""
    p = c.wp.to_torch(c.wd.cam_xpos)[:n, c.cam].cpu().numpy().astype(np.float64)
    r = c.wp.to_torch(c.wd.cam_xmat)[:n, c.cam].cpu().numpy().astype(np.float64).reshape(n, 3, 3)
    return float(np.abs(p - xpos[:n]).max()), float(np.abs(r - xmat[:n]).max())


def timed(fn, reps: int, warm: int = 3) -> float:
    import warp as wp
    for _ in range(warm):
        fn()
    wp.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        wp.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


# --------------------------------------------------------------------------- the camera model

class Pinhole:
    """The projection the megakernel's own ray table implies, recovered FROM that table.

    ``compute_ray`` builds ``normalize(x, y, -znear)`` with ``x`` linear in the pixel column and ``y``
    linear (and decreasing) in the row, so ``rx/-rz`` and ``ry/-rz`` are affine in the pixel index and
    two rays are enough to invert it. Recovering it rather than recomputing ``tan(fovy/2)`` from the
    model means a sensorsize/intrinsic camera needs no special case, and ``roundtrip_err`` asserts it.
    """

    def __init__(self, ray: torch.Tensor, res: int):
        r = ray.reshape(res, res, 3).to(torch.float64)
        u = (r[..., 0] / -r[..., 2])
        v = (r[..., 1] / -r[..., 2])
        # u is affine in the column, v affine in the row.
        self.res = int(res)
        self.au = float(u[0, 1] - u[0, 0])
        self.bu = float(u[0, 0])
        self.av = float(v[1, 0] - v[0, 0])
        self.bv = float(v[0, 0])

    def project(self, pc: torch.Tensor):
        """``(...,3)`` camera-frame points -> float pixel ``(col, row)`` and validity (z < 0)."""
        z = pc[..., 2]
        ok = z < -1e-6
        zz = torch.where(ok, z, torch.full_like(z, -1.0))
        u = pc[..., 0] / -zz
        v = pc[..., 1] / -zz
        return (u - self.bu) / self.au, (v - self.bv) / self.av, ok

    def roundtrip_err(self, ray: torch.Tensor) -> float:
        r = ray.reshape(self.res, self.res, 3).to(torch.float64)
        col, row, _ = self.project(r)
        jj, ii = torch.meshgrid(torch.arange(self.res, device=r.device, dtype=torch.float64),
                                torch.arange(self.res, device=r.device, dtype=torch.float64),
                                indexing="ij")
        return float(torch.max((col - ii).abs().max(), (row - jj).abs().max()))


# --------------------------------------------------------------------------- reprojection

def unproject(ray: torch.Tensor, depth: torch.Tensor, xpos: torch.Tensor, xmat: torch.Tensor,
              sky_depth: float = SKY_DEPTH) -> torch.Tensor:
    """``(P,3)`` rays + ``(N,P)`` PLANAR depth + pose -> ``(N,P,3)`` world points.

    A miss (depth exactly 0.0) is placed at ``sky_depth`` along its own ray, which is what makes the
    skybox come out right: the skybox is a function of the world ray direction alone, so a point at
    that distance is carried by a rotation and unmoved by any translation this family sees.
    """
    dz = -ray[:, 2]                                            # cos(theta) to the optical axis, > 0
    dist = torch.where(depth > 0, depth, torch.full_like(depth, float(sky_depth))) / dz
    pc = ray.unsqueeze(0) * dist.unsqueeze(-1)                 # (N,P,3) camera frame
    return xpos.unsqueeze(1) + torch.einsum("nij,npj->npi", xmat, pc)


def splat(P: torch.Tensor, src_rgb: torch.Tensor, xpos_b: torch.Tensor, xmat_b: torch.Tensor,
          cam: Pinhole, res: int):
    """Forward-splat world points into camera B with a z-buffer. Returns (rgb, depth, hole).

    One source pixel writes one target pixel, resolved by ``scatter_reduce`` over a single int64 key
    ``(depth_bits << 32) | source_index``: for POSITIVE floats the IEEE-754 bit pattern is monotone,
    so the minimum key is the nearest surface and its low half names the pixel that won. That is one
    pass and no atomics loop, and it makes the winner's index available for free.
    """
    n, p = P.shape[0], P.shape[1]
    rel = P - xpos_b.unsqueeze(1)
    pc = torch.einsum("nji,npj->npi", xmat_b, rel)             # R^T (P - t)
    col, row, ok = cam.project(pc)
    ci = torch.round(col).to(torch.int64)
    ri = torch.round(row).to(torch.int64)
    ok = ok & (ci >= 0) & (ci < res) & (ri >= 0) & (ri < res)
    z = (-pc[..., 2]).to(torch.float32).clamp_(min=1e-6)
    key = (z.view(torch.int32).to(torch.int64) << 32) | torch.arange(
        p, device=P.device, dtype=torch.int64).unsqueeze(0).expand(n, p)
    INIT = torch.iinfo(torch.int64).max
    tgt = ri * res + ci
    buf = torch.full((n, res * res), INIT, dtype=torch.int64, device=P.device)
    buf.scatter_reduce_(1, torch.where(ok, tgt, torch.zeros_like(tgt)),
                        torch.where(ok, key, torch.full_like(key, INIT)), "amin",
                        include_self=True)
    hole = buf == INIT
    q = res * res
    idx = (buf & 0xFFFFFFFF).clamp_(0, p - 1)
    rgb = torch.gather(src_rgb.reshape(n, p, 3), 1, idx.unsqueeze(-1).expand(n, q, 3))
    dep = (buf >> 32).to(torch.int32).view(torch.float32)
    dep = torch.where(hole, torch.zeros_like(dep), dep)
    return (rgb.reshape(n, res, res, 3), dep.reshape(n, res, res), hole.reshape(n, res, res))


def fill_holes(rgb: torch.Tensor, dep: torch.Tensor, hole: torch.Tensor, iters: int = 8):
    """Pull-push the cheap way: dilate the filled region into the holes, nearest-first.

    Deliberately the dumbest fill there is. It exists so the frontier has a row that costs nothing and
    the re-cast row has something to be better than.
    """
    r = rgb.permute(0, 3, 1, 2).to(torch.float32)
    d = dep.unsqueeze(1)
    h = hole.unsqueeze(1)
    for _ in range(iters):
        if not bool(h.any()):
            break
        w = (~h).to(torch.float32)
        num = torch.nn.functional.avg_pool2d(r * w, 3, 1, 1, count_include_pad=False)
        den = torch.nn.functional.avg_pool2d(w, 3, 1, 1, count_include_pad=False)
        dn = torch.nn.functional.avg_pool2d(d * w, 3, 1, 1, count_include_pad=False)
        got = den > 0
        r = torch.where(h & got, num / den.clamp(min=1e-8), r)
        d = torch.where(h & got, dn / den.clamp(min=1e-8), d)
        h = h & ~got
    return (r.round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1), d.squeeze(1),
            h.squeeze(1))


# --------------------------------------------------------------------------- metrics

def metrics(ours: np.ndarray, ref: np.ndarray) -> dict:
    """The four the visual tier declares, computed the way ``testing/conformance.py`` computes them."""
    from stable_envs.testing.conformance import ssim
    a = ours.astype(np.int16)
    b = ref.astype(np.int16)
    d = np.abs(a - b)
    return {"mean_abs": float(d.mean()), "max_abs": int(d.max()),
            "within8_frac": float((d.max(-1) <= 8).mean()),
            "exact_frac": float((d.max(-1) == 0).mean()),
            "ssim": ssim(ours, ref)}


def interior_metrics(ours: np.ndarray, ref: np.ndarray) -> dict:
    """``interior_*``: the same numbers restricted to pixels 2 px away from any edge in EITHER frame.

    ``conformance.pixel_agreement``'s rule, reproduced here so a probe does not need an env."""
    try:
        import scipy.ndimage as ndi
    except ImportError:
        return {}
    lum = lambda x: x.astype(np.float64) @ [0.299, 0.587, 0.114]
    e = np.zeros(ours.shape[:3], bool)
    for f in (lum(ours), lum(ref)):
        gx = ndi.sobel(f, axis=2)
        gy = ndi.sobel(f, axis=1)
        e |= (np.hypot(gx, gy) > 8.0)
    m = ~ndi.binary_dilation(e, np.ones((1, 5, 5), bool))
    if not m.any():
        return {"interior_frac": 0.0}
    d = np.abs(ours.astype(np.int16) - ref.astype(np.int16)).max(-1)[m]
    return {"interior_frac": float(m.mean()), "interior_mean_abs": float(d.mean()),
            "interior_exact_frac": float((d == 0).mean())}


def jpeg_roundtrip(arr: np.ndarray, quality: int = 95) -> np.ndarray:
    """What Lance stores: the frame through libjpeg at ``quality`` and back."""
    import io

    from PIL import Image
    out = np.empty_like(arr)
    for i in range(len(arr)):
        b = io.BytesIO()
        Image.fromarray(arr[i]).save(b, format="JPEG", quality=int(quality))
        b.seek(0)
        out[i] = np.asarray(Image.open(b).convert("RGB"))
    return out
