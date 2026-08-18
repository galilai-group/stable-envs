"""Shared plumbing for the MJWarp re-measurement of the six locomotion domains.

Everything here exists so that the two renderers are shown THE SAME SCENE. Three of the four ways
they can silently differ are handled in ``stable_envs.render.mjwarp``; the fourth is specific to this
family and is the reason this file exists at all:

  ``mujoco_warp.kinematics`` does NOT compute ``subtree_com``, and a ``trackcom`` camera's position
  IS ``subtree_com[body] + cam_poscom0``. So the four-call frame loop the repo uses everywhere else
  (kinematics -> camlight -> refit_bvh -> render) leaves the camera FROZEN at whatever
  ``put_data`` uploaded. ``com_pos`` has to run between the first two.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("WARP_CACHE_PATH", os.path.expanduser("~/.cache/warp"))

import numpy as np

ENVS = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5", "Ant-v5", "Humanoid-v5", "Swimmer-v5"]

VISIBLE_GROUPS = (0, 1, 2)


# --------------------------------------------------------------------------- reference side

def make_env(env_id, size):
    import gymnasium as gym
    env = gym.make(env_id, render_mode="rgb_array", width=int(size), height=int(size))
    u = env.unwrapped
    u.model.vis.global_.offwidth = max(int(u.model.vis.global_.offwidth), int(size))
    u.model.vis.global_.offheight = max(int(u.model.vis.global_.offheight), int(size))
    env.reset(seed=0)
    u.render()
    return env, u


def track_cam(m):
    import mujoco
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "track")


def gym_shot(u):
    u.mujoco_renderer.viewer.make_context_current()
    return np.ascontiguousarray(u.render())


_RCACHE = {}


def renderer(m, size, tag="ref"):
    """One ``mujoco.Renderer`` per (model, size, tag), built once and NEVER closed."""
    import mujoco
    key = (id(m), int(size), tag)
    hit = _RCACHE.get(key)
    if hit is None:
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), int(size))
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), int(size))
        hit = _RCACHE[key] = mujoco.Renderer(m, height=int(size), width=int(size))
    return hit


def mj_render(m, d, cam, size, qpos_batch, tag="ref"):
    """``(B,size,size,3)`` uint8 from MuJoCo's own rasterizer at the model camera ``cam``."""
    import mujoco
    r = renderer(m, size, tag)
    out = []
    for q in np.asarray(qpos_batch, np.float64):
        d.qpos[:] = q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=int(cam))
        out.append(np.ascontiguousarray(r.render()))
    return np.stack(out)


def mj_segment(m, d, cam, size, qpos_batch, tag="seg"):
    """``(B,size,size,2)`` int32 [objid, objtype] from MuJoCo's own segmentation buffer."""
    import mujoco
    r = renderer(m, size, tag)
    out = []
    r.enable_segmentation_rendering()
    try:
        for q in np.asarray(qpos_batch, np.float64):
            d.qpos[:] = q
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera=int(cam))
            out.append(np.ascontiguousarray(r.render())[..., :2].astype(np.int32))
    finally:
        r.disable_segmentation_rendering()
    return np.stack(out)


# --------------------------------------------------------------------------- MJWarp side

class MjwRenderer:
    """MJWarp with every known correction applied, and ``com_pos`` in the frame loop.

    Corrections, all of them measured elsewhere in this repo and none of them optional:
      * ``gl_rounding()`` around context creation AND the first launch (truncate -> round).
      * ``prepare_model`` for geoms AND sites (zero alpha would be painted opaque).
      * BGRA -> RGB channel swap.
      * shadows ON (MuJoCo shadows by default), skybox ON, groups 0-2 on both sides.
    """

    def __init__(self, m, cam, res, nworld, shadows=True, textures=True, skybox=True,
                 specular=True, fast_math=True, prepare=True):
        import mujoco
        import torch
        import warp as wp

        import mujoco_warp as mjw
        from stable_envs.render import mjwarp as MW

        self.prepared = MW.prepare_model(m, VISIBLE_GROUPS) if prepare else None
        self.texture_gaps = MW.texture_gaps(m, VISIBLE_GROUPS)
        self.m_mj, self.res, self.nworld = m, int(res), int(nworld)
        d0 = mujoco.MjData(m)
        mujoco.mj_forward(m, d0)
        self.wm = mjw.put_model(m)
        self.wd = mjw.put_data(m, d0, nworld=int(nworld))
        with MW.gl_rounding():
            self.rc = MW.create_render_context(
                m, nworld=int(nworld), cam_res=(int(res), int(res)), render_rgb=True,
                render_seg=True,
                cam_active=[i == int(cam) for i in range(m.ncam)],
                enabled_geom_groups=list(VISIBLE_GROUPS), use_textures=textures,
                use_shadows=shadows, render_skybox=skybox, enable_specular=specular,
                use_fast_math=fast_math)
            self._frame()
            wp.synchronize()
        self.wp, self.mjw, self.torch = wp, mjw, torch
        self.bgra = wp.to_torch(self.rc.rgb_data).view(torch.uint8).reshape(
            int(nworld), int(res), int(res), 4)
        self.seg = wp.to_torch(self.rc.seg_data).reshape(int(nworld), int(res), int(res), 2)

    def _frame(self, com=True):
        self.mjw = __import__("mujoco_warp") if not hasattr(self, "mjw") else self.mjw
        mjw = self.mjw
        mjw.kinematics(self.wm, self.wd)
        if com:
            # THE ONE THIS FAMILY NEEDS. A trackcom camera is subtree_com + cam_poscom0, and
            # kinematics does not touch subtree_com.
            mjw.com_pos(self.wm, self.wd)
        mjw.camlight(self.wm, self.wd)
        mjw.refit_bvh(self.wm, self.wd, self.rc)
        mjw.render(self.wm, self.wd, self.rc)

    def render(self, qpos_batch, com=True):
        q = np.asarray(qpos_batch, np.float32)
        n = q.shape[0]
        assert n <= self.nworld, f"{n} states into {self.nworld} worlds"
        qt = self.wp.to_torch(self.wd.qpos)
        qt[:n] = self.torch.as_tensor(q, device=qt.device, dtype=qt.dtype)
        if n < self.nworld:
            qt[n:] = qt[n - 1:n]
        self.torch.cuda.synchronize()
        self._frame(com=com)
        self.wp.synchronize()
        rgb = self.bgra[:n][..., [2, 1, 0]].contiguous().cpu().numpy()
        return rgb

    def cam_xpos(self):
        return self.wp.to_torch(self.wd.cam_xpos).cpu().numpy()


# --------------------------------------------------------------------------- scoring

def interior_mask(frames):
    import scipy.ndimage as ndi
    out = []
    for f in frames:
        g = f.astype(np.int16)
        flat = (np.abs(np.diff(g, axis=0)).sum(-1)[:, :-1]
                + np.abs(np.diff(g, axis=1)).sum(-1)[:-1]) == 0
        out.append(ndi.minimum_filter(flat, size=3))
    return np.stack(out)


def agreement(a, b, extra_mask=None):
    """``a`` under test, ``b`` the reference. Same definitions as ``testing/conformance.py``."""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
    res = {"mean_abs": round(float(d.mean()), 4), "exact_frac": round(float((d == 0).mean()), 6),
           "within8_frac": round(float((d <= 8).mean()), 6), "max_abs": int(d.max())}
    m = interior_mask(b)
    if extra_mask is not None:
        m = m & extra_mask[:, :-1, :-1]
    di = d[:, :-1, :-1][m]
    res["interior_frac"] = round(float(m.mean()), 6)
    res["interior_n"] = int(di.size)
    if di.size:
        res["interior_exact"] = round(float((di == 0).mean()), 6)
        res["interior_mean"] = round(float(di.mean()), 4)
        res["interior_within8"] = round(float((di <= 8).mean()), 6)
    else:
        res["interior_exact"] = None
        res["interior_mean"] = None
        res["interior_within8"] = None
    return res


# --------------------------------------------------------------------------- states

def sample_qpos(env_id, n, seed=0, steps=60):
    """The env's own reset distribution plus a random-policy rollout, exactly as the shipped four."""
    import gymnasium as gym
    env = gym.make(env_id)
    env.action_space.seed(seed)
    u = env.unwrapped
    rng = np.random.default_rng(seed)
    lo = np.asarray(env.action_space.low, np.float64)
    hi = np.asarray(env.action_space.high, np.float64)
    out = np.empty((int(n), int(u.model.nq)), np.float64)
    for i in range(int(n)):
        env.reset(seed=int(rng.integers(0, 2 ** 31 - 1)))
        for _ in range(int(rng.integers(0, steps + 1))):
            _, _, term, trunc, _ = env.step(rng.uniform(lo, hi).astype(np.float32))
            if term or trunc:
                env.reset(seed=int(rng.integers(0, 2 ** 31 - 1)))
        out[i] = np.asarray(u.data.qpos, np.float64)
    env.close()
    return out


def jsave(path, obj):
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    print("wrote", path)
