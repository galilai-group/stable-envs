"""Shared plumbing for the HumanoidBench probes. One task per process, always.

Two things this family needs that the repo's other MuJoCo harnesses do not:

  * every task renders through a model camera whose ``mode`` is **TRACKCOM** (``cam_default``,
    ``cam_maze``, ``cam_tabletop``, ``cam_hurdle``, ``cam_basketball``, ``cam_hand_visible``,
    ``cam_inhand``); only ``cam_kitchen`` is a world-body camera. ``mujoco_warp.kinematics`` does not
    write ``subtree_com``, so ``com_pos`` has to run between ``kinematics`` and ``camlight`` or the
    camera never leaves its ``put_data`` pose.
  * the per-episode draw goes through the process-global ``np.random`` (``Task.reset_model`` calls
    ``np.random.uniform``), not through ``self.np_random``, so a state pool has to seed both.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("WARP_CACHE_PATH", os.path.expanduser("~/.cache/warp"))

import numpy as np

LOGS = "/mnt/vast/home/randall.balestriero/stable_envs_logs/humanoidbench"
VISIBLE_GROUPS = (0, 1, 2)
DEFAULT_RGBA = np.array([0.5, 0.5, 0.5, 1.0])

#: The 30 h1hand scenes plus h1's 20; `kitchen` is excluded everywhere because its MJCF does not
#: compile on mujoco 3.11 at all (see probe1).
LOCO = ["walk", "run", "stand", "reach", "stair", "slide", "crawl", "hurdle", "pole", "maze",
        "balance_simple", "balance_hard", "sit_simple", "sit_hard"]


def mj311_shim() -> list:
    """Let HumanoidBench's vendored dm_control named-indexer build on ``mujoco >= 3.11``.

    ``HumanoidEnv.__init__`` builds `self.named` by walking dm_control's static field table
    (``dmc_deps/dmc_sizes.py``, generated against mujoco 3.1) and calling ``getattr`` for every entry.
    Three model fields and fourteen data fields were renamed or removed after that table was
    generated, so construction dies with ``AttributeError: 'MjModel' object has no attribute
    'flex_xvert0'`` before any pixel exists. ``struct_indexer`` already SKIPS any field whose value is
    not an ndarray, so returning ``None`` for exactly the absent names is the whole fix; the named
    indexer then simply has 17 fewer entries, none of which HumanoidBench reads (it uses
    ``xmat``, ``site_xpos``, ``subtree_com``, ``sensordata``, ``geom_rgba``, ``body_dofadr``).

    Returns the names it shimmed so a caller can report them rather than trust them.
    """
    import humanoid_bench.dmc_deps.dmc_wrapper as W
    from humanoid_bench.dmc_deps import dmc_sizes as S

    import mujoco
    probe_m = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><geom size='.1'/><freejoint/></body></worldbody></mujoco>")
    probe_d = mujoco.MjData(probe_m)
    absent = {"mjmodel": frozenset(f for f in S.array_sizes["mjmodel"] if not hasattr(probe_m, f)),
              "mjdata": frozenset(f for f in S.array_sizes["mjdata"] if not hasattr(probe_d, f))}
    for cls, slot, key in ((W.MjModelWrapper, "_model", "mjmodel"),
                           (W.MjDataWrapper, "_data", "mjdata")):
        if getattr(cls, "_se_mj311_shim", False):
            continue
        gone = absent[key]

        def getter(self, name, _slot=slot, _gone=gone):
            inner = object.__getattribute__(self, _slot)
            if name in _gone and not hasattr(inner, name):
                return None
            return getattr(inner, name)

        cls.__getattr__ = getter
        cls._se_mj311_shim = True

    # gymnasium >= 1.0 moved the camera onto ``MujocoRenderer.__init__`` and left ``render`` taking
    # only the mode; HumanoidBench (pinned to gymnasium 0.29.1) still calls
    # ``render(mode, camera_id, camera_name)``. ``MujocoEnv.__init__`` already forwarded
    # ``camera_name`` to the constructor, so dropping the two extra arguments renders through exactly
    # the same camera. The two tasks that draw a goal marker override ``render`` and are patched the
    # same way, marker logic untouched.
    from humanoid_bench.envs.push import Push
    from humanoid_bench.envs.reach import Reach
    from humanoid_bench.tasks import Task

    def _render(self):
        return self._env.mujoco_renderer.render(self._env.render_mode)

    def _render_marker(self):
        for mk in self._env.viewer._markers:
            if mk["objid"] == 789:
                mk["pos"] = self.goal
                break
        else:
            self._env.viewer.add_marker(pos=self.goal, size=0.05, objid=789,
                                        rgba=(0.8, 0.28, 0.28, 1.0), label="")
        return self._env.mujoco_renderer.render(self._env.render_mode)

    Task.render = _render
    Reach.render = _render_marker
    Push.render = _render_marker
    return sorted(absent["mjmodel"] | absent["mjdata"])


def make_env(task: str, robot: str = "h1hand", size: int = 256):
    import gymnasium as gym

    import humanoid_bench  # noqa: F401  (registers the ids)
    mj311_shim()
    np.random.seed(0)
    env = gym.make(f"{robot}-{task}-v0", render_mode="rgb_array", width=int(size), height=int(size))
    u = env.unwrapped
    u.model.vis.global_.offwidth = max(int(u.model.vis.global_.offwidth), int(size))
    u.model.vis.global_.offheight = max(int(u.model.vis.global_.offheight), int(size))
    env.reset(seed=0)
    u.render()
    return env, u


def cam_id(u):
    import mujoco
    return mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_CAMERA, u.camera_name)


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


def set_state(m, d, q):
    """``qpos <- q``, ``qvel <- 0``, forward, and ASSERT the round-trip is bit-exact.

    ``mj_step`` leaves ``data.xpos`` one substep behind ``data.qpos``; every frame in these probes is
    taken after an explicit ``mj_forward`` so that the picture is a function of the recorded state and
    not of the integrator's phase. The assert catches the one thing ``mj_forward`` legitimately
    changes -- a free joint's quaternion is renormalised in place.
    """
    import mujoco
    q = np.asarray(q, np.float64)
    d.qpos[:] = q
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    err = float(np.abs(np.asarray(d.qpos) - q).max())
    assert err < 1e-9, f"qpos round-trip {err:.3e}"


def mj_render(m, d, cam, size, qpos_batch, tag="ref"):
    r = renderer(m, size, tag)
    out = []
    for q in np.asarray(qpos_batch, np.float64):
        set_state(m, d, q)
        r.update_scene(d, camera=int(cam))
        out.append(np.ascontiguousarray(r.render()))
    return np.stack(out)


def mj_segment(m, d, cam, size, qpos_batch, tag="seg"):
    r = renderer(m, size, tag)
    out = []
    r.enable_segmentation_rendering()
    try:
        for q in np.asarray(qpos_batch, np.float64):
            set_state(m, d, q)
            r.update_scene(d, camera=int(cam))
            out.append(np.ascontiguousarray(r.render())[..., :2].astype(np.int32))
    finally:
        r.disable_segmentation_rendering()
    return np.stack(out)


# --------------------------------------------------------------------------- MJWarp side

class MjwRenderer:
    """MJWarp with every correction this repo knows about, and ``com_pos`` in the frame loop."""

    def __init__(self, m, cam, res, nworld, shadows=True, textures=True, skybox=True,
                 specular=True, prepare=True, capture=False):
        import mujoco
        import torch
        import warp as wp

        import mujoco_warp as mjw
        from stable_envs.render import mjwarp as MW

        self.prepared = MW.prepare_model(m, VISIBLE_GROUPS) if prepare else None
        self.texture_gaps = MW.texture_gaps(m, VISIBLE_GROUPS)
        self.needs_com = MW.needs_com_pos(m)
        self.m_mj, self.res, self.nworld = m, int(res), int(nworld)
        # A render context never runs a collision pass, but ``put_model`` validates one anyway and
        # refuses a non-zero ``geom_margin`` on BOX/MESH under MULTICCD.
        keep_margin = np.array(m.geom_margin, copy=True)
        keep_pair = np.array(m.pair_margin, copy=True)
        m.geom_margin[:] = 0.0
        m.pair_margin[:] = 0.0
        try:
            d0 = mujoco.MjData(m)
            mujoco.mj_forward(m, d0)
            self.wm = mjw.put_model(m)
            self.wd = mjw.put_data(m, d0, nworld=int(nworld))
        finally:
            m.geom_margin[:] = keep_margin
            m.pair_margin[:] = keep_pair
        self.wp, self.mjw, self.torch = wp, mjw, torch
        with MW.gl_rounding():
            self.rc = MW.create_render_context(
                m, nworld=int(nworld), cam_res=(int(res), int(res)), render_rgb=True,
                render_seg=True, cam_active=[i == int(cam) for i in range(m.ncam)],
                enabled_geom_groups=list(VISIBLE_GROUPS), use_textures=textures,
                use_shadows=shadows, render_skybox=skybox, enable_specular=specular)
            self._frame()
            wp.synchronize()
        self.bgra = wp.to_torch(self.rc.rgb_data).view(torch.uint8).reshape(
            int(nworld), int(res), int(res), 4)
        self.seg = wp.to_torch(self.rc.seg_data).reshape(int(nworld), int(res), int(res), 2)
        self.graph = None
        if capture:
            with wp.ScopedCapture() as capt:
                self._frame()
            self.graph = capt.graph

    def _frame(self, com=True):
        mjw = self.mjw
        mjw.kinematics(self.wm, self.wd)
        if com and self.needs_com:
            mjw.com_pos(self.wm, self.wd)
        mjw.camlight(self.wm, self.wd)
        mjw.refit_bvh(self.wm, self.wd, self.rc)
        mjw.render(self.wm, self.wd, self.rc)

    def upload(self, qpos_batch):
        q = np.asarray(qpos_batch, np.float32)
        n = q.shape[0]
        assert n <= self.nworld, f"{n} states into {self.nworld} worlds"
        qt = self.wp.to_torch(self.wd.qpos)
        qt[:n] = self.torch.as_tensor(q, device=qt.device, dtype=qt.dtype)
        if n < self.nworld:
            qt[n:] = qt[n - 1:n]
        self.torch.cuda.synchronize()
        return n

    def render(self, qpos_batch, com=True):
        n = self.upload(qpos_batch)
        if self.graph is not None and com:
            self.wp.capture_launch(self.graph)
        else:
            self._frame(com=com)
        self.wp.synchronize()
        return self.bgra[:n][..., [2, 1, 0]].contiguous().cpu().numpy()

    def cam_xpos(self):
        return self.wp.to_torch(self.wd.cam_xpos).cpu().numpy()


def resolved_model(path, size=256):
    """Recompile the MJCF with one material per geom carrying MuJoCo's own RESOLVED colour.

    MJWarp reads a geom's colour from its material whenever it has one; MuJoCo prefers the geom's own
    ``rgba`` unless that is the compiler default. H1's ``black``/``white`` materials are worn by geoms
    that also set ``rgba``, so uncorrected the robot comes out one colour.
    """
    import mujoco
    spec = mujoco.MjSpec.from_file(path)
    base = spec.compile()
    for i in range(base.ngeom):
        spec.add_material(name=f"se:mat{i}")
    m = spec.compile()
    assert m.ngeom == base.ngeom and m.nq == base.nq, "surgery changed the model"
    mat0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "se:mat0")
    for g in range(m.ngeom):
        k = mat0 + g
        old = int(base.geom_matid[g])
        rgba = np.asarray(base.geom_rgba[g], np.float64)
        if old >= 0 and np.allclose(rgba, DEFAULT_RGBA):
            rgba = np.asarray(base.mat_rgba[old], np.float64)
        if old >= 0:
            m.mat_specular[k] = base.mat_specular[old]
            m.mat_shininess[k] = base.mat_shininess[old]
            m.mat_emission[k] = base.mat_emission[old]
            m.mat_reflectance[k] = base.mat_reflectance[old]
            m.mat_texid[k] = base.mat_texid[old]
            m.mat_texrepeat[k] = base.mat_texrepeat[old]
            m.mat_texuniform[k] = base.mat_texuniform[old]
        else:
            m.mat_specular[k] = 0.5
            m.mat_shininess[k] = 0.5
            m.mat_emission[k] = 0.0
            m.mat_reflectance[k] = 0.0
            m.mat_texid[k] = -1
        m.mat_rgba[k] = rgba
        m.geom_rgba[g] = rgba
        m.geom_matid[g] = k
    m.vis.quality.offsamples = int(base.vis.quality.offsamples)
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), 4 * int(size))
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), 4 * int(size))
    return m, mat0


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
        res["interior_exact"] = res["interior_mean"] = res["interior_within8"] = None
    return res


def sample_qpos(task, robot, n, seed=0, steps=60, size=64):
    """The env's own reset distribution plus a random-policy rollout.

    Seeds BOTH generators: ``gymnasium``'s (for the ``randomness`` jitter on ``qpos``) and the
    process-global ``np.random`` that ``Task.reset_model`` draws its per-episode placement from.
    """
    import gymnasium as gym

    import humanoid_bench  # noqa: F401
    mj311_shim()
    np.random.seed(int(seed))
    env = gym.make(f"{robot}-{task}-v0", render_mode=None, width=size, height=size)
    env.action_space.seed(int(seed))
    u = env.unwrapped
    rng = np.random.default_rng(int(seed))
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
