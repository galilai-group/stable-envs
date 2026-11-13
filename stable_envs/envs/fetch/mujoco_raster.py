"""Fetch configuration of the MuJoCo rasterizer (``backend="raster"``).

The scene is the render model of the OpenGL backend with the translucent markers in their drawn groups: its
geoms, materials, lights, skybox and camera, read by ``render.mujoco_scene`` (MuJoCo 3.11). Per frame, MuJoCo's
kinematics gives the geom poses (with the mocap welds that ``reset_mocap2body_xpos`` follows), the free camera
gives the view, and the look leaves give the colours: ``floor.color``, ``table.color``, ``object.color``,
``arm.color``, ``goal.color``, ``rendering.transparent_arm``, ``light.diffuse`` (model lights) and ``sky.color``
(a flat sky away from its default). ``render.mujoco_raster`` draws the frames on CPU.
"""

from __future__ import annotations

import copy
import threading

import numpy as np
import torch

from ...render import mujoco_raster as core
from ...render import mujoco_scene as ms

_LOCK = threading.RLock()
_CONFIGS: dict = {}


class _Config:
    """Packed scene of one Fetch scene, its kinematics model and the drawn index of each look slot."""


def _build(scene):
    import mujoco

    from . import envs as fetch

    b = fetch._build_bundle(scene)
    m = copy.copy(b.model)
    fetch._restore(m, b.pristine)
    markers, groups = b.info["markers"]
    m.geom_group[markers] = groups
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    c = _Config()
    c.bundle, c.model, c.data = b, m, d
    c.scene = ms.pack(ms.extract(m, d, int(b.info["cam"])))
    index = {int(g): i for i, g in enumerate(c.scene.objid)}
    c.slot = lambda geoms: [index[int(g)] for g in geoms if int(g) >= 0 and int(g) in index]
    c.welds = []
    for e in range(m.neq):
        if int(m.eq_type[e]) != int(mujoco.mjtEq.mjEQ_WELD):
            continue
        a, o = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
        c.welds.append((int(m.body_mocapid[a]), o) if int(m.body_mocapid[a]) >= 0 else (int(m.body_mocapid[o]), a))
    c.sky_default = fetch.sky_color(m)
    c.light_diffuse = np.asarray(b.pristine["light_diffuse"], np.float64)
    return c


def config(scene):
    with _LOCK:
        if scene not in _CONFIGS:
            _CONFIGS[scene] = _build(scene)
        return _CONFIGS[scene]


def _quat_mat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def frames(scene, rows, values):
    """Per-frame inputs of ``render.mujoco_raster`` for compact Fetch ``rows`` with look ``values``."""
    import mujoco

    from . import envs as fetch

    c = config(scene)
    p, b, info = c.scene, c.bundle, c.bundle.info
    B, G = len(rows), len(p.objid)
    gpos = np.empty((B, G, 3))
    gmat = np.empty((B, G, 9))
    cams = np.empty((B, 12))
    with _LOCK:
        m, d = c.model, c.data
        nq, goal = b.nq, info["goal_mocap"]
        for i in range(B):
            d.qpos[:] = rows[i, :nq]
            d.mocap_pos[goal] = rows[i, nq : nq + 3]
            mujoco.mj_kinematics(m, d)
            for mocap, body in c.welds:
                d.mocap_pos[mocap] = d.xpos[body]
                d.mocap_quat[mocap] = d.xquat[body]
            if c.welds:
                mujoco.mj_kinematics(m, d)
            gpos[i] = d.geom_xpos[p.objid]
            gmat[i] = d.geom_xmat[p.objid]
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
        inv = inv.reshape(-1)
        for u, (ds, az, el) in enumerate(uniq):
            pos, quat = fetch._free_cam_pose(
                m, d, cd["lookat"], cd["distance"] * ds, cd["azimuth"] + az, cd["elevation"] + el
            )
            r = _quat_mat(np.asarray(quat, np.float64))
            cams[inv == u] = np.concatenate([pos, r[:, 0], r[:, 1], -r[:, 2]])
    rgba = np.repeat(p.rgba[None], B, 0)
    if "rendering.transparent_arm" in values:
        flag = values["rendering.transparent_arm"][:, 0] > 0.5
        rgba[np.ix_(flag, c.slot(info["robot"]), [3])] = fetch.ARM_ALPHA
    for name, geoms in (
        ("floor", [info["floor"]]),
        ("table", [info["table"]]),
        ("object", [info["object"]]),
        ("arm", info["arm"]),
        ("goal", [info["target"]]),
    ):
        idx = c.slot(geoms)
        if name + ".color" in values and idx:
            rgba[:, idx, :3] = values[name + ".color"][:, None, :3]
    lights = ms.headlight_rows(p.lights, cams)
    for k, light in enumerate(p.light_ids):
        if light >= 0:
            lights[:, k, 8:11] = c.light_diffuse[light] * values["light.diffuse"][:, :1]
    sky = np.full((B, 3), -1.0)
    if "sky.color" in values:
        for i, color in enumerate(values["sky.color"]):
            color = np.asarray(color, np.float64).reshape(3)
            if not np.array_equal(np.round(color, 6), np.round(c.sky_default, 6)):
                sky[i] = (np.clip(color, 0.0, 1.0) * 255.0).astype(np.uint8)
    return gpos, gmat, rgba, cams, lights, sky


def render(env, states=None, size=None, supersample=None, variation=None, style=None, out=None):
    """Fetch frames of compact ``states`` (the live lanes by default) on CPU."""
    env._check_open("render")
    if env.device.type != "cpu":
        raise ValueError("backend='raster' draws Fetch frames on CPU only; on CUDA use backend='auto'")
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
    inputs = frames(env.SCENE, rows, values)
    scene = config(env.SCENE).scene
    host = None if out is None else out.view(-1, size, size, 3).numpy()
    images = np.empty((len(rows), size, size, 3), np.uint8) if host is None else host
    core.draw(scene, *inputs, ss, images)
    if host is not None:
        return out
    return torch.from_numpy(images).reshape(shape)


def semantic(env, states):
    """Segmentation and planar depth of compact ``states`` at ``env.size``.

    Segmentation is ``(B, H, W, 2)`` ``(objid, objtype)`` pairs in the model of ``reference_env`` (the goal geom
    is the site ``target0``); depth is ``(B, H, W)``.
    """
    import mujoco

    from ...testing.conformance import reference_env

    rows = states.detach().cpu().numpy() if hasattr(states, "detach") else np.asarray(states)
    rows = np.asarray(rows, np.float64).reshape(-1, env.state_dim)
    style = env.resolve_style(None, None, batch=len(rows))
    values = {k: style[k].detach().cpu().numpy().reshape(len(rows), -1) for k in env.variation_space.names()}
    c = config(env.SCENE)
    gpos, gmat, rgba, cams, _, _ = frames(env.SCENE, rows, values)
    index, depth = core.semantic(c.scene, gpos, gmat, rgba, cams, int(env.size))
    keys = getattr(c, "reference_keys", None)
    if keys is None:
        ref = reference_env(type(env), int(env.size)).unwrapped.model
        site = int(mujoco.mj_name2id(ref, mujoco.mjtObj.mjOBJ_SITE, "target0"))
        extra = {int(c.bundle.info["target"]): (site, ms.OBJ_SITE)}
        keys = c.reference_keys = ms.geom_keys(c.model, ref, extra)
    return ms.segmentation(index, c.scene.objid, keys), depth
