"""Cube and Scene configuration of the MuJoCo rasterizer (``backend="raster"``).

A configuration is the private render model of the OpenGL backend for one value of the compiled look leaves
(``cube.size``, ``floor.color``, ``camera.angle_delta``, ``light.intensity``): its geoms, the ``front_pixels``
camera, the lights, the shadow light and the floor texture, read by ``render.mujoco_scene``. The arm meshes have
one vertex per ``ARM_CELL`` grid cell. Per row, MuJoCo's kinematics gives the geom poses and the source's colour
rules give the geom colours of ``cube.color``, ``agent.color``, ``lock_color`` and the Scene buttons, with the
see-through arm and the hidden goal cube. ``render.mujoco_raster`` draws the frames on CPU.
"""

from __future__ import annotations

import copy
import hashlib
import threading

import numpy as np
import torch

from ...render import mujoco_raster as core
from ...render import mujoco_scene as ms
from ...render._resources import BoundedCache
from .simulation import render_state as pixels
from .simulation.compatibility import MODEL_VALUES, CompatibleCube, CompatibleScene

_LOCK = threading.RLock()
_RIGS = BoundedCache(32, name="ogbench.raster_rig", on_evict=lambda rig: rig.env.close())
#: Arm meshes are drawn, and cast shadows, simplified on grids of these many metres (0 keeps them exact).
ARM_CELL = (0.002, 0.002)


def _key(family, columns, index, cell):
    h = hashlib.sha256(f"{family}:{cell}".encode())
    for k in MODEL_VALUES:
        if k != "agent.color":
            h.update(np.ascontiguousarray(columns["appearance", k][index]).tobytes())
    return h.hexdigest()


class _Rig:
    """The private render model of one configuration, its packed scene and its per-row poses and colours."""

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
        cast = None
        if cast_cell != view_cell:
            if cast_cell:
                cluster_meshes(exact, cast_cell)
            cast = ms.extract(exact, d, "front_pixels")
        self.scene = ms.pack(s, cast)
        self.objid = self.scene.objid
        self.looks = {}

    def rgba(self, look):
        """Drawn geom colours for one (cube, agent, lock, buttons) look, as the OpenGL path writes them."""
        hit = self.looks.get(look)
        if hit is not None:
            return hit
        env, m = self.env, self.env.model
        cube, agent, lock, buttons = look
        env.variation_space._walk("cube.color")._value = (
            np.frombuffer(cube, np.float32).reshape(1, 3).astype(np.float64)
        )
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
        out = np.array([ms.geom_rgba(m, int(g)) for g in self.objid], np.float64)
        if len(self.looks) > 256:
            self.looks.clear()
        self.looks[look] = out
        return out

    def frames(self, columns, indices):
        """Per-frame inputs of ``render.mujoco_raster`` for rows ``indices`` of host RenderState ``columns``."""
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
        p = self.scene
        cams = np.repeat(p.cam[None], n, 0)
        lights = np.repeat(p.lights[None], n, 0)
        sky = np.full((n, 3), -1.0)
        return gpos, gmat, rgba, cams, lights, sky


def _rig(family, columns, index, cell):
    key = _key(family, columns, index, cell)
    rig = _RIGS.get(key)
    if rig is None:
        rig = _RIGS.put(key, _Rig(family, columns, index, cell))
    return rig


def _groups(family, columns, cell):
    groups = {}
    for i in range(len(columns["renderer", "size"])):
        groups.setdefault(_key(family, columns, i, cell), []).append(i)
    return groups


def draw(family, columns, images, cell=None):
    """Draw validated host RenderState ``columns`` into ``(B, size, size, 3)`` uint8 ``images`` on CPU, in place."""
    cell = ARM_CELL if cell is None else cell
    ss = int(columns["renderer", "supersample"][0, 0])
    with _LOCK:
        for indices in _groups(family, columns, cell).values():
            rig = _rig(family, columns, indices[0], cell)
            inputs = rig.frames(columns, indices)
            whole = len(indices) == len(images)
            out = images if whole else images[np.asarray(indices)]
            core.draw(rig.scene, *inputs, ss, out)
            if not whole:
                images[np.asarray(indices)] = out


def frames(family, columns, device, chunk_size=None):
    """Frames of host RenderState ``columns`` on CPU ``device``."""
    from .manipulation_opengl import check_versions

    check_versions("raster")
    if torch.device(device).type != "cpu":
        raise ValueError("backend='raster' draws Cube/Scene frames on CPU only; on CUDA use backend='auto'")
    b, size = len(columns["renderer", "size"]), int(columns["renderer", "size"][0, 0])
    images = np.empty((b, size, size, 3), np.uint8)
    draw(family, columns, images)
    return torch.from_numpy(images)


def semantic(family, columns, cls):
    """Segmentation and planar depth of host RenderState ``columns`` at the frame size.

    Segmentation is ``(B, H, W, 2)`` ``(objid, objtype)`` pairs in the model of ``cls.reference_env``; depth is
    ``(B, H, W)``.
    """
    from ...testing.conformance import reference_env

    size = int(columns["renderer", "size"][0, 0])
    b = len(columns["renderer", "size"])
    seg = np.full((b, size, size, 2), -1, np.int64)
    depth = np.zeros((b, size, size), np.float32)
    with _LOCK:
        for indices in _groups(family, columns, ARM_CELL).values():
            rig = _rig(family, columns, indices[0], ARM_CELL)
            gpos, gmat, rgba, cams, _, _ = rig.frames(columns, indices)
            index, d = core.semantic(rig.scene, gpos, gmat, rgba, cams, size)
            keys = getattr(rig, "reference_keys", None)
            if keys is None:
                ref = reference_env(cls, size).unwrapped.model
                keys = rig.reference_keys = ms.geom_keys(rig.env.model, ref)
            seg[np.asarray(indices)] = ms.segmentation(index, rig.scene.objid, keys)
            depth[np.asarray(indices)] = d
    return seg, depth
