import copy
import threading

import numpy as np
import torch

from ..._core.mujoco_gl import mujoco_renderer
from ..._core.plate_texels import substitute, texel_key
from ...render._resources import retain_process_lifetime
from .simulation.physics import tilt_quat

_LOCK = threading.RLock()
_CACHE = {}


def render(env, states, style, size, out):
    env._check_open("render")
    import mujoco
    from OpenGL import GL

    key = (env.DOMAIN, env.TASK, size)
    with _LOCK:
        if key not in _CACHE:
            model = copy.copy(env._model())
            pristine = env._pristine()
            for name, value in pristine.items():
                getattr(model, name)[:] = value
            model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), size)
            model.vis.global_.offheight = max(int(model.vis.global_.offheight), size)
            data = mujoco.MjData(model)
            renderer = mujoco_renderer(model, height=size, width=size)
            renderer._gl_context.make_current()
            vendor = GL.glGetString(GL.GL_VENDOR).decode()
            gl_device = GL.glGetString(GL.GL_RENDERER).decode()
            entry = [model, data, renderer, vendor, gl_device, None]
            retain_process_lifetime(entry, "Trackcom fidelity OpenGL context must remain alive with sibling contexts")
            _CACHE[key] = entry
        entry = _CACHE[key]
        model, data, renderer, vendor, gl_device = entry[:5]
        software = any(s in gl_device.lower() for s in ("llvmpipe", "softpipe", "swrast", "software rasterizer"))
        if env.device.type == "cpu" and not software:
            raise RuntimeError(f"CPU fidelity rendering requires software OpenGL, found {vendor}: {gl_device}")
        if env.device.type == "cuda" and (software or "nvidia" not in vendor.lower()):
            raise RuntimeError(f"CUDA fidelity rendering requires NVIDIA OpenGL, found {vendor}: {gl_device}")
        pr = env._pristine()
        rows = states.detach().cpu().numpy()
        colors = {name: style[f"material.{name}"].detach().cpu().numpy() for name in env._used_materials()}
        lights = style["light.diffuse_scale"].detach().cpu().numpy()
        masses = env._row_masses(style, 0, len(rows))
        masses = {name: value.detach().cpu().numpy() for name, value in (masses or {}).items()}
        slots, tilt = env._floor_slots(), env._floor_tilt()
        floors = None if slots is None else style["floor.color"].detach().cpu().numpy().astype(np.float64)
        angles = None if tilt is None else style[tilt[0]].detach().cpu().numpy().astype(np.float64).reshape(-1)
        level = (
            None if tilt is None else (env._model().geom_quat[tilt[1]].copy(), env._model().geom_sameframe[tilt[1]])
        )
        for i, row in enumerate(rows):
            for name, value in pr.items():
                getattr(model, name)[:] = value
            for name, values in colors.items():
                model.material(name).rgba[:3] = values[i]
            model.light_diffuse[:] = np.clip(pr["light_diffuse"] * lights[i], 0, 1)
            for name, value in masses.items():
                getattr(model, name)[:] = value[i]
            for (name, k), value in zip(env._slots(), row[env.nq :]):
                getattr(model, name).reshape(-1)[k] = value
            if floors is not None:
                colours = floors[i].reshape(2, 3)
                # entry[5] keys the colors this context's texture holds; None is the compiled default.
                if texel_key(slots, colours) != entry[5]:
                    substitute(model, renderer, slots, colours)
                    entry[5] = texel_key(slots, colours)
            if angles is not None:
                quat = tilt_quat(angles[i])
                model.geom_quat[tilt[1]] = quat
                model.geom_sameframe[tilt[1]] = level[1] if np.array_equal(quat, level[0]) else 0
            data.qpos[:] = row[: env.nq]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=env.CAMERA)
            out[i].copy_(torch.as_tensor(renderer.render().copy(), device=env.device))
