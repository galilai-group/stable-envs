"""Explicit OpenGL rendering of the trackcom model, including floor reflections."""

import copy
import threading

import numpy as np
import torch

from ...render._resources import retain_process_lifetime

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
            renderer = mujoco.Renderer(model, height=size, width=size)
            renderer._gl_context.make_current()
            vendor = GL.glGetString(GL.GL_VENDOR).decode()
            gl_device = GL.glGetString(GL.GL_RENDERER).decode()
            entry = (model, data, renderer, vendor, gl_device)
            retain_process_lifetime(entry, "Trackcom fidelity OpenGL context must remain alive with sibling contexts")
            _CACHE[key] = entry
        model, data, renderer, vendor, gl_device = _CACHE[key]
        software = any(s in gl_device.lower() for s in ("llvmpipe", "softpipe", "swrast", "software rasterizer"))
        if env.device.type == "cpu" and not software:
            raise RuntimeError(f"CPU fidelity rendering requires software OpenGL, found {vendor}: {gl_device}")
        if env.device.type == "cuda" and (software or "nvidia" not in vendor.lower()):
            raise RuntimeError(f"CUDA fidelity rendering requires NVIDIA OpenGL, found {vendor}: {gl_device}")
        pr = env._pristine()
        rows = states.detach().cpu().numpy()
        colors = {name: style[f"material.{name}"].detach().cpu().numpy() for name in env._used_materials()}
        lights = style["light.diffuse_scale"].detach().cpu().numpy()
        for i, row in enumerate(rows):
            for name, value in pr.items():
                getattr(model, name)[:] = value
            for name, values in colors.items():
                model.material(name).rgba[:3] = values[i]
            model.light_diffuse[:] = np.clip(pr["light_diffuse"] * lights[i], 0, 1)
            for (name, k), value in zip(env._slots(), row[env.nq :]):
                getattr(model, name).reshape(-1)[k] = value
            data.qpos[:] = row[: env.nq]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=env.CAMERA)
            out[i].copy_(torch.as_tensor(renderer.render().copy(), device=env.device))
