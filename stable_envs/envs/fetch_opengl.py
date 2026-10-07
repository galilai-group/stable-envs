"""Explicit CPU/GPU OpenGL backend for the existing Fetch render-only model.

Uses the same compact pose and appearance contract as MJWarp. The four hidden
translucent mocap markers remain a declared limitation of that model.
Match the source's 480px raster before resizing; supersampling adds density there.
"""

import copy
import threading

import numpy as np
import torch

from ..render._resources import retain_process_lifetime

_LOCK = threading.RLock()
_CACHE = {}


def render(env, states=None, size=None, supersample=None, variation=None, style=None, out=None):
    env._check_open("render")
    import cv2
    import mujoco
    from OpenGL import GL

    from . import fetch

    rows = torch.as_tensor(env._render_state if states is None else states, device=env.device)
    lead = tuple(rows.shape[:-1])
    rows = rows.reshape(-1, env.state_dim).detach().cpu().numpy()
    size, ss = int(size or env.size), int(env.ss if supersample is None else supersample)
    if size < 1 or not 1 <= ss <= 4:
        raise ValueError("Invalid Fetch render size/supersample")
    style = env.resolve_style(style, variation, batch=len(rows))
    values = {k: style[k].detach().cpu().numpy() for k in env.variation_space.names()}
    native_size = 480
    raster_size = native_size * ss
    key = (env.SCENE, raster_size)
    images = []
    with _LOCK:
        b = fetch._build_bundle(env.SCENE)
        if key not in _CACHE:
            model = copy.copy(b.model)
            fetch._restore(model, b.pristine)
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, raster_size)
            model.vis.global_.offheight = max(model.vis.global_.offheight, raster_size)
            data = mujoco.MjData(model)
            renderer = mujoco.Renderer(model, height=raster_size, width=raster_size)
            renderer._gl_context.make_current()
            entry = (
                model,
                data,
                renderer,
                GL.glGetString(GL.GL_VENDOR).decode(),
                GL.glGetString(GL.GL_RENDERER).decode(),
            )
            retain_process_lifetime(entry, "Fetch fidelity OpenGL context must remain alive with sibling contexts")
            _CACHE[key] = entry
        model, data, renderer, vendor, gl_device = _CACHE[key]
        software = any(s in gl_device.lower() for s in ("llvmpipe", "softpipe", "swrast", "software rasterizer"))
        if env.device.type == "cpu" and not software:
            raise RuntimeError(f"CPU rendering requires software OpenGL, found {vendor}: {gl_device}")
        if env.device.type == "cuda" and (software or "nvidia" not in vendor.lower()):
            raise RuntimeError(f"CUDA rendering requires NVIDIA OpenGL, found {vendor}: {gl_device}")
        for i, row in enumerate(rows):
            fetch._restore(model, b.pristine)
            for name, geoms in (
                ("floor", [b.info["floor"]]),
                ("table", [b.info["table"]]),
                ("object", [b.info["object"]]),
                ("arm", b.info["arm"]),
                ("goal", [b.info["target"]]),
            ):
                if name + ".color" not in values:
                    continue
                for g in geoms:
                    if g >= 0:
                        model.geom_rgba[g, :3] = values[name + ".color"][i]
                        model.mat_rgba[b.info["mat0"] + g, :3] = values[name + ".color"][i]
            model.light_diffuse[:] = b.pristine["light_diffuse"] * values["light.diffuse"][i]
            cam = b.cam_defaults
            pos, quat = fetch._free_cam_pose(
                model,
                data,
                cam["lookat"],
                cam["distance"] * values["camera.distance_scale"].reshape(len(rows), -1)[i, 0],
                cam["azimuth"] + values["camera.azimuth_delta"].reshape(len(rows), -1)[i, 0],
                cam["elevation"] + values["camera.elevation_delta"].reshape(len(rows), -1)[i, 0],
            )
            model.cam_pos[b.info["cam"]], model.cam_quat[b.info["cam"]] = pos, quat
            data.qpos[:] = row[: b.nq]
            data.mocap_pos[b.info["goal_mocap"]] = row[b.nq : b.nq + 3]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=b.info["cam"])
            image = renderer.render().copy()
            if ss > 1:
                image = np.rint(image.reshape(native_size, ss, native_size, ss, 3).mean((1, 3))).astype(np.uint8)
            images.append(cv2.resize(image, (size, size)))
    result = torch.as_tensor(np.stack(images), device=env.device).reshape(*lead, size, size, 3)
    if out is not None:
        if (
            out.shape != result.shape
            or out.device != result.device
            or out.dtype != result.dtype
            or not out.is_contiguous()
        ):
            raise ValueError("Invalid Fetch render output buffer")
        out.copy_(result)
        return out
    return result
