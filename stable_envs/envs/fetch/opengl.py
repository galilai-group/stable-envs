"""Explicit CPU/GPU OpenGL backend for the existing Fetch render-only model.

Uses the same compact pose and appearance contract as MJWarp. The four hidden
translucent mocap markers remain a declared limitation of that model.
Match the source's 480px raster before resizing; supersampling adds density there.
"""

import copy
import threading

import numpy as np
import torch

from ..._core.mujoco_gl import mujoco_renderer
from ...render._resources import retain_process_lifetime

_LOCK = threading.RLock()
_CACHE = {}


class _Readback:
    """Collect resized frames in a bounded OpenGL buffer, then read each chunk once."""

    def __init__(self, renderer, size: int, batch: int):
        import mujoco
        from OpenGL import GL

        if batch < 1:
            raise ValueError("Cannot render an empty Fetch batch")
        self.renderer, self.size = renderer, size
        self.images = np.empty((batch, size, size, 3), dtype=np.uint8)
        renderer._gl_context.make_current()
        self.capacity = min(batch, max(1, int(GL.glGetIntegerv(GL.GL_MAX_RENDERBUFFER_SIZE)) // size))
        context = renderer._mjr_context
        for index, width, height in ((0, renderer.width, renderer.height), (1, size, size * self.capacity)):
            if context.auxWidth[index] != width or context.auxHeight[index] != height:
                mujoco.mjr_addAux(index, width, height, 0, context)
        mujoco.mjr_restoreBuffer(context)

    def render(self, index: int) -> None:
        import mujoco
        from OpenGL import GL

        renderer, size = self.renderer, self.size
        context = renderer._mjr_context
        width, height = renderer.width, renderer.height
        slot = index % self.capacity
        mujoco.mjr_render(renderer._rect, renderer.scene, context)
        try:
            # Resolve multisampling at native size before resizing and flipping each frame.
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, int(context.auxFBO[0]))
            GL.glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, int(context.auxFBO[0]))
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, int(context.auxFBO[1]))
            GL.glBlitFramebuffer(
                0, height, width, 0, 0, slot * size, size, (slot + 1) * size, GL.GL_COLOR_BUFFER_BIT, GL.GL_LINEAR
            )
            if slot + 1 == self.capacity or index + 1 == len(self.images):
                GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, int(context.auxFBO[1]))
                GL.glReadPixels(
                    0,
                    0,
                    size,
                    (slot + 1) * size,
                    GL.GL_RGB,
                    GL.GL_UNSIGNED_BYTE,
                    self.images[index - slot : index + 1],
                )
        finally:
            mujoco.mjr_restoreBuffer(context)


def render(env, states=None, size=None, supersample=None, variation=None, style=None, out=None):
    env._check_open("render")
    import cv2
    import mujoco
    from OpenGL import GL

    from . import envs as fetch

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
            renderer = mujoco_renderer(model, height=raster_size, width=raster_size)
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
        readback = _Readback(renderer, size, len(rows)) if ss == 1 else None
        sky = None
        for i, row in enumerate(rows):
            fetch._restore(model, b.pristine)
            want = None
            if "sky.color" in values and fetch._flat_sky(model, values["sky.color"][i]):
                want = tuple(np.round(values["sky.color"][i].astype(np.float64), 6).tolist())
            if want != sky:
                renderer._gl_context.make_current()
                mujoco.mjr_uploadTexture(model, renderer._mjr_context, b.info["sky"])
                sky = want
            if "rendering.transparent_arm" in values and values["rendering.transparent_arm"].reshape(-1)[i] > 0.5:
                model.geom_rgba[b.info["robot"], 3] = fetch.ARM_ALPHA
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
            if readback is not None:
                readback.render(i)
            else:
                image = renderer.render()
                image = np.rint(image.reshape(native_size, ss, native_size, ss, 3).mean((1, 3))).astype(np.uint8)
                images.append(cv2.resize(image, (size, size)))
        if sky is not None:
            fetch._restore(model, b.pristine)
            renderer._gl_context.make_current()
            mujoco.mjr_uploadTexture(model, renderer._mjr_context, b.info["sky"])
    images = readback.images if readback is not None else np.stack(images)
    result = torch.as_tensor(images, device=env.device).reshape(*lead, size, size, 3)
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
