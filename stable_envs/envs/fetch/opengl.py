"""OpenGL backend for the Fetch render-only model, on CPU (software GL) or GPU.

Uses the same compact pose and appearance contract as MJWarp. Unlike MJWarp, it draws the
translucent mocap markers where Gymnasium Robotics draws them. Frames are drawn at the requested
size with the model's multisampling, as Gymnasium Robotics draws them at that size; supersampling
draws at a multiple of the size and averages. On Linux, large CPU batches draw in render worker
processes, each with its own OpenGL context.
"""

import copy
import threading
from collections import OrderedDict

import numpy as np
import torch

from ... import tuning
from ..._core.mujoco_gl import mujoco_renderer
from ...render._resources import retain_process_lifetime

PATH = "fetch-opengl"
_LOCK = threading.RLock()
_CACHE = {}
_SKIES = {}


class _Readback:
    """Collect flipped frames in a bounded OpenGL buffer, then read each chunk once into ``images``."""

    def __init__(self, renderer, images: np.ndarray):
        import mujoco
        from OpenGL import GL

        if len(images) < 1:
            raise ValueError("Cannot render an empty Fetch batch")
        self.renderer, self.images = renderer, images
        size = renderer.width
        renderer._gl_context.make_current()
        self.capacity = min(len(images), max(1, int(GL.glGetIntegerv(GL.GL_MAX_RENDERBUFFER_SIZE)) // size))
        context = renderer._mjr_context
        for index, height in ((0, size), (1, size * self.capacity)):
            if context.auxWidth[index] != size or context.auxHeight[index] != height:
                mujoco.mjr_addAux(index, size, height, 0, context)
        mujoco.mjr_restoreBuffer(context)

    def render(self, index: int) -> None:
        import mujoco
        from OpenGL import GL

        renderer = self.renderer
        context = renderer._mjr_context
        size = renderer.width
        slot = index % self.capacity
        mujoco.mjr_render(renderer._rect, renderer.scene, context)
        try:
            # Resolve multisampling as mjr_readPixels does, then copy the frame flipped into its slot.
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, int(context.auxFBO[0]))
            GL.glBlitFramebuffer(0, 0, size, size, 0, 0, size, size, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, int(context.auxFBO[0]))
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, int(context.auxFBO[1]))
            GL.glBlitFramebuffer(
                0, size, size, 0, 0, slot * size, size, (slot + 1) * size, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST
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


class _SkyTextures:
    """Skybox textures of one OpenGL context, keyed by their texel bytes, so each look uploads once.

    The context's own texture keeps the registered sky. The others are kept least recently used first.
    """

    def __init__(self, context, slot: int):
        self.context, self.slot = context, slot
        self.default = int(context.texture[slot])
        self.textures = OrderedDict()

    def bind(self, model, key) -> None:
        """Point the context's skybox at the texture of ``key`` (None: the registered sky)."""
        import mujoco
        from OpenGL import GL

        tex = self.default if key is None else self.textures.get(key)
        if tex is None:
            tex = int(GL.glGenTextures(1))
            self.context.texture[self.slot] = tex
            mujoco.mjr_uploadTexture(model, self.context, self.slot)
            self.textures[key] = tex
        elif key is not None:
            self.textures.move_to_end(key)
        self.context.texture[self.slot] = tex

    def trim(self, capacity: int) -> None:
        from OpenGL import GL

        while len(self.textures) > capacity:
            GL.glDeleteTextures([self.textures.popitem(last=False)[1]])


class _Drawer:
    """Picklable factory of a render worker's drawer: ``draw(args, frames)`` with this process's OpenGL."""

    def __call__(self):
        def draw_rows(args, frames):
            scene, rows, values, ss = args
            draw(scene, rows, values, ss, "cpu", frames)

        return draw_rows


def render(env, states=None, size=None, supersample=None, variation=None, style=None, out=None):
    env._check_open("render")
    from ..._execution import render_pool

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
    host = None if out is None or env.device.type != "cpu" else out.view(-1, size, size, 3).numpy()
    images = None
    if env.device.type == "cpu":
        tuning.on_render(PATH, size, len(rows), env)
        workers = render_pool.pool(
            PATH, _Drawer(), len(rows), PATH, size, software_gl=True, name=f"{type(env).__name__} OpenGL"
        )
        if workers is not None:
            with workers.lock:
                images = workers.render(
                    len(rows),
                    (size, size, 3),
                    lambda part: (env.SCENE, rows[part], {k: v[part] for k, v in values.items()}, ss),
                    host,
                )
    if images is None:
        images = np.empty((len(rows), size, size, 3), np.uint8) if host is None else host
        draw(env.SCENE, rows, values, ss, env.device.type, images)
    if host is not None:
        return out
    result = torch.from_numpy(images).to(env.device).reshape(shape)
    return result if out is None else out.copy_(result)


def draw(scene, rows, values, ss, device_type, images):
    """Draw compact ``rows`` with appearance ``values`` into ``(B, size, size, 3)`` uint8 ``images`` in place."""
    import mujoco
    from gymnasium_robotics.utils import mujoco_utils
    from OpenGL import GL

    from . import envs as fetch

    size = images.shape[1]
    raster_size = size * ss
    key = (scene, raster_size)
    with _LOCK:
        b = fetch._build_bundle(scene)
        if key not in _CACHE:
            model = copy.copy(b.model)
            fetch._restore(model, b.pristine)
            markers, groups = b.info["markers"]
            model.geom_group[markers] = groups
            # An offscreen buffer of the frame's size, as Gymnasium Robotics allocates for its renderer.
            model.vis.global_.offwidth = model.vis.global_.offheight = raster_size
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
            _SKIES[key] = _SkyTextures(renderer._mjr_context, b.info["sky"])
        model, data, renderer, vendor, gl_device = _CACHE[key]
        skies = _SKIES[key]
        software = any(s in gl_device.lower() for s in ("llvmpipe", "softpipe", "swrast", "software rasterizer"))
        if device_type == "cpu" and not software:
            raise RuntimeError(f"CPU rendering requires software OpenGL, found {vendor}: {gl_device}")
        if device_type == "cuda" and (software or "nvidia" not in vendor.lower()):
            raise RuntimeError(f"CUDA rendering requires NVIDIA OpenGL, found {vendor}: {gl_device}")
        readback = _Readback(renderer, images) if ss == 1 else None
        sky = None
        for i, row in enumerate(rows):
            fetch._restore(model, b.pristine)
            want = None
            if "sky.color" in values and fetch._flat_sky(model, values["sky.color"][i]):
                want = fetch._sky_texels(model)[0].tobytes()
            if want != sky:
                renderer._gl_context.make_current()
                skies.bind(model, want)
                sky = want
            if "rendering.transparent_arm" in values and values["rendering.transparent_arm"][i, 0] > 0.5:
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
                cam["distance"] * values["camera.distance_scale"][i, 0],
                cam["azimuth"] + values["camera.azimuth_delta"][i, 0],
                cam["elevation"] + values["camera.elevation_delta"][i, 0],
            )
            model.cam_pos[b.info["cam"]], model.cam_quat[b.info["cam"]] = pos, quat
            data.qpos[:] = row[: b.nq]
            data.mocap_pos[b.info["goal_mocap"]] = row[b.nq : b.nq + 3]
            mujoco.mj_forward(model, data)
            # Gymnasium Robotics puts the mocap markers on the gripper before it draws.
            mujoco_utils.reset_mocap2body_xpos(model, data)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=b.info["cam"])
            if readback is not None:
                readback.render(i)
            else:
                image = renderer.render()
                images[i] = np.rint(image.reshape(size, ss, size, ss, 3).mean((1, 3))).astype(np.uint8)
        renderer._gl_context.make_current()
        if sky is not None:
            fetch._restore(model, b.pristine)
            skies.bind(model, None)
        skies.trim(max(len(rows), 64))
