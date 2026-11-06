from __future__ import annotations

import hashlib
import threading
from functools import lru_cache
from importlib.metadata import version

import numpy as np
import torch
from tensordict import TensorDict

from ..._core.mujoco_gl import mujoco_renderer
from ...render._resources import retain_process_lifetime
from .simulation import render_state as pixels
from .simulation.compatibility import MODEL_VALUES, CompatibleCube, CompatibleScene

# agent.color is applied per row to the agent materials, so it does not select a compiled model.
COMPILED_VALUES = tuple(name for name in MODEL_VALUES if name != "agent.color")

_LOCK = threading.RLock()
_CACHE = {}
# Renderer entries by family, resolution and packed configuration bytes, which skips hashing each call.
_ENTRIES: dict = {}
_PINNED: dict = {}
_SIZE = ("renderer", "size")
_SCRATCH: dict = {}
# The appearance last written into each private renderer model.
_APPLIED: dict = {}
_BUTTONS: dict = {}
_KINEMATICS: dict = {}
_PINNED_ROWS = 8


@lru_cache
def check_versions(renderer, *, warp=False):
    """Raise ImportError unless the pinned MuJoCo and OGBench, and with ``warp`` MJWarp and Warp, are installed."""
    if version("mujoco") != "3.17.0" or version("ogbench") != "1.2.1":
        raise ImportError(f"Cube/Scene {renderer} rendering requires mujoco==3.17.0 and ogbench==1.2.1")
    if warp and (version("mujoco-warp") != "3.17.0" or version("warp-lang") != "1.17.0"):
        raise ImportError("Cube/Scene GPU rendering requires mujoco-warp==3.17.0 and warp-lang==1.17.0")


def _key(family, row, resolution):
    h = hashlib.sha256(f"{family}:{resolution}".encode())
    for k in COMPILED_VALUES:
        h.update(row["appearance", k].numpy().tobytes())
    return h.hexdigest()


def _renderer(family, row, resolution):
    import mujoco  # noqa: F401 — selects PYOPENGL_PLATFORM before PyOpenGL loads
    from OpenGL import GL

    key = _key(family, row, resolution)
    if key not in _CACHE:
        env = (CompatibleCube if family == "Cube" else CompatibleScene)(seed=0)
        for k in MODEL_VALUES:
            env.variation_space._walk(k)._value = row["appearance", k].numpy().copy()
        env._mjcf_model = env.modify_mjcf_model(env.build_mjcf_model())
        env.compile_model_and_data()
        env.model.vis.global_.offwidth = max(int(env.model.vis.global_.offwidth), resolution)
        env.model.vis.global_.offheight = max(int(env.model.vis.global_.offheight), resolution)
        renderer = mujoco_renderer(env.model, height=resolution, width=resolution)
        renderer._gl_context.make_current()
        vendor = GL.glGetString(GL.GL_VENDOR).decode()
        device = GL.glGetString(GL.GL_RENDERER).decode()
        entry = (env, renderer, vendor, device)
        retain_process_lifetime(
            entry, "Cube/Scene fidelity OpenGL context: releasing one invalidates other live contexts"
        )
        _CACHE[key] = entry
    return _CACHE[key]


def _clear_errors():
    """Drop GL errors MuJoCo's C calls left set, which PyOpenGL would raise at its next call.

    The source's 100-degree spot cutoff sets GL_INVALID_VALUE in every ``mjr_render``.
    """
    from OpenGL import GL

    for _ in range(8):
        if GL.glGetError() == GL.GL_NO_ERROR:
            break


class _Tiles:
    """Resolve each frame as ``mjr_readPixels`` does, flip it into one tile of an auxiliary buffer, read many at once."""

    def __init__(self, renderer, images, ss, count):
        import mujoco
        from OpenGL import GL

        self.renderer, self.images, self.ss = renderer, images, ss
        self.resolution = resolution = renderer.width
        if renderer.height != resolution or images.shape[1] * ss != resolution:
            raise ValueError("Cube/Scene OpenGL tiles need square frames at the renderer resolution")
        renderer._gl_context.make_current()
        _clear_errors()
        context = renderer._mjr_context
        limit = int(GL.glGetIntegerv(GL.GL_MAX_RENDERBUFFER_SIZE)) // resolution
        self.capacity = max(1, min(count, limit, (64 << 20) // (3 * resolution * resolution)))
        height = resolution * self.capacity
        if context.auxWidth[0] != resolution or context.auxHeight[0] < height:
            mujoco.mjr_addAux(0, resolution, height, 0, context)
        mujoco.mjr_restoreBuffer(context)
        self.stage = np.empty((self.capacity, resolution, resolution, 3), np.uint8)
        self.rows = []

    def draw(self, index):
        import mujoco
        from OpenGL import GL

        renderer, size = self.renderer, self.resolution
        context = renderer._mjr_context
        renderer._gl_context.make_current()
        mujoco.mjr_render(renderer._rect, renderer.scene, context)
        _clear_errors()
        slot = len(self.rows)
        try:
            source = int(context.offFBO)
            if context.offSamples:
                GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, source)
                GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
                GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, int(context.offFBO_r))
                GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
                GL.glBlitFramebuffer(0, 0, size, size, 0, 0, size, size, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
                source = int(context.offFBO_r)
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, source)
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, int(context.auxFBO[0]))
            GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
            # Row 0 of the tile is the top image row, as Renderer.render returns it.
            GL.glBlitFramebuffer(
                0, size, size, 0, 0, slot * size, size, (slot + 1) * size, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST
            )
        finally:
            mujoco.mjr_restoreBuffer(context)
        self.rows.append(index)
        if len(self.rows) == self.capacity:
            self.flush()

    def flush(self):
        import mujoco
        from OpenGL import GL

        count = len(self.rows)
        if not count:
            return
        context = self.renderer._mjr_context
        self.renderer._gl_context.make_current()
        _clear_errors()
        try:
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, int(context.auxFBO[0]))
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
            GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
            GL.glReadPixels(
                0, 0, self.resolution, count * self.resolution, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, self.stage[:count]
            )
        finally:
            mujoco.mjr_restoreBuffer(context)
        for frame, index in zip(self.stage[:count], self.rows):
            if self.ss == 1:
                self.images[index] = frame
            else:
                box_mean(frame, self.ss, self.images[index])
        self.rows = []


def _read_one(renderer, out):
    """``Renderer.render(out=out)`` for RGB frames, without its per-call flag bookkeeping."""
    import mujoco

    scratch = _SCRATCH.get(out.shape)
    if scratch is None:
        scratch = _SCRATCH[out.shape] = np.empty(out.shape, np.uint8)
    renderer._gl_context.make_current()
    mujoco.mjr_render(renderer._rect, renderer.scene, renderer._mjr_context)
    mujoco.mjr_readPixels(scratch, None, renderer._rect, renderer._mjr_context)
    np.copyto(out, scratch[::-1])


def box_mean(image, ss, out):
    """Write the round-half-even mean of each ``ss`` x ``ss`` block, as ``np.rint`` of the float mean gives."""
    blocks = image.reshape(len(out), ss, out.shape[1], ss, 3)
    total = np.zeros(out.shape, np.uint16)
    for y in range(ss):
        for x in range(ss):
            total += blocks[:, y, :, x]
    count = ss * ss
    quotient, remainder = np.divmod(total, count)
    # Ties occur only when the count divides twice the sum exactly.
    up = (2 * remainder > count) | ((2 * remainder == count) & (quotient % 2 == 1))
    np.add(quotient, up, out=out, casting="unsafe")


def _groups(host):
    """Group exact model bytes without constructing a TensorDict for each row."""
    host = columns(host) if isinstance(host, TensorDict) else host
    b = len(host[_SIZE])
    parts = [np.ascontiguousarray(host["appearance", name]).reshape(b, -1).view(np.uint8) for name in COMPILED_VALUES]
    packed = np.concatenate(parts, axis=1)
    groups = {}
    for index, row in enumerate(packed):
        groups.setdefault(row.tobytes(), []).append(index)
    return groups


def _configuration_groups(host):
    """Row indices of each compiled configuration, in first-row order."""
    return _groups(host).values()


def _can_use_render_kinematics(env, renderer):
    """Keep the fast update restricted to the audited fixed RGB scene."""
    import mujoco

    model = env.model
    if model.nflex or model.nskin or model.nplugin or model.nsensor:
        return False
    if model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP:
        return False
    if np.any(model.cam_mode != mujoco.mjtCamLight.mjCAMLIGHT_FIXED):
        return False
    if np.any(model.light_mode != mujoco.mjtCamLight.mjCAMLIGHT_FIXED):
        return False
    if np.any(model.wrap_type != mujoco.mjtWrap.mjWRAP_JOINT):
        return False
    default = mujoco.MjvOption()
    options = renderer._scene_option
    return (
        np.array_equal(options.flags, default.flags)
        and options.frame == default.frame
        and options.label == default.label
    )


def _apply_render_button_colors(env, lock_colors):
    """Final CompatibleScene colors; no simulation update on this private model."""
    for index, state in enumerate(env._cur_button_states):
        color = lock_colors[index] if state == 0 else env._colors["white"][:3]
        for gid in env._button_geom_ids_list[index]:
            env.model.geom_rgba[gid] = [*color, 1.0]
        env.model.material(("drawer_handle", "window_handle")[index]).rgba[:] = [*color, 1.0]


def render(family, state, device, chunk_size=None, validated=False):
    """Render named states through OpenGL using candidate assets, in worker processes for large batches.

    CPU output requires software OpenGL. CUDA output uses NVIDIA OpenGL plus
    host readback and upload.
    """
    # OpenGL consumes host model inputs anyway. Validate that same detached copy
    # rather than synchronizing CUDA once for every scalar validation predicate.
    host, device, size = prepare(family, state, device, chunk_size, validated)
    return frames(family, columns(host), device)


def columns(host):
    """NumPy columns of a host RenderState, keyed like it."""
    return {key: value.numpy() for key, value in host.items(True, True)}


def frames(family, columns, device):
    """Frames of host RenderState columns on ``device``, across worker processes for large batches."""
    from . import manipulation_opengl_pool

    check_versions("OpenGL")
    device = torch.device(device)
    b, size = len(columns[_SIZE]), int(columns[_SIZE][0, 0])
    shared = manipulation_opengl_pool.draw(family, columns, device.type, size)
    if shared is not None:
        return torch.from_numpy(shared.copy()) if device.type == "cpu" else torch.from_numpy(shared).to(device)
    if device.type == "cuda" and b <= _PINNED_ROWS:
        # A reused pinned buffer uploads small batches faster than fresh pageable memory.
        with _LOCK:
            staging = _PINNED.get((b, size))
            if staging is None:
                staging = _PINNED[b, size] = torch.empty((b, size, size, 3), dtype=torch.uint8, pin_memory=True)
            draw(family, columns, device.type, staging.numpy())
            result = staging.to(device, non_blocking=True)
            torch.cuda.current_stream(device).synchronize()
            return result
    images = np.empty((b, size, size, 3), dtype=np.uint8)
    draw(family, columns, device.type, images)
    return torch.from_numpy(images) if device.type == "cpu" else torch.from_numpy(images).to(device)


def prepare(family, state, device, chunk_size=None, validated=False):
    """Validate a render State unless ``validated``; return its host copy, the torch device and the frame size."""
    host = state.cpu()
    if any(value.requires_grad for value in host.values(True, True)):
        host = host.detach()
    if not validated:
        pixels.validate(family, host)
    check_versions("OpenGL")
    if chunk_size is not None and (type(chunk_size) is not int or chunk_size < 1):
        raise ValueError("chunk_size must be a positive integer")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Cube/Scene OpenGL supports CPU software or NVIDIA GPU rendering")
    return host, device, int(host["renderer", "size"][0, 0])


class _Row:
    """One state of NumPy columns, read like a RenderState row."""

    def __init__(self, columns, index):
        self.columns, self.index = columns, index

    def __getitem__(self, key):
        return torch.from_numpy(np.array(self.columns[key][self.index]))


def draw(family, host, device_type, images):
    """Draw validated host states (NumPy columns or a RenderState) into ``images`` with this process's OpenGL."""
    import mujoco

    host = columns(host) if isinstance(host, TensorDict) else host
    size, ss = (int(host["renderer", k][0, 0]) for k in ("size", "supersample"))
    poses = {name: host["pose", name] for name in ("qpos", "mocap_pos", "mocap_quat")}
    cube_colors = host["appearance", "cube.color"]
    agent_colors = host["appearance", "agent.color"]
    if family == "Scene":
        lock_colors = host["appearance", "lock_color"]
        buttons = host["pose", "buttons"]
    high_resolution = np.empty((size * ss, size * ss, 3), dtype=np.uint8) if ss != 1 else None
    with _LOCK:
        for packed, indices in _groups(host).items():
            entry = _ENTRIES.get((family, size * ss, packed))
            if entry is None:
                entry = _ENTRIES[family, size * ss, packed] = _renderer(family, _Row(host, indices[0]), size * ss)
            env, renderer, vendor, gl_device = entry
            software = any(s in gl_device.lower() for s in ("llvmpipe", "softpipe", "swrast", "software rasterizer"))
            if device_type == "cpu" and not software:
                raise RuntimeError(
                    f"CPU rendering requires software OpenGL, found {vendor}: {gl_device}. "
                    "Launch a separate process with Mesa software rendering configured before importing MuJoCo."
                )
            if device_type == "cuda" and (software or "nvidia" not in vendor.lower()):
                raise RuntimeError(f"CUDA rendering requires NVIDIA OpenGL, found {vendor}: {gl_device}")
            if family == "Scene":
                # The private model and renderer options do not change, so the guard runs once per renderer.
                render_kinematics = _KINEMATICS.get(id(env))
                if render_kinematics is None:
                    render_kinematics = _KINEMATICS[id(env)] = _can_use_render_kinematics(env, renderer)
            else:
                render_kinematics = False
            # One frame reads back faster through MuJoCo's own readback than through PyOpenGL calls.
            tiles = (
                _Tiles(renderer, images, ss, len(indices))
                if len(indices) > 1 and hasattr(renderer, "_mjr_context")
                else None
            )
            for index in indices:
                for name, values in poses.items():
                    getattr(env.data, name)[:] = values[index]
                # Colours this loop writes stay in the private model, so an unchanged appearance is not rewritten.
                look = (cube_colors[index].tobytes(), agent_colors[index].tobytes())
                if _APPLIED.get(id(env)) != look:
                    env.variation_space._walk("cube.color")._value = cube_colors[index].copy()
                    for material in pixels.AGENT_MATERIALS:
                        env.model.material(material).rgba[:3] = agent_colors[index]
                    env._cube_colors_from_variation()
                    for ids in env._cube_target_geom_ids_list:
                        env.model.geom_rgba[ids, 3] = 0.2
                    _APPLIED[id(env)] = look
                if family == "Scene":
                    env.variation_space._walk("lock_color")._value = lock_colors[index].copy()
                    env._cur_button_states = buttons[index].copy()
                    if render_kinematics:
                        buttons_look = (lock_colors[index].tobytes(), buttons[index].tobytes())
                        if _BUTTONS.get(id(env)) != buttons_look:
                            _apply_render_button_colors(env, lock_colors[index])
                            _BUTTONS[id(env)] = buttons_look
                        mujoco.mj_kinematics(env.model, env.data)
                        mujoco.mj_camlight(env.model, env.data)
                    else:
                        # Upstream applies button physics and calls mj_forward.
                        env._apply_button_states()
                        _BUTTONS.pop(id(env), None)
                else:
                    mujoco.mj_forward(env.model, env.data)
                renderer.update_scene(env.data, camera="front_pixels")
                if tiles is not None:
                    tiles.draw(index)
                elif ss == 1 and hasattr(renderer, "_mjr_context"):
                    _read_one(renderer, images[index])
                elif ss == 1:
                    renderer.render(out=images[index])
                else:
                    renderer.render(out=high_resolution)
                    box_mean(high_resolution, ss, images[index])
            if tiles is not None:
                tiles.flush()
