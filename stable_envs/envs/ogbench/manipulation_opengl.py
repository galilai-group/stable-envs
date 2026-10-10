from __future__ import annotations

import hashlib
import threading
from functools import lru_cache
from importlib.metadata import version

import numpy as np
import torch

from ...render._resources import retain_process_lifetime
from .simulation import render_state as pixels
from .simulation.compatibility import MODEL_VALUES, CompatibleCube, CompatibleScene

# agent.color is applied per row to the agent materials, so it does not select a compiled model.
COMPILED_VALUES = tuple(name for name in MODEL_VALUES if name != "agent.color")

_LOCK = threading.RLock()
_CACHE = {}


def _key(family, row, resolution):
    h = hashlib.sha256(f"{family}:{resolution}".encode())
    for k in COMPILED_VALUES:
        h.update(row["appearance", k].numpy().tobytes())
    return h.hexdigest()


def _renderer(family, row, resolution):
    import mujoco
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
        renderer = mujoco.Renderer(env.model, height=resolution, width=resolution)
        renderer._gl_context.make_current()
        vendor = GL.glGetString(GL.GL_VENDOR).decode()
        device = GL.glGetString(GL.GL_RENDERER).decode()
        entry = (env, renderer, vendor, device)
        retain_process_lifetime(
            entry, "Cube/Scene fidelity OpenGL context: releasing one invalidates other live contexts"
        )
        _CACHE[key] = entry
    return _CACHE[key]


@lru_cache(maxsize=1)
def _check_versions():
    # Distribution metadata does not change during the lifetime of a renderer.
    if version("mujoco") != "3.12.0" or version("ogbench") != "1.2.1":
        raise ImportError("Cube/Scene OpenGL rendering requires mujoco==3.12.0 and ogbench==1.2.1")


def _configuration_groups(host):
    """Group exact model bytes without constructing a TensorDict for each row."""
    parts = [
        np.ascontiguousarray(host["appearance", name].numpy()).reshape(len(host), -1).view(np.uint8)
        for name in COMPILED_VALUES
    ]
    packed = np.concatenate(parts, axis=1)
    groups = {}
    for index, row in enumerate(packed):
        groups.setdefault(row.tobytes(), []).append(index)
    return groups.values()


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


def render(family, state, device, chunk_size=None):
    """Render named states serially through OpenGL using candidate assets.

    CPU output requires software OpenGL. CUDA output uses NVIDIA OpenGL plus
    host readback and upload; its throughput is separate from batched MJWarp."""
    import mujoco

    # OpenGL consumes host model inputs anyway. Validate that same detached copy
    # rather than synchronizing CUDA once for every scalar validation predicate.
    host = state.detach().cpu()
    pixels.validate(family, host)
    _check_versions()
    if chunk_size is not None and (type(chunk_size) is not int or chunk_size < 1):
        raise ValueError("chunk_size must be a positive integer")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Cube/Scene OpenGL supports CPU software or NVIDIA GPU rendering")
    size, ss = (int(host["renderer", k][0, 0]) for k in ("size", "supersample"))
    poses = {name: host["pose", name].numpy() for name in ("qpos", "mocap_pos", "mocap_quat")}
    cube_colors = host["appearance", "cube.color"].numpy()
    agent_colors = host["appearance", "agent.color"].numpy()
    if family == "Scene":
        lock_colors = host["appearance", "lock_color"].numpy()
        buttons = host["pose", "buttons"].numpy()
    images = np.empty((len(host), size, size, 3), dtype=np.uint8)
    high_resolution = np.empty((size * ss, size * ss, 3), dtype=np.uint8) if ss != 1 else None
    with _LOCK:
        for indices in _configuration_groups(host):
            env, renderer, vendor, gl_device = _renderer(family, host[indices[0]], size * ss)
            software = any(s in gl_device.lower() for s in ("llvmpipe", "softpipe", "swrast", "software rasterizer"))
            if device.type == "cpu" and not software:
                raise RuntimeError(
                    f"CPU rendering requires software OpenGL, found {vendor}: {gl_device}. "
                    "Launch a separate process with Mesa software rendering configured before importing MuJoCo."
                )
            if device.type == "cuda" and (software or "nvidia" not in vendor.lower()):
                raise RuntimeError(f"CUDA rendering requires NVIDIA OpenGL, found {vendor}: {gl_device}")
            render_kinematics = family == "Scene" and _can_use_render_kinematics(env, renderer)
            for index in indices:
                for name, values in poses.items():
                    getattr(env.data, name)[:] = values[index]
                env.variation_space._walk("cube.color")._value = cube_colors[index].copy()
                for material in pixels.AGENT_MATERIALS:
                    env.model.material(material).rgba[:3] = agent_colors[index]
                env._cube_colors_from_variation()
                for ids in env._cube_target_geom_ids_list:
                    env.model.geom_rgba[ids, 3] = 0.2
                if family == "Scene":
                    env.variation_space._walk("lock_color")._value = lock_colors[index].copy()
                    env._cur_button_states = buttons[index].copy()
                    if render_kinematics:
                        _apply_render_button_colors(env, lock_colors[index])
                        mujoco.mj_kinematics(env.model, env.data)
                        mujoco.mj_camlight(env.model, env.data)
                    else:
                        # Upstream applies button physics and calls mj_forward.
                        env._apply_button_states()
                else:
                    mujoco.mj_forward(env.model, env.data)
                renderer.update_scene(env.data, camera="front_pixels")
                if ss == 1:
                    renderer.render(out=images[index])
                else:
                    renderer.render(out=high_resolution)
                    images[index] = np.rint(high_resolution.reshape(size, ss, size, ss, 3).mean((1, 3)))
    return torch.as_tensor(images, dtype=torch.uint8, device=device)
