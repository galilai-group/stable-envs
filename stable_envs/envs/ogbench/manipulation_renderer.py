from __future__ import annotations

import threading

import numpy as np
import torch

from ..._core.config import log
from ..._core.warp_cuda import require_warp_cuda
from . import manipulation as core
from .manipulation_opengl import check_versions
from .simulation import render_state as pixels

_LOCK = threading.RLock()
OPENGL = ("opengl", "opengl_fast")
# Backends that draw from host RenderState columns.
HOST = (*OPENGL, "numba")


VIEW = ("camera.angle_delta", "light.intensity")
# Compiled arrays a configuration changes that the fixed-camera MJWarp and OpenGL passes never read.
_INERT = (
    "body_inertia",
    "body_invweight0",
    "body_mass",
    "body_subtreemass",
    "bvh_aabb",
    "cam_mat0",
    "cam_poscom0",
    "dof_M0",
    "dof_invweight0",
    "dof_length",
    "geom_aabb",
    "geom_rbound",
    "light_poscom0",
)
_VIEWS = core._RES.BoundedCache(1024, name="ogbench.view")
_LOCKS = core._RES.BoundedCache(1024, name="ogbench.lock")


def _groups(host, names):
    """Group rows by the exact bytes of the named appearance leaves; return keys, first rows and row groups."""
    columns = [
        np.ascontiguousarray(host["appearance", name].numpy()).reshape(len(host), -1).view(np.uint8) for name in names
    ]
    keys, first, group = {}, [], np.empty(len(host), np.int64)
    for index, row in enumerate(np.concatenate(columns, axis=1)):
        key = row.tobytes()
        if key not in keys:
            keys[key] = len(first)
            first.append(index)
        group[index] = keys[key]
    return list(keys), first, group


def configured_bundle(family, sample_pattern="grid"):
    """Compile the family's render model once, at default configuration."""
    token = f"stable-envs-{family}" + ("" if sample_pattern == "grid" else "/" + sample_pattern)
    with _LOCK:
        bundle = core.get_bundle(token)
        if bundle is None:
            import mujoco

            env = pixels.build_model(family)
            if int(env.model.joint("object_joint_0").qposadr[0]) != pixels.OBJECT_QPOS_ADR:
                raise ValueError(f"{family}: object_joint_0 has an unexpected qpos address")
            fixed = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
            if np.any(env.model.cam_mode != fixed) or np.any(env.model.light_mode != fixed):
                raise ValueError(f"{family}: per-world cameras and lights assume fixed modes")
            bundle = core.build_bundle(
                env, camera="front_pixels", size=core.SPEC[family]["size"], nbuttons=2 if family == "Scene" else 0
            )
            bundle.sample_pattern = sample_pattern
            core.register_bundle(token, bundle)
        return token, bundle


class _View:
    __slots__ = ("key", "fields")


def _view(family, bundle, host, index, value):
    """Camera and light arrays of a configuration, from its own compile at default cube size."""
    key = (family, value)
    hit = _VIEWS.get(key)
    if hit is not None:
        return hit
    env = pixels.build_model(family, host[index], VIEW)
    try:
        changed = [
            name
            for name in dir(env.model)
            if isinstance(value := getattr(env.model, name, None), np.ndarray)
            and not np.array_equal(value, getattr(bundle.model, name))
        ]
        if set(changed) - {*core.VIEW_FIELDS, *_INERT}:
            raise ValueError(f"{family}: camera/light configuration changed unbatched arrays {changed}")
        view = _View()
        view.key = key
        view.fields = {name: np.array(getattr(env.model, name), copy=True) for name in core.VIEW_FIELDS}
    finally:
        env.close()
    return _VIEWS.put(key, view)


class _Lock:
    __slots__ = ("key", "tables")


def _lock(token, bundle, value):
    """Scene button colour tables for one lock_color, applied by the source rule to the shared model."""
    key = (token, value.tobytes())
    hit = _LOCKS.get(key)
    if hit is not None:
        return hit
    u = bundle.unwrapped
    node = u.variation_space._walk("lock_color")
    saved = node.value.copy()
    node.set_value(value.copy())
    try:
        tables = core._button_tables(u, bundle.model, bundle.nbuttons)
    finally:
        node.set_value(saved)
        u._apply_button_states()
    zero_g, zero_m = bundle.button_tables[(0,) * bundle.nbuttons]
    geoms = {int(i) for g, _ in tables.values() for i in np.flatnonzero(np.any(g != zero_g, axis=1))}
    materials = {int(i) for _, m in tables.values() for i in np.flatnonzero(np.any(m != zero_m, axis=1))}
    static = [g for g in range(bundle.model.ngeom) if g not in set(bundle.movers)]
    if geoms - set(bundle.movers) or materials & {int(bundle.model.geom_matid[g]) for g in static}:
        raise ValueError("lock_color colours a static geom, which the cached background would bake")
    lock = _Lock()
    lock.key = key
    lock.tables = tables
    return _LOCKS.put(key, lock)


class _Worlds:
    """Per-world configuration arrays of one RenderState batch, sliced alongside its rows."""

    __slots__ = ("fields", "view", "views", "lock", "locks")

    def __getitem__(self, index):
        out = _Worlds()
        out.fields = {name: value if len(value) == 1 else value[index] for name, value in self.fields.items()}
        out.view, out.views = self.view[index], self.views
        out.lock, out.locks = (None, None) if self.locks is None else (self.lock[index], self.locks)
        return out


def _worlds(family, token, bundle, host, device):
    worlds = _Worlds()
    keys, first, worlds.view = _groups(host, VIEW)
    worlds.views = [_view(family, bundle, host, index, key) for key, index in zip(keys, first)]
    # Cube and target are boxes whose compiled size is the leaf itself; every other geom keeps its size.
    size = host["appearance", "cube.size"].numpy()
    size = size[:1] if np.all(size == size[0]) else size
    geom_size = torch.as_tensor(bundle.model.geom_size, device=device, dtype=torch.float32)
    geom_size = geom_size.expand(len(size), *geom_size.shape).clone()
    geom_size[:, torch.as_tensor(bundle.cube_geoms, device=device)] = torch.as_tensor(
        size[:, :, None], device=device, dtype=torch.float32
    )
    worlds.fields = {"geom_size": geom_size}
    for name in core.VIEW_FIELDS:
        stack = torch.as_tensor(np.stack([view.fields[name] for view in worlds.views]), device=device)
        stack = stack.to(torch.float32)
        worlds.fields[name] = stack if len(stack) == 1 else stack[torch.as_tensor(worlds.view, device=device)]
    worlds.lock = worlds.locks = None
    if family == "Scene":
        _, first, worlds.lock = _groups(host, ("lock_color",))
        lock = host["appearance", "lock_color"].numpy()
        worlds.locks = [_lock(token, bundle, lock[index]) for index in first]
    return worlds


class _ConfiguredRenderer(core.OGBManipBase):
    """Renderer over a configured bundle; render and plate algorithms are inherited."""

    def __init__(self, token, bundle, device, size, supersample):
        super().__init__(
            num_envs=1,
            size=size,
            supersample=supersample,
            device=device,
            backend="mjwarp",
            bundle=bundle,
            cache_key=token,
        )

    def _rows(self, state):
        columns = [state["pose", "qpos"], state["pose", "mocap_pos"]]
        if self._b.nbuttons:
            columns.append(state["pose", "buttons"])
        return torch.cat(columns, dim=-1).to(self.device, torch.float32)

    def _agent(self, state):
        return state["appearance", "agent.color"].reshape(len(state), 3).to(self.device, torch.float32)

    def raster(self, state, worlds):
        cube = self._cube_colours(
            {"cube.color": state["appearance", "cube.color"].detach().cpu().numpy().reshape(len(state), 3)},
            len(state),
        )
        return self._raster(
            self._rows(state),
            self.size,
            cube,
            state["pose", "mocap_quat"].reshape(len(state), self._b.nmocap, 4),
            self._agent(state),
            worlds,
        )

    def draw(self, state, worlds):
        return super()._render(
            self._rows(state),
            mocap_quat=state["pose", "mocap_quat"].reshape(len(state), self._b.nmocap, 4),
            agent_color=self._agent(state),
            variation={
                "cube.color": state["appearance", "cube.color"].reshape(len(state), 3),
                "floor.color": state["appearance", "floor.color"].reshape(len(state), 6),
            },
            worlds=worlds,
        )


_RENDERERS = core._RES.BoundedCache(32, name="ogbench.renderer", on_evict=lambda renderer: renderer.close())


def configured_renderer(key, device, size, supersample):
    """Return the cached renderer of a registered bundle; callers hold ``_LOCK``."""
    bundle = core.get_bundle(key)
    cache_key = (key, str(device), int(size), int(supersample))
    renderer = _RENDERERS.get(cache_key)
    if renderer is None or renderer._b is not bundle:
        renderer = _RENDERERS.put(cache_key, _ConfiguredRenderer(key, bundle, device, size, supersample))
    return renderer


def draws_on_host(backend, device, supersample):
    """Whether ``backend`` draws through MuJoCo OpenGL from host inputs.

    ``auto`` does so on CPU, and on CUDA at the source's own 1x sampling; CUDA ``auto`` at 2x or more
    keeps the MJWarp fidelity renderer.
    """
    return backend in HOST or (backend == "auto" and (torch.device(device).type == "cpu" or supersample == 1))


def host_backend(backend, device):
    """The host backend that ``backend`` draws with on ``device``: opengl, opengl_fast or numba.

    ``auto`` draws with opengl_fast on CPU and with opengl on CUDA.
    """
    if backend in HOST:
        return backend
    return "opengl_fast" if torch.device(device).type == "cpu" else "opengl"


def _supersample(state):
    try:
        return int(state["renderer", "supersample"].reshape(-1)[0])
    except Exception:  # noqa: BLE001 - validation reports the malformed State
        return None


def render(family, state, device, backend="auto", chunk_size=None, validated=False):
    if draws_on_host(backend, device, _supersample(state)):
        from . import manipulation_opengl

        chosen = host_backend(backend, device)
        if backend == "auto":
            log("DEBUG", f"Renderer backend 'auto' chose {chosen} on {device}.")
        if chosen == "numba":
            from . import manipulation_numba

            host, device, _ = manipulation_opengl.prepare(family, state, device, chunk_size, validated)
            return manipulation_numba.frames(family, manipulation_opengl.columns(host), device)
        return manipulation_opengl.render(family, state, device, chunk_size, validated, chosen == "opengl_fast")
    if backend == "auto":
        backend = "mjwarp_fidelity"
        log("DEBUG", f"Renderer backend 'auto' chose {backend} on {device}.")
    host = state.detach().cpu()
    pixels.validate(family, host)
    if backend not in ("auto", "mjwarp", "mjwarp_fidelity"):
        raise ValueError("Cube/Scene rendering supports mjwarp, mjwarp_fidelity, opengl, opengl_fast or numba backends")
    if torch.device(device).type != "cuda":
        raise NotImplementedError("Cube/Scene CPU mesh rendering is unavailable; renderer/GPU execution requires CUDA")
    check_versions("unified", warp=True)
    if chunk_size is not None and (type(chunk_size) is not int or chunk_size < 1):
        raise ValueError("chunk_size must be a positive integer")
    size, ss = (int(host["renderer", key][0, 0]) for key in ("size", "supersample"))
    result = torch.empty((len(state), size, size, 3), dtype=torch.uint8, device=device)
    # Shared caches hold mutable render buffers; serialize callers.
    wp = require_warp_cuda("OGBench manipulation GPU rendering")
    with _LOCK, torch.cuda.device(device), wp.ScopedDevice(str(device)):
        pattern = "gl4" if family == "Scene" and backend == "mjwarp_fidelity" and ss == 2 else "grid"
        token, bundle = configured_bundle(family, pattern)
        worlds = _worlds(family, token, bundle, host, device)
        limit = chunk_size or len(state)
        active = set()
        device_index = torch.cuda.current_device()
        step = core.mjw_worlds()
        for start in range(0, len(state), limit):
            count = min(limit, len(state) - start)
            for offset in range(0, count, step):
                nw = 1 << (min(step, count - offset) - 1).bit_length()
                if backend == "mjwarp_fidelity":
                    active.add((device_index, token + "/fidelity/glass", size, nw))
                    active.add((device_index, token + "/fidelity/opaque", size * ss, nw))
                else:
                    active.add((device_index, token, size * ss, nw))
        with core._CTX_CACHE.working_set(active):
            if backend == "mjwarp_fidelity":
                from . import manipulation_fidelity

                for start in range(0, len(state), limit):
                    selected = slice(start, start + limit)
                    result[selected] = manipulation_fidelity.draw(
                        token, bundle, state[selected], device, size, ss, worlds[selected]
                    )
                return result
            renderer = configured_renderer(token, device, size, ss)
            for start in range(0, len(state), limit):
                selected = slice(start, start + limit)
                result[selected] = renderer.draw(state[selected], worlds[selected])
    return result
