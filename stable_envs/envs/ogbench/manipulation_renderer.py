"""Supply configured models and named pixel inputs to the unchanged frozen renderer.

The frozen module owns rasterization, MJWarp, corrections and context policy.
Its caches are namespaced by immutable compiled configuration at this boundary.
"""

from __future__ import annotations

import hashlib
import threading
from importlib.metadata import version

import torch

from ..._adapters import ogbench_render_state as pixels
from ...env import Env
from . import manipulation as frozen

_LOCK = threading.RLock()
FROZEN_SHA256 = "1c7c5b4a00581d5d6e412da38ce1d139595bd230050fb43cac7d339ec81e9e0b"


def configuration_key(family, row):
    h = hashlib.sha256(family.encode())
    for key in (*pixels.STATIC, *(("lock_color",) if family == "Scene" else ())):
        h.update(row["appearance", key].detach().cpu().contiguous().numpy().tobytes())
    return f"stable-envs-{family}-{h.hexdigest()}"


def configured_bundle(family, row):
    """Populate the frozen model-provider cache without its obsolete constructors."""
    import mujoco

    token = configuration_key(family, row)
    with _LOCK:
        if token in frozen._BUNDLE_CACHE:
            return token, frozen._BUNDLE_CACHE[token]
        env = pixels.build_model(family, row)
        model = env.model
        b = frozen._Bundle()
        b.unwrapped = env
        b.size = frozen.SPEC[family]["size"]
        b.cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_pixels")
        assert b.cam >= 0 and int(model.joint("object_joint_0").qposadr[0]) == 14
        b.movers = frozen._movers(model)
        b.nq, b.nv, b.nmocap = model.nq, model.nv, model.nmocap
        b.nbuttons = 2 if family == "Scene" else 0
        b.state_names = (
            tuple(f"qpos_{i}" for i in range(b.nq))
            + ("goal_x", "goal_y", "goal_z")
            + tuple(f"button_{i}" for i in range(b.nbuttons))
        )
        b.pristine = {name: getattr(model, name).copy() for name in ["geom_group", "geom_rgba", "mat_rgba"]}
        b.button_tables = frozen._button_tables(env, model, b.nbuttons)
        b.var_defaults = frozen._variation_defaults(env)
        b.cube_geoms = frozen._cube_geoms(env, model)
        b.grid_tex = frozen._grid_texture(model, b.var_defaults["floor.color"])
        frozen._BUNDLE_CACHE[token] = b
        return token, b


class _ConfiguredRenderer(frozen.OGBManipBase):
    """Only model/input wiring differs; the render and plate algorithms are inherited."""

    def __init__(self, token, bundle, device, size, supersample):
        self.CELL, self._b = token, bundle
        self.state_dim = bundle.nq + 3 + bundle.nbuttons
        self.state_names = bundle.state_names
        Env.__init__(self, num_envs=1, size=size, supersample=supersample, device=device, backend="mjwarp")
        self._mocap_input = None
        self._mocap_offset = 0

    def _identity_quat(self, nmoc, n):
        # The inherited uploader's hook now receives the explicit named mocap input.
        start = self._mocap_offset
        self._mocap_offset += n
        return self._mocap_input[start : start + n].reshape(n, nmoc, 4)

    def draw(self, state):
        columns = [state["pose", "qpos"], state["pose", "mocap_pos"]]
        if self._b.nbuttons:
            columns.append(state["pose", "buttons"])
        rows = torch.cat(columns, dim=-1).to(self.device, torch.float32)
        self._mocap_input = state["pose", "mocap_quat"].to(self.device, torch.float32)
        self._mocap_offset = 0
        try:
            return super()._render(
                rows,
                variation={
                    "cube.color": state["appearance", "cube.color"].reshape(len(state), 3),
                    "floor.color": state["appearance", "floor.color"].reshape(len(state), 6),
                },
            )
        finally:
            self._mocap_input = None
            self._mocap_offset = 0


def render(family, state, device, backend="auto", chunk_size=None):
    if backend == "opengl" or (backend == "auto" and torch.device(device).type == "cpu"):
        from . import manipulation_opengl

        return manipulation_opengl.render(family, state, device, chunk_size)
    host = state.detach().cpu()
    pixels.validate(family, host)
    if backend not in ("auto", "mjwarp", "mjwarp_fidelity"):
        raise ValueError("Cube/Scene rendering supports mjwarp, Cube mjwarp_fidelity, or explicit opengl backend")
    if backend == "mjwarp_fidelity" and family != "Cube":
        raise ValueError("mjwarp_fidelity is currently validated only for Cube")
    if torch.device(device).type != "cuda":
        raise NotImplementedError("Cube/Scene CPU mesh rendering is unavailable; renderer/GPU execution requires CUDA")
    if version("mujoco") != "3.12.0" or version("ogbench") != "1.2.1":
        raise ImportError("Cube/Scene unified rendering requires mujoco==3.12.0 and ogbench==1.2.1")
    if version("mujoco-warp") != "3.12.0" or version("warp-lang") != "1.17.0":
        raise ImportError("Cube/Scene frozen GPU rendering requires mujoco-warp==3.12.0 and warp-lang==1.17.0")
    if chunk_size is not None and (type(chunk_size) is not int or chunk_size < 1):
        raise ValueError("chunk_size must be a positive integer")
    size, ss = (int(host["renderer", key][0, 0]) for key in ("size", "supersample"))
    result = torch.empty((len(state), size, size, 3), dtype=torch.uint8, device=device)
    # Frozen caches hold mutable render buffers; serialize users of this caller boundary.
    import warp as wp

    with _LOCK, torch.cuda.device(device), wp.ScopedDevice(str(device)):
        groups = {}
        for i, row in enumerate(host):
            groups.setdefault(configuration_key(family, row), []).append(i)
        prepared = []
        active = set()
        device_index = torch.cuda.current_device()
        for indices in groups.values():
            token, bundle = configured_bundle(family, host[indices[0]])
            prepared.append((token, bundle, indices))
            limit = chunk_size or len(indices)
            for start in range(0, len(indices), limit):
                count = min(limit, len(indices) - start)
                for offset in range(0, count, frozen.MJW_CHUNK):
                    n = min(frozen.MJW_CHUNK, count - offset)
                    nw = 1 << (n - 1).bit_length()
                    active.add((device_index, token, size * ss, nw))
                    if backend == "mjwarp_fidelity":
                        active.add((device_index, token + "/fidelity/opaque", size * ss, nw))
        with frozen._CTX_CACHE.working_set(active):
            for token, bundle, indices in prepared:
                if backend == "mjwarp_fidelity":
                    from . import manipulation_fidelity

                    limit = chunk_size or len(indices)
                    for start in range(0, len(indices), limit):
                        selected = indices[start : start + limit]
                        result[selected] = manipulation_fidelity.draw(token, bundle, state[selected], device, size, ss)
                    continue
                renderer = _ConfiguredRenderer(token, bundle, device, size, ss)
                try:
                    limit = chunk_size or len(indices)
                    for start in range(0, len(indices), limit):
                        selected = indices[start : start + limit]
                        result[selected] = renderer.draw(state[selected])
                finally:
                    renderer.close()
    return result
