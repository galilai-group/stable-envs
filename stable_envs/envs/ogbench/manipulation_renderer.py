from __future__ import annotations

import hashlib
import threading
from functools import lru_cache
from importlib.metadata import version

import numpy as np
import torch

from . import manipulation as core
from .simulation import render_state as pixels

_LOCK = threading.RLock()


@lru_cache(maxsize=1)
def _check_versions():
    if version("mujoco") != "3.12.0" or version("ogbench") != "1.2.1":
        raise ImportError("Cube/Scene unified rendering requires mujoco==3.12.0 and ogbench==1.2.1")
    if version("mujoco-warp") != "3.12.0" or version("warp-lang") != "1.17.0":
        raise ImportError("Cube/Scene frozen GPU rendering requires mujoco-warp==3.12.0 and warp-lang==1.17.0")


def configuration_key(family, row):
    h = hashlib.sha256(family.encode())
    for key in (*pixels.STATIC, *(("lock_color",) if family == "Scene" else ())):
        h.update(row["appearance", key].detach().cpu().contiguous().numpy().tobytes())
    return f"stable-envs-{family}-{h.hexdigest()}"


def _configuration_groups(family, host):
    """Group exact model bytes without constructing a TensorDict for every row."""
    keys = (*pixels.STATIC, *(("lock_color",) if family == "Scene" else ()))
    columns = [
        np.ascontiguousarray(host["appearance", key].numpy()).reshape(len(host), -1).view(np.uint8) for key in keys
    ]
    groups = {}
    for index, row in enumerate(np.concatenate(columns, axis=1)):
        groups.setdefault(row.tobytes(), []).append(index)
    return groups.values()


def configured_bundle(family, row, sample_pattern="grid"):
    """Cache the compiled render model under its immutable configuration key."""
    token = configuration_key(family, row)
    if sample_pattern != "grid":
        token += "/" + sample_pattern
    with _LOCK:
        bundle = core.get_bundle(token)
        if bundle is None:
            env = pixels.build_model(family, row)
            if int(env.model.joint("object_joint_0").qposadr[0]) != pixels.OBJECT_QPOS_ADR:
                raise ValueError(f"{family}: object_joint_0 has an unexpected qpos address")
            bundle = core.build_bundle(
                env, camera="front_pixels", size=core.SPEC[family]["size"], nbuttons=2 if family == "Scene" else 0
            )
            bundle.sample_pattern = sample_pattern
            core.register_bundle(token, bundle)
        return token, bundle


class _ConfiguredRenderer(core.OGBManipBase):
    """Only model/input wiring differs; the render and plate algorithms are inherited."""

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

    def raster(self, state):
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
        )

    def draw(self, state):
        return super()._render(
            self._rows(state),
            mocap_quat=state["pose", "mocap_quat"].reshape(len(state), self._b.nmocap, 4),
            agent_color=self._agent(state),
            variation={
                "cube.color": state["appearance", "cube.color"].reshape(len(state), 3),
                "floor.color": state["appearance", "floor.color"].reshape(len(state), 6),
            },
        )


def render(family, state, device, backend="auto", chunk_size=None):
    if backend == "opengl" or (backend == "auto" and torch.device(device).type == "cpu"):
        from . import manipulation_opengl

        return manipulation_opengl.render(family, state, device, chunk_size)
    if backend == "auto":
        backend = "mjwarp_fidelity"
    host = state.detach().cpu()
    pixels.validate(family, host)
    if backend not in ("auto", "mjwarp", "mjwarp_fidelity"):
        raise ValueError("Cube/Scene rendering supports mjwarp, mjwarp_fidelity, or explicit opengl backend")
    if torch.device(device).type != "cuda":
        raise NotImplementedError("Cube/Scene CPU mesh rendering is unavailable; renderer/GPU execution requires CUDA")
    _check_versions()
    if chunk_size is not None and (type(chunk_size) is not int or chunk_size < 1):
        raise ValueError("chunk_size must be a positive integer")
    size, ss = (int(host["renderer", key][0, 0]) for key in ("size", "supersample"))
    result = torch.empty((len(state), size, size, 3), dtype=torch.uint8, device=device)
    # Frozen caches hold mutable render buffers; serialize users of this caller boundary.
    import warp as wp

    with _LOCK, torch.cuda.device(device), wp.ScopedDevice(str(device)):
        prepared = []
        active = set()
        device_index = torch.cuda.current_device()
        for indices in _configuration_groups(family, host):
            pattern = "gl4" if family == "Scene" and backend == "mjwarp_fidelity" and ss == 2 else "grid"
            token, bundle = configured_bundle(family, host[indices[0]], pattern)
            prepared.append((token, bundle, indices))
            limit = chunk_size or len(indices)
            for start in range(0, len(indices), limit):
                count = min(limit, len(indices) - start)
                for offset in range(0, count, core.MJW_CHUNK):
                    n = min(core.MJW_CHUNK, count - offset)
                    nw = 1 << (n - 1).bit_length()
                    if backend == "mjwarp_fidelity":
                        active.add((device_index, token + "/fidelity/glass", size, nw))
                        active.add((device_index, token + "/fidelity/opaque", size * ss, nw))
                    else:
                        active.add((device_index, token, size * ss, nw))
        with core._CTX_CACHE.working_set(active):
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
