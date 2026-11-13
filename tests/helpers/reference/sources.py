"""How each source library draws: its GPU renderer, and the inputs its reference renderer takes for a State."""

from __future__ import annotations

import numpy as np
import torch

#: How the source of each family draws on a GPU. Families that are not listed draw on CPU only.
GPU_RENDER: dict[str, str] = {
    "dmcontrol": "mujoco-egl",
    "fetch": "mujoco-egl",
    "ogbench": "mujoco-egl",
    "pyflyt": "pybullet-egl",
}

_REFERENCE_ENVS: dict = {}


def reference_gpu_render(cls) -> str | None:
    """Return ``"mujoco-egl"`` or ``"pybullet-egl"`` for a source that draws on a GPU, or None."""
    return GPU_RENDER.get(getattr(cls, "family", ""))


def _draws_fields(cls) -> bool:
    """PushT draws named render fields with the source's pygame code, not compact rows."""
    return hasattr(cls, "_render_fields_of")


def reference_env(cls, size: int | None = None):
    """Borrow the process-lifetime source renderer of ``cls`` at ``size``; do not close it."""
    if not _draws_fields(cls):
        from stable_envs.testing import conformance

        return conformance.reference_env(cls, int(size or cls.native_size or cls.default_size))
    key = (cls, size)
    if key not in _REFERENCE_ENVS:
        _REFERENCE_ENVS[key] = cls(num_envs=1, size=size, device="cpu", backend="source")
    return _REFERENCE_ENVS[key]


def reference_inputs(env, state):
    """Return what :func:`reference_render` draws for full State ``state``.

    This is the compact render rows on the host, or for PushT one dict of render fields per environment.
    """
    if _draws_fields(type(env)):
        fields = env._render_fields_of(state)
        return [{k: v[i] for k, v in fields.items()} for i in range(len(fields["recipe/config"]))]
    env._check_open("reference_inputs")
    return np.asarray(torch.as_tensor(env._pack_render_state(state)).detach().cpu(), np.float32)


def reference_render(cls, env, inputs) -> np.ndarray:
    """Draw ``inputs`` from :func:`reference_inputs` with the source renderer ``env`` of ``cls``."""
    if _draws_fields(cls):
        return np.stack([np.asarray(env._projection.render([row], env.size))[0] for row in inputs])
    return np.asarray(cls.reference_render(env, inputs))


def has_source_renderer(cls) -> bool:
    """Whether :func:`reference_env` can build a source renderer for ``cls``."""
    from .certification import has_reference_renderer

    return _draws_fields(cls) or has_reference_renderer(cls)
