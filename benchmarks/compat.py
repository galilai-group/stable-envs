"""Class attributes the harness reads, with the values they have where a stable-envs version does not define them."""

#: Source GPU renderer per family package: what ``reference_gpu_render`` declares where it exists.
_GPU_RENDER = {"dmcontrol": "mujoco-egl", "fetch": "mujoco-egl", "ogbench": "mujoco-egl", "rocket": "pybullet-egl"}


def native_size(cls):
    """The size the class draws at natively, or None."""
    size = getattr(cls, "native_size", None)
    if size is None:
        size = getattr(cls, "fixed_size", None)
    if size is None and any(getattr(cls, "native", None) or ()):
        size = cls.native[1] or None
    return size


def reference_gpu_render(cls):
    """The source's GPU renderer ("mujoco-egl", "pybullet-egl"), or None when the source draws on CPU only."""
    if hasattr(cls, "reference_gpu_render"):
        return cls.reference_gpu_render
    parts = cls.__module__.split(".")
    return _GPU_RENDER.get(parts[2]) if len(parts) > 2 and parts[1] == "envs" else None


#: Families whose compact rows draw only on some devices (``row_render_devices`` where it exists).
_ROW_DEVICES = {"rocket": ("cuda",)}


def row_render_devices(cls):
    """Devices where the class draws compact rows, such as ``sample_states`` returns."""
    if hasattr(cls, "row_render_devices"):
        return tuple(cls.row_render_devices)
    parts = cls.__module__.split(".")
    return _ROW_DEVICES.get(parts[2], ("cpu", "cuda")) if len(parts) > 2 and parts[1] == "envs" else ("cpu", "cuda")


def render_rows(env, rows, **kwargs):
    """Draw compact render rows: ``render_rows`` where it exists, else the renderer that ``render`` calls."""
    if hasattr(env, "render_rows"):
        return env.render_rows(rows, **kwargs)
    import torch

    return env._render(torch.as_tensor(rows, device=env.device, dtype=torch.float32), **kwargs)


def reference_inputs(env, state):
    """What the class's ``reference_render`` draws for full State ``state``: its compact render rows."""
    if hasattr(env, "reference_inputs"):
        return env.reference_inputs(state)
    import numpy as np
    import torch

    return np.asarray(torch.as_tensor(env._pack_render_state(state)).detach().cpu(), np.float32)
