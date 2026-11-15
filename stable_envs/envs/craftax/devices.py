"""The JAX devices of Craftax: the number of CPU devices, and whether JAX has a GPU backend.

This module does not import Craftax, because importing Craftax makes JAX arrays, and that starts the
JAX backends.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import pkgutil

import jax

from ..._core.config import log
from ..._core.cpus import cpu_limit


def host_device_count() -> int:
    """One XLA host device for each CPU that this process can use."""
    return cpu_limit()


def _ask_for_host_devices() -> str | None:
    """Ask JAX for ``host_device_count()`` CPU devices; returns why it could not, or None.

    JAX reads the count once, when it starts its CPU backend, and the count holds for the whole process.
    A count that the user set (``jax_num_cpu_devices``, ``JAX_NUM_CPU_DEVICES`` or
    ``--xla_force_host_platform_device_count`` in ``XLA_FLAGS``) is kept.
    """
    if "xla_force_host_platform_device_count" in os.environ.get("XLA_FLAGS", ""):
        return None
    count = getattr(jax.config, "jax_num_cpu_devices", None)
    if count is None:
        return "this JAX version has no jax_num_cpu_devices setting"
    if count >= 0:
        return None
    try:
        jax.config.update("jax_num_cpu_devices", host_device_count())
    except RuntimeError:
        return "JAX started before the first CPU Craftax environment"
    return None


@functools.cache
def use_host_devices() -> None:
    """Before the first CPU Craftax environment, ask JAX for ``host_device_count()`` CPU devices.

    Logs one warning when JAX keeps one CPU device where Craftax would use more.
    """
    reason = _ask_for_host_devices()
    count = host_device_count()
    if reason is not None and count > 1 and len(jax.devices("cpu")) == 1:
        log(
            "WARNING",
            f"JAX has one CPU device ({reason}), so Craftax steps its CPU lanes on one device instead of {count}. "
            "Make the first CPU Craftax env before other JAX work, or set "
            f"XLA_FLAGS=--xla_force_host_platform_device_count={count} before you start Python.",
        )


def gpu_backend_installed() -> bool:
    """Whether a JAX GPU plugin is installed; finding it does not start JAX."""
    spec = importlib.util.find_spec("jax_plugins")
    if spec is None or not spec.submodule_search_locations:
        return False
    return any("cuda" in m.name or "rocm" in m.name for m in pkgutil.iter_modules(spec.submodule_search_locations))


def has_gpu() -> bool:
    """Whether JAX has a GPU backend. This starts the JAX backends."""
    try:
        return bool(jax.devices("gpu"))
    except RuntimeError:
        return False
