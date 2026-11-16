"""Process settings that JAX reads when it starts a backend.

JAX reserves most of the GPU memory when it starts a GPU backend, and leaves Torch little of it.
``prepare`` turns that off before the first JAX backend of the process starts.
"""

from __future__ import annotations

import os
import sys

from .config import log

PREALLOCATE = "XLA_PYTHON_CLIENT_PREALLOCATE"
#: Variables that configure the JAX GPU allocator. When one is set, the user chose the memory behavior.
ALLOCATOR_VARS = (PREALLOCATE, "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_ALLOCATOR")

_prepared = False


def _started_backends() -> tuple[str, ...]:
    """The platforms of the JAX backends that this process started, or () when none started."""
    if "jax" not in sys.modules:
        return ()
    try:
        from jax._src import xla_bridge
    except ImportError:
        return ()
    if not xla_bridge.backends_are_initialized():
        return ()
    return tuple(xla_bridge.backends())


def prepare() -> None:
    """Turn off JAX's GPU preallocation, before a JAX backend of stable-envs starts.

    Sets ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` when JAX has not started a backend yet. A value that is already
    set is kept. When JAX already runs on a GPU without the variable, logs one warning with the fix.
    """
    global _prepared
    if _prepared:
        return
    _prepared = True
    started = _started_backends()
    if not started:
        os.environ.setdefault(PREALLOCATE, "false")
        return
    if any(name in os.environ for name in ALLOCATOR_VARS):
        return
    if any(platform in ("cuda", "rocm", "gpu") for platform in started):
        log(
            "WARNING",
            f"JAX started on the GPU before stable-envs could set {PREALLOCATE}=false, so it preallocated most "
            f"of the GPU memory, and Torch may run out of memory. Set {PREALLOCATE}=false in the environment, "
            "or build the stable-envs environment before other code uses JAX.",
        )
