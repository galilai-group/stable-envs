"""Whether Warp can use CUDA here, and why not.

A Warp build whose CUDA toolkit is newer than the driver supports falls back to CPU
arrays without raising, so GPU render paths check this before building contexts.
"""

from __future__ import annotations

import os

#: Lowest Linux driver release for each CUDA major version.
DRIVER_FOR_CUDA = {12: "525.60.13", 13: "580.65.06"}
#: The Warp release the GPU extras and profiles are certified with.
CERTIFIED_WARP = "1.17.0"


def _version(pair) -> str:
    return ".".join(str(int(v)) for v in pair)


def warp_runtime():
    """Warp's initialised runtime, which records the toolkit and driver CUDA versions."""
    import warp as wp

    wp.init()
    try:
        from warp._src import context
    except ImportError:
        context = wp.context
    return context.runtime


def warp_cuda_problem() -> str | None:
    """Return why Warp cannot run on CUDA in this process, or None when it can."""
    import warp as wp

    runtime = warp_runtime()
    if wp.is_cuda_available():
        return None
    toolkit = getattr(runtime, "toolkit_version", None)
    driver = getattr(runtime, "driver_version", None)
    name = f"Warp {wp.config.version}"
    if toolkit is None:
        return f"{name} was built without CUDA support"
    if driver is None:
        return f"{name} found no NVIDIA CUDA driver"
    if tuple(driver) < tuple(getattr(runtime, "min_driver_version", None) or toolkit):
        need = DRIVER_FOR_CUDA.get(int(toolkit[0]))
        return (
            f"{name} is built with CUDA {_version(toolkit)} but the NVIDIA driver supports CUDA "
            f"{_version(driver)} at most, so Warp fell back to CPU. Upgrade the driver"
            + (f" to {need} or newer" if need else "")
            + f", or install warp-lang=={CERTIFIED_WARP}"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return f"{name} sees no CUDA device" + (f" (CUDA_VISIBLE_DEVICES={visible!r})" if visible is not None else "")


def require_warp_cuda(what: str):
    """Return the ``warp`` module, or raise when ``what`` would run on Warp's CPU fallback."""
    problem = warp_cuda_problem()
    if problem is not None:
        raise RuntimeError(f"{what} needs Warp on CUDA: {problem}. `stable-envs doctor` checks this machine.")
    import warp as wp

    return wp
