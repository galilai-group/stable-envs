"""Configure GL before importing MuJoCo or dm_control.

On macOS, bind MuJoCo to its native backend before importing dm_control with
rendering disabled. This avoids GLFW initialization during discovery. Restore
``glfw`` afterward for Gymnasium, which rejects ``cgl``. On Linux, an unset
``MUJOCO_GL`` becomes EGL when libEGL and libGL load, else OSMesa when libOSMesa
loads, else ``disable``: MuJoCo then imports without a renderer, physics and the
Numba renderers still work, and only a MuJoCo OpenGL renderer raises, with the fix.
Explicit settings are kept.
"""

from __future__ import annotations

import atexit
import ctypes
import functools
import importlib
import importlib.util
import os
import sys
import threading

_DARWIN_MUJOCO = frozenset({"cgl", "glfw"})
_LINUX_MUJOCO = frozenset({"egl", "osmesa", "glfw"})
GL_VARS = ("MUJOCO_GL", "PYOPENGL_PLATFORM", "MUJOCO_EGL_DEVICE_ID")
#: The GL variables as the process received them, before configure_mujoco_gl fills defaults.
INITIAL_GL_ENV = {name: os.environ.get(name) for name in GL_VARS}

#: Libraries each Linux backend loads through PyOpenGL: one name of every group must load.
GL_LIBRARIES = {
    "egl": (("libEGL.so.1", "libEGL.so"), ("libOpenGL.so.0", "libGL.so.1", "libOpenGL.so", "libGL.so")),
    "osmesa": (("libOSMesa.so.8", "libOSMesa.so.6", "libOSMesa.so"),),
}

NO_RENDERER_FIX = (
    "Headless Linux: install EGL (apt install libegl1 libgl1; NVIDIA machines also need the driver's "
    "libEGL_nvidia) and keep MUJOCO_GL=egl, or install OSMesa (apt install libosmesa6) and set "
    "MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa. With a display, MUJOCO_GL=glfw. "
    "`stable-envs doctor` tests each backend on this machine."
)


def selected_backend(environ=None, platform: str = sys.platform) -> str:
    """Return the MUJOCO_GL backend MuJoCo binds when stable_envs imports it.

    Unset on Linux, it follows a PYOPENGL_PLATFORM that names a MuJoCo backend, else EGL.
    """
    environ = os.environ if environ is None else environ
    backend = environ.get("MUJOCO_GL")
    if platform == "darwin":
        return backend if backend in _DARWIN_MUJOCO else "cgl"
    if backend is not None:
        return backend
    pyopengl = (environ.get("PYOPENGL_PLATFORM") or "").lower()
    return pyopengl if pyopengl in _LINUX_MUJOCO else "egl"


@functools.cache
def library_loads(name: str) -> bool:
    """Whether the shared library ``name`` loads in this process."""
    try:
        ctypes.CDLL(name)
    except OSError:
        return False
    return True


def gl_libraries_load(backend: str) -> bool:
    """Whether the shared libraries ``backend`` needs load in this process; True for backends without a list."""
    groups = GL_LIBRARIES.get(backend)
    return groups is None or all(any(library_loads(name) for name in group) for group in groups)


def resolved_backend(environ=None, platform: str = sys.platform, loads=gl_libraries_load) -> str:
    """Return the ``MUJOCO_GL`` value that ``configure_mujoco_gl`` sets.

    An explicit ``MUJOCO_GL`` is kept. Unset on Linux, the backend from ``selected_backend`` is used when its
    libraries load. Without ``PYOPENGL_PLATFORM``, EGL falls back to OSMesa, then to ``"disable"``.
    """
    environ = os.environ if environ is None else environ
    backend = selected_backend(environ, platform)
    if not platform.startswith("linux") or environ.get("MUJOCO_GL") is not None:
        return backend
    candidates = (backend,) if environ.get("PYOPENGL_PLATFORM") else (backend, "osmesa")
    for candidate in dict.fromkeys(candidates):
        if loads(candidate):
            return candidate
    return "disable"


def configure_mujoco_gl() -> str:
    """Configure import-time GL selection and return the resulting environment value."""
    if sys.platform == "darwin":
        if "mujoco" not in sys.modules:
            os.environ["MUJOCO_GL"] = selected_backend()
            try:
                import mujoco  # noqa: F401
            except Exception:  # noqa: BLE001 — optional extra; discovery records the miss
                pass
        have_dmc = importlib.util.find_spec("dm_control") is not None
        if have_dmc and "dm_control._render" not in sys.modules:
            os.environ["MUJOCO_GL"] = "disable"
            try:
                importlib.import_module("dm_control.mujoco")
            except Exception:  # noqa: BLE001 — a backend that is not installed
                return os.environ["MUJOCO_GL"]
        # Gymnasium's OffScreenViewer only accepts glfw/egl/osmesa. dm_control is already
        # imported (or absent), so restoring glfw does not re-enter GLFW init in this process.
        os.environ["MUJOCO_GL"] = "glfw"
        return os.environ["MUJOCO_GL"]
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = resolved_backend()
    return os.environ["MUJOCO_GL"]


def _renderer_import_error() -> Exception | None:
    # mujoco swallows the ImportError that leaves it without Renderer; importing again recovers it.
    for name in ("mujoco.rendering.classic.renderer", "mujoco.renderer"):
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as e:
            if e.name != name:
                return e
        except Exception as e:  # noqa: BLE001 — reported to the user
            return e
    return None


_GL_LOCK = threading.RLock()
_GL_OWNED: dict[int, tuple] = {}
_GL_EXIT_HOOKED = False


def track_gl(resource, release=None):
    """Own an object that holds an OpenGL context until ``release_gl`` or interpreter exit; return it.

    ``release`` defaults to ``release_renderer``. Exit releases newest first, before MuJoCo ends its EGL display.
    """
    global _GL_EXIT_HOOKED
    with _GL_LOCK:
        _GL_OWNED[id(resource)] = (resource, release or release_renderer)
        if not _GL_EXIT_HOOKED:
            # Registered after MuJoCo's EGL display hook, so it runs first.
            _GL_EXIT_HOOKED = True
            atexit.register(release_all_gl)
    return resource


def release_renderer(renderer) -> None:
    """Free a ``mujoco.Renderer``: make its context current so its GL objects are the ones freed, then close it."""
    context = getattr(renderer, "_gl_context", None)
    if context is not None:
        try:
            context.make_current()
        except Exception:  # noqa: BLE001 — a context that is already gone has nothing to free
            pass
    renderer.close()


def close_source_env(env) -> None:
    """Close an upstream OGBench environment together with the ``mujoco.Renderer`` it keeps outside ``close``."""
    unwrapped = getattr(env, "unwrapped", env)
    for name in ("custom_renderer", "_renderer"):
        renderer = getattr(unwrapped, name, None)
        if renderer is not None:
            setattr(unwrapped, name, None)
            release_renderer(renderer)
    env.close()


def track_source_env(env):
    """Own an upstream OGBench environment; ``release_gl`` closes it with its renderer."""
    return track_gl(env, close_source_env)


def release_gl(resource) -> None:
    """Release one tracked object once; an untracked object is ignored."""
    with _GL_LOCK:
        entry = _GL_OWNED.pop(id(resource), None)
    if entry is not None:
        entry[1](resource)


def release_all_gl() -> int:
    """Release every tracked object, newest first, and return how many were released.

    Interpreter exit calls this. Caches of other families keep their renderers for the process lifetime, so a
    call while they can still render leaves them with a closed renderer. A failing release does not stop the
    others; the first failure is raised afterward.
    """
    with _GL_LOCK:
        entries = list(_GL_OWNED.values())[::-1]
        _GL_OWNED.clear()
    first = None
    for resource, release in entries:
        try:
            release(resource)
        except Exception as e:  # noqa: BLE001
            first = first or e
    if first is not None:
        raise first
    return len(entries)


def mujoco_renderer(model, *, height: int, width: int):
    """Construct ``mujoco.Renderer``, failing with the GL setup to change when no OpenGL context can be made.

    The renderer is tracked: ``release_gl`` frees it, and interpreter exit frees whatever remains.
    """
    import mujoco

    backend = os.environ.get("MUJOCO_GL", "<unset>")
    renderer = getattr(mujoco, "Renderer", None)
    if backend == "disable" and INITIAL_GL_ENV["MUJOCO_GL"] is None:
        raise RuntimeError(
            "MuJoCo has no OpenGL renderer: neither libEGL with libGL nor libOSMesa loads on this machine, "
            f"so stable-envs imported MuJoCo with MUJOCO_GL=disable. {NO_RENDERER_FIX}"
        )
    if renderer is None:
        cause = _renderer_import_error()
        detail = f"MuJoCo has no OpenGL renderer with MUJOCO_GL={backend}"
    else:
        try:
            return track_gl(renderer(model, height=height, width=width))
        except ValueError:
            raise
        except Exception as e:  # noqa: BLE001 — re-raised with the fix
            cause = e
            detail = f"MuJoCo could not create an OpenGL context with MUJOCO_GL={backend}"
    pyopengl = os.environ.get("PYOPENGL_PLATFORM")
    if cause is not None and "PYOPENGL_PLATFORM" in str(cause) and pyopengl:
        raise RuntimeError(
            f"PYOPENGL_PLATFORM={pyopengl} conflicts with MUJOCO_GL={backend}: unset PYOPENGL_PLATFORM "
            f"or set both to the same backend. {NO_RENDERER_FIX}"
        ) from cause
    reason = f" ({type(cause).__name__}: {cause})" if cause is not None else ""
    raise RuntimeError(f"{detail}{reason}. {NO_RENDERER_FIX}") from cause
