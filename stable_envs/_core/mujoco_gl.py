"""Configure GL before importing MuJoCo or dm_control.

On macOS, bind MuJoCo to its native backend before importing dm_control with
rendering disabled. This avoids GLFW initialization during discovery. Restore
``glfw`` afterward for Gymnasium, which rejects ``cgl``. On other platforms,
default to offscreen EGL and preserve explicit settings.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys

_DARWIN_MUJOCO = frozenset({"cgl", "glfw"})
_LINUX_MUJOCO = frozenset({"egl", "osmesa", "glfw"})
GL_VARS = ("MUJOCO_GL", "PYOPENGL_PLATFORM", "MUJOCO_EGL_DEVICE_ID")
#: The GL variables as the process received them, before configure_mujoco_gl fills defaults.
INITIAL_GL_ENV = {name: os.environ.get(name) for name in GL_VARS}

NO_RENDERER_FIX = (
    "Headless Linux: install EGL (apt install libegl1; NVIDIA machines also need the driver's "
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
    os.environ.setdefault("MUJOCO_GL", selected_backend())
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


def mujoco_renderer(model, *, height: int, width: int):
    """Construct ``mujoco.Renderer``, failing with the GL setup to change when no OpenGL context can be made."""
    import mujoco

    backend = os.environ.get("MUJOCO_GL", "<unset>")
    renderer = getattr(mujoco, "Renderer", None)
    if renderer is None:
        cause = _renderer_import_error()
        detail = f"MuJoCo has no OpenGL renderer with MUJOCO_GL={backend}"
    else:
        try:
            return renderer(model, height=height, width=width)
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
