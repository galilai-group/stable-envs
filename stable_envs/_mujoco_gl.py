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


def configure_mujoco_gl() -> str:
    """Configure import-time GL selection and return the resulting environment value."""
    if sys.platform == "darwin":
        if "mujoco" not in sys.modules:
            backend = os.environ.get("MUJOCO_GL")
            os.environ["MUJOCO_GL"] = backend if backend in _DARWIN_MUJOCO else "cgl"
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
    os.environ.setdefault("MUJOCO_GL", "egl")
    return os.environ["MUJOCO_GL"]
