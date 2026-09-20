"""Pick a MuJoCo GL backend that will not hang this process.

``MUJOCO_GL`` is read when a renderer is first imported, and the wrong value is not a skip — it is
either an abort or a hang:

* **Linux, unset.** MuJoCo defaults to GLFW, which needs ``DISPLAY``. On a headless GPU node that
  surfaces as ``glfw.GLFWError: X11: The DISPLAY environment variable is missing`` raised out of
  every fixture, then SIGABRT, so pytest never prints a summary. EGL is the offscreen default there.
* **macOS, ``egl``.** Not a mujoco backend (3.12 raises ``invalid value for environment variable
  MUJOCO_GL: egl``). dm_control still tries ``dlopen(EGL)`` and loses the family.
* **macOS, ``glfw`` / unset.** ``import mujoco`` and ``mujoco.Renderer`` return. ``from dm_control
  import mujoco`` does not: ``dm_control._render`` initialises GLFW and blocks in NSApp. That is the
  hang ``import stable_envs`` used to be, because discovery imports the DM Control family at
  package import.

Darwin therefore loads mujoco first with its native offscreen backend (``cgl``), then imports
dm_control under ``disable`` — a value it accepts without opening a window — so discovery cannot take
the GLFW path. mujoco's renderer stays the one that was already imported. After that bind,
``MUJOCO_GL`` is restored to ``glfw`` because gymnasium's ``OffScreenViewer`` only accepts
``glfw`` / ``egl`` / ``osmesa``. dm_control is already imported, so the restore does not re-enter
GLFW init in this process. An explicit ``MUJOCO_GL`` that is valid for mujoco on this host (``cgl``,
``glfw``) is honoured for that first import; ``egl`` / ``osmesa`` are not valid here and are treated
as unset.

Call this before importing mujoco or dm_control. ``discover()`` does; env modules that can be
imported directly do too. An explicit setting on Linux always wins.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys

_DARWIN_MUJOCO = frozenset({"cgl", "glfw"})


def configure_mujoco_gl() -> str:
    """Set ``MUJOCO_GL`` if needed and return the value later importers will see."""
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
