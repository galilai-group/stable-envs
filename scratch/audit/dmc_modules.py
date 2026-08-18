"""Is dm_control's own EGL context class importable-and-already-imported after `import stable_envs`?

resource_guard patches `mujoco.egl.GLContext` and nothing else, so every EGL context a dm_control
reference env creates is invisible to the rule the guard exists to enforce. Patching it means either
importing dm_control from the guard (a 1-2 s cost and an EGL display creation for the 180 sessions
that never touch it) or patching only what is ALREADY imported. This decides which.
"""
import sys

import stable_envs as se  # noqa: F401

for m in ("dm_control", "dm_control._render", "dm_control._render.base",
          "dm_control._render.pyopengl.egl_renderer", "dm_control.mujoco.engine"):
    print(f"{m:48s} imported={m in sys.modules}")
print("DMC registered:", sum(n.startswith('DMC') for n in se.REGISTRY))
b = sys.modules.get("dm_control._render.base")
if b is not None:
    print("ContextBase methods:", [x for x in ("__init__", "_free_unconditionally", "free", "__del__")
                                   if hasattr(b.ContextBase, x)])
