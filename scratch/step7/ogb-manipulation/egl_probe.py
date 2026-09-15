"""Fail fast if this node cannot create an EGL context.

A node whose EGL is broken makes MuJoCo's official renderer raise, or worse render
something degenerate, and the result reads as a RENDERER finding rather than a node
fault. W4 hit exactly that on gpu2104: torch/cuda healthy, EGL context creation broken.
This runs before any measurement and exits nonzero with the device list on failure.
"""
import os, sys

os.environ.setdefault("MUJOCO_GL", "egl")
try:
    from OpenGL import EGL
    d = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
    print("eglGetDisplay(EGL_DEFAULT_DISPLAY) ->", d)
except Exception as e:
    print("EGL import/display probe:", type(e).__name__, e)

import mujoco
import numpy as np

XML = """<mujoco><worldbody>
  <light pos='0 0 2'/><geom type='plane' size='1 1 .1'/>
  <body pos='0 0 .3'><freejoint/><geom type='box' size='.1 .1 .1' rgba='1 0 0 1'/></body>
  <camera name='c' pos='1 0 1' xyaxes='0 1 0 -.7 0 .7'/>
</worldbody></mujoco>"""
m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
try:
    r = mujoco.Renderer(m, height=64, width=64)
    r.update_scene(d, camera="c")
    f = r.render()
except Exception as e:
    print("FATAL: EGL context/render failed on this node:", type(e).__name__, e)
    sys.exit(9)
try:
    from OpenGL import GL
    print("GL_RENDERER", GL.glGetString(GL.GL_RENDERER).decode())
    print("GL_VERSION", GL.glGetString(GL.GL_VERSION).decode())
except Exception as e:
    print("GL strings unreadable:", type(e).__name__)
# A context that creates but renders a constant frame is a node fault too.
if f.std() < 1e-6:
    print("FATAL: EGL rendered a CONSTANT frame -- node fault, not a renderer result")
    sys.exit(9)
print("EGL OK: 64x64 frame, mean %.2f std %.2f, %d distinct colours"
      % (f.mean(), f.std(), len(np.unique(f.reshape(-1, 3), axis=0))))
