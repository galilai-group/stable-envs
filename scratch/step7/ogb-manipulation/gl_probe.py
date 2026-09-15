"""Pick a WORKING MuJoCo GL backend on this node, and refuse to measure without one.

Two different failures wear the same traceback and must not be confused:

* **A node fault.** A context that will not create, or that creates and paints one flat colour.
  A measurement taken there reads as a RENDERER finding when it is a broken node.
  NOT a fault: an `EGLError` raised from `mujoco.egl.GLContext.__del__` at interpreter exit. That
  is teardown noise after a successful render -- the reported `gpu2104` failure was exactly this --
  which is why the verdict below is the child's EXIT CODE plus a non-constant frame, never the
  presence of a traceback in its output.
* **The wrong backend for the partition.** A `batch` node has no GPU, so EGL's
  `PLATFORM_DEVICE` path cannot initialise at all -- `Cannot initialize a EGL device display`.
  That is not a fault; Mesa's surfaceless EGL or `osmesa` (llvmpipe) is what renders there, which
  is exactly what the Step-6 CPU scripts probe for.

So: try each candidate in order, render a real lit scene, and accept the first that returns a
NON-CONSTANT frame -- a context that creates but paints one colour is a fault too. Prints
`MUJOCO_GL=<name>` on the last line for the caller to `eval`, and exits 9 if nothing works.

    eval "$(python gl_probe.py --require egl)"      # GPU node: EGL or nothing
    eval "$(python gl_probe.py egl osmesa glfw)"    # CPU node: first that works
"""
from __future__ import annotations

import argparse
import os
import sys

XML = """<mujoco><worldbody>
  <light pos='0 0 2' dir='0 0 -1'/><geom type='plane' size='1 1 .1' rgba='.3 .4 .5 1'/>
  <body pos='0 0 .3'><freejoint/><geom type='box' size='.1 .1 .1' rgba='1 0 0 1'/></body>
  <camera name='c' pos='1 0 1' xyaxes='0 1 0 -.7 0 .7'/>
</worldbody></mujoco>"""


def try_backend(name: str) -> tuple[bool, str]:
    """Render 64x64 under ``name`` in a CHILD process.

    A child, because a failed GL init leaves the OpenGL bindings in a state the next attempt
    inherits; probing three backends in one interpreter makes the second answer depend on the
    first.
    """
    import subprocess
    code = (
        "import os,sys;os.environ['MUJOCO_GL']=%r\n"
        "import mujoco,numpy as np\n"
        "m=mujoco.MjModel.from_xml_string(%r);d=mujoco.MjData(m);mujoco.mj_forward(m,d)\n"
        "r=mujoco.Renderer(m,64,64);r.update_scene(d,camera='c');f=r.render()\n"
        "try:\n"
        "    from OpenGL import GL;rn=GL.glGetString(GL.GL_RENDERER).decode()\n"
        "except Exception:\n"
        "    rn='unreadable'\n"
        "assert f.std()>1e-6,'CONSTANT FRAME'\n"
        "print('RENDERER=%%s MEAN=%%.2f STD=%%.2f'%%(rn,f.mean(),f.std()))\n"
    ) % (name, XML)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    out = (p.stdout + p.stderr).strip().replace("\n", " | ")[-400:]
    return p.returncode == 0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", nargs="*", default=None)
    ap.add_argument("--require", default=None,
                    help="accept only this backend; anything else is a node fault")
    a = ap.parse_args()
    cands = [a.require] if a.require else (a.candidates or ["egl", "osmesa", "glfw"])
    for c in cands:
        ok, msg = try_backend(c)
        print(f"# {c}: {'OK' if ok else 'FAIL'} {msg}", file=sys.stderr)
        if ok:
            print(f"export MUJOCO_GL={c}")
            return 0
    print(f"# FATAL: no working GL backend on {os.uname().nodename} among {cands}", file=sys.stderr)
    print("export MUJOCO_GL=__NONE__")
    return 9


if __name__ == "__main__":
    sys.exit(main())
