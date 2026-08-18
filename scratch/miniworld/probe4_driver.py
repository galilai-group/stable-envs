"""Probe 4: is a MiniWorld frame a function of the SCENE, or of the GL DRIVER?

This is G1 asked at the level that decides whether a reimplementation is possible at all. T1/T2
measured two terms an analytic renderer would have to reproduce; this asks whether the target is
even well-defined. Two CONFORMING OpenGL implementations are handed the identical scene, the
identical textures and the identical state, and their frames are diffed:

    NVIDIA (libEGL_nvidia)  vs  Mesa llvmpipe (LIBGL_ALWAYS_SOFTWARE=1)

Everything the GL spec leaves implementation-defined lives in this diff and nowhere else:
glGenerateMipmap's downsample filter, the screen-space derivative used to pick a LOD, the
trilinear blend, the GL_POLYGON tessellation of the floor/ceiling n-gons, the GL_QUADS diagonal,
and the 16x MSAA resolve. If the two disagree on a large fraction of pixels then MiniWorld's
frames are a function of (state, scene, DRIVER), the reference is not a specification, and no
analytic renderer can be bit-exact against it -- the same shape of finding as the ALE no-go, where
the picture only exists after a cycle-accurate emulation.

Run as: probe4_driver.py write <tag>   (once per GL implementation, separate processes)
        probe4_driver.py compare       (diff the two)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pyglet

pyglet.options["headless"] = True

OUT = os.environ.get("MW_OUT", "/mnt/vast/home/randall.balestriero/logs/miniworld")
IDS = ["MiniWorld-OneRoom-v0", "MiniWorld-Hallway-v0", "MiniWorld-TMazeLeft-v0",
       "MiniWorld-FourRooms-v0", "MiniWorld-Sidewalk-v0", "MiniWorld-CollectHealth-v0"]

#: Fixed states, written directly, so the two processes render EXACTLY the same scene and no RNG
#: or episode history can differ between them.
POSES = {
    "MiniWorld-OneRoom-v0": [([2.0, 0.0, 3.0], 0.30), ([5.0, 0.0, 5.0], 1.90),
                             ([8.0, 0.0, 2.0], -2.40), ([1.5, 0.0, 8.5], 0.95)],
    "MiniWorld-Hallway-v0": [([0.5, 0.0, 0.0], 0.05), ([4.0, 0.0, 1.0], -0.35),
                             ([8.0, 0.0, -1.0], 0.40), ([2.0, 0.0, 0.5], 2.10)],
    "MiniWorld-TMazeLeft-v0": [([0.0, 0.0, 0.0], 0.10), ([5.0, 0.0, 0.0], -0.20),
                               ([9.0, 0.0, 2.0], 1.30), ([10.0, 0.0, -4.0], 2.80)],
    "MiniWorld-FourRooms-v0": [([3.0, 0.0, 3.0], 0.60), ([8.0, 0.0, 8.0], -1.10),
                               ([12.0, 0.0, 4.0], 2.30), ([5.0, 0.0, 11.0], -2.90)],
    "MiniWorld-Sidewalk-v0": [([2.0, 0.0, 2.0], 0.10), ([4.0, 0.0, 8.0], -0.50),
                              ([3.0, 0.0, 14.0], 0.80), ([2.5, 0.0, 20.0], 1.60)],
    "MiniWorld-CollectHealth-v0": [([5.0, 0.0, 5.0], 0.20), ([10.0, 0.0, 10.0], -1.40),
                                   ([14.0, 0.0, 4.0], 2.60), ([3.0, 0.0, 13.0], -0.70)],
}


def gl_strings():
    """pyglet's glGetString returns LP_c_ubyte, which has no .decode()."""
    import ctypes

    from pyglet.gl import GL_RENDERER, GL_VENDOR, GL_VERSION, glGetString
    return tuple(ctypes.cast(glGetString(k), ctypes.c_char_p).value.decode()
                 for k in (GL_VENDOR, GL_RENDERER, GL_VERSION))


def force_software():
    """Make pyglet take its 'no EGL device' fallback, i.e. eglGetDisplay(EGL_DEFAULT_DISPLAY).

    Under EGL_PLATFORM=surfaceless + the Mesa vendor json that is llvmpipe. It is the only way to
    get a SECOND conforming GL implementation on this node: eglQueryDevicesEXT under the Mesa
    vendor lists only DRM hardware devices (card6-8) and this account is not in the `video` group.
    """
    import warnings

    from pyglet.libs.egl import eglext
    eglext.eglQueryDevicesEXT = lambda max_devices, devices, num_devices: 1
    warnings.simplefilter("ignore")


def render_all(tag):
    if "mesa" in tag:
        force_software()
    import gymnasium as gym
    import miniworld  # noqa: F401
    from pyglet.gl import GLint, glGetIntegerv

    import miniworld.opengl as MO
    force = os.environ.get("MW_FORCE_SAMPLES")
    if force:
        # Attribution run: pin the multisample count on BOTH sides. As shipped the two are not
        # comparable -- NVIDIA honours the requested 16 while llvmpipe caps at GL_MAX_SAMPLES=4 --
        # so without this the driver diff and the MSAA diff are confounded.
        n_force = int(force)
        _FB = MO.FrameBuffer
        class FBn(_FB):                                            # noqa: E306
            def __init__(self, w, h, n=1):
                super().__init__(w, h, n_force)
        MO.FrameBuffer = FBn
        import miniworld.miniworld as MM
        MM.FrameBuffer = FBn

    out = {}
    vendor = None
    ms = GLint()
    for eid in IDS:
        env = gym.make(eid, render_mode="rgb_array")
        u = env.unwrapped
        u.reset(seed=0)
        if vendor is None:
            glGetIntegerv(0x8D57, ms)                              # GL_MAX_SAMPLES_EXT
            vendor = gl_strings()
            print(f"[{tag}] GL_VENDOR   = {vendor[0]}")
            print(f"[{tag}] GL_RENDERER = {vendor[1]}")
            print(f"[{tag}] GL_VERSION  = {vendor[2]}")
            print(f"[{tag}] GL_MAX_SAMPLES = {ms.value}   MW_FORCE_SAMPLES="
                  f"{os.environ.get('MW_FORCE_SAMPLES')}", flush=True)
        frames = []
        for pos, dr in POSES[eid]:
            u.agent.pos = np.array(pos, np.float64)
            u.agent.dir = float(dr)
            u._render_static()
            frames.append(u.render())
        out[eid] = np.stack(frames)
        env.close()
    np.savez_compressed(os.path.join(OUT, f"driver_{tag}.npz"),
                        vendor=np.array(vendor, object), **out)
    print(f"[{tag}] wrote {OUT}/driver_{tag}.npz", flush=True)


def compare():
    import scipy.ndimage as ndi
    a = np.load(os.path.join(OUT, "driver_nvidia.npz"), allow_pickle=True)
    b = np.load(os.path.join(OUT, "driver_mesa.npz"), allow_pickle=True)
    print("NVIDIA:", list(a["vendor"]))
    print("Mesa  :", list(b["vendor"]))
    print()
    print(f"{'id':32s} {'all-exact':>10s} {'within8':>9s} {'mean':>8s} {'max':>6s} "
          f"{'interior%':>10s} {'int-exact':>10s}")
    for eid in IDS:
        A, B = a[eid].astype(np.int16), b[eid].astype(np.int16)
        d = np.abs(A - B).max(-1)
        flat = []
        for f in a[eid]:
            g = f.astype(np.int16)
            fl = (np.abs(np.diff(g, axis=0)).sum(-1)[:, :-1]
                  + np.abs(np.diff(g, axis=1)).sum(-1)[:-1]) == 0
            flat.append(ndi.minimum_filter(fl, size=3))
        m = np.stack(flat)
        di = d[:, :-1, :-1][m]
        ie = float((di == 0).mean()) if di.size else float("nan")
        print(f"{eid:32s} {(d == 0).mean():>10.4f} {(d <= 8).mean():>9.4f} "
              f"{d.mean():>8.3f} {d.max():>6d} {100*m.mean():>9.2f}% {ie:>10.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "write":
        render_all(sys.argv[2])
    else:
        compare()
