"""Does ``madrona_mjx`` run at all on this cluster, and if so on which scene?

``docs/research/backend3d.md`` records it as "builds, does not run": the raytracer path reaches
``MWCudaExecutor::run -> an illegal memory access was encountered`` at B=8, 64x64. That was measured
on an OGBench scene. Before spending a day on it this re-runs the failure from the smallest possible
input upwards, so the error is attributed to something -- scene size, batch, resolution, or the build
itself -- rather than left as "does not run".

Ladder, each step in the SAME process so a step that dies takes the ladder with it and the last line
printed is the last thing that worked:

  1. import + executor construction on a two-geom MJCF
  2. one render at B=2, 32 px
  3. the same at B=64, 256 px
  4. the humanoid scene, then the kitchen, if either survives

Anything past step 1 that survives is worth a fidelity comparison; nothing here is a throughput
number until it draws the right picture.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

TWO_GEOM = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="5 5 0.1" rgba="0.6 0.6 0.6 1"/>
    <body pos="0 0 0.5">
      <joint type="free"/>
      <geom type="box" size="0.2 0.2 0.2" rgba="0.9 0.2 0.2 1"/>
    </body>
    <camera name="cam" pos="2 2 1.5" xyaxes="-1 1 0 -0.3 -0.3 1"/>
  </worldbody>
</mujoco>
"""


def step(label, fn):
    print(f"--- {label}", flush=True)
    t0 = time.perf_counter()
    try:
        out = fn()
    except BaseException as e:                                       # noqa: BLE001
        print(json.dumps({"step": label, "error": f"{type(e).__name__}: {e}"}), flush=True)
        raise
    print(json.dumps({"step": label, "ok": True, "seconds": time.perf_counter() - t0,
                      "result": out if isinstance(out, (int, float, str, type(None))) else
                      type(out).__name__}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--rasterizer", action="store_true")
    ap.add_argument("--groups", default="0,1,2",
                    type=lambda x: tuple(int(v) for v in x.split(",")))
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--dump", default="")
    ap.add_argument("--no-collide", action="store_true")
    a = ap.parse_args()

    import jax
    import jax.numpy as jp                                            # noqa: F401
    import mujoco
    import numpy as np
    from mujoco import mjx
    print(json.dumps({"jax": jax.__version__, "mujoco": mujoco.__version__,
                      "devices": [str(d) for d in jax.devices()]}), flush=True)

    from madrona_mjx.renderer import BatchRenderer
    print("imported BatchRenderer", flush=True)

    if a.xml:
        # mujoco 3.3 (what madrona_mjx pins through jax<0.6) rejects attributes MuJoCo 3.11 wrote.
        # ``bench_madrona.load_model`` strips them one compile error at a time and reports which.
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "backend3d"))
        from bench_madrona import load_model
        mjm = load_model(a.xml)
    else:
        mjm = mujoco.MjModel.from_xml_string(TWO_GEOM)
    print(json.dumps({"ngeom": int(mjm.ngeom), "nmesh": int(mjm.nmesh),
                      "ntex": int(mjm.ntex), "ncam": int(mjm.ncam)}), flush=True)
    if a.no_collide:
        # ``madrona_mjx`` takes an ``mjx.Model``, and ``mjx.put_model`` refuses any model MJX cannot
        # SIMULATE even when the caller only wants to draw it: a RoboCasa kitchen dies on
        # "(mjGEOM_CYLINDER, mjGEOM_MESH) collisions not implemented". Clearing contype/conaffinity
        # removes the pairs from the enumeration and changes no pixel -- nothing here steps physics.
        mjm.geom_contype[:] = 0
        mjm.geom_conaffinity[:] = 0
    m = mjx.put_model(mjm)

    # ONE renderer per process, and that is a hard limit rather than a convention: a second
    # construction aborts the interpreter in ``madrona::setCudaHeapSize`` (cuda_exec.cpp:283) with no
    # catchable exception. An earlier version of this ladder built a throwaway renderer to time the
    # construction and then built the real one, and died there -- which is what "does not run" looked
    # like the first time.
    renderer = step("construct", lambda: BatchRenderer(
        m, a.cam, a.batch, a.res, a.res, np.array(list(a.groups)), None, False, a.rasterizer,
        None))

    keys = jax.random.split(jax.random.PRNGKey(0), a.batch)

    def init_one(key):
        d = mjx.make_data(m)
        d = d.replace(qpos=d.qpos + 0.02 * jax.random.uniform(key, shape=(m.nq,)))
        return mjx.kinematics(m, d)

    v_data = step("vmap-kinematics", lambda: jax.vmap(init_one)(keys))
    tok, rgb, depth = step("renderer.init",
                           lambda: jax.vmap(renderer.init, in_axes=(0, None))(v_data, m))
    jax.block_until_ready(rgb)
    print(json.dumps({"rgb_shape": list(rgb.shape)}), flush=True)

    @jax.jit
    def frame(v):
        v = jax.vmap(mjx.kinematics, in_axes=(None, 0))(m, v)
        _, out, _ = renderer.render(tok, v)
        return out

    out = step("first-render", lambda: np.asarray(jax.block_until_ready(frame(v_data))).mean())
    if a.dump:
        import imageio.v2 as imageio
        a0 = np.asarray(frame(v_data))[0]
        imageio.imwrite(a.dump, a0[..., :3].astype(np.uint8))
        print(json.dumps({"dumped": a.dump, "shape": list(a0.shape)}), flush=True)
    for _ in range(3):
        o = frame(v_data)
    jax.block_until_ready(o)
    t0 = time.perf_counter()
    n = 20
    for _ in range(n):
        o = frame(v_data)
    jax.block_until_ready(o)
    dt = time.perf_counter() - t0
    print(json.dumps({"backend": "madrona_mjx",
                      "label": "rasterizer" if a.rasterizer else "raytracer",
                      "xml": a.xml or "two-geom", "batch": a.batch, "res": a.res,
                      "ms_per_batch": 1e3 * dt / n, "fps": a.batch * n / dt,
                      "mean_pixel": float(out)}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
