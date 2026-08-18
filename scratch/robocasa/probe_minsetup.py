"""How low can MJWarp's per-scene setup go? Price the last two terms: ``put_model`` and ``put_data``.

``probe_fastsetup.py`` removed the big one -- ``create_render_context`` fell 8.98 s -> 1.07 s once
the 388 MB texture blob stopped being re-uploaded once per texture. What is left is
``put_model`` (3.98 s) + ``put_data`` (3.11 s) + the MJCF compile (1.68 s).

The question for those two is whether their cost is BYTES or FIELD COUNT. ``put_model`` moves about
100 MB, which an H200 ingests in ~10 ms, so 3.98 s cannot be the transfer -- it is ~500 individual
``wp.array`` constructions in Python. If that is right the cost is nearly independent of the model,
and the render path (which reads about twenty ``Model`` fields and seven ``Data`` fields) can skip
almost all of it. This measures it the only way that settles it: run ``put_model`` on a two-geom
model and see whether it is still seconds.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("WARP_CACHE_PATH", "/mnt/vast/home/randall.balestriero/.cache/warp")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

TRIVIAL = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="2 2 .1"/>
    <body pos="0 0 .5"><freejoint/><geom type="box" size=".1 .1 .1" rgba=".8 .2 .2 1"/></body>
  </worldbody>
  <visual><global offwidth="256" offheight="256"/></visual>
</mujoco>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=os.path.join(OUT, "corpus", "L52S03", "ep000.xml"))
    ap.add_argument("--nworld", type=int, default=256)
    ap.add_argument("--out", default=os.path.join(OUT, "minsetup.json"))
    args = ap.parse_args()

    import mujoco
    import warp as wp
    import mujoco_warp as mjw

    rec = {"mujoco": mujoco.__version__, "nworld": args.nworld}

    # --- trivial model: whatever this costs is FIXED overhead, not transfer ---------------------
    mt = mujoco.MjModel.from_xml_string(TRIVIAL)
    dt = mujoco.MjData(mt); mujoco.mj_forward(mt, dt)
    mjw.put_model(mt); wp.synchronize()                     # warm the warp module loads
    t0 = time.perf_counter(); mjw.put_model(mt); wp.synchronize()
    rec["trivial_put_model_s"] = time.perf_counter() - t0
    t0 = time.perf_counter(); mjw.put_data(mt, dt, nworld=args.nworld); wp.synchronize()
    rec["trivial_put_data_s"] = time.perf_counter() - t0
    if hasattr(mjw, "make_data"):
        t0 = time.perf_counter(); mjw.make_data(mt, nworld=args.nworld); wp.synchronize()
        rec["trivial_make_data_s"] = time.perf_counter() - t0
    rec["trivial_model"] = dict(ngeom=int(mt.ngeom), nbody=int(mt.nbody), nq=int(mt.nq))

    # --- the real kitchen ------------------------------------------------------------------------
    mujoco.MjModel.from_xml_path(args.xml)
    t0 = time.perf_counter(); mjm = mujoco.MjModel.from_xml_path(args.xml)
    rec["kitchen_compile_s"] = time.perf_counter() - t0
    mjd = mujoco.MjData(mjm); mujoco.mj_forward(mjm, mjd)
    t0 = time.perf_counter(); mjw.put_model(mjm); wp.synchronize()
    rec["kitchen_put_model_s"] = time.perf_counter() - t0
    t0 = time.perf_counter(); mjw.put_data(mjm, mjd, nworld=args.nworld); wp.synchronize()
    rec["kitchen_put_data_s"] = time.perf_counter() - t0
    if hasattr(mjw, "make_data"):
        t0 = time.perf_counter(); mjw.make_data(mjm, nworld=args.nworld); wp.synchronize()
        rec["kitchen_make_data_s"] = time.perf_counter() - t0
    rec["kitchen_model"] = dict(ngeom=int(mjm.ngeom), nbody=int(mjm.nbody), nq=int(mjm.nq),
                                nmesh=int(mjm.nmesh), ntex=int(mjm.ntex))

    # --- how much of `put_model` is per-FIELD Python overhead? ------------------------------------
    import dataclasses
    from mujoco_warp._src import types as T
    fields = [f.name for f in dataclasses.fields(T.Model)]
    rec["model_field_count"] = len(fields)
    rec["put_model_s_per_field_trivial"] = rec["trivial_put_model_s"] / len(fields)
    rec["put_model_bytes_kitchen_MB"] = float(
        sum(getattr(mjm, n).nbytes for n in dir(mjm)
            if isinstance(getattr(mjm, n, None), np.ndarray)) / 1e6)

    # --- MJCF compile: how much of it is the collision half? --------------------------------------
    stripped = os.path.join(OUT, "stripped_scene.xml")
    if os.path.exists(stripped):
        mujoco.MjModel.from_xml_path(stripped)
        t0 = time.perf_counter(); ms = mujoco.MjModel.from_xml_path(stripped)
        rec["stripped_compile_s"] = time.perf_counter() - t0
        rec["stripped_model"] = dict(ngeom=int(ms.ngeom), nmesh=int(ms.nmesh), ntex=int(ms.ntex))
        t0 = time.perf_counter(); mjw.put_model(ms); wp.synchronize()
        rec["stripped_put_model_s"] = time.perf_counter() - t0

    print(json.dumps(rec, indent=1), flush=True)
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
