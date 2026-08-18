"""Benchmark madrona_mjx (the deprecated Madrona batch renderer for MJX).

The bundled `scripts/benchmark.py` measures physics + rendering together and does not
run as shipped (it references an undefined `sys` after the `dr()` helper), so this is a
render-only harness matching the other backends: a batch of qpos goes through
`mjx.forward` for kinematics, then `renderer.render` produces (B, H, W, 4) uint8.

Madrona owns the whole pipeline internally, so unlike nvdiffrast there is no geometry
glue to write -- the cost is entirely in the build, not the integration.

Only one renderer may exist per process (documented, and enforced by a hard failure in
`setCudaHeapSize`), so the caller must run one batch size per invocation.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

from madrona_mjx.renderer import BatchRenderer


def load_model(xml, max_strip=20):
    """madrona_mjx pins jax<0.6, which pins mujoco-mjx to 3.3; the benchmark scenes were
    written by MuJoCo 3.9, whose MJCF schema is a superset. Strip whatever 3.3 rejects,
    one attribute per compile error, and report what was dropped.

    Every stripped attribute is cosmetic (texture colourspace, light type), so geometry
    and camera -- what the throughput number depends on -- are unaffected.
    """
    import re
    import shutil
    import tempfile
    import xml.etree.ElementTree as ET

    tmpdir = tempfile.mkdtemp()
    tmp = os.path.join(tmpdir, os.path.basename(xml))
    shutil.copy(xml, tmp)
    tree = ET.parse(xml)
    for el in tree.iter():
        f = el.get("file")
        if f:
            s = os.path.join(os.path.dirname(xml), os.path.basename(f))
            if os.path.exists(s):
                shutil.copy(s, os.path.join(tmpdir, os.path.basename(f)))

    dropped = []
    for _ in range(max_strip):
        try:
            m = mujoco.MjModel.from_xml_path(tmp)
            if dropped:
                print(json.dumps({"note": "stripped for mujoco 3.3 schema", "attrs": dropped}), flush=True)
            return m
        except ValueError as e:
            hit = re.search(r"unrecognized attribute: '([^']+)'\s*\nElement '([^']+)'", str(e))
            if not hit:
                raise
            attr, elem = hit.groups()
            dropped.append(f"{elem}.{attr}")
            t = ET.parse(tmp)
            for el in t.iter(elem):
                el.attrib.pop(attr, None)
            t.write(tmp)
    raise RuntimeError(f"still failing after stripping {dropped}")


def run(xml, batch, res, iters=20, warmup=3, use_rasterizer=False):
    mjm = load_model(xml)
    m = mjx.put_model(mjm)

    renderer = BatchRenderer(
        m, 0, batch, res, res,
        np.array([0, 1, 2]), None, False, use_rasterizer, None,
    )

    rng = jax.random.PRNGKey(0)
    keys = jax.random.split(rng, batch)

    def init_one(key):
        # kinematics + camlight is everything the renderer reads; the full `forward`
        # drags in a cuSolver factorisation that is not part of what is being measured
        d = mjx.make_data(m)
        d = d.replace(qpos=d.qpos + 0.05 * jax.random.uniform(key, shape=(m.nq,)))
        d = mjx.kinematics(m, d)
        return d

    v_data = jax.vmap(init_one)(keys)
    render_token, rgb, depth = jax.vmap(renderer.init, in_axes=(0, None))(v_data, m)
    jax.block_until_ready(rgb)

    @jax.jit
    def frame(v_data):
        # mjx 3.3's camlight touches `light_type`, a field its own Model lacks; the
        # benchmark cameras are world-fixed, so kinematics already fills cam_xpos/xmat
        v_data = jax.vmap(mjx.kinematics, in_axes=(None, 0))(m, v_data)
        _, rgb, _ = renderer.render(render_token, v_data)
        return rgb

    for _ in range(warmup):
        out = frame(v_data)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = frame(v_data)
    jax.block_until_ready(out)
    dt = time.perf_counter() - t0

    fps = batch * iters / dt
    row = {"backend": "madrona_mjx", "scene": os.path.basename(xml),
           "label": "rasterizer" if use_rasterizer else "raytracer",
           "batch": batch, "res": res, "ms_per_batch": 1e3 * dt / iters,
           "fps": fps, "mpix_s": fps * res * res / 1e6,
           "out_shape": list(out.shape), "out_dtype": str(out.dtype)}
    print(json.dumps(row), flush=True)
    return row, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="antmaze_medium.xml")
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes"))
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--batches", default="64,256,1024")
    ap.add_argument("--rasterizer", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--images", default="")
    args = ap.parse_args()

    print(jax.devices(), flush=True)
    rows = []
    for scene in args.scenes.split(","):
        xml = os.path.join(args.dir, scene)
        if not os.path.exists(xml):
            continue
        for b in [int(x) for x in args.batches.split(",")]:
            try:
                row, out = run(xml, b, args.res, use_rasterizer=args.rasterizer)
                rows.append(row)
                if args.images:
                    import imageio.v2 as imageio
                    os.makedirs(args.images, exist_ok=True)
                    a = np.asarray(out[0])
                    imageio.imwrite(os.path.join(args.images, scene.replace(".xml", "_madrona.png")),
                                    a[..., :3].astype(np.uint8))
            except Exception as e:
                print(json.dumps({"backend": "madrona_mjx", "scene": scene, "batch": b,
                                  "error": f"{type(e).__name__}: {e}"}), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)
