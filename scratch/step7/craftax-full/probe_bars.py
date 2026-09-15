"""The two bars a Step-2/Step-3 tier is measured against, for full Craftax at its registered size.

* ``official f/s`` -- the denominator of ``S_cpu``. Measured **un-jitted** (the reference this
  renderer is certified against, per W1's ruling) *and* jitted, because the jitted number is what a
  reader will otherwise assume and the two differ by orders of magnitude.
* ``decode f/s per core`` -- one core of JPEG-q90 decode of **this env's own frame**, the Step-3
  ``bench_vs_decode.decode_baseline`` protocol, which is the denominator of ``cores_cpu``.

From those two, the absolute f/s a renderer must reach for ``S_cpu >= 5`` (CPU_FAST) and
``cores_cpu >= 1``.

Apple arm64, one torch thread. **Not certification numbers** -- the Oscar batch-node sweep is.

Writes ``phase0/bars.json``.
"""
import io
import json
import os
import pickle
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "..", "examples", "results", "step7", "craftax-full", "phase0")
POOL = os.path.join(HERE, "_pool.pkl")


def decode_fps(frame, quality=90, min_frames=40):
    from PIL import Image
    h, w = int(frame.shape[0]), int(frame.shape[1])
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    blob = buf.getvalue()
    n = max(min_frames, int(4e7 / max(h * w, 1)))
    for _ in range(20):
        np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
    t0 = time.perf_counter()
    for _ in range(n):
        np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
    dt = (time.perf_counter() - t0) / n
    return 1.0 / dt, len(blob), n


def main():
    import jax
    import torch
    from craftax.craftax import constants as C
    from craftax.craftax.renderer import make_craftax_pixel_renderer

    torch.set_num_threads(1)
    pool = pickle.load(open(POOL, "rb"))
    states = (pool["rollout"][:16] + pool["level_shifted"])[:16]
    rep = {"host": platform.platform(), "machine": platform.machine(),
           "note": "one core, one torch thread, Apple arm64 -- NOT a certification number",
           "jax": jax.__version__, "threads": 1}

    out = {}
    for t in (C.BLOCK_PIXEL_SIZE_AGENT, C.BLOCK_PIXEL_SIZE_IMG):
        t = int(t)
        render = make_craftax_pixel_renderer(t)
        jrender = jax.jit(render)
        shape = ((C.OBS_DIM[0] + C.INVENTORY_OBS_HEIGHT) * t, C.OBS_DIM[1] * t, 3)

        # un-jitted: expensive, so few reps
        n = 4
        render(states[0]).block_until_ready()
        t0 = time.perf_counter()
        for i in range(n):
            render(states[i % len(states)]).block_until_ready()
        dt_un = (time.perf_counter() - t0) / n
        # jitted
        jrender(states[0]).block_until_ready()
        m = 200
        t0 = time.perf_counter()
        for i in range(m):
            jrender(states[i % len(states)]).block_until_ready()
        dt_j = (time.perf_counter() - t0) / m

        frame = np.round(np.asarray(render(states[0]))).clip(0, 255).astype(np.uint8)
        dfps, nbytes, ndec = decode_fps(frame)

        rec = {
            "tile_px": t, "frame_shape": list(shape),
            "frame_bytes_uint8": int(np.prod(shape)),
            "official_unjitted_fps": 1.0 / dt_un,
            "official_jitted_fps": 1.0 / dt_j,
            "jit_speedup": dt_un / dt_j,
            "decode_fps_per_core": dfps,
            "jpeg_q90_bytes": int(nbytes),
            "decode_reps": int(ndec),
            "target_fps_for_S_cpu_ge_5_vs_unjitted": 5.0 / dt_un,
            "target_fps_for_S_cpu_ge_5_vs_jitted": 5.0 / dt_j,
            "target_fps_for_cores_cpu_ge_1": dfps,
        }
        out[str(t)] = rec
    rep["bars"] = out

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "bars.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    sys.exit(main())
