"""Time the OFFICIAL full-Craftax renderer on the GPU, and refuse to if it is not on the GPU.

Imports **no torch**: it runs in W1's ``$SCR/venvs/se7-craftax-gpu`` (jax[cuda12] 0.11.1 + craftax
1.6.1), which has no torch in it, so this cannot be folded into ``bench.py``. The companion half --
our renderer's throughput on the same card -- runs from ``$SCR/venvs/se7-craftax`` in the SAME job,
which is what "same card, same job" requires without modifying either venv.

**The backend is asserted, not assumed.** ``$SCR/venvs/se7-craftax`` is a ``jax[cpu]`` build, so
``jax.default_backend()`` there is ``'cpu'`` inside a GPU job whatever ``JAX_PLATFORMS`` says, and a
timing taken from it would be a CPU number wearing a GPU label -- inadmissible as a reference for
``R``. This exits non-zero rather than produce one.

Timed as craftax executes it: ``jax.jit``, warm, ``block_until_ready``, compile excluded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="gpu")
    a = ap.parse_args()

    import jax
    from importlib import metadata
    from craftax.craftax.renderer import make_craftax_pixel_renderer
    from craftax.craftax_env import make_craftax_env_from_name

    backend = jax.default_backend()
    devices = [str(d) for d in jax.devices()]
    rec = {"tag": a.tag, "size": a.size, "reps": a.reps, "jax": jax.__version__,
           "craftax": metadata.version("craftax"), "jax_default_backend": backend,
           "jax_devices": devices,
           "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
           "host": os.uname().nodename}
    if backend != "gpu":
        rec["error"] = (f"jax.default_backend() is {backend!r} and the devices are {devices}. A "
                        f"reference timed here is a {backend.upper()} timing, and using it as the "
                        f"GPU reference for R would be inferring a same-card number from something "
                        f"that never touched the card.")
        with open(a.out, "w") as f:
            json.dump(rec, f, indent=2)
        print(json.dumps(rec, indent=2))
        return 1

    env = make_craftax_env_from_name("Craftax-Pixels-v1", auto_reset=False)
    _obs, state = env.reset(jax.random.PRNGKey(0), env.default_params)
    render = make_craftax_pixel_renderer(int(a.size))
    jr = jax.jit(render)

    t0 = time.perf_counter()
    jr(state).block_until_ready()
    rec["compile_seconds"] = time.perf_counter() - t0          # excluded from the timing below
    for _ in range(20):
        jr(state).block_until_ready()

    t0 = time.perf_counter()
    for _ in range(a.reps):
        jr(state).block_until_ready()
    dt = (time.perf_counter() - t0) / a.reps
    rec["official_jitted_fps"] = 1.0 / dt
    rec["official_seconds_per_frame"] = dt

    # the un-jitted function too, for the record -- it is what parity is claimed against
    t0 = time.perf_counter()
    for _ in range(4):
        np.asarray(render(state))
    rec["official_unjitted_fps"] = 4.0 / (time.perf_counter() - t0)
    rec["note"] = ("jitted is the number a speed ratio may use -- the renderer as a user runs it. "
                   "Parity stays against the un-jitted function and is a uint8 claim.")

    with open(a.out, "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
