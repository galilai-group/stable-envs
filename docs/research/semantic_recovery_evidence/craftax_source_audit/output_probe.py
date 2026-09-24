"""Measure source cached output versus pure output at the exact restored engine State."""

import importlib.util
import json
from pathlib import Path

import jax
import numpy as np

HERE = Path(__file__).resolve().parent
sp = importlib.util.spec_from_file_location("research", HERE / "replay_probe.py")
r = importlib.util.module_from_spec(sp)
sp.loader.exec_module(r)


def frame(x):
    return (np.clip(np.asarray(x), 0, 1) * 255).astype(np.uint8)


results = {}
for classic in [True, False]:
    clsname = "Craftax" + ("Classic" if classic else "") + "PixelsWrapper"
    a = getattr(r.mod, clsname)(backend="cpu", render_mode="rgb_array")
    a.reset(seed=17)
    base = a._state
    out = []
    obsfn = jax.jit(a._env.get_obs)
    for kind in ["mine_tree", "craft", "combat", "damage", "night", "terminal", "near_limit", "stochastic_grass"]:
        a._state, act = r.fixture(base, classic, kind, a._params)
        a._key = jax.random.PRNGKey(2026)
        a.step(act)
        cached = a._last_obs.copy()
        pure = np.asarray(obsfn(a._state))
        f1, f2 = frame(cached), frame(pure)
        row = {
            "fixture": kind,
            "float_exact": bool(np.array_equal(cached, pure)),
            "max_float_abs": float(np.abs(cached - pure).max()),
            "frame_exact": bool(np.array_equal(f1, f2)),
            "different_channels": int(np.count_nonzero(f1 != f2)),
            "max_uint8_abs": int(np.abs(f1.astype(int) - f2.astype(int)).max()),
        }
        out.append(row)
        if not row["frame_exact"]:
            np.savez_compressed(
                HERE / (clsname + "-" + kind + "-output-boundary.npz"),
                cached=cached,
                pure=pure,
                cached_frame=f1,
                pure_frame=f2,
            )
    results[clsname] = out
    print(clsname, [(x["fixture"], x["different_channels"]) for x in out], flush=True)
(HERE / "output-boundary.json").write_text(json.dumps(results, indent=2) + "\n")
print("CRAFTAX_OUTPUT_BOUNDARY_MEASURED", flush=True)
