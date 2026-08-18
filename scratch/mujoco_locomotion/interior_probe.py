"""Which INTERIOR pixels disagree, how many, and what are they touching?

The conformance floor is exactness on interior pixels — those whose 3x3 neighbourhood in the
reference is one flat colour — so a single disagreeing pixel is a failure and deserves to be named
rather than tolerated. This counts them over many states and classifies each one as: inside a mover,
within one pixel of a mover silhouette, or neither. It also re-scores at supersample 4, so "is this
a coverage artifact" is answered rather than assumed.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import scipy.ndimage as ndi

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import stable_envs as se                                                    # noqa: E402
from stable_envs.envs import mujoco_locomotion as ML                        # noqa: E402
from stable_envs.testing import conformance as C                            # noqa: E402
from scratch.mujoco_locomotion.parity import mover_mask                     # noqa: E402


def run(name, n=256, size=None, chunk=64):
    cls = se.REGISTRY[name]
    size = int(size or cls.default_size)
    env = cls(batch=chunk, size=size, device="cuda")
    ref = C.reference_env(cls, size)
    rng = np.random.default_rng(0)
    rec = {"env": name, "size": size, "n": n, "by_ss": {}}
    states = env.sample_states(n, rng=rng)
    official = np.concatenate([cls.reference_render(ref, states[i:i + chunk])
                               for i in range(0, n, chunk)])
    mm = mover_mask(cls, ref, states, size)
    near = np.stack([ndi.binary_dilation(x, iterations=1) for x in mm])

    for ss in (2, 4):
        ours = np.concatenate([env.render(states[i:i + chunk], supersample=ss).cpu().numpy()
                               for i in range(0, n, chunk)])
        d = np.abs(ours.astype(np.int16) - official.astype(np.int16)).max(-1)
        m = C.interior_mask(official)                       # (N, H-1, W-1)
        di = d[:, :-1, :-1]
        bad = m & (di > 0)
        inside = mm[:, :-1, :-1] & bad
        edge = near[:, :-1, :-1] & ~mm[:, :-1, :-1] & bad
        elsewhere = bad & ~near[:, :-1, :-1]
        rec["by_ss"][ss] = {
            "interior_px": int(m.sum()),
            "interior_bad": int(bad.sum()),
            "interior_exact_frac": round(float(1 - bad.sum() / max(1, m.sum())), 9),
            "bad_inside_mover": int(inside.sum()),
            "bad_within_1px_of_mover": int(edge.sum()),
            "bad_elsewhere": int(elsewhere.sum()),
            "max_bad_diff": int(di[bad].max()) if bad.any() else 0,
            "all_exact": round(float((d == 0).mean()), 6),
            "mean_abs": round(float(d.mean()), 6)}
    return rec


if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(ML.ENVS)
    out = []
    for nm in names:
        try:
            r = run(nm)
        except Exception as e:                                              # noqa: BLE001
            import traceback
            traceback.print_exc()
            r = {"env": nm, "error": f"{type(e).__name__}: {e}"}
        out.append(r)
        print(json.dumps(r), flush=True)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "interior_probe.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p)
