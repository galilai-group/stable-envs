"""Is the ray-cast movers' residual a SYSTEMATIC bias, or per-pixel shading disagreement?

The distinction decides whether it is worth chasing. ``fetch.py``'s largest single fidelity win came
from a signed look at exactly this: the difference on flat interior pixels was ``{0: 86.8%, -1: 13.1%}``
and nothing else, which is one float quantised two ways and is not predicted by any shading
hypothesis. If Meta-World's mover residual has that shape there is a rule to fix; if it is a symmetric
spread of several levels it is MuJoCo's per-vertex Gouraud interpolation against a per-pixel analytic
shader, which is the limit ``docs/research/ogbench.md`` refuted three ways.

Measured on the INTERIOR of the movers (the mask eroded by one pixel), so a silhouette -- where two
rasterizers are entitled to disagree about coverage -- cannot be mistaken for a shading term.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def run(task: str, size: int, n: int) -> dict:
    import scipy.ndimage as ndi
    import torch

    import stable_envs as se
    from stable_envs.envs.metaworld import _build_bundle, _class_name, _context
    from stable_envs.testing import conformance as C

    cls = se.REGISTRY[_class_name(task)]
    env = cls(batch=n, size=size, device="cuda")
    b = _build_bundle(task)
    states = env.sample_states(n, rng=np.random.default_rng(0))
    ref = C.reference_env(cls, size)
    official = cls.reference_render(ref, states).astype(np.int16)
    ours = env.render(states).cpu().numpy().astype(np.int16)
    c = _context(task, size, n)
    mv = torch.isin(c.seg[:n], torch.as_tensor(b.movers, device="cuda").to(c.seg.dtype)).cpu().numpy()
    inner = np.stack([ndi.binary_erosion(f, iterations=1) for f in mv])

    sd = (official - ours)[inner]
    vals, counts = np.unique(sd, return_counts=True)
    order = np.argsort(-counts)[:12]
    return {"task": task, "size": size, "n": n,
            "mover_frac": round(float(mv.mean()), 5),
            "mover_interior_frac": round(float(inner.mean()), 5),
            "abs_mean_on_mover_interior": round(float(np.abs(sd).mean()), 3),
            "exact_frac_on_mover_interior": round(float((sd == 0).mean()), 4),
            "within1": round(float((np.abs(sd) <= 1).mean()), 4),
            "within4": round(float((np.abs(sd) <= 4).mean()), 4),
            "signed_hist": [[int(vals[i]), round(float(counts[i] / sd.size), 5)] for i in order]}


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "reach-v3"
    print("RESULT " + json.dumps(run(task, int(sys.argv[2]) if len(sys.argv) > 2 else 480,
                                     int(sys.argv[3]) if len(sys.argv) > 3 else 8)), flush=True)
