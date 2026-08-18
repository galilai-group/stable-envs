"""Re-measure Fetch's per-scene interior agreement after the gl_rounding CUDA-graph fix.

The class comment quotes 0.991894 / 0.992075 / 0.992240 / 0.994751, measured before the capture was
moved inside `with MW.gl_rounding()`. That fix changed 15.8% of Fetch's channels, so those four numbers
describe a renderer that no longer exists. The floor (0.99) is conservative either way; the comment is
what is stale, and a stale MEASURED value is worse than no value because it reads as current evidence.
"""
import numpy as np, torch, stable_envs as se
from stable_envs.testing import conformance as C

names = [n for n in sorted(se.REGISTRY) if n.startswith("Fetch")][:8]
print(f"{len(names)} Fetch cells\n{'env':32s} {'interior':>10s} {'all-px':>9s} {'mean':>7s}")
worst = 1.0
for n in names:
    cls = se.REGISTRY[n]
    env = se.make(n, batch=64, device="cuda")
    ref = cls.reference_env(size=env.size)
    st = env.sample_states(64, rng=np.random.default_rng(0))
    off = cls.reference_render(ref, st)
    ours = env.render(st).cpu().numpy()
    r = C.pixel_agreement(ours, off)
    ie = r["interior_exact_frac"]
    if ie is not None: worst = min(worst, ie)
    print(f"{n:32s} {('%.6f'%ie) if ie is not None else '   n/a':>10s} "
          f"{r['exact_frac']:9.5f} {r['mean_abs']:7.4f}", flush=True)
    del env; torch.cuda.empty_cache()
print(f"\nworst interior across cells: {worst:.6f}   (declared floor 0.99)")
