"""Where does LunarLander's render time go? Disable one stage at a time and time the rest.

The first landed version rendered at 272 frames/s -- slower than the official pygame renderer -- so
the question is which of the 31 polygons, 114 anti-aliased edges and 10 flag repetitions is paying
for it, before any of them is optimised.
"""
import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import stable_envs as se                                             # noqa: E402
import stable_envs.envs.lunarlander as L                             # noqa: E402
import stable_envs.render.gfxdraw as G                               # noqa: E402

B = int(sys.argv[1]) if len(sys.argv) > 1 else 256
e = se.make("LunarLander", batch=B, device="cuda")
st = torch.as_tensor(e.sample_states(B, rng=np.random.default_rng(0)), device="cuda")


def bench(label, n=5):
    for _ in range(2):
        e.render(st)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        e.render(st)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    print(f"  {label:34s} {B/dt:9,.0f} f/s   {1000*dt:7.1f} ms/batch")


bench("everything")

orig_flags = L.LunarLander._flags
L.LunarLander._flags = lambda *a, **k: None
bench("no flags at all")
L.LunarLander._flags = orig_flags

orig_aa = G.aaline
G.aaline = lambda *a, **k: None
bench("no anti-aliased edges")
G.aaline = orig_aa

orig_clip = G.clip_line
G.clip_line = lambda x1, y1, x2, y2, W, H, iters=6: (
    torch.ones_like(x1, dtype=torch.bool), x1, y1, x2, y2)
bench("aa on, clip_line stubbed out")
G.clip_line = orig_clip

orig_p = L.LunarLander._particle
L.LunarLander._particle = lambda *a, **k: None
bench("no particles")
L.LunarLander._particle = orig_p

orig_b = L.LunarLander._body
L.LunarLander._body = lambda *a, **k: None
bench("no rigid bodies")
L.LunarLander._body = orig_b

G.aaline = lambda *a, **k: None
L.LunarLander._flags = lambda *a, **k: None
bench("fills only (no aa, no flags)")
G.aaline = orig_aa
L.LunarLander._flags = orig_flags
