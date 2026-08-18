"""LunarLander parity against the official renderer, decomposed by which primitive disagrees.

    python scratch/lunarlander/validate.py [n_states]

Prints interior and all-pixel agreement, then classifies every differing pixel by the colour the
REFERENCE has there -- which names the primitive: white is the sky or a flagpole, black the terrain,
(128,102,230) the lander body, (77,77,128) the ``pygame.draw.aalines`` outline this module does not
reimplement, (204,204,0) a flag, anything warm an exhaust particle.
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import stable_envs as se                                             # noqa: E402
from stable_envs.testing import conformance as C                     # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
NAMED = {(255, 255, 255): "sky/pole", (0, 0, 0): "terrain", (128, 102, 230): "lander body",
         (77, 77, 128): "aalines outline", (204, 204, 0): "flag"}

e = se.make("LunarLander", batch=8, device="cuda")
print(f"state_dim {e.state_dim}  names {len(e.state_names)}")
st = e.sample_states(N, rng=np.random.default_rng(0))
from stable_envs.envs.lunarlander import PART_OFF                    # noqa: E402
alive = st[:, PART_OFF + 2::3] >= 0            # ttl is stored AS DRAWN, so no decrement here
print(f"particles drawn per state: min {alive.sum(1).min()} max {alive.sum(1).max()} "
      f"mean {alive.sum(1).mean():.2f}")
ref = C.reference_env(type(e), e.size)
off = type(e).reference_render(ref, st)
ours = e.render(st).cpu().numpy()
print(f"frame {ours.shape} vs official {off.shape}")
r = C.pixel_agreement(ours, off)
print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})

d = np.abs(ours.astype(np.int16) - off.astype(np.int16)).max(-1)
bad = d > 0
print(f"\n{int(bad.sum())} differing px over {N} frames ({int(bad.sum())/N:.1f}/frame)")
pairs = {}
for a, b in zip(map(tuple, off[bad].tolist()), map(tuple, ours[bad].tolist())):
    pairs[(a, b)] = pairs.get((a, b), 0) + 1
for (a, b), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:14]:
    print(f"  {n:7d}  ref {str(a):18s} {NAMED.get(a,''):16s} -> ours {str(b):18s} {NAMED.get(b,'')}")

# how much is the un-ported pygame.draw.aalines outline? Every pixel whose reference colour is
# color2, or whose OURS colour is the body colour where the reference has something else nearby.
ours_body = (ours == np.array((128, 102, 230), np.uint8)).all(-1)
ref_body = (off == np.array((128, 102, 230), np.uint8)).all(-1)
outline = int((bad & ours_body & ~ref_body).sum())
fringe = int((bad & ((off == 0).all(-1) | (ours == 0).all(-1))).sum())
print(f"\nbreakdown: {outline} px ({outline/N:.1f}/frame) are ours=body where the reference has "
      f"something else -- the un-ported pygame.draw.aalines outline")
print(f"           {fringe} px ({fringe/N:.1f}/frame) involve pure black, i.e. the terrain's own "
      f"boundary: the pygame.draw.polygon vs gfxdraw.filled_polygon fill rule")

m = C.interior_mask(off)
di = d[:, :-1, :-1]
print(f"interior: {int((di[m] > 0).sum())} differing, exact {float((di[m] == 0).mean()):.7f}")

# determinism + throughput
a = e.render(st).cpu().numpy()
print(f"determinism: {np.array_equal(a, ours)}")
import time                                                         # noqa: E402
B = 256
big = e.sample_states(B, rng=np.random.default_rng(1))
eb = se.make("LunarLander", batch=B, device="cuda")
t = torch.as_tensor(big, device="cuda")
for _ in range(2):
    eb.render(t)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    eb.render(t)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / 5
print(f"GPU render B={B}: {B/dt:,.0f} frames/s ({B*400*600*3/dt/1e9:.2f} GB/s)")
t0 = time.perf_counter()
for i in range(20):
    type(e).reference_render(ref, st[i:i + 1])
print(f"official CPU renderer: {20/(time.perf_counter()-t0):,.0f} frames/s")
