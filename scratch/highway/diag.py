"""Where do Racetrack/Parking disagree with the reference, and which primitive should have painted it?

Classifies every differing pixel by the colour the REFERENCE has there, which names the primitive
directly: grey is the road plane, white a lane marking, yellow an obstacle, blue the goal landmark,
green the ego, red a crash, (60,60,60) the outline or a tyre.

    python scratch/highway/diag.py Parking [n]
"""
import os
import sys

os.environ["SDL_VIDEODRIVER"] = "offscreen"
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import stable_envs as se                                             # noqa: E402
from stable_envs.testing import conformance as C                     # noqa: E402

NAME = sys.argv[1] if len(sys.argv) > 1 else "Parking"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
PALETTE = {(100, 100, 100): "road", (255, 255, 255): "marking", (200, 200, 0): "obstacle",
           (100, 200, 255): "landmark/traffic", (50, 200, 0): "ego", (255, 100, 100): "crashed",
           (60, 60, 60): "outline/tyre", (73, 294, 375): "?"}

e = se.make(NAME, batch=8, device="cuda")
st = e.sample_states(N, rng=np.random.default_rng(0))
ref = C.reference_env(type(e), e.size)
off = type(e).reference_render(ref, st)
ours = e.render(st).cpu().numpy()
d = np.abs(ours.astype(np.int16) - off.astype(np.int16)).max(-1)
print(f"{NAME}: {int((d > 0).sum())} differing px over {N} frames "
      f"({int((d > 0).sum()) / N:.1f}/frame), all-exact {float((d == 0).mean()):.6f}")

m = C.interior_mask(off)
di = d[:, :-1, :-1]
print(f"interior: {int((di[m] > 0).sum())} differing interior px "
      f"({float((di[m] == 0).mean()):.6f} exact)")

ys, xs = np.where(d.reshape(-1, *d.shape[1:]).any(0))
print(f"differing pixels span rows {ys.min()}..{ys.max()} cols {xs.min()}..{xs.max()} "
      f"(frame {off.shape[1]}x{off.shape[2]})")

bad = d > 0
refc = off[bad]
ourc = ours[bad]
pairs = {}
for a, b in zip(map(tuple, refc.tolist()), map(tuple, ourc.tolist())):
    pairs[(a, b)] = pairs.get((a, b), 0) + 1
print("\ntop (reference colour -> ours) pairs:")
for (a, b), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:12]:
    print(f"  {n:6d}  ref {a} {PALETTE.get(a,''):18s} -> ours {b} {PALETTE.get(b,'')}")

# interior only
mi = np.zeros_like(bad)
mi[:, :-1, :-1] = m
badi = bad & mi
pairs = {}
for a, b in zip(map(tuple, off[badi].tolist()), map(tuple, ours[badi].tolist())):
    pairs[(a, b)] = pairs.get((a, b), 0) + 1
print("\ninterior only:")
for (a, b), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:12]:
    print(f"  {n:6d}  ref {a} {PALETTE.get(a,''):18s} -> ours {b} {PALETTE.get(b,'')}")
    zs, zy, zx = np.where(badi & (off == np.array(a, np.uint8)).all(-1)
                          & (ours == np.array(b, np.uint8)).all(-1))
    print(f"          e.g. frame {zs[0]} at (row {zy[0]}, col {zx[0]}); rows {zy.min()}..{zy.max()} "
          f"cols {zx.min()}..{zx.max()}")
