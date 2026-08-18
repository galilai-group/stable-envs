"""Can crafter's texture resize be reproduced without PIL, and how big is the source art?

Two questions, both prerequisites for shipping a Crafter atlas that does not import crafter:
  1. what do the 56 assets weigh at their native size, and what would an embedded blob cost;
  2. does the standard nearest-neighbour rule ``floor((i+0.5)*src/dst)`` reproduce
     ``PIL.Image.resize(..., NEAREST)`` for every (src, dst) crafter ever asks for.
"""
from __future__ import annotations

import base64
import zlib

import numpy as np
from PIL import Image

import crafter
from crafter import constants, engine

env = crafter.Env(seed=0)
tex = env._textures
orig = tex._originals
shapes = {k: v.shape for k, v in orig.items()}
print(f"{len(orig)} assets; distinct shapes: {sorted({v for v in shapes.values()})}")
for k in sorted(shapes):
    if shapes[k][:2] != (16, 16):
        print("  non-16x16:", k, shapes[k])

# what an embedded blob costs, per channel count
tot = 0
for k, v in orig.items():
    tot += v.nbytes
print(f"raw bytes {tot}")
names = sorted(orig)
rgba = np.zeros((len(names), 16, 16, 4), np.uint8)
for i, n in enumerate(names):
    a = orig[n]
    rgba[i, ..., :a.shape[-1]] = a
    if a.shape[-1] == 3:
        rgba[i, ..., 3] = 255
blob = base64.b64encode(zlib.compress(rgba.tobytes(), 9))
print(f"packed (N,16,16,4) uint8 -> zlib+b64 = {len(blob)} bytes")

# the resize rule
bad = []
for src in sorted({s[0] for s in shapes.values()} | {s[1] for s in shapes.values()}):
    for dst in range(1, 65):
        ix = ((np.arange(dst) + 0.5) * src / dst).astype(np.int64).clip(0, src - 1)
        for n in names:
            a = orig[n]
            want = np.array(Image.fromarray(a).resize((dst, dst), resample=Image.NEAREST))
            got = a[ix][:, ix]
            if not np.array_equal(want, got):
                bad.append((n, src, dst, int(np.abs(want.astype(int) - got.astype(int)).max())))
            break                                   # one asset per (src,dst) is enough to falsify
print(f"nearest rule floor((i+0.5)*src/dst): {len(bad)} mismatches; first {bad[:3]}")

# full check on every asset at the sizes crafter actually asks for (unit, 0.8*unit, 0.6*unit)
bad2 = []
for size in range(3, 33):
    for frac in (1.0, 0.8, 0.6):
        u = int(frac * size)
        if u < 1:
            continue
        for n in names:
            a = orig[n]
            want = tex.get(n, (u, u))
            src = a.shape[0]
            ix = ((np.arange(u) + 0.5) * src / u).astype(np.int64).clip(0, src - 1)
            got = a[ix][:, ix]
            if not np.array_equal(want, got):
                bad2.append((n, size, frac, u))
print(f"against crafter's own Textures.get over units 3-32 x {{1.0,0.8,0.6}}: {len(bad2)} mismatches"
      f"{'' if not bad2 else '  first ' + str(bad2[:5])}")

# what does 'unit' mean for the item view offsets at each size?
for size in (32, 64, 96, 128):
    unit = size // 9
    print(f"size {size}: unit {unit}, item off {int(0.1*unit)}/{int(0.8*unit)}, "
          f"amount off {int(0.4*unit)}/{int(0.6*unit)}, border {(size - unit*9)//2}")
